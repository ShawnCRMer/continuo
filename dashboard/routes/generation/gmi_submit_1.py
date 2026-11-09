from __future__ import annotations


# --- GMI Cloud submit -----------------------------------------------------------
# Client request ids whose submit coroutine is still running in this process.
# A GMI row in ``submitting`` without a provider id whose client id is not here
# can no longer receive one (lost response or a restart): it becomes
# ``submission_unknown`` and is never sent again.
_GMI_IN_FLIGHT: set[str] = set()
_GMI_ACTIVE_WINDOW_SECONDS = 2 * 3600.0


class _GmiProblem(ValueError):
    def __init__(self, problem: str) -> None:
        self.problem = problem if problem in _gmi_module.PROBLEMS else "args"
        super().__init__(self.problem)


def _gmi_problem_response(problem: str) -> dict[str, Any]:
    response = _v4_error("invalid_request")
    response["error"] = f"{response['error']}: {_gmi_module.PROBLEMS[problem]}"
    response["problem"] = problem
    return response


def _gmi_is_local(value: str) -> bool:
    return bool(_request_identity_module.looks_like_local_path(value))


def _gmi_clip_seconds(path: str) -> float:
    """Duration of a local clip (ffprobe); patched in tests."""
    return float(probe_media(path)["duration"])


def _gmi_prepare_media(
    request: SubmitRequest, *, pin: bool
) -> tuple[dict[str, Any], Any | None, list[tuple[str, int, Any]]]:
    """Return identity args (local paths -> path HMACs) and optional pinned leases.

    Uses the same approved-root boundary and path identity as the Managed V4
    media path.  Remote values must be ``https://`` links and stay unchanged.
    """
    args = request.args
    roots = tuple(_request_roots(request.approved_roots, request.project_ref))
    identity_args: dict[str, Any] = dict(args)
    boundary: Any | None = None
    key: bytes | None = None
    pinned: list[tuple[str, int, Any]] = []
    try:
        for field in _gmi_module.MEDIA_FIELDS:
            raw = args.get(field)
            values = _gmi_module.media_values(args, field)
            if not values:
                continue
            projected = list(values)
            for index, item in enumerate(values):
                if not isinstance(item, str) or not item.strip() or not _gmi_is_local(item):
                    continue
                local_value = _v4_local_path_value(item)
                # The only private input exception is Image's backend-chosen
                # opening-frame folder, never a caller-nominated approved root.
                staged_root = prepared_cache_dir() / "opening-frames"
                is_staged = Path(local_value).parent == staged_root
                if not roots and not is_staged:
                    raise _GmiProblem("local_root")
                if boundary is None:
                    boundary = _v4_path_boundary()
                identity = (
                    boundary.validate_staged_input(local_value, staged_root)
                    if is_staged else boundary.validate_input(local_value, roots)
                )
                if key is None:
                    key = boundary.profile_identity_key()
                projected[index] = _request_identity_module.path_identity(identity.canonical_path, key)
                if field == "input_video" and pin:
                    try:
                        seconds = _gmi_clip_seconds(identity.canonical_path)
                    except Exception as exc:  # noqa: BLE001 - finite reason only
                        raise _GmiProblem("clip_unreadable") from exc
                    duration = args.get("duration_seconds", _gmi_module.DEFAULT_DURATION_SECONDS)
                    if not math.isfinite(seconds) or seconds <= 0:
                        raise _GmiProblem("clip_unreadable")
                    if seconds + duration > _gmi_module.MAX_TOTAL_SECONDS:
                        raise _GmiProblem("clip_length")
                if pin:
                    lease = (
                        boundary.pin_staged_media(local_value, staged_root=staged_root, field=field, index=index)
                        if is_staged else boundary.pin_local_media(
                            local_value, approved_roots=roots, field=field, index=index
                        )
                    )
                    if isinstance(lease, tuple):
                        raise ValueError("media pin returned multiple leases")
                    pinned.append((field, index, lease))
            identity_args[field] = projected if isinstance(raw, list) else projected[0]
        return identity_args, boundary, pinned
    except BaseException:
        for _field, _index, lease in pinned:
            try:
                boundary.abort_pinned_media(lease)
            except BaseException:
                pass
        raise


def _gmi_active_count() -> int:
    """Active GMI takes: bound and unfinished, or a submit running right now."""
    store = _comfy_store()
    connection = store._connect()
    try:
        rows = connection.execute(
            "SELECT client_request_id, provider_job_id FROM studio_jobs WHERE provider_id = ? "
            "AND state IN ('submitting','queued','running') AND created_at >= ?",
            (GMI_PROVIDER_ID, time.time() - _GMI_ACTIVE_WINDOW_SECONDS),
        ).fetchall()
    finally:
        connection.close()
    return sum(1 for row in rows if row[1] is not None or row[0] in _GMI_IN_FLIGHT)


def _gmi_admission_limit() -> int:
    try:
        target = _generation_registry.resolve(GMI_PROVIDER_ID, _gmi_module.TARGET_ID)
        return int(target.capabilities["admission"]["max_active_studio_jobs"])
    except Exception:  # noqa: BLE001 - fail closed to one job
        return 1


def _gmi_submit_reason(outcome: Any) -> str:
    reason = getattr(outcome, "reason_code", None)
    if getattr(outcome, "route_reason", None) == "credit_required":
        return "provider_refused"
    if reason in generation.contract.REASON_CODES:
        return reason
    return "submission_unknown"


async def _gmi_submit_route(request: SubmitRequest) -> dict[str, Any]:
    if _generation_registry.is_paid_video_provider(GMI_PROVIDER_ID):  # lane p5-vid: before any key or upload
        return _generation_registry.paid_target_refusal()
    """Contract-4 submit for GMI Cloud: validate, pin, ledger, upload, submit once."""
    problem = _gmi_module.validate_args(request.args, is_local=_gmi_is_local)
    if problem is not None:
        return _gmi_problem_response(problem)
    boundary: Any | None = None
    pinned: list[tuple[str, int, Any]] = []
    in_flight = False
    submit_called = False
    ledgered_row: Any | None = None
    ledgered_operation: Any | None = None
    project_id: str | None = None
    client_request_id = request.client_request_id
    failure_stage = "output folder"
    try:
        directory, basename, project_id = _v4_destination(request)
        failure_stage = "picture or clip"
        store = _comfy_store()
        existing = _read_client_row(client_request_id)
        if existing is not None:
            identity_args, _boundary, _none = _gmi_prepare_media(request, pin=False)
            _request_json, request_hash = _v4_request_identity(request, project_id, identity_args)
            if getattr(existing, "row_contract", None) != 4:
                return _v4_error("invalid_request")
            if (
                existing.provider_id != request.provider_id
                or existing.target_id != request.target_id
                or existing.request_hash != request_hash
                or existing.project_id != project_id
            ):
                return _v4_error("invalid_request")
            _private_ops().validate_exact_retry(
                existing, destination=directory, requested_basename=basename
            )
            if existing.provider_job_id is not None or existing.is_terminal:
                return _receipt_from_job(existing)
            if client_request_id in _GMI_IN_FLIGHT:
                return _receipt_from_job(existing)
            # GMI has no correlation marker: an accepted row without a request
            # id is never submitted again.
            return _receipt_from_job(_v4_finish_without_job(existing, project_id, "submission_unknown"))
        if client_request_id in _GMI_IN_FLIGHT:
            return _v4_error("provider_busy")

        identity_args, boundary, pinned = _gmi_prepare_media(request, pin=True)
        client = _gmi_client()
        if client is None:
            raise _GmiNotReady("auth_required")
        if _gmi_active_count() >= _gmi_admission_limit():
            raise _GmiNotReady("provider_busy")
        request_json, request_hash = _v4_request_identity(request, project_id, identity_args)

        _GMI_IN_FLIGHT.add(client_request_id)
        in_flight = True
        studio_job_id = "studio:" + uuid.uuid4().hex
        ops = _private_ops()
        prepared = ops.prepare(
            studio_job_id=studio_job_id,
            client_request_id=client_request_id,
            provider_id=request.provider_id,
            target_id=request.target_id,
            project_id=project_id,
            destination=directory,
            requested_basename=basename,
            request_hash=request_hash,
        )
        ledgered = False
        try:
            row = store.create_or_get_v4(
                studio_job_id=studio_job_id,
                client_request_id=client_request_id,
                provider_id=request.provider_id,
                target_id=request.target_id,
                project_id=project_id,
                request_json=request_json,
                request_hash=request_hash,
                destination_identity=prepared.destination_identity,
                requested_basename=basename,
            )
            operation = ops.mark_ledgered(prepared, row)
            ledgered = True
            ops.mark_submit_started(operation)
        except BaseException:
            if not ledgered:
                ops.abort_prepared(prepared)
            raise

        deadline = time.monotonic() + _GMI_SUBMIT_BUDGET_SECONDS
        operational_args: dict[str, Any] = dict(request.args)
        ledgered_row, ledgered_operation = row, operation
        while pinned:
            field, index, lease = pinned.pop(0)
            try:
                data = boundary.consume_pinned_media(lease, max_bytes=_gmi_module.max_bytes_for(field))
            except BaseException:
                try:
                    boundary.abort_pinned_media(lease)
                except BaseException:
                    pass
                raise
            try:
                uploaded = await client.upload(field, data, deadline=deadline)
            finally:
                del data
            if not isinstance(uploaded, str):
                _GMI_LOG.warning("gmi upload refused for %s", row.studio_job_id)
                _gmi_remember_reason(row.studio_job_id, getattr(uploaded, "public_message", None))
                failed = _v4_finish_receipt_failure(row, project_id, _gmi_submit_reason(uploaded), operation)
                return _receipt_from_job(failed)
            if isinstance(request.args.get(field), list):
                values = list(operational_args[field])
                values[index] = uploaded
                operational_args[field] = values
            else:
                operational_args[field] = uploaded

        payload = _gmi_module.build_payload(operational_args)
        submit_called = True
        outcome = await client.submit(payload)
        if outcome.kind == "accepted" and outcome.provider_job_id:
            row = store.bind_provider_job(
                studio_job_id=row.studio_job_id,
                provider_job_id=outcome.provider_job_id,
                expected_project_id=project_id,
            )
            ops.mark_submit_result(operation, row)
            return _receipt_from_job(row)
        _gmi_remember_reason(row.studio_job_id, getattr(outcome, "public_message", None))
        failed = _v4_finish_receipt_failure(row, project_id, _gmi_submit_reason(outcome), operation)
        return _receipt_from_job(failed)
    except BaseException as exc:
        if boundary is not None:
            for _field, _index, lease in pinned:
                try:
                    boundary.abort_pinned_media(lease)
                except BaseException:
                    pass
        if ledgered_row is not None and not submit_called:
            # Nothing was sent to GMI yet: close the row so it neither blocks
            # admission nor turns into a misleading "unknown" later.
            reason = getattr(exc, "reason_code", None)
            if reason not in generation.contract.REASON_CODES:
                reason = "cancelled" if isinstance(exc, asyncio.CancelledError) else "unknown"
            try:
                _v4_finish_receipt_failure(ledgered_row, project_id, reason, ledgered_operation)
            except BaseException:
                pass
        if isinstance(exc, asyncio.CancelledError):
            raise
        if isinstance(exc, _GmiProblem):
            return _gmi_problem_response(exc.problem)
        if isinstance(exc, _GmiNotReady):
            return _v4_error(exc.reason, prefix="Generation submission failed")
        if isinstance(exc, _gmi_module.GmiModelRefused):
            return _v4_error("invalid_request", prefix="Generation submission failed")
        if type(exc) is ValueError and str(exc) in _V4_DESTINATION_MESSAGES:
            _GMI_LOG.warning("gmi submit failed: ValueError: %s", str(exc))
        elif isinstance(exc, studio_core.ProjectError):
            # ProjectError messages are finite literals (no paths): log them.
            stage = failure_stage
            _GMI_LOG.warning(
                "gmi submit failed: ProjectError code=%s stage=%s message=%s",
                getattr(exc, "code", "?"), stage, str(exc),
            )
            response = _v4_error("invalid_request", prefix="Generation submission failed")
            response["error"] = f"Generation submission failed: {stage} refused ({str(exc)})"
            return response
        else:
            _GMI_LOG.warning("gmi submit failed: %s", type(exc).__name__)
        return _v4_exception(exc, prefix="Generation submission failed")
    finally:
        if in_flight:
            _GMI_IN_FLIGHT.discard(client_request_id)


class _GmiNotReady(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)
