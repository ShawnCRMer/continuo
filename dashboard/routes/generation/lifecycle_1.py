from __future__ import annotations


def _gmi_download_client(**kwargs: Any) -> Any:
    """Download client for GMI results: the allow-listed host, no redirects."""
    kwargs["follow_redirects"] = False
    kwargs.pop("max_redirects", None)
    return _gmi_http_factory(**kwargs)


async def _v4_deliver_ready(row: Any, operation: Any, output_identity: Any, provider: str) -> Any:
    if provider == GMI_PROVIDER_ID and not _gmi_module.is_allowed_download_url(output_identity):
        raise ValueError("generation output host is not allowed")
    ops = _private_ops()
    delivery: Any | None = None
    sink: Any | None = None
    try:
        delivery = ops.open_delivery(operation)
        sink = ops.open_sink(delivery)
        if provider == "comfyui-local":
            digest = await _comfy_provider_module.comfy_stream_output_v4(
                client=_comfy_client_module.ComfyUIClient(),
                output_identity=output_identity,
                sink=sink,
            )
        elif provider == GMI_PROVIDER_ID:
            if not _gmi_module.is_allowed_download_url(output_identity):
                raise ValueError("generation output host is not allowed")
            digest = await _stream_managed_v4(
                output_identity, sink, client_factory=_gmi_download_client
            )
        else:
            digest = await _stream_managed_v4(output_identity, sink)
        ops.mark_downloaded(delivery, digest)
        published = ops.publish_or_adopt(delivery, digest)
        updated = _comfy_store().complete_v4_delivery(
            studio_job_id=row.studio_job_id,
            expected_project_id=row.project_id,
            published_identity=published.published_identity,
            output_sha256=published.output_sha256,
            output_bytes=published.output_bytes,
        )
        ops.finalize_ready_receipt(operation, published, updated)
        # Lane p13b-med (MED-017): the backend records a project take's recipe at publication, as ComfyUI's
        # SaveImage writes provenance on the server; a take Hermes made with no Video page open is a take too.
        _stamp_published_take(updated)
        return updated
    except BaseException as original_exc:
        cleanup_exc: BaseException | None = None
        if sink is not None:
            for method_name in ("request_abort", "worker_finished", "abort_owned"):
                method = getattr(sink, method_name, None)
                if callable(method):
                    try:
                        method()
                    except BaseException as caught:
                        if cleanup_exc is None:
                            cleanup_exc = caught
        if delivery is not None:
            try:
                ops.abort_delivery(delivery)
            except BaseException as caught:
                if cleanup_exc is None:
                    cleanup_exc = caught
        if cleanup_exc is not None:
            raise cleanup_exc from original_exc
        raise


async def _v4_poll_route(request: PollRequest, looked_up: Any) -> dict[str, Any]:
    expected_project_id, field_present = _v4_expected_scope(request, looked_up)
    try:
        row = _comfy_store().get_scoped(
            studio_job_id=request.id.strip(),
            expected_project_id=expected_project_id,
            field_present=field_present,
        )
        ops = _private_ops()
        if row.state == "ready":
            operation = ops.reconcile_ready_receipt(row)
            await _h3_measure_take(row, operation)
            return _receipt_from_job(row, operation=operation)
        if row.is_terminal:
            return _receipt_from_job(row)
        operation = ops.load_for_job(row)
        recovered = ops.recover_delivery_before_provider(operation, row)
        if recovered is not None:
            row = _comfy_store().complete_v4_delivery(
                studio_job_id=row.studio_job_id,
                expected_project_id=row.project_id,
                published_identity=recovered.published_identity,
                output_sha256=recovered.output_sha256,
                output_bytes=recovered.output_bytes,
            )
            ops.finalize_ready_receipt(operation, recovered, row)
            await _h3_measure_take(row, operation)
            return _receipt_from_job(row, operation=operation)
        if not row.provider_job_id:
            if (
                row.provider_id == GMI_PROVIDER_ID
                and row.state == "submitting"
                and row.client_request_id not in _GMI_IN_FLIGHT
            ):
                # The submit that owned this row is gone (lost answer or a
                # restart).  GMI has no correlation marker: never resend.
                row = _v4_finish_without_job(row, row.project_id, "submission_unknown")
            if (
                row.provider_id == "comfyui-local"
                and row.state == "submitting"
                and row.client_request_id not in _V4_SUBMITS_IN_FLIGHT
            ):
                return await _v4_recover_lost_submit(row)  # lane w33-rel2 item 2 (VID-169)
            return _receipt_from_job(row)
        if row.provider_id == "comfyui-local":
            profile, _graph = _comfy_assets(row.target_id)
            manifest = profile.manifest if hasattr(profile, "manifest") else profile
            outcome = await _comfy_provider_module.comfy_poll_v4(
                client=_comfy_client_module.ComfyUIClient(),
                provider_job_id=row.provider_job_id,
                manifest=manifest,
            )
        elif row.provider_id == "flux3-managed":
            outcome = await flux3_provider.poll_managed(provider_job_id=row.provider_job_id)
        elif row.provider_id == GMI_PROVIDER_ID:
            gmi_client = _gmi_client()
            if gmi_client is None:
                # Key removed while the take runs: keep the row, report it.
                return _receipt_from_job(row)
            outcome = await gmi_client.poll(row.provider_job_id)
            if getattr(outcome, "kind", None) == "failed":
                _gmi_remember_reason(row.studio_job_id, getattr(outcome, "public_message", None))
        else:
            return _v4_error("invalid_request")
        kind = _v4_outcome_kind(outcome)
        status = getattr(outcome, "status", None)
        if kind == "ready" or status == "ready":
            output = getattr(outcome, "output_identity", None) or getattr(outcome, "sample_url", None)
            if not output:
                row = _comfy_store().transition_scoped(
                    studio_job_id=row.studio_job_id,
                    expected_project_id=row.project_id,
                    allowed_from=(row.state,),
                    new_status="failed",
                    reason_code="output_invalid",
                )
                return _receipt_from_job(row)
            row = await _v4_deliver_ready(row, operation, output, row.provider_id)
            await _h3_collect_sidecar(row, operation, output)
            await _h3_measure_take(row, operation)
            return _receipt_from_job(row, operation=operation)
        if kind in {"queued", "running"} or status in {"queued", "running"}:
            new_state = "queued" if (kind == "queued" or status == "queued") else "running"
            row = _comfy_store().transition_scoped(
                studio_job_id=row.studio_job_id,
                expected_project_id=row.project_id,
                allowed_from=(row.state,),
                new_status=new_state,
            )
            return _receipt_from_job(row)
        if status in {"cancelled", "failed", "moderated"}:
            row = _comfy_store().transition_scoped(
                studio_job_id=row.studio_job_id,
                expected_project_id=row.project_id,
                allowed_from=(row.state,),
                new_status="cancelled" if status == "cancelled" else "failed",
                reason_code={
                    "cancelled": "cancelled",
                    "failed": "execution_failed",
                    "moderated": "moderated",
                }[status],
            )
            ops.release_terminal_without_output(operation, row.state)
            return _receipt_from_job(row)
        if kind == "transport_error":
            return _receipt_from_job(row)
        if kind == "missing":
            updated_at = getattr(row, "updated_at", None)
            if isinstance(updated_at, (int, float)) and time.time() - float(updated_at) < 60.0:
                return _receipt_from_job(row)
        if kind in {"cancelled", "failed", "missing", "refused", "malformed"}:
            reason = _v4_status_reason(
                outcome,
                {"cancelled": "cancelled", "failed": "execution_failed", "missing": "job_lost"}.get(kind, "unreachable"),
            )
            row = _comfy_store().transition_scoped(
                studio_job_id=row.studio_job_id,
                expected_project_id=row.project_id,
                allowed_from=(row.state,),
                new_status="cancelled" if kind == "cancelled" else "failed",
                reason_code=reason,
            )
            ops.release_terminal_without_output(operation, row.state)
            return _receipt_from_job(row)
        return _v4_error("unknown")
    except BaseException as exc:
        if isinstance(exc, asyncio.CancelledError):
            raise
        return _v4_exception(exc, prefix="Generation poll failed")


# Lane w27-rows item 5 (P3, Hermes texts): Hermes wrote aspect_ratio "16:9" and requested_basename "harbour-dusk" for an
# H3 target and was refused. The catalog's exact option ("16:9 (Widescreen)") is what the target takes; a short ratio
# that names exactly one option of the target's aspect_ratio control is read as that option, and a requested_basename
# without an extension gets ".mp4" (a video submit only writes MP4). Anything else is left to the existing refusals.
def _submit_normalised(request: SubmitRequest) -> SubmitRequest:
    changes: dict[str, Any] = {}
    ratio = request.args.get("aspect_ratio") if isinstance(request.args, dict) else None
    if isinstance(ratio, str) and re.fullmatch(r"\d{1,2}:\d{1,2}", ratio.strip()):
        try:
            options = [option for provider in _generation_registry.catalog().get("providers", [])
                       if provider.get("id") == request.provider_id for target in provider.get("targets", [])
                       if target.get("id") == request.target_id for control in target.get("controls", [])
                       if control.get("id") == "aspect_ratio" for option in control.get("options") or []]
        except Exception:  # noqa: BLE001 - without the catalog the value goes on as sent
            options = []
        named = [o for o in options if isinstance(o, str) and (o == ratio.strip() or o.startswith(ratio.strip() + " ("))]
        if len(named) == 1 and named[0] != ratio:
            changes["args"] = {**request.args, "aspect_ratio": named[0]}
    name = request.requested_basename
    if isinstance(name, str) and name.strip() and "." not in Path(name.strip()).name:
        changes["requested_basename"] = name.strip() + ".mp4"
    return request.model_copy(update=changes) if changes else request


@router.post("/submit")
async def submit(request: SubmitRequest) -> dict[str, Any]:
    # Lane p5-vid (VID-020): a paid video target (FLUX 3 / Nous Portal, Wan 3.0 / GMI Cloud, and the legacy
    # FLUX shape without a provider id) is refused before any ledger, credential or network access.
    if _generation_registry.is_paid_video_provider(request.provider_id) or (
        request.provider_id is None and request.schema != "studio.generation-request/1"
        and _generation_registry.is_paid_video_provider("flux3-managed")
    ):
        result = _generation_registry.paid_target_refusal()
        _job_refusal_note(request, result)
        return result
    try:  # lane p5-prj: approved folders come from the server; Hermes naming another folder is refused
        _request_roots(request.approved_roots, request.project_ref)
    except _FolderNotApproved:
        return _folder_refusal()
    if request.provider_id is not None or request.schema == "studio.generation-request/1":
        request = _submit_normalised(request)
        result = await _v4_submit_route(request)
        reason_text = _submit_reason_text(result)  # lane w11-fixb: the reason alone, beside error
        if reason_text and not result.get("reason_text"):
            result = {**result, "reason_text": reason_text}
        if result.get("reason_code") not in V4_REFUSALS_BEFORE_A_JOB:  # lane p10-vid: a refused OUT is no job
            _job_refusal_note(request, result)  # lane w11-backend M-35: a refusal is a failed job with its cause
        _announce_shoot_job(result, submitted=True)
        return result
    if request.mode not in MODES:
        return _error("Generation mode not supported.")
    args = dict(request.args)
    staged: Path | None = None
    try:
        continuation_source = args.get("input_video") if request.mode == "video_continuation" else None
        local_continuation = (
            isinstance(continuation_source, str)
            and bool(continuation_source.strip())
            and not continuation_source.strip().lower().startswith(("http://", "https://", "data:"))
        )
        if local_continuation and request.input_media_sha256 is None:
            return _error("A verified continuation hash is required; prepare the local source again.")
        if request.input_media_sha256 is not None:
            expected_hash = request.input_media_sha256
            if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
                return _error("Prepared continuation hash is invalid; prepare the source again.")
            if request.mode != "video_continuation":
                return _error("Prepared continuation identity is valid only for video continuation.")
            source = args.get("input_video")
            try:
                submitted = _confined_local_path(source, _request_roots(request.approved_roots, request.project_ref, path=source))
                staged = _stage_verified_media(submitted, expected_hash)
                args["input_video"] = str(staged)
            except (RecipeError, OSError, TypeError, ValueError) as exc:
                return _error(f"Prepared continuation verification failed: {exc}")
        problem = flux3_provider.validate(request.mode, args)
        if problem:
            return _error(problem)
        return _response(await flux3_provider.submit(request.mode, args))
    finally:
        if staged is not None:
            _safe_unlink(staged)


@router.post("/poll")
async def poll(request: PollRequest) -> dict[str, Any]:
    result = await _poll_route(request)
    await _h3x_release_after(result)
    await _h3k_advance_after(result)  # lane w2-h3ctl: Generate x N submits its next take when this one ended
    _announce_shoot_job(result)
    return result


async def _poll_route(request: PollRequest) -> dict[str, Any]:
    studio_job_id = request.id.strip()
    if not studio_job_id:
        return _error("A job id is required.")
    try:
        looked_up = _read_ledger_row(studio_job_id)
    except _comfy_job_store_module.JobNotFoundError:
        # Exactly one read-only lookup is the sole compatibility allowance.
        if studio_job_id.startswith(("studio_", "studio:")):
            return _v4_error("invalid_request")
        # Pre-V20 managed jobs have no ledger row.  Preserve their exact
        # historical body ({id, optional save_to}) and route only through the
        # unchanged FLUX adapter; no row, project, or Studio state is created.
        if getattr(request, "model_fields_set", set()) - {"id", "save_to"}:
            return _v4_error("invalid_request")
        args: dict[str, Any] = {"id": studio_job_id}
        destination = flux3_provider.poll_save_destination(request.save_to)
        if destination:
            args["save_to"] = destination
        return _response(await flux3_provider.poll(args))
    except BaseException as exc:
        return _v4_exception(exc, prefix="Generation poll failed")
    if getattr(looked_up, "row_contract", None) == 4:
        return await _v4_poll_route(request, looked_up)
    if getattr(looked_up, "row_contract", None) == 3:
        # Contract-3 Studio rows are the original Comfy lifecycle contract;
        # keep its poll/publication path byte-for-byte isolated from V4.
        if getattr(looked_up, "provider_id", None) == "comfyui-local":
            return await _comfy_poll_route(request)
        args: dict[str, Any] = {"id": studio_job_id}
        destination = flux3_provider.poll_save_destination(request.save_to)
        if destination:
            args["save_to"] = destination
        return _response(await flux3_provider.poll(args))
    return _v4_error("invalid_request")


@router.post("/cancel")
async def cancel(request: CancelRequest) -> dict[str, Any]:
    result = await _cancel_route(request)
    _announce_shoot_job(result)
    return result


async def _cancel_route(request: CancelRequest) -> dict[str, Any]:
    studio_job_id = request.id.strip()
    if not studio_job_id:
        return _error("Generation request is invalid")
    try:
        looked_up = _read_ledger_row(studio_job_id)
    except _comfy_job_store_module.JobNotFoundError:
        return _v4_error("invalid_request")
    except BaseException as exc:
        return _v4_exception(exc, prefix="Generation cancellation failed")
    if getattr(looked_up, "row_contract", None) == 3:
        if not studio_job_id.startswith("studio:"):
            return _error("Generation request is invalid")
        try:
            record = await _comfy_provider_module.comfy_cancel(
                job_store=_comfy_store(),
                client=_comfy_client_module.ComfyUIClient(),
                studio_job_id=studio_job_id,
            )
            return _receipt_from_job(record)
        except Exception as exc:
            return _comfy_error(exc, "Comfy cancellation failed")
    if getattr(looked_up, "row_contract", None) != 4:
        return _v4_error("invalid_request")
    expected_project_id, field_present = _v4_expected_scope(request, looked_up)
    try:
        row = _comfy_store().get_scoped(
            studio_job_id=studio_job_id,
            expected_project_id=expected_project_id,
            field_present=field_present,
        )
        if row.provider_id in ("flux3-managed", GMI_PROVIDER_ID):
            # GMI documents no cancel endpoint.
            return _v4_error("cancel_unsupported")
        if row.state == "ready":
            operation = _private_ops().reconcile_ready_receipt(row)
            return _receipt_from_job(row, operation=operation)
        if row.is_terminal:
            return _receipt_from_job(row)
        ops = _private_ops()
        operation = ops.load_for_job(row)
        recovered = ops.recover_delivery_before_provider(operation, row)
        if recovered is not None:
            row = _comfy_store().complete_v4_delivery(
                studio_job_id=row.studio_job_id,
                expected_project_id=row.project_id,
                published_identity=recovered.published_identity,
                output_sha256=recovered.output_sha256,
                output_bytes=recovered.output_bytes,
            )
            ops.finalize_ready_receipt(operation, recovered, row)
            return _receipt_from_job(row, operation=operation)
        if not row.provider_job_id:
            return _receipt_from_job(row)
        outcome = await _comfy_provider_module.comfy_cancel_v4(
            client=_comfy_client_module.ComfyUIClient(), provider_job_id=row.provider_job_id
        )
        if _v4_outcome_kind(outcome) != "accepted":
            return _v4_error(_v4_status_reason(outcome, "unreachable"), prefix="Comfy cancellation failed")
        row = _comfy_store().transition_scoped(
            studio_job_id=row.studio_job_id,
            expected_project_id=row.project_id,
            allowed_from=(row.state,),
            new_status="cancelled",
            reason_code="cancelled",
        )
        ops.release_terminal_without_output(operation, "cancelled")
        return _receipt_from_job(row)
    except BaseException as exc:
        if isinstance(exc, asyncio.CancelledError):
            raise
        return _v4_exception(exc, prefix="Generation cancellation failed")


# --- Lane w33-rel2 item 2 (VID-169, independent review R7): a lost submit answer never leaves a take "submitting" ----
# ComfyUI accepted the prompt but its answer never reached Hermes (dropped connection, restart): the row is submitting
# without a prompt id. /poll now settles it, never sending anything again: past a short grace (the submit may still be
# uploading in another process) ComfyUI's queue and history are searched for the row's client_request_id; found = the
# row is bound to that prompt and polls on as usual; not found by the bounded wait = failed submission_unknown with a
# plain cause, and the page's Retry sends a new request. Reference: an idempotency-key lookup after an ambiguous
# network failure (Stripe, S3 multipart): look the request up, never send it twice.
V4_LOST_SUBMIT_GRACE_SECONDS = 20.0
V4_LOST_SUBMIT_LIMIT_SECONDS = 300.0
V4_LOST_SUBMIT_CAUSE = ("The answer to this submit was lost and ComfyUI does not list the take in its queue or history, "
                        "so it was not made.")


async def _v4_recover_lost_submit(row: Any) -> dict[str, Any]:
    stamp = getattr(row, "updated_at", None)
    age = time.time() - float(stamp) if isinstance(stamp, (int, float)) else V4_LOST_SUBMIT_LIMIT_SECONDS
    if age < V4_LOST_SUBMIT_GRACE_SECONDS:
        return _receipt_from_job(row)
    recovered = await _comfy_provider_module.comfy_recover_submit_v4(
        client=_comfy_client_module.ComfyUIClient(), client_request_id=row.client_request_id
    )
    if _v4_outcome_kind(recovered) == "accepted" and getattr(recovered, "provider_job_id", None):
        bound = _comfy_store().bind_provider_job(
            studio_job_id=row.studio_job_id,
            provider_job_id=recovered.provider_job_id,
            expected_project_id=row.project_id,
        )
        return _receipt_from_job(bound)
    if age < V4_LOST_SUBMIT_LIMIT_SECONDS:
        return _receipt_from_job(row)
    failed = _v4_finish_without_job(row, row.project_id, "submission_unknown")
    _job_cause_note(row.studio_job_id, V4_LOST_SUBMIT_CAUSE)
    return {**_receipt_from_job(failed), "reason_text": V4_LOST_SUBMIT_CAUSE}
