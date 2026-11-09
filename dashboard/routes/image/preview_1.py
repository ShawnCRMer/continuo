from __future__ import annotations


# --- Film Suite page: one read-only Jobs list over both ledgers -------------

class ImagePreviewRequest(_ProjectRequest):
    """Latest live frame of one of the scope's own running generations."""

    schema_: Literal["studio.image-producer-operation/1"] = Field(alias="schema")
    kind: Literal["image_producer_preview"]
    owner_scope: str = Field(pattern=_OWNER_SCOPE_PATTERN, min_length=18, max_length=47)
    image_producer_job_id: str = Field(pattern=r"^imgjob_[0-9a-f]{32}$", min_length=39, max_length=39)


_IMAGE_PREVIEW_JOB_STATES = {
    "accepted": "queued", "submitting": "queued", "queued": "queued",
    "ready": "done", "ingested": "done", "published": "done",
    "failed": "error", "quarantined": "error", "cancelled": "interrupted",
}


@project_router.post("/image/preview")
async def image_producer_preview(request: ImagePreviewRequest) -> Any:
    """Live preview for one owned job: {state, step, steps, data_url, updated_at}.

    Read-only and cheap: answers from the in-memory preview hub (ComfyUI's
    websocket, own prompts only) and the ledger; never polls or submits.
    The prompt id comes from the owner-checked ledger row, never the caller.
    """
    unknown = JSONResponse(
        status_code=404,
        content={"error": {"code": "job_unknown", "message": "Generation job is unknown"}},
    )
    try:
        async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
            root = _image_producer_root(request.owner_scope)
            handoff = _image_operations_handoff(root, request.owner_scope)
            jobs = handoff.producer_jobs()
            states = handoff.image_states()
        try:
            job = jobs.get_job(request.image_producer_job_id)
        except Exception as exc:  # noqa: BLE001 - narrowed below
            if type(exc).__name__ != "ImageJobNotFoundError" and not isinstance(exc, KeyError):
                raise
            return unknown
        basis = (job.get("request_snapshot") or {}).get("context_basis") or {}
        try:
            states.require_owned(basis.get("context_id"))
        except Exception:  # noqa: BLE001 - not owned answers like unknown
            return unknown
        prompt_id = (job.get("provider") or {}).get("prompt_id")
        hub = _image_preview_hub()
        snapshot: dict[str, Any] = {"state": None, "step": None, "steps": None, "node": None, "updated_at": None}
        data_url = None
        if type(prompt_id) is str:
            _image_preview_watch(job)
            snapshot = hub.snapshot(prompt_id)
            data_url = await hub.data_url(prompt_id)
        ledger_state = job.get("state")
        if ledger_state in {"queued", "running"}:
            # Live: the hub knows best ("waiting" before the first message).
            state = snapshot["state"] or "queued"
            if state == "waiting":
                state = "queued"
        else:
            state = _IMAGE_PREVIEW_JOB_STATES.get(ledger_state, "queued")
            data_url = None if state != "queued" else data_url
        if state in {"done", "error", "interrupted"}:
            data_url = None
        return {
            "schema": "studio.image-producer-operation/1",
            "kind": "image_producer_preview_result",
            "owner_scope": request.owner_scope,
            "image_producer_job_id": request.image_producer_job_id,
            "state": state,
            "step": snapshot["step"],
            "steps": snapshot["steps"],
            "data_url": data_url,
            "updated_at": snapshot["updated_at"],
        }
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_film_error_response(exc)

_JOBS_LIST_SCHEMA = "studio.jobs-list/1"
_JOB_ID_RE = re.compile(r"^imgjob_[0-9a-f]{32}$")
_OWNER_SCOPE_RE = re.compile(_OWNER_SCOPE_PATTERN)
_LIVE_PREVIEW_SCHEMA = "studio.image-preview-live/1"


def _krea_adapter_module() -> Any:
    """The loaded Krea adapter module (its package name differs per install)."""
    module = sys.modules.get(f"{_GENERATION_PACKAGE_NAME}.krea2_create_adapter")
    if module is not None and hasattr(module, "live_preview_node"):
        return module
    for name, candidate in tuple(sys.modules.items()):
        if name.endswith(".krea2_create_adapter") and hasattr(candidate, "live_preview_node"):
            return candidate
    return None


def _edit_adapter_module() -> Any:
    """The loaded Qwen edit adapter module, or None before the first edit coordinator."""
    module = sys.modules.get(f"{_GENERATION_PACKAGE_NAME}.qwen_image_edit_adapter")
    if module is not None and hasattr(module, "live_preview_node"):
        return module
    for name, candidate in tuple(sys.modules.items()):
        if name.endswith(".qwen_image_edit_adapter") and hasattr(candidate, "live_preview_node"):
            return candidate
    return None


@project_router.get("/image/preview/live")
async def image_preview_live(request: Request) -> Any:
    """Live step view of one owned job, like KJNodes' Model Preview Override.

    ``GET /image/preview/live?job=imgjob_…&scope=film_owner:…[&since=<rev>]``.
    Answers from the in-memory hub only (no ComfyUI call): step/total, picture
    size, sigma/delta, step times, ETA, the per-step arrays and the latest
    picture. ``rev`` grows with every change; ``since`` equal to it (or a
    matching ``If-None-Match``) returns ``{"unchanged": true}`` without the
    picture, so a renderer can poll it a few times a second.
    """
    unknown = JSONResponse(status_code=404, content={"error": {"code": "job_unknown", "message": "Generation job is unknown"}})
    params = request.query_params
    job_id, scope, since = params.get("job", ""), params.get("scope", "film_owner:scratch"), params.get("since")
    # Hermes Desktop appends ?profile=<name> (and may add other host keys); only our own keys are validated.
    if not _JOB_ID_RE.match(job_id) or not _OWNER_SCOPE_RE.match(scope) \
            or (since is not None and not (since.isdigit() and len(since) <= 12)):
        return JSONResponse(status_code=422, content={"error": {"code": "invalid_preview_request", "message": "Preview request is invalid"}})
    try:
        async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
            root = _image_producer_root(scope)
            handoff = _image_operations_handoff(root, scope)
            jobs = handoff.producer_jobs()
            states = handoff.image_states()
        try:
            job = jobs.get_job(job_id)
        except Exception as exc:  # noqa: BLE001 - narrowed below
            if type(exc).__name__ != "ImageJobNotFoundError" and not isinstance(exc, KeyError):
                raise
            return unknown
        basis = (job.get("request_snapshot") or {}).get("context_basis") or {}
        try:
            states.require_owned(basis.get("context_id"))
        except Exception:  # noqa: BLE001 - not owned answers like unknown
            return unknown
        prompt_id = (job.get("provider") or {}).get("prompt_id")
        hub = _image_preview_hub()
        live = None
        if type(prompt_id) is str:
            _image_preview_watch(job)
            live = hub.live(prompt_id)
        ledger_state = job.get("state")
        if ledger_state in {"queued", "running"}:
            state = (live or {}).get("state") or "queued"
            if state == "waiting":
                state = "queued"
        else:
            state = _IMAGE_PREVIEW_JOB_STATES.get(ledger_state, "queued")
            if ledger_state == "published":
                await _image_steps_persist(job)
        adapter = _krea_adapter_module()
        inserted = adapter.live_preview_node(prompt_id) if adapter is not None else None
        known = bool(adapter is not None and adapter.live_preview_known(prompt_id))
        edit_adapter = _edit_adapter_module()
        if edit_adapter is not None and edit_adapter.live_preview_known(prompt_id):
            # R4 edit jobs (Qwen): their own preview node bookkeeping.
            inserted = edit_adapter.live_preview_node(prompt_id)
            known = True
        probe = adapter.live_preview_probe_state() if adapter is not None else None
        if inserted:
            preview = "on"
        elif known or probe is False:
            preview = "node_missing"
        elif not hub.available:
            preview = "unavailable"
        else:
            preview = "unknown"
        rev = int((live or {}).get("rev") or 0) * 16 + {"queued": 1, "running": 2, "no_preview": 3, "connecting": 4,
                                                         "done": 5, "error": 6, "interrupted": 7, "unavailable": 8}.get(state, 0)
        etag = f'W/"{job_id[7:15]}-{rev}"'
        head = {"schema": _LIVE_PREVIEW_SCHEMA, "image_producer_job_id": job_id, "owner_scope": scope,
                "state": state, "preview": preview, "rev": rev}
        if (since is not None and int(since) == rev) or request.headers.get("if-none-match") == etag:
            return JSONResponse({**head, "unchanged": True}, headers={"ETag": etag, "Cache-Control": "no-store"})
        live = live or {}
        terminal = state in {"done", "error", "interrupted"}
        body = {**head, "unchanged": False,
                **{key: live.get(key) for key in ("step", "total", "w", "h", "sigma", "delta", "step_ms", "avg_step_ms", "eta_ms")},
                "source": live.get("source"),
                "sigmas": live.get("sigmas") or [], "deltas": live.get("deltas") or [], "step_times": live.get("step_times") or [],
                "image_b64": None if terminal else live.get("image_b64"), "mime": None if terminal else live.get("mime")}
        return JSONResponse(body, headers={"ETag": etag, "Cache-Control": "no-store"})
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_film_error_response(exc)

_STEP_HISTORY_ROUTE_SCHEMA = "studio.image-step-history/1"


@project_router.get("/image/steps")
async def image_steps(request: Request) -> Any:
    """R2d: the stored diffusion step history of one owned take.

    ``GET /image/steps?job=imgjob_…&scope=film_owner:…[&step=N]``: the stats
    (sigmas/deltas/step times/total/size) and the list of kept step frames;
    with ``step`` also that frame's bytes (base64).  A take that is published
    is persisted first; one still in memory (not published yet) is answered
    from the hub.  404 for unknown/foreign jobs, like the live route; host
    query keys (``profile``) are ignored.
    """
    unknown = JSONResponse(status_code=404, content={"error": {"code": "job_unknown", "message": "Generation job is unknown"}})
    params = request.query_params
    job_id, scope, step_raw = params.get("job", ""), params.get("scope", "film_owner:scratch"), params.get("step")
    if not _JOB_ID_RE.match(job_id) or not _OWNER_SCOPE_RE.match(scope) \
            or (step_raw is not None and not (step_raw.isdigit() and len(step_raw) <= 4)):
        return JSONResponse(status_code=422, content={"error": {"code": "invalid_steps_request", "message": "Step history request is invalid"}})
    try:
        async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
            root = _image_producer_root(scope)
            handoff = _image_operations_handoff(root, scope)
            jobs = handoff.producer_jobs()
            states = handoff.image_states()
        try:
            job = jobs.get_job(job_id)
        except Exception as exc:  # noqa: BLE001 - narrowed below
            if type(exc).__name__ != "ImageJobNotFoundError" and not isinstance(exc, KeyError):
                raise
            return unknown
        basis = (job.get("request_snapshot") or {}).get("context_basis") or {}
        try:
            states.require_owned(basis.get("context_id"))
        except Exception:  # noqa: BLE001 - not owned answers like unknown
            return unknown
        await _image_steps_persist(job)
        store = _image_step_store()
        meta = await asyncio.to_thread(store.read, job_id) if store is not None else None
        source = "stored"
        memory = None
        if meta is None:
            prompt_id = (job.get("provider") or {}).get("prompt_id")
            memory = _image_preview_hub().history(prompt_id) if type(prompt_id) is str else None
            source = "memory" if memory is not None else "none"
            meta = memory or {}
        frames = [{key: frame.get(key) for key in ("step", "mime", "sigma", "delta", "step_ms", "bytes")}
                  for frame in (meta.get("frames") or []) if isinstance(frame, dict)]
        body: dict[str, Any] = {
            "schema": _STEP_HISTORY_ROUTE_SCHEMA, "image_producer_job_id": job_id, "owner_scope": scope,
            "source": source, "total": meta.get("total"), "w": meta.get("w"), "h": meta.get("h"),
            "sigmas": meta.get("sigmas") or [], "deltas": meta.get("deltas") or [], "step_times": meta.get("step_times") or [],
            "frames": frames,
        }
        if step_raw is not None:
            step = int(step_raw)
            picked: tuple[bytes, str] | None = None
            if source == "stored":
                picked = await asyncio.to_thread(store.frame, job_id, step)
                encoded = base64.b64encode(picked[0]).decode("ascii") if picked else None
                mime = picked[1] if picked else None
            else:
                hit = next((f for f in (memory or {}).get("frames") or [] if f.get("step") == step), None)
                encoded, mime = (hit["image_b64"], hit["mime"]) if hit else (None, None)
            if encoded is None:
                return JSONResponse(status_code=404, content={"error": {"code": "step_unknown", "message": "This step has no stored frame"}})
            body["frame"] = {"step": step, "mime": mime, "image_b64": encoded}
        return JSONResponse(body, headers={"Cache-Control": "no-store"})
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_film_error_response(exc)
