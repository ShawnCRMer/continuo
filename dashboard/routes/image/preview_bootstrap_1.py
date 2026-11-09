from __future__ import annotations


# --- Live render preview (masterplan 2.2) ------------------------------------
# One websocket reader for the whole plugin, loaded unregistered like the
# model catalog.  It only ever watches prompt ids the ledger hands it.
def _load_comfyui_preview_module() -> Any:
    module_path = Path(__file__).with_name("generation") / "comfyui_preview.py"
    spec = importlib.util.spec_from_file_location("film_suite_comfyui_preview", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("The live preview module is unavailable.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_comfyui_preview_module = _load_comfyui_preview_module()
_IMAGE_PREVIEW_HUB = _comfyui_preview_module.PreviewHub()
# prompt id -> image_producer_job_id, for preview.frame payloads (bounded).
_IMAGE_PREVIEW_JOBS: dict[str, str] = {}


def _image_preview_changed(prompt_id: str, rev: int, state: str) -> None:
    film_events.preview_frame(prompt_id, rev, state, _IMAGE_PREVIEW_JOBS.get(prompt_id))


_IMAGE_PREVIEW_HUB.on_change = _image_preview_changed


def _image_preview_hub() -> Any:
    return _IMAGE_PREVIEW_HUB


def _image_preview_watch(job: Mapping[str, Any] | None) -> None:
    """Watch one ledger job's own prompt; never raises into a route."""
    try:
        prompt_id = ((job or {}).get("provider") or {}).get("prompt_id")
        if type(prompt_id) is str and (job or {}).get("state") in {"queued", "running", "submitting"}:
            hub = _image_preview_hub()
            job_id = (job or {}).get("image_producer_job_id")
            if type(job_id) is str:
                _IMAGE_PREVIEW_JOBS.pop(prompt_id, None)
                _IMAGE_PREVIEW_JOBS[prompt_id] = job_id
                while len(_IMAGE_PREVIEW_JOBS) > 64:
                    _IMAGE_PREVIEW_JOBS.pop(next(iter(_IMAGE_PREVIEW_JOBS)))
            if getattr(hub, "on_change", None) is None:
                hub.on_change = _image_preview_changed
            hub.watch(prompt_id)
            hub.ensure_started()
    except Exception:  # noqa: BLE001 - preview is a convenience, never a failure
        pass


_IMAGE_STEP_STORES: dict[str, Any] = {}
_IMAGE_STEP_SAVING: set[str] = set()


def _image_step_store() -> Any:
    """R2d: the take step-history store in the plugin's own data dir (never user folders)."""
    home = _active_hermes_home()
    if home is None:
        return None
    root = str(state_root(Path(home) / "plugin-data") / "image-step-history")
    store = _IMAGE_STEP_STORES.get(root)
    if store is None:
        store = _IMAGE_STEP_STORES[root] = _comfyui_preview_module.StepHistoryStore(root)
    return store


async def _image_steps_persist(job: Mapping[str, Any] | None) -> bool:
    """Write a published take's step history to disk once; never raises into a route.

    The hub keeps every step frame past completion until this ran, so the
    history is never dropped before it is on disk.
    """
    try:
        job = job or {}
        job_id = job.get("image_producer_job_id")
        prompt_id = (job.get("provider") or {}).get("prompt_id")
        if job.get("state") != "published" or type(prompt_id) is not str or type(job_id) is not str or job_id in _IMAGE_STEP_SAVING:
            return False
        hub = _image_preview_hub()
        history = hub.history(prompt_id)
        store = _image_step_store()
        if history is None or store is None:
            return False
        _IMAGE_STEP_SAVING.add(job_id)
        try:
            saved = await asyncio.to_thread(store.save, job_id, history)
        finally:
            _IMAGE_STEP_SAVING.discard(job_id)
        if saved or store.has(job_id):
            hub.mark_persisted(prompt_id)
        return bool(saved)
    except Exception:  # noqa: BLE001 - history is a convenience, never a failure
        return False
