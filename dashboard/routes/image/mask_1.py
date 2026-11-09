from __future__ import annotations


# --- R3 mask jobs (studio.image-mask-job/1) ----------------------------------
# BiRefNet background and SAM3 click/box/text masks for an owned, published
# picture.  Loaded unregistered like the preview hub; the module owns the graph
# builder, the ComfyUI round trip and a bounded result cache.
def _load_mask_jobs_module() -> Any:
    module_path = Path(__file__).with_name("generation") / "image_mask_jobs.py"
    spec = importlib.util.spec_from_file_location("film_suite_image_mask_jobs", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("The mask job module is unavailable.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_mask_jobs_module = _load_mask_jobs_module()
_IMAGE_MASK_JOBS = _mask_jobs_module.MaskJobService(lambda: _comfy_client_module.ComfyUIClient(), _comfy_client_module.urlopen)
_MASK_WAIT_MS_MAX = 20_000


def _image_mask_service() -> Any:
    return _IMAGE_MASK_JOBS


class ImageMaskJobPicture(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    image_producer_job_id: str = Field(pattern=r"^imgjob_[0-9a-f]{32}$", min_length=39, max_length=39)


class ImageMaskJobRequest(BaseModel):
    """One mask job; host keys the desktop may add (``profile`` …) are ignored."""

    model_config = ConfigDict(extra="ignore", strict=True)

    schema_: Literal["studio.image-mask-job/1"] = Field(alias="schema")
    owner_scope: str = Field(pattern=_OWNER_SCOPE_PATTERN, min_length=18, max_length=47)
    picture: ImageMaskJobPicture
    mode: Literal["background", "points", "box", "text"]
    points: list[dict[str, Any]] | None = Field(default=None, max_length=16)
    box: list[int] | None = Field(default=None, min_length=4, max_length=4)
    text: str | None = Field(default=None, max_length=120)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    refine_iterations: int = Field(default=2, ge=0, le=5)
    wait_ms: int = Field(default=0, ge=0, le=_MASK_WAIT_MS_MAX)


@project_router.post("/image/mask-job")
async def image_mask_job(request: ImageMaskJobRequest) -> Any:
    """Start (or answer) one mask job for an owned, published picture.

    The same request (picture, mode, prompt) is one job: a second call returns
    the running state or the cached result.  ``wait_ms`` long-polls up to 20 s.
    A foreign or unknown picture answers 404 like the picture route; ComfyUI
    being down is a finite ``comfyui_unreachable`` answer, never a hang.
    """
    unknown = JSONResponse(status_code=404, content={"error": {"code": "job_unknown", "message": "Generation job is unknown"}})
    job_id = request.picture.image_producer_job_id
    try:
        async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
            root = _image_producer_root(request.owner_scope)
            handoff = _image_operations_handoff(root, request.owner_scope)
            jobs = handoff.producer_jobs()
            states = handoff.image_states()
            _collector, reader = handoff.result_services()
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
        if job.get("state") != "published":
            return {"error": {"code": "picture_not_ready", "message": "The picture is not ready yet"}}
        data, artifact = await asyncio.to_thread(reader.read_published_job, job_id)
        width, height, sha = int(artifact["width"]), int(artifact["height"]), str(artifact["artifact_sha256"])
        raw = request.model_dump(by_alias=False, exclude={"schema_", "owner_scope", "picture", "wait_ms"})
        try:
            spec = _mask_jobs_module.normalize(raw, width, height)
        except _mask_jobs_module.MaskJobError as exc:
            return JSONResponse(status_code=422, content={"error": {"code": exc.code, "message": exc.message}})
        key = _mask_jobs_module.cache_key(sha, spec)
        service = _image_mask_service()
        state = service.start(key, spec, data, sha, (width, height))
        if state.get("state") == "running" and request.wait_ms:
            state = await service.wait(key, request.wait_ms) or state
        return {"schema": _mask_jobs_module.SCHEMA, "owner_scope": request.owner_scope, "image_producer_job_id": job_id,
                "picture": {"sha256": sha, "width": width, "height": height}, "request": spec, **state}
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_film_error_response(exc)


@project_router.get("/image/mask-capabilities")
async def image_mask_capabilities() -> Any:
    """Which AI mask kinds (subject, background, object, words, sky) the image engine can run now.

    Read-only: asks ComfyUI's /object_info (cached two minutes), never runs a job.  An engine that
    does not answer is ``engine: off`` with no kinds; a missing node or model is named per kind.
    """
    try:
        return JSONResponse(await _image_mask_service().capabilities(), headers={"Cache-Control": "no-store"})
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_film_error_response(exc)


# --- Relight stage 1: MoGe-2 normals (studio.image-geometry-job/1) ------------
# The live-light shader in the renderer needs the surface normals of the picture;
# one cached job per picture, on the mask job machinery (queueing, long-poll, failures).
def _load_geometry_jobs_module() -> Any:
    module_path = Path(__file__).with_name("generation") / "image_geometry_jobs.py"
    spec = importlib.util.spec_from_file_location("film_suite_image_geometry_jobs", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("The geometry job module is unavailable.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_geometry_jobs_module = _load_geometry_jobs_module()
_IMAGE_GEOMETRY_JOBS = _geometry_jobs_module.make_service(_mask_jobs_module, lambda: _comfy_client_module.ComfyUIClient(), _comfy_client_module.urlopen)


def _image_geometry_service() -> Any:
    return _IMAGE_GEOMETRY_JOBS


class ImageGeometryJobRequest(BaseModel):
    """One geometry job; host keys the desktop may add (``profile`` …) are ignored."""

    model_config = ConfigDict(extra="ignore", strict=True)

    schema_: Literal["studio.image-geometry-job/1"] = Field(alias="schema")
    owner_scope: str = Field(pattern=_OWNER_SCOPE_PATTERN, min_length=18, max_length=47)
    picture: ImageMaskJobPicture
    wait_ms: int = Field(default=0, ge=0, le=_MASK_WAIT_MS_MAX)


@project_router.post("/image/geometry-job")
async def image_geometry_job(request: ImageGeometryJobRequest) -> Any:
    """Start (or answer) the normal map job of an owned, published picture.

    One picture is one job: a second call returns the running state or the cached
    normals.  Same owner checks and finite failures as ``/image/mask-job``.
    """
    unknown = JSONResponse(status_code=404, content={"error": {"code": "job_unknown", "message": "Generation job is unknown"}})
    job_id = request.picture.image_producer_job_id
    try:
        async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
            root = _image_producer_root(request.owner_scope)
            handoff = _image_operations_handoff(root, request.owner_scope)
            jobs = handoff.producer_jobs()
            states = handoff.image_states()
            _collector, reader = handoff.result_services()
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
        if job.get("state") != "published":
            return {"error": {"code": "picture_not_ready", "message": "The picture is not ready yet"}}
        data, artifact = await asyncio.to_thread(reader.read_published_job, job_id)
        width, height, sha = int(artifact["width"]), int(artifact["height"]), str(artifact["artifact_sha256"])
        key = _geometry_jobs_module.cache_key(sha)
        service = _image_geometry_service()
        state = service.start(key, {"mode": "geometry"}, data, sha, (width, height))
        if state.get("state") == "running" and request.wait_ms:
            state = await service.wait(key, request.wait_ms) or state
        return {"schema": _geometry_jobs_module.SCHEMA, "owner_scope": request.owner_scope, "image_producer_job_id": job_id,
                "picture": {"sha256": sha, "width": width, "height": height}, **state}
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_film_error_response(exc)
