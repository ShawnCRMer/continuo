from __future__ import annotations


# --- Upscale (studio.image-upscale-request/1, lane w1-imgout) ----------------------------------
# The local SeedVR2 7B upscales an owned picture into a NEW published picture of the same work
# (the Original is never touched); generation/image_upscale.py owns the graph and the snapshot.
# The answer carries the job id (poll it with /projects/image-producer/status like any run) and
# the History row the page writes for the new version.
_UPSCALE_ENGINE_PROBE: dict[str, Any] = {"until": 0.0, "health": None}
_UPSCALE_ENGINE_TTL_SECONDS = 120.0


def _upscale_module() -> Any:
    from importlib import import_module
    return import_module(_image_contract_module().__package__ + ".image_upscale")


class ImageUpscaleRequest(BaseModel):
    """studio.image-upscale-request/1; host keys the desktop may add (``profile`` ...) are ignored."""

    model_config = ConfigDict(extra="ignore")

    schema_: Literal["studio.image-upscale-request/1"] = Field(alias="schema")
    owner_scope: str = Field(pattern=_OWNER_SCOPE_PATTERN, min_length=18, max_length=47)
    operation_id: str = Field(pattern=r"^imgop_[0-9a-f]{32}$", min_length=38, max_length=38)
    source: ImageEditSource
    factor: Any = None
    long_edge: Any = None
    tile_size: Any = None
    tile_overlap: Any = None
    color_correction: Any = None
    seed: Any = None


async def _upscale_engine_health(client: Any) -> dict[str, Any]:
    """SeedVR2 classes and files from /object_info (GET only, cached; raises when the engine is down)."""
    now = time.monotonic()
    if now < _UPSCALE_ENGINE_PROBE["until"] and _UPSCALE_ENGINE_PROBE["health"] is not None:
        return _UPSCALE_ENGINE_PROBE["health"]
    upscale = _upscale_module()
    info: dict[str, Any] = {}
    for name in upscale.CLASSES:
        body = await client.object_info(name)
        if isinstance(body, dict) and isinstance(body.get(name), dict):
            info[name] = body[name]
    health = upscale.static_health(info)
    _UPSCALE_ENGINE_PROBE.update(until=now + (_UPSCALE_ENGINE_TTL_SECONDS if health["state"] == "ready_static" else 10.0), health=health)
    return health


def _upscale_answer(request: ImageUpscaleRequest, job: Mapping[str, Any], source_job_id: str, replayed: bool) -> dict[str, Any]:
    upscale = _upscale_module()
    snapshot = job["request_snapshot"]
    recipe = upscale.recipe(snapshot)
    out = {"schema": "studio.image-upscale-operation/1", "kind": "image_upscale_status", "owner_scope": request.owner_scope,
           "operation_id": request.operation_id, "image_producer_job_id": job["image_producer_job_id"], "state": job.get("state"),
           "replayed": replayed, "source": {"image_producer_job_id": source_job_id, "sha256": recipe["parent_sha256"]},
           "width": recipe["width"], "height": recipe["height"], "upscale": recipe,
           "history": {"label": f"{recipe['label']} · {upscale.MODEL_LABEL}", "kind": "upscale", "parent_sha256": recipe["parent_sha256"]},
           "poll": {"route": "/projects/image-producer/status", "job_key": "image_producer_job_id"}}
    for optional in ("failure_reason", "quarantine_reason"):
        if job.get(optional):
            out[optional] = job[optional]
    return out


@project_router.post("/image/upscale")
async def image_upscale_submit(request: ImageUpscaleRequest) -> Any:
    """Upscale an owned picture 2x / 4x / to a long edge with the local SeedVR2 7B.

    The result is a new picture of the source's work (from a kept revision: that revision's
    work, so Keep makes the next version); the Original stays as it is.  The same operation id
    with the same request is a replay (no second render); with another request a conflict.
    """
    unknown = _edit_error(404, "picture_unknown", "The picture is unknown")
    try:
        upscale = _upscale_module()
        async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
            root = _image_producer_root(request.owner_scope)
            handoff = _image_operations_handoff(root, request.owner_scope)
            jobs = handoff.producer_jobs()
            states = handoff.image_states()
            _collector, reader = handoff.result_services()
        source_job_id = _edit_source_job_id(states, jobs, request.source)
        source_job = _edit_owned_job(jobs, states, source_job_id) if source_job_id else None
        if source_job is None:
            return unknown
        if source_job.get("state") != "published":
            return _edit_error(409, "picture_not_ready", "The picture is not ready yet")
        artifact = source_job["artifact"]
        if request.source.sha256 and request.source.sha256 != artifact["artifact_sha256"]:
            return _edit_error(409, "source_changed", "The picture changed since it was selected")
        raw = {k: v for k, v in request.model_dump(by_alias=False, include={"factor", "long_edge", "tile_size", "tile_overlap",
                                                                          "color_correction", "seed"}).items() if v is not None}
        try:
            settings = upscale.normalize_request(raw, (artifact["width"], artifact["height"]))
        except upscale.UpscaleError as exc:
            return _edit_error(422, exc.code, exc.message)
        key = "imgop_" + hashlib.sha256(("upscale:" + request.owner_scope + ":" + request.operation_id).encode()).hexdigest()[:32]
        try:
            previous = jobs.get_by_idempotency_key(key)
        except Exception as exc:  # noqa: BLE001 - only an unknown key means "not submitted yet"
            if type(exc).__name__ != "ImageJobNotFoundError":
                raise
            previous = None
        sha = artifact["artifact_sha256"]
        intent = {"schema": upscale.SCHEMA, "source_job_id": source_job_id, "source_sha256": sha,
                  "source_size": [artifact["width"], artifact["height"]], "image": upscale.upload_name(sha),
                  "mode": settings["mode"], "factor": settings["factor"], "long_edge": settings["long_edge"],
                  "tile_size": settings["tile_size"], "tile_overlap": settings["tile_overlap"],
                  "color_correction": settings["color_correction"], "model": upscale.MODEL_FILE, "vae": upscale.VAE_FILE}
        if previous is not None:
            states.require_owned(previous["request_snapshot"]["context_basis"]["context_id"])
            prior = previous["request_snapshot"]
            if prior["effective_input"]["upscale"] != intent or prior["artist_input"]["seed"] != settings["seed"]:
                return _edit_error(409, "idempotency_conflict", "This operation id was already used for another upscale. Use a new operation id.")
            return _upscale_answer(request, previous, source_job_id, True)
        client = _comfy_client_module.ComfyUIClient()
        try:
            health = await _upscale_engine_health(client)
        except Exception as exc:  # noqa: BLE001 - engine down is a finite answer
            if getattr(exc, "reason_code", None) == "unreachable" or isinstance(exc, OSError):
                return _edit_engine_unreachable()
            raise
        if health["state"] != "ready_static":
            missing = health["missing_classes"] + health["missing_files"]
            return _edit_error(503, "upscale_not_ready", "The image engine cannot upscale: missing " + ", ".join(missing))
        home = _compose_revision_home(states, request.source.revision_id) if request.source.revision_id else None
        if home is not None:
            claim = states.load_current_owner(home[0], home[1], "saved_revision", request.source.revision_id)
        else:
            basis_in = source_job["request_snapshot"]["context_basis"]
            claim = states.load_current_owner(basis_in["context_id"], basis_in["work_id"], basis_in["basis_kind"], basis_in.get("revision_id"))
        basis = {k: claim[k] for k in ("context_id", "work_id", "basis_kind", "revision_id", "draft_generation", "basis_content_sha256")}
        policy: dict[str, Any] = {"mode": "specified" if basis["revision_id"] is not None else "current",
                                  "owner_token": claim["owner_token"], "expected_revision": claim["expected_revision"]}
        if basis["revision_id"] is not None:
            policy.update(context_id=basis["context_id"], work_id=basis["work_id"], revision_id=basis["revision_id"])
        try:
            snapshot = upscale.build_snapshot(intent, settings["seed"], settings["width"], settings["height"], basis, policy)
        except upscale.UpscaleError as exc:
            return _edit_error(422, exc.code, "The upscale request is invalid")
        source_png, _artifact = await asyncio.to_thread(reader.read_published_job, source_job_id)
        try:
            await asyncio.to_thread(_edit_service_module.upload_crop, _comfy_client_module.urlopen, client.origin,
                                    intent["image"], source_png)
        except _edit_service_module.EditServiceError as exc:
            return _edit_error(502, exc.code, exc.message)
        except OSError:
            return _edit_engine_unreachable()
        async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
            coordinator = handoff.producer_coordinator()
        job = await coordinator.submit(key, snapshot, client_request_id=key)
        _image_preview_watch(job)
        _announce_image_jobs([(job["image_producer_job_id"], None, job.get("state"))], request.owner_scope)
        return _upscale_answer(request, job, source_job_id, False)
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_film_error_response(exc)


@project_router.get("/image/upscale/capabilities")
async def image_upscale_capabilities(request: Request) -> Any:
    """What Upscale offers (factors, long edge, tile and colour options) and whether the engine has it now."""
    upscale = _upscale_module()
    body: dict[str, Any] = {
        "schema": "studio.image-upscale-capabilities/1", "request_schema": "studio.image-upscale-request/1",
        "model": upscale.MODEL_LABEL, "factors": list(upscale.FACTORS),
        "long_edge": {"min": 512, "max": upscale.OUTPUT_MAX_SIDE, "step": 2}, "output_max_side": upscale.OUTPUT_MAX_SIDE,
        "tile_size": {"options": list(upscale.TILE_SIZES), "default": upscale.DEFAULTS["tile_size"]},
        "tile_overlap": {"options": list(upscale.TILE_OVERLAPS), "default": upscale.DEFAULTS["tile_overlap"]},
        "color_correction": {"options": list(upscale.COLOR_CORRECTIONS), "default": upscale.DEFAULTS["color_correction"]},
    }
    try:
        health = await _upscale_engine_health(_comfy_client_module.ComfyUIClient())
        body.update(available=health["state"] == "ready_static", health=health,
                    reason=None if health["state"] == "ready_static" else "Missing " + ", ".join(health["missing_classes"] + health["missing_files"]))
    except Exception:  # noqa: BLE001 - engine down is a state
        body.update(available=False, health={"state": "unreachable"}, reason="The image engine is not running")
    return JSONResponse(body, headers={"Cache-Control": "no-store"})
