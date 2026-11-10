from __future__ import annotations


# Video upscale with RelayVSR (lane w23-vsr, VID-153), the default video upscaler: one finished video take of
# this scope (named by its job id; the backend reads the file it published, never a path from the caller) is
# uploaded byte-exact to ComfyUI temp/ and run through the pinned relayvsr-video-upscale-v1 graph with the
# artist's own settings (detail, model, size, filter, seed, IN..OUT). The result is a new take whose take facts
# ``upscale`` link it to its source; it polls, lists and cuts like any take. The project remembers the choice.
class VideoUpscaleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    client_request_id: str = Field(min_length=1, max_length=200)
    studio_job_id: str = Field(pattern=r"^studio:[0-9a-f]{32}$")
    model: Literal["auto", "x2", "x4"] | None = None
    detail: float | None = None
    size: Literal["auto", "native", "1080p", "1440p", "2160p", "custom"] | None = None
    width: int | None = None
    height: int | None = None
    filter: Literal["lanczos", "bicubic", "bilinear", "area", "nearest-exact"] | None = None
    seed: int | None = None
    in_frame: int | None = None
    out_frame: int | None = None
    save_to: str | None = None
    requested_basename: str | None = None
    approved_roots: list[str] = Field(default_factory=list)
    project_id: str | None = Field(default=None, pattern=r"^prj_[0-9a-f]{32}$")
    project_ref: ProjectReference | None = None

    @field_validator("detail", mode="before")
    @classmethod
    def _detail_number(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("detail is a number from 0 to 1")
        return float(value) if isinstance(value, int) else value


class VideoUpscaleSettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    studio_job_id: str | None = Field(default=None, pattern=r"^studio:[0-9a-f]{32}$")
    settings: dict[str, Any] = Field(default_factory=dict)
    remember: bool = False
    project_id: str | None = Field(default=None, pattern=r"^prj_[0-9a-f]{32}$")
    project_ref: ProjectReference | None = None


_VSR_SETTING_FIELDS = ("model", "detail", "size", "width", "height", "filter", "seed", "in_frame", "out_frame")


def _vsr_is_upscale(request: SubmitRequest) -> bool:
    return request.provider_id == "comfyui-local" and request.target_id == _vsr_module.TARGET_ID


def _vsr_project_directory(project_ref: Any) -> str | None:
    """The folder of the NAMED project, or None. Lane w34-sec (scan finding 11): a caller directory is used only after
    its text is a local path (a network / device path is never opened) and its project.json, read through the project
    store's boundary, names this project_id; any other folder is no project here and nothing is read or written."""
    if project_ref is None:
        return None
    try:
        directory = _project_reference_directory(project_ref)
        if _local_path_module.remote_path_reason(directory) is not None:
            return None
        _project_store_factory().read_project(directory, expected_project_id=project_ref.project_id)
        return directory
    except Exception:  # noqa: BLE001 - an unreachable project only means nothing is remembered
        return None


def _vsr_source(take: str, project_id: str | None) -> tuple[Any, Path]:
    """The finished video take of this scope and its published file."""
    try:
        row = _read_ledger_row(take)
    except _comfy_job_store_module.JobNotFoundError:  # a take of another ledger (a copied project) is named, never a 500
        row = None
    if row is None or getattr(row, "row_contract", None) != 4 or row.state != "ready" or row.project_id != project_id:
        raise _vsr_module.VideoUpscaleError("That take is not a finished take of this project on this machine")
    try:
        ops = _private_ops()
        path = Path(ops.saved_path_for_receipt(ops.load_for_job(row)))
    except (PrivateOperationError, OSError, ValueError):
        raise _vsr_module.VideoUpscaleError("The take's file cannot be read") from None
    if path.suffix.lower() != ".mp4" or not path.is_file():
        raise _vsr_module.VideoUpscaleError("That take is not a video file")
    return row, path


def _vsr_prepare_request(request: SubmitRequest) -> dict[str, Any]:
    """Source, settings (request > project memory > defaults), plan and durable identity; no side effect."""
    args = dict(request.args)
    take = args.pop("take", None)
    if type(take) is not str:
        raise _vsr_module.VideoUpscaleError("take must be a finished take's studio job id")
    if "sha256" in args:  # a durable retry carries the identity it was made with
        args.pop("sha256")
    remembered = _vsr_module.read_remembered(directory) if (directory := _vsr_project_directory(request.project_ref)) else {}
    settings = _vsr_module.normalize_settings(args, remembered)
    row, path = _vsr_source(take, request.project_id)
    source = _vsr_module.probe(path, FFPROBE_PATH)
    plan = _vsr_module.plan(settings, source)
    digest = _vsr_module.sha256_file(path)
    return {"settings": settings, "plan": plan, "path": path, "sha256": digest, "source_job": row.studio_job_id,
            "identity": _vsr_module.identity_args(row.studio_job_id, digest, settings), "directory": directory}


async def _vsr_ram_free() -> int | None:
    try:
        stats = await _comfy_client_module.ComfyUIClient().system_stats()
        value = (stats.get("system") or {}).get("ram_free") if isinstance(stats, Mapping) else None
        return int(value) if type(value) in (int, float) else None
    except Exception:  # noqa: BLE001 - an unreachable engine is refused later by static health
        return None


async def _vsr_prepare_fresh(request: SubmitRequest) -> dict[str, Any]:
    prepared = await asyncio.to_thread(_vsr_prepare_request, request)
    _vsr_module.check_memory(prepared["plan"], ram_free=await _vsr_ram_free())
    return prepared


async def _vsr_upload_and_compile(manifest: Any, graph: Any, prepared: Mapping[str, Any]) -> dict[str, Any]:
    name = _vsr_module.upload_name(prepared["sha256"])

    def upload() -> str:
        data = Path(prepared["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != prepared["sha256"]:
            raise _vsr_module.VideoUpscaleError("the take changed after it was checked")
        return _vsr_module.upload_temp(data, name)

    uploaded = await asyncio.to_thread(upload)
    plan = prepared["plan"]
    compiled_graph = _vsr_module.compile_request(graph=graph, manifest=manifest, plan=plan, upload=uploaded)
    return {"graph": compiled_graph, "effective_frames": plan["frames"], "canvas": dict(plan["canvas"]),
            "media": [{"role": "source", "key": "source/0", "sha256": prepared["sha256"]}], "prompt_sent": None,
            "upscale": _vsr_module.take_facts(plan, prepared["source_job"], prepared["sha256"])}


@router.post("/generation/take/upscale")
async def video_take_upscale(request: VideoUpscaleRequest) -> dict[str, Any]:
    """Upscale one finished video take with RelayVSR into a new linked take; poll it like any take."""
    settings = {name: getattr(request, name) for name in _VSR_SETTING_FIELDS if getattr(request, name) is not None}
    # w28-fix item 4c (VID-159, HER parity): either half names the project, as the hand path does (the Video page never
    # names folders); a mismatched pair is refused by SubmitRequest. The folders are then the project's own.
    project_ref = request.project_ref
    project_id = request.project_id if request.project_id is not None else (project_ref.project_id if project_ref is not None else None)
    if project_ref is None and project_id is not None:
        project_ref = ProjectReference(project_id=project_id)
    submit_request = SubmitRequest(
        schema="studio.generation-request/1", provider_id="comfyui-local", target_id=_vsr_module.TARGET_ID,
        client_request_id=request.client_request_id, args={"take": request.studio_job_id, **settings},
        save_to=request.save_to, requested_basename=request.requested_basename,
        approved_roots=_request_roots(request.approved_roots, project_ref),
        **({"project_id": project_id, "project_ref": project_ref} if project_id is not None else {}),
    )
    answer = await submit(submit_request)
    details = answer.get("details") if isinstance(answer, Mapping) else None
    if isinstance(details, Mapping) and details.get("id") and (directory := _vsr_project_directory(project_ref)):
        try:  # the project remembers the artist's (or Hermes') last choice
            remembered = _vsr_module.read_remembered(directory)
            _vsr_module.write_remembered(directory, _vsr_module.normalize_settings(settings, remembered))
        except (OSError, _vsr_module.VideoUpscaleError):
            pass
    return answer


@router.post("/generation/take/upscale/settings")
async def video_take_upscale_settings(request: VideoUpscaleSettingsRequest) -> dict[str, Any]:
    """Every choice, the defaults, the project's remembered settings and (for a take) what the job would be."""
    directory = _vsr_project_directory(request.project_ref)
    try:
        remembered = _vsr_module.read_remembered(directory) if directory else {}
        settings = _vsr_module.normalize_settings(request.settings, remembered)
        if request.remember:
            if directory is None:
                return {"error": "Settings are remembered per project; open a project first", "reason_code": "project_folder_missing"}
            remembered = await asyncio.to_thread(_vsr_module.write_remembered, directory, settings)
        answer: dict[str, Any] = {"target_id": _vsr_module.TARGET_ID, "options": _vsr_module.settings_options(),
                                  "defaults": dict(_vsr_module.DEFAULTS), "remembered": remembered, "settings": settings}
        if request.studio_job_id is not None:
            _row, path = await asyncio.to_thread(_vsr_source, request.studio_job_id, request.project_id)
            source = await asyncio.to_thread(_vsr_module.probe, path, FFPROBE_PATH)
            plan = _vsr_module.plan(settings, source)
            free = await _vsr_ram_free()
            room = None if free is None else max(0, free - _vsr_module.RAM_FLOOR)
            answer["source"] = source
            answer["plan"] = {key: plan[key] for key in ("model", "size", "filter", "detail", "seed", "native", "canvas", "scaled",
                                                          "in_frame", "out_frame", "frames", "fps", "audio", "memory_bytes")}
            answer["memory"] = {"need_bytes": plan["memory_bytes"], "free_bytes": room,
                                "fits": None if room is None else plan["memory_bytes"] <= room}
        return answer
    except _vsr_module.VideoUpscaleError as exc:
        return {"error": str(exc), "reason_code": exc.reason_code}
