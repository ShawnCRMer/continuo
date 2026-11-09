from __future__ import annotations


# --- Scene analysis (lane w6-scene item 3) ----------------------------------------------------------------------
# POST /scene/analyse starts one engine job (scene_core/scene_engine.py: one runner subprocess, GPU only on an
# empty ComfyUI queue) for a take (take_id = its studio job id, resolved through the ledger like Take A/B) or a
# file of the approved folders (path, optionally media_id file:<sha256> which must match). The answer is a job;
# GET or POST /scene/map reads it: the job state, and when done the studio.scene-map/1 (cuts, speakers, sync,
# motion on one clock, frames + seconds). Maps are kept per file content (sha256) under the Film Suite state
# root, so a second read never runs the engine again. Engine absent -> engine.state "not_installed" ("Not
# installed"), never another analyzer. Answers never carry a path.
_SCENE_ENGINE_NAME = "film_suite_scene_engine"
_SCENE_JOBS: dict[str, dict[str, Any]] = {}
_SCENE_JOBS_LOCK = _thread.allocate_lock()
_SCENE_JOBS_KEEP = 32


def _scene_engine() -> Any:
    module = sys.modules.get(_SCENE_ENGINE_NAME)
    if module is not None:
        return module
    # The unit runs in plugin_api's namespace: __file__ is dashboard/plugin_api.py.
    spec = importlib.util.spec_from_file_location(_SCENE_ENGINE_NAME, Path(__file__).with_name("scene_core") / "scene_engine.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("The scene engine module is unavailable.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_SCENE_ENGINE_NAME] = module
    spec.loader.exec_module(module)
    return module


class SceneAnalyseRequest(_ProjectRequest):
    """One of: ``take_id`` (a finished take of this project) or ``path`` (inside the approved folders)."""

    project_ref: ProjectReference
    take_id: str | None = Field(default=None, min_length=1, max_length=120)
    media_id: str | None = Field(default=None, pattern=r"^file:[0-9a-f]{64}$")
    path: str | None = Field(default=None, min_length=1, max_length=4096)
    parts: list[Literal["cuts", "speakers", "sync", "motion"]] = Field(default_factory=lambda: ["cuts", "speakers", "sync", "motion"], min_length=1, max_length=4)
    refresh: bool = False


class SceneMapRequest(_ProjectRequest):
    project_ref: ProjectReference | None = None
    job_id: str | None = Field(default=None, pattern=r"^scn_[0-9a-f]{32}$")
    take_id: str | None = Field(default=None, min_length=1, max_length=120)
    media_id: str | None = Field(default=None, pattern=r"^file:[0-9a-f]{64}$")
    path: str | None = Field(default=None, min_length=1, max_length=4096)


def _scene_error(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "retryable": False}}


class _SceneError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message


def _scene_cache_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    return Path(state_root(Path(base) / "hermes")) / "scene"


def _scene_file(project_ref: ProjectReference, take_id: str | None, media_id: str | None, path: str | None) -> tuple[Path, str, dict[str, Any]]:
    """The file to analyse, its sha256 and the public source record (ids only)."""
    if (take_id is None) == (path is None):
        raise _SceneError("invalid_request", "Needs exactly one of a take or a file.")
    directory = _project_reference_directory(project_ref)
    manifest = _project_store_factory().read_project(directory, expected_project_id=project_ref.project_id)
    roots = [*manifest["approved_roots"], str(Path(os.path.realpath(directory)))]
    if take_id is not None:
        try:
            row = _read_ledger_row(take_id)
        except Exception:  # noqa: BLE001 - unknown id or unreadable ledger: one finite answer
            raise _SceneError("take_unknown", "That take is not a finished take of this project.") from None
        if getattr(row, "state", None) != "ready" or getattr(row, "project_id", None) != project_ref.project_id:
            raise _SceneError("take_unknown", "That take is not a finished take of this project.")
        try:
            ops = _private_ops()
            file = Path(ops.saved_path_for_receipt(ops.load_for_job(row)))
        except Exception:  # noqa: BLE001
            raise _SceneError("take_unreadable", "The take's file cannot be read.") from None
    else:
        try:
            file = resolve_approved_path(path, roots)
        except (OSError, ValueError):
            raise _SceneError("file_outside_roots", "The file is outside the project's approved folders.") from None
    if not file.is_file():
        raise _SceneError("file_missing", "The file is missing.")
    digest = hashlib.sha256()
    with file.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    sha = digest.hexdigest()
    if media_id is not None and media_id != "file:" + sha:
        raise _SceneError("media_mismatch", "The file is not that Media item.")
    source = {"take_id": take_id, "media_id": "file:" + sha, "sha256": sha}
    return file, sha, source


def _scene_cached(sha: str) -> dict[str, Any] | None:
    try:
        data = json.loads((_scene_cache_dir() / f"{sha}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("schema") == "studio.scene-map/1" else None


def _scene_job_run(job: dict[str, Any], file: Path) -> None:
    engine = _scene_engine()
    try:
        out = engine.run(file, parts=tuple(job["parts"]))
        if not out.get("ok"):
            if out.get("reason") == "not_installed":
                job.update(state="done", map=engine.not_installed_map(job["source"]))
            else:
                job.update(state="failed", error={"code": out.get("reason") or "runner_failed", "message": "The scene engine failed.",
                                                  "retryable": True}, engine_job=out.get("job"))
            return
        scene_map = engine.build_map(out, source=job["source"])
        cache = _scene_cache_dir()
        cache.mkdir(parents=True, exist_ok=True)
        tmp = cache / f"{job['source']['sha256']}.json.part"
        tmp.write_text(json.dumps(scene_map), encoding="utf-8")
        tmp.replace(cache / f"{job['source']['sha256']}.json")
        job.update(state="done", map=scene_map)
    except Exception as exc:  # noqa: BLE001 - the job answers failed, the reason without a path
        job.update(state="failed", error={"code": "runner_failed", "message": f"The scene engine failed ({type(exc).__name__}).", "retryable": True})
    finally:
        job["finished_at"] = time.time()


def _scene_analyse_sync(request: SceneAnalyseRequest) -> dict[str, Any]:
    try:
        file, sha, source = _scene_file(request.project_ref, request.take_id, request.media_id, request.path)
    except _SceneError as exc:
        return _scene_error(exc.code, exc.message)
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    engine = _scene_engine()
    if engine.locate() is None:
        return {"job_id": None, "state": "done", "map": engine.not_installed_map(source)}
    cached = None if request.refresh else _scene_cached(sha)
    if cached is not None and all(cached.get(p) is not None for p in request.parts):
        return {"job_id": None, "state": "done", "map": {**cached, "source": {**cached.get("source", {}), **source}}}
    job_id = "scn_" + uuid.uuid4().hex
    job = {"job_id": job_id, "state": "running", "source": source, "parts": list(request.parts), "started_at": time.time()}
    with _SCENE_JOBS_LOCK:
        _SCENE_JOBS[job_id] = job
        while len(_SCENE_JOBS) > _SCENE_JOBS_KEEP:
            _SCENE_JOBS.pop(next(iter(_SCENE_JOBS)))
    _thread.start_new_thread(_scene_job_run, (job, file))
    return {"job_id": job_id, "state": "running", "source": source}


def _scene_public(job: dict[str, Any]) -> dict[str, Any]:
    out = {"job_id": job["job_id"], "state": job["state"], "source": job["source"]}
    if job["state"] == "done":
        out["map"] = job["map"]
    if job["state"] == "failed":
        out["error"] = job["error"]
    return out


def _scene_map_sync(request: SceneMapRequest) -> dict[str, Any]:
    if request.job_id is not None:
        job = _SCENE_JOBS.get(request.job_id)
        if job is None:
            return _scene_error("job_unknown", "That scene job is unknown.")
        return _scene_public(job)
    if request.project_ref is None:
        return _scene_error("invalid_request", "Name the job, or the project with the take or the file.")
    try:
        _file, sha, source = _scene_file(request.project_ref, request.take_id, request.media_id, request.path)
    except _SceneError as exc:
        return _scene_error(exc.code, exc.message)
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    cached = _scene_cached(sha)
    if cached is not None:
        return {"job_id": None, "state": "done", "map": {**cached, "source": {**cached.get("source", {}), **source}}}
    engine = _scene_engine()
    if engine.locate() is None:
        return {"job_id": None, "state": "done", "map": engine.not_installed_map(source)}
    running = next((j for j in _SCENE_JOBS.values() if j["source"]["sha256"] == sha and j["state"] == "running"), None)
    return _scene_public(running) if running else {"job_id": None, "state": "none", "source": source}


@project_router.post("/scene/analyse")
async def scene_analyse(request: SceneAnalyseRequest) -> Any:
    """Start the scene engine on a take or a file: cuts, speakers, sync, motion (answers a job, or the stored map)."""
    return await asyncio.to_thread(_scene_analyse_sync, request)


@project_router.post("/scene/map")
async def scene_map(request: SceneMapRequest) -> Any:
    """Read a scene job, or the stored studio.scene-map/1 of a take or a file."""
    return await asyncio.to_thread(_scene_map_sync, request)


@project_router.get("/scene/map")
async def scene_map_get(job_id: str | None = None) -> Any:
    """GET form for a job id (the page polls it)."""
    try:
        request = SceneMapRequest(job_id=job_id)
    except Exception:  # noqa: BLE001 - malformed id: the same finite answer
        return _scene_error("invalid_request", "That is not a scene job id.")
    return await asyncio.to_thread(_scene_map_sync, request)


# --- Camera move of one clip (lane w9-chain item 4): POST /scene/camera -----------------------------------------------
# scene_core/camera_flow.py measured in the scene engine's venv (its OpenCV; the python from the engine's engine.json):
# cumulative zoom, tx, ty, rot_deg and the RANSAC inlier ratio of a file inside the approved folders (path) or of a
# finished take (take_id). Kept per file content (sha256) for this process. No engine -> a plain refusal.
_CAMERA_FLOW_NAME = "film_suite_camera_flow"
_CAMERA_CACHE: dict[str, dict[str, Any]] = {}
_CAMERA_CACHE_KEEP = 64


def _camera_flow() -> Any:
    module = sys.modules.get(_CAMERA_FLOW_NAME)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(_CAMERA_FLOW_NAME, Path(__file__).with_name("scene_core") / "camera_flow.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("The camera measure is unavailable.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_CAMERA_FLOW_NAME] = module
    spec.loader.exec_module(module)
    return module


class SceneCameraRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    path: str | None = Field(default=None, min_length=1, max_length=4096)
    take_id: str | None = Field(default=None, min_length=1, max_length=120)
    approved_roots: list[str] = Field(default_factory=list, max_length=64)
    project_ref: ProjectReference | None = None


def _camera_take_file(take_id: str, project_id: str | None = None) -> Path:
    try:
        take = _shoot_take(take_id, project_id)  # lane p8-cut: a copied project's takes (CUT-104)
    except Exception:  # noqa: BLE001 - unknown id or unreadable ledger: one finite answer
        raise _SceneError("take_unknown", "That take is not a finished take.") from None
    if getattr(take, "state", None) not in ("ready", "succeeded") or not getattr(take, "published_path", None):
        raise _SceneError("take_unknown", "That take is not a finished take.")
    if project_id is not None and getattr(take, "project_id", None) != project_id:
        raise _SceneError("take_unknown", "That take is not a finished take of this project.")
    return Path(take.published_path)


def _camera_measure(file: Path) -> dict[str, Any]:
    """The single-clip part {zoom, tx, ty, rot_deg, inliers} (raises _SceneError with a plain reason)."""
    flow = _camera_flow()
    if not file.is_file():
        raise _SceneError("file_missing", "The file is missing.")
    digest = hashlib.sha256()
    with file.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    sha = digest.hexdigest()
    if sha in _CAMERA_CACHE:
        return dict(_CAMERA_CACHE[sha])
    try:
        part = flow.public(flow.run(file, _scene_engine().engine_dir()))
    except flow.CameraFlowUnavailable as exc:
        raise _SceneError("engine_missing", str(exc)) from None
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        raise _SceneError("measure_failed", f"The camera move could not be measured: {exc}") from None
    if len(_CAMERA_CACHE) >= _CAMERA_CACHE_KEEP:
        _CAMERA_CACHE.pop(next(iter(_CAMERA_CACHE)))
    _CAMERA_CACHE[sha] = dict(part)
    return part


def _scene_camera_sync(request: SceneCameraRequest) -> dict[str, Any]:
    if (request.path is None) == (request.take_id is None):
        return _scene_error("invalid_request", "Needs exactly one of a take or a file.")
    try:
        if request.take_id is not None:
            file = _camera_take_file(request.take_id)
        else:
            try:
                file = _confined_local_path(request.path, _request_roots(request.approved_roots, request.project_ref, path=request.path))
            except _FolderNotApproved:
                return {**_scene_error(_FolderNotApproved.reason_code, str(_FolderNotApproved())), **_folder_refusal()}
            except RecipeError:
                return _scene_error("file_outside_roots", "The file is missing or outside the approved folders.")
        return _camera_measure(file)
    except _SceneError as exc:
        return _scene_error(exc.code, exc.message)


@project_router.post("/scene/camera")
async def scene_camera(request: SceneCameraRequest) -> Any:
    """The camera move of one clip (studio.camera-flow/1 single-clip part), measured on the CPU."""
    return await asyncio.to_thread(_scene_camera_sync, request)
