from __future__ import annotations


# --- File QC of the written movie (lane w5-review item 2) ----------------------------------------------------
# /projects/timeline/qc measures a WRITTEN file with ffprobe/ffmpeg (review/file_qc.py): the audio-video start
# offset (ms and frames), freeze and black spans with their times, integrated loudness and true peak (the
# export's own measure, timeline_export/render_mix.measure) and video vs audio duration. The file is a finished
# film export job of this session (job_id: its package's film file) or any file the artist names inside the
# project's approved folders (path). The answer is cached by file identity (path, size, mtime), so the Render
# queue and the agent read the same measurement once. Answers never carry a path.
_QC_MODULE_NAME = "film_suite_review_file_qc"
_QC_CACHE: dict[tuple[str, int, int], dict[str, Any]] = {}
_QC_CACHE_MAX = 64


def _qc_module() -> Any:
    module = sys.modules.get(_QC_MODULE_NAME)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(_QC_MODULE_NAME, Path(__file__).with_name("review") / "file_qc.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("The QC module is unavailable.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_QC_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


class TimelineQcRequest(_ProjectRequest):
    """One of: ``job_id`` (a done film export of this session) or ``path`` (inside the approved folders)."""

    project_ref: ProjectReference
    job_id: str | None = Field(default=None, pattern=r"^tlexp_[0-9a-f]{32}$", min_length=38, max_length=38)
    path: str | None = Field(default=None, min_length=1, max_length=4096)
    refresh: bool = False


def _qc_job_file(job_id: str) -> Path:
    module = _qc_module()
    # Lane w20-dlv (CUT-006): any done job of the render queue, after a restart and from Hermes' process too.
    jobs = _timeline_export_jobs_knowing(job_id)
    if jobs is None:
        raise module.QcError("job_unknown", "Export job is unknown.")
    state = _timeline_export_stored(jobs, job_id)[1]
    if state.get("state") != "done" or state.get("kind") != "film":
        raise module.QcError("not_a_film", "Only a finished film render has a movie to check.")
    found = _timeline_export_done_folder(job_id)
    folder = None if found is None else found[1]
    name = ((state.get("result") or {}).get("files") or {}).get("film")
    if folder is None or not isinstance(name, str):
        raise module.QcError("not_a_film", "Only a finished film render has a movie to check.")
    return Path(folder) / name


def _qc_measure(path: Path, refresh: bool) -> dict[str, Any]:
    module = _qc_module()
    if not path.is_file():
        raise module.QcError("file_missing", "The file is missing.")
    stat_ = path.stat()
    key = (os.path.normcase(str(path)), stat_.st_size, stat_.st_mtime_ns)
    if not refresh and key in _QC_CACHE:
        return _QC_CACHE[key]
    try:
        _timeline_export_package()
        measure = importlib.import_module(f"{_TIMELINE_EXPORT_PACKAGE_NAME}.render_mix").measure
    except Exception:  # noqa: BLE001 - without the export package the file is checked without loudness
        measure = None
    answer = module.check_file(path, ffmpeg=Path(FFMPEG_PATH), ffprobe=Path(FFPROBE_PATH), measure_loudness=measure)
    answer = {**answer, "file": path.name, "bytes": stat_.st_size}
    if len(_QC_CACHE) >= _QC_CACHE_MAX:
        _QC_CACHE.pop(next(iter(_QC_CACHE)))
    _QC_CACHE[key] = answer
    return answer


@project_router.post("/projects/timeline/qc")
async def projects_timeline_qc(request: TimelineQcRequest) -> Any:
    """Measure a written movie: A/V offset, freeze and black spans, loudness / true peak, durations."""
    module = _qc_module()
    if (request.job_id is None) == (request.path is None):
        return {"error": {"code": "invalid_request", "message": "Needs exactly one of an export job or a file.", "retryable": False}}
    try:
        directory = _project_reference_directory(request.project_ref)
        manifest = _project_store_factory().read_project(directory, expected_project_id=request.project_ref.project_id)
        roots = [*manifest["approved_roots"], str(Path(os.path.realpath(directory)))]
        if request.job_id is not None:
            path = _qc_job_file(request.job_id)
        else:
            try:
                path = resolve_approved_path(request.path, roots)
            except (OSError, ValueError):
                return {"error": {"code": "file_outside_roots", "message": "The file is outside the project's approved folders.", "retryable": False}}
        answer = await asyncio.to_thread(_qc_measure, path, request.refresh)
        return {**answer, "job_id": request.job_id}
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    except module.QcError as exc:
        return {"error": exc.as_dict()}
    except Exception:  # noqa: BLE001 - finite plugin boundary
        return {"error": {"code": "qc_failed", "message": "The check failed.", "retryable": True}}
