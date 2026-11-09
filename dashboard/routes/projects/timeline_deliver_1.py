from __future__ import annotations


# --- Deliver: check before render + render queue (lane w2-cutdel) --------------------------
# /projects/timeline/export/check reads one cut like the export does (the project's own resolver,
# or caller-named files confined to the approved folders) and answers every finding at once:
# offline sources (block, with clip ids and "locate"), rate mismatches (a 48 fps take on a 24 fps
# cut) and variable-frame-rate sources (warn). /projects/timeline/export/queue lists the export
# jobs in their order (one renders at a time, the rest wait their turn), removes a waiting job and
# clears finished ones. Both are read/act on the in-process ExportJobs of timeline_export_1.py.
# Answers never carry a path.


class TimelineExportCheckRequest(_ProjectRequest):
    """The same document / sources / range as /projects/timeline/export, without starting a job."""

    project_ref: ProjectReference
    timeline_id: str | None = Field(default=None, min_length=1, max_length=120)
    document: dict[str, Any]
    sources: dict[str, str] = Field(default_factory=dict, max_length=2000)
    music: dict[str, str] = Field(default_factory=dict, max_length=8)
    range: TimelineExportRange | None = None


class TimelineExportQueueRequest(_ProjectRequest):
    """``list`` (default), ``remove`` one waiting job (job_id), ``clear_finished``, or ``again`` (lane p12-cut, CUT-006):
    render a failed / cancelled / done job's stored request again as a new job (also after a restart)."""

    project_ref: ProjectReference
    action: Literal["list", "remove", "clear_finished", "again"] = "list"
    job_id: str | None = Field(default=None, pattern=r"^tlexp_[0-9a-f]{32}$", min_length=38, max_length=38)


@project_router.post("/projects/timeline/export/check")
async def projects_timeline_export_check(request: TimelineExportCheckRequest) -> Any:
    try:
        package = _timeline_export_package()
        checker = importlib.import_module(f"{_TIMELINE_EXPORT_PACKAGE_NAME}.export_check")
    except Exception:  # noqa: BLE001 - finite plugin boundary
        return {"error": {"code": "export_unavailable", "message": "Timeline export is not installed."}}
    try:
        directory = _project_reference_directory(request.project_ref)
        manifest = _project_store_factory().read_project(directory, expected_project_id=request.project_ref.project_id)
        project_dir = Path(os.path.realpath(directory))
        roots = [*manifest["approved_roots"], str(project_dir)]
        document = request.document
        if document.get("project_id") not in (None, request.project_ref.project_id):
            return {"error": {"code": "project_scope_mismatch", "message": "The timeline belongs to another project."}}
        ffprobe = Path(FFPROBE_PATH)

        def confine(raw: str) -> Path:
            try:
                return resolve_approved_path(raw, roots)
            except (OSError, ValueError):
                raise package.ExportError("source_outside_roots", "A source is outside the project's approved folders.") from None

        if request.sources or request.music:
            resolver_module = sys.modules[f"{_TIMELINE_EXPORT_PACKAGE_NAME}.resolver"]
            resolver = resolver_module.ProbeResolver(request.sources, request.music, ffprobe=ffprobe, confine=confine)
        else:
            resolver, _bound = await asyncio.to_thread(_timeline_export_bound_resolver, document, request.project_ref.project_id, ffprobe)
        rng = None if request.range is None else (request.range.in_frame, request.range.out_frame)
        answer = await asyncio.to_thread(checker.check_document, document, resolver, timeline_id=request.timeline_id, rng=rng)
        # Owner after p7-cut (PRJ-015, Project decision 10): the package name the next export gets by the project rule
        # (page deliver, reserving nothing), so Deliver's Writes line previews exactly what /projects/timeline/export writes.
        if isinstance(answer, dict):
            try:
                folder = _project_output_dir(directory, request.project_ref.project_id)
                now = _naming._datetime.datetime.now()
                name = _naming.preview(directory, folder, manifest.get("name") or "Project", "deliver", now=now)
                answer = {**answer, "package_name": name, "package_stamp": now.strftime("%Y%m%d-%H%M%S")}
            except Exception:  # noqa: BLE001 - the preview never fails the check
                pass
        return answer
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    except package.ExportError as exc:
        return {"error": exc.as_dict()}
    except Exception:  # noqa: BLE001 - finite plugin boundary
        return {"error": {"code": "check_failed", "message": "The check failed."}}


def _timeline_export_files_missing(directory: Any, result: Any) -> bool:
    """Lane w22-cut (CUT-006 polish): True when a done job's written folder or any of its files is gone (read now)."""
    if not isinstance(result, dict) or not isinstance(result.get("folder"), str) or not result["folder"]:
        return False
    folder = Path(result["folder"])
    if not folder.is_absolute():
        folder = Path(directory) / folder
    if not folder.is_dir():
        return True
    files = result.get("files") if isinstance(result.get("files"), dict) else {}
    return any(isinstance(name, str) and name and not (folder / name).is_file() for name in files.values())


@project_router.post("/projects/timeline/export/queue")
async def projects_timeline_export_queue(request: TimelineExportQueueRequest) -> Any:
    try:
        directory = _project_reference_directory(request.project_ref)
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    project_id = request.project_ref.project_id
    # Lane p12-cut (CUT-006): the project's stored queue is read back on the first look after a start.
    try:
        jobs = _timeline_export_jobs_for(project_id, directory)
    except Exception:  # noqa: BLE001 - finite plugin boundary: the export package is missing
        jobs = _TIMELINE_EXPORT_STATE.get("jobs")
    answer: dict[str, Any] = {"schema": "studio.timeline-export-queue/1", "action": request.action}
    if request.action == "again":
        stored = None if jobs is None or request.job_id is None else jobs.request_of(request.job_id)
        if stored is None:
            return JSONResponse(status_code=404, content={"error": {"code": "job_unknown", "message": "Export job is unknown"}})
        started = await projects_timeline_export(TimelineExportRequest.model_validate({**stored, "wait_ms": 0}))
        if not isinstance(started, dict) or "error" in started:
            return started
        answer["again"] = started.get("job_id")
    if request.action == "remove":
        if request.job_id is None:
            return {"error": {"code": "invalid_request", "message": "remove needs job_id.", "retryable": False, "next": "No job_id."}}
        try:
            removed = None if jobs is None else jobs.remove(request.job_id)
        except Exception as exc:  # noqa: BLE001 - ExportError of the package: not_waiting
            return {"error": {"code": getattr(exc, "code", "remove_failed"), "message": getattr(exc, "message", "The job could not be removed."),
                              "retryable": False, "next": "A running job is not removed; continuo_timeline_export_cancel cancels it."}}
        if removed is None:
            return JSONResponse(status_code=404, content={"error": {"code": "job_unknown", "message": "Export job is unknown"}})
        answer["removed"] = [request.job_id]
    elif request.action == "clear_finished":
        answer["cleared"] = [] if jobs is None else jobs.clear_finished(project_id)
    rows = [] if jobs is None else jobs.queue(project_id)
    # Lane w22-cut (CUT-006 polish; Resolve marks a file that is gone as offline): a done job whose written folder or
    # files are gone reads files_missing, checked now on every list (page and Hermes alike).
    for row in rows:
        if row.get("state") == "done":
            row["files_missing"] = _timeline_export_files_missing(directory, row.get("result"))
    # A done job's result names its folder relative to the project (or the artist's chosen folder).
    answer["jobs"] = rows
    answer["running"] = next((row["job_id"] for row in rows if row["state"] == "running"), None)
    answer["waiting"] = sum(1 for row in rows if row["state"] == "waiting")
    return answer
