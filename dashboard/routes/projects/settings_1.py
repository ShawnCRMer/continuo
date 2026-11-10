from __future__ import annotations

# --- Project output folder (lane p5c-frm; Project charter decisions 1, 2, 4; PRJ-001, VID-057/061) -------------
# Every page writes what Continuo makes into ONE folder of the project: the manifest's output_folder (relative to
# the project folder when inside it), else a legacy Shoot state outputFolder (project "test" keeps wan_tests2),
# else <project>/output. The backend creates the default on first write. The folder is always one of the project's
# approved roots (_manifest_roots_of). Hermes reads it in continuo_project_state and cannot change it: the only
# writer is PUT /projects/settings/output from Project settings, refused for caller "hermes" like a folder approval.


def _project_output_resolve(directory: str, project_id: str) -> tuple[Path, bool]:
    manifest = _project_store_factory().read_project(directory, expected_project_id=project_id)
    state_output = None
    if not manifest.get("output_folder"):
        try:
            state_output = _project_state_output(directory)
        except Exception:  # noqa: BLE001 - an unreadable state has no legacy folder
            state_output = None
    return studio_core.project_store.output_folder_of(directory, manifest, state_output)


def _project_output_dir(directory: str, project_id: str, *, create: bool = False) -> Path:
    """The project's output folder; with create, the default <project>/output is made on first write."""
    folder, default = _project_output_resolve(directory, project_id)
    if create and default and not folder.exists():
        folder.mkdir(parents=False, exist_ok=True)
    return folder


def _project_state_output(directory: str) -> str | None:
    """The legacy Video output folder of the project's Shoot state, read straight from its file (no repair)."""
    path = Path(directory) / "state" / "shoot.json"
    if not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    state = value.get("state") if isinstance(value, dict) else None
    folder = state.get("outputFolder") if isinstance(state, dict) else None
    return folder if isinstance(folder, str) and folder.strip() else None


def _project_output_public(directory: str, project_id: str) -> dict[str, Any]:
    folder, default = _project_output_resolve(directory, project_id)
    return {"output_folder": str(folder), "output_folder_default": default}


def _project_output_name(project_name: str, page: str = "video", suffix: str = ".mp4") -> str:
    """The project rule (charter decision 3): <Project>_<page>_free_<date-time>; a typed name wins elsewhere."""
    safe = "".join(ch if ch.isalnum() or ch in "-" else "-" for ch in str(project_name or "Project")).strip("-")[:40] or "Project"
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime()) + f"-{uuid.uuid4().hex[:4]}"
    return f"{safe}_{page}_free_{stamp}{suffix}"


# Lane p6-name (Project charter decision 10): the ONE naming rule, studio_core/naming.py, called by every write site at
# submit time. The template and the per-page take counters live in <project>/naming.json (travel with the folder).
_naming = importlib.import_module(studio_core.__name__ + ".naming")


def _project_result_name(directory: str | Path, folder: str | Path, project_name: Any, page: str, suffix: str, *,
                         typed: Any = None, scene: Any = None) -> str:
    """The result's file name by the project rule; a typed name wins (counter only on collision)."""
    return _naming.name_for(directory, folder, project_name or "Project", page, scene=scene, typed=typed, suffix=suffix)


def _scope_project(owner_scope: Any) -> tuple[str, str, str] | None:
    """(directory, project_id, display name) of a film_owner:prj_… scope; None for Loose work or an unknown project."""
    project_id = str(owner_scope or "")[len("film_owner:"):]
    if not project_id.startswith("prj_"):
        return None
    try:
        directory = _recent_directory(_recent_project_store_factory(), project_id)
        manifest = _project_store_factory().read_project(directory, expected_project_id=project_id)
    except Exception:  # noqa: BLE001 - no project: no rule name
        return None
    return str(directory), project_id, str(manifest.get("name") or "Project")


def _image_result_names_reserve(owner_scope: Any, job_ids: Any) -> dict[str, str]:
    """Image submit (lane p6-name): each new job gets its rule name now, so Jobs shows it while it runs."""
    named: dict[str, str] = {}
    try:
        scope = _scope_project(owner_scope)
        if scope is None:
            return named
        directory, project_id, name = scope
        known = _naming.job_names(directory)
        folder = _project_output_dir(directory, project_id)
        for job_id in job_ids or ():
            if not isinstance(job_id, str) or not job_id:
                continue
            if job_id in known:  # a replayed submit keeps its name
                named[job_id] = known[job_id]
                continue
            named[job_id] = _project_result_name(directory, folder, name, "image", ".png")
            _naming.record_job(directory, job_id, named[job_id])
    except Exception:  # noqa: BLE001 - a name never fails a submit
        pass
    return named


def _image_result_names(owner_scope: Any) -> dict[str, str]:
    scope = _scope_project(owner_scope)
    return _naming.job_names(scope[0]) if scope else {}


def _image_keep_to_output(owner_scope: Any, job_id: str, data: bytes, typed: Any = None) -> str | None:
    """Keep (lane p6-name, Project decision 2 + 10): the picture is also written to <output>/<name>.png, the name
    reserved at submit (or the rule's next one); a typed file name wins. The same bytes already there are reused."""
    scope = _scope_project(owner_scope)
    if scope is None:
        return None
    directory, project_id, project_name = scope
    folder = _project_output_dir(directory, project_id, create=True)
    if isinstance(typed, str) and typed.strip():
        name = _project_result_name(directory, folder, project_name, "image", ".png", typed=typed)
    else:
        name = _naming.job_names(directory).get(job_id)
        if name and (Path(folder) / name).is_file() and (Path(folder) / name).read_bytes() == data:
            return name
        if not name or (Path(folder) / name).exists():
            name = _project_result_name(directory, folder, project_name, "image", ".png")
    target = Path(folder) / name
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return name


async def _image_keep_file(handoff: Any, request: Any) -> str | None:
    """Keep (lane p6-name, Project decision 2 + 10): the kept picture is a file of the project's output folder too."""
    try:
        _collector, reader = handoff.result_services()
        data, _artifact = await asyncio.to_thread(reader.read_published_job, request.image_producer_job_id)
        return await asyncio.to_thread(_image_keep_to_output, request.owner_scope, request.image_producer_job_id, data,
                                       getattr(request, "file_name", None))
    except Exception:  # noqa: BLE001 - the revision is kept; the file is reported missing (None)
        return None


class ProjectOutputWriteRequest(_ProjectRequest):
    project_ref: ProjectReference
    output_folder: str | None = Field(default=None, max_length=32767)


@project_router.put("/projects/settings/output")
async def projects_settings_output(request: ProjectOutputWriteRequest) -> dict[str, Any]:
    """Project settings › Output folder: Change… (a folder) or Reset (null = <project>/output). Artist only."""
    if _film_caller() == "hermes":  # the same rule as a folder approval (Project charter decision 4)
        return _folder_refusal()
    try:
        def write() -> dict[str, Any] | None:  # BCK-227: the whole transaction off the event loop
            directory = _project_reference_directory(request.project_ref)
            if request.output_folder is not None and not Path(request.output_folder.strip()).is_dir():
                return None
            _project_store_factory().write_output_folder(
                directory, request.project_ref.project_id,
                None if request.output_folder is None else request.output_folder.strip())
            if request.output_folder is not None:  # BCK-226: the artist's hand choice, recorded on this computer only
                _output_consents().record(request.project_ref.project_id, directory, request.output_folder.strip())
            return _project_output_public(directory, request.project_ref.project_id)
        public = await asyncio.to_thread(write)
        if public is None:
            return {"ok": False, "reason_code": "folder_missing", "message": "That folder does not exist."}
    except studio_core.ProjectError as exc:
        if getattr(exc, "code", None) == studio_core.project_store.PROJECT_FOLDER_OFFLINE:  # lane p13-prjb finding 1
            return {"ok": False, "reason_code": exc.code, "message": str(exc), "error": {"code": exc.code, "message": str(exc)}}
        return {"ok": False, **_project_error_response(exc)}
    try:
        film_events.shoot_changed(request.project_ref.project_id, _film_caller())
    except Exception:  # noqa: BLE001 - an event never fails a route
        pass
    return {"ok": True, "reason_code": None, "message": None, **public}


# POST /projects/state answers the output folder too (Hermes reads it in continuo_project_state; the page in
# Project settings): wrapped here, the route body above stays as it was.
_projects_state_read_before_output = projects_state


async def _projects_state_with_output(request: ProjectStateReadRequest) -> dict[str, Any]:
    answer = await _projects_state_read_before_output(request)
    if isinstance(answer, dict) and "state" in answer:
        return await asyncio.to_thread(_projects_state_output_answer, request, answer)  # BCK-227: off the event loop
    return answer


def _projects_state_output_answer(request: ProjectStateReadRequest, answer: dict[str, Any]) -> dict[str, Any]:
    try:
        directory = _project_reference_directory(request.project_ref)
        answer = {**answer, **_project_output_public(directory, request.project_ref.project_id)}
        # Lane p5d-frm (Project settings): the project folder and its approved folders, read-only (Hermes reads the
        # list, Project decision 4; only Media's hand changes it).
        manifest = _project_store_factory().read_project(directory, expected_project_id=request.project_ref.project_id)
        # Lane w20-vid (VID-061): the output folder is one of the approved folders for every reader of this answer
        # (page and Hermes), the same list submit admission uses (_manifest_roots_of); listed once it exists.
        answer = {**answer, "project_folder": str(Path(directory)),
                  "approved_roots": _with_output_root(list(manifest["approved_roots"]), _project_output_existing(request.project_ref.project_id, directory)),
                  # lane p13-prj item 3 (PRJ-019): the approved folders that are offline now (unplugged drive)
                  "offline_roots": studio_core.project_store.offline_roots(manifest["approved_roots"])}
    except Exception:  # noqa: BLE001 - the state still reads
        pass
    return answer


projects_state = _projects_state_with_output  # noqa: F811 - the facade name answers the same as the route
for _route in project_router.routes:
    if getattr(_route, "path", None) == "/projects/state" and "POST" in getattr(_route, "methods", ()):
        _route.endpoint = _projects_state_with_output
        _route.dependant.call = _projects_state_with_output


# The output folder is always one of the project's approved roots (Project charter decision 2): appended to the
# manifest roots every route reads through _request_roots (routes/shared/paths_1.py), resolved now, never cached.
_manifest_roots_before_output = _manifest_roots_of


# Lane p13-prjc finding 1: the media write destination (trim) is refused at a pending re-base's old place. Wrapped here
# because shared/paths_2.py (not re-loaded after paths_1.py) defines it; the roots carry the blocked folders.
_confined_output_path_before_rebase = _confined_output_path


def _confined_output_path(path: str, approved_roots: list[str]) -> Path:  # noqa: F811 - append-only extension
    if _rebase_pending_hit(path, getattr(approved_roots, "rebase_offline", None),
                           getattr(approved_roots, "rebase_blocked", ()), getattr(approved_roots, "rebase_own", None)):
        raise _RebasePending(studio_core.project_store.project_rebase_pending_message(approved_roots.rebase_offline))
    return _confined_output_path_before_rebase(path, approved_roots)


def _manifest_roots_of(reference: Any) -> list[str]:  # noqa: F811 - append-only extension
    roots = list(_manifest_roots_before_output(reference))
    try:
        project_id = getattr(reference, "project_id", None)
        directory = getattr(reference, "directory", None) or _recent_directory(_recent_project_store_factory(), project_id)
    except Exception:  # noqa: BLE001 - fail closed: only the manifest roots
        return roots
    try:
        folder = str(_project_output_dir(directory, project_id))
    except Exception:  # noqa: BLE001 - BCK-226: an output folder that is no destination is no root either
        folder = None
    media = _project_media_dir(directory)
    return _with_output_root(roots, folder, str(media) if media.is_dir() else None)


# Lane p5d-frm (Project decision 2 + the 28.10. write-sites line): the output folder is a folder of the project for
# every reader, not only for _manifest_roots_of. Generated results (voice takes, stems, chain takes) land there, so
# Cut (_timeline_footage_roots: Add to cut, the footage resolver) and every request that names a project
# (_request_roots: Media's folder listing, recipe reads) also read it, once it exists. Resolved now, never cached.
def _project_output_existing(project_id: Any, directory: Any = None) -> str | None:
    try:
        if directory is None:
            directory = _recent_directory(_recent_project_store_factory(), project_id)
        folder = _project_output_dir(directory, project_id)
    except Exception:  # noqa: BLE001 - fail closed: no output folder
        return None
    return str(folder) if folder.is_dir() else None


def _project_media_dir(directory: str | Path, *, create: bool = False) -> Path:
    """<project>/media: where imports of a project without an approved folder go (Project decision 1)."""
    folder = Path(directory) / "media"
    if create:
        folder.mkdir(exist_ok=True)
    return folder


def _project_media_existing(project_id: Any, directory: Any = None) -> str | None:
    try:
        if directory is None:
            directory = _recent_directory(_recent_project_store_factory(), project_id)
        folder = _project_media_dir(directory)
    except Exception:  # noqa: BLE001 - fail closed
        return None
    return str(folder) if folder.is_dir() else None


def _with_output_root(roots: list[str], *folders: str | None) -> list[str]:
    roots = list(roots)
    for folder in folders:
        if folder and not any(os.path.normcase(os.path.abspath(r)) == os.path.normcase(os.path.abspath(folder)) for r in roots):
            roots.append(folder)
    return roots


_timeline_footage_roots_before_output = _timeline_footage_roots


def _timeline_footage_roots(project_id: str, directory: str | None = None) -> list[str]:  # noqa: F811 - append-only
    roots = list(_timeline_footage_roots_before_output(project_id, directory))
    return _with_output_root(roots, _project_output_existing(project_id, directory), _project_media_existing(project_id, directory))


# lane p12-med item 2: the output/media extension of _request_roots lives in its one definition (routes/shared/paths_1.py).


# Lane p13-prj item 3 (PRJ-019, BCK-002, PRJ-004; Resolve: the project opens, media on a missing volume are Offline):
# /projects/open and /projects/open-recent answer each approved folder with its state now (online | offline), so Media
# can show an unplugged drive's folder offline. Wrapped here (append-only), the route bodies stay as they were; the
# manifest is only read, never written, for this answer.
_projects_open_before_folders = projects_open
_projects_open_recent_before_folders = projects_open_recent


def _project_folders_answer(answer: Any, project_id: Any, directory: Any) -> Any:
    if not isinstance(answer, dict) or "project" not in answer:
        return answer
    try:
        if directory is None:
            directory = _recent_directory(_recent_project_store_factory(), project_id)
        manifest = _project_store_factory().read_project(directory, expected_project_id=project_id)
        gone = set(studio_core.project_store.offline_roots(manifest["approved_roots"]))
        folders = [{"path": root, "state": "offline" if root in gone else "online"} for root in manifest["approved_roots"]]
    except Exception:  # noqa: BLE001 - the open still answers
        return answer
    return {**answer, "folders": folders}


async def _projects_open_with_folders(request: ProjectOpenRequest) -> dict[str, Any]:
    answer = await _projects_open_before_folders(request)
    project = answer.get("project") if isinstance(answer, dict) else None
    return await asyncio.to_thread(_project_folders_answer, answer, project.get("project_id") if isinstance(project, dict) else None,
                                   request.directory)  # BCK-227


async def _projects_open_recent_with_folders(request: ProjectOpenRecentRequest) -> dict[str, Any]:
    return await asyncio.to_thread(_project_folders_answer, await _projects_open_recent_before_folders(request), request.project_id, None)


_projects_open_with_folders.__name__ = "projects_open"  # the route keeps its name (route surface, p5e-frm)
_projects_open_recent_with_folders.__name__ = "projects_open_recent"
projects_open = _projects_open_with_folders  # noqa: F811 - the facade name answers the same as the route
projects_open_recent = _projects_open_recent_with_folders  # noqa: F811
for _route in project_router.routes:
    if getattr(_route, "path", None) == "/projects/open" and "POST" in getattr(_route, "methods", ()):
        _route.endpoint = _projects_open_with_folders
        _route.dependant.call = _projects_open_with_folders
    if getattr(_route, "path", None) == "/projects/open-recent" and "POST" in getattr(_route, "methods", ()):
        _route.endpoint = _projects_open_recent_with_folders
        _route.dependant.call = _projects_open_recent_with_folders


# Lane p13-prjc finding 1: the project's output folder is a write destination too (default submit, voice takes, stems,
# timeline export all make it with create=True). A pre-wave copy may store it absolute at the original's place; while
# the re-base is pending that folder is refused like every other write there. Reads (create=False) are unchanged.
_project_output_dir_before_rebase = _project_output_dir


def _project_output_dir(directory: str, project_id: str, *, create: bool = False) -> Path:  # noqa: F811
    if create:
        offline, blocked = studio_core.project_store.rebase_pending_roots(
            directory, project_id, list(_project_store_factory().read_project(directory, expected_project_id=project_id)["approved_roots"]))
        folder, _default = _project_output_resolve(directory, project_id)
        if _rebase_pending_hit(str(folder), offline, blocked, str(directory)):
            raise studio_core.ProjectError(studio_core.project_store.PROJECT_REBASE_PENDING,
                                           studio_core.project_store.project_rebase_pending_message(offline))
    return _project_output_dir_before_rebase(directory, project_id, create=create)


# --- Lane w33-rel2 item 0 (BCK-226, independent review S1): the output folder text of project.json is no grant ------
# A project folder can come from anywhere (a download, a colleague's drive); its project.json (and a legacy Shoot state)
# names the output folder as text. That text is a destination only when the folder, links resolved, lies inside the
# project folder, inside one of the project's approved folders, or when the artist chose it on THIS computer in Project
# settings (a machine-local record under LOCALAPPDATA, never in the project). Otherwise it is no approved root and every
# write there is refused with output_folder_not_approved and a reason naming the setting. Reference: Resolve /
# Premiere open a foreign project with its render locations offline until they are linked again on this machine.
import threading  # noqa: E402 - the composition namespace has no threading import

OUTPUT_FOLDER_REFUSAL = ("The output folder in Project settings is outside this project and its approved folders, "
                         "and it was not chosen on this computer.")
_OUTPUT_CONSENT_LOCK = threading.Lock()


class _OutputConsents:
    """The output folders the artist chose by hand on this computer: (project_id, project folder, folder), canonical."""

    MAX = 512

    def __init__(self, path: Path) -> None:
        self.path = path

    def _entries(self) -> list[dict[str, str]]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []  # none or unreadable: nothing chosen (fail closed)
        rows = raw.get("entries") if isinstance(raw, dict) else None
        return [r for r in rows or () if isinstance(r, dict) and all(isinstance(r.get(k), str) for k in ("project_id", "project", "folder"))]

    def holds(self, project_id: str, directory: Any, folder: Any) -> bool:
        key = {"project_id": str(project_id), "project": _approved_roots_module.canonical(str(directory)),
               "folder": _approved_roots_module.canonical(str(folder))}
        return None not in key.values() and key in self._entries()

    def record(self, project_id: str, directory: Any, folder: Any) -> None:
        key = {"project_id": str(project_id), "project": _approved_roots_module.canonical(str(directory)),
               "folder": _approved_roots_module.canonical(str(folder))}
        if None in key.values():
            return
        with _OUTPUT_CONSENT_LOCK:
            rows = [r for r in self._entries() if r != key][-(self.MAX - 1):] + [key]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
            try:
                with open(temporary, "w", encoding="utf-8") as handle:
                    json.dump({"schema": "studio.output-folder-consents/1", "entries": rows}, handle, indent=1)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)


def _output_consents() -> _OutputConsents:
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    return _OutputConsents(state_root(Path(base) / "hermes") / "output-folder-consents.json")


def _project_output_refusal(directory: Any, project_id: str, folder: Any) -> str | None:
    """None when ``folder`` may receive this project's files; else the plain reason (BCK-226)."""
    target = _approved_roots_module.canonical(str(folder))
    project = _approved_roots_module.canonical(str(directory))
    if target is None or project is None:
        return OUTPUT_FOLDER_REFUSAL
    if _approved_roots_module.within(target, project):
        return None
    try:
        manifest = _project_store_factory().read_project(directory, expected_project_id=project_id)
        roots = [c for c in (_approved_roots_module.canonical(r) for r in manifest["approved_roots"]) if c]
    except Exception:  # noqa: BLE001 - an unreadable manifest approves nothing
        roots = []
    if any(_approved_roots_module.within(target, root) for root in roots):
        return None
    return None if _output_consents().holds(project_id, directory, folder) else OUTPUT_FOLDER_REFUSAL


_project_output_dir_before_consent = _project_output_dir


def _project_output_dir(directory: str, project_id: str, *, create: bool = False) -> Path:  # noqa: F811
    folder, _default = _project_output_resolve(directory, project_id)
    reason = _project_output_refusal(directory, project_id, folder)
    if reason is not None:
        raise studio_core.ProjectError("output_folder_not_approved", reason)
    return _project_output_dir_before_consent(directory, project_id, create=create)


_project_output_public_before_consent = _project_output_public


def _project_output_public(directory: str, project_id: str) -> dict[str, Any]:  # noqa: F811
    public = _project_output_public_before_consent(directory, project_id)
    reason = _project_output_refusal(directory, project_id, public["output_folder"])
    return public if reason is None else {**public, "output_folder_reason": reason}


_v4_exception_before_consent = _v4_exception


def _v4_exception(exc: BaseException, *, prefix: str | None = None) -> dict[str, Any]:  # noqa: F811
    if isinstance(exc, studio_core.ProjectError) and getattr(exc, "code", None) == "output_folder_not_approved":
        return _v4_named("output_folder_not_approved", prefix, str(exc))
    return _v4_exception_before_consent(exc, prefix=prefix)


_project_error_response_before_consent = _project_error_response


def _project_error_response(exc: BaseException) -> dict[str, Any]:  # noqa: F811
    if getattr(exc, "code", None) == "output_folder_not_approved":  # BCK-226: Deliver, Use in Cut, voice, stems
        return {"error": {"code": "output_folder_not_approved", "message": OUTPUT_FOLDER_REFUSAL}}
    return _project_error_response_before_consent(exc)
