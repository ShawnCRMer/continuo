from __future__ import annotations


# Lane w34-sec (scan findings 1-7, 11-13): the one lexical guard against UNC / device paths
# (studio_core/local_path.py), shared by every route that takes a path from a request or from project data.
_local_path_spec = importlib.util.spec_from_file_location(
    "film_suite_local_path", Path(__file__).with_name("studio_core") / "local_path.py")
_local_path_module = importlib.util.module_from_spec(_local_path_spec)
_local_path_spec.loader.exec_module(_local_path_module)


class RecipeError(RuntimeError):
    pass


def resolve_approved_path(path: str, approved_roots: list[str]) -> Path:
    """Resolve a local path and reject traversal, symlink, drive, and UNC escapes."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("A local path is required.")
    raw_path = path.strip()
    # Lane w34-sec (finding 6): every UNC / device spelling (\\host, //host, /\host, \/host, \\?\, \\.\) is refused
    # on its text: realpath would contact the host before the containment check below. A remote root is skipped.
    if raw_path.startswith(("\\", "//")) or _local_path_module.remote_path_reason(raw_path) is not None:
        raise ValueError("UNC paths are not allowed outside an approved local root.")
    if not approved_roots:
        raise ValueError("At least one approved root is required.")

    candidate = Path(os.path.realpath(os.path.expanduser(path.strip())))
    roots = [Path(os.path.realpath(os.path.expanduser(root.strip()))) for root in approved_roots
             if isinstance(root, str) and root.strip() and _local_path_module.remote_path_reason(root) is None]
    if not roots:
        raise ValueError("At least one approved root is required.")

    for root in roots:
        try:
            common = os.path.commonpath([os.path.normcase(str(candidate)), os.path.normcase(str(root))])
        except ValueError:
            continue
        if os.path.normcase(common) == os.path.normcase(str(root)):
            return candidate
    raise ValueError("Path is outside every approved root.")


def _confined_local_path(path: str, approved_roots: list[str]) -> Path:
    try:
        roots = list(dict.fromkeys([*(approved_roots or []), str(prepared_cache_dir())]))
        candidate = resolve_approved_path(path, roots)
    except (OSError, ValueError) as exc:
        raise RecipeError(str(exc)) from exc
    if not candidate.is_file():
        raise RecipeError("The local video file does not exist.")
    return candidate


# --- Approved folders are decided here, never by the request (lane p5-prj, Project charter decision 4) ---
# Every route that takes approved_roots calls _request_roots(request.approved_roots, project_ref) once; the
# rule itself lives in studio_core/approved_roots.py (one resolver, no cache).
_approved_roots_spec = importlib.util.spec_from_file_location(
    "film_suite_approved_roots", Path(__file__).with_name("studio_core") / "approved_roots.py"
)
_approved_roots_module = importlib.util.module_from_spec(_approved_roots_spec)
_approved_roots_spec.loader.exec_module(_approved_roots_module)


class _FolderNotApproved(RecipeError):
    """Hermes named a folder no one approved, or a request read a file in a folder that is not approved."""

    reason_code = _approved_roots_module.REASON_CODE

    def __init__(self) -> None:
        super().__init__(_approved_roots_module.MESSAGE)


def _folder_refusal() -> dict[str, Any]:
    return {"ok": False, "error": _approved_roots_module.MESSAGE, "reason_code": _approved_roots_module.REASON_CODE,
            "message": _approved_roots_module.MESSAGE}


def _loose_work_folders() -> Any:
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    return _approved_roots_module.LooseWorkFolders(
        state_root(Path(base) / "hermes") / _approved_roots_module.LOOSE_FILE_NAME)


def _manifest_roots_of(reference: Any) -> list[str]:
    """The manifest roots of one project reference, read now; an unreadable project approves nothing."""
    try:
        project_id = getattr(reference, "project_id", None)
        directory = getattr(reference, "directory", None) or _recent_directory(_recent_project_store_factory(), project_id)
        manifest = _project_store_factory().read_project(directory, expected_project_id=project_id)
        return list(manifest["approved_roots"])
    except Exception:  # noqa: BLE001 - fail closed
        return []


def _recent_projects_roots() -> list[str]:
    """The manifest roots of every project in the Recent list, each read now."""
    try:
        entries, _snapshot = _recent_project_store_factory()._read_entries()
    except Exception:  # noqa: BLE001 - no readable registry: no project roots
        return []
    roots: list[str] = []
    for entry in entries or ():
        try:
            manifest = _project_store_factory().read_project(entry["directory"], expected_project_id=entry["project_id"])
            roots.extend(manifest["approved_roots"])
        except Exception:  # noqa: BLE001 - one unreadable project approves nothing
            continue
    return roots


def _request_roots_narrowed(request_roots: Any, project_ref: Any = None, *, path: Any = None, server_list: bool = False) -> list[str]:
    """The roots this request may use, read from the server now, never cached (lane p12-med item 2, BCK-001).

    Server list: with ``project_ref`` the project's manifest folders plus its own output/ and media/ folders
    (``_manifest_roots_of``), without one the Loose-work list. Project charter decision 4: "A request can never
    name its own folders": a sent list only narrows inside the server's list. For Hermes a root the server does
    not hold refuses the whole request; for the artist it is dropped, and a list with nothing the server holds
    (empty, or stale after a move) reads the server's whole list. A ``path`` outside every server folder is refused ``folder_not_approved`` for both, so a removed folder
    is refused on the very next request by name.
    """
    owned = [str(prepared_cache_dir())]
    caller = _film_caller()
    requested = list(request_roots or ())
    if project_ref is not None and requested:
        # Lane p13-prjb finding 4 (cold review p13-prj Minor 2): Hermes sends back the server's own list from
        # /projects/state, which names the project's offline folders too. Those are dropped before the narrowing
        # check (they are no destination now), never refused as a missing folder. Any other folder is narrowed as before.
        offline = {os.path.normcase(os.path.normpath(root)) for root in
                   studio_core.project_store.offline_roots(_manifest_roots_of(project_ref)) if isinstance(root, str)}
        requested = [root for root in requested
                     if not (isinstance(root, str) and root and os.path.normcase(os.path.normpath(root)) in offline)]
    try:
        kept, _dropped = _approved_roots_module.effective_roots(
            requested, project_ref, caller,
            project_roots=_manifest_roots_of, known_project_roots=_recent_projects_roots,
            loose=_loose_work_folders(), owned_roots=owned,
        )
    except _approved_roots_module.FolderNotApproved:
        raise _FolderNotApproved() from None
    if project_ref is not None:
        pid, pdir = getattr(project_ref, "project_id", None), getattr(project_ref, "directory", None)
        server = _with_output_root(list(_manifest_roots_of(project_ref)), _project_output_existing(pid, pdir),
                                   _project_media_existing(pid, pdir))
    else:
        server = list(_loose_work_folders().read()["approved_roots"])
    server = [root for root in dict.fromkeys(server) if isinstance(root, str) and os.path.isdir(root)]
    if not server_list:
        # Every other route: the sent list narrowed to the server's list (p5-prj); a guard that needs a root
        # ("without approved root fails before admission") still sees an empty list.
        paths = path if isinstance(path, (list, tuple)) else [path]
        if _dropped and any(_approved_roots_module.path_refused(p, kept, _dropped, owned) for p in paths if p):
            raise _FolderNotApproved()
        roots = kept
        if project_ref is not None:
            roots = _with_output_root(list(kept), _project_output_existing(pid, pdir), _project_media_existing(pid, pdir))
        return roots
    # The media routes (routes/media/**, server_list=True): a sent list only narrows inside the server's list; with
    # nothing the server holds (empty, or stale after a move) they read the server's whole list, and a path outside
    # every server folder is refused folder_not_approved by name (a removed folder on the very next request).
    if kept:
        roots = kept
        if project_ref is not None:
            roots = _with_output_root(list(kept), _project_output_existing(pid, pdir), _project_media_existing(pid, pdir))
    else:
        roots = server
    paths = path if isinstance(path, (list, tuple)) else [path]
    allowed = [c for c in (_approved_roots_module.canonical(r) for r in [*roots, *owned]) if c]
    # Lane w22-cut (CUT-177): the path is checked against the SERVER's list, as the docstring says. A file of the same
    # project outside the sent (narrowed) folder, e.g. pictures/ while Media shows clips/, is served from the server
    # folder that holds it; on 042e516f it answered 403 folder_not_approved and Cut's viewer stayed black.
    # Only for the page's request that names its project: Media's current folder is incidental there. Hermes' sent
    # folders stay a deliberate narrowing (test_audio_stems), and in Loose work a sent folder narrows (test_media_sound).
    held = [r for r in (server if project_ref is not None and _film_caller() != "hermes" else []) if r not in roots]
    for item in paths:
        target = _approved_roots_module.canonical(item) if item else None
        if target is None or any(_approved_roots_module.within(target, parent) for parent in allowed):
            continue
        holder = [r for r in held if (c := _approved_roots_module.canonical(r)) and _approved_roots_module.within(target, c)]
        if not holder:
            raise _FolderNotApproved()
        roots = [*roots, *(r for r in holder if r not in roots)]
    return roots


# --- Lane p13-prjc finding 1 (second cold review p13-prj, Minor 1): a pending re-base never writes into the old place ---
# A moved or copied project opened while a folder is offline keeps its manifest as stored (p13-prjb finding 2), so its
# folders may still name the ORIGINAL. Every writer that confines a destination asks _rebase_pending_refusal first;
# reads are untouched (Resolve keeps the old links until relink).
class _RebasePending(RecipeError):
    """A write into a moved or copied project's old place while its re-base waits for an offline folder."""

    reason_code = "project_rebase_pending"


class _ServerRoots(list):
    """The roots _request_roots answers, plus the roots a pending re-base blocks for writes (read now, never cached)."""

    rebase_blocked: tuple[str, ...] = ()
    rebase_offline: str | None = None
    rebase_own: str | None = None


def _rebase_pending_of(project_ref: Any) -> tuple[str | None, list[str], str | None]:
    """(offline folder as stored, roots at the old place, this project's folder) of one project reference, read now."""
    if project_ref is None:
        return None, [], None
    try:
        project_id = getattr(project_ref, "project_id", None)
        directory = getattr(project_ref, "directory", None) or _recent_directory(_recent_project_store_factory(), project_id)
        manifest = _project_store_factory().read_project(directory, expected_project_id=project_id)
        offline, blocked = studio_core.project_store.rebase_pending_roots(directory, project_id, list(manifest["approved_roots"]))
        return offline, blocked, str(directory)
    except Exception:  # noqa: BLE001 - an unreadable project approves nothing anyway (_manifest_roots_of)
        return None, [], None


def _rebase_pending_hit(target: Any, offline: str | None, blocked: Any, own: str | None) -> bool:
    if not offline or not blocked or not isinstance(target, str) or not target:
        return False
    canonical = _approved_roots_module.canonical(target)
    if canonical is None:
        return False
    own_canonical = _approved_roots_module.canonical(own) if own else None
    if own_canonical is not None and _approved_roots_module.within(canonical, own_canonical):
        return False  # the project's own folder (its output/) always admits
    return any(_approved_roots_module.within(canonical, parent)
               for parent in (_approved_roots_module.canonical(root) for root in blocked) if parent)


def _rebase_pending_refusal(project_ref: Any, target: Any) -> None:
    """Raise _RebasePending when ``target`` lies at the old place of a project whose re-base is pending."""
    offline, blocked, own = _rebase_pending_of(project_ref)
    if _rebase_pending_hit(target, offline, blocked, own):
        raise _RebasePending(studio_core.project_store.project_rebase_pending_message(offline))


def _request_roots(request_roots: Any, project_ref: Any = None, *, path: Any = None, server_list: bool = False) -> list[str]:
    """The one entry every route calls (p12-med: one definition): _request_roots_narrowed's list, carrying the roots a
    pending re-base blocks for writes (lane p13-prjc finding 1). The list itself is unchanged, so reads stay."""
    roots = _ServerRoots(_request_roots_narrowed(request_roots, project_ref, path=path, server_list=server_list))
    offline, blocked, own = _rebase_pending_of(project_ref)
    if offline and blocked:
        roots.rebase_offline, roots.rebase_blocked, roots.rebase_own = offline, tuple(blocked), own
    return roots


def _sent_without_offline(request_roots: Any, project_ref: Any) -> list[Any]:
    """The sent list without this project's offline folders (lane p13-prjc finding 2, second cold review Minor 2).

    The same drop as in _request_roots (p13-prjb finding 4), for the stale-list check of _v4_destination: a list made
    only of the project's offline folders (any spelling) is left empty and behaves exactly like an empty sent list.
    """
    requested = list(request_roots or ())
    if project_ref is None or not requested:
        return requested
    offline = {os.path.normcase(os.path.normpath(root)) for root in
               studio_core.project_store.offline_roots(_manifest_roots_of(project_ref)) if isinstance(root, str)}
    return [root for root in requested
            if not (isinstance(root, str) and root and os.path.normcase(os.path.normpath(root)) in offline)]
