from __future__ import annotations

# --- Recent with picture and count (lane p5e-frm item 5; frame decision 10; FRM-029/031/034, REV-1/3/71) -------------
# GET /projects/recent answers per project, besides id, name and available:
#   item_count  the Media listing's files_found for the project's folders (its approved roots, the output folder and
#               media/, the folders the Media listing reads): one counter for Start and Media (lane w14-shell);
#   cover       the newest picture of those folders, else the newest clip (Start draws it through POST /media/thumb,
#               which renders a clip's poster frame): {"path", "kind", "root"} (root = the approved folder holding it,
#               the request's approved_roots), or null when the project holds neither.
# The walk is bounded (depth MEDIA_LIST_MAX_DEPTH, RECENT_COUNT_SCAN_BUDGET entries and RECENT_COUNT_TIME_BUDGET_S
# per project, no link following, the Media listing's skipped folders); a project that cannot be read answers
# item_count 0 and cover null (never an error for the list). A missing project is not walked.
# Start re-reads Recent on the existing contract events (no polling, no new event name): media.changed (a picture,
# clip or sound came or went: the count and the cover move) and script.changed (the scene count moves). A project made
# or opened through a route (POST /projects/create, /projects/open, /projects/open-recent; Hermes'
# continuo_project_create too) announces media.changed for its scope film_owner:<project_id>: a new scope of Media.

RECENT_COUNT_SCAN_BUDGET = 6_000
RECENT_COUNT_TIME_BUDGET_S = 0.6
_RECENT_KINDS = {"image", "video", "audio"}


def _recent_project_folders(directory: str, project_id: str) -> list[str]:
    """The folders Media lists for one project: approved roots + output folder + media/ (each once, existing)."""
    reference = ProjectReference(project_id=project_id, directory=directory)
    roots = [str(root) for root in _manifest_roots_of(reference)]
    return [root for root in roots if os.path.isdir(root)]


def _recent_media_facts(roots: list[str]) -> dict[str, Any]:
    """Count pictures, clips and sounds under ``roots`` (bounded walk) and find the newest picture and clip."""
    deadline = time.monotonic() + RECENT_COUNT_TIME_BUDGET_S
    scanned = 0
    seen: set[str] = set()
    count = 0
    newest: dict[str, tuple[float, str, str]] = {}
    for root in roots:
        queue: collections.deque[tuple[str, int]] = collections.deque([(root, 0)])
        while queue:
            folder, depth = queue.popleft()
            if time.monotonic() >= deadline or scanned >= RECENT_COUNT_SCAN_BUDGET:
                return {"count": count, "newest": newest, "truncated": True}
            try:
                with os.scandir(folder) as iterator:
                    for entry in iterator:
                        scanned += 1
                        if scanned >= RECENT_COUNT_SCAN_BUDGET:
                            break
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                if not _media_dir_is_skipped(entry) and depth < MEDIA_LIST_MAX_DEPTH:
                                    queue.append((entry.path, depth + 1))
                                continue
                            kind = MEDIA_EXTENSIONS.get(os.path.splitext(entry.name)[1].lower())
                            if kind not in _RECENT_KINDS or not entry.is_file(follow_symlinks=False):
                                continue
                            if os.name == "nt" and _media_entry_is_link(entry):
                                continue
                            key = os.path.normcase(os.path.abspath(entry.path))
                            if key in seen:
                                continue
                            seen.add(key)
                            count += 1
                            if kind in ("image", "video"):
                                modified = entry.stat(follow_symlinks=False).st_mtime
                                if kind not in newest or modified > newest[kind][0]:
                                    newest[kind] = (modified, entry.path, root)
                        except OSError:
                            continue
            except OSError:
                continue
    return {"count": count, "newest": newest, "truncated": False}


def _recent_scene_count(directory: str, project_id: str) -> int:
    try:
        read = _script_store().read(directory, project_id=project_id)
        state = read.get("state") if read.get("exists") else None
        if not state:
            return 0
        return len(_script_module("schema").numbered_ids(state))
    except Exception:  # noqa: BLE001 - a project without a readable script has no scenes
        return 0


def _recent_project_facts(directory: str, project_id: str) -> dict[str, Any]:
    try:
        roots = _recent_project_folders(directory, project_id)
    except Exception:  # noqa: BLE001 - an unreadable project counts nothing
        roots = []
    media = _recent_media_facts(roots) if roots else {"count": 0, "newest": {}}
    newest = media["newest"]
    cover = None
    for kind in ("image", "video"):
        if kind in newest:
            cover = {"path": newest[kind][1], "kind": kind, "root": newest[kind][2]}
            break
    # Lane w14-shell (FRM-008, MED-046; Resolve's Project Manager shows the count its Media Pool shows): item_count is
    # the Media listing's own counter, files_found of the bounded /media/list walk over the same folders, so Start and
    # Media's All media never disagree. Scenes are not media and no longer count.
    try:
        counted = int(_scan_media_roots(roots)["summary"]["files_found"]) if roots else 0
    except Exception:  # noqa: BLE001 - an unreadable project counts nothing
        counted = 0
    return {"item_count": counted, "cover": cover}


def _recent_with_facts(answer: Any) -> Any:
    if not isinstance(answer, dict) or not isinstance(answer.get("projects"), list):
        return answer
    store = _recent_project_store_factory()
    projects = []
    for project in answer["projects"]:
        if not isinstance(project, dict):
            projects.append(project)
            continue
        facts: dict[str, Any] = {"item_count": 0, "cover": None}
        if project.get("available") is not False:
            try:
                facts = _recent_project_facts(_recent_directory(store, project["project_id"]), project["project_id"])
            except Exception:  # noqa: BLE001 - the list still reads
                pass
        projects.append({**project, **facts})
    # w25-cutfix (PRJ-063): rows that share a name carry their folder's name (the last path part, never a path), the
    # second line Start shows so equal names are told apart (Resolve's Project Manager lists the location likewise).
    names = collections.Counter(p.get("name") for p in projects if isinstance(p, dict))
    for project in projects:
        if isinstance(project, dict) and names[project.get("name")] > 1:
            try:
                project["folder"] = os.path.basename(os.path.normpath(_recent_directory(store, project["project_id"])))[:200]
            except Exception:  # noqa: BLE001 - the list still reads
                pass
    return {**answer, "projects": projects}


_projects_recent_before_facts = projects_recent


async def _projects_recent_with_facts() -> dict[str, Any]:
    answer = await _projects_recent_before_facts()
    return await asyncio.to_thread(_recent_with_facts, answer)


projects_recent = _projects_recent_with_facts  # noqa: F811 - the facade name answers the same as the route
for _route in project_router.routes:
    if getattr(_route, "path", None) == "/projects/recent" and "GET" in getattr(_route, "methods", ()):
        _route.endpoint = _projects_recent_with_facts
        _route.dependant.call = _projects_recent_with_facts


def _announce_project(answer: Any) -> Any:
    """A project made or opened by a route is a new Media scope: media.changed {owner_scope: film_owner:<id>}."""
    try:
        project = answer.get("project") if isinstance(answer, dict) else None
        if isinstance(project, dict) and isinstance(project.get("project_id"), str):
            film_events.media_changed("film_owner:" + project["project_id"])
    except Exception:  # noqa: BLE001 - an event never fails its route
        pass
    return answer


def _announcing(path: str, before: Any) -> Any:
    async def route(request: Any) -> Any:
        return _announce_project(await before(request))
    route.__name__ = before.__name__
    route.__doc__ = before.__doc__
    route.__annotations__ = dict(getattr(before, "__annotations__", {}))
    for _r in project_router.routes:
        if getattr(_r, "path", None) == path and "POST" in getattr(_r, "methods", ()):
            _r.endpoint = route
            _r.dependant.call = route
    return route


projects_create = _announcing("/projects/create", projects_create)  # noqa: F811
projects_open = _announcing("/projects/open", projects_open)  # noqa: F811
projects_open_recent = _announcing("/projects/open-recent", projects_open_recent)  # noqa: F811
