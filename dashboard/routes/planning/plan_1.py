from __future__ import annotations

# --- Planning layer (lane w20-plan item 2): the shot record studio.shot-plan/1 ----------------------------------------
# One record per project (state/shot-plan.json), keyed by the script's shot id; the pure rules live in
# dashboard/planning/shot_plan.py (loaded by path, like dashboard/script). Read joins the old ShotMargin data (Media marks
# shot:<id>) once, durably; write takes per-shot patches against expected_revision (a stale write is refused with
# plan_conflict, never last-writer-wins). Hand and Hermes use the same two routes (continuo_shot_plan_read / _write).
# Lane w20-plan2 (architecture review finding 2): every planning file goes through planning/store.py (ProjectStore's
# cross-process project lock, pinned folder + manifest scope, bounded checked reads, atomic publish); never reset.
_PLANNING_PACKAGE_NAME = "film_suite_planning_core"


def _planning_module(name: str) -> Any:
    with _PACKAGE_LOAD_LOCK:
        if _PLANNING_PACKAGE_NAME not in sys.modules:
            directory = Path(__file__).with_name("planning")
            spec = importlib.util.spec_from_file_location(_PLANNING_PACKAGE_NAME, directory / "__init__.py", submodule_search_locations=[str(directory)])
            if spec is None or spec.loader is None:
                raise studio_core.ProjectError("planning_unavailable", "planning core is unavailable")
            package = importlib.util.module_from_spec(spec)
            sys.modules[_PLANNING_PACKAGE_NAME] = package
            try:
                spec.loader.exec_module(package)
            except BaseException:
                sys.modules.pop(_PLANNING_PACKAGE_NAME, None)
                raise
        return importlib.import_module(f"{_PLANNING_PACKAGE_NAME}.{name}")


class ShotPlanReadRequest(_ProjectRequest):
    project_ref: ProjectReference


class ShotPlanWriteRequest(_ProjectRequest):
    project_ref: ProjectReference
    expected_revision: int = Field(ge=0)
    changes: list[dict[str, Any]] = Field(default_factory=list, max_length=500)
    board: dict[str, Any] | None = None


def _planning_project(reference: ProjectReference) -> str:
    return _project_reference_directory(reference)


def _planning_open(reference: ProjectReference, directory: str) -> Any:
    """One locked transaction over the project's planning files (planning/store.py)."""
    return _planning_module("store").PlanningStore(_project_store_factory()).open(directory, reference.project_id)


def _shot_plan_marked(project_id: str) -> dict[str, list[str]]:
    """Media marks shot:<id> of the project's scope: the pictures the old shot margin showed."""
    marked: dict[str, list[str]] = {}
    for item_id, mark in _media_marks_read(f"film_owner:{project_id}")["items"].items():
        for tag in mark.get("tags") or ():
            if isinstance(tag, str) and _SHOT_TAG_RE.fullmatch(tag):
                marked.setdefault(tag[len("shot:"):], []).append(item_id)
    return marked


def _shot_plan_read(request: ShotPlanReadRequest) -> dict[str, Any]:
    plan = _planning_module("shot_plan")
    directory = _planning_project(request.project_ref)
    try:
        marked = _shot_plan_marked(request.project_ref.project_id)
    except Exception:  # noqa: BLE001 - unreadable marks never fail the plan read; nothing joins this time
        marked = {}
    with _planning_open(request.project_ref, directory) as tx:
        document = tx.load("shot-plan.json")
        migrated = plan.join_marks(document, marked) + (1 if plan.migrate(document) else 0)
        if migrated:
            document["revision"] = int(document.get("revision") or 0) + 1
            tx.save("shot-plan.json", document)
    return {"ok": True, "reason_code": None, "message": None, "migrated": migrated, **plan.public(document)}


def _shot_plan_write(request: ShotPlanWriteRequest) -> dict[str, Any]:
    plan = _planning_module("shot_plan")
    directory = _planning_project(request.project_ref)
    with _planning_open(request.project_ref, directory) as tx:
        document = tx.load("shot-plan.json")
        revision = int(document.get("revision") or 0)
        if request.expected_revision != revision:
            return {"ok": False, "reason_code": "plan_conflict", "message": "The plan changed meanwhile.", "revision": revision}
        # Lane w20-plan2 contract 1: shot elements resolve to the catalogue's ids and existing versions.
        catalogue = tx.load("elements.json")
        _planning_module("elements").migrate(catalogue)
        touched = plan.apply(document, request.changes, request.board, _film_caller(), datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), plan.resolver(catalogue))
        document["revision"] = revision + 1
        tx.save("shot-plan.json", document)
    return {"ok": True, "reason_code": None, "message": None, "touched": touched, **plan.public(document)}


def _planning_answer(work: Any) -> dict[str, Any]:
    plan = _planning_module("shot_plan")
    try:
        return work()
    except plan.PlanRefusal as exc:
        return {"ok": False, "reason_code": exc.reason_code, "message": exc.message}
    except _planning_module("store").PlanningStoreRefusal as exc:
        return {"ok": False, "reason_code": exc.reason_code, "message": exc.message}
    except studio_core.ProjectError as exc:
        return {"ok": False, **_project_error_response(exc)}


@project_router.post("/projects/shot-plan/read")
async def projects_shot_plan_read(request: ShotPlanReadRequest) -> dict[str, Any]:
    """The project's shot record: shots {id: {details, camera, pictures, elements}}, board_order, vocabulary, revision."""
    return await asyncio.to_thread(_planning_answer, lambda: _shot_plan_read(request))


@project_router.post("/projects/shot-plan/write")
async def projects_shot_plan_write(request: ShotPlanWriteRequest) -> dict[str, Any]:
    """Per-shot patches (details keys, null removes; pictures/elements/camera replace) and one scene's board order."""
    answer = await asyncio.to_thread(_planning_answer, lambda: _shot_plan_write(request))
    if answer.get("ok"):
        try:
            film_events.media_changed("film_owner:" + request.project_ref.project_id)
        except Exception:  # noqa: BLE001 - an event never fails a route
            pass
    return answer


# --- Planning layer (lane w20-plan item 6): elements with versions studio.elements/1 ----------------------------------
# state/elements.json. Read seeds from the script (characters, places) and the Media names (references.json, read only;
# Media keeps writing it unchanged); write upserts by key against expected_revision (elements_conflict). Same routes for
# the hand (portraits in the Script inspector) and Hermes (continuo_elements_read / _write).
class ElementsReadRequest(_ProjectRequest):
    project_ref: ProjectReference


class ElementsWriteRequest(_ProjectRequest):
    project_ref: ProjectReference
    expected_revision: int = Field(ge=0)
    changes: list[dict[str, Any]] = Field(default_factory=list, max_length=200)
    remove: list[str] = Field(default_factory=list, max_length=200)


def _elements_seed_sources(reference: ProjectReference, directory: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    derived = None
    try:
        result = _script_store().read(directory, project_id=reference.project_id)
        if result["exists"]:
            derived = _script_view(result["state"], "derived")["derived"]
    except Exception:  # noqa: BLE001 - an unreadable script seeds nothing this time
        derived = None
    try:
        references = _references_public(_references_document(directory))["references"]
    except Exception:  # noqa: BLE001 - unreadable Media names seed nothing this time
        references = []
    return derived, references


def _elements_read(request: ElementsReadRequest) -> dict[str, Any]:
    els = _planning_module("elements")
    directory = _planning_project(request.project_ref)
    derived, references = _elements_seed_sources(request.project_ref, directory)
    with _planning_open(request.project_ref, directory) as tx:
        document = tx.load("elements.json")
        seeded = els.seed(document, derived, references)
        if seeded:
            document["revision"] = int(document.get("revision") or 0) + 1
            tx.save("elements.json", document)
    return {"ok": True, "reason_code": None, "message": None, "seeded": seeded, **els.public(document)}


def _elements_write(request: ElementsWriteRequest) -> dict[str, Any]:
    els = _planning_module("elements")
    directory = _planning_project(request.project_ref)
    with _planning_open(request.project_ref, directory) as tx:
        document = tx.load("elements.json")
        revision = int(document.get("revision") or 0)
        if request.expected_revision != revision:
            return {"ok": False, "reason_code": "elements_conflict", "message": "The elements changed meanwhile.", "revision": revision}
        # Lane w20-plan2 contract 1: versions and elements shots still reference are not removable.
        shots = tx.load("shot-plan.json")
        touched = els.apply(document, request.changes, request.remove, _film_caller(), datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), _planning_module("shot_plan").elements_in_use(shots))
        document["revision"] = revision + 1
        tx.save("elements.json", document)
    return {"ok": True, "reason_code": None, "message": None, "touched": touched, **els.public(document)}


def _elements_answer(work: Any) -> dict[str, Any]:
    els = _planning_module("elements")
    try:
        return work()
    except els.ElementsRefusal as exc:
        return {"ok": False, "reason_code": exc.reason_code, "message": exc.message}
    except _planning_module("store").PlanningStoreRefusal as exc:
        return {"ok": False, "reason_code": exc.reason_code, "message": exc.message}
    except studio_core.ProjectError as exc:
        return {"ok": False, **_project_error_response(exc)}


@project_router.post("/projects/elements/read")
async def projects_elements_read(request: ElementsReadRequest) -> dict[str, Any]:
    """Elements (character, place, prop) with their versions and pictures; script and Media names seed them."""
    return await asyncio.to_thread(_elements_answer, lambda: _elements_read(request))


@project_router.post("/projects/elements/write")
async def projects_elements_write(request: ElementsWriteRequest) -> dict[str, Any]:
    """Upsert elements by key (name, kind, versions with pictures) and remove keys, against expected_revision."""
    answer = await asyncio.to_thread(_elements_answer, lambda: _elements_write(request))
    if answer.get("ok"):
        try:
            film_events.media_changed("film_owner:" + request.project_ref.project_id)
        except Exception:  # noqa: BLE001 - an event never fails a route
            pass
    return answer
