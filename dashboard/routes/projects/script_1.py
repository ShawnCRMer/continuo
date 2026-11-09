from __future__ import annotations


# --- Script (studio.script/1): the project's screenplay, one verb per hand action --------
#
# One document per project, <project>/script/script.json (+ kept revisions and a plain
# script.fountain beside it), through dashboard/script (loaded by path on first use). The
# artist's hand and Hermes use the same routes; every change names expected_revision (the
# document revision last read) and by. A stale change to the same scene (or the order, or a
# card) answers HTTP 409 with code *_conflict and details {current, proposed}; nothing is
# saved, never last-writer-wins. Changes to different scenes never conflict. Other refusals
# are {error: {code, message, retryable, next}} like the timeline. No route returns a path
# outside the project (exports answer exports/script/<name>). Every saved write announces
# script.changed {project_id, revision, by, scenes} (film_events); a refused or unchanged
# write announces nothing.
_SCRIPT_PACKAGE_NAME = "film_suite_script_core"


def _script_module(name: str) -> Any:
    with _PACKAGE_LOAD_LOCK:
        if _SCRIPT_PACKAGE_NAME not in sys.modules:
            directory = Path(__file__).with_name("script")
            spec = importlib.util.spec_from_file_location(
                _SCRIPT_PACKAGE_NAME, directory / "__init__.py", submodule_search_locations=[str(directory)],
            )
            if spec is None or spec.loader is None:
                raise studio_core.ProjectError("script_unavailable", "script core is unavailable")
            package = importlib.util.module_from_spec(spec)
            sys.modules[_SCRIPT_PACKAGE_NAME] = package
            try:
                spec.loader.exec_module(package)
                for part in ("schema", "fountain", "fdx", "ops", "store"):
                    importlib.import_module(f"{_SCRIPT_PACKAGE_NAME}.{part}")
            except BaseException:
                for loaded in list(sys.modules):
                    if loaded == _SCRIPT_PACKAGE_NAME or loaded.startswith(_SCRIPT_PACKAGE_NAME + "."):
                        sys.modules.pop(loaded, None)
                raise
        return importlib.import_module(f"{_SCRIPT_PACKAGE_NAME}.{name}")


def _script_error(exc: BaseException) -> Any:
    store = _script_module("store")
    code = getattr(exc, "code", None)
    if type(code) is not str or code not in store.ERRORS:
        code = "script_unavailable"
    message = str(exc) if isinstance(exc, ValueError) and code != "script_unavailable" else "Script is unavailable"
    body = {"error": store.error_body(code, message, getattr(exc, "details", None))}
    # w16-scr (SCR-073): review_stale_base is a conflict too (HTTP 409 with the record now).
    if code.endswith("_conflict") or code == "review_stale_base":
        return JSONResponse(status_code=409, content=body)
    return body


def _script_view(state: dict[str, Any] | None, view: str) -> dict[str, Any]:
    schema, fountain = _script_module("schema"), _script_module("fountain")
    state = state if state is not None else schema.default_state()
    out: dict[str, Any] = {"derived": schema.derive(state, fountain.elements)}
    if view in ("structured", "both"):
        out["state"] = state
    if view in ("fountain", "both"):
        out["fountain"] = fountain.write(state, ids=True)
    return out


class ScriptReadRequest(_ProjectRequest):
    project_ref: ProjectReference
    view: Literal["structured", "fountain", "both"] = "structured"


class ScriptRevisionsRequest(_ProjectRequest):
    project_ref: ProjectReference
    revision: int | None = Field(default=None, ge=1)
    view: Literal["structured", "fountain", "both"] = "structured"


class _ScriptChange(_ProjectRequest):
    project_ref: ProjectReference
    expected_revision: int = Field(ge=0)
    by: Literal["artist", "hermes"] = "artist"


class ScriptPlace(_ProjectRequest):
    """Where in the order: index, or before/after a scene or act break id (one of them)."""
    index: int | None = Field(default=None, ge=0)
    before: str | None = Field(default=None, max_length=12)
    after: str | None = Field(default=None, max_length=12)


class ScriptSceneAddRequest(_ScriptChange):
    text: str = Field(default="", max_length=65536)
    synopsis: str = Field(default="", max_length=2000)
    # The stable id of each Shot line (sh-1a2b3c4d, or null for a new one), in text order: the
    # editor carries them through edits and undo; without it the backend aligns old and new lines.
    shots: list[str | None] | None = Field(default=None, max_length=500)
    at: ScriptPlace | None = None
    aside: bool = False


class ScriptSceneWriteRequest(_ScriptChange):
    scene: str = Field(max_length=12)
    text: str | None = Field(default=None, max_length=65536)
    synopsis: str | None = Field(default=None, max_length=2000)
    shots: list[str | None] | None = Field(default=None, max_length=500)


class ScriptSceneOmitRequest(_ScriptChange):
    scene: str = Field(max_length=12)
    omitted: bool = True


class ScriptSceneDeleteRequest(_ScriptChange):
    scene: str = Field(max_length=12)
    force: bool = False


class ScriptMoveRequest(_ScriptChange):
    items: list[str] = Field(min_length=1, max_length=2200)
    at: ScriptPlace | None = None


class ScriptAsideRequest(_ScriptChange):
    scenes: list[str] = Field(min_length=1, max_length=2000)
    aside: bool = True
    at: ScriptPlace | None = None


class ScriptBreakAddRequest(_ScriptChange):
    title: str = Field(min_length=1, max_length=120)
    at: ScriptPlace | None = None


class ScriptBreakRenameRequest(_ScriptChange):
    act_break: str = Field(alias="break", max_length=12)
    title: str = Field(min_length=1, max_length=120)


class ScriptBreakRemoveRequest(_ScriptChange):
    act_break: str = Field(alias="break", max_length=12)


class ScriptCardPlaceRequest(_ScriptChange):
    scene: str = Field(max_length=12)
    x: float | None = None
    y: float | None = None
    clear: bool = False


class ScriptNumbersRequest(_ScriptChange):
    locked: bool


class ScriptPageSizeRequest(_ScriptChange):
    size: Literal["letter", "a4"]


class ScriptTitleEntry(_ProjectRequest):
    key: str = Field(min_length=1, max_length=40)
    value: str = Field(max_length=1000)


class ScriptTitlePageRequest(_ScriptChange):
    entries: list[ScriptTitleEntry] = Field(max_length=20)


class ScriptWriteRequest(_ScriptChange):
    fountain: str = Field(max_length=8 * 1024 * 1024)


class ScriptImportRequest(_ScriptChange):
    format: Literal["fountain", "fdx"]
    text: str = Field(min_length=1, max_length=8 * 1024 * 1024)
    name: str = Field(default="", max_length=120)


class ScriptExportRequest(_ProjectRequest):
    project_ref: ProjectReference
    format: Literal["fountain", "fdx", "pdf"]
    name: str = Field(default="", max_length=100)


def _script_store() -> Any:
    return _script_module("store").ScriptStore(_project_store_factory())


def _script_place(at: ScriptPlace | None) -> dict[str, Any] | None:
    if at is None:
        return None
    given = {key: value for key, value in at.model_dump().items() if value is not None}
    if len(given) != 1:
        raise _script_module("schema").ScriptError("invalid_op", "at: needs exactly one of index, before, after")
    return given


_SCRIPT_LABELS = {
    "scene_add": "add scene", "scene_write": "write scene", "scene_omit": "omit scene", "scene_delete": "delete scene",
    "move": "move", "aside": "aside", "break_add": "add act break", "break_rename": "rename act break",
    "break_remove": "remove act break", "card_place": "place card", "numbers_lock": "scene numbers",
    "title_page": "title page", "write_fountain": "write script", "restore": "restore", "page_size": "page size",
}


def _script_by(request: Any) -> str:
    """The author of a script change (w15-scr, SCR-072): an agent call is always Hermes' revision, whatever it claims."""
    return "hermes" if _film_caller() == "hermes" else request.by


async def _script_change(request: _ScriptChange, verb: str, args: dict[str, Any]) -> Any:
    by = _script_by(request)
    try:
        directory = _project_reference_directory(request.project_ref)
        ops = _script_module("ops")

        def mutate(state: Any, _current: int, revision: int) -> Any:
            return ops.apply(state, verb, args, expected_revision=request.expected_revision, revision=revision, by=by)

        label = ("Hermes: " if by == "hermes" else "") + _SCRIPT_LABELS[verb]
        done = await asyncio.to_thread(
            _script_store().change, directory, project_id=request.project_ref.project_id,
            expected_revision=request.expected_revision, mutate=mutate, label=label, by=by,
            wholesale=verb == "write_fountain",
        )
        answer: dict[str, Any] = {"saved": done["saved"], "revision": done["revision"], **done["result"],
                                  # w14-fix (S4): false when the save could not update the review record (damaged file).
                                  "review_saved": bool(done.get("review_saved", True))}
        if done["saved"]:
            film_events.script_changed(request.project_ref.project_id, done["revision"], by, ops.touched(done["result"]))
        state = done["state"]
        sid = done["result"].get("scene")
        if state is not None and isinstance(sid, str) and sid in state["scenes"]:
            answer["scene_state"] = ops.public_scene(sid, state["scenes"][sid])
        if state is not None and verb in ("scene_add", "move", "aside", "break_add", "break_remove", "numbers_lock", "scene_delete", "write_fountain"):
            schema = _script_module("schema")
            answer["order"] = state["order"]
            answer["numbers"] = schema.numbers(state)
        return answer
    except (studio_core.ProjectError, ValueError) as exc:
        return _script_error(exc)


_LEGACY_SHOT_TAG_RE = re.compile(r"^shot:s-[0-9a-f]{8}\.\d{1,3}$")
_SHOT_TAG_RE = re.compile(r"^shot:sh-[0-9a-f]{8}$")


def _script_shot_marks(project_id: str, shots: list[dict[str, Any]]) -> dict[str, Any]:
    """The pictures assigned to the script's shots (Media mark tags of the project's scope).

    A stopgap tag ``shot:<scene id>.<n>`` (written before shots had ids) is migrated once, here:
    to ``shot:<id>`` of the shot at that position now, saved through the marks store; a tag that
    matches no shot stays as it is and is listed in ``missing`` (the UI says "Shot missing"),
    never dropped. Answers {by_shot: {shot id: [item ids]}, missing: [{item_id, tag}], migrated}."""
    scope = f"film_owner:{project_id}"
    marks = _media_marks_read(scope)
    by_legacy = {shot["legacy_tag"]: shot["tag"] for shot in shots}
    known = {shot["tag"] for shot in shots}
    changes = []
    for item_id, mark in marks["items"].items():
        tags = [tag for tag in mark.get("tags") or () if isinstance(tag, str)]
        moved = [by_legacy.get(tag, tag) if _LEGACY_SHOT_TAG_RE.fullmatch(tag) else tag for tag in tags]
        if moved != tags:
            changes.append(MediaMarkChange(item_id=item_id, tags=moved))
    migrated = 0
    if changes:
        request = MediaMarksWriteRequest.model_validate({"schema": "studio.media-marks/1", "owner_scope": scope,
                                                         "expected_revision": marks["revision"], "changes": [c.model_dump(exclude_none=True) for c in changes]})
        marks = _media_marks_write(request)
        migrated = len(changes)
        try:
            film_events.media_changed(scope)
        except Exception:  # noqa: BLE001 - an event never fails a route
            pass
    by_shot: dict[str, list[str]] = {}
    missing = []
    for item_id, mark in marks["items"].items():
        for tag in mark.get("tags") or ():
            if not isinstance(tag, str) or not (_SHOT_TAG_RE.fullmatch(tag) or _LEGACY_SHOT_TAG_RE.fullmatch(tag)):
                continue
            if tag in known:
                by_shot.setdefault(tag[len("shot:"):], []).append(item_id)
            else:
                missing.append({"item_id": item_id, "tag": tag})
    return {"by_shot": by_shot, "missing": missing, "migrated": migrated}


@project_router.post("/projects/script/read")
async def projects_script_read(request: ScriptReadRequest) -> Any:
    try:
        directory = _project_reference_directory(request.project_ref)
        result = await asyncio.to_thread(_script_store().read, directory, project_id=request.project_ref.project_id)
        view = _script_view(result["state"], request.view)
        if result["exists"]:
            # Pictures per shot (Media marks shot:<id>); legacy shot:<scene>.<n> tags migrate once here.
            try:
                view["derived"]["shot_pictures"] = await asyncio.to_thread(_script_shot_marks, request.project_ref.project_id, view["derived"]["shots"])
            except Exception:  # noqa: BLE001 - the marks never fail the script read
                view["derived"]["shot_pictures"] = None
        return {key: result[key] for key in ("exists", "revision", "updated_at", "label", "by")} | view
    except (studio_core.ProjectError, ValueError) as exc:
        return _script_error(exc)


@project_router.post("/projects/script/revisions")
async def projects_script_revisions(request: ScriptRevisionsRequest) -> Any:
    try:
        directory = _project_reference_directory(request.project_ref)
        store = _script_store()
        if request.revision is None:
            return await asyncio.to_thread(store.revisions, directory, project_id=request.project_ref.project_id)
        result = await asyncio.to_thread(store.read_revision, directory, project_id=request.project_ref.project_id, revision=request.revision)
        return {key: result[key] for key in ("revision", "updated_at", "label", "by")} | _script_view(result["state"], request.view)
    except (studio_core.ProjectError, ValueError) as exc:
        return _script_error(exc)


@project_router.post("/projects/script/scene/add")
async def projects_script_scene_add(request: ScriptSceneAddRequest) -> Any:
    try:
        at = _script_place(request.at)
    except ValueError as exc:
        return _script_error(exc)
    args = {"text": request.text, "synopsis": request.synopsis, "at": at, "aside": request.aside}
    if request.shots is not None:
        args["shots"] = request.shots
    return await _script_change(request, "scene_add", args)


@project_router.post("/projects/script/scene/write")
async def projects_script_scene_write(request: ScriptSceneWriteRequest) -> Any:
    args: dict[str, Any] = {"scene": request.scene}
    if request.text is not None:
        args["text"] = request.text
    if request.synopsis is not None:
        args["synopsis"] = request.synopsis
    if request.shots is not None:
        args["shots"] = request.shots
    return await _script_change(request, "scene_write", args)


@project_router.post("/projects/script/scene/omit")
async def projects_script_scene_omit(request: ScriptSceneOmitRequest) -> Any:
    return await _script_change(request, "scene_omit", {"scene": request.scene, "omitted": request.omitted})


@project_router.post("/projects/script/scene/delete")
async def projects_script_scene_delete(request: ScriptSceneDeleteRequest) -> Any:
    return await _script_change(request, "scene_delete", {"scene": request.scene, "force": request.force})


@project_router.post("/projects/script/move")
async def projects_script_move(request: ScriptMoveRequest) -> Any:
    try:
        at = _script_place(request.at)
    except ValueError as exc:
        return _script_error(exc)
    return await _script_change(request, "move", {"items": request.items, "at": at})


@project_router.post("/projects/script/aside")
async def projects_script_aside(request: ScriptAsideRequest) -> Any:
    try:
        at = _script_place(request.at)
    except ValueError as exc:
        return _script_error(exc)
    return await _script_change(request, "aside", {"scenes": request.scenes, "aside": request.aside, "at": at})


@project_router.post("/projects/script/break/add")
async def projects_script_break_add(request: ScriptBreakAddRequest) -> Any:
    try:
        at = _script_place(request.at)
    except ValueError as exc:
        return _script_error(exc)
    return await _script_change(request, "break_add", {"title": request.title, "at": at})


@project_router.post("/projects/script/break/rename")
async def projects_script_break_rename(request: ScriptBreakRenameRequest) -> Any:
    return await _script_change(request, "break_rename", {"break": request.act_break, "title": request.title})


@project_router.post("/projects/script/break/remove")
async def projects_script_break_remove(request: ScriptBreakRemoveRequest) -> Any:
    return await _script_change(request, "break_remove", {"break": request.act_break})


@project_router.post("/projects/script/card/place")
async def projects_script_card_place(request: ScriptCardPlaceRequest) -> Any:
    if request.clear:
        return await _script_change(request, "card_place", {"scene": request.scene, "card": None})
    return await _script_change(request, "card_place", {"scene": request.scene, "x": request.x, "y": request.y})


@project_router.post("/projects/script/numbers")
async def projects_script_numbers(request: ScriptNumbersRequest) -> Any:
    return await _script_change(request, "numbers_lock", {"locked": request.locked})


@project_router.post("/projects/script/page-size")
async def projects_script_page_size(request: ScriptPageSizeRequest) -> Any:
    """US Letter (default) or A4 for the editor, the PDF and derived.pages (one page engine)."""
    return await _script_change(request, "page_size", {"size": request.size})


@project_router.post("/projects/script/title-page")
async def projects_script_title_page(request: ScriptTitlePageRequest) -> Any:
    return await _script_change(request, "title_page", {"entries": [entry.model_dump() for entry in request.entries]})


@project_router.post("/projects/script/write")
async def projects_script_write(request: ScriptWriteRequest) -> Any:
    base = None
    if request.expected_revision > 0:
        try:  # the state the writer read (kept revisions never change): untouched parts never conflict
            directory = _project_reference_directory(request.project_ref)
            base = (await asyncio.to_thread(_script_store().read_revision, directory,
                                            project_id=request.project_ref.project_id, revision=request.expected_revision))["state"]
        except (studio_core.ProjectError, ValueError):
            base = None
    return await _script_change(request, "write_fountain", {"fountain": request.fountain, "base": base})


@project_router.post("/projects/script/import")
async def projects_script_import(request: ScriptImportRequest) -> Any:
    by = _script_by(request)
    try:
        directory = _project_reference_directory(request.project_ref)
        ops = _script_module("ops")
        reader = _script_module("fdx" if request.format == "fdx" else "fountain")
        parsed = await asyncio.to_thread(reader.parse, request.text)
        report: dict[str, Any] = {}

        def mutate(state: Any, current: int, revision: int) -> Any:
            if request.expected_revision != current:
                raise _script_module("schema").ScriptError(
                    "script_conflict", f"importing replaces the script and needs its current revision ({current})")
            built, found = ops.import_state(parsed, revision, by)
            report.update(found)
            return built, found, True

        label = ("Hermes: " if by == "hermes" else "") + f"import {request.format}" + (f" {request.name}" if request.name else "")
        done = await asyncio.to_thread(
            _script_store().change, directory, project_id=request.project_ref.project_id,
            expected_revision=request.expected_revision, mutate=mutate, label=label, by=by, wholesale=True,
        )
        schema = _script_module("schema")
        if done["saved"]:
            film_events.script_changed(request.project_ref.project_id, done["revision"], by, ())
        return {"saved": True, "revision": done["revision"], "imported": report,
                "order": done["state"]["order"], "numbers": schema.numbers(done["state"]),
                "aside": schema.aside_ids(done["state"])}
    except (studio_core.ProjectError, ValueError) as exc:
        return _script_error(exc)


@project_router.post("/projects/script/export")
async def projects_script_export(request: ScriptExportRequest) -> Any:
    try:
        directory = _project_reference_directory(request.project_ref)
        project_id = request.project_ref.project_id
        store = _script_store()
        current = await asyncio.to_thread(store.read, directory, project_id=project_id)
        if not current["exists"]:
            raise _script_module("schema").ScriptError("not_found", "the project has no script yet")
        state = current["state"]
        extra: dict[str, Any] = {}
        if request.format == "pdf":
            # The PDF is laid out by pages.py (one page count: editor, PDF, derived.pages).
            printed = await asyncio.to_thread(_script_module("pdf").write, state, _script_module("fountain").elements)
            data, lost = printed["data"], {}
            extra = {"pages": printed["pages"], "title_page": printed["title_page"]}
        else:
            if request.format == "fdx":
                text, lost = _script_module("fdx").write(state)
            else:
                text, lost = _script_module("fountain").write(state, ids=False), {}
            data = text.encode("utf-8")
        base = request.name or "script"
        name = f"{base}-r{current['revision']}.{request.format}"
        relative = await asyncio.to_thread(store.export, directory, project_id=project_id, name=name, data=data)
        return {"revision": current["revision"], "format": request.format, "file": relative, "bytes": len(data), "not_exported": lost, **extra}
    except (studio_core.ProjectError, ValueError) as exc:
        return _script_error(exc)


# --- History (lane w10-script item 3, board E21 inspector Scene | History) --------------
# Undo / Restore in the History tab: a kept revision comes back as a new revision (label
# "back to version N", by artist or hermes), like the Cut's Restore. Whole document, so
# expected_revision is the current revision; the page reloads the editor after it.
class ScriptRestoreRequest(_ScriptChange):
    revision: int = Field(ge=1)


class ScriptReviewUndo(_ProjectRequest):
    scene: str = Field(max_length=12)
    text: str = Field(max_length=65536)


class ScriptReviewRequest(_ProjectRequest):
    """Hermes' changes under review (lane w14-scr, SCR-071): read; baselines {scene id: text|null} accepts; clear = Accept
    all; undo [{scene, text}] + expected_revision writes those scenes back as one revision "Undo Hermes change", by the
    caller (the page: the artist; an agent tool call: Hermes, labelled "Hermes: ..."); only scenes in the record."""
    project_ref: ProjectReference
    baselines: dict[str, str | None] | None = Field(default=None, max_length=2000)
    clear: bool = False
    undo: list[ScriptReviewUndo] | None = Field(default=None, max_length=2000)
    expected_revision: int | None = Field(default=None, ge=0)
    # w14-fix (S2 + S3): baselines / clear name the record's hermes_revision they were decided on (review_conflict).
    expected_hermes_revision: int | None = Field(default=None, ge=0)
    # w16-scr (SCR-073): the script revision the baselines were computed on (review_stale_base).
    base_revision: int | None = Field(default=None, ge=0)
    # w15-scr (SCR-072): the page names Hermes when Hermes asked for the undo through the page bridge.
    by: Literal["artist", "hermes"] = "artist"


@project_router.post("/projects/script/review")
async def projects_script_review(request: ScriptReviewRequest) -> Any:
    try:
        directory = _project_reference_directory(request.project_ref)
        store = _script_store()
        if request.undo:
            if request.expected_revision is None:
                raise _script_module("schema").ScriptError("invalid_op", "undo: needs expected_revision")
            ops = _script_module("ops")
            undo = [item.model_dump() for item in request.undo]
            who = _script_by(request)  # Hermes' undo is Hermes' revision (w14-fix S1; page bridge: w15-scr SCR-072)

            def mutate(state: Any, _current: int, revision: int) -> Any:
                if state is None:
                    raise _script_module("schema").ScriptError("not_found", "the project has no script yet")
                changed = False
                for item in undo:
                    state, _result, did = ops.apply(state, "scene_write", {"scene": item["scene"], "text": item["text"]},
                                                    expected_revision=request.expected_revision, revision=revision, by=who)
                    changed = changed or did
                return state, {"undone": [item["scene"] for item in undo]}, changed

            label = ("Hermes: " if who == "hermes" else "") + _script_module("review").UNDO_LABEL + ("s" if len(undo) > 1 else "")
            done = await asyncio.to_thread(store.change, directory, project_id=request.project_ref.project_id,
                                           expected_revision=request.expected_revision, mutate=mutate, label=label, by=who,
                                           undo_scenes=[item["scene"] for item in undo])
            if done["saved"]:
                film_events.script_changed(request.project_ref.project_id, done["revision"], who, [item["scene"] for item in undo])
        out = await asyncio.to_thread(store.review, directory, project_id=request.project_ref.project_id,
                                      baselines=request.baselines, clear=request.clear,
                                      expected_hermes_revision=request.expected_hermes_revision,
                                      base_revision=request.base_revision)
        if request.baselines or request.clear:
            film_events.script_changed(request.project_ref.project_id, out["revision"], _film_caller(), ())
        if request.undo:
            out = {**out, "saved": done["saved"], "undone": done["result"]["undone"]}
        return out
    except (studio_core.ProjectError, ValueError) as exc:
        return _script_error(exc)


@project_router.post("/projects/script/restore")
async def projects_script_restore(request: ScriptRestoreRequest) -> Any:
    by = _script_by(request)
    try:
        directory = _project_reference_directory(request.project_ref)
        store = _script_store()
        kept = await asyncio.to_thread(store.read_revision, directory, project_id=request.project_ref.project_id, revision=request.revision)
        ops = _script_module("ops")

        def mutate(state: Any, current: int, revision: int) -> Any:
            if state is None:
                raise _script_module("schema").ScriptError("not_found", "the project has no script yet")
            return ops.apply(state, "restore", {"target": kept["state"], "target_revision": request.revision, "current": current},
                             expected_revision=request.expected_revision, revision=revision, by=by)

        label = ("Hermes: " if by == "hermes" else "") + f"back to version {request.revision}"
        done = await asyncio.to_thread(
            store.change, directory, project_id=request.project_ref.project_id,
            expected_revision=request.expected_revision, mutate=mutate, label=label, by=by, wholesale=True,
        )
        if done["saved"]:
            film_events.script_changed(request.project_ref.project_id, done["revision"], by, ())
        return {"saved": done["saved"], "revision": done["revision"], "restored": request.revision}
    except (studio_core.ProjectError, ValueError) as exc:
        return _script_error(exc)
