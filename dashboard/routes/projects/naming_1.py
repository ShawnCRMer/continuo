from __future__ import annotations

# --- Lane p6-name items 3 + 4 (PRJ-002/013/014, Project charter decisions 9 + 10) -----------------------------------
# PUT /projects/settings/naming: the project's file-name template (Project settings › File names, and Hermes'
# continuo_project_naming_write: it is not a folder, so decision 4 does not apply). A template that could not stay
# unique (no {take} and no {date}) is refused with reason name_template_not_unique; null = the default.
# POST /projects/rename: the display name only (manifest + recent list); the folder is never renamed. By hand (the
# name field in Project settings) and by Hermes (continuo_project_rename). Both announce project.changed.
# POST /projects/state also answers name_template, name_template_default and name_example (the next real name).


class ProjectNamingWriteRequest(_ProjectRequest):
    project_ref: ProjectReference
    template: str | None = Field(default=None, max_length=400)


class ProjectRenameRequest(_ProjectRequest):
    project_ref: ProjectReference
    name: str = Field(min_length=1, max_length=120)


def _project_naming_public(directory: str, project_id: str) -> dict[str, Any]:
    manifest = _project_store_factory().read_project(directory, expected_project_id=project_id)
    return _naming.public(directory, _project_output_dir(directory, project_id), manifest.get("name") or "Project")


def _project_changed(project_id: str, name: str) -> None:
    try:
        film_events.project_changed(project_id, name, _film_caller())
    except Exception:  # noqa: BLE001 - an event never fails a route
        pass


@project_router.put("/projects/settings/naming")
async def projects_settings_naming(request: ProjectNamingWriteRequest) -> dict[str, Any]:
    """Project settings › File names: the one naming template of the project (artist and Hermes)."""
    try:
        directory = _project_reference_directory(request.project_ref)
        _project_store_factory().read_project(directory, expected_project_id=request.project_ref.project_id)
        await asyncio.to_thread(_naming.write_template, directory, request.template)
        public = _project_naming_public(directory, request.project_ref.project_id)
    except _naming.NamingError as exc:
        return {"ok": False, "reason_code": exc.reason_code, "message": exc.message}
    except studio_core.ProjectError as exc:
        return {"ok": False, **_project_error_response(exc)}
    try:
        film_events.shoot_changed(request.project_ref.project_id, _film_caller())
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True, "reason_code": None, "message": None, **public}


@project_router.post("/projects/rename")
async def projects_rename(request: ProjectRenameRequest) -> dict[str, Any]:
    """Rename the project: its display name only (manifest and recent list); the folder stays where it is."""
    name = request.name.strip()
    if not name or any(ord(ch) < 32 for ch in name):
        return {"ok": False, "reason_code": "name_invalid", "message": "The name is empty or has a control character."}
    try:
        directory = _project_reference_directory(request.project_ref)
        store = _project_store_factory()
        manifest = store.read_project(directory, expected_project_id=request.project_ref.project_id)
        if manifest["name"] != name:
            store.write_metadata(directory, request.project_ref.project_id, manifest["revision"], name, list(manifest["approved_roots"]))
        try:
            _recent_project_store_factory().record_recent(request.project_ref.project_id, name, directory)
        except Exception:  # noqa: BLE001 - the recent list catches up on the next open
            pass
    except studio_core.ProjectError as exc:
        return {"ok": False, **_project_error_response(exc)}
    _project_changed(request.project_ref.project_id, name)
    return {"ok": True, "reason_code": None, "message": None, "project_id": request.project_ref.project_id, "name": name,
            "folder_renamed": False}


_projects_state_read_before_naming = projects_state


async def _projects_state_with_naming(request: ProjectStateReadRequest) -> dict[str, Any]:
    answer = await _projects_state_read_before_naming(request)
    if isinstance(answer, dict) and "state" in answer:
        try:
            directory = _project_reference_directory(request.project_ref)
            answer = {**answer, **_project_naming_public(directory, request.project_ref.project_id)}
        except Exception:  # noqa: BLE001 - the state still reads
            pass
    return answer


projects_state = _projects_state_with_naming  # noqa: F811 - the facade name answers the same as the route
for _route in project_router.routes:
    if getattr(_route, "path", None) == "/projects/state" and "POST" in getattr(_route, "methods", ()):
        _route.endpoint = _projects_state_with_naming
        _route.dependant.call = _projects_state_with_naming
