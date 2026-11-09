from __future__ import annotations


@project_router.post("/projects/state")
async def projects_state(request: ProjectStateReadRequest) -> dict[str, Any]:
    try:
        directory = _project_reference_directory(request.project_ref)
        return _project_store_factory().read_shoot_state(
            directory,
            project_id=request.project_ref.project_id,
        )
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)


@project_router.put("/projects/state")
async def projects_state_write(request: ProjectStateWriteRequest) -> dict[str, Any]:
    # Lane p5d-frm (Project decision 2 + 4): a legacy outputFolder in the Video state is the project's output folder
    # while the manifest names none, so changing it is a folder decision: the artist's hand only, like a folder approval.
    if _film_caller() == "hermes" and _state_output_changed(request):
        return _folder_refusal()
    try:
        directory = _project_reference_directory(request.project_ref)
        written = _project_store_factory().write_shoot_state(
            directory,
            project_id=request.project_ref.project_id,
            expected_revision=request.expected_revision,
            state=request.state,
        )
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    # Lane w27-rows item 1: a save that changed nothing kept the revision (write_shoot_state); nothing to announce.
    if isinstance(written, dict) and written.get("revision") == request.expected_revision:
        return written
    # shoot.changed (lane w5-backend item 2): who wrote it - an agent tool call is Hermes.
    try:
        film_events.shoot_changed(request.project_ref.project_id, _film_caller())
    except Exception:  # noqa: BLE001 - an event never fails a route
        pass
    return written


@project_router.put("/projects/metadata")
async def projects_metadata(request: ProjectMetadataRequest) -> dict[str, Any]:
    if _film_caller() == "hermes":  # lane p5-prj (HER-012): only the artist's hand approves folders
        return _folder_refusal()
    try:
        directory = _project_reference_directory(request.project_ref)
        metadata = _project_store_factory().write_metadata(
            directory,
            project_id=request.project_ref.project_id,
            expected_revision=request.expected_revision,
            name=request.name,
            approved_roots=request.approved_roots,
        )
        return _public_metadata(metadata)
    except studio_core.ProjectError as exc:
        if getattr(exc, "code", None) == studio_core.project_store.PROJECT_FOLDER_OFFLINE:  # lane p13-prjb finding 1
            return {"error": {"code": exc.code, "message": str(exc)}}
        return _project_error_response(exc)


def _state_output_changed(request: Any) -> bool:
    """True when a state write would change the stored legacy outputFolder (absent = unchanged)."""
    state = request.state if isinstance(request.state, dict) else {}
    if "outputFolder" not in state:
        return False
    try:
        stored = _project_state_output(_project_reference_directory(request.project_ref)) or ""
    except Exception:  # noqa: BLE001 - unreadable: only an empty value is unchanged
        stored = ""
    return str(state.get("outputFolder") or "").strip() != stored.strip()
