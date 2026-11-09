from __future__ import annotations


def _project_store_factory() -> Any:
    return studio_core.ProjectStore(hermes_home=_active_hermes_home())


def _recent_project_store_factory() -> Any:
    return studio_core.RecentProjectStore(hermes_home=_active_hermes_home())


def _project_error_response(exc: BaseException) -> dict[str, Any]:
    code = getattr(exc, "code", None)
    if type(code) is not str or code not in _PROJECT_ERROR_CODES:
        code = "project_unavailable"
    return {"error": {"code": code, "message": _PROJECT_ERROR_MESSAGES[code]}}


def _project_call(operation: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        return operation(*args, **kwargs)
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)


def _public_project_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep project/open responses on the exact path-free public shape."""
    project = payload["project"]
    shoot = payload["shoot"]
    result: dict[str, Any] = {
        "project": {
            "project_id": project["project_id"],
            "name": project["name"],
            "revision": project["revision"],
        },
        "shoot": {
            "revision": shoot["revision"],
            "state": shoot["state"],
        },
    }
    if payload.get("warning") == getattr(studio_core, "RECENT_WARNING", "recents_unavailable"):
        result["warning"] = "recents_unavailable"
    return result


def _public_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metadata[key]
        for key in ("project_id", "name", "revision", "created_at", "updated_at")
        if key in metadata
    }
