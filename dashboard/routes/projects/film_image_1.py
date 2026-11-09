from __future__ import annotations


@project_router.post("/projects/film-image/preview")
async def film_image_preview(
    request: FilmImagePreviewRequest,
    http_request: Request,
) -> dict[str, Any]:
    """Create a session-only new-reference preview without durable writes."""
    try:
        caller_scope_id = _image_film_caller_scope_id(http_request)
        payload = request.model_dump(by_alias=True, mode="python")
        async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
            root = _initialized_image_film_root()
            handoff = root.read_cached_project_operations(request.target_scope.project_id)
            return handoff.preview(payload, caller_scope_id)
    except _ImageFilmCallerScopeError as exc:
        return _image_film_error_response(exc)
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_film_error_response(exc)


film_image_preview_new_reference = film_image_preview


@project_router.post("/projects/film-image/preview/cancel")
async def film_image_preview_cancel(
    request: FilmImagePreviewCancelRequest,
    http_request: Request,
) -> dict[str, Any]:
    """Consume a matching session-only preview token without a store write."""
    try:
        caller_scope_id = _image_film_caller_scope_id(http_request)
        payload = request.model_dump(by_alias=True, mode="python")
        async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
            root = _initialized_image_film_root()
            handoff = root.read_cached_project_operations(request.target_scope.project_id)
            return handoff.cancel(payload, caller_scope_id)
    except _ImageFilmCallerScopeError as exc:
        return _image_film_error_response(exc)
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_film_error_response(exc)


film_image_preview_cancel_operation = film_image_preview_cancel


@project_router.post("/projects/film-image/prepare")
async def film_image_prepare(
    request: FilmImagePrepareRequest,
    http_request: Request,
) -> dict[str, Any]:
    """Prepare one new-reference operation through the initialized project handoff."""
    try:
        caller_scope_id = _image_film_caller_scope_id(http_request)
        payload = request.model_dump(by_alias=True, mode="python")
        async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
            root = _initialized_image_film_root()
            handoff = root.read_cached_project_operations(request.target_scope.project_id)
            return handoff.prepare(payload, caller_scope_id)
    except _ImageFilmCallerScopeError as exc:
        return _image_film_error_response(exc)
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_film_error_response(exc)


@project_router.post("/projects/film-image/execute")
async def film_image_execute(
    request: FilmImageExecuteRequest,
    http_request: Request,
) -> dict[str, Any]:
    """Execute one prepared new-reference operation through the cached handoff."""
    try:
        caller_scope_id = _image_film_caller_scope_id(http_request)
        payload = request.model_dump(by_alias=True, mode="python")
        async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
            root = _initialized_image_film_root()
            handoff = root.read_cached_project_operations(request.target_scope.project_id)
            return handoff.execute(payload, caller_scope_id)
    except _ImageFilmCallerScopeError as exc:
        return _image_film_error_response(exc)
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_film_error_response(exc)


@project_router.post("/projects/film-image/status")
async def film_image_status(
    request: FilmImageStatusRequest,
    http_request: Request,
) -> dict[str, Any]:
    """Read one accepted Image/Film operation through the cached handoff."""
    try:
        caller_scope_id = _image_film_caller_scope_id(http_request)
        payload = request.model_dump(by_alias=True, mode="python")
        project_id = request.project_id
        async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
            root = _initialized_image_film_root()
            handoff = root.read_cached_project_operations(project_id)
            return handoff.status(payload, caller_scope_id)
    except _ImageFilmCallerScopeError as exc:
        return _image_film_error_response(exc)
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_film_error_response(exc)
