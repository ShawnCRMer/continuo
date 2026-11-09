from __future__ import annotations


@image_catalog_router.post("/image/sources/catalog")
async def image_source_catalog(
    request: ImageSourceCatalogRequest,
    http_request: Request,
) -> dict[str, Any]:
    """Read the already initialized profile-global Image source catalog."""
    try:
        caller_scope_id = _image_film_caller_scope_id(http_request)
        async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
            root = _initialized_image_film_root()
            return root.profile_image_source_catalog(caller_scope_id)
    except _ImageFilmCallerScopeError as exc:
        return _image_catalog_error_response(exc)
    except _ImageFilmCompositionError as exc:
        return _image_catalog_error_response(exc)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_catalog_error_response(exc)
