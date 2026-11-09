from __future__ import annotations


router = APIRouter()


# Lane w11-backend M-2 / U-42: these routes are no project request; a bad body
# names its first wrong field (a field name of the request model, never a value).
_NAMED_FIELD_PATHS = frozenset({"/media/feed", "/media/marks/read", "/projects/timeline/export", "/projects/timeline/export/status"})  # p13b-cut: export wait_ms


def _named_field_error(exc: Any) -> JSONResponse | None:
    try:
        location = exc.errors()[0].get("loc") or ()
        field = next((str(part) for part in reversed(location) if isinstance(part, str) and part != "body"), None)
    except Exception:  # noqa: BLE001 - fall back to the finite project answer
        return None
    if not field or len(field) > 64 or not field.replace("_", "").isalnum():
        return None
    return JSONResponse(status_code=422, content={"error": {
        "code": "invalid_request", "field": field, "message": f"The field {field} is missing or invalid"}})


class FiniteValidationRoute(APIRoute):
    """Translate only request-model failures into a finite project response."""

    def get_route_handler(self):
        original_route_handler = super().get_route_handler()
        # Lane p5b-bck (BCK-032): the desktop mounts this router under /api/plugins/<name>, so the
        # route's path carries that prefix; match the route's own path at the end, not the whole path.
        named = any(self.path == p or self.path.endswith(p) for p in _NAMED_FIELD_PATHS)

        async def finite_handler(request):
            try:
                return await original_route_handler(request)
            except RequestValidationError as exc:
                answer = _named_field_error(exc) if named else None
                if answer is not None:
                    return answer
                return JSONResponse(
                    status_code=422,
                    content={
                        "error": {
                            "code": "invalid_project",
                            "message": "Project request is invalid",
                        }
                    },
                )

        return finite_handler


class ImageCatalogValidationRoute(APIRoute):
    """Translate catalog request-model failures to its finite public error."""

    def get_route_handler(self):
        original_route_handler = super().get_route_handler()

        async def finite_handler(request):
            try:
                return await original_route_handler(request)
            except RequestValidationError:
                return JSONResponse(
                    status_code=422,
                    content={
                        "error": {
                            "code": "invalid_image_catalog_request",
                            "message": "Image catalog request is invalid",
                        }
                    },
                )

        return finite_handler
