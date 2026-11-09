"""Style browser HTTP routes (masterplan package 6). See README.md.

Independently importable: ``create_style_router`` receives every dependency,
so this unit adds no declarations to the frozen ``plugin_api`` namespace.
No ``from __future__ import annotations``: the module is not in
``sys.modules``, so FastAPI must see real request-model classes.
"""
import asyncio
import importlib.util
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import JSONResponse

_DASHBOARD = Path(__file__).resolve().parents[2]
_PROJECT_ID = r"^prj_[0-9a-f]{32}$"
_BOARD_ID = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"


def _load(name: str, relative: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _DASHBOARD / relative)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"The style module is unavailable: {relative}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


boards = _load("film_suite_style_boards", "generation/style_boards.py")
categories = _load("film_suite_style_categories", "generation/style_categories.py")


class StyleValidationRoute(APIRoute):
    """Request-model failures are one finite answer, never a framework echo."""

    def get_route_handler(self):
        original = super().get_route_handler()

        async def finite_handler(request):
            try:
                return await original(request)
            except RequestValidationError:
                return JSONResponse(status_code=422, content={"error": {"code": "invalid_style_request", "message": "Style request is invalid"}})

        return finite_handler


class _StyleRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)


class StyleSearchRequest(_StyleRequest):
    query: str = Field(default="", max_length=200)
    limit: int = Field(default=4000, ge=1, le=4000)


class StyleDetailRequest(_StyleRequest):
    id: str = Field(pattern=_BOARD_ID, min_length=36, max_length=36)


class StyleThumbsRequest(_StyleRequest):
    ids: list[str] = Field(min_length=1, max_length=60)
    # True: answer from the disk cache only, at once; uncached ids are left out.
    cached_only: bool = False


# No project_id: the user's own categories ("My categories", with Favourites).
class StyleCategoriesReadRequest(_StyleRequest):
    project_id: str | None = Field(default=None, pattern=_PROJECT_ID, min_length=36, max_length=36)


class StyleCategoriesWriteRequest(_StyleRequest):
    project_id: str | None = Field(default=None, pattern=_PROJECT_ID, min_length=36, max_length=36)
    expected_revision: int = Field(ge=0)
    categories: list[dict[str, Any]] = Field(max_length=64)


_STATUS = {
    "invalid_style_request": 422,
    "style_unknown": 404, "style_index_unavailable": 503, "style_index_invalid": 503,
    "invalid_style_categories": 422, "style_category_exists": 409, "style_categories_conflict": 409,
    "style_categories_unavailable": 503, "project_not_found": 404, "project_unavailable": 503,
}


def _error(code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=_STATUS.get(code, 503), content={"error": {"code": code, "message": message}})


def create_style_router(
    *,
    hermes_home: Callable[[], Path | None],
    state_root: Callable[..., Path],
    require_project: Callable[[str], None],
    fetch_thumb: Callable[[str], bytes] | None = None,
) -> APIRouter:
    """``require_project`` raises for an id that names no recent project."""
    router = APIRouter(route_class=StyleValidationRoute)
    caches: dict[str, Any] = {}

    def plugin_data(*parts: str) -> Path | None:
        home = hermes_home()
        return None if home is None else Path(state_root(Path(home) / "plugin-data")).joinpath(*parts)

    def thumb_cache() -> Any:
        root = plugin_data("style-thumbs")
        key = str(root)
        if key not in caches:
            caches[key] = boards.ThumbCache(root, fetch_thumb or boards.fetch_thumb)
        return caches[key]

    def category_store() -> Any:
        root = plugin_data("style-categories")
        if root is None:
            raise categories.StyleCategoryError("style_categories_unavailable", "Categories are unavailable")
        return categories.StyleCategoryStore(root)

    def project(project_id: str | None) -> JSONResponse | None:
        if project_id is None:
            return None
        try:
            require_project(project_id)
        except Exception as exc:  # noqa: BLE001 - finite project answer
            code = getattr(exc, "code", None)
            if code == "project_not_found":
                return _error("project_not_found", "The project is not open on this computer")
            return _error("project_unavailable", "The project is unavailable")
        return None

    @router.post("/styles/search")
    async def styles_search(request: StyleSearchRequest) -> Any:
        try:
            return boards.search(boards.load_index(), request.query, limit=request.limit)
        except boards.StyleBoardError as exc:
            return _error(exc.code, exc.message)

    @router.post("/styles/detail")
    async def styles_detail(request: StyleDetailRequest) -> Any:
        try:
            return boards.detail(boards.load_index(), request.id)
        except boards.StyleBoardError as exc:
            return _error(exc.code, exc.message)

    @router.post("/styles/thumbs")
    async def styles_thumbs(request: StyleThumbsRequest) -> Any:
        if any(not isinstance(item, str) or not boards.BOARD_ID_RE.fullmatch(item) for item in request.ids):
            return _error("invalid_style_request", "Style request is invalid")
        try:
            index = boards.load_index()
        except boards.StyleBoardError as exc:
            return _error(exc.code, exc.message)
        cache = thumb_cache()
        return {"thumbs": await asyncio.to_thread(lambda: cache.many(index, list(request.ids), cached_only=request.cached_only))}

    @router.post("/styles/categories/read")
    async def styles_categories_read(request: StyleCategoriesReadRequest) -> Any:
        refused = project(request.project_id)
        if refused is not None:
            return refused
        try:
            document = category_store().read(request.project_id or categories.USER)
        except categories.StyleCategoryError as exc:
            return _error(exc.code, exc.message)
        return {"revision": document["revision"], "categories": document["categories"]}

    @router.post("/styles/categories/write")
    async def styles_categories_write(request: StyleCategoriesWriteRequest) -> Any:
        refused = project(request.project_id)
        if refused is not None:
            return refused
        try:
            index = boards.load_index()
            document = category_store().write(
                request.project_id or categories.USER, request.expected_revision, request.categories,
                known_board=lambda board_id: board_id in index["by_id"],
            )
        except (boards.StyleBoardError, categories.StyleCategoryError) as exc:
            return _error(exc.code, exc.message)
        return {"revision": document["revision"], "categories": document["categories"]}

    return router
