from __future__ import annotations


# --- Image workbench renderer bridge -------------------------------------
# Same transport as the Music bridge: the pane posts path-free snapshots and
# polls bounded named actions queued by the agent tools, applies them with the
# handlers its own buttons use, and acknowledges. Nothing here starts work.


def _image_bridge_store() -> Any:
    hermes_home = _active_hermes_home()
    if hermes_home is None:
        raise image_bridge.ImageBridgeValidationError("Image bridge storage is unavailable")
    return image_bridge.ImageBridgeStore(state_root(hermes_home / "plugins", "image-bridge"))


_IMAGE_BRIDGE_MESSAGES = {
    "invalid_image_bridge_request": "Image bridge request is invalid.",
    "image_bridge_unavailable": "Image bridge is unavailable.",
}


def _image_bridge_error(exc: Exception | None) -> dict[str, Any]:
    code = "invalid_image_bridge_request" if exc is None or isinstance(exc, image_bridge.ImageBridgeValidationError) else "image_bridge_unavailable"
    return {"ok": False, "reason_code": code, "message": _IMAGE_BRIDGE_MESSAGES[code]}


class ImageBridgeValidationRoute(APIRoute):
    """Strict request models with a finite, neutral error envelope."""

    def get_route_handler(self):
        original_route_handler = super().get_route_handler()

        async def finite_handler(request):
            try:
                return await original_route_handler(request)
            except RequestValidationError:
                return JSONResponse(status_code=422, content=_image_bridge_error(None))

        return finite_handler


image_bridge_router = APIRouter(route_class=ImageBridgeValidationRoute)


class ImageBridgeStateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    window_id: str = Field(pattern=r"^win_[0-9a-f]{32}$", min_length=36, max_length=36)
    state: dict[str, Any]


class ImageBridgeActionsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    window_id: str = Field(pattern=r"^win_[0-9a-f]{32}$", min_length=36, max_length=36)
    since: float = Field(default=0.0, ge=0)


class ImageBridgeAckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    window_id: str = Field(pattern=r"^win_[0-9a-f]{32}$", min_length=36, max_length=36)
    action_id: str = Field(pattern=r"^act_[0-9a-f]{32}$", min_length=36, max_length=36)
    ack: dict[str, Any]


class ImageBridgeViewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    window_id: str = Field(pattern=r"^win_[0-9a-f]{32}$", min_length=36, max_length=36)
    action_id: str = Field(pattern=r"^act_[0-9a-f]{32}$", min_length=36, max_length=36)
    data_url: str = Field(max_length=image_bridge.VIEW_MAX_CHARS)
    width: int
    height: int


@image_bridge_router.post("/image/bridge/state")
async def image_bridge_state(request: ImageBridgeStateRequest) -> dict[str, Any]:
    try:
        record = await asyncio.to_thread(_image_bridge_store().write_snapshot, request.window_id, request.state)
        return {
            "ok": True,
            "schema": image_bridge.STATE_SCHEMA,
            "window_id": record["window_id"],
            "revision": record["revision"],
            "observed_at": record["observed_at"],
        }
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_bridge_error(exc)


@image_bridge_router.post("/image/bridge/actions")
async def image_bridge_actions(request: ImageBridgeActionsRequest) -> dict[str, Any]:
    try:
        actions = await asyncio.to_thread(_image_bridge_store().pending_actions, request.window_id, since=request.since)
        return {"ok": True, "actions": actions}
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_bridge_error(exc)


@image_bridge_router.post("/image/bridge/ack")
async def image_bridge_ack(request: ImageBridgeAckRequest) -> dict[str, Any]:
    try:
        record = await asyncio.to_thread(_image_bridge_store().ack_action, request.window_id, request.action_id, request.ack)
        return {
            "ok": True,
            "schema": image_bridge.ACK_SCHEMA,
            "action_id": record["action_id"],
            "window_id": record["window_id"],
        }
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_bridge_error(exc)


@image_bridge_router.post("/image/bridge/view")
async def image_bridge_view(request: ImageBridgeViewRequest) -> dict[str, Any]:
    """The picture a pending view_edit action asked for; the agent tool returns it as an image."""
    try:
        await asyncio.to_thread(_image_bridge_store().put_view, request.window_id, request.action_id, request.data_url, request.width, request.height)
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_bridge_error(exc)


router.include_router(image_bridge_router)
