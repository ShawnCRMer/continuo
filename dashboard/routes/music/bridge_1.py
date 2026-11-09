from __future__ import annotations


def _music_bridge_store() -> Any:
    hermes_home = _active_hermes_home()
    if hermes_home is None:
        raise music_bridge.MusicBridgeValidationError("Music bridge storage is unavailable")
    return music_bridge.MusicBridgeStore(state_root(hermes_home / "plugins", "music-bridge"))


_MUSIC_NEUTRAL_MESSAGES = {
    "invalid_music_request": "Music request is invalid.",
    "music_evidence_missing": "Music evidence is unavailable. Song not analysed.",
    "invalid_music_evidence": "Music evidence is invalid.",
    "invalid_music_interpretation": "Music interpretation is invalid.",
    "music_candidate_missing": "Selected music candidate is unavailable in the current evidence.",
    "music_conflict": "Music interpretation changed elsewhere.",
    "music_scope_mismatch": "Music interpretation belongs to another context.",
    "music_state_unavailable": "Music state is unavailable.",
}


def _music_neutral_error(reason_code: str) -> dict[str, Any]:
    if reason_code not in _MUSIC_NEUTRAL_MESSAGES:
        reason_code = "music_state_unavailable"
    return {
        "ok": False,
        "reason_code": reason_code,
        "message": _MUSIC_NEUTRAL_MESSAGES[reason_code],
    }


class MusicValidationRoute(APIRoute):
    """Keep music request validation strict while returning a finite envelope."""

    def get_route_handler(self):
        original_route_handler = super().get_route_handler()

        async def finite_handler(request):
            try:
                return await original_route_handler(request)
            except RequestValidationError:
                return JSONResponse(
                    status_code=422,
                    content=_music_neutral_error("invalid_music_request"),
                )

        return finite_handler


music_preview_router = APIRouter(route_class=MusicValidationRoute)


class MusicBridgeStateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    window_id: str = Field(pattern=r"^win_[0-9a-f]{32}$", min_length=36, max_length=36)
    state: dict[str, Any]


class MusicBridgeActionsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    window_id: str = Field(pattern=r"^win_[0-9a-f]{32}$", min_length=36, max_length=36)
    since: float = Field(default=0.0, ge=0)
    context_epoch: str | None = Field(default=None, max_length=160)


class MusicBridgeAckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    window_id: str = Field(pattern=r"^win_[0-9a-f]{32}$", min_length=36, max_length=36)
    action_id: str = Field(pattern=r"^act_[0-9a-f]{32}$", min_length=36, max_length=36)
    ack: dict[str, Any]



def _music_bridge_error(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, music_bridge.MusicBridgeValidationError):
        return _music_neutral_error("invalid_music_request")
    return _music_neutral_error("music_state_unavailable")


@router.post("/music/bridge/state")
async def music_bridge_state(request: MusicBridgeStateRequest) -> dict[str, Any]:
    try:
        record = await asyncio.to_thread(_music_bridge_store().write_snapshot, request.window_id, request.state)
        return {
            "ok": True,
            "schema": music_bridge.STATE_SCHEMA,
            "window_id": record["window_id"],
            "revision": record["revision"],
            "observed_at": record["observed_at"],
        }
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _music_bridge_error(exc)


@router.post("/music/bridge/actions")
async def music_bridge_actions(request: MusicBridgeActionsRequest) -> dict[str, Any]:
    try:
        actions = await asyncio.to_thread(
            _music_bridge_store().pending_actions,
            request.window_id,
            since=request.since,
            context_epoch=request.context_epoch,
        )
        return {"ok": True, "actions": actions}
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _music_bridge_error(exc)


@router.post("/music/bridge/ack")
async def music_bridge_ack(request: MusicBridgeAckRequest) -> dict[str, Any]:
    try:
        record = await asyncio.to_thread(_music_bridge_store().ack_action, request.window_id, request.action_id, request.ack)
        return {
            "ok": True,
            "schema": music_bridge.ACK_SCHEMA,
            "action_id": record["action_id"],
            "window_id": record["window_id"],
        }
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _music_bridge_error(exc)
