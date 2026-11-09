from __future__ import annotations

# Lane p12-aud item 3 (AUD-010): the song's IN-OUT is a stored mark of the project, not a passing selection.
# Reference: Pixaroma load_audio help.mjs:49-50 (typed IN = the same value as a drag); Audacity's selection toolbar (the
# selection is a value you read and type); charter audio "IN-OUT is a stored mark". One small document per project,
# <project>/state/music-ranges.json {schema, ranges: {<track sha256>: {start, end}}}, atomic replace, newest 500 songs.
# The page saves every range the artist or Hermes sets and restores it on open; no entry = the whole song.

_MUSIC_RANGES_SCHEMA = "studio.music-ranges/1"
_MUSIC_RANGES_MAX = 500
_MUSIC_RANGES_LOCK = _thread.allocate_lock()


class MusicRangeReadRequest(_MusicRequest):
    project_ref: MusicProjectReference
    track_sha256: str = _MUSIC_SHA256_FIELD


class MusicRangeWriteRequest(_MusicRequest):
    project_ref: MusicProjectReference
    track_sha256: str = _MUSIC_SHA256_FIELD
    start: float | None = Field(default=None, ge=0, le=86400)
    end: float | None = Field(default=None, ge=0, le=86400)
    by: Literal["artist", "hermes"] = "artist"


def _music_ranges_path(project_ref: MusicProjectReference) -> Path:
    return Path(_music_project_directory(project_ref)) / "state" / "music-ranges.json"


def _music_ranges_load(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError:
        return {}
    if not isinstance(value, dict) or value.get("schema") != _MUSIC_RANGES_SCHEMA or not isinstance(value.get("ranges"), dict):
        raise ValueError("music ranges document is invalid")
    return value["ranges"]


def _music_range_read_sync(request: MusicRangeReadRequest) -> dict[str, Any]:
    entry = _music_ranges_load(_music_ranges_path(request.project_ref)).get(request.track_sha256)
    if not isinstance(entry, dict):
        return {"ok": True, "track_sha256": request.track_sha256, "range": None}
    return {"ok": True, "track_sha256": request.track_sha256, "range": {"start": entry["start"], "end": entry["end"]}, "by": entry.get("by", "artist")}


def _music_range_write_sync(request: MusicRangeWriteRequest) -> dict[str, Any]:
    clear = request.start is None and request.end is None
    if not clear and (request.start is None or request.end is None or request.end <= request.start):
        raise ValueError("music_range_invalid")
    path = _music_ranges_path(request.project_ref)
    with _MUSIC_RANGES_LOCK:
        ranges = dict(_music_ranges_load(path))
        ranges.pop(request.track_sha256, None)
        if not clear:
            ranges[request.track_sha256] = {"start": round(float(request.start), 6), "end": round(float(request.end), 6), "by": request.by}
        while len(ranges) > _MUSIC_RANGES_MAX:
            ranges.pop(next(iter(ranges)))
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
        try:
            temp.write_text(json.dumps({"schema": _MUSIC_RANGES_SCHEMA, "ranges": ranges}, ensure_ascii=False), encoding="utf-8")
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
    stored = ranges.get(request.track_sha256)
    return {"ok": True, "track_sha256": request.track_sha256, "range": None if stored is None else {"start": stored["start"], "end": stored["end"]}}


@router.post("/music/range/read")
async def music_range_read(request: MusicRangeReadRequest) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(_music_range_read_sync, request)
    except Exception:  # noqa: BLE001 - the route names the reason
        return _music_neutral_error("music_state_unavailable")


@router.post("/music/range")
async def music_range_write(request: MusicRangeWriteRequest) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(_music_range_write_sync, request)
    except ValueError as exc:
        return {"ok": False, "reason_code": "music_range_invalid", "message": "OUT must be after IN."} if str(exc) == "music_range_invalid" else _music_neutral_error("music_state_unavailable")
    except Exception:  # noqa: BLE001 - the route names the reason
        return _music_neutral_error("music_state_unavailable")
