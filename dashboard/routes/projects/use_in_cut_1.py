from __future__ import annotations


# --- Use in Cut (lane w19-g7hermes item 5, CUT-071): the Audio page's "Use in Cut" button as one route ---------
#
# The hand button (desktop src/audio/audio-use-in-cut.js + timeline/view/timeline-song.js songOps) has two cases:
# * an analysed song with a map (track_sha256 + map_id): place_music on the music lane at the start of the cut,
#   the IN-OUT range (seconds) as source_in_frame / duration_frames at the cut's rate, through /projects/timeline/apply;
# * a song without a map (a sound file of the project's folders): /projects/timeline/audio/add with place
#   {role: music, at_frame: 0}, the whole file.
# This route runs exactly those two calls (called, not changed), so the agent and the hand land the same revision.
# expected_revision defaults to the cut's current revision, as the button reads it right before placing.


class UseInCutRange(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)


class UseInCutSong(BaseModel):
    track_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    map_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    seconds: float = Field(gt=0)
    name: str = Field(default="", max_length=80)


class UseInCutRequest(_ProjectRequest):
    project_ref: ProjectReference
    song: UseInCutSong | None = None
    path: str | None = Field(default=None, min_length=1, max_length=4096)
    range: UseInCutRange | None = None
    expected_revision: int | None = Field(default=None, ge=0)
    by: Literal["artist", "hermes"] = "hermes"
    request_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{8,64}$")


def _use_in_cut_error(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "retryable": False, "next": None}}


def _use_in_cut_song_op(song: UseInCutSong, span: UseInCutRange | None, rate: dict[str, int]) -> tuple[dict[str, Any], str]:
    """songOps of timeline-song.js: the range (or the whole song) in frames at the cut rate, placed at frame 0."""
    fps = rate["num"] / rate["den"]
    start, end = (0.0, song.seconds) if span is None else (max(0.0, span.start), min(song.seconds, span.end))
    source_in = round(start * fps)
    duration = max(1, int(end * fps) - source_in)
    op = {"op": "place_music", "track_sha256": song.track_sha256, "map_id": song.map_id,
          "record_start_frame": 0, "source_in_frame": source_in, "duration_frames": duration}
    return op, f"Place {song.name or 'the song'} at the start"[:_TIMELINE_LABEL_MAX]


def _use_in_cut_song_from_path(request: UseInCutRequest) -> UseInCutSong | dict[str, Any]:
    try:
        roots = _request_roots([], request.project_ref, path=request.path, server_list=True)
        opened = _music_open_sync(request.path, roots, None, None, request.project_ref)
    except _FolderNotApproved:
        return _use_in_cut_error("outside_roots", "The file is outside the project's folders.")
    except Exception:  # noqa: BLE001 - not a readable sound file
        return _use_in_cut_error("use_in_cut_invalid", "The sound file could not be read.")
    if not opened.get("analysed") or not opened.get("map_id"):
        return _use_in_cut_error("song_not_analysed", "The song has no analysis yet: analyse it (continuo_music_analyse_start), or send path without range for the whole file.")
    name = Path(request.path).stem[:80]
    return UseInCutSong(track_sha256=opened["track_sha256"], map_id=opened["map_id"], seconds=float(opened["duration"]), name=name)


@project_router.post("/projects/timeline/use-in-cut")
async def projects_timeline_use_in_cut(request: UseInCutRequest) -> dict[str, Any]:
    if (request.song is None) == (request.path is None):
        return _use_in_cut_error("use_in_cut_invalid", "Send song (an analysed song with its map) or path (a sound file), not both.")
    if request.range is not None and request.range.end <= request.range.start:
        return _use_in_cut_error("use_in_cut_invalid", "range end must be after its start.")
    if request.path is not None and request.range is not None:
        # Lane w25-audio2 item 5 (HER-088): a path with a range is the Audio page's Use in Cut on an analysed song: the
        # song and its map are resolved here from the file's stored analysis (by content hash) and the project's
        # interpretation, as the page does, so Hermes needs no id it cannot get.
        song = await asyncio.to_thread(_use_in_cut_song_from_path, request)
        if isinstance(song, dict):
            return song
        request = request.model_copy(update={"song": song, "path": None})
    try:
        directory = _project_reference_directory(request.project_ref)
        project_id = request.project_ref.project_id
        store = _project_store_factory()
        revision, state = await asyncio.to_thread(_timeline_current, store, directory, project_id)
    except (studio_core.ProjectError, ValueError) as exc:
        return _timeline_error(exc)
    expected = revision if request.expected_revision is None else request.expected_revision
    if request.path is not None:
        answer = await projects_timeline_audio_add(TimelineAudioAddRequest(
            project_ref=request.project_ref, paths=[request.path],
            place=TimelineAudioPlace(expected_revision=expected, role="music", at_frame=0, by=request.by)))
        if isinstance(answer, dict) and not answer.get("error") and not answer.get("placed"):
            errors = answer.get("errors") or []
            first = errors[0] if errors and isinstance(errors[0], dict) else {}
            return {**_use_in_cut_error(str(first.get("code") or "not_placed"), str(first.get("message") or "Nothing was placed.")),
                    "errors": errors}
        return {"use": "file", **answer} if isinstance(answer, dict) else answer
    op, label = _use_in_cut_song_op(request.song, request.range, _timeline_rate_of(state, {}))
    answer = await projects_timeline_apply(TimelineApplyRequest(
        project_ref=request.project_ref, expected_revision=expected, ops=[op], label=label, by=request.by,
        request_id=request.request_id))
    return {"use": "song", "placed": op, **answer} if isinstance(answer, dict) and not answer.get("error") else answer
