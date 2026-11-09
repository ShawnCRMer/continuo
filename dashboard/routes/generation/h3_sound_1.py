from __future__ import annotations


# H3 sound (lane fs-h3aud): the lengths Shoot shows and the agent reads ("9.4 of 15 s").
# Read-only: confines every file to the approved roots, measures it with ffprobe and
# applies the same span / limit rules the submit pre-fit applies (h3_media), so the
# total and the refusal here are exactly what a submit would do.  Nothing is uploaded.
_H3_SOUND_SCALARS = {"prompt": "sound check", "seed": 1, "aspect_ratio": "16:9 (Widescreen)", "megapixels": 0.5,
                     "steps": 12, "sampler_name": "er_sde", "scheduler": "beta"}
_H3_SOUND_KEYS = frozenset({"duration_seconds", "reference_images", "reference_videos", "reference_audios",
                            "reference_video_sound", "sound_track"})


class H3SoundCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_id: str
    args: dict[str, Any] = Field(default_factory=dict)
    approved_roots: list[str] = Field(default_factory=list)
    project_ref: ProjectReference | None = None


_H3_SOUND_WORDS = (
    ("reference_audios need at least one", "Voice and music need a reference picture or clip"),
    ("reference_audios must be a list of at most 3", "At most 3 voice or music files"),
    ("may last at most 15 seconds together", "Voice and music may last at most 15 s together"),
    ("must last 2 to 15 seconds", "Each voice or music file must last 2 to 15 s"),
    ("must be one of", "Sound file not WAV, MP3, FLAC, M4A, OGG or Opus"),
    ("reference_video_sound", "The clip's sound cannot be used"),
)


def _h3_sound_problem(exc: BaseException, total: float | None = None) -> str:
    """One plain line for Shoot and the agent; never a field name. Over the limit it says by how much."""
    text = str(exc) or "the sound cannot be used"
    if total is not None and "15 seconds together" in text and total > _h3m_module.AUDIO_TOTAL_SECONDS:
        return f"Voice and music are {total - _h3m_module.AUDIO_TOTAL_SECONDS:.1f} s too long"
    for needle, words in _H3_SOUND_WORDS:
        if needle in text:
            return words
    if "_" in text:
        return "This sound cannot be used here"
    return text[0].upper() + text[1:]


@router.post("/generation/sound/check")
async def h3_sound_check(request: H3SoundCheckRequest) -> dict[str, Any]:
    """Measure the sound of an H3 Frames/References request: per file its length and the part used."""
    if request.target_id not in _h3c_module.TARGET_IDS:
        return _v4_error("invalid_request")
    try:
        roots = _request_roots(request.approved_roots, request.project_ref)
    except _FolderNotApproved:
        return _folder_refusal()
    extra = set(request.args) - _H3_SOUND_KEYS
    if extra:
        return {"error": "Only the sound fields and duration_seconds are read here", "reason_code": "invalid_request"}
    args = {**_H3_SOUND_SCALARS, "duration_seconds": 5, **request.args}
    answer: dict[str, Any] = {"limit_seconds": _h3m_module.AUDIO_TOTAL_SECONDS, "min_seconds": _h3m_module.AUDIO_MIN_SECONDS,
                              "max_seconds": _h3m_module.AUDIO_MAX_SECONDS, "items": [], "total_seconds": 0.0, "problem": None}
    try:
        plan = _h3c_module.validate_args(request.target_id, args)
    except _h3c_module.H3ConditioningError as exc:
        # Lane w30-fix3 item 4 (VID-165): a length outside the target's range is a Duration problem, named with its range
        # next to the field (problem_field), never the sound sentence; the sound is then not judged.
        if "duration_seconds" in str(exc):
            return {**answer, "problem": "Duration must be a whole number from 4 to 15 s", "problem_field": "duration_seconds",
                    "duration_range": {"min": 4, "max": 15}}
        # The spans with a typed end are known without reading a file (as validate_args counts them).
        audios = args.get("reference_audios") if isinstance(args.get("reference_audios"), list) else []
        known = sum(float(a["end"]) - float(a.get("start") or 0) for a in audios
                    if isinstance(a, dict) and isinstance(a.get("end"), (int, float)) and isinstance(a.get("start") or 0, (int, float)))
        return {**answer, "problem": _h3_sound_problem(exc, round(known, 3))}
    answer["shot_seconds"] = round(_h3c_module.shot_seconds(plan), 3)
    labels = {item["key"]: item["label"] for item in _h3c_module.audio_labels(plan)}
    wanted = [item for item in _h3c_module.media_items(plan) if item["kind"] == "audio" or item.get("sound")]
    try:
        resolved = _h3m_module.resolve_media(
            wanted, roots=tuple(roots), boundary=_v4_path_boundary(),
            local_path=_v4_local_path_value, path_identity=_request_identity_module.path_identity,
            staged_root=prepared_cache_dir() / "opening-frames")
    except (_h3m_module.H3MediaError, ValueError, OSError) as exc:
        return {**answer, "problem": _h3_sound_problem(exc) if isinstance(exc, _h3m_module.H3MediaError) else "A sound file is outside your folders or cannot be read"}
    spans = []
    for item in resolved:
        entry: dict[str, Any] = {"key": item["key"]}
        if item["key"] in labels:
            entry["label"] = labels[item["key"]]
        try:
            if item["kind"] == "audio":
                seconds = await asyncio.to_thread(_h3m_module.audio_seconds, item["path"], FFMPEG_PATH)
                start, end = _h3m_module.audio_span(item, seconds)
                entry.update(seconds=round(seconds, 3), used_seconds=round(end - start, 3), start=start, role=item.get("audio_role") or item["role"])
                spans.append((item, end - start))
            else:
                streams = await asyncio.to_thread(_h3m_module.audio_streams, item["path"], FFMPEG_PATH)
                entry.update(role="clip_sound", has_sound=bool(streams))
                if not streams:
                    answer["problem"] = answer["problem"] or "A reference clip has no sound to use"
        except _h3m_module.H3MediaError as exc:
            entry["problem"] = _h3_sound_problem(exc)
            answer["problem"] = answer["problem"] or entry["problem"]
        answer["items"].append(entry)
    answer["total_seconds"] = round(sum(seconds for item, seconds in spans if item.get("fit") == "reference"), 3)
    try:
        _h3m_module.check_audio_lengths(spans)
    except _h3m_module.H3MediaError as exc:
        answer["problem"] = answer["problem"] or _h3_sound_problem(exc, answer["total_seconds"])
    return answer
