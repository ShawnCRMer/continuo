from __future__ import annotations


# --- Timeline sound files: the artist's audio as cut sources (file:<sha256>), Add to cut -----
#
# /projects/timeline/audio/add registers sound files (wav, mp3, flac, opus, ogg, m4a, aac) of
# the project's approved folders exactly like footage (projects/timeline_footage_1.py): the
# path is confined to the manifest's folders, the content hashed once into the same private
# index, the file probed (sample rate, channels, decoded samples, codec) and measured once
# (ffmpeg ebur128: integrated LUFS, true peak, LRA), cached with the facts by content hash.
# The answer carries ids and facts, never a path. With ``place`` it is the Timeline's "Add to
# cut": the files land one after another on the sound track of the chosen role (A1-A3; the
# track is made when the cut has none) at ``at_frame``, through the same algebra, revision
# history and timeline.changed event as /projects/timeline/apply (one apply = one undo step).

_TIMELINE_AUDIO_MAX = 50


class TimelineAudioPlace(BaseModel):
    expected_revision: int = Field(ge=0)
    role: Literal["dialogue", "music", "effects"] = "dialogue"
    track: str | None = Field(default=None, min_length=1, max_length=64)
    at_frame: int = Field(default=0, ge=0)
    timeline: str | None = Field(default=None, min_length=1, max_length=64)
    by: Literal["artist", "hermes"] = "hermes"


class TimelineAudioAddRequest(_ProjectRequest):
    project_ref: ProjectReference
    paths: list[str] = Field(min_length=1, max_length=_TIMELINE_AUDIO_MAX)
    place: TimelineAudioPlace | None = None


def _timeline_audio_index(project_id: str) -> Any:
    footage = _timeline_module("footage")
    index = _timeline_footage_index(project_id)
    index._probe = lambda file: footage.probe_media(file, ffprobe=FFPROBE_PATH)
    return index


def _timeline_audio_public(added: dict[str, Any]) -> dict[str, Any]:
    facts = added["facts"]
    samples, rate = facts.get("samples"), facts.get("sample_rate")
    return {"ref": added["ref"], "name": added["name"], "kind": "audio", "samples": samples, "sample_rate": rate,
            "channels": facts.get("channels"), "duration_seconds": round(samples / rate, 3) if samples and rate else None,
            "codec": facts.get("codec") or "", "loudness": dict(facts.get("loudness") or {})}


def _timeline_audio_register(directory: str, project_id: str, paths: list[str]) -> dict[str, Any]:
    footage = _timeline_module("footage")
    roots = _timeline_footage_roots(project_id, directory)
    index = _timeline_audio_index(project_id)
    added: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for position, raw in enumerate(paths):
        try:
            if not roots:
                raise footage.FootageError("outside_roots", "No project folder.")
            try:
                path = resolve_approved_path(raw, roots)
            except (OSError, ValueError):
                raise footage.FootageError("outside_roots", "The file is outside the project's folders.") from None
            if not path.is_file():
                raise footage.FootageError("file_missing", "The file does not exist.")
            if path.suffix.lower() not in footage.AUDIO_SUFFIXES:
                raise footage.FootageError("unsupported", "Only sound files (WAV, MP3, FLAC, Opus, Ogg, M4A, AAC) go onto a sound track.")
            entry = index.add(path)
            if entry["facts"].get("kind") != "audio":
                raise footage.FootageError("unsupported", "The file holds no sound.")
            added.append(_timeline_audio_public(entry))
        except footage.FootageError as exc:
            errors.append({"index": position, "code": exc.code, "message": str(exc)})
        except OSError:
            errors.append({"index": position, "code": "unreadable", "message": "The file could not be read."})
    return {"audio": added, "errors": errors}


def _timeline_audio_place(directory: str, project_id: str, audio: list[dict[str, Any]], place: TimelineAudioPlace) -> dict[str, Any]:
    """One apply of place_audio ops, the files back to back from ``at_frame`` on one sound track."""
    store = _project_store_factory()
    revision, before = _timeline_current(store, directory, project_id)
    if place.expected_revision != revision:
        raise studio_core.ProjectError("timeline_conflict", f"the timeline is at revision {revision}")
    ops_module = _timeline_module("ops")
    timeline = ops_module._timeline(before, {"op": "place_audio", **({"timeline": place.timeline} if place.timeline else {})})
    rate = timeline["rate"]
    footage = _timeline_module("footage")
    target: dict[str, Any] = {"track": place.track} if place.track else {"role": place.role}
    ops: list[dict[str, Any]] = []
    at = place.at_frame
    for entry in audio:
        facts = {"samples": entry["samples"], "sample_rate": entry["sample_rate"]}
        frames = footage.audio_frames(facts, rate)
        if frames < 1:
            continue
        op = {"op": "place_audio", "at_frame": at, **target, "clip": {
            "source": {"kind": "media", "ref": entry["ref"]}, "source_rate": dict(rate), "label": entry["name"][:200],
            "media_info": {"frames": frames, "sample_rate": entry["sample_rate"], "channels": entry["channels"]}}}
        if place.timeline:
            op["timeline"] = place.timeline
        ops.append(op)
        at += frames
    if not ops:
        raise studio_core.ProjectError("invalid_op", "no sound file to place")
    after, results = ops_module.apply_ops(before, ops)
    ducking = _timeline_reduck(before, after, ops, project_id)  # lane p12-cutb: the duck lands in this same revision
    track = results[0]["track"]
    label = f"Add {len(ops)} sound{'s' if len(ops) != 1 else ''} to {track}"
    by = _caller_by(place.by)  # lane w34-sec (finding 18): a Hermes call is Hermes' revision
    written = store.write_timeline(directory, project_id=project_id, expected_revision=revision, state=after, label=label, by=by)
    film_events.timeline_changed(project_id, written["revision"], by)
    summary = _timeline_module("document").summarize(before, after)
    return {"track": track, "revision": written["revision"], "label": label, "created": summary["created"],
            "placed_refs": summary["placed_refs"], **({"ducking": ducking} if ducking is not None else {})}


def _timeline_audio_add(directory: str, project_id: str, paths: list[str], place: TimelineAudioPlace | None) -> dict[str, Any]:
    result = _timeline_audio_register(directory, project_id, paths)
    if place is not None and result["audio"]:
        result["placed"] = _timeline_audio_place(directory, project_id, result["audio"], place)
    return result


@project_router.post("/projects/timeline/audio/add")
async def projects_timeline_audio_add(request: TimelineAudioAddRequest) -> dict[str, Any]:
    try:
        directory = _project_reference_directory(request.project_ref)
        return await asyncio.to_thread(_timeline_audio_add, directory, request.project_ref.project_id, list(request.paths), request.place)
    except (studio_core.ProjectError, ValueError) as exc:
        return _timeline_error(exc)
