from __future__ import annotations


# --- Timeline ducking (lane fs-duck): speech regions for auto_duck -------------------------
#
# continuo_timeline_apply / the artist's "Duck under dialogue" send auto_duck without ``speech``;
# _timeline_enrich (projects/timeline_1.py) asks _timeline_duck_speech to fill it: every audible
# key-role source of the cut as it stands at that op (earlier ops of the batch applied) is resolved
# and run through the local speech detector (dashboard/timeline/speech.py, Silero VAD 6.2 on CPU),
# cached per file content sha256 under <state root>/speech (beside the music analysis cache).
# Answers never carry a path.

def _timeline_speech_cache_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    return state_root(Path(base) / "hermes") / "speech"


def _timeline_duck_speech(op: dict[str, Any], index: int, project_id: str, state: dict[str, Any]) -> dict[str, Any]:
    ops_module, volume = _timeline_module("ops"), _timeline_module("ops_volume")
    speech = _timeline_module("speech")
    resolver = _timeline_module("resolver")
    lookups = _timeline_lookups(project_id)
    lookups.pop("music_file", None)
    timeline = ops_module._timeline(state, op)
    key_role = op.get("key_role", "dialogue")
    if key_role not in ("dialogue", "music", "effects"):
        return op  # the algebra refuses it with its own message
    out: dict[str, list[list[int]]] = {}
    for source in volume.key_sources(timeline, key_role):
        ref = source["ref"]
        if ref in out:
            continue
        fact = resolver.resolve_ref(ref, **lookups)
        if fact["status"] != "online" or not fact.get("path"):
            error = studio_core.ProjectError("source_offline", f"ops[{index}]: {ref} is offline ({fact.get('reason')})")
            error.op_index = index
            raise error
        try:
            found = speech.regions_ms(fact["path"], ffmpeg=FFMPEG_PATH, cache_dir=_timeline_speech_cache_dir())
        except speech.SpeechError as exc:
            error = studio_core.ProjectError("timeline_unavailable", f"ops[{index}]: {exc}")
            error.op_index = index
            raise error from None
        out[ref] = found["regions"]
    return {**op, "speech": out}


# --- Lane p12-cutb (CUT-043, GEM-045): ducking is on by default in the cut --------------------------------------
#
# Reference: Premiere Essential Sound auto-ducking writes volume keyframes on the music under dialogue [PR-DUCK];
# filmcraft writes speech regions as Volume keyframes [FC-A:504-507]. After every applied batch that changes where
# dialogue sound is (dialogue-role sound clips, picture clips whose own sound is on, role / mute changes) or where
# music clips are, every unlocked music-role track whose ducking is on (no record = on with the defaults) is re-keyed
# with its stored parameters (ops_volume.reduck) in the SAME document the edit writes: one revision, one Undo. A batch
# that itself runs auto_duck or clear_volume_keys is the artist's own ducking step and is left as it is. Keys in the
# document make the Cut preview and the export the same mix by construction. Nothing happens on read: an old cut
# changes at its next such edit. Without the speech detector (or when it fails) no key is touched and the answer and
# the next read carry ONE named notice.

_TIMELINE_DUCK_NOTICES = {
    "speech_engine_missing": "Speech detector not installed · ducking is off",
    "speech_engine_mismatch": "Speech detector not installed · ducking is off",
    "speech_failed": "Speech detection failed · ducking is off",
    "source_offline": "Dialogue offline · ducking is off",
}
_TIMELINE_DUCK_NOTICE: dict[str, str] = {}  # project id -> the last edit's notice (the read shows it until a duck lands)
_TIMELINE_DUCK_LOG = __import__("logging").getLogger("film_suite.timeline.duck")


def _timeline_reduck(before: dict[str, Any], after: dict[str, Any], ops: list[dict[str, Any]], project_id: str) -> dict[str, Any] | None:
    """Re-key the music of every timeline whose duck inputs this batch changed (``after`` is changed in place).
    Answers None (nothing to do), {"tracks": [...], "changed": [...]} or {"notice": text}."""
    if any(op.get("op") in ("auto_duck", "clear_volume_keys") for op in ops):
        _TIMELINE_DUCK_NOTICE.pop(project_id, None)
        return None
    _timeline_module("ops")  # ops first: ops_volume imports ops, which imports ops_volume back (a fresh process)
    volume, speech = _timeline_module("ops_volume"), _timeline_module("speech")
    old = {t.get("id"): t for t in (before or {}).get("timelines") or ()}
    lookups = None
    tracks: list[str] = []
    changed: list[str] = []
    for timeline in after.get("timelines") or ():
        on = [t for t in timeline["tracks"] if t["kind"] == "audio" and t["role"] == "music" and not t["locked"] and volume.duck_on(t)]
        if not on:
            continue
        previous = old.get(timeline["id"])
        if previous is not None and volume.duck_inputs(previous) == volume.duck_inputs(timeline):
            continue
        regions: dict[str, list[list[int]]] = {}
        for source in volume.key_sources(timeline, "dialogue"):
            ref = source["ref"]
            if ref in regions:
                continue
            if lookups is None:
                lookups = _timeline_lookups(project_id)
                lookups.pop("music_file", None)
            fact = _timeline_module("resolver").resolve_ref(ref, **lookups)
            if fact["status"] != "online" or not fact.get("path"):
                _TIMELINE_DUCK_NOTICE[project_id] = _TIMELINE_DUCK_NOTICES["source_offline"]
                return {"notice": _TIMELINE_DUCK_NOTICE[project_id]}
            try:
                regions[ref] = speech.regions_ms(fact["path"], ffmpeg=FFMPEG_PATH, cache_dir=_timeline_speech_cache_dir())["regions"]
            except speech.SpeechError as exc:
                _TIMELINE_DUCK_NOTICE[project_id] = _TIMELINE_DUCK_NOTICES.get(exc.code, _TIMELINE_DUCK_NOTICES["speech_failed"])
                return {"notice": _TIMELINE_DUCK_NOTICE[project_id]}
            except Exception as exc:  # noqa: BLE001 - lane p12-cutc: an automatic helper never costs the artist an edit
                _TIMELINE_DUCK_LOG.warning("ducking skipped: speech detection failed for %s", ref, exc_info=exc)
                _TIMELINE_DUCK_NOTICE[project_id] = _TIMELINE_DUCK_NOTICES["speech_failed"]
                return {"notice": _TIMELINE_DUCK_NOTICE[project_id]}
        changed += volume.reduck(timeline, {ref: [tuple(r) for r in found] for ref, found in regions.items()})
        tracks += [t["id"] for t in on]
    if not tracks:
        return None
    _TIMELINE_DUCK_NOTICE.pop(project_id, None)
    return {"tracks": tracks, "changed": changed}


def _timeline_ducking_state(state: dict[str, Any], project_id: str) -> dict[str, Any] | None:
    """The read's ``ducking``: every music-role sound track's switch + the last edit's notice; None without music."""
    _timeline_module("ops")  # ops first (see _timeline_reduck): a read can be the first timeline call of a process
    volume = _timeline_module("ops_volume")
    timelines = (state or {}).get("timelines") or ()
    wanted = (state or {}).get("active")
    timeline = next((t for t in timelines if t.get("id") == wanted), timelines[0] if timelines else None)
    if not isinstance(timeline, dict):
        return None
    tracks = {t["id"]: {"on": volume.duck_on(t)} for t in timeline.get("tracks") or () if t.get("kind") == "audio" and t.get("role") == "music"}
    notice = _TIMELINE_DUCK_NOTICE.get(project_id) or _timeline_duck_standing_notice(timeline, volume)
    if not tracks and notice is None:
        return None
    return {"tracks": tracks, **({"notice": notice} if notice else {})}


# --- Lane p12-cutc: the notice survives a restart while its cause holds ----------------------------------------
#
# _TIMELINE_DUCK_NOTICE lives in memory only; after an app restart a duck-on music track would read Duck on while the
# music does not dip. The read therefore also derives the notice from the state: a duck-on music track has dialogue or
# effects sound under its clips, none of its clips carry automatic keys, and the detector is unavailable (model file
# missing, sha mismatch or runtime not importable; speech.available(), no inference, cached per process).

def _timeline_duck_standing_notice(timeline: dict[str, Any], volume: Any) -> str | None:
    under = [s for role in ("dialogue", "effects") for s in volume.key_sources(timeline, role)]
    if not under:
        return None
    for track in timeline.get("tracks") or ():
        if track.get("kind") != "audio" or track.get("role") != "music" or not volume.duck_on(track):
            continue
        items = track.get("items") or ()
        if any(k.get("auto") for i in items for k in i.get("volume_keys") or ()):
            continue
        spans = [(i["record_start_frame"], i["record_start_frame"] + i["duration_frames"]) for i in items]
        if not any(a < s["record_start"] + s["duration"] and s["record_start"] < b for a, b in spans for s in under):
            continue
        if not _timeline_module("speech").available():
            return _TIMELINE_DUCK_NOTICES["speech_engine_missing"]
        return None
    return None
