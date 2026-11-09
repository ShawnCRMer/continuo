"""Volume automation on sound clips and dialogue ducking by role (lane fs-duck).

Twin of ``desktop/film-suite/src/timeline/ops/timeline-volume.js``; the shared fixtures
``tests/backend/fixtures/timeline/volume.json`` run against both.

A sound clip may carry ``volume_keys``: ``[{source_frame, gain_db, auto}]``, strictly rising in
``source_frame`` (the clip's own source frames, like FCPXML / FCP7 keyframes in clip media time), so
move / trim / roll / slip / split keep every key on the same sound without a change to those verbs.
``gain_db`` (0.1 dB steps, -60..+24) is added to the clip + track gain; between keys the level is
linear in dB, before the first and after the last key it holds. A clip without keys has no field.

``auto_duck`` lowers every sound of ``target_role`` (default music) under every audible sound of
``key_role`` (default dialogue: the sound tracks of that role + picture clips whose own sound is on)
wherever that sound holds speech. Speech comes in ``speech: {source ref: [[start_ms, end_ms], ...]}``
(source time of the file); the route fills it from the speech detector when the caller leaves it
out. Each speech region becomes a duck: down from ``lookahead_ms`` before the first word over
``attack_ms``, held ``hold_ms`` after the last, up over ``release_ms``; ducks closer than that merge.
Its keys are ``auto: true``: a re-run replaces only those; a key the artist set or moved by hand is
``auto: false`` and stays. The parameters are kept on the target track (``duck``) for the next run.

Lane p12-cutb (CUT-043, GEM-045): ducking is on by default in the cut. A music track without a ``duck`` record, or with
``duck.on`` absent, counts as on with ``DUCK_DEFAULTS``; the route re-runs the duck after every edit that moves dialogue
or music (``reduck``, routes/projects/timeline_duck_1.py) in the same revision. A track's auto-only clear is Off: its
auto keys go, the record stays with ``on: false``, and later edits leave it alone until ``auto_duck`` turns it on again.
"""
from __future__ import annotations

from typing import Any

from . import ops as _ops
from .schema import AUDIO_ROLES, GAIN_MAX_DB, GAIN_MIN_DB, MAX_VOLUME_KEYS, fail, gain_tenths, gain_value

DUCK_DEFAULTS = {"depth_db": 12, "attack_ms": 250, "release_ms": 600, "lookahead_ms": 250, "hold_ms": 500}
_DUCK_LIMITS = {"attack_ms": 5000, "release_ms": 10000, "lookahead_ms": 5000, "hold_ms": 10000}
MAX_SPEECH_REGIONS = 20000
MAX_SPEECH_MS = 24 * 3600 * 1000


def _floor_div(n: int, d: int) -> int:
    return n // d


def _round_div(n: int, d: int) -> int:
    """n / d rounded half up (d > 0), exact on integers."""
    return (2 * n + d) // (2 * d)


def _sound(timeline: dict[str, Any], clip_id: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    track, _index, item = _ops._clip(timeline, clip_id)
    if track["kind"] != "audio":
        _ops._bad(f"{clip_id} is a picture clip; volume keys are for sound clips")
    return track, item


def _key_gain(value: Any, name: str = "gain_db") -> int:
    tenths = gain_tenths(value)
    if tenths is None or not GAIN_MIN_DB * 10 <= tenths <= GAIN_MAX_DB * 10:
        _ops._bad(f"{name} must be a number in 0.1 dB steps in {GAIN_MIN_DB}..{GAIN_MAX_DB}")
    return tenths


def _store(item: dict[str, Any], keys: list[dict[str, Any]]) -> None:
    keys.sort(key=lambda k: k["source_frame"])
    if len(keys) > MAX_VOLUME_KEYS:
        fail("too_many_keys", f"{item['id']} would hold more than {MAX_VOLUME_KEYS} volume keys")
    if keys:
        item["volume_keys"] = keys
    else:
        item.pop("volume_keys", None)


def op_set_volume_key(state, args, ctx):
    """Add or replace a key at ``source_frame`` (a hand key); ``from_frame`` moves the key there."""
    _ops._args(args, ("clip", "source_frame", "gain_db"), ("from_frame",))
    track, item = _sound(_ops._timeline(state, args), args["clip"])
    _ops._unlocked(track)
    frame = _ops._int(args["source_frame"], "source_frame", 0, item["media_info"]["frames"])
    tenths = _key_gain(args["gain_db"])
    keys = [dict(k) for k in item.get("volume_keys") or ()]
    if "from_frame" in args:
        origin = _ops._int(args["from_frame"], "from_frame", 0)
        if not any(k["source_frame"] == origin for k in keys):
            fail("not_found", f"{item['id']} has no volume key at source frame {origin}")
        keys = [k for k in keys if k["source_frame"] != origin]
    keys = [k for k in keys if k["source_frame"] != frame]
    keys.append({"source_frame": frame, "gain_db": gain_value(tenths), "auto": False})
    _store(item, keys)
    return {"op": "set_volume_key", "keys": len(keys)}


def op_remove_volume_key(state, args, ctx):
    _ops._args(args, ("clip", "source_frame"))
    track, item = _sound(_ops._timeline(state, args), args["clip"])
    _ops._unlocked(track)
    frame = _ops._int(args["source_frame"], "source_frame", 0)
    keys = [dict(k) for k in item.get("volume_keys") or ()]
    if not any(k["source_frame"] == frame for k in keys):
        fail("not_found", f"{item['id']} has no volume key at source frame {frame}")
    keys = [k for k in keys if k["source_frame"] != frame]
    _store(item, keys)
    return {"op": "remove_volume_key", "keys": len(keys)}


def op_clear_volume_keys(state, args, ctx):
    """Every key (or only the auto ones) of one clip or of every clip on one track; a track's
    auto-only clear also forgets its ``duck`` (ducking off)."""
    _ops._args(args, (), ("clip", "track", "auto_only"))
    timeline = _ops._timeline(state, args)
    if ("clip" in args) == ("track" in args):
        _ops._bad("clear_volume_keys takes clip or track")
    auto_only = args.get("auto_only", False)
    if type(auto_only) is not bool:
        _ops._bad("auto_only must be true or false")
    if "clip" in args:
        track, item = _sound(timeline, args["clip"])
        items = [item]
    else:
        track = _ops._track(timeline, args["track"])
        if track["kind"] != "audio":
            _ops._bad(f"{args['track']} is not a sound track")
        items = list(track["items"])
    _ops._unlocked(track)
    if "track" in args and auto_only:  # Off: the parameters stay, ducking is off until auto_duck (p12-cutb)
        track["duck"] = {**{k: v for k, v in (track.get("duck") or {"key_role": "dialogue", **DUCK_DEFAULTS}).items() if k != "on"}, "on": False}
    removed = 0
    for item in items:
        keys = [dict(k) for k in item.get("volume_keys") or ()]
        kept = [k for k in keys if auto_only and not k["auto"]]
        removed += len(keys) - len(kept)
        _store(item, kept)
    return {"op": "clear_volume_keys", "removed": removed}


# --- auto_duck ---------------------------------------------------------------------------

def _ms(value: Any, name: str, maximum: int) -> int:
    return _ops._int(value, name, 0, maximum)


def _speech(value: Any) -> dict[str, list[tuple[int, int]]]:
    if type(value) is not dict:
        _ops._bad("speech must be an object {source ref: [[start_ms, end_ms], ...]}")
    out: dict[str, list[tuple[int, int]]] = {}
    count = 0
    for ref, regions in value.items():
        if type(regions) is not list:
            _ops._bad(f"speech[{ref}] must be a list of [start_ms, end_ms]")
        last = -1
        parsed = []
        for region in regions:
            if (type(region) is not list or len(region) != 2 or any(type(v) is not int for v in region)
                    or not last <= region[0] < region[1] <= MAX_SPEECH_MS):
                _ops._bad(f"speech[{ref}] regions must be rising [start_ms, end_ms] pairs of integers")
            last = region[1]
            parsed.append((region[0], region[1]))
        count += len(parsed)
        out[ref] = parsed
    if count > MAX_SPEECH_REGIONS:
        _ops._bad(f"speech holds more than {MAX_SPEECH_REGIONS} regions")
    return out


def key_sources(timeline: dict[str, Any], key_role: str) -> list[dict[str, Any]]:
    """Every audible sound of ``key_role``: {ref, record_start, duration, source_in, source_rate}."""
    out = []
    for track in _ops._audio_tracks(timeline):
        if track["role"] != key_role or track["mute"]:
            continue
        for item in track["items"]:
            if item["enabled"]:
                out.append({"ref": item["source"]["ref"], "record_start": item["record_start_frame"], "duration": item["duration_frames"],
                            "source_in": item["source_in_frame"], "source_rate": item["source_rate"]})
    if key_role == "dialogue":
        for track in _ops._video_tracks(timeline):
            if not track["enabled"]:
                continue
            for index, item in enumerate(track["items"]):
                if item["type"] == "clip" and item["enabled"] and item["audio"] == "on" and _ops._moving(item) and item["media_info"]["has_audio"]:
                    out.append({"ref": item["source"]["ref"], "record_start": _ops._starts(track)[index], "duration": item["duration_frames"],
                                "source_in": item["source_in_frame"], "source_rate": item["source_rate"]})
    return out


def _record_frame(source: dict[str, Any], ms: int, rate: dict[str, int], ceil: bool) -> int:
    """Source time ``ms`` of a key sound -> record frame (floor for a start, ceil for an end)."""
    sn, sd = source["source_rate"]["num"], source["source_rate"]["den"]
    numerator = (ms * sn - 1000 * source["source_in"] * sd) * rate["num"]
    denominator = 1000 * sn * rate["den"]
    step = -_floor_div(-numerator, denominator) if ceil else _floor_div(numerator, denominator)
    return source["record_start"] + step


def speech_frames(timeline: dict[str, Any], key_role: str, speech: dict[str, list[tuple[int, int]]]) -> list[tuple[int, int]]:
    """Record-frame regions [start, end) where a key sound holds speech, merged, sorted."""
    rate = timeline["rate"]
    regions = []
    for source in key_sources(timeline, key_role):
        if source["ref"] not in speech:
            fail("no_speech", f"speech regions of {source['ref']} are missing")
        lo, hi = source["record_start"], source["record_start"] + source["duration"]
        for start_ms, end_ms in speech[source["ref"]]:
            start = max(lo, _record_frame(source, start_ms, rate, False))
            end = min(hi, _record_frame(source, end_ms, rate, True))
            if start < end:
                regions.append((start, end))
    regions.sort()
    merged: list[list[int]] = []
    for start, end in regions:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def _frames_of_ms(ms: int, rate: dict[str, int]) -> int:
    return _round_div(ms * rate["num"], 1000 * rate["den"])


def duck_points(regions: list[tuple[int, int]], params: dict[str, int], rate: dict[str, int]) -> list[tuple[int, int]]:
    """The duck envelope in record frames as (frame, tenths of dB) points, 0 outside the ducks."""
    depth = params["depth_tenths"]
    look, attack = _frames_of_ms(params["lookahead_ms"], rate), _frames_of_ms(params["attack_ms"], rate)
    hold, release = _frames_of_ms(params["hold_ms"], rate), _frames_of_ms(params["release_ms"], rate)
    ducks: list[list[int]] = []
    for start, end in regions:
        down0 = start - look
        down1 = down0 + attack
        up0 = max(end + hold, down1)
        if ducks and down0 <= ducks[-1][3]:
            ducks[-1][2] = max(ducks[-1][2], up0)
            ducks[-1][3] = ducks[-1][2] + release
            continue
        ducks.append([down0, down1, up0, up0 + release])
    points: list[tuple[int, int]] = []
    for down0, down1, up0, up1 in ducks:
        points += [(down0, 0), (down1, -depth), (up0, -depth), (up1, 0)]
    out: list[tuple[int, int]] = []
    for frame, value in points:  # a zero-length ramp keeps its later point
        if out and out[-1][0] == frame:
            out[-1] = (frame, value)
        else:
            out.append((frame, value))
    return out


def _value_at(points: list[tuple[int, int]], frame: int) -> int:
    """Tenths of dB of the envelope at ``frame`` (linear between points, rounded half up)."""
    if not points or frame <= points[0][0]:
        return points[0][1] if points else 0
    for (f0, v0), (f1, v1) in zip(points, points[1:]):
        if f0 <= frame <= f1:
            return v0 + _round_div((v1 - v0) * (frame - f0), f1 - f0) if f1 > f0 else v1
    return points[-1][1]


def _source_frame(item: dict[str, Any], record: int, rate: dict[str, int]) -> int:
    sn, sd = item["source_rate"]["num"], item["source_rate"]["den"]
    return item["source_in_frame"] + _round_div((record - item["record_start_frame"]) * sn * rate["den"], sd * rate["num"])


def clip_keys(item: dict[str, Any], points: list[tuple[int, int]], rate: dict[str, int]) -> list[dict[str, Any]]:
    """The auto keys of one target clip: the envelope's points inside the clip, plus its value at a
    clip edge that cuts into a duck (the level holds before the first and after the last key)."""
    lo, hi = item["record_start_frame"], item["record_start_frame"] + item["duration_frames"]
    chosen = [(f, v) for f, v in points if lo <= f <= hi]
    for edge in (lo, hi):
        if not any(f == edge for f, _v in chosen) and _value_at(points, edge):
            chosen.append((edge, _value_at(points, edge)))
    chosen.sort()
    keys: list[dict[str, Any]] = []
    for frame, value in chosen:
        source_frame = _source_frame(item, frame, rate)
        if not 0 <= source_frame <= item["media_info"]["frames"]:
            continue
        key = {"source_frame": source_frame, "gain_db": gain_value(value), "auto": True}
        if keys and keys[-1]["source_frame"] == source_frame:
            keys[-1] = key
        else:
            keys.append(key)
    return keys


def _role(value: Any, name: str) -> str:
    if type(value) is not str or value not in AUDIO_ROLES:
        _ops._bad(f"{name} must be one of " + ", ".join(AUDIO_ROLES))
    return value


def duck_params(args: dict[str, Any], track: dict[str, Any] | None = None) -> dict[str, Any]:
    """The run's parameters: the op's, else the target track's last run, else the defaults."""
    base = dict(DUCK_DEFAULTS)
    if track is not None and track.get("duck"):
        base.update({k: v for k, v in track["duck"].items() if k in DUCK_DEFAULTS})
    depth = gain_tenths(args.get("depth_db", base["depth_db"]))
    if depth is None or not 1 <= depth <= 600:
        _ops._bad("depth_db must be in 0.1..60")
    out = {"depth_db": gain_value(depth), "depth_tenths": depth}
    for name, maximum in _DUCK_LIMITS.items():
        out[name] = _ms(args.get(name, base[name]), name, maximum)
    return out


def duck_on(track: dict[str, Any]) -> bool:
    """A sound track's ducking switch: no record or no ``on`` = on (p12-cutb, on by default)."""
    return (track.get("duck") or {}).get("on", True) is not False


def duck_inputs(timeline: dict[str, Any], key_role: str = "dialogue", target_role: str = "music") -> Any:
    """What the duck depends on: where the key-role sound is and where the target-role clips are."""
    targets = [(t["id"], t["locked"], duck_on(t), [(i["id"], i["record_start_frame"], i["duration_frames"], i["source_in_frame"])
               for i in t["items"]]) for t in _ops._audio_tracks(timeline) if t["role"] == target_role]
    return (sorted((s["ref"], s["record_start"], s["duration"], s["source_in"]) for s in key_sources(timeline, key_role)), targets)


def reduck(timeline: dict[str, Any], speech: dict[str, list[tuple[int, int]]], key_role: str = "dialogue", target_role: str = "music") -> list[str]:
    """Re-key every unlocked ``target_role`` track whose ducking is on with its stored parameters (no record = the
    defaults); hand keys stay. Answers the ids of the clips whose keys changed. The record itself is not written."""
    regions = speech_frames(timeline, key_role, speech)
    changed: list[str] = []
    for track in _ops._audio_tracks(timeline):
        if track["role"] != target_role or track["locked"] or not duck_on(track):
            continue
        record = track.get("duck") or {}
        params = duck_params({}, track)
        if record.get("key_role", key_role) != key_role:
            continue
        points = duck_points(regions, params, timeline["rate"])
        for item in track["items"]:
            manual = [dict(k) for k in item.get("volume_keys") or () if not k["auto"]]
            taken = {k["source_frame"] for k in manual}
            fresh = [k for k in clip_keys(item, points, timeline["rate"]) if k["source_frame"] not in taken]
            before = item.get("volume_keys")
            _store(item, manual + fresh)
            if item.get("volume_keys") != before:
                changed.append(item["id"])
    return changed


def op_auto_duck(state, args, ctx):
    _ops._args(args, (), ("target_role", "key_role", "depth_db", "attack_ms", "release_ms", "lookahead_ms", "hold_ms", "speech"))
    timeline = _ops._timeline(state, args)
    target_role = _role(args.get("target_role", "music"), "target_role")
    key_role = _role(args.get("key_role", "dialogue"), "key_role")
    if target_role == key_role:
        _ops._bad("target_role and key_role must differ")
    targets = [t for t in _ops._audio_tracks(timeline) if t["role"] == target_role]
    if not targets:
        fail("no_target", f"the cut has no {target_role} sound track")
    speech = _speech(args.get("speech", {}))
    regions = speech_frames(timeline, key_role, speech)
    changed: list[str] = []
    skipped: list[str] = []
    auto_keys = 0
    for track in targets:
        if track["locked"]:
            skipped.append(track["id"])
            continue
        params = duck_params(args, track)
        points = duck_points(regions, params, timeline["rate"])
        for item in track["items"]:
            manual = [dict(k) for k in item.get("volume_keys") or () if not k["auto"]]
            taken = {k["source_frame"] for k in manual}
            fresh = [k for k in clip_keys(item, points, timeline["rate"]) if k["source_frame"] not in taken]
            before = item.get("volume_keys")
            _store(item, manual + fresh)
            auto_keys += len(fresh)
            if item.get("volume_keys") != before:
                changed.append(item["id"])
        track["duck"] = {"key_role": key_role, **{k: params[k] for k in DUCK_DEFAULTS}}
    if len(skipped) == len(targets):
        fail("locked", "every " + target_role + " sound track is locked")
    used = {k: params[k] for k in DUCK_DEFAULTS}
    return {"op": "auto_duck", "regions": len(regions), "changed": changed, "auto_keys": auto_keys, "skipped": skipped, "params": used}


VOLUME_OPS = {
    "set_volume_key": op_set_volume_key, "remove_volume_key": op_remove_volume_key,
    "clear_volume_keys": op_clear_volume_keys, "auto_duck": op_auto_duck,
}
