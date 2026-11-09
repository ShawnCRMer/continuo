"""Sound-track verbs (A1-A3): place, gain, fades, track settings, a take's own sound linked.

Twin of ``desktop/film-suite/src/timeline/ops/timeline-audio.js``; the shared fixtures
``tests/backend/fixtures/timeline/audio.json`` run against both. Sound clips live on
``kind: "audio"`` tracks at their own record frames (like V2), so move / trim / roll / slip /
split / delete are the picture verbs (ops.py) unchanged; this module adds what only sound has.

A linked sound (``link: {clip, offset_frames}``) is a take's own sound on a sound track: the
post-pass (``sync_sound``) keeps it on its picture clip (start, length, source in, source
rate, source), a split of the picture splits it, deleting the picture removes it, and its own
edges are not edited (``linked``) until ``unlink_sound``. Gain and fades stay its own.
"""
from __future__ import annotations

import copy
from typing import Any

from . import ops as _ops
from .schema import AUDIO_ROLES, GAIN_MAX_DB, GAIN_MIN_DB, MAX_AUDIO_TRACKS, fail, gain_tenths, gain_value, new_audio_track

_SOUND_FIELDS = ("id", "source", "source_in_frame", "duration_frames", "source_rate", "media_info", "gain_db",
                 "fade_in_frames", "fade_out_frames", "enabled", "label", "note")


def _gain(value: Any, name: str = "gain_db") -> int | float:
    tenths = gain_tenths(value)
    if tenths is None or not GAIN_MIN_DB * 10 <= tenths <= GAIN_MAX_DB * 10:
        _ops._bad(f"{name} must be a number in 0.1 dB steps in {GAIN_MIN_DB}..{GAIN_MAX_DB}")
    return gain_value(tenths)


def _sound(timeline: dict[str, Any], item_id: Any) -> tuple[dict[str, Any], int, dict[str, Any]]:
    track, index, item = _ops._clip(timeline, item_id)
    if track["kind"] != "audio":
        _ops._bad(f"{item_id} is a picture clip; this verb is for sound clips")
    return track, index, item


def _audio_track(timeline: dict[str, Any], track_id: Any) -> dict[str, Any]:
    track = _ops._track(timeline, track_id)
    if track["kind"] != "audio":
        _ops._bad(f"{track_id} is not a sound track")
    return track


def _free_track_id(timeline: dict[str, Any]) -> str:
    used = {track["id"] for track in timeline["tracks"]}
    return next(f"A{n}" for n in range(1, MAX_AUDIO_TRACKS + 2) if f"A{n}" not in used)


def _add_track(timeline: dict[str, Any], role: str, ident: str | None = None) -> dict[str, Any]:
    if len(_ops._audio_tracks(timeline)) >= MAX_AUDIO_TRACKS:
        fail("track_full", f"a cut holds at most {MAX_AUDIO_TRACKS} sound tracks (A1-A3)")
    track = new_audio_track(ident or _free_track_id(timeline), role)
    timeline["tracks"].append(track)
    return track


def _role(value: Any) -> str:
    if value not in AUDIO_ROLES or type(value) is not str:
        _ops._bad("role must be one of " + ", ".join(AUDIO_ROLES))
    return value


def _overlaps(track: dict[str, Any], start: int, end: int, skip: tuple[str, ...] = ()) -> str | None:
    for item in track["items"]:
        if item["id"] not in skip and item["record_start_frame"] < end and start < item["record_start_frame"] + item["duration_frames"]:
            return item["id"]
    return None


def op_add_audio_track(state, args, ctx):
    _ops._args(args, ("role",), ("id",))
    timeline = _ops._timeline(state, args)
    ident = args.get("id")
    if ident is not None and ident in _ops._all_ids(state):
        _ops._bad(f"track id {ident} is already used")
    track = _add_track(timeline, _role(args["role"]), ident)
    return {"op": "add_audio_track", "created": [track["id"]]}


def op_remove_audio_track(state, args, ctx):
    _ops._args(args, ("track",))
    timeline = _ops._timeline(state, args)
    track = _ops._unlocked(_audio_track(timeline, args["track"]))
    if track["items"]:
        fail("not_empty", f"{track['id']} still holds {len(track['items'])} sound clips")
    timeline["tracks"] = [t for t in timeline["tracks"] if t is not track]
    return {"op": "remove_audio_track", "removed": [track["id"]]}


def op_set_track(state, args, ctx):
    """Track settings: role, gain_db, mute, solo, locked (a locked track still mixes and can be unlocked)."""
    _ops._args(args, ("track",), ("role", "gain_db", "mute", "solo", "locked"))
    timeline = _ops._timeline(state, args)
    track = _audio_track(timeline, args["track"])
    if not set(args) & {"role", "gain_db", "mute", "solo", "locked"}:
        _ops._bad("set_track takes role, gain_db, mute, solo and/or locked")
    if "role" in args:
        track["role"] = _role(args["role"])
    if "gain_db" in args:
        track["gain_db"] = _gain(args["gain_db"])
    for key in ("mute", "solo", "locked"):
        if key in args:
            if type(args[key]) is not bool:
                _ops._bad(f"{key} must be true or false")
            track[key] = args[key]
    return {"op": "set_track"}


def _target_track(timeline: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """``track`` by id, else the first sound track of ``role`` (made when the cut has room)."""
    if "track" in args:
        return _audio_track(timeline, args["track"])
    role = _role(args.get("role", "dialogue"))
    found = next((t for t in _ops._audio_tracks(timeline) if t["role"] == role), None)
    return found if found is not None else _add_track(timeline, role)


def _make_sound(state, timeline, spec, start: int, reserved: set[str]) -> dict[str, Any]:
    if type(spec) is not dict:
        _ops._bad("clip must be an object")
    for key in spec:
        if key not in _SOUND_FIELDS:
            _ops._bad(f"clip: unknown field {key}")
    for key in ("source", "media_info"):
        if key not in spec:
            _ops._bad(f"clip: missing field {key}")
    ident = spec.get("id") or _ops.new_id(state, "a", reserved)
    if ident in _ops._all_ids(state) or ident in reserved:
        _ops._bad(f"clip id {ident} is already used")
    info = spec["media_info"]
    if type(info) is not dict:
        _ops._bad("clip: media_info must be an object")
    item = {
        "id": ident, "type": "clip", "duration_frames": 0, "source": copy.deepcopy(spec["source"]),
        "source_in_frame": spec.get("source_in_frame", 0), "source_rate": copy.deepcopy(spec.get("source_rate", timeline["rate"])),
        "media_info": {"frames": info.get("frames"), "sample_rate": info.get("sample_rate"), "channels": info.get("channels")},
        "record_start_frame": start, "gain_db": _gain(spec.get("gain_db", 0)),
        "fade_in_frames": spec.get("fade_in_frames", 0), "fade_out_frames": spec.get("fade_out_frames", 0),
        "enabled": spec.get("enabled", True), "label": spec.get("label", ""), "note": spec.get("note", ""),
    }
    rate = item["source_rate"]
    frames = item["media_info"]["frames"]
    if (type(frames) is not int or type(item["source_in_frame"]) is not int or type(rate) is not dict
            or any(type(rate.get(k)) is not int or rate.get(k) < 1 for k in ("num", "den"))):
        _ops._bad("clip: media_info.frames, source_in_frame or source_rate is invalid")
    if "duration_frames" in spec:
        item["duration_frames"] = spec["duration_frames"]
    else:
        item["duration_frames"] = _ops._to_rec(frames - item["source_in_frame"], item, timeline["rate"])
    return item


def op_place_audio(state, args, ctx):
    """A sound file or a take's sound on A1-A3 at ``at_frame`` (free time only; ``attach`` follows a picture)."""
    _ops._args(args, ("clip", "at_frame"), ("track", "role", "attach"))
    timeline = _ops._timeline(state, args)
    if "track" in args and "role" in args:
        _ops._bad("place_audio takes track or role, not both")
    at = _ops._int(args["at_frame"], "at_frame", 0)
    track = _ops._unlocked(_target_track(timeline, args))
    reserved: set[str] = set()
    item = _make_sound(state, timeline, args["clip"], at, reserved)
    if args.get("attach") is not None:
        target_track, target_index, _target = _ops._picture(timeline, args["attach"])
        item["attach"] = {"clip": args["attach"], "offset_frames": at - _ops._starts(target_track)[target_index]}
    end = at + item["duration_frames"] if type(item["duration_frames"]) is int else at
    hit = _overlaps(track, at, end)
    if hit is not None:
        fail("overlap", f"{item['id']} would overlap {hit} on {track['id']}")
    track["items"].append(item)
    _ops._sort_overlay(track)
    return {"op": "place_audio", "track": track["id"], "created": [item["id"]]}


def op_set_gain(state, args, ctx):
    _ops._args(args, ("clip", "gain_db"))
    track, _index, item = _sound(_ops._timeline(state, args), args["clip"])
    _ops._unlocked(track)
    item["gain_db"] = _gain(args["gain_db"])
    return {"op": "set_gain", "gain_db": item["gain_db"]}


def op_set_fades(state, args, ctx):
    """Fade lengths in frames, clamped so both fades fit in the clip (the answer names what applied)."""
    _ops._args(args, ("clip",), ("fade_in_frames", "fade_out_frames"))
    track, _index, item = _sound(_ops._timeline(state, args), args["clip"])
    _ops._unlocked(track)
    if "fade_in_frames" not in args and "fade_out_frames" not in args:
        _ops._bad("set_fades takes fade_in_frames and/or fade_out_frames")
    duration = item["duration_frames"]
    fade_in = _ops._int(args.get("fade_in_frames", item["fade_in_frames"]), "fade_in_frames", 0)
    fade_out = _ops._int(args.get("fade_out_frames", item["fade_out_frames"]), "fade_out_frames", 0)
    if "fade_in_frames" in args:
        fade_in = min(fade_in, duration - min(fade_out, duration))
        fade_out = min(fade_out, duration - fade_in)
    else:
        fade_out = min(fade_out, duration - fade_in)
    item["fade_in_frames"], item["fade_out_frames"] = fade_in, fade_out
    return {"op": "set_fades", "fade_in_frames": fade_in, "fade_out_frames": fade_out}


def op_link_sound(state, args, ctx):
    """A picture clip's own sound onto a sound track, linked (moves, trims, splits and goes with it).

    The picture's ``audio`` becomes ``off``: the sound is now the linked clip, never twice."""
    _ops._args(args, ("clip",), ("track", "role", "gain_db"))
    timeline = _ops._timeline(state, args)
    if "track" in args and "role" in args:
        _ops._bad("link_sound takes track or role, not both")
    picture_track, index, picture = _ops._picture(timeline, args["clip"])
    _ops._unlocked(picture_track)
    if not _ops._moving(picture) or not picture["media_info"]["has_audio"]:
        fail("no_sound", f"{picture['id']} has no sound of its own")
    if picture.get("retime") is not None:
        fail("retimed", f"{picture['id']} changes speed; its sound is not retimed yet")
    for track in _ops._audio_tracks(timeline):
        for item in track["items"]:
            if (item.get("link") or {}).get("clip") == picture["id"]:
                fail("already_linked", f"{picture['id']} is linked to {item['id']} on {track['id']}")
    track = _ops._unlocked(_target_track(timeline, args))
    start = _ops._starts(picture_track)[index]
    item = _make_sound(state, timeline, {
        "source": {"kind": picture["source"]["kind"], "ref": picture["source"]["ref"]},
        "source_in_frame": picture["source_in_frame"], "duration_frames": picture["duration_frames"],
        "source_rate": picture["source_rate"], "media_info": {"frames": picture["media_info"]["frames"]},
        "gain_db": args.get("gain_db", 0), "label": picture["label"],
    }, start, set())
    item["link"] = {"clip": picture["id"], "offset_frames": 0}
    hit = _overlaps(track, start, start + item["duration_frames"])
    if hit is not None:
        fail("overlap", f"{item['id']} would overlap {hit} on {track['id']}")
    track["items"].append(item)
    _ops._sort_overlay(track)
    picture["audio"] = "off"
    return {"op": "link_sound", "track": track["id"], "created": [item["id"]]}


def op_unlink_sound(state, args, ctx):
    _ops._args(args, ("clip",))
    track, _index, item = _sound(_ops._timeline(state, args), args["clip"])
    _ops._unlocked(track)
    if item.get("link") is None:
        _ops._bad(f"{item['id']} is not linked")
    item["link"] = None
    return {"op": "unlink_sound"}


AUDIO_OPS = {
    "add_audio_track": op_add_audio_track, "remove_audio_track": op_remove_audio_track, "set_track": op_set_track,
    "place_audio": op_place_audio, "set_gain": op_set_gain, "set_fades": op_set_fades,
    "link_sound": op_link_sound, "unlink_sound": op_unlink_sound,
}

from .ops_volume import VOLUME_OPS  # noqa: E402  (volume keys + auto_duck, lane fs-duck; ops.py takes AUDIO_OPS whole)

AUDIO_OPS.update(VOLUME_OPS)


def split_linked(state: dict[str, Any], timeline: dict[str, Any], left_id: str, right_id: str, reserved: set[str]) -> None:
    """A picture cut in two: each sound linked to it is cut too (the right part links the right picture)."""
    for track in _ops._audio_tracks(timeline):
        for index, item in enumerate(list(track["items"])):
            if (item.get("link") or {}).get("clip") != left_id:
                continue
            right = copy.deepcopy(item)
            right["id"] = _ops.new_id(state, "a", reserved)
            reserved.add(right["id"])
            right["link"]["clip"] = right_id
            right["fade_in_frames"], item["fade_out_frames"] = 0, 0
            track["items"].insert(track["items"].index(item) + 1, right)


def sync_sound(timeline: dict[str, Any], clips: dict[str, dict[str, Any]], edges: dict[str, tuple[int, int]]) -> None:
    """Keep every linked sound on its picture; a deleted picture takes its sound with it, a picture
    that no longer has sound (swapped to a still) releases it in place."""
    for track in _ops._audio_tracks(timeline):
        kept = []
        for item in track["items"]:
            link = item.get("link")
            if link is not None:
                picture = clips.get(link["clip"])
                if picture is None:
                    continue  # the picture is gone: its own sound goes with it
                if not _ops._moving(picture) or not picture["media_info"]["has_audio"]:
                    item["link"] = None
                else:
                    if item["source"]["ref"] != picture["source"]["ref"]:
                        item["media_info"] = {"frames": picture["media_info"]["frames"], "sample_rate": None, "channels": None}
                        item.pop("volume_keys", None)  # keys sit in the old sound's frames
                    else:
                        item["media_info"]["frames"] = picture["media_info"]["frames"]
                    item["source"] = {"kind": picture["source"]["kind"], "ref": picture["source"]["ref"]}
                    item["source_rate"] = copy.deepcopy(picture["source_rate"])
                    item["source_in_frame"] = picture["source_in_frame"]
                    item["duration_frames"] = picture["duration_frames"]
                    item["record_start_frame"] = edges[picture["id"]][0] + link["offset_frames"]
                    if item["record_start_frame"] < 0:
                        fail("out_of_range", f"{item['id']} would start before frame 0")
            # A trim, roll or split may leave fades longer than the clip: they shrink with it.
            item["fade_in_frames"] = min(item["fade_in_frames"], item["duration_frames"])
            item["fade_out_frames"] = min(item["fade_out_frames"], item["duration_frames"] - item["fade_in_frames"])
            kept.append(item)
        track["items"] = kept
        _ops._sort_overlay(track)
