"""Timeline edit algebra (Python twin of ``desktop/film-suite/src/timeline/ops``).

Every verb is a pure function of ``(state, args)``: the input is never
mutated, the output is validated with ``schema.validate_state`` before it is
returned, and a refusal is a ``TimelineError`` with a stable code. Deltas are
clamped first (the legal delta is computed before anything moves, after
filmcraft's ``clamp_trim``) and the result names the requested and the applied
delta, so a caller can ask "how far can this go" with ``dry_run``.

The shape of trim/roll/slip/split/overwrite/ripple delete follows the
storytold/filmcraft edit crate (MIT OR Apache-2.0, crates/edit/src/lib.rs:
insert 278, overwrite 260, split 309-341, delete 372/443, trim 654-781,
roll 887-934, slip 936-948), ported to integer frames and a magnetic
storyline. Ripple of attached overlay clips and markers is a post-pass that
compares edges before and after (OpenCut classic ripple/diff.ts).

The shared fixture suite ``tests/backend/fixtures/timeline/*.json`` runs
against this module and the JS twin; both must give byte-identical results.
"""
from __future__ import annotations

import copy
import re
from typing import Any, Callable

from .schema import (MAX_FRAME, TimelineError, clip_source_frames, fail, frozen, handles, reversed_clip, sound_follows,
                     used_source_frames, validate_state)

INF = MAX_FRAME
_NUMBER_RE = re.compile(r"^(-?)(\d+)(?:\.(\d+))?(?:[eE]([+-]?\d+))?$")


# --- small helpers ------------------------------------------------------------

def _bad(message: str) -> None:
    fail("invalid_op", message)


def _args(args: dict[str, Any], required: tuple[str, ...], optional: tuple[str, ...] = ()) -> None:
    allowed = set(required) | set(optional) | {"op", "timeline"}
    for key in args:
        if key not in allowed:
            _bad(f"{args['op']}: unknown argument {key}")
    for key in required:
        if key not in args:
            _bad(f"{args['op']}: missing argument {key}")


def _int(value: Any, name: str, minimum: int = -INF, maximum: int = INF) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _bad(f"{name} must be an integer in {minimum}..{maximum}")
    return value


def _id_list(value: Any, name: str) -> list[str]:
    if type(value) is not list or not value or any(type(v) is not str for v in value) or len(set(value)) != len(value):
        _bad(f"{name} must be a non-empty list of distinct ids")
    return value


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(value, max(lo, hi)))


def _timeline(state: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    wanted = args.get("timeline", state["active"])
    for timeline in state["timelines"]:
        if timeline["id"] == wanted:
            return timeline
    fail("not_found", f"timeline {wanted} does not exist")
    raise AssertionError


def _track(timeline: dict[str, Any], track_id: str) -> dict[str, Any]:
    for track in timeline["tracks"]:
        if track["id"] == track_id:
            return track
    fail("not_found", f"track {track_id} does not exist")
    raise AssertionError


def _video_tracks(timeline: dict[str, Any]) -> list[dict[str, Any]]:
    return [track for track in timeline["tracks"] if track["kind"] == "video"]


def _audio_tracks(timeline: dict[str, Any]) -> list[dict[str, Any]]:
    return [track for track in timeline["tracks"] if track["kind"] == "audio"]


def _item_tracks(timeline: dict[str, Any]) -> list[dict[str, Any]]:
    """Every track that holds items: pictures (V1, V2) and sound (A1-A3)."""
    return [track for track in timeline["tracks"] if track["kind"] in ("video", "audio")]


def _free(track: dict[str, Any]) -> bool:
    """Items at their own record frames (V2 and the sound tracks), not a magnetic storyline."""
    return track["kind"] == "audio" or track["role"] == "overlay"


def _storyline(timeline: dict[str, Any]) -> dict[str, Any]:
    return next(track for track in timeline["tracks"] if track.get("role") == "storyline")


def _unlocked(track: dict[str, Any]) -> dict[str, Any]:
    if track["locked"]:
        fail("locked", f"track {track['id']} is locked")
    return track


def _find(timeline: dict[str, Any], item_id: Any) -> tuple[dict[str, Any], int]:
    for track in _item_tracks(timeline):
        for index, item in enumerate(track["items"]):
            if item["id"] == item_id:
                return track, index
    fail("not_found", f"clip {item_id} does not exist")
    raise AssertionError


def _clip(timeline: dict[str, Any], item_id: Any) -> tuple[dict[str, Any], int, dict[str, Any]]:
    track, index = _find(timeline, item_id)
    item = track["items"][index]
    if item["type"] != "clip":
        _bad(f"{item_id} is a gap, not a clip")
    return track, index, item


def _picture(timeline: dict[str, Any], item_id: Any) -> tuple[dict[str, Any], int, dict[str, Any]]:
    """A picture clip (V1/V2): the verbs that only pictures have (swap, clip sound, markers)."""
    track, index, item = _clip(timeline, item_id)
    if track["kind"] != "video":
        _bad(f"{item_id} is a sound clip; this verb is for picture clips")
    return track, index, item


def _not_linked(item: dict[str, Any]) -> None:
    if item.get("link") is not None:
        fail("linked", f"{item['id']} is the linked sound of {item['link']['clip']}")


def _starts(track: dict[str, Any]) -> list[int]:
    if _free(track):
        return [item["record_start_frame"] for item in track["items"]]
    out, at = [], 0
    for item in track["items"]:
        out.append(at)
        at += item["duration_frames"]
    return out


def _end(track: dict[str, Any]) -> int:
    if not track["items"]:
        return 0
    starts = _starts(track)
    return starts[-1] + track["items"][-1]["duration_frames"]


def _all_ids(state: dict[str, Any]) -> set[str]:
    ids = {slot["id"] for slot in state["slots"]}
    for timeline in state["timelines"]:
        ids.add(timeline["id"])
        ids.update(marker["id"] for marker in timeline["markers"])
        ids.update(transition["id"] for transition in timeline["transitions"])
        for track in timeline["tracks"]:
            ids.add(track["id"])
            ids.update(item["id"] for item in track.get("items") or ())
    return ids


def new_id(state: dict[str, Any], prefix: str, reserved: set[str] = frozenset()) -> str:
    """Deterministic: one past the highest ``<prefix><n>`` in use (both twins)."""
    pattern = re.compile("^" + re.escape(prefix) + r"(\d{1,9})$")
    highest = 0
    for ident in _all_ids(state) | set(reserved):
        match = pattern.fullmatch(ident)
        if match:
            highest = max(highest, int(match.group(1)))
    return f"{prefix}{highest + 1}"


def _moving(item: dict[str, Any]) -> bool:
    return item["type"] == "clip" and item["media_info"]["frames"] is not None


def _to_src(frames: int, item: dict[str, Any], rate: dict[str, int]) -> int:
    """Record frames -> source frames, toward zero."""
    sr = item["source_rate"]
    magnitude = abs(frames) * sr["num"] * rate["den"] // (sr["den"] * rate["num"])
    return magnitude if frames >= 0 else -magnitude


def _to_rec(frames: int, item: dict[str, Any], rate: dict[str, int]) -> int:
    sr = item["source_rate"]
    return frames * sr["den"] * rate["num"] // (sr["num"] * rate["den"])


def _head(item: dict[str, Any], rate: dict[str, int]) -> int:
    """Record frames of media before the in point (unbounded for stills, gaps and freezes)."""
    return handles(item, rate)[0] if _moving(item) else INF


def _tail(item: dict[str, Any], rate: dict[str, int]) -> int:
    """Record frames of media after the out point."""
    return handles(item, rate)[1] if _moving(item) else INF


def _retimed(item: dict[str, Any]) -> bool:
    return item["type"] == "clip" and item.get("retime") is not None


def _edge_moved(item: dict[str, Any], edge: str, old_duration: int, rate: dict[str, int]) -> None:
    """After a retimed clip's ``edge`` moved (its duration is already new): the source end that edge
    shows follows, the other end stays. A forward clip's in edge shows its lowest source frame, a
    reversed clip's out edge does; a freeze keeps its one frame."""
    if not _moving(item) or frozen(item):
        return
    if (edge == "in") != reversed_clip(item):
        high = item["source_in_frame"] + clip_source_frames({**item, "duration_frames": old_duration}, rate)
        item["source_in_frame"] = high - clip_source_frames(item, rate)


def _shift_in(item: dict[str, Any], applied: int, rate: dict[str, int]) -> None:
    """The in edge moved ``applied`` record frames later and the duration shrank by it."""
    if _retimed(item):
        item["duration_frames"] -= applied
        _edge_moved(item, "in", item["duration_frames"] + applied, rate)
        return
    if _moving(item):
        item["source_in_frame"] += _to_src(applied, item, rate)
    item["duration_frames"] -= applied


def _shift_out(item: dict[str, Any], applied: int, rate: dict[str, int]) -> None:
    """The out edge moved ``applied`` record frames later (the duration grew by it)."""
    item["duration_frames"] += applied
    if _retimed(item):
        _edge_moved(item, "out", item["duration_frames"] - applied, rate)


def _sort_overlay(track: dict[str, Any]) -> None:
    if _free(track):
        track["items"].sort(key=lambda item: item["record_start_frame"])


# --- clip construction ----------------------------------------------------------

def _make_clip(state: dict[str, Any], timeline: dict[str, Any], spec: Any, *, overlay: bool, start: int, reserved: set[str]) -> dict[str, Any]:
    if type(spec) is not dict:
        _bad("clip must be an object")
    allowed = {"id", "source", "source_in_frame", "duration_frames", "source_rate", "media_info", "audio", "enabled", "label", "note"}
    for key in spec:
        if key not in allowed:
            _bad(f"clip: unknown field {key}")
    for key in ("source", "media_info"):
        if key not in spec:
            _bad(f"clip: missing field {key}")
    ident = spec.get("id") or new_id(state, "c", reserved)
    if ident in _all_ids(state) or ident in reserved:
        _bad(f"clip id {ident} is already used")
    item: dict[str, Any] = {
        "id": ident,
        "type": "clip",
        "duration_frames": 0,
        "source": copy.deepcopy(spec["source"]),
        "source_in_frame": spec.get("source_in_frame", 0),
        "source_rate": copy.deepcopy(spec.get("source_rate", timeline["rate"])),
        "media_info": copy.deepcopy(spec["media_info"]),
        "audio": spec.get("audio", "off"),
        "enabled": spec.get("enabled", True),
        "label": spec.get("label", ""),
        "note": spec.get("note", ""),
        "anchors": {"in": None, "out": None},
    }
    rate = item["source_rate"]
    if (type(item["media_info"]) is not dict or type(item["source_in_frame"]) is not int or type(rate) is not dict
            or any(type(rate.get(k)) is not int or rate.get(k) < 1 for k in ("num", "den"))):
        _bad("clip: media_info, source_in_frame or source_rate is invalid")
    if "duration_frames" in spec:
        item["duration_frames"] = spec["duration_frames"]
    elif item["media_info"].get("frames") is not None and type(item["media_info"]["frames"]) is int:
        item["duration_frames"] = _to_rec(item["media_info"]["frames"] - item["source_in_frame"], item, timeline["rate"])
    else:
        _bad("clip: a still needs duration_frames")
    if overlay:
        item["record_start_frame"] = start
    return item


def _split_item(state: dict[str, Any], timeline: dict[str, Any], track: dict[str, Any], index: int, at: int, reserved: set[str]) -> str:
    """Blade one item at record frame ``at`` (strictly inside). Returns the right part's id."""
    item = track["items"][index]
    start = _starts(track)[index]
    offset = at - start
    right = copy.deepcopy(item)
    right["id"] = new_id(state, "g" if item["type"] == "gap" else "a" if track["kind"] == "audio" else "c", reserved)
    reserved.add(right["id"])
    right["duration_frames"] = item["duration_frames"] - offset
    item["duration_frames"] = offset
    if item["type"] == "clip":
        if _retimed(item):
            _split_retimed(item, right, item["duration_frames"] + right["duration_frames"], timeline["rate"])
        elif _moving(item):
            right["source_in_frame"] = item["source_in_frame"] + _to_src(offset, item, timeline["rate"])
        if "anchors" in item:
            right["anchors"] = {"in": None, "out": item["anchors"]["out"]}
            item["anchors"] = {"in": item["anchors"]["in"], "out": None}
        else:  # a sound clip: the left part keeps the fade in, the right part the fade out
            right["fade_in_frames"], item["fade_out_frames"] = 0, 0
            item["fade_in_frames"] = min(item["fade_in_frames"], item["duration_frames"])
            right["fade_out_frames"] = min(right["fade_out_frames"], right["duration_frames"])
        if _free(track):
            right["record_start_frame"] = at
            if right.get("attach") is not None:
                right["attach"]["offset_frames"] += offset
    track["items"].insert(index + 1, right)
    if track["kind"] == "video" and item["type"] == "clip":
        split_linked(state, timeline, item["id"], right["id"], reserved)
    return right["id"]


def _split_retimed(left: dict[str, Any], right: dict[str, Any], whole: int, rate: dict[str, int]) -> None:
    """Source ranges of the two parts of a retimed clip (durations already set): a forward clip's
    right part starts where the left part's frames run to; a reversed clip's right part keeps the
    lowest frames and the left part keeps the highest end."""
    if frozen(left) or not _moving(left):
        return
    if reversed_clip(left):
        high = left["source_in_frame"] + clip_source_frames({**left, "duration_frames": whole}, rate)
        left["source_in_frame"] = high - clip_source_frames(left, rate)
    else:
        right["source_in_frame"] = left["source_in_frame"] + _to_src(left["duration_frames"], {**left, "source_rate": _eff(left)}, rate)


def _eff(item: dict[str, Any]) -> dict[str, int]:
    from .schema import effective_rate
    return effective_rate(item)


# --- verbs ---------------------------------------------------------------------------

def _insert(state: dict[str, Any], args: dict[str, Any], ctx: dict[str, Any], *, append: bool) -> dict[str, Any]:
    if append:
        _args(args, ("clip",), ("track", "attach"))
    else:
        _args(args, ("clip", "at_frame"), ("track", "attach"))
    timeline = _timeline(state, args)
    track = _unlocked(_track(timeline, args.get("track", _storyline(timeline)["id"])))
    if track["kind"] != "video":
        _bad("clips go on a video track")
    at = _end(track) if append else _int(args["at_frame"], "at_frame", 0)
    reserved: set[str] = set()
    result: dict[str, Any] = {"op": args["op"]}
    if track["role"] == "storyline":
        if "attach" in args:
            _bad("attach is for overlay clips")
        if at > _end(track):
            fail("out_of_range", f"at_frame {at} is past the end ({_end(track)})")
        starts = _starts(track)
        index = len(track["items"])
        for i, start in enumerate(starts):
            end = start + track["items"][i]["duration_frames"]
            if start == at:
                index = i
                break
            if start < at < end:
                result["split"] = _split_item(state, timeline, track, i, at, reserved)
                index = i + 1
                break
        clip = _make_clip(state, timeline, args["clip"], overlay=False, start=0, reserved=reserved)
        track["items"].insert(index, clip)
    else:
        clip = _make_clip(state, timeline, args["clip"], overlay=True, start=at, reserved=reserved)
        if args.get("attach") is not None:
            target_track, target_index, _target = _clip(timeline, args["attach"])
            if target_track["role"] != "storyline":
                _bad("attach names a storyline clip")
            clip["attach"] = {"clip": args["attach"], "offset_frames": at - _starts(target_track)[target_index]}
        end = at + clip["duration_frames"] if type(clip["duration_frames"]) is int else at
        for i, item in enumerate(track["items"]):
            if item["record_start_frame"] < end and at < item["record_start_frame"] + item["duration_frames"]:
                fail("overlap", f"{clip['id']} would overlap {item['id']} on {track['id']}")
        track["items"].append(clip)
        _sort_overlay(track)
    result["created"] = [clip["id"]]
    return result


def op_append(state, args, ctx):
    return _insert(state, args, ctx, append=True)


def op_insert(state, args, ctx):
    return _insert(state, args, ctx, append=False)


def op_overwrite(state, args, ctx):
    _args(args, ("clip", "at_frame"), ("track",))
    timeline = _timeline(state, args)
    track = _unlocked(_track(timeline, args.get("track", "V2")))
    if track["kind"] != "video" or track["role"] != "overlay":
        fail("not_on_storyline", "overwrite places on an overlay track; the storyline is always rippled")
    at = _int(args["at_frame"], "at_frame", 0)
    reserved: set[str] = set()
    clip = _make_clip(state, timeline, args["clip"], overlay=True, start=at, reserved=reserved)
    reserved.add(clip["id"])
    end = at + _int(clip["duration_frames"], "duration_frames", 1)
    removed: list[str] = []
    index = 0
    while index < len(track["items"]):
        item = track["items"][index]
        start, stop = item["record_start_frame"], item["record_start_frame"] + item["duration_frames"]
        if stop <= at or start >= end:
            index += 1
            continue
        if start >= at and stop <= end:
            removed.append(item["id"])
            track["items"].pop(index)
            continue
        if start < at and stop > end:
            _split_item(state, timeline, track, index, end, reserved)
            continue  # the left part now ends at ``end`` and is cut next round
        if start < at:
            _shift_out(item, (at - start) - item["duration_frames"], timeline["rate"])
            item["anchors"]["out"] = None
        else:
            cut = end - start
            _shift_in(item, cut, timeline["rate"])
            item["record_start_frame"] = end
            item["anchors"]["in"] = None
            if item.get("attach") is not None:
                item["attach"]["offset_frames"] += cut
        index += 1
    track["items"].append(clip)
    _sort_overlay(track)
    return {"op": "overwrite", "created": [clip["id"]], "removed": removed}


def op_split(state, args, ctx):
    _args(args, ("at_frame",), ("tracks",))
    timeline = _timeline(state, args)
    at = _int(args["at_frame"], "at_frame", 1)
    if "tracks" in args:
        tracks = [_unlocked(_track(timeline, t)) for t in _id_list(args["tracks"], "tracks")]
    else:
        tracks = [t for t in _item_tracks(timeline) if not t["locked"]]
    created: list[str] = []
    reserved: set[str] = set()
    for track in tracks:
        if track["kind"] not in ("video", "audio"):
            _bad("split cuts picture and sound tracks")
        starts = _starts(track)
        for index, item in enumerate(track["items"]):
            if item.get("link") is not None:
                continue  # a linked sound is cut with its picture
            if item["type"] == "clip" and starts[index] < at < starts[index] + item["duration_frames"]:
                created.append(_split_item(state, timeline, track, index, at, reserved))
                break
    if not created:
        fail("nothing_to_split", f"no clip runs across frame {at}")
    return {"op": "split", "created": created}


def _neighbours(track: dict[str, Any], index: int) -> tuple[int, int]:
    items = track["items"]
    prev_end = 0 if index == 0 else items[index - 1]["record_start_frame"] + items[index - 1]["duration_frames"]
    next_start = INF if index + 1 >= len(items) else items[index + 1]["record_start_frame"]
    return prev_end, next_start


def op_trim(state, args, ctx):
    _args(args, ("clip", "edge", "delta"))
    timeline = _timeline(state, args)
    track, index = _find(timeline, args["clip"])
    _unlocked(track)
    item = track["items"][index]
    _not_linked(item)
    edge = args["edge"]
    if edge not in ("in", "out"):
        _bad("edge must be in or out")
    delta = _int(args["delta"], "delta")
    rate = timeline["rate"]
    overlay = _free(track)
    if edge == "in":
        lo = -_head(item, rate)
        if overlay:
            prev_end, _next = _neighbours(track, index)
            lo = max(lo, -(item["record_start_frame"] - prev_end))
        applied = _clamp(delta, lo, item["duration_frames"] - ctx["min_frames"])
    else:
        hi = _tail(item, rate)
        if overlay:
            _prev, next_start = _neighbours(track, index)
            hi = min(hi, next_start - (item["record_start_frame"] + item["duration_frames"]))
        applied = _clamp(delta, -(item["duration_frames"] - ctx["min_frames"]), hi)
    if applied and item["type"] == "clip" and "anchors" in item:
        item["anchors"][edge] = None  # an edge the artist moves leaves its beat
    if edge == "in":
        _shift_in(item, applied, rate)
        if overlay:
            item["record_start_frame"] += applied
            if item.get("attach") is not None:
                item["attach"]["offset_frames"] += applied
    else:
        _shift_out(item, applied, rate)
    return {"op": "trim", "requested": delta, "applied": applied}


def op_roll(state, args, ctx):
    _args(args, ("left", "delta"), ("right",))
    timeline = _timeline(state, args)
    track, index = _find(timeline, args["left"])
    _unlocked(track)
    if index + 1 >= len(track["items"]):
        fail("not_adjacent", f"{args['left']} has no item after it")
    left, right = track["items"][index], track["items"][index + 1]
    _not_linked(left)
    _not_linked(right)
    if "right" in args and args["right"] != right["id"]:
        fail("not_adjacent", f"{args['right']} does not follow {args['left']}")
    if _free(track) and left["record_start_frame"] + left["duration_frames"] != right["record_start_frame"]:
        fail("not_adjacent", "the clips do not touch")
    delta = _int(args["delta"], "delta")
    rate = timeline["rate"]
    lo = max(-(left["duration_frames"] - ctx["min_frames"]), -_head(right, rate))
    hi = min(right["duration_frames"] - ctx["min_frames"], _tail(left, rate))
    applied = _clamp(delta, lo, hi)
    if applied:
        for item, edge in ((left, "out"), (right, "in")):
            if item["type"] == "clip" and "anchors" in item:
                item["anchors"][edge] = None
    _shift_out(left, applied, rate)
    _shift_in(right, applied, rate)
    if _free(track):
        right["record_start_frame"] += applied
        if right.get("attach") is not None:
            right["attach"]["offset_frames"] += applied
    return {"op": "roll", "requested": delta, "applied": applied}


def op_slip(state, args, ctx):
    _args(args, ("clip", "delta"))
    timeline = _timeline(state, args)
    track, _index, item = _clip(timeline, args["clip"])
    _unlocked(track)
    _not_linked(item)
    delta = _int(args["delta"], "delta")
    if not _moving(item):
        return {"op": "slip", "requested": delta, "applied": 0}
    used = clip_source_frames(item, timeline["rate"])
    new_in = _clamp(item["source_in_frame"] + delta, 0, item["media_info"]["frames"] - used)
    applied = new_in - item["source_in_frame"]
    item["source_in_frame"] = new_in
    return {"op": "slip", "requested": delta, "applied": applied}


def op_move(state, args, ctx):
    _args(args, ("clips",), ("to_index", "delta", "to_track"))
    timeline = _timeline(state, args)
    ids = _id_list(args["clips"], "clips")
    located = [_find(timeline, ident) for ident in ids]
    track = located[0][0]
    if any(t is not track for t, _i in located):
        _bad("move takes clips of one track")
    _unlocked(track)
    for t, i in located:
        _not_linked(t["items"][i])
    if "to_track" in args:
        if track["kind"] != "audio":
            _bad("to_track moves sound clips between sound tracks")
        target = _unlocked(_track(timeline, args["to_track"]))
        if target["kind"] != "audio":
            _bad("to_track names a sound track")
        if target is not track:
            moving = [item for item in track["items"] if item["id"] in ids]
            track["items"] = [item for item in track["items"] if item["id"] not in ids]
            target["items"].extend(moving)
            track = target
    if track["kind"] == "video" and track["role"] == "storyline":
        if "delta" in args or "to_index" not in args:
            _bad("storyline move takes to_index (reorder)")
        moving = [item for item in track["items"] if item["id"] in ids]
        rest = [item for item in track["items"] if item["id"] not in ids]
        to_index = _int(args["to_index"], "to_index", 0, len(rest))
        track["items"] = rest[:to_index] + moving + rest[to_index:]
        return {"op": "move"}
    if "to_index" in args or ("delta" not in args and "to_track" not in args):
        _bad("overlay move takes delta (frames)")
    delta = _int(args.get("delta", 0), "delta")
    for item in track["items"]:
        if item["id"] in ids:
            if item["record_start_frame"] + delta < 0:
                fail("out_of_range", f"{item['id']} would start before frame 0")
            item["record_start_frame"] += delta
            if item.get("attach") is not None:
                item["attach"]["offset_frames"] += delta
    _sort_overlay(track)
    return {"op": "move"}


def op_delete(state, args, ctx):
    _args(args, ("clips",), ("ripple",))
    timeline = _timeline(state, args)
    ids = _id_list(args["clips"], "clips")
    ripple = args.get("ripple", True)
    if type(ripple) is not bool:
        _bad("ripple must be true or false")
    for ident in ids:
        _unlocked(_find(timeline, ident)[0])
    reserved: set[str] = set()
    for track in _item_tracks(timeline):
        if _free(track) or ripple:
            track["items"] = [item for item in track["items"] if item["id"] not in ids]
            continue
        items = []
        for item in track["items"]:
            if item["id"] in ids:
                gap_id = new_id(state, "g", reserved)
                reserved.add(gap_id)
                item = {"id": gap_id, "type": "gap", "duration_frames": item["duration_frames"]}
            items.append(item)
        track["items"] = items
    return {"op": "delete", "removed": list(ids)}


def op_swap_take(state, args, ctx):
    _args(args, ("clip", "ref", "media_info"), ("kind", "source_rate"))
    timeline = _timeline(state, args)
    track, _index, item = _picture(timeline, args["clip"])
    _unlocked(track)
    info = args["media_info"]
    if type(info) is not dict or "frames" not in info:
        _bad("media_info must give frames (null for a still)")
    item["source"]["ref"] = args["ref"]
    item["source"]["kind"] = args.get("kind", item["source"]["kind"])
    item["media_info"] = copy.deepcopy(info)
    if "source_rate" in args:
        item["source_rate"] = copy.deepcopy(args["source_rate"])
    frames = info["frames"]
    if frames is None:
        item["source_in_frame"] = 0
    else:
        _int(frames, "media_info.frames", 1)
        used = clip_source_frames(item, timeline["rate"])
        if frames < used:
            fail("take_too_short", f"the take has {frames} frames; the clip needs {used}")
        item["source_in_frame"] = min(item["source_in_frame"], frames - used)
    slot_id = item["source"].get("slot")
    for slot in state["slots"]:
        if slot["id"] == slot_id:
            if args["ref"] not in slot["takes"]:
                slot["takes"].append(args["ref"])
            slot["selected"] = args["ref"]
    return {"op": "swap_take", "source_in_frame": item["source_in_frame"]}


def _set_clip_field(state, args, field, allowed):
    _args(args, ("clip", field))
    timeline = _timeline(state, args)
    track, _index, item = (_clip if field == "enabled" else _picture)(timeline, args["clip"])
    _unlocked(track)
    if args[field] not in allowed or type(args[field]) is not type(allowed[0]):
        _bad(f"{field} must be one of {allowed}")
    if field == "audio" and args[field] == "on" and not sound_follows(item):
        fail("retimed", f"{item['id']} changes speed; its own sound plays only at one speed of 50 .. 200 % forward (a ramp, hold, reverse or freeze keeps it off)")
    item[field] = args[field]
    return {"op": args["op"]}


def op_set_audio(state, args, ctx):
    return _set_clip_field(state, args, "audio", ("off", "on"))


def op_set_enabled(state, args, ctx):
    return _set_clip_field(state, args, "enabled", (True, False))


def op_set_label(state, args, ctx):
    _args(args, ("clip",), ("label", "note"))
    timeline = _timeline(state, args)
    track, _index, item = _clip(timeline, args["clip"])
    _unlocked(track)
    if "label" not in args and "note" not in args:
        _bad("set_label takes label and/or note")
    for key in ("label", "note"):
        if key in args:
            item[key] = args[key]
    return {"op": "set_label"}


def _sort_markers(timeline: dict[str, Any]) -> None:
    timeline["markers"].sort(key=lambda marker: marker["frame"])


def op_add_marker(state, args, ctx):
    _args(args, ("frame",), ("id", "name", "note", "kind", "color", "shot_id", "by", "duration_frames", "attach_clip"))
    timeline = _timeline(state, args)
    _int(args["frame"], "frame", 0)
    ident = args.get("id") or new_id(state, "m")
    if ident in _all_ids(state):
        _bad(f"marker id {ident} is already used")
    marker = {
        "id": ident, "frame": args["frame"], "duration_frames": args.get("duration_frames", 0),
        "name": args.get("name", ""), "note": args.get("note", ""), "kind": args.get("kind", "note"),
        "color": args.get("color", ""), "shot_id": args.get("shot_id"), "by": args.get("by", "artist"),
    }
    if args.get("attach_clip") is not None:
        _picture(timeline, args["attach_clip"])
        marker["attach_clip"] = args["attach_clip"]
    timeline["markers"].append(marker)
    _sort_markers(timeline)
    return {"op": "add_marker", "created": [ident]}


def _marker(timeline: dict[str, Any], ident: Any) -> dict[str, Any]:
    for marker in timeline["markers"]:
        if marker["id"] == ident:
            return marker
    fail("not_found", f"marker {ident} does not exist")
    raise AssertionError


def op_move_marker(state, args, ctx):
    _args(args, ("id", "frame"))
    timeline = _timeline(state, args)
    _marker(timeline, args["id"])["frame"] = _int(args["frame"], "frame", 0)
    _sort_markers(timeline)
    return {"op": "move_marker"}


def op_remove_marker(state, args, ctx):
    _args(args, ("id",))
    timeline = _timeline(state, args)
    marker = _marker(timeline, args["id"])
    timeline["markers"] = [m for m in timeline["markers"] if m is not marker]
    return {"op": "remove_marker"}


def _music_track(timeline: dict[str, Any]) -> dict[str, Any]:
    for track in timeline["tracks"]:
        if track["kind"] == "music":
            return track
    fail("no_music_lane", "the timeline has no music lane")
    raise AssertionError


# Lane w11-cut item 1 (M-99, owner decision): the song has ONE representation. The sound track with role music IS the
# Music lane: place_music puts the song there as a sound clip (source file:<track_sha256>, the analysis pinned as the
# clip's map_id), so ducking, fades and gain keys work on it; the M1 lane of a new cut stays empty. A legacy document
# whose M1 lane holds the song still reads and fits; placing a song again moves it to the music track.
_SHA_RE = re.compile(r"[0-9a-f]{64}")


def _song_items(timeline: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """(track, item) of every sound clip of a music-role track that is a song file (file:<sha256>)."""
    return [(track, item) for track in _audio_tracks(timeline) if track["role"] == "music" for item in track["items"]
            if item["source"]["kind"] == "media" and _FILE_RE.fullmatch(item["source"]["ref"])]


_FILE_RE = re.compile(r"file:[0-9a-f]{64}")


def song_of(timeline: dict[str, Any], music_map: Any = None) -> dict[str, Any] | None:
    """The song the cut's beats come from, as {track_sha256, map_id, record_start_frame, source_in_frame,
    duration_frames}: the legacy M1 lane when it holds one, else the song clip of the music track (the one whose
    pinned map_id is the map's, else whose recording is the map's track_sha256, else the first)."""
    lane = next((t for t in timeline["tracks"] if t["kind"] == "music"), None)
    if lane is not None and lane.get("music") is not None:
        return lane["music"]
    songs = _song_items(timeline)
    if not songs:
        return None
    wanted = music_map if type(music_map) is dict else {}
    pick = next((i for _t, i in songs if i.get("map_id") is not None and i.get("map_id") == wanted.get("map_id")), None) \
        or next((i for _t, i in songs if i.get("map_id") is None and i["source"]["ref"][5:] == wanted.get("track_sha256")), None) \
        or next((i for _t, i in songs if i["source"]["ref"][5:] == wanted.get("track_sha256")), None) \
        or songs[0][1]
    return {"track_sha256": pick["source"]["ref"][5:], "map_id": pick.get("map_id"), "record_start_frame": pick["record_start_frame"],
            "source_in_frame": pick["source_in_frame"], "duration_frames": pick["duration_frames"]}


def op_place_music(state, args, ctx):
    _args(args, ("track_sha256", "map_id", "duration_frames"), ("record_start_frame", "source_in_frame", "gain_db"))
    timeline = _timeline(state, args)
    for key in ("track_sha256", "map_id"):
        if type(args[key]) is not str or not _SHA_RE.fullmatch(args[key]):
            _bad(f"{key} must be a sha256 (64 hex)")
    start = _int(args.get("record_start_frame", 0), "record_start_frame", 0)
    source_in = _int(args.get("source_in_frame", 0), "source_in_frame", 0)
    duration = _int(args["duration_frames"], "duration_frames", 1)
    from .ops_audio import _add_track, _gain  # noqa: PLC0415  (ops_audio imports this module)
    lane = next((t for t in timeline["tracks"] if t["kind"] == "music"), None)
    if lane is not None and lane.get("music") is not None:
        _unlocked(lane)["music"] = None
    track = next((t for t in _audio_tracks(timeline) if t["role"] == "music"), None)
    track = _unlocked(track if track is not None else _add_track(timeline, "music"))
    # The song has one representation: a clip placed by Use in Cut (map_id) or, in a cut from before the song
    # lane, the same song file placed as a plain sound (no map_id) is replaced, not refused as an overlap.
    same = "file:" + args["track_sha256"]
    old = {item["id"] for _t, item in _song_items(timeline) if _t is track and ("map_id" in item or item["source"]["ref"] == same)}
    track["items"] = [item for item in track["items"] if item["id"] not in old]
    end = start + duration
    for item in track["items"]:
        if item["record_start_frame"] < end and start < item["record_start_frame"] + item["duration_frames"]:
            fail("overlap", f"the song would overlap {item['id']} on {track['id']}")
    ident = new_id(state, "a")
    track["items"].append({
        "id": ident, "type": "clip", "duration_frames": duration, "source": {"kind": "media", "ref": "file:" + args["track_sha256"]},
        "source_in_frame": source_in, "source_rate": copy.deepcopy(timeline["rate"]),
        "media_info": {"frames": source_in + duration, "sample_rate": None, "channels": None},
        "record_start_frame": start, "gain_db": _gain(args.get("gain_db", 0)), "fade_in_frames": 0, "fade_out_frames": 0,
        "enabled": True, "label": "Song", "note": "", "map_id": args["map_id"],
    })
    _sort_overlay(track)
    return {"op": "place_music", "track": track["id"], "created": [ident], "removed": sorted(old)}


def op_remove_music(state, args, ctx):
    _args(args, ())
    timeline = _timeline(state, args)
    lane = next((t for t in timeline["tracks"] if t["kind"] == "music"), None)
    songs = [(t, i) for t, i in _song_items(timeline) if "map_id" in i]
    if lane is not None and lane.get("music") is not None:
        _unlocked(lane)["music"] = None
    for track, item in songs:
        _unlocked(track)["items"] = [i for i in track["items"] if i is not item]
    return {"op": "remove_music"}


def _decimal(value: Any, name: str) -> tuple[int, int]:
    """An exact rational ``(p, q)`` from a JSON number's shortest decimal form."""
    if type(value) is int:
        return value, 1
    if type(value) is not float:
        _bad(f"{name} must be a number")
    text = repr(value)
    match = _NUMBER_RE.fullmatch(text)
    if match is None:
        _bad(f"{name} must be finite")
    sign, whole, fraction, exponent = match.groups()
    fraction = fraction or ""
    p = int(whole + fraction) * (-1 if sign else 1)
    q = 10 ** len(fraction)
    shift = int(exponent or 0)
    if shift >= 0:
        p *= 10 ** shift
    else:
        q *= 10 ** -shift
    return p, q


def music_event_frames(timeline: dict[str, Any], music_map: dict[str, Any], grid: str) -> list[dict[str, int]]:
    """The map's events of one kind as record frames: nearest frame + sub-frame residual (ppm)."""
    music = song_of(timeline, music_map)
    if music is None:
        fail("no_music_lane", "no music is placed on the timeline")
    pinned = music["map_id"] if music["map_id"] is not None else None
    # P3S-11 (CUT-072): a tempo fix makes a new map of the SAME song (same track_sha256). Fit to beats follows the
    # current map and re-pins the placed song's map_id in the same op; a map of another song is still refused.
    # Twin of timeline-music.js musicEventFrames.
    if type(music_map) is not dict or music_map.get("track_sha256") != music["track_sha256"]:
        fail("music_map_mismatch", "Song map changed")
    if pinned is not None and music_map.get("map_id") != pinned:
        new_map = music_map.get("map_id")
        if type(new_map) is not str or not _SHA_RE.fullmatch(new_map):
            fail("music_map_mismatch", "Song map changed")
        lane = next((t for t in timeline["tracks"] if t["kind"] == "music"), None)
        if lane is not None and lane.get("music") is not None:
            _unlocked(lane)["music"]["map_id"] = new_map
        else:
            for track, item in _song_items(timeline):
                if item.get("map_id") == pinned and item["source"]["ref"][5:] == music["track_sha256"]:
                    _unlocked(track)
                    item["map_id"] = new_map
        music = {**music, "map_id": new_map}
    family, key = {"beat": ("beats", "time_seconds"), "downbeat": ("downbeats", "time_seconds"), "section": ("sections", "start_seconds")}[grid]
    rows = music_map.get(family)
    if type(rows) is not list:
        _bad(f"map.{family} must be a list")
    rate = timeline["rate"]
    base = music["record_start_frame"] - music["source_in_frame"]
    lo, hi = music["record_start_frame"], music["record_start_frame"] + music["duration_frames"]
    out = []
    for index, row in enumerate(rows):
        if type(row) is not dict:
            _bad(f"map.{family}[{index}] must be an object")
        p, q = _decimal(row.get(key), f"map.{family}[{index}].{key}")
        num, den = p * rate["num"], q * rate["den"]  # exact frame position in the music source
        nearest = (2 * num + den) // (2 * den)
        residual = (2 * (num - nearest * den) * 1_000_000 + den) // (2 * den)
        frame = base + nearest
        if lo <= frame <= hi:
            out.append({"index": index, "frame": frame, "residual_ppm": residual})
    return out


def op_fit_to_beats(state, args, ctx):
    _args(args, ("clips", "grid", "map"), ("every",))
    timeline = _timeline(state, args)
    grid = args["grid"]
    if grid not in ("beat", "downbeat", "section"):
        _bad("grid must be beat, downbeat or section")
    every = _int(args.get("every", 1), "every", 1, 1024)
    events = [e for e in music_event_frames(timeline, args["map"], grid) if e["index"] % every == 0]
    track = _unlocked(_storyline(timeline))
    ids = _id_list(args["clips"], "clips")
    for ident in ids:
        found_track, _index, _item = _clip(timeline, ident)
        if found_track is not track:
            _bad("fit_to_beats fits storyline clips")
    rate = timeline["rate"]
    # w31-fix4 item 4 (P1 HER-090; Premiere "Automate to Sequence" and Resolve's beat markers report what a beat edit
    # did): every skipped clip says why, and the answer says where the song sits on the cut. index is the map's own
    # event number counted from the start of the song FILE (beat 27 = 27.28 s into the recording), frame the cut frame
    # it lands on: only events inside the placed song's range on the cut count, and a clip's new end must lie within
    # its media (its handles).
    music = song_of(timeline, args["map"])
    first, last = music["record_start_frame"], music["record_start_frame"] + music["duration_frames"]
    family, key = {"beat": ("beats", "time_seconds"), "downbeat": ("downbeats", "time_seconds"), "section": ("sections", "start_seconds")}[grid]
    seconds = {e["index"]: args["map"][family][e["index"]][key] for e in events}
    fitted, skipped = [], []
    for item in [item for item in track["items"] if item["id"] in ids]:
        index = track["items"].index(item)
        start = _starts(track)[index]
        end = start + item["duration_frames"]
        longest = INF if not _moving(item) else item["duration_frames"] + _tail(item, rate)
        best = None
        for event in events:
            if start + ctx["min_frames"] <= event["frame"] <= start + longest:
                if best is None or abs(event["frame"] - end) <= abs(best["frame"] - end):
                    best = event
        if best is None:
            skipped.append(_fit_skip(item["id"], start, longest, events, first, last, ctx["min_frames"]))
            continue
        item["duration_frames"] = best["frame"] - start
        ctx["pending_anchors"].append((item["id"], {
            "map_id": args["map"]["map_id"], "kind": grid, "index": best["index"],
            "offset_frames": 0, "residual_ppm": best["residual_ppm"],
        }))
        fitted.append({"clip": item["id"], "frame": best["frame"], "kind": grid, "index": best["index"], "residual_ppm": best["residual_ppm"],
                       "song_seconds": seconds[best["index"]]})
    song = {"first_frame": first, "last_frame": last, "source_in_frame": music["source_in_frame"], "events": len(events)}
    return {"op": "fit_to_beats", "fitted": fitted, "skipped": skipped, "song": song, "note": _fit_note(grid, song, fitted, skipped)}


def _fit_note(grid: str, song: dict[str, int], fitted: list[Any], skipped: list[dict[str, Any]]) -> str:
    """One sentence for Hermes and the page: what was fitted and what limits the fit."""
    word = {"beat": "beats", "downbeat": "bars", "section": "sections"}[grid]
    text = f"Fitted {len(fitted)} of {len(fitted) + len(skipped)} clips to {word}; the song covers frames {song['first_frame']}-{song['last_frame']} of the cut ({song['events']} {word})."
    reasons: dict[str, int] = {}
    for row in skipped:
        reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
    words = {"past_song_end": "start after the song ends", "past_last_beat": f"start after its last {word[:-1]}",
             "before_song_start": "end before the song starts", "no_beat_in_reach": f"have no {word[:-1]} within their media"}
    parts = [f"{n} {words[r]}" for r, n in reasons.items()]
    return text + (" Not moved: " + ", ".join(parts) + "." if parts else "")


def _fit_skip(clip: str, start: int, longest: Any, events: list[dict[str, int]], first: int, last: int, min_frames: int) -> dict[str, Any]:
    """Why fit_to_beats left a clip alone (w31-fix4 item 4, HER-090)."""
    if start + min_frames > last:
        return {"clip": clip, "reason": "past_song_end", "song_last_frame": last}
    if longest != INF and start + longest < first:
        return {"clip": clip, "reason": "before_song_start", "song_first_frame": first}
    later = next((e["frame"] for e in events if e["frame"] >= start + min_frames), None)
    if later is None:
        return {"clip": clip, "reason": "past_last_beat", "song_last_frame": last}
    return {"clip": clip, "reason": "no_beat_in_reach", "reach_frame": start + longest, "next_event_frame": later}


def op_put_slot(state, args, ctx):
    _args(args, ("slot",))
    slot = args["slot"]
    if type(slot) is not dict:
        _bad("slot must be an object")
    state["slots"] = [s for s in state["slots"] if s["id"] != slot["id"]] + [slot]
    return {"op": "put_slot"}


def op_add_timeline(state, args, ctx):
    _args(args, ("name",), ("id", "rate", "activate"))
    from .schema import new_timeline
    ident = args.get("id") or new_id(state, "tl_")
    if ident in _all_ids(state):
        _bad(f"timeline id {ident} is already used")
    state["timelines"].append(new_timeline(ident, args["name"], args.get("rate")))
    if args.get("activate", True) is True:
        state["active"] = ident
    return {"op": "add_timeline", "created": [ident]}


def op_set_active(state, args, ctx):
    _args(args, ("id",))
    _timeline(state, {"timeline": args["id"]})
    state["active"] = args["id"]
    return {"op": "set_active"}


def op_rename_timeline(state, args, ctx):
    _args(args, ("name",))
    _timeline(state, args)["name"] = args["name"]
    return {"op": "rename_timeline"}


OPS: dict[str, Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], dict[str, Any]]] = {
    "append": op_append, "insert": op_insert, "overwrite": op_overwrite, "split": op_split,
    "trim": op_trim, "roll": op_roll, "slip": op_slip, "move": op_move, "delete": op_delete,
    "swap_take": op_swap_take, "set_audio": op_set_audio, "set_enabled": op_set_enabled,
    "set_label": op_set_label, "add_marker": op_add_marker, "move_marker": op_move_marker,
    "remove_marker": op_remove_marker, "place_music": op_place_music, "remove_music": op_remove_music,
    "fit_to_beats": op_fit_to_beats, "put_slot": op_put_slot, "add_timeline": op_add_timeline,
    "set_active": op_set_active, "rename_timeline": op_rename_timeline,
}


from .ops_audio import AUDIO_OPS, split_linked, sync_sound  # noqa: E402  (the sound verbs reuse the helpers above)

OPS.update(AUDIO_OPS)

from .ops_continue import CONTINUE_OPS  # noqa: E402  (lane fs-obvb: place_with_parent, a continuation take at its parent's join)

OPS.update(CONTINUE_OPS)

from .ops_pro import PRO_OPS, STATE_OPS, curve_guard, curve_heads, fit_transitions  # noqa: E402  (format, retime, transitions)

OPS.update(PRO_OPS)

from .ops_cut import CUT_OPS  # noqa: E402  (lane w2-cutdel: lift_range, extract_range, slide)

OPS.update(CUT_OPS)

from .nest import NEST_OPS, NEST_STATE_OPS  # noqa: E402  (lane w23-bins: delete_timeline; nested cuts)

OPS.update(NEST_OPS)


# --- post-pass and entry points ---------------------------------------------------------

def _edges(timeline: dict[str, Any]) -> dict[str, tuple[int, int]]:
    edges = {}
    for track in _video_tracks(timeline):
        for start, item in zip(_starts(track), track["items"]):
            edges[item["id"]] = (start, start + item["duration_frames"])
    return edges


def _merge_gaps(track: dict[str, Any]) -> None:
    merged: list[dict[str, Any]] = []
    for item in track["items"]:
        if item["type"] == "gap" and merged and merged[-1]["type"] == "gap":
            merged[-1]["duration_frames"] += item["duration_frames"]
        else:
            merged.append(item)
    track["items"] = merged


def _post_pass(timeline: dict[str, Any], before: dict[str, tuple[int, int]], pending: list) -> None:
    _merge_gaps(_storyline(timeline))
    after = _edges(timeline)
    clips = {item["id"]: item for track in _video_tracks(timeline) for item in track["items"] if item["type"] == "clip"}
    for track in _item_tracks(timeline):
        if not _free(track):
            continue
        for item in track["items"]:
            attach = item.get("attach")
            if attach is None:
                continue
            if attach["clip"] in clips and clips[attach["clip"]] is not item:
                item["record_start_frame"] = after[attach["clip"]][0] + attach["offset_frames"]
                if item["record_start_frame"] < 0:
                    fail("out_of_range", f"{item['id']} would start before frame 0")
            else:
                item["attach"] = None
        _sort_overlay(track)
    sync_sound(timeline, clips, _edges(timeline))
    for marker in timeline["markers"]:
        target = marker.get("attach_clip")
        if target is None:
            continue
        if target in after and target in before:
            marker["frame"] += after[target][0] - before[target][0]
            if marker["frame"] < 0:
                fail("out_of_range", f"marker {marker['id']} would move before frame 0")
        elif target not in after:
            marker["attach_clip"] = None
    _sort_markers(timeline)
    after = _edges(timeline)
    for ident, item in clips.items():
        if ident not in before or ident not in after:
            continue
        for edge, position in (("in", 0), ("out", 1)):
            anchor = item["anchors"][edge]
            if anchor is not None:
                anchor["offset_frames"] += after[ident][position] - before[ident][position]
    for ident, anchor in pending:
        clips[ident]["anchors"]["out"] = anchor
    fit_transitions(timeline, before)


def apply_op(state: dict[str, Any], op: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply one verb by name; returns ``(new_state, result)``. ``state`` is untouched."""
    if type(op) is not dict or type(op.get("op")) is not str:
        _bad("an op is an object with an op name")
    handler = OPS.get(op["op"])
    if handler is None:
        fail("unknown_op", f"unknown op {op['op']}")
    work = copy.deepcopy(state)
    args = copy.deepcopy(op)
    target = _timeline(work, args) if op["op"] not in ("add_timeline", "set_active", "put_slot", *STATE_OPS, *NEST_STATE_OPS) else None
    before = _edges(target) if target is not None else {}
    ctx: dict[str, Any] = {"min_frames": 1, "pending_anchors": []}
    heads = curve_heads(target)  # lane w6-tempo: speed keys count from a clip's first frame
    result = handler(work, args, ctx)
    if target is not None:
        _post_pass(target, before, ctx["pending_anchors"])
        curve_guard(heads, target, op["op"])
    validate_state(work)
    return work, result


def apply_ops(state: dict[str, Any], ops: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply a batch atomically: all verbs or none. A refusal carries ``op_index``."""
    if type(ops) is not list or not 1 <= len(ops) <= 500:
        _bad("ops must be a list of 1..500 ops")
    validate_state(state)
    results = []
    current = state
    for index, op in enumerate(ops):
        try:
            current, result = apply_op(current, op)
        except TimelineError as exc:
            exc.op_index = index
            raise
        results.append(result)
    return current, results


def clamp_op(state: dict[str, Any], op: Any) -> dict[str, Any]:
    """Clamp-first query: how far would this trim/roll/slip go, without applying it."""
    _new, result = apply_op(state, op)
    return {"requested": result.get("requested"), "applied": result.get("applied")}
