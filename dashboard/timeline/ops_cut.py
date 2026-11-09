"""Range edits over IN-OUT and slide (lane w2-cutdel, Cut A: Resolve Cut style).

Twin of ``desktop/film-suite/src/timeline/ops/timeline-cut.js``; the shared fixtures
``tests/backend/fixtures/timeline/cut.json`` run against both.

- ``lift_range {in_frame, out_frame, tracks?}``: record frames ``[in_frame, out_frame)`` are taken out
  of the chosen tracks (default: every unlocked picture and sound track) and leave a gap. Clips that
  cross IN or OUT are cut there; the parts inside go. On V1 the hole becomes one gap (none at the end of
  the cut: a cut never ends in black); on V2 and A1-A3 it stays empty track time. A sound linked to a
  picture is not cut by itself: it follows its picture (and goes when its picture goes).
- ``extract_range {in_frame, out_frame, tracks?}``: the same, then the hole closes (ripple): later items
  of the chosen tracks move left by its length. When V1 is chosen the cut's free markers ripple too
  (a marker inside the range lands on IN); attached overlays and markers follow their clip as always.
  Extract over a gap alone closes that gap.
- ``slide {clip, delta}``: a storyline clip moves ``delta`` frames between its two neighbours and keeps
  its own source and length; the left neighbour's out point and the right neighbour's in point move
  with it, so the cut keeps its length. Clamped first like roll (the neighbours' handles, one frame
  each at least); the result names the requested and the applied delta.

Locked tracks are skipped by default and refused when named. A range that takes nothing out is
refused with ``nothing_in_range``.
"""
from __future__ import annotations

from typing import Any

from . import ops as _ops
from .schema import fail

STATE_OPS: tuple[str, ...] = ()


def _tracks(timeline: dict[str, Any], args: dict[str, Any]) -> list[dict[str, Any]]:
    if "tracks" in args:
        tracks = [_ops._unlocked(_ops._track(timeline, ident)) for ident in _ops._id_list(args["tracks"], "tracks")]
        for track in tracks:
            if track["kind"] not in ("video", "audio"):
                _ops._bad("a range edit cuts picture and sound tracks")
        return tracks
    return [track for track in _ops._item_tracks(timeline) if not track["locked"]]


def _blade(state: dict[str, Any], timeline: dict[str, Any], track: dict[str, Any], at: int, reserved: set[str]) -> None:
    starts = _ops._starts(track)
    for index, item in enumerate(track["items"]):
        if item.get("link") is not None:
            continue  # a linked sound is cut with its picture
        if starts[index] < at < starts[index] + item["duration_frames"]:
            _ops._split_item(state, timeline, track, index, at, reserved)
            return


def _range_edit(state: dict[str, Any], args: dict[str, Any], *, ripple: bool) -> dict[str, Any]:
    _ops._args(args, ("in_frame", "out_frame"), ("tracks",))
    timeline = _ops._timeline(state, args)
    lo = _ops._int(args["in_frame"], "in_frame", 0)
    hi = _ops._int(args["out_frame"], "out_frame", lo + 1)
    tracks = _tracks(timeline, args)
    story = _ops._storyline(timeline)
    existing = {item["id"] for track in _ops._item_tracks(timeline) for item in track["items"]}
    reserved: set[str] = set()
    removed: list[str] = []
    cut_clip = cut_any = False
    for track in tracks:
        top = min(hi, _ops._end(track)) if track is story else hi
        if lo >= top:
            continue
        _blade(state, timeline, track, lo, reserved)
        _blade(state, timeline, track, top, reserved)
        kept: list[dict[str, Any]] = []
        hole_at = None
        for start, item in zip(_ops._starts(track), track["items"]):
            if lo <= start and start + item["duration_frames"] <= top and item.get("link") is None:
                cut_any = True
                if item["type"] == "clip":
                    cut_clip = True
                    if item["id"] in existing:
                        removed.append(item["id"])
                if hole_at is None:
                    hole_at = len(kept)
                continue
            if ripple and _ops._free(track) and start >= top:
                item["record_start_frame"] -= hi - lo
            kept.append(item)
        if track is story and not ripple and hole_at is not None and hole_at < len(kept):
            gap = _ops.new_id(state, "g", reserved)
            reserved.add(gap)
            kept.insert(hole_at, {"id": gap, "type": "gap", "duration_frames": top - lo})
        track["items"] = kept
    if not (cut_any if ripple else cut_clip):
        fail("nothing_in_range", f"no clip lies in frames {lo}..{hi - 1} of the chosen tracks")
    # Items the range moved stay where it put them: an attached item of a chosen track is re-anchored
    # to its clip (the post-pass then keeps it there) and an item whose clip went is released.
    edges = _ops._edges(timeline)
    for track in tracks:
        if not _ops._free(track):
            continue
        for item in track["items"]:
            attach = item.get("attach")
            if attach is not None:
                if attach["clip"] in edges:
                    attach["offset_frames"] = item["record_start_frame"] - edges[attach["clip"]][0]
                else:
                    item["attach"] = None
    if ripple and story in tracks:
        for marker in timeline["markers"]:
            if marker.get("attach_clip") is not None:
                if marker["attach_clip"] in edges:
                    continue  # an attached marker follows its clip (post-pass)
                marker["attach_clip"] = None
            if marker["frame"] >= hi:
                marker["frame"] -= hi - lo
            elif marker["frame"] >= lo:
                marker["frame"] = lo
    created = [item["id"] for track in tracks for item in track["items"]
               if item["type"] == "clip" and item["id"] not in existing and item.get("link") is None]
    return {"op": args["op"], "removed": removed, "created": created, "range": {"in_frame": lo, "out_frame": hi}}


def op_lift_range(state, args, ctx):
    return _range_edit(state, args, ripple=False)


def op_extract_range(state, args, ctx):
    return _range_edit(state, args, ripple=True)


def op_slide(state, args, ctx):
    _ops._args(args, ("clip", "delta"))
    timeline = _ops._timeline(state, args)
    track, index, item = _ops._picture(timeline, args["clip"])
    if track["role"] != "storyline":
        _ops._bad("slide moves a storyline clip (V1) between its neighbours")
    _ops._unlocked(track)
    if index == 0 or index + 1 >= len(track["items"]):
        fail("not_adjacent", f"{item['id']} needs a neighbour on both sides to slide")
    left, right = track["items"][index - 1], track["items"][index + 1]
    _ops._not_linked(left)
    _ops._not_linked(right)
    delta = _ops._int(args["delta"], "delta")
    rate = timeline["rate"]
    lo = max(-(left["duration_frames"] - ctx["min_frames"]), -_ops._head(right, rate))
    hi = min(right["duration_frames"] - ctx["min_frames"], _ops._tail(left, rate))
    applied = _ops._clamp(delta, lo, hi)
    if applied:
        for neighbour, edge in ((left, "out"), (right, "in")):
            if neighbour["type"] == "clip":
                neighbour["anchors"][edge] = None  # a moved edge leaves its beat
    _ops._shift_out(left, applied, rate)
    _ops._shift_in(right, applied, rate)
    return {"op": "slide", "requested": delta, "applied": applied}


def picture_end(timeline: dict[str, Any]) -> int:
    """The record frame where the last picture clip (V1, V2, ...) ends; 0 without one."""
    end = 0
    for track in _ops._video_tracks(timeline):
        for start, item in zip(_ops._starts(track), track["items"]):
            if item["type"] == "clip":
                end = max(end, start + item["duration_frames"])
    return end


def op_trim_to_picture(state, args, ctx):
    """Lane p6-cut (cut decision 13): the song ends where the picture ends. The song (the legacy M1 lane or the
    song clips of the music track) is cut at the last picture frame: a song clip crossing it loses its tail (a fade
    out longer than what is left shrinks to fit), a song clip that starts at or after it goes. Refused with
    ``no_music_lane`` without a song, ``no_picture`` without a picture clip and ``nothing_to_trim`` when the song
    already ends with the picture."""
    _ops._args(args, ())
    timeline = _ops._timeline(state, args)
    end = picture_end(timeline)
    lane = next((t for t in timeline["tracks"] if t["kind"] == "music"), None)
    legacy = lane is not None and lane.get("music") is not None
    songs = _ops._song_items(timeline)
    if not legacy and not songs:
        fail("no_music_lane", "no song is placed on the timeline")
    if end == 0:
        fail("no_picture", "the cut has no picture clip to trim the song to")
    trimmed: list[str] = []
    removed: list[str] = []
    cut = 0
    if legacy:
        music = lane["music"]
        over = music["record_start_frame"] + music["duration_frames"] - end
        if over > 0:
            if music["record_start_frame"] >= end:
                _ops._bad("the song starts after the picture ends")
            _ops._unlocked(lane)
            music["duration_frames"] -= over
            cut += over
            trimmed.append(lane["id"])
    for track, item in songs:
        start = item["record_start_frame"]
        over = start + item["duration_frames"] - end
        if over <= 0:
            continue
        _ops._unlocked(track)
        if start >= end:
            track["items"] = [i for i in track["items"] if i is not item]
            removed.append(item["id"])
            cut += item["duration_frames"]
            continue
        item["duration_frames"] -= over
        item["fade_in_frames"] = min(item["fade_in_frames"], item["duration_frames"])
        item["fade_out_frames"] = min(item["fade_out_frames"], item["duration_frames"] - item["fade_in_frames"])
        cut += over
        trimmed.append(item["id"])
    if not cut:
        fail("nothing_to_trim", "the song already ends with the picture")
    return {"op": "trim_to_picture", "picture_end": end, "trimmed": trimmed, "removed": removed, "frames": cut}


CUT_OPS = {"lift_range": op_lift_range, "extract_range": op_extract_range, "slide": op_slide,
           "trim_to_picture": op_trim_to_picture}
