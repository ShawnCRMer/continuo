"""Continuation verb: ``place_with_parent`` (lane fs-obvb).

Twin of ``desktop/film-suite/src/timeline/ops/timeline-continue.js``; the shared fixtures
``tests/backend/fixtures/timeline/continue.json`` run against both.

A MiniMax H3 continuation take (extend / prepend / bridge) is made from its parent take's
latents, so it joins its parent at frames its ``.mctx`` header names (``seams``, source frames
of the parent, see ``generation/h3_mctx.py``):

* ``extends``  — the parent plays to source frame ``seams.exit`` (exclusive), the new take from
  its frame 0: the parent's out point moves to ``exit``, the take goes in right after it;
* ``prepends`` — the new take plays whole, then the parent from source frame ``seams.entry``:
  the parent's in point moves to ``entry`` (its out point stays), the take goes in right before;
* ``bridges``  — the first parent exits at ``seams.exit``, the take plays whole, the second
  parent (``next``) enters at ``seams.entry``; ``next`` moves to right after the take.

One verb, one undo step, on the storyline (magnetic: later clips ripple; attached overlays,
markers and linked sounds follow in the post-pass). The new clip copies the parent's ``audio``
so the sound runs on across the join. The result names the record frame of every join.
"""
from __future__ import annotations

from typing import Any

from . import ops as _ops

RELATIONS = ("extends", "prepends", "bridges")


def _seam(seams: dict[str, Any], key: str) -> int:
    if key not in seams:
        _ops._bad(f"place_with_parent: seams.{key} is required for this relation")
    return _ops._int(seams[key], f"seams.{key}", 0)


def _storyline_clip(timeline: dict[str, Any], item_id: Any) -> tuple[dict[str, Any], int, dict[str, Any]]:
    track, index, item = _ops._picture(timeline, item_id)
    if track["role"] != "storyline":
        _ops._bad(f"{item_id} is not on the storyline")
    if not _ops._moving(item):
        _ops._bad(f"{item_id} is a still; only a take has a continuation")
    return track, index, item


def _source_out(item: dict[str, Any], rate: dict[str, int]) -> int:
    """Source frame just past the clip's out point."""
    return item["source_in_frame"] + _ops.used_source_frames(item["duration_frames"], item["source_rate"], rate)


def _exit_at(item: dict[str, Any], frame: int, rate: dict[str, int]) -> None:
    """Out point at source frame ``frame`` (exclusive); the in point stays."""
    if not item["source_in_frame"] < frame <= item["media_info"]["frames"]:
        _ops.fail("out_of_range", f"{item['id']}: the join frame {frame} is outside its frames "
                                  f"{item['source_in_frame'] + 1}..{item['media_info']['frames']}")
    duration = _ops._to_rec(frame - item["source_in_frame"], item, rate)
    if duration < 1:
        _ops.fail("out_of_range", f"{item['id']} would be shorter than one frame")
    if duration != item["duration_frames"]:
        item["anchors"]["out"] = None
    item["duration_frames"] = duration


def _enter_at(item: dict[str, Any], frame: int, rate: dict[str, int]) -> None:
    """In point at source frame ``frame``; the out point stays."""
    out = _source_out(item, rate)
    if not 0 <= frame < out:
        _ops.fail("out_of_range", f"{item['id']}: the join frame {frame} is outside its frames 0..{out - 1}")
    duration = _ops._to_rec(out - frame, item, rate)
    if duration < 1:
        _ops.fail("out_of_range", f"{item['id']} would be shorter than one frame")
    if frame != item["source_in_frame"]:
        item["anchors"]["in"] = None
    item["source_in_frame"] = frame
    item["duration_frames"] = duration


def op_place_with_parent(state, args, ctx):
    _ops._args(args, ("clip", "take", "relation", "seams"), ("next",))
    timeline = _ops._timeline(state, args)
    relation = args["relation"]
    if relation not in RELATIONS:
        _ops._bad("relation must be extends, prepends or bridges")
    seams = args["seams"]
    if type(seams) is not dict or any(key not in ("exit", "entry") for key in seams):
        _ops._bad("seams is an object with exit and/or entry (source frames of the parent)")
    if ("next" in args) != (relation == "bridges"):
        _ops._bad("next (the second parent) is given for bridges only")
    rate = timeline["rate"]
    track, index, parent = _storyline_clip(timeline, args["clip"])
    _ops._unlocked(track)
    reserved: set[str] = set()
    take = _ops._make_clip(state, timeline, args["take"], overlay=False, start=0, reserved=reserved)
    if not _ops._moving(take):
        _ops._bad("take: a continuation is a take with frames")
    take["audio"] = args["take"].get("audio", parent["audio"])
    if relation == "extends":
        _exit_at(parent, _seam(seams, "exit"), rate)
        track["items"].insert(index + 1, take)
    elif relation == "prepends":
        _enter_at(parent, _seam(seams, "entry"), rate)
        track["items"].insert(index, take)
    else:
        if args["next"] == args["clip"]:
            _ops._bad("next is the second parent, not the first")
        _next_track, _next_index, second = _storyline_clip(timeline, args["next"])
        exit_frame, entry_frame = _seam(seams, "exit"), _seam(seams, "entry")
        _exit_at(parent, exit_frame, rate)
        _enter_at(second, entry_frame, rate)
        track["items"].remove(second)
        at = track["items"].index(parent) + 1
        track["items"][at:at] = [take, second]
    starts = dict(zip((item["id"] for item in track["items"]), _ops._starts(track)))
    if relation == "prepends":
        joins = [starts[parent["id"]]]
    elif relation == "extends":
        joins = [starts[take["id"]]]
    else:
        joins = [starts[take["id"]], starts[args["next"]]]
    return {"op": "place_with_parent", "created": [take["id"]], "joins": joins}


CONTINUE_OPS = {"place_with_parent": op_place_with_parent}
