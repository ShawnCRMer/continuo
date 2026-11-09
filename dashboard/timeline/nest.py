"""Cuts as items (lane w23-bins, CUT-180): a cut placed inside another cut (a nested clip).

Reference (study w23-bins-STUDY.md): DaVinci Resolve's nested timelines and Premiere's nested sequences. A
timeline dragged into a timeline is ONE clip; it is live (edits of the inner cut show in every instance) and
keeps its own length (Resolve: "editing an original timeline does nothing to change the duration of nested
instances ... those areas of the nested timeline simply go black").

Here a nested clip is an ordinary clip whose ``source`` is ``{kind: 'cut', ref: <timeline id>}``; its
``media_info.frames`` is the inner cut's length when it was placed (the clip's own length rule), its
``source_rate`` the inner cut's rate. Rules (refusals carry a reason code, the same for the hand and Hermes):
- ``nest_cycle``: a cut never contains itself, directly or through other cuts;
- ``nest_rate``: the inner cut runs at the outer cut's rate (Resolve conforms; we refuse instead of resampling
  a cut silently);
- ``not_found``: the ref names a cut of this document.
Twin: ``desktop/film-suite/src/timeline/ops/timeline-nest.js``.
"""
from __future__ import annotations

import copy
from typing import Any

from .schema import fail

KIND = "cut"


def _picture_tracks(timeline: dict[str, Any]) -> list[dict[str, Any]]:
    return [track for track in timeline.get("tracks") or () if track.get("kind") == "video"]


def nested_refs(timeline: dict[str, Any]) -> list[str]:
    """The cut ids this cut's picture tracks place directly (first-seen order)."""
    out: list[str] = []
    for track in _picture_tracks(timeline):
        for item in track.get("items") or ():
            source = item.get("source") if item.get("type") == "clip" else None
            if isinstance(source, dict) and source.get("kind") == KIND and source.get("ref") not in out:
                out.append(source.get("ref"))
    return out


def timeline_length(timeline: dict[str, Any]) -> int:
    """The cut's length: the end of its last item on any track (Cut's Length; timeline_export.model.timeline_end)."""
    end = 0
    for track in timeline.get("tracks") or ():
        if track.get("kind") == "music":
            music = track.get("music")
            if isinstance(music, dict):
                end = max(end, int(music.get("record_start_frame") or 0) + int(music.get("duration_frames") or 0))
            continue
        if track.get("kind") == "video" and track.get("role") == "storyline":
            end = max(end, sum(int(i.get("duration_frames") or 0) for i in track.get("items") or ()))
            continue
        for item in track.get("items") or ():
            if item.get("type") == "clip" and type(item.get("record_start_frame")) is int:
                end = max(end, item["record_start_frame"] + int(item.get("duration_frames") or 0))
    return end


def by_id(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {t["id"]: t for t in state.get("timelines") or () if isinstance(t, dict) and isinstance(t.get("id"), str)}


def users(state: dict[str, Any], ident: str) -> list[str]:
    """The cuts that place cut ``ident`` directly (document order)."""
    return [t["id"] for t in state["timelines"] if ident in nested_refs(t)]


def check_state(state: dict[str, Any]) -> None:
    """Every nested clip names a cut of the document at its own rate, and no cut contains itself."""
    cuts = by_id(state)
    for timeline in state["timelines"]:
        for ref in nested_refs(timeline):
            inner = cuts.get(ref)
            if inner is None:
                fail("not_found", f"{timeline['id']}: the nested cut {ref} does not exist")
            if inner["rate"] != timeline["rate"]:
                fail("nest_rate", f"{timeline['id']}: the nested cut {inner['name']} runs at {inner['rate']['num']}/{inner['rate']['den']} fps, "
                                  f"this cut at {timeline['rate']['num']}/{timeline['rate']['den']}")
    # Depth-first over the nest graph; a back edge is a cycle.
    state_of: dict[str, int] = {}

    def visit(ident: str, path: list[str]) -> None:
        state_of[ident] = 1
        for ref in nested_refs(cuts[ident]):
            if state_of.get(ref) == 1:
                names = " > ".join(cuts[i]["name"] for i in [*path, ident, ref])
                fail("nest_cycle", f"a cut cannot contain itself ({names})")
            if state_of.get(ref) is None:
                visit(ref, [*path, ident])
        state_of[ident] = 2

    for ident in cuts:
        if state_of.get(ident) is None:
            visit(ident, [])


def media_info(state: dict[str, Any], ref: Any, outer: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, int]]:
    """``(media_info, source_rate)`` of cut ``ref`` as a nested source; refuses an unknown or empty cut."""
    inner = by_id(state).get(ref) if isinstance(ref, str) else None
    if inner is None:
        fail("not_found", f"the cut {ref} does not exist")
    frames = timeline_length(inner)
    if frames < 1:
        fail("nest_empty", f"the cut {inner['name']} is empty; there is nothing to place")
    if outer is not None and outer["id"] == inner["id"]:
        fail("nest_cycle", f"a cut cannot contain itself ({inner['name']})")
    info = {"frames": frames, "width": inner["format"]["width"], "height": inner["format"]["height"], "has_audio": False}
    return info, copy.deepcopy(inner["rate"])


def op_delete_timeline(state, args, ctx):
    """Delete one cut (Media's Delete, Resolve's Media Pool Delete Timeline). Never the last cut (``last_cut``),
    never a cut another cut places (``cut_in_use``: Resolve leaves black holes; we keep the film whole). When it was
    the active cut, the next cut in the list (else the previous) becomes active."""
    from . import ops
    ops._args(args, ("id",))
    ident = args["id"]
    cuts = by_id(state)
    if ident not in cuts:
        fail("not_found", f"timeline {ident} does not exist")
    if len(state["timelines"]) == 1:
        fail("last_cut", f"{cuts[ident]['name']} is the project's only cut")
    holders = [t for t in users(state, ident) if t != ident]
    if holders:
        fail("cut_in_use", f"{cuts[ident]['name']} is inside " + ", ".join(cuts[t]["name"] for t in holders))
    index = next(i for i, t in enumerate(state["timelines"]) if t["id"] == ident)
    del state["timelines"][index]
    if state["active"] == ident:
        state["active"] = state["timelines"][min(index, len(state["timelines"]) - 1)]["id"]
    return {"op": "delete_timeline", "removed": [ident]}


NEST_OPS = {"delete_timeline": op_delete_timeline}
NEST_STATE_OPS: tuple[str, ...] = ("delete_timeline",)


def _poster(state: dict[str, Any], timeline: dict[str, Any], seen: set[str]) -> dict[str, Any] | None:
    """The first storyline clip's source; a nested cut's first frame is its own first clip (as Resolve shows it)."""
    seen.add(timeline["id"])
    for track in _picture_tracks(timeline):
        if track.get("role") != "storyline":
            continue
        for item in track.get("items") or ():
            if item.get("type") != "clip":
                continue
            source = item["source"]
            if source.get("kind") != KIND:
                return source
            inner = by_id(state).get(source.get("ref"))
            if inner is None or inner["id"] in seen:
                return None
            return _poster(state, inner, seen)
        return None
    return None


def cuts_list(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Media's facts of every cut, in document order (the cut selector's order): Media and Hermes read this."""
    out = []
    for timeline in state["timelines"]:
        frames = timeline_length(timeline)
        rate = timeline["rate"]
        clips = [item for track in _picture_tracks(timeline) for item in track.get("items") or () if item.get("type") == "clip"]
        poster = _poster(state, timeline, set())
        out.append({
            "id": timeline["id"], "name": timeline["name"], "active": timeline["id"] == state["active"],
            "frames": frames, "rate": copy.deepcopy(rate), "seconds": round(frames * rate["den"] / rate["num"], 3),
            "width": timeline["format"]["width"], "height": timeline["format"]["height"], "clips": len(clips),
            "poster": {"kind": poster["kind"], "ref": poster["ref"]} if poster else None,
            "nests": nested_refs(timeline), "nested_in": users(state, timeline["id"]),
        })
    return out

