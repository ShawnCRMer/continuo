"""Takes keep their Video shot (lane p7-cut item 3, VID-099).

The Video page shoots one shot per project: its takes are the finished jobs of the project's
shoot state (``state/shoot.json`` ``jobs``, newest first, each ``{id: studio:..., savedPath}``).
A take placed into the cut (Video's Add to Cut, Place with its parent, Hermes ``append`` /
``insert`` / ``overwrite`` / ``place_with_parent`` of a ``kind: 'take'`` source) is a clip whose
``source.ref`` is one of those ids; this pass gives it ``source.slot`` = the shot's slot and keeps
that slot's ``takes`` in submission order (Take 1 first, as the Video page numbers them), so the
inspector lists "Takes of this shot" and ``swap_take`` works.

It runs after every apply (one revision, one undo step with the verb) and on every read, so an
existing cut whose take clips have no link gets it when the take's shot is known; nothing is
written by a read (no migration file). A clip that already names a slot is left alone; a take
that is not one of the shot's jobs is left alone. Pure: no I/O, the caller passes the jobs.
"""
from __future__ import annotations

import copy
from typing import Any, Iterable

from .resolver import TAKE_RE

SLOT_ID = "video-shot"
SLOT_NAME = "Video shot"


def shot_takes(jobs: Any) -> list[str]:
    """The shot's finished take ids, oldest first (the Video page lists jobs newest first)."""
    out: list[str] = []
    for job in reversed(jobs if isinstance(jobs, list) else []):
        if not isinstance(job, dict):
            continue
        ident = job.get("id")
        if type(ident) is str and TAKE_RE.fullmatch(ident) and job.get("savedPath") and ident not in out:
            out.append(ident)
    return out


def _take_clips(state: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for timeline in state.get("timelines") or []:
        for track in timeline.get("tracks") or []:
            for item in track.get("items") or []:
                source = item.get("source") if isinstance(item, dict) else None
                if isinstance(source, dict) and source.get("kind") == "take" and type(source.get("ref")) is str:
                    yield item


def wants_link(state: Any) -> bool:
    """True when the shot's jobs are worth reading: an unlinked take clip, or the shot's slot exists."""
    if not isinstance(state, dict):
        return False
    if any(s.get("id") == SLOT_ID for s in state.get("slots") or [] if isinstance(s, dict)):
        return True
    return any(not item["source"].get("slot") for item in _take_clips(state))


def link_takes(state: dict[str, Any], takes: list[str]) -> dict[str, Any]:
    """``state`` with every unlinked take clip of the shot linked to its slot (a copy when anything changes)."""
    if not takes or not isinstance(state, dict):
        return state
    known = set(takes)
    slots = state.get("slots") or []
    slot = next((s for s in slots if isinstance(s, dict) and s.get("id") == SLOT_ID), None)
    hits = [item["source"]["ref"] for item in _take_clips(state) if not item["source"].get("slot") and item["source"]["ref"] in known]
    if not hits and slot is None:
        return state
    # The shot's takes in submission order, then any take a swap added that the shot no longer lists.
    merged = list(takes) + [ref for ref in (slot or {}).get("takes", []) if ref not in known]
    selected = (slot or {}).get("selected")
    if selected not in merged:
        selected = hits[0] if hits else None
    fresh = {"id": SLOT_ID, "name": (slot or {}).get("name") or SLOT_NAME, "shot_id": (slot or {}).get("shot_id"), "takes": merged, "selected": selected}
    if not hits and slot == fresh:
        return state
    out = copy.deepcopy(state)
    for item in _take_clips(out):
        if not item["source"].get("slot") and item["source"]["ref"] in known:
            item["source"]["slot"] = SLOT_ID
    out["slots"] = [s for s in out.get("slots") or [] if not (isinstance(s, dict) and s.get("id") == SLOT_ID)] + [fresh]
    return out
