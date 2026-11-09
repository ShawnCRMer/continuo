"""What needs redoing: the computed ``status`` of a chain (spec §3). Never stored.

``compute(chain, bindings, take_inputs, shot_fps, readiness)``:
  ready / reasons  - why Shoot is not possible now (the route passes the support reasons in ``readiness``)
  stale            - per take whose inputs_sha256 differs from now, every difference in plain words
  marks            - per mark whether the picture it uses no longer matches the mark (frame or guide)
"""
from __future__ import annotations

from typing import Any, Mapping

from . import document as D


def base_reasons(chain: Mapping[str, Any]) -> list[str]:
    reasons = []
    if chain.get("guide") is None:
        reasons.append("The shot has no guide clip.")
    if not chain.get("marks"):
        reasons.append("The shot has no marks.")
    if chain.get("step") is None:
        reasons.append("No video step is set.")
    for mark in chain.get("marks") or ():
        if mark.get("picture") is None:
            reasons.append(f"Mark {mark['label']} has no picture.")
    return reasons


def mark_status(chain: Mapping[str, Any], bindings: Mapping[str, Any], mark: Mapping[str, Any]) -> dict[str, Any]:
    guide = chain.get("guide")
    picture = mark.get("picture") or {}
    why = None
    if guide is None:
        why = "the guide was removed" if mark.get("still") else None
    elif picture.get("kind") == "refined":
        bound = bindings.get(mark["mark_id"]) or {}
        if bound.get("guide_sha256") not in (None, guide["sha256"]):
            why = "the picture was refined from a still of another guide"
        elif bound.get("frame") is not None and bound["frame"] != mark["frame"]:
            why = f"the still was pulled at frame {bound['frame']}; the mark is at {mark['frame']}"
    elif mark.get("still"):
        still = mark["still"]
        if still["guide_sha256"] != guide["sha256"]:
            why = "the still was pulled from another guide"
        elif still["frame"] != mark["frame"]:
            why = f"the still was pulled at frame {still['frame']}; the mark is at {mark['frame']}"
    if why is None and guide is not None and mark["frame"] > guide["frames"] - 1:
        why = f"the mark is past the guide's last frame {guide['frames'] - 1}"
    return {"mark_id": mark["mark_id"], "picture_stale": why is not None, "why": why}


def take_why(before: Mapping[str, Any] | None, chain: Mapping[str, Any]) -> list[str]:
    """Every difference between the inputs a take was shot from and the chain now, in plain words."""
    if before is None:
        return ["the take's inputs were not recorded"]
    why: list[str] = []
    if before.get("guide_sha256") != (chain.get("guide") or {}).get("sha256"):
        why.append("guide changed")
    then = {m["mark_id"]: m for m in before.get("marks") or ()}
    now = {m["mark_id"]: m for m in chain.get("marks") or ()}
    for mark in chain.get("marks") or ():
        old = then.get(mark["mark_id"])
        if old is None:
            why.append(f"mark {mark['label']} added")
            continue
        if old["frame"] != mark["frame"]:
            why.append(f"mark {mark['label']} moved {old['frame']} → {mark['frame']}")
        elif old.get("picture_sha256") != (mark.get("picture") or {}).get("sha256"):
            kind = (mark.get("picture") or {}).get("kind")
            why.append(f"picture of {mark['label']} refined" if kind == "refined" else f"picture of {mark['label']} changed")
    for mark_id, old in then.items():
        if mark_id not in now:
            why.append(f"mark {old['label']} removed")
    if before.get("step") != chain.get("step"):
        why.append("step changed")
    return why


def compute(chain: Mapping[str, Any], bindings: Mapping[str, Any], take_inputs: Mapping[str, Any], shot_fps: int,
            readiness: list[str]) -> dict[str, Any]:
    current = D.inputs_sha256(chain, shot_fps)
    stale = []
    for take in chain.get("takes") or ():
        if take["inputs_sha256"] != current:
            stale.append({"studio_job_id": take["studio_job_id"],
                          "why": take_why(take_inputs.get(take["studio_job_id"]), chain) or ["inputs changed"]})
    reasons = list(readiness)
    return {"ready": not reasons, "reasons": reasons, "stale": stale,
            "marks": [mark_status(chain, bindings, m) for m in chain.get("marks") or ()]}
