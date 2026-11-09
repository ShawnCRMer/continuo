"""The apply verbs of studio.shot-chain/1. Pure: every effect comes in through ``env``.

env (route-supplied callables):
  probe_guide(path, approved_roots) -> {path, sha256, fps, frames, width, height, label}   (guide_unreadable)
  pull_still(guide, frame) -> {image_job_id, path, sha256}            the full-size guide frame as an owned picture
  kept_picture(image_job_id) -> {image_job_id, path, sha256}           a kept Image result (picture_not_kept)
  check_step(target_id, args) -> args                                  (step_unknown / invalid_step_args)

``apply(chain, bindings, ops, env)`` -> (chain, bindings, results); the input is never changed. A failing op
raises ChainError with its op_index and nothing of the apply is kept.
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Mapping

from . import document as D

VERBS = ("create", "set_guide", "clear_guide", "add_mark", "move_mark", "rename_mark", "remove_mark",
         "bind_picture", "unbind_picture", "set_step")
_ARGS = {
    "create": ({"label"}, {"shot_id"}),
    "set_guide": ({"path", "approved_roots"}, set()),
    "clear_guide": (set(), set()),
    "add_mark": ({"frame", "label"}, set()),
    "move_mark": ({"mark_id", "frame"}, set()),
    "rename_mark": ({"mark_id", "label"}, set()),
    "remove_mark": ({"mark_id"}, set()),
    "bind_picture": ({"mark_id", "image_job_id"}, set()),
    "unbind_picture": ({"mark_id"}, set()),
    "set_step": ({"target_id", "args"}, set()),
}


def _bad(index: int, message: str, code: str = "invalid_op") -> None:
    D.fail(code, f"ops[{index}]: {message}", index)


def check_ops(ops: Any, *, creating: bool) -> list[dict[str, Any]]:
    if not isinstance(ops, list) or not ops:
        D.fail("invalid_op", "ops must be a non-empty list")
    for index, op in enumerate(ops):
        if not isinstance(op, dict) or op.get("op") not in VERBS:
            _bad(index, f"op must be one of {', '.join(VERBS)}")
        need, may = _ARGS[op["op"]]
        names = set(op) - {"op"}
        if need - names:
            _bad(index, f"{op['op']} needs {', '.join(sorted(need - names))}")
        if names - need - may:
            _bad(index, f"{op['op']} does not take {', '.join(sorted(names - need - may))}")
        if op["op"] == "create" and (index != 0 or not creating):
            _bad(index, "create must be the first op of an apply without chain_id")
    if creating and ops[0]["op"] != "create":
        D.fail("invalid_op", "an apply without chain_id starts with create", 0)
    return ops


def _mark(chain: Mapping[str, Any], mark_id: Any, index: int) -> dict[str, Any]:
    for mark in chain["marks"]:
        if mark["mark_id"] == mark_id:
            return mark
    _bad(index, f"no mark {mark_id!r} in this chain", "mark_unknown")
    raise AssertionError


def _label(value: Any, index: int, field: str = "label") -> str:
    if type(value) is not str or not value.strip() or len(value) > D.LABEL_MAX:
        _bad(index, f"{field} must be text of 1 to {D.LABEL_MAX} characters")
    return value.strip()


def _frame(chain: Mapping[str, Any], value: Any, index: int, ignore: str | None = None) -> int:
    if type(value) is not int or value < 0:
        _bad(index, "frame must be a whole number from 0")
    guide = chain["guide"]
    if guide is None:
        _bad(index, "the chain has no guide clip (set_guide)", "no_guide")
    if value > guide["frames"] - 1:
        _bad(index, f"frame {value} is outside the guide (frames 0 to {guide['frames'] - 1})", "frame_outside_guide")
    for mark in chain["marks"]:
        if mark["frame"] == value and mark["mark_id"] != ignore:
            _bad(index, f"mark {mark['label']} is already at frame {value}", "mark_frame_taken")
    return value


def _still(env: Mapping[str, Callable[..., Any]], guide: Mapping[str, Any], frame: int, index: int) -> dict[str, Any]:
    try:
        pulled = env["pull_still"](guide, frame)
    except D.ChainError as exc:
        _bad(index, str(exc), exc.code)
    return {"image_job_id": pulled["image_job_id"], "path": pulled["path"], "sha256": pulled["sha256"],
            "guide_sha256": guide["sha256"], "frame": frame}


def _picture_of_still(still: Mapping[str, Any]) -> dict[str, Any]:
    return {"image_job_id": still["image_job_id"], "path": still["path"], "sha256": still["sha256"], "kind": "still"}


def apply(chain: Mapping[str, Any], bindings: Mapping[str, Any], ops: list[dict[str, Any]],
          env: Mapping[str, Callable[..., Any]]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    chain = copy.deepcopy(dict(chain))
    bindings = copy.deepcopy(dict(bindings))
    results: list[dict[str, Any]] = []
    for index, op in enumerate(ops):
        verb = op["op"]
        result: dict[str, Any] = {"op": verb}
        if verb == "create":
            chain["label"] = _label(op["label"], index)
            shot = op.get("shot_id")
            if shot is not None and (type(shot) is not str or not D.SHOT_ID_RE.fullmatch(shot)):
                _bad(index, "shot_id must be sh- and 8 hex digits")
            chain["shot_id"] = shot
            result["chain_id"] = chain["chain_id"]
        elif verb == "set_guide":
            roots = op["approved_roots"]
            if type(op["path"]) is not str or not op["path"].strip() or not isinstance(roots, list) or not all(type(r) is str for r in roots):
                _bad(index, "set_guide needs path (text) and approved_roots (list of folders)")
            try:
                chain["guide"] = dict(env["probe_guide"](op["path"], roots))
            except D.ChainError as exc:
                _bad(index, str(exc), exc.code)
            result["guide"] = {k: chain["guide"][k] for k in ("media_id", "fps", "frames")}
        elif verb == "clear_guide":
            chain["guide"] = None
        elif verb == "add_mark":
            frame = _frame(chain, op["frame"], index)
            mark_id = D.new_mark_id({m["mark_id"] for m in chain["marks"]})
            if len(chain["marks"]) >= D.MAX_MARKS:
                _bad(index, f"a chain holds at most {D.MAX_MARKS} marks")
            still = _still(env, chain["guide"], frame, index)
            chain["marks"].append({"mark_id": mark_id, "frame": frame, "label": _label(op["label"], index),
                                   "still": still, "picture": _picture_of_still(still)})
            chain["marks"].sort(key=lambda m: m["frame"])
            result.update(mark_id=mark_id, still=still["image_job_id"])
        elif verb == "move_mark":
            mark = _mark(chain, op["mark_id"], index)
            frame = _frame(chain, op["frame"], index, ignore=mark["mark_id"])
            mark["frame"] = frame
            mark["still"] = _still(env, chain["guide"], frame, index)
            if mark["picture"] is None or mark["picture"]["kind"] == "still":
                mark["picture"] = _picture_of_still(mark["still"])
            chain["marks"].sort(key=lambda m: m["frame"])
            result.update(mark_id=mark["mark_id"], still=mark["still"]["image_job_id"])
        elif verb == "rename_mark":
            _mark(chain, op["mark_id"], index)["label"] = _label(op["label"], index)
        elif verb == "remove_mark":
            mark = _mark(chain, op["mark_id"], index)
            chain["marks"] = [m for m in chain["marks"] if m is not mark]
            bindings.pop(mark["mark_id"], None)
        elif verb == "bind_picture":
            mark = _mark(chain, op["mark_id"], index)
            job = op["image_job_id"]
            if type(job) is not str or not D.IMAGE_JOB_RE.fullmatch(job):
                _bad(index, "image_job_id must be imgjob_ and 32 hex digits")
            try:
                picture = env["kept_picture"](job)
            except D.ChainError as exc:
                _bad(index, str(exc), exc.code)
            mark["picture"] = {"image_job_id": picture["image_job_id"], "path": picture["path"],
                               "sha256": picture["sha256"], "kind": "refined"}
            still = mark["still"] or {}
            bindings[mark["mark_id"]] = {"frame": mark["frame"], "guide_sha256": still.get("guide_sha256"),
                                         "still_sha256": still.get("sha256")}
        elif verb == "unbind_picture":
            mark = _mark(chain, op["mark_id"], index)
            mark["picture"] = _picture_of_still(mark["still"]) if mark["still"] else None
            bindings.pop(mark["mark_id"], None)
        elif verb == "set_step":
            if type(op["target_id"]) is not str or not isinstance(op["args"], dict):
                _bad(index, "set_step needs target_id (text) and args (object)")
            try:
                args = env["check_step"](op["target_id"], op["args"])
            except D.ChainError as exc:
                _bad(index, str(exc), exc.code)
            chain["step"] = {"target_id": op["target_id"], "args": args}
        results.append(result)
    return chain, bindings, results
