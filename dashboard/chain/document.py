"""studio.shot-chain/1: the document, its stored envelope, ids, errors and inputs_sha256.

The public document is exactly the spec shape (chain_id, project_id, shot_id, label, revision, updated_at, by,
guide, marks, step, takes). The stored envelope adds what only the status needs and the spec does not show:
``bindings`` (per mark: the frame, guide and still a refined picture was bound against) and ``take_inputs``
(per take: the inputs it was shot from, so ``why`` can name every difference). Neither is ever answered.
"""
from __future__ import annotations

import hashlib
import json
import re
import secrets
from typing import Any, Mapping

SCHEMA = "studio.shot-chain/1"
ENVELOPE_SCHEMA = "studio.shot-chain-file/1"
REQUESTS_SCHEMA = "studio.shot-chain-requests/1"
MAX_BYTES = 4 * 1024 * 1024
REQUESTS_MAX_BYTES = 4 * 1024 * 1024
KEEP_REVISIONS = 50
KEEP_REQUESTS = 200
MAX_MARKS = 64
MAX_TAKES = 500
LABEL_MAX = 120
REASON_MAX = 200
CHAIN_ID_RE = re.compile(r"^ch-[0-9a-f]{8}$")
MARK_ID_RE = re.compile(r"^mk-[0-9a-f]{8}$")
SHOT_ID_RE = re.compile(r"^sh-[0-9a-f]{8}$")
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
IMAGE_JOB_RE = re.compile(r"^imgjob_[0-9a-f]{32}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")

ERRORS: dict[str, tuple[bool, str]] = {
    "chain_conflict": (True, "The chain changed since this revision."),
    "request_id_reused": (False, "This request_id already applied other ops; a retry repeats the same ops with the same request_id."),
    "invalid_chain": (False, "The named field is invalid; nothing was saved."),
    "invalid_op": (False, "The named argument of the op is invalid; nothing was saved."),
    "chain_unknown": (False, "No chain of the project has this chain_id (continuo_chain_list lists them)."),
    "mark_unknown": (False, "No mark of the chain has this mark_id."),
    "mark_frame_taken": (False, "One frame holds one mark; another mark is on this frame."),
    "frame_outside_guide": (False, "Frame outside the guide (0 to its last frame)."),
    "no_guide": (False, "No guide clip (set_guide sets one)."),
    "guide_unreadable": (False, "Guide not a video inside the approved folders that ffprobe can read."),
    "still_failed": (True, "The guide frame could not become a picture."),
    "picture_not_kept": (False, "Picture not kept (continuo_image_keep keeps it)."),
    "step_unknown": (False, "Step unknown (continuo_video_steps lists the steps)."),
    "invalid_step_args": (False, "The named step argument is invalid."),
    "not_supported": (False, "The step cannot make this shot; the reasons name why."),
    "take_unknown": (False, "No take of this chain has this studio_job_id."),
    "submit_failed": (True, "Nothing was recorded; the same client_request_id stays valid."),
    "measure_unavailable": (False, "The camera measure cannot run on this machine; the reason says why."),
    "chain_unavailable": (True, "The chain could not be read or saved."),
}


class ChainError(ValueError):
    def __init__(self, code: str, message: str, op_index: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.op_index = op_index


def fail(code: str, message: str, op_index: int | None = None) -> None:
    raise ChainError(code, message, op_index)


def error_body(code: str, message: str, op_index: int | None = None) -> dict[str, Any]:
    retryable, next_step = ERRORS.get(code, (False, "The chain was not changed."))
    body: dict[str, Any] = {"code": code, "message": message, "retryable": retryable, "next": next_step}
    if op_index is not None:
        body["op_index"] = op_index
    return body


def new_chain_id() -> str:
    return "ch-" + secrets.token_hex(4)


def new_mark_id(taken: set[str]) -> str:
    while True:
        candidate = "mk-" + secrets.token_hex(4)
        if candidate not in taken:
            return candidate


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def ops_sha256(ops: Any) -> str:
    return hashlib.sha256(canonical(ops)).hexdigest()


def new_chain(chain_id: str, project_id: str, label: str, shot_id: str | None, stamp: str, by: str) -> dict[str, Any]:
    return {"schema": SCHEMA, "chain_id": chain_id, "project_id": project_id, "shot_id": shot_id, "label": label,
            "revision": 0, "updated_at": stamp, "by": by, "guide": None, "marks": [], "step": None, "takes": []}


# --------------------------------------------------------------------------- frames and inputs


def shot_frame(guide: Mapping[str, Any] | None, guide_frame: int, shot_fps: int, offset_frame: int = 0) -> int:
    """A mark's guide frame on the shot clock: offset, then guide fps -> shot fps (equal fps: the same number)."""
    fps = float((guide or {}).get("fps") or shot_fps)
    return int(round((guide_frame - offset_frame) * shot_fps / fps))


def inputs_of(chain: Mapping[str, Any], shot_fps: int) -> dict[str, Any]:
    """What a take is made of: guide sha256, the step and per mark (in frame order) its shot frame and picture."""
    guide = chain.get("guide")
    marks = sorted(chain.get("marks") or (), key=lambda m: m["frame"])
    return {
        "guide_sha256": guide["sha256"] if guide else None,
        "step": chain.get("step"),
        "marks": [[shot_frame(guide, m["frame"], shot_fps), (m.get("picture") or {}).get("sha256")] for m in marks],
    }


def inputs_sha256(chain: Mapping[str, Any], shot_fps: int) -> str:
    return hashlib.sha256(canonical(inputs_of(chain, shot_fps))).hexdigest()


def take_record(chain: Mapping[str, Any], shot_fps: int) -> dict[str, Any]:
    """The private memory of one take's inputs, readable by status: guide, step and every mark."""
    return {"guide_sha256": (chain.get("guide") or {}).get("sha256"), "step": chain.get("step"),
            "marks": [{"mark_id": m["mark_id"], "label": m["label"], "frame": m["frame"],
                       "picture_sha256": (m.get("picture") or {}).get("sha256"),
                       "kind": (m.get("picture") or {}).get("kind")} for m in chain.get("marks") or ()]}


# --------------------------------------------------------------------------- envelope


def envelope(project_id: str, chain: dict[str, Any], *, bindings: dict[str, Any], take_inputs: dict[str, Any],
             reason: str = "") -> dict[str, Any]:
    return {"schema": ENVELOPE_SCHEMA, "project_id": project_id, "chain": chain, "bindings": bindings,
            "take_inputs": take_inputs, "reason": reason}


def payload(value: dict[str, Any]) -> bytes:
    data = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(data) > MAX_BYTES:
        fail("invalid_chain", "the chain document exceeds 4 MiB")
    return data


def _text(value: Any, field: str, *, max_length: int = LABEL_MAX, empty: bool = False) -> str:
    if type(value) is not str or len(value) > max_length or (not empty and not value.strip()):
        fail("invalid_chain", f"{field} must be text of 1 to {max_length} characters")
    return value


def _int(value: Any, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        fail("invalid_chain", f"{field} must be a whole number from {minimum}")
    return value


def check_chain(chain: Any, project_id: str) -> dict[str, Any]:
    """Shape and invariants of a stored/answered document; raises invalid_chain."""
    keys = {"schema", "chain_id", "project_id", "shot_id", "label", "revision", "updated_at", "by", "guide", "marks", "step", "takes"}
    if not isinstance(chain, dict) or set(chain) != keys or chain["schema"] != SCHEMA:
        fail("invalid_chain", "not a studio.shot-chain/1 document")
    if type(chain["chain_id"]) is not str or not CHAIN_ID_RE.fullmatch(chain["chain_id"]):
        fail("invalid_chain", "chain_id must be ch- and 8 hex digits")
    if chain["project_id"] != project_id:
        fail("invalid_chain", "the chain belongs to another project")
    if chain["shot_id"] is not None and (type(chain["shot_id"]) is not str or not SHOT_ID_RE.fullmatch(chain["shot_id"])):
        fail("invalid_chain", "shot_id must be sh- and 8 hex digits or null")
    _text(chain["label"], "label")
    _int(chain["revision"], "revision")
    if chain["by"] not in ("artist", "hermes"):
        fail("invalid_chain", "by must be artist or hermes")
    guide = chain["guide"]
    if guide is not None:
        if not isinstance(guide, dict) or set(guide) != {"path", "media_id", "sha256", "fps", "frames", "width", "height", "label"}:
            fail("invalid_chain", "guide has the wrong fields")
        if type(guide["sha256"]) is not str or not SHA_RE.fullmatch(guide["sha256"]) or guide["media_id"] != "file:" + guide["sha256"]:
            fail("invalid_chain", "guide sha256 / media_id are wrong")
        _int(guide["frames"], "guide.frames", 1)
    marks = chain["marks"]
    if not isinstance(marks, list) or len(marks) > MAX_MARKS:
        fail("invalid_chain", f"marks must be a list of at most {MAX_MARKS}")
    seen_ids, seen_frames = set(), set()
    for mark in marks:
        if not isinstance(mark, dict) or set(mark) != {"mark_id", "frame", "label", "still", "picture"}:
            fail("invalid_chain", "a mark has the wrong fields")
        if type(mark["mark_id"]) is not str or not MARK_ID_RE.fullmatch(mark["mark_id"]) or mark["mark_id"] in seen_ids:
            fail("invalid_chain", "mark_id must be unique mk- and 8 hex digits")
        if _int(mark["frame"], "mark frame") in seen_frames:
            fail("invalid_chain", "two marks on one frame")
        seen_ids.add(mark["mark_id"]); seen_frames.add(mark["frame"])
        _text(mark["label"], "mark label")
        if mark["still"] is not None and set(mark["still"]) != {"image_job_id", "path", "sha256", "guide_sha256", "frame"}:
            fail("invalid_chain", "a still has the wrong fields")
        if mark["picture"] is not None and (set(mark["picture"]) != {"image_job_id", "path", "sha256", "kind"}
                                            or mark["picture"]["kind"] not in ("still", "refined")):
            fail("invalid_chain", "a picture has the wrong fields")
    step = chain["step"]
    if step is not None and (not isinstance(step, dict) or set(step) != {"target_id", "args"} or not isinstance(step["args"], dict)):
        fail("invalid_chain", "step must be {target_id, args} or null")
    takes = chain["takes"]
    if not isinstance(takes, list) or len(takes) > MAX_TAKES:
        fail("invalid_chain", "takes must be a list")
    for take in takes:
        if not isinstance(take, dict) or set(take) != {"studio_job_id", "inputs_sha256", "made_at", "camera"}:
            fail("invalid_chain", "a take has the wrong fields")
    return chain


def check_envelope(value: Any, project_id: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != ENVELOPE_SCHEMA or value.get("project_id") != project_id:
        fail("invalid_chain", "not a chain file of this project")
    if set(value) != {"schema", "project_id", "chain", "bindings", "take_inputs", "reason"}:
        fail("invalid_chain", "the chain file has the wrong fields")
    check_chain(value["chain"], project_id)
    if not isinstance(value["bindings"], dict) or not isinstance(value["take_inputs"], dict):
        fail("invalid_chain", "the chain file's bookkeeping is wrong")
    return value


def requests_payload(project_id: str, chain_id: str, entries: list[dict[str, Any]]) -> bytes:
    data = json.dumps({"schema": REQUESTS_SCHEMA, "project_id": project_id, "chain_id": chain_id,
                       "entries": entries[-KEEP_REQUESTS:]}, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")
    if len(data) > REQUESTS_MAX_BYTES:
        fail("chain_unavailable", "the kept chain requests exceed their size")
    return data


def check_requests(value: Any, project_id: str, chain_id: str) -> list[dict[str, Any]]:
    if (not isinstance(value, dict) or value.get("schema") != REQUESTS_SCHEMA or value.get("project_id") != project_id
            or value.get("chain_id") != chain_id or not isinstance(value.get("entries"), list)):
        fail("invalid_chain", "the kept chain requests are unreadable")
    return value["entries"]
