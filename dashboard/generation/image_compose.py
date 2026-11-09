"""A picture the Suite composed becomes a picture of the ledger (``operation: "compose"``).

Accept as version (Original + generative layers, flattened) and Expand (a larger
canvas) produce pixels no model run produced.  Kept revisions, edits, masks and
previews are all keyed by a published producer job, so the composed PNG goes
through the same ledger instead of a second material authority: the route
uploads it to the local engine's temp folder and a passthrough job
(core ``LoadImage -> PreviewImage``, measured pixel-identical on the real
engine) publishes it like any other run.  The snapshot carries the PNG's
sha256, size and a small recipe so the job is honest about what it is.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any
from .comfy_failure import failure_reason as _comfy_failure_reason  # lane w24-rest (GEM-056)

SCHEMA = "studio.image-compose/1"
TARGET_ID = "filmsuite-compose-v1"
KINDS = ("accept", "expand")
OUTPUT_NODE = "2"
_RECIPE_MAX = 4096
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_JOB = re.compile(r"^imgjob_[0-9a-f]{32}$")
_UPLOAD = re.compile(r"^filmsuite-compose-[0-9a-f]{32}\.png \[temp\]$")
_KEYS = {"schema", "kind", "source_job_id", "picture_sha256", "image", "recipe"}
ALPHA_EPSILON = 0.002


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def graph(image: str, alpha: tuple[int, int] | None = None) -> dict[str, Any]:
    """The whole workflow: load the uploaded PNG, hand it back unchanged.

    ``alpha`` (width, height): the picture has transparency (Remove background).  Core LoadImage
    gives RGB plus the mask 1 - a; JoinImageWithAlpha puts back 1 - mask, which drifts by one
    level on 95 of the 256 alpha values (float32 rounding, measured on the real engine).  Taking
    0.002 off the mask first (SolidMask + MaskComposite subtract, clamped at 0) lands every level
    exactly: 0 of 256 wrong, colours unchanged (measured 2026-10-19, prompt 03cdc647).
    """
    if alpha is None:
        return {"1": {"class_type": "LoadImage", "inputs": {"image": image}},
                OUTPUT_NODE: {"class_type": "PreviewImage", "inputs": {"images": ["1", 0]}}}
    width, height = alpha
    return {"1": {"class_type": "LoadImage", "inputs": {"image": image}},
            "3": {"class_type": "SolidMask", "inputs": {"value": ALPHA_EPSILON, "width": width, "height": height}},
            "4": {"class_type": "MaskComposite", "inputs": {"destination": ["1", 1], "source": ["3", 0], "x": 0, "y": 0, "operation": "subtract"}},
            "5": {"class_type": "JoinImageWithAlpha", "inputs": {"image": ["1", 0], "alpha": ["4", 0]}},
            OUTPUT_NODE: {"class_type": "PreviewImage", "inputs": {"images": ["5", 0]}}}


MANIFEST = {"schema": SCHEMA, "target_id": TARGET_ID, "classes": ["LoadImage", "PreviewImage"],
            "outputs": {"image": {"node_id": OUTPUT_NODE, "key": "images", "type": "temp"}}}
CAPABILITY_SHA256 = hashlib.sha256(_canonical(MANIFEST)).hexdigest()
GRAPH_SHA256 = hashlib.sha256(_canonical(graph("{image}"))).hexdigest()
# A picture with transparency: the alpha-keeping passthrough (its own graph fingerprint).
GRAPH_ALPHA_SHA256 = hashlib.sha256(_canonical(graph("{image}", (0, 0)))).hexdigest()
ALPHA_CLASSES = ("JoinImageWithAlpha", "LoadImage", "MaskComposite", "PreviewImage", "SolidMask")


class ComposeError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def upload_name(picture_sha256: str) -> str:
    return f"filmsuite-compose-{picture_sha256[:32]}.png [temp]"


def build_snapshot(compose: Mapping[str, Any], width: int, height: int, basis: dict[str, Any],
                   policy: dict[str, Any]) -> dict[str, Any]:
    """The ledger request snapshot of one composed picture (validated, canonical)."""
    e = dict(compose)
    # ``alpha: true`` only on pictures with transparency; every older snapshot keeps the plain shape.
    alpha_bad = "alpha" in e and e["alpha"] is not True
    if set(e) not in (_KEYS, _KEYS | {"alpha"}) or e["schema"] != SCHEMA or e["kind"] not in KINDS or alpha_bad:
        raise ComposeError("invalid_compose", "compose-shape")
    if not isinstance(e["source_job_id"], str) or not _JOB.match(e["source_job_id"]):
        raise ComposeError("invalid_compose", "compose-source")
    if not isinstance(e["picture_sha256"], str) or not _HEX64.match(e["picture_sha256"]):
        raise ComposeError("invalid_compose", "compose-sha256")
    if e["image"] != upload_name(e["picture_sha256"]) or not _UPLOAD.match(e["image"]):
        raise ComposeError("invalid_compose", "compose-image")
    if not isinstance(e["recipe"], dict) or len(_canonical(e["recipe"])) > _RECIPE_MAX:
        raise ComposeError("invalid_compose", "compose-recipe")
    for value in (width, height):
        if isinstance(value, bool) or not isinstance(value, int) or not 16 <= value <= 16_384:
            raise ComposeError("invalid_compose", "compose-size")
    label = "Accepted edit" if e["kind"] == "accept" else "Expanded picture"
    return {
        "operation": "compose",
        "target_id": TARGET_ID,
        "artist_input": {"prompt": label, "seed": 0, "width": width, "height": height},
        "effective_input": {"compose": copy.deepcopy(e), "graph": graph(e["image"], (width, height) if e.get("alpha") else None)},
        "context_basis": copy.deepcopy(basis),
        "attach_policy": copy.deepcopy(policy),
        "capability_manifest_sha256": CAPABILITY_SHA256,
        "api_graph_sha256": GRAPH_ALPHA_SHA256 if e.get("alpha") else GRAPH_SHA256,
    }


def is_compose_snapshot(snapshot: Any) -> bool:
    return isinstance(snapshot, Mapping) and snapshot.get("operation") == "compose" and snapshot.get("target_id") == TARGET_ID


def rebuild_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild from the snapshot's own intent (the ledger compares canonical bytes)."""
    effective, artist = snapshot.get("effective_input"), snapshot.get("artist_input")
    if not isinstance(effective, Mapping) or set(effective) != {"compose", "graph"} or not isinstance(artist, Mapping):
        raise ComposeError("invalid_compose", "compose-snapshot-shape")
    return build_snapshot(effective["compose"], artist.get("width"), artist.get("height"),
                          snapshot["context_basis"], snapshot["attach_policy"])


def png_size(data: bytes) -> tuple[int, int]:
    """Width and height of a PNG from its IHDR (the route decodes it fully before this)."""
    if len(data) < 24 or not data.startswith(b"\x89PNG\r\n\x1a\n") or data[12:16] != b"IHDR":
        raise ComposeError("invalid_picture", "The picture is not a PNG")
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


class ComposeAdapter:
    """Provider seam for compose jobs: submit the passthrough graph, read its one output."""

    target_id = TARGET_ID

    def __init__(self, client: Any) -> None:
        self.client = client

    @staticmethod
    def _identity(job_id: str, client_request_id: str, snapshot_sha: str | None, graph_sha: str = GRAPH_SHA256) -> dict[str, Any]:
        return {"provider_id": "local-comfyui", "image_producer_job_id": job_id, "client_request_id": client_request_id,
                "target_id": TARGET_ID, "request_snapshot_sha256": snapshot_sha,
                "capability_manifest_sha256": CAPABILITY_SHA256, "api_graph_sha256": graph_sha}

    @staticmethod
    def _owned(prompt: Any, pid: str, body_graph: Mapping[str, Any], extra: Mapping[str, Any]) -> bool:
        return (isinstance(prompt, (list, tuple)) and len(prompt) >= 4 and prompt[1] == pid
                and prompt[2] == body_graph and prompt[3] == extra)

    async def submit(self, *, image_producer_job_id: str, request_snapshot: Mapping[str, Any], client_request_id: str,
                     request_snapshot_sha256: str | None = None) -> dict[str, Any]:
        if not is_compose_snapshot(request_snapshot):
            return {"kind": "malformed"}
        try:
            expected = rebuild_snapshot(request_snapshot)
        except (ComposeError, KeyError, TypeError):
            return {"kind": "malformed"}
        if _canonical(expected) != _canonical(dict(request_snapshot)):
            return {"kind": "malformed"}
        extra = {"studio": self._identity(image_producer_job_id, client_request_id, request_snapshot_sha256, expected["api_graph_sha256"])}
        receipt = await self.client.submit(expected["effective_input"]["graph"], client_id=client_request_id, extra_data=extra)
        pid = receipt.get("prompt_id") if isinstance(receipt, Mapping) else None
        if not isinstance(pid, str) or not pid or len(pid) > 256 or receipt.get("node_errors"):
            return {"kind": "malformed"}
        return {"kind": "accepted", "provider_prompt_id": pid}

    async def recover_submit(self, client_request_id: str, **kwargs: Any) -> dict[str, Any]:
        # A passthrough is cheap and its output is content-addressed: an unknown submit stays
        # unresolved (the coordinator quarantines it) rather than guessing an engine prompt.
        return {"kind": "unresolved"}

    recover = recover_submit

    async def poll(self, provider_prompt_id: str, *, expected_identity: Mapping[str, Any]) -> dict[str, Any]:
        history = await self.client.history(provider_prompt_id)
        entry = history.get(provider_prompt_id) if isinstance(history, Mapping) else None
        if entry is not None:
            prompt = entry.get("prompt")
            extra = prompt[3] if isinstance(prompt, (list, tuple)) and len(prompt) >= 4 else None
            if not isinstance(extra, Mapping) or extra.get("studio") != dict(expected_identity):
                return {"kind": "malformed"}
            status = entry.get("status") or {}
            if status.get("status_str") in {"error", "failed"}:
                return {"kind": "failed", "reason": _comfy_failure_reason(entry)}
            if status.get("status_str") == "success" or status.get("completed") is True:
                images = (entry.get("outputs", {}).get(OUTPUT_NODE) or {}).get("images")
                if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], Mapping) \
                        or images[0].get("type") != "temp" or images[0].get("subfolder") != "":
                    return {"kind": "malformed"}
                return {"kind": "ready", "output_identity": {"filename": images[0].get("filename"), "subfolder": "",
                                                             "type": "temp", "mime_type": "image/png", "extension": ".png"}}
        queue = await self.client.queue()
        for key, kind in (("queue_running", "running"), ("queue_pending", "queued")):
            for item in queue.get(key, []) if isinstance(queue, Mapping) else []:
                if isinstance(item, (list, tuple)) and len(item) >= 2 and item[1] == provider_prompt_id:
                    return {"kind": kind}
        return {"kind": "missing"}

    async def cancel(self, provider_prompt_id: str) -> dict[str, Any]:
        return {"kind": "unsupported"}


__all__ = ["ALPHA_CLASSES", "CAPABILITY_SHA256", "ComposeAdapter", "ComposeError", "GRAPH_ALPHA_SHA256", "GRAPH_SHA256", "KINDS", "SCHEMA", "TARGET_ID",
           "build_snapshot", "graph", "is_compose_snapshot", "png_size", "rebuild_snapshot", "upload_name"]
