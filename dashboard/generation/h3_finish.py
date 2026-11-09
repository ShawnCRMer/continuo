"""MiniMax H3 finishing as its own operation on a finished take (lane w1-h3).

A finished local H3 take that carries its ``.mctx.safetensors`` sidecar is
finished from its stored AV latent (no VAE round trip before the upscale):

* **upscale** (catalog stage ``upscale``, method ``lbh``): the Minimax H3 latent
  upscaler (3D) to 1.5 MP, aligned to 32 px, exactly the artist's RC2 node
  ``1332 LBH upscale`` (``MinimaxH3LatentUpscaler3D`` megapixels 1.5 / align 32 /
  temporal chunking on / fp16), then the take's own video VAE decodes it;
* **rife** (catalog stage ``rife``): RIFE v4.26 x2 (RC2 ``1572``/``1573``), 24 -> 48 fps.

Each switch is on/off; at least one is on.  The graph is pinned
(``workflows/minimax-h3-finish-v1``) and nodes of a switched-off stage are
removed structurally, like the conditioning targets.  The decoded frames are
cut to the take's delivered frames (``pinned_head_frames`` .. + ``delivered_frames``)
and the audio is decoded from the same stored latent and cut alike, so a
continuation take is finished exactly as it was delivered.  The result is a new
take linked to its source (``finish`` take facts); the source pair is staged
read-only into ComfyUI/output and never changed.  Temporal 2x/3x is not part of
this operation (the catalog marks it ``needs_padding_fix``).

Standalone module (no package-relative imports, no dataclasses).
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

TARGET_ID = "minimax-h3-finish-v1"
PROVIDER_ID = "comfyui-local"
UPSCALE_MEGAPIXELS = 1.5
UPSCALE_ALIGN = 32
LATENT_DOWNSAMPLE = 16  # the upscaler's VAE_DOWNSAMPLE
RIFE_MULTIPLIER = 2
FPS = 24
ARG_KEYS = frozenset({"take", "upscale", "rife"})
_TAKE_ID = re.compile(r"studio:[0-9a-f]{32}")
_PINS = {"manifest_sha256": "9fbecb0cd75aceef79179cdee22c58a017cd1d8e0ab2658a63c46a6be4ade369",
         "api_sha256": "acda458f2f9159317afe48a5ecb728658f84faed8d7cdcdb049c9b7eac016a3d", "node_count": 14}
# Graph nodes the compile writes or removes.
_LOAD, _SPLIT, _UPSCALE, _JOIN, _DECODE, _CUT, _CUT_AUDIO = "40", "41", "42", "43", "15", "44", "45"
_RIFE_MODEL, _RIFE, _VIDEO = "50", "51", "18"


class H3FinishError(ValueError):
    """Refused before any side effect; ``reason_code`` says why."""

    def __init__(self, message: str, reason_code: str = "invalid_request") -> None:
        super().__init__(message)
        self.reason_code = reason_code


class FinishProfile:
    """Validated pinned profile; every read returns a fresh copy."""

    __slots__ = ("target_id", "_manifest_json", "_graph_json")

    def __init__(self, manifest: Mapping[str, Any], graph: Mapping[str, Any]) -> None:
        self.target_id = TARGET_ID
        self._manifest_json = json.dumps(manifest, ensure_ascii=False)
        self._graph_json = json.dumps(graph, ensure_ascii=False)

    @property
    def id(self) -> str:
        return self.target_id

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(self._manifest_json)

    @property
    def graph(self) -> dict[str, Any]:
        return json.loads(self._graph_json)


def _resolve(graph: Any, pointer: str) -> Any:
    node = graph
    for part in pointer.strip("/").split("/"):
        if not isinstance(node, dict) or part not in node:
            raise H3FinishError(f"graph is missing pointer {pointer}", "workflow_drift")
        node = node[part]
    return node


def _is_link(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and type(value[1]) is int


def validate_pinned_graph(graph: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    for pointer, expected in manifest["fixed_assertions"].items():
        actual = _resolve(graph, pointer)
        if actual != expected or type(actual) is not type(expected):
            raise H3FinishError(f"fixed assertion differs at {pointer}", "workflow_drift")
    for name, model in manifest["models"].items():
        if _resolve(graph, model["assertion_pointer"]) != model["filename"]:
            raise H3FinishError(f"model {name} differs from its pinned file", "workflow_drift")
    for pointer in manifest["mutable_json_pointers"]:
        if _is_link(_resolve(graph, pointer)):
            raise H3FinishError(f"mutable pointer {pointer} names a link", "workflow_drift")
    if sorted({node["class_type"] for node in graph.values()}) != sorted(manifest["required_classes"]):
        raise H3FinishError("graph classes differ from required_classes", "workflow_drift")
    for node_id, node in graph.items():
        for value in node["inputs"].values():
            if _is_link(value) and value[0] not in graph:
                raise H3FinishError(f"node {node_id} links to a missing node", "workflow_drift")


def load_profile(workflow_root: Path | str, *, pins: Mapping[str, Any] | None = None) -> FinishProfile:
    pins = pins or _PINS
    directory = Path(workflow_root) / TARGET_ID
    manifest_raw = (directory / "manifest.json").read_bytes()
    api_raw = (directory / "workflow-api.json").read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != pins["manifest_sha256"]:
        raise H3FinishError("workflow manifest bytes do not match the pinned SHA-256", "workflow_drift")
    if hashlib.sha256(api_raw).hexdigest() != pins["api_sha256"]:
        raise H3FinishError("workflow API bytes do not match the pinned SHA-256", "workflow_drift")
    manifest, graph = json.loads(manifest_raw), json.loads(api_raw)
    if manifest.get("id") != TARGET_ID or manifest.get("provider_id") != PROVIDER_ID:
        raise H3FinishError("workflow manifest names another target", "workflow_drift")
    if manifest["workflow"]["api_sha256"] != pins["api_sha256"] or len(graph) != pins["node_count"]:
        raise H3FinishError("workflow manifest declares another graph", "workflow_drift")
    validate_pinned_graph(graph, manifest)
    return FinishProfile(manifest, graph)


# --------------------------------------------------------------------------- request


def validate_args(args: Mapping[str, Any]) -> dict[str, Any]:
    """``{take: <studio job id>, upscale: bool, rife: bool}``; at least one switch on.

    The source is named by its job id only: the backend reads the file it
    published for that job, so no path ever enters the request.
    """
    if not isinstance(args, Mapping):
        raise H3FinishError("args must be an object")
    if set(args) != ARG_KEYS:
        raise H3FinishError("finishing takes exactly take, upscale and rife")
    take = args["take"]
    if type(take) is not str or not _TAKE_ID.fullmatch(take):
        raise H3FinishError("take must be a finished take's studio job id")
    if type(args["upscale"]) is not bool or type(args["rife"]) is not bool:
        raise H3FinishError("upscale and rife are on or off")
    if not (args["upscale"] or args["rife"]):
        raise H3FinishError("neither upscale nor RIFE 48 fps is on")
    return {"take": take, "upscale": args["upscale"], "rife": args["rife"]}


def identity_args(plan: Mapping[str, Any], source_sha256: str) -> dict[str, Any]:
    """Durable identity: the source take's job id and the content SHA-256 of its mp4, plus both switches."""
    return {"take": plan["take"], "sha256": source_sha256, "upscale": bool(plan["upscale"]), "rife": bool(plan["rife"])}


def upscaled_size(width: int, height: int) -> tuple[int, int]:
    """The upscaler's own target arithmetic (megapixels mode, align 32, latent grid 16)."""
    w_in, h_in = width // LATENT_DOWNSAMPLE, height // LATENT_DOWNSAMPLE
    aspect = w_in / h_in
    h_target = (UPSCALE_MEGAPIXELS * 1024 * 1024 / aspect) ** 0.5
    w_target = h_target * aspect
    w_aligned = round(w_target / UPSCALE_ALIGN) * UPSCALE_ALIGN
    h_aligned = round(h_target / UPSCALE_ALIGN) * UPSCALE_ALIGN
    w_out = max(1, int((round(w_aligned / LATENT_DOWNSAMPLE) * LATENT_DOWNSAMPLE) // LATENT_DOWNSAMPLE))
    h_out = max(1, int((round(h_aligned / LATENT_DOWNSAMPLE) * LATENT_DOWNSAMPLE) // LATENT_DOWNSAMPLE))
    return w_out * LATENT_DOWNSAMPLE, h_out * LATENT_DOWNSAMPLE


def output_facts(plan: Mapping[str, Any], facts: Mapping[str, Any]) -> dict[str, Any]:
    """What the finished take will be: canvas, frame count and fps."""
    width, height = int(facts["width"]), int(facts["height"])
    delivered = int(facts["delivered_frames"])
    if plan["upscale"]:
        width, height = upscaled_size(width, height)
    frames = (delivered - 1) * RIFE_MULTIPLIER + 1 if plan["rife"] else delivered
    return {"canvas": {"width": width, "height": height}, "frames": frames, "fps": FPS * (RIFE_MULTIPLIER if plan["rife"] else 1)}


def compile_request(*, graph: Mapping[str, Any], manifest: Mapping[str, Any], plan: Mapping[str, Any],
                    facts: Mapping[str, Any], staged: str) -> dict[str, Any]:
    """Clone the pinned graph, point it at the staged take, remove the switched-off stage."""
    validate_pinned_graph(graph, manifest)
    if type(staged) is not str or not staged.endswith(".mp4") or ".." in staged:
        raise H3FinishError("the staged take does not match the request")
    head, delivered = int(facts.get("pinned_head_frames", 0)), int(facts["delivered_frames"])
    if delivered < 2:
        raise H3FinishError("this take is too short to finish")
    g = copy.deepcopy(dict(graph))
    g[_LOAD]["inputs"]["clip"] = staged
    g[_CUT]["inputs"]["batch_index"] = head
    g[_CUT]["inputs"]["length"] = delivered
    g[_CUT_AUDIO]["inputs"]["start_index"] = round(head / FPS, 6)
    g[_CUT_AUDIO]["inputs"]["duration"] = round(delivered / FPS, 6)
    if not plan["upscale"]:
        for node_id in (_SPLIT, _UPSCALE, _JOIN):
            g.pop(node_id)
        g[_DECODE]["inputs"]["samples"] = [_LOAD, 2]
    if not plan["rife"]:
        for node_id in (_RIFE_MODEL, _RIFE):
            g.pop(node_id)
        g[_VIDEO]["inputs"]["images"] = [_CUT, 0]
        g[_VIDEO]["inputs"]["fps"] = float(FPS)
    required = set(manifest["required_classes"])
    for node_id, node in g.items():
        if node["class_type"] not in required:
            raise H3FinishError(f"node {node_id} class is not pinned")
        for value in node["inputs"].values():
            if _is_link(value) and value[0] not in g:
                raise H3FinishError(f"node {node_id} input points at a removed node")
            if isinstance(value, str) and value.startswith("__"):
                raise H3FinishError(f"node {node_id} still carries a placeholder")
    out = output_facts(plan, facts)
    return {
        "graph": g,
        "effective_frames": out["frames"],
        "canvas": out["canvas"],
        "media": [{"role": "source", "key": "source/0", "sha256": facts["self_id"]}],
        "prompt_sent": None,
        "finish": {"relation": "finishes", "source": facts["self_id"], "upscale": plan["upscale"], "rife": plan["rife"],
                   "upscale_method": "lbh" if plan["upscale"] else None,
                   "megapixels": UPSCALE_MEGAPIXELS if plan["upscale"] else None,
                   "fps": out["fps"], "frames": out["frames"], "canvas": out["canvas"],
                   "source_canvas": {"width": int(facts["width"]), "height": int(facts["height"])},
                   "source_frames": delivered},
    }


__all__ = [
    "ARG_KEYS",
    "FinishProfile",
    "H3FinishError",
    "TARGET_ID",
    "compile_request",
    "identity_args",
    "load_profile",
    "output_facts",
    "upscaled_size",
    "validate_args",
    "validate_pinned_graph",
]
