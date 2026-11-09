"""FLUX.3 Klein box edit adapter (Image Box): boxes with an intent, read by the model itself.

The pinned graph (``workflows/flux3-klein-box-edit-v1``) runs the distilled FLUX.3 Klein
through NKD Klein Presampling (model_family Klein 3) and Postsampling.  The Suite writes
the whole positive itself - the artist's caption, one sentence per box the caption does
not name, and BFL's element list ``{id, from, src_bbox, tgt_bbox, desc}`` with the boxes
on the crop (``image_box.compile_prompt``) - so ``boxes_json`` stays empty and what the
model reads is exactly the instruction the ledger holds.

A ledger job changes exactly three declared pointers - the uploaded crop, the instruction
and the sampler seed.  Submit, recover, poll, cancel and the live preview node are the Qwen
adapter's, with this target's identity and nodes.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from .image_contract import ContractError, validate_snapshot
from .image_edit_contract import EDIT_TARGETS, KLEINBOX_TARGET_ID
from . import qwen_image_edit_adapter as _base
from .qwen_image_edit_adapter import (LIVE_PREVIEW_CLIENT_ID, LIVE_PREVIEW_METHOD, LIVE_PREVIEW_NODE_CLASS,  # noqa: F401 - read by the composition root
                                      QwenEditAdapterError, QwenImageEditAdapter, _equal, get_pointer, set_pointer)

TARGET_ID = KLEINBOX_TARGET_ID
_TARGET = EDIT_TARGETS[TARGET_ID]
CAPABILITY_SHA256 = _TARGET["capability_manifest_sha256"]
GRAPH_SHA256 = _TARGET["api_graph_sha256"]

MUTABLE_POINTERS = {
    "image": "/11/inputs/image",
    "instruction": "/20/inputs/positive",
    "seed": "/7/inputs/seed",
}
OUTPUT_NODE = "9"
SENT_PROMPT_NODE = "52"
SAMPLER_NODE = "7"
NODE_COUNT = 10

REQUIRED_CLASSES = (
    "CLIPLoader", "KSampler", "LoadImage", "NKDKleinPostsampling", "NKDKleinPresampling", "PreviewAny", "PreviewImage",
    "UNETLoader", "VAEDecode", "VAELoader",
)

_base.SENT_PROMPT_NODES[TARGET_ID] = SENT_PROMPT_NODE
RUNTIME_ASSET_ROOT = _base._default_asset_root(__file__, TARGET_ID)


def validate_assets(manifest: Mapping[str, Any], graph: Mapping[str, Any]) -> None:
    if manifest.get("schema") != "studio.image-edit-capability/1" or manifest.get("id") != TARGET_ID \
            or manifest.get("family") != _TARGET["family"] or manifest.get("variant") != _TARGET["variant"]:
        raise QwenEditAdapterError("edit capability manifest identity changed")
    if not _equal(manifest.get("mutable_json_pointers"), MUTABLE_POINTERS):
        raise QwenEditAdapterError("edit mutable pointer declarations changed")
    workflow = manifest.get("workflow") or {}
    if workflow.get("api_sha256") != GRAPH_SHA256 or workflow.get("node_count") != NODE_COUNT or len(graph) != NODE_COUNT:
        raise QwenEditAdapterError("edit workflow fingerprint changed")
    for pointer, expected in (manifest.get("fixed_assertions") or {}).items():
        if not _equal(get_pointer(graph, pointer), expected):
            raise QwenEditAdapterError(f"pinned edit graph assertion changed: {pointer}")
    files = _TARGET["files"]
    for pointer, expected in (("/1/inputs/unet_name", files["diffusion_model"]), ("/2/inputs/clip_name", files["text_encoder"]),
                              ("/3/inputs/vae_name", files["vae"])):
        if get_pointer(graph, pointer) != expected:
            raise QwenEditAdapterError("pinned edit graph files changed")
    if graph.get(OUTPUT_NODE, {}).get("inputs") != {"images": ["21", 0]} or graph.get(SENT_PROMPT_NODE, {}).get("inputs") != {"source": ["20", 7]}:
        raise QwenEditAdapterError("pinned edit outputs changed")
    pre = graph["20"]["inputs"]
    if graph[SAMPLER_NODE]["inputs"].get("model") != ["20", 0] or pre.get("ref_images.ref_0") != ["11", 0] \
            or pre.get("model") != ["1", 0] or graph["21"]["inputs"].get("bundle") != ["20", 4] or "mask" in pre or "boxes" in pre:
        raise QwenEditAdapterError("pinned edit routes changed")
    for node in graph.values():
        if not isinstance(node, Mapping) or not isinstance(node.get("class_type"), str) or not isinstance(node.get("inputs"), Mapping):
            raise QwenEditAdapterError("pinned edit graph node shape changed")
        if node["class_type"] not in REQUIRED_CLASSES:
            raise QwenEditAdapterError("pinned edit graph has an undeclared class")


class Flux3KleinBoxEditAdapter(QwenImageEditAdapter):
    """Compile and operate one FLUX.3 Klein box edit job through an injected client."""

    target_id = TARGET_ID
    capability_sha256 = CAPABILITY_SHA256
    graph_sha256 = GRAPH_SHA256
    output_node = OUTPUT_NODE
    sampler_node = SAMPLER_NODE
    tiny_vae_name = None

    def _default_root(self):
        return RUNTIME_ASSET_ROOT

    def load_assets(self) -> tuple[dict[str, Any], dict[str, Any]]:
        manifest = _base._read_pinned(self.asset_root, "manifest.json", CAPABILITY_SHA256, TARGET_ID)
        graph = _base._read_pinned(self.asset_root, "workflow-api.json", GRAPH_SHA256, TARGET_ID)
        validate_assets(manifest, graph)
        return manifest, graph

    def compile_graph(self, request_snapshot: Mapping[str, Any]) -> dict[str, Any]:
        """One validated box edit snapshot -> the API graph, changing exactly the three declared pointers."""
        try:
            snapshot = copy.deepcopy(dict(request_snapshot))
            validate_snapshot(snapshot)
        except (ContractError, TypeError, ValueError, RecursionError) as exc:
            raise QwenEditAdapterError("invalid edit request snapshot") from exc
        if snapshot.get("operation") != "edit" or snapshot.get("target_id") != TARGET_ID:
            raise QwenEditAdapterError("snapshot is not a Klein box edit")
        _manifest, graph = self.load_assets()
        edit = snapshot["effective_input"]["edit"]
        artist = snapshot["artist_input"]
        compiled = copy.deepcopy(graph)
        set_pointer(compiled, MUTABLE_POINTERS["image"], edit["image"])
        set_pointer(compiled, MUTABLE_POINTERS["instruction"], artist["prompt"])
        set_pointer(compiled, MUTABLE_POINTERS["seed"], artist["seed"])
        return compiled

    compile = compile_graph


__all__ = [
    "CAPABILITY_SHA256", "Flux3KleinBoxEditAdapter", "GRAPH_SHA256", "MUTABLE_POINTERS", "REQUIRED_CLASSES", "RUNTIME_ASSET_ROOT", "TARGET_ID",
]
