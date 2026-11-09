"""Krea 2 Edit adapter (w24-krea): Krea 2 General Edit V2 behind the edit contract.

The pinned graphs are the Suite's ``krea2-general-edit-v2`` workflow (comfyui-krea2edit 2.0.0, model card
huggingface.co/conradlocke/krea2-identity-edit, General Edit V2: Turbo 8 steps, CFG 1, ref_boost 4,
grounding 768) reduced to one picture, the Suite's crop.  Two targets, chosen per operation by the render
study (docs/studies/w24-krea-RESEARCH.md):

- ``krea2-edit-kroma-v1``: UNET Kroma (the artist's Turbo model; Krea 2 tensor keys) for Replace, through
  the Suite's latent-mask Replace step (SetLatentNoiseMask + DifferentialDiffusion on KSampler 27).
- ``krea2-edit-turbo-v1``: UNET Krea 2 Turbo for Restyle and Relight (Kroma left the photo unchanged).

A ledger edit job changes exactly three declared pointers - the uploaded crop, the instruction and the
seed - plus, for a Replace snapshot, the declared latent-mask insertion.  No prompt enhancer, no live
preview node (``live_preview: false`` in the capabilities).  Submit, recover, poll and cancel are the
Qwen adapter's, with this target's identity and nodes.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from .image_contract import ContractError, validate_snapshot
from .image_edit_contract import EDIT_TARGETS, KREA_LOOK_TARGET_ID, KREA_REPLACE_TARGET_ID
from . import qwen_image_edit_adapter as _base
from .qwen_image_edit_adapter import (LIVE_PREVIEW_CLIENT_ID, LIVE_PREVIEW_METHOD,  # noqa: F401 - read by the composition root
                                      QwenEditAdapterError, QwenImageEditAdapter, _equal, get_pointer, set_pointer)

TARGET_ID = KREA_REPLACE_TARGET_ID
TARGET_IDS = (KREA_REPLACE_TARGET_ID, KREA_LOOK_TARGET_ID)

MUTABLE_POINTERS = {
    "image": "/5/inputs/image",
    "instruction": "/2/inputs/value",
    "seed": "/27/inputs/seed",
}
OUTPUT_NODE = "99"
SAMPLER_NODE = "27"
NODE_COUNT = 16

REQUIRED_CLASSES = (
    "EmptySD3LatentImage", "GetImageSize", "ImageScaleToTotalPixels", "KSampler", "Krea2EditGroundedEncode",
    "Krea2EditModelPatch", "LoadImage", "LoraLoaderModelOnly", "PreviewImage", "PrimitiveStringMultiline",
    "CLIPLoader", "UNETLoader", "VAEDecode", "VAEEncode", "VAELoader",
    "DifferentialDiffusion", "SetLatentNoiseMask",  # added at compile time (Replace: the latent mask)
)
# The Replace step on this graph: KSampler 27 samples SetLatentNoiseMask(VAEEncode(crop 5), mask = 5's alpha);
# DifferentialDiffusion wraps 27's model (the Krea2EditModelPatch 24 output). Measured in the study (round 2).
LATENT_MASK_POINTERS = {"sampler": "27", "latent_key": "latent_image", "load_node": "5", "vae_node": "12",
                        "model_node": "27", "model_key": "model"}

RUNTIME_ASSET_ROOT = _base._default_asset_root(__file__, KREA_REPLACE_TARGET_ID)


def validate_assets(manifest: Mapping[str, Any], graph: Mapping[str, Any], target_id: str) -> None:
    target = EDIT_TARGETS[target_id]
    if manifest.get("schema") != "studio.image-edit-capability/1" or manifest.get("id") != target_id \
            or manifest.get("family") != target["family"] or manifest.get("variant") != target["variant"]:
        raise QwenEditAdapterError("edit capability manifest identity changed")
    if not _equal(manifest.get("mutable_json_pointers"), MUTABLE_POINTERS):
        raise QwenEditAdapterError("edit mutable pointer declarations changed")
    workflow = manifest.get("workflow") or {}
    if workflow.get("api_sha256") != target["api_graph_sha256"] or workflow.get("node_count") != NODE_COUNT or len(graph) != NODE_COUNT:
        raise QwenEditAdapterError("edit workflow fingerprint changed")
    for pointer, expected in (manifest.get("fixed_assertions") or {}).items():
        if not _equal(get_pointer(graph, pointer), expected):
            raise QwenEditAdapterError(f"pinned edit graph assertion changed: {pointer}")
    files = target["files"]
    for pointer, expected in (("/10/inputs/unet_name", files["diffusion_model"]), ("/11/inputs/clip_name", files["text_encoder"]),
                              ("/12/inputs/vae_name", files["vae"]), ("/13/inputs/lora_name", files["lora"])):
        if get_pointer(graph, pointer) != expected:
            raise QwenEditAdapterError("pinned edit graph files changed")
    if graph.get(OUTPUT_NODE, {}).get("inputs") != {"images": ["28", 0]}:
        raise QwenEditAdapterError("pinned edit outputs changed")
    if graph["27"]["inputs"].get("model") != ["24", 0] or graph["24"]["inputs"].get("source_image") != ["5", 0] \
            or graph["25"]["inputs"].get("image") != ["5", 0] or graph["26"]["inputs"].get("image") != ["5", 0] \
            or graph["20"]["inputs"].get("image") != ["5", 0]:
        raise QwenEditAdapterError("pinned edit routes changed")
    for node in graph.values():
        if not isinstance(node, Mapping) or not isinstance(node.get("class_type"), str) or not isinstance(node.get("inputs"), Mapping):
            raise QwenEditAdapterError("pinned edit graph node shape changed")
        if node["class_type"] not in REQUIRED_CLASSES:
            raise QwenEditAdapterError("pinned edit graph has an undeclared class")


class Krea2EditAdapter(QwenImageEditAdapter):
    """Compile and operate one Krea 2 Edit job (Kroma, Replace) through an injected client."""

    target_id = KREA_REPLACE_TARGET_ID
    capability_sha256 = EDIT_TARGETS[KREA_REPLACE_TARGET_ID]["capability_manifest_sha256"]
    graph_sha256 = EDIT_TARGETS[KREA_REPLACE_TARGET_ID]["api_graph_sha256"]
    output_node = OUTPUT_NODE
    sampler_node = SAMPLER_NODE
    tiny_vae_name = None

    def __init__(self, client: Any, **kwargs: Any) -> None:
        kwargs["kj_preview"] = False  # live_preview: false in the capabilities
        super().__init__(client, **kwargs)

    def _default_root(self):
        return RUNTIME_ASSET_ROOT

    def load_assets(self) -> tuple[dict[str, Any], dict[str, Any]]:
        manifest = _base._read_pinned(self.asset_root, "manifest.json", self.capability_sha256, self.target_id)
        graph = _base._read_pinned(self.asset_root, "workflow-api.json", self.graph_sha256, self.target_id)
        validate_assets(manifest, graph, self.target_id)
        return manifest, graph

    def compile_graph(self, request_snapshot: Mapping[str, Any]) -> dict[str, Any]:
        """One validated edit snapshot -> the API graph: the three declared pointers (+ Replace: the latent mask)."""
        try:
            snapshot = copy.deepcopy(dict(request_snapshot))
            validate_snapshot(snapshot)
        except (ContractError, TypeError, ValueError, RecursionError) as exc:
            raise QwenEditAdapterError("invalid edit request snapshot") from exc
        if snapshot.get("operation") != "edit" or snapshot.get("target_id") != self.target_id:
            raise QwenEditAdapterError("snapshot is not a Krea 2 Edit job of this target")
        _manifest, graph = self.load_assets()
        edit = snapshot["effective_input"]["edit"]
        artist = snapshot["artist_input"]
        compiled = copy.deepcopy(graph)
        set_pointer(compiled, MUTABLE_POINTERS["image"], edit["image"])
        set_pointer(compiled, MUTABLE_POINTERS["instruction"], artist["prompt"])
        set_pointer(compiled, MUTABLE_POINTERS["seed"], artist["seed"])
        if "latent_mask" in edit:
            _base.insert_latent_mask(compiled, **LATENT_MASK_POINTERS)
            compiled[SAMPLER_NODE]["inputs"]["denoise"] = float(snapshot["effective_input"]["denoise"])
        elif edit.get("strength_curve") == "start_v1":  # w24-edit2 (IMG-130): Restyle Strength = the start on the schedule
            _base.insert_strength_start(compiled, snapshot, sampler=SAMPLER_NODE, latent_key="latent_image", load_node="5", vae_node="12")
        return compiled

    compile = compile_graph


class Krea2EditTurboAdapter(Krea2EditAdapter):
    """The same graph on Krea 2 Turbo (Restyle, Relight)."""

    target_id = KREA_LOOK_TARGET_ID
    capability_sha256 = EDIT_TARGETS[KREA_LOOK_TARGET_ID]["capability_manifest_sha256"]
    graph_sha256 = EDIT_TARGETS[KREA_LOOK_TARGET_ID]["api_graph_sha256"]


__all__ = [
    "Krea2EditAdapter", "Krea2EditTurboAdapter", "LATENT_MASK_POINTERS", "MUTABLE_POINTERS", "REQUIRED_CLASSES",
    "RUNTIME_ASSET_ROOT", "TARGET_ID", "TARGET_IDS",
]
