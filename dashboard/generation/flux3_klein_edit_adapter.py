"""FLUX.3 Klein (distilled) edit adapter: the second family behind the edit contract.

The pinned graph is the official ComfyUI template ``image_flux3_klein_distilled_image_edit``
flattened to API form (``workflows/flux3-klein-sd-edit-v1``; see its manifest for every
change).  The Suite's crop goes in as ``<ref_image_0>`` through ``ReferenceMultiLatent``;
the result is the edited crop at the crop's size, pasted back by the Suite like Qwen's.

A ledger edit job changes exactly four declared pointers - the uploaded crop, the
instruction, the sampler seed and the prompt-upsampler switch.  Klein has no negative
prompt (distilled, cfg 1: ``ConditioningZeroOut``).  Submit, recover, poll, cancel and
the live preview node are the Qwen adapter's, with this target's identity and nodes.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from .image_contract import ContractError, validate_snapshot
from .image_edit_contract import EDIT_TARGETS, KLEIN_TARGET_ID
from . import qwen_image_edit_adapter as _base
from .qwen_image_edit_adapter import (LIVE_PREVIEW_CLIENT_ID, LIVE_PREVIEW_METHOD, LIVE_PREVIEW_NODE_CLASS,  # noqa: F401 - read by the composition root
                                      QwenEditAdapterError, QwenImageEditAdapter, _equal, get_pointer, set_pointer)

TARGET_ID = KLEIN_TARGET_ID
_TARGET = EDIT_TARGETS[TARGET_ID]
CAPABILITY_SHA256 = _TARGET["capability_manifest_sha256"]
GRAPH_SHA256 = _TARGET["api_graph_sha256"]

MUTABLE_POINTERS = {
    "image": "/11/inputs/image",
    "instruction": "/34/inputs/value",
    "seed": "/7/inputs/seed",
    "prompt_enhancer": "/51/inputs/switch",
}
OUTPUT_NODE = "9"
SENT_PROMPT_NODE = "52"
SAMPLER_NODE = "7"
NODE_COUNT = 16
ENHANCER_SWITCH_NODE = "51"
ENHANCER_SEED_POINTER = "/47/inputs/sampling_mode.seed"

REQUIRED_CLASSES = (
    "CLIPLoader", "CLIPTextEncode", "ComfySwitchNode", "ConditioningZeroOut", "KSampler", "LoadImage",
    "PreviewAny", "PreviewImage", "PrimitiveStringMultiline", "ReferenceMultiLatent", "TextGenerate",
    "UNETLoader", "VAEDecode", "VAELoader",
    "DifferentialDiffusion", "SetLatentNoiseMask", "VAEEncode",  # added at compile time (p13-img latent mask)
)
# p13-img: KSampler 7's latent was ReferenceMultiLatent 17 output 2 = the reference latent x 0
# (comfy_extras/nodes_edit_model.py :51); with the latent mask it is SetLatentNoiseMask(VAEEncode(crop)).
LATENT_MASK_POINTERS = {"sampler": "7", "latent_key": "latent_image", "load_node": "11", "vae_node": "3",
                        "model_node": "7", "model_key": "model", "denoise": "/7/inputs/denoise"}

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
                              ("/3/inputs/vae_name", files["vae"]), ("/46/inputs/clip_name", files["prompt_enhancer"])):
        if get_pointer(graph, pointer) != expected:
            raise QwenEditAdapterError("pinned edit graph files changed")
    if graph.get(OUTPUT_NODE, {}).get("inputs") != {"images": ["8", 0]} or graph.get(SENT_PROMPT_NODE, {}).get("inputs") != {"source": ["51", 0]}:
        raise QwenEditAdapterError("pinned edit outputs changed")
    if graph[SAMPLER_NODE]["inputs"].get("model") != ["1", 0] or graph["4"]["inputs"].get("text") != ["52", 0] \
            or graph["17"]["inputs"].get("images.image0") != ["11", 0]:
        raise QwenEditAdapterError("pinned edit routes changed")
    for node in graph.values():
        if not isinstance(node, Mapping) or not isinstance(node.get("class_type"), str) or not isinstance(node.get("inputs"), Mapping):
            raise QwenEditAdapterError("pinned edit graph node shape changed")
        if node["class_type"] not in REQUIRED_CLASSES:
            raise QwenEditAdapterError("pinned edit graph has an undeclared class")


class Flux3KleinEditAdapter(QwenImageEditAdapter):
    """Compile and operate one FLUX.3 Klein distilled edit job through an injected client."""

    target_id = TARGET_ID
    capability_sha256 = CAPABILITY_SHA256
    graph_sha256 = GRAPH_SHA256
    output_node = OUTPUT_NODE
    sampler_node = SAMPLER_NODE
    tiny_vae_name = None  # no Klein tiny VAE is installed: latent2rgb preview

    def _default_root(self):
        return RUNTIME_ASSET_ROOT

    def load_assets(self) -> tuple[dict[str, Any], dict[str, Any]]:
        manifest = _base._read_pinned(self.asset_root, "manifest.json", CAPABILITY_SHA256, TARGET_ID)
        graph = _base._read_pinned(self.asset_root, "workflow-api.json", GRAPH_SHA256, TARGET_ID)
        validate_assets(manifest, graph)
        return manifest, graph

    def compile_graph(self, request_snapshot: Mapping[str, Any]) -> dict[str, Any]:
        """One validated edit snapshot -> the API graph: the four declared pointers (+ the enhancer guard when on)."""
        try:
            snapshot = copy.deepcopy(dict(request_snapshot))
            validate_snapshot(snapshot)
        except (ContractError, TypeError, ValueError, RecursionError) as exc:
            raise QwenEditAdapterError("invalid edit request snapshot") from exc
        if snapshot.get("operation") != "edit" or snapshot.get("target_id") != TARGET_ID:
            raise QwenEditAdapterError("snapshot is not a Klein edit")
        _manifest, graph = self.load_assets()
        edit = snapshot["effective_input"]["edit"]
        artist = snapshot["artist_input"]
        compiled = copy.deepcopy(graph)
        set_pointer(compiled, MUTABLE_POINTERS["image"], edit["image"])
        set_pointer(compiled, MUTABLE_POINTERS["instruction"], artist["prompt"])
        set_pointer(compiled, MUTABLE_POINTERS["seed"], artist["seed"])
        set_pointer(compiled, MUTABLE_POINTERS["prompt_enhancer"], bool(edit["prompt_enhancer"]))
        if edit["prompt_enhancer"]:  # enhancer guard (see qwen_image_edit_adapter): 51.on_true 47 -> guard, fallback 34
            _base.insert_enhancer_guard(compiled, ENHANCER_SWITCH_NODE)
            set_pointer(compiled, ENHANCER_SEED_POINTER, artist["seed"])  # p13-img (IMG-130): the run's seed, not 0
        if "latent_mask" in edit:
            _base.insert_latent_mask(compiled, **{k: v for k, v in LATENT_MASK_POINTERS.items() if k != "denoise"})
            set_pointer(compiled, LATENT_MASK_POINTERS["denoise"], edit["latent_mask"]["denoise"])
        elif edit.get("strength_curve") == "start_v1":  # w24-edit2 (IMG-130): Restyle Strength = the start on the schedule
            _base.insert_strength_start(compiled, snapshot, sampler="7", latent_key="latent_image", load_node="11", vae_node="3")
        if not edit["prompt_enhancer"]:
            _base.prune_unchosen(compiled, ENHANCER_SWITCH_NODE)  # w24-rest MOD-017: enhancer OFF sends no upsampler node
        return compiled

    compile = compile_graph


__all__ = [
    "CAPABILITY_SHA256", "Flux3KleinEditAdapter", "GRAPH_SHA256", "MUTABLE_POINTERS", "REQUIRED_CLASSES", "RUNTIME_ASSET_ROOT", "TARGET_ID",
]
