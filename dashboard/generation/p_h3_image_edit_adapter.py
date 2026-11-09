"""P-H3-Image edit adapter behind the Suite's edit contract (/image/edit: Fill, Replace, Remove, ...).

Target ``p-h3-image-lora-crop-edit-v1``; the pinned graph is the catalog edit target's ``workflows/p-h3-image-lora-edit-v1`` (PrunaAI's
``p-h3-image_lora_edit.json`` flattened; see its manifest). The Suite's crop arrives at its work size
(4 MP, the card's edit size; ``EDIT_TARGETS`` work_resolution 2048, grid 32), so the compile wires the
crop straight into ``PH3ImageConditioning.images.image_1`` (ImageScaleToTotalPixels 31 is removed: the
crop is already at the size it would produce) and the result is the edited crop at that size, pasted back
by the Suite inside the soft mask like the other edit families.

A ledger edit job changes exactly three declared pointers: the uploaded crop, the instruction and the
noise seed. Distilled, no CFG, no negative prompt, no prompt upsampler (not installed). It samples with
``SamplerCustomAdvanced``, so the KSampler live preview does not apply. Submit, recover, poll and cancel
are the Qwen adapter's, with this target's identity and nodes.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from .image_contract import ContractError, validate_snapshot
from .image_edit_contract import EDIT_TARGETS, PH3_TARGET_ID
from . import qwen_image_edit_adapter as _base
from .qwen_image_edit_adapter import (LIVE_PREVIEW_CLIENT_ID, LIVE_PREVIEW_METHOD,  # noqa: F401 - read by the composition root
                                      QwenEditAdapterError, QwenImageEditAdapter, _equal, get_pointer, set_pointer)

TARGET_ID = PH3_TARGET_ID
# The catalog edit folder this crop edit reads. Its own target id keeps the two routes apart: the composition root
# routes the catalog id to the workflow runtime and this id to this adapter.
ASSET_FOLDER = "p-h3-image-lora-edit-v1"
_TARGET = EDIT_TARGETS[TARGET_ID]
CAPABILITY_SHA256 = _TARGET["capability_manifest_sha256"]
GRAPH_SHA256 = _TARGET["api_graph_sha256"]

MUTABLE_POINTERS = {
    "image": "/5/inputs/image",
    "instruction": "/2/inputs/value",
    "seed": "/18/inputs/noise_seed",
}
OUTPUT_NODE = "99"
SENT_PROMPT_NODE = "90"
SAMPLER_NODE = "20"
SCALE_NODE = "31"
NODE_COUNT = 17
FIXED = {
    "/10/inputs/unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "/13/inputs/lora_name": "p-h3-image_lora_bf16.safetensors",
    "/13/inputs/strength_model": 1.0,
    "/14/inputs/shift_video": 1.0,
    "/14/inputs/shift_audio": 1.0,
    "/11/inputs/clip_name": "p-h3-image_text_encoder_int8_convrot.safetensors",
    "/11/inputs/type": "minimax",
    "/12/inputs/vae_name": "p-h3-image_vae_int8_convrot.safetensors",
    "/17/inputs/sampler_name": "euler",
    "/19/inputs/sigmas": "1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0",
    "/99/class_type": "PreviewImage",
}
REQUIRED_CLASSES = (
    "BasicGuider", "CLIPLoader", "KSamplerSelect", "LoadImage", "LoraLoaderBypassModelOnly", "ManualSigmas",
    "MiniMaxH3SigmaShift", "PH3ImageConditioning", "PreviewAny", "PreviewImage", "PrimitiveStringMultiline",
    "RandomNoise", "SamplerCustomAdvanced", "UNETLoader", "VAEDecode", "VAELoader", "ImageScaleToTotalPixels",
    "LoraLoaderModelOnly",  # not in the graph: the edit route's engine check reads the loras folder's names from it
)

_base.SENT_PROMPT_NODES[TARGET_ID] = SENT_PROMPT_NODE
RUNTIME_ASSET_ROOT = _base._default_asset_root(__file__, ASSET_FOLDER)


def validate_assets(manifest: Mapping[str, Any], graph: Mapping[str, Any]) -> None:
    if manifest.get("schema") != "studio.image-workflow/1" or manifest.get("id") != ASSET_FOLDER \
            or manifest.get("family") != "p_h3_image" or manifest.get("operation") != "edit":
        raise QwenEditAdapterError("edit capability manifest identity changed")
    workflow = manifest.get("workflow") or {}
    if workflow.get("api_sha256") != GRAPH_SHA256 or workflow.get("node_count") != NODE_COUNT or len(graph) != NODE_COUNT:
        raise QwenEditAdapterError("edit workflow fingerprint changed")
    for pointer, expected in FIXED.items():
        if not _equal(get_pointer(graph, pointer), expected):
            raise QwenEditAdapterError(f"pinned edit graph assertion changed: {pointer}")
    for pointer in MUTABLE_POINTERS.values():
        get_pointer(graph, pointer)
    if graph[SCALE_NODE]["inputs"].get("image") != ["5", 0] or graph["15"]["inputs"].get("images.image_1") != [SCALE_NODE, 0] \
            or graph["15"]["inputs"].get("prompt") != ["2", 0] or graph[OUTPUT_NODE]["inputs"] != {"images": ["21", 0]}:
        raise QwenEditAdapterError("pinned edit routes changed")
    for node in graph.values():
        if not isinstance(node, Mapping) or node.get("class_type") not in REQUIRED_CLASSES:
            raise QwenEditAdapterError("pinned edit graph has an undeclared class")


class PH3ImageEditAdapter(QwenImageEditAdapter):
    """Compile and operate one P-H3-Image edit job through an injected client."""

    target_id = TARGET_ID
    capability_sha256 = CAPABILITY_SHA256
    graph_sha256 = GRAPH_SHA256
    output_node = OUTPUT_NODE
    sampler_node = SAMPLER_NODE
    tiny_vae_name = None

    def __init__(self, client: Any, **kwargs: Any) -> None:
        kwargs["kj_preview"] = False  # SamplerCustomAdvanced: no KSampler to put the preview node before
        super().__init__(client, **kwargs)

    def _default_root(self):
        return RUNTIME_ASSET_ROOT

    def load_assets(self) -> tuple[dict[str, Any], dict[str, Any]]:
        manifest = _base._read_pinned(self.asset_root, "manifest.json", CAPABILITY_SHA256, ASSET_FOLDER)
        graph = _base._read_pinned(self.asset_root, "workflow-api.json", GRAPH_SHA256, ASSET_FOLDER)
        validate_assets(manifest, graph)
        return manifest, graph

    def compile_graph(self, request_snapshot: Mapping[str, Any]) -> dict[str, Any]:
        """One validated edit snapshot -> the API graph: the three declared pointers, the crop straight into image_1."""
        try:
            snapshot = copy.deepcopy(dict(request_snapshot))
            validate_snapshot(snapshot)
        except (ContractError, TypeError, ValueError, RecursionError) as exc:
            raise QwenEditAdapterError("invalid edit request snapshot") from exc
        if snapshot.get("operation") != "edit" or snapshot.get("target_id") != TARGET_ID:
            raise QwenEditAdapterError("snapshot is not a P-H3-Image edit")
        _manifest, graph = self.load_assets()
        edit = snapshot["effective_input"]["edit"]
        artist = snapshot["artist_input"]
        compiled = copy.deepcopy(graph)
        set_pointer(compiled, MUTABLE_POINTERS["image"], edit["image"])
        set_pointer(compiled, MUTABLE_POINTERS["instruction"], artist["prompt"])
        set_pointer(compiled, MUTABLE_POINTERS["seed"], artist["seed"])
        # The crop is already at its 4 MP work size: no second resample.
        compiled.pop(SCALE_NODE)
        compiled["15"]["inputs"]["images.image_1"] = ["5", 0]
        return compiled

    compile = compile_graph


__all__ = ["ASSET_FOLDER", "CAPABILITY_SHA256", "GRAPH_SHA256", "MUTABLE_POINTERS", "PH3ImageEditAdapter", "REQUIRED_CLASSES",
           "RUNTIME_ASSET_ROOT", "TARGET_ID"]
