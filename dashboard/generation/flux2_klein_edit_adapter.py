"""FLUX.2 Klein 9B KV edit adapter: the fast edit model behind the edit contract.

The pinned graph is the official ComfyUI template ``image_flux2_klein_9b_kv_image_edit``
flattened to API form with one reference picture (``workflows/flux2-klein-9b-kv-edit-v1``;
see its manifest for every change).  The Suite's crop goes in through ``ReferenceLatent``
on the prompt and on the zeroed negative; the result is the edited crop at the crop's
size, pasted back by the Suite like Qwen's.

A ledger edit job changes exactly three declared pointers - the uploaded crop, the
instruction and the noise seed.  The template has no prompt upsampler and no negative
prompt (distilled, cfg 1).  It samples with ``SamplerCustomAdvanced``, so the KSampler
live preview does not apply (4 steps, about 2-3 s).  Submit, recover, poll and cancel
are the Qwen adapter's, with this target's identity and nodes.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from .image_contract import ContractError, validate_snapshot
from .image_edit_contract import EDIT_TARGETS, KLEIN2_TARGET_ID
from . import qwen_image_edit_adapter as _base
from .qwen_image_edit_adapter import (LIVE_PREVIEW_CLIENT_ID, LIVE_PREVIEW_METHOD,  # noqa: F401 - read by the composition root
                                      QwenEditAdapterError, QwenImageEditAdapter, _equal, get_pointer, set_pointer)

TARGET_ID = KLEIN2_TARGET_ID
_TARGET = EDIT_TARGETS[TARGET_ID]
CAPABILITY_SHA256 = _TARGET["capability_manifest_sha256"]
GRAPH_SHA256 = _TARGET["api_graph_sha256"]

MUTABLE_POINTERS = {
    "image": "/76/inputs/image",
    "instruction": "/135/inputs/text",
    "seed": "/125/inputs/noise_seed",
}
OUTPUT_NODE = "94"
SAMPLER_NODE = "123"
NODE_COUNT = 19

REQUIRED_CLASSES = (
    "CFGGuider", "CLIPLoader", "CLIPTextEncode", "ConditioningZeroOut", "EmptyFlux2LatentImage", "Flux2Scheduler",
    "FluxKVCache", "GetImageSize", "KSamplerSelect", "LoadImage", "PreviewImage", "RandomNoise", "ReferenceLatent",
    "SamplerCustomAdvanced", "UNETLoader", "VAEDecode", "VAEEncode", "VAELoader",
    "DifferentialDiffusion", "SetLatentNoiseMask", "SplitSigmasDenoise", "SplitSigmas",  # added at compile time (p13-img latent mask)
)
# p13-img: SamplerCustomAdvanced 123's latent was EmptyFlux2LatentImage 129; with the latent mask it is
# SetLatentNoiseMask(VAEEncode(crop)), and DifferentialDiffusion patches the guider's model (138 <- 139).
# Below denoise 1.0 the sigmas go through SplitSigmasDenoise 942 (comfy_extras/nodes_custom_sampler.py 228-250,
# low half) on Flux2Scheduler 137 with the step count raised so the low half keeps the template's 4 steps.
LATENT_MASK_POINTERS = {"sampler": "123", "latent_key": "latent_image", "load_node": "76", "vae_node": "127",
                        "model_node": "138", "model_key": "model"}
SCHEDULER_NODE = "137"

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
    for pointer, expected in (("/126/inputs/unet_name", files["diffusion_model"]), ("/133/inputs/clip_name", files["text_encoder"]),
                              ("/127/inputs/vae_name", files["vae"])):
        if get_pointer(graph, pointer) != expected:
            raise QwenEditAdapterError("pinned edit graph files changed")
    if graph.get(OUTPUT_NODE, {}).get("inputs") != {"images": ["124", 0]}:
        raise QwenEditAdapterError("pinned edit outputs changed")
    if graph["138"]["inputs"].get("model") != ["139", 0] or graph["117"]["inputs"].get("pixels") != ["76", 0] \
            or graph["128"]["inputs"].get("image") != ["76", 0] or graph["123"]["inputs"].get("guider") != ["138", 0]:
        raise QwenEditAdapterError("pinned edit routes changed")
    for node in graph.values():
        if not isinstance(node, Mapping) or not isinstance(node.get("class_type"), str) or not isinstance(node.get("inputs"), Mapping):
            raise QwenEditAdapterError("pinned edit graph node shape changed")
        if node["class_type"] not in REQUIRED_CLASSES:
            raise QwenEditAdapterError("pinned edit graph has an undeclared class")


class Flux2KleinEditAdapter(QwenImageEditAdapter):
    """Compile and operate one FLUX.2 Klein 9B KV edit job through an injected client."""

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
        manifest = _base._read_pinned(self.asset_root, "manifest.json", CAPABILITY_SHA256, TARGET_ID)
        graph = _base._read_pinned(self.asset_root, "workflow-api.json", GRAPH_SHA256, TARGET_ID)
        validate_assets(manifest, graph)
        return manifest, graph

    def compile_graph(self, request_snapshot: Mapping[str, Any]) -> dict[str, Any]:
        """One validated edit snapshot -> the API graph, changing exactly the three declared pointers."""
        try:
            snapshot = copy.deepcopy(dict(request_snapshot))
            validate_snapshot(snapshot)
        except (ContractError, TypeError, ValueError, RecursionError) as exc:
            raise QwenEditAdapterError("invalid edit request snapshot") from exc
        if snapshot.get("operation") != "edit" or snapshot.get("target_id") != TARGET_ID:
            raise QwenEditAdapterError("snapshot is not a FLUX.2 Klein edit")
        _manifest, graph = self.load_assets()
        edit = snapshot["effective_input"]["edit"]
        artist = snapshot["artist_input"]
        compiled = copy.deepcopy(graph)
        set_pointer(compiled, MUTABLE_POINTERS["image"], edit["image"])
        set_pointer(compiled, MUTABLE_POINTERS["instruction"], artist["prompt"])
        set_pointer(compiled, MUTABLE_POINTERS["seed"], artist["seed"])
        if "latent_mask" in edit:
            _base.insert_latent_mask(compiled, **LATENT_MASK_POINTERS)
            denoise = float(edit["latent_mask"]["denoise"])
            if denoise < 1.0:
                if _base.LATENT_SPLIT_NODE_ID in compiled or compiled[SCHEDULER_NODE]["class_type"] != "Flux2Scheduler":
                    raise QwenEditAdapterError("latent mask insertion point changed")
                steps = int(compiled[SCHEDULER_NODE]["inputs"]["steps"])
                compiled[SCHEDULER_NODE]["inputs"]["steps"] = int(round(steps / denoise))
                compiled[_base.LATENT_SPLIT_NODE_ID] = {"class_type": "SplitSigmasDenoise",
                                                        "inputs": {"sigmas": [SCHEDULER_NODE, 0], "denoise": denoise}}
                compiled[SAMPLER_NODE]["inputs"]["sigmas"] = [_base.LATENT_SPLIT_NODE_ID, 1]
        elif edit.get("strength_curve") == "start_v1":  # w24-edit2 (IMG-130): Restyle Strength = the start on the schedule
            _base.insert_strength_start(compiled, snapshot, sampler=SAMPLER_NODE, latent_key="latent_image", load_node="76",
                                        vae_node="127", scheduler=SCHEDULER_NODE)
        return compiled

    compile = compile_graph


__all__ = [
    "CAPABILITY_SHA256", "Flux2KleinEditAdapter", "GRAPH_SHA256", "MUTABLE_POINTERS", "REQUIRED_CLASSES", "RUNTIME_ASSET_ROOT", "TARGET_ID",
]
