"""P-H3-Image (PrunaAI LoRA on the MiniMax H3 FL2VA transformer): Create and Edit catalog targets.

Create (``p-h3-image-lora-t2i-v1``) is the shared manifest-driven compile of the catalog create adapter.
Edit (``p-h3-image-lora-edit-v1``) follows PrunaAI's ``p-h3-image_lora_edit.json``: Picture 1 (role
``target``) goes through ImageScaleToTotalPixels (4 MP default, steps of 32) into
``PH3ImageConditioning.images.image_1``; references become ``LoadImage`` 1000+i -> ``images.image_{i+2}``
(Picture 2..8, the card's "up to 8 input images"). The artist prompt is written byte-for-byte; the
references' roles are a separate, visible suffix (``<Picture N> is the <role> reference.``), the same
way the Qwen edit names its roles. The node puts "Edit <Picture 1>. " in front of a short instruction.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any, Mapping

_CREATE = None


def _create():
    global _CREATE
    if _CREATE is None:
        path = Path(__file__).with_name("catalog_create_workflow_adapter.py")
        spec = importlib.util.spec_from_file_location("p_h3_image_create_base", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CREATE = module
    return _CREATE


WorkflowRequestError = _create().WorkflowRequestError
REQUEST_SCHEMA = _create().REQUEST_SCHEMA
FAMILY = "p_h3_image"
CREATE_TARGET = "p-h3-image-lora-t2i-v1"
EDIT_TARGET = "p-h3-image-lora-edit-v1"
MAX_PICTURES = 8  # model card: "edits with up to 8 input images" (Picture 1 + 7 references)

# w24-h3img item 4 (memory). Measured 07.11.2026 on the RTX 5090 (32 GB) under ComfyUI --enable-dynamic-vram
# --vram-headroom 2: 4, 8, 12 and 16 MP all rendered; nvidia-smi peaked at 29.6-29.8 GB in every size (dynamic VRAM
# fills to the headroom and streams the rest of the weights), a cold load took 2.0-2.5 GB of system RAM on top.
# So what decides "fits" is the weight set against graphics memory plus free RAM, like the video upscaler's RAM guard:
# FL2VA int8 20.97 GB + LoRA 1.96 GB + text encoder 5.12 GB + VAE 3.15 GB (model-sources.json bytes) + the measured
# 2.5 GB working set; RAM keeps the video upscaler's 4 GB reserve.
WEIGHT_BYTES = 20_970_379_616 + 1_956_195_360 + 5_120_870_504 + 3_146_996_440
WORKING_BYTES = int(2.5 * 1024 ** 3)
RAM_FLOOR = 4 * 1024 ** 3


def _gb(value: float) -> str:
    return f"{value / 1024 ** 3:.1f} GB"


class MemoryRefusal(WorkflowRequestError):
    """Refused before queueing: the machine cannot hold P-H3-Image; ``reason_code`` names it."""

    reason_code = "p_h3_memory"


def check_memory(stats, megapixels: float) -> None:
    """Refuse when graphics memory plus free RAM above the reserve cannot hold the weights and the working set.

    ``stats`` is ComfyUI's /system_stats; unreadable stats never refuse (the engine check answers later).
    """
    try:
        device = (stats.get("devices") or [])[0]
        vram = int(device["vram_total"])
        ram_free = int(stats["system"]["ram_free"])
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return
    need = WEIGHT_BYTES + WORKING_BYTES
    room = vram + max(ram_free - RAM_FLOOR, 0)
    if need > room:
        raise MemoryRefusal(
            f"P-H3-Image at {megapixels:g} MP needs {_gb(need)} of graphics memory plus free system memory for its "
            f"weights; this machine has {_gb(vram)} graphics memory and {_gb(max(ram_free - RAM_FLOOR, 0))} free above "
            f"the {_gb(RAM_FLOOR)} reserve (close other programs or models, then retry)")


# w28-fix item 3 (IMG-220): the 29 GB weight set streams into graphics memory; another model ComfyUI still holds there
# (the owner's 26 GB H3 video model) made the prefetch fail. Free first when less than the weights' resident part is free.
FREE_FIRST_BYTES = 24 * 1024 ** 3


def free_first(stats) -> bool:
    """True when ComfyUI's /system_stats show the GPU holding another model (less than FREE_FIRST_BYTES free)."""
    try:
        device = (stats.get("devices") or [])[0]
        return int(device["vram_free"]) < FREE_FIRST_BYTES
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return False


# w31-fix4 item 2 (P2 IMG-216): Steps. The card samples 8 steps on ManualSigmas over the checkpoint's own grid ("The 8 steps
# run over the checkpoint's own sampling grid. Pass `sigmas` to use another one.", PrunaAI/P-H3-Image card, Performance).
# The graph has no steps input, so Steps N writes N + 1 sigmas resampled along that grid (linear in step position): 8 is
# the card's string byte for byte, every other count keeps its start (1.0), its end (0.0) and its shape.
CARD_SIGMAS = ("1.0", "0.984753", "0.967905", "0.951688", "0.925372", "0.881923", "0.810083", "0.528718", "0.0")
STEPS_SLOT_CLASS = "ManualSigmas"


def sigmas_for_steps(steps: int) -> str:
    """The ManualSigmas string for ``steps`` steps on the card's grid (8 -> the card's own string)."""
    if type(steps) is not int or steps < 1:
        raise WorkflowRequestError("steps invalid")
    card = len(CARD_SIGMAS) - 1
    out = []
    for k in range(steps + 1):
        at = k * card / steps
        low = int(math.floor(at))
        if at == low:
            out.append(CARD_SIGMAS[low])
            continue
        a, b = float(CARD_SIGMAS[low]), float(CARD_SIGMAS[low + 1])
        out.append(f"{a + (b - a) * (at - low):.6f}".rstrip("0"))
    return ", ".join(out)



def role_suffix(refs: list[Mapping[str, Any]]) -> str:
    """The visible line naming each reference's role (Picture 1 is the edited picture)."""
    lines = [f"<Picture {index + 1}> is the {ref['role']} reference." for index, ref in enumerate(refs) if index > 0]
    return ("\n\n" + " ".join(lines)) if lines else ""


class PH3ImageWorkflowAdapter(_create().CatalogCreateWorkflowAdapter):
    """Create = the shared create compile; Edit = Picture 1 + up to 7 references."""

    def __init__(self, *, asset_root: Path | None = None):
        here = Path(__file__).resolve()
        repository = here.parents[4] / "workflows"
        installed = here.parents[1] / "workflows"
        pinned = repository / EDIT_TARGET / "manifest.json"
        self.asset_root = Path(asset_root) if asset_root is not None else (repository if pinned.is_file() else installed)

    def _entry(self, target_id: str) -> tuple[str, dict]:
        catalog = _create()._shared().load_catalog_module().catalog_view()  # shared, read-only
        for family in catalog["families"]:
            if family["id"] != FAMILY:
                continue
            for variant in family["variants"]:
                workflow = variant.get("workflow")
                if workflow and workflow["descriptor"]["target_id"] == target_id:
                    return family["id"], workflow
        raise WorkflowRequestError("workflow target not offered")

    def load_assets(self, target_id: str) -> tuple[dict, dict]:
        if target_id == CREATE_TARGET:
            return super().load_assets(target_id)
        family, entry = self._entry(target_id)
        shared = _create()._shared()
        folder = self.asset_root / target_id
        manifest = shared._read_asset(folder, "manifest.json", entry["manifest_sha256"])
        graph = shared._read_asset(folder, "workflow-api.json", entry["api_sha256"])
        if (manifest.get("schema") != "studio.image-workflow/1" or manifest.get("id") != target_id
                or manifest.get("family") != family or manifest.get("operation") != "edit"
                or manifest.get("descriptor") != entry["descriptor"]
                or manifest["workflow"]["api_sha256"] != entry["api_sha256"]
                or manifest["workflow"]["node_count"] != len(graph)
                or manifest["references"]["max"] != MAX_PICTURES):
            raise WorkflowRequestError("workflow identity mismatch")
        return manifest, graph

    def _slot_value(self, slot: Mapping[str, Any], values: dict) -> Any:
        """Steps (IMG-216) writes the card grid resampled to that many steps into ManualSigmas."""
        if slot["id"] == "steps" and slot["class_type"] == STEPS_SLOT_CLASS:
            return sigmas_for_steps(values["steps"])
        return values[slot["id"]]

    def compile(self, request: Mapping[str, Any]) -> dict:
        if type(request) is dict and request.get("target_id") == CREATE_TARGET:
            return super().compile(request)
        shared = _create()._shared()
        if (type(request) is not dict or set(request) != {"schema", "target_id", "prompt", "controls", "references"}
                or request["schema"] != REQUEST_SCHEMA or type(request["target_id"]) is not str):
            raise WorkflowRequestError("request shape invalid")
        manifest, graph = self.load_assets(request["target_id"])
        prompt = request["prompt"]
        if type(prompt) is not str or not prompt.strip() or len(prompt) > 16384 or "\x00" in prompt:
            raise WorkflowRequestError("prompt invalid")
        refs = request["references"]
        if type(refs) is not list or not 1 <= len(refs) <= manifest["references"]["max"]:
            raise WorkflowRequestError("reference count invalid")
        roles = next(r["roles"] for r in manifest["descriptor"]["media_roles"] if r["id"] == "references")
        for index, ref in enumerate(refs):
            if (type(ref) is not dict or set(ref) != {"image", "role"}
                    or type(ref["image"]) is not str or not shared._UPLOAD.fullmatch(ref["image"])
                    or ref["role"] not in (["target"] if index == 0 else roles)):
                raise WorkflowRequestError("reference image or role invalid")
        values = self._values(manifest, request["controls"])
        for slot in manifest["slots"]:
            shared._set_pointer(graph, slot["pointer"], self._slot_value(slot, values))
        spec = manifest["references"]
        graph[spec["target_node"]]["inputs"]["image"] = refs[0]["image"]
        conditioning = graph[spec["conditioning_node"]]["inputs"]
        for index, ref in enumerate(refs[1:]):
            node_id = str(spec["reference_node_base"] + index)
            if node_id in graph:
                raise WorkflowRequestError("reference node collision")
            graph[node_id] = {"class_type": "LoadImage", "inputs": {"image": ref["image"]}}
            conditioning[f"images.image_{index + 2}"] = [node_id, 0]
        graph[manifest["prompt"]["node_id"]]["inputs"]["value"] = prompt + role_suffix(refs)
        return graph

    def memory_check(self, request: Mapping[str, Any], stats: Any) -> None:
        """Generic workflow-submit hook: refuse by name before the ledger when the machine cannot hold the model."""
        manifest, _ = self.load_assets(request["target_id"])
        check_memory(stats, float(self._values(manifest, request["controls"])["megapixels"]))

    def output_size(self, request: dict, references: list[dict]) -> tuple[int, int]:
        if request.get("target_id") == CREATE_TARGET:
            return super().output_size(request, references)
        if not references:
            raise WorkflowRequestError("edit target required")
        manifest, _ = self.load_assets(request["target_id"])
        values = self._values(manifest, request["controls"])
        w, h = references[0]["width"], references[0]["height"]
        step = manifest["size"]["resolution_steps"]
        # ImageScaleToTotalPixels: scale to megapixels * 1024 * 1024, each side rounded to resolution_steps.
        scale = math.sqrt(values["megapixels"] * 1024 * 1024 / (w * h))
        return max(step, round(w * scale / step) * step), max(step, round(h * scale / step) * step)


def create_adapter():
    """Catalog adapter-factory protocol; generic admission has no model branch."""
    return PH3ImageWorkflowAdapter()
