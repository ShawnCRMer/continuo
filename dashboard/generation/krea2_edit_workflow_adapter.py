"""Manifest-driven compiler for the pinned Krea 2 Edit V2 targets (comfyui-krea2edit >= 2.0.0).

Image 1 is required (the picture to edit, the person photo, or the scene in
two-image mode); image 2 is the optional person. With one image the second
loader and every ``*_b`` input are structurally pruned, never fed a
placeholder. The artist prompt is written byte-for-byte. Output size follows
image 1's aspect at the chosen megapixels, the same rounding as the pinned
ImageScaleToTotalPixels node.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any, Mapping

_CREATE = None


def _create():
    """The Create adapter supplies the shared static health / effective-prompt logic."""
    global _CREATE
    if _CREATE is None:
        path = Path(__file__).with_name("catalog_create_workflow_adapter.py")
        spec = importlib.util.spec_from_file_location("krea2_edit_create_base", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CREATE = module
    return _CREATE


WorkflowRequestError = _create().WorkflowRequestError
REQUEST_SCHEMA = _create().REQUEST_SCHEMA
FAMILY = "krea2"
ROLES = ("target", "identity")


class Krea2EditWorkflowAdapter(_create().CatalogCreateWorkflowAdapter):
    """Edit-only, one or two owned references; transport is the generic runtime's."""

    def __init__(self, *, asset_root: Path | None = None):
        here = Path(__file__).resolve()
        repository = here.parents[4] / "workflows"
        installed = here.parents[1] / "workflows"
        pinned = repository / "krea2-identity-edit-v2" / "manifest.json"
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
                or manifest["references"]["max"] != 2):
            raise WorkflowRequestError("workflow identity mismatch")
        return manifest, graph

    def compile(self, request: Mapping[str, Any]) -> dict:
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
        for index, ref in enumerate(refs):
            if (type(ref) is not dict or set(ref) != {"image", "role"}
                    or type(ref["image"]) is not str or not shared._UPLOAD.fullmatch(ref["image"])
                    or ref["role"] != ROLES[index]):
                raise WorkflowRequestError("reference image or role invalid")
        values = self._values(manifest, request["controls"])
        for slot in manifest["slots"]:
            shared._set_pointer(graph, slot["pointer"], values[slot["id"]])
        graph[manifest["prompt"]["node_id"]]["inputs"]["value"] = prompt
        spec = manifest["references"]
        graph[spec["source_node"]]["inputs"]["image"] = refs[0]["image"]
        if len(refs) == 2:
            graph[spec["subject_node"]]["inputs"]["image"] = refs[1]["image"]
        else:
            # Single image: no placeholder second loader, no *_b inputs.
            graph.pop(spec["subject_node"])
            for node_id, name in spec["subject_inputs"]:
                graph[node_id]["inputs"].pop(name)
        return graph

    def output_size(self, request: dict, references: list[dict]) -> tuple[int, int]:
        if not references:
            raise WorkflowRequestError("edit target required")
        manifest, _ = self.load_assets(request["target_id"])
        values = self._values(manifest, request["controls"])
        w, h = references[0]["width"], references[0]["height"]
        step = manifest["size"]["resolution_steps"]
        scale = math.sqrt(values["megapixels"] * 1024 * 1024 / (w * h))
        return max(step, round(w * scale / step) * step), max(step, round(h * scale / step) * step)


def create_adapter():
    """Catalog adapter-factory protocol; generic admission has no model branch."""
    return Krea2EditWorkflowAdapter()
