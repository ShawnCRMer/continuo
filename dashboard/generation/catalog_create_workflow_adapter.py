"""Manifest-driven text-to-image compiler for pinned catalog Create targets.

One adapter serves every create-only family whose graph needs no reference
uploads (Anima, FLUX.2 Klein T2I, Ideogram 4, MOSS-Image, FLUX.3 Klein). Every
model difference lives in the pinned manifest: typed slots, the prompt node,
the size node, an optional prompt format check and optional pruned branches.
The artist prompt is written byte-for-byte; nothing is rewritten here.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Mapping

_QWEN = None


def _shared():
    """Reuse the pinned-asset/slot helpers of the Qwen boundary (file-path safe)."""
    global _QWEN
    if _QWEN is None:
        path = Path(__file__).with_name("qwen_image_workflow_adapter.py")
        spec = importlib.util.spec_from_file_location("catalog_create_shared", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _QWEN = module
    return _QWEN


WorkflowRequestError = _shared().WorkflowRequestError
REQUEST_SCHEMA = _shared().REQUEST_SCHEMA
FAMILIES = frozenset({"anima", "flux2_klein", "ideogram4", "moss_image", "flux3_klein", "p_h3_image"})


class CatalogCreateWorkflowAdapter:
    """Create-only, zero-reference compilation; transport is the generic runtime's."""

    def __init__(self, *, asset_root: Path | None = None):
        here = Path(__file__).resolve()
        repository = here.parents[4] / "workflows"
        installed = here.parents[1] / "workflows"
        pinned = repository / "anima-turbo-create-v1" / "manifest.json"
        self.asset_root = Path(asset_root) if asset_root is not None else (repository if pinned.is_file() else installed)

    def _entry(self, target_id: str) -> tuple[str, dict]:
        catalog = _shared().load_catalog_module().catalog_view()  # shared, read-only
        for family in catalog["families"]:
            if family["id"] not in FAMILIES:
                continue
            for variant in family["variants"]:
                workflow = variant.get("workflow")
                if workflow and workflow["descriptor"]["target_id"] == target_id:
                    return family["id"], workflow
        raise WorkflowRequestError("workflow target not offered")

    def load_assets(self, target_id: str) -> tuple[dict, dict]:
        family, entry = self._entry(target_id)
        folder = self.asset_root / target_id
        manifest = _shared()._read_asset(folder, "manifest.json", entry["manifest_sha256"])
        graph = _shared()._read_asset(folder, "workflow-api.json", entry["api_sha256"])
        if (manifest.get("schema") != "studio.image-workflow/1" or manifest.get("id") != target_id
                or manifest.get("family") != family or manifest.get("operation") != "create"
                or manifest.get("descriptor") != entry["descriptor"]
                or manifest["workflow"]["api_sha256"] != entry["api_sha256"]
                or manifest["workflow"]["node_count"] != len(graph)
                or manifest["references"]["max"] != 0):
            raise WorkflowRequestError("workflow identity mismatch")
        return manifest, graph

    @staticmethod
    def _values(manifest: dict, controls: Any) -> dict:
        offered = {c["id"]: c for c in manifest["descriptor"]["controls"]}
        if type(controls) is not dict or set(controls) - set(offered):
            raise WorkflowRequestError("controls not offered")
        values = {}
        for name, control in offered.items():
            values[name] = copy.deepcopy(controls.get(name, control["default"]))
            _shared()._validate_control(control, values[name])
        return values

    def _slot_value(self, slot: Mapping[str, Any], values: dict) -> Any:
        """The value a slot writes into the graph: the control's value (a family adapter may derive it)."""
        return values[slot["id"]]

    def compile(self, request: Mapping[str, Any]) -> dict:
        if (type(request) is not dict or set(request) != {"schema", "target_id", "prompt", "controls", "references"}
                or request["schema"] != REQUEST_SCHEMA or type(request["target_id"]) is not str):
            raise WorkflowRequestError("request shape invalid")
        manifest, graph = self.load_assets(request["target_id"])
        prompt = request["prompt"]
        if type(prompt) is not str or not prompt.strip() or len(prompt) > 16384 or "\x00" in prompt:
            raise WorkflowRequestError("prompt invalid")
        if manifest["prompt"].get("format") == "json_object":
            # Ideogram 4 reads a structured caption; prose is refused, not rewritten.
            try:
                parsed = json.loads(prompt)
            except ValueError as exc:
                raise WorkflowRequestError("prompt must be a JSON object") from exc
            if type(parsed) is not dict or not parsed:
                raise WorkflowRequestError("prompt must be a JSON object")
        if request["references"] != []:
            raise WorkflowRequestError("reference count invalid")
        values = self._values(manifest, request["controls"])
        for slot in manifest["slots"]:
            _shared()._set_pointer(graph, slot["pointer"], self._slot_value(slot, values))
        graph[manifest["prompt"]["node_id"]]["inputs"]["value"] = prompt
        for branch in manifest.get("branches", []):
            if not values[branch["control"]]:
                # OFF is structurally pruned: absent weights never block or load.
                graph[branch["switch_node"]]["inputs"].pop("on_true")
                for node_id in branch["nodes"]:
                    graph.pop(node_id)
        return graph

    def output_size(self, request: dict, references: list[dict]) -> tuple[int, int]:
        if references:
            raise WorkflowRequestError("reference count invalid")
        manifest, graph = self.load_assets(request["target_id"])
        values = self._values(manifest, request["controls"])
        a, b = map(int, values["aspect_ratio"].split(" ")[0].split(":"))
        multiple = graph[manifest["size"]["node_id"]]["inputs"]["multiple"]
        scale = math.sqrt(values["megapixels"] * 1024 * 1024 / (a * b))
        return round(a * scale / multiple) * multiple, round(b * scale / multiple) * multiple

    def static_health(self, request: dict, object_info: Mapping[str, Any]) -> dict:
        graph = self.compile(request)
        manifest, _ = self.load_assets(request["target_id"])
        missing_classes, missing_files, schema_mismatches = [], [], []
        for node in graph.values():
            class_type = node["class_type"]
            schema = object_info.get(class_type)
            if not isinstance(schema, Mapping):
                missing_classes.append(class_type)
                continue
            inputs = schema.get("input") or {}
            declared = {**inputs.get("required", {}), **inputs.get("optional", {})}
            for name in ("unet_name", "clip_name", "vae_name", "lora_name"):
                if name not in node["inputs"]:
                    continue
                spec = declared.get(name)
                options = (spec[0] if isinstance(spec, list) and spec and isinstance(spec[0], list)
                           else spec[1].get("options") if isinstance(spec, list) and len(spec) > 1 and isinstance(spec[1], Mapping) else None)
                if not isinstance(options, list):
                    schema_mismatches.append(f"{class_type}.{name}")
                elif node["inputs"][name] not in options:
                    missing_files.append(node["inputs"][name])
        for slot in manifest["slots"]:
            schema = object_info.get(slot["class_type"])
            if not isinstance(schema, Mapping):
                continue
            live = ((schema.get("input") or {}).get(slot["schema_group"]) or {}).get(slot["input_name"])
            expected = slot["schema"]
            compatible = isinstance(live, list) and bool(live)
            if compatible:
                if isinstance(expected[0], list):
                    # A file picker: every offered option must still exist.
                    compatible = isinstance(live[0], list) and all(v in live[0] for v in expected[0])
                else:
                    compatible = live == expected
            if not compatible:
                schema_mismatches.append(f"{slot['class_type']}.{slot['input_name']}")
        return {"state": "missing" if missing_classes or missing_files or schema_mismatches else "ready_static",
                "missing_classes": sorted(set(missing_classes)), "missing_files": sorted(set(missing_files)),
                "schema_mismatches": sorted(set(schema_mismatches)), "inference_tested": False,
                "availability": manifest["descriptor"]["availability"]}

    def effective_prompt(self, entry: dict, expected_body: dict) -> str | None:
        queued = entry.get("prompt") if isinstance(entry, Mapping) else None
        if (not isinstance(queued, (list, tuple)) or len(queued) < 4
                or queued[2] != expected_body["prompt"] or not _shared().echo_owned(queued[3], expected_body["extra_data"])):
            raise WorkflowRequestError("history is not owned by this request")
        manifest, _ = self.load_assets(expected_body["extra_data"]["studio"]["target_id"])
        node = (entry.get("outputs") or {}).get(manifest["outputs"]["effective_prompt"]["node_id"])
        text = node.get("text") if isinstance(node, Mapping) else None
        if not isinstance(text, list) or len(text) != 1 or type(text[0]) is not str:
            return None
        if len(text[0]) > 16384:
            raise WorkflowRequestError("effective prompt too large")
        return text[0]


def create_adapter():
    """Catalog adapter-factory protocol; generic admission has no model branch."""
    return CatalogCreateWorkflowAdapter()


def asset_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()
