"""Additive Qwen PE v3 Create/Edit compiler and injected transport boundary.

This boundary is deliberately NOT registered as a durable Image job route yet.
Callers must resolve/upload owned media and durably admit a job before submit;
this class neither uploads arbitrary paths nor retries an ambiguous submission.
Legacy crop-edit assets, ledger snapshots and adapter remain byte-stable.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

REQUEST_SCHEMA = "studio.image-workflow-request/1"
_UPLOAD = re.compile(r"^filmsuite-(?:edit|ref)-[0-9a-f]{32}\.png \[temp\]$")
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_CATALOG_MODULE = None


class WorkflowRequestError(ValueError):
    """A finite refusal before any transport I/O."""


def echo_owned(echoed: Any, sent: Mapping[str, Any]) -> bool:
    """Whether ComfyUI's stored ``prompt[3]`` is the ``extra_data`` the Suite sent.

    The server adds its own keys (``client_id``, ``create_time`` in 0.41) to
    what it stores; every key the Suite sent must come back unchanged.
    """
    return isinstance(echoed, Mapping) and all(key in echoed and echoed[key] == value for key, value in sent.items())


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def load_catalog_module():
    """Also works when the host loads this leaf unregistered by file path."""
    global _CATALOG_MODULE
    if _CATALOG_MODULE is None:
        path = Path(__file__).with_name("model_catalog.py")
        spec = importlib.util.spec_from_file_location("image_workflow_catalog", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CATALOG_MODULE = module
    return _CATALOG_MODULE


def _read_asset(folder: Path, name: str, expected: str) -> dict:
    path = folder / name
    try:
        if path.is_symlink() or folder.is_symlink():
            raise WorkflowRequestError("workflow asset is a link")
        resolved = path.resolve(strict=True)
        resolved.relative_to(folder.resolve(strict=True))
        if not resolved.is_file() or resolved.stat().st_size > 512 * 1024:
            raise WorkflowRequestError("workflow asset size invalid")
        raw = resolved.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise WorkflowRequestError("workflow asset fingerprint mismatch")
        value = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise WorkflowRequestError("workflow asset unavailable or invalid") from exc
    if type(value) is not dict:
        raise WorkflowRequestError("workflow asset malformed")
    return value


def _set_pointer(graph: dict, pointer: str, value: Any) -> None:
    parts = pointer.split("/")[1:]
    node = graph
    for part in parts[:-1]:
        node = node[part]
    if parts[-1] not in node:
        raise WorkflowRequestError("declared slot missing")
    node[parts[-1]] = copy.deepcopy(value)


def _validate_control(control: Mapping, value: Any) -> None:
    kind = control["kind"]
    valid = ((kind == "boolean" and type(value) is bool)
             or (kind == "integer" and type(value) is int)
             or (kind == "number" and type(value) in (float, int) and math.isfinite(value))
             or (kind == "textarea" and type(value) is str and len(value) <= 16384 and "\x00" not in value)
             or (kind == "select" and type(value) is str and value in control["options"]))
    if not valid:
        raise WorkflowRequestError(f"invalid control: {control['id']}")
    if kind in ("number", "integer"):
        if not control["min"] <= value <= control["max"]:
            raise WorkflowRequestError(f"control out of range: {control['id']}")
        if kind == "integer" and (value - control["min"]) % control.get("step", 1):
            raise WorkflowRequestError(f"control not on grid: {control['id']}")


class QwenImageWorkflowAdapter:
    """Pinned family compilation only; injected transport never drains a queue."""

    def __init__(self, client: Any = None, *, asset_root: Path | None = None):
        self.client = client
        here = Path(__file__).resolve()
        repository = here.parents[4] / "workflows"
        installed = here.parents[1] / "workflows"
        pinned = repository / "qwen-image-2.1-create-pe-v3-v2" / "manifest.json"
        self.asset_root = Path(asset_root) if asset_root is not None else (repository if pinned.is_file() else installed)

    def _entry(self, target_id: str) -> dict:
        catalog = load_catalog_module().catalog_view()  # shared, read-only
        family = next(f for f in catalog["families"] if f["id"] == "qwen_image_2_1")
        for variant in family["variants"]:
            workflow = variant.get("workflow")
            if workflow and workflow["descriptor"]["target_id"] == target_id:
                return workflow
        raise WorkflowRequestError("workflow target not offered")

    def load_assets(self, target_id: str) -> tuple[dict, dict]:
        entry = self._entry(target_id)
        folder = self.asset_root / target_id
        manifest = _read_asset(folder, "manifest.json", entry["manifest_sha256"])
        graph = _read_asset(folder, "workflow-api.json", entry["api_sha256"])
        if (manifest.get("schema") != "studio.image-workflow/1" or manifest.get("id") != target_id
                or manifest.get("family") != "qwen_image_2_1" or manifest.get("descriptor") != entry["descriptor"]
                or manifest["workflow"]["api_sha256"] != entry["api_sha256"]
                or manifest["workflow"]["node_count"] != len(graph)):
            raise WorkflowRequestError("workflow identity mismatch")
        return manifest, graph

    def compile(self, request: Mapping[str, Any]) -> dict:
        if (type(request) is not dict or set(request) != {"schema", "target_id", "prompt", "controls", "references"}
                or request["schema"] != REQUEST_SCHEMA or type(request["target_id"]) is not str):
            raise WorkflowRequestError("request shape invalid")
        manifest, graph = self.load_assets(request["target_id"])
        prompt = request["prompt"]
        if type(prompt) is not str or not prompt.strip() or len(prompt) > 16384 or "\x00" in prompt:
            raise WorkflowRequestError("prompt invalid")
        # Preserve artist bytes. Role instructions are a separate, visible suffix.
        values = request["controls"]
        offered = {c["id"]: c for c in manifest["descriptor"]["controls"]}
        if type(values) is not dict or set(values) - set(offered):
            raise WorkflowRequestError("controls not offered")
        effective = {}
        for name, control in offered.items():
            effective[name] = copy.deepcopy(values.get(name, control["default"]))
            _validate_control(control, effective[name])
        refs = request["references"]
        if type(refs) is not list or len(refs) > manifest["references"]["max"]:
            raise WorkflowRequestError("reference count invalid")
        editing = manifest["operation"] == "edit"
        roles = next(r["roles"] for r in manifest["descriptor"]["media_roles"] if r["id"] == "references")
        for index, ref in enumerate(refs):
            if (type(ref) is not dict or set(ref) != {"image", "role"}
                    or type(ref["image"]) is not str or not _UPLOAD.fullmatch(ref["image"])
                    or type(ref["role"]) is not str
                    or ref["role"] not in (["target"] if editing and index == 0 else roles)):
                raise WorkflowRequestError("reference image or role invalid")
        if editing and not refs:
            raise WorkflowRequestError("edit target required")
        for slot in manifest["slots"]:
            _set_pointer(graph, slot["pointer"], effective[slot["id"]])
        encoder = graph["490"]["inputs"]
        batch = graph["503"]["inputs"]
        for name in list(encoder):
            if name.startswith("images."):
                del encoder[name]
        batch.clear()
        graph.pop("492")
        role_lines = []
        for index, ref in enumerate(refs):
            node_id = str(manifest["references"]["additional_node_base"] + index)
            if node_id in graph:
                raise WorkflowRequestError("reference node collision")
            graph[node_id] = {"class_type": "LoadImage", "inputs": {"image": ref["image"]}}
            encoder[f"images.image_{index + 1}"] = [node_id, 0]
            batch[f"images.image{index}"] = [node_id, 0]
            role_lines.append(f"<image{index + 1}>: {'edit target; preserve everything not requested' if ref['role'] == 'target' else ref['role'] + ' reference'}.")
        graph["502"]["inputs"]["value"] = prompt + ("\n\nReference roles:\n" + "\n".join(role_lines) if refs else "")
        enhancer = manifest["enhancer"]
        if not refs:
            graph.pop("503")
            graph["504"]["inputs"].pop("image", None)
        if effective["prompt_enhancer"]:
            graph["500"]["inputs"]["clip_name"] = enhancer["i2i_model" if refs else "t2i_model"]
        else:
            # OFF is structurally pruned, not just lazy-unselected: an absent PE
            # checkpoint must not block validation or load any enhancer weights.
            graph["505"]["inputs"].pop("on_true")
            for node_id in enhancer["branch_nodes"]:
                graph.pop(node_id, None)
        return graph

    def output_size(self, request: dict, references: list[dict]) -> tuple[int, int]:
        manifest, graph = self.load_assets(request["target_id"])
        values = {c["id"]: request["controls"].get(c["id"], c["default"]) for c in manifest["descriptor"]["controls"]}
        if manifest["descriptor"]["output"]["size_policy"] == "first_reference":
            if not references:
                raise WorkflowRequestError("edit target required")
            w, h = references[0]["width"], references[0]["height"]
            resolution = values["reference_resolution"]
            if resolution:
                ratio = w / h
                return max(32, round(math.sqrt(resolution * resolution * ratio) / 32) * 32), max(32, round(math.sqrt(resolution * resolution / ratio) / 32) * 32)
            return max(32, round(w / 32) * 32), max(32, round(h / 32) * 32)
        a, b = map(int, values["aspect_ratio"].split(" ")[0].split(":"))
        node = next(n for n in graph.values() if n["class_type"] == "ResolutionSelector")
        multiple = node["inputs"]["multiple"]
        scale = math.sqrt(values["megapixels"] * 1024 * 1024 / (a * b))
        return round(a * scale / multiple) * multiple, round(b * scale / multiple) * multiple

    def static_health(self, request: dict, object_info: Mapping[str, Any]) -> dict:
        """Read-only admission facts for the chosen PE branch; never inference."""
        graph = self.compile(request)
        manifest, _ = self.load_assets(request["target_id"])
        missing_classes, missing_files, schema_mismatches = [], [], []
        for node in graph.values():
            class_type = node["class_type"]
            schema = object_info.get(class_type)
            if not isinstance(schema, Mapping):
                if class_type not in missing_classes:
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
            # File/options inventories can grow; scalar types and bounds may
            # not silently change under a pinned control declaration.
            compatible = isinstance(live, list) and bool(live)
            if compatible:
                if isinstance(expected[0], list):
                    compatible = isinstance(live[0], list) and all(v in live[0] for v in expected[0])
                else:
                    compatible = live == expected
            if not compatible:
                schema_mismatches.append(f"{slot['class_type']}.{slot['input_name']}")
        return {"state": "missing" if missing_classes or missing_files or schema_mismatches else "ready_static",
                "missing_classes": sorted(set(missing_classes)), "missing_files": sorted(set(missing_files)),
                "schema_mismatches": sorted(set(schema_mismatches)), "inference_tested": False,
                "availability": manifest["descriptor"]["availability"]}

    def submit_body(self, request: dict, client_request_id: str) -> dict:
        if type(client_request_id) is not str or not _TOKEN.fullmatch(client_request_id):
            raise WorkflowRequestError("client request id invalid")
        graph = self.compile(request)
        entry = self._entry(request["target_id"])
        identity = {"target_id": request["target_id"], "client_request_id": client_request_id,
                    "request_sha256": hashlib.sha256(canonical_bytes(request)).hexdigest(),
                    "manifest_sha256": entry["manifest_sha256"], "api_sha256": entry["api_sha256"]}
        return {"prompt": graph, "client_id": client_request_id, "extra_data": {"studio": identity}}

    async def submit(self, request: dict, client_request_id: str) -> str:
        if self.client is None:
            raise WorkflowRequestError("transport is not configured")
        body = self.submit_body(request, client_request_id)
        receipt = await self.client.submit(body["prompt"], client_id=body["client_id"], extra_data=body["extra_data"])
        if (not isinstance(receipt, Mapping) or receipt.get("node_errors")
                or type(receipt.get("prompt_id")) is not str or not receipt["prompt_id"].strip()
                or len(receipt["prompt_id"]) > 256 or any(ord(c) < 32 for c in receipt["prompt_id"])):
            raise WorkflowRequestError("submit receipt invalid")
        return receipt["prompt_id"]

    def effective_prompt(self, entry: dict, expected_body: dict) -> str | None:
        queued = entry.get("prompt") if isinstance(entry, Mapping) else None
        if (not isinstance(queued, (list, tuple)) or len(queued) < 4
                or queued[2] != expected_body["prompt"] or not echo_owned(queued[3], expected_body["extra_data"])):
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
    return QwenImageWorkflowAdapter()
