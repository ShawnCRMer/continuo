"""Additive descriptor-driven snapshots; historical producer bytes are unchanged."""
from __future__ import annotations
import copy
from . import image_contract as c, model_catalog
from . import image_workflow_run as run
from importlib import import_module


def entries():
    # Shared read-only catalog view: every job validation reaches this via
    # known_target_ids(); callers read entries and copy what they keep.
    return {v["workflow"]["descriptor"]["target_id"]: (f, v["workflow"])
            for f in model_catalog.catalog_view()["families"] if f["id"] in model_catalog.IMAGE_WORKFLOW_ADAPTERS
            for v in f["variants"] if v.get("workflow")}


def compiler(target_id):
    # The adapter registry is backend-only; controls and roles remain catalog data.
    family, entry = entries()[target_id]
    module_name = model_catalog.IMAGE_WORKFLOW_ADAPTERS[family["id"]]
    return import_module("." + module_name, __package__).create_adapter()


def build_snapshot(request, owned_references, basis, policy):
    target = request["target_id"]
    family, entry = entries()[target]
    desc = entry["descriptor"]
    values = {control["id"]: copy.deepcopy(request["controls"].get(control["id"], control["default"])) for control in desc["controls"]}
    if set(request["controls"]) - set(values):
        raise c.ContractError("controls-not-offered")
    effective = {"schema": desc["request_schema"], "target_id": target, "prompt": request["prompt"], "controls": values,
                 "references": [{"image": "filmsuite-ref-" + ref["sha256"][:32] + ".png [temp]", "role": ref["role"]} for ref in owned_references]}
    if "loras" in request:
        # w1-imgout: a LoRA chain after the target's model node. Without LoRAs the key is absent,
        # so every historical snapshot keeps its exact bytes.
        try:
            loras = run.normalize_loras(request["loras"])
        except run.RunError as exc:
            raise c.ContractError("workflow-loras-invalid") from exc
        if not loras or loras != request["loras"]:
            raise c.ContractError("workflow-loras-invalid")
        effective["loras"] = loras
    adapter = compiler(target)
    run.compile_request(adapter, effective)
    width, height = adapter.output_size(run.strip(effective)[0], owned_references)
    c._strict_int(width, "width", 16, c.MAX_DIMENSION, 8)
    c._strict_int(height, "height", 16, c.MAX_DIMENSION, 8)
    if width * height > c.MAX_PIXELS:
        raise c.ContractError("output-size-exceeded")
    c.validate_basis(basis); c.validate_policy(policy)
    return {"operation": "workflow_edit" if desc["capabilities"]["edit"] else "workflow_create", "target_id": target,
            "artist_input": {"prompt": request["prompt"], "seed": values.get("seed", 0), "width": width, "height": height},
            "effective_input": {"workflow": effective, "owned_references": copy.deepcopy(owned_references), "output": copy.deepcopy(desc["output"])},
            "context_basis": copy.deepcopy(basis), "attach_policy": copy.deepcopy(policy),
            "capability_manifest_sha256": entry["manifest_sha256"], "api_graph_sha256": entry["api_sha256"]}


def validate_snapshot(snapshot):
    effective = snapshot["effective_input"]
    if type(effective) is not dict or set(effective) != {"workflow", "owned_references", "output"}:
        raise c.ContractError("workflow-effective-shape")
    refs = effective["owned_references"]
    if type(refs) is not list or len(refs) > 16:
        raise c.ContractError("workflow-reference-shape")
    for ref in refs:
        if type(ref) is not dict or set(ref) != {"image_producer_job_id", "sha256", "width", "height", "role"}:
            raise c.ContractError("workflow-reference-shape")
        c._id_value(ref["image_producer_job_id"], "imgjob_", "reference-job")
        c._hex64(ref["sha256"], "reference-sha256")
        c._strict_int(ref["width"], "reference-width", 1, c.MAX_DIMENSION)
        c._strict_int(ref["height"], "reference-height", 1, c.MAX_DIMENSION)
    request = effective["workflow"]
    rebuilt = build_snapshot(request, refs, snapshot["context_basis"], snapshot["attach_policy"])
    raw = c.canonical_bytes(snapshot)
    if raw != c.canonical_bytes(rebuilt):
        # Lane w29-fix2 W29F-P0 (Install 134 refused: four persisted P-H3-Image jobs no longer decoded after w28 IMG-219
        # changed that workflow's manifest): a job persisted under an EARLIER pin of the same workflow keeps decoding for
        # ever. It is checked against its own recorded pin (a known earlier pair of this target) and by structure, never
        # against today's manifest; a new submit still builds with the current one (build_snapshot). Reference: Lightroom
        # / Resolve open a catalog or project made by an older version and keep every picture in it.
        if not _earlier_pin(snapshot, rebuilt):
            raise c.ContractError("workflow-snapshot-mismatch")
    return raw


_PIN_HISTORY = None


def pin_history():
    """target_id -> [(manifest_sha256, api_sha256)] of every earlier committed pin (workflow_pin_history.v1.json)."""
    global _PIN_HISTORY
    if _PIN_HISTORY is None:
        import json
        from pathlib import Path
        doc = json.loads((Path(__file__).with_name("workflow_pin_history.v1.json")).read_text(encoding="utf-8"))
        _PIN_HISTORY = {target: {(p["manifest_sha256"], p["api_sha256"]) for p in pins} for target, pins in doc["earlier"].items()}
    return _PIN_HISTORY


def _earlier_pin(snapshot, rebuilt):
    target = snapshot.get("target_id")
    pin = (snapshot.get("capability_manifest_sha256"), snapshot.get("api_graph_sha256"))
    if pin not in pin_history().get(target, set()):
        return False
    # Structure: the same operation and target, the request as the workflow schema has it, a real picture size.
    effective, request = snapshot["effective_input"], snapshot["effective_input"]["workflow"]
    if snapshot["operation"] != rebuilt["operation"] or target != rebuilt["target_id"]:
        return False
    if type(request) is not dict or request.get("target_id") != target or request.get("schema") != rebuilt["effective_input"]["workflow"]["schema"]:
        return False
    if type(request.get("controls")) is not dict or type(request.get("prompt")) is not str:
        return False
    artist = snapshot["artist_input"]
    if type(artist) is not dict or set(artist) != {"prompt", "seed", "width", "height"} or artist["prompt"] != request["prompt"]:
        return False
    c._strict_int(artist["width"], "width", 16, c.MAX_DIMENSION, 8)
    c._strict_int(artist["height"], "height", 16, c.MAX_DIMENSION, 8)
    if artist["width"] * artist["height"] > c.MAX_PIXELS:
        return False
    return type(effective["output"]) is dict
