"""Offline-safe Krea2 CREATE adapter.

The adapter is the provider boundary for the first image producer target.  It
loads only the checked-in runtime assets, compiles an in-memory graph, and
translates the small Comfy transport surface into provider-neutral outcomes.
It has no dependency on publication, image state, film adoption, or routes.
"""
from __future__ import annotations

import copy
import hashlib
import json
import ntpath
import os
import stat
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contract import COMFY_ENRICHMENT_KEYS
from .comfy_failure import failure_reason as _comfy_failure_reason  # lane w24-rest (GEM-056)
from .image_contract import (
    CAPABILITY_SHA256,
    GRAPH_SHA256,
    PROVIDER_ID,
    TARGET_ID,
    ContractError,
    canonical_bytes,
    sha256,
    validate_operation,
    validate_snapshot,
)
from .model_catalog import krea_profile

# Single source: the pinned Turbo graph's file names and fixed sampler values
# come from the shared model catalog (family krea2, variant turbo). The graph
# and manifest are hash-pinned, so a catalog edit that disagrees with them
# fails load_assets instead of silently changing a render.
_KREA_TURBO = krea_profile("turbo")



def _default_asset_root(module_file: os.PathLike[str] | str) -> Path:
    """Where the pinned recipe lives, in a checkout or in the installed plugin.

    Repository layout: ``<repo>/workflows`` (parents[4] from this file).
    Installed layout: the drift-safe installer copies approved workflow
    directories to ``plugins/<id>/dashboard/workflows/`` - the sibling of
    ``generation/``. Same ladder the H3 path uses. Only a directory that
    actually holds this target counts; the hash check still runs on load.
    """
    here = Path(module_file).resolve()
    repository = here.parents[4] / "workflows"
    if (repository / TARGET_ID / "manifest.json").is_file():
        return repository
    installed = here.parents[1] / "workflows"
    if (installed / TARGET_ID / "manifest.json").is_file():
        return installed
    return repository


RUNTIME_ASSET_ROOT = _default_asset_root(__file__)
RUNTIME_TARGET_DIR = TARGET_ID
_MANIFEST_FILENAME = "manifest.json"
_GRAPH_FILENAME = "workflow-api.json"
_NODE_IDS = frozenset({"167", "289", "291", "292", "35", "36", "37", "38", "39"})
_MUTABLE_POINTERS = {
    "prompt": "/35/inputs/text",
    "seed": "/38/inputs/seed",
    "width": "/37/inputs/width",
    "height": "/37/inputs/height",
}
_FIXED_ASSERTIONS = {
    "/167/class_type": "PreviewImage",
    "/289/inputs/weight_dtype": "default",
    "/291/inputs/device": "default",
    "/291/inputs/type": "krea2",
    "/292/inputs/vae_name": _KREA_TURBO["vae"],
    "/37/inputs/batch_size": 1,
    "/38/inputs/cfg": _KREA_TURBO["preset"]["cfg"],
    "/38/inputs/denoise": 1.0,
    "/38/inputs/sampler_name": _KREA_TURBO["preset"]["sampler"],
    "/38/inputs/scheduler": _KREA_TURBO["preset"]["scheduler"],
    "/38/inputs/steps": _KREA_TURBO["preset"]["steps"],
}
_MODEL_ASSERTIONS = {
    "/289/inputs/unet_name": _KREA_TURBO["model"],
    "/291/inputs/clip_name": _KREA_TURBO["encoder"],
    "/292/inputs/vae_name": _KREA_TURBO["vae"],
}


class Krea2AdapterError(ValueError):
    """Raised when the pinned Krea2 boundary or an input is invalid."""


# Live preview (masterplan 2.2): the websocket client id the plugin's single
# preview reader connects with; equal to comfyui_preview.PREVIEW_CLIENT_ID
# (pinned by tests/backend/test_comfyui_preview.py).
LIVE_PREVIEW_CLIENT_ID = "filmsuite-image-preview"
LIVE_PREVIEW_METHOD = "latent2rgb"

# Live step preview like KJNodes' "Model Preview Override" (R2 part B).
# When ComfyUI registers the node, one node is inserted between the model
# chain and the sampler; nothing else in the submitted graph changes. It
# pushes 'kj_preview_override' websocket events (picture + sigma/delta/step
# time) to the prompt's client id, which the plugin's preview reader relays.
LIVE_PREVIEW_NODE_CLASS = "ModelPreviewOverrideKJ"
LIVE_PREVIEW_NODE_ID = "500"
LIVE_PREVIEW_NODE_INPUTS = {"max_resolution": 768, "jpeg_quality": 80, "suppress_default_preview": True,
                            "preview_frames": 1, "preview_fps": 12}
_PREVIEW_PROBE_TTL_SECONDS = 60.0
_PREVIEW_PROBE_RETRY_SECONDS = 10.0
_PREVIEW_NODES_MAX = 64
_preview_probe: dict[str, Any] = {"until": 0.0, "present": None}
# prompt id -> inserted node id (None = not inserted); bounded, newest last.
_PREVIEW_NODES: "OrderedDict[str, str | None]" = OrderedDict()


def insert_live_preview(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``graph`` with the preview node before the sampler.

    Only node ``LIVE_PREVIEW_NODE_ID`` is added and only ``/38/inputs/model``
    is re-pointed at it; every other byte of the graph stays as compiled.
    """
    compiled = copy.deepcopy(dict(graph))
    sampler = compiled.get("38")
    if LIVE_PREVIEW_NODE_ID in compiled or not isinstance(sampler, dict) or sampler.get("class_type") != "KSampler":
        raise Krea2AdapterError("live preview insertion point changed")
    model_from = sampler.get("inputs", {}).get("model")
    if not (isinstance(model_from, list) and len(model_from) == 2 and isinstance(model_from[0], str)):
        raise Krea2AdapterError("live preview insertion point changed")
    compiled[LIVE_PREVIEW_NODE_ID] = {"class_type": LIVE_PREVIEW_NODE_CLASS,
                                      "inputs": {"model": list(model_from), **LIVE_PREVIEW_NODE_INPUTS}}
    sampler["inputs"]["model"] = [LIVE_PREVIEW_NODE_ID, 0]
    return compiled


def live_preview_node(prompt_id: Any) -> str | None:
    """The preview node inserted for one of our prompts, or None."""
    return _PREVIEW_NODES.get(prompt_id) if isinstance(prompt_id, str) else None


def live_preview_known(prompt_id: Any) -> bool:
    return isinstance(prompt_id, str) and prompt_id in _PREVIEW_NODES


def live_preview_probe_state() -> bool | None:
    """Last answer of the node probe: True/False, None before the first one."""
    return _preview_probe["present"]


def _remember_preview_node(prompt_id: str, node_id: str | None) -> None:
    _PREVIEW_NODES[prompt_id] = node_id
    _PREVIEW_NODES.move_to_end(prompt_id)
    while len(_PREVIEW_NODES) > _PREVIEW_NODES_MAX:
        _PREVIEW_NODES.popitem(last=False)


@dataclass(frozen=True)
class SubmitOutcome:
    kind: str
    provider_prompt_id: str | None = None
    reason: str | None = None

    @property
    def prompt_id(self) -> str | None:
        return self.provider_prompt_id


@dataclass(frozen=True)
class PollOutcome:
    kind: str
    status: str
    output_identity: dict[str, str] | None = None
    reason: str | None = None


@dataclass(frozen=True)
class CancelOutcome:
    kind: str
    reason: str | None = None


@dataclass(frozen=True)
class AssetBundle:
    """Private validated asset bytes represented as copy-on-read JSON values."""

    manifest_json: str
    graph_json: str

    @property
    def manifest(self) -> dict[str, Any]:
        return _read_object(self.manifest_json)

    @property
    def graph(self) -> dict[str, Any]:
        return _read_object(self.graph_json)


class Krea2CreateAdapter:
    """Compile and operate one Krea2 CREATE target through an injected client."""

    def __init__(
        self,
        client: Any,
        *,
        asset_root: os.PathLike[str] | str | None = None,
        submit_client_id: str | None = None,
        preview_method: str | None = None,
        kj_preview: bool = False,
    ) -> None:
        self.client = client
        self.asset_root = _resolve_asset_root(RUNTIME_ASSET_ROOT if asset_root is None else asset_root)
        self._assets: AssetBundle | None = None
        # Insert the KJ live preview node when ComfyUI registers it (probed
        # with GET /object_info, cached); off keeps the submission unchanged.
        self.kj_preview = bool(kj_preview)
        # Additive live-preview wiring (masterplan 2.2).  ComfyUI sends a
        # prompt's progress and preview frames only to the websocket whose
        # clientId equals the prompt's client_id, and samples with the
        # server's default preview method unless the prompt names one.  None
        # keeps the legacy submission (client_id = job id, no method).
        if submit_client_id is not None and not _valid_token(submit_client_id):
            raise Krea2AdapterError("submit client id invalid")
        if preview_method not in (None, "latent2rgb", "taesd", "auto", "none"):
            raise Krea2AdapterError("preview method invalid")
        self.submit_client_id = submit_client_id
        self.preview_method = preview_method

    def _load_bundle(self) -> AssetBundle:
        # Revalidate the files on every load.  Caching a previously trusted
        # parse would let an on-disk tamper, replacement, or reparse-point
        # change go unnoticed for the lifetime of an adapter instance.
        target_dir = _safe_directory(self.asset_root, RUNTIME_TARGET_DIR)
        manifest_raw = _read_verified_asset(target_dir, _MANIFEST_FILENAME, CAPABILITY_SHA256)
        graph_raw = _read_verified_asset(target_dir, _GRAPH_FILENAME, GRAPH_SHA256)
        manifest = _parse_object(manifest_raw, "manifest")
        graph = _parse_object(graph_raw, "workflow graph")
        _validate_manifest(manifest)
        _validate_source_graph(graph)
        self._assets = AssetBundle(
            manifest_json=json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
            graph_json=json.dumps(graph, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        )
        return self._assets

    def load_assets(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return independent copies of the verified runtime manifest and graph."""
        bundle = self._load_bundle()
        return bundle.manifest, bundle.graph

    @property
    def manifest(self) -> dict[str, Any]:
        return self.load_assets()[0]

    @property
    def graph(self) -> dict[str, Any]:
        return self.load_assets()[1]

    def static_health(self) -> dict[str, Any]:
        """Report static asset health; this method never calls the client."""
        self._load_bundle()
        return {
            "target_id": TARGET_ID,
            "provider_id": "local-comfyui",
            "policy": "static_only",
            "inference_tested": False,
            "runtime_generation_proven": False,
        }

    def compile_graph(
        self,
        request_snapshot: Mapping[str, Any],
        *,
        graph: Mapping[str, Any] | None = None,
        manifest: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compile one validated snapshot by changing exactly four pointers."""
        try:
            snapshot = copy.deepcopy(dict(request_snapshot))
            validate_snapshot(snapshot)
        except (ContractError, TypeError, ValueError, RecursionError) as exc:
            raise Krea2AdapterError("invalid image request snapshot") from exc

        loaded_manifest, loaded_graph = self.load_assets()
        if manifest is not None and not _json_equal(manifest, loaded_manifest):
            raise Krea2AdapterError("runtime capability manifest changed")
        source = loaded_graph if graph is None else copy.deepcopy(dict(graph))
        if not _json_equal(source, loaded_graph):
            raise Krea2AdapterError("runtime API graph changed")
        _validate_manifest(loaded_manifest)
        _validate_source_graph(source)

        compiled = copy.deepcopy(source)
        artist = snapshot["artist_input"]
        _set_pointer(compiled, _MUTABLE_POINTERS["prompt"], artist["prompt"])
        _set_pointer(compiled, _MUTABLE_POINTERS["seed"], artist["seed"])
        _set_pointer(compiled, _MUTABLE_POINTERS["width"], artist["width"])
        _set_pointer(compiled, _MUTABLE_POINTERS["height"], artist["height"])
        spec = snapshot["effective_input"].get("render_spec")
        if spec is not None:
            return self.compile_render_spec(spec, artist["prompt"], source=source)
        return compiled

    def compile_render_spec(self, spec: Mapping[str, Any], prompt: str, *, source: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Compile only a hash-pinned profile and its separately declared mutations."""
        from .krea2_render_spec import LORA_MAX, validate_spec
        validated = validate_spec(dict(spec))
        profile = validated["profile"]
        if source is None:
            source = self.load_assets()[1]
        if profile == "raw":
            target = _safe_directory(self.asset_root, "krea2-raw-t2i-create")
            manifest_path = _safe_file(target, "manifest.json")
            manifest = _parse_object(manifest_path.read_bytes(), "raw mutations")
            baseline = _parse_object(_safe_file(target, "workflow-api.json").read_bytes(), "raw graph")
        else:
            target = _safe_directory(self.asset_root, RUNTIME_TARGET_DIR)
            manifest = _parse_object(_safe_file(target, "render-mutations.v1.json").read_bytes(), "turbo mutations")
            baseline = copy.deepcopy(dict(source))
        if manifest.get("schema") != "studio.krea2-graph-mutations/1" or manifest.get("profile") != profile:
            raise Krea2AdapterError("render mutation manifest invalid")
        actual = hashlib.sha256(json.dumps(baseline, indent=2, ensure_ascii=False).encode("utf-8") + b"\n").hexdigest()
        if profile == "turbo":
            # The source is already checked against GRAPH_SHA256 by load_assets.
            actual = GRAPH_SHA256
        if actual != manifest.get("base_sha256"):
            raise Krea2AdapterError("render graph fingerprint mismatch")
        declared = {"prompt": "/35/inputs/text", "seed": "/38/inputs/seed", "width": "/37/inputs/width", "height": "/37/inputs/height", "steps": "/38/inputs/steps", "cfg": "/38/inputs/cfg", "sampler": "/38/inputs/sampler_name", "scheduler": "/38/inputs/scheduler", "batch_size": "/37/inputs/batch_size"}
        if profile == "raw":
            declared["negative"] = "/300/inputs/text"
            if baseline.get("300") != {"class_type": "CLIPTextEncode", "inputs": {"clip": ["291", 0], "text": ""}} or baseline["38"]["inputs"]["negative"] != ["300", 0] or baseline["289"]["inputs"]["unet_name"] != "krea2_raw_int8_convrot.safetensors":
                raise Krea2AdapterError("raw baseline topology changed")
            turbo = self.load_assets()[1]
            expected = copy.deepcopy(turbo)
            expected["289"]["inputs"]["unet_name"] = "krea2_raw_int8_convrot.safetensors"
            expected["38"]["inputs"]["negative"] = ["300", 0]
            expected["300"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["291", 0], "text": ""}}
            if not _json_equal(baseline, expected):
                raise Krea2AdapterError("raw baseline topology changed")
        if manifest.get("mutations") != declared or manifest.get("insertion") != {"class_type": "LoraLoaderModelOnly", "ids": [str(400 + index) for index in range(LORA_MAX)], "model_from": ["289", 0], "model_into": "/38/inputs/model", "inputs": ["model", "lora_name", "strength_model"]}:
            raise Krea2AdapterError("render mutation declarations changed")
        compiled = copy.deepcopy(baseline)
        for key, pointer in declared.items():
            _set_pointer(compiled, pointer, prompt if key == "prompt" else validated[key])
        parent = ["289", 0]
        for index, lora in enumerate(validated["loras"]):
            node_id = manifest["insertion"]["ids"][index]
            if node_id in compiled:
                raise Krea2AdapterError("LoRA node collides with baseline")
            compiled[node_id] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": parent, "lora_name": lora["file"].replace("/", os.sep), "strength_model": lora["strength"]}}
            parent = [node_id, 0]
        _set_pointer(compiled, "/38/inputs/model", parent)
        if validated.get("schema") == "studio.krea2-render-spec/2":
            # v2: one ledger job is one picture (the run fans out into jobs),
            # and the model files are the explicit per-role choice.  Only the
            # declared batch pointer and the three loader file inputs change.
            _set_pointer(compiled, "/37/inputs/batch_size", 1)
            files = validated["files"]
            _set_pointer(compiled, "/289/inputs/unet_name", files["diffusion_model"])
            _set_pointer(compiled, "/291/inputs/clip_name", files["text_encoder"])
            _set_pointer(compiled, "/292/inputs/vae_name", files["vae"])
        return compiled

    compile = compile_graph
    compile_request = compile_graph

    async def submit(
        self,
        *,
        image_producer_job_id: str,
        request_snapshot: Mapping[str, Any],
        client_request_id: str,
        request_snapshot_sha256: str | None = None,
    ) -> SubmitOutcome:
        """Submit exactly one compiled graph, translating malformed receipts."""
        try:
            _validate_job_id(image_producer_job_id)
            _validate_token(client_request_id)
            snapshot = copy.deepcopy(dict(request_snapshot))
            snapshot_bytes = validate_snapshot(snapshot)
            snapshot_hash = sha256(snapshot_bytes)
            if request_snapshot_sha256 is not None and request_snapshot_sha256 != snapshot_hash:
                raise Krea2AdapterError("request snapshot hash mismatch")
            compiled = self.compile_graph(snapshot)
        except (ContractError, Krea2AdapterError, TypeError, ValueError, RecursionError):
            return SubmitOutcome("malformed", reason="invalid_request")

        preview_node = None
        if self.kj_preview and await self._kj_preview_present():
            try:
                compiled = insert_live_preview(compiled)
                preview_node = LIVE_PREVIEW_NODE_ID
            except Krea2AdapterError:
                preview_node = None  # preview off; the render itself is unchanged
        metadata = {
            "provider_id": PROVIDER_ID,
            "image_producer_job_id": image_producer_job_id,
            "client_request_id": client_request_id,
            "target_id": TARGET_ID,
            "request_snapshot_sha256": snapshot_hash,
            "capability_manifest_sha256": CAPABILITY_SHA256,
            "api_graph_sha256": GRAPH_SHA256,
        }
        extra_data: dict[str, Any] = {"studio": metadata}
        if self.preview_method is not None:
            extra_data["preview_method"] = self.preview_method
        try:
            receipt = await self.client.submit(
                compiled,
                client_id=self.submit_client_id or image_producer_job_id,
                extra_data=extra_data,
            )
        except Exception:
            return SubmitOutcome("transport_error", reason="submission_unknown")
        if not isinstance(receipt, Mapping):
            return SubmitOutcome("malformed", reason="submission_unknown")
        try:
            node_errors = receipt.get("node_errors")
            if node_errors:
                return SubmitOutcome("malformed", reason="invalid_request")
            prompt_id = receipt.get("prompt_id")
        except Exception:
            return SubmitOutcome("malformed", reason="submission_unknown")
        if not _valid_prompt_id(prompt_id):
            return SubmitOutcome("malformed", reason="submission_unknown")
        if self.kj_preview:
            _remember_preview_node(prompt_id, preview_node)
        return SubmitOutcome("accepted", provider_prompt_id=prompt_id)

    async def _kj_preview_present(self) -> bool:
        """Whether ComfyUI registers the preview node (GET only, cached)."""
        now = time.monotonic()
        if _preview_probe["present"] is not None and now < _preview_probe["until"]:
            return bool(_preview_probe["present"])
        probe = getattr(self.client, "object_info", None)
        present = False
        if probe is not None:
            try:
                info = await probe(LIVE_PREVIEW_NODE_CLASS)
                present = isinstance(info, Mapping) and isinstance(info.get(LIVE_PREVIEW_NODE_CLASS), Mapping)
            except Exception:  # noqa: BLE001 - ComfyUI off or old: preview just off
                present = False
        _preview_probe["present"] = present
        _preview_probe["until"] = now + (_PREVIEW_PROBE_TTL_SECONDS if present else _PREVIEW_PROBE_RETRY_SECONDS)
        return present

    async def recover_submit(
        self,
        client_request_id: str,
        *,
        image_producer_job_id: str | None = None,
        request_snapshot_sha256: str | None = None,
    ) -> SubmitOutcome:
        """Find one fully correlated provider job without issuing a submit."""
        if not _valid_token(client_request_id):
            return SubmitOutcome("malformed", reason="invalid_request")
        if image_producer_job_id is None or request_snapshot_sha256 is None:
            return SubmitOutcome("unresolved", reason="submission_unknown")
        try:
            _validate_job_id(image_producer_job_id)
            _validate_sha256(request_snapshot_sha256)
        except Krea2AdapterError:
            return SubmitOutcome("malformed", reason="invalid_request")
        try:
            queue = await self.client.queue()
            history = await self.client.recent_history(max_items=100)
        except Exception:
            return SubmitOutcome("transport_error", reason="submission_unknown")
        try:
            matches = _correlation_matches(
                queue,
                history,
                {
                    "provider_id": PROVIDER_ID,
                    "image_producer_job_id": image_producer_job_id,
                    "client_request_id": client_request_id,
                    "target_id": TARGET_ID,
                    "request_snapshot_sha256": request_snapshot_sha256,
                    "capability_manifest_sha256": CAPABILITY_SHA256,
                    "api_graph_sha256": GRAPH_SHA256,
                },
            )
        except Krea2AdapterError:
            return SubmitOutcome("malformed", reason="submission_unknown")
        if len(matches) != 1:
            return SubmitOutcome("unresolved", reason="submission_unknown")
        return SubmitOutcome("accepted", provider_prompt_id=next(iter(matches)))

    recover = recover_submit

    async def poll(
        self,
        provider_prompt_id: str,
        *,
        expected_identity: Mapping[str, Any],
    ) -> PollOutcome:
        """Poll one owned provider ID and revalidate its complete Studio identity."""
        try:
            _validate_prompt_id(provider_prompt_id)
            expected = _validate_correlation_identity(expected_identity)
        except Krea2AdapterError:
            raise
        try:
            history = await self.client.history(provider_prompt_id)
        except Exception:
            return PollOutcome("transport_error", "missing", reason="unreachable")
        if not isinstance(history, Mapping):
            return PollOutcome("malformed", "missing", reason="invalid_history")
        entry_present = provider_prompt_id in history
        entry = history[provider_prompt_id] if entry_present else None
        if not entry_present and "status" in history:
            entry = history
            entry_present = True
        if not entry_present:
            return await self._poll_queue(provider_prompt_id, expected)
        if not isinstance(entry, Mapping):
            return PollOutcome("malformed", "missing", reason="invalid_history")
        prompt = entry.get("prompt")
        if (
            not isinstance(prompt, (list, tuple))
            or len(prompt) < 2
            or prompt[1] != provider_prompt_id
            or not _metadata_matches(_metadata_from_prompt(prompt), expected)
        ):
            return PollOutcome("malformed", "missing", reason="result_authority_invalid")
        status = entry.get("status")
        if not isinstance(status, Mapping):
            return PollOutcome("malformed", "missing", reason="invalid_history")
        status_str = status.get("status_str")
        completed = status.get("completed")
        if status_str in {"error", "failed"}:
            return PollOutcome("failed", "failed", reason=_comfy_failure_reason(entry))
        if status_str in {"interrupted", "cancelled"}:
            return PollOutcome("cancelled", "cancelled", reason="cancelled")
        if status_str == "success" or completed is True:
            try:
                descriptor = _extract_png_descriptor(entry)
            except Krea2AdapterError:
                return PollOutcome("malformed", "missing", reason="output_invalid")
            return PollOutcome("ready", "ready", output_identity=descriptor)
        if status_str in {"running", "executing"}:
            return PollOutcome("running", "running")
        if status_str in {"queued", "pending"}:
            return PollOutcome("queued", "queued")
        return await self._poll_queue(provider_prompt_id, expected)

    async def _poll_queue(
        self,
        provider_prompt_id: str,
        expected_identity: Mapping[str, Any],
    ) -> PollOutcome:
        try:
            queue = await self.client.queue()
        except Exception:
            return PollOutcome("transport_error", "missing", reason="unreachable")
        try:
            queue_state = _owned_queue_state(queue, provider_prompt_id, expected_identity)
        except Krea2AdapterError:
            return PollOutcome("malformed", "missing", reason="result_authority_invalid")
        if queue_state == "running":
            return PollOutcome("running", "running")
        if queue_state == "queued":
            return PollOutcome("queued", "queued")
        return PollOutcome("missing", "missing", reason="job_lost")

    async def cancel(self, provider_prompt_id: str) -> CancelOutcome:
        """Cancel exactly one provider prompt; persistence remains coordinator-owned."""
        _validate_prompt_id(provider_prompt_id)
        try:
            response = await self.client.cancel_prompt(provider_prompt_id)
        except Exception as exc:
            # A ComfyUI without the jobs API answers 404/405: stop the owned
            # prompt through the queue + targeted interrupt instead.
            if getattr(exc, "http_status", None) not in (404, 405) or not callable(getattr(self.client, "interrupt_prompt", None)):
                return CancelOutcome("transport_error", reason="unreachable")
            try:
                response = await self.client.interrupt_prompt(provider_prompt_id)
            except Exception:
                return CancelOutcome("transport_error", reason="unreachable")
        if isinstance(response, Mapping) and response.get("cancelled") is True:
            return CancelOutcome("accepted")
        return CancelOutcome("malformed", reason="invalid_cancel_receipt")

    cancel_prompt = cancel


def graph_differences(left: Any, right: Any, path: str = "") -> list[str]:
    """Return JSON pointer-like differences for focused topology tests."""
    if type(left) is not type(right):
        return [path or "<root>"]
    if isinstance(left, Mapping):
        differences: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}/{key}" if path else f"/{key}"
            if key not in left or key not in right:
                differences.append(child)
            else:
                differences.extend(graph_differences(left[key], right[key], child))
        return differences
    if isinstance(left, list):
        if len(left) != len(right):
            return [path or "<root>"]
        differences: list[str] = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(graph_differences(left_item, right_item, f"{path}/{index}"))
        return differences
    return [] if left == right else [path or "<root>"]


def _resolve_asset_root(value: os.PathLike[str] | str) -> Path:
    try:
        root = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise Krea2AdapterError("runtime workflow assets unavailable") from exc
    if not root.is_dir():
        raise Krea2AdapterError("runtime workflow assets unavailable")
    return root


def _safe_directory(root: Path, relative_name: str) -> Path:
    candidate = root / relative_name
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
        mode = os.stat(resolved, follow_symlinks=False).st_mode
    except (OSError, RuntimeError, ValueError) as exc:
        raise Krea2AdapterError("runtime workflow target unavailable") from exc
    if candidate.is_symlink() or not stat.S_ISDIR(mode):
        raise Krea2AdapterError("runtime workflow target unavailable")
    return resolved


def _safe_file(directory: Path, name: str) -> Path:
    if not name or name != Path(name).name or name != ntpath.basename(name):
        raise Krea2AdapterError("runtime workflow asset name invalid")
    candidate = directory / name
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(directory)
        mode = os.stat(resolved, follow_symlinks=False).st_mode
    except (OSError, RuntimeError, ValueError) as exc:
        raise Krea2AdapterError("runtime workflow asset unavailable") from exc
    if candidate.is_symlink() or not stat.S_ISREG(mode):
        raise Krea2AdapterError("runtime workflow asset unavailable")
    return resolved


def _read_verified_asset(directory: Path, name: str, expected_hash: str) -> bytes:
    path = _safe_file(directory, name)
    try:
        data = path.read_bytes()
    except (OSError, RuntimeError) as exc:
        raise Krea2AdapterError("runtime workflow asset unavailable") from exc
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise Krea2AdapterError("runtime workflow asset fingerprint mismatch")
    return data


def _parse_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError, RecursionError) as exc:
        raise Krea2AdapterError(f"{label} is malformed") from exc
    if not isinstance(value, dict):
        raise Krea2AdapterError(f"{label} must be an object")
    return value


def _read_object(raw: str) -> dict[str, Any]:
    return _parse_object(raw.encode("utf-8"), "runtime asset")


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if not isinstance(manifest, Mapping):
        raise Krea2AdapterError("capability manifest is not an object")
    required = {
        "schema", "version", "id", "provider_id", "operation", "prompt_policy", "models",
        "parameters", "mutable_json_pointers", "fixed_assertions", "sampler", "output", "health",
        "policy", "workflow", "provenance", "source", "source_contracts",
    }
    if set(manifest) != required:
        raise Krea2AdapterError("capability manifest shape changed")
    if (
        manifest["schema"] != "studio.image-capability/1"
        or not _json_equal(manifest["version"], 1)
        or manifest["id"] != TARGET_ID
        or manifest["provider_id"] != "comfyui-local"
        or manifest["operation"] != "create"
        or manifest["prompt_policy"] != "literal"
    ):
        raise Krea2AdapterError("capability manifest identity changed")
    if not _json_equal(manifest["mutable_json_pointers"], _MUTABLE_POINTERS):
        raise Krea2AdapterError("mutable JSON pointer declarations changed")
    if not _json_equal(manifest["fixed_assertions"], _FIXED_ASSERTIONS):
        raise Krea2AdapterError("fixed graph assertions changed")
    workflow = manifest["workflow"]
    if (
        not isinstance(workflow, Mapping)
        or workflow.get("api_sha256") != GRAPH_SHA256
        or not _json_equal(workflow.get("node_count"), 9)
    ):
        raise Krea2AdapterError("workflow fingerprint changed")
    health = manifest["health"]
    if not _json_equal(health, {"inference_tested": False, "policy": "static_only", "runtime_generation_proven": False}):
        raise Krea2AdapterError("health policy changed")
    policy = manifest["policy"]
    if (
        not isinstance(policy, Mapping)
        or policy.get("graph_mutation") != "declared_pointers_only"
        or not _json_equal(policy.get("mutable_controls"), ["prompt", "seed", "width", "height"])
        or policy.get("optional_lora") is not False
        or policy.get("runtime_execution_attempted") is not False
    ):
        raise Krea2AdapterError("graph mutation policy changed")
    output = manifest["output"]
    if (
        not isinstance(output, Mapping)
        or output.get("node_id") != "167"
        or not _json_equal(output.get("accepted_keys"), ["images"])
        or output.get("class_type") != "PreviewImage"
        or output.get("descriptor", {}).get("type") != "temp"
        or not _json_equal(output.get("descriptor", {}).get("accepted_extensions"), [".png"])
    ):
        raise Krea2AdapterError("output contract changed")
    if not isinstance(manifest["parameters"], Mapping) or set(manifest["parameters"]) != set(_MUTABLE_POINTERS):
        raise Krea2AdapterError("parameter declarations changed")


def _validate_source_graph(graph: Mapping[str, Any]) -> None:
    if not isinstance(graph, Mapping) or set(graph) != _NODE_IDS:
        raise Krea2AdapterError("pinned graph topology changed")
    for node_id, node in graph.items():
        if not isinstance(node_id, str) or not isinstance(node, Mapping):
            raise Krea2AdapterError("pinned graph node shape changed")
        if not isinstance(node.get("class_type"), str) or not isinstance(node.get("inputs"), Mapping):
            raise Krea2AdapterError("pinned graph node shape changed")
    for pointer, expected in {**_FIXED_ASSERTIONS, **_MODEL_ASSERTIONS}.items():
        if not _json_equal(_get_pointer(graph, pointer), expected):
            raise Krea2AdapterError("pinned graph assertion changed")
    if not _json_equal(graph["167"]["inputs"], {"images": ["39", 0]}):
        raise Krea2AdapterError("pinned output route changed")
    if not _json_equal(graph["39"]["inputs"], {"samples": ["38", 0], "vae": ["292", 0]}):
        raise Krea2AdapterError("pinned decode route changed")
    if not _json_equal(graph["38"]["inputs"], {
        "cfg": 1.0,
        "denoise": 1.0,
        "latent_image": ["37", 0],
        "model": ["289", 0],
        "negative": ["36", 0],
        "positive": ["35", 0],
        "sampler_name": "euler",
        "scheduler": "simple",
        "seed": 123456789,
        "steps": 8,
    }):
        raise Krea2AdapterError("pinned sampler route changed")


def _get_pointer(value: Any, pointer: str) -> Any:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise Krea2AdapterError("JSON pointer invalid")
    current = value
    for token in pointer.split("/")[1:]:
        token = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or token not in current:
            raise Krea2AdapterError("JSON pointer target missing")
        current = current[token]
    return current


def _set_pointer(value: Any, pointer: str, new_value: Any) -> None:
    parts = pointer.split("/")[1:]
    current = value
    for token in parts[:-1]:
        token = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or token not in current:
            raise Krea2AdapterError("mutable JSON pointer target missing")
        current = current[token]
    token = parts[-1].replace("~1", "/").replace("~0", "~")
    if not isinstance(current, dict) or token not in current:
        raise Krea2AdapterError("mutable JSON pointer target missing")
    current[token] = copy.deepcopy(new_value)


def _json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(_json_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_json_equal(a, b) for a, b in zip(left, right))
    return left == right


def _validate_job_id(value: Any) -> None:
    if type(value) is not str or len(value) != 39 or not value.startswith("imgjob_") or any(char not in "0123456789abcdef" for char in value[7:]):
        raise Krea2AdapterError("image producer job ID invalid")


def _valid_token(value: Any) -> bool:
    return type(value) is str and 1 <= len(value) <= 128 and all(char in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:-" for char in value)


def _validate_token(value: Any) -> None:
    if not _valid_token(value):
        raise Krea2AdapterError("client request ID invalid")


def _validate_sha256(value: Any) -> None:
    if type(value) is not str or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise Krea2AdapterError("snapshot hash invalid")


def _valid_prompt_id(value: Any) -> bool:
    return (
        type(value) is str
        and 1 <= len(value) <= 256
        and value == value.strip()
        and not any(ord(char) < 0x20 or char == "\x7f" for char in value)
    )


def _validate_prompt_id(value: Any) -> None:
    if not _valid_prompt_id(value):
        raise Krea2AdapterError("provider prompt ID invalid")


def _metadata_from_prompt(prompt: Any) -> Mapping[str, Any] | None:
    if not isinstance(prompt, (list, tuple)) or len(prompt) < 4:
        return None
    extra_data = prompt[3]
    if not isinstance(extra_data, Mapping) or not isinstance(extra_data.get("studio"), Mapping):
        return None
    return extra_data["studio"]


def _validate_correlation_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "provider_id",
        "image_producer_job_id",
        "client_request_id",
        "target_id",
        "request_snapshot_sha256",
        "capability_manifest_sha256",
        "api_graph_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise Krea2AdapterError("provider correlation identity shape is invalid")
    identity = dict(value)
    _validate_job_id(identity["image_producer_job_id"])
    _validate_token(identity["client_request_id"])
    for key in ("request_snapshot_sha256", "capability_manifest_sha256", "api_graph_sha256"):
        _validate_sha256(identity[key])
    if (
        identity["provider_id"] != PROVIDER_ID
        or identity["target_id"] != TARGET_ID
        or identity["capability_manifest_sha256"] != CAPABILITY_SHA256
        or identity["api_graph_sha256"] != GRAPH_SHA256
    ):
        raise Krea2AdapterError("provider correlation identity is not authoritative")
    return identity


def _correlation_matches(queue: Any, history: Any, expected: Mapping[str, Any]) -> set[str]:
    expected = _validate_correlation_identity(expected)
    occurrences: dict[str, list[Mapping[str, Any] | None]] = {}
    if not isinstance(queue, Mapping) or not isinstance(history, Mapping):
        raise Krea2AdapterError("provider lookup envelope malformed")

    def record(prompt_id: Any, metadata: Mapping[str, Any] | None) -> None:
        if _valid_prompt_id(prompt_id):
            occurrences.setdefault(prompt_id, []).append(metadata)

    for key in ("queue_running", "queue_pending"):
        items = queue.get(key, [])
        if not isinstance(items, list):
            raise Krea2AdapterError("provider queue envelope malformed")
        for item in items:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            record(item[1], _metadata_from_prompt(item) if len(item) >= 4 else None)

    for prompt_id, entry in history.items():
        metadata = None
        if isinstance(entry, Mapping):
            prompt = entry.get("prompt")
            if isinstance(prompt, (list, tuple)) and len(prompt) >= 2:
                inner_prompt_id = prompt[1]
                if _valid_prompt_id(inner_prompt_id) and inner_prompt_id != prompt_id:
                    raise Krea2AdapterError("history prompt identity mismatch")
                if inner_prompt_id == prompt_id:
                    metadata = _metadata_from_prompt(prompt)
        record(prompt_id, metadata)

    matches: set[str] = set()
    for prompt_id, identities in occurrences.items():
        if not any(_metadata_mentions_job(identity, expected) for identity in identities):
            continue
        if not all(_metadata_matches(identity, expected) for identity in identities):
            raise Krea2AdapterError("provider correlation identity conflicts")
        matches.add(prompt_id)
    return matches


def _metadata_mentions_job(metadata: Mapping[str, Any] | None, expected: Mapping[str, Any]) -> bool:
    return metadata is not None and (
        metadata.get("client_request_id") == expected["client_request_id"]
        or metadata.get("image_producer_job_id") == expected["image_producer_job_id"]
    )


def _metadata_matches(metadata: Mapping[str, Any] | None, expected: Mapping[str, Any]) -> bool:
    return (
        metadata is not None
        and set(metadata) == set(expected)
        and all(metadata.get(key) == value for key, value in expected.items())
    )


def _owned_queue_state(
    queue: Any,
    provider_prompt_id: str,
    expected: Mapping[str, Any],
) -> str | None:
    expected = _validate_correlation_identity(expected)
    if not isinstance(queue, Mapping):
        raise Krea2AdapterError("provider queue envelope malformed")
    states: list[str] = []
    for key, state_name in (("queue_running", "running"), ("queue_pending", "queued")):
        items = queue.get(key, [])
        if not isinstance(items, list):
            raise Krea2AdapterError("provider queue envelope malformed")
        for item in items:
            if not isinstance(item, (list, tuple)) or len(item) < 4 or not _valid_prompt_id(item[1]):
                raise Krea2AdapterError("provider queue item malformed")
            if item[1] != provider_prompt_id:
                continue
            if not _metadata_matches(_metadata_from_prompt(item), expected):
                raise Krea2AdapterError("provider queue ownership conflicts")
            states.append(state_name)
    if len(states) > 1:
        raise Krea2AdapterError("provider queue ownership is ambiguous")
    return states[0] if states else None


def _extract_png_descriptor(entry: Mapping[str, Any]) -> dict[str, str]:
    outputs = entry.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {"167"}:
        raise Krea2AdapterError("output node is not authoritative")
    node = outputs.get("167")
    if not isinstance(node, Mapping) or set(node) != {"images"}:
        raise Krea2AdapterError("output key is not authoritative")
    images = node.get("images")
    if not isinstance(images, list) or len(images) != 1:
        raise Krea2AdapterError("output descriptor count is not one")
    raw = images[0]
    if not isinstance(raw, Mapping) or not {"filename", "subfolder", "type"}.issubset(raw):
        raise Krea2AdapterError("output descriptor shape is invalid")
    if set(raw) - {"filename", "subfolder", "type", "mime_type", "extension"} - COMFY_ENRICHMENT_KEYS:
        raise Krea2AdapterError("output descriptor shape is invalid")
    filename = raw.get("filename")
    subfolder = raw.get("subfolder")
    folder_type = raw.get("type")
    if (
        type(filename) is not str
        or not 1 <= len(filename) <= 255
        or type(subfolder) is not str
        or subfolder != ""
        or folder_type != "temp"
        or filename != ntpath.basename(filename)
        or "/" in filename
        or "\\" in filename
        or any(char in filename for char in '<>:"|?*')
        or "\x00" in filename
        or filename in {".", ".."}
        or filename != filename.rstrip(" .")
        or filename.split(".", 1)[0].casefold() in {
            "con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))
        }
        or not filename.casefold().endswith(".png")
        or any(ord(char) < 0x20 or 0x7F <= ord(char) <= 0x9F for char in filename)
    ):
        raise Krea2AdapterError("output descriptor is not a safe PNG")
    if "mime_type" in raw and raw["mime_type"] != "image/png":
        raise Krea2AdapterError("output descriptor MIME is invalid")
    if "extension" in raw and raw["extension"] != ".png":
        raise Krea2AdapterError("output descriptor extension is invalid")
    return {
        "filename": filename,
        "subfolder": "",
        "type": "temp",
        "mime_type": "image/png",
        "extension": ".png",
    }


__all__ = [
    "AssetBundle",
    "CancelOutcome",
    "Krea2AdapterError",
    "Krea2CreateAdapter",
    "PollOutcome",
    "RUNTIME_ASSET_ROOT",
    "SubmitOutcome",
    "graph_differences",
]
