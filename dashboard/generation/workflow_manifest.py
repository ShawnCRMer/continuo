"""Load and validate immutable, pinned ComfyUI workflow profiles.

The loader deliberately treats the manifest and API graph as one executable
artifact.  It verifies every file before exposing a profile and keeps the
validated JSON sources private so every read/compile operation returns a fresh
mutable copy.
"""

from __future__ import annotations

import hashlib
import json
import math
import ntpath
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


def _load_h3_megapixels():
    """The one H3 megapixel table (``h3_megapixels.py`` next to this file; lane p13-vid VID-005).

    Loaded by path so this module works both as a package member and when loaded alone by path."""
    import hashlib as _hashlib
    import importlib.util as _importlib_util
    import sys as _sys
    from pathlib import Path as _Path

    path = _Path(__file__).with_name("h3_megapixels.py")
    name = "film_suite_h3_megapixels_" + _hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
    module = _sys.modules.get(name)
    if module is None:
        spec = _importlib_util.spec_from_file_location(name, path)
        module = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _sys.modules[name] = module
    return module


H3_MP = _load_h3_megapixels()


TARGET_ID = "minimax-h3-t2v"
SCHEMA = "studio.comfy-workflow/1"
PROVIDER_ID = "comfyui-local"

_EXPECTED_API_SHA256 = "b1ac67824d04413657af1ff17dde899c5d0250a1d78aaea8a2edcf62e5af995b"
_EXPECTED_EDITOR_SHA256 = "31ab33fdb053a7834cc866bd7aa08b887518fc656e4a796c89779c6b5e1786e6"
_EXPECTED_LICENSE_SHA256 = "5dccd61c80c847cd6b22fa98124010163ab8646e573557a16a5cae31122f2056"
_EXPECTED_MANIFEST_SHA256 = "3d243b565d52b2adb200a6de2bca99843240df134ca211437686d7f1e816aa07"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PROFILE_VALIDATION_TOKEN = object()

_EXPECTED_RUNTIME = {
    "comfyui_versions": ["0.30.0", "0.30.1", "0.31.0", "0.37.0", "0.41.0"],
    "frontend_versions": ["1.47.12", "1.48.7", "1.53.6", "1.55.17"],
    "workflow_templates_versions": ["0.11.27", "0.11.37", "0.11.70", "0.11.86"],
    # Lists = runtimes WE tested; health runs an untested runtime when the live
    # node schema matches schema_fingerprint (generation/runtime_policy.py).
    "policy": "schema_verified",
}
_EXPECTED_WORKFLOW = {
    "api_file": "workflow-api.json",
    "api_sha256": _EXPECTED_API_SHA256,
    "editor_file": "official-editor-template.json",
    "editor_sha256": _EXPECTED_EDITOR_SHA256,
    "editor_distribution": "comfyui-workflow-templates-json 0.1.27",
    "license": "MIT",
    "license_file": "COMFYUI-WORKFLOW-TEMPLATES-LICENSE.txt",
    "node_count": 18,
    "source": "Official installed MiniMax H3 template plus a successful local API export; mutable literals were replaced with Studio defaults without changing graph topology. fs-obvb: the save is a take pair (mp4 + mctx sidecar) by H3TrimAndSaveVideoWithMCtx (pins unwired, a root take); its path reaches /history through PreviewAny.",
}
_EXPECTED_PARAMETERS = {
    "prompt": {
        "pointer": "/105:104/inputs/prompt",
        "type": "string",
        "required": True,
    },
    "duration_seconds": {
        "pointer": "/105:111/inputs/value",
        "type": "number",
        "integer_only": True,
        "minimum": 4,
        "maximum": 15,
    },
    "seed": {
        "pointer": "/124/inputs/value",
        "type": "integer",
        "minimum": 0,
        "maximum": 4294967295,
    },
    "aspect_ratio": {
        "pointer": "/115/inputs/aspect_ratio",
        "type": "string",
        "enum": [
            "21:9 (Ultrawide)",
            "16:9 (Widescreen)",
            "4:3 (Standard)",
            "1:1 (Square)",
            "3:4 (Portrait Standard)",
            "9:16 (Portrait Widescreen)",
        ],
    },
    "megapixels": {
        "pointer": "/115/inputs/megapixels",
        "type": "number",
        "enum": list(H3_MP.options("minimax-h3-t2v")),
    },
}
_EXPECTED_FIXED_ASSERTIONS = {
    "/115/inputs/multiple": 32,
    "/105:107/inputs/expression": "max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17",
    "/105:9/inputs/steps": 20,
    "/105:17/inputs/sampler_name": "res_multistep",
    "/92/class_type": "H3TrimAndSaveVideoWithMCtx",
    "/105:104/class_type": "MiniMaxH3ImageToVideo",
}
_EXPECTED_MODELS = {
    "diffusion_model": {
        "folder": "diffusion_models",
        "filename": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "assertion_pointer": "/105:6/inputs/unet_name",
    },
    "text_encoder": {
        "folder": "text_encoders",
        "filename": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "assertion_pointer": "/105:13/inputs/clip_name",
    },
    "video_vae": {
        "folder": "vae",
        "filename": "minimax_h3_video_vae_fp16.safetensors",
        "assertion_pointer": "/105:11/inputs/vae_name",
    },
    "audio_vae": {
        "folder": "vae",
        "filename": "minimax_h3_audio_vae_fp32.safetensors",
        "assertion_pointer": "/105:24/inputs/vae_name",
    },
}
_EXPECTED_REQUIRED_CLASSES = [
    "BasicGuider",
    "BasicScheduler",
    "CLIPLoader",
    "ComfyMathExpression",
    "H3TrimAndSaveVideoWithMCtx",
    "INTConstant",
    "KSamplerSelect",
    "MiniMaxH3ImageToVideo",
    "PreviewAny",
    "PrimitiveFloat",
    "RandomNoise",
    "ResolutionSelector",
    "SamplerCustomAdvanced",
    "UNETLoader",
    "VAEDecode",
    "VAEDecodeAudio",
    "VAELoader",
]
_EXPECTED_OUTPUT = {
    # lane fs-obvb: the take pair's mp4 is named by the path text PreviewAny 93 shows
    "node_id": "93",
    "accepted_keys": ["text"],
    "required_folder_type": "output",
    "extension": ".mp4",
    "mime": "video/mp4",
    "retrieve_via": "/view",
    "path_text_folder": "studio-h3",
    "sidecar_suffix": ".mctx.safetensors",
    # a job submitted on the SaveVideo graph before fs-obvb is still read by its output
    "legacy_output": {"node_id": "92", "accepted_keys": ["images", "videos", "video"]},
}
_EXPECTED_POLICY = {
    "graph_mutation": "declared_pointers_only",
    "custom_node_install": "never",
    "model_download": "never_automatic",
    "submission_requires_explicit_user_intent": True,
    "overwrite_comfy_output": False,
}
_EXPECTED_CAPABILITIES = {
    "modalities": ["text_to_video"],
    "audio": "required_generated",
    "duration": {
        "unit": "seconds",
        "min": 4.0,
        "max": 15.0,
        "default": 5.0,
        "step": 1.0,
        "request_type": "integer_seconds",
        "fps": 24,
        "effective_frames": {
            "rule": "round_fps_then_align_modulo",
            "minimum_frames": 5,
            "modulus": 17,
            "remainder": 5,
        },
    },
    "aspect_ratios": [
        "21:9 (Ultrawide)",
        "16:9 (Widescreen)",
        "4:3 (Standard)",
        "1:1 (Square)",
        "3:4 (Portrait Standard)",
        "9:16 (Portrait Widescreen)",
    ],
    "megapixel_presets": list(H3_MP.options("minimax-h3-t2v")),
    "seed": True,
    "negative_prompt": False,
    "first_frame": False,
    "last_frame": False,
}


class WorkflowManifestError(ValueError):
    """Stable domain error raised for every invalid or unavailable profile."""


class _DuplicateKeyError(ValueError):
    pass


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _json_equal(actual: Any, expected: Any) -> bool:
    """Compare JSON values with integer/float and boolean types kept distinct."""
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return (
            set(actual) == set(expected)
            and all(_json_equal(actual[key], expected[key]) for key in actual)
        )
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(
            _json_equal(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _require_object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkflowManifestError(f"{path} must be an object")
    return value


def _require_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise WorkflowManifestError(f"{path} is missing keys: {', '.join(missing)}")
    if unknown:
        raise WorkflowManifestError(f"{path} contains unknown keys: {', '.join(unknown)}")


def _require_exact(actual: Any, expected: Any, path: str) -> None:
    if not _json_equal(actual, expected):
        raise WorkflowManifestError(f"{path} is not pinned to the approved value")


def _require_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise WorkflowManifestError(f"{path} must be non-empty text")
    return value


def _require_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkflowManifestError(f"{path} must be an integer")
    return value


def _require_number(value: Any, path: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkflowManifestError(f"{path} must be a number")
    if isinstance(value, float) and not math.isfinite(value):
        raise WorkflowManifestError(f"{path} must be finite")
    return value


def _require_sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise WorkflowManifestError(f"{path} must be a lowercase SHA-256 hex digest")
    return value


def _resolve_root(workflow_root: os.PathLike[str] | str | None) -> Path:
    raw_root: os.PathLike[str] | str = (
        Path(__file__).with_name("workflows") if workflow_root is None else workflow_root
    )
    try:
        root = Path(raw_root).resolve(strict=False)
        if not root.exists() or not root.is_dir():
            raise WorkflowManifestError(f"workflow root is not an existing directory: {root}")
        return root
    except WorkflowManifestError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise WorkflowManifestError("workflow root cannot be resolved") from None


def _safe_file(root: Path, relative_name: Any, label: str) -> Path:
    if not isinstance(relative_name, str) or not relative_name or "\x00" in relative_name:
        raise WorkflowManifestError(f"{label} must be a relative file path below the workflow root")

    # pathlib uses the host flavour.  Check both flavours so Windows drive,
    # UNC, and drive-relative names are rejected even from a POSIX-like shell.
    windows_drive, _ = ntpath.splitdrive(relative_name)
    if (
        Path(relative_name).is_absolute()
        or ntpath.isabs(relative_name)
        or bool(windows_drive)
        or relative_name.startswith(("/", "\\"))
    ):
        raise WorkflowManifestError(f"{label} must be a relative file path below the workflow root")

    candidate = root / relative_name
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        raise WorkflowManifestError(f"{label} escapes the workflow root") from None

    try:
        if not resolved.exists() or not resolved.is_file():
            raise WorkflowManifestError(f"{label} is missing or is not a file")
    except WorkflowManifestError:
        raise
    except OSError:
        raise WorkflowManifestError(f"{label} cannot be inspected") from None
    return resolved


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except (OSError, RuntimeError):
        raise WorkflowManifestError(f"cannot read {label}") from None


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_bytes(path, label)
    try:
        text = raw.decode("utf-8")
        value = json.loads(text, object_pairs_hook=_object_no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateKeyError, TypeError, ValueError):
        raise WorkflowManifestError(f"{label} contains malformed JSON") from None
    if not isinstance(value, dict):
        raise WorkflowManifestError(f"{label} must contain a JSON object")
    return value, raw


def _decode_pointer_token(token: str, pointer: str) -> str:
    decoded: list[str] = []
    index = 0
    while index < len(token):
        character = token[index]
        if character != "~":
            decoded.append(character)
            index += 1
            continue
        if index + 1 >= len(token) or token[index + 1] not in "01":
            raise WorkflowManifestError(f"JSON pointer {pointer!r} contains an invalid escape")
        decoded.append("~" if token[index + 1] == "0" else "/")
        index += 2
    return "".join(decoded)


def _resolve_pointer(graph: Any, pointer: Any, context: str) -> Any:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise WorkflowManifestError(f"{context} must be an absolute JSON pointer")
    current = graph
    for raw_token in pointer.split("/")[1:]:
        token = _decode_pointer_token(raw_token, pointer)
        if isinstance(current, dict):
            if token not in current:
                raise WorkflowManifestError(f"{context} target does not exist: {pointer}")
            current = current[token]
        elif isinstance(current, list):
            if token == "-" or not token.isdigit():
                raise WorkflowManifestError(f"{context} target does not exist: {pointer}")
            index = int(token)
            if index >= len(current):
                raise WorkflowManifestError(f"{context} target does not exist: {pointer}")
            current = current[index]
        else:
            raise WorkflowManifestError(f"{context} target does not exist: {pointer}")
    return current


def _is_graph_link(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
        and not isinstance(value[1], bool)
    )


def _validate_pointer_type(value: Any, declaration: Mapping[str, Any], context: str) -> None:
    declared_type = declaration.get("type")
    if declared_type == "string" and not isinstance(value, str):
        raise WorkflowManifestError(f"{context} pointer type is {_json_type(value)}, expected string")
    if declared_type == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
        raise WorkflowManifestError(f"{context} pointer type is {_json_type(value)}, expected integer")
    if declared_type == "number":
        _require_number(value, f"{context} pointer value")
    if declared_type == "boolean" and not isinstance(value, bool):
        raise WorkflowManifestError(f"{context} pointer type is {_json_type(value)}, expected boolean")
    if declared_type not in {"string", "integer", "number", "boolean"}:
        raise WorkflowManifestError(f"{context} declares unsupported pointer type")

    if declaration.get("integer_only") is True:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise WorkflowManifestError(f"{context} pointer value must be an integer")
        if isinstance(value, float) and not value.is_integer():
            raise WorkflowManifestError(f"{context} pointer value must be an integer")

    if "minimum" in declaration and value < declaration["minimum"]:
        raise WorkflowManifestError(f"{context} pointer value is below its minimum")
    if "maximum" in declaration and value > declaration["maximum"]:
        raise WorkflowManifestError(f"{context} pointer value is above its maximum")
    if "enum" in declaration and not any(_json_equal(value, option) for option in declaration["enum"]):
        raise WorkflowManifestError(f"{context} pointer value is outside its enum")


def _validate_parameter_declarations(manifest: dict[str, Any]) -> None:
    parameters = _require_object(manifest["parameters"], "parameters")
    _require_keys(parameters, set(_EXPECTED_PARAMETERS), "parameters")
    for name, declaration_value in parameters.items():
        declaration = _require_object(declaration_value, f"parameters.{name}")
        expected_keys = set(_EXPECTED_PARAMETERS[name])
        _require_keys(declaration, expected_keys, f"parameters.{name}")
        _require_text(declaration.get("pointer"), f"parameters.{name}.pointer")
        declared_type = declaration.get("type")
        if not isinstance(declared_type, str):
            raise WorkflowManifestError(f"parameters.{name}.type must be text")
        if declared_type not in {"string", "integer", "number", "boolean"}:
            raise WorkflowManifestError(f"parameters.{name}.type is unsupported")
        if "required" in declaration and not isinstance(declaration["required"], bool):
            raise WorkflowManifestError(f"parameters.{name}.required must be boolean")
        if "integer_only" in declaration and not isinstance(declaration["integer_only"], bool):
            raise WorkflowManifestError(f"parameters.{name}.integer_only must be boolean")
        for bound_name in ("minimum", "maximum"):
            if bound_name in declaration:
                _require_int(declaration[bound_name], f"parameters.{name}.{bound_name}")
        if "minimum" in declaration and "maximum" in declaration:
            if declaration["minimum"] > declaration["maximum"]:
                raise WorkflowManifestError(f"parameters.{name} has reversed bounds")
        if "enum" in declaration:
            if not isinstance(declaration["enum"], list) or not declaration["enum"]:
                raise WorkflowManifestError(f"parameters.{name}.enum must be a non-empty list")
        _require_exact(declaration, _EXPECTED_PARAMETERS[name], f"parameters.{name}")


def _validate_static_manifest(manifest: dict[str, Any]) -> None:
    _require_keys(
        manifest,
        {
            "schema",
            "id",
            "version",
            "display_name",
            "provider_id",
            "prompt_policy",
            "runtime_compatibility",
            "workflow",
            "capabilities",
            "parameters",
            "fixed_assertions",
            "models",
            "required_classes",
            "schema_fingerprint",
            "output",
            "policy",
        },
        "manifest",
    )
    _require_exact(manifest["schema"], SCHEMA, "manifest.schema")
    _require_exact(manifest["id"], TARGET_ID, "manifest.id")
    _require_exact(manifest["version"], 1, "manifest.version")
    _require_exact(manifest["display_name"], "MiniMax H3 · Text to Video", "manifest.display_name")
    _require_exact(manifest["provider_id"], PROVIDER_ID, "manifest.provider_id")
    _require_exact(manifest["prompt_policy"], "literal", "manifest.prompt_policy")

    runtime = _require_object(manifest["runtime_compatibility"], "runtime_compatibility")
    _require_keys(runtime, set(_EXPECTED_RUNTIME), "runtime_compatibility")
    _require_exact(runtime, _EXPECTED_RUNTIME, "runtime_compatibility")

    workflow = _require_object(manifest["workflow"], "workflow")
    _require_keys(workflow, set(_EXPECTED_WORKFLOW), "workflow")
    _require_text(workflow["api_file"], "workflow.api_file")
    _require_text(workflow["editor_file"], "workflow.editor_file")
    _require_sha256(workflow["api_sha256"], "workflow.api_sha256")
    _require_sha256(workflow["editor_sha256"], "workflow.editor_sha256")
    _require_int(workflow["node_count"], "workflow.node_count")
    _require_exact(workflow, _EXPECTED_WORKFLOW, "workflow")

    capabilities = _require_object(manifest["capabilities"], "capabilities")
    _require_keys(capabilities, set(_EXPECTED_CAPABILITIES), "capabilities")
    _require_exact(capabilities, _EXPECTED_CAPABILITIES, "capabilities")

    _validate_parameter_declarations(manifest)

    fixed = _require_object(manifest["fixed_assertions"], "fixed_assertions")
    _require_exact(fixed, _EXPECTED_FIXED_ASSERTIONS, "fixed_assertions")

    models = _require_object(manifest["models"], "models")
    _require_keys(models, set(_EXPECTED_MODELS), "models")
    for model_name, declaration_value in models.items():
        declaration = _require_object(declaration_value, f"models.{model_name}")
        _require_keys(declaration, {"folder", "filename", "assertion_pointer"}, f"models.{model_name}")
        _require_text(declaration["folder"], f"models.{model_name}.folder")
        _require_text(declaration["filename"], f"models.{model_name}.filename")
        _require_text(declaration["assertion_pointer"], f"models.{model_name}.assertion_pointer")
        _require_exact(declaration, _EXPECTED_MODELS[model_name], f"models.{model_name}")

    required_classes = manifest["required_classes"]
    if not isinstance(required_classes, list) or not all(isinstance(item, str) for item in required_classes):
        raise WorkflowManifestError("required_classes must be a list of text")
    _require_exact(required_classes, _EXPECTED_REQUIRED_CLASSES, "required_classes")

    output = _require_object(manifest["output"], "output")
    _require_keys(output, set(_EXPECTED_OUTPUT), "output")
    _require_exact(output, _EXPECTED_OUTPUT, "output")

    policy = _require_object(manifest["policy"], "policy")
    _require_keys(policy, set(_EXPECTED_POLICY), "policy")
    _require_exact(policy, _EXPECTED_POLICY, "policy")

    fingerprint = _require_object(manifest["schema_fingerprint"], "schema_fingerprint")
    _require_keys(fingerprint, {"nodes"}, "schema_fingerprint")
    nodes = _require_object(fingerprint["nodes"], "schema_fingerprint.nodes")
    if sorted(nodes) != sorted(manifest["required_classes"]):
        raise WorkflowManifestError("schema_fingerprint must name exactly the required classes")


def _validate_parameter_pointers(manifest: dict[str, Any], graph: dict[str, Any]) -> None:
    parameters = manifest["parameters"]
    for name, declaration in parameters.items():
        pointer = declaration["pointer"]
        value = _resolve_pointer(graph, pointer, f"parameters.{name}")
        if _is_graph_link(value):
            raise WorkflowManifestError(f"parameters.{name} pointer identifies a graph link")
        _validate_pointer_type(value, declaration, f"parameters.{name}")


def _validate_manifest_and_graph(manifest: dict[str, Any], graph: dict[str, Any]) -> None:
    _validate_parameter_pointers(manifest, graph)

    fixed = manifest["fixed_assertions"]
    for pointer, expected_value in fixed.items():
        actual_value = _resolve_pointer(graph, pointer, "fixed assertion")
        _require_exact(actual_value, expected_value, f"fixed assertion {pointer}")

    models = manifest["models"]
    for model_name, declaration in models.items():
        pointer = declaration["assertion_pointer"]
        actual_filename = _resolve_pointer(graph, pointer, f"models.{model_name}.assertion_pointer")
        if not isinstance(actual_filename, str):
            raise WorkflowManifestError(f"models.{model_name} assertion pointer must identify text")
        if actual_filename != declaration["filename"]:
            raise WorkflowManifestError(f"models.{model_name} assertion does not match its filename")

    required_classes = manifest["required_classes"]
    graph_classes: set[str] = set()
    for node_id, node in graph.items():
        if not isinstance(node_id, str) or not isinstance(node, dict):
            raise WorkflowManifestError("workflow API graph must contain object nodes")
        class_type = node.get("class_type")
        if not isinstance(class_type, str):
            raise WorkflowManifestError(f"workflow API node {node_id} has no class_type")
        graph_classes.add(class_type)
    missing_classes = sorted(set(required_classes) - graph_classes)
    if missing_classes:
        raise WorkflowManifestError(f"required_classes missing from graph: {', '.join(missing_classes)}")
    _require_exact(manifest["workflow"]["node_count"], len(graph), "workflow.node_count")

    output = manifest["output"]
    if output["node_id"] not in graph:
        raise WorkflowManifestError("output node_id does not exist in graph")


@dataclass(frozen=True)
class WorkflowProfile:
    """Validated profile with private JSON sources and copy-on-read accessors."""

    target_id: str
    provider_id: str
    manifest_path: Path
    api_path: Path
    editor_path: Path
    license_path: Path
    _manifest_json: str
    _graph_json: str
    _validation_token: object | None = field(default=None, repr=False, compare=False)

    @property
    def id(self) -> str:
        return self.target_id

    @property
    def workflow_id(self) -> str:
        return self.target_id

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(self._manifest_json, object_pairs_hook=_object_no_duplicates)

    @property
    def graph(self) -> dict[str, Any]:
        return json.loads(self._graph_json, object_pairs_hook=_object_no_duplicates)

    @property
    def api_graph(self) -> dict[str, Any]:
        return self.graph

    def compile_base_graph(self) -> dict[str, Any]:
        if self._validation_token is not _PROFILE_VALIDATION_TOKEN:
            raise WorkflowManifestError("compile_base_graph requires a validated workflow profile")
        return self.graph


def load_workflow_profile(
    target_id: str = TARGET_ID,
    workflow_root: os.PathLike[str] | str | None = None,
) -> WorkflowProfile:
    """Load one pinned workflow profile below ``workflow_root``.

    If no root is supplied, only the installed-plugin location beside this
    module is used.  There is intentionally no repository fallback.
    """

    try:
        # Accept the convenient load_workflow_profile(root) form while keeping
        # target_id first for callers that select among registered targets.
        if isinstance(target_id, os.PathLike):
            if workflow_root is None:
                workflow_root = target_id
                target_id = TARGET_ID
            elif isinstance(workflow_root, str):
                target_id, workflow_root = workflow_root, target_id
        elif (
            workflow_root is None
            and isinstance(target_id, str)
            and target_id != TARGET_ID
            and (
                Path(target_id).exists()
                or "/" in target_id
                or "\\" in target_id
            )
        ):
            workflow_root = target_id
            target_id = TARGET_ID
        if target_id != TARGET_ID:
            raise WorkflowManifestError(f"unknown workflow target: {target_id!r}")
        root = _resolve_root(workflow_root)
        target_dir = _safe_file(root, target_id + "/manifest.json", "workflow manifest")
        manifest, manifest_raw = _read_json(target_dir, "workflow manifest")
        if hashlib.sha256(manifest_raw).hexdigest() != _EXPECTED_MANIFEST_SHA256:
            raise WorkflowManifestError("workflow manifest bytes do not match pinned manifest SHA-256")
        _validate_static_manifest(manifest)
        workflow = manifest["workflow"]

        api_path = _safe_file(root, f"{target_id}/{workflow['api_file']}", "workflow API file")
        editor_path = _safe_file(root, f"{target_id}/{workflow['editor_file']}", "workflow editor file")
        license_path = _safe_file(root, f"{target_id}/{workflow['license_file']}", "workflow license file")

        api_raw = _read_bytes(api_path, "workflow API file")
        editor_raw = _read_bytes(editor_path, "workflow editor file")
        license_raw = _read_bytes(license_path, "workflow license file")

        declared_api_hash = workflow["api_sha256"]
        declared_editor_hash = workflow["editor_sha256"]
        actual_api_hash = hashlib.sha256(api_raw).hexdigest()
        actual_editor_hash = hashlib.sha256(editor_raw).hexdigest()
        actual_license_hash = hashlib.sha256(license_raw).hexdigest()
        if declared_api_hash != actual_api_hash or declared_api_hash != _EXPECTED_API_SHA256:
            raise WorkflowManifestError("workflow API file does not match pinned api_sha256")
        if declared_editor_hash != actual_editor_hash or declared_editor_hash != _EXPECTED_EDITOR_SHA256:
            raise WorkflowManifestError("workflow editor file does not match pinned editor_sha256")
        if actual_license_hash != _EXPECTED_LICENSE_SHA256:
            raise WorkflowManifestError("workflow license hash does not match approved content")

        try:
            graph = json.loads(
                api_raw.decode("utf-8"),
                object_pairs_hook=_object_no_duplicates,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateKeyError, TypeError, ValueError):
            raise WorkflowManifestError("workflow API file contains malformed JSON") from None
        if not isinstance(graph, dict):
            raise WorkflowManifestError("workflow API graph must be a JSON object")

        _validate_manifest_and_graph(manifest, graph)
        manifest_json = json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        graph_json = json.dumps(graph, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        return WorkflowProfile(
            target_id=target_id,
            provider_id=manifest["provider_id"],
            manifest_path=target_dir,
            api_path=api_path,
            editor_path=editor_path,
            license_path=license_path,
            _manifest_json=manifest_json,
            _graph_json=graph_json,
            _validation_token=_PROFILE_VALIDATION_TOKEN,
        )
    except WorkflowManifestError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError, KeyError):
        raise WorkflowManifestError("workflow profile could not be loaded") from None


def load_workflow_manifest(
    target_id: str = TARGET_ID,
    workflow_root: os.PathLike[str] | str | None = None,
) -> WorkflowProfile:
    """Compatibility name for :func:`load_workflow_profile`."""

    return load_workflow_profile(target_id=target_id, workflow_root=workflow_root)


def compile_base_graph(profile: WorkflowProfile) -> dict[str, Any]:
    """Return a fresh API graph copy from a validated profile."""

    if not isinstance(profile, WorkflowProfile) or profile._validation_token is not _PROFILE_VALIDATION_TOKEN:
        raise WorkflowManifestError("compile_base_graph requires a validated workflow profile")
    try:
        return profile.compile_base_graph()
    except (TypeError, ValueError, json.JSONDecodeError):
        raise WorkflowManifestError("validated workflow graph could not be copied") from None


SAMPLING_V2_TARGET_ID = "minimax-h3-t2v-sampling-v2"
_SAMPLING_V2_MEGAPIXELS = list(H3_MP.options(SAMPLING_V2_TARGET_ID))
_SAMPLING_V2_CONTROLS = {
    "steps": {"pointer": "/105:9/inputs/steps", "minimum": 1, "maximum": 100},
    "sampler_name": {"pointer": "/105:17/inputs/sampler_name", "enum": ["res_multistep", "euler"]},
    "scheduler": {"pointer": "/105:9/inputs/scheduler", "enum": ["simple", "beta"]},
    "megapixels": {"pointer": "/115/inputs/megapixels", "enum": _SAMPLING_V2_MEGAPIXELS},
}


class SamplingV2Profile:
    """Versioned projection of the validated v1 profile (same pinned graph).

    ``minimax-h3-t2v`` stays byte-stable: the v2 target reuses its manifest,
    graph and assertions and only adds the ``sampling`` declarations (steps,
    sampler, scheduler) plus the Draft / Native megapixel enum (h3_megapixels).  ``manifest`` returns a
    fresh copy on every read, like :class:`WorkflowProfile`.
    """

    __slots__ = ("_manifest_json",)

    def __init__(self, base: WorkflowProfile) -> None:
        if not isinstance(base, WorkflowProfile) or base._validation_token is not _PROFILE_VALIDATION_TOKEN:
            raise WorkflowManifestError("the sampling v2 projection requires a validated workflow profile")
        manifest = base.manifest
        manifest["id"] = SAMPLING_V2_TARGET_ID
        manifest["parameters"]["megapixels"]["enum"] = list(_SAMPLING_V2_MEGAPIXELS)
        manifest["sampling"] = json.loads(json.dumps(_SAMPLING_V2_CONTROLS))
        self._manifest_json = json.dumps(manifest, ensure_ascii=False)

    @property
    def target_id(self) -> str:
        return SAMPLING_V2_TARGET_ID

    @property
    def id(self) -> str:
        return SAMPLING_V2_TARGET_ID

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(self._manifest_json)


def project_sampling_v2(profile: WorkflowProfile) -> SamplingV2Profile:
    """Return the ``minimax-h3-t2v-sampling-v2`` view of a validated v1 profile."""

    return SamplingV2Profile(profile)


__all__ = [
    "TARGET_ID",
    "SAMPLING_V2_TARGET_ID",
    "SamplingV2Profile",
    "project_sampling_v2",
    "SCHEMA",
    "PROVIDER_ID",
    "WorkflowManifestError",
    "WorkflowProfile",
    "load_workflow_profile",
    "load_workflow_manifest",
    "compile_base_graph",
]
