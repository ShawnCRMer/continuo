"""Provider-neutral generation contracts for the Studio backend.

The contracts in this module are deliberately independent of FastAPI, Pydantic,
and any provider client.  They validate the boundary data that later provider
adapters will exchange and provide public/backend serialization as separate
operations so vendor diagnostics cannot accidentally become UI payloads.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


GENERATION_REQUEST_SCHEMA = "studio.generation-request/1"
# Keys ComfyUI's asset/media enrichment adds to a history output descriptor since its October 2026
# assets work (comfy_execution/asset_enrichment.py: id, preview_id; media_enrichment.py: metadata).
# They describe the file, never locate it: adapters ignore them and keep validating filename,
# subfolder and type strictly. Rejecting them quarantined every finished picture (output_invalid).
COMFY_ENRICHMENT_KEYS = frozenset({"id", "preview_id", "metadata"})
NORMALIZED_STATUSES = frozenset({"queued", "running", "ready", "failed", "moderated", "cancelled"})
HEALTH_STATES = frozenset({"ready_static", "unreachable", "unverified_runtime", "incompatible", "misconfigured"})
STATIC_HEALTH_CHECKS = frozenset(
    {
        "reachable",
        "runtime_version",
        "required_classes",
        "input_schema",
        "required_models",
        "workflow_integrity",
    }
)
_STATUS_MAX_DEPTH = 32
_PUBLIC_ROW_CONTRACT = 1
_PUBLIC_TEXT_MAX_CHARS = 256
_PUBLIC_PROMPT_MAX_BYTES = 256 * 1024
_MAX_PUBLIC_JSON_DEPTH = 64
_PUBLIC_SCHEMES = ("http:", "https:", "file:", "data:", "ftp:", "ws:", "wss:", "mailto:", "urn:")
_PRIVATE_KEY_TEXT = re.compile(r"(?i)(?:private[\s_-]*key|api[\s_-]*key|access[\s_-]*key)")
_FLUX_ASPECT_RATIOS = frozenset({"auto", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "9:21"})
_H3_ASPECT_RATIOS = frozenset(
    {
        "21:9 (Ultrawide)",
        "16:9 (Widescreen)",
        "4:3 (Standard)",
        "1:1 (Square)",
        "3:4 (Portrait Standard)",
        "9:16 (Portrait Widescreen)",
    }
)
_PUBLIC_ASPECT_RATIOS = _FLUX_ASPECT_RATIOS | _H3_ASPECT_RATIOS
_PUBLIC_RESOLUTIONS = frozenset({"720p"})
_HEALTH_POLICIES = frozenset({"static_only"})
_PUBLIC_CREATIVE_KEYS = frozenset(
    {
        "prompt",
        "aspect_ratio",
        "duration",
        "duration_seconds",
        "resolution",
        "megapixels",
        "steps",
        "sampler_name",
        "scheduler",
        "generate_audio",
        "grounding",
        "seed",
        "version",
        "keyframe_indices",
    }
)

# Public receipt fields are intentionally a narrow projection.  These keys are
# reserved for backend diagnostics or path-bearing execution state and are
# rejected anywhere below result, requested, or effective.  This is an exact
# normalized-key set, not a substring search: ordinary prose such as
# ``workflow details`` remains valid when it is a value or an unrelated key.
_PRIVATE_PUBLIC_KEYS = frozenset(
    {
        "diagnostic",
        "diagnostic_detail",
        "raw",
        "raw_status",
        "raw_detail",
        "raw_error",
        "raw_errors",
        "vendor",
        "backend",
        "traceback",
        "stack",
        "exception",
        "node_error",
        "node_errors",
        "node_id",
        "node_ids",
        "errors",
        "message",
        "detail",
        "details",
        "status_message",
        "provider_message",
        "provider_detail",
        "raw_message",
        "error",
        "failure",
        "reason",
        "log",
        "logs",
        "status_str",
        "save_to",
        "approved_roots",
        "input_image",
        "input_images",
        "input_video",
        "workflow",
        "workflow_path",
        "workflow_paths",
        "manifest",
        "manifest_path",
        "json_pointer",
        "json_pointers",
        "download_url",
        "download_urls",
        "model_path",
        "model_paths",
        "path",
        "paths",
        "destination",
        "destination_path",
        "project_ref",
        "project_directory",
        "private_record_path",
        "frozen_destination",
        "published_path",
        "publication_path",
        "output_path",
        "request_json",
        "request_hash",
    }
)

_CAPABILITY_CONTROL_KEYS = frozenset(
    {"id", "kind", "required", "min", "max", "step", "options", "default", "prompt_policy"}
)
_CAPABILITY_OUTPUT_KEYS = frozenset({"kind", "resolution", "audio", "fps"})
_CAPABILITY_ADMISSION_KEYS = frozenset({"max_active_studio_jobs", "min_submit_interval_seconds"})
_CAPABILITY_TARGET_KEYS = frozenset(
    {"id", "label", "input_kind", "controls", "output", "admission", "health_policy"}
)
_CAPABILITY_CONTROL_KINDS = frozenset(
    {"textarea", "integer", "select", "boolean", "image", "keyframes", "video", "audio", "informational"}
)
_PUBLIC_RESULT_KEYS = frozenset({"saved", "artifact", "description"})
_PUBLIC_ARTIFACT_KEYS = frozenset({"labels"})
_PUBLIC_EFFECTIVE_KEYS = frozenset({"fps", "frames", "duration_seconds"})

REASON_CODES = frozenset(
    {
        "unreachable",
        "unverified_runtime",
        "missing_node",
        "incompatible_node_schema",
        "missing_model",
        "workflow_drift",
        "invalid_request",
        "provider_busy",
        "rate_limited",
        "provider_refused",
        "auth_required",
        "submission_unknown",
        "job_lost",
        "execution_failed",
        "output_missing",
        "output_invalid",
        "save_failed",
        "moderated",
        "cancelled",
        "unknown",
    }
)

# Keep this mapping finite and intentionally free of backend/vendor wording.
PUBLIC_REASON_MESSAGES = MappingProxyType(
    {
        "unreachable": "Provider unavailable",
        "unverified_runtime": "Runtime not verified",
        "missing_node": "Required node missing",
        "incompatible_node_schema": "Provider input schema is incompatible",
        "missing_model": "Required model missing",
        "workflow_drift": "Workflow integrity check failed",
        "invalid_request": "Generation request is invalid",
        "provider_busy": "Provider is busy",
        "rate_limited": "Generation is rate-limited",
        "provider_refused": "Provider refused the generation",
        "auth_required": "Provider authentication is required",
        "submission_unknown": "Submission status is unknown",
        "job_lost": "Generation job was lost",
        "execution_failed": "Generation failed",
        "output_missing": "Generation output is missing",
        "output_invalid": "Generation output is invalid",
        "save_failed": "Saving the generation failed",
        "moderated": "Generation was moderated",
        "cancelled": "Generation was cancelled",
        "unknown": "Unknown generation error",
    }
)

# Lane p12-shl (BCK-023; R3 "finite named reason codes per cause"): a /submit refused BEFORE a ledger row exists
# names its own cause. These codes never enter the Studio ledger (its SQLite CHECK is built from REASON_CODES);
# the plain text comes from the code, never from matching an English message.
REFUSAL_REASON_MESSAGES = MappingProxyType(
    {
        "output_folder_missing": "No output folder",
        "output_name_missing": "Output file name missing",
        "output_name_invalid": "File name not a valid MP4 name",
        "folders_changed": "Media folders changed",
        "output_folder_not_approved": "Output folder not approved",
        "folders_invalid": "Approved media folders invalid",
        "project_mismatch": "The take does not match the open project",
        "project_folder_missing": "The project folder was not found",
        "project_unavailable": "The project folder is unavailable right now",
        "path_refused": "A chosen folder or file cannot be used",
        "unknown_target": "This provider does not offer that model",
        "request_id_missing": "Each take needs an id",
        "request_id_conflict": "This id was already used for another take",
        "payload_invalid": "Some settings are missing or not valid",
        "settings_rejected": "The model did not accept these settings",
        "save_to_not_folder": "The output must be a folder, not a file",  # lane p13-prj item 1
        "project_folder_offline": "A project folder is offline",  # lane p13-prjb finding 1 (the route names the folder)
        "project_rebase_pending": "Moving this project is not finished",  # lane p13-prjc finding 1 (names the folder)
    }
)
REFUSAL_REASON_CODES = frozenset(REFUSAL_REASON_MESSAGES)


def _freeze_json(
    value: Any,
    field_name: str,
    *,
    reject_private_keys: bool = False,
    _active: set[int] | None = None,
    _depth: int = 0,
) -> Any:
    """Validate a JSON value and store it in recursively immutable form."""
    if _depth > _MAX_PUBLIC_JSON_DEPTH:
        raise ValueError(f"{field_name} exceeds the maximum JSON depth")
    active = set() if _active is None else _active
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must contain only finite JSON numbers")
        return value
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in active:
            raise ValueError(f"{field_name} must not contain cyclic JSON")
        active.add(marker)
        try:
            frozen: dict[str, Any] = {}
            try:
                items = value.items()
                for key, nested in items:
                    # Validate primitive keys before membership, casefold, or
                    # insertion into the frozen dict.
                    if type(key) is not str:
                        raise ValueError(f"{field_name} keys must be primitive text")
                    if reject_private_keys and key.casefold() in _PRIVATE_PUBLIC_KEYS:
                        raise ValueError(f"{field_name} contains a reserved private key")
                    frozen[key] = _freeze_json(
                        nested,
                        f"{field_name}.{key}",
                        reject_private_keys=reject_private_keys,
                        _active=active,
                        _depth=_depth + 1,
                    )
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError(f"{field_name} must be a finite JSON object") from exc
            return MappingProxyType(frozen)
        finally:
            active.remove(marker)
    if type(value) in (list, tuple):
        marker = id(value)
        if marker in active:
            raise ValueError(f"{field_name} must not contain cyclic JSON")
        active.add(marker)
        try:
            return tuple(
                _freeze_json(
                    item,
                    f"{field_name}[{index}]",
                    reject_private_keys=reject_private_keys,
                    _active=active,
                    _depth=_depth + 1,
                )
                for index, item in enumerate(value)
            )
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"{field_name} must be a finite JSON list") from exc
        finally:
            active.remove(marker)
    raise ValueError(f"{field_name} must contain only JSON values")


def _json_copy(value: Any, field_name: str, *, reject_private_keys: bool = False) -> Any:
    """Validate a JSON value recursively and return immutable storage."""
    return _freeze_json(value, field_name, reject_private_keys=reject_private_keys)


def _thaw_json(value: Any) -> Any:
    """Return a fresh mutable JSON-compatible copy for serialization."""
    if isinstance(value, Mapping):
        return {key: _thaw_json(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be non-empty text")
    try:
        if not value.strip():
            raise ValueError(f"{field_name} must be non-empty text")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"{field_name} must be non-empty text") from exc
    return value


def _optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _text(value, field_name)


def _bounded_text(value: Any, field_name: str, *, max_chars: int = 256) -> str:
    text = _text(value, field_name)
    try:
        length = len(text)
    except Exception as exc:
        raise ValueError(f"{field_name} must be non-empty text") from exc
    if length > max_chars:
        raise ValueError(f"{field_name} exceeds {max_chars} characters")
    return text


def _primitive_text(value: Any, field_name: str, *, max_chars: int = _PUBLIC_TEXT_MAX_CHARS) -> str:
    """Validate public text without invoking overridden str-subclass methods."""
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be primitive non-empty text")
    if len(value) > max_chars:
        raise ValueError(f"{field_name} exceeds {max_chars} characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field_name} must be valid UTF-8") from exc
    return value


def _mapping(value: Any, field_name: str, *, reject_private_keys: bool = False) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return _json_copy(value, field_name, reject_private_keys=reject_private_keys)


def _list(value: Any, field_name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a list")
    return _json_copy(list(value), field_name)


def _object_list(value: Any, field_name: str) -> tuple[Mapping[str, Any], ...]:
    items = _list(value, field_name)
    if not all(isinstance(item, Mapping) for item in items):
        raise ValueError(f"{field_name} must contain only objects")
    return items


def _capability_mapping(value: Any, allowed: frozenset[str], field_name: str) -> dict[str, Any]:
    """Copy one capability object and reject every non-catalog key."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    copied = _thaw_json(_json_copy(value, field_name, reject_private_keys=True))
    unknown = set(copied) - allowed
    if unknown:
        raise ValueError(f"{field_name} contains unknown public capability fields")
    return copied


def _capability_text(value: Any, field_name: str) -> str:
    """Validate path-free primitive public capability text."""
    text = _primitive_text(value, field_name)
    if text != text.strip() or "\x00" in text or "/" in text or "\\" in text:
        raise ValueError(f"{field_name} must be path-free text")
    lowered = text.casefold()
    if lowered.startswith(_PUBLIC_SCHEMES):
        raise ValueError(f"{field_name} must not contain a URL")
    return text


def _capability_label(value: Any, field_name: str) -> str:
    """Validate a catalog label; only the default empty label is permitted."""
    if type(value) is not str:
        raise ValueError(f"{field_name} must be primitive text")
    if value == "":
        return value
    return _capability_text(value, field_name)


def _public_id(value: Any, field_name: str) -> str:
    """Validate an opaque public ID without treating colon IDs as URLs."""
    return _capability_text(value, field_name)


def _public_result_text(value: Any, field_name: str) -> str:
    text = _capability_text(value, field_name)
    if _PRIVATE_KEY_TEXT.search(text):
        raise ValueError(f"{field_name} must not contain private key material")
    return text


def _generation_id(value: Any, field_name: str) -> str:
    """Validate request IDs without moving JobStore's subclass boundary earlier."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be non-empty text")
    try:
        primitive = str(value)
    except Exception as exc:
        raise ValueError(f"{field_name} must be non-empty text") from exc
    _public_id(primitive, field_name)
    return value


def _capability_scalar(value: Any, field_name: str) -> Any:
    if type(value) is str:
        return _capability_text(value, field_name)
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite")
        return value
    raise ValueError(f"{field_name} must be a finite primitive value")


def _capability_number(value: Any, field_name: str, *, integer: bool = False, minimum: int | float | None = None) -> int | float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number")
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number")
    if integer and type(value) is not int:
        raise ValueError(f"{field_name} must be a whole number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}")
    return value


def _capability_control(value: Any, field_name: str) -> Mapping[str, Any]:
    control = _capability_mapping(value, _CAPABILITY_CONTROL_KEYS, field_name)
    if "id" not in control or "kind" not in control:
        raise ValueError(f"{field_name} requires id and kind")
    control_id = _capability_text(control["id"], f"{field_name}.id")
    kind = _capability_text(control["kind"], f"{field_name}.kind")
    if kind not in _CAPABILITY_CONTROL_KINDS:
        raise ValueError(f"{field_name}.kind is not supported")
    normalized: dict[str, Any] = {"id": control_id, "kind": kind}
    if "required" in control:
        if type(control["required"]) is not bool:
            raise ValueError(f"{field_name}.required must be boolean")
        normalized["required"] = control["required"]

    numeric: dict[str, int | float] = {}
    for name in ("min", "max", "step"):
        if name in control:
            numeric[name] = _capability_number(
                control[name],
                f"{field_name}.{name}",
                integer=kind == "integer",
            )
    if "min" in numeric and "max" in numeric and numeric["min"] > numeric["max"]:
        raise ValueError(f"{field_name}.min cannot exceed max")
    if "step" in numeric and numeric["step"] <= 0:
        raise ValueError(f"{field_name}.step must be positive")
    normalized.update(numeric)

    options: list[Any] | None = None
    if "options" in control:
        if kind != "select":
            raise ValueError(f"{field_name}.options requires a select control")
        raw_options = control["options"]
        if isinstance(raw_options, (str, bytes, bytearray)) or not isinstance(raw_options, Sequence):
            raise ValueError(f"{field_name}.options must be a list")
        options = [
            _capability_scalar(option, f"{field_name}.options[{index}]")
            for index, option in enumerate(raw_options)
        ]
        normalized["options"] = options

    if "default" in control:
        default = control["default"]
        if kind == "boolean":
            if type(default) is not bool:
                raise ValueError(f"{field_name}.default must be boolean")
        elif kind == "integer":
            default = _capability_number(default, f"{field_name}.default", integer=True)
        else:
            default = _capability_scalar(default, f"{field_name}.default")
        if options is not None and default not in options:
            raise ValueError(f"{field_name}.default must be one of options")
        normalized["default"] = default

    if "prompt_policy" in control:
        normalized["prompt_policy"] = _capability_text(control["prompt_policy"], f"{field_name}.prompt_policy")
    return _json_copy(normalized, field_name, reject_private_keys=True)


def _capability_controls(value: Any, field_name: str) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a list")
    return tuple(_capability_control(item, f"{field_name}[{index}]") for index, item in enumerate(value))


def _capability_output(value: Any, field_name: str) -> Mapping[str, Any]:
    output = _capability_mapping(value, _CAPABILITY_OUTPUT_KEYS, field_name)
    normalized: dict[str, Any] = {}
    for name, nested in output.items():
        if name == "fps":
            normalized[name] = _capability_number(nested, f"{field_name}.{name}", minimum=0)
        else:
            normalized[name] = _capability_text(nested, f"{field_name}.{name}")
    return _json_copy(normalized, field_name, reject_private_keys=True)


def _capability_admission(value: Any, field_name: str) -> Mapping[str, Any]:
    admission = _capability_mapping(value, _CAPABILITY_ADMISSION_KEYS, field_name)
    normalized: dict[str, Any] = {}
    if "max_active_studio_jobs" in admission:
        normalized["max_active_studio_jobs"] = _capability_number(
            admission["max_active_studio_jobs"],
            f"{field_name}.max_active_studio_jobs",
            integer=True,
            minimum=1,
        )
    if "min_submit_interval_seconds" in admission:
        normalized["min_submit_interval_seconds"] = _capability_number(
            admission["min_submit_interval_seconds"],
            f"{field_name}.min_submit_interval_seconds",
            minimum=0,
        )
    return _json_copy(normalized, field_name, reject_private_keys=True)


def _capability_target(value: Any, field_name: str) -> Mapping[str, Any]:
    target = _capability_mapping(value, _CAPABILITY_TARGET_KEYS, field_name)
    for name in ("id", "label", "input_kind"):
        if name not in target:
            raise ValueError(f"{field_name} requires {name}")
    normalized: dict[str, Any] = {
        "id": _capability_text(target["id"], f"{field_name}.id"),
        "label": _capability_text(target["label"], f"{field_name}.label"),
        "input_kind": _capability_text(target["input_kind"], f"{field_name}.input_kind"),
        "controls": _capability_controls(target.get("controls", []), f"{field_name}.controls"),
        "output": _capability_output(target.get("output", {}), f"{field_name}.output"),
        "admission": _capability_admission(target.get("admission", {}), f"{field_name}.admission"),
        "health_policy": None,
    }
    if target.get("health_policy") is not None:
        health_policy = _capability_text(target["health_policy"], f"{field_name}.health_policy")
        if health_policy not in _HEALTH_POLICIES:
            raise ValueError(f"{field_name}.health_policy is unsupported")
        normalized["health_policy"] = health_policy
    return _json_copy(normalized, field_name, reject_private_keys=True)


def _capability_targets(value: Any, field_name: str) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a list")
    return tuple(_capability_target(item, f"{field_name}[{index}]") for index, item in enumerate(value))


def _reason_code(code: Any) -> str:
    if type(code) is str:
        candidate = code.strip().casefold()
        if candidate in REASON_CODES:
            return candidate
    return "unknown"


def public_reason(code: Any) -> str:
    """Return finite user-facing language for a normalized reason code.

    Unknown codes, including arbitrary backend prose, intentionally collapse to
    the single public ``unknown`` message.
    """

    if type(code) is str and code in REFUSAL_REASON_MESSAGES:  # lane p12-shl: a refusal's own code
        return REFUSAL_REASON_MESSAGES[code]
    return PUBLIC_REASON_MESSAGES[_reason_code(code)]


def _status_text(raw_status: Any, *, _active: set[int] | None = None, _depth: int = 0) -> str | None:
    if type(raw_status) is str:
        try:
            return " ".join(raw_status.strip().casefold().replace("_", " ").replace("-", " ").split())
        except Exception:
            return None
    if not isinstance(raw_status, Mapping) or _depth >= _STATUS_MAX_DEPTH:
        return None

    active = set() if _active is None else _active
    marker = id(raw_status)
    if marker in active:
        return None
    active.add(marker)
    missing = object()
    try:
        # ComfyUI can report completed together with an error flag.  Inspect
        # failure flags before any status/success field so failure is terminal.
        for key in ("error", "failed", "failure"):
            try:
                value = raw_status.get(key, missing)
            except Exception:
                return None
            if value is True:
                return "error"
        # Prefer explicit status strings, including status_str == error, over
        # completed/success flags.  Nested details are bounded and cycle-safe.
        for key in ("status_str", "status", "state", "status_text", "message", "details", "data"):
            try:
                nested_value = raw_status.get(key, missing)
            except Exception:
                return None
            if nested_value is not missing:
                nested = _status_text(nested_value, _active=active, _depth=_depth + 1)
                if nested:
                    return nested
        try:
            interrupted = raw_status.get("interrupted", missing)
            cancelled = raw_status.get("cancelled", missing)
            completed = raw_status.get("completed", missing)
            success = raw_status.get("success", missing)
        except Exception:
            return None
        if interrupted is True or cancelled is True:
            return "interrupted"
        if completed is True or success is True:
            return "completed"
        return None
    except Exception:
        return None
    finally:
        active.remove(marker)


_MODERATED_ALIASES = frozenset({"moderated", "request moderated", "content moderated", "content policy"})
_CANCELLED_ALIASES = frozenset({"cancelled", "canceled", "interrupted", "aborted"})
_FAILED_ALIASES = frozenset({"error", "failed", "failure", "not found", "lost", "job lost"})
_READY_ALIASES = frozenset({"ready", "completed", "complete", "success", "succeeded", "finished", "done"})
_RUNNING_ALIASES = frozenset({"running", "generating", "executing", "processing", "in progress"})
_QUEUED_ALIASES = frozenset({"queued", "queue", "pending", "submitted", "accepted", "created", "waiting"})


def normalize_status(provider_id: str, raw_status: Any) -> str:
    """Normalize known FLUX/Comfy aliases into the six public states.

    Status aliases are explicit rather than substring-based: values such as
    ``incomplete`` and ``not ready`` are unknown and safely become ``failed``.
    Mapping failure flags take precedence over completed/success fields, and raw
    vendor values are never returned.
    """

    _public_id(provider_id, "provider_id")
    status = _status_text(raw_status)
    if not status:
        return "failed"

    if status in _MODERATED_ALIASES:
        return "moderated"
    if status in _CANCELLED_ALIASES:
        return "cancelled"
    if status in _FAILED_ALIASES:
        return "failed"
    if status in _READY_ALIASES:
        return "ready"
    if status in _RUNNING_ALIASES:
        return "running"
    if status in _QUEUED_ALIASES:
        return "queued"
    return "failed"


def _retry_after(value: Any) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError("retry_after must be a non-negative finite number")
    try:
        finite = math.isfinite(float(value))
    except OverflowError as exc:
        raise ValueError("retry_after must be a non-negative finite number") from exc
    if not finite:
        raise ValueError("retry_after must be a non-negative finite number")
    return value


def _optional_nonnegative_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative whole number")
    return value


@dataclass(frozen=True)
class ProviderCapabilities:
    """Public capability projection for one provider target."""

    provider_id: str
    label: str = ""
    target_id: str | None = None
    input_kind: str | None = None
    controls: list[dict[str, Any]] = field(default_factory=list)
    output: dict[str, Any] = field(default_factory=dict)
    admission: dict[str, Any] = field(default_factory=dict)
    endpoint_label: str | None = None
    health_policy: str | None = None
    targets: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        _capability_text(self.provider_id, "provider_id")
        _capability_label(self.label, "label")
        if self.target_id is not None:
            _capability_text(self.target_id, "target_id")
        if self.input_kind is not None:
            _capability_text(self.input_kind, "input_kind")
        if self.endpoint_label is not None:
            # The catalog explicitly permits the local endpoint label.  It is
            # not an execution path and is retained as a public field.
            _capability_text(self.endpoint_label, "endpoint_label")
        if self.health_policy is not None:
            health_policy = _capability_text(self.health_policy, "health_policy")
            if health_policy not in _HEALTH_POLICIES:
                raise ValueError("health_policy is unsupported")
        object.__setattr__(self, "controls", _capability_controls(self.controls, "controls"))
        object.__setattr__(self, "output", _capability_output(self.output, "output"))
        object.__setattr__(self, "admission", _capability_admission(self.admission, "admission"))
        object.__setattr__(self, "targets", _capability_targets(self.targets, "targets"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "target_id": self.target_id,
            "label": self.label,
            "input_kind": self.input_kind,
            "controls": _thaw_json(self.controls),
            "output": _thaw_json(self.output),
            "admission": _thaw_json(self.admission),
            "endpoint_label": self.endpoint_label,
            "health_policy": self.health_policy,
            "targets": _thaw_json(self.targets),
        }

    def to_json(self) -> str:
        return _dump(self.to_dict())

    as_dict = to_dict
    as_json = to_json


def _snapshot_mapping(value: Any, field_name: str) -> dict[str, Any]:
    """Copy mapping entries after validating primitive keys and safe iteration."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    try:
        items = list(value.items())
    except Exception as exc:
        raise ValueError(f"{field_name} must be a finite object") from exc
    result: dict[str, Any] = {}
    for key, nested in items:
        if type(key) is not str:
            raise ValueError(f"{field_name} keys must be primitive text")
        result[key] = nested
    return result


def _public_project_id(value: Any) -> str | None:
    if value is None:
        return None
    if type(value) is not str or len(value) != 36 or not value.startswith("prj_"):
        raise ValueError("project_id must be prj_<32 lowercase hex> or null")
    if any(character not in "0123456789abcdef" for character in value[4:]):
        raise ValueError("project_id must be prj_<32 lowercase hex> or null")
    return value


def _public_basename(value: Any) -> str:
    text = _primitive_text(value, "requested_basename")
    if (
        text in {".", ".."}
        or "\x00" in text
        or any(separator in text for separator in ("/", "\\", ":"))
    ):
        raise ValueError("requested_basename must be a path-free basename")
    return text


def _public_number(value: Any, field_name: str) -> int | float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number")
    try:
        if not math.isfinite(float(value)):
            raise ValueError(f"{field_name} must be a finite number")
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number") from exc
    return value


def _public_creative_value(key: str, value: Any) -> Any:
    if key == "prompt":
        prompt = _primitive_text(value, "requested.prompt", max_chars=_PUBLIC_PROMPT_MAX_BYTES)
        try:
            if len(prompt.encode("utf-8")) > _PUBLIC_PROMPT_MAX_BYTES:
                raise ValueError("requested.prompt exceeds 256 KiB")
        except UnicodeError as exc:
            raise ValueError("requested.prompt must be valid UTF-8") from exc
        return prompt
    if key == "duration":
        if type(value) is not int or not 5 <= value <= 20:
            raise ValueError("requested.duration must be a whole number from 5 to 20")
        return value
    if key == "duration_seconds":
        if type(value) is not int or not 4 <= value <= 15:
            raise ValueError("requested.duration_seconds must be a whole number from 4 to 15")
        return value
    if key == "aspect_ratio":
        text = _capability_text(value, "requested.aspect_ratio")
        if text not in _PUBLIC_ASPECT_RATIOS:
            raise ValueError("requested.aspect_ratio is not an approved preset")
        return text
    if key == "resolution":
        text = _capability_text(value, "requested.resolution")
        if text not in _PUBLIC_RESOLUTIONS:
            raise ValueError("requested.resolution is not an approved preset")
        return text
    if key == "steps":
        if type(value) is not int or not 1 <= value <= 100:
            raise ValueError("requested.steps must be a whole number from 1 to 100")
        return value
    if key == "sampler_name":
        if type(value) is not str or value not in {"res_multistep", "euler"}:
            raise ValueError("requested.sampler_name is not an approved preset")
        return value
    if key == "scheduler":
        if type(value) is not str or value not in {"simple", "beta"}:
            raise ValueError("requested.scheduler is not an approved preset")
        return value
    if key == "megapixels":
        if type(value) not in (int, float) or value not in {0.5, 1.0, 1.5, 2.0}:
            raise ValueError("requested.megapixels is not an approved preset")
        return float(value)
    if key == "version":
        text = _capability_text(value, "requested.version")
        return text
    if key in {"generate_audio", "grounding"}:
        if type(value) is not bool:
            raise ValueError(f"requested.{key} must be boolean")
        return value
    if key == "keyframe_indices":
        if type(value) is not list or len(value) > 10:
            raise ValueError("requested.keyframe_indices must be a bounded list")
        indices: list[int] = []
        for index in value:
            if type(index) is not int or isinstance(index, bool) or index < 0 or index > 480:
                raise ValueError("requested.keyframe_indices must contain bounded whole numbers")
            indices.append(index)
        return indices
    number = _public_number(value, f"requested.{key}")
    if abs(float(number)) > 2**53:
        raise ValueError(f"requested.{key} exceeds the public numeric bound")
    if key == "seed" and (type(number) is not int or number < 0 or number > 4294967295):
        raise ValueError("requested.seed must be a bounded whole number")
    return number


def _public_requested(
    source: Any,
    *,
    reject_unknown: bool = True,
    validate_values: bool = True,
) -> dict[str, Any] | None:
    if source is None:
        return None
    source_map = _snapshot_mapping(source, "requested")
    if "args" in source_map:
        wrapper_unknown = set(source_map) - {"schema", "args"}
        if wrapper_unknown and reject_unknown:
            raise ValueError("requested contains non-creative fields")
        if "schema" in source_map:
            schema = source_map["schema"]
            if type(schema) is not str or schema != GENERATION_REQUEST_SCHEMA:
                raise ValueError("requested schema is unsupported")
        args_value = source_map["args"]
    else:
        args_value = source_map
    args = _snapshot_mapping(args_value, "requested.args")
    requested: dict[str, Any] = {}
    for key, value in args.items():
        if key not in _PUBLIC_CREATIVE_KEYS:
            if reject_unknown:
                raise ValueError(f"requested.{key} is not public creative data")
            continue
        if not validate_values:
            if key in {"aspect_ratio", "resolution", "version"}:
                _capability_text(value, f"requested.{key}")
            requested[key] = value
            continue
        try:
            requested[key] = _public_creative_value(key, value)
        except ValueError:
            if reject_unknown:
                raise
            # The public job projection is a compatibility filter for ledger
            # rows; invalid optional creative values are omitted there.  A
            # ProviderReceipt uses strict mode and rejects construction.
            continue
    return requested or None


def _public_result(value: Any) -> Any:
    """Validate the finite public provider result schema."""
    if value is None:
        return None
    copied = _thaw_json(_json_copy(value, "result", reject_private_keys=True))
    if not isinstance(copied, dict):
        raise ValueError("result must be an object or null")
    unknown = set(copied) - _PUBLIC_RESULT_KEYS
    if unknown:
        raise ValueError("result contains unknown public fields")
    normalized: dict[str, Any] = {}
    if "saved" in copied:
        if type(copied["saved"]) is not bool:
            raise ValueError("result.saved must be boolean")
        normalized["saved"] = copied["saved"]
    if "description" in copied:
        normalized["description"] = _public_result_text(copied["description"], "result.description")
    if "artifact" in copied:
        artifact = copied["artifact"]
        if not isinstance(artifact, dict) or set(artifact) - _PUBLIC_ARTIFACT_KEYS:
            raise ValueError("result.artifact contains unknown public fields")
        if "labels" in artifact:
            labels = artifact["labels"]
            if type(labels) is not list or any(type(label) is not str for label in labels):
                raise ValueError("result.artifact.labels must be a list of primitive text")
            normalized["artifact"] = {
                "labels": [_public_result_text(label, "result.artifact.labels") for label in labels]
            }
        else:
            normalized["artifact"] = {}
    return _json_copy(normalized, "result", reject_private_keys=True)


def _public_effective(value: Any) -> Any:
    """Validate the finite public effective-output metadata schema."""
    if value is None:
        return None
    copied = _thaw_json(_json_copy(value, "effective", reject_private_keys=True))
    if not isinstance(copied, dict):
        raise ValueError("effective must be an object or null")
    unknown = set(copied) - _PUBLIC_EFFECTIVE_KEYS
    if unknown:
        raise ValueError("effective contains unknown public fields")
    normalized: dict[str, int | float] = {}
    for name, nested in copied.items():
        number = _public_number(nested, f"effective.{name}")
        if number < 0 or (name == "frames" and type(number) is not int):
            raise ValueError(f"effective.{name} is out of bounds")
        normalized[name] = number
    return _json_copy(normalized, "effective", reject_private_keys=True)


def public_job_projection(record: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Return the finite, path-free public §4.3 job projection.

    The ledger owns private destinations and publication paths.  This function
    exposes only the authenticated job identity, normalized state, bounded
    creative values, and non-sensitive timestamps; Task7 adds any scoped saved
    path later from its receipt/coordinator boundary.
    """
    if not isinstance(record, Mapping):
        try:
            to_dict = getattr(record, "to_dict", None)
            if not callable(to_dict):
                raise ValueError("record must be an object")
            record = to_dict()
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("record must be an object") from exc
    raw = _snapshot_mapping(record, "record")

    raw_id = raw.get("id", raw.get("studio_job_id"))
    job_id = _public_id(raw_id, "id")
    row_contract = raw.get("row_contract", _PUBLIC_ROW_CONTRACT)
    if type(row_contract) is not int or isinstance(row_contract, bool) or row_contract < 1:
        raise ValueError("row_contract must be a positive finite integer")
    provider_id = _public_id(raw.get("provider_id"), "provider_id")
    target_id = _public_id(raw.get("target_id"), "target_id")
    status_source = raw.get("status", raw.get("state"))
    status = normalize_status(provider_id, status_source)
    basename = _public_basename(raw.get("requested_basename"))

    result: dict[str, Any] = {
        "id": job_id,
        "row_contract": row_contract,
        "project_id": _public_project_id(raw.get("project_id")),
        "status": status,
        "provider_id": provider_id,
        "target_id": target_id,
        "requested_basename": basename,
    }
    for key in ("created_at", "updated_at", "queued_at", "running_at", "completed_at"):
        if key not in raw or raw[key] is None:
            continue
        try:
            result[key] = _public_number(raw[key], key)
        except ValueError:
            continue
    if "reason_code" in raw:
        result["reason_code"] = None if raw["reason_code"] is None else _reason_code(raw["reason_code"])
    source = raw.get("requested", raw.get("request"))
    requested = _public_requested(source, reject_unknown=False)
    if requested is not None:
        result["requested"] = requested
    return _thaw_json(_json_copy(result, "public_job"))


@dataclass(frozen=True)
class GenerationRequest:
    """Validated versioned request with the artist prompt kept byte-for-byte."""

    provider_id: str
    target_id: str
    args: dict[str, Any]
    schema: str = GENERATION_REQUEST_SCHEMA
    client_request_id: str | None = None
    save_to: str | None = None
    approved_roots: list[str] = field(default_factory=list)
    project_id: str | None = None

    def __post_init__(self) -> None:
        _generation_id(self.provider_id, "provider_id")
        _generation_id(self.target_id, "target_id")
        if type(self.schema) is not str or self.schema != GENERATION_REQUEST_SCHEMA:
            raise ValueError(f"schema must be {GENERATION_REQUEST_SCHEMA}")
        if self.client_request_id is not None:
            _bounded_text(self.client_request_id, "client_request_id")
        if self.save_to is not None:
            if type(self.save_to) is not str:
                raise ValueError("save_to must be non-empty text")
            _text(self.save_to, "save_to")

        args = _mapping(self.args, "args")
        prompt = args.get("prompt")
        if type(prompt) is not str or not prompt.strip():
            raise ValueError("args.prompt must be non-empty text")
        try:
            prompt_bytes = len(prompt.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise ValueError("args.prompt must be valid UTF-8") from exc
        if prompt_bytes > 256 * 1024:
            raise ValueError("args.prompt exceeds 256 KiB")
        object.__setattr__(self, "args", args)

        roots = _list(self.approved_roots, "approved_roots")
        if not all(type(root) is str and root.strip() for root in roots):
            raise ValueError("approved_roots must contain non-empty text")
        object.__setattr__(self, "approved_roots", roots)
        if self.project_id is not None:
            if (
                type(self.project_id) is not str
                or len(self.project_id) != 36
                or not self.project_id.startswith("prj_")
                or any(character not in "0123456789abcdef" for character in self.project_id[4:])
            ):
                raise ValueError("project_id must be prj_<32 lowercase hex> or null")

    @property
    def prompt(self) -> str:
        return self.args["prompt"]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "client_request_id": self.client_request_id,
            "provider_id": self.provider_id,
            "target_id": self.target_id,
            "args": _thaw_json(self.args),
            "save_to": self.save_to,
            "approved_roots": _thaw_json(self.approved_roots),
        }
        if self.project_id is not None:
            payload["project_id"] = self.project_id
        return payload

    def to_json(self) -> str:
        return _dump(self.to_dict())

    as_dict = to_dict
    as_json = to_json


@dataclass(frozen=True)
class ProviderReceipt:
    """Provider result with a public projection and private diagnostics."""

    provider_id: str
    target_id: str
    provider_job_id: str | None = None
    status: str = "failed"
    id: str | None = None
    client_request_id: str | None = None
    result: Any = None
    saved_path: str | None = None
    saved_bytes: int | None = None
    requested: dict[str, Any] | None = None
    effective: dict[str, Any] | None = None
    reason_code: str | None = None
    retry_after: int | float | None = None
    diagnostic_detail: Any = None

    def __post_init__(self) -> None:
        _public_id(self.provider_id, "provider_id")
        _public_id(self.target_id, "target_id")
        if self.provider_job_id is not None:
            _public_id(self.provider_job_id, "provider_job_id")
        if self.id is not None:
            _public_id(self.id, "id")
        if self.client_request_id is not None:
            _public_id(self.client_request_id, "client_request_id")
        object.__setattr__(self, "status", normalize_status(self.provider_id, self.status))
        if self.saved_path is not None:
            raise ValueError("saved_path is authenticated publication state, not a public receipt field")
        object.__setattr__(self, "saved_bytes", _optional_nonnegative_int(self.saved_bytes, "saved_bytes"))
        if self.requested is not None:
            object.__setattr__(
                self,
                "requested",
                _json_copy(
                    _public_requested(self.requested, reject_unknown=True, validate_values=True),
                    "requested",
                    reject_private_keys=True,
                ),
            )
        if self.effective is not None:
            object.__setattr__(self, "effective", _public_effective(self.effective))
        if self.result is not None:
            object.__setattr__(self, "result", _public_result(self.result))
        if self.diagnostic_detail is not None:
            object.__setattr__(self, "diagnostic_detail", _json_copy(self.diagnostic_detail, "diagnostic_detail"))
        if self.reason_code is not None:
            object.__setattr__(self, "reason_code", _reason_code(self.reason_code))
        object.__setattr__(self, "retry_after", _retry_after(self.retry_after))

    def to_dict(self) -> dict[str, Any]:
        details: dict[str, Any] = {
            "id": self.id,
            "provider_job_id": self.provider_job_id,
            "provider_id": self.provider_id,
            "target_id": self.target_id,
            "status": self.status,
            "saved_path": self.saved_path,
            "saved_bytes": self.saved_bytes,
            "requested": _thaw_json(self.requested),
            "effective": _thaw_json(self.effective),
        }
        if self.client_request_id is not None:
            details["client_request_id"] = self.client_request_id
        return {
            "result": _thaw_json(self.result),
            "details": details,
            "reason_code": self.reason_code,
            "retry_after": self.retry_after,
        }

    def to_backend_dict(self) -> dict[str, Any]:
        return {
            "public": self.to_dict(),
            "diagnostic_detail": _thaw_json(self.diagnostic_detail),
        }

    def to_json(self) -> str:
        return _dump(self.to_dict())

    def to_backend_json(self) -> str:
        return _dump(self.to_backend_dict())

    as_dict = to_dict
    as_json = to_json


_HEALTH_RUNTIME_KEYS = frozenset(
    {"comfyui_version", "required_frontend_version", "installed_templates_version", "required_templates_version"}
)
_HEALTH_VERSION_TEXT = re.compile(r"^[0-9A-Za-z][0-9A-Za-z.+_-]{0,39}$")
_HEALTH_DETAIL_TEXT = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.+-]{0,79}$")


@dataclass(frozen=True)
class StaticHealth:
    """Truthful static dependency result; it never claims inference was tested."""

    provider_id: str
    target_id: str
    state: str
    checks: dict[str, bool] = field(default_factory=dict)
    inference_tested: bool = False
    reason_code: str | None = None
    diagnostic_detail: Any = None
    # Pinned local targets (runtime policy ``schema_verified``): whether the live
    # runtime is one we (or a published take on this install) tested, the live
    # versions, and the node class + input that drifted.  Absent elsewhere.
    runtime_tested: bool | None = None
    runtime: Any = None
    reason_detail: str | None = None

    def __post_init__(self) -> None:
        _public_id(self.provider_id, "provider_id")
        _public_id(self.target_id, "target_id")
        _primitive_text(self.state, "state")
        if self.runtime_tested is not None and type(self.runtime_tested) is not bool:
            raise ValueError("runtime_tested must be boolean")
        if self.runtime is not None:
            if not isinstance(self.runtime, Mapping) or set(self.runtime) != _HEALTH_RUNTIME_KEYS or not all(
                    type(value) is str and _HEALTH_VERSION_TEXT.match(value) for value in self.runtime.values()):
                raise ValueError("runtime must name the four runtime versions")
            object.__setattr__(self, "runtime", MappingProxyType(dict(sorted(self.runtime.items()))))
        if self.reason_detail is not None and (type(self.reason_detail) is not str or not _HEALTH_DETAIL_TEXT.match(self.reason_detail)):
            raise ValueError("reason_detail must name one node class and input")
        if self.state not in HEALTH_STATES:
            raise ValueError(f"state must be one of {sorted(HEALTH_STATES)}")
        if self.state == "ready_static" and self.reason_code is not None:
            raise ValueError("ready_static cannot carry a reason_code")
        checks = _mapping(self.checks, "checks")
        if not all(isinstance(value, bool) for value in checks.values()):
            raise ValueError("checks values must be boolean")
        if set(checks) - STATIC_HEALTH_CHECKS:
            raise ValueError("checks contains an unknown static health key")
        if self.state == "ready_static":
            if set(checks) != STATIC_HEALTH_CHECKS or any(not checks[name] for name in STATIC_HEALTH_CHECKS):
                raise ValueError("ready_static requires every static check to be true")
        checks = MappingProxyType({name: checks.get(name, False) for name in sorted(STATIC_HEALTH_CHECKS)})
        object.__setattr__(self, "checks", checks)
        if not isinstance(self.inference_tested, bool):
            raise ValueError("inference_tested must be boolean")
        if self.inference_tested:
            raise ValueError("inference_tested must be false for static health")
        if self.reason_code is not None:
            object.__setattr__(self, "reason_code", _reason_code(self.reason_code))
        if self.diagnostic_detail is not None:
            object.__setattr__(self, "diagnostic_detail", _json_copy(self.diagnostic_detail, "diagnostic_detail"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "target_id": self.target_id,
            "state": self.state,
            "checks": _thaw_json(self.checks),
            "inference_tested": self.inference_tested,
            "reason_code": self.reason_code,
            **({"runtime_tested": self.runtime_tested} if self.runtime_tested is not None else {}),
            **({"runtime": dict(self.runtime)} if self.runtime is not None else {}),
            **({"reason_detail": self.reason_detail} if self.reason_detail is not None else {}),
        }

    def to_backend_dict(self) -> dict[str, Any]:
        return {"public": self.to_dict(), "diagnostic_detail": _thaw_json(self.diagnostic_detail)}

    def to_json(self) -> str:
        return _dump(self.to_dict())

    as_dict = to_dict
    as_json = to_json


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
