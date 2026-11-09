"""Pure, path-private durable generation request identities.

This module is intentionally independent from ``studio_core`` and from provider
clients.  Callers perform path validation/canonicalisation at their boundary and
pass the resulting strings plus the profile key here.  The only path operation
performed here is the domain-separated HMAC of an already canonical string.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


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


GENERATION_REQUEST_SCHEMA = "studio.generation-request/1"
PROFILE_KEY_SIZE = 32
MAX_DURABLE_REQUEST_BYTES = 1024 * 1024
MAX_PROMPT_BYTES = 256 * 1024
MAX_ID_CHARS = 256

MANAGED_NONPATH_KEYS = frozenset(
    {
        "prompt",
        "aspect_ratio",
        "duration",
        "resolution",
        "generate_audio",
        "grounding",
        "seed",
        "version",
        "keyframe_indices",
    }
)
MANAGED_MEDIA_KEYS = frozenset({"input_image", "input_images", "input_video"})
MANAGED_ARG_KEYS = MANAGED_NONPATH_KEYS | MANAGED_MEDIA_KEYS
MANAGED_MODES = frozenset(
    {"text_to_video", "image_to_video", "keyframes_to_video", "video_continuation"}
)
COMFY_H3_KEYS = frozenset({"prompt", "duration_seconds", "seed", "aspect_ratio", "megapixels"})

# This is deliberately kept byte-for-byte in parity with the managed FLUX 3
# tool.  Base64 is checked first because JPEG payloads begin with ``/9j/``.
_WINDOWS_DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_BASE64_PAYLOAD = re.compile(r"^[A-Za-z0-9+/\r\n]+={0,2}[\r\n]*$")
_MIN_BASE64_PAYLOAD_LENGTH = 256

_EXCLUDED_TRANSIENT_KEYS = frozenset(
    {
        "project_id",
        "project_ref",
        "project_directory",
        "project_folder",
        "directory",
        "save_to",
        "requested_basename",
        "destination",
        "destination_identity",
        "frozen_destination",
        "published_path",
        "publication_state",
        "publication",
        "manifest",
        "manifest_path",
        "state",
        "state_path",
        "output_path",
        "private_record_path",
        "identity_key",
        "key_path",
        "temp_path",
    }
)
_ALLOWED_REQUEST_KEYS = frozenset(
    {"schema", "client_request_id", "provider_id", "target_id", "args", "approved_roots"}
) | _EXCLUDED_TRANSIENT_KEYS
_DURABLE_IDENTITY_KEYS = frozenset(
    {"schema", "client_request_id", "provider_id", "target_id", "args", "approved_roots"}
)
_COMFY_H3_V2_KEYS = frozenset({"steps", "sampler_name", "scheduler"})
# Pinned H3 conditioning targets (lane fs-h3b3).  Media enter the durable
# identity as {path HMAC, content SHA-256}: a changed file is a new request.
H3_CONDITIONING_MEDIA_KEYS = {
    "minimax-h3-frames-v1": frozenset({"keyframes", "sound_track"}),
    "minimax-h3-references-v1": frozenset({"keyframes", "reference_images", "reference_videos", "control_video",
                                           "reference_audios", "reference_video_sound", "sound_track"}),
}
H3_CONDITIONING_TARGETS = frozenset(H3_CONDITIONING_MEDIA_KEYS)
# Camera field (lane w1-h3): the normalized camera and the server-compiled prompt enter
# the identity together, so the same camera is the same request.
_H3_CAMERA_IDENTITY_KEYS = frozenset({"camera", "prompt_compiled"})
_H3_CAMERA_FIELDS = frozenset({"move", "direction", "amplitude", "speed"})
# Inspector controls and Generate x N (lane w2-h3ctl): scalars plus the user LoRA list; a request
# without them keeps exactly its earlier identity.  count/take_index name one take of a batch.
_H3_CONTROL_IDENTITY_KEYS = frozenset({"turbo", "turbo_strength", "recipe", "sigma_shift_video", "sigma_shift_audio",
                                       "attention", "sparse", "sparsity", "cache", "engine", "count", "take_index"})
# Continuation targets on take sidecars (lane fs-obvb): each source take enters
# the durable identity as {path HMAC, content SHA-256 of its mp4}.
H3_CONTINUATION_MEDIA_KEYS = {
    "minimax-h3-extend-mctx-v1": ("source",),
    "minimax-h3-prepend-mctx-v1": ("source",),
    "minimax-h3-bridge-mctx-v1": ("source", "target"),
}
H3_CONTINUATION_TARGETS = frozenset(H3_CONTINUATION_MEDIA_KEYS)
# Finishing on a finished take (lane w1-h3): {take: job id, sha256: its mp4, upscale, rife}.
H3_FINISH_TARGET = "minimax-h3-finish-v1"
# Video upscale on a finished take (lane w23-vsr): {take, sha256, model, detail, size, width?, height?, filter, seed, in_frame?, out_frame?}.
VSR_UPSCALE_TARGET = "relayvsr-video-upscale-v1"
_H3_CONTINUATION_SCALARS = frozenset({"prompt", "duration_seconds", "seed", "steps", "sampler_name", "scheduler"})
_H3_MEDIA_KEYS = frozenset().union(*H3_CONDITIONING_MEDIA_KEYS.values())
_H3_MEDIA_LIMITS = {"keyframes": 10, "reference_images": 9, "reference_videos": 3, "reference_audios": 3}
_DURABLE_ARG_KEYS = MANAGED_ARG_KEYS | COMFY_H3_KEYS | _COMFY_H3_V2_KEYS | _H3_MEDIA_KEYS
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


class RequestIdentityError(ValueError):
    """Finite error raised before durable/provider side effects."""

    def __init__(self, message: str, *, code: str = "invalid_request") -> None:
        self.code = code
        self.reason_code = code
        super().__init__(f"{code}: {message}")


IdentityError = RequestIdentityError
PrivateStateUnavailable = RequestIdentityError


def _invalid(message: str) -> RequestIdentityError:
    return RequestIdentityError(message)


def _megapixels(target_id: str, value: Any) -> float:
    """The one H3 megapixel list (h3_megapixels); outside it: ``megapixels_not_offered``."""
    try:
        return H3_MP.check(target_id, value)
    except ValueError as exc:
        raise RequestIdentityError(str(exc).split(": ", 1)[-1], code=H3_MP.CODE) from None


def _private_state(message: str) -> RequestIdentityError:
    return RequestIdentityError(message, code="private_state_unavailable")


def _mapping_failure(field_name: str) -> RequestIdentityError:
    return _invalid(f"{field_name} mapping could not be read")


def _safe_mapping_keys(value: Mapping[str, Any], field_name: str) -> list[Any]:
    try:
        return list(value.keys())
    except Exception as exc:
        raise _mapping_failure(field_name) from exc


def _safe_mapping_items(value: Mapping[str, Any], field_name: str) -> list[tuple[Any, Any]]:
    try:
        return list(value.items())
    except Exception as exc:
        raise _mapping_failure(field_name) from exc


def _safe_mapping_names(value: Mapping[str, Any], field_name: str) -> list[Any]:
    try:
        return list(value)
    except Exception as exc:
        raise _mapping_failure(field_name) from exc


def _safe_mapping_get(value: Mapping[str, Any], name: str, default: Any, field_name: str) -> Any:
    try:
        return value.get(name, default)
    except Exception as exc:
        raise _mapping_failure(field_name) from exc


def _safe_mapping_item(value: Mapping[str, Any], name: str, field_name: str) -> Any:
    try:
        return value[name]
    except Exception as exc:
        raise _mapping_failure(field_name) from exc


def _validate_key(key: Any) -> bytes:
    if not isinstance(key, bytes) or len(key) != PROFILE_KEY_SIZE:
        raise _private_state("identity key must be exactly 32 bytes")
    return key


def _validate_text(value: Any, field_name: str, *, required: bool = True) -> str | None:
    if value is None and not required:
        return None
    # These values cross an identity boundary.  Reject subclasses before hash,
    # equality, or any string method can be overridden by an attacker.
    if type(value) is not str:
        raise _invalid(f"{field_name} must be text")
    if required and not value:
        raise _invalid(f"{field_name} must be non-empty text")
    if value != value.strip():
        raise _invalid(f"{field_name} must not contain surrounding whitespace")
    if len(value) > MAX_ID_CHARS:
        raise _invalid(f"{field_name} exceeds {MAX_ID_CHARS} characters")
    return value


def _freeze_json(value: Any, field_name: str, active: set[int]) -> Any:
    """Validate JSON recursively and return an immutable representation."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise _invalid(f"{field_name} must contain only finite JSON numbers")
        return value
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in active:
            raise _invalid(f"{field_name} must not contain cyclic JSON")
        active.add(marker)
        try:
            frozen: dict[str, Any] = {}
            try:
                items = value.items()
                for key, nested in items:
                    if type(key) is not str:
                        raise _invalid(f"{field_name} keys must be text")
                    frozen[key] = _freeze_json(nested, f"{field_name}.{key}", active)
            except RequestIdentityError:
                raise
            except Exception as exc:
                raise _mapping_failure(field_name) from exc
            return MappingProxyType(frozen)
        finally:
            active.remove(marker)
    if type(value) in (list, tuple):
        marker = id(value)
        if marker in active:
            raise _invalid(f"{field_name} must not contain cyclic JSON")
        active.add(marker)
        try:
            return tuple(_freeze_json(item, f"{field_name}[{index}]", active) for index, item in enumerate(value))
        finally:
            active.remove(marker)
    raise _invalid(f"{field_name} must contain only JSON values")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_json(value: Any, *, field_name: str = "request_json") -> str:
    try:
        frozen = _freeze_json(value, field_name, set())
        encoded = json.dumps(
            _thaw_json(frozen),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except RequestIdentityError:
        raise
    except Exception as exc:
        raise _invalid(f"{field_name} is not supported JSON") from exc
    try:
        size = len(encoded.encode("utf-8"))
    except UnicodeError as exc:
        raise _invalid(f"{field_name} must be valid UTF-8") from exc
    if size > MAX_DURABLE_REQUEST_BYTES:
        raise _invalid("request_json exceeds 1 MiB")
    return encoded


def _hash_path(key: bytes, canonical_path: str) -> str:
    if type(canonical_path) is not str:
        raise _invalid("canonical local path must be text")
    try:
        raw = canonical_path.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise _invalid("canonical local path must be valid UTF-8") from exc
    return hmac.new(key, b"path\0" + raw, hashlib.sha256).hexdigest()


def path_identity(canonical_path: str, key: bytes) -> str:
    """Return ``HMAC-SHA256(key, b'path\\0' + canonical_path)``."""
    return _hash_path(_validate_key(key), canonical_path)


path_hmac = path_identity
hmac_path_identity = path_identity


def _looks_like_local_path(value: str) -> bool:
    """Match the frozen FLUX 3 managed-media local-path classifier."""
    if type(value) is not str:
        raise _invalid("media value must be primitive text")
    if len(value) >= _MIN_BASE64_PAYLOAD_LENGTH and _BASE64_PAYLOAD.match(value):
        return False
    if value.startswith("file://"):
        return True
    if value == "~" or value.startswith(("~/", "~\\")):
        return True
    if value.startswith(("/", "./", "../", ".\\", "..\\")):
        return True
    if _WINDOWS_DRIVE_PATH.match(value) or value.startswith("\\\\"):
        return True
    return False


looks_like_local_path = _looks_like_local_path


def _media_identity(value: Any, key: bytes, field_name: str) -> Any:
    if type(value) is not str:
        raise _invalid(f"args.{field_name} must be text")
    return _hash_path(key, value) if _looks_like_local_path(value) else value


def _media_identities(value: Any, key: bytes, field_name: str) -> list[str]:
    if type(value) is not list or not 1 <= len(value) <= 10:
        raise _invalid(f"args.{field_name} must contain 1–10 strings")
    return [_media_identity(item, key, field_name) for item in value]


def _validate_managed_mode(mode: Any) -> str | None:
    if mode is None:
        return None
    if type(mode) is not str or mode not in MANAGED_MODES:
        raise _invalid("mode is unsupported")
    return mode


def _validate_prompt(prompt: Any) -> str:
    if type(prompt) is not str or not prompt.strip():
        raise _invalid("args.prompt must be non-empty text")
    try:
        prompt_size = len(prompt.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise _invalid("args.prompt must be valid UTF-8") from exc
    if prompt_size > MAX_PROMPT_BYTES:
        raise _invalid("args.prompt exceeds 256 KiB")
    return prompt


def _scalar(value: Any, field_name: str) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _invalid(f"{field_name} must be finite")
        return value
    raise _invalid(f"{field_name} must be a scalar JSON value")


def _managed_args(args: Any, key: bytes, *, mode: str | None = None) -> dict[str, Any]:
    if not isinstance(args, Mapping):
        raise _invalid("args must be an object")
    mode = _validate_managed_mode(mode)
    keys = _safe_mapping_keys(args, "args")
    for name in keys:
        if type(name) is not str or name not in MANAGED_ARG_KEYS:
            raise _invalid("args contains an unknown key")

    result: dict[str, Any] = {}
    for name in keys:
        value = _safe_mapping_item(args, name, "args")
        if name == "prompt":
            result[name] = _validate_prompt(value)
        elif name == "input_image" or name == "input_video":
            media_value = _media_identity(value, key, name)
            if mode != "text_to_video":
                result[name] = media_value
        elif name == "input_images":
            media_value = _media_identities(value, key, name)
            if mode != "text_to_video":
                result[name] = media_value
        elif name == "keyframe_indices":
            if not isinstance(value, list):
                raise _invalid("args.keyframe_indices must be a list")
            indices: list[int] = []
            for index in value:
                if isinstance(index, bool) or not isinstance(index, int):
                    raise _invalid("args.keyframe_indices must contain whole numbers")
                indices.append(index)
            result[name] = indices
        else:
            result[name] = _scalar(value, f"args.{name}")
    # A managed generation always has an artist prompt.  This also prevents a
    # caller accidentally using this boundary as an arbitrary JSON hasher.
    _validate_prompt(result.get("prompt"))
    return result


def managed_args_identity(args: Mapping[str, Any], key: bytes, *, mode: str | None = None) -> dict[str, Any]:
    """Validate and project the exact managed argument allowlist."""
    return _managed_args(args, _validate_key(key), mode=mode)


def without_media(args: Mapping[str, Any]) -> dict[str, Any]:
    """Return the managed text-to-video argument shape without media fields."""
    if not isinstance(args, Mapping):
        raise _invalid("args must be an object")
    keys = _safe_mapping_keys(args, "args")
    for name in keys:
        if type(name) is not str:
            raise _invalid("args contains an unknown key")
    return {name: _safe_mapping_item(args, name, "args") for name in keys if name not in MANAGED_MEDIA_KEYS}


def _request_mapping(request: Any, fields: Mapping[str, Any]) -> Mapping[str, Any]:
    if request is None:
        return fields
    if isinstance(request, Mapping):
        return request
    try:
        to_dict = getattr(request, "to_dict", None)
        if callable(to_dict):
            candidate = to_dict()
            if isinstance(candidate, Mapping):
                return candidate
    except Exception as exc:
        raise _invalid("request mapping could not be read") from exc
    raise _invalid("request must be an object")


def _build_request_from_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    candidate = dict(fields)
    candidate.pop("key", None)
    return candidate


def _request_identity_payload(request: Mapping[str, Any], key: bytes, *, mode: str | None) -> dict[str, Any]:
    for name in _safe_mapping_names(request, "request"):
        if type(name) is not str or name not in _ALLOWED_REQUEST_KEYS:
            raise _invalid("request contains an unknown field")

    schema = _safe_mapping_get(request, "schema", GENERATION_REQUEST_SCHEMA, "request")
    if type(schema) is not str or schema != GENERATION_REQUEST_SCHEMA:
        raise _invalid("schema is unsupported")
    client_request_id = _validate_text(
        _safe_mapping_get(request, "client_request_id", None, "request"),
        "client_request_id",
    )
    provider_id = _validate_text(_safe_mapping_get(request, "provider_id", None, "request"), "provider_id")
    target_id = _validate_text(_safe_mapping_get(request, "target_id", None, "request"), "target_id")
    args = _managed_args(_safe_mapping_get(request, "args", None, "request"), key, mode=mode)

    roots = _safe_mapping_get(request, "approved_roots", [], "request")
    if not isinstance(roots, list):
        raise _invalid("approved_roots must be a list")
    root_ids: list[str] = []
    try:
        root_values = list(roots)
    except Exception as exc:
        raise _invalid("approved_roots must be a finite list") from exc
    for root in root_values:
        if type(root) is not str or not root:
            raise _invalid("approved_roots must contain canonical local path strings")
        root_ids.append(_hash_path(key, root))

    # Do not copy or recursively inspect any excluded path/private fields.  They
    # are validated and retained by the operational/ledger boundary instead.
    return {
        "schema": schema,
        "client_request_id": client_request_id,
        "provider_id": provider_id,
        "target_id": target_id,
        "args": args,
        "approved_roots": root_ids,
    }


def _unique_json_object(pairs: list[tuple[Any, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if type(name) is not str:
            raise ValueError("JSON object keys must be primitive text")
        if name in result:
            raise ValueError("JSON object contains duplicate keys")
        result[name] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON constant {value} is not finite")


def _parse_durable_request_json(request_json: str) -> dict[str, Any]:
    try:
        parsed = json.loads(
            request_json,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except Exception as exc:
        raise _invalid("request_json must be a canonical JSON object") from exc
    if type(parsed) is not dict:
        raise _invalid("request_json must encode a JSON object")
    try:
        canonical = _canonical_json(parsed)
    except RequestIdentityError:
        raise
    except Exception as exc:
        raise _invalid("request_json must be a canonical JSON object") from exc
    if canonical != request_json:
        raise _invalid("request_json must use canonical JSON")
    return parsed


def _validate_durable_payload(payload: dict[str, Any]) -> None:
    unknown = set(payload) - _DURABLE_IDENTITY_KEYS
    if unknown:
        raise _invalid("request_json contains an unknown or private field")
    required = {"schema", "client_request_id", "provider_id", "target_id", "args", "approved_roots"}
    if required - set(payload):
        raise _invalid("request_json is missing a durable identity field")
    if type(payload["schema"]) is not str or payload["schema"] != GENERATION_REQUEST_SCHEMA:
        raise _invalid("schema is unsupported")
    _validate_text(payload["client_request_id"], "client_request_id")
    _validate_text(payload["provider_id"], "provider_id")
    _validate_text(payload["target_id"], "target_id")

    args = payload["args"]
    if type(args) is not dict:
        raise _invalid("request_json.args must be an object")
    if payload["target_id"] in H3_CONDITIONING_TARGETS or payload["target_id"] in H3_CONTINUATION_TARGETS \
            or payload["target_id"] in (H3_FINISH_TARGET, VSR_UPSCALE_TARGET):
        _comfy_args(args, target_id=payload["target_id"])
        if "approved_roots" in payload:
            roots = payload["approved_roots"]
            if type(roots) is not list or any(type(root) is not str or _SHA256_HEX.fullmatch(root) is None for root in roots):
                raise _invalid("request_json.approved_roots must contain path HMACs")
        return
    if set(args) & _H3_MEDIA_KEYS:
        raise _invalid("request_json args carry H3 media for a target without media roles")
    if set(args) - _DURABLE_ARG_KEYS:
        raise _invalid("request_json.args contains an unknown or private field")
    if payload["target_id"] == "minimax-h3-t2v" and set(args) & _COMFY_H3_V2_KEYS:
        raise _invalid("request_json v1 args contain a v2 sampling field")
    if payload["target_id"] == "minimax-h3-t2v-sampling-v2" and not _COMFY_H3_V2_KEYS <= set(args):
        raise _invalid("request_json v2 args are missing sampling fields")
    if payload["target_id"] == "minimax-h3-t2v-sampling-v2" and set(args) != COMFY_H3_KEYS | _COMFY_H3_V2_KEYS:
        raise _invalid("request_json v2 args do not match the frozen allowlist")
    if set(args) & _COMFY_H3_V2_KEYS and payload["target_id"] != "minimax-h3-t2v-sampling-v2":
        raise _invalid("sampling fields require the versioned H3 v2 target")
    if "prompt" not in args:
        raise _invalid("request_json.args.prompt must be non-empty text")
    for name, value in args.items():
        if name == "prompt":
            _validate_prompt(value)
        elif name in {"input_image", "input_video"}:
            if type(value) is not str:
                raise _invalid(f"request_json.args.{name} must be text")
            if _looks_like_local_path(value):
                raise _invalid(f"request_json.args.{name} must contain a path HMAC, not a local path")
        elif name == "input_images":
            if type(value) is not list or not 1 <= len(value) <= 10 or any(type(item) is not str for item in value):
                raise _invalid("request_json.args.input_images must contain 1–10 strings")
            if any(_looks_like_local_path(item) for item in value):
                raise _invalid("request_json.args.input_images must contain path HMACs, not local paths")
        elif name == "keyframe_indices":
            if type(value) is not list or any(type(item) is not int for item in value):
                raise _invalid("request_json.args.keyframe_indices must contain whole numbers")
        elif name == "steps":
            if type(value) is not int or not 1 <= value <= 100:
                raise _invalid("request_json.args.steps is invalid")
        elif name == "sampler_name":
            if type(value) is not str or value not in {"res_multistep", "euler"}:
                raise _invalid("request_json.args.sampler_name is invalid")
        elif name == "scheduler":
            if type(value) is not str or value not in {"simple", "beta"}:
                raise _invalid("request_json.args.scheduler is invalid")
        elif name == "megapixels" and payload["target_id"] == "minimax-h3-t2v-sampling-v2":
            _megapixels(payload["target_id"], value)
        else:
            _scalar(value, f"request_json.args.{name}")

    if "approved_roots" in payload:
        roots = payload["approved_roots"]
        if type(roots) is not list or any(type(root) is not str or _SHA256_HEX.fullmatch(root) is None for root in roots):
            raise _invalid("request_json.approved_roots must contain path HMACs")


@dataclass(frozen=True)
class DurableRequestIdentity:
    """Frozen canonical durable request JSON and its SHA-256."""

    request_json: str
    request_hash: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.request_json) is not str:
            raise _invalid("request_json must be primitive text")
        if type(self.request_hash) is not str:
            raise _invalid("request_hash must be primitive text")
        try:
            request_bytes = self.request_json.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise _invalid("request_json must be valid UTF-8") from exc
        if len(request_bytes) > MAX_DURABLE_REQUEST_BYTES:
            raise _invalid("request_json exceeds 1 MiB")
        parsed = _parse_durable_request_json(self.request_json)
        _validate_durable_payload(parsed)
        if _SHA256_HEX.fullmatch(self.request_hash) is None:
            raise _invalid("request_hash must be 64 lowercase hexadecimal characters")
        expected_hash = hashlib.sha256(request_bytes).hexdigest()
        if self.request_hash != expected_hash:
            raise _invalid("request_hash does not match request_json")
        try:
            frozen_payload = _freeze_json(self.payload, "payload", set())
            payload_json = _canonical_json(_thaw_json(frozen_payload), field_name="payload")
        except RequestIdentityError:
            raise
        except Exception as exc:
            raise _invalid("payload must be a finite JSON object") from exc
        if payload_json != self.request_json:
            raise _invalid("payload does not match request_json")
        object.__setattr__(self, "payload", frozen_payload)

    @property
    def json(self) -> str:
        return self.request_json

    @property
    def hash(self) -> str:
        return self.request_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_json": self.request_json,
            "request_hash": self.request_hash,
            "payload": _thaw_json(self.payload),
        }

    as_dict = to_dict

    def __getitem__(self, name: str) -> Any:
        return self.to_dict()[name]


RequestIdentity = DurableRequestIdentity


def _finish(payload: Mapping[str, Any]) -> DurableRequestIdentity:
    request_json = _canonical_json(payload)
    request_hash = hashlib.sha256(request_json.encode("utf-8")).hexdigest()
    return DurableRequestIdentity(request_json, request_hash, payload)


def build_durable_request_identity(
    request: Any = None,
    key: bytes | None = None,
    *,
    mode: str | None = None,
    text_to_video: bool = False,
    **fields: Any,
) -> DurableRequestIdentity:
    """Build the path-private Managed durable request identity.

    ``request`` may be a validated ``GenerationRequest`` or its mapping.  For
    callers that have separate fields, omit ``request`` and pass the request
    fields as keywords.  ``project_id`` and all operational paths are ignored
    by design; the project is a separate retry dimension.
    """
    validated_key = _validate_key(key)
    if text_to_video:
        mode = "text_to_video"
    request_map = _request_mapping(request, _build_request_from_fields(fields))
    payload = _request_identity_payload(request_map, validated_key, mode=mode)
    return _finish(payload)


build_request_identity = build_durable_request_identity
build_durable_identity = build_durable_request_identity
durable_request_identity = build_durable_request_identity
make_durable_request_identity = build_durable_request_identity


def _h3_media_entry(value: Any, field_name: str) -> dict[str, str]:
    if type(value) is not dict or set(value) != {"path", "sha256"}:
        raise _invalid(f"{field_name} must be a path HMAC and a content SHA-256")
    for name in ("path", "sha256"):
        if type(value[name]) is not str or _SHA256_HEX.fullmatch(value[name]) is None:
            raise _invalid(f"{field_name}.{name} must be 64 lowercase hex characters")
    return {"path": value["path"], "sha256": value["sha256"]}


def _h3_conditioning_args(args: Any, target_id: str) -> dict[str, Any]:
    """Exact durable allowlist of a pinned H3 conditioning target."""
    if not isinstance(args, Mapping):
        raise _invalid("args must be an object")
    allowed = (COMFY_H3_KEYS | _COMFY_H3_V2_KEYS | H3_CONDITIONING_MEDIA_KEYS[target_id] | _H3_CAMERA_IDENTITY_KEYS
               | _H3_CONTROL_IDENTITY_KEYS | {"loras"})
    names = _safe_mapping_keys(args, "args")
    if any(type(name) is not str or name not in allowed for name in names):
        raise _invalid("H3 args contain an unknown key")
    if (COMFY_H3_KEYS | _COMFY_H3_V2_KEYS) - set(names):
        raise _invalid("H3 args are missing a required control")
    if ("camera" in names) != ("prompt_compiled" in names):
        raise _invalid("H3 camera and its compiled prompt travel together")
    result: dict[str, Any] = {}
    for name in names:
        value = _safe_mapping_item(args, name, "args")
        if name in ("prompt", "prompt_compiled"):
            result[name] = _validate_prompt(value)
        elif name == "camera":
            if type(value) is not dict or not {"move", "amplitude", "speed"} <= set(value) <= _H3_CAMERA_FIELDS \
                    or any(type(item) is not str for item in value.values()):
                raise _invalid("args.camera must be a closed camera term")
            result[name] = dict(value)
        elif name == "keyframes":
            if type(value) is not list or len(value) > _H3_MEDIA_LIMITS[name]:
                raise _invalid("args.keyframes must hold at most 10 entries")
            frames: list[dict[str, Any]] = []
            for index, entry in enumerate(value):
                if type(entry) is not dict or set(entry) != {"frame", "source"} or type(entry["frame"]) is not int or entry["frame"] < 0:
                    raise _invalid(f"args.keyframes[{index}] must be a frame and a source")
                frames.append({"frame": entry["frame"], "source": _h3_media_entry(entry["source"], f"args.keyframes[{index}].source")})
            result[name] = frames
        elif name in ("reference_images", "reference_videos"):
            if type(value) is not list or len(value) > _H3_MEDIA_LIMITS[name]:
                raise _invalid(f"args.{name} holds too many entries")
            result[name] = [_h3_media_entry(entry, f"args.{name}[{index}]") for index, entry in enumerate(value)]
        elif name == "control_video":
            result[name] = _h3_media_entry(value, "args.control_video")
        elif name == "reference_audios":
            if type(value) is not list or len(value) > _H3_MEDIA_LIMITS[name]:
                raise _invalid("args.reference_audios holds too many entries")
            result[name] = [_h3_audio_entry(entry, f"args.reference_audios[{index}]", track=False) for index, entry in enumerate(value)]
        elif name == "sound_track":
            result[name] = _h3_audio_entry(value, "args.sound_track", track=True)
        elif name == "reference_video_sound":
            if type(value) is not list or len(value) > _H3_MEDIA_LIMITS["reference_videos"] or any(type(i) is not int or i < 0 for i in value):
                raise _invalid("args.reference_video_sound must list reference clip indices")
            result[name] = list(value)
        elif name == "megapixels":
            _megapixels(target_id, value)
            result[name] = value
        elif name == "loras":
            if type(value) is not list or len(value) > 8 or any(
                    type(item) is not dict or set(item) != {"file", "strength"} or type(item["file"]) is not str
                    or type(item["strength"]) not in (int, float) or not math.isfinite(item["strength"]) for item in value):
                raise _invalid("args.loras must list {file, strength}")
            result[name] = [{"file": item["file"], "strength": item["strength"]} for item in value]
        else:
            result[name] = _scalar(value, f"args.{name}")
    return result


def _h3_continuation_args(args: Any, target_id: str) -> dict[str, Any]:
    """Exact durable allowlist of an H3 continuation target (lane fs-obvb)."""
    if not isinstance(args, Mapping):
        raise _invalid("args must be an object")
    media = H3_CONTINUATION_MEDIA_KEYS[target_id]
    names = _safe_mapping_keys(args, "args")
    # lane p7-vid (VID-097): extend / prepend may name the take's chosen frame (source_frame, optional).
    optional = {"source_frame"} if target_id != "minimax-h3-bridge-mctx-v1" else set()
    if set(names) - optional != _H3_CONTINUATION_SCALARS | set(media) or any(type(name) is not str for name in names):
        raise _invalid("H3 continuation args do not match the allowlist")
    result: dict[str, Any] = {}
    for name in names:
        value = _safe_mapping_item(args, name, "args")
        if name == "prompt":
            result[name] = _validate_prompt(value)
        elif name in media:
            result[name] = _h3_media_entry(value, f"args.{name}")
        else:
            result[name] = _scalar(value, f"args.{name}")
    return result


def _h3_finish_args(args: Any) -> dict[str, Any]:
    """Exact durable allowlist of H3 finishing (lane w1-h3): source job id, its content SHA-256, two switches."""
    if not isinstance(args, Mapping):
        raise _invalid("args must be an object")
    names = _safe_mapping_keys(args, "args")
    if set(names) != {"take", "sha256", "upscale", "rife"} or any(type(name) is not str for name in names):
        raise _invalid("H3 finishing args do not match the allowlist")
    take = _safe_mapping_item(args, "take", "args")
    digest = _safe_mapping_item(args, "sha256", "args")
    if type(take) is not str or re.fullmatch(r"studio:[0-9a-f]{32}", take) is None:
        raise _invalid("args.take must be a studio job id")
    if type(digest) is not str or _SHA256_HEX.fullmatch(digest) is None:
        raise _invalid("args.sha256 must be 64 lowercase hex characters")
    switches = {name: _safe_mapping_item(args, name, "args") for name in ("upscale", "rife")}
    if any(type(value) is not bool for value in switches.values()) or not any(switches.values()):
        raise _invalid("args.upscale and args.rife are booleans, at least one on")
    return {"take": take, "sha256": digest, **switches}


def _vsr_upscale_args(args: Any) -> dict[str, Any]:
    """Exact durable allowlist of the RelayVSR video upscale (lane w23-vsr): source job + its SHA-256 + every setting."""
    if not isinstance(args, Mapping):
        raise _invalid("args must be an object")
    names = _safe_mapping_keys(args, "args")
    required = {"take", "sha256", "model", "detail", "size", "filter", "seed"}
    optional = {"width", "height", "in_frame", "out_frame"}
    if any(type(name) is not str for name in names) or not required <= set(names) or set(names) - required - optional:
        raise _invalid("video upscale args do not match the allowlist")
    take = _safe_mapping_item(args, "take", "args")
    digest = _safe_mapping_item(args, "sha256", "args")
    if type(take) is not str or re.fullmatch(r"studio:[0-9a-f]{32}", take) is None:
        raise _invalid("args.take must be a studio job id")
    if type(digest) is not str or _SHA256_HEX.fullmatch(digest) is None:
        raise _invalid("args.sha256 must be 64 lowercase hex characters")
    result: dict[str, Any] = {"take": take, "sha256": digest}
    for name in sorted(set(names) - {"take", "sha256"}):
        value = _safe_mapping_item(args, name, "args")
        if name == "detail":
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise _invalid("args.detail must be a number from 0 to 1")
        elif name in ("model", "size", "filter"):
            if type(value) is not str or not 0 < len(value) <= 20:
                raise _invalid(f"args.{name} must be a short string")
        elif type(value) is not int or value < 0:
            raise _invalid(f"args.{name} must be a non-negative whole number")
        result[name] = value
    return result


def _h3_audio_entry(value: Any, field_name: str, *, track: bool) -> dict[str, Any]:
    """Durable audio entry: {source: {path HMAC, sha256}, start?, end?} (+ role, subject? for a reference)."""
    allowed = {"source", "start", "end"} | (set() if track else {"role", "subject"})
    if type(value) is not dict or "source" not in value or set(value) - allowed or (not track and "role" not in value):
        raise _invalid(f"{field_name} must be a source with its role and optional start/end")
    result: dict[str, Any] = {"source": _h3_media_entry(value["source"], f"{field_name}.source")}
    for name in ("start", "end"):
        if name in value:
            number = value[name]
            if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
                raise _invalid(f"{field_name}.{name} must be a non-negative number")
            result[name] = number
    if not track:
        if value["role"] not in ("voice_of", "music_bed"):
            raise _invalid(f"{field_name}.role is unknown")
        result["role"] = value["role"]
        if "subject" in value:
            if type(value["subject"]) is not int or value["subject"] < 1:
                raise _invalid(f"{field_name}.subject must be a positive whole number")
            result["subject"] = value["subject"]
    return result


def _comfy_args(args: Any, *, target_id: str = "minimax-h3-t2v") -> dict[str, Any]:
    if not isinstance(args, Mapping):
        raise _invalid("args must be an object")
    if target_id in H3_CONDITIONING_TARGETS:
        return _h3_conditioning_args(args, target_id)
    if target_id in H3_CONTINUATION_TARGETS:
        return _h3_continuation_args(args, target_id)
    if target_id == H3_FINISH_TARGET:
        return _h3_finish_args(args)
    if target_id == VSR_UPSCALE_TARGET:
        return _vsr_upscale_args(args)
    result: dict[str, Any] = {}
    try:
        items = _safe_mapping_items(args, "args")
        for name, value in items:
            allowed = COMFY_H3_KEYS if target_id == "minimax-h3-t2v" else COMFY_H3_KEYS | {"steps", "sampler_name", "scheduler"} if target_id == "minimax-h3-t2v-sampling-v2" else frozenset()
            if type(name) is not str or name not in allowed:
                raise _invalid("Comfy args contains an unknown key")
            if name == "prompt":
                result[name] = _validate_prompt(value)
            else:
                result[name] = _scalar(value, f"args.{name}")
        if target_id == "minimax-h3-t2v-sampling-v2":
            if set(result) != COMFY_H3_KEYS | _COMFY_H3_V2_KEYS:
                raise _invalid("v2 args are missing required H3 controls")
            if type(result.get("duration_seconds")) is not int or not 4 <= result["duration_seconds"] <= 15:
                raise _invalid("duration_seconds is invalid")
            if type(result.get("seed")) is not int or not 0 <= result["seed"] <= 4294967295:
                raise _invalid("seed is invalid")
            if type(result.get("aspect_ratio")) is not str or result["aspect_ratio"] not in {
                "21:9 (Ultrawide)", "16:9 (Widescreen)", "4:3 (Standard)",
                "1:1 (Square)", "3:4 (Portrait Standard)", "9:16 (Portrait Widescreen)",
            }:
                raise _invalid("aspect_ratio is invalid")
            if type(result.get("steps")) is not int or not 1 <= result["steps"] <= 100:
                raise _invalid("steps must be a whole number from 1 to 100")
            if result.get("sampler_name") not in {"res_multistep", "euler"} or type(result.get("sampler_name")) is not str:
                raise _invalid("sampler_name is unsupported")
            if result.get("scheduler") not in {"simple", "beta"} or type(result.get("scheduler")) is not str:
                raise _invalid("scheduler is unsupported")
            _megapixels(target_id, result.get("megapixels"))
        elif target_id != "minimax-h3-t2v":
            raise _invalid("Comfy target is unsupported")
        elif "megapixels" in result:
            _megapixels(target_id, result["megapixels"])
    except RequestIdentityError:
        raise
    except Exception as exc:
        raise _mapping_failure("args") from exc
    _validate_prompt(result.get("prompt"))
    return result


def build_comfy_request_identity(
    request: Any = None,
    key: bytes | None = None,
    *,
    schema: str = GENERATION_REQUEST_SCHEMA,
    client_request_id: str | None = None,
    provider_id: str = "comfyui-local",
    target_id: str = "minimax-h3-t2v",
    args: Mapping[str, Any] | None = None,
    approved_roots: list[str] | tuple[str, ...] = (),
    **fields: Any,
) -> DurableRequestIdentity:
    """Build Comfy's H3-only durable identity; local controls are unsupported."""
    validated_key = _validate_key(key)
    if request is not None:
        request_map = _request_mapping(request, {})
        allowed_fields = {"schema", "client_request_id", "provider_id", "target_id", "args", "approved_roots"}
        for name in _safe_mapping_names(request_map, "request"):
            if type(name) is not str or name not in allowed_fields:
                raise _invalid("Comfy request contains an unknown field")
        schema = _safe_mapping_get(request_map, "schema", schema, "request")
        client_request_id = _safe_mapping_get(request_map, "client_request_id", client_request_id, "request")
        provider_id = _safe_mapping_get(request_map, "provider_id", provider_id, "request")
        target_id = _safe_mapping_get(request_map, "target_id", target_id, "request")
        args = _safe_mapping_get(request_map, "args", args, "request")
        approved_roots = _safe_mapping_get(request_map, "approved_roots", approved_roots, "request")
    if fields:
        allowed_fields = {"schema", "client_request_id", "provider_id", "target_id", "args", "approved_roots"}
        for name in fields:
            if type(name) is not str or name not in allowed_fields:
                raise _invalid("Comfy request contains an unknown field")
        schema = fields.get("schema", schema)
        client_request_id = fields.get("client_request_id", client_request_id)
        provider_id = fields.get("provider_id", provider_id)
        target_id = fields.get("target_id", target_id)
        args = fields.get("args", args)
        approved_roots = fields.get("approved_roots", approved_roots)
    if type(schema) is not str or schema != GENERATION_REQUEST_SCHEMA:
        raise _invalid("schema is unsupported")
    payload = {
        "schema": schema,
        "client_request_id": _validate_text(client_request_id, "client_request_id"),
        "provider_id": _validate_text(provider_id, "provider_id"),
        "target_id": _validate_text(target_id, "target_id"),
        "args": _comfy_args(args, target_id=_validate_text(target_id, "target_id")),
        "approved_roots": [],
    }
    if type(approved_roots) not in (list, tuple):
        raise _invalid("approved_roots must be a list")
    for root in approved_roots:
        if type(root) is not str or not root:
            raise _invalid("approved_roots must contain canonical local path strings")
        payload["approved_roots"].append(_hash_path(validated_key, root))
    return _finish(payload)


build_comfy_durable_identity = build_comfy_request_identity
build_comfy_identity = build_comfy_request_identity


__all__ = [
    "COMFY_H3_KEYS",
    "DurableRequestIdentity",
    "H3_CONDITIONING_MEDIA_KEYS",
    "H3_CONDITIONING_TARGETS",
    "H3_CONTINUATION_MEDIA_KEYS",
    "H3_CONTINUATION_TARGETS",
    "IdentityError",
    "MANAGED_ARG_KEYS",
    "MANAGED_MEDIA_KEYS",
    "MANAGED_MODES",
    "MANAGED_NONPATH_KEYS",
    "MAX_DURABLE_REQUEST_BYTES",
    "MAX_ID_CHARS",
    "MAX_PROMPT_BYTES",
    "PrivateStateUnavailable",
    "RequestIdentity",
    "RequestIdentityError",
    "build_comfy_durable_identity",
    "build_comfy_identity",
    "build_comfy_request_identity",
    "build_durable_identity",
    "build_durable_request_identity",
    "build_request_identity",
    "durable_request_identity",
    "hmac_path_identity",
    "looks_like_local_path",
    "managed_args_identity",
    "make_durable_request_identity",
    "path_hmac",
    "path_identity",
    "without_media",
]
