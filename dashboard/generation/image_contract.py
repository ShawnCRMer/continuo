"""Strict, provider-neutral image producer contract for the U4b slice.

This module is intentionally self-contained.  It knows the closed image
contract and its durable envelopes, but it does not import a provider,
ComfyUI, a route layer, or the NONPRODUCTION oracle.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import struct
import zlib
from dataclasses import dataclass
from typing import Any, Mapping


IMAGE_CAPABILITY_SCHEMA = "studio.image-capability/1"
IMAGE_PRODUCER_JOB_SCHEMA = "studio.image-producer-job/1"
IMAGE_ARTIFACT_SCHEMA = "studio.image-artifact-ref/2"
IMAGE_STATE_SCHEMA = "studio.image-state/2"
IMAGE_OPERATION_SCHEMA = "studio.image-producer-operation/1"

PROVIDER_ID = "local-comfyui"
TARGET_ID = "krea2-turbo-t2i-create"
CAPABILITY_ID = TARGET_ID
GRAPH_SHA256 = "67cafc4d0cf88cf29d3b992d20455e63788307e32d846b581944a4e87892a02a"
CAPABILITY_SHA256 = "16da82babfa04c180d4576a2fe48af09c59def6e201e414b06d6c864d5f53206"
# 32 MiB until w1-imgout: a SeedVR2 4x of a 1024 picture is a 34.3 MB PNG (measured, 4096 x 4096),
# an 8192 x 8192 upscale up to ~200 MB. MAX_PIXELS (8192 x 8192) is unchanged.
MAX_BYTES = 268_435_456
MAX_DIMENSION = 16_384
MAX_PIXELS = 67_108_864
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

LEGAL_TRANSITIONS: dict[str, frozenset[str]] = {
    "accepted": frozenset({"submitting", "failed", "cancelled"}),
    "submitting": frozenset({"queued", "failed", "cancelled", "quarantined"}),
    "queued": frozenset({"running", "failed", "cancelled", "quarantined"}),
    "running": frozenset({"ready", "failed", "cancelled", "quarantined"}),
    "ready": frozenset({"ingested", "failed", "quarantined", "cancelled"}),
    "ingested": frozenset({"published", "failed", "quarantined"}),
    "published": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
    "quarantined": frozenset(),
}
STATES = frozenset(LEGAL_TRANSITIONS)


class ContractError(ValueError):
    """Raised when a caller or persisted envelope violates the image contract."""


# The store exposes this name so callers do not need to know which contract
# module owns the boundary exception.
ImageContractError = ContractError


class ImageSchemaError(RuntimeError):
    """Raised when bytes already persisted by the image ledger are invalid."""


def canonical_bytes(value: Any) -> bytes:
    """Serialize the only accepted identity form: compact sorted UTF-8 JSON."""
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError) as exc:
        raise ContractError("non-canonical-json") from exc
    return encoded


def parse_canonical(raw: bytes) -> Any:
    if type(raw) is not bytes:
        raise ContractError("request-bytes-required")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ContractError("invalid-json") from exc
    if canonical_bytes(value) != raw:
        raise ContractError("non-canonical-json")
    return value


def sha256(value: bytes) -> str:
    if type(value) is not bytes:
        raise ContractError("bytes-required")
    return hashlib.sha256(value).hexdigest()


def _strict_int(value: Any, name: str, low: int, high: int, multiple: int | None = None) -> int:
    if type(value) is not int or value < low or value > high:
        raise ContractError(f"{name}-invalid")
    if multiple is not None and value % multiple:
        raise ContractError(f"{name}-invalid")
    return value


def _strict_str(value: Any, name: str, low: int = 0, high: int = 16_384) -> str:
    if type(value) is not str or not low <= len(value) <= high:
        raise ContractError(f"{name}-invalid")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ContractError(f"{name}-invalid") from exc
    return value


_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_WINDOWS_INVALID_BASENAME_CHARS = frozenset('<>:"/\\|?*')
_WINDOWS_RESERVED_BASENAMES = frozenset(
    {
        "con", "prn", "aux", "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)
_ID_PATTERNS = {
    "imgctx_": re.compile(r"^imgctx_[0-9a-f]{32}$"),
    "imgwork_": re.compile(r"^imgwork_[0-9a-f]{32}$"),
    "imgref_": re.compile(r"^imgref_[0-9a-f]{32}$"),
    "imgrev_": re.compile(r"^imgrev_[0-9a-f]{32}$"),
    "imgjob_": re.compile(r"^imgjob_[0-9a-f]{32}$"),
    "imgmat_": re.compile(r"^imgmat_[0-9a-f]{32}$"),
    "imgop_": re.compile(r"^imgop_[0-9a-f]{32}$"),
    "imgbind_": re.compile(r"^imgbind_[0-9a-f]{32}$"),
    "imgadopt_": re.compile(r"^imgadopt_[0-9a-f]{32}$"),
    "filmref_": re.compile(r"^filmref_[0-9a-f]{32}$"),
    "prj_": re.compile(r"^prj_[0-9a-f]{32}$"),
    "imgq_": re.compile(r"^imgq_[0-9a-f]{32}$"),
    "imgkey_": re.compile(r"^imgkey_[0-9a-f]{32}$"),
}


def _strict_token(value: Any, name: str) -> str:
    if type(value) is not str or _TOKEN_PATTERN.fullmatch(value) is None:
        raise ContractError(f"{name}-invalid")
    return value


def _id(prefix: str, seed: bytes) -> str:
    return prefix + hashlib.sha256(seed).hexdigest()[:32]


def _id_value(value: Any, prefix: str, name: str) -> str:
    pattern = _ID_PATTERNS.get(prefix)
    if type(value) is not str or pattern is None or pattern.fullmatch(value) is None:
        raise ContractError(f"{name}-invalid")
    return value


def _hex64(value: Any, name: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ContractError(f"{name}-invalid")
    return value


def _closed(value: Any, required: set[str], optional: set[str] | None, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ContractError(f"{name}-shape")
    optional = set() if optional is None else optional
    if not required.issubset(value) or set(value) - required - optional:
        raise ContractError(f"{name}-shape")
    return value


def validate_basis(basis: dict[str, Any]) -> dict[str, Any]:
    required = {"context_id", "work_id", "basis_kind", "revision_id", "draft_generation", "basis_content_sha256"}
    if type(basis) is not dict or set(basis) != required:
        raise ContractError("basis-shape")
    _id_value(basis["context_id"], "imgctx_", "context-id")
    _id_value(basis["work_id"], "imgwork_", "work-id")
    _hex64(basis["basis_content_sha256"], "basis-content-sha256")
    if basis["basis_kind"] == "saved_revision":
        _id_value(basis["revision_id"], "imgrev_", "revision-id")
        if basis["draft_generation"] is not None:
            raise ContractError("basis-alternative")
    elif basis["basis_kind"] == "draft":
        if basis["revision_id"] is not None:
            raise ContractError("basis-alternative")
        _strict_int(basis["draft_generation"], "draft-generation", 0, 9_007_199_254_740_991)
    else:
        raise ContractError("basis-kind")
    return copy.deepcopy(basis)


def validate_policy(policy: dict[str, Any]) -> dict[str, Any]:
    if type(policy) is not dict or policy.get("mode") not in ("current", "specified"):
        raise ContractError("attach-policy-shape")
    expected = {"mode", "owner_token", "expected_revision"}
    if policy["mode"] == "specified":
        expected |= {"context_id", "work_id", "revision_id"}
    if set(policy) != expected:
        raise ContractError("attach-policy-shape")
    _strict_token(policy["owner_token"], "owner-token")
    _strict_int(policy["expected_revision"], "expected-revision", 0, 9_007_199_254_740_991)
    if policy["mode"] == "specified":
        _id_value(policy["context_id"], "imgctx_", "context-id")
        _id_value(policy["work_id"], "imgwork_", "work-id")
        _id_value(policy["revision_id"], "imgrev_", "revision-id")
    return copy.deepcopy(policy)


class ModelFileMissing(ContractError):
    """w24-models: a well-formed request names a model file ComfyUI cannot load now (moment of use)."""

    def __init__(self, role: str, name: str, folder: str) -> None:
        super().__init__("model-file-missing")
        self.role, self.name, self.folder = role, name, folder


def default_render_spec(seed: int, width: int, height: int) -> dict[str, Any]:
    from .krea2_render_spec import default_spec
    return default_spec(seed, width, height)


def validate_render_spec(value: Any, *, check_files: bool = True) -> dict[str, Any]:
    from .krea2_render_spec import validate_spec
    return validate_spec(value, check_files=check_files)


def build_snapshot(
    prompt: str,
    seed: int,
    width: int,
    height: int,
    basis: dict[str, Any],
    attach_policy: dict[str, Any],
    capability_hash: str = CAPABILITY_SHA256,
    graph_hash: str = GRAPH_SHA256,
    render_spec: dict[str, Any] | None = None,
    *,
    check_files: bool = True,
) -> dict[str, Any]:
    """The canonical create snapshot.  ``check_files=False`` rebuilds a STORED snapshot (w24-models):
    shape and integrity only, never whether today's disk still holds the named model files."""
    _strict_str(prompt, "prompt", 1, 16_384)
    _strict_int(seed, "seed", 0, 18_446_744_073_709_551_615)
    _strict_int(width, "width", 16, MAX_DIMENSION, 8)
    _strict_int(height, "height", 16, MAX_DIMENSION, 8)
    from .krea2_render_spec import default_spec, validate_spec, MODELS, KREA_FILES, MissingModelFile
    try:
        spec = validate_spec(default_spec(seed, width, height) if render_spec is None else render_spec,
                             check_files=check_files)
    except MissingModelFile as exc:
        raise ModelFileMissing(exc.role, exc.name, exc.folder) from exc
    except ValueError as exc:
        raise ContractError("render-spec-invalid") from exc
    if (spec["seed"], spec["width"], spec["height"]) != (seed, width, height):
        raise ContractError("render-spec-artist-mismatch")
    validate_basis(basis)
    validate_policy(attach_policy)
    _hex64(capability_hash, "capability-hash")
    _hex64(graph_hash, "graph-hash")
    if capability_hash != CAPABILITY_SHA256 or graph_hash != GRAPH_SHA256:
        raise ContractError("unrecognized-capability-or-graph")
    if spec.get("schema") == "studio.krea2-render-spec/2":
        # Additive v2: the chosen files are the effective files, and a ledger
        # job always renders exactly one picture of its run.
        model, encoder, vae, batch = (spec["files"]["diffusion_model"], spec["files"]["text_encoder"],
                                      spec["files"]["vae"], 1)
    else:
        model, encoder, vae, batch = MODELS[spec["profile"]], KREA_FILES["encoder"], KREA_FILES["vae"], spec["batch_size"]
    return {
        "operation": "create",
        "target_id": TARGET_ID,
        "artist_input": {"prompt": prompt, "seed": seed, "width": width, "height": height},
        "effective_input": {
            "model": model,
            "encoder": encoder,
            "vae": vae,
            "steps": spec["steps"],
            "cfg": spec["cfg"] if render_spec is not None else 1,
            "sampler": spec["sampler"],
            "scheduler": spec["scheduler"],
            "denoise": 1,
            "batch": batch,
            **({"render_spec": copy.deepcopy(spec)} if render_spec is not None else {}),
        },
        "context_basis": copy.deepcopy(basis),
        "attach_policy": copy.deepcopy(attach_policy),
        "capability_manifest_sha256": capability_hash,
        "api_graph_sha256": graph_hash,
    }


def _edit_contract() -> Any:
    from . import image_edit_contract
    return image_edit_contract


def edit_target_ids() -> frozenset[str]:
    """Target ids of the pinned edit workflows (R4, additive to the Krea create target)."""
    return frozenset(_edit_contract().EDIT_TARGETS)


def _compose() -> Any:
    from . import image_compose
    return image_compose


def _import() -> Any:
    from . import image_import
    return image_import


def _upscale() -> Any:
    from . import image_upscale
    return image_upscale


def known_target_ids() -> frozenset[str]:
    from .image_workflow_contract import entries
    return frozenset({TARGET_ID, _compose().TARGET_ID, _import().TARGET_ID, _upscale().TARGET_ID}) | edit_target_ids() | frozenset(entries())


def snapshot_target_id(snapshot: Any) -> str:
    """The target a validated snapshot runs on (create snapshots: the Krea target)."""
    if type(snapshot) is dict and snapshot.get("operation") in {"edit", "workflow_create", "workflow_edit", "compose", "import", "upscale"} and snapshot.get("target_id") in known_target_ids():
        return snapshot["target_id"]
    return TARGET_ID


def validate_snapshot(snapshot: dict[str, Any]) -> bytes:
    required = {
        "operation", "target_id", "artist_input", "effective_input", "context_basis",
        "attach_policy", "capability_manifest_sha256", "api_graph_sha256",
    }
    if type(snapshot) is not dict or set(snapshot) != required:
        raise ContractError("snapshot-unknown-shape")
    if snapshot.get("operation") in {"workflow_create", "workflow_edit"}:
        from .image_workflow_contract import validate_snapshot as validate_workflow
        return validate_workflow(snapshot)
    if snapshot.get("operation") == "compose":
        # A picture the Suite composed (Accept as version, Expand): accepted only when a rebuild
        # from its own intent gives the same canonical bytes (image_compose).
        compose = _compose()
        validate_basis(snapshot["context_basis"])
        validate_policy(snapshot["attach_policy"])
        try:
            expected = compose.rebuild_snapshot(snapshot)
        except (compose.ComposeError, TypeError, KeyError, ValueError) as exc:
            raise ContractError("compose-snapshot-invalid") from exc
        snapshot_bytes = canonical_bytes(snapshot)
        if snapshot_bytes != canonical_bytes(expected):
            raise ContractError("effective-input-or-shape-mismatch")
        return snapshot_bytes
    if snapshot.get("operation") == "upscale":
        # SeedVR2 upscale of an owned picture (a new version): accepted only when a rebuild
        # from its own intent gives the same canonical bytes, graph included (image_upscale).
        upscale = _upscale()
        validate_basis(snapshot["context_basis"])
        validate_policy(snapshot["attach_policy"])
        try:
            expected = upscale.rebuild_snapshot(snapshot)
        except (upscale.UpscaleError, TypeError, KeyError, ValueError) as exc:
            raise ContractError("upscale-snapshot-invalid") from exc
        snapshot_bytes = canonical_bytes(snapshot)
        if snapshot_bytes != canonical_bytes(expected):
            raise ContractError("effective-input-or-shape-mismatch")
        return snapshot_bytes
    if snapshot.get("operation") == "import":
        # A picture from disk or the clipboard: accepted only when a rebuild from its own
        # provenance gives the same canonical bytes (image_import).
        imported = _import()
        validate_basis(snapshot["context_basis"])
        validate_policy(snapshot["attach_policy"])
        try:
            expected = imported.rebuild_snapshot(snapshot)
        except (imported.ImageImportError, TypeError, KeyError, ValueError) as exc:
            raise ContractError("import-snapshot-invalid") from exc
        snapshot_bytes = canonical_bytes(snapshot)
        if snapshot_bytes != canonical_bytes(expected):
            raise ContractError("effective-input-or-shape-mismatch")
        return snapshot_bytes
    if snapshot.get("operation") == "edit":
        # R4 edit variant (studio.image-edit-request/1): accepted only when a
        # rebuild from its own intent gives the same canonical bytes.
        edit = _edit_contract()
        if type(snapshot["artist_input"]) is not dict or set(snapshot["artist_input"]) != {"prompt", "seed", "width", "height"}:
            raise ContractError("artist-input-shape")
        validate_basis(snapshot["context_basis"])
        validate_policy(snapshot["attach_policy"])
        try:
            expected = edit.rebuild_edit_snapshot(snapshot)
        except (edit.EditRequestError, TypeError, KeyError, ValueError) as exc:
            raise ContractError("edit-snapshot-invalid") from exc
        snapshot_bytes = canonical_bytes(snapshot)
        if snapshot_bytes != canonical_bytes(expected):
            raise ContractError("effective-input-or-shape-mismatch")
        return snapshot_bytes
    artist = snapshot["artist_input"]
    if type(artist) is not dict or set(artist) != {"prompt", "seed", "width", "height"}:
        raise ContractError("artist-input-shape")
    expected = build_snapshot(
        artist["prompt"], artist["seed"], artist["width"], artist["height"],
        snapshot["context_basis"], snapshot["attach_policy"],
        snapshot["capability_manifest_sha256"], snapshot["api_graph_sha256"],
        snapshot["effective_input"].get("render_spec") if type(snapshot.get("effective_input")) is dict else None,
        check_files=False,  # w24-models: a stored job is read by shape and integrity, never against today's disk
    )
    snapshot_bytes = canonical_bytes(snapshot)
    if snapshot_bytes != canonical_bytes(expected):
        raise ContractError("effective-input-or-shape-mismatch")
    return snapshot_bytes


def validate_png_byte_size(byte_size: Any) -> None:
    if type(byte_size) is not int or byte_size < 1 or byte_size > MAX_BYTES:
        raise ContractError("png-size-exceeded")


def _png_facts(data: bytes) -> dict[str, Any]:
    if type(data) is not bytes or not data:
        raise ContractError("png-empty")
    validate_png_byte_size(len(data))
    if not data.startswith(PNG_SIGNATURE):
        raise ContractError("png-signature")
    pos = len(PNG_SIGNATURE)
    ihdr: tuple[int, int, int] | None = None
    idat = bytearray()
    seen_iend = False
    chunks = 0
    while pos < len(data):
        if pos + 12 > len(data):
            raise ContractError("png-truncated")
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        kind = data[pos + 4:pos + 8]
        end = pos + 12 + length
        if end > len(data):
            raise ContractError("png-truncated")
        payload = data[pos + 8:pos + 8 + length]
        crc = struct.unpack(">I", data[pos + 8 + length:end])[0]
        if zlib.crc32(kind + payload) & 0xFFFFFFFF != crc:
            raise ContractError("png-crc")
        chunks += 1
        if kind == b"IHDR":
            if ihdr is not None or length != 13:
                raise ContractError("png-ihdr")
            width, height, depth, color, compression, filter_method, interlace = struct.unpack(">IIBBBBB", payload)
            if depth != 8 or color not in (2, 6) or compression != 0 or filter_method != 0 or interlace != 0:
                raise ContractError("png-format")
            if not 1 <= width <= MAX_DIMENSION or not 1 <= height <= MAX_DIMENSION:
                raise ContractError("png-dimensions")
            if width * height > MAX_PIXELS:
                raise ContractError("png-pixels")
            ihdr = (width, height, color)
        elif kind == b"IDAT":
            if ihdr is None:
                raise ContractError("png-order")
            idat.extend(payload)
        elif kind == b"IEND":
            if length != 0 or seen_iend:
                raise ContractError("png-iend")
            seen_iend = True
            if end != len(data):
                raise ContractError("png-trailing")
        pos = end
    if pos != len(data) or ihdr is None or not idat or not seen_iend:
        raise ContractError("png-structure")
    width, height, color = ihdr
    channels = 3 if color == 2 else 4
    row = width * channels
    expected = (row + 1) * height
    # Lane w34-sec (scan finding 26): inflate at most expected + 1 bytes. zlib.decompress materialised the whole stream
    # (a few MiB of zeros inflate to GiB) before the length check; more output than IHDR allows, or input left over
    # after it, is png-decode now without allocating more.
    inflater = zlib.decompressobj()
    try:
        raw = inflater.decompress(bytes(idat), expected + 1)
    except zlib.error as exc:
        raise ContractError("png-data") from exc
    if not inflater.eof and len(raw) < expected:
        raise ContractError("png-data")  # a truncated stream, as zlib.decompress answered it
    if len(raw) != expected or inflater.unconsumed_tail or not inflater.eof:
        raise ContractError("png-decode")
    if any(raw[index * (row + 1)] > 4 for index in range(height)):
        raise ContractError("png-decode")
    return {
        "format": "png",
        "mime_type": "image/png",
        "byte_size": len(data),
        "width": width,
        "height": height,
        "pixel_count": width * height,
        "sha256": sha256(data),
        "chunks": chunks,
    }


def validate_output(
    data: bytes,
    descriptor_count: int,
    descriptor: dict[str, Any] | None,
    expected_width: int,
    expected_height: int,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    if type(descriptor_count) is not int or descriptor_count != 1:
        raise ContractError("descriptor-count")
    required = {"filename", "subfolder", "type", "mime_type", "extension"}
    if type(descriptor) is not dict or set(descriptor) != required:
        raise ContractError("descriptor-shape")
    _validate_descriptor(descriptor)
    facts = _png_facts(data)
    if facts["width"] != expected_width or facts["height"] != expected_height:
        raise ContractError("requested-dimensions")
    if expected_sha256 is not None and facts["sha256"] != expected_sha256:
        raise ContractError("hash-mismatch")
    return facts


def validate_artifact_ref(ref: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema", "material_id", "origin", "producer_job_id", "producer_job_schema",
        "request_snapshot_sha256", "artifact_sha256", "format", "mime_type",
        "byte_size", "width", "height", "artifact_version",
    }
    if type(ref) is not dict or set(ref) != required:
        raise ContractError("artifact-ref-shape")
    if ref["schema"] != IMAGE_ARTIFACT_SCHEMA or ref["origin"] != "generated" or ref["producer_job_schema"] != IMAGE_PRODUCER_JOB_SCHEMA:
        raise ContractError("artifact-ref-provenance")
    _id_value(ref["material_id"], "imgmat_", "material-id")
    _id_value(ref["producer_job_id"], "imgjob_", "producer-job-id")
    _hex64(ref["request_snapshot_sha256"], "request-snapshot-sha256")
    _hex64(ref["artifact_sha256"], "artifact-sha256")
    if ref["format"] != "png" or ref["mime_type"] != "image/png":
        raise ContractError("artifact-ref-format")
    _strict_int(ref["byte_size"], "byte-size", 1, MAX_BYTES)
    _strict_int(ref["width"], "width", 1, MAX_DIMENSION)
    _strict_int(ref["height"], "height", 1, MAX_DIMENSION)
    _strict_int(ref["artifact_version"], "artifact-version", 1, 9_007_199_254_740_991)
    if ref["artifact_version"] != 1:
        raise ContractError("artifact-version")
    return copy.deepcopy(ref)


def _validate_descriptor(descriptor: Any) -> dict[str, str]:
    required = {"filename", "subfolder", "type", "mime_type", "extension"}
    if type(descriptor) is not dict or set(descriptor) != required:
        raise ContractError("operation-descriptor-shape")
    filename = _strict_str(descriptor["filename"], "descriptor-filename", 1, 255)
    subfolder = _strict_str(descriptor["subfolder"], "descriptor-subfolder", 0, 255)
    mime_type = _strict_str(descriptor["mime_type"], "descriptor-mime-type", 0, 16_384)
    extension = _strict_str(descriptor["extension"], "descriptor-extension", 1, 16_384)
    for value, name in (
        (filename, "descriptor-filename"),
        (subfolder, "descriptor-subfolder"),
        (mime_type, "descriptor-mime-type"),
        (extension, "descriptor-extension"),
    ):
        if any(ord(char) < 0x20 or 0x7F <= ord(char) <= 0x9F for char in value):
            raise ContractError(f"{name}-control")
    if (
        any(char in _WINDOWS_INVALID_BASENAME_CHARS for char in filename)
        or filename in {".", ".."}
        or filename != filename.rstrip(" .")
        or filename.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_BASENAMES
        or not filename.casefold().endswith(".png")
        or subfolder != ""
        or descriptor["type"] != "temp"
        or mime_type != "image/png"
        or extension != ".png"
    ):
        raise ContractError("descriptor-authority")
    return descriptor


def validate_operation(operation: dict[str, Any]) -> dict[str, Any]:
    if type(operation) is not dict:
        raise ContractError("operation-shape")
    kind = operation.get("kind")
    job_key = "image_producer_job_id"
    if kind == "submit":
        if set(operation) != {"kind", job_key, "client_request_id", "request_snapshot_sha256"}:
            raise ContractError("operation-submit-shape")
        _id_value(operation[job_key], "imgjob_", job_key)
        _strict_token(operation["client_request_id"], "client-request-id")
        _hex64(operation["request_snapshot_sha256"], "request-snapshot-sha256")
    elif kind == "status":
        if set(operation) != {"kind", job_key, "provider_prompt_id"}:
            raise ContractError("operation-status-shape")
        _id_value(operation[job_key], "imgjob_", job_key)
        _strict_str(operation["provider_prompt_id"], "provider-prompt-id", 1, 256)
    elif kind == "cancel":
        required = {"kind", job_key, "cancel_scope", "provider_prompt_id"}
        if set(operation) != required:
            raise ContractError("operation-cancel-shape")
        _id_value(operation[job_key], "imgjob_", job_key)
        if operation["cancel_scope"] == "local":
            if operation["provider_prompt_id"] is not None:
                raise ContractError("operation-local-cancel-provider-id")
        elif operation["cancel_scope"] == "provider":
            _strict_str(operation["provider_prompt_id"], "provider-prompt-id", 1, 256)
        else:
            raise ContractError("operation-cancel-scope")
    elif kind == "retry":
        if set(operation) != {"kind", job_key, "retry_key"}:
            raise ContractError("operation-retry-shape")
        _id_value(operation[job_key], "imgjob_", job_key)
        _strict_token(operation["retry_key"], "retry-key")
    elif kind == "ingest":
        required = {
            "kind", job_key, "provider_id", "provider_prompt_id", "target_id",
            "capability_manifest_sha256", "api_graph_sha256", "request_snapshot_sha256",
            "context_basis", "result_sha256", "descriptors",
        }
        if set(operation) != required:
            raise ContractError("operation-ingest-shape")
        _id_value(operation[job_key], "imgjob_", job_key)
        if operation["provider_id"] != PROVIDER_ID or operation["target_id"] not in known_target_ids():
            raise ContractError("operation-ingest-authority")
        _strict_str(operation["provider_prompt_id"], "provider-prompt-id", 1, 256)
        _hex64(operation["capability_manifest_sha256"], "capability-manifest-sha256")
        _hex64(operation["api_graph_sha256"], "api-graph-sha256")
        _hex64(operation["request_snapshot_sha256"], "request-snapshot-sha256")
        validate_basis(operation["context_basis"])
        _hex64(operation["result_sha256"], "result-sha256")
        if type(operation["descriptors"]) is not list or len(operation["descriptors"]) > 2:
            raise ContractError("operation-descriptor-count")
        for descriptor in operation["descriptors"]:
            _validate_descriptor(descriptor)
    elif kind == "attach":
        base = {"kind", job_key, "mode", "expected_revision", "owner_token"}
        if not base.issubset(operation):
            raise ContractError("operation-attach-shape")
        _id_value(operation[job_key], "imgjob_", job_key)
        if operation["mode"] not in ("current", "specified"):
            raise ContractError("operation-attach-mode")
        _strict_int(operation["expected_revision"], "expected-revision", 0, 9_007_199_254_740_991)
        _strict_token(operation["owner_token"], "owner-token")
        if operation["mode"] == "current":
            if set(operation) != base:
                raise ContractError("operation-current-attach-shape")
        else:
            required = base | {"context_id", "work_id", "revision_id"}
            if set(operation) != required:
                raise ContractError("operation-specified-attach-shape")
            _id_value(operation["context_id"], "imgctx_", "context-id")
            _id_value(operation["work_id"], "imgwork_", "work-id")
            _id_value(operation["revision_id"], "imgrev_", "revision-id")
    elif kind == "response":
        required = {"kind", "operation", job_key, "status"}
        if not required.issubset(operation) or set(operation) - required not in (set(), {"reason"}):
            raise ContractError("operation-response-shape")
        if operation["operation"] not in {"submit", "status", "cancel", "retry", "ingest", "attach"}:
            raise ContractError("operation-response-operation")
        _id_value(operation[job_key], "imgjob_", job_key)
        if operation["status"] not in STATES | {"conflict"}:
            raise ContractError("operation-response-status")
        if "reason" in operation:
            _strict_str(operation["reason"], "operation-reason", 0, 160)
    else:
        raise ContractError("operation-kind")
    return copy.deepcopy(operation)


def build_capability() -> dict[str, Any]:
    return {
        "schema": IMAGE_CAPABILITY_SCHEMA,
        "capability_id": CAPABILITY_ID,
        "provider_id": PROVIDER_ID,
        "target_id": TARGET_ID,
        "operation": "create",
        "graph_sha256": GRAPH_SHA256,
        "api_graph_sha256": GRAPH_SHA256,
        "models": {
            "diffusion_model": {
                "filename": "krea2_turbo_int8_convrot.safetensors",
                "folder": "diffusion_models",
                "pointer": "/289/inputs/unet_name",
                "source_node": 289,
            },
            "text_encoder": {
                "filename": "qwen3vl_4b_fp8_scaled.safetensors",
                "folder": "text_encoders",
                "pointer": "/291/inputs/clip_name",
                "source_node": 291,
            },
            "vae": {
                "filename": "qwen_image_vae.safetensors",
                "folder": "vae",
                "pointer": "/292/inputs/vae_name",
                "source_node": 292,
            },
        },
        "accepted_controls": ["prompt", "seed", "width", "height"],
        "output_contract": {
            "format": "png",
            "mime_type": "image/png",
            "descriptor_count": 1,
            "accepted_keys": ["images"],
            "retrieve_via": "/view",
            "type": "temp",
        },
        "health_policy": "static_only",
        "admission_limit": 1,
        "graph_mutation": "declared_pointers_only",
    }


def validate_capability(capability: dict[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "schema", "capability_id", "provider_id", "target_id", "operation", "graph_sha256",
        "api_graph_sha256", "models", "accepted_controls", "output_contract", "health_policy",
        "admission_limit", "graph_mutation",
    }
    if type(capability) is not dict or set(capability) != expected_keys:
        raise ContractError("capability-shape")
    if capability != build_capability():
        raise ContractError("capability-mismatch")
    return copy.deepcopy(capability)


def _validate_artifact_summary(value: Any) -> None:
    required = {"material_id", "artifact_sha256", "byte_size", "width", "height", "format", "mime_type", "artifact_version"}
    if type(value) is not dict or set(value) != required:
        raise ContractError("artifact-summary-shape")
    _id_value(value["material_id"], "imgmat_", "material-id")
    _hex64(value["artifact_sha256"], "artifact-sha256")
    if value["format"] != "png" or value["mime_type"] != "image/png":
        raise ContractError("artifact-summary-format")
    _strict_int(value["byte_size"], "byte-size", 1, MAX_BYTES)
    _strict_int(value["width"], "width", 1, MAX_DIMENSION)
    _strict_int(value["height"], "height", 1, MAX_DIMENSION)
    _strict_int(value["artifact_version"], "artifact-version", 1, 9_007_199_254_740_991)
    if value["artifact_version"] != 1:
        raise ContractError("artifact-version")


def validate_job(job: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema", "image_producer_job_id", "idempotency_key", "client_request_id", "request_snapshot",
        "request_snapshot_sha256", "capability_manifest_sha256", "api_graph_sha256", "context_basis",
        "attach_policy", "provider", "submit_intent", "state", "attempt", "submission_count", "artifact",
        "revision_count", "retry_keys", "quarantine_record_ids",
    }
    optional = {
        "last_quarantine_record_id", "quarantine_reason", "failure_reason", "publication_path",
        "attached_target", "attached_revision_id", "pending_bytes_sha256", "pending_output",
        "result_evidence", "submit_reconciliation", "sent_prompt",
    }
    if type(job) is not dict or not required.issubset(job) or set(job) - required - optional:
        raise ContractError("job-shape")
    if "sent_prompt" in job:
        _strict_str(job["sent_prompt"], "sent-prompt", 0, 16_384)
    if job["schema"] != IMAGE_PRODUCER_JOB_SCHEMA:
        raise ContractError("job-schema")
    _id_value(job["image_producer_job_id"], "imgjob_", "image-producer-job-id")
    _strict_token(job["idempotency_key"], "idempotency-key")
    _strict_token(job["client_request_id"], "client-request-id")
    snapshot_bytes = validate_snapshot(job["request_snapshot"])
    identity = canonical_bytes({
        "idempotency_key": job["idempotency_key"],
        "request_snapshot": job["request_snapshot"],
        "capability_manifest_sha256": job["request_snapshot"]["capability_manifest_sha256"],
        "api_graph_sha256": job["request_snapshot"]["api_graph_sha256"],
        "context_basis": job["request_snapshot"]["context_basis"],
        "attach_policy": job["request_snapshot"]["attach_policy"],
    })
    if job["image_producer_job_id"] != _id("imgjob_", identity):
        raise ContractError("job-id-identity")
    snapshot_hash = sha256(snapshot_bytes)
    if job["request_snapshot_sha256"] != snapshot_hash:
        raise ContractError("request-snapshot-hash")
    _hex64(job["capability_manifest_sha256"], "capability-manifest-sha256")
    _hex64(job["api_graph_sha256"], "api-graph-sha256")
    if job["capability_manifest_sha256"] != job["request_snapshot"]["capability_manifest_sha256"] or job["api_graph_sha256"] != job["request_snapshot"]["api_graph_sha256"]:
        raise ContractError("job-fingerprint")
    if job["context_basis"] != job["request_snapshot"]["context_basis"] or job["attach_policy"] != job["request_snapshot"]["attach_policy"]:
        raise ContractError("job-snapshot-identity")
    validate_basis(job["context_basis"])
    validate_policy(job["attach_policy"])
    provider = job["provider"]
    if type(provider) is not dict or set(provider) != {"provider_id", "target_id", "client_request_id", "prompt_id"}:
        raise ContractError("provider-shape")
    if provider["provider_id"] != PROVIDER_ID or provider["target_id"] != snapshot_target_id(job["request_snapshot"]):
        raise ContractError("provider-identity")
    _strict_token(provider["client_request_id"], "provider-client-request-id")
    if provider["prompt_id"] is not None:
        _strict_str(provider["prompt_id"], "provider-prompt-id", 1, 256)
    intent = job["submit_intent"]
    if type(intent) is not dict or set(intent) != {"attempt", "client_request_id", "request_snapshot_sha256"}:
        raise ContractError("submit-intent-shape")
    _strict_int(intent["attempt"], "attempt", 1, 16)
    _strict_token(intent["client_request_id"], "submit-client-request-id")
    _hex64(intent["request_snapshot_sha256"], "submit-snapshot-hash")
    if intent["attempt"] != job["attempt"] or intent["client_request_id"] != provider["client_request_id"] or intent["request_snapshot_sha256"] != snapshot_hash:
        raise ContractError("submit-intent-conflict")
    if job["state"] not in STATES:
        raise ContractError("job-state")
    _strict_int(job["attempt"], "attempt", 1, 16)
    _strict_int(job["submission_count"], "submission-count", 0, 16)
    if job["submission_count"] > job["attempt"]:
        raise ContractError("submission-count")
    if job["artifact"] is not None:
        _validate_artifact_summary(job["artifact"])
    _strict_int(job["revision_count"], "revision-count", 0, 1)
    if type(job["retry_keys"]) is not list or len(job["retry_keys"]) > 15 or len(set(job["retry_keys"])) != len(job["retry_keys"]):
        raise ContractError("retry-keys")
    for key in job["retry_keys"]:
        _strict_token(key, "retry-key")
    if type(job["quarantine_record_ids"]) is not list or len(job["quarantine_record_ids"]) > 64:
        raise ContractError("quarantine-record-ids")
    for record_id in job["quarantine_record_ids"]:
        _id_value(record_id, "imgq_", "quarantine-record-id")
    if len(set(job["quarantine_record_ids"])) != len(job["quarantine_record_ids"]):
        raise ContractError("quarantine-record-ids")
    if "last_quarantine_record_id" in job:
        _id_value(job["last_quarantine_record_id"], "imgq_", "last-quarantine-record-id")
    for field in ("quarantine_reason", "failure_reason"):
        if field in job:
            _strict_str(job[field], field.replace("_", "-"), 1, 160)
    if "publication_path" in job and (type(job["publication_path"]) is not str or re.fullmatch(r"materials/[0-9a-f]{64}\.png", job["publication_path"]) is None):
        raise ContractError("publication-path")
    has_attached_target = "attached_target" in job
    has_attached_revision = "attached_revision_id" in job
    if has_attached_target != has_attached_revision:
        raise ContractError("attached-projection-pair")
    if has_attached_target:
        target = job["attached_target"]
        if type(target) is not list or len(target) != 3:
            raise ContractError("attached-target")
        _id_value(target[0], "imgctx_", "attached-context-id")
        _id_value(target[1], "imgwork_", "attached-work-id")
        if target[2] is not None:
            _id_value(target[2], "imgrev_", "attached-revision-id")
        _id_value(job["attached_revision_id"], "imgrev_", "attached-revision-id")
        basis = job["context_basis"]
        expected_target = [basis["context_id"], basis["work_id"], basis["revision_id"]]
        if job["state"] != "published" or target != expected_target:
            raise ContractError("attached-target-authority")
        expected_revision_id = "imgrev_" + sha256(canonical_bytes({
            "job_id": job["image_producer_job_id"], "target": expected_target,
        }))[:32]
        if job["attached_revision_id"] != expected_revision_id:
            raise ContractError("attached-revision-authority")
    has_pending_hash = "pending_bytes_sha256" in job
    has_pending_output = "pending_output" in job
    if has_pending_hash != has_pending_output:
        raise ContractError("pending-output-pair")
    output = None
    if has_pending_hash:
        _hex64(job["pending_bytes_sha256"], "pending-bytes-sha256")
        output = job["pending_output"]
        if type(output) is not dict or set(output) != {"format", "mime_type", "byte_size", "width", "height", "pixel_count", "sha256", "chunks"}:
            raise ContractError("pending-output-shape")
        _hex64(output["sha256"], "pending-output-sha256")
        _strict_int(output["byte_size"], "byte-size", 1, MAX_BYTES)
        _strict_int(output["width"], "width", 1, MAX_DIMENSION)
        _strict_int(output["height"], "height", 1, MAX_DIMENSION)
        _strict_int(output["pixel_count"], "pixel-count", 1, MAX_PIXELS)
        _strict_int(output["chunks"], "chunks", 1, 2**31 - 1)
        if output["format"] != "png" or output["mime_type"] != "image/png":
            raise ContractError("pending-output-format")
        if job["pending_bytes_sha256"] != output["sha256"]:
            raise ContractError("pending-output-hash-mismatch")
        requested = job["request_snapshot"]["artist_input"]
        if output["width"] != requested["width"] or output["height"] != requested["height"]:
            raise ContractError("pending-output-requested-dimensions")
        if output["pixel_count"] != output["width"] * output["height"]:
            raise ContractError("pending-output-pixel-count")
    if "submit_reconciliation" in job:
        reconciliation = job["submit_reconciliation"]
        if type(reconciliation) is not dict:
            raise ContractError("submit-reconciliation-shape")
        expected = {"status", "attempt", "client_request_id", "request_snapshot_sha256"}
        if reconciliation.get("status") == "bound":
            expected.add("provider_prompt_id")
        if set(reconciliation) != expected or reconciliation["status"] not in ("unresolved", "bound"):
            raise ContractError("submit-reconciliation-shape")
        _strict_int(reconciliation["attempt"], "reconciliation-attempt", 1, 16)
        _strict_token(reconciliation["client_request_id"], "reconciliation-client-request-id")
        _hex64(reconciliation["request_snapshot_sha256"], "reconciliation-snapshot-hash")
        if reconciliation["attempt"] != intent["attempt"] or reconciliation["client_request_id"] != intent["client_request_id"] or reconciliation["request_snapshot_sha256"] != intent["request_snapshot_sha256"]:
            raise ContractError("submit-reconciliation-conflict")
        if reconciliation["status"] == "bound":
            _strict_str(reconciliation["provider_prompt_id"], "reconciliation-provider-prompt-id", 1, 256)
            if reconciliation["provider_prompt_id"] != provider["prompt_id"] or provider["prompt_id"] is None:
                raise ContractError("submit-reconciliation-provider-conflict")
        elif provider["prompt_id"] is not None:
            raise ContractError("submit-reconciliation-provider-conflict")
    if "result_evidence" in job:
        evidence = job["result_evidence"]
        required_evidence = {
            "provider_id", "provider_prompt_id", "target_id", "capability_manifest_sha256",
            "api_graph_sha256", "request_snapshot_sha256", "context_basis", "result_sha256",
        }
        if type(evidence) is not dict or set(evidence) != required_evidence:
            raise ContractError("result-evidence-shape")
        if evidence["provider_id"] != PROVIDER_ID or evidence["target_id"] != provider["target_id"]:
            raise ContractError("result-evidence-authority")
        _strict_str(evidence["provider_prompt_id"], "evidence-provider-prompt-id", 1, 256)
        _hex64(evidence["capability_manifest_sha256"], "evidence-capability-hash")
        _hex64(evidence["api_graph_sha256"], "evidence-graph-hash")
        _hex64(evidence["request_snapshot_sha256"], "evidence-snapshot-hash")
        _hex64(evidence["result_sha256"], "evidence-result-hash")
        validate_basis(evidence["context_basis"])
        if (
            evidence["provider_prompt_id"] != provider["prompt_id"]
            or provider["prompt_id"] is None
            or evidence["provider_id"] != provider["provider_id"]
            or evidence["target_id"] != provider["target_id"]
            or evidence["capability_manifest_sha256"] != job["capability_manifest_sha256"]
            or evidence["api_graph_sha256"] != job["api_graph_sha256"]
            or evidence["request_snapshot_sha256"] != job["request_snapshot_sha256"]
            or evidence["context_basis"] != job["context_basis"]
        ):
            raise ContractError("result-evidence-conflict")
        if output is not None and evidence["result_sha256"] != output["sha256"]:
            raise ContractError("result-evidence-output-conflict")

    state = job["state"]
    prompt_id = provider["prompt_id"]
    submission_count = job["submission_count"]
    if state == "accepted" and (prompt_id is not None or submission_count != 0):
        raise ContractError("accepted-state-conflict")
    if state in {"queued", "running", "ready", "ingested", "published"}:
        if prompt_id is None or submission_count < job["attempt"]:
            raise ContractError("provider-bound-state-conflict")

    has_result_evidence = "result_evidence" in job
    has_publication_path = "publication_path" in job
    if state in {"accepted", "submitting"} and (
        has_pending_output or has_result_evidence or has_publication_path or job["artifact"] is not None
    ):
        raise ContractError("state-evidence-conflict")
    if state in {"queued", "running", "ready", "failed", "cancelled", "quarantined"} and (
        has_pending_output or has_result_evidence or has_publication_path or job["artifact"] is not None
    ):
        raise ContractError("state-evidence-conflict")
    if state == "ingested":
        if not has_pending_output or not has_result_evidence or job["artifact"] is not None or has_publication_path:
            raise ContractError("ingested-state-evidence")
    if state == "published":
        if not has_pending_output or not has_result_evidence or job["artifact"] is None or not has_publication_path:
            raise ContractError("published-state-evidence")
        artifact = job["artifact"]
        if (
            output is None
            or artifact["artifact_sha256"] != output["sha256"]
            or artifact["byte_size"] != output["byte_size"]
            or artifact["width"] != output["width"]
            or artifact["height"] != output["height"]
            or artifact["format"] != output["format"]
            or artifact["mime_type"] != output["mime_type"]
        ):
            raise ContractError("published-output-conflict")
        if job["publication_path"] != f"materials/{artifact['artifact_sha256']}.png":
            raise ContractError("publication-path-artifact-conflict")

    if "submit_reconciliation" in job:
        reconciliation = job["submit_reconciliation"]
        if reconciliation["status"] == "unresolved":
            if state != "quarantined" or job.get("quarantine_reason") != "submit-reconciliation-required" or prompt_id is not None:
                raise ContractError("submit-reconciliation-state")
        elif state in {"accepted", "submitting"}:
            raise ContractError("submit-reconciliation-state")

    quarantine_ids = job["quarantine_record_ids"]
    if quarantine_ids:
        if "last_quarantine_record_id" in job and job["last_quarantine_record_id"] != quarantine_ids[-1]:
            raise ContractError("quarantine-record-order")
    elif "last_quarantine_record_id" in job:
        raise ContractError("quarantine-record-order")
    if state in {"accepted", "submitting", "queued", "running", "ready", "ingested"} and (
        "quarantine_reason" in job or "last_quarantine_record_id" in job
    ):
        raise ContractError("active-quarantine-marker-conflict")
    if state == "quarantined" and (
        not quarantine_ids
        or "quarantine_reason" not in job
        or "last_quarantine_record_id" not in job
    ):
        raise ContractError("quarantined-state-evidence")
    if state == "quarantined" and job["last_quarantine_record_id"] != quarantine_ids[-1]:
        raise ContractError("quarantine-record-order")
    if state not in {"quarantined", "published"} and "quarantine_reason" in job:
        raise ContractError("quarantine-marker-state")
    if quarantine_ids and state != "quarantined" and not (state == "published" and "quarantine_reason" in job):
        retry_history = job["attempt"] > 1 and len(job["retry_keys"]) == job["attempt"] - 1
        reconciliation = job.get("submit_reconciliation")
        bound_reconciliation = (
            type(reconciliation) is dict
            and reconciliation.get("status") == "bound"
            and reconciliation.get("attempt") == job["attempt"]
            and reconciliation.get("client_request_id") == intent["client_request_id"]
            and reconciliation.get("request_snapshot_sha256") == job["request_snapshot_sha256"]
            and provider["prompt_id"] is not None
            and job["submission_count"] >= job["attempt"]
        )
        if not retry_history and not bound_reconciliation:
            raise ContractError("quarantine-history-unproven")
    return copy.deepcopy(job)


def build_generated_artifact_ref(published_job: dict[str, Any]) -> dict[str, Any]:
    """Derive the generated provenance record from one validated published job."""
    job = validate_job(published_job)
    if job["state"] != "published":
        raise ContractError("artifact-ref-job-not-published")
    artifact = job["artifact"]
    if type(artifact) is not dict:
        raise ContractError("artifact-ref-job-artifact")
    digest = artifact["artifact_sha256"]
    expected_material_id = "imgmat_" + hashlib.sha256(digest.encode("ascii")).hexdigest()[:32]
    if artifact["material_id"] != expected_material_id:
        raise ContractError("artifact-ref-material-identity")
    if job["publication_path"] != f"materials/{digest}.png":
        raise ContractError("artifact-ref-publication-path")
    return validate_artifact_ref({
        "schema": IMAGE_ARTIFACT_SCHEMA,
        "material_id": artifact["material_id"],
        "origin": "generated",
        "producer_job_id": job["image_producer_job_id"],
        "producer_job_schema": IMAGE_PRODUCER_JOB_SCHEMA,
        "request_snapshot_sha256": job["request_snapshot_sha256"],
        "artifact_sha256": digest,
        "format": artifact["format"],
        "mime_type": artifact["mime_type"],
        "byte_size": artifact["byte_size"],
        "width": artifact["width"],
        "height": artifact["height"],
        "artifact_version": artifact["artifact_version"],
    })


def _validate_reference(value: Any) -> None:
    required = {"reference_id", "material_id", "role", "instruction", "preserve"}
    if type(value) is not dict or set(value) != required:
        raise ContractError("reference-shape")
    _id_value(value["reference_id"], "imgref_", "reference-id")
    _id_value(value["material_id"], "imgmat_", "reference-material-id")
    for field, high in (("role", 80), ("instruction", 2048), ("preserve", 2048)):
        _strict_str(value[field], field, 0, high)


def _validate_imported(value: Any) -> None:
    if type(value) is not dict or set(value) != {"material_id", "origin", "job_id"} or value["origin"] != "imported" or value["job_id"] is not None:
        raise ContractError("imported-artifact-shape")
    _id_value(value["material_id"], "imgmat_", "imported-material-id")


def _validate_content(value: Any, *, draft: bool) -> None:
    required = {"intent", "intent_sha256", "notes", "references", "alternatives"}
    if draft:
        required |= {"based_on_revision_id", "draft_generation"}
    if type(value) is not dict or set(value) != required:
        raise ContractError("revision-content-shape")
    _strict_str(value["intent"], "intent", 0, 16_384)
    _hex64(value["intent_sha256"], "intent-sha256")
    if value["intent_sha256"] != sha256(value["intent"].encode("utf-8")):
        raise ContractError("intent-sha256")
    _strict_str(value["notes"], "notes", 0, 16_384)
    if type(value["references"]) is not list or len(value["references"]) > 32:
        raise ContractError("references-shape")
    for reference in value["references"]:
        _validate_reference(reference)
    if type(value["alternatives"]) is not list or len(value["alternatives"]) > 128:
        raise ContractError("alternatives-shape")
    for alternative in value["alternatives"]:
        if draft:
            _validate_imported(alternative)
        elif type(alternative) is dict and alternative.get("origin") == "generated":
            validate_artifact_ref(alternative)
        else:
            _validate_imported(alternative)
    if draft:
        if value["based_on_revision_id"] is not None:
            _id_value(value["based_on_revision_id"], "imgrev_", "based-on-revision-id")
        _strict_int(value["draft_generation"], "draft-generation", 0, 9_007_199_254_740_991)


def _validate_receipt(value: Any) -> None:
    required = {
        "operation_id", "operation_kind", "issued_at", "expires_at", "payload_sha256",
        "expected_revision", "target_scope", "status", "committed_revision", "result",
    }
    if type(value) is not dict or set(value) != required:
        raise ContractError("receipt-shape")
    _id_value(value["operation_id"], "imgop_", "operation-id")
    if value["operation_kind"] not in {
        "image_create_work", "image_update_draft", "image_save_revision", "image_branch_revision",
        "image_register_material", "image_relink_material", "film_image_adopt",
    }:
        raise ContractError("operation-kind")
    for field in ("issued_at", "expires_at"):
        if type(value[field]) is not str or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", value[field]) is None:
            raise ContractError(f"{field}-timestamp")
    _hex64(value["payload_sha256"], "payload-sha256")
    _strict_int(value["expected_revision"], "expected-revision", 0, 9_007_199_254_740_991)
    _strict_int(value["committed_revision"], "committed-revision", 1, 9_007_199_254_740_991)
    if value["status"] != "committed":
        raise ContractError("receipt-status")
    result_required = {"context_id", "work_id", "revision_id", "material_id", "binding_id", "adoption_id"}
    result = value["result"]
    if type(result) is not dict or set(result) != result_required:
        raise ContractError("receipt-result-shape")
    for field, prefix in (("context_id", "imgctx_"), ("work_id", "imgwork_"), ("revision_id", "imgrev_"), ("material_id", "imgmat_"), ("binding_id", "imgbind_"), ("adoption_id", "imgadopt_")):
        if result[field] is not None:
            _id_value(result[field], prefix, field)
    scope = value["target_scope"]
    if type(scope) is not dict or "scope_kind" not in scope:
        raise ContractError("target-scope-shape")
    if scope["scope_kind"] == "image_context":
        if set(scope) != {"scope_kind", "context_id", "work_id", "material_id"}:
            raise ContractError("image-scope-shape")
        _id_value(scope["context_id"], "imgctx_", "scope-context-id")
        for field, prefix in (("work_id", "imgwork_"), ("material_id", "imgmat_")):
            if scope[field] is not None:
                _id_value(scope[field], prefix, f"scope-{field}")
    elif scope["scope_kind"] == "film_binding":
        if set(scope) != {"scope_kind", "project_id", "binding_id", "film_reference_id"}:
            raise ContractError("film-scope-shape")
        _id_value(scope["project_id"], "prj_", "scope-project-id")
        if scope["binding_id"] is not None:
            _id_value(scope["binding_id"], "imgbind_", "scope-binding-id")
        _id_value(scope["film_reference_id"], "filmref_", "scope-film-reference-id")
    else:
        raise ContractError("target-scope-kind")


def validate_imported_artifact_ref(ref: dict[str, Any]) -> dict[str, Any]:
    """Validate the unchanged v1 imported-only artifact shape."""
    required = {"material_id", "origin", "job_id"}
    if type(ref) is not dict or set(ref) != required or ref["origin"] != "imported" or ref["job_id"] is not None:
        raise ContractError("imported-artifact-provenance")
    _id_value(ref["material_id"], "imgmat_", "material-id")
    return copy.deepcopy(ref)


def validate_image_state(state: dict[str, Any]) -> dict[str, Any]:
    required = {"schema", "context_id", "revision", "updated_at", "receipt_key_fingerprint", "receipt_key_enrollment_id", "state"}
    if type(state) is not dict or set(state) != required or state["schema"] != IMAGE_STATE_SCHEMA:
        raise ContractError("image-state-shape")
    _id_value(state["context_id"], "imgctx_", "context-id")
    _strict_int(state["revision"], "state-revision", 1, 9_007_199_254_740_991)
    if type(state["updated_at"]) is not str or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", state["updated_at"]) is None:
        raise ContractError("state-timestamp")
    _hex64(state["receipt_key_fingerprint"], "receipt-key-fingerprint")
    _id_value(state["receipt_key_enrollment_id"], "imgkey_", "receipt-key-enrollment-id")
    body = state["state"]
    if type(body) is not dict or set(body) != {"works", "materials", "receipts", "receipt_floor_at"}:
        raise ContractError("state-body-shape")
    if type(body["works"]) is not list or len(body["works"]) > 128:
        raise ContractError("works-shape")
    for work in body["works"]:
        if type(work) is not dict or set(work) != {"work_id", "name", "draft", "saved_revisions"}:
            raise ContractError("work-shape")
        _id_value(work["work_id"], "imgwork_", "work-id")
        _strict_str(work["name"], "work-name", 1, 160)
        _validate_content(work["draft"], draft=True)
        if type(work["saved_revisions"]) is not list or len(work["saved_revisions"]) > 256:
            raise ContractError("saved-revisions-shape")
        for revision in work["saved_revisions"]:
            required_revision = {"revision_id", "work_id", "parent_revision_id", "saved_at", "content", "selected_artifact", "material_ids"}
            if type(revision) is not dict or set(revision) != required_revision:
                raise ContractError("saved-revision-shape")
            _id_value(revision["revision_id"], "imgrev_", "revision-id")
            _id_value(revision["work_id"], "imgwork_", "revision-work-id")
            if revision["parent_revision_id"] is not None:
                _id_value(revision["parent_revision_id"], "imgrev_", "parent-revision-id")
            if type(revision["saved_at"]) is not str or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", revision["saved_at"]) is None:
                raise ContractError("saved-at")
            _validate_content(revision["content"], draft=False)
            if revision["selected_artifact"] is not None:
                _id_value(revision["selected_artifact"], "imgmat_", "selected-artifact")
            if type(revision["material_ids"]) is not list or len(revision["material_ids"]) > 128 or len(set(revision["material_ids"])) != len(revision["material_ids"]):
                raise ContractError("material-ids-shape")
            for material_id in revision["material_ids"]:
                _id_value(material_id, "imgmat_", "material-id")
    if type(body["materials"]) is not list or len(body["materials"]) > 4096:
        raise ContractError("materials-shape")
    for material in body["materials"]:
        required_material = {"material_id", "sha256", "media_type", "label", "availability"}
        if type(material) is not dict or set(material) != required_material:
            raise ContractError("material-shape")
        _id_value(material["material_id"], "imgmat_", "material-id")
        _hex64(material["sha256"], "material-sha256")
        if material["media_type"] != "image" or material["availability"] not in {"available", "missing", "denied"}:
            raise ContractError("material-authority")
        _strict_str(material["label"], "material-label", 0, 260)
    if type(body["receipts"]) is not list or len(body["receipts"]) > 128:
        raise ContractError("receipts-shape")
    for receipt in body["receipts"]:
        _validate_receipt(receipt)
    if type(body["receipt_floor_at"]) is not str or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", body["receipt_floor_at"]) is None:
        raise ContractError("receipt-floor")
    return copy.deepcopy(state)


@dataclass(frozen=True)
class ImageRequest:
    """Typed request envelope used before a job is admitted to the ledger."""

    idempotency_key: str
    request_snapshot: Mapping[str, Any]
    client_request_id: str | None = None

    def __post_init__(self) -> None:
        _strict_token(self.idempotency_key, "idempotency-key")
        snapshot = copy.deepcopy(dict(self.request_snapshot))
        validate_snapshot(snapshot)
        object.__setattr__(self, "request_snapshot", snapshot)
        client = self.idempotency_key if self.client_request_id is None else self.client_request_id
        _strict_token(client, "client-request-id")
        object.__setattr__(self, "client_request_id", client)

    def to_dict(self) -> dict[str, Any]:
        return {
            "idempotency_key": self.idempotency_key,
            "client_request_id": self.client_request_id,
            "request_snapshot": copy.deepcopy(dict(self.request_snapshot)),
        }


@dataclass(frozen=True)
class ImageProducerJob:
    """Validated typed view of a durable producer job envelope."""

    value: Mapping[str, Any]

    def __post_init__(self) -> None:
        normalized = validate_job(dict(self.value))
        object.__setattr__(self, "value", normalized)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.value))

    @property
    def image_producer_job_id(self) -> str:
        return self.value["image_producer_job_id"]


@dataclass(frozen=True)
class ImageProducerOperation:
    """Validated typed view of a closed producer operation wire."""

    value: Mapping[str, Any]

    def __post_init__(self) -> None:
        normalized = validate_operation(dict(self.value))
        object.__setattr__(self, "value", normalized)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.value))


ImageProducerJobEnvelope = ImageProducerJob
ImageProducerOperationEnvelope = ImageProducerOperation

__all__ = [
    "IMAGE_CAPABILITY_SCHEMA", "IMAGE_PRODUCER_JOB_SCHEMA", "IMAGE_ARTIFACT_SCHEMA", "IMAGE_STATE_SCHEMA", "IMAGE_OPERATION_SCHEMA",
    "PROVIDER_ID", "TARGET_ID", "CAPABILITY_ID", "GRAPH_SHA256", "CAPABILITY_SHA256", "MAX_BYTES", "MAX_DIMENSION", "MAX_PIXELS",
    "LEGAL_TRANSITIONS", "ContractError", "ImageContractError", "ImageSchemaError", "canonical_bytes", "parse_canonical", "sha256",
    "build_snapshot", "validate_snapshot", "validate_basis", "validate_policy", "validate_output", "validate_png_byte_size",
    "build_generated_artifact_ref", "validate_artifact_ref", "validate_imported_artifact_ref", "validate_operation", "build_capability", "validate_capability", "validate_job", "validate_image_state",
    "ImageRequest", "ImageProducerJob", "ImageProducerJobEnvelope", "ImageProducerOperation", "ImageProducerOperationEnvelope",
]
