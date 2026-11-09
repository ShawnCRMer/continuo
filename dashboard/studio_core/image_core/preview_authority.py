"""Bounded, in-process authority for deliberate Image-to-Film previews.

The authority owns only ephemeral preview/preparation evidence.  It never writes
an ImageState or FilmBindingStore revision.  Durable adoption is implemented by
``film_adoption.FilmAdoptionService``.
"""
from __future__ import annotations

import base64
import copy
import datetime as _datetime
import hashlib
import hmac
import json
import re
import secrets
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    from .image_store import HmacImageOwnerAuthority, ImageStateStore
    from .image_store import contract as _image_contract
    from .film_binding_store import FilmBindingStore, FilmBindingStoreError
    from .film_binding_contract import MAX_RECEIPTS
except (ImportError, AttributeError):  # pragma: no cover - direct module probes
    HmacImageOwnerAuthority = None
    ImageStateStore = None
    FilmBindingStore = None
    _image_contract = None
    FilmBindingStoreError = ()
    MAX_RECEIPTS = 128


V2_SCHEMA = "studio.image-operation/2"
BRANCH_VERSION = "new_reference/1"
PREVIEW_KIND = "film_image_preview_new_reference"
PREVIEW_READY_KIND = "film_image_preview_new_reference_ready"
CANCEL_KIND = "film_image_preview_cancel"
CANCELLED_KIND = "film_image_preview_cancelled"
PREPARE_KIND = "image_prepare_operation"
ADOPT_KIND = "film_image_adopt_new_reference"
PREPARED_KIND = "image_operation_prepared"
RECEIPT_KIND = "image_operation_receipt"
STATUS_KIND = "image_operation_status"
STATUS_RESULT_KIND = "image_operation_status_result"
ERROR_KIND = "image_operation_error"

MAX_SAFE_INTEGER = 9_007_199_254_740_991
MAX_WIRE_BYTES = 512 * 1024
MAX_PENDING_PREVIEWS = 128
MAX_PENDING_OPERATIONS = 128
PREVIEW_TTL_SECONDS = 10 * 60
OPERATION_TTL_SECONDS = 24 * 60 * 60
TOKEN_BYTES = 32

_CONTEXT_RE = re.compile(r"^imgctx_[0-9a-f]{32}$")
_WORK_RE = re.compile(r"^imgwork_[0-9a-f]{32}$")
_REVISION_RE = re.compile(r"^imgrev_[0-9a-f]{32}$")
_MATERIAL_RE = re.compile(r"^imgmat_[0-9a-f]{32}$")
_BINDING_RE = re.compile(r"^imgbind_[0-9a-f]{32}$")
_ADOPTION_RE = re.compile(r"^imgadopt_[0-9a-f]{32}$")
_FILM_REFERENCE_RE = re.compile(r"^filmref_[0-9a-f]{32}$")
_OPERATION_RE = re.compile(r"^imgop_[0-9a-f]{32}$")
_PROJECT_RE = re.compile(r"^prj_[0-9a-f]{32}$")
_ENROLLMENT_RE = re.compile(r"^imgkey_[0-9a-f]{32}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{43}$")
_TIMESTAMP_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
_OWNER_RE = re.compile(r"^film_owner:prj_[0-9a-f]{32}$")


class ImageAdoptionError(ValueError):
    """Stable fail-closed error for malformed public adoption values."""

    def __init__(self, code: str) -> None:
        self.code = code
        self.reason_code = code
        super().__init__(code)


PreviewAuthorityError = ImageAdoptionError


@dataclass(slots=True)
class _PreviewRecord:
    session_id: str
    request: dict[str, Any]
    response: dict[str, Any]
    snapshot: dict[str, Any]
    snapshot_sha256: str
    target: Any
    issued_at: str
    expires_epoch: float
    state: str = "open"
    operation_id: str | None = None


@dataclass(slots=True)
class _OperationRecord:
    session_id: str
    token: str
    projection: dict[str, Any]
    prepared: dict[str, Any]
    expires_epoch: float
    state: str = "pending"
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class FilmTemporalEvidence:
    """Validated temporal evidence from one persisted Film state."""

    updated_at: str
    highwater_at: str


def validate_film_state_temporal(state: Any) -> FilmTemporalEvidence | None:
    """Validate the complete bounded temporal relation of a persisted Film state."""
    if state is None:
        return None
    try:
        if type(state) is not dict:
            raise ValueError("film state shape")
        updated_at = state["updated_at"]
        body = state["state"]
        if type(body) is not dict:
            raise ValueError("film state body shape")
        if type(updated_at) is not str or _TIMESTAMP_RE.fullmatch(updated_at) is None:
            raise ValueError("film updated timestamp")
        updated_value = _datetime.datetime.strptime(updated_at, "%Y-%m-%dT%H:%M:%SZ")
        floor = body["receipt_floor_at"]
        if type(floor) is not str or _TIMESTAMP_RE.fullmatch(floor) is None:
            raise ValueError("receipt floor timestamp")
        floor_value = _datetime.datetime.strptime(floor, "%Y-%m-%dT%H:%M:%SZ")
        if floor_value > updated_value:
            raise ValueError("receipt floor after publication")
        receipts = body["receipts"]
        if type(receipts) is not list or len(receipts) > MAX_RECEIPTS:
            raise ValueError("receipt list")
        highwater = floor
        highwater_value = floor_value
        for receipt in receipts:
            if type(receipt) is not dict:
                raise ValueError("receipt shape")
            issued_at = receipt["issued_at"]
            expires_at = receipt["expires_at"]
            if (
                type(issued_at) is not str
                or _TIMESTAMP_RE.fullmatch(issued_at) is None
                or type(expires_at) is not str
                or _TIMESTAMP_RE.fullmatch(expires_at) is None
            ):
                raise ValueError("receipt timestamp")
            issued_value = _datetime.datetime.strptime(issued_at, "%Y-%m-%dT%H:%M:%SZ")
            expires_value = _datetime.datetime.strptime(expires_at, "%Y-%m-%dT%H:%M:%SZ")
            if issued_value > updated_value:
                raise ValueError("receipt issued after publication")
            if expires_value != issued_value + _datetime.timedelta(seconds=OPERATION_TTL_SECONDS):
                raise ValueError("receipt expiry is not correlated to issuance")
            if issued_value > highwater_value:
                highwater = issued_at
                highwater_value = issued_value
        return FilmTemporalEvidence(updated_at=updated_at, highwater_at=highwater)
    except ImageAdoptionError:
        raise
    except Exception as exc:
        raise ImageAdoptionError("image_state_unavailable") from exc


class PreviewAuthority:
    """Hold bounded session-only preview and prepare authority.

    ``image_store`` and ``binding_store`` are already-authorized owner objects.
    The current caller/session is passed out of band; it is never serialized in
    the closed operation dictionaries.
    """

    _FILM_STORE_INSTANCE_METHODS = frozenset({
        "load", "read", "get_current", "current_revision", "mint_binding_target", "mint_ids", "abandon_binding_target",
        "publish_initial", "append_initial", "publish_next", "append_next", "upgrade_v1_to_v2",
    })

    @classmethod
    def _require_initialized_film_store(cls, binding_store: Any) -> dict[str, Any]:
        if FilmBindingStore is None or type(binding_store) is not FilmBindingStore:
            raise ImageAdoptionError("film-binding-store-required")
        try:
            values = {
                name: object.__getattribute__(binding_store, name)
                for name in (
                    "project_dir", "boundary", "authority", "clock", "project_id",
                    "receipt_key_enrollment_id", "receipt_key_fingerprint", "_project_identity",
                    "_state_identity", "_issued_targets", "_target_expiry_heap", "_target_sequence",
                )
            }
            instance_keys = set(vars(binding_store))
        except Exception as exc:
            raise ImageAdoptionError("film-binding-store-required") from exc
        if instance_keys & cls._FILM_STORE_INSTANCE_METHODS:
            raise ImageAdoptionError("film-binding-store-required")
        if (
            not isinstance(values["project_dir"], Path)
            or values["boundary"] is None
            or not callable(values["clock"])
            or values["_project_identity"] is None
            or values["_state_identity"] is not None and not hasattr(values["_state_identity"], "is_directory")
            or type(values["_issued_targets"]) is not dict
            or type(values["_target_expiry_heap"]) is not list
            or type(values["_target_sequence"]) is not int
            or values["_target_sequence"] < 0
        ):
            raise ImageAdoptionError("film-binding-store-required")
        _validate_id(values["project_id"], _PROJECT_RE, "project-id-invalid")
        enrollment_id, fingerprint = cls._validate_film_authority(
            values["authority"],
            stored_enrollment_id=values["receipt_key_enrollment_id"],
            stored_fingerprint=values["receipt_key_fingerprint"],
        )
        if enrollment_id != values["receipt_key_enrollment_id"] or fingerprint != values["receipt_key_fingerprint"]:
            raise ImageAdoptionError("film-owner-enrollment-mismatch")
        values["owner_enrollment_id"] = enrollment_id
        values["owner_fingerprint"] = fingerprint
        return values

    def __init__(
        self,
        image_store: Any,
        binding_store: Any,
        *,
        clock: Callable[[], str] | None = None,
        token_factory: Callable[[int], bytes] | None = None,
        max_pending_previews: int = MAX_PENDING_PREVIEWS,
        max_pending_operations: int = MAX_PENDING_OPERATIONS,
    ) -> None:
        if image_store is None:
            raise ImageAdoptionError("image-store-required")
        if binding_store is None:
            raise ImageAdoptionError("binding-store-required")
        root, boundary, image_authority, root_identity = self._require_initialized_image_store(image_store)
        film_values = self._require_initialized_film_store(binding_store)
        project_id = film_values["project_id"]
        owner = film_values["authority"]
        enrollment_id = film_values["owner_enrollment_id"]
        fingerprint = film_values["owner_fingerprint"]
        stored_enrollment_id = film_values["receipt_key_enrollment_id"]
        stored_fingerprint = film_values["receipt_key_fingerprint"]
        if enrollment_id != stored_enrollment_id or fingerprint != stored_fingerprint:
            raise ImageAdoptionError("film-owner-enrollment-mismatch")
        self._pin_image_store(image_store, enrollment_id, fingerprint)
        self._image_contexts_root = root
        self._image_boundary = boundary
        self._image_authority = image_authority
        self._image_root_identity = root_identity
        if type(max_pending_previews) is not int or not 1 <= max_pending_previews <= MAX_PENDING_PREVIEWS:
            raise ImageAdoptionError("preview-capacity-invalid")
        if type(max_pending_operations) is not int or not 1 <= max_pending_operations <= MAX_PENDING_OPERATIONS:
            raise ImageAdoptionError("operation-capacity-invalid")
        self.image_store = image_store
        self.binding_store = binding_store
        self._film_project_dir = film_values["project_dir"]
        self._film_boundary = film_values["boundary"]
        self._film_owner_authority = owner
        self._film_clock = film_values["clock"]
        self._film_project_identity = film_values["_project_identity"]
        self._film_state_identity = film_values["_state_identity"]
        self._film_issued_targets = film_values["_issued_targets"]
        self._film_target_expiry_heap = film_values["_target_expiry_heap"]
        self._film_owner_enrollment_id = enrollment_id
        self._film_owner_fingerprint = fingerprint
        self.project_id = project_id
        self.owner_scope = f"film_owner:{project_id}"
        self.clock = clock or self._utc_second
        self._token_factory = token_factory or secrets.token_bytes
        self.max_pending_previews = max_pending_previews
        self.max_pending_operations = max_pending_operations
        self._previews: dict[str, _PreviewRecord] = {}
        self._operations: dict[str, _OperationRecord] = {}
        self._completed_sessions: dict[str, str] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _safe_dependency_attr(value: Any, name: str) -> Any:
        try:
            return getattr(value, name)
        except Exception as exc:
            raise ImageAdoptionError("image_state_unavailable") from exc

    @classmethod
    def _validate_film_authority(
        cls,
        authority: Any,
        *,
        stored_enrollment_id: Any = None,
        stored_fingerprint: Any = None,
    ) -> tuple[str, str]:
        if HmacImageOwnerAuthority is None or type(authority) is not HmacImageOwnerAuthority:
            raise ImageAdoptionError("film-owner-signer-required")
        try:
            sealed = object.__getattribute__(authority, "_sealed")
            key = object.__getattribute__(authority, "_key")
            enrollment_id = object.__getattribute__(authority, "receipt_key_enrollment_id")
            fingerprint = object.__getattribute__(authority, "receipt_key_fingerprint")
            actual_fingerprint = hashlib.sha256(key).hexdigest()
        except Exception as exc:
            raise ImageAdoptionError("film-owner-signer-required") from exc
        if (
            sealed is not True
            or type(key) is not bytes
            or len(key) != 32
            or type(enrollment_id) is not str
            or _ENROLLMENT_RE.fullmatch(enrollment_id) is None
            or type(fingerprint) is not str
            or _SHA256_RE.fullmatch(fingerprint) is None
            or not hmac.compare_digest(actual_fingerprint, fingerprint)
        ):
            raise ImageAdoptionError("film-owner-signer-required")
        if stored_enrollment_id is not None and enrollment_id != stored_enrollment_id:
            raise ImageAdoptionError("film-owner-enrollment-mismatch")
        if stored_fingerprint is not None and fingerprint != stored_fingerprint:
            raise ImageAdoptionError("film-owner-enrollment-mismatch")
        return enrollment_id, fingerprint

    @classmethod
    def _require_initialized_image_store(cls, image_store: Any) -> tuple[Path, Any, Any, Any]:
        if ImageStateStore is None or type(image_store) is not ImageStateStore:
            raise ImageAdoptionError("image-source-store-required")
        try:
            root = getattr(image_store, "image_contexts_root")
            boundary = getattr(image_store, "boundary")
            authority = getattr(image_store, "authority")
            clock = getattr(image_store, "clock")
            root_identity = getattr(image_store, "_root_identity")
        except Exception as exc:
            raise ImageAdoptionError("image-source-store-required") from exc
        if (
            not isinstance(root, Path)
            or root.name != "image-contexts"
            or boundary is None
            or not callable(clock)
            or root_identity is None
        ):
            raise ImageAdoptionError("image-source-store-required")
        cls._validate_film_authority(authority)
        try:
            ImageStateStore._check_root(image_store)
        except Exception as exc:
            raise ImageAdoptionError("image-source-store-required") from exc
        return root, boundary, authority, root_identity

    @classmethod
    def _pin_image_store(cls, image_store: Any, enrollment_id: str, fingerprint: str) -> None:
        cls._require_initialized_image_store(image_store)
        try:
            authority = getattr(image_store, "authority")
            source_enrollment = object.__getattribute__(authority, "receipt_key_enrollment_id")
            source_fingerprint = object.__getattribute__(authority, "receipt_key_fingerprint")
        except Exception as exc:
            raise ImageAdoptionError("image-source-store-required") from exc
        if source_enrollment != enrollment_id or source_fingerprint != fingerprint:
            raise ImageAdoptionError("film-owner-enrollment-mismatch")

    def _revalidate_film_store(self) -> None:
        """Recheck the accepted Film store boundary before every use."""
        store = self.binding_store
        if FilmBindingStore is None or type(store) is not FilmBindingStore:
            raise ImageAdoptionError("image_state_unavailable")
        try:
            if set(vars(store)) & self._FILM_STORE_INSTANCE_METHODS:
                raise ImageAdoptionError("image_state_unavailable")
            project_dir = object.__getattribute__(store, "project_dir")
            boundary = object.__getattribute__(store, "boundary")
            authority = object.__getattribute__(store, "authority")
            clock = object.__getattribute__(store, "clock")
            project_id = object.__getattribute__(store, "project_id")
            project_identity = object.__getattribute__(store, "_project_identity")
            state_identity = object.__getattribute__(store, "_state_identity")
            issued_targets = object.__getattribute__(store, "_issued_targets")
            target_expiry_heap = object.__getattribute__(store, "_target_expiry_heap")
            sequence = object.__getattribute__(store, "_target_sequence")
            enrollment_id = object.__getattribute__(store, "receipt_key_enrollment_id")
            fingerprint = object.__getattribute__(store, "receipt_key_fingerprint")
        except ImageAdoptionError:
            raise
        except Exception as exc:
            raise ImageAdoptionError("image_state_unavailable") from exc
        if (
            project_dir is not self._film_project_dir
            or boundary is not self._film_boundary
            or authority is not self._film_owner_authority
            or clock is not self._film_clock
            or project_id != self.project_id
            or project_identity is not self._film_project_identity
            or issued_targets is not self._film_issued_targets
            or type(target_expiry_heap) is not list
            or type(sequence) is not int
            or sequence < 0
            or enrollment_id != self._film_owner_enrollment_id
            or fingerprint != self._film_owner_fingerprint
        ):
            raise ImageAdoptionError("image_state_unavailable")
        if self._film_state_identity is not None and state_identity is not self._film_state_identity:
            raise ImageAdoptionError("image_state_unavailable")
        if self._film_state_identity is None and state_identity is not None:
            self._film_state_identity = state_identity
        self._validate_film_authority(
            authority,
            stored_enrollment_id=self._film_owner_enrollment_id,
            stored_fingerprint=self._film_owner_fingerprint,
        )

    def _film_store_load(self) -> dict[str, Any] | None:
        self._revalidate_film_store()
        try:
            result = FilmBindingStore.load(self.binding_store)
        except ImageAdoptionError:
            raise
        except Exception:
            raise
        self._revalidate_film_store()
        return result

    def _film_store_current_revision(self) -> int:
        self._revalidate_film_store()
        result = FilmBindingStore.current_revision(self.binding_store)
        self._revalidate_film_store()
        return result

    def _film_store_mint_binding_target(self) -> Any:
        self._revalidate_film_store()
        result = FilmBindingStore.mint_binding_target(self.binding_store)
        self._revalidate_film_store()
        return result

    def _film_store_abandon_binding_target(self, target: Any) -> Any:
        self._revalidate_film_store()
        result = FilmBindingStore.abandon_binding_target(self.binding_store, target)
        if result is not None or any(issued is target for issued in self._film_issued_targets):
            raise ImageAdoptionError("image_state_unavailable")
        self._revalidate_film_store()
        return result

    def _film_store_publish_initial(self, candidate: dict[str, Any], operation_id: str, *, target: Any) -> Any:
        self._revalidate_film_store()
        result = FilmBindingStore.publish_initial(self.binding_store, candidate, operation_id, target=target)
        self._revalidate_film_store()
        return result

    def _film_store_publish_next(self, expected_revision: int, candidate: dict[str, Any], operation_id: str, *, target: Any) -> Any:
        self._revalidate_film_store()
        result = FilmBindingStore.publish_next(
            self.binding_store, expected_revision, candidate, operation_id, target=target
        )
        self._revalidate_film_store()
        return result

    def _film_state(self) -> tuple[dict[str, Any] | None, FilmTemporalEvidence | None]:
        try:
            state = self._film_store_load()
            if state is None:
                return None, None
            if (
                state.get("receipt_key_enrollment_id") != self._film_owner_enrollment_id
                or state.get("receipt_key_fingerprint") != self._film_owner_fingerprint
            ):
                raise ImageAdoptionError("image_state_unavailable")
            temporal = validate_film_state_temporal(state)
            return state, temporal
        except ImageAdoptionError:
            raise
        except Exception as exc:
            raise ImageAdoptionError("image_state_unavailable") from exc

    def _revalidate_image_store(self) -> None:
        """Recheck the exact initialized source and its pinned root chain."""
        store = self.image_store
        if ImageStateStore is None or type(store) is not ImageStateStore:
            raise ImageAdoptionError("image-source-store-required")
        try:
            if (
                getattr(store, "image_contexts_root") is not self._image_contexts_root
                or getattr(store, "boundary") is not self._image_boundary
                or getattr(store, "authority") is not self._image_authority
                or getattr(store, "_root_identity") is not self._image_root_identity
            ):
                raise ImageAdoptionError("image-source-store-required")
            self._validate_film_authority(
                self._image_authority,
                stored_enrollment_id=self._film_owner_enrollment_id,
                stored_fingerprint=self._film_owner_fingerprint,
            )
            ImageStateStore._check_root(store)
        except ImageAdoptionError:
            raise
        except Exception as exc:
            raise ImageAdoptionError("image-source-store-required") from exc

    def __copy__(self) -> "PreviewAuthority":
        raise ImageAdoptionError("authority-not-copyable")

    def __deepcopy__(self, memo: dict[int, Any]) -> "PreviewAuthority":
        raise ImageAdoptionError("authority-not-copyable")

    def __reduce__(self) -> Any:
        raise ImageAdoptionError("authority-not-copyable")

    def __reduce_ex__(self, protocol: int) -> Any:
        raise ImageAdoptionError("authority-not-copyable")

    def __getstate__(self) -> Any:
        raise ImageAdoptionError("authority-not-copyable")

    @staticmethod
    def _utc_second() -> str:
        return _datetime.datetime.now(_datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def _now(self) -> tuple[str, float]:
        try:
            value = self.clock()
            if type(value) is not str or _TIMESTAMP_RE.fullmatch(value) is None:
                raise ValueError("invalid timestamp")
            parsed = _datetime.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
            return value, parsed.replace(tzinfo=_datetime.timezone.utc).timestamp()
        except ImageAdoptionError:
            raise
        except Exception as exc:
            raise ImageAdoptionError("clock-invalid") from exc

    def _durable_highwater(self) -> tuple[str, float] | None:
        """Recover only the admitted Film receipt/floor logical-time evidence."""
        state, temporal = self._film_state()
        if state is None or temporal is None:
            return None
        return temporal.highwater_at, self._epoch(temporal.highwater_at)

    def _logical_now(self) -> tuple[str, float]:
        wall_value, wall_epoch = self._now()
        durable = self._durable_highwater()
        if durable is None or wall_epoch >= durable[1]:
            return wall_value, wall_epoch
        return durable

    @staticmethod
    def _timestamp_after(value: str, seconds: int) -> str:
        try:
            parsed = _datetime.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
            return (parsed.replace(tzinfo=_datetime.timezone.utc) + _datetime.timedelta(seconds=seconds)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        except Exception as exc:
            raise ImageAdoptionError("timestamp-invalid") from exc

    @staticmethod
    def _epoch(value: str) -> float:
        try:
            parsed = _datetime.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
            return parsed.replace(tzinfo=_datetime.timezone.utc).timestamp()
        except Exception as exc:
            raise ImageAdoptionError("timestamp-invalid") from exc

    @staticmethod
    def _canonical(value: Any) -> bytes:
        try:
            raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        except Exception as exc:
            raise ImageAdoptionError("operation-canonical-json-invalid") from exc
        if len(raw) > MAX_WIRE_BYTES:
            raise ImageAdoptionError("operation-size-exceeded")
        return raw

    @classmethod
    def _digest(cls, value: Any) -> str:
        return hashlib.sha256(cls._canonical(value)).hexdigest()

    @staticmethod
    def _copy(value: Any) -> Any:
        try:
            return copy.deepcopy(value)
        except Exception as exc:
            raise ImageAdoptionError("operation-value-invalid") from exc

    @staticmethod
    def _session(session_id: Any) -> str:
        if type(session_id) is not str or not 1 <= len(session_id) <= 256 or "\x00" in session_id:
            raise ImageAdoptionError("session-invalid")
        try:
            session_id.encode("utf-8")
        except UnicodeError as exc:
            raise ImageAdoptionError("session-invalid") from exc
        return session_id

    @staticmethod
    def _keys(value: Any, required: set[str], name: str) -> dict[str, Any]:
        if type(value) is not dict or any(type(key) is not str for key in value):
            raise ImageAdoptionError(f"{name}-shape")
        try:
            actual = set(value)
        except Exception as exc:
            raise ImageAdoptionError(f"{name}-shape") from exc
        if actual != required:
            raise ImageAdoptionError(f"{name}-shape")
        return value

    @staticmethod
    def _strict_int(value: Any, name: str, low: int, high: int) -> int:
        if type(value) is not int or value < low or value > high:
            raise ImageAdoptionError(f"{name}-invalid")
        return value

    @staticmethod
    def _strict_text(value: Any, name: str, low: int, high: int) -> str:
        if type(value) is not str or not low <= len(value) <= high:
            raise ImageAdoptionError(f"{name}-invalid")
        try:
            value.encode("utf-8")
        except UnicodeError as exc:
            raise ImageAdoptionError(f"{name}-invalid") from exc
        return value

    @staticmethod
    def _strict_hash(value: Any, name: str) -> str:
        if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
            raise ImageAdoptionError(f"{name}-invalid")
        return value

    @staticmethod
    def _strict_const(value: Any, expected: str, name: str) -> str:
        if type(value) is not str or value != expected:
            raise ImageAdoptionError(f"{name}-invalid")
        return value

    @staticmethod
    def _strict_timestamp(value: Any, name: str) -> str:
        if type(value) is not str or _TIMESTAMP_RE.fullmatch(value) is None:
            raise ImageAdoptionError(f"{name}-invalid")
        try:
            _datetime.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError as exc:
            raise ImageAdoptionError(f"{name}-invalid") from exc
        return value

    @staticmethod
    def _strict_token(value: Any, name: str = "preview-token") -> str:
        if type(value) is not str or _TOKEN_RE.fullmatch(value) is None:
            raise ImageAdoptionError(f"{name}-invalid")
        return value

    def _project(self, value: Any) -> str:
        if type(value) is not str or _PROJECT_RE.fullmatch(value) is None:
            raise ImageAdoptionError("project-id-invalid")
        if value != self.project_id:
            raise ImageAdoptionError("project-scope-mismatch")
        return value

    def _validate_target(self, value: Any, *, assigned: bool) -> dict[str, Any]:
        target = self._keys(value, {"scope_kind", "project_id", "binding_id", "film_reference_id"}, "target-scope")
        self._project(target["project_id"])
        if assigned:
            self._strict_const(target["scope_kind"], "film_binding", "target-scope-kind")
            _validate_id(target["binding_id"], _BINDING_RE, "binding-id-invalid")
            _validate_id(target["film_reference_id"], _FILM_REFERENCE_RE, "film-reference-id-invalid")
        else:
            if (
                type(target["scope_kind"]) is not str
                or target["scope_kind"] != "film_binding_new_reference"
                or target["binding_id"] is not None
                or target["film_reference_id"] is not None
            ):
                raise ImageAdoptionError("unassigned-target-invalid")
        return self._copy(target)

    def _validate_intent(self, value: Any) -> dict[str, Any]:
        intent = self._keys(
            value,
            {
                "branch_version",
                "target_scope",
                "expected_revision",
                "label",
                "source_context_id",
                "source_work_id",
                "source_revision_id",
                "snapshot_sha256",
            },
            "operation-intent",
        )
        self._strict_const(intent["branch_version"], BRANCH_VERSION, "branch-version")
        self._validate_target(intent["target_scope"], assigned=True)
        self._strict_intent_revision(intent["expected_revision"])
        self._strict_text(intent["label"], "label", 1, 160)
        _validate_id(intent["source_context_id"], _CONTEXT_RE, "source-context-id-invalid")
        _validate_id(intent["source_work_id"], _WORK_RE, "source-work-id-invalid")
        _validate_id(intent["source_revision_id"], _REVISION_RE, "source-revision-id-invalid")
        self._strict_hash(intent["snapshot_sha256"], "snapshot-sha256")
        return self._copy(intent)

    def _strict_intent_revision(self, value: Any) -> int:
        return self._strict_int(value, "expected-revision", 0, MAX_SAFE_INTEGER)

    def _validate_preview_request(self, request: Any) -> dict[str, Any]:
        value = self._keys(
            request,
            {
                "schema",
                "kind",
                "branch_version",
                "expected_revision",
                "target_scope",
                "label",
                "source_context_id",
                "source_work_id",
                "source_revision_id",
            },
            "preview-request",
        )
        if (
            type(value["schema"]) is not str
            or type(value["kind"]) is not str
            or type(value["branch_version"]) is not str
            or value["schema"] != V2_SCHEMA
            or value["kind"] != PREVIEW_KIND
            or value["branch_version"] != BRANCH_VERSION
        ):
            raise ImageAdoptionError("operation-discriminator-invalid")
        self._strict_intent_revision(value["expected_revision"])
        self._validate_target(value["target_scope"], assigned=False)
        self._strict_text(value["label"], "label", 1, 160)
        _validate_id(value["source_context_id"], _CONTEXT_RE, "source-context-id-invalid")
        _validate_id(value["source_work_id"], _WORK_RE, "source-work-id-invalid")
        _validate_id(value["source_revision_id"], _REVISION_RE, "source-revision-id-invalid")
        return self._copy(value)

    def _validate_owner_scope(self, value: Any) -> str:
        if type(value) is not str or _OWNER_RE.fullmatch(value) is None:
            raise ImageAdoptionError("owner-scope-invalid")
        if value != self.owner_scope:
            raise ImageAdoptionError("owner-scope-mismatch")
        return value

    def _validate_assigned_common(self, value: Any, *, kind: str, required: set[str]) -> dict[str, Any]:
        request = self._keys(value, required, f"{kind}-request")
        if (
            type(request["schema"]) is not str
            or type(request["kind"]) is not str
            or type(request["branch_version"]) is not str
            or request["schema"] != V2_SCHEMA
            or request["branch_version"] != BRANCH_VERSION
            or request["kind"] != kind
        ):
            raise ImageAdoptionError("operation-discriminator-invalid")
        self._validate_owner_scope(request["owner_scope"])
        _validate_id(request["context_id"], _CONTEXT_RE, "context-id-invalid")
        self._strict_intent_revision(request["expected_revision"])
        self._validate_target(request["target_scope"], assigned=True)
        intent = self._validate_intent(request["operation_intent"])
        if request["context_id"] != intent["source_context_id"] or request["target_scope"] != intent["target_scope"] or request["expected_revision"] != intent["expected_revision"]:
            raise ImageAdoptionError("operation-intent-mismatch")
        return self._copy(request)

    def _validate_prepare_request(self, value: Any) -> dict[str, Any]:
        request = self._validate_assigned_common(
            value,
            kind=PREPARE_KIND,
            required={
                "schema", "kind", "branch_version", "owner_scope", "context_id", "target_kind",
                "expected_revision", "target_scope", "operation_intent", "payload",
            },
        )
        self._strict_const(request["target_kind"], ADOPT_KIND, "target-kind")
        payload = self._keys(request["payload"], {"preview_token"}, "payload")
        self._strict_token(payload["preview_token"])
        return request

    def _validate_execute_request(self, value: Any) -> dict[str, Any]:
        request = self._validate_assigned_common(
            value,
            kind=ADOPT_KIND,
            required={
                "schema", "kind", "branch_version", "owner_scope", "context_id", "operation_id", "issued_at",
                "expected_revision", "target_scope", "operation_intent", "payload",
            },
        )
        _validate_id(request["operation_id"], _OPERATION_RE, "operation-id-invalid")
        self._strict_timestamp(request["issued_at"], "issued-at")
        payload = self._keys(request["payload"], {"preview_token"}, "payload")
        self._strict_token(payload["preview_token"])
        return request

    def _validate_cancel_request(self, value: Any) -> dict[str, Any]:
        request = self._keys(
            value,
            {
                "schema", "kind", "branch_version", "owner_scope", "expected_revision", "target_scope",
                "operation_intent", "preview_token",
            },
            "cancel-request",
        )
        if (
            type(request["schema"]) is not str
            or type(request["kind"]) is not str
            or type(request["branch_version"]) is not str
            or request["schema"] != V2_SCHEMA
            or request["kind"] != CANCEL_KIND
            or request["branch_version"] != BRANCH_VERSION
        ):
            raise ImageAdoptionError("operation-discriminator-invalid")
        self._validate_owner_scope(request["owner_scope"])
        self._strict_intent_revision(request["expected_revision"])
        self._validate_target(request["target_scope"], assigned=True)
        intent = self._validate_intent(request["operation_intent"])
        if request["target_scope"] != intent["target_scope"] or request["expected_revision"] != intent["expected_revision"]:
            raise ImageAdoptionError("operation-intent-mismatch")
        self._strict_token(request["preview_token"])
        return self._copy(request)

    def _validate_status_request(self, value: Any) -> dict[str, Any]:
        request = self._keys(
            value,
            {"schema", "kind", "branch_version", "owner_scope", "operation_id", "operation_kind"},
            "status-request",
        )
        if (
            type(request["schema"]) is not str
            or type(request["kind"]) is not str
            or type(request["branch_version"]) is not str
            or request["schema"] != V2_SCHEMA
            or request["kind"] != STATUS_KIND
            or request["branch_version"] != BRANCH_VERSION
        ):
            raise ImageAdoptionError("operation-discriminator-invalid")
        self._validate_owner_scope(request["owner_scope"])
        _validate_id(request["operation_id"], _OPERATION_RE, "operation-id-invalid")
        self._strict_const(request["operation_kind"], ADOPT_KIND, "operation-kind")
        return self._copy(request)

    @staticmethod
    def _assigned_scope(project_id: str, target: Any) -> dict[str, Any]:
        return {
            "scope_kind": "film_binding",
            "project_id": project_id,
            "binding_id": target.binding_id,
            "film_reference_id": target.film_reference_id,
        }

    @staticmethod
    def _unassigned_scope(project_id: str) -> dict[str, Any]:
        return {
            "scope_kind": "film_binding_new_reference",
            "project_id": project_id,
            "binding_id": None,
            "film_reference_id": None,
        }

    def _error(
        self,
        operation_kind: str,
        reason: str,
        *,
        target_scope: dict[str, Any] | None = None,
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        conflict = {"image_operation_conflict", "image_conflict"}
        expired = {"image_operation_expired", "image_preview_expired"}
        no_effect = {"image_operation_no_effect", "image_preview_cancelled", "image_preview_consumed", "image_capacity"}
        failed = {
            "image_wrong_session", "image_source_unavailable", "image_authorization_denied",
            "image_operation_not_yet_valid", "image_key_unavailable", "image_state_unavailable",
        }
        if reason in conflict:
            status = "conflict"
        elif reason in expired:
            status = "expired"
        elif reason in no_effect:
            status = "no_effect"
        elif reason in failed:
            status = "failed"
        else:
            # Malformed values are represented as a stable conflict envelope;
            # callers never receive a raw parser/KeyError/TypeError.
            reason = "image_operation_conflict"
            status = "conflict"
        return {
            "schema": V2_SCHEMA,
            "kind": ERROR_KIND,
            "branch_version": BRANCH_VERSION,
            "owner_scope": self.owner_scope,
            "operation_kind": operation_kind if operation_kind in {PREVIEW_KIND, CANCEL_KIND, PREPARE_KIND, ADOPT_KIND} else ADOPT_KIND,
            "status": status,
            "reason_code": reason,
            "operation_id": operation_id,
            "target_scope": self._copy(target_scope or self._unassigned_scope(self.project_id)),
        }

    def _safe_error(self, operation_kind: str, reason: str, **kwargs: Any) -> dict[str, Any]:
        try:
            return self._error(operation_kind, reason, **kwargs)
        except Exception:
            return self._error(ADOPT_KIND, "image_operation_conflict")

    def _purge(self, now: float) -> None:
        # Each registry is bounded independently; no unbounded sweep is used.
        for token, record in list(self._previews.items())[: self.max_pending_previews]:
            if record.state == "open" and record.expires_epoch <= now:
                try:
                    self._film_store_abandon_binding_target(record.target)
                except FilmBindingStoreError as exc:
                    # The Store removes an already-expired target before
                    # reporting its stable expiry error; that is successful
                    # cleanup, not a failed cancellation.
                    if exc.code != "binding-target-expired":
                        raise ImageAdoptionError("image_state_unavailable") from exc
                record.state = "expired"
        for operation_id, record in list(self._operations.items())[: self.max_pending_operations]:
            if record.state == "pending" and record.expires_epoch <= now:
                record.state = "expired"

    def _evict_terminal_previews(self) -> None:
        while len(self._previews) >= self.max_pending_previews:
            victim = next((token for token, record in self._previews.items() if record.state != "open"), None)
            if victim is None:
                return
            record = self._previews[victim]
            try:
                self._film_store_abandon_binding_target(record.target)
            except FilmBindingStoreError as exc:
                # A terminal record may already have released its target (for
                # example, a consumed preview).  Any other refusal leaves the
                # record in place so cleanup failure cannot hide a leak.
                if exc.code not in {"binding-target-invalid", "binding-target-expired"}:
                    raise ImageAdoptionError("image_state_unavailable") from exc
            del self._previews[victim]

    def _evict_terminal_operations(self) -> None:
        while len(self._operations) >= self.max_pending_operations:
            victim = next((operation_id for operation_id, record in self._operations.items() if record.state != "pending"), None)
            if victim is None:
                return
            del self._operations[victim]

    def _find_session_preview(self, session_id: str) -> _PreviewRecord | None:
        for record in self._previews.values():
            if record.session_id == session_id and record.state == "open":
                return record
        return None

    @staticmethod
    def _snapshot_identity(snapshot: dict[str, Any]) -> dict[str, Any]:
        """Return the source identity while excluding mutable availability."""
        identity = PreviewAuthority._copy(snapshot)
        for material in identity["materials"]:
            material.pop("availability", None)
        return identity

    def _validate_source_state(self, state: Any, context_id: str) -> dict[str, Any]:
        if type(state) is not dict:
            raise ImageAdoptionError("image_source_unavailable")
        validator = getattr(_image_contract, "validate_image_state", None)
        if callable(validator):
            try:
                state = validator(state)
            except Exception as exc:
                raise ImageAdoptionError("image_source_unavailable") from exc
        if type(state.get("context_id")) is not str or state["context_id"] != context_id:
            raise ImageAdoptionError("image_source_unavailable")
        try:
            body = state["state"]
            works = body["works"]
            materials = body["materials"]
        except (KeyError, TypeError) as exc:
            raise ImageAdoptionError("image_source_unavailable") from exc
        enrollment = self._film_owner_enrollment_id
        fingerprint = self._film_owner_fingerprint
        if (
            type(enrollment) is not str
            or type(fingerprint) is not str
            or state.get("receipt_key_enrollment_id") != enrollment
            or state.get("receipt_key_fingerprint") != fingerprint
        ):
            raise ImageAdoptionError("image_authorization_denied")
        if type(works) is not list or type(materials) is not list:
            raise ImageAdoptionError("image_source_unavailable")
        return self._copy(state)

    def _load_snapshot(
        self,
        context_id: str,
        work_id: str,
        revision_id: str,
        *,
        require_available: bool,
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        try:
            self._revalidate_image_store()
            # These class-bound calls are intentional.  An exact-type store can
            # still have mutable instance attributes, so instance replacements
            # of the loader/verifier are not an authorization path.
            state = self._validate_source_state(ImageStateStore.load(self.image_store, context_id), context_id)
            try:
                claim = ImageStateStore.load_current_owner(self.image_store, context_id, work_id, "saved_revision", revision_id)
                claim_fields = {
                    "schema",
                    "context_id",
                    "work_id",
                    "basis_kind",
                    "revision_id",
                    "draft_generation",
                    "basis_content_sha256",
                    "expected_revision",
                }
                if (
                    type(claim) is not dict
                    or set(claim) != claim_fields | {"owner_token"}
                    or claim.get("context_id") != context_id
                    or claim.get("work_id") != work_id
                    or claim.get("basis_kind") != "saved_revision"
                    or claim.get("revision_id") != revision_id
                ):
                    raise ImageAdoptionError("image_authorization_denied")
                owner_token = claim["owner_token"]
                verified = ImageStateStore.verify_current_owner(self.image_store, claim, owner_token)
                if verified is not True and (type(verified) is not dict or verified != claim):
                    raise ImageAdoptionError("image_authorization_denied")
            except ImageAdoptionError:
                raise
            except Exception as exc:
                raise ImageAdoptionError("image_authorization_denied") from exc
            state_part = state["state"]
            works = state_part["works"]
            materials = state_part["materials"]
            work_matches = [item for item in works if type(item) is dict and item.get("work_id") == work_id]
            if len(work_matches) != 1:
                raise ImageAdoptionError("image_source_unavailable")
            revisions = work_matches[0]["saved_revisions"]
            revision_matches = [item for item in revisions if type(item) is dict and item.get("revision_id") == revision_id]
            if len(revision_matches) != 1:
                raise ImageAdoptionError("image_source_unavailable")
            revision = self._copy(revision_matches[0])
            if revision.get("work_id") != work_id:
                raise ImageAdoptionError("image_source_unavailable")
            if state["revision"] != claim["expected_revision"]:
                raise ImageAdoptionError("image_source_unavailable")
            try:
                basis_hash = _image_contract.sha256(_image_contract.canonical_bytes(revision["content"]))
            except Exception as exc:
                raise ImageAdoptionError("image_source_unavailable") from exc
            if claim["basis_content_sha256"] != basis_hash:
                raise ImageAdoptionError("image_source_unavailable")
            material_ids = revision["material_ids"]
            if (
                type(material_ids) is not list
                or len(material_ids) > 128
                or any(type(item) is not str or _MATERIAL_RE.fullmatch(item) is None for item in material_ids)
                or len(set(material_ids)) != len(material_ids)
            ):
                raise ImageAdoptionError("image_source_unavailable")
            closure: set[str] = set()
            content = revision["content"]
            for reference in content["references"]:
                closure.add(reference["material_id"])
            for alternative in content["alternatives"]:
                closure.add(alternative["material_id"])
            selected_artifact = revision["selected_artifact"]
            if selected_artifact is not None:
                closure.add(selected_artifact)
            if closure != set(material_ids):
                raise ImageAdoptionError("image_source_unavailable")
            by_id: dict[str, dict[str, Any]] = {}
            for material in materials:
                if type(material) is not dict or type(material.get("material_id")) is not str:
                    raise ImageAdoptionError("image_source_unavailable")
                material_id = material["material_id"]
                if material_id in by_id:
                    raise ImageAdoptionError("image_source_unavailable")
                by_id[material_id] = self._copy(material)
            selected: list[dict[str, Any]] = []
            for material_id in material_ids:
                material = by_id.get(material_id)
                if (
                    material is None
                    or type(material.get("media_type")) is not str
                    or material["media_type"] != "image"
                    or type(material.get("sha256")) is not str
                    or _SHA256_RE.fullmatch(material["sha256"]) is None
                    or type(material.get("availability")) is not str
                    or material["availability"] not in {"available", "missing", "denied"}
                ):
                    raise ImageAdoptionError("image_source_unavailable")
                selected.append(material)
            snapshot = {"revision": revision, "materials": selected}
            digest = self._digest(snapshot)
            if require_available and any(item["availability"] != "available" for item in selected):
                raise ImageAdoptionError("image_source_unavailable")
            return state, snapshot, digest
        except ImageAdoptionError:
            raise
        except Exception as exc:
            raise ImageAdoptionError("image_source_unavailable") from exc

    def _source_matches(self, record: _PreviewRecord) -> None:
        _state, snapshot, digest = self._load_snapshot(
            record.request["source_context_id"],
            record.request["source_work_id"],
            record.request["source_revision_id"],
            require_available=True,
        )
        if (
            record.snapshot_sha256 != self._digest(record.snapshot)
            or digest != self._digest(record.snapshot)
            or self._snapshot_identity(snapshot) != self._snapshot_identity(record.snapshot)
        ):
            raise ImageAdoptionError("image_source_unavailable")
        if any(item["availability"] != "available" for item in snapshot["materials"]):
            raise ImageAdoptionError("image_source_unavailable")

    def _token(self) -> str:
        try:
            raw = self._token_factory(TOKEN_BYTES)
        except Exception as exc:
            raise ImageAdoptionError("preview-token-unavailable") from exc
        if type(raw) is not bytes or len(raw) != TOKEN_BYTES:
            raise ImageAdoptionError("preview-token-unavailable")
        token = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
        if _TOKEN_RE.fullmatch(token) is None:
            raise ImageAdoptionError("preview-token-unavailable")
        return token

    def _operation_signature(self, projection: dict[str, Any]) -> str:
        raw = self._canonical(projection)
        owner = self._film_owner_authority
        enrollment_id, fingerprint = self._validate_film_authority(
            owner,
            stored_enrollment_id=self._film_owner_enrollment_id,
            stored_fingerprint=self._film_owner_fingerprint,
        )
        if enrollment_id != self._film_owner_enrollment_id or fingerprint != self._film_owner_fingerprint:
            raise ImageAdoptionError("film-owner-enrollment-mismatch")
        try:
            # The accepted HMAC authority is deliberately called through its
            # class, never through mutable instance attributes.
            signed = HmacImageOwnerAuthority.sign_owner_claim(owner, raw)
        except ImageAdoptionError:
            raise
        except Exception as exc:
            raise ImageAdoptionError("image_state_unavailable") from exc
        if type(signed) is not str or re.fullmatch(r"imgown_[0-9a-f]{64}", signed) is None:
            raise ImageAdoptionError("image_state_unavailable")
        try:
            verified = HmacImageOwnerAuthority.verify_owner_claim(owner, raw, signed)
        except ImageAdoptionError:
            raise
        except Exception as exc:
            raise ImageAdoptionError("image_state_unavailable") from exc
        if verified is not True:
            raise ImageAdoptionError("image_state_unavailable")
        digest = signed[len("imgown_"):]
        seconds = int(self._epoch(projection["issued_at"]))
        return f"imgop_{seconds:016x}{digest[:16]}"

    def _projection(self, request: dict[str, Any], payload_sha256: str) -> dict[str, Any]:
        return {
            "schema": request["schema"],
            "branch_version": request["branch_version"],
            "operation_kind": ADOPT_KIND,
            "owner_scope": request["owner_scope"],
            "expected_revision": request["expected_revision"],
            "target_scope": self._copy(request["target_scope"]),
            "operation_intent": self._copy(request["operation_intent"]),
            "payload_sha256": payload_sha256,
            "issued_at": request["issued_at"],
        }

    def preview(self, request: dict[str, Any], session_id: str) -> dict[str, Any]:
        try:
            session_id = self._session(session_id)
        except ImageAdoptionError as exc:
            return self._safe_error(PREVIEW_KIND, exc.code, target_scope=self._unassigned_scope(self.project_id))
        try:
            request = self._validate_preview_request(request)
        except ImageAdoptionError as exc:
            return self._safe_error(PREVIEW_KIND, exc.code, target_scope=self._unassigned_scope(self.project_id))
        with self._lock:
            try:
                issued_at, now = self._logical_now()
                self._purge(now)
                existing = self._find_session_preview(session_id)
                if existing is not None:
                    if existing.request == request and existing.expires_epoch > now:
                        return self._copy(existing.response)
                    # One open preview per session is deliberate: changing any
                    # frozen field cannot allocate a second owner target.
                    return self._safe_error(PREVIEW_KIND, "image_operation_conflict", target_scope=self._unassigned_scope(self.project_id))
                open_count = sum(1 for record in self._previews.values() if record.state == "open")
                if open_count >= self.max_pending_previews:
                    return self._safe_error(PREVIEW_KIND, "image_capacity", target_scope=self._unassigned_scope(self.project_id))
                _state, snapshot, snapshot_sha256 = self._load_snapshot(
                    request["source_context_id"], request["source_work_id"], request["source_revision_id"], require_available=True
                )
                if request["expected_revision"] != self._film_revision():
                    return self._safe_error(PREVIEW_KIND, "image_conflict", target_scope=self._unassigned_scope(self.project_id))
                # Validate the capability factory before reserving a Film
                # target.  A failed/invalid token factory must be completely
                # allocation-free, including the Store's pending-target
                # registry.
                token = self._token()
                # Release terminal records before reserving a new target so a
                # failed cleanup cannot strand this preview's fresh target.
                self._evict_terminal_previews()
                target = self._film_store_mint_binding_target()
                if not all(type(getattr(target, name, None)) is str for name in ("binding_id", "film_reference_id")):
                    raise ImageAdoptionError("binding-target-invalid")
                _validate_id(target.binding_id, _BINDING_RE, "binding-id-invalid")
                _validate_id(target.film_reference_id, _FILM_REFERENCE_RE, "film-reference-id-invalid")
                assigned = self._assigned_scope(self.project_id, target)
                intent = {
                    "branch_version": BRANCH_VERSION,
                    "target_scope": self._copy(assigned),
                    "expected_revision": request["expected_revision"],
                    "label": request["label"],
                    "source_context_id": request["source_context_id"],
                    "source_work_id": request["source_work_id"],
                    "source_revision_id": request["source_revision_id"],
                    "snapshot_sha256": snapshot_sha256,
                }
                expires_at = self._timestamp_after(issued_at, PREVIEW_TTL_SECONDS)
                response = {
                    "schema": V2_SCHEMA,
                    "kind": PREVIEW_READY_KIND,
                    "branch_version": BRANCH_VERSION,
                    "owner_scope": self.owner_scope,
                    "expected_revision": request["expected_revision"],
                    "target_scope": self._copy(assigned),
                    "operation_intent": self._copy(intent),
                    "preview_token": token,
                    "expires_at": expires_at,
                    "label": request["label"],
                    "source_context_id": request["source_context_id"],
                    "source_work_id": request["source_work_id"],
                    "source_revision_id": request["source_revision_id"],
                    "snapshot_sha256": snapshot_sha256,
                    "snapshot": self._copy(snapshot),
                }
                token = response["preview_token"]
                self._canonical(response)
                self._previews[token] = _PreviewRecord(
                    session_id=session_id,
                    request=self._copy(request),
                    response=self._copy(response),
                    snapshot=self._copy(snapshot),
                    snapshot_sha256=snapshot_sha256,
                    target=target,
                    issued_at=issued_at,
                    expires_epoch=self._epoch(expires_at),
                )
                return self._copy(response)
            except ImageAdoptionError as exc:
                return self._safe_error(PREVIEW_KIND, exc.code, target_scope=self._unassigned_scope(self.project_id))
            except Exception as exc:
                reason = "image_capacity" if isinstance(exc, FilmBindingStoreError) and exc.code == "pending-target-capacity-exceeded" else "image_state_unavailable"
                return self._safe_error(PREVIEW_KIND, reason, target_scope=self._unassigned_scope(self.project_id))

    film_image_preview_new_reference = preview
    preview_new_reference = preview

    def prepare(self, request: dict[str, Any], session_id: str) -> dict[str, Any]:
        try:
            session_id = self._session(session_id)
        except ImageAdoptionError as exc:
            return self._safe_error(PREPARE_KIND, exc.code, target_scope=self._unassigned_scope(self.project_id))
        target_scope: dict[str, Any] | None = None
        try:
            request = self._validate_prepare_request(request)
            target_scope = request["target_scope"]
        except ImageAdoptionError as exc:
            return self._safe_error(PREPARE_KIND, exc.code, target_scope=target_scope or self._unassigned_scope(self.project_id))
        with self._lock:
            try:
                _issued_at, now = self._logical_now()
                self._purge(now)
                token = request["payload"]["preview_token"]
                record = self._previews.get(token)
                if record is None:
                    return self._safe_error(PREPARE_KIND, "image_preview_expired", target_scope=target_scope)
                if record.session_id != session_id:
                    return self._safe_error(PREPARE_KIND, "image_wrong_session", target_scope=target_scope)
                if record.state == "cancelled":
                    return self._safe_error(PREPARE_KIND, "image_preview_cancelled", target_scope=target_scope)
                if record.state == "consumed":
                    return self._safe_error(PREPARE_KIND, "image_preview_consumed", target_scope=target_scope)
                if record.state == "expired" or record.expires_epoch <= now:
                    record.state = "expired"
                    return self._safe_error(PREPARE_KIND, "image_preview_expired", target_scope=target_scope)
                if (
                    request["owner_scope"] != self.owner_scope
                    or request["context_id"] != record.response["source_context_id"]
                    or request["expected_revision"] != record.response["expected_revision"]
                    or request["target_scope"] != record.response["target_scope"]
                    or request["operation_intent"] != record.response["operation_intent"]
                ):
                    return self._safe_error(PREPARE_KIND, "image_operation_conflict", target_scope=target_scope)
                if request["expected_revision"] != self._film_revision():
                    return self._safe_error(PREPARE_KIND, "image_conflict", target_scope=target_scope)
                self._source_matches(record)
                payload_sha256 = self._digest(request["payload"])
                issued_at = record.issued_at
                projection = self._projection({**request, "kind": ADOPT_KIND, "issued_at": issued_at}, payload_sha256)
                operation_id = self._operation_signature(projection)
                previous = self._operations.get(operation_id)
                if previous is not None:
                    if previous.projection != projection or previous.token != token or previous.session_id != session_id:
                        return self._safe_error(PREPARE_KIND, "image_operation_conflict", target_scope=target_scope)
                    return self._copy(previous.prepared)
                self._evict_terminal_operations()
                pending_count = sum(1 for item in self._operations.values() if item.state == "pending")
                if pending_count >= self.max_pending_operations:
                    return self._safe_error(PREPARE_KIND, "image_capacity", target_scope=target_scope)
                prepared = {
                    "schema": V2_SCHEMA,
                    "kind": PREPARED_KIND,
                    "branch_version": BRANCH_VERSION,
                    "owner_scope": self.owner_scope,
                    "context_id": request["context_id"],
                    "target_kind": ADOPT_KIND,
                    "operation_id": operation_id,
                    "issued_at": issued_at,
                    "expires_at": self._timestamp_after(issued_at, OPERATION_TTL_SECONDS),
                    "expected_revision": request["expected_revision"],
                    "target_scope": self._copy(request["target_scope"]),
                    "operation_intent": self._copy(request["operation_intent"]),
                    "payload_sha256": payload_sha256,
                }
                self._operations[operation_id] = _OperationRecord(
                    session_id=session_id,
                    token=token,
                    projection=self._copy(projection),
                    prepared=self._copy(prepared),
                    expires_epoch=self._epoch(prepared["expires_at"]),
                )
                record.operation_id = operation_id
                return self._copy(prepared)
            except ImageAdoptionError as exc:
                return self._safe_error(PREPARE_KIND, exc.code, target_scope=target_scope)
            except Exception:
                return self._safe_error(PREPARE_KIND, "image_state_unavailable", target_scope=target_scope)

    image_prepare_operation = prepare
    prepare_operation = prepare

    @staticmethod
    def prepare_request(preview_response: dict[str, Any]) -> dict[str, Any]:
        try:
            required = {
                "schema", "kind", "branch_version", "owner_scope", "expected_revision", "target_scope",
                "operation_intent", "preview_token", "source_context_id",
            }
            if type(preview_response) is not dict or not required.issubset(preview_response):
                raise ImageAdoptionError("preview-response-shape")
            return {
                "schema": V2_SCHEMA,
                "kind": PREPARE_KIND,
                "branch_version": BRANCH_VERSION,
                "owner_scope": preview_response["owner_scope"],
                "context_id": preview_response["source_context_id"],
                "target_kind": ADOPT_KIND,
                "expected_revision": preview_response["expected_revision"],
                "target_scope": copy.deepcopy(preview_response["target_scope"]),
                "operation_intent": copy.deepcopy(preview_response["operation_intent"]),
                "payload": {"preview_token": preview_response["preview_token"]},
            }
        except ImageAdoptionError:
            raise
        except Exception as exc:
            raise ImageAdoptionError("preview-response-shape") from exc

    @staticmethod
    def execute_request(prepare_request: dict[str, Any], prepared_response: dict[str, Any]) -> dict[str, Any]:
        try:
            return {
                "schema": V2_SCHEMA,
                "kind": ADOPT_KIND,
                "branch_version": BRANCH_VERSION,
                "owner_scope": prepare_request["owner_scope"],
                "context_id": prepare_request["context_id"],
                "operation_id": prepared_response["operation_id"],
                "issued_at": prepared_response["issued_at"],
                "expected_revision": prepare_request["expected_revision"],
                "target_scope": copy.deepcopy(prepare_request["target_scope"]),
                "operation_intent": copy.deepcopy(prepare_request["operation_intent"]),
                "payload": copy.deepcopy(prepare_request["payload"]),
            }
        except Exception as exc:
            raise ImageAdoptionError("prepared-response-shape") from exc

    def cancel(self, request: dict[str, Any], session_id: str) -> dict[str, Any]:
        try:
            session_id = self._session(session_id)
        except ImageAdoptionError as exc:
            return self._safe_error(CANCEL_KIND, exc.code, target_scope=self._unassigned_scope(self.project_id))
        target_scope: dict[str, Any] | None = None
        try:
            request = self._validate_cancel_request(request)
            target_scope = request["target_scope"]
        except ImageAdoptionError as exc:
            return self._safe_error(CANCEL_KIND, exc.code, target_scope=target_scope or self._unassigned_scope(self.project_id))
        with self._lock:
            try:
                _issued_at, now = self._logical_now()
                self._purge(now)
                record = self._previews.get(request["preview_token"])
                if record is None:
                    return self._safe_error(CANCEL_KIND, "image_preview_expired", target_scope=target_scope)
                if record.session_id != session_id:
                    return self._safe_error(CANCEL_KIND, "image_wrong_session", target_scope=target_scope)
                if record.state == "consumed":
                    return self._safe_error(CANCEL_KIND, "image_preview_consumed", target_scope=target_scope)
                if record.state == "expired" or record.expires_epoch <= now:
                    record.state = "expired"
                    return self._safe_error(CANCEL_KIND, "image_preview_expired", target_scope=target_scope)
                if record.state != "open" or request["target_scope"] != record.response["target_scope"] or request["expected_revision"] != record.response["expected_revision"] or request["operation_intent"] != record.response["operation_intent"]:
                    return self._safe_error(CANCEL_KIND, "image_operation_conflict", target_scope=target_scope)
                self._film_store_abandon_binding_target(record.target)
                record.state = "cancelled"
                if record.operation_id is not None:
                    operation = self._operations.get(record.operation_id)
                    if operation is not None and operation.state == "pending":
                        operation.state = "cancelled"
                return {
                    "schema": V2_SCHEMA,
                    "kind": CANCELLED_KIND,
                    "branch_version": BRANCH_VERSION,
                    "owner_scope": self.owner_scope,
                    "expected_revision": request["expected_revision"],
                    "target_scope": self._copy(request["target_scope"]),
                    "operation_intent": self._copy(request["operation_intent"]),
                    "preview_token": request["preview_token"],
                    "status": "cancelled",
                    "writes": False,
                }
            except ImageAdoptionError as exc:
                return self._safe_error(CANCEL_KIND, exc.code, target_scope=target_scope)
            except Exception:
                return self._safe_error(CANCEL_KIND, "image_state_unavailable", target_scope=target_scope)

    film_image_preview_cancel = cancel
    cancel_preview = cancel

    @staticmethod
    def cancel_request(preview_response: dict[str, Any]) -> dict[str, Any]:
        try:
            return {
                "schema": V2_SCHEMA,
                "kind": CANCEL_KIND,
                "branch_version": BRANCH_VERSION,
                "owner_scope": preview_response["owner_scope"],
                "expected_revision": preview_response["expected_revision"],
                "target_scope": copy.deepcopy(preview_response["target_scope"]),
                "operation_intent": copy.deepcopy(preview_response["operation_intent"]),
                "preview_token": preview_response["preview_token"],
            }
        except Exception as exc:
            raise ImageAdoptionError("preview-response-shape") from exc

    def _film_revision(self) -> int:
        try:
            state, _temporal = self._film_state()
            revision = self._film_store_current_revision()
            if state is None:
                if revision != 0:
                    raise ImageAdoptionError("image_state_unavailable")
            elif revision != state.get("revision"):
                raise ImageAdoptionError("image_state_unavailable")
        except ImageAdoptionError:
            raise
        except Exception as exc:
            raise ImageAdoptionError("image_state_unavailable") from exc
        return self._strict_int(revision, "film-revision", 0, MAX_SAFE_INTEGER)



def _validate_id(value: Any, pattern: re.Pattern[str], code: str) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise ImageAdoptionError(code)
    return value


__all__ = [
    "ADOPT_KIND",
    "BRANCH_VERSION",
    "CANCELLED_KIND",
    "CANCEL_KIND",
    "ERROR_KIND",
    "FilmTemporalEvidence",
    "ImageAdoptionError",
    "MAX_PENDING_OPERATIONS",
    "MAX_PENDING_PREVIEWS",
    "OPERATION_TTL_SECONDS",
    "PREPARED_KIND",
    "PREPARE_KIND",
    "PREVIEW_KIND",
    "PREVIEW_READY_KIND",
    "PREVIEW_TTL_SECONDS",
    "PreviewAuthority",
    "PreviewAuthorityError",
    "RECEIPT_KIND",
    "STATUS_KIND",
    "STATUS_RESULT_KIND",
    "V2_SCHEMA",
    "validate_film_state_temporal",
]
