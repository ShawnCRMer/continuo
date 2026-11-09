"""Project-owned append-only persistence for Image-to-Film bindings.

This module owns the ``<project>/state/image-bindings`` revision history and
nothing else.  It does not inspect Image contexts, issue preview tokens, build
Adoptions, or perform deliberate adoption.  A later adoption service may use
``mint_binding_target`` and the append primitives after it has independently
constructed and authorized a complete contract-valid candidate.
"""
from __future__ import annotations

import copy
import datetime
import hashlib
import heapq
import importlib.util
import json
import os
import re
import secrets
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator
# These owner modules can also be loaded as isolated top-level packages.
_migration_spec = importlib.util.spec_from_file_location(
    "film_suite_owner_migration", Path(__file__).resolve().parents[2] / "generation" / "legacy_plugin_migration.py"
)
_migration_module = importlib.util.module_from_spec(_migration_spec)
_migration_spec.loader.exec_module(_migration_module)
state_root = _migration_module.state_root
TOMBSTONE_PATTERN = _migration_module.TOMBSTONE_PATTERN

from .film_binding_contract import (
    BINDINGS_V1_SCHEMA,
    BINDINGS_V2_SCHEMA,
    MAX_BINDING_BYTES,
    MAX_SAFE_INTEGER,
    FilmBindingContractError,
    project_bindings_v1_to_v2,
    validate_image_bindings,
)


MAX_STATE_FILES = 256
# POSIX PathBoundary cleanup may retain private handle-bound tombstone names.
# Keep their admission separate from the state-record limit and bounded by one
# maximum publication history, rather than allowing residue to grow freely.
MAX_INTERNAL_RESIDUES = MAX_STATE_FILES
MAX_STATE_BYTES = MAX_BINDING_BYTES
MAX_MANIFEST_BYTES = 256 * 1024
MAX_PENDING_TARGETS = 128
TARGET_TTL_SECONDS = 900
_STATE_FILE_RE = re.compile(r"^r([0-9]{20})-(append|upgrade)-(imgop_[0-9a-f]{32})\.json$")
_INTERNAL_TOMBSTONE_RE = re.compile(TOMBSTONE_PATTERN)
_OPERATION_KINDS = frozenset({"append", "upgrade"})
_OPERATION_RE = re.compile(r"^imgop_[0-9a-f]{32}$")
_BINDING_RE = re.compile(r"^imgbind_[0-9a-f]{32}$")
_FILM_REFERENCE_RE = re.compile(r"^filmref_[0-9a-f]{32}$")
_PROJECT_RE = re.compile(r"^prj_[0-9a-f]{32}$")
_ENROLLMENT_RE = re.compile(r"^imgkey_[0-9a-f]{32}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_TIMESTAMP_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")


class FilmBindingStoreError(ValueError):
    """Stable fail-closed error for the FilmBindingStore boundary."""

    def __init__(self, code: str) -> None:
        self.code = code
        self.reason_code = code
        super().__init__(code)


BindingStoreError = FilmBindingStoreError


@dataclass(frozen=True, slots=True)
class BindingTarget:
    """Opaque owner-issued IDs for one future binding publication.

    The nonce is intentionally not part of the persisted contract.  The store
    accepts a target only when this exact object was minted by that store
    instance, preventing a caller from substituting a hand-built ID pair.
    """

    binding_id: str
    film_reference_id: str
    _nonce: bytes = field(repr=False, compare=True)


@dataclass(frozen=True, slots=True)
class _DescriptorIdentity:
    """Identity captured from the creator descriptor before it is closed."""

    canonical_path: str
    volume_serial: int
    file_id: int
    is_directory: bool
    size: int
    missing_leaf: bool = False

    def same_object(self, other: Any) -> bool:
        return (
            bool(getattr(other, "missing_leaf", False)) is False
            and self.volume_serial == getattr(other, "volume_serial", None)
            and self.file_id == getattr(other, "file_id", None)
            and self.is_directory == bool(getattr(other, "is_directory", False))
        )


class FilmBindingStore:
    """Own one project's bounded immutable binding revision history."""

    def __init__(
        self,
        project_dir: Path | str,
        *,
        boundary: Any,
        authority: Any,
        clock: Callable[[], str] | None = None,
        expected_project_id: str | None = None,
    ) -> None:
        if boundary is None:
            raise FilmBindingStoreError("path-boundary-required")
        if authority is None:
            raise FilmBindingStoreError("owner-authority-required")
        try:
            normalized_project_dir = Path(project_dir)
            project_text = str(normalized_project_dir)
        except Exception as exc:
            raise FilmBindingStoreError("project-path-invalid") from exc
        if "\x00" in project_text:
            raise FilmBindingStoreError("project-path-invalid")
        self.project_dir = normalized_project_dir
        self.boundary = boundary
        self.authority = authority
        self.clock = clock or self._utc_second
        self.project_id = self._read_trusted_project_id(expected_project_id)
        self._project_identity = self._pin_directory(self.project_dir, "project-identity-unavailable")
        self._state_identity = None
        state_dir = self.project_dir / "state"
        if os.path.lexists(state_dir):
            if state_dir.is_symlink() or not state_dir.is_dir():
                raise FilmBindingStoreError("state-directory-invalid")
            self._state_identity = self._pin_directory(state_dir, "state-parent-unavailable")
        enrollment_id = getattr(authority, "receipt_key_enrollment_id", None)
        fingerprint = getattr(authority, "receipt_key_fingerprint", None)
        if type(enrollment_id) is not str or _ENROLLMENT_RE.fullmatch(enrollment_id) is None:
            raise FilmBindingStoreError("owner-enrollment-id-invalid")
        if type(fingerprint) is not str or _HEX64_RE.fullmatch(fingerprint) is None:
            raise FilmBindingStoreError("owner-fingerprint-invalid")
        self.receipt_key_enrollment_id = enrollment_id
        self.receipt_key_fingerprint = fingerprint
        self._issued_targets: dict[BindingTarget, float] = {}
        self._target_expiry_heap: list[tuple[float, int, BindingTarget]] = []
        self._target_sequence = 0

    def __copy__(self) -> FilmBindingStore:
        raise FilmBindingStoreError("store-not-copyable")

    def __deepcopy__(self, memo: dict[int, Any]) -> FilmBindingStore:
        raise FilmBindingStoreError("store-not-copyable")

    def __reduce__(self) -> Any:
        raise FilmBindingStoreError("store-not-copyable")

    def __reduce_ex__(self, protocol: int) -> Any:
        raise FilmBindingStoreError("store-not-copyable")

    def __getstate__(self) -> Any:
        raise FilmBindingStoreError("store-not-copyable")

    def _clock_timestamp(self) -> str:
        try:
            value = self.clock()
            if type(value) is not str or _TIMESTAMP_RE.fullmatch(value) is None:
                raise ValueError("invalid clock timestamp")
            datetime.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
            return value
        except FilmBindingStoreError:
            raise
        except Exception as exc:
            raise FilmBindingStoreError("clock-invalid") from exc

    def _clock_epoch(self) -> float:
        value = self._clock_timestamp()
        timestamp = datetime.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        return timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()

    def _find_pending_target(self, target: BindingTarget) -> tuple[BindingTarget, float] | None:
        for issued, expires_at in self._issued_targets.items():
            if issued is target:
                return issued, expires_at
        return None

    def _remove_pending_target(self, target: BindingTarget) -> bool:
        for issued in tuple(self._issued_targets):
            if issued is target:
                del self._issued_targets[issued]
                self._target_expiry_heap = [
                    (expires_at, index, pending)
                    for index, (pending, expires_at) in enumerate(self._issued_targets.items())
                ]
                heapq.heapify(self._target_expiry_heap)
                self._target_sequence = len(self._issued_targets)
                return True
        return False

    def _consume_replayed_target(self, state: dict[str, Any]) -> None:
        bindings = state["state"]["bindings"]
        for issued in tuple(self._issued_targets):
            if any(
                binding["binding_id"] == issued.binding_id
                and binding["film_reference_id"] == issued.film_reference_id
                for binding in bindings
            ):
                self._remove_pending_target(issued)
                return

    def _purge_expired_targets(self, now: float) -> None:
        purged = 0
        while self._target_expiry_heap and self._target_expiry_heap[0][0] <= now and purged < MAX_PENDING_TARGETS:
            expires_at, _sequence, target = heapq.heappop(self._target_expiry_heap)
            for issued, current_expiry in tuple(self._issued_targets.items()):
                if issued is target:
                    if current_expiry == expires_at:
                        del self._issued_targets[issued]
                    break
            purged += 1

    def _require_live_target(self, target: BindingTarget) -> None:
        found = self._find_pending_target(target)
        if found is None:
            raise FilmBindingStoreError("binding-target-invalid")
        issued, expires_at = found
        now = self._clock_epoch()
        if expires_at <= now:
            self._remove_pending_target(issued)
            self._purge_expired_targets(now)
            raise FilmBindingStoreError("binding-target-expired")
        self._purge_expired_targets(now)

    @staticmethod
    def _utc_second() -> str:
        import datetime

        return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _close(handle: Any) -> None:
        close = getattr(handle, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _strict_project_id(value: Any, code: str = "project-id-invalid") -> str:
        if type(value) is not str or _PROJECT_RE.fullmatch(value) is None:
            raise FilmBindingStoreError(code)
        return value

    @staticmethod
    def _strict_operation_id(value: Any) -> str:
        if type(value) is not str or _OPERATION_RE.fullmatch(value) is None:
            raise FilmBindingStoreError("operation-id-invalid")
        return value

    @staticmethod
    def _strict_revision(value: Any, *, positive: bool) -> int:
        minimum = 1 if positive else 0
        if type(value) is not int or value < minimum or value > MAX_SAFE_INTEGER:
            raise FilmBindingStoreError("revision-invalid")
        return value

    def _read_trusted_project_id(self, expected_project_id: str | None) -> str:
        expected = None if expected_project_id is None else self._strict_project_id(expected_project_id, "expected-project-id-invalid")
        reader = getattr(self.boundary, "read_manifest", None)
        if not callable(reader):
            raise FilmBindingStoreError("manifest-reader-missing")
        try:
            manifest = reader(
                self.project_dir,
                max_bytes=MAX_MANIFEST_BYTES,
                object_pairs_hook=self._reject_duplicate_pairs,
            )
        except FilmBindingStoreError:
            raise
        except Exception as exc:
            raise FilmBindingStoreError("project-manifest-unavailable") from exc
        if type(manifest) is not dict:
            raise FilmBindingStoreError("project-manifest-invalid")
        project_id = self._strict_project_id(manifest.get("project_id"), "project-manifest-invalid")
        if expected is not None and project_id != expected:
            raise FilmBindingStoreError("project-id-mismatch")
        return project_id

    def _path_boundary_module(self) -> Any:
        module = getattr(self, "_loaded_path_boundary", None)
        if module is not None:
            return module
        path = Path(__file__).resolve().parents[1] / "path_boundary.py"
        spec = importlib.util.spec_from_file_location("_film_binding_store_path_boundary", path)
        if spec is None or spec.loader is None:
            raise FilmBindingStoreError("path-boundary-unavailable")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(spec.name, None)
            raise FilmBindingStoreError("path-boundary-unavailable") from exc
        self._loaded_path_boundary = module
        return module

    def _pin_directory(self, path: Path, code: str) -> Any:
        pin = getattr(self.boundary, "pin_directory", None)
        try:
            if callable(pin):
                identity = pin(path)
            else:
                handle, identity = self._open(path, directory=True)
                self._close(handle)
        except FilmBindingStoreError:
            raise
        except Exception as exc:
            raise FilmBindingStoreError(code) from exc
        if identity is None or bool(getattr(identity, "is_directory", True)) is not True:
            raise FilmBindingStoreError(code)
        return identity

    def _open(self, path: Path, *, directory: bool) -> tuple[Any, Any]:
        opener = getattr(self.boundary, "open_existing", None)
        try:
            if callable(opener):
                handle, identity = opener(path, directory=directory)
            else:
                handle, identity = self._path_boundary_module()._open_existing(path, directory=directory)
        except FilmBindingStoreError:
            raise
        except Exception as exc:
            raise FilmBindingStoreError("state-object-unavailable") from exc
        if handle is None or identity is None:
            raise FilmBindingStoreError("state-object-unavailable")
        if bool(getattr(identity, "is_directory", directory)) is not directory:
            self._close(handle)
            raise FilmBindingStoreError("state-object-type-invalid")
        if any(bool(getattr(identity, name, False)) for name in ("is_reparse", "is_reparse_point", "reparse")):
            self._close(handle)
            raise FilmBindingStoreError("state-reparse-object")
        return handle, identity

    def _capture_stage_identity(self, fd: int, path: Path) -> Any:
        capture = getattr(self.boundary, "identity_from_fd", None)
        if callable(capture):
            return capture(fd, path)
        if not callable(getattr(self.boundary, "open_existing", None)):
            module = self._path_boundary_module()
            identity_from_handle = getattr(module, "_identity_from_handle", None)
            native_handle_type = getattr(module, "_NativeHandle", None)
            if callable(identity_from_handle) and callable(native_handle_type):
                raw = fd
                if os.name == "nt":
                    import msvcrt

                    raw = msvcrt.get_osfhandle(fd)
                return identity_from_handle(native_handle_type(raw, "file"), path)
        stat_result = os.fstat(fd)
        return _DescriptorIdentity(
            str(path),
            int(stat_result.st_dev),
            int(stat_result.st_ino),
            False,
            int(stat_result.st_size),
        )

    @staticmethod
    def _same_identity(left: Any, right: Any) -> bool:
        if left is None or right is None:
            return False
        for candidate, other in ((left, right), (right, left)):
            same = getattr(candidate, "same_object", None)
            if callable(same):
                try:
                    if not bool(same(other)):
                        return False
                except Exception:
                    return False
                break
        if bool(getattr(left, "is_directory", False)) != bool(getattr(right, "is_directory", False)):
            return False
        left_id = getattr(left, "file_id", None)
        right_id = getattr(right, "file_id", None)
        if left_id is None or right_id is None or left_id != right_id:
            return False
        left_volume = getattr(left, "volume_serial", getattr(left, "volume_id", None))
        right_volume = getattr(right, "volume_serial", getattr(right, "volume_id", None))
        if left_volume is not None or right_volume is not None:
            if left_volume != right_volume:
                return False
        if not bool(getattr(left, "is_directory", False)) and getattr(left, "size", None) != getattr(right, "size", None):
            return False
        return True

    def _revalidate_project(self, expected: Any) -> None:
        revalidate = getattr(self.boundary, "revalidate_directory", None)
        try:
            if callable(revalidate):
                revalidate(self.project_dir, expected)
            else:
                _handle, identity = self._open(self.project_dir, directory=True)
                self._close(_handle)
                if not self._same_identity(expected, identity):
                    raise OSError("project identity changed")
            current_id = self._read_trusted_project_id(self.project_id)
            if current_id != self.project_id:
                raise FilmBindingStoreError("project-id-changed")
        except FilmBindingStoreError:
            raise
        except Exception as exc:
            raise FilmBindingStoreError("project-identity-changed") from exc

    def _ensure_directory(self, path: Path) -> None:
        ensure = getattr(self.boundary, "ensure_private_node", None)
        try:
            if callable(ensure):
                ensure(path, directory=True)
            else:
                self._path_boundary_module().ensure_private_node(path, directory=True)
        except FilmBindingStoreError:
            raise
        except Exception as exc:
            raise FilmBindingStoreError("state-directory-unavailable") from exc

    def _binding_root(self, *, create: bool) -> tuple[Path, Any] | None:
        state_dir = self.project_dir / "state"
        root = state_dir / "image-bindings"
        if create:
            self._ensure_directory(state_dir)
        if not os.path.lexists(state_dir):
            if self._state_identity is not None:
                raise FilmBindingStoreError("state-parent-identity-changed")
            return None
        if state_dir.is_symlink() or not state_dir.is_dir():
            raise FilmBindingStoreError("state-directory-invalid")
        observed_state = self._pin_directory(state_dir, "state-parent-unavailable")
        if self._state_identity is None:
            self._state_identity = observed_state
        elif not self._same_identity(self._state_identity, observed_state):
            raise FilmBindingStoreError("state-parent-identity-changed")
        if create:
            self._ensure_directory(root)
        elif not os.path.lexists(root):
            return None
        if root.is_symlink() or not root.is_dir():
            raise FilmBindingStoreError("state-directory-invalid")
        return root, self._pin_directory(root, "state-parent-unavailable")

    def _read_handle(self, handle: Any) -> bytes:
        reader = None
        for name in ("read_handle_bytes", "read_handle", "read_bytes"):
            candidate = getattr(self.boundary, name, None)
            if callable(candidate):
                reader = candidate
                break
        try:
            if callable(reader):
                raw = reader(handle, max_bytes=MAX_STATE_BYTES)
            else:
                raw = self._path_boundary_module().read_handle_bytes(handle, max_bytes=MAX_STATE_BYTES)
        except FilmBindingStoreError:
            raise
        except Exception as exc:
            raise FilmBindingStoreError("state-read-failed") from exc
        if type(raw) is not bytes or len(raw) > MAX_STATE_BYTES:
            raise FilmBindingStoreError("state-size-invalid")
        return raw

    @staticmethod
    def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    @classmethod
    def _parse_canonical(cls, raw: bytes) -> dict[str, Any]:
        try:
            value = json.loads(raw.decode("utf-8"), object_pairs_hook=cls._reject_duplicate_pairs)
            if type(value) is not dict:
                raise ValueError("state must be an object")
            canonical = cls._canonical_bytes(value)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError, MemoryError, OverflowError) as exc:
            raise FilmBindingStoreError("state-invalid") from exc
        if canonical != raw:
            raise FilmBindingStoreError("state-not-canonical")
        return value

    @staticmethod
    def _canonical_bytes(value: Any) -> bytes:
        try:
            return json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, OverflowError, RecursionError, MemoryError) as exc:
            raise FilmBindingStoreError("state-not-canonical") from exc

    def _validate_envelope(self, value: Any, *, schema: str | None = None, revision: int | None = None) -> tuple[dict[str, Any], bytes]:
        if type(value) is not dict:
            raise FilmBindingStoreError("candidate-shape")
        try:
            normalized = validate_image_bindings(value, self.project_id)
        except FilmBindingContractError as exc:
            raise FilmBindingStoreError("state-contract-invalid") from exc
        except (TypeError, ValueError, OverflowError, RecursionError, MemoryError) as exc:
            raise FilmBindingStoreError("state-contract-invalid") from exc
        if schema is not None and normalized["schema"] != schema:
            raise FilmBindingStoreError("writer-schema-invalid")
        if revision is not None and normalized["revision"] != revision:
            raise FilmBindingStoreError("revision-sequence-invalid")
        if normalized["receipt_key_enrollment_id"] != self.receipt_key_enrollment_id:
            raise FilmBindingStoreError("owner-enrollment-mismatch")
        if normalized["receipt_key_fingerprint"] != self.receipt_key_fingerprint:
            raise FilmBindingStoreError("owner-fingerprint-mismatch")
        raw = self._canonical_bytes(normalized)
        if len(raw) > MAX_STATE_BYTES:
            raise FilmBindingStoreError("state-size-exceeded")
        return normalized, raw

    def _enumerate(self, root: Path) -> list[str]:
        names: list[str] = []
        state_records = 0
        internal_residues = 0
        try:
            with os.scandir(root) as iterator:
                for entry in iterator:
                    names.append(entry.name)
                    if len(names) > MAX_STATE_FILES + MAX_INTERNAL_RESIDUES:
                        raise FilmBindingStoreError("state-directory-breadth-exceeded")
                    if _STATE_FILE_RE.fullmatch(entry.name) is not None:
                        state_records += 1
                        if state_records > MAX_STATE_FILES:
                            raise FilmBindingStoreError("state-file-count-exceeded")
                    elif _INTERNAL_TOMBSTONE_RE.fullmatch(entry.name) is not None:
                        internal_residues += 1
                        if internal_residues > MAX_INTERNAL_RESIDUES:
                            raise FilmBindingStoreError("state-internal-residue-count-exceeded")
        except FilmBindingStoreError:
            raise
        except Exception as exc:
            raise FilmBindingStoreError("state-enumeration-failed") from exc
        return names

    def _load_records_locked(self, *, create: bool = False) -> tuple[Path | None, Any, list[tuple[str, dict[str, Any], bytes]]]:
        located = self._binding_root(create=create)
        if located is None:
            self._revalidate_state_parent()
            return None, None, []
        root, root_identity = located
        names = self._enumerate(root)
        records: list[tuple[str, dict[str, Any], bytes]] = []
        seen_revisions: set[int] = set()
        seen_operations: set[str] = set()
        total_bytes = 0
        for name in names:
            if _INTERNAL_TOMBSTONE_RE.fullmatch(name) is not None:
                continue
            match = _STATE_FILE_RE.fullmatch(name)
            if match is None:
                raise FilmBindingStoreError("state-unknown-child")
            revision = int(match.group(1))
            operation_id = match.group(3)
            if revision in seen_revisions or operation_id in seen_operations:
                raise FilmBindingStoreError("state-ambiguous-history")
            path = root / name
            handle, identity = self._open(path, directory=False)
            try:
                size = getattr(identity, "size", None)
                if type(size) is not int or size < 0 or size > MAX_STATE_BYTES or total_bytes + size > MAX_STATE_BYTES * MAX_STATE_FILES:
                    raise FilmBindingStoreError("state-size-exceeded")
                raw = self._read_handle(handle)
                if len(raw) != size:
                    raise FilmBindingStoreError("state-size-changed")
            finally:
                self._close(handle)
            value = self._parse_canonical(raw)
            normalized, canonical = self._validate_envelope(value, revision=revision)
            if canonical != raw:
                raise FilmBindingStoreError("state-not-canonical")
            seen_revisions.add(revision)
            seen_operations.add(operation_id)
            total_bytes += len(raw)
            records.append((name, normalized, raw))
            self._revalidate_root(root, root_identity)
        if records:
            records.sort(key=lambda item: item[1]["revision"])
            expected = 1
            for _name, state, _raw in records:
                if state["revision"] != expected:
                    raise FilmBindingStoreError("state-revision-gap")
                expected += 1
        self._revalidate_root(root, root_identity)
        return root, root_identity, records

    def _revalidate_state_parent(self) -> None:
        if self._state_identity is None:
            return
        state_dir = self.project_dir / "state"
        try:
            _handle, identity = self._open(state_dir, directory=True)
            self._close(_handle)
            if not self._same_identity(self._state_identity, identity):
                raise FilmBindingStoreError("state-parent-identity-changed")
        except FilmBindingStoreError:
            raise
        except Exception as exc:
            raise FilmBindingStoreError("state-parent-identity-changed") from exc

    def _revalidate_root(self, root: Path, expected: Any) -> None:
        try:
            self._revalidate_state_parent()
            _handle, identity = self._open(root, directory=True)
            self._close(_handle)
            if not self._same_identity(expected, identity):
                raise FilmBindingStoreError("state-parent-identity-changed")
        except FilmBindingStoreError:
            raise
        except Exception as exc:
            raise FilmBindingStoreError("state-parent-identity-changed") from exc

    def _revalidate_final_chain(self, root: Path | None, root_identity: Any) -> None:
        if root is None or root_identity is None:
            self._revalidate_state_parent()
        else:
            self._revalidate_root(root, root_identity)
        self._revalidate_project(self._project_identity)

    @contextmanager
    def _project_guard(self) -> Iterator[Any]:
        lock_factory = getattr(self.boundary, "project_lock", None)
        if not callable(lock_factory):
            raise FilmBindingStoreError("project-lock-missing")
        lock = None
        try:
            lock = lock_factory(self.project_dir)
            acquired = lock.acquire()
            if acquired is False:
                raise FilmBindingStoreError("project-lock-unavailable")
            self._revalidate_project(self._project_identity)
            yield lock
        except FilmBindingStoreError:
            raise
        except Exception as exc:
            raise FilmBindingStoreError("project-lock-unavailable") from exc
        finally:
            if lock is not None:
                try:
                    close = getattr(lock, "close", None)
                    if callable(close):
                        close()
                    else:
                        lock.release()
                except Exception as exc:
                    raise FilmBindingStoreError("project-lock-release-failed") from exc

    def _find_operation(self, records: list[tuple[str, dict[str, Any], bytes]], operation_id: str, raw: bytes) -> dict[str, Any] | None:
        expected_kind = "append"
        for name, state, prior_raw in records:
            match = _STATE_FILE_RE.fullmatch(name)
            assert match is not None
            if match.group(3) == operation_id:
                if match.group(2) != expected_kind:
                    raise FilmBindingStoreError("operation-conflict")
                if prior_raw != raw:
                    raise FilmBindingStoreError("operation-conflict")
                return copy.deepcopy(state)
        return None

    def _find_upgrade_operation(
        self,
        records: list[tuple[str, dict[str, Any], bytes]],
        operation_id: str,
    ) -> tuple[dict[str, Any], int] | None:
        for name, state, operation_raw in records:
            match = _STATE_FILE_RE.fullmatch(name)
            assert match is not None
            if match.group(3) != operation_id:
                continue
            if match.group(2) != "upgrade":
                raise FilmBindingStoreError("operation-conflict")
            if state["schema"] != BINDINGS_V2_SCHEMA:
                raise FilmBindingStoreError("upgrade-operation-invalid")
            predecessor = next(
                (
                    prior_state
                    for _prior_name, prior_state, _prior_raw in records
                    if prior_state["revision"] == state["revision"] - 1
                    and prior_state["schema"] == BINDINGS_V1_SCHEMA
                ),
                None,
            )
            if predecessor is None:
                raise FilmBindingStoreError("upgrade-operation-invalid")
            projected = project_bindings_v1_to_v2(predecessor, self.project_id)
            projected["revision"] = state["revision"]
            projected["updated_at"] = state["updated_at"]
            try:
                _normalized, projected_raw = self._validate_envelope(
                    projected,
                    schema=BINDINGS_V2_SCHEMA,
                    revision=state["revision"],
                )
            except FilmBindingStoreError as exc:
                raise FilmBindingStoreError("upgrade-operation-invalid") from exc
            if projected_raw != operation_raw:
                raise FilmBindingStoreError("upgrade-operation-invalid")
            return copy.deepcopy(state), predecessor["revision"]
        return None

    @staticmethod
    def _binding_map(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {binding["binding_id"]: binding for binding in state["state"]["bindings"]}

    def _check_owner_target(
        self,
        previous: dict[str, Any] | None,
        candidate: dict[str, Any],
        target: BindingTarget | None,
        *,
        initial: bool,
    ) -> None:
        previous_bindings = {} if previous is None else self._binding_map(previous)
        candidate_bindings = self._binding_map(candidate)
        missing = set(previous_bindings) - set(candidate_bindings)
        if missing:
            raise FilmBindingStoreError("binding-history-truncated")
        for binding_id, old in previous_bindings.items():
            new = candidate_bindings[binding_id]
            if new["label"] != old["label"] or new["film_reference_id"] != old["film_reference_id"]:
                raise FilmBindingStoreError("binding-identity-mutated")
            old_adoptions = old["adoptions"]
            new_adoptions = new["adoptions"]
            if len(new_adoptions) < len(old_adoptions) or new_adoptions[: len(old_adoptions)] != old_adoptions:
                raise FilmBindingStoreError("adoption-history-mutated")
            if len(new_adoptions) == len(old_adoptions) and new["active_adoption_id"] != old["active_adoption_id"]:
                raise FilmBindingStoreError("active-adoption-mutated")
        new_ids = set(candidate_bindings) - set(previous_bindings)
        if initial:
            new_ids = set(candidate_bindings)
        if not new_ids:
            if target is not None:
                raise FilmBindingStoreError("binding-target-unused")
            return
        if len(new_ids) != 1 or target is None:
            raise FilmBindingStoreError("binding-target-invalid")
        self._require_live_target(target)
        new_id = next(iter(new_ids))
        new_binding = candidate_bindings[new_id]
        if new_id != target.binding_id or new_binding["film_reference_id"] != target.film_reference_id:
            raise FilmBindingStoreError("binding-target-mismatch")
        if not _BINDING_RE.fullmatch(target.binding_id) or not _FILM_REFERENCE_RE.fullmatch(target.film_reference_id):
            raise FilmBindingStoreError("binding-target-invalid")

    def _check_append_transition(
        self,
        previous: dict[str, Any] | None,
        candidate: dict[str, Any],
        target: BindingTarget | None,
        *,
        initial: bool,
    ) -> None:
        self._check_owner_target(previous, candidate, target, initial=initial)
        if previous is not None:
            old_receipts = previous["state"]["receipts"]
            new_receipts = candidate["state"]["receipts"]
            if len(new_receipts) < len(old_receipts) or new_receipts[: len(old_receipts)] != old_receipts:
                raise FilmBindingStoreError("receipt-history-mutated")
            if candidate["state"]["receipt_floor_at"] != previous["state"]["receipt_floor_at"]:
                raise FilmBindingStoreError("receipt-floor-mutated")

    def _write_stage(self, root: Path, operation_id: str, raw: bytes, root_identity: Any) -> tuple[Path, Any, Any]:
        stage = root / f".film-binding-stage-{operation_id}.json"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd: int | None = None
        handle = None
        identity = None
        created = False
        try:
            try:
                fd = os.open(stage, flags, 0o600)
            except FileExistsError as exc:
                raise FilmBindingStoreError("state-stage-collision") from exc
            except (OSError, ValueError) as exc:
                raise FilmBindingStoreError("state-stage-failed") from exc
            created = True
            view = memoryview(raw)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("short state write")
                view = view[written:]
            flush = getattr(self.boundary, "flush_private_file", None)
            if callable(flush):
                flush(fd)
            else:
                os.fsync(fd)
            identity = self._capture_stage_identity(fd, stage)
            os.close(fd)
            fd = None
            handle, opened_identity = self._open(stage, directory=False)
            if not self._same_identity(identity, opened_identity):
                raise FilmBindingStoreError("state-stage-identity-changed")
            identity = opened_identity
            if self._read_handle(handle) != raw:
                raise FilmBindingStoreError("state-stage-mismatch")
            return stage, handle, identity
        except BaseException as exc:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._close(handle)
            if created:
                if identity is None:
                    raise FilmBindingStoreError("uncertain-cleanup") from exc
                self._cleanup_exact(stage, identity, root_identity, hashlib.sha256(raw).hexdigest())
            raise

    def _link_no_replace(self, source: Any, parent: Any, name: str) -> None:
        try:
            link = getattr(self.boundary, "link_no_replace", None)
            if callable(link):
                link(source, parent, name)
                return
            module = self._path_boundary_module()
            if os.name == "nt":
                module.win_link_handle_no_replace(source, parent, name)
            else:
                module._posix_linkat_empty_path(int(source.raw), int(parent.raw), name)
        except FileExistsError:
            raise
        except FilmBindingStoreError:
            raise
        except Exception as exc:
            raise FilmBindingStoreError("state-publication-failed") from exc

    def _flush_pinned_windows_parent(self, root: Path, root_identity: Any) -> None:
        module = self._path_boundary_module()
        create_file = getattr(module, "_CreateFileW", None)
        native_handle_type = getattr(module, "_NativeHandle", None)
        identity_from_handle = getattr(module, "_identity_from_handle", None)
        flush_file_buffers = getattr(module, "_FlushFileBuffers", None)
        if not all(callable(item) for item in (create_file, native_handle_type, identity_from_handle, flush_file_buffers)):
            raise OSError("native parent flush is unavailable")
        native_path = getattr(module, "_strip_extended_prefix")(str(root))
        if len(native_path) >= 240 and not native_path.startswith("\\\\?\\"):
            native_path = "\\\\?\\" + native_path
        raw = create_file(
            native_path,
            getattr(module, "_GENERIC_WRITE") | getattr(module, "_FILE_READ_ATTRIBUTES"),
            getattr(module, "_FILE_SHARE_READ")
            | getattr(module, "_FILE_SHARE_WRITE")
            | getattr(module, "_FILE_SHARE_DELETE"),
            None,
            getattr(module, "_OPEN_EXISTING"),
            getattr(module, "_FILE_FLAG_BACKUP_SEMANTICS") | getattr(module, "_FILE_FLAG_OPEN_REPARSE_POINT"),
            None,
        )
        if raw in (None, getattr(module, "_INVALID_HANDLE_VALUE")):
            raise OSError("pinned parent could not be opened")
        handle = native_handle_type(raw, "directory")
        try:
            observed = identity_from_handle(handle, root)
            if not self._same_identity(root_identity, observed):
                raise OSError("pinned parent identity changed")
            if not bool(flush_file_buffers(handle.raw)):
                raise OSError("pinned parent flush failed")
        finally:
            self._close(handle)

    def _flush_parent(self, root: Path, root_identity: Any, parent_handle: Any = None) -> None:
        flush_handle = getattr(self.boundary, "fsync_directory_handle", None)
        flush = getattr(self.boundary, "fsync_parent", None)
        try:
            self._revalidate_root(root, root_identity)
            if callable(flush_handle):
                if parent_handle is None:
                    raise OSError("pinned parent handle unavailable")
                flush_handle(parent_handle, expected=root_identity)
            elif parent_handle is not None and getattr(parent_handle, "raw", None) is not None:
                if os.name == "nt":
                    self._flush_pinned_windows_parent(root, root_identity)
                else:
                    os.fsync(int(parent_handle.raw))
            elif callable(flush):
                flush(root)
            else:
                self._path_boundary_module()._fsync_parent(root)
            self._revalidate_root(root, root_identity)
        except FilmBindingStoreError:
            raise
        except Exception as exc:
            raise FilmBindingStoreError("state-durability-failed") from exc

    def _cleanup_exact(self, path: Path, identity: Any, root_identity: Any, digest: str) -> None:
        cleanup = getattr(self.boundary, "unlink_handle_bound", None)
        try:
            if callable(cleanup):
                removed = cleanup(
                    path,
                    identity,
                    parent_expected=root_identity,
                    expected_sha256=digest,
                    max_bytes=MAX_STATE_BYTES,
                )
            else:
                removed = self._path_boundary_module().unlink_handle_bound(
                    path,
                    identity,
                    parent_expected=root_identity,
                    expected_sha256=digest,
                    max_bytes=MAX_STATE_BYTES,
                )
        except Exception as exc:
            raise FilmBindingStoreError("uncertain-cleanup") from exc
        if removed is not True:
            raise FilmBindingStoreError("uncertain-cleanup")

    def _publish_locked(
        self,
        root: Path,
        root_identity: Any,
        candidate: dict[str, Any],
        raw: bytes,
        operation_id: str,
        operation_kind: str,
    ) -> dict[str, Any]:
        name = f"r{candidate['revision']:020d}-{operation_kind}-{operation_id}.json"
        target = root / name
        parent_handle = None
        stage_handle = None
        stage = None
        stage_identity = None
        stage_cleaned = False
        target_handle = None
        try:
            try:
                parent_handle, parent_observed = self._open(root, directory=True)
                if not self._same_identity(root_identity, parent_observed):
                    raise FilmBindingStoreError("state-parent-identity-changed")
                stage, stage_handle, stage_identity = self._write_stage(root, operation_id, raw, root_identity)
                self._revalidate_project(self._project_identity)
                self._revalidate_root(root, root_identity)
                self._link_no_replace(stage_handle, parent_handle, name)
                self._revalidate_root(root, root_identity)
                self._flush_parent(root, root_identity, parent_handle)
            except FileExistsError as exc:
                self._close(stage_handle)
                stage_handle = None
                if stage is not None and stage_identity is not None:
                    self._cleanup_exact(stage, stage_identity, root_identity, hashlib.sha256(raw).hexdigest())
                raise FilmBindingStoreError("state-publication-collision") from exc
            except FilmBindingStoreError:
                self._close(stage_handle)
                stage_handle = None
                if stage is not None and stage_identity is not None:
                    try:
                        self._cleanup_exact(stage, stage_identity, root_identity, hashlib.sha256(raw).hexdigest())
                    except FilmBindingStoreError:
                        pass
                raise
            except Exception as exc:
                self._close(stage_handle)
                stage_handle = None
                if stage is not None and stage_identity is not None:
                    try:
                        self._cleanup_exact(stage, stage_identity, root_identity, hashlib.sha256(raw).hexdigest())
                    except FilmBindingStoreError:
                        pass
                raise FilmBindingStoreError("state-publication-failed") from exc
            finally:
                self._close(stage_handle)
                stage_handle = None
            try:
                target_handle, target_identity = self._open(target, directory=False)
                target_raw = self._read_handle(target_handle)
                if not self._same_identity(stage_identity, target_identity) or target_raw != raw:
                    raise FilmBindingStoreError("state-publication-collision")
                self._close(target_handle)
                target_handle = None
                # This second object check is the publication linearization
                # point: it proves the pathname still names the exact object
                # whose bytes were read while the project lock is held.
                target_handle, revalidated_target_identity = self._open(target, directory=False)
                try:
                    revalidated_target_raw = self._read_handle(target_handle)
                finally:
                    self._close(target_handle)
                    target_handle = None
                if (
                    not self._same_identity(target_identity, revalidated_target_identity)
                    or revalidated_target_raw != raw
                ):
                    raise FilmBindingStoreError("state-publication-collision")
                self._cleanup_exact(stage, stage_identity, root_identity, hashlib.sha256(raw).hexdigest())
                stage_cleaned = True
                self._revalidate_root(root, root_identity)
                self._flush_parent(root, root_identity, parent_handle)
                # No target check can be meaningfully "last" under an
                # external race.  The precise guarantee is the exact target
                # identity at the second object check above, followed by the
                # pinned flush and these project/state/root chain checks.
                self._revalidate_final_chain(root, root_identity)
                return copy.deepcopy(candidate)
            except FilmBindingStoreError as exc:
                if not stage_cleaned:
                    try:
                        self._cleanup_exact(stage, stage_identity, root_identity, hashlib.sha256(raw).hexdigest())
                    except FilmBindingStoreError as cleanup_exc:
                        raise cleanup_exc from exc
                raise
            except Exception as exc:
                if not stage_cleaned:
                    try:
                        self._cleanup_exact(stage, stage_identity, root_identity, hashlib.sha256(raw).hexdigest())
                    except FilmBindingStoreError as cleanup_exc:
                        raise cleanup_exc from exc
                raise FilmBindingStoreError("state-publication-failed") from exc
        finally:
            self._close(target_handle)
            self._close(parent_handle)

    def _execute_publish(
        self,
        candidate: Any,
        operation_id: str,
        *,
        expected_revision: int,
        initial: bool,
        target: BindingTarget | None,
        operation_kind: str = "append",
    ) -> dict[str, Any]:
        operation_id = self._strict_operation_id(operation_id)
        if operation_kind not in _OPERATION_KINDS:
            raise FilmBindingStoreError("operation-kind-invalid")
        if initial:
            self._strict_revision(expected_revision, positive=False)
            required_revision = 1
        else:
            self._strict_revision(expected_revision, positive=True)
            required_revision = expected_revision + 1
        normalized, raw = self._validate_envelope(candidate, schema=BINDINGS_V2_SCHEMA, revision=required_revision)
        with self._project_guard():
            root, root_identity, records = self._load_records_locked(create=True)
            assert root is not None and root_identity is not None
            if operation_kind == "append":
                replay = self._find_operation(records, operation_id, raw)
            else:
                upgrade_replay = self._find_upgrade_operation(records, operation_id)
                replay = None if upgrade_replay is None else upgrade_replay[0]
            if replay is not None:
                if target is not None:
                    raise FilmBindingStoreError("binding-target-unused")
                self._consume_replayed_target(replay)
                self._revalidate_final_chain(root, root_identity)
                return replay
            if len(records) >= MAX_STATE_FILES:
                raise FilmBindingStoreError("state-file-count-exceeded")
            current = records[-1][1] if records else None
            if initial:
                if records or expected_revision != 0:
                    raise FilmBindingStoreError("initial-cas-invalid")
            elif not records or current is None or current["revision"] != expected_revision:
                raise FilmBindingStoreError("revision-conflict")
            self._check_append_transition(current, normalized, target, initial=initial)
            result = self._publish_locked(root, root_identity, normalized, raw, operation_id, operation_kind)
            if target is not None:
                self._remove_pending_target(target)
            self._revalidate_final_chain(root, root_identity)
            return result

    def current_revision(self) -> int:
        with self._project_guard():
            root, root_identity, records = self._load_records_locked(create=False)
            result = records[-1][1]["revision"] if records else 0
            self._revalidate_final_chain(root, root_identity)
            return result

    def load(self) -> dict[str, Any] | None:
        with self._project_guard():
            root, root_identity, records = self._load_records_locked(create=False)
            result = None if not records else copy.deepcopy(records[-1][1])
            self._revalidate_final_chain(root, root_identity)
            return result

    read = load
    get_current = load

    def mint_binding_target(self) -> BindingTarget:
        with self._project_guard():
            now = self._clock_epoch()
            self._purge_expired_targets(now)
            if len(self._issued_targets) >= MAX_PENDING_TARGETS:
                raise FilmBindingStoreError("pending-target-capacity-exceeded")
            root, root_identity, records = self._load_records_locked(create=False)
            used_bindings: set[str] = set()
            used_references: set[str] = set()
            for _name, state, _raw in records:
                for binding in state["state"]["bindings"]:
                    used_bindings.add(binding["binding_id"])
                    used_references.add(binding["film_reference_id"])
            while True:
                binding_id = "imgbind_" + secrets.token_hex(16)
                film_reference_id = "filmref_" + secrets.token_hex(16)
                if (
                    binding_id not in used_bindings
                    and film_reference_id not in used_references
                    and not any(
                        issued.binding_id == binding_id or issued.film_reference_id == film_reference_id
                        for issued in self._issued_targets
                    )
                ):
                    break
            target = BindingTarget(binding_id, film_reference_id, secrets.token_bytes(32))
            self._target_sequence += 1
            expires_at = now + TARGET_TTL_SECONDS
            self._issued_targets[target] = expires_at
            heapq.heappush(self._target_expiry_heap, (expires_at, self._target_sequence, target))
            self._revalidate_final_chain(root, root_identity)
            return target

    def abandon_binding_target(self, target: BindingTarget) -> None:
        """Release one still-pending target without publishing state."""
        if type(target) is not BindingTarget:
            raise FilmBindingStoreError("binding-target-invalid")
        with self._project_guard():
            root, root_identity, _records = self._load_records_locked(create=False)
            found = self._find_pending_target(target)
            if found is None:
                self._revalidate_final_chain(root, root_identity)
                raise FilmBindingStoreError("binding-target-invalid")
            issued, expires_at = found
            now = self._clock_epoch()
            if expires_at <= now:
                self._remove_pending_target(issued)
                self._purge_expired_targets(now)
                self._revalidate_final_chain(root, root_identity)
                raise FilmBindingStoreError("binding-target-expired")
            self._remove_pending_target(issued)
            self._revalidate_final_chain(root, root_identity)

    mint_ids = mint_binding_target

    def publish_initial(
        self,
        candidate: dict[str, Any],
        operation_id: str,
        *,
        target: BindingTarget | None = None,
    ) -> dict[str, Any]:
        return self._execute_publish(candidate, operation_id, expected_revision=0, initial=True, target=target)

    def publish_next(
        self,
        expected_revision: int,
        candidate: dict[str, Any],
        operation_id: str,
        *,
        target: BindingTarget | None = None,
    ) -> dict[str, Any]:
        return self._execute_publish(
            candidate,
            operation_id,
            expected_revision=expected_revision,
            initial=False,
            target=target,
        )

    append_initial = publish_initial
    append_next = publish_next

    def upgrade_v1_to_v2(
        self,
        operation_id: str,
        *,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        operation_id = self._strict_operation_id(operation_id)
        supplied_revision = None if expected_revision is None else self._strict_revision(expected_revision, positive=True)
        with self._project_guard():
            root, root_identity, records = self._load_records_locked(create=False)
            if not records:
                raise FilmBindingStoreError("state-missing")
            replay = self._find_upgrade_operation(records, operation_id)
            if replay is not None:
                replay_state, source_revision = replay
                if supplied_revision is not None and supplied_revision != source_revision:
                    raise FilmBindingStoreError("revision-conflict")
                self._revalidate_final_chain(root, root_identity)
                return replay_state
            if len(records) >= MAX_STATE_FILES:
                raise FilmBindingStoreError("state-file-count-exceeded")
            current = records[-1][1]
            if current["schema"] != BINDINGS_V1_SCHEMA:
                raise FilmBindingStoreError("upgrade-source-not-v1")
            current_revision = current["revision"]
            expected = current_revision if supplied_revision is None else supplied_revision
            if expected != current_revision:
                raise FilmBindingStoreError("revision-conflict")
            projected = project_bindings_v1_to_v2(current, self.project_id)
            projected["revision"] = current_revision + 1
            projected["updated_at"] = self._clock_timestamp()
        return self._execute_publish(
            projected,
            operation_id,
            expected_revision=expected,
            initial=False,
            target=None,
            operation_kind="upgrade",
        )


__all__ = [
    "BindingStoreError",
    "BindingTarget",
    "FilmBindingStore",
    "FilmBindingStoreError",
    "MAX_PENDING_TARGETS",
    "MAX_INTERNAL_RESIDUES",
    "MAX_STATE_BYTES",
    "MAX_STATE_FILES",
    "TARGET_TTL_SECONDS",
]
