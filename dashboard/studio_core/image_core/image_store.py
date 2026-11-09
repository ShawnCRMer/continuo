from __future__ import annotations

import copy
import datetime as _datetime
import hashlib
import secrets
import hmac
import importlib
import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Protocol

try:
    contract = importlib.import_module("film_suite_generation.image_contract")
except ImportError:  # pragma: no cover - direct isolated module loading
    contract = None
    for _name, _module in tuple(sys.modules.items()):
        if _name.endswith(".image_contract") and hasattr(_module, "validate_image_state"):
            contract = _module
            break
    if contract is None:
        _path = Path(__file__).parent.parent.parent / "generation" / "image_contract.py"
        _spec = importlib.util.spec_from_file_location("_image_state_store_contract", _path)
        if _spec is None or _spec.loader is None:
            raise ImportError("image contract unavailable")
        contract = importlib.util.module_from_spec(_spec)
        sys.modules[_spec.name] = contract
        _spec.loader.exec_module(contract)


IMAGE_OWNER_CLAIM_SCHEMA = "studio.image-owner-claim/1"
MAX_STATE_FILES = 256
MAX_STATE_BYTES = 4 * 1024 * 1024
MAX_CONTEXTS = 256
MAX_FENCED_RESULT_BYTES = max(MAX_STATE_BYTES, int(getattr(contract, "MAX_BYTES", MAX_STATE_BYTES)))
_OWNER_TOKEN_RE = re.compile(r"^imgown_[0-9a-f]{64}$")
_CONTEXT_RE = re.compile(r"^imgctx_[0-9a-f]{32}$")
_WORK_RE = re.compile(r"^imgwork_[0-9a-f]{32}$")
_REVISION_RE = re.compile(r"^imgrev_[0-9a-f]{32}$")
_OPERATION_RE = re.compile(r"^imgop_[0-9a-f]{32}$")
_ENROLLMENT_RE = re.compile(r"^imgkey_[0-9a-f]{32}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_STATE_FILE_RE = re.compile(r"^r([0-9]{20})-(imgop_[0-9a-f]{32})\.json$")
_CLAIM_FIELDS = frozenset(
    {
        "schema",
        "context_id",
        "work_id",
        "basis_kind",
        "revision_id",
        "draft_generation",
        "basis_content_sha256",
        "expected_revision",
    }
)


class ImageStateError(ValueError):
    """Finite fail-closed error for the standalone Image state owner."""

    def __init__(self, code: str) -> None:
        self.code = code
        self.reason_code = code
        super().__init__(code)


ImageStoreError = ImageStateError


class ImageOwnerAuthority(Protocol):
    receipt_key_enrollment_id: str
    receipt_key_fingerprint: str

    def sign_owner_claim(self, canonical_claim: bytes) -> str: ...

    def verify_owner_claim(self, canonical_claim: bytes, token: str) -> bool: ...


class HmacImageOwnerAuthority:
    """Issue and verify exact-byte owner claims from an already-loaded key."""

    __slots__ = ("receipt_key_enrollment_id", "receipt_key_fingerprint", "_key", "_sealed")

    def __init__(
        self,
        key: bytes | None = None,
        receipt_key_enrollment_id: str | None = None,
        receipt_key_fingerprint: str | None = None,
        *,
        profile_key: bytes | None = None,
    ) -> None:
        if key is None:
            key = profile_key
        elif profile_key is not None:
            raise ImageStateError("owner-key-ambiguous")
        if type(key) is not bytes or len(key) != 32:
            raise ImageStateError("owner-key-length-invalid")
        if type(receipt_key_enrollment_id) is not str or _ENROLLMENT_RE.fullmatch(receipt_key_enrollment_id) is None:
            raise ImageStateError("owner-enrollment-id-invalid")
        if type(receipt_key_fingerprint) is not str or _HEX64_RE.fullmatch(receipt_key_fingerprint) is None:
            raise ImageStateError("owner-fingerprint-invalid")
        actual = hashlib.sha256(key).hexdigest()
        if not hmac.compare_digest(actual, receipt_key_fingerprint):
            raise ImageStateError("owner-fingerprint-mismatch")
        object.__setattr__(self, "receipt_key_enrollment_id", receipt_key_enrollment_id)
        object.__setattr__(self, "receipt_key_fingerprint", receipt_key_fingerprint)
        object.__setattr__(self, "_key", key)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("owner authority is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("owner authority is immutable")
        object.__delattr__(self, name)

    @staticmethod
    def _claim_bytes(canonical_claim: bytes) -> bytes:
        if type(canonical_claim) is not bytes:
            raise ImageStateError("owner-claim-bytes-required")
        return canonical_claim

    def sign_owner_claim(self, canonical_claim: bytes) -> str:
        return "imgown_" + hmac.new(self._key, self._claim_bytes(canonical_claim), hashlib.sha256).hexdigest()

    def verify_owner_claim(self, canonical_claim: bytes, token: str) -> bool:
        if type(token) is not str or _OWNER_TOKEN_RE.fullmatch(token) is None:
            return False
        try:
            expected = self.sign_owner_claim(canonical_claim)
        except ImageStateError:
            return False
        return hmac.compare_digest(expected, token)


class ImageStateStore:
    """Append-only owner of one profile-private ``studio.image-state/2`` context."""

    def __init__(
        self,
        image_contexts_root: Path | str,
        boundary: Any = None,
        authority: ImageOwnerAuthority | None = None,
        *,
        clock: Callable[[], str] | None = None,
    ) -> None:
        if boundary is None:
            raise ImageStateError("path-boundary-required")
        if authority is None:
            raise ImageStateError("owner-authority-required")
        root = Path(image_contexts_root)
        if root.name != "image-contexts":
            raise ImageStateError("image-contexts-root-invalid")
        enrollment_id = getattr(authority, "receipt_key_enrollment_id", None)
        fingerprint = getattr(authority, "receipt_key_fingerprint", None)
        if type(enrollment_id) is not str or _ENROLLMENT_RE.fullmatch(enrollment_id) is None:
            raise ImageStateError("owner-enrollment-id-invalid")
        if type(fingerprint) is not str or _HEX64_RE.fullmatch(fingerprint) is None:
            raise ImageStateError("owner-fingerprint-invalid")
        self.image_contexts_root = root
        self.boundary = boundary
        self.authority = authority
        self.clock = clock or self._utc_second
        self._root_identity = self._open_and_close(root, directory=True)[1]

    @staticmethod
    def _utc_second() -> str:
        return _datetime.datetime.now(_datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _close(handle: Any) -> None:
        close = getattr(handle, "close", None)
        if callable(close):
            close()

    def _open(self, path: Path, *, directory: bool) -> tuple[Any, Any]:
        opener = getattr(self.boundary, "open_existing", None)
        if not callable(opener):
            raise ImageStateError("path-boundary-open-missing")
        try:
            handle, identity = opener(path, directory=directory)
        except ImageStateError:
            raise
        except Exception as exc:
            raise ImageStateError("state-object-unavailable") from exc
        if handle is None or identity is None:
            raise ImageStateError("state-object-unavailable")
        if bool(getattr(identity, "is_directory", directory)) is not directory:
            self._close(handle)
            raise ImageStateError("state-object-type-invalid")
        if any(bool(getattr(identity, name, False)) for name in ("is_reparse", "is_reparse_point", "reparse")):
            self._close(handle)
            raise ImageStateError("state-reparse-object")
        return handle, identity

    def _open_and_close(self, path: Path, *, directory: bool) -> tuple[None, Any]:
        handle, identity = self._open(path, directory=directory)
        self._close(handle)
        return None, identity

    @staticmethod
    def _same_identity(left: Any, right: Any) -> bool:
        missing = object()

        def attribute(value: Any, names: tuple[str, ...]) -> Any:
            for name in names:
                try:
                    result = getattr(value, name, missing)
                except Exception:
                    return missing
                if result is not missing:
                    return result
            return missing

        for candidate, other in ((left, right), (right, left)):
            try:
                same = getattr(candidate, "same_object", None)
            except Exception:
                return False
            if callable(same):
                try:
                    if not bool(same(other)):
                        return False
                except Exception:
                    return False
                break

        left_directory = attribute(left, ("is_directory",))
        right_directory = attribute(right, ("is_directory",))
        if left_directory is missing or right_directory is missing:
            return False
        if bool(left_directory) != bool(right_directory):
            return False

        left_id = attribute(left, ("file_id",))
        right_id = attribute(right, ("file_id",))
        if left_id is missing or right_id is missing or left_id is None or right_id is None:
            return False

        volume_attributes = ("volume_serial", "volume_id", "volume")
        volume_proof = False
        for name in volume_attributes:
            left_volume = attribute(left, (name,))
            right_volume = attribute(right, (name,))
            if left_volume is missing and right_volume is missing:
                continue
            if (
                left_volume is missing
                or right_volume is missing
                or left_volume is None
                or right_volume is None
                or left_volume != right_volume
            ):
                return False
            volume_proof = True
        if not volume_proof:
            # Test/adapter identities may carry the volume in a structured
            # file_id.  Require the complete tuple rather than matching only
            # selected components of it.
            if not (
                isinstance(left_id, tuple)
                and isinstance(right_id, tuple)
                and len(left_id) >= 3
                and len(left_id) == len(right_id)
            ):
                return False
        try:
            same_file = left_id == right_id
        except Exception:
            return False
        return bool(same_file)

    def _check_root(self) -> None:
        handle, identity = self._open(self.image_contexts_root, directory=True)
        try:
            if not self._same_identity(self._root_identity, identity):
                raise ImageStateError("image-contexts-root-identity-changed")
        finally:
            self._close(handle)

    def _read(self, handle: Any) -> bytes:
        reader = None
        for name in ("read_handle_bytes", "read_handle", "read_bytes"):
            candidate = getattr(self.boundary, name, None)
            if callable(candidate):
                reader = candidate
                break
        if reader is None:
            raise ImageStateError("path-boundary-read-missing")
        try:
            data = reader(handle, max_bytes=MAX_STATE_BYTES)
        except Exception as exc:
            raise ImageStateError("state-read-failed") from exc
        if type(data) is not bytes or len(data) > MAX_STATE_BYTES:
            raise ImageStateError("state-size-invalid")
        return data

    def _enumerate_child_entries(
        self,
        parent_handle: Any,
        parent_path: Path,
        *,
        max_entries: int,
        error_code: str,
    ) -> list[tuple[str, Any]]:
        """Enumerate direct child names and identities from one pinned handle."""
        enumerate_entries = getattr(self.boundary, "enumerate_child_entries", None)
        if not callable(enumerate_entries):
            raise ImageStateError("path-boundary-enumeration-identity-missing")
        try:
            entries = enumerate_entries(
                parent_handle,
                parent_path,
                max_entries=max_entries,
            )
        except ImageStateError:
            raise
        except Exception as exc:
            boundary_code = getattr(exc, "code", None)
            message = str(exc).casefold()
            if boundary_code == "directory-entry-limit" or "entry count exceeded" in message:
                if error_code.startswith("context"):
                    raise ImageStateError("context-count-exceeded") from exc
                raise ImageStateError("state-file-count-exceeded") from exc
            raise ImageStateError(error_code) from exc
        if type(entries) is not list or len(entries) > max_entries:
            raise ImageStateError(error_code)
        validated: list[tuple[str, Any]] = []
        for entry in entries:
            if (
                type(entry) is not tuple
                or len(entry) != 2
                or type(entry[0]) is not str
                or not entry[0]
                or entry[1] is None
            ):
                raise ImageStateError(error_code)
            validated.append((entry[0], entry[1]))
        return validated

    @staticmethod
    def _same_enumerated_identity(expected: Any, observed: Any, *, directory: bool) -> bool:
        """Match identity, type, and - for regular files - the captured size.

        Size is a genuine tamper signal for a FILE: bytes changing under a
        pinned handle means someone rewrote it.  For a DIRECTORY the number is
        filesystem bookkeeping, not content: on Windows it steps from 0 to 4096
        as soon as a second entry lands, so comparing it made a context
        unreadable the moment it held a second revision.  Identity and type
        still have to match either way.
        """
        if not ImageStateStore._same_identity(expected, observed):
            return False
        try:
            if bool(getattr(expected, "is_directory")) is not directory:
                return False
            if bool(getattr(observed, "is_directory")) is not directory:
                return False
            if directory:
                return True
            expected_size = int(getattr(expected, "size"))
            observed_size = int(getattr(observed, "size"))
            if expected_size < 0 or observed_size != expected_size:
                return False
        except Exception:
            return False
        return True

    @staticmethod
    def _same_enumerated_entries(
        expected: Any,
        observed: Any,
        *,
        directory: bool,
    ) -> bool:
        """Require the same unique ordered names and complete child identities."""
        if type(expected) is not list or type(observed) is not list or len(expected) != len(observed):
            return False
        expected_names: list[str] = []
        observed_names: list[str] = []
        for entries, names in ((expected, expected_names), (observed, observed_names)):
            for entry in entries:
                if (
                    type(entry) is not tuple
                    or len(entry) != 2
                    or type(entry[0]) is not str
                    or not entry[0]
                    or entry[1] is None
                ):
                    return False
                if entry[0] in names:
                    return False
                names.append(entry[0])
        if expected_names != observed_names or set(expected_names) != set(observed_names):
            return False
        return all(
            ImageStateStore._same_enumerated_identity(
                expected_entry[1],
                observed_entry[1],
                directory=directory,
            )
            for expected_entry, observed_entry in zip(expected, observed)
        )

    def _open_child(
        self,
        parent_handle: Any,
        parent_path: Path,
        name: str,
        *,
        directory: bool,
    ) -> tuple[Any, Any]:
        """Open a named child relative to the already-pinned parent handle."""
        open_child = getattr(self.boundary, "open_child_existing", None)
        if not callable(open_child):
            raise ImageStateError("path-boundary-open-relative-missing")
        try:
            handle, identity = open_child(
                parent_handle,
                parent_path,
                name,
                directory=directory,
            )
        except ImageStateError:
            raise
        except Exception as exc:
            raise ImageStateError("state-object-unavailable") from exc
        if handle is None or identity is None:
            raise ImageStateError("state-object-unavailable")
        if bool(getattr(identity, "is_directory", directory)) is not directory:
            self._close(handle)
            raise ImageStateError("state-object-type-invalid")
        if any(bool(getattr(identity, field, False)) for field in ("is_reparse", "is_reparse_point", "reparse")):
            self._close(handle)
            raise ImageStateError("state-reparse-object")
        return handle, identity

    def _load_records_from_pinned_state(
        self,
        context_id: str,
        state_handle: Any,
        state_path: Path,
    ) -> list[tuple[str, dict[str, Any], bytes]]:
        """Load complete revisions using only one pinned state-directory handle."""
        entries = self._enumerate_child_entries(
            state_handle,
            state_path,
            max_entries=MAX_STATE_FILES,
            error_code="state-enumeration-failed",
        )
        records: list[tuple[str, dict[str, Any], bytes]] = []
        seen_revisions: set[int] = set()
        total_bytes = 0
        for name, enumerated_identity in entries:
            match = _STATE_FILE_RE.fullmatch(name)
            if match is None:
                raise ImageStateError("state-unknown-child")
            revision = int(match.group(1))
            try:
                handle, identity = self._open_child(
                    state_handle,
                    state_path,
                    name,
                    directory=False,
                )
            except ImageStateError as exc:
                if exc.code in {"state-object-unavailable", "state-object-type-invalid", "state-reparse-object"}:
                    raise ImageStateError("state-file-identity-changed") from exc
                raise
            try:
                if not self._same_enumerated_identity(
                    enumerated_identity,
                    identity,
                    directory=False,
                ):
                    raise ImageStateError("state-file-identity-changed")
                file_size = int(getattr(identity, "size", -1))
                if file_size < 0 or file_size > MAX_STATE_BYTES or total_bytes + file_size > MAX_STATE_BYTES:
                    raise ImageStateError("state-size-exceeded")
                raw = self._read(handle)
                if len(raw) != file_size:
                    raise ImageStateError("state-size-changed")
            finally:
                self._close(handle)
            try:
                value = contract.parse_canonical(raw)
                normalized = contract.validate_image_state(value)
            except Exception as exc:
                raise ImageStateError("state-invalid") from exc
            if normalized["revision"] != revision or normalized["context_id"] != context_id:
                raise ImageStateError("state-filename-mismatch")
            if (
                normalized["receipt_key_enrollment_id"] != self.authority.receipt_key_enrollment_id
                or normalized["receipt_key_fingerprint"] != self.authority.receipt_key_fingerprint
            ):
                raise ImageStateError("state-enrollment-mismatch")
            if revision in seen_revisions:
                raise ImageStateError("state-duplicate-revision")
            seen_revisions.add(revision)
            total_bytes += len(raw)
            records.append((name, normalized, raw))
        final_entries = self._enumerate_child_entries(
            state_handle,
            state_path,
            max_entries=MAX_STATE_FILES,
            error_code="state-enumeration-failed",
        )
        if not self._same_enumerated_entries(entries, final_entries, directory=False):
            raise ImageStateError("state-file-identity-changed")
        if not records:
            raise ImageStateError("state-missing")
        records.sort(key=lambda item: item[1]["revision"])
        return records

    def _revalidate_state_parent(self, state_dir: Path, expected: Any) -> None:
        handle, identity = self._open(state_dir, directory=True)
        try:
            if not self._same_identity(expected, identity):
                raise ImageStateError("state-parent-identity-changed")
        finally:
            self._close(handle)

    def _ensure_private_node(self, path: Path, *, directory: bool) -> None:
        ensure = getattr(self.boundary, "ensure_private_node", None)
        if callable(ensure):
            try:
                ensure(path, directory=directory)
            except Exception as exc:
                raise ImageStateError("state-directory-unavailable") from exc
            return
        try:
            self._open_and_close(path, directory=directory)
        except ImageStateError as exc:
            raise ImageStateError("state-directory-unavailable") from exc

    def _ensure_context_dirs(self, context_id: str) -> tuple[Path, Path, Any, Any]:
        self._check_root()
        context_dir = self.image_contexts_root / context_id
        state_dir = context_dir / "state"
        self._ensure_private_node(context_dir, directory=True)
        self._ensure_private_node(state_dir, directory=True)
        self._check_root()
        _, context_identity = self._open_and_close(context_dir, directory=True)
        _, state_identity = self._open_and_close(state_dir, directory=True)
        self._validate_context_children(context_dir)
        return context_dir, state_dir, context_identity, state_identity

    def _validate_context_children(self, context_dir: Path) -> None:
        entries: list[Any] = []
        try:
            with os.scandir(context_dir) as iterator:
                for entry in iterator:
                    entries.append(entry)
                    if len(entries) > 1:
                        raise ImageStateError("context-unknown-child")
        except ImageStateError:
            raise
        except Exception as exc:
            raise ImageStateError("context-enumeration-failed") from exc
        if len(entries) != 1 or entries[0].name != "state":
            raise ImageStateError("context-unknown-child")
        self._open_and_close(context_dir / "state", directory=True)

    def _context_dirs(self, context_id: str, *, create: bool) -> tuple[Path, Path, Any, Any]:
        self._validate_context_id(context_id)
        if create:
            return self._ensure_context_dirs(context_id)
        context_dir = self.image_contexts_root / context_id
        state_dir = context_dir / "state"
        self._check_root()
        _, context_identity = self._open_and_close(context_dir, directory=True)
        _, state_identity = self._open_and_close(state_dir, directory=True)
        self._validate_context_children(context_dir)
        return context_dir, state_dir, context_identity, state_identity

    def _revalidate_context(self, context_dir: Path, expected: Any) -> None:
        handle, identity = self._open(context_dir, directory=True)
        try:
            if not self._same_identity(expected, identity):
                raise ImageStateError("context-identity-changed")
        finally:
            self._close(handle)

    def _lock(self, context_dir: Path) -> Any:
        factory = getattr(self.boundary, "project_lock", None)
        if not callable(factory):
            raise ImageStateError("path-boundary-lock-missing")
        try:
            lock = factory(context_dir)
            acquired = lock.acquire()
            if acquired is False:
                raise ImageStateError("state-lock-unavailable")
            return lock
        except ImageStateError:
            raise
        except Exception as exc:
            raise ImageStateError("state-lock-unavailable") from exc

    @staticmethod
    def _validate_context_id(value: Any) -> str:
        if type(value) is not str or _CONTEXT_RE.fullmatch(value) is None:
            raise ImageStateError("context-id-invalid")
        return value

    @staticmethod
    def _validate_work_id(value: Any) -> str:
        if type(value) is not str or _WORK_RE.fullmatch(value) is None:
            raise ImageStateError("work-id-invalid")
        return value

    @staticmethod
    def _validate_revision_id(value: Any) -> str:
        if type(value) is not str or _REVISION_RE.fullmatch(value) is None:
            raise ImageStateError("revision-id-invalid")
        return value

    @staticmethod
    def _validate_operation_id(value: Any) -> str:
        if type(value) is not str or _OPERATION_RE.fullmatch(value) is None:
            raise ImageStateError("operation-id-invalid")
        return value

    def _validate_candidate(self, candidate: Any, context_id: str, *, revision: int | None = None) -> tuple[dict[str, Any], bytes]:
        if type(candidate) is not dict:
            raise ImageStateError("candidate-shape")
        try:
            normalized = contract.validate_image_state(candidate)
            raw = contract.canonical_bytes(normalized)
        except Exception as exc:
            raise ImageStateError("candidate-invalid") from exc
        if normalized["context_id"] != context_id:
            raise ImageStateError("candidate-context-mismatch")
        if revision is not None and normalized["revision"] != revision:
            raise ImageStateError("candidate-revision-invalid")
        if (
            normalized["receipt_key_enrollment_id"] != self.authority.receipt_key_enrollment_id
            or normalized["receipt_key_fingerprint"] != self.authority.receipt_key_fingerprint
        ):
            raise ImageStateError("state-enrollment-mismatch")
        return normalized, raw

    def _load_records_locked(
        self,
        context_id: str,
        state_dir: Path,
        expected_state_identity: Any = None,
    ) -> list[tuple[str, dict[str, Any], bytes]]:
        """Load one complete state set from a single pinned state directory."""
        state_parent_handle, state_parent_identity = self._open(state_dir, directory=True)
        try:
            if expected_state_identity is not None and not self._same_identity(
                expected_state_identity,
                state_parent_identity,
            ):
                raise ImageStateError("state-parent-identity-changed")
            return self._load_records_from_pinned_state(
                context_id,
                state_parent_handle,
                state_dir,
            )
        finally:
            self._close(state_parent_handle)

    def _load_locked(
        self,
        context_id: str,
        state_dir: Path,
        expected_state_identity: Any = None,
    ) -> dict[str, Any]:
        records = self._load_records_locked(context_id, state_dir, expected_state_identity)
        return copy.deepcopy(records[-1][1])

    def load(self, context_id: str) -> dict[str, Any]:
        context_id = self._validate_context_id(context_id)
        context_dir, state_dir, context_identity, state_identity = self._context_dirs(context_id, create=False)
        lock = self._lock(context_dir)
        try:
            self._revalidate_context(context_dir, context_identity)
            result = self._load_locked(context_id, state_dir, state_identity)
            self._revalidate_context(context_dir, context_identity)
            return result
        finally:
            self._close(lock)

    def _read_fenced(
        self,
        context_id: str,
        consumer: Callable[[dict[str, Any]], bytes],
        *,
        max_bytes: int = MAX_STATE_BYTES,
    ) -> bytes:
        """Run one bounded read while the context's cross-process lock is held.

        The callback receives a defensive copy of the validated current state.
        It is deliberately invoked before the final context check so a caller
        cannot publish bytes from a state that was changed by another process
        between provenance validation and the byte read.
        """
        context_id = self._validate_context_id(context_id)
        if not callable(consumer):
            raise ImageStateError("fenced-read-consumer-invalid")
        if (
            type(max_bytes) is not int
            or isinstance(max_bytes, bool)
            or max_bytes < 1
            or max_bytes > MAX_FENCED_RESULT_BYTES
        ):
            raise ImageStateError("fenced-read-bound-invalid")

        context_dir, state_dir, context_identity, state_identity = self._context_dirs(context_id, create=False)
        lock = self._lock(context_dir)
        result: bytes | None = None
        pending: BaseException | None = None
        pending_traceback = None
        try:
            try:
                self._revalidate_context(context_dir, context_identity)
                state = self._load_locked(context_id, state_dir, state_identity)
                consumed = consumer(copy.deepcopy(state))
                if type(consumed) is not bytes or len(consumed) > max_bytes:
                    raise ImageStateError("fenced-read-result-invalid")
                result = consumed
            except BaseException as exc:
                pending = exc
                pending_traceback = exc.__traceback__
            finally:
                try:
                    self._revalidate_context(context_dir, context_identity)
                except BaseException as exc:
                    if pending is None:
                        pending = exc
                        pending_traceback = exc.__traceback__
                try:
                    self._close(lock)
                except BaseException as exc:
                    if pending is None:
                        pending = exc
                        pending_traceback = exc.__traceback__
        except BaseException:
            # The nested cleanup above intentionally captures every exception
            # so a process-control exception from the consumer cannot be
            # replaced by lock or final-check cleanup.
            raise
        if pending is not None:
            raise pending.with_traceback(pending_traceback)
        if result is None:
            raise ImageStateError("fenced-read-result-invalid")
        return result

    def enumerate_contexts(self) -> list[dict[str, Any]]:
        """Read every published context through pinned directory handles.

        The root, each context, its single ``state`` child, and every revision
        filename are all enumerated/opened relative to the handle that owns the
        corresponding name.  Paths are retained only as diagnostic identity
        fallbacks; they never select an object for this catalog read.
        """
        root_handle, root_identity = self._open(self.image_contexts_root, directory=True)
        try:
            if not self._same_identity(self._root_identity, root_identity):
                raise ImageStateError("image-contexts-root-identity-changed")
            entries = self._enumerate_child_entries(
                root_handle,
                self.image_contexts_root,
                max_entries=MAX_CONTEXTS,
                error_code="context-enumeration-failed",
            )
            names = [name for name, _identity in entries]
            entry_identities = {name: identity for name, identity in entries}
            if len(set(names)) != len(names):
                raise ImageStateError("context-duplicate-child")
            for name in names:
                if _CONTEXT_RE.fullmatch(name) is None:
                    raise ImageStateError("context-unknown-child")
            states: list[dict[str, Any]] = []
            for context_id in sorted(names):
                context_path = self.image_contexts_root / context_id
                try:
                    expected_context_handle, expected_context_identity = self._open(
                        context_path,
                        directory=True,
                    )
                except ImageStateError as exc:
                    if exc.code in {"state-object-unavailable", "state-object-type-invalid", "state-reparse-object"}:
                        raise ImageStateError("context-identity-changed") from exc
                    raise
                try:
                    if not self._same_enumerated_identity(
                        entry_identities[context_id],
                        expected_context_identity,
                        directory=True,
                    ):
                        raise ImageStateError("context-identity-changed")
                finally:
                    self._close(expected_context_handle)
                context_lock = self._lock(context_path)
                try:
                    context_handle, context_identity = self._open_child(
                        root_handle,
                        self.image_contexts_root,
                        context_id,
                        directory=True,
                    )
                    try:
                        if (
                            not self._same_enumerated_identity(
                                entry_identities[context_id],
                                context_identity,
                                directory=True,
                            )
                            or not self._same_identity(expected_context_identity, context_identity)
                        ):
                            raise ImageStateError("context-identity-changed")
                        context_entries = self._enumerate_child_entries(
                            context_handle,
                            context_path,
                            max_entries=2,
                            error_code="context-enumeration-failed",
                        )
                        context_names = [name for name, _identity in context_entries]
                        if len(context_names) != 1 or context_names[0] != "state":
                            raise ImageStateError("context-unknown-child")
                        state_path = context_path / "state"
                        try:
                            state_handle, state_identity = self._open_child(
                                context_handle,
                                context_path,
                                "state",
                                directory=True,
                            )
                        except ImageStateError as exc:
                            if exc.code in {"state-object-unavailable", "state-object-type-invalid", "state-reparse-object"}:
                                raise ImageStateError("context-identity-changed") from exc
                            raise
                        try:
                            if not self._same_enumerated_identity(
                                context_entries[0][1],
                                state_identity,
                                directory=True,
                            ):
                                raise ImageStateError("state-parent-identity-changed")
                            records = self._load_records_from_pinned_state(
                                context_id,
                                state_handle,
                                state_path,
                            )
                            states.append(copy.deepcopy(records[-1][1]))
                            final_context_entries = self._enumerate_child_entries(
                                context_handle,
                                context_path,
                                max_entries=2,
                                error_code="context-enumeration-failed",
                            )
                            if not self._same_enumerated_entries(
                                context_entries,
                                final_context_entries,
                                directory=True,
                            ):
                                raise ImageStateError("context-identity-changed")
                        finally:
                            self._close(state_handle)
                    finally:
                        self._close(context_handle)
                finally:
                    self._close(context_lock)
                final_root_entries = self._enumerate_child_entries(
                    root_handle,
                    self.image_contexts_root,
                    max_entries=MAX_CONTEXTS,
                    error_code="context-enumeration-failed",
                )
                if not self._same_enumerated_entries(
                    entries,
                    final_root_entries,
                    directory=True,
                ):
                    raise ImageStateError("image-contexts-root-identity-changed")
            if not names:
                final_root_entries = self._enumerate_child_entries(
                    root_handle,
                    self.image_contexts_root,
                    max_entries=MAX_CONTEXTS,
                    error_code="context-enumeration-failed",
                )
                if not self._same_enumerated_entries(
                    entries,
                    final_root_entries,
                    directory=True,
                ):
                    raise ImageStateError("image-contexts-root-identity-changed")
            return states
        finally:
            self._close(root_handle)

    def _link_no_replace(self, source: Any, parent: Any, name: str) -> None:
        try:
            link = getattr(self.boundary, "link_no_replace", None)
            if callable(link):
                link(source, parent, name)
                return
            if os.name == "nt":
                link = getattr(self.boundary, "win_link_handle_no_replace", None)
                if callable(link):
                    link(source, parent, name)
                    return
            else:
                link = getattr(self.boundary, "posix_link_no_replace", None)
                if callable(link):
                    source_fd = getattr(source, "raw", source)
                    parent_fd = getattr(parent, "raw", parent)
                    link(source_fd, parent_fd, name)
                    return
            raise ImageStateError("path-boundary-link-missing")
        except FileExistsError:
            raise
        except ImageStateError:
            raise
        except Exception as exc:
            raise ImageStateError("state-publication-failed") from exc

    def _fsync_parent(self, path: Path) -> None:
        flush = getattr(self.boundary, "fsync_parent", None)
        if not callable(flush):
            flush = getattr(self.boundary, "_fsync_parent", None)
        if not callable(flush):
            raise ImageStateError("path-boundary-fsync-missing")
        try:
            flush(path)
        except Exception as exc:
            raise ImageStateError("state-durability-failed") from exc

    def _cleanup_exact(
        self,
        path: Path,
        identity: Any,
        digest: str | None,
        *,
        parent_path: Path | None = None,
        parent_identity: Any = None,
        max_bytes: int = MAX_STATE_BYTES,
    ) -> None:
        if parent_path is not None:
            try:
                parent_handle, current_parent_identity = self._open(parent_path, directory=True)
            except ImageStateError as exc:
                raise ImageStateError("state-parent-identity-changed") from exc
            try:
                if parent_identity is None or not self._same_identity(parent_identity, current_parent_identity):
                    raise ImageStateError("state-parent-identity-changed")
            finally:
                self._close(parent_handle)
        cleanup = getattr(self.boundary, "unlink_handle_bound", None)
        if not callable(cleanup):
            raise ImageStateError("uncertain-cleanup")
        try:
            removed = cleanup(
                path,
                identity,
                parent_expected=parent_identity,
                expected_sha256=digest,
                max_bytes=max_bytes,
            )
        except Exception as exc:
            raise ImageStateError("uncertain-cleanup") from exc
        if removed is not True:
            raise ImageStateError("uncertain-cleanup")

    def _write_stage(
        self,
        state_dir: Path,
        operation_id: str,
        raw: bytes,
        parent_identity: Any,
    ) -> tuple[Path, Any, Any]:
        stage = state_dir / f".image-state-stage-{operation_id}.json"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        fd: int | None = None
        stage_handle = None
        stage_identity = None
        created = False
        try:
            try:
                fd = os.open(stage, flags, 0o600)
            except FileExistsError as exc:
                raise ImageStateError("state-stage-collision") from exc
            except Exception as exc:
                raise ImageStateError("state-stage-failed") from exc
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
            os.close(fd)
            fd = None
            stage_handle, stage_identity = self._open(stage, directory=False)
            if self._read(stage_handle) != raw:
                raise ImageStateError("state-stage-mismatch")
            return stage, stage_handle, stage_identity
        except BaseException as exc:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._close(stage_handle)
            if created:
                cleanup_identity = stage_identity
                if cleanup_identity is None:
                    try:
                        cleanup_handle, cleanup_identity = self._open(stage, directory=False)
                    except Exception as cleanup_exc:
                        raise ImageStateError("uncertain-cleanup") from exc
                    finally:
                        if "cleanup_handle" in locals():
                            self._close(cleanup_handle)
                try:
                    self._cleanup_exact(
                        stage,
                        cleanup_identity,
                        hashlib.sha256(raw).hexdigest(),
                        parent_path=state_dir,
                        parent_identity=parent_identity,
                        max_bytes=max(MAX_STATE_BYTES, len(raw)),
                    )
                except ImageStateError as cleanup_exc:
                    raise cleanup_exc from exc
            raise

    def _publish_locked(
        self,
        context_id: str,
        state_dir: Path,
        candidate: dict[str, Any],
        raw: bytes,
        operation_id: str,
    ) -> dict[str, Any]:
        name = f"r{candidate['revision']:020d}-{operation_id}.json"
        target = state_dir / name
        pinned_parent_handle, pinned_parent_identity = self._open(state_dir, directory=True)
        self._close(pinned_parent_handle)
        stage, stage_handle, stage_identity = self._write_stage(
            state_dir,
            operation_id,
            raw,
            pinned_parent_identity,
        )
        parent_handle = None
        try:
            parent_handle, parent_identity = self._open(state_dir, directory=True)
            if not self._same_identity(pinned_parent_identity, parent_identity):
                raise ImageStateError("state-parent-identity-changed")
            source_check_handle, source_check_identity = self._open(stage, directory=False)
            try:
                if not self._same_identity(stage_identity, source_check_identity):
                    raise ImageStateError("state-source-identity-changed")
            finally:
                self._close(source_check_handle)
            try:
                self._link_no_replace(stage_handle, parent_handle, name)
            except FileExistsError:
                pass
            current_parent_handle, current_parent_identity = self._open(state_dir, directory=True)
            try:
                if not self._same_identity(parent_identity, current_parent_identity):
                    raise ImageStateError("state-parent-identity-changed")
            finally:
                self._close(current_parent_handle)
            self._fsync_parent(state_dir)
        except BaseException as exc:
            self._close(parent_handle)
            self._close(stage_handle)
            try:
                self._cleanup_exact(
                    stage,
                    stage_identity,
                    hashlib.sha256(raw).hexdigest(),
                    parent_path=state_dir,
                    parent_identity=pinned_parent_identity,
                    max_bytes=max(MAX_STATE_BYTES, len(raw)),
                )
            except ImageStateError as cleanup_exc:
                raise cleanup_exc from exc
            raise
        finally:
            self._close(parent_handle)
            self._close(stage_handle)
        try:
            target_handle, target_identity = self._open(target, directory=False)
            try:
                target_raw = self._read(target_handle)
            finally:
                self._close(target_handle)
        except BaseException as exc:
            try:
                self._cleanup_exact(
                    stage,
                    stage_identity,
                    hashlib.sha256(raw).hexdigest(),
                    parent_path=state_dir,
                    parent_identity=pinned_parent_identity,
                    max_bytes=max(MAX_STATE_BYTES, len(raw)),
                )
            except ImageStateError as cleanup_exc:
                raise cleanup_exc from exc
            raise
        if not self._same_identity(stage_identity, target_identity) or target_raw != raw:
            try:
                self._cleanup_exact(
                    stage,
                    stage_identity,
                    hashlib.sha256(raw).hexdigest(),
                    parent_path=state_dir,
                    parent_identity=pinned_parent_identity,
                    max_bytes=max(MAX_STATE_BYTES, len(raw)),
                )
            except ImageStateError as cleanup_exc:
                raise cleanup_exc from ImageStateError("state-publication-collision")
            raise ImageStateError("state-publication-collision")
        try:
            self._fsync_parent(state_dir)
            self._cleanup_exact(
                stage,
                stage_identity,
                hashlib.sha256(raw).hexdigest(),
                parent_path=state_dir,
                parent_identity=pinned_parent_identity,
                max_bytes=max(MAX_STATE_BYTES, len(raw)),
            )
        except ImageStateError as exc:
            if exc.code == "uncertain-cleanup":
                raise
            try:
                self._cleanup_exact(
                    stage,
                    stage_identity,
                    hashlib.sha256(raw).hexdigest(),
                    parent_path=state_dir,
                    parent_identity=pinned_parent_identity,
                    max_bytes=max(MAX_STATE_BYTES, len(raw)),
                )
            except ImageStateError as cleanup_exc:
                raise cleanup_exc from exc
            raise
        return copy.deepcopy(candidate)

    def ensure_working_context(self, operation_id: str) -> dict[str, Any]:
        """Open (or find) the context a user works in before naming anything.

        The image area is for working, not for filling in a form: a first
        generation must not require the user to name and save something first.
        So the first run publishes one working context holding a single unnamed
        work with an empty draft, and every later run reuses it - a session of
        experiments belongs in one place, not one context per prompt.

        ``operation_id`` makes the publication idempotent, so a retried submit
        cannot strand a second context.
        """
        operation_id = self._validate_operation_id(operation_id)
        for state in self.enumerate_contexts():
            works = (state.get("state") or {}).get("works") or []
            if works:
                return {
                    "context_id": state["context_id"],
                    "work_id": works[0]["work_id"],
                    "draft_generation": (works[0].get("draft") or {}).get("draft_generation", 0),
                }
        return self.create_working_context(operation_id)

    def create_working_context(self, operation_id: str, context_id: str | None = None) -> dict[str, Any]:
        """Publish a new working context: one unnamed work with an empty draft.

        ``context_id`` lets an owner index claim the id BEFORE anything is
        published, so a crash between the two can never leave an unowned
        context that some other scope would then see.
        """
        operation_id = self._validate_operation_id(operation_id)
        context_id = (
            "imgctx_" + secrets.token_hex(16)
            if context_id is None
            else self._validate_context_id(context_id)
        )
        work_id = "imgwork_" + secrets.token_hex(16)
        now = self._utc_second()
        published = self.publish_initial(
            {
                "schema": "studio.image-state/2",
                "context_id": context_id,
                "revision": 1,
                "updated_at": now,
                "receipt_key_fingerprint": self.authority.receipt_key_fingerprint,
                "receipt_key_enrollment_id": self.authority.receipt_key_enrollment_id,
                "state": {
                    "works": [{
                        "work_id": work_id,
                        "name": "Untitled",
                        "draft": {
                            "intent": "",
                            "intent_sha256": hashlib.sha256(b"").hexdigest(),
                            "notes": "",
                            "references": [],
                            "alternatives": [],
                            "based_on_revision_id": None,
                            "draft_generation": 0,
                        },
                        "saved_revisions": [],
                    }],
                    "materials": [],
                    "receipts": [],
                    "receipt_floor_at": now,
                },
            },
            operation_id,
        )
        work = published["state"]["works"][0]
        return {
            "context_id": published["context_id"],
            "work_id": work["work_id"],
            "draft_generation": work["draft"]["draft_generation"],
        }

    def keep_draft_as_revision(
        self,
        context_id: str,
        work_id: str,
        name: str,
        operation_id: str,
    ) -> dict[str, Any]:
        """Turn the current draft into a named saved revision.

        This is the step that makes experimentation durable, and it happens
        after the work, never before it: the user generates, looks, and only
        then decides something is worth keeping.  ``operation_id`` (the job id)
        makes it idempotent, so a retried keep cannot publish twice.
        """
        context_id = self._validate_context_id(context_id)
        work_id = self._validate_work_id(work_id)
        if type(operation_id) is not str or not operation_id:
            raise ImageStateError("operation-id-invalid")
        # The caller identifies the run by its job id; the ledger wants an
        # operation id.  Derive one deterministically so a retried keep maps to
        # the same publication and cannot produce a second revision.
        source_operation = operation_id
        publication_operation = self._validate_operation_id(
            "imgop_" + hashlib.sha256(("keep:" + operation_id).encode("utf-8")).hexdigest()[:32]
        )
        if type(name) is not str or not name.strip() or len(name) > 200:
            raise ImageStateError("work-name-invalid")
        current = self.load(context_id)
        candidate = copy.deepcopy(current)
        works = candidate["state"]["works"]
        matching = [work for work in works if work["work_id"] == work_id]
        if len(matching) != 1:
            raise ImageStateError("work-missing")
        work = matching[0]
        # The revision id is derived from the job too, so a retried keep
        # rebuilds the identical candidate and publish_next recognises it as
        # the same publication instead of appending a second revision.
        revision_id = "imgrev_" + hashlib.sha256(
            ("revision:" + source_operation).encode("utf-8")
        ).hexdigest()[:32]
        for revision in work["saved_revisions"]:
            if revision["revision_id"] == revision_id:
                return {
                    "context_id": context_id,
                    "work_id": work_id,
                    "revision_id": revision_id,
                    "name": work["name"],
                }
        draft = copy.deepcopy(work["draft"])
        parent = draft.get("based_on_revision_id")
        content = {key: draft[key] for key in ("intent", "intent_sha256", "notes", "references", "alternatives")}
        work["name"] = name.strip()
        work["saved_revisions"].append({
            "revision_id": revision_id,
            "work_id": work_id,
            "parent_revision_id": parent,
            "saved_at": self._utc_second(),
            "content": content,
            "selected_artifact": None,
            "material_ids": [],
        })
        work["draft"] = {
            **copy.deepcopy(content),
            "based_on_revision_id": revision_id,
            "draft_generation": int(draft.get("draft_generation") or 0) + 1,
        }
        candidate["revision"] = int(current["revision"]) + 1
        candidate["updated_at"] = self._utc_second()
        self.publish_next(context_id, int(current["revision"]), candidate, publication_operation)
        return {
            "context_id": context_id,
            "work_id": work_id,
            "revision_id": revision_id,
            "name": work["name"],
        }

    def keep_generated(
        self,
        context_id: str,
        basis: dict[str, Any],
        name: str,
        artifact: dict[str, Any],
        intent: str,
    ) -> dict[str, Any]:
        """Keep a finished run WITH its picture, under the name the user gave it.

        A run on the unnamed working draft becomes its own named work, so two
        things kept one after another keep two names - naming one never renames
        the other.  A run that continued from a saved revision becomes the next
        revision of that same work.  The picture is bound as the revision's
        selected artifact, so the revision is a picture, not just a label.

        Idempotent per run: the revision and work ids derive from the job, so a
        retried keep returns the first result instead of publishing twice.
        """
        context_id = self._validate_context_id(context_id)
        try:
            artifact = contract.validate_artifact_ref(copy.deepcopy(artifact))
        except Exception as exc:
            raise ImageStateError("artifact-invalid") from exc
        if type(name) is not str or not name.strip() or len(name.strip()) > 160:
            raise ImageStateError("work-name-invalid")
        name = name.strip()
        if type(intent) is not str or len(intent) > 16_384:
            raise ImageStateError("intent-invalid")
        if type(basis) is not dict:
            raise ImageStateError("basis-invalid")
        basis_kind = self._validate_basis_kind(basis.get("basis_kind"))
        source_work_id = self._validate_work_id(basis.get("work_id"))
        job_id = artifact["producer_job_id"]
        digest = lambda label: hashlib.sha256((label + ":" + job_id).encode("utf-8")).hexdigest()[:32]
        revision_id = "imgrev_" + digest("kept-revision")
        publication_operation = self._validate_operation_id("imgop_" + digest("kept-publication"))

        current = self.load(context_id)
        for work in current["state"]["works"]:
            for revision in work["saved_revisions"]:
                if revision["revision_id"] == revision_id:
                    return {"context_id": context_id, "work_id": work["work_id"],
                            "revision_id": revision_id, "name": work["name"], "replayed": True}

        candidate = copy.deepcopy(current)
        works = candidate["state"]["works"]
        if basis_kind == "saved_revision":
            parent = self._validate_revision_id(basis.get("revision_id"))
            matching = [work for work in works if work["work_id"] == source_work_id]
            if len(matching) != 1 or not any(r["revision_id"] == parent for r in matching[0]["saved_revisions"]):
                raise ImageStateError("work-missing")
            work = matching[0]
            work["name"] = name
        else:
            if not any(work["work_id"] == source_work_id for work in works):
                raise ImageStateError("work-missing")
            parent = None
            work = {
                "work_id": "imgwork_" + digest("kept-work"),
                "name": name,
                "draft": None,
                "saved_revisions": [],
            }
            works.append(work)
        intent_sha = hashlib.sha256(intent.encode("utf-8")).hexdigest()
        content = {
            "intent": intent,
            "intent_sha256": intent_sha,
            "notes": "",
            "references": [],
            "alternatives": [copy.deepcopy(artifact)],
        }
        work["saved_revisions"].append({
            "revision_id": revision_id,
            "work_id": work["work_id"],
            "parent_revision_id": parent,
            "saved_at": self._utc_second(),
            "content": content,
            "selected_artifact": artifact["material_id"],
            "material_ids": [artifact["material_id"]],
        })
        # The work's draft follows its newest kept revision, so continuing
        # from it starts where the user left off.
        work["draft"] = {
            "intent": intent,
            "intent_sha256": intent_sha,
            "notes": "",
            "references": [],
            "alternatives": [],
            "based_on_revision_id": revision_id,
            "draft_generation": int(((work.get("draft") or {}).get("draft_generation")) or 0) + (1 if work.get("draft") else 0),
        }
        material = {
            "material_id": artifact["material_id"],
            "sha256": artifact["artifact_sha256"],
            "media_type": "image",
            "label": "generated.png",
            "availability": "available",
        }
        materials = candidate["state"]["materials"]
        same = [item for item in materials if item["material_id"] == material["material_id"]]
        if same and same != [material]:
            raise ImageStateError("material-conflict")
        if not same:
            materials.append(material)
        candidate["revision"] = int(current["revision"]) + 1
        candidate["updated_at"] = self._utc_second()
        self.publish_next(context_id, int(current["revision"]), candidate, publication_operation)
        return {"context_id": context_id, "work_id": work["work_id"],
                "revision_id": revision_id, "name": name, "replayed": False}

    def publish_initial(self, candidate: dict[str, Any], operation_id: str) -> dict[str, Any]:
        context_id = self._validate_context_id(candidate.get("context_id") if isinstance(candidate, dict) else None)
        operation_id = self._validate_operation_id(operation_id)
        normalized, raw = self._validate_candidate(candidate, context_id, revision=1)
        context_dir, state_dir, context_identity, state_identity = self._context_dirs(context_id, create=True)
        lock = self._lock(context_dir)
        try:
            self._revalidate_context(context_dir, context_identity)
            try:
                records = self._load_records_locked(context_id, state_dir, state_identity)
            except ImageStateError as exc:
                if exc.code != "state-missing":
                    raise
                records = []
            target_name = f"r{1:020d}-{operation_id}.json"
            if records:
                for name, state, prior_raw in records:
                    if name == target_name:
                        if prior_raw == raw:
                            self._revalidate_context(context_dir, context_identity)
                            return copy.deepcopy(state)
                        raise ImageStateError("state-operation-conflict")
                    if name.endswith(f"-{operation_id}.json"):
                        raise ImageStateError("state-operation-conflict")
                raise ImageStateError("state-initial-conflict")
            result = self._publish_locked(context_id, state_dir, normalized, raw, operation_id)
            self._revalidate_context(context_dir, context_identity)
            return result
        finally:
            self._close(lock)

    def publish_next(
        self,
        context_id: str,
        expected_revision: int,
        candidate: dict[str, Any],
        operation_id: str,
    ) -> dict[str, Any]:
        context_id = self._validate_context_id(context_id)
        if type(expected_revision) is not int or isinstance(expected_revision, bool) or expected_revision <= 0:
            raise ImageStateError("expected-revision-invalid")
        operation_id = self._validate_operation_id(operation_id)
        normalized, raw = self._validate_candidate(candidate, context_id)
        context_dir, state_dir, context_identity, state_identity = self._context_dirs(context_id, create=False)
        lock = self._lock(context_dir)
        try:
            self._revalidate_context(context_dir, context_identity)
            records = self._load_records_locked(context_id, state_dir, state_identity)
            target_name = f"r{normalized['revision']:020d}-{operation_id}.json"
            current = records[-1][1]
            for name, state, prior_raw in records:
                if name == target_name:
                    if prior_raw == raw:
                        self._revalidate_context(context_dir, context_identity)
                        return copy.deepcopy(state)
                    raise ImageStateError("state-operation-conflict")
                if name.endswith(f"-{operation_id}.json") and prior_raw != raw:
                    raise ImageStateError("state-operation-conflict")
            if current["revision"] != expected_revision:
                raise ImageStateError("state-revision-conflict")
            if normalized["revision"] != expected_revision + 1:
                raise ImageStateError("state-revision-sequence")
            result = self._publish_locked(context_id, state_dir, normalized, raw, operation_id)
            self._revalidate_context(context_dir, context_identity)
            return result
        finally:
            self._close(lock)

    @staticmethod
    def _validate_basis_kind(value: Any) -> str:
        if type(value) is not str or value not in ("draft", "saved_revision"):
            raise ImageStateError("basis-kind-invalid")
        return value

    def _claim_fields(
        self,
        state: dict[str, Any],
        context_id: str,
        work_id: str,
        basis_kind: str,
        revision_id: str | None,
    ) -> dict[str, Any]:
        works = [work for work in state["state"]["works"] if work["work_id"] == work_id]
        if len(works) != 1:
            raise ImageStateError("work-missing-or-ambiguous")
        work = works[0]
        if basis_kind == "saved_revision":
            if revision_id is None:
                raise ImageStateError("revision-required")
            revisions = [item for item in work["saved_revisions"] if item["revision_id"] == revision_id]
            if len(revisions) != 1:
                raise ImageStateError("revision-missing-or-ambiguous")
            if revisions[0]["work_id"] != work_id:
                raise ImageStateError("revision-work-mismatch")
            content = revisions[0]["content"]
            draft_generation = None
        else:
            if revision_id is not None:
                raise ImageStateError("draft-revision-alternative")
            content = work["draft"]
            draft_generation = content["draft_generation"]
        fields = {
            "schema": IMAGE_OWNER_CLAIM_SCHEMA,
            "context_id": context_id,
            "work_id": work_id,
            "basis_kind": basis_kind,
            "revision_id": revision_id,
            "draft_generation": draft_generation,
            "basis_content_sha256": contract.sha256(contract.canonical_bytes(content)),
            "expected_revision": state["revision"],
        }
        try:
            if basis_kind == "saved_revision":
                contract.validate_basis({
                    "context_id": context_id,
                    "work_id": work_id,
                    "basis_kind": basis_kind,
                    "revision_id": revision_id,
                    "draft_generation": None,
                    "basis_content_sha256": fields["basis_content_sha256"],
                })
            else:
                contract.validate_basis({
                    "context_id": context_id,
                    "work_id": work_id,
                    "basis_kind": basis_kind,
                    "revision_id": None,
                    "draft_generation": draft_generation,
                    "basis_content_sha256": fields["basis_content_sha256"],
                })
        except Exception as exc:
            raise ImageStateError("owner-basis-invalid") from exc
        return fields

    def _validated_claim(self, claim: Any) -> dict[str, Any]:
        if type(claim) is not dict or set(claim) != _CLAIM_FIELDS:
            raise ImageStateError("owner-claim-shape")
        if type(claim["schema"]) is not str or claim["schema"] != IMAGE_OWNER_CLAIM_SCHEMA:
            raise ImageStateError("owner-claim-schema")
        self._validate_context_id(claim["context_id"])
        self._validate_work_id(claim["work_id"])
        self._validate_basis_kind(claim["basis_kind"])
        if claim["revision_id"] is not None:
            self._validate_revision_id(claim["revision_id"])
        if claim["basis_kind"] == "saved_revision" and claim["revision_id"] is None:
            raise ImageStateError("revision-required")
        if claim["basis_kind"] == "draft" and claim["revision_id"] is not None:
            raise ImageStateError("draft-revision-alternative")
        if claim["draft_generation"] is not None and (
            type(claim["draft_generation"]) is not int
            or isinstance(claim["draft_generation"], bool)
            or claim["draft_generation"] < 0
            or claim["draft_generation"] > 9_007_199_254_740_991
        ):
            raise ImageStateError("draft-generation-invalid")
        if claim["basis_kind"] == "saved_revision" and claim["draft_generation"] is not None:
            raise ImageStateError("draft-generation-alternative")
        if claim["basis_kind"] == "draft" and claim["draft_generation"] is None:
            raise ImageStateError("draft-generation-required")
        if type(claim["basis_content_sha256"]) is not str or _HEX64_RE.fullmatch(claim["basis_content_sha256"]) is None:
            raise ImageStateError("basis-content-sha256-invalid")
        if (
            type(claim["expected_revision"]) is not int
            or isinstance(claim["expected_revision"], bool)
            or claim["expected_revision"] <= 0
            or claim["expected_revision"] > 9_007_199_254_740_991
        ):
            raise ImageStateError("expected-revision-invalid")
        try:
            contract.validate_basis({
                "context_id": claim["context_id"],
                "work_id": claim["work_id"],
                "basis_kind": claim["basis_kind"],
                "revision_id": claim["revision_id"],
                "draft_generation": claim["draft_generation"],
                "basis_content_sha256": claim["basis_content_sha256"],
            })
        except Exception as exc:
            raise ImageStateError("owner-basis-invalid") from exc
        return copy.deepcopy(claim)

    def _owner_claim_locked(
        self,
        context_id: str,
        state_dir: Path,
        work_id: str,
        basis_kind: str,
        revision_id: str | None,
        *,
        issue_token: bool,
        expected_state_identity: Any = None,
    ) -> dict[str, Any]:
        state = self._load_locked(context_id, state_dir, expected_state_identity)
        fields = self._claim_fields(state, context_id, work_id, basis_kind, revision_id)
        fields = self._validated_claim(fields)
        if not issue_token:
            return fields
        try:
            canonical = contract.canonical_bytes(fields)
            token = self.authority.sign_owner_claim(canonical)
        except Exception as exc:
            raise ImageStateError("owner-token-invalid") from exc
        if type(token) is not str or _OWNER_TOKEN_RE.fullmatch(token) is None:
            raise ImageStateError("owner-token-invalid")
        return {**fields, "owner_token": token}

    def load_current_owner(
        self,
        context_id: str,
        work_id: str,
        basis_kind: str,
        revision_id: str | None = None,
    ) -> dict[str, Any]:
        context_id = self._validate_context_id(context_id)
        work_id = self._validate_work_id(work_id)
        basis_kind = self._validate_basis_kind(basis_kind)
        if revision_id is not None:
            self._validate_revision_id(revision_id)
        context_dir, state_dir, context_identity, state_identity = self._context_dirs(context_id, create=False)
        lock = self._lock(context_dir)
        try:
            self._revalidate_context(context_dir, context_identity)
            result = self._owner_claim_locked(
                context_id,
                state_dir,
                work_id,
                basis_kind,
                revision_id,
                issue_token=True,
                expected_state_identity=state_identity,
            )
            self._revalidate_context(context_dir, context_identity)
            return result
        finally:
            self._close(lock)

    def verify_current_owner(self, claim: dict[str, Any], owner_token: str) -> dict[str, Any]:
        if type(owner_token) is not str or _OWNER_TOKEN_RE.fullmatch(owner_token) is None:
            raise ImageStateError("owner-token-invalid")
        if type(claim) is dict and "owner_token" in claim:
            embedded_token = claim["owner_token"]
            if (
                set(claim) != _CLAIM_FIELDS | {"owner_token"}
                or type(embedded_token) is not str
                or embedded_token != owner_token
            ):
                raise ImageStateError("owner-claim-shape")
            claim = {key: value for key, value in claim.items() if key != "owner_token"}
        fields = self._validated_claim(claim)
        try:
            canonical = contract.canonical_bytes(fields)
        except Exception as exc:
            raise ImageStateError("owner-claim-invalid") from exc
        try:
            verified = self.authority.verify_owner_claim(canonical, owner_token)
        except Exception as exc:
            raise ImageStateError("owner-token-invalid") from exc
        if verified is not True:
            raise ImageStateError("owner-token-invalid")
        context_dir, state_dir, context_identity, state_identity = self._context_dirs(fields["context_id"], create=False)
        lock = self._lock(context_dir)
        try:
            self._revalidate_context(context_dir, context_identity)
            current = self._owner_claim_locked(
                fields["context_id"],
                state_dir,
                fields["work_id"],
                fields["basis_kind"],
                fields["revision_id"],
                issue_token=False,
                expected_state_identity=state_identity,
            )
            try:
                current_canonical = contract.canonical_bytes(current)
            except Exception as exc:
                raise ImageStateError("owner-claim-invalid") from exc
            if current_canonical != canonical:
                raise ImageStateError("owner-claim-stale")
            self._revalidate_context(context_dir, context_identity)
            return {**fields, "owner_token": owner_token}
        finally:
            self._close(lock)


__all__ = [
    "IMAGE_OWNER_CLAIM_SCHEMA",
    "MAX_STATE_BYTES",
    "MAX_STATE_FILES",
    "ImageOwnerAuthority",
    "HmacImageOwnerAuthority",
    "ImageStateError",
    "ImageStoreError",
    "ImageStateStore",
]
