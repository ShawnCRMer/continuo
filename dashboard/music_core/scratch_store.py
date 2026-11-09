"""Private No-Project Interpretation scratch storage.

The store owns only the scratch namespace.  Filesystem and locking primitives
are supplied by the composition root so this neutral package does not depend on
``studio_core``.
"""

from __future__ import annotations

import contextlib
import datetime as _datetime
import hashlib
import json
import os
import re
import secrets
import stat
from pathlib import Path
from typing import Any, Callable, Iterator

from .contracts import (
    MusicContractError,
    build_interpretation,
    canonical_json_bytes,
    validate_interpretation,
    validate_interpretation_body,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INTERPRETATION_MAX_BYTES = 4 * 1024 * 1024


class ScratchStoreError(MusicContractError):
    """Stable, path-free errors raised by the scratch store."""

    _MESSAGES = {
        "invalid_track": "invalid scratch track id",
        "invalid_revision": "invalid scratch revision",
        "invalid_body": "invalid scratch interpretation body",
        "scratch_unavailable": "scratch store unavailable",
        "scratch_corrupt": "scratch interpretation is corrupt",
        "scratch_conflict": "scratch interpretation revision conflict",
        "scratch_write_failed": "scratch interpretation write failed",
        "private_node_invalid": "scratch store node is invalid",
    }

    def __init__(self, code: str) -> None:
        self.code = code if code in self._MESSAGES else "scratch_unavailable"
        self.public_message = self._MESSAGES[self.code]
        # Never include OS/provider exception text: it can contain paths.
        super().__init__(self.public_message)


class ScratchStore:
    """Profile-private CAS storage for an artist-authored Interpretation body."""

    def __init__(
        self,
        root: os.PathLike[str] | str,
        boundary: Any,
        lock_factory: Callable[[Path], Any] | None = None,
    ) -> None:
        try:
            self._root = Path(root)
        except (TypeError, ValueError, OSError):
            raise ScratchStoreError("scratch_unavailable") from None
        if boundary is None:
            raise ScratchStoreError("scratch_unavailable")
        self._boundary = boundary
        factory = lock_factory or getattr(boundary, "project_lock", None)
        if not callable(factory):
            raise ScratchStoreError("scratch_unavailable")
        self._lock_factory = factory

    @property
    def root(self) -> None:
        """The store never exposes its filesystem locator."""

        return None

    def read(self, track_sha256: str) -> dict[str, Any]:
        track = self._validate_track(track_sha256)
        if not self._lexists(self._root):
            return self._empty_result()
        try:
            self._validate_directory(self._root)
            scratch = self._scratch_directory()
            if not self._lexists(scratch):
                return self._empty_result()
            self._validate_directory(scratch)
            target = self._target(track)
            if not self._lexists(target):
                return self._empty_result()
            self._ensure_private_existing(target, directory=False)
            interpretation = self._read_interpretation(target, track)
            return self._result(interpretation)
        except ScratchStoreError:
            raise
        except FileNotFoundError:
            return self._empty_result()
        except Exception:
            raise ScratchStoreError("scratch_corrupt") from None

    def write(
        self,
        track_sha256: str,
        expected_revision: int,
        value: Any,
    ) -> dict[str, Any]:
        track = self._validate_track(track_sha256)
        expected = self._validate_revision(expected_revision)
        try:
            body = validate_interpretation_body(value)
        except Exception:
            raise ScratchStoreError("invalid_body") from None

        self._prepare_layout()
        try:
            with self._lock_scope(self._scratch_directory()):
                target = self._target(track)
                current = self._read_current_for_write(target, track)
                current_revision = 0 if current is None else current["revision"]
                if expected != current_revision:
                    raise ScratchStoreError("scratch_conflict")
                interpretation = build_interpretation(
                    track_sha256=track,
                    scope={"kind": "scratch"},
                    revision=current_revision + 1,
                    updated_at=_now_timestamp(),
                    value=body,
                )
                interpretation = validate_interpretation(interpretation)
                payload = canonical_json_bytes(interpretation)
                if len(payload) > _INTERPRETATION_MAX_BYTES:
                    raise ScratchStoreError("invalid_body")
                self._publish(target, payload, prior=current)
                return self._result(interpretation)
        except ScratchStoreError:
            raise
        except Exception:
            raise ScratchStoreError("scratch_write_failed") from None

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {"exists": False, "revision": 0, "interpretation": None}

    @staticmethod
    def _result(interpretation: dict[str, Any]) -> dict[str, Any]:
        return {
            "exists": True,
            "revision": interpretation["revision"],
            "interpretation": interpretation,
        }

    @staticmethod
    def _validate_track(value: Any) -> str:
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise ScratchStoreError("invalid_track")
        return value

    @staticmethod
    def _validate_revision(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ScratchStoreError("invalid_revision")
        return value

    def _scratch_directory(self) -> Path:
        return self._root / "scratch"

    def _target(self, track: str) -> Path:
        return self._scratch_directory() / f"{track}.json"

    def _prepare_layout(self) -> None:
        try:
            self._ensure_private_node(self._root, directory=True)
            self._ensure_private_node(self._scratch_directory(), directory=True)
            self._validate_directory(self._root)
            self._validate_directory(self._scratch_directory())
        except ScratchStoreError:
            raise
        except Exception:
            raise ScratchStoreError("scratch_unavailable") from None

    def _validate_directory(self, path: Path) -> None:
        self._ensure_private_existing(path, directory=True)

    def _ensure_private_node(self, path: Path, *, directory: bool) -> None:
        fn = getattr(self._boundary, "ensure_private_node", None)
        if not callable(fn):
            fn = getattr(self._boundary, "ensure_private", None)
        if not callable(fn):
            raise ScratchStoreError("scratch_unavailable")
        try:
            fn(path, directory=directory)
        except Exception:
            raise ScratchStoreError("scratch_unavailable") from None

    def _ensure_private_existing(self, path: Path, *, directory: bool) -> None:
        self._reject_reparse_components(path)
        try:
            info = path.lstat()
        except FileNotFoundError:
            raise ScratchStoreError("private_node_invalid") from None
        except Exception:
            raise ScratchStoreError("private_node_invalid") from None
        if directory:
            valid_type = stat.S_ISDIR(info.st_mode)
            expected_mode = 0o700
        else:
            valid_type = stat.S_ISREG(info.st_mode)
            expected_mode = 0o600
        if not valid_type:
            raise ScratchStoreError("private_node_invalid")
        if os.name != "nt" and stat.S_IMODE(info.st_mode) != expected_mode:
            raise ScratchStoreError("private_node_invalid")
        verify = getattr(self._boundary, "require_private_node", None)
        if os.name == "nt":
            if not callable(verify):
                raise ScratchStoreError("private_node_invalid")
            try:
                verify(path, directory=directory)
            except Exception:
                raise ScratchStoreError("private_node_invalid") from None

    def _reject_reparse_components(self, path: Path) -> None:
        checker = getattr(self._boundary, "reject_reparse_components", None)
        if not callable(checker):
            checker = getattr(self._boundary, "_reject_reparse_components", None)
        if callable(checker):
            try:
                checker(path)
            except Exception:
                raise ScratchStoreError("private_node_invalid") from None
            return
        anchor = Path(path.anchor) if path.anchor else Path()
        current = anchor
        parts = path.parts[len(anchor.parts):] if anchor.parts else path.parts
        try:
            for part in parts:
                current = current / part
                if current.is_symlink():
                    raise ScratchStoreError("private_node_invalid")
        except ScratchStoreError:
            raise
        except Exception:
            raise ScratchStoreError("private_node_invalid") from None

    def _read_current_for_write(self, target: Path, track: str) -> dict[str, Any] | None:
        if not self._lexists(target):
            return None
        self._ensure_private_existing(target, directory=False)
        return self._read_interpretation(target, track)

    def _read_interpretation(self, target: Path, track: str) -> dict[str, Any]:
        raw = self._read_raw(target)
        try:
            value = json.loads(raw.decode("utf-8"))
            interpretation = validate_interpretation(value)
        except Exception:
            raise ScratchStoreError("scratch_corrupt") from None
        if (
            interpretation["track_sha256"] != track
            or interpretation["scope"] != {"kind": "scratch"}
            or canonical_json_bytes(interpretation) != raw
        ):
            raise ScratchStoreError("scratch_corrupt")
        return interpretation

    def _read_raw(self, path: Path) -> bytes:
        try:
            with path.open("rb") as handle:
                raw = handle.read(_INTERPRETATION_MAX_BYTES + 1)
        except FileNotFoundError:
            raise ScratchStoreError("scratch_corrupt") from None
        except Exception:
            raise ScratchStoreError("scratch_corrupt") from None
        if len(raw) > _INTERPRETATION_MAX_BYTES:
            raise ScratchStoreError("scratch_corrupt")
        return raw

    def _publish(
        self,
        target: Path,
        payload: bytes,
        *,
        prior: dict[str, Any] | None,
    ) -> None:
        directory = target.parent
        temporary = directory / f".studio-scratch-{secrets.token_hex(16)}.tmp"
        backup = directory / f".studio-scratch-backup-{secrets.token_hex(16)}.tmp"
        staged = False
        target_removed = False
        backup_created = False
        new_published = False
        try:
            self._write_temp(temporary, payload)
            staged = True
            self._flush_parent(directory)
            if prior is None:
                self._link_no_replace(temporary, directory, target.name)
                new_published = True
            else:
                self._ensure_private_existing(target, directory=False)
                self._link_no_replace(target, directory, backup.name)
                backup_created = True
                self._unlink_file(target, expected=prior)
                target_removed = True
                self._link_no_replace(temporary, directory, target.name)
                new_published = True
            self._flush_parent(directory)
        except ScratchStoreError:
            if new_published and prior is not None:
                self._restore_prior(target, backup, payload)
            elif new_published and prior is None:
                self._unlink_file_best_effort(target, expected_bytes=payload)
            raise
        except Exception:
            if new_published and prior is not None:
                self._restore_prior(target, backup, payload)
            elif new_published and prior is None:
                self._unlink_file_best_effort(target, expected_bytes=payload)
            raise ScratchStoreError("scratch_write_failed") from None
        finally:
            restored = False
            if target_removed and not new_published:
                restored = self._restore_prior(target, backup, None)
            if backup_created and prior is not None and (not target_removed or restored or self._lexists(target)):
                self._unlink_file_best_effort(backup, expected_bytes=canonical_json_bytes(prior))
            if staged:
                self._unlink_file_best_effort(temporary, expected_bytes=payload)

    def _restore_prior(self, target: Path, backup: Path, current_payload: bytes | None) -> bool:
        try:
            if self._lexists(target):
                if current_payload is None or self._read_raw(target) != current_payload:
                    return False
                self._unlink_file(target, expected_bytes=current_payload)
            if self._lexists(backup):
                self._link_no_replace(backup, target.parent, target.name)
                return True
        except Exception:
            # Preserve whatever verified prior object remains; never replace it.
            return False
        return False

    def _write_temp(self, path: Path, payload: bytes) -> None:
        fd: int | None = None
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
            fd = os.open(path, flags, 0o600)
            with os.fdopen(fd, "wb", closefd=True) as handle:
                fd = None
                handle.write(payload)
                handle.flush()
                flush = getattr(self._boundary, "flush_private_file", None)
                if callable(flush):
                    flush(handle.fileno())
                else:
                    os.fsync(handle.fileno())
            self._ensure_private_node(path, directory=False)
            self._ensure_private_existing(path, directory=False)
        except ScratchStoreError:
            self._unlink_file_best_effort(path, expected_bytes=payload)
            raise
        except Exception:
            self._unlink_file_best_effort(path, expected_bytes=payload)
            raise ScratchStoreError("scratch_write_failed") from None
        finally:
            if fd is not None:
                with contextlib.suppress(OSError):
                    os.close(fd)

    def _link_no_replace(self, temporary: Path, directory: Path, name: str) -> None:
        open_existing = self._primitive("open_existing", "_open_existing")
        if os.name == "nt":
            native_link = self._primitive("win_link_handle_no_replace")
            if callable(open_existing) and callable(native_link):
                source = parent = None
                try:
                    source, _ = open_existing(temporary, directory=False)
                    parent, _ = open_existing(directory, directory=True)
                    native_link(source, parent, name)
                    return
                finally:
                    if source is not None:
                        source.close()
                    if parent is not None:
                        parent.close()
        else:
            posix_link = self._primitive("posix_link_no_replace", "_posix_linkat_empty_path")
            if callable(open_existing) and callable(posix_link):
                source = parent = None
                try:
                    source, _ = open_existing(temporary, directory=False)
                    parent, _ = open_existing(directory, directory=True)
                    posix_link(int(source.raw), int(parent.raw), name)
                    return
                finally:
                    if source is not None:
                        source.close()
                    if parent is not None:
                        parent.close()
        os.link(temporary, directory / name)

    def _unlink_file(self, path: Path, *, expected: dict[str, Any] | None = None, expected_bytes: bytes | None = None) -> None:
        if expected_bytes is None and expected is not None:
            expected_bytes = canonical_json_bytes(expected)
        unlink = self._primitive("unlink_handle_bound")
        open_existing = self._primitive("open_existing", "_open_existing")
        if callable(unlink) and callable(open_existing):
            source = parent = None
            try:
                source, source_identity = open_existing(path, directory=False)
                parent, parent_identity = open_existing(path.parent, directory=True)
                result = unlink(
                    path,
                    source_identity,
                    parent_expected=parent_identity,
                    expected_sha256=hashlib.sha256(expected_bytes or b"").hexdigest(),
                    max_bytes=_INTERPRETATION_MAX_BYTES,
                )
                if not result:
                    raise ScratchStoreError("scratch_write_failed")
                return
            finally:
                if source is not None:
                    source.close()
                if parent is not None:
                    parent.close()
        path.unlink()

    def _unlink_file_best_effort(self, path: Path, *, expected_bytes: bytes) -> None:
        try:
            if not self._lexists(path):
                return
            self._ensure_private_existing(path, directory=False)
            if self._read_raw(path) != expected_bytes:
                return
            self._unlink_file(path, expected_bytes=expected_bytes)
        except Exception:
            pass

    def _flush_parent(self, path: Path) -> None:
        flush = self._primitive("fsync_parent", "_fsync_parent")
        if callable(flush):
            try:
                flush(path)
            except Exception:
                raise ScratchStoreError("scratch_write_failed") from None
            return
        if os.name == "nt":
            raise ScratchStoreError("scratch_write_failed")
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except Exception:
            raise ScratchStoreError("scratch_write_failed") from None

    @contextlib.contextmanager
    def _lock_scope(self, directory: Path) -> Iterator[None]:
        try:
            lock = self._lock_factory(directory)
            if hasattr(lock, "__enter__") and hasattr(lock, "__exit__"):
                with lock:
                    yield
                return
            lock.acquire()
            try:
                yield
            finally:
                lock.release()
                close = getattr(lock, "close", None)
                if callable(close):
                    close()
        except ScratchStoreError:
            raise
        except Exception:
            raise ScratchStoreError("scratch_unavailable") from None

    def _primitive(self, *names: str) -> Callable[..., Any] | None:
        for name in names:
            value = getattr(self._boundary, name, None)
            if callable(value):
                return value
        return None

    @staticmethod
    def _lexists(path: Path) -> bool:
        try:
            path.lstat()
            return True
        except FileNotFoundError:
            return False
        except Exception:
            raise ScratchStoreError("scratch_unavailable") from None


def _now_timestamp() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = ["ScratchStore", "ScratchStoreError"]
