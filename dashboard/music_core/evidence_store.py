"""Immutable, path-free global stores for neutral music artifacts.

The store deliberately depends on an injected filesystem boundary.  That keeps
this package independent from ``studio_core`` while allowing the composition
root to provide its reparse-aware, handle-relative primitives.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import secrets
import stat
import threading
from pathlib import Path
from typing import Any, Callable, Iterator

from .contracts import (
    MusicContractError,
    canonical_json_bytes,
    compute_evidence_id,
    compute_waveform_id,
    validate_evidence,
    validate_waveform,
)


_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_PATH_VALUE_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|/|file://)")
_KIND_CONFIG: dict[str, tuple[Callable[[Any], dict[str, Any]], Callable[[Any], str], str, int]] = {
    "evidence": (validate_evidence, compute_evidence_id, "evidence_id", 64 * 1024 * 1024),
    "waveform": (validate_waveform, compute_waveform_id, "waveform_id", 128 * 1024 * 1024),
}
_LAYOUT_LOCK = threading.Lock()


class MusicStoreError(MusicContractError):
    """Finite, path-free errors raised by the immutable artifact store."""

    _MESSAGES = {
        "invalid_id": "invalid music artifact id",
        "invalid_artifact": "invalid music artifact",
        "relation_mismatch": "evidence and waveform are unrelated",
        "private_store_unavailable": "music artifact store unavailable",
        "private_node_invalid": "music artifact store node is invalid",
        "artifact_not_found": "music artifact not found",
        "artifact_corrupt": "music artifact is corrupt",
        "artifact_collision": "music artifact id collision",
        "publication_failed": "music artifact publication failed",
    }

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code if code in self._MESSAGES else "private_store_unavailable"
        self.public_message = message or self._MESSAGES[self.code]
        # Never include OS/provider exception text here: it can contain paths.
        super().__init__(self.public_message)


class EvidenceStore:
    """Content-addressed immutable Evidence/Waveform storage.

    ``root`` and all filesystem operations are injected.  A boundary may expose
    the public names used by the composition root (``ensure_private_node``,
    ``flush_private_file``, ``fsync_parent``, ``open_existing`` and the
    platform-specific no-replace link primitive), or the corresponding
    underscored path-boundary helpers.  A conservative stdlib fallback is kept
    for small test doubles; it never replaces an existing target.
    """

    def __init__(
        self,
        root: os.PathLike[str] | str,
        boundary: Any,
        lock_factory: Callable[[Path], Any] | None = None,
    ) -> None:
        try:
            self._root = Path(root)
        except (TypeError, ValueError, OSError) as exc:
            raise MusicStoreError("private_store_unavailable") from None
        if boundary is None:
            raise MusicStoreError("private_store_unavailable")
        self._boundary = boundary
        factory = lock_factory or getattr(boundary, "project_lock", None)
        if not callable(factory):
            raise MusicStoreError("private_store_unavailable")
        self._lock_factory = factory

    @property
    def root(self) -> None:
        """The store intentionally exposes no filesystem locator."""

        return None

    def publish_evidence(self, artifact: Any) -> str:
        normalized = self._validate_for_write("evidence", artifact)
        return self._publish_validated("evidence", normalized)

    def publish_waveform(
        self,
        artifact: Any,
        *,
        evidence: Any | None = None,
        evidence_id: str | None = None,
    ) -> str:
        normalized = self._validate_for_write("waveform", artifact)
        if evidence is not None and evidence_id is not None:
            raise MusicStoreError("relation_mismatch")
        if evidence is not None:
            related = self._validate_for_write("evidence", evidence)
            self._check_relation(related, normalized)
        elif evidence_id is not None:
            related = self.read_evidence(evidence_id)
            self._check_relation(related, normalized)
        return self._publish_validated("waveform", normalized)

    def publish_pair(self, evidence: Any, waveform: Any) -> tuple[str, str]:
        """Validate and publish a related Evidence/Waveform pair."""

        evidence_value = self._validate_for_write("evidence", evidence)
        waveform_value = self._validate_for_write("waveform", waveform)
        self._check_relation(evidence_value, waveform_value)
        self._prepare_layout()
        try:
            with self._lock_scope(self._root):
                self._publish_validated_unlocked("evidence", evidence_value)
                self._publish_validated_unlocked("waveform", waveform_value)
        except MusicStoreError:
            raise
        except Exception as exc:
            raise MusicStoreError("publication_failed") from None
        return evidence_value["evidence_id"], waveform_value["waveform_id"]

    # Short aliases make the store convenient for the route composition root.
    publish_bundle = publish_pair

    def publish(self, evidence: Any, waveform: Any | None = None) -> str | tuple[str, str]:
        if waveform is None:
            if isinstance(evidence, dict) and evidence.get("schema") == "studio.music-waveform/1":
                return self.publish_waveform(evidence)
            return self.publish_evidence(evidence)
        return self.publish_pair(evidence, waveform)

    def read_evidence(self, evidence_id: str, *, waveform_id: str | None = None) -> dict[str, Any]:
        evidence = self._read_validated("evidence", evidence_id)
        if waveform_id is not None:
            waveform = self._read_validated("waveform", waveform_id)
            self._check_relation(evidence, waveform)
        return evidence

    def read_waveform(self, waveform_id: str, *, evidence_id: str | None = None) -> dict[str, Any]:
        waveform = self._read_validated("waveform", waveform_id)
        if evidence_id is not None:
            evidence = self._read_validated("evidence", evidence_id)
            self._check_relation(evidence, waveform)
        return waveform

    get_evidence = read_evidence
    get_waveform = read_waveform

    def read_pair(
        self,
        evidence_id: str,
        waveform_id: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        evidence = self._read_validated("evidence", evidence_id)
        selected_waveform_id = waveform_id or evidence["waveform"]["waveform_id"]
        waveform = self._read_validated("waveform", selected_waveform_id)
        self._check_relation(evidence, waveform)
        return evidence, waveform

    get_pair = read_pair

    def _validate_for_write(self, kind: str, artifact: Any) -> dict[str, Any]:
        validator, id_computer, id_field, _cap = self._config(kind)
        self._reject_path_values(artifact)
        try:
            normalized = validator(artifact)
            expected_id = id_computer(normalized)
        except MusicContractError as exc:
            raise MusicStoreError("invalid_artifact") from None
        except Exception as exc:
            raise MusicStoreError("invalid_artifact") from None
        if normalized.get(id_field) != expected_id:
            raise MusicStoreError("invalid_artifact")
        return normalized

    def _publish_validated(self, kind: str, artifact: dict[str, Any]) -> str:
        self._prepare_layout()
        try:
            with self._lock_scope(self._root):
                return self._publish_validated_unlocked(kind, artifact)
        except MusicStoreError:
            raise
        except Exception as exc:
            raise MusicStoreError("publication_failed") from None

    def _publish_validated_unlocked(self, kind: str, artifact: dict[str, Any]) -> str:
        _validator, _id_computer, id_field, cap = self._config(kind)
        artifact_id = artifact[id_field]
        directory = self._kind_directory(kind)
        target = directory / f"{artifact_id}.json"
        payload = canonical_json_bytes(artifact)

        self._validate_directory(directory)
        if self._lexists(target):
            self._ensure_private_existing(target, directory=False)
            if self._read_raw(target, cap) == payload:
                return artifact_id
            raise MusicStoreError("artifact_collision")

        temporary = directory / f".studio-music-{secrets.token_hex(16)}.tmp"
        published = False
        try:
            self._write_temp(temporary, payload)
            # The hook is intentionally tiny and private: crash/fault tests can
            # stop after a fully flushed sibling without affecting public API.
            self._before_publish()
            self._flush_parent(directory)
            try:
                self._link_no_replace(temporary, directory, target.name)
            except FileExistsError:
                # The winner is authoritative.  Compare bytes, never replace it.
                self._ensure_private_existing(target, directory=False)
                if self._read_raw(target, cap) != payload:
                    raise MusicStoreError("artifact_collision")
                return artifact_id
            published = True
            self._flush_parent(directory)
            return artifact_id
        except MusicStoreError:
            if published:
                self._remove_published_if_owned(target, directory, payload, cap)
            raise
        except Exception as exc:
            if published:
                self._remove_published_if_owned(target, directory, payload, cap)
            raise MusicStoreError("publication_failed") from None
        finally:
            self._remove_own_temp(temporary)

    def _read_validated(self, kind: str, artifact_id: str) -> dict[str, Any]:
        _validator, id_computer, id_field, cap = self._config(kind)
        self._require_id(artifact_id)
        self._validate_existing_layout()
        target = self._kind_directory(kind) / f"{artifact_id}.json"
        try:
            self._ensure_private_existing(target, directory=False)
            raw = self._read_raw(target, cap)
            try:
                value = json.loads(raw.decode("utf-8"))
            except Exception as exc:
                raise MusicStoreError("artifact_corrupt") from None
            self._reject_path_values(value)
            try:
                normalized = _validator(value)
                expected_id = id_computer(normalized)
            except Exception as exc:
                raise MusicStoreError("artifact_corrupt") from None
            if normalized.get(id_field) != artifact_id or expected_id != artifact_id:
                raise MusicStoreError("artifact_corrupt")
            # The on-disk representation is itself immutable canonical content.
            if canonical_json_bytes(normalized) != raw:
                raise MusicStoreError("artifact_corrupt")
            return normalized
        except MusicStoreError:
            raise
        except FileNotFoundError as exc:
            raise MusicStoreError("artifact_not_found") from None
        except Exception as exc:
            raise MusicStoreError("artifact_corrupt") from None

    def _config(self, kind: str):
        try:
            return _KIND_CONFIG[kind]
        except KeyError as exc:
            raise MusicStoreError("private_store_unavailable") from None

    def _prepare_layout(self) -> None:
        with _LAYOUT_LOCK:
            try:
                self._ensure_private_node(self._root, directory=True)
                self._ensure_private_node(self._kind_directory("evidence"), directory=True)
                self._ensure_private_node(self._kind_directory("waveform"), directory=True)
                self._validate_directory(self._root)
                self._validate_directory(self._kind_directory("evidence"))
                self._validate_directory(self._kind_directory("waveform"))
            except MusicStoreError:
                raise
            except Exception as exc:
                raise MusicStoreError("private_store_unavailable") from None

    def _validate_existing_layout(self) -> None:
        try:
            for path in (self._root, self._kind_directory("evidence"), self._kind_directory("waveform")):
                self._validate_directory(path)
        except MusicStoreError:
            raise
        except Exception as exc:
            raise MusicStoreError("private_store_unavailable") from None

    def _kind_directory(self, kind: str) -> Path:
        if kind not in _KIND_CONFIG:
            raise MusicStoreError("private_store_unavailable")
        return self._root / kind

    def _validate_directory(self, path: Path) -> None:
        self._ensure_private_existing(path, directory=True)
        self._validate_confined(path, path.parent, directory=True, allow_missing=False)

    def _ensure_private_node(self, path: Path, *, directory: bool) -> None:
        if self._lexists(path):
            self._ensure_private_existing(path, directory=directory)
            return
        fn = self._primitive("ensure_private_node", "ensure_private")
        if fn is not None:
            try:
                fn(path, directory=directory)
            except Exception as exc:
                raise MusicStoreError("private_store_unavailable") from None
        else:
            try:
                if directory:
                    path.mkdir(parents=True, exist_ok=True)
                    if os.name != "nt":
                        os.chmod(path, 0o700)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    os.close(fd)
            except Exception as exc:
                raise MusicStoreError("private_store_unavailable") from None

    def _ensure_private_existing(self, path: Path, *, directory: bool) -> None:
        if not self._lexists(path):
            raise MusicStoreError("artifact_not_found" if not directory else "private_node_invalid")
        try:
            self._reject_reparse_components(path)
            info = path.lstat()
        except Exception as exc:
            raise MusicStoreError("private_node_invalid") from None
        if directory:
            valid_type = stat.S_ISDIR(info.st_mode)
            expected_mode = 0o700
        else:
            valid_type = stat.S_ISREG(info.st_mode)
            expected_mode = 0o600
        if not valid_type:
            raise MusicStoreError("private_node_invalid")
        if os.name != "nt" and stat.S_IMODE(info.st_mode) != expected_mode:
            raise MusicStoreError("private_node_invalid")
        verify = self._primitive("require_private_node", "verify_private_node")
        if os.name == "nt":
            if verify is None:
                raise MusicStoreError("private_node_invalid")
            try:
                verify(path, directory=directory)
            except Exception as exc:
                raise MusicStoreError("private_node_invalid") from None

    def _validate_confined(self, path: Path, parent: Path, *, directory: bool, allow_missing: bool) -> None:
        fn = self._primitive("validate_confined_path", "validate_confined")
        if fn is None:
            self._reject_reparse_components(path)
            try:
                path.relative_to(parent)
            except ValueError as exc:
                raise MusicStoreError("private_node_invalid") from None
            return
        try:
            fn(path, (parent,), directory=directory, allow_missing_leaf=allow_missing)
        except Exception as exc:
            raise MusicStoreError("private_node_invalid") from None

    def _reject_reparse_components(self, path: Path) -> None:
        # The injected PathBoundary performs the authoritative Windows reparse
        # walk.  This fallback still rejects symlink components on every host.
        for component in self._components(path):
            try:
                if component.is_symlink():
                    raise MusicStoreError("private_node_invalid")
            except MusicStoreError:
                raise
            except OSError as exc:
                raise MusicStoreError("private_node_invalid") from None

    @staticmethod
    def _components(path: Path) -> list[Path]:
        parts = path.parts
        if not parts:
            return []
        current = Path(parts[0])
        result = [current]
        for part in parts[1:]:
            current = current / part
            result.append(current)
        return result

    @staticmethod
    def _lexists(path: Path) -> bool:
        try:
            path.lstat()
            return True
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise MusicStoreError("private_node_invalid") from None

    def _read_raw(self, path: Path, cap: int) -> bytes:
        try:
            with path.open("rb") as handle:
                raw = handle.read(cap + 1)
        except FileNotFoundError:
            raise MusicStoreError("artifact_not_found")
        except Exception as exc:
            raise MusicStoreError("artifact_corrupt") from None
        if len(raw) > cap:
            raise MusicStoreError("artifact_corrupt")
        return raw

    def _write_temp(self, path: Path, payload: bytes) -> None:
        fd: int | None = None
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
            fd = os.open(path, flags, 0o600)
            with os.fdopen(fd, "wb", closefd=True) as handle:
                fd = None
                handle.write(payload)
                handle.flush()
                flush = self._primitive("flush_private_file", "_flush_private_file")
                if flush is not None:
                    flush(handle.fileno())
                else:
                    os.fsync(handle.fileno())
            private = self._primitive("ensure_private_node", "ensure_private")
            if private is not None:
                private(path, directory=False)
            elif os.name != "nt":
                os.chmod(path, 0o600)
            self._ensure_private_existing(path, directory=False)
        except MusicStoreError:
            raise
        except Exception as exc:
            raise MusicStoreError("publication_failed") from None
        finally:
            if fd is not None:
                with contextlib.suppress(OSError):
                    os.close(fd)

    def _link_no_replace(self, temporary: Path, directory: Path, name: str) -> None:
        open_existing = self._primitive("open_existing", "_open_existing")
        if os.name == "nt":
            native_link = self._primitive("win_link_handle_no_replace")
            if open_existing is not None and native_link is not None:
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
            if open_existing is not None and posix_link is not None:
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
        # os.link is an atomic same-directory no-replace operation on the
        # supported fallback platforms; unlike os.replace it cannot clobber.
        os.link(temporary, directory / name)

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
        except MusicStoreError:
            raise
        except Exception as exc:
            raise MusicStoreError("private_store_unavailable") from None

    def _flush_parent(self, path: Path) -> None:
        flush = self._primitive("fsync_parent", "_fsync_parent")
        if flush is not None:
            try:
                flush(path)
                return
            except Exception as exc:
                raise MusicStoreError("publication_failed") from None
        if os.name == "nt":
            raise MusicStoreError("publication_failed")
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except Exception as exc:
            raise MusicStoreError("publication_failed") from None

    def _remove_published_if_owned(
        self,
        target: Path,
        directory: Path,
        payload: bytes,
        cap: int,
    ) -> None:
        """Best-effort rollback of only this call's exact published object."""

        try:
            if not self._lexists(target):
                return
            self._ensure_private_existing(target, directory=False)
            if self._read_raw(target, cap) != payload:
                return
            unlink = self._primitive("unlink_handle_bound")
            open_existing = self._primitive("open_existing", "_open_existing")
            if unlink is not None and open_existing is not None:
                source = parent = None
                try:
                    source, source_identity = open_existing(target, directory=False)
                    parent, parent_identity = open_existing(directory, directory=True)
                    unlink(
                        target,
                        source_identity,
                        parent_expected=parent_identity,
                        expected_sha256=hashlib.sha256(payload).hexdigest(),
                        max_bytes=cap,
                    )
                    return
                finally:
                    if source is not None:
                        source.close()
                    if parent is not None:
                        parent.close()
            target.unlink()
        except Exception:
            # The target is safer than an unproven deletion after a race.
            return

    @staticmethod
    def _remove_own_temp(path: Path) -> None:
        with contextlib.suppress(FileNotFoundError, OSError):
            path.unlink()

    def _primitive(self, *names: str) -> Callable[..., Any] | None:
        for name in names:
            value = getattr(self._boundary, name, None)
            if callable(value):
                return value
        return None

    @staticmethod
    def _require_id(value: Any) -> str:
        if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
            raise MusicStoreError("invalid_id")
        return value

    @classmethod
    def _reject_path_values(cls, value: Any) -> None:
        def walk(item: Any) -> None:
            if isinstance(item, str):
                if _PATH_VALUE_RE.match(item):
                    raise MusicStoreError("invalid_artifact")
                return
            if isinstance(item, dict):
                for key, child in item.items():
                    if isinstance(key, str) and _PATH_VALUE_RE.match(key):
                        raise MusicStoreError("invalid_artifact")
                    walk(child)
                return
            if isinstance(item, list):
                for child in item:
                    walk(child)

        try:
            walk(value)
        except MusicStoreError:
            raise
        except Exception as exc:
            raise MusicStoreError("invalid_artifact") from None

    @staticmethod
    def _check_relation(evidence: dict[str, Any], waveform: dict[str, Any]) -> None:
        try:
            evidence_waveform_id = evidence["waveform"]["waveform_id"]
            if (
                evidence["track_sha256"] != waveform["track_sha256"]
                or evidence_waveform_id != waveform["waveform_id"]
            ):
                raise MusicStoreError("relation_mismatch")
        except (KeyError, TypeError) as exc:
            raise MusicStoreError("relation_mismatch") from None

    def _before_publish(self) -> None:
        """Fault-injection seam; it performs no work in production."""


__all__ = ["EvidenceStore", "MusicStoreError"]
