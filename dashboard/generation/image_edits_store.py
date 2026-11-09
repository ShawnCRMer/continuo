"""Image edits kept with the owner scope: develop values and History per picture.

Document ``studio.image-edits/1``: one small JSON file per picture, keyed by the
sha256 of the Original (the key Develop uses), in one folder per owner scope (a
project's ``state/image-edits/``, or the profile's private plugin data for No
project). Each picture has its own revision, so editing one picture never
rewrites, conflicts with or evicts another one.

* A write names ``expected_revision``; a different current revision is a
  conflict (the caller answers 409 with the current revision), never a silent
  overwrite.
* Bounds are refusals, never evictions: ``PICTURE_MAX_BYTES`` per picture,
  ``PICTURES_MAX`` pictures and ``SCOPE_MAX_BYTES`` per scope.
* Atomic: the new bytes go to a temp file in the same folder (flushed and
  fsynced), then ``os.replace``; a reader sees the old or the new file, never a
  part.
* Nothing is deleted: an emptied picture keeps its file with ``develop`` and
  ``history`` set to null and its revision.

Pure module (stdlib only); loaded by path from routes/image/edits_1.py.
"""
from __future__ import annotations

import json
import math
import os
import re
import stat
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "studio.image-edits/1"
HISTORY_SCHEMA = "studio.image-history/1"
# The renderer's History budget is 250,000 characters per picture
# (sel-history-store.js HIST_PICTURE_CHARS); develop values incl. masks fit in the rest.
PICTURE_MAX_BYTES = 512 * 1024
PICTURES_MAX = 5000
SCOPE_MAX_BYTES = 256 * 1024 * 1024
# A History keeps SEL_HISTORY_MAX + 1 = 61 rows.
HISTORY_ROWS_MAX = 61
JSON_MAX_DEPTH = 32
READ_MAX = 16
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_FILE_RE = re.compile(r"^([0-9a-f]{64})\.json$")
_TEMP_RE = re.compile(r"^\.[0-9a-f]{64}\.[0-9a-f]{32}\.tmp$")
_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


class ImageEditsError(Exception):
    """A finite, path-free refusal: ``code`` is the public error code."""

    def __init__(self, code: str, message: str, *, revision: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.revision = revision


def folder_lock(folder: Path) -> threading.RLock:
    key = os.path.normcase(os.path.abspath(str(folder)))
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = _LOCKS[key] = threading.RLock()
        return lock


def _depth_ok(value: Any, depth: int = 0) -> bool:
    if depth > JSON_MAX_DEPTH:
        return False
    if isinstance(value, dict):
        return all(isinstance(k, str) and _depth_ok(v, depth + 1) for k, v in value.items())
    if isinstance(value, list):
        return all(_depth_ok(v, depth + 1) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return value is None or isinstance(value, (str, int, bool))


def validate_item(sha256: str, develop: Any, history: Any) -> None:
    """The shape the pane writes: develop values (a JSON object) and the packed History."""
    if not isinstance(sha256, str) or not SHA_RE.fullmatch(sha256):
        raise ImageEditsError("invalid_image_edits", "The picture key must be the sha256 of the Original.")
    if develop is not None and not isinstance(develop, dict):
        raise ImageEditsError("invalid_image_edits", "Develop values must be an object or null.")
    if history is not None:
        if not isinstance(history, dict) or history.get("schema") != HISTORY_SCHEMA or history.get("sha") != sha256:
            raise ImageEditsError("invalid_image_edits", "The History must be a studio.image-history/1 entry of this picture.")
        rows, at = history.get("rows"), history.get("at")
        if (not isinstance(rows, list) or not 1 <= len(rows) <= HISTORY_ROWS_MAX or type(at) is not int
                or not 0 <= at < len(rows) or not isinstance(history.get("devs"), list)
                or not isinstance(history.get("masks"), list)):
            raise ImageEditsError("invalid_image_edits", "The History rows or position are invalid.")
    if not _depth_ok(develop) or not _depth_ok(history):
        raise ImageEditsError("invalid_image_edits", "The edits are nested too deeply or hold an invalid number.")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _regular(path: Path) -> os.stat_result | None:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ImageEditsError("image_edits_unavailable", "The edits could not be read.") from exc
    reparse = getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if not stat.S_ISREG(info.st_mode) or reparse:
        raise ImageEditsError("image_edits_damaged", "The edits of this picture are damaged.")
    return info


def check_folder(folder: Path) -> bool:
    """True when the folder exists as a real directory (never a link or junction)."""
    try:
        info = os.lstat(folder)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ImageEditsError("image_edits_unavailable", "The edits could not be read.") from exc
    reparse = getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if not stat.S_ISDIR(info.st_mode) or reparse:
        raise ImageEditsError("image_edits_damaged", "The edits folder is not a folder.")
    return True


def _parse(raw: bytes, scope: str, sha256: str) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("duplicate key")
            out[key] = value
        return out

    def no_constant(_value: str) -> None:
        raise ValueError("constant")

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicates, parse_constant=no_constant)
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise ImageEditsError("image_edits_damaged", "The edits of this picture are damaged.") from None
    if (not isinstance(value, dict) or set(value) != {"schema", "scope", "sha256", "revision", "updated_at", "develop", "history"}
            or value["schema"] != SCHEMA or value["scope"] != scope or value["sha256"] != sha256
            or type(value["revision"]) is not int or value["revision"] < 1):
        raise ImageEditsError("image_edits_damaged", "The edits of this picture are damaged.")
    try:
        validate_item(sha256, value["develop"], value["history"])
    except ImageEditsError:
        raise ImageEditsError("image_edits_damaged", "The edits of this picture are damaged.") from None
    return value


def read_one(folder: Path | None, scope: str, sha256: str) -> dict[str, Any] | None:
    if folder is None or not check_folder(folder):
        return None
    target = folder / f"{sha256}.json"
    info = _regular(target)
    if info is None:
        return None
    if info.st_size > PICTURE_MAX_BYTES:
        raise ImageEditsError("image_edits_damaged", "The edits of this picture are damaged.")
    try:
        with open(target, "rb") as handle:
            raw = handle.read(PICTURE_MAX_BYTES + 1)
    except OSError as exc:
        raise ImageEditsError("image_edits_unavailable", "The edits could not be read.") from exc
    if len(raw) > PICTURE_MAX_BYTES:
        raise ImageEditsError("image_edits_damaged", "The edits of this picture are damaged.")
    return _parse(raw, scope, sha256)


def index(folder: Path | None) -> dict[str, Any]:
    """Every picture of the scope (sha256 -> bytes), without content; bounded by PICTURES_MAX."""
    pictures: dict[str, int] = {}
    total = 0
    if folder is not None and check_folder(folder):
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    match = _FILE_RE.fullmatch(entry.name)
                    if not match or not entry.is_file(follow_symlinks=False):
                        continue
                    size = entry.stat(follow_symlinks=False).st_size
                    pictures[match.group(1)] = size
                    total += size
                    if len(pictures) > PICTURES_MAX + 1:
                        break
        except OSError as exc:
            raise ImageEditsError("image_edits_unavailable", "The edits could not be read.") from exc
    return {"pictures": pictures, "count": len(pictures), "bytes": total}


def read_many(folder: Path | None, scope: str, shas: list[str]) -> dict[str, Any]:
    pictures: dict[str, Any] = {}
    missing: list[str] = []
    for sha256 in shas:
        record = read_one(folder, scope, sha256)
        if record is None:
            missing.append(sha256)
        else:
            pictures[sha256] = {k: record[k] for k in ("revision", "updated_at", "develop", "history")}
    return {"pictures": pictures, "missing": missing}


def _fsync_dir(folder: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(folder, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_one(folder: Path, scope: str, sha256: str, expected_revision: int, develop: Any, history: Any) -> dict[str, Any]:
    """Write one picture's edits with expected_revision. Caller holds the folder (and project) lock."""
    validate_item(sha256, develop, history)
    if type(expected_revision) is not int or expected_revision < 0:
        raise ImageEditsError("invalid_image_edits", "expected_revision must be a whole number from 0.")
    if not check_folder(folder):
        raise ImageEditsError("image_edits_unavailable", "The edits folder is missing.")
    current = read_one(folder, scope, sha256)
    revision = current["revision"] if current else 0
    if revision != expected_revision:
        raise ImageEditsError("image_edits_conflict", "The edits of this picture changed meanwhile.", revision=revision)
    document = {"schema": SCHEMA, "scope": scope, "sha256": sha256, "revision": revision + 1, "updated_at": _now(), "develop": develop, "history": history}
    try:
        payload = json.dumps(document, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError):
        raise ImageEditsError("invalid_image_edits", "The edits hold a value that is not JSON.") from None
    if len(payload) > PICTURE_MAX_BYTES:
        raise ImageEditsError("image_edits_too_large", f"The edits of this picture are larger than {PICTURE_MAX_BYTES // 1024} KB; nothing was changed.")
    seen = index(folder)
    old_size = seen["pictures"].get(sha256, 0)
    if current is None and seen["count"] >= PICTURES_MAX:
        raise ImageEditsError("image_edits_full", f"This scope keeps edits of {PICTURES_MAX} pictures already; nothing was removed.")
    if seen["bytes"] - old_size + len(payload) > SCOPE_MAX_BYTES:
        raise ImageEditsError("image_edits_full", f"The edits of this scope reached {SCOPE_MAX_BYTES // (1024 * 1024)} MB; nothing was removed.")
    target = folder / f"{sha256}.json"
    temporary = folder / f".{sha256}.{uuid.uuid4().hex}.tmp"
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        fd = os.open(temporary, flags, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_dir(folder)
    except OSError as exc:
        raise ImageEditsError("image_edits_unavailable", "The edits could not be saved.") from exc
    finally:
        try:
            if os.path.lexists(temporary):
                os.unlink(temporary)
        except OSError:
            pass
    return {"revision": document["revision"], "updated_at": document["updated_at"], "bytes": len(payload),
            "scope_pictures": seen["count"] + (0 if current else 1), "scope_bytes": seen["bytes"] - old_size + len(payload)}
