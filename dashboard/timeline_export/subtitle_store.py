"""state/subtitles.json behind ProjectStore's boundary (lane w23-subs), the way planning/store.py keeps the planning
files: the project's cross-process lock, the pinned folder revalidated and its manifest checked against the request's
project id, a bounded read that rejects a BOM, duplicate keys and NaN, the full document check (subtitles.check_document)
and ProjectStore's atomic create/replace. A damaged file is a named refusal (subtitles_document_invalid) and is never
reset; a missing one reads as empty.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator

from . import subtitles as S

NAME = "subtitles.json"
MAX_BYTES = 8 * 1024 * 1024


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate key")
        out[key] = value
    return out


def _no_constant(_value: str) -> None:
    raise ValueError("NaN or Infinity")


def loads(raw: bytes) -> dict[str, Any]:
    try:
        if raw.startswith(b"\xef\xbb\xbf"):
            raise ValueError("BOM")
        doc = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicates, parse_constant=_no_constant)
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise S.SubtitleRefusal("subtitles_document_invalid", f"{NAME} is not valid JSON; it was left as it is.") from None
    problem = S.check_document(doc)
    if problem:
        raise S.SubtitleRefusal("subtitles_document_invalid", f"{NAME} is damaged ({problem}); it was left as it is.")
    return doc


def dumps(doc: dict[str, Any]) -> bytes:
    problem = S.check_document(doc)
    if problem:
        raise S.SubtitleRefusal("subtitles_document_invalid", f"{NAME} would be invalid ({problem}); nothing was saved.")
    data = json.dumps(doc, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(data) > MAX_BYTES:
        raise S.SubtitleRefusal("subtitles_too_large", f"{NAME} would exceed 8 MiB; nothing was saved.")
    return data


class _Transaction:
    def __init__(self, owner: "SubtitleStore", project_dir: Path, identity: Any) -> None:
        self.owner, self.project_dir, self.identity = owner, project_dir, identity

    def load(self) -> dict[str, Any]:
        target = self.project_dir / "state" / NAME
        if not os.path.lexists(target):
            return S.empty()
        if target.is_symlink() or not target.is_file():
            raise S.SubtitleRefusal("subtitles_document_invalid", f"{NAME} is not a plain file; it was left as it is.")
        return loads(self.owner.ps._read_regular(self.owner.boundary, target, max_bytes=MAX_BYTES))

    def save(self, doc: dict[str, Any]) -> None:
        ps, boundary, store = self.owner.ps, self.owner.boundary, self.owner.store
        data = dumps(doc)
        boundary.revalidate_directory(self.project_dir, self.identity)
        state_dir = store._ensure_state_dir(self.project_dir)
        state_identity = boundary.pin_directory(state_dir)
        target = state_dir / NAME
        try:
            if os.path.lexists(target):
                store._atomic_replace_file(state_dir, target, data, max_bytes=MAX_BYTES * 2, expected_directory_identity=state_identity)
            else:
                store._atomic_create_file(self.project_dir, state_dir, target, data, project_identity=self.identity, directory_identity=state_identity)
        except ps.ProjectError as exc:
            if exc.code.startswith("music_"):
                raise ps.ProjectError("project_io_failed", f"{NAME} could not be saved") from exc
            raise


class SubtitleStore:
    def __init__(self, store: Any) -> None:
        self.store = store
        self.boundary = store.boundary
        self.ps = sys.modules[type(store).__module__]

    @contextlib.contextmanager
    def open(self, directory: str | Path, project_id: Any) -> Iterator[_Transaction]:
        project_id = self.ps._project_id(project_id)
        project_dir = self.store._directory(directory)
        identity = self.boundary.pin_directory(project_dir)
        with self.boundary.project_lock(project_dir):
            self.boundary.revalidate_directory(project_dir, identity)
            if self.store._manifest_locked(project_dir)["project_id"] != project_id:
                raise self.ps.ProjectError("project_scope_mismatch", "the subtitles belong to another project")
            yield _Transaction(self, project_dir, identity)
