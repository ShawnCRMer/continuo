"""Storage of ``studio.script/1``: ``<project>/script/script.json`` and kept revisions.

The envelope is the timeline's: ``{schema, project_id, revision, updated_at, label, by,
state}``. Each revision is also kept as ``script.json.r<revision>`` (newest
``KEEP_REVISIONS``), and after each save the plain ``script.fountain`` beside it is
regenerated (readable, diffable; the json is the truth, the Fountain file is never read
back). Exports land in ``<project>/exports/script/``.

Bytes go through ProjectStore's atomic create/replace primitives under the project lock
(the same ones the timeline uses); this module only adds the script's shape. The
project-wide JSON reader caps objects at 256 keys, so the script document has its own
bounded reader (duplicate keys, NaN and a BOM are refused the same way).
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

from . import fountain
from . import review as review_rules
from .schema import AUTHORS, SCHEMA, ScriptError, fail, validate_state

MAX_BYTES = 4 * 1024 * 1024
KEEP_REVISIONS = 100
ENVELOPE_KEYS = {"schema", "project_id", "revision", "updated_at", "label", "by", "state"}
_PROJECT_RE = re.compile(r"^prj_[0-9a-f]{32}$")
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_REVISION_RE = re.compile(r"^script\.json\.r([1-9][0-9]{0,8})$")
_EXPORT_NAME_RE = re.compile(r"[^A-Za-z0-9._ -]+")

# Stable error codes -> (retryable, what to do next).
ERRORS: dict[str, tuple[bool, str]] = {
    "scene_conflict": (True, "The scene changed since this revision (current is in details)."),
    "order_conflict": (True, "The scene order changed since this revision."),
    "card_conflict": (True, "The card was moved after this revision."),
    "script_conflict": (True, "The script changed since this revision."),
    "invalid_script": (False, "The named field is invalid; nothing was saved."),
    "invalid_op": (False, "The named argument is invalid."),
    "unknown_op": (False, "Op not among the script tools."),
    "not_found": (True, "No such id in the current script."),
    "not_in_order": (False, "Scene aside, not in the order (script_aside with aside: false brings it back)."),
    "scene_numbered": (False, "The scene has a locked number; omitting keeps it, deleting needs force: true."),
    "scene_too_large": (False, "One scene holds at most 64 KiB of text."),
    "script_too_large": (False, "The script is capped at 2000 scenes, 200 act breaks and 4 MiB."),
    "revision_unknown": (True, "No kept revision with this id."),
    "review_conflict": (True, "Hermes changed the script since this review was read (the record now is in details)."),
    "review_stale_base": (True, "A scene changed after the revision this baseline was computed on (the record now is in details)."),
    "review_unknown_scene": (True, "No such scene in the current script; the review is older than the script."),
    "invalid_import": (False, "Not a Fountain or FDX file; nothing was imported."),
    "import_too_large": (False, "Import files are capped at 8 MiB."),
    "script_unavailable": (True, "The project folder could not be read or written."),
    "project_conflict": (True, "The project changed since it was opened."),
    "project_not_found": (False, "No project open."),
    "project_scope_mismatch": (False, "The script belongs to another project."),
}


def error_body(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    retryable, next_step = ERRORS.get(code, (False, "The script was not changed."))
    body: dict[str, Any] = {"code": code, "message": message, "retryable": retryable, "next": next_step}
    if details:
        body["details"] = details
    return body


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            fail("invalid_script", "script JSON contains duplicate keys")
        out[key] = value
    return out


def _reject_constant(_value: str) -> None:
    fail("invalid_script", "script JSON number is invalid")


def loads(raw: bytes) -> Any:
    if len(raw) > MAX_BYTES:
        fail("script_too_large", "the script document exceeds 4 MiB")
    if raw.startswith(b"\xef\xbb\xbf"):
        fail("invalid_script", "script JSON must not contain a UTF-8 BOM")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicates, parse_constant=_reject_constant)
    except ScriptError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError):
        fail("invalid_script", "script JSON is not valid UTF-8 JSON")


def payload(envelope: dict[str, Any]) -> bytes:
    data = json.dumps(envelope, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(data) > MAX_BYTES:
        fail("script_too_large", "the script document exceeds 4 MiB")
    return data


def check(value: Any, project_id: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != ENVELOPE_KEYS:
        fail("invalid_script", "envelope: must hold exactly " + ", ".join(sorted(ENVELOPE_KEYS)))
    if value["schema"] != SCHEMA:
        fail("invalid_script", f"envelope.schema: must be {SCHEMA}")
    if type(value["project_id"]) is not str or not _PROJECT_RE.fullmatch(value["project_id"]) or value["project_id"] != project_id:
        fail("invalid_script", "envelope.project_id: belongs to another project")
    if type(value["revision"]) is not int or value["revision"] < 1:
        fail("invalid_script", "envelope.revision: must be an integer >= 1")
    if type(value["updated_at"]) is not str or not _TIMESTAMP_RE.fullmatch(value["updated_at"]):
        fail("invalid_script", "envelope.updated_at: must be an ISO UTC timestamp")
    if type(value["label"]) is not str or len(value["label"]) > 120:
        fail("invalid_script", "envelope.label: text of at most 120 characters")
    if value["by"] not in AUTHORS:
        fail("invalid_script", "envelope.by: must be artist or hermes")
    validate_state(value["state"])
    for scene in value["state"]["scenes"].values():
        if scene["revision"] > value["revision"] or (scene["card"] and scene["card"]["revision"] > value["revision"]):
            fail("invalid_script", "envelope: a scene is newer than the document")
    if value["state"]["order_revision"] > value["revision"]:
        fail("invalid_script", "envelope: the order is newer than the document")
    return value


def public(envelope: dict[str, Any] | None) -> dict[str, Any]:
    if envelope is None:
        return {"exists": False, "revision": 0, "updated_at": None, "label": "", "by": "artist", "state": None}
    return {"exists": True, **{key: envelope[key] for key in ("revision", "updated_at", "label", "by", "state")}}


class ScriptStore:
    """The script files of one ProjectStore (its boundary, lock and atomic primitives)."""

    def __init__(self, store: Any) -> None:
        self.store = store
        self.boundary = store.boundary
        self.ps = sys.modules[type(store).__module__]  # project_store: the shared primitives

    # --- locked helpers ----------------------------------------------------------

    def _open(self, directory: str | Path, project_id: Any) -> tuple[str, Path, Any]:
        project_id = self.ps._project_id(project_id)
        project_dir = self.store._directory(directory)
        return project_id, project_dir, self.boundary.pin_directory(project_dir)

    def _scope(self, project_dir: Path, identity: Any, project_id: str) -> None:
        self.boundary.revalidate_directory(project_dir, identity)
        if self.store._manifest_locked(project_dir)["project_id"] != project_id:
            raise self.ps.ProjectError("project_scope_mismatch", "script belongs to another project")

    def _file(self, path: Path, project_id: str) -> dict[str, Any]:
        raw = self.ps._read_regular(self.boundary, path, max_bytes=MAX_BYTES)
        return check(loads(raw), project_id)

    def _current(self, project_dir: Path, project_id: str) -> dict[str, Any] | None:
        target = project_dir / "script" / "script.json"
        if not os.path.lexists(target):
            return None
        return self._file(target, project_id)

    def _revision_files(self, script_dir: Path) -> list[tuple[int, Path]]:
        if not os.path.lexists(script_dir):
            return []
        found = []
        for name in self.ps._members(script_dir):
            match = _REVISION_RE.fullmatch(name)
            if match:
                found.append((int(match.group(1)), script_dir / name))
        return sorted(found, reverse=True)

    def _ensure_dir(self, parent: Path, name: str) -> Path:
        path = parent / name
        if not os.path.lexists(path):
            try:
                path.mkdir()
            except FileExistsError:
                pass
            except OSError as exc:
                raise self.ps.ProjectError("script_unavailable", f"{name} folder could not be created") from exc
        if path.is_symlink() or not path.is_dir():
            raise self.ps.ProjectError("script_unavailable", f"{name} is not a plain folder")
        return path

    def _put(self, project_dir: Path, project_identity: Any, directory: Path, name: str, data: bytes) -> None:
        """Create or atomically replace one file of a project folder."""
        directory_identity = self.boundary.pin_directory(directory)
        target = directory / name
        try:
            if os.path.lexists(target):
                self.store._atomic_replace_file(directory, target, data, max_bytes=MAX_BYTES * 2,
                                                expected_directory_identity=directory_identity)
            else:
                self.store._atomic_create_file(project_dir, directory, target, data,
                                               project_identity=project_identity, directory_identity=directory_identity)
        except self.ps.ProjectError as exc:
            if exc.code == "music_conflict":
                raise self.ps.ProjectError("script_conflict", "script revision conflict") from exc
            if exc.code.startswith("music_"):
                raise self.ps.ProjectError("script_unavailable", "script could not be saved") from exc
            raise

    # --- public ------------------------------------------------------------------

    def read(self, directory: str | Path, *, project_id: str) -> dict[str, Any]:
        project_id, project_dir, identity = self._open(directory, project_id)
        with self.boundary.project_lock(project_dir):
            self._scope(project_dir, identity, project_id)
            return public(self._current(project_dir, project_id))

    def change(self, directory: str | Path, *, project_id: str, expected_revision: int,
               mutate: Callable[[dict[str, Any] | None, int, int], tuple[dict[str, Any], dict[str, Any], bool]],
               label: str, by: str, dry_run: bool = False, undo_scenes: Any = None,
               wholesale: bool = False) -> dict[str, Any]:
        """Run ``mutate(state_or_None, current_revision, next_revision)`` under the project
        lock and save its state as the next revision when it changed anything. ``undo_scenes``
        (Undo Hermes change): every scene must be in the review record read under the same lock. ``wholesale``: the
        verb replaces scene text as a whole (restore, import, write_fountain; review.after_save)."""
        if type(expected_revision) is not int or expected_revision < 0:
            fail("invalid_op", "expected_revision must be an integer >= 0")
        if by not in AUTHORS:
            fail("invalid_op", "by must be artist or hermes")
        project_id, project_dir, identity = self._open(directory, project_id)
        with self.boundary.project_lock(project_dir):
            self._scope(project_dir, identity, project_id)
            current = self._current(project_dir, project_id)
            revision = 0 if current is None else current["revision"]
            if expected_revision > revision:
                fail("script_conflict", f"expected_revision {expected_revision} is newer than the script (revision {revision})")
            if undo_scenes is not None:
                review_rules.admit_undo(self._review_read(project_dir, project_id),
                                        None if current is None else current["state"], undo_scenes)
            state, result, changed = mutate(None if current is None else current["state"], revision, revision + 1)
            if not changed or dry_run:
                return {"saved": False, "revision": revision, "result": result, "state": state if current or changed else None}
            stamp = self.ps._now_timestamp() if current is None else self.ps._next_timestamp(current["updated_at"])
            envelope = check({"schema": SCHEMA, "project_id": project_id, "revision": revision + 1, "updated_at": stamp,
                              "label": label[:120], "by": by, "state": state}, project_id)
            data = payload(envelope)
            script_dir = self._ensure_dir(project_dir, "script")
            self._put(project_dir, identity, script_dir, "script.json", data)
            kept = f"script.json.r{revision + 1}"
            self._put(project_dir, identity, script_dir, kept, data)
            for _old, path in self._revision_files(script_dir)[KEEP_REVISIONS:]:
                self.ps._unlink_owned_file(self.boundary, path, expected=self.boundary.pin_file(path), max_bytes=MAX_BYTES)
            review_saved = True
            try:
                self._review_after(project_dir, identity, script_dir, project_id, current, state, by, revision + 1, stamp, label,
                                   wholesale)
            except (self.ps.ProjectError, ScriptError, OSError):
                review_saved = False  # the review never blocks a save; the page shows what it can read
            fountain_written = True
            try:
                self._put(project_dir, identity, script_dir, "script.fountain",
                          fountain.write(state, ids=False).encode("utf-8"))
            except (self.ps.ProjectError, OSError):
                fountain_written = False  # the json is the truth; the next save writes it again
            return {"saved": True, "revision": revision + 1, "updated_at": stamp, "result": result,
                    "state": state, "fountain_written": fountain_written, "review_saved": review_saved}

    # --- Hermes' changes under review (lane w14-scr, SCR-071): the rules live in review.py ---------------

    def _review_read(self, project_dir: Path, project_id: str) -> dict[str, Any]:
        target = project_dir / "script" / "review.json"
        if not os.path.lexists(target):
            return review_rules.empty()
        return review_rules.check(loads(self.ps._read_regular(self.boundary, target, max_bytes=MAX_BYTES)))

    def _review_write(self, project_dir: Path, identity: Any, script_dir: Path, record: dict[str, Any]) -> None:
        data = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self._put(project_dir, identity, script_dir, "review.json", data)

    def _review_after(self, project_dir: Path, identity: Any, script_dir: Path, project_id: str,
                      current: dict[str, Any] | None, state: dict[str, Any], by: str, revision: int,
                      stamp: str, label: str, wholesale: bool = False) -> None:
        record = self._review_read(project_dir, project_id)
        after = review_rules.after_save(record, None if current is None else current["state"], state, by, revision, stamp, label,
                                        wholesale)
        if after != record:
            self._review_write(project_dir, identity, script_dir, after)

    def review(self, directory: str | Path, *, project_id: str, baselines: dict[str, Any] | None = None,
               clear: bool = False, expected_hermes_revision: int | None = None,
               base_revision: int | None = None) -> dict[str, Any]:
        """Read the review record; ``baselines`` {scene id: text | None} moves baselines (Accept), ``clear``
        empties it (Accept all); both need ``expected_hermes_revision`` = the record's ``hermes_revision`` they were
        decided on (else review_conflict with the record now). Answers the record with the script ``revision``."""
        project_id, project_dir, identity = self._open(directory, project_id)
        with self.boundary.project_lock(project_dir):
            self._scope(project_dir, identity, project_id)
            current = self._current(project_dir, project_id)
            state = None if current is None else current["state"]
            try:
                record = self._review_read(project_dir, project_id)
            except ScriptError:
                # w14-fix (S4): an unreadable record that is a regular file (bad JSON, wrong shape) is replaced by the
                # empty record on Accept all; there is no readable hermes_revision to compare, so any value is taken.
                # A directory named review.json is a ProjectError and still refuses.
                if not clear or baselines is not None:
                    raise
                if expected_hermes_revision is None:
                    fail("invalid_op", "baselines and clear need expected_hermes_revision (the record's hermes_revision)")
                record = review_rules.empty()
                self._review_write(project_dir, identity, self._ensure_dir(project_dir, "script"), record)
                return {**record, "revision": 0 if current is None else current["revision"]}
            if baselines is not None or clear:
                revision = 0 if current is None else current["revision"]
                review_rules.admit_decision(record, expected_hermes_revision,
                                            {**review_rules.prune(record, state), "revision": revision})
                # w16-scr (SCR-073): a window's baseline computed before the artist's later save is refused.
                review_rules.admit_base(state, baselines, base_revision, {**review_rules.prune(record, state), "revision": revision})
                record = review_rules.apply(record, state, baselines, clear)
                self._review_write(project_dir, identity, self._ensure_dir(project_dir, "script"), record)
            else:
                record = review_rules.prune(record, state)
            return {**record, "revision": 0 if current is None else current["revision"]}

    def revisions(self, directory: str | Path, *, project_id: str) -> dict[str, Any]:
        project_id, project_dir, identity = self._open(directory, project_id)
        with self.boundary.project_lock(project_dir):
            self._scope(project_dir, identity, project_id)
            current = self._current(project_dir, project_id)
            rows = []
            for number, path in self._revision_files(project_dir / "script"):
                envelope = self._file(path, project_id)
                if envelope["revision"] != number:
                    fail("invalid_script", f"{path.name} holds revision {envelope['revision']}")
                rows.append({key: envelope[key] for key in ("revision", "updated_at", "label", "by")})
            return {"revision": 0 if current is None else current["revision"], "revisions": rows}

    def read_revision(self, directory: str | Path, *, project_id: str, revision: int) -> dict[str, Any]:
        if type(revision) is not int or revision < 1:
            fail("invalid_op", "revision must be an integer >= 1")
        project_id, project_dir, identity = self._open(directory, project_id)
        with self.boundary.project_lock(project_dir):
            self._scope(project_dir, identity, project_id)
            path = project_dir / "script" / f"script.json.r{revision}"
            if not os.path.lexists(path):
                fail("revision_unknown", f"revision {revision} is not kept")
            envelope = self._file(path, project_id)
            if envelope["revision"] != revision:
                fail("invalid_script", f"{path.name} holds revision {envelope['revision']}")
            return public(envelope)

    def export(self, directory: str | Path, *, project_id: str, name: str, data: bytes) -> str:
        """Write one export file into ``<project>/exports/script/``; answers the relative path."""
        project_id, project_dir, identity = self._open(directory, project_id)
        safe = _EXPORT_NAME_RE.sub("-", name).strip(" .-")[:120] or "script"
        with self.boundary.project_lock(project_dir):
            self._scope(project_dir, identity, project_id)
            exports = self._ensure_dir(project_dir, "exports")
            folder = self._ensure_dir(exports, "script")
            self._put(project_dir, identity, folder, safe, data)
            return f"exports/script/{safe}"


__all__ = ["ERRORS", "KEEP_REVISIONS", "MAX_BYTES", "ScriptStore", "check", "error_body", "loads", "payload", "public"]
