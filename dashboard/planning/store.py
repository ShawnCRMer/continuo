"""The planning documents of one project behind ProjectStore's boundary (lane w20-plan2, architecture review finding 2).

state/shot-plan.json (studio.shot-plan/1), state/elements.json (studio.elements/1) and state/previz.json (studio.previz/1)
are read and published ONLY here, the way dashboard/script/store.py does it for the script: the project's cross-process
lock (PathBoundary.project_lock, a named mutex on Windows, flock elsewhere), the pinned project folder revalidated and
its manifest checked against the request's project id, bounded reads (4 MiB) that reject a BOM, duplicate keys and
NaN/Infinity, a full nested check of every document (each pure module's ``check_document``), and ProjectStore's atomic
create/replace (staged temp, identity checks, fsync, read-back). A malformed document is a named refusal
(``planning_document_invalid``) and is never reset or overwritten; a missing one reads as empty.

    with PlanningStore(project_store).open(directory, project_id) as tx:
        plan = tx.load("shot-plan.json")          # checked, or empty
        ...
        tx.save("shot-plan.json", plan)            # inside the same lock

Pure modules keep the rules; this module owns no rule but "what is on disk is well-formed".
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator

from . import elements, previz, shot_plan

MAX_BYTES = 4 * 1024 * 1024
DOCUMENTS = {"shot-plan.json": shot_plan, "elements.json": elements, "previz.json": previz}


class PlanningStoreRefusal(Exception):
    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code, self.message = reason_code, message


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate key")
        out[key] = value
    return out


def _no_constant(_value: str) -> None:
    raise ValueError("NaN or Infinity")


def loads(name: str, raw: bytes) -> dict[str, Any]:
    """Parse and fully check one stored document; any problem is planning_document_invalid naming the file."""
    if len(raw) > MAX_BYTES:
        raise PlanningStoreRefusal("planning_document_too_large", f"{name} is larger than 4 MiB; it was left as it is.")
    try:
        if raw.startswith(b"\xef\xbb\xbf"):
            raise ValueError("BOM")
        doc = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicates, parse_constant=_no_constant)
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise PlanningStoreRefusal("planning_document_invalid", f"{name} is not valid JSON; it was left as it is.") from None
    problem = DOCUMENTS[name].check_document(doc)
    if problem:
        raise PlanningStoreRefusal("planning_document_invalid", f"{name} is damaged ({problem}); it was left as it is.")
    return doc


def dumps(name: str, doc: dict[str, Any]) -> bytes:
    problem = DOCUMENTS[name].check_document(doc)
    if problem:  # a rule bug never reaches the disk
        raise PlanningStoreRefusal("planning_document_invalid", f"{name} would be invalid ({problem}); nothing was saved.")
    data = json.dumps(doc, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(data) > MAX_BYTES:
        raise PlanningStoreRefusal("planning_document_too_large", f"{name} would exceed 4 MiB; nothing was saved.")
    return data


class _Transaction:
    def __init__(self, owner: "PlanningStore", project_dir: Path, identity: Any, project_id: str) -> None:
        self.owner, self.project_dir, self.identity, self.project_id = owner, project_dir, identity, project_id

    def load(self, name: str) -> dict[str, Any]:
        ps, boundary = self.owner.ps, self.owner.boundary
        target = self.project_dir / "state" / name
        if not os.path.lexists(target):
            return DOCUMENTS[name].empty()
        if target.is_symlink() or not target.is_file():
            raise PlanningStoreRefusal("planning_document_invalid", f"{name} is not a plain file; it was left as it is.")
        try:
            raw = ps._read_regular(boundary, target, max_bytes=MAX_BYTES)
        except ps.ProjectError as exc:
            if "too large" in str(exc):
                raise PlanningStoreRefusal("planning_document_too_large", f"{name} is larger than 4 MiB; it was left as it is.") from None
            raise
        return loads(name, raw)

    def save(self, name: str, doc: dict[str, Any]) -> None:
        self._publish(name, dumps(name, doc))

    def _publish(self, name: str, data: bytes, max_bytes: int = MAX_BYTES * 2) -> None:
        ps, boundary, store = self.owner.ps, self.owner.boundary, self.owner.store
        boundary.revalidate_directory(self.project_dir, self.identity)
        state_dir = store._ensure_state_dir(self.project_dir)
        state_identity = boundary.pin_directory(state_dir)
        target = state_dir / name
        try:
            if os.path.lexists(target):
                store._atomic_replace_file(state_dir, target, data, max_bytes=max_bytes, expected_directory_identity=state_identity)
            else:
                store._atomic_create_file(self.project_dir, state_dir, target, data, project_identity=self.identity, directory_identity=state_identity)
        except ps.ProjectError as exc:
            if exc.code.startswith("music_"):  # ProjectStore's create primitive names its music origin; ours is planning
                raise ps.ProjectError("project_io_failed", f"{name} could not be saved") from exc
            raise

    def _raw(self, name: str, max_bytes: int = MAX_BYTES) -> bytes | None:
        target = self.project_dir / "state" / name
        if not os.path.lexists(target):
            return None
        return self.owner.ps._read_regular(self.owner.boundary, target, max_bytes=max_bytes)

    # Lane w22-pv (SCR-082, review codex3 F5): several planning documents commit both-or-neither. The old bytes go into
    # one journal first (its publication is the commit's start), then each document is published, then the journal is
    # marked done (the commit point). A failure in between rolls the published ones back at once; a process that dies in
    # between leaves the journal pending and the next open of the planning files restores the complete old state.
    def commit(self, docs: dict[str, dict[str, Any]]) -> None:
        names = [n for n in ("shot-plan.json", "elements.json", "previz.json") if n in docs]
        if len(names) == 1:
            self.save(names[0], docs[names[0]])
            return
        payloads = {n: dumps(n, docs[n]) for n in names}
        old = {}
        for n in names:
            raw = self._raw(n)
            old[n] = None if raw is None else raw.decode("utf-8")
        self._publish(JOURNAL, json.dumps({"schema": JOURNAL_SCHEMA, "state": "pending", "old": old}).encode("utf-8"), MAX_BYTES * 8)
        published: list[str] = []
        try:
            for n in names:
                self.save(n, docs[n])
                published.append(n)
        except Exception:
            _restore(self, {n: old[n] for n in published})
            self._publish(JOURNAL, json.dumps({"schema": JOURNAL_SCHEMA, "state": "done"}).encode("utf-8"), MAX_BYTES * 8)
            raise
        self._publish(JOURNAL, json.dumps({"schema": JOURNAL_SCHEMA, "state": "done"}).encode("utf-8"), MAX_BYTES * 8)


JOURNAL = "planning-commit.json"
JOURNAL_SCHEMA = "studio.planning-commit/1"


def _restore(tx: _Transaction, old: dict[str, str | None]) -> None:
    for name, text in old.items():
        tx._publish(name, text.encode("utf-8") if text is not None else dumps(name, DOCUMENTS[name].empty()))


def _recover(tx: _Transaction) -> None:
    """A pending journal (a commit that never reached its commit point): restore every document it names, then close it."""
    raw = tx._raw(JOURNAL, MAX_BYTES * 8)
    if raw is None:
        return
    try:
        journal = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise PlanningStoreRefusal("planning_document_invalid", f"{JOURNAL} is not readable; it was left as it is.") from None
    if not isinstance(journal, dict) or journal.get("schema") != JOURNAL_SCHEMA or journal.get("state") not in ("pending", "done"):
        raise PlanningStoreRefusal("planning_document_invalid", f"{JOURNAL} is invalid; it was left as it is.")
    if journal["state"] == "done":
        return
    old = journal.get("old")
    if not isinstance(old, dict) or not all(n in DOCUMENTS and (v is None or isinstance(v, str)) for n, v in old.items()):
        raise PlanningStoreRefusal("planning_document_invalid", f"{JOURNAL} is invalid; it was left as it is.")
    for name, text in old.items():
        if text is not None:
            loads(name, text.encode("utf-8"))  # the old document is checked before it is published back
    _restore(tx, old)
    tx._publish(JOURNAL, json.dumps({"schema": JOURNAL_SCHEMA, "state": "done"}).encode("utf-8"), MAX_BYTES * 8)


class PlanningStore:
    """The planning files of one ProjectStore (its boundary, lock and atomic primitives)."""

    def __init__(self, store: Any) -> None:
        self.store = store
        self.boundary = store.boundary
        self.ps = sys.modules[type(store).__module__]  # project_store: the shared primitives

    @contextlib.contextmanager
    def open(self, directory: str | Path, project_id: Any) -> Iterator[_Transaction]:
        project_id = self.ps._project_id(project_id)
        project_dir = self.store._directory(directory)
        identity = self.boundary.pin_directory(project_dir)
        with self.boundary.project_lock(project_dir):
            self.boundary.revalidate_directory(project_dir, identity)
            if self.store._manifest_locked(project_dir)["project_id"] != project_id:
                raise self.ps.ProjectError("project_scope_mismatch", "the plan belongs to another project")
            tx = _Transaction(self, project_dir, identity, project_id)
            _recover(tx)  # w22-pv SCR-082: a commit a dead process left half-way is rolled back first
            yield tx
