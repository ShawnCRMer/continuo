"""Who owns an image context: scratch, or exactly one project.

The image state store is profile-wide and its record schema has no owner
field, so ownership is kept beside it in a small index instead of inside the
records (the store schema is a frozen contract).  Every scope reads the store
only through :class:`ScopedImageStates`, which filters and guards by this index.

Rules:

- an id is claimed for its owner BEFORE the context is published, so a crash
  in between leaves at most an unused claim, never an unowned context;
- a context with no claim was made before owners existed.  Generation only
  ever worked in scratch then, so it belongs to scratch - kept, not lost, and
  not visible to any project;
- a claim is never moved to another owner.
"""
from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

SCRATCH_OWNER = "film_owner:scratch"
_OWNER_RE = re.compile(r"^film_owner:(scratch|prj_[0-9a-f]{32})$")
_CONTEXT_RE = re.compile(r"^imgctx_[0-9a-f]{32}$")


class ImageContextOwnerError(ValueError):
    """Finite failure: the context is not this scope's, or the index is unusable."""

    def __init__(self, code: str) -> None:
        self.code = code
        self.reason_code = code
        super().__init__(code)


def _owner(value: Any) -> str:
    if type(value) is not str or _OWNER_RE.fullmatch(value) is None:
        raise ImageContextOwnerError("owner-invalid")
    return value


def _context(value: Any) -> str:
    if type(value) is not str or _CONTEXT_RE.fullmatch(value) is None:
        raise ImageContextOwnerError("context-invalid")
    return value


class ImageContextOwnerIndex:
    """Durable context -> owner map (SQLite, WAL, FULL sync, like the job ledger)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._closed = False
        connection = self._connect()
        try:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS image_context_owners ("
                " context_id TEXT PRIMARY KEY,"
                " owner_scope TEXT NOT NULL)"
            )
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        if self._closed:
            raise ImageContextOwnerError("index-closed")
        connection = sqlite3.connect(self.path, timeout=10.0, isolation_level=None, check_same_thread=False)
        if str(connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower() != "wal":
            connection.close()
            raise ImageContextOwnerError("index-journal")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def owners(self) -> dict[str, str]:
        with self._lock:
            connection = self._connect()
            try:
                return {row[0]: row[1] for row in connection.execute(
                    "SELECT context_id, owner_scope FROM image_context_owners")}
            finally:
                connection.close()

    def owner_of(self, context_id: str) -> str:
        """The owner; an unclaimed context predates owners and is scratch's."""
        context_id = _context(context_id)
        with self._lock:
            connection = self._connect()
            try:
                row = connection.execute(
                    "SELECT owner_scope FROM image_context_owners WHERE context_id = ?",
                    (context_id,),
                ).fetchone()
            finally:
                connection.close()
        return row[0] if row else SCRATCH_OWNER

    def claim(self, context_id: str, owner_scope: str) -> None:
        context_id, owner_scope = _context(context_id), _owner(owner_scope)
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT owner_scope FROM image_context_owners WHERE context_id = ?",
                    (context_id,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO image_context_owners (context_id, owner_scope) VALUES (?, ?)",
                        (context_id, owner_scope),
                    )
                elif row[0] != owner_scope:
                    connection.execute("ROLLBACK")
                    raise ImageContextOwnerError("context-owned-elsewhere")
                connection.execute("COMMIT")
            except ImageContextOwnerError:
                raise
            except BaseException:
                try:
                    connection.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            finally:
                connection.close()

    def close(self) -> None:
        with self._lock:
            self._closed = True


class ScopedImageStates:
    """The image store as ONE owner sees it.

    Only the operations the Image routes use are exposed.  Everything that
    names a context checks ownership first, so a caller cannot reach another
    scope's work by naming its id.
    """

    __slots__ = ("_store", "_index", "_owner")

    def __init__(self, store: Any, index: ImageContextOwnerIndex, owner_scope: str) -> None:
        self._store = store
        self._index = index
        self._owner = _owner(owner_scope)

    @property
    def owner_scope(self) -> str:
        return self._owner

    def require_owned(self, context_id: str) -> str:
        if self._index.owner_of(context_id) != self._owner:
            raise ImageContextOwnerError("context-not-owned")
        return context_id

    def _mine(self, contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        owners = self._index.owners()
        return [
            context for context in contexts
            if owners.get(context["context_id"], SCRATCH_OWNER) == self._owner
        ]

    def enumerate_contexts(self) -> list[dict[str, Any]]:
        return self._mine(self._store.enumerate_contexts())

    def ensure_working_context(self, operation_id: str) -> dict[str, Any]:
        # Serialised per index so two first submits cannot open two contexts.
        with self._index._lock:
            for state in self.enumerate_contexts():
                works = (state.get("state") or {}).get("works") or []
                if works:
                    return {
                        "context_id": state["context_id"],
                        "work_id": works[0]["work_id"],
                        "draft_generation": (works[0].get("draft") or {}).get("draft_generation", 0),
                    }
            import secrets

            context_id = "imgctx_" + secrets.token_hex(16)
            self._index.claim(context_id, self._owner)
            return self._store.create_working_context(operation_id, context_id)

    def load_current_owner(self, context_id: str, work_id: str, basis_kind: str, revision_id: str | None = None) -> dict[str, Any]:
        self.require_owned(context_id)
        return self._store.load_current_owner(context_id, work_id, basis_kind, revision_id)

    def keep_draft_as_revision(self, context_id: str, work_id: str, name: str, operation_id: str) -> dict[str, Any]:
        self.require_owned(context_id)
        return self._store.keep_draft_as_revision(context_id, work_id, name, operation_id)

    def keep_generated(self, context_id: str, basis: dict[str, Any], name: str, artifact: dict[str, Any], intent: str) -> dict[str, Any]:
        self.require_owned(context_id)
        return self._store.keep_generated(context_id, basis, name, artifact, intent)

    def load(self, context_id: str) -> dict[str, Any]:
        self.require_owned(context_id)
        return self._store.load(context_id)


__all__ = [
    "SCRATCH_OWNER",
    "ImageContextOwnerError",
    "ImageContextOwnerIndex",
    "ScopedImageStates",
]
