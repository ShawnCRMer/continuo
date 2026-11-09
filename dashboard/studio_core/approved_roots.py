"""Approved folders are decided by the backend, never by the request (Project charter decision 4).

The artist approves folders by hand: for a project in its manifest (``approved_roots``, written only by
``PUT /projects/metadata``), without a project in the Loose-work list (written only by
``PUT /projects/loose/folders``). A request may still name roots (the page sends them with its
``project_ref``), but it can only NARROW the server's list: a request root that is not the server's root or inside one is dropped.
For Hermes (caller ``"hermes"``) a dropped root refuses the whole request with ``folder_not_approved``.
Nothing is cached: every call reads the manifest / list file again, so a removed folder is refused on
the very next request.

The server's list (owner rule, charter decision 4; the wave-1 union of every Recent project is gone, p5b):
- a request WITH ``project_ref``: that project's manifest ``approved_roots`` and nothing else;
- a request WITHOUT ``project_ref``: the Loose-work list and nothing else. A folder approved only in a
  project is readable only by a request that names that project (the page sends ``project_ref`` with
  every rootful request while a project is open).
Request roots only narrow: a root outside the server's list is dropped for the artist and refused
``folder_not_approved`` for Hermes. The backend's own cache folders stay allowed.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

REASON_CODE = "folder_not_approved"
MESSAGE = "Folder not approved."
LOOSE_SCHEMA = "studio.loose-folders/1"
LOOSE_FILE_NAME = "loose-work-folders.json"
LOOSE_MAX_ROOTS = 64

_LOG = logging.getLogger("film_suite.approved_roots")
_LOOSE_LOCK = threading.RLock()


class FolderNotApproved(ValueError):
    """A request named a folder the artist did not approve (Hermes), or read a file in one."""

    reason_code = REASON_CODE

    def __init__(self, message: str = MESSAGE) -> None:
        super().__init__(message)


def canonical(root: Any) -> str | None:
    """Comparable form of a local folder path, or None for anything that is not one."""
    if not isinstance(root, str) or not root.strip():
        return None
    text = root.strip()
    if text.startswith(("\\\\", "//")):
        return None
    try:
        return os.path.normcase(os.path.realpath(os.path.expanduser(text)))
    except (OSError, ValueError):
        return None


def within(child: str, parent: str) -> bool:
    """True when canonical ``child`` is ``parent`` or inside it."""
    try:
        return os.path.normcase(os.path.commonpath([child, parent])) == parent
    except ValueError:
        return False


def narrow(request_roots: Iterable[Any], server_roots: Iterable[Any], owned_roots: Iterable[Any] = ()) -> tuple[list[str], list[str]]:
    """Split the request's roots into (kept, dropped).

    Kept: a request root equal to or inside a server root or a backend-owned folder (request order, unique).
    Dropped: everything else, including text that is not a local folder path.
    """
    allowed = [c for c in (canonical(r) for r in [*server_roots, *owned_roots]) if c]
    kept: list[str] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for root in request_roots or ():
        key = canonical(root)
        if key is not None and any(within(key, parent) for parent in allowed):
            if key not in seen:
                seen.add(key)
                kept.append(root.strip())
        else:
            dropped.append(root if isinstance(root, str) else repr(root))
    return kept, dropped


def path_refused(path: Any, kept: Iterable[str], dropped: Iterable[str], owned_roots: Iterable[Any] = ()) -> bool:
    """A file under a dropped root and under no kept / owned root (the artist read a removed folder)."""
    target = canonical(path)
    if target is None:
        return False
    if any(within(target, c) for c in (canonical(r) for r in [*kept, *owned_roots]) if c):
        return False
    return any(within(target, c) for c in (canonical(r) for r in dropped) if c)


class LooseWorkFolders:
    """The server-side Loose-work list: one small JSON file under the plugin's state folder.

    ``exists()`` is False until the artist writes it (route or one-time migration); Hermes never creates it.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.is_file()

    def read(self) -> dict[str, Any]:
        with _LOOSE_LOCK:
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return {"schema": LOOSE_SCHEMA, "revision": 0, "approved_roots": []}
            except (OSError, ValueError):
                # An unreadable list approves nothing (fail closed); the artist writes it again by hand.
                return {"schema": LOOSE_SCHEMA, "revision": 0, "approved_roots": [], "unreadable": True}
        roots = raw.get("approved_roots") if isinstance(raw, dict) else None
        revision = raw.get("revision") if isinstance(raw, dict) else None
        if not isinstance(roots, list) or not all(isinstance(r, str) for r in roots) or type(revision) is not int:
            return {"schema": LOOSE_SCHEMA, "revision": 0, "approved_roots": [], "unreadable": True}
        return {"schema": LOOSE_SCHEMA, "revision": revision, "approved_roots": list(roots)}

    def write(self, roots: list[str], *, expected_revision: int | None) -> dict[str, Any]:
        """Replace the list (compare-and-set on the revision when given). Returns the new document."""
        if len(roots) > LOOSE_MAX_ROOTS:
            raise ValueError("too many approved folders")
        with _LOOSE_LOCK:
            current = self.read()
            if expected_revision is not None and expected_revision != current["revision"]:
                raise LooseConflict(current["revision"])
            document = {"schema": LOOSE_SCHEMA, "revision": current["revision"] + 1, "approved_roots": list(roots)}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp = tempfile.mkstemp(prefix=".loose-", suffix=".json", dir=str(self.path.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(document, handle, ensure_ascii=False, indent=2)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp, self.path)
            except BaseException:
                try:
                    os.unlink(temp)
                except OSError:
                    pass
                raise
            return document

    def seed(self, roots: list[str]) -> bool:
        """One-time migration: create the list with exactly ``roots`` if it does not exist yet."""
        with _LOOSE_LOCK:
            if self.exists():
                return False
            self.write(list(dict.fromkeys(roots)), expected_revision=None)
        _LOG.info("Loose-work approved folders created from the page's first request (%d folder(s))", len(roots))
        return True


class LooseConflict(Exception):
    def __init__(self, revision: int) -> None:
        super().__init__("loose-work folders changed since this revision")
        self.revision = revision


def effective_roots(
    request_roots: Iterable[Any],
    project_ref: Any,
    caller: str,
    *,
    project_roots: Callable[[Any], list[str]],
    known_project_roots: Callable[[], list[str]],
    loose: LooseWorkFolders,
    owned_roots: Iterable[Any] = (),
) -> tuple[list[str], list[str]]:
    """The roots a request may use: the request's list narrowed to the server's list.

    With ``project_ref``: the server's list is that project's manifest roots (read now).
    Without: the Loose-work list only. ``known_project_roots`` serves only the one-time seed: while the
    Loose-work list does not exist, the artist's first request that carries roots creates it with those of
    its roots no project approves (possibly none), so a project's folder never becomes a Loose-work folder
    by migration; Hermes never creates or changes it.
    Raises ``FolderNotApproved`` when Hermes names a folder outside the server's list.
    """
    requested = [r for r in (request_roots or ())]
    owned = list(owned_roots)
    if project_ref is not None:
        server = list(project_roots(project_ref))
    else:
        if caller != "hermes" and requested and not loose.exists():
            _kept, unknown = narrow(requested, list(known_project_roots()), owned)
            loose.seed([r for r in unknown if canonical(r)])
        server = list(loose.read()["approved_roots"])
    kept, dropped = narrow(requested, server, owned)
    if dropped and caller == "hermes":
        raise FolderNotApproved()
    return kept, dropped
