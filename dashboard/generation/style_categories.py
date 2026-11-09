"""Style categories: one small durable JSON document per project, plus one
per user ("My categories", ``user.json``) that always starts with the
built-in Favourites category (fixed id and name, never dropped).

Documents live under the plugin state root (``style-categories/<project_id>.json``),
never in the project folder, so project manifests and shoot state stay
byte-identical. Writes are whole-document compare-and-swap on ``revision``:
staged to an owned temp name, fsynced, published with ``os.replace``, then
read back. Old owned temps are swept on read. Names derive only from the
validated project id; nothing a caller sends becomes a path.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import stat
import threading
import time
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "studio.style-categories/1"
PROJECT_ID_RE = re.compile(r"^prj_[0-9a-f]{32}$")
CATEGORY_ID_RE = re.compile(r"^cat_[0-9a-f]{12}$")
BOARD_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
TEMP_RE = re.compile(r"^\.cat-[0-9a-f]{32}\.tmp$")
MAX_CATEGORIES = 64
MAX_BOARDS = 2000
NAME_MAX = 60
DOC_MAX_BYTES = 1024 * 1024
TEMP_MAX_AGE_S = 24 * 60 * 60
USER = "user"
FAVOURITES_ID = "cat_000000000000"
FAVOURITES_NAME = "Favourites"


class StyleCategoryError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _invalid(message: str) -> None:
    raise StyleCategoryError("invalid_style_categories", message)


def new_category_id() -> str:
    return "cat_" + secrets.token_hex(6)


def normalize(categories: Any, known_board: Any = None) -> list[dict[str, Any]]:
    """Validate a caller's list; names unique (case-insensitive), ids stable."""
    if not isinstance(categories, list) or len(categories) > MAX_CATEGORIES:
        _invalid("categories must be a list of at most 64 entries")
    out: list[dict[str, Any]] = []
    names: set[str] = set()
    ids: set[str] = set()
    for raw in categories:
        if not isinstance(raw, dict) or set(raw) - {"id", "name", "boards"}:
            _invalid("category entries have id, name and boards only")
        name = re.sub(r"\s+", " ", str(raw.get("name") if isinstance(raw.get("name"), str) else "")).strip()
        if not name or len(name) > NAME_MAX:
            _invalid("category names are 1 to 60 characters")
        if name.casefold() in names:
            raise StyleCategoryError("style_category_exists", "A category with this name already exists")
        names.add(name.casefold())
        category_id = raw.get("id")
        if category_id is None:
            category_id = new_category_id()
        if not isinstance(category_id, str) or not CATEGORY_ID_RE.fullmatch(category_id) or category_id in ids:
            _invalid("category id is invalid")
        ids.add(category_id)
        boards_raw = raw.get("boards", [])
        if not isinstance(boards_raw, list) or len(boards_raw) > MAX_BOARDS:
            _invalid("category boards must be a list of at most 2000 ids")
        boards: list[str] = []
        for board in boards_raw:
            if not isinstance(board, str) or not BOARD_ID_RE.fullmatch(board):
                _invalid("board id is invalid")
            if known_board is not None and not known_board(board):
                raise StyleCategoryError("style_unknown", "The style is unknown")
            if board not in boards:
                boards.append(board)
        out.append({"id": category_id, "name": name, "boards": boards})
    return out


def _with_favourites(categories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The user document: Favourites first, with its fixed name; nothing else may take that name."""
    favourite = next((item for item in categories if item["id"] == FAVOURITES_ID), None)
    rest = [item for item in categories if item["id"] != FAVOURITES_ID]
    if any(item["name"].casefold() == FAVOURITES_NAME.casefold() for item in rest):
        raise StyleCategoryError("style_category_exists", "A category with this name already exists")
    return [{"id": FAVOURITES_ID, "name": FAVOURITES_NAME, "boards": favourite["boards"] if favourite else []}, *rest]


def _empty(project_id: str) -> dict[str, Any]:
    categories = _with_favourites([]) if project_id == USER else []
    return {"schema": SCHEMA, "project_id": project_id, "revision": 0, "categories": categories}


def _encode(document: dict[str, Any]) -> bytes:
    return (json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


class StyleCategoryStore:
    _locks: dict[str, threading.Lock] = {}
    _locks_guard = threading.Lock()

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _lock(self, project_id: str) -> threading.Lock:
        key = str(self.root) + "\0" + project_id
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    def _target(self, project_id: str) -> Path:
        if project_id == USER:
            return self.root / "user.json"
        if not isinstance(project_id, str) or not PROJECT_ID_RE.fullmatch(project_id):
            _invalid("project id is invalid")
        return self.root / f"{project_id}.json"

    def _directory(self, create: bool) -> Path | None:
        if not os.path.lexists(self.root):
            if not create:
                return None
            try:
                self.root.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise StyleCategoryError("style_categories_unavailable", "Categories are unavailable") from exc
        if self.root.is_symlink() or not self.root.is_dir():
            raise StyleCategoryError("style_categories_unavailable", "Categories are unavailable")
        if os.name != "nt":
            os.chmod(self.root, 0o700)
        return self.root

    def _sweep(self, directory: Path) -> None:
        cutoff = time.time() - TEMP_MAX_AGE_S
        try:
            entries = list(directory.iterdir())
        except OSError:
            return
        for entry in entries:
            if not TEMP_RE.fullmatch(entry.name):
                continue
            try:
                info = entry.lstat()
                if stat.S_ISREG(info.st_mode) and info.st_mtime < cutoff:
                    entry.unlink()
            except OSError:
                continue

    def _read_locked(self, project_id: str, *, sweep: bool) -> dict[str, Any]:
        target = self._target(project_id)
        directory = self._directory(create=False)
        if directory is None:
            return _empty(project_id)
        if sweep:
            self._sweep(directory)
        if not os.path.lexists(target):
            return _empty(project_id)
        try:
            info = target.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_size > DOC_MAX_BYTES:
                raise StyleCategoryError("style_categories_unavailable", "Categories are unavailable")
            with open(target, "rb") as handle:
                raw = handle.read(DOC_MAX_BYTES + 1)
        except OSError as exc:
            raise StyleCategoryError("style_categories_unavailable", "Categories are unavailable") from exc
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise StyleCategoryError("style_categories_unavailable", "Saved categories are unreadable") from exc
        if (not isinstance(value, dict) or value.get("schema") != SCHEMA or value.get("project_id") != project_id
                or type(value.get("revision")) is not int or value["revision"] < 1):
            raise StyleCategoryError("style_categories_unavailable", "Saved categories are unreadable")
        try:
            categories = normalize(value.get("categories"))
            if project_id == USER:
                categories = _with_favourites(categories)
        except StyleCategoryError as exc:
            raise StyleCategoryError("style_categories_unavailable", "Saved categories are unreadable") from exc
        return {"schema": SCHEMA, "project_id": project_id, "revision": value["revision"], "categories": categories}

    def read(self, project_id: str) -> dict[str, Any]:
        with self._lock(project_id):
            return self._read_locked(project_id, sweep=True)

    def write(self, project_id: str, expected_revision: int, categories: Iterable[Any], known_board: Any = None) -> dict[str, Any]:
        if type(expected_revision) is not int or expected_revision < 0:
            _invalid("expected revision is invalid")
        normalized = normalize(list(categories) if not isinstance(categories, list) else categories, known_board)
        if project_id == USER:
            normalized = _with_favourites(normalized)
        with self._lock(project_id):
            current = self._read_locked(project_id, sweep=False)
            if current["revision"] != expected_revision:
                raise StyleCategoryError("style_categories_conflict", "Categories changed elsewhere")
            document = {"schema": SCHEMA, "project_id": project_id, "revision": expected_revision + 1, "categories": normalized}
            payload = _encode(document)
            if len(payload) > DOC_MAX_BYTES:
                _invalid("categories are too large")
            directory = self._directory(create=True)
            target = self._target(project_id)
            temp = directory / (".cat-" + secrets.token_hex(16) + ".tmp")
            try:
                with open(temp, "xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp, target)
                with open(target, "rb") as handle:
                    if handle.read(DOC_MAX_BYTES + 1) != payload:
                        raise StyleCategoryError("style_categories_unavailable", "Categories could not be saved")
                if os.name != "nt":
                    fd = os.open(directory, os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
            except OSError as exc:
                raise StyleCategoryError("style_categories_unavailable", "Categories could not be saved") from exc
            finally:
                try:
                    if os.path.lexists(temp):
                        temp.unlink()
                except OSError:
                    pass
            return document
