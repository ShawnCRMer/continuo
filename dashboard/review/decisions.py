"""Take A/B review: comparisons, per-frame notes and decisions, per project (lane w5-review).

One append-only JSON Lines file per project (``<state>/review/<project_id>.jsonl``): every line is
one record and no line is ever rewritten. Three kinds:

* ``comparison``: two takes of one shot opened side by side (take ids + sha256 of both files, blind
  yes/no, the presented order: which take sits in slot A and which in slot B, shuffled when blind).
* ``note``: a note on one frame of an open comparison ``{frame, text}``.
* ``decision``: "Prefer A/B" for a comparison: the chosen take, both sha256, blind, the notes made
  while comparing, time and by (artist | hermes). A comparison has at most one decision.

The UI and the agent call the same routes (routes/projects/review_1.py), so a comparison Hermes
starts is the one the artist sees, and the other way round.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = "studio.take-review/1"
RECORD_MAX = 20_000  # lines read back per project; older lines stay in the file
NOTE_TEXT_MAX = 500
_PROJECT_RE = re.compile(r"^prj_[0-9a-f]{32}$")
_LOCK = threading.Lock()


class ReviewError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "retryable": False}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class ReviewStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, project_id: str) -> Path:
        if not _PROJECT_RE.fullmatch(project_id):
            raise ReviewError("invalid_project", "The project id is invalid.")
        return self.root / f"{project_id}.jsonl"

    def records(self, project_id: str) -> list[dict[str, Any]]:
        path = self._path(project_id)
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except ValueError:
                    continue  # a torn last line (crash mid-append) is skipped, never repaired in place
                if isinstance(value, dict) and value.get("kind") in {"comparison", "note", "decision"}:
                    rows.append(value)
        return rows[-RECORD_MAX:]

    def _append(self, project_id: str, record: dict[str, Any]) -> dict[str, Any]:
        path = self._path(project_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with open(path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        return record

    # -- the three verbs ------------------------------------------------------------------------

    def start(self, project_id: str, *, shot: str, take_a: dict[str, Any], take_b: dict[str, Any], blind: bool, by: str) -> dict[str, Any]:
        """Open a comparison. ``take_*`` = {id, label, sha256}; blind shuffles which take is slot A."""
        if take_a["id"] == take_b["id"]:
            raise ReviewError("same_take", "Both sides are the same take.")
        pair = [take_a, take_b]
        swapped = bool(blind and secrets.randbelow(2))
        if swapped:
            pair.reverse()
        record = {
            "schema": SCHEMA, "kind": "comparison", "comparison_id": "cmp_" + secrets.token_hex(16),
            "shot": shot, "blind": bool(blind), "by": by, "at": round(time.time(), 3),
            # slot A and slot B as presented; ``picked`` keeps the order the takes were named in.
            "a": dict(pair[0]), "b": dict(pair[1]), "picked": [take_a["id"], take_b["id"]],
        }
        with _LOCK:
            return self._append(project_id, record)

    def _comparison(self, rows: list[dict[str, Any]], comparison_id: str) -> dict[str, Any]:
        found = next((r for r in rows if r["kind"] == "comparison" and r.get("comparison_id") == comparison_id), None)
        if found is None:
            raise ReviewError("comparison_unknown", "That comparison is not in this project.")
        return found

    def note(self, project_id: str, comparison_id: str, *, frame: int, text: str, by: str) -> dict[str, Any]:
        text = " ".join(str(text).split())[:NOTE_TEXT_MAX]
        if not text:
            raise ReviewError("empty_note", "A note needs text.")
        with _LOCK:
            rows = self.records(project_id)
            cmp_ = self._comparison(rows, comparison_id)
            record = {"schema": SCHEMA, "kind": "note", "note_id": "nte_" + secrets.token_hex(8), "comparison_id": comparison_id,
                      "shot": cmp_["shot"], "frame": int(frame), "text": text, "by": by, "at": round(time.time(), 3)}
            return self._append(project_id, record)

    def decide(self, project_id: str, comparison_id: str, *, chosen: str, by: str) -> dict[str, Any]:
        """``chosen`` is the slot ("a" | "b"); the record names the take and both files' sha256."""
        with _LOCK:
            rows = self.records(project_id)
            cmp_ = self._comparison(rows, comparison_id)
            if any(r["kind"] == "decision" and r.get("comparison_id") == comparison_id for r in rows):
                raise ReviewError("already_decided", "This comparison already has a decision.")
            notes = [{"frame": r["frame"], "text": r["text"], "by": r["by"]} for r in rows
                     if r["kind"] == "note" and r.get("comparison_id") == comparison_id]
            take = cmp_[chosen]
            record = {
                "schema": SCHEMA, "kind": "decision", "comparison_id": comparison_id, "shot": cmp_["shot"],
                "take_ids": [cmp_["a"]["id"], cmp_["b"]["id"]], "sha256": [cmp_["a"]["sha256"], cmp_["b"]["sha256"]],
                "blind": cmp_["blind"], "chosen_slot": chosen, "chosen": take["id"], "chosen_sha256": take["sha256"],
                "notes": notes, "by": by, "at": round(time.time(), 3),
            }
            return self._append(project_id, record)

    # -- read -----------------------------------------------------------------------------------

    def view(self, project_id: str, *, shot: str | None = None) -> dict[str, Any]:
        """Comparisons (newest first) with their notes and decision; ``open`` = undecided ones."""
        rows = self.records(project_id)
        if shot is not None:
            rows = [r for r in rows if r.get("shot") == shot]
        decisions = {r["comparison_id"]: r for r in rows if r["kind"] == "decision"}
        notes: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            if r["kind"] == "note":
                notes.setdefault(r["comparison_id"], []).append({k: r[k] for k in ("note_id", "frame", "text", "by", "at")})
        comparisons = []
        for r in reversed(rows):
            if r["kind"] != "comparison":
                continue
            comparisons.append({**r, "notes": notes.get(r["comparison_id"], []), "decision": decisions.get(r["comparison_id"])})
        return {"schema": SCHEMA, "project_id": project_id, "comparisons": comparisons,
                "open": [c["comparison_id"] for c in comparisons if c["decision"] is None],
                "decisions": [d for d in reversed(rows) if d["kind"] == "decision"]}
