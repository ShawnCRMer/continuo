"""Shots of ``studio.script/1``: stable ids for the Shot lines of a scene (board E21).

A Shot is a lone upper-case line of a scene (ANGLE ON, CLOSE ON THE LETTER, ...), read by
exactly the rule of the editor's classifier (desktop/film-suite/src/script/editor/fountain.js
``read``/``lone``); the backend's ``fountain.elements`` folds it into Action, so this module is
the one backend reader of Shots.

Each scene holds ``shots``: one id ``sh-<8 hex>`` per Shot line, in text order. The ids live
in the document, never in the scene text. Numbering ``<scene number>.<n>`` is display only.
An id survives:

* edits of other lines and of the shot's own text: ``reconcile`` aligns the old Shot lines
  with the new ones (equal lines keep their id, an edited line in place keeps its id, a
  new line gets a new id);
* undo/redo in the editor: the editor carries the id on the line and sends ``shots``
  (``given``) with the scene text, so a shot that comes back keeps its id;
* Fountain and FDX: ``[[shot sh-xxxxxxxx]]`` at the end of the Shot line (a Fountain note:
  other programs hide it, the PDF never prints scene notes) and the FDX paragraph
  ``<Paragraph Type="Shot" ContinuoShot="sh-...">``.

A scene stored before shot ids existed has none; ``ids_of`` derives stable ones for it
(``sh-`` + sha256(scene id, n)), the same on every read until the scene is next written.
"""
from __future__ import annotations

import difflib
import hashlib
import re
import secrets
from typing import Any, Iterable

from .schema import is_heading

SHOT_ID_RE = re.compile(r"^sh-[0-9a-f]{8}$")
SHOT_NOTE_RE = re.compile(r"\s*\[\[\s*shot\s+(sh-[0-9a-f]{8})\s*\]\]\s*$")
MAX_SHOTS = 500  # per scene

_TRANSITION_RE = re.compile(r"^[^a-z]*TO:$")
_EXTENSION_RE = re.compile(r"\s*\([^()]*\)")
_SHOT_RE = re.compile(r"^(?:ANGLE ON|ANGLE|CLOSE ON|CLOSE UP|CLOSE-UP|CLOSEUP|EXTREME CLOSE|ECU|CU|INSERT|POV|WIDE ON|WIDE SHOT|WIDE|MEDIUM SHOT|MEDIUM|TWO SHOT|ESTABLISHING|AERIAL|BACK TO|REVERSE ON|REVERSE|TRACKING|OVER THE SHOULDER|OTS|SERIES OF SHOTS|MONTAGE|ON )\b")


def _is_cue(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s.startswith("@"):
        return len(s) > 1
    if s[:1] in "!.>~=#" or s.startswith("[[") or is_heading(s):
        return False
    name = _EXTENSION_RE.sub("", re.sub(r"\^$", "", s).rstrip())
    if ":" in name:
        return False
    return bool(re.search(r"[A-Z]", name)) and not re.search(r"[a-z]", name) and not _TRANSITION_RE.match(s)


def _lone_is_shot(line: str) -> bool:
    """The editor's ``lone``: a lone upper-case line is a Shot, a waiting cue or Action."""
    s = line.strip()
    if not re.search(r"[A-Z]", s) or re.search(r"[a-z]", s) or s[:1] in "!.>~=#@" or _TRANSITION_RE.match(s):
        return False
    if _SHOT_RE.match(s):
        return True
    if s[-1:] in ".!?":
        return False
    return len([w for w in re.split(r"\s+", _EXTENSION_RE.sub("", s)) if w]) > 3


def shot_lines(text: str) -> list[tuple[int, str]]:
    """The Shot lines of one scene's Fountain: [(line index in ``text.split('\\n')``, line)]."""
    lines = str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[list[int]] = []
    current: list[int] = []
    for index, line in enumerate(lines):
        if line.strip():
            current.append(index)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    out: list[tuple[int, str]] = []
    seen_any = False
    for block in blocks:
        first = lines[block[0]].strip()
        if is_heading(first) and not seen_any:
            seen_any = True
            block = block[1:]
            if not block:
                continue
            first = lines[block[0]].strip()
        seen_any = True
        if len(block) == 1 and first == "===":
            continue
        if len(block) == 1 and first.startswith(">"):
            continue
        if len(block) == 1 and _TRANSITION_RE.match(first) and not first.startswith("!"):
            continue
        if all(lines[i].strip().startswith("~") for i in block):
            continue
        if len(block) >= 2 and _is_cue(lines[block[0]]):
            continue
        if len(block) == 1 and not lines[block[0]].startswith("!") and _lone_is_shot(first):
            out.append((block[0], first))
    return out


def new_shot_id(taken: set[str]) -> str:
    while True:
        value = f"sh-{secrets.token_hex(4)}"
        if value not in taken:
            taken.add(value)
            return value


def derived_ids(scene_id: str, count: int) -> list[str]:
    """Stable ids for a scene stored before shot ids existed (same on every read)."""
    return [f"sh-{hashlib.sha256(f'{scene_id}:{n}'.encode()).hexdigest()[:8]}" for n in range(1, count + 1)]


def ids_of(scene_id: str, scene: dict[str, Any]) -> list[str]:
    """The scene's shot ids: stored, or derived when the scene predates shot ids."""
    stored = scene.get("shots")
    if isinstance(stored, list):
        return list(stored)
    return derived_ids(scene_id, len(shot_lines(scene["text"])))


def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def reconcile(old_text: str, old_ids: list[str], new_text: str, *, given: list[Any] | None = None,
              taken: set[str] | None = None) -> list[str]:
    """The ids of ``new_text``'s Shot lines.

    ``given`` (the editor's ids, or ids read from ``[[shot ...]]`` notes): one entry per new
    Shot line, an id or None. A well-formed id not taken by another scene is kept; a None
    entry is aligned with the old lines like a write without ids. ``taken``: ids used by the
    other scenes of the document (filled with the answer)."""
    taken = set() if taken is None else taken
    old = [line for _i, line in shot_lines(old_text)]
    new = [line for _i, line in shot_lines(new_text)]
    if len(old_ids) != len(old):
        old_ids = (list(old_ids) + [None] * len(old))[: len(old)]  # type: ignore[list-item]
    out: list[str | None] = [None] * len(new)
    used: set[str] = set()
    if given is not None and len(given) == len(new):
        for j, value in enumerate(given):
            if isinstance(value, str) and SHOT_ID_RE.fullmatch(value) and value not in taken and value not in used:
                out[j] = value
                used.add(value)
    # Align the old lines with the new ones that have no id yet.
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            pairs = list(zip(range(i1, i2), range(j1, j2)))
        elif tag == "replace":
            pairs = []
            free = list(range(i1, i2))
            for j in range(j1, j2):
                best = max(free, key=lambda i: _similar(old[i], new[j]), default=None)
                if best is not None and _similar(old[best], new[j]) >= 0.5:
                    pairs.append((best, j))
                    free = [i for i in free if i > best]
        else:
            pairs = []
        for i, j in pairs:
            value = old_ids[i]
            if out[j] is None and isinstance(value, str) and value not in used and value not in taken:
                out[j] = value
                used.add(value)
    pool = taken | used
    result = [value if value is not None else new_shot_id(pool) for value in out]
    taken.update(result)
    return result


def ids_from_notes(text: str, notes: Iterable[tuple[str, str]]) -> list[str | None] | None:
    """The ids of ``text``'s Shot lines from ``[[shot ...]]`` notes ((stripped line, id) in text
    order), None when there were no notes."""
    notes = list(notes)
    if not notes:
        return None
    out: list[str | None] = []
    for _index, line in shot_lines(text):
        found = next((k for k, (said, _id) in enumerate(notes) if said == line), None)
        if found is None:
            out.append(None)
        else:
            out.append(notes[found][1])
            notes = notes[found + 1:]
    return out


def split_notes(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Scene text without ``[[shot ...]]`` notes, and the notes as (stripped line, id)."""
    lines, notes = [], []
    for line in str(text).split("\n"):
        match = SHOT_NOTE_RE.search(line)
        if match:
            line = line[: match.start()].rstrip()
            notes.append((line.strip(), match.group(1)))
        lines.append(line)
    return "\n".join(lines), notes


def ensure(state: dict[str, Any]) -> dict[str, Any]:
    """Every scene of ``state`` holds ``shots`` (derived ids for scenes that predate them)."""
    for sid, scene in state["scenes"].items():
        if not isinstance(scene.get("shots"), list):
            scene["shots"] = derived_ids(sid, len(shot_lines(scene["text"])))
    return state


def check(state: dict[str, Any], fail: Any) -> None:
    """``shots`` (when present): one unique id per Shot line, unique in the whole document."""
    seen: set[str] = set()
    for sid, scene in state["scenes"].items():
        if "shots" not in scene:
            continue
        ids = scene["shots"]
        where = f"scenes.{sid}.shots"
        if type(ids) is not list or len(ids) > MAX_SHOTS:
            fail("invalid_script", f"{where}: a list of at most {MAX_SHOTS} shot ids")
        for value in ids:
            if type(value) is not str or not SHOT_ID_RE.fullmatch(value):
                fail("invalid_script", f"{where}: shot ids look like sh-1a2b3c4d")
            if value in seen:
                fail("invalid_script", f"{where}: {value} is used twice")
            seen.add(value)
        if len(ids) != len(shot_lines(scene["text"])):
            fail("invalid_script", f"{where}: one id per Shot line of the text")


def listing(state: dict[str, Any], numbers: dict[str, str], order_ids: list[str]) -> list[dict[str, Any]]:
    """``derived.shots``: the shots of the scenes in the order (omitted excluded), in order:
    {id, scene, n, number (<scene number>.<n>, display only), text, tag, legacy_tag}.
    ``tag`` is the Media mark that assigns a picture (``shot:<id>``); ``legacy_tag`` the
    stopgap ``shot:<scene id>.<n>`` it replaces (migrated once on read)."""
    out: list[dict[str, Any]] = []
    for sid in order_ids:
        scene = state["scenes"][sid]
        if scene["omitted"]:
            continue
        ids = ids_of(sid, scene)
        for n, ((_index, line), shot_id) in enumerate(zip(shot_lines(scene["text"]), ids), start=1):
            number = numbers.get(sid)
            out.append({"id": shot_id, "scene": sid, "n": n, "number": f"{number}.{n}" if number else str(n), "text": line,
                        "tag": f"shot:{shot_id}", "legacy_tag": f"shot:{sid}.{n}"})
    return out


__all__ = ["MAX_SHOTS", "SHOT_ID_RE", "SHOT_NOTE_RE", "check", "derived_ids", "ensure", "ids_from_notes", "ids_of",
           "listing", "new_shot_id", "reconcile", "shot_lines", "split_notes"]
