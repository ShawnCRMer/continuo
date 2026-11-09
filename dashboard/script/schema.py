"""``studio.script/1``: exact shape, bounds, invariants and the derived views.

The document is the ``state`` of a project envelope ``{schema, project_id, revision,
updated_at, label, by, state}`` stored as ``<project>/script/script.json``::

    state = {
      "title_page":     [{"key": "Title", "value": "..."}, ...],
      "scenes":         {"s-7f3a19c0": {"text", "synopsis", "card", "number", "numbered", "omitted",
                                        "revision", "by"}},
      "order":          ["s-7f3a19c0", {"break": "act", "id": "a-0c1d2e3f", "title": "ACT TWO"}, ...],
                        (a deeper Fountain section ## … ###### is a break with "level": 2-6)
      "order_revision": 0,
      "numbers_locked": False,
    }

* Content (``scenes``), place in the order (``order``) and card (``card``) are separate:
  reordering never touches a scene's text, and a scene that is not in ``order`` is
  *aside* (derived, never stored twice).
* A scene's ``text`` is Fountain: an optional heading as its first line, then the body.
  IDs, scene numbers, synopsis lines (``=``), sections (``#``) and boneyard markers never
  live in the text; the document holds them.
* ``revision`` is per scene (text, synopsis, omitted); ``order_revision`` covers the order,
  act breaks and numbering. Two writers on different scenes never conflict.
* ``number`` is stored only while numbers are locked (FDX style ``12A``); unlocked numbers
  are derived 1..n over the scenes in the order (an omitted scene keeps its slot).
  ``numbered`` stays true once a scene had a locked number: such a scene is omitted, not
  deleted (delete needs ``force``).
* ``shots`` (optional, lane w5-script): one stable id ``sh-<8 hex>`` per Shot line of the
  text, in text order (``shots.py``); a scene stored before shot ids has none and reads
  derived ids. Numbering ``<scene number>.<n>`` is display only (``derived.shots``).
* ``card`` is null or ``{x, y, revision}``: its own revision, so moving a card never
  conflicts with editing the scene's text.
* Room for dialogue line ids: a later revision of this schema may add an optional
  ``lines`` map per scene (line id -> anchor); unknown keys are refused today, so nothing
  can be stored there by accident before that contract exists.

Characters and places are derived read-only views (``derive``), never stored.
"""
from __future__ import annotations

import math
import re
import secrets
from typing import Any

from . import pages  # the one page count (editor twin: desktop/film-suite/src/script/editor/pages.js)

SCHEMA = "studio.script/1"
MAX_SCENES = 2000
MAX_SCENE_TEXT_BYTES = 64 * 1024
MAX_SYNOPSIS = 2000
MAX_BREAKS = 200
MAX_TITLE = 120
MAX_TITLE_PAGE = 20
MAX_TITLE_KEY = 40
MAX_TITLE_VALUE = 1000
MAX_CARD = 1_000_000
AUTHORS = ("artist", "hermes")

SCENE_ID_RE = re.compile(r"^s-[0-9a-f]{8}$")
BREAK_ID_RE = re.compile(r"^a-[0-9a-f]{8}$")
NUMBER_RE = re.compile(r"^[0-9A-Z]{1,8}$")
ID_NOTE_RE = re.compile(r"\[\[\s*(?:[sa]-[0-9a-f]{8}(?:\s+omitted)?|shot\s+sh-[0-9a-f]{8})\s*\]\]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_TITLE_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 _-]{0,39}$")

# Fountain scene heading: INT./EXT./EST./INT./EXT./I/E (spec), or forced with one leading dot.
HEADING_RE = re.compile(r"^(?:int|ext|est|int\.?/ext|i/e)[. ]", re.IGNORECASE)
NUMBER_IN_HEADING_RE = re.compile(r"\s+#([^#\s]+)#\s*$")


class ScriptError(ValueError):
    """A finite, public-safe refusal: ``code`` is stable, ``message`` names the field."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.reason_code = code
        super().__init__(message)


def fail(code: str, message: str) -> None:
    raise ScriptError(code, message)


def default_state() -> dict[str, Any]:
    return {"title_page": [], "scenes": {}, "order": [], "order_revision": 0, "numbers_locked": False}


def new_id(prefix: str, taken: set[str]) -> str:
    while True:
        value = f"{prefix}-{secrets.token_hex(4)}"
        if value not in taken:
            return value


def is_heading(line: str) -> bool:
    stripped = line.strip()
    if stripped.startswith(".") and not stripped.startswith("..") and len(stripped) > 1:
        return True
    return HEADING_RE.match(stripped) is not None


def heading_of(text: str) -> str | None:
    """The scene's heading (first non-blank line) when it is one, without a forcing dot."""
    for line in text.split("\n"):
        if line.strip():
            if is_heading(line):
                stripped = line.strip()
                return stripped[1:] if stripped.startswith(".") else stripped
            return None
    return None


def check_text(text: Any, where: str) -> str:
    """A scene's Fountain text: bounded, LF only, one heading at most and only first."""
    if type(text) is not str:
        fail("invalid_script", f"{where}: must be text")
    if len(text.encode("utf-8")) > MAX_SCENE_TEXT_BYTES:
        fail("scene_too_large", f"{where}: a scene holds at most 64 KiB of text")
    if "\r" in text or _CONTROL_RE.search(text):
        fail("invalid_script", f"{where}: line ends must be LF and the text holds no control characters")
    if ID_NOTE_RE.search(text):
        fail("invalid_script", f"{where}: scene ids live in the document, not in the text")
    lines = text.split("\n")
    first = next((i for i, line in enumerate(lines) if line.strip()), None)
    previous_blank = True
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped in ("/*", "*/") or stripped.startswith("/*") or stripped.endswith("*/"):
            fail("invalid_script", f"{where}: boneyard /* */ is not scene text; a scene set aside holds it")
        if stripped.startswith("#"):
            fail("invalid_script", f"{where}: a section line (#) is an act break, not scene text")
        if stripped.startswith("=") and not stripped.startswith("==="):
            fail("invalid_script", f"{where}: a synopsis line (=) belongs in synopsis")
        if index != first and previous_blank and is_heading(line):
            fail("invalid_script", f"{where}: line {index + 1} is a second scene heading; one scene holds one heading")
        if index == first and is_heading(line) and NUMBER_IN_HEADING_RE.search(stripped):
            fail("invalid_script", f"{where}: scene numbers are set with number_lock, not in the heading")
        previous_blank = not stripped
    return text


def _check_card(card: Any, where: str) -> None:
    if card is None:
        return
    if type(card) is not dict or set(card) != {"x", "y", "revision"}:
        fail("invalid_script", f"{where}: must be null or {{x, y, revision}}")
    if type(card["revision"]) is not int or card["revision"] < 1:
        fail("invalid_script", f"{where}.revision: must be an integer >= 1")
    for key in ("x", "y"):
        value = card[key]
        if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > MAX_CARD:
            fail("invalid_script", f"{where}.{key}: must be a finite number within +-{MAX_CARD}")


def check_title(value: Any, where: str) -> str:
    if type(value) is not str or not value.strip() or len(value) > MAX_TITLE or _CONTROL_RE.search(value) or "\n" in value:
        fail("invalid_script", f"{where}: must be one line of 1..{MAX_TITLE} characters")
    return value


def check_synopsis(value: Any, where: str) -> str:
    if type(value) is not str or len(value) > MAX_SYNOPSIS or "\r" in value or _CONTROL_RE.search(value):
        fail("invalid_script", f"{where}: must be text of at most {MAX_SYNOPSIS} characters, LF line ends")
    return value


SCENE_KEYS = {"text", "synopsis", "card", "number", "numbered", "omitted", "revision", "by"}
STATE_KEYS = {"title_page", "scenes", "order", "order_revision", "numbers_locked"}
PAGE_SIZES = ("letter", "a4")  # optional "page_size" (absent = letter): pages.PAGE_SIZES


def validate_state(state: Any) -> dict[str, Any]:
    if type(state) is not dict or set(state) - {"page_size"} != STATE_KEYS:
        fail("invalid_script", "state: must hold exactly " + ", ".join(sorted(STATE_KEYS)) + " (and optional page_size)")
    if "page_size" in state and state["page_size"] not in PAGE_SIZES:
        fail("invalid_script", "page_size: letter or a4")
    page = state["title_page"]
    if type(page) is not list or len(page) > MAX_TITLE_PAGE:
        fail("invalid_script", f"title_page: must be a list of at most {MAX_TITLE_PAGE} entries")
    for i, entry in enumerate(page):
        if type(entry) is not dict or set(entry) != {"key", "value"}:
            fail("invalid_script", f"title_page[{i}]: must be {{key, value}}")
        if type(entry["key"]) is not str or not _TITLE_KEY_RE.fullmatch(entry["key"]):
            fail("invalid_script", f"title_page[{i}].key: letters, digits, space, _ or -, at most {MAX_TITLE_KEY}")
        value = entry["value"]
        if type(value) is not str or len(value) > MAX_TITLE_VALUE or "\r" in value or _CONTROL_RE.search(value):
            fail("invalid_script", f"title_page[{i}].value: text of at most {MAX_TITLE_VALUE} characters")
    scenes = state["scenes"]
    if type(scenes) is not dict or len(scenes) > MAX_SCENES:
        fail("invalid_script", f"scenes: at most {MAX_SCENES} scenes")
    numbers: set[str] = set()
    for sid, scene in scenes.items():
        where = f"scenes.{sid}"
        if type(sid) is not str or not SCENE_ID_RE.fullmatch(sid):
            fail("invalid_script", f"{where}: scene ids look like s-1a2b3c4d")
        if type(scene) is not dict or set(scene) - {"shots"} != SCENE_KEYS:
            fail("invalid_script", f"{where}: must hold exactly " + ", ".join(sorted(SCENE_KEYS)) + " (and optional shots)")
        check_text(scene["text"], f"{where}.text")
        check_synopsis(scene["synopsis"], f"{where}.synopsis")
        _check_card(scene["card"], f"{where}.card")
        number = scene["number"]
        if number is not None:
            if type(number) is not str or not NUMBER_RE.fullmatch(number):
                fail("invalid_script", f"{where}.number: null or 1-8 of 0-9 A-Z")
            if number in numbers:
                fail("invalid_script", f"{where}.number: {number} is used twice")
            numbers.add(number)
        if type(scene["numbered"]) is not bool or (number is not None and not scene["numbered"]):
            fail("invalid_script", f"{where}.numbered: must be true or false, true while it has a number")
        if type(scene["omitted"]) is not bool:
            fail("invalid_script", f"{where}.omitted: must be true or false")
        if type(scene["revision"]) is not int or scene["revision"] < 1:
            fail("invalid_script", f"{where}.revision: must be an integer >= 1")
        if scene["by"] not in AUTHORS:
            fail("invalid_script", f"{where}.by: must be artist or hermes")
    if numbers and not state["numbers_locked"]:
        fail("invalid_script", "scenes: stored numbers exist only while numbers_locked is true")
    if type(state["numbers_locked"]) is not bool:
        fail("invalid_script", "numbers_locked: must be true or false")
    if type(state["order_revision"]) is not int or state["order_revision"] < 0:
        fail("invalid_script", "order_revision: must be an integer >= 0")
    order = state["order"]
    if type(order) is not list or len(order) > MAX_SCENES + MAX_BREAKS:
        fail("invalid_script", "order: must be a list of scene ids and act breaks")
    seen: set[str] = set()
    breaks = 0
    for i, item in enumerate(order):
        if type(item) is str:
            if item not in scenes:
                fail("invalid_script", f"order[{i}]: {item} is not a scene of this script")
            key = item
        elif type(item) is dict:
            if set(item) - {"level"} != {"break", "id", "title"} or item["break"] != "act":
                fail("invalid_script", f"order[{i}]: an act break is {{break: 'act', id, title}} (+ level 2-6 for a deeper section)")
            if "level" in item and (type(item["level"]) is not int or not 2 <= item["level"] <= 6):
                fail("invalid_script", f"order[{i}].level: 2-6 (a section ## to ######; an act break # has none)")
            if type(item["id"]) is not str or not BREAK_ID_RE.fullmatch(item["id"]):
                fail("invalid_script", f"order[{i}].id: act break ids look like a-1a2b3c4d")
            check_title(item["title"], f"order[{i}].title")
            key = item["id"]
            breaks += 1
        else:
            fail("invalid_script", f"order[{i}]: must be a scene id or an act break")
        if key in seen:
            fail("invalid_script", f"order[{i}]: {key} appears twice (a scene has one place in the order)")
        seen.add(key)
    if breaks > MAX_BREAKS:
        fail("invalid_script", f"order: at most {MAX_BREAKS} act breaks")
    from . import shots as _shots  # shot ids: one per Shot line, unique in the document
    _shots.check(state, fail)
    return state


def scene_ids_in_order(state: dict[str, Any]) -> list[str]:
    return [item for item in state["order"] if type(item) is str]


def aside_ids(state: dict[str, Any]) -> list[str]:
    placed = set(scene_ids_in_order(state))
    return [sid for sid in state["scenes"] if sid not in placed]


def is_numbered_block(scene: dict[str, Any]) -> bool:
    """A block that is a scene: it starts with a heading, or it is omitted (keeps its number).
    Text before the first heading (a preamble, a cast list) is not a scene (charter Script
    decision 11, Fountain spec, Final Draft, Highland): no number, not counted, no strip cell."""
    return bool(scene["omitted"]) or heading_of(scene["text"]) is not None


def numbered_ids(state: dict[str, Any]) -> list[str]:
    """The scenes in the order that carry a number (blocks without a heading are skipped)."""
    return [sid for sid in scene_ids_in_order(state) if is_numbered_block(state["scenes"][sid])]


def numbers(state: dict[str, Any]) -> dict[str, str]:
    """The number each scene in the order shows: stored while locked, else 1..n from the first
    heading (a block without a heading has none)."""
    if state["numbers_locked"]:
        return {sid: state["scenes"][sid]["number"] for sid in scene_ids_in_order(state) if state["scenes"][sid]["number"]}
    return {sid: str(i + 1) for i, sid in enumerate(numbered_ids(state))}


def _letters_after(base: str) -> list[str]:
    letters = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    out = []
    if base[-1:].isalpha():
        out += [base[:-1] + letter for letter in letters if letter > base[-1]]
    out += [base + letter for letter in letters]
    return [value for value in out if NUMBER_RE.fullmatch(value)]


def assign_locked_numbers(state: dict[str, Any]) -> list[str]:
    """While locked, a scene that enters the order without a number gets one after its
    numbered predecessor (12 -> 12A -> 12B), before the first (1 -> A1). Returns the ids."""
    if not state["numbers_locked"]:
        return []
    scenes = state["scenes"]
    used = {scene["number"] for scene in scenes.values() if scene["number"]}
    ids = numbered_ids(state)  # a block without a heading never gets a number (decision 11)
    assigned = []
    for index, sid in enumerate(ids):
        if scenes[sid]["number"]:
            continue
        previous = next((scenes[p]["number"] for p in reversed(ids[:index]) if scenes[p]["number"]), None)
        if previous is not None:
            candidates = _letters_after(previous)
        else:
            following = next((scenes[n]["number"] for n in ids[index + 1:] if scenes[n]["number"]), None)
            base = following or "1"
            candidates = [letter + base for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"] + [str(n) for n in range(1, 100000)]
            candidates = [value for value in candidates if NUMBER_RE.fullmatch(value)]
        number = next((value for value in candidates if value not in used), None)
        if number is None:
            fail("invalid_op", f"no free scene number after {previous}; unlock and lock numbers again")
        scenes[sid]["number"] = number
        used.add(number)
        assigned.append(sid)
    return assigned


_EXTENSION_RE = re.compile(r"\s*\((?:[^()]*)\)\s*$")
_PLACE_PREFIX_RE = re.compile(r"^(?:\.|(?:int\.?/ext|ext\.?/int|i/e|int|ext|est)\.?\s+)", re.IGNORECASE)


def speaker_name(cue: str) -> str:
    """A character cue without extensions ((V.O.), (CONT'D)), dual mark ^ or forcing @."""
    name = cue.strip()
    if name.endswith("^"):
        name = name[:-1].rstrip()
    if name.startswith("@"):
        name = name[1:]
    while True:
        stripped = _EXTENSION_RE.sub("", name)
        if stripped == name:
            break
        name = stripped
    return name.strip()


def place_name(heading: str) -> str:
    """The place of a heading: without INT./EXT. and the time of day after the last ' - '."""
    text = NUMBER_IN_HEADING_RE.sub("", heading).strip()
    text = _PLACE_PREFIX_RE.sub("", text, count=1).strip()
    if " - " in text:
        text = text.rsplit(" - ", 1)[0].strip()
    return text.upper()


def line_id(scene_id: str, n: int) -> str:
    """The id of the n-th spoken line of a scene, in Voice's id form (``line_<12 hex>``): stable
    while the lines before it in its scene stay (the same rule as shots.derived_ids)."""
    import hashlib
    return "line_" + hashlib.sha256(f"{scene_id}:line:{n}".encode()).hexdigest()[:12]


# Charter Script decision 14: a centered line alone between scenes whose text is an act word is an act break
# (list and strip only; the text is unchanged, so Fountain/FDX/PDF are byte-identical). "ENDE ..." closes the script.
_ACT_RE = re.compile(r"^(?:(?:AKT|ACT)\s+\S+|COLD\s+OPEN|TEASER|TAG|PROLOG(?:UE)?|EPILOG(?:UE)?|(?:ENDE|END)(?:\s+.*)?)$", re.IGNORECASE)
_END_RE = re.compile(r"^(?:ENDE|END)(?:\s+.*)?$", re.IGNORECASE)
ACT_RE = _ACT_RE  # ops.move keeps act lines at their gap (p12-scrb)


def act_marks(state: dict[str, Any], elements_of: Any, shown: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """The author's act marks: (acts [{title, first_scene}], end {title, after_scene} | None). A mark is a centered
    element (``> AKT 1 <``) at the END of a scene's text (or of the preamble), i.e. alone between two scenes; it opens
    the act of the next numbered scene. An end word (``ENDE ...``) or a mark with no scene after it is the end mark."""
    acts: list[dict[str, Any]] = []
    end: dict[str, Any] | None = None
    pending: str | None = None
    last: str | None = None
    for sid in scene_ids_in_order(state):
        scene = state["scenes"][sid]
        if scene["omitted"]:
            continue
        if sid in shown:
            if pending is not None:
                if _END_RE.match(pending) and acts:
                    end = {"title": pending, "after_scene": last}
                else:
                    acts.append({"title": pending, "first_scene": sid})
                pending = None
            last = sid
        elements = [e for e in elements_of(scene["text"]) if e["type"] != "page_break"]
        tail: str | None = None
        for element in reversed(elements):
            if element["type"] != "centered":
                break
            text = " ".join(element["text"].split())
            if _ACT_RE.match(text):
                tail = text
                break
        if tail is not None:
            pending = tail
    if pending is not None and last is not None:
        end = {"title": pending, "after_scene": last}
    return acts, end


# Lane p13-scr (SCR-009/058): a character's first appearance in action is set in caps (screenplay convention; Final
# Draft, Highland and Arc Studio list characters from the cues). A caps run in an ACTION element is a character when it
# is a cue name of the script or carries the age convention (``OLD PETE, 70`` / ``MARA (30s)``). Headings, transitions,
# centered lines and dialogue never introduce. The editor's twin is src/script/editor/characters.js (same rules).
_CAPS_WORD = r"[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ'’-]*"
CAPS_RUN_RE = re.compile(r"(?<![^\W_])" + _CAPS_WORD + r"(?: " + _CAPS_WORD + r")*(?![^\W_])")
_AGE_AFTER_RE = re.compile(r"^\s*(?:,\s*\d|\(\s*\d)")
# Lane w22-pv (SCR-078): a leading article of a caps run ("A FERRYMAN (70s)") is English, not part of the name; a cue
# that really starts with one ("THE BOSS") still matches whole, because cue names are tried before the article rule.
_ARTICLES = frozenset({"A", "AN", "THE"})


def caps_names(text: str, cues: set[str]) -> list[str]:
    """The character names a piece of action text introduces or mentions in caps, in order."""
    from .fountain import printed
    plain = printed(text)
    out: list[str] = []
    for m in CAPS_RUN_RE.finditer(plain):
        words = m.group(0).split(" ")
        lead = 1 if len(words) > 1 and words[0] in _ARTICLES else 0
        k = 0
        while k < len(words):
            for j in range(len(words), k, -1):
                name = " ".join(words[k:j])
                if name in cues or (j == len(words) and k == lead and any(c.isalpha() for c in name) and len(name) > 1
                                    and _AGE_AFTER_RE.match(plain[m.end():])):
                    out.append(name)
                    k = j
                    break
            else:
                k += 1
    return out


def derive(state: dict[str, Any], elements_of: Any) -> dict[str, Any]:
    """Read-only views: numbers, pages, aside, characters and places of the scenes in the
    order (omitted scenes excluded). ``elements_of(text)`` is the Fountain classifier.
    ``pages`` is the one page count (``pages.py``, twin of the editor's ``pages.js``):
    {total, lines_per_page, scenes: {id: {from, to, eighths}}}. ``lines`` are the spoken lines
    (board E21 inspector "Lines"): {id (line_<12 hex>, the form Voice addresses lines by), scene,
    n, character, text, direction (the parenthetical before it)}."""
    from . import shots as _shots
    shown = numbers(state)
    acts, act_end = act_marks(state, elements_of, shown)
    characters: dict[str, dict[str, Any]] = {}
    places: dict[str, dict[str, Any]] = {}
    lines: list[dict[str, Any]] = []
    cues = {speaker_name(e["text"]) for sid in scene_ids_in_order(state) if not state["scenes"][sid]["omitted"]
            for e in elements_of(state["scenes"][sid]["text"]) if e["type"] == "character"} - {""}

    def character(name: str) -> dict[str, Any]:
        return characters.setdefault(name, {"name": name, "scenes": [], "lines": 0, "speaks": [], "introduced": None})

    for sid in scene_ids_in_order(state):
        scene = state["scenes"][sid]
        if scene["omitted"]:
            continue
        speaker, direction, n = "", "", 0
        for element in elements_of(scene["text"]):
            if element["type"] == "dialogue" and speaker:
                n += 1
                lines.append({"id": line_id(sid, n), "scene": sid, "n": n, "character": speaker,
                              "text": element["text"], "direction": direction})
                direction = ""
            elif element["type"] == "parenthetical" and speaker:
                direction = element["text"].strip()
            elif element["type"] != "character":
                speaker, direction = "", ""
            if element["type"] == "character":
                speaker, direction = speaker_name(element["text"]), ""
                name = speaker_name(element["text"])
                if not name:
                    continue
                entry = character(name)
                entry["lines"] += 1
                if sid not in entry["scenes"]:
                    entry["scenes"].append(sid)
                if sid not in entry["speaks"]:
                    entry["speaks"].append(sid)
            elif element["type"] == "action" and sid in shown:
                for name in caps_names(element["text"], cues):
                    entry = character(name)
                    if entry["introduced"] is None:
                        entry["introduced"] = sid
                    if sid not in entry["scenes"]:
                        entry["scenes"].append(sid)
            elif element["type"] == "heading":
                name = place_name(element["text"])
                if name:
                    entry = places.setdefault(name, {"name": name, "scenes": []})
                    if sid not in entry["scenes"]:
                        entry["scenes"].append(sid)
    return {
        "numbers": shown,
        "pages": pages.of_state(state, elements_of, scene_ids_in_order(state)),
        "aside": aside_ids(state),
        "shots": _shots.listing(state, shown, scene_ids_in_order(state)),
        "characters": sorted(characters.values(), key=lambda c: (-c["lines"], c["name"])),
        "lines": lines,
        "places": sorted(places.values(), key=lambda p: (-len(p["scenes"]), p["name"])),
        "acts": acts,
        "act_end": act_end,
    }
