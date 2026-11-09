"""One page count everywhere: the screenplay page-break rules for ``studio.script/1``.

Twin of ``desktop/film-suite/src/script/editor/pages.js``. The editor draws its page breaks
from that one; Hermes reads ``derived.pages`` of ``continuo_script_read`` from this one; a
PDF writer lays out the same rows. Both twins answer the shared cases in
``tests/backend/fixtures/script/pages/*.json`` (``tests/backend/test_script_pages.py`` and
``checks/script-editor.cjs``), so they cannot drift apart silently.

The page (US Letter, Courier 12 pt = 10 characters per inch, 6 lines per inch):

* 11 in tall, 1 in top and bottom margin -> 54 lines of script per page.
* Element columns from the left page edge (inches) -> characters per line: heading, action,
  shot, centered 1.5-7.5 (60); character 3.5-7.5 (40); parenthetical 3.0-5.5 (25);
  dialogue and lyric 2.5-6.0 (35); transition 5.5-7.5 (20). Dual dialogue: two 3 in
  columns (character 22, parenthetical 24, dialogue 26 characters).
* Blank lines before an element: heading 2; action, shot, centered, lyric, character,
  transition 1; parenthetical and dialogue 0; none at the top of a page.

Break rules (the usual ones; informed by OpenDraft ``editor/pagination.ts``, MIT, rewritten):

* A heading stays with the block after it (never alone at the foot of a page).
* A character cue stays with its speech. A speech that does not fit may split between two of
  its rows (hard lines and parentheticals) when at least 2 spoken lines stay and at least 2
  go, and the page does not end on a parenthetical. The page then ends with (MORE) (a line
  kept free for it) and the next page starts with ``NAME (CONT'D)`` (one line). Otherwise
  the whole block goes to the next page.
* Dual dialogue (two speeches side by side) is one block as deep as its deeper column and
  never splits.
* ``===`` starts a new page. Omitted scenes do not print.
* Automatic (CONT'D) (Final Draft, OpenDraft; SCR-017): a cue whose speaker spoke last in the
  same scene, with something that is not speech in between (Action, a Shot, ...), prints as
  ``NAME (CONT'D)`` unless the cue already carries a (CONT'D) of its own. The engine never
  writes it into the text, so a typed (CONT'D) stays as typed and the program only changes
  what it set itself; the editor shows the same cue (``auto_contd``), the PDF prints it.
* A row deeper than a whole page runs on (a ``within`` break, never a blank page).

Wrapping (``wrap``): break at spaces (spaces at a wrap hang at the line end), after a hyphen
that is followed by a non-hyphen, before and after an em dash, after ``!``/``?`` before a
letter or digit; a word longer than the line is cut at the margin. That is what Chromium
does for these characters with ``white-space: pre-wrap; overflow-wrap: anywhere`` in a column
``cpl`` characters wide (compared against the browser by checks/script-editor.cjs).
"""
from __future__ import annotations

import re
from typing import Any, Callable

from . import emphasis

_NOTE_RE = re.compile(r"\[\[.*?\]\]", re.S)


def printed(text: str) -> str:
    """What a text prints (the engine wraps this): notes ``[[…]]`` never print, emphasis
    markers and escape backslashes neither (``emphasis.plain``)."""
    return emphasis.plain(_NOTE_RE.sub("", text))


def units(text: str) -> list[tuple[int, str]]:
    """(index, printed text) of an element's lines; a line that held only notes is gone."""
    return [(u, printed(unit)) for u, unit in enumerate(text.split("\n"))
            if not (unit.strip() and not printed(unit).strip())]


LINES_PER_PAGE = 54
CPL = {
    "heading": 60, "action": 60, "shot": 60, "centered": 60, "page_break": 60,
    "character": 40, "parenthetical": 25, "dialogue": 35, "lyric": 35, "transition": 20,
}
DUAL_CPL = {"character": 22, "parenthetical": 24, "dialogue": 26}
SPACE_BEFORE = {
    "heading": 2, "action": 1, "shot": 1, "centered": 1, "lyric": 1, "character": 1,
    "transition": 1, "parenthetical": 0, "dialogue": 0, "page_break": 0,
}
SPEECH = ("parenthetical", "dialogue")
EM_DASH = "\u2014"

# Page sizes (charter decision 4, SCR-041/047). The script document stores ``page_size``
# (absent = letter). Letter = this table's defaults (US Letter 8.5 x 11 in; its page breaks
# equal Final Draft's on the Brick & Steel sample, tests/backend/test_p5_scr_conformance.py).
# A4 (210 x 297 mm = 595 x 842 pt) keeps every left column and the 1 in top and bottom
# margins; the page is 0.27 in narrower, so the right-running columns lose characters
# (Character 38 instead of 40, Action 57 instead of 60 would leave 1 in right margin; the A4
# templates of Final Draft, Fade In and Highland narrow the right margin instead and keep
# the Action line, which gives Character 38, Dialogue 35), and 0.69 in taller: 58 lines.
PAGE_SIZES = {
    "letter": {"width": 612, "height": 792, "lines": 54, "cpl": {}},
    "a4": {"width": 595, "height": 842, "lines": 58, "cpl": {"character": 38}},
}


def page_size(state: dict[str, Any] | None) -> str:
    size = (state or {}).get("page_size", "letter")
    return size if size in PAGE_SIZES else "letter"


def metrics_for(size: str) -> dict[str, Any]:
    """The page metrics of one page size (``letter`` or ``a4``)."""
    spec = PAGE_SIZES[size]
    return metrics(lines=spec["lines"], cpl=spec["cpl"])


def metrics(**override: Any) -> dict[str, Any]:
    """The page metrics; a case or a comparison may override single values."""
    out: dict[str, Any] = {"lines": LINES_PER_PAGE, "cpl": dict(CPL), "dual_cpl": dict(DUAL_CPL),
                           "space_before": dict(SPACE_BEFORE)}
    for key, value in override.items():
        out[key] = {**out[key], **value} if isinstance(out.get(key), dict) else value
    return out


def _breaks_before(word: str, i: int) -> bool:
    before, here = word[i - 1], word[i]
    if before == "-" and here != "-":
        return True
    if (here == EM_DASH) != (before == EM_DASH):
        return True
    return before in "!?" and here.isalnum()


def _pieces(word: str) -> list[str]:
    out: list[str] = []
    start = 0
    for i in range(1, len(word)):
        if _breaks_before(word, i):
            out.append(word[start:i])
            start = i
    out.append(word[start:])
    return out


def wrap(text: str, cpl: int) -> list[str]:
    """The printed lines of one text at ``cpl`` characters per line (``\\n`` = a new line)."""
    out: list[str] = []
    for segment in text.split("\n"):
        line = ""      # the current line without the spaces that may hang
        pending = 0    # spaces since the last piece
        i, n = 0, len(segment)
        while i < n:
            j = i
            while j < n and segment[j] == " ":
                j += 1
            pending += j - i
            k = j
            while k < n and segment[k] != " ":
                k += 1
            if k == j:
                break  # trailing spaces hang at the end of the line
            for piece in _pieces(segment[j:k]):
                if len(line) + pending + len(piece) <= cpl:
                    line += " " * pending + piece
                else:
                    if line or pending:
                        out.append(line + (" " * pending if not line else ""))
                    rest = piece
                    while len(rest) > cpl:
                        out.append(rest[:cpl])
                        rest = rest[cpl:]
                    line = rest
                pending = 0
            i = k
        out.append(line)
    return out


def count(text: str, cpl: int) -> int:
    return len(wrap(text, cpl))


def speaker(cue: str) -> str:
    """The name of a cue without extensions, dual mark ^ and forcing @, upper case."""
    name = cue.strip()
    if name.endswith("^"):
        name = name[:-1].rstrip()
    if name.startswith("@"):
        name = name[1:]
    while name.endswith(")") and "(" in name:
        name = name[: name.rindex("(")].rstrip()
    return name.upper()


CONTD = " (CONT'D)"
_CONTD_RE = re.compile(r"\(\s*CONT'D\s*\)", re.I)
_SPEECH_KINDS = ("dialogue", "parenthetical")


def auto_contd(scenes: list[dict[str, Any]]) -> set[tuple[int, int]]:
    """(scene, element) of every cue that prints an automatic `` (CONT'D)`` (the module rule;
    the twin of the editor's rule in complete.js: same speaker again in the same scene after
    something printed that is not speech; a heading or a dual cue starts over)."""
    out: set[tuple[int, int]] = set()
    for s, scene in enumerate(scenes):
        last, between = None, False
        for e, element in enumerate(scene["elements"]):
            kind = element["type"]
            if kind == "character" and not element.get("dual"):
                who = speaker(printed(element["text"]))
                if who and who == last and between and not _CONTD_RE.search(element["text"]):
                    out.add((s, e))
                if who:
                    last = who
                between = False
                continue
            if kind == "character":
                last, between = None, False
                continue
            if kind not in _SPEECH_KINDS and printed(element["text"]).strip():
                between = True
            if kind == "heading":
                last = None
    return out


def cue_text(element: dict[str, Any], contd: bool) -> str:
    """The text a cue prints: its own text plus the automatic (CONT'D) when the rule sets one."""
    return element["text"] + CONTD if contd else element["text"]


def _rows(element: dict[str, Any], s: int, e: int, m: dict[str, Any], contd: bool = False) -> list[dict[str, Any]]:
    cpl = m["cpl"].get(element["type"], 60)
    return [{"n": count(unit, cpl), "at": [s, e, u], "kind": element["type"]}
            for u, unit in units(cue_text(element, contd))]


def groups(scenes: list[dict[str, Any]], m: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep-together groups in print order: {sb, rows, speech, head, kind}. ``head`` counts
    the rows before the first spoken row (heading and cue rows never split from it)."""
    # An element that prints nothing (only notes) takes no room.
    flat = [(s, e, element) for s, scene in enumerate(scenes) for e, element in enumerate(scene["elements"])
            if element["type"] == "page_break" or units(element["text"])]
    contd = auto_contd(scenes)
    blocks: list[dict[str, Any]] = []
    i = 0
    while i < len(flat):
        s, e, element = flat[i]
        kind = element["type"]
        if kind == "page_break":
            blocks.append({"sb": 0, "rows": [{"n": 0, "at": [s, e, 0], "kind": kind}], "speech": None, "head": 0, "kind": kind})
            i += 1
            continue
        if element.get("dual"):
            columns: list[int] = []
            j = i
            while j < len(flat) and flat[j][0] == s and flat[j][2].get("dual"):
                item = flat[j][2]
                if item["type"] == "character" or not columns:
                    if len(columns) == 2:
                        break
                    columns.append(0)
                columns[-1] += count(printed(item["text"]), m["dual_cpl"].get(item["type"], m["dual_cpl"]["dialogue"]))
                j += 1
            blocks.append({"sb": m["space_before"]["character"], "rows": [{"n": max(columns), "at": [s, e, 0], "kind": "dual"}],
                           "speech": None, "head": 0, "kind": "dual"})
            i = j
            continue
        if kind == "character":
            j = i + 1
            while j < len(flat) and flat[j][0] == s and flat[j][2]["type"] in SPEECH and not flat[j][2].get("dual"):
                j += 1
            if j > i + 1:
                rows = _rows(element, s, e, m, (s, e) in contd)
                head = len(rows)
                for s2, e2, item in flat[i + 1:j]:
                    rows += _rows(item, s2, e2, m)
                blocks.append({"sb": m["space_before"]["character"], "rows": rows, "speech": speaker(element["text"]),
                               "head": head, "kind": "speech"})
                i = j
                continue
        blocks.append({"sb": m["space_before"].get(kind, 1), "rows": _rows(element, s, e, m, (s, e) in contd), "speech": None, "head": 0, "kind": kind})
        i += 1
    out: list[dict[str, Any]] = []
    k = 0
    while k < len(blocks):
        block = blocks[k]
        if block["kind"] == "heading" and k + 1 < len(blocks) and blocks[k + 1]["kind"] not in ("heading", "page_break"):
            nxt = blocks[k + 1]
            first = {**nxt["rows"][0], "n": nxt["rows"][0]["n"] + nxt["sb"]}
            out.append({"sb": block["sb"], "rows": block["rows"] + [first] + nxt["rows"][1:], "speech": nxt["speech"],
                        "head": len(block["rows"]) + nxt["head"], "kind": nxt["kind"]})
            k += 2
            continue
        out.append(block)
        k += 1
    return out


def _split_at(group: dict[str, Any], room: int) -> int | None:
    """The fullest legal split of a speech whose rows before it fit in ``room`` lines."""
    rows, head = group["rows"], group["head"]
    for k in range(len(rows) - 1, head, -1):
        before = sum(row["n"] for row in rows[:k])
        said = sum(row["n"] for row in rows[head:k])
        after = sum(row["n"] for row in rows[k:])
        if before <= room and said >= 2 and after >= 2 and rows[k - 1]["kind"] != "parenthetical":
            return k
    return None


def paginate(scenes: list[dict[str, Any]], m: dict[str, Any] | None = None, trace: list[Any] | None = None) -> dict[str, Any]:
    """``scenes``: [{id, elements: [{type, text, dual?}]}] in print order.

    Returns {pages, breaks, scenes}. A break {page, at: [scene, element, unit], more, within,
    ended} names the first row of a new page (``ended``: lines used on the page before); ``more`` is the speaker whose speech continues there
    (the page before ends with (MORE), this one opens with NAME (CONT'D)); ``within`` marks a
    row deeper than a page that runs on. ``scenes`` gives each scene's first and last page and
    its length in eighths of a page (at least 1 when it prints anything).

    ``trace`` (a list) receives the layout as it is made, for the PDF writer: ("row", row,
    extra) for every placed row (``extra`` blank lines before it) and ("turn", more) for every
    page turn before a row; a ``within`` run-on is not traced (the row simply fills the page).
    The trace changes nothing in the answer."""
    m = m or metrics()
    lines = m["lines"]
    state = {"page": 1, "used": 0, "top": 0, "started": False}
    breaks: list[dict[str, Any]] = []
    spans: dict[int, list[int]] = {}

    def mark(scene: int, n: int) -> None:
        span = spans.setdefault(scene, [state["page"], state["page"], 0])
        span[1] = state["page"]
        span[2] += n

    def turn(at: list[int], more: str | None) -> None:
        if trace is not None:
            trace.append(("turn", more))
        ended = state["used"]
        state["page"] += 1
        state["used"] = state["top"] = 1 if more else 0
        breaks.append({"page": state["page"], "at": list(at), "more": more, "within": False, "ended": ended})
        if more:
            mark(at[0], 1)

    def place(row: dict[str, Any], extra: int = 0) -> None:
        if trace is not None:
            trace.append(("row", row, extra))
        n = row["n"] + extra
        while state["used"] + n > lines:  # deeper than a page: it runs on
            take = lines - state["used"]
            mark(row["at"][0], take)
            n -= take
            state["page"] += 1
            state["used"] = state["top"] = 0
            breaks.append({"page": state["page"], "at": list(row["at"]), "more": None, "within": True, "ended": lines})
        mark(row["at"][0], n)
        state["used"] += n

    for group in groups(scenes, m):
        rows = group["rows"]
        if group["kind"] == "page_break":
            if state["started"] and state["used"] > state["top"]:
                turn(rows[0]["at"], None)
            continue
        state["started"] = True
        total = sum(row["n"] for row in rows)
        sb = group["sb"] if state["used"] > state["top"] else 0
        if state["used"] + sb + total <= lines:
            for index, row in enumerate(rows):
                place(row, sb if index == 0 else 0)
            continue
        start = 0
        if state["used"] > state["top"]:
            split = _split_at(group, lines - state["used"] - sb - 1) if group["speech"] else None
            if split is not None:
                for index, row in enumerate(rows[:split]):
                    place(row, sb if index == 0 else 0)
                turn(rows[split]["at"], group["speech"])
                start = split
            else:
                turn(rows[0]["at"], None)
        for index in range(start, len(rows)):
            row = rows[index]
            spoken = group["speech"] is not None and index >= group["head"]
            limit = lines - 1 if spoken and index + 1 < len(rows) else lines
            if index > start and state["used"] > state["top"] and state["used"] + row["n"] > limit:
                turn(row["at"], group["speech"] if spoken else None)
            place(row)
    out_scenes = []
    for s, scene in enumerate(scenes):
        span = spans.get(s)
        if span is None:
            out_scenes.append({"id": scene.get("id"), "from": None, "to": None, "eighths": 0})
        else:
            out_scenes.append({"id": scene.get("id"), "from": span[0], "to": span[1],
                               "eighths": max(1, (span[2] * 8 + lines // 2) // lines)})
    return {"pages": state["page"] if state["started"] else 0, "breaks": breaks, "scenes": out_scenes}


def of_state(state: dict[str, Any], elements_of: Callable[[str], list[dict[str, Any]]], order_ids: list[str]) -> dict[str, Any]:
    """``derived.pages`` of a document: the page count of the scenes in the order (omitted
    ones excluded) on the document's page size, and per printed scene {from, to, eighths}.
    ``breaks``: where each page from 2 on starts, {page, scene, element, unit} (element = index
    in the scene's Fountain elements, unit = its line) and ``more`` = the speaker whose speech
    runs over (the page before ends with (MORE), this one opens with ``NAME (CONT'D)``).
    ``contd``: the cues that print an automatic (CONT'D), {scene, element}. The editor draws
    and the PDF prints exactly these (one rule, this module)."""
    shown = [sid for sid in order_ids if not state["scenes"][sid]["omitted"]]
    size = page_size(state)
    m = metrics_for(size)
    scenes = [{"id": sid, "elements": elements_of(state["scenes"][sid]["text"])} for sid in shown]
    result = paginate(scenes, m)
    return {
        "total": result["pages"],
        "size": size,
        "lines_per_page": m["lines"],
        "scenes": {entry["id"]: {"from": entry["from"], "to": entry["to"], "eighths": entry["eighths"]} for entry in result["scenes"]},
        "breaks": [{"page": b["page"], "scene": shown[b["at"][0]], "element": b["at"][1], "unit": b["at"][2], "more": b["more"]}
                   for b in result["breaks"]],
        "contd": [{"scene": shown[s], "element": e} for s, e in sorted(auto_contd(scenes))],
    }


__all__ = ["CONTD", "auto_contd", "cue_text", "CPL", "DUAL_CPL", "LINES_PER_PAGE", "SPACE_BEFORE", "count", "groups", "metrics", "of_state", "paginate", "speaker", "wrap"]
