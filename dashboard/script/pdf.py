"""The screenplay as PDF, written by hand (no library): PDF 1.4, base-14 Courier 12 pt, US Letter.

The page layout is ``pages.py``'s and nothing else: ``paginate(..., trace=...)`` hands over
every row it places and every page it turns, and this writer only draws them. So the PDF has
the page count of ``derived.pages`` and of the editor (one count everywhere), its (MORE) /
NAME (CONT'D) where the editor shows them, and the lines broken by ``pages.wrap``.

Geometry (72 pt = 1 in; Courier 12 = 7.2 pt per character, 12 pt per line):

* 612 x 792 pt. Script line k (1..54) has its baseline at 72 + 12 k pt from the top edge
  (1 in top margin, 54 lines, 1 in bottom margin).
* Columns from the left edge (in): heading, action, shot 1.5; character 3.5; parenthetical
  3.0; dialogue, lyric 2.5; transition flush right at 7.1 (Final Draft); centered around 4.5. Dual dialogue:
  two columns at 1.5 and 4.5 in, cue 8 characters in, parenthetical 3 (as the editor draws it).
* Page numbers ``N.`` top right (baseline 0.5 in + 12 pt) from page 2 on; scene numbers left
  (1.0 in) and right (7.625 in) of the heading while numbers are locked.
* A title page (when the script has one) comes first and is not counted: Title, Credit,
  Author(s) and Source centred from line 18; every other entry bottom left.

Text is written in WinAnsiEncoding (cp1252); a character outside it prints as ``?``. Fountain
emphasis prints as it reads (``emphasis.py``): ``**bold**`` in Courier-Bold, ``*italic*`` in
Courier-Oblique, both in Courier-BoldOblique, ``_underline_`` with a 0.6 pt stroke under the
run; markers, escape backslashes and notes ``[[…]]`` never print. The engine wraps the printed
text (``pages.printed``), so the line breaks are those of the editor, which hides the markers.
"""
from __future__ import annotations

import zlib
from typing import Any, Callable

import re

from . import emphasis
from . import pages as layout
from .schema import numbers, scene_ids_in_order

PAGE_W, PAGE_H = 612, 792
TOP, LINE, CHAR = 72, 12, 7.2
LEFT = {"heading": 108, "action": 108, "shot": 108, "character": 252, "parenthetical": 216,
        "dialogue": 180, "lyric": 180, "transition": None, "centered": None}
DUAL_LEFT = (108, 324)
DUAL_INDENT = {"character": 8, "parenthetical": 3, "dialogue": 0}
NUMBER_LEFT, NUMBER_RIGHT = 72, 549
CENTRED_KEYS = ("title", "credit", "author", "authors", "source")


def _escape(text: str) -> bytes:
    raw = text.encode("cp1252", errors="replace")
    return raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


class _Page:
    def __init__(self, height: int = PAGE_H) -> None:
        self.height = height  # the page's own height: US Letter 792 pt, A4 842 pt
        self.body: list[bytes] = []   # script text, in reading order
        self.marks: list[bytes] = []  # scene and page numbers, written after the text

    def text(self, x: float, line: int, value: str, font: str = "F1", marks: bool = False, top: float | None = None) -> None:
        if not value:
            return
        y = self.height - (top if top is not None else TOP + LINE * line)
        op = b"BT /%s 12 Tf %.2f %.2f Td (%s) Tj ET" % (font.encode(), x, y, _escape(value))
        (self.marks if marks else self.body).append(op)

    def styled(self, x: float, line: int, value: str, styles: list[frozenset[str]], base: str = "F1") -> None:
        """One printed line whose characters carry emphasis styles: one text run per style."""
        y = self.height - (TOP + LINE * line)
        start = 0
        while start < len(value):
            end = start
            while end < len(value) and styles[end] == styles[start]:
                end += 1
            style = styles[start]
            font = _FONT[("bold" in style, "italic" in style or base == "F2")]
            piece = value[start:end]
            left = x + CHAR * start
            if piece.strip():
                self.body.append(b"BT /%s 12 Tf %.2f %.2f Td (%s) Tj ET" % (font.encode(), left, y, _escape(piece)))
            if "underline" in style:
                self.body.append(b"0.6 w %.2f %.2f m %.2f %.2f l S" % (left, y - 1.8, left + CHAR * len(piece), y - 1.8))
            start = end

    def stream(self) -> bytes:
        return b"\n".join(self.body + self.marks) + b"\n"


_FONT = {(False, False): "F1", (False, True): "F2", (True, False): "F3", (True, True): "F4"}


def _printed_styles(text: str) -> tuple[str, list[frozenset[str]]]:
    """The printed text of one unit and the emphasis style of each of its characters."""
    walked = emphasis._walk(re.sub(r"\[\[.*?\]\]", "", text, flags=re.S))
    return "".join(ch for ch, _s, _i in walked), [st for _ch, st, _i in walked]


def _pieces_of(unit: str, lines: list[str]) -> list[int]:
    """Where each wrapped line starts in ``unit`` (wrap only drops spaces at the breaks)."""
    out, at = [], 0
    for piece in lines:
        found = unit.find(piece, at) if piece else at
        out.append(found if found >= 0 else at)
        at = (found if found >= 0 else at) + len(piece)
    return out


def _x(kind: str, value: str, right: float = 540) -> float:
    """Left edge of a printed line; ``right`` is the page's right text edge (1 in from the side
    on Letter, 0.73 in on A4 so the 60-character Action line stays)."""
    if kind == "transition":  # Final Draft: flush right at 7.1 in (FDX Transition RightIndent 7.10)
        return right - 4 * CHAR - CHAR * len(value)
    if kind == "centered":
        return 324 - CHAR * len(value) / 2
    return LEFT.get(kind) or 108


def _dual_columns(elements: list[dict[str, Any]], start: int) -> list[list[dict[str, Any]]]:
    """The two speeches of a dual block from element ``start`` (pages.groups' grouping)."""
    columns: list[list[dict[str, Any]]] = []
    for item in elements[start:]:
        if not item.get("dual"):
            break
        if item["type"] == "character" or not columns:
            if len(columns) == 2:
                break
            columns.append([])
        columns[-1].append(item)
    return columns


def _title_page(entries: list[dict[str, str]], height: int = PAGE_H, lines: int = 54) -> _Page | None:
    if not entries:
        return None
    page = _Page(height)
    line = 18
    corner: list[str] = []
    for entry in entries:
        values = [_printed_styles(v.strip()) for v in str(entry["value"]).split("\n")]
        if str(entry["key"]).strip().lower() in CENTRED_KEYS:
            for value, styles in values:
                page.styled(_x("centered", value), line, value, styles)
                line += 1
            line += 1
        else:
            corner.extend(values)
    for k, (value, styles) in enumerate(corner[-12:]):
        page.styled(108, lines - min(len(corner), 12) + 1 + k, value, styles)
    return page


def layout_of(state: dict[str, Any], elements_of: Callable[[str], list[dict[str, Any]]]) -> dict[str, Any]:
    """The printed pages of a script: {pages: [_Page], result (paginate's answer), title}."""
    printed = [sid for sid in scene_ids_in_order(state) if not state["scenes"][sid]["omitted"]]
    scenes = [{"id": sid, "elements": elements_of(state["scenes"][sid]["text"])} for sid in printed]
    shown = numbers(state) if state["numbers_locked"] else {}
    size = layout.PAGE_SIZES[layout.page_size(state)]
    m = layout.metrics_for(layout.page_size(state))
    right = 108 + CHAR * m["cpl"]["action"]  # 540 pt: the right end of the Action line (both sizes)
    trace: list[Any] = []
    result = layout.paginate(scenes, m, trace=trace)
    contd = layout.auto_contd(scenes)
    out: list[_Page] = []
    used = 0

    def new_page() -> None:
        nonlocal used
        out.append(_Page(size["height"]))
        used = 0

    def line_out(x: float, value: str, font: str = "F1") -> None:
        nonlocal used
        if not out or used >= m["lines"]:  # a row deeper than a page runs on
            new_page()
        used += 1
        out[-1].text(x, used, value, font)

    def blank() -> None:
        nonlocal used
        if not out or used >= m["lines"]:
            new_page()
        used += 1

    for step in trace:
        if step[0] == "turn":
            more = step[1]
            if more:
                out[-1].text(LEFT["character"], used + 1, "(MORE)")
            new_page()
            if more:
                line_out(LEFT["character"], f"{more} (CONT'D)")
            continue
        _kind, row, extra = step
        s, e, u = row["at"]
        element = scenes[s]["elements"][e]
        if row["kind"] == "page_break":
            continue
        if row["kind"] == "dual":
            columns = _dual_columns(scenes[s]["elements"], e)
            drawn = [[(item["type"], piece) for item in column
                      for piece in layout.wrap(layout.printed(item["text"]), m["dual_cpl"].get(item["type"], m["dual_cpl"]["dialogue"]))]
                     for column in columns]
            for _ in range(extra + row["n"] - max(len(c) for c in drawn)):
                blank()
            for k in range(max(len(c) for c in drawn)):
                blank()
                for side, column in enumerate(drawn):
                    if k < len(column):
                        kind, piece = column[k]
                        out[-1].text(DUAL_LEFT[side] + CHAR * DUAL_INDENT.get(kind, 0), used, piece)
            continue
        kind = element["type"]
        unit, styles = _printed_styles(layout.cue_text(element, (s, e) in contd).split("\n")[u])
        lines = layout.wrap(unit, m["cpl"].get(kind, 60))
        starts = _pieces_of(unit, lines)
        for _ in range(extra + row["n"] - len(lines)):
            blank()
        first = True
        for piece, at in zip(lines, starts):
            if any(styles[at:at + len(piece)]):
                line_out(_x(kind, piece, right), "", "F1")
                out[-1].styled(_x(kind, piece, right), used, piece, styles[at:at + len(piece)], "F2" if kind == "lyric" else "F1")
            else:
                line_out(_x(kind, piece, right), piece, "F2" if kind == "lyric" else "F1")
            if first and kind == "heading" and u == 0 and e == 0:
                number = shown.get(scenes[s]["id"])
                if number:
                    out[-1].text(NUMBER_LEFT, used, number, marks=True)
                    out[-1].text(NUMBER_RIGHT, used, number, marks=True)
            first = False
    for index, page in enumerate(out):
        if index:
            page.text(right - CHAR * len(f"{index + 1}."), 0, f"{index + 1}.", marks=True, top=36 + LINE)
    title = _title_page(state["title_page"], size["height"], m["lines"])
    return {"pages": out, "result": result, "title": title, "size": size}


def _document(pages: list[_Page], title: str, width: int = PAGE_W) -> bytes:
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    catalog = add(b"")  # filled below
    tree = add(b"")
    regular = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>")
    oblique = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier-Oblique /Encoding /WinAnsiEncoding >>")
    bold = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier-Bold /Encoding /WinAnsiEncoding >>")
    both = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier-BoldOblique /Encoding /WinAnsiEncoding >>")
    kids = []
    for page in pages:
        data = zlib.compress(page.stream(), 9)
        content = add(b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(data) + data + b"\nendstream")
        kids.append(add(b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 %d %d] /Resources << /Font << /F1 %d 0 R /F2 %d 0 R /F3 %d 0 R /F4 %d 0 R >> >> /Contents %d 0 R >>"
                        % (tree, width, page.height, regular, oblique, bold, both, content)))
    objects[catalog - 1] = b"<< /Type /Catalog /Pages %d 0 R >>" % tree
    objects[tree - 1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (b" ".join(b"%d 0 R" % k for k in kids), len(kids))
    info = add(b"<< /Producer (Continuo Film Suite) /Title (%s) >>" % _escape(title[:200]))
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root %d 0 R /Info %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, catalog, info, xref)
    return bytes(out)


def script_title(state: dict[str, Any]) -> str:
    """The script's title as plain text (first line of the title page's Title, no emphasis
    markers, no notes): the PDF's document title. The page header and Hermes' suite state use
    the JS twin (emphasis.js printed) on the same first line."""
    raw = next((str(e["value"]).split("\n")[0] for e in state["title_page"] if str(e["key"]).lower() == "title"), "")
    return _printed_styles(raw.strip())[0]


def write(state: dict[str, Any], elements_of: Callable[[str], list[dict[str, Any]]]) -> dict[str, Any]:
    """{data: PDF bytes, pages: script pages (= derived.pages.total), title_page: bool}."""
    made = layout_of(state, elements_of)
    pages = made["pages"]
    if len(pages) != made["result"]["pages"]:
        raise ValueError(f"pdf layout drew {len(pages)} pages, pages.py counts {made['result']['pages']}")
    if not pages:
        pages = [_Page(made["size"]["height"])]  # an empty script still prints one empty page
    title = script_title(state) or "Script"
    every = ([made["title"]] if made["title"] else []) + pages
    return {"data": _document(every, title, made["size"]["width"]), "pages": made["result"]["pages"],
            "title_page": made["title"] is not None, "size": layout.page_size(state)}


__all__ = ["layout_of", "write"]
