"""Final Draft (FDX) import and export for ``studio.script/1``.

Import reads every ``<Paragraph>`` of ``<Content>`` (also inside ``<DualDialogue>``, which
Final Draft wraps in a Paragraph of its own), joins all ``<Text>`` runs of a paragraph
(revision runs included), and builds the same items ``fountain.parse`` returns, so one
importer turns both into a document:

* ``Scene Heading`` starts a scene; ``Number`` is its locked number; text ``OMITTED``
  is an omitted scene; ``SceneProperties`` Title and Summary become the synopsis.
* ``New Act`` is an act break; ``End of Act`` closes it (dropped, reported).
* Character / Dialogue / Parenthetical / Transition / Lyrics keep their element; the two
  speakers of a ``<DualDialogue>`` become a Fountain dual pair (``^``).
* Action with ``Alignment="Center"`` is centred text (``> THE END <``, SCR-048).
* A Text run's ``Style`` Bold / Italic / Underline becomes Fountain emphasis
  (``**`` ``*`` ``_``), ``AllCaps`` upper-cases the run (SCR-049).
* ``StartsNewPage="Yes"`` is a page break ``===`` before the paragraph (SCR-051).
* Action, General, Shot, Sound Effect, Music Cue and unknown types become action
  (non-Action types are counted in ``dropped.types``). Revision marks, page layout and
  fonts are not kept (counted).
* ``TitlePage`` (SCR-050): the keys Title, Credit, Author, Source, Draft date, Contact are
  read back from Final Draft's layout (centred blocks: title, credit line ``Written by``,
  author, ``Story by`` source; left blocks: a date, then the contact), or exactly from
  our own export (each paragraph carries ``ContinuoKey``). Other text becomes ``Notes``.
* ``ScriptNotes`` written by our export come back as Fountain notes ``[[…]]`` on their
  paragraph (another program's notes are counted); ``ContinuoAside`` holds the scenes set
  aside (Fountain's boneyard).

A DOCTYPE or ENTITY declaration is refused (no entity expansion), input is capped.

Export writes the scenes in the order, act breaks as ``New Act``, numbers on every
heading, omitted scenes as ``OMITTED`` headings with their number, the synopsis as
``SceneProperties``/``Summary``, dual dialogue inside ``<DualDialogue>``, centred text
with ``Alignment="Center"``, emphasis as Text ``Style``, ``===`` as ``StartsNewPage``, notes
as ``<ScriptNotes>``, aside scenes in ``<ContinuoAside>`` and the title page with its keys.
Deeper sections (``##``) have no FDX element and are counted (``sections_not_exported``).
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any

from . import emphasis
from . import fountain
from . import shots as shot_ids
from .schema import NUMBER_RE, ScriptError, aside_ids, numbers

MAX_FDX_BYTES = 8 * 1024 * 1024
_DECL_RE = re.compile(r"<!\s*(DOCTYPE|ENTITY)", re.IGNORECASE)
_ACTION_LIKE = {"Action", "General", "Shot", "Sound Effect", "Music Cue", "Cast List"}
_NOTE_RE = re.compile(r"\[\[.*?\]\]", re.S)
_CENTRED_KEYS = ("title", "credit", "author", "authors", "source")
_CREDIT_RE = re.compile(r"^(?:(?:written|screenplay|teleplay|script)\s+)?by$", re.IGNORECASE)
_SOURCE_RE = re.compile(r"^(?:story\s+by|based\s+on|from\b|adapted\s+from)", re.IGNORECASE)
_DATE_RE = re.compile(r"^(?:\d{1,4}[./-]\d{1,2}[./-]\d{1,4}|.*\b(?:19|20)\d\d\b.*|draft\b.*)$", re.IGNORECASE)
_STYLED = {"Action", "General", "Dialogue", "Lyrics", "Sound Effect", "Music Cue", "Cast List"}


def _text_of(paragraph: ET.Element) -> str:
    runs = paragraph.findall("Text")
    return "".join("".join(run.itertext()) for run in runs).replace("\r\n", "\n").replace("\r", "\n")


def _fountain_of(paragraph: ET.Element) -> str:
    """The paragraph's runs as Fountain: Style -> emphasis markers, AllCaps -> upper case,
    literal ``*`` ``_`` escaped. Markers never span a line break."""
    out = []
    for run in paragraph.findall("Text"):
        value = "".join(run.itertext()).replace("\r\n", "\n").replace("\r", "\n")
        names = set((run.get("Style") or "").split("+"))
        if "AllCaps" in names:
            value = value.upper()
        style = {key for key, name in (("bold", "Bold"), ("italic", "Italic"), ("underline", "Underline")) if name in names}
        out.append("\n".join(emphasis.markup(piece, style) for piece in value.split("\n")))
    return "".join(out)


def _escape_action(text: str) -> str:
    """Action text that Fountain would read as something else is forced with ``!``."""
    lines = text.split("\n")
    first = lines[0].strip()
    if first and (fountain.is_heading(first) or (len(lines) > 1 and fountain._is_cue(first))
                  or first.startswith(("#", "=", ">", "~", ".", "@", "[[")) or fountain._TRANSITION_RE.match(first)):
        lines[0] = "!" + lines[0]
    return "\n".join(lines)


def _cue(text: str, dual: bool) -> str:
    name = " ".join(text.split())
    if not fountain._is_cue(name):  # e.g. "McCOY" or "DR. NO: X" would read as action: force it
        name = "@" + name
    return name + (" ^" if dual else "")


def _title_page_in(page: ET.Element) -> list[dict[str, str]]:
    content = page.find("Content")
    paragraphs = list((content if content is not None else page).iter("Paragraph"))
    if any(p.get("ContinuoKey") for p in paragraphs):  # our own export: exact keys
        entries: list[dict[str, str]] = []
        last = None
        for p in paragraphs:
            key = p.get("ContinuoKey")
            if not key:
                last = None
                continue
            value = _fountain_of(p).strip()
            if last == key:
                entries[-1]["value"] += "\n" + value
            else:
                entries.append({"key": key[:40], "value": value})
            last = key
        return [{"key": e["key"], "value": e["value"][:1000]} for e in entries]
    # Final Draft's own layout: blocks of consecutive non-empty paragraphs, by alignment.
    blocks: list[tuple[str, list[str]]] = []
    for p in paragraphs:
        value = _fountain_of(p).strip()
        if not re.search(r"\w", emphasis.plain(value)):  # Final Draft pads the page with "." lines
            blocks.append(("", []))
            continue
        align = "center" if p.get("Alignment") == "Center" else "left"
        if blocks and blocks[-1][0] == align and blocks[-1][1]:
            blocks[-1][1].append(value)
        else:
            blocks.append((align, [value]))
    entries = []
    seen: set[str] = set()

    def put(key: str, lines: list[str]) -> None:
        if key in seen:
            key = "Notes"
        if key == "Notes" and "Notes" in seen:
            entry = next(e for e in entries if e["key"] == "Notes")
            entry["value"] = (entry["value"] + "\n" + "\n".join(lines))[:1000]
            return
        seen.add(key)
        entries.append({"key": key, "value": "\n".join(lines)[:1000]})

    after_credit = False
    for align, lines in blocks:
        if not lines:
            continue
        first = emphasis.plain(lines[0])
        credit = False
        if align == "center":
            if "Title" not in seen:
                put("Title", lines)
            elif _CREDIT_RE.match(first) and "Credit" not in seen:
                put("Credit", lines)
                credit = True
            elif after_credit and "Author" not in seen:
                put("Author", lines)
            elif _SOURCE_RE.match(first) and "Source" not in seen:
                put("Source", lines)
            else:
                put("Notes", lines)
        elif len(lines) == 1 and _DATE_RE.match(first) and "Draft date" not in seen:
            put("Draft date", lines)
        else:
            put("Contact", lines)
        after_credit = credit
    order = ["Title", "Credit", "Author", "Source", "Draft date", "Contact", "Notes"]
    return sorted(entries, key=lambda e: order.index(e["key"]))


def _paragraphs(container: ET.Element) -> list[tuple[ET.Element, bool, int]]:
    """Every content paragraph in order: (paragraph, in a dual block, its cue index there)."""
    out = []
    for child in container:
        dual = child if child.tag == "DualDialogue" else (child.find("DualDialogue") if child.tag == "Paragraph" else None)
        if dual is not None:
            cues = 0
            for p in dual.iter("Paragraph"):
                if p.get("Type") == "Character":
                    cues += 1
                out.append((p, True, cues))
        elif child.tag == "Paragraph":
            out.append((child, False, 0))
    return out


def _items(container: ET.Element, dropped: dict[str, Any], notes: dict[int, list[str]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    scene: dict[str, Any] | None = None
    block: list[str] = []  # the current Fountain block (a speech) of the scene

    def close_block() -> None:
        if block and scene is not None:
            scene["blocks"].append("\n".join(block))
            block.clear()

    def ensure_scene() -> dict[str, Any]:
        nonlocal scene
        if scene is None:
            scene = {"kind": "scene", "id": None, "blocks": [], "synopsis": "", "number": None, "omitted": False}
            items.append(scene)
        return scene

    for index, (paragraph, in_dual, cue_index) in enumerate(_paragraphs(container)):
        kind = paragraph.get("Type") or "Action"
        text = _fountain_of(paragraph) if kind in _STYLED else _text_of(paragraph)
        noted = "".join(f" [[{note}]]" for note in notes.get(index, []))
        if paragraph.get("StartsNewPage") == "Yes" and scene is not None and kind not in ("Scene Heading", "New Act") and text.strip():
            close_block()
            scene["blocks"].append("===")
        if kind in ("Scene Heading", "Scene Heading (Top of Page)"):
            close_block()
            number = paragraph.get("Number")
            number = number.strip().upper() if isinstance(number, str) and number.strip() else None
            if number is not None and not NUMBER_RE.fullmatch(number):
                dropped.setdefault("numbers", []).append(number)
                number = None
            heading = " ".join(text.split())
            omitted = heading.upper() == "OMITTED"
            synopsis_parts = []
            props = paragraph.find("SceneProperties")
            if props is not None:
                title = (props.get("Title") or "").strip()
                summary = "\n".join(_text_of(p).strip() for p in props.iter("Paragraph") if _text_of(p).strip())
                synopsis_parts = [part for part in (title, summary) if part]
            scene = {"kind": "scene", "id": None, "blocks": [], "synopsis": "\n".join(synopsis_parts)[:2000],
                     "number": number, "omitted": omitted}
            items.append(scene)
            if not omitted and heading:
                scene["blocks"].append((heading if fountain.is_heading(heading) else "." + heading) + noted)
            continue
        if kind == "New Act":
            close_block()
            title = " ".join(text.split())[:120]
            if title:
                items.append({"kind": "break", "id": None, "title": title})
                scene = None
            continue
        if kind == "End of Act":
            dropped["end_of_act"] += 1
            continue
        if not text.strip() and kind != "Dialogue":
            continue
        current = ensure_scene()
        if kind == "Character":
            close_block()
            block.append(_cue(text, in_dual and cue_index >= 2))
        elif kind == "Parenthetical":
            if not block:
                current["blocks"].append(_escape_action(text.strip()) + noted)
                continue
            stripped = text.strip()
            block.append(stripped if stripped.startswith("(") else f"({stripped})")
        elif kind == "Dialogue":
            if not block:
                current["blocks"].append(_escape_action(text.strip()) + noted)
                continue
            lines = [line.strip() or fountain.TWO_SPACES for line in text.strip().split("\n")]
            lines[-1] += noted
            block.extend(lines)
        elif kind == "Transition":
            close_block()
            stripped = " ".join(text.split())
            current["blocks"].append(stripped if fountain._TRANSITION_RE.match(stripped) else "> " + stripped)
        elif kind == "Lyrics":
            close_block()
            current["blocks"].append("\n".join("~" + line.strip() for line in text.strip().split("\n")) + noted)
        elif kind == "Shot" and shot_ids.SHOT_ID_RE.fullmatch(paragraph.get("ContinuoShot") or ""):
            # Our own export: the shot's stable id rides along as the Fountain note the importer reads.
            close_block()
            current["blocks"].append(" ".join(text.split()) + f" [[shot {paragraph.get('ContinuoShot')}]]")
        elif kind in ("Action", "General") and paragraph.get("Alignment") == "Center":
            close_block()
            current["blocks"].append("\n".join(f"> {line.strip()} <" for line in text.strip("\n").split("\n") if line.strip()) + noted)
        else:
            close_block()
            if kind not in ("Action",):
                dropped["types"][kind] = dropped["types"].get(kind, 0) + 1
            current["blocks"].append(_escape_action(text.strip("\n")) + noted)
    close_block()
    for item in items:
        if item["kind"] == "scene":
            item["text"] = "\n\n".join(item.pop("blocks"))
    return items


def parse(data: bytes | str) -> dict[str, Any]:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    if len(raw) > MAX_FDX_BYTES:
        raise ScriptError("import_too_large", "an FDX file is capped at 8 MiB")
    if _DECL_RE.search(raw.decode("utf-8", "ignore")):
        raise ScriptError("invalid_import", "FDX with a DOCTYPE or ENTITY declaration is refused")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ScriptError("invalid_import", f"not FDX: {exc}") from None
    if root.tag != "FinalDraft":
        raise ScriptError("invalid_import", "not FDX: the root element is not FinalDraft")
    dropped: dict[str, Any] = {"types": {}, "script_notes": 0, "revision_runs": 0, "end_of_act": 0, "title_page": []}
    dropped["revision_runs"] = sum(1 for run in root.iter("Text") if run.get("RevisionID") not in (None, "0"))
    notes: dict[str, dict[int, list[str]]] = {"content": {}, "aside": {}}
    for note in root.findall(".//ScriptNote"):
        where, at = note.get("ContinuoIn"), note.get("ContinuoParagraph")
        text = " ".join(_text_of(p).strip() for p in note.iter("Paragraph")).strip().replace("]]", "] ]")
        if where in notes and at is not None and at.isdigit() and text:
            notes[where].setdefault(int(at), []).append(text)
        else:
            dropped["script_notes"] += 1  # a note of another program: its anchor is not kept

    page = root.find("TitlePage")
    title_page = _title_page_in(page) if page is not None else []
    content = root.find("Content")
    items = _items(content if content is not None else ET.Element("Content"), dropped, notes["content"])
    holder = root.find("ContinuoAside")
    aside = [item for item in _items(holder, dropped, notes["aside"]) if item["kind"] == "scene"] if holder is not None else []
    return {"title_page": title_page, "items": items, "aside": aside, "dropped": dropped}


# --- export ----------------------------------------------------------------------

_STYLE_ORDER = (("bold", "Bold"), ("italic", "Italic"), ("underline", "Underline"))
_FDX_KIND = {"action": "Action", "character": "Character", "dialogue": "Dialogue", "parenthetical": "Parenthetical",
             "transition": "Transition", "lyric": "Lyrics", "centered": "Action"}


def _paragraph(parent: ET.Element, kind: str | None, text: str, **attrs: str) -> ET.Element:
    """A paragraph of plain text (headings, cues, transitions carry no emphasis)."""
    paragraph = ET.SubElement(parent, "Paragraph", **({"Type": kind} if kind else {}), **attrs)
    ET.SubElement(paragraph, "Text").text = text
    return paragraph


def _styled(parent: ET.Element, kind: str | None, text: str, **attrs: str) -> ET.Element:
    """A paragraph whose Fountain emphasis becomes Text runs with ``Style`` (SCR-049)."""
    paragraph = ET.SubElement(parent, "Paragraph", **({"Type": kind} if kind else {}), **attrs)
    for value, style in emphasis.runs(re.sub(r"[ \t]*\[\[.*?\]\]", "", text, flags=re.S)) or [("", frozenset())]:
        run = ET.SubElement(paragraph, "Text")
        names = "+".join(name for key, name in _STYLE_ORDER if key in style)
        if names:
            run.set("Style", names)
        run.text = value
    return paragraph


class _Writer:
    """Writes scenes into one container; remembers the notes of each paragraph (ScriptNotes)."""

    def __init__(self, container: ET.Element, where: str, notes: list[tuple[str, int, str]]) -> None:
        self.container, self.where, self.notes = container, where, notes
        self.count = 0
        self.new_page = False

    def add(self, parent: ET.Element, kind: str, text: str, *, styled: bool, **attrs: str) -> ET.Element:
        if self.new_page:
            attrs["StartsNewPage"] = "Yes"
            self.new_page = False
        for note in _NOTE_RE.findall(text):
            if not note[2:].lstrip().startswith("shot "):
                self.notes.append((self.where, self.count, note[2:-2].strip()))
        self.count += 1
        if styled:
            return _styled(parent, kind, text, **attrs)
        return _paragraph(parent, kind, " ".join(fountain.strip_notes(text).split()), **attrs)

    def scene(self, sid: str, scene: dict[str, Any], number: str | None) -> None:
        content = self.container
        attrs = {"Number": number} if number else {}
        if scene["omitted"]:
            self.add(content, "Scene Heading", "OMITTED", styled=False, **attrs)
            return
        parts = fountain.elements(scene["text"])
        # Shot lines (shots.py) leave as FDX Shot paragraphs with their stable id (ContinuoShot).
        pending_shots = [(line, shot_id) for (_i, line), shot_id in zip(shot_ids.shot_lines(scene["text"]), shot_ids.ids_of(sid, scene))]
        heading = parts[0]["text"] if parts and parts[0]["type"] == "heading" else ""
        if heading or scene["synopsis"] or number:
            head = self.add(content, "Scene Heading", heading, styled=False, **attrs)
            if scene["synopsis"]:
                props = ET.SubElement(head, "SceneProperties", Title="")
                summary = ET.SubElement(props, "Summary")
                for line in scene["synopsis"].split("\n"):
                    p = ET.SubElement(summary, "Paragraph")
                    ET.SubElement(p, "Text").text = line
        dual: ET.Element | None = None
        for element in parts[1:] if heading else parts:
            kind = element["type"]
            if kind == "heading":
                continue
            if kind == "page_break":  # === : the next paragraph starts a new page (SCR-051)
                self.new_page = True
                dual = None
                continue
            target = content
            if element.get("dual"):
                if kind == "character" and dual is None:
                    dual = ET.SubElement(content, "DualDialogue")
                target = dual if dual is not None else content
            elif kind == "character" or kind not in ("dialogue", "parenthetical"):
                dual = None
            fdx_kind = _FDX_KIND[kind]
            attrs = {"Alignment": "Center"} if kind == "centered" else {}
            if kind == "action" and pending_shots and element["text"].strip() == pending_shots[0][0]:
                fdx_kind, attrs = "Shot", {"ContinuoShot": pending_shots.pop(0)[1]}
            plain = kind in ("character", "transition", "parenthetical") or fdx_kind == "Shot"
            if fdx_kind == "Shot":  # its id note stays on the line (the importer reads it back)
                _paragraph(target, fdx_kind, element["text"], **attrs)
                self.count += 1
                continue
            self.add(target, fdx_kind, element["text"], styled=not plain, **attrs)


def _title_page_out(root: ET.Element, entries: list[dict[str, str]]) -> None:
    """Final Draft's layout: Title, Credit, Author(s), Source centred from the upper third,
    every other key bottom left; each paragraph names its key (``ContinuoKey``)."""
    tp_content = ET.SubElement(ET.SubElement(root, "TitlePage"), "Content")
    centred = [e for e in entries if e["key"].strip().lower() in _CENTRED_KEYS]
    corner = [e for e in entries if e["key"].strip().lower() not in _CENTRED_KEYS]
    for _ in range(17):
        _paragraph(tp_content, None, "", Alignment="Left")
    for group, align in ((centred, "Center"), (corner, "Left")):
        for k, entry in enumerate(group):
            if k or align == "Left":
                _paragraph(tp_content, None, "", Alignment=align)
            for line in entry["value"].split("\n"):
                _styled(tp_content, None, line, Alignment=align, ContinuoKey=entry["key"])


def write(state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """The script as FDX text, and what had no place in FDX (deeper sections, counted)."""
    root = ET.Element("FinalDraft", DocumentType="Script", Template="No", Version="5")
    content = ET.SubElement(root, "Content")
    shown = numbers(state)
    notes: list[tuple[str, int, str]] = []
    main = _Writer(content, "content", notes)
    lost = {"aside_not_exported": 0, "sections_not_exported": 0}
    for item in state["order"]:
        if type(item) is dict:
            if item.get("level", 1) > 1:
                lost["sections_not_exported"] += 1
                continue
            main.add(content, "New Act", item["title"], styled=False)
            continue
        main.scene(item, state["scenes"][item], shown.get(item))
    aside = aside_ids(state)
    if aside:  # scenes set aside (Fountain's boneyard): an element of their own, Final Draft skips it
        writer = _Writer(ET.SubElement(root, "ContinuoAside"), "aside", notes)
        for sid in aside:
            writer.scene(sid, state["scenes"][sid], None)
    if notes:
        script_notes = ET.SubElement(root, "ScriptNotes")
        for k, (where, index, text) in enumerate(notes, start=1):
            note = ET.SubElement(script_notes, "ScriptNote", ID=str(k), ContinuoIn=where, ContinuoParagraph=str(index))
            ET.SubElement(ET.SubElement(note, "Paragraph"), "Text").text = text
    if state["title_page"]:
        _title_page_out(root, state["title_page"])
    ET.indent(root, space="  ")
    text = '<?xml version="1.0" encoding="UTF-8" standalone="no" ?>\n' + ET.tostring(root, encoding="unicode") + "\n"
    return text, lost
