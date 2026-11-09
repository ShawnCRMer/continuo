"""Fountain for ``studio.script/1``: classify scene text, read and write whole scripts.

Three views of one document, all produced here:

* ``elements(text)``   one scene's Fountain classified into screenplay elements (the
                        derived characters/places and the FDX export read these).
* ``write(state, ids=True)``  the full script for Hermes: one ``[[s-xxxxxxxx]]`` note
                        line before each scene and ``[[a-xxxxxxxx]]`` before each act
                        break; aside scenes follow in one boneyard ``/* */`` at the end.
  ``write(state, ids=False)`` the plain export / ``script.fountain`` (same, no notes).
* ``parse(text)``      a whole Fountain script back into title page, order items and
                        aside scenes. Top-level sections ``#`` are act breaks, boneyard
                        scenes are aside, ``= synopsis`` lines are the scene's synopsis,
                        ``#12A#`` in a heading is its number, ``OMITTED`` is an omitted
                        scene. Deeper sections ``##``, ``###`` … are kept as structure
                        (an order item ``{break: act, level: n}``; they never print).
* ``printed(text)``    what one element's text prints: notes ``[[…]]`` removed, emphasis
                        markers and escape backslashes gone (``emphasis.plain``). Notes,
                        markers and escapes stay in the stored text (round trip).

A line of two spaces inside a block (a speech) keeps the block together: it is an empty
line of the element, not the blank line that ends it (fountain.io/syntax#line-breaks).

Fountain spec: https://fountain.io/syntax (headings, character cues, dual ``^``,
parentheticals, transitions, centred ``> <``, lyrics ``~``, notes, boneyard, sections,
synopses, page breaks ``===``, title page). Written for this backend; why screenplay-tools
was not vendored: docs/references/script-interchange-evaluation.md.
"""
from __future__ import annotations

import re
from typing import Any

from . import emphasis
from . import shots as shot_ids
from .schema import (
    BREAK_ID_RE,
    NUMBER_IN_HEADING_RE,
    NUMBER_RE,
    SCENE_ID_RE,
    aside_ids,
    is_heading,
    numbers,
)

_ID_LINE_RE = re.compile(r"^\s*\[\[\s*((?:s|a)-[0-9a-f]{8})\s*\]\]\s*$")
_TITLE_KEY_LINE_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 _-]{0,39}):(.*)$")
_OMITTED_RE = re.compile(r"^\.?OMITTED(?:\s+#([^#\s]+)#)?\s*$")
_SECTION_RE = re.compile(r"^(#+)\s*(.*?)\s*$")
_TRANSITION_RE = re.compile(r"^[^a-z]*TO:$")
_CHARACTER_EXT_RE = re.compile(r"\s*\([^()]*\)")
_NOTE_RE = re.compile(r"\[\[.*?\]\]", re.S)
TWO_SPACES = "  "


def is_kept_blank(line: str) -> bool:
    """A line of (at least) two spaces and nothing else: an empty line inside a block."""
    return len(line) >= 2 and line.strip(" ") == "" and "	" not in line


def strip_notes(text: str) -> str:
    """Text without its notes ``[[…]]`` (a note never prints)."""
    return _NOTE_RE.sub("", text)


def printed(text: str) -> str:
    """What a piece of element text prints: no notes, no emphasis markers, no escapes."""
    return emphasis.plain(strip_notes(text))


def _normalize(text: str) -> str:
    if text.startswith("\ufeff"):
        text = text[1:]
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _is_cue(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("@"):
        return len(stripped) > 1
    if stripped.startswith(("!", ".", ">", "~", "=", "#", "[[")) or is_heading(stripped):
        return False
    name = _CHARACTER_EXT_RE.sub("", stripped.rstrip("^").rstrip())
    if ":" in name:  # "SFX: A DOOR CREAKS." is a sound line, not a speaker (force one with @)
        return False
    return bool(re.search(r"[A-Z]", name)) and not re.search(r"[a-z]", name) and not _TRANSITION_RE.match(stripped)


def elements(text: str) -> list[dict[str, Any]]:
    """Classify one scene's Fountain text. Each element is {type, text} (+ ``dual`` on a
    character and its lines). Types: heading, action, character, parenthetical,
    dialogue, transition, centered, lyric, page_break."""
    out: list[dict[str, Any]] = []
    lines = _normalize(text).split("\n")
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.strip() or (current and is_kept_blank(line)):
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    for block in blocks:
        first = block[0].strip()
        if is_heading(first) and not out:
            out.append({"type": "heading", "text": first[1:] if first.startswith(".") else first})
            block = block[1:]
            if not block:
                continue
            first = block[0].strip()
        if len(block) == 1 and first == "===":
            out.append({"type": "page_break", "text": ""})
            continue
        if all(line.strip().startswith(">") and line.strip().endswith("<") and len(line.strip()) > 1 for line in block):
            out.append({"type": "centered", "text": "\n".join(line.strip()[1:-1].strip() for line in block)})
            continue
        if len(block) == 1 and (first.startswith(">") or (_TRANSITION_RE.match(first) and not first.startswith("!"))):
            out.append({"type": "transition", "text": first[1:].strip() if first.startswith(">") else first})
            continue
        if all(line.strip().startswith("~") for line in block):
            out.append({"type": "lyric", "text": "\n".join(line.strip()[1:].strip() for line in block)})
            continue
        if len(block) >= 2 and _is_cue(block[0]):
            cue = first[1:] if first.startswith("@") else first
            dual = cue.endswith("^")
            if dual:
                cue = cue[:-1].rstrip()
                for previous in reversed(out):
                    if previous["type"] in ("dialogue", "parenthetical", "character"):
                        previous["dual"] = True
                        if previous["type"] == "character":
                            break
                    else:
                        break
            out.append({"type": "character", "text": cue, **({"dual": True} if dual else {})})
            speech: list[str] = []

            def flush() -> None:
                if speech:
                    out.append({"type": "dialogue", "text": "\n".join(speech), **({"dual": True} if dual else {})})
                    speech.clear()

            for line in block[1:]:
                stripped = line.strip()
                if stripped.startswith("(") and stripped.endswith(")"):
                    flush()
                    out.append({"type": "parenthetical", "text": stripped, **({"dual": True} if dual else {})})
                else:
                    speech.append(stripped)  # a two-space line is an empty line of the speech
            flush()
            continue
        body = "\n".join(line[1:] if index == 0 and line.startswith("!") else line for index, line in enumerate(block))
        out.append({"type": "action", "text": body})
    return out


# --- writing -------------------------------------------------------------------


def _title_page_lines(entries: list[dict[str, str]]) -> list[str]:
    lines: list[str] = []
    for entry in entries:
        value = entry["value"]
        if "\n" in value:
            lines.append(f"{entry['key']}:")
            lines.extend("   " + part for part in value.split("\n"))
        else:
            lines.append(f"{entry['key']}: {value}")
    return lines


def _scene_lines(scene: dict[str, Any], number: str | None, *, show_number: bool, sid: str = "") -> list[str]:
    tag = f" #{number}#" if show_number and number else ""
    if scene["omitted"]:
        return [f".OMITTED{tag}"]
    lines = scene["text"].strip("\n").split("\n") if scene["text"].strip() else []
    # Each Shot line carries its stable id as a Fountain note (other programs hide notes, the PDF
    # never prints them, the importer reads the id back): ``CLOSE ON THE LETTER [[shot sh-1a2b3c4d]]``.
    if lines:
        for (index, _line), shot_id in zip(shot_ids.shot_lines("\n".join(lines)), shot_ids.ids_of(sid, scene)):
            lines[index] = lines[index].rstrip() + f" [[shot {shot_id}]]"
    synopsis = [f"= {line}" if line else "=" for line in scene["synopsis"].split("\n")] if scene["synopsis"] else []
    first = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first is not None and is_heading(lines[first]):
        heading = lines[first].rstrip()
        return lines[: first] + [heading + tag] + ([""] + synopsis if synopsis else []) + lines[first + 1:]
    return (synopsis + [""] if synopsis else []) + lines


def write(state: dict[str, Any], *, ids: bool) -> str:
    """The whole script as Fountain; ``ids`` adds the id note lines Hermes maps back by."""
    shown = numbers(state)
    show_numbers = state["numbers_locked"] or ids
    parts: list[list[str]] = []
    if state["title_page"]:
        parts.append(_title_page_lines(state["title_page"]))
    for item in state["order"]:
        if type(item) is dict:
            parts.append(([f"[[{item['id']}]]"] if ids else []) + ["#" * item.get("level", 1) + f" {item['title']}"])
        else:
            scene = state["scenes"][item]
            parts.append(([f"[[{item}]]"] if ids else []) + _scene_lines(scene, shown.get(item), show_number=show_numbers, sid=item))
    aside = aside_ids(state)
    if aside:
        bone: list[str] = ["/*"]
        for sid in aside:
            bone += ([f"[[{sid}]]"] if ids else []) + _scene_lines(state["scenes"][sid], None, show_number=False, sid=sid) + [""]
        bone[-1] = "*/"
        parts.append(bone)
    return "\n\n".join("\n".join(part) for part in parts) + "\n"


# --- reading -------------------------------------------------------------------


def _title_page(lines: list[str]) -> tuple[list[dict[str, str]], int]:
    """Title page entries and the index of the first line after it (0 when none)."""
    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1
    if start >= len(lines) or not _TITLE_KEY_LINE_RE.match(lines[start]) or is_heading(lines[start]):
        return [], 0
    entries: list[dict[str, str]] = []
    index = start
    while index < len(lines) and lines[index].strip():
        line = lines[index]
        match = _TITLE_KEY_LINE_RE.match(line)
        if match and not line.startswith((" ", "\t")):
            entries.append({"key": match.group(1).strip(), "value": match.group(2).strip()})
        elif entries and line.startswith((" ", "\t")):
            value = entries[-1]["value"]
            entries[-1]["value"] = (value + "\n" if value else "") + line.strip()
        else:
            return [], 0  # not a title page after all: a key line followed by prose
        index += 1
    return entries, index


def _split_boneyard(lines: list[str]) -> tuple[list[str], list[list[str]], list[str]]:
    """Main lines, each boneyard's lines, and inline comments removed from main lines."""
    main: list[str] = []
    bones: list[list[str]] = []
    inline: list[str] = []
    inside: list[str] | None = None
    for line in lines:
        rest = line
        while True:
            if inside is None:
                at = rest.find("/*")
                if at < 0:
                    main.append(rest)
                    break
                before, rest = rest[:at], rest[at + 2:]
                end = rest.find("*/")
                if end >= 0:  # an inline comment on one line: removed, reported
                    inline.append(rest[:end].strip())
                    rest = before + rest[end + 2:]
                    continue
                if before.strip():
                    main.append(before)
                inside = []
                if rest.strip():
                    inside.append(rest)
                break
            end = rest.find("*/")
            if end < 0:
                inside.append(rest)
                break
            if rest[:end].strip():
                inside.append(rest[:end])
            bones.append(inside)
            inside = None
            rest = rest[end + 2:]
            if not rest.strip():
                break
    if inside is not None:  # an unclosed boneyard runs to the end (Fountain readers agree)
        bones.append(inside)
    return main, bones, inline


def _scenes(lines: list[str], dropped: dict[str, list[str]], *, allow_breaks: bool) -> list[dict[str, Any]]:
    """Order items ({kind: scene|break}) from lines without title page or boneyard."""
    items: list[dict[str, Any]] = []
    scene: dict[str, Any] | None = None
    pending_id: str | None = None
    pending_synopsis: list[str] = []

    def start(kind: str, **values: Any) -> dict[str, Any]:
        nonlocal pending_id, scene
        item = {"kind": kind, "id": pending_id, **values}
        pending_id = None
        items.append(item)
        scene = item if kind == "scene" else None
        return item

    def new_scene(first_line: str | None, number: str | None = None, omitted: bool = False) -> None:
        nonlocal pending_synopsis
        start("scene", lines=[first_line] if first_line is not None else [], synopsis=list(pending_synopsis),
              number=number, omitted=omitted)
        pending_synopsis = []

    previous_blank = True
    for raw in lines:
        line = TWO_SPACES if is_kept_blank(raw) and scene is not None else raw.rstrip()
        stripped = line.strip()
        note = _ID_LINE_RE.match(line)
        if note:
            pending_id = note.group(1)
            scene = None
            previous_blank = True
            continue
        section = _SECTION_RE.match(stripped) if stripped.startswith("#") and not is_heading(stripped) else None
        if section:
            level, title = len(section.group(1)), section.group(2)
            if allow_breaks and title:
                start("break", title=title[:120], **({"level": min(level, 6)} if level > 1 else {}))
            else:
                dropped["sections"].append(stripped)
            previous_blank = True
            continue
        if stripped.startswith("=") and not stripped.startswith("==="):
            text = stripped[1:].strip()
            if scene is not None and not scene["omitted"] and not any(l.strip() for l in scene["lines"][1:]) and scene["lines"] and is_heading(scene["lines"][0]):
                scene["synopsis"].append(text)
                del scene["lines"][1:]  # only blank lines so far: the synopsis replaces them
            else:
                pending_synopsis.append(text)
            continue
        omitted = _OMITTED_RE.match(stripped) if previous_blank else None
        if omitted:
            new_scene(None, number=omitted.group(1), omitted=True)
            previous_blank = False
            continue
        if previous_blank and stripped and is_heading(stripped):
            number = None
            match = NUMBER_IN_HEADING_RE.search(stripped)
            if match:
                number = match.group(1)
                stripped = NUMBER_IN_HEADING_RE.sub("", stripped).rstrip()
            new_scene(stripped, number=number)
            previous_blank = False
            continue
        if scene is None:
            if not stripped:
                previous_blank = True
                continue
            new_scene(None)
        scene["lines"].append(line)
        previous_blank = not stripped and line != TWO_SPACES
    for item in items:
        if item["kind"] != "scene":
            continue
        body = item.pop("lines")
        while body and not body[-1].strip():
            body.pop()
        while body and not body[0].strip():
            body.pop(0)
        item["text"] = "\n".join(body)
        item["synopsis"] = "\n".join(item["synopsis"]).strip("\n")
        if item["number"] is not None and not NUMBER_RE.fullmatch(item["number"]):
            dropped["numbers"].append(item["number"])
            item["number"] = None
    return items


def parse(text: str) -> dict[str, Any]:
    """A whole Fountain script: {title_page, items, aside, dropped}. ``items`` are scenes
    ({kind: scene, id, text, synopsis, number, omitted}) and act breaks ({kind: break, id,
    title}) in order; ``aside`` are the boneyard's scenes; ``id`` is the note's id or None."""
    lines = _normalize(text).split("\n")
    title_page, after = _title_page(lines)
    main, bones, inline = _split_boneyard(lines[after:])
    dropped: dict[str, list[str]] = {"sections": [], "numbers": [], "comments": inline}
    items = _scenes(main, dropped, allow_breaks=True)
    aside: list[dict[str, Any]] = []
    for bone in bones:
        found = _scenes(bone, dropped, allow_breaks=False)
        aside.extend(item for item in found if item["kind"] == "scene" and (item["text"] or item["synopsis"] or item["omitted"]))
    for item in items + aside:
        if item.get("id") is not None:
            wanted = SCENE_ID_RE if item["kind"] == "scene" else BREAK_ID_RE
            if not wanted.fullmatch(item["id"]):
                item["id"] = None
    return {"title_page": title_page, "items": items, "aside": aside, "dropped": dropped}
