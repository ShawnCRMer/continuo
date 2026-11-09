"""Fountain emphasis and escapes inside text (twin: desktop/film-suite/src/script/editor/emphasis.js).

``runs(text)`` splits text into printed runs ``(text, style)``; ``style`` is a frozenset of
``bold``, ``italic``, ``underline``. The rules (fountain.io/syntax#emphasis, read the same way
by Beat and OpenDraft):

* ``***x***`` bold italic, ``**x**`` bold, ``*x*`` italic, ``_x_`` underline; they nest
  (``_**BRICK & STEEL**_``, ``_the *Leupold Mark 4* scope_``).
* A marker opens before a non-space and closes after one, on the same line; a marker
  without its partner stays text (a lone ``*`` or ``_`` prints, ``2 * 3`` stays).
* ``\\*`` and ``\\_`` are the literal character; the backslash never prints.

The scan is length preserving: every source character keeps its offset, markers become
open/close sentinels, so ``markers(text)`` names exactly the offsets that never print (the
editor hides those everywhere but on the caret line). ``plain`` is what prints: the page
engine wraps it, the PDF draws it.
"""
from __future__ import annotations

import re

_ESC = "\ue000"   # the backslash of an escape (never prints)
_MASK = "\ue001"  # the escaped character's slot (prints the real character, never a marker)
_PAD = "\ue002"   # the rest of a marker longer than its styles (never prints)
_OPEN = {"bold": "\ue010", "italic": "\ue012", "underline": "\ue014"}
_CLOSE = {"bold": "\ue011", "italic": "\ue013", "underline": "\ue015"}
_STYLE = {v: (k, 1) for k, v in _OPEN.items()} | {v: (k, -1) for k, v in _CLOSE.items()}
# (pattern, styles) in match order; the content is non-greedy and never crosses a line end.
_PASSES = [
    (re.compile(r"\*\*\*(?=[^\s*])([^\n]*?[^\s*])\*\*\*"), ("bold", "italic")),
    (re.compile(r"\*\*(?=[^\s*])([^\n]*?[^\s*])\*\*"), ("bold",)),
    (re.compile(r"\*(?=[^\s*])([^\n*]*?[^\s*])\*"), ("italic",)),
    (re.compile(r"\*(?=[^\s*])([^\s*])\*"), ("italic",)),
    (re.compile(r"_(?=[^\s_])([^\n_]*?[^\s_])_"), ("underline",)),
    (re.compile(r"_(?=[^\s_])([^\s_])_"), ("underline",)),
]


def _work(text: str) -> str:
    work = list(text)
    i = 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text) and text[i + 1] in "*_":
            work[i], work[i + 1] = _ESC, _MASK
            i += 2
            continue
        i += 1
    s = "".join(work)
    for pattern, styles in _PASSES:
        def sub(match: re.Match[str], styles: tuple[str, ...] = styles) -> str:
            half = (len(match.group(0)) - len(match.group(1))) // 2
            opens = "".join(_OPEN[st] for st in styles)
            closes = "".join(_CLOSE[st] for st in reversed(styles))
            return (opens + _PAD * half)[:half] + match.group(1) + (closes + _PAD * half)[:half]
        s = pattern.sub(sub, s)
    return s


def _walk(text: str) -> list[tuple[str, frozenset[str], int]]:
    """(character, style, offset) of every printed character."""
    depth = {"bold": 0, "italic": 0, "underline": 0}
    out: list[tuple[str, frozenset[str], int]] = []
    for i, w in enumerate(_work(text)):
        if w in _STYLE:
            name, step = _STYLE[w]
            depth[name] += step
            continue
        if w in (_ESC, _PAD):
            continue
        out.append((text[i], frozenset(k for k, v in depth.items() if v > 0), i))
    return out


def runs(text: str) -> list[tuple[str, frozenset[str]]]:
    out: list[tuple[str, frozenset[str]]] = []
    for ch, style, _i in _walk(text):
        if out and out[-1][1] == style:
            out[-1] = (out[-1][0] + ch, style)
        else:
            out.append((ch, style))
    return out


def plain(text: str) -> str:
    """The printed text: no emphasis markers, no escape backslashes."""
    return "".join(ch for ch, _s, _i in _walk(text))


def markers(text: str) -> list[int]:
    """Offsets in ``text`` that never print: emphasis markers and escape backslashes."""
    printed = {i for _ch, _s, i in _walk(text)}
    return [i for i in range(len(text)) if i not in printed]


def escape(text: str) -> str:
    """Literal text as Fountain: a backslash before ``*`` and ``_``."""
    return re.sub(r"([*_])", r"\\\1", text)


def markup(text: str, style: frozenset[str] | set[str] | tuple[str, ...]) -> str:
    """One styled run as Fountain (FDX import): underline outside, then bold, then italic;
    spaces at the ends stay outside the markers."""
    body = escape(text)
    core = body.strip()
    if not core or not style:
        return body
    lead, trail = body[: len(body) - len(body.lstrip())], body[len(body.rstrip()):]
    stars = ("**" if "bold" in style else "") + ("*" if "italic" in style else "")
    inner = stars + core + stars
    if "underline" in style:
        inner = "_" + inner + "_"
    return lead + inner + trail


__all__ = ["escape", "markers", "markup", "plain", "runs"]
