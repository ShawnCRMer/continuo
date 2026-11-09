"""Subtitles of a cut (lane w23-subs, CUT-173 / CUT-011): the pure rules of studio.subtitles/1.

One document per project (state/subtitles.json): ``timelines {<timeline id>: {track, cues}}`` and a revision. A cue is
``{id st-xxxxxxxx, start, end (record frames of that cut, end exclusive), text ('\\n' = a typed line break), words
[{w, start, end}] (the text's tokens in order, with record frames), edited, by}``. Frames against the cut's rate, so
SRT / WebVTT times are frame-exact by construction (``frame * den / num`` to the millisecond).

Reference (docs/studies/w23-subs-STUDY.md): Resolve's subtitle track and Create Subtitles from Audio (max characters
per line, lines, gap), Premiere's Create captions (length, duration, gap, single line / single word), FCP's caption
clips (move, trim, blade, split); the numbers are the Netflix Timed Text Style Guide's (42 characters, two lines,
5/6 s - 7 s, 2-frame gap, chaining under 0.5 s, shot changes) with the task's stricter 1 s minimum and 17 cps, and
the BBC's 0.3 s per word. Every number is a track rule the artist (or Hermes) can change.

No I/O here: the store (subtitle_store.py), the routes (routes/projects/subtitles_1.py) and the rasteriser
(subtitle_raster.py) use these functions.
"""
from __future__ import annotations

import copy
import difflib
import math
import re
import secrets
from fractions import Fraction
from typing import Any, Iterable

SCHEMA = "studio.subtitles/1"
MAX_CUES = 5000
MAX_TEXT = 500
MAX_WORDS = 200
TRACK_ID = "ST1"

PRESETS: dict[str, dict[str, Any]] = {
    # Film / TV: bottom centre, up to two lines on a soft box (Netflix TTSG; BBC 68 % width online).
    "classic": {
        "style": {"preset": "classic", "font": "sans", "size": 5.0, "color": "#FFFFFF", "highlight": "#FFFFFF",
                  "background": "#000000", "background_opacity": 0.6, "outline": 0.0, "outline_color": "#000000",
                  "x": 50.0, "y": 92.0, "uppercase": False, "active_word": False},
        "rules": {"max_chars": 42, "lines": 2, "min_seconds": 1.0, "max_seconds": 7.0, "cps": 17.0, "gap_frames": 2,
                  "max_words": 0, "pause_seconds": 0.7, "linger_seconds": 0.5, "chain_seconds": 0.5},
    },
    # Social captions: word by word, the spoken word highlighted, larger, centred (CapCut / Descript active word).
    # Lane w24-rest (owner look review): in the lower third above the platform UI safe zone (CapCut / Submagic place
    # social captions there, not mid-frame - y 72 % read as mid-frame) with a heavy stroke that holds over bright surf.
    "social": {
        "style": {"preset": "social", "font": "sans_bold", "size": 8.0, "color": "#FFFFFF", "highlight": "#FFD60A",
                  "background": "#000000", "background_opacity": 0.0, "outline": 20.0, "outline_color": "#000000",
                  "x": 50.0, "y": 85.0, "uppercase": True, "active_word": True},
        "rules": {"max_chars": 20, "lines": 1, "min_seconds": 0.4, "max_seconds": 3.0, "cps": 25.0, "gap_frames": 0,
                  "max_words": 3, "pause_seconds": 0.5, "linger_seconds": 0.2, "chain_seconds": 0.5},
    },
}
FONTS = ("sans", "sans_bold", "serif", "mono")
STYLE_KEYS = tuple(PRESETS["classic"]["style"])
RULE_KEYS = tuple(PRESETS["classic"]["rules"])
_RULE_BOUNDS = {"max_chars": (8, 80), "lines": (1, 3), "min_seconds": (0.1, 10.0), "max_seconds": (0.5, 20.0), "cps": (5.0, 60.0),
                "gap_frames": (0, 24), "max_words": (0, 50), "pause_seconds": (0.1, 5.0), "linger_seconds": (0.0, 3.0),
                "chain_seconds": (0.0, 3.0)}
_STYLE_BOUNDS = {"size": (1.0, 20.0), "background_opacity": (0.0, 1.0), "outline": (0.0, 40.0), "x": (0.0, 100.0), "y": (0.0, 100.0)}
_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
_ID = re.compile(r"^st-[0-9a-f]{8}$")
VERBS = ("set_text", "set_times", "move", "split", "merge", "delete", "add", "set_track", "resplit")

# Netflix TTSG I.10: break after punctuation, before conjunctions and prepositions; never after an article,
# a subject pronoun or an auxiliary (it would separate it from its noun / verb).
_WEAK_END = frozenset("a an the my your his her its our their this that these those to of in on at by for with from "
                      "and or but nor so i you he she we they is are was were be been am will would can could shall "
                      "should may might must do does did has have had not mr mrs ms dr".split())
_GOOD_START = frozenset("and or but nor so because when while if that which who where after before until than "
                        "to of in on at by for with from into over under about through".split())


class SubtitleRefusal(ValueError):
    def __init__(self, reason_code: str, message: str, index: int | None = None) -> None:
        super().__init__(message)
        self.reason_code, self.message, self.index = reason_code, message, index


def fail(code: str, message: str, index: int | None = None) -> None:
    raise SubtitleRefusal(code, message, index)


# --- document ---------------------------------------------------------------------------------------------------

def empty() -> dict[str, Any]:
    return {"schema": SCHEMA, "revision": 0, "timelines": {}}


def default_track(preset: str = "classic") -> dict[str, Any]:
    p = PRESETS[preset]
    return {"id": TRACK_ID, "enabled": True, "style": dict(p["style"]), "rules": dict(p["rules"])}


def timeline_entry(doc: dict[str, Any], timeline_id: str, *, create: bool = False) -> dict[str, Any]:
    entry = doc["timelines"].get(timeline_id)
    if entry is None:
        entry = {"track": default_track(), "cues": []}
        if create:
            doc["timelines"][timeline_id] = entry
    return entry


def new_id(taken: Iterable[str] = ()) -> str:
    taken = set(taken)
    while True:
        cid = "st-" + secrets.token_hex(4)
        if cid not in taken:
            return cid


def _check_style(style: Any) -> str | None:
    if not isinstance(style, dict) or set(style) != set(STYLE_KEYS):
        return "style keys"
    if style["preset"] not in PRESETS or style["font"] not in FONTS:
        return "style preset/font"
    for key in ("color", "highlight", "background", "outline_color"):
        if not isinstance(style[key], str) or not _COLOR.match(style[key]):
            return f"style {key}"
    for key, (lo, hi) in _STYLE_BOUNDS.items():
        v = style[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
            return f"style {key}"
    if not isinstance(style["uppercase"], bool) or not isinstance(style["active_word"], bool):
        return "style flags"
    return None


def _check_rules(rules: Any) -> str | None:
    if not isinstance(rules, dict) or set(rules) != set(RULE_KEYS):
        return "rules keys"
    for key, (lo, hi) in _RULE_BOUNDS.items():
        v = rules[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
            return f"rules {key}"
        if key in ("max_chars", "lines", "gap_frames", "max_words") and not isinstance(v, int):
            return f"rules {key}"
    if rules["min_seconds"] > rules["max_seconds"]:
        return "rules min > max"
    return None


def _check_cue(cue: Any) -> str | None:
    if not isinstance(cue, dict) or set(cue) != {"id", "start", "end", "text", "words", "edited", "by"}:
        return "cue keys"
    if not isinstance(cue["id"], str) or not _ID.match(cue["id"]):
        return "cue id"
    for key in ("start", "end"):
        if isinstance(cue[key], bool) or not isinstance(cue[key], int) or cue[key] < 0:
            return "cue frames"
    if cue["end"] <= cue["start"]:
        return "cue length"
    if not isinstance(cue["text"], str) or not cue["text"].strip() or len(cue["text"]) > MAX_TEXT:
        return "cue text"
    if not isinstance(cue["edited"], bool) or cue["by"] not in ("artist", "hermes", "transcription"):
        return "cue flags"
    words = cue["words"]
    if not isinstance(words, list) or len(words) > MAX_WORDS:
        return "cue words"
    for w in words:
        if not isinstance(w, dict) or set(w) != {"w", "start", "end"} or not isinstance(w["w"], str):
            return "word keys"
        if any(isinstance(w[k], bool) or not isinstance(w[k], int) for k in ("start", "end")) or w["end"] < w["start"]:
            return "word frames"
    if words and [w["w"] for w in words] != tokens(cue["text"]):
        return "words do not match the text"
    return None


def check_document(doc: Any) -> str | None:
    if not isinstance(doc, dict) or set(doc) != {"schema", "revision", "timelines"} or doc["schema"] != SCHEMA:
        return "document keys"
    if isinstance(doc["revision"], bool) or not isinstance(doc["revision"], int) or doc["revision"] < 0:
        return "revision"
    if not isinstance(doc["timelines"], dict) or len(doc["timelines"]) > 200:
        return "timelines"
    for tid, entry in doc["timelines"].items():
        if not isinstance(tid, str) or not 1 <= len(tid) <= 120 or not isinstance(entry, dict) or set(entry) != {"track", "cues"}:
            return "timeline entry"
        track = entry["track"]
        # Lane w24-rest (ST1 lock): "locked" is optional (documents written before it read unlocked).
        if not isinstance(track, dict) or set(track) - {"locked"} != {"id", "enabled", "style", "rules"} or track["id"] != TRACK_ID or not isinstance(track["enabled"], bool)                 or not isinstance(track.get("locked", False), bool):
            return "track"
        problem = _check_style(track["style"]) or _check_rules(track["rules"])
        if problem:
            return problem
        cues = entry["cues"]
        if not isinstance(cues, list) or len(cues) > MAX_CUES:
            return "cues"
        ids = set()
        for i, cue in enumerate(cues):
            problem = _check_cue(cue)
            if problem:
                return f"cues[{i}]: {problem}"
            if cue["id"] in ids:
                return f"cues[{i}]: duplicate id"
            ids.add(cue["id"])
            if i and cues[i - 1]["end"] > cue["start"]:
                return f"cues[{i}]: overlaps the cue before"
    return None


# --- text ---------------------------------------------------------------------------------------------------------

def tokens(text: str) -> list[str]:
    return text.split()


def _bare(word: str) -> str:
    return re.sub(r"[^\w']+", "", word.casefold())


def _break_score(top: list[str], bottom: list[str]) -> float:
    a, b = len(" ".join(top)), len(" ".join(bottom))
    score = abs(a - b) + (4 if a > b else 0)          # balanced, bottom-heavy pyramid
    last, first = top[-1], bottom[0]
    if re.search(r"[.,;:!?…]$", last):
        score -= 30                                    # after punctuation (first choice in the Netflix list)
    elif _bare(last) in _WEAK_END:
        score += 25                                    # never after an article / pronoun / auxiliary
    if _bare(first) in _GOOD_START:
        score -= 6                                     # before a conjunction or preposition
    if len(top) == 1 or len(bottom) == 1:
        score += 8                                     # no orphan word on a line
    return score


def layout_lines(text: str, rules: dict[str, Any]) -> list[str]:
    """The cue's lines as shown and exported: typed line breaks kept; an over-long line is broken at the best point
    (Netflix line treatment), into at most ``rules.lines`` lines when it fits, else as few as fit ``max_chars``."""
    out: list[str] = []
    limit = int(rules["max_chars"])
    for typed in text.split("\n"):
        words = typed.split()
        if not words:
            continue
        out.extend(_wrap(words, limit))
    return out or [text.strip()]


def _wrap(words: list[str], limit: int) -> list[str]:
    line = " ".join(words)
    if len(line) <= limit or len(words) == 1:
        return [line]
    best = None
    for k in range(1, len(words)):
        top, bottom = words[:k], words[k:]
        if len(" ".join(top)) > limit:
            break
        if len(" ".join(bottom)) > limit:
            continue
        score = _break_score(top, bottom)
        if best is None or score < best[0]:
            best = (score, k)
    if best is not None:
        return [" ".join(words[:best[1]]), " ".join(words[best[1]:])]
    # Too long for two lines: fill greedily (the QC flag 'chars' names it).
    out, cur = [], []
    for w in words:
        if cur and len(" ".join(cur + [w])) > limit:
            out.append(" ".join(cur))
            cur = []
        cur.append(w)
    if cur:
        out.append(" ".join(cur))
    return out


def fits(words: list[str], rules: dict[str, Any]) -> bool:
    lines = layout_lines(" ".join(words), rules)
    return len(lines) <= int(rules["lines"]) and all(len(l) <= int(rules["max_chars"]) for l in lines)


def display_text(text: str, rules: dict[str, Any]) -> str:
    return "\n".join(layout_lines(text, rules))


# --- time -----------------------------------------------------------------------------------------------------------

def _rate(rate: dict[str, int]) -> Fraction:
    return Fraction(int(rate["num"]), int(rate["den"]))


def frames_of(seconds: float, rate: dict[str, int]) -> int:
    return int(round(Fraction(seconds).limit_denominator(100000) * _rate(rate)))


def ms_of(frame: int, rate: dict[str, int]) -> int:
    """The millisecond of a frame's start, rounded half up (exact: back to the same frame for any rate <= 1000)."""
    value = Fraction(frame) / _rate(rate) * 1000
    return int(math.floor(value + Fraction(1, 2)))


def frame_of_ms(ms: int, rate: dict[str, int]) -> int:
    return int(math.floor(Fraction(ms, 1000) * _rate(rate) + Fraction(1, 2)))


def _stamp(ms: int, sep: str) -> str:
    h, rest = divmod(ms, 3600000)
    m, rest = divmod(rest, 60000)
    s, milli = divmod(rest, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{milli:03d}"


def srt(cues: list[dict[str, Any]], rate: dict[str, int], rules: dict[str, Any], style: dict[str, Any] | None = None) -> str:
    """SubRip: numbered cues, HH:MM:SS,mmm, the lines as shown. UTF-8, CRLF-free, blank line between cues."""
    parts = []
    for n, cue in enumerate(cues, 1):
        text = display_text(cue["text"], rules)
        if style is not None and style.get("uppercase"):
            text = text.upper()
        parts.append(f"{n}\n{_stamp(ms_of(cue['start'], rate), ',')} --> {_stamp(ms_of(cue['end'], rate), ',')}\n{text}\n")
    return "\n".join(parts)


def vtt(cues: list[dict[str, Any]], rate: dict[str, int], rules: dict[str, Any], style: dict[str, Any] | None = None) -> str:
    """WebVTT (W3C): header, cue ids, HH:MM:SS.mmm, line:/position: from the track style; '&', '<', '>' escaped."""
    out = ["WEBVTT", ""]
    settings = ""
    if style is not None:
        settings = f" line:{int(round(style['y']))}% position:{int(round(style['x']))}% align:center"
    for cue in cues:
        text = display_text(cue["text"], rules)
        if style is not None and style.get("uppercase"):
            text = text.upper()
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        out += [cue["id"], f"{_stamp(ms_of(cue['start'], rate), '.')} --> {_stamp(ms_of(cue['end'], rate), '.')}{settings}", text, ""]
    return "\n".join(out)


def parse_srt(text: str) -> list[tuple[int, int, str]]:
    """(start ms, end ms, text) of every SRT cue (used by the validators and tests)."""
    out = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.strip().split("\n")
        if len(lines) < 3 or not lines[0].strip().isdigit():
            raise ValueError(f"bad SRT block: {block[:40]!r}")
        m = re.fullmatch(r"(\d\d):(\d\d):(\d\d),(\d{3}) --> (\d\d):(\d\d):(\d\d),(\d{3})", lines[1].strip())
        if not m:
            raise ValueError(f"bad SRT time line: {lines[1]!r}")
        g = [int(x) for x in m.groups()]
        out.append(((g[0] * 3600 + g[1] * 60 + g[2]) * 1000 + g[3], (g[4] * 3600 + g[5] * 60 + g[6]) * 1000 + g[7], "\n".join(lines[2:])))
    return out


# --- QC (Resolve: CPS red over the limit; FCP: the caption turns red) ------------------------------------------------

def qc(cue: dict[str, Any], rate: dict[str, int], rules: dict[str, Any]) -> list[str]:
    flags = []
    seconds = Fraction(cue["end"] - cue["start"]) / _rate(rate)
    chars = len(cue["text"].replace("\n", " ").strip())
    if seconds > 0 and Fraction(chars) / seconds > Fraction(rules["cps"]).limit_denominator(1000) + Fraction(1, 100):
        flags.append("cps")
    lines = layout_lines(cue["text"], rules)
    if len(lines) > int(rules["lines"]) or any(len(l) > int(rules["max_chars"]) for l in lines):
        flags.append("chars")
    if cue["end"] - cue["start"] < frames_of(rules["min_seconds"], rate):
        flags.append("short")
    if cue["end"] - cue["start"] > frames_of(rules["max_seconds"], rate):
        flags.append("long")
    return flags


# --- building cues from timed words (Create subtitles from audio) --------------------------------------------------

def build_cues(words: list[dict[str, Any]], *, rate: dict[str, int], rules: dict[str, Any], cuts: Iterable[int] = (),
               limit_end: int | None = None, taken: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Split record-frame words into cues by the rules: a new cue at a pause, a sentence end, a picture cut the speech
    does not cross, or when the next word would break the line, word, duration limits. Then time each cue: in on the
    first word, out half a second after the last (linger) or at the reading-speed and minimum duration, never past
    the next cue minus the gap, never across a cut the speech did not cross, gaps under chain_seconds closed."""
    words = [w for w in sorted(words, key=lambda w: (w["start"], w["end"])) if w["w"].strip()]
    cuts = sorted(set(int(c) for c in cuts))
    pause = frames_of(rules["pause_seconds"], rate)
    max_frames = frames_of(rules["max_seconds"], rate)
    max_words = int(rules["max_words"])
    groups: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []

    def cut_between(a: int, b: int) -> bool:
        return any(a <= c <= b for c in cuts)

    for w in words:
        if cur:
            prev = cur[-1]
            text_now = [x["w"] for x in cur]
            reasons = (
                w["start"] - prev["end"] >= pause,
                cut_between(prev["end"], w["start"]),
                not fits(text_now + [w["w"]], rules),
                w["end"] - cur[0]["start"] > max_frames,
                bool(max_words) and len(cur) >= max_words,
                re.search(r"[.!?…]$", prev["w"]) is not None and len(" ".join(text_now)) >= int(rules["max_chars"]) * 0.5,
            )
            if any(reasons):
                groups.append(cur)
                cur = []
        cur.append(w)
    if cur:
        groups.append(cur)

    taken = set(taken)
    gap = int(rules["gap_frames"])
    linger = frames_of(rules["linger_seconds"], rate)
    chain = frames_of(rules["chain_seconds"], rate)
    min_frames = frames_of(rules["min_seconds"], rate)
    cues = []
    for g in groups:
        cid = new_id(taken)
        taken.add(cid)
        text = display_text(" ".join(x["w"] for x in g), rules)
        cues.append({"id": cid, "start": g[0]["start"], "end": max(g[-1]["end"], g[0]["start"] + 1), "text": text,
                     "words": [{"w": x["w"], "start": x["start"], "end": x["end"]} for x in g], "edited": False, "by": "transcription"})
    for i, cue in enumerate(cues):
        speech_end = cue["end"]
        chars = len(cue["text"].replace("\n", " "))
        reading = int(math.ceil(Fraction(chars) / Fraction(rules["cps"]).limit_denominator(1000) * _rate(rate)))
        target = max(speech_end + linger, cue["start"] + min_frames, cue["start"] + reading)
        target = min(target, max(cue["start"] + max_frames, speech_end))
        nxt = cues[i + 1]["start"] if i + 1 < len(cues) else None
        if nxt is not None:
            target = min(target, nxt - gap)
            if nxt - gap - target < chain:            # chaining: close a short gap to the rule's gap
                target = nxt - gap
        after = [c for c in cuts if speech_end <= c < target]
        if after:                                     # the speech ends before this cut: do not hang over it
            target = after[0]
        if limit_end is not None:
            target = min(target, limit_end)
        cue["end"] = max(target, speech_end if nxt is None else min(speech_end, nxt - gap), cue["start"] + 1)
    return cues


def merge_transcription(existing: list[dict[str, Any]], new: list[dict[str, Any]], lo: int, hi: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """A new transcription of [lo, hi): unedited cues in that range are replaced; every edited cue stays and new cues
    that would overlap it are dropped (the artist's and Hermes' corrections survive)."""
    kept = [c for c in existing if c["edited"] or c["end"] <= lo or c["start"] >= hi]
    replaced = len(existing) - len(kept)
    added = 0
    for cue in new:
        if any(cue["start"] < c["end"] and c["start"] < cue["end"] for c in kept):
            clash = [c for c in kept if cue["start"] < c["end"] and c["start"] < cue["end"]]
            if all(not c["edited"] for c in clash):
                continue
            # The same speech as an edited cue (most of its words fall inside it, also after the cue was moved or
            # retimed by hand): the edit stands for it; never leave a fragment of it next to the edit.
            inside = sum(1 for w in cue["words"] if any(c["start"] <= (w["start"] + w["end"]) / 2 < c["end"] for c in clash))
            if cue["words"] and inside * 2 >= len(cue["words"]):
                continue
            # trim the new cue off an edited neighbour where the speech allows it, else drop it
            start, end = cue["start"], cue["end"]
            for c in clash:
                if c["start"] <= start < c["end"]:
                    start = c["end"]
                if c["start"] < end <= c["end"]:
                    end = c["start"]
            words = [w for w in cue["words"] if w["start"] >= start and w["end"] <= end]
            if end - start < 1 or not words or any(start < c["end"] and c["start"] < end for c in kept):
                continue
            cue = {**cue, "start": start, "end": end, "words": words, "text": " ".join(w["w"] for w in words)}
        kept.append(cue)
        added += 1
    kept.sort(key=lambda c: c["start"])
    return kept, {"replaced": replaced, "added": added, "kept_edited": sum(1 for c in kept if c["edited"])}


# --- edits (hand = Hermes) ------------------------------------------------------------------------------------------

def _find(cues: list[dict[str, Any]], cid: Any, index: int) -> int:
    for i, c in enumerate(cues):
        if c["id"] == cid:
            return i
    fail("cue_unknown", f"ops[{index}]: no cue {cid}", index)
    raise AssertionError


def _int(value: Any, name: str, index: int, lo: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or (lo is not None and value < lo):
        fail("invalid_op", f"ops[{index}]: {name} must be a whole frame" + (f" >= {lo}" if lo is not None else ""), index)
    return value


def _text(value: Any, index: int) -> str:
    if not isinstance(value, str):
        fail("invalid_op", f"ops[{index}]: text must be a string", index)
    text = "\n".join(" ".join(line.split()) for line in value.replace("\r\n", "\n").split("\n")).strip("\n")
    if not text.strip():
        fail("text_empty", f"ops[{index}]: the text is empty (delete the cue instead)", index)
    if len(text) > MAX_TEXT:
        fail("text_too_long", f"ops[{index}]: at most {MAX_TEXT} characters", index)
    if len(tokens(text)) > MAX_WORDS:
        fail("text_too_long", f"ops[{index}]: at most {MAX_WORDS} words", index)
    return text


def retime_words(old: list[dict[str, Any]], text: str, start: int, end: int) -> list[dict[str, Any]]:
    """Words of a typed text: a word that matches the old words keeps its time; the others share the time between
    their timed neighbours by their length (no word is invented, none loses its place)."""
    new = tokens(text)
    if not new:
        return []
    if len(old) == len(new):
        return [{"w": n, "start": o["start"], "end": o["end"]} for n, o in zip(new, old)]
    times: list[tuple[int, int] | None] = [None] * len(new)
    matcher = difflib.SequenceMatcher(a=[_bare(w["w"]) for w in old], b=[_bare(w) for w in new], autojunk=False)
    for a, b, size in matcher.get_matching_blocks():
        for k in range(size):
            times[b + k] = (old[a + k]["start"], old[a + k]["end"])
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(new):
        if times[i] is not None:
            out.append({"w": new[i], "start": times[i][0], "end": times[i][1]})
            i += 1
            continue
        j = i
        while j < len(new) and times[j] is None:
            j += 1
        lo = out[-1]["end"] if out else start
        hi = times[j][0] if j < len(new) else end
        hi = max(hi, lo)
        span, run = hi - lo, new[i:j]
        total = sum(len(w) + 1 for w in run)
        at, acc = lo, 0
        for w in run:
            acc += len(w) + 1
            nxt = lo + (span * acc) // total
            out.append({"w": w, "start": at, "end": max(at, nxt)})
            at = max(at, nxt)
        i = j
    return out


def _sorted_ok(cues: list[dict[str, Any]], index: int) -> None:
    cues.sort(key=lambda c: (c["start"], c["end"]))
    for a, b in zip(cues, cues[1:]):
        if a["end"] > b["start"]:
            fail("overlap", f"ops[{index}]: {a['id']} and {b['id']} would overlap", index)


def _clamp_words(cue: dict[str, Any]) -> None:
    for w in cue["words"]:
        w["start"] = min(max(w["start"], cue["start"]), cue["end"])
        w["end"] = min(max(w["end"], w["start"]), cue["end"])


def apply_ops(entry: dict[str, Any], ops: list[dict[str, Any]], *, by: str, rate: dict[str, int], cuts: Iterable[int] = (),
              total_frames: int | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply edit verbs to one cut's subtitle entry, atomically (a refusal leaves the entry as it was)."""
    entry = copy.deepcopy(entry)
    cues: list[dict[str, Any]] = entry["cues"]
    results: list[dict[str, Any]] = []
    for index, op in enumerate(ops):
        if not isinstance(op, dict) or op.get("op") not in VERBS:
            fail("invalid_op", f"ops[{index}]: op must be one of " + ", ".join(VERBS), index)
        verb = op["op"]
        track = entry["track"]
        # Lane w24-rest (ST1 lock, owner decision: parity with V/A tracks): a locked ST1 refuses every edit with the code a
        # locked picture track answers ("locked"); only the lock itself and Show / Hide stay (Resolve: a locked track can
        # still be hidden).
        if track.get("locked") and (verb != "set_track" or set(op) - {"op", "locked", "enabled"}):
            fail("locked", f"ops[{index}]: track {TRACK_ID} is locked", index)
        if verb == "set_track":
            if "locked" in op:
                if not isinstance(op["locked"], bool):
                    fail("invalid_op", f"ops[{index}]: locked is true or false", index)
                track["locked"] = op["locked"]
            if "preset" in op:
                if op["preset"] not in PRESETS:
                    fail("preset_unknown", f"ops[{index}]: preset must be classic or social", index)
                track["style"] = dict(PRESETS[op["preset"]]["style"])
                track["rules"] = dict(PRESETS[op["preset"]]["rules"])
            if "enabled" in op:
                if not isinstance(op["enabled"], bool):
                    fail("invalid_op", f"ops[{index}]: enabled is true or false", index)
                track["enabled"] = op["enabled"]
            for key, holder, check in (("style", track["style"], _check_style), ("rules", track["rules"], _check_rules)):
                patch = op.get(key)
                if patch is None:
                    continue
                if not isinstance(patch, dict) or any(k not in holder or k == "preset" for k in patch):
                    fail(f"{key}_invalid", f"ops[{index}]: unknown {key} key", index)
                trial = {**holder, **patch}
                problem = check(trial)
                if problem:
                    fail(f"{key}_invalid", f"ops[{index}]: {problem}", index)
                holder.update(patch)
            results.append({"op": verb, "track": copy.deepcopy(track)})
            continue
        if verb == "add":
            start, end = _int(op.get("start"), "start", index, 0), _int(op.get("end"), "end", index, 1)
            if end <= start:
                fail("invalid_op", f"ops[{index}]: end must be after start", index)
            text = _text(op.get("text"), index)
            cue = {"id": new_id(c["id"] for c in cues), "start": start, "end": end, "text": text,
                   "words": retime_words([], text, start, end), "edited": True, "by": by}
            cues.append(cue)
            _sorted_ok(cues, index)
            results.append({"op": verb, "id": cue["id"]})
            continue
        if verb == "resplit":
            lo = _int(op.get("in_frame", 0), "in_frame", index, 0)
            hi = op.get("out_frame")
            hi = (max((c["end"] for c in cues), default=0) + 1) if hi is None else _int(hi, "out_frame", index, 1)
            words = [w for c in cues if not c["edited"] and c["start"] >= lo and c["end"] <= hi for w in c["words"]]
            fresh = build_cues(words, rate=rate, rules=track["rules"], cuts=cuts, limit_end=total_frames, taken=[c["id"] for c in cues])
            entry["cues"], counts = merge_transcription(cues, fresh, lo, hi)
            cues = entry["cues"]
            results.append({"op": verb, **counts})
            continue
        if verb == "merge":
            ids = op.get("ids")
            if not isinstance(ids, list) or len(ids) < 2 or len(set(map(str, ids))) != len(ids):
                fail("invalid_op", f"ops[{index}]: merge takes ids (two or more cues)", index)
            picked = sorted((_find(cues, cid, index) for cid in ids))
            if picked != list(range(picked[0], picked[0] + len(picked))):
                fail("not_adjacent", f"ops[{index}]: the cues to merge are not neighbours", index)
            parts = [cues[i] for i in picked]
            text = " ".join(" ".join(p["text"].split()) for p in parts)
            if len(text) > MAX_TEXT:
                fail("text_too_long", f"ops[{index}]: the merged text is longer than {MAX_TEXT} characters", index)
            merged = {"id": parts[0]["id"], "start": parts[0]["start"], "end": parts[-1]["end"], "text": text,
                      "words": [w for p in parts for w in p["words"]] if all(p["words"] for p in parts) else retime_words([], text, parts[0]["start"], parts[-1]["end"]),
                      "edited": True, "by": by}
            cues[picked[0]:picked[-1] + 1] = [merged]
            results.append({"op": verb, "id": merged["id"], "removed": [p["id"] for p in parts[1:]]})
            continue
        i = _find(cues, op.get("id"), index)
        cue = cues[i]
        if verb == "delete":
            del cues[i]
            results.append({"op": verb, "id": cue["id"]})
            continue
        if verb == "set_text":
            text = _text(op.get("text"), index)
            cue["words"] = retime_words(cue["words"], text, cue["start"], cue["end"])
            cue["text"] = text
        elif verb == "set_times":
            start = cue["start"] if op.get("start") is None else _int(op.get("start"), "start", index, 0)
            end = cue["end"] if op.get("end") is None else _int(op.get("end"), "end", index, 1)
            if end <= start:
                fail("invalid_op", f"ops[{index}]: end must be after start", index)
            cue["start"], cue["end"] = start, end
            _clamp_words(cue)
        elif verb == "move":
            delta = _int(op.get("delta"), "delta", index)
            if cue["start"] + delta < 0:
                fail("invalid_op", f"ops[{index}]: the cue would start before the cut", index)
            cue["start"] += delta
            cue["end"] += delta
            for w in cue["words"]:
                w["start"] += delta
                w["end"] += delta
        elif verb == "split":
            at = _int(op.get("at_frame"), "at_frame", index, 0)
            if not cue["start"] < at < cue["end"]:
                fail("split_outside", f"ops[{index}]: at_frame must lie inside the cue", index)
            words = cue["words"] or retime_words([], cue["text"], cue["start"], cue["end"])
            k = sum(1 for w in words if (w["start"] + w["end"]) / 2 < at)
            k = min(max(k, 1), len(words) - 1) if len(words) > 1 else 0
            if len(words) < 2:
                fail("split_one_word", f"ops[{index}]: a one-word cue cannot be split", index)
            left = {"id": cue["id"], "start": cue["start"], "end": at, "text": " ".join(w["w"] for w in words[:k]),
                    "words": [dict(w) for w in words[:k]], "edited": True, "by": by}
            right = {"id": new_id(c["id"] for c in cues), "start": at, "end": cue["end"], "text": " ".join(w["w"] for w in words[k:]),
                     "words": [dict(w) for w in words[k:]], "edited": True, "by": by}
            for part in (left, right):
                _clamp_words(part)
            cues[i:i + 1] = [left, right]
            results.append({"op": verb, "id": left["id"], "created": right["id"]})
            _sorted_ok(cues, index)
            continue
        cue["edited"], cue["by"] = True, by
        _sorted_ok(cues, index)
        results.append({"op": verb, "id": cue["id"]})
    problem = check_document({"schema": SCHEMA, "revision": 0, "timelines": {"t": entry}})
    if problem:
        fail("invalid_op", f"the result would be invalid ({problem})")
    return entry, results


# --- states (what is on screen when): one per cue, or one per spoken word with the active word ---------------------

def states(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """[{start, end, cue, active}] covering every cue: Classic = the cue; Social (active_word) = one state per word,
    from that word's start to the next word's start (the first from the cue's in-time)."""
    out = []
    active_word = entry["track"]["style"]["active_word"]
    for cue in entry["cues"]:
        words = cue["words"]
        if not active_word or not words:
            out.append({"start": cue["start"], "end": cue["end"], "cue": cue["id"], "active": None})
            continue
        bounds = [cue["start"]] + [min(max(w["start"], cue["start"]), cue["end"]) for w in words[1:]] + [cue["end"]]
        for k in range(len(words)):
            a, b = bounds[k], max(bounds[k + 1], bounds[k])
            if b > a:
                out.append({"start": a, "end": b, "cue": cue["id"], "active": k})
    return out


def crop(entry: dict[str, Any], in_frame: int, out_frame: int) -> dict[str, Any]:
    """The cues of record frames [in, out) rebased to 0 (an export range): cues are clipped at the range ends."""
    out = copy.deepcopy(entry)
    cues = []
    for cue in out["cues"]:
        if cue["end"] <= in_frame or cue["start"] >= out_frame:
            continue
        cue["start"], cue["end"] = max(cue["start"], in_frame) - in_frame, min(cue["end"], out_frame) - in_frame
        for w in cue["words"]:
            w["start"] = min(max(w["start"], in_frame), out_frame) - in_frame
            w["end"] = min(max(w["end"], in_frame), out_frame) - in_frame
        cues.append(cue)
    out["cues"] = cues
    return out


def public(entry: dict[str, Any], rate: dict[str, int]) -> dict[str, Any]:
    """A cut's subtitles as the page and Hermes read them: track, cues with their shown lines and QC flags."""
    rules = entry["track"]["rules"]
    return {"track": copy.deepcopy(entry["track"]),
            "cues": [{**copy.deepcopy(c), "lines": layout_lines(c["text"], rules), "qc": qc(c, rate, rules)} for c in entry["cues"]]}
