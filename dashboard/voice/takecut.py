"""Re-cut pauses of a take without re-speaking it (lane w4-audio, board E15 ⑭).

The take's words carry times (``[{w, start, end}]``, seconds, from the local recognizer).  A pause
is the gap between two words.  ``plan`` turns the asked pause lengths into a list of pieces of the
source (kept speech and the kept part of each pause) and digital silence (a pause made longer),
cut in the middle of each pause so no word is touched; ``splice`` applies the plan to 16-bit PCM
with a 4 ms fade at every join (no clicks); the new word times are the old ones shifted by every
earlier change.  Standalone (stdlib only); the route reads and writes the files.
"""
from __future__ import annotations

import array
import math
from typing import Any, Mapping

MAX_PAUSE_SECONDS = 10.0
FADE_SECONDS = 0.004


class TakeCutError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _seconds(value: Any, where: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= MAX_PAUSE_SECONDS:
        raise TakeCutError("invalid_pauses", f"{where} must be 0 to {MAX_PAUSE_SECONDS:g} seconds")
    return float(value)


def clean_words(words: Any) -> list[dict[str, Any]]:
    """Words in time order with start <= end (raises when there are none to cut between)."""
    out = []
    for item in words or []:
        if not isinstance(item, Mapping):
            continue
        start, end = item.get("start"), item.get("end")
        if type(start) not in (int, float) or type(end) not in (int, float) or not math.isfinite(start) or not math.isfinite(end):
            continue
        out.append({"w": str(item.get("w") or "")[:80], "start": max(0.0, float(start)), "end": max(float(start), float(end))})
    out.sort(key=lambda w: w["start"])
    for left, right in zip(out, out[1:]):
        if right["start"] < left["end"]:
            right["start"] = left["end"]
            right["end"] = max(right["end"], right["start"])
    return out


def pauses_of(words: list[dict[str, Any]]) -> list[float]:
    """The pause after each word but the last (seconds, 3 decimals)."""
    return [round(max(0.0, b["start"] - a["end"]), 3) for a, b in zip(words, words[1:])]


def plan(words: Any, total_seconds: float, *, gaps: Mapping[Any, Any] | None = None, max_pause: Any = None,
         end_pause: Any = None) -> dict[str, Any]:
    """{pieces: [("keep", a, b) | ("silence", seconds)], words, pauses_before, pauses_after, seconds}.

    ``gaps`` {index: seconds} sets the pause after word ``index``; ``max_pause`` shortens every longer
    pause to it; ``end_pause`` sets the sound after the last word.  Nothing asked = refused.
    """
    timed = clean_words(words)
    if len(timed) < 2 and end_pause is None:
        raise TakeCutError("no_words", "The take needs word timings with at least two words")
    if not timed:
        raise TakeCutError("no_words", "The take has no word timings yet")
    total = float(total_seconds)
    before = pauses_of(timed)
    wanted = list(before)
    if max_pause is not None:
        cap = _seconds(max_pause, "max_pause")
        wanted = [min(g, cap) for g in wanted]
    for key, value in dict(gaps or {}).items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            raise TakeCutError("invalid_pauses", "a pause is named by the index of the word before it") from None
        if not 0 <= index < len(before):
            raise TakeCutError("invalid_pauses", f"pause {index} does not exist (the take has {len(before)})")
        wanted[index] = _seconds(value, f"pause {index}")
    wanted = [round(g, 3) for g in wanted]
    tail_before = max(0.0, total - timed[-1]["end"])
    tail = _seconds(end_pause, "end_pause") if end_pause is not None else tail_before
    if wanted == before and abs(tail - tail_before) < 0.0005:
        raise TakeCutError("nothing_to_cut", "Those are the pauses the take already has")
    pieces: list[tuple] = []
    cursor = 0.0
    shift = 0.0
    moved = [dict(timed[0])]
    for i, (g, g2) in enumerate(zip(before, wanted)):
        end_i, start_next = timed[i]["end"], timed[i + 1]["start"]
        if g2 <= g:
            pieces.append(("keep", cursor, end_i + g2 / 2))
            cursor = start_next - g2 / 2
        else:
            mid = end_i + g / 2
            pieces.append(("keep", cursor, mid))
            pieces.append(("silence", g2 - g))
            cursor = mid
        shift += g2 - g
        word = dict(timed[i + 1])
        word["start"], word["end"] = round(word["start"] + shift, 3), round(word["end"] + shift, 3)
        moved.append(word)
    last_end = timed[-1]["end"]
    if tail <= tail_before:
        pieces.append(("keep", cursor, last_end + tail))
    else:
        pieces.append(("keep", cursor, total))
        pieces.append(("silence", tail - tail_before))
    pieces = [p for p in pieces if p[0] == "silence" or p[2] - p[1] > 1e-6]
    seconds = sum(p[1] if p[0] == "silence" else p[2] - p[1] for p in pieces)
    return {"pieces": pieces, "words": moved, "pauses_before": before, "pauses_after": wanted,
            "end_before": round(tail_before, 3), "end_after": round(tail, 3), "seconds": round(seconds, 3)}


def splice(pcm: bytes, sample_rate: int, channels: int, pieces: list[tuple]) -> bytes:
    """Apply a plan to interleaved 16-bit PCM; a short fade where two pieces meet."""
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - len(pcm) % (2 * channels)])
    frames = len(samples) // channels
    fade = max(1, int(round(FADE_SECONDS * sample_rate)))
    out = array.array("h")
    for n, piece in enumerate(pieces):
        if piece[0] == "silence":
            out.extend(array.array("h", bytes(2 * channels * int(round(piece[1] * sample_rate)))))
            continue
        a = max(0, min(frames, int(round(piece[1] * sample_rate))))
        b = max(a, min(frames, int(round(piece[2] * sample_rate))))
        chunk = samples[a * channels:b * channels]
        count = b - a
        # Fade in where a cut piece starts after another, fade out where another follows.
        if n > 0:
            for k in range(min(fade, count)):
                for c in range(channels):
                    chunk[k * channels + c] = int(chunk[k * channels + c] * k / fade)
        if n < len(pieces) - 1:
            for k in range(min(fade, count)):
                at = (count - 1 - k) * channels
                for c in range(channels):
                    chunk[at + c] = int(chunk[at + c] * k / fade)
        out.extend(chunk)
    return out.tobytes()


__all__ = ["FADE_SECONDS", "MAX_PAUSE_SECONDS", "TakeCutError", "clean_words", "pauses_of", "plan", "splice"]
