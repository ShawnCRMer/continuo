"""Rational time: frame <-> seconds, FCPXML time strings, SMPTE timecode.

All conversions of the export live here (TIMELINE-INPUT.md section 2.2).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any

from .errors import ExportError


def rate_of(value: Any, where: str) -> Fraction:
    """``{num, den}`` (schema form) -> Fraction frames per second."""
    if isinstance(value, Fraction):
        rate = value
    elif isinstance(value, dict):
        num, den = value.get("num"), value.get("den", 1)
        if type(num) is not int or type(den) is not int or num <= 0 or den <= 0:
            raise ExportError("invalid_timeline", f"{where} must be a rational rate with positive whole num and den.")
        rate = Fraction(num, den)
    elif type(value) is int and value > 0:
        rate = Fraction(value)
    else:
        raise ExportError("invalid_timeline", f"{where} must be a rational rate {{num, den}}.")
    if rate <= 0 or rate > 1000:
        raise ExportError("invalid_timeline", f"{where} is outside 0-1000 frames per second.")
    return rate


def seconds(frames: int | Fraction, rate: Fraction) -> Fraction:
    return Fraction(frames) / rate


def fcpx_time(frames: int, rate: Fraction) -> str:
    """Frames at ``rate`` as FCPXML rational seconds: ``N*den/num s``, never a decimal."""
    if frames == 0:
        return "0s"
    numerator = frames * rate.denominator
    denominator = rate.numerator
    if denominator == 1:
        return f"{numerator}s"
    return f"{numerator}/{denominator}s"


def fcpx_seconds(value: Fraction) -> str:
    """Any exact rational seconds value as an FCPXML time string."""
    value = Fraction(value)
    if value == 0:
        return "0s"
    if value.denominator == 1:
        return f"{value.numerator}s"
    return f"{value.numerator}/{value.denominator}s"


def parse_fcpx_time(text: str) -> Fraction:
    raw = (text or "").strip()
    if not raw.endswith("s"):
        raise ValueError(f"not an FCPXML time: {text!r}")
    raw = raw[:-1]
    if "/" in raw:
        a, b = raw.split("/", 1)
        return Fraction(int(a), int(b))
    return Fraction(raw)


def ffmpeg_seconds(value: Fraction) -> str:
    """Seconds for an ffmpeg option; nine decimals keep sub-sample precision at 48 kHz."""
    return f"{float(value):.9f}"


def ffmpeg_rate(rate: Fraction) -> str:
    return f"{rate.numerator}/{rate.denominator}"


def timecode_base(rate: Fraction) -> int:
    """Nominal frames per timecode second (24 for 24 and 23.976, 30 for 29.97)."""
    return -(-rate.numerator // rate.denominator)


def is_drop_rate(rate: Fraction) -> bool:
    """SMPTE 12M drop frame exists only for 30000/1001 (29.97) and 60000/1001 (59.94)."""
    return rate in (Fraction(30000, 1001), Fraction(60000, 1001))


def _dropped(rate: Fraction) -> int:
    """Frame numbers skipped per minute in drop frame: 2 at 29.97, 4 at 59.94."""
    return round(timecode_base(rate) / 15)


def timecode(frame: int, rate: Fraction, drop: bool = False) -> str:
    """SMPTE timecode: non-drop ``HH:MM:SS:FF``; with ``drop`` at 29.97 / 59.94 the drop-frame
    ``HH:MM:SS;FF`` (frame numbers 0..d-1 skipped at every minute except each tenth, SMPTE 12M)."""
    if frame < 0:
        raise ExportError("invalid_timeline", "A timecode cannot be negative.")
    base = timecode_base(rate)
    sep = ":"
    if drop and is_drop_rate(rate):
        d = _dropped(rate)
        per_minute = base * 60 - d
        per_ten = per_minute * 10 + d
        tens, rest = divmod(frame, per_ten)
        frame = frame + 9 * d * tens + (d * ((rest - d) // per_minute) if rest >= d else 0)
        sep = ";"
    ff = frame % base
    total_seconds = frame // base
    return f"{total_seconds // 3600 % 24:02d}:{total_seconds // 60 % 60:02d}:{total_seconds % 60:02d}{sep}{ff:02d}"


def timecode_frames(text: str, rate: Fraction) -> int:
    """Read ``HH:MM:SS:FF`` (non-drop) or ``HH:MM:SS;FF`` / ``HH;MM;SS;FF`` (drop frame) back to frames."""
    raw = (text or "").strip()
    drop = ";" in raw
    hh, mm, ss, ff = (int(x) for x in raw.replace(";", ":").split(":"))
    base = timecode_base(rate)
    frames = ((hh * 60 + mm) * 60 + ss) * base + ff
    if drop and is_drop_rate(rate):
        minutes = hh * 60 + mm
        frames -= _dropped(rate) * (minutes - minutes // 10)
    return frames
