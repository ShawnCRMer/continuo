"""Speed curves (lane w6-tempo) for the interchange writers (lane w7-export): pure, exact.

``timeline/speed_curve.py`` is the one mapping (read only): record offset ``x`` -> source position
``P(x)``. Each writer carries as much of it as its own format allows:

- FCPXML ``timeMap`` and FCP7 ``Time Remap`` graphdict: a list of points ``(x, P(x))`` that makes the
  mapping exact at every record frame. Where ``P`` is a straight line (one speed, a hold, a step
  segment) the segment's two ends suffice; a ramp (``linear`` or ``smooth`` between two different
  non-zero speeds) is quadratic / quartic in ``x``, so it gets a point on every record frame.
- OTIO and EDL have one constant speed per clip: ``constant_speed`` says whether the curve IS one
  speed over the clip (then they carry it exactly), ``average_speed`` is what they write otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .model import PlanClip, speed_curve

# Formats that carry any curve exactly vs. only one constant speed per clip.
RAMP_FORMATS = ("FCPXML", "FCP7 XML")
CONSTANT_ONLY_FORMATS = ("OTIO", "EDL")


@dataclass(frozen=True)
class Point:
    """``x``: record offset in the clip (frames); ``position``: source frames passed (``P(x)``);
    ``ease``: shape of the stretch from this point to the next (``linear`` | ``smooth``).

    ``step`` needs no extra point: the SPEED steps at the next key, the position (its integral)
    does not jump, so a step segment is a straight line with a corner at the next key's point."""

    x: Fraction
    position: Fraction
    ease: str = "linear"


def q_of(clip: PlanClip, rate: Fraction) -> Fraction:
    return clip.source_rate / rate


def _ramps(a: Fraction, b: Fraction, ease: str) -> bool:
    """A segment whose position is not a straight line (two different non-zero speeds, eased)."""
    return a != b and a != 0 and b != 0 and ease != "step"


def segments(keys: list[dict[str, Any]], duration: int) -> list[tuple[int, int, Fraction, Fraction, str, bool]]:
    """``(start, end, a, b, ease, ramp)`` for every stretch of the clip ``0 .. duration``."""
    out = []
    for index, key in enumerate(keys):
        start = key["frame"]
        if start >= duration:
            break
        last = index + 1 >= len(keys)
        end = duration if last else min(keys[index + 1]["frame"], duration)
        a = Fraction(key["percent"])
        b = a if last else Fraction(keys[index + 1]["percent"])
        ease = "linear" if last else key.get("ease", "linear")
        out.append((start, end, a, b, ease, not last and _ramps(a, b, ease)))
    return out


def points(clip: PlanClip, rate: Fraction) -> list[Point]:
    """Points ``(x, P(x))`` over ``0 .. duration`` that make the mapping exact at every record frame."""
    sc = speed_curve()
    keys, q = clip.keys(), q_of(clip, rate)
    out: list[Point] = []
    for start, end, _a, _b, ease, ramp in segments(keys, clip.duration):
        xs = range(start, end) if ramp else [start]
        shape = "smooth" if ramp and ease == "smooth" else "linear"
        out.extend(Point(Fraction(x), sc.position(keys, q, x), shape) for x in xs)
    out.append(Point(Fraction(clip.duration), sc.position(keys, q, clip.duration)))
    return out


def edge_rates(clip: PlanClip, rate: Fraction) -> tuple[Fraction, Fraction]:
    """Source frames per record frame at the clip's first and last frame (the handles run at these)."""
    sc = speed_curve()
    keys, q = clip.keys(), q_of(clip, rate)
    return Fraction(keys[0]["percent"]) * q / 100, sc.speed_at(keys, clip.duration) * q / 100


def constant_speed(clip: PlanClip, rate: Fraction) -> Fraction | None:
    """The one speed (ratio, 0 = a hold) when the curve is a single speed over the whole clip, else None."""
    parts = segments(clip.keys(), clip.duration)
    if any(ramp for *_rest, ramp in parts):
        return None
    speeds = {a for _s, _e, a, _b, _ease, _ramp in parts}
    return Fraction(speeds.pop(), 100) if len(speeds) == 1 else None


def average_speed(clip: PlanClip, rate: Fraction) -> Fraction:
    """Exact average speed of the clip (source frames read / (q * duration)): what a one-speed format writes."""
    sc = speed_curve()
    q = q_of(clip, rate)
    return sc.position(clip.keys(), q, clip.duration) / (q * clip.duration)


def is_curved(clip: PlanClip) -> bool:
    return clip.kind == "clip" and clip.retime is not None and bool(clip.retime.curve) and not clip.is_still


def ramped(clip: PlanClip, rate: Fraction) -> bool:
    """A curved clip a one-speed format (OTIO, EDL) cannot carry exactly."""
    return is_curved(clip) and constant_speed(clip, rate) is None
