"""Speed curves (lane w6-tempo): the ONE mapping from a clip's record frames to its source frames.

Pure (fractions only): the timeline algebra (schema / ops_pro), the export (timeline_export/model,
render_film) and the agent read it; the JS twin (desktop/film-suite/src/timeline/ops/timeline-speed.js)
is held equal by the shared fixtures (tests/backend/fixtures/timeline/speed.json).

A picture clip's ``retime.curve`` is a list of keys ``{frame, percent, ease}``: ``frame`` is a record
frame inside the clip (0 = its first frame, strictly rising, the first key at 0), ``percent`` the
speed there (0 = hold, else 1 .. 10 000), ``ease`` the shape of the segment that starts at the key
(``linear``, ``smooth`` = smoothstep, ``step`` = the key's speed until the next key). A hold never
ramps: a 0 % key holds until the next key and the segment into a 0 % key keeps its own speed (the
picture cuts into the hold; a slow stop ramps to 1 %). After the last key the speed stays at the
last key's. A
constant speed is the curve with one key, stored as before (``retime.speed``; no ``curve``), so old
documents read unchanged. A hold is a key pair at 0 %.

Source position (source frames after ``source_in_frame``) at record offset ``x`` is the integral of
the speed: ``P(x) = sum over segments of integral(speed(u) du) * q`` with ``q = source_rate / rate``
(source frames per record frame at 100 %). Record frame ``k`` shows source frame ``floor(P(k))``
(forward) or ``last - floor(P(k))`` (reverse); the clip reads ``max(1, ceil(P(duration)))`` source
frames and its duration is the longest ``d`` with ``P(d) <= source frames``. Every value is exact.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Iterable, Sequence

EASES = ("linear", "smooth", "step")
MAX_KEYS = 200
MAX_PERCENT = 10_000


def keys_of(item: dict[str, Any]) -> list[dict[str, Any]] | None:
    """The clip's speed keys, or None for a clip without a curve."""
    retime = item.get("retime")
    if not retime:
        return None
    curve = retime.get("curve")
    return curve if curve else None


def q_of(source_rate: dict[str, int] | Fraction, rate: dict[str, int] | Fraction) -> Fraction:
    """Source frames per record frame at 100 %."""
    sr = source_rate if isinstance(source_rate, Fraction) else Fraction(source_rate["num"], source_rate["den"])
    r = rate if isinstance(rate, Fraction) else Fraction(rate["num"], rate["den"])
    return sr / r


def _segment_integral(a: Fraction, b: Fraction, length: int, ease: str, x: Fraction) -> Fraction:
    """Integral of the speed (in percent) over the first ``x`` frames of a segment of ``length``
    frames going from ``a`` to ``b`` percent."""
    if a == b or ease == "step" or a == 0 or b == 0:
        return a * x
    t = x / length
    if ease == "smooth":  # speed = a + (b - a) * (3 t^2 - 2 t^3); integral of S = L (t^3 - t^4 / 2)
        return a * x + (b - a) * length * (t ** 3 - t ** 4 / 2)
    return a * x + (b - a) * x * t / 2


def position(keys: Sequence[dict[str, Any]], q: Fraction, x: int | Fraction) -> Fraction:
    """Source frames passed after ``x`` record frames (exact)."""
    x = Fraction(x)
    total = Fraction(0)
    for index, key in enumerate(keys):
        start = key["frame"]
        if x <= start:
            break
        a = Fraction(key["percent"])
        if index + 1 < len(keys):
            nxt = keys[index + 1]
            length = nxt["frame"] - start
            span = min(x, Fraction(nxt["frame"])) - start
            total += _segment_integral(a, Fraction(nxt["percent"]), length, key.get("ease", "linear"), span)
        else:
            total += a * (x - start)
    return total * q / 100


def speed_at(keys: Sequence[dict[str, Any]], x: int | Fraction) -> Fraction:
    """Speed in percent at record offset ``x`` (the curve's value, exact)."""
    x = Fraction(x)
    for index, key in enumerate(keys):
        if index + 1 < len(keys) and x >= keys[index + 1]["frame"]:
            continue
        a = Fraction(key["percent"])
        if index + 1 >= len(keys) or x <= key["frame"]:
            return a
        nxt = keys[index + 1]
        t = (x - key["frame"]) / (nxt["frame"] - key["frame"])
        if key.get("ease", "linear") == "step" or a == 0 or nxt["percent"] == 0:
            return a
        if key.get("ease", "linear") == "smooth":
            t = 3 * t ** 2 - 2 * t ** 3
        return a + (Fraction(nxt["percent"]) - a) * t
    return Fraction(keys[-1]["percent"])


def _floor(value: Fraction) -> int:
    return value.numerator // value.denominator


def _ceil(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def source_frames(keys: Sequence[dict[str, Any]], q: Fraction, duration: int) -> int:
    """Source frames a clip of ``duration`` record frames reads (a whole hold reads one)."""
    return max(1, _ceil(position(keys, q, duration)))


def duration_for(keys: Sequence[dict[str, Any]], q: Fraction, frames: int) -> int:
    """The longest record duration whose source range fits ``frames`` source frames (>= 1)."""
    last = keys[-1]
    if last["percent"] <= 0:
        raise ValueError("the last key holds: the clip would never end")
    tail_rate = Fraction(last["percent"]) * q / 100
    lo, hi = 1, last["frame"] + max(2, _ceil((frames - position(keys, q, last["frame"])) / tail_rate) + 2)
    if position(keys, q, lo) > frames:
        return 1
    while lo < hi:  # largest d with P(d) <= frames
        mid = (lo + hi + 1) // 2
        if position(keys, q, mid) <= frames:
            lo = mid
        else:
            hi = mid - 1
    return lo


def offsets(keys: Sequence[dict[str, Any]], q: Fraction, first: int, count: int) -> list[int]:
    """``floor(P(k))`` for record offsets ``first .. first + count - 1`` (source offsets, forward)."""
    return [_floor(position(keys, q, k)) for k in range(first, first + count)]


def constant_keys(speed: dict[str, int] | Fraction) -> list[dict[str, Any]]:
    """A constant speed as the one-key curve (percent may be a fraction here: never stored)."""
    s = speed if isinstance(speed, Fraction) else Fraction(speed["num"], speed["den"])
    return [{"frame": 0, "percent": s * 100, "ease": "linear"}]


def average_speed(keys: Sequence[dict[str, Any]], q: Fraction, duration: int) -> dict[str, int]:
    """The clip's average speed as whole percent (>= 1 %), what readers without curves see."""
    avg = position(keys, q, duration) / (q * duration) * 100
    pct = max(1, min(MAX_PERCENT, _floor(avg + Fraction(1, 2))))
    from math import gcd
    k = gcd(pct, 100)
    return {"num": pct // k, "den": 100 // k}


def check_keys(keys: Any, where: str) -> str | None:
    """Shape of a key list: None when valid, else the reason (``where`` prefixes it)."""
    if type(keys) is not list or not 2 <= len(keys) <= MAX_KEYS:
        return f"{where} must be a list of 2..{MAX_KEYS} keys (one speed is retime.speed)"
    previous = -1
    for index, key in enumerate(keys):
        path = f"{where}[{index}]"
        if type(key) is not dict or set(key) != {"frame", "percent", "ease"}:
            return f"{path} must be {{frame, percent, ease}}"
        frame, percent = key["frame"], key["percent"]
        if type(frame) is not int or frame <= previous or frame < 0:
            return f"{path}.frame must be a whole frame after the previous key"
        if index == 0 and frame != 0:
            return f"{path}.frame: the first key sits on the clip's first frame (0)"
        if type(percent) is not int or not (percent == 0 or 1 <= percent <= MAX_PERCENT):
            return f"{path}.percent must be 0 (hold) or 1..{MAX_PERCENT}"
        if key["ease"] not in EASES:
            return f"{path}.ease must be " + " | ".join(EASES)
        previous = frame
    if keys[-1]["percent"] == 0:
        return f"{where}: the last key holds; a hold ends with a key that moves on"
    return None


def source_map(item: dict[str, Any], rate: dict[str, int], first: int = 0, count: int | None = None) -> list[int]:
    """The absolute source frame each record frame ``first ..`` of a moving picture clip shows
    (speed, curve, reverse, freeze): the one mapping every reader uses."""
    duration = item["duration_frames"]
    count = duration - first if count is None else count
    retime = item.get("retime")
    source_in = item["source_in_frame"]
    if retime is None:
        keys, q = [{"frame": 0, "percent": 100, "ease": "linear"}], q_of(item["source_rate"], rate)
    elif retime["speed"]["num"] == 0 and not retime.get("curve"):
        return [source_in] * count
    else:
        keys = keys_of(item) or constant_keys(retime["speed"])
        q = q_of(item["source_rate"], rate)
    steps = offsets(keys, q, first, count)
    if retime is not None and retime.get("reverse"):
        last = source_in + source_frames(keys, q, duration) - 1
        return [last - s for s in steps]
    return [source_in + s for s in steps]


def key_rows(item: dict[str, Any], rate: dict[str, int]) -> list[dict[str, Any]]:
    """Each key with the source frame it lands on (for op results, the UI and the agent)."""
    keys = keys_of(item)
    if not keys:
        return []
    q = q_of(item["source_rate"], rate)
    shown = source_map(item, rate)
    return [{**key, "source_frame": shown[min(key["frame"], len(shown) - 1)],
             "source_position": str(position(keys, q, key["frame"]))} for key in keys]


def normalise(keys: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"frame": k["frame"], "percent": k["percent"], "ease": k.get("ease", "linear")} for k in keys]
