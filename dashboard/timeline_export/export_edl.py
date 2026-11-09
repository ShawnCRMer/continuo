"""CMX3600 EDL fallback for V1 (TIMELINE-INPUT.md 6.2; archived interchange skill section 3).

One video event per storyline clip, ``BL`` (black) events for gaps, cuts only.
Source timecode is the source's own frames at the sequence timecode base, as
our tested RC4 EDL does; record timecode starts at ``start_tc_frame``.  Each
event carries ``* FROM CLIP NAME`` and ``* SOURCE FILE`` so Resolve can relink.
Reels are 8 characters, unique per source (FilmCraft ``edl.rs`` ``reel()``).

Not representable and therefore omitted, named in the README: V2 overlays,
music, clip audio, stills as moving media, mixed source rates (a 48 fps source
is listed with source timecode at the timeline base, which Resolve relinks by
file name; the FCPXML carries the exact rational source rate).

``studio.timeline/1.1``: a retimed clip gets an ``M2`` line (reel, speed in frames per second at
the timeline base, negative when reversed, 0 for a freeze, and the event's source in); its source
out stays in + record length (CMX: the M2 line gives the real speed). A speed curve (lane
w7-export) that is one speed over the clip is that M2 exactly (a hold: 0.0); a ramp is its average
speed, and the package warns that EDL cannot carry it. A V1 dissolve is the CMX
pair: the outgoing clip ends where the transition starts, a zero-length cut on the outgoing clip at
that point, then a ``D`` event of the incoming clip with the transition length, its source in moved
back into its head handle. A dip is two dissolves through ``BL`` (white: ``BL`` with a
``* DIP TO WHITE`` comment, CMX has no white).
"""
from __future__ import annotations

import re
from fractions import Fraction
from pathlib import Path

from .model import Plan
from .timebase import timecode as _timecode, timecode_base, timecode_frames

_REEL_RE = re.compile(r"[^A-Z0-9]")


def _reel(stem: str, used: dict[str, str], ref: str) -> str:
    base = _REEL_RE.sub("", stem.upper())[:8] or "AX"
    reel, n = base, 1
    while used.get(reel, ref) != ref:
        n += 1
        suffix = str(n)
        reel = base[: 8 - len(suffix)] + suffix
    used[reel] = ref
    return reel


def _src_base(clip, offset: int, rate: Fraction) -> int:
    """The source frame shown ``offset`` record frames into a clip, counted at the timeline base."""
    if clip.is_still:
        return offset
    return int(Fraction(clip.source_frame_at(offset, rate)) / (clip.source_rate / rate))


def _speed(clip, rate: Fraction) -> Fraction:
    """The clip's one speed: a speed curve (lane w7-export) that is one speed over the clip is that
    speed exactly (0 = a hold); a ramp is its exact average (CMX M2 has one speed per event)."""
    if clip.retime.curve:
        from .export_curve import average_speed, constant_speed
        speed = constant_speed(clip, rate)
        return average_speed(clip, rate) if speed is None else speed
    return Fraction(0) if clip.retime.freeze else clip.retime.speed


def fcm(plan: Plan) -> str:
    """CMX 3600 frame code mode: DROP FRAME exactly when the plan's timecode is drop frame."""
    return "FCM: DROP FRAME" if plan.drop_frame else "FCM: NON-DROP FRAME"


def _bound(plan: Plan):
    """``timecode(frame, rate)`` in the plan's frame code mode (``;`` before the frames in DF)."""
    return lambda frame, rate: _timecode(frame, rate, plan.drop_frame)


def _m2(reel: str, clip, src_in: int, rate: Fraction, drop: bool = False) -> str:
    base = Fraction(timecode_base(rate))
    speed = _speed(clip, rate) * clip.source_rate / rate * base
    value = float(-speed if clip.retime.reverse else speed)
    return f"M2   {reel:<8}{value:>9.1f}                {_timecode(src_in, rate, drop)}"


def build_edl(plan: Plan, remuxes: dict[str, Path], *, title: str, track: str = "V1") -> str:
    """One CMX3600 EDL per video track (CMX3600 carries one picture track): ``V1`` = the storyline
    with BL events for gaps and the ``* LOC:`` markers, ``V2`` = the overlays at their record times."""
    rate = plan.rate
    timecode = _bound(plan)
    if track == "V1" and (plan.transitions or any(c.retime is not None for c in plan.storyline)):
        return _build_v1_pro(plan, remuxes, title=title)
    lines = [f"TITLE: {title[:70]}", fcm(plan), ""]
    used: dict[str, str] = {}
    number = 0
    for clip in (plan.storyline if track == "V1" else plan.overlays):
        number += 1
        rec_in = plan.start_tc_frame + clip.record_start
        rec_out = rec_in + clip.duration
        if clip.kind == "gap" or clip.info is None:
            lines.append(f"{number:03d}  BL       V     C        {timecode(0, rate)} {timecode(clip.duration, rate)} {timecode(rec_in, rate)} {timecode(rec_out, rate)}")
            lines.append("")
            continue
        path = remuxes.get(clip.source["ref"], clip.info.path)
        reel = _reel(clip.info.path.stem, used, clip.source["ref"])
        if clip.is_still:
            src_in, src_out = 0, clip.duration
        else:
            # Source frames counted at the timeline base: exact when source rate == timeline rate.
            ratio = clip.source_rate / rate
            src_in = int(Fraction(clip.source_in) / ratio)
            src_out = src_in + clip.duration
        lines.append(f"{number:03d}  {reel:<8} V     C        {timecode(src_in, rate)} {timecode(src_out, rate)} {timecode(rec_in, rate)} {timecode(rec_out, rate)}")
        if clip.retime is not None and not clip.is_still:
            lines.append(_m2(reel, clip, src_in, rate, plan.drop_frame))
        lines.append(f"* FROM CLIP NAME: {path.name}")
        lines.append(f"* SOURCE FILE: {path}")
        if clip.label:
            lines.append(f"* COMMENT: {clip.label[:120]}")
        if not clip.enabled:
            lines.append("* COMMENT: disabled in the timeline")
        lines.append("")
    for marker in plan.markers if track == "V1" else []:
        if marker.frame < plan.total_frames:
            name = (marker.name or marker.id).replace("\n", " ")[:120]
            lines.append(f"* LOC: {timecode(plan.start_tc_frame + marker.frame, rate)} BLUE    {name}")
    return "\r\n".join(lines).rstrip("\r\n") + "\r\n"


def _build_v1_pro(plan: Plan, remuxes: dict[str, Path], *, title: str) -> str:
    """V1 with retimed clips (M2) and transitions (D events); record frames exact.

    A clip with an incoming transition starts its event at the transition start (its source in
    moved back into its head) and the outgoing clip gets the zero-length cut at that frame under
    the same event number; a clip with an outgoing transition ends its event at that transition's
    start (the D event carries the overlap)."""
    rate = plan.rate
    timecode = _bound(plan)
    lines = [f"TITLE: {title[:70]}", fcm(plan), ""]
    used: dict[str, str] = {}
    number = [0]
    tc0 = plan.start_tc_frame
    by_left = {t.left: t for t in plan.transitions}
    by_right = {t.right: t for t in plan.transitions}
    clips = {c.id: c for c in plan.storyline}

    def reel_of(clip) -> str:
        return "BL" if clip is None or clip.kind == "gap" or clip.info is None else _reel(clip.info.path.stem, used, clip.source["ref"])

    def line(clip, kind: str, first: int, length: int, rec_in: int, *, transition: int = 0, same: bool = False) -> None:
        if not same:
            number[0] += 1
        reel = reel_of(clip)
        src_in = 0 if reel == "BL" else _src_base(clip, first, rate)
        what = f"D    {transition:03d}" if kind == "D" else "C       "
        lines.append(f"{number[0]:03d}  {reel:<8} V     {what} {timecode(src_in, rate)} {timecode(src_in + length, rate)} "
                     f"{timecode(tc0 + rec_in, rate)} {timecode(tc0 + rec_in + length, rate)}")
        if reel != "BL" and clip.retime is not None and not clip.is_still:
            lines.append(_m2(reel, clip, src_in, rate, plan.drop_frame))
        if reel != "BL" and length:
            path = remuxes.get(clip.source["ref"], clip.info.path)
            lines.append(f"* FROM CLIP NAME: {path.name}")
            lines.append(f"* SOURCE FILE: {path}")
            if clip.label:
                lines.append(f"* COMMENT: {clip.label[:120]}")
            if not clip.enabled:
                lines.append("* COMMENT: disabled in the timeline")

    for clip in plan.storyline:
        out = by_left.get(clip.id)
        end = clip.duration - (out.before if out is not None else 0)  # offset where this clip's event ends
        incoming = by_right.get(clip.id)
        if incoming is None:
            line(clip, "C", 0, end, clip.record_start)
            lines.append("")
            continue
        left = clips[incoming.left]
        first = -incoming.before
        if incoming.kind == "dissolve":
            line(left, "C", left.duration - incoming.before, 0, incoming.start)
            line(clip, "D", first, end - first, incoming.start, transition=incoming.duration, same=True)
            lines.append(f"* TRANSITION: cross dissolve {incoming.duration} f, {incoming.before} before / {incoming.after} after the cut")
        else:
            half = incoming.duration // 2
            line(left, "C", left.duration - incoming.before, 0, incoming.start)
            line(None, "D", 0, half, incoming.start, transition=half, same=True)
            colour = "white" if incoming.kind == "dip_white" else "black"
            lines.append(f"* TRANSITION: dip to {colour} {incoming.duration} f" + (" (CMX has no white: BL stands for it)" if colour == "white" else ""))
            lines.append("")
            line(None, "C", 0, 0, incoming.start + half)
            line(clip, "D", first + half, end - first - half, incoming.start + half, transition=incoming.duration - half, same=True)
        lines.append("")
    for marker in plan.markers:
        if marker.frame < plan.total_frames:
            name = (marker.name or marker.id).replace("\n", " ")[:120]
            lines.append(f"* LOC: {timecode(plan.start_tc_frame + marker.frame, rate)} BLUE    {name}")
    return "\r\n".join(lines).rstrip("\r\n") + "\r\n"


def verify_edl(text: str, plan: Plan) -> dict:
    """Read the V1 EDL back: every record frame of the storyline covered once in order (cuts and the
    D events' overlaps), the D events' lengths equal the transitions, an M2 line per retimed event."""
    failures: list[str] = []
    rate = plan.rate
    events = []
    m2 = 0
    for raw in text.splitlines():
        parts = raw.split()
        if parts and parts[0] == "M2":
            m2 += 1
            continue
        if len(parts) >= 8 and parts[0].isdigit() and parts[2] == "V":
            kind = parts[3]
            length = int(parts[4]) if kind == "D" else 0
            tcs = parts[-4:]
            events.append({"num": parts[0], "reel": parts[1], "kind": kind, "length": length,
                           "rec_in": _tc_frames(tcs[2], rate), "rec_out": _tc_frames(tcs[3], rate),
                           "src_in": _tc_frames(tcs[0], rate), "src_out": _tc_frames(tcs[1], rate)})
    cursor = plan.start_tc_frame
    dissolves = 0
    for e in events:
        if e["rec_out"] - e["rec_in"] != e["src_out"] - e["src_in"]:
            failures.append(f"event {e['num']} record and source lengths differ")
        if e["rec_in"] != cursor:
            failures.append(f"event {e['num']} starts at {e['rec_in'] - plan.start_tc_frame}, expected {cursor - plan.start_tc_frame}")
        cursor = e["rec_out"]
        if e["kind"] == "D":
            dissolves += 1
    if cursor != plan.start_tc_frame + plan.total_frames:
        failures.append(f"the events end at {cursor - plan.start_tc_frame}, the storyline at {plan.total_frames}")
    want_d = sum(1 if t.kind == "dissolve" else 2 for t in plan.transitions)
    if dissolves != want_d:
        failures.append(f"{dissolves} D events, expected {want_d}")
    want_m2 = sum(1 for c in plan.storyline if c.kind == "clip" and c.retime is not None and not c.is_still)
    if m2 < want_m2:
        failures.append(f"{m2} M2 lines, expected at least {want_m2}")
    return {"ok": not failures, "failures": failures, "events": len(events), "dissolve_events": dissolves, "m2": m2}


def _tc_frames(text: str, rate: Fraction) -> int:
    return timecode_frames(text, rate)


def parse_edl_events(text: str) -> list[dict[str, str]]:
    events = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 8 and parts[0].isdigit() and parts[2] == "V":
            events.append({"num": parts[0], "reel": parts[1], "src_in": parts[4], "src_out": parts[5], "rec_in": parts[6], "rec_out": parts[7]})
    return events


def build_marker_edl(plan: Plan, *, title: str) -> str:
    """Timeline markers in the EDL form Resolve itself writes with Timelines > Export > Timeline
    Markers to EDL and reads with Timelines > Import > Timeline Markers from EDL (Resolve manual,
    "Exporting Timeline Markers to EDL"; sample event ``001  001  V  C  00:00:00:00 00:00:00:01
    00:00:00:00 00:00:00:01 |C:ResolveColorBlue |M:Intro |D:1``). FCPXML markers do not reach Resolve
    (EXPORT-PROOF-RESOLVE.md: FCPXML 0/2), so this file carries them for a Resolve import."""
    rate = plan.rate
    timecode = _bound(plan)
    lines = [f"TITLE: {title[:70]}", fcm(plan), ""]
    number = 0
    for marker in plan.markers:
        if marker.frame >= plan.total_frames:
            continue
        number += 1
        at = plan.start_tc_frame + marker.frame
        length = max(1, marker.duration)
        name = (marker.name or marker.id).replace("\n", " ").replace("|", "/")[:120]
        lines.append(f"{number:03d}  001      V     C        {timecode(at, rate)} {timecode(at + 1, rate)} {timecode(at, rate)} {timecode(at + 1, rate)}  ")
        lines.append(f" |C:ResolveColorBlue |M:{name} |D:{length}")
        lines.append("")
    return "\r\n".join(lines).rstrip("\r\n") + "\r\n"
