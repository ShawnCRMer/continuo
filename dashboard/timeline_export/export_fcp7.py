"""Final Cut Pro 7 XML (``xmeml`` version 5) for Premiere Pro; DaVinci Resolve reads it too.

Ported from FilmCraft's ``crates/interchange/src/fcp7.rs`` (MIT OR Apache-2.0,
https://github.com/storytold/filmcraft): ``export``/``write_sequence``/
``write_clip``/``write_file``/``write_rate``/``write_markers``.  Reduced to what
our timeline holds (hard cuts, no effects or transitions) and rewritten in Python.

Mapping:
- V1 -> video track 1: one ``clipitem`` per storyline clip at its record frames;
  gaps are empty track time (FCP7 has no gap element); disabled clips keep
  ``enabled FALSE``.
- V2 -> video track 2 at its record frames.
- music -> audio track 1, clip audio set to on -> audio track 2 (from the
  original file; the picture uses the video-only remux).
- sound tracks A1-A3 -> one audio track each after those, ``MZ.TrackName`` "A<n> <role>", gain
  and fades as an Audio Levels filter (linear level, keyframes at the fade edges);
- ``start/end`` are sequence frames from 0; the start timecode is the
  sequence ``timecode`` (``start_tc_frame``).
- ``in/out`` and the clipitem ``duration`` are in sequence frames (like
  FilmCraft: ``frames_round(seq_rate, source_time)``); ``pproTicksIn/Out``
  carry the exact source time at 254 016 000 000 ticks per second, so a 48 fps
  source keeps its exact source frame in Premiere.
- every ``file`` carries its own rate and its full ``duration`` in its frames,
  so all handles stay available; files are written once and referenced by id.

``studio.timeline/1.1`` (nothing of it is written for a cut that does not use it):
- a retimed clip carries the FCP7 ``Time Remap`` filter (``effectid timeremap``: ``speed`` in
  percent, ``reverse``, ``frameblending`` for blend / flow; a freeze is variable speed with a
  ``graphdict`` that holds one frame; a speed curve / hold is variable speed with a ``graphdict``
  exact at every record frame, ``curve_graph``); its ``in``/``out`` span the media it reads
  (``out - in = round(length * speed)``, a freeze one frame) while ``start``/``end`` stay the
  record frames;
- a V1 transition is a ``transitionitem`` on video track 1 between the two clipitems
  (``start``/``end`` = the transition's record frames, ``alignment`` center | start-black |
  end-black for centre / start / end, ``Cross Dissolve`` or ``Dip to Color Dissolve`` with its
  colour); the clipitems keep their cut points and the handles stay in their files.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from urllib.parse import quote

from .errors import ExportError
from .model import Plan, PlanClip, speed_curve
from .timebase import timecode

XMEML_VERSION = "5"
PPRO_TICKS_PER_SECOND = 254_016_000_000


def timebase(rate: Fraction) -> tuple[int, bool]:
    """FCP7 rate: integer timebase plus NTSC flag (FilmCraft ``timebase``)."""
    if rate.denominator == 1:
        return rate.numerator, False
    if rate.denominator == 1001 and rate.numerator % 1000 == 0:
        return rate.numerator // 1000, True
    raise ExportError("invalid_timeline", f"FCP7 XML cannot express a rate of {rate.numerator}/{rate.denominator} fps.")


def path_url(path: Path) -> str:
    """Premiere's form: ``file://localhost/C:/a%20b/c.mp4``."""
    text = str(Path(path).resolve()).replace("\\", "/")
    if text.startswith("//?/"):
        text = text[4:]
    if text.startswith("//"):
        return "file:" + quote(text, safe="/:")
    if not text.startswith("/"):
        text = "/" + text
    return "file://localhost" + quote(text, safe="/:")


def _floor(value: Fraction) -> int:
    return value.numerator // value.denominator


def _round(value: Fraction) -> int:
    return _floor(value + Fraction(1, 2))


@dataclass
class Fcp7:
    text: str
    expected_storyline_clips: int
    expected_music_clips: int
    expected_sound_clips: int | None = None  # A1-A3 clips; None without sound tracks
    expected_transitions: int = 0
    expected_retimed: int = 0
    expected_nested: int = 0  # lane w23-bins: clipitems on V1 holding a nested sequence (a nested cut)


def _text(parent: ET.Element, tag: str, value: object) -> ET.Element:
    node = ET.SubElement(parent, tag)
    node.text = str(value)
    return node


def _bool(value: bool) -> str:
    return "TRUE" if value else "FALSE"


def _rate(parent: ET.Element, rate: Fraction) -> None:
    tb, ntsc = timebase(rate)
    node = ET.SubElement(parent, "rate")
    _text(node, "timebase", tb)
    _text(node, "ntsc", _bool(ntsc))


def _timecode(parent: ET.Element, frame: int, rate: Fraction, drop: bool = False) -> None:
    node = ET.SubElement(parent, "timecode")
    _rate(node, rate)
    _text(node, "string", timecode(frame, rate, drop))
    _text(node, "frame", frame)
    _text(node, "displayformat", "DF" if drop else "NDF")


def build_fcp7(plan: Plan, remuxes: dict[str, Path], *, name: str) -> Fcp7:
    rate = plan.rate
    timebase(rate)  # refuse inexpressible rates before writing anything
    total = plan.total_frames
    files: dict[str, str] = {}
    still_lengths: dict[str, int] = {}
    for clip in [*plan.storyline, *plan.overlays]:
        if clip.kind == "clip" and clip.info is not None and clip.info.is_still:
            ref = clip.source["ref"]
            still_lengths[ref] = max(still_lengths.get(ref, 0), clip.duration)
    counter = {"clip": 0}

    root = ET.Element("xmeml", {"version": XMEML_VERSION})
    sequence = ET.SubElement(root, "sequence", {"id": "sequence-1"})
    _text(sequence, "name", name)
    _text(sequence, "duration", total)
    _rate(sequence, rate)
    _timecode(sequence, plan.start_tc_frame, rate, plan.drop_frame)
    media = ET.SubElement(sequence, "media")

    def file_element(parent: ET.Element, key: str, path: Path, *, f_rate: Fraction, frames: int,
                     width: int | None, height: int | None, audio: tuple[int, int] | None) -> None:
        if key in files:
            ET.SubElement(parent, "file", {"id": files[key]})
            return
        files[key] = f"file-{len(files) + 1}"
        node = ET.SubElement(parent, "file", {"id": files[key]})
        _text(node, "name", path.name)
        _text(node, "pathurl", path_url(path))
        _rate(node, f_rate)
        _text(node, "duration", frames)
        _timecode(node, 0, f_rate)
        node_media = ET.SubElement(node, "media")
        if width is not None:
            video = ET.SubElement(node_media, "video")
            chars = ET.SubElement(video, "samplecharacteristics")
            _rate(chars, f_rate)
            _text(chars, "width", width)
            _text(chars, "height", height)
            _text(chars, "anamorphic", "FALSE")
            _text(chars, "pixelaspectratio", "square")
            _text(chars, "fielddominance", "none")
        if audio is not None:
            node_audio = ET.SubElement(node_media, "audio")
            chars = ET.SubElement(node_audio, "samplecharacteristics")
            _text(chars, "depth", 16)
            _text(chars, "samplerate", audio[0])
            _text(node_audio, "channelcount", audio[1])

    def clipitem(track: ET.Element, label: str, *, enabled: bool, record_start: int, length: int,
                 source_seconds: Fraction, media_seconds: Fraction | None, audio_track: bool = False,
                 media_length: int | None = None) -> ET.Element:
        del audio_track  # kept for call-site readability: audio items get a <sourcetrack> after the file
        counter["clip"] += 1
        item = ET.SubElement(track, "clipitem", {"id": f"clipitem-{counter['clip']}"})
        _text(item, "name", label)
        _text(item, "enabled", _bool(enabled))
        in_f = _round(source_seconds * rate)
        out_f = in_f + (length if media_length is None else media_length)  # derive out from the record span (FilmCraft #340)
        media_frames = out_f if media_seconds is None else max(out_f, _floor(media_seconds * rate))
        _text(item, "duration", media_frames)
        _rate(item, rate)
        _text(item, "start", record_start)
        _text(item, "end", record_start + length)
        _text(item, "in", in_f)
        _text(item, "out", out_f)
        _text(item, "pproTicksIn", _floor(source_seconds * PPRO_TICKS_PER_SECOND))
        _text(item, "pproTicksOut", _floor((source_seconds + Fraction(length if media_length is None else media_length) / rate) * PPRO_TICKS_PER_SECOND))
        return item

    def picture(track: ET.Element, clip: PlanClip, *, enabled: bool, offset: int = 0) -> None:
        if offset:  # lane w23-bins: a piece inside a nested sequence counts its frames from the nest's start
            from dataclasses import replace
            clip = replace(clip, record_start=clip.record_start - offset)
        info = clip.info
        ref = clip.source["ref"]
        if info.is_still:
            length = still_lengths[ref]
            item = clipitem(track, clip.label or info.path.stem, enabled=enabled, record_start=clip.record_start,
                            length=clip.duration, source_seconds=Fraction(0), media_seconds=Fraction(length) / rate, audio_track=False)
            file_element(item, "v:" + ref, info.path, f_rate=rate, frames=length, width=info.width, height=info.height, audio=None)
        else:
            remux = remuxes[ref]
            retime = clip.retime
            media_length = None
            source_in = clip.source_in
            if retime is not None and retime.curve:
                # A speed curve: in/out span the graph's media frames (sequence-rate, like every clipitem).
                values = [v for _w, v in curve_graph(clip, rate)]
                in_f = _round(Fraction(source_in) / clip.source_rate * rate)
                media_length = max(1, -(-max(values).numerator // max(values).denominator) - in_f)
            elif retime is not None:
                # In/out span the media the clip reads (sequence-rate frames like every clipitem).
                media_length = 1 if retime.freeze else max(1, _round(Fraction(clip.duration) * retime.speed))
            item = clipitem(track, clip.label or remux.stem, enabled=enabled, record_start=clip.record_start, length=clip.duration,
                            source_seconds=Fraction(source_in) / clip.source_rate,
                            media_seconds=Fraction(info.frames) / clip.source_rate, audio_track=False, media_length=media_length)
            file_element(item, "v:" + ref, remux, f_rate=clip.source_rate, frames=info.frames, width=info.width, height=info.height, audio=None)
            if retime is not None:
                _time_remap(item, clip, rate)

    video = ET.SubElement(media, "video")
    fmt = ET.SubElement(ET.SubElement(video, "format"), "samplecharacteristics")
    _rate(fmt, rate)
    _text(fmt, "width", plan.width)
    _text(fmt, "height", plan.height)
    _text(fmt, "anamorphic", "FALSE")
    _text(fmt, "pixelaspectratio", "square")
    _text(fmt, "fielddominance", "none")
    v1 = ET.SubElement(video, "track")
    after_clip = {t.left: t for t in plan.transitions}
    # Lane w23-bins (CUT-180): a nested cut is a clipitem holding its own <sequence> (xmeml's nested sequence, as
    # Premiere writes a nest); its V1 pieces sit in that sequence at frames counted from the nest's start.
    from .export_nest import storyline_runs
    nests = [0]
    for nest_id, nest_name, run in storyline_runs(plan.storyline, plan.nest_ids):
        if nest_id is None:
            for clip in run:
                if clip.kind == "clip":
                    picture(v1, clip, enabled=clip.enabled)
                    if clip.id in after_clip:
                        _transition_item(v1, after_clip[clip.id], rate)
            continue
        start, frames = run[0].record_start, sum(c.duration for c in run)
        nests[0] += 1
        counter["clip"] += 1
        holder = ET.SubElement(v1, "clipitem", {"id": f"clipitem-{counter['clip']}"})
        _text(holder, "name", nest_name)
        _text(holder, "enabled", "TRUE")
        _text(holder, "duration", frames)
        _rate(holder, rate)
        for tag, value in (("start", start), ("end", start + frames), ("in", 0), ("out", frames)):
            _text(holder, tag, value)
        nested = ET.SubElement(holder, "sequence", {"id": f"sequence-{nests[0] + 1}"})
        _text(nested, "name", nest_name)
        _text(nested, "duration", frames)
        _rate(nested, rate)
        _timecode(nested, 0, rate, plan.drop_frame)
        n_video = ET.SubElement(ET.SubElement(nested, "media"), "video")
        n_fmt = ET.SubElement(ET.SubElement(n_video, "format"), "samplecharacteristics")
        _rate(n_fmt, rate)
        _text(n_fmt, "width", plan.width)
        _text(n_fmt, "height", plan.height)
        n_v1 = ET.SubElement(n_video, "track")
        for clip in run:
            if clip.kind == "clip":
                picture(n_v1, clip, enabled=clip.enabled, offset=start)
                if clip.id in after_clip:
                    from dataclasses import replace
                    t = after_clip[clip.id]
                    _transition_item(n_v1, replace(t, cut=t.cut - start), rate)
        _text(n_v1, "enabled", "TRUE")
        _text(n_v1, "locked", "FALSE")
    _text(v1, "enabled", "TRUE")
    _text(v1, "locked", "FALSE")
    if plan.overlays:
        v2 = ET.SubElement(video, "track")
        for clip in plan.overlays:
            picture(v2, clip, enabled=clip.enabled)
        _text(v2, "enabled", _bool(plan.overlay_enabled))
        _text(v2, "locked", "FALSE")

    audio = ET.SubElement(media, "audio")
    afmt = ET.SubElement(ET.SubElement(audio, "format"), "samplecharacteristics")
    _text(afmt, "depth", 16)
    _text(afmt, "samplerate", 48000)
    expected_music = 0
    if plan.music is not None:
        music = plan.music
        a1 = ET.SubElement(audio, "track")
        item = clipitem(a1, music.info.path.stem, enabled=True, record_start=music.record_start, length=music.duration,
                        source_seconds=Fraction(music.source_in) / rate, media_seconds=music.info.duration, audio_track=True)
        file_element(item, "m:" + music.track_sha256, music.info.link or music.info.path, f_rate=rate, frames=_floor(music.info.duration * rate),
                     width=None, height=None, audio=(music.info.sample_rate, music.info.channels))
        source_track = ET.SubElement(item, "sourcetrack")
        _text(source_track, "mediatype", "audio")
        _text(source_track, "trackindex", 1)
        _text(a1, "enabled", "TRUE")
        _text(a1, "locked", "FALSE")
        expected_music = 1
    voiced = [c for c in [*plan.storyline, *plan.overlays] if c.kind == "clip" and c.audio and c.enabled and c.info is not None]
    if voiced:
        a2 = ET.SubElement(audio, "track")
        for clip in sorted(voiced, key=lambda c: c.record_start):
            item = clipitem(a2, (clip.label or clip.info.path.stem) + " audio", enabled=True, record_start=clip.record_start,
                            length=clip.duration, source_seconds=Fraction(clip.source_in) / clip.source_rate,
                            media_seconds=Fraction(clip.info.frames) / clip.source_rate, audio_track=True)
            file_element(item, "a:" + clip.source["ref"], clip.info.path, f_rate=clip.source_rate, frames=clip.info.frames,
                         width=clip.info.width, height=clip.info.height, audio=(48000, 2))
            source_track = ET.SubElement(item, "sourcetrack")
            _text(source_track, "mediatype", "audio")
            _text(source_track, "trackindex", 1)
        _text(a2, "enabled", "TRUE")
        _text(a2, "locked", "FALSE")

    expected_sound = 0
    for track in plan.sound_tracks:
        # A1-A3: one audio track each, named like Premiere names them (MZ.TrackName), the level as
        # an Audio Levels filter (linear gain = clip + track gain; fades as keyframes 0 -> gain -> 0).
        node_track = ET.SubElement(audio, "track", {"MZ.TrackName": f"{track.id} {track.role}"})
        for sound in track.clips:
            if sound.record_start >= total:
                continue
            length = min(sound.duration, total - sound.record_start)
            info = sound.info
            media_seconds = Fraction(info.samples or 0, info.sample_rate) if info.samples else None
            item = clipitem(node_track, sound.label or sound.id, enabled=track.audible, record_start=sound.record_start, length=length,
                            source_seconds=Fraction(sound.source_in) / sound.source_rate, media_seconds=media_seconds, audio_track=True)
            frames = _floor(media_seconds * rate) if media_seconds is not None else _round(Fraction(sound.source_in) / sound.source_rate * rate) + length
            linked = info.link or info.path  # p12-cut (CUT-021): the original, never the analysis proxy
            file_element(item, "s:" + str(linked), linked, f_rate=rate, frames=frames, width=None, height=None,
                         audio=(info.sample_rate, info.channels))
            source_track = ET.SubElement(item, "sourcetrack")
            _text(source_track, "mediatype", "audio")
            _text(source_track, "trackindex", 1)
            _levels(item, sound.gain_db, sound.fade_in, sound.fade_out if length == sound.duration else 0, length,
                    int(item.findtext("in")), _key_points(sound, rate))
            expected_sound += 1
        _text(node_track, "enabled", _bool(track.audible))
        _text(node_track, "locked", "FALSE")

    for marker in plan.markers:
        if marker.frame >= total:
            continue
        node = ET.SubElement(sequence, "marker")
        _text(node, "comment", marker.note)
        _text(node, "name", marker.name or marker.id)
        _text(node, "in", marker.frame)
        _text(node, "out", marker.frame + marker.duration if marker.duration else -1)
    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode")
    text = '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n' + body + "\n"
    return Fcp7(text=text, expected_storyline_clips=len(plan.storyline_clips), expected_music_clips=expected_music, expected_nested=nests[0],
                expected_sound_clips=expected_sound if plan.sound_tracks else None,
                expected_transitions=len(plan.transitions),
                expected_retimed=sum(1 for c in [*plan.storyline, *plan.overlays] if c.kind == "clip" and c.retime is not None and not c.is_still))


_ALIGN = {(True, False): "start-black", (False, True): "end-black"}
_DIP = {"dip_black": (0, 0, 0), "dip_white": (255, 255, 255)}


def _transition_item(track: ET.Element, transition, rate: Fraction) -> None:
    """``<transitionitem>`` on the cut: record frames, alignment, the dissolve (or dip) effect."""
    node = ET.SubElement(track, "transitionitem")
    _rate(node, rate)
    _text(node, "start", transition.start)
    _text(node, "end", transition.start + transition.duration)
    _text(node, "alignment", _ALIGN.get((transition.before == 0, transition.after == 0), "center"))
    effect = ET.SubElement(node, "effect")
    dip = transition.kind in _DIP
    for tag, value in (("name", "Dip to Color Dissolve" if dip else "Cross Dissolve"),
                       ("effectid", "Dip to Color Dissolve" if dip else "Cross Dissolve"),
                       ("effectcategory", "Dissolve"), ("effecttype", "transition"), ("mediatype", "video")):
        _text(effect, tag, value)
    if dip:
        parameter = ET.SubElement(effect, "parameter")
        _text(parameter, "parameterid", "color")
        _text(parameter, "name", "Color")
        value = ET.SubElement(parameter, "value")
        for channel, level in zip(("alpha", "red", "green", "blue"), (255, *_DIP[transition.kind])):
            _text(value, channel, level)


def _time_remap(item: ET.Element, clip: PlanClip, rate: Fraction) -> None:
    """FCP7 ``Time Remap`` filter: constant speed (percent), reverse, frame blending; a freeze is
    variable speed holding one source frame."""
    retime = clip.retime
    effect = ET.SubElement(ET.SubElement(item, "filter"), "effect")
    for tag, value in (("name", "Time Remap"), ("effectid", "timeremap"), ("effectcategory", "motion"),
                       ("effecttype", "motion"), ("mediatype", "video")):
        _text(effect, tag, value)

    def param(pid: str, name: str, value: object, **extra: object) -> ET.Element:
        node = ET.SubElement(effect, "parameter")
        _text(node, "parameterid", pid)
        _text(node, "name", name)
        for key, v in extra.items():
            _text(node, key, v)
        _text(node, "value", value)
        return node

    if retime.curve:
        # Variable speed (lane w7-export): ``speed`` is the clip's average (what a reader without
        # the graph shows), the graphdict carries the curve exactly at every record frame.
        from .export_curve import average_speed
        param("variablespeed", "variablespeed", 1, valuemin=0, valuemax=1)
        param("speed", "speed", f"{float(average_speed(clip, rate) * 100):.6g}", valuemin=-100000, valuemax=100000)
        param("reverse", "reverse", _bool(retime.reverse))
        param("frameblending", "frameblending", _bool(retime.sampling != "nearest"))
        graph = param("graphdict", "graphdict", 0)
        for when, value in curve_graph(clip, rate):
            key = ET.SubElement(graph, "keyframe")
            _text(key, "when", when)
            _text(key, "value", _frame_text(value, clip.source_rate / rate, retime.reverse))
        return
    param("variablespeed", "variablespeed", 1 if retime.freeze else 0, valuemin=0, valuemax=1)
    param("speed", "speed", f"{float(retime.speed * 100):.6g}", valuemin=-100000, valuemax=100000)
    param("reverse", "reverse", _bool(retime.reverse))
    param("frameblending", "frameblending", _bool(retime.sampling != "nearest"))
    if retime.freeze:
        graph = param("graphdict", "graphdict", 0)
        held = _round(Fraction(clip.source_in) / clip.source_rate * rate)
        for when in (0, clip.duration):
            key = ET.SubElement(graph, "keyframe")
            _text(key, "when", when)
            _text(key, "value", held)


def _frame_text(value: Fraction, q: Fraction = Fraction(1), reverse: bool = False) -> str:
    """A graphdict value as a decimal (FCP7 allows a fractional value, "blended into the next
    frame"): exact when it has <= 9 decimals, else rounded towards the frame it shows (down forward,
    up reversed) with as many decimals as keep that frame: ``floor(value * q)`` forward, the frame
    that ends at ``value * q`` reversed (``q`` = source frames per sequence frame)."""
    def shown(v: Fraction) -> int:
        s = v * q
        return -(-s.numerator // s.denominator) - 1 if reverse else s.numerator // s.denominator

    for digits in range(0, 19):
        scale = 10 ** digits
        cut = value * scale
        whole = -(-cut.numerator // cut.denominator) if reverse else cut.numerator // cut.denominator
        approx = Fraction(whole, scale)
        if approx == value or (digits >= 9 and shown(approx) == shown(value)):
            break
    sign, whole = ("-", -whole) if whole < 0 else ("", whole)
    text = str(whole).rjust(digits + 1, "0")
    return sign + (text if not digits else f"{text[:-digits]}.{text[-digits:]}".rstrip("0").rstrip("."))


def curve_graph(clip: PlanClip, rate: Fraction) -> list[tuple[int, Fraction]]:
    """A speed curve (lane w6-tempo) as FCP7 ``graphdict`` keyframes ``(when, value)``.

    ``when`` = record frame in the clipitem, 0 .. its length; ``value`` = the media frame there at
    the sequence timebase (like ``in``/``out``; a 48 fps take counts halves): forward
    ``(source_in + P(when)) / q``, reversed ``(end - P(when)) / q`` (``end`` = one past the clip's last
    source frame), ``P`` from ``timeline/speed_curve.py``. One key per record frame: ``value`` is a
    decimal in FCP7, so a line between two rounded keys could land on the wrong frame; per-frame
    keys are exact at every frame (Resolve itself imports a variable speed as one key per frame).
    A hold is a run of keys with one value; a step key a corner of the line, never a jump."""
    sc = speed_curve()
    keys, q = clip.keys(), clip.source_rate / rate
    end = Fraction(clip.source_in + clip.source_frames_used(rate))
    out = []
    for when in range(clip.duration + 1):
        p = sc.position(keys, q, when)
        out.append((when, (end - p if clip.retime.reverse else clip.source_in + p) / q))
    return out


def _level(gain_db: float) -> str:
    """FCP7's Audio Levels ``Level`` is linear gain (Apple: decibels = 20 * log10(Level))."""
    return f"{10 ** (gain_db / 20):.6g}"


FCP7_STEP_DB = 0.5  # FCP7 / Premiere interpolate Level linearly in gain: a dB ramp is cut into steps this fine


def _key_points(sound: Any, rate: Fraction) -> list[tuple[int, float]]:
    """A clip's volume keys as (frame from its in point at the sequence rate, dB), a ramp subdivided so
    the straight line FCP7 draws in linear Level stays within FCP7_STEP_DB of the dB-linear mix."""
    keys = [(int(round((Fraction(frame - sound.source_in) / sound.source_rate) * rate)), db) for frame, db in sound.volume_keys]
    out: list[tuple[int, float]] = []
    for (f0, d0), (f1, d1) in zip(keys, keys[1:]):
        out.append((f0, d0))
        steps = int(abs(d1 - d0) // (2 * FCP7_STEP_DB))
        for k in range(1, steps + 1):
            frame = f0 + round((f1 - f0) * k / (steps + 1))
            if f0 < frame < f1:
                out.append((frame, d0 + (d1 - d0) * (frame - f0) / (f1 - f0)))
    if keys:
        out.append(keys[-1])
    kept: list[tuple[int, float]] = []
    for frame, db in out:
        if kept and kept[-1][0] == frame:
            kept[-1] = (frame, db)
        else:
            kept.append((frame, db))
    return kept


def _levels(item: ET.Element, gain_db: float, fade_in: int, fade_out: int, length: int, in_frame: int,
            keys: list[tuple[int, float]] | None = None) -> None:
    """The Audio Levels filter of a sound clipitem; nothing when the clip is at 0 dB without fades.

    Keyframes: ``when`` in frames of the clip's media (the clip starts at its ``in``); a fade in
    rises from 0 at the in point to the gain, a fade out falls to 0 at the out point. Premiere's
    reading of ``when`` is UNVERIFIED here (no Premiere on this machine)."""
    if not gain_db and not fade_in and not fade_out and not keys:
        return
    effect = ET.SubElement(ET.SubElement(item, "filter"), "effect")
    for tag, value in (("name", "Audio Levels"), ("effectid", "audiolevels"), ("effectcategory", "audiolevels"),
                       ("effecttype", "audiolevels"), ("mediatype", "audio")):
        _text(effect, tag, value)
    parameter = ET.SubElement(effect, "parameter")
    _text(parameter, "parameterid", "level")
    _text(parameter, "name", "Level")
    _text(parameter, "valuemin", 0)
    _text(parameter, "valuemax", "3.98109")
    _text(parameter, "value", _level(gain_db))
    if keys:
        # Volume keys (lane fs-duck): one keyframe per (subdivided) key, and the fade corners; each
        # value = fade factor x 10^((gain + key dB) / 20), the key level held outside the keys.
        def key_db(when: int) -> float:
            if when <= keys[0][0]:
                return keys[0][1]
            for (f0, d0), (f1, d1) in zip(keys, keys[1:]):
                if f0 <= when <= f1:
                    return d0 + (d1 - d0) * (when - f0) / (f1 - f0) if f1 > f0 else d1
            return keys[-1][1]

        def fade(when: int) -> float:
            factor = 1.0
            if fade_in and when < fade_in:
                factor = when / fade_in
            if fade_out and when > length - fade_out:
                factor = min(factor, (length - when) / fade_out)
            return max(0.0, factor)

        whens = sorted({w for w, _d in keys if 0 <= w <= length} | {0, length} | ({fade_in} if fade_in else set())
                       | ({length - fade_out} if fade_out else set()))
        for when in whens:
            key = ET.SubElement(parameter, "keyframe")
            _text(key, "when", in_frame + when)
            factor = fade(when)
            _text(key, "value", f"{factor * 10 ** ((gain_db + key_db(when)) / 20):.6g}" if factor else "0")
    elif fade_in or fade_out:
        points = [(0, "0" if fade_in else _level(gain_db)), (fade_in, _level(gain_db)),
                  (length - fade_out, _level(gain_db)), (length, "0" if fade_out else _level(gain_db))]
        kept: list[tuple[int, str]] = []
        for when, value in points:
            if not kept or kept[-1][0] != when:
                kept.append((when, value))
        for when, value in kept:
            key = ET.SubElement(parameter, "keyframe")
            _text(key, "when", in_frame + when)
            _text(key, "value", value)
