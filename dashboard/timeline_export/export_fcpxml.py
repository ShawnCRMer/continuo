"""FCPXML 1.9 for DaVinci Resolve from a resolved plan (TIMELINE-INPUT.md 6.2).

Structure (FilmCraft ``fcpxml.rs:1-15`` mapping; our tested Resolve file
``THE-BOOTH-RC4-MOVING.fcpxml``):

- one ``format`` per (rate, size); the sequence uses the timeline format;
- one ``asset`` per unique picture source; moving assets point to the
  full-length video-only remux and always carry ``start="0s"`` and an explicit
  rational ``duration`` (Resolve imports a duration-less moving asset as a
  still: ``resolve-fcpxml-moving-media-handoff.md``); stills carry none;
- V1 -> ``spine`` of ``asset-clip`` and ``gap``; V2 -> connected clips
  ``lane="1"``; music -> connected audio clip ``lane="-1" audioRole="music"``;
  clip audio set to on -> connected ``lane="-2"`` audio clip from the original
  file (``srcEnable="audio"``), because the remux has no audio;
- sound tracks A1-A3 -> connected audio clips on lanes -4 / -5 / -6 with ``audioRole``
  ``<role>.A<n>`` (dialogue / music / effects), ``adjust-volume amount`` = clip + track gain,
  fades as ``<param name="amount"><fadeIn type="linear" duration=.../></param>``; a muted
  (or not solo) track's clips are ``enabled="0"``;
- markers on the spine element that covers their frame;
- times as ``N/24s`` style rationals, never decimals; ``tcStart`` from
  ``start_tc_frame``.

Every ``<asset>`` names its file in a ``<media-rep kind="original-media" src=...>``
child, as FCPXML 1.9 requires (``<!ELEMENT asset (media-rep+, metadata?)>``;
``src`` is not an asset attribute in 1.9) and as Resolve 21 writes it itself.

``studio.timeline/1.1`` (nothing of it is written for a cut that does not use it):
- a retimed clip carries ``<timeMap frameSampling=...>`` with two linear ``timept`` (clip time ->
  media time over the whole asset; reversed: falling; freeze: constant) and its ``start`` in the
  retimed clip time (FCPXML 1.9 DTD ``timeMap``/``timept``); nearest = ``floor``, blend =
  ``frame-blending``, flow = ``optical-flow``; a speed curve / hold (lane w7-export) is the same
  timeMap with its points exact at every record frame (``curve_map``);
- fill / stretch: ``<adjust-conform type="fill">`` / fit + ``<adjust-transform scale>`` that turns
  the fit into the stretch (FCPXML has no stretch conform);
- a V1 transition is a spine ``<transition>`` at the cut minus its part before the cut, with the
  Cross Dissolve effect (FxPlug 4731E73A...). FCPXML has no standard dip: a dip is written as that
  dissolve named "Dip to Black" / "Dip to White" (README says so).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr

from .model import Plan, PlanClip
from . import colour
from .timebase import fcpx_seconds, fcpx_time

FCPXML_VERSION = "1.9"
CROSS_DISSOLVE_UID = "FxPlug:4731E73A-8DAC-4113-9A30-AE85B1761265"
FRAME_SAMPLING = {"nearest": "floor", "blend": "frame-blending", "flow": "optical-flow", "ai": "optical-flow"}
TRANSITION_NAMES = {"dissolve": "Cross Dissolve", "dip_black": "Dip to Black", "dip_white": "Dip to White"}


def file_uri(path: Path) -> str:
    """``file:///C:/a%20b/c.mp4``; percent-encodes everything outside the unreserved set."""
    from urllib.parse import quote

    text = str(Path(path).resolve()).replace("\\", "/")
    if text.startswith("//?/"):
        text = text[4:]
    if text.startswith("//"):
        return "file:" + quote(text, safe="/:")
    if not text.startswith("/"):
        text = "/" + text
    return "file://" + quote(text, safe="/:")


@dataclass
class AssetRef:
    id: str
    name: str
    src: Path
    moving: bool
    has_audio_asset: bool
    frames: int | None
    rate: Fraction | None
    format_id: str


@dataclass
class Fcpxml:
    text: str
    assets: dict[str, AssetRef] = field(default_factory=dict)
    expected_spine_clips: int = 0
    expected_compounds: int = 0  # lane w23-bins: ref-clips of compound clips (nested cuts) on the spine
    expected_music_clips: int = 0
    expected_sound_clips: int | None = None  # A1-A3 clips; None without sound tracks
    expected_transitions: int = 0
    expected_retimed: int = 0


def _attrs(pairs: list[tuple[str, Any]]) -> str:
    return "".join(f" {key}={quoteattr(str(value))}" for key, value in pairs if value is not None)


def _asset(pairs: list[tuple[str, Any]], src: Path) -> str:
    """``<asset ...>`` with its one ``<media-rep>`` child (FCPXML 1.9)."""
    rep = f"<media-rep{_attrs([('kind', 'original-media'), ('src', file_uri(src))])}/>"
    return f"    <asset{_attrs(pairs)}>\n      {rep}\n    </asset>"


def _format_name(rate: Fraction, height: int) -> str:
    label = f"{float(rate):.2f}".rstrip("0").rstrip(".").replace(".", "")
    return f"FFVideoFormat{height}p{label}"


def build_fcpxml(plan: Plan, remuxes: dict[str, Path], *, project_name: str, event_name: str, captions: dict[str, Any] | None = None) -> Fcpxml:
    """``remuxes`` maps ``source.ref`` -> video-only remux path for every moving source."""
    rate = plan.rate
    next_id = [0]

    def rid() -> str:
        next_id[0] += 1
        return f"r{next_id[0]}"

    resources: list[str] = []
    formats: dict[tuple[Fraction, int, int], str] = {}

    def format_id(f_rate: Fraction, width: int, height: int, space: str | None = None) -> str:
        space = space or plan.color.fcpxml  # the sequence: the cut's dominant space; a source: its own
        key = (f_rate, width, height, space)
        if key not in formats:
            formats[key] = rid()
            resources.append(
                f"    <format{_attrs([('id', formats[key]), ('name', _format_name(f_rate, height)), ('frameDuration', fcpx_time(1, f_rate)), ('width', width), ('height', height), ('colorSpace', space)])}/>"
            )
        return formats[key]

    sequence_format = format_id(rate, plan.width, plan.height)
    assets: dict[str, AssetRef] = {}
    used_names: set[str] = set()

    def unique_name(stem: str) -> str:
        name, n = stem, 2
        while name.lower() in used_names:
            name, n = f"{stem} ({n})", n + 1
        used_names.add(name.lower())
        return name

    def picture_asset(clip: PlanClip) -> AssetRef:
        ref = clip.source["ref"]
        if ref in assets:
            return assets[ref]
        info = clip.info
        assert info is not None
        if info.is_still:
            asset = AssetRef(rid(), unique_name(info.path.stem), info.path, False, False, None, None, sequence_format)
            resources.append(_asset([('id', asset.id), ('name', asset.name), ('start', '0s'), ('hasVideo', 1), ('format', asset.format_id)], asset.src))
        else:
            fmt = format_id(clip.source_rate, info.width, info.height, colour.of_source(info).fcpxml)
            remux = remuxes[ref]
            asset = AssetRef(rid(), unique_name(remux.stem), remux, True, False, info.frames, clip.source_rate, fmt)
            resources.append(_asset([('id', asset.id), ('name', asset.name), ('start', '0s'), ('duration', fcpx_time(info.frames, clip.source_rate)), ('hasVideo', 1), ('hasAudio', 0), ('format', fmt)], remux))
        assets[ref] = asset
        return asset

    def audio_asset(clip: PlanClip) -> AssetRef:
        key = "audio:" + clip.source["ref"]
        if key in assets:
            return assets[key]
        info = clip.info
        asset = AssetRef(rid(), unique_name(info.path.stem + " audio"), info.path, True, True, info.frames, clip.source_rate, "")
        resources.append(_asset([('id', asset.id), ('name', asset.name), ('start', '0s'), ('duration', fcpx_time(info.frames, clip.source_rate)), ('hasVideo', 0), ('hasAudio', 1), ('audioSources', 1), ('audioChannels', 2), ('audioRate', 48000)], info.path))
        assets[key] = asset
        return asset

    def sound_asset(sound: Any) -> AssetRef:
        key = "sound:" + str(sound.info.path)
        if key in assets:
            return assets[key]
        info = sound.info
        frames = int(Fraction(info.samples or 0, info.sample_rate) * sound.source_rate)  # whole source frames in the file
        linked = info.link or info.path  # p12-cut (CUT-021): the original, never the analysis proxy
        asset = AssetRef(rid(), unique_name(linked.stem), linked, True, True, frames, sound.source_rate, "")
        resources.append(_asset([('id', asset.id), ('name', asset.name), ('start', '0s'), ('duration', fcpx_time(frames, sound.source_rate)), ('hasVideo', 0), ('hasAudio', 1), ('audioSources', 1), ('audioChannels', info.channels), ('audioRate', info.sample_rate)], linked))  # p12-cutb: src = the linked file
        assets[key] = asset
        return asset

    tc0 = Fraction(plan.start_tc_frame) / rate

    def rec(frame: int) -> Fraction:
        return tc0 + Fraction(frame) / rate

    # Spine elements: (record_start, record_end, local_start, open_tag, close_tag, children)
    def spine_of(clips: list[PlanClip], origin: int) -> list[dict[str, Any]]:
        """Spine elements of ``clips``; an element's offset is ``origin + record_start`` frames."""
        out: list[dict[str, Any]] = []
        for clip in clips:
            if clip.kind == "gap":
                local = Fraction(origin + clip.record_start) / rate
                out.append({"clip": clip, "local": local, "children": [], "open":
                            f"<gap{_attrs([('name', clip.label or 'Gap'), ('offset', fcpx_time(origin + clip.record_start, rate)), ('start', fcpx_time(origin + clip.record_start, rate)), ('duration', fcpx_time(clip.duration, rate))])}",
                            "close": "gap"})
                continue
            asset = picture_asset(clip)
            local = Fraction(0) if asset.rate is None else Fraction(clip.source_in) / asset.rate
            start_text = "0s" if asset.rate is None else fcpx_time(clip.source_in, asset.rate)  # source frames at the source rate
            intrinsic = _picture_params(clip, asset, plan)
            if clip.retime is not None:
                local = retimed_start(clip, asset, rate)
                start_text = fcpx_seconds(local)
            out.append({"clip": clip, "local": local, "children": list(intrinsic), "open":
                        f"<asset-clip{_attrs([('ref', asset.id), ('name', clip.label or asset.name), ('offset', fcpx_time(origin + clip.record_start, rate)), ('start', start_text), ('duration', fcpx_time(clip.duration, rate)), ('enabled', None if clip.enabled else 0)])}",
                        "close": "asset-clip"})
            if clip.audio and clip.enabled:
                out[-1]["children"].append(_audio_child(audio_asset(clip), clip, local, local, rate, lane=-2))
        return out

    def with_transitions(elements: list[dict[str, Any]], origin: int) -> list[dict[str, Any]]:
        mine = [t for t in plan.transitions if any(e["clip"] is not None and e["clip"].id == t.left for e in elements)]
        if not mine:
            return elements
        at = {element["clip"].id: index for index, element in enumerate(elements)}
        for transition in sorted(mine, key=lambda t: -at[t.left]):
            tag = (f"<transition{_attrs([('name', TRANSITION_NAMES[transition.kind]), ('offset', fcpx_time(origin + transition.start, rate)), ('duration', fcpx_time(transition.duration, rate))])}>"
                   f"<filter-video{_attrs([('ref', effect_id()), ('name', 'Cross Dissolve')])}/></transition>")
            elements.insert(at[transition.left] + 1, {"clip": None, "children": [], "raw": tag})
        return elements

    effect = [None]

    def effect_id() -> str:
        if effect[0] is None:
            effect[0] = rid()
            resources.append(f"    <effect{_attrs([('id', effect[0]), ('name', 'Cross Dissolve'), ('uid', CROSS_DISSOLVE_UID)])}/>")
        return effect[0]

    def lines_of(elements: list[dict[str, Any]], indent: str) -> list[str]:
        out = []
        for element in elements:
            if element.get("raw"):
                out.append(f"{indent}{element['raw']}")
            elif element["children"]:
                out.append(f"{indent}{element['open']}>")
                out.extend(f"{indent}  {child}" for child in element["children"])
                out.append(f"{indent}</{element['close']}>")
            else:
                out.append(f"{indent}{element['open']}/>")
        return out

    # Lane w23-bins (CUT-180): a nested cut is a compound clip: a <media> resource holding its own sequence (its V1
    # pieces at offsets from 0) and a <ref-clip> on the spine (as Final Cut Pro writes a compound clip).
    from types import SimpleNamespace
    from .export_nest import storyline_runs
    spine: list[dict[str, Any]] = []
    compounds = 0
    for nest_id, nest_name, run in storyline_runs(plan.storyline, plan.nest_ids):
        if nest_id is None:
            spine.extend(spine_of(run, plan.start_tc_frame))
            continue
        start, frames = run[0].record_start, sum(c.duration for c in run)
        media_id = rid()
        inner = with_transitions(spine_of(run, -start), -start)
        resources.append("\n".join([
            f"    <media{_attrs([('id', media_id), ('name', nest_name)])}>",
            f"      <sequence{_attrs([('format', sequence_format), ('duration', fcpx_time(frames, rate)), ('tcStart', '0s'), ('tcFormat', 'NDF'), ('audioLayout', 'stereo'), ('audioRate', '48k')])}>",
            "        <spine>", *lines_of(inner, "          "), "        </spine>", "      </sequence>", "    </media>"]))
        compounds += 1
        holder = SimpleNamespace(id=nest_id, kind="nest", record_start=start, record_end=start + frames, duration=frames)
        spine.append({"clip": holder, "local": Fraction(0), "children": [], "open":
                      f"<ref-clip{_attrs([('ref', media_id), ('name', nest_name), ('offset', fcpx_time(plan.start_tc_frame + start, rate)), ('start', '0s'), ('duration', fcpx_time(frames, rate))])}",
                      "close": "ref-clip"})

    def covering(frame: int) -> dict[str, Any]:
        for element in spine:
            if element["clip"] is None:
                continue
            if element["clip"].record_start <= frame < element["clip"].record_end:
                return element
        return spine[-1]

    def local_at(element: dict[str, Any], frame: int) -> Fraction:
        return element["local"] + Fraction(frame - element["clip"].record_start) / rate

    for clip in plan.overlays:
        parent = covering(clip.record_start)
        asset = picture_asset(clip)
        start = Fraction(0) if asset.rate is None else Fraction(clip.source_in) / asset.rate
        offset = local_at(parent, clip.record_start)
        enabled = clip.enabled and plan.overlay_enabled
        start_text = '0s' if asset.rate is None else fcpx_time(clip.source_in, asset.rate)
        if clip.retime is not None:
            start_text = fcpx_seconds(retimed_start(clip, asset, rate))
        params = "".join(_picture_params(clip, asset, plan))
        tag = f"<asset-clip{_attrs([('ref', asset.id), ('lane', 1), ('name', clip.label or asset.name), ('offset', fcpx_seconds(offset)), ('start', start_text), ('duration', fcpx_time(clip.duration, rate)), ('enabled', None if enabled else 0)])}"
        parent["children"].append(f"{tag}>{params}</asset-clip>" if params else f"{tag}/>")
        if clip.audio and enabled:
            parent["children"].append(_audio_child(audio_asset(clip), clip, offset, start, rate, lane=-3))

    expected_music = 0
    if plan.music is not None:
        music = plan.music
        music_frames = int(music.info.duration * rate)  # whole sequence frames inside the recording
        linked = music.info.link or music.info.path
        asset = AssetRef(rid(), unique_name(linked.stem), linked, True, True, music_frames, rate, "")
        resources.append(_asset([('id', asset.id), ('name', asset.name), ('start', '0s'), ('duration', fcpx_time(music_frames, rate)), ('hasVideo', 0), ('hasAudio', 1), ('audioSources', 1), ('audioChannels', music.info.channels), ('audioRate', music.info.sample_rate)], linked))  # p12-cutb: src = the linked file
        assets["music:" + music.track_sha256] = asset
        parent = covering(music.record_start)
        gain = f"<adjust-volume amount={quoteattr(f'{music.gain_db:g}dB')}/>" if music.gain_db else ""
        tag = f"<asset-clip{_attrs([('ref', asset.id), ('lane', -1), ('name', asset.name), ('offset', fcpx_seconds(local_at(parent, music.record_start))), ('start', fcpx_time(music.source_in, rate)), ('duration', fcpx_time(music.duration, rate)), ('audioRole', 'music')])}"
        parent["children"].insert(0, f"{tag}>{gain}</asset-clip>" if gain else f"{tag}/>")
        expected_music = 1

    total = plan.total_frames
    expected_sound = 0
    for lane, track in enumerate(plan.sound_tracks, start=4):
        # A1-A3: connected audio clips, one lane per track (-4, -5, -6), role.subrole = role.A<n>.
        for sound in track.clips:
            if sound.record_start >= total:
                continue
            asset = sound_asset(sound)
            parent = covering(sound.record_start)
            children = _volume(sound.gain_db, sound.fade_in, sound.fade_out, rate, sound.volume_keys, sound.source_rate)
            tag = f"<asset-clip{_attrs([('ref', asset.id), ('lane', -lane), ('name', sound.label or sound.id), ('offset', fcpx_seconds(local_at(parent, sound.record_start))), ('start', fcpx_time(sound.source_in, sound.source_rate)), ('duration', fcpx_time(sound.duration, rate)), ('enabled', None if track.audible else 0), ('audioRole', f'{track.role}.{track.id}')])}"
            parent["children"].append(f"{tag}>{children}</asset-clip>" if children else f"{tag}/>")
            expected_sound += 1
            expected_music += 1 if track.role == "music" else 0  # p12-cut: the song on its music track is a music clip

    # Lane w23-subs (CUT-011): the cut's subtitles as FCPXML 1.9 iTT captions (DTD: caption (text*, text-style-def*),
    # anchored like connected clips, role "iTT?captionFormat=ITT.<lang>"), lane 2 above V2; before the markers
    # (anchor items precede marker items in a clip).
    if captions is not None and plan.storyline:
        from xml.sax.saxutils import escape as _escape
        from .subtitles import display_text as _display_text
        style = captions["track"]["style"]
        placement = "top" if style["y"] < 50 else "bottom"
        for number, cue in enumerate(captions["cues"], start=1):
            if cue["start"] >= total:
                continue
            parent = covering(cue["start"])
            duration = min(cue["end"], total) - cue["start"]
            text = _display_text(cue["text"], captions["track"]["rules"])
            if style["uppercase"]:
                text = text.upper()
            ts = f"ts{rid()}"
            parent["children"].append(
                f"<caption{_attrs([('lane', 2), ('offset', fcpx_seconds(local_at(parent, cue['start']))), ('name', text.replace(chr(10), ' ')[:60]), ('duration', fcpx_time(duration, rate)), ('role', 'iTT?captionFormat=ITT.en')])}>"
                f"<text{_attrs([('placement', placement)])}><text-style{_attrs([('ref', ts)])}>{_escape(text)}</text-style></text>"
                f"<text-style-def{_attrs([('id', ts)])}><text-style{_attrs([('font', '.AppleSystemUIFont'), ('fontSize', '13'), ('fontFace', 'Regular'), ('fontColor', '1 1 1 1'), ('backgroundColor', '0 0 0 1')])}/></text-style-def></caption>")

    for marker in plan.markers:
        if marker.frame >= total:
            continue
        parent = covering(marker.frame)
        parent["children"].append(
            f"<marker{_attrs([('start', fcpx_seconds(local_at(parent, marker.frame))), ('duration', fcpx_time(max(1, marker.duration), rate)), ('value', marker.name or marker.id), ('note', marker.note or None)])}/>"
        )

    with_transitions(spine, plan.start_tc_frame)  # before the resources are written: it may add the effect
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<!DOCTYPE fcpxml>",
        f'<fcpxml version="{FCPXML_VERSION}">',
        "  <resources>",
        *resources,
        "  </resources>",
        "  <library>",
        f"    <event{_attrs([('name', event_name)])}>",
        f"      <project{_attrs([('name', project_name)])}>",
        f"        <sequence{_attrs([('format', sequence_format), ('duration', fcpx_time(total, rate)), ('tcStart', fcpx_time(plan.start_tc_frame, rate)), ('tcFormat', 'DF' if plan.drop_frame else 'NDF'), ('audioLayout', 'stereo'), ('audioRate', '48k')])}>",
        "          <spine>",
    ]
    lines.extend(lines_of(spine, "            "))
    lines += ["          </spine>", "        </sequence>", "      </project>", "    </event>", "  </library>", "</fcpxml>", ""]
    return Fcpxml(
        text="\n".join(lines), assets=assets,
        expected_spine_clips=len(plan.storyline_clips), expected_music_clips=expected_music, expected_compounds=compounds,
        expected_sound_clips=expected_sound if plan.sound_tracks else None,
        expected_transitions=len(plan.transitions),
        expected_retimed=sum(1 for c in [*plan.storyline, *plan.overlays] if c.kind == "clip" and c.retime is not None and not c.is_still),
    )


def retimed_start(clip: PlanClip, asset: AssetRef, rate: Fraction) -> Fraction:
    """A retimed clip's ``start`` on its timeMap's clip-time axis. Forward: the media time of its in
    point over the speed. Reversed (media = asset_end - t * speed): the end of its source range, so
    the clip starts on its last shown frame and runs down to its in point. Freeze: 0."""
    retime = clip.retime
    if retime.curve:
        return curve_map(clip, asset, rate)[0]
    if retime.freeze:
        return Fraction(0)
    if retime.reverse:
        end = Fraction(clip.source_in + clip.source_frames_used(rate)) / asset.rate
        return (Fraction(asset.frames) / asset.rate - end) / retime.speed
    return Fraction(clip.source_in) / asset.rate / retime.speed


def time_map(clip: PlanClip, asset: AssetRef, rate: Fraction) -> str:
    """``<timeMap>`` of a retimed clip over its whole asset: clip time -> media time (linear)."""
    retime = clip.retime
    if retime.curve:
        start, body = curve_map(clip, asset, rate)
        return f"<timeMap{_attrs([('frameSampling', FRAME_SAMPLING[retime.sampling])])}>{body}</timeMap>"
    media = Fraction(asset.frames) / asset.rate
    if retime.freeze:
        held = fcpx_time(clip.source_in, asset.rate)
        points = [("0s", held), (fcpx_time(clip.duration, rate), held)]
    else:
        span = fcpx_seconds(media / retime.speed)
        points = [("0s", fcpx_seconds(media) if retime.reverse else "0s"), (span, "0s" if retime.reverse else fcpx_seconds(media))]
    body = "".join(f"<timept{_attrs([('time', t), ('value', v), ('interp', 'linear')])}/>" for t, v in points)
    return f"<timeMap{_attrs([('frameSampling', FRAME_SAMPLING[retime.sampling])])}>{body}</timeMap>"


def curve_map(clip: PlanClip, asset: AssetRef, rate: Fraction) -> tuple[Fraction, str]:
    """A speed curve (lane w6-tempo) as ``timeMap`` points: ``(start, timept...)``.

    Clip time ``start + x / rate`` (``x`` = record offset) maps to media time ``(source_in + P(x)) /
    asset_rate`` (forward) or ``(end - P(x)) / asset_rate`` (reverse, ``end`` = one past the clip's
    last source frame, as the constant reverse does), with ``P`` from ``timeline/speed_curve.py``:
    one ``timept`` per key, one per record frame inside a ramp (its position is quadratic / quartic,
    so a frame-exact map needs them; Resolve itself imports a variable speed as one key per frame),
    ``interp`` ``smooth2`` inside a smooth ramp, else ``linear``. A hold is two points with the
    same ``value``. A ``step`` key steps the SPEED, not the position: its segment is a straight line
    whose corner is the next key's point (two points at one time would be a jump of the picture,
    which a step does not make). A head and a tail point run at the edge speeds to the asset's
    ends, so transitions find their handles; a clip that starts on a hold has no head."""
    from .export_curve import edge_rates, points

    span = Fraction(asset.frames) / asset.rate
    head_rate, tail_rate = (r * rate / asset.rate for r in edge_rates(clip, rate))  # media s per clip s
    end = Fraction(clip.source_in + clip.source_frames_used(rate))
    reverse = clip.retime.reverse

    def media(position: Fraction) -> Fraction:
        return (end - position) / asset.rate if reverse else (clip.source_in + position) / asset.rate

    rows = points(clip, rate)
    first = media(rows[0].position)
    room = (span - first) if reverse else first
    start = room / head_rate if head_rate else Fraction(0)
    out: list[tuple[Fraction, Fraction, str]] = []
    if start:
        out.append((Fraction(0), span if reverse else Fraction(0), "linear"))
    out += [(start + p.x / rate, media(p.position), "smooth2" if p.ease == "smooth" else "linear") for p in rows]
    last_t, last_v, _interp = out[-1]
    rest = last_v if reverse else span - last_v
    if rest > 0 and tail_rate:  # a clip that ends inside a hold has no tail
        out.append((last_t + rest / tail_rate, Fraction(0) if reverse else span, "linear"))
    body = "".join(f"<timept{_attrs([('time', fcpx_seconds(t)), ('value', fcpx_seconds(v)), ('interp', i)])}/>" for t, v, i in out)
    return start, body


def _picture_params(clip: PlanClip, asset: AssetRef, plan: Plan) -> list[str]:
    """Timing and intrinsic children of a picture asset-clip, in DTD order (timeMap, adjust-conform,
    adjust-transform); empty for a 1.0 cut."""
    out: list[str] = []
    if clip.retime is not None and asset.rate is not None:
        out.append(time_map(clip, asset, plan.rate))
    if plan.fit == "fill":
        out.append('<adjust-conform type="fill"/>')
    elif plan.fit == "stretch" and clip.info is not None and clip.info.width and clip.info.height:
        w, h = clip.info.width, clip.info.height
        fit = min(Fraction(plan.width, w), Fraction(plan.height, h))
        sx, sy = Fraction(plan.width, w) / fit, Fraction(plan.height, h) / fit
        if sx != 1 or sy != 1:
            out.append(f'<adjust-transform scale="{float(sx):.6g} {float(sy):.6g}"/>')
    return out


def _audio_child(asset: AssetRef, clip: PlanClip, offset: Fraction, start: Fraction, rate: Fraction, *, lane: int) -> str:
    return (
        f"<asset-clip{_attrs([('ref', asset.id), ('lane', lane), ('name', asset.name), ('offset', fcpx_seconds(offset)), ('start', fcpx_seconds(start)), ('duration', fcpx_time(clip.duration, rate)), ('srcEnable', 'audio'), ('audioRole', 'dialogue')])}/>"
    )


def _volume(gain_db: float, fade_in: int, fade_out: int, rate: Fraction, keys: tuple = (), source_rate: Fraction | None = None) -> str:
    """``<adjust-volume>`` of a sound clip: the gain (clip + track) and linear fades on its amount.

    Volume automation (lane fs-duck): ``<keyframeAnimation>`` on the amount, one ``<keyframe>`` per
    key at the clip's media time (source frame / source rate, like ``start``), value = gain + key dB,
    ``interp="linear" curve="linear"`` (straight in dB between keys, as the mix renders it)."""
    if keys:
        fades = ""
        if fade_in:
            fades += f"<fadeIn{_attrs([('type', 'linear'), ('duration', fcpx_time(fade_in, rate))])}/>"
        if fade_out:
            fades += f"<fadeOut{_attrs([('type', 'linear'), ('duration', fcpx_time(fade_out, rate))])}/>"
        frames = "".join(f"<keyframe{_attrs([('time', fcpx_time(frame, source_rate)), ('value', f'{gain_db + db:g}dB'), ('interp', 'linear'), ('curve', 'linear')])}/>"
                         for frame, db in keys)
        return (f"<adjust-volume amount={quoteattr(f'{gain_db:g}dB')}><param name=\"amount\">{fades}"
                f"<keyframeAnimation>{frames}</keyframeAnimation></param></adjust-volume>")
    if not gain_db and not fade_in and not fade_out:
        return ""
    fades = ""
    if fade_in:
        fades += f"<fadeIn{_attrs([('type', 'linear'), ('duration', fcpx_time(fade_in, rate))])}/>"
    if fade_out:
        fades += f"<fadeOut{_attrs([('type', 'linear'), ('duration', fcpx_time(fade_out, rate))])}/>"
    param = f"<param name=\"amount\">{fades}</param>" if fades else ""
    return f"<adjust-volume amount={quoteattr(f'{gain_db:g}dB')}>{param}</adjust-volume>" if param else f"<adjust-volume amount={quoteattr(f'{gain_db:g}dB')}/>"
