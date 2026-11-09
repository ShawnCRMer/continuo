"""Validate a ``studio.timeline/1`` document and resolve it into an export plan.

The plan is the single input of render, FCPXML and EDL: integer frames at the
sequence rate for record times, source frames at the source rate, resolved
source facts.  Validation is the export-relevant subset of TIMELINE-INPUT.md
section 2.4; it fails closed (an offline source stops the export, it is never
silently dropped).

Music lane frames (``record_start_frame``, ``source_in_frame``,
``duration_frames``) are sequence-rate frames: a recording has no frame rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Mapping

from .errors import ExportError
from .resolver import MusicInfo, SoundInfo, SourceInfo, SourceResolver
from . import colour
from .timebase import is_drop_rate, rate_of

SCHEMA = "studio.timeline/1"
SCHEMAS = (SCHEMA, "studio.timeline/1.1")
MAX_ITEMS = 2000
FITS = ("fit", "fill", "stretch")
SAMPLINGS = ("nearest", "blend", "flow", "ai")  # ai: RIFE in-betweens through the local ComfyUI at export (lane w6-tempo)
TRANSITION_KINDS = ("dissolve", "dip_black", "dip_white")


def speed_curve():
    """dashboard/timeline/speed_curve.py, the one record -> source mapping (lane w6-tempo). Loaded by
    path: this package is also loaded on its own (tests, the export route), outside ``dashboard``."""
    import importlib.util
    import sys
    from pathlib import Path
    name = "film_suite_speed_curve"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "timeline" / "speed_curve.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@dataclass(frozen=True)
class PlanRetime:
    """Speed of a picture clip: constant (``speed == 0``: a freeze of ``source_in``) or a speed curve
    (``curve``: keys ``{frame, percent, ease}``, lane w6-tempo; ``speed`` is then 1)."""

    speed: Fraction
    reverse: bool
    sampling: str
    curve: tuple = ()

    @property
    def freeze(self) -> bool:
        return self.speed == 0 and not self.curve

    @property
    def sound_follows(self) -> bool:
        """One speed of 50 .. 200 % forward: the clip's own sound plays along, pitch-corrected (atempo)."""
        return not self.curve and not self.reverse and Fraction(1, 2) <= self.speed <= 2

    @property
    def tempo(self) -> Fraction:
        return Fraction(1) if self.curve else self.speed


@dataclass(frozen=True)
class PlanClip:
    id: str
    kind: str  # "clip" | "gap"
    track: str
    record_start: int
    duration: int
    source: Mapping[str, Any] | None = None
    info: SourceInfo | None = None
    source_in: int = 0
    source_rate: Fraction | None = None
    audio: bool = False
    enabled: bool = True
    label: str = ""
    retime: PlanRetime | None = None

    @property
    def record_end(self) -> int:
        return self.record_start + self.duration

    @property
    def is_still(self) -> bool:
        return self.info is not None and self.info.is_still

    def step(self, rate: Fraction) -> Fraction:
        """Source frames per record frame: source_rate * speed / rate."""
        speed = Fraction(1) if self.retime is None else self.retime.speed
        return (self.source_rate or rate) * speed / rate

    def keys(self) -> list[dict[str, Any]] | None:
        return [dict(k) for k in self.retime.curve] if self.retime is not None and self.retime.curve else None

    def source_frames_used(self, rate: Fraction, duration: int | None = None) -> int:
        """Source frames the clip spans: ceil(duration * source_rate * speed / rate); a freeze reads one."""
        if self.source_rate is None:
            return 0
        if self.retime is not None and self.retime.freeze:
            return 1
        if self.keys():
            sc = speed_curve()
            return sc.source_frames(self.keys(), self.source_rate / rate, self.duration if duration is None else duration)
        exact = Fraction(self.duration if duration is None else duration) * self.step(rate)
        return -(-exact.numerator // exact.denominator)

    def source_frame_at(self, offset: int, rate: Fraction) -> int:
        """The source frame shown ``offset`` record frames into the clip (negative: in its head handle).
        Forward: ``source_in + floor(offset * step)``; reversed: counted down from the clip's last
        source frame; a freeze always shows ``source_in``."""
        if self.retime is not None and self.retime.freeze:
            return self.source_in
        keys = self.keys()
        if keys:
            sc = speed_curve()
            q = self.source_rate / rate
            # Handles (offsets outside the clip) run at the edge keys' speeds.
            if offset < 0:
                moved = Fraction(offset) * q * keys[0]["percent"] / 100
            else:
                moved = sc.position(keys, q, offset)
        else:
            moved = Fraction(offset) * self.step(rate)
        index = moved.numerator // moved.denominator
        if self.retime is not None and self.retime.reverse:
            return self.source_in + self.source_frames_used(rate) - 1 - index
        return self.source_in + index


@dataclass(frozen=True)
class PlanTransition:
    """A V1 transition on the cut between ``left`` and ``right`` at record frame ``cut``:
    ``before`` frames of it lie before the cut, ``after`` frames after."""

    id: str
    kind: str
    left: str
    right: str
    cut: int
    before: int
    after: int

    @property
    def duration(self) -> int:
        return self.before + self.after

    @property
    def start(self) -> int:
        return self.cut - self.before


@dataclass(frozen=True)
class PlanMusic:
    record_start: int
    source_in: int
    duration: int
    gain_db: float
    track_sha256: str
    map_id: str
    info: MusicInfo


@dataclass(frozen=True)
class PlanSound:
    """One sound clip on A1-A3. Frames: record and duration at the sequence rate, source_in at source_rate."""

    id: str
    track: str
    role: str
    record_start: int
    duration: int
    source: Mapping[str, Any]
    info: SoundInfo
    source_in: int
    source_rate: Fraction
    gain_db: float  # clip gain + track gain, as the mix applies it
    clip_gain_db: float
    fade_in: int
    fade_out: int
    label: str = ""
    # Volume automation (lane fs-duck): (source_frame, dB added to gain_db), rising; () = none.
    volume_keys: tuple[tuple[int, float], ...] = ()

    @property
    def record_end(self) -> int:
        return self.record_start + self.duration


@dataclass(frozen=True)
class PlanSoundTrack:
    id: str
    role: str
    gain_db: float
    audible: bool  # not muted and, when any track is solo, solo itself
    clips: tuple[PlanSound, ...]


@dataclass(frozen=True)
class PlanMarker:
    id: str
    frame: int
    duration: int
    name: str
    note: str


@dataclass
class Plan:
    timeline_id: str
    name: str
    rate: Fraction
    width: int
    height: int
    fit: str
    start_tc_frame: int
    storyline: list[PlanClip]
    overlays: list[PlanClip]
    overlay_enabled: bool
    music: PlanMusic | None
    markers: list[PlanMarker]
    sound_tracks: list[PlanSoundTrack] = field(default_factory=list)
    transitions: list[PlanTransition] = field(default_factory=list)
    # A sound track is solo: only solo tracks sound (the song lane and clip sound are left out).
    sound_solo: bool = False
    # Drop-frame timecode (SMPTE 12M): only at 29.97 / 59.94, default on there (as Resolve).
    drop_frame: bool = False
    # Colour space of the cut, from its sources (colour.dominant); FCPXML and the film are tagged with it.
    color: colour.ColourSpace = colour.REC709
    project_id: str = ""
    revision: int | None = None
    warnings: list[str] = field(default_factory=list)
    # Lane w23-bins (CUT-180): the names of the cuts nested in this one (their clips are in the storyline/overlays).
    nests: list[str] = field(default_factory=list)
    # The outer nested clip ids -> the nested cut's name; a piece's id is "<nested clip id>~<inner id>".
    nest_ids: dict[str, str] = field(default_factory=dict)

    @property
    def total_frames(self) -> int:
        return sum(item.duration for item in self.storyline)

    @property
    def storyline_clips(self) -> list[PlanClip]:
        return [item for item in self.storyline if item.kind == "clip"]

    @property
    def sounds(self) -> list[PlanSound]:
        """Every audible sound clip of A1-A3 (mute / solo applied), in track order."""
        return [clip for track in self.sound_tracks if track.audible for clip in track.clips]


def _invalid(message: str, details: Any = None) -> ExportError:
    return ExportError("invalid_timeline", message, details)


def _frame(value: Any, where: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise _invalid(f"{where} must be a whole number >= {minimum}.")
    return value


def _text(value: Any, limit: int = 400) -> str:
    return value[:limit] if isinstance(value, str) else ""


def load_timeline(document: Mapping[str, Any], timeline_id: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(envelope, timeline)``; accepts the envelope or its ``state``."""
    if not isinstance(document, Mapping):
        raise _invalid("The timeline document must be an object.")
    envelope = dict(document)
    if "schema" in envelope and envelope.get("schema") not in SCHEMAS:
        raise _invalid(f"The timeline document schema must be {SCHEMA} or {SCHEMAS[1]}.")
    state = envelope.get("state") if "state" in envelope else envelope
    if not isinstance(state, Mapping) or not isinstance(state.get("timelines"), list) or not state["timelines"]:
        raise _invalid("The timeline document has no timelines.")
    wanted = timeline_id or state.get("active")
    for timeline in state["timelines"]:
        if isinstance(timeline, Mapping) and timeline.get("id") == wanted:
            return envelope, dict(timeline)
    raise ExportError("timeline_not_found", "The requested timeline is not in the document.")


def _clip(raw: Any, track: str, record_start: int, rate: Fraction, resolver: SourceResolver, offline: list[str]) -> PlanClip:
    if not isinstance(raw, Mapping):
        raise _invalid(f"Track {track} has an item that is not an object.")
    item_id = raw.get("id")
    if not isinstance(item_id, str) or not item_id:
        raise _invalid(f"Track {track} has an item without an id.")
    duration = _frame(raw.get("duration_frames"), f"{item_id}.duration_frames", minimum=1)
    if raw.get("type") == "gap":
        return PlanClip(item_id, "gap", track, record_start, duration, label=_text(raw.get("label")))
    if raw.get("type") != "clip":
        raise _invalid(f"{item_id} must be a clip or a gap.")
    source = raw.get("source")
    if not isinstance(source, Mapping) or not isinstance(source.get("ref"), str) or not source["ref"]:
        raise _invalid(f"{item_id} has no source reference.")
    if source.get("kind") not in {"take", "picture", "media"}:  # a nested cut ("cut") is flattened before
        raise _invalid(f"{item_id}.source.kind must be take, picture or media.")
    audio = raw.get("audio", "off")
    if audio not in {"on", "off"}:
        raise _invalid(f"{item_id}.audio must be on or off.")
    enabled = raw.get("enabled", True)
    if type(enabled) is not bool:
        raise _invalid(f"{item_id}.enabled must be true or false.")
    info = resolver.resolve(source)
    if info is None:
        offline.append(item_id)
        return PlanClip(item_id, "clip", track, record_start, duration, source=dict(source))
    source_in = _frame(raw.get("source_in_frame", 0), f"{item_id}.source_in_frame")
    source_rate: Fraction | None = None
    if not info.is_still:
        source_rate = rate_of(raw.get("source_rate") or info.rate, f"{item_id}.source_rate")
        if info.rate is not None and source_rate != info.rate:
            raise _invalid(f"{item_id}.source_rate does not match the source file's measured rate.")
    retime = _retime(raw.get("retime"), item_id) if source_rate is not None else None
    clip = PlanClip(
        item_id, "clip", track, record_start, duration, source=dict(source), info=info,
        source_in=source_in if source_rate is not None else 0, source_rate=source_rate,
        audio=audio == "on" and info.has_audio and (retime is None or retime.sound_follows), enabled=enabled,
        label=_text(raw.get("label")),
        retime=retime,
    )
    if info.frames is not None and source_in + clip.source_frames_used(rate) > info.frames:
        raise _invalid(
            f"{item_id} reads past the end of its source "
            f"(in {source_in} + {clip.source_frames_used(rate)} frames > {info.frames}).",
        )
    return clip


def _retime(raw: Any, item_id: str) -> PlanRetime | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or not isinstance(raw.get("speed"), Mapping):
        raise _invalid(f"{item_id}.retime must be {{speed, reverse, sampling}}.")
    num, den = raw["speed"].get("num"), raw["speed"].get("den")
    if type(num) is not int or type(den) is not int or num < 0 or den < 1 or num > 100 * den:
        raise _invalid(f"{item_id}.retime.speed must be a ratio of 0 .. 100.")
    if raw.get("sampling") not in SAMPLINGS or type(raw.get("reverse")) is not bool:
        raise _invalid(f"{item_id}.retime needs reverse true|false and sampling nearest|blend|flow|ai.")
    curve = raw.get("curve")
    if curve is not None:
        why = speed_curve().check_keys(curve, f"{item_id}.retime.curve")
        if why:
            raise _invalid(why + ".")
        return PlanRetime(Fraction(1), raw["reverse"], raw["sampling"], tuple(dict(k) for k in curve))
    return PlanRetime(Fraction(num, den), raw["reverse"] and num > 0, raw["sampling"])


def _plan_transitions(timeline: Mapping[str, Any], storyline: list[PlanClip], rate: Fraction) -> list[PlanTransition]:
    """Transitions on V1 cuts with the handles they need checked against the resolved media."""
    raw = timeline.get("transitions") or []
    if not isinstance(raw, list):
        raise _invalid("transitions must be a list.")
    index = {clip.id: n for n, clip in enumerate(storyline)}
    out: list[PlanTransition] = []
    for item in raw:
        if not isinstance(item, Mapping) or item.get("kind") not in TRANSITION_KINDS:
            raise _invalid("Every transition needs a kind dissolve | dip_black | dip_white.")
        left, right = item.get("left"), item.get("right")
        if left not in index or right not in index or index[right] != index[left] + 1:
            raise _invalid(f"Transition {item.get('id')} must join two adjacent storyline clips.")
        a, b = storyline[index[left]], storyline[index[right]]
        if a.kind != "clip" or b.kind != "clip":
            raise _invalid(f"Transition {item.get('id')} must join two clips.")
        duration = _frame(item.get("duration_frames"), f"transition {item.get('id')}.duration_frames", minimum=1)
        alignment = item.get("alignment", "center")
        before, after = {"start": (0, duration), "end": (duration, 0)}.get(alignment, (duration // 2, duration - duration // 2))
        for clip, frames, side in ((a, after, "after its out point"), (b, before, "before its in point")):
            if not frames or clip.is_still or clip.info is None or clip.info.frames is None:
                continue
            if clip.retime is not None and clip.retime.freeze:
                continue
            first = clip.source_frame_at(-frames if clip is b else clip.duration + frames - 1, rate)
            if not 0 <= first < clip.info.frames:
                raise ExportError("no_handles", f"{clip.label or clip.id} has no media {frames} frames {side}; the transition {item.get('id')} needs it.")
        out.append(PlanTransition(str(item.get("id")), item["kind"], a.id, b.id, b.record_start, before, after))
    return out


def offline_message(picture_and_music: list[str], sound: list[str], music: list[str]) -> str:
    """Lane p10-vid (CUT-113): the offline refusal says what Deliver's check row says ("Offline: 1 picture clip.",
    export_check.py), so Hermes and the button ("Add to render queue: 1 picture clip offline") name the same count:
    picture clips, sound clips and the song lane apart."""
    counts = {"picture": len([ident for ident in picture_and_music if ident not in music]), "sound": len(sound),
              "music": len(music)}
    parts = [f"{n} {kind}" + ("" if kind == "music" else " clip" + ("s" if n > 1 else "")) for kind, n in counts.items() if n]
    return "Offline: " + ", ".join(parts) + "."


TAIL_ID = "black-tail"


def timeline_end(timeline: Mapping[str, Any]) -> int:
    """w22-dlv (CUT-178): the ONE render extent. Like Resolve's Deliver "Entire timeline" and Cut's Length
    (timeline-model.js: the end of the last item on any track), the end of the last item on any track: the storyline
    (gaps included), each overlay clip, each sound clip and the song lane, muted or disabled ones too (they keep their
    place on the timeline); markers are not items. The film, its sound, the range check and every interchange writer
    use this end; where the picture stops earlier the storyline continues in black (build_plan's tail gap)."""
    end = 0
    for track in timeline.get("tracks") or ():
        if not isinstance(track, Mapping):
            continue
        kind = track.get("kind")
        if kind == "music":
            music = track.get("music")
            if isinstance(music, Mapping) and type(music.get("duration_frames")) is int:
                end = max(end, int(music.get("record_start_frame") or 0) + music["duration_frames"])
            continue
        if kind == "video" and track.get("role") == "storyline":
            end = max(end, sum(i["duration_frames"] for i in track.get("items") or ()
                               if isinstance(i, Mapping) and type(i.get("duration_frames")) is int))
            continue
        for item in track.get("items") or ():
            if (isinstance(item, Mapping) and item.get("type") == "clip" and type(item.get("record_start_frame")) is int
                    and type(item.get("duration_frames")) is int):
                end = max(end, item["record_start_frame"] + item["duration_frames"])
    return end


def build_plan(document: Mapping[str, Any], resolver: SourceResolver, *, timeline_id: str | None = None) -> Plan:
    # Lane w23-bins (CUT-180): a nested cut is exported as what it shows (export_nest.py), like Resolve renders it.
    from .export_nest import flatten_document
    document, nest_warnings = flatten_document(document, timeline_id)
    nests = list(document.pop("__nests", []))
    nest_ids = dict(document.pop("__nest_ids", {}))
    envelope, timeline = load_timeline(document, timeline_id)
    rate = rate_of(timeline.get("rate"), "rate")
    fmt = timeline.get("format")
    if not isinstance(fmt, Mapping):
        raise _invalid("The timeline has no format.")
    width = _frame(fmt.get("width"), "format.width", minimum=16)
    height = _frame(fmt.get("height"), "format.height", minimum=16)
    if width % 2 or height % 2 or width > 8192 or height > 8192:
        raise _invalid("The timeline format must be even and at most 8192 pixels.")
    fit = fmt.get("fit", "fit")
    if fit not in FITS:
        raise _invalid("The timeline format fit must be fit, fill or stretch.")
    start_tc = _frame(timeline.get("start_tc_frame", 0), "start_tc_frame")
    tc_format = timeline.get("tc_format", "DF")
    if tc_format not in ("DF", "NDF"):
        raise _invalid("The timeline tc_format must be DF or NDF.")
    drop_frame = tc_format == "DF" and is_drop_rate(rate)
    tracks = timeline.get("tracks")
    if not isinstance(tracks, list):
        raise _invalid("The timeline has no tracks.")
    seen_ids: set[str] = set()
    offline: list[str] = []
    music_offline: list[str] = []
    storyline: list[PlanClip] = []
    overlays: list[PlanClip] = []
    overlay_enabled = True
    music: PlanMusic | None = None
    storyline_found = False
    raw_sound_tracks: list[Mapping[str, Any]] = []
    for track in tracks:
        if not isinstance(track, Mapping) or not isinstance(track.get("id"), str):
            raise _invalid("Every track needs an id.")
        track_id = track["id"]
        kind = track.get("kind")
        if kind == "video":
            items = track.get("items")
            if not isinstance(items, list) or len(items) > MAX_ITEMS:
                raise _invalid(f"Track {track_id} items must be a list of at most {MAX_ITEMS}.")
            role = track.get("role")
            if role == "storyline":
                if storyline_found:
                    raise _invalid("The timeline has more than one storyline.")
                storyline_found = True
                if track.get("enabled", True) is not True:
                    raise _invalid("The storyline track cannot be disabled for export.")
                cursor = 0
                for raw in items:
                    clip = _clip(raw, track_id, cursor, rate, resolver, offline)
                    storyline.append(clip)
                    cursor = clip.record_end
            elif role == "overlay":
                if overlays:
                    raise _invalid("v1 exports one overlay track.")
                overlay_enabled = track.get("enabled", True) is True
                for raw in items:
                    start = _frame(raw.get("record_start_frame") if isinstance(raw, Mapping) else None, "overlay record_start_frame")
                    if isinstance(raw, Mapping) and raw.get("type") == "gap":
                        raise _invalid("Gaps on the overlay track stay implicit.")
                    overlays.append(_clip(raw, track_id, start, rate, resolver, offline))
            else:
                raise _invalid(f"Track {track_id} role must be storyline or overlay.")
        elif kind == "music":
            if music is not None:
                raise _invalid("v1 exports one music lane.")
            raw = track.get("music")
            if raw is None or track.get("enabled", True) is not True:
                # The empty lane of every new cut (music: null), or a muted lane (the header's mute
                # toggle = enabled:false): no song in the film or the interchange.
                continue
            if not isinstance(raw, Mapping):
                raise _invalid(f"Music track {track_id} has no music.")
            info = resolver.resolve_music(raw)
            if info is None:
                offline.append(track_id)
                music_offline.append(track_id)
                continue
            gain = raw.get("gain_db", 0)
            if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not -60 <= gain <= 24:
                raise _invalid("Music gain_db must be between -60 and 24.")
            source_in = _frame(raw.get("source_in_frame", 0), "music.source_in_frame")
            duration = _frame(raw.get("duration_frames"), "music.duration_frames", minimum=1)
            if Fraction(source_in + duration) / rate > info.duration:
                raise _invalid("The music lane reads past the end of the recording.")
            music = PlanMusic(
                _frame(raw.get("record_start_frame", 0), "music.record_start_frame"), source_in, duration,
                float(gain), str(raw.get("track_sha256") or ""), str(raw.get("map_id") or ""), info,
            )
        elif kind == "audio":
            raw_sound_tracks.append(track)
        else:
            raise _invalid(f"Track {track_id} kind must be video, music or audio.")
    offline_before_sound = len(offline)
    sound_tracks = _sound_tracks(raw_sound_tracks, rate, resolver, offline)
    for item in [*storyline, *overlays, *(clip for track in sound_tracks for clip in track.clips)]:
        if item.id in seen_ids:
            raise _invalid(f"Item id {item.id} is not unique.")
        seen_ids.add(item.id)
    if not storyline_found or not storyline:
        raise _invalid("The storyline is empty.")
    transitions = [] if offline else _plan_transitions(timeline, storyline, rate)
    if offline:
        raise ExportError("source_offline", offline_message(offline[:offline_before_sound], offline[offline_before_sound:],
                                                            music_offline), {"items": offline})
    # w22-dlv (CUT-178): the storyline runs to the timeline's end; the picture's missing frames are black (one gap).
    picture_end = sum(item.duration for item in storyline)
    end = timeline_end(timeline)
    if end > picture_end:
        if TAIL_ID in seen_ids:
            raise _invalid(f"Item id {TAIL_ID} is reserved for the black tail.")
        storyline.append(PlanClip(TAIL_ID, "gap", storyline[0].track, picture_end, end - picture_end, label="Black"))
    total = sum(item.duration for item in storyline)
    ordered = sorted(overlays, key=lambda clip: clip.record_start)
    for before, after in zip(ordered, ordered[1:]):
        if after.record_start < before.record_end:
            raise _invalid(f"Overlay clips {before.id} and {after.id} overlap.")
    plan_warnings: list[str] = list(nest_warnings)
    for clip in ordered:
        if clip.record_start >= total:
            raise _invalid(f"Overlay clip {clip.id} starts after the storyline ends.")
        if clip.record_end > total:
            plan_warnings.append(f"Overlay clip {clip.id} runs past the storyline end; the film cuts it at the end.")
    for track in sound_tracks:
        for sound in track.clips:
            if sound.record_start >= total:
                plan_warnings.append(f"Sound clip {sound.id} on {track.id} starts after the storyline ends and is not in the film.")
            elif sound.record_end > total:
                plan_warnings.append(f"Sound clip {sound.id} on {track.id} runs past the storyline end; the film cuts it at the end.")
    if any(track.audible for track in sound_tracks) and any(not track.audible for track in sound_tracks):
        silent = ", ".join(track.id for track in sound_tracks if not track.audible)
        plan_warnings.append(f"Sound tracks {silent} are not in the mix (muted, or another track is solo).")
    if any(_solo(t) for t in raw_sound_tracks):
        plan_warnings.append("A sound track is solo: the mix holds only the solo tracks (no song lane, no clip sound).")
    if music is not None:
        if music.record_start >= total:
            raise _invalid("The music lane starts after the storyline ends.")
        if music.record_start + music.duration > total:
            plan_warnings.append("The music lane runs past the storyline end; the film cuts it at the end.")
    markers: list[PlanMarker] = []
    for raw in timeline.get("markers") or []:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("id"), str):
            raise _invalid("Every marker needs an id.")
        frame = _frame(raw.get("frame"), f"marker {raw['id']}.frame")
        if frame >= total:
            plan_warnings.append(f"Marker {raw['id']} lies after the storyline end and is listed only in markers.csv.")
        markers.append(PlanMarker(raw["id"], frame, _frame(raw.get("duration_frames", 0), "marker duration"), _text(raw.get("name"), 200), _text(raw.get("note"), 2000)))
    revision = envelope.get("revision")
    return Plan(
        timeline_id=str(timeline.get("id")), name=_text(timeline.get("name"), 200) or "Timeline", rate=rate,
        width=width, height=height, fit=fit, start_tc_frame=start_tc, drop_frame=drop_frame,
        color=colour.dominant([c.info for c in [*storyline, *ordered] if c.kind == "clip"]), storyline=storyline, overlays=ordered,
        overlay_enabled=overlay_enabled, music=music, markers=sorted(markers, key=lambda m: (m.frame, m.id)),
        sound_tracks=sound_tracks, sound_solo=any(_solo(t) for t in raw_sound_tracks), transitions=transitions,
        project_id=str(envelope.get("project_id") or ""), revision=revision if type(revision) is int else None,
        warnings=plan_warnings, nests=nests, nest_ids=nest_ids,
    )


AUDIO_ROLES = ("dialogue", "music", "effects")


def _solo(track: Mapping[str, Any]) -> bool:
    return track.get("solo") is True


def _gain(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not -60 <= value <= 24:
        raise _invalid(f"{where} must be a gain between -60 and 24 dB.")
    return float(value)


def _volume_keys(value: Any, item_id: str) -> tuple[tuple[int, float], ...]:
    """A sound clip's volume keys as (source_frame, dB), strictly rising; absent = ()."""
    if value is None:
        return ()
    if not isinstance(value, list) or not value:
        raise _invalid(f"{item_id}.volume_keys must be a non-empty list.")
    out: list[tuple[int, float]] = []
    for key in value:
        if not isinstance(key, Mapping):
            raise _invalid(f"{item_id}.volume_keys holds objects only.")
        frame = _frame(key.get("source_frame"), f"{item_id}.volume_keys.source_frame")
        if out and frame <= out[-1][0]:
            raise _invalid(f"{item_id}.volume_keys must rise in source frames.")
        out.append((frame, _gain(key.get("gain_db"), f"{item_id}.volume_keys.gain_db")))
    return tuple(out)


def _sound_tracks(tracks: list[Mapping[str, Any]], rate: Fraction, resolver: SourceResolver, offline: list[str]) -> list[PlanSoundTrack]:
    """A1-A3 (``kind: audio``): every clip resolved to its sound file; mute / solo decide audibility."""
    if len(tracks) > 3:
        raise _invalid("A cut exports at most three sound tracks (A1-A3).")
    if not tracks:
        return []
    resolve_sound = getattr(resolver, "resolve_sound", None)
    any_solo = any(_solo(track) for track in tracks)
    out: list[PlanSoundTrack] = []
    for track in tracks:
        track_id = track["id"]
        role = track.get("role")
        if role not in AUDIO_ROLES:
            raise _invalid(f"Sound track {track_id} role must be dialogue, music or effects.")
        track_gain = _gain(track.get("gain_db", 0), f"{track_id}.gain_db")
        items = track.get("items")
        if not isinstance(items, list) or len(items) > MAX_ITEMS:
            raise _invalid(f"Track {track_id} items must be a list of at most {MAX_ITEMS}.")
        clips: list[PlanSound] = []
        for raw in items:
            if not isinstance(raw, Mapping) or raw.get("type") != "clip" or not isinstance(raw.get("id"), str):
                raise _invalid(f"Sound track {track_id} holds clips only.")
            item_id = raw["id"]
            if raw.get("enabled", True) is not True:
                continue
            source = raw.get("source")
            if not isinstance(source, Mapping) or not isinstance(source.get("ref"), str) or not source["ref"]:
                raise _invalid(f"{item_id} has no source reference.")
            duration = _frame(raw.get("duration_frames"), f"{item_id}.duration_frames", minimum=1)
            start = _frame(raw.get("record_start_frame"), f"{item_id}.record_start_frame")
            source_in = _frame(raw.get("source_in_frame", 0), f"{item_id}.source_in_frame")
            source_rate = rate_of(raw.get("source_rate") or {"num": rate.numerator, "den": rate.denominator}, f"{item_id}.source_rate")
            fade_in = _frame(raw.get("fade_in_frames", 0), f"{item_id}.fade_in_frames")
            fade_out = _frame(raw.get("fade_out_frames", 0), f"{item_id}.fade_out_frames")
            if fade_in + fade_out > duration:
                raise _invalid(f"{item_id} fades are longer than the clip.")
            clip_gain = _gain(raw.get("gain_db", 0), f"{item_id}.gain_db")
            info = resolve_sound(source) if resolve_sound is not None else None
            if info is None:
                offline.append(item_id)
                continue
            if info.samples is not None:
                needed = Fraction(source_in) / source_rate + Fraction(duration) / rate
                if needed * info.sample_rate > info.samples:
                    raise _invalid(f"{item_id} reads past the end of its sound file.")
            clips.append(PlanSound(item_id, track_id, role, start, duration, dict(source), info, source_in, source_rate,
                                   clip_gain + track_gain, clip_gain, fade_in, fade_out, _text(raw.get("label")),
                                   _volume_keys(raw.get("volume_keys"), item_id)))
        clips.sort(key=lambda clip: clip.record_start)
        for before, after in zip(clips, clips[1:]):
            if after.record_start < before.record_end:
                raise _invalid(f"Sound clips {before.id} and {after.id} overlap on {track_id}.")
        audible = track.get("mute") is not True and (not any_solo or _solo(track))
        out.append(PlanSoundTrack(track_id, role, track_gain, audible, tuple(clips)))
    return out
