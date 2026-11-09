"""``studio.timeline/1``: exact document shape and the section 2.4 invariants.

The document is the ``state`` of a project envelope ``{schema, project_id,
revision, updated_at, state}`` stored as ``state/timeline.json``. Time is in
integer frames: record times at the timeline's rational rate, source times in
the source's own frames (``source_in_frame`` at ``source_rate``). Seconds never
appear here; a musical anchor keeps the frame it was set on plus the residual
between that frame and the exact event position (millionths of a frame).

``validate_state`` checks the exact shape (unknown keys are refused, so a
typo never silently persists) and the structural invariants: unique ids,
integer frames >= 0, durations >= 1, source ranges inside moving media, a
contiguous storyline of clips and explicit gaps, no overlap on overlay tracks.
Whether a ``source.ref`` still resolves is reported by the resolver as
``offline``; an unresolvable clip is never dropped.

Sound tracks (``kind: "audio"``, at most ``MAX_AUDIO_TRACKS`` = A1..A3): a track has a
``role`` (dialogue | music | effects), ``gain_db``, ``mute``, ``solo`` and ``locked``; its
items are sound clips at their own record frames (no gaps, no overlap, like V2):
``source`` is a take's own sound or a file (``take`` | ``media``), ``source_in_frame`` at
``source_rate`` (a sound file is read at the timeline rate), ``media_info`` gives the
source length in those frames (``sample_rate`` / ``channels`` when probed), ``gain_db``
in 0.1 dB steps (-60..+24), ``fade_in_frames + fade_out_frames <= duration_frames``.
``attach`` (follows a picture clip through ripple) or ``link`` (a take's own sound: also
cut, split and removed with its picture) names a picture clip. The song lane (``music``)
is unchanged; a document without sound tracks reads and exports exactly as before.

Volume automation (lane fs-duck, optional, absent = none): a sound clip's ``volume_keys`` are
``[{source_frame, gain_db, auto}]`` strictly rising in its own source frames (0..media frames,
at most ``MAX_VOLUME_KEYS``), ``gain_db`` in 0.1 dB steps added to clip + track gain, linear in dB
between keys and held outside them; ``auto`` marks keys ``auto_duck`` made. A sound track's
``duck`` keeps the parameters of its last ``auto_duck`` run (ops_volume.py).

``studio.timeline/1.1`` (additive; a document that uses none of it stays ``studio.timeline/1``
byte for byte, see ``schema_of``):

- ``format.fit`` may also be ``stretch`` (fit = letterbox, fill = crop, stretch = distort);
- a picture clip may carry ``retime {speed {num, den}, reverse, sampling}``: constant speed as an
  exact ratio (1/2 = 50 %), ``speed.num == 0`` is a freeze (the clip holds ``source_in_frame``),
  ``reverse`` plays the source range backwards, ``sampling`` (nearest | blend | flow) is how frames
  between source frames are made for slow speeds. A retimed clip reads
  ``ceil(duration * speed * source_rate / rate)`` source frames (a freeze reads one);
- (lane w6-tempo) ``retime.curve`` (optional, absent = constant speed): speed keys
  ``[{frame, percent, ease}]`` in the clip's own record frames (``speed_curve.py`` has the one
  mapping); ``speed`` is then ``1/1`` (the curve is the speed). A hold is a key pair at 0 %. The
  clip reads ``ceil(P(duration))`` source frames; every key lies inside the clip;
- ``transitions`` holds cross dissolves and dips to black / white on a cut between two adjacent
  storyline clips: ``{id, kind, left, right, duration_frames, alignment}``, ``alignment`` centre |
  start | end splits the duration into the part before the cut (the left clip plays on into its
  handle) and after it (the right clip starts early, from its handle). The handles must exist.
"""
from __future__ import annotations

import copy
import re
from typing import Any, Callable

SCHEMA = "studio.timeline/1"
SCHEMA_1_1 = "studio.timeline/1.1"
SCHEMAS = (SCHEMA, SCHEMA_1_1)
FITS = ("fit", "fill", "stretch")
SAMPLINGS = ("nearest", "blend", "flow", "ai")  # ai: RIFE in-betweens through the local ComfyUI at export (lane w6-tempo)
TRANSITION_KINDS = ("dissolve", "dip_black", "dip_white")
ALIGNMENTS = ("center", "start", "end")
MAX_SPEED = 100  # 10 000 %
MAX_TRANSITIONS = 1000
MAX_TIMELINES = 32
MAX_TRACKS = 16
MAX_ITEMS = 1000
MAX_MARKERS = 1000
MAX_SLOTS = 1000
MAX_FRAME = 2**40  # ~1450 years at 24 fps; keeps JS integer math exact
MAX_RATE = 1_000_000
MAX_AUDIO_TRACKS = 3
AUDIO_ROLES = ("dialogue", "music", "effects")
GAIN_MIN_DB, GAIN_MAX_DB = -60, 24
MAX_VOLUME_KEYS = 2000
DUCK_KEYS = {"key_role", "depth_db", "attack_ms", "release_ms", "lookahead_ms", "hold_ms"}

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


class TimelineError(ValueError):
    """A finite, public-safe refusal: ``code`` is stable, ``message`` names the field."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.reason_code = code
        super().__init__(message)


def fail(code: str, message: str) -> None:
    raise TimelineError(code, message)


def _invalid(path: str, what: str) -> None:
    fail("invalid_timeline", f"{path}: {what}")


def _exact(value: Any, path: str, required: set[str], optional: set[str] = frozenset()) -> dict[str, Any]:
    if type(value) is not dict:
        _invalid(path, "must be an object")
    keys = set(value)
    missing = required - keys
    extra = keys - required - optional
    if missing:
        _invalid(path, "missing " + ", ".join(sorted(missing)))
    if extra:
        _invalid(path, "unknown " + ", ".join(sorted(extra)))
    return value


def _int(value: Any, path: str, *, minimum: int = 0, maximum: int = MAX_FRAME) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _invalid(path, f"must be an integer in {minimum}..{maximum}")
    return value


def _bool(value: Any, path: str) -> bool:
    if type(value) is not bool:
        _invalid(path, "must be true or false")
    return value


def _text(value: Any, path: str, *, maximum: int, minimum: int = 0) -> str:
    if type(value) is not str or not minimum <= len(value) <= maximum or _CONTROL_RE.search(value.replace("\n", "")):
        _invalid(path, f"must be text of {minimum}..{maximum} characters")
    return value


def _ident(value: Any, path: str) -> str:
    if type(value) is not str or _ID_RE.fullmatch(value) is None:
        _invalid(path, "must be an id (letters, digits, _ . : -; at most 64)")
    return value


def _sha(value: Any, path: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        _invalid(path, "must be a lowercase sha256")
    return value


def _enum(value: Any, path: str, options: tuple[str, ...]) -> str:
    if value not in options or type(value) is not str:
        _invalid(path, "must be one of " + "|".join(options))
    return value


def _rate(value: Any, path: str) -> dict[str, int]:
    rate = _exact(value, path, {"num", "den"})
    _int(rate["num"], path + ".num", minimum=1, maximum=MAX_RATE)
    _int(rate["den"], path + ".den", minimum=1, maximum=MAX_RATE)
    return rate


def used_source_frames(duration: int, source_rate: dict[str, int], rate: dict[str, int]) -> int:
    """Source frames a clip of ``duration`` record frames consumes (rounded up)."""
    numerator = duration * source_rate["num"] * rate["den"]
    denominator = source_rate["den"] * rate["num"]
    return -(-numerator // denominator)


def _curve_q(item: dict[str, Any], rate: dict[str, int]):
    from .speed_curve import keys_of, q_of
    keys = keys_of(item)
    return (keys, q_of(item["source_rate"], rate)) if keys else (None, None)


def speed_of(item: dict[str, Any]) -> dict[str, int] | None:
    retime = item.get("retime")
    return None if retime is None else retime["speed"]


def frozen(item: dict[str, Any]) -> bool:
    speed = speed_of(item)
    return speed is not None and speed["num"] == 0


def reversed_clip(item: dict[str, Any]) -> bool:
    retime = item.get("retime")
    return retime is not None and retime["reverse"] is True


def sound_follows(item: dict[str, Any]) -> bool:
    """True when a retimed clip's own sound plays retimed (pitch-corrected): one speed of 50 .. 200 %
    forward. A curve, a hold, a freeze or reverse keeps it off (lane w6-tempo, decision d)."""
    retime = item.get("retime")
    if retime is None:
        return True
    speed = retime["speed"]
    return (not retime.get("curve") and not retime["reverse"]
            and speed["den"] <= 2 * speed["num"] and speed["num"] <= 2 * speed["den"])


def effective_rate(item: dict[str, Any]) -> dict[str, int]:
    """Source frames per second of record time: the source rate times the clip's speed."""
    rate, speed = item["source_rate"], speed_of(item)
    if speed is None:
        return rate
    return {"num": rate["num"] * speed["num"], "den": rate["den"] * speed["den"]}


def clip_source_frames(item: dict[str, Any], rate: dict[str, int]) -> int:
    """Source frames a picture clip reads: a freeze reads one, a retimed clip duration * speed."""
    if frozen(item):
        return 1
    keys, q = _curve_q(item, rate)
    if keys:
        from .speed_curve import source_frames
        return source_frames(keys, q, item["duration_frames"])
    return used_source_frames(item["duration_frames"], effective_rate(item), rate)


def _rec_floor(frames: int, eff: dict[str, int], rate: dict[str, int]) -> int:
    return frames * eff["den"] * rate["num"] // (eff["num"] * rate["den"])


def handles(item: dict[str, Any], rate: dict[str, int]) -> tuple[int, int]:
    """Record frames of media before the clip's first frame and after its last (``MAX_FRAME`` for
    stills, gaps and freezes). A reversed clip's first frame is its last source frame."""
    if item["type"] != "clip" or item["media_info"]["frames"] is None or frozen(item):
        return MAX_FRAME, MAX_FRAME
    keys, q = _curve_q(item, rate)
    if keys:  # a curve's handles run at its edge speeds (a hold at an edge repeats that frame)
        from .speed_curve import position
        used = clip_source_frames(item, rate)
        low, high = item["source_in_frame"], item["media_info"]["frames"] - item["source_in_frame"] - used
        first, last = keys[0]["percent"], keys[-1]["percent"]
        rest = high + used - position(keys, q, item["duration_frames"])
        b = MAX_FRAME if first == 0 else int(low * 100 // (q * first))
        a = int(rest * 100 // (q * last))
        return (a, b) if reversed_clip(item) else (b, a)
    eff = effective_rate(item)
    before = _rec_floor(item["source_in_frame"], eff, rate)
    after = _rec_floor(item["media_info"]["frames"] - item["source_in_frame"], eff, rate) - item["duration_frames"]
    return (after, before) if reversed_clip(item) else (before, after)


def transition_split(transition: dict[str, Any]) -> tuple[int, int]:
    """Frames of a transition before the cut and after it (centre: the odd frame goes after)."""
    duration, alignment = transition["duration_frames"], transition["alignment"]
    if alignment == "start":
        return 0, duration
    if alignment == "end":
        return duration, 0
    return duration // 2, duration - duration // 2


def uses_1_1(state: dict[str, Any]) -> bool:
    """True when the state needs ``studio.timeline/1.1`` (stretch, a retimed clip or a transition)."""
    for timeline in state.get("timelines") or ():
        if timeline.get("transitions") or (timeline.get("format") or {}).get("fit") == "stretch":
            return True
        for track in timeline.get("tracks") or ():
            for item in track.get("items") or ():
                if item.get("retime") is not None or (item.get("source") or {}).get("kind") == "cut":
                    return True
    return False


def schema_of(state: dict[str, Any]) -> str:
    return SCHEMA_1_1 if uses_1_1(state) else SCHEMA


def _retime(value: Any, path: str) -> None:
    retime = _exact(value, path, {"speed", "reverse", "sampling"}, {"curve"})
    speed = _exact(retime["speed"], path + ".speed", {"num", "den"})
    _int(speed["num"], path + ".speed.num", maximum=MAX_RATE)
    _int(speed["den"], path + ".speed.den", minimum=1, maximum=MAX_RATE)
    if speed["num"] > MAX_SPEED * speed["den"] or (speed["num"] and speed["num"] * 100 < speed["den"]):
        _invalid(path + ".speed", "must be a freeze (0) or 1 % .. 10 000 %")
    if speed["num"] == speed["den"] and retime["reverse"] is False and "curve" not in retime:
        _invalid(path, "100 % forward is no retime (use null)")
    if "curve" in retime:
        from .speed_curve import check_keys
        why = check_keys(retime["curve"], path + ".curve")
        if why:
            fail("invalid_timeline", why)
        if speed != {"num": 1, "den": 1}:
            _invalid(path + ".speed", "is 1/1 when a speed curve is set (the curve is the speed)")
    _bool(retime["reverse"], path + ".reverse")
    if speed["num"] == 0 and retime["reverse"]:
        _invalid(path + ".reverse", "a freeze holds one frame; it has no direction")
    _enum(retime["sampling"], path + ".sampling", SAMPLINGS)


def _anchor(value: Any, path: str) -> None:
    if value is None:
        return
    anchor = _exact(value, path, {"map_id", "kind", "index", "offset_frames", "residual_ppm"})
    _sha(anchor["map_id"], path + ".map_id")
    _enum(anchor["kind"], path + ".kind", ("beat", "downbeat", "section"))
    _int(anchor["index"], path + ".index", maximum=1_000_000)
    _int(anchor["offset_frames"], path + ".offset_frames", minimum=-MAX_FRAME)
    _int(anchor["residual_ppm"], path + ".residual_ppm", minimum=-500_000, maximum=500_000)


_CLIP_KEYS = {
    "id", "type", "duration_frames", "source", "source_in_frame", "source_rate",
    "media_info", "audio", "enabled", "label", "note", "anchors",
}


def _clip(item: dict[str, Any], path: str, *, storyline: bool, rate: dict[str, int]) -> None:
    required = set(_CLIP_KEYS) if storyline else _CLIP_KEYS | {"record_start_frame"}
    _exact(item, path, required, {"retime"} if storyline else {"attach", "retime"})
    _int(item["duration_frames"], path + ".duration_frames", minimum=1)
    source = _exact(item["source"], path + ".source", {"kind", "ref"}, {"slot"})
    kind = _enum(source["kind"], path + ".source.kind", ("take", "picture", "media", "cut"))
    _text(source["ref"], path + ".source.ref", minimum=1, maximum=4096)
    if source.get("slot") is not None:
        _ident(source["slot"], path + ".source.slot")
    source_in = _int(item["source_in_frame"], path + ".source_in_frame")
    source_rate = _rate(item["source_rate"], path + ".source_rate")
    info = _exact(item["media_info"], path + ".media_info", {"frames", "width", "height", "has_audio"}, {"probe_sha"})
    frames = info["frames"]
    if frames is not None:
        _int(frames, path + ".media_info.frames", minimum=1)
    for key in ("width", "height"):
        if info[key] is not None:
            _int(info[key], f"{path}.media_info.{key}", minimum=1, maximum=65536)
    _bool(info["has_audio"], path + ".media_info.has_audio")
    if info.get("probe_sha") is not None:
        _text(info["probe_sha"], path + ".media_info.probe_sha", minimum=1, maximum=128)
    if kind == "picture" and frames is not None:
        _invalid(path + ".media_info.frames", "a picture is a still (null)")
    if kind in ("take", "cut") and frames is None:
        _invalid(path + ".media_info.frames", f"a {kind} has a frame count")
    if kind == "cut" and (item.get("retime") is not None or item["audio"] != "off"):
        _invalid(path, "a nested cut keeps its own speed and sound (no retime, audio off)")
    if item.get("retime") is not None:
        _retime(item["retime"], path + ".retime")
        if frames is None:
            _invalid(path + ".retime", "a still has no speed")
    if frames is None:
        if source_in != 0:
            _invalid(path + ".source_in_frame", "a still starts at 0")
    elif source_in + clip_source_frames(item, rate) > frames:
        _invalid(path, "source range runs past the end of the media")
    keys = (item.get("retime") or {}).get("curve")
    if keys and keys[-1]["frame"] >= item["duration_frames"]:
        fail("speed_curve", f"{path}: {item['id']}'s speed key at frame {keys[-1]['frame']} would lie past its last "
             f"frame ({item['duration_frames'] - 1})")
    _enum(item["audio"], path + ".audio", ("off", "on"))
    if item["audio"] == "on" and not sound_follows(item):
        _invalid(path + ".audio", "a retimed clip's own sound plays only at one speed of 50 .. 200 % forward")
    _bool(item["enabled"], path + ".enabled")
    _text(item["label"], path + ".label", maximum=200)
    _text(item["note"], path + ".note", maximum=2000)
    anchors = _exact(item["anchors"], path + ".anchors", {"in", "out"})
    _anchor(anchors["in"], path + ".anchors.in")
    _anchor(anchors["out"], path + ".anchors.out")
    if not storyline:
        _int(item["record_start_frame"], path + ".record_start_frame")
        if item.get("attach") is not None:
            attach = _exact(item["attach"], path + ".attach", {"clip", "offset_frames"})
            _ident(attach["clip"], path + ".attach.clip")
            _int(attach["offset_frames"], path + ".attach.offset_frames", minimum=-MAX_FRAME)


def gain_tenths(value: Any) -> int | None:
    """A gain in dB as whole tenths (``-3.5`` -> ``-35``), or None when it is not a 0.1 dB step.

    Exact: the number's shortest decimal form is read (the same digits in both twins),
    never a float product, so 0.1-dB steps never drift."""
    if type(value) is int:
        return value * 10
    if type(value) is not float or value != value or value in (float("inf"), float("-inf")):
        return None
    match = re.fullmatch(r"(-?)(\d+)(?:\.(\d+))?(?:[eE]([+-]?\d+))?", repr(value))
    if match is None:
        return None
    sign, whole, fraction, exponent = match.groups()
    fraction = fraction or ""
    digits = int(whole + fraction)
    shift = int(exponent or 0) - len(fraction) + 1  # value * 10 = digits * 10**shift
    if shift >= 0:
        tenths = digits * 10 ** shift
    else:
        tenths, rest = divmod(digits, 10 ** -shift)
        if rest:
            return None
    return -tenths if sign else tenths


def gain_value(tenths: int) -> int | float:
    """The canonical JSON number of a gain: an integer when whole, else one decimal."""
    return tenths // 10 if tenths % 10 == 0 else tenths / 10


def _gain(value: Any, path: str) -> None:
    tenths = gain_tenths(value)
    if tenths is None or not GAIN_MIN_DB * 10 <= tenths <= GAIN_MAX_DB * 10:
        _invalid(path, f"must be a gain in 0.1 dB steps in {GAIN_MIN_DB}..{GAIN_MAX_DB} (whole dB as an integer)")


_SOUND_KEYS = {
    "id", "type", "duration_frames", "source", "source_in_frame", "source_rate", "media_info",
    "record_start_frame", "gain_db", "fade_in_frames", "fade_out_frames", "enabled", "label", "note",
}


def _follow(value: Any, path: str) -> None:
    follow = _exact(value, path, {"clip", "offset_frames"})
    _ident(follow["clip"], path + ".clip")
    _int(follow["offset_frames"], path + ".offset_frames", minimum=-MAX_FRAME)


def _sound(item: dict[str, Any], path: str, rate: dict[str, int]) -> None:
    _exact(item, path, _SOUND_KEYS, {"attach", "link", "volume_keys", "map_id"})
    if item.get("map_id") is not None:  # the song's pinned analysis (place_music, lane w11-cut M-99)
        _sha(item["map_id"], path + ".map_id")
    if item["type"] != "clip":
        _invalid(path + ".type", "a sound track holds clips only (time between them is empty)")
    duration = _int(item["duration_frames"], path + ".duration_frames", minimum=1)
    source = _exact(item["source"], path + ".source", {"kind", "ref"})
    _enum(source["kind"], path + ".source.kind", ("take", "media"))
    _text(source["ref"], path + ".source.ref", minimum=1, maximum=4096)
    source_in = _int(item["source_in_frame"], path + ".source_in_frame")
    source_rate = _rate(item["source_rate"], path + ".source_rate")
    info = _exact(item["media_info"], path + ".media_info", {"frames", "sample_rate", "channels"})
    frames = _int(info["frames"], path + ".media_info.frames", minimum=1)
    if info["sample_rate"] is not None:
        _int(info["sample_rate"], path + ".media_info.sample_rate", minimum=1000, maximum=768_000)
    if info["channels"] is not None:
        _int(info["channels"], path + ".media_info.channels", minimum=1, maximum=64)
    if source_in + used_source_frames(duration, source_rate, rate) > frames:
        _invalid(path, "source range runs past the end of the sound")
    _int(item["record_start_frame"], path + ".record_start_frame")
    _gain(item["gain_db"], path + ".gain_db")
    fade_in = _int(item["fade_in_frames"], path + ".fade_in_frames")
    fade_out = _int(item["fade_out_frames"], path + ".fade_out_frames")
    if fade_in + fade_out > duration:
        _invalid(path, "fade_in_frames + fade_out_frames is longer than the clip")
    _bool(item["enabled"], path + ".enabled")
    _text(item["label"], path + ".label", maximum=200)
    _text(item["note"], path + ".note", maximum=2000)
    if item.get("attach") is not None and item.get("link") is not None:
        _invalid(path, "a sound clip follows one picture clip: attach or link")
    for key in ("attach", "link"):
        if item.get(key) is not None:
            _follow(item[key], f"{path}.{key}")
    if "volume_keys" in item:
        _volume_keys(item["volume_keys"], path + ".volume_keys", frames)


def _volume_keys(keys: Any, path: str, frames: int) -> None:
    if type(keys) is not list or not 1 <= len(keys) <= MAX_VOLUME_KEYS:
        _invalid(path, f"must be a list of 1..{MAX_VOLUME_KEYS} keys (no keys: leave the field out)")
    previous = -1
    for index, key in enumerate(keys):
        kpath = f"{path}[{index}]"
        _exact(key, kpath, {"source_frame", "gain_db", "auto"})
        frame = _int(key["source_frame"], kpath + ".source_frame", maximum=frames)
        if frame <= previous:
            _invalid(kpath + ".source_frame", "keys must rise strictly in source frames")
        previous = frame
        _gain(key["gain_db"], kpath + ".gain_db")
        _bool(key["auto"], kpath + ".auto")


def _duck(value: Any, path: str) -> None:
    # Lane p12-cutb (CUT-043): ``on`` (optional, absent = on) is the track's ducking switch; Off keeps the parameters.
    duck = _exact(value, path, DUCK_KEYS, {"on"})
    if "on" in duck:
        _bool(duck["on"], path + ".on")
    _enum(duck["key_role"], path + ".key_role", AUDIO_ROLES)
    tenths = gain_tenths(duck["depth_db"])
    if tenths is None or not 1 <= tenths <= 600:
        _invalid(path + ".depth_db", "must be in 0.1..60 dB (0.1 dB steps)")
    for key, maximum in (("attack_ms", 5000), ("release_ms", 10000), ("lookahead_ms", 5000), ("hold_ms", 10000)):
        _int(duck[key], f"{path}.{key}", maximum=maximum)


def _music(value: Any, path: str) -> None:
    if value is None:
        return
    music = _exact(value, path, {"track_sha256", "map_id", "record_start_frame", "source_in_frame", "duration_frames", "gain_db"})
    _sha(music["track_sha256"], path + ".track_sha256")
    _sha(music["map_id"], path + ".map_id")
    _int(music["record_start_frame"], path + ".record_start_frame")
    _int(music["source_in_frame"], path + ".source_in_frame")
    _int(music["duration_frames"], path + ".duration_frames", minimum=1)
    _int(music["gain_db"], path + ".gain_db", minimum=-60, maximum=24)


def _marker(value: Any, path: str) -> None:
    marker = _exact(value, path, {"id", "frame", "duration_frames", "name", "note", "kind", "color", "shot_id", "by"}, {"attach_clip"})
    _ident(marker["id"], path + ".id")
    _int(marker["frame"], path + ".frame")
    _int(marker["duration_frames"], path + ".duration_frames")
    _text(marker["name"], path + ".name", maximum=200)
    _text(marker["note"], path + ".note", maximum=2000)
    _enum(marker["kind"], path + ".kind", ("note", "shot", "music"))
    _text(marker["color"], path + ".color", maximum=32)
    if marker["shot_id"] is not None:
        _ident(marker["shot_id"], path + ".shot_id")
    _enum(marker["by"], path + ".by", ("artist", "hermes"))
    if marker.get("attach_clip") is not None:
        _ident(marker["attach_clip"], path + ".attach_clip")


def _timeline(value: Any, path: str) -> None:
    timeline = _exact(value, path, {"id", "name", "rate", "format", "start_tc_frame", "tracks", "transitions", "markers"})
    _ident(timeline["id"], path + ".id")
    _text(timeline["name"], path + ".name", minimum=1, maximum=120)
    rate = _rate(timeline["rate"], path + ".rate")
    fmt = _exact(timeline["format"], path + ".format", {"width", "height", "fit"})
    _int(fmt["width"], path + ".format.width", minimum=16, maximum=16384)
    _int(fmt["height"], path + ".format.height", minimum=16, maximum=16384)
    _enum(fmt["fit"], path + ".format.fit", FITS)
    _int(timeline["start_tc_frame"], path + ".start_tc_frame")
    if type(timeline["transitions"]) is not list or len(timeline["transitions"]) > MAX_TRANSITIONS:
        _invalid(path + ".transitions", f"must be a list of at most {MAX_TRANSITIONS}")
    tracks = timeline["tracks"]
    if type(tracks) is not list or not 1 <= len(tracks) <= MAX_TRACKS:
        _invalid(path + ".tracks", f"must hold 1..{MAX_TRACKS} tracks")
    ids: set[str] = set()

    def claim(ident: str, where: str) -> None:
        if ident in ids:
            _invalid(where, f"id {ident} is not unique")
        ids.add(ident)

    storylines = musics = audios = 0
    clip_ids: set[str] = set()
    for t_index, track in enumerate(tracks):
        tpath = f"{path}.tracks[{t_index}]"
        if type(track) is not dict:
            _invalid(tpath, "must be an object")
        kind = _enum(track.get("kind"), tpath + ".kind", ("video", "music", "audio"))
        if kind == "audio":
            _exact(track, tpath, {"id", "kind", "role", "gain_db", "mute", "solo", "locked", "items"}, {"duck"})
            audios += 1
            if "duck" in track:
                _duck(track["duck"], tpath + ".duck")
            _enum(track["role"], tpath + ".role", AUDIO_ROLES)
            _gain(track["gain_db"], tpath + ".gain_db")
            _bool(track["mute"], tpath + ".mute")
            _bool(track["solo"], tpath + ".solo")
            items = track["items"]
            if type(items) is not list or len(items) > MAX_ITEMS:
                _invalid(tpath + ".items", f"must be a list of at most {MAX_ITEMS}")
            previous_end = 0
            for i_index, item in enumerate(items):
                ipath = f"{tpath}.items[{i_index}]"
                if type(item) is not dict:
                    _invalid(ipath, "must be an object")
                _ident(item.get("id"), ipath + ".id")
                claim(item["id"], ipath + ".id")
                _sound(item, ipath, rate)
                if item["record_start_frame"] < previous_end:
                    fail("overlap", f"{ipath}: overlaps the previous clip on {track['id']}")
                previous_end = item["record_start_frame"] + item["duration_frames"]
            _ident(track["id"], tpath + ".id")
            claim(track["id"], tpath + ".id")
            _bool(track["locked"], tpath + ".locked")
            continue
        if kind == "music":
            _exact(track, tpath, {"id", "kind", "enabled", "locked", "music"})
            musics += 1
            _music(track["music"], tpath + ".music")
        else:
            _exact(track, tpath, {"id", "kind", "role", "enabled", "locked", "items"})
            role = _enum(track["role"], tpath + ".role", ("storyline", "overlay"))
            storylines += role == "storyline"
            items = track["items"]
            if type(items) is not list or len(items) > MAX_ITEMS:
                _invalid(tpath + ".items", f"must be a list of at most {MAX_ITEMS}")
            previous_end = 0
            for i_index, item in enumerate(items):
                ipath = f"{tpath}.items[{i_index}]"
                if type(item) is not dict:
                    _invalid(ipath, "must be an object")
                item_type = _enum(item.get("type"), ipath + ".type", ("clip", "gap"))
                _ident(item.get("id"), ipath + ".id")
                claim(item["id"], ipath + ".id")
                if item_type == "gap":
                    if role != "storyline":
                        _invalid(ipath, "gaps are explicit on the storyline only")
                    _exact(item, ipath, {"id", "type", "duration_frames"})
                    _int(item["duration_frames"], ipath + ".duration_frames", minimum=1)
                    continue
                clip_ids.add(item["id"])
                _clip(item, ipath, storyline=role == "storyline", rate=rate)
                if role == "overlay":
                    start = item["record_start_frame"]
                    if start < previous_end:
                        fail("overlap", f"{ipath}: overlaps the previous clip on {track['id']}")
                    previous_end = start + item["duration_frames"]
        _ident(track["id"], tpath + ".id")
        claim(track["id"], tpath + ".id")
        _bool(track["enabled"], tpath + ".enabled")
        _bool(track["locked"], tpath + ".locked")
    if storylines != 1:
        _invalid(path + ".tracks", "needs exactly one storyline track")
    if musics > 1:
        _invalid(path + ".tracks", "holds at most one music lane")
    if audios > MAX_AUDIO_TRACKS:
        _invalid(path + ".tracks", f"holds at most {MAX_AUDIO_TRACKS} sound tracks (A1-A3)")
    retimed = {item["id"] for track in tracks if track["kind"] == "video" for item in track["items"]
               if item["type"] == "clip" and item.get("retime") is not None}
    for t_index, track in enumerate(tracks):
        for i_index, item in enumerate(track.get("items") or ()):
            for key in ("attach", "link"):
                follow = item.get(key)
                if follow is not None and follow["clip"] not in clip_ids:
                    _invalid(f"{path}.tracks[{t_index}].items[{i_index}].{key}.clip", "names no picture clip")
                if key == "link" and follow is not None and follow["clip"] in retimed:
                    fail("retimed", f"{path}.tracks[{t_index}].items[{i_index}].link: {follow['clip']} is retimed; its sound cannot follow it yet")
    _transitions(timeline, path, rate, claim)
    markers = timeline["markers"]
    if type(markers) is not list or len(markers) > MAX_MARKERS:
        _invalid(path + ".markers", f"must be a list of at most {MAX_MARKERS}")
    for m_index, marker in enumerate(markers):
        mpath = f"{path}.markers[{m_index}]"
        _marker(marker, mpath)
        claim(marker["id"], mpath + ".id")
        if marker.get("attach_clip") is not None and marker["attach_clip"] not in clip_ids:
            _invalid(mpath + ".attach_clip", "names no clip")


def _transitions(timeline: dict[str, Any], path: str, rate: dict[str, int], claim: Callable[[str, str], None]) -> None:
    """Each transition sits on a cut between two adjacent storyline clips, one per cut, inside both
    clips (with the other transition of the clip) and on media the clips have (handles)."""
    storyline = next(track for track in timeline["tracks"] if track.get("role") == "storyline")
    items = storyline["items"]
    position = {item["id"]: index for index, item in enumerate(items)}
    used: dict[str, int] = {}
    cuts: set[str] = set()
    for t_index, transition in enumerate(timeline["transitions"]):
        tpath = f"{path}.transitions[{t_index}]"
        _exact(transition, tpath, {"id", "kind", "left", "right", "duration_frames", "alignment"})
        _ident(transition["id"], tpath + ".id")
        claim(transition["id"], tpath + ".id")
        _enum(transition["kind"], tpath + ".kind", TRANSITION_KINDS)
        _enum(transition["alignment"], tpath + ".alignment", ALIGNMENTS)
        _int(transition["duration_frames"], tpath + ".duration_frames", minimum=1)
        left, right = transition["left"], transition["right"]
        if left not in position or right not in position or position[right] != position[left] + 1:
            _invalid(tpath, "joins two adjacent clips on the storyline (left, then right)")
        left_item, right_item = items[position[left]], items[position[right]]
        if left_item["type"] != "clip" or right_item["type"] != "clip":
            _invalid(tpath, "joins two clips, not a gap")
        if left in cuts:
            _invalid(tpath, "a cut holds one transition")
        cuts.add(left)
        before, after = transition_split(transition)
        used[left] = used.get(left, 0) + before
        used[right] = used.get(right, 0) + after
        for ident in (left, right):
            if used[ident] > items[position[ident]]["duration_frames"]:
                _invalid(tpath, f"is longer than clip {ident} can hold")
        if handles(left_item, rate)[1] < after:
            fail("no_handles", f"{tpath}: {left} has {handles(left_item, rate)[1]} frames after its out point; the transition needs {after}")
        if handles(right_item, rate)[0] < before:
            fail("no_handles", f"{tpath}: {right} has {handles(right_item, rate)[0]} frames before its in point; the transition needs {before}")


def validate_state(value: Any) -> dict[str, Any]:
    """Validate a complete timeline ``state``; returns it unchanged (never repaired)."""
    state = _exact(value, "state", {"active", "timelines", "slots"})
    timelines = state["timelines"]
    if type(timelines) is not list or not 1 <= len(timelines) <= MAX_TIMELINES:
        _invalid("state.timelines", f"must hold 1..{MAX_TIMELINES} timelines")
    seen: set[str] = set()
    for index, timeline in enumerate(timelines):
        _timeline(timeline, f"state.timelines[{index}]")
        if timeline["id"] in seen:
            _invalid(f"state.timelines[{index}].id", "is not unique")
        seen.add(timeline["id"])
    if state["active"] not in seen:
        _invalid("state.active", "names no timeline")
    from .nest import check_state  # lane w23-bins (CUT-180): nested cuts name a cut at the same rate, no cycles
    check_state(state)
    slots = state["slots"]
    if type(slots) is not list or len(slots) > MAX_SLOTS:
        _invalid("state.slots", f"must be a list of at most {MAX_SLOTS}")
    slot_ids: set[str] = set()
    for index, slot in enumerate(slots):
        spath = f"state.slots[{index}]"
        _exact(slot, spath, {"id", "name", "shot_id", "takes", "selected"})
        _ident(slot["id"], spath + ".id")
        if slot["id"] in slot_ids:
            _invalid(spath + ".id", "is not unique")
        slot_ids.add(slot["id"])
        _text(slot["name"], spath + ".name", maximum=120)
        if slot["shot_id"] is not None:
            _ident(slot["shot_id"], spath + ".shot_id")
        if type(slot["takes"]) is not list or len(slot["takes"]) > 200:
            _invalid(spath + ".takes", "must be a list of at most 200 refs")
        for t_index, ref in enumerate(slot["takes"]):
            _text(ref, f"{spath}.takes[{t_index}]", minimum=1, maximum=4096)
        if slot["selected"] is not None and slot["selected"] not in slot["takes"]:
            _invalid(spath + ".selected", "must be one of its takes")
    return state


def new_timeline(identifier: str = "tl_1", name: str = "Cut 1", rate: dict[str, int] | None = None) -> dict[str, Any]:
    """An empty cut: storyline V1, overlay V2 and an empty music lane M1."""
    return {
        "id": identifier,
        "name": name,
        "rate": copy.deepcopy(rate) if rate else {"num": 24, "den": 1},
        "format": {"width": 1280, "height": 704, "fit": "fit"},
        "start_tc_frame": 86400,
        "tracks": [
            {"id": "V1", "kind": "video", "role": "storyline", "enabled": True, "locked": False, "items": []},
            {"id": "V2", "kind": "video", "role": "overlay", "enabled": True, "locked": False, "items": []},
            {"id": "M1", "kind": "music", "enabled": True, "locked": False, "music": None},
        ],
        "transitions": [],
        "markers": [],
    }


def new_audio_track(identifier: str, role: str) -> dict[str, Any]:
    """An empty sound track (A1 dialogue, A2 music, A3 effects by default)."""
    return {"id": identifier, "kind": "audio", "role": role, "gain_db": 0, "mute": False, "solo": False,
            "locked": False, "items": []}


def default_state() -> dict[str, Any]:
    return {"active": "tl_1", "timelines": [new_timeline()], "slots": []}
