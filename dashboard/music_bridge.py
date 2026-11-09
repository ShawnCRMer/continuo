"""Small, file-backed bridge between the Music renderer and Hermes tools.

The renderer owns the truth for ephemeral playback state.  This module only
transports bounded, path-free snapshots and named actions through the private
plugin data directory.  Project JSON remains a separate durable authority.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

# Store hygiene (Weg A1): a window that has not reported for a day is gone;
# its state file and action folder are removed at first use, then hourly.
PRUNE_AGE_SECONDS = 24 * 3600.0
PRUNE_INTERVAL_SECONDS = 3600.0
# A window silent this long is gone: removed on read with its actions (suite_bridge's value).
STALE_WINDOW_SECONDS = 3600.0
_PRUNED_AT: dict[str, float] = {}
_PRUNE_LOCK = threading.Lock()


STATE_SCHEMA = "studio.music-ui-state/1"
ACTION_SCHEMA = "studio.music-ui-action/1"
ACK_SCHEMA = "studio.music-ui-ack/1"
_WINDOW_RE = re.compile(r"^win_[0-9a-f]{32}$")
_ACTION_RE = re.compile(r"^act_[0-9a-f]{32}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_ACTIONS = frozenset({
    "set_selection",
    "seek",
    "play",
    "pause",
    "set_loop",
    "return_to_in",
    "reveal_playhead",
    "set_layers",
    "mark_in",
    "mark_out",
    "fit_selection",
    # fs-inout: Alt+X (no IN-OUT: the whole song).
    "clear_range",
    # fs-inout (E18): Analyse / Stop on the page (payload engine: beat_this | librosa, optional).
    "analyse",
    "stop_analysis",
    # Lane p13b-aud item 2: the engine select (payload engine: beat_this | librosa): shows that engine's stored result.
    "show_engine",
    # fs-music4: Space (the song from the playhead to its end) and the Snap toggle (N).
    "play_song",
    "set_snap",
    # fs-music4 S3: the inspector's grid corrections (x2/3 x3/4 x4/3 x1.5, nudge, Beat here).
    "set_tempo_factor",
    "nudge_grid",
    "beat_here",
    # fs-music4 S4: markers and hot cues (Timeline markers, kind music) and bar jumps.
    "add_marker",
    "set_cue",
    "jump_cue",
    "jump_bars",
    # Lane p13-aud item 2 (AUD-007; R4 Serato X / Ableton "Set 1.1.1 Here" / Logic time signature hint): B = Bar 1 on the
    # beat at the playhead (stored at once, one undo step) and the meter select (beats per bar), both stored in
    # /music/interpretation.
    "set_bar_one",
    "set_meter",
    # w3-audio: the Audio page's Create | Edit and its stems panel (separate / stop / play / to_media / solo / mute).
    "set_audio_mode",
    "stems",
    # w7-spur: the Music track heads (eyes and "+ N hidden"): hidden is the complete list of hidden track ids.
    "set_rows",
    # Lane w17-ainsp item 1 (AUD-077; the Image inspector's ✕ and Ctrl+[): hide / show the Audio inspector.
    "set_inspector",
})
# Lane p14-sam item 2 (HER-051; live 01.11. Hermes sent seek {"time"} and was refused with no hint): ONE table of every
# action's payload. The validator checks key sets against it and the continuo_music_action description Hermes reads is
# built from it (payload_description), so the two can never disagree. (key, required, value) per key; () = no payload.
ACTION_PAYLOADS: dict[str, tuple[tuple[str, bool, str], ...]] = {
    "set_selection": (("start", True, "seconds"), ("end", True, "seconds > start")),
    "seek": (("position", True, "seconds"),),
    "set_loop": (("enabled", True, "bool"),),
    "analyse": (("engine", False, "beat_this|librosa"),),
    "show_engine": (("engine", True, "beat_this|librosa"),),
    "set_audio_mode": (("mode", True, "create|edit"),),
    "stems": (("do", True, "separate|stop|play|to_media|solo|mute|split"), ("split", False, "vocals_rest|four"),
              ("stem", False, "vocals|drums|bass|other|rest"), ("on", False, "bool")),
    "set_snap": (("enabled", True, "bool"),),
    "add_marker": (("snap", False, "bool"),),
    "set_cue": (("n", True, "1-9"),),
    "jump_cue": (("n", True, "1-9"),),
    "set_meter": (("beats_per_bar", True, "1-32"),),
    "jump_bars": (("count", True, "1-64"), ("direction", False, "-1|1 (default 1)")),
    "set_tempo_factor": (("factor", True, "1/2|2/3|3/4|4/3|3/2|2/1"),),
    "nudge_grid": (("ms", True, "-1000..1000 (not 0)"),),
    "set_rows": (("hidden", True, "list of track ids"),),
    "set_layers": (("accents", False, "bool"), ("energy", False, "bool")),
    "set_inspector": (("open", True, "bool"),),
}


def payload_spec(action: str) -> str:
    """'seek {position: seconds}', 'jump_bars {count: 1-64, direction?: -1|1 (default 1)}', 'play {}'."""
    keys = ", ".join(f"{key}{'' if required else '?'}: {value}" for key, required, value in ACTION_PAYLOADS.get(action, ()))
    return f"{action} {{{keys}}}"


def payload_description() -> str:
    """The payload of every action, for the tool description (one line, actions in name order)."""
    with_payload = "; ".join(payload_spec(action) for action in sorted(_ALLOWED_ACTIONS) if ACTION_PAYLOADS.get(action))
    without = ", ".join(sorted(action for action in _ALLOWED_ACTIONS if not ACTION_PAYLOADS.get(action)))
    return f"Payload per action (? = optional): {with_payload}. No payload ({{}}): {without}."


_MARKER_ID_RE = re.compile(r"^music:[0-9a-f]{12}:(?:m[0-9]{1,6}|cue[1-9])$")
_MARKERS_MAX = 200
# set_tempo_factor takes a factor id, or the plain number for 1.5 / 0.75.
# Lane p12-aud item 5 (AUD-008; Rekordbox/Serato x1/2 x2, MX CHANGELOG:267 half-beat grid): Hermes halves and doubles too.
_TEMPO_FACTORS = {"2/3": "2/3", "3/4": "3/4", "4/3": "4/3", "3/2": "3/2", 1.5: "3/2", 0.75: "3/4",
                  "1/2": "1/2", "2/1": "2/1", "2": "2/1", "0.5": "1/2", 0.5: "1/2", 2.0: "2/1"}
_NUDGE_MS_LIMIT = 1000
# Optional "music" block (fs-music): what the analysis shows on the work
# surface (key, accents, energy), the view layers and the timing edit on the
# waveform. Old renderers omit it; every field is bounded and path-free.
_MUSIC_KEYS = {"key", "accents", "energy", "layers", "timing"}
# fs-beat: optional "tempo" block (older renderers do not send it).
# fs-music4: optional "editor" block (what the transport plays, loop).
_MUSIC_OPTIONAL_KEYS = {"tempo", "editor", "markers", "inspector", "readout"}
# Lane w17-ainsp item 1: the Audio inspector Hermes reads (shown or hidden, which tab).
_INSPECTOR_TABS = {"song", "history", "marker", "stem"}
_PLAY_MODES = {"song", "range"}
_TEMPO_KEYS = {"word", "reason", "status", "meter_status", "engine", "decoder"}
_TRUST_WORDS = {"sure", "unsure", "varies", "unknown"}
_ENGINE_IDS = {"beat_this", "librosa_v6"}
_SHORT_CODE_RE = re.compile(r"[a-z_]{1,24}")
_LAYER_KEYS = {"accents", "energy"}
_ROW_ID_RE = re.compile(r"[a-z0-9_]{1,40}")
_TIMING_STATES = {"none", "default", "dirty", "saving", "saved", "conflict"}
_KEY_NAMES = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}


class MusicBridgeValidationError(ValueError):
    """A public bridge envelope is malformed or contains an unsupported field."""


def _finite_number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise MusicBridgeValidationError(f"{label} must be a finite number")
    number = float(value)
    if minimum is not None and number < minimum:
        raise MusicBridgeValidationError(f"{label} is out of range")
    return number


def _window_id(value: Any) -> str:
    text = str(value or "")
    if not _WINDOW_RE.fullmatch(text):
        raise MusicBridgeValidationError("window_id is invalid")
    return text


def _action_id(value: Any) -> str:
    text = str(value or "")
    if not _ACTION_RE.fullmatch(text):
        raise MusicBridgeValidationError("action_id is invalid")
    return text


def _optional_text(value: Any, label: str, maximum: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > maximum:
        raise MusicBridgeValidationError(f"{label} is invalid")
    return value


def _validate_state(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise MusicBridgeValidationError("state must be an object")
    expected = {
        "schema",
        "context_key",
        "context_epoch",
        "project_id",
        "active_tab",
        "surface_visible",
        "track",
        "analysis",
        "selection",
        "transport",
        "notice",
    }
    if set(state) - {"music"} != expected:
        raise MusicBridgeValidationError("state fields are invalid")
    if "music" in state:
        _validate_music_block(state["music"])
    if state["schema"] != STATE_SCHEMA:
        raise MusicBridgeValidationError("state schema is invalid")
    for key, maximum in (("context_key", 160), ("context_epoch", 160), ("active_tab", 32)):
        if not isinstance(state[key], str) or not state[key] or len(state[key]) > maximum:
            raise MusicBridgeValidationError(f"{key} is invalid")
    project_id = state["project_id"]
    if project_id is not None and (not isinstance(project_id, str) or not re.fullmatch(r"prj_[0-9a-f]{32}", project_id)):
        raise MusicBridgeValidationError("project_id is invalid")
    if not isinstance(state["surface_visible"], bool):
        raise MusicBridgeValidationError("surface_visible is invalid")

    track = state["track"]
    if not isinstance(track, dict) or set(track) - {"selected", "name", "sha256", "duration"}:
        raise MusicBridgeValidationError("track fields are invalid")
    if not isinstance(track.get("selected"), bool):
        raise MusicBridgeValidationError("track.selected is invalid")
    name = _optional_text(track.get("name"), "track.name", 260)
    digest = track.get("sha256")
    if digest is not None and (not isinstance(digest, str) or not _SHA_RE.fullmatch(digest)):
        raise MusicBridgeValidationError("track.sha256 is invalid")
    duration = track.get("duration")
    if duration is not None:
        _finite_number(duration, "track.duration", minimum=0)
    if track["selected"] and not name:
        raise MusicBridgeValidationError("selected track needs a name")

    analysis = state["analysis"]
    if not isinstance(analysis, dict) or set(analysis) != {"status", "available", "duration"}:
        raise MusicBridgeValidationError("analysis fields are invalid")
    if analysis["status"] not in {"none", "loading", "ready", "error"} or not isinstance(analysis["available"], bool):
        raise MusicBridgeValidationError("analysis status is invalid")
    if analysis["duration"] is not None:
        _finite_number(analysis["duration"], "analysis.duration", minimum=0)

    selection = state["selection"]
    if selection is not None:
        if not isinstance(selection, dict) or set(selection) != {"start", "end"}:
            raise MusicBridgeValidationError("selection fields are invalid")
        start = _finite_number(selection["start"], "selection.start", minimum=0)
        end = _finite_number(selection["end"], "selection.end", minimum=0)
        if end <= start:
            raise MusicBridgeValidationError("selection range is invalid")

    transport = state["transport"]
    transport_keys = {"status", "ready", "playing", "logical_time", "media_duration", "error_code"}
    if not isinstance(transport, dict) or set(transport) != transport_keys:
        raise MusicBridgeValidationError("transport fields are invalid")
    if not isinstance(transport["status"], str) or len(transport["status"]) > 40:
        raise MusicBridgeValidationError("transport.status is invalid")
    if not isinstance(transport["ready"], bool) or not isinstance(transport["playing"], bool):
        raise MusicBridgeValidationError("transport flags are invalid")
    for key in ("logical_time", "media_duration"):
        if transport[key] is not None:
            _finite_number(transport[key], f"transport.{key}", minimum=0)
    _optional_text(transport["error_code"], "transport.error_code", 96)

    notice = state["notice"]
    if not isinstance(notice, dict) or set(notice) != {"tone", "text"}:
        raise MusicBridgeValidationError("notice fields are invalid")
    if notice["tone"] not in {"info", "success", "warning", "error"}:
        raise MusicBridgeValidationError("notice.tone is invalid")
    if not isinstance(notice["text"], str) or len(notice["text"]) > 240:
        raise MusicBridgeValidationError("notice.text is invalid")
    return json.loads(json.dumps(state, ensure_ascii=False, allow_nan=False))


def _validate_music_block(music: Any) -> None:
    if not isinstance(music, dict) or not _MUSIC_KEYS <= set(music) <= _MUSIC_KEYS | _MUSIC_OPTIONAL_KEYS:
        raise MusicBridgeValidationError("music fields are invalid")
    key = music["key"]
    if key is not None:
        if not isinstance(key, dict) or set(key) != {"name", "mode", "sure"}:
            raise MusicBridgeValidationError("music.key fields are invalid")
        if key["name"] not in _KEY_NAMES or key["mode"] not in {"major", "minor"} or not isinstance(key["sure"], bool):
            raise MusicBridgeValidationError("music.key is invalid")
    accents = music["accents"]
    if not isinstance(accents, dict) or set(accents) != {"count", "strongest"}:
        raise MusicBridgeValidationError("music.accents fields are invalid")
    if not isinstance(accents["count"], int) or isinstance(accents["count"], bool) or not 0 <= accents["count"] <= 100000:
        raise MusicBridgeValidationError("music.accents.count is invalid")
    if not isinstance(accents["strongest"], list) or len(accents["strongest"]) > 8:
        raise MusicBridgeValidationError("music.accents.strongest is invalid")
    for row in accents["strongest"]:
        if not isinstance(row, dict) or set(row) != {"time", "level"}:
            raise MusicBridgeValidationError("music.accents.strongest row is invalid")
        _finite_number(row["time"], "music.accents.time", minimum=0)
        if not 0 <= _finite_number(row["level"], "music.accents.level", minimum=0) <= 1:
            raise MusicBridgeValidationError("music.accents.level is invalid")
    energy = music["energy"]
    if not isinstance(energy, dict) or set(energy) != {"points", "peak_time"}:
        raise MusicBridgeValidationError("music.energy fields are invalid")
    if not isinstance(energy["points"], int) or isinstance(energy["points"], bool) or not 0 <= energy["points"] <= 1000000:
        raise MusicBridgeValidationError("music.energy.points is invalid")
    if energy["peak_time"] is not None:
        _finite_number(energy["peak_time"], "music.energy.peak_time", minimum=0)
    layers = music["layers"]
    if not isinstance(layers, dict) or set(layers) != _LAYER_KEYS or not all(isinstance(layers[k], bool) for k in _LAYER_KEYS):
        raise MusicBridgeValidationError("music.layers is invalid")
    timing = music["timing"]
    # Lane p13-aud item 2: beats_per_bar (the artist's meter, null while detected) is optional for older pages.
    # Lane p13b-aud item 1: bars (how many bar lines the artist's grid draws, null without one) is optional too.
    if not isinstance(timing, dict) or set(timing) - {"beats_per_bar", "bars"} != {"state", "preview_bpm", "bar_one"}:
        raise MusicBridgeValidationError("music.timing fields are invalid")
    if timing["state"] not in _TIMING_STATES:
        raise MusicBridgeValidationError("music.timing.state is invalid")
    for name in ("preview_bpm", "bar_one"):
        if timing[name] is not None:
            _finite_number(timing[name], f"music.timing.{name}", minimum=0)
    beats = timing.get("beats_per_bar")
    if beats is not None and (not isinstance(beats, int) or isinstance(beats, bool) or not 1 <= beats <= 32):
        raise MusicBridgeValidationError("music.timing.beats_per_bar is invalid")
    bars = timing.get("bars")
    if bars is not None and (not isinstance(bars, int) or isinstance(bars, bool) or not 1 <= bars <= 100000):
        raise MusicBridgeValidationError("music.timing.bars is invalid")
    # fs-beat: the one tempo trust value the head and inspector show, plus its engine.
    if "tempo" in music:
        tempo = music["tempo"]
        if not isinstance(tempo, dict) or set(tempo) != _TEMPO_KEYS:
            raise MusicBridgeValidationError("music.tempo fields are invalid")
        if tempo["word"] not in _TRUST_WORDS:
            raise MusicBridgeValidationError("music.tempo.word is invalid")
        if tempo["engine"] is not None and tempo["engine"] not in _ENGINE_IDS:
            raise MusicBridgeValidationError("music.tempo.engine is invalid")
        for name in ("reason", "status", "meter_status", "decoder"):
            value = tempo[name]
            if value is not None and (not isinstance(value, str) or not _SHORT_CODE_RE.fullmatch(value)):
                raise MusicBridgeValidationError(f"music.tempo.{name} is invalid")
    if "editor" in music:
        _validate_editor_block(music["editor"])
    if "markers" in music:
        _validate_markers_block(music["markers"])
    if "readout" in music:
        _validate_readout_block(music["readout"])
    if "inspector" in music:
        inspector = music["inspector"]
        # w17-ainspc item 4: optional "markers", the Markers section head text the page shows ("4 sections", "none").
        if not isinstance(inspector, dict) or not {"open", "tab"} <= set(inspector) <= {"open", "tab", "markers"} or not isinstance(inspector["open"], bool) or inspector["tab"] not in _INSPECTOR_TABS:
            raise MusicBridgeValidationError("music.inspector is invalid")
        if "markers" in inspector and (not isinstance(inspector["markers"], str) or not 0 < len(inspector["markers"]) <= 40):
            raise MusicBridgeValidationError("music.inspector.markers is invalid")


def _validate_readout_block(readout: Any) -> None:
    # Lane w25-audio2 (Audio board 2 additions): the transport's Bar.Beat ("12.3", "–" before Bar 1), the level meter
    # (0-100, null while stopped), the numbered sections and the pickup length (s before Bar 1).
    if not isinstance(readout, dict) or not {"bar_beat", "level"} <= set(readout) <= {"bar_beat", "level", "sections", "pickup"}:
        raise MusicBridgeValidationError("music.readout is invalid")
    bb = readout["bar_beat"]
    if bb is not None and (not isinstance(bb, str) or not re.fullmatch(r"(\d{1,5}\.\d{1,2}|–)", bb)):
        raise MusicBridgeValidationError("music.readout.bar_beat is invalid")
    if readout["level"] is not None and not 0 <= _finite_number(readout["level"], "music.readout.level", minimum=0) <= 100:
        raise MusicBridgeValidationError("music.readout.level is invalid")
    if "pickup" in readout:
        _finite_number(readout["pickup"], "music.readout.pickup", minimum=0)
    if "sections" in readout:
        rows = readout["sections"]
        if not isinstance(rows, list) or len(rows) > 64:
            raise MusicBridgeValidationError("music.readout.sections is invalid")
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"name", "time", "bar"} or not isinstance(row["name"], str) or not 0 < len(row["name"]) <= 40:
                raise MusicBridgeValidationError("music.readout.sections row is invalid")
            _finite_number(row["time"], "music.readout.sections.time", minimum=0)
            if row["bar"] is not None and (not isinstance(row["bar"], int) or isinstance(row["bar"], bool) or row["bar"] < 1):
                raise MusicBridgeValidationError("music.readout.sections.bar is invalid")


def _validate_markers_block(markers: Any) -> None:
    if not isinstance(markers, list) or len(markers) > _MARKERS_MAX:
        raise MusicBridgeValidationError("music.markers is invalid")
    for row in markers:
        if not isinstance(row, dict) or set(row) != {"id", "time", "frame", "cue", "by"}:
            raise MusicBridgeValidationError("music.markers row fields are invalid")
        if not isinstance(row["id"], str) or not _MARKER_ID_RE.fullmatch(row["id"]):
            raise MusicBridgeValidationError("music.markers id is invalid")
        _finite_number(row["time"], "music.markers.time", minimum=0)
        if not isinstance(row["frame"], int) or isinstance(row["frame"], bool) or row["frame"] < 0:
            raise MusicBridgeValidationError("music.markers frame is invalid")
        if row["cue"] is not None and (not isinstance(row["cue"], int) or isinstance(row["cue"], bool) or not 1 <= row["cue"] <= 9):
            raise MusicBridgeValidationError("music.markers cue is invalid")
        if row["by"] not in {"artist", "hermes"}:
            raise MusicBridgeValidationError("music.markers by is invalid")


def _validate_editor_block(editor: Any) -> None:
    if not isinstance(editor, dict) or set(editor) != {"play_mode", "loop", "snap"}:
        raise MusicBridgeValidationError("music.editor fields are invalid")
    if editor["play_mode"] is not None and editor["play_mode"] not in _PLAY_MODES:
        raise MusicBridgeValidationError("music.editor.play_mode is invalid")
    if not isinstance(editor["loop"], bool) or not isinstance(editor["snap"], bool):
        raise MusicBridgeValidationError("music.editor flags are invalid")


def _validate_action_payload(action: str, payload: Any) -> dict[str, Any]:
    if action not in _ALLOWED_ACTIONS:
        raise MusicBridgeValidationError("action is unsupported")
    if not isinstance(payload, dict):
        raise MusicBridgeValidationError("action payload must be an object")
    spec = ACTION_PAYLOADS.get(action, ())
    if spec:
        keys, required = set(payload), {key for key, needed, _value in spec if needed}
        if not required <= keys or keys - {key for key, _needed, _value in spec}:
            raise MusicBridgeValidationError(f"{action} payload is invalid: {payload_spec(action)}")
    if action == "set_selection":
        if set(payload) != {"start", "end"}:
            raise MusicBridgeValidationError(f"set_selection payload is invalid: {payload_spec('set_selection')}")
        start = _finite_number(payload["start"], "payload.start", minimum=0)
        end = _finite_number(payload["end"], "payload.end", minimum=0)
        if end <= start:
            raise MusicBridgeValidationError("set_selection range is invalid")
        return {"start": start, "end": end}
    if action == "seek":
        if set(payload) != {"position"}:
            raise MusicBridgeValidationError(f"seek payload is invalid: {payload_spec('seek')}")
        return {"position": _finite_number(payload["position"], "payload.position", minimum=0)}
    if action == "set_loop":
        if set(payload) != {"enabled"} or not isinstance(payload["enabled"], bool):
            raise MusicBridgeValidationError(f"set_loop payload is invalid: {payload_spec('set_loop')}")
        return {"enabled": payload["enabled"]}
    if action == "analyse":
        if set(payload) - {"engine"} or ("engine" in payload and payload["engine"] not in {"beat_this", "librosa"}):
            raise MusicBridgeValidationError(f"analyse payload is invalid: {payload_spec('analyse')}")
        return {"engine": payload["engine"]} if "engine" in payload else {}
    if action == "show_engine":
        if set(payload) != {"engine"} or payload["engine"] not in {"beat_this", "librosa"}:
            raise MusicBridgeValidationError(f"show_engine payload is invalid: {payload_spec('show_engine')}")
        return {"engine": payload["engine"]}
    if action == "set_audio_mode":
        if set(payload) != {"mode"} or payload["mode"] not in {"create", "edit"}:
            raise MusicBridgeValidationError(f"set_audio_mode payload is invalid: {payload_spec('set_audio_mode')}")
        return {"mode": payload["mode"]}
    if action == "stems":
        if set(payload) - {"do", "split", "stem", "on"} or payload.get("do") not in {"separate", "stop", "play", "to_media", "solo", "mute", "split"}:
            raise MusicBridgeValidationError(f"stems payload is invalid: {payload_spec('stems')}")
        if "split" in payload and payload["split"] not in {"vocals_rest", "four"}:
            raise MusicBridgeValidationError("stems split is invalid")
        if "stem" in payload and payload["stem"] not in {"vocals", "drums", "bass", "other", "rest"}:
            raise MusicBridgeValidationError("stems stem is invalid")
        if "on" in payload and not isinstance(payload["on"], bool):
            raise MusicBridgeValidationError("stems on is invalid")
        return dict(payload)
    if action == "set_snap":
        if set(payload) != {"enabled"} or not isinstance(payload["enabled"], bool):
            raise MusicBridgeValidationError(f"set_snap payload is invalid: {payload_spec('set_snap')}")
        return {"enabled": payload["enabled"]}
    if action == "add_marker":
        if set(payload) - {"snap"} or ("snap" in payload and not isinstance(payload["snap"], bool)):
            raise MusicBridgeValidationError(f"add_marker payload is invalid: {payload_spec('add_marker')}")
        return {"snap": payload.get("snap", False)}
    if action in {"set_cue", "jump_cue"}:
        cue = payload.get("n") if set(payload) == {"n"} else None
        if not isinstance(cue, int) or isinstance(cue, bool) or not 1 <= cue <= 9:
            raise MusicBridgeValidationError(f"{action} payload is invalid: {payload_spec(action)}")
        return {"n": cue}
    if action == "set_meter":
        beats = payload.get("beats_per_bar") if set(payload) == {"beats_per_bar"} else None
        if not isinstance(beats, int) or isinstance(beats, bool) or not 1 <= beats <= 32:
            raise MusicBridgeValidationError(f"set_meter payload is invalid: {payload_spec('set_meter')}")
        return {"beats_per_bar": beats}
    if action == "jump_bars":
        # Lane p13-aud item 2 (AUD-007 bench: {count} alone was refused "payload is invalid"): direction defaults to 1.
        payload = {"direction": 1, **payload} if "direction" not in payload else payload
        direction, count = payload.get("direction"), payload.get("count")
        if set(payload) != {"direction", "count"} or direction not in (-1, 1) or isinstance(direction, bool) \
                or not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 64:
            raise MusicBridgeValidationError(f"jump_bars payload is invalid: {payload_spec('jump_bars')}")
        return {"direction": direction, "count": count}
    if action == "set_tempo_factor":
        factor = payload.get("factor") if set(payload) == {"factor"} else None
        key = factor if isinstance(factor, str) else float(factor) if isinstance(factor, (int, float)) and not isinstance(factor, bool) else None
        if key is None or key not in _TEMPO_FACTORS:
            raise MusicBridgeValidationError(f"set_tempo_factor payload is invalid: {payload_spec('set_tempo_factor')}")
        return {"factor": _TEMPO_FACTORS[key]}
    if action == "nudge_grid":
        if set(payload) != {"ms"}:
            raise MusicBridgeValidationError(f"nudge_grid payload is invalid: {payload_spec('nudge_grid')}")
        ms = _finite_number(payload["ms"], "payload.ms", minimum=-_NUDGE_MS_LIMIT)
        if ms == 0 or ms > _NUDGE_MS_LIMIT:
            raise MusicBridgeValidationError("nudge_grid ms is out of range")
        return {"ms": ms}
    if action == "set_rows":
        hidden = payload.get("hidden") if set(payload) == {"hidden"} else None
        if not isinstance(hidden, list) or len(hidden) > 40 or not all(isinstance(i, str) and _ROW_ID_RE.fullmatch(i) for i in hidden):
            raise MusicBridgeValidationError(f"set_rows payload is invalid: {payload_spec('set_rows')}")
        return {"hidden": list(dict.fromkeys(hidden))}
    if action == "set_layers":
        if not payload or set(payload) - _LAYER_KEYS or not all(isinstance(value, bool) for value in payload.values()):
            raise MusicBridgeValidationError(f"set_layers payload is invalid: {payload_spec('set_layers')}")
        return dict(payload)
    if action == "set_inspector":
        if set(payload) != {"open"} or not isinstance(payload["open"], bool):
            raise MusicBridgeValidationError(f"set_inspector payload is invalid: {payload_spec('set_inspector')}")
        return {"open": payload["open"]}
    if payload:
        raise MusicBridgeValidationError(f"{action} does not accept a payload")
    return {}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _semantic_state(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    projection = json.loads(_canonical(value))
    transport = projection.get("transport")
    if isinstance(transport, dict):
        transport["logical_time"] = 0
    return projection


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _atomic_write(path: Path, value: dict[str, Any], *, durable: bool = True) -> None:
    """Replace ``path`` atomically; ``durable=False`` skips fsync (heartbeat-only rewrites)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.flush()
            if durable:
                os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


class MusicBridgeStore:
    """Atomic path-per-envelope storage for renderer observations and actions."""

    def __init__(self, root: Path, clock: Callable[[], float] | None = None) -> None:
        self.root = Path(root)
        self.clock = clock or time.time

    def prune_if_due(self) -> list[str]:
        """Delete window files and action dirs idle for 24 h: first use after start, then hourly."""
        key = str(self.root)
        now = float(self.clock())
        with _PRUNE_LOCK:
            last = _PRUNED_AT.get(key)
            if last is not None and 0.0 <= now - last < PRUNE_INTERVAL_SECONDS:
                return []
            _PRUNED_AT[key] = now
        return self.prune(now)

    def prune(self, now: float | None = None) -> list[str]:
        """Remove idle windows' state files and action dirs; returns the removed window ids."""
        cutoff = (float(self.clock()) if now is None else now) - PRUNE_AGE_SECONDS
        removed: list[str] = []
        try:
            states = list(self.state_dir.glob("win_*.json")) if self.state_dir.is_dir() else []
        except OSError:
            states = []
        for path in states:
            try:
                if _WINDOW_RE.match(path.stem) and path.stat().st_mtime < cutoff:
                    # Gone with its actions (HER-048): a pending file newer than the state must not keep the folder.
                    self.forget(path.stem)
                    removed.append(path.stem)
            except OSError:
                continue
        try:
            folders = list(self.action_dir.glob("win_*")) if self.action_dir.is_dir() else []
        except OSError:
            folders = []
        for folder in folders:
            try:
                if not _WINDOW_RE.match(folder.name) or folder.is_symlink() or not folder.is_dir():
                    continue
                if (self.state_dir / f"{folder.name}.json").exists():
                    continue  # the window still reports; its actions are its own business
                newest = max([folder.stat().st_mtime] + [item.stat().st_mtime for item in folder.rglob("*")])
                if newest < cutoff:
                    shutil.rmtree(folder, ignore_errors=True)
                    if folder.name not in removed:
                        removed.append(folder.name)
            except OSError:
                continue
        return removed

    @property
    def state_dir(self) -> Path:
        return self.root / "states"

    @property
    def action_dir(self) -> Path:
        return self.root / "actions"

    def _state_path(self, window_id: str) -> Path:
        return self.state_dir / f"{_window_id(window_id)}.json"

    def _pending_path(self, window_id: str, action_id: str) -> Path:
        return self.action_dir / _window_id(window_id) / "pending" / f"{_action_id(action_id)}.json"

    def _ack_path(self, window_id: str, action_id: str) -> Path:
        return self.action_dir / _window_id(window_id) / "acks" / f"{_action_id(action_id)}.json"

    def write_snapshot(self, window_id: str, state: dict[str, Any]) -> dict[str, Any]:
        window_id = _window_id(window_id)
        state = _validate_state(state)
        path = self._state_path(window_id)
        previous = _read_json(path) if path.exists() else None
        previous_state = previous.get("state") if previous else None
        revision = int(previous.get("revision", 0)) if previous else 0
        changed = previous is None or _semantic_state(previous_state) != _semantic_state(state)
        if changed:
            revision += 1
        record = {
            "schema": STATE_SCHEMA,
            "window_id": window_id,
            "revision": revision,
            "observed_at": float(self.clock()),
            "state": state,
        }
        # Same semantic state: only the heartbeat (observed_at) moves. Still
        # written - freshness is how the agent sees a connected window - but
        # without fsync; losing it in a crash costs one heartbeat.
        _atomic_write(path, record, durable=changed)
        self.prune_if_due()
        return record

    def forget(self, window_id: str) -> None:
        """Remove a gone window: its state file and its pending actions, acks and views."""
        window_id = _window_id(window_id)
        self._state_path(window_id).unlink(missing_ok=True)
        shutil.rmtree(self.action_dir / window_id, ignore_errors=True)

    def list_snapshots(
        self,
        window_id: str | None = None,
        max_age_seconds: float = 5.0,
        forget_after_seconds: float | None = None,
    ) -> list[dict[str, Any]]:
        """Rows for reporting windows; with ``forget_after_seconds`` a window silent that long is removed on read."""
        paths: list[Path]
        if window_id is not None:
            paths = [self._state_path(window_id)]
        elif self.state_dir.is_dir():
            paths = sorted(self.state_dir.glob("win_*.json"))
        else:
            paths = []
        now = float(self.clock())
        if window_id is None:
            self.prune_if_due()
            paths = [path for path in paths if path.exists()]
        rows: list[dict[str, Any]] = []
        for path in paths:
            record = _read_json(path)
            if forget_after_seconds is not None and path.exists() and _WINDOW_RE.match(path.stem):
                try:
                    observed = float(record["observed_at"]) if record else path.stat().st_mtime
                except (OSError, KeyError, TypeError, ValueError):
                    observed = now
                if now - observed > forget_after_seconds:
                    self.forget(path.stem)
                    continue
            if not record or record.get("schema") != STATE_SCHEMA:
                continue
            try:
                wid = _window_id(record.get("window_id"))
                revision = int(record.get("revision"))
                observed_at = float(record.get("observed_at"))
                state = _validate_state(record.get("state"))
            except (MusicBridgeValidationError, TypeError, ValueError):
                continue
            age = max(0.0, now - observed_at)
            fresh = age <= max(0.0, float(max_age_seconds))
            rows.append({
                "schema": STATE_SCHEMA,
                "window_id": wid,
                "revision": revision,
                "observed_at": observed_at,
                "age_seconds": round(age, 3),
                "freshness": "fresh" if fresh else "stale",
                "connected": fresh,
                "state": state,
            })
        rows.sort(key=lambda row: (row["connected"], row["observed_at"]), reverse=True)
        return rows

    def enqueue_action(
        self,
        window_id: str,
        action: str,
        payload: dict[str, Any],
        expected_revision: int | None = None,
        context_epoch: str | None = None,
    ) -> dict[str, Any]:
        window_id = _window_id(window_id)
        payload = _validate_action_payload(action, payload)
        if expected_revision is not None and (isinstance(expected_revision, bool) or int(expected_revision) < 1):
            raise MusicBridgeValidationError("expected_revision is invalid")
        if context_epoch is not None and (not isinstance(context_epoch, str) or not context_epoch or len(context_epoch) > 160):
            raise MusicBridgeValidationError("context_epoch is invalid")
        created_at = float(self.clock())
        action_id = "act_" + uuid.uuid4().hex
        record = {
            "schema": ACTION_SCHEMA,
            "action_id": action_id,
            "window_id": window_id,
            "action": action,
            "payload": payload,
            "expected_revision": expected_revision,
            "context_epoch": context_epoch,
            "created_at": created_at,
            "expires_at": created_at + 10.0,
        }
        _atomic_write(self._pending_path(window_id, action_id), record)
        return record

    def pending_actions(
        self,
        window_id: str,
        since: float = 0.0,
        context_epoch: str | None = None,
    ) -> list[dict[str, Any]]:
        window_id = _window_id(window_id)
        pending = self.action_dir / window_id / "pending"
        if not pending.is_dir():
            return []
        now = float(self.clock())
        rows: list[dict[str, Any]] = []
        for path in sorted(pending.glob("act_*.json")):
            record = _read_json(path)
            if not record:
                continue
            try:
                action_id = _action_id(record.get("action_id"))
                if record.get("window_id") != window_id or float(record.get("expires_at")) <= now:
                    path.unlink(missing_ok=True)
                    continue
                if float(record.get("created_at")) < float(since):
                    continue
                if context_epoch is not None and record.get("context_epoch") not in {None, context_epoch}:
                    continue
                _validate_action_payload(record.get("action"), record.get("payload"))
                _action_id(action_id)
            except (MusicBridgeValidationError, TypeError, ValueError):
                continue
            rows.append(record)
        rows.sort(key=lambda row: (float(row["created_at"]), row["action_id"]))
        return rows

    def ack_action(self, window_id: str, action_id: str, ack: dict[str, Any]) -> dict[str, Any]:
        window_id = _window_id(window_id)
        action_id = _action_id(action_id)
        if not isinstance(ack, dict) or set(ack) != {"ok", "reason_code", "message", "applied", "revision", "context_epoch"}:
            raise MusicBridgeValidationError("ack fields are invalid")
        if not isinstance(ack["ok"], bool) or (ack["reason_code"] is not None and not isinstance(ack["reason_code"], str)):
            raise MusicBridgeValidationError("ack result is invalid")
        if ack["message"] is not None and (not isinstance(ack["message"], str) or len(ack["message"]) > 240):
            raise MusicBridgeValidationError("ack message is invalid")
        if ack["applied"] is not None and not isinstance(ack["applied"], dict):
            raise MusicBridgeValidationError("ack applied state is invalid")
        if isinstance(ack["revision"], bool) or not isinstance(ack["revision"], int) or ack["revision"] < 0:
            raise MusicBridgeValidationError("ack revision is invalid")
        if not isinstance(ack["context_epoch"], str) or len(ack["context_epoch"]) > 160:
            raise MusicBridgeValidationError("ack context_epoch is invalid")
        record = {
            "schema": ACK_SCHEMA,
            "action_id": action_id,
            "window_id": window_id,
            **json.loads(_canonical(ack)),
        }
        _atomic_write(self._ack_path(window_id, action_id), record)
        self._pending_path(window_id, action_id).unlink(missing_ok=True)
        return record

    def consume_ack(self, window_id: str, action_id: str) -> dict[str, Any] | None:
        path = self._ack_path(window_id, action_id)
        record = _read_json(path)
        if record is None:
            return None
        path.unlink(missing_ok=True)
        return record
