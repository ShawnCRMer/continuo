"""Voices of the characters, the lines they speak and the takes' Audio-page state.

One document per project (``studio.voice-project/1``), kept beside the timeline's
footage index in the plugin's private state root (``voice/<project_id>.json``),
written whole with ``expected_revision`` like the project's other documents.

* voice: ``{id, name, source: clip|description, clip: {ref, name, in, out} | null,
  words, description, seed | null, profile}``.  ``clip.ref`` is the owned copy's
  Media id ``file:<sha256>``; ``in``/``out`` trim it (seconds, ``out`` null = end).
  ``words`` is what the clip says, typed or transcribed; never invented (empty =
  unknown; OmniVoice then transcribes with its local Whisper, Drama 3 refuses).
* characters: ``{name: voice_id}``.
* line: ``{id, character, text, direction}``; ``text`` may hold neutral cues
  ``{cue:sigh}``; ``direction`` is the bracket note ``[softer, to himself]``.
* takes: ``{studio_job_id: {line_id, selected, gain_db, facts}}``: the Audio page's own
  choices (selected, non-destructive gain) and what the backend measured (facts:
  target_id, seed, seconds, loudness, took_seconds, warnings, error; words [{w, start, end}] once read, source
  engine|recording|recut, recut_of, pauses).  Kept, rating and tags are the Media marks of the take (one store).

Standalone (stdlib only).
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "studio.voice-project/1"
MAX_BYTES = 2 * 1024 * 1024
MAX_VOICES = 200
MAX_LINES = 2000
MAX_TAKES = 5000
MAX_FACT_BYTES = 32000
_ID = re.compile(r"^(?:voice|line)_[0-9a-f]{12}$")
_TAKE = re.compile(r"^studio:[0-9a-f-]{36}$")
_REF = re.compile(r"^file:[0-9a-f]{64}$")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_LOCK = threading.Lock()


class VoiceStoreError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def new_id(kind: str) -> str:
    return f"{kind}_{uuid.uuid4().hex[:12]}"


def _bad(message: str) -> None:
    raise VoiceStoreError("invalid_voice_state", message)


def _text(value: Any, where: str, *, max_len: int, empty: bool = True) -> str:
    if type(value) is not str or len(value) > max_len or _CONTROL.search(value) or (not empty and not value.strip()):
        _bad(f"{where} must be text of at most {max_len} characters")
    return value


def _seconds(value: Any, where: str, *, nullable: bool) -> float | None:
    if value is None and nullable:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 3600:
        _bad(f"{where} must be seconds")
    return float(value)


def _closed(value: Any, keys: set[str], where: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        _bad(f"{where} must have exactly {sorted(keys)}")
    return value


def validate_body(body: Mapping[str, Any]) -> dict[str, Any]:
    """The artist-editable part (voices, characters, lines, takes); raises VoiceStoreError."""
    body = _closed(dict(body), {"voices", "characters", "lines", "takes"}, "voice state")
    voices = body["voices"]
    if type(voices) is not list or len(voices) > MAX_VOICES:
        _bad("voices must be a list")
    ids = set()
    for index, voice in enumerate(voices):
        at = f"voices[{index}]"
        _closed(voice, {"id", "name", "source", "clip", "words", "description", "seed", "profile"}, at)
        if type(voice["id"]) is not str or not voice["id"].startswith("voice_") or not _ID.fullmatch(voice["id"]) or voice["id"] in ids:
            _bad(f"{at}.id is invalid")
        ids.add(voice["id"])
        _text(voice["name"], f"{at}.name", max_len=80, empty=False)
        if voice["source"] not in ("clip", "description"):
            _bad(f"{at}.source must be clip or description")
        clip = voice["clip"]
        if clip is not None:
            _closed(clip, {"ref", "name", "in", "out"}, f"{at}.clip")
            if type(clip["ref"]) is not str or not _REF.fullmatch(clip["ref"]):
                _bad(f"{at}.clip.ref must be a Media sound id")
            _text(clip["name"], f"{at}.clip.name", max_len=200, empty=False)
            start = _seconds(clip["in"], f"{at}.clip.in", nullable=False)
            end = _seconds(clip["out"], f"{at}.clip.out", nullable=True)
            if end is not None and end - start < 1.0:
                _bad(f"{at}.clip OUT must be at least 1 s after IN")
        if voice["source"] == "clip" and clip is None:
            _bad(f"{at} speaks from a clip but has none")
        _text(voice["words"], f"{at}.words", max_len=4000)
        _text(voice["description"], f"{at}.description", max_len=400)
        if voice["source"] == "description" and not voice["description"].strip():
            _bad(f"{at} speaks from a description but has none")
        if voice["seed"] is not None and (type(voice["seed"]) is not int or not 0 <= voice["seed"] <= 2147483647):
            _bad(f"{at}.seed must be a whole number")
        profile = voice["profile"]
        if type(profile) is not dict or len(json.dumps(profile)) > 2000:
            _bad(f"{at}.profile must be a small object")
    characters = body["characters"]
    if type(characters) is not dict or len(characters) > MAX_VOICES:
        _bad("characters must be an object")
    for name, voice_id in characters.items():
        _text(name, "character name", max_len=80, empty=False)
        if voice_id not in ids:
            _bad(f"character {name} names an unknown voice")
    lines = body["lines"]
    if type(lines) is not list or len(lines) > MAX_LINES:
        _bad("lines must be a list")
    line_ids = set()
    for index, line in enumerate(lines):
        at = f"lines[{index}]"
        _closed(line, {"id", "character", "text", "direction"}, at)
        if type(line["id"]) is not str or not line["id"].startswith("line_") or not _ID.fullmatch(line["id"]) or line["id"] in line_ids:
            _bad(f"{at}.id is invalid")
        line_ids.add(line["id"])
        _text(line["character"], f"{at}.character", max_len=80, empty=False)
        _text(line["text"], f"{at}.text", max_len=2000, empty=False)
        _text(line["direction"], f"{at}.direction", max_len=200)
    takes = body["takes"]
    if type(takes) is not dict or len(takes) > MAX_TAKES:
        _bad("takes must be an object")
    selected: set[str] = set()
    for take_id, take in takes.items():
        if not _TAKE.fullmatch(take_id):
            _bad("a take id is invalid")
        _closed(take, {"line_id", "selected", "gain_db", "facts"}, f"takes[{take_id}]")
        # 32 KB: the facts carry the take's timed words (up to 400 x {w, start, end}) and its pauses.
        if type(take["facts"]) is not dict or len(json.dumps(take["facts"])) > MAX_FACT_BYTES:
            _bad("take facts must be a small object")
        if take["line_id"] not in line_ids:
            _bad(f"take {take_id} names an unknown line")
        if type(take["selected"]) is not bool:
            _bad("selected must be true or false")
        if take["selected"]:
            if take["line_id"] in selected:
                _bad("one selected take per line")
            selected.add(take["line_id"])
        gain = take["gain_db"]
        if type(gain) not in (int, float) or not math.isfinite(gain) or not -24 <= gain <= 24:
            _bad("gain_db must be -24 to +24 dB")
    return copy.deepcopy(body)


def empty_body() -> dict[str, Any]:
    return {"voices": [], "characters": {}, "lines": [], "takes": {}}


def _path(state_dir: Path, project_id: str) -> Path:
    if not re.fullmatch(r"prj_[0-9a-f]{32}", project_id):
        raise VoiceStoreError("invalid_voice_state", "project id is invalid")
    return Path(state_dir) / f"{project_id}.json"


def read(state_dir: Path, project_id: str) -> dict[str, Any]:
    target = _path(state_dir, project_id)
    if not target.exists():
        return {"revision": 0, **empty_body()}
    try:
        raw = target.read_bytes()
        if len(raw) > MAX_BYTES:
            raise ValueError
        value = json.loads(raw.decode("utf-8"))
        if value.get("schema") != SCHEMA or value.get("project_id") != project_id or type(value.get("revision")) is not int:
            raise ValueError
        body = validate_body({k: value[k] for k in ("voices", "characters", "lines", "takes")})
    except (OSError, ValueError, KeyError, AttributeError) as exc:
        raise VoiceStoreError("voice_state_unavailable", "The voices of this project could not be read") from exc
    return {"revision": value["revision"], **body}


def write(state_dir: Path, project_id: str, expected_revision: int, body: Mapping[str, Any]) -> dict[str, Any]:
    clean = validate_body(body)
    with _LOCK:
        current = read(state_dir, project_id)
        if current["revision"] != expected_revision:
            raise VoiceStoreError("voice_state_conflict", f"The voices changed meanwhile (revision {current['revision']})")
        document = {"schema": SCHEMA, "project_id": project_id, "revision": expected_revision + 1, **clean}
        payload = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(payload) > MAX_BYTES:
            raise VoiceStoreError("invalid_voice_state", "Too many voices, lines or takes")
        target = _path(state_dir, project_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".voice.", suffix=".tmp", dir=str(target.parent))
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
            os.replace(name, target)
        finally:
            if os.path.exists(name):
                os.unlink(name)
    return {"revision": document["revision"], **clean}


def update(state_dir: Path, project_id: str, change: Any) -> dict[str, Any]:
    """Read, change (a function of the body), write; for backend-side edits (new takes)."""
    with _LOCK:
        current = read(state_dir, project_id)
        body = {k: copy.deepcopy(current[k]) for k in ("voices", "characters", "lines", "takes")}
        change(body)
    return write(state_dir, project_id, current["revision"], body)


def voice_of_line(body: Mapping[str, Any], line_id: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    line = next((item for item in body["lines"] if item["id"] == line_id), None)
    if line is None:
        raise VoiceStoreError("line_not_found", "That line is not in this project")
    voice_id = body["characters"].get(line["character"])
    voice = next((item for item in body["voices"] if item["id"] == voice_id), None)
    return line, voice


__all__ = ["SCHEMA", "VoiceStoreError", "empty_body", "new_id", "read", "update", "validate_body", "voice_of_line", "write"]
