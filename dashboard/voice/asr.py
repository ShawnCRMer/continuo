"""Local speech recognition for voice clips and takes (lane fs-voice).

The Suite has no ASR route of its own; ComfyUI's Qwen3-ASR loader has no weights
installed.  faster-whisper (CTranslate2) is installed in the Hermes agent venv
with Systran's converted models cached by Hugging Face (large-v3, medium, base):
this module runs it in a child process with ``local_files_only=True`` (never a
download), on the CPU (int8) so the GPU stays the artist's.  Nothing is invented:
a failure answers ``None`` and the words stay empty and editable.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# One way to start an engine python (lane p14-sam, AUD-062): dashboard/engine_process, loaded by path like its siblings.
_engine_process_spec = importlib.util.spec_from_file_location(
    "film_suite_engine_process", Path(__file__).resolve().parents[1] / "engine_process" / "__init__.py")
engine_process = importlib.util.module_from_spec(_engine_process_spec)
_engine_process_spec.loader.exec_module(engine_process)

_SCRIPT = r"""
import json, sys
from faster_whisper import WhisperModel
path, model, words = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
m = WhisperModel(model, device="cpu", compute_type="int8", local_files_only=True)
segments, info = m.transcribe(path, beam_size=5, word_timestamps=words, vad_filter=False)
out = {"language": info.language, "text": "", "words": []}
parts = []
for s in segments:
    parts.append(s.text.strip())
    for w in (s.words or []):
        out["words"].append({"w": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3)})
out["text"] = " ".join(p for p in parts if p)
print(json.dumps(out, ensure_ascii=False))
"""


def python_with_whisper() -> str | None:
    """A Python that imports faster_whisper: this one, else the Hermes agent venv."""
    candidates = [sys.executable]
    root = os.environ.get("HERMES_AGENT_ROOT") or str(Path(os.environ.get("LOCALAPPDATA", "")) / "hermes" / "hermes-agent")
    candidates.append(str(Path(root) / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")))
    # w18-amap: the backend itself runs from the Hermes agent root (hermes_cli); its venv is there even when
    # LOCALAPPDATA points elsewhere (a separate home) and HERMES_AGENT_ROOT is not set.
    try:
        import importlib.util as _iu
        spec = _iu.find_spec("hermes_cli")
        if spec is not None and spec.origin:
            candidates.append(str(Path(spec.origin).resolve().parents[1] / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")))
    except (ImportError, ValueError):
        pass
    for candidate in candidates:
        if not candidate or not Path(candidate).is_file():
            continue
        try:
            done = subprocess.run(engine_process.engine_command(candidate, "-c", "import faster_whisper"), capture_output=True,
                                  timeout=60, check=False, env=engine_process.engine_env())
        except (OSError, subprocess.TimeoutExpired):
            continue
        if done.returncode == 0:
            return candidate
    return None


def transcribe(path: str | os.PathLike[str], *, model: str = "large-v3", words: bool = False, timeout: float = 600.0) -> dict[str, Any] | None:
    """{language, text, words[{w,start,end}]} or None when no local recognizer could run."""
    python = python_with_whisper()
    if python is None:
        return None
    # Isolated like every engine (engine_process): the backend's PYTHONPATH must not reach the recognizer's venv;
    # -X utf8 because -I ignores PYTHONIOENCODING and the words are read back as UTF-8.
    env = engine_process.engine_env({"HF_HUB_OFFLINE": "1"})
    try:
        done = subprocess.run(engine_process.engine_command(python, "-c", _SCRIPT, path, model, "1" if words else "0", flags=("-X", "utf8")),
                              capture_output=True, timeout=timeout, check=False, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0:
        return None
    try:
        lines = [line for line in done.stdout.decode("utf-8", "replace").splitlines() if line.startswith("{")]
        value = json.loads(lines[-1])
    except (ValueError, IndexError):
        return None
    if not isinstance(value, dict) or not isinstance(value.get("text"), str):
        return None
    return value


# ---------------------------------------------------------------- songs (lane w18-amap, AUD-135)
# The words of a sung vocal stem without pasted lyrics, as Premiere's text-based editing and Descript show a
# transcript: every word at its time, grouped into lines (whisper segments).  Silero VAD v6 decides first whether
# anything is sung (the copy faster-whisper ships: engines/silero-vad holds v6.2, whose stateful ONNX inputs
# (state, sr) the faster-whisper wrapper cannot feed); nothing sung answers speech_s ~ 0 and the large model is
# never loaded.
SONG_MODEL = "large-v3"
SONG_MIN_SPEECH_S = 1.0

_SONG_SCRIPT = r"""
import json, os, sys
import faster_whisper
from faster_whisper import WhisperModel, decode_audio
from faster_whisper import vad as fw_vad
path, model = sys.argv[1], sys.argv[2]
vad_model = "silero-vad-v6 (faster-whisper " + faster_whisper.__version__ + ")"
audio = decode_audio(path, sampling_rate=16000)
spans = fw_vad.get_speech_timestamps(audio, fw_vad.VadOptions(min_silence_duration_ms=500), sampling_rate=16000)
speech_s = round(sum(s["end"] - s["start"] for s in spans) / 16000.0, 3)
out = {"version": faster_whisper.__version__, "model": model, "vad": vad_model, "speech_s": speech_s,
       "duration_s": round(len(audio) / 16000.0, 3), "language": None, "segments": []}
if speech_s >= float(sys.argv[3]):
    m = WhisperModel(model, device="cpu", compute_type="int8", local_files_only=True)
    segments, info = m.transcribe(audio, beam_size=5, word_timestamps=True, vad_filter=True,
                                  vad_parameters={"min_silence_duration_ms": 500}, condition_on_previous_text=False)
    out["language"] = info.language
    for s in segments:
        if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:
            continue
        words = [{"w": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3), "p": round(w.probability, 3)}
                 for w in (s.words or []) if w.word.strip()]
        if not words:
            continue
        out["segments"].append({"text": s.text.strip(), "start": round(s.start, 3), "end": round(s.end, 3),
                                "avg_logprob": round(s.avg_logprob, 3), "words": words})
print(json.dumps(out, ensure_ascii=False))
"""


def whisper_version(python: str | None = None) -> str | None:
    """The faster-whisper version the song transcriber would run (part of the song map's cache key)."""
    python = python or python_with_whisper()
    if python is None:
        return None
    try:
        done = subprocess.run(engine_process.engine_command(python, "-c", "import faster_whisper; print(faster_whisper.__version__)"),
                              capture_output=True, timeout=60, check=False, env=engine_process.engine_env())
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = done.stdout.decode("utf-8", "replace").strip().splitlines()
    return text[-1].strip() if done.returncode == 0 and text else None


def song_transcriber_id(version: str | None) -> str | None:
    return f"faster-whisper-{version}/{SONG_MODEL}/cpu-int8" if version else None


def transcribe_song(path: str | os.PathLike[str], *, model: str = SONG_MODEL, timeout: float = 1800.0) -> dict[str, Any] | None:
    """{version, model, vad, speech_s, duration_s, language, segments[{text,start,end,words[{w,start,end,p}]}]} or None."""
    python = python_with_whisper()
    if python is None:
        return None
    env = engine_process.engine_env({"HF_HUB_OFFLINE": "1"})
    try:
        done = subprocess.run(engine_process.engine_command(python, "-c", _SONG_SCRIPT, str(path), model, str(SONG_MIN_SPEECH_S), flags=("-X", "utf8")),
                              capture_output=True, timeout=timeout, check=False, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0:
        return None
    try:
        lines = [line for line in done.stdout.decode("utf-8", "replace").splitlines() if line.startswith("{")]
        value = json.loads(lines[-1])
    except (ValueError, IndexError):
        return None
    if not isinstance(value, dict) or not isinstance(value.get("segments"), list):
        return None
    return value


__all__ = ["python_with_whisper", "transcribe", "transcribe_song", "whisper_version", "song_transcriber_id"]
