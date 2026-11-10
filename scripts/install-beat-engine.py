#!/usr/bin/env python3
"""Install the optional Beat This! engine for Continuo's Audio analysis.

Creates the engine folder that ``dashboard/music_core/beat_engine.py`` looks for
(``engine_dir()``: ``FILM_SUITE_BEAT_ENGINE_DIR``, else ``%LOCALAPPDATA%\\hermes\\film-suite\\engines\\beat-this``,
else ``~/AppData/Local/hermes/film-suite/engines/beat-this`` on Linux and macOS), with:

- ``venv/``          its own Python environment (torch + beat_this, never the Hermes environment)
- ``final0.ckpt``    the Beat This! final0 checkpoint, checked against ``CHECKPOINT_SHA256``
- ``runner.py``      a copy of ``dashboard/music_core/beat_runner.py``
- ``engine.json``    the record ``beat_engine.locate()`` reads (schema ``studio.beat-engine/1``)

Usage:
    python scripts/install-beat-engine.py [--python PATH] [--device cpu|cuda] [--dir PATH]

``--python`` is the interpreter used to create the engine venv (3.10-3.12 recommended for torch wheels;
default: the one running this script). ``--device cpu`` installs CPU-only torch (about 0.9 GB instead of
about 4.3 GB); Continuo still runs it when no GPU is free. Uses ``uv`` when it is on PATH, else ``venv`` + pip.
Re-running is safe: an existing complete install is checked and kept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _engine_constants() -> dict[str, str]:
    """Read the constants from beat_engine.py as text, so this script needs no third-party packages."""
    import re
    src = (ROOT / "dashboard" / "music_core" / "beat_engine.py").read_text(encoding="utf-8")
    out = {}
    for name in ("CHECKPOINT", "CHECKPOINT_SHA256", "ENGINE_SCHEMA"):
        m = re.search(rf'^{name} = "([^"]+)"', src, re.M)
        if not m:
            raise SystemExit(f"cannot read {name} from beat_engine.py")
        out[name] = m.group(1)
    return out


class _Engine:
    """The parts of beat_engine this installer needs (same rules as beat_engine.engine_dir/locate)."""
    def __init__(self):
        self.__dict__.update(_engine_constants())

    @staticmethod
    def engine_dir() -> Path:
        override = os.environ.get("FILM_SUITE_BEAT_ENGINE_DIR")
        if override:
            return Path(override)
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "hermes" / "film-suite" / "engines" / "beat-this"

    def locate(self):
        try:
            record = json.loads((self.engine_dir() / "engine.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        ok = isinstance(record, dict) and record.get("schema") == self.ENGINE_SCHEMA and all(
            isinstance(record.get(k), str) for k in ("python", "runner", "checkpoint"))
        return record if ok and Path(record["python"]).is_file() and Path(record["runner"]).is_file() else None


beat_engine = _Engine()

CHECKPOINT_URL = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt"
BEAT_THIS = "beat_this @ git+https://github.com/CPJKU/beat_this@b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c"
TORCH = ("torch==2.7.1", "torchaudio==2.7.1")
EXTRAS = ("numpy", "soundfile", "soxr", "einops", "rotary-embedding-torch")
TORCH_INDEX = {"cpu": "https://download.pytorch.org/whl/cpu", "cuda": "https://download.pytorch.org/whl/cu128"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def make_venv(venv: Path, base_python: str, device: str) -> Path:
    uv = shutil.which("uv")
    py = venv_python(venv)
    if not py.is_file():
        if uv:
            run([uv, "venv", "--python", base_python, str(venv)])
        else:
            run([base_python, "-m", "venv", str(venv)])
    if uv:
        pip = [uv, "pip", "install", "--python", str(py)]
    else:
        run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
        pip = [str(py), "-m", "pip", "install"]
    run(pip + ["--index-url", TORCH_INDEX[device], *TORCH])
    run(pip + [BEAT_THIS, *EXTRAS])
    run([str(py), "-I", "-c", "import torch, beat_this; print('torch', torch.__version__)"])
    return py


def fetch_checkpoint(dest: Path) -> None:
    if dest.is_file() and sha256(dest) == beat_engine.CHECKPOINT_SHA256:
        print("checkpoint already present and verified")
        return
    part = dest.with_suffix(".part")
    print(f"downloading {CHECKPOINT_URL}", flush=True)
    with urllib.request.urlopen(CHECKPOINT_URL, timeout=120) as response, part.open("wb") as out:  # noqa: S310
        shutil.copyfileobj(response, out)
    got = sha256(part)
    if got != beat_engine.CHECKPOINT_SHA256:
        part.unlink(missing_ok=True)
        raise SystemExit(f"checkpoint sha256 mismatch: {got} (expected {beat_engine.CHECKPOINT_SHA256})")
    part.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--python", default=sys.executable, help="interpreter for the engine venv")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if shutil.which("nvidia-smi") else "cpu")
    ap.add_argument("--dir", default=None, help="engine folder (default: beat_engine.engine_dir())")
    args = ap.parse_args()

    if args.dir:
        os.environ["FILM_SUITE_BEAT_ENGINE_DIR"] = args.dir
    target = beat_engine.engine_dir()
    target.mkdir(parents=True, exist_ok=True)
    print(f"engine folder: {target}")
    if args.dir:
        print(f"note: Continuo finds this folder only when FILM_SUITE_BEAT_ENGINE_DIR={target} is set for Hermes")

    py = make_venv(target / "venv", args.python, args.device)
    ckpt = target / f"{beat_engine.CHECKPOINT}.ckpt"
    fetch_checkpoint(ckpt)
    runner = target / "runner.py"
    shutil.copyfile(ROOT / "dashboard" / "music_core" / "beat_runner.py", runner)
    record = {
        "schema": beat_engine.ENGINE_SCHEMA,
        "python": str(py),
        "runner": str(runner),
        "checkpoint": str(ckpt),
        "checkpoint_sha256": beat_engine.CHECKPOINT_SHA256,
        "versions": {"beat_this": BEAT_THIS.rsplit("@", 1)[-1][:12], "torch": TORCH[0].split("==")[1], "device": args.device},
    }
    tmp = target / "engine.json.part"
    tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
    tmp.replace(target / "engine.json")
    if beat_engine.locate() is None:
        raise SystemExit("engine.json written but beat_engine.locate() does not accept it")
    print("Beat This! engine installed. Continuo picks it as the default beat engine on the next analysis.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
