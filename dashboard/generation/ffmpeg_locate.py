"""Where ffmpeg/ffprobe are (w20-cat, CATALOG.md #23: no hard-coded C:/ffmpeg path).

Order: the explicit environment override (``FLUX3_FFMPEG`` / ``FLUX3_FFPROBE``, then ``FFMPEG_PATH``'s folder), then
``PATH`` (``shutil.which``), then the conventional Windows folder ``<SystemDrive>\\ffmpeg\\bin`` when it holds the
tool. Without any of them the bare name is returned: the subprocess call fails with "not found" and the music and
render routes report ffmpeg as unavailable, as before on a machine without ffmpeg.

Standalone (stdlib only): loaded by path from music_analysis.py, routes/shared/constants_1.py, timeline_export and
the publishing module.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

_OVERRIDES = {"ffmpeg": "FLUX3_FFMPEG", "ffprobe": "FLUX3_FFPROBE"}


def _exe(name: str) -> str:
    return name + (".exe" if os.name == "nt" else "")


def locate(name: str) -> str:
    """The path of ``ffmpeg`` or ``ffprobe`` by the order above, as a string."""
    if name not in _OVERRIDES:
        raise ValueError(name)
    override = os.environ.get(_OVERRIDES[name])
    if override and override.strip():
        return override.strip()
    configured = os.environ.get("FFMPEG_PATH")
    if configured and configured.strip():
        return str(Path(configured.strip()).with_name(_exe(name)))
    found = shutil.which(name)
    if found:
        return found
    if os.name == "nt":
        drive = os.environ.get("SystemDrive") or ""
        if drive:
            candidate = Path(drive + "\\") / "ffmpeg" / "bin" / _exe(name)
            if candidate.is_file():
                return str(candidate)
    return _exe(name)

