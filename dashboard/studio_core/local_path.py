"""The one lexical guard against remote and device paths (lane w34-sec, scan findings 1-7, 11, 12, 13).

On Windows every UNC spelling (``\\\\host\\share``, ``//host/share``, the mixed ``/\\host`` and ``\\/host``) and every
device or NT spelling (``\\\\?\\``, ``\\\\.\\``, ``//?/``, ``\\??\\``) makes ``realpath``/``stat``/``exists``/``open``
contact a host or a device before any containment check can answer, and an SMB host is sent the user's NTLM
authentication. A path text that names project data (a manifest root, a chain guide, a control-layer index, a render
queue folder, an output folder, a music tool path, a video-step source, an upscale project folder) is therefore checked
HERE, on its text only, before any filesystem call. A local drive path (``C:\\...``, ``C:/...``) or a rooted POSIX path
passes. This module is stdlib only and touches nothing: ``path_boundary.py``, ``approved_roots.py``,
``routes/shared/paths_1.py``, ``control/extract.py`` and the plugin's ``__init__.py`` load it by file path.
"""
from __future__ import annotations

import ntpath
import re
from typing import Any

REMOTE_PATH_MESSAGE = "network (UNC) and device paths are not local paths"
_LOCAL_DRIVE_RE = re.compile(r"^[A-Za-z]:$")


def remote_path_reason(value: Any) -> str | None:
    """None for a path text that may be used locally; else why it is refused. Never touches the filesystem."""
    if not isinstance(value, str):
        return "path must be text"
    text = value.strip()
    if "\x00" in text:
        return "path contains forbidden characters"
    unified = text.replace("/", "\\")
    # Two leading separators in any '/' '\\' mix: UNC (\\host\share), extended (\\?\), device (\\.\).
    if unified.startswith("\\\\"):
        return REMOTE_PATH_MESSAGE
    # The NT object namespace (\??\UNC\host\share, \??\C:\...) is passed through by Win32 as written.
    if unified.startswith("\\??\\"):
        return REMOTE_PATH_MESSAGE
    drive, _tail = ntpath.splitdrive(text)
    if drive and not _LOCAL_DRIVE_RE.fullmatch(drive):
        return REMOTE_PATH_MESSAGE
    return None


def is_local_path_text(value: Any) -> bool:
    return remote_path_reason(value) is None
