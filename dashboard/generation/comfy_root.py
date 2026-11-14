"""The one ComfyUI root of the Film Suite (lane w7-rest item 3).

``FILM_SUITE_COMFY_ROOT`` is the single source; without it the root is ComfyUI-Easy-Install's
folder in the user's Documents.  Every reader goes through ``comfy_root()``: the model catalog
(``model_catalog.py``), ``GET /models/missing`` (``routes/generation/model_sources_1.py``), the
Krea2 render spec's models folder, the scene engine's MediaPipe import and its installer.

Stdlib only, no package imports: it is loaded by relative import and by file path alike.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV = "FILM_SUITE_COMFY_ROOT"


def _desktop_app_base_path() -> Path | None:
    """macOS ComfyUI Desktop: its chosen install folder, from ~/Library/Application Support/ComfyUI/config.json."""
    import json
    try:
        data = json.loads((Path.home() / "Library" / "Application Support" / "ComfyUI" / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    base = data.get("basePath") if isinstance(data, dict) else None
    return Path(base) if isinstance(base, str) and base.strip() else None


def default_comfy_root() -> Path:
    """ComfyUI-Easy-Install's folder in Documents (the Windows default).  Elsewhere the first existing root of the
    usual installs: ComfyUI Desktop's base path, ~/Documents/ComfyUI, ~/ComfyUI, ~/comfy/ComfyUI (comfy-cli)."""
    easy_install = Path.home() / "Documents" / "ComfyUI-Easy-Install" / "ComfyUI"
    if os.name == "nt" or easy_install.is_dir():
        return easy_install
    candidates = [_desktop_app_base_path(), Path.home() / "Documents" / "ComfyUI", Path.home() / "ComfyUI",
                  Path.home() / "comfy" / "ComfyUI"]
    for candidate in candidates:
        if candidate is not None and (candidate / "models").is_dir():
            return candidate
    return easy_install


def comfy_root(fallback: str | os.PathLike[str] | None = None) -> Path:
    """``FILM_SUITE_COMFY_ROOT``, else ``fallback`` (a root recorded at install time), else the default."""
    configured = (os.environ.get(ENV) or "").strip()
    if configured:
        return Path(configured)
    if fallback:
        return Path(fallback)
    return default_comfy_root()


__all__ = ["ENV", "comfy_root", "default_comfy_root"]
