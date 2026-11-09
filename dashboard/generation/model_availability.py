"""Is a model file available? ComfyUI's own view first (lane w24-models item 2).

ComfyUI loads a file from ``models/<folder>`` AND from every folder its ``extra_model_paths.yaml``
declares (another drive included).  "Installed" therefore means "ComfyUI lists it", never "it sits
under ComfyUI/models".  The answer comes from:

1. the engine's live list per folder kind (authoritative while fresh) (loader options from ``/object_info`` or ``GET /models/<folder>``),
   recorded by the routes that already ask the engine (``record_live``), kept for ``LIVE_TTL_SECONDS`` and
   dropped as soon as the engine stops answering (``forget_live``), so a restarted ComfyUI is read anew;
2. without a fresh list (engine down, a process that has not asked yet): ``models/<folder>`` with ComfyUI's
   alias folders (unet, clip) plus the folders ``extra_model_paths.yaml`` declares, in ComfyUI's order: the
   folders ComfyUI builds its list from, so a file on another drive counts.  The submit check re-reads the
   engine's list before it refuses, so a model installed a moment ago counts without restarting Hermes
   (ComfyUI re-scans its folders for ``/object_info``).

Reading stored history never calls this module: a stored recipe names files, it does not need them.
Stdlib only besides the sibling ``model_catalog`` (relative import, or by file path for the agent tools).
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

if __package__:
    from .model_catalog import model_roots as _model_roots
else:  # loaded by file path (the agent tools)
    import importlib.util as _util
    _spec = _util.spec_from_file_location("film_suite_availability_model_catalog", Path(__file__).resolve().with_name("model_catalog.py"))
    _module = _util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    _model_roots = _module.model_roots

LIVE_TTL_SECONDS = 30.0
# Render-spec file role -> ComfyUI folder kind.
ROLE_KINDS = {"diffusion_model": "diffusion_models", "text_encoder": "text_encoders", "vae": "vae", "lora": "loras"}
_ALIASES = {"diffusion_models": ("diffusion_models", "unet"), "text_encoders": ("text_encoders", "clip")}
_LOCK = threading.Lock()
_LIVE: dict[str, tuple[float, frozenset[str]]] = {}


def _norm(name: str) -> str:
    return name.replace("\\", "/")


def record_live(lists: Mapping[str, Iterable[str]], *, ttl: float = LIVE_TTL_SECONDS) -> None:
    """Store the engine's file names per folder kind (``diffusion_models``, ``loras``, ...)."""
    until = time.monotonic() + ttl
    with _LOCK:
        for kind, names in lists.items():
            _LIVE[kind] = (until, frozenset(_norm(n) for n in names if isinstance(n, str)))


def forget_live() -> None:
    """The engine did not answer: its next list is read fresh (restart, other model folders)."""
    with _LOCK:
        _LIVE.clear()


def live_names(kind: str) -> frozenset[str] | None:
    with _LOCK:
        entry = _LIVE.get(kind)
    if entry is None or time.monotonic() >= entry[0]:
        return None
    return entry[1]


def local_roots(kind: str, models_root: Path) -> tuple[Path, ...]:
    """Where ComfyUI looks for ``kind``: models/<kind> (+ aliases), then extra_model_paths.yaml.

    ``models_root`` is the models folder itself; the yaml sits next to it (ComfyUI's root).
    """
    models_root = Path(models_root)
    comfy = models_root.parent
    found = _model_roots(comfy)
    defaults = [models_root / sub for sub in _ALIASES.get(kind, (kind,))]
    extra = [root for root in found.get(kind, ()) if root not in [comfy / "models" / sub for sub in _ALIASES.get(kind, (kind,))]]
    return tuple(dict.fromkeys([*defaults, *extra]))


def local_file(kind: str, name: str, models_root: Path) -> Path | None:
    """The first regular file named ``name`` in ComfyUI's folder order, else None."""
    parts = _norm(name).split("/")
    if any(part in ("", ".", "..") for part in parts):
        return None
    for root in local_roots(kind, models_root):
        path = Path(root).joinpath(*parts)
        try:
            if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(Path(root).resolve()):
                return path
        except OSError:
            continue
    return None


def present(kind: str, name: Any, models_root: Path, *, live: bool = True) -> bool:
    """True when ComfyUI can load ``name`` from folder kind ``kind``."""
    if type(name) is not str or not name:
        return False
    names = live_names(kind) if live else None
    if names is not None:
        return _norm(name) in names  # the running engine's own list decides
    # No fresh list (engine down, a process that has not asked yet): the folders ComfyUI builds its list from.
    return local_file(kind, name, models_root) is not None


def folder_of(role: str) -> str:
    return ROLE_KINDS[role]


__all__ = ["LIVE_TTL_SECONDS", "ROLE_KINDS", "folder_of", "forget_live", "live_names", "local_file", "local_roots",
           "present", "record_live"]
