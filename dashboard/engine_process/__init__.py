"""One way to start a foreign engine python (lane p14-sam, AUD-062).

Every owned engine (song map, scene analysis, camera flow, LaMa, Beat This!, speech recognition) runs in its own
venv.  The backend itself runs with PYTHONPATH / PYTHONHOME pointing at the plugin's dependency env (another NumPy):
inherited, the engine imports the backend's packages and dies (live 01.11.: "No module named
numpy._core._multiarray_umath" on every song-map track).  Same isolation as ``uv run`` or an embedded venv: the
child sees no ``PYTHON*`` variable, no ``VIRTUAL_ENV`` / ``CONDA_PREFIX``, and runs with ``-I`` (no environment
paths, no user site, no script folder on ``sys.path``; a venv's own ``.pth`` files still apply, which is how the
song engine reaches its ``vendor/`` folder; runners that need their own folder insert it themselves).
"""
from __future__ import annotations

import os
from typing import Mapping

_DROPPED = frozenset({"VIRTUAL_ENV", "CONDA_PREFIX", "__PYVENV_LAUNCHER__"})


def engine_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The backend's environment without anything that steers a Python interpreter, plus ``extra``."""
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith("PYTHON") and key.upper() not in _DROPPED}
    env.update(extra or {})
    return env


def engine_command(python: str | os.PathLike[str], *args: str | os.PathLike[str], flags: tuple[str, ...] = ()) -> list[str]:
    """``[python, -I, *flags, *args]``: the engine's interpreter, isolated."""
    return [str(python), "-I", *flags, *(str(a) for a in args)]


__all__ = ["engine_command", "engine_env"]
