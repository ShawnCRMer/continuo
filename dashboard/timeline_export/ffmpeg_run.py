"""ffmpeg process runner with progress, cancellation and a finite error."""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Sequence

from .errors import ExportError

ProgressFn = Callable[[int], None]
CancelFn = Callable[[], bool]


class Cancelled(ExportError):
    def __init__(self) -> None:
        super().__init__("cancelled", "The export was cancelled.")


def run_ffmpeg(
    command: Sequence[str],
    *,
    on_frame: ProgressFn | None = None,
    should_cancel: CancelFn | None = None,
    timeout: float = 3600.0,
    what: str = "ffmpeg",
) -> None:
    """Run ffmpeg with ``-progress pipe:1``; report ``frame=`` lines; kill on cancel."""
    args = [str(command[0]), "-hide_banner", "-nostdin", "-loglevel", "error", "-progress", "pipe:1", *map(str, command[1:])]
    try:
        proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except FileNotFoundError as exc:
        raise ExportError("ffmpeg_unavailable", "ffmpeg is not available at the configured path.") from exc
    stderr_chunks: list[str] = []

    def drain() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            if sum(map(len, stderr_chunks)) < 16_000:
                stderr_chunks.append(line)

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout
    cancelled = False
    assert proc.stdout is not None
    for line in proc.stdout:
        if should_cancel is not None and should_cancel():
            cancelled = True
            proc.kill()
            break
        if time.monotonic() > deadline:
            proc.kill()
            proc.wait()
            raise ExportError("ffmpeg_timeout", f"{what} timed out.")
        if on_frame is not None and line.startswith("frame="):
            value = line.split("=", 1)[1].strip()
            if value.isdigit():
                on_frame(int(value))
    proc.wait()
    reader.join(timeout=5)
    if cancelled:
        raise Cancelled()
    if proc.returncode != 0:
        tail = "".join(stderr_chunks).strip().splitlines()[-3:]
        raise ExportError("ffmpeg_failed", f"{what} failed.", {"stderr": tail})


def ffmpeg_version(ffmpeg: Path) -> str:
    try:
        proc = subprocess.run([str(ffmpeg), "-version"], capture_output=True, text=True, encoding="utf-8", timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    first = proc.stdout.splitlines()[0] if proc.stdout else ""
    return first[:120] or "unknown"
