"""Film Suite push events (Weg A1, contract: docs/plans/FILM-SUITE-WEG-A.md 3b).

Events are hints that something changed; the existing routes stay the truth
and the renderer keeps a slow safety-net poll. Every helper here therefore:

* never raises into its caller (a route, the preview reader, an agent tool);
* is a no-op when no Hermes gateway runs in this process (tests, CLI tools,
  ``hermes plugins validate``) - the host import happens lazily, only once the
  gateway module is already loaded, so importing this file never pulls in the
  transport stack;
* sends only small, path-free payloads: ids, states, counters. Every payload
  carries ``v: 1`` and ``at`` (epoch ms). ``job_ids`` is capped at 50 and
  ``preview.frame`` is throttled to <= 8 per second per job.

Tests replace ``_send`` to record what would have been broadcast.
"""
from __future__ import annotations

import re
import sys
import threading
import time
from typing import Any, Callable, Iterable

PLUGIN_ID = "film-suite"
VERSION = 1
EVENTS = ("hello", "jobs.changed", "media.changed", "bridge.action", "preview.frame", "timeline.changed", "script.changed", "shoot.changed",
          "project.changed")  # lane p6-name: rename (display name), by hand or by Hermes
TIMELINE_AUTHORS = frozenset({"artist", "hermes"})
JOB_REASONS = frozenset({"submitted", "state", "published", "cancelled", "kept"})
SURFACES = frozenset({"image", "music", "suite"})
MAX_JOB_IDS = 50
MAX_SCRIPT_SCENES = 50
PREVIEW_MAX_PER_SECOND = 8
_PREVIEW_MIN_INTERVAL = 1.0 / PREVIEW_MAX_PER_SECOND
_PREVIEW_TRACKED_MAX = 256
# Ids, scopes and states only: no slashes, backslashes, spaces or drive colons
# followed by a separator can pass, so a path can never ride along.
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.:\-]{0,127}$")
_PATHLIKE_RE = re.compile(r"^[A-Za-z]:[\\/]")

_clock: Callable[[], float] = time.monotonic
_lock = threading.Lock()
# prompt/job key -> (last sent monotonic time, last sent state)
_preview_sent: dict[str, tuple[float, str]] = {}


def _token(value: Any) -> str | None:
    if type(value) is not str or not _TOKEN_RE.match(value) or _PATHLIKE_RE.match(value):
        return None
    return value


def _send(event: str, payload: dict[str, Any]) -> bool:
    """Hand one event to Hermes; False when nobody in this process can deliver it."""
    if not any(name == "tui_gateway" or name.startswith("tui_gateway.") for name in tuple(sys.modules)):
        # No gateway in this process (CLI chat, cron, tests): there is nobody
        # to deliver to, and importing tui_gateway.server just to drop the
        # event would redirect stdout as an import side effect. A
        # turn-isolation compute host has tui_gateway loaded and relays.
        return False
    try:
        from hermes_cli.plugin_events import broadcast_plugin_event
    except Exception:  # noqa: BLE001 - older Hermes without the event door
        return False
    broadcast_plugin_event(PLUGIN_ID, event, payload)
    return True


def emit(event: str, payload: dict[str, Any] | None = None) -> bool:
    """Broadcast one contract event; never raises, returns whether it was handed over."""
    try:
        if event not in EVENTS:
            return False
        body = dict(payload or {})
        body["v"] = VERSION
        body["at"] = int(time.time() * 1000)
        return bool(_send(event, body))
    except Exception:  # noqa: BLE001 - an event can never fail its caller
        return False


def hello() -> bool:
    return emit("hello", {"events": list(EVENTS)})


def jobs_changed(job_ids: Iterable[Any], reason: str, owner_scope: Any = None, by: Any = None) -> bool:
    try:
        if reason not in JOB_REASONS:
            return False
        ids: list[str] = []
        for value in job_ids or ():
            token = _token(value)
            if token is not None and token not in ids:
                ids.append(token)
            if len(ids) >= MAX_JOB_IDS:
                break
        if not ids:
            return False
        payload: dict[str, Any] = {"job_ids": ids, "reason": reason}
        scope = _token(owner_scope)
        if scope is not None:
            payload["owner_scope"] = scope
        # w31-fix4 item 0 (IMG-217): a submit names its sender, so an open Image page shows a Hermes job at once.
        if by in TIMELINE_AUTHORS:
            payload["by"] = by
        return emit("jobs.changed", payload)
    except Exception:  # noqa: BLE001
        return False


def media_changed(owner_scope: Any = None) -> bool:
    scope = _token(owner_scope)
    return emit("media.changed", {"owner_scope": scope} if scope is not None else {})


def timeline_changed(project_id: Any, revision: Any, by: Any) -> bool:
    """A timeline write landed (/projects/timeline/write or /apply): project, new revision, author."""
    project = _token(project_id)
    if project is None or type(revision) is not int or revision < 1 or by not in TIMELINE_AUTHORS:
        return False
    return emit("timeline.changed", {"project_id": project, "revision": revision, "by": by})


def shoot_changed(project_id: Any, by: Any) -> bool:
    """The Shoot (Video page) state of a project was written (PUT /projects/state, the page's save or
    Hermes' continuo_project_state_write): project and author. The page's History lists Hermes' changes."""
    project = _token(project_id)
    if project is None or by not in TIMELINE_AUTHORS:
        return False
    return emit("shoot.changed", {"project_id": project, "by": by})


def script_changed(project_id: Any, revision: Any, by: Any, scene_ids: Iterable[Any] = ()) -> bool:
    """A script write landed (every /projects/script/* write route): project, new revision,
    author and the scene ids it touched (at most 50; empty = the order, the title page or the
    whole script). The open Script page reads again; its own writes do not bounce."""
    try:
        project = _token(project_id)
        if project is None or type(revision) is not int or revision < 1 or by not in TIMELINE_AUTHORS:
            return False
        ids: list[str] = []
        for value in scene_ids or ():
            token = _token(value)
            if token is not None and token not in ids:
                ids.append(token)
            if len(ids) >= MAX_SCRIPT_SCENES:
                break
        return emit("script.changed", {"project_id": project, "revision": revision, "by": by, "scenes": ids})
    except Exception:  # noqa: BLE001
        return False


def bridge_action(surface: str, window_id: Any, action_id: Any) -> bool:
    window, action = _token(window_id), _token(action_id)
    if surface not in SURFACES or window is None or action is None:
        return False
    return emit("bridge.action", {"surface": surface, "window_id": window, "action_id": action})


def preview_frame(prompt_id: Any, rev: Any, state: Any, image_producer_job_id: Any = None) -> bool:
    """A new live step or a preview state change; frames are throttled per job.

    A state change (waiting -> running -> done ...) always passes; further
    frames of the same state pass at most ``PREVIEW_MAX_PER_SECOND`` times a
    second. The picture itself is never sent - the renderer reads it from
    ``/image/preview/live`` only when it is visible.
    """
    try:
        prompt = _token(prompt_id)
        status = _token(state)
        if prompt is None or status is None or type(rev) is not int or rev < 0:
            return False
        job = _token(image_producer_job_id)
        key = job or prompt
        now = _clock()
        with _lock:
            last = _preview_sent.get(key)
            if last is not None and last[1] == status and now - last[0] < _PREVIEW_MIN_INTERVAL:
                return False
            _preview_sent[key] = (now, status)
            if len(_preview_sent) > _PREVIEW_TRACKED_MAX:
                for old in sorted(_preview_sent, key=lambda k: _preview_sent[k][0])[: len(_preview_sent) - _PREVIEW_TRACKED_MAX]:
                    _preview_sent.pop(old, None)
        payload: dict[str, Any] = {"prompt_id": prompt, "rev": rev, "state": status}
        if job is not None:
            payload["image_producer_job_id"] = job
        return emit("preview.frame", payload)
    except Exception:  # noqa: BLE001
        return False


def project_changed(project_id: Any, name: Any, by: Any) -> bool:
    """The project's display name changed (POST /projects/rename, the page or Hermes' continuo_project_rename)."""
    project = _token(project_id)
    if project is None or by not in TIMELINE_AUTHORS or not isinstance(name, str) or not name.strip():
        return False
    return emit("project.changed", {"project_id": project, "name": name.strip()[:120], "by": by})


# Lane w23-subs (CUT-173): the cut's subtitle track changed (/projects/subtitles/apply or a finished transcription),
# by the page or by Hermes; the open Cut reads its subtitles again. Appended to the contract.
EVENTS = EVENTS + ("subtitles.changed",)


def subtitles_changed(project_id: Any, revision: Any, by: Any) -> bool:
    project = _token(project_id)
    if project is None or type(revision) is not int or revision < 1 or by not in TIMELINE_AUTHORS:
        return False
    return emit("subtitles.changed", {"project_id": project, "revision": revision, "by": by})
