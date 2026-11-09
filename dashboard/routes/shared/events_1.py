from __future__ import annotations


# --- Push events (Weg A1, docs/plans/FILM-SUITE-WEG-A.md 3b) -----------------
# Hints only: the routes stay the truth and the renderer keeps a slow poll as
# a safety net. Every announce helper below swallows its own failures, so an
# event can never change a route's answer.
_FILM_EVENTS_PATH = Path(__file__).with_name("film_events.py")
_FILM_EVENTS_SPEC = importlib.util.spec_from_file_location("film_suite_events", _FILM_EVENTS_PATH)
if _FILM_EVENTS_SPEC is None or _FILM_EVENTS_SPEC.loader is None:
    raise RuntimeError("The Continuo event helper is unavailable.")
film_events = importlib.util.module_from_spec(_FILM_EVENTS_SPEC)
_FILM_EVENTS_SPEC.loader.exec_module(film_events)


def _announce_image_jobs(changes: Any, owner_scope: Any = None) -> None:
    """``changes``: (job_id, state before or None for a new job, state after)."""
    try:
        by_reason: dict[str, list[str]] = {}
        published = False
        for job_id, before, after in changes or ():
            if before == after or not after:
                continue
            if before is None:
                reason = "submitted"
                _job_authors_note([job_id])  # lane w5-backend: the new job's author (hermes | artist)
            elif after == "published":
                reason, published = "published", True
            elif after == "cancelled":
                reason = "cancelled"
            else:
                reason = "state"
            by_reason.setdefault(reason, []).append(job_id)
        for reason, job_ids in by_reason.items():
            if reason == "submitted":  # w31-fix4 item 0 (IMG-217): Hermes' direct submits are announced as Hermes'
                film_events.jobs_changed(job_ids, reason, owner_scope, by=_film_caller())
            else:
                film_events.jobs_changed(job_ids, reason, owner_scope)
        if published:
            film_events.media_changed(owner_scope)
    except Exception:  # noqa: BLE001 - an event never fails a route
        pass


# Shoot receipt status last announced per studio job id (bounded): a poll that
# returns the same status as before stays silent.
_SHOOT_ANNOUNCED: dict[str, str] = {}


def _announce_shoot_job(result: Any, *, submitted: bool = False) -> None:
    """One Shoot receipt from /submit, /poll or /cancel; only a new status is announced."""
    try:
        details = result.get("details") if isinstance(result, dict) else None
        if not isinstance(details, dict) or type(details.get("id")) is not str or type(details.get("status")) is not str:
            return
        job_id, status = details["id"], details["status"]
        before = _SHOOT_ANNOUNCED.pop(job_id, None)
        _SHOOT_ANNOUNCED[job_id] = status
        while len(_SHOOT_ANNOUNCED) > 512:
            _SHOOT_ANNOUNCED.pop(next(iter(_SHOOT_ANNOUNCED)))
        if before == status:
            return
        project_id = details.get("project_id")
        scope = f"film_owner:{project_id}" if type(project_id) is str and project_id else None
        if submitted:
            _job_authors_note_shoot(job_id)  # lane w5-backend: the new take's author (hermes | artist)
        reason = "submitted" if submitted else {"ready": "published", "cancelled": "cancelled"}.get(status, "state")
        film_events.jobs_changed([details["id"]], reason, scope)
        if status == "ready":
            film_events.media_changed(scope)
    except Exception:  # noqa: BLE001 - an event never fails a route
        pass


@router.post("/events/ping")
async def events_ping() -> dict[str, Any]:
    """Confirm event mode: broadcast ``hello`` and name the events this backend sends.

    The body and query are ignored on purpose (the desktop adds ``?profile=``)."""
    film_events.hello()
    return {"ok": True, "v": film_events.VERSION, "events": list(film_events.EVENTS)}


film_events.hello()
