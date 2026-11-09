from __future__ import annotations

import contextvars

# --- Who did it: Hermes or the artist (lane w5-backend items 1-2) -------------------------------------
#
# A Hermes tool call runs the very routes the page calls (agent_tools.py: in-process ASGI). agent_tools
# sets _FILM_CALLER to "hermes" around its dispatch; every page request leaves the default "artist".
# The submit paths note the caller per NEW job id (the image announce of a new row, the Shoot announce of
# a submitted take) in one small private SQLite table under the Hermes state folder; /jobs/list answers it
# as ``by``. A job without a note (sent before this existed) reads "artist". The first note wins: a replay
# of the same request by the other side never takes a job over. The takes of a Generate x N batch that the
# backend submits later inherit the author of the batch's first take (same client request id stem).
# Notes never fail a route: every helper swallows its own errors.

_FILM_CALLER = contextvars.ContextVar("film_suite_caller", default="artist")
_JOB_AUTHORS_MAX = 20_000
_JOB_AUTHORS_LOCK = _thread.allocate_lock()


def _film_caller() -> str:
    """"hermes" inside an agent tool call, else "artist"."""
    return "hermes" if _FILM_CALLER.get() == "hermes" else "artist"


def _job_authors_path() -> Path:
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    return Path(state_root(Path(base) / "hermes")) / "job-authors.sqlite3"


def _job_authors_connect() -> sqlite3.Connection:
    path = _job_authors_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=5)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS job_authors (job_id TEXT PRIMARY KEY, by TEXT NOT NULL CHECK (by IN ('artist', 'hermes')), at REAL NOT NULL)"
    )
    return connection


def _job_authors_note(job_ids: Any, by: str | None = None) -> None:
    try:
        who = by if by in ("artist", "hermes") else _film_caller()
        ids = [job_id for job_id in (job_ids or ()) if type(job_id) is str and 0 < len(job_id) <= 160]
        if not ids:
            return
        now = time.time()
        with _JOB_AUTHORS_LOCK:
            connection = _job_authors_connect()
            try:
                with connection:
                    connection.executemany("INSERT OR IGNORE INTO job_authors (job_id, by, at) VALUES (?, ?, ?)", [(job_id, who, now) for job_id in ids])
                    (count,) = connection.execute("SELECT COUNT(*) FROM job_authors").fetchone()
                    if count > _JOB_AUTHORS_MAX:
                        connection.execute(
                            "DELETE FROM job_authors WHERE job_id IN (SELECT job_id FROM job_authors ORDER BY at ASC LIMIT ?)",
                            (count - _JOB_AUTHORS_MAX,),
                        )
            finally:
                connection.close()
    except Exception:  # noqa: BLE001 - a note never fails a route
        pass


def _job_authors_read(job_ids: Any) -> dict[str, str]:
    """job id -> "artist" | "hermes" for the ids that carry a note."""
    ids = [job_id for job_id in (job_ids or ()) if type(job_id) is str and job_id]
    found: dict[str, str] = {}
    if not ids or not _job_authors_path().is_file():
        return found
    try:
        with _JOB_AUTHORS_LOCK:
            connection = _job_authors_connect()
            try:
                for start in range(0, len(ids), 400):
                    chunk = ids[start:start + 400]
                    rows = connection.execute(
                        f"SELECT job_id, by FROM job_authors WHERE job_id IN ({','.join('?' * len(chunk))})", chunk
                    ).fetchall()
                    found.update({job_id: who for job_id, who in rows})
            finally:
                connection.close()
    except Exception:  # noqa: BLE001 - unknown reads as the artist
        return {}
    return found


def _job_authors_note_shoot(studio_job_id: Any) -> None:
    """A submitted Shoot take: its author, by take id and by client request id (batch stem inherited)."""
    try:
        if type(studio_job_id) is not str or not studio_job_id:
            return
        crid = None
        try:
            crid = getattr(_read_ledger_row(studio_job_id), "client_request_id", None)
        except Exception:  # noqa: BLE001 - a row we cannot read is noted by id only
            crid = None
        who = None
        if type(crid) is str and ":take-" in crid:
            who = _job_authors_read(["crid:" + crid.split(":take-")[0]]).get("crid:" + crid.split(":take-")[0])
        keys = [studio_job_id] + (["crid:" + crid] if type(crid) is str and crid else [])
        _job_authors_note(keys, who)
    except Exception:  # noqa: BLE001
        pass


def _job_authors_apply(jobs: list[dict[str, Any]]) -> None:
    authors = _job_authors_read([job.get("job_id") for job in jobs])
    for job in jobs:
        job["by"] = authors.get(job.get("job_id"), "artist")
