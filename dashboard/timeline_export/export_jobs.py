"""In-process export jobs: start, status with long-poll, cancel.

Same shape as the Image mask jobs (``generation/image_mask_jobs.py``
``MaskJobService``: ``start`` -> ``running``, ``wait(ms)`` long-poll, ``peek``),
with three differences an export needs: a stable job id the caller polls, a
cancel flag the ffmpeg runner checks per progress line, and one export at a
time (ffmpeg shares the machine with ComfyUI) so later jobs wait their turn.

The same request (key) while a job for it is waiting or running returns that
job instead of starting a second one.

Render queue (lane w2-cutdel): jobs run strictly in the order they were started (a FIFO turn, not
a lock any waiting thread may win), each carries its queue ``number``; ``queue`` lists them in order
with a ``position`` for the waiting ones, ``remove`` takes a waiting job out (it never runs) and
``clear_finished`` drops done / failed / cancelled jobs from the list.

Push (lane w4-cutdel): every change of the queue is announced as ``jobs.changed`` through the suite's
event channel (``dashboard/film_events.py``): ``submitted`` when a job is added, ``state`` when it starts,
finishes or fails or finished jobs are cleared, ``cancelled`` when it is stopped or removed. The scope is
``film_owner:<project_id>``; ids only, never a path. An event never fails or delays the queue.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .errors import ExportError

_EVENTS: dict[str, Any] = {}


def _film_events() -> Any:
    """The suite's event helper (dashboard/film_events.py), loaded once; None when it cannot load."""
    if "module" not in _EVENTS:
        module = None
        try:
            import importlib.util
            from pathlib import Path
            path = Path(__file__).resolve().parent.parent / "film_events.py"
            spec = importlib.util.spec_from_file_location("film_suite_export_events", path)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
        except Exception:  # noqa: BLE001 - no events, the queue still works
            module = None
        _EVENTS["module"] = module
    return _EVENTS["module"]

MAX_JOBS = 32
JOB_ID_PATTERN = r"^tlexp_[0-9a-f]{32}$"

Work = Callable[[Callable[[float, str], None], Callable[[], bool]], dict[str, Any]]


@dataclass
class _Job:
    id: str
    key: str
    kind: str
    created: float
    state: str = "waiting"  # waiting | running | done | failed | cancelled
    progress: float = 0.0
    stage: str = ""
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    task: Any = None
    number: int = 0
    label: str = ""
    project_id: str | None = None
    request: dict[str, Any] | None = None  # the full export request (Render again after a restart)
    settings: dict[str, Any] | None = None  # what renders: kind, quality, loudness, range, name, folder (path-free)
    stored: bool = False  # w20-dlv: its row was written to the project's queue store once
    removed: bool = False  # w20-dlv: another process took it out of the queue while it waited
    started_at: float = field(default_factory=time.time)  # wall clock: elapsed for another process
    written: float = 0.0  # monotonic time of the last progress write


# Lane p12-cut (CUT-006, GEM-056): the queue is written to the project (state/export-queue.json, atomic) so it survives a
# restart. Reference: Resolve's Render Queue keeps each job with its settings and range [RV-RQ]; filmcraft snapshots the
# project + resolved settings per queue item [FC-A:696-700]. A job waiting or running when the app closed comes back as
# failed / app_closed ("App closed while rendering"); its stored request renders it again.
#
# Lane w20-dlv (CUT-006, HER-047, HER-031): the store is the ONE queue of the project for every process. The artist's
# app and Hermes' tool process each hold an ExportJobs; Resolve has one render queue per project whatever started the
# job [RV-RQ]. Every row names its owner (pid + token). A process lists the store (its own jobs overlaid live), so it
# sees jobs another process started after its first look; Stop on another process's job writes cancel_requested, which
# the owner's render reads through its cancelled() callable; Remove deletes a waiting row and the owner's waiting job
# leaves; numbers come from the store, so they never collide; one job renders at a time across both processes (the
# turn is claimed inside the store lock). A row whose owner is gone is failed / app_closed. Every read and write holds
# the store's lock file (msvcrt / fcntl), so os.replace never meets a reader.
QUEUE_SCHEMA = "studio.timeline-export-queue-store/1"
APP_CLOSED = {"code": "app_closed", "message": "App closed while rendering"}
ACTIVE = frozenset({"waiting", "running"})
_POLL_S = 0.2  # how often a waiting job / a long-poll on another process's job re-reads the store
_PROGRESS_WRITE_S = 0.25  # at most one progress write per this many seconds (a stage change writes at once)


def _owner_slots() -> dict[str, str]:
    """slot -> owner token of the queue each plugin instance of THIS process uses (shared by every copy of this module)."""
    import sys
    import types
    holder = sys.modules.get("_film_suite_export_owners")
    if holder is None:
        holder = sys.modules.setdefault("_film_suite_export_owners", types.ModuleType("_film_suite_export_owners"))
    if not hasattr(holder, "slots"):
        holder.slots = {}
    return holder.slots


def _pid_alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # access denied: it exists
        try:
            code = ctypes.c_ulong()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _owner_alive(owner: Any, me: str | None = None) -> bool:
    """Is the process (queue) that wrote a row still there? In this process only the queue a plugin instance uses
    counts (a replaced queue object is a closed app); another process counts while its pid runs."""
    if type(owner) is not str or ":" not in owner:
        return False
    if owner == me:
        return True
    try:
        pid = int(owner.split(":", 1)[0])
    except ValueError:
        return False
    if pid == os.getpid():
        return owner in _owner_slots().values()
    return _pid_alive(pid)


def _project_root_exists(store: Path) -> bool:
    """Lane w22-cut (CUT-176; Resolve marks a moved project offline and never recreates its folder): the queue file
    ``<project>/state/export-queue.json`` is read or written only while its project folder exists. A moved-away folder
    is never made again by a poll, a Deliver read or a lookup; the routes bind a store only after the manifest validates."""
    return store.parent.parent.is_dir()


class _StoreLock:
    """Exclusive lock on ``<store>.lock`` across processes (blocks until it is free)."""

    def __init__(self, store: Path) -> None:
        self.path = store.with_name(store.name + ".lock")
        self.fh: Any = None

    def __enter__(self) -> "_StoreLock":
        # Lane w22-cut (CUT-176): only state/ itself, never the project folder above it (callers checked it exists).
        self.path.parent.mkdir(exist_ok=True)
        self.fh = open(self.path, "a+b")
        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    self.fh.seek(0)
                    msvcrt.locking(self.fh.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:  # LK_LOCK gave up after its own 10 s; the holder is still writing
                    continue
        else:
            import fcntl
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *_exc: Any) -> None:
        try:
            if os.name == "nt":
                import msvcrt
                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
        finally:
            self.fh.close()


class ExportJobs:
    def __init__(self, *, clock: Callable[[], float] = time.monotonic, events: Any = None) -> None:
        self._events = events
        self._jobs: "OrderedDict[str, _Job]" = OrderedDict()  # the jobs THIS process renders
        self._lock = threading.Lock()
        self._turn = threading.Condition(self._lock)  # one export at a time, in start order
        self._running: str | None = None
        self._numbers = 0
        self._clock = clock
        self._stores: dict[str, Path] = {}  # project_id -> its state/export-queue.json
        self._owner = f"{os.getpid()}:{uuid.uuid4().hex[:12]}"

    def claim_slot(self, slot: str) -> None:
        """This object is the queue of plugin instance ``slot`` in this process (an older one there is closed)."""
        _owner_slots()[slot] = self._owner

    # --- the store ---------------------------------------------------------------------------------------------
    def _read_store(self, store: Path) -> tuple[list[dict[str, Any]], int]:
        try:
            saved = json.loads(store.read_text(encoding="utf-8"))
            rows = saved["jobs"] if saved.get("schema") == QUEUE_SCHEMA and isinstance(saved.get("jobs"), list) else []
            last = int(saved.get("last_number") or 0)
        except (OSError, ValueError, TypeError, AttributeError):
            rows, last = [], 0
        rows = [row for row in rows if isinstance(row, dict) and isinstance(row.get("job_id"), str)]
        return rows, max([last, *(int(row.get("number") or 0) for row in rows)])

    def _closed(self, row: dict[str, Any]) -> bool:
        """An active row whose owner is gone (not one of this process's live jobs)."""
        if row.get("state") not in ACTIVE:
            return False
        if row.get("owner") == self._owner:
            return row["job_id"] not in self._jobs
        return not _owner_alive(row.get("owner"), self._owner)

    def _normal(self, rows: list[dict[str, Any]]) -> bool:
        changed = False
        for row in rows:
            if self._closed(row):
                row.update(state="failed", error=dict(APP_CLOSED), cancel_requested=False)
                changed = True
        return changed

    def _row(self, job: _Job, old: dict[str, Any] | None) -> dict[str, Any]:
        return {"job_id": job.id, "key": job.key, "kind": job.kind, "state": job.state, "number": job.number, "label": job.label,
                "progress": round(job.progress, 4), "stage": job.stage, "result": job.result, "error": job.error,
                "request": job.request, "settings": job.settings, "owner": self._owner, "started_at": job.started_at,
                "cancel_requested": bool(old and old.get("cancel_requested")) and job.state in ACTIVE}

    def _write(self, store: Path, rows: list[dict[str, Any]], last: int) -> None:
        rows = sorted(rows, key=lambda row: int(row.get("number") or 0))
        while len(rows) > MAX_JOBS:
            old = next((row for row in rows if row.get("state") not in ACTIVE), None)
            if old is None:
                break
            rows.remove(old)
        try:
            temp = store.with_name(store.name + ".tmp")
            temp.write_text(json.dumps({"schema": QUEUE_SCHEMA, "last_number": last, "jobs": rows}, ensure_ascii=False, default=str),
                            encoding="utf-8")
            os.replace(temp, store)
        except OSError:
            pass  # the queue still works in memory; the next change writes again

    def _txn(self, project_id: str | None, fn: Callable[[list[dict[str, Any]], dict[str, int]], Any]) -> Any:
        """Read-modify-write one project's store under its lock; this process's jobs are written as they are now.
        Call with self._lock held. ``fn(rows, meta)`` may change rows / meta["last"]; its answer is returned."""
        store = self._stores.get(project_id or "")
        if store is None or not _project_root_exists(store):
            # Lane w22-cut (CUT-176): a project folder that is gone reads like no store: no file, no folder.
            return fn([], {"last": self._numbers})
        with _StoreLock(store):
            rows, last = self._read_store(store)
            before = (json.dumps(rows, sort_keys=True, default=str), last)
            meta = {"last": max(last, self._numbers)}
            self._sync_own(project_id, rows)
            self._normal(rows)
            out = fn(rows, meta)
            self._numbers = max(self._numbers, meta["last"])
            self._sync_own(project_id, rows)
            if (json.dumps(rows, sort_keys=True, default=str), meta["last"]) != before or not store.is_file():
                self._write(store, rows, meta["last"])
        return out

    def _sync_own(self, project_id: str | None, rows: list[dict[str, Any]]) -> None:
        """Put this process's jobs of the project into ``rows``; a stored own job whose row another process deleted
        (Remove / Clear finished) leaves here too (a waiting one is cancelled and never runs)."""
        by_id = {row["job_id"]: n for n, row in enumerate(rows)}
        for job in list(self._jobs.values()):
            if job.project_id != project_id:
                continue
            if job.id in by_id:
                old = rows[by_id[job.id]]
                if old.get("cancel_requested") and job.state in ACTIVE:
                    job.cancel.set()
                rows[by_id[job.id]] = self._row(job, old)
                job.stored = True
            elif job.stored and job.state != "running":
                if job.state == "waiting":
                    job.removed = True
                    job.cancel.set()
                self._jobs.pop(job.id, None)
            elif job.removed:
                continue
            else:
                rows.append(self._row(job, None))
                by_id[job.id] = len(rows) - 1
                job.stored = True

    def _store_rows(self, project_id: str) -> list[dict[str, Any]]:
        def view(rows: list[dict[str, Any]], _meta: dict[str, int]) -> list[dict[str, Any]]:
            return json.loads(json.dumps(sorted(rows, key=lambda row: int(row.get("number") or 0)), default=str))
        return self._txn(project_id, view)

    def _find_row(self, job_id: str) -> tuple[str, dict[str, Any]] | None:
        """(project_id, row) of a job another process renders, from the stores this process knows."""
        for project_id in list(self._stores):
            for row in self._store_rows(project_id):
                if row["job_id"] == job_id:
                    return project_id, row
        return None

    def _row_snapshot(self, row: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "job_id": row["job_id"], "kind": str(row.get("kind") or ""), "state": str(row.get("state") or ""),
            "number": int(row.get("number") or 0), "progress": round(float(row.get("progress") or 0.0), 4),
            "stage": str(row.get("stage") or ""),
            "elapsed_ms": max(0, int((time.time() - float(row.get("started_at") or time.time())) * 1000)),
        }
        if row.get("result") is not None:
            out["result"] = row["result"]
        if row.get("error") is not None:
            out["error"] = row["error"]
        return out

    def restore(self, project_id: str, store: Path) -> None:
        """Bind ``project_id``'s jobs to ``store`` (the one queue of the project); a job an earlier process left
        waiting or running is marked failed / app_closed."""
        with self._lock:
            if project_id in self._stores:
                return
            self._stores[project_id] = store
            self._txn(project_id, lambda rows, meta: None)

    def _save_locked(self, project_id: str | None) -> None:
        """Write this process's jobs of one project (call with the lock held)."""
        if (project_id or "") in self._stores:
            self._txn(project_id, lambda rows, meta: None)

    def _save(self, project_id: str | None) -> None:
        with self._lock:
            self._save_locked(project_id)

    def request_of(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                return None if job.request is None else json.loads(json.dumps(job.request))
            found = self._find_row(job_id)
        request = None if found is None else found[1].get("request")
        return json.loads(json.dumps(request)) if isinstance(request, dict) else None

    def stored(self, job_id: str) -> tuple[str | None, dict[str, Any]] | None:
        """(project_id, the job's row as stored: state, kind, result) for a job of this process or of the stores it
        knows (lane w20-dlv: Show in folder and the file check after a restart or from another process)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                return job.project_id, {"state": job.state, "kind": job.kind, "result": job.result}
            found = self._find_row(job_id)
        if found is None:
            return None
        return found[0], {"state": found[1].get("state"), "kind": found[1].get("kind"), "result": found[1].get("result")}

    def _announce(self, job_ids: list[str], reason: str, project_id: str | None) -> None:
        """One ``jobs.changed`` for these export jobs (hint only; never raises)."""
        try:
            events = self._events if self._events is not None else _film_events()
            if events is None or not job_ids:
                return
            scope = f"film_owner:{project_id}" if type(project_id) is str and project_id else None
            events.jobs_changed(job_ids, reason, scope)
        except Exception:  # noqa: BLE001 - an event never fails the queue
            pass

    def snapshot(self, job: _Job) -> dict[str, Any]:
        out: dict[str, Any] = {
            "job_id": job.id, "kind": job.kind, "state": job.state, "number": job.number,
            "progress": round(job.progress, 4), "stage": job.stage,
            "elapsed_ms": int((self._clock() - job.created) * 1000),
        }
        if job.result is not None:
            out["result"] = job.result
        if job.error is not None:
            out["error"] = job.error
        return out

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                return self.snapshot(job)
            found = self._find_row(job_id)
        return None if found is None else self._row_snapshot(found[1])

    def _claim_turn(self, job: _Job) -> bool:
        """Inside the store lock: may ``job`` render now (no running job of a live owner, no earlier waiting one)?
        If so it becomes running in the store at once. Call with self._lock held."""
        if (job.project_id or "") not in self._stores:
            return True

        def claim(rows: list[dict[str, Any]], _meta: dict[str, int]) -> bool:
            if job.cancel.is_set():
                return False
            for row in rows:
                if row["job_id"] == job.id:
                    continue
                if row.get("state") == "running":
                    return False
                if row.get("state") == "waiting" and int(row.get("number") or 0) < job.number and not row.get("cancel_requested"):
                    return False
            job.state = "running"
            return True

        return bool(self._txn(job.project_id, claim))

    def start(self, key: str, kind: str, work: Work, *, label: str = "", project_id: str | None = None,
              request: dict[str, Any] | None = None, settings: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            for job in self._jobs.values():
                if job.key == key and job.state in ACTIVE:
                    return self.snapshot(job)
            job = _Job(id="tlexp_" + uuid.uuid4().hex, key=key, kind=kind, created=self._clock(), number=0,
                       label=label, project_id=project_id, request=request, settings=settings)

            def add(rows: list[dict[str, Any]], meta: dict[str, int]) -> dict[str, Any] | None:
                # The same settings already in the queue of another process: that job (Deliver: "Show Job N").
                same = next((row for row in rows if row.get("key") == key and row.get("state") in ACTIVE and key), None)
                if same is not None:
                    return self._row_snapshot(same)
                meta["last"] += 1
                job.number = meta["last"]
                self._jobs[job.id] = job
                return None

            same = self._txn(project_id, add)
            if same is not None:
                return same
            if (project_id or "") not in self._stores:
                self._numbers = max(self._numbers, job.number)
            self._trim()

        def report(fraction: float, stage: str) -> None:
            moved = stage != job.stage
            job.progress = max(job.progress, min(1.0, float(fraction)))
            job.stage = stage
            if (job.project_id or "") in self._stores and (moved or not job.written or self._clock() - job.written >= _PROGRESS_WRITE_S):
                job.written = self._clock() or 1e-9
                self._save(job.project_id)

        checked = {"at": None}

        def cancelled() -> bool:
            # Stop from another process arrives as cancel_requested in the store (read at most every _POLL_S).
            if not job.cancel.is_set() and (job.project_id or "") in self._stores:
                now = self._clock()
                if checked["at"] is None or now - checked["at"] >= _POLL_S:
                    checked["at"] = now
                    self._save(job.project_id)
            return job.cancel.is_set()

        def run() -> None:
            took_turn = False
            try:
                with self._turn:
                    bound = (job.project_id or "") in self._stores
                    while True:
                        if job.cancel.is_set():
                            raise ExportError("cancelled", "The export was cancelled.")
                        if self._running is None and self._next() is job and self._claim_turn(job):
                            break
                        self._turn.wait(_POLL_S if bound else None)
                        if bound:
                            self._save_locked(job.project_id)  # another process may have stopped or removed it
                    self._running, took_turn = job.id, True
                    job.state = "running"
                    self._save_locked(job.project_id)
                self._announce([job.id], "state", job.project_id)
                job.result = work(report, cancelled)
                job.progress, job.stage, job.state = 1.0, "done", "done"
            except ExportError as exc:
                job.error = exc.as_dict()
                job.state = "cancelled" if exc.code == "cancelled" else "failed"
            except Exception as exc:  # noqa: BLE001 - finite job boundary
                job.error = {"code": "export_failed", "message": "The export failed.", "details": {"type": type(exc).__name__}}
                job.state = "failed"
            finally:
                with self._turn:
                    if took_turn:
                        self._running = None
                    if not job.removed:
                        self._save_locked(job.project_id)  # stored before anyone woken by finished reads the store
                    job.finished.set()
                    self._turn.notify_all()
                self._announce([job.id], "cancelled" if job.state == "cancelled" else "state", job.project_id)

        # Own thread, not the loop's executor: a job outlives the request (and
        # asyncio.run's executor shutdown would otherwise block until it ends).
        thread = threading.Thread(target=run, name=f"timeline-export-{job.id}", daemon=True)
        job.task = thread
        snap = self.snapshot(job)
        self._announce([job.id], "submitted", project_id)
        thread.start()
        return snap

    async def wait(self, job_id: str, wait_ms: int) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            if wait_ms > 0 and not job.finished.is_set():
                await asyncio.to_thread(job.finished.wait, wait_ms / 1000.0)
            return self.snapshot(job)
        # Another process renders it: read the store until it finishes or the wait ends.
        deadline = time.monotonic() + max(0, wait_ms) / 1000.0
        while True:
            state = await asyncio.to_thread(self.get, job_id)
            if state is None or state["state"] not in ACTIVE or time.monotonic() >= deadline:
                return state
            await asyncio.sleep(min(_POLL_S, max(0.0, deadline - time.monotonic())))

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        with self._turn:
            job = self._jobs.get(job_id)
            if job is not None:
                if job.state in ACTIVE:
                    job.cancel.set()
                    self._turn.notify_all()
                return self.snapshot(job)
            found = self._find_row(job_id)
            if found is None:
                return None
            project_id, row = found
            if row.get("state") in ACTIVE:
                def ask(rows: list[dict[str, Any]], _meta: dict[str, int]) -> dict[str, Any] | None:
                    for one in rows:
                        if one["job_id"] == job_id:
                            if one.get("state") in ACTIVE:
                                one["cancel_requested"] = True
                            return dict(one)
                    return None
                row = self._txn(project_id, ask) or row
        self._announce([job_id], "cancelled", project_id)
        return self._row_snapshot(row)

    def _next(self) -> _Job | None:
        """The waiting job whose turn is next (lowest number, not cancelled). Call with the lock held."""
        waiting = [job for job in self._jobs.values() if job.state == "waiting" and not job.cancel.is_set()]
        return min(waiting, key=lambda job: job.number) if waiting else None

    def queue(self, project_id: str | None = None) -> list[dict[str, Any]]:
        """Every job in start order (the project's whole queue, every process); a waiting job has ``position``
        (1 = runs next), others None."""
        with self._lock:
            if project_id is not None and project_id in self._stores:
                stored = self._store_rows(project_id)
                live = {job.id: job for job in self._jobs.values()}
                rows = []
                for row in stored:
                    job = live.get(row["job_id"])
                    out = self.snapshot(job) if job is not None else self._row_snapshot(row)
                    out["label"] = job.label if job is not None else str(row.get("label") or "")
                    out["settings"] = job.settings if job is not None else row.get("settings")
                    out["_stop"] = bool(row.get("cancel_requested")) or (job is not None and job.cancel.is_set())
                    rows.append(out)
                waiting = [row["job_id"] for row in rows if row["state"] == "waiting" and not row["_stop"]]
                for row in rows:
                    row.pop("_stop")
                    row["position"] = waiting.index(row["job_id"]) + 1 if row["job_id"] in waiting else None
                return rows
            jobs = sorted((job for job in self._jobs.values() if project_id is None or job.project_id in (None, project_id)),
                          key=lambda job: job.number)
            waiting = [job.id for job in sorted(self._jobs.values(), key=lambda job: job.number)
                       if job.state == "waiting" and not job.cancel.is_set()]
            rows = []
            for job in jobs:
                row = self.snapshot(job)
                row["label"] = job.label
                row["settings"] = job.settings
                row["position"] = waiting.index(job.id) + 1 if job.id in waiting else None
                rows.append(row)
            return rows

    def remove(self, job_id: str) -> dict[str, Any] | None:
        """Take a waiting job out of the queue: it never runs and leaves the list. ``None`` when unknown;
        a running or finished job is refused with ``not_waiting``."""
        with self._turn:
            job = self._jobs.get(job_id)
            if job is not None:
                if job.state != "waiting":
                    raise ExportError("not_waiting", f"Job {job.number} is {job.state}; only a waiting job can be removed.")
                job.cancel.set()
                job.removed = True
                self._jobs.pop(job_id)

                def drop_own(rows: list[dict[str, Any]], _meta: dict[str, int]) -> None:
                    rows[:] = [row for row in rows if row["job_id"] != job_id]
                self._txn(job.project_id, drop_own)
                self._turn.notify_all()
                snap, project_id = self.snapshot(job), job.project_id
            else:
                found = self._find_row(job_id)
                if found is None:
                    return None
                project_id = found[0]

                def drop(rows: list[dict[str, Any]], _meta: dict[str, int]) -> dict[str, Any] | None:
                    row = next((one for one in rows if one["job_id"] == job_id), None)
                    if row is None:
                        return None
                    if row.get("state") != "waiting":
                        raise ExportError("not_waiting", f"Job {row.get('number')} is {row.get('state')}; only a waiting job can be removed.")
                    rows.remove(row)
                    return row
                row = self._txn(project_id, drop)
                if row is None:
                    return None
                snap = self._row_snapshot(row)
        self._announce([job_id], "cancelled", project_id)
        return {**snap, "removed": True}

    def clear_finished(self, project_id: str | None = None) -> list[str]:
        """Drop done / failed / cancelled jobs (of one project, or all); answers their ids."""
        with self._lock:
            gone = [job_id for job_id, job in self._jobs.items() if job.finished.is_set()
                    and (project_id is None or job.project_id in (None, project_id))]
            projects = {self._jobs[job_id].project_id for job_id in gone}
            for job_id in gone:
                self._jobs.pop(job_id)
            if project_id is not None:
                projects.add(project_id)
            for pid in projects:
                if (pid or "") not in self._stores:
                    continue

                def drop(rows: list[dict[str, Any]], _meta: dict[str, int]) -> list[str]:
                    out = [row["job_id"] for row in rows if row.get("state") not in ACTIVE]
                    rows[:] = [row for row in rows if row.get("state") in ACTIVE]
                    return out
                gone += [job_id for job_id in self._txn(pid, drop) if job_id not in gone]
        if gone:
            self._announce(gone, "state", project_id)
        return gone

    def _trim(self) -> None:
        while len(self._jobs) > MAX_JOBS:
            for job_id, job in self._jobs.items():
                if job.finished.is_set():
                    self._jobs.pop(job_id)
                    break
            else:
                return
