from __future__ import annotations


_JOB_LABEL_MAX = 120
_IMAGE_JOB_STATE_VIEW = {
    "accepted": "queued", "submitting": "queued", "queued": "queued", "running": "running",
    # "ready"/"ingested": ComfyUI has finished; only the local pickup is left.
    # Nothing renders any more, so these are never counted as active work.
    "ready": "done", "ingested": "done", "published": "done",
    "failed": "failed", "quarantined": "failed", "cancelled": "cancelled",
}
_VIDEO_JOB_STATE_VIEW = {
    "submitting": "queued", "queued": "queued", "running": "running", "ready": "done",
    "failed": "failed", "moderated": "failed", "cancelled": "cancelled",
}


class JobsListRequest(_ProjectRequest):
    schema_: Literal["studio.jobs-list/1"] = Field(alias="schema")
    owner_scope: str = Field(pattern=_OWNER_SCOPE_PATTERN, min_length=18, max_length=47)
    limit: int = Field(default=40, ge=1, le=200)
    # lane p5c-frm: a dismissed failure is left out unless asked for (then marked dismissed: true).
    include_dismissed: bool = False

    @property
    def project_id(self) -> str | None:
        suffix = self.owner_scope[len("film_owner:"):]
        return None if suffix == "scratch" else suffix


def _job_label(text: Any) -> str:
    value = " ".join(str(text or "").split())
    if len(value) > _JOB_LABEL_MAX:
        value = value[: _JOB_LABEL_MAX - 1].rstrip() + "\u2026"
    return value


def _finite_epoch(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _image_scope_services(owner_scope: str) -> tuple[Any, Any, Any]:
    with _IMAGE_FILM_ROOT_LOCK:
        root = _image_producer_root(owner_scope)
        handoff = _image_operations_handoff(root, owner_scope)
        return handoff, handoff.producer_jobs(), handoff.image_states()


def _image_window(limit: int) -> int:
    # One read serves listing, reconciliation and collection: the window is
    # the largest any of them needs (reconcile/collect looked at 200 rows).
    return min(500, max(200, limit * 5))


def _image_jobs_for_scope(owner_scope: str, limit: int, rows: list[dict[str, Any]] | None = None,
                          services: tuple[Any, Any, Any] | None = None) -> list[dict[str, Any]]:
    _handoff, jobs, states = services or _image_scope_services(owner_scope)
    # The ledger is profile-wide: scan a bounded window, keep the scope's own.
    if rows is None:
        rows = jobs.list_recent(limit=_image_window(limit))
    try:
        times = jobs.job_times([job["image_producer_job_id"] for job in rows])
    except Exception:  # noqa: BLE001 - a ledger without time columns lists untimed
        times = {}
    listed: list[dict[str, Any]] = []
    try:
        result_names = _image_result_names(owner_scope)  # lane p6-name: names reserved at submit
    except Exception:  # noqa: BLE001 - no names: None
        result_names = {}
    for job in rows:
        basis = (job.get("request_snapshot") or {}).get("context_basis") or {}
        try:
            states.require_owned(basis.get("context_id"))
        except Exception:  # noqa: BLE001 - not owned: not listed
            continue
        artist = (job.get("request_snapshot") or {}).get("artist_input") or {}
        raw = str(job.get("state") or "")
        error = job.get("failure_reason") or job.get("quarantine_reason")
        stamp = times.get(job["image_producer_job_id"]) or {}
        view = _IMAGE_JOB_STATE_VIEW.get(raw, "queued")
        # lane p7-jobs (VID-102): "cancel_unknown" is only ever written by a cancel someone asked for
        # (ImageProducerCoordinator.cancel, the provider never confirmed it): the job was stopped, not failed.
        if raw == "quarantined" and job.get("quarantine_reason") == "cancel_unknown":
            view = "cancelled"
        listed.append({
            "module": "image",
            "job_id": job["image_producer_job_id"],
            "state": view,
            "raw_state": raw,
            "label": _job_label(artist.get("prompt")),
            "result_name": result_names.get(job["image_producer_job_id"]),
            "target_id": (job.get("provider") or {}).get("target_id"),
            # Rows written before the ledger kept times read None: unknown,
            # never guessed. started_at is when ComfyUI accepted the prompt.
            "created_at": _finite_epoch(stamp.get("created_at")),
            "started_at": _finite_epoch(stamp.get("submitted_at")),
            "finished_at": _finite_epoch(stamp.get("updated_at")) if view in ("done", "failed", "cancelled") else None,
            "error": _job_label(error) if error else None,
            # lane p12-shl (GEM-053): the failure's finite code when the ledger wrote one (else the page reads the text).
            "reason_code": _job_reason_code(error) if view == "failed" else None,
            "_run_key": str(job.get("idempotency_key") or ""),
            # Lane w25-imgfix item 1 (IMG-212): the model the run used in the artist's words ("Krea 2 Turbo").
            "model_label": _image_job_model_label(job),
        })
    return _image_runs_folded(listed)[:limit]


# Lane w25-imgfix item 1 (IMG-212): a Jobs row names the run the way the artist chose it (Lightroom's activity bar
# names the task, Resolve's render queue the preset + clip; the output file is a detail). The model is read from the
# job's own render spec (profile turbo|raw -> the model catalog's "Krea 2" + "Turbo"), else the workflow's catalog
# label without its "<runs on> · " prefix. Unknown answers None (the page then keeps its older naming).
def _image_job_model_label(job: dict[str, Any]) -> str | None:
    snapshot = job.get("request_snapshot") or {}
    effective = snapshot.get("effective_input") or {}
    spec = effective.get("render_spec") if isinstance(effective, dict) else None
    profile = spec.get("profile") if isinstance(spec, dict) else None
    target = (job.get("provider") or {}).get("target_id") or snapshot.get("target_id")
    if snapshot.get("operation") == "create" and profile is None:
        profile = "turbo"  # a v1 create (no render spec) renders the default Turbo profile (image_contract)
    try:
        if profile in ("turbo", "raw") and snapshot.get("operation") == "create":
            catalog = _model_catalog_module.catalog_view()
            family = next(f for f in catalog["families"] if f["id"] == "krea2")
            variant = _model_catalog_module.find_variant(catalog, "krea2", profile)
            return f"{family['label']} {variant['label']}"
    except Exception:  # noqa: BLE001 - without the catalog the workflow label below names it
        pass
    namer = globals().get("_job_workflow_label")
    label = namer(target) if callable(namer) else None
    if not isinstance(label, str) or not label or label == target:
        return None
    return label.split(" · ", 1)[1] if " · " in label else label


def _media_workflow_extras(job: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    # Media named a P-H3/Ideogram/Qwen picture "Krea 2 Turbo"; the item now names its model like its Jobs row.
    """w31-fix4 item 1 (P1 MED-100; media/feed_1.py image items): model_label, reference_count and loras of a catalog workflow picture; {} otherwise."""
    try:
        if snapshot.get("operation") not in {"workflow_create", "workflow_edit"}:
            return {}
        workflow = (snapshot.get("effective_input") or {}).get("workflow") or {}
        namer = globals().get("_image_job_model_label")
        label = namer(job) if callable(namer) else None
        loras = [{"file": str(l.get("file")), "strength": l.get("strength")} for l in (workflow.get("loras") or []) if isinstance(l, dict) and isinstance(l.get("file"), str)]
        return {"model_label": label if isinstance(label, str) and label else None,
                "reference_count": len(workflow.get("references") or []), **({"loras": loras} if loras else {})}
    except Exception:  # noqa: BLE001 - the picture still lists without them
        return {}

# lane w20-img (IMG-046): a Generate x N run is N ledger jobs keyed "<op>" and "<op>:run-<i>" (producer_1.py,
# workflow_1.py). The list shows it as ONE job, the way ComfyUI shows one queue entry for a batch: the row of member 0
# (its job id is the run id the page follows) with run = {size, settled, failed} and the state over all members.
_RUN_MEMBER_RE = re.compile(r"^(.+):run-([1-9][0-9]*)$")


def _image_runs_folded(listed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    order: list[str] = []
    for row in listed:
        key = row.pop("_run_key", "")
        match = _RUN_MEMBER_RE.match(key)
        base, index = (match.group(1), int(match.group(2))) if match else (key or row["job_id"], 0)
        if base not in groups:
            groups[base] = []
            order.append(base)
        groups[base].append((index, row))
    folded: list[dict[str, Any]] = []
    for base in order:
        members = sorted(groups[base], key=lambda item: item[0])
        head = dict(members[0][1])
        if len(members) > 1:
            states = [row["state"] for _index, row in members]
            # A run with a settled member and a waiting one is under way (ComfyUI: the batch entry is running).
            if "running" in states or ("queued" in states and any(st not in ("running", "queued") for st in states)):
                head["state"] = "running"
            elif "queued" in states:
                head["state"] = "queued"
            elif "failed" in states:
                # Lane w24-rest (GEM-056): a settled run with a failed member is a failed run (run.failed says how many),
                # never "done"; its reason is the first failed member's (the artist's window: "done", run.failed 1, no reason).
                head["state"] = "failed"
            elif all(state == "cancelled" for state in states):
                head["state"] = "cancelled"
            else:
                head["state"] = "done"
            settled = [row for _index, row in members if row["state"] not in ("running", "queued")]
            if head["state"] in ("running", "queued"):
                head["finished_at"] = None
            else:
                stamps = [row["finished_at"] for row in settled if row["finished_at"] is not None]
                head["finished_at"] = max(stamps) if stamps else None
            starts = [row["started_at"] for _index, row in members if row["started_at"] is not None]
            head["started_at"] = min(starts) if starts else None
            if head["state"] != "failed":
                head["error"], head["reason_code"] = None, None
            elif head["error"] is None or head["state"] != members[0][1]["state"]:
                first = next((row for _index, row in members if row["state"] == "failed" and row["error"]), members[0][1])
                head["error"], head["reason_code"] = first["error"], first["reason_code"]
            head["run"] = {"size": len(members), "settled": len(settled),
                           "failed": sum(1 for row in settled if row["state"] == "failed")}
        folded.append(head)
    return folded


def _video_jobs_for_scope(project_id: str | None, limit: int) -> list[dict[str, Any]]:
    listed: list[dict[str, Any]] = []
    records = _comfy_store().list_recent(project_id=project_id, limit=limit)
    causes = _job_causes_read([record.studio_job_id for record in records if record.state in ("failed", "moderated")])
    for record in records:
        try:
            prompt = (record.request.get("args") or {}).get("prompt")
        except Exception:  # noqa: BLE001 - an unreadable request still lists
            prompt = None
        raw = record.state
        reason_code = record.reason_code if raw in ("failed", "moderated") else None
        # Lane p12-shl (GEM-053, owner 31.10.): a finished take whose published file is gone is Failed output_missing,
        # never Done. One stat per finished row with an output path; never a hash, never a read.
        if raw == "ready" and isinstance(getattr(record, "published_path", None), str) and record.published_path:
            try:
                present = Path(record.published_path).stat() is not None
            except OSError:
                present = False
            if not present:
                raw, reason_code = "failed", "output_missing"
        facts = _h3_public_facts(record.studio_job_id) if _h3_is_take(record) else None
        motion = {key: facts[key] for key in ("motion_activity", "frozen") if key in facts} if facts else {}
        listed.append({
            **motion,
            "module": "shoot",
            "job_id": record.studio_job_id,
            "state": _VIDEO_JOB_STATE_VIEW.get(raw, "queued"),
            "raw_state": raw,
            "label": _job_label(prompt or record.requested_basename),
            # lane p6-name (GEM-068): the file name the take gets, known at submit (project rule or the typed name).
            "result_name": record.requested_basename if isinstance(record.requested_basename, str) and record.requested_basename else None,
            "target_id": record.target_id,
            "created_at": _finite_epoch(record.created_at),
            "started_at": _finite_epoch(record.running_at),
            "finished_at": _finite_epoch(record.completed_at),
            # lane w11-backend M-35: the sentence recorded when it failed, else the reason code's sentence.
            "error": _job_label(causes.get(record.studio_job_id) or generation.contract.public_reason(reason_code))
            if raw in ("failed", "moderated") and (reason_code or record.studio_job_id in causes) else (
                "Moderated by the provider" if raw == "moderated" else None
            ),
            "reason_code": _job_reason_code(reason_code),
        })
    try:  # refused submits (no ledger row) are jobs of the project too
        listed.extend(_job_refusals_for_scope(project_id, limit))
    except Exception:  # noqa: BLE001 - the ledger's jobs still list
        pass
    listed.sort(key=lambda job: job.get("created_at") or 0.0, reverse=True)
    return listed[:limit]


# Lane w20-dlv (GEM-056, HER-047, HER-031): the project's Deliver render jobs are jobs of the Jobs list too, the same rows
# the render queue answers (continuo_timeline_export_queue); Stop / Render again by job_id through
# continuo_timeline_export_cancel / continuo_timeline_export_queue action again. Path-free.
_EXPORT_JOB_STATE_VIEW = {"waiting": "queued", "running": "running", "done": "done", "failed": "failed", "cancelled": "cancelled"}


def _export_jobs_for_scope(project_id: str | None, limit: int) -> list[dict[str, Any]]:
    if project_id is None:
        return []
    try:
        directory = _project_reference_directory(ProjectReference(project_id=project_id))
    except studio_core.ProjectError as exc:
        if exc.code != "project_not_found":
            raise
        return []  # not a recent project: it has no render queue (no Deliver job), which is not an outage
    rows = _timeline_export_jobs_for(project_id, Path(directory)).queue(project_id)
    project_dir = Path(os.path.realpath(directory))
    now = time.time()
    listed: list[dict[str, Any]] = []
    for row in rows:
        raw = row.get("state")
        error = row.get("error") if isinstance(row.get("error"), dict) else None
        failed = raw in ("failed", "cancelled") and error is not None
        started = now - max(0, int(row.get("elapsed_ms") or 0)) / 1000.0
        listed.append({
            "module": "export",
            "job_id": row["job_id"],
            "state": _EXPORT_JOB_STATE_VIEW.get(raw, "queued"),
            "raw_state": raw,
            "kind": row.get("kind"),
            "number": row.get("number"),
            "label": _job_label(row.get("label") or f"Job {row.get('number')}"),
            "progress": row.get("progress"),
            "stage": row.get("stage") or None,
            "position": row.get("position"),
            "settings": row.get("settings"),
            "result_name": None,
            "created_at": _finite_epoch(started),
            "started_at": _finite_epoch(started) if raw != "waiting" else None,
            "finished_at": None,
            "error": _job_label(error.get("message") or error.get("code")) if failed else None,
            "reason_code": (error.get("code") if type(error.get("code")) is str and error.get("code") else None) if failed else None,
        })
    listed.sort(key=lambda job: job.get("number") or 0, reverse=True)
    return _timeline_export_answer({"jobs": listed[:limit]}, project_dir)["jobs"]  # strips the project folder from any text


# A finished run ("ready": ComfyUI is done, the picture still sits in its temp
# output) used to be brought home only by an Image pane polling its status. A
# pane closed at the wrong moment left the job "ready" forever. The Jobs list
# is polled by the Film Suite page, so it also collects the scope's OWN ready
# jobs: a few per call, each at most once per retry window, never a submit,
# never a running job, never another scope's job.
_READY_COLLECT_PER_CALL = 2
_READY_COLLECT_RETRY_SECONDS = 60.0
_READY_COLLECT_TIMEOUT_SECONDS = 20.0
_READY_COLLECT_LAST_TRY: dict[str, float] = {}


def _ready_collect_clock() -> float:
    return time.monotonic()


async def _collect_owned_ready_image_jobs(owner_scope: str, rows: list[dict[str, Any]] | None = None,
                                          services: tuple[Any, Any, Any] | None = None) -> list[str]:
    handoff, jobs, states = services or _image_scope_services(owner_scope)
    now = _ready_collect_clock()
    due: list[str] = []
    for job in (rows if rows is not None else jobs.list_recent(limit=200))[:200]:
        if job.get("state") != "ready":
            continue
        basis = (job.get("request_snapshot") or {}).get("context_basis") or {}
        try:
            states.require_owned(basis.get("context_id"))
        except Exception:  # noqa: BLE001 - not this scope's job: never touched
            continue
        job_id = job["image_producer_job_id"]
        last = _READY_COLLECT_LAST_TRY.get(job_id)
        if last is not None and now - last < _READY_COLLECT_RETRY_SECONDS:
            continue
        due.append(job_id)
        if len(due) >= _READY_COLLECT_PER_CALL:
            break
    if not due:
        return []
    async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
        coordinator = handoff.producer_coordinator()
        collector, _reader = handoff.result_services()
    for job_id in due:
        _READY_COLLECT_LAST_TRY[job_id] = now

        async def bring_home(job_id: str = job_id) -> None:
            # A ready job answers from the coordinator's descriptor cache; after
            # a restart it polls the provider once for that one owned prompt.
            # A prompt ComfyUI forgot becomes quarantined "job_lost" there.
            job = await coordinator.status(job_id)
            if not isinstance(job, dict) or job.get("state") != "ready":
                return
            await collector.collect(job_id, job.get("output_descriptor"))

        try:
            await asyncio.wait_for(bring_home(), timeout=_READY_COLLECT_TIMEOUT_SECONDS)
        except Exception:  # noqa: BLE001 - the ledger holds the outcome; retried later
            continue
    if len(_READY_COLLECT_LAST_TRY) > 2_000:
        cutoff = now - _READY_COLLECT_RETRY_SECONDS
        for key in [key for key, at in _READY_COLLECT_LAST_TRY.items() if at < cutoff]:
            _READY_COLLECT_LAST_TRY.pop(key, None)
    return due


# A queued/running row used to leave that state only while the Image pane
# that submitted it kept polling its status. A reload, a project switch or one
# network error left it "running" forever (ghost jobs). The Jobs list is polled
# by the page, so it also reconciles the scope's OWN unfinished rows through
# the coordinator (the same call the status route makes): a few per call, each
# at most once per window, never a submit. ComfyUI no longer knowing a prompt
# ends the row quarantined ``job_lost`` (the coordinator's existing path). The
# first list call per scope after the backend starts is the startup sweep: it
# reconciles every unfinished owned row of the window, not just a few.
_RECONCILE_STATES = frozenset({"queued", "running"})
_RECONCILE_PER_CALL = 10
_RECONCILE_SWEEP_MAX = 60
_RECONCILE_RETRY_SECONDS = 8.0
_RECONCILE_TIMEOUT_SECONDS = 10.0
_RECONCILE_LAST_TRY: dict[str, float] = {}
_RECONCILE_SWEPT_SCOPES: set[str] = set()


async def _reconcile_owned_unfinished_image_jobs(owner_scope: str, rows: list[dict[str, Any]] | None = None,
                                                 services: tuple[Any, Any, Any] | None = None) -> list[str]:
    handoff, jobs, states = services or _image_scope_services(owner_scope)
    sweep = owner_scope not in _RECONCILE_SWEPT_SCOPES
    bound = _RECONCILE_SWEEP_MAX if sweep else _RECONCILE_PER_CALL
    now = _ready_collect_clock()
    due: list[str] = []
    for job in (rows if rows is not None else jobs.list_recent(limit=200))[:200]:
        if job.get("state") not in _RECONCILE_STATES or not (job.get("provider") or {}).get("prompt_id"):
            continue
        basis = (job.get("request_snapshot") or {}).get("context_basis") or {}
        try:
            states.require_owned(basis.get("context_id"))
        except Exception:  # noqa: BLE001 - not this scope's job: never touched
            continue
        job_id = job["image_producer_job_id"]
        last = _RECONCILE_LAST_TRY.get(job_id)
        if not sweep and last is not None and now - last < _RECONCILE_RETRY_SECONDS:
            continue
        due.append(job_id)
        if len(due) >= bound:
            break
    _RECONCILE_SWEPT_SCOPES.add(owner_scope)
    if not due:
        return []
    async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
        coordinator = handoff.producer_coordinator()
    for job_id in due:
        _RECONCILE_LAST_TRY[job_id] = now

    async def one(job_id: str) -> None:
        try:
            await asyncio.wait_for(coordinator.status(job_id), timeout=_RECONCILE_TIMEOUT_SECONDS)
        except Exception:  # noqa: BLE001 - the ledger keeps its state; retried later
            pass

    await asyncio.gather(*(one(job_id) for job_id in due))
    if len(_RECONCILE_LAST_TRY) > 2_000:
        cutoff = now - _RECONCILE_RETRY_SECONDS
        for key in [key for key, at in _RECONCILE_LAST_TRY.items() if at < cutoff]:
            _RECONCILE_LAST_TRY.pop(key, None)
    return due


@project_router.post("/jobs/list")
async def jobs_list(request: JobsListRequest) -> dict[str, Any]:
    """Jobs panel feed: Image + Shoot jobs of one owner scope, and its project's Deliver render jobs (w20-dlv).

    Never submits. The only writes are reconciling the scope's own unfinished
    rows with the provider and bringing home its finished-but-uncollected
    pictures (see above). Each
    source degrades on its own so one unavailable ledger does not hide the
    other's jobs."""
    sources: dict[str, str] = {}
    jobs: list[dict[str, Any]] = []
    # The image ledger is read ONCE per call; reconciliation and collection
    # work on that window and only the rows they touched are re-read.
    services = rows = None
    try:
        services = _image_scope_services(request.owner_scope)
        rows = services[1].list_recent(limit=_image_window(request.limit))
    except Exception:  # noqa: BLE001 - each step below degrades on its own
        services = rows = None
    touched: list[str] = []
    try:
        touched += await _reconcile_owned_unfinished_image_jobs(request.owner_scope, rows, services) or []
    except Exception:  # noqa: BLE001 - listing never depends on reconciliation
        pass
    try:
        touched += await _collect_owned_ready_image_jobs(request.owner_scope, rows, services) or []
    except Exception:  # noqa: BLE001 - listing never depends on collection
        pass
    if rows is not None and touched:
        wanted = set(touched)
        fresh: list[dict[str, Any]] = []
        changes: list[tuple[str, Any, Any]] = []
        for job in rows:
            if job.get("image_producer_job_id") in wanted:
                before = job.get("state")
                try:
                    job = services[1].get_job(job["image_producer_job_id"])
                except Exception:  # noqa: BLE001 - keep the row as read
                    pass
                changes.append((job.get("image_producer_job_id"), before, job.get("state")))
            fresh.append(job)
        rows = fresh
        # Only rows whose state really moved; an unchanged reconcile is silent.
        _announce_image_jobs(changes, request.owner_scope)
    try:
        jobs.extend(_image_jobs_for_scope(request.owner_scope, request.limit, rows, services))
        sources["image"] = "ok"
    except Exception:  # noqa: BLE001 - reported as a source state
        sources["image"] = "unavailable"
    try:
        jobs.extend(_video_jobs_for_scope(request.project_id, request.limit))
        sources["shoot"] = "ok"
    except Exception:  # noqa: BLE001 - reported as a source state
        sources["shoot"] = "unavailable"
    if request.project_id is not None:  # lane w20-dlv: Deliver's render queue (a scratch scope has none)
        try:
            jobs.extend(_export_jobs_for_scope(request.project_id, request.limit))
            sources["export"] = "ok"
        except Exception:  # noqa: BLE001 - reported as a source state
            sources["export"] = "unavailable"
    for job in jobs:  # lane p6-name: every listed job names its result file (None = not known)
        job.setdefault("result_name", None)
    _job_authors_apply(jobs)  # by: "hermes" | "artist" (no note = "artist")
    # lane p5c-frm: dismissed failures are marked and never counted (jobs_dismiss_1.py).
    counts = _jobs_mark_dismissed(request.owner_scope, jobs)
    if not request.include_dismissed:
        jobs = [job for job in jobs if not job["dismissed"]]
    return {"schema": _JOBS_LIST_SCHEMA, "owner_scope": request.owner_scope, "jobs": jobs, "sources": sources, "counts": counts}
