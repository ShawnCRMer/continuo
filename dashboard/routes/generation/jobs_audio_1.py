from __future__ import annotations


# --- Audio jobs in the Jobs list (lane w24-rest, Jobs parity) -------------------------------------------------------
# The Jobs popover lists the Audio page's runs (Stems, the beat analysis, the song map / words, subtitles from audio);
# continuo_jobs_list listed image, shoot and export only, so Hermes did not see what the panel showed. These runs live in
# this process's job tables (audio/stems_1.py, music/analysis_jobs_1.py, music/song_map_1.py, projects/subtitles_1.py);
# they are listed as module "music" rows of the same shape as every other row. Reference: Resolve's / Logic's one
# background-task list for every kind of work.
_AUDIO_JOB_STATE = {"running": "running", "done": "done", "error": "failed", "failed": "failed", "stopped": "cancelled"}
_AUDIO_JOB_TABLES = (
    ("_AUDIO_STEMS_JOBS", "Stems"),
    ("_MUSIC_JOBS", "Beat analysis"),
    ("_SONG_MAP_JOBS", "Song map"),
    ("_SUBTITLES_JOBS", "Subtitles from audio"),
)


def _audio_jobs_for_scope(limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for table_name, label in _AUDIO_JOB_TABLES:
        table = globals().get(table_name)
        if not isinstance(table, dict):
            continue
        for job in list(table.values()):
            if not isinstance(job, dict) or not isinstance(job.get("job_id"), str):
                continue
            raw = job.get("status", job.get("state"))
            state = _AUDIO_JOB_STATE.get(raw, "running")
            error = job.get("error") if isinstance(job.get("error"), dict) else None
            started = _finite_epoch(job.get("started")) if isinstance(job.get("started"), (int, float)) else None
            finished = _finite_epoch(job.get("finished")) if isinstance(job.get("finished"), (int, float)) else None
            kind = "Words" if table_name == "_SONG_MAP_JOBS" and job.get("kind") == "words" else label
            rows.append({
                "module": "music",
                "job_id": job["job_id"],
                "state": state,
                "raw_state": raw if isinstance(raw, str) else None,
                "label": kind,
                "progress": round(float(job.get("progress") or 0.0), 3),
                "stage": job.get("stage") or None,
                "result_name": None,
                "created_at": started,
                "started_at": started,
                "finished_at": finished,
                "error": _job_label(error.get("message") or error.get("code")) if state == "failed" and error else None,
                "reason_code": (error.get("code") if isinstance(error.get("code"), str) and error.get("code") else None) if state == "failed" and error else None,
            })
    rows.sort(key=lambda row: row.get("created_at") or 0, reverse=True)
    return rows[:limit]


_jobs_list_before_w24_rest_audio = jobs_list


async def _jobs_list_with_audio(request: JobsListRequest) -> dict[str, Any]:
    """The Jobs list plus this process's Audio runs (module music), like the Jobs popover."""
    answer = await _jobs_list_before_w24_rest_audio(request)
    try:
        audio = _audio_jobs_for_scope(request.limit)
        answer["sources"]["music"] = "ok"
    except Exception:  # noqa: BLE001 - reported as a source state
        audio = []
        answer["sources"]["music"] = "unavailable"
    for job in audio:
        job["by"] = "artist"
    counts = _jobs_mark_dismissed(request.owner_scope, audio)
    if not request.include_dismissed:
        audio = [job for job in audio if not job["dismissed"]]
    answer["jobs"] = [*answer["jobs"], *audio]
    for key, value in counts.items():
        if isinstance(value, int) and isinstance(answer["counts"].get(key), int):
            answer["counts"][key] += value
    return answer


jobs_list = _jobs_list_with_audio  # noqa: F811 - the facade name answers the same as the route (as job_names_1.py)
# Loaded after router.include_router(project_router) (plugin_api.py): both routers carry the /jobs/list route object.
for _route in [*project_router.routes, *router.routes]:
    if getattr(_route, "path", None) == "/jobs/list" and "POST" in getattr(_route, "methods", ()):
        _route.endpoint = _jobs_list_with_audio
        _route.dependant.call = _jobs_list_with_audio
