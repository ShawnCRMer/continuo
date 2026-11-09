from __future__ import annotations

# --- Song map (lane w6-song): one clock, measured tracks (studio.song-map/1) ----------------------------------
#
# POST /music/song-map/start {song, approved_roots, project_ref?} starts one job: the beat grid of the file (the
# stored analysis, else the default beat engine, the same as Analyse), the vocal/instrumental stems through the
# Suite's audio-stems-v1 "vocals_rest" workflow on the local ComfyUI (stored stems are reused; the work reads
# GET /queue before its submit and only appends), then ONE song-engine subprocess (music_core/song_map.py).
# GET /music/song-map/status?job_id= reads the job; GET /music/song-map?song=&approved_roots= reads the map cached
# for (audio sha256, engine.json sha256) with the running job if any.  Lyrics come from the song's Interpretation
# (project or scratch); none -> the Words track says "No lyrics".  Agent tools: continuo_song_map_start and
# continuo_song_map_read (agent_tools.ROUTE_TOOLS) answer what the Music page reads.

def _song_map_load() -> Any:
    existing = sys.modules.get("film_suite_song_map")
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location("film_suite_song_map", Path(__file__).with_name("music_core") / "song_map.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("The song map module is unavailable.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["film_suite_song_map"] = module
    spec.loader.exec_module(module)
    return module


_song_map_module = _song_map_load()
_SONG_MAP_JOBS: dict[str, dict[str, Any]] = {}
_SONG_MAP_LOCK = _thread.allocate_lock()
_SONG_MAP_KEEP = 16


class SongMapStartRequest(BaseModel):
    song: str = Field(min_length=1, max_length=4096)
    approved_roots: list[str] = Field(default_factory=list, max_length=32)
    project_ref: MusicProjectReference | None = None


def _song_map_cache_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    return Path(state_root(Path(base) / "hermes")) / "song-map"


def _song_map_lyrics(digest: str, project_ref: MusicProjectReference | None) -> str | None:
    """The lyrics text of the song's Interpretation (artist or transcription draft), or None."""
    try:
        answer = _music_read_interpretation(digest, project_ref)
    except Exception:  # noqa: BLE001 - unreadable lyrics are "no lyrics", never a crash
        return None
    lyrics = ((answer or {}).get("interpretation") or {}).get("lyrics") if isinstance(answer, dict) else None
    # w24-rest AUD-066: an uncorrected transcription draft is the recognizer's own words (already on time): it is no
    # lyrics to align. Once the artist (or Hermes) corrects it, its source is "artist" and the aligner places it.
    if isinstance(lyrics, dict) and lyrics.get("source") == "transcription_draft":
        return None
    text = (lyrics or {}).get("original_text") if isinstance(lyrics, dict) else None
    return text if isinstance(text, str) and text.strip() else None


def _song_map_stems(source: Path, digest: str) -> dict[str, Any]:
    """Vocals + instrumental of the Suite's audio-stems-v1 vocals_rest split (stored ones reused)."""
    out_dir = _audio_stems_dir(digest, "vocals_rest")
    try:
        kept = _audio_stems_module.stored(out_dir)
        if kept is None:
            _audio_stems_module.run(source=source, split="vocals_rest", comfy=_audio_stems_comfy(), out_dir=out_dir, ffmpeg=FFMPEG_PATH)
            source_state = "separated"
        else:
            source_state = "stored"
    except Exception as exc:  # noqa: BLE001 - the tracks that need stems say why
        return {"split": "vocals_rest", "error": getattr(exc, "code", None) or "stems_failed", "vocals": None, "instrumental": None}
    return {"split": "vocals_rest", "source": source_state, "error": None,
            "vocals": str(out_dir / "vocals.flac"), "instrumental": str(out_dir / "rest.flac")}


def _song_map_grid(source: Path, digest: str, progress: Any) -> tuple[list[float], list[float], float]:
    _load_music_analysis()
    analysis = music_analysis.cached_analysis(source, digest)
    if analysis is None:
        analysis = music_analysis.analyze_cached(source, music_analysis.default_engine(), progress)
    return list(analysis.get("beats") or []), list(analysis.get("downbeats") or []), float(analysis.get("duration") or 0.0)


def _song_map_public(job: dict[str, Any]) -> dict[str, Any]:
    out = {"ok": True, "job_id": job["job_id"], "track_sha256": job["track_sha256"], "status": job["status"],
           "progress": round(float(job["progress"]), 3), "stage": job["stage"], "kind": job.get("kind", "map")}
    if job["status"] == "done":
        out["map"] = job["result"]
    if job["status"] == "error":
        out["error"] = job["error"]
    return out


def _song_map_run(job: dict[str, Any], source: Path, project_ref: MusicProjectReference | None) -> None:
    def progress(fraction: float, stage: str) -> None:
        job["progress"], job["stage"] = max(job["progress"], min(0.99, float(fraction))), stage

    digest = job["track_sha256"]
    try:
        progress(0.02, "beats")
        beats, downbeats, duration = _song_map_grid(source, digest, lambda f, s: progress(0.02 + 0.08 * float(f), "beats"))
        lyrics = _song_map_lyrics(digest, project_ref)
        result = _song_map_module.build(cache_dir=_song_map_cache_dir(), audio_path=source, track_sha256=digest, duration=duration,
                                        beats=beats, downbeats=downbeats, lyrics=lyrics,
                                        stems=lambda: _song_map_stems(source, digest), progress=progress)
        job["result"] = result
        if lyrics is None:
            _song_map_write_draft(digest, project_ref, result)
        job["status"], job["progress"], job["stage"] = "done", 1.0, "done"
    except Exception as exc:  # noqa: BLE001 - the job reports a plain failure
        job["status"], job["stage"] = "error", "error"
        job["error"] = {"code": "song_map_failed", "message": "The song map failed", "detail": type(exc).__name__}


def _song_map_draft_text(result: dict[str, Any] | None) -> str | None:
    """The recognized lines of a transcribed Words track, one per line (lines without text, the sung-but-unheard
    stretches, are left out); None when nothing was recognized."""
    words = next((t for t in ((result or {}).get("tracks") or []) if t.get("id") == "words"), None)
    if not words or words.get("status") != "ready" or words.get("source") != "transcribed":
        return None
    lines = [str(ln.get("text") or "").strip() for ln in words.get("lines") or []]
    text = chr(10).join(line for line in lines if line)
    return text or None


def _song_map_write_draft(digest: str, project_ref: MusicProjectReference | None, result: dict[str, Any] | None) -> None:
    """w24-rest AUD-066 (reference: Logic / Premiere transcribe a draft the editor corrects): a song without lyrics gets
    the recognizer's lines as a lyrics DRAFT in its Interpretation (source transcription_draft), in the project or the
    scratch scope the map was made for. Never over existing lyrics; Hermes reads and corrects it with
    continuo_music_interpretation_read / _write, the Audio inspector's Lyrics section shows the same text."""
    text = _song_map_draft_text(result)
    if text is None:
        return
    try:
        current = _music_read_interpretation(digest, project_ref)
        interp = (current or {}).get("interpretation")
        if interp and interp.get("lyrics") is not None:
            return
        contracts = _music_core_module("contracts")
        body = ({k: interp[k] for k in ("tempo", "meter", "downbeat", "sections", "lyrics")} if interp else contracts.default_interpretation_body())
        words = next(t for t in result["tracks"] if t.get("id") == "words")
        language = words.get("language") if isinstance(words.get("language"), str) and re.fullmatch(r"[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*", words.get("language")) else None
        body["lyrics"] = {"original_text": text, "language": language, "source": "transcription_draft",
                          "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "lines": [], "words": []}
        _music_write_interpretation(MusicInterpretationWriteRequest(track_sha256=digest, project_ref=project_ref,
                                                                    expected_revision=int((current or {}).get("revision") or 0), value=body))
    except Exception:  # noqa: BLE001 - the map stands without its draft; the next run writes it
        return


def _song_map_start_sync(request: SongMapStartRequest) -> dict[str, Any]:
    source = _confined_music_path(request.song, _request_roots(request.approved_roots, request.project_ref, path=request.song, server_list=True))
    digest = _song_map_module.sha256_file(source)
    with _SONG_MAP_LOCK:
        for job in _SONG_MAP_JOBS.values():
            if job["track_sha256"] == digest and job["status"] == "running":
                return _song_map_public(job)
        job = {"job_id": f"smap_{uuid.uuid4().hex}", "track_sha256": digest, "status": "running", "progress": 0.0,
               "stage": "starting", "result": None, "error": None, "started": time.time()}
        _SONG_MAP_JOBS[job["job_id"]] = job
        for old in sorted(_SONG_MAP_JOBS.values(), key=lambda row: row["started"])[:-_SONG_MAP_KEEP]:
            if old["status"] != "running":
                _SONG_MAP_JOBS.pop(old["job_id"], None)
    _thread.start_new_thread(_song_map_run, (job, source, request.project_ref))
    return _song_map_public(job)


@router.post("/music/song-map/start")
async def music_song_map_start(request: SongMapStartRequest) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(_song_map_start_sync, request)
    except _FolderNotApproved:
        return _folder_refusal()
    except Exception as exc:  # noqa: BLE001 - route boundary
        return {"ok": False, "reason_code": "song_map_refused", "message": _music_error(exc)}


_SONG_MAP_JOB_RE = re.compile(r"smap_[0-9a-f]{32}")


@router.get("/music/song-map/status")
async def music_song_map_status(request: Request) -> Any:
    job_id = request.query_params.get("job_id") or ""
    if not _SONG_MAP_JOB_RE.fullmatch(job_id):
        return JSONResponse(status_code=422, content={"error": {"code": "invalid_request", "message": "job_id is smap_ + 32 hex"}})
    job = _SONG_MAP_JOBS.get(job_id)
    if job is None:
        return {"ok": False, "reason_code": "song_map_job_unknown", "message": "No such song map job (it ended long ago or the backend restarted)."}
    return _song_map_public(job)


# Lane w18-amapc (AUD-139): a map stored before word recognition (no words_transcriber, or another recognizer / line
# rule) gets its words by a words-only cache migration the READ starts, as Logic / Pro Tools convert an older project
# on open: the read answers the stored map at once with a job (kind "words", stage "words", progress) that redoes only
# the words from the stored vocal stem (song_map.migrate_words, the branch build() uses); a second read joins it; a
# failure is stored in the map (words_migration) and not retried for that recognizer/rule. No stored vocal stem: no job
# (a read never queues a ComfyUI separation); the Words track says why. Page and Hermes read the same.
def _song_map_words_run(job: dict[str, Any], stored: dict[str, Any], vocals: str) -> None:
    def progress(fraction: float, stage: str) -> None:
        job["progress"], job["stage"] = max(job["progress"], min(0.99, float(fraction))), stage

    try:
        clk = stored.get("clock") or {}
        result = _song_map_module.migrate_words(cache_dir=_song_map_cache_dir(), cached=stored, clock=_song_map_module.Clock(clk.get("beats") or [], clk.get("downbeats") or []),
                                                vocals=vocals, progress=progress)
        job["result"] = result
        job["status"], job["progress"], job["stage"] = "done", 1.0, "done"
    except Exception as exc:  # noqa: BLE001 - the job reports a plain failure (stored by migrate_words)
        job["status"], job["stage"] = "error", "error"
        job["error"] = {"code": "song_map_words_failed", "message": "Reading the words failed", "detail": type(exc).__name__}


def _song_map_words_migration(digest: str, stored: dict[str, Any] | None) -> dict[str, Any] | None:
    """The running words job of this file (joined or started), or None (current words, no stem, failed before)."""
    key = _song_map_module.words_outdated(stored)
    if key is None:
        return None
    kept = _audio_stems_module.stored(_audio_stems_dir(digest, "vocals_rest"))
    vocals = _audio_stems_dir(digest, "vocals_rest") / "vocals.flac"
    if kept is None or not vocals.is_file():
        for track in stored.get("tracks") or []:
            if track.get("id") == "words" and track.get("status") != "ready":
                track["reason"], track["reason_text"] = "no_stored_vocal_stem", _song_map_module.REASON_TEXT["no_stored_vocal_stem"]
        return None
    with _SONG_MAP_LOCK:
        for job in _SONG_MAP_JOBS.values():
            if job["track_sha256"] == digest and job["status"] == "running":
                return job
        job = {"job_id": f"smap_{uuid.uuid4().hex}", "track_sha256": digest, "status": "running", "progress": 0.05, "kind": "words",
               "stage": "words", "words_key": key, "result": None, "error": None, "started": time.time()}
        _SONG_MAP_JOBS[job["job_id"]] = job
    _thread.start_new_thread(_song_map_words_run, (job, json.loads(json.dumps(stored)), str(vocals)))
    return job


def _song_map_read_sync(song: str, approved_roots: list[str]) -> dict[str, Any]:
    source = _confined_music_path(song, approved_roots)
    digest = _song_map_module.sha256_file(source)
    stored = _song_map_module.read_cached(_song_map_cache_dir(), digest)
    record, engine_sha = _song_map_module.locate()
    running = next((j for j in _SONG_MAP_JOBS.values() if j["track_sha256"] == digest and j["status"] == "running"), None)
    if running is None and stored is not None:
        running = _song_map_words_migration(digest, stored)
    out = {"ok": True, "schema": _song_map_module.SCHEMA, "track_sha256": digest, "map": stored,
           "engine": {"installed": record is not None, "engine_sha256": engine_sha},
           "job": _song_map_public(running) if running else None}
    if stored is None and record is None:
        # Not installed: the tracks are named, never silently filled by another method.
        out["map"] = _song_map_module.assemble(track_sha256=digest, duration=0.0, clock=_song_map_module.Clock([], []),
                                               engine_sha256=None, record=None, run=None, lyrics_sha256=None, stems_state={})
    return out


@router.get("/music/song-map")
async def music_song_map_read(request: Request) -> Any:
    song = request.query_params.get("song") or ""
    approved_roots = request.query_params.getlist("approved_roots")
    project_id = request.query_params.get("project_id")  # lane p5b-prj: the page names its project (charter decision 4)
    if not song or len(song) > 4096 or len(approved_roots) > 32 or (project_id is not None and not re.fullmatch(r"prj_[0-9a-f]{32}", project_id)):
        return JSONResponse(status_code=422, content={"error": {"code": "invalid_request", "message": "song is a sound file path"}})
    try:
        project_ref = MusicProjectReference(project_id=project_id) if project_id else None
        return await asyncio.to_thread(_song_map_read_sync, song, _request_roots(approved_roots, project_ref, path=song, server_list=True))
    except _FolderNotApproved:
        return _folder_refusal()
    except Exception as exc:  # noqa: BLE001 - route boundary
        return {"ok": False, "reason_code": "song_map_refused", "message": _music_error(exc)}
