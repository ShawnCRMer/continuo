from __future__ import annotations


# --- Subtitles (lane w23-subs, CUT-173 / CUT-011): the cut's subtitle track ST1 --------------------------------------
#
# Reference (docs/studies/w23-subs-STUDY.md): Resolve's subtitle track + Create Subtitles from Audio, Premiere's
# Create captions, FCP's caption clips; Netflix / BBC readability numbers. One document per project
# (state/subtitles.json, studio.subtitles/1) keyed by the cut's timeline id, with its own revision (a stale write is
# refused with subtitles_conflict, never last-writer-wins). The pure rules live in timeline_export/subtitles.py, the
# file in timeline_export/subtitle_store.py, the one rasteriser of the viewer and the burn-in in
# timeline_export/subtitle_raster.py. The page and Hermes use the same routes (continuo_subtitles_*); an agent call
# is told apart by _film_caller. Answers never carry a path.

_SUBTITLES_JOBS: dict[str, dict[str, Any]] = {}
_SUBTITLES_JOBS_LOCK = _thread.RLock()
_SUBTITLES_WAIT_MS_MAX = 20_000
_SUBTITLES_ASR_LOCK = _thread.RLock()


def _subtitles_module(name: str) -> Any:
    _timeline_export_package()
    return sys.modules[f"{_TIMELINE_EXPORT_PACKAGE_NAME}.{name}"] if f"{_TIMELINE_EXPORT_PACKAGE_NAME}.{name}" in sys.modules else importlib.import_module(f"{_TIMELINE_EXPORT_PACKAGE_NAME}.{name}")


class SubtitlesReadRequest(_ProjectRequest):
    project_ref: ProjectReference
    timeline_id: str | None = Field(default=None, min_length=1, max_length=120)


class SubtitlesApplyRequest(_ProjectRequest):
    project_ref: ProjectReference
    expected_revision: int = Field(ge=0)
    timeline_id: str | None = Field(default=None, min_length=1, max_length=120)
    ops: list[dict[str, Any]] = Field(min_length=1, max_length=200)


class SubtitlesRange(BaseModel):
    in_frame: int = Field(ge=0)
    out_frame: int = Field(ge=1)


class SubtitlesTranscribeRequest(_ProjectRequest):
    project_ref: ProjectReference
    timeline_id: str | None = Field(default=None, min_length=1, max_length=120)
    range: SubtitlesRange | None = None
    # Which sound is heard: the dialogue (dialogue tracks + picture clips with their sound on), or every audible sound.
    source: Literal["dialogue", "all"] = "dialogue"
    # The look and rules of a track that has none yet (a track already set keeps its own).
    preset: Literal["classic", "social"] | None = None
    language: str | None = Field(default=None, pattern=r"^[a-z]{2}$")
    wait_ms: int = Field(default=0, ge=0, le=_SUBTITLES_WAIT_MS_MAX)


class SubtitlesStatusRequest(_ProjectRequest):
    job_id: str = Field(pattern=r"^subtx_[0-9a-f]{32}$")
    wait_ms: int = Field(default=0, ge=0, le=_SUBTITLES_WAIT_MS_MAX)


class SubtitlesExportRequest(_ProjectRequest):
    project_ref: ProjectReference
    timeline_id: str | None = Field(default=None, min_length=1, max_length=120)
    format: Literal["srt", "vtt"] = "srt"
    range: SubtitlesRange | None = None
    folder: str | None = Field(default=None, min_length=1, max_length=2000)
    name: str | None = Field(default=None, min_length=1, max_length=60)


class SubtitlesFrameRequest(_ProjectRequest):
    project_ref: ProjectReference
    timeline_id: str | None = Field(default=None, min_length=1, max_length=120)
    cue: str = Field(pattern=r"^st-[0-9a-f]{8}$")
    active: int | None = Field(default=None, ge=0, le=500)


def _subtitles_refusal(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, studio_core.ProjectError):
        return {"ok": False, "reason_code": getattr(exc, "code", "project_unavailable"), **_project_error_response(exc)}
    code = getattr(exc, "reason_code", "subtitles_invalid")
    return {"ok": False, "reason_code": code, "message": getattr(exc, "message", str(exc)), "error": {"code": code, "message": getattr(exc, "message", str(exc))}}


def _subtitles_cut(project_ref: ProjectReference, timeline_id: str | None) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """(directory, the timeline state, the chosen timeline) of the project; refuses a missing cut."""
    directory = _project_reference_directory(project_ref)
    _revision, state = _timeline_current(_project_store_factory(), directory, project_ref.project_id)
    wanted = timeline_id or state.get("active")
    timeline = next((t for t in state.get("timelines") or () if t.get("id") == wanted), None)
    if timeline is None:
        raise _subtitles_module("subtitles").SubtitleRefusal("timeline_not_found", f"No cut {wanted}.")
    return directory, state, timeline


def _subtitles_cut_facts(timeline: dict[str, Any]) -> tuple[dict[str, int], int, list[int], dict[str, int]]:
    """(rate, total frames, V1 cut frames, format {width, height}) of a cut."""
    model = _subtitles_module("model")
    total = model.timeline_end(timeline)
    cuts, at = [], 0
    for track in timeline.get("tracks") or ():
        if track.get("kind") == "video" and track.get("role") == "storyline":
            for item in track.get("items") or ():
                if at:
                    cuts.append(at)
                at += int(item.get("duration_frames") or 0)
    fmt = timeline.get("format") or {}
    return dict(timeline["rate"]), total, cuts, {"width": int(fmt.get("width") or 1920), "height": int(fmt.get("height") or 1080)}


def _subtitles_store() -> Any:
    return _subtitles_module("subtitle_store").SubtitleStore(_project_store_factory())


def _subtitles_answer(doc: dict[str, Any], timeline: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    S = _subtitles_module("subtitles")
    rate, total, cuts, fmt = _subtitles_cut_facts(timeline)
    entry = S.timeline_entry(doc, timeline["id"])
    return {"ok": True, "reason_code": None, "revision": doc["revision"], "timeline_id": timeline["id"], "rate": rate,
            "total_frames": total, "format": fmt, "cuts": cuts, **S.public(entry, rate), "states": S.states(entry), **(extra or {})}


def _subtitles_announce(project_id: str, revision: int) -> None:
    try:
        film_events.subtitles_changed(project_id, revision, _film_caller())
    except Exception:  # noqa: BLE001 - an event never fails a route
        pass


@project_router.post("/projects/subtitles/read")
async def projects_subtitles_read(request: SubtitlesReadRequest) -> dict[str, Any]:
    """The cut's subtitle track ST1: style, rules, cues {id, start, end, text, words, lines, qc, edited, by}, states."""
    def work() -> dict[str, Any]:
        directory, _state, timeline = _subtitles_cut(request.project_ref, request.timeline_id)
        with _subtitles_store().open(directory, request.project_ref.project_id) as tx:
            doc = tx.load()
        return _subtitles_answer(doc, timeline)
    try:
        return await asyncio.to_thread(work)
    except (studio_core.ProjectError, ValueError) as exc:
        return _subtitles_refusal(exc)


@project_router.post("/projects/subtitles/apply")
async def projects_subtitles_apply(request: SubtitlesApplyRequest) -> dict[str, Any]:
    """Edit verbs on the cut's subtitles, atomically, against expected_revision (the hand and Hermes alike)."""
    def work() -> dict[str, Any]:
        S = _subtitles_module("subtitles")
        directory, _state, timeline = _subtitles_cut(request.project_ref, request.timeline_id)
        rate, total, cuts, _fmt = _subtitles_cut_facts(timeline)
        with _subtitles_store().open(directory, request.project_ref.project_id) as tx:
            doc = tx.load()
            if request.expected_revision != doc["revision"]:
                return {"ok": False, "reason_code": "subtitles_conflict", "message": f"The subtitles are at revision {doc['revision']}.",
                        "revision": doc["revision"], "error": {"code": "subtitles_conflict", "message": f"The subtitles are at revision {doc['revision']}."}}
            entry = S.timeline_entry(doc, timeline["id"])
            entry, results = S.apply_ops(entry, request.ops, by=_film_caller(), rate=rate, cuts=cuts, total_frames=total)
            doc["timelines"][timeline["id"]] = entry
            doc["revision"] += 1
            tx.save(doc)
        _subtitles_announce(request.project_ref.project_id, doc["revision"])
        return _subtitles_answer(doc, timeline, {"results": results})
    try:
        return await asyncio.to_thread(work)
    except (studio_core.ProjectError, ValueError) as exc:
        return _subtitles_refusal(exc)


# --- Create subtitles from audio ------------------------------------------------------------------------------------

def _subtitles_asr_cache() -> Path:
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    return state_root(Path(base) / "hermes") / "subtitles-asr"


def _subtitles_transcribe_file(path: str, language: str | None) -> dict[str, Any] | None:
    """{language, words [{w, start, end} seconds]} of one sound file by the product's Words engine (voice/asr.py:
    faster-whisper large-v3, CPU int8, local files only), or None when no local recognizer can run."""
    spec = importlib.util.spec_from_file_location("film_suite_voice_asr_subtitles", Path(__file__).resolve().parent / "voice" / "asr.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    del language  # the recognizer detects the language itself (Resolve: Auto)
    return module.transcribe(path, model="large-v3", words=True, timeout=3600.0)


def _subtitles_words_of(path: str, sha: str, language: str | None) -> dict[str, Any] | None:
    """The cached words of a file (by content sha256), else a fresh transcription stored for next time."""
    cache = _subtitles_asr_cache()
    target = cache / f"{sha}.json"
    try:
        cached = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(cached, dict) and isinstance(cached.get("words"), list):
            return cached
    except (OSError, ValueError):
        pass
    with _SUBTITLES_ASR_LOCK:
        found = _subtitles_transcribe_file(path, language)
    if found is None:
        return None
    try:
        cache.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(f".{uuid.uuid4().hex}.tmp")
        temp.write_text(json.dumps({"language": found.get("language"), "words": found.get("words") or []}, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, target)
    except OSError:
        pass
    return found


def _subtitles_sources(timeline: dict[str, Any], source: str) -> list[dict[str, Any]]:
    """Every audible sound of the cut that is heard as it was recorded: sound clips of the chosen roles on unmuted
    tracks, and picture clips whose own sound is on (not retimed: their words would not sit where they are heard)."""
    roles = ("dialogue",) if source == "dialogue" else ("dialogue", "music", "effects")
    out = []
    for track in timeline.get("tracks") or ():
        if track.get("kind") == "audio" and track.get("role") in roles and not track.get("mute"):
            for item in track.get("items") or ():
                if item.get("enabled", True):
                    out.append({"ref": item["source"]["ref"], "record_start": item["record_start_frame"], "duration": item["duration_frames"],
                                "source_in": item["source_in_frame"], "source_rate": item["source_rate"], "kind": "sound"})
        if track.get("kind") == "video" and track.get("enabled", True):
            at = 0
            storyline = track.get("role") == "storyline"
            for item in track.get("items") or ():
                start = at if storyline else int(item.get("record_start_frame") or 0)
                at += int(item.get("duration_frames") or 0)
                if (item.get("type") == "clip" and item.get("enabled", True) and item.get("audio") == "on" and item.get("retime") is None
                        and (item.get("media_info") or {}).get("has_audio") and (item.get("media_info") or {}).get("frames") is not None):
                    out.append({"ref": item["source"]["ref"], "record_start": start, "duration": item["duration_frames"],
                                "source_in": item["source_in_frame"], "source_rate": item["source_rate"], "kind": "picture"})
    return out


def _subtitles_record_words(words: list[dict[str, Any]], src: dict[str, Any], rate: dict[str, int]) -> list[dict[str, Any]]:
    """Source-second words of one placed sound -> record-frame words inside the part of the source the cut uses."""
    from fractions import Fraction as _F
    sr = _F(src["source_rate"]["num"], src["source_rate"]["den"])
    tr = _F(rate["num"], rate["den"])
    head = _F(src["source_in"]) / sr
    tail = head + _F(src["duration"]) / tr
    out = []
    for w in words:
        a, b = _F(str(w["start"])), _F(str(w["end"]))
        mid = (a + b) / 2
        if not head <= mid < tail:
            continue
        start = src["record_start"] + int(math.floor((max(a, head) - head) * tr + _F(1, 2)))
        end = src["record_start"] + int(math.floor((min(b, tail) - head) * tr + _F(1, 2)))
        out.append({"w": str(w["w"]).strip(), "start": start, "end": max(end, start + 1)})
    return [w for w in out if w["w"]]


def _subtitles_dedupe(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One word where two sources carry the same speech (a picture's sound and its linked dialogue track)."""
    out: list[dict[str, Any]] = []
    for w in sorted(words, key=lambda x: (x["start"], x["end"])):
        if out and w["start"] < out[-1]["end"] and (min(w["end"], out[-1]["end"]) - w["start"]) * 2 >= (w["end"] - w["start"]):
            continue
        out.append(w)
    return out


def _subtitles_transcribe_work(request: SubtitlesTranscribeRequest, by: str, progress: Any) -> dict[str, Any]:
    S = _subtitles_module("subtitles")
    directory, state, timeline = _subtitles_cut(request.project_ref, request.timeline_id)
    rate, total, cuts, _fmt = _subtitles_cut_facts(timeline)
    lo = request.range.in_frame if request.range else 0
    hi = request.range.out_frame if request.range else total
    if hi <= lo:
        raise S.SubtitleRefusal("invalid_range", "The range needs in < out.")
    resolver = _timeline_module("resolver")
    lookups = _timeline_lookups(request.project_ref.project_id)
    lookups.pop("music_file", None)
    sources = [s for s in _subtitles_sources(timeline, request.source) if s["record_start"] < hi and s["record_start"] + s["duration"] > lo]
    if not sources:
        raise S.SubtitleRefusal("no_dialogue", "No audible dialogue in the range." if request.source == "dialogue" else "No audible sound in the range.")
    words: list[dict[str, Any]] = []
    files: dict[str, dict[str, Any] | None] = {}
    language = None
    for n, src in enumerate(sources):
        progress(n / max(1, len(sources)), "transcribe")
        fact = resolver.resolve_ref(src["ref"], **lookups)
        if fact.get("status") != "online" or not fact.get("path"):
            raise S.SubtitleRefusal("source_offline", f"{src['ref']} is offline ({fact.get('reason') or 'offline'}).")
        path = fact["path"]
        if path not in files:
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            files[path] = _subtitles_words_of(path, digest.hexdigest(), request.language)
            if files[path] is None:
                raise S.SubtitleRefusal("asr_unavailable", "The local speech recognizer (faster-whisper large-v3) is not installed.")
            language = language or files[path].get("language")
        words += _subtitles_record_words(files[path].get("words") or [], src, rate)
    words = [w for w in _subtitles_dedupe(words) if lo <= w["start"] < hi]
    progress(0.95, "split")
    with _subtitles_store().open(directory, request.project_ref.project_id) as tx:
        doc = tx.load()
        fresh_track = timeline["id"] not in doc["timelines"]
        entry = S.timeline_entry(doc, timeline["id"], create=True)
        if fresh_track and request.preset:
            entry["track"] = S.default_track(request.preset)
        new = S.build_cues(words, rate=rate, rules=entry["track"]["rules"], cuts=cuts, limit_end=min(total, hi),
                           taken=[c["id"] for c in entry["cues"]])
        entry["cues"], counts = S.merge_transcription(entry["cues"], new, lo, hi)
        doc["revision"] += 1
        tx.save(doc)
    _subtitles_announce(request.project_ref.project_id, doc["revision"])
    return {"revision": doc["revision"], "timeline_id": timeline["id"], "range": {"in_frame": lo, "out_frame": hi}, "words": len(words),
            "language": language, "cues": len(entry["cues"]), "by": by, **counts}


def _subtitles_job_public(job: dict[str, Any]) -> dict[str, Any]:
    return {k: job[k] for k in ("job_id", "state", "progress", "stage", "result", "error")}


@project_router.post("/projects/subtitles/transcribe")
async def projects_subtitles_transcribe(request: SubtitlesTranscribeRequest) -> dict[str, Any]:
    """Create subtitles from audio: a job that transcribes the cut's sound locally and writes the cues."""
    try:
        _subtitles_cut(request.project_ref, request.timeline_id)
    except (studio_core.ProjectError, ValueError) as exc:
        return _subtitles_refusal(exc)
    by = _film_caller()
    job_id = "subtx_" + uuid.uuid4().hex
    job = {"job_id": job_id, "state": "running", "progress": 0.0, "stage": "start", "result": None, "error": None, "done": _thread.allocate_lock()}
    job["done"].acquire()
    with _SUBTITLES_JOBS_LOCK:
        _SUBTITLES_JOBS[job_id] = job

    def progress(fraction: float, stage: str) -> None:
        job["progress"], job["stage"] = round(float(fraction), 3), stage

    def run() -> None:
        try:
            job["result"] = _subtitles_transcribe_work(request, by, progress)
            job["state"], job["progress"], job["stage"] = "done", 1.0, "done"
        except Exception as exc:  # noqa: BLE001 - a job answers its failure, never raises
            refusal = _subtitles_refusal(exc) if isinstance(exc, (studio_core.ProjectError, ValueError)) else {"reason_code": "transcribe_failed", "message": "The transcription failed."}
            job["state"], job["error"] = "failed", {"code": refusal["reason_code"], "message": refusal.get("message") or "The transcription failed."}
        finally:
            job["done"].release()

    _thread.start_new_thread(run, ())
    if request.wait_ms:
        await asyncio.to_thread(job["done"].acquire, True, request.wait_ms / 1000.0) and job["done"].release()
    return {"ok": job["state"] != "failed", "reason_code": job["error"]["code"] if job["error"] else None, **_subtitles_job_public(job)}


@project_router.post("/projects/subtitles/transcribe/status")
async def projects_subtitles_transcribe_status(request: SubtitlesStatusRequest) -> dict[str, Any]:
    with _SUBTITLES_JOBS_LOCK:
        job = _SUBTITLES_JOBS.get(request.job_id)
    if job is None:
        return {"ok": False, "reason_code": "job_unknown", "message": "No such transcription in this process.", "error": {"code": "job_unknown", "message": "No such transcription in this process."}}
    if request.wait_ms and job["state"] == "running":
        await asyncio.to_thread(job["done"].acquire, True, request.wait_ms / 1000.0) and job["done"].release()
    return {"ok": job["state"] != "failed", "reason_code": job["error"]["code"] if job["error"] else None, **_subtitles_job_public(job)}


# --- Export SRT / WebVTT, and the viewer's picture of one cue -------------------------------------------------------

def _subtitles_entry_for_export(project_ref: ProjectReference, timeline_id: str | None, frame_range: Any) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    S = _subtitles_module("subtitles")
    directory, _state, timeline = _subtitles_cut(project_ref, timeline_id)
    with _subtitles_store().open(directory, project_ref.project_id) as tx:
        doc = tx.load()
    entry = S.timeline_entry(doc, timeline["id"])
    if frame_range is not None:
        entry = S.crop(entry, frame_range.in_frame, frame_range.out_frame)
    return directory, timeline, entry, doc


@project_router.post("/projects/subtitles/export")
async def projects_subtitles_export(request: SubtitlesExportRequest) -> dict[str, Any]:
    """Write the cut's subtitles as SRT or WebVTT into the project output folder (or a chosen approved folder)."""
    def work() -> dict[str, Any]:
        S = _subtitles_module("subtitles")
        directory, timeline, entry, doc = _subtitles_entry_for_export(request.project_ref, request.timeline_id, request.range)
        if not entry["cues"]:
            raise S.SubtitleRefusal("no_subtitles", "The cut has no subtitles in the range.")
        rate = dict(timeline["rate"])
        writer = S.srt if request.format == "srt" else S.vtt
        text = writer(entry["cues"], rate, entry["track"]["rules"], entry["track"]["style"])
        manifest = _project_store_factory().read_project(directory, expected_project_id=request.project_ref.project_id)
        project_dir = Path(os.path.realpath(directory))
        root = Path(os.path.realpath(_project_output_dir(directory, request.project_ref.project_id, create=True)))
        if request.folder is not None:
            try:
                root = resolve_approved_path(request.folder, [*manifest["approved_roots"], str(project_dir)])
            except (OSError, ValueError):
                raise S.SubtitleRefusal("folder_outside_roots", "The folder is outside the project's approved folders.") from None
        base = _subtitles_module("export_package").safe_name(request.name or timeline.get("name") or "subtitles")
        target = root / f"{base}.{request.format}"
        n = 2
        while target.exists():
            target = root / f"{base}-{n}.{request.format}"
            n += 1
        target.write_bytes(text.encode("utf-8"))
        shown = Path(os.path.relpath(target, project_dir)).as_posix() if Path(os.path.realpath(target)).is_relative_to(project_dir) else target.name
        return {"ok": True, "reason_code": None, "file": shown, "format": request.format, "cues": len(entry["cues"]),
                "revision": doc["revision"], "text": text}
    try:
        return await asyncio.to_thread(work)
    except (studio_core.ProjectError, ValueError, OSError) as exc:
        return _subtitles_refusal(exc) if not isinstance(exc, OSError) else {"ok": False, "reason_code": "export_failed", "message": "The file could not be written."}


@project_router.post("/projects/subtitles/frame")
async def projects_subtitles_frame(request: SubtitlesFrameRequest) -> dict[str, Any]:
    """The frame-sized transparent PNG of one cue (and its active word) at the cut's format: the same picture the
    burn-in lays over the film (subtitle_raster), so the viewer shows what is exported."""
    def work() -> dict[str, Any]:
        S = _subtitles_module("subtitles")
        raster = _subtitles_module("subtitle_raster")
        directory, timeline, entry, doc = _subtitles_entry_for_export(request.project_ref, request.timeline_id, None)
        cue = next((c for c in entry["cues"] if c["id"] == request.cue), None)
        if cue is None:
            raise S.SubtitleRefusal("cue_unknown", f"No cue {request.cue}.")
        _rate, _total, _cuts, fmt = _subtitles_cut_facts(timeline)
        key = raster.state_key(entry["track"], cue, request.active, fmt["width"], fmt["height"])
        data = raster.png(entry["track"], cue, request.active, fmt["width"], fmt["height"])
        return {"ok": True, "reason_code": None, "key": key, "width": fmt["width"], "height": fmt["height"],
                "png_base64": base64.b64encode(data).decode("ascii"), "revision": doc["revision"]}
    try:
        return await asyncio.to_thread(work)
    except (studio_core.ProjectError, ValueError) as exc:
        return _subtitles_refusal(exc)


def _subtitles_for_export(project_id: str, directory: str, timeline_id: str | None, document: dict[str, Any]) -> dict[str, Any] | None:
    """The stored subtitle entry of the exported cut (None when it has no cues): read by the export route."""
    S = _subtitles_module("subtitles")
    state = document["state"] if "state" in document else document
    wanted = timeline_id or state.get("active")
    try:
        with _subtitles_store().open(directory, project_id) as tx:
            doc = tx.load()
    except (studio_core.ProjectError, ValueError):
        return None
    entry = doc["timelines"].get(wanted)
    return entry if entry and entry["cues"] else None
