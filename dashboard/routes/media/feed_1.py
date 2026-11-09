from __future__ import annotations


# --- Film Suite page: one read-only Media feed over both ledgers ------------
#
# The Media screen shows what the scope has MADE: published Image results
# (kept or not, with the prompt and render spec the ledger recorded) and the
# project's finished Shoot clips. Nothing here writes, polls a provider or
# returns a local path; pictures come through the picture route, clip posters
# through /media/feed/poster, both re-checking ownership.

_MEDIA_FEED_SCHEMA = "studio.media-feed/1"
_MEDIA_FEED_VIDEO_SETTINGS = (
    "duration_seconds", "seed", "aspect_ratio", "resolution", "fps", "audio", "generate_audio", "grounding", "version",
)


class MediaFeedRequest(JobsListRequest):
    """The Media library view: the same three facets, sort and search the
    artist sets in Media, so an agent reads exactly the grid the artist sees.
    ``before`` is the previous answer's ``next_before`` (one page after it)."""
    schema_: Literal["studio.media-feed/1"] = Field(alias="schema")
    limit: int = Field(default=120, ge=1, le=500)
    kind: Literal["all", "image", "video", "audio"] = "all"
    # Rejected items are hidden unless mark is "rejected" (only them) or
    # include_rejected is true (everything, e.g. for an agent's full listing).
    mark: Literal["any", "kept", "rated", "rejected"] = "any"
    min_rating: int = Field(default=1, ge=1, le=5)
    include_rejected: bool = False
    query: str = Field(default="", max_length=200)
    sort: Literal["newest", "oldest", "rating", "kept", "name"] = "newest"
    before: str | None = Field(default=None, min_length=1, max_length=600)


def _load_take_ids_module() -> Any:
    """generation/take_ids.py, the ledger's one take-id definition, by path (not a package member)."""
    source = Path(__file__).with_name("generation") / "take_ids.py"
    spec = importlib.util.spec_from_file_location("film_suite_take_ids", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"The take id definition is unavailable: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_take_ids_module = _load_take_ids_module()


class MediaFeedPosterRequest(_ProjectRequest):
    schema_: Literal["studio.media-feed/1"] = Field(alias="schema")
    kind: Literal["media_feed_poster"]
    owner_scope: str = Field(pattern=_OWNER_SCOPE_PATTERN, min_length=18, max_length=47)
    studio_job_id: str = Field(pattern=_take_ids_module.STUDIO_JOB_ID_PATTERN, min_length=39, max_length=43)


def _iso_epoch(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return _finite_epoch(parsed.timestamp())


def _kept_image_index(states: Any) -> dict[str, dict[str, Any]]:
    """producer job id -> {name, kept_at} for every kept revision in scope."""
    kept: dict[str, dict[str, Any]] = {}
    for context in states.enumerate_contexts():
        for work in (context.get("state") or {}).get("works") or ():
            for revision in work.get("saved_revisions") or ():
                content = revision.get("content") or {}
                selected = revision.get("selected_artifact")
                for item in content.get("alternatives") or ():
                    if not (isinstance(item, dict) and item.get("origin") == "generated" and item.get("material_id") == selected):
                        continue
                    job_id = item.get("producer_job_id")
                    at = _iso_epoch(revision.get("saved_at"))
                    known = kept.get(job_id)
                    if isinstance(job_id, str) and (known is None or (at or 0) > (known["kept_at"] or 0)):
                        kept[job_id] = {"name": _job_label(work.get("name")) or None, "kept_at": at}
    return kept


def _material_time(material_root: Path | None, relative: Any) -> float | None:
    """When the picture was brought home: the published file's mtime. Only a
    number leaves this function; the path never does."""
    if material_root is None or not isinstance(relative, str) or not relative:
        return None
    try:
        root = Path(material_root).resolve()
        candidate = (root / relative).resolve()
        candidate.relative_to(root)
        return _finite_epoch(candidate.stat().st_mtime)
    except (OSError, ValueError):
        return None


def _feed_project_is_recent(project_id: str) -> bool:
    try:
        entries, _snapshot = _recent_project_store_factory()._read_entries()
    except Exception:  # noqa: BLE001 - unreadable registry: not recent
        return False
    return any(isinstance(e, dict) and e.get("project_id") == project_id for e in entries or ())


def _image_feed_root(owner_scope: str) -> Any:
    """The profile root for the read-only feed.

    A recent project not yet opened in this session still has its pictures (the
    Start screen counts them before anything is opened), so a recent project's
    scope may bring the root up here, as scratch does. Any other scope keeps the
    producer rule: only a project the application already opened.
    """
    if owner_scope != _SCRATCH_SCOPE and _feed_project_is_recent(owner_scope[len("film_owner:"):]):
        return _image_film_root()
    return _image_producer_root(owner_scope)


def _image_feed_handoff(root: Any, owner_scope: str) -> Any:
    """An opened project reads through its bundle; a recent, unopened one through
    the read-only owner view (no project directory, nothing writable)."""
    if owner_scope == _SCRATCH_SCOPE:
        return root.scratch_operations()
    project_id = owner_scope[len("film_owner:"):]
    try:
        return root.read_cached_project_operations(project_id)
    except Exception:  # noqa: BLE001 - not opened in this session
        if not _feed_project_is_recent(project_id) or not hasattr(root, "read_project_media"):
            raise
        return root.read_project_media(project_id)


def _image_media_for_scope(owner_scope: str, limit: int, window: dict[str, int] | None = None) -> list[dict[str, Any]]:
    with _IMAGE_FILM_ROOT_LOCK:
        root = _image_feed_root(owner_scope)
        handoff = _image_feed_handoff(root, owner_scope)
        jobs = handoff.producer_jobs()
        states = handoff.image_states()
        try:
            _collector, reader = handoff.result_services()
            material_root = getattr(reader, "material_root", None)
        except Exception:  # noqa: BLE001 - times are informative, never required
            material_root = None
    try:
        kept = _kept_image_index(states)
    except Exception:  # noqa: BLE001 - kept marks degrade to "not kept"
        kept = {}
    items: list[dict[str, Any]] = []
    rows = jobs.list_recent(limit=500)  # the ledgers cap a window at 500
    if window is not None:
        window["rows"] = len(rows)
    for job in rows:
        if job.get("state") != "published":
            continue
        snapshot = job.get("request_snapshot") or {}
        if snapshot.get("operation") == "edit":
            # A Generative edit publishes the model's raw crop; it lands in the
            # picture as a layer (/image/edit/result), never as a picture of its own.
            continue
        basis = snapshot.get("context_basis") or {}
        try:
            states.require_owned(basis.get("context_id"))
        except Exception:  # noqa: BLE001 - not owned: not listed
            continue
        job_id = job["image_producer_job_id"]
        artist = snapshot.get("artist_input") or {}
        artifact = job.get("artifact") or {}
        mark = kept.get(job_id)
        prompt = str(artist.get("prompt") or "")
        items.append({
            "kind": "image",
            "module": "image",
            "item_id": job_id,
            "label": (mark and mark["name"]) or _job_label(prompt) or "Untitled",
            "prompt": prompt,
            "render_spec": _job_render_spec(job),
            "width": artifact.get("width") if type(artifact.get("width")) is int else artist.get("width"),
            "height": artifact.get("height") if type(artifact.get("height")) is int else artist.get("height"),
            "target_id": (job.get("provider") or {}).get("target_id"),
            "finished_at": _material_time(material_root, job.get("publication_path")),
            "kept": mark is not None,
            "kept_name": mark["name"] if mark else None,
            "kept_at": mark["kept_at"] if mark else None, **_media_workflow_extras(job, snapshot), **({"source_sha256": src} if isinstance(src := ((snapshot.get("effective_input") or {}).get("imported") or {}).get("source_sha256"), str) else {}),  # w25-cutfix: one tile with its folder copy (same bytes)
        })
        if len(items) >= limit:
            break
    return items


def _video_setting(value: Any) -> Any:
    if isinstance(value, bool) or (isinstance(value, int) and not isinstance(value, bool)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, str) and len(value) <= 40 and not any(mark in value for mark in ("/", "\\", ":")):
        return value
    return None


def _video_media_for_scope(project_id: str | None, limit: int, window: dict[str, int] | None = None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    rows = _comfy_store().list_recent(project_id=project_id, limit=500)
    music = _music_gen_seconds_by_job(project_id) if rows else {}  # w17-acre MED-056: a music take's measured length
    if window is not None:
        window["rows"] = len(rows)
    for record in rows:
        if record.state != "ready":
            continue
        try:
            args = dict(record.request.get("args") or {})
        except Exception:  # noqa: BLE001 - an unreadable request still lists
            args = {}
        prompt = str(args.get("prompt") or "")
        # A finished generated sound (a voice take later) is a take whose file is a sound file:
        # kind "audio", the same fields, the same id as its recipe link (lane fs-msnd).
        sound = _media_take_is_sound(record)
        settings = {**{key: _video_setting(args[key]) for key in _MEDIA_FEED_VIDEO_SETTINGS if key in args and _video_setting(args[key]) is not None},
                    **({"duration_seconds": music[record.studio_job_id]} if sound and record.studio_job_id in music else {})}
        items.append({
            "kind": "audio" if sound else "video",
            "module": "shoot",
            "item_id": record.studio_job_id,
            "label": _job_label(prompt or record.requested_basename) or ("Sound" if sound else "Clip"),
            "prompt": prompt,
            "settings": settings,
            "target_id": record.target_id,
            "finished_at": _finite_epoch(record.completed_at),
            "has_file": _media_take_has_file(record),  # lane w24-rest (VID-030/035): a V4 take's file too
            "kept": False,
            "kept_name": None,
            "kept_at": None,
        })
        if len(items) >= limit:
            break
    return items


def _media_take_is_sound(record: Any) -> bool:
    """A take whose finished (or requested) file is a sound file (MEDIA_EXTENSIONS kind audio)."""
    for name in (getattr(record, "published_path", None), getattr(record, "requested_basename", None)):
        if isinstance(name, str) and name:
            return MEDIA_EXTENSIONS.get(os.path.splitext(name)[1].lower()) == "audio"
    return False


def _media_sort_key(item: Mapping[str, Any]) -> float:
    for key in ("finished_at", "kept_at"):
        value = item.get(key)
        if isinstance(value, float):
            return value
    return float("-inf")


@project_router.post("/media/feed")
async def media_feed(request: MediaFeedRequest) -> dict[str, Any]:
    """Media screen feed: what one owner scope has made, as the artist's
    library view (type, mark, search, sort) one page at a time.

    Read-only. Each source degrades on its own. A generated sound is a Shoot
    take with a sound file (kind "audio"); there is no separate audio source. The
    ledgers are read in their newest 500-row window; ``window_limited`` says
    when older results exist beyond it."""
    sources: dict[str, str] = {}
    items: list[dict[str, Any]] = []
    limited = False
    try:
        found, full = await asyncio.to_thread(_image_media_window, request.owner_scope)  # w29-fix2: off the loop (ledger + decode)
        items.extend(found)
        limited = limited or full
        sources["image"] = "ok"
    except Exception:  # noqa: BLE001 - reported as a source state
        sources["image"] = "unavailable"
    try:
        found, full = await asyncio.to_thread(_video_media_window, request.project_id)
        items.extend(found)
        limited = limited or full
        sources["shoot"] = "ok"
    except Exception:  # noqa: BLE001 - reported as a source state
        sources["shoot"] = "unavailable"
    try:
        marks = (await asyncio.to_thread(_media_marks_read, request.owner_scope))["items"]
        marks_state = "ok"
    except Exception:  # noqa: BLE001 - marks degrade to "unmarked"
        marks = {}
        marks_state = "unavailable"
    # Stable: equal/unknown times keep ledger order (itself newest first).
    items.sort(key=_media_sort_key, reverse=True)
    for seq, item in enumerate(items):
        _merge_media_mark(item, marks.get(item["item_id"]))
        item["_seq"] = seq
    for item in items:
        item["bin"] = _media_item_bin(item)
    try:
        view = _media_view(items, request)
    except ValueError:
        return _media_feed_error(400, "media_cursor_invalid", "That page marker is not valid for this view.")
    bin_counts = _media_bin_counts(items, request)
    return {"schema": _MEDIA_FEED_SCHEMA, "owner_scope": request.owner_scope, "sources": sources, **view, "bin_counts": bin_counts,
            "marks": marks_state, "window_limited": limited}


# --- Bins (lane w5-backend item 4): the column's bin counts, exact over the whole window -------------
#
# Each item carries ``bin``: rejected | kept | imported | None (only in All media). ``bin_counts``
# answers every bin over the scope's whole window (never one page, never one type or search):
# all (not rejected), kept, imported, rejected, and running / failed from the job ledgers (the
# work not yet a result). failed = failed and not sent again: a failed job is superseded when a
# later job of the same module, target and request label exists (Retry sends the same request).

_MEDIA_IMPORT_TARGET_ID = "filmsuite-import-v1"
_MEDIA_JOB_WINDOW = 200


def _media_item_bin(item: Mapping[str, Any]) -> str | None:
    if item.get("rejected"):
        return "rejected"
    if item.get("kept"):
        return "kept"
    if item.get("target_id") == _MEDIA_IMPORT_TARGET_ID:
        return "imported"
    return None


def _media_bin_jobs(request: MediaFeedRequest) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    try:
        jobs.extend(_image_jobs_for_scope(request.owner_scope, _MEDIA_JOB_WINDOW))
    except Exception:  # noqa: BLE001 - an unavailable ledger counts nothing
        pass
    try:
        jobs.extend(_video_jobs_for_scope(request.project_id, _MEDIA_JOB_WINDOW))
    except Exception:  # noqa: BLE001
        pass
    return jobs


def _media_bin_counts(items: list[dict[str, Any]], request: MediaFeedRequest, jobs: list[dict[str, Any]] | None = None) -> dict[str, int]:
    published = {item["item_id"] for item in items}
    rows = [job for job in (jobs if jobs is not None else _media_bin_jobs(request)) if job.get("job_id") not in published]
    running = sum(1 for job in rows if job.get("state") in ("queued", "running"))

    # Each ledger lists newest first: a job sent again sits before the failed one in its own module.
    def sent_again(index: int, job: Mapping[str, Any]) -> bool:
        return any(
            other.get("module") == job.get("module") and other.get("target_id") == job.get("target_id")
            and other.get("label") == job.get("label")
            for other in rows[:index]
        )
    failed = sum(1 for index, job in enumerate(rows) if job.get("state") == "failed" and not sent_again(index, job))
    return {
        "all": sum(1 for item in items if not item.get("rejected")),
        "kept": sum(1 for item in items if item.get("bin") == "kept"),
        "imported": sum(1 for item in items if item.get("bin") == "imported" or (not item.get("rejected") and item.get("target_id") == _MEDIA_IMPORT_TARGET_ID)),
        "rejected": sum(1 for item in items if item.get("rejected")),
        "running": running,
        "failed": failed,
    }


# --- Library view: facets, search, sort, cursor ------------------------------

_MEDIA_WINDOW = 500  # the ledgers cap one read at 500 rows


def _image_media_window(owner_scope: str) -> tuple[list[dict[str, Any]], bool]:
    window = {"rows": 0}
    items = _image_media_for_scope(owner_scope, _MEDIA_WINDOW, window)
    return items, window["rows"] >= _MEDIA_WINDOW


def _video_media_window(project_id: str | None) -> tuple[list[dict[str, Any]], bool]:
    window = {"rows": 0}
    items = _video_media_for_scope(project_id, _MEDIA_WINDOW, window)
    return items, window["rows"] >= _MEDIA_WINDOW


def _merge_media_mark(item: dict[str, Any], mark: Any) -> None:
    mark = mark if isinstance(mark, dict) else {}
    item["rating"] = mark.get("rating", 0) if type(mark.get("rating")) is int else 0
    item["rejected"] = mark.get("rejected") is True
    item["tags"] = [tag for tag in mark.get("tags") or () if isinstance(tag, str)]
    if item["kind"] in ("video", "audio") and mark.get("kept") is True:
        # A clip is kept in the marks; a picture stays kept through Image.
        item["kept"] = True
        at = mark.get("kept_at")
        item["kept_at"] = float(at) if isinstance(at, (int, float)) and not isinstance(at, bool) and math.isfinite(at) else None


def _media_time(item: Mapping[str, Any]) -> float:
    value = _media_sort_key(item)
    return value if math.isfinite(value) else float("-inf")


def _media_order_key(item: Mapping[str, Any], sort: str) -> tuple:
    """Ascending tuple = display order. Unknown times always sort last."""
    t = _media_time(item)
    known = 0 if math.isfinite(t) else 1
    if sort == "oldest":
        return (known, t if known == 0 else 0.0, item["_seq"])
    newest = (known, -t if known == 0 else 0.0, item["_seq"])
    if sort == "rating":
        return (-int(item.get("rating") or 0), *newest)
    if sort == "kept":
        return (0 if item.get("kept") else 1, *newest)
    if sort == "name":  # w24-media (Premiere's Name sort): digit runs by number, text by code point, 64 floats padded
        return (*([x for r in re.findall(r"\d+|\D+", str(item.get("kept_name") or item.get("label") or "").casefold()) for x in ((0.0, float(int(r[:15]))) if r.isdigit() else (1.0, *map(float, map(ord, r))))] + [-1.0] * 64)[:64], *newest)
    return newest


def _media_matches(item: Mapping[str, Any], request: MediaFeedRequest, *, facet_kind: bool = True, facet_mark: bool = True) -> bool:
    if facet_kind and request.kind != "all" and item["kind"] != request.kind:
        return False
    if request.mark == "rejected":
        if facet_mark and not item.get("rejected"):
            return False
    elif item.get("rejected") and not request.include_rejected:
        return False
    if facet_mark and request.mark == "kept" and not item.get("kept"):
        return False
    if facet_mark and request.mark == "rated" and int(item.get("rating") or 0) < request.min_rating:
        return False
    terms = request.query.casefold().split()
    if terms:
        text = " ".join([str(item.get("prompt") or ""), str(item.get("label") or ""), str(item.get("kept_name") or ""), *item.get("tags", ())]).casefold()
        if not all(term in text for term in terms):
            return False
    return True


def _media_cursor(item: Mapping[str, Any], sort: str) -> str:
    key = list(_media_order_key(item, sort)[:-1])
    raw = json.dumps({"id": item["item_id"], "sort": sort, "key": [k if math.isfinite(k) else None for k in key]}, separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")


def _media_after_cursor(ordered: list[dict[str, Any]], before: str, sort: str) -> int:
    """Index of the first item after the cursor. The cursor names the last
    item of the previous page; if that item left the view (a new mark), the
    page continues at the first item that sorts after its recorded key."""
    try:
        padded = before + "=" * (-len(before) % 4)
        value = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
        item_id, key = value["id"], value["key"]
        if value.get("sort") != sort or not isinstance(item_id, str) or not isinstance(key, list):
            raise ValueError
        key = tuple(float("inf") if k is None else float(k) for k in key)
    except Exception:  # noqa: BLE001 - finite request error
        raise ValueError("media-cursor-invalid") from None
    for index, item in enumerate(ordered):
        if item["item_id"] == item_id:
            return index + 1
    for index, item in enumerate(ordered):
        if _media_order_key(item, sort)[:-1] > key:
            return index
    return len(ordered)


def _media_view(items: list[dict[str, Any]], request: MediaFeedRequest) -> dict[str, Any]:
    # Counts answer "how many would this chip show" with the other facets set.
    counts = {
        "all": sum(_media_matches(i, request, facet_kind=False) for i in items),
        "image": sum(_media_matches(i, request, facet_kind=False) and i["kind"] == "image" for i in items),
        "video": sum(_media_matches(i, request, facet_kind=False) and i["kind"] == "video" for i in items),
        "audio": sum(_media_matches(i, request, facet_kind=False) and i["kind"] == "audio" for i in items),
    }
    # Mark chips: the same view with the mark facet swapped (type and search kept).
    for mark in ("any", "kept", "rated", "rejected"):
        swapped = request.model_copy(update={"mark": mark})
        counts["mark_" + mark] = sum(_media_matches(i, swapped) for i in items)
    cleared = request.model_copy(update={"mark": "any", "kind": "all"})
    counts["everything"] = sum(_media_matches(i, cleared) for i in items)
    ordered = sorted((i for i in items if _media_matches(i, request)), key=lambda i: _media_order_key(i, request.sort))
    start = _media_after_cursor(ordered, request.before, request.sort) if request.before else 0
    page = ordered[start:start + request.limit]
    more = start + len(page) < len(ordered)
    out = []
    for item in page:
        cursor = _media_cursor(item, request.sort)
        clean = {k: v for k, v in item.items() if k != "_seq"}
        out.append((clean, cursor))
    return {
        "items": [clean for clean, _ in out],
        "total": len(ordered),
        "counts": counts,
        "has_more": more,
        "next_before": out[-1][1] if more and out else None,
    }


def _video_poster_jpeg(path: str, width: int) -> bytes:
    fd, name = tempfile.mkstemp(prefix="flux3-poster-", suffix=".jpg")
    os.close(fd)
    temporary = Path(name)
    try:
        _run([
            str(FFMPEG_PATH), "-hide_banner", "-loglevel", "error", "-ss", "0.1", "-i", str(path),
            "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "4", "-f", "image2", "-vcodec", "mjpeg",
            "-pix_fmt", "yuvj420p", "-y", str(temporary),
        ])
        jpeg = temporary.read_bytes()
        if not jpeg.startswith(b"\xff\xd8"):
            raise RecipeError("ffmpeg produced no poster frame.")
        return jpeg
    finally:
        temporary.unlink(missing_ok=True)


def _media_feed_error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def _shoot_take(studio_job_id: str, project_id: str | None = None) -> Any:
    """One Shoot take by id with its finished file as ``published_path``, either contract (Contract 3 carries the
    path; a V4 row's file is the one its operation record published, re-verified by size and sha256, else None).
    A ledger row wins; with a ``project_id`` an id the ledger lacks resolves from the project (``_project_take``)."""
    try:
        return _shoot_take_ledger(studio_job_id)
    except _comfy_job_store_module.JobNotFoundError:
        if project_id is None:
            raise
    return _project_take(studio_job_id, project_id)


_PROJECT_TAKE_RECIPE_MAX = 1 << 20
_TAKE_SHA: dict[tuple[str, int, int], str] = {}  # sha256 per (path, size, mtime): hashed once, never per frame/poster


def _project_take(studio_job_id: str, project_id: str) -> Any:
    """p8-cut/b (CUT-104/VID-106; project decision 2, 29.10.): the Video state's ready row (savedPath inside the project or an
    approved folder, its recipe naming the id), or no row: the recipe alone. Sidecar self_id = sha256. Else not_found."""
    missing = _comfy_job_store_module.JobNotFoundError
    try:
        directory = _project_reference_directory(ProjectReference(project_id=project_id))
        manifest = (store := _project_store_factory()).read_project(directory, expected_project_id=project_id)
        jobs = store.read_shoot_state(directory, project_id=project_id)["state"].get("jobs")
        roots = [Path(directory).resolve(strict=True)] + [Path(root).resolve(strict=True) for root in (manifest.get("approved_roots") or [])
                                                         if isinstance(root, str) and Path(root).is_dir()]
    except Exception:  # noqa: BLE001 - no readable project: the take is unknown
        raise missing() from None
    row = next((job for job in jobs if isinstance(job, dict) and job.get("id") == studio_job_id), None) if isinstance(jobs, list) else None
    if row is None:
        return _recipe_take(studio_job_id, project_id, roots)
    saved = row.get("savedPath")
    try:
        path = Path(saved).resolve(strict=True)
        recipe = path.with_name(path.stem + ".flux3.json")
        if (row.get("status") != "ready" or not Path(saved).is_absolute() or not path.is_file() or not any(path.is_relative_to(r) for r in roots)
                or not recipe.is_file() or recipe.stat().st_size > _PROJECT_TAKE_RECIPE_MAX):
            raise missing()
        named = json.loads(recipe.read_text(encoding="utf-8"))
        if not isinstance(named, dict) or named.get("job_id") != studio_job_id or not _take_sidecar_ok(path):
            raise missing()
    except Exception:  # noqa: BLE001 - unreadable file, recipe or sidecar, or refused: unknown
        raise missing() from None
    return SimpleNamespace(studio_job_id=studio_job_id, project_id=project_id, row_contract=4, state="ready",
                           published_path=str(path), request={"project_id": project_id})


def _take_sha256(path: Path) -> str:
    if (key := (str(path), (st := path.stat()).st_size, st.st_mtime_ns)) not in _TAKE_SHA:
        _TAKE_SHA.pop(next(iter(_TAKE_SHA))) if len(_TAKE_SHA) >= 256 else None
        _TAKE_SHA[key] = _h3_mctx_module.sha256_file(path)
    return _TAKE_SHA[key]


def _take_sidecar_ok(path: Path) -> bool:
    return not (car := _h3_mctx_module.sidecar_path(path)).exists() or _h3_mctx_module.read_header(car).get("self_id") == _take_sha256(path)


def _recipe_take(studio_job_id: str, project_id: str, roots: list[Path]) -> Any:
    """p8-cutb: a recipe in output/ or an approved folder (depth 1, max 500, size cap) naming the id + sha256 of <stem>.mp4."""
    try:
        scan = ([out.resolve()] if (out := roots[0] / "output").is_dir() and out.resolve().is_relative_to(roots[0]) else []) + roots[1:]
        names = [(root, e.name) for root in scan for e in sorted(os.scandir(root), key=lambda e: e.name) if e.name.endswith(".flux3.json")]
    except OSError:  # an unreadable folder: the take is unknown
        raise _comfy_job_store_module.JobNotFoundError() from None
    for root, name in names[:500]:
        try:
            recipe, path = (root / name).resolve(strict=True), (root / (name[:-11] + ".mp4")).resolve(strict=True)
            if {recipe.parent, path.parent} != {root} or not path.is_file() or recipe.stat().st_size > _PROJECT_TAKE_RECIPE_MAX:
                continue
            shas, stack = set(), [named := json.loads(recipe.read_text(encoding="utf-8"))]
            while stack:  # published_file_sha256 sits somewhere in the document
                node = stack.pop()
                shas |= {node["published_file_sha256"]} if isinstance(node, dict) and isinstance(node.get("published_file_sha256"), str) else set()
                stack += list(node.values()) if isinstance(node, dict) else node if isinstance(node, list) else []
            if (isinstance(named, dict) and named.get("job_id") == studio_job_id and len(shas) == 1 and _take_sha256(path) in shas
                    and _take_sidecar_ok(path)):
                return SimpleNamespace(studio_job_id=studio_job_id, project_id=project_id, row_contract=4, state="ready",
                                       published_path=str(path), request={"project_id": project_id})
        except Exception:  # noqa: BLE001 - unreadable recipe or file: the next recipe
            continue
    raise _comfy_job_store_module.JobNotFoundError()


def _shoot_take_ledger(studio_job_id: str) -> Any:
    """The ledger part of ``_shoot_take`` (base behaviour): Contract 3 by get_job, Contract 4 by its row."""
    try:
        return _comfy_store().get_job(studio_job_id)
    except Exception as exc:  # noqa: BLE001 - only the Contract-3 guard falls through
        if getattr(exc, "reason_code", None) != "project_scope_mismatch":
            raise
    row = _read_ledger_row(studio_job_id)
    if getattr(row, "row_contract", None) != 4:
        raise _comfy_job_store_module.JobNotFoundError()
    path = None
    if row.state == "ready":
        try:
            ops = _private_ops()
            path = ops.saved_path_for_receipt(ops.load_for_job(row))
        except (PrivateOperationError, OSError, ValueError):
            path = None
    return SimpleNamespace(studio_job_id=row.studio_job_id, project_id=row.project_id, row_contract=4,
                           state=row.state, published_path=path, request=row.request)


@project_router.post("/media/feed/poster")
async def media_feed_poster(request: MediaFeedPosterRequest):
    """One small JPEG of a finished clip of this scope's project. The clip is
    addressed by its ledger id; a clip of another project answers like an
    unknown one, and its path never leaves this function."""
    suffix = request.owner_scope[len("film_owner:"):]
    project_id = None if suffix == "scratch" else suffix
    try:
        record = _shoot_take(request.studio_job_id, project_id)
    except Exception:  # noqa: BLE001 - unknown or unreadable: not found
        return _media_feed_error(404, "media_not_found", "That clip is not in this project.")
    if getattr(record, "project_id", None) != project_id:
        return _media_feed_error(404, "media_not_found", "That clip is not in this project.")
    if record.state != "ready" or not record.published_path:
        return _media_feed_error(409, "poster_not_ready", "The clip has no finished file yet.")
    try:
        jpeg = await asyncio.to_thread(_video_poster_jpeg, record.published_path, 320)
    except Exception:  # noqa: BLE001 - finite plugin boundary
        return _media_feed_error(409, "poster_unavailable", "No frame could be read from this clip.")
    return {"schema": _MEDIA_FEED_SCHEMA, "mime_type": "image/jpeg", "data_base64": base64.b64encode(jpeg).decode("ascii")}



# --- Media marks: rating, reject, clip keep, tags ---------------------------
#
# The artist's marks on results, per owner scope: a project keeps them in its
# own state folder (state/media-marks.json, expected_revision like the music
# timing), No project in the profile's private plugin data. A picture's Keep
# stays the Image keep route; a clip's Keep lives here. Nothing is deleted.

_MEDIA_MARKS_SCHEMA = "studio.media-marks/1"
# A sound file of the approved folders is marked by its content id file:<sha256> (lane fs-msnd), the
# id /media/list and /projects/timeline/audio/add give it: the same store, the same keys.
_MEDIA_MARK_ITEM = r"^(imgjob_[0-9a-f]{32}|file:[0-9a-f]{64}|" + _take_ids_module.STUDIO_JOB_ID_BODY + ")$"
_MEDIA_MARKS_MAX = 5000
_MEDIA_MARKS_LOCK = __import__("threading").RLock()


class MediaMarkChange(_ProjectRequest):
    item_id: str = Field(pattern=_MEDIA_MARK_ITEM, min_length=39, max_length=69)
    rating: int | None = Field(default=None, ge=0, le=5)
    rejected: bool | None = None
    kept: bool | None = None
    tags: list[str] | None = Field(default=None, max_length=20)

    @field_validator("tags")
    @classmethod
    def _tag_text(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and any(not 1 <= len(tag.strip()) <= 40 for tag in value):
            raise ValueError("a tag is 1 to 40 characters")
        return value


class MediaMarksReadRequest(_ProjectRequest):
    schema_: Literal["studio.media-marks/1"] = Field(alias="schema")
    owner_scope: str = Field(pattern=_OWNER_SCOPE_PATTERN, min_length=18, max_length=47)


class MediaMarksWriteRequest(MediaMarksReadRequest):
    """Several items in one call (a whole cut marks its takes at once). Each
    change sets only the fields it names; the others stay as they were."""
    expected_revision: int = Field(ge=0)
    changes: list[MediaMarkChange] = Field(min_length=1, max_length=500)


def _media_marks_project(owner_scope: str) -> str | None:
    suffix = owner_scope[len("film_owner:"):]
    return None if suffix == "scratch" else suffix


def _media_marks_scratch_file() -> Path:
    home = _active_hermes_home()
    if home is None:
        raise studio_core.ProjectError("project_unavailable", "no profile home")
    folder = Path(state_root(Path(home) / "plugin-data", "media-marks"))
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "no-project.json"


def _media_marks_read(owner_scope: str) -> dict[str, Any]:
    project_id = _media_marks_project(owner_scope)
    if project_id is not None:
        directory = _recent_directory(_recent_project_store_factory(), project_id)
        return _project_store_factory().read_media_marks(directory, project_id=project_id)
    with _MEDIA_MARKS_LOCK:
        target = _media_marks_scratch_file()
        if not target.exists():
            return {"exists": False, "revision": 0, "updated_at": None, "items": {}}
        value = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema") != _MEDIA_MARKS_SCHEMA or not isinstance(value.get("items"), dict) or type(value.get("revision")) is not int:
            raise studio_core.ProjectError("project_unavailable", "media marks are damaged")
        return {"exists": True, "revision": value["revision"], "updated_at": value.get("updated_at"), "items": value["items"]}


def _media_mark_apply(items: dict[str, Any], change: MediaMarkChange, now: float) -> None:
    mark = dict(items.get(change.item_id) or {})
    if change.rating is not None:
        mark["rating"] = change.rating
    if change.rejected is not None:
        mark["rejected"] = change.rejected
    if change.kept is not None:
        mark["kept"] = change.kept
        mark["kept_at"] = now if change.kept else None
    if change.tags is not None:
        seen: list[str] = []
        for tag in change.tags:
            tag = " ".join(tag.split())
            if tag and tag.casefold() not in {t.casefold() for t in seen}:
                seen.append(tag)
        mark["tags"] = seen
    clean = {k: v for k, v in mark.items() if v not in (None, False, 0, [])}
    if clean:
        items[change.item_id] = clean
    else:
        items.pop(change.item_id, None)


def _media_marks_write(request: MediaMarksWriteRequest) -> dict[str, Any]:
    for change in request.changes:
        if change.kept is not None and change.item_id.startswith("imgjob_"):
            raise ValueError("picture-keep-is-image")
    with _MEDIA_MARKS_LOCK:
        current = _media_marks_read(request.owner_scope)
        if current["revision"] != request.expected_revision:
            raise studio_core.ProjectError("project_conflict", "media marks changed")
        items = json.loads(json.dumps(current["items"]))
        now = time.time()
        for change in request.changes:
            _media_mark_apply(items, change, now)
        if len(items) > _MEDIA_MARKS_MAX:
            raise ValueError("too-many-marks")
        project_id = _media_marks_project(request.owner_scope)
        if project_id is not None:
            directory = _recent_directory(_recent_project_store_factory(), project_id)
            return _project_store_factory().write_media_marks(directory, project_id=project_id, expected_revision=request.expected_revision, items=items)
        target = _media_marks_scratch_file()
        document = {"schema": _MEDIA_MARKS_SCHEMA, "revision": request.expected_revision + 1, "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "items": items}
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(document, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return {"exists": True, "revision": document["revision"], "updated_at": document["updated_at"], "items": items}


def _media_marks_failure(exc: BaseException) -> JSONResponse:
    code = getattr(exc, "code", None)
    if code in ("project_conflict",):
        return _media_feed_error(409, "media_marks_conflict", "The marks changed meanwhile.")
    if isinstance(exc, ValueError) and str(exc) == "picture-keep-is-image":
        return _media_feed_error(422, "picture_keep_is_image", "A picture is kept through Image (continuo_image_keep), not through marks.")
    if isinstance(exc, ValueError) and str(exc) == "too-many-marks":
        return _media_feed_error(422, "media_marks_full", "This project holds too many marks.")
    if code == "project_not_found":
        return _media_feed_error(404, "project_not_found", "Project not found")
    return _media_feed_error(503, "media_marks_unavailable", "The marks could not be read or saved.")


@project_router.post("/media/marks/read")
async def media_marks_read(request: MediaMarksReadRequest):
    """The artist's marks for this scope: {item_id: {rating, rejected, kept, tags}} and the revision to write against."""
    try:
        result = await asyncio.to_thread(_media_marks_read, request.owner_scope)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _media_marks_failure(exc)
    return {"schema": _MEDIA_MARKS_SCHEMA, "owner_scope": request.owner_scope, **result}


@project_router.post("/media/marks/write")
async def media_marks_write(request: MediaMarksWriteRequest):
    """Set rating / rejected / clip kept / tags of one or many items with expected_revision."""
    try:
        result = await asyncio.to_thread(_media_marks_write, request)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _media_marks_failure(exc)
    try:
        film_events.media_changed(request.owner_scope)
    except Exception:  # noqa: BLE001 - an event never fails a route
        pass
    return {"schema": _MEDIA_MARKS_SCHEMA, "owner_scope": request.owner_scope, **result}
