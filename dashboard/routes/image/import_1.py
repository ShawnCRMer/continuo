from __future__ import annotations


# --- pictures from disk or the clipboard (studio.image-import-request/1) ---------------------------
# The decoding, the snapshot and the ledger drive live in generation/image_import.py.
class ImageImportRequest(_ProjectRequest):
    """A local file inside an approved root, or pasted bytes, becomes an owned picture."""

    schema_: Literal["studio.image-import-request/1"] = Field(alias="schema")
    owner_scope: str = Field(pattern=_OWNER_SCOPE_PATTERN, min_length=18, max_length=47)
    operation_id: str = Field(pattern=r"^imgop_[0-9a-f]{32}$", min_length=38, max_length=38)
    path: str | None = Field(default=None, min_length=1, max_length=4096)
    approved_roots: list[str] = Field(default_factory=list, max_length=64)
    project_ref: ProjectReference | None = None
    # 64 MB of picture bytes, base64 (image_import.MAX_SOURCE_B64).
    data_b64: str | None = Field(default=None, min_length=8, max_length=89_478_488)
    file_name: str | None = Field(default=None, max_length=255)


@project_router.post("/projects/image-producer/import")
async def image_producer_import(request: ImageImportRequest) -> Any:
    """Make a local picture (or pasted bytes) an owned, published picture of the ledger.

    The picture is decoded once (PNG, JPEG, WebP, TIFF at 8 or 16 bit; EXIF orientation applied;
    an embedded colour profile converted to sRGB) and copied write-once into the profile's
    materials under its sha256.  The job is an ordinary producer job with an ``imported``
    provenance and no render spec, so Edit, Keep, compose, Use in Shoot and the Media feed
    take it unchanged.  The same bytes in the same scope give the same job.
    """
    def refuse(code: str, message: str) -> JSONResponse:
        return JSONResponse(status_code=422, content={"error": {"code": code, "message": message}})

    try:
        imported = importlib.import_module(_image_contract_module().__package__ + ".image_import")
        if (request.path is None) == (request.data_b64 is None):
            return refuse("invalid_import", "Needs exactly one of a file path or picture bytes")
        try:
            if request.path is not None:
                if not request.approved_roots:
                    return refuse("path_refused", imported.OUTSIDE_ROOTS)
                try:
                    roots = _request_roots(request.approved_roots, request.project_ref, path=request.path)
                except _FolderNotApproved:
                    return JSONResponse(status_code=403, content=_folder_refusal())
                data, name = await asyncio.to_thread(imported.read_local, request.path, roots,
                                                     _confined_local_path)
            else:
                try:
                    data = base64.b64decode(request.data_b64, validate=True)
                except (ValueError, TypeError):
                    return refuse("not_a_picture", imported.NOT_A_PICTURE)
                if len(data) > imported.MAX_SOURCE_BYTES:
                    return refuse("picture_too_large", imported.TOO_LARGE_FILE)
                name = request.file_name or "Pasted picture"
            png, facts = await asyncio.to_thread(imported.decode, data)
        except imported.ImageImportError as exc:
            return refuse(exc.code, exc.message)
        key = imported.idempotency_key(request.owner_scope, facts["source_sha256"])
        async with imported.LOCKS.setdefault(key, asyncio.Lock()):
            async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
                root = _image_producer_root(request.owner_scope)
                handoff = _image_operations_handoff(root, request.owner_scope)
                jobs = handoff.producer_jobs()
                states = handoff.image_states()
                collector, _reader = handoff.result_services()
            try:
                job = jobs.get_by_idempotency_key(key)
            except Exception as exc:  # noqa: BLE001 - only an unknown key means "not imported yet"
                if type(exc).__name__ != "ImageJobNotFoundError":
                    raise
                job = None
            replayed = job is not None
            if job is None:
                working = states.ensure_working_context(request.operation_id)
                claim = states.load_current_owner(working["context_id"], working["work_id"], "draft", None)
                basis = {k: claim[k] for k in ("context_id", "work_id", "basis_kind", "revision_id",
                                               "draft_generation", "basis_content_sha256")}
                policy = {"mode": "current", "owner_token": claim["owner_token"], "expected_revision": claim["expected_revision"]}
                provenance = {"schema": imported.SCHEMA, "file_name": imported.clean_file_name(name),
                              **{k: facts[k] for k in ("source_sha256", "picture_sha256", "source_format", "bit_depth",
                                                       "orientation", "icc", "alpha")}}
                snapshot = imported.build_snapshot(provenance, facts["width"], facts["height"], basis, policy)
                job, _existing = jobs.accept(key, snapshot, key)
            else:
                # The key carries the scope; the owner check still answers for the context.
                states.require_owned(job["request_snapshot"]["context_basis"]["context_id"])
            job_id = job["image_producer_job_id"]
            before = job.get("state")
            if before != "published":
                job = await asyncio.to_thread(imported.publish, jobs, collector.ingestor, job, png)
                _announce_image_jobs([(job_id, before if replayed else None, job.get("state"))], request.owner_scope)
                film_events.media_changed(request.owner_scope)
        if job.get("state") != "published":
            return JSONResponse(status_code=500, content={"error": {"code": "import_failed",
                                                                    "message": "The picture could not be imported"}})
        snapshot = job["request_snapshot"]
        return {"schema": "studio.image-producer-operation/1", "kind": "image_import_result", "owner_scope": request.owner_scope,
                "operation_id": request.operation_id, "image_producer_job_id": job_id, "state": "published",
                "replayed": replayed, "width": snapshot["artist_input"]["width"], "height": snapshot["artist_input"]["height"],
                "imported": snapshot["effective_input"]["imported"], "render_spec": None}
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_film_error_response(exc)
