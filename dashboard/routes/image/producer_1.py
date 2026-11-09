from __future__ import annotations

import copy


@project_router.post("/projects/film-image/inspect")
async def film_image_inspect(request: FilmImageInspectRequest) -> dict[str, Any]:
    """Inspect one Film binding through a trusted cached-project handoff.

    This endpoint deliberately does not call ``_image_film_root``: that helper
    is the cold-start/bootstrap path.  The read route is valid only after the
    application has initialized the trusted root and opened the requested
    project.  An unavailable or unopened project fails closed instead of
    constructing project stores or creating filesystem state as a side effect
    of inspection.
    """
    try:
        async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
            root = _initialized_image_film_root()
            handoff = root.read_cached_project(request.project_id)
            projection = handoff.inspect(
                request.binding_id,
                request.film_reference_id,
            )
            contract = importlib.import_module(
                f"{_STUDIO_CORE_PACKAGE_NAME}.image_core.film_binding_contract"
            )
            return contract.validate_image_binding_inspect_response(
                projection,
                request.project_id,
            )
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_film_error_response(exc)


def _image_film_caller_scope_id(http_request: Request) -> str:
    """Resolve the verified caller without accepting identity from the body."""
    try:
        scope = resolve_trusted_image_film_caller_scope(http_request)
        scope_id = scope.scope_id
    except _ImageFilmCallerScopeError:
        raise
    except Exception as exc:
        raise _ImageFilmCallerScopeError("caller_identity_invalid") from exc
    if type(scope_id) is not str or _IMAGE_CALLER_SCOPE_RE.fullmatch(scope_id) is None:
        raise _ImageFilmCallerScopeError("caller_identity_invalid")
    return scope_id


@project_router.post("/projects/image-producer/catalog")
async def image_producer_catalog(request: ImageBuilderCatalogRequest) -> dict[str, Any]:
    """Return the bundled, read-only tile catalog with small inline thumbnails."""
    catalog_dir = Path(__file__).resolve().parent / "image_builder"
    catalog_path = catalog_dir / "krea2-builder-catalog.v1.json"
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        for option in payload.get("options", []):
            relative = option.pop("thumb", None)
            if not relative:
                option["thumb_data_url"] = None
                continue
            tile_path = (catalog_dir / relative).resolve()
            if catalog_dir.resolve() not in tile_path.parents or tile_path.suffix.lower() != ".jpg":
                raise ValueError("catalog tile path is invalid")
            encoded = base64.b64encode(tile_path.read_bytes()).decode("ascii")
            option["thumb_data_url"] = "data:image/jpeg;base64," + encoded
    except (OSError, ValueError, json.JSONDecodeError):
        raise HTTPException(status_code=503, detail="Image builder catalog unavailable")
    contract = _image_contract_module()
    from importlib import import_module
    spec_module = import_module(contract.__package__ + ".krea2_render_spec")
    payload["sampling"] = {"samplers": list(spec_module.SAMPLERS), "schedulers": list(spec_module.SCHEDULERS), "batch_max": 1}
    try:
        payload["render_spec_v2"] = await asyncio.to_thread(_render_spec_v2_catalog, spec_module)
    except Exception:  # noqa: BLE001 - v1 catalog stays usable without v2
        payload["render_spec_v2"] = None
    return payload


def _render_spec_v2_catalog(spec_module: Any) -> dict[str, Any]:
    """Every v2 limit, path-free, so the UI never hard-codes one.

    Blocking (stats model files): run off the event loop.
    """
    files = {}
    for role, items in spec_module.v2_file_choices().items():
        files[role] = [
            {
                "name": item["name"],
                "variant": item["variant"],
                "installed": spec_module.custom_file_folder(role, item["name"]) is not None,  # w24-models: ComfyUI's view
            }
            for item in items
        ]
    profiles = {}
    for profile in ("turbo", "raw"):
        default = spec_module.default_spec_v2(0, 1024, 1024, profile)
        profiles[profile] = {
            "values": {key: default[key] for key in ("steps", "cfg", "sampler", "scheduler")},
            "files": dict(default["files"]),
            "negative_allowed": profile != "turbo",
        }
    return {
        "schema": spec_module.SCHEMA_V2,
        "files": files,
        "profiles": profiles,
        "batch": {"min": spec_module.V2_BATCH_MIN, "max": spec_module.V2_BATCH_MAX},
        "size": {"min_px": spec_module.V2_SIZE_MIN, "max_px": spec_module.V2_SIZE_MAX,
                 "step": spec_module.V2_SIZE_STEP, "max_pixels": spec_module.V2_MAX_PIXELS},
        "aspect_presets": spec_module.aspect_preset_sizes(1.0),
        "steps": {"min": 1, "max": 100},
        "cfg": {"min": 0.1, "max": 20},
        "lora_max": spec_module.LORA_MAX,
        "lora_strength": {"min": 0, "max": 2},
        "samplers": list(spec_module.SAMPLERS),
        "schedulers": list(spec_module.SCHEDULERS),
    }


def _job_render_spec(job: Mapping[str, Any]) -> dict[str, Any] | None:
    snapshot = job.get("request_snapshot") or {}
    if snapshot.get("operation") in {"workflow_create", "workflow_edit"}:
        artist = snapshot["artist_input"]
        return {"schema": "studio.image-workflow-settings/1", "target_id": snapshot["target_id"],
                "controls": copy.deepcopy(snapshot["effective_input"]["workflow"]["controls"]),
                "width": artist["width"], "height": artist["height"]}
    if snapshot.get("operation") in {"edit", "compose", "upscale"}:
        # Edit variants, composed and upscaled pictures carry no Krea render spec.
        return None
    effective = snapshot.get("effective_input") or {}
    spec = effective.get("render_spec")
    if isinstance(spec, dict):
        return spec
    artist = snapshot.get("artist_input") or {}
    if all(type(artist.get(k)) is int for k in ("seed", "width", "height")):
        return _image_contract_module().default_render_spec(artist["seed"], artist["width"], artist["height"])
    return None


@project_router.post("/projects/image-producer/keep")
async def image_producer_keep(request: ImageProducerKeepRequest) -> dict[str, Any]:
    """Turn a finished run into a named saved revision.

    The origin is read from the ledger, never from the caller: the request
    snapshot the job ran under already names the context and work, so a caller
    cannot redirect a keep at someone else's work by naming it back at us.
    """
    try:
        async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
            root = _image_producer_root(request.owner_scope)
            handoff = _image_operations_handoff(root, request.owner_scope)
            jobs = handoff.producer_jobs()
            states = handoff.image_states()
        try:
            job = jobs.get_job(request.image_producer_job_id)
        except Exception as exc:  # noqa: BLE001 - finite ledger boundary
            # Same rule as the status route: only an unknown id is a 404.
            if type(exc).__name__ != "ImageJobNotFoundError" and not isinstance(exc, KeyError):
                raise
            return JSONResponse(
                status_code=404,
                content={"error": {"code": "job_not_found", "message": "Job not found"}},
            )
        snapshot = job.get("request_snapshot") or {}
        basis = snapshot.get("context_basis") or {}
        context_id = basis.get("context_id")
        work_id = basis.get("work_id")
        # A job of another scope answers like an unknown one (as status does).
        try:
            states.require_owned(context_id)
        except Exception:  # noqa: BLE001 - not owned, or no context at all
            return JSONResponse(
                status_code=404,
                content={"error": {"code": "job_not_found", "message": "Job not found"}},
            )
        if job.get("state") not in {"ready", "published"}:
            return _image_film_error_response(
                studio_core.ProjectError("invalid_project", "Only a finished run can be kept")
            )
        if job.get("state") != "published":
            # Keeping means keeping the picture; until it is home there is
            # nothing to bind, and a name alone is not a kept image.
            return {"error": {"code": "picture_not_ready", "message": "The picture is not home yet"}}
        if snapshot.get("operation") == "edit":
            # An edit variant's published picture is the edited crop, not the
            # whole picture; accepting an edit as a version needs the stitched
            # layer (desktop edit-layer package), so it is refused honestly.
            return {"error": {"code": "edit_keep_unavailable", "message": "Accepting an edit as a version is not available yet"}}
        if not context_id or not work_id:
            return _image_film_error_response(
                studio_core.ProjectError("invalid_project", "This run has no context to keep")
            )
        artifact = _image_contract_module().build_generated_artifact_ref(job)
        intent = str(((snapshot.get("artist_input") or {}).get("prompt")) or "")
        kept = states.keep_generated(context_id, basis, request.name, artifact, intent)
        file_name = await _image_keep_file(handoff, request)  # lane p6-name: <output>/<rule name>.png
        # Lane w6-link: a run made for a Script shot is that shot's picture once kept (its recipe's shot_id).
        shot_id = _image_job_shot(request.image_producer_job_id)
        if shot_id:
            await asyncio.to_thread(_image_keep_assign_shot, request.owner_scope, request.image_producer_job_id, shot_id)
        film_events.jobs_changed([request.image_producer_job_id], "kept", request.owner_scope)
        film_events.media_changed(request.owner_scope)
        return {
            **({"shot_id": shot_id} if shot_id else {}),
            "schema": "studio.image-producer-operation/1",
            "kind": "image_producer_keep_result",
            "owner_scope": request.owner_scope,
            "context_id": kept["context_id"],
            "work_id": kept["work_id"],
            "revision_id": kept["revision_id"],
            "name": kept["name"],
            "render_spec": _job_render_spec(job),
            "file_name": file_name,
        }
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_film_error_response(exc)


def _image_revision_summary(revision: dict[str, Any], jobs: Any) -> dict[str, Any]:
    """What the pane may know about one kept revision - named field by field.

    The picture is referenced by the run that made it, never by a file: the
    pane asks the picture route for that run, which checks ownership again.
    Prompt and seed come from the ledger's request snapshot, so what is shown
    under a picture is what actually produced it.
    """
    summary: dict[str, Any] = {
        "revision_id": revision["revision_id"],
        "parent_revision_id": revision.get("parent_revision_id"),
        "saved_at": revision.get("saved_at"),
    }
    content = revision.get("content") or {}
    selected = revision.get("selected_artifact")
    generated = [
        item for item in (content.get("alternatives") or ())
        if isinstance(item, dict) and item.get("origin") == "generated"
        and item.get("material_id") == selected
    ]
    if selected is None or len(generated) != 1:
        return summary
    artifact = generated[0]
    summary["image_producer_job_id"] = artifact.get("producer_job_id")
    summary["width"] = artifact.get("width")
    summary["height"] = artifact.get("height")
    summary["prompt"] = content.get("intent") or ""
    try:
        recorded = jobs.get_job(artifact["producer_job_id"])
        artist = (recorded.get("request_snapshot") or {}).get("artist_input") or {}
        summary["render_spec"] = _job_render_spec(recorded)
        if recorded["request_snapshot"].get("operation") in {"workflow_create", "workflow_edit"}:
            summary["sent_prompt"] = recorded.get("sent_prompt")
        if type(artist.get("seed")) is int:
            summary["seed"] = artist["seed"]
    except Exception:  # noqa: BLE001 - seed is informative, never required
        pass
    return summary


@project_router.post("/projects/image-producer/contexts")
async def image_producer_contexts(request: ImageProducerContextsRequest) -> dict[str, Any]:
    """List the saved revisions a generation can build on.

    The catalog is rebuilt field by field rather than forwarded: the stored
    state carries owner tokens and publication paths, and neither is catalog
    data.  A future field added to the ledger therefore cannot leak here by
    default - it has to be named explicitly.
    """
    try:
        async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
            root = _image_producer_root(request.owner_scope)
            handoff = _image_operations_handoff(root, request.owner_scope)
            states = handoff.image_states()
            jobs = handoff.producer_jobs()
        contexts: list[dict[str, Any]] = []
        for context in states.enumerate_contexts():
            works: list[dict[str, Any]] = []
            for work in (context.get("state") or {}).get("works") or ():
                revisions = [
                    _image_revision_summary(revision, jobs)
                    for revision in (work.get("saved_revisions") or ())
                ]
                works.append({
                    "work_id": work["work_id"],
                    "title": work.get("name"),
                    "saved_revisions": revisions,
                })
            contexts.append({"context_id": context["context_id"], "works": works})
        return {
            "schema": "studio.image-producer-operation/1",
            "kind": "image_producer_contexts_result",
            "owner_scope": request.owner_scope,
            "contexts": contexts,
        }
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_film_error_response(exc)


@project_router.post("/projects/image-producer/submit")
async def image_producer_submit(request: ImageProducerSubmitRequest) -> dict[str, Any]:
    """Start one generation on the project the application already opened.

    The route does not build a provider client or adapter: it asks the
    composition root for this project's coordinator, so provider identity and
    network authority stay where they were decided.  The snapshot is passed
    through untouched - the ledger validates its own envelope and must remain
    the only authority on what a valid request is.
    """
    try:
        async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
            root = _image_producer_root(request.owner_scope)
            handoff = _image_operations_handoff(root, request.owner_scope)
            coordinator = handoff.producer_coordinator()
            states = handoff.image_states()
        # The request contract is assembled here, never sent by the caller:
        # capability and graph hashes, the effective model parameters and the
        # owner token are backend facts.  A caller sends intent - prompt, seed,
        # size - plus which saved revision it is working from, and cannot reach
        # past that into what the ledger accepts.
        if request.revision_id is None:
            # Nothing named yet: the image area is for working, not for filling
            # in a form first.  The store opens a working context on first use
            # and the run builds on its draft; naming happens later, and only
            # if the result turned out worth keeping.
            working = states.ensure_working_context(request.operation_id)
            context_id = working["context_id"]
            work_id = working["work_id"]
            basis_kind = "draft"
            revision_id = None
        else:
            context_id = request.context_id
            work_id = request.work_id
            basis_kind = "saved_revision"
            revision_id = request.revision_id
        claim = states.load_current_owner(context_id, work_id, basis_kind, revision_id)
        basis = {
            key: claim[key]
            for key in (
                "context_id", "work_id", "basis_kind",
                "revision_id", "draft_generation", "basis_content_sha256",
            )
        }
        attach_policy: dict[str, Any] = {
            "mode": "specified" if revision_id is not None else "current",
            "owner_token": claim["owner_token"],
            "expected_revision": claim["expected_revision"],
        }
        if revision_id is not None:
            attach_policy["context_id"] = context_id
            attach_policy["work_id"] = work_id
            attach_policy["revision_id"] = revision_id
        contract = _image_contract_module()
        requested_spec = request.render_spec or contract.default_render_spec(request.seed, request.width, request.height)
        refused = await _image_model_missing(requested_spec) or await _image_model_files_admit(requested_spec) or await _image_lora_admit(requested_spec)
        if refused is not None:
            return refused
        if requested_spec.get("schema") == "studio.krea2-render-spec/2":
            # Render spec v2: a run of N pictures is N ledger jobs, one picture
            # each, on the unchanged single-image publication seam.  Member 0
            # carries the caller's operation id (so a 1-picture run is one job
            # exactly like before); members 1..N-1 derive theirs from it, so a
            # replayed submit replays the whole run and never renders twice.
            from importlib import import_module
            spec_module = import_module(contract.__package__ + ".krea2_render_spec")
            members = []
            for index in range(requested_spec["batch_size"]):
                member_spec = spec_module.run_member_spec(requested_spec, index)
                key = request.operation_id if index == 0 else f"{request.operation_id}:run-{index}"
                member_snapshot = contract.build_snapshot(
                    request.prompt, member_spec["seed"], request.width, request.height, basis, attach_policy,
                    render_spec=member_spec,
                )
                member_job = await coordinator.submit(key, member_snapshot, client_request_id=key)
                _image_preview_watch(member_job)
                members.append((member_spec, member_job))
            job = members[0][1]
            _announce_image_jobs(
                [(member_job["image_producer_job_id"], None, member_job.get("state")) for _spec, member_job in members],
                request.owner_scope,
            )
        else:
            members = None
            snapshot = contract.build_snapshot(
                request.prompt, request.seed, request.width, request.height, basis, attach_policy,
                render_spec=requested_spec,
            )
            job = await coordinator.submit(
                request.operation_id,
                snapshot,
                client_request_id=request.operation_id,
            )
            _image_preview_watch(job)
            _announce_image_jobs([(job["image_producer_job_id"], None, job.get("state"))], request.owner_scope)
        result_names = _image_result_names_reserve(request.owner_scope, [m[1]["image_producer_job_id"] for m in members] if members is not None else [job["image_producer_job_id"]])  # lane p6-name
        provider = job.get("provider") or {}
        _image_job_recipe_write([m[1]["image_producer_job_id"] for m in members] if members is not None else [job["image_producer_job_id"]],
                                target_id=provider.get("target_id"), prompt=request.prompt, shot_id=request.shot_id)
        result: dict[str, Any] = {
            "schema": "studio.image-producer-operation/1",
            "kind": "image_producer_submit_result",
            "owner_scope": request.owner_scope,
            "operation_id": request.operation_id,
            "image_producer_job_id": job["image_producer_job_id"],
            "state": job["state"],
            "target_id": provider.get("target_id"),
            "provider_id": provider.get("provider_id"),
            "render_spec": _job_render_spec(job),
            "result_name": result_names.get(job["image_producer_job_id"]),
        }
        if (job.get("request_snapshot") or {}).get("operation") in {"workflow_create", "workflow_edit"}:
            result["sent_prompt"] = job.get("sent_prompt")
            result["request_contract"] = "studio.image-workflow-request/1"
        for optional in ("failure_reason", "quarantine_reason"):
            if job.get(optional) is not None:
                result[optional] = job[optional]
        result.update({"shot_id": _image_job_shot(job["image_producer_job_id"])} if _image_job_shot(job["image_producer_job_id"]) else {})
        if members is not None:
            result["run"] = {
                "batch_size": len(members),
                "jobs": [
                    {
                        "image_producer_job_id": member_job["image_producer_job_id"],
                        "batch_index": member_spec["batch_index"],
                        "seed": member_spec["seed"],
                        "state": member_job["state"],
                        **({"failure_reason": member_job["failure_reason"]} if member_job.get("failure_reason") else {}),
                        **({"quarantine_reason": member_job["quarantine_reason"]} if member_job.get("quarantine_reason") else {}),
                    }
                    for member_spec, member_job in members
                ],
            }
        return result
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_film_error_response(exc)


class ImageProducerPictureRequest(_ProjectRequest):
    schema_: str = Field(alias="schema", pattern=r"^studio\.image-producer-operation/1$")
    kind: str = Field(pattern=r"^image_producer_picture$")
    owner_scope: str = Field(pattern=_OWNER_SCOPE_PATTERN)
    image_producer_job_id: str = Field(pattern=r"^imgjob_[0-9a-f]{32}$")


@project_router.post("/projects/image-producer/picture")
async def image_producer_picture(request: ImageProducerPictureRequest) -> Any:
    """Hand a finished run's picture to the scope that owns it.

    The renderer's only door is JSON, so the PNG travels base64-encoded. The
    bytes are read through the material reader, which derives the file from
    the published job and re-verifies size and hash - a caller names a job,
    never a file.  A job of another scope answers like an unknown one.
    """
    unknown = JSONResponse(
        status_code=404,
        content={"error": {"code": "job_unknown", "message": "Generation job is unknown"}},
    )
    try:
        async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
            # Read-only: a recent project's picture shows on the Start screen before it is opened.
            root = _image_feed_root(request.owner_scope)
            handoff = _image_feed_handoff(root, request.owner_scope)
            jobs = handoff.producer_jobs()
            states = handoff.image_states()
            _collector, reader = handoff.result_services()
        try:
            job = jobs.get_job(request.image_producer_job_id)
        except Exception as exc:  # noqa: BLE001 - narrowed below
            if type(exc).__name__ != "ImageJobNotFoundError" and not isinstance(exc, KeyError):
                raise
            return unknown
        basis = (job.get("request_snapshot") or {}).get("context_basis") or {}
        try:
            states.require_owned(basis.get("context_id"))
        except Exception:  # noqa: BLE001 - not owned, or no context at all
            return unknown
        if job.get("state") != "published":
            return {"error": {"code": "picture_not_ready", "message": "The picture is not ready yet"}}
        data, artifact = await asyncio.to_thread(reader.read_published_job, request.image_producer_job_id)
        return {
            "schema": "studio.image-producer-operation/1",
            "kind": "image_producer_picture_result",
            "owner_scope": request.owner_scope,
            "image_producer_job_id": request.image_producer_job_id,
            "mime_type": artifact["mime_type"],
            "width": artifact["width"],
            "height": artifact["height"],
            "sha256": artifact["artifact_sha256"],
            "data_base64": base64.b64encode(data).decode("ascii"),
        }
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_film_error_response(exc)


class ImageProducerStageOpeningFrameRequest(_ProjectRequest):
    schema_: str = Field(alias="schema", pattern=r"^studio\.image-producer-operation/1$")
    kind: str = Field(pattern=r"^image_producer_stage_opening_frame$")
    owner_scope: str = Field(pattern=_OWNER_SCOPE_PATTERN)
    image_producer_job_id: str = Field(pattern=r"^imgjob_[0-9a-f]{32}$")


_OPENING_FRAME_SUFFIX = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}


class _OpeningFrameStageError(Exception):
    """The picture could not be staged as it was published."""


def _opening_frame_dir() -> Path:
    directory = prepared_cache_dir() / "opening-frames"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _sha256_regular_file(path: Path) -> str | None:
    """Hash a regular, non-link file; ``None`` when it is anything else."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        raise _OpeningFrameStageError("The staged name is taken by something that is not a picture file.")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stage_opening_frame_bytes(data: bytes, sha256: str, mime_type: str) -> Path:
    """Write one published picture into the prepared cache, write-once.

    The name is the content hash, so a second call (or another run with the
    same picture) finds the same file and never rewrites it.  A file already
    under that name is only used when its bytes hash to the name; anything
    else is refused and left untouched.  New bytes go to a private temp file
    in the same folder and are linked into place without replacing.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", sha256 or ""):
        raise _OpeningFrameStageError("The picture has no usable fingerprint.")
    if hashlib.sha256(data).hexdigest() != sha256:
        raise _OpeningFrameStageError("The picture does not match its published fingerprint.")
    suffix = _OPENING_FRAME_SUFFIX.get(mime_type)
    if suffix is None:
        raise _OpeningFrameStageError("Only PNG, JPEG or WebP pictures can open a clip.")
    folder = _opening_frame_dir()
    final = folder / f"{sha256}{suffix}"
    existing = _sha256_regular_file(final)
    if existing is not None:
        if existing != sha256:
            raise _OpeningFrameStageError("A different file already uses this picture's name.")
        return final
    fd, temp_name = tempfile.mkstemp(prefix=".stage-", suffix=suffix, dir=str(folder))
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp, final)  # never replaces an existing name
        except FileExistsError:
            pass  # a concurrent call won; its file is verified below
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
    if _sha256_regular_file(final) != sha256:
        raise _OpeningFrameStageError("The staged picture did not verify.")
    return final


@project_router.post("/projects/image-producer/stage-opening-frame")
async def image_producer_stage_opening_frame(request: ImageProducerStageOpeningFrameRequest) -> Any:
    """Make a finished picture usable as a Shoot opening frame.

    Shoot takes a local file path, confined to approved roots plus the
    prepared cache.  The ownership check is the one ``/picture`` uses (a job
    of another scope answers like an unknown one); the bytes come from the
    same hash-verified reader, and are staged write-once under their content
    hash in ``prepared/opening-frames``.  A caller names a job, never a file.
    """
    unknown = JSONResponse(
        status_code=404,
        content={"error": {"code": "job_unknown", "message": "Generation job is unknown"}},
    )
    try:
        async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
            root = _image_producer_root(request.owner_scope)
            handoff = _image_operations_handoff(root, request.owner_scope)
            jobs = handoff.producer_jobs()
            states = handoff.image_states()
            _collector, reader = handoff.result_services()
        try:
            job = jobs.get_job(request.image_producer_job_id)
        except Exception as exc:  # noqa: BLE001 - narrowed below
            if type(exc).__name__ != "ImageJobNotFoundError" and not isinstance(exc, KeyError):
                raise
            return unknown
        basis = (job.get("request_snapshot") or {}).get("context_basis") or {}
        try:
            states.require_owned(basis.get("context_id"))
        except Exception:  # noqa: BLE001 - not owned, or no context at all
            return unknown
        if job.get("state") != "published":
            return {"error": {"code": "picture_not_ready", "message": "The picture is not ready yet"}}
        data, artifact = await asyncio.to_thread(reader.read_published_job, request.image_producer_job_id)
        try:
            staged = await asyncio.to_thread(
                _stage_opening_frame_bytes, data, artifact["artifact_sha256"], artifact["mime_type"]
            )
        except _OpeningFrameStageError as exc:
            return {"error": {"code": "opening_frame_unavailable", "message": str(exc)}}
        return {
            "schema": "studio.image-producer-operation/1",
            "kind": "image_producer_opening_frame_result",
            "owner_scope": request.owner_scope,
            "image_producer_job_id": request.image_producer_job_id,
            "path": str(staged),
            "sha256": artifact["artifact_sha256"],
            "width": artifact["width"],
            "height": artifact["height"],
        }
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_film_error_response(exc)


@project_router.post("/projects/image-producer/cancel")
async def image_producer_cancel(request: ImageProducerCancelRequest) -> Any:
    """Stop one owned job: the coordinator cancels the store-owned prompt.

    Another scope's job answers like an unknown one. A finished job is
    returned unchanged; an unreachable ComfyUI leaves the job as it was.
    """
    unknown = JSONResponse(
        status_code=404,
        content={"error": {"code": "job_unknown", "message": "Generation job is unknown"}},
    )
    try:
        async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
            root = _image_producer_root(request.owner_scope)
            handoff = _image_operations_handoff(root, request.owner_scope)
            jobs = handoff.producer_jobs()
            states = handoff.image_states()
        try:
            job = jobs.get_job(request.image_producer_job_id)
        except Exception as exc:  # noqa: BLE001 - narrowed below
            if type(exc).__name__ != "ImageJobNotFoundError" and not isinstance(exc, KeyError):
                raise
            return unknown
        basis = (job.get("request_snapshot") or {}).get("context_basis") or {}
        try:
            states.require_owned(basis.get("context_id"))
        except Exception:  # noqa: BLE001 - not owned, or no context at all
            return unknown
        if job.get("state") not in _IMAGE_TERMINAL_STATES:
            async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
                coordinator = handoff.producer_coordinator()
            before = job.get("state")
            cancelled = await coordinator.cancel(request.image_producer_job_id)
            job = cancelled if isinstance(cancelled, dict) and cancelled.get("state") else job
            _announce_image_jobs([(job["image_producer_job_id"], before, job.get("state"))], request.owner_scope)
        await _image_run_members_cancel(job, jobs, handoff, request.owner_scope)  # w20-img IMG-046: Stop stops the run
        result: dict[str, Any] = {
            "schema": "studio.image-producer-operation/1",
            "kind": "image_producer_cancel_result",
            "owner_scope": request.owner_scope,
            "image_producer_job_id": job["image_producer_job_id"],
            "state": job["state"],
        }
        for optional in ("failure_reason", "quarantine_reason"):
            if job.get(optional) is not None:
                result[optional] = job[optional]
        return result
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_film_error_response(exc)


@project_router.post("/projects/image-producer/status")
async def image_producer_status(request: ImageProducerStatusRequest) -> dict[str, Any]:
    """Report one generation job from the project-owned ledger.

    Like the Film read route this never calls ``_image_film_root``: reading a
    job is valid only after the application opened the project.  The response
    is assembled field by field from the ledger envelope instead of forwarding
    it, so local storage detail (publication path, adapter internals) cannot
    reach a caller even if the envelope grows new fields later.
    """
    try:
        async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
            root = _image_producer_root(request.owner_scope)
            handoff = _image_operations_handoff(root, request.owner_scope)
            jobs = handoff.producer_jobs()
            try:
                job = jobs.get_job(request.image_producer_job_id)
            except Exception as exc:  # noqa: BLE001 - narrowed below
                # The ledger signals an unknown id with its own error type; a
                # KeyError only appears in tests with a mapping-like store.
                # Anything else is a real failure and must not become a 404.
                if type(exc).__name__ != "ImageJobNotFoundError" and not isinstance(exc, KeyError):
                    raise
                return JSONResponse(
                    status_code=404,
                    content={"error": {"code": "job_unknown", "message": "Generation job is unknown"}},
                )
            coordinator = handoff.producer_coordinator()
            states = handoff.image_states()
            collector, _reader = handoff.result_services()
        # The job ledger is profile-wide; a job belongs to whoever owns the
        # context it ran on.  Another scope gets the same answer as for an
        # unknown id, so a job id cannot be used to probe other projects.
        basis = (job.get("request_snapshot") or {}).get("context_basis") or {}
        try:
            states.require_owned(basis.get("context_id"))
        except Exception:  # noqa: BLE001 - not owned, or no context at all
            return JSONResponse(
                status_code=404,
                content={"error": {"code": "job_unknown", "message": "Generation job is unknown"}},
            )
        # The ledger never learns on its own that a provider finished: the
        # coordinator is what reconciles a run into a terminal state. Reading
        # alone would leave a caller polling a job that finished minutes ago.
        # A job that is already terminal is answered from the ledger, so a
        # finished run costs no further provider round trip.
        polled = False
        state_before = job.get("state")
        if job.get("state") not in _IMAGE_TERMINAL_STATES:
            polled = True
            reconciled = await coordinator.status(request.image_producer_job_id)
            if isinstance(reconciled, dict) and reconciled.get("state"):
                job = reconciled
        picture_issue: str | None = None
        if job.get("state") == "ready":
            # Finished in ComfyUI is not finished for the user: the picture
            # still sits in ComfyUI's temp output. Bring it home once; after
            # that the job is published and this branch is never taken again.
            # A ready job read straight from the ledger carries no descriptor,
            # so ask the coordinator once (cached, or a re-poll after a
            # restart) - never twice in one request.
            if not polled and not isinstance(job.get("output_descriptor"), Mapping):
                reconciled = await coordinator.status(request.image_producer_job_id)
                if isinstance(reconciled, dict) and reconciled.get("state"):
                    job = reconciled
            try:
                job = await collector.collect(
                    request.image_producer_job_id, job.get("output_descriptor")
                )
            except Exception as exc:  # noqa: BLE001 - reported, never raised
                picture_issue = str(getattr(exc, "code", "") or "collect-failed")
                job = handoff.producer_jobs().get_job(request.image_producer_job_id)
        _announce_image_jobs([(job["image_producer_job_id"], state_before, job.get("state"))], request.owner_scope)
        provider = job.get("provider") or {}
        result: dict[str, Any] = {
            "schema": "studio.image-producer-operation/1",
            "kind": "image_producer_status_result",
            "owner_scope": request.owner_scope,
            "image_producer_job_id": job["image_producer_job_id"],
            "state": job["state"],
            "target_id": provider.get("target_id"),
            "provider_id": provider.get("provider_id"),
            "render_spec": _job_render_spec(job),
        }
        if (job.get("request_snapshot") or {}).get("operation") in {"workflow_create", "workflow_edit"}:
            result["sent_prompt"] = job.get("sent_prompt")
            result["request_contract"] = "studio.image-workflow-request/1"
        for optional in ("failure_reason", "quarantine_reason"):
            if job.get(optional) is not None:
                result[optional] = job[optional]
        result.update({"shot_id": _image_job_shot(job["image_producer_job_id"])} if _image_job_shot(job["image_producer_job_id"]) else {})
        if job["state"] == "published":
            result["picture"] = "available"
            await _image_steps_persist(job)
        elif job["state"] == "ready":
            result["picture"] = "unavailable" if picture_issue else "pending"
        descriptor = job.get("output_descriptor")
        if job["state"] == "ready" and isinstance(descriptor, Mapping):
            # Retrieval identity only; the local path stays inside the plugin.
            result["output"] = {
                key: descriptor[key]
                for key in ("filename", "subfolder", "type")
                if isinstance(descriptor.get(key), str)
            }
        return result
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_film_error_response(exc)
