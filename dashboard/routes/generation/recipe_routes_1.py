from __future__ import annotations


@router.post("/recipe/stamp")
async def recipe_stamp(request: RecipeStampRequest) -> dict[str, Any]:
    try:
        v4_take = _v4_take_file(request.job_id)
        if v4_take is not None:
            return _stamp_v4_take(request, *v4_take)
        # Lane p13c-safe item 1: never embed into a take at a copied project's old place while its re-base is pending.
        _rebase_pending_refusal(request.project_ref, request.path)
        roots = _request_roots(request.approved_roots, request.project_ref, path=request.path)
        path = _confined_local_path(request.path, roots)
        # For a versioned studio job the provider job ID is the ledger's
        # write-once Comfy prompt ID, not the studio row id. Resolve it here
        # so the stamp embeds the real provider identity. A successful lookup
        # with no row omits the field; an unavailable ledger fails closed.
        provider_job_id = _provider_prompt_id(request.job_id)
        envelope = _recipe_envelope(
            request.recipe,
            job_id=request.job_id,
            shot=request.shot,
            take=request.take,
            parent=request.parent,
            approved_roots=roots,
        )
        if provider_job_id:
            envelope["provider_job_id"] = provider_job_id
        elif request.job_id is not None and not _is_studio_job_id(request.job_id):
            # Legacy FLUX jobs have no ledger row; the studio id (which is
            # also the provider job id there) is the honest value.
            envelope["provider_job_id"] = request.job_id
        else:
            # A studio job whose provider prompt id is unknown must not be
            # mislabeled; drop the field entirely.
            envelope.pop("provider_job_id", None)
        sidecar = _stamp_file(path, envelope)
        return {"stamped": True, "path": str(path), "sidecar": str(sidecar)}
    except (OSError, sqlite3.DatabaseError):
        return _error("Recipe stamping failed: local media or generation ledger is unavailable.")
    except (RecipeError, TypeError, ValueError) as exc:
        if isinstance(exc, _FolderNotApproved):
            return _folder_refusal()
        if isinstance(exc, _RebasePending):
            return {"error": str(exc), "reason_code": exc.reason_code}
        return _error(f"Recipe stamping failed: {exc}")


def _v4_take_file(job_id: str | None) -> tuple[Any, Path] | None:
    """A Contract-4 take's row and the file the backend itself published for it, or None
    for any other job (legacy path). The file comes from the take's private operation
    record and is re-verified by size and sha256; the caller never names the folder, so a
    default-destination take (Downloads) is stamped without its folder being an approved root."""
    if not _is_studio_job_id(job_id):
        return None
    try:
        _comfy_store().get_job(job_id)
        return None  # a Contract-3 row: the legacy path below
    except _comfy_job_store_module.JobNotFoundError:
        return None
    except _comfy_job_store_module.ProjectScopeError:
        pass  # get_job refuses every Contract-4 row; ledger access failures propagate
    row = _read_ledger_row(job_id)
    if getattr(row, "row_contract", None) != 4:
        return None
    if row.state != "ready":
        raise RecipeError("The take has no finished file yet.")
    ops = _private_ops()
    return row, Path(ops.saved_path_for_receipt(ops.load_for_job(row)))


def _stamp_v4_take(request: RecipeStampRequest, row: Any, published: Path) -> dict[str, Any]:
    """Record a V4 take's recipe in its sidecar only. The media bytes stay as published:
    the private record re-verifies the file's sha256 on every receipt, poster and timeline
    read, so an in-place remux would orphan the take. Only the exact file of this job id is
    accepted, and only its sidecar is written: nothing an agent could not already ask the
    backend to write elsewhere."""
    requested = os.path.normcase(os.path.realpath(os.path.expanduser(request.path.strip())))
    if not request.path.strip() or requested != os.path.normcase(os.path.realpath(published)):
        raise RecipeError("The path is not this take's file.")
    # A take published before the project was copied lies at the old place: no sidecar there while the re-base waits.
    _rebase_pending_refusal(request.project_ref, str(published))
    envelope = _recipe_envelope(
        request.recipe, job_id=request.job_id, shot=request.shot, take=request.take,
        parent=request.parent, approved_roots=_request_roots(request.approved_roots, request.project_ref),
    )
    if row.provider_job_id:
        envelope["provider_job_id"] = str(row.provider_job_id)
    else:
        envelope.pop("provider_job_id", None)
    sidecar = _sidecar_path(published)
    with StampPublicationLocks(published, sidecar):
        existing = _read_json(sidecar)
        if existing is not None:
            if existing.get("job_id") != request.job_id or existing.get("schema") != envelope.get("schema"):
                raise RecipeError("The clip already has a recipe of another take.")
            return {"stamped": True, "path": str(published), "sidecar": str(sidecar), "embedded": False}
        digest = _published_file_sha256(published)
        envelope["source_media_sha256"] = digest
        envelope["published_file_sha256"] = digest
        _validate_known_recipe_metadata(envelope)
        write_recipe_sidecar(published, envelope)
    return {"stamped": True, "path": str(published), "sidecar": str(sidecar), "embedded": False}



def _published_take_entry(row: Any) -> dict[str, Any] | None:
    """The renderer-equivalent recipe entry of a V4 take, from its stored request only (lane p13b-med, MED-017).

    Filled from the row: provider, target, client request id, the submitted slot values (args) and the requested
    file name. Not in the row, so never written: shot, take number, lineage parent and the page's mode."""
    request = row.request if isinstance(getattr(row, "request", None), dict) else {}
    args = request.get("args")
    if not isinstance(args, dict):
        return None
    entry: dict[str, Any] = {
        "args": args,
        "providerId": row.provider_id,
        "targetId": row.target_id,
        "clientRequestId": row.client_request_id,
        "filename": row.requested_basename,
    }
    return entry


def _stamp_published_take(row: Any) -> None:
    """Write a just-published project take's sidecar, the same one ``/recipe/stamp`` writes (lane p13b-med).

    Only a Contract-4 take of a project; an existing sidecar is never replaced (the same job's: nothing to do, the page's
    later stamp stays idempotent; another job's: kept, as ``_stamp_v4_take`` refuses it). The file and its receipt are
    already final here, so a sidecar that cannot be written never fails the take: it is logged and the page's stamp
    stays the fallback writer."""
    if getattr(row, "row_contract", None) != 4 or getattr(row, "state", None) != "ready" or not getattr(row, "project_id", None):
        return
    try:
        entry = _published_take_entry(row)
        if entry is None:
            return
        published = Path(_private_ops().saved_path_for_receipt(_private_ops().load_for_job(row)))
        envelope = _recipe_envelope(entry, job_id=row.studio_job_id, approved_roots=[])
        envelope["asset_identity"] = published.name  # the file the backend published (a suffix may differ from the request)
        if row.provider_job_id:
            envelope["provider_job_id"] = str(row.provider_job_id)
        else:
            envelope.pop("provider_job_id", None)
        sidecar = _sidecar_path(published)
        with StampPublicationLocks(published, sidecar):
            if sidecar.exists():
                return
            digest = _published_file_sha256(published)
            envelope["source_media_sha256"] = digest
            envelope["published_file_sha256"] = digest
            _validate_known_recipe_metadata(envelope)
            write_recipe_sidecar(published, envelope)
    except Exception as exc:  # noqa: BLE001 - a published take never fails on its sidecar
        import logging

        logging.getLogger("film_suite.recipe").warning(
            "publication stamp skipped for %s: %s", getattr(row, "studio_job_id", "?"), type(exc).__name__
        )


def _is_studio_job_id(job_id: str | None) -> bool:
    return isinstance(job_id, str) and job_id.startswith("studio:")


def _provider_prompt_id(job_id: str | None) -> str | None:
    """Resolve the Comfy provider prompt ID; propagate ledger access failures."""
    if not job_id or not _is_studio_job_id(job_id):
        return None
    store = _comfy_store()
    try:
        record = store.get_job(job_id)
    except _comfy_job_store_module.JobNotFoundError:
        return None
    if record is not None and record.provider_job_id:
        return str(record.provider_job_id)
    return None


def _validate_stored_identity_fields(recipe: dict[str, Any]) -> None:
    identity_aliases = (
        ("job_id", "jobId", "Stored recipe job identity"),
        ("provider_job_id", "providerJobId", "Stored provider job identity"),
        ("root_job_id", "rootJobId", "Stored root job identity"),
        ("provider_id", "providerId", "Stored recipe provider identity"),
        ("target_id", "targetId", "Stored recipe target identity"),
    )
    hash_aliases = (
        ("source_media_sha256", "sourceMediaSha256", "Stored source media hash"),
        ("published_file_sha256", "publishedFileSha256", "Stored published media hash"),
        ("workflow_api_hash", "workflowApiHash", "Stored workflow API hash"),
        ("editor_template_hash", "editorTemplateHash", "Stored editor template hash"),
        ("prepared_sha256", "preparedSha256", "Stored prepared media hash"),
        ("file_sha256", "fileSha256", "Stored file hash"),
    )
    for snake, camel, label in identity_aliases:
        for key in (snake, camel):
            if key in recipe:
                _present_portable_identity(recipe, key, label)
        _require_equal_present_aliases(recipe, (snake, camel), label)
    for snake, camel, label in hash_aliases:
        for key in (snake, camel):
            _present_sha256_identity(recipe, key, label)
        _require_equal_present_aliases(recipe, (snake, camel), label)


def _validate_known_recipe_metadata(recipe: dict[str, Any] | None) -> None:
    if not isinstance(recipe, dict):
        return
    schema = recipe.get("schema")
    if "schema" in recipe and (
        not isinstance(schema, str)
        or not schema
        or schema != schema.strip()
        or len(schema) > 128
        or "\0" in schema
    ):
        raise RecipeError("Stored recipe schema is invalid.")
    if schema == "studio.derivation/1":
        direct = _safe_derivation_record(recipe.get("derivation"))
        if direct is None:
            raise RecipeError("Stored media derivation metadata is invalid.")
        if "derivations" in recipe:
            raw_chain = recipe["derivations"]
            if (
                not isinstance(raw_chain, list)
                or not raw_chain
                or len(raw_chain) > MAX_DERIVATION_CHAIN_LENGTH
            ):
                raise RecipeError("Stored media derivation history is invalid.")
            validated = [_safe_derivation_record(item) for item in raw_chain]
            if any(item is None for item in validated) or validated[-1] != direct:
                raise RecipeError("Stored media derivation history is invalid.")
        _validate_stored_identity_fields(recipe)
        return
    if schema not in {"studio.recipe/1", "flux3.recipe/2"}:
        return

    entry = recipe.get("entry")
    if not isinstance(entry, dict):
        raise RecipeError("Stored recipe entry is invalid.")
    _validate_recipe_entry_identity_fields(entry)
    _validate_stored_identity_fields(recipe)

    if schema == "studio.recipe/1":
        provider_id = _present_portable_identity(
            recipe, "provider_id", "Stored recipe provider identity"
        )
        target_id = _present_portable_identity(
            recipe, "target_id", "Stored recipe target identity"
        )
        if provider_id is None or target_id is None:
            raise RecipeError("Stored Studio recipe identity is missing.")
        if "providerId" in entry:
            entry_provider_id = _present_portable_identity(
                entry, "providerId", "Stored entry provider identity"
            )
            if entry_provider_id != provider_id:
                raise RecipeError("Stored recipe provider identity is conflicting.")
        if "targetId" in entry:
            entry_target_id = _present_portable_identity(
                entry, "targetId", "Stored entry target identity"
            )
            if entry_target_id != target_id:
                raise RecipeError("Stored recipe target identity is conflicting.")
        if "provider_id" in entry and entry["provider_id"] != provider_id:
            raise RecipeError("Stored recipe provider identity is conflicting.")
        if "target_id" in entry and entry["target_id"] != target_id:
            raise RecipeError("Stored recipe target identity is conflicting.")
    else:
        if "providerId" in entry:
            _present_portable_identity(entry, "providerId", "Stored entry provider identity")
        if "targetId" in entry:
            _present_portable_identity(entry, "targetId", "Stored entry target identity")

    if "parent" in recipe:
        _lineage_parent(entry, recipe["parent"], [])


def _normalized_recipe_projection(recipe: dict[str, Any] | None) -> dict[str, Any] | None:
    """Section 3.8: provider-neutral projection of any stored recipe.

    ``recipe`` stays byte-compatible; this sibling projection exposes the
    provider/target identity and a stable entry for Edit in Create, whatever
    the stored schema (studio.recipe/1 or flux3.recipe/2).
    """
    if not isinstance(recipe, dict):
        return None
    entry = recipe.get("entry")
    if not isinstance(entry, dict):
        return None
    provider_id = recipe.get("provider_id") or entry.get("providerId") or "flux3-managed"
    target_id = recipe.get("target_id") or entry.get("targetId") or "text-to-video"
    return {
        "schema": "studio.recipe-view/1",
        "provider_id": provider_id,
        "target_id": target_id,
        "entry": entry,
        "source_schema": recipe.get("schema") or "flux3.recipe/2",
    }


@router.post("/recipe/read")
async def recipe_read(request: RecipeReadRequest) -> dict[str, Any]:
    try:
        path = _confined_local_path(request.path, _request_roots(request.approved_roots, request.project_ref, path=request.path))
        recipe = _read_embedded_recipe(path)
    except RecipeError as exc:
        if isinstance(exc, _FolderNotApproved):
            return {"recipe": None, "normalized_recipe": None, **_folder_refusal()}
        return {
            "recipe": None,
            "normalized_recipe": None,
            "error": f"Stored recipe metadata is invalid: {exc}",
        }

    if recipe is None:
        recipe = _read_json(_sidecar_path(path))
    try:
        _validate_known_recipe_metadata(recipe)
    except RecipeError as exc:
        return {
            "recipe": None,
            "normalized_recipe": None,
            "error": f"Stored recipe metadata is invalid: {exc}",
        }
    return {
        "recipe": recipe,
        "normalized_recipe": _normalized_recipe_projection(recipe),
    }


@router.post("/recipe/backfill")
async def recipe_backfill(request: RecipeBackfillRequest) -> dict[str, Any]:
    stamped = 0
    skipped = 0
    failed = 0
    errors: list[dict[str, Any]] = []

    for job in request.jobs:
        saved_path = job.get("savedPath") or job.get("saved_path")
        entry = job.get("entry")
        if not saved_path or not isinstance(entry, dict):
            skipped += 1
            continue
        existing = await recipe_read(
            RecipeReadRequest(path=str(saved_path), approved_roots=_request_roots(request.approved_roots, request.project_ref), project_ref=request.project_ref)
        )
        if existing.get("recipe") is not None:
            skipped += 1
            continue
        result = await recipe_stamp(
            RecipeStampRequest(
                path=str(saved_path),
                recipe=entry,
                job_id=job.get("id"),
                shot=job.get("shot"),
                take=job.get("take"),
                parent=job.get("parent"),
                approved_roots=_request_roots(request.approved_roots, request.project_ref), project_ref=request.project_ref,
            )
        )
        if result.get("stamped"):
            stamped += 1
        else:
            failed += 1
            errors.append({"id": job.get("id"), "path": saved_path, "error": result.get("error", "unknown error")})

    return {"stamped": stamped, "skipped": skipped, "failed": failed, "errors": errors}
