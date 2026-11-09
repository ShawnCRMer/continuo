from __future__ import annotations


# --- Picture export with the recipe in the file; restore by drop (lane w1-imgout) ----------------
# generation/image_export.py writes PNG 8/16, JPEG, WebP and TIFF 8/16 with the sRGB profile and,
# when metadata is on, the Suite's studio.image-recipe/1 (PNG iTXt, XMP otherwise).  Like the
# timeline and script exports, a picture is exported into its project: <project>/exports/images/
# (No project has no exports folder and is refused with that reason).  A file is never replaced:
# a taken name gets -2, -3 ...  /image/recipe/read answers the recipe of a dropped file.
_IMAGE_EXPORT_NAME_RE = re.compile(r"[^A-Za-z0-9._ -]+")
_IMAGE_EXPORT_NEEDS_PROJECT = "No project open: exports go into a project's exports folder."


def _image_export_module() -> Any:
    from importlib import import_module
    return import_module(_image_contract_module().__package__ + ".image_export")


class ImageExportSize(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["original", "long_edge", "percent"] = "original"
    value: int | float | None = None


class ImageExportPixels(BaseModel):
    """The picture as Edit shows it (M-43): the flattened PNG the page drew, its sha256 and what it holds."""

    model_config = ConfigDict(extra="forbid")

    png_b64: str = Field(min_length=8, max_length=_COMPOSE_B64_MAX)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    recipe: dict[str, Any] = Field(default_factory=dict)


class ImageExportRequest(BaseModel):
    """studio.image-export-request/1; host keys the desktop may add (``profile`` ...) are ignored."""

    model_config = ConfigDict(extra="ignore")

    schema_: Literal["studio.image-export-request/1"] = Field(alias="schema")
    owner_scope: str = Field(pattern=_OWNER_SCOPE_PATTERN, min_length=18, max_length=47)
    source: ImageEditSource
    format: Literal["png", "jpeg", "webp", "tiff"] = "png"
    bit_depth: Any = None
    quality: Any = None
    lossless: Any = False
    size: ImageExportSize | None = None
    color_profile: Literal["srgb", "display_p3"] = "srgb"
    metadata: Any = True
    name: str = Field(default="", max_length=100)
    # M-43: `pixels` is what Edit shows (develop, masks, layers, crop); without it a picture with saved
    # edits is refused (export_has_edits) unless `original` asks for the unedited picture on purpose.
    pixels: ImageExportPixels | None = None
    original: bool = False


_IMAGE_EXPORT_HAS_EDITS = ("The picture has edits that are not in the published picture. Export it from Edit "
                           "(continuo_image_action export_edit writes it as shown), or send original=true for the unedited picture.")


def _image_export_saved_develop(owner_scope: str, sha256: str) -> bool:
    """True when the scope keeps develop values for this picture (studio.image-edits/1 keeps none for a neutral one)."""
    def work(folder: Any) -> bool:
        record = _image_edits_store.read_one(folder, owner_scope, sha256)
        return bool(record and record.get("develop"))
    try:
        return bool(_image_edits_in_scope(owner_scope, False, work))
    except Exception:  # noqa: BLE001 - unreadable edits: the export goes on as asked
        return False


def _image_export_pixels(pixels: ImageExportPixels) -> bytes:
    try:
        data = base64.b64decode(pixels.png_b64, validate=True)
    except (ValueError, TypeError):
        raise ValueError("pixels are not base64")
    if hashlib.sha256(data).hexdigest() != pixels.sha256:
        raise ValueError("pixels do not match their sha256")
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("pixels are not a PNG")
    return data


class ImageRecipeReadRequest(_ProjectRequest):
    """A dropped picture: its bytes, or a file inside an approved root (read once, never kept)."""

    schema_: Literal["studio.image-recipe-read/1"] = Field(alias="schema")
    data_b64: str | None = Field(default=None, min_length=8, max_length=89_478_488)
    path: str | None = Field(default=None, min_length=1, max_length=4096)
    approved_roots: list[str] = Field(default_factory=list, max_length=64)
    project_ref: ProjectReference | None = None


def _image_export_parent(job: Mapping[str, Any], jobs: Any) -> str | None:
    """The sha256 of the picture this one was made from (edit, upscale, compose: its source)."""
    snap = job.get("request_snapshot") or {}
    effective = snap.get("effective_input") or {}
    for key, field in (("upscale", "source_job_id"), ("compose", "source_job_id")):
        source = (effective.get(key) or {}).get(field)
        if source:
            try:
                return (jobs.get_job(source).get("artifact") or {}).get("artifact_sha256")
            except Exception:  # noqa: BLE001 - parent gone: unknown, never guessed
                return None
    edit = effective.get("edit") or {}
    return edit.get("source_sha256")


def _image_export_write(project_id: str, file_name: str, data: bytes) -> str:
    """Create <project>/exports/images/<name> (never replacing a file); answers the relative path."""
    store = _project_store_factory()
    ps = sys.modules[type(store).__module__]
    directory = store._directory(_recent_directory(_recent_project_store_factory(), project_id))
    identity = store.boundary.pin_directory(directory)
    with store.boundary.project_lock(directory):
        store.boundary.revalidate_directory(directory, identity)
        if store._manifest_locked(directory)["project_id"] != project_id:
            raise studio_core.ProjectError("project_scope_mismatch", "the picture belongs to another project")
        folder = directory
        for part in ("exports", "images"):
            folder = folder / part
            if not os.path.lexists(folder):
                try:
                    folder.mkdir()
                except FileExistsError:
                    pass
            if folder.is_symlink() or not folder.is_dir():
                raise studio_core.ProjectError("export_unavailable", f"{part} is not a plain folder")
        folder_identity = store.boundary.pin_directory(folder)
        stem, dot, ext = file_name.rpartition(".")
        for n in range(1, 1000):
            name = file_name if n == 1 else f"{stem}-{n}{dot}{ext}"
            target = folder / name
            if os.path.lexists(target):
                continue
            temp = ps._stage_temp(folder, data)
            try:
                store.boundary.revalidate_directory(directory, identity)
                store.boundary.revalidate_directory(folder, folder_identity)
                ps._replace_temp(temp, target, no_replace=True)
            except FileExistsError:
                continue
            finally:
                if os.path.lexists(temp):
                    try:
                        os.unlink(temp)
                    except OSError:
                        pass
            return f"exports/images/{name}"
    raise studio_core.ProjectError("export_unavailable", "no free file name")


@project_router.post("/image/export")
async def image_export(request: ImageExportRequest) -> Any:
    """Export an owned picture (or a kept version) as a file in the project's exports/images/ folder.

    The pixels are the published picture's; the recipe is read from the picture's own ledger
    snapshot (model, prompt, seed, settings, parent sha256).  Metadata off writes no recipe.
    """
    unknown = _edit_error(404, "picture_unknown", "The picture is unknown")
    try:
        exporter = _image_export_module()
        project_id = request.owner_scope[len("film_owner:"):]
        if project_id == "scratch":
            return _edit_error(409, "export_needs_project", _IMAGE_EXPORT_NEEDS_PROJECT)
        async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
            root = _image_producer_root(request.owner_scope)
            handoff = _image_operations_handoff(root, request.owner_scope)
            jobs = handoff.producer_jobs()
            states = handoff.image_states()
            _collector, reader = handoff.result_services()
        source_job_id = _edit_source_job_id(states, jobs, request.source)
        job = _edit_owned_job(jobs, states, source_job_id) if source_job_id else None
        if job is None:
            return unknown
        if job.get("state") != "published":
            return _edit_error(409, "picture_not_ready", "The picture is not ready yet")
        artifact = job["artifact"]
        if request.source.sha256 and request.source.sha256 != artifact["artifact_sha256"]:
            return _edit_error(409, "source_changed", "The picture changed since it was selected")
        raw = {"format": request.format, "color_profile": request.color_profile, "metadata": request.metadata,
               "lossless": request.lossless}
        for key in ("bit_depth", "quality"):
            if getattr(request, key) is not None:
                raw[key] = getattr(request, key)
        if request.size is not None:
            raw["size"] = request.size.model_dump()
        try:
            settings = exporter.normalize(raw, (artifact["width"], artifact["height"]))
        except exporter.ExportError as exc:
            return _edit_error(422, exc.code, exc.message)
        recipe = exporter.recipe_for(job, parent_sha256=_image_export_parent(job, jobs))
        if request.source.revision_id:
            recipe["revision_id"] = request.source.revision_id
        shown_png = None
        if request.pixels is not None and not request.original:
            try:
                shown_png = _image_export_pixels(request.pixels)
            except ValueError as exc:
                return _edit_error(422, "invalid_pixels", str(exc))
            recipe["edit"] = {"sha256": request.pixels.sha256, **{k: v for k, v in request.pixels.recipe.items() if k != "sha256"}}
        elif not request.original and _image_export_saved_develop(request.owner_scope, artifact["artifact_sha256"]):
            return _edit_error(409, "export_has_edits", _IMAGE_EXPORT_HAS_EDITS)
        recipe["export"] = {k: settings[k] for k in ("format", "bit_depth", "quality", "lossless", "color_profile", "size", "width", "height")}
        if shown_png is not None:
            source_png = shown_png
            # The PNG header (IHDR) names the size; exporter.encode decodes and checks the pixels.
            shown_size = (int.from_bytes(shown_png[16:20], "big"), int.from_bytes(shown_png[20:24], "big"))
            if shown_size != (artifact["width"], artifact["height"]):
                # A crop: the size settings apply to the picture as shown.
                try:
                    settings = exporter.normalize(raw, shown_size)
                except exporter.ExportError as exc:
                    return _edit_error(422, exc.code, exc.message)
                recipe["export"] = {k: settings[k] for k in ("format", "bit_depth", "quality", "lossless", "color_profile", "size", "width", "height")}
        else:
            source_png, _artifact = await asyncio.to_thread(reader.read_published_job, source_job_id)
        try:
            made = await asyncio.to_thread(exporter.encode, source_png, settings, recipe)
        except exporter.ExportError as exc:
            return _edit_error(422, exc.code, exc.message)
        stem = _IMAGE_EXPORT_NAME_RE.sub("-", request.name).strip(" .-")[:80] or ("picture-" + artifact["artifact_sha256"][:8])
        file_name = f"{stem}-{settings['width']}x{settings['height']}.{exporter.FORMATS[settings['format']]['ext']}"
        relative = await asyncio.to_thread(_image_export_write, project_id, file_name, made["data"])
        return {"schema": "studio.image-export-result/1", "owner_scope": request.owner_scope, "file": relative,
                "bytes": len(made["data"]), "sha256": hashlib.sha256(made["data"]).hexdigest(),
                "format": settings["format"], "bit_depth": made["bit_depth"], "width": made["width"], "height": made["height"],
                "icc": made["icc"], "alpha": made["alpha"], "metadata": made["metadata"],
                "recipe": recipe if made["metadata"] else None, "warnings": made["warnings"],
                "source": {"image_producer_job_id": source_job_id, "sha256": artifact["artifact_sha256"]},
                "shown": shown_png is not None}
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_film_error_response(exc)


@project_router.get("/image/export/options")
async def image_export_options(request: Request) -> Any:
    """Formats, bit depths, quality, size modes and colour profiles Export offers (with reasons)."""
    exporter = _image_export_module()
    return JSONResponse({
        "schema": "studio.image-export-options/1", "request_schema": exporter.REQUEST_SCHEMA,
        "formats": [{"id": k, "extension": v["ext"], "bit_depths": list(v["bit_depths"]), "quality": v["quality"],
                     "default_bit_depth": exporter.DEFAULT_BIT_DEPTH[k], **({"lossless": True} if k == "webp" else {})}
                    for k, v in exporter.FORMATS.items()],
        "quality": {"min": 1, "max": 100, "default": 92}, "size_modes": list(exporter.SIZE_MODES),
        "long_edge": {"min": 16, "max": exporter.MAX_SIDE}, "percent": {"min": 1, "max": 400},
        "color_profiles": [{"id": "srgb", "available": True}, {"id": "display_p3", "available": False, "reason": exporter.P3_REASON}],
        "metadata": {"default": True, "recipe_schema": exporter.RECIPE_SCHEMA,
                     "where": {"png": "iTXt " + exporter.PNG_KEY, "jpeg": "XMP", "webp": "XMP", "tiff": "XMP"}},
        "folder": "exports/images", "needs_project": True,
    }, headers={"Cache-Control": "no-store"})


@project_router.post("/image/recipe/read")
async def image_recipe_read(request: ImageRecipeReadRequest) -> Any:
    """The Suite recipe a dropped picture carries (studio.image-recipe/1), so the page can offer its settings."""
    def refuse(code: str, message: str) -> JSONResponse:
        return JSONResponse(status_code=422, content={"error": {"code": code, "message": message}})

    try:
        exporter = _image_export_module()
        imported = importlib.import_module(_image_contract_module().__package__ + ".image_import")
        if (request.path is None) == (request.data_b64 is None):
            return refuse("invalid_recipe_read", "Needs exactly one of a file path or picture bytes")
        if request.path is not None:
            if not request.approved_roots:
                return refuse("path_refused", imported.OUTSIDE_ROOTS)
            try:
                roots = _request_roots(request.approved_roots, request.project_ref, path=request.path)
            except _FolderNotApproved:
                return JSONResponse(status_code=403, content=_folder_refusal())
            try:
                data, _name = await asyncio.to_thread(imported.read_local, request.path, roots, _confined_local_path)
            except imported.ImageImportError as exc:
                return refuse(exc.code, exc.message)
        else:
            try:
                data = base64.b64decode(request.data_b64, validate=True)
            except (ValueError, TypeError):
                return refuse("not_a_picture", imported.NOT_A_PICTURE)
        found = await asyncio.to_thread(exporter.read_recipe, data)
        return {"schema": "studio.image-recipe-read-result/1", **found}
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_film_error_response(exc)
