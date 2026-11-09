from __future__ import annotations


# --- Image edits: develop values + History kept with the owner scope ---------
# studio.image-edits/1 (generation/image_edits_store.py): one file per picture,
# keyed by the Original's sha256, each with its own expected_revision. A project
# keeps them in its own state folder (state/image-edits/<sha256>.json, so they
# move with the project folder); No project keeps them in the profile's private
# plugin data. Bounds refuse, never evict; nothing is deleted. Before this the
# pane kept them in plugin localStorage (64 / 16 pictures, then the oldest went).
def _load_image_edits_store_module() -> Any:
    """generation/image_edits_store.py by path (stdlib only, not a generation package member)."""
    source = Path(__file__).with_name("generation") / "image_edits_store.py"
    spec = importlib.util.spec_from_file_location("film_suite_image_edits_store", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"The image edits store is unavailable: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_image_edits_store = _load_image_edits_store_module()


class ImageEditsReadRequest(_ProjectRequest):
    """`sha256s` names the pictures to read (up to 16); without it the answer lists every picture of the scope."""
    schema_: Literal["studio.image-edits/1"] = Field(alias="schema")
    owner_scope: str = Field(pattern=_OWNER_SCOPE_PATTERN, min_length=18, max_length=47)
    sha256s: list[str] | None = Field(default=None, min_length=1, max_length=16)

    @field_validator("sha256s")
    @classmethod
    def _sha_list(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and (any(not _image_edits_store.SHA_RE.fullmatch(s) for s in value) or len(set(value)) != len(value)):
            raise ValueError("each sha256 is 64 lowercase hex characters, once")
        return value


class ImageEditsWriteRequest(_ProjectRequest):
    """One picture's develop values and packed History, written against its revision (0 = none yet)."""
    schema_: Literal["studio.image-edits/1"] = Field(alias="schema")
    owner_scope: str = Field(pattern=_OWNER_SCOPE_PATTERN, min_length=18, max_length=47)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$", min_length=64, max_length=64)
    expected_revision: int = Field(ge=0)
    develop: dict[str, Any] | None
    history: dict[str, Any] | None


def _image_edits_scratch_folder(create: bool) -> Path:
    home = _active_hermes_home()
    if home is None:
        raise _image_edits_store.ImageEditsError("image_edits_unavailable", "No profile is active.")
    folder = Path(state_root(Path(home) / "plugin-data", "image-edit-catalog")) / "no-project"
    if create:
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise _image_edits_store.ImageEditsError("image_edits_unavailable", "The edits folder could not be created.") from exc
    return folder


def _image_edits_in_scope(owner_scope: str, create: bool, work: Any) -> Any:
    """Run `work(folder)` with the scope's edits folder under its locks (project lock first)."""
    project_id = owner_scope[len("film_owner:"):]
    if project_id == "scratch":
        folder = _image_edits_scratch_folder(create)
        with _image_edits_store.folder_lock(folder):
            return work(folder)
    store = _project_store_factory()
    directory = store._directory(_recent_directory(_recent_project_store_factory(), project_id))
    identity = store.boundary.pin_directory(directory)
    with store.boundary.project_lock(directory):
        store.boundary.revalidate_directory(directory, identity)
        manifest = store._manifest_locked(directory)
        if manifest["project_id"] != project_id:
            raise studio_core.ProjectError("project_scope_mismatch", "image edits belong to another project")
        state_dir = directory / "state"
        if create:
            # state/ holds shoot.json first; a folder beside a missing shoot state would read as damaged.
            if store._state_locked(directory, manifest) is None:
                store._create_default_state_locked(directory, manifest)
            state_dir = store._ensure_state_dir(directory)
        elif not _image_edits_store.check_folder(state_dir):
            return work(None)
        folder = state_dir / "image-edits"
        if create and not os.path.lexists(folder):
            try:
                folder.mkdir()
            except FileExistsError:
                pass
            except OSError as exc:
                raise _image_edits_store.ImageEditsError("image_edits_unavailable", "The edits folder could not be created.") from exc
        store.boundary.revalidate_directory(directory, identity)
        with _image_edits_store.folder_lock(folder):
            return work(folder)


def _image_edits_read(request: ImageEditsReadRequest) -> dict[str, Any]:
    def work(folder: Path | None) -> dict[str, Any]:
        seen = _image_edits_store.index(folder)
        out: dict[str, Any] = {"count": seen["count"], "bytes": seen["bytes"],
                               "limits": {"picture_bytes": _image_edits_store.PICTURE_MAX_BYTES, "pictures": _image_edits_store.PICTURES_MAX, "scope_bytes": _image_edits_store.SCOPE_MAX_BYTES}}
        if request.sha256s is None:
            out["index"] = sorted(seen["pictures"])
            return out
        return {**out, **_image_edits_store.read_many(folder, request.owner_scope, request.sha256s)}
    return _image_edits_in_scope(request.owner_scope, False, work)


def _image_edits_write(request: ImageEditsWriteRequest) -> dict[str, Any]:
    _image_edits_store.validate_item(request.sha256, request.develop, request.history)
    return _image_edits_in_scope(request.owner_scope, True, lambda folder: _image_edits_store.write_one(
        folder, request.owner_scope, request.sha256, request.expected_revision, request.develop, request.history))


_IMAGE_EDITS_STATUS = {"image_edits_conflict": 409, "invalid_image_edits": 422, "image_edits_too_large": 413,
                       "image_edits_full": 422, "image_edits_damaged": 409, "image_edits_unavailable": 503}


def _image_edits_failure(exc: BaseException) -> JSONResponse:
    if isinstance(exc, _image_edits_store.ImageEditsError):
        content: dict[str, Any] = {"error": {"code": exc.code, "message": exc.message}}
        if exc.revision is not None:
            content["revision"] = exc.revision
        return JSONResponse(status_code=_IMAGE_EDITS_STATUS.get(exc.code, 503), content=content)
    code = getattr(exc, "code", None)
    if code == "project_not_found":
        return _edit_error(404, "project_not_found", "Project not found")
    return _edit_error(503, "image_edits_unavailable", "The edits could not be read or saved.")


@project_router.post("/image/edits/read")
async def image_edits_read(request: ImageEditsReadRequest):
    """The scope's kept edits: with sha256s their develop values, History and revision; without, the list of pictures."""
    try:
        result = await asyncio.to_thread(_image_edits_read, request)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_edits_failure(exc)
    return {"schema": _image_edits_store.SCHEMA, "owner_scope": request.owner_scope, **result}


@project_router.post("/image/edits/write")
async def image_edits_write(request: ImageEditsWriteRequest):
    """Keep one picture's develop values and History with expected_revision (409 + current revision on conflict)."""
    try:
        result = await asyncio.to_thread(_image_edits_write, request)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_edits_failure(exc)
    return {"schema": _image_edits_store.SCHEMA, "owner_scope": request.owner_scope, "sha256": request.sha256, **result}
