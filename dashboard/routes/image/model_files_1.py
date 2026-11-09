from __future__ import annotations


# --- WP6 M2: every model file ComfyUI can load, per Krea 2 file role --------
# The lists come from the running engine's /object_info loader inputs (GET
# only, cached briefly).  A file outside the catalog may be submitted only when
# its bare name is an exact member of this live list; the render-spec contract
# then checks it is a regular file in one of the role's model folders.
_IMAGE_MODEL_LOADERS = {"diffusion_model": ("UNETLoader", "unet_name"),
                        "text_encoder": ("CLIPLoader", "clip_name"),
                        "vae": ("VAELoader", "vae_name")}
_IMAGE_MODEL_FILES_TTL_SECONDS = 30.0
_IMAGE_MODEL_FILES_CACHE: dict[str, Any] = {"until": 0.0, "lists": None}


class ImageModelFilesRequest(_ProjectRequest):
    schema_: Literal["studio.image-producer-operation/1"] = Field(alias="schema")
    kind: Literal["image_producer_model_files"]
    refresh: bool = False


def _image_loader_options(body: Any, class_name: str, field: str) -> list[str]:
    info = body.get(class_name) if isinstance(body, dict) else None
    spec = (((info or {}).get("input") or {}).get("required") or {}).get(field) if isinstance(info, dict) else None
    values: Any = []
    if isinstance(spec, list) and spec:
        if isinstance(spec[0], list):
            values = spec[0]
        elif len(spec) > 1 and isinstance(spec[1], dict):
            values = spec[1].get("options") or []
    return [value for value in values if isinstance(value, str)][:4096]


async def _image_live_model_files(refresh: bool = False) -> dict[str, list[str]]:
    """Loader options per role; raises the client's error when unreachable."""
    now = time.monotonic()
    cached = _IMAGE_MODEL_FILES_CACHE["lists"]
    if not refresh and cached is not None and now < _IMAGE_MODEL_FILES_CACHE["until"]:
        return cached
    client = _comfy_client_module.ComfyUIClient()
    lists: dict[str, list[str]] = {}
    try:
        for role, (class_name, field) in _IMAGE_MODEL_LOADERS.items():
            lists[role] = _image_loader_options(await client.object_info(class_name), class_name, field)
    except Exception:
        _image_model_availability().forget_live()  # w24-models: a restarted engine is read anew
        raise
    _IMAGE_MODEL_FILES_CACHE.update(until=now + _IMAGE_MODEL_FILES_TTL_SECONDS, lists=lists)
    availability = _image_model_availability()
    availability.record_live({availability.folder_of(role): names for role, names in lists.items()},
                             ttl=_IMAGE_MODEL_FILES_TTL_SECONDS)
    return lists


def _image_model_availability() -> Any:
    """w24-models: the one answer to "can ComfyUI load this file" (live list, else its folders)."""
    from importlib import import_module
    return import_module(_image_contract_module().__package__ + ".model_availability")


async def _image_model_missing(spec: Any) -> JSONResponse | None:
    """w24-models item 4: a recipe naming a file ComfyUI cannot load is refused at submit, named.

    The engine's lists are read first (cached briefly), so a file on another drive through
    extra_model_paths counts; an engine that does not answer leaves the folder fallback.
    """
    from importlib import import_module
    spec_module = import_module(_image_contract_module().__package__ + ".krea2_render_spec")
    missing: list[dict[str, Any]] = []
    for refresh in (False, True):  # a missing file is re-read once: a model installed a moment ago counts
        for read in (_image_live_model_files, _image_live_loras):
            try:
                await read(refresh)
            except Exception:  # noqa: BLE001 - engine down: folders answer
                _image_model_availability().forget_live()
        try:
            missing = await asyncio.to_thread(spec_module.missing_files, spec)
        except ValueError:
            return None  # malformed: the contract's own refusal follows
        if not missing:
            return None
    first = missing[0]
    return JSONResponse(status_code=422, content={"error": {
        "code": "model_file_missing", "message": f"Model file missing: {first['name']} ({first['folder']})",
        "missing": missing}})


def _image_model_files_rows(spec_module: Any, lists: dict[str, list[str]]) -> dict[str, list[dict[str, Any]]]:
    choices = spec_module.v2_file_choices()
    rows: dict[str, list[dict[str, Any]]] = {}
    for role, names in lists.items():
        known = {item["name"]: item["variant"] for item in choices.get(role, [])}
        rows[role] = [
            {"name": name, "known": name in known, **({"variant": known[name]} if name in known else {})}
            for name in sorted(set(names), key=str.lower)
            if name in known or spec_module.custom_file_folder(role, name) is not None
        ]
    return rows


def _image_model_files_local(spec_module: Any) -> dict[str, list[dict[str, Any]]]:
    """Per role, the model files in the role's ComfyUI folders (engine down), tagged like the live rows."""
    availability = _image_model_availability()
    root = Path(spec_module.MODEL_ROOT)
    lists: dict[str, list[str]] = {}
    for role in _IMAGE_MODEL_LOADERS:
        names: list[str] = []
        for folder in availability.local_roots(availability.folder_of(role), root):
            try:
                names += [entry.name for entry in Path(folder).iterdir() if entry.is_file() and not entry.is_symlink()]
            except OSError:
                continue
        lists[role] = names[:4096]
    return _image_model_files_rows(spec_module, lists)


@project_router.post("/projects/image-producer/model-files")
async def image_producer_model_files(request: ImageModelFilesRequest) -> Any:
    """Every file ComfyUI lists for the diffusion model, text encoder and VAE loaders."""
    from importlib import import_module
    spec_module = import_module(_image_contract_module().__package__ + ".krea2_render_spec")
    try:
        lists = await _image_live_model_files(request.refresh)
    except Exception:  # noqa: BLE001 - engine not answering is a state, not an error
        # w24-models item 2: the engine is down, so the files come from the folders ComfyUI lists them from
        # (models/ + extra_model_paths.yaml) as ``local``; ``roles`` stays the engine's own (none).
        local = await asyncio.to_thread(_image_model_files_local, spec_module)
        return {"schema": "studio.image-model-files/1", "state": "unreachable", "roles": None, "local": local}
    rows = await asyncio.to_thread(_image_model_files_rows, spec_module, lists)
    return {"schema": "studio.image-model-files/1", "state": "ready", "roles": rows}


async def _image_model_files_admit(spec: Any) -> JSONResponse | None:
    """A v2 spec naming a non-catalog file needs that name in the live list."""
    if not isinstance(spec, dict) or spec.get("schema") != "studio.krea2-render-spec/2" or not isinstance(spec.get("files"), dict):
        return None
    from importlib import import_module
    choices = import_module(_image_contract_module().__package__ + ".krea2_render_spec").v2_file_choices()
    custom = {role: name for role, name in spec["files"].items()
              if role in _IMAGE_MODEL_LOADERS and name not in {item["name"] for item in choices.get(role, [])}}
    if not custom:
        return None
    try:
        lists = await _image_live_model_files()
    except Exception:  # noqa: BLE001
        return _edit_engine_unreachable()
    for role, name in custom.items():
        if type(name) is not str or name not in lists.get(role, []):
            return _edit_error(422, "model_file_unknown", "The image engine does not list this model file")
    return None
