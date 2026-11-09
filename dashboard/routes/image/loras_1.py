from __future__ import annotations


# --- WP7 LoRA library ------------------------------------------------------
# Every LoRA the running engine lists (LoraLoader / LoraLoaderModelOnly
# lora_name, GET /object_info, cached briefly) plus the builder catalog's
# LoRAs, with local metadata (sidecars, safetensors header, preview picture)
# and an on-demand Civitai lookup by SHA256.  Caches live in plugin-data only.
# A submit naming a non-catalog LoRA is admitted only when the name is an
# exact member of the live list (same rule as WP6 model files).
def _load_lora_library_module() -> Any:
    module_path = Path(__file__).with_name("generation") / "lora_library.py"
    spec = importlib.util.spec_from_file_location("film_suite_lora_library", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("The LoRA library module is unavailable.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_lora_library_module = _load_lora_library_module()
_IMAGE_LORA_LOADERS = (("LoraLoader", "lora_name"), ("LoraLoaderModelOnly", "lora_name"))
_IMAGE_LORA_TTL_SECONDS = 30.0
_IMAGE_LORA_CACHE: dict[str, Any] = {"until": 0.0, "names": None}
_IMAGE_LORA_LIBRARIES: dict[str, Any] = {}
_IMAGE_LORA_NAME = r"^[^\\:*?\"<>|\x00-\x1f]{1,240}\.safetensors$"


class ImageLorasRequest(_ProjectRequest):
    schema_: Literal["studio.image-producer-operation/1"] = Field(alias="schema")
    kind: Literal["image_producer_loras"]
    refresh: bool = False


class ImageLoraThumbsRequest(_ProjectRequest):
    schema_: Literal["studio.image-producer-operation/1"] = Field(alias="schema")
    kind: Literal["image_producer_lora_thumbs"]
    names: list[str] = Field(min_length=1, max_length=60)


class ImageLoraInfoRequest(_ProjectRequest):
    schema_: Literal["studio.image-producer-operation/1"] = Field(alias="schema")
    kind: Literal["image_producer_lora_info"]
    name: str = Field(pattern=_IMAGE_LORA_NAME, min_length=13, max_length=240)
    fetch: bool = False


def _image_lora_library() -> Any:
    """One library per plugin-data root; None when no profile home resolves."""
    home = _active_hermes_home()
    try:
        root = None if home is None else Path(state_root(Path(home) / "plugin-data", "lora-library"))
    except Exception:  # noqa: BLE001 - a conflicting legacy root means no cache, not a failure
        root = None
    key = str(root)
    if key not in _IMAGE_LORA_LIBRARIES:
        _IMAGE_LORA_LIBRARIES[key] = _lora_library_module.LoraLibrary(root)
    return _IMAGE_LORA_LIBRARIES[key]


async def _image_live_loras(refresh: bool = False) -> list[str]:
    """LoRA names the engine lists (``/`` separators); raises when unreachable."""
    now = time.monotonic()
    cached = _IMAGE_LORA_CACHE["names"]
    if not refresh and cached is not None and now < _IMAGE_LORA_CACHE["until"]:
        return cached
    client = _comfy_client_module.ComfyUIClient()
    names: list[str] = []
    for class_name, field in _IMAGE_LORA_LOADERS:
        try:
            body = await client.object_info(class_name)
        except Exception:
            if class_name == "LoraLoader":
                _image_model_availability().forget_live()  # w24-models
                raise
            continue  # one loader class is enough
        for value in _image_loader_options(body, class_name, field):
            name = _lora_library_module.normalize_name(value)
            if name not in names:
                names.append(name)
    _IMAGE_LORA_CACHE.update(until=now + _IMAGE_LORA_TTL_SECONDS, names=names)
    _image_model_availability().record_live({"loras": names}, ttl=_IMAGE_LORA_TTL_SECONDS)  # w24-models
    return names


def _image_lora_catalog() -> dict[str, dict[str, Any]]:
    try:
        catalog_path = Path(__file__).resolve().parent / "image_builder" / "krea2-builder-catalog.v1.json"
        options = json.loads(catalog_path.read_text(encoding="utf-8"))["options"]
    except (OSError, ValueError, KeyError):
        return {}
    return {o["file"]: o for o in options if o.get("axis") == "lora" and isinstance(o.get("file"), str)}


def _image_lora_root() -> Path:
    from importlib import import_module
    return Path(import_module(_image_contract_module().__package__ + ".krea2_render_spec").MODEL_ROOT) / "loras"


def _image_lora_path(root: Path, name: str) -> Path | None:
    """w24-models: the LoRA file where ComfyUI finds it: models/loras, then extra_model_paths.yaml folders."""
    path = _lora_library_module.lora_file(root, name)
    if path is None and _lora_library_module.name_ok(name):
        path = _image_model_availability().local_file("loras", name, Path(root).parent)
    return path


def _image_lora_row(library: Any, root: Path, name: str, catalog: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    path = _image_lora_path(root, name)
    known = catalog.get(name)
    if path is None and known is None:
        return None
    local = library.local(path) if path is not None else {"triggers": [], "base_model": "", "has_preview": False}
    record = library.civitai_cached(path) if path is not None else None
    triggers = (record or {}).get("triggers") or local.get("triggers") or []
    if not triggers and known and known.get("text"):
        triggers = _lora_library_module._words(str(known["text"]))
    row: dict[str, Any] = {
        "name": name, "installed": path is not None, "known": known is not None,
        "label": str((known or {}).get("label") or (record or {}).get("model_name") or local.get("model_name") or Path(name).stem)[:120],
        "base_model": str((record or {}).get("base_model") or local.get("base_model") or "")[:80],
        "triggers": triggers[:24], "has_preview": bool(local.get("has_preview")) or bool((record or {}).get("examples")),
        "civitai": _lora_library_module.civitai_state(record),
        "folder": name.rsplit("/", 1)[0] if "/" in name else "",
    }
    if known is not None:
        row["key"] = known.get("key")
        row["catalog_thumb"] = known.get("thumb")
        if isinstance(known.get("strength"), (int, float)):
            row["strength"] = known["strength"]
    elif isinstance(local.get("strength"), float):
        row["strength"] = local["strength"]
    return row


def _image_lora_rows(names: list[str] | None) -> list[dict[str, Any]]:
    library = _image_lora_library()
    root = _image_lora_root()
    catalog = _image_lora_catalog()
    ordered = list(dict.fromkeys([*(names or []), *catalog]))
    rows = [row for name in ordered if (row := _image_lora_row(library, root, name, catalog)) is not None]
    return sorted(rows, key=lambda row: (not row["known"], row["label"].lower()))


async def _image_lora_allowed(name: str) -> bool:
    """A catalog LoRA, or an exact member of the engine's live list."""
    if name in _image_lora_catalog():
        return True
    try:
        return name in await _image_live_loras()
    except Exception:  # noqa: BLE001
        return False


@project_router.post("/projects/image-producer/loras")
async def image_producer_loras(request: ImageLorasRequest) -> Any:
    """Every LoRA the engine lists plus the catalog's, with local metadata."""
    try:
        names = await _image_live_loras(request.refresh)
        state = "ready"
    except Exception:  # noqa: BLE001 - engine not answering is a state, not an error
        names, state = None, "unreachable"
    rows = await asyncio.to_thread(_image_lora_rows, names)
    return {"schema": "studio.image-loras/1", "state": state, "loras": rows}


@project_router.post("/projects/image-producer/lora-thumbs")
async def image_producer_lora_thumbs(request: ImageLoraThumbsRequest) -> Any:
    """Small preview pictures (data URLs) for up to 60 listed LoRAs."""
    library = _image_lora_library()
    root = _image_lora_root()
    wanted = [name for name in dict.fromkeys(request.names) if _lora_library_module.name_ok(name)]
    allowed = [name for name in wanted if await _image_lora_allowed(name)]

    def read() -> dict[str, str | None]:
        out: dict[str, str | None] = {}
        for name in allowed:
            path = _image_lora_path(root, name)
            out[name] = library.thumb(path) if path is not None else None
        return out
    return {"thumbs": await asyncio.to_thread(read)}


@project_router.post("/projects/image-producer/lora-info")
async def image_producer_lora_info(request: ImageLoraInfoRequest) -> Any:
    """One LoRA's detail; fetch=true asks Civitai once by the file's SHA256."""
    name = request.name
    if not _lora_library_module.name_ok(name) or not await _image_lora_allowed(name):
        return _edit_error(404, "lora_unknown", "The image engine does not list this LoRA")
    library = _image_lora_library()
    root = _image_lora_root()
    path = _image_lora_path(root, name)
    if path is None:
        return _edit_error(404, "lora_not_installed", "This LoRA file is not installed")
    error = None
    if request.fetch:
        try:
            await asyncio.to_thread(library.fetch_civitai, path)
        except _lora_library_module.LoraLibraryError as exc:
            error = {"code": exc.code, "message": exc.message}
        except OSError:
            error = {"code": "lora_unreadable", "message": "This LoRA file could not be read"}

    def detail() -> dict[str, Any]:
        catalog = _image_lora_catalog()
        row = _image_lora_row(library, root, name, catalog) or {"name": name}
        local = library.local(path)
        record = library.civitai_cached(path)
        examples = []
        for item in (record or {}).get("examples") or []:
            url = library.example_data_url(record["sha256"], int(item.get("index", -1)))
            if url:
                examples.append({"src": url, "prompt": item.get("prompt") or "", "width": item.get("width"), "height": item.get("height")})
        return {
            **row, "thumb": library.thumb(path), "examples": examples,
            "description": str((record or {}).get("description") or local.get("description") or "")[:2000],
            "model_name": (record or {}).get("model_name") or local.get("model_name") or "",
            "version_name": (record or {}).get("version_name") or local.get("version_name") or "",
            "tags": local.get("tags") or [], "sources": local.get("sources") or [],
            "link": _lora_library_module.civitai_link(record, local),
            "fetched_at": (record or {}).get("fetched_at"),
        }
    body = await asyncio.to_thread(detail)
    body["schema"] = "studio.image-lora-info/1"
    if error is not None:
        body["fetch_error"] = error
    return body


async def _image_lora_admit(spec: Any) -> JSONResponse | None:
    """A spec naming a non-catalog LoRA needs that name in the live list."""
    if not isinstance(spec, dict) or not isinstance(spec.get("loras"), list):
        return None
    catalog = _image_lora_catalog()
    custom = [item.get("file") for item in spec["loras"] if isinstance(item, dict) and item.get("file") not in catalog]
    if not custom:
        return None
    try:
        names = await _image_live_loras()
    except Exception:  # noqa: BLE001
        return _edit_engine_unreachable()
    for name in custom:
        if type(name) is not str or name not in names:
            return _edit_error(422, "lora_unknown", "The image engine does not list this LoRA")
    return None
