from __future__ import annotations


# --- R4 model-agnostic edit (studio.image-edit-request/1) --------------------
# The Suite owns the operation, the common request, crop & stitch and the
# result; a model family declares capabilities and maps the request onto its
# own pinned ComfyUI workflow (Qwen-Image 2.1 PE v3 first).  N variants are N
# ledger jobs on the existing coordinator/publication seam; each publishes the
# edited CROP, and GET /image/edit/result stitches it back into the untouched
# original (patch = RGBA at source size, alpha = the soft mask).
def _load_edit_service_module() -> Any:
    module_path = Path(__file__).with_name("generation") / "image_edit_service.py"
    spec = importlib.util.spec_from_file_location("film_suite_image_edit_service", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("The edit service module is unavailable.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_edit_service_module = _load_edit_service_module()
_EDIT_STORES: dict[str, Any] = {}
_EDIT_RESULT_CACHE: "collections.OrderedDict[str, dict[str, Any]]" = collections.OrderedDict()
_EDIT_RESULT_CACHE_MAX = 16
_EDIT_ENGINE_PROBE: dict[str, Any] = {"until": 0.0, "error": None}
_EDIT_ENGINE_TTL_SECONDS = 120.0
_EDIT_MASK_B64_MAX = 48 * 1024 * 1024
_EDIT_ID_RE = re.compile(r"^imgedit_[0-9a-f]{32}$")


def _edit_contract_module() -> Any:
    from importlib import import_module
    return import_module(_image_contract_module().__package__ + ".image_edit_contract")


def _edit_adapter_import() -> Any:
    from importlib import import_module
    return import_module(_image_contract_module().__package__ + ".qwen_image_edit_adapter")


# The adapter module and class of each pinned edit target (one per family).
_EDIT_FAMILY_MODULES = {"qwen-image-2.1-edit-pe-v3": ("qwen_image_edit_adapter", "QwenImageEditAdapter"),
                        "flux3-klein-sd-edit-v1": ("flux3_klein_edit_adapter", "Flux3KleinEditAdapter"),
                        "flux2-klein-9b-kv-edit-v1": ("flux2_klein_edit_adapter", "Flux2KleinEditAdapter"),
                        "flux3-klein-box-edit-v1": ("flux3_klein_box_adapter", "Flux3KleinBoxEditAdapter"), "p-h3-image-lora-crop-edit-v1": ("p_h3_image_edit_adapter", "PH3ImageEditAdapter"), "krea2-edit-kroma-v1": ("krea2_edit_adapter", "Krea2EditAdapter"), "krea2-edit-turbo-v1": ("krea2_edit_adapter", "Krea2EditTurboAdapter")}  # w24-h3img: P-H3 crop edit ;   w24-krea: Krea 2 Edit


def _edit_family_adapter(target_id: str) -> tuple[Any, Any]:
    """(module, adapter class) of one edit target."""
    from importlib import import_module
    module_name, class_name = _EDIT_FAMILY_MODULES[target_id]
    module = import_module(_image_contract_module().__package__ + "." + module_name)
    return module, getattr(module, class_name)


def _edit_store() -> Any:
    home = _active_hermes_home()
    root = None if home is None else str(state_root(Path(home) / "plugin-data") / "image-edits")
    key = root or "<memory>"
    store = _EDIT_STORES.get(key)
    if store is None:
        store = _EDIT_STORES[key] = _edit_service_module.EditStore(root)
    return store


def _edit_error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def _edit_engine_unreachable() -> JSONResponse:
    return _edit_error(503, "comfyui_unreachable", "The image engine is not running. Start it and retry.")


async def _edit_engine_check(client: Any, target: Mapping[str, Any]) -> tuple[str, str] | None:
    """Classes and files of the edit workflow, from /object_info (cached; GET only).

    Returns None when everything is there, else (code, message).  Raises the
    client's error when the engine does not answer at all.
    """
    now = time.monotonic()
    target_id = target.get("target_id") or "qwen-image-2.1-edit-pe-v3"
    if now < _EDIT_ENGINE_PROBE["until"] and _EDIT_ENGINE_PROBE.get("target", "qwen-image-2.1-edit-pe-v3") == target_id:
        return _EDIT_ENGINE_PROBE["error"]
    adapter, adapter_type = _edit_family_adapter(target_id)
    problem: tuple[str, str] | None = None
    infos: dict[str, Any] = {}
    try:
        adapter_type(client).load_assets()
    except Exception:
        # Live once: the workflow was never installed, and every edit died as a vague engine error.
        problem = ("edit_workflow_missing", "The edit workflow is not installed. Reinstall Continuo")
    for name in adapter.REQUIRED_CLASSES if problem is None else ():
        body = await client.object_info(name)
        info = body.get(name) if isinstance(body, dict) else None
        if not isinstance(info, dict):
            problem = ("edit_node_missing", f"The image engine has no {name} node")
            break
        infos[name] = info

    def options(class_name: str, field: str) -> list[Any]:
        spec = (((infos.get(class_name) or {}).get("input") or {}).get("required") or {}).get(field)
        if isinstance(spec, list) and spec:
            if isinstance(spec[0], list):
                return spec[0]
            if len(spec) > 1 and isinstance(spec[1], dict):
                return list(spec[1].get("options") or [])
        return []

    if problem is None:
        files = target["files"]
        for class_name, field, wanted in (("UNETLoader", "unet_name", files["diffusion_model"]),
                                          ("CLIPLoader", "clip_name", files["text_encoder"]),
                                          ("CLIPLoader", "clip_name", files["prompt_enhancer"]),
                                          ("VAELoader", "vae_name", files["vae"]),
                                          ("LoraLoaderModelOnly", "lora_name", files["lora"])):
            if wanted is None:  # this family needs no file of this role (Klein: no LoRA)
                continue
            if wanted not in options(class_name, field):
                problem = ("edit_model_missing", f"The image engine has no {wanted}")
                break
    _EDIT_ENGINE_PROBE.update(until=now + (_EDIT_ENGINE_TTL_SECONDS if problem is None else 10.0), error=problem, target=target_id)
    return problem


class ImageEditSource(BaseModel):
    model_config = ConfigDict(extra="ignore")

    image_producer_job_id: str | None = Field(default=None, pattern=r"^imgjob_[0-9a-f]{32}$")
    revision_id: str | None = Field(default=None, pattern=r"^imgrev_[0-9a-f]{32}$")
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class ImageEditMask(BaseModel):
    model_config = ConfigDict(extra="ignore")

    png_b64: str = Field(min_length=8, max_length=_EDIT_MASK_B64_MAX)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class ImageEditRequest(BaseModel):
    """studio.image-edit-request/1; host keys the desktop may add (``profile`` ...) are ignored."""

    model_config = ConfigDict(extra="ignore")

    schema_: Literal["studio.image-edit-request/1"] = Field(alias="schema")
    owner_scope: str = Field(pattern=_OWNER_SCOPE_PATTERN, min_length=18, max_length=47)
    operation_id: str = Field(pattern=r"^imgop_[0-9a-f]{32}$", min_length=38, max_length=38)
    source: ImageEditSource
    mask: ImageEditMask | None = None  # Outpaint: none, the Suite makes the canvas margin the area
    base: ImageEditMask | None = None  # the picture the model renders from: the pixels below the layer, Relight's live light, or Outpaint's shown frame
    operation: str = Field(min_length=1, max_length=40)
    prompt: Any = None
    negative: Any = None
    references: Any = None
    strength: Any = None
    variants: Any = 1
    seed: Any = 0
    model: Any = None
    options: Any = None
    boxes: Any = None  # operation "box": [{id, intent, desc, bbox [x, y, w, h], target_bbox?}] in source pixels


def _edit_owned_job(jobs: Any, states: Any, job_id: str) -> dict[str, Any] | None:
    try:
        job = jobs.get_job(job_id)
    except Exception as exc:  # noqa: BLE001 - narrowed below
        if type(exc).__name__ != "ImageJobNotFoundError" and not isinstance(exc, KeyError):
            raise
        return None
    basis = (job.get("request_snapshot") or {}).get("context_basis") or {}
    try:
        states.require_owned(basis.get("context_id"))
    except Exception:  # noqa: BLE001 - not owned answers like unknown
        return None
    return job


def _edit_source_job_id(states: Any, jobs: Any, source: ImageEditSource) -> str | None:
    """The producer job behind the source: given directly, or the kept revision's picture."""
    if source.image_producer_job_id:
        return source.image_producer_job_id
    if not source.revision_id:
        return None
    for context in states.enumerate_contexts():
        for work in (context.get("state") or {}).get("works") or ():
            for revision in work.get("saved_revisions") or ():
                if revision.get("revision_id") == source.revision_id:
                    return _image_revision_summary(revision, jobs).get("image_producer_job_id")
    return None


_EDIT_FAILURE_TEXT = {
    "submit-reconciliation-required": "The image engine did not take the edit. Is ComfyUI running?", "submission_unknown": "The image engine did not take the edit. Is ComfyUI running?",
    "unreachable": "The image engine is not reachable. Is ComfyUI running?", "execution_failed": "The image engine stopped with an error on this edit",
    "job_lost": "The image engine lost this edit. Was it restarted?", "output_invalid": "The image engine returned no usable picture", "cancelled": "Stopped",
}


def _edit_failure_text(job: Mapping[str, Any]) -> str | None:
    """One readable English sentence for a failed/quarantined variant (codes stay in the record)."""
    if str(job.get("state") or "") not in ("failed", "quarantined", "cancelled"):
        return None
    # A stopped variant carries no reason code: its state is the reason.
    code = str(job.get("failure_reason") or job.get("quarantine_reason") or ("cancelled" if job.get("state") == "cancelled" else ""))
    return _EDIT_FAILURE_TEXT.get(code, "The image engine could not make this edit")


def _edit_variant_view(job: Mapping[str, Any], member: Mapping[str, Any]) -> dict[str, Any]:
    raw = str(job.get("state") or "")
    view = {"variant_index": member["variant_index"], "seed": member["seed"],
            "image_producer_job_id": member["image_producer_job_id"],
            "state": _IMAGE_JOB_STATE_VIEW.get(raw, "queued"), "raw_state": raw,
            "result": "available" if raw == "published" else ("pending" if raw not in ("failed", "cancelled", "quarantined") else "none")}
    for optional in ("failure_reason", "quarantine_reason"):
        if job.get(optional):
            view[optional] = job[optional]
    text = _edit_failure_text(job)
    if text:
        view["failure_text"] = text
    return view


def _edit_aggregate(views: list[dict[str, Any]]) -> str:
    states = {v["state"] for v in views}
    if states & {"queued", "running"}:
        return "running" if "running" in states else "queued"
    published = sum(1 for v in views if v["raw_state"] == "published")
    if published == len(views):
        return "done"
    if published:
        return "partial"
    return "failed"


def _edit_fingerprint(intent: Mapping[str, Any], source_job: str, mask_sha: str) -> str:
    body = {k: v for k, v in intent.items() if k != "warnings"}
    return hashlib.sha256(json.dumps({"intent": body, "source": source_job, "mask": mask_sha}, sort_keys=True).encode()).hexdigest()


@project_router.post("/image/edit")
async def image_edit_submit(request: ImageEditRequest) -> Any:
    """Start one model-agnostic edit (fill / replace / remove) on an owned picture.

    Returns at once with the edit id and one ledger job per variant.  The
    same ``operation_id`` with the same request is a replay (no second
    render); with a different request it is ``idempotency_conflict`` - the
    desktop clears the operation id after every answered submit.
    """
    unknown = _edit_error(404, "picture_unknown", "The picture is unknown")
    contract = None
    try:
        contract = _edit_contract_module()
        async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
            root = _image_producer_root(request.owner_scope)
            handoff = _image_operations_handoff(root, request.owner_scope)
            jobs = handoff.producer_jobs()
            states = handoff.image_states()
            _collector, reader = handoff.result_services()
        source_job_id = _edit_source_job_id(states, jobs, request.source)
        if source_job_id is None:
            return unknown
        source_job = _edit_owned_job(jobs, states, source_job_id)
        if source_job is None:
            return unknown
        if source_job.get("state") != "published":
            return _edit_error(409, "picture_not_ready", "The picture is not ready yet")
        raw = request.model_dump(by_alias=False, exclude={"schema_", "owner_scope", "operation_id", "source", "mask", "base"})
        try:
            intent = contract.normalize_request(raw)
        except contract.EditRequestError as exc:
            return _edit_error(422, exc.code, exc.message)
        outpaint = intent["operation"] == "outpaint"
        request = await _image_box_selection(request, intent, reader, source_job_id) if intent.get("boxes") else request  # image/box_1.py
        if outpaint != (request.mask is None):
            return _edit_error(422, "invalid_edit_request", "Outpaint takes no selection; every other edit needs one")
        mask_png = b""
        if not outpaint:
            try:
                mask_png = base64.b64decode(request.mask.png_b64, validate=True)
            except (ValueError, TypeError):
                return _edit_error(422, "invalid_mask", "The selection mask is not valid base64")
            if request.mask.sha256 and request.mask.sha256 != hashlib.sha256(mask_png).hexdigest():
                return _edit_error(422, "mask_hash_mismatch", "The selection mask does not match its fingerprint")
        mask_sha = hashlib.sha256(mask_png).hexdigest() if not outpaint else "outpaint:" + ",".join(map(str, intent["canvas"]))
        base_png = None
        if request.base is not None:  # the pixels below the layer (or Relight's live light) are the picture the model sees
            try:
                base_png = base64.b64decode(request.base.png_b64, validate=True)
            except (ValueError, TypeError):
                return _edit_error(422, "invalid_base", "The light base is not valid base64")
            if request.base.sha256 and request.base.sha256 != hashlib.sha256(base_png).hexdigest():
                return _edit_error(422, "invalid_base", "The base must match its fingerprint")
        edit_id = contract.edit_id_for(request.owner_scope, request.operation_id)
        fingerprint = _edit_fingerprint(intent, source_job_id, mask_sha + ("|base:" + hashlib.sha256(base_png).hexdigest() if base_png else ""))
        store = _edit_store()
        known = store.get(edit_id)
        if known is not None:
            if known.get("fingerprint") != fingerprint:
                return _edit_error(409, "idempotency_conflict",
                                   "This operation id was already used for a different edit. Use a new operation id for every edit.")
            return await _edit_status_body(known, jobs, states, handoff, refresh=False)
        source_png, artifact = await asyncio.to_thread(reader.read_published_job, source_job_id)
        if request.source.sha256 and request.source.sha256 != artifact["artifact_sha256"]:
            return _edit_error(409, "source_changed", "The picture changed since it was selected")
        target = {**contract.EDIT_TARGETS[intent["target_id"]], "target_id": intent["target_id"]}
        if base_png is not None:  # Outpaint: the shown frame (crop + layers, A-02), any size; else the picture's
            try:
                size = _edit_png_size(base_png)
                same = (size[0] >= 1 and size[1] >= 1) if outpaint else size == _edit_png_size(source_png)
            except Exception:  # noqa: BLE001 - any unreadable base is the same finite answer
                same = False
            if not same:
                return _edit_error(422, "invalid_base", "The base must be a PNG the size of the picture (Outpaint: of the shown frame)")
            source_png = base_png
        try:
            if outpaint:  # the picture on the larger canvas; the grey margin (and the seam band) is the area
                contract.outpaint_canvas_size(_edit_png_size(source_png), intent["canvas"])  # finite answer first
                made = await asyncio.to_thread(_edit_service_module.outpaint_canvas, source_png, intent["canvas"])
                source_png, mask_png = made["canvas_png"], made["mask_png"]
                mask_sha = hashlib.sha256(mask_png).hexdigest()
            facts = await asyncio.to_thread(_edit_service_module.prepare, source_png, mask_png, contract=contract,
                                            factor=intent["context_factor"], target=target, neutral=contract.neutral_for(intent),
                                            cover=contract.box_cover(intent), latent=contract.latent_mask_for(intent))  # p13-img
        except _edit_service_module.EditServiceError as exc:  # p13-img: a missing / failing LaMa engine has its own status
            return _edit_error(_edit_service_module.REFUSAL_STATUS.get(exc.code, 422), exc.code, exc.message)
        except contract.EditRequestError as exc:
            return _edit_error(422, exc.code, exc.message)
        client = _comfy_client_module.ComfyUIClient()
        try:
            problem = await _edit_engine_check(client, target)
        except Exception as exc:  # noqa: BLE001 - engine down is a finite answer
            if getattr(exc, "reason_code", None) == "unreachable" or isinstance(exc, OSError):
                return _edit_engine_unreachable()
            raise
        if problem is not None:
            return _edit_error(503, problem[0], problem[1])
        name = contract.upload_name(facts["crop_sha256"])
        try:
            await asyncio.to_thread(_edit_service_module.upload_crop, _comfy_client_module.urlopen, client.origin, name, facts["crop_png"])
        except _edit_service_module.EditServiceError as exc:
            return _edit_error(502, exc.code, exc.message)
        except OSError:
            return _edit_engine_unreachable()
        store.save_mask(mask_png)
        if base_png is not None:
            store.save_mask(base_png)  # content-addressed like the mask: the result is stitched over it
        basis_in = source_job["request_snapshot"]["context_basis"]
        claim = states.load_current_owner(basis_in["context_id"], basis_in["work_id"], basis_in["basis_kind"], basis_in.get("revision_id"))
        basis = {key: claim[key] for key in ("context_id", "work_id", "basis_kind", "revision_id", "draft_generation", "basis_content_sha256")}
        policy: dict[str, Any] = {"mode": "specified" if basis["revision_id"] is not None else "current",
                                  "owner_token": claim["owner_token"], "expected_revision": claim["expected_revision"]}
        if basis["revision_id"] is not None:
            policy.update(context_id=basis["context_id"], work_id=basis["work_id"], revision_id=basis["revision_id"])
        async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
            coordinator = handoff.producer_coordinator()
        members = []
        for index in range(intent["variants"]):
            edit = {
                "schema": contract.SNAPSHOT_EDIT_SCHEMA, "edit_id": edit_id, "operation": intent["operation"],
                "prompt": intent["prompt"], "negative": intent["negative"], "target": intent["target"],
                "instruction_mode": intent["instruction_mode"], "prompt_enhancer": intent["prompt_enhancer"],
                "color_match": intent["color_match"], "context_factor": intent["context_factor"],
                "references": intent["references"], "variants": intent["variants"], "variant_index": index,
                "source_sha256": artifact["artifact_sha256"], "source_size": facts["source_size"],
                "mask_sha256": mask_sha, "mask_bbox": facts["mask_bbox"], "crop_box": facts["crop_box"],
                "work_size": facts["work_size"], "crop_sha256": facts["crop_sha256"], "image": name,
                **contract.strength_fields(intent), **facts.get("snapshot_edit", {})}  # Strength <100 % + its curve; p13-img latent mask
            if outpaint:  # the snapshot's sizes are the picture's; the crop lies on the canvas
                edit.update(source_size=made["source_size"], canvas=intent["canvas"])
            edit.update({"boxes": intent["boxes"]} if intent.get("box_mode") == "native" else {})  # FLUX.3 Klein reads them
            seed = intent["seed"] + index
            snapshot = contract.build_edit_snapshot(edit, intent["target_id"], basis, policy, seed)
            key = contract.variant_key(request.operation_id, index)
            job = await coordinator.submit(key, snapshot, client_request_id=key)
            _image_preview_watch(job)
            _announce_image_jobs([(job["image_producer_job_id"], None, job.get("state"))], request.owner_scope)
            members.append({"variant_index": index, "seed": seed, "image_producer_job_id": job["image_producer_job_id"],
                            "instruction": snapshot["artist_input"]["prompt"]})
        record = {
            "schema": "studio.image-edit-record/1", "edit_id": edit_id, "owner_scope": request.owner_scope,
            "operation_id": request.operation_id, "fingerprint": fingerprint, "created_ms": _edit_service_module.now_ms(),
            "intent": intent, "target_id": intent["target_id"], "family": intent["family"], "variant": intent["variant"],
            "source": {"image_producer_job_id": source_job_id, "sha256": artifact["artifact_sha256"],
                       "width": facts["source_size"][0], "height": facts["source_size"][1]},
            **({"canvas": {"margins": intent["canvas"], "source_size": made["source_size"],
                           "band_px": made["band_px"]}} if outpaint else {}),
            "mask": {"sha256": mask_sha, "bbox": facts["mask_bbox"], "coverage_pct": facts["mask_coverage_pct"]},
            **({"base": {"sha256": hashlib.sha256(base_png).hexdigest()}} if base_png is not None else {}),
            "crop": {"box": facts["crop_box"], "work_size": facts["work_size"], "sha256": facts["crop_sha256"], "upload": name,
                     "neutral": facts["neutral"], **facts.get("record_crop", {})},  # p13-img: latent mask + LaMa pre-fill facts
            "variants": members,
        }
        store.put(record)
        return await _edit_status_body(record, jobs, states, handoff, refresh=False)
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        if contract is not None and isinstance(exc, contract.EditRequestError):
            return _edit_error(422, exc.code, exc.message)
        return _image_film_error_response(exc)


def _edit_png_size(data: bytes) -> tuple[int, int]:
    """Width and height of a published PNG from its IHDR (ledger pictures are PNG)."""
    if len(data) < 24 or not data.startswith(b"\x89PNG\r\n\x1a\n") or data[12:16] != b"IHDR":
        raise _edit_service_module.EditServiceError("invalid_picture", "The picture is not a PNG")
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


async def _edit_status_body(record: Mapping[str, Any], jobs: Any, states: Any, handoff: Any, *, refresh: bool) -> dict[str, Any]:
    """One edit's state: per-variant ledger state (reconciled + collected when ``refresh``)."""
    views = []
    coordinator = collector = None
    for member in record["variants"]:
        job = jobs.get_job(member["image_producer_job_id"])
        state_before = job.get("state")
        if refresh and job.get("state") not in _IMAGE_TERMINAL_STATES:
            if coordinator is None:
                async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
                    coordinator = handoff.producer_coordinator()
            reconciled = await coordinator.status(member["image_producer_job_id"])
            if isinstance(reconciled, dict) and reconciled.get("state"):
                job = reconciled
        if refresh and job.get("state") == "ready":
            if coordinator is None:
                async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
                    coordinator = handoff.producer_coordinator()
            if not isinstance(job.get("output_descriptor"), Mapping):
                reconciled = await coordinator.status(member["image_producer_job_id"])
                if isinstance(reconciled, dict) and reconciled.get("state"):
                    job = reconciled
            if collector is None:
                async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
                    collector, _reader = handoff.result_services()
            try:
                job = await collector.collect(member["image_producer_job_id"], job.get("output_descriptor"))
            except Exception:  # noqa: BLE001 - reported through the ledger state, never raised
                job = jobs.get_job(member["image_producer_job_id"])
        if job.get("state") in {"queued", "running", "submitting"}:
            _image_preview_watch(job)
        if refresh:
            _announce_image_jobs([(member["image_producer_job_id"], state_before, job.get("state"))], record["owner_scope"])
        views.append(_edit_variant_view(job, member))
    intent = record["intent"]
    return {
        "schema": "studio.image-edit-operation/1", "kind": "image_edit_status", "owner_scope": record["owner_scope"],
        "edit_id": record["edit_id"], "operation_id": record["operation_id"], "operation": intent["operation"],
        "state": _edit_aggregate(views), "variants": views,
        "model": {"family": record["family"], "variant": record["variant"], "target_id": record["target_id"]},
        "source": record["source"], "mask": record["mask"],
        **({"canvas": record["canvas"]} if record.get("canvas") else {}),
        "crop": {"x": record["crop"]["box"][0], "y": record["crop"]["box"][1], "w": record["crop"]["box"][2],
                 "h": record["crop"]["box"][3], "work_w": record["crop"]["work_size"][0], "work_h": record["crop"]["work_size"][1]},
        "instruction": record["variants"][0]["instruction"] if record["variants"] else None,
        "warnings": intent.get("warnings") or [],
        "preview": {"route": "/image/preview/live", "job_key": "job", "note": "poll it with each running variant's image_producer_job_id"},
    }


def _edit_query(request: Request) -> tuple[str, str] | None:
    params = request.query_params
    edit_id, scope = params.get("edit", ""), params.get("scope", _SCRATCH_SCOPE)
    if not _EDIT_ID_RE.match(edit_id) or not _OWNER_SCOPE_RE.match(scope):
        return None
    return edit_id, scope


def _edit_record_for(edit_id: str, scope: str) -> dict[str, Any] | None:
    record = _edit_store().get(edit_id)
    if record is None or record.get("owner_scope") != scope:
        return None
    return record


@project_router.get("/image/edit/status")
async def image_edit_status(request: Request) -> Any:
    """``GET /image/edit/status?edit=imgedit_…&scope=film_owner:…``; host query keys are ignored."""
    parsed = _edit_query(request)
    if parsed is None:
        return _edit_error(422, "invalid_edit_request", "Edit status request is invalid")
    edit_id, scope = parsed
    try:
        record = _edit_record_for(edit_id, scope)
        if record is None:
            return _edit_error(404, "edit_unknown", "The edit is unknown")
        async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
            root = _image_producer_root(scope)
            handoff = _image_operations_handoff(root, scope)
            jobs = handoff.producer_jobs()
            states = handoff.image_states()
        if any(_edit_owned_job(jobs, states, m["image_producer_job_id"]) is None for m in record["variants"]):
            return _edit_error(404, "edit_unknown", "The edit is unknown")
        return await _edit_status_body(record, jobs, states, handoff, refresh=True)
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_film_error_response(exc)


async def _edit_history_facts(prompt_id: Any) -> dict[str, Any]:
    """The prompt-enhancer output and the engine time of one variant (from ComfyUI history)."""
    if type(prompt_id) is not str:
        return {}
    adapter = _edit_adapter_import()
    try:
        history = await _comfy_client_module.ComfyUIClient().history(prompt_id)
    except Exception:  # noqa: BLE001 - history gone or engine down: facts unknown
        return {}
    entry = history.get(prompt_id) if isinstance(history, dict) else None
    return {"sent_prompt": adapter.sent_prompt_from_history(entry), "comfy_ms": adapter.execution_ms_from_history(entry)}


@project_router.get("/image/edit/result")
async def image_edit_result(request: Request) -> Any:
    """``GET /image/edit/result?edit=…&variant=N&scope=…[&composite=1]`` (studio.image-edit-result/1).

    The patch is an RGBA PNG at source size whose alpha is the soft mask:
    placed at (0, 0) over the locked original it changes only the selection.
    The original is never changed.
    """
    parsed = _edit_query(request)
    variant_raw = request.query_params.get("variant", "0")
    composite = request.query_params.get("composite") in ("1", "true")
    if parsed is None or not (variant_raw.isdigit() and len(variant_raw) == 1):
        return _edit_error(422, "invalid_edit_request", "Edit result request is invalid")
    edit_id, scope = parsed
    index = int(variant_raw)
    try:
        record = _edit_record_for(edit_id, scope)
        if record is None or index >= len(record["variants"]):
            return _edit_error(404, "edit_unknown", "The edit is unknown")
        member = record["variants"][index]
        async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
            root = _image_producer_root(scope)
            handoff = _image_operations_handoff(root, scope)
            jobs = handoff.producer_jobs()
            states = handoff.image_states()
            _collector, reader = handoff.result_services()
        job = _edit_owned_job(jobs, states, member["image_producer_job_id"])
        source_job = _edit_owned_job(jobs, states, record["source"]["image_producer_job_id"])
        if job is None or source_job is None:
            return _edit_error(404, "edit_unknown", "The edit is unknown")
        if job.get("state") != "published":
            return _edit_error(409, "result_not_ready", "This variant is not ready yet")
        cache_key = f"{member['image_producer_job_id']}|{int(composite)}"
        store = _edit_store()

        async def build() -> Any:  # IMG-221: once per variant, shared and kept (routes/image/edit_result_1.py)
            output_png, _artifact = await asyncio.to_thread(reader.read_published_job, member["image_producer_job_id"])
            source_png, source_artifact = await asyncio.to_thread(reader.read_published_job, record["source"]["image_producer_job_id"])
            if source_artifact["artifact_sha256"] != record["source"]["sha256"]:
                return _edit_error(409, "source_changed", "The original picture changed")
            store = _edit_store()
            mask_png = store.mask(record["mask"]["sha256"])
            if record.get("canvas") and mask_png is None:  # Outpaint: the area is rebuilt from the picture
                mask_png = b""
            if mask_png is None:
                return _edit_error(410, "edit_mask_gone", "The selection of this edit is no longer stored")
            if record.get("base"):  # Relight on the live light: the result lies over that base
                source_png = store.mask(record["base"]["sha256"])
                if source_png is None:
                    return _edit_error(410, "edit_mask_gone", "The light base of this edit is no longer stored")
            if record.get("canvas"):  # the same canvas the edit was made from (deterministic)
                made = await asyncio.to_thread(_edit_service_module.outpaint_canvas, source_png, record["canvas"]["margins"])
                source_png, mask_png = made["canvas_png"], made["mask_png"]
            stitched = await asyncio.to_thread(_edit_service_module.stitch, source_png, mask_png, output_png, record["crop"]["box"],
                                               color_match=bool(record["intent"].get("color_match")),
                                               neutral=record["crop"].get("neutral") or False, latent=record["crop"].get("latent_mask"), **_edit_contract_module().paste_for(record["intent"]))
            if "sent_prompt" not in member:
                facts = await _edit_history_facts((job.get("provider") or {}).get("prompt_id"))
                if facts:
                    member.update(facts)
                    store.put(record)
            width, height = record["source"]["width"], record["source"]["height"]  # Outpaint: the canvas
            x, y, w, h = record["crop"]["box"]
            body: dict[str, Any] = {
                "schema": "studio.image-edit-result/1", "owner_scope": scope, "edit_id": edit_id, "variant_index": index,
                "image_producer_job_id": member["image_producer_job_id"], "seed": member["seed"],
                "operation": record["intent"]["operation"],
                "model": {"family": record["family"], "variant": record["variant"], "target_id": record["target_id"]},
                "patch": {"mime": "image/png", "png_b64": base64.b64encode(stitched["patch_png"]).decode("ascii"),
                          "width": width, "height": height, "sha256": hashlib.sha256(stitched["patch_png"]).hexdigest(),
                          "alpha": "soft_mask"},
                "placement": {"x": 0, "y": 0, "w": width, "h": height},
                "crop": {"x": x, "y": y, "w": w, "h": h, "work_w": record["crop"]["work_size"][0], "work_h": record["crop"]["work_size"][1],
                         "model_output": stitched["model_output_size"]},
                "mask": record["mask"],
                "effective_prompt": {"instruction": member["instruction"], "sent_prompt": member.get("sent_prompt"),
                                     "prompt_enhancer": bool(record["intent"].get("prompt_enhancer"))},
                "timings": {"comfy_ms": member.get("comfy_ms"), "submitted_ms": record.get("created_ms")},
                "proof": {"unmasked_sha256_before": stitched["unmasked_sha256_before"],
                          "unmasked_sha256_after": stitched["unmasked_sha256_after"],
                          "unmasked_identical": stitched["unmasked_sha256_before"] == stitched["unmasked_sha256_after"],
                          "changed_outside_mask": stitched["changed_outside_mask"], "changed_pixels": stitched["changed_pixels"],
                          "changed_bbox": stitched["changed_bbox"], "grow_px": stitched["grow_px"], **({"latent_mask": record["crop"]["latent_mask"]} if record["crop"].get("latent_mask") else {}), **({"sky": stitched["sky"]} if stitched.get("sky") else {})},
                "source": record["source"], "warnings": record["intent"].get("warnings") or [],
            }
            if record.get("canvas"):  # Outpaint: the patch is the whole new picture; the original sits inside it
                left, top = record["canvas"]["margins"][:2]
                body["canvas"] = {"margins": record["canvas"]["margins"], "width": width, "height": height,
                                  "source": {"x": left, "y": top, "w": record["canvas"]["source_size"][0],
                                             "h": record["canvas"]["source_size"][1]}}
            if composite:
                body["composite"] = {"mime": "image/png", "png_b64": base64.b64encode(stitched["composite_png"]).decode("ascii")}
            return body
        return await _edit_result_shared(cache_key, build, store)
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_film_error_response(exc)


@project_router.get("/image/edit/capabilities")
async def image_edit_capabilities(request: Request) -> Any:
    """Per-family edit capabilities (studio.image-edit-capabilities/1); host query keys are ignored."""
    try:
        contract = _edit_contract_module()
        caps = contract.load_capabilities()
        caps["wired_operations"] = list(contract.OPERATIONS_WIRED)
        caps["request_schema"] = contract.REQUEST_SCHEMA
        caps["result_schema"] = contract.RESULT_SCHEMA
        return JSONResponse(caps, headers={"Cache-Control": "no-store"})
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_film_error_response(exc)


# --- composed pictures (Accept as version, Expand) ------------------------------------------------
_COMPOSE_B64_MAX = 96 * 1024 * 1024
_COMPOSE_WAIT_SECONDS = 20.0
_COMPOSE_DONE = frozenset({"published", "failed", "cancelled", "quarantined"})


def _compose_module() -> Any:
    from importlib import import_module
    return import_module(_image_contract_module().__package__ + ".image_compose")


class ImageComposeRequest(BaseModel):
    """studio.image-compose-request/1: a picture the Suite flattened becomes a picture of the ledger."""

    model_config = ConfigDict(extra="ignore")

    schema_: Literal["studio.image-compose-request/1"] = Field(alias="schema")
    owner_scope: str = Field(pattern=_OWNER_SCOPE_PATTERN, min_length=18, max_length=47)
    operation_id: str = Field(pattern=r"^imgop_[0-9a-f]{32}$", min_length=38, max_length=38)
    source: ImageEditSource
    kind: Literal["accept", "expand"]
    png_b64: str = Field(min_length=8, max_length=_COMPOSE_B64_MAX)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    recipe: dict[str, Any] = Field(default_factory=dict)


def _compose_same_pixels(sent: bytes, published: bytes) -> bool:
    """The engine re-encodes the PNG: compare decoded RGBA pixels, not bytes."""
    from PIL import Image  # type: ignore[import-not-found]
    with Image.open(io.BytesIO(sent)) as a, Image.open(io.BytesIO(published)) as b:
        if a.size != b.size:
            return False
        if a.convert("RGBA").tobytes() == b.convert("RGBA").tobytes():
            return True
        # A picture with transparency must come back with the same alpha, never flattened.
        return "A" not in a.getbands() and a.convert("RGB").tobytes() == b.convert("RGB").tobytes()


def _compose_revision_home(states: Any, revision_id: str) -> tuple[str, str] | None:
    """Context and work of a kept revision (Accept as version continues that work)."""
    for context in states.enumerate_contexts():
        for work in (context.get("state") or {}).get("works") or ():
            if any(r.get("revision_id") == revision_id for r in work.get("saved_revisions") or ()):
                return context.get("context_id"), work.get("work_id")
    return None


def _compose_has_alpha(data: bytes) -> bool:
    """True when the PNG carries transparency (Remove background): the alpha passthrough keeps it."""
    from PIL import Image  # type: ignore[import-not-found]
    with Image.open(io.BytesIO(data)) as image:
        if image.mode not in {"RGBA", "LA"} and not (image.mode == "P" and "transparency" in image.info):
            return False
        return image.convert("RGBA").getchannel("A").getextrema()[0] < 255


def _compose_check_png(data: bytes) -> tuple[int, int]:
    """Decode the whole PNG once (a truncated or hostile file never reaches the engine)."""
    from PIL import Image  # type: ignore[import-not-found]
    width, height = _compose_module().png_size(data)
    if width * height > _edit_service_module.MAX_SOURCE_PIXELS:
        raise _edit_service_module.EditServiceError("picture_too_large", "The picture is too large")
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "PNG" or image.mode not in {"RGB", "RGBA", "L", "LA", "P"}:
            raise _edit_service_module.EditServiceError("invalid_picture", "The picture is not a PNG")
        image.load()
    return width, height


@project_router.post("/image/compose")
async def image_compose_submit(request: ImageComposeRequest) -> Any:
    """Publish a picture the Suite composed as a picture of the ledger (studio.image-compose-request/1).

    The pixels come from the caller (Original + accepted layers, or an expanded
    canvas); the source picture names whose work it continues.  The engine only
    hands the PNG back unchanged, so the result is a normal published run:
    picture, keep, edit and mask routes all take it by its job id.
    """
    unknown = _edit_error(404, "picture_unknown", "The picture is unknown")
    try:
        compose = _compose_module()
        try:
            data = base64.b64decode(request.png_b64, validate=True)
        except (ValueError, TypeError):
            return _edit_error(422, "invalid_picture", "The picture is not valid base64")
        picture_sha = hashlib.sha256(data).hexdigest()
        if request.sha256 and request.sha256 != picture_sha:
            return _edit_error(422, "picture_hash_mismatch", "The picture does not match its fingerprint")
        try:
            width, height = await asyncio.to_thread(_compose_check_png, data)
            alpha = await asyncio.to_thread(_compose_has_alpha, data)
        except (compose.ComposeError, _edit_service_module.EditServiceError) as exc:
            return _edit_error(422, exc.code, exc.message)
        except Exception:  # noqa: BLE001 - any decoder failure is a bad picture
            return _edit_error(422, "invalid_picture", "The picture is not a PNG")
        async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
            root = _image_producer_root(request.owner_scope)
            handoff = _image_operations_handoff(root, request.owner_scope)
            jobs = handoff.producer_jobs()
            states = handoff.image_states()
        source_job_id = _edit_source_job_id(states, jobs, request.source)
        source_job = _edit_owned_job(jobs, states, source_job_id) if source_job_id else None
        if source_job is None:
            return unknown
        if source_job.get("state") != "published":
            return _edit_error(409, "picture_not_ready", "The picture is not ready yet")
        # From a kept revision the composed picture continues that work (keeping it makes the next
        # revision); otherwise it builds on what the source run built on.
        home = _compose_revision_home(states, request.source.revision_id) if request.source.revision_id else None
        if home is not None:
            claim = states.load_current_owner(home[0], home[1], "saved_revision", request.source.revision_id)
        else:
            basis_in = source_job["request_snapshot"]["context_basis"]
            claim = states.load_current_owner(basis_in["context_id"], basis_in["work_id"], basis_in["basis_kind"], basis_in.get("revision_id"))
        basis = {key: claim[key] for key in ("context_id", "work_id", "basis_kind", "revision_id", "draft_generation", "basis_content_sha256")}
        policy: dict[str, Any] = {"mode": "specified" if basis["revision_id"] is not None else "current",
                                  "owner_token": claim["owner_token"], "expected_revision": claim["expected_revision"]}
        if basis["revision_id"] is not None:
            policy.update(context_id=basis["context_id"], work_id=basis["work_id"], revision_id=basis["revision_id"])
        name = compose.upload_name(picture_sha)
        try:
            snapshot = compose.build_snapshot({"schema": compose.SCHEMA, "kind": request.kind, "source_job_id": source_job_id,
                                               "picture_sha256": picture_sha, "image": name, "recipe": request.recipe,
                                               **({"alpha": True} if alpha else {})},
                                              width, height, basis, policy)
        except compose.ComposeError as exc:
            return _edit_error(422, exc.code, "The composed picture request is invalid")
        client = _comfy_client_module.ComfyUIClient()
        try:
            await asyncio.to_thread(_edit_service_module.upload_crop, _comfy_client_module.urlopen, client.origin, name, data)
        except _edit_service_module.EditServiceError as exc:
            return _edit_error(502, exc.code, exc.message)
        except OSError:
            return _edit_engine_unreachable()
        async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
            coordinator = handoff.producer_coordinator()
            collector, reader = handoff.result_services()
        job = await coordinator.submit(request.operation_id, snapshot, client_request_id=request.operation_id)
        job_id = job["image_producer_job_id"]
        _announce_image_jobs([(job_id, None, job.get("state"))], request.owner_scope)
        # A passthrough takes well under a second: publish it here so the caller can keep it at once.
        deadline = asyncio.get_running_loop().time() + _COMPOSE_WAIT_SECONDS
        while job.get("state") not in _COMPOSE_DONE and asyncio.get_running_loop().time() < deadline:
            before = job.get("state")
            reconciled = await coordinator.status(job_id)
            if isinstance(reconciled, dict) and reconciled.get("state"):
                job = reconciled
            if job.get("state") == "ready":
                try:
                    job = await collector.collect(job_id, job.get("output_descriptor"))
                except Exception:  # noqa: BLE001 - reported through the ledger state
                    job = jobs.get_job(job_id)
            if job.get("state") != before:
                _announce_image_jobs([(job_id, before, job.get("state"))], request.owner_scope)
            if job.get("state") not in _COMPOSE_DONE:
                await asyncio.sleep(0.25)
        state = job.get("state")
        error = _edit_failure_text(job) if state in {"failed", "quarantined"} else None
        if state == "published":
            published, _artifact = await asyncio.to_thread(reader.read_published_job, job_id)
            if not await asyncio.to_thread(_compose_same_pixels, data, published):
                return _edit_error(502, "compose_mismatch", "The image engine changed the picture")
        return {"schema": "studio.image-compose-operation/1", "kind": "image_compose_status", "owner_scope": request.owner_scope,
                "operation_id": request.operation_id, "image_producer_job_id": job_id, "state": state,
                "source": {"image_producer_job_id": source_job_id}, "compose_kind": request.kind,
                "width": width, "height": height, "picture_sha256": picture_sha,
                "error": error}
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_film_error_response(exc)
