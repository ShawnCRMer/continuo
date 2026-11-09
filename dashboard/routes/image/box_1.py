from __future__ import annotations


# --- Image Box (studio.image-box-find/1) ---------------------------------------
# Boxes with an intent go through POST /image/edit (operation "box"); this area holds
# the box selection hook of that route, Find (a local detector proposes boxes for a
# text query) and Final prompt (what a box edit would send, without running it).
def _load_box_module(name: str, file: str) -> Any:
    module_path = Path(__file__).with_name("generation") / file
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("The image box module is unavailable.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_box_find_module = _load_box_module("film_suite_image_box_find", "image_box_find.py")
_IMAGE_BOX_FIND = _box_find_module.FindService(
    lambda: _comfy_client_module.ComfyUIClient(),
    lambda origin, name, data: _edit_service_module.upload_crop(_comfy_client_module.urlopen, origin, name, data),
    lambda label, count: _edit_contract_module().box_module().detection_ids(label, count))
_BOX_WAIT_MS_MAX = 20_000


async def _image_box_selection(request: Any, intent: Mapping[str, Any], reader: Any, source_job_id: str) -> Any:
    """A box edit's selection: the boxes as a mask (Box -> mask) or the area pasted back (FLUX.3 Klein).

    Limit (board E19, lane w4-image): an optional ``mask`` is a HARD limit.  The area the edit may change is the
    boxes' area AND the limit (per pixel the smaller value), so every pixel outside the limit stays the input's."""
    contract = _edit_contract_module()
    source_png, artifact = await asyncio.to_thread(reader.read_published_job, source_job_id)
    size = _edit_png_size(source_png)
    limit = _image_box_limit(request.mask, size, contract) if request.mask is not None else None
    boxes = contract.box_module()
    try:
        boxes.normalize_boxes(intent["boxes"], size)
        if intent["box_mode"] == "native" and any(b["intent"] == "move" for b in intent["boxes"]):
            moved = await _image_box_move(request.model_copy(update={"mask": None}), intent, boxes, source_png, str(artifact["artifact_sha256"]), size)
            return _image_box_limited(moved, limit, contract)
        made = await asyncio.to_thread(boxes.box_mask if intent["box_mode"] == "mask" else boxes.region_mask, intent["boxes"], size)
    except boxes.BoxError as exc:
        raise contract.EditRequestError(exc.code, exc.message) from exc
    return _image_box_limited(request.model_copy(update={"mask": ImageEditMask(png_b64=base64.b64encode(made["png"]).decode("ascii"))}), limit, contract)


def _image_box_limit(mask: Any, size: tuple[int, int], contract: Any) -> Any:
    """The Limit mask of a box edit (8-bit, picture size, white = may change); its fingerprint is checked here."""
    try:
        data = base64.b64decode(mask.png_b64, validate=True)
    except (ValueError, TypeError) as exc:
        raise contract.EditRequestError("invalid_mask", "The limit mask is not valid base64") from exc
    if mask.sha256 and mask.sha256 != hashlib.sha256(data).hexdigest():
        raise contract.EditRequestError("mask_hash_mismatch", "The limit mask does not match its fingerprint")
    try:
        limit = _edit_service_module.decode_mask(data, size)
    except _edit_service_module.EditServiceError as exc:
        raise contract.EditRequestError(exc.code, exc.message.replace("selection mask", "limit mask")) from exc
    if not (limit > 0).any():
        raise contract.EditRequestError("empty_mask", "The limit is empty")
    return limit


def _image_box_limited(request: Any, limit: Any, contract: Any) -> Any:
    """The request with its box area cut to the Limit (pixel-wise minimum); unchanged without a limit."""
    if limit is None:
        return request
    import numpy as np
    from PIL import Image
    area = np.asarray(Image.open(io.BytesIO(base64.b64decode(request.mask.png_b64))).convert("L"), dtype=np.uint8)
    joined = np.minimum(area, limit)
    if not (joined > 0).any():
        raise contract.EditRequestError("empty_mask", "The limit leaves nothing of the boxes to change")
    out = io.BytesIO()
    Image.fromarray(joined, mode="L").save(out, format="PNG")
    return request.model_copy(update={"mask": ImageEditMask(png_b64=base64.b64encode(out.getvalue()).decode("ascii"))})


async def _image_box_move(request: Any, intent: dict[str, Any], boxes: Any, source_png: bytes, sha: str, size: tuple[int, int]) -> Any:
    """Move carries the object (wave 4, owner decision): (a) the object is the single SAM3 instance that best
    matches its box (else the box, feathered); (b) a real Remove pass of the object runs as its own FLUX.3
    Klein job on the original picture and is pasted back only over the object grown by two rings; (c) the
    object's own pixels are pasted onto that result at the target; (d) the edit itself is the ring pass:
    the model sees that picture (the request's base) and only a narrow ring around the carried object
    (plus a ground strip with a contact shadow under a grounded object) takes its output (the mask).
    The intent's boxes become the ring pass's boxes; ``intent["move"]`` records the facts.
    Wave 5: other SAM3 instances next to the object (occluders) are kept by the Remove pass; the part they hid
    is completed by the ring pass (a Change box, nothing carried); ``intent["move"][i]["generated_mask_png_b64"]``
    marks the pixels the model made (hidden part and ground) for the layer."""
    contract = _edit_contract_module()
    if request.base is not None:
        raise contract.EditRequestError("invalid_edit_request", "A Move takes no base picture")
    client = _comfy_client_module.ComfyUIClient()
    upload = lambda origin, name, data: _edit_service_module.upload_crop(_comfy_client_module.urlopen, origin, name, data)  # noqa: E731
    masks, occluders = {}, {}
    for box in intent["boxes"]:
        if box["intent"] == "move":
            masks[box["id"]] = await _box_find_module.segment_in_box(client, upload, source_png, sha, box["bbox"], box["desc"])
            if masks[box["id"]] is not None:
                occluders[box["id"]] = await _box_find_module.occluders_near(client, upload, source_png, sha, boxes.move_occluder_box(box["bbox"], size), masks[box["id"]])
    removed_png, removal = await _image_box_move_remove(client, upload, intent, boxes, source_png, sha, masks, occluders)
    carry = await asyncio.to_thread(boxes.move_carry, source_png, intent["boxes"], masks, removed_png, occluders)
    completed = {}
    if carry.get("completion_png"):  # wave 6 (owner decision c): hidden part + ground, FLUX.2 Klein's Add path
        base_png, completed = await _image_box_move_complete(client, upload, intent, boxes, carry)
        carry = {**carry, "base_png": base_png}
    mask_png = carry["mask_png"]
    others = [b for b in intent["boxes"] if b["intent"] not in ("move", "keep")]
    if others:  # Add / Change / Remove boxes next to the Move: their own paste-back area, never over a carried core
        import numpy as np
        from PIL import Image
        region = await asyncio.to_thread(boxes.region_mask, others, size)
        core = np.asarray(Image.open(io.BytesIO(carry["core_png"])).convert("L")) > 0
        joined = np.maximum(np.asarray(Image.open(io.BytesIO(region["png"])).convert("L")), np.asarray(Image.open(io.BytesIO(mask_png)).convert("L")))
        out = io.BytesIO()
        Image.fromarray(np.where(core, 0, joined).astype(np.uint8), mode="L").save(out, format="PNG")
        mask_png = out.getvalue()
    intent["boxes"] = carry["model_boxes"]
    intent["move"] = [{**{k: m[k] for k in ("id", "landing", "scale", "mask_source", "ring_px", "grounded", "ground", "occluder_px", "hidden_px", "hidden", "generated_px")},
                       "generated_mask_png_b64": base64.b64encode(m["generated_png"]).decode("ascii"), "remove": removal,
                       **({"complete": completed} if completed else {})}
                      for m in carry["moves"]]
    b64 = lambda data: base64.b64encode(data).decode("ascii")  # noqa: E731
    return request.model_copy(update={"mask": ImageEditMask(png_b64=b64(mask_png)), "base": ImageEditMask(png_b64=b64(carry["base_png"]))})


async def _image_box_move_remove(client: Any, upload: Any, intent: Mapping[str, Any], boxes: Any, source_png: bytes, sha: str,
                                 masks: Mapping[str, Any], occluders: Mapping[str, Any] | None = None) -> tuple[bytes, dict[str, Any]]:
    """The Move's Remove pass: the pinned FLUX.3 Klein box graph on the ORIGINAL picture's crop, Remove at
    the source box (Keep boxes kept), its output pasted back seam-free (``move_blend``) only over the
    object grown by two rings.  Returns the removed picture (PNG, picture size) and facts.  Queued behind other jobs,
    never cancelled; a failed or late pass is a finite refusal (nothing is carried onto a smear)."""
    contract = _edit_contract_module()
    plan = await asyncio.to_thread(boxes.move_remove_plan, source_png, intent["boxes"], masks, occluders)
    target = {**contract.EDIT_TARGETS[intent["target_id"]], "target_id": intent["target_id"]}
    try:
        problem = await _edit_engine_check(client, target)
    except Exception as exc:  # noqa: BLE001 - engine down is a finite answer
        if getattr(exc, "reason_code", None) == "unreachable" or isinstance(exc, OSError):
            raise contract.EditRequestError("comfyui_unreachable", "ComfyUI is not running.") from exc
        raise
    if problem is not None:
        raise contract.EditRequestError(problem[0], problem[1])
    prefilled = await asyncio.to_thread(boxes.move_prefill, source_png, plan["mask_png"], plan.get("keep_out_png"))  # the model never sees the object
    facts = await asyncio.to_thread(_edit_service_module.prepare, prefilled, plan["mask_png"], contract=contract,
                                    factor=intent["context_factor"], target=target, neutral=False, cover=boxes.cover(plan["boxes"]))
    instruction = boxes.compile_prompt("", plan["boxes"], facts["crop_box"])["prompt"]
    module, adapter_type = _edit_family_adapter(intent["target_id"])
    _manifest, graph = adapter_type(client).load_assets()
    graph = json.loads(json.dumps(graph))
    name = contract.upload_name(facts["crop_sha256"])
    for key, value in (("image", name), ("instruction", instruction), ("seed", int(intent["seed"]))):
        module.set_pointer(graph, module.MUTABLE_POINTERS[key], value)
    key = _box_find_module.remove_cache_key(sha, plan["mask_png"], plan["boxes"], intent["seed"], intent["target_id"])
    output = await _box_find_module.remove_pass(client, upload, key, graph, facts["crop_png"], name, module.OUTPUT_NODE, intent["target_id"])
    if output is None:
        raise contract.EditRequestError("box_move_remove_failed", "The Move could not clear the object's old place")
    stitched = await asyncio.to_thread(boxes.move_blend, source_png, plan["mask_png"], output, facts["crop_box"])
    # wave 6: no low-frequency ghost in a plain area (the model's drift from the prefill taken out where the surroundings are smooth)
    clean = await asyncio.to_thread(boxes.move_unghost, source_png, plan["mask_png"], stitched["composite_png"], plan.get("keep_out_png"))
    return clean["composite_png"], {"crop": facts["crop_box"], "instruction": instruction, "seed": int(intent["seed"]), "unghost_px": clean["unghost_px"],
                                       "changed_pixels": stitched["changed_pixels"], "changed_outside_mask": stitched["changed_outside_mask"],
                                       "protected_px": sum(m.get("protected_px", 0) for m in plan["moves"])}


async def _image_box_move_complete(client: Any, upload: Any, intent: Mapping[str, Any], boxes: Any,
                                   carry: Mapping[str, Any]) -> tuple[bytes, dict[str, Any]]:
    """The Move's completion job (wave 6, owner decision c): the part an occluder hid and the ground under the foot,
    generated by FLUX.2 Klein's Add path (the local flux2-klein-9b-kv-edit-v1 workflow) on the carried picture,
    cropped around the target box with MOVE_COMPLETE_CONTEXT (1.6x) context, the output pasted back seam-free
    (``move_blend``) only inside the completion mask (0 on the carried core).  The prompt comes from the object's
    name.  Queued behind other jobs, never cancelled; a failed or late job is a finite refusal."""
    contract = _edit_contract_module()
    target_id = boxes.MOVE_COMPLETE_TARGET_ID
    target = {**contract.EDIT_TARGETS[target_id], "target_id": target_id}
    try:
        problem = await _edit_engine_check(client, target)
    except Exception as exc:  # noqa: BLE001 - engine down is a finite answer
        if getattr(exc, "reason_code", None) == "unreachable" or isinstance(exc, OSError):
            raise contract.EditRequestError("comfyui_unreachable", "ComfyUI is not running.") from exc
        raise
    if problem is not None:
        raise contract.EditRequestError(problem[0], problem[1])
    base_png, mask_png = carry["base_png"], carry["completion_png"]
    jobs = list(carry.get("completion") or [])
    cover = [min(j["target_bbox"][0] for j in jobs), min(j["target_bbox"][1] for j in jobs),
             max(j["target_bbox"][0] + j["target_bbox"][2] for j in jobs), max(j["target_bbox"][1] + j["target_bbox"][3] for j in jobs)]
    facts = await asyncio.to_thread(_edit_service_module.prepare, base_png, mask_png, contract=contract, factor=boxes.MOVE_COMPLETE_CONTEXT,
                                    target=target, neutral=False, cover=cover)
    prompt = " and ".join(dict.fromkeys(j["prompt"] for j in jobs))
    instruction = contract.instruction({"operation": "fill", "prompt": prompt}, contract.where_phrase(facts["mask_bbox"], facts["crop_box"]), target_id)
    module, adapter_type = _edit_family_adapter(target_id)
    _manifest, graph = adapter_type(client).load_assets()
    graph = json.loads(json.dumps(graph))
    name = contract.upload_name(facts["crop_sha256"])
    for key, value in (("image", name), ("instruction", instruction), ("seed", int(intent["seed"]))):
        module.set_pointer(graph, module.MUTABLE_POINTERS[key], value)
    key = _box_find_module.remove_cache_key(hashlib.sha256(base_png).hexdigest(), mask_png, [instruction], intent["seed"], target_id)
    output = await _box_find_module.remove_pass(client, upload, key, graph, facts["crop_png"], name, module.OUTPUT_NODE, target_id,
                                                kind="box_move_complete")
    if output is None:
        raise contract.EditRequestError("box_move_complete_failed", "The Move could not complete the object and its ground")
    stitched = await asyncio.to_thread(boxes.move_blend, base_png, mask_png, output, facts["crop_box"])
    return stitched["composite_png"], {"target_id": target_id, "crop": facts["crop_box"], "instruction": instruction, "seed": int(intent["seed"]),
                                       "changed_pixels": stitched["changed_pixels"], "changed_outside_mask": stitched["changed_outside_mask"]}


class ImageBoxFindRequest(BaseModel):
    """Find: boxes for a text query on an owned, published picture; host keys are ignored."""

    model_config = ConfigDict(extra="ignore")

    schema_: Literal["studio.image-box-find/1"] = Field(alias="schema")
    owner_scope: str = Field(pattern=_OWNER_SCOPE_PATTERN, min_length=18, max_length=47)
    source: ImageEditSource
    query: str = Field(min_length=1, max_length=120)
    threshold: float = 0.4
    max_boxes: int = 8
    wait_ms: int = Field(default=0, ge=0, le=_BOX_WAIT_MS_MAX)


@project_router.post("/image/box/find")
async def image_box_find(request: ImageBoxFindRequest) -> Any:
    """Run the local detector (SAM3 text) on a picture: boxes ``{id, label, score, bbox [x, y, w, h]}``.

    The same picture + query + threshold is one job (a repeat answers the running state or the
    cached boxes); ``wait_ms`` long-polls up to 20 s.  A foreign picture answers like an unknown one.
    """
    unknown = _edit_error(404, "picture_unknown", "The picture is unknown")
    try:
        async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
            root = _image_producer_root(request.owner_scope)
            handoff = _image_operations_handoff(root, request.owner_scope)
            jobs = handoff.producer_jobs()
            states = handoff.image_states()
            _collector, reader = handoff.result_services()
        job_id = _edit_source_job_id(states, jobs, request.source)
        job = _edit_owned_job(jobs, states, job_id) if job_id else None
        if job is None:
            return unknown
        if job.get("state") != "published":
            return _edit_error(409, "picture_not_ready", "The picture is not ready yet")
        try:
            spec = _box_find_module.normalize(request.model_dump(include={"query", "threshold", "max_boxes"}))
        except _box_find_module.FindError as exc:
            return _edit_error(422, exc.code, exc.message)
        data, artifact = await asyncio.to_thread(reader.read_published_job, job_id)
        width, height, sha = int(artifact["width"]), int(artifact["height"]), str(artifact["artifact_sha256"])
        key = _box_find_module.cache_key(sha, spec)
        state = _IMAGE_BOX_FIND.start(key, spec, data, sha, (width, height))
        if state.get("state") == "running" and request.wait_ms:
            state = await _IMAGE_BOX_FIND.wait(key, request.wait_ms) or state
        return {"schema": _box_find_module.SCHEMA, "owner_scope": request.owner_scope, "image_producer_job_id": job_id,
                "picture": {"sha256": sha, "width": width, "height": height}, "request": spec, **state}
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_film_error_response(exc)


class ImageBoxPromptRequest(BaseModel):
    """Final prompt: what a box edit would send, for a picture size; nothing runs."""

    model_config = ConfigDict(extra="ignore")

    schema_: Literal["studio.image-box-prompt/1"] = Field(alias="schema")
    width: int = Field(ge=1, le=16384)
    height: int = Field(ge=1, le=16384)
    prompt: str = Field(default="", max_length=2000)
    boxes: list[dict[str, Any]] = Field(min_length=1, max_length=16)
    model: dict[str, Any] | None = None
    context_factor: float = Field(default=1.2, ge=1.0, le=4.0)


@project_router.post("/image/box/prompt")
async def image_box_prompt(request: ImageBoxPromptRequest) -> Any:
    """The compiled box edit without running it: for FLUX.3 Klein the final prompt (caption, box
    sentences, BFL list on the crop) and the crop; for a mask model the operation and the mask facts.
    """
    contract = _edit_contract_module()
    boxes = contract.box_module()
    try:
        intent = contract.normalize_request({"operation": "box", "prompt": request.prompt, "boxes": request.boxes,
                                             "model": request.model or {"family": "flux3-klein"},
                                             "options": {"context_factor": request.context_factor}})
        size = (request.width, request.height)
        normalized = boxes.normalize_boxes(intent["boxes"], size)
        made = (boxes.box_mask if intent["box_mode"] == "mask" else boxes.region_mask)(normalized, size)
        mask = _edit_service_module.decode_mask(made["png"], size)
        bbox = _edit_service_module.mask_bbox(mask)
        cover = boxes.cover(normalized)
        target = contract.EDIT_TARGETS[intent["target_id"]]
        around = (min(bbox[0], cover[0]), min(bbox[1], cover[1]), max(bbox[2], cover[2]), max(bbox[3], cover[3]))
        crop = contract.crop_box(around, size, factor=intent["context_factor"], min_side=contract.MIN_CROP_SIDE, multiple=int(target["grid"]))
        body: dict[str, Any] = {"schema": "studio.image-box-prompt/1", "box_mode": intent["box_mode"], "operation": intent["operation"],
                                "model": {"family": intent["family"], "variant": intent["variant"], "target_id": intent["target_id"]},
                                "boxes": normalized, "crop": crop}
        if intent["box_mode"] == "native":
            compiled = boxes.compile_prompt(intent["prompt"], normalized, crop)
            body.update(final_prompt=compiled["prompt"], sentences=compiled["sentences"], rows=compiled["rows"])
        else:
            body.update(prompt=intent["prompt"], instruction=contract.instruction({**intent, "target": ""},
                        contract.where_phrase(bbox, crop), intent["target_id"]),
                        mask={"bbox": made["bbox"], "coverage_pct": made["coverage_pct"], "feather_px": made["feather_px"]})
        return body
    except contract.EditRequestError as exc:
        return _edit_error(422, exc.code, exc.message)
    except boxes.BoxError as exc:
        return _edit_error(422, exc.code, exc.message)
    except Exception as exc:  # noqa: BLE001 - finite plugin boundary
        return _image_film_error_response(exc)
