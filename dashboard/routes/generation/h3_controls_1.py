from __future__ import annotations


# H3 Video inspector controls (lane w2-h3ctl): sampling (turbo, recipe, sigma shift, engine), user
# LoRAs and performance (attention, sparse, cache) on the pinned Frames / References graphs, plus
# Generate x N.  The pure rules and the structural compile live in generation/h3_controls.py; here:
# the live admission before the ledger (only classes the engine lists, only listed MiniMax H3 LoRAs),
# the fan-out of one count request into N takes and the read-only descriptor route.
_h3k_module = _load_h3_conditioning_module("h3_controls.py", "film_suite_h3_controls")


class H3ControlsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_id: str | None = None


async def _h3k_object_info(client: Any, classes: Any) -> dict[str, Any]:
    """One live /object_info read per class; a class the engine lacks is simply absent."""
    info: dict[str, Any] = {}
    for class_type in sorted(set(classes)):
        try:
            body = await client.object_info(class_type)
        except _comfy_client_module.ComfyClientError as exc:
            if getattr(exc, "status", None) == 404 or "404" in str(exc):
                continue
            raise
        if isinstance(body, Mapping) and class_type in body:
            info[class_type] = body[class_type]
    return info


def _h3k_lora_family(name: str) -> str:
    """MiniMax H3 / another family / '' from the file's metadata, else its file name."""
    base = ""
    try:
        row = _image_lora_row(_image_lora_library(), _image_lora_root(), name, {})
        base = (row or {}).get("base_model") or ""
    except Exception:  # noqa: BLE001 - an unreadable header leaves the name rule
        base = ""
    return _h3k_module.lora_family(base, name, _lora_library_module.base_family)


async def _h3k_admit(plan: Mapping[str, Any]) -> None:
    """Refuse controls the live engine cannot run, before the ledger (raises H3ControlError)."""
    controls = plan.get("controls")
    if controls is None:
        return
    needs = _h3k_module.needs(controls)
    client = _comfy_client_module.ComfyUIClient()
    try:
        info = await _h3k_object_info(client, [need[0] for need in needs])
        live = await _image_live_loras() if controls["loras"] else None
    except Exception as exc:  # noqa: BLE001 - the engine is down: nothing can be checked
        raise _h3k_module.H3ControlError("The engine is unreachable; the controls cannot be checked", "unreachable") from exc
    for need in needs:
        problem = _h3k_module.class_problem(need, info)
        if problem is not None:
            raise _h3k_module.H3ControlError(f"Unavailable: {problem}", "missing_node")
    if controls["loras"]:
        await asyncio.to_thread(_h3k_module.admit_loras, controls["loras"], live, _h3k_lora_family)


def _h3k_fans_out(request: SubmitRequest) -> bool:
    """A count request (count 2-8, no take_index yet) on a pinned H3 conditioning target."""
    return (request.provider_id == "comfyui-local" and request.target_id in _h3k_module.TARGET_IDS
            and isinstance(request.args, Mapping) and "count" in request.args and "take_index" not in request.args)


def _h3k_take_basename(basename: str | None, index: int) -> str | None:
    if basename is None or index == 0:
        return basename
    stem, dot, suffix = basename.rpartition(".")
    return f"{stem}-take{index + 1}.{suffix}" if dot else f"{basename}-take{index + 1}"


# Batches whose later takes still wait (the ledger runs one local job at a time): batch id -> request.
_H3K_PENDING: "collections.OrderedDict[str, SubmitRequest]" = collections.OrderedDict()
_H3K_PENDING_MAX = 64


async def _h3k_submit_batch(request: SubmitRequest) -> dict[str, Any]:
    """Generate x N: one request, N takes (seed + i), each its own durable request.

    Take i is submitted as ``client_request_id:take-i`` (take 0 keeps the id) with ``take_index``.
    The ledger runs one local job at a time, so take i+1 is submitted when take i has ended: a poll
    that sees a take of a waiting batch end advances it, and so does a retry of the same request.
    A retry replays every submitted take's receipt and never makes a take twice.  Count 1 is
    exactly a request without ``count``.
    """
    try:
        count = request.args["count"]
        if isinstance(count, bool) or type(count) is not int or not _h3k_module.COUNT_MIN <= count <= _h3k_module.COUNT_MAX:
            raise _h3k_module.H3ControlError(f"count must be a whole number from {_h3k_module.COUNT_MIN} to {_h3k_module.COUNT_MAX}")
        seed = request.args.get("seed")
        if isinstance(seed, bool) or type(seed) is not int:
            raise _h3k_module.H3ControlError("seed must be a whole number")
    except _h3k_module.H3ControlError as exc:
        return {"error": str(exc), "reason_code": "invalid_request"}
    if count == 1:
        args = {k: v for k, v in request.args.items() if k != "count"}
        return await _v4_submit_route(request.model_copy(update={"args": args}))
    batch = _h3k_module.batch_id(request.client_request_id or "")
    takes: list[dict[str, Any]] = []
    first: dict[str, Any] | None = None
    waiting = False
    for index in range(count):
        take_seed = _h3k_module.take_seed(seed, index)
        crid = _h3k_module.take_client_request_id(request.client_request_id or "", index)
        row = await asyncio.to_thread(_read_client_row, crid) if index else None
        if waiting or (index and row is None and not _h3k_previous_ended(takes)):
            waiting = True
            takes.append({"index": index, "seed": take_seed, "id": None, "status": "waiting"})
            continue
        take = request.model_copy(update={
            "args": {**request.args, "seed": take_seed, "take_index": index},
            "client_request_id": crid,
            "requested_basename": _h3k_take_basename(request.requested_basename, index),
        })
        result = await _v4_submit_route(take)
        details = result.get("details") if isinstance(result, dict) else None
        takes.append({"index": index, "seed": take_seed, "id": (details or {}).get("id"),
                      "status": (details or {}).get("status"), "reason_code": result.get("reason_code") if isinstance(result, dict) else None,
                      **({"error": result["error"]} if isinstance(result, dict) and result.get("error") else {})})
        if first is None:
            first = result
        if details is None:
            break  # a refused take stops the batch: the next ones would be refused for the same reason
        _announce_shoot_job(result, submitted=True)
    if waiting:
        _H3K_PENDING[batch] = request
        _H3K_PENDING.move_to_end(batch)
        while len(_H3K_PENDING) > _H3K_PENDING_MAX:
            _H3K_PENDING.popitem(last=False)
        _h3k_start_runner(batch)  # the backend advances the batch itself; no page poll needed
    else:
        _H3K_PENDING.pop(batch, None)
    answer = dict(first or {})
    answer["batch"] = {"id": batch, "count": count, "seed": "seed + index", "takes": takes}
    return answer


_H3K_ENDED = frozenset({"ready", "failed", "cancelled", "moderated"})


def _h3k_previous_ended(takes: list[dict[str, Any]]) -> bool:
    """The take before this one has ended (the ledger admits the next local job only then)."""
    if not takes or takes[-1].get("id") is None:
        return False
    row = _read_client_row_by_job(takes[-1]["id"])
    return row is not None and _normalized_state(row.state) in _H3K_ENDED


def _read_client_row_by_job(studio_job_id: str) -> Any:
    try:
        return _read_ledger_row(studio_job_id)
    except Exception:  # noqa: BLE001 - unreadable means not ended
        return None


async def _h3k_advance_after(result: Any) -> None:
    """After a poll: a take of a waiting batch has ended -> submit the batch's next take."""
    details = result.get("details") if isinstance(result, Mapping) else None
    if not isinstance(details, Mapping) or details.get("target_id") not in _h3k_module.TARGET_IDS \
            or details.get("status") not in _H3K_ENDED or not _H3K_PENDING:
        return
    batch = (details.get("batch") or {}).get("id") if isinstance(details.get("batch"), Mapping) else None
    request = _H3K_PENDING.get(batch) if batch else None
    if request is None:
        return
    try:
        await _h3k_submit_batch(request)
    except Exception:  # noqa: BLE001 - a poll never fails because the next take could not start
        pass


# Generate x N on its own (wave 3, lane w3-gen): one runner task per waiting batch follows the batch's
# active take through the same /poll the page sends (reconcile, publish, measure, events) and, when it
# has ended, submits the next take - one at a time, appended to ComfyUI's queue behind everyone else's
# jobs (never in front, never cancelling).  It ends when no take of the batch waits any more.
_H3K_RUNNERS: dict[str, Any] = {}
_H3K_RUNNER_POLL_SECONDS = 4.0
_H3K_RUNNER_MAX_SECONDS = 8 * 3600.0


def _h3k_start_runner(batch: str) -> None:
    task = _H3K_RUNNERS.get(batch)
    if task is not None and not task.done():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # no event loop (a sync caller): the page's poll still advances the batch
        return
    _H3K_RUNNERS[batch] = loop.create_task(_h3k_run_batch(batch))


def _h3k_active_take(request: SubmitRequest) -> Any:
    """The batch's latest submitted take row (None before take 1)."""
    last = None
    for index in range(int(request.args["count"])):
        row = _read_client_row(_h3k_module.take_client_request_id(request.client_request_id or "", index))
        if row is None:
            break
        last = row
    return last


async def _h3k_run_batch(batch: str) -> None:
    started = time.monotonic()
    request = _H3K_PENDING.get(batch)
    try:
        while request is not None and time.monotonic() - started < _H3K_RUNNER_MAX_SECONDS:
            await asyncio.sleep(_H3K_RUNNER_POLL_SECONDS)
            row = await asyncio.to_thread(_h3k_active_take, request)
            if batch not in _H3K_PENDING:  # every take is submitted: follow the last one until it is published
                if row is None or _normalized_state(row.state) in _H3K_ENDED:
                    return
            if row is not None and _normalized_state(row.state) not in _H3K_ENDED:
                fields = {"id": row.studio_job_id}
                if getattr(row, "project_id", None) is not None:
                    fields["expected_project_id"] = row.project_id
                await poll(PollRequest(**fields))  # the page's poll: publishes the take, then advances the batch
            else:
                await _h3k_submit_batch(request)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - a broken runner leaves the batch to the page's poll and a retry
        pass
    finally:
        if _H3K_RUNNERS.get(batch) is asyncio.current_task():
            _H3K_RUNNERS.pop(batch, None)


def _h3k_take_facts(request: SubmitRequest, plan: Mapping[str, Any] | None) -> dict[str, Any]:
    """The take facts of the controls and the batch (empty for a request without them)."""
    facts: dict[str, Any] = {}
    if plan is not None and plan.get("controls") is not None:
        facts["controls"] = _h3k_module.requested(plan["controls"])
    batch = _h3k_module.batch_args(request.args) if isinstance(request.args, Mapping) else None
    if batch is not None:
        base = (request.client_request_id or "").rsplit(":take-", 1)[0] if batch["index"] else request.client_request_id or ""
        facts["batch"] = {"id": _h3k_module.batch_id(base), "count": batch["count"], "index": batch["index"],
                          "seed": request.args.get("seed")}
    return facts


async def _h3k_live_problems() -> tuple[dict[tuple[str, Any], str | None] | None, list[str] | None]:
    """Per (control, value) the first live problem (None = available); (None, None) when unreachable."""
    needs = _h3k_module.option_needs()
    client = _comfy_client_module.ComfyUIClient()
    try:
        info = await _h3k_object_info(client, [n[0] for rows in needs.values() for n in rows])
        live = await _image_live_loras()
    except Exception:  # noqa: BLE001 - engine down: every option is unchecked
        return None, None
    problems: dict[tuple[str, Any], str | None] = {}
    for key, rows in needs.items():
        problems[key] = next((p for p in (_h3k_module.class_problem(n, info) for n in rows) if p), None)
    for key in (("recipe", "native"), ("sparse", "off"), ("cache", "none"), ("engine", "native")):
        problems[key] = None
    if live is not None:
        live = [name for name in live if (await asyncio.to_thread(_h3k_lora_family, name)) == _h3k_module.H3_FAMILY
                and name.rsplit("/", 1)[-1] not in _h3k_module.PINNED_TURBO_LORAS]
    return problems, live


@router.post("/generation/h3/controls")
async def h3_controls(request: H3ControlsRequest) -> dict[str, Any]:
    """The H3 inspector controls: options, defaults, live availability (with the reason) and render proof."""
    if request.target_id is not None and request.target_id not in _h3k_module.TARGET_IDS:
        return _v4_error("invalid_request")
    problems, loras = await _h3k_live_problems()
    answer = _h3k_module.descriptor(problems, lora_names=loras)
    answer["engine"] = "unchecked" if problems is None else "checked"
    return answer
