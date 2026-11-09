from __future__ import annotations


# H3 finishing (lane w1-h3): latent upscale (lbh, 1.5 MP, multiple of 32) and RIFE 48 fps, each on/off,
# as its own operation on a finished local H3 take.  The source take is named by its job id; the backend
# reads the file it published for that job (never a path from the caller), checks its sidecar pair,
# stages the pair read-only into ComfyUI/output and submits the pinned minimax-h3-finish-v1 graph.
# The result is a new take with ``finish`` take facts that link it to its source; the source stays as it is.
class H3TakeFinishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    client_request_id: str = Field(min_length=1, max_length=200)
    studio_job_id: str = Field(pattern=r"^studio:[0-9a-f]{32}$")
    upscale: bool = True
    rife: bool = False
    save_to: str | None = None
    requested_basename: str | None = None
    approved_roots: list[str] = Field(default_factory=list)
    project_id: str | None = Field(default=None, pattern=r"^prj_[0-9a-f]{32}$")
    project_ref: ProjectReference | None = None


def _h3f_is_finish(request: SubmitRequest) -> bool:
    return request.provider_id == "comfyui-local" and request.target_id == _h3f_module.TARGET_ID


def _h3f_source(request: SubmitRequest, plan: Mapping[str, Any]) -> tuple[Any, Path, dict[str, Any]]:
    """The finished source take of this scope, its published file and its verified pair facts."""
    row = _read_ledger_row(plan["take"])
    if (row is None or getattr(row, "row_contract", None) != 4 or not _h3_is_take(row)
            or row.target_id == _h3f_module.TARGET_ID or row.state != "ready" or row.project_id != request.project_id):
        raise _h3f_module.H3FinishError("That take is not a finished H3 take of this scope")
    try:
        ops = _private_ops()
        path = Path(ops.saved_path_for_receipt(ops.load_for_job(row)))
    except (PrivateOperationError, OSError, ValueError):
        raise _h3f_module.H3FinishError("The take's file cannot be read") from None
    try:
        facts = _h3_mctx_module.verify_pair(path)
    except _h3_mctx_module.MctxError as exc:
        raise _h3f_module.H3FinishError(str(exc), getattr(exc, "reason_code", "no_sidecar")) from None
    return row, path, facts


def _h3f_prepare_request(request: SubmitRequest) -> tuple[dict[str, Any], dict[str, Any], tuple[Any, Path, dict[str, Any]]]:
    plan = _h3f_module.validate_args(request.args)
    source = _h3f_source(request, plan)
    return plan, _h3f_module.identity_args(plan, source[2]["self_id"]), source


async def _h3f_prepare_fresh(request: SubmitRequest) -> dict[str, Any]:
    """A verified source pair before the ledger: a bad take never creates a row."""
    plan, identity, (row, path, facts) = await asyncio.to_thread(_h3f_prepare_request, request)
    return {"plan": plan, "identity": identity, "path": path, "facts": facts, "source_job": row.studio_job_id}


async def _h3f_stage_and_compile(manifest: Any, graph: Any, studio_job_id: str, prepared: Mapping[str, Any]) -> dict[str, Any]:
    """Stage the source pair (read-only copy) into ComfyUI/output/studio-h3/staged, then compile."""
    upload = _h3x_upload or _h3x_module.upload_output
    names = _h3x_module.staged_names(studio_job_id, 1)
    staged: list[str] = []

    def stage() -> list[str]:
        staged.extend(_h3x_module.stage_pairs([(prepared["facts"], str(prepared["path"]))], names, upload))
        return staged

    try:
        await asyncio.to_thread(stage)
        compiled = _h3f_module.compile_request(graph=graph, manifest=manifest, plan=prepared["plan"],
                                               facts=prepared["facts"], staged=staged[0])
    except BaseException:
        await _h3x_release_paths(staged)
        raise
    compiled["staged"] = list(staged)
    compiled["finish"] = {**compiled["finish"], "source_job": prepared["source_job"]}
    return compiled


@router.post("/generation/take/finish")
async def h3_take_finish(request: H3TakeFinishRequest) -> dict[str, Any]:
    """Finish one finished H3 take (upscale and/or RIFE 48 fps) into a new linked take; poll it like any take."""
    submit_request = SubmitRequest(
        schema="studio.generation-request/1", provider_id="comfyui-local", target_id=_h3f_module.TARGET_ID,
        client_request_id=request.client_request_id,
        args={"take": request.studio_job_id, "upscale": request.upscale, "rife": request.rife},
        save_to=request.save_to, requested_basename=request.requested_basename,
        approved_roots=_request_roots(request.approved_roots, request.project_ref),
        **({"project_id": request.project_id, "project_ref": request.project_ref} if request.project_id is not None else {}),
    )
    return await submit(submit_request)
