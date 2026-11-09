from __future__ import annotations


# --- Video steps (lane w9-chain item 1): studio.video-step/1 manifests and the support check ---------------------
# GET /generation/video-steps lists one manifest per video target that can end a shot chain (generation/video_steps.py,
# every number read from the target's own module and catalog entry per request). POST /generation/video-step/support
# answers whether one concrete request fits a target: supported, plain reasons, the effective frames, canvas,
# resolved keyframe frames and what the target does with the guide. Neither route compiles a graph or talks to the
# engine; files are only confined to the approved folders and probed. Agent tools: continuo_video_steps,
# continuo_video_step_support.
_VIDEO_STEPS_NAME = "film_suite_video_steps"


def _video_steps_module() -> Any:
    module = sys.modules.get(_VIDEO_STEPS_NAME)
    if module is not None:
        return module
    # The unit runs in plugin_api's namespace: __file__ is dashboard/plugin_api.py.
    spec = importlib.util.spec_from_file_location(_VIDEO_STEPS_NAME, Path(__file__).with_name("generation") / "video_steps.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("The video step module is unavailable.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_VIDEO_STEPS_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_VIDEO_STEPS_NAME, None)
        raise
    return module


class VideoStepSupportRequest(BaseModel):
    """Loose on purpose: a wrong field value is a reason in the answer, never a 422."""

    model_config = ConfigDict(extra="forbid")

    target_id: str = Field(default="", max_length=120)
    duration_seconds: Any = None
    frames: Any = None
    keyframes: list[dict[str, Any]] = Field(default_factory=list, max_length=64)
    motion_guide: dict[str, Any] | None = None
    approved_roots: list[str] = Field(default_factory=list, max_length=64)


def _video_steps_source() -> Any:
    return _video_steps_module().StepSource(_h3c_module)


def _video_steps_evidence() -> Any:
    return _video_steps_module().read_evidence(Path(__file__).with_name("generation") / "model_evidence.v1.json")


def _video_step_support(body: dict[str, Any]) -> dict[str, Any]:
    steps = _video_steps_module()
    return steps.support(_video_steps_source(), _generation_registry.catalog(), body,
                         confine=_confined_local_path, probe=probe_media)


@router.get("/generation/video-steps")
async def generation_video_steps() -> dict[str, Any]:
    try:
        steps = _video_steps_module()
        return {"steps": steps.manifests(_video_steps_source(), _generation_registry.catalog(), _video_steps_evidence())}
    except Exception:  # noqa: BLE001 - finite plugin boundary
        return {"error": {"code": "video_steps_unavailable", "message": "The video steps cannot be read.", "retryable": True}}


@router.post("/generation/video-step/support")
async def generation_video_step_support(request: VideoStepSupportRequest) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(_video_step_support, request.model_dump())
    except Exception:  # noqa: BLE001 - finite plugin boundary
        return {"error": {"code": "video_steps_unavailable", "message": "The support check cannot run.", "retryable": True}}
