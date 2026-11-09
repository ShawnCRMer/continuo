from __future__ import annotations


# Paid FLUX 3 managed adapter from generation/paid/ through the one optional loader (shared/bootstrap_1.py);
# without paid/ (catalog build) a stand-in with no modes that refuses every call (w20-cat).
flux3_provider = _paid_loader.flux3_provider(flux3)
MODES = flux3_provider.handlers
_flux3_provider_module = _paid_loader.loaded("flux3_provider")
_FLUX3_PROVIDER_SPEC = getattr(_flux3_provider_module, "__spec__", None)

_FLUX3_REGISTRY_PATH = Path(__file__).with_name("generation") / "registry.py"
_FLUX3_REGISTRY_NAME = f"{_GENERATION_PACKAGE_NAME}.registry"
_FLUX3_REGISTRY_SPEC = importlib.util.spec_from_file_location(_FLUX3_REGISTRY_NAME, _FLUX3_REGISTRY_PATH)
if _FLUX3_REGISTRY_SPEC is None or _FLUX3_REGISTRY_SPEC.loader is None:
    raise RuntimeError("The local generation registry is unavailable.")
_generation_registry = importlib.util.module_from_spec(_FLUX3_REGISTRY_SPEC)
sys.modules[_FLUX3_REGISTRY_NAME] = _generation_registry
try:
    _FLUX3_REGISTRY_SPEC.loader.exec_module(_generation_registry)
finally:
    sys.modules.pop(_FLUX3_REGISTRY_NAME, None)


def _load_comfy_lifecycle_modules() -> tuple[Any, Any, Any]:
    """Load comfy client, provider, and job store without polluting the package.

    The provider and job store import siblings via relative imports, so each
    is loaded under its generation-package submodule name during exec and then
    popped, mirroring the registry loader.  The client module is standalone;
    it is temporarily registered under its generation name so the provider's
    ``from .comfyui_client import ...`` resolves without a re-import.
    """
    generation_dir = Path(__file__).with_name("generation")

    client_path = generation_dir / "comfyui_client.py"
    client_spec = importlib.util.spec_from_file_location(
        f"{_GENERATION_PACKAGE_NAME}.comfyui_client", client_path
    )
    if client_spec is None or client_spec.loader is None:
        raise RuntimeError("The local ComfyUI client is unavailable.")
    client_module = importlib.util.module_from_spec(client_spec)
    # The client resolves ``from .bounded_download import ...`` during exec,
    # which registers ``{package}.bounded_download`` as a side effect.  Pop it
    # together with the client so the synthetic generation package keeps only
    # its persistent submodules and a reload does not accumulate names.
    _bounded_download_name = f"{_GENERATION_PACKAGE_NAME}.bounded_download"
    # Bind client and provider to the bounded_download module that
    # private_operations already imported (shared/bootstrap_1.py), so
    # DownloadDigest/ExclusiveSink are one class object across the V4 delivery
    # path (strict isinstance checks in mark_downloaded / open_sink).
    if _bounded_download_module is None:
        raise RuntimeError("The local bounded download module is unavailable.")
    sys.modules[_bounded_download_name] = _bounded_download_module
    sys.modules[client_spec.name] = client_module
    try:
        client_spec.loader.exec_module(client_module)
    finally:
        sys.modules.pop(client_spec.name, None)
        sys.modules.pop(_bounded_download_name, None)

    job_store_path = generation_dir / "job_store.py"
    job_store_spec = importlib.util.spec_from_file_location(
        f"{_GENERATION_PACKAGE_NAME}.job_store", job_store_path
    )
    if job_store_spec is None or job_store_spec.loader is None:
        raise RuntimeError("The local generation job store is unavailable.")
    job_store_module = importlib.util.module_from_spec(job_store_spec)
    sys.modules[job_store_spec.name] = job_store_module
    sys.modules[_FLUX3_REGISTRY_NAME] = _generation_registry
    try:
        job_store_spec.loader.exec_module(job_store_module)
    finally:
        sys.modules.pop(job_store_spec.name, None)
        sys.modules.pop(_FLUX3_REGISTRY_NAME, None)

    provider_path = generation_dir / "comfyui_provider.py"
    provider_spec = importlib.util.spec_from_file_location(
        f"{_GENERATION_PACKAGE_NAME}.comfyui_provider", provider_path
    )
    if provider_spec is None or provider_spec.loader is None:
        raise RuntimeError("The local ComfyUI provider is unavailable.")
    provider_module = importlib.util.module_from_spec(provider_spec)
    sys.modules[provider_spec.name] = provider_module
    sys.modules[client_spec.name] = client_module
    sys.modules[job_store_spec.name] = job_store_module
    sys.modules[_bounded_download_name] = _bounded_download_module
    try:
        provider_spec.loader.exec_module(provider_module)
    finally:
        sys.modules.pop(provider_spec.name, None)
        sys.modules.pop(client_spec.name, None)
        sys.modules.pop(job_store_spec.name, None)
        sys.modules.pop(_bounded_download_name, None)

    return client_module, provider_module, job_store_module


_comfy_client_module, _comfy_provider_module, _comfy_job_store_module = _load_comfy_lifecycle_modules()


def _load_h3_conditioning_module(filename: str, name: str) -> Any:
    """Load a standalone H3 conditioning module (no package imports), unregistered."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("generation") / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError("The local H3 conditioning module is unavailable.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Pinned H3 conditioning targets (minimax-h3-frames-v1 / -references-v1):
# compile + live-schema health, media pre-fit/upload + motion activity.
_h3c_module = _load_h3_conditioning_module("h3_conditioning.py", "film_suite_h3_conditioning")
_h3m_module = _load_h3_conditioning_module("h3_media.py", "film_suite_h3_media")
# lane fs-obvb: take sidecars (mctx_v1 header) and the three continuation targets.
_h3x_module = _load_h3_conditioning_module("h3_continuation.py", "film_suite_h3_continuation")
_h3_mctx_module = _h3x_module.mctx
_h3x_profiles: dict[str, Any] = {}
# lane w1-h3: finishing (latent upscale + RIFE 48 fps) as its own operation on a finished take.
_h3f_module = _load_h3_conditioning_module("h3_finish.py", "film_suite_h3_finish")
_h3f_profile: Any = None
# lane w23-vsr (VID-153): RelayVSR video upscale on a finished take, the default video upscaler.
_vsr_module = _load_h3_conditioning_module("video_upscale.py", "film_suite_video_upscale")
_vsr_profile: Any = None
# Tests replace these; production stages through ComfyUI /upload/image (type output).
_h3x_upload: Any = None
_h3x_release: Any = None
_comfy_h3_profiles: dict[str, Any] = {}
# Tests replace this; production uploads to ComfyUI temp/ on the fixed loopback origin.
_h3_media_upload: Any = None
_comfy_job_store: Any = None
_comfy_workflow_profile: Any = None
_comfy_graph: dict[str, Any] | None = None
_comfy_sampling_v2_profile: Any = None
_private_operations: Any = None
_private_operations_store_marker: Any = None


# --- GMI Cloud (Wan 3.0) -----------------------------------------------------
# Standalone adapter loaded unregistered (no package imports, no dataclasses).
# Contract: docs/plans/film-suite-gmi-wan3-contract.md.
# Loaded from generation/paid/ by the optional loader; without paid/ a stand-in (no key, no client, refusals).
_gmi_module = _paid_loader.gmi()
GMI_PROVIDER_ID = _gmi_module.PROVIDER_ID
# Tests replace this with a client over ``httpx.MockTransport``.
_gmi_http_factory: Any = create_ssrf_safe_async_client
# Uploads plus the submit call are bounded well below the stale window, so a
# ``submitting`` row older than the window can no longer be in flight.
_GMI_SUBMIT_BUDGET_SECONDS = 170.0
_GMI_LOG = __import__("logging").getLogger("film_suite.gmi")
_GMI_REASON_TEXT: "collections.OrderedDict[str, str]" = collections.OrderedDict()
_GMI_REASON_TEXT_LIMIT = 256


def _gmi_api_key() -> str | None:
    """The GMI key, read only inside generation/paid/ (never logged); ``None`` without paid/."""
    return _gmi_module.read_api_key()


def _gmi_client() -> Any | None:
    key = _gmi_api_key()
    if key is None:
        return None
    return _gmi_module.GmiClient(api_key=key, client_factory=_gmi_http_factory)


def _gmi_remember_reason(studio_job_id: str, text: Any) -> None:
    if not isinstance(text, str) or not text:
        return
    _GMI_REASON_TEXT[studio_job_id] = text
    _GMI_REASON_TEXT.move_to_end(studio_job_id)
    while len(_GMI_REASON_TEXT) > _GMI_REASON_TEXT_LIMIT:
        _GMI_REASON_TEXT.popitem(last=False)


V4_ROUTE_REASON_MESSAGES = MappingProxyType(
    {
        "project_scope_mismatch": "Generation job belongs to a different project context",
        "cancel_unsupported": "Cancellation is not supported for this provider",
        "private_state_unavailable": "Generation private state is unavailable",
        # lane fs-obvb: a continuation offers nothing for a take without its sidecar
        "no_sidecar": "This take was made before continuation existed",
        "pair_broken": "This take was changed after it was made; it cannot be continued",
        # lane p7-vid (VID-097): Continue from a chosen frame that leaves no room for the held window
        "source_frame_out_of_range": "The chosen frame leaves no room for the held window of this take",
        # lane w23-vsr (VID-153): the upscaled frames would not fit in the engine's free memory
        "vsr_memory": "This upscale needs more memory than is free; x2 or a shorter IN–OUT fits",
        # lane w24-rest (GEM-056): a ComfyUI image run that ran out of memory (generation/comfy_failure.py)
        "out_of_memory": "Not enough memory",
    }
)
V4_ROUTE_REASON_CODES = frozenset(V4_ROUTE_REASON_MESSAGES)
# lane p10-vid (VID-116): refusals the page makes before Generate too; they never become a (failed) job of the list.
V4_REFUSALS_BEFORE_A_JOB = frozenset({"source_frame_out_of_range"})


def _private_ops() -> Any:
    """Return the profile-private Contract-4 operation service.

    Tests replace ``_private_operations`` with a fake.  In production the
    service is scoped to the same profile as the ledger; the store marker also
    prevents a test/temporary ledger from accidentally reusing another ledger's
    private records.
    """
    global _private_operations, _private_operations_store_marker
    store = _comfy_store()
    marker = getattr(store, "path", None)
    if _private_operations is not None and (
        marker is None or _private_operations_store_marker == marker
    ):
        return _private_operations
    if _private_operations_store_marker != marker:
        root = None
        if marker is not None:
            root = Path(marker).parent / "v4-private"
        _private_operations = PrivateOperations(
            hermes_home=_active_hermes_home(),
            private_root=root,
        )
        _private_operations_store_marker = marker
    return _private_operations


def _comfy_workflow_root() -> Path:
    """Resolve the workflow root that contains the pinned H3 target dir.

    Repository layout: ``<repo>/workflows`` (parents[3] from the dashboard
    file).  Installed layout (the drift-safe installer copies approved
    workflow directories to ``dashboard/workflows``): the dashboard sibling.
    """
    candidate = Path(__file__).resolve().parents[3] / "workflows"
    if (candidate / "minimax-h3-t2v" / "manifest.json").exists():
        return candidate
    sibling = Path(__file__).resolve().parent / "workflows"
    if (sibling / "minimax-h3-t2v" / "manifest.json").exists():
        return sibling
    return Path(__file__).with_name("generation") / "workflows"


def _comfy_assets(target_id: str = "minimax-h3-t2v") -> tuple[Any, dict[str, Any]]:
    """Load (profile, graph) for a pinned H3 target, cached per process.

    ``minimax-h3-t2v-sampling-v2`` is a versioned projection of the same
    validated v1 profile and graph.  The pinned conditioning targets load their
    own graphs.  Any other id returns the v1 profile, so the callers'
    manifest-id check refuses it exactly as before.
    """
    global _comfy_workflow_profile, _comfy_graph, _comfy_sampling_v2_profile, _h3f_profile, _vsr_profile
    if target_id == _vsr_module.TARGET_ID:
        if _vsr_profile is None:
            _vsr_profile = _vsr_module.load_profile(_comfy_workflow_root())
        return _vsr_profile, _vsr_profile.graph
    if target_id == _h3f_module.TARGET_ID:
        if _h3f_profile is None:
            _h3f_profile = _h3f_module.load_profile(_comfy_workflow_root())
        return _h3f_profile, _h3f_profile.graph
    if target_id in _h3x_module.TARGET_IDS:
        profile = _h3x_profiles.get(target_id)
        if profile is None:
            profile = _h3x_module.load_profile(target_id, _comfy_workflow_root())
            _h3x_profiles[target_id] = profile
        return profile, profile.graph
    if target_id in _h3c_module.TARGET_IDS:
        profile = _comfy_h3_profiles.get(target_id)
        if profile is None:
            profile = _h3c_module.load_h3_profile(target_id, _comfy_workflow_root())
            _comfy_h3_profiles[target_id] = profile
        return profile, profile.graph
    if _comfy_workflow_profile is None or _comfy_graph is None:
        manifest_path = Path(__file__).with_name("generation") / "workflow_manifest.py"
        manifest_spec = importlib.util.spec_from_file_location(
            "film_suite_workflow_manifest", manifest_path
        )
        if manifest_spec is None or manifest_spec.loader is None:
            raise RuntimeError("The local workflow manifest module is unavailable.")
        manifest_module = importlib.util.module_from_spec(manifest_spec)
        sys.modules["film_suite_workflow_manifest"] = manifest_module
        try:
            manifest_spec.loader.exec_module(manifest_module)
        finally:
            sys.modules.pop("film_suite_workflow_manifest", None)

        root = _comfy_workflow_root()
        _comfy_workflow_profile = manifest_module.load_workflow_manifest(
            target_id="minimax-h3-t2v",
            workflow_root=root,
        )
        _comfy_graph = manifest_module.compile_base_graph(_comfy_workflow_profile)
        _comfy_sampling_v2_profile = manifest_module.project_sampling_v2(_comfy_workflow_profile)
    if target_id == "minimax-h3-t2v-sampling-v2":
        return _comfy_sampling_v2_profile, _comfy_graph
    return _comfy_workflow_profile, _comfy_graph


async def _comfy_static_health(*, client: Any, manifest: Any, graph: Any) -> Any:
    """Static health dispatched by target: pinned conditioning targets add LoRA,
    model patch and live-schema checks; every other target is unchanged."""
    tested = _h3_runtime_records()
    if isinstance(manifest, Mapping) and (manifest.get("id") in _h3c_module.TARGET_IDS
                                          or manifest.get("id") in _h3x_module.TARGET_IDS
                                          or manifest.get("id") in (_h3f_module.TARGET_ID, _vsr_module.TARGET_ID)):
        return await _h3c_module.conditioning_static_health(
            graph=graph,
            manifest=manifest,
            client=client,
            health_type=generation.StaticHealth,
            client_error=_comfy_client_module.ComfyClientError,
            tested=tested,
            policy=_comfy_provider_module.runtime_policy,
        )
    return await _comfy_provider_module.static_health(client=client, manifest=manifest, graph=graph, tested=tested)


def _comfy_store() -> Any:
    global _comfy_job_store
    if _comfy_job_store is None:
        _comfy_job_store = _comfy_job_store_module.JobStore()
    return _comfy_job_store
