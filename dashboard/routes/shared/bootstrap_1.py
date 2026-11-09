from __future__ import annotations
# Paid providers (w20-cat, catalog rules 9/11): one optional loader for ``generation/paid/``. The local install
# keeps paid/ and loads it as before; the catalog build omits it and every paid name below is a refusing stand-in.
# The legacy Classic-FLUX core tool (removed upstream in migration v39) is probed inside paid/flux3_core.py; absent, ``flux3`` reports unavailable and its handlers fail only when invoked.
_paid_loader_spec = importlib.util.spec_from_file_location(
    "film_suite_paid_loader", Path(__file__).with_name("generation") / "paid_loader.py"
)
if _paid_loader_spec is None or _paid_loader_spec.loader is None:
    raise RuntimeError("The paid provider loader is unavailable.")
_paid_loader = importlib.util.module_from_spec(_paid_loader_spec)
_paid_loader_spec.loader.exec_module(_paid_loader)
_PAID_PRESENT = _paid_loader.present()
flux3 = _paid_loader.flux_core()

from tools.url_safety import create_ssrf_safe_async_client

if os.name == "nt":
    _KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _CREATE_MUTEX = _KERNEL32.CreateMutexW
    _CREATE_MUTEX.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    _CREATE_MUTEX.restype = wintypes.HANDLE
    _WAIT_FOR_SINGLE_OBJECT = _KERNEL32.WaitForSingleObject
    _WAIT_FOR_SINGLE_OBJECT.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _WAIT_FOR_SINGLE_OBJECT.restype = wintypes.DWORD
    _RELEASE_MUTEX = _KERNEL32.ReleaseMutex
    _RELEASE_MUTEX.argtypes = [wintypes.HANDLE]
    _RELEASE_MUTEX.restype = wintypes.BOOL
    _CLOSE_HANDLE = _KERNEL32.CloseHandle
    _CLOSE_HANDLE.argtypes = [wintypes.HANDLE]
    _CLOSE_HANDLE.restype = wintypes.BOOL
    _WAIT_OBJECT_0 = 0x00000000
    _WAIT_ABANDONED = 0x00000080
    _WAIT_TIMEOUT = 0x00000102
    _WAIT_FAILED = 0xFFFFFFFF

_MUSIC_ANALYSIS_PATH = Path(__file__).with_name("music_analysis.py")
_MUSIC_SPEC = importlib.util.spec_from_file_location("film_suite_music_analysis", _MUSIC_ANALYSIS_PATH)
if _MUSIC_SPEC is None or _MUSIC_SPEC.loader is None:
    raise RuntimeError("The local music analysis engine is unavailable.")
music_analysis: Any | None = None
_MUSIC_LOAD_LOCK = _thread.allocate_lock()


def _load_music_analysis() -> Any:
    """Load the local music engine only for an explicit analysis request."""
    global music_analysis
    if music_analysis is not None:
        return music_analysis
    _MUSIC_LOAD_LOCK.acquire()
    try:
        if music_analysis is not None:
            return music_analysis
        package_prefix = f"{_MUSIC_SPEC.name}."
        previous = {
            name: module
            for name, module in sys.modules.items()
            if name == _MUSIC_SPEC.name or name.startswith(package_prefix)
        }
        for name in previous:
            sys.modules.pop(name, None)
        module: Any | None = None
        try:
            module = importlib.util.module_from_spec(_MUSIC_SPEC)
            sys.modules[_MUSIC_SPEC.name] = module
            _MUSIC_SPEC.loader.exec_module(module)
        except BaseException:
            for name in list(sys.modules):
                if name == _MUSIC_SPEC.name or name.startswith(package_prefix):
                    sys.modules.pop(name, None)
            sys.modules.update(previous)
            music_analysis = None
            raise
        music_analysis = module
        return module
    finally:
        _MUSIC_LOAD_LOCK.release()


_GENERATION_PACKAGE_NAME = "film_suite_generation"
_STUDIO_CORE_PACKAGE_NAME = "film_suite_studio_core"
_MUSIC_CORE_PACKAGE_NAME = "film_suite_music_core"
_PACKAGE_LOAD_LOCK = _thread.RLock()


def _load_generation_package() -> Any:
    with _PACKAGE_LOAD_LOCK:
        return _load_generation_package_unlocked()


def _load_generation_package_unlocked() -> Any:
    """Load generation contracts under a stable package name for top-level imports.

    Hermes loads this file as a top-level module, so ``dashboard.generation``
    cannot be imported through normal package semantics.  Replace any previous
    synthetic package and its descendants before each load; this keeps gateway
    reloads deterministic while restoring the previous registration if the new
    package fails during execution.
    """

    generation_dir = Path(__file__).with_name("generation")
    generation_init = generation_dir / "__init__.py"
    previous = {
        name: module
        for name, module in sys.modules.items()
        if name == _GENERATION_PACKAGE_NAME or name.startswith(f"{_GENERATION_PACKAGE_NAME}.")
    }
    for name in sorted(previous, key=len, reverse=True):
        sys.modules.pop(name, None)

    try:
        spec = importlib.util.spec_from_file_location(
            _GENERATION_PACKAGE_NAME,
            generation_init,
            submodule_search_locations=[str(generation_dir)],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("The local generation contract package is unavailable.")

        package = importlib.util.module_from_spec(spec)
        sys.modules[_GENERATION_PACKAGE_NAME] = package
        spec.loader.exec_module(package)
    except BaseException:
        for name in list(sys.modules):
            if name == _GENERATION_PACKAGE_NAME or name.startswith(f"{_GENERATION_PACKAGE_NAME}."):
                sys.modules.pop(name, None)
        sys.modules.update(previous)
        raise
    return package


def _load_studio_core_package() -> Any:
    with _PACKAGE_LOAD_LOCK:
        return _load_studio_core_package_unlocked()


def _load_studio_core_package_unlocked() -> Any:
    """Load studio_core under a stable synthetic package name.

    The dashboard API is executed as a top-level module by Hermes.  Keep the
    package and every descendant registered only under the synthetic name, and
    replace the complete prior tree on reload.  If execution fails, restore the
    previous tree rather than leaving half-imported modules behind.
    """
    studio_dir = Path(__file__).with_name("studio_core")
    studio_init = studio_dir / "__init__.py"
    previous = {
        name: module
        for name, module in sys.modules.items()
        if name == _STUDIO_CORE_PACKAGE_NAME
        or name.startswith(f"{_STUDIO_CORE_PACKAGE_NAME}.")
    }
    for name in sorted(previous, key=len, reverse=True):
        sys.modules.pop(name, None)

    try:
        spec = importlib.util.spec_from_file_location(
            _STUDIO_CORE_PACKAGE_NAME,
            studio_init,
            submodule_search_locations=[str(studio_dir)],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("The local studio core package is unavailable.")

        package = importlib.util.module_from_spec(spec)
        sys.modules[_STUDIO_CORE_PACKAGE_NAME] = package
        spec.loader.exec_module(package)
        project_store = importlib.import_module(f"{_STUDIO_CORE_PACKAGE_NAME}.project_store")
        # ``studio_core.__init__`` predates RecentProjectStore and cannot be
        # changed as part of this task.  Project the route-facing public surface
        # onto the synthetic package after the package has executed, while
        # retaining the real descendant module in sys.modules for relative
        # imports and reload isolation.
        for symbol in (
            "ProjectStore",
            "RecentProjectStore",
            "ProjectError",
            "ProjectManifestError",
            "read_project",
            "MANIFEST_SCHEMA",
            "MANIFEST_MAX_BYTES",
            "STATE_SCHEMA",
            "STATE_MAX_BYTES",
            "RECENT_WARNING",
        ):
            if hasattr(project_store, symbol):
                setattr(package, symbol, getattr(project_store, symbol))
    except BaseException:
        for name in list(sys.modules):
            if name == _STUDIO_CORE_PACKAGE_NAME or name.startswith(f"{_STUDIO_CORE_PACKAGE_NAME}."):
                sys.modules.pop(name, None)
        sys.modules.update(previous)
        raise
    return package


def _load_music_core_package() -> Any:
    with _PACKAGE_LOAD_LOCK:
        return _load_music_core_package_unlocked()


def _load_music_core_package_unlocked() -> Any:
    """Load music_core under a stable synthetic package name."""
    music_core_dir = Path(__file__).with_name("music_core")
    music_core_init = music_core_dir / "__init__.py"
    previous = {
        name: module
        for name, module in sys.modules.items()
        if name == _MUSIC_CORE_PACKAGE_NAME
        or name.startswith(f"{_MUSIC_CORE_PACKAGE_NAME}.")
    }
    for name in sorted(previous, key=len, reverse=True):
        sys.modules.pop(name, None)

    try:
        spec = importlib.util.spec_from_file_location(
            _MUSIC_CORE_PACKAGE_NAME,
            music_core_init,
            submodule_search_locations=[str(music_core_dir)],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("The local music core package is unavailable.")

        package = importlib.util.module_from_spec(spec)
        sys.modules[_MUSIC_CORE_PACKAGE_NAME] = package
        spec.loader.exec_module(package)
    except BaseException:
        for name in list(sys.modules):
            if name == _MUSIC_CORE_PACKAGE_NAME or name.startswith(f"{_MUSIC_CORE_PACKAGE_NAME}."):
                sys.modules.pop(name, None)
        sys.modules.update(previous)
        raise
    return package


# The one ``bounded_download`` module object shared by private operations and
# the Comfy client/provider (see ``_load_private_operations_module``).
_bounded_download_module: Any = None


def _load_private_operations_module() -> Any:
    """Load ``studio_core.private_operations`` with temporary dependency registration.

    The module resolves ``film_suite_generation.bounded_download`` and the
    ``studio_core.path_boundary`` sibling during exec.  Both parent packages are
    already registered, so register only the private-operations submodule for the
    duration of ``exec_module`` (its frozen dataclass needs
    ``sys.modules[__module__]``) and then pop it together with the transiently
    imported generation leaf so neither synthetic package tree accumulates names
    on reload.  ``plugin_api`` keeps the returned module object as the sole
    reference; nothing is left registered.

    The transiently imported ``bounded_download`` module object is retained in
    ``_bounded_download_module`` so the Comfy client/provider loader binds to
    the *same* module.  Otherwise each loader executes the file again and
    ``DownloadDigest`` exists as two distinct classes, which makes the strict
    ``isinstance`` check in ``mark_downloaded`` reject every real Comfy digest
    (publication ends ``output_invalid``).
    """
    global _bounded_download_module
    with _PACKAGE_LOAD_LOCK:
        module_name = f"{_STUDIO_CORE_PACKAGE_NAME}.private_operations"
        bounded_name = f"{_GENERATION_PACKAGE_NAME}.bounded_download"
        module_path = Path(__file__).with_name("studio_core") / "private_operations.py"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("The local private operations module is unavailable.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            bounded_module = sys.modules.get(bounded_name)
            if bounded_module is None or getattr(bounded_module, "DownloadDigest", None) is not module.DownloadDigest:
                raise RuntimeError("The local bounded download module is unavailable.")
            _bounded_download_module = bounded_module
        finally:
            sys.modules.pop(module_name, None)
            sys.modules.pop(bounded_name, None)
        return module


def _load_request_identity_module() -> Any:
    """Load the Task-5 serializer without extending the synthetic package tree."""
    with _PACKAGE_LOAD_LOCK:
        module_name = f"{_GENERATION_PACKAGE_NAME}.request_identity"
        try:
            return importlib.import_module(module_name)
        finally:
            sys.modules.pop(module_name, None)


_previous_image_film_root = globals().get("_IMAGE_FILM_ROOT")
_previous_image_film_root_lock = globals().get("_IMAGE_FILM_ROOT_LOCK")
_IMAGE_FILM_ROOT: Any | None = None
_IMAGE_FILM_ROOT_PROFILE: Any | None = None
_IMAGE_FILM_ROOT_LOCK = (
    _previous_image_film_root_lock
    if _previous_image_film_root_lock is not None
    else _thread.RLock()
)
class _ImageFilmRootLockAsync:
    """Lane w30-fix3 item 3 (w29 open question): async routes wait for ``_IMAGE_FILM_ROOT_LOCK`` without blocking the
    event loop. The lock is tried without blocking; while another thread holds it the coroutine yields
    (``asyncio.sleep``) and tries again, so every other route keeps answering. Once taken, the critical section runs on
    the loop thread exactly as before (its body has no ``await``), and is released there. Reference: asyncio docs, never
    block the loop on a threading primitive; Starlette runs blocking work off the loop."""

    __slots__ = ()

    async def __aenter__(self) -> None:
        delay = 0.001
        while not _IMAGE_FILM_ROOT_LOCK.acquire(blocking=False):
            await asyncio.sleep(delay)
            delay = min(delay * 2, 0.02)

    async def __aexit__(self, *exc: Any) -> None:
        _IMAGE_FILM_ROOT_LOCK.release()


_IMAGE_FILM_ROOT_LOCK_ASYNC = _ImageFilmRootLockAsync()


if _previous_image_film_root is not None:
    with _IMAGE_FILM_ROOT_LOCK:
        _IMAGE_FILM_ROOT = None
        _IMAGE_FILM_ROOT_PROFILE = None
        try:
            _previous_image_film_root.close()
        except Exception:
            composition_module = sys.modules.get(
                f"{_STUDIO_CORE_PACKAGE_NAME}.image_core.composition_root"
            )
            error_type = getattr(composition_module, "ImageFilmCompositionError", RuntimeError)
            raise error_type("image_film_unavailable") from None


generation = _load_generation_package()
_request_identity_module = _load_request_identity_module()
studio_core = _load_studio_core_package()
_image_film_composition_module = importlib.import_module(
    f"{_STUDIO_CORE_PACKAGE_NAME}.image_core.composition_root"
)
_ImageFilmCompositionError = _image_film_composition_module.ImageFilmCompositionError
_image_film_caller_scope_module = importlib.import_module(
    f"{_STUDIO_CORE_PACKAGE_NAME}.image_core.caller_scope"
)
_ImageFilmCallerScopeError = _image_film_caller_scope_module.ImageFilmCallerScopeError
resolve_trusted_image_film_caller_scope = _image_film_caller_scope_module.resolve_trusted_image_film_caller_scope
music_core = _load_music_core_package()
