"""One optional loader for the paid providers in ``generation/paid/`` (w20-cat, catalog rules 9/11).

The local install keeps ``paid/`` and loads it exactly as before. The catalog build omits ``paid/``; then every
name the routes bind at import time is an inert stand-in that refuses with ``paid_unavailable`` before any
network or credential access, so the shared routes load unchanged and answer truthfully. Absence is a file check
on ``paid/__init__.py`` (the pattern of the old ``find_spec`` probe in ``routes/shared/bootstrap_1.py``).

Standalone on purpose: no package imports, loaded by path from the route bootstrap.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any

PAID_DIR = Path(__file__).resolve().with_name("paid")
REASON_CODE = "paid_unavailable"
MESSAGE = "Paid models are not part of this install."


def present() -> bool:
    return (PAID_DIR / "__init__.py").is_file()


def load(name: str, *, module_name: str | None = None, register: bool = False) -> ModuleType | None:
    """``paid/<name>.py`` as a standalone module, or ``None`` when ``paid/`` is absent."""
    path = PAID_DIR / f"{name}.py"
    if not present() or not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(module_name or f"film_suite_paid_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"The paid module {name} could not be loaded.")
    module = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


_LOADED: dict[str, ModuleType] = {}


def loaded(name: str) -> ModuleType | None:
    """The paid module a loader call above executed (route-loader tests read its dataclasses), else ``None``."""
    return _LOADED.get(name)


def refusal() -> dict[str, Any]:
    return {"ok": False, "error": MESSAGE, "reason_code": REASON_CODE, "reason_text": MESSAGE}


class UnavailableFluxCore:
    """Stand-in for the Hermes core FLUX tool: unavailable, and any ``_handle_*`` fails only when invoked."""

    @staticmethod
    def check_bfl_requirements() -> bool:
        return False

    def __getattr__(self, name: str):
        if name.startswith("_handle_"):
            async def _unavailable(*_args, **_kwargs):
                raise RuntimeError("Classic FLUX is unavailable on this Hermes install.")
            return _unavailable
        raise AttributeError(name)


def flux_core() -> Any:
    module = load("flux3_core")
    core = module.core_module() if module is not None else None
    return core if core is not None else UnavailableFluxCore()


class UnavailableFlux3Provider:
    """Stand-in for ``paid/flux3_provider.Flux3Provider`` without ``paid/``: no modes, every call refuses."""

    handlers: dict[str, Any] = {}

    def supports(self, mode: str) -> bool:
        return False

    def validate(self, mode: str, args: Any) -> str:
        return REASON_CODE

    async def submit(self, mode: str, args: Any) -> str:
        return json.dumps(refusal())

    async def poll(self, args: Any) -> str:
        return json.dumps(refusal())

    def poll_save_destination(self, save_to: Any) -> None:
        return None

    @staticmethod
    async def submit_managed(**_kwargs: Any) -> dict[str, Any]:
        return refusal()

    @staticmethod
    async def poll_managed(**_kwargs: Any) -> dict[str, Any]:
        return refusal()

    @staticmethod
    async def upload_managed_media(**_kwargs: Any) -> str:
        raise ValueError("managed media upload is unavailable")


def flux3_provider(core: Any) -> Any:
    """``Flux3Provider(core)`` from ``paid/``, its gateway helper registered for the exec only; else the stand-in."""
    vendor_name = "film_suite_flux3_provider_vendor"
    vendor = load("_managed_gateway_vendor", module_name=vendor_name, register=True)
    if vendor is None:
        return UnavailableFlux3Provider()
    try:
        # Frozen dataclasses resolve sys.modules[cls.__module__] while each class is built: register the synthetic
        # module for the exec, then pop it (reload-safe, as before the move).
        module = load("flux3_provider", module_name="film_suite_flux3_provider", register=True)
    finally:
        sys.modules.pop("film_suite_flux3_provider", None)
        sys.modules.pop(vendor_name, None)
    _LOADED["flux3_provider"] = module
    return module.Flux3Provider(core)


class _GmiModelRefused(ValueError):
    pass


class _UnavailableGmiClient:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(MESSAGE)


class UnavailableGmi:
    """Stand-in for ``paid/gmi_cloud_provider`` without ``paid/``: the import-time names, no key, no client."""

    PROVIDER_ID = "gmi-cloud"
    TARGET_ID = "wan3-video"
    CREDIT_MESSAGE = MESSAGE
    DEFAULT_DURATION_SECONDS = 5
    MAX_TOTAL_SECONDS = 30
    MEDIA_FIELDS = MappingProxyType({})
    PROBLEMS = MappingProxyType({"args": MESSAGE})
    GmiModelRefused = _GmiModelRefused
    GmiClient = _UnavailableGmiClient

    @staticmethod
    def read_api_key() -> None:
        return None

    @staticmethod
    def media_values(_args: Any, _field: str) -> list[Any]:
        return []

    @staticmethod
    def validate_args(*_args: Any, **_kwargs: Any) -> str:
        return "args"

    @staticmethod
    def max_bytes_for(_field: str) -> int:
        return 0

    @staticmethod
    def is_allowed_download_url(_url: Any) -> bool:
        return False

    @staticmethod
    def build_payload(_args: Any) -> dict[str, Any]:
        raise RuntimeError(MESSAGE)

    @staticmethod
    def build_request_identity(*_args: Any, **_kwargs: Any) -> tuple[str, str]:
        raise RuntimeError(MESSAGE)


def gmi() -> Any:
    module = load("gmi_cloud_provider", module_name="film_suite_gmi_cloud_provider")
    return module if module is not None else UnavailableGmi()


__all__ = ["MESSAGE", "PAID_DIR", "REASON_CODE", "flux3_provider", "flux_core", "gmi", "load", "present", "refusal"]
