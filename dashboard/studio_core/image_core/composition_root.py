"""Trusted internal Image/Film composition for one active profile.

This module is deliberately an internal composition root.  It is not an HTTP
surface and it does not select a provider, material root, or runtime service.
The only authority it creates comes from the accepted profile receipt-key
enrollment boundary.  HTTP callers are resolved by ``caller_scope`` at the
route boundary; this root never accepts a caller-provided identity, session ID,
dependency, factory, or path.
"""
from __future__ import annotations
from ..legacy_plugin_migration import state_root


import hashlib
import hmac
import importlib
import os
import re
import sys
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import path_boundary
from ..project_store import ProjectStore
from .film_adoption import FilmAdoptionService
from .film_binding_store import FilmBindingStore, FilmBindingStoreError
from .image_catalog_contract import ImageCatalogError, build_catalog
from .image_context_owners import (
    SCRATCH_OWNER,
    ImageContextOwnerError,
    ImageContextOwnerIndex,
    ScopedImageStates,
)
from .image_store import HmacImageOwnerAuthority, ImageStateError, ImageStateStore
from .receipt_key_enrollment import ReceiptKeyEnrollment, ReceiptKeyEnrollmentError


_PROJECT_ID_RE = re.compile(r"^prj_[0-9a-f]{32}$")
_CONTEXT_ID_RE = re.compile(r"^imgctx_[0-9a-f]{32}$")
_CALLER_SCOPE_RE = re.compile(r"^imgcaller_[0-9a-f]{64}$")
_ROOT_OPEN_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class _ProfileCacheAnchor:
    """Immutable normalized spelling plus the opened profile directory identity."""

    path: str
    identity: Any


@dataclass(frozen=True, slots=True)
class _ActiveProfileSnapshot:
    """A repeated active-home observation and its opened directory identity."""

    path: str | None
    anchor: _ProfileCacheAnchor | None


class ImageFilmCompositionError(ValueError):
    """Finite failure at the trusted Image/Film composition boundary."""

    CODES = frozenset(
        {
            "image_key_unavailable",
            "image_state_unavailable",
            "image_state_not_private",
            "project_not_found",
            "invalid_project",
            "project_unavailable",
            "project_scope_mismatch",
            "root_closed",
            "image_film_unavailable",
            "image_catalog_unavailable",
            "image_catalog_state_unavailable",
            "image_catalog_identity_changed",
            "image_catalog_limit_exceeded",
            "image_catalog_response_too_large",
        }
    )

    def __init__(self, code: str) -> None:
        if code not in self.CODES:
            code = "image_film_unavailable"
        self.code = code
        self.reason_code = code
        self.reason = code
        super().__init__(code)


_ROOT_CODES = ImageFilmCompositionError.CODES


def _active_hermes_home() -> Path | None:
    """Resolve the current profile without accepting a caller override."""
    from hermes_constants import get_hermes_home

    value = get_hermes_home()
    return None if value is None else Path(value)


def _normalised_profile_path(home: Path | str | None) -> str | None:
    if home is None:
        return None
    text = os.fspath(home)
    if type(text) is not str:
        raise TypeError("profile home must be text")
    return os.path.normcase(os.path.abspath(text))


def _profile_anchor_for_home(home: Path | str | None) -> _ProfileCacheAnchor | None:
    """Capture a profile anchor from a real opened directory.

    ``exists``/``is_dir`` only classify the ordinary no-anchor states.  The
    identity used for binding always comes from the opened native directory;
    failures of that operation remain exceptions for the finite caller
    boundary to map rather than becoming an indistinguishable missing profile.
    """
    normalized = _normalised_profile_path(home)
    if normalized is None:
        return None
    path = Path(normalized)
    if not path.exists() or not path.is_dir():
        return None
    path_boundary._reject_reparse_components(path)
    handle, identity = path_boundary._open_existing(path, directory=True)
    try:
        if not bool(getattr(identity, "is_directory", False)):
            return None
        return _ProfileCacheAnchor(normalized, identity)
    finally:
        handle.close()


def _same_profile_anchor(left: _ProfileCacheAnchor | None, right: _ProfileCacheAnchor | None) -> bool:
    if left is None or right is None or left.path != right.path:
        return False
    try:
        forward = getattr(left.identity, "same_object", None)
        reverse = getattr(right.identity, "same_object", None)
        return callable(forward) and bool(forward(right.identity)) and (
            not callable(reverse) or bool(reverse(left.identity))
        )
    except Exception:
        return False


def _same_optional_profile_anchor(
    left: _ProfileCacheAnchor | None,
    right: _ProfileCacheAnchor | None,
) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return _same_profile_anchor(left, right)


def _bind_composition_lock(lock: Any) -> None:
    """Use the plugin factory's reload-stable gate for root construction too."""
    global _ROOT_OPEN_LOCK
    if hasattr(lock, "__enter__") and hasattr(lock, "__exit__"):
        _ROOT_OPEN_LOCK = lock


def _stable_active_home() -> tuple[Path | None, str | None, _ProfileCacheAnchor | None] | None:
    """Resolve and anchor the active home before profile enrollment."""
    try:
        with _ROOT_OPEN_LOCK:
            first_home = _active_hermes_home()
            first_path = _normalised_profile_path(first_home)
            first_anchor = _profile_anchor_for_home(first_home)
            second_home = _active_hermes_home()
            second_path = _normalised_profile_path(second_home)
            if first_path != second_path:
                return None
            return first_home, first_path, first_anchor
    except Exception:
        raise ImageFilmCompositionError("image_film_unavailable") from None


def _stable_active_profile_snapshot(active_resolver: Any | None = None) -> _ActiveProfileSnapshot | None:
    """Linearize active profile spelling and native directory identity together.

    Hermes exposes a context-local home override or a process ``HERMES_HOME``
    value, but no switch lock/generation that this plugin can share.  The shared
    gate serializes every composition/factory transition.  Each bounded sample
    captures the native identity, observes the resolver again, and then takes a
    final native reanchor.  That final reanchor is the identity publication
    point; a transition observed before it fails closed.  A future unsynchronized
    transition after that point cannot be predicted.
    """
    resolver = _active_hermes_home if active_resolver is None else active_resolver
    try:
        with _ROOT_OPEN_LOCK:
            samples: list[_ActiveProfileSnapshot] = []
            for _ in range(2):
                home = resolver()
                path = _normalised_profile_path(home)
                anchor = _profile_anchor_for_home(home)
                post_home = resolver()
                post_path = _normalised_profile_path(post_home)
                if path != post_path:
                    return None
                final_anchor = _profile_anchor_for_home(post_home)
                if not _same_optional_profile_anchor(anchor, final_anchor):
                    return None
                samples.append(_ActiveProfileSnapshot(post_path, final_anchor))
            first = samples[0]
            if any(
                sample.path != first.path
                or not _same_optional_profile_anchor(first.anchor, sample.anchor)
                for sample in samples[1:]
            ):
                return None
            return samples[-1]
    except Exception:
        raise ImageFilmCompositionError("image_film_unavailable") from None


def _raise(code: str) -> Any:
    raise ImageFilmCompositionError(code) from None


def _exception_code(exc: BaseException) -> str | None:
    try:
        code = object.__getattribute__(exc, "code")
    except Exception:
        return None
    return code if type(code) is str else None


def _map_exception(exc: BaseException, *, default: str = "image_film_unavailable") -> ImageFilmCompositionError:
    """Map accepted domain failures while hiding all implementation detail."""
    if isinstance(exc, ImageFilmCompositionError):
        return exc
    if isinstance(exc, ReceiptKeyEnrollmentError):
        code = _exception_code(exc)
        if code in {"image_key_unavailable", "image_state_unavailable", "image_state_not_private"}:
            return ImageFilmCompositionError(code)
        return ImageFilmCompositionError("image_state_unavailable")
    if isinstance(exc, path_boundary.ProjectError):
        code = _exception_code(exc)
        if code in {"project_not_found", "invalid_project", "project_unavailable"}:
            return ImageFilmCompositionError(code)
        if code == "project_io_failed":
            return ImageFilmCompositionError("project_unavailable")
        return ImageFilmCompositionError("invalid_project")
    if isinstance(exc, ImageStateError):
        return ImageFilmCompositionError("image_state_unavailable")
    if isinstance(exc, ImageContextOwnerError):
        if exc.code in {"context-not-owned", "context-owned-elsewhere"}:
            return ImageFilmCompositionError("image_context_not_owned")
        return ImageFilmCompositionError("image_state_unavailable")
    if isinstance(exc, ImageCatalogError):
        code = _exception_code(exc)
        if code in {
            "image_catalog_state_unavailable",
            "image_catalog_identity_changed",
            "image_catalog_limit_exceeded",
            "image_catalog_response_too_large",
        }:
            return ImageFilmCompositionError(code)
        return ImageFilmCompositionError("image_catalog_state_unavailable")
    if isinstance(exc, FilmBindingStoreError):
        code = _exception_code(exc)
        if code in {"project-id-mismatch", "expected-project-id-invalid", "project-id-invalid"}:
            return ImageFilmCompositionError("project_scope_mismatch")
        if code in {
            "project-identity-changed",
            "state-parent-identity-changed",
            "project-path-invalid",
            "project-lock-missing",
            "project-lock-unavailable",
            "project-lock-release-failed",
        }:
            return ImageFilmCompositionError("project_unavailable")
        if code in {"project-manifest-unavailable", "project-manifest-invalid", "state-directory-invalid"}:
            return ImageFilmCompositionError("invalid_project")
        return ImageFilmCompositionError("image_state_unavailable")
    return ImageFilmCompositionError(default)


def _strict_project_id(value: Any) -> str:
    if type(value) is not str or _PROJECT_ID_RE.fullmatch(value) is None:
        _raise("invalid_project")
    return value


def _generation_package_name() -> str:
    """Find the already-registered synthetic generation package.

    The dashboard loader uses ``film_suite_generation`` in production, while
    repository tests deliberately use collision-safe package names.  Selection
    is based on the registered package path, never on an arbitrary caller
    module or a global dashboard ``sys.path`` entry.
    """
    generation_dir = Path(__file__).resolve().parents[2] / "generation"
    stable = "film_suite_generation"
    package_root = (__package__ or "").split(".", 1)[0]
    candidates = [stable]
    if package_root.endswith("_studio"):
        candidates.insert(0, package_root[: -len("_studio")] + "_generation")
    elif package_root.endswith("_studio_core"):
        candidates.insert(0, package_root[: -len("_studio_core")] + "_generation")
    for name in candidates:
        module = sys.modules.get(name)
        try:
            if module is not None and list(getattr(module, "__path__", ())) == [str(generation_dir)]:
                return name
        except Exception:
            continue
    for name, module in tuple(sys.modules.items()):
        if "." in name or not hasattr(module, "__path__"):
            continue
        try:
            paths = list(module.__path__)
        except Exception:
            continue
        if len(paths) == 1 and Path(paths[0]).resolve() == generation_dir:
            return name
    return stable


def _generation_component(module_name: str, symbol: str) -> Any:
    module = importlib.import_module(f"{_generation_package_name()}.{module_name}")
    return getattr(module, symbol)


def _live_preview_adapter(adapter_type: Any, client: Any) -> Any:
    """The Krea adapter, submitting under the plugin's preview client id.

    ComfyUI sends progress and preview frames only to the websocket whose
    clientId matches the prompt's client_id; the single preview reader in
    plugin_api listens with that id.  Read-only for ComfyUI otherwise.
    """
    module = importlib.import_module(adapter_type.__module__)
    extra = {"kj_preview": True} if hasattr(module, "LIVE_PREVIEW_NODE_CLASS") else {}
    return adapter_type(
        client,
        submit_client_id=module.LIVE_PREVIEW_CLIENT_ID,
        preview_method=module.LIVE_PREVIEW_METHOD,
        **extra,
    )


def _with_edit_routes(create_adapter: Any, client: Any, jobs: Any, reader: Any = None) -> Any:
    """R4: one provider seam for create AND edit jobs on the shared ledger.

    Create jobs keep going to the Krea adapter unchanged; edit jobs go to the
    adapter of their own pinned edit target (Qwen-Image 2.1 PE v3 first).  The
    route is chosen by the job's target id, never by a caller.  When the edit
    modules cannot load, the Krea adapter is returned unchanged.
    """
    try:
        qwen_type = _generation_component("qwen_image_edit_adapter", "QwenImageEditAdapter")
        routing_type = _generation_component("qwen_image_edit_adapter", "EditRoutingAdapter")
        qwen = _live_preview_adapter(qwen_type, client)
    except Exception:  # noqa: BLE001 - edits off; creating pictures is unaffected
        return create_adapter

    def target_of(job_id: str) -> Any:
        return ((jobs.get_job(job_id) or {}).get("provider") or {}).get("target_id")

    runtime_type = _generation_component("image_workflow_runtime", "ImageWorkflowRuntime")
    entries = _generation_component("image_workflow_contract", "entries")
    runtime = runtime_type(client, jobs, reader)
    routes = {qwen.target_id: qwen, **{target: runtime for target in entries()}}
    try:  # FLUX.3 Klein edit (the artist's alternative edit model): same seam, its own pinned graph
        klein = _live_preview_adapter(_generation_component("flux3_klein_edit_adapter", "Flux3KleinEditAdapter"), client)
        routes[klein.target_id] = klein
    except Exception:  # noqa: BLE001 - Klein edits off; Qwen edits are unaffected
        pass
    try:  # FLUX.2 Klein edit (the fast edit model): same seam, its own pinned graph
        klein2 = _live_preview_adapter(_generation_component("flux2_klein_edit_adapter", "Flux2KleinEditAdapter"), client)
        routes[klein2.target_id] = klein2
    except Exception:  # noqa: BLE001 - FLUX.2 Klein edits off; the other edit models are unaffected
        pass
    try:  # FLUX.3 Klein with boxes (Image Box, NKD Klein nodes): same seam, its own pinned graph
        kleinbox = _live_preview_adapter(_generation_component("flux3_klein_box_adapter", "Flux3KleinBoxEditAdapter"), client)
        routes[kleinbox.target_id] = kleinbox
    except Exception:  # noqa: BLE001 - box edits off; the other edit models are unaffected
        pass
    try:  # w24-h3img: P-H3-Image edit (PrunaAI LoRA on H3 FL2VA): same seam, the catalog edit folder's graph
        ph3 = _live_preview_adapter(_generation_component("p_h3_image_edit_adapter", "PH3ImageEditAdapter"), client)
        routes[ph3.target_id] = ph3
    except Exception:  # noqa: BLE001 - P-H3-Image edits off; the other edit models are unaffected
        pass
    try:  # w24-krea: Krea 2 Edit (Replace on Kroma, Restyle and Relight on Krea 2 Turbo): same seam, its own pinned graphs
        for krea_class in ("Krea2EditAdapter", "Krea2EditTurboAdapter"):
            krea = _live_preview_adapter(_generation_component("krea2_edit_adapter", krea_class), client)
            routes[krea.target_id] = krea
    except Exception:  # noqa: BLE001 - Krea edits off; the other edit models are unaffected
        pass
    try:  # SeedVR2 upscale (a new version of an owned picture): same seam, its own graph
        upscale = _generation_component("image_upscale", "UpscaleAdapter")(client)
        routes[upscale.target_id] = upscale
    except Exception:  # noqa: BLE001 - upscaling off; creating and editing are unaffected
        pass
    try:  # composed pictures (Accept as version, Expand): a passthrough on the same ledger
        compose = _generation_component("image_compose", "ComposeAdapter")(client)
        routes[compose.target_id] = compose
    except Exception:  # noqa: BLE001 - composing off; creating and editing are unaffected
        pass
    try:  # w6-gptimage: GPT Image through Hermes' openai-codex provider (subscription), same ledger, never ComfyUI
        subscription = _generation_component("paid.subscription_image_provider", "SubscriptionImageRuntime")(jobs, reader)
        for target in subscription.target_ids:
            routes[target] = subscription
    except Exception:  # noqa: BLE001 - GPT Image off; local models are unaffected
        for target in ("gpt-image-2-create-v1", "gpt-image-2-edit-v1"):
            routes.pop(target, None)
    return routing_type(create_adapter, routes, target_of)


class _PrivateImageBoundary:
    """Only the private path primitives required by ``ImageStateStore``.

    ``PathBoundary`` intentionally rejects paths inside the Hermes home when it
    is used for user projects.  Image contexts are profile-private state, so the
    adapter reuses its hardened handle primitives without projecting a broad
    alternate path service.
    """

    __slots__ = (
        "_boundary_module",
        "_profile_boundary",
        "_profile_key",
        "_image_root",
        "hermes_home",
        "_lock_dir",
    )

    def __init__(self, profile_boundary: Any, profile_key: bytes, image_root: Path) -> None:
        self._boundary_module = path_boundary
        self._profile_boundary = profile_boundary
        self._profile_key = profile_key
        self._image_root = Path(image_root)
        self.hermes_home = profile_boundary.hermes_home
        self._lock_dir = state_root(self.hermes_home / "plugin-data")

    @staticmethod
    def _private_path_text(value: os.PathLike[str] | str) -> str:
        try:
            text = os.fspath(value)
        except Exception as exc:
            raise OSError("private image path is unavailable") from exc
        if type(text) is not str:
            raise OSError("private image path is unavailable")
        if "\x00" in text or any(ord(char) < 32 or ord(char) == 127 for char in text):
            raise OSError("private image path is unavailable")
        if any(unicodedata.category(char) in {"Cc", "Cs"} for char in text):
            raise OSError("private image path is unavailable")
        components = [part for part in re.split(r"[\\\\/]", text) if part not in ("", ".")]
        if any(component == ".." for component in components):
            raise OSError("private image path is unavailable")
        return text

    @staticmethod
    def _reject_windows_component_chars(text: str, *, child: bool = False) -> None:
        if os.name != "nt":
            return
        invalid = '<>"|?*'
        if child:
            # A child name never owns a drive prefix; every colon is ADS or a
            # drive-relative spelling and must be rejected before native open.
            invalid += ":"
        components = [part for part in re.split(r"[\\\\/]", text) if part not in ("", ".")]
        if any(any(char in invalid for char in component) for component in components):
            raise OSError("private image path is unavailable")

    def _check_private_path(self, value: os.PathLike[str] | str) -> Path:
        try:
            text = self._private_path_text(value)
            # Reuse the accepted boundary grammar before any private primitive.  The
            # adapter's lexical parent check remains explicit because PathBoundary's
            # descriptor walks must never receive a path that can escape this root.
            self._boundary_module._reject_path_spelling(text)
            self._reject_windows_component_chars(text)
            path = Path(text)
            self._boundary_module._reject_reparse_components(path)
            path.relative_to(self._image_root)
        except OSError:
            raise
        except Exception as exc:
            raise OSError("private image path is unavailable") from exc
        return path

    def ensure_private_node(self, path: Path, *, directory: bool) -> None:
        self._boundary_module.ensure_private_node(self._check_private_path(path), directory=directory)

    def open_existing(self, path: Path, *, directory: bool) -> tuple[Any, Any]:
        return self._boundary_module._open_existing(self._check_private_path(path), directory=directory)

    def create_exclusive_child(
        self,
        parent_handle: Any,
        parent_path: Path,
        name: str,
    ) -> tuple[Any, Any]:
        self._check_private_child_name(name)
        return self._boundary_module._create_exclusive_child(
            parent_handle,
            self._check_private_path(parent_path),
            name,
        )

    def _check_private_child_name(self, name: str) -> None:
        if type(name) is not str or not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise OSError("private image child name is invalid")
        text = self._private_path_text(name)
        candidate = str(self._image_root / text)
        try:
            self._boundary_module._reject_path_spelling(candidate)
            self._reject_windows_component_chars(text, child=True)
        except OSError:
            raise
        except Exception as exc:
            raise OSError("private image child name is invalid") from exc

    def open_child_existing(self, parent_handle: Any, parent_path: Path, name: str, *, directory: bool) -> tuple[Any, Any]:
        self._check_private_child_name(name)
        try:
            parent = Path(os.fspath(parent_path))
        except (TypeError, ValueError) as exc:
            raise OSError("private image diagnostic path is unavailable") from exc
        return self._boundary_module._open_child_existing(parent_handle, parent, name, directory=directory)

    def enumerate_child_entries(self, parent_handle: Any, parent_path: Path, *, max_entries: int) -> list[tuple[str, Any]]:
        try:
            diagnostic_path = Path(os.fspath(parent_path))
        except (TypeError, ValueError) as exc:
            raise OSError("private image diagnostic path is unavailable") from exc
        return self._boundary_module.enumerate_child_entries(
            parent_handle,
            diagnostic_path,
            max_entries=max_entries,
        )

    def enumerate_child_names(self, parent_handle: Any, parent_path: Path, *, max_entries: int) -> list[str]:
        try:
            diagnostic_path = Path(os.fspath(parent_path))
        except (TypeError, ValueError) as exc:
            raise OSError("private image diagnostic path is unavailable") from exc
        return self._boundary_module.enumerate_child_names(
            parent_handle,
            diagnostic_path,
            max_entries=max_entries,
        )

    def read_handle_bytes(self, handle: Any, *, max_bytes: int) -> bytes:
        return self._profile_boundary.read_handle_bytes(handle, max_bytes=max_bytes)

    def pin_directory(self, path: Path) -> Any:
        handle, identity = self.open_existing(path, directory=True)
        handle.close()
        return identity

    def revalidate_directory(self, path: Path, expected: Any) -> None:
        handle, identity = self.open_existing(path, directory=True)
        handle.close()
        if not self._same_identity(expected, identity):
            raise OSError("private image directory identity changed")

    @staticmethod
    def _same_identity(left: Any, right: Any) -> bool:
        try:
            same = getattr(left, "same_object", None)
            if not callable(same) or not bool(same(right)):
                return False
            reverse = getattr(right, "same_object", None)
            return not callable(reverse) or bool(reverse(left))
        except Exception:
            return False

    def project_lock(self, path: Path) -> Any:
        target = self._check_private_path(path)
        try:
            relative = target.relative_to(self._image_root)
        except ValueError as exc:
            raise OSError("private image lock path is invalid") from exc
        if not relative.parts:
            purpose = b"image-private-root-lock\0"
            prefix = "Local\\HermesFilmSuite-ImagePrivateRoot-"
            lock_prefix = ".image-private-root-lock-"
        elif self._image_root.name == "image-contexts" and len(relative.parts) == 1 and _CONTEXT_ID_RE.fullmatch(relative.parts[0]) is not None:
            purpose = b"image-context-lock\0"
            prefix = "Local\\HermesFilmSuite-ImageContext-"
            lock_prefix = ".image-context-lock-"
        else:
            raise OSError("private image lock path is invalid")
        handle, identity = self.open_existing(target, directory=True)
        handle.close()
        canonical = str(getattr(identity, "canonical_path", target))
        digest = hmac.new(
            self._profile_key,
            purpose + canonical.encode("utf-8", "surrogatepass"),
            hashlib.sha256,
        ).hexdigest()
        name = prefix + digest
        if os.name == "nt":
            return self._boundary_module.NamedProjectLock(name)
        self._boundary_module.ensure_private_node(self._lock_dir, directory=True)
        lock_file = self._lock_dir / (lock_prefix + digest)
        self._boundary_module.ensure_private_node(lock_file, directory=False)
        return self._boundary_module.NamedProjectLock(name, lock_file=lock_file)

    def link_no_replace(self, source: Any, parent: Any, name: str) -> None:
        if os.name == "nt":
            return self._boundary_module.win_link_handle_no_replace(source, parent, name)
        return self._boundary_module._posix_linkat_empty_path(int(source.raw), int(parent.raw), name)

    def win_link_handle_no_replace(self, source: Any, parent: Any, name: str) -> None:
        return self._boundary_module.win_link_handle_no_replace(source, parent, name)

    def posix_link_no_replace(self, source_fd: int, parent_fd: int, name: str) -> None:
        return self._boundary_module._posix_linkat_empty_path(source_fd, parent_fd, name)

    def flush_private_file(self, fd: int) -> None:
        return self._boundary_module._flush_private_file(fd)

    def fsync_parent(self, path: Path) -> None:
        return self._boundary_module._fsync_parent(self._check_private_path(path))

    def fsync_directory_handle(self, parent_handle: Any) -> None:
        return self._boundary_module._fsync_directory_handle(parent_handle)

    def unlink_handle_bound(self, path: Path, expected: Any, **kwargs: Any) -> bool:
        return self._boundary_module.unlink_handle_bound(self._check_private_path(path), expected, **kwargs)

    def dispose_exclusive_child(
        self,
        parent_handle: Any,
        parent_path: Path,
        name: str,
        child_handle: Any,
        expected: Any,
        **kwargs: Any,
    ) -> bool:
        self._check_private_child_name(name)
        return self._boundary_module.dispose_exclusive_child(
            parent_handle,
            self._check_private_path(parent_path),
            name,
            child_handle,
            expected,
            **kwargs,
        )


def _close_partial_producer(producer_jobs: Any | None, *, preserve_original_control: bool) -> None:
    if producer_jobs is None:
        return
    try:
        producer_jobs.close()
    except BaseException as cleanup_error:
        if preserve_original_control:
            return
        if isinstance(cleanup_error, Exception):
            raise ImageFilmCompositionError("image_film_unavailable") from None
        raise


class ProjectBundle:
    """Opaque per-project view whose supported boundary is root-scoped."""

    __slots__ = (
        "_root",
        "_closed",
        "_project_dir",
        "_project_identity",
        "_project_id",
        "_image_owner",
        "_film_owner",
        "_image_store",
        "_film_store",
        "_producer_jobs",
        "_producer_coordinator",
        "_attach",
        "_adoption",
    )

    def __init__(
        self,
        root: "ImageFilmCompositionRoot",
        project_dir: Path,
        project_identity: Any,
        project_id: str,
        image_owner: HmacImageOwnerAuthority,
        image_store: ImageStateStore,
        film_store: FilmBindingStore,
        producer_jobs: Any,
        attach: Any,
        adoption: FilmAdoptionService,
    ) -> None:
        self._root = root
        self._closed = False
        self._project_dir = project_dir
        self._project_identity = project_identity
        self._project_id = project_id
        self._image_owner = image_owner
        self._film_owner = image_owner
        self._image_store = image_store
        self._film_store = film_store
        self._producer_jobs = producer_jobs
        self._producer_coordinator: Any = None
        self._attach = attach
        self._adoption = adoption

    def _require_open(self) -> None:
        root = self._root
        if self._closed or root is None:
            _raise("root_closed")
        root._require_open()
        if self._closed or root._closed:
            _raise("root_closed")

    def _component(self, field_name: str) -> Any:
        root = self._root
        if self._closed or root is None:
            _raise("root_closed")
        with _ROOT_OPEN_LOCK:
            with root._lock:
                self._require_open()
                return getattr(self, field_name)

    def _invalidate(self) -> None:
        self._closed = True
        self._root = None  # type: ignore[assignment]
        self._project_dir = None  # type: ignore[assignment]
        self._project_identity = None
        self._image_owner = None  # type: ignore[assignment]
        self._film_owner = None  # type: ignore[assignment]
        self._image_store = None  # type: ignore[assignment]
        self._film_store = None  # type: ignore[assignment]
        self._producer_jobs = None
        self._producer_coordinator = None
        self._attach = None
        self._adoption = None  # type: ignore[assignment]

    @property
    def project_id(self) -> str:
        return self._component("_project_id")

    @property
    def image_owner(self) -> HmacImageOwnerAuthority:
        return self._component("_image_owner")

    @property
    def film_owner(self) -> HmacImageOwnerAuthority:
        return self._component("_film_owner")

    @property
    def image_store(self) -> ImageStateStore:
        return self._component("_image_store")

    @property
    def film_store(self) -> FilmBindingStore:
        return self._component("_film_store")

    @property
    def producer_jobs(self) -> Any:
        return self._component("_producer_jobs")

    @property
    def attach(self) -> Any:
        return self._component("_attach")

    @property
    def adoption(self) -> FilmAdoptionService:
        return self._component("_adoption")


class CachedProjectReadHandoff:
    """Read-only handoff for a project that is already cached by this root.

    The handoff intentionally exposes no store or mutating service.  It keeps
    the root and bundle locks around the liveness check and the actual inspect
    call, so a concurrent close cannot turn a successful check into a half-live
    read.  It never opens a project or constructs a project-scoped component.
    """

    __slots__ = ("_root", "_bundle", "_project_id")

    def __init__(self, root: "ImageFilmCompositionRoot", bundle: ProjectBundle) -> None:
        self._root = root
        self._bundle = bundle
        self._project_id = bundle._project_id

    @property
    def project_id(self) -> str:
        with _ROOT_OPEN_LOCK:
            with self._root._lock:
                self._root._require_open()
                self._bundle._require_open()
                return self._project_id

    def inspect(
        self,
        binding_id: str | None = None,
        film_reference_id: str | None = None,
    ) -> dict[str, Any]:
        with _ROOT_OPEN_LOCK:
            with self._root._lock:
                self._root._require_open()
                self._bundle._require_open()
                adoption = self._bundle._adoption
                if adoption is None:
                    _raise("root_closed")
                return adoption.inspect(binding_id, film_reference_id)


def _result_services(root: "ImageFilmCompositionRoot") -> tuple[Any, Any]:
    """The profile's one result collector and its material reader.

    One collector for the whole profile, so its per-job lock really is the
    only way a job is fetched and published; the job ledger is profile-wide.
    Caller holds ``_ROOT_OPEN_LOCK`` and ``root._lock``.
    """
    root._require_open()
    reader = root._image_material_reader
    ingestor = root._image_ingestor
    jobs = root._producer_jobs
    if reader is None or ingestor is None or jobs is None:
        _raise("root_closed")
    collector = root._result_collector
    if collector is None:
        try:
            client_type = _generation_component("comfyui_client", "ComfyUIClient")
            collector_type = _generation_component("image_collect", "ImageResultCollector")
            # The client pins its own loopback origin; do not widen it here.
            fetch = client_type().view
            try:  # w6-gptimage: GPT Image pictures come from the subscription runtime, never from ComfyUI's /view
                fetch = _generation_component("paid.subscription_image_provider", "wrap_fetch")(fetch)
            except Exception:  # noqa: BLE001 - provider module absent: ComfyUI pictures only
                pass
            collector = collector_type(job_store=jobs, ingestor=ingestor, fetch=fetch)
        except Exception as exc:  # noqa: BLE001 - finite composition boundary
            raise _map_exception(exc, default="image_state_unavailable") from None
        root._result_collector = collector
    return collector, reader


class ScratchOperationHandoff:
    """Image work with no project open.

    Images and jobs are profile-scoped storage already; a project only ever
    narrowed who may reach them.  Someone trying something out must not have to
    create a project first, so this hands out the same profile-owned stores
    under an owner scope of its own.

    It is a SEPARATE place, not a shortcut into a project: it carries no
    project id, so nothing it creates can land in a project's catalog and a
    project's work never shows up here.
    """

    __slots__ = ("_root",)

    def __init__(self, root: "ImageFilmCompositionRoot") -> None:
        self._root = root

    def image_states(self) -> ImageStateStore:
        with _ROOT_OPEN_LOCK:
            with self._root._lock:
                self._root._require_open()
                store = self._root._image_store
                index = self._root._context_owners
                if store is None or index is None:
                    _raise("root_closed")
                return ScopedImageStates(store, index, SCRATCH_OWNER)

    def producer_jobs(self) -> Any:
        with _ROOT_OPEN_LOCK:
            with self._root._lock:
                self._root._require_open()
                jobs = self._root._producer_jobs
                if jobs is None:
                    _raise("root_closed")
                return jobs

    def producer_coordinator(self) -> Any:
        """Build the scratch coordinator once, like a project's own."""
        with _ROOT_OPEN_LOCK:
            with self._root._lock:
                self._root._require_open()
                existing = self._root._scratch_coordinator
                if existing is not None:
                    return existing
                jobs = self._root._producer_jobs
                if jobs is None:
                    _raise("root_closed")
                try:
                    client_type = _generation_component("comfyui_client", "ComfyUIClient")
                    adapter_type = _generation_component("krea2_create_adapter", "Krea2CreateAdapter")
                    coordinator_type = _generation_component("image_coordinator", "ImageProducerCoordinator")
                    client = client_type()
                    coordinator = coordinator_type(jobs, _with_edit_routes(_live_preview_adapter(adapter_type, client), client, jobs, self._root._image_material_reader))
                except Exception as exc:  # noqa: BLE001 - finite composition boundary
                    raise _map_exception(exc, default="image_state_unavailable") from None
                self._root._scratch_coordinator = coordinator
                return coordinator

    def result_services(self) -> tuple[Any, Any]:
        """Collector (ready -> published) and reader for finished runs."""
        with _ROOT_OPEN_LOCK:
            with self._root._lock:
                return _result_services(self._root)


class ProjectMediaReadHandoff:
    """Read-only view of what one project owner scope has made, without opening it.

    The Start screen lists recent projects with a picture and an item count
    before any of them is opened in this session.  Images and jobs are
    profile-wide stores; a project only narrows ownership.  This hands out the
    same stores scoped to ``film_owner:<project_id>`` for reading: no project
    directory is touched, no bundle is built, nothing is submitted through it.
    """

    __slots__ = ("_root", "_owner")

    def __init__(self, root: "ImageFilmCompositionRoot", project_id: str) -> None:
        self._root = root
        self._owner = "film_owner:" + project_id

    def image_states(self) -> ImageStateStore:
        with _ROOT_OPEN_LOCK:
            with self._root._lock:
                self._root._require_open()
                store = self._root._image_store
                index = self._root._context_owners
                if store is None or index is None:
                    _raise("root_closed")
                return ScopedImageStates(store, index, self._owner)

    def producer_jobs(self) -> Any:
        with _ROOT_OPEN_LOCK:
            with self._root._lock:
                self._root._require_open()
                jobs = self._root._producer_jobs
                if jobs is None:
                    _raise("root_closed")
                return jobs

    def result_services(self) -> tuple[Any, Any]:
        with _ROOT_OPEN_LOCK:
            with self._root._lock:
                return _result_services(self._root)


class CachedProjectOperationHandoff:
    """Opaque handoff for bounded, session-scoped Image/Film operations.

    The handoff exposes only the accepted application service methods.  It does
    not expose a store, authority, project path, factory, or caller identity;
    the caller scope is supplied separately by the HTTP route.
    """

    __slots__ = ("_root", "_bundle", "_project_id")

    def __init__(self, root: "ImageFilmCompositionRoot", bundle: ProjectBundle) -> None:
        self._root = root
        self._bundle = bundle
        self._project_id = bundle._project_id

    @property
    def project_id(self) -> str:
        with _ROOT_OPEN_LOCK:
            with self._root._lock:
                self._root._require_open()
                self._bundle._require_open()
                return self._project_id

    def producer_coordinator(self) -> Any:
        """Return this project's generation coordinator, built once.

        The provider adapter and its loopback client are chosen here, in the
        composition root, never at the HTTP edge: a route must not decide
        provider identity or network authority.  Built lazily so opening a
        project stays free of provider work, and cached so two callers cannot
        hold two descriptor caches for the same jobs.
        """
        with _ROOT_OPEN_LOCK:
            with self._root._lock:
                self._root._require_open()
                self._bundle._require_open()
                existing = self._bundle._producer_coordinator
                if existing is not None:
                    return existing
                jobs = self._bundle._producer_jobs
                if jobs is None:
                    _raise("root_closed")
                try:
                    client_type = _generation_component("comfyui_client", "ComfyUIClient")
                    adapter_type = _generation_component("krea2_create_adapter", "Krea2CreateAdapter")
                    coordinator_type = _generation_component("image_coordinator", "ImageProducerCoordinator")
                    # The client pins its own loopback origin; do not widen it here.
                    client = client_type()
                    coordinator = coordinator_type(jobs, _with_edit_routes(_live_preview_adapter(adapter_type, client), client, jobs, self._root._image_material_reader))
                except Exception as exc:  # noqa: BLE001 - finite composition boundary
                    raise _map_exception(exc, default="image_state_unavailable") from None
                self._bundle._producer_coordinator = coordinator
                return coordinator

    def result_services(self) -> tuple[Any, Any]:
        """Collector (ready -> published) and reader for finished runs."""
        with _ROOT_OPEN_LOCK:
            with self._root._lock:
                self._bundle._require_open()
                return _result_services(self._root)

    def image_states(self) -> ImageStateStore:
        """Return this project's image state store.

        Same guard as the other operation accessors: the root and the bundle
        must both still be open, so a closed project cannot hand out a store
        that would write into a directory nobody owns anymore.
        """
        with _ROOT_OPEN_LOCK:
            with self._root._lock:
                self._root._require_open()
                self._bundle._require_open()
                store = self._bundle._image_store
                index = self._root._context_owners
                if store is None or index is None:
                    _raise("root_closed")
                return ScopedImageStates(store, index, "film_owner:" + self._bundle._project_id)

    def producer_jobs(self) -> Any:
        """Return the cached bundle's own generation job store.

        Generation is a project-scoped operation like adoption, so it must run
        on the store the root already opened.  A caller that built its own
        store would own a second ledger for the same project; that is why this
        accessor exists instead of letting a route construct one.  It hands
        out no publication path and performs no filesystem work.
        """
        with _ROOT_OPEN_LOCK:
            with self._root._lock:
                self._root._require_open()
                self._bundle._require_open()
                producer = self._bundle._producer_jobs
                if producer is None:
                    _raise("root_closed")
                return producer

    def preview(self, request: dict[str, Any], caller_scope_id: str) -> dict[str, Any]:
        with _ROOT_OPEN_LOCK:
            with self._root._lock:
                self._root._require_open()
                self._bundle._require_open()
                adoption = self._bundle._adoption
                if adoption is None:
                    _raise("root_closed")
                return adoption.preview(request, caller_scope_id)

    def cancel(self, request: dict[str, Any], caller_scope_id: str) -> dict[str, Any]:
        with _ROOT_OPEN_LOCK:
            with self._root._lock:
                self._root._require_open()
                self._bundle._require_open()
                adoption = self._bundle._adoption
                if adoption is None:
                    _raise("root_closed")
                return adoption.cancel(request, caller_scope_id)

    def prepare(self, request: dict[str, Any], caller_scope_id: str) -> dict[str, Any]:
        with _ROOT_OPEN_LOCK:
            with self._root._lock:
                self._root._require_open()
                self._bundle._require_open()
                adoption = self._bundle._adoption
                if adoption is None:
                    _raise("root_closed")
                return adoption.prepare(request, caller_scope_id)

    def execute(self, request: dict[str, Any], caller_scope_id: str) -> dict[str, Any]:
        with _ROOT_OPEN_LOCK:
            with self._root._lock:
                self._root._require_open()
                self._bundle._require_open()
                adoption = self._bundle._adoption
                if adoption is None:
                    _raise("root_closed")
                return adoption.execute(request, caller_scope_id)

    def status(self, request: dict[str, Any], caller_scope_id: str) -> dict[str, Any]:
        with _ROOT_OPEN_LOCK:
            with self._root._lock:
                self._root._require_open()
                self._bundle._require_open()
                adoption = self._bundle._adoption
                if adoption is None:
                    _raise("root_closed")
                return adoption.status(request, caller_scope_id)

class ImageFilmCompositionRoot:
    """Own one profile's Image and Film authorities and shared services."""

    __slots__ = (
        "_closed",
        "_lock",
        "_profile_home",
        "_profile_anchor",
        "_profile_boundary",
        "_image_boundary",
        "_image_material_boundary",
        "_image_material_root",
        "_image_owner",
        "_image_store",
        "_producer_jobs",
        "_image_ingestor",
        "_image_material_reader",
        "_scratch_coordinator",
        "_context_owners",
        "_result_collector",
        "_project_store",
        "_bundles",
    )

    def __init__(self) -> None:
        raise TypeError("ImageFilmCompositionRoot is built by ImageFilmCompositionRoot.open()")

    @classmethod
    def open(cls) -> "ImageFilmCompositionRoot":
        with _ROOT_OPEN_LOCK:
            return cls._open_unlocked()

    @classmethod
    def _open_unlocked(cls) -> "ImageFilmCompositionRoot":
        producer_jobs: Any | None = None
        try:
            initial = _stable_active_home()
            if initial is None:
                _raise("image_film_unavailable")
            if initial[0] is None:
                _raise("image_key_unavailable")
            home, initial_path, initial_anchor = initial
            if initial_anchor is None:
                # The first call is allowed to create the missing profile, but
                # its result is deliberately discarded.  Pin the now-existing
                # native home, then take a separate authoritative read under
                # that pin's identity envelope.
                ReceiptKeyEnrollment.open(home)
                try:
                    profile_anchor = _profile_anchor_for_home(home)
                except Exception:
                    _raise("image_film_unavailable")
                if profile_anchor is None:
                    _raise("image_state_unavailable")
                enrollment = ReceiptKeyEnrollment.open(home)
                try:
                    authoritative_anchor = _profile_anchor_for_home(home)
                except Exception:
                    _raise("image_film_unavailable")
                if not _same_profile_anchor(profile_anchor, authoritative_anchor):
                    _raise("image_film_unavailable")
            else:
                # Existing profiles are already pinned before the one
                # authoritative enrollment load.
                enrollment = ReceiptKeyEnrollment.open(home)
            evidence = enrollment.evidence
            key = enrollment.key
            if type(key) is not bytes or len(key) != 32:
                _raise("image_key_unavailable")
            profile_boundary = path_boundary.PathBoundary(hermes_home=home)
            profile_home = Path(profile_boundary.hermes_home)
            try:
                current_anchor = _profile_anchor_for_home(profile_home)
                profile_home_path = _normalised_profile_path(profile_home)
            except Exception:
                _raise("image_film_unavailable")
            if initial_anchor is not None:
                if not _same_profile_anchor(initial_anchor, current_anchor):
                    _raise("image_film_unavailable")
            elif not _same_profile_anchor(profile_anchor, current_anchor):
                _raise("image_film_unavailable")
            profile_anchor = current_anchor
            if profile_anchor is None:
                _raise("image_state_unavailable")
            if profile_home_path != initial_path:
                _raise("image_film_unavailable")
            image_root = state_root(profile_home) / "image-contexts"
            image_boundary = _PrivateImageBoundary(profile_boundary, key, image_root)
            image_boundary._boundary_module.ensure_private_node(image_root.parent, directory=True)
            image_boundary.ensure_private_node(image_root, directory=True)
            # Generated bytes have one profile-global private root.  Resolve it
            # only after the receipt key and the active profile identity are
            # pinned, before the producer store is constructed.
            image_material_root = state_root(profile_home / "plugin-data") / "image-materials"
            image_material_boundary = _PrivateImageBoundary(
                profile_boundary,
                key,
                image_material_root,
            )
            image_material_boundary._boundary_module.ensure_private_node(
                image_material_root.parent,
                directory=True,
            )
            image_material_boundary.ensure_private_node(image_material_root, directory=True)
            image_material_boundary.ensure_private_node(
                image_material_root / "materials",
                directory=True,
            )
            image_material_boundary.ensure_private_node(
                image_material_root / ".image-staging",
                directory=True,
            )
            image_owner = HmacImageOwnerAuthority(
                key,
                evidence.enrollment_id,
                evidence.receipt_key_fingerprint,
            )
            image_store = ImageStateStore(
                image_root,
                boundary=image_boundary,
                authority=image_owner,
            )
            job_store_type = _generation_component("image_job_store", "ImageProducerJobStore")
            producer_jobs = job_store_type(
                state_root(profile_home / "plugin-data") / "image-producer-jobs.sqlite3"
            )
            context_owners = ImageContextOwnerIndex(
                state_root(profile_home / "plugin-data") / "image-context-owners.sqlite3"
            )
            project_store = ProjectStore(boundary=profile_boundary)
            ingestor_type = _generation_component("image_ingest", "ImageResultIngestor")
            material_reader_type = _generation_component(
                "image_material_read",
                "ImagePublishedMaterialReadAuthority",
            )
            image_ingestor = ingestor_type(
                job_store=producer_jobs,
                material_root=image_material_root,
                boundary=image_material_boundary,
            )
            image_material_reader = material_reader_type(
                image_store=image_store,
                job_store=producer_jobs,
                material_root=image_material_root,
                boundary=image_material_boundary,
            )
            root = object.__new__(cls)
            root._closed = False
            root._lock = threading.RLock()
            root._profile_home = profile_home
            root._profile_anchor = profile_anchor
            root._profile_boundary = profile_boundary
            root._image_boundary = image_boundary
            root._image_material_boundary = image_material_boundary
            root._image_material_root = image_material_root
            root._image_owner = image_owner
            root._image_store = image_store
            root._producer_jobs = producer_jobs
            root._image_ingestor = image_ingestor
            root._image_material_reader = image_material_reader
            root._project_store = project_store
            root._scratch_coordinator = None
            root._context_owners = context_owners
            root._result_collector = None
            root._bundles = {}
            final = _stable_active_profile_snapshot()
            if (
                final is None
                or initial_path != final.path
                or not _same_profile_anchor(profile_anchor, final.anchor)
            ):
                raise OSError("profile home changed during construction")
            return root
        except ImageFilmCompositionError as exc:
            _close_partial_producer(producer_jobs, preserve_original_control=False)
            raise exc
        except (ReceiptKeyEnrollmentError, ImageStateError, path_boundary.ProjectError, FilmBindingStoreError) as exc:
            _close_partial_producer(producer_jobs, preserve_original_control=False)
            raise _map_exception(exc, default="image_state_unavailable") from None
        except Exception as exc:
            _close_partial_producer(producer_jobs, preserve_original_control=False)
            raise _map_exception(exc) from None
        except BaseException:
            _close_partial_producer(producer_jobs, preserve_original_control=True)
            raise

    def _require_open(self) -> None:
        with _ROOT_OPEN_LOCK:
            with self._lock:
                if self._closed:
                    _raise("root_closed")
                try:
                    current = _stable_active_profile_snapshot()
                except Exception as exc:
                    try:
                        self._invalidate_for_profile_failure()
                    except BaseException:
                        # Cleanup failures are observable: process-control remains
                        # unchanged and ordinary close failures are already finite.
                        raise
                    raise _map_exception(exc, default="image_film_unavailable") from None
                except BaseException:
                    try:
                        self._invalidate_for_profile_failure()
                    except BaseException:
                        # Preserve the original process-control exception after the
                        # one permitted cleanup attempt.
                        pass
                    raise
                if (
                    current is None
                    or current.path != getattr(self._profile_anchor, "path", None)
                    or not _same_profile_anchor(self._profile_anchor, current.anchor)
                ):
                    self._invalidate_for_profile_failure()
                    _raise("image_film_unavailable")

    def _invalidate_for_profile_failure(self) -> None:
        self.close()

    @staticmethod
    def _same_identity(left: Any, right: Any) -> bool:
        try:
            same = getattr(left, "same_object", None)
            if not callable(same) or not bool(same(right)):
                return False
            reverse = getattr(right, "same_object", None)
            return not callable(reverse) or bool(reverse(left))
        except Exception:
            return False

    @staticmethod
    def _cache_path(identity: Any) -> str | None:
        try:
            value = getattr(identity, "canonical_path")
        except Exception:
            return None
        return value if type(value) is str else None

    @staticmethod
    def _cache_identity(identity: Any) -> tuple[Any, Any, bool]:
        try:
            volume = getattr(identity, "volume_serial")
            file_id = getattr(identity, "file_id")
            is_directory = getattr(identity, "is_directory")
            value = (volume, file_id, bool(is_directory))
            hash(value)
            return value
        except Exception:
            _raise("project_unavailable")

    def _discard_bundle(self, key: tuple[str, Any]) -> None:
        bundle = self._bundles.pop(key, None)
        if bundle is not None:
            bundle._invalidate()

    def open_project(self, project_dir: str | Path, *, expected_project_id: str) -> ProjectBundle:
        with _ROOT_OPEN_LOCK:
            with self._lock:
                self._require_open()
                expected = _strict_project_id(expected_project_id)
                try:
                    opened_dir = self._profile_boundary.validate_project_directory(project_dir, operation="open")
                    identity = self._profile_boundary.pin_directory(opened_dir)
                    manifest = self._project_store.read_project(opened_dir)
                    trusted_id = manifest.get("project_id") if isinstance(manifest, dict) else None
                    if trusted_id != expected:
                        _raise("project_scope_mismatch")
                    self._profile_boundary.revalidate_directory(opened_dir, identity)
                except ImageFilmCompositionError:
                    raise
                except Exception as exc:
                    raise _map_exception(exc, default="project_unavailable") from None

                current_path = self._cache_path(identity)
                key = (expected, self._cache_identity(identity))
                for cached_key, cached_bundle in tuple(self._bundles.items()):
                    if cached_key == key:
                        continue
                    old_path = self._cache_path(cached_bundle._project_identity)
                    if current_path is not None and old_path == current_path:
                        self._discard_bundle(cached_key)
                        _raise("project_unavailable")
                    # Lane w30-fix3 item 2 (BCK-221): the same project_id at another live folder (a file-system copy of
                    # the project) closes the older location's bundle cleanly, so exactly one location of a project is
                    # open and Image reads it; the bundle of the other folder is rebuilt when that folder is opened again.
                    if cached_key[0] == expected:
                        self._discard_bundle(cached_key)

                cached = self._bundles.get(key)
                if cached is not None:
                    try:
                        self._profile_boundary.revalidate_directory(cached._project_dir, cached._project_identity)
                        fresh = self._project_store.read_project(cached._project_dir)
                        if fresh.get("project_id") != expected:
                            self._discard_bundle(key)
                            _raise("project_scope_mismatch")
                        return cached
                    except ImageFilmCompositionError:
                        raise
                    except Exception as exc:
                        self._discard_bundle(key)
                        raise _map_exception(exc, default="project_unavailable") from None

                film_store: Any = None
                adoption: Any = None
                attach: Any = None
                bundle: ProjectBundle | None = None
                try:
                    film_store = FilmBindingStore(
                        opened_dir,
                        boundary=self._profile_boundary,
                        authority=self._image_owner,
                        expected_project_id=expected,
                    )
                    adoption_type = FilmAdoptionService
                    adoption = adoption_type(self._image_store, film_store)
                    attach_type = _generation_component("image_attach", "ImageAttachService")
                    attach = attach_type(self._producer_jobs, self._image_store)
                    bundle = ProjectBundle(
                        self,
                        opened_dir,
                        identity,
                        expected,
                        self._image_owner,
                        self._image_store,
                        film_store,
                        self._producer_jobs,
                        attach,
                        adoption,
                    )
                    final_manifest = self._project_store.read_project(opened_dir)
                    if not isinstance(final_manifest, dict) or final_manifest.get("project_id") != expected:
                        _raise("project_scope_mismatch")
                    self._profile_boundary.revalidate_directory(opened_dir, identity)
                    self._bundles[key] = bundle
                    return bundle
                except ImageFilmCompositionError:
                    if bundle is not None:
                        bundle._invalidate()
                    raise
                except Exception as exc:
                    if bundle is not None:
                        bundle._invalidate()
                    raise _map_exception(exc, default="image_film_unavailable") from None
                except BaseException:
                    if bundle is not None:
                        bundle._invalidate()
                    raise

    def profile_image_source_catalog(self, caller_scope_id: str) -> dict[str, Any]:
        """Return the bounded profile-global Image source projection.

        This is the only catalog handoff.  It accepts only the already resolved
        caller scope identifier and exposes neither the store nor any profile
        filesystem capability to the route layer.
        """
        if type(caller_scope_id) is not str or _CALLER_SCOPE_RE.fullmatch(caller_scope_id) is None:
            _raise("image_catalog_unavailable")
        with _ROOT_OPEN_LOCK:
            with self._lock:
                self._require_open()
                store = self._image_store
                if store is None:
                    _raise("image_catalog_unavailable")
                try:
                    return build_catalog(store.enumerate_contexts())
                except ImageCatalogError as exc:
                    raise _map_exception(
                        exc,
                        default="image_catalog_state_unavailable",
                    ) from None
                except ImageStateError as exc:
                    if exc.code in {
                        "image-contexts-root-identity-changed",
                        "context-identity-changed",
                        "state-identity-changed",
                        "state-file-identity-changed",
                        "state-parent-identity-changed",
                    }:
                        raise ImageFilmCompositionError("image_catalog_identity_changed") from None
                    if exc.code in {"context-count-exceeded", "state-file-count-exceeded"}:
                        raise ImageFilmCompositionError("image_catalog_limit_exceeded") from None
                    raise ImageFilmCompositionError("image_catalog_state_unavailable") from None
                except Exception as exc:
                    raise _map_exception(
                        exc,
                        default="image_catalog_state_unavailable",
                    ) from None

    def scratch_operations(self) -> ScratchOperationHandoff:
        """Hand out the profile-owned stores for work with no project open."""
        with self._lock:
            self._require_open()
        return ScratchOperationHandoff(self)

    def read_cached_project(self, expected_project_id: str) -> CachedProjectReadHandoff:
        """Return a read-only handoff for one already-opened project.

        This is deliberately distinct from :meth:`open_project`.  Read-only
        inspection must not resolve a recent-project pointer, open a new
        project, consult ``ProjectStore``, construct
        ``FilmBindingStore``/adoption/attach services, or create any
        filesystem node.  A project is readable here only after a trusted
        caller has already populated this root's bundle cache.
        """
        with _ROOT_OPEN_LOCK:
            with self._lock:
                self._require_open()
                expected = _strict_project_id(expected_project_id)
                matches = [
                    bundle
                    for (project_id, _identity), bundle in self._bundles.items()
                    if project_id == expected
                ]
                if not matches:
                    _raise("project_unavailable")
                if len(matches) != 1:
                    _raise("project_unavailable")
                bundle = matches[0]
                try:
                    bundle._require_open()
                    self._profile_boundary.revalidate_directory(
                        bundle._project_dir,
                        bundle._project_identity,
                    )
                except ImageFilmCompositionError:
                    raise
                except Exception as exc:
                    self._discard_bundle((expected, self._cache_identity(bundle._project_identity)))
                    raise _map_exception(exc, default="project_unavailable") from None
                return CachedProjectReadHandoff(self, bundle)

    def read_project_media(self, expected_project_id: str) -> ProjectMediaReadHandoff:
        """Read-only media of one project owner scope, opened or not (see ProjectMediaReadHandoff)."""
        with self._lock:
            self._require_open()
            expected = _strict_project_id(expected_project_id)
        return ProjectMediaReadHandoff(self, expected)

    def read_cached_project_operations(self, expected_project_id: str) -> CachedProjectOperationHandoff:
        """Return an opaque handoff for bounded session-only Image/Film calls.

        Like :meth:`read_cached_project`, this never opens a project or creates
        a component.  The separate handoff keeps the inspect surface read-only
        while allowing the Preview/Cancel route to use the already accepted
        service without exposing the bundle or its stores.
        """
        read_handoff = self.read_cached_project(expected_project_id)
        return CachedProjectOperationHandoff(read_handoff._root, read_handoff._bundle)

    def close(self) -> None:
        with _ROOT_OPEN_LOCK:
            with self._lock:
                if self._closed:
                    return
                self._closed = True
                for bundle in tuple(self._bundles.values()):
                    bundle._invalidate()
                self._bundles.clear()
                producer_jobs = self._producer_jobs
                self._producer_jobs = None  # type: ignore[assignment]
                self._image_store = None  # type: ignore[assignment]
                self._image_owner = None  # type: ignore[assignment]
                self._image_boundary = None  # type: ignore[assignment]
                self._image_material_boundary = None  # type: ignore[assignment]
                self._image_material_root = None  # type: ignore[assignment]
                self._image_ingestor = None  # type: ignore[assignment]
                self._result_collector = None
                self._image_material_reader = None  # type: ignore[assignment]
                self._project_store = None  # type: ignore[assignment]
                self._profile_boundary = None  # type: ignore[assignment]
                self._profile_home = None  # type: ignore[assignment]
                self._profile_anchor = None
                owners = getattr(self, "_context_owners", None)
                self._context_owners = None
                if owners is not None:
                    owners.close()
                if producer_jobs is None:
                    return
                try:
                    producer_jobs.close()
                except Exception:
                    raise ImageFilmCompositionError("image_film_unavailable") from None


__all__ = [
    "CachedProjectReadHandoff",
    "ImageFilmCompositionError",
    "ImageFilmCompositionRoot",
    "ProjectBundle",
]
