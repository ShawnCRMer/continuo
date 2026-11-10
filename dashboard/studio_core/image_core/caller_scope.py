"""Resolve the only caller principals trusted by Image/Film authority.

Hermes' hosted authentication middleware exposes a provider-verified
``request.state.session``.  A Hermes Desktop backend is different: it runs on
loopback and the legacy dashboard gate authenticates the Desktop shell with
the operator-minted ``X-Hermes-Session-Token``.  That gate does not attach a
session object, so this module calls its existing verifier through a narrow
adapter and additionally requires the Desktop/loopback server envelope.

Loopback locality, an arbitrary request-state flag, a token-shaped value, or a
service ``TokenPrincipal`` is never enough.  The returned identifier is an
opaque principal scope for the Image/Film session authority; it is not a
browser, conversation, or user-session identifier.
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
import hashlib
import json
import os
import secrets
import threading
from dataclasses import dataclass
from typing import Any


_CALLER_SCOPE_SCHEMA = "studio.image-film-caller-scope/1"
_HOSTED_SCOPE_KIND = "host_session_principal"
_DESKTOP_SCOPE_KIND = "hermes_desktop_loopback_principal"
_DESKTOP_SCOPE_PROVIDER = "hermes_desktop"
_DESKTOP_SCOPE_PRINCIPAL = "desktop"
_CALLER_SCOPE_PREFIX = "imgcaller_"
_MAX_ID_TEXT = 256
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_DESKTOP_SCOPE_NONCE = None
_DESKTOP_SCOPE_AUTH_STATE = None
_DESKTOP_SCOPE_NONCE_LOCK = threading.Lock()
_AGENT_SENTINEL = object()
_AGENT_CALL = contextvars.ContextVar("studio_trusted_agent_call", default=None)


@contextmanager
def trusted_agent_call():
    # Child asyncio tasks inherit context, including after the parent resets it.
    # External server tasks and threads started outside this call do not.
    token = _AGENT_CALL.set(_AGENT_SENTINEL)
    try:
        yield
    finally:
        _AGENT_CALL.reset(token)


class ImageFilmCallerScopeError(ValueError):
    """Finite fail-closed error for the host caller identity boundary."""

    CODES = frozenset({"caller_identity_unavailable", "caller_identity_invalid"})

    def __init__(self, code: str) -> None:
        if type(code) is not str or code not in self.CODES:
            code = "caller_identity_unavailable"
        self.code = code
        self.reason_code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class TrustedImageFilmCallerScope:
    """Opaque Image/Film authority scope derived from one verified principal.

    ``user_id`` is populated only for a hosted user principal.  Desktop is an
    explicitly authenticated local application principal, not a hosted user,
    so it carries ``principal_id="desktop"`` and an empty user/org pair.
    """

    scope_id: str
    provider: str
    org_id: str
    user_id: str

    @property
    def principal_kind(self) -> str:
        if self.provider == _DESKTOP_SCOPE_PROVIDER and not self.org_id and not self.user_id:
            return _DESKTOP_SCOPE_KIND
        return _HOSTED_SCOPE_KIND

    @property
    def principal_id(self) -> str:
        return _DESKTOP_SCOPE_PRINCIPAL if self.principal_kind == _DESKTOP_SCOPE_KIND else self.user_id

    @property
    def kind(self) -> str:
        """Compatibility alias for the canonical principal-kind label."""
        return self.principal_kind


def _fail(code: str) -> Any:
    raise ImageFilmCallerScopeError(code) from None


def _read_request_auth_state(request: Any) -> Any:
    """Read only Hermes-owned authentication state from ``request``.

    The loopback token gate intentionally does not populate ``state.session``.
    ``None`` therefore means "no hosted session", not "authenticated Desktop";
    Desktop authentication is established separately by the core verifier.
    """
    if request is None:
        _fail("caller_identity_unavailable")
    try:
        state = getattr(request, "state")
    except Exception:
        _fail("caller_identity_unavailable")
    try:
        token_principal = getattr(state, "token_principal", None)
        token_authenticated = getattr(state, "token_authenticated", False)
    except Exception:
        _fail("caller_identity_unavailable")
    # Image/Film accepts the interactive Hosted session and the explicitly
    # authenticated Desktop loopback principal only.  A service principal is a
    # different trust contract, even if Hermes has already verified it.
    if token_principal is not None or token_authenticated is not False:
        _fail("caller_identity_unavailable")
    try:
        session = getattr(state, "session", None)
    except Exception:
        _fail("caller_identity_unavailable")
    return state, session


def _verified_session_type() -> type[Any]:
    try:
        from hermes_cli.dashboard_auth.base import Session
    except Exception:
        _fail("caller_identity_unavailable")
    return Session


def _desktop_loopback_request_is_authenticated(request: Any) -> str:
    """Ask Hermes' existing loopback gate whether this request has its token.

    This is deliberately an adapter around the host's verifier, not a second
    header parser.  The host owns the token value, comparison policy, and
    legacy Bearer compatibility path.  If that seam is absent or fails, this
    boundary fails closed rather than treating loopback as an identity.
    """
    try:
        app = getattr(request, "app")
        app_state = getattr(app, "state")
        bound_host = getattr(app_state, "bound_host")
        auth_required = getattr(app_state, "auth_required")
    except Exception:
        _fail("caller_identity_unavailable")
    if type(bound_host) is not str or bound_host not in _LOOPBACK_HOSTS:
        _fail("caller_identity_unavailable")
    if auth_required is not False or os.environ.get("HERMES_DESKTOP") != "1":
        _fail("caller_identity_unavailable")
    try:
        from hermes_cli import web_server

        verifier = getattr(web_server, "_has_valid_session_token")
        authenticated = verifier(request)
    except Exception:
        _fail("caller_identity_unavailable")
    if type(authenticated) is not bool or not authenticated:
        _fail("caller_identity_unavailable")
    # Only a verified Desktop HTTP request may establish the process scope.
    global _DESKTOP_SCOPE_NONCE, _DESKTOP_SCOPE_AUTH_STATE
    try:
        with _DESKTOP_SCOPE_NONCE_LOCK:
            if _DESKTOP_SCOPE_NONCE is None:
                _DESKTOP_SCOPE_NONCE = secrets.token_hex(32)
            _DESKTOP_SCOPE_AUTH_STATE = app_state
            nonce = _DESKTOP_SCOPE_NONCE
    except Exception:
        _fail("caller_identity_unavailable")
    if type(nonce) is not str or len(nonce) != 64:
        _fail("caller_identity_unavailable")
    return nonce


def _scope_from_projection(
    projection: dict[str, str],
    *,
    provider: str,
    org_id: str,
    user_id: str,
) -> TrustedImageFilmCallerScope:
    try:
        canonical = json.dumps(
            projection,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, UnicodeError, ValueError):
        _fail("caller_identity_invalid")
    scope_id = _CALLER_SCOPE_PREFIX + hashlib.sha256(canonical).hexdigest()
    return TrustedImageFilmCallerScope(
        scope_id=scope_id,
        provider=provider,
        org_id=org_id,
        user_id=user_id,
    )


def _identity_text(session: Any, field: str, *, allow_empty: bool = False) -> str:
    try:
        value = object.__getattribute__(session, field)
    except Exception:
        _fail("caller_identity_invalid")
    if type(value) is not str:
        _fail("caller_identity_invalid")
    if not allow_empty and not value:
        _fail("caller_identity_invalid")
    if len(value) > _MAX_ID_TEXT or "\x00" in value:
        _fail("caller_identity_invalid")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        _fail("caller_identity_invalid")
    try:
        value.encode("utf-8")
    except UnicodeError:
        _fail("caller_identity_invalid")
    return value


def resolve_trusted_image_film_caller_scope(request: Any) -> TrustedImageFilmCallerScope:
    """Resolve a verified Hosted or authenticated Desktop caller on ``request``.

    Hosted requests use ``request.state.session``, which Hermes populates only
    after provider verification.  Desktop loopback requests use the existing
    Hermes session-token verifier, but only when the server is explicitly the
    Hermes Desktop loopback backend.  No request JSON, caller-provided session
    ID, arbitrary state marker, locality, or service ``TokenPrincipal`` can
    create a scope.
    """
    _state, session = _read_request_auth_state(request)
    if session is not None:
        session_type = _verified_session_type()
        if type(session) is not session_type:
            _fail("caller_identity_invalid")

        provider = _identity_text(session, "provider")
        org_id = _identity_text(session, "org_id", allow_empty=True)
        user_id = _identity_text(session, "user_id")
        return _scope_from_projection(
            {
                "schema": _CALLER_SCOPE_SCHEMA,
                "kind": _HOSTED_SCOPE_KIND,
                "provider": provider,
                "org_id": org_id,
                "user_id": user_id,
            },
            provider=provider,
            org_id=org_id,
            user_id=user_id,
        )

    if _AGENT_CALL.get() is _AGENT_SENTINEL:
        # A marker is not an identity: it can only reuse an established scope.
        if os.environ.get("HERMES_DESKTOP") != "1":
            _fail("caller_identity_unavailable")
        if getattr(getattr(getattr(request, "app", None), "state", None), "auth_required", None) is True:
            _fail("caller_identity_unavailable")
        with _DESKTOP_SCOPE_NONCE_LOCK:
            desktop_server_instance = _DESKTOP_SCOPE_NONCE
            verified_state = _DESKTOP_SCOPE_AUTH_STATE
        try:
            bound_host = getattr(verified_state, "bound_host", None)
            if type(bound_host) is not str or bound_host not in _LOOPBACK_HOSTS or getattr(verified_state, "auth_required", None) is not False:
                _fail("caller_identity_unavailable")
        except ImageFilmCallerScopeError:
            raise
        except Exception:
            _fail("caller_identity_unavailable")
        if type(desktop_server_instance) is not str or len(desktop_server_instance) != 64:
            _fail("caller_identity_unavailable")
    else:
        desktop_server_instance = _desktop_loopback_request_is_authenticated(request)
    return _scope_from_projection(
        {
            "schema": _CALLER_SCOPE_SCHEMA,
            "kind": _DESKTOP_SCOPE_KIND,
            "provider": _DESKTOP_SCOPE_PROVIDER,
            "principal_id": _DESKTOP_SCOPE_PRINCIPAL,
            "server_instance": desktop_server_instance,
        },
        provider=_DESKTOP_SCOPE_PROVIDER,
        org_id="",
        user_id="",
    )


__all__ = [
    "ImageFilmCallerScopeError",
    "TrustedImageFilmCallerScope",
    "resolve_trusted_image_film_caller_scope",
]
