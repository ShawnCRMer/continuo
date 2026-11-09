from __future__ import annotations


@router.get("/health")
async def health() -> dict[str, Any]:
    """Report only whether the existing managed service adapter is usable."""
    return {"available": bool(flux3.check_bfl_requirements())}


def _v4_error(reason_code: str, *, prefix: str | None = None) -> dict[str, Any]:
    """Return a finite Contract-4 route error without private diagnostics."""
    if reason_code in V4_ROUTE_REASON_MESSAGES:
        message = V4_ROUTE_REASON_MESSAGES[reason_code]
    elif reason_code in generation.contract.REASON_CODES:
        message = generation.contract.public_reason(reason_code)
    else:
        reason_code = "invalid_request"
        message = generation.contract.public_reason(reason_code)
    return {"error": f"{prefix}: {message}" if prefix else message, "reason_code": reason_code}


# Only these literal, path-free destination diagnostics may cross the route
# boundary. Never reflect arbitrary ValueError text (which may contain secrets).
# Lane p12-shl (BCK-023): each literal names its own refusal code; the shown text is that code's plain text
# (generation.contract.REFUSAL_REASON_MESSAGES). P3 f3: "output directory and basename must be supplied together"
# is a missing file name, never "No output folder".
_V4_DESTINATION_CODES = MappingProxyType({
    "project scope is invalid": "project_mismatch",
    "project output destination is invalid": "output_folder_missing",
    "output directory and basename must be supplied together": "output_name_missing",
    "requested basename is invalid": "output_name_invalid",
    "approved roots do not match the project manifest": "folders_changed",
    "an explicit output requires an approved root": "output_folder_not_approved",
    "approved roots are invalid": "folders_invalid",
})
# The GMI route reads the literal set to decide what it may log; its texts now come from the codes.
_V4_DESTINATION_MESSAGES = MappingProxyType({literal: generation.contract.public_reason(code)
                                             for literal, code in _V4_DESTINATION_CODES.items()})


# Lane w11-backend M-2: a path refusal of the project boundary (ProjectError,
# a ValueError subclass) is named by its fixed boundary diagnostic; the texts
# are literals of path_boundary.py, never a path. Unlisted ones fall back per code.
_V4_PROJECT_MESSAGES = MappingProxyType({
    "path is outside approved roots": "The output folder is not an approved media folder",
    "media is outside approved roots": "A chosen medium is not inside an approved media folder",
    "media requires at least one approved root": "No approved media folder for local media",
    "path must be a directory": "The output folder is not a folder",
    "path must be a regular file": "A chosen medium is not a file",
    "path does not exist": "A chosen folder or file does not exist",
    "path is inside HERMES_HOME": "Folder inside the Hermes data folder",
    "path type is not allowed": "A chosen folder or file is of a type that cannot be used",
    "symlink object is not allowed": "A chosen folder or file is a link",
    "symlink components are not allowed": "A chosen path runs through a link",
    "reparse components are not allowed": "A chosen path runs through a link",
    "path must be an absolute local path": "Path not absolute",
    "path must be absolute": "Path not absolute",
    "path contains forbidden characters": "A chosen path contains characters Windows does not allow",
    "project identity does not match": "The generation request does not match the open project",
})
_V4_PROJECT_CODE_MESSAGES = MappingProxyType({
    "project_not_found": "The project folder was not found",
    "project_unavailable": "The project folder is unavailable right now",
    "invalid_project": "A chosen folder or file cannot be used",
})
# Lane p12-shl (BCK-023): the refusal code of a project-boundary refusal, by its fixed literal, else by its code.
_V4_PROJECT_REFUSAL_CODES = MappingProxyType({
    "path is outside approved roots": "output_folder_not_approved",
    "project identity does not match": "project_mismatch",
})
_V4_PROJECT_CODE_REFUSALS = MappingProxyType({
    "project_not_found": "project_folder_missing",
    "project_unavailable": "project_unavailable",
    "invalid_project": "path_refused",
})
_V4_SUBMIT_PREFIX = "Generation submission failed"


def _v4_named(code: str, prefix: str | None, message: str | None = None) -> dict[str, Any]:
    """A named refusal: its own finite code (REFUSAL_REASON_CODES); the text is the code's, or a finer fixed literal."""
    if code not in generation.contract.REFUSAL_REASON_CODES:
        raise ValueError("unnamed refusal code")
    text = message or generation.contract.public_reason(code)
    return {"error": f"{prefix}: {text}" if prefix else text, "reason_code": code}


# Lane w11-fixb item 3: the reason of a /submit refusal as its own field (reason_text), without the
# generic prefix the M-2 mapping writes into ``error`` ("Generation submission failed: ..."), so the page
# shows only the reason; a path-like text is never sent. Lane p12-shl (BCK-023): the plain words come from
# the refusal's code, never from matching an English message ("Generation request is invalid" was turned
# into "The model did not accept these settings" for causes the model never saw).
def _submit_reason_text(result: Any) -> str | None:
    if not isinstance(result, dict) or type(result.get("error")) is not str or result.get("details") is not None:
        return None
    text = " ".join(result["error"].split())
    while True:
        head, sep, rest = text.partition(": ")
        if not sep or head not in ("Generation submission failed", "Generation request is invalid", "Generation request was refused"):
            break
        text = rest
    if text == generation.contract.public_reason("invalid_request"):  # the catch-all never blames the model
        text = generation.contract.public_reason("payload_invalid")
    if not text or len(text) > 300 or re.search(r"[A-Za-z]:[\\/]|\\\\|/", text):
        return None
    return text


def _v4_exception(exc: BaseException, *, prefix: str | None = None) -> dict[str, Any]:
    if type(exc) is ValueError and str(exc) in _V4_DESTINATION_CODES:
        return _v4_named(_V4_DESTINATION_CODES[str(exc)], prefix)
    if isinstance(exc, studio_core.ProjectError) and getattr(exc, "code", None) not in V4_ROUTE_REASON_CODES:
        message = _V4_PROJECT_MESSAGES.get(str(exc)) or _V4_PROJECT_CODE_MESSAGES.get(getattr(exc, "code", None))
        code = (_V4_PROJECT_REFUSAL_CODES.get(str(exc)) if getattr(exc, "code", None) == "invalid_project" else None) \
            or _V4_PROJECT_CODE_REFUSALS.get(getattr(exc, "code", None))
        if message and code:
            return _v4_named(code, prefix, message)
    # The H3 validators raise fixed, path-free sentences (like H3ControlError): settings that are not valid.
    if isinstance(exc, (_h3c_module.H3ConditioningError, _h3m_module.H3MediaError)):
        return _v4_named("payload_invalid", prefix, str(exc))
    reason = getattr(exc, "reason_code", None)
    if reason in V4_ROUTE_REASON_CODES:
        return _v4_error(reason, prefix=prefix)
    if reason in generation.contract.REASON_CODES:
        return _v4_error(reason, prefix=prefix)
    if isinstance(exc, _comfy_job_store_module.JobNotFoundError):
        return _v4_error("invalid_request", prefix=prefix)
    if isinstance(exc, _comfy_job_store_module.ProjectScopeError):
        return _v4_error("project_scope_mismatch", prefix=prefix)
    if isinstance(exc, (ValueError, TypeError)) and not isinstance(
        exc, PrivateOperationError
    ):
        if prefix == _V4_SUBMIT_PREFIX:  # lane p12-shl: on /submit a refused value is a setting, never the model
            return _v4_named("payload_invalid", prefix)
        return _v4_error("invalid_request", prefix=prefix)
    if isinstance(exc, (PrivateOperationError, OSError, ValueError, TypeError)):
        return _v4_error(getattr(exc, "code", "private_state_unavailable"), prefix=prefix)
    return _v4_error("unknown", prefix=prefix)


def _read_ledger_row(studio_job_id: str) -> Any:
    """Perform the one read-only lookup used to choose Contract 3 vs 4."""
    store = _comfy_store()
    connector = getattr(store, "_connect", None)
    if callable(connector):
        connection = connector()
        try:
            row = connection.execute(
                "SELECT * FROM studio_jobs WHERE studio_job_id = ?", (studio_job_id,)
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise _comfy_job_store_module.JobNotFoundError()
        converter = getattr(store, "_row_to_record", None)
        if not callable(converter):
            raise RuntimeError("ledger projection is unavailable")
        return converter(row)
    for name in ("read_row", "lookup_row", "get_any", "get_record"):
        method = getattr(store, name, None)
        if callable(method):
            return method(studio_job_id)
    raise RuntimeError("ledger read boundary is unavailable")


def _read_client_row(client_request_id: str) -> Any | None:
    store = _comfy_store()
    connector = getattr(store, "_connect", None)
    if callable(connector):
        connection = connector()
        try:
            row = connection.execute(
                "SELECT * FROM studio_jobs WHERE client_request_id = ?", (client_request_id,)
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        converter = getattr(store, "_row_to_record", None)
        if callable(converter):
            return converter(row)
        return None
    for name in ("read_by_client_request_id", "lookup_client_request", "get_by_client_request_id"):
        method = getattr(store, name, None)
        if callable(method):
            try:
                return method(client_request_id)
            except _comfy_job_store_module.JobNotFoundError:
                return None
    return None


def _v4_expected_scope(request: PollRequest | CancelRequest, row: Any) -> tuple[str | None, bool]:
    fields = getattr(request, "model_fields_set", set())
    present = "expected_project_id" in fields
    # Direct Python callers from the pre-V20 API did not carry the nullable
    # scope field.  Keep that call shape only for a no-project row; raw HTTP
    # parsing below remains strict for project rows.
    if not present and getattr(row, "project_id", None) is None:
        present = True
    return request.expected_project_id, present
