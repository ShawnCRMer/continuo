from __future__ import annotations


def _music_project_directory(project_ref: MusicProjectReference) -> str:
    return _recent_directory(_recent_project_store_factory(), project_ref.project_id)


def _music_interpretation_failure_code(exc: BaseException) -> str:
    missing = object()
    try:
        code = getattr(exc, "code", missing)
    except BaseException:
        return "music_state_unavailable"
    if code is missing:
        try:
            reason_code = getattr(exc, "reason_code", missing)
        except BaseException:
            return "music_state_unavailable"
        if reason_code == "music_candidate_missing":
            return "music_candidate_missing"
        if isinstance(exc, (ValueError, TypeError)):
            return "invalid_music_interpretation"
        return "music_state_unavailable"
    if type(code) is not str:
        return "music_state_unavailable"
    if code in {"music_evidence_missing", "invalid_music_evidence", "music_candidate_missing"}:
        return code
    if code in {"invalid_music_interpretation", "invalid_body", "scratch_corrupt"}:
        return "invalid_music_interpretation"
    if code in {"music_conflict", "scratch_conflict"}:
        return "music_conflict"
    if code == "music_scope_mismatch":
        return "music_scope_mismatch"
    if code == "invalid_music_request":
        return "invalid_music_request"
    return "music_state_unavailable"


class _MusicEvidenceAwareWriteError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _music_evidence_aware_body(request: MusicInterpretationWriteRequest) -> dict[str, Any]:
    """Validate a Phase-3 body against the exact immutable Evidence snapshot."""
    if request.evidence_id is None:
        return request.value

    root, evidence_store, _scratch_store, _boundary = _music_store_context(include_scratch=False)
    root_failure = _music_root_preflight(root)
    if root_failure is not None:
        raise _MusicEvidenceAwareWriteError(root_failure)
    try:
        evidence = evidence_store.read_evidence(request.evidence_id)
    except Exception as exc:
        raise _MusicEvidenceAwareWriteError(_music_evidence_failure_code(exc)) from None
    if not isinstance(evidence, dict) or evidence.get("track_sha256") != request.track_sha256:
        raise _MusicEvidenceAwareWriteError("invalid_music_evidence")
    body = _music_core_module("contracts").validate_interpretation_against_evidence(
        request.value,
        evidence,
        request.track_sha256,
    )
    _music_core_module("resolver").preview_music_map(evidence, body)
    return body


def _music_read_interpretation(
    track_sha256: str,
    project_ref: MusicProjectReference | None,
) -> dict[str, Any]:
    _root, _evidence_store, scratch_store, _boundary = _music_store_context()
    if project_ref is None:
        return scratch_store.read(track_sha256)
    directory = _music_project_directory(project_ref)
    return _project_store_factory().read_music_interpretation(
        directory,
        project_id=project_ref.project_id,
        track_sha256=track_sha256,
    )


def _music_write_interpretation(
    request: MusicInterpretationWriteRequest,
) -> dict[str, Any]:
    body = _music_evidence_aware_body(request)
    _root, _evidence_store, scratch_store, _boundary = _music_store_context(
        include_scratch=request.project_ref is None
    )
    if request.project_ref is None:
        if scratch_store is None:
            raise RuntimeError("scratch store is unavailable")
        return scratch_store.write(
            request.track_sha256,
            request.expected_revision,
            body,
        )
    directory = _music_project_directory(request.project_ref)
    return _project_store_factory().write_music_interpretation(
        directory,
        project_id=request.project_ref.project_id,
        track_sha256=request.track_sha256,
        expected_revision=request.expected_revision,
        value=body,
    )
