from __future__ import annotations


class MusicAnalyzeRequest(BaseModel):
    path: str
    approved_roots: list[str] = Field(default_factory=list)
    project_ref: ProjectReference | None = None


class MusicPeaksRequest(BaseModel):
    path: str
    approved_roots: list[str] = Field(default_factory=list)
    project_ref: ProjectReference | None = None


class _MusicRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class MusicProjectReference(_MusicRequest):
    project_id: str = Field(pattern=r"^prj_[0-9a-f]{32}$", min_length=36, max_length=36)


_MUSIC_SHA256_FIELD = Field(pattern=r"^[0-9a-f]{64}$", min_length=64, max_length=64)


class MusicEvidenceImportRequest(_MusicRequest):
    track_sha256: str = _MUSIC_SHA256_FIELD
    # Lane p13b-aud item 2: the engine whose stored analysis is shown; its bars and tempo map, not the last run's.
    engine: str | None = Field(default=None, pattern=r"^(beat_this|librosa)$")


class MusicEvidenceReadRequest(_MusicRequest):
    track_sha256: str = _MUSIC_SHA256_FIELD
    evidence_id: str = _MUSIC_SHA256_FIELD


class MusicInterpretationReadRequest(_MusicRequest):
    track_sha256: str = _MUSIC_SHA256_FIELD
    project_ref: MusicProjectReference | None = None


class MusicInterpretationWriteRequest(_MusicRequest):
    track_sha256: str = _MUSIC_SHA256_FIELD
    evidence_id: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        min_length=64,
        max_length=64,
    )
    project_ref: MusicProjectReference | None = None
    expected_revision: int = Field(ge=0)
    value: dict[str, Any]

    @field_validator("evidence_id", mode="before")
    @classmethod
    def reject_explicit_null_evidence_id(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("evidence_id must be omitted or a non-null SHA-256 string")
        return value


class MusicMapResolveRequest(_MusicRequest):
    track_sha256: str = _MUSIC_SHA256_FIELD
    evidence_id: str = _MUSIC_SHA256_FIELD
    project_ref: MusicProjectReference | None = None


class MusicMapPreviewRequest(_MusicRequest):
    track_sha256: str = _MUSIC_SHA256_FIELD
    evidence_id: str = _MUSIC_SHA256_FIELD
    value: dict[str, Any]


class MusicWaveformRequest(_MusicRequest):
    track_sha256: str = _MUSIC_SHA256_FIELD
    evidence_id: str = _MUSIC_SHA256_FIELD
