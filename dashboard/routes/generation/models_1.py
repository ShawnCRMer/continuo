from __future__ import annotations


class SubmitRequest(BaseModel):
    """Legacy-compatible input model; V4 shape is selected after row lookup."""

    model_config = ConfigDict(extra="forbid", strict=True)

    mode: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    schema: str | None = None
    provider_id: str | None = None
    target_id: str | None = None
    client_request_id: str | None = None
    input_media_sha256: str | None = None
    save_to: str | None = None
    requested_basename: str | None = None
    approved_roots: list[str] = Field(default_factory=list)
    project_id: str | None = Field(default=None, pattern=r"^prj_[0-9a-f]{32}$")
    project_ref: ProjectReference | None = None

    @model_validator(mode="after")
    def validate_project_pair(self) -> "SubmitRequest":
        if self.schema == "studio.generation-request/1":
            present_id = "project_id" in self.model_fields_set
            present_ref = "project_ref" in self.model_fields_set
            if present_id != present_ref:
                raise ValueError("project_id and project_ref must be supplied together")
            if present_id and self.project_ref is not None and self.project_ref.project_id != self.project_id:
                raise ValueError("project_ref project_id does not match project_id")
        return self


class PollRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    # save_to is retained only so already-installed Contract-3 callers keep
    # their historical request model.  The V4 body parser rejects it.
    save_to: str | None = None
    expected_project_id: str | None = Field(default=None, pattern=r"^prj_[0-9a-f]{32}$")


class CancelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    expected_project_id: str | None = Field(default=None, pattern=r"^prj_[0-9a-f]{32}$")


class GenerationHealthRequest(BaseModel):
    provider_id: str
    target_id: str


class RecipeStampRequest(BaseModel):
    path: str
    recipe: dict[str, Any]
    approved_roots: list[str] = Field(default_factory=list)
    project_ref: ProjectReference | None = None
    job_id: str | None = None
    shot: str | None = None
    take: int | None = None
    parent: dict[str, Any] | None = None


class RecipeReadRequest(BaseModel):
    path: str
    approved_roots: list[str] = Field(default_factory=list)
    project_ref: ProjectReference | None = None


class RecipeBackfillRequest(BaseModel):
    jobs: list[dict[str, Any]] = Field(default_factory=list)
    approved_roots: list[str] = Field(default_factory=list)
    project_ref: ProjectReference | None = None
