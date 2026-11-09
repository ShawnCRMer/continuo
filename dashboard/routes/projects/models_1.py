from __future__ import annotations


class _ProjectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ProjectReference(_ProjectRequest):
    project_id: str = Field(pattern=r"^prj_[0-9a-f]{32}$", min_length=36, max_length=36)
    # Public recent/open responses intentionally contain no directory.  An
    # ID-only reference is resolved through the backend-private recent registry;
    # callers that already hold a picker-selected directory may still send it.
    directory: str | None = Field(default=None, min_length=1, max_length=32767)


class ProjectCreateRequest(_ProjectRequest):
    directory: str = Field(min_length=1, max_length=32767)
    name: str = Field(min_length=1, max_length=120)
    copy_state: dict[str, Any] | None = None


class ProjectOpenRequest(_ProjectRequest):
    directory: str = Field(min_length=1, max_length=32767)


class ProjectOpenRecentRequest(_ProjectRequest):
    project_id: str = Field(pattern=r"^prj_[0-9a-f]{32}$", min_length=36, max_length=36)


class ProjectStateReadRequest(_ProjectRequest):
    project_ref: ProjectReference


class ProjectStateWriteRequest(_ProjectRequest):
    project_ref: ProjectReference
    expected_revision: int = Field(ge=1)
    state: dict[str, Any]


class ProjectMetadataRequest(_ProjectRequest):
    project_ref: ProjectReference
    expected_revision: int = Field(ge=1)
    name: str = Field(min_length=1, max_length=120)
    approved_roots: list[str]


class FilmImagePreviewTargetScope(_ProjectRequest):
    scope_kind: Literal["film_binding_new_reference"]
    project_id: str = Field(pattern=r"^prj_[0-9a-f]{32}$", min_length=36, max_length=36)
    binding_id: Literal[None] = Field(...)
    film_reference_id: Literal[None] = Field(...)


class FilmImageAssignedTargetScope(_ProjectRequest):
    scope_kind: Literal["film_binding"]
    project_id: str = Field(pattern=r"^prj_[0-9a-f]{32}$", min_length=36, max_length=36)
    binding_id: str = Field(
        pattern=r"^imgbind_[0-9a-f]{32}$",
        min_length=40,
        max_length=40,
    )
    film_reference_id: str = Field(
        pattern=r"^filmref_[0-9a-f]{32}$",
        min_length=40,
        max_length=40,
    )


class FilmImageOperationIntent(_ProjectRequest):
    branch_version: Literal["new_reference/1"]
    target_scope: FilmImageAssignedTargetScope
    expected_revision: int = Field(ge=0)
    label: str = Field(min_length=1, max_length=160)
    source_context_id: str = Field(pattern=r"^imgctx_[0-9a-f]{32}$", min_length=39, max_length=39)
    source_work_id: str = Field(pattern=r"^imgwork_[0-9a-f]{32}$", min_length=40, max_length=40)
    source_revision_id: str = Field(pattern=r"^imgrev_[0-9a-f]{32}$", min_length=39, max_length=39)
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", min_length=64, max_length=64)


class FilmImagePreviewRequest(_ProjectRequest):
    schema_: Literal["studio.image-operation/2"] = Field(alias="schema")
    kind: Literal["film_image_preview_new_reference"]
    branch_version: Literal["new_reference/1"]
    expected_revision: int = Field(ge=0)
    target_scope: FilmImagePreviewTargetScope
    label: str = Field(min_length=1, max_length=160)
    source_context_id: str = Field(pattern=r"^imgctx_[0-9a-f]{32}$", min_length=39, max_length=39)
    source_work_id: str = Field(pattern=r"^imgwork_[0-9a-f]{32}$", min_length=40, max_length=40)
    source_revision_id: str = Field(pattern=r"^imgrev_[0-9a-f]{32}$", min_length=39, max_length=39)


FilmImagePreviewNewReferenceRequest = FilmImagePreviewRequest


# Work with no project open lives under its own owner scope. The literal is
# exact - no project id may hide behind it - so scratch is a separate place
# with the same boundary a project has, not a way into one.
_SCRATCH_SCOPE = "film_owner:scratch"
_OWNER_SCOPE_PATTERN = r"^film_owner:(scratch|prj_[0-9a-f]{32})$"


class FilmImagePreviewCancelRequest(_ProjectRequest):
    schema_: Literal["studio.image-operation/2"] = Field(alias="schema")
    kind: Literal["film_image_preview_cancel"]
    branch_version: Literal["new_reference/1"]
    owner_scope: str = Field(pattern=_OWNER_SCOPE_PATTERN, min_length=18, max_length=47)
    expected_revision: int = Field(ge=0)
    target_scope: FilmImageAssignedTargetScope
    operation_intent: FilmImageOperationIntent
    preview_token: str = Field(pattern=r"^[A-Za-z0-9_-]{43}$", min_length=43, max_length=43)


class FilmImageOperationPayload(_ProjectRequest):
    preview_token: str = Field(pattern=r"^[A-Za-z0-9_-]{43}$", min_length=43, max_length=43)


def _validate_assigned_operation_identity(request: Any) -> Any:
    target = request.target_scope
    intent = request.operation_intent
    if (
        target.project_id != intent.target_scope.project_id
        or target != intent.target_scope
        or request.expected_revision != intent.expected_revision
        or request.context_id != intent.source_context_id
        or request.owner_scope != f"film_owner:{target.project_id}"
    ):
        raise ValueError("operation intent does not match the request")
    return request


class FilmImagePrepareRequest(_ProjectRequest):
    schema_: Literal["studio.image-operation/2"] = Field(alias="schema")
    kind: Literal["image_prepare_operation"]
    branch_version: Literal["new_reference/1"]
    owner_scope: str = Field(pattern=_OWNER_SCOPE_PATTERN, min_length=18, max_length=47)
    context_id: str = Field(pattern=r"^imgctx_[0-9a-f]{32}$", min_length=39, max_length=39)
    target_kind: Literal["film_image_adopt_new_reference"]
    expected_revision: int = Field(ge=0)
    target_scope: FilmImageAssignedTargetScope
    operation_intent: FilmImageOperationIntent
    payload: FilmImageOperationPayload

    @model_validator(mode="after")
    def validate_operation_identity(self) -> "FilmImagePrepareRequest":
        return _validate_assigned_operation_identity(self)


class FilmImageExecuteRequest(_ProjectRequest):
    schema_: Literal["studio.image-operation/2"] = Field(alias="schema")
    kind: Literal["film_image_adopt_new_reference"]
    branch_version: Literal["new_reference/1"]
    owner_scope: str = Field(pattern=_OWNER_SCOPE_PATTERN, min_length=18, max_length=47)
    context_id: str = Field(pattern=r"^imgctx_[0-9a-f]{32}$", min_length=39, max_length=39)
    operation_id: str = Field(pattern=r"^imgop_[0-9a-f]{32}$", min_length=38, max_length=38)
    issued_at: str = Field(
        pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$",
        min_length=20,
        max_length=20,
    )
    expected_revision: int = Field(ge=0)
    target_scope: FilmImageAssignedTargetScope
    operation_intent: FilmImageOperationIntent
    payload: FilmImageOperationPayload

    @field_validator("issued_at")
    @classmethod
    def validate_issued_at(cls, value: str) -> str:
        try:
            datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError as exc:
            raise ValueError("issued_at must be a valid UTC second timestamp") from exc
        return value

    @model_validator(mode="after")
    def validate_operation_identity(self) -> "FilmImageExecuteRequest":
        return _validate_assigned_operation_identity(self)


class FilmImageStatusRequest(_ProjectRequest):
    schema_: Literal["studio.image-operation/2"] = Field(alias="schema")
    kind: Literal["image_operation_status"]
    branch_version: Literal["new_reference/1"]
    owner_scope: str = Field(pattern=_OWNER_SCOPE_PATTERN, min_length=18, max_length=47)
    operation_id: str = Field(pattern=r"^imgop_[0-9a-f]{32}$", min_length=38, max_length=38)
    operation_kind: Literal["film_image_adopt_new_reference"]

    @property
    def project_id(self) -> str | None:
        """The project this scope names, or None for scratch."""
        suffix = self.owner_scope[len("film_owner:"):]
        return None if suffix == "scratch" else suffix
