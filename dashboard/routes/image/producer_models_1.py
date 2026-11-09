from __future__ import annotations


def _image_contract_module() -> Any:
    """Load the image request contract from the backend's own generation package.

    Imported through the same resolver the composition root uses so the route
    can never bind a different copy of the contract than the ledger validates
    against.
    """
    with _PACKAGE_LOAD_LOCK:
        module_name = f"{_GENERATION_PACKAGE_NAME}.image_contract"
        cached = sys.modules.get(module_name)
        if cached is not None:
            return cached
        module_path = Path(__file__).with_name("generation") / "image_contract.py"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("The image request contract is unavailable.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        return module


def _image_producer_root(owner_scope: str) -> Any:
    """The profile root for an Image producer call.

    A project scope only ever reaches a project the application already opened,
    so it must never bootstrap anything.  Scratch is different: 'Continue
    without a project' opens no project, so nothing else would ever open the
    profile root - scratch opens it itself, through the same guarded path the
    project routes use.
    """
    if owner_scope == _SCRATCH_SCOPE:
        return _image_film_root()
    return _initialized_image_film_root()


def _image_operations_handoff(root: Any, owner_scope: str) -> Any:
    """Resolve the stores for an owner scope: a cached project, or scratch."""
    if owner_scope == _SCRATCH_SCOPE:
        return root.scratch_operations()
    return root.read_cached_project_operations(owner_scope[len("film_owner:"):])


class ImageProducerKeepRequest(_ProjectRequest):
    """Name a finished result so it becomes something to build on later."""

    schema_: Literal["studio.image-producer-operation/1"] = Field(alias="schema")
    kind: Literal["image_producer_keep"]
    owner_scope: str = Field(pattern=_OWNER_SCOPE_PATTERN, min_length=18, max_length=47)
    image_producer_job_id: str = Field(pattern=r"^imgjob_[0-9a-f]{32}$", min_length=39, max_length=39)
    name: str = Field(min_length=1, max_length=200)
    # lane p6-name: a typed file name for the output folder (wins over the project rule); None = the rule's name.
    file_name: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("name")
    @classmethod
    def _name_is_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("a name must not be blank")
        return stripped

    @property
    def project_id(self) -> str | None:
        """The project this scope names, or None for scratch."""
        suffix = self.owner_scope[len("film_owner:"):]
        return None if suffix == "scratch" else suffix


class ImageBuilderCatalogRequest(_ProjectRequest):
    schema_: Literal["studio.image-producer-operation/1"] = Field(alias="schema")
    kind: Literal["image_producer_catalog"]


class ImageProducerContextsRequest(_ProjectRequest):
    """Read the project's image contexts so a caller can pick a basis."""

    schema_: Literal["studio.image-producer-operation/1"] = Field(alias="schema")
    kind: Literal["image_producer_contexts"]
    owner_scope: str = Field(pattern=_OWNER_SCOPE_PATTERN, min_length=18, max_length=47)

    @property
    def project_id(self) -> str | None:
        """The project this scope names, or None for scratch."""
        suffix = self.owner_scope[len("film_owner:"):]
        return None if suffix == "scratch" else suffix


class ImageProducerSubmitRequest(_ProjectRequest):
    """Submit exactly one generation into the opened project's ledger.

    ``operation_id`` is the caller's idempotency key: a repeated submit with
    the same id replays the existing job instead of starting a second one, so
    a retried request cannot cost a second run.
    """

    schema_: Literal["studio.image-producer-operation/1"] = Field(alias="schema")
    kind: Literal["image_producer_submit"]
    owner_scope: str = Field(pattern=_OWNER_SCOPE_PATTERN, min_length=18, max_length=47)
    operation_id: str = Field(pattern=r"^imgop_[0-9a-f]{32}$", min_length=38, max_length=38)
    prompt: str = Field(min_length=1, max_length=16_384)
    seed: int = Field(ge=0, le=18_446_744_073_709_551_615)
    width: int = Field(ge=16, le=4096, multiple_of=8)
    height: int = Field(ge=16, le=4096, multiple_of=8)
    render_spec: dict[str, Any] | None = None
    # Optional, and only meaningful together: a caller either names the saved
    # revision it is building on, or names nothing and the store opens a
    # working context.  Half a basis is a caller error, never a guess.
    context_id: str | None = Field(default=None, pattern=r"^imgctx_[0-9a-f]{32}$")
    work_id: str | None = Field(default=None, pattern=r"^imgwork_[0-9a-f]{32}$")
    revision_id: str | None = Field(default=None, pattern=r"^imgrev_[0-9a-f]{32}$")
    # Lane w6-link: the Script shot this run makes a frame for ("+ Frame"); recorded in the job's recipe.
    shot_id: str | None = Field(default=None, pattern=r"^sh-[0-9a-f]{8}$", min_length=11, max_length=11)

    @model_validator(mode="after")
    def _basis_is_whole_or_absent(self) -> "ImageProducerSubmitRequest":
        named = (self.context_id, self.work_id, self.revision_id)
        if any(part is not None for part in named) and not all(part is not None for part in named):
            raise ValueError("a basis must name context, work and revision together")
        if self.render_spec is not None:
            # w24-models: shape here; whether ComfyUI can load the files is the submit's named refusal.
            spec = _image_contract_module().validate_render_spec(self.render_spec, check_files=False)
            if (spec["seed"], spec["width"], spec["height"]) != (self.seed, self.width, self.height):
                raise ValueError("render_spec seed and size must match request")
            if spec.get("schema") == "studio.krea2-render-spec/2" and spec["batch_index"] != 0:
                # A run is requested as a whole; its members are backend facts.
                raise ValueError("render_spec batch_index must be 0 in a request")
        return self

    @property
    def project_id(self) -> str | None:
        """The project this scope names, or None for scratch."""
        suffix = self.owner_scope[len("film_owner:"):]
        return None if suffix == "scratch" else suffix




# A job in one of these states is done with the provider; polling it again
# would cost a round trip for an answer the ledger already holds.
_IMAGE_TERMINAL_STATES = frozenset({"ready", "failed", "cancelled", "quarantined", "published"})


class ImageProducerCancelRequest(_ProjectRequest):
    """Stop one owned generation job (Generate's Stop)."""

    schema_: Literal["studio.image-producer-operation/1"] = Field(alias="schema")
    kind: Literal["image_producer_cancel"]
    owner_scope: str = Field(pattern=_OWNER_SCOPE_PATTERN, min_length=18, max_length=47)
    image_producer_job_id: str = Field(pattern=r"^imgjob_[0-9a-f]{32}$", min_length=39, max_length=39)

    @property
    def project_id(self) -> str | None:
        suffix = self.owner_scope[len("film_owner:"):]
        return None if suffix == "scratch" else suffix


class ImageProducerStatusRequest(_ProjectRequest):
    """Read one generation job of the explicitly opened project.

    The project is derived from the validated owner scope, never from a
    caller-supplied directory: a route must not be able to point the trusted
    root at a different project by writing a path into the body.
    """

    schema_: Literal["studio.image-producer-operation/1"] = Field(alias="schema")
    kind: Literal["image_producer_status"]
    owner_scope: str = Field(pattern=_OWNER_SCOPE_PATTERN, min_length=18, max_length=47)
    image_producer_job_id: str = Field(pattern=r"^imgjob_[0-9a-f]{32}$", min_length=39, max_length=39)

    @property
    def project_id(self) -> str | None:
        """The project this scope names, or None for scratch."""
        suffix = self.owner_scope[len("film_owner:"):]
        return None if suffix == "scratch" else suffix


# Lane w19-g7hermes item 6 (HER-046; reference /projects/open-recent in projects/routes_1.py): each process has its own
# composition root and its own cache of opened projects. The page's backend opens a project on Start; a Hermes tool call
# runs in Hermes' own process, where nothing opened it, so every project-scope Image route (use_in_shoot, jobs_list's
# image rows, import, keep, ...) answered image_film_unavailable. For a Hermes call the project of the scope is opened
# here exactly as open-recent opens it (its directory from the recent list, inside the backend only; the root checks
# the manifest's project id). Page calls are unchanged: they still reach only a project the app already opened.
_image_producer_root_before_w19 = _image_producer_root


def _image_producer_root(owner_scope: str) -> Any:  # noqa: F811 - append-only extension
    if owner_scope == _SCRATCH_SCOPE or _film_caller() != "hermes":
        return _image_producer_root_before_w19(owner_scope)
    project_id = owner_scope[len("film_owner:"):]
    root = _image_film_root()
    try:
        root.read_cached_project_operations(project_id)  # the handoff _image_operations_handoff takes
    except _ImageFilmCompositionError:
        try:
            directory = _recent_directory(_recent_project_store_factory(), project_id)
        except studio_core.ProjectError:
            raise _ImageFilmCompositionError("project_unavailable") from None
        root.open_project(directory, expected_project_id=project_id)
    return root


# lane w20-img (IMG-046): the Jobs list shows a Generate x N run as the row of member 0 (generation/jobs_1.py), so
# stopping that row stops the run: every unfinished member "<op>:run-<i>" is cancelled too (ComfyUI: deleting a batch
# entry stops the whole batch). A member key itself is never a run head.
async def _image_run_members_cancel(job: dict[str, Any], jobs: Any, handoff: Any, owner_scope: str) -> None:
    run_key = str(job.get("idempotency_key") or "")
    if not run_key or ":run-" in run_key:
        return
    index = 1
    while True:
        try:
            member = jobs.get_by_idempotency_key(f"{run_key}:run-{index}")
        except Exception as exc:  # noqa: BLE001 - narrowed below: no such member ends the run
            if type(exc).__name__ != "ImageJobNotFoundError" and not isinstance(exc, KeyError):
                raise
            return
        if member.get("state") not in _IMAGE_TERMINAL_STATES:
            async with _IMAGE_FILM_ROOT_LOCK_ASYNC:
                coordinator = handoff.producer_coordinator()
            before = member.get("state")
            stopped = await coordinator.cancel(member["image_producer_job_id"])
            after = stopped.get("state") if isinstance(stopped, dict) else before
            _announce_image_jobs([(member["image_producer_job_id"], before, after)], owner_scope)
        index += 1
