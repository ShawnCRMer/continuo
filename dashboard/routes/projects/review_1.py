from __future__ import annotations


# --- Take A/B review (lane w5-review item 1) -------------------------------------------------------------------
# Two takes of one shot side by side, locked to one clock, optionally blind; "Prefer A/B" stores a decision.
# The records live in one append-only JSON Lines file per project (review/decisions.py). The take files are
# hashed here (sha256), so a decision names exactly the bytes that were compared. A take is named by its studio
# job id (resolved through the ledger to its published file, which must belong to this project) or by a path
# inside the project's approved folders. Every route is an agent tool; answers never carry a path.
_REVIEW_MODULE_NAME = "film_suite_review_decisions"


def _review_module() -> Any:
    module = sys.modules.get(_REVIEW_MODULE_NAME)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(_REVIEW_MODULE_NAME, Path(__file__).with_name("review") / "decisions.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("The review module is unavailable.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_REVIEW_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _review_store() -> Any:
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    return _review_module().ReviewStore(Path(state_root(Path(base) / "hermes")) / "review")


class ReviewTake(_ProjectRequest):
    """One take: its studio job id (or another stable id with ``path``), the label the page shows, optional file."""

    id: str = Field(min_length=1, max_length=120)
    label: str = Field(default="", max_length=60)
    path: str | None = Field(default=None, min_length=1, max_length=4096)


class ReviewStartRequest(_ProjectRequest):
    project_ref: ProjectReference
    shot: str = Field(default="shot", min_length=1, max_length=120)
    takes: list[ReviewTake] = Field(min_length=2, max_length=2)
    blind: bool = False
    by: Literal["artist", "hermes"] = "artist"


class ReviewReadRequest(_ProjectRequest):
    project_ref: ProjectReference
    shot: str | None = Field(default=None, min_length=1, max_length=120)


class ReviewNoteRequest(_ProjectRequest):
    project_ref: ProjectReference
    comparison_id: str = Field(pattern=r"^cmp_[0-9a-f]{32}$")
    frame: int = Field(ge=0, le=10_000_000)
    text: str = Field(min_length=1, max_length=500)
    by: Literal["artist", "hermes"] = "artist"


class ReviewDecideRequest(_ProjectRequest):
    project_ref: ProjectReference
    comparison_id: str = Field(pattern=r"^cmp_[0-9a-f]{32}$")
    chosen: Literal["a", "b"]
    by: Literal["artist", "hermes"] = "artist"


def _review_error(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "retryable": False}}


def _review_take_file(take: ReviewTake, project_id: str, roots: list[str]) -> Path:
    """The take's file: a path inside the approved folders, or the ledger's published file of this project."""
    module = _review_module()
    if take.path is not None:
        try:
            path = resolve_approved_path(take.path, roots)
        except (OSError, ValueError):
            raise module.ReviewError("take_outside_roots", "A take file is outside the project's approved folders.") from None
    else:
        try:
            row = _read_ledger_row(take.id)
        except Exception:  # noqa: BLE001 - unknown id or unreadable ledger: the same finite answer
            raise module.ReviewError("take_unknown", "That take is not a finished take of this project.") from None
        if getattr(row, "state", None) != "ready" or getattr(row, "project_id", None) != project_id:
            raise module.ReviewError("take_unknown", "That take is not a finished take of this project.")
        try:
            ops = _private_ops()
            path = Path(ops.saved_path_for_receipt(ops.load_for_job(row)))
        except Exception:  # noqa: BLE001
            raise module.ReviewError("take_unreadable", "The take's file cannot be read.") from None
    if not path.is_file():
        raise module.ReviewError("take_missing", "A take file is missing.")
    return path


def _review_project(reference: ProjectReference) -> tuple[str, list[str]]:
    directory = _project_reference_directory(reference)
    manifest = _project_store_factory().read_project(directory, expected_project_id=reference.project_id)
    return reference.project_id, [*manifest["approved_roots"], str(Path(os.path.realpath(directory)))]


def _review_call(work: Any) -> Any:
    module = _review_module()
    try:
        return work()
    except studio_core.ProjectError as exc:
        return _project_error_response(exc)
    except module.ReviewError as exc:
        return {"error": exc.as_dict()}


@project_router.post("/projects/review/start")
async def projects_review_start(request: ReviewStartRequest) -> Any:
    """Open an A/B comparison of two takes (blind shuffles which take is A); answers the comparison record."""
    def work() -> Any:
        project_id, roots = _review_project(request.project_ref)
        module = _review_module()
        takes = []
        for take in request.takes:
            path = _review_take_file(take, project_id, roots)
            takes.append({"id": take.id, "label": take.label, "sha256": module.sha256_file(path)})
        record = _review_store().start(project_id, shot=request.shot, take_a=takes[0], take_b=takes[1], blind=request.blind, by=request.by)
        return {"comparison": {**record, "notes": [], "decision": None}}
    return await asyncio.to_thread(_review_call, work)


@project_router.post("/projects/review/read")
async def projects_review_read(request: ReviewReadRequest) -> Any:
    """The project's comparisons (newest first, with notes and decision), the open ones and every decision."""
    def work() -> Any:
        project_id, _roots = _review_project(request.project_ref)
        return _review_store().view(project_id, shot=request.shot)
    return await asyncio.to_thread(_review_call, work)


@project_router.post("/projects/review/note")
async def projects_review_note(request: ReviewNoteRequest) -> Any:
    """A note on one frame of a comparison; it is a mark on the shot's track and goes into the decision."""
    def work() -> Any:
        project_id, _roots = _review_project(request.project_ref)
        return {"note": _review_store().note(project_id, request.comparison_id, frame=request.frame, text=request.text, by=request.by)}
    return await asyncio.to_thread(_review_call, work)


@project_router.post("/projects/review/decide")
async def projects_review_decide(request: ReviewDecideRequest) -> Any:
    """Prefer A or B: stores the decision record (take ids, sha256 of both, blind, chosen, notes, time, by)."""
    def work() -> Any:
        project_id, _roots = _review_project(request.project_ref)
        return {"decision": _review_store().decide(project_id, request.comparison_id, chosen=request.chosen, by=request.by)}
    return await asyncio.to_thread(_review_call, work)
