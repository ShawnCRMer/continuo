from __future__ import annotations


# --- Timeline footage: the artist's own files as cut sources (file:<sha256>) ---------
#
# /projects/timeline/footage/add registers files the artist picked in Media "Your folders"
# (or the agent names) for ONE project: each path is confined to the project's approved
# folders as the manifest holds them (never to roots the caller sends), hashed once, probed,
# and answered by id + facts, never by path. The id goes into a clip's source
# {kind: "media", ref: "file:<sha256>"} through /projects/timeline/apply like a take; the
# resolver (dashboard/timeline/resolver.py) binds it back to the file through the private
# index under the Hermes state folder (dashboard/timeline/footage.py). A moved, changed or
# no-longer-approved file resolves offline with its reason; the clip stays in the cut.

_TIMELINE_FOOTAGE_MAX = 50


class TimelineFootageAddRequest(_ProjectRequest):
    project_ref: ProjectReference
    paths: list[str] = Field(min_length=1, max_length=_TIMELINE_FOOTAGE_MAX)


def _timeline_footage_index(project_id: str) -> Any:
    footage = _timeline_module("footage")
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    path = state_root(Path(base) / "hermes") / "timeline-footage" / f"{project_id}.json"
    # relink_probe: a relinked file (a project from another home) may be a picture, clip or sound.
    return footage.FootageIndex(path, project_id, probe=lambda file: footage.probe_footage(file, ffprobe=FFPROBE_PATH),
                                relink_probe=lambda file: footage.probe_media(file, ffprobe=FFPROBE_PATH))


def _timeline_footage_roots(project_id: str, directory: str | None = None) -> list[str]:
    """The project's approved folders, read from its manifest now (a removed folder is no longer one)."""
    if directory is None:
        directory = _project_reference_directory(ProjectReference(project_id=project_id))
    return list(_project_store_factory().read_project(directory, expected_project_id=project_id)["approved_roots"])


def _timeline_footage_lookup(project_id: str) -> Any:
    """Resolver lookup for one read: ref -> (file, facts) of this project's footage, or the offline
    reason. The project's folders are read once per lookup set, on the first footage ref."""
    footage = _timeline_module("footage")
    roots: list[list[str]] = []
    indexes: list[Any] = []  # one index per lookup set: a relink lists the folders once per read

    def locate(ref: str) -> tuple[Path, dict[str, Any]]:
        if not roots:
            try:
                roots.append(_timeline_footage_roots(project_id))
            except (studio_core.ProjectError, OSError, ValueError) as exc:
                raise footage.FootageError("ledger_unavailable", "The project's folders could not be read.") from exc
        if not indexes:
            indexes.append(_timeline_footage_index(project_id))
        return indexes[0].locate(ref, roots[0])

    return locate


def _timeline_footage_public(added: dict[str, Any]) -> dict[str, Any]:
    facts = added["facts"]
    rate = facts.get("rate")
    seconds = None
    if facts.get("frames") and isinstance(rate, dict):
        seconds = round(facts["frames"] * rate["den"] / rate["num"], 3)
    return {"ref": added["ref"], "name": added["name"], "kind": facts.get("kind"), "frames": facts.get("frames"),
            "rate": rate, "duration_seconds": seconds, "width": facts.get("width"), "height": facts.get("height"),
            "has_audio": bool(facts.get("has_audio")), "codec": facts.get("codec") or "", "vfr": bool(facts.get("vfr"))}


def _timeline_footage_add(directory: str, project_id: str, paths: list[str]) -> dict[str, Any]:
    footage = _timeline_module("footage")
    roots = _timeline_footage_roots(project_id, directory)
    index = _timeline_footage_index(project_id)
    added: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for position, raw in enumerate(paths):
        try:
            if not roots:
                raise footage.FootageError("outside_roots", "No project folder.")
            try:
                path = resolve_approved_path(raw, roots)
            except (OSError, ValueError):
                raise footage.FootageError("outside_roots", "The file is outside the project's folders.") from None
            if not path.is_file():
                raise footage.FootageError("file_missing", "The file does not exist.")
            added.append(_timeline_footage_public(index.add(path)))
        except footage.FootageError as exc:
            errors.append({"index": position, "code": exc.code, "message": str(exc)})
        except OSError:
            errors.append({"index": position, "code": "unreadable", "message": "The file could not be read."})
    return {"footage": added, "errors": errors}


@project_router.post("/projects/timeline/footage/add")
async def projects_timeline_footage_add(request: TimelineFootageAddRequest) -> dict[str, Any]:
    try:
        directory = _project_reference_directory(request.project_ref)
        return await asyncio.to_thread(_timeline_footage_add, directory, request.project_ref.project_id, list(request.paths))
    except (studio_core.ProjectError, ValueError) as exc:
        return _timeline_error(exc)
