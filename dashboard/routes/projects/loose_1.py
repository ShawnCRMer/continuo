from __future__ import annotations


# --- Loose-work approved folders (lane p5-prj, Project charter decisions 4 and 8) ---------------------------
# Without a project the artist's approved folders live server-side in one small JSON file (approved_roots.py,
# LooseWorkFolders). Only this page route writes it, from the page's own folder picker (commitApprovedRoots,
# no-project branch); it is not an agent tool and refuses caller "hermes". expected_revision null replaces the
# list as the page holds it (the page does not track the revision yet); an integer is compare-and-set.


class LooseFoldersWriteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    expected_revision: int | None = Field(default=None, ge=0)
    approved_roots: list[str] = Field(max_length=64)


@project_router.put("/projects/loose/folders")
async def projects_loose_folders(request: LooseFoldersWriteRequest) -> dict[str, Any]:
    if _film_caller() == "hermes":
        return _folder_refusal()
    roots: list[str] = []
    for root in request.approved_roots:
        if _approved_roots_module.canonical(root) is None or not Path(root.strip()).is_dir():
            return {"error": {"code": "invalid_project", "message": "Project request is invalid"}}
        roots.append(root.strip())
    try:
        written = _loose_work_folders().write(list(dict.fromkeys(roots)), expected_revision=request.expected_revision)
    except _approved_roots_module.LooseConflict as exc:
        return {"error": {"code": "project_conflict", "message": "Project changed since this revision"}, "revision": exc.revision}
    except (OSError, ValueError):
        return {"error": {"code": "project_io_failed", "message": "Project could not be saved"}}
    return {"revision": written["revision"], "approved_roots": written["approved_roots"]}
