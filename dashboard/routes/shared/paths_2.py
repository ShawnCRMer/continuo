from __future__ import annotations


def _confined_output_path(path: str, approved_roots: list[str]) -> Path:
    try:
        candidate = resolve_approved_path(path, approved_roots)
    except (OSError, ValueError) as exc:
        raise RecipeError(str(exc)) from exc
    if candidate.exists() and candidate.is_dir():
        raise RecipeError("The trim destination must be a file path.")
    if not candidate.parent.is_dir():
        raise RecipeError("The trim destination folder does not exist.")
    return candidate


# Lane w34-sec (scan finding 17): the media trim's container and its exclusive publish (routes/media/routes_1.py).
_TRIM_CONTAINER_SUFFIX = ".mp4"


def _trim_destination_new(destination: Path) -> None:
    """The trim is a NEW file: ffmpeg writes MP4, so the name says .mp4, and an existing file (an original, an earlier
    take, a sidecar) is refused, never replaced. The publish (_publish_new_file) is no-replace too."""
    if destination.suffix.lower() != _TRIM_CONTAINER_SUFFIX:
        raise RecipeError(f"Trim output must be an {_TRIM_CONTAINER_SUFFIX} file (the trim is written as MP4).")
    if os.path.lexists(destination):
        raise RecipeError("Trim output must be a new file; an existing file is never replaced.")


def _publish_new_file(temporary: Path, destination: Path) -> None:
    """Lane w34-sec (finding 17): move ``temporary`` to ``destination`` only when nothing is there (exclusive create).
    A hard link is created no-replace on every file system that supports it; os.rename is the fallback (no-replace on
    Windows). An existing destination raises and stays as it was."""
    try:
        os.link(temporary, destination)
    except FileExistsError:
        raise RecipeError("Trim output must be a new file; an existing file is never replaced.") from None
    except OSError:
        if os.name != "nt" or os.path.lexists(destination):
            raise RecipeError("Trim output must be a new file; an existing file is never replaced.") from None
        try:
            os.rename(temporary, destination)
        except FileExistsError:
            raise RecipeError("Trim output must be a new file; an existing file is never replaced.") from None
        return
    os.unlink(temporary)
