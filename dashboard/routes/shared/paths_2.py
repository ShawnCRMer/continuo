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
