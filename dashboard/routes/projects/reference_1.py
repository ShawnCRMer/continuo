from __future__ import annotations


def _project_reference_directory(reference: ProjectReference) -> str:
    """Resolve a project reference without exposing private recent paths.

    Picker/open callers may supply a concrete directory.  Path-free public
    recent projections supply only the immutable project id; resolve that id
    through the backend-private recent registry and never return the path in a
    public response.
    """

    if isinstance(reference.directory, str) and reference.directory:
        return reference.directory
    return _recent_directory(_recent_project_store_factory(), reference.project_id)
