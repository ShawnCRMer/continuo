"""Safe project foundation primitives for the Film Suite backend."""

from .path_boundary import (
    IMAGE_LIMIT,
    MAX_INPUTS,
    VIDEO_LIMIT,
    WINDOWS_JUNCTION_PRIVILEGE_UNAVAILABLE,
    NamedProjectLock,
    PathBoundary,
    PathIdentity,
    PinnedMedia,
    ProjectError,
    abort_pinned_media,
    consume_pinned_media,
    create_junction,
    ensure_private_node,
    pin_local_media,
    profile_identity_key,
)
from .project_store import ProjectStore, read_music_interpretation, read_project, write_music_interpretation

__all__ = [
    "IMAGE_LIMIT",
    "MAX_INPUTS",
    "VIDEO_LIMIT",
    "WINDOWS_JUNCTION_PRIVILEGE_UNAVAILABLE",
    "NamedProjectLock",
    "PathBoundary",
    "PathIdentity",
    "PinnedMedia",
    "ProjectError",
    "ProjectStore",
    "abort_pinned_media",
    "consume_pinned_media",
    "create_junction",
    "ensure_private_node",
    "pin_local_media",
    "profile_identity_key",
    "read_music_interpretation",
    "read_project",
    "write_music_interpretation",
]
