from __future__ import annotations


def _music_core_module(name: str) -> Any:
    return importlib.import_module(f"{_MUSIC_CORE_PACKAGE_NAME}.{name}")


class _MusicStoreBoundary:
    """Expose the hardened path primitives expected by music_core stores."""

    def __init__(self, boundary: Any, path_boundary_module: Any) -> None:
        self._boundary = boundary
        self._path_boundary_module = path_boundary_module
        self.hermes_home = boundary.hermes_home
        def require_private_node(node: Path, *, directory: bool) -> None:
            path_boundary_module._require_private_windows_acl(node)
        self.ensure_private_node = path_boundary_module.ensure_private_node
        self.require_private_node = require_private_node
        self.flush_private_file = path_boundary_module._flush_private_file
        self.fsync_parent = path_boundary_module._fsync_parent
        self.open_existing = path_boundary_module._open_existing
        self.open_child_existing = path_boundary_module._open_child_existing
        self.win_link_handle_no_replace = path_boundary_module.win_link_handle_no_replace
        self.posix_link_no_replace = path_boundary_module._posix_linkat_empty_path
        self.unlink_handle_bound = path_boundary_module.unlink_handle_bound

    def validate_confined_path(
        self,
        value: os.PathLike[str] | str,
        approved_roots: Any,
        *,
        directory: bool = False,
        allow_missing_leaf: bool = False,
    ) -> Any:
        """Confine fixed profile-global store nodes without project-home rejection."""
        target = Path(value)
        module = self._path_boundary_module
        module._reject_reparse_components(target)
        if target.exists():
            if target.is_symlink() or target.is_dir() is not directory:
                raise OSError("music store node has an invalid type")
        elif not allow_missing_leaf:
            raise FileNotFoundError(str(target))
        target_real = Path(os.path.realpath(str(target)))
        for root in approved_roots:
            root_path = Path(root)
            module._reject_reparse_components(root_path)
            if not root_path.exists() or not root_path.is_dir():
                continue
            try:
                target_real.relative_to(Path(os.path.realpath(str(root_path))))
                return target_real
            except ValueError:
                continue
        raise OSError("music store node is outside its private root")

    def project_lock(self, directory: os.PathLike[str] | str) -> Any:
        return self._boundary.project_lock(directory)


def _music_store_context(*, include_scratch: bool = True) -> tuple[Path, Any, Any | None, Any]:
    """Build profile-private Evidence/Scratch stores without exposing paths."""
    profile_boundary = studio_core.PathBoundary(hermes_home=_active_hermes_home())
    profile_home = profile_boundary.hermes_home
    # PathBoundary reserves its configured home for project paths.  Global music
    # state lives inside the active profile home, so use a private sibling anchor
    # solely for the lock identity while retaining the active profile as root.
    store_boundary = studio_core.PathBoundary(hermes_home=profile_home / ".music-lock-anchor")
    path_boundary_module = importlib.import_module(f"{_STUDIO_CORE_PACKAGE_NAME}.path_boundary")
    injected = _MusicStoreBoundary(store_boundary, path_boundary_module)
    root = state_root(profile_home) / "music"
    evidence_store_module = _music_core_module("evidence_store")
    scratch_store_module = _music_core_module("scratch_store")
    evidence_store = evidence_store_module.EvidenceStore(
        root=root,
        boundary=injected,
        lock_factory=injected.project_lock,
    )
    scratch_store = None
    if include_scratch:
        scratch_store = scratch_store_module.ScratchStore(
            root=root,
            boundary=injected,
            lock_factory=injected.project_lock,
        )
    return root, evidence_store, scratch_store, profile_boundary


def _music_owned_catalog_path() -> Path:
    return Path(__file__).with_name("known-recordings.v1.json")


def _music_require_regular_nonreparse(path: Path, *, directory: bool) -> None:
    path_boundary_module = importlib.import_module(f"{_STUDIO_CORE_PACKAGE_NAME}.path_boundary")
    path_boundary_module._reject_reparse_components(path)
    info = path.lstat()
    if directory:
        valid = stat.S_ISDIR(info.st_mode)
    else:
        valid = stat.S_ISREG(info.st_mode)
    if not valid or path.is_symlink():
        raise OSError("music private node is invalid")


def _music_root_preflight(root: Path) -> str | None:
    """Classify the store root without weakening the store's private boundary."""
    try:
        info = root.lstat()
    except FileNotFoundError:
        return "music_evidence_missing"
    except (OSError, TypeError, ValueError):
        return "music_state_unavailable"
    if root.is_symlink() or not stat.S_ISDIR(info.st_mode):
        return "music_state_unavailable"
    return None


def _music_legacy_cache_file(track_sha256: str, engine: str | None = None) -> Path | None:
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    directory = state_root(Path(base) / "hermes") / "analysis"
    # Lane p13b-aud item 2: the named engine's stored result (<sha>.<engine>.json) when there is one, else the last run.
    named = directory / f"{track_sha256}.{engine}.json" if engine in ("beat_this", "librosa") else None
    target = named if named is not None and named.is_file() else directory / f"{track_sha256}.json"
    try:
        _music_require_regular_nonreparse(directory, directory=True)
        _music_require_regular_nonreparse(target, directory=False)
    except (OSError, ValueError, TypeError, RuntimeError):
        return None
    return target


def _music_read_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _music_import_v6_from_cache(track_sha256: str, engine: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    cache_file = _music_legacy_cache_file(track_sha256, engine)
    if cache_file is None:
        raise LookupError("music evidence cache is missing")
    try:
        payload = _music_read_json_file(cache_file)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        raise ValueError("music evidence cache is invalid") from None
    if not isinstance(payload, dict) or payload.get("source_sha256") != track_sha256:
        raise ValueError("music evidence cache is invalid")
    try:
        catalog_path = _music_owned_catalog_path()
        _music_require_regular_nonreparse(catalog_path, directory=False)
        catalog = _music_read_json_file(catalog_path)
        evidence, waveform, _audit = _music_core_module("import_v6").import_v6(
            payload,
            catalog_record=catalog,
        )
    except Exception:
        raise ValueError("music evidence or owned catalog is invalid") from None
    return evidence, waveform


@router.post("/music/evidence/import-v6")
async def music_evidence_import_v6(request: MusicEvidenceImportRequest) -> dict[str, Any]:
    try:
        evidence, waveform = await asyncio.to_thread(
            _music_import_v6_from_cache,
            request.track_sha256,
            request.engine,
        )
        _root, evidence_store, _scratch_store, _boundary = _music_store_context()
        evidence_id, waveform_id = await asyncio.to_thread(
            evidence_store.publish_pair,
            evidence,
            waveform,
        )
        return {
            "ok": True,
            "track_sha256": request.track_sha256,
            "evidence_id": evidence_id,
            "waveform_id": waveform_id,
            "evidence_schema": "studio.music-evidence/1",
            "waveform_schema": "studio.music-waveform/1",
            "published": {"evidence": True, "waveform": True},
        }
    except LookupError:
        return _music_neutral_error("music_evidence_missing")
    except (ValueError, TypeError):
        return _music_neutral_error("invalid_music_evidence")
    except Exception:
        return _music_neutral_error("music_state_unavailable")


def _music_evidence_failure_code(exc: BaseException) -> str:
    missing = object()
    try:
        code = getattr(exc, "code", missing)
    except BaseException:
        return "music_state_unavailable"
    if code is missing:
        if isinstance(exc, (ValueError, TypeError)):
            return "invalid_music_evidence"
        return "music_state_unavailable"
    if type(code) is not str:
        return "music_state_unavailable"
    if code == "artifact_not_found":
        return "music_evidence_missing"
    if code == "private_node_invalid":
        return "music_state_unavailable"
    if code in {"invalid_artifact", "relation_mismatch", "artifact_corrupt", "artifact_collision"}:
        return "invalid_music_evidence"
    return "music_state_unavailable"


@router.post("/music/evidence/read")
async def music_evidence_read(request: MusicEvidenceReadRequest) -> dict[str, Any]:
    try:
        _root, evidence_store, _scratch_store, _boundary = _music_store_context()
        evidence = await asyncio.to_thread(evidence_store.read_evidence, request.evidence_id)
        if evidence.get("track_sha256") != request.track_sha256:
            return _music_neutral_error("invalid_music_evidence")
        return {"ok": True, "evidence": evidence}
    except Exception as exc:
        return _music_neutral_error(_music_evidence_failure_code(exc))
