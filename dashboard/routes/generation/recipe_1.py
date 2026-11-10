from __future__ import annotations


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(".flux3.json")


def write_recipe_sidecar(path: Path | str, envelope: dict[str, Any]) -> Path:
    """Atomically write the sidecar without changing the media bytes."""
    target = _sidecar_path(Path(path))
    temporary: Path | None = None
    try:
        fd, name = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=".tmp", dir=str(target.parent))
        os.close(fd)
        temporary = Path(name)
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(envelope, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_embedded_recipe(path: Path) -> dict[str, Any] | None:
    try:
        result = _run(
            [
                str(FFPROBE_PATH),
                "-v",
                "error",
                "-show_entries",
                "format_tags=comment",
                "-of",
                "json",
                str(path),
            ]
        )
        payload = json.loads(result.stdout)
        comment = payload.get("format", {}).get("tags", {}).get("comment")
        if not isinstance(comment, str) or not comment.strip():
            return None
        value = json.loads(comment)
    except (RecipeError, TypeError, AttributeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    _validate_known_recipe_metadata(value)
    schema = value.get("schema")
    if schema in {
        "studio.derivation/1",
        "studio.recipe/1",
        "flux3.recipe/2",
    }:
        return value
    if isinstance(value.get("entry"), dict):
        return value
    return None


def _read_native_comfy_metadata(path: Path) -> dict[str, str]:
    """Read native ComfyUI workflow/prompt tags without interpreting them."""
    try:
        result = _run(
            [
                str(FFPROBE_PATH),
                "-v",
                "error",
                "-show_entries",
                "format_tags=workflow,prompt",
                "-of",
                "json",
                str(path),
            ]
        )
        payload = json.loads(result.stdout)
    except (UnicodeDecodeError, TypeError, json.JSONDecodeError) as exc:
        raise RecipeError("Could not read native ComfyUI metadata.") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("format"), dict):
        raise RecipeError("Could not read native ComfyUI metadata.")
    format_payload = payload["format"]
    if "tags" not in format_payload:
        return {}
    tags = format_payload["tags"]
    if not isinstance(tags, dict):
        raise RecipeError("Could not read native ComfyUI metadata.")
    native: dict[str, str] = {}
    for key in ("workflow", "prompt"):
        if key not in tags:
            continue
        value = tags[key]
        if not isinstance(value, str):
            raise RecipeError("Could not read native ComfyUI metadata.")
        native[key] = value
    return native


_DROP_PORTABLE_VALUE = object()
_PORTABLE_PROMPT_KEYS = {"prompt", "negative_prompt", "system_prompt", "workflow"}
_PORTABLE_MEDIA_KEYS = {"input_video", "input_image", "input_images", "reference_images"}
_LOCAL_ONLY_RECIPE_KEYS = {
    "folder",
    "saveTo",
    "source_path",
    "prepared_path",
    "keyframeSources",
}


def _looks_like_local_recipe_path(value: str) -> bool:
    text = value.strip()
    lower = text.lower()
    if not text or lower.startswith(("http://", "https://", "data:")):
        return False
    return bool(
        lower.startswith("file:")
        or re.match(r"^[A-Za-z]:", text)
        or text.startswith(("\\\\", "//", "/", "./", "../", ".\\", "..\\", "~/", "~\\"))
    )


def _portable_identity(value: Any, label: str, max_length: int = 256) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > max_length
        or "\0" in value
        or "/" in value
        or "\\" in value
        or _looks_like_local_recipe_path(value)
    ):
        raise RecipeError(f"{label} is invalid.")
    return value


def _present_portable_identity(
    source: dict[str, Any], key: str, label: str, max_length: int = 256
) -> str | None:
    if key not in source:
        return None
    identity = _portable_identity(source[key], label, max_length)
    if identity is None:
        raise RecipeError(f"{label} is invalid.")
    return identity


def _present_sha256_identity(source: dict[str, Any], key: str, label: str) -> str | None:
    if key not in source:
        return None
    value = source[key]
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise RecipeError(f"{label} is invalid.")
    return value


def _validate_recipe_entry_identity_fields(entry: dict[str, Any]) -> None:
    """Validate direct entry identities without interpreting creative descendants."""
    identity_aliases = (
        ("job_id", "jobId", "Recipe entry job identity"),
        ("provider_job_id", "providerJobId", "Recipe entry provider job identity"),
        ("root_job_id", "rootJobId", "Recipe entry root job identity"),
        ("provider_id", "providerId", "Recipe entry provider identity"),
        ("target_id", "targetId", "Recipe entry target identity"),
    )
    hash_aliases = (
        ("source_media_sha256", "sourceMediaSha256", "Recipe entry source media hash"),
        ("published_file_sha256", "publishedFileSha256", "Recipe entry published media hash"),
        ("workflow_api_hash", "workflowApiHash", "Recipe entry workflow API hash"),
        ("editor_template_hash", "editorTemplateHash", "Recipe entry editor template hash"),
        ("prepared_sha256", "preparedSha256", "Recipe entry prepared media hash"),
        ("file_sha256", "fileSha256", "Recipe entry file hash"),
    )
    for snake, camel, label in identity_aliases:
        for key in (snake, camel):
            if key in entry:
                _present_portable_identity(entry, key, label)
        _require_equal_present_aliases(entry, (snake, camel), label)
    for snake, camel, label in hash_aliases:
        for key in (snake, camel):
            _present_sha256_identity(entry, key, label)
        _require_equal_present_aliases(entry, (snake, camel), label)


def _require_equal_present_aliases(source: dict[str, Any], keys: tuple[str, ...], label: str) -> None:
    values = [source[key] for key in keys if key in source]
    if len(values) > 1 and any(value != values[0] for value in values[1:]):
        raise RecipeError(f"{label} aliases conflict.")


def _present_portable_alias(
    source: dict[str, Any], snake: str, camel: str, label: str, max_length: int = 256
) -> str | None:
    values = [
        _present_portable_identity(source, key, label, max_length)
        for key in (snake, camel)
        if key in source
    ]
    if len(values) > 1 and any(value != values[0] for value in values[1:]):
        raise RecipeError(f"{label} aliases conflict.")
    return values[0] if values else None


def _present_sha256_alias(source: dict[str, Any], snake: str, camel: str, label: str) -> str | None:
    values = [
        _present_sha256_identity(source, key, label)
        for key in (snake, camel)
        if key in source
    ]
    if len(values) > 1 and any(value != values[0] for value in values[1:]):
        raise RecipeError(f"{label} aliases conflict.")
    return values[0] if values else None


def _portable_recipe_value(value: Any, key: str | None = None) -> Any:
    if key in _LOCAL_ONLY_RECIPE_KEYS:
        return _DROP_PORTABLE_VALUE
    if key in _PORTABLE_PROMPT_KEYS:
        # Artist-authored creative payloads are opaque.  A workflow can be a
        # structured object, not only a string, and path-like descendants are
        # creative data rather than portable media references.
        return value
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for child_key, child_value in value.items():
            if not isinstance(child_key, str):
                continue
            cleaned = _portable_recipe_value(child_value, child_key)
            if cleaned is not _DROP_PORTABLE_VALUE:
                result[child_key] = cleaned
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            cleaned = _portable_recipe_value(item, key)
            if cleaned is not _DROP_PORTABLE_VALUE:
                result.append(cleaned)
        return result
    if isinstance(value, str) and key in _PORTABLE_MEDIA_KEYS:
        text = value.strip()
        if text.lower().startswith(("http://", "https://", "data:")):
            return value
        return _DROP_PORTABLE_VALUE
    if isinstance(value, str) and key not in _PORTABLE_PROMPT_KEYS and _looks_like_local_recipe_path(value):
        return _DROP_PORTABLE_VALUE
    return value


def _portable_recipe_entry(entry: dict[str, Any]) -> dict[str, Any]:
    _validate_recipe_entry_identity_fields(entry)
    cleaned = _portable_recipe_value(entry)
    return cleaned if isinstance(cleaned, dict) else {}


_STUDIO_RECIPE_PROVIDERS = frozenset({"comfyui-local", "local-voice"})


def _recipe_envelope(
    entry: dict[str, Any],
    *,
    job_id: str | None = None,
    shot: str | None = None,
    take: int | None = None,
    parent: dict[str, Any] | None = None,
    approved_roots: list[str] | None = None,
) -> dict[str, Any]:
    roots = approved_roots or []
    job_id = _portable_identity(job_id, "Recipe job identity")
    shot = _portable_identity(shot, "Recipe shot identity")
    if entry.get("schema") == "flux3.recipe/2":
        source_entry = entry.get("entry")
        if not isinstance(source_entry, dict):
            raise RecipeError("Versioned FLUX recipe entry is invalid.")
        _validate_recipe_entry_identity_fields(source_entry)
        selected_parent = parent if parent is not None else entry.get("parent")
        safe_parent = _lineage_parent(source_entry, selected_parent, roots)
        embedded_entry = _portable_recipe_entry(source_entry)
        if "parent" in embedded_entry and safe_parent is not None:
            embedded_entry["parent"] = safe_parent
        elif "parent" in embedded_entry:
            embedded_entry.pop("parent")
        envelope = {
            "schema": "flux3.recipe/2",
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "entry": embedded_entry,
            "parent": safe_parent,
            "tool": {"plugin": "film-suite"},
        }
        if job_id is not None:
            envelope["job_id"] = job_id
        if shot is not None:
            envelope["shot"] = shot
        if take is not None:
            envelope["take"] = take
        return envelope
    _validate_recipe_entry_identity_fields(entry)
    provider_id = _present_portable_alias(
        entry, "provider_id", "providerId", "Recipe provider identity"
    )
    _present_portable_alias(entry, "target_id", "targetId", "Recipe target identity")
    # Every studio take (ComfyUI and the local voice takes: recorded and re-cut, lane w5-backend item 3)
    # carries the one provider-neutral studio.recipe/1 envelope; flux3.recipe/2 stays the managed FLUX format.
    if provider_id in _STUDIO_RECIPE_PROVIDERS:
        return _studio_recipe_envelope(entry, job_id=job_id, parent=parent, roots=roots)
    safe_parent = _lineage_parent(entry, parent, roots)
    embedded_entry = _portable_recipe_entry(entry)
    if "parent" in embedded_entry and safe_parent is not None:
        embedded_entry["parent"] = safe_parent
    elif "parent" in embedded_entry:
        embedded_entry.pop("parent")
    envelope = {
        "schema": "flux3.recipe/2",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "entry": embedded_entry,
        "parent": safe_parent,
        "tool": {"plugin": "film-suite"},
    }
    if job_id is not None:
        envelope["job_id"] = job_id
    if shot is not None:
        envelope["shot"] = shot
    if take is not None:
        envelope["take"] = take
    return envelope


def _studio_recipe_envelope(
    entry: dict[str, Any],
    *,
    job_id: str | None,
    parent: dict[str, Any] | None,
    roots: list[str],
) -> dict[str, Any]:
    """Section 3.8 provenance: provider-neutral schema for new H3 outputs.

    The embedded stamp carries the pre-stamp media hash as
    ``source_media_sha256``; the post-stamp ``published_file_sha256`` is
    computed separately and never claimed to be embedded in the same MP4.
    The asset identity is collision-safe (basename, never an absolute user
    path); the absolute local destination stays in mutable job state.
    """
    args = entry.get("args") or {}
    # Sanitized entry: the renderer's frozen entry carries absolute local
    # paths (folder/saveTo) that must never be embedded.  Keep the recipe
    # slots, and expose the published asset identity as its basename only.
    safe_parent = _lineage_parent(entry, parent, roots)
    embedded_entry = _portable_recipe_entry(entry)
    if "parent" in embedded_entry and safe_parent is not None:
        embedded_entry["parent"] = safe_parent
    elif "parent" in embedded_entry:
        embedded_entry.pop("parent")
    save_to = entry.get("saveTo") or ""
    asset_identity = Path(save_to).name if save_to else (entry.get("filename") or "")
    provider_id = _present_portable_alias(
        entry, "provider_id", "providerId", "Studio provider identity"
    )
    target_id = _present_portable_alias(
        entry, "target_id", "targetId", "Studio target identity"
    )
    envelope = {
        "schema": "studio.recipe/1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "provider_id": provider_id,
        "target_id": target_id,
        # Exact artist-authored prompt and all submitted slot values.
        "entry": embedded_entry,
        "prompt": args.get("prompt", ""),
        "duration_seconds": args.get("duration_seconds"),
        "effective_frames": args.get("effective_frames"),
        "seed": args.get("seed"),
        "aspect_ratio": args.get("aspect_ratio"),
        "megapixels": args.get("megapixels"),
        "asset_identity": asset_identity,
        "parent": safe_parent,
        "tool": {"plugin": "film-suite"},
    }
    if job_id is not None:
        envelope["job_id"] = job_id
    # Comfy targets: workflow API hash and editor-template hash come from the
    # pinned manifest, not from the renderer.
    workflow_hashes = _comfy_workflow_hashes(envelope["target_id"])
    if workflow_hashes:
        envelope["workflow_api_hash"] = workflow_hashes[0]
        envelope["editor_template_hash"] = workflow_hashes[1]
        envelope["effective_frames"] = _manifest_effective_frames(
            envelope["target_id"], envelope.get("duration_seconds")
        )
    return envelope


def _manifest_effective_frames(target_id: str | None, duration_seconds: Any) -> int | None:
    """Compute the manifest's named duration rule server-side.

    The renderer submits only the requested seconds; the effective frame
    count is a manifest rule (round_fps_then_align_modulo) and is computed
    here so the stamp embeds the real value.
    """
    try:
        manifest_path = _comfy_manifest_path(target_id)
        if manifest_path is None or not manifest_path.exists():
            return None
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        rule = (payload.get("capabilities") or {}).get("duration") or {}
        effective = rule.get("effective_frames") or {}
        if effective.get("rule") != "round_fps_then_align_modulo":
            return None
        if duration_seconds is None:
            return None
        seconds = NumberLike(duration_seconds)
        minimum = NumberLike(rule.get("min"))
        maximum = NumberLike(rule.get("max"))
        if (
            seconds is None
            or (minimum is not None and seconds < minimum)
            or (maximum is not None and seconds > maximum)
            or (rule.get("request_type") == "integer_seconds" and not seconds.is_integer())
        ):
            raise RecipeError("Recipe duration is invalid.")
        fps = int(effective.get("fps") or rule.get("fps") or 24)
        minimum_frames = int(effective.get("minimum_frames") or 5)
        modulus = int(effective.get("modulus") or 17)
        remainder = int(effective.get("remainder") or 5)
        base = max(minimum_frames, round(seconds * fps))
        return base + ((remainder - base) % modulus)
    except (OSError, ValueError, TypeError):
        return None


def NumberLike(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _comfy_workflow_hashes(target_id: str | None) -> tuple[str, str] | None:
    """Read api_sha256 / editor_sha256 from the pinned target manifest."""
    try:
        manifest_path = _comfy_manifest_path(target_id)
        if manifest_path is None or not manifest_path.exists():
            return None
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        workflow = payload.get("workflow") or {}
        api_hash = workflow.get("api_sha256")
        editor_hash = workflow.get("editor_sha256")
        if isinstance(api_hash, str) and isinstance(editor_hash, str) and api_hash and editor_hash:
            return api_hash, editor_hash
    except (OSError, ValueError, TypeError):
        return None
    return None


def _comfy_manifest_path(target_id: str | None) -> Path | None:
    if not target_id:
        return None
    candidate = Path(__file__).resolve().parents[3] / "workflows" / target_id / "manifest.json"
    return candidate if candidate.is_file() else None


def _published_file_sha256(path: Path) -> str:
    """Post-stamp SHA-256 of the published file, kept in job/sidecar only."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stamp_request_projection(recipe: dict[str, Any], *, ignore_added_source_hash: bool) -> dict[str, Any]:
    projected = dict(recipe)
    projected.pop("created_at", None)
    if ignore_added_source_hash:
        projected.pop("source_media_sha256", None)
    return projected


def _stamp_write_once_identity_matches(existing: dict[str, Any], requested: dict[str, Any]) -> bool:
    if existing.get("schema") != requested.get("schema"):
        return False
    if existing.get("schema") == "studio.derivation/1":
        return all(
            existing.get(key) == requested.get(key)
            for key in ("derivation", "derivations", "root_job_id")
        )
    job_id = requested.get("job_id")
    return isinstance(job_id, str) and bool(job_id) and existing.get("job_id") == job_id


def _stamp_identity_matches(existing: dict[str, Any], requested: dict[str, Any]) -> bool:
    if not _stamp_write_once_identity_matches(existing, requested):
        return False
    ignore_added_source_hash = "source_media_sha256" not in requested
    return _stamp_request_projection(
        existing, ignore_added_source_hash=ignore_added_source_hash
    ) == _stamp_request_projection(
        requested, ignore_added_source_hash=False
    )


def _stamp_file(path: Path, envelope: dict[str, Any]) -> Path:
    with StampPublicationLocks(path, _sidecar_path(path)):
        return _stamp_file_locked(path, envelope)


def _stamp_file_locked(path: Path, envelope: dict[str, Any]) -> Path:
    # Ledger-idempotent: an envelope whose stamp is already embedded (same
    # job id, same media) is returned unchanged — no second remux, no second
    # sidecar.  Interrupted `pending` work is retried; succeeded work is
    # never re-stamped.
    _validate_known_recipe_metadata(envelope)
    existing = _read_embedded_recipe(path)
    if existing is not None and _stamp_write_once_identity_matches(existing, envelope):
        if not _stamp_identity_matches(existing, envelope):
            raise RecipeError("The existing stamp identity belongs to a different recipe request.")
        sidecar = _sidecar_path(path)
        record = _read_json(sidecar)
        _validate_known_recipe_metadata(record)
        expected_record = dict(existing)
        expected_record["published_file_sha256"] = _published_file_sha256(path)
        if record == expected_record:
            return sidecar
        raise RecipeError("The stamped media does not have its matching recipe sidecar.")

    before = probe_media(path)
    native_comfy_metadata = _read_native_comfy_metadata(path)
    # The pre-stamp media hash is embedded as source_media_sha256.  The
    # post-stamp published_file_sha256 is computed separately below and kept
    # in the sidecar record, never claimed to be embedded in the same MP4.
    source_hash = _published_file_sha256(path)
    envelope = dict(envelope)
    envelope["source_media_sha256"] = source_hash
    comment = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
    temporary: Path | None = None
    temporary_sidecar: Path | None = None
    sidecar_backup_candidate: Path | None = None
    sidecar_backup: Path | None = None
    media_committed = False
    rollback_complete = False
    try:
        temporary = path.with_name(f".{path.stem}.flux3-stamp-{uuid.uuid4().hex}{path.suffix}")
        _run(
            [
                str(FFMPEG_PATH),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(path),
                "-c",
                "copy",
                "-map",
                "0",
                "-map_metadata",
                "0",
                "-metadata",
                f"comment={comment}",
                "-movflags",
                "use_metadata_tags",
                str(temporary),
            ]
        )
        after = probe_media(temporary)
        if after["stream_count"] != before["stream_count"] or abs(after["duration"] - before["duration"]) > 0.01:
            raise RecipeError("ffmpeg changed the media duration or stream count.")
        if _read_native_comfy_metadata(temporary) != native_comfy_metadata:
            raise RecipeError("ffmpeg did not preserve native ComfyUI metadata.")
        sidecar_envelope = dict(envelope)
        sidecar_envelope["published_file_sha256"] = _published_file_sha256(temporary)
        temporary_sidecar = write_recipe_sidecar(temporary, sidecar_envelope)
        sidecar_target = _sidecar_path(path)
        if sidecar_target.is_file():
            sidecar_backup_candidate = sidecar_target.with_name(
                f".{sidecar_target.stem}.flux3-sidecar-backup-{uuid.uuid4().hex}{sidecar_target.suffix}"
            )
            shutil.copy2(sidecar_target, sidecar_backup_candidate)
            sidecar_backup = sidecar_backup_candidate
            sidecar_backup_candidate = None
        try:
            os.replace(temporary_sidecar, sidecar_target)
            temporary_sidecar = None
        except BaseException:
            rollback_complete = True
            raise
        try:
            os.replace(temporary, path)
            temporary = None
            media_committed = True
        except BaseException:
            try:
                if sidecar_backup is not None:
                    os.replace(sidecar_backup, sidecar_target)
                    sidecar_backup = None
                else:
                    sidecar_target.unlink(missing_ok=True)
                rollback_complete = True
            except OSError as rollback_error:
                raise RecipeError("Recipe publication failed and the original sidecar could not be restored.") from rollback_error
            raise
        if sidecar_backup is not None:
            _safe_unlink(sidecar_backup)
            sidecar_backup = None
        return sidecar_target
    finally:
        if temporary is not None:
            _safe_unlink(temporary)
        if temporary_sidecar is not None:
            _safe_unlink(temporary_sidecar)
        if sidecar_backup_candidate is not None:
            _safe_unlink(sidecar_backup_candidate)
        if sidecar_backup is not None and (media_committed or rollback_complete):
            _safe_unlink(sidecar_backup)


# --- The Script shot in an Image job's recipe (lane w6-link item 1) ---------------------------------
# An Image job started from the Script's "+ Frame" names its shot (shot_id, sh-xxxxxxxx) on submit. The
# shot is written once per NEW job id into that job's studio.recipe/1 record (the provider-neutral envelope
# above, one small JSON file per job under the Hermes state folder) and read back by job id: the status
# answer shows it and Keep assigns the kept picture to the shot from it (Media mark shot:<id>), so the link
# no longer lives only in the open window. The first record wins: a replay never moves a job to another
# shot. A job without shot has no record. Records never fail a route: the helpers swallow their own errors.
_IMAGE_SHOT_ID_RE = re.compile(r"^sh-[0-9a-f]{8}$")
_IMAGE_JOB_ID_RE = re.compile(r"^imgjob_[0-9a-f]{32}$")
_IMAGE_RECIPE_PROVIDER = "local-comfyui"


def _image_job_recipes_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    return Path(state_root(Path(base) / "hermes")) / "image-recipes"


def _image_job_recipe_write(job_ids: Any, *, target_id: Any, prompt: Any, shot_id: Any) -> None:
    """studio.recipe/1 with shot_id for each new job of a run (write-once, never raises)."""
    try:
        if type(shot_id) is not str or not _IMAGE_SHOT_ID_RE.fullmatch(shot_id):
            return
        target = target_id if type(target_id) is str and target_id else "image"
        folder = _image_job_recipes_dir()
        folder.mkdir(parents=True, exist_ok=True)
        for job_id in job_ids or ():
            if type(job_id) is not str or not _IMAGE_JOB_ID_RE.fullmatch(job_id):
                continue
            path = folder / f"{job_id}.json"
            if path.exists():
                continue
            entry = {"providerId": _IMAGE_RECIPE_PROVIDER, "targetId": target,
                     "args": {"prompt": str(prompt or "")[:16_384]}, "shot_id": shot_id}
            envelope = _studio_recipe_envelope(entry, job_id=job_id, parent=None, roots=[])
            envelope["shot_id"] = shot_id
            _validate_known_recipe_metadata(envelope)
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            try:
                temporary.write_text(json.dumps(envelope, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
                if not path.exists():
                    os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001 - a recipe note never fails a submit
        pass


def _image_job_recipe_read(job_id: Any) -> dict[str, Any] | None:
    """The job's studio.recipe/1 record, or None (no shot, unreadable, not ours)."""
    try:
        if type(job_id) is not str or not _IMAGE_JOB_ID_RE.fullmatch(job_id):
            return None
        value = _read_json(_image_job_recipes_dir() / f"{job_id}.json")
        if not value or value.get("schema") != "studio.recipe/1" or value.get("job_id") != job_id:
            return None
        if type(value.get("shot_id")) is not str or not _IMAGE_SHOT_ID_RE.fullmatch(value["shot_id"]):
            return None
        _validate_known_recipe_metadata(value)
        return value
    except Exception:  # noqa: BLE001 - unknown reads as no shot
        return None


def _image_job_shot(job_id: Any) -> str | None:
    recipe = _image_job_recipe_read(job_id)
    return recipe["shot_id"] if recipe else None


# Keep of a run made for a Script shot (producer_1.py image_producer_keep): the Media mark helpers of
# media/feed_1.py resolve at call time (same plugin_api namespace).
def _image_keep_assign_shot(owner_scope: str, job_id: str, shot_id: str) -> None:
    """The kept picture's Media mark shot:<id> (the mark "Assign to shot..." writes); other tags stay."""
    for _attempt in range(3):
        try:
            current = _media_marks_read(owner_scope)
            tags = [t for t in ((current["items"].get(job_id) or {}).get("tags") or []) if not str(t).startswith("shot:")]
            _media_marks_write(MediaMarksWriteRequest.model_validate({
                "schema": "studio.media-marks/1", "owner_scope": owner_scope, "expected_revision": current["revision"],
                "changes": [{"item_id": job_id, "tags": tags + ["shot:" + shot_id]}]}))
            return
        except Exception as exc:  # noqa: BLE001 - a conflict is read again; anything else leaves the keep as it is
            if getattr(exc, "code", None) != "project_conflict":
                return


# --- The artist's own words as an Image job's title (lane w35-ui-3, critic run 3 #11) ------------------------------
# A JSON-caption model (Ideogram 4, prompt_grammar ideogram_json) gets a caption the page builds from the artist's plain
# words; the ledger's artist_input.prompt is that built JSON (the snapshot is rebuilt byte-exactly, so it stays as it is)
# and Jobs, Media and the Image stage named the run "{"high_level_description":...". The page now sends the words it
# built the caption from as ``title``; it is written once per new job id next to the recipes (one small JSON file) and is
# the human-facing name. The caption stays the job's prompt (metadata: Media's caption, the recipe, the export).
# A job written before (no title file) reads the caption's own one-sentence summary, never raw JSON.
_IMAGE_TITLE_MAX = 2_000


def _image_job_titles_dir() -> Path:
    return _image_job_recipes_dir().parent / "image-titles"


def _image_job_title_write(job_ids: Any, title: Any) -> None:
    """studio.image-title/1 for each new job of a run (write-once, never raises)."""
    try:
        text = " ".join(str(title).split()) if type(title) is str else ""
        if not text:
            return
        folder = _image_job_titles_dir()
        folder.mkdir(parents=True, exist_ok=True)
        for job_id in job_ids or ():
            if type(job_id) is not str or not _IMAGE_JOB_ID_RE.fullmatch(job_id):
                continue
            path = folder / f"{job_id}.json"
            if path.exists():
                continue
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            try:
                temporary.write_text(json.dumps({"schema": "studio.image-title/1", "job_id": job_id, "title": text[:_IMAGE_TITLE_MAX]},
                                                ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
                if not path.exists():
                    os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001 - a title note never fails a submit
        pass


def _image_job_title(job_id: Any) -> str | None:
    """The artist's words a job was submitted with, or None (no title written, unreadable)."""
    try:
        if type(job_id) is not str or not _IMAGE_JOB_ID_RE.fullmatch(job_id):
            return None
        value = _read_json(_image_job_titles_dir() / f"{job_id}.json")
        title = value.get("title") if isinstance(value, dict) and value.get("job_id") == job_id else None
        return title if type(title) is str and title.strip() else None
    except Exception:  # noqa: BLE001 - unknown reads as no title
        return None


def _image_human_prompt(job_id: Any, prompt: Any) -> str:
    """The human-facing words of an Image job: its prompt, or for a JSON caption the artist's title (else the caption's
    summary sentence). Only a prompt that is a JSON object is looked up, so plain prompts cost nothing."""
    text = str(prompt or "")
    if not text.lstrip().startswith("{"):
        return text
    title = _image_job_title(job_id)
    if title:
        return title
    try:
        caption = json.loads(text)
    except ValueError:
        return text
    summary = caption.get("high_level_description") if isinstance(caption, dict) else None
    return summary if type(summary) is str and summary.strip() else text


def _image_human_title(job: Any) -> str:
    """A ledger job's human title (the status / submit answers' ``title``)."""
    job = job if isinstance(job, dict) else {}
    return _image_human_prompt(job.get("image_producer_job_id"), ((job.get("request_snapshot") or {}).get("artist_input") or {}).get("prompt"))
