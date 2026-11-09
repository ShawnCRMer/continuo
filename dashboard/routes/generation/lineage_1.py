from __future__ import annotations


def _safe_derivation_record(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    source_hash = value.get("of")
    operation = value.get("op")
    params = value.get("params")
    if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", source_hash):
        return None
    if not isinstance(operation, str) or operation not in {"trim", "prepare", "remux"}:
        return None
    if not isinstance(params, dict):
        return None

    expected_keys = {
        "trim": {"start", "end", "frame_accurate"},
        "prepare": {"start", "end", "in_frame", "out_frame", "fps", "frame_accurate"},
        "remux": set(),
    }[operation]
    if set(params) != expected_keys:
        return None

    if operation == "remux":
        return {"of": source_hash, "op": operation, "params": {}}

    start = params["start"]
    end = params["end"]
    frame_accurate = params["frame_accurate"]
    if (
        isinstance(start, bool)
        or not isinstance(start, (int, float))
        or isinstance(end, bool)
        or not isinstance(end, (int, float))
        or not 0 <= start < end <= MAX_DERIVATION_TIME_SECONDS
        or not math.isfinite(float(start))
        or not math.isfinite(float(end))
        or not isinstance(frame_accurate, bool)
    ):
        return None

    if operation == "prepare":
        in_frame = params["in_frame"]
        out_frame = params["out_frame"]
        fps = params["fps"]
        if (
            isinstance(in_frame, bool)
            or not isinstance(in_frame, int)
            or isinstance(out_frame, bool)
            or not isinstance(out_frame, int)
            or not 0 <= in_frame <= out_frame <= MAX_DERIVATION_FRAME
            or isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or not 0 < fps <= MAX_DERIVATION_FPS
            or not math.isfinite(float(fps))
        ):
            return None
    return {"of": source_hash, "op": operation, "params": dict(params)}


def _derivation_envelope(source: Path, operation: str, params: dict[str, Any]) -> dict[str, Any]:
    """Build fresh identity and a bounded hash-only chain for derived media."""
    current = _safe_derivation_record({"of": _sha256_file(source), "op": operation, "params": params})
    if current is None:
        raise RecipeError("Media derivation parameters are invalid.")
    inherited: list[dict[str, Any]] = []
    root_job_id: str | None = None
    source_recipe = _read_embedded_recipe(source)
    if isinstance(source_recipe, dict):
        if source_recipe.get("schema") == "studio.derivation/1":
            raw_chain = source_recipe.get("derivations")
            if raw_chain is not None and not isinstance(raw_chain, list):
                raise RecipeError("Continuation derivation history is invalid or too long.")
            if isinstance(raw_chain, list):
                if not raw_chain or len(raw_chain) >= MAX_DERIVATION_CHAIN_LENGTH:
                    raise RecipeError("Continuation derivation history is invalid or too long.")
                validated = [_safe_derivation_record(item) for item in raw_chain]
                if any(record is None for record in validated):
                    raise RecipeError("Continuation derivation history is invalid or too long.")
                inherited = [record for record in validated if record is not None]
                direct = _safe_derivation_record(source_recipe.get("derivation"))
                if direct is None or direct != inherited[-1]:
                    raise RecipeError("Continuation derivation history is invalid or too long.")
            else:
                record = _safe_derivation_record(source_recipe.get("derivation"))
                if record is None:
                    raise RecipeError("Continuation derivation history is invalid or too long.")
                inherited = [record]
            candidate_job_id = _present_portable_alias(
                source_recipe,
                "root_job_id",
                "rootJobId",
                "Continuation root job identity",
            )
        else:
            candidate_job_id = _present_portable_alias(
                source_recipe,
                "job_id",
                "jobId",
                "Continuation root job identity",
            )
        if candidate_job_id is not None:
            root_job_id = candidate_job_id
    envelope = {
        "schema": "studio.derivation/1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "derivation": current,
        "derivations": [*inherited, current],
        "tool": {"plugin": "film-suite"},
    }
    if root_job_id:
        envelope["root_job_id"] = root_job_id
    return envelope


def _derivation_response(envelope: dict[str, Any], derived_path: Path) -> dict[str, Any]:
    current = envelope["derivation"]
    return {
        "derivation": current,
        "derivations": envelope["derivations"],
        "root_job_id": envelope.get("root_job_id"),
        "source_sha256": current["of"],
        "prepared_sha256": _sha256_file(derived_path),
    }


def derive_lineage(job_id: str, recipes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Derive chain position by walking parent pointers; never persist an index."""
    current = job_id
    seen: set[str] = set()
    job_ids: list[str] = []
    while isinstance(current, str) and current and current not in seen:
        seen.add(current)
        job_ids.append(current)
        recipe = recipes.get(current) if isinstance(recipes, dict) else None
        parent = recipe.get("parent") if isinstance(recipe, dict) else None
        next_job = parent.get("job_id") if isinstance(parent, dict) else None
        if not isinstance(next_job, str) or not next_job:
            break
        current = next_job
    return {"chain_index": max(0, len(job_ids) - 1), "job_ids": job_ids}


def _lineage_parent(
    entry: dict[str, Any], parent: dict[str, Any] | None, approved_roots: list[str]
) -> dict[str, Any] | None:
    candidate = parent
    if candidate is None and "parent" in entry:
        candidate = entry["parent"]
    if candidate is None:
        return None
    if not isinstance(candidate, dict):
        raise RecipeError("Continuation parent metadata is invalid.")
    # Only safe lineage keys are embedded: absolute local paths
    # (source_path/prepared_path) and any other user-local detail must never
    # travel with the shared media file.
    result: dict[str, Any] = {}
    if "method" in candidate:
        result["method"] = _present_portable_identity(candidate, "method", "Continuation parent method", 64)
    parent_job_id = _present_portable_alias(
        candidate, "job_id", "jobId", "Continuation parent job id"
    )
    if parent_job_id is not None:
        result["job_id"] = parent_job_id
    hashes = {
        key: value
        for key, value in (
            (
                "prepared_sha256",
                _present_sha256_alias(
                    candidate,
                    "prepared_sha256",
                    "preparedSha256",
                    "Continuation parent file hash",
                ),
            ),
            (
                "file_sha256",
                _present_sha256_alias(
                    candidate,
                    "file_sha256",
                    "fileSha256",
                    "Continuation parent file hash",
                ),
            ),
        )
        if value is not None
    }
    if len(set(hashes.values())) > 1:
        raise RecipeError("Continuation parent file hash is conflicting.")
    if hashes:
        result["file_sha256"] = hashes.get("prepared_sha256") or hashes["file_sha256"]
    raw_trim = candidate.get("trim")
    if raw_trim is not None:
        if not isinstance(raw_trim, dict):
            raise RecipeError("Continuation trim history is invalid.")
        safe_trim: dict[str, Any] = {}
        for key in ("start", "end", "fps", "actual_duration"):
            if key not in raw_trim:
                continue
            value = raw_trim[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= value <= MAX_DERIVATION_TIME_SECONDS
                or not math.isfinite(float(value))
            ):
                raise RecipeError("Continuation trim history is invalid.")
            safe_trim[key] = value
        for key in ("inFrame", "outFrame", "in_frame", "out_frame"):
            if key not in raw_trim:
                continue
            value = raw_trim[key]
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_DERIVATION_FRAME:
                raise RecipeError("Continuation trim history is invalid.")
            safe_trim[key] = value
        if "frame_accurate" in raw_trim:
            if not isinstance(raw_trim["frame_accurate"], bool):
                raise RecipeError("Continuation trim history is invalid.")
            safe_trim["frame_accurate"] = raw_trim["frame_accurate"]
        if "start" not in safe_trim or "end" not in safe_trim or safe_trim["end"] <= safe_trim["start"]:
            raise RecipeError("Continuation trim history is invalid.")
        result["trim"] = safe_trim
    raw_derivations = candidate.get("derivations")
    if raw_derivations is not None:
        if (
            not isinstance(raw_derivations, list)
            or not raw_derivations
            or len(raw_derivations) > MAX_DERIVATION_CHAIN_LENGTH
        ):
            raise RecipeError("Continuation derivation history is invalid.")
        safe_derivations = [_safe_derivation_record(item) for item in raw_derivations]
        if any(item is None for item in safe_derivations):
            raise RecipeError("Continuation derivation history is invalid.")
        result["derivations"] = safe_derivations
    if result.get("method") == "continuation" and not hashes:
        raise RecipeError("Continuation parent file hash is required.")
    return result if result else None
