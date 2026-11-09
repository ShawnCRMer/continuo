from __future__ import annotations


@router.post("/music/analyze")
async def music_analyze(request: MusicAnalyzeRequest) -> dict[str, Any]:


    cache_file: Path | None = None
    cache_preexisting = False
    try:
        _load_music_analysis()

        source = _confined_music_path(request.path, _request_roots(request.approved_roots, request.project_ref, path=request.path, server_list=True))
        digest, cache_file, proxy_file = _music_cache_paths(source)
        cache_preexisting = cache_file.is_file()
        evict_analysis_cache(protected={digest})
        # Lane w25-audio2 item 5 (HER-088): a stored analysis is found by the file's content hash (the same file copied to
        # another folder is the same song), the default engine's first, as /music/open shows it; only without one it runs.
        stored = None
        for name in [music_analysis.default_engine()] + [n for n in music_analysis.ENGINE_CHOICES if n != music_analysis.default_engine()]:
            stored = stored or await asyncio.to_thread(music_analysis.stored_engine_analysis, source, name, digest)
        analysis = stored if stored is not None else await asyncio.to_thread(music_analysis.analyze_cached, source)
        proxy = await asyncio.to_thread(_ensure_music_proxy, source, proxy_file)
        evict_analysis_cache(protected={digest})
        return _music_metadata(analysis, proxy)
    except _FolderNotApproved:
        return _folder_refusal()
    except Exception as exc:  # noqa: BLE001 - route boundary translates every engine failure
        if cache_file is not None and not cache_preexisting:
            cache_file.unlink(missing_ok=True)
        return _error(_music_error(exc))


@router.post("/music/peaks")
async def music_peaks(request: MusicPeaksRequest) -> dict[str, Any]:
    try:
        engine_error = None
        try:
            _load_music_analysis()
        except Exception as exc:  # noqa: BLE001 - lane p12-aud (AUD-059): the waveform never needs the engine
            engine_error = _music_error(exc)
        source = _confined_music_path(request.path, _request_roots(request.approved_roots, request.project_ref, path=request.path, server_list=True))
        cached = None if engine_error is not None else await asyncio.to_thread(_read_music_cache_only, source)
        if cached is None:
            # Lane p12-aud (AUD-059): no stored analysis -> the waveform of the FFmpeg playback proxy (as /music/open).
            try:
                opened = await asyncio.to_thread(_music_open_sync, request.path, _request_roots(request.approved_roots, request.project_ref, path=request.path, server_list=True), engine_error or "")
            except Exception:  # noqa: BLE001 - not a readable sound file: the old answer
                return _error("Song not analysed.")
            return {"ok": True, "peaks": opened["peaks"], "duration": opened["duration"], "buckets_per_sec": MUSIC_PEAKS_PER_SECOND,
                    "track_sha256": opened["track_sha256"], "proxy_path": opened["proxy_path"], "analysed": False}
        return {
            "ok": True,
            "peaks": cached.get("peaks", []),
            "duration": cached.get("duration", 0.0),
            "buckets_per_sec": MUSIC_PEAKS_PER_SECOND,
        }
    except _FolderNotApproved:
        return _folder_refusal()
    except Exception as exc:  # noqa: BLE001 - route boundary translates every engine failure
        return _error(_music_error(exc))


@router.post("/music/interpretation/read")
async def music_interpretation_read(request: MusicInterpretationReadRequest) -> dict[str, Any]:
    try:
        result = await asyncio.to_thread(
            _music_read_interpretation,
            request.track_sha256,
            request.project_ref,
        )
        return {"ok": True, **result}
    except Exception as exc:
        return _music_neutral_error(_music_interpretation_failure_code(exc))


async def music_interpretation_write(request: MusicInterpretationWriteRequest) -> dict[str, Any]:
    try:
        result = await asyncio.to_thread(_music_write_interpretation, request)
        return {"ok": True, **result}
    except Exception as exc:
        answer = _music_neutral_error(_music_interpretation_failure_code(exc))
        # w28-fix item 5b (HER-TXT-2): a value with the wrong keys (the whole artifact, a typo) names them; the body keys
        # are music_core/contracts.validate_interpretation_body's.
        _MUSIC_INTERPRETATION_BODY_KEYS = ("tempo", "meter", "downbeat", "sections", "lyrics")
        keys = set(request.value) if isinstance(request.value, dict) else set()
        unknown = sorted(keys - set(_MUSIC_INTERPRETATION_BODY_KEYS))
        missing = sorted(set(_MUSIC_INTERPRETATION_BODY_KEYS) - keys)
        if answer.get("reason_code") == "invalid_music_interpretation" and (unknown or missing):
            parts = [f"unknown: {', '.join(unknown)}"] if unknown else []
            parts += [f"missing: {', '.join(missing)}"] if missing else []
            answer["message"] = (f"Music interpretation is invalid ({'; '.join(parts)}); "
                                 f"value is the body only: {', '.join(_MUSIC_INTERPRETATION_BODY_KEYS)}.")
        return answer


router.add_api_route(
    "/music/interpretation",
    music_interpretation_write,
    methods=["PUT"],
    route_class_override=MusicValidationRoute,
)


@router.post("/music/waveform")
async def music_waveform(request: MusicWaveformRequest) -> dict[str, Any]:
    try:
        root, evidence_store, _scratch_store, _boundary = _music_store_context()
        root_failure = _music_root_preflight(root)
        if root_failure is not None:
            return _music_neutral_error(root_failure)
        evidence = await asyncio.to_thread(evidence_store.read_evidence, request.evidence_id)
        if evidence.get("track_sha256") != request.track_sha256:
            return _music_neutral_error("invalid_music_evidence")
        waveform_id = evidence.get("waveform", {}).get("waveform_id")
        if not isinstance(waveform_id, str):
            return _music_neutral_error("invalid_music_evidence")
        waveform = await asyncio.to_thread(
            evidence_store.read_waveform,
            waveform_id,
            evidence_id=request.evidence_id,
        )
        if waveform.get("track_sha256") != request.track_sha256:
            return _music_neutral_error("invalid_music_evidence")
        return {"ok": True, "waveform": waveform}
    except Exception as exc:
        return _music_neutral_error(_music_evidence_failure_code(exc))


@router.post("/music/map/resolve")
async def music_map_resolve(request: MusicMapResolveRequest) -> dict[str, Any]:
    try:
        root, evidence_store, _scratch_store, _boundary = _music_store_context()
        root_failure = _music_root_preflight(root)
        if root_failure is not None:
            return _music_neutral_error(root_failure)
        evidence = await asyncio.to_thread(evidence_store.read_evidence, request.evidence_id)
        if evidence.get("track_sha256") != request.track_sha256:
            return _music_neutral_error("invalid_music_evidence")
    except Exception as exc:
        return _music_neutral_error(_music_evidence_failure_code(exc))

    try:
        interpretation_result = await asyncio.to_thread(
            _music_read_interpretation,
            request.track_sha256,
            request.project_ref,
        )
    except Exception as exc:
        return _music_neutral_error(_music_interpretation_failure_code(exc))

    try:
        interpretation = interpretation_result.get("interpretation") if interpretation_result.get("exists") else None
        result = await asyncio.to_thread(
            _music_core_module("resolver").resolve_music_map,
            evidence,
            interpretation,
        )
        return {"ok": True, "map": result}
    except (ValueError, TypeError):
        return _music_neutral_error("invalid_music_interpretation")
    except Exception:
        return _music_neutral_error("music_state_unavailable")


@music_preview_router.post("/music/map/preview")
async def music_map_preview(request: MusicMapPreviewRequest) -> dict[str, Any]:
    try:
        root, evidence_store, _scratch_store, _boundary = _music_store_context()
        root_failure = _music_root_preflight(root)
        if root_failure is not None:
            return _music_neutral_error(root_failure)
        try:
            evidence = await asyncio.to_thread(evidence_store.read_evidence, request.evidence_id)
        except Exception as exc:
            return _music_neutral_error(_music_evidence_failure_code(exc))
        if evidence.get("track_sha256") != request.track_sha256:
            return _music_neutral_error("invalid_music_evidence")

        try:
            contracts = _music_core_module("contracts")
            resolver = _music_core_module("resolver")
            validated_body = await asyncio.to_thread(
                contracts.validate_interpretation_against_evidence,
                request.value,
                evidence,
                request.track_sha256,
            )
            preview = await asyncio.to_thread(resolver.preview_music_map, evidence, validated_body)
            success_envelope = {"ok": True, "preview": preview}
            if len(contracts.canonical_json_bytes(success_envelope)) > 64 * 1024 * 1024:
                return _music_neutral_error("invalid_music_interpretation")
        except Exception as exc:
            if getattr(exc, "reason_code", None) == "music_candidate_missing":
                return _music_neutral_error("music_candidate_missing")
            if isinstance(exc, (ValueError, TypeError)):
                return _music_neutral_error("invalid_music_interpretation")
            return _music_neutral_error("music_state_unavailable")
        return success_envelope
    except Exception:
        return _music_neutral_error("music_state_unavailable")


router.include_router(music_preview_router)
