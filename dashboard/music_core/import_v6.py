"""Strict, path-free conversion of a legacy ``flux3.music/6`` cache entry.

This adapter intentionally accepts an already decoded JSON object only.  It
never resolves or opens the embedded source path, imports an audio library, or
loads the mutable legacy catalog implicitly.  All persisted artifacts are
built as fresh dictionaries and passed through the sibling Type-C validators.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
from decimal import Decimal, InvalidOperation
from typing import Any

from .contracts import (
    MusicContractError,
    canonical_json_bytes,
    compute_evidence_id,
    compute_waveform_id,
    validate_evidence,
    validate_waveform,
)


_V6_SCHEMA = "flux3.music/6"
_V6_KEYS = frozenset(
    {
        "schema",
        "source_path",
        "source_sha256",
        "duration",
        "sample_rate",
        "analyzed_at",
        "tempo_curve",
        "tempo_curve_raw",
        "tempo_curve_status",
        "tempo_global",
        "beats",
        "meter",
        "downbeats",
        "downbeat_confidence",
        "sections",
        "accents",
        "energy",
        "key",
        "mode",
        "key_confidence",
        "brightness",
        "peaks",
        "integer_spans",
        "identified_recording",
    }
)
# fs-beat: payloads now record the beat engine (optional, so older v6 entries still import).
_V6_OPTIONAL_KEYS = frozenset({"engine"})
_ENGINE_IDS = {"beat_this", "librosa_v6"}
# Evidence ``method`` per engine: (tempo candidates, beats, downbeats, meter candidates).
_ENGINE_METHODS = {
    "beat_this": ("beat_this_final0_median_ibi", "beat_this_final0_minimal", "beat_this_final0_minimal", "beat_this_downbeat_fold"),
    "librosa_v6": ("legacy_v6_global_tempo", "legacy_v6_beat_track", "legacy_v6_downbeat", "legacy_v6_meter"),
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_CATALOG_SCHEMA = "flux3.known-recordings/1"
_CATALOG_KEYS = frozenset({"schema", "description", "recordings"})
_CATALOG_ROW_KEYS = frozenset({"id", "title", "tempo", "meter", "structure", "evidence"})
_CATALOG_MAX_BYTES = 1 * 1024 * 1024
_CATALOG_TEMPO_RELATIONS = {
    "detected",
    "alternate",
    "four_over_three",
    "primary",
    "half",
    "double",
    "published_candidate",
    "detected_candidate",
    "dj_full_tempo_candidate",
    "artist_grid",
    "local_grid",
    "catalogued",
}
_CATALOG_CITATION_KINDS = {
    "published_score",
    "peer_reviewed_analysis",
    "educational_score_analysis",
    "institutional_music_history",
    "published_sheet_music",
    "recording_metadata",
}
_LOSS_FLAGS = {
    "source_path_removed",
    "analyzed_at_removed",
    "duration_rounded_to_milliseconds",
    "legacy_runtime_unrecorded",
    "legacy_stream_unpinned",
    "application_projection_removed",
    "raw_signal_sections_lost_by_v6_override",
    "catalog_details_absent_from_v6",
    "catalog_snapshot_not_bound_to_v6_analysis",
    "legacy_selected_claims_mixed",
}


# These are the immutable v1 analyzer constants from the neutral contract.
_PARAMETERS = {
    "target_sample_rate_hz": 22050,
    "main_hop_samples": 512,
    "tempo_refinement_hop_samples": 128,
    "waveform_requested_bucket_rate_hz": 200,
    "minimum_probe_seconds": 3.0,
}
_PARAMETER_HASH = hashlib.sha256(
    canonical_json_bytes(_PARAMETERS)
).hexdigest()


def _fail(path: str, message: str) -> None:
    raise MusicContractError(f"{path}: {message}")


def _object(value: Any, path: str, required: set[str], optional: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "expected an object")
    allowed = set(required) | (optional or set())
    keys = set(value)
    if any(not isinstance(key, str) for key in value):
        _fail(path, "object keys must be strings")
    unknown = keys - allowed
    missing = set(required) - keys
    if unknown:
        _fail(path, f"unknown key(s): {', '.join(sorted(unknown))}")
    if missing:
        _fail(path, f"missing key(s): {', '.join(sorted(missing))}")
    return value


def _number(value: Any, path: str, *, minimum: float | None = None, maximum: float | None = None) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "expected a JSON number (boolean is not a number)")
    if isinstance(value, float) and not math.isfinite(value):
        _fail(path, "must be finite")
    if minimum is not None and value < minimum:
        _fail(path, f"must be >= {minimum}")
    if maximum is not None and value > maximum:
        _fail(path, f"must be <= {maximum}")
    return value


def _integer(value: Any, path: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, "expected an integer (boolean is not an integer)")
    if minimum is not None and value < minimum:
        _fail(path, f"must be >= {minimum}")
    if maximum is not None and value > maximum:
        _fail(path, f"must be <= {maximum}")
    return value


def _string(value: Any, path: str, *, nonempty: bool = True, max_bytes: int = 1024 * 1024) -> str:
    if not isinstance(value, str):
        _fail(path, "expected a string")
    if nonempty and not value:
        _fail(path, "must not be empty")
    if "\x00" in value or any(ord(char) < 32 for char in value):
        _fail(path, "C0 control characters are not allowed")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        _fail(path, f"must be valid UTF-8: {exc}")
    if size > max_bytes:
        _fail(path, f"UTF-8 length exceeds {max_bytes} bytes")
    return value


def _optional_number(value: Any, path: str, *, minimum: float | None = None, maximum: float | None = None) -> int | float | None:
    if value is None:
        return None
    return _number(value, path, minimum=minimum, maximum=maximum)


def _optional_confidence(value: Any, path: str) -> int | float | None:
    return _optional_number(value, path, minimum=0, maximum=1)


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha_text(value: Any, path: str) -> str:
    value = _string(value, path, max_bytes=64)
    if _SHA256_RE.fullmatch(value) is None:
        _fail(path, "must be exactly 64 lowercase hexadecimal characters")
    return value


def _stable_candidate_id(family: str, body: dict[str, Any]) -> str:
    return f"{family}.{_sha(body)}"


def _v6_curve(value: Any, path: str, duration: float) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        _fail(path, "expected an array")
    out: list[dict[str, Any]] = []
    previous = -float("inf")
    for index, row in enumerate(value):
        item = _object(row, f"{path}[{index}]", {"time", "bpm"})
        time_value = _number(item["time"], f"{path}[{index}].time", minimum=0, maximum=duration)
        bpm = _number(item["bpm"], f"{path}[{index}].bpm", minimum=0, maximum=400)
        if float(time_value) < previous - 1e-6:
            _fail(path, "rows must be ordered by time")
        previous = float(time_value)
        out.append({"time_seconds": time_value, "bpm": bpm})
    return out


def _v6_event_times(value: Any, path: str, duration: float) -> list[int | float]:
    if not isinstance(value, list):
        _fail(path, "expected an array")
    return [
        _number(item, f"{path}[{index}]", minimum=0, maximum=duration)
        for index, item in enumerate(value)
    ]


def _validate_v6_payload(payload: object) -> dict[str, Any]:
    # canonical_json_bytes performs the shared finite/depth/list/string checks
    # before any field is consumed, without normalizing or mutating payload.
    canonical_json_bytes(payload)
    item = _object(payload, "$", set(_V6_KEYS), set(_V6_OPTIONAL_KEYS))
    if item["schema"] != _V6_SCHEMA:
        _fail("$.schema", "must equal flux3.music/6")
    _string(item["source_path"], "$.source_path", nonempty=False)
    source_sha256 = _sha_text(item["source_sha256"], "$.source_sha256")
    duration = float(_number(item["duration"], "$.duration", minimum=0))
    if duration <= 0:
        _fail("$.duration", "must be > 0")
    _integer(item["sample_rate"], "$.sample_rate", minimum=8000, maximum=384000)
    _string(item["analyzed_at"], "$.analyzed_at")

    _v6_curve(item["tempo_curve"], "$.tempo_curve", duration)
    _v6_curve(item["tempo_curve_raw"], "$.tempo_curve_raw", duration)
    _string(item["tempo_curve_status"], "$.tempo_curve_status")

    tempo = _object(
        item["tempo_global"],
        "$.tempo_global",
        {"bpm", "candidates", "confidence", "status", "source"},
        {"triplet_bpm", "octave_margin", "octave_ambiguous", "alt_bpm", "reason", "local_iqr_pct"},
    )
    _optional_number(tempo["bpm"], "$.tempo_global.bpm", minimum=20, maximum=400)
    _optional_confidence(tempo["confidence"], "$.tempo_global.confidence")
    if tempo["status"] not in {"fixed", "ambiguous", "variable", "none"}:
        _fail("$.tempo_global.status", "has an unsupported v6 status")
    if tempo["source"] not in {"signal", "recording_catalog"}:
        _fail("$.tempo_global.source", "has an unsupported v6 source")
    if not isinstance(tempo["candidates"], list):
        _fail("$.tempo_global.candidates", "expected an array")
    for index, candidate in enumerate(tempo["candidates"]):
        row = _object(
            candidate,
            f"$.tempo_global.candidates[{index}]",
            {"bpm", "relation", "source", "confidence", "selected_for_grid"},
        )
        _number(row["bpm"], f"$.tempo_global.candidates[{index}].bpm", minimum=20, maximum=400)
        if row["relation"] not in _CATALOG_TEMPO_RELATIONS:
            _fail(f"$.tempo_global.candidates[{index}].relation", "has an unsupported tempo relation")
        if row["source"] not in {"signal", "recording_catalog"}:
            _fail(f"$.tempo_global.candidates[{index}].source", "has an unsupported v6 source")
        _optional_confidence(row["confidence"], f"$.tempo_global.candidates[{index}].confidence")
        if not isinstance(row["selected_for_grid"], bool):
            _fail(f"$.tempo_global.candidates[{index}].selected_for_grid", "expected a JSON boolean")
    for key in ("triplet_bpm", "octave_margin", "alt_bpm"):
        if key in tempo:
            _optional_number(tempo[key], f"$.tempo_global.{key}")
    if "octave_ambiguous" in tempo and tempo["octave_ambiguous"] is not None and not isinstance(tempo["octave_ambiguous"], bool):
        _fail("$.tempo_global.octave_ambiguous", "expected a JSON boolean or null")
    if "local_iqr_pct" in tempo:
        _optional_number(tempo["local_iqr_pct"], "$.tempo_global.local_iqr_pct", minimum=0)
    if "reason" in tempo:
        _string(tempo["reason"], "$.tempo_global.reason", max_bytes=64)
    if "engine" in item:
        engine = item["engine"]
        if not isinstance(engine, dict) or engine.get("id") not in _ENGINE_IDS:
            _fail("$.engine.id", f"must be one of {sorted(_ENGINE_IDS)}")
        canonical_json_bytes(engine)

    _v6_event_times(item["beats"], "$.beats", duration)
    meter = _object(
        item["meter"],
        "$.meter",
        {"value", "confidence", "assumed", "status", "source", "candidates", "regions"},
    )
    if meter["value"] is not None:
        _integer(meter["value"], "$.meter.value", minimum=1, maximum=32)
    _optional_confidence(meter["confidence"], "$.meter.confidence")
    if not isinstance(meter["assumed"], bool):
        _fail("$.meter.assumed", "expected a JSON boolean")
    if meter["status"] not in {"fixed", "mixed", "ambiguous", "none"}:
        _fail("$.meter.status", "has an unsupported v6 status")
    if meter["source"] not in {"signal", "recording_catalog"}:
        _fail("$.meter.source", "has an unsupported v6 source")
    if not isinstance(meter["candidates"], list):
        _fail("$.meter.candidates", "expected an array")
    for index, candidate in enumerate(meter["candidates"]):
        row = _object(candidate, f"$.meter.candidates[{index}]", {"value", "label"})
        _integer(row["value"], f"$.meter.candidates[{index}].value", minimum=1, maximum=32)
        _string(row["label"], f"$.meter.candidates[{index}].label", max_bytes=128)
    if not isinstance(meter["regions"], list):
        _fail("$.meter.regions", "expected an array")

    _v6_event_times(item["downbeats"], "$.downbeats", duration)
    _optional_confidence(item["downbeat_confidence"], "$.downbeat_confidence")

    if not isinstance(item["sections"], list):
        _fail("$.sections", "expected an array")
    for index, section in enumerate(item["sections"]):
        row = _object(
            section,
            f"$.sections[{index}]",
            {"start", "end", "label", "confidence"},
            {"source"},
        )
        _number(row["start"], f"$.sections[{index}].start", minimum=0, maximum=duration)
        _number(row["end"], f"$.sections[{index}].end", minimum=0, maximum=duration)
        if float(row["start"]) > float(row["end"]) + 1e-6:
            _fail(f"$.sections[{index}]", "start must be <= end")
        _string(row["label"], f"$.sections[{index}].label", max_bytes=256)
        _optional_confidence(row["confidence"], f"$.sections[{index}].confidence")
        if "source" in row:
            _string(row["source"], f"$.sections[{index}].source")

    if not isinstance(item["accents"], list):
        _fail("$.accents", "expected an array")
    for index, accent in enumerate(item["accents"]):
        row = _object(accent, f"$.accents[{index}]", {"time", "strength", "salience"})
        _number(row["time"], f"$.accents[{index}].time", minimum=0, maximum=duration)
        _number(row["strength"], f"$.accents[{index}].strength", minimum=0, maximum=1)
        _number(row["salience"], f"$.accents[{index}].salience", minimum=0)

    if not isinstance(item["energy"], list):
        _fail("$.energy", "expected an array")
    for index, sample in enumerate(item["energy"]):
        row = _object(sample, f"$.energy[{index}]", {"time", "value"})
        _number(row["time"], f"$.energy[{index}].time", minimum=0, maximum=duration)
        _number(row["value"], f"$.energy[{index}].value", minimum=0)

    for field in ("key", "mode"):
        if item[field] is not None:
            _string(item[field], f"$.{field}", max_bytes=32)
    _optional_confidence(item["key_confidence"], "$.key_confidence")
    brightness = _object(item["brightness"], "$.brightness", {"hz", "normalized"})
    _number(brightness["hz"], "$.brightness.hz", minimum=0)
    _number(brightness["normalized"], "$.brightness.normalized", minimum=0, maximum=1)

    if not isinstance(item["peaks"], list):
        _fail("$.peaks", "expected an array")
    for index, peak in enumerate(item["peaks"]):
        if not isinstance(peak, list) or len(peak) != 2:
            _fail(f"$.peaks[{index}]", "must contain exactly min and max")
        minimum = _number(peak[0], f"$.peaks[{index}][0]", minimum=-1, maximum=1)
        maximum = _number(peak[1], f"$.peaks[{index}][1]", minimum=-1, maximum=1)
        if float(minimum) > float(maximum):
            _fail(f"$.peaks[{index}]", "min must be <= max")

    if item["identified_recording"] is not None:
        identified = _object(
            item["identified_recording"],
            "$.identified_recording",
            {"id", "title", "match", "evidence"},
        )
        _string(identified["id"], "$.identified_recording.id", max_bytes=256)
        if identified["title"] is not None:
            _string(identified["title"], "$.identified_recording.title", max_bytes=256)
        canonical_json_bytes(identified["match"])
        canonical_json_bytes(identified["evidence"])
    return item


def _catalog_tempo(value: Any, path: str) -> dict[str, Any]:
    item = _object(
        value,
        path,
        {"status", "bpm", "candidates"},
        {"range_bpm", "trend", "curve_anchor_bpm"},
    )
    if item["status"] not in {"fixed", "ambiguous", "variable", "none"}:
        _fail(f"{path}.status", "has an unsupported catalog tempo status")
    _optional_number(item["bpm"], f"{path}.bpm", minimum=20, maximum=400)
    if not isinstance(item["candidates"], list):
        _fail(f"{path}.candidates", "expected an array")
    candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(item["candidates"]):
        row = _object(candidate, f"{path}.candidates[{index}]", {"bpm", "relation"})
        candidates.append(
            {
                "bpm": _number(row["bpm"], f"{path}.candidates[{index}].bpm", minimum=20, maximum=400),
                "relation": _catalog_enum(row["relation"], f"{path}.candidates[{index}].relation", _CATALOG_TEMPO_RELATIONS),
            }
        )
    out: dict[str, Any] = {
        "status": item["status"],
        "bpm": item["bpm"],
        "candidates": candidates,
    }
    if "range_bpm" in item:
        range_value = item["range_bpm"]
        if not isinstance(range_value, list) or len(range_value) != 2:
            _fail(f"{path}.range_bpm", "must be an ascending two-value array")
        low = _number(range_value[0], f"{path}.range_bpm[0]", minimum=20, maximum=400)
        high = _number(range_value[1], f"{path}.range_bpm[1]", minimum=20, maximum=400)
        if float(low) >= float(high):
            _fail(f"{path}.range_bpm", "must be ascending")
        out["range_bpm"] = [low, high]
    if "trend" in item:
        out["trend"] = _catalog_enum(item["trend"], f"{path}.trend", {"expressive", "accelerando"})
    if "curve_anchor_bpm" in item:
        out["curve_anchor_bpm"] = _number(item["curve_anchor_bpm"], f"{path}.curve_anchor_bpm", minimum=20, maximum=400)
    return out


def _catalog_meter(value: Any, path: str) -> dict[str, Any]:
    item = _object(value, path, {"status", "value"}, {"candidates", "regions", "feel"})
    if item["status"] not in {"fixed", "ambiguous", "mixed", "none"}:
        _fail(f"{path}.status", "has an unsupported catalog meter status")
    if item["value"] is not None:
        _integer(item["value"], f"{path}.value", minimum=1, maximum=32)
    out: dict[str, Any] = {"status": item["status"], "value": item["value"]}
    if "candidates" in item:
        if not isinstance(item["candidates"], list):
            _fail(f"{path}.candidates", "expected an array")
        candidates = []
        for index, candidate in enumerate(item["candidates"]):
            row = _object(candidate, f"{path}.candidates[{index}]", {"value", "label"})
            candidates.append(
                {
                    "value": _integer(row["value"], f"{path}.candidates[{index}].value", minimum=1, maximum=32),
                    "label": _string(row["label"], f"{path}.candidates[{index}].label", max_bytes=128),
                }
            )
        out["candidates"] = candidates
    if "regions" in item:
        if not isinstance(item["regions"], list) or item["regions"]:
            _fail(f"{path}.regions", "catalog regions must be empty in Phase 2")
        out["regions"] = []
    if "feel" in item:
        out["feel"] = _catalog_enum(item["feel"], f"{path}.feel", {"shuffle"})
    return out


def _catalog_structure(value: Any, path: str) -> dict[str, Any]:
    item = _object(value, path, {"status", "sections"})
    if item["status"] not in {"continuous", "sourced", "artist_validated"}:
        _fail(f"{path}.status", "has an unsupported catalog structure status")
    if not isinstance(item["sections"], list) or not 1 <= len(item["sections"]) <= 256:
        _fail(f"{path}.sections", "must contain 1 to 256 rows")
    out_sections: list[dict[str, Any]] = []
    previous = -float("inf")
    for index, section in enumerate(item["sections"]):
        row = _object(section, f"{path}.sections[{index}]", {"start", "label"})
        start = _number(row["start"], f"{path}.sections[{index}].start", minimum=0)
        label = _string(row["label"], f"{path}.sections[{index}].label", max_bytes=256)
        if float(start) < previous - 1e-6:
            _fail(f"{path}.sections", "rows must be ordered by start")
        previous = float(start)
        out_sections.append({"start": start, "label": label})
    return {"status": item["status"], "sections": out_sections}


def _catalog_evidence(value: Any, path: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or len(value) > 64:
        _fail(path, "must be an array of at most 64 citations")
    out: list[dict[str, str]] = []
    for index, citation in enumerate(value):
        row = _object(citation, f"{path}[{index}]", {"kind", "url"})
        kind = _catalog_enum(row["kind"], f"{path}[{index}].kind", _CATALOG_CITATION_KINDS)
        url = _string(row["url"], f"{path}[{index}].url", max_bytes=2048)
        if not url.startswith("https://"):
            _fail(f"{path}[{index}].url", "must be an absolute https:// URL")
        out.append({"kind": kind, "url": url})
    return out


def _catalog_enum(value: Any, path: str, allowed: set[str]) -> str:
    value = _string(value, path)
    if value not in allowed:
        _fail(path, f"must be one of {sorted(allowed)}")
    return value


def _catalog_row(value: Any, path: str) -> dict[str, Any]:
    item = _object(value, path, {"id", "title", "tempo", "meter"}, {"structure", "evidence"})
    row = {
        "id": _string(item["id"], f"{path}.id", max_bytes=256),
        "title": _string(item["title"], f"{path}.title", max_bytes=256),
        "tempo": _catalog_tempo(item["tempo"], f"{path}.tempo"),
        "meter": _catalog_meter(item["meter"], f"{path}.meter"),
    }
    if "structure" in item:
        row["structure"] = _catalog_structure(item["structure"], f"{path}.structure")
    if "evidence" in item:
        row["evidence"] = _catalog_evidence(item["evidence"], f"{path}.evidence")
    return row


def _load_catalog(catalog_record: object, track_sha256: str) -> tuple[dict[str, Any] | None, bool]:
    """Validate a bounded caller-supplied catalog and select one exact row.

    The accepted value is either the owned catalog document or a one-entry
    SHA-keyed mapping.  A complete valid document may legitimately have no
    row for this track.  A one-entry mapping is an explicit exact-record
    request, so a wrong key fails closed.  Every sibling in a complete
    document is validated before a selected row is returned.
    """
    size = len(canonical_json_bytes(catalog_record))
    if size > _CATALOG_MAX_BYTES:
        _fail("catalog_record", "catalog exceeds 1 MiB")
    if not isinstance(catalog_record, dict):
        _fail("catalog_record", "expected an owned catalog object")

    keys = set(catalog_record)
    if keys == set(_CATALOG_KEYS):
        document = _object(catalog_record, "catalog_record", set(_CATALOG_KEYS))
        if document["schema"] != _CATALOG_SCHEMA:
            _fail("catalog_record.schema", "must equal flux3.known-recordings/1")
        _string(document["description"], "catalog_record.description", max_bytes=4096)
        recordings = document["recordings"]
        if not isinstance(recordings, dict):
            _fail("catalog_record.recordings", "expected an object")
        normalized: dict[str, dict[str, Any]] = {}
        for key, row in recordings.items():
            if not isinstance(key, str) or _SHA256_RE.fullmatch(key) is None:
                _fail("catalog_record.recordings", "keys must be lowercase SHA-256 values")
            normalized[key] = _catalog_row(row, f"catalog_record.recordings[{key}]")
        selected = normalized.get(track_sha256)
        return selected, selected is not None

    # A direct row is deliberately not accepted: without a SHA-keyed outer
    # key there is no trustworthy way to establish exact-track provenance.
    if len(keys) != 1:
        _fail("catalog_record", "must be a complete catalog or one SHA-keyed row")
    key = next(iter(keys))
    if not isinstance(key, str) or _SHA256_RE.fullmatch(key) is None:
        _fail("catalog_record", "must be keyed by a lowercase SHA-256")
    row = _catalog_row(catalog_record[key], f"catalog_record[{key}]")
    if key != track_sha256:
        _fail("catalog_record", "row key does not match payload.source_sha256")
    return row, True


def _legacy_source(source: str) -> str:
    return "legacy_hybrid" if source == "recording_catalog" else "signal"


def _candidate(family: str, body: dict[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result["candidate_id"] = _stable_candidate_id(family, body)
    return result


def _engine_methods(payload: dict[str, Any]) -> tuple[str, str, str, str]:
    engine = payload.get("engine") if isinstance(payload.get("engine"), dict) else None
    return _ENGINE_METHODS[engine["id"] if engine else "librosa_v6"]


def _tempo_family(payload: dict[str, Any], duration: float) -> dict[str, Any]:
    global_tempo = payload["tempo_global"]
    # A variable tempo has no single grid BPM: the family is "ambiguous" so the map
    # never builds a regular grid from it without the artist's choice.
    status = "unavailable" if global_tempo["status"] == "none" else (
        "ambiguous" if global_tempo["status"] in {"ambiguous", "variable"} else "available"
    )
    candidates = []
    for source_row in global_tempo["candidates"]:
        body = {
            "bpm": source_row["bpm"],
            "relation": source_row["relation"],
            "source": _legacy_source(source_row["source"]),
            "method": _engine_methods(payload)[0],
            "confidence": source_row["confidence"],
            "selected_for_grid": source_row["selected_for_grid"],
        }
        candidates.append(_candidate("tempo", body))
    diagnostics = {
        "triplet_bpm": global_tempo.get("triplet_bpm"),
        "octave_margin": global_tempo.get("octave_margin"),
        "octave_ambiguous": global_tempo.get("octave_ambiguous"),
        "alt_bpm": global_tempo.get("alt_bpm"),
    }
    return {
        "status": status,
        "raw_curve": _v6_curve(payload["tempo_curve_raw"], "$.tempo_curve_raw", duration),
        "candidates": candidates,
        "legacy_selected": {
            "bpm": global_tempo["bpm"],
            "status": global_tempo["status"],
            "source": global_tempo["source"],
            "confidence": global_tempo["confidence"],
        },
        "legacy_curve_status": payload["tempo_curve_status"],
        "legacy_normalized_curve_source": "legacy_hybrid",
        "legacy_normalized_curve": _v6_curve(payload["tempo_curve"], "$.tempo_curve", duration),
        "legacy_diagnostics": diagnostics,
    }


def _meter_family(payload: dict[str, Any]) -> dict[str, Any]:
    meter = payload["meter"]
    status = "unavailable" if meter["status"] == "none" else (
        "ambiguous" if meter["status"] in {"ambiguous", "mixed"} else "available"
    )
    candidates = []
    for source_row in meter["candidates"]:
        body = {
            "beats_per_bar": source_row["value"],
            "label": source_row["label"],
            "source": _legacy_source(meter["source"]),
            "method": _engine_methods(payload)[3],
            "confidence": meter["confidence"],
        }
        candidates.append(_candidate("meter", body))
    return {
        "status": status,
        "candidates": candidates,
        "legacy_selected": {
            "beats_per_bar": meter["value"],
            "status": meter["status"],
            "assumed": meter["assumed"],
            "source": meter["source"],
            "confidence": meter["confidence"],
        },
        "regions": [],
    }


def _events(times: list[int | float], kind: str, *, source: str, method: str, confidence: int | float | None) -> list[dict[str, Any]]:
    return [
        {
            "event_id": f"{kind}.{index:06d}",
            "time_seconds": time_value,
            "source": source,
            "method": method,
            "confidence": confidence,
        }
        for index, time_value in enumerate(times, 1)
    ]


def _sections_family(payload: dict[str, Any], duration: float, overridden: bool) -> dict[str, Any]:
    if overridden:
        return {"status": "unavailable", "candidates": []}
    candidates = []
    for source_row in payload["sections"]:
        body = {
            "start_seconds": source_row["start"],
            "end_seconds": source_row["end"],
            "label": source_row["label"],
            "source": "signal",
            "method": "legacy_v6_sections",
            "confidence": source_row["confidence"],
        }
        candidates.append(_candidate("section", body))
    return {"status": "available" if candidates else "unavailable", "candidates": candidates}


def _accents_family(payload: dict[str, Any]) -> dict[str, Any]:
    accents = payload["accents"]
    raw_values = [float(row["salience"]) for row in accents]
    scale = max(raw_values, default=1.0)
    # The legacy producer's raw onset salience is unbounded, while the v1
    # neutral event contract is explicitly [0, 1].  Normalize only this
    # incompatible scale, preserving relative ordering; already-normalized
    # values are left unchanged.
    normalized = []
    for index, row in enumerate(accents, 1):
        salience = float(row["salience"]) / scale if scale > 1.0 else row["salience"]
        normalized.append(
            {
                "event_id": f"accent.{index:06d}",
                "time_seconds": row["time"],
                "strength": row["strength"],
                "salience": salience,
                "source": "signal",
                "method": "legacy_v6_accents",
                "confidence": None,
            }
        )
    return {"status": "available" if normalized else "unavailable", "events": normalized}


def _catalog_facts(row: dict[str, Any] | None) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "match": None,
        "recording_id": None,
        "title": None,
        "catalog_snapshot_hash": "",
        "tempo": None,
        "meter": None,
        "structure": None,
        "citations": [],
    }
    if row is not None:
        facts.update(
            {
                "match": "sha256",
                "recording_id": row["id"],
                "title": row["title"],
                "tempo": copy.deepcopy(row["tempo"]),
                "meter": copy.deepcopy(row["meter"]),
                "structure": copy.deepcopy(row.get("structure")),
                "citations": copy.deepcopy(row.get("evidence", [])),
            }
        )
    projection = {key: value for key, value in facts.items() if key != "catalog_snapshot_hash"}
    facts["catalog_snapshot_hash"] = _sha(projection)
    return facts


def _duration_places(value: Any) -> int | None:
    try:
        exponent = Decimal(str(value)).as_tuple().exponent
    except (InvalidOperation, ValueError):
        return None
    return max(0, -exponent) if isinstance(exponent, int) else None


def _audit_report(
    payload: dict[str, Any],
    raw_payload_sha256: str,
    loss_flags: list[str],
    sections_overridden: bool,
    catalog_match: bool,
) -> dict[str, Any]:
    duration_places = _duration_places(payload["duration"])
    return {
        "ok": True,
        "schema": _V6_SCHEMA,
        "top_level_keys": sorted(payload),
        "top_level_key_count": len(payload),
        "top_level_keys_exact": set(payload) - set(_V6_OPTIONAL_KEYS) == set(_V6_KEYS),
        "source_path_present": True,
        "analyzed_at_present": True,
        "analyzed_at": payload["analyzed_at"],
        "raw_payload_sha256": raw_payload_sha256,
        "integer_spans": {"excluded": True, "status": "application-specific"},
        "catalog_overridden": sections_overridden,
        "raw_signal_sections_lost_by_v6_override": sections_overridden,
        "duration_rounding": {
            "at_most_3_decimals": duration_places is not None and duration_places <= 3,
            "stored_decimal_places": duration_places,
        },
        "legacy_default_audio_stream": {"pinned_stream": False, "selected": True},
        "catalog_snapshot_absent": not catalog_match,
        "loss_flags": loss_flags,
    }


def _import_v6(payload: object, *, catalog_record: object | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    item = _validate_v6_payload(payload)
    raw_payload_sha256 = _sha(item)
    normalized_payload = copy.deepcopy(item)
    for field in ("source_path", "analyzed_at", "peaks", "integer_spans"):
        normalized_payload.pop(field, None)
    normalized_source_sha256 = _sha(normalized_payload)

    catalog_row: dict[str, Any] | None = None
    catalog_match = False
    if catalog_record is not None:
        catalog_row, catalog_match = _load_catalog(catalog_record, item["source_sha256"])

    sections_overridden = any("source" in row for row in item["sections"])
    mixed = (
        item["tempo_global"]["source"] == "recording_catalog"
        or item["meter"]["source"] == "recording_catalog"
        or sections_overridden
    )
    loss_flags = {
        "source_path_removed",
        "analyzed_at_removed",
        "duration_rounded_to_milliseconds",
        "legacy_runtime_unrecorded",
        "legacy_stream_unpinned",
        "application_projection_removed",
    }
    if sections_overridden:
        loss_flags.add("raw_signal_sections_lost_by_v6_override")
    if item["identified_recording"] is not None and not catalog_match:
        loss_flags.add("catalog_details_absent_from_v6")
    if catalog_match:
        loss_flags.add("catalog_snapshot_not_bound_to_v6_analysis")
    if mixed:
        loss_flags.add("legacy_selected_claims_mixed")
    sorted_loss_flags = sorted(loss_flags)
    if not loss_flags <= _LOSS_FLAGS:
        _fail("migration.loss_flags", "contains an unsupported loss flag")

    duration = float(item["duration"])
    sample_rate = item["sample_rate"]
    peaks = copy.deepcopy(item["peaks"])
    bucket_count = len(peaks)
    waveform = {
        "schema": "studio.music-waveform/1",
        "track_sha256": item["source_sha256"],
        "waveform_id": "",
        "requested_bucket_rate_hz": 200,
        "estimated_effective_bucket_rate_hz": round(bucket_count / duration, 6),
        "estimated_bucket_span_seconds": round(duration / bucket_count, 6),
        "bucket_count": bucket_count,
        "duration_seconds": item["duration"],
        "buckets": peaks,
    }
    waveform["waveform_id"] = compute_waveform_id(waveform)
    waveform = validate_waveform(waveform)

    tempo = _tempo_family(item, duration)
    meter = _meter_family(item)
    downbeat_source = "legacy_hybrid" if item["meter"]["source"] == "recording_catalog" else "signal"
    downbeat_dependencies = ["beats"]
    if downbeat_source == "legacy_hybrid":
        downbeat_dependencies.append("catalog_facts:meter")
    downbeats = {
        "status": "available" if item["downbeats"] else "unavailable",
        "events": _events(
            item["downbeats"],
            "downbeat",
            source=downbeat_source,
            method=_engine_methods(item)[2],
            confidence=item["downbeat_confidence"],
        ),
        "family_confidence": item["downbeat_confidence"],
        "dependencies": sorted(downbeat_dependencies),
    }
    beats = {
        "status": "available" if item["beats"] else "unavailable",
        "events": _events(item["beats"], "beat", source="signal", method=_engine_methods(item)[1], confidence=None),
    }
    energy_hop = round(sample_rate / 20)
    energy = {
        "status": "available" if item["energy"] else "unavailable",
        "samples": [
            {"time_seconds": row["time"], "value": row["value"]}
            for row in item["energy"]
        ],
        "requested_sample_rate_hz": 20,
        "hop_samples": energy_hop,
        "estimated_effective_sample_rate_hz": round(sample_rate / energy_hop, 6),
        "source": "signal",
        "method": "legacy_v6_energy",
    }
    tonality_available = item["key"] is not None and item["mode"] is not None
    tonality = {
        "status": "available" if tonality_available else "unavailable",
        "candidates": [],
    }
    if tonality_available:
        tonality["candidates"] = [
            _candidate(
                "tonality",
                {
                    "key": item["key"],
                    "mode": item["mode"],
                    "confidence": item["key_confidence"],
                    "source": "signal",
                    "method": "legacy_v6_key_mode",
                    "planning_constraint": False,
                },
            )
        ]
    brightness = {
        "status": "available",
        "hz": item["brightness"]["hz"],
        "normalized": item["brightness"]["normalized"],
        "source": "signal",
        "method": "legacy_v6_brightness",
    }
    legacy_identified = copy.deepcopy(item["identified_recording"])
    evidence = {
        "schema": "studio.music-evidence/1",
        "track_sha256": item["source_sha256"],
        "evidence_id": "",
        "importer": {"id": "studio.music-v6-import", "version": "1"},
        "analyzer": {
            "id": "legacy-music-v6",
            "version": "unrecorded",
            "parameter_hash": _PARAMETER_HASH,
            "parameters": copy.deepcopy(_PARAMETERS),
            "runtime": {"status": "unrecorded_legacy"},
            "decoded_audio": {
                "duration_seconds": item["duration"],
                "sample_rate_hz": sample_rate,
                "channels": 1,
                "stream_selector": "legacy_default_audio_stream",
            },
        },
        "waveform": {
            "waveform_id": waveform["waveform_id"],
            "requested_bucket_rate_hz": waveform["requested_bucket_rate_hz"],
            "estimated_effective_bucket_rate_hz": waveform["estimated_effective_bucket_rate_hz"],
            "estimated_bucket_span_seconds": waveform["estimated_bucket_span_seconds"],
            "bucket_count": waveform["bucket_count"],
            "encoding": "minmax-f32-json/1",
        },
        "families": {
            "tempo": tempo,
            "beats": beats,
            "meter": meter,
            "downbeats": downbeats,
            "sections": _sections_family(item, duration, sections_overridden),
            "accents": _accents_family(item),
            "energy": energy,
            "tonality": tonality,
            "brightness": brightness,
            "catalog_facts": _catalog_facts(catalog_row),
            "lyrics_alignment_candidates": [],
        },
        "migration": {
            "source_schema": _V6_SCHEMA,
            "normalized_source_sha256": normalized_source_sha256,
            "legacy_identified_recording": legacy_identified,
            "loss_flags": sorted_loss_flags,
        },
    }
    evidence["evidence_id"] = compute_evidence_id(evidence)
    evidence = validate_evidence(evidence)
    audit = _audit_report(item, raw_payload_sha256, sorted_loss_flags, sections_overridden, catalog_match)
    return evidence, waveform, audit


def import_v6(payload: object, *, catalog_record: object | None = None) -> tuple[dict, dict, dict]:
    """Import one complete v6 cache object into Evidence, Waveform and audit.

    ``catalog_record`` is optional.  When supplied it must be a strict owned
    catalog document or a one-entry SHA-keyed catalog mapping; no catalog file
    is consulted implicitly.
    """
    try:
        return _import_v6(payload, catalog_record=catalog_record)
    except MusicContractError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise MusicContractError(f"payload: invalid flux3.music/6 object ({exc})") from exc


__all__ = ["import_v6"]
