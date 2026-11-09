"""Strict Type-C contracts for neutral music artifacts.

The module deliberately contains no I/O or application-specific imports.  The
validators accept JSON-shaped values, return a detached normalized copy, and
fail closed on every object boundary.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from datetime import datetime
from decimal import Context, Decimal, DecimalException, Inexact, ROUND_HALF_EVEN, Rounded, localcontext
from fractions import Fraction
from typing import Any


class MusicContractError(ValueError):
    """Raised when a value does not satisfy the frozen Type-C contract."""


class MusicCandidateMissingError(MusicContractError):
    """Raised when an authored tempo candidate is absent from bound Evidence."""

    reason_code = "music_candidate_missing"


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,95}\Z")
_ANALYZER_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_EVENT_ID_RE = re.compile(r"(?:beat|downbeat|accent)\.[0-9]{6}\Z")
_BAR_ID_RE = re.compile(r"bar\.[0-9]{6}\Z")
_TIMESTAMP_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")
_PROJECT_RE = re.compile(r"prj_[0-9a-f]{32}\Z")
_LANGUAGE_RE = re.compile(r"[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*\Z")

_MAX_DEPTH = 12
_MAX_KEYS = 256
_MAX_STRING_BYTES = 1024 * 1024
_MAX_LIST = 2_000_000
_EVIDENCE_BYTES = 64 * 1024 * 1024
_WAVEFORM_BYTES = 128 * 1024 * 1024
_INTERPRETATION_BYTES = 4 * 1024 * 1024
_MAP_BYTES = 64 * 1024 * 1024
_TIME_TOLERANCE = Decimal("0.000001")
_UNIT_FACTORS = {
    "whole": Fraction(4, 1),
    "half": Fraction(2, 1),
    "quarter": Fraction(1, 1),
    "eighth": Fraction(1, 2),
    "dotted_quarter": Fraction(3, 2),
}
_BAR_CAPABLE_METER_STATUSES = {"fixed", "mixed", "beats_per_bar_only"}

_TEMPO_RELATIONS = {
    "detected", "alternate", "four_over_three", "primary", "half", "double",
    "published_candidate", "detected_candidate", "dj_full_tempo_candidate",
    "artist_grid", "local_grid", "catalogued",
}
_LOSS_FLAGS = {
    "source_path_removed", "analyzed_at_removed", "duration_rounded_to_milliseconds",
    "legacy_runtime_unrecorded", "legacy_stream_unpinned", "application_projection_removed",
    "raw_signal_sections_lost_by_v6_override", "catalog_details_absent_from_v6",
    "catalog_snapshot_not_bound_to_v6_analysis", "legacy_selected_claims_mixed",
}
_WARNING_ORDER = {
    "selected_candidate_missing": 0,
    "regular_grid_anchor_missing": 1,
    "bar_math_withheld": 2,
    "legacy_evidence_loss": 3,
    "catalog_snapshot_absent": 4,
}
_SOURCE_REF_RE = re.compile(
    r"(?:"
    r"interpretation:(?:tempo\.(?:mode|selected_candidate_id|bpm|beat_unit|grid_source)|"
    r"meter\.(?:mode|numerator|denominator)|meter\.region\." + _ID_RE.pattern[:-2] + r"|"
    r"downbeat\.(?:mode|anchor_seconds)|section\." + _ID_RE.pattern[:-2] + r"|lyrics\.text_sha256)|"
    r"evidence:(?:" + _ID_RE.pattern[:-2] + r"|energy|migration\.loss_flags)|"
    r"catalog:(?:tempo|meter|structure|catalog_snapshot_hash)|"
    r"map:(?:tempo|meter|downbeat\.[0-9]{6})"
    r")\Z"
)


def _fail(path: str, message: str) -> None:
    raise MusicContractError(f"{path}: {message}")


def _object(value: Any, path: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "expected an object")
    actual = set(value)
    unknown = actual - keys
    missing = keys - actual
    if unknown:
        _fail(path, f"unknown key(s): {', '.join(sorted(map(str, unknown)))}")
    if missing:
        _fail(path, f"missing key(s): {', '.join(sorted(missing))}")
    if any(not isinstance(key, str) for key in value):
        _fail(path, "object keys must be strings")
    return value


def _object_with_optional(value: Any, path: str, required: set[str], optional: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "expected an object")
    actual = set(value)
    unknown = actual - required - optional
    missing = required - actual
    if unknown:
        _fail(path, f"unknown key(s): {', '.join(sorted(map(str, unknown)))}")
    if missing:
        _fail(path, f"missing key(s): {', '.join(sorted(missing))}")
    if any(not isinstance(key, str) for key in value):
        _fail(path, "object keys must be strings")
    return value


def _optional_object(value: Any, path: str, keys: set[str]) -> dict[str, Any] | None:
    if value is None:
        return None
    return _object(value, path, keys)


def _string(value: Any, path: str, *, max_bytes: int = _MAX_STRING_BYTES, ascii_only: bool = False,
            allow_controls: bool = False, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        _fail(path, "expected a string")
    value = str.__str__(value)
    if nonempty and not value:
        _fail(path, "must not be empty")
    if "\x00" in value:
        _fail(path, "NUL is not allowed")
    if allow_controls:
        has_forbidden_control = any(ord(char) < 32 and char not in "\n\r\t" for char in value)
    else:
        has_forbidden_control = any(ord(char) < 32 for char in value)
    if has_forbidden_control:
        _fail(path, "C0 control characters are not allowed")
    if ascii_only:
        try:
            value.encode("ascii")
        except UnicodeEncodeError:
            _fail(path, "must be ASCII")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        _fail(path, f"must be valid UTF-8: {exc}")
    if size > max_bytes:
        _fail(path, f"UTF-8 length exceeds {max_bytes} bytes")
    return value


def _numeric_projection_and_value(value: int | float, path: str) -> tuple[Decimal, int | float]:
    """Project a JSON number without invoking numeric-subclass hooks."""
    try:
        encoded = _canonical_json_bytes_low_level(value)
        text = encoded.decode("ascii")
        projection = Decimal(text)
        normalized = json.loads(text, parse_int=int, parse_float=float)
    except (MusicContractError, DecimalException, UnicodeDecodeError, OverflowError, TypeError, ValueError) as exc:
        _fail(path, f"must be a valid finite JSON number: {exc}")
    if not isinstance(normalized, (int, float)) or isinstance(normalized, bool) or not projection.is_finite():
        _fail(path, "must be a finite JSON number")
    if type(value) in (int, float):
        return projection, value
    return projection, normalized


def _number(value: Any, path: str, *, minimum: float | None = None, maximum: float | None = None) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "expected a JSON number (boolean is not a number)")
    projection, normalized = _numeric_projection_and_value(value, path)
    if minimum is not None and projection < _canonical_decimal(minimum):
        _fail(path, f"must be >= {minimum}")
    if maximum is not None and projection > _canonical_decimal(maximum):
        _fail(path, f"must be <= {maximum}")
    return normalized


def _integer(value: Any, path: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, "expected an integer (boolean is not an integer)")
    projection, normalized = _numeric_projection_and_value(value, path)
    if not isinstance(normalized, int) or isinstance(normalized, bool):
        _fail(path, "expected an integer (boolean is not an integer)")
    if minimum is not None and projection < _canonical_decimal(minimum):
        _fail(path, f"must be >= {minimum}")
    if maximum is not None and projection > _canonical_decimal(maximum):
        _fail(path, f"must be <= {maximum}")
    return normalized


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, "expected a JSON boolean")
    return value


def _enum(value: Any, path: str, allowed: set[str]) -> str:
    value = _string(value, path)
    if value not in allowed:
        _fail(path, f"must be one of {sorted(allowed)}")
    return value


def _array(value: Any, path: str, *, maximum: int = _MAX_LIST) -> list[Any]:
    if not isinstance(value, list):
        _fail(path, "expected an array")
    if len(value) > maximum:
        _fail(path, f"array length exceeds {maximum}")
    return value


def _walk_json(value: Any, path: str = "$", depth: int = 1) -> None:
    if depth > _MAX_DEPTH:
        _fail(path, f"JSON depth exceeds {_MAX_DEPTH}")
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, str):
            value = str.__str__(value)
            try:
                encoded_size = len(value.encode("utf-8"))
            except UnicodeEncodeError as exc:
                _fail(path, f"must be valid UTF-8: {exc}")
            if encoded_size > _MAX_STRING_BYTES:
                _fail(path, f"UTF-8 length exceeds {_MAX_STRING_BYTES} bytes")
        elif isinstance(value, float) and not math.isfinite(value):
            _fail(path, "must be finite")
        return
    if isinstance(value, list):
        if len(value) > _MAX_LIST:
            _fail(path, f"array length exceeds {_MAX_LIST}")
        for index, item in enumerate(value):
            _walk_json(item, f"{path}[{index}]", depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > _MAX_KEYS:
            _fail(path, f"object key count exceeds {_MAX_KEYS}")
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(path, "object keys must be strings")
            _string(key, f"{path}.<key>", nonempty=False)
            _walk_json(item, f"{path}.{key}", depth + 1)
        return
    _fail(path, f"unsupported JSON value type {type(value).__name__}")


def _canonical_json_bytes_low_level(value: Any) -> bytes:
    """Serialize detached built-ins, or one scalar numeric value, only."""
    if isinstance(value, int) and not isinstance(value, bool):
        text = int.__repr__(value)
        return text.encode("ascii")
    if isinstance(value, float):
        text = float.__repr__(value)
        return text.encode("ascii")
    _walk_json(value)
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return text.encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise MusicContractError(f"$: cannot canonicalize JSON: {exc}") from exc


def canonical_json_bytes(value: Any) -> bytes:
    """Return canonical UTF-8 JSON bytes for a finite JSON-shaped value."""
    return _canonical_json_bytes_low_level(_detached(value))


def _canonical_numeric_projection(value: int | float) -> Decimal:
    return Decimal(canonical_json_bytes(value).decode("ascii"))


def _published_numeric_projection(value: Decimal) -> Decimal:
    """Project an internal Decimal exactly as the resolver publishes it."""
    return _canonical_numeric_projection(float(value))


def _canonical_decimal(value: int | float) -> Decimal:
    """Parse a validated JSON number without crossing a binary-float boundary."""
    return Decimal(canonical_json_bytes(value).decode("ascii"))


def _fresh_decimal_context(*values: Decimal) -> Context:
    """Create a load-bearing context large enough for the supplied magnitudes."""
    magnitude_digits = max(
        (max(len(value.as_tuple().digits), value.adjusted() + 1) for value in values),
        default=1,
    )
    return Context(prec=max(50, magnitude_digits + 32), rounding=ROUND_HALF_EVEN)


def _waveform_estimates(duration: int | float, bucket_count: int) -> tuple[Decimal, Decimal]:
    """Return the canonical six-place waveform rate/span equations."""
    decimal_duration = _canonical_decimal(duration)
    decimal_count = _canonical_decimal(bucket_count)
    context = _fresh_decimal_context(decimal_duration, decimal_count)
    context.traps[Inexact] = False
    context.traps[Rounded] = False
    with localcontext(context):
        quantum = Decimal("0.000001")
        return (
            (decimal_count / decimal_duration).quantize(quantum, rounding=ROUND_HALF_EVEN),
            (decimal_duration / decimal_count).quantize(quantum, rounding=ROUND_HALF_EVEN),
        )


def _waveform_value_matches(value: int | float, expected: Decimal) -> bool:
    """Match exact equations while accepting their JSON-number publication."""
    actual = _canonical_numeric_projection(value)
    if actual == expected:
        return True
    try:
        published = _published_numeric_projection(expected)
    except (OverflowError, ValueError):
        return False
    return actual == published


def _legacy_waveform_estimates(duration: int | float, bucket_count: int) -> tuple[Decimal, Decimal] | None:
    """Match the frozen importer’s Python-round publication from canonical input."""
    canonical_duration = _canonical_numeric_projection(duration)
    try:
        runtime_duration = float(canonical_duration)
    except (OverflowError, TypeError, ValueError):
        return None
    if not math.isfinite(runtime_duration) or (canonical_duration > 0 and runtime_duration <= 0):
        return None
    try:
        rate = round(bucket_count / runtime_duration, 6)
        span = round(runtime_duration / bucket_count, 6)
    except (OverflowError, TypeError, ValueError, ZeroDivisionError):
        return None
    return _canonical_numeric_projection(rate), _canonical_numeric_projection(span)


def _waveform_equations_match(
    duration: int | float,
    bucket_count: int,
    rate: int | float,
    span: int | float,
) -> bool:
    try:
        expected_rate, expected_span = _waveform_estimates(duration, bucket_count)
        if (_waveform_value_matches(rate, expected_rate) and
                _waveform_value_matches(span, expected_span)):
            return True
        legacy = _legacy_waveform_estimates(duration, bucket_count)
        return legacy is not None and _waveform_value_matches(rate, legacy[0]) and _waveform_value_matches(span, legacy[1])
    except (DecimalException, OverflowError, TypeError, ValueError, ZeroDivisionError):
        return False


def _timeline_before(left: Decimal, right: Decimal) -> bool:
    """Return whether ``left`` is earlier than ``right`` beyond the timeline tolerance."""
    left_numerator, left_denominator = left.as_integer_ratio()
    right_numerator, right_denominator = right.as_integer_ratio()
    tolerance_numerator, tolerance_denominator = _TIME_TOLERANCE.as_integer_ratio()
    return (
        (left_numerator * tolerance_denominator + tolerance_numerator * left_denominator) * right_denominator
        < right_numerator * left_denominator * tolerance_denominator
    )


def _timeline_after(left: Decimal, right: Decimal) -> bool:
    """Return whether ``left`` is later than ``right`` beyond the timeline tolerance."""
    return _timeline_before(right, left)


def _detached(value: Any, path: str = "$", depth: int = 1) -> Any:
    """Detach bounded JSON-shaped data without invoking user-defined hooks."""
    if depth > _MAX_DEPTH:
        _fail(path, f"JSON depth exceeds {_MAX_DEPTH}")
    if value is None or type(value) is bool:
        return value
    if isinstance(value, str):
        try:
            normalized = str.__str__(value)
        except Exception as exc:
            _fail(path, f"cannot normalize string: {exc}")
        if type(normalized) is not str:
            _fail(path, "string normalization did not produce a built-in string")
        return normalized
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _numeric_projection_and_value(value, path)[1]
    if isinstance(value, list):
        try:
            length = list.__len__(value)
        except Exception as exc:
            _fail(path, f"cannot inspect array: {exc}")
        if length > _MAX_LIST:
            _fail(path, f"array length exceeds {_MAX_LIST}")
        output: list[Any] = []
        for index in range(length):
            try:
                item = list.__getitem__(value, index)
            except Exception as exc:
                _fail(f"{path}[{index}]", f"cannot read array item: {exc}")
            output.append(_detached(item, f"{path}[{index}]", depth + 1))
        return output
    if isinstance(value, dict):
        try:
            length = dict.__len__(value)
        except Exception as exc:
            _fail(path, f"cannot inspect object: {exc}")
        if length > _MAX_KEYS:
            _fail(path, f"object key count exceeds {_MAX_KEYS}")
        output: dict[str, Any] = {}
        try:
            items = dict.items(value)
            iterator = iter(items)
        except Exception as exc:
            _fail(path, f"cannot inspect object entries: {exc}")
        for key, item in iterator:
            normalized_key = _detached(key, f"{path}.<key>", depth + 1)
            if type(normalized_key) is not str:
                _fail(f"{path}.<key>", "object keys must be strings")
            if normalized_key in output:
                _fail(f"{path}.<key>", "duplicate keys after normalization")
            output[normalized_key] = _detached(item, f"{path}.{normalized_key}", depth + 1)
        return output
    _fail(path, f"unsupported JSON value type {type(value).__name__}")


def _artifact_size(value: dict[str, Any], cap: int, name: str) -> None:
    size = len(canonical_json_bytes(value))
    if size > cap:
        _fail("$", f"{name} artifact exceeds {cap} canonical bytes")


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _validate_sha(value: Any, path: str) -> str:
    value = _string(value, path, max_bytes=64, ascii_only=True)
    if _SHA256_RE.fullmatch(value) is None:
        _fail(path, "must be exactly 64 lowercase hexadecimal characters")
    return value


def _validate_id(value: Any, path: str, *, prefix: str | None = None) -> str:
    value = _string(value, path, max_bytes=96, ascii_only=True)
    if _ID_RE.fullmatch(value) is None:
        _fail(path, "has an invalid stable ID")
    if prefix is not None and not value.startswith(prefix):
        _fail(path, f"must start with {prefix!r}")
    return value


def _time(value: Any, path: str, duration: int | float | None = None) -> int | float:
    value = _number(value, path)
    projection = _canonical_numeric_projection(value)
    if _timeline_before(projection, Decimal(0)):
        _fail(path, "is before the track start")
    if duration is not None and _timeline_after(projection, _canonical_numeric_projection(duration)):
        _fail(path, "is outside the track duration")
    return value


def _time_range(start: Any, end: Any, path: str, duration: int | float | None = None) -> tuple[int | float, int | float]:
    start_value = _time(start, f"{path}.start_seconds", duration)
    end_value = _time(end, f"{path}.end_seconds", duration)
    if _timeline_after(_canonical_numeric_projection(start_value), _canonical_numeric_projection(end_value)):
        _fail(path, "start_seconds must be <= end_seconds")
    return start_value, end_value


def _confidence(value: Any, path: str, *, nullable: bool = True) -> int | float | None:
    if value is None and nullable:
        return None
    return _number(value, path, minimum=0, maximum=1)


def _validate_timestamp(value: Any, path: str) -> str:
    value = _string(value, path, max_bytes=20, ascii_only=True)
    if _TIMESTAMP_RE.fullmatch(value) is None:
        _fail(path, "must use UTC YYYY-MM-DDTHH:MM:SSZ")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        _fail(path, f"is not a valid UTC timestamp: {exc}")
    return value


def _candidate_id(family: str, body: dict[str, Any]) -> str:
    return f"{family}.{_sha(body)}"


def _validate_candidate_rows(value: Any, path: str, family: str, keys: set[str],
                             body_validator, *, maximum: int = 2_000_000) -> list[dict[str, Any]]:
    rows = _array(value, path, maximum=maximum)
    result: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    for index, row in enumerate(rows):
        item = _object(row, f"{path}[{index}]", keys)
        normalized = body_validator(item, f"{path}[{index}]")
        supplied = _validate_id(item["candidate_id"], f"{path}[{index}].candidate_id", prefix=f"{family}.")
        expected = _candidate_id(family, {key: normalized[key] for key in normalized if key != "candidate_id"})
        if supplied != expected:
            _fail(f"{path}[{index}].candidate_id", "does not match candidate content")
        body_bytes = canonical_json_bytes({key: normalized[key] for key in normalized if key != "candidate_id"})
        if body_bytes in seen:
            continue
        seen.add(body_bytes)
        normalized["candidate_id"] = supplied
        result.append(normalized)
    return result


def _event_rows(value: Any, path: str, kind: str, duration: int | float | None, *, accents: bool = False) -> list[dict[str, Any]]:
    rows = _array(value, path)
    result: list[tuple[int, dict[str, Any]]] = []
    keys = {"event_id", "time_seconds", "source", "method", "confidence"}
    if accents:
        keys |= {"strength", "salience"}
    for index, row in enumerate(rows):
        item = _object(row, f"{path}[{index}]", keys)
        event_id = _string(item["event_id"], f"{path}[{index}].event_id", max_bytes=32, ascii_only=True)
        if _EVENT_ID_RE.fullmatch(event_id) is None or not event_id.startswith(f"{kind}."):
            _fail(f"{path}[{index}].event_id", f"must be a {kind} ordinal ID")
        normalized: dict[str, Any] = {
            "event_id": event_id,
            "time_seconds": _time(item["time_seconds"], f"{path}[{index}].time_seconds", duration),
            "source": _enum(item["source"], f"{path}[{index}].source", {"signal", "recording_catalog", "legacy_hybrid", "artist"}),
            "method": _string(item["method"], f"{path}[{index}].method", max_bytes=128),
            "confidence": _confidence(item["confidence"], f"{path}[{index}].confidence"),
        }
        if accents:
            normalized["strength"] = _number(item["strength"], f"{path}[{index}].strength", minimum=0, maximum=1)
            normalized["salience"] = _number(item["salience"], f"{path}[{index}].salience", minimum=0, maximum=1)
        result.append((index, normalized))
    if accents:
        result.sort(key=lambda pair: (_canonical_numeric_projection(pair[1]["time_seconds"]), _canonical_numeric_projection(pair[1]["strength"]), _canonical_numeric_projection(pair[1]["salience"]), pair[0]))
    else:
        result.sort(key=lambda pair: (_canonical_numeric_projection(pair[1]["time_seconds"]), pair[0]))
    output: list[dict[str, Any]] = []
    for ordinal, (_, row) in enumerate(result, 1):
        row["event_id"] = f"{kind}.{ordinal:06d}"
        output.append(row)
    return output


def _curve_rows(value: Any, path: str, duration: int | float) -> list[dict[str, Any]]:
    rows = _array(value, path)
    out: list[dict[str, Any]] = []
    previous: Decimal | None = None
    for index, row in enumerate(rows):
        item = _object(row, f"{path}[{index}]", {"time_seconds", "bpm"})
        time_value = _time(item["time_seconds"], f"{path}[{index}].time_seconds", duration)
        projection = _canonical_numeric_projection(time_value)
        if previous is not None and _timeline_before(projection, previous):
            _fail(path, "rows must be ordered by time_seconds")
        previous = projection
        out.append({"time_seconds": time_value, "bpm": _number(item["bpm"], f"{path}[{index}].bpm", minimum=0, maximum=400)})
    return out


def _candidate_common(item: dict[str, Any], path: str, *, body_keys: set[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"candidate_id": item["candidate_id"]}
    for key in body_keys:
        if key == "candidate_id":
            continue
        result[key] = item[key]
    result["source"] = _enum(item["source"], f"{path}.source", {"signal", "recording_catalog", "legacy_hybrid", "artist"})
    result["method"] = _string(item["method"], f"{path}.method", max_bytes=128)
    result["confidence"] = _confidence(item["confidence"], f"{path}.confidence")
    return result


def _tempo_candidate(item: dict[str, Any], path: str) -> dict[str, Any]:
    out = _candidate_common(item, path, body_keys={"candidate_id", "bpm", "relation", "source", "method", "confidence", "selected_for_grid"})
    out["bpm"] = _number(item["bpm"], f"{path}.bpm", minimum=20, maximum=400)
    out["relation"] = _enum(item["relation"], f"{path}.relation", _TEMPO_RELATIONS)
    out["selected_for_grid"] = _boolean(item["selected_for_grid"], f"{path}.selected_for_grid")
    return out


def _meter_candidate(item: dict[str, Any], path: str) -> dict[str, Any]:
    out = _candidate_common(item, path, body_keys={"candidate_id", "beats_per_bar", "label", "source", "method", "confidence"})
    out["beats_per_bar"] = _integer(item["beats_per_bar"], f"{path}.beats_per_bar", minimum=1, maximum=32)
    out["label"] = _string(item["label"], f"{path}.label", max_bytes=128)
    return out


def _section_candidate(item: dict[str, Any], path: str, duration: float) -> dict[str, Any]:
    out = _candidate_common(item, path, body_keys={"candidate_id", "start_seconds", "end_seconds", "label", "source", "method", "confidence"})
    out["start_seconds"], out["end_seconds"] = _time_range(item["start_seconds"], item["end_seconds"], path, duration)
    out["label"] = _string(item["label"], f"{path}.label", max_bytes=256)
    return out


def _tonality_candidate(item: dict[str, Any], path: str) -> dict[str, Any]:
    keys = {"candidate_id", "key", "mode", "confidence", "source", "method", "planning_constraint"}
    out = _candidate_common(item, path, body_keys=keys)
    out["key"] = _string(item["key"], f"{path}.key", max_bytes=32)
    out["mode"] = _string(item["mode"], f"{path}.mode", max_bytes=32)
    out["planning_constraint"] = _boolean(item["planning_constraint"], f"{path}.planning_constraint")
    if out["planning_constraint"] is not False:
        _fail(f"{path}.planning_constraint", "must be false")
    return out


def _validate_tempo(value: Any, path: str, duration: int | float) -> dict[str, Any]:
    keys = {"status", "raw_curve", "candidates", "legacy_selected", "legacy_curve_status", "legacy_normalized_curve_source", "legacy_normalized_curve", "legacy_diagnostics"}
    item = _object(value, path, keys)
    status = _enum(item["status"], f"{path}.status", {"available", "ambiguous", "unavailable"})
    out = {
        "status": status,
        "raw_curve": _curve_rows(item["raw_curve"], f"{path}.raw_curve", duration),
        "candidates": _validate_candidate_rows(item["candidates"], f"{path}.candidates", "tempo", {"candidate_id", "bpm", "relation", "source", "method", "confidence", "selected_for_grid"}, _tempo_candidate),
        "legacy_selected": None,
        "legacy_curve_status": _enum(item["legacy_curve_status"], f"{path}.legacy_curve_status", {"normalized", "unavailable", "suppressed_no_stable_pulse"}),
        "legacy_normalized_curve_source": _enum(item["legacy_normalized_curve_source"], f"{path}.legacy_normalized_curve_source", {"legacy_hybrid"}),
        "legacy_normalized_curve": _curve_rows(item["legacy_normalized_curve"], f"{path}.legacy_normalized_curve", duration),
        "legacy_diagnostics": None,
    }
    selected = _object(item["legacy_selected"], f"{path}.legacy_selected", {"bpm", "status", "source", "confidence"})
    out["legacy_selected"] = {
        "bpm": None if selected["bpm"] is None else _number(selected["bpm"], f"{path}.legacy_selected.bpm", minimum=20, maximum=400),
        "status": _enum(selected["status"], f"{path}.legacy_selected.status", {"fixed", "ambiguous", "variable", "none"}),
        "source": _enum(selected["source"], f"{path}.legacy_selected.source", {"signal", "recording_catalog"}),
        "confidence": _confidence(selected["confidence"], f"{path}.legacy_selected.confidence"),
    }
    diagnostics = _object(item["legacy_diagnostics"], f"{path}.legacy_diagnostics", {"triplet_bpm", "octave_margin", "octave_ambiguous", "alt_bpm"})
    out["legacy_diagnostics"] = {
        "triplet_bpm": None if diagnostics["triplet_bpm"] is None else _number(diagnostics["triplet_bpm"], f"{path}.legacy_diagnostics.triplet_bpm"),
        "octave_margin": None if diagnostics["octave_margin"] is None else _number(diagnostics["octave_margin"], f"{path}.legacy_diagnostics.octave_margin"),
        "octave_ambiguous": None if diagnostics["octave_ambiguous"] is None else _boolean(diagnostics["octave_ambiguous"], f"{path}.legacy_diagnostics.octave_ambiguous"),
        "alt_bpm": None if diagnostics["alt_bpm"] is None else _number(diagnostics["alt_bpm"], f"{path}.legacy_diagnostics.alt_bpm"),
    }
    return out


def _validate_beats(value: Any, path: str, duration: float) -> dict[str, Any]:
    item = _object(value, path, {"status", "events"})
    return {"status": _enum(item["status"], f"{path}.status", {"available", "ambiguous", "unavailable"}),
            "events": _event_rows(item["events"], f"{path}.events", "beat", duration)}


def _validate_meter(value: Any, path: str) -> dict[str, Any]:
    item = _object(value, path, {"status", "candidates", "legacy_selected", "regions"})
    out = {
        "status": _enum(item["status"], f"{path}.status", {"available", "ambiguous", "unavailable"}),
        "candidates": _validate_candidate_rows(item["candidates"], f"{path}.candidates", "meter", {"candidate_id", "beats_per_bar", "label", "source", "method", "confidence"}, _meter_candidate),
        "legacy_selected": None,
        "regions": _array(item["regions"], f"{path}.regions", maximum=256),
    }
    selected = _object(item["legacy_selected"], f"{path}.legacy_selected", {"beats_per_bar", "status", "assumed", "source", "confidence"})
    out["legacy_selected"] = {
        "beats_per_bar": None if selected["beats_per_bar"] is None else _integer(selected["beats_per_bar"], f"{path}.legacy_selected.beats_per_bar", minimum=1, maximum=32),
        "status": _enum(selected["status"], f"{path}.legacy_selected.status", {"fixed", "mixed", "ambiguous", "none"}),
        "assumed": _boolean(selected["assumed"], f"{path}.legacy_selected.assumed"),
        "source": _enum(selected["source"], f"{path}.legacy_selected.source", {"signal", "recording_catalog"}),
        "confidence": _confidence(selected["confidence"], f"{path}.legacy_selected.confidence"),
    }
    if out["regions"]:
        _fail(f"{path}.regions", "must be empty in the v1 v6 evidence contract")
    # Evidence v6 meter regions are not populated by Phase 2.
    for index, region in enumerate(out["regions"]):
        if not isinstance(region, dict):
            _fail(f"{path}.regions[{index}]", "expected an object")
        _walk_json(region, f"{path}.regions[{index}]")
    return out


def _validate_downbeats(value: Any, path: str, duration: float) -> dict[str, Any]:
    item = _object(value, path, {"status", "events", "family_confidence", "dependencies"})
    deps = _array(item["dependencies"], f"{path}.dependencies", maximum=16)
    dependencies = [_string(dep, f"{path}.dependencies[{index}]", max_bytes=128, ascii_only=True) for index, dep in enumerate(deps)]
    if dependencies != sorted(set(dependencies)):
        _fail(f"{path}.dependencies", "must be sorted and unique")
    return {"status": _enum(item["status"], f"{path}.status", {"available", "ambiguous", "unavailable"}),
            "events": _event_rows(item["events"], f"{path}.events", "downbeat", duration),
            "family_confidence": _confidence(item["family_confidence"], f"{path}.family_confidence"),
            "dependencies": dependencies}


def _validate_sections(value: Any, path: str, duration: float) -> dict[str, Any]:
    item = _object(value, path, {"status", "candidates"})
    return {"status": _enum(item["status"], f"{path}.status", {"available", "ambiguous", "unavailable"}),
            "candidates": _validate_candidate_rows(item["candidates"], f"{path}.candidates", "section", {"candidate_id", "start_seconds", "end_seconds", "label", "source", "method", "confidence"}, lambda row, row_path: _section_candidate(row, row_path, duration))}


def _validate_accents(value: Any, path: str, duration: float) -> dict[str, Any]:
    item = _object(value, path, {"status", "events"})
    return {"status": _enum(item["status"], f"{path}.status", {"available", "ambiguous", "unavailable"}),
            "events": _event_rows(item["events"], f"{path}.events", "accent", duration, accents=True)}


def _validate_energy(value: Any, path: str, duration: float, sample_rate: int | None = None) -> dict[str, Any]:
    item = _object(value, path, {"status", "samples", "requested_sample_rate_hz", "hop_samples", "estimated_effective_sample_rate_hz", "source", "method"})
    status = _enum(item["status"], f"{path}.status", {"available", "unavailable"})
    samples = _array(item["samples"], f"{path}.samples")
    output_samples: list[dict[str, Any]] = []
    previous: Decimal | None = None
    for index, sample in enumerate(samples):
        row = _object(sample, f"{path}.samples[{index}]", {"time_seconds", "value"})
        time_value = _time(row["time_seconds"], f"{path}.samples[{index}].time_seconds", duration)
        projection = _canonical_numeric_projection(time_value)
        if previous is not None and _timeline_before(projection, previous):
            _fail(f"{path}.samples", "rows must be ordered by time_seconds")
        previous = projection
        output_samples.append({"time_seconds": time_value, "value": _number(row["value"], f"{path}.samples[{index}].value", minimum=0)})
    requested = _integer(item["requested_sample_rate_hz"], f"{path}.requested_sample_rate_hz", minimum=1)
    hop = _integer(item["hop_samples"], f"{path}.hop_samples", minimum=1)
    estimate = _number(item["estimated_effective_sample_rate_hz"], f"{path}.estimated_effective_sample_rate_hz", minimum=0)
    if sample_rate is not None:
        expected_hop = round(sample_rate / 20)
        if requested != 20 or hop != expected_hop:
            _fail(path, "legacy energy metadata does not match the 20 Hz hop equation")
        expected = round(sample_rate / hop, 6)
        if float(estimate) != expected:
            _fail(f"{path}.estimated_effective_sample_rate_hz", "does not match the sample-rate equation")
    return {"status": status, "samples": output_samples, "requested_sample_rate_hz": requested, "hop_samples": hop,
            "estimated_effective_sample_rate_hz": estimate,
            "source": _enum(item["source"], f"{path}.source", {"signal", "recording_catalog", "legacy_hybrid", "artist"}),
            "method": _string(item["method"], f"{path}.method", max_bytes=128)}


def _validate_tonality(value: Any, path: str) -> dict[str, Any]:
    item = _object(value, path, {"status", "candidates"})
    status = _enum(item["status"], f"{path}.status", {"available", "unavailable"})
    candidates = _validate_candidate_rows(item["candidates"], f"{path}.candidates", "tonality", {"candidate_id", "key", "mode", "confidence", "source", "method", "planning_constraint"}, _tonality_candidate)
    if status == "unavailable" and candidates:
        _fail(f"{path}.candidates", "must be empty when status is unavailable")
    return {"status": status, "candidates": candidates}


def _validate_brightness(value: Any, path: str) -> dict[str, Any]:
    item = _object(value, path, {"status", "hz", "normalized", "source", "method"})
    status = _enum(item["status"], f"{path}.status", {"available", "unavailable"})
    if status == "available":
        hz = _number(item["hz"], f"{path}.hz", minimum=0)
        normalized = _number(item["normalized"], f"{path}.normalized", minimum=0, maximum=1)
    else:
        if item["hz"] is not None or item["normalized"] is not None:
            _fail(path, "unavailable brightness requires null numeric fields")
        hz = normalized = None
    return {"status": status, "hz": hz, "normalized": normalized,
            "source": _enum(item["source"], f"{path}.source", {"signal"}),
            "method": _enum(item["method"], f"{path}.method", {"legacy_v6_brightness"})}


def _catalog_tempo(value: Any, path: str) -> dict[str, Any]:
    item = _object_with_optional(value, path, {"status", "bpm", "candidates"}, {"range_bpm", "trend", "curve_anchor_bpm"})
    out: dict[str, Any] = {
        "status": _enum(item["status"], f"{path}.status", {"fixed", "ambiguous", "variable", "none"}),
        "bpm": None if item["bpm"] is None else _number(item["bpm"], f"{path}.bpm", minimum=20, maximum=400),
    }
    candidates = _array(item["candidates"], f"{path}.candidates")
    out["candidates"] = []
    for index, candidate in enumerate(candidates):
        row = _object(candidate, f"{path}.candidates[{index}]", {"bpm", "relation"})
        out["candidates"].append({
            "bpm": _number(row["bpm"], f"{path}.candidates[{index}].bpm", minimum=20, maximum=400),
            "relation": _enum(row["relation"], f"{path}.candidates[{index}].relation", _TEMPO_RELATIONS),
        })
    if "range_bpm" in item:
        range_value = _array(item["range_bpm"], f"{path}.range_bpm", maximum=2)
        if len(range_value) != 2:
            _fail(f"{path}.range_bpm", "must be an ascending two-value array")
        low = _number(range_value[0], f"{path}.range_bpm[0]", minimum=20, maximum=400)
        high = _number(range_value[1], f"{path}.range_bpm[1]", minimum=20, maximum=400)
        if _canonical_numeric_projection(low) >= _canonical_numeric_projection(high):
            _fail(f"{path}.range_bpm", "must be ascending")
        out["range_bpm"] = [low, high]
    if "trend" in item:
        out["trend"] = _enum(item["trend"], f"{path}.trend", {"expressive", "accelerando"})
    if "curve_anchor_bpm" in item:
        out["curve_anchor_bpm"] = _number(item["curve_anchor_bpm"], f"{path}.curve_anchor_bpm", minimum=20, maximum=400)
    return out


def _catalog_meter(value: Any, path: str) -> dict[str, Any]:
    item = _object_with_optional(value, path, {"status", "value"}, {"candidates", "regions", "feel"})
    out: dict[str, Any] = {
        "status": _enum(item["status"], f"{path}.status", {"fixed", "ambiguous", "mixed", "none"}),
        "value": None if item["value"] is None else _integer(item["value"], f"{path}.value", minimum=1, maximum=32),
    }
    if "candidates" in item:
        candidates = _array(item["candidates"], f"{path}.candidates")
        out["candidates"] = []
        for index, candidate in enumerate(candidates):
            row = _object(candidate, f"{path}.candidates[{index}]", {"value", "label"})
            out["candidates"].append({
                "value": _integer(row["value"], f"{path}.candidates[{index}].value", minimum=1, maximum=32),
                "label": _string(row["label"], f"{path}.candidates[{index}].label", max_bytes=128),
            })
    if "regions" in item:
        regions = _array(item["regions"], f"{path}.regions", maximum=0)
        out["regions"] = regions
    if "feel" in item:
        out["feel"] = _enum(item["feel"], f"{path}.feel", {"shuffle"})
    return out


def _catalog_structure(value: Any, path: str, duration: int | float) -> dict[str, Any]:
    item = _object(value, path, {"status", "sections"})
    out: list[dict[str, Any]] = []
    sections = _array(item["sections"], f"{path}.sections", maximum=256)
    if not sections:
        _fail(f"{path}.sections", "must contain 1 to 256 rows")
    previous: Decimal | None = None
    for index, section in enumerate(sections):
        row = _object(section, f"{path}.sections[{index}]", {"start", "label"})
        start = _number(row["start"], f"{path}.sections[{index}].start", minimum=0, maximum=duration)
        projection = _canonical_numeric_projection(start)
        if previous is not None and _timeline_before(projection, previous):
            _fail(f"{path}.sections", "rows must be ordered by start")
        previous = projection
        out.append({"start": start, "label": _string(row["label"], f"{path}.sections[{index}].label", max_bytes=256)})
    return {
        "status": _enum(item["status"], f"{path}.status", {"continuous", "sourced", "artist_validated"}),
        "sections": out,
    }


def _catalog_citations(value: Any, path: str) -> list[dict[str, str]]:
    citations = _array(value, path, maximum=64)
    out: list[dict[str, str]] = []
    kinds = {"published_score", "peer_reviewed_analysis", "educational_score_analysis", "institutional_music_history", "published_sheet_music", "recording_metadata"}
    for index, citation in enumerate(citations):
        row = _object(citation, f"{path}[{index}]", {"kind", "url"})
        url = _string(row["url"], f"{path}[{index}].url", max_bytes=2048)
        if not url.startswith("https://"):
            _fail(f"{path}[{index}].url", "must be an absolute https:// URL")
        out.append({
            "kind": _enum(row["kind"], f"{path}[{index}].kind", kinds),
            "url": url,
        })
    return out


def _validate_catalog_facts(value: Any, path: str, duration: int | float) -> dict[str, Any]:
    item = _object(value, path, {"match", "recording_id", "title", "catalog_snapshot_hash", "tempo", "meter", "structure", "citations"})
    match = item["match"]
    if match is not None:
        match = _enum(match, f"{path}.match", {"sha256"})
    recording_id = None if item["recording_id"] is None else _string(item["recording_id"], f"{path}.recording_id", max_bytes=256)
    title = None if item["title"] is None else _string(item["title"], f"{path}.title", max_bytes=256)
    tempo = None if item["tempo"] is None else _catalog_tempo(item["tempo"], f"{path}.tempo")
    meter = None if item["meter"] is None else _catalog_meter(item["meter"], f"{path}.meter")
    structure = None if item["structure"] is None else _catalog_structure(item["structure"], f"{path}.structure", duration)
    catalog_hash = _validate_sha(item["catalog_snapshot_hash"], f"{path}.catalog_snapshot_hash")
    citations = _catalog_citations(item["citations"], f"{path}.citations")
    out = {
        "match": match,
        "recording_id": recording_id,
        "title": title,
        "catalog_snapshot_hash": catalog_hash,
        "tempo": tempo,
        "meter": meter,
        "structure": structure,
        "citations": citations,
    }
    if match is None:
        if any(value is not None for value in (recording_id, title, tempo, meter, structure)) or citations:
            _fail(path, "null match requires the exact neutral catalog projection")
    elif any(value is None for value in (recording_id, title, tempo, meter)):
        _fail(path, "sha256 match requires recording_id, title, tempo, and meter")
    projection = dict(out)
    projection.pop("catalog_snapshot_hash")
    if _sha(projection) != catalog_hash:
        _fail(f"{path}.catalog_snapshot_hash", "does not match catalog facts content")
    return out


def _validate_migration(value: Any, path: str) -> dict[str, Any] | None:
    if value is None:
        return None
    item = _object(value, path, {"source_schema", "normalized_source_sha256", "legacy_identified_recording", "loss_flags"})
    flags = _array(item["loss_flags"], f"{path}.loss_flags", maximum=len(_LOSS_FLAGS))
    normalized_flags = [_enum(flag, f"{path}.loss_flags[{index}]", _LOSS_FLAGS) for index, flag in enumerate(flags)]
    if normalized_flags != sorted(set(normalized_flags)):
        _fail(f"{path}.loss_flags", "must be sorted and unique")
    legacy = item["legacy_identified_recording"]
    if legacy is not None:
        legacy_obj = _object(legacy, f"{path}.legacy_identified_recording", {"id", "title", "match", "evidence"})
        legacy = {"id": _string(legacy_obj["id"], f"{path}.legacy_identified_recording.id", max_bytes=256),
                  "title": None if legacy_obj["title"] is None else _string(legacy_obj["title"], f"{path}.legacy_identified_recording.title", max_bytes=256),
                  "match": legacy_obj["match"], "evidence": legacy_obj["evidence"]}
        _walk_json(legacy["match"], f"{path}.legacy_identified_recording.match")
        _walk_json(legacy["evidence"], f"{path}.legacy_identified_recording.evidence")
    return {"source_schema": _enum(item["source_schema"], f"{path}.source_schema", {"flux3.music/6"}),
            "normalized_source_sha256": _validate_sha(item["normalized_source_sha256"], f"{path}.normalized_source_sha256"),
            "legacy_identified_recording": legacy, "loss_flags": normalized_flags}


def validate_evidence(value: Any) -> dict[str, Any]:
    """Validate and normalize a complete ``studio.music-evidence/1`` artifact."""
    artifact = _detached(value)
    _artifact_size(artifact, _EVIDENCE_BYTES, "Evidence")
    top = _object(artifact, "$", {"schema", "track_sha256", "evidence_id", "importer", "analyzer", "waveform", "families", "migration"})
    if top["schema"] != "studio.music-evidence/1":
        _fail("$.schema", "must equal studio.music-evidence/1")
    track = _validate_sha(top["track_sha256"], "$.track_sha256")
    evidence_id = _validate_sha(top["evidence_id"], "$.evidence_id")
    importer = _object(top["importer"], "$.importer", {"id", "version"})
    importer_out = {"id": _enum(importer["id"], "$.importer.id", {"studio.music-v6-import"}), "version": _enum(importer["version"], "$.importer.version", {"1"})}
    analyzer = _object(top["analyzer"], "$.analyzer", {"id", "version", "parameter_hash", "parameters", "runtime", "decoded_audio"})
    parameters = _object(analyzer["parameters"], "$.analyzer.parameters", {"target_sample_rate_hz", "main_hop_samples", "tempo_refinement_hop_samples", "waveform_requested_bucket_rate_hz", "minimum_probe_seconds"})
    expected_parameters = {"target_sample_rate_hz": 22050, "main_hop_samples": 512, "tempo_refinement_hop_samples": 128, "waveform_requested_bucket_rate_hz": 200, "minimum_probe_seconds": 3.0}
    if canonical_json_bytes(parameters) != canonical_json_bytes(expected_parameters):
        _fail("$.analyzer.parameters", "must equal the frozen v1 parameter object")
    parameter_hash = _validate_sha(analyzer["parameter_hash"], "$.analyzer.parameter_hash")
    if parameter_hash != _sha(parameters):
        _fail("$.analyzer.parameter_hash", "does not match parameters content")
    runtime = _object(analyzer["runtime"], "$.analyzer.runtime", {"status"})
    runtime_out = {"status": _enum(runtime["status"], "$.analyzer.runtime.status", {"unrecorded_legacy"})}
    decoded = _object(analyzer["decoded_audio"], "$.analyzer.decoded_audio", {"duration_seconds", "sample_rate_hz", "channels", "stream_selector"})
    duration = _number(decoded["duration_seconds"], "$.analyzer.decoded_audio.duration_seconds", minimum=0)
    if _canonical_numeric_projection(duration) <= Decimal(0):
        _fail("$.analyzer.decoded_audio.duration_seconds", "must be > 0")
    sample_rate = _integer(decoded["sample_rate_hz"], "$.analyzer.decoded_audio.sample_rate_hz", minimum=8000, maximum=384000)
    channels = _integer(decoded["channels"], "$.analyzer.decoded_audio.channels", minimum=1, maximum=64)
    decoded_out = {"duration_seconds": duration, "sample_rate_hz": sample_rate, "channels": channels,
                   "stream_selector": _enum(decoded["stream_selector"], "$.analyzer.decoded_audio.stream_selector", {"legacy_default_audio_stream"})}
    analyzer_out = {"id": _string(analyzer["id"], "$.analyzer.id", max_bytes=64, ascii_only=True), "version": _enum(analyzer["version"], "$.analyzer.version", {"unrecorded"}),
                    "parameter_hash": parameter_hash, "parameters": copy.deepcopy(expected_parameters), "runtime": runtime_out, "decoded_audio": decoded_out}
    if _ANALYZER_ID_RE.fullmatch(analyzer_out["id"]) is None:
        _fail("$.analyzer.id", "has an invalid analyzer ID")
    wave = _object(top["waveform"], "$.waveform", {"waveform_id", "requested_bucket_rate_hz", "estimated_effective_bucket_rate_hz", "estimated_bucket_span_seconds", "bucket_count", "encoding"})
    wave_out = {"waveform_id": _validate_sha(wave["waveform_id"], "$.waveform.waveform_id"),
                "requested_bucket_rate_hz": _integer(wave["requested_bucket_rate_hz"], "$.waveform.requested_bucket_rate_hz", minimum=1),
                "estimated_effective_bucket_rate_hz": _number(wave["estimated_effective_bucket_rate_hz"], "$.waveform.estimated_effective_bucket_rate_hz", minimum=0),
                "estimated_bucket_span_seconds": _number(wave["estimated_bucket_span_seconds"], "$.waveform.estimated_bucket_span_seconds", minimum=0),
                "bucket_count": _integer(wave["bucket_count"], "$.waveform.bucket_count", minimum=1, maximum=_MAX_LIST),
                "encoding": _enum(wave["encoding"], "$.waveform.encoding", {"minmax-f32-json/1"})}
    if wave_out["requested_bucket_rate_hz"] != 200:
        _fail("$.waveform.requested_bucket_rate_hz", "must equal 200")
    if not _waveform_equations_match(
            duration,
            wave_out["bucket_count"],
            wave_out["estimated_effective_bucket_rate_hz"],
            wave_out["estimated_bucket_span_seconds"]):
        _fail("$.waveform", "estimates do not match bucket_count/duration equations")
    families = _object(top["families"], "$.families", {"tempo", "beats", "meter", "downbeats", "sections", "accents", "energy", "tonality", "brightness", "catalog_facts", "lyrics_alignment_candidates"})
    families_out = {
        "tempo": _validate_tempo(families["tempo"], "$.families.tempo", duration),
        "beats": _validate_beats(families["beats"], "$.families.beats", duration),
        "meter": _validate_meter(families["meter"], "$.families.meter"),
        "downbeats": _validate_downbeats(families["downbeats"], "$.families.downbeats", duration),
        "sections": _validate_sections(families["sections"], "$.families.sections", duration),
        "accents": _validate_accents(families["accents"], "$.families.accents", duration),
        "energy": _validate_energy(families["energy"], "$.families.energy", duration, sample_rate),
        "tonality": _validate_tonality(families["tonality"], "$.families.tonality"),
        "brightness": _validate_brightness(families["brightness"], "$.families.brightness"),
        "catalog_facts": _validate_catalog_facts(families["catalog_facts"], "$.families.catalog_facts", duration),
        "lyrics_alignment_candidates": _array(families["lyrics_alignment_candidates"], "$.families.lyrics_alignment_candidates", maximum=0),
    }
    migration_out = _validate_migration(top["migration"], "$.migration")
    result = {"schema": "studio.music-evidence/1", "track_sha256": track, "evidence_id": evidence_id,
              "importer": importer_out, "analyzer": analyzer_out, "waveform": wave_out, "families": families_out, "migration": migration_out}
    _artifact_size(result, _EVIDENCE_BYTES, "Evidence")
    expected_id = compute_evidence_id(result)
    if evidence_id != expected_id:
        _fail("$.evidence_id", "does not match complete artifact content")
    return result


def _validate_scope(value: Any, path: str, *, allow_default: bool = False) -> dict[str, str]:
    if not isinstance(value, dict):
        _fail(path, "expected scope object")
    value = _detached(value, path)
    if allow_default and set(value) == {"kind"} and value.get("kind") == "default":
        return {"kind": "default"}
    kind = value.get("kind")
    if kind == "scratch":
        _object(value, path, {"kind"})
        return {"kind": "scratch"}
    if kind == "project":
        _object(value, path, {"kind", "project_id"})
        project_id = _string(value["project_id"], f"{path}.project_id", max_bytes=36, ascii_only=True)
        if _PROJECT_RE.fullmatch(project_id) is None:
            _fail(f"{path}.project_id", "must match prj_<32 lowercase hex>")
        return {"kind": "project", "project_id": project_id}
    _fail(f"{path}.kind", "must be project or scratch")


def _validate_tempo_body(value: Any, path: str) -> dict[str, Any]:
    item = _object(value, path, {"mode", "selected_candidate_id", "bpm", "beat_unit", "grid_source", "confirmed_by_artist"})
    mode = _enum(item["mode"], f"{path}.mode", {"inherit", "fixed", "variable", "no_pulse"})
    selected = None if item["selected_candidate_id"] is None else _validate_id(
        item["selected_candidate_id"], f"{path}.selected_candidate_id", prefix="tempo."
    )
    bpm = None if item["bpm"] is None else _number(item["bpm"], f"{path}.bpm", minimum=20, maximum=400)
    if mode == "fixed":
        if bpm is None or selected is not None:
            _fail(path, "fixed tempo requires bpm and no selected candidate")
    elif bpm is not None:
        _fail(f"{path}.bpm", f"must be null for mode {mode}")
    if mode == "no_pulse" and selected is not None:
        _fail(f"{path}.selected_candidate_id", "must be null for no_pulse")
    grid_source = _enum(item["grid_source"], f"{path}.grid_source", {"inherit", "detected_beats", "regular_grid"})
    if mode == "no_pulse" and grid_source != "inherit":
        _fail(f"{path}.grid_source", "no_pulse requires inherit")
    return {"mode": mode, "selected_candidate_id": selected, "bpm": bpm,
            "beat_unit": _enum(item["beat_unit"], f"{path}.beat_unit", {"whole", "half", "quarter", "eighth", "dotted_quarter"}),
            "grid_source": grid_source, "confirmed_by_artist": _boolean(item["confirmed_by_artist"], f"{path}.confirmed_by_artist")}


def _validate_meter_body(value: Any, path: str, duration: int | float | None) -> dict[str, Any]:
    item = _object(value, path, {"mode", "numerator", "denominator", "regions", "confirmed_by_artist"})
    mode = _enum(item["mode"], f"{path}.mode", {"inherit", "fixed", "mixed", "none"})
    numerator = None if item["numerator"] is None else _integer(item["numerator"], f"{path}.numerator", minimum=1, maximum=32)
    denominator = None if item["denominator"] is None else _integer(item["denominator"], f"{path}.denominator", minimum=1, maximum=16)
    if denominator is not None and denominator not in {1, 2, 4, 8, 16}:
        _fail(f"{path}.denominator", "must be one of 1, 2, 4, 8, 16")
    regions = _array(item["regions"], f"{path}.regions", maximum=256)
    if mode == "fixed":
        if numerator is None or denominator is None or regions:
            _fail(path, "fixed meter requires numerator, denominator, and no regions")
    elif mode in {"inherit", "none"}:
        if numerator is not None or denominator is not None or regions:
            _fail(path, f"{mode} meter requires null numerator/denominator and empty regions")
    else:  # mixed
        if numerator is not None or denominator is not None or not regions:
            _fail(path, "mixed meter requires null top-level signature and regions")
    output_regions: list[dict[str, Any]] = []
    previous_end: Decimal | None = None
    seen_region_ids: set[str] = set()
    for index, region in enumerate(regions):
        row = _object(region, f"{path}.regions[{index}]", {"region_id", "start_seconds", "end_seconds", "numerator", "denominator", "confirmed_by_artist"})
        rid = _validate_id(row["region_id"], f"{path}.regions[{index}].region_id")
        if rid in seen_region_ids:
            _fail(f"{path}.regions[{index}].region_id", "must be unique")
        seen_region_ids.add(rid)
        start, end = _time_range(row["start_seconds"], row["end_seconds"], f"{path}.regions[{index}]", duration)
        rn = _integer(row["numerator"], f"{path}.regions[{index}].numerator", minimum=1, maximum=32)
        rd = _integer(row["denominator"], f"{path}.regions[{index}].denominator", minimum=1, maximum=16)
        if rd not in {1, 2, 4, 8, 16}:
            _fail(f"{path}.regions[{index}].denominator", "must be one of 1, 2, 4, 8, 16")
        start_projection = _canonical_numeric_projection(start)
        end_projection = _canonical_numeric_projection(end)
        if previous_end is not None and _timeline_before(start_projection, previous_end):
            _fail(f"{path}.regions", "regions must be ordered and non-overlapping")
        previous_end = end_projection
        output_regions.append({"region_id": rid, "start_seconds": start, "end_seconds": end, "numerator": rn, "denominator": rd,
                               "confirmed_by_artist": _boolean(row["confirmed_by_artist"], f"{path}.regions[{index}].confirmed_by_artist")})
    return {"mode": mode, "numerator": numerator, "denominator": denominator, "regions": output_regions,
            "confirmed_by_artist": _boolean(item["confirmed_by_artist"], f"{path}.confirmed_by_artist")}


def _validate_downbeat_body(value: Any, path: str, duration: int | float | None) -> dict[str, Any]:
    item = _object(value, path, {"mode", "anchor_seconds", "confirmed_by_artist"})
    mode = _enum(item["mode"], f"{path}.mode", {"inherit", "anchor", "none"})
    anchor = None if item["anchor_seconds"] is None else _time(item["anchor_seconds"], f"{path}.anchor_seconds", duration)
    if mode == "anchor" and anchor is None:
        _fail(path, "anchor mode requires anchor_seconds")
    if mode != "anchor" and anchor is not None:
        _fail(f"{path}.anchor_seconds", f"must be null for mode {mode}")
    return {"mode": mode, "anchor_seconds": anchor, "confirmed_by_artist": _boolean(item["confirmed_by_artist"], f"{path}.confirmed_by_artist")}


def _validate_section_body(value: Any, path: str, duration: int | float | None) -> dict[str, Any]:
    item = _object(value, path, {"section_id", "start_seconds", "end_seconds", "label", "status"})
    start, end = _time_range(item["start_seconds"], item["end_seconds"], path, duration)
    return {"section_id": _validate_id(item["section_id"], f"{path}.section_id"), "start_seconds": start, "end_seconds": end,
            "label": _string(item["label"], f"{path}.label", max_bytes=256),
            "status": _enum(item["status"], f"{path}.status", {"artist_draft", "artist_confirmed"})}


def _validate_lyrics(value: Any, path: str) -> dict[str, Any] | None:
    if value is None:
        return None
    item = _object(value, path, {"original_text", "language", "source", "text_sha256", "lines", "words"})
    original = _string(item["original_text"], f"{path}.original_text", max_bytes=_MAX_STRING_BYTES, allow_controls=True, nonempty=False)
    language = None if item["language"] is None else _string(item["language"], f"{path}.language", max_bytes=64, ascii_only=True)
    if language is not None and _LANGUAGE_RE.fullmatch(language) is None:
        _fail(f"{path}.language", "is not a bounded BCP-47-style language tag")
    text_hash = _validate_sha(item["text_sha256"], f"{path}.text_sha256")
    if text_hash != hashlib.sha256(original.encode("utf-8")).hexdigest():
        _fail(f"{path}.text_sha256", "does not match original_text UTF-8 bytes")
    lines = _array(item["lines"], f"{path}.lines", maximum=0)
    words = _array(item["words"], f"{path}.words", maximum=0)
    return {"original_text": original, "language": language, "source": _enum(item["source"], f"{path}.source", {"artist", "transcription_draft"}),
            "text_sha256": text_hash, "lines": lines, "words": words}


def validate_interpretation_body(value: Any, *, duration_seconds: int | float | None = None) -> dict[str, Any]:
    """Validate the strict artist-authored Interpretation body."""
    body = _detached(value)
    top = _object(body, "$", {"tempo", "meter", "downbeat", "sections", "lyrics"})
    duration = None if duration_seconds is None else _number(duration_seconds, "duration_seconds", minimum=0)
    sections = _array(top["sections"], "$.sections", maximum=256)
    output_sections: list[dict[str, Any]] = []
    previous_end: Decimal | None = None
    seen_ids: set[str] = set()
    for index, section in enumerate(sections):
        normalized = _validate_section_body(section, f"$.sections[{index}]", duration)
        if normalized["section_id"] in seen_ids:
            _fail(f"$.sections[{index}].section_id", "must be unique")
        start_projection = _canonical_numeric_projection(normalized["start_seconds"])
        end_projection = _canonical_numeric_projection(normalized["end_seconds"])
        if previous_end is not None and _timeline_before(start_projection, previous_end):
            _fail("$.sections", "sections must be ordered and non-overlapping")
        seen_ids.add(normalized["section_id"])
        previous_end = end_projection
        output_sections.append(normalized)
    result = {"tempo": _validate_tempo_body(top["tempo"], "$.tempo"),
              "meter": _validate_meter_body(top["meter"], "$.meter", duration),
              "downbeat": _validate_downbeat_body(top["downbeat"], "$.downbeat", duration),
              "sections": output_sections, "lyrics": _validate_lyrics(top["lyrics"], "$.lyrics")}
    return result


def validate_interpretation(value: Any, *, duration_seconds: int | float | None = None) -> dict[str, Any]:
    """Validate a persisted ``studio.music-interpretation/1`` artifact."""
    artifact = _detached(value)
    _artifact_size(artifact, _INTERPRETATION_BYTES, "Interpretation")
    top = _object(artifact, "$", {"schema", "track_sha256", "scope", "revision", "updated_at", "tempo", "meter", "downbeat", "sections", "lyrics"})
    if top["schema"] != "studio.music-interpretation/1":
        _fail("$.schema", "must equal studio.music-interpretation/1")
    scope = _validate_scope(top["scope"], "$.scope")
    revision = _integer(top["revision"], "$.revision", minimum=1)
    updated_at = _validate_timestamp(top["updated_at"], "$.updated_at")
    body = validate_interpretation_body({key: top[key] for key in ("tempo", "meter", "downbeat", "sections", "lyrics")}, duration_seconds=duration_seconds)
    result = {"schema": "studio.music-interpretation/1", "track_sha256": _validate_sha(top["track_sha256"], "$.track_sha256"),
              "scope": scope, "revision": revision, "updated_at": updated_at, **body}
    _artifact_size(result, _INTERPRETATION_BYTES, "Interpretation")
    return result


def default_interpretation_body() -> dict[str, Any]:
    """Return a detached all-inherit Phase-2 Interpretation body."""
    return {"tempo": {"mode": "inherit", "selected_candidate_id": None, "bpm": None, "beat_unit": "quarter", "grid_source": "inherit", "confirmed_by_artist": False},
            "meter": {"mode": "inherit", "numerator": None, "denominator": None, "regions": [], "confirmed_by_artist": False},
            "downbeat": {"mode": "inherit", "anchor_seconds": None, "confirmed_by_artist": False},
            "sections": [], "lyrics": None}


def build_interpretation(*, track_sha256: Any, scope: Any, revision: Any, updated_at: Any, value: Any) -> dict[str, Any]:
    """Build and validate the backend-owned Interpretation envelope."""
    body = validate_interpretation_body(value)
    result = {"schema": "studio.music-interpretation/1", "track_sha256": _validate_sha(track_sha256, "track_sha256"),
              "scope": _validate_scope(scope, "scope"), "revision": _integer(revision, "revision", minimum=1),
              "updated_at": _validate_timestamp(updated_at, "updated_at"), **body}
    _artifact_size(result, _INTERPRETATION_BYTES, "Interpretation")
    return result


def _effective_validation_bpm(body: dict[str, Any], evidence: dict[str, Any]) -> int | float | None:
    tempo = body["tempo"]
    if tempo["mode"] == "fixed":
        return tempo["bpm"]

    candidates = evidence["families"]["tempo"]["candidates"]
    selected_id = tempo["selected_candidate_id"]
    if selected_id is not None:
        selected = next((row for row in candidates if row["candidate_id"] == selected_id), None)
        return None if selected is None else selected["bpm"]

    catalog = evidence["families"]["catalog_facts"]
    catalog_tempo = catalog.get("tempo")
    if catalog.get("match") == "sha256" and catalog.get("citations") and isinstance(catalog_tempo, dict):
        status = catalog_tempo.get("status")
        bpm = catalog_tempo.get("bpm")
        if status == "fixed":
            if isinstance(bpm, (int, float)) and not isinstance(bpm, bool):
                projection = _canonical_numeric_projection(bpm)
                if Decimal("20") <= projection <= Decimal("400"):
                    return bpm
            return None
        if status in {"ambiguous", "variable", "none"}:
            return None

    if evidence["families"]["tempo"]["status"] == "ambiguous" or not candidates:
        return None

    def sort_key(row: dict[str, Any]) -> tuple[Decimal, str]:
        confidence = Decimal("-1") if row["confidence"] is None else _canonical_numeric_projection(row["confidence"])
        return -confidence, row["candidate_id"]

    selected_for_grid = [row for row in candidates if row["selected_for_grid"]]
    return sorted(selected_for_grid or candidates, key=sort_key)[0]["bpm"]


def _unique_timing_match(events: list[dict[str, Any]], phase_time: int | float) -> bool:
    context = Context(prec=50, rounding=ROUND_HALF_EVEN)
    context.traps[Inexact] = False
    context.traps[Rounded] = False
    with localcontext(context):
        phase = Decimal(canonical_json_bytes(phase_time).decode("ascii"))
        matches = [
            event for event in events
            if abs(Decimal(canonical_json_bytes(event["time_seconds"]).decode("ascii")) - phase) <= _TIME_TOLERANCE
        ]
    return len(matches) == 1


def _regular_grid_match(phase_time: int | float, target_time: int | float, bpm: int | float) -> bool:
    decimal_phase = _canonical_decimal(phase_time)
    decimal_target = _canonical_decimal(target_time)
    decimal_bpm = _canonical_decimal(bpm)
    context = _fresh_decimal_context(decimal_phase, decimal_target, decimal_bpm)
    context.traps[Inexact] = False
    context.traps[Rounded] = False
    with localcontext(context):
        interval = Decimal(60) / decimal_bpm
        ordinal = ((decimal_target - decimal_phase) / interval).to_integral_value(rounding=ROUND_HALF_EVEN)
        emitted = (decimal_phase + ordinal * interval).quantize(Decimal("0.000000001"), rounding=ROUND_HALF_EVEN)
        return _waveform_value_matches(target_time, emitted)


def _validate_integral_pulse_count(numerator: int, denominator: int, beat_unit: str, path: str) -> None:
    pulse_count = Fraction(numerator * 4, denominator) / _UNIT_FACTORS[beat_unit]
    if pulse_count.denominator != 1 or not 1 <= pulse_count.numerator <= 256:
        _fail(path, "meter signature does not produce an integral pulse count")


def _effective_validation_anchor(body: dict[str, Any], evidence: dict[str, Any]) -> int | float | None:
    authored = body["downbeat"]
    if authored["mode"] == "anchor":
        return authored["anchor_seconds"]
    if authored["mode"] == "inherit":
        family = evidence["families"]["downbeats"]
        if family["status"] != "unavailable" and family["events"]:
            return family["events"][0]["time_seconds"]
    return None


def validate_interpretation_against_evidence(body: Any, evidence: Any, track_sha256: Any) -> dict[str, Any]:
    """Validate an authored body against one exact, immutable Evidence artifact."""
    track = _validate_sha(track_sha256, "track_sha256")
    validated_evidence = validate_evidence(evidence)
    if validated_evidence["track_sha256"] != track:
        _fail("track_sha256", "does not match Evidence track_sha256")
    duration = validated_evidence["analyzer"]["decoded_audio"]["duration_seconds"]
    validated_body = validate_interpretation_body(body, duration_seconds=duration)
    tempo = validated_body["tempo"]
    selected_candidate_id = tempo["selected_candidate_id"]
    if selected_candidate_id is not None:
        candidates = validated_evidence["families"]["tempo"]["candidates"]
        if not any(row["candidate_id"] == selected_candidate_id for row in candidates):
            raise MusicCandidateMissingError(
                "$.tempo.selected_candidate_id: is absent from Evidence.families.tempo.candidates"
            )
    grid_source = tempo["grid_source"]
    if tempo["mode"] == "fixed" and grid_source == "inherit":
        _fail("$.tempo.grid_source", "fixed tempo requires explicit detected_beats or regular_grid")
    if tempo["mode"] == "variable" and grid_source == "regular_grid":
        _fail("$.tempo.grid_source", "variable tempo cannot request regular_grid")
    if grid_source == "detected_beats":
        beats = validated_evidence["families"]["beats"]
        if beats["status"] == "unavailable" or not beats["events"]:
            _fail("$.tempo.grid_source", "detected_beats requires a usable Evidence beats sequence")
    if grid_source == "regular_grid":
        effective_bpm = _effective_validation_bpm(validated_body, validated_evidence)
        if effective_bpm is None:
            _fail("$.tempo.grid_source", "regular_grid requires effective fixed BPM")
        downbeat = validated_body["downbeat"]
        if downbeat["mode"] == "anchor":
            has_anchor = True
        elif downbeat["mode"] == "inherit":
            family = validated_evidence["families"]["downbeats"]
            has_anchor = family["status"] != "unavailable" and bool(family["events"])
        else:
            has_anchor = False
        if not has_anchor:
            _fail("$.downbeat", "regular_grid requires a valid Bar-1 anchor")
        if validated_body["meter"]["mode"] == "mixed":
            anchor = _effective_validation_anchor(validated_body, validated_evidence)
            if anchor is None:
                _fail("$.downbeat", "mixed meter requires a global Bar-1 anchor")
            for index, region in enumerate(validated_body["meter"]["regions"]):
                try:
                    matches_grid = _regular_grid_match(anchor, region["start_seconds"], effective_bpm)
                except DecimalException:
                    _fail(f"$.meter.regions[{index}].start_seconds", "regular-grid arithmetic is invalid")
                if not matches_grid:
                    _fail(f"$.meter.regions[{index}].start_seconds", "must lie on the global regular grid")
    effective_detected = tempo["mode"] != "no_pulse" and grid_source == "detected_beats"
    if tempo["mode"] != "no_pulse" and grid_source == "inherit" and validated_evidence["families"]["beats"]["status"] != "unavailable" and validated_evidence["families"]["beats"]["events"]:
        effective_detected = True
    if effective_detected:
        beats = validated_evidence["families"]["beats"]["events"]
        anchor = _effective_validation_anchor(validated_body, validated_evidence)
        meter_mode = validated_body["meter"]["mode"]
        if meter_mode in {"fixed", "mixed"} and anchor is None:
            _fail("$.downbeat", "meter requires a valid Bar-1 anchor")
        if anchor is not None and not _unique_timing_match(beats, anchor):
            _fail("$.downbeat.anchor_seconds", "must match exactly one effective detected beat")
        if meter_mode == "mixed":
            for index, region in enumerate(validated_body["meter"]["regions"]):
                if not _unique_timing_match(beats, region["start_seconds"]):
                    _fail(f"$.meter.regions[{index}].start_seconds", "must match exactly one effective detected beat")
    meter = validated_body["meter"]
    beat_unit = tempo["beat_unit"]
    if tempo["mode"] != "no_pulse" and meter["mode"] == "fixed":
        _validate_integral_pulse_count(meter["numerator"], meter["denominator"], beat_unit, "$.meter")
    elif tempo["mode"] != "no_pulse" and meter["mode"] == "mixed":
        for index, region in enumerate(meter["regions"]):
            _validate_integral_pulse_count(region["numerator"], region["denominator"], beat_unit, f"$.meter.regions[{index}]")
    return validated_body


def _source_refs(value: Any, path: str) -> list[str]:
    refs = _array(value, path, maximum=256)
    normalized = [_string(ref, f"{path}[{index}]", max_bytes=128, ascii_only=True) for index, ref in enumerate(refs)]
    if any(_SOURCE_REF_RE.fullmatch(ref) is None for ref in normalized):
        _fail(path, "contains a source reference outside the frozen grammar")
    if normalized != sorted(set(normalized)):
        _fail(path, "must be sorted and unique")
    return normalized


def _map_event_rows(value: Any, path: str, kind: str, duration: int | float, *, accents: bool = False) -> list[dict[str, Any]]:
    rows = _array(value, path)
    keys = {"event_id", "time_seconds", "source_refs"}
    if accents:
        keys |= {"strength", "salience"}
    out: list[dict[str, Any]] = []
    previous: Decimal | None = None
    for index, row in enumerate(rows):
        item = _object(row, f"{path}[{index}]", keys)
        event_id = _string(item["event_id"], f"{path}[{index}].event_id", max_bytes=32, ascii_only=True)
        expected_re = re.compile(rf"{kind}\.[0-9]{{6}}\Z")
        if expected_re.fullmatch(event_id) is None:
            _fail(f"{path}[{index}].event_id", "has an invalid event ID")
        if event_id != f"{kind}.{index + 1:06d}":
            _fail(f"{path}[{index}].event_id", "must be the chronological ordinal for this row")
        time_value = _time(item["time_seconds"], f"{path}[{index}].time_seconds", duration)
        projection = _canonical_numeric_projection(time_value)
        if previous is not None and _timeline_before(projection, previous):
            _fail(path, "rows must be ordered by time_seconds")
        previous = projection
        normalized = {"event_id": event_id, "time_seconds": time_value, "source_refs": _source_refs(item["source_refs"], f"{path}[{index}].source_refs")}
        if accents:
            normalized["strength"] = _number(item["strength"], f"{path}[{index}].strength", minimum=0, maximum=1)
            normalized["salience"] = _number(item["salience"], f"{path}[{index}].salience", minimum=0, maximum=1)
        out.append(normalized)
    return out


def _map_section_rows(value: Any, path: str, duration: int | float) -> list[dict[str, Any]]:
    rows = _array(value, path, maximum=256)
    out: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    previous_end: Decimal | None = None
    for index, row in enumerate(rows):
        item = _object(row, f"{path}[{index}]", {"section_id", "start_seconds", "end_seconds", "label", "label_kind", "confidence", "source_refs"})
        start, end = _time_range(item["start_seconds"], item["end_seconds"], f"{path}[{index}]", duration)
        normalized = {"section_id": _validate_id(item["section_id"], f"{path}[{index}].section_id", prefix="section."),
                      "start_seconds": start, "end_seconds": end, "label": _string(item["label"], f"{path}[{index}].label", max_bytes=256),
                      "label_kind": _enum(item["label_kind"], f"{path}[{index}].label_kind", {"neutral", "semantic"}),
                      "confidence": _confidence(item["confidence"], f"{path}[{index}].confidence"),
                      "source_refs": _source_refs(item["source_refs"], f"{path}[{index}].source_refs")}
        identity_row = {key: normalized[key] for key in normalized if key != "section_id"}
        expected = f"section.{_sha(identity_row)}"
        if normalized["section_id"] != expected:
            _fail(f"{path}[{index}].section_id", "does not match effective section content")
        body_bytes = canonical_json_bytes(identity_row)
        if body_bytes in seen:
            continue
        seen.add(body_bytes)
        start_projection = _canonical_numeric_projection(start)
        end_projection = _canonical_numeric_projection(end)
        if previous_end is not None and _timeline_before(start_projection, previous_end):
            _fail(path, "sections must be ordered and non-overlapping")
        previous_end = end_projection
        out.append(normalized)
    return out


def _map_warnings(value: Any, path: str) -> list[dict[str, Any]]:
    rows = _array(value, path, maximum=len(_WARNING_ORDER))
    seen: set[bytes] = set()
    out: list[dict[str, Any]] = []
    exact_refs = {
        "selected_candidate_missing": ["interpretation:tempo.selected_candidate_id"],
        "regular_grid_anchor_missing": ["interpretation:tempo.grid_source"],
        "bar_math_withheld": ["map:meter", "map:tempo"],
        "legacy_evidence_loss": ["evidence:migration.loss_flags"],
        "catalog_snapshot_absent": ["catalog:catalog_snapshot_hash"],
    }
    for index, row in enumerate(rows):
        item = _object(row, f"{path}[{index}]", {"code", "source_refs"})
        code = _enum(item["code"], f"{path}[{index}].code", set(_WARNING_ORDER))
        refs = _source_refs(item["source_refs"], f"{path}[{index}].source_refs")
        if refs != exact_refs[code]:
            _fail(f"{path}[{index}].source_refs", f"do not match warning code {code}")
        normalized = {"code": code, "source_refs": refs}
        identity = canonical_json_bytes(normalized)
        if identity not in seen:
            seen.add(identity)
            out.append(normalized)
    out.sort(key=lambda row: (_WARNING_ORDER[row["code"]], canonical_json_bytes(row["source_refs"])))
    return out


def _validate_map_tempo(value: Any, path: str) -> dict[str, Any]:
    item = _object(value, path, {"status", "bpm", "beat_unit", "grid_source", "confidence", "source_refs"})
    status = _enum(item["status"], f"{path}.status", {"fixed", "variable", "ambiguous", "no_pulse", "unavailable"})
    bpm = None if item["bpm"] is None else _number(item["bpm"], f"{path}.bpm", minimum=20, maximum=400)
    if status == "fixed" and bpm is None:
        _fail(path, "fixed tempo requires bpm")
    if status != "fixed" and bpm is not None:
        _fail(f"{path}.bpm", "only fixed tempo may carry bpm")
    return {"status": status, "bpm": bpm, "beat_unit": _enum(item["beat_unit"], f"{path}.beat_unit", {"whole", "half", "quarter", "eighth", "dotted_quarter"}),
            "grid_source": _enum(item["grid_source"], f"{path}.grid_source", {"inherit", "detected_beats", "regular_grid"}),
            "confidence": _confidence(item["confidence"], f"{path}.confidence"), "source_refs": _source_refs(item["source_refs"], f"{path}.source_refs")}


def _validate_map_meter(value: Any, path: str, duration: float) -> dict[str, Any]:
    item = _object(value, path, {"status", "beats_per_bar", "numerator", "denominator", "regions", "confidence", "source_refs"})
    status = _enum(item["status"], f"{path}.status", {"fixed", "mixed", "beats_per_bar_only", "ambiguous", "none", "unavailable"})
    bpb = None if item["beats_per_bar"] is None else _integer(item["beats_per_bar"], f"{path}.beats_per_bar", minimum=1, maximum=32)
    numerator = None if item["numerator"] is None else _integer(item["numerator"], f"{path}.numerator", minimum=1, maximum=32)
    denominator = None if item["denominator"] is None else _integer(item["denominator"], f"{path}.denominator", minimum=1, maximum=16)
    if denominator is not None and denominator not in {1, 2, 4, 8, 16}:
        _fail(f"{path}.denominator", "must be one of 1, 2, 4, 8, 16")
    regions = _array(item["regions"], f"{path}.regions", maximum=256)
    source_refs = _source_refs(item["source_refs"], f"{path}.source_refs")
    if status == "fixed" and (bpb is None or numerator is None or bpb != numerator or denominator is None or regions):
        _fail(path, "fixed meter requires beats_per_bar equal to numerator, a signature and no regions")
    if status == "mixed" and (bpb is not None or numerator is not None or denominator is not None):
        _fail(path, "mixed meter requires regions and no global signature")
    if status == "mixed" and not regions and "catalog:meter" not in source_refs:
        _fail(path, "mixed meter requires regions and no global signature")
    if status == "beats_per_bar_only" and (bpb is None or numerator is not None or denominator is not None or regions):
        _fail(path, "beats_per_bar_only requires only beats_per_bar")
    if status in {"ambiguous", "none", "unavailable"} and (bpb is not None or numerator is not None or denominator is not None or regions):
        _fail(path, f"{status} meter cannot carry meter claims")
    output_regions: list[dict[str, Any]] = []
    seen_region_ids: set[str] = set()
    previous_start: Decimal | None = None
    previous_end: Decimal | None = None
    for index, region in enumerate(regions):
        row = _object(region, f"{path}.regions[{index}]", {"region_id", "start_seconds", "end_seconds", "numerator", "denominator", "source_refs"})
        region_path = f"{path}.regions[{index}]"
        region_id = _validate_id(row["region_id"], f"{region_path}.region_id")
        if region_id in seen_region_ids:
            _fail(f"{region_path}.region_id", "must be unique")
        start, end = _time_range(row["start_seconds"], row["end_seconds"], region_path, duration)
        start_projection = _canonical_numeric_projection(start)
        end_projection = _canonical_numeric_projection(end)
        if previous_start is not None and start_projection < previous_start:
            _fail(f"{path}.regions", "regions must be ordered and non-overlapping")
        if previous_end is not None and start_projection < previous_end:
            _fail(f"{path}.regions", "regions must be ordered and non-overlapping")
        seen_region_ids.add(region_id)
        previous_start = start_projection
        previous_end = end_projection
        rn = _integer(row["numerator"], f"{region_path}.numerator", minimum=1, maximum=32)
        rd = _integer(row["denominator"], f"{region_path}.denominator", minimum=1, maximum=16)
        if rd not in {1, 2, 4, 8, 16}:
            _fail(f"{region_path}.denominator", "invalid denominator")
        output_regions.append({"region_id": region_id, "start_seconds": start, "end_seconds": end,
                               "numerator": rn, "denominator": rd, "source_refs": _source_refs(row["source_refs"], f"{region_path}.source_refs")})

    catalog_refs = ["catalog:catalog_snapshot_hash", "catalog:meter"]
    if status == "fixed":
        expected_source_refs = [["interpretation:meter.denominator", "interpretation:meter.mode", "interpretation:meter.numerator"]]
    elif status == "mixed":
        if output_regions:
            expected_source_refs = [["interpretation:meter.mode", *sorted(
                f"interpretation:meter.region.{region['region_id']}" for region in output_regions
            )]]
            for index, region in enumerate(output_regions):
                region_ref = [f"interpretation:meter.region.{region['region_id']}"]
                if region["source_refs"] != region_ref:
                    _fail(f"{path}.regions[{index}].source_refs", "do not match authored meter-region provenance")
        else:
            expected_source_refs = [catalog_refs]
    elif status == "beats_per_bar_only":
        expected_source_refs = [catalog_refs]
        if len(source_refs) == 1 and source_refs[0].startswith("evidence:"):
            candidate_id = source_refs[0][len("evidence:"):]
            if candidate_id.startswith("meter.") and _ID_RE.fullmatch(candidate_id) is not None:
                expected_source_refs.append(source_refs)
    elif status == "ambiguous":
        expected_source_refs = [[], catalog_refs]
    elif status == "none":
        expected_source_refs = [["interpretation:meter.mode"], catalog_refs]
    else:
        expected_source_refs = [[]]
    if source_refs not in expected_source_refs:
        _fail(f"{path}.source_refs", "do not match the meter claim provenance")

    return {"status": status, "beats_per_bar": bpb, "numerator": numerator, "denominator": denominator,
            "regions": output_regions, "confidence": _confidence(item["confidence"], f"{path}.confidence"), "source_refs": source_refs}


def _validate_map_energy(value: Any, path: str, duration: int | float) -> dict[str, Any]:
    item = _object(value, path, {"requested_sample_rate_hz", "hop_samples", "estimated_effective_sample_rate_hz", "samples", "source_refs"})
    samples = _array(item["samples"], f"{path}.samples")
    output: list[dict[str, Any]] = []
    previous: Decimal | None = None
    for index, sample in enumerate(samples):
        row = _object(sample, f"{path}.samples[{index}]", {"time_seconds", "value"})
        time_value = _time(row["time_seconds"], f"{path}.samples[{index}].time_seconds", duration)
        projection = _canonical_numeric_projection(time_value)
        if previous is not None and _timeline_before(projection, previous):
            _fail(f"{path}.samples", "rows must be ordered by time_seconds")
        previous = projection
        output.append({"time_seconds": time_value, "value": _number(row["value"], f"{path}.samples[{index}].value", minimum=0)})
    requested = _integer(item["requested_sample_rate_hz"], f"{path}.requested_sample_rate_hz", minimum=1)
    if requested != 20:
        _fail(f"{path}.requested_sample_rate_hz", "must equal the legacy 20 Hz request")
    return {"requested_sample_rate_hz": requested,
            "hop_samples": _integer(item["hop_samples"], f"{path}.hop_samples", minimum=1),
            "estimated_effective_sample_rate_hz": _number(item["estimated_effective_sample_rate_hz"], f"{path}.estimated_effective_sample_rate_hz", minimum=0),
            "samples": output, "source_refs": _source_refs(item["source_refs"], f"{path}.source_refs")}


def _map_bar_pulse_count(numerator: int, denominator: int, beat_unit: str) -> int | None:
    pulse_count = Fraction(numerator * 4, denominator) / _UNIT_FACTORS[beat_unit]
    if pulse_count.denominator != 1 or not 1 <= pulse_count.numerator <= 256:
        return None
    return pulse_count.numerator


def _map_unique_phase_index(beats: list[dict[str, Any]], phase_time: int | float) -> int | None:
    fresh_context = Context(prec=50, rounding=ROUND_HALF_EVEN)
    fresh_context.traps[Inexact] = False
    fresh_context.traps[Rounded] = False
    with localcontext(fresh_context):
        phase = _canonical_numeric_projection(phase_time)
        matches = [
            index for index, beat in enumerate(beats)
            if abs(_canonical_numeric_projection(beat["time_seconds"]) - phase) <= _TIME_TOLERANCE
        ]
    return matches[0] if len(matches) == 1 else None


def _expected_v2_bar_projection(
    bars: list[dict[str, Any]],
    beats: list[dict[str, Any]],
    downbeats: list[dict[str, Any]],
    meter: dict[str, Any],
    tempo: dict[str, Any],
    duration: int | float,
    path: str,
) -> tuple[list[tuple[bytes, bytes, str | None, list[str]]], bool]:
    """Build resolver-v2 bars and the exact bar-math withholding state."""
    if meter["status"] not in _BAR_CAPABLE_METER_STATUSES or tempo["status"] == "no_pulse":
        return [], False
    if meter["status"] == "mixed" and not meter["regions"]:
        return [], False
    if not beats or not downbeats:
        return [], True

    duration_projection = _canonical_numeric_projection(duration)
    candidates: list[tuple[Decimal, int, Decimal, bytes, bytes, str | None, list[str]]] = []
    withheld = False

    def append_group(
        phase_index: int,
        pulse_count: int,
        region_id: str | None,
        region_end: int | float | None,
        refs: list[str],
    ) -> None:
        bar_index = phase_index
        while bar_index < len(beats):
            start = beats[bar_index]["time_seconds"]
            start_projection = _canonical_numeric_projection(start)
            if start_projection >= duration_projection:
                break
            if region_end is not None and start_projection >= _canonical_numeric_projection(region_end):
                break
            next_index = bar_index + pulse_count
            next_time = duration if next_index >= len(beats) else beats[next_index]["time_seconds"]
            end_values = (next_time, duration) if region_end is None else (next_time, region_end, duration)
            end = min(end_values, key=_canonical_numeric_projection)
            if _canonical_numeric_projection(end) == duration_projection:
                end = duration
            end_projection = _canonical_numeric_projection(end)
            if end_projection < start_projection:
                break
            candidates.append((
                start_projection,
                len(candidates),
                end_projection,
                canonical_json_bytes(start),
                canonical_json_bytes(end),
                region_id,
                refs,
            ))
            if len(candidates) > len(bars):
                _fail(path, "contains more bars than deterministic resolver v2 projection")
            bar_index = next_index

    if meter["status"] in {"fixed", "beats_per_bar_only"}:
        phase_index = _map_unique_phase_index(beats, downbeats[0]["time_seconds"])
        if phase_index is None:
            return [], True
        if meter["status"] == "beats_per_bar_only":
            pulse_count = int(meter["beats_per_bar"])
        else:
            pulse_count = _map_bar_pulse_count(int(meter["numerator"]), int(meter["denominator"]), tempo["beat_unit"])
        if pulse_count is None:
            return [], True
        append_group(phase_index, pulse_count, None, None, ["map:tempo", "map:meter", "map:downbeat.000001"])
    elif meter["status"] == "mixed":
        anchor = _canonical_numeric_projection(downbeats[0]["time_seconds"])
        containing_regions = [
            index for index, region in enumerate(meter["regions"])
            if _canonical_numeric_projection(region["start_seconds"]) <= anchor
            < _canonical_numeric_projection(region["end_seconds"])
        ]
        if len(containing_regions) != 1:
            return [], True
        anchor_region_index = containing_regions[0]
        global_phase_index = _map_unique_phase_index(beats, downbeats[0]["time_seconds"])
        if global_phase_index is None:
            return [], True
        for index, region in enumerate(meter["regions"]):
            pulse_count = _map_bar_pulse_count(int(region["numerator"]), int(region["denominator"]), tempo["beat_unit"])
            if pulse_count is None:
                withheld = True
                continue
            if index < anchor_region_index:
                continue
            if index == anchor_region_index:
                phase_index = global_phase_index
                refs = ["map:tempo", "map:meter", f"interpretation:meter.region.{region['region_id']}", "map:downbeat.000001"]
            else:
                phase_index = _map_unique_phase_index(beats, region["start_seconds"])
                if phase_index is None:
                    withheld = True
                    continue
                refs = ["map:tempo", "map:meter", f"interpretation:meter.region.{region['region_id']}"]
            append_group(phase_index, pulse_count, region["region_id"], region["end_seconds"], refs)

    candidates.sort(key=lambda item: (item[0], item[1]))
    return [
        (start_bytes, end_bytes, region_id, sorted(set(refs)))
        for _, _, _, start_bytes, end_bytes, region_id, refs in candidates
    ], withheld


def _validate_v2_bar_projection(
    bars: list[dict[str, Any]],
    beats: list[dict[str, Any]],
    downbeats: list[dict[str, Any]],
    meter: dict[str, Any],
    tempo: dict[str, Any],
    duration: int | float,
    path: str,
    warnings: list[dict[str, Any]],
) -> None:
    """Validate the deterministic v2 bar projection from normalized map data."""
    expected, withheld = _expected_v2_bar_projection(bars, beats, downbeats, meter, tempo, duration, path)
    if len(bars) != len(expected):
        _fail(path, "does not match the deterministic resolver v2 bar count")
    for index, (bar, (start, end, region_id, refs)) in enumerate(zip(bars, expected)):
        if canonical_json_bytes(bar["start_seconds"]) != start:
            _fail(f"{path}[{index}].start_seconds", "does not match the authoritative resolver v2 bar start bytes")
        if canonical_json_bytes(bar["end_seconds"]) != end:
            _fail(f"{path}[{index}].end_seconds", "does not match the authoritative resolver v2 bar end bytes")
        if bar["meter_region_id"] != region_id:
            _fail(f"{path}[{index}].meter_region_id", "does not match the deterministic resolver v2 bar region")
        if bar["source_refs"] != refs:
            _fail(f"{path}[{index}].source_refs", "does not match the deterministic resolver v2 bar provenance")
    bar_warnings = [row for row in warnings if row["code"] == "bar_math_withheld"]
    if len(bar_warnings) != (1 if withheld else 0):
        _fail("$.warnings", "bar_math_withheld presence does not match resolver v2 bar projection")

def validate_music_map(value: Any) -> dict[str, Any]:
    """Validate and normalize a complete ``studio.music-map/1`` artifact."""
    artifact = _detached(value)
    _artifact_size(artifact, _MAP_BYTES, "Effective Map")
    top = _object(artifact, "$", {"schema", "track_sha256", "map_id", "evidence_id", "interpretation", "resolver", "duration_seconds", "tempo", "beats", "meter", "downbeats", "bars", "sections", "accents", "energy", "lyrics", "warnings"})
    if top["schema"] != "studio.music-map/1":
        _fail("$.schema", "must equal studio.music-map/1")
    duration = _number(top["duration_seconds"], "$.duration_seconds", minimum=0)
    if _canonical_numeric_projection(duration) <= Decimal(0):
        _fail("$.duration_seconds", "must be > 0")
    interpretation = _object(top["interpretation"], "$.interpretation", {"scope", "revision"})
    interpretation_out = {"scope": _validate_scope(interpretation["scope"], "$.interpretation.scope", allow_default=True),
                          "revision": _integer(interpretation["revision"], "$.interpretation.revision", minimum=0)}
    if interpretation_out["scope"]["kind"] == "default" and interpretation_out["revision"] != 0:
        _fail("$.interpretation.revision", "default interpretation must have revision 0")
    if interpretation_out["scope"]["kind"] != "default" and interpretation_out["revision"] < 1:
        _fail("$.interpretation.revision", "persisted interpretation must have revision >= 1")
    resolver = _object(top["resolver"], "$.resolver", {"id", "version"})
    resolver_out = {"id": _enum(resolver["id"], "$.resolver.id", {"studio.music-map-resolver"}), "version": _enum(resolver["version"], "$.resolver.version", {"1", "2"})}
    tempo = _validate_map_tempo(top["tempo"], "$.tempo")
    beats = _map_event_rows(top["beats"], "$.beats", "beat", duration)
    downbeats = _map_event_rows(top["downbeats"], "$.downbeats", "downbeat", duration)
    accents = _map_event_rows(top["accents"], "$.accents", "accent", duration, accents=True)
    meter = _validate_map_meter(top["meter"], "$.meter", duration)
    region_index = {region["region_id"]: region for region in meter["regions"]}
    effective_beat_times = {
        _canonical_numeric_projection(beat["time_seconds"])
        for beat in beats
    }
    global_downbeat_ref = None if not downbeats else f"map:{downbeats[0]['event_id']}"
    global_anchor = None if not downbeats else _canonical_numeric_projection(downbeats[0]["time_seconds"])
    seen_region_bars: set[str] = set()
    bars_in = _array(top["bars"], "$.bars", maximum=_MAX_LIST)
    bars: list[dict[str, Any]] = []
    previous_end: Decimal | None = None
    for index, row in enumerate(bars_in):
        item = _object(row, f"$.bars[{index}]", {"bar_id", "number", "start_seconds", "end_seconds", "meter_region_id", "source_refs"})
        if item["bar_id"] != f"bar.{index + 1:06d}":
            _fail(f"$.bars[{index}].bar_id", "must be chronological bar ordinal")
        number = _integer(item["number"], f"$.bars[{index}].number", minimum=1)
        if number != index + 1:
            _fail(f"$.bars[{index}].number", "must equal global bar ordinal")
        start, end = _time_range(item["start_seconds"], item["end_seconds"], f"$.bars[{index}]", duration)
        start_projection = _canonical_numeric_projection(start)
        end_projection = _canonical_numeric_projection(end)
        if previous_end is not None and _timeline_before(start_projection, previous_end):
            _fail("$.bars", "bars must be ordered and non-overlapping")
        previous_end = end_projection
        meter_region_id = None if item["meter_region_id"] is None else _validate_id(item["meter_region_id"], f"$.bars[{index}].meter_region_id")
        source_refs = _source_refs(item["source_refs"], f"$.bars[{index}].source_refs")
        if meter["status"] == "mixed":
            if meter_region_id is None:
                _fail(f"$.bars[{index}].meter_region_id", "mixed bars require a meter region")
            region = region_index.get(meter_region_id)
            if region is None:
                _fail(f"$.bars[{index}].meter_region_id", "does not reference an existing meter region")
            region_start = _canonical_numeric_projection(region["start_seconds"])
            region_end = _canonical_numeric_projection(region["end_seconds"])
            if end_projection > region_end:
                _fail(f"$.bars[{index}]", "must be fully contained in its meter region")
            is_first_region_bar = meter_region_id not in seen_region_bars
            contains_global_anchor = global_anchor is not None and region_start <= global_anchor < region_end
            if start_projection < region_start:
                near_boundary_start = (
                    is_first_region_bar
                    and not contains_global_anchor
                    and abs(start_projection - region_start) <= _TIME_TOLERANCE
                    and start_projection in effective_beat_times
                )
                if not near_boundary_start:
                    _fail(f"$.bars[{index}]", "must be fully contained in its meter region")
            expected_refs = {"map:tempo", "map:meter", f"interpretation:meter.region.{meter_region_id}"}
            if contains_global_anchor:
                expected_refs.add(global_downbeat_ref)
            if source_refs != sorted(expected_refs):
                _fail(f"$.bars[{index}].source_refs", "do not match meter region provenance")
            seen_region_bars.add(meter_region_id)
        elif meter["status"] in {"fixed", "beats_per_bar_only"}:
            if meter_region_id is not None:
                _fail(f"$.bars[{index}].meter_region_id", f"{meter['status']} bars must not reference a meter region")
            if global_downbeat_ref is None:
                _fail("$.bars", f"{meter['status']} bars require a global downbeat")
            expected_refs = sorted({"map:tempo", "map:meter", global_downbeat_ref})
            if source_refs != expected_refs:
                _fail(f"$.bars[{index}].source_refs", f"do not match {meter['status']} meter provenance")
        bars.append({"bar_id": _string(item["bar_id"], f"$.bars[{index}].bar_id", max_bytes=16, ascii_only=True),
                     "number": number, "start_seconds": start, "end_seconds": end,
                     "meter_region_id": meter_region_id, "source_refs": source_refs})
    result = {"schema": "studio.music-map/1", "track_sha256": _validate_sha(top["track_sha256"], "$.track_sha256"),
              "map_id": _validate_sha(top["map_id"], "$.map_id"), "evidence_id": _validate_sha(top["evidence_id"], "$.evidence_id"),
              "interpretation": interpretation_out, "resolver": resolver_out, "duration_seconds": duration,
              "tempo": tempo, "beats": beats,
              "meter": meter, "downbeats": downbeats, "bars": bars,
              "sections": _map_section_rows(top["sections"], "$.sections", duration), "accents": accents,
              "energy": _validate_map_energy(top["energy"], "$.energy", duration), "lyrics": None,
              "warnings": _map_warnings(top["warnings"], "$.warnings")}
    if top["lyrics"] is not None:
        _fail("$.lyrics", "must be null in Phase 2")
    if resolver_out["version"] == "2":
        _validate_v2_bar_projection(bars, beats, downbeats, meter, tempo, duration, "$.bars", result["warnings"])
    _artifact_size(result, _MAP_BYTES, "Effective Map")
    expected = compute_map_id(result)
    if result["map_id"] != expected:
        _fail("$.map_id", "does not match complete artifact content")
    return result


def _artifact_id(value: Any, field: str) -> str:
    detached = _detached(value)
    if not isinstance(detached, dict):
        _fail("$", "artifact must be an object")
    body = dict(detached)
    body.pop(field, None)
    return _sha(body)


def compute_evidence_id(artifact_without_id: Any) -> str:
    """Compute the bare SHA-256 Evidence artifact identity."""
    return _artifact_id(artifact_without_id, "evidence_id")


def compute_waveform_id(artifact_without_id: Any) -> str:
    """Compute the bare SHA-256 Waveform artifact identity."""
    return _artifact_id(artifact_without_id, "waveform_id")


def compute_map_id(artifact_without_id: Any) -> str:
    """Compute the bare SHA-256 Effective Map artifact identity."""
    return _artifact_id(artifact_without_id, "map_id")


def music_timing_projection(map_or_preview: Any) -> dict[str, Any]:
    """Return the detached canonical Type-C timing projection."""
    value = _detached(map_or_preview)
    if not isinstance(value, dict):
        _fail("$", "artifact must be an object")
    keys = ("track_sha256", "evidence_id", "resolver", "duration_seconds", "tempo", "beats", "meter", "downbeats", "bars", "warnings")
    missing = [key for key in keys if key not in value]
    if missing:
        _fail("$", f"missing key(s): {', '.join(missing)}")
    return {key: value[key] for key in keys}


__all__ = [
    "MusicContractError", "MusicCandidateMissingError", "validate_evidence", "validate_waveform", "validate_interpretation_body",
    "validate_interpretation", "validate_interpretation_against_evidence", "validate_music_map", "music_timing_projection", "canonical_json_bytes", "compute_evidence_id",
    "compute_waveform_id", "compute_map_id", "default_interpretation_body", "build_interpretation",
]


def validate_waveform(value: Any) -> dict[str, Any]:
    """Validate and normalize a complete ``studio.music-waveform/1`` artifact."""
    artifact = _detached(value)
    _artifact_size(artifact, _WAVEFORM_BYTES, "Waveform")
    top = _object(artifact, "$", {"schema", "track_sha256", "waveform_id", "requested_bucket_rate_hz", "estimated_effective_bucket_rate_hz", "estimated_bucket_span_seconds", "bucket_count", "duration_seconds", "buckets"})
    if top["schema"] != "studio.music-waveform/1":
        _fail("$.schema", "must equal studio.music-waveform/1")
    duration = _number(top["duration_seconds"], "$.duration_seconds", minimum=0)
    if _canonical_numeric_projection(duration) <= Decimal(0):
        _fail("$.duration_seconds", "must be > 0")
    count = _integer(top["bucket_count"], "$.bucket_count", minimum=1, maximum=_MAX_LIST)
    buckets = _array(top["buckets"], "$.buckets", maximum=_MAX_LIST)
    if len(buckets) != count:
        _fail("$.buckets", "length must equal bucket_count")
    normalized_buckets: list[list[int | float]] = []
    for index, bucket in enumerate(buckets):
        pair = _array(bucket, f"$.buckets[{index}]", maximum=2)
        if len(pair) != 2:
            _fail(f"$.buckets[{index}]", "must contain exactly min and max")
        minimum = _number(pair[0], f"$.buckets[{index}][0]", minimum=-1, maximum=1)
        maximum = _number(pair[1], f"$.buckets[{index}][1]", minimum=-1, maximum=1)
        if _canonical_numeric_projection(minimum) > _canonical_numeric_projection(maximum):
            _fail(f"$.buckets[{index}]", "min must be <= max")
        normalized_buckets.append([minimum, maximum])
    rate = _number(top["estimated_effective_bucket_rate_hz"], "$.estimated_effective_bucket_rate_hz", minimum=0)
    span = _number(top["estimated_bucket_span_seconds"], "$.estimated_bucket_span_seconds", minimum=0)
    if not _waveform_equations_match(duration, count, rate, span):
        _fail("$", "waveform estimates do not match bucket_count/duration equations")
    result = {"schema": "studio.music-waveform/1", "track_sha256": _validate_sha(top["track_sha256"], "$.track_sha256"),
              "waveform_id": _validate_sha(top["waveform_id"], "$.waveform_id"), "requested_bucket_rate_hz": _integer(top["requested_bucket_rate_hz"], "$.requested_bucket_rate_hz", minimum=1),
              "estimated_effective_bucket_rate_hz": rate, "estimated_bucket_span_seconds": span, "bucket_count": count,
              "duration_seconds": duration, "buckets": normalized_buckets}
    if result["requested_bucket_rate_hz"] != 200:
        _fail("$.requested_bucket_rate_hz", "must equal 200")
    _artifact_size(result, _WAVEFORM_BYTES, "Waveform")
    if result["waveform_id"] != compute_waveform_id(result):
        _fail("$.waveform_id", "does not match complete artifact content")
    return result
