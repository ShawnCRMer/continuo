"""Deterministic Phase-2 resolver for the neutral Effective Music Map artifact."""

from __future__ import annotations

import copy
from decimal import Context, Decimal, Inexact, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, Rounded, localcontext
from fractions import Fraction
from typing import Any

from .contracts import (
    _canonical_numeric_projection as _canonical_numeric_projection,
    canonical_json_bytes as _canonical_json_bytes,
    compute_map_id as _compute_map_id,
    default_interpretation_body as _default_interpretation_body,
    _detached as _detached,
    _INTERPRETATION_BYTES as _INTERPRETATION_BYTES,
    _MAX_LIST as _MAX_REGULAR_GRID_EVENTS,
    _MAP_BYTES as _MAP_BYTES,
    MusicContractError as _MusicContractError,
    validate_evidence as _validate_evidence,
    validate_interpretation_against_evidence as _validate_interpretation_against_evidence,
    validate_interpretation as _validate_interpretation,
    validate_interpretation_body as _validate_interpretation_body,
    validate_music_map as _validate_music_map,
    _timeline_before as _timeline_before,
)


__all__ = ["preview_music_map", "resolve_music_map"]

_RESOLVER = {"id": "studio.music-map-resolver", "version": "2"}
_WARNING_ORDER = {
    "selected_candidate_missing": 0,
    "regular_grid_anchor_missing": 1,
    "bar_math_withheld": 2,
    "legacy_evidence_loss": 3,
    "catalog_snapshot_absent": 4,
}
_SEMANTIC_LABELS = {
    "intro", "verse", "pre-chorus", "pre chorus", "chorus", "refrain", "hook",
    "bridge", "outro", "ending", "breakdown", "drop", "solo", "instrumental",
}
_UNIT_FACTORS = {
    "whole": Fraction(4, 1),
    "half": Fraction(2, 1),
    "quarter": Fraction(1, 1),
    "eighth": Fraction(1, 2),
    "dotted_quarter": Fraction(3, 2),
}
_BAR_CAPABLE_METER_STATUSES = {"fixed", "mixed", "beats_per_bar_only"}


class _PreviewEncodedSizeBudget:
    """Track exact canonical bytes without materializing unbounded output rows."""

    def __init__(self, preview_floor: dict[str, Any], cap: int) -> None:
        self._size = len(_canonical_json_bytes(preview_floor))
        self._cap = cap
        self._counts: dict[str, int] = {}

    @property
    def size(self) -> int:
        return self._size

    def replace(self, field: str, value: object) -> None:
        empty = {} if field in {"tempo", "meter"} else []
        self._reserve(len(_canonical_json_bytes(value)) - len(_canonical_json_bytes(empty)))

    def append(self, field: str, value: object) -> None:
        self.append_encoded(field, len(_canonical_json_bytes(value)))

    def append_encoded(self, field: str, encoded_size: int) -> None:
        count = self._counts.get(field, 0)
        self._reserve(encoded_size + (1 if count else 0))
        self._counts[field] = count + 1

    def append_event(
        self,
        field: str,
        kind: str,
        ordinal: int,
        time_seconds: int | float,
        source_refs: list[str] | str,
        *,
        strength: int | float | None = None,
        salience: int | float | None = None,
    ) -> None:
        event_id = f"{kind}.{ordinal:06d}"
        encoded_size = _event_encoded_size(
            event_id,
            time_seconds,
            source_refs,
            strength=strength,
            salience=salience,
        )
        self.append_encoded(field, encoded_size)

    def append_bar(
        self,
        field: str,
        ordinal: int,
        start_seconds: int | float,
        end_seconds: int | float,
        meter_region_id: str | None,
        source_refs: list[str],
    ) -> None:
        self.append_encoded(
            field,
            _bar_encoded_size(
                f"bar.{ordinal:06d}",
                ordinal,
                start_seconds,
                end_seconds,
                meter_region_id,
                source_refs,
            ),
        )

    def _reserve(self, increment: int) -> None:
        self._size += increment
        if self._size > self._cap:
            raise _MusicContractError(
                f"Preview artifact exceeds {self._cap} canonical bytes"
            )


class _LazyEventSeries:
    """A bounded event view used by the dry-run and by bar phase calculations."""

    def __init__(self, count: int, time_at) -> None:
        self._count = count
        self._time_at = time_at

    def __len__(self) -> int:
        return self._count

    def time_at(self, index: int) -> int | float:
        return self._time_at(index)


def _event_time(events: object, index: int) -> int | float:
    if isinstance(events, _LazyEventSeries):
        return events.time_at(index)
    return events[index]["time_seconds"]


def _event_count(events: object) -> int:
    return len(events)


def _source_refs_encoded_size(source_refs: list[str] | str) -> int:
    if isinstance(source_refs, str):
        # Dry-run copied evidence events have exactly one ASCII-safe ref and do
        # not need a one-item list allocation merely to count its JSON bytes.
        return len(_canonical_json_bytes(source_refs)) + 2
    return len(_canonical_json_bytes(source_refs))


def _event_encoded_size(
    event_id: str,
    time_seconds: int | float,
    source_refs: list[str] | str,
    *,
    strength: int | float | None = None,
    salience: int | float | None = None,
) -> int:
    """Return the exact canonical size of one map event without making its dict."""
    size = len(b'{"event_id":') + len(_canonical_json_bytes(event_id))
    if salience is not None:
        size += len(b',"salience":') + len(_canonical_json_bytes(salience))
    size += len(b',"source_refs":') + _source_refs_encoded_size(source_refs)
    if strength is not None:
        size += len(b',"strength":') + len(_canonical_json_bytes(strength))
    return size + len(b',"time_seconds":') + len(_canonical_json_bytes(time_seconds)) + len(b"}")


def _bar_encoded_size(
    bar_id: str,
    number: int,
    start_seconds: int | float,
    end_seconds: int | float,
    meter_region_id: str | None,
    source_refs: list[str],
) -> int:
    """Return the exact canonical size of one map bar without making its dict."""
    size = len(b'{"bar_id":') + len(_canonical_json_bytes(bar_id))
    size += len(b',"end_seconds":') + len(_canonical_json_bytes(end_seconds))
    size += len(b',"meter_region_id":') + len(_canonical_json_bytes(meter_region_id))
    size += len(b',"number":') + len(_canonical_json_bytes(number))
    size += len(b',"source_refs":') + len(_canonical_json_bytes(source_refs))
    return size + len(b',"start_seconds":') + len(_canonical_json_bytes(start_seconds)) + len(b"}")


def _refs(*values: str | None) -> list[str]:
    return sorted({value for value in values if value is not None})


def _duration(evidence: dict[str, Any]) -> int | float:
    return evidence["analyzer"]["decoded_audio"]["duration_seconds"]


def _interpretation(value: object, duration: int | float) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a validated body and the small, persisted envelope projection."""
    if value is None:
        return _default_interpretation_body(), {"scope": {"kind": "default"}, "revision": 0}
    value = _detached(value)
    is_default_marker = isinstance(value, str) and value in ("default", "none")
    if isinstance(value, dict):
        is_default_marker = is_default_marker or value == {"kind": "default"} or value == {"kind": "none"}
        is_default_marker = is_default_marker or value == {"scope": {"kind": "default"}, "revision": 0}
    if is_default_marker:
        return _default_interpretation_body(), {"scope": {"kind": "default"}, "revision": 0}
    if isinstance(value, dict) and set(value) == {"tempo", "meter", "downbeat", "sections", "lyrics"}:
        return _validate_interpretation_body(value, duration_seconds=duration), {"scope": {"kind": "default"}, "revision": 0}
    envelope = _validate_interpretation(value, duration_seconds=duration)
    body = {key: envelope[key] for key in ("tempo", "meter", "downbeat", "sections", "lyrics")}
    return body, {"scope": envelope["scope"], "revision": envelope["revision"]}


def _confidence(value: Any) -> Decimal:
    return Decimal("-1") if value is None else _canonical_numeric_projection(value)


def _candidate_choice(candidates: list[dict[str, Any]], *, selected_for_grid: bool = False) -> dict[str, Any] | None:
    best = None
    best_key = None
    for row in candidates:
        if selected_for_grid and not row.get("selected_for_grid"):
            continue
        key = (-_confidence(row.get("confidence")), str(row["candidate_id"]))
        if best_key is None or key < best_key:
            best = row
            best_key = key
    return best


def _catalog_sourced(catalog: dict[str, Any]) -> bool:
    # A hash alone is the stable empty projection, not a recording fact.
    return catalog.get("match") == "sha256" and bool(catalog.get("citations"))


def _catalog_present(catalog: dict[str, Any]) -> bool:
    return catalog.get("match") == "sha256"


def _catalog_tempo(catalog: dict[str, Any]) -> tuple[dict[str, Any] | None, float | None]:
    value = catalog.get("tempo")
    if not isinstance(value, dict):
        return None, None
    status = value.get("status")
    bpm = value.get("bpm")
    if status == "fixed" and isinstance(bpm, (int, float)) and not isinstance(bpm, bool):
        candidates = value.get("candidates")
        confidence = None
        if isinstance(candidates, list) and candidates:
            confidence = next((row.get("confidence") for row in candidates if isinstance(row, dict) and row.get("bpm") == bpm), None)
        return {"status": "fixed", "bpm": bpm}, confidence
    if status in {"ambiguous", "none"}:
        return {"status": "unavailable" if status == "none" else status, "bpm": None}, None
    if status == "variable":
        return {"status": "variable", "bpm": None}, None
    return None, None


def _signal_tempo(evidence: dict[str, Any], selected_id: str | None) -> tuple[dict[str, Any] | None, bool]:
    family = evidence["families"]["tempo"]
    candidates = family["candidates"]
    if selected_id is not None:
        for candidate in candidates:
            if candidate["candidate_id"] == selected_id:
                return candidate, True
        return None, False
    # v6 selected_for_grid wins over confidence, then candidate_id.
    selected = _candidate_choice(candidates, selected_for_grid=True)
    return selected or _candidate_choice(candidates), True


def _effective_grid_source(evidence: dict[str, Any], requested: str, status: str) -> str:
    if requested == "inherit" and status != "no_pulse":
        beats = evidence["families"]["beats"]
        if beats["status"] != "unavailable" and beats["events"]:
            return "detected_beats"
    return requested


def _tempo_claim(
    evidence: dict[str, Any],
    body: dict[str, Any],
    catalog: dict[str, Any],
    warnings: list[dict[str, Any]],
    *,
    budget: _PreviewEncodedSizeBudget | None = None,
) -> dict[str, Any]:
    authored = body["tempo"]
    mode = authored["mode"]
    beat_unit = authored["beat_unit"]
    grid_requested = authored["grid_source"]
    if mode == "fixed":
        return {
            "status": "fixed", "bpm": authored["bpm"], "beat_unit": beat_unit,
            "grid_source": _effective_grid_source(evidence, grid_requested, "fixed"), "confidence": 1.0,
            "source_refs": _refs("interpretation:tempo.bpm", "interpretation:tempo.beat_unit", "interpretation:tempo.grid_source", "interpretation:tempo.mode"),
        }
    if mode == "no_pulse":
        return {
            "status": "no_pulse", "bpm": None, "beat_unit": beat_unit,
            "grid_source": grid_requested, "confidence": None,
            "source_refs": _refs("interpretation:tempo.beat_unit", "interpretation:tempo.grid_source", "interpretation:tempo.mode"),
        }
    if mode == "variable":
        return {
            "status": "variable", "bpm": None, "beat_unit": beat_unit,
            "grid_source": grid_requested, "confidence": None,
            "source_refs": _refs("interpretation:tempo.beat_unit", "interpretation:tempo.grid_source", "interpretation:tempo.mode"),
        }

    selected_id = authored["selected_candidate_id"]
    selected = None
    selected_valid = True
    if selected_id is not None:
        selected, selected_valid = _signal_tempo(evidence, selected_id)
        if not selected_valid:
            _warning(
                warnings,
                "selected_candidate_missing",
                ["interpretation:tempo.selected_candidate_id"],
                budget=budget,
            )

    family = evidence["families"]["tempo"]
    catalog_claim, catalog_confidence = _catalog_tempo(catalog) if _catalog_sourced(catalog) else (None, None)
    # The explicit Interpretation selection is the only evidence claim that
    # precedes an exact-recording catalog fact.  Ordinary signal candidates do
    # not outrank the catalog.
    if selected is not None:
        return {
            "status": "fixed", "bpm": selected["bpm"], "beat_unit": beat_unit,
            "grid_source": _effective_grid_source(evidence, grid_requested, "fixed"), "confidence": selected.get("confidence"),
            "source_refs": _refs(f"evidence:{selected['candidate_id']}", "interpretation:tempo.selected_candidate_id"),
        }
    if catalog_claim is not None:
        return {
            "status": catalog_claim["status"], "bpm": catalog_claim["bpm"], "beat_unit": beat_unit,
            "grid_source": _effective_grid_source(evidence, grid_requested, catalog_claim["status"]), "confidence": catalog_confidence,
            "source_refs": _refs("catalog:catalog_snapshot_hash", "catalog:tempo"),
        }
    selected, _ = _signal_tempo(evidence, None)
    # Keep the evidence family status only when it carries a neutral status.
    if family["status"] == "ambiguous":
        ambiguous_candidate = _candidate_choice(family["candidates"])
        source_refs = [] if ambiguous_candidate is None else [f"evidence:{ambiguous_candidate['candidate_id']}"]
        return {
            "status": "ambiguous", "bpm": None, "beat_unit": beat_unit,
            "grid_source": _effective_grid_source(evidence, grid_requested, "ambiguous"), "confidence": None,
            "source_refs": source_refs,
        }
    if selected is None:
        return {
            "status": "unavailable", "bpm": None, "beat_unit": beat_unit,
            "grid_source": _effective_grid_source(evidence, grid_requested, "unavailable"), "confidence": None, "source_refs": [],
        }
    return {
        "status": "fixed", "bpm": selected["bpm"], "beat_unit": beat_unit,
        "grid_source": _effective_grid_source(evidence, grid_requested, "fixed"), "confidence": selected.get("confidence"),
        "source_refs": [f"evidence:{selected['candidate_id']}"],
    }


def _copy_evidence_events(
    rows: list[dict[str, Any]],
    kind: str,
    *,
    accents: bool = False,
    budget: _PreviewEncodedSizeBudget | None = None,
    budget_field: str | None = None,
    materialize: bool = True,
) -> list[dict[str, Any]] | _LazyEventSeries:
    # Evidence validation already emits each family in its authoritative order.
    # Only compatibility callers that request materialized output need the
    # defensive sort used by the old private helper; the dry-run keeps no index
    # list proportional to the event count.
    if materialize:
        ordered = sorted(
            enumerate(rows),
            key=lambda pair: (_canonical_numeric_projection(pair[1]["time_seconds"]), pair[0]),
        )
        indexed_rows = (indexed_row[1] for indexed_row in ordered)
        time_at = lambda index: ordered[index][1]["time_seconds"]
    else:
        ordered = rows
        indexed_rows = iter(ordered)
        time_at = lambda index: ordered[index]["time_seconds"]
    series = _LazyEventSeries(len(ordered), time_at)
    output: list[dict[str, Any]] | None = [] if materialize else None
    for ordinal, row in enumerate(indexed_rows, 1):
        source_ref = f"evidence:{row['event_id']}"
        if budget is not None and budget_field is not None:
            budget.append_event(
                budget_field,
                kind,
                ordinal,
                row["time_seconds"],
                source_ref if not materialize else [source_ref],
                strength=row.get("strength") if accents else None,
                salience=row.get("salience") if accents else None,
            )
        if materialize:
            source_refs = [source_ref]
            item = {
                "event_id": f"{kind}.{ordinal:06d}",
                "time_seconds": row["time_seconds"],
                "source_refs": source_refs,
            }
            if accents:
                item["strength"] = row["strength"]
                item["salience"] = row["salience"]
            assert output is not None
            output.append(item)
    return output if materialize else series


def _regular_beats(
    anchor: int | float,
    duration: int | float,
    bpm: int | float,
    beat_unit: str,
    refs: list[str],
    *,
    budget: _PreviewEncodedSizeBudget | None = None,
    materialize: bool = True,
) -> list[dict[str, Any]] | _LazyEventSeries:
    first_k, last_k = _regular_grid_indices(anchor, duration, bpm)
    cardinality = max(0, last_k - first_k + 1)
    if cardinality > _MAX_REGULAR_GRID_EVENTS:
        raise _MusicContractError(
            "$.tempo.grid_source: regular_grid output exceeds "
            f"the {_MAX_REGULAR_GRID_EVENTS}-event cap"
        )
    if cardinality == 0:
        return []
    fresh_context = Context(prec=50, rounding=ROUND_HALF_EVEN)
    fresh_context.traps[Inexact] = False
    fresh_context.traps[Rounded] = False
    with localcontext(fresh_context):
        decimal_anchor = Decimal(_canonical_json_bytes(anchor).decode("ascii"))
        decimal_bpm = Decimal(_canonical_json_bytes(bpm).decode("ascii"))
        if not decimal_bpm.is_finite() or decimal_bpm <= 0:
            return []
        delta = Decimal(60) / decimal_bpm
        quantum = Decimal("0.000000001")
        previous = None
        for k in range(first_k, last_k + 1):
            time_value = (decimal_anchor + Decimal(k) * delta).quantize(quantum, rounding=ROUND_HALF_EVEN)
            if previous is not None and time_value == previous:
                return []
            previous = time_value

    def time_for_k(k: int) -> float:
        with localcontext(fresh_context):
            time_value = (decimal_anchor + Decimal(k) * delta).quantize(quantum, rounding=ROUND_HALF_EVEN)
            return float(time_value)

    def time_at(index: int) -> float:
        return time_for_k(first_k + index)

    series = _LazyEventSeries(cardinality, time_at)
    source_refs = sorted(set(refs))
    output: list[dict[str, Any]] | None = [] if materialize else None
    for ordinal, k in enumerate(range(first_k, last_k + 1), 1):
        time_value = time_for_k(k)
        if budget is not None:
            budget.append_event("beats", "beat", ordinal, time_value, source_refs)
        if materialize:
            assert output is not None
            output.append({"event_id": f"beat.{ordinal:06d}", "time_seconds": time_value, "source_refs": source_refs})
    return output if materialize else series


def _canonical_fraction(value: int | float) -> Fraction:
    return Fraction(Decimal(_canonical_json_bytes(value).decode("ascii")))


def _ceil_fraction(value: Fraction) -> int:
    quotient, remainder = divmod(value.numerator, value.denominator)
    return quotient + (1 if remainder else 0)


def _regular_grid_indices(anchor: int | float, duration: int | float, bpm: int | float) -> tuple[int, int]:
    """Return exact inclusive pulse indices without allocating the grid."""
    anchor_value = _canonical_fraction(anchor)
    duration_value = _canonical_fraction(duration)
    bpm_value = _canonical_fraction(bpm)
    if bpm_value <= 0:
        return 1, 0
    interval = Fraction(60, 1) / bpm_value
    first_k = _ceil_fraction(-anchor_value / interval)
    last_k = (duration_value - anchor_value) // interval
    return first_k, last_k


def _regular_grid_cardinality(anchor: int | float, duration: int | float, bpm: int | float) -> int:
    first_k, last_k = _regular_grid_indices(anchor, duration, bpm)
    return max(0, last_k - first_k + 1)


def _downbeats(
    evidence: dict[str, Any],
    authored: dict[str, Any],
    *,
    no_pulse: bool,
    budget: _PreviewEncodedSizeBudget | None = None,
    materialize: bool = True,
) -> list[dict[str, Any]] | _LazyEventSeries:
    if no_pulse or authored["mode"] == "none":
        return []
    if authored["mode"] == "anchor":
        source_refs = _refs("interpretation:downbeat.anchor_seconds", "interpretation:downbeat.mode")
        anchor = authored["anchor_seconds"]
        if budget is not None:
            budget.append_event("downbeats", "downbeat", 1, anchor, source_refs)
        if not materialize:
            return _LazyEventSeries(1, lambda _index: anchor)
        return [{"event_id": "downbeat.000001", "time_seconds": anchor, "source_refs": source_refs}]
    family = evidence["families"]["downbeats"]
    if family["status"] == "unavailable":
        return []
    return _copy_evidence_events(
        family["events"],
        "downbeat",
        budget=budget,
        budget_field="downbeats",
        materialize=materialize,
    )


def _beats(
    evidence: dict[str, Any],
    authored: dict[str, Any],
    tempo: dict[str, Any],
    downbeats: list[dict[str, Any]] | _LazyEventSeries,
    duration: int | float,
    warnings: list[dict[str, Any]],
    *,
    budget: _PreviewEncodedSizeBudget | None = None,
    materialize: bool = True,
) -> list[dict[str, Any]] | _LazyEventSeries:
    if tempo["status"] == "no_pulse":
        return []
    source = tempo["grid_source"]
    if source == "regular_grid":
        if tempo["status"] != "fixed":
            return []
        if not downbeats:
            _warning(
                warnings,
                "regular_grid_anchor_missing",
                ["interpretation:tempo.grid_source"],
                budget=budget,
            )
            return []
        anchor = _event_time(downbeats, 0)
        if authored["downbeat"]["mode"] == "anchor":
            refs = ["interpretation:downbeat.anchor_seconds", "map:tempo"]
        else:
            refs = ["map:downbeat.000001", "map:tempo"]
        return _regular_beats(
            anchor,
            duration,
            tempo["bpm"],
            tempo["beat_unit"],
            refs,
            budget=budget,
            materialize=materialize,
        )
    family = evidence["families"]["beats"]
    if family["status"] == "unavailable":
        return []
    return _copy_evidence_events(
        family["events"],
        "beat",
        budget=budget,
        budget_field="beats",
        materialize=materialize,
    )


def _meter_candidate(evidence: dict[str, Any]) -> dict[str, Any] | None:
    return _candidate_choice(evidence["families"]["meter"]["candidates"])


def _meter_claim(evidence: dict[str, Any], body: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
    authored = body["meter"]
    mode = authored["mode"]
    if mode == "fixed":
        return {
            "status": "fixed", "beats_per_bar": authored["numerator"], "numerator": authored["numerator"], "denominator": authored["denominator"],
            "regions": [], "confidence": 1.0,
            "source_refs": _refs("interpretation:meter.denominator", "interpretation:meter.mode", "interpretation:meter.numerator"),
        }
    if mode == "none":
        return {"status": "none", "beats_per_bar": None, "numerator": None, "denominator": None, "regions": [], "confidence": None, "source_refs": ["interpretation:meter.mode"]}
    if mode == "mixed":
        regions = []
        top_refs: list[str] = ["interpretation:meter.mode"]
        for row in authored["regions"]:
            ref = f"interpretation:meter.region.{row['region_id']}"
            regions.append({
                "region_id": row["region_id"], "start_seconds": row["start_seconds"], "end_seconds": row["end_seconds"],
                "numerator": row["numerator"], "denominator": row["denominator"], "source_refs": [ref],
            })
            top_refs.append(ref)
        return {"status": "mixed", "beats_per_bar": None, "numerator": None, "denominator": None, "regions": regions, "confidence": 1.0, "source_refs": _refs(*top_refs)}

    if _catalog_sourced(catalog) and isinstance(catalog.get("meter"), dict):
        catalog_meter = catalog["meter"]
        status = catalog_meter.get("status")
        value = catalog_meter.get("value")
        if status == "fixed" and isinstance(value, int) and not isinstance(value, bool):
            return {"status": "beats_per_bar_only", "beats_per_bar": value, "numerator": None, "denominator": None, "regions": [], "confidence": None, "source_refs": _refs("catalog:catalog_snapshot_hash", "catalog:meter")}
        if status in {"mixed", "ambiguous", "none"}:
            return {"status": status, "beats_per_bar": None, "numerator": None, "denominator": None, "regions": [], "confidence": None, "source_refs": _refs("catalog:catalog_snapshot_hash", "catalog:meter")}
    family = evidence["families"]["meter"]
    candidate = _meter_candidate(evidence)
    if family["status"] == "ambiguous":
        return {"status": "ambiguous", "beats_per_bar": None, "numerator": None, "denominator": None, "regions": [], "confidence": None, "source_refs": []}
    if candidate is not None:
        return {"status": "beats_per_bar_only", "beats_per_bar": candidate["beats_per_bar"], "numerator": None, "denominator": None, "regions": [], "confidence": candidate.get("confidence"), "source_refs": [f"evidence:{candidate['candidate_id']}"]}
    return {"status": "unavailable", "beats_per_bar": None, "numerator": None, "denominator": None, "regions": [], "confidence": None, "source_refs": []}


def _bar_pulse_count(numerator: int, denominator: int, beat_unit: str) -> int | None:
    pulse_count = Fraction(numerator * 4, denominator) / _UNIT_FACTORS[beat_unit]
    if pulse_count.denominator != 1 or not 1 <= pulse_count.numerator <= 256:
        return None
    return pulse_count.numerator


def _unique_phase_index(events: list[dict[str, Any]] | _LazyEventSeries, phase_time: int | float) -> int | None:
    fresh_context = Context(prec=50, rounding=ROUND_HALF_EVEN)
    fresh_context.traps[Inexact] = False
    fresh_context.traps[Rounded] = False
    with localcontext(fresh_context):
        decimal_phase = Decimal(_canonical_json_bytes(phase_time).decode("ascii"))
        match_index = None
        match_count = 0
        for index in range(_event_count(events)):
            if abs(Decimal(_canonical_json_bytes(_event_time(events, index)).decode("ascii")) - decimal_phase) <= Decimal("0.000001"):
                match_index = index
                match_count += 1
    return match_index if match_count == 1 else None


def _decimal_projection(value: int | float) -> Decimal:
    return _canonical_numeric_projection(value)


def _bars(
    beats: list[dict[str, Any]] | _LazyEventSeries,
    downbeats: list[dict[str, Any]] | _LazyEventSeries,
    meter: dict[str, Any],
    duration: int | float,
    beat_unit: str,
    *,
    bar_math_suppressed: bool = False,
    budget: _PreviewEncodedSizeBudget | None = None,
    materialize: bool = True,
) -> tuple[list[dict[str, Any]] | None, bool]:
    if meter["status"] not in _BAR_CAPABLE_METER_STATUSES:
        return [], False
    if meter["status"] == "mixed" and not meter["regions"]:
        return [], False
    if not beats or not downbeats:
        return [], not bar_math_suppressed and meter["status"] in _BAR_CAPABLE_METER_STATUSES
    fresh_context = Context(prec=50, rounding=ROUND_HALF_EVEN)
    fresh_context.traps[Inexact] = False
    fresh_context.traps[Rounded] = False
    with localcontext(fresh_context):
        duration_projection = _decimal_projection(duration)
        withheld = False

        if meter["status"] in {"fixed", "beats_per_bar_only"} and meter["beats_per_bar"]:
            phase_index = _unique_phase_index(beats, _event_time(downbeats, 0))
            if phase_index is None:
                return [], True
            pulse_count = (
                int(meter["beats_per_bar"])
                if meter["status"] == "beats_per_bar_only"
                else _bar_pulse_count(int(meter["numerator"]), int(meter["denominator"]), beat_unit)
            )
            if pulse_count is None:
                return [], True
            groups = [(phase_index, None, pulse_count, 1, None)]
        elif meter["status"] == "mixed":
            anchor_time = _event_time(downbeats, 0)
            anchor_projection = _decimal_projection(anchor_time)
            anchor_region_index = next(
                (
                    index for index, region in enumerate(meter["regions"])
                    if _decimal_projection(region["start_seconds"]) <= anchor_projection < _decimal_projection(region["end_seconds"])
                ),
                None,
            )
            if anchor_region_index is None:
                return [], True
            global_phase_index = _unique_phase_index(beats, anchor_time)
            if global_phase_index is None:
                return [], True
            groups: list[tuple[int, str, int, int, int | float]] = []
            for index, region in enumerate(meter["regions"]):
                pulse_count = _bar_pulse_count(int(region["numerator"]), int(region["denominator"]), beat_unit)
                if pulse_count is None:
                    withheld = True
                    continue
                if index < anchor_region_index:
                    continue
                if index == anchor_region_index:
                    phase_index = global_phase_index
                    phase_downbeat_ordinal: int | None = 1
                else:
                    phase_index = _unique_phase_index(beats, region["start_seconds"])
                    phase_downbeat_ordinal = None
                    if phase_index is None:
                        withheld = True
                        continue
                groups.append((phase_index, region["region_id"], pulse_count, phase_downbeat_ordinal or 0, region["end_seconds"]))
        else:
            return [], True

        output: list[dict[str, Any]] | None = [] if materialize else None
        bar_ordinal = 0
        for phase_index, region_id, pulse_count, downbeat_ordinal, region_end in groups:
            bar_start = phase_index
            while bar_start < _event_count(beats):
                start = _event_time(beats, bar_start)
                start_projection = _decimal_projection(start)
                if start_projection >= duration_projection or (region_end is not None and start_projection >= _decimal_projection(region_end)):
                    break
                next_start = bar_start + pulse_count
                next_time = duration if next_start >= _event_count(beats) else _event_time(beats, next_start)
                end_values = (next_time, duration) if region_end is None else (next_time, region_end, duration)
                end = min(end_values, key=lambda value: _decimal_projection(value))
                if _decimal_projection(end) == duration_projection:
                    end = duration
                if _decimal_projection(end) < start_projection:
                    break
                if bar_ordinal >= _MAX_REGULAR_GRID_EVENTS:
                    raise _MusicContractError(
                        f"bar output exceeds the {_MAX_REGULAR_GRID_EVENTS}-event cap"
                    )
                refs = ["map:tempo", "map:meter"]
                if region_id is None:
                    refs.append(f"map:downbeat.{downbeat_ordinal:06d}")
                else:
                    refs.append(f"interpretation:meter.region.{region_id}")
                    if downbeat_ordinal:
                        refs.append(f"map:downbeat.{downbeat_ordinal:06d}")
                source_refs = _refs(*refs)
                bar_ordinal += 1
                if budget is not None:
                    budget.append_bar("bars", bar_ordinal, start, end, region_id, source_refs)
                if materialize:
                    output.append({
                        "bar_id": f"bar.{bar_ordinal:06d}",
                        "number": bar_ordinal,
                        "start_seconds": start,
                        "end_seconds": end,
                        "meter_region_id": region_id,
                        "source_refs": source_refs,
                    })
                bar_start = next_start

        return output, withheld


def _is_semantic(label: str) -> bool:
    return label.strip().casefold() in _SEMANTIC_LABELS


def _effective_section(row: dict[str, Any], source_refs: list[str], source_kind: str) -> dict[str, Any]:
    label = str(row["label"])
    return {
        "start_seconds": row["start_seconds"], "end_seconds": row["end_seconds"], "label": label,
        "label_kind": "semantic" if source_kind in {"artist", "catalog"} and _is_semantic(label) else "neutral",
        "confidence": row.get("confidence"), "source_refs": _refs(*source_refs),
    }


def _sections(evidence: dict[str, Any], body: dict[str, Any], catalog: dict[str, Any], duration: int | float) -> list[dict[str, Any]]:
    authored = body["sections"]
    rows: list[dict[str, Any]] = []
    if authored:
        for row in authored:
            identity = _effective_section(row, [f"interpretation:section.{row['section_id']}"], "artist")
            rows.append(identity)
    elif _catalog_sourced(catalog) and isinstance(catalog.get("structure"), dict) and isinstance(catalog["structure"].get("sections"), list):
        catalog_rows = catalog["structure"]["sections"]
        for index, row in enumerate(catalog_rows):
            if not isinstance(row, dict) or not isinstance(row.get("start"), (int, float)):
                continue
            end = duration
            if index + 1 < len(catalog_rows) and isinstance(catalog_rows[index + 1], dict):
                end = catalog_rows[index + 1].get("start", duration)
            if _canonical_numeric_projection(end) < _canonical_numeric_projection(row["start"]):
                continue
            identity = _effective_section(
                {"start_seconds": row["start"], "end_seconds": end, "label": row.get("label", "A"), "confidence": None},
                ["catalog:catalog_snapshot_hash", "catalog:structure"], "catalog",
            )
            rows.append(identity)
    else:
        family = evidence["families"]["sections"]
        if family["status"] != "unavailable":
            for row in family["candidates"]:
                rows.append(_effective_section(row, [f"evidence:{row['candidate_id']}"], "signal"))
    rows.sort(key=lambda row: (_canonical_numeric_projection(row["start_seconds"]), _canonical_numeric_projection(row["end_seconds"]), row["label"], _canonical_json_bytes(row["source_refs"])))
    output: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    previous_end: Decimal | None = None
    for row in rows:
        start_projection = _canonical_numeric_projection(row["start_seconds"])
        end_projection = _canonical_numeric_projection(row["end_seconds"])
        if previous_end is not None and _timeline_before(start_projection, previous_end):
            continue
        identity = _canonical_json_bytes(row)
        if identity in seen:
            continue
        seen.add(identity)
        section = dict(row)
        section["section_id"] = "section." + _sha(row)
        output.append({"section_id": section.pop("section_id"), **section})
        previous_end = end_projection
    return output


def _sha(value: object) -> str:
    # _canonical_json_bytes is supplied by the frozen sibling contract.
    import hashlib
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _warning(
    warnings: list[dict[str, Any]],
    code: str,
    refs: list[str],
    *,
    budget: _PreviewEncodedSizeBudget | None = None,
) -> None:
    row = {"code": code, "source_refs": list(dict.fromkeys(refs))}
    if row not in warnings:
        if budget is not None:
            budget.append("warnings", row)
        warnings.append(row)


def _preflight_timing_caps(
    evidence_value: dict[str, Any],
    body: dict[str, Any],
    tempo: dict[str, Any],
) -> tuple[int, int]:
    """Reject finite timing output over the list cap before copying/generating it."""
    if tempo["status"] == "no_pulse":
        return 0, 0

    downbeat_family = evidence_value["families"]["downbeats"]
    if body["downbeat"]["mode"] == "inherit" and downbeat_family["status"] != "unavailable":
        downbeat_count = len(downbeat_family["events"])
    else:
        downbeat_count = 1 if body["downbeat"]["mode"] == "anchor" else 0
    if downbeat_count > _MAX_REGULAR_GRID_EVENTS:
        raise _MusicContractError(
            f"downbeat output exceeds the {_MAX_REGULAR_GRID_EVENTS}-event cap"
        )

    if tempo["grid_source"] == "regular_grid":
        if tempo["status"] != "fixed" or downbeat_count == 0:
            return downbeat_count, 0
        anchor = (body["downbeat"]["anchor_seconds"] if body["downbeat"]["mode"] == "anchor"
                  else downbeat_family["events"][0]["time_seconds"])
        cardinality = _regular_grid_cardinality(anchor, _duration(evidence_value), tempo["bpm"])
    else:
        beat_family = evidence_value["families"]["beats"]
        cardinality = len(beat_family["events"]) if beat_family["status"] != "unavailable" else 0
    if cardinality > _MAX_REGULAR_GRID_EVENTS:
        if tempo["grid_source"] == "regular_grid":
            raise _MusicContractError(
                "$.tempo.grid_source: regular_grid output exceeds "
                f"the {_MAX_REGULAR_GRID_EVENTS}-event cap"
            )
        raise _MusicContractError(
            f"beat output exceeds the {_MAX_REGULAR_GRID_EVENTS}-event cap"
        )
    return downbeat_count, cardinality


def _resolve_timing_projection(evidence_value: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    """Build the shared authoritative timing fields without persistence metadata."""
    projection = _build_timing_projection(evidence_value, body)
    assert projection is not None
    return projection


def _build_timing_projection(
    evidence_value: dict[str, Any],
    body: dict[str, Any],
    *,
    budget: _PreviewEncodedSizeBudget | None = None,
    materialize: bool = True,
) -> dict[str, Any] | None:
    """Run the shared timing algorithm, optionally without output-row allocation."""
    catalog = evidence_value["families"]["catalog_facts"]
    warnings: list[dict[str, Any]] = []
    duration = _duration(evidence_value)

    tempo = _tempo_claim(evidence_value, body, catalog, warnings, budget=budget)
    if budget is not None:
        budget.replace("tempo", tempo)
    _preflight_timing_caps(evidence_value, body, tempo)
    downbeats = _downbeats(
        evidence_value,
        body["downbeat"],
        no_pulse=tempo["status"] == "no_pulse",
        budget=budget,
        materialize=materialize,
    )
    beats = _beats(
        evidence_value,
        body,
        tempo,
        downbeats,
        duration,
        warnings,
        budget=budget,
        materialize=materialize,
    )
    meter = _meter_claim(evidence_value, body, catalog)
    if budget is not None:
        budget.replace("meter", meter)
    bars, bar_math_withheld = _bars(
        beats,
        downbeats,
        meter,
        duration,
        tempo["beat_unit"],
        bar_math_suppressed=tempo["status"] == "no_pulse",
        budget=budget,
        materialize=materialize,
    )

    # Bar math is withheld when a bar-capable meter cannot justify a grouping
    # because the effective pulse/phase is missing or the meter math is
    # incompatible. Explicit no_pulse and meter none remain intentional
    # suppressions. The frozen contract names the two references map:tempo and
    # map:meter; source_refs are stored sorted (map:meter < map:tempo), which
    # the sibling validator accepts. The warning preserves the real map state
    # (that bar math was intentionally withheld) rather than silently dropping it.
    if bar_math_withheld:
        _warning(warnings, "bar_math_withheld", _refs("map:tempo", "map:meter"), budget=budget)
    if evidence_value["migration"] is not None and evidence_value["migration"]["loss_flags"]:
        _warning(warnings, "legacy_evidence_loss", ["evidence:migration.loss_flags"], budget=budget)
    if not _catalog_present(catalog):
        _warning(warnings, "catalog_snapshot_absent", ["catalog:catalog_snapshot_hash"], budget=budget)

    if not materialize:
        return None
    assert isinstance(bars, list)
    return {
        "tempo": tempo,
        "beats": beats,
        "meter": meter,
        "downbeats": downbeats,
        "bars": bars,
        "warnings": sorted(warnings, key=lambda row: (_WARNING_ORDER[row["code"]], _canonical_json_bytes(row["source_refs"]))),
    }


def preview_music_map(evidence: object, interpretation: object) -> dict[str, Any]:
    """Project an Evidence-bound, non-persisted authoritative timing Preview."""
    evidence_value = _validate_evidence(evidence)
    duration = _duration(evidence_value)
    body = _validate_interpretation_against_evidence(
        interpretation,
        evidence_value,
        evidence_value["track_sha256"],
    )
    body_bytes = _canonical_json_bytes(body)
    if len(body_bytes) > _INTERPRETATION_BYTES:
        raise _MusicContractError(
            f"Interpretation body exceeds {_INTERPRETATION_BYTES} canonical bytes"
        )
    draft_fingerprint = _sha(body)
    preview_floor = {
        "schema": "studio.music-map-preview/1",
        "track_sha256": evidence_value["track_sha256"],
        "evidence_id": evidence_value["evidence_id"],
        "draft_fingerprint": draft_fingerprint,
        "resolver": copy.deepcopy(_RESOLVER),
        "duration_seconds": duration,
        "tempo": {},
        "beats": [],
        "meter": {},
        "downbeats": [],
        "bars": [],
        "warnings": [],
    }
    if len(_canonical_json_bytes(preview_floor)) > _MAP_BYTES:
        raise _MusicContractError(
            f"Preview artifact exceeds {_MAP_BYTES} canonical bytes"
        )
    budget = _PreviewEncodedSizeBudget(preview_floor, _MAP_BYTES)
    _build_timing_projection(evidence_value, body, budget=budget, materialize=False)
    timing = _build_timing_projection(evidence_value, body, materialize=True)
    assert timing is not None
    preview: dict[str, Any] = {
        "schema": "studio.music-map-preview/1",
        "track_sha256": evidence_value["track_sha256"],
        "evidence_id": evidence_value["evidence_id"],
        "draft_fingerprint": draft_fingerprint,
        "resolver": copy.deepcopy(_RESOLVER),
        "duration_seconds": duration,
        **timing,
    }
    preview_bytes = _canonical_json_bytes(preview)
    if len(preview_bytes) > _MAP_BYTES:
        raise _MusicContractError(
            f"Preview artifact exceeds {_MAP_BYTES} canonical bytes"
        )
    return preview


def resolve_music_map(evidence: object, interpretation: object) -> dict[str, Any]:
    """Resolve validated Evidence and Interpretation into a content-bound map."""
    evidence_value = _validate_evidence(evidence)
    duration = _duration(evidence_value)
    body, interpretation_projection = _interpretation(interpretation, duration)
    timing = _resolve_timing_projection(evidence_value, body)
    catalog = evidence_value["families"]["catalog_facts"]

    sections = _sections(evidence_value, body, catalog, duration)
    accents_family = evidence_value["families"]["accents"]
    accents = [] if accents_family["status"] == "unavailable" else _copy_evidence_events(accents_family["events"], "accent", accents=True)
    energy_family = evidence_value["families"]["energy"]
    energy = {
        "requested_sample_rate_hz": energy_family["requested_sample_rate_hz"],
        "hop_samples": energy_family["hop_samples"],
        "estimated_effective_sample_rate_hz": energy_family["estimated_effective_sample_rate_hz"],
        "samples": copy.deepcopy(energy_family["samples"]),
        "source_refs": ["evidence:energy"] if energy_family["status"] == "available" else [],
    }

    result: dict[str, Any] = {
        "schema": "studio.music-map/1",
        "track_sha256": evidence_value["track_sha256"],
        "map_id": "",
        "evidence_id": evidence_value["evidence_id"],
        "interpretation": interpretation_projection,
        "resolver": copy.deepcopy(_RESOLVER),
        "duration_seconds": evidence_value["analyzer"]["decoded_audio"]["duration_seconds"],
        "tempo": timing["tempo"],
        "beats": timing["beats"],
        "meter": timing["meter"],
        "downbeats": timing["downbeats"],
        "bars": timing["bars"],
        "sections": sections,
        "accents": accents,
        "energy": energy,
        "lyrics": None,
        "warnings": timing["warnings"],
    }
    result["map_id"] = _compute_map_id(result)
    return _validate_music_map(result)
