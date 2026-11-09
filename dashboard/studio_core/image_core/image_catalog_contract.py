from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    _image_contract = importlib.import_module("film_suite_generation.image_contract")
except ImportError:  # pragma: no cover - direct isolated module loading
    _image_contract = None
    for _name, _module in tuple(sys.modules.items()):
        if _name.endswith(".image_contract") and hasattr(_module, "validate_image_state"):
            _image_contract = _module
            break
    if _image_contract is None:
        _path = Path(__file__).parent.parent.parent / "generation" / "image_contract.py"
        _spec = importlib.util.spec_from_file_location("_image_source_catalog_contract", _path)
        if _spec is None or _spec.loader is None:
            raise ImportError("image contract unavailable")
        _image_contract = importlib.util.module_from_spec(_spec)
        sys.modules[_spec.name] = _image_contract
        _spec.loader.exec_module(_image_contract)


CATALOG_REQUEST_SCHEMA = "studio.image-source-catalog-request/1"
CATALOG_SCHEMA = "studio.image-source-catalog/1"
CATALOG_KIND = "image_source_catalog"
MAX_CONTEXTS = 256
MAX_ENTRIES = 2_048
MAX_RESPONSE_BYTES = 1 * 1024 * 1024
MAX_INTENT_UTF8_BYTES = 240

_CONTEXT_RE = re.compile(r"^imgctx_[0-9a-f]{32}$")
_WORK_RE = re.compile(r"^imgwork_[0-9a-f]{32}$")
_REVISION_RE = re.compile(r"^imgrev_[0-9a-f]{32}$")
_MATERIAL_RE = re.compile(r"^imgmat_[0-9a-f]{32}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TIMESTAMP_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")


class ImageCatalogError(ValueError):
    """Finite failure for the profile Image source catalog boundary."""

    CODES = frozenset(
        {
            "image_catalog_state_unavailable",
            "image_catalog_identity_changed",
            "image_catalog_limit_exceeded",
            "image_catalog_response_too_large",
        }
    )

    def __init__(self, code: str) -> None:
        if type(code) is not str or code not in self.CODES:
            code = "image_catalog_state_unavailable"
        self.code = code
        self.reason_code = code
        super().__init__(code)


def _fail(code: str) -> Any:
    raise ImageCatalogError(code) from None


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError) as exc:
        raise ImageCatalogError("image_catalog_state_unavailable") from exc


def validate_catalog_request(value: Any) -> dict[str, str]:
    if (
        type(value) is not dict
        or set(value) != {"schema", "kind"}
        or value["schema"] != CATALOG_REQUEST_SCHEMA
        or value["kind"] != CATALOG_KIND
    ):
        raise ImageCatalogError("image_catalog_state_unavailable")
    return {"schema": value["schema"], "kind": value["kind"]}


def _strict_id(value: Any, pattern: re.Pattern[str]) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        _fail("image_catalog_state_unavailable")
    return value


def _timestamp(value: Any) -> str:
    if type(value) is not str or _TIMESTAMP_RE.fullmatch(value) is None:
        _fail("image_catalog_state_unavailable")
    return value


def _intent_projection(intent: Any) -> tuple[str, bool]:
    if type(intent) is not str:
        _fail("image_catalog_state_unavailable")
    try:
        encoded = intent.encode("utf-8")
    except UnicodeError:
        _fail("image_catalog_state_unavailable")
    if len(encoded) <= MAX_INTENT_UTF8_BYTES:
        return intent, False
    output: list[str] = []
    used = 0
    for character in intent:
        encoded_character = character.encode("utf-8")
        if used + len(encoded_character) > MAX_INTENT_UTF8_BYTES:
            break
        output.append(character)
        used += len(encoded_character)
    return "".join(output), True


def _material_closure(revision: dict[str, Any]) -> set[str]:
    content = revision["content"]
    closure: set[str] = set()
    for reference in content["references"]:
        closure.add(_strict_id(reference["material_id"], _MATERIAL_RE))
    for alternative in content["alternatives"]:
        closure.add(_strict_id(alternative["material_id"], _MATERIAL_RE))
    selected = revision["selected_artifact"]
    if selected is not None:
        closure.add(_strict_id(selected, _MATERIAL_RE))
    material_ids = revision["material_ids"]
    if (
        type(material_ids) is not list
        or len(material_ids) != len(set(material_ids))
        or any(type(value) is not str or _MATERIAL_RE.fullmatch(value) is None for value in material_ids)
        or set(material_ids) != closure
    ):
        _fail("image_catalog_state_unavailable")
    return closure


def _selected_origin(revision: dict[str, Any]) -> str | None:
    selected = revision["selected_artifact"]
    if selected is None:
        return None
    matches = [
        alternative
        for alternative in revision["content"]["alternatives"]
        if alternative.get("material_id") == selected
    ]
    if len(matches) != 1:
        _fail("image_catalog_state_unavailable")
    alternative = matches[0]
    if alternative.get("origin") == "imported":
        try:
            _image_contract.validate_imported_artifact_ref(alternative)
        except Exception as exc:
            raise ImageCatalogError("image_catalog_state_unavailable") from exc
        return "imported"
    if alternative.get("origin") == "generated":
        try:
            _image_contract.validate_artifact_ref(alternative)
        except Exception as exc:
            raise ImageCatalogError("image_catalog_state_unavailable") from exc
        return "generated"
    _fail("image_catalog_state_unavailable")


def _project_entry(state: dict[str, Any], work: dict[str, Any], revision: dict[str, Any]) -> dict[str, Any]:
    context_id = _strict_id(state.get("context_id"), _CONTEXT_RE)
    work_id = _strict_id(work.get("work_id"), _WORK_RE)
    revision_id = _strict_id(revision.get("revision_id"), _REVISION_RE)
    if revision.get("work_id") != work_id:
        _fail("image_catalog_state_unavailable")
    context_revision = state.get("revision")
    if type(context_revision) is not int or isinstance(context_revision, bool) or context_revision < 1:
        _fail("image_catalog_state_unavailable")
    updated_at = _timestamp(state.get("updated_at"))
    saved_at = _timestamp(revision.get("saved_at"))
    origin = _selected_origin(revision)
    _material_closure(revision)

    materials = state["state"]["materials"]
    by_id: dict[str, dict[str, Any]] = {}
    for material in materials:
        material_id = _strict_id(material.get("material_id"), _MATERIAL_RE)
        if material_id in by_id:
            _fail("image_catalog_state_unavailable")
        by_id[material_id] = material

    selected = revision["selected_artifact"]
    material_projection: dict[str, Any] | None = None
    if selected is not None:
        material = by_id.get(selected)
        if material is not None:
            if (
                material.get("media_type") != "image"
                or type(material.get("label")) is not str
                or type(material.get("sha256")) is not str
                or _SHA256_RE.fullmatch(material["sha256"]) is None
                or material.get("availability") not in {"available", "missing", "denied"}
            ):
                _fail("image_catalog_state_unavailable")
            if material["availability"] == "available":
                material_projection = {
                    "material_id": selected,
                    "label": material["label"],
                    "sha256": material["sha256"],
                    "media_type": material["media_type"],
                    "availability": material["availability"],
                }

    intent_summary, intent_truncated = _intent_projection(revision["content"].get("intent"))
    return {
        "context_id": context_id,
        "context_revision": context_revision,
        "updated_at": updated_at,
        "work_id": work_id,
        "work_name": work.get("name"),
        "revision_id": revision_id,
        "saved_at": saved_at,
        "intent_summary": intent_summary,
        "intent_truncated": intent_truncated,
        "selected_artifact": selected,
        "material": material_projection,
        "origin": origin,
        "allowed_action": "select" if material_projection is not None else "unavailable",
    }


def _state_entries(state: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        normalized = _image_contract.validate_image_state(state)
    except Exception as exc:
        raise ImageCatalogError("image_catalog_state_unavailable") from exc
    works = normalized["state"]["works"]
    work_ids: set[str] = set()
    entries: list[dict[str, Any]] = []
    for work in works:
        work_id = _strict_id(work.get("work_id"), _WORK_RE)
        if work_id in work_ids:
            _fail("image_catalog_state_unavailable")
        work_ids.add(work_id)
        revision_ids: set[str] = set()
        for revision in work["saved_revisions"]:
            revision_id = _strict_id(revision.get("revision_id"), _REVISION_RE)
            if revision_id in revision_ids:
                _fail("image_catalog_state_unavailable")
            revision_ids.add(revision_id)
            entries.append(_project_entry(normalized, work, revision))
    return entries


def validate_catalog_response(value: Any) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {"schema", "entries"} or value.get("schema") != CATALOG_SCHEMA:
        _fail("image_catalog_state_unavailable")
    entries = value["entries"]
    if type(entries) is not list or len(entries) > MAX_ENTRIES:
        _fail("image_catalog_limit_exceeded")
    required_entry = {
        "context_id", "context_revision", "updated_at", "work_id", "work_name",
        "revision_id", "saved_at", "intent_summary", "intent_truncated",
        "selected_artifact", "material", "origin", "allowed_action",
    }
    required_material = {"material_id", "label", "sha256", "media_type", "availability"}
    for entry in entries:
        if type(entry) is not dict or set(entry) != required_entry:
            _fail("image_catalog_state_unavailable")
        _strict_id(entry["context_id"], _CONTEXT_RE)
        _strict_id(entry["work_id"], _WORK_RE)
        _strict_id(entry["revision_id"], _REVISION_RE)
        if type(entry["context_revision"]) is not int or isinstance(entry["context_revision"], bool) or entry["context_revision"] < 1:
            _fail("image_catalog_state_unavailable")
        _timestamp(entry["updated_at"])
        _timestamp(entry["saved_at"])
        if type(entry["work_name"]) is not str:
            _fail("image_catalog_state_unavailable")
        if type(entry["intent_summary"]) is not str or len(entry["intent_summary"].encode("utf-8")) > MAX_INTENT_UTF8_BYTES:
            _fail("image_catalog_state_unavailable")
        if type(entry["intent_truncated"]) is not bool:
            _fail("image_catalog_state_unavailable")
        if entry["selected_artifact"] is not None:
            _strict_id(entry["selected_artifact"], _MATERIAL_RE)
        if entry["origin"] is not None and entry["origin"] not in {"imported", "generated"}:
            _fail("image_catalog_state_unavailable")
        if entry["allowed_action"] not in {"select", "unavailable"}:
            _fail("image_catalog_state_unavailable")
        material = entry["material"]
        if material is not None:
            if (
                type(material) is not dict
                or set(material) != required_material
                or entry["selected_artifact"] is None
                or material.get("material_id") != entry["selected_artifact"]
                or entry["allowed_action"] != "select"
            ):
                _fail("image_catalog_state_unavailable")
            _strict_id(material["material_id"], _MATERIAL_RE)
            if (
                type(material["label"]) is not str
                or type(material["sha256"]) is not str
                or _SHA256_RE.fullmatch(material["sha256"]) is None
            ):
                _fail("image_catalog_state_unavailable")
            if material["media_type"] != "image" or material["availability"] != "available":
                _fail("image_catalog_state_unavailable")
        elif entry["allowed_action"] != "unavailable":
            _fail("image_catalog_state_unavailable")
        if entry["selected_artifact"] is None and entry["origin"] is not None:
            _fail("image_catalog_state_unavailable")
        if entry["selected_artifact"] is not None and entry["origin"] not in {"imported", "generated"}:
            _fail("image_catalog_state_unavailable")
    if len(canonical_bytes(value)) > MAX_RESPONSE_BYTES:
        _fail("image_catalog_response_too_large")
    return copy.deepcopy(value)


def build_catalog(states: list[dict[str, Any]]) -> dict[str, Any]:
    if type(states) is not list or len(states) > MAX_CONTEXTS:
        raise ImageCatalogError("image_catalog_limit_exceeded")
    context_ids: set[str] = set()
    entries: list[dict[str, Any]] = []
    for state in states:
        if type(state) is not dict:
            raise ImageCatalogError("image_catalog_state_unavailable")
        context_id = _strict_id(state.get("context_id"), _CONTEXT_RE)
        if context_id in context_ids:
            raise ImageCatalogError("image_catalog_state_unavailable")
        context_ids.add(context_id)
        entries.extend(_state_entries(state))
        if len(entries) > MAX_ENTRIES:
            raise ImageCatalogError("image_catalog_limit_exceeded")
    # Stable sorting keeps the explicit ID tie-breaker ascending while the
    # second pass orders the canonical UTC-second timestamp descending.
    entries.sort(key=lambda item: (item["context_id"], item["work_id"], item["revision_id"]))
    entries.sort(key=lambda item: item["saved_at"], reverse=True)
    result = {"schema": CATALOG_SCHEMA, "entries": entries}
    validate_catalog_response(result)
    if len(canonical_bytes(result)) > MAX_RESPONSE_BYTES:
        raise ImageCatalogError("image_catalog_response_too_large")
    return copy.deepcopy(result)


__all__ = [
    "CATALOG_KIND",
    "CATALOG_REQUEST_SCHEMA",
    "CATALOG_SCHEMA",
    "ImageCatalogError",
    "MAX_CONTEXTS",
    "MAX_ENTRIES",
    "MAX_INTENT_UTF8_BYTES",
    "MAX_RESPONSE_BYTES",
    "build_catalog",
    "canonical_bytes",
    "validate_catalog_request",
    "validate_catalog_response",
]
