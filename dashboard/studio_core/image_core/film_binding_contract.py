"""Pure validation and compatibility projection for Film Image bindings.

This module is deliberately limited to the persisted binding envelope.  It has
no store, filesystem, route, preview, adoption, provider, or runtime effects.
The generated alternative is validated by the accepted U4b producer contract;
this module only validates binding-local cross references and closure.
"""
from __future__ import annotations

import copy
import importlib
import importlib.util
import math
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable


BINDINGS_V1_SCHEMA = "studio.image-bindings/1"
BINDINGS_V2_SCHEMA = "studio.image-bindings/2"
IMAGE_BINDING_INSPECT_SCHEMA = "studio.image-binding-inspect/1"
MAX_SAFE_INTEGER = 9_007_199_254_740_991
MAX_BINDING_BYTES = 8 * 1024 * 1024
# A binding inspect serializes the complete binding plus repeated active-adoption
# and snapshot subtrees.  Each subtree is bounded by the persisted state cap;
# the fourth allowance covers the closed response envelope.  Keep this response
# cap separate so MAX_BINDING_BYTES remains the persisted-state admission bound.
MAX_INSPECT_RESPONSE_BYTES = 4 * MAX_BINDING_BYTES
MAX_DEPTH = 16
MAX_BINDINGS = 128
MAX_ADOPTIONS_PER_BINDING = 256
MAX_RECEIPTS = 128
MAX_SNAPSHOT_MATERIALS = 128
MAX_BINDING_GRAPH_NODES = 250_000

_PROJECT_RE = re.compile(r"^prj_[0-9a-f]{32}$")
_CONTEXT_RE = re.compile(r"^imgctx_[0-9a-f]{32}$")
_WORK_RE = re.compile(r"^imgwork_[0-9a-f]{32}$")
_REFERENCE_RE = re.compile(r"^imgref_[0-9a-f]{32}$")
_MATERIAL_RE = re.compile(r"^imgmat_[0-9a-f]{32}$")
_REVISION_RE = re.compile(r"^imgrev_[0-9a-f]{32}$")
_OPERATION_RE = re.compile(r"^imgop_[0-9a-f]{32}$")
_BINDING_RE = re.compile(r"^imgbind_[0-9a-f]{32}$")
_ADOPTION_RE = re.compile(r"^imgadopt_[0-9a-f]{32}$")
_FILM_REFERENCE_RE = re.compile(r"^filmref_[0-9a-f]{32}$")
_ENROLLMENT_RE = re.compile(r"^imgkey_[0-9a-f]{32}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_TIMESTAMP_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")


def _load_producer_contract() -> Any:
    accepted_path = (Path(__file__).resolve().parents[2] / "generation" / "image_contract.py").resolve()
    try:
        module = importlib.import_module("film_suite_generation.image_contract")
    except ModuleNotFoundError as exc:
        if exc.name not in {"film_suite_generation", "film_suite_generation.image_contract"}:
            raise
    else:
        module_path = getattr(module, "__file__", None)
        if module_path is not None and Path(module_path).resolve() == accepted_path:
            return module

    spec = importlib.util.spec_from_file_location("_u5_image_producer_contract", accepted_path)
    if spec is None or spec.loader is None or spec.origin is None:
        raise ImportError("accepted image producer contract unavailable")
    if Path(spec.origin).resolve() != accepted_path:
        raise ImportError("accepted image producer contract path mismatch")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


_producer_contract = _load_producer_contract()


class FilmBindingContractError(ValueError):
    """Raised when a persisted Film binding envelope is not contract-valid."""

    def __init__(self, code: str) -> None:
        self.code = code
        self.reason_code = code
        super().__init__(code)


BindingContractError = FilmBindingContractError
ContractError = FilmBindingContractError


def _fail(code: str) -> None:
    raise FilmBindingContractError(code)


def _strict_int(value: Any, name: str, low: int, high: int) -> int:
    if type(value) is not int or value < low or value > high:
        _fail(f"{name}-invalid")
    return value


def _strict_str(value: Any, name: str, low: int, high: int) -> str:
    if type(value) is not str or not low <= len(value) <= high:
        _fail(f"{name}-invalid")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise FilmBindingContractError(f"{name}-invalid") from exc
    return value


def _closed(value: Any, required: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != required:
        _fail(f"{name}-shape")
    return value


def _id(value: Any, pattern: re.Pattern[str], name: str) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        _fail(f"{name}-invalid")
    return value


def _hex64(value: Any, name: str) -> str:
    return _id(value, _HEX64_RE, name)


def _timestamp(value: Any, name: str) -> str:
    return _id(value, _TIMESTAMP_RE, name)


def _preflight_json(value: Any) -> None:
    """Reject cycles, non-JSON objects, non-finite numbers, and deep graphs.

    The walk is iterative so malformed caller objects cannot exhaust Python's
    recursion limit before the closed contract validator gets a chance to reject
    them.  Shared (but acyclic) containers remain valid JSON values.
    """
    stack: list[tuple[Iterator[Any], int, frozenset[int]]] = []
    nodes = 0

    def visit(current: Any, depth: int, ancestors: frozenset[int]) -> None:
        nonlocal nodes
        if depth > MAX_DEPTH:
            _fail("binding-depth-exceeded")
        if type(current) is dict:
            marker = id(current)
            if marker in ancestors:
                _fail("binding-cycle")
            nodes += 1
            if nodes + child_entries > MAX_BINDING_GRAPH_NODES:
                _fail("binding-node-count-exceeded")
            next_ancestors = ancestors | {marker}
            def children() -> Iterator[Any]:
                for key, child in current.items():
                    if type(key) is not str:
                        _fail("binding-non-json-key")
                    try:
                        key.encode("utf-8")
                    except UnicodeError as exc:
                        raise FilmBindingContractError("binding-non-json-key") from exc
                    yield child

            stack.append((children(), depth + 1, next_ancestors))
        elif type(current) is list:
            marker = id(current)
            if marker in ancestors:
                _fail("binding-cycle")
            nodes += 1
            if nodes + child_entries > MAX_BINDING_GRAPH_NODES:
                _fail("binding-node-count-exceeded")
            next_ancestors = ancestors | {marker}
            stack.append((iter(current), depth + 1, next_ancestors))
        elif type(current) is str:
            try:
                if len(current.encode("utf-8")) > MAX_BINDING_BYTES:
                    _fail("binding-size-exceeded")
            except UnicodeError as exc:
                raise FilmBindingContractError("binding-string-invalid") from exc
        elif type(current) is float:
            if not math.isfinite(current):
                _fail("binding-non-finite-number")
        elif type(current) in (int, bool) or current is None:
            return
        else:
            _fail("binding-non-json-value")

    child_entries = 0
    visit(value, 1, frozenset())
    while stack:
        children, child_depth, ancestors = stack[-1]
        try:
            child = next(children)
        except StopIteration:
            stack.pop()
            continue
        child_entries += 1
        if nodes + child_entries > MAX_BINDING_GRAPH_NODES:
            _fail("binding-node-count-exceeded")
        visit(child, child_depth, ancestors)


def _canonical_size(
    value: dict[str, Any], *, max_bytes: int = MAX_BINDING_BYTES, error_code: str = "binding-size-exceeded"
) -> None:
    try:
        raw = _producer_contract.canonical_bytes(value)
    except Exception as exc:
        raise FilmBindingContractError("binding-non-canonical-json") from exc
    if len(raw) > max_bytes:
        _fail(error_code)


def _validate_project(value: Any, name: str = "project-id") -> str:
    return _id(value, _PROJECT_RE, name)


def _validate_artifact(value: Any, *, allow_generated: bool) -> str:
    if type(value) is not dict:
        _fail("artifact-shape")
    origin = value.get("origin")
    try:
        if origin == "generated":
            if not allow_generated:
                _fail("generated-artifact-not-allowed")
            _producer_contract.validate_artifact_ref(value)
            return value["material_id"]
        _producer_contract.validate_imported_artifact_ref(value)
        return value["material_id"]
    except FilmBindingContractError:
        raise
    except Exception as exc:
        # Preserve the accepted canonical producer validator's stable reason
        # (not a second, weaker generated-artifact implementation).
        raise FilmBindingContractError(str(exc) or "artifact-invalid") from exc


def _validate_reference(value: Any) -> tuple[str, str]:
    required = {"reference_id", "material_id", "role", "instruction", "preserve"}
    _closed(value, required, "reference")
    reference_id = _id(value["reference_id"], _REFERENCE_RE, "reference-id")
    material_id = _id(value["material_id"], _MATERIAL_RE, "reference-material-id")
    _strict_str(value["role"], "reference-role", 0, 80)
    _strict_str(value["instruction"], "reference-instruction", 0, 2048)
    _strict_str(value["preserve"], "reference-preserve", 0, 2048)
    return reference_id, material_id


def _validate_content(value: Any, *, allow_generated: bool) -> tuple[set[str], dict[str, dict[str, Any]]]:
    required = {"intent", "intent_sha256", "notes", "references", "alternatives"}
    _closed(value, required, "revision-content")
    intent = _strict_str(value["intent"], "intent", 0, 16_384)
    _hex64(value["intent_sha256"], "intent-sha256")
    try:
        expected_hash = _producer_contract.sha256(intent.encode("utf-8"))
    except Exception as exc:
        raise FilmBindingContractError("intent-sha256") from exc
    if value["intent_sha256"] != expected_hash:
        _fail("intent-sha256")
    _strict_str(value["notes"], "notes", 0, 16_384)

    references = value["references"]
    if type(references) is not list or len(references) > 32:
        _fail("references-shape")
    reference_ids: set[str] = set()
    closure: set[str] = set()
    for reference in references:
        reference_id, material_id = _validate_reference(reference)
        if reference_id in reference_ids:
            _fail("duplicate-reference-id")
        reference_ids.add(reference_id)
        closure.add(material_id)

    alternatives = value["alternatives"]
    if type(alternatives) is not list or len(alternatives) > 128:
        _fail("alternatives-shape")
    alternative_material_ids: set[str] = set()
    artifacts: dict[str, dict[str, Any]] = {}
    for alternative in alternatives:
        material_id = _validate_artifact(alternative, allow_generated=allow_generated)
        if material_id in alternative_material_ids:
            _fail("duplicate-alternative-material-id")
        alternative_material_ids.add(material_id)
        closure.add(material_id)
        if alternative.get("origin") == "generated":
            artifacts[material_id] = alternative

    return closure, artifacts


def _validate_saved_revision(
    value: Any,
    *,
    allow_generated: bool,
    source_work_id: str | None = None,
    source_revision_id: str | None = None,
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    required = {
        "revision_id",
        "work_id",
        "parent_revision_id",
        "saved_at",
        "content",
        "selected_artifact",
        "material_ids",
    }
    _closed(value, required, "saved-revision")
    revision_id = _id(value["revision_id"], _REVISION_RE, "revision-id")
    work_id = _id(value["work_id"], _WORK_RE, "revision-work-id")
    if source_work_id is not None and work_id != source_work_id:
        _fail("source-work-revision-mismatch")
    if source_revision_id is not None and revision_id != source_revision_id:
        _fail("source-id-revision-mismatch")
    if value["parent_revision_id"] is not None:
        _id(value["parent_revision_id"], _REVISION_RE, "parent-revision-id")
    _timestamp(value["saved_at"], "saved-at")
    closure, artifacts = _validate_content(value["content"], allow_generated=allow_generated)
    selected = value["selected_artifact"]
    if selected is not None:
        _id(selected, _MATERIAL_RE, "selected-artifact")
        closure.add(selected)
    material_ids = value["material_ids"]
    if type(material_ids) is not list or len(material_ids) > MAX_SNAPSHOT_MATERIALS:
        _fail("material-ids-shape")
    if any(type(material_id) is not str for material_id in material_ids):
        _fail("material-ids-shape")
    for material_id in material_ids:
        _id(material_id, _MATERIAL_RE, "material-id")
    if len(set(material_ids)) != len(material_ids):
        _fail("material-ids-duplicate")
    if material_ids != sorted(material_ids):
        _fail("material-ids-unsorted")
    if set(material_ids) != closure:
        _fail("material-closure-mismatch")
    return closure, artifacts


def _validate_material(value: Any) -> str:
    required = {"material_id", "sha256", "media_type", "label", "availability"}
    _closed(value, required, "snapshot-material")
    material_id = _id(value["material_id"], _MATERIAL_RE, "snapshot-material-id")
    _hex64(value["sha256"], "snapshot-material-sha256")
    if value["media_type"] != "image":
        _fail("snapshot-material-media-type")
    _strict_str(value["label"], "snapshot-material-label", 0, 260)
    if type(value["availability"]) is not str or value["availability"] not in {
        "available",
        "missing",
        "denied",
    }:
        _fail("snapshot-material-availability")
    return material_id


def _validate_snapshot(value: Any, *, allow_generated: bool, source_work_id: str, source_revision_id: str) -> None:
    _closed(value, {"revision", "materials"}, "snapshot")
    closure, artifacts = _validate_saved_revision(
        value["revision"],
        allow_generated=allow_generated,
        source_work_id=source_work_id,
        source_revision_id=source_revision_id,
    )
    materials = value["materials"]
    if type(materials) is not list or len(materials) > MAX_SNAPSHOT_MATERIALS:
        _fail("snapshot-materials-shape")
    material_by_id: dict[str, dict[str, Any]] = {}
    for material in materials:
        material_id = _validate_material(material)
        if material_id in material_by_id:
            _fail("duplicate-snapshot-material-id")
        material_by_id[material_id] = material
    if list(material_by_id) != sorted(material_by_id):
        _fail("snapshot-materials-unsorted")
    if set(material_by_id) != closure:
        _fail("snapshot-material-closure-mismatch")
    for material_id, artifact in artifacts.items():
        # This is a structural cross-reference only.  It does not authenticate
        # arbitrary hashes against saved source bytes.
        if material_by_id[material_id]["sha256"] != artifact["artifact_sha256"]:
            _fail("generated-material-hash-mismatch")


def _validate_receipt(
    value: Any,
    *,
    expected_project_id: str,
    envelope_revision: int,
    allow_new_reference: bool,
) -> tuple[str, str | None, str]:
    required = {
        "operation_id",
        "operation_kind",
        "issued_at",
        "expires_at",
        "payload_sha256",
        "expected_revision",
        "target_scope",
        "status",
        "committed_revision",
        "result",
    }
    _closed(value, required, "receipt")
    operation_id = _id(value["operation_id"], _OPERATION_RE, "operation-id")
    operation_kind = value["operation_kind"]
    allowed_kinds = {"film_image_adopt"}
    if allow_new_reference:
        allowed_kinds.add("film_image_adopt_new_reference")
    if type(operation_kind) is not str or operation_kind not in allowed_kinds:
        _fail("receipt-operation-kind")
    _timestamp(value["issued_at"], "receipt-issued-at")
    _timestamp(value["expires_at"], "receipt-expires-at")
    _hex64(value["payload_sha256"], "receipt-payload-sha256")
    minimum_expected_revision = 0 if operation_kind == "film_image_adopt_new_reference" else 1
    expected_revision = _strict_int(
        value["expected_revision"],
        "receipt-expected-revision",
        minimum_expected_revision,
        MAX_SAFE_INTEGER,
    )
    committed_revision = _strict_int(value["committed_revision"], "receipt-committed-revision", 1, MAX_SAFE_INTEGER)
    if committed_revision > envelope_revision:
        _fail("receipt-committed-revision-future")
    if committed_revision != expected_revision + 1:
        _fail("receipt-cas-mismatch")
    if value["status"] != "committed":
        _fail("receipt-status")
    scope = value["target_scope"]
    _closed(scope, {"scope_kind", "project_id", "binding_id", "film_reference_id"}, "receipt-target-scope")
    if scope["scope_kind"] != "film_binding":
        _fail("receipt-target-scope-kind")
    project_id = _validate_project(scope["project_id"], "receipt-project-id")
    if project_id != expected_project_id:
        _fail("receipt-project-id-mismatch")
    binding_id = _id(scope["binding_id"], _BINDING_RE, "receipt-binding-id")
    film_reference_id = _id(scope["film_reference_id"], _FILM_REFERENCE_RE, "receipt-film-reference-id")
    result = value["result"]
    _closed(result, {"context_id", "work_id", "revision_id", "material_id", "binding_id", "adoption_id"}, "receipt-result")
    for field, pattern in (
        ("context_id", _CONTEXT_RE),
        ("work_id", _WORK_RE),
        ("revision_id", _REVISION_RE),
        ("material_id", _MATERIAL_RE),
    ):
        if result[field] is not None:
            _id(result[field], pattern, f"receipt-result-{field}")
    _id(result["binding_id"], _BINDING_RE, "receipt-result-binding_id")
    _id(result["adoption_id"], _ADOPTION_RE, "receipt-result-adoption_id")
    return operation_id, binding_id, film_reference_id


def _validate_adoption(value: Any, *, allow_generated: bool) -> tuple[str, str, str, str, dict[str, Any]]:
    required = {
        "adoption_id",
        "source_context_id",
        "source_work_id",
        "source_revision_id",
        "snapshot",
        "adopted_at",
        "operation_id",
    }
    _closed(value, required, "adoption")
    adoption_id = _id(value["adoption_id"], _ADOPTION_RE, "adoption-id")
    context_id = _id(value["source_context_id"], _CONTEXT_RE, "source-context-id")
    work_id = _id(value["source_work_id"], _WORK_RE, "source-work-id")
    revision_id = _id(value["source_revision_id"], _REVISION_RE, "source-revision-id")
    _timestamp(value["adopted_at"], "adopted-at")
    operation_id = _id(value["operation_id"], _OPERATION_RE, "adoption-operation-id")
    _validate_snapshot(
        value["snapshot"],
        allow_generated=allow_generated,
        source_work_id=work_id,
        source_revision_id=revision_id,
    )
    embedded = value["snapshot"]["revision"]
    if embedded["work_id"] != work_id or embedded["revision_id"] != revision_id:
        _fail("source-id-revision-mismatch")
    return adoption_id, context_id, work_id, revision_id, value


def _validate_binding(value: Any, *, allow_generated: bool) -> tuple[str, str, dict[str, dict[str, Any]], dict[str, str]]:
    required = {"binding_id", "label", "film_reference_id", "active_adoption_id", "adoptions"}
    _closed(value, required, "binding")
    binding_id = _id(value["binding_id"], _BINDING_RE, "binding-id")
    _strict_str(value["label"], "binding-label", 1, 160)
    film_reference_id = _id(value["film_reference_id"], _FILM_REFERENCE_RE, "film-reference-id")
    active_adoption_id = _id(value["active_adoption_id"], _ADOPTION_RE, "active-adoption-id")
    adoptions = value["adoptions"]
    if type(adoptions) is not list or not adoptions or len(adoptions) > MAX_ADOPTIONS_PER_BINDING:
        _fail("adoptions-shape")
    adoption_by_id: dict[str, dict[str, Any]] = {}
    adoption_operation_ids: set[str] = set()
    adoption_to_operation: dict[str, str] = {}
    for adoption in adoptions:
        adoption_id, _context_id, _work_id, _revision_id, normalized = _validate_adoption(
            adoption, allow_generated=allow_generated
        )
        if adoption_id in adoption_by_id:
            _fail("duplicate-adoption-id")
        operation_id = normalized["operation_id"]
        if operation_id in adoption_operation_ids:
            _fail("duplicate-adoption-operation-id")
        adoption_operation_ids.add(operation_id)
        adoption_by_id[adoption_id] = normalized
        adoption_to_operation[adoption_id] = operation_id
    if active_adoption_id not in adoption_by_id:
        _fail("active-adoption-missing")
    return binding_id, film_reference_id, adoption_by_id, adoption_to_operation


def validate_image_bindings(bindings: dict[str, Any], expected_project_id: str) -> dict[str, Any]:
    """Validate a /1 or /2 envelope against a trusted project identity.

    The returned value is a deep copy.  Validation is structural and
    cross-reference based; valid-looking hashes are not treated as proof that
    the referenced source bytes were authenticated.
    """
    _preflight_json(bindings)
    _closed(
        bindings,
        {
            "schema",
            "project_id",
            "revision",
            "updated_at",
            "receipt_key_fingerprint",
            "receipt_key_enrollment_id",
            "state",
        },
        "bindings-envelope",
    )
    schema = bindings["schema"]
    if type(schema) is not str or schema not in {BINDINGS_V1_SCHEMA, BINDINGS_V2_SCHEMA}:
        _fail("bindings-schema")
    allow_generated = schema == BINDINGS_V2_SCHEMA
    trusted_project_id = _validate_project(expected_project_id, "expected-project-id")
    project_id = _validate_project(bindings["project_id"])
    if project_id != trusted_project_id:
        _fail("project-id-mismatch")
    revision = _strict_int(bindings["revision"], "bindings-revision", 1, MAX_SAFE_INTEGER)
    _timestamp(bindings["updated_at"], "bindings-updated-at")
    _hex64(bindings["receipt_key_fingerprint"], "receipt-key-fingerprint")
    _id(bindings["receipt_key_enrollment_id"], _ENROLLMENT_RE, "receipt-key-enrollment-id")

    state = _closed(bindings["state"], {"bindings", "receipts", "receipt_floor_at"}, "bindings-state")
    binding_values = state["bindings"]
    if type(binding_values) is not list or len(binding_values) > MAX_BINDINGS:
        _fail("bindings-list-shape")
    _timestamp(state["receipt_floor_at"], "receipt-floor-at")

    bindings_by_id: dict[str, dict[str, Any]] = {}
    film_refs: dict[str, str] = {}
    adoptions_by_id: dict[str, tuple[str, dict[str, Any]]] = {}
    adoption_operation_ids: set[str] = set()
    for binding in binding_values:
        binding_id, film_reference_id, adoption_by_id, operation_ids = _validate_binding(
            binding, allow_generated=allow_generated
        )
        if binding_id in bindings_by_id:
            _fail("duplicate-binding-id")
        if film_reference_id in film_refs:
            _fail("duplicate-film-reference-id")
        for operation_id in operation_ids.values():
            if operation_id in adoption_operation_ids:
                _fail("duplicate-adoption-operation-id")
            adoption_operation_ids.add(operation_id)
        bindings_by_id[binding_id] = binding
        film_refs[film_reference_id] = binding_id
        for adoption_id, adoption in adoption_by_id.items():
            if adoption_id in adoptions_by_id:
                _fail("duplicate-adoption-id")
            adoptions_by_id[adoption_id] = (binding_id, adoption)

    receipts = state["receipts"]
    if type(receipts) is not list or len(receipts) > MAX_RECEIPTS:
        _fail("receipts-shape")
    receipts_by_id: dict[str, dict[str, Any]] = {}
    for receipt in receipts:
        operation_id, binding_id, film_reference_id = _validate_receipt(
            receipt,
            expected_project_id=trusted_project_id,
            envelope_revision=revision,
            allow_new_reference=allow_generated,
        )
        if operation_id in receipts_by_id:
            _fail("duplicate-receipt-operation-id")
        receipts_by_id[operation_id] = receipt
        if binding_id is not None:
            if binding_id not in bindings_by_id:
                _fail("receipt-binding-missing")
            binding = bindings_by_id[binding_id]
            if binding["film_reference_id"] != film_reference_id:
                _fail("receipt-film-reference-mismatch")
        result = receipt["result"]
        result_binding_id = result["binding_id"]
        result_adoption_id = result["adoption_id"]
        if result_binding_id is not None:
            if result_binding_id not in bindings_by_id:
                _fail("receipt-result-binding-missing")
            if binding_id is not None and result_binding_id != binding_id:
                _fail("receipt-result-binding-mismatch")
        if result_adoption_id is not None:
            found = adoptions_by_id.get(result_adoption_id)
            if found is None:
                _fail("receipt-result-adoption-missing")
            found_binding_id, adoption = found
            if result_binding_id != found_binding_id:
                _fail("receipt-result-adoption-binding-mismatch")
            if (
                result["context_id"] != adoption["source_context_id"]
                or result["work_id"] != adoption["source_work_id"]
                or result["revision_id"] != adoption["source_revision_id"]
            ):
                _fail("receipt-result-source-mismatch")
            if operation_id != adoption["operation_id"]:
                _fail("receipt-result-operation-mismatch")

    _canonical_size(bindings)
    return copy.deepcopy(bindings)


def validate_image_bindings_v1(bindings: dict[str, Any], expected_project_id: str) -> dict[str, Any]:
    """Validate the historical imported-only /1 envelope without rewriting it."""
    result = validate_image_bindings(bindings, expected_project_id)
    if result["schema"] != BINDINGS_V1_SCHEMA:
        _fail("bindings-v1-required")
    return result


def validate_image_bindings_v2(bindings: dict[str, Any], expected_project_id: str) -> dict[str, Any]:
    """Validate the additive /2 envelope with imported/generated alternatives."""
    result = validate_image_bindings(bindings, expected_project_id)
    if result["schema"] != BINDINGS_V2_SCHEMA:
        _fail("bindings-v2-required")
    return result


def validate_image_binding_inspect_response(
    response: dict[str, Any], expected_project_id: str
) -> dict[str, Any]:
    """Validate the additive read-only Film binding inspection projection.

    This is intentionally separate from the historical
    ``film_image_inspect_result`` operation envelope.  The projection repeats
    the active adoption and immutable snapshot for consumers, so validation
    also proves those repeated values are identical to the selected Binding.
    The response version is independent of the persisted binding-state version;
    both imported and accepted generated artifacts are therefore supported.
    """
    _preflight_json(response)
    _closed(
        response,
        {
            "schema",
            "project_id",
            "bindings_revision",
            "binding",
            "active_adoption",
            "snapshot",
            "availability",
            "use_allowed",
        },
        "image-binding-inspect-response",
    )
    if response["schema"] != IMAGE_BINDING_INSPECT_SCHEMA:
        _fail("image-binding-inspect-schema")
    expected = _validate_project(expected_project_id, "expected-project-id")
    project_id = _validate_project(response["project_id"], "inspect-project-id")
    if project_id != expected:
        _fail("inspect-project-id-mismatch")
    _strict_int(response["bindings_revision"], "bindings-revision", 1, MAX_SAFE_INTEGER)
    if type(response["availability"]) is not str or response["availability"] not in {
        "available",
        "missing",
        "denied",
        "changed",
    }:
        _fail("inspect-availability")
    if type(response["use_allowed"]) is not bool:
        _fail("inspect-use-allowed")

    binding_id, film_reference_id, adoption_by_id, _operation_ids = _validate_binding(
        response["binding"], allow_generated=True
    )
    adoption_id, _context_id, source_work_id, source_revision_id, active_adoption = _validate_adoption(
        response["active_adoption"],
        allow_generated=True,
    )
    if response["binding"]["active_adoption_id"] != adoption_id:
        _fail("inspect-active-adoption-mismatch")
    stored_active = adoption_by_id.get(adoption_id)
    if stored_active is None or stored_active != active_adoption:
        _fail("inspect-active-adoption-mismatch")
    _validate_snapshot(
        response["snapshot"],
        allow_generated=True,
        source_work_id=source_work_id,
        source_revision_id=source_revision_id,
    )
    if response["snapshot"] != active_adoption["snapshot"]:
        _fail("inspect-snapshot-mismatch")
    if response["binding"]["binding_id"] != binding_id:
        _fail("inspect-binding-mismatch")
    if response["binding"]["film_reference_id"] != film_reference_id:
        _fail("inspect-film-reference-mismatch")
    _canonical_size(
        response,
        max_bytes=MAX_INSPECT_RESPONSE_BYTES,
        error_code="inspect-response-size-exceeded",
    )
    return copy.deepcopy(response)


def validate_image_binding_inspect_response_v1(
    response: dict[str, Any], expected_project_id: str
) -> dict[str, Any]:
    """Named entry point for callers that require this exact response version."""
    return validate_image_binding_inspect_response(response, expected_project_id)


def project_bindings_v1_to_v2(bindings: dict[str, Any], expected_project_id: str) -> dict[str, Any]:
    """Purely project a valid /1 envelope to /2, preserving every other field."""
    source = validate_image_bindings_v1(bindings, expected_project_id)
    projected = copy.deepcopy(source)
    projected["schema"] = BINDINGS_V2_SCHEMA
    return validate_image_bindings_v2(projected, expected_project_id)


# Friendly aliases for callers that use the shorter contract terminology.
validate_binding_state = validate_image_bindings
validate_bindings = validate_image_bindings
project_v1_to_v2 = project_bindings_v1_to_v2
upgrade_bindings_v1_to_v2 = project_bindings_v1_to_v2


__all__ = [
    "BINDINGS_V1_SCHEMA",
    "BINDINGS_V2_SCHEMA",
    "IMAGE_BINDING_INSPECT_SCHEMA",
    "MAX_SAFE_INTEGER",
    "MAX_BINDING_BYTES",
    "MAX_INSPECT_RESPONSE_BYTES",
    "MAX_DEPTH",
    "FilmBindingContractError",
    "BindingContractError",
    "ContractError",
    "validate_image_bindings",
    "validate_image_bindings_v1",
    "validate_image_bindings_v2",
    "validate_image_binding_inspect_response",
    "validate_image_binding_inspect_response_v1",
    "validate_binding_state",
    "validate_bindings",
    "project_bindings_v1_to_v2",
    "project_v1_to_v2",
    "upgrade_bindings_v1_to_v2",
]
