"""Deliberate new-reference Image-to-Film adoption over accepted stores."""
from __future__ import annotations

import copy
import re
from typing import Any

from .film_binding_contract import (
    BINDINGS_V1_SCHEMA,
    BINDINGS_V2_SCHEMA,
    FilmBindingContractError,
    project_bindings_v1_to_v2,
    validate_image_binding_inspect_response,
)
from .film_binding_store import FilmBindingStoreError
from .preview_authority import (
    ADOPT_KIND,
    BRANCH_VERSION,
    ERROR_KIND,
    ImageAdoptionError,
    PreviewAuthority,
    PREPARE_KIND,
    RECEIPT_KIND,
    STATUS_RESULT_KIND,
    V2_SCHEMA,
    _ADOPTION_RE,
    _BINDING_RE,
    _FILM_REFERENCE_RE,
    _OPERATION_RE,
    _validate_id,
)


class FilmAdoptionError(ImageAdoptionError):
    """Stable domain error alias for callers that name the adoption service."""


class FilmAdoptionService(PreviewAuthority):
    """Application service for one deliberate new Film reference.

    Preview and prepare use the inherited session authority and are read-only
    with respect to both accepted stores.  Only ``execute`` publishes one
    append-only FilmBindingStore revision.
    """

    def __init__(self, image_store: Any, binding_store: Any, **kwargs: Any) -> None:
        try:
            super().__init__(image_store, binding_store, **kwargs)
        except ImageAdoptionError as exc:
            raise FilmAdoptionError(exc.code) from exc
        self._adoption_lock = self._lock

    def _durable_receipt(self, operation_id: str) -> dict[str, Any] | None:
        """Recover a complete v2 wire receipt from Store evidence.

        The accepted persisted binding receipt intentionally has no wire-only
        owner/branch/intent fields.  Those fields are recovered from the
        immutable Adoption snapshot and Binding label, never from caller data.
        """
        _validate_id(operation_id, _OPERATION_RE, "operation-id-invalid")
        try:
            state, _temporal = self._film_state()
        except Exception as exc:
            raise ImageAdoptionError("image_state_unavailable") from exc
        if state is None:
            return None
        if type(state) is not dict:
            raise ImageAdoptionError("image_state_unavailable")
        try:
            if (
                state.get("receipt_key_enrollment_id") != self._film_owner_enrollment_id
                or state.get("receipt_key_fingerprint") != self._film_owner_fingerprint
            ):
                raise ValueError("binding owner evidence mismatch")
            state_revision = state["revision"]
            self._strict_int(state_revision, "film-revision", 1, 9_007_199_254_740_991)
            receipts = state["state"]["receipts"]
            bindings = state["state"]["bindings"]
            if type(receipts) is not list or type(bindings) is not list:
                raise ValueError("binding evidence shape")
        except (KeyError, TypeError, ValueError) as exc:
            raise ImageAdoptionError("image_state_unavailable") from exc
        matching = [item for item in receipts if type(item) is dict and item.get("operation_id") == operation_id]
        if not matching:
            return None
        if len(matching) != 1:
            raise ImageAdoptionError("image_state_unavailable")
        durable = copy.deepcopy(matching[0])
        try:
            result = durable["result"]
            binding_id = result["binding_id"]
            adoption_id = result["adoption_id"]
            binding_matches = [item for item in bindings if item.get("binding_id") == binding_id]
            if len(binding_matches) != 1:
                raise ValueError("binding evidence missing")
            binding = binding_matches[0]
            adoptions = [item for item in binding["adoptions"] if item.get("adoption_id") == adoption_id]
            if len(adoptions) != 1:
                raise ValueError("adoption evidence missing")
            adoption = adoptions[0]
            snapshot = copy.deepcopy(adoption["snapshot"])
            source_context_id = adoption["source_context_id"]
            source_work_id = adoption["source_work_id"]
            source_revision_id = adoption["source_revision_id"]
            snapshot_sha256 = self._digest(snapshot)
            intent = {
                "branch_version": BRANCH_VERSION,
                "target_scope": copy.deepcopy(durable["target_scope"]),
                "expected_revision": durable["expected_revision"],
                "label": binding["label"],
                "source_context_id": source_context_id,
                "source_work_id": source_work_id,
                "source_revision_id": source_revision_id,
                "snapshot_sha256": snapshot_sha256,
            }
            self._validate_intent(intent)
            self._strict_intent_revision(durable["expected_revision"])
            self._strict_int(durable["committed_revision"], "committed-revision", 1, 9_007_199_254_740_991)
            if durable["committed_revision"] != durable["expected_revision"] + 1:
                raise ValueError("receipt revision mismatch")
            if (
                result["context_id"] != source_context_id
                or result["work_id"] != source_work_id
                or result["revision_id"] != source_revision_id
                or result["material_id"] is not None
                or result["binding_id"] != binding_id
                or result["adoption_id"] != adoption_id
            ):
                raise ValueError("receipt result mismatch")
            if adoption["snapshot"] != snapshot or adoption["operation_id"] != operation_id:
                raise ValueError("adoption evidence mismatch")
            projection = {
                "schema": V2_SCHEMA,
                "branch_version": BRANCH_VERSION,
                "operation_kind": ADOPT_KIND,
                "owner_scope": self.owner_scope,
                "expected_revision": durable["expected_revision"],
                "target_scope": copy.deepcopy(durable["target_scope"]),
                "operation_intent": copy.deepcopy(intent),
                "payload_sha256": durable["payload_sha256"],
                "issued_at": durable["issued_at"],
            }
            if operation_id != self._operation_signature(projection):
                raise ValueError("operation identity mismatch")
            if durable["operation_kind"] != ADOPT_KIND:
                raise ValueError("wrong operation kind")
            if durable["target_scope"]["project_id"] != self.project_id:
                raise ValueError("wrong project")
            if durable["target_scope"]["binding_id"] != binding_id or durable["target_scope"]["film_reference_id"] != binding["film_reference_id"]:
                raise ValueError("target evidence mismatch")
            if adoption["operation_id"] != operation_id:
                raise ValueError("adoption operation mismatch")
            return {
                "schema": V2_SCHEMA,
                "kind": RECEIPT_KIND,
                "branch_version": BRANCH_VERSION,
                "owner_scope": self.owner_scope,
                "operation_id": durable["operation_id"],
                "operation_kind": ADOPT_KIND,
                "issued_at": durable["issued_at"],
                "expires_at": durable["expires_at"],
                "payload_sha256": durable["payload_sha256"],
                "expected_revision": durable["expected_revision"],
                "target_scope": copy.deepcopy(durable["target_scope"]),
                "operation_intent": intent,
                "status": "committed",
                "committed_revision": durable["committed_revision"],
                "result": copy.deepcopy(result),
            }
        except ImageAdoptionError:
            raise
        except Exception as exc:
            raise ImageAdoptionError("image_state_unavailable") from exc

    def _receipt_matches(self, request: dict[str, Any], receipt: dict[str, Any]) -> bool:
        try:
            return (
                request["schema"] == receipt["schema"]
                and request["branch_version"] == receipt["branch_version"]
                and request["owner_scope"] == receipt["owner_scope"]
                and request["operation_id"] == receipt["operation_id"]
                and request["issued_at"] == receipt["issued_at"]
                and request["expected_revision"] == receipt["expected_revision"]
                and request["target_scope"] == receipt["target_scope"]
                and request["operation_intent"] == receipt["operation_intent"]
                and self._digest(request["payload"]) == receipt["payload_sha256"]
                and request["context_id"] == receipt["operation_intent"]["source_context_id"]
            )
        except (KeyError, TypeError, ValueError, RecursionError, MemoryError):
            return False

    def _base_candidate(self, expected_revision: int, updated_at: str) -> dict[str, Any]:
        try:
            current, _temporal = self._film_state()
        except Exception as exc:
            raise ImageAdoptionError("image_state_unavailable") from exc
        if expected_revision == 0:
            if current is not None:
                raise ImageAdoptionError("image_conflict")
            return {
                "schema": BINDINGS_V2_SCHEMA,
                "project_id": self.project_id,
                "revision": 1,
                "updated_at": updated_at,
                "receipt_key_fingerprint": self._film_owner_fingerprint,
                "receipt_key_enrollment_id": self._film_owner_enrollment_id,
                "state": {"bindings": [], "receipts": [], "receipt_floor_at": updated_at},
            }
        if type(current) is not dict or current.get("revision") != expected_revision:
            raise ImageAdoptionError("image_conflict")
        if current.get("schema") == BINDINGS_V1_SCHEMA:
            try:
                candidate = project_bindings_v1_to_v2(current, self.project_id)
            except Exception as exc:
                raise ImageAdoptionError("image_state_unavailable") from exc
        elif current.get("schema") == BINDINGS_V2_SCHEMA:
            candidate = copy.deepcopy(current)
        else:
            raise ImageAdoptionError("image_state_unavailable")
        candidate["revision"] = expected_revision + 1
        candidate["updated_at"] = updated_at
        return candidate

    @staticmethod
    def _adoption_id(operation_id: str) -> str:
        import hashlib

        return "imgadopt_" + hashlib.sha256(operation_id.encode("ascii")).hexdigest()[:32]

    def _candidate(
        self,
        record: Any,
        operation_id: str,
        issued_at: str,
        updated_at: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        request = record.request if hasattr(record, "request") else record.response
        response = record.response
        expected_revision = response["expected_revision"]
        candidate = self._base_candidate(expected_revision, updated_at)
        adoption_id = self._adoption_id(operation_id)
        adoption = {
            "adoption_id": adoption_id,
            "source_context_id": response["source_context_id"],
            "source_work_id": response["source_work_id"],
            "source_revision_id": response["source_revision_id"],
            "snapshot": copy.deepcopy(record.snapshot),
            "adopted_at": issued_at,
            "operation_id": operation_id,
        }
        target = response["target_scope"]
        binding = {
            "binding_id": target["binding_id"],
            "label": response["label"],
            "film_reference_id": target["film_reference_id"],
            "active_adoption_id": adoption_id,
            "adoptions": [adoption],
        }
        candidate["state"]["bindings"].append(binding)
        payload = {"preview_token": record.response["preview_token"]}
        durable_receipt = {
            "operation_id": operation_id,
            "operation_kind": ADOPT_KIND,
            "issued_at": issued_at,
            "expires_at": self._timestamp_after(issued_at, 24 * 60 * 60),
            "payload_sha256": self._digest(payload),
            "expected_revision": expected_revision,
            "target_scope": copy.deepcopy(target),
            "status": "committed",
            "committed_revision": expected_revision + 1,
            "result": {
                "context_id": response["source_context_id"],
                "work_id": response["source_work_id"],
                "revision_id": response["source_revision_id"],
                "material_id": None,
                "binding_id": target["binding_id"],
                "adoption_id": adoption_id,
            },
        }
        candidate["state"]["receipts"].append(durable_receipt)
        return candidate, durable_receipt

    def _map_store_error(self, exc: Exception) -> str:
        if not isinstance(exc, FilmBindingStoreError):
            return "image_state_unavailable"
        code = exc.code
        if code in {"revision-conflict", "initial-cas-invalid", "upgrade-source-not-v1"}:
            return "image_conflict"
        if code in {"binding-target-invalid", "binding-target-mismatch", "binding-target-expired", "binding-target-unused", "operation-conflict"}:
            return "image_operation_conflict"
        return "image_state_unavailable"

    def execute(self, request: dict[str, Any], session_id: str) -> dict[str, Any]:
        operation_id: str | None = None
        target_scope: dict[str, Any] | None = None
        try:
            request = self._validate_execute_request(request)
            operation_id = request["operation_id"]
            target_scope = request["target_scope"]
        except ImageAdoptionError as exc:
            return self._safe_error(ADOPT_KIND, exc.code, target_scope=target_scope or self._unassigned_scope(self.project_id), operation_id=operation_id)
        try:
            session_id = self._session(session_id)
        except ImageAdoptionError as exc:
            return self._safe_error(ADOPT_KIND, exc.code, target_scope=target_scope, operation_id=operation_id)
        with self._adoption_lock:
            try:
                token_for_auth = request["payload"]["preview_token"]
                preview_for_auth = self._previews.get(token_for_auth)
                if preview_for_auth is not None and preview_for_auth.session_id != session_id:
                    return self._safe_error(ADOPT_KIND, "image_wrong_session", target_scope=target_scope, operation_id=operation_id)
                completed_session = self._completed_sessions.get(operation_id) if operation_id is not None else None
                if completed_session is not None and completed_session != session_id:
                    return self._safe_error(ADOPT_KIND, "image_wrong_session", target_scope=target_scope, operation_id=operation_id)
                # Durable evidence is authoritative and is intentionally looked
                # up before the ephemeral token/session record.
                durable = self._durable_receipt(operation_id)
                if durable is not None:
                    if not self._receipt_matches(request, durable):
                        return self._safe_error(ADOPT_KIND, "image_operation_conflict", target_scope=target_scope, operation_id=operation_id)
                    return durable
                try:
                    operation = self._operations.get(operation_id)
                except Exception as exc:
                    raise ImageAdoptionError("image_operation_conflict") from exc
                if operation is None:
                    return self._safe_error(ADOPT_KIND, "image_preview_expired", target_scope=target_scope, operation_id=operation_id)
                if operation.session_id != session_id:
                    return self._safe_error(ADOPT_KIND, "image_wrong_session", target_scope=target_scope, operation_id=operation_id)
                if operation.state == "cancelled":
                    return self._safe_error(ADOPT_KIND, "image_preview_cancelled", target_scope=target_scope, operation_id=operation_id)
                if operation.state == "failed":
                    return self._safe_error(ADOPT_KIND, operation.failure_reason or "image_source_unavailable", target_scope=target_scope, operation_id=operation_id)
                if operation.state == "expired":
                    return self._safe_error(ADOPT_KIND, "image_operation_expired", target_scope=target_scope, operation_id=operation_id)
                if operation.projection != self._projection(request, self._digest(request["payload"])) or self._operation_signature(operation.projection) != operation_id:
                    return self._safe_error(ADOPT_KIND, "image_operation_conflict", target_scope=target_scope, operation_id=operation_id)
                now_at, now = self._logical_now()
                if operation.expires_epoch <= now:
                    operation.state = "expired"
                    return self._safe_error(ADOPT_KIND, "image_operation_expired", target_scope=target_scope, operation_id=operation_id)
                token = request["payload"]["preview_token"]
                preview = self._previews.get(token)
                if preview is None or preview.operation_id != operation_id:
                    return self._safe_error(ADOPT_KIND, "image_preview_expired", target_scope=target_scope, operation_id=operation_id)
                if preview.session_id != session_id:
                    return self._safe_error(ADOPT_KIND, "image_wrong_session", target_scope=target_scope, operation_id=operation_id)
                if preview.state == "cancelled":
                    return self._safe_error(ADOPT_KIND, "image_preview_cancelled", target_scope=target_scope, operation_id=operation_id)
                if preview.state == "consumed":
                    return self._safe_error(ADOPT_KIND, "image_preview_consumed", target_scope=target_scope, operation_id=operation_id)
                if preview.state == "expired" or preview.expires_epoch <= now:
                    preview.state = "expired"
                    operation.state = "expired"
                    return self._safe_error(ADOPT_KIND, "image_preview_expired", target_scope=target_scope, operation_id=operation_id)
                if request["issued_at"] != operation.prepared["issued_at"] or request["payload"] != {"preview_token": preview.response["preview_token"]}:
                    return self._safe_error(ADOPT_KIND, "image_operation_conflict", target_scope=target_scope, operation_id=operation_id)
                if request["expected_revision"] != self._film_revision():
                    operation.state = "conflict"
                    return self._safe_error(ADOPT_KIND, "image_conflict", target_scope=target_scope, operation_id=operation_id)
                self._source_matches(preview)
                updated_at, _ = self._logical_now()
                candidate, _durable = self._candidate(preview, operation_id, operation.prepared["issued_at"], updated_at)
                if request["expected_revision"] == 0:
                    self._film_store_publish_initial(candidate, operation_id, target=preview.target)
                else:
                    self._film_store_publish_next(request["expected_revision"], candidate, operation_id, target=preview.target)
                committed = self._durable_receipt(operation_id)
                if committed is None or not self._receipt_matches(request, committed):
                    raise ImageAdoptionError("image_state_unavailable")
                # Consumption follows read-back of the durable Store evidence.
                preview.state = "consumed"
                operation.state = "committed"
                self._completed_sessions[operation_id] = session_id
                while len(self._completed_sessions) > self.max_pending_operations:
                    self._completed_sessions.pop(next(iter(self._completed_sessions)))
                del self._operations[operation_id]
                return committed
            except ImageAdoptionError as exc:
                if exc.code == "image_source_unavailable":
                    operation = self._operations.get(operation_id) if operation_id is not None else None
                    if operation is not None:
                        operation.state = "failed"
                        operation.failure_reason = "image_source_unavailable"
                # A publication may have succeeded before the caller observed
                # an exception.  Re-read the Store and recover only exact proof.
                if operation_id is not None:
                    try:
                        durable = self._durable_receipt(operation_id)
                    except ImageAdoptionError:
                        durable = None
                    if durable is not None and self._receipt_matches(request, durable):
                        preview = self._previews.get(request["payload"]["preview_token"])
                        if preview is not None:
                            preview.state = "consumed"
                        self._completed_sessions[operation_id] = session_id
                        while len(self._completed_sessions) > self.max_pending_operations:
                            self._completed_sessions.pop(next(iter(self._completed_sessions)))
                        self._operations.pop(operation_id, None)
                        return durable
                return self._safe_error(ADOPT_KIND, exc.code, target_scope=target_scope, operation_id=operation_id)
            except Exception as exc:
                code = self._map_store_error(exc)
                if operation_id is not None:
                    try:
                        durable = self._durable_receipt(operation_id)
                    except ImageAdoptionError:
                        durable = None
                    if durable is not None and self._receipt_matches(request, durable):
                        preview = self._previews.get(request["payload"]["preview_token"])
                        if preview is not None:
                            preview.state = "consumed"
                        self._completed_sessions[operation_id] = session_id
                        while len(self._completed_sessions) > self.max_pending_operations:
                            self._completed_sessions.pop(next(iter(self._completed_sessions)))
                        self._operations.pop(operation_id, None)
                        return durable
                return self._safe_error(ADOPT_KIND, code, target_scope=target_scope, operation_id=operation_id)

    film_image_adopt_new_reference = execute
    adopt_new_reference = execute

    def status(self, request: dict[str, Any], session_id: str | None = None) -> dict[str, Any]:
        operation_id: str | None = None
        try:
            request = self._validate_status_request(request)
            operation_id = request["operation_id"]
        except ImageAdoptionError as exc:
            return self._safe_error(ADOPT_KIND, exc.code, operation_id=operation_id)
        try:
            if session_id is not None:
                session_id = self._session(session_id)
            with self._adoption_lock:
                if session_id is not None:
                    completed_session = self._completed_sessions.get(operation_id)
                    if completed_session is not None and completed_session != session_id:
                        return self._safe_error(ADOPT_KIND, "image_wrong_session", operation_id=operation_id)
                durable = self._durable_receipt(operation_id)
                if durable is not None:
                    completed_session = self._completed_sessions.get(operation_id)
                    if completed_session is not None and completed_session != session_id:
                        return self._safe_error(ADOPT_KIND, "image_wrong_session", operation_id=operation_id)
                    if session_id is None or completed_session is None or completed_session != session_id:
                        return {
                            "schema": V2_SCHEMA,
                            "kind": STATUS_RESULT_KIND,
                            "branch_version": BRANCH_VERSION,
                            "owner_scope": self.owner_scope,
                            "operation_id": operation_id,
                            "operation_kind": ADOPT_KIND,
                            "status": "unknown",
                            "reason_code": "image_operation_unknown",
                            "receipt": None,
                        }
                    return {
                        "schema": V2_SCHEMA,
                        "kind": STATUS_RESULT_KIND,
                        "branch_version": BRANCH_VERSION,
                        "owner_scope": self.owner_scope,
                        "operation_id": operation_id,
                        "operation_kind": ADOPT_KIND,
                        "status": "committed",
                        "reason_code": None,
                        "receipt": durable,
                    }
                operation = self._operations.get(operation_id)
                if operation is not None and session_id is not None and operation.session_id != session_id:
                    return self._safe_error(ADOPT_KIND, "image_wrong_session", operation_id=operation_id)
                operation_state = None if operation is None else operation.state
                if operation_state == "pending":
                    _at, now = self._logical_now()
                    if operation.expires_epoch <= now:
                        operation_state = "expired"
                    else:
                        return {
                            "schema": V2_SCHEMA,
                            "kind": STATUS_RESULT_KIND,
                            "branch_version": BRANCH_VERSION,
                            "owner_scope": self.owner_scope,
                            "operation_id": operation_id,
                            "operation_kind": ADOPT_KIND,
                            "status": "pending",
                            "reason_code": "image_operation_pending",
                            "receipt": None,
                        }
                if operation is not None and operation_state == "expired":
                    return {
                        "schema": V2_SCHEMA,
                        "kind": STATUS_RESULT_KIND,
                        "branch_version": BRANCH_VERSION,
                        "owner_scope": self.owner_scope,
                        "operation_id": operation_id,
                        "operation_kind": ADOPT_KIND,
                        "status": "expired",
                        "reason_code": "image_operation_expired",
                        "receipt": None,
                    }
                if operation is not None and operation.state == "failed" and operation.failure_reason is not None:
                    return {
                        "schema": V2_SCHEMA,
                        "kind": STATUS_RESULT_KIND,
                        "branch_version": BRANCH_VERSION,
                        "owner_scope": self.owner_scope,
                        "operation_id": operation_id,
                        "operation_kind": ADOPT_KIND,
                        "status": "failed",
                        "reason_code": operation.failure_reason,
                        "receipt": None,
                    }
                state, _temporal = self._film_state()
                floor = None if state is None else state.get("state", {}).get("receipt_floor_at")
                issued_second = int(operation_id[len("imgop_"): len("imgop_") + 16], 16)
                if type(floor) is str:
                    floor_second = int(self._epoch(floor))
                else:
                    floor_second = -1
                status = "expired" if issued_second <= floor_second else "unknown"
                reason = "image_operation_expired" if status == "expired" else "image_operation_unknown"
                return {
                    "schema": V2_SCHEMA,
                    "kind": STATUS_RESULT_KIND,
                    "branch_version": BRANCH_VERSION,
                    "owner_scope": self.owner_scope,
                    "operation_id": operation_id,
                    "operation_kind": ADOPT_KIND,
                    "status": status,
                    "reason_code": reason,
                    "receipt": None,
                }
        except ImageAdoptionError as exc:
            return self._safe_error(ADOPT_KIND, exc.code, operation_id=operation_id)
        except Exception:
            return self._safe_error(ADOPT_KIND, "image_state_unavailable", operation_id=operation_id)

    image_operation_status = status

    @staticmethod
    def status_request(execute_or_prepared: dict[str, Any]) -> dict[str, Any]:
        try:
            return {
                "schema": V2_SCHEMA,
                "kind": "image_operation_status",
                "branch_version": BRANCH_VERSION,
                "owner_scope": execute_or_prepared["owner_scope"],
                "operation_id": execute_or_prepared["operation_id"],
                "operation_kind": ADOPT_KIND,
            }
        except Exception as exc:
            raise ImageAdoptionError("prepared-response-shape") from exc

    def inspect(
        self,
        binding_id: str | None = None,
        film_reference_id: str | None = None,
    ) -> dict[str, Any]:
        if binding_id is None and film_reference_id is None:
            raise FilmAdoptionError("binding-target-required")
        if binding_id is not None:
            _validate_id(binding_id, _BINDING_RE, "binding-id-invalid")
        if film_reference_id is not None:
            _validate_id(film_reference_id, _FILM_REFERENCE_RE, "film-reference-id-invalid")
        try:
            state, _temporal = self._film_state()
        except Exception as exc:
            raise FilmAdoptionError("image_state_unavailable") from exc
        if state is None:
            raise FilmAdoptionError("binding-missing")
        try:
            bindings = state["state"]["bindings"]
            matches = [
                item for item in bindings
                if (binding_id is None or item["binding_id"] == binding_id)
                and (film_reference_id is None or item["film_reference_id"] == film_reference_id)
            ]
            if len(matches) != 1:
                raise FilmAdoptionError("binding-missing")
            binding = copy.deepcopy(matches[0])
            active = next(item for item in binding["adoptions"] if item["adoption_id"] == binding["active_adoption_id"])
            snapshot = copy.deepcopy(active["snapshot"])
            availability, use_allowed = self._current_availability(active, snapshot)
            projection = {
                "schema": "studio.image-binding-inspect/1",
                "project_id": self.project_id,
                "bindings_revision": state["revision"],
                "binding": binding,
                "active_adoption": copy.deepcopy(active),
                "snapshot": snapshot,
                "availability": availability,
                "use_allowed": use_allowed,
            }
            try:
                return validate_image_binding_inspect_response(projection, self.project_id)
            except FilmBindingContractError as exc:
                raise FilmAdoptionError("image_state_unavailable") from exc
        except FilmAdoptionError:
            raise
        except Exception as exc:
            raise FilmAdoptionError("image_state_unavailable") from exc

    film_image_inspect = inspect

    def _current_availability(self, adoption: dict[str, Any], snapshot: dict[str, Any]) -> tuple[str, bool]:
        try:
            _state, current_snapshot, _digest = self._load_snapshot(
                adoption["source_context_id"], adoption["source_work_id"], adoption["source_revision_id"], require_available=False
            )
        except ImageAdoptionError as exc:
            return ("denied", False) if exc.code == "image_authorization_denied" else ("missing", False)
        if self._snapshot_identity(current_snapshot) != self._snapshot_identity(snapshot):
            return "changed", False
        statuses = [item["availability"] for item in current_snapshot["materials"]]
        if any(status == "denied" for status in statuses):
            return "denied", False
        if any(status == "missing" for status in statuses):
            return "missing", False
        return "available", True


ImageFilmAdoptionService = FilmAdoptionService


__all__ = [
    "FilmAdoptionError",
    "FilmAdoptionService",
    "ImageFilmAdoptionService",
]
