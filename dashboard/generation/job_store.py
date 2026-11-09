"""Durable, provider-neutral Studio generation ownership ledger.

The ledger is deliberately independent from provider submission.  It records the
request and ownership decisions before a later adapter talks to a provider.
Every mutating operation opens its own SQLite connection, uses an explicit
transaction, and closes the connection before returning so two store instances
exercise the same durable constraints.
"""

from __future__ import annotations
from .legacy_plugin_migration import state_root


import hashlib
import json
import math
import ntpath
import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    from hermes_constants import get_hermes_home
except ImportError:  # pragma: no cover - repository tests install Hermes' root
    def get_hermes_home() -> Path:
        return Path.home() / "AppData" / "Local" / "hermes"

from .contract import GENERATION_REQUEST_SCHEMA, GenerationRequest, REASON_CODES
from .registry import GenerationDomainError, InvalidRequestError


SCHEMA_VERSION = 4
DEFAULT_CONNECTION_ID = "default"
DEFAULT_DESTINATION = Path.home() / "Downloads"
DEFAULT_BASENAME = "generation.mp4"
_NONTERMINAL_STATES = frozenset({"submitting", "queued", "running"})
_TERMINAL_STATES = frozenset({"ready", "failed", "moderated", "cancelled"})
_STATES = _NONTERMINAL_STATES | _TERMINAL_STATES
_STAMP_STATES = frozenset({"not_started", "pending", "succeeded", "failed"})
_PUBLICATION_STATES = frozenset({"unpublished", "publishing", "published"})
_ALLOWED_TRANSITIONS = {
    "submitting": frozenset({"submitting", "queued", "running", *_TERMINAL_STATES}),
    "queued": frozenset({"queued", "running", *_TERMINAL_STATES}),
    "running": frozenset({"running", *_TERMINAL_STATES}),
    **{state: frozenset({state}) for state in _TERMINAL_STATES},
}
_LEGACY_OUTPUT_FAILURE_REASONS = frozenset({"output_missing", "output_invalid"})
_WINDOWS_RESERVED_BASENAMES = frozenset(
    {"con", "prn", "aux", "nul", *(f"com{index}" for index in range(1, 10)), *(f"lpt{index}" for index in range(1, 10))}
)
_WINDOWS_INVALID_BASENAME_CHARS = frozenset('<>:"/\\|?*')
_SQLITE_PUBLISHED_PATH_CHILD_FUNCTION = "studio_published_path_is_child"
_HEX64_SQL = "typeof({field}) = 'text' AND length({field}) = 64 AND {field} NOT GLOB '*[^0-9a-f]*'"
_PROJECT_ID_SQL = "typeof({field}) = 'text' AND length({field}) = 36 AND substr({field}, 1, 4) = 'prj_' AND substr({field}, 5) NOT GLOB '*[^0-9a-f]*'"
_REQUEST_HASH_SQL = _HEX64_SQL.format(field="request_hash")
_DESTINATION_ID_SQL = _HEX64_SQL.format(field="destination_identity")
_PUBLISHED_ID_SQL = _HEX64_SQL.format(field="published_identity")
_PROJECT_ID_NEW_SQL = _PROJECT_ID_SQL.format(field="NEW.project_id")
_DESTINATION_ID_NEW_SQL = _HEX64_SQL.format(field="NEW.destination_identity")
_PUBLISHED_ID_NEW_SQL = _HEX64_SQL.format(field="NEW.published_identity")


class JobStoreError(GenerationDomainError):
    """Base class for stable ledger errors."""

    def __init__(self, reason_code: str, message: str | None = None) -> None:
        self.reason_code = reason_code if reason_code in REASON_CODES else "unknown"
        self.public_message = {
            "provider_busy": "Provider is busy",
            "invalid_request": "Generation request is invalid",
            "unknown": "Unknown generation error",
        }.get(self.reason_code, "Unknown generation error")
        ValueError.__init__(self, message or self.public_message)

    def to_result(self) -> dict[str, str]:
        return {"reason_code": self.reason_code}


class ProviderBusyError(JobStoreError):
    def __init__(self) -> None:
        super().__init__("provider_busy")


class LeaseConflictError(JobStoreError):
    def __init__(self) -> None:
        super().__init__("provider_busy", "Publication lease is owned by another worker")


class JobNotFoundError(InvalidRequestError):
    pass


class ProjectScopeError(InvalidRequestError):
    """The caller supplied a project scope different from the persisted row."""

    def __init__(self) -> None:
        super().__init__("Generation job belongs to a different project context")
        self.reason_code = "project_scope_mismatch"
        self.public_message = "Generation job belongs to a different project context"


# Keep the descriptive name used by the initial V4 API while exposing the
# shorter stable error name for callers that only care about scope isolation.
ProjectScopeMismatchError = ProjectScopeError


class SchemaError(RuntimeError):
    pass


@dataclass(frozen=True)
class JobRecord:
    studio_job_id: str
    client_request_id: str
    provider_id: str
    target_id: str
    provider_job_id: str | None
    connection_id: str
    request_json: str
    request_hash: str
    project_id: str | None
    row_contract: int
    destination_identity: str | None
    published_identity: str | None
    frozen_destination: str | None
    requested_basename: str
    state: str
    created_at: float
    updated_at: float
    queued_at: float | None
    running_at: float | None
    completed_at: float | None
    reason_code: str | None
    provider_output_identity_json: str | None
    publication_state: str
    publication_lease_owner: str | None
    publication_lease_token: str | None
    publication_lease_expires_at: float | None
    published_path: str | None
    source_media_sha256: str | None
    published_file_sha256: str | None
    stamp_state: str
    stamp_attempt_count: int

    @property
    def request(self) -> dict[str, Any]:
        return JobStore._row_request_json(self.request_json)

    @property
    def is_terminal(self) -> bool:
        return self.state in _TERMINAL_STATES

    def to_dict(self) -> dict[str, Any]:
        return {
            "studio_job_id": self.studio_job_id,
            "client_request_id": self.client_request_id,
            "provider_id": self.provider_id,
            "target_id": self.target_id,
            "provider_job_id": self.provider_job_id,
            "connection_id": self.connection_id,
            "request_json": self.request_json,
            "request_hash": self.request_hash,
            "project_id": self.project_id,
            "row_contract": self.row_contract,
            "destination_identity": self.destination_identity,
            "published_identity": self.published_identity,
            "frozen_destination": self.frozen_destination,
            "requested_basename": self.requested_basename,
            "state": self.state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "queued_at": self.queued_at,
            "running_at": self.running_at,
            "completed_at": self.completed_at,
            "reason_code": self.reason_code,
            "provider_output_identity_json": self.provider_output_identity_json,
            "publication_state": self.publication_state,
            "publication_lease_owner": self.publication_lease_owner,
            "publication_lease_token": self.publication_lease_token,
            "publication_lease_expires_at": self.publication_lease_expires_at,
            "published_path": self.published_path,
            "source_media_sha256": self.source_media_sha256,
            "published_file_sha256": self.published_file_sha256,
            "stamp_state": self.stamp_state,
            "stamp_attempt_count": self.stamp_attempt_count,
        }

    as_dict = to_dict


@dataclass(frozen=True)
class PublicationLease:
    studio_job_id: str
    owner_id: str
    token: str
    expires_at: float


_PUBLICATION_METADATA_ALL_NULL_SQL = """
published_path IS NULL
AND source_media_sha256 IS NULL
AND published_file_sha256 IS NULL
"""
_PUBLICATION_METADATA_COMPLETE_SQL = """
typeof(published_path) = 'text'
AND length(trim(published_path)) > 0
AND typeof(source_media_sha256) = 'text'
AND length(trim(source_media_sha256)) > 0
AND typeof(published_file_sha256) = 'text'
AND length(trim(published_file_sha256)) > 0
"""
_PUBLICATION_METADATA_ALL_NULL_NEW_SQL = """
NEW.published_path IS NULL
AND NEW.source_media_sha256 IS NULL
AND NEW.published_file_sha256 IS NULL
"""
_PUBLICATION_METADATA_COMPLETE_NEW_SQL = """
typeof(NEW.published_path) = 'text'
AND length(trim(NEW.published_path)) > 0
AND typeof(NEW.source_media_sha256) = 'text'
AND length(trim(NEW.source_media_sha256)) > 0
AND typeof(NEW.published_file_sha256) = 'text'
AND length(trim(NEW.published_file_sha256)) > 0
"""
_PUBLICATION_LEASE_ALL_NULL_SQL = """
publication_lease_owner IS NULL
AND publication_lease_token IS NULL
AND publication_lease_expires_at IS NULL
"""
_PUBLICATION_LEASE_ALL_NULL_NEW_SQL = """
NEW.publication_lease_owner IS NULL
AND NEW.publication_lease_token IS NULL
AND NEW.publication_lease_expires_at IS NULL
"""
_PUBLICATION_LEASE_COMPLETE_SQL = """
typeof(publication_lease_owner) = 'text'
AND length(trim(publication_lease_owner)) > 0
AND typeof(publication_lease_token) = 'text'
AND length(trim(publication_lease_token)) > 0
AND typeof(publication_lease_expires_at) IN ('integer', 'real')
AND publication_lease_expires_at = publication_lease_expires_at
AND publication_lease_expires_at <= 1.7976931348623157e308
AND publication_lease_expires_at >= -1.7976931348623157e308
"""
_PUBLICATION_LEASE_COMPLETE_NEW_SQL = """
typeof(NEW.publication_lease_owner) = 'text'
AND length(trim(NEW.publication_lease_owner)) > 0
AND typeof(NEW.publication_lease_token) = 'text'
AND length(trim(NEW.publication_lease_token)) > 0
AND typeof(NEW.publication_lease_expires_at) IN ('integer', 'real')
AND NEW.publication_lease_expires_at = NEW.publication_lease_expires_at
AND NEW.publication_lease_expires_at <= 1.7976931348623157e308
AND NEW.publication_lease_expires_at >= -1.7976931348623157e308
"""
_PUBLICATION_LEASE_COMPLETE_OLD_SQL = """
typeof(OLD.publication_lease_owner) = 'text'
AND length(trim(OLD.publication_lease_owner)) > 0
AND typeof(OLD.publication_lease_token) = 'text'
AND length(trim(OLD.publication_lease_token)) > 0
AND typeof(OLD.publication_lease_expires_at) IN ('integer', 'real')
AND OLD.publication_lease_expires_at = OLD.publication_lease_expires_at
AND OLD.publication_lease_expires_at <= 1.7976931348623157e308
AND OLD.publication_lease_expires_at >= -1.7976931348623157e308
"""
_PUBLICATION_LEASE_MATCH_OLD_NEW_SQL = """
NEW.publication_lease_owner IS OLD.publication_lease_owner
AND NEW.publication_lease_token IS OLD.publication_lease_token
AND NEW.publication_lease_expires_at IS OLD.publication_lease_expires_at
"""
_REASON_CODES_SQL = "(" + ", ".join(
    "'" + reason_code.replace("'", "''") + "'" for reason_code in sorted(REASON_CODES)
) + ")"
_REASON_CODE_INVALID_SQL = f"""
reason_code IS NOT NULL
AND (
    typeof(reason_code) <> 'text'
    OR reason_code NOT IN {_REASON_CODES_SQL}
)
"""
_REASON_CODE_INVALID_NEW_SQL = f"""
NEW.reason_code IS NOT NULL
AND (
    typeof(NEW.reason_code) <> 'text'
    OR NEW.reason_code NOT IN {_REASON_CODES_SQL}
)
"""
_STAMP_STATE_INVALID_SQL = """
(
    typeof(stamp_state) <> 'text'
    OR stamp_state NOT IN ('not_started','pending','succeeded','failed')
    OR typeof(stamp_attempt_count) <> 'integer'
    OR (
        stamp_state = 'not_started'
        AND stamp_attempt_count <> 0
    )
    OR (
        stamp_state IN ('pending','succeeded','failed')
        AND stamp_attempt_count <= 0
    )
)
"""
_STAMP_STATE_INVALID_NEW_SQL = """
(
    typeof(NEW.stamp_state) <> 'text'
    OR NEW.stamp_state NOT IN ('not_started','pending','succeeded','failed')
    OR typeof(NEW.stamp_attempt_count) <> 'integer'
    OR (
        NEW.stamp_state = 'not_started'
        AND NEW.stamp_attempt_count <> 0
    )
    OR (
        NEW.stamp_state IN ('pending','succeeded','failed')
        AND NEW.stamp_attempt_count <= 0
    )
)
"""
_STAMP_INSERT_STATE_INVALID_NEW_SQL = """
(
    typeof(NEW.stamp_state) <> 'text'
    OR NEW.stamp_state IS NOT 'not_started'
    OR typeof(NEW.stamp_attempt_count) <> 'integer'
    OR NEW.stamp_attempt_count IS NOT 0
)
"""
_PUBLICATION_OUTPUT_IDENTITY_COMPLETE_SQL = """
typeof(provider_output_identity_json) = 'text'
AND length(trim(provider_output_identity_json)) > 0
AND json_valid(provider_output_identity_json) = 1
"""
_PUBLICATION_OUTPUT_IDENTITY_COMPLETE_NEW_SQL = """
typeof(NEW.provider_output_identity_json) = 'text'
AND length(trim(NEW.provider_output_identity_json)) > 0
AND json_valid(NEW.provider_output_identity_json) = 1
"""
_PUBLICATION_OUTPUT_IDENTITY_INVALID_SQL = f"""
provider_output_identity_json IS NOT NULL
AND NOT ({_PUBLICATION_OUTPUT_IDENTITY_COMPLETE_SQL})
"""
_PUBLICATION_OUTPUT_IDENTITY_INVALID_NEW_SQL = f"""
NEW.provider_output_identity_json IS NOT NULL
AND NOT ({_PUBLICATION_OUTPUT_IDENTITY_COMPLETE_NEW_SQL})
"""
_V4_OUTPUT_IDENTITY_COMPLETE_TEMPLATE = """
typeof({field}) = 'text'
AND length(trim({field})) > 0
AND json_valid({field}) = 1
AND CASE WHEN json_valid({field}) = 1 THEN
    json_type({field}) = 'object'
    AND json_type({field}, '$.bytes') = 'integer'
    AND json_extract({field}, '$.bytes') >= 0
    AND json_type({field}, '$.published_identity') = 'text'
    AND length(json_extract({field}, '$.published_identity')) = 64
    AND json_extract({field}, '$.published_identity') NOT GLOB '*[^0-9a-f]*'
    AND json_type({field}, '$.sha256') = 'text'
    AND length(json_extract({field}, '$.sha256')) = 64
    AND json_extract({field}, '$.sha256') NOT GLOB '*[^0-9a-f]*'
    AND {field} = json_object(
        'bytes', json_extract({field}, '$.bytes'),
        'published_identity', json_extract({field}, '$.published_identity'),
        'sha256', json_extract({field}, '$.sha256')
    )
ELSE 0 END
"""
_V4_OUTPUT_IDENTITY_COMPLETE_SQL = _V4_OUTPUT_IDENTITY_COMPLETE_TEMPLATE.format(
    field="provider_output_identity_json"
)
_V4_OUTPUT_IDENTITY_COMPLETE_NEW_SQL = _V4_OUTPUT_IDENTITY_COMPLETE_TEMPLATE.format(
    field="NEW.provider_output_identity_json"
)
_V4_DELIVERY_RELATION_INVALID_SQL = f"""
row_contract = 4
AND NOT (
    (
        published_identity IS NULL
        AND provider_output_identity_json IS NULL
        AND state <> 'ready'
    )
    OR (
        state = 'ready'
        AND ({_V4_OUTPUT_IDENTITY_COMPLETE_SQL})
        AND published_identity IS NOT NULL
        AND json_extract(provider_output_identity_json, '$.published_identity') = published_identity
    )
)
"""
_V4_DELIVERY_RELATION_INVALID_NEW_SQL = f"""
NEW.row_contract = 4
AND NOT (
    (
        NEW.published_identity IS NULL
        AND NEW.provider_output_identity_json IS NULL
        AND NEW.state <> 'ready'
    )
    OR (
        NEW.state = 'ready'
        AND ({_V4_OUTPUT_IDENTITY_COMPLETE_NEW_SQL})
        AND NEW.published_identity IS NOT NULL
        AND json_extract(NEW.provider_output_identity_json, '$.published_identity') = NEW.published_identity
    )
)
"""
_V4_DELIVERY_OLD_ELIGIBLE_SQL = f"""
OLD.row_contract = 4
AND OLD.state IN ('submitting', 'queued', 'running', 'ready')
AND OLD.published_identity IS NULL
AND OLD.provider_output_identity_json IS NULL
AND OLD.reason_code IS NULL
AND OLD.publication_state = 'unpublished'
AND OLD.publication_lease_owner IS NULL
AND OLD.publication_lease_token IS NULL
AND OLD.publication_lease_expires_at IS NULL
AND OLD.published_path IS NULL
AND OLD.source_media_sha256 IS NULL
AND OLD.published_file_sha256 IS NULL
"""
_V4_DELIVERY_TRANSITION_INVALID_NEW_SQL = f"""
NEW.row_contract = 4
AND NEW.state = 'ready'
AND NEW.published_identity IS NOT NULL
AND ({_V4_OUTPUT_IDENTITY_COMPLETE_NEW_SQL})
AND json_extract(NEW.provider_output_identity_json, '$.published_identity') = NEW.published_identity
AND NOT (
    (
        OLD.row_contract = 4
        AND OLD.state = 'ready'
        AND OLD.published_identity IS NEW.published_identity
        AND OLD.provider_output_identity_json IS NEW.provider_output_identity_json
    )
    OR ({_V4_DELIVERY_OLD_ELIGIBLE_SQL})
)
"""
_PUBLICATION_PUBLISHING_ELIGIBLE_SQL = f"""
state = 'ready'
AND ({_PUBLICATION_OUTPUT_IDENTITY_COMPLETE_SQL})
AND ({_PUBLICATION_LEASE_COMPLETE_SQL})
"""
_PUBLICATION_PUBLISHING_ELIGIBLE_NEW_SQL = f"""
NEW.state = 'ready'
AND ({_PUBLICATION_OUTPUT_IDENTITY_COMPLETE_NEW_SQL})
AND ({_PUBLICATION_LEASE_COMPLETE_NEW_SQL})
"""
_PUBLICATION_METADATA_INVALID_SQL = f"""
(
    NOT (
        ({_PUBLICATION_METADATA_ALL_NULL_SQL})
        OR ({_PUBLICATION_METADATA_COMPLETE_SQL})
    )
    OR (
        publication_state = 'unpublished'
        AND NOT (
            ({_PUBLICATION_METADATA_ALL_NULL_SQL})
            AND ({_PUBLICATION_LEASE_ALL_NULL_SQL})
        )
    )
    OR (
        publication_state = 'publishing'
        AND NOT ({_PUBLICATION_METADATA_ALL_NULL_SQL})
    )
    OR (
        publication_state = 'published'
        AND NOT ({_PUBLICATION_METADATA_COMPLETE_SQL})
    )
)
"""
_PUBLICATION_STATE_INVARIANT_INVALID_NEW_SQL = f"""
(
    (
        NEW.publication_state = 'unpublished'
        AND NOT (
            ({_PUBLICATION_METADATA_ALL_NULL_NEW_SQL})
            AND ({_PUBLICATION_LEASE_ALL_NULL_NEW_SQL})
        )
    )
    OR (
        NEW.publication_state = 'publishing'
        AND NOT ({_PUBLICATION_METADATA_ALL_NULL_NEW_SQL})
    )
    OR (
        NEW.publication_state = 'published'
        AND NOT ({_PUBLICATION_METADATA_COMPLETE_NEW_SQL})
    )
)
"""
_PUBLICATION_STATE_INVALID_SQL = f"""
(
    typeof(studio_job_id) <> 'text'
    OR length(trim(studio_job_id)) <= 0
    OR typeof(state) <> 'text'
    OR state NOT IN ('submitting','queued','running','ready','failed','moderated','cancelled')
    OR typeof(publication_state) <> 'text'
    OR publication_state NOT IN ('unpublished','publishing','published')
    OR ({_REASON_CODE_INVALID_SQL})
    OR ({_STAMP_STATE_INVALID_SQL})
    OR ({_PUBLICATION_OUTPUT_IDENTITY_INVALID_SQL})
    OR ({_PUBLICATION_METADATA_INVALID_SQL})
    OR (
        publication_state = 'publishing'
        AND NOT ({_PUBLICATION_PUBLISHING_ELIGIBLE_SQL})
    )
    OR (
        publication_state = 'published'
        AND NOT (
            state = 'ready'
            AND ({_PUBLICATION_OUTPUT_IDENTITY_COMPLETE_SQL})
            AND ({_PUBLICATION_METADATA_COMPLETE_SQL})
            AND ({_PUBLICATION_LEASE_COMPLETE_SQL})
        )
    )
)
"""

_READY_WITHOUT_OUTPUT_REPAIR_SQL = f"""
CASE WHEN
    OLD.state = 'ready'
    AND NEW.state = 'failed'
    AND OLD.provider_id = 'comfyui-local'
    AND OLD.provider_output_identity_json IS NULL
    AND OLD.publication_state = 'unpublished'
    AND OLD.reason_code IS NULL
    AND NEW.reason_code IN ('output_missing','output_invalid')
    AND NEW.provider_output_identity_json IS NULL
    AND NEW.publication_state = 'unpublished'
THEN 1 ELSE 0 END = 1
"""

_LIFECYCLE_TRANSITIONS_SQL = f"""
(
    (OLD.state = 'submitting' AND NEW.state IN ('submitting','queued','running','ready','failed','moderated','cancelled'))
    OR (OLD.state = 'queued' AND NEW.state IN ('queued','running','ready','failed','moderated','cancelled'))
    OR (OLD.state = 'running' AND NEW.state IN ('running','ready','failed','moderated','cancelled'))
    OR (OLD.state = 'ready' AND NEW.state = 'ready')
    OR (OLD.state = 'failed' AND NEW.state = 'failed')
    OR (OLD.state = 'moderated' AND NEW.state = 'moderated')
    OR (OLD.state = 'cancelled' AND NEW.state = 'cancelled')
    OR ({_READY_WITHOUT_OUTPUT_REPAIR_SQL})
)
"""

_V4_ROW_SHAPE_INVALID_NEW_SQL = f"""
(
    NEW.row_contract = 4
    AND (
        (NEW.project_id IS NOT NULL AND NOT ({_PROJECT_ID_NEW_SQL}))
        OR NEW.destination_identity IS NULL
        OR NOT ({_DESTINATION_ID_NEW_SQL})
        OR NEW.frozen_destination IS NOT NULL
        OR NEW.published_path IS NOT NULL
        OR NEW.published_identity IS NOT NULL
        OR NEW.provider_output_identity_json IS NOT NULL
        OR (NEW.provider_job_id IS NOT NULL AND (
            typeof(NEW.provider_job_id) <> 'text'
            OR length(trim(NEW.provider_job_id)) <= 0
        ))
        OR typeof(NEW.client_request_id) <> 'text' OR length(trim(NEW.client_request_id)) <= 0
        OR typeof(NEW.provider_id) <> 'text' OR length(trim(NEW.provider_id)) <= 0
        OR typeof(NEW.target_id) <> 'text' OR length(trim(NEW.target_id)) <= 0
        OR typeof(NEW.connection_id) <> 'text' OR length(trim(NEW.connection_id)) <= 0
        OR typeof(NEW.request_json) <> 'text' OR length(trim(NEW.request_json)) <= 0
        OR json_valid(NEW.request_json) <> 1 OR json_type(NEW.request_json) <> 'object'
        OR NOT ({_REQUEST_HASH_SQL.replace('request_hash', 'NEW.request_hash')})
        OR typeof(NEW.requested_basename) <> 'text' OR length(trim(NEW.requested_basename)) <= 0
        OR instr(NEW.requested_basename, '/') > 0 OR instr(NEW.requested_basename, char(92)) > 0
        OR NEW.requested_basename NOT GLOB '*.mp4'
        OR NEW.requested_basename GLOB '*[. ]'
    )
)
"""
_CONTRACT_INVALID_NEW_SQL = f"""
(
    NEW.row_contract IS NOT 4
    OR ({_V4_ROW_SHAPE_INVALID_NEW_SQL})
)
"""

_CONTRACT_INSERT_TRIGGER_STATEMENT = f"""
CREATE TRIGGER studio_jobs_contract_on_insert
BEFORE INSERT ON studio_jobs
WHEN {_CONTRACT_INVALID_NEW_SQL}
BEGIN
    SELECT RAISE(ABORT, 'invalid Studio row contract');
END
"""

# Keep this schema in one explicit migration so a future version can add a
# migration without silently recreating or dropping the user ledger.
_DURABILITY_TRIGGER_STATEMENTS = (
    "DROP TRIGGER IF EXISTS studio_jobs_provider_job_id_write_once",
    "DROP TRIGGER IF EXISTS studio_jobs_provider_output_identity_write_once",
    "DROP TRIGGER IF EXISTS studio_jobs_provider_output_identity_json_on_insert",
    "DROP TRIGGER IF EXISTS studio_jobs_provider_output_identity_json_on_update",
    "DROP TRIGGER IF EXISTS studio_jobs_publication_metadata_all_or_none_on_insert",
    "DROP TRIGGER IF EXISTS studio_jobs_publication_metadata_all_or_none_on_update",
    "DROP TRIGGER IF EXISTS studio_jobs_published_metadata_on_insert",
    "DROP TRIGGER IF EXISTS studio_jobs_published_metadata_on_update",
    "DROP TRIGGER IF EXISTS studio_jobs_publication_lease_on_insert",
    "DROP TRIGGER IF EXISTS studio_jobs_publication_lease_on_update",
    "DROP TRIGGER IF EXISTS studio_jobs_publication_state_on_insert",
    "DROP TRIGGER IF EXISTS studio_jobs_publication_state_on_update",
    "DROP TRIGGER IF EXISTS studio_jobs_publication_state_invariant_on_insert",
    "DROP TRIGGER IF EXISTS studio_jobs_publication_state_invariant_on_update",
    "DROP TRIGGER IF EXISTS studio_jobs_publication_published_on_insert",
    "DROP TRIGGER IF EXISTS studio_jobs_publication_published_on_update",
    "DROP TRIGGER IF EXISTS studio_jobs_publication_path_confinement_on_update",
    "DROP TRIGGER IF EXISTS studio_jobs_publication_metadata_write_once",
    "DROP TRIGGER IF EXISTS studio_jobs_immutable_request",
    "DROP TRIGGER IF EXISTS studio_jobs_immutable_id",
    "DROP TRIGGER IF EXISTS studio_jobs_immutable_id_on_insert",
    "DROP TRIGGER IF EXISTS studio_jobs_lifecycle_monotonic",
    "DROP TRIGGER IF EXISTS studio_jobs_lifecycle_state_on_insert",
    "DROP TRIGGER IF EXISTS studio_jobs_terminal_reason_write_once",
    "DROP TRIGGER IF EXISTS studio_jobs_terminal_lifecycle_write_once",
    "DROP TRIGGER IF EXISTS studio_jobs_reason_code_on_insert",
    "DROP TRIGGER IF EXISTS studio_jobs_reason_code_on_update",
    "DROP TRIGGER IF EXISTS studio_jobs_stamp_state_on_insert",
    "DROP TRIGGER IF EXISTS studio_jobs_stamp_state_invariant_on_update",
    "DROP TRIGGER IF EXISTS studio_jobs_publication_terminal_write_once",
    "DROP TRIGGER IF EXISTS studio_jobs_stamp_terminal_write_once",
    "DROP TRIGGER IF EXISTS studio_jobs_stamp_state_machine",
    "DROP TRIGGER IF EXISTS studio_jobs_contract_on_insert",
    "DROP TRIGGER IF EXISTS studio_jobs_contract_shape_on_update",
    "DROP TRIGGER IF EXISTS studio_jobs_project_identity_write_once",
    "DROP TRIGGER IF EXISTS studio_jobs_destination_identity_write_once",
    "DROP TRIGGER IF EXISTS studio_jobs_published_identity_write_once",
    "DROP TRIGGER IF EXISTS studio_jobs_published_identity_on_insert",
    "DROP TRIGGER IF EXISTS studio_jobs_published_identity_on_update",
    "DROP TRIGGER IF EXISTS studio_jobs_v4_contract_shape_on_insert",
    "DROP TRIGGER IF EXISTS studio_jobs_v4_request_shape_on_insert",
    "DROP TRIGGER IF EXISTS studio_jobs_v4_request_shape_on_update",
    "DROP TRIGGER IF EXISTS studio_jobs_v4_delivery_relation_on_update",
    "DROP TRIGGER IF EXISTS studio_jobs_v4_delivery_transition_on_update",
    _CONTRACT_INSERT_TRIGGER_STATEMENT,
    """
    CREATE TRIGGER studio_jobs_immutable_request
    BEFORE UPDATE OF client_request_id, provider_id, target_id, connection_id,
        request_json, request_hash, frozen_destination, requested_basename
    ON studio_jobs
    WHEN OLD.client_request_id IS NOT NEW.client_request_id
      OR OLD.provider_id IS NOT NEW.provider_id
      OR OLD.target_id IS NOT NEW.target_id
      OR OLD.connection_id IS NOT NEW.connection_id
      OR OLD.request_json IS NOT NEW.request_json
      OR OLD.request_hash IS NOT NEW.request_hash
      OR OLD.frozen_destination IS NOT NEW.frozen_destination
      OR OLD.requested_basename IS NOT NEW.requested_basename
    BEGIN
        SELECT RAISE(ABORT, 'immutable Studio request');
    END
    """,
    f"""
    CREATE TRIGGER studio_jobs_v4_contract_shape_on_insert
    BEFORE INSERT ON studio_jobs
    WHEN {_V4_ROW_SHAPE_INVALID_NEW_SQL}
    BEGIN
        SELECT RAISE(ABORT, 'invalid V4 row identity shape');
    END
    """,
    f"""
    CREATE TRIGGER studio_jobs_v4_request_shape_on_insert
    BEFORE INSERT ON studio_jobs
    WHEN NEW.row_contract = 4
     AND (typeof(NEW.request_json) <> 'text' OR json_valid(NEW.request_json) <> 1
       OR typeof(NEW.request_hash) <> 'text' OR length(NEW.request_hash) <> 64
       OR NEW.request_hash GLOB '*[^0-9a-f]*')
    BEGIN
        SELECT RAISE(ABORT, 'invalid V4 request identity');
    END
    """,
    f"""
    CREATE TRIGGER studio_jobs_v4_request_shape_on_update
    BEFORE UPDATE OF project_id, row_contract, destination_identity, published_identity,
        frozen_destination, published_path, request_json, request_hash ON studio_jobs
    WHEN NEW.row_contract = 4
     AND (typeof(NEW.request_json) <> 'text' OR json_valid(NEW.request_json) <> 1
       OR typeof(NEW.request_hash) <> 'text' OR length(NEW.request_hash) <> 64
       OR NEW.request_hash GLOB '*[^0-9a-f]*'
       OR NEW.frozen_destination IS NOT NULL OR NEW.published_path IS NOT NULL)
    BEGIN
        SELECT RAISE(ABORT, 'invalid V4 request identity');
    END
    """,
    f"""
    CREATE TRIGGER studio_jobs_v4_delivery_relation_on_update
    BEFORE UPDATE OF row_contract, state, provider_output_identity_json, published_identity
    ON studio_jobs
    WHEN {_V4_DELIVERY_RELATION_INVALID_NEW_SQL}
    BEGIN
        SELECT RAISE(ABORT, 'invalid V4 published output identity relation');
    END
    """,
    f"""
    CREATE TRIGGER studio_jobs_v4_delivery_transition_on_update
    BEFORE UPDATE OF row_contract, state, provider_output_identity_json, published_identity
    ON studio_jobs
    WHEN {_V4_DELIVERY_TRANSITION_INVALID_NEW_SQL}
    BEGIN
        SELECT RAISE(ABORT, 'V4 delivery identity must be established atomically');
    END
    """,
    """
    CREATE TRIGGER studio_jobs_project_identity_write_once
    BEFORE UPDATE OF project_id ON studio_jobs
    WHEN OLD.row_contract = 4 AND OLD.project_id IS NOT NEW.project_id
    BEGIN
        SELECT RAISE(ABORT, 'project_id is write-once');
    END
    """,
    """
    CREATE TRIGGER studio_jobs_destination_identity_write_once
    BEFORE UPDATE OF destination_identity ON studio_jobs
    WHEN OLD.row_contract = 4 AND OLD.destination_identity IS NOT NEW.destination_identity
    BEGIN
        SELECT RAISE(ABORT, 'destination_identity is write-once');
    END
    """,
    f"""
    CREATE TRIGGER studio_jobs_published_identity_on_insert
    BEFORE INSERT ON studio_jobs
    WHEN NEW.published_identity IS NOT NULL AND NOT ({_PUBLISHED_ID_NEW_SQL})
    BEGIN
        SELECT RAISE(ABORT, 'published_identity must be lowercase 64-hex text');
    END
    """,
    f"""
    CREATE TRIGGER studio_jobs_published_identity_on_update
    BEFORE UPDATE OF published_identity ON studio_jobs
    WHEN NEW.published_identity IS NOT NULL AND NOT ({_PUBLISHED_ID_NEW_SQL})
    BEGIN
        SELECT RAISE(ABORT, 'published_identity must be lowercase 64-hex text');
    END
    """,
    """
    CREATE TRIGGER studio_jobs_published_identity_write_once
    BEFORE UPDATE OF published_identity ON studio_jobs
    WHEN OLD.published_identity IS NOT NULL AND NEW.published_identity IS NOT OLD.published_identity
    BEGIN
        SELECT RAISE(ABORT, 'published_identity is write-once');
    END
    """,
    f"""
    CREATE TRIGGER studio_jobs_contract_shape_on_update
    BEFORE UPDATE OF row_contract, project_id, destination_identity, published_identity,
        frozen_destination, published_path ON studio_jobs
    WHEN NEW.row_contract <> OLD.row_contract
      OR (NEW.row_contract = 3 AND (
          NEW.project_id IS NOT NULL OR NEW.destination_identity IS NOT NULL
          OR NEW.published_identity IS NOT NULL OR NEW.frozen_destination IS NULL
      ))
      OR (NEW.row_contract = 4 AND (
          (NEW.project_id IS NOT NULL AND NOT ({_PROJECT_ID_NEW_SQL}))
          OR NEW.destination_identity IS NULL OR NOT ({_DESTINATION_ID_NEW_SQL})
          OR NEW.frozen_destination IS NOT NULL OR NEW.published_path IS NOT NULL
      ))
    BEGIN
        SELECT RAISE(ABORT, 'invalid Studio row contract shape');
    END
    """,
    """
    CREATE TRIGGER studio_jobs_immutable_id_on_insert
    BEFORE INSERT ON studio_jobs
    WHEN typeof(NEW.studio_job_id) <> 'text'
      OR length(trim(NEW.studio_job_id)) <= 0
    BEGIN
        SELECT RAISE(ABORT, 'studio_job_id must be non-empty text');
    END
    """,
    """
    CREATE TRIGGER studio_jobs_immutable_id
    BEFORE UPDATE OF studio_job_id ON studio_jobs
    WHEN OLD.studio_job_id IS NOT NEW.studio_job_id
    BEGIN
        SELECT RAISE(ABORT, 'studio_job_id is immutable');
    END
    """,
    """
    CREATE TRIGGER studio_jobs_lifecycle_state_on_insert
    BEFORE INSERT ON studio_jobs
    WHEN typeof(NEW.state) <> 'text'
      OR NEW.state NOT IN ('submitting','queued','running','ready','failed','moderated','cancelled')
    BEGIN
        SELECT RAISE(ABORT, 'invalid lifecycle state');
    END
    """,
    f"""
    CREATE TRIGGER studio_jobs_reason_code_on_insert
    BEFORE INSERT ON studio_jobs
    WHEN {_REASON_CODE_INVALID_NEW_SQL}
    BEGIN
        SELECT RAISE(ABORT, 'invalid reason code');
    END
    """,
    f"""
    CREATE TRIGGER studio_jobs_reason_code_on_update
    BEFORE UPDATE OF reason_code ON studio_jobs
    WHEN {_REASON_CODE_INVALID_NEW_SQL}
    BEGIN
        SELECT RAISE(ABORT, 'invalid reason code');
    END
    """,
    f"""
    CREATE TRIGGER studio_jobs_stamp_state_on_insert
    BEFORE INSERT ON studio_jobs
    WHEN {_STAMP_INSERT_STATE_INVALID_NEW_SQL}
    BEGIN
        SELECT RAISE(ABORT, 'invalid stamp state');
    END
    """,
    f"""
    CREATE TRIGGER studio_jobs_stamp_state_invariant_on_update
    BEFORE UPDATE OF stamp_state, stamp_attempt_count ON studio_jobs
    WHEN {_STAMP_STATE_INVALID_NEW_SQL}
    BEGIN
        SELECT RAISE(ABORT, 'invalid stamp state');
    END
    """,
    f"""
    CREATE TRIGGER studio_jobs_lifecycle_monotonic
    BEFORE UPDATE OF state ON studio_jobs
    WHEN typeof(NEW.state) <> 'text'
      OR NOT ({_LIFECYCLE_TRANSITIONS_SQL})
    BEGIN
        SELECT RAISE(ABORT, 'lifecycle state cannot regress');
    END
    """,
    f"""
    CREATE TRIGGER studio_jobs_terminal_reason_write_once
    BEFORE UPDATE OF reason_code ON studio_jobs
    WHEN OLD.state IN ('ready','failed','moderated','cancelled')
     AND NEW.reason_code IS NOT OLD.reason_code
     AND NOT ({_READY_WITHOUT_OUTPUT_REPAIR_SQL})
    BEGIN
        SELECT RAISE(ABORT, 'terminal reason_code is write-once');
    END
    """,
    """
    CREATE TRIGGER studio_jobs_terminal_lifecycle_write_once
    BEFORE UPDATE OF updated_at, queued_at, running_at, completed_at ON studio_jobs
    WHEN OLD.state IN ('ready','failed','moderated','cancelled')
     AND (
        NEW.updated_at IS NOT OLD.updated_at
        OR
        NEW.queued_at IS NOT OLD.queued_at
        OR NEW.running_at IS NOT OLD.running_at
        OR NEW.completed_at IS NOT OLD.completed_at
     )
    BEGIN
        SELECT RAISE(ABORT, 'terminal lifecycle timestamps are write-once');
    END
    """,
    """
    CREATE TRIGGER studio_jobs_provider_job_id_write_once
    BEFORE UPDATE OF provider_job_id ON studio_jobs
    WHEN (
        NEW.row_contract = 4
        AND NEW.provider_job_id IS NOT NULL
        AND (typeof(NEW.provider_job_id) <> 'text' OR length(trim(NEW.provider_job_id)) <= 0)
    ) OR (
        OLD.provider_job_id IS NOT NULL
        AND (NEW.provider_job_id IS NULL OR NEW.provider_job_id IS NOT OLD.provider_job_id)
    )
    BEGIN
        SELECT RAISE(ABORT, 'provider_job_id is write-once');
    END
    """,
    """
    CREATE TRIGGER studio_jobs_provider_output_identity_write_once
    BEFORE UPDATE OF provider_output_identity_json ON studio_jobs
    WHEN OLD.provider_output_identity_json IS NOT NULL
     AND (NEW.provider_output_identity_json IS NULL
          OR NEW.provider_output_identity_json IS NOT OLD.provider_output_identity_json)
    BEGIN
        SELECT RAISE(ABORT, 'provider_output_identity_json is write-once');
    END
    """,
    f"""
    CREATE TRIGGER studio_jobs_provider_output_identity_json_on_insert
    BEFORE INSERT ON studio_jobs
    WHEN {_PUBLICATION_OUTPUT_IDENTITY_INVALID_NEW_SQL}
    BEGIN
        SELECT RAISE(ABORT, 'provider_output_identity_json must be valid non-empty JSON text');
    END
    """,
    f"""
    CREATE TRIGGER studio_jobs_provider_output_identity_json_on_update
    BEFORE UPDATE OF provider_output_identity_json ON studio_jobs
    WHEN {_PUBLICATION_OUTPUT_IDENTITY_INVALID_NEW_SQL}
    BEGIN
        SELECT RAISE(ABORT, 'provider_output_identity_json must be valid non-empty JSON text');
    END
    """,
    f"""
    CREATE TRIGGER studio_jobs_publication_metadata_all_or_none_on_insert
    BEFORE INSERT ON studio_jobs
    WHEN NOT (
        ({_PUBLICATION_METADATA_ALL_NULL_NEW_SQL})
        OR ({_PUBLICATION_METADATA_COMPLETE_NEW_SQL})
    )
    BEGIN
        SELECT RAISE(ABORT, 'publication metadata must be all absent or complete and non-empty');
    END
    """,
    f"""
    CREATE TRIGGER studio_jobs_publication_metadata_all_or_none_on_update
    BEFORE UPDATE OF publication_state, published_path, source_media_sha256, published_file_sha256
    ON studio_jobs
    WHEN NOT (
        ({_PUBLICATION_METADATA_ALL_NULL_NEW_SQL})
        OR ({_PUBLICATION_METADATA_COMPLETE_NEW_SQL})
    )
    BEGIN
        SELECT RAISE(ABORT, 'publication metadata must be all absent or complete and non-empty');
    END
    """,
    f"""
    CREATE TRIGGER studio_jobs_published_metadata_on_insert
    BEFORE INSERT ON studio_jobs
    WHEN NEW.publication_state = 'published'
     AND NOT ({_PUBLICATION_METADATA_COMPLETE_NEW_SQL})
    BEGIN
        SELECT RAISE(ABORT, 'published jobs require complete non-empty publication metadata');
    END
    """,
    f"""
    CREATE TRIGGER studio_jobs_publication_lease_on_insert
    BEFORE INSERT ON studio_jobs
    WHEN NEW.publication_state = 'publishing'
     AND NOT ({_PUBLICATION_PUBLISHING_ELIGIBLE_NEW_SQL})
    BEGIN
        SELECT RAISE(ABORT, 'publishing jobs require ready output and a complete finite publication lease');
    END
    """,
    f"""
    CREATE TRIGGER studio_jobs_publication_lease_on_update
    BEFORE UPDATE OF publication_state, state, provider_output_identity_json,
        publication_lease_owner, publication_lease_token, publication_lease_expires_at
    ON studio_jobs
    WHEN NEW.publication_state = 'publishing'
     AND NOT ({_PUBLICATION_PUBLISHING_ELIGIBLE_NEW_SQL})
    BEGIN
        SELECT RAISE(ABORT, 'publishing jobs require ready output and a complete finite publication lease');
    END
    """,
    """
    CREATE TRIGGER studio_jobs_publication_state_on_insert
    BEFORE INSERT ON studio_jobs
    WHEN typeof(NEW.publication_state) <> 'text'
      OR NEW.publication_state NOT IN ('unpublished','publishing','published')
    BEGIN
        SELECT RAISE(ABORT, 'invalid publication state');
    END
    """,
    """
    CREATE TRIGGER studio_jobs_publication_state_on_update
    BEFORE UPDATE OF publication_state ON studio_jobs
    WHEN typeof(NEW.publication_state) <> 'text'
      OR NEW.publication_state NOT IN ('unpublished','publishing','published')
    BEGIN
        SELECT RAISE(ABORT, 'invalid publication state');
    END
    """,
    f"""
    CREATE TRIGGER studio_jobs_publication_state_invariant_on_insert
    BEFORE INSERT ON studio_jobs
    WHEN {_PUBLICATION_STATE_INVARIANT_INVALID_NEW_SQL}
    BEGIN
        SELECT RAISE(ABORT, 'invalid publication state fields');
    END
    """,
    f"""
    CREATE TRIGGER studio_jobs_publication_state_invariant_on_update
    BEFORE UPDATE OF publication_state, publication_lease_owner, publication_lease_token,
        publication_lease_expires_at, published_path, source_media_sha256, published_file_sha256
    ON studio_jobs
    WHEN {_PUBLICATION_STATE_INVARIANT_INVALID_NEW_SQL}
    BEGIN
        SELECT RAISE(ABORT, 'invalid publication state fields');
    END
    """,
    f"""
    CREATE TRIGGER studio_jobs_publication_published_on_insert
    BEFORE INSERT ON studio_jobs
    WHEN NEW.publication_state IS NOT 'unpublished'
    BEGIN
        SELECT RAISE(ABORT, 'publication rows must start unpublished');
    END
    """,
    f"""
    CREATE TRIGGER studio_jobs_publication_published_on_update
    BEFORE UPDATE OF publication_state, state, provider_output_identity_json,
        publication_lease_owner, publication_lease_token, publication_lease_expires_at,
        published_path, source_media_sha256, published_file_sha256
    ON studio_jobs
    WHEN NEW.publication_state = 'published'
     AND OLD.publication_state <> 'published'
     AND NOT (
        NEW.state = 'ready'
        AND ({_PUBLICATION_OUTPUT_IDENTITY_COMPLETE_NEW_SQL})
        AND ({_PUBLICATION_METADATA_COMPLETE_NEW_SQL})
        AND OLD.publication_state = 'publishing'
        AND ({_PUBLICATION_LEASE_COMPLETE_OLD_SQL})
        AND (
            ({_PUBLICATION_LEASE_ALL_NULL_NEW_SQL})
            OR (
                ({_PUBLICATION_LEASE_COMPLETE_NEW_SQL})
                AND ({_PUBLICATION_LEASE_MATCH_OLD_NEW_SQL})
            )
        )
    )
    BEGIN
        SELECT RAISE(ABORT, 'published jobs require ready output and a prior publication lease');
    END
    """,
    """
    CREATE TRIGGER studio_jobs_publication_metadata_write_once
    BEFORE UPDATE OF published_path, source_media_sha256, published_file_sha256
    ON studio_jobs
    WHEN OLD.publication_state = 'published'
     AND (NEW.published_path IS NOT OLD.published_path
          OR NEW.source_media_sha256 IS NOT OLD.source_media_sha256
          OR NEW.published_file_sha256 IS NOT OLD.published_file_sha256)
    BEGIN
        SELECT RAISE(ABORT, 'published publication metadata is write-once');
    END
    """,
    f"""
    CREATE TRIGGER studio_jobs_publication_terminal_write_once
    BEFORE UPDATE OF publication_state, state, provider_output_identity_json,
        publication_lease_owner, publication_lease_token, publication_lease_expires_at,
        published_path, source_media_sha256, published_file_sha256
    ON studio_jobs
    WHEN OLD.publication_state = 'published'
     AND NOT (
        NEW.publication_state = 'published'
        AND NEW.state = 'ready'
        AND ({_PUBLICATION_OUTPUT_IDENTITY_COMPLETE_NEW_SQL})
        AND ({_PUBLICATION_METADATA_COMPLETE_NEW_SQL})
        AND NEW.publication_lease_owner IS OLD.publication_lease_owner
        AND NEW.publication_lease_token IS OLD.publication_lease_token
        AND NEW.publication_lease_expires_at IS OLD.publication_lease_expires_at
     )
    BEGIN
        SELECT RAISE(ABORT, 'published publication state is terminal');
    END
    """,
    """
    CREATE TRIGGER studio_jobs_stamp_state_machine
    BEFORE UPDATE OF stamp_state, stamp_attempt_count ON studio_jobs
    WHEN typeof(NEW.stamp_state) <> 'text'
      OR typeof(NEW.stamp_attempt_count) <> 'integer'
      OR NOT (
        (
            NEW.stamp_state IS OLD.stamp_state
            AND NEW.stamp_attempt_count IS OLD.stamp_attempt_count
        )
        OR (
            NEW.stamp_state = 'pending'
            AND OLD.stamp_state IN ('not_started','pending','failed')
            AND NEW.stamp_attempt_count = OLD.stamp_attempt_count + 1
        )
        OR (
            OLD.stamp_state = 'pending'
            AND NEW.stamp_state IN ('succeeded','failed')
            AND NEW.stamp_attempt_count = OLD.stamp_attempt_count
        )
      )
    BEGIN
        SELECT RAISE(ABORT, 'invalid stamp state transition');
    END
    """,
    f"""
    CREATE TRIGGER studio_jobs_publication_path_confinement_on_update
    BEFORE UPDATE OF published_path ON studio_jobs
    WHEN OLD.publication_state = 'publishing'
     AND NEW.publication_state = 'published'
    BEGIN
        SELECT RAISE(ABORT, 'published_path must be inside frozen_destination')
        WHERE {_SQLITE_PUBLISHED_PATH_CHILD_FUNCTION}(NEW.frozen_destination, NEW.published_path) <> 1;
    END
    """,
)
_DURABILITY_TRIGGERS = "\n".join(f"{statement};" for statement in _DURABILITY_TRIGGER_STATEMENTS)

_CREATE_TABLE_SCHEMA = f"""
CREATE TABLE studio_jobs (
    studio_job_id TEXT PRIMARY KEY NOT NULL CHECK (typeof(studio_job_id) = 'text' AND length(trim(studio_job_id)) > 0),
    client_request_id TEXT NOT NULL UNIQUE,
    provider_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    provider_job_id TEXT,
    connection_id TEXT NOT NULL,
    request_json TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    frozen_destination TEXT,
    requested_basename TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('submitting','queued','running','ready','failed','moderated','cancelled')),
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    queued_at REAL,
    running_at REAL,
    completed_at REAL,
    reason_code TEXT,
    provider_output_identity_json TEXT,
    publication_state TEXT NOT NULL DEFAULT 'unpublished' CHECK (publication_state IN ('unpublished','publishing','published')),
    publication_lease_owner TEXT,
    publication_lease_token TEXT,
    publication_lease_expires_at REAL,
    published_path TEXT,
    source_media_sha256 TEXT,
    published_file_sha256 TEXT,
    stamp_state TEXT NOT NULL DEFAULT 'not_started' CHECK (stamp_state IN ('not_started','pending','succeeded','failed')),
    stamp_attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (stamp_attempt_count >= 0),
    project_id TEXT,
    row_contract INTEGER NOT NULL CHECK (typeof(row_contract) = 'integer' AND row_contract IN (3, 4)),
    destination_identity TEXT,
    published_identity TEXT,
    CHECK (project_id IS NULL OR ({_PROJECT_ID_SQL.format(field='project_id')})),
    CHECK (destination_identity IS NULL OR ({_DESTINATION_ID_SQL})),
    CHECK (published_identity IS NULL OR ({_PUBLISHED_ID_SQL})),
    CHECK (
        row_contract = 3
        OR (
            row_contract = 4
            AND destination_identity IS NOT NULL
            AND frozen_destination IS NULL
            AND published_path IS NULL
            AND (project_id IS NULL OR ({_PROJECT_ID_SQL.format(field='project_id')}))
            AND ({_DESTINATION_ID_SQL})
            AND (published_identity IS NULL OR ({_PUBLISHED_ID_SQL}))
            AND typeof(client_request_id) = 'text' AND length(trim(client_request_id)) > 0
            AND typeof(provider_id) = 'text' AND length(trim(provider_id)) > 0
            AND typeof(target_id) = 'text' AND length(trim(target_id)) > 0
            AND typeof(connection_id) = 'text' AND length(trim(connection_id)) > 0
            AND typeof(request_json) = 'text' AND json_valid(request_json) = 1 AND json_type(request_json) = 'object'
            AND ({_REQUEST_HASH_SQL})
            AND typeof(requested_basename) = 'text' AND length(trim(requested_basename)) > 0
            AND instr(requested_basename, '/') = 0 AND instr(requested_basename, char(92)) = 0
            AND requested_basename GLOB '*.mp4'
            AND requested_basename NOT GLOB '*[. ]'
        )
    ),
    CHECK (
        row_contract = 4
        OR (
            row_contract = 3
            AND project_id IS NULL
            AND destination_identity IS NULL
            AND published_identity IS NULL
            AND frozen_destination IS NOT NULL
        )
    ),
    CHECK (
        provider_output_identity_json IS NULL
        OR ({_PUBLICATION_OUTPUT_IDENTITY_COMPLETE_SQL})
    ),
    CHECK (NOT ({_V4_DELIVERY_RELATION_INVALID_SQL})),
    CHECK (NOT ({_REASON_CODE_INVALID_SQL})),
    CHECK (NOT ({_STAMP_STATE_INVALID_SQL})),
    CHECK (
        ({_PUBLICATION_METADATA_ALL_NULL_SQL})
        OR ({_PUBLICATION_METADATA_COMPLETE_SQL})
    ),
    CHECK (
        publication_state <> 'published'
        OR ({_PUBLICATION_METADATA_COMPLETE_SQL})
    ),
    CHECK (
        publication_state <> 'unpublished'
        OR (
            ({_PUBLICATION_METADATA_ALL_NULL_SQL})
            AND ({_PUBLICATION_LEASE_ALL_NULL_SQL})
        )
    ),
    CHECK (
        publication_state <> 'publishing'
        OR (
            ({_PUBLICATION_METADATA_ALL_NULL_SQL})
            AND ({_PUBLICATION_PUBLISHING_ELIGIBLE_SQL})
        )
    )
)
"""

_CREATE_SCHEMA = _CREATE_TABLE_SCHEMA + f"""
;
CREATE UNIQUE INDEX IF NOT EXISTS studio_jobs_provider_identity
    ON studio_jobs(provider_id, provider_job_id)
    WHERE provider_job_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS studio_jobs_one_active_comfy_connection
    ON studio_jobs(provider_id, connection_id)
    WHERE provider_id = 'comfyui-local' AND state IN ('submitting','queued','running');
CREATE INDEX IF NOT EXISTS studio_jobs_state_index ON studio_jobs(state);
{_DURABILITY_TRIGGERS}
"""
_SCHEMA_INDEX_STATEMENTS = (
    ("studio_jobs_provider_identity", """
        CREATE UNIQUE INDEX studio_jobs_provider_identity
        ON studio_jobs(provider_id, provider_job_id)
        WHERE provider_job_id IS NOT NULL
    """),
    ("studio_jobs_one_active_comfy_connection", """
        CREATE UNIQUE INDEX studio_jobs_one_active_comfy_connection
        ON studio_jobs(provider_id, connection_id)
        WHERE provider_id = 'comfyui-local' AND state IN ('submitting','queued','running')
    """),
    ("studio_jobs_state_index", "CREATE INDEX studio_jobs_state_index ON studio_jobs(state)"),
)
_FRESH_SCHEMA_STATEMENTS = (
    ("studio_jobs_table", _CREATE_TABLE_SCHEMA),
    *_SCHEMA_INDEX_STATEMENTS,
    *((f"durability_{index:02d}", statement) for index, statement in enumerate(_DURABILITY_TRIGGER_STATEMENTS)),
)



def _normalize_transition_reason(reason_code: Any) -> str | None:
    """Validate and normalize a lifecycle reason before finite-set membership."""
    if reason_code is None:
        return None
    if not isinstance(reason_code, str):
        raise _invalid("reason_code must be text")
    try:
        hash(reason_code)
        normalized = str.casefold(str.strip(reason_code))
        hash(normalized)
    except (TypeError, ValueError, AttributeError, RecursionError) as exc:
        raise _invalid("reason_code must be hashable text") from exc
    if normalized not in REASON_CODES:
        raise _invalid("reason_code is not recognized")
    return normalized


def _validate_windows_path_component(component: str, field_name: str) -> None:
    """Apply the same lexical rules to every non-root Windows path component."""
    if (
        not component
        or component in {".", ".."}
        or component[-1] in ". "
        or any(character in _WINDOWS_INVALID_BASENAME_CHARS for character in component)
        or any(ord(character) < 32 for character in component)
        or component.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_BASENAMES
    ):
        raise _invalid(f"{field_name} contains an invalid Windows path component")


def _validate_windows_drive_components(value: str, field_name: str) -> None:
    """Validate non-root components without resolving or touching the filesystem."""
    drive, tail = ntpath.splitdrive(value)
    if not drive or len(drive) != 2 or drive[1] != ":" or not tail.startswith("\\"):
        return
    components = tail.replace("/", "\\").lstrip("\\").split("\\")
    if components and components[-1] == "":
        components.pop()
    if any(component == "" for component in components):
        raise _invalid(f"{field_name} contains an invalid Windows path component")
    for component in components:
        _validate_windows_path_component(component, field_name)


def _invalid(message: str | None = None) -> InvalidRequestError:
    return InvalidRequestError(message)


def _validate_json(
    value: Any,
    field_name: str = "request",
    _active_containers: set[int] | None = None,
) -> Any:
    """Return plain JSON containers without coercing keys, numbers, or strings."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _invalid(f"{field_name} contains a non-finite JSON number")
        return value
    active_containers = _active_containers if _active_containers is not None else set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active_containers:
            raise _invalid(f"{field_name} contains a cyclic container")
        active_containers.add(identity)
        try:
            result: dict[str, Any] = {}
            for key, nested in value.items():
                if not isinstance(key, str):
                    raise _invalid(f"{field_name} object keys must be text")
                result[key] = _validate_json(nested, f"{field_name}.{key}", active_containers)
            return result
        finally:
            active_containers.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active_containers:
            raise _invalid(f"{field_name} contains a cyclic container")
        active_containers.add(identity)
        try:
            return [
                _validate_json(nested, f"{field_name}[{index}]", active_containers)
                for index, nested in enumerate(value)
            ]
        finally:
            active_containers.remove(identity)
    raise _invalid(f"{field_name} contains an unsupported value")


def canonical_request_json(request: GenerationRequest | Mapping[str, Any]) -> str:
    """Encode an already validated request with stable bytes and no rewriting."""
    try:
        if isinstance(request, GenerationRequest):
            value: Any = request.to_dict()
        elif isinstance(request, Mapping):
            value = request
        else:
            raise _invalid()
        normalized = _validate_json(value)
        encoded = json.dumps(
            normalized,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        encoded.encode("utf-8")
    except RecursionError as exc:
        raise _invalid() from exc
    except (TypeError, UnicodeError, ValueError, OverflowError) as exc:
        raise _invalid() from exc
    return encoded


def canonical_request_bytes(request: GenerationRequest | Mapping[str, Any]) -> bytes:
    return canonical_request_json(request).encode("utf-8")


def canonical_request_hash(request: GenerationRequest | Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_request_bytes(request)).hexdigest()


def _normalized_lexical_path(value: str, field_name: str) -> tuple[str, str]:
    """Return a normalized path and its lexical platform family.

    Publication authorization is deliberately lexical: it must not touch the
    filesystem, follow links, or depend on whether a destination exists yet.
    Windows paths are normalized with Windows rules and compared case-folded.
    Native absolute POSIX paths remain usable for portable repository tests.
    """
    if "\x00" in value:
        raise _invalid(f"{field_name} must not contain NUL bytes")
    if any(part == ".." for part in value.replace("/", "\\").split("\\")):
        raise _invalid(f"{field_name} must not contain traversal")
    windows_value = value.replace("/", "\\")
    if windows_value.startswith("\\\\"):
        raise _invalid(f"{field_name} must be an absolute local path")
    drive, tail = ntpath.splitdrive(windows_value)
    if drive:
        if (
            len(drive) != 2
            or not drive[0].isalpha()
            or drive[1] != ":"
            or not tail.startswith("\\")
        ):
            raise _invalid(f"{field_name} must be an absolute local path")
        _validate_windows_drive_components(windows_value, field_name)
        return "windows", ntpath.normpath(windows_value).casefold()
    if not value.startswith("/") or value.startswith("//"):
        raise _invalid(f"{field_name} must be an absolute local path")
    return "posix", os.path.normpath(value)


def _ensure_published_path_is_child(frozen_destination: str, published_path: str) -> None:
    """Require a published path to be a non-directory lexical child."""
    if published_path.endswith(("/", "\\")):
        raise _invalid("published_path must name a file child")
    destination_kind, normalized_destination = _normalized_lexical_path(
        frozen_destination,
        "frozen_destination",
    )
    published_kind, normalized_published = _normalized_lexical_path(
        published_path,
        "published_path",
    )
    if destination_kind != published_kind or normalized_destination == normalized_published:
        raise _invalid("published_path must be inside frozen_destination")
    try:
        if destination_kind == "windows":
            common = ntpath.commonpath((normalized_destination, normalized_published))
        else:
            common = os.path.commonpath((normalized_destination, normalized_published))
    except ValueError as exc:
        raise _invalid("published_path must be inside frozen_destination") from exc
    if common != normalized_destination:
        raise _invalid("published_path must be inside frozen_destination")


def _sqlite_published_path_is_child(frozen_destination: Any, published_path: Any) -> int:
    """Return SQLite truth for the lexical publication-child invariant."""
    if not isinstance(frozen_destination, str) or not isinstance(published_path, str):
        return 0
    try:
        _ensure_published_path_is_child(frozen_destination, published_path)
    except (AttributeError, InvalidRequestError, OverflowError, RecursionError, TypeError, ValueError):
        return 0
    return 1


class JobStore:
    """SQLite-backed ledger whose admission and publication decisions are durable."""

    def __init__(self, path: str | os.PathLike[str] | None = None, *, clock: Callable[[], float] | None = None) -> None:
        self.path = Path(path) if path is not None else state_root(Path(get_hermes_home()) / "plugin-data") / "generation-jobs.sqlite3"
        self.path = self.path.expanduser()
        self._clock = clock or time.time
        self._closed = False
        self._ensure_parent()
        self._migrate()

    def _ensure_parent(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass

    def _connect(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("JobStore is closed")
        connection = sqlite3.connect(
            self.path,
            timeout=10.0,
            isolation_level=None,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.create_function(
            _SQLITE_PUBLISHED_PATH_CHILD_FUNCTION,
            2,
            _sqlite_published_path_is_child,
            deterministic=True,
        )
        journal_mode = str(connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
        if journal_mode != "wal":
            connection.close()
            raise SchemaError("Studio ledger requires WAL journal mode")
        connection.execute("PRAGMA synchronous=FULL")
        if int(connection.execute("PRAGMA synchronous").fetchone()[0]) != 2:
            connection.close()
            raise SchemaError("Studio ledger requires FULL synchronous mode")
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA foreign_keys=ON")
        if int(connection.execute("PRAGMA foreign_keys").fetchone()[0]) != 1:
            connection.close()
            raise SchemaError("Studio ledger requires foreign keys")
        return connection

    @staticmethod
    def _install_durability_triggers(connection: sqlite3.Connection) -> None:
        for statement in _DURABILITY_TRIGGER_STATEMENTS:
            connection.execute(statement)

    @staticmethod
    def _execute_contract3_insert(
        connection: sqlite3.Connection,
        statement: str,
        parameters: tuple[Any, ...],
    ) -> sqlite3.Cursor:
        """Run the legacy adapter insert while keeping raw SQL fail-closed.

        SQLite has no connection-local trigger authorization primitive.  The
        adapter therefore performs the already-validated Contract-3 insert in
        the same IMMEDIATE transaction in which it drops and restores only the
        contract gate.  Other writers cannot commit during this transaction,
        and rollback restores the dropped trigger as well as the row.
        """
        connection.execute("DROP TRIGGER IF EXISTS studio_jobs_contract_on_insert")
        try:
            return connection.execute(statement, parameters)
        finally:
            connection.execute(_CONTRACT_INSERT_TRIGGER_STATEMENT)

    @staticmethod
    def _validate_migration_published_paths(connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            """
            SELECT frozen_destination, published_path
            FROM studio_jobs
            WHERE published_path IS NOT NULL
            """
        ).fetchall()
        for row in rows:
            try:
                _ensure_published_path_is_child(row["frozen_destination"], row["published_path"])
            except (AttributeError, InvalidRequestError, OverflowError, RecursionError, TypeError, ValueError) as exc:
                raise SchemaError(
                    "cannot migrate Studio ledger with published_path outside frozen_destination"
                ) from exc

    @staticmethod
    def _validate_migration_durable_state(connection: sqlite3.Connection) -> None:
        invalid = connection.execute(
            f"SELECT 1 FROM studio_jobs WHERE {_PUBLICATION_STATE_INVALID_SQL} LIMIT 1"
        ).fetchone()
        if invalid is not None:
            raise SchemaError("cannot migrate Studio ledger with invalid publication state")
        JobStore._validate_migration_published_paths(connection)

    def _rebuild_legacy_to_v4(self, connection: sqlite3.Connection) -> None:
        """Rebuild a V1/V2/V3 table atomically while preserving legacy bytes."""
        self._validate_migration_durable_state(connection)
        connection.execute(_CREATE_TABLE_SCHEMA.replace("CREATE TABLE studio_jobs", "CREATE TABLE studio_jobs_v4", 1))
        connection.execute(
            """
            INSERT INTO studio_jobs_v4 (
                studio_job_id, client_request_id, provider_id, target_id, provider_job_id,
                connection_id, request_json, request_hash, project_id, row_contract,
                destination_identity, published_identity, frozen_destination, requested_basename,
                state, created_at, updated_at, queued_at, running_at, completed_at, reason_code,
                provider_output_identity_json, publication_state, publication_lease_owner,
                publication_lease_token, publication_lease_expires_at, published_path,
                source_media_sha256, published_file_sha256, stamp_state, stamp_attempt_count
            )
            SELECT
                studio_job_id, client_request_id, provider_id, target_id, provider_job_id,
                connection_id, request_json, request_hash, NULL, 3,
                NULL, NULL, frozen_destination, requested_basename,
                state, created_at, updated_at, queued_at, running_at, completed_at, reason_code,
                provider_output_identity_json, publication_state, publication_lease_owner,
                publication_lease_token, publication_lease_expires_at, published_path,
                source_media_sha256, published_file_sha256, stamp_state, stamp_attempt_count
            FROM studio_jobs
            """
        )
        connection.execute("DROP TABLE studio_jobs")
        connection.execute("ALTER TABLE studio_jobs_v4 RENAME TO studio_jobs")
        connection.execute(
            "CREATE UNIQUE INDEX studio_jobs_provider_identity ON studio_jobs(provider_id, provider_job_id) WHERE provider_job_id IS NOT NULL"
        )
        connection.execute(
            "CREATE UNIQUE INDEX studio_jobs_one_active_comfy_connection ON studio_jobs(provider_id, connection_id) WHERE provider_id = 'comfyui-local' AND state IN ('submitting','queued','running')"
        )
        connection.execute("CREATE INDEX studio_jobs_state_index ON studio_jobs(state)")
        self._install_durability_triggers(connection)
        connection.execute("PRAGMA user_version=4")

    def _migrate_v1_to_v2(self, connection: sqlite3.Connection) -> None:
        # Retained as a named compatibility hook for older callers; Task 6
        # performs the complete V1->V4 rebuild in one outer transaction.
        self._validate_migration_durable_state(connection)

    def _migrate_v2_to_v3(self, connection: sqlite3.Connection) -> None:
        self._validate_migration_durable_state(connection)

    def _migrate(self) -> None:
        connection = self._connect()
        try:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise SchemaError(f"unsupported Studio ledger schema version: {version}")
            if version == 0:
                connection.executescript(
                    "BEGIN IMMEDIATE;\n"
                    + _CREATE_SCHEMA
                    + f"\nPRAGMA user_version={SCHEMA_VERSION};\nCOMMIT;"
                )
            elif version in {1, 2, 3}:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    self._rebuild_legacy_to_v4(connection)
                    connection.execute("COMMIT")
                except Exception:
                    if connection.in_transaction:
                        connection.execute("ROLLBACK")
                    raise
            elif version == SCHEMA_VERSION:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    self._validate_migration_durable_state(connection)
                    self._install_durability_triggers(connection)
                    connection.execute("COMMIT")
                except Exception:
                    if connection.in_transaction:
                        connection.execute("ROLLBACK")
                    raise
        finally:
            connection.close()
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def close(self) -> None:
        self._closed = True

    def _now(self, now: float | None = None) -> float:
        value = self._clock() if now is None else now
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _invalid("ledger time must be a finite number")
        try:
            normalized = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise _invalid("ledger time must be a finite number") from exc
        if not math.isfinite(normalized):
            raise _invalid("ledger time must be a finite number")
        return normalized

    @staticmethod
    def _text(value: Any, field_name: str) -> str:
        if not isinstance(value, str):
            raise _invalid(f"{field_name} must be non-empty text")
        try:
            hash(value)
        except TypeError as exc:
            raise _invalid(f"{field_name} must be hashable text") from exc
        if not value.strip() or "\x00" in value:
            raise _invalid(f"{field_name} must be non-empty text")
        try:
            value.encode("utf-8")
        except UnicodeError as exc:
            raise _invalid(f"{field_name} must be valid text") from exc
        return value

    @staticmethod
    def _state(value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise _invalid("state must be non-empty text")
        try:
            hash(value)
        except TypeError as exc:
            raise _invalid("state must be hashable text") from exc
        return value

    @staticmethod
    def _basename(value: Any) -> str:
        value = JobStore._text(value, "requested_basename")
        try:
            hash(value)
        except TypeError as exc:
            raise _invalid("requested_basename must be hashable text") from exc
        if (
            value in {".", ".."}
            or value[-1] in ". "
            or any(character in _WINDOWS_INVALID_BASENAME_CHARS for character in value)
            or any(ord(character) < 32 for character in value)
            or value.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_BASENAMES
        ):
            raise _invalid()
        return value

    @staticmethod
    def _path_text(value: Any, field_name: str) -> str:
        if isinstance(value, os.PathLike):
            try:
                value = os.fspath(value)
            except (TypeError, ValueError, OverflowError, RecursionError) as exc:
                raise _invalid(f"{field_name} must be valid text") from exc
        value = JobStore._text(value, field_name)
        _normalized_lexical_path(value, field_name)
        return value

    @staticmethod
    def _coerce_request(request: GenerationRequest | Mapping[str, Any]) -> GenerationRequest:
        if isinstance(request, GenerationRequest):
            result = request
        elif isinstance(request, Mapping):
            try:
                result = GenerationRequest(**dict(request))
            except (TypeError, ValueError, OverflowError, RecursionError) as exc:
                raise _invalid() from exc
        else:
            raise _invalid()
        if result.schema != GENERATION_REQUEST_SCHEMA:
            raise _invalid()
        JobStore._text(result.client_request_id, "client_request_id")
        JobStore._text(result.provider_id, "provider_id")
        JobStore._text(result.target_id, "target_id")
        if result.save_to is not None:
            JobStore._path_text(result.save_to, "save_to")
        for root in result.approved_roots:
            JobStore._path_text(root, "approved_root")
        return result

    def create_or_get_job(
        self,
        request: GenerationRequest | Mapping[str, Any],
        *,
        provider_job_id: str | None = None,
        connection_id: str | None = None,
        frozen_destination: str | os.PathLike[str] | None = None,
        requested_basename: str | None = None,
        state: str = "queued",
        now: float | None = None,
    ) -> JobRecord:
        request = self._coerce_request(request)
        state = self._state(state)
        if state not in _STATES:
            raise _invalid()
        canonical = canonical_request_json(request)
        request_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        provider_job_id = None if provider_job_id is None else self._text(provider_job_id, "provider_job_id")
        connection_id = request.provider_id if connection_id is None else connection_id
        connection_id = self._text(connection_id, "connection_id")
        destination_value = DEFAULT_DESTINATION if frozen_destination is None else frozen_destination
        destination = self._path_text(destination_value, "frozen_destination")
        basename = self._basename(DEFAULT_BASENAME if requested_basename is None else requested_basename)
        timestamp = self._now(now)
        queued_at = timestamp if state in _NONTERMINAL_STATES else None
        running_at = timestamp if state == "running" else None
        completed_at = timestamp if state in _TERMINAL_STATES else None
        studio_job_id = f"studio:{uuid.uuid4()}"
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM studio_jobs WHERE client_request_id = ?",
                (request.client_request_id,),
            ).fetchone()
            if existing is not None:
                self._assert_contract3_row(existing)
                if (
                    existing["request_json"] != canonical
                    or existing["request_hash"] != request_hash
                    or existing["provider_id"] != request.provider_id
                    or existing["target_id"] != request.target_id
                    or existing["provider_job_id"] != provider_job_id
                    or existing["connection_id"] != connection_id
                    or existing["frozen_destination"] != destination
                    or existing["requested_basename"] != basename
                ):
                    raise _invalid()
                connection.execute("COMMIT")
                return self._row_to_record(existing)
            try:
                self._execute_contract3_insert(
                    connection,
                    """
                    INSERT INTO studio_jobs (
                        studio_job_id, client_request_id, provider_id, target_id,
                        provider_job_id, connection_id, request_json, request_hash,
                        project_id, row_contract, destination_identity, published_identity,
                        frozen_destination, requested_basename, state, created_at,
                        updated_at, queued_at, running_at, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, 3, NULL, NULL, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        studio_job_id,
                        request.client_request_id,
                        request.provider_id,
                        request.target_id,
                        provider_job_id,
                        connection_id,
                        canonical,
                        request_hash,
                        destination,
                        basename,
                        state,
                        timestamp,
                        timestamp,
                        queued_at,
                        running_at,
                        completed_at,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                active = connection.execute(
                    """
                    SELECT 1 FROM studio_jobs
                    WHERE provider_id = 'comfyui-local'
                      AND connection_id = ?
                      AND state IN ('submitting','queued','running')
                    LIMIT 1
                    """,
                    (connection_id,),
                ).fetchone()
                if active is not None:
                    raise ProviderBusyError() from exc
                raise _invalid() from exc
            created = connection.execute("SELECT * FROM studio_jobs WHERE studio_job_id = ?", (studio_job_id,)).fetchone()
            connection.execute("COMMIT")
            self._chmod_private_file()
            return self._row_to_record(created)
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    create_job = create_or_get_job

    @staticmethod
    def _v4_project_id(value: Any, field_name: str = "project_id") -> str | None:
        if value is None:
            return None
        if type(value) is not str or re.fullmatch(r"prj_[0-9a-f]{32}", value) is None:
            raise _invalid(f"{field_name} must be prj_<32 lowercase hex> or null")
        return value

    @staticmethod
    def _v4_hex(value: Any, field_name: str) -> str:
        if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise _invalid(f"{field_name} must be lowercase 64-hex text")
        return value

    @staticmethod
    def _v4_request_identity(request_json: Any, request_hash: Any) -> tuple[str, str]:
        if type(request_json) is not str or not request_json.strip():
            raise _invalid("request_json must be non-empty JSON text")
        try:
            parsed = json.loads(request_json, parse_constant=JobStore._reject_json_constant)
        except (TypeError, UnicodeError, ValueError, OverflowError, RecursionError) as exc:
            raise _invalid("request_json must be valid JSON") from exc
        if not isinstance(parsed, dict):
            raise _invalid("request_json must be a JSON object")
        normalized_hash = JobStore._v4_hex(request_hash, "request_hash")
        actual_hash = hashlib.sha256(request_json.encode("utf-8")).hexdigest()
        if actual_hash != normalized_hash:
            raise _invalid("request_hash does not match request_json")
        return request_json, normalized_hash

    @staticmethod
    def _v4_basename(value: Any) -> str:
        normalized = JobStore._basename(value)
        if not normalized.endswith(".mp4"):
            raise _invalid("requested_basename must name an MP4 file")
        return normalized

    @staticmethod
    def _assert_v4_scope(
        row: sqlite3.Row,
        expected_project_id: Any,
        *,
        field_present: bool = True,
    ) -> None:
        JobStore._validate_persisted_row(row)
        if JobStore._row_contract(row["row_contract"]) != 4:
            raise _invalid("operation requires a Contract-4 ledger row")
        expected = JobStore._v4_project_id(expected_project_id)
        if not field_present or row["project_id"] != expected:
            raise ProjectScopeMismatchError()

    @staticmethod
    def _assert_contract3_row(
        row: sqlite3.Row,
        *,
        check_output_identity: bool = True,
    ) -> None:
        """Reject V4 rows at every legacy API boundary before any side effect."""
        if JobStore._row_contract(row["row_contract"]) != 3:
            raise ProjectScopeError()
        JobStore._validate_persisted_row(row, check_output_identity=check_output_identity)

    @staticmethod
    def _scoped_fields_match(row: sqlite3.Row, values: Mapping[str, Any]) -> bool:
        return all(row[field_name] == value for field_name, value in values.items())

    @staticmethod
    def _v4_delivery_identity(
        published_identity: Any,
        output_sha256: Any,
        output_bytes: Any,
    ) -> tuple[str, dict[str, Any]]:
        published_identity = JobStore._v4_hex(published_identity, "published_identity")
        output_sha256 = JobStore._v4_hex(output_sha256, "output_sha256")
        if isinstance(output_bytes, bool) or type(output_bytes) is not int or output_bytes < 0:
            raise _invalid("output_bytes must be a non-negative integer")
        if output_bytes > 9223372036854775807:
            raise _invalid("output_bytes exceeds the SQLite integer range")
        identity = {
            "bytes": output_bytes,
            "published_identity": published_identity,
            "sha256": output_sha256,
        }
        return canonical_request_json(identity), identity

    @staticmethod
    def _parse_v4_delivery_identity(value: Any) -> dict[str, Any]:
        if type(value) is not str or not value.strip():
            raise SchemaError("invalid V4 provider output identity")
        try:
            parsed = json.loads(value, parse_constant=JobStore._reject_json_constant)
        except (TypeError, UnicodeError, ValueError, OverflowError, RecursionError) as exc:
            raise SchemaError("invalid V4 provider output identity") from exc
        if type(parsed) is not dict or set(parsed) != {"bytes", "published_identity", "sha256"}:
            raise SchemaError("invalid V4 provider output identity")
        if (
            isinstance(parsed["bytes"], bool)
            or type(parsed["bytes"]) is not int
            or parsed["bytes"] < 0
            or parsed["bytes"] > 9223372036854775807
            or type(parsed["published_identity"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", parsed["published_identity"]) is None
            or type(parsed["sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", parsed["sha256"]) is None
        ):
            raise SchemaError("invalid V4 provider output identity")
        try:
            canonical = canonical_request_json(parsed)
        except InvalidRequestError as exc:
            raise SchemaError("invalid V4 provider output identity") from exc
        if value != canonical:
            raise SchemaError("invalid V4 provider output identity")
        return parsed

    @staticmethod
    def _assert_expected_v4_delivery_identity(
        value: Any,
        expected_json: str,
        expected_identity: Mapping[str, Any],
    ) -> None:
        try:
            parsed = JobStore._parse_v4_delivery_identity(value)
        except SchemaError as exc:
            raise _invalid("provider output identity is not a V4 delivery identity") from exc
        if value != expected_json or parsed != dict(expected_identity):
            raise _invalid("provider output identity does not match the V4 delivery")

    def create_or_get_v4(
        self,
        *,
        studio_job_id: str,
        client_request_id: str,
        provider_id: str,
        target_id: str,
        project_id: str | None,
        request_json: str,
        request_hash: str,
        destination_identity: str,
        requested_basename: str,
    ) -> JobRecord:
        """Create or replay a preallocated immutable Contract-4 ledger row."""
        studio_job_id = self._text(studio_job_id, "studio_job_id")
        client_request_id = self._text(client_request_id, "client_request_id")
        provider_id = self._text(provider_id, "provider_id")
        target_id = self._text(target_id, "target_id")
        project_id = self._v4_project_id(project_id)
        request_json, request_hash = self._v4_request_identity(request_json, request_hash)
        destination_identity = self._v4_hex(destination_identity, "destination_identity")
        requested_basename = self._v4_basename(requested_basename)
        timestamp = self._now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            by_id = connection.execute(
                "SELECT * FROM studio_jobs WHERE studio_job_id = ?",
                (studio_job_id,),
            ).fetchone()
            by_client = connection.execute(
                "SELECT * FROM studio_jobs WHERE client_request_id = ?",
                (client_request_id,),
            ).fetchone()
            if by_id is not None or by_client is not None:
                if by_id is None or by_client is None or by_id["studio_job_id"] != by_client["studio_job_id"]:
                    raise _invalid("Studio job identity is already bound to another row")
                row = by_id
                self._assert_v4_scope(row, project_id)
                expected = {
                    "studio_job_id": studio_job_id,
                    "client_request_id": client_request_id,
                    "provider_id": provider_id,
                    "target_id": target_id,
                    "request_json": request_json,
                    "request_hash": request_hash,
                    "destination_identity": destination_identity,
                    "requested_basename": requested_basename,
                }
                if not self._scoped_fields_match(row, expected):
                    raise _invalid("Contract-4 identity does not match the existing row")
                connection.execute("COMMIT")
                return self._row_to_record(row)
            try:
                connection.execute(
                    """
                    INSERT INTO studio_jobs (
                        studio_job_id, client_request_id, provider_id, target_id,
                        provider_job_id, connection_id, request_json, request_hash,
                        frozen_destination, requested_basename, state, created_at,
                        updated_at, queued_at, running_at, completed_at,
                        project_id, row_contract, destination_identity, published_identity
                    ) VALUES (?, ?, ?, ?, NULL, ?, ?, ?, NULL, ?, 'submitting', ?, ?, ?, NULL, NULL, ?, 4, ?, NULL)
                    """,
                    (
                        studio_job_id,
                        client_request_id,
                        provider_id,
                        target_id,
                        provider_id,
                        request_json,
                        request_hash,
                        requested_basename,
                        timestamp,
                        timestamp,
                        timestamp,
                        project_id,
                        destination_identity,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                active = connection.execute(
                    """
                    SELECT 1 FROM studio_jobs
                    WHERE provider_id = 'comfyui-local'
                      AND connection_id = ?
                      AND state IN ('submitting','queued','running')
                    LIMIT 1
                    """,
                    (provider_id,),
                ).fetchone()
                if active is not None:
                    raise ProviderBusyError() from exc
                raise _invalid() from exc
            row = connection.execute(
                "SELECT * FROM studio_jobs WHERE studio_job_id = ?",
                (studio_job_id,),
            ).fetchone()
            connection.execute("COMMIT")
            self._chmod_private_file()
            return self._row_to_record(row)
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def get_scoped(
        self,
        *,
        studio_job_id: str,
        expected_project_id: str | None,
        field_present: bool = True,
    ) -> JobRecord:
        studio_job_id = self._text(studio_job_id, "studio_job_id")
        if not isinstance(field_present, bool):
            raise _invalid("field_present must be boolean")
        expected_project_id = self._v4_project_id(expected_project_id)
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM studio_jobs WHERE studio_job_id = ?",
                (studio_job_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise JobNotFoundError()
        self._assert_v4_scope(row, expected_project_id, field_present=field_present)
        return self._row_to_record(row)

    def bind_provider_job(
        self,
        *,
        studio_job_id: str,
        provider_job_id: str,
        expected_project_id: str | None,
    ) -> JobRecord:
        studio_job_id = self._text(studio_job_id, "studio_job_id")
        provider_job_id = self._text(provider_job_id, "provider_job_id")
        expected_project_id = self._v4_project_id(expected_project_id)
        timestamp = self._now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM studio_jobs WHERE studio_job_id = ?", (studio_job_id,)).fetchone()
            if row is None:
                raise JobNotFoundError()
            self._assert_v4_scope(row, expected_project_id)
            if row["provider_job_id"] is not None:
                if row["provider_job_id"] == provider_job_id:
                    connection.execute("COMMIT")
                    return self._row_to_record(row)
                raise _invalid("provider_job_id is write-once")
            if row["state"] not in _NONTERMINAL_STATES:
                raise _invalid("provider_job_id can only be bound to a nonterminal row")
            next_state = "queued" if row["state"] == "submitting" else row["state"]
            queued_at = row["queued_at"] if row["queued_at"] is not None else timestamp
            connection.execute(
                """
                UPDATE studio_jobs
                SET provider_job_id = ?, state = ?, queued_at = ?, updated_at = ?
                WHERE studio_job_id = ?
                """,
                (provider_job_id, next_state, queued_at, timestamp, studio_job_id),
            )
            updated = connection.execute("SELECT * FROM studio_jobs WHERE studio_job_id = ?", (studio_job_id,)).fetchone()
            connection.execute("COMMIT")
            return self._row_to_record(updated)
        except sqlite3.IntegrityError as exc:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise _invalid() from exc
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def finish_submit_without_job(
        self,
        *,
        studio_job_id: str,
        expected_project_id: str | None,
        terminal_status: str,
        reason_code: str | None,
    ) -> JobRecord:
        studio_job_id = self._text(studio_job_id, "studio_job_id")
        expected_project_id = self._v4_project_id(expected_project_id)
        terminal_status = self._state(terminal_status)
        if terminal_status not in _TERMINAL_STATES or terminal_status == "ready":
            raise _invalid("terminal_status must be a non-ready terminal state")
        normalized_reason = _normalize_transition_reason(reason_code)
        if normalized_reason is None:
            raise _invalid("terminal_status requires a reason_code")
        timestamp = self._now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM studio_jobs WHERE studio_job_id = ?", (studio_job_id,)).fetchone()
            if row is None:
                raise JobNotFoundError()
            self._assert_v4_scope(row, expected_project_id)
            current_state = self._row_state(row["state"])
            if current_state in _TERMINAL_STATES:
                if current_state == terminal_status and row["reason_code"] == normalized_reason:
                    connection.execute("COMMIT")
                    return self._row_to_record(row)
                raise _invalid("terminal lifecycle state is write-once")
            if row["provider_job_id"] is not None:
                raise _invalid("cannot finish a submit that already has a provider job")
            completed_at = row["completed_at"] if row["completed_at"] is not None else timestamp
            connection.execute(
                """
                UPDATE studio_jobs
                SET state = ?, reason_code = ?, updated_at = ?, completed_at = ?
                WHERE studio_job_id = ?
                """,
                (terminal_status, normalized_reason, timestamp, completed_at, studio_job_id),
            )
            updated = connection.execute("SELECT * FROM studio_jobs WHERE studio_job_id = ?", (studio_job_id,)).fetchone()
            connection.execute("COMMIT")
            return self._row_to_record(updated)
        except sqlite3.IntegrityError as exc:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise _invalid() from exc
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def transition_scoped(
        self,
        *,
        studio_job_id: str,
        expected_project_id: str | None,
        allowed_from: Any,
        new_status: str,
        reason_code: str | None = None,
    ) -> JobRecord:
        studio_job_id = self._text(studio_job_id, "studio_job_id")
        expected_project_id = self._v4_project_id(expected_project_id)
        new_status = self._state(new_status)
        if new_status == "ready":
            raise _invalid("ready is completed only through complete_v4_delivery")
        try:
            if isinstance(allowed_from, str):
                allowed_values = (allowed_from,)
            else:
                allowed_values = tuple(allowed_from)
        except (TypeError, ValueError, RecursionError) as exc:
            raise _invalid("allowed_from must contain lifecycle states") from exc
        if not allowed_values or any(self._state(value) not in _STATES for value in allowed_values):
            raise _invalid("allowed_from must contain lifecycle states")
        allowed_values = tuple(self._state(value) for value in allowed_values)
        normalized_reason = _normalize_transition_reason(reason_code)
        timestamp = self._now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM studio_jobs WHERE studio_job_id = ?", (studio_job_id,)).fetchone()
            if row is None:
                raise JobNotFoundError()
            self._assert_v4_scope(row, expected_project_id)
            current_state = self._row_state(row["state"])
            if current_state not in allowed_values or new_status not in _ALLOWED_TRANSITIONS[current_state]:
                raise _invalid("lifecycle transition is not allowed")
            if current_state == new_status:
                if current_state in _TERMINAL_STATES and normalized_reason is not None and normalized_reason != row["reason_code"]:
                    raise _invalid("terminal reason_code is write-once")
                connection.execute("COMMIT")
                return self._row_to_record(row)
            queued_at = row["queued_at"]
            running_at = row["running_at"]
            completed_at = row["completed_at"]
            if new_status in _NONTERMINAL_STATES and queued_at is None:
                queued_at = timestamp
            if new_status == "running" and running_at is None:
                running_at = timestamp
            if new_status in _TERMINAL_STATES and completed_at is None:
                completed_at = timestamp
            connection.execute(
                """
                UPDATE studio_jobs
                SET state = ?, reason_code = ?, updated_at = ?, queued_at = ?,
                    running_at = ?, completed_at = ?
                WHERE studio_job_id = ?
                """,
                (new_status, normalized_reason, timestamp, queued_at, running_at, completed_at, studio_job_id),
            )
            updated = connection.execute("SELECT * FROM studio_jobs WHERE studio_job_id = ?", (studio_job_id,)).fetchone()
            connection.execute("COMMIT")
            return self._row_to_record(updated)
        except sqlite3.IntegrityError as exc:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise _invalid() from exc
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def complete_v4_delivery(
        self,
        *,
        studio_job_id: str,
        expected_project_id: str | None,
        published_identity: str,
        output_sha256: str,
        output_bytes: int,
    ) -> JobRecord:
        studio_job_id = self._text(studio_job_id, "studio_job_id")
        expected_project_id = self._v4_project_id(expected_project_id)
        output_identity, expected_identity = self._v4_delivery_identity(
            published_identity,
            output_sha256,
            output_bytes,
        )
        timestamp = self._now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM studio_jobs WHERE studio_job_id = ?", (studio_job_id,)).fetchone()
            if row is None:
                raise JobNotFoundError()
            self._assert_v4_scope(row, expected_project_id)
            if row["published_identity"] is not None:
                if row["published_identity"] != published_identity:
                    raise _invalid("published_identity is write-once")
                self._assert_expected_v4_delivery_identity(
                    row["provider_output_identity_json"],
                    output_identity,
                    expected_identity,
                )
                if row["state"] != "ready":
                    raise _invalid("completed V4 delivery has an invalid lifecycle state")
                connection.execute("COMMIT")
                return self._row_to_record(row)
            if row["state"] not in _NONTERMINAL_STATES and row["state"] != "ready":
                raise _invalid("cannot complete delivery from a terminal failure")
            if row["provider_output_identity_json"] is not None:
                self._assert_expected_v4_delivery_identity(
                    row["provider_output_identity_json"],
                    output_identity,
                    expected_identity,
                )
                output_identity_value = row["provider_output_identity_json"]
            else:
                output_identity_value = output_identity
            queued_at = row["queued_at"] if row["queued_at"] is not None else timestamp
            completed_at = row["completed_at"] if row["completed_at"] is not None else timestamp
            connection.execute(
                """
                UPDATE studio_jobs
                SET published_identity = ?, provider_output_identity_json = ?, state = 'ready',
                    reason_code = NULL, updated_at = ?, queued_at = ?, completed_at = ?
                WHERE studio_job_id = ?
                """,
                (published_identity, output_identity_value, timestamp, queued_at, completed_at, studio_job_id),
            )
            updated = connection.execute("SELECT * FROM studio_jobs WHERE studio_job_id = ?", (studio_job_id,)).fetchone()
            connection.execute("COMMIT")
            return self._row_to_record(updated)
        except sqlite3.IntegrityError as exc:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise _invalid() from exc
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def get_job(self, studio_job_id: str) -> JobRecord:
        studio_job_id = self._text(studio_job_id, "studio_job_id")
        connection = self._connect()
        try:
            row = connection.execute("SELECT * FROM studio_jobs WHERE studio_job_id = ?", (studio_job_id,)).fetchone()
        finally:
            connection.close()
        if row is None:
            raise JobNotFoundError()
        self._assert_contract3_row(row)
        return self._row_to_record(row)

    def get_by_client_request_id(self, client_request_id: str) -> JobRecord:
        client_request_id = self._text(client_request_id, "client_request_id")
        connection = self._connect()
        try:
            row = connection.execute("SELECT * FROM studio_jobs WHERE client_request_id = ?", (client_request_id,)).fetchone()
        finally:
            connection.close()
        if row is None:
            raise JobNotFoundError()
        self._assert_contract3_row(row)
        return self._row_to_record(row)

    def list_recent(self, *, project_id: str | None, limit: int = 50) -> list[JobRecord]:
        """Newest jobs of one project (or the projectless scope), read-only.

        Unlike ``count_jobs`` this accepts Contract-3 and Contract-4 rows alike:
        it is a display projection, not a lifecycle operation. A row that no
        longer validates is skipped instead of hiding every other job."""
        if type(limit) is not int or not 1 <= limit <= 500:
            raise _invalid("limit must be between 1 and 500")
        if project_id is not None:
            project_id = self._text(project_id, "project_id")
        # Contract-3 rows carry their project only inside the frozen request;
        # Contract-4 rows carry it in the column. Scan a bounded newest window
        # and match on the effective project of each row.
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT * FROM studio_jobs ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (min(2_000, limit * 10),),
            ).fetchall()
        finally:
            connection.close()
        records: list[JobRecord] = []
        for row in rows:
            try:
                record = self._row_to_record(row)
                effective = record.project_id
                if effective is None:
                    requested = record.request.get("project_id")
                    effective = requested if isinstance(requested, str) and requested else None
            except Exception:  # noqa: BLE001 - one bad row must not blank the list
                continue
            if effective != project_id:
                continue
            records.append(record)
            if len(records) >= limit:
                break
        return records

    def count_jobs(self) -> int:
        connection = self._connect()
        try:
            rows = connection.execute("SELECT * FROM studio_jobs").fetchall()
            # Preserve the legacy API's deterministic V4 scope rejection before
            # validating Contract-3 rows in a mixed ledger.
            for row in rows:
                if JobStore._row_contract(row["row_contract"]) == 4:
                    raise ProjectScopeError()
            for row in rows:
                self._assert_contract3_row(row)
            return len(rows)
        finally:
            connection.close()

    def transition(self, studio_job_id: str, state: str, *, reason_code: str | None = None, now: float | None = None) -> JobRecord:
        studio_job_id = self._text(studio_job_id, "studio_job_id")
        state = self._state(state)
        if state not in _STATES:
            raise _invalid()
        normalized_reason = _normalize_transition_reason(reason_code)
        timestamp = self._now(now)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM studio_jobs WHERE studio_job_id = ?", (studio_job_id,)).fetchone()
            if row is None:
                raise JobNotFoundError()
            self._assert_contract3_row(row)
            current_state = self._row_state(row["state"])
            if state not in _ALLOWED_TRANSITIONS[current_state]:
                raise _invalid()
            if current_state == state:
                if current_state in _TERMINAL_STATES:
                    if normalized_reason is not None and normalized_reason != row["reason_code"]:
                        raise _invalid()
                connection.execute("COMMIT")
                return self._row_to_record(row)
            if current_state in _TERMINAL_STATES:
                if normalized_reason is None:
                    normalized_reason = row["reason_code"]
                elif row["reason_code"] is not None and normalized_reason != row["reason_code"]:
                    raise _invalid()
            queued_at = row["queued_at"]
            running_at = row["running_at"]
            completed_at = row["completed_at"]
            if state in _NONTERMINAL_STATES and queued_at is None:
                queued_at = timestamp
            if state == "running" and running_at is None:
                running_at = timestamp
            if state in _TERMINAL_STATES:
                if completed_at is None:
                    completed_at = timestamp
            connection.execute(
                """
                UPDATE studio_jobs
                SET state = ?, reason_code = ?, updated_at = ?, queued_at = ?,
                    running_at = ?, completed_at = ?
                WHERE studio_job_id = ?
                """,
                (state, normalized_reason, timestamp, queued_at, running_at, completed_at, studio_job_id),
            )
            updated = connection.execute("SELECT * FROM studio_jobs WHERE studio_job_id = ?", (studio_job_id,)).fetchone()
            connection.execute("COMMIT")
            return self._row_to_record(updated)
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def update_provider_job(self, studio_job_id: str, provider_job_id: str) -> JobRecord:
        provider_job_id = self._text(provider_job_id, "provider_job_id")
        return self._update_fields(studio_job_id, {"provider_job_id": provider_job_id})

    def set_output_identity(self, studio_job_id: str, identity: Any) -> JobRecord:
        identity_json = canonical_request_json(identity)
        return self._update_fields(studio_job_id, {"provider_output_identity_json": identity_json})

    def backfill_ready_output_identity(self, studio_job_id: str, identity: Any) -> JobRecord:
        """Atomically backfill one legacy unpublished ready row's output identity."""
        studio_job_id = self._text(studio_job_id, "studio_job_id")
        identity_json = canonical_request_json(identity)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM studio_jobs WHERE studio_job_id = ?",
                (studio_job_id,),
            ).fetchone()
            if row is None:
                raise JobNotFoundError()
            self._assert_contract3_row(row)
            if not (
                row["provider_id"] == "comfyui-local"
                and row["state"] == "ready"
                and row["reason_code"] is None
                and row["publication_state"] == "unpublished"
            ):
                raise _invalid()
            if row["provider_output_identity_json"] == identity_json:
                connection.execute("COMMIT")
                return self._row_to_record(row)
            if row["provider_output_identity_json"] is not None:
                raise _invalid()
            cursor = connection.execute(
                """
                UPDATE studio_jobs
                SET provider_output_identity_json = ?
                WHERE studio_job_id = ?
                  AND provider_id = 'comfyui-local'
                  AND state = 'ready'
                  AND reason_code IS NULL
                  AND provider_output_identity_json IS NULL
                  AND publication_state = 'unpublished'
                """,
                (identity_json, studio_job_id),
            )
            if cursor.rowcount != 1:
                raise _invalid()
            updated = connection.execute(
                "SELECT * FROM studio_jobs WHERE studio_job_id = ?",
                (studio_job_id,),
            ).fetchone()
            connection.execute("COMMIT")
            return self._row_to_record(updated)
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def fail_ready_without_output(self, studio_job_id: str, reason_code: str) -> JobRecord:
        """Atomically reclassify one legacy unpublished ready row with no output."""
        studio_job_id = self._text(studio_job_id, "studio_job_id")
        normalized_reason = _normalize_transition_reason(reason_code)
        if normalized_reason not in _LEGACY_OUTPUT_FAILURE_REASONS:
            raise _invalid()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM studio_jobs WHERE studio_job_id = ?",
                (studio_job_id,),
            ).fetchone()
            if row is None:
                raise JobNotFoundError()
            self._assert_contract3_row(row)
            if (
                row["provider_id"] == "comfyui-local"
                and row["state"] == "failed"
                and row["reason_code"] == normalized_reason
                and row["provider_output_identity_json"] is None
                and row["publication_state"] == "unpublished"
            ):
                connection.execute("COMMIT")
                return self._row_to_record(row)
            if not (
                row["provider_id"] == "comfyui-local"
                and row["state"] == "ready"
                and row["reason_code"] is None
                and row["provider_output_identity_json"] is None
                and row["publication_state"] == "unpublished"
            ):
                raise _invalid()
            cursor = connection.execute(
                """
                UPDATE studio_jobs
                SET state = 'failed', reason_code = ?
                WHERE studio_job_id = ?
                  AND provider_id = 'comfyui-local'
                  AND state = 'ready'
                  AND reason_code IS NULL
                  AND provider_output_identity_json IS NULL
                  AND publication_state = 'unpublished'
                """,
                (normalized_reason, studio_job_id),
            )
            if cursor.rowcount != 1:
                raise _invalid()
            updated = connection.execute(
                "SELECT * FROM studio_jobs WHERE studio_job_id = ?",
                (studio_job_id,),
            ).fetchone()
            connection.execute("COMMIT")
            return self._row_to_record(updated)
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def _update_fields(self, studio_job_id: str, fields: Mapping[str, Any]) -> JobRecord:
        studio_job_id = self._text(studio_job_id, "studio_job_id")
        allowed = {"provider_job_id", "provider_output_identity_json"}
        if set(fields) - allowed:
            raise _invalid()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM studio_jobs WHERE studio_job_id = ?", (studio_job_id,)).fetchone()
            if row is None:
                raise JobNotFoundError()
            self._assert_contract3_row(row)
            for field_name, value in fields.items():
                existing_value = row[field_name]
                if existing_value is not None and existing_value != value:
                    raise _invalid()
            if all(row[field_name] == value for field_name, value in fields.items()):
                connection.execute("COMMIT")
                return self._row_to_record(row)
            assignments = ", ".join(f"{key} = ?" for key in fields)
            values = [fields[key] for key in fields]
            if row["state"] not in _TERMINAL_STATES:
                assignments += ", updated_at = ?"
                values.append(self._now())
            values.append(studio_job_id)
            try:
                connection.execute(
                    f"UPDATE studio_jobs SET {assignments} WHERE studio_job_id = ?",
                    tuple(values),
                )
            except sqlite3.IntegrityError as exc:
                raise _invalid() from exc
            row = connection.execute("SELECT * FROM studio_jobs WHERE studio_job_id = ?", (studio_job_id,)).fetchone()
            connection.execute("COMMIT")
            return self._row_to_record(row)
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def acquire_publication_lease(
        self,
        studio_job_id: str,
        owner_id: str,
        *,
        now: float | None = None,
        lease_seconds: float = 300.0,
    ) -> PublicationLease | None:
        studio_job_id = self._text(studio_job_id, "studio_job_id")
        owner_id = self._text(owner_id, "owner_id")
        if isinstance(lease_seconds, bool) or not isinstance(lease_seconds, (int, float)) or lease_seconds <= 0:
            raise _invalid()
        try:
            lease_value = float(lease_seconds)
        except (OverflowError, TypeError, ValueError) as exc:
            raise _invalid() from exc
        if not math.isfinite(lease_value) or lease_value <= 0:
            raise _invalid()
        timestamp = self._now(now)
        try:
            expires_at = timestamp + lease_value
        except OverflowError as exc:
            raise _invalid() from exc
        if not math.isfinite(expires_at):
            raise _invalid()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM studio_jobs WHERE studio_job_id = ?", (studio_job_id,)).fetchone()
            if row is None:
                raise JobNotFoundError()
            self._assert_contract3_row(row, check_output_identity=False)
            lease_expires_at = self._row_finite_float(
                row["publication_lease_expires_at"],
                "publication_lease_expires_at",
            )
            if row["state"] != "ready":
                connection.execute("COMMIT")
                return None
            output_identity = row["provider_output_identity_json"]
            if output_identity is None:
                connection.execute("COMMIT")
                return None
            try:
                self._row_output_identity_json(output_identity)
            except SchemaError as exc:
                raise _invalid() from exc
            if row["publication_state"] == "published":
                connection.execute("COMMIT")
                return None
            if row["publication_lease_token"] and lease_expires_at is not None:
                if lease_expires_at > timestamp:
                    connection.execute("COMMIT")
                    return None
            token = uuid.uuid4().hex
            try:
                connection.execute(
                    """
                    UPDATE studio_jobs
                    SET publication_state = 'publishing', publication_lease_owner = ?,
                        publication_lease_token = ?, publication_lease_expires_at = ?
                    WHERE studio_job_id = ?
                    """,
                    (owner_id, token, expires_at, studio_job_id),
                )
            except sqlite3.IntegrityError as exc:
                raise _invalid() from exc
            connection.execute("COMMIT")
            return PublicationLease(studio_job_id, owner_id, token, expires_at)
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def release_publication_lease(
        self,
        studio_job_id: str,
        owner_id: str,
        token: str,
        *,
        now: float | None = None,
        published_path: str | None = None,
        source_media_sha256: str | None = None,
        published_file_sha256: str | None = None,
    ) -> JobRecord:
        studio_job_id = self._text(studio_job_id, "studio_job_id")
        owner_id = self._text(owner_id, "owner_id")
        token = self._text(token, "token")
        publication_fields = (published_path, source_media_sha256, published_file_sha256)
        supplied_publication_fields = tuple(value is not None for value in publication_fields)
        if any(supplied_publication_fields) and not all(supplied_publication_fields):
            raise _invalid()
        if published_path is not None:
            published_path = self._path_text(published_path, "published_path")
        if source_media_sha256 is not None:
            source_media_sha256 = self._text(source_media_sha256, "source_media_sha256")
        if published_file_sha256 is not None:
            published_file_sha256 = self._text(published_file_sha256, "published_file_sha256")
        timestamp = self._now(now)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM studio_jobs WHERE studio_job_id = ?", (studio_job_id,)).fetchone()
            if row is None:
                raise JobNotFoundError()
            self._assert_contract3_row(row, check_output_identity=False)
            lease_expires_at = self._row_finite_float(
                row["publication_lease_expires_at"],
                "publication_lease_expires_at",
            )
            if (
                row["publication_lease_owner"] != owner_id
                or row["publication_lease_token"] != token
                or lease_expires_at is None
                or lease_expires_at <= timestamp
            ):
                raise LeaseConflictError()
            if row["publication_state"] == "published":
                raise _invalid()
            if row["state"] != "ready" or row["provider_output_identity_json"] is None:
                raise _invalid()
            try:
                self._row_output_identity_json(row["provider_output_identity_json"])
            except SchemaError as exc:
                raise _invalid() from exc
            if published_path is not None:
                _ensure_published_path_is_child(row["frozen_destination"], published_path)
                publication_state = "published"
                publication_lease_owner = row["publication_lease_owner"]
                publication_lease_token = row["publication_lease_token"]
                publication_lease_expires_at = lease_expires_at
            else:
                publication_state = "unpublished"
                publication_lease_owner = None
                publication_lease_token = None
                publication_lease_expires_at = None
            try:
                connection.execute(
                    """
                    UPDATE studio_jobs
                    SET publication_state = ?, publication_lease_owner = ?,
                        publication_lease_token = ?, publication_lease_expires_at = ?,
                        published_path = ?, source_media_sha256 = ?, published_file_sha256 = ?
                    WHERE studio_job_id = ?
                    """,
                    (
                        publication_state,
                        publication_lease_owner,
                        publication_lease_token,
                        publication_lease_expires_at,
                        published_path,
                        source_media_sha256,
                        published_file_sha256,
                        studio_job_id,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise _invalid() from exc
            updated = connection.execute("SELECT * FROM studio_jobs WHERE studio_job_id = ?", (studio_job_id,)).fetchone()
            connection.execute("COMMIT")
            return self._row_to_record(updated)
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def begin_stamp(self, studio_job_id: str) -> JobRecord:
        studio_job_id = self._text(studio_job_id, "studio_job_id")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM studio_jobs WHERE studio_job_id = ?", (studio_job_id,)).fetchone()
            if row is None:
                raise JobNotFoundError()
            self._assert_contract3_row(row)
            current_state = self._row_state(row["state"])
            current_stamp_state, current_attempts = self._row_stamp_fields(
                row["stamp_state"],
                row["stamp_attempt_count"],
            )
            if current_stamp_state == "succeeded":
                connection.execute("COMMIT")
                return self._row_to_record(row)
            attempts = current_attempts + 1
            if current_state in _TERMINAL_STATES:
                connection.execute(
                    """
                    UPDATE studio_jobs
                    SET stamp_state = 'pending', stamp_attempt_count = ?
                    WHERE studio_job_id = ?
                    """,
                    (attempts, studio_job_id),
                )
            else:
                connection.execute(
                    """
                    UPDATE studio_jobs
                    SET stamp_state = 'pending', stamp_attempt_count = ?, updated_at = ?
                    WHERE studio_job_id = ?
                    """,
                    (attempts, self._now(), studio_job_id),
                )
            updated = connection.execute("SELECT * FROM studio_jobs WHERE studio_job_id = ?", (studio_job_id,)).fetchone()
            connection.execute("COMMIT")
            return self._row_to_record(updated)
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def finish_stamp(self, studio_job_id: str, *, success: bool) -> JobRecord:
        if not isinstance(success, bool):
            raise _invalid()
        state = "succeeded" if success else "failed"
        return self._update_stamp(studio_job_id, state)

    def _update_stamp(self, studio_job_id: str, state: str) -> JobRecord:
        studio_job_id = self._text(studio_job_id, "studio_job_id")
        if state not in {"succeeded", "failed"}:
            raise _invalid()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM studio_jobs WHERE studio_job_id = ?", (studio_job_id,)).fetchone()
            if row is None:
                raise JobNotFoundError()
            self._assert_contract3_row(row)
            current_state = self._row_state(row["state"])
            current_stamp_state, attempts = self._row_stamp_fields(
                row["stamp_state"],
                row["stamp_attempt_count"],
            )
            if attempts == 0:
                raise _invalid()
            if current_stamp_state == state:
                connection.execute("COMMIT")
                return self._row_to_record(row)
            if current_stamp_state != "pending":
                raise _invalid()
            if current_state in _TERMINAL_STATES:
                connection.execute(
                    "UPDATE studio_jobs SET stamp_state = ? WHERE studio_job_id = ?",
                    (state, studio_job_id),
                )
            else:
                connection.execute(
                    "UPDATE studio_jobs SET stamp_state = ?, updated_at = ? WHERE studio_job_id = ?",
                    (state, self._now(), studio_job_id),
                )
            row = connection.execute("SELECT * FROM studio_jobs WHERE studio_job_id = ?", (studio_job_id,)).fetchone()
            connection.execute("COMMIT")
            return self._row_to_record(row)
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def _chmod_private_file(self) -> None:
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    @staticmethod
    def _row_finite_float(value: Any, field_name: str, *, required: bool = False) -> float | None:
        if value is None:
            if required:
                raise SchemaError(f"invalid Studio ledger field: {field_name}")
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SchemaError(f"invalid Studio ledger field: {field_name}")
        try:
            normalized = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise SchemaError(f"invalid Studio ledger field: {field_name}") from exc
        if not math.isfinite(normalized):
            raise SchemaError(f"invalid Studio ledger field: {field_name}")
        return normalized

    @staticmethod
    def _row_state(value: Any) -> str:
        if not isinstance(value, str):
            raise SchemaError("invalid Studio ledger field: state")
        try:
            allowed = value in _STATES
        except (TypeError, ValueError, AttributeError, RecursionError) as exc:
            raise SchemaError("invalid Studio ledger field: state") from exc
        if not allowed:
            raise SchemaError("invalid Studio ledger field: state")
        return value

    @staticmethod
    def _row_stamp_state(value: Any) -> str:
        if not isinstance(value, str):
            raise SchemaError("invalid Studio ledger field: stamp_state")
        try:
            allowed = value in _STAMP_STATES
        except (TypeError, ValueError, AttributeError, RecursionError) as exc:
            raise SchemaError("invalid Studio ledger field: stamp_state") from exc
        if not allowed:
            raise SchemaError("invalid Studio ledger field: stamp_state")
        return value

    @staticmethod
    def _row_stamp_fields(stamp_state: Any, stamp_attempt_count: Any) -> tuple[str, int]:
        normalized_state = JobStore._row_stamp_state(stamp_state)
        normalized_attempts = JobStore._row_stamp_attempt_count(stamp_attempt_count)
        if (
            (normalized_state == "not_started" and normalized_attempts != 0)
            or (normalized_state != "not_started" and normalized_attempts <= 0)
        ):
            raise SchemaError("invalid Studio ledger stamp accounting")
        return normalized_state, normalized_attempts

    @staticmethod
    def _row_request_json(value: Any) -> dict[str, Any]:
        if not isinstance(value, str) or not value.strip():
            raise SchemaError("invalid Studio ledger field: request_json")
        try:
            parsed = json.loads(value, parse_constant=JobStore._reject_json_constant)
        except (TypeError, UnicodeError, ValueError, OverflowError, RecursionError) as exc:
            raise SchemaError("invalid Studio ledger field: request_json") from exc
        if not isinstance(parsed, dict):
            raise SchemaError("invalid Studio ledger field: request_json")
        return parsed

    @staticmethod
    def _row_stamp_attempt_count(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SchemaError("invalid Studio ledger field: stamp_attempt_count")
        return value

    @staticmethod
    def _row_reason_code(value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise SchemaError("invalid Studio ledger field: reason_code")
        try:
            allowed = value in REASON_CODES
        except (TypeError, ValueError, AttributeError, RecursionError) as exc:
            raise SchemaError("invalid Studio ledger field: reason_code") from exc
        if not allowed:
            raise SchemaError("invalid Studio ledger field: reason_code")
        return value

    @staticmethod
    def _row_output_identity_json(value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise SchemaError("invalid Studio ledger field: provider_output_identity_json")
        try:
            json.loads(value, parse_constant=JobStore._reject_json_constant)
        except (TypeError, UnicodeError, ValueError, RecursionError) as exc:
            raise SchemaError("invalid Studio ledger field: provider_output_identity_json") from exc
        return value

    @staticmethod
    def _row_publication_state(value: Any) -> str:
        if not isinstance(value, str):
            raise SchemaError("invalid Studio ledger field: publication_state")
        try:
            allowed = value in _PUBLICATION_STATES
        except (TypeError, ValueError, AttributeError, RecursionError) as exc:
            raise SchemaError("invalid Studio ledger field: publication_state") from exc
        if not allowed:
            raise SchemaError("invalid Studio ledger field: publication_state")
        return value

    @staticmethod
    def _row_optional_nonempty_text(value: Any, field_name: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise SchemaError(f"invalid Studio ledger field: {field_name}")
        return value

    @staticmethod
    def _row_optional_opaque(value: Any, field_name: str) -> str | bytes | None:
        if value is None:
            return None
        if isinstance(value, str):
            if not value.strip():
                raise SchemaError(f"invalid Studio ledger field: {field_name}")
            return value
        if isinstance(value, bytes):
            if not value:
                raise SchemaError(f"invalid Studio ledger field: {field_name}")
            return value
        raise SchemaError(f"invalid Studio ledger field: {field_name}")

    @staticmethod
    def _row_required_text(value: Any, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise SchemaError(f"invalid Studio ledger field: {field_name}")
        return value

    @staticmethod
    def _row_publication_metadata(
        published_path: Any,
        source_media_sha256: Any,
        published_file_sha256: Any,
    ) -> tuple[str | None, str | None, str | None]:
        values = (published_path, source_media_sha256, published_file_sha256)
        names = ("published_path", "source_media_sha256", "published_file_sha256")
        if all(value is None for value in values):
            return (None, None, None)
        normalized = tuple(
            JobStore._row_optional_nonempty_text(value, name)
            for value, name in zip(values, names)
        )
        if any(value is None for value in normalized):
            raise SchemaError("invalid Studio ledger publication metadata")
        return normalized  # type: ignore[return-value]

    @staticmethod
    def _reject_json_constant(value: str) -> Any:
        raise ValueError(f"invalid JSON constant: {value}")

    @staticmethod
    def _row_contract(value: Any) -> int:
        if isinstance(value, bool) or type(value) is not int or value not in (3, 4):
            raise SchemaError("invalid Studio ledger field: row_contract")
        return value

    @staticmethod
    def _row_project_id(value: Any) -> str | None:
        if value is None:
            return None
        if type(value) is not str or re.fullmatch(r"prj_[0-9a-f]{32}", value) is None:
            raise SchemaError("invalid Studio ledger field: project_id")
        return value

    @staticmethod
    def _row_hex_identity(value: Any, field_name: str) -> str | None:
        if value is None:
            return None
        if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise SchemaError(f"invalid Studio ledger field: {field_name}")
        return value

    @staticmethod
    def _validate_v4_row_shape(row: sqlite3.Row) -> tuple[str | None, str, str | None]:
        project_id = JobStore._row_project_id(row["project_id"])
        destination_identity = JobStore._row_hex_identity(row["destination_identity"], "destination_identity")
        published_identity = JobStore._row_hex_identity(row["published_identity"], "published_identity")
        state = JobStore._row_state(row["state"])
        output_identity = row["provider_output_identity_json"]
        if published_identity is None:
            if output_identity is not None or state == "ready":
                raise SchemaError("invalid Studio ledger V4 delivery identity relation")
        else:
            if state != "ready" or output_identity is None:
                raise SchemaError("invalid Studio ledger V4 delivery identity relation")
            parsed_output = JobStore._parse_v4_delivery_identity(output_identity)
            if parsed_output["published_identity"] != published_identity:
                raise SchemaError("invalid Studio ledger V4 delivery identity relation")
        if destination_identity is None or row["frozen_destination"] is not None or row["published_path"] is not None:
            raise SchemaError("invalid Studio ledger V4 row shape")
        JobStore._row_request_json(row["request_json"])
        request_hash = row["request_hash"]
        if type(request_hash) is not str or re.fullmatch(r"[0-9a-f]{64}", request_hash) is None:
            raise SchemaError("invalid Studio ledger field: request_hash")
        if hashlib.sha256(row["request_json"].encode("utf-8")).hexdigest() != request_hash:
            raise SchemaError("invalid Studio ledger request identity")
        for field_name in ("client_request_id", "provider_id", "target_id", "connection_id"):
            value = row[field_name]
            if type(value) is not str or not value.strip():
                raise SchemaError(f"invalid Studio ledger field: {field_name}")
        basename = row["requested_basename"]
        if (
            type(basename) is not str
            or not basename.strip()
            or "/" in basename
            or chr(92) in basename
            or not basename.casefold().endswith(".mp4")
            or basename[-1] in ". "
        ):
            raise SchemaError("invalid Studio ledger field: requested_basename")
        return project_id, destination_identity, published_identity

    @staticmethod
    def _validate_persisted_publication_relation(
        *,
        state: str,
        publication_state: str,
        provider_output_identity_json: Any,
        publication_lease_owner: Any,
        publication_lease_token: Any,
        publication_lease_expires_at: Any,
        published_path: Any,
        source_media_sha256: Any,
        published_file_sha256: Any,
        check_output_identity: bool,
    ) -> None:
        """Validate the persisted publication state machine as one relation."""
        metadata_values = (published_path, source_media_sha256, published_file_sha256)
        lease_values = (
            publication_lease_owner,
            publication_lease_token,
            publication_lease_expires_at,
        )
        metadata_all_null = all(value is None for value in metadata_values)
        metadata_complete = all(value is not None for value in metadata_values)
        lease_all_null = all(value is None for value in lease_values)
        lease_complete = all(value is not None for value in lease_values)

        if not (metadata_all_null or metadata_complete):
            raise SchemaError("invalid Studio ledger publication metadata")
        if not (lease_all_null or lease_complete):
            raise SchemaError("invalid Studio ledger publication lease")

        if publication_state == "unpublished":
            if not metadata_all_null or not lease_all_null:
                raise SchemaError("invalid Studio ledger unpublished publication relation")
            return

        if publication_state == "publishing":
            if (
                state != "ready"
                or not metadata_all_null
                or not lease_complete
                or provider_output_identity_json is None
            ):
                raise SchemaError("invalid Studio ledger publishing publication relation")
            # Deliberate check_output_identity=False callers retain their
            # stable invalid_request mapping only for unpublished rows. A
            # publishing row must already contain a valid output identity.
            JobStore._row_output_identity_json(provider_output_identity_json)
            return

        if publication_state == "published":
            if state != "ready" or not metadata_complete or provider_output_identity_json is None:
                raise SchemaError("invalid Studio ledger published publication relation")
            # A fresh API release may clear the active lease fields; migrated
            # legacy rows may preserve one complete audit tuple. No partial or
            # replaced tuple is a valid published shape.
            JobStore._row_output_identity_json(provider_output_identity_json)
            return

        raise SchemaError("invalid Studio ledger publication relation")

    @staticmethod
    def _validate_persisted_row(row: sqlite3.Row, *, check_output_identity: bool = True) -> None:
        """Fail closed on any corrupted persisted field before a mutation writes."""
        row_contract = JobStore._row_contract(row["row_contract"])
        if row_contract == 4:
            JobStore._validate_v4_row_shape(row)
        else:
            if row["project_id"] is not None or row["destination_identity"] is not None or row["published_identity"] is not None:
                raise SchemaError("invalid Contract-3 identity fields")
            JobStore._row_required_text(row["frozen_destination"], "frozen_destination")
        JobStore._row_required_text(row["studio_job_id"], "studio_job_id")
        JobStore._row_required_text(row["client_request_id"], "client_request_id")
        JobStore._row_required_text(row["provider_id"], "provider_id")
        JobStore._row_required_text(row["target_id"], "target_id")
        if row_contract == 4:
            JobStore._row_optional_nonempty_text(row["provider_job_id"], "provider_job_id")
        else:
            JobStore._row_optional_opaque(row["provider_job_id"], "provider_job_id")
        JobStore._row_required_text(row["connection_id"], "connection_id")
        if row_contract == 3:
            JobStore._row_required_text(row["frozen_destination"], "frozen_destination")
        JobStore._row_required_text(row["requested_basename"], "requested_basename")
        JobStore._row_required_text(row["request_hash"], "request_hash")
        persisted_state = JobStore._row_state(row["state"])
        JobStore._row_stamp_fields(row["stamp_state"], row["stamp_attempt_count"])
        JobStore._row_request_json(row["request_json"])
        JobStore._row_reason_code(row["reason_code"])
        persisted_output_identity = row["provider_output_identity_json"]
        if check_output_identity:
            JobStore._row_output_identity_json(persisted_output_identity)
        persisted_publication_state = JobStore._row_publication_state(row["publication_state"])
        JobStore._row_optional_nonempty_text(row["publication_lease_owner"], "publication_lease_owner")
        JobStore._row_optional_nonempty_text(row["publication_lease_token"], "publication_lease_token")
        JobStore._row_finite_float(row["publication_lease_expires_at"], "publication_lease_expires_at")
        published_path, source_media_sha256, published_file_sha256 = JobStore._row_publication_metadata(
            row["published_path"],
            row["source_media_sha256"],
            row["published_file_sha256"],
        )
        if row_contract == 3 and published_path is not None:
            try:
                _ensure_published_path_is_child(row["frozen_destination"], published_path)
            except (InvalidRequestError, GenerationDomainError) as exc:
                raise SchemaError("invalid Studio ledger published path relation") from exc
        JobStore._validate_persisted_publication_relation(
            state=persisted_state,
            publication_state=persisted_publication_state,
            provider_output_identity_json=persisted_output_identity,
            publication_lease_owner=row["publication_lease_owner"],
            publication_lease_token=row["publication_lease_token"],
            publication_lease_expires_at=row["publication_lease_expires_at"],
            published_path=row["published_path"],
            source_media_sha256=row["source_media_sha256"],
            published_file_sha256=row["published_file_sha256"],
            check_output_identity=check_output_identity,
        )
        JobStore._row_finite_float(row["created_at"], "created_at", required=True)
        JobStore._row_finite_float(row["updated_at"], "updated_at", required=True)
        JobStore._row_finite_float(row["queued_at"], "queued_at")
        JobStore._row_finite_float(row["running_at"], "running_at")
        JobStore._row_finite_float(row["completed_at"], "completed_at")

    @staticmethod
    def _row_to_record(row: sqlite3.Row | None) -> JobRecord:
        if row is None:
            raise JobNotFoundError()
        try:
            JobStore._validate_persisted_row(row)
            state = JobStore._row_state(row["state"])
            stamp_state, stamp_attempt_count = JobStore._row_stamp_fields(
                row["stamp_state"],
                row["stamp_attempt_count"],
            )
            request_json = row["request_json"]
            JobStore._row_request_json(request_json)
            published_path, source_media_sha256, published_file_sha256 = JobStore._row_publication_metadata(
                row["published_path"],
                row["source_media_sha256"],
                row["published_file_sha256"],
            )
            return JobRecord(
                studio_job_id=row["studio_job_id"],
                client_request_id=row["client_request_id"],
                provider_id=row["provider_id"],
                target_id=row["target_id"],
                provider_job_id=row["provider_job_id"],
                connection_id=row["connection_id"],
                request_json=request_json,
                request_hash=row["request_hash"],
                project_id=JobStore._row_project_id(row["project_id"]),
                row_contract=JobStore._row_contract(row["row_contract"]),
                destination_identity=JobStore._row_hex_identity(row["destination_identity"], "destination_identity"),
                published_identity=JobStore._row_hex_identity(row["published_identity"], "published_identity"),
                frozen_destination=row["frozen_destination"],
                requested_basename=row["requested_basename"],
                state=state,
                created_at=JobStore._row_finite_float(row["created_at"], "created_at", required=True),
                updated_at=JobStore._row_finite_float(row["updated_at"], "updated_at", required=True),
                queued_at=JobStore._row_finite_float(row["queued_at"], "queued_at"),
                running_at=JobStore._row_finite_float(row["running_at"], "running_at"),
                completed_at=JobStore._row_finite_float(row["completed_at"], "completed_at"),
                reason_code=JobStore._row_reason_code(row["reason_code"]),
                provider_output_identity_json=JobStore._row_output_identity_json(
                    row["provider_output_identity_json"]
                ),
                publication_state=JobStore._row_publication_state(row["publication_state"]),
                publication_lease_owner=JobStore._row_optional_nonempty_text(
                    row["publication_lease_owner"], "publication_lease_owner"
                ),
                publication_lease_token=JobStore._row_optional_nonempty_text(
                    row["publication_lease_token"], "publication_lease_token"
                ),
                publication_lease_expires_at=JobStore._row_finite_float(
                    row["publication_lease_expires_at"],
                    "publication_lease_expires_at",
                ),
                published_path=published_path,
                source_media_sha256=source_media_sha256,
                published_file_sha256=published_file_sha256,
                stamp_state=stamp_state,
                stamp_attempt_count=stamp_attempt_count,
            )
        except SchemaError:
            raise
        except (KeyError, IndexError, TypeError, ValueError, OverflowError) as exc:
            raise SchemaError("invalid Studio ledger row") from exc


__all__ = [
    "DEFAULT_BASENAME",
    "DEFAULT_CONNECTION_ID",
    "JobNotFoundError",
    "JobRecord",
    "JobStore",
    "JobStoreError",
    "LeaseConflictError",
    "PublicationLease",
    "ProviderBusyError",
    "SCHEMA_VERSION",
    "SchemaError",
    "canonical_request_bytes",
    "canonical_request_hash",
    "canonical_request_json",
]
