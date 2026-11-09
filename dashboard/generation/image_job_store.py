"""Dedicated durable SQLite ledger for the U4b Image producer slice.

The ledger is deliberately separate from the video ``JobStore``.  It records
local image authority, immutable request snapshots, submit intent, provider
correlation, lifecycle state, retry identity, and quarantine evidence.  It
never imports or calls a provider.
"""
from __future__ import annotations
from .legacy_plugin_migration import state_root


import copy
import json
import math
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    from hermes_constants import get_hermes_home
except ImportError:  # pragma: no cover - repository tests install Hermes' root
    def get_hermes_home() -> Path:
        return Path.home() / "AppData" / "Local" / "hermes"

from .image_contract import (
    CAPABILITY_SHA256,
    GRAPH_SHA256,
    IMAGE_PRODUCER_JOB_SCHEMA,
    LEGAL_TRANSITIONS,
    PROVIDER_ID,
    STATES,
    TARGET_ID,
    ContractError,
    ImageRequest,
    ImageSchemaError,
    canonical_bytes,
    sha256,
    snapshot_target_id,
    validate_job,
    validate_operation,
    validate_snapshot,
)


ImageContractError = ContractError
SCHEMA_VERSION = 2
DEFAULT_PATH = state_root(Path(get_hermes_home()) / "plugin-data") / "image-producer-jobs.sqlite3"


class ImageJobNotFoundError(ImageContractError):
    """No local image producer job exists for the requested identity."""


class ImageProducerJobStore:
    """SQLite-backed image ledger with fail-closed durable identity checks."""

    def __init__(self, path: str | os.PathLike[str] | None = None, *, clock: Callable[[], float] | None = None) -> None:
        self.path = (
            Path(path).expanduser()
            if path is not None
            else state_root(Path(get_hermes_home()) / "plugin-data") / "image-producer-jobs.sqlite3"
        )
        self._clock = clock
        self._closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass
        self._migrate()
        self._verify_ledger()

    def _verify_ledger(self) -> None:
        """Ledger-wide integrity on open: every job row and every quarantine
        relation, including orphan evidence.  Reads afterwards validate only
        the rows they return; writes repeat this check in their transaction."""
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            self._validate_quarantine_relations(connection)
            connection.execute("COMMIT")
        finally:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("ImageProducerJobStore is closed")
        connection = sqlite3.connect(
            self.path,
            timeout=10.0,
            isolation_level=None,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        journal_mode = str(connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
        if journal_mode != "wal":
            connection.close()
            raise ImageSchemaError("image ledger requires WAL journal mode")
        connection.execute("PRAGMA synchronous=FULL")
        if int(connection.execute("PRAGMA synchronous").fetchone()[0]) != 2:
            connection.close()
            raise ImageSchemaError("image ledger requires FULL synchronous mode")
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _migrate(self) -> None:
        connection = self._connect()
        try:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise ImageSchemaError(f"unsupported image ledger schema version: {version}")
            if version == 0:
                try:
                    connection.executescript(
                        """
                        BEGIN IMMEDIATE;
                        CREATE TABLE image_producer_jobs (
                            image_producer_job_id TEXT PRIMARY KEY,
                            idempotency_key TEXT NOT NULL UNIQUE,
                            client_request_id TEXT NOT NULL UNIQUE,
                            request_snapshot_json TEXT NOT NULL,
                            request_snapshot_sha256 TEXT NOT NULL,
                            provider_prompt_id TEXT,
                            job_json TEXT NOT NULL,
                            state TEXT NOT NULL DEFAULT 'accepted'
                        );
                        CREATE UNIQUE INDEX image_producer_jobs_provider_prompt
                            ON image_producer_jobs(provider_prompt_id)
                            WHERE provider_prompt_id IS NOT NULL;
                        CREATE INDEX image_producer_jobs_state
                            ON image_producer_jobs(state);
                        CREATE TRIGGER image_producer_jobs_identity_immutable
                        BEFORE UPDATE ON image_producer_jobs
                        WHEN NEW.image_producer_job_id IS NOT OLD.image_producer_job_id
                          OR NEW.idempotency_key IS NOT OLD.idempotency_key
                          OR NEW.client_request_id IS NOT OLD.client_request_id
                          OR NEW.request_snapshot_json IS NOT OLD.request_snapshot_json
                          OR NEW.request_snapshot_sha256 IS NOT OLD.request_snapshot_sha256
                        BEGIN
                            SELECT RAISE(ABORT, 'image producer request identity is immutable');
                        END;
                        CREATE TRIGGER image_producer_jobs_provider_prompt_write_once
                        BEFORE UPDATE OF provider_prompt_id ON image_producer_jobs
                        WHEN OLD.provider_prompt_id IS NOT NULL
                          AND NEW.provider_prompt_id IS NOT OLD.provider_prompt_id
                          AND NOT (
                              NEW.provider_prompt_id IS NULL
                              AND json_extract(NEW.job_json, '$.state') = 'accepted'
                              AND json_extract(NEW.job_json, '$.attempt') = json_extract(OLD.job_json, '$.attempt') + 1
                          )
                        BEGIN
                            SELECT RAISE(ABORT, 'provider prompt correlation is write-once');
                        END;
                        CREATE TABLE image_producer_quarantine (
                            record_id TEXT PRIMARY KEY,
                            image_producer_job_id TEXT NOT NULL REFERENCES image_producer_jobs(image_producer_job_id),
                            ordinal INTEGER NOT NULL DEFAULT 0 CHECK (ordinal >= 0),
                            reason TEXT NOT NULL,
                            original_state TEXT,
                            result_sha256 TEXT,
                            evidence_json TEXT,
                            payload BLOB,
                            recoverable INTEGER NOT NULL CHECK (recoverable = 1),
                            created_at REAL NOT NULL,
                            UNIQUE (image_producer_job_id, ordinal)
                        );
                        PRAGMA user_version=2;
                        COMMIT;
                        """
                    )
                    # state is duplicated only for a cheap ledger index.  It is
                    # written together with job_json and never exposed as the
                    # authority independent of the validated envelope.
                except Exception:
                    if connection.in_transaction:
                        connection.execute("ROLLBACK")
                    raise
            elif version == 1:
                try:
                    self._migrate_v1_to_v2(connection)
                except Exception:
                    if connection.in_transaction:
                        connection.execute("ROLLBACK")
                    raise
            self._ensure_time_columns(connection)
        finally:
            connection.close()
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    # Job timestamps (epoch seconds) live in plain columns beside the
    # validated envelope, never inside it: the envelope and its hashes stay
    # byte-identical.  Rows written before the columns existed read NULL,
    # which the Jobs list reports as unknown instead of guessing.
    _TIME_COLUMNS = ("created_at", "submitted_at", "updated_at")

    @classmethod
    def _ensure_time_columns(cls, connection: sqlite3.Connection) -> None:
        present = {row[1] for row in connection.execute("PRAGMA table_info(image_producer_jobs)")}
        missing = [name for name in cls._TIME_COLUMNS if name not in present]
        if not missing:
            return
        connection.execute("BEGIN IMMEDIATE")
        try:
            present = {row[1] for row in connection.execute("PRAGMA table_info(image_producer_jobs)")}
            for name in cls._TIME_COLUMNS:
                if name not in present:
                    connection.execute(f"ALTER TABLE image_producer_jobs ADD COLUMN {name} REAL")
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise

    def _now(self) -> float:
        value = float(self._clock()) if self._clock is not None else time.time()
        return value if math.isfinite(value) and value >= 0 else time.time()

    def job_times(self, job_ids: list[str]) -> dict[str, dict[str, float | None]]:
        """created/submitted/updated epoch seconds per job id (None if unknown)."""
        ids = [self._job_id(job_id) for job_id in job_ids][:500]
        if not ids:
            return {}
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT image_producer_job_id, created_at, submitted_at, updated_at FROM image_producer_jobs"
                f" WHERE image_producer_job_id IN ({','.join('?' for _ in ids)})",
                ids,
            ).fetchall()
        finally:
            connection.close()

        def finite(value: Any) -> float | None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            return float(value) if math.isfinite(float(value)) and value >= 0 else None

        return {
            row[0]: {name: finite(row[name]) for name in self._TIME_COLUMNS}
            for row in rows
        }

    def _migrate_v1_to_v2(self, connection: sqlite3.Connection) -> None:
        """Upgrade the unshipped v1 quarantine table without accepting weak rows."""
        connection.execute("BEGIN IMMEDIATE")
        try:
            jobs: dict[str, dict[str, Any]] = {}
            for row in connection.execute("SELECT * FROM image_producer_jobs").fetchall():
                try:
                    normalized = validate_job(json.loads(row["job_json"]))
                    if (
                        row["image_producer_job_id"] != normalized["image_producer_job_id"]
                        or row["idempotency_key"] != normalized["idempotency_key"]
                        or row["client_request_id"] != normalized["client_request_id"]
                        or row["request_snapshot_json"] != canonical_bytes(normalized["request_snapshot"]).decode("utf-8")
                        or row["request_snapshot_sha256"] != normalized["request_snapshot_sha256"]
                        or row["provider_prompt_id"] != normalized["provider"]["prompt_id"]
                        or row["state"] != normalized["state"]
                    ):
                        raise ImageSchemaError("image ledger identity columns disagree with job envelope")
                except ImageSchemaError:
                    raise
                except Exception as exc:
                    raise ImageSchemaError("invalid persisted image producer job") from exc
                if normalized["image_producer_job_id"] in jobs:
                    raise ImageSchemaError("duplicate persisted image producer job")
                jobs[normalized["image_producer_job_id"]] = normalized

            legacy_rows = connection.execute("SELECT * FROM image_producer_quarantine").fetchall()
            rows_by_id: dict[str, sqlite3.Row] = {}
            for row in legacy_rows:
                try:
                    self._validate_quarantine_row(row)
                    record_id = self._record_id(row["record_id"])
                    owner = row["image_producer_job_id"]
                    if type(owner) is not str or owner not in jobs:
                        raise ImageSchemaError("quarantine record has no valid owning job")
                except ImageSchemaError:
                    raise
                except Exception as exc:
                    raise ImageSchemaError("invalid persisted quarantine record") from exc
                if record_id in rows_by_id:
                    raise ImageSchemaError("duplicate persisted quarantine record")
                rows_by_id[record_id] = row

            ordinals: dict[str, int] = {}
            for job_id, job in jobs.items():
                for ordinal, record_id in enumerate(job["quarantine_record_ids"]):
                    row = rows_by_id.get(record_id)
                    if row is None or row["image_producer_job_id"] != job_id:
                        raise ImageSchemaError("quarantine evidence is not durably owned by its job")
                    ordinals[record_id] = ordinal
            if set(ordinals) != set(rows_by_id):
                raise ImageSchemaError("orphan persisted quarantine record")

            connection.execute("ALTER TABLE image_producer_quarantine RENAME TO image_producer_quarantine_v1")
            connection.execute(
                """
                CREATE TABLE image_producer_quarantine (
                    record_id TEXT PRIMARY KEY,
                    image_producer_job_id TEXT NOT NULL REFERENCES image_producer_jobs(image_producer_job_id),
                    ordinal INTEGER NOT NULL DEFAULT 0 CHECK (ordinal >= 0),
                    reason TEXT NOT NULL,
                    original_state TEXT,
                    result_sha256 TEXT,
                    evidence_json TEXT,
                    payload BLOB,
                    recoverable INTEGER NOT NULL CHECK (recoverable = 1),
                    created_at REAL NOT NULL,
                    UNIQUE (image_producer_job_id, ordinal)
                )
                """
            )
            for record_id, row in rows_by_id.items():
                connection.execute(
                    """
                    INSERT INTO image_producer_quarantine (
                        record_id, image_producer_job_id, ordinal, reason, original_state,
                        result_sha256, evidence_json, payload, recoverable, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["record_id"],
                        row["image_producer_job_id"],
                        ordinals[record_id],
                        row["reason"],
                        row["original_state"],
                        row["result_sha256"],
                        row["evidence_json"],
                        row["payload"],
                        row["recoverable"],
                        row["created_at"],
                    ),
                )
            connection.execute("DROP TABLE image_producer_quarantine_v1")
            connection.execute("PRAGMA user_version=2")
            connection.execute("COMMIT")
        except ImageSchemaError:
            raise
        except Exception as exc:
            raise ImageSchemaError("cannot migrate image ledger schema v1") from exc

    @staticmethod
    def _job_id(value: Any) -> str:
        if type(value) is not str or not value.startswith("imgjob_") or len(value) != 39 or any(char not in "0123456789abcdef" for char in value[7:]):
            raise ImageContractError("image-producer-job-id-invalid")
        return value

    @staticmethod
    def _record_id(value: Any) -> str:
        if type(value) is not str or not value.startswith("imgq_") or len(value) != 37 or any(char not in "0123456789abcdef" for char in value[5:]):
            raise ImageContractError("quarantine-record-id-invalid")
        return value

    @classmethod
    def _validate_quarantine_row(cls, row: sqlite3.Row) -> None:
        """Validate every field before durable quarantine evidence is exposed."""
        try:
            cls._record_id(row["record_id"])
            owner = row["image_producer_job_id"]
            if type(owner) is not str or not re.fullmatch(r"imgjob_[0-9a-f]{32}", owner):
                raise ImageSchemaError("invalid persisted quarantine owner")
            ordinal = row["ordinal"]
            if type(ordinal) is not int or ordinal < 0:
                raise ImageSchemaError("invalid persisted quarantine ordinal")
            reason = row["reason"]
            if type(reason) is not str or not 1 <= len(reason) <= 160:
                raise ImageSchemaError("invalid persisted quarantine reason")
            original_state = row["original_state"]
            if type(original_state) is not str or original_state not in STATES:
                raise ImageSchemaError("invalid persisted quarantine original state")
            result_sha256 = row["result_sha256"]
            if result_sha256 is not None and (
                type(result_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", result_sha256) is None
            ):
                raise ImageSchemaError("invalid persisted quarantine result hash")
            payload = row["payload"]
            if payload is not None and type(payload) is not bytes:
                raise ImageSchemaError("invalid persisted quarantine payload")
            if payload is None:
                if result_sha256 is not None:
                    raise ImageSchemaError("quarantine payload/hash relation is invalid")
            elif result_sha256 != sha256(payload):
                raise ImageSchemaError("quarantine payload/hash relation is invalid")
            evidence_json = row["evidence_json"]
            if evidence_json is not None:
                if type(evidence_json) is not str:
                    raise ImageSchemaError("invalid persisted quarantine evidence")
                evidence = json.loads(evidence_json)
                if type(evidence) is not dict:
                    raise ImageSchemaError("invalid persisted quarantine evidence")
                if canonical_bytes(evidence).decode("utf-8") != evidence_json:
                    raise ImageSchemaError("non-canonical persisted quarantine evidence")
            recoverable = row["recoverable"]
            if type(recoverable) is not int or recoverable != 1:
                raise ImageSchemaError("invalid persisted quarantine recoverability")
            created_at = row["created_at"]
            if type(created_at) not in (int, float) or not math.isfinite(float(created_at)) or created_at < 0:
                raise ImageSchemaError("invalid persisted quarantine timestamp")
        except ImageSchemaError:
            raise
        except (ContractError, KeyError, TypeError, ValueError, OverflowError, json.JSONDecodeError) as exc:
            raise ImageSchemaError("invalid persisted quarantine record") from exc

    @staticmethod
    def _token(value: Any, name: str) -> str:
        if type(value) is not str or not value or len(value) > 128 or any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:-" for char in value):
            raise ImageContractError(f"{name}-invalid")
        return value

    @staticmethod
    def _prompt(value: Any) -> str:
        if type(value) is not str or not 1 <= len(value) <= 256:
            raise ImageContractError("provider-prompt-id-invalid")
        return value

    @staticmethod
    def _decode_job_row(row: sqlite3.Row) -> dict[str, Any]:
        try:
            job = json.loads(row["job_json"])
            normalized = validate_job(job)
            if (
                row["image_producer_job_id"] != normalized["image_producer_job_id"]
                or row["idempotency_key"] != normalized["idempotency_key"]
                or row["client_request_id"] != normalized["client_request_id"]
                or row["request_snapshot_json"] != canonical_bytes(normalized["request_snapshot"]).decode("utf-8")
                or row["request_snapshot_sha256"] != normalized["request_snapshot_sha256"]
                or row["provider_prompt_id"] != normalized["provider"]["prompt_id"]
                or row["state"] != normalized["state"]
            ):
                raise ImageSchemaError("image ledger identity columns disagree with job envelope")
            return normalized
        except ImageSchemaError:
            raise
        except Exception as exc:
            raise ImageSchemaError("invalid persisted image producer job") from exc

    @staticmethod
    def _validate_quarantine_relations(connection: sqlite3.Connection, job: Mapping[str, Any] | None = None) -> None:
        try:
            job_rows = connection.execute("SELECT * FROM image_producer_jobs").fetchall()
            quarantine_rows = connection.execute(
                "SELECT * FROM image_producer_quarantine ORDER BY image_producer_job_id, ordinal"
            ).fetchall()
        except sqlite3.Error as exc:
            raise ImageSchemaError("cannot read image quarantine evidence") from exc
        jobs: dict[str, dict[str, Any]] = {}
        for job_row in job_rows:
            normalized = ImageProducerJobStore._decode_job_row(job_row)
            jobs[normalized["image_producer_job_id"]] = normalized
        if job is not None:
            current_job = validate_job(dict(job))
            jobs[current_job["image_producer_job_id"]] = current_job
        rows_by_owner: dict[str, list[sqlite3.Row]] = {}
        for quarantine_row in quarantine_rows:
            ImageProducerJobStore._validate_quarantine_row(quarantine_row)
            owner = quarantine_row["image_producer_job_id"]
            if owner not in jobs:
                raise ImageSchemaError("image quarantine evidence has an orphan row")
            rows_by_owner.setdefault(owner, []).append(quarantine_row)
        for owner, owner_job in jobs.items():
            record_ids = owner_job["quarantine_record_ids"]
            rows = rows_by_owner.get(owner, [])
            if len(rows) != len(record_ids):
                raise ImageSchemaError("image job quarantine evidence is incomplete")
            for ordinal, (record_id, row) in enumerate(zip(record_ids, rows)):
                if row["record_id"] != record_id or row["ordinal"] != ordinal:
                    raise ImageSchemaError("image job quarantine evidence order or ownership is invalid")

    @classmethod
    def _validate_own_quarantine(cls, job: Mapping[str, Any], rows: list[sqlite3.Row]) -> None:
        """One job's evidence: exactly its listed records, owned, in ordinal order.

        ``rows`` are the quarantine rows whose owner column is this job,
        ordered by ordinal.  This is the per-job half of
        :meth:`_validate_quarantine_relations`; the ledger-wide half (orphan
        rows) runs on open and inside every write transaction.
        """
        job_id = job["image_producer_job_id"]
        record_ids = job["quarantine_record_ids"]
        for row in rows:
            cls._validate_quarantine_row(row)
            if row["image_producer_job_id"] != job_id:
                raise ImageSchemaError("image job quarantine evidence order or ownership is invalid")
        if len(rows) != len(record_ids):
            raise ImageSchemaError("image job quarantine evidence is incomplete")
        for ordinal, (record_id, row) in enumerate(zip(record_ids, rows)):
            if row["record_id"] != record_id or row["ordinal"] != ordinal:
                raise ImageSchemaError("image job quarantine evidence order or ownership is invalid")

    @staticmethod
    def _own_quarantine_rows(connection: sqlite3.Connection, job_id: str) -> list[sqlite3.Row]:
        try:
            return connection.execute(
                "SELECT * FROM image_producer_quarantine WHERE image_producer_job_id = ? ORDER BY ordinal",
                (job_id,),
            ).fetchall()
        except sqlite3.Error as exc:
            raise ImageSchemaError("cannot read image quarantine evidence") from exc

    def _row_job(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        whole_ledger: bool = False,
    ) -> dict[str, Any]:
        """Decode one row and check its own quarantine evidence.

        Reads validate only what they return, so a read costs O(1) rows, not
        O(ledger).  Write transactions pass ``whole_ledger=True`` to keep the
        ledger-wide relation check (orphan rows, every job's evidence) before
        they commit anything.
        """
        normalized = self._decode_job_row(row)
        if whole_ledger:
            self._validate_quarantine_relations(connection, normalized)
        else:
            self._validate_own_quarantine(
                normalized,
                self._own_quarantine_rows(connection, normalized["image_producer_job_id"]),
            )
        return normalized

    @staticmethod
    def _job_values(job: Mapping[str, Any]) -> tuple[Any, ...]:
        normalized = validate_job(dict(job))
        snapshot_json = canonical_bytes(normalized["request_snapshot"]).decode("utf-8")
        return (
            normalized["image_producer_job_id"],
            normalized["idempotency_key"],
            normalized["client_request_id"],
            snapshot_json,
            normalized["request_snapshot_sha256"],
            normalized["provider"]["prompt_id"],
            canonical_bytes(normalized).decode("utf-8"),
            normalized["state"],
        )

    @classmethod
    def _insert_job(cls, connection: sqlite3.Connection, job: Mapping[str, Any], now: float | None = None) -> None:
        connection.execute(
            """
            INSERT INTO image_producer_jobs (
                image_producer_job_id, idempotency_key, client_request_id,
                request_snapshot_json, request_snapshot_sha256,
                provider_prompt_id, job_json, state
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            cls._job_values(job),
        )
        cls._touch(connection, job, time.time() if now is None else now, created=True)

    @staticmethod
    def _touch(connection: sqlite3.Connection, job: Mapping[str, Any], now: float, *, created: bool = False) -> None:
        """Stamp the time columns; submitted_at is set once, when a prompt binds."""
        bound = (job.get("provider") or {}).get("prompt_id") is not None
        connection.execute(
            "UPDATE image_producer_jobs SET updated_at = ?,"
            " created_at = CASE WHEN ? THEN COALESCE(created_at, ?) ELSE created_at END,"
            " submitted_at = CASE WHEN ? THEN COALESCE(submitted_at, ?) ELSE submitted_at END"
            " WHERE image_producer_job_id = ?",
            (now, 1 if created else 0, now, 1 if bound else 0, now, job["image_producer_job_id"]),
        )

    @classmethod
    def _update_job(cls, connection: sqlite3.Connection, job: Mapping[str, Any], now: float | None = None) -> None:
        values = cls._job_values(job)
        cursor = connection.execute(
            """
            UPDATE image_producer_jobs
            SET idempotency_key = ?, client_request_id = ?, request_snapshot_json = ?,
                request_snapshot_sha256 = ?, provider_prompt_id = ?, job_json = ?, state = ?
            WHERE image_producer_job_id = ?
            """,
            (
                values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[0],
            ),
        )
        if cursor.rowcount != 1:
            raise ImageJobNotFoundError("unknown-local-job")
        cls._touch(connection, job, time.time() if now is None else now)

    @staticmethod
    def _fetch_by_id(connection: sqlite3.Connection, job_id: str) -> sqlite3.Row | None:
        return connection.execute(
            "SELECT * FROM image_producer_jobs WHERE image_producer_job_id = ?",
            (job_id,),
        ).fetchone()

    @staticmethod
    def _make_job(idempotency_key: str, client_request_id: str, snapshot: dict[str, Any]) -> dict[str, Any]:
        snapshot_bytes = validate_snapshot(snapshot)
        snapshot_hash = sha256(snapshot_bytes)
        identity = canonical_bytes({
            "idempotency_key": idempotency_key,
            "request_snapshot": snapshot,
            "capability_manifest_sha256": snapshot["capability_manifest_sha256"],
            "api_graph_sha256": snapshot["api_graph_sha256"],
            "context_basis": snapshot["context_basis"],
            "attach_policy": snapshot["attach_policy"],
        })
        job_id = "imgjob_" + __import__("hashlib").sha256(identity).hexdigest()[:32]
        return {
            "schema": IMAGE_PRODUCER_JOB_SCHEMA,
            "image_producer_job_id": job_id,
            "idempotency_key": idempotency_key,
            "client_request_id": client_request_id,
            "request_snapshot": copy.deepcopy(snapshot),
            "request_snapshot_sha256": snapshot_hash,
            "capability_manifest_sha256": snapshot["capability_manifest_sha256"],
            "api_graph_sha256": snapshot["api_graph_sha256"],
            "context_basis": copy.deepcopy(snapshot["context_basis"]),
            "attach_policy": copy.deepcopy(snapshot["attach_policy"]),
            "provider": {
                "provider_id": PROVIDER_ID,
                "target_id": snapshot_target_id(snapshot),
                "client_request_id": client_request_id,
                "prompt_id": None,
            },
            "submit_intent": {
                "attempt": 1,
                "client_request_id": client_request_id,
                "request_snapshot_sha256": snapshot_hash,
            },
            "state": "accepted",
            "attempt": 1,
            "submission_count": 0,
            "artifact": None,
            "revision_count": 0,
            "retry_keys": [],
            "quarantine_record_ids": [],
        }

    def _create_or_get(self, request: ImageRequest) -> tuple[dict[str, Any], bool]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._validate_quarantine_relations(connection)
            existing = connection.execute(
                "SELECT * FROM image_producer_jobs WHERE idempotency_key = ?",
                (request.idempotency_key,),
            ).fetchone()
            if existing is not None:
                job = self._row_job(connection, existing)
                if (
                    job["request_snapshot_sha256"] != sha256(validate_snapshot(dict(request.request_snapshot)))
                    or job["client_request_id"] != request.client_request_id
                ):
                    raise ImageContractError("idempotency-conflict")
                connection.execute("COMMIT")
                return job, True
            by_client = connection.execute(
                "SELECT 1 FROM image_producer_jobs WHERE client_request_id = ?",
                (request.client_request_id,),
            ).fetchone()
            if by_client is not None:
                raise ImageContractError("client-request-id-conflict")
            job = self._make_job(request.idempotency_key, request.client_request_id or request.idempotency_key, dict(request.request_snapshot))
            validate_job(job)
            self._insert_job(connection, job, self._now())
            connection.execute("COMMIT")
            return copy.deepcopy(job), False
        except sqlite3.IntegrityError as exc:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise ImageContractError("image-job-identity-conflict") from exc
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def accept(
        self,
        idempotency_key: str,
        snapshot: dict[str, Any],
        client_request_id: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        request = ImageRequest(idempotency_key, snapshot, client_request_id)
        return self._create_or_get(request)

    def create_or_get_job(
        self,
        idempotency_key: str,
        snapshot: dict[str, Any],
        *,
        client_request_id: str | None = None,
    ) -> dict[str, Any]:
        return self.accept(idempotency_key, snapshot, client_request_id)[0]

    create_job = create_or_get_job

    def get_job(self, image_producer_job_id: str) -> dict[str, Any]:
        image_producer_job_id = self._job_id(image_producer_job_id)
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            row = self._fetch_by_id(connection, image_producer_job_id)
            if row is None:
                raise ImageJobNotFoundError("unknown-local-job")
            job = self._row_job(connection, row)
            connection.execute("COMMIT")
            return job
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def get_by_idempotency_key(self, idempotency_key: str) -> dict[str, Any]:
        idempotency_key = self._token(idempotency_key, "idempotency-key")
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            row = connection.execute("SELECT * FROM image_producer_jobs WHERE idempotency_key = ?", (idempotency_key,)).fetchone()
            if row is None:
                raise ImageJobNotFoundError("unknown-local-job")
            job = self._row_job(connection, row)
            connection.execute("COMMIT")
            return job
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def get_by_client_request_id(self, client_request_id: str) -> dict[str, Any]:
        client_request_id = self._token(client_request_id, "client-request-id")
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            row = connection.execute("SELECT * FROM image_producer_jobs WHERE client_request_id = ?", (client_request_id,)).fetchone()
            if row is None:
                raise ImageJobNotFoundError("unknown-local-job")
            job = self._row_job(connection, row)
            connection.execute("COMMIT")
            return job
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def count_jobs(self) -> int:
        connection = self._connect()
        try:
            self._validate_quarantine_relations(connection)
            return int(connection.execute("SELECT COUNT(*) FROM image_producer_jobs").fetchone()[0])
        finally:
            connection.close()

    def list_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Newest jobs first, read-only. A row that no longer validates is
        skipped rather than hiding every other job from a listing."""
        if type(limit) is not int or not 1 <= limit <= 500:
            raise ImageContractError("list-limit-invalid")
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            rows = connection.execute(
                "SELECT * FROM image_producer_jobs ORDER BY rowid DESC LIMIT ?", (limit,)
            ).fetchall()
            # One batched read of the listed jobs' own quarantine evidence;
            # each row is checked against exactly its own rows, as in get_job.
            ids = [row["image_producer_job_id"] for row in rows if type(row["image_producer_job_id"]) is str]
            evidence: dict[str, list[sqlite3.Row]] = {}
            if ids:
                for quarantine_row in connection.execute(
                    "SELECT * FROM image_producer_quarantine"
                    f" WHERE image_producer_job_id IN ({','.join('?' for _ in ids)})"
                    " ORDER BY image_producer_job_id, ordinal",
                    ids,
                ).fetchall():
                    evidence.setdefault(quarantine_row["image_producer_job_id"], []).append(quarantine_row)
            connection.execute("COMMIT")
            jobs: list[dict[str, Any]] = []
            for row in rows:
                try:
                    job = self._decode_job_row(row)
                    self._validate_own_quarantine(job, evidence.get(job["image_producer_job_id"], []))
                    jobs.append(job)
                except Exception:  # noqa: BLE001 - one bad row must not blank the list
                    continue
            return jobs
        finally:
            connection.close()

    def _mutate(self, job_id: str, mutator: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
        job_id = self._job_id(job_id)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = self._fetch_by_id(connection, job_id)
            if row is None:
                raise ImageJobNotFoundError("unknown-local-job")
            # Own evidence of the stored envelope here; the ledger-wide check
            # below runs with the updated envelope before anything commits.
            job = self._row_job(connection, row)
            updated = mutator(job)
            validate_job(updated)
            self._validate_quarantine_relations(connection, updated)
            self._update_job(connection, updated, self._now())
            connection.execute("COMMIT")
            return copy.deepcopy(updated)
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def transition(self, image_producer_job_id: str, new_state: str, *, reason: str | None = None) -> dict[str, Any]:
        if type(new_state) is not str or new_state not in STATES:
            raise ImageContractError("job-state-invalid")
        if reason is not None and (type(reason) is not str or not 1 <= len(reason) <= 160):
            raise ImageContractError("transition-reason-invalid")
        if new_state == "quarantined":
            current = self.get_job(image_producer_job_id)
            if new_state not in LEGAL_TRANSITIONS[current["state"]]:
                raise ImageContractError(f"illegal-transition:{current['state']}->{new_state}")
            return self.quarantine(image_producer_job_id, reason or "explicit-transition")

        def apply(job: dict[str, Any]) -> dict[str, Any]:
            current = job["state"]
            if new_state not in LEGAL_TRANSITIONS[current]:
                raise ImageContractError(f"illegal-transition:{current}->{new_state}")
            if new_state == "failed":
                job["failure_reason"] = reason or "provider-failed"
            job["state"] = new_state
            return job

        return self._mutate(image_producer_job_id, apply)

    transition_state = transition

    def record_sent_prompt(self, image_producer_job_id: str, text: str) -> dict[str, Any]:
        def apply(job):
            if "sent_prompt" in job and job["sent_prompt"] != text:
                raise ImageContractError("sent-prompt-write-once")
            job["sent_prompt"] = text
            return job
        return self._mutate(image_producer_job_id, apply)

    def mark_submitting(self, image_producer_job_id: str) -> dict[str, Any]:
        return self.transition(image_producer_job_id, "submitting")

    def record_submission_attempt(self, image_producer_job_id: str) -> dict[str, Any]:
        def apply(job: dict[str, Any]) -> dict[str, Any]:
            if job["state"] != "submitting":
                raise ImageContractError("submission-state")
            if job["submission_count"] >= job["attempt"]:
                raise ImageContractError("submission-attempt-already-recorded")
            job["submission_count"] += 1
            return job

        return self._mutate(image_producer_job_id, apply)

    def bind_provider_prompt(self, image_producer_job_id: str, provider_prompt_id: str) -> dict[str, Any]:
        provider_prompt_id = self._prompt(provider_prompt_id)
        image_producer_job_id = self._job_id(image_producer_job_id)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = self._fetch_by_id(connection, image_producer_job_id)
            if row is None:
                raise ImageJobNotFoundError("unknown-local-job")
            # Ledger-wide: the exact-replay branch below commits without a
            # later relation check.
            job = self._row_job(connection, row, whole_ledger=True)
            current = job["provider"]["prompt_id"]
            if current is not None and current != provider_prompt_id:
                raise ImageContractError("provider-prompt-id-write-once")
            owner = connection.execute(
                "SELECT image_producer_job_id FROM image_producer_jobs WHERE provider_prompt_id = ?",
                (provider_prompt_id,),
            ).fetchone()
            if owner is not None and owner[0] != image_producer_job_id:
                raise ImageContractError("provider-correlation-not-owned")
            if current == provider_prompt_id:
                connection.execute("COMMIT")
                return job
            if job["state"] not in {"submitting", "queued", "running", "ready"}:
                reconciling = (
                    job["state"] == "quarantined"
                    and job.get("quarantine_reason") == "submit-reconciliation-required"
                    and job.get("submit_reconciliation", {}).get("status") == "unresolved"
                )
                if not reconciling:
                    raise ImageContractError("provider-prompt-state")
            job["provider"]["prompt_id"] = provider_prompt_id
            job["submission_count"] = max(job["submission_count"], job["attempt"])
            if job["state"] in {"submitting", "quarantined"}:
                job["state"] = "queued"
            job.pop("quarantine_reason", None)
            job.pop("last_quarantine_record_id", None)
            job["submit_reconciliation"] = {
                "status": "bound",
                "attempt": job["attempt"],
                "client_request_id": job["provider"]["client_request_id"],
                "request_snapshot_sha256": job["request_snapshot_sha256"],
                "provider_prompt_id": provider_prompt_id,
            }
            validate_job(job)
            self._validate_quarantine_relations(connection, job)
            self._update_job(connection, job, self._now())
            connection.execute("COMMIT")
            return copy.deepcopy(job)
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    bind_provider_job = bind_provider_prompt
    reconcile_provider_prompt = bind_provider_prompt

    def mark_submit_unresolved(
        self,
        image_producer_job_id: str,
        *,
        payload: bytes | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Quarantine an ambiguous submit without permitting a blind retry.

        This is a ledger-only seam: a caller records that the provider boundary
        became ambiguous, and a later caller may reconcile an explicitly owned
        provider prompt through :meth:`reconcile_submit`.
        """
        image_producer_job_id = self._job_id(image_producer_job_id)
        if payload is not None and type(payload) is not bytes:
            raise ImageContractError("quarantine-payload-invalid")
        if evidence is not None and type(evidence) is not dict:
            raise ImageContractError("quarantine-evidence-invalid")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = self._fetch_by_id(connection, image_producer_job_id)
            if row is None:
                raise ImageJobNotFoundError("unknown-local-job")
            job = self._row_job(connection, row)
            if (
                job["state"] != "submitting"
                or job["provider"]["prompt_id"] is not None
                or job["submission_count"] < job["attempt"]
            ):
                raise ImageContractError("submit-reconciliation-state")
            ordinal = int(connection.execute(
                "SELECT COALESCE(MAX(ordinal), -1) + 1 FROM image_producer_quarantine WHERE image_producer_job_id = ?",
                (image_producer_job_id,),
            ).fetchone()[0])
            record_id = "imgq_" + __import__("hashlib").sha256(
                canonical_bytes({"job_id": image_producer_job_id, "reason": "submit-reconciliation-required", "ordinal": ordinal})
            ).hexdigest()[:32]
            evidence_json = None if evidence is None else canonical_bytes(evidence).decode("utf-8")
            connection.execute(
                """
                INSERT INTO image_producer_quarantine (
                    record_id, image_producer_job_id, ordinal, reason, original_state,
                    result_sha256, evidence_json, payload, recoverable, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    record_id,
                    image_producer_job_id,
                    ordinal,
                    "submit-reconciliation-required",
                    job["state"],
                    None if payload is None else sha256(payload),
                    evidence_json,
                    payload,
                    0.0 if self._clock is None else float(self._clock()),
                ),
            )
            intent = job["submit_intent"]
            job["quarantine_record_ids"].append(record_id)
            job["last_quarantine_record_id"] = record_id
            job["quarantine_reason"] = "submit-reconciliation-required"
            job["submit_reconciliation"] = {
                "status": "unresolved",
                "attempt": intent["attempt"],
                "client_request_id": intent["client_request_id"],
                "request_snapshot_sha256": intent["request_snapshot_sha256"],
            }
            job["state"] = "quarantined"
            validate_job(job)
            self._validate_quarantine_relations(connection, job)
            self._update_job(connection, job, self._now())
            connection.execute("COMMIT")
            return copy.deepcopy(job)
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def reconcile_submit(self, image_producer_job_id: str, provider_prompt_id: str) -> dict[str, Any]:
        """Bind only an externally authorized prompt; never submit from quarantine."""
        image_producer_job_id = self._job_id(image_producer_job_id)
        provider_prompt_id = self._prompt(provider_prompt_id)
        job = self.get_job(image_producer_job_id)
        if job["state"] != "quarantined" or job.get("quarantine_reason") != "submit-reconciliation-required":
            raise ImageContractError("submit-reconciliation-not-required")
        reconciliation = job.get("submit_reconciliation")
        if type(reconciliation) is not dict or reconciliation.get("status") != "unresolved":
            raise ImageContractError("submit-reconciliation-state")
        return self.bind_provider_prompt(image_producer_job_id, provider_prompt_id)

    def submit_operation(self, image_producer_job_id: str) -> dict[str, Any]:
        job = self.get_job(image_producer_job_id)
        intent = job["submit_intent"]
        return validate_operation({
            "kind": "submit",
            "image_producer_job_id": job["image_producer_job_id"],
            "client_request_id": intent["client_request_id"],
            "request_snapshot_sha256": intent["request_snapshot_sha256"],
        })

    def status_operation(self, image_producer_job_id: str, provider_prompt_id: str) -> dict[str, Any]:
        job = self.get_job(image_producer_job_id)
        self._prompt(provider_prompt_id)
        if job["provider"]["prompt_id"] != provider_prompt_id:
            raise ImageContractError("provider-correlation-not-owned")
        return validate_operation({
            "kind": "status",
            "image_producer_job_id": job["image_producer_job_id"],
            "provider_prompt_id": provider_prompt_id,
        })

    def cancel_operation(self, image_producer_job_id: str, provider_prompt_id: str | None) -> dict[str, Any]:
        job = self.get_job(image_producer_job_id)
        if job["state"] == "accepted" and provider_prompt_id is None:
            return validate_operation({
                "kind": "cancel",
                "image_producer_job_id": job["image_producer_job_id"],
                "cancel_scope": "local",
                "provider_prompt_id": None,
            })
        if job["state"] in {"published", "failed", "cancelled", "quarantined"}:
            raise ImageContractError("cancel-terminal")
        if provider_prompt_id is None or job["provider"]["prompt_id"] != provider_prompt_id:
            raise ImageContractError("cancel-provider-correlation-not-owned")
        return validate_operation({
            "kind": "cancel",
            "image_producer_job_id": job["image_producer_job_id"],
            "cancel_scope": "provider",
            "provider_prompt_id": provider_prompt_id,
        })

    def cancel(self, image_producer_job_id: str, provider_prompt_id: str | None) -> dict[str, Any]:
        self.cancel_operation(image_producer_job_id, provider_prompt_id)
        return self.transition(image_producer_job_id, "cancelled")

    def quarantine(
        self,
        image_producer_job_id: str,
        reason: str,
        *,
        payload: bytes | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        image_producer_job_id = self._job_id(image_producer_job_id)
        if type(reason) is not str or not 1 <= len(reason) <= 160:
            raise ImageContractError("quarantine-reason-invalid")
        if payload is not None and type(payload) is not bytes:
            raise ImageContractError("quarantine-payload-invalid")
        if evidence is not None and type(evidence) is not dict:
            raise ImageContractError("quarantine-evidence-invalid")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = self._fetch_by_id(connection, image_producer_job_id)
            if row is None:
                raise ImageJobNotFoundError("unknown-local-job")
            job = self._row_job(connection, row)
            ordinal = int(connection.execute(
                "SELECT COALESCE(MAX(ordinal), -1) + 1 FROM image_producer_quarantine WHERE image_producer_job_id = ?",
                (image_producer_job_id,),
            ).fetchone()[0])
            record_id = "imgq_" + __import__("hashlib").sha256(canonical_bytes({"job_id": image_producer_job_id, "reason": reason, "ordinal": ordinal})).hexdigest()[:32]
            evidence_json = None if evidence is None else canonical_bytes(evidence).decode("utf-8")
            connection.execute(
                """
                INSERT INTO image_producer_quarantine (
                    record_id, image_producer_job_id, ordinal, reason, original_state,
                    result_sha256, evidence_json, payload, recoverable, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    record_id,
                    image_producer_job_id,
                    ordinal,
                    reason,
                    job["state"],
                    None if payload is None else sha256(payload),
                    evidence_json,
                    payload,
                    0.0 if self._clock is None else float(self._clock()),
                ),
            )
            job["quarantine_record_ids"].append(record_id)
            job["last_quarantine_record_id"] = record_id
            job["quarantine_reason"] = reason
            if job["state"] == "ingested":
                for field in ("pending_output", "pending_bytes_sha256", "result_evidence"):
                    job.pop(field, None)
            if job["state"] != "published":
                job["state"] = "quarantined"
            validate_job(job)
            self._validate_quarantine_relations(connection, job)
            self._update_job(connection, job, self._now())
            connection.execute("COMMIT")
            return copy.deepcopy(job)
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def get_quarantine(self, record_id: str) -> dict[str, Any]:
        record_id = self._record_id(record_id)
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            row = connection.execute(
                "SELECT * FROM image_producer_quarantine WHERE record_id = ?",
                (record_id,),
            ).fetchone()
            if row is None:
                raise ImageContractError("unknown-quarantine-record")
            self._validate_quarantine_row(row)
            owner_id = row["image_producer_job_id"]
            if type(owner_id) is not str:
                raise ImageSchemaError("quarantine record has no owning job")
            owner_row = self._fetch_by_id(connection, owner_id)
            if owner_row is None:
                raise ImageSchemaError("quarantine record owner is missing")
            self._row_job(connection, owner_row)
            evidence = None if row["evidence_json"] is None else json.loads(row["evidence_json"])
            if evidence is not None and type(evidence) is not dict:
                raise ImageSchemaError("invalid persisted quarantine evidence")
            result = {
                "record_id": row["record_id"],
                "image_producer_job_id": owner_id,
                "reason": row["reason"],
                "original_state": row["original_state"],
                "result_sha256": row["result_sha256"],
                "evidence": evidence,
                "payload": None if row["payload"] is None else bytes(row["payload"]),
                "recoverable": bool(row["recoverable"]),
            }
            connection.execute("COMMIT")
            return result
        except ImageSchemaError:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        except Exception as exc:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            if isinstance(exc, ImageContractError):
                raise
            raise ImageSchemaError("invalid persisted quarantine record") from exc
        finally:
            connection.close()

    def retry(self, image_producer_job_id: str, retry_key: str) -> dict[str, Any]:
        retry_key = self._token(retry_key, "retry-key")
        self.get_job(image_producer_job_id)
        validate_operation({
            "kind": "retry",
            "image_producer_job_id": image_producer_job_id,
            "retry_key": retry_key,
        })

        def apply(job: dict[str, Any]) -> dict[str, Any]:
            if job.get("quarantine_reason") == "submit-reconciliation-required":
                if job.get("submit_reconciliation", {}).get("status") != "bound":
                    raise ImageContractError("submit-reconciliation-required")
            if job.get("submit_reconciliation", {}).get("status") == "unresolved":
                raise ImageContractError("submit-reconciliation-required")
            if retry_key in job["retry_keys"]:
                return job
            if job["state"] not in {"failed", "cancelled", "quarantined"}:
                raise ImageContractError("retry-state")
            if job["attempt"] >= 16:
                raise ImageContractError("retry-limit")
            attempt = job["attempt"] + 1
            suffix = f":retry-{attempt}-{sha256(retry_key.encode('utf-8'))[:8]}"
            base = job["client_request_id"]
            client_request_id = base[: max(1, 128 - len(suffix))] + suffix
            job["retry_keys"] = list(job["retry_keys"]) + [retry_key]
            job["attempt"] = attempt
            job["state"] = "accepted"
            job["provider"] = {
                "provider_id": PROVIDER_ID,
                "target_id": snapshot_target_id(job["request_snapshot"]),
                "client_request_id": client_request_id,
                "prompt_id": None,
            }
            job["submit_intent"] = {
                "attempt": attempt,
                "client_request_id": client_request_id,
                "request_snapshot_sha256": job["request_snapshot_sha256"],
            }
            job["submission_count"] = 0
            job["artifact"] = None
            job["revision_count"] = 0
            for field in ("pending_output", "pending_bytes_sha256", "result_evidence", "submit_reconciliation", "failure_reason", "quarantine_reason"):
                job.pop(field, None)
            job.pop("last_quarantine_record_id", None)
            return job

        return self._mutate(image_producer_job_id, apply)

    def retry_operation(self, image_producer_job_id: str, retry_key: str) -> dict[str, Any]:
        return validate_operation({
            "kind": "retry",
            "image_producer_job_id": self._job_id(image_producer_job_id),
            "retry_key": retry_key,
        })

    def record_ingested(
        self,
        image_producer_job_id: str,
        *,
        pending_output: dict[str, Any],
        result_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """Durably admit one validated result envelope, with exact replay only."""
        if type(pending_output) is not dict or type(result_evidence) is not dict:
            raise ImageContractError("ingest-envelope-invalid")

        def apply(job: dict[str, Any]) -> dict[str, Any]:
            if job["state"] == "ingested":
                if (
                    job.get("pending_output") == pending_output
                    and job.get("pending_bytes_sha256") == pending_output.get("sha256")
                    and job.get("result_evidence") == result_evidence
                ):
                    return job
                raise ImageContractError("ingest-replay-conflict")
            if job["state"] != "ready":
                raise ImageContractError("ingest-state")
            job["pending_output"] = copy.deepcopy(pending_output)
            job["pending_bytes_sha256"] = pending_output.get("sha256")
            job["result_evidence"] = copy.deepcopy(result_evidence)
            job["state"] = "ingested"
            return job

        return self._mutate(image_producer_job_id, apply)

    def record_published(
        self,
        image_producer_job_id: str,
        *,
        artifact: dict[str, Any],
        publication_path: str,
    ) -> dict[str, Any]:
        """Durably commit the content-addressed publication envelope."""
        if type(artifact) is not dict or type(publication_path) is not str:
            raise ImageContractError("publication-envelope-invalid")

        def apply(job: dict[str, Any]) -> dict[str, Any]:
            if job["state"] == "published":
                if job.get("artifact") == artifact and job.get("publication_path") == publication_path:
                    return job
                raise ImageContractError("publication-replay-conflict")
            if job["state"] != "ingested":
                raise ImageContractError("publication-state")
            job["artifact"] = copy.deepcopy(artifact)
            job["publication_path"] = publication_path
            job["state"] = "published"
            return job

        return self._mutate(image_producer_job_id, apply)

    def record_attached(
        self,
        image_producer_job_id: str,
        *,
        target: list[Any],
        revision_id: str,
    ) -> dict[str, Any]:
        """Durably project one completed Image-State attachment onto its published job."""
        if type(target) is not list or type(revision_id) is not str:
            raise ImageContractError("attachment-envelope-invalid")
        requested_target = copy.deepcopy(target)

        def apply(job: dict[str, Any]) -> dict[str, Any]:
            current_target = job.get("attached_target")
            current_revision = job.get("attached_revision_id")
            if current_target is not None or current_revision is not None:
                if current_target == requested_target and current_revision == revision_id:
                    return job
                raise ImageContractError("attachment-replay-conflict")
            if job["state"] != "published":
                raise ImageContractError("attachment-state")
            job["attached_target"] = copy.deepcopy(requested_target)
            job["attached_revision_id"] = revision_id
            return job

        return self._mutate(image_producer_job_id, apply)

    def close(self) -> None:
        self._closed = True


__all__ = [
    "SCHEMA_VERSION", "DEFAULT_PATH", "ImageContractError", "ImageJobNotFoundError", "ImageSchemaError", "ImageProducerJobStore",
]
