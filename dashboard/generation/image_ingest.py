"""Offline Image result ingest and content-addressed publication.

The module accepts only a bounded iterable of already-authorized PNG bytes.  It
has no provider, URL, network, ComfyUI, route, or Image State dependency.  The
composition root supplies the approved material root and the handle-bound
filesystem primitives through ``boundary``.
"""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import image_contract as contract
from .bounded_download import DownloadError, ExclusiveSink


MAX_BYTES = contract.MAX_BYTES


class ImageIngestError(ValueError):
    """Finite failure for offline ingest/publication."""


class ImageResultIngestor:
    """Ingest one durable ready Image job and publish its validated PNG."""

    def __init__(
        self,
        *,
        job_store: Any,
        material_root: Path | str,
        boundary: Any,
        stage_root: Path | str | None = None,
        sink_factory: Any = ExclusiveSink,
        max_bytes: int = MAX_BYTES,
    ) -> None:
        if boundary is None:
            raise ImageIngestError("path-boundary-required")
        if type(max_bytes) is not int or max_bytes < 1 or max_bytes > MAX_BYTES:
            raise ImageIngestError("max-bytes-invalid")
        self.job_store = job_store
        self.material_root = Path(material_root)
        self.stage_root = Path(stage_root) if stage_root is not None else self.material_root / ".image-staging"
        self.sink_factory = sink_factory
        self.boundary = boundary
        self.max_bytes = max_bytes
        self.materials_root = self.material_root / "materials"
        self._require_directory_root(self.material_root, "material-root-invalid")
        self._require_directory_root(self.stage_root, "stage-root-invalid")
        self._require_directory_root(self.materials_root, "materials-root-invalid")

    def ingest_and_publish(
        self,
        image_producer_job_id: str,
        *,
        stream: Iterable[bytes],
        descriptor_count: int,
        descriptor: dict[str, Any] | None,
        result_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """Serialize the local material boundary before any job or stage inspection."""
        lock_factory = getattr(self.boundary, "project_lock", None)
        if not callable(lock_factory):
            raise ImageIngestError("ingest-lock-missing")
        lock = None
        try:
            lock = lock_factory(self.material_root)
            acquired = lock.acquire()
            if acquired is False:
                raise ImageIngestError("ingest-lock-unavailable")
        except BaseException as exc:
            if lock is not None:
                self._close_lock_quietly(lock)
            if isinstance(exc, ImageIngestError):
                raise
            if not isinstance(exc, Exception):
                raise
            raise ImageIngestError("ingest-lock-unavailable") from exc
        try:
            return self._ingest_and_publish_locked(
                image_producer_job_id,
                stream=stream,
                descriptor_count=descriptor_count,
                descriptor=descriptor,
                result_evidence=result_evidence,
            )
        finally:
            if lock is not None:
                close = getattr(lock, "close", None)
                if callable(close):
                    close()
                else:
                    release = getattr(lock, "release", None)
                    if callable(release):
                        release()

    def _ingest_and_publish_locked(
        self,
        image_producer_job_id: str,
        *,
        stream: Iterable[bytes],
        descriptor_count: int,
        descriptor: dict[str, Any] | None,
        result_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """Run the durable ready -> ingested -> published offline sequence under the boundary lock."""
        job = self.job_store.get_job(image_producer_job_id)
        if job["state"] == "published":
            self._validate_authority(job, image_producer_job_id, descriptor_count, descriptor, result_evidence)
            return job
        if job["state"] not in {"ready", "ingested"}:
            raise ImageIngestError("ingest-state")

        try:
            self._validate_authority(job, image_producer_job_id, descriptor_count, descriptor, result_evidence)
        except ImageIngestError as exc:
            self._quarantine(image_producer_job_id, str(exc), evidence=result_evidence if isinstance(result_evidence, dict) else None)
            raise
        if job["state"] == "ready":
            job = self._ingest_ready(
                job,
                stream=stream,
                descriptor_count=descriptor_count,
                descriptor=descriptor,
                result_evidence=result_evidence,
            )
            if job["state"] != "ingested":
                return job
        return self._publish_ingested(
            job,
            descriptor_count=descriptor_count,
            descriptor=descriptor,
            result_evidence=result_evidence,
        )

    def _validate_authority(
        self,
        job: Mapping[str, Any],
        job_id: str,
        descriptor_count: int,
        descriptor: dict[str, Any] | None,
        evidence: dict[str, Any],
    ) -> None:
        """Validate descriptor/evidence authority before consuming any stream."""
        if type(evidence) is not dict:
            raise ImageIngestError("result-evidence-shape")
        required_evidence = {
            "provider_id",
            "provider_prompt_id",
            "target_id",
            "capability_manifest_sha256",
            "api_graph_sha256",
            "request_snapshot_sha256",
            "context_basis",
            "result_sha256",
        }
        if set(evidence) != required_evidence:
            raise ImageIngestError("result-evidence-shape")
        if job.get("state") in {"ingested", "published"} and job.get("result_evidence") != evidence:
            raise ImageIngestError("result-evidence-replay-conflict")
        operation = {
            "kind": "ingest",
            "image_producer_job_id": job_id,
            "provider_id": evidence.get("provider_id"),
            "provider_prompt_id": evidence.get("provider_prompt_id"),
            "target_id": evidence.get("target_id"),
            "capability_manifest_sha256": evidence.get("capability_manifest_sha256"),
            "api_graph_sha256": evidence.get("api_graph_sha256"),
            "request_snapshot_sha256": evidence.get("request_snapshot_sha256"),
            "context_basis": evidence.get("context_basis"),
            "result_sha256": evidence.get("result_sha256"),
            "descriptors": [copy.deepcopy(descriptor)] if descriptor_count == 1 else [],
        }
        try:
            contract.validate_operation(operation)
        except contract.ContractError as exc:
            raise ImageIngestError(str(exc)) from exc
        expected = {
            "provider_id": contract.PROVIDER_ID,
            "provider_prompt_id": job["provider"]["prompt_id"],
            "target_id": contract.snapshot_target_id(job["request_snapshot"]),
            "capability_manifest_sha256": job["capability_manifest_sha256"],
            "api_graph_sha256": job["api_graph_sha256"],
            "request_snapshot_sha256": job["request_snapshot_sha256"],
            "context_basis": job["context_basis"],
        }
        if any(evidence.get(key) != value for key, value in expected.items()):
            raise ImageIngestError("result-evidence-conflict")
        if type(descriptor_count) is not int or descriptor_count != 1:
            raise ImageIngestError("descriptor-count")

    def _stage_path(self, job: Mapping[str, Any]) -> Path:
        job_id = job["image_producer_job_id"]
        attempt = job["attempt"]
        return self.stage_root / f".{job_id}-attempt-{attempt}.png.part"

    def _open_existing(self, path: Path, *, directory: bool):
        opener = getattr(self.boundary, "open_existing", None)
        if not callable(opener):
            opener = getattr(self.boundary, "_open_existing", None)
        if not callable(opener):
            raise ImageIngestError("path-boundary-open-missing")
        try:
            return opener(path, directory=directory)
        except FileNotFoundError:
            raise
        except Exception as exc:
            if getattr(exc, "code", None) in {"project_not_found", "project_unavailable"}:
                raise FileNotFoundError(path) from exc
            raise ImageIngestError("path-boundary-open-failed") from exc

    @staticmethod
    def _close(handle: Any) -> None:
        close = getattr(handle, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _close_lock_quietly(lock: Any) -> None:
        """Release a lock object even when acquisition did not complete."""

        close = getattr(lock, "close", None)
        try:
            if callable(close):
                close()
                return
            release = getattr(lock, "release", None)
            if callable(release):
                release()
        except Exception:
            # The original acquisition failure is the actionable error.  The
            # close was still attempted, which prevents a NamedProjectLock
            # handle from being abandoned on the failure path.
            pass

    @staticmethod
    def _regular(identity: Any) -> bool:
        return not any(
            bool(getattr(identity, name, False))
            for name in ("is_directory", "is_reparse", "is_reparse_point", "reparse")
        )

    def _require_directory_root(self, path: Path, reason: str) -> None:
        handle = None
        try:
            handle, identity = self._open_existing(path, directory=True)
        except (FileNotFoundError, ImageIngestError) as exc:
            raise ImageIngestError(reason) from exc
        try:
            if not bool(getattr(identity, "is_directory", False)) or any(
                bool(getattr(identity, name, False))
                for name in ("is_reparse", "is_reparse_point", "reparse")
            ):
                raise ImageIngestError(reason)
        finally:
            self._close(handle)

    def _read_handle(self, handle: Any) -> bytes:
        reader = getattr(self.boundary, "read_handle", None)
        if not callable(reader):
            reader = getattr(self.boundary, "read_handle_bytes", None)
        if not callable(reader):
            reader = getattr(self.boundary, "read_bytes", None)
        if not callable(reader):
            raise ImageIngestError("path-boundary-read-missing")
        try:
            data = reader(handle, max_bytes=self.max_bytes)
        except Exception as exc:
            raise ImageIngestError("bounded-read-failed") from exc
        if type(data) is not bytes or len(data) > self.max_bytes:
            raise ImageIngestError("bounded-read-failed")
        return data

    def _read_object(self, path: Path) -> tuple[bytes, Any]:
        handle = identity = None
        try:
            handle, identity = self._open_existing(path, directory=False)
            if not self._regular(identity):
                raise ImageIngestError("publication-nonregular")
            return self._read_handle(handle), identity
        except FileNotFoundError:
            raise
        finally:
            if handle is not None:
                self._close(handle)

    def _read_object_from_parent(self, parent_handle: Any, path: Path) -> tuple[bytes, Any]:
        """Read one published child through the pinned publication parent."""
        opener = getattr(self.boundary, "open_child_existing", None)
        if not callable(opener):
            opener = getattr(self.boundary, "_open_child_existing", None)
        if not callable(opener):
            # Explicit legacy test seams do not expose a child-open primitive;
            # retain their path-based read contract without weakening the
            # canonical boundary implementation.
            return self._read_object(path)
        handle = None
        try:
            handle, identity = opener(
                parent_handle,
                path.parent,
                path.name,
                directory=False,
            )
            if not self._regular(identity):
                raise ImageIngestError("publication-nonregular")
            return self._read_handle(handle), identity
        except FileNotFoundError:
            raise
        except ImageIngestError:
            raise
        except Exception as exc:
            if getattr(exc, "code", None) in {"project_not_found", "project_unavailable"}:
                raise FileNotFoundError(path) from exc
            raise ImageIngestError("path-boundary-open-failed") from exc
        finally:
            if handle is not None:
                self._close(handle)

    def _unlink_owned(
        self,
        path: Path,
        identity: Any,
        *,
        expected_sha256: str | None = None,
        parent_expected: Any | None = None,
    ) -> bool:
        unlink = getattr(self.boundary, "unlink_handle_bound", None)
        if not callable(unlink):
            raise ImageIngestError("path-boundary-unlink-missing")
        try:
            kwargs = {
                "expected_sha256": expected_sha256,
                "max_bytes": MAX_BYTES,
            }
            if parent_expected is not None:
                kwargs["parent_expected"] = parent_expected
            return bool(
                unlink(
                    path,
                    identity,
                    **kwargs,
                )
            )
        except Exception as exc:
            raise ImageIngestError("owned-cleanup-failed") from exc

    def _cleanup_stage(
        self,
        stage: Path,
        identity: Any,
        *,
        expected_sha256: str | None,
    ) -> bool:
        """Delete only the exact owned staging object; never fall back to a pathname unlink."""
        if identity is None:
            return True
        try:
            return self._unlink_owned(stage, identity, expected_sha256=expected_sha256)
        except ImageIngestError:
            return False

    def _make_sink(self, stage: Path) -> ExclusiveSink:
        """Create the canonical stage through one pinned stage-parent handle."""

        # A path-only factory remains an explicit test seam.  Production's
        # default sink is the only path that is allowed to request the native
        # handle-relative CREATE_NEW primitive.
        if self.sink_factory is not ExclusiveSink:
            return self.sink_factory(stage)
        creator = getattr(self.boundary, "create_exclusive_child", None)
        if not callable(creator):
            creator = getattr(self.boundary, "_create_exclusive_child", None)
        if not callable(creator):
            # Older injected doubles use the compatible path-only constructor.
            return self.sink_factory(stage)
        parent_handle = None
        try:
            parent_handle, _parent_identity = self._open_existing(self.stage_root, directory=True)
            return self.sink_factory(
                stage,
                boundary=self.boundary,
                parent_handle=parent_handle,
            )
        except Exception:
            if parent_handle is not None:
                self._close(parent_handle)
            raise

    def _quarantine(
        self,
        job_id: str,
        reason: str,
        *,
        payload: bytes | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            return self.job_store.quarantine(
                job_id,
                reason[:160] or "image-ingest-failed",
                payload=payload,
                evidence=evidence,
            )
        except Exception as exc:
            raise ImageIngestError("quarantine-failed") from exc

    def _validate_bytes(
        self,
        data: bytes,
        job: Mapping[str, Any],
        descriptor_count: int,
        descriptor: dict[str, Any] | None,
        result_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        requested = job["request_snapshot"]["artist_input"]
        try:
            return contract.validate_output(
                data,
                descriptor_count,
                descriptor,
                requested["width"],
                requested["height"],
                result_evidence["result_sha256"],
            )
        except contract.ContractError as exc:
            raise ImageIngestError(str(exc)) from exc

    def _ingest_ready(
        self,
        job: dict[str, Any],
        *,
        stream: Iterable[bytes],
        descriptor_count: int,
        descriptor: dict[str, Any] | None,
        result_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        stage = self._stage_path(job)
        data: bytes | None = None
        stage_identity: Any = None

        # A flushed stage may survive a process restart before record_ingested.
        # Keep the opened identity even when the bounded read fails so cleanup
        # remains object-bound rather than pathname-authoritative.
        stage_handle = None
        try:
            stage_handle, stage_identity = self._open_existing(stage, directory=False)
            if not self._regular(stage_identity):
                return self._quarantine(
                    job["image_producer_job_id"],
                    "staged-output-nonregular",
                    evidence=result_evidence,
                )
            try:
                data = self._read_handle(stage_handle)
            except ImageIngestError:
                self._close(stage_handle)
                stage_handle = None
                if not self._cleanup_stage(stage, stage_identity, expected_sha256=None):
                    return self._quarantine(
                        job["image_producer_job_id"],
                        "staged-output-cleanup-failed",
                        evidence=result_evidence,
                    )
                data = None
                stage_identity = None
        except FileNotFoundError:
            pass
        except ImageIngestError:
            return self._quarantine(
                job["image_producer_job_id"],
                "staged-output-invalid",
                evidence=result_evidence,
            )
        finally:
            if stage_handle is not None:
                self._close(stage_handle)

        if data is not None:
            try:
                facts = self._validate_bytes(data, job, descriptor_count, descriptor, result_evidence)
            except ImageIngestError as exc:
                if stage_identity is None:
                    raise
                try:
                    removed = self._unlink_owned(
                        stage,
                        stage_identity,
                        expected_sha256=hashlib.sha256(data).hexdigest(),
                    )
                    if not removed:
                        raise ImageIngestError("staged-output-cleanup-failed")
                except ImageIngestError:
                    return self._quarantine(job["image_producer_job_id"], "staged-output-cleanup-failed", payload=data, evidence=result_evidence)
                data = None
                stage_identity = None
            if data is not None:
                job = self.job_store.record_ingested(
                    job["image_producer_job_id"],
                    pending_output=facts,
                    result_evidence=dict(result_evidence),
                )
                return job

        sink = None
        total = 0
        stream_error: Exception | None = None
        try:
            sink = self._make_sink(stage)
            try:
                for chunk in stream:
                    if type(chunk) is not bytes:
                        raise DownloadError("download chunk is not bytes")
                    size = len(chunk)
                    if size > self.max_bytes - total:
                        raise DownloadError("download exceeds maximum size")
                    sink.write(chunk)
                    total += size
            except Exception as exc:
                stream_error = exc
            finally:
                sink.worker_finished()

            if stream_error is not None:
                sink.request_abort()
                cleanup_failed = False
                try:
                    sink.abort_owned()
                except Exception:
                    cleanup_failed = True
                return self._quarantine(
                    job["image_producer_job_id"],
                    "staged-output-cleanup-failed" if cleanup_failed else "stream-failed",
                    evidence=result_evidence,
                )
            sink.flush_and_close()
        except FileExistsError:
            # Another caller owns the deterministic exclusive stage. This
            # caller has no authority to abort or quarantine that operation.
            return self.job_store.get_job(job["image_producer_job_id"])
        except ImageIngestError:
            return self._quarantine(job["image_producer_job_id"], "stream-failed", evidence=result_evidence)
        except Exception as exc:
            if sink is not None:
                sink.request_abort()
                cleanup_failed = False
                try:
                    sink.abort_owned()
                except Exception:
                    cleanup_failed = True
            return self._quarantine(
                job["image_producer_job_id"],
                "staged-output-cleanup-failed" if cleanup_failed else "stage-failed",
                evidence=result_evidence,
            )

        try:
            data, stage_identity = self._read_object(stage)
            facts = self._validate_bytes(data, job, descriptor_count, descriptor, result_evidence)
        except ImageIngestError as exc:
            payload = data if isinstance(data, bytes) else None
            reason = str(exc)
            if stage_identity is not None:
                removed = self._cleanup_stage(
                    stage,
                    stage_identity,
                    expected_sha256=hashlib.sha256(payload).hexdigest() if isinstance(payload, bytes) else None,
                )
                if not removed:
                    reason = "staged-output-cleanup-failed"
            return self._quarantine(job["image_producer_job_id"], reason, payload=payload, evidence=result_evidence)

        return self.job_store.record_ingested(
            job["image_producer_job_id"],
            pending_output=facts,
            result_evidence=dict(result_evidence),
        )

    def _link_no_replace(self, source: Any, parent: Any, name: str) -> None:
        link = getattr(self.boundary, "link_no_replace", None)
        if callable(link):
            link(source, parent, name)
            return
        if os.name == "nt":
            link = getattr(self.boundary, "win_link_handle_no_replace", None)
            if not callable(link):
                raise ImageIngestError("path-boundary-link-missing")
            link(source, parent, name)
            return
        link = getattr(self.boundary, "posix_link_no_replace", None)
        if not callable(link):
            link = getattr(self.boundary, "_posix_linkat_empty_path", None)
        if not callable(link):
            raise ImageIngestError("path-boundary-link-missing")
        source_fd = getattr(source, "raw", getattr(source, "fileno", lambda: source)())
        parent_fd = getattr(parent, "raw", getattr(parent, "fileno", lambda: parent)())
        link(source_fd, parent_fd, name)

    def _publish_ingested(
        self,
        job: dict[str, Any],
        *,
        descriptor_count: int,
        descriptor: dict[str, Any] | None,
        result_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        digest = job["pending_output"]["sha256"]
        publication_path = f"materials/{digest}.png"
        target = self.materials_root / f"{digest}.png"
        stage = self._stage_path(job)
        target_created = False
        target_preexisting = False
        target_identity = None
        stage_identity = None
        parent_identity = None
        stage_handle = parent_handle = None
        publication_durable = False
        try:
            try:
                stage_handle, stage_identity = self._open_existing(stage, directory=False)
                if not self._regular(stage_identity):
                    raise ImageIngestError("staged-nonregular")
            except FileNotFoundError:
                stage_handle = None

            try:
                parent_handle, parent_identity = self._open_existing(self.materials_root, directory=True)
            except FileNotFoundError as exc:
                raise ImageIngestError("materials-root-unavailable") from exc

            if stage_handle is not None:
                try:
                    self._link_no_replace(stage_handle, parent_handle, target.name)
                    target_created = True
                except FileExistsError:
                    target_preexisting = True
                    target_created = False
            elif job["state"] == "ingested":
                # Restart may have completed no-replace publication already.
                # Existence and authority are established only by the later
                # handle-bound open/readback, never by pathname inspection.
                target_preexisting = True
            else:
                raise ImageIngestError("missing-staged-output")
            self._boundary_fsync_parent(self.materials_root, parent_handle=parent_handle)
            publication_durable = True
        except ImageIngestError as exc:
            if stage_handle is not None:
                self._close(stage_handle)
                stage_handle = None
            if parent_handle is not None:
                self._close(parent_handle)
                parent_handle = None
            reason = str(exc) or "publication-failed"
            if target_created:
                try:
                    removed = self._unlink_owned(
                        target,
                        target_identity or stage_identity,
                        parent_expected=parent_identity,
                        expected_sha256=digest,
                    )
                    if not removed:
                        reason = "publication-target-cleanup-failed"
                except ImageIngestError:
                    reason = "publication-target-cleanup-failed"
            if not self._cleanup_stage(stage, stage_identity, expected_sha256=digest):
                reason = "staged-output-cleanup-failed"
            return self._quarantine(job["image_producer_job_id"], reason, evidence=result_evidence)
        except OSError:
            if stage_handle is not None:
                self._close(stage_handle)
                stage_handle = None
            if parent_handle is not None:
                self._close(parent_handle)
                parent_handle = None
            reason = "publication-io-failed"
            if target_created:
                try:
                    removed = self._unlink_owned(
                        target,
                        target_identity or stage_identity,
                        parent_expected=parent_identity,
                        expected_sha256=digest,
                    )
                    if not removed:
                        reason = "publication-target-cleanup-failed"
                except ImageIngestError:
                    reason = "publication-target-cleanup-failed"
            if not self._cleanup_stage(stage, stage_identity, expected_sha256=digest):
                reason = "staged-output-cleanup-failed"
            return self._quarantine(job["image_producer_job_id"], reason, evidence=result_evidence)
        finally:
            if stage_handle is not None:
                self._close(stage_handle)
            if parent_handle is not None and not publication_durable:
                self._close(parent_handle)
                parent_handle = None

        try:
            readback, target_identity = self._read_object_from_parent(parent_handle, target)
            facts = self._validate_bytes(
                readback,
                job,
                descriptor_count,
                descriptor,
                result_evidence,
            )
            if facts != job["pending_output"]:
                raise ImageIngestError("publication-readback-mismatch")
            self._boundary_fsync_parent(self.materials_root, parent_handle=parent_handle)
        except (FileNotFoundError, ImageIngestError) as exc:
            target_cleanup_failed = False
            if target_created:
                try:
                    removed = self._unlink_owned(
                        target,
                        target_identity or stage_identity,
                        parent_expected=parent_identity,
                        expected_sha256=digest,
                    )
                    target_cleanup_failed = not removed
                except ImageIngestError:
                    target_cleanup_failed = True
            if isinstance(exc, FileNotFoundError):
                reason = "missing-staged-output"
            elif target_preexisting:
                reason = "publication-collision"
            else:
                reason = str(exc) if isinstance(exc, ImageIngestError) else "publication-readback-mismatch"
            if target_cleanup_failed:
                reason = "publication-target-cleanup-failed"
            if not self._cleanup_stage(stage, stage_identity, expected_sha256=digest):
                reason = "staged-output-cleanup-failed"
            return self._quarantine(job["image_producer_job_id"], reason, evidence=result_evidence)
        finally:
            if parent_handle is not None:
                self._close(parent_handle)
                parent_handle = None

        artifact = {
            "material_id": "imgmat_" + hashlib.sha256(digest.encode("ascii")).hexdigest()[:32],
            "artifact_sha256": digest,
            "byte_size": job["pending_output"]["byte_size"],
            "width": job["pending_output"]["width"],
            "height": job["pending_output"]["height"],
            "format": "png",
            "mime_type": "image/png",
            "artifact_version": 1,
        }
        if stage_identity is None:
            stage_data = None
            try:
                stage_data, stage_identity = self._read_object(stage)
            except FileNotFoundError:
                stage_identity = None
        else:
            stage_data = None
        if stage_identity is not None:
            removed = self._cleanup_stage(
                stage,
                stage_identity,
                expected_sha256=hashlib.sha256(stage_data).hexdigest() if isinstance(stage_data, bytes) else digest,
            )
            if not removed:
                # The final object is valid and durable, but the staging object
                # is still retryable residue.  Leave the durable job in
                # ``ingested`` and let the next call clean this exact object
                # before recording publication.
                raise ImageIngestError("staged-output-cleanup-failed")
        return self.job_store.record_published(
            job["image_producer_job_id"],
            artifact=artifact,
            publication_path=publication_path,
        )

    def _boundary_fsync_parent(self, path: Path, *, parent_handle: Any | None = None) -> None:
        if parent_handle is not None:
            flush_handle = getattr(self.boundary, "fsync_directory_handle", None)
            if not callable(flush_handle):
                flush_handle = getattr(self.boundary, "_fsync_directory_handle", None)
            if callable(flush_handle):
                try:
                    flush_handle(parent_handle)
                except Exception as exc:
                    raise ImageIngestError("parent-flush-failed") from exc
                return
        flush = getattr(self.boundary, "fsync_parent", None)
        if not callable(flush):
            flush = getattr(self.boundary, "_fsync_parent", None)
        if not callable(flush):
            raise ImageIngestError("path-boundary-parent-flush-missing")
        try:
            flush(path)
        except Exception as exc:
            raise ImageIngestError("parent-flush-failed") from exc


ImageIngestor = ImageResultIngestor


__all__ = ["MAX_BYTES", "ImageIngestError", "ImageResultIngestor", "ImageIngestor"]
