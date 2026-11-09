"""Bring a finished run's picture home: fetch once, validate, publish.

A ready job only means ComfyUI finished.  The picture still sits in ComfyUI's
temp output, which ComfyUI clears on its own.  This collector turns ``ready``
into ``published``: it fetches the bytes through the pinned client (``/view``,
loopback only), and hands them to the ingestor, which validates them against
the job's own ledger facts and publishes them content-addressed.

Authority stays in the ledger.  The output descriptor comes from the
coordinator's validated poll, never from a caller, and the evidence is built
from the job record - so a caller cannot point this at another file.
"""
from __future__ import annotations

import asyncio
import hashlib
from typing import Any, Awaitable, Callable, Mapping

from . import image_contract as contract


class ImageCollectError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        self.reason_code = code
        super().__init__(code)


class ImageResultCollector:
    """ready -> published, idempotent, one run at a time per job."""

    def __init__(
        self,
        *,
        job_store: Any,
        ingestor: Any,
        fetch: Callable[[Mapping[str, str]], Awaitable[bytes]],
    ) -> None:
        if job_store is None or ingestor is None or not callable(fetch):
            raise ImageCollectError("collector-owner-required")
        self.job_store = job_store
        self.ingestor = ingestor
        self.fetch = fetch
        self._locks: dict[str, asyncio.Lock] = {}

    @staticmethod
    def _evidence(job: Mapping[str, Any], result_sha256: str) -> dict[str, Any]:
        return {
            "provider_id": contract.PROVIDER_ID,
            "provider_prompt_id": job["provider"]["prompt_id"],
            "target_id": job["provider"]["target_id"],
            "capability_manifest_sha256": job["capability_manifest_sha256"],
            "api_graph_sha256": job["api_graph_sha256"],
            "request_snapshot_sha256": job["request_snapshot_sha256"],
            "context_basis": dict(job["context_basis"]),
            "result_sha256": result_sha256,
        }

    async def collect(self, image_producer_job_id: str, descriptor: Mapping[str, Any] | None) -> dict[str, Any]:
        lock = self._locks.setdefault(image_producer_job_id, asyncio.Lock())
        async with lock:
            job = self.job_store.get_job(image_producer_job_id)
            if job["state"] == "published":
                return job
            if job["state"] != "ready":
                raise ImageCollectError("collect-state")
            if not isinstance(descriptor, Mapping) or any(
                not isinstance(descriptor.get(key), str) for key in ("filename", "subfolder", "type")
            ):
                raise ImageCollectError("collect-descriptor-missing")
            reference = {key: descriptor[key] for key in ("filename", "subfolder", "type")}
            try:
                data = await self.fetch(reference)
            except Exception as exc:  # noqa: BLE001 - transport is not ledger authority
                if getattr(exc, "http_status", None) == 404:
                    # ComfyUI answered and no longer has the file (its temp
                    # output was cleared). Waiting will not bring it back, so
                    # end the job honestly instead of leaving it "ready".
                    self.job_store.quarantine(image_producer_job_id, "output_missing")
                    raise ImageCollectError("collect-output-missing") from exc
                raise ImageCollectError("collect-fetch-failed") from exc
            if type(data) is not bytes or not data or len(data) > contract.MAX_BYTES:
                raise ImageCollectError("collect-fetch-failed")
            evidence = self._evidence(job, hashlib.sha256(data).hexdigest())
            # The ingestor re-validates everything against the job and
            # quarantines on any mismatch; it is synchronous file work.
            return await asyncio.to_thread(
                self.ingestor.ingest_and_publish,
                image_producer_job_id,
                stream=[data],
                descriptor_count=1,
                descriptor=dict(descriptor),
                result_evidence=evidence,
            )


__all__ = ["ImageCollectError", "ImageResultCollector"]
