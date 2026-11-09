"""Provider binding for catalog Image targets, over the existing durable ledger."""
from __future__ import annotations
import asyncio
from collections.abc import Mapping
from . import comfyui_client, image_contract as c, image_workflow_contract as workflow
from . import image_workflow_run as run
from .comfy_failure import failure_reason as _comfy_failure_reason  # lane w24-rest (GEM-056)
from .image_edit_service import upload_crop
from .qwen_image_workflow_adapter import echo_owned


class ImageWorkflowRuntime:
    def __init__(self, client, jobs, reader):
        self.client, self.jobs, self.reader = client, jobs, reader

    def _body(self, job):
        request = job["request_snapshot"]["effective_input"]["workflow"]
        adapter = workflow.compiler(request["target_id"])
        identity = {"provider_id": c.PROVIDER_ID, "image_producer_job_id": job["image_producer_job_id"],
                    "client_request_id": job["provider"]["client_request_id"], "target_id": request["target_id"],
                    "request_snapshot_sha256": job["request_snapshot_sha256"],
                    "capability_manifest_sha256": job["capability_manifest_sha256"], "api_graph_sha256": job["api_graph_sha256"]}
        return adapter, {"prompt": run.compile_request(adapter, request), "client_id": job["provider"]["client_request_id"], "extra_data": {"studio": identity}}

    async def health(self, request):
        adapter = workflow.compiler(request["target_id"])
        graph = run.compile_request(adapter, request)
        info = {}
        for name in sorted({n["class_type"] for n in graph.values()}):
            info.update(await self.client.object_info(name))
        health = adapter.static_health(run.strip(request)[0], info)
        missing = run.lora_health(graph, info)
        if missing:  # a LoRA of the chain the engine does not list
            health = {**health, "state": "missing", "missing_files": sorted(set(health.get("missing_files", [])) | set(missing))}
        return health

    async def submit(self, *, image_producer_job_id, request_snapshot, client_request_id, request_snapshot_sha256=None):
        job = self.jobs.get_job(image_producer_job_id)
        if (c.validate_snapshot(dict(request_snapshot)) != c.validate_snapshot(job["request_snapshot"])
                or client_request_id != job["provider"]["client_request_id"]
                or request_snapshot_sha256 != job["request_snapshot_sha256"]):
            return {"kind": "malformed"}
        adapter, body = self._body(job)
        for ref in request_snapshot["effective_input"]["owned_references"]:
            data, artifact = await asyncio.to_thread(self.reader.read_published_job, ref["image_producer_job_id"])
            if (artifact["artifact_sha256"], artifact["width"], artifact["height"]) != (ref["sha256"], ref["width"], ref["height"]):
                return {"kind": "malformed"}
            name = "filmsuite-ref-" + ref["sha256"][:32] + ".png [temp]"
            await asyncio.to_thread(upload_crop, comfyui_client.urlopen, self.client.origin, name, data)
        receipt = await self.client.submit(body["prompt"], client_id=body["client_id"], extra_data=body["extra_data"])
        pid = receipt.get("prompt_id") if isinstance(receipt, Mapping) else None
        if not isinstance(pid, str) or not pid or len(pid) > 256 or any(ord(ch) < 32 for ch in pid) or receipt.get("node_errors"):
            return {"kind": "malformed"}
        return {"kind": "accepted", "provider_prompt_id": pid}

    async def recover_submit(self, client_request_id, *, image_producer_job_id=None, request_snapshot_sha256=None):
        job = self.jobs.get_job(image_producer_job_id)
        if client_request_id != job["provider"]["client_request_id"] or request_snapshot_sha256 != job["request_snapshot_sha256"]:
            return {"kind": "unresolved"}
        _, body = self._body(job)
        queue, history = await self.client.queue(), await self.client.recent_history(max_items=100)
        found = set()
        candidates = [(pid, entry.get("prompt")) for pid, entry in history.items() if isinstance(entry, Mapping)]
        candidates += [(item[1], item) for key in ("queue_running", "queue_pending") for item in queue.get(key, []) if isinstance(item, (list, tuple)) and len(item) >= 4]
        for pid, prompt in candidates:
            if self._owned(prompt, pid, body):
                found.add(pid)
        return {"kind": "accepted", "provider_prompt_id": next(iter(found))} if len(found) == 1 else {"kind": "unresolved"}

    @staticmethod
    def _owned(prompt, pid, body):
        return (isinstance(prompt, (list, tuple)) and len(prompt) >= 4 and prompt[1] == pid
                and prompt[2] == body["prompt"] and echo_owned(prompt[3], body["extra_data"]))

    async def poll(self, provider_prompt_id, *, expected_identity):
        job = self.jobs.get_job(expected_identity["image_producer_job_id"])
        adapter, body = self._body(job)
        if job["provider"]["prompt_id"] != provider_prompt_id or expected_identity != body["extra_data"]["studio"]:
            return {"kind": "malformed"}
        history = await self.client.history(provider_prompt_id)
        entry = history.get(provider_prompt_id)
        if entry is not None:
            if not self._owned(entry.get("prompt"), provider_prompt_id, body):
                return {"kind": "malformed"}
            status = entry.get("status") or {}
            if status.get("status_str") in {"error", "failed"}:
                return {"kind": "failed", "reason": _comfy_failure_reason(entry)}
            if status.get("status_str") == "success" or status.get("completed") is True:
                manifest, _ = adapter.load_assets(expected_identity["target_id"])
                output = manifest["outputs"]["image"]
                images = (entry.get("outputs", {}).get(output["node_id"]) or {}).get(output["key"])
                if not isinstance(images, list) or len(images) != 1 or images[0].get("type") != output["type"]:
                    return {"kind": "malformed"}
                descriptor = {k: images[0].get(k) for k in ("filename", "subfolder", "type")}
                descriptor.update(mime_type="image/png", extension=".png")
                try:
                    c._validate_descriptor(descriptor)
                    sent = adapter.effective_prompt(entry, body)
                except (ValueError, TypeError, KeyError):
                    return {"kind": "malformed"}
                if sent is None:
                    return {"kind": "malformed"}
                return {"kind": "ready", "output_identity": descriptor, "sent_prompt": sent}
        queue = await self.client.queue()
        for key, kind in (("queue_running", "running"), ("queue_pending", "queued")):
            for item in queue.get(key, []):
                if isinstance(item, (list, tuple)) and len(item) >= 4 and item[1] == provider_prompt_id:
                    return {"kind": kind if self._owned(item, provider_prompt_id, body) else "malformed"}
        return {"kind": "missing"}

    async def cancel(self, provider_prompt_id):
        return {"kind": "unsupported"}
