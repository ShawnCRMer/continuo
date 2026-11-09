"""R3 mask jobs: BiRefNet background and SAM3 click/box/text masks via ComfyUI.

Loaded unregistered by ``plugin_api`` (like ``comfyui_preview``), so no bare
relative imports and no ``@dataclass``.  One job = one small ComfyUI graph:

* ``background``: LoadImage -> LoadBackgroundRemovalModel(birefnet) ->
  RemoveBackground -> MaskToImage -> PreviewImage (mask = foreground).
* ``points`` / ``box`` / ``text``: LoadImage + CheckpointLoaderSimple(sam3.1) ->
  SAM3_Detect (positive/negative coords JSON, a bounding box, or the
  checkpoint's CLIP text conditioning) -> MaskToImage -> PreviewImage.

The picture is uploaded to ComfyUI's temp folder (``/upload/image`` with
``type=temp``) under its content hash, the mask is read back through ``/view``
and returned as an 8-bit grayscale PNG at source size (255 = selected) with its
bounding box and coverage.  Grow/feather happen in the renderer, so a result
is cached by (picture sha256, mode, prompt).  Every network call goes through
the fixed-origin client in a worker thread; nothing blocks the event loop.
Jobs queue normally behind other work (never ``front``), and are never
cancelled or reordered by this module.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import time
import urllib.request
import uuid
from collections import OrderedDict
from typing import Any, Callable, Mapping

SCHEMA = "studio.image-mask-job/1"
CAPABILITIES_SCHEMA = "studio.image-mask-capabilities/1"
CLIENT_ID = "filmsuite-mask"
SAM3_FILE = "sam3.1_multiplex_fp16.safetensors"
BIREFNET_FILE = "birefnet.safetensors"
MODES = ("background", "points", "box", "text")
MAX_POINTS = 16
MAX_TEXT = 120
UPLOAD_MAX_BYTES = 64 * 1024 * 1024
JOB_TIMEOUT_SECONDS = 900.0
POLL_SECONDS = 0.25
CLASS_TTL_SECONDS = 120.0
CACHE_ENTRIES = 48

_log = logging.getLogger("film_suite.image.mask")


class MaskJobError(Exception):
    """A finite, user-facing failure with a stable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def required_classes(mode: str) -> tuple[str, ...]:
    if mode == "background":
        return ("LoadImage", "LoadBackgroundRemovalModel", "RemoveBackground", "MaskToImage", "PreviewImage")
    base = ("LoadImage", "CheckpointLoaderSimple", "SAM3_Detect", "MaskToImage", "PreviewImage")
    if mode == "points":
        return base + ("PrimitiveString",)
    if mode == "box":
        return base + ("PrimitiveBoundingBox",)
    return base + ("CLIPTextEncode",)


def normalize(request: Mapping[str, Any], width: int, height: int) -> dict[str, Any]:
    """Validate and canonicalise one request against the picture size."""
    mode = request.get("mode")
    if mode not in MODES:
        raise MaskJobError("invalid_mask_request", "Mask request is invalid")
    out: dict[str, Any] = {"mode": mode}
    if mode == "background":
        return out
    threshold = request.get("threshold", 0.5)
    refine = request.get("refine_iterations", 2)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0.0 <= float(threshold) <= 1.0:
        raise MaskJobError("invalid_mask_request", "Mask request is invalid")
    if isinstance(refine, bool) or not isinstance(refine, int) or not 0 <= refine <= 5:
        raise MaskJobError("invalid_mask_request", "Mask request is invalid")
    out["threshold"] = round(float(threshold), 3)
    out["refine_iterations"] = refine
    if mode == "points":
        points = request.get("points")
        if not isinstance(points, list) or not 1 <= len(points) <= MAX_POINTS:
            raise MaskJobError("invalid_mask_request", "Mask request is invalid")
        clean = []
        for point in points:
            if not isinstance(point, Mapping):
                raise MaskJobError("invalid_mask_request", "Mask request is invalid")
            x, y, label = point.get("x"), point.get("y"), point.get("label", 1)
            if any(isinstance(v, bool) or not isinstance(v, int) for v in (x, y)) or label not in (0, 1) \
                    or not (0 <= x < width and 0 <= y < height):
                raise MaskJobError("invalid_mask_request", "Mask request is invalid")
            clean.append({"x": x, "y": y, "label": label})
        if not any(p["label"] == 1 for p in clean):
            raise MaskJobError("invalid_mask_request", "Mask request is invalid")
        out["points"] = clean
    elif mode == "box":
        box = request.get("box")
        if not isinstance(box, list) or len(box) != 4 or any(isinstance(v, bool) or not isinstance(v, int) for v in box):
            raise MaskJobError("invalid_mask_request", "Mask request is invalid")
        x, y, w, h = box
        if x < 0 or y < 0 or w < 2 or h < 2 or x + w > width or y + h > height:
            raise MaskJobError("invalid_mask_request", "Mask request is invalid")
        out["box"] = [x, y, w, h]
    else:
        text = request.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT or any(ord(c) < 32 for c in text):
            raise MaskJobError("invalid_mask_request", "Mask request is invalid")
        out["text"] = " ".join(text.split())
    return out


def build_graph(spec: Mapping[str, Any], image_name: str) -> dict[str, Any]:
    """The API-format graph for one normalised request (byte-stable)."""
    graph: dict[str, Any] = {"1": {"class_type": "LoadImage", "inputs": {"image": image_name}}}
    if spec["mode"] == "background":
        graph["2"] = {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": BIREFNET_FILE}}
        graph["3"] = {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["2", 0], "image": ["1", 0]}}
    else:
        graph["2"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SAM3_FILE}}
        detect: dict[str, Any] = {"model": ["2", 0], "image": ["1", 0], "threshold": spec["threshold"],
                                  "refine_iterations": spec["refine_iterations"], "individual_masks": False}
        if spec["mode"] == "points":
            pos = [{"x": p["x"], "y": p["y"]} for p in spec["points"] if p["label"] == 1]
            neg = [{"x": p["x"], "y": p["y"]} for p in spec["points"] if p["label"] == 0]
            graph["6"] = {"class_type": "PrimitiveString", "inputs": {"value": json.dumps(pos, separators=(",", ":"))}}
            detect["positive_coords"] = ["6", 0]
            if neg:
                graph["7"] = {"class_type": "PrimitiveString", "inputs": {"value": json.dumps(neg, separators=(",", ":"))}}
                detect["negative_coords"] = ["7", 0]
        elif spec["mode"] == "box":
            x, y, w, h = spec["box"]
            graph["6"] = {"class_type": "PrimitiveBoundingBox", "inputs": {"x": x, "y": y, "width": w, "height": h}}
            detect["bboxes"] = ["6", 0]
        else:
            graph["6"] = {"class_type": "CLIPTextEncode", "inputs": {"text": spec["text"], "clip": ["2", 1]}}
            detect["conditioning"] = ["6", 0]
        graph["3"] = {"class_type": "SAM3_Detect", "inputs": detect}
    graph["4"] = {"class_type": "MaskToImage", "inputs": {"mask": ["3", 0]}}
    graph["5"] = {"class_type": "PreviewImage", "inputs": {"images": ["4", 0]}}
    return graph


def cache_key(picture_sha256: str, spec: Mapping[str, Any]) -> str:
    return hashlib.sha256((picture_sha256 + "|" + json.dumps(spec, sort_keys=True, separators=(",", ":"))).encode()).hexdigest()


def _multipart(fields: Mapping[str, str], file_field: str, filename: str, data: bytes) -> tuple[bytes, str]:
    boundary = "----filmsuite" + uuid.uuid4().hex
    parts = []
    for key, value in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode())
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; filename=\"{filename}\"\r\n"
                 "Content-Type: image/png\r\n\r\n".encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), boundary


def _one_object(mask: Any) -> Any:
    """Single-object clean-up for click/box masks: fill holes, keep the largest region.

    SAM3's box path returns a speckled mask on real pictures (measured: 301
    regions and 17,303 hole pixels on a 1024 px apple); one object is one
    region.  Needs numpy + scipy (present in the Hermes venv); without them
    the mask is returned unchanged.
    """
    try:
        import numpy as np  # type: ignore[import-not-found]
        from scipy import ndimage  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - optional clean-up
        return mask
    on = np.asarray(mask) >= 128
    if not on.any():
        return mask
    filled = ndimage.binary_fill_holes(on)
    labels, count = ndimage.label(filled)
    if count > 1:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        filled = labels == int(sizes.argmax())
    return Image.fromarray((filled * 255).astype("uint8"), mode="L")


def mask_from_png(data: bytes, width: int, height: int, *, one_object: bool = False) -> tuple[bytes, list[int] | None, float]:
    """MaskToImage output -> 8-bit L PNG at source size, bbox [x,y,w,h] (>= 128) and coverage %."""
    from PIL import Image  # type: ignore[import-not-found]

    with Image.open(io.BytesIO(data)) as image:
        image.load()
        mask = image.convert("L")
    if mask.size != (width, height):
        mask = mask.resize((width, height), Image.Resampling.BILINEAR)
    if one_object:
        mask = _one_object(mask)
    binary = mask.point(lambda v: 255 if v >= 128 else 0)
    box = binary.getbbox()
    hist = binary.histogram()
    coverage = round(100.0 * hist[255] / float(width * height), 3)
    out = io.BytesIO()
    mask.save(out, format="PNG", optimize=False)
    bbox = None if box is None else [box[0], box[1], box[2] - box[0], box[3] - box[1]]
    return out.getvalue(), bbox, coverage


def refine_sky_png(mask_png: bytes, picture: bytes) -> tuple[bytes, list[int] | None, float]:
    """w24-edit2 (owner decision 2, Photoshop Select Sky + Refine Edge / Lightroom Select Sky): SAM3's sky matte refined
    by colour (image_edit_service.refine_sky): sky seen through small enclosed gaps and in the top rows joins it, thin
    dark lines through it (rigging) leave it. Without numpy/scipy the mask is returned unchanged."""
    from PIL import Image  # type: ignore[import-not-found]
    try:
        import numpy as np  # type: ignore[import-not-found]
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location("film_suite_image_edit_service_sky", Path(__file__).with_name("image_edit_service.py"))
        service = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(service)  # type: ignore[union-attr]
        with Image.open(io.BytesIO(mask_png)) as image:
            mask = np.asarray(image.convert("L"), dtype=np.uint8)
        with Image.open(io.BytesIO(picture)) as image:
            rgb = np.asarray(image.convert("RGB").resize((mask.shape[1], mask.shape[0])), dtype=np.uint8)
        refined, _facts = service.refine_sky(rgb, mask)
    except ImportError:
        refined = None
    out = io.BytesIO()
    if refined is None:
        out.write(mask_png)
        img = Image.open(io.BytesIO(mask_png)).convert("L")
    else:
        img = Image.fromarray(refined, mode="L")
        img.save(out, format="PNG", optimize=False)
    binary = img.point(lambda v: 255 if v >= 128 else 0)
    box = binary.getbbox()
    coverage = round(100.0 * binary.histogram()[255] / float(img.size[0] * img.size[1]), 3)
    return out.getvalue(), (None if box is None else [box[0], box[1], box[2] - box[0], box[3] - box[1]]), coverage


class _Entry:
    __slots__ = ("task", "result", "started", "spec")

    def __init__(self, spec: dict[str, Any]) -> None:
        self.task: asyncio.Task | None = None
        self.result: dict[str, Any] | None = None
        self.started = time.monotonic()
        self.spec = spec


class MaskJobService:
    """In-memory jobs keyed by (picture sha, mode, prompt); bounded LRU of results."""

    def __init__(self, client_factory: Callable[[], Any], opener: Callable[..., Any], *, clock: Callable[[], float] = time.monotonic) -> None:
        self._client_factory = client_factory
        self._opener = opener
        self._clock = clock
        self._entries: "OrderedDict[str, _Entry]" = OrderedDict()
        self._classes: dict[str, tuple[float, dict[str, Any] | None]] = {}

    # -- public -----------------------------------------------------------
    def peek(self, key: str) -> dict[str, Any] | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.result is not None:
            if entry.result.get("state") == "failed":
                # Failures are reported once and never cached: the next request retries.
                self._entries.pop(key, None)
                return entry.result
            self._entries.move_to_end(key)
            return entry.result
        if entry.task is not None and entry.task.done():
            # A task of a closed loop (tests) or one that died without a result.
            self._entries.pop(key, None)
            return None
        return {"state": "running", "elapsed_ms": int((self._clock() - entry.started) * 1000)}

    def start(self, key: str, spec: dict[str, Any], picture: bytes, picture_sha256: str, size: tuple[int, int]) -> dict[str, Any]:
        known = self.peek(key)
        if known is not None:
            return known
        entry = _Entry(spec)
        self._entries[key] = entry
        entry.task = asyncio.get_running_loop().create_task(self._run(key, entry, picture, picture_sha256, size))
        self._trim()
        return {"state": "running", "elapsed_ms": 0}

    async def wait(self, key: str, wait_ms: int) -> dict[str, Any] | None:
        entry = self._entries.get(key)
        if entry is not None and entry.result is None and entry.task is not None and wait_ms > 0:
            try:
                await asyncio.wait_for(asyncio.shield(entry.task), wait_ms / 1000.0)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001 - the result carries the failure
                pass
        return self.peek(key)

    # -- internals --------------------------------------------------------
    def _trim(self) -> None:
        while len(self._entries) > CACHE_ENTRIES:
            for key, entry in self._entries.items():
                if entry.result is not None:
                    self._entries.pop(key)
                    break
            else:
                return

    async def _class_info(self, client: Any, name: str) -> dict[str, Any] | None:
        now = self._clock()
        hit = self._classes.get(name)
        if hit is not None and now - hit[0] < CLASS_TTL_SECONDS:
            return hit[1]
        body = await client.object_info(name)
        info = body.get(name) if isinstance(body, dict) else None
        self._classes[name] = (now, info if isinstance(info, dict) else None)
        return self._classes[name][1]

    async def check_nodes(self, client: Any, mode: str) -> None:
        for name in required_classes(mode):
            if await self._class_info(client, name) is None:
                raise MaskJobError("mask_node_missing", f"ComfyUI has no {name} node")
        loader, field, wanted = (("LoadBackgroundRemovalModel", "bg_removal_name", BIREFNET_FILE) if mode == "background"
                                 else ("CheckpointLoaderSimple", "ckpt_name", SAM3_FILE))
        info = await self._class_info(client, loader) or {}
        spec = ((info.get("input") or {}).get("required") or {}).get(field)
        options: list[Any] = []
        if isinstance(spec, list) and spec:
            if isinstance(spec[0], list):
                options = spec[0]
            elif len(spec) > 1 and isinstance(spec[1], dict):
                options = list(spec[1].get("options") or [])
        if wanted not in options:
            raise MaskJobError("mask_model_missing", f"ComfyUI has no {wanted}")

    async def capabilities(self) -> dict[str, Any]:
        """Which AI mask kinds the engine can run now, without running one (read-only /object_info).

        ``engine`` is ``off`` when ComfyUI does not answer (the kinds are then unknown, not missing);
        otherwise every family names what it lacks.  Subject/Background use BiRefNet, Object/Words/Sky SAM3.
        """
        client = self._client_factory()
        families: dict[str, dict[str, Any]] = {}
        try:
            for family, mode in (("birefnet", "background"), ("sam3_points", "points"), ("sam3_box", "box"), ("sam3_text", "text")):
                try:
                    await self.check_nodes(client, mode)
                    families[family] = {"ok": True}
                except MaskJobError as exc:
                    families[family] = {"ok": False, "missing": exc.message.replace("ComfyUI has no ", "")}
        except Exception as exc:  # noqa: BLE001 - an unreachable engine is an answer, not an error
            if getattr(exc, "reason_code", None) == "unreachable" or isinstance(exc, OSError):
                return {"schema": CAPABILITIES_SCHEMA, "engine": "off", "kinds": {}}
            raise
        pick = lambda *names: next((families[n] for n in names if not families[n]["ok"]), {"ok": True})
        kinds = {"subject": families["birefnet"], "background": families["birefnet"],
                 "object": pick("sam3_points", "sam3_box"), "words": families["sam3_text"], "sky": families["sam3_text"]}
        return {"schema": CAPABILITIES_SCHEMA, "engine": "on", "kinds": kinds}

    def _upload(self, origin: str, name: str, data: bytes) -> str:
        body, boundary = _multipart({"type": "temp", "overwrite": "true"}, "image", name, data)
        request = urllib.request.Request(origin + "/upload/image", data=body, method="POST",
                                         headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with self._opener(request, timeout=30) as response:
            raw = response.read(1 << 20)
        answer = json.loads(raw.decode("utf-8"))
        stored = answer.get("name") if isinstance(answer, dict) else None
        if not isinstance(stored, str) or not stored:
            raise MaskJobError("mask_upload_failed", "ComfyUI did not take the picture")
        sub = answer.get("subfolder") or ""
        return f"{sub + '/' if sub else ''}{stored} [temp]"

    async def _run(self, key: str, entry: _Entry, picture: bytes, picture_sha256: str, size: tuple[int, int]) -> None:
        spec = entry.spec
        started = self._clock()
        result: dict[str, Any]
        prompt_id = None
        try:
            client = self._client_factory()
            await self.check_nodes(client, spec["mode"])
            if len(picture) > UPLOAD_MAX_BYTES:
                raise MaskJobError("mask_picture_too_large", "The picture is too large for a mask job")
            name = await asyncio.to_thread(self._upload, client.origin, f"filmsuite-mask-{picture_sha256[:32]}.png", picture)
            graph = build_graph(spec, name)
            submitted = await client.submit(graph, client_id=CLIENT_ID, extra_data={"studio": {"kind": "mask_job", "mode": spec["mode"]}})
            prompt_id = submitted.get("prompt_id") if isinstance(submitted, dict) else None
            if not isinstance(prompt_id, str) or not prompt_id:
                errors = submitted.get("node_errors") if isinstance(submitted, dict) else None
                raise MaskJobError("mask_job_rejected", "ComfyUI refused the mask job" + (f" ({next(iter(errors))})" if errors else ""))
            submit_at = self._clock()
            outputs = None
            while self._clock() - started < JOB_TIMEOUT_SECONDS:
                history = await client.history(prompt_id)
                item = history.get(prompt_id) if isinstance(history, dict) else None
                if isinstance(item, dict):
                    status = item.get("status") or {}
                    if status.get("status_str") == "error":
                        messages = [m for m in status.get("messages") or [] if isinstance(m, list) and m and m[0] == "execution_error"]
                        detail = (messages[0][1] or {}).get("exception_message", "") if messages and isinstance(messages[0][1], dict) else ""
                        raise MaskJobError("mask_job_failed", ("ComfyUI could not make the mask: " + str(detail).strip().splitlines()[0][:160]) if detail else "ComfyUI could not make the mask")
                    if status.get("completed") or status.get("status_str") == "success":
                        outputs = item.get("outputs") or {}
                        break
                await asyncio.sleep(POLL_SECONDS)
            if outputs is None:
                raise MaskJobError("mask_job_timeout", "The mask job did not finish in time")
            images = ((outputs.get("5") or {}).get("images") or [])
            if not images or not isinstance(images[0], dict):
                raise MaskJobError("mask_job_failed", "ComfyUI made no mask")
            ref = {k: str(images[0][k]) for k in ("filename", "subfolder", "type") if k in images[0]}
            data = await client.view(ref)
            png, bbox, coverage = await asyncio.to_thread(mask_from_png, data, size[0], size[1], one_object=spec["mode"] in ("points", "box"))
            if spec["mode"] == "text" and spec.get("text", "").lower() == "sky":  # w24-edit2: Select Sky + Refine Edge
                png, bbox, coverage = await asyncio.to_thread(refine_sky_png, png, picture)
            done_at = self._clock()
            result = {"state": "done", "prompt_id": prompt_id, "empty": bbox is None,
                      "mask": {"png_b64": base64.b64encode(png).decode("ascii"), "width": size[0], "height": size[1], "bbox": bbox,
                               "coverage_pct": coverage, "sha256": hashlib.sha256(png).hexdigest()},
                      "model": ({"class": "RemoveBackground", "file": BIREFNET_FILE} if spec["mode"] == "background"
                                else {"class": "SAM3_Detect", "file": SAM3_FILE}),
                      "ms": int((done_at - started) * 1000), "comfy_ms": int((done_at - submit_at) * 1000)}
        except MaskJobError as exc:
            result = {"state": "failed", "prompt_id": prompt_id, "error": {"code": exc.code, "message": exc.message}}
        except Exception as exc:  # noqa: BLE001 - finite answer, cause chain to the log
            reason = getattr(exc, "reason_code", None)
            chain, cause = [], exc
            while cause is not None and len(chain) < 6:
                chain.append(f"{type(cause).__name__}:{getattr(cause, 'reason_code', '') or getattr(cause, 'code', '')}")
                cause = cause.__cause__ or cause.__context__
            _log.warning("mask job failed: %s", " <- ".join(chain), exc_info=exc)
            if reason == "unreachable" and prompt_id is None:
                result = {"state": "failed", "prompt_id": None, "error": {"code": "comfyui_unreachable", "message": "ComfyUI is not running."}}
            else:
                result = {"state": "failed", "prompt_id": prompt_id, "error": {"code": "mask_job_failed", "message": "The mask job failed"}}
        entry.result = result


__all__ = ["CAPABILITIES_SCHEMA", "MaskJobError", "MaskJobService", "SCHEMA", "build_graph", "cache_key", "mask_from_png", "normalize", "required_classes"]
