"""Image Box Find: a local detector proposes boxes for a text query (SAM3 text detection).

Loaded unregistered by the box routes (no bare relative imports, no ``@dataclass``).  One
job = one small ComfyUI graph: LoadImage + CheckpointLoaderSimple(sam3.1) -> CLIPTextEncode
(the query) -> SAM3_Detect(individual_masks) -> PreviewAny(bboxes).  The detector's
``BOUNDING_BOX`` list ({x, y, width, height, score} in picture pixels) comes back as the
PreviewAny text; the answer is boxes ``{id, label, score, bbox [x, y, w, h]}``, highest
score first, ids numbered left to right as NKD Box Detections numbers them.

RT-DETR (``RTDETR_detect``) is live too, but no RT-DETR model file is installed and it
only knows the 80 COCO classes; SAM3 takes any words.  Jobs queue behind other work
(never ``front``), are never cancelled, and are cached by (picture sha256, query, threshold).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Callable, Mapping

SCHEMA = "studio.image-box-find/1"
CLIENT_ID = "filmsuite-box-find"
SAM3_FILE = "sam3.1_multiplex_fp16.safetensors"
REQUIRED_CLASSES = ("LoadImage", "CheckpointLoaderSimple", "CLIPTextEncode", "SAM3_Detect", "PreviewAny")
MAX_QUERY = 120
MAX_BOXES = 16
TIMEOUT_SECONDS = 900.0
POLL_SECONDS = 0.25
CACHE_ENTRIES = 32


class FindError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def normalize(request: Mapping[str, Any]) -> dict[str, Any]:
    query = request.get("query")
    if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY or any(ord(c) < 32 for c in query):
        raise FindError("invalid_find_request", f"Find needs words for what to look for (up to {MAX_QUERY} characters)")
    threshold = request.get("threshold", 0.4)
    limit = request.get("max_boxes", 8)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0.05 <= float(threshold) <= 0.95:
        raise FindError("invalid_find_request", "The threshold must be between 0.05 and 0.95")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_BOXES:
        raise FindError("invalid_find_request", f"Find answers 1 to {MAX_BOXES} boxes")
    return {"query": " ".join(query.split()), "threshold": round(float(threshold), 3), "max_boxes": limit}


def build_graph(spec: Mapping[str, Any], image_name: str) -> dict[str, Any]:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SAM3_FILE}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": spec["query"], "clip": ["2", 1]}},
        "4": {"class_type": "SAM3_Detect", "inputs": {"model": ["2", 0], "image": ["1", 0], "conditioning": ["3", 0],
                                                      "threshold": spec["threshold"], "refine_iterations": 0, "individual_masks": True}},
        "5": {"class_type": "PreviewAny", "inputs": {"source": ["4", 1]}},
    }


def cache_key(picture_sha256: str, spec: Mapping[str, Any]) -> str:
    return hashlib.sha256((picture_sha256 + "|" + json.dumps(spec, sort_keys=True, separators=(",", ":"))).encode()).hexdigest()


def parse_boxes(text: Any, width: int, height: int, spec: Mapping[str, Any], ids: Callable[[str, int], list[str]]) -> list[dict[str, Any]]:
    """The PreviewAny text of SAM3's BOUNDING_BOX output -> boxes in whole picture pixels, clipped to the picture."""
    try:
        value = json.loads(text) if isinstance(text, str) else None
    except ValueError:
        value = None
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]  # one frame
    if not isinstance(value, list):
        raise FindError("find_failed", "The detector answered no boxes")
    found = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        try:
            x0, y0 = max(0.0, float(item["x"])), max(0.0, float(item["y"]))
            x1, y1 = min(float(width), float(item["x"]) + float(item["width"])), min(float(height), float(item["y"]) + float(item["height"]))
            score = float(item.get("score", 1.0))
        except (KeyError, TypeError, ValueError):
            continue
        x, y = int(x0), int(y0)
        w, h = int(round(x1)) - x, int(round(y1)) - y
        if w >= 2 and h >= 2:
            found.append({"label": spec["query"], "score": round(score, 4), "bbox": [x, y, w, h]})
    found.sort(key=lambda b: -b["score"])
    found = found[: spec["max_boxes"]]
    for box, box_id in zip(sorted(found, key=lambda b: (b["bbox"][0], b["bbox"][1])), ids(spec["query"], len(found))):
        box["id"] = box_id
    return [{"id": b["id"], "label": b["label"], "score": b["score"], "bbox": b["bbox"]} for b in found]


class FindService:
    """In-memory find jobs keyed by (picture sha, query, threshold); bounded LRU of results."""

    def __init__(self, client_factory: Callable[[], Any], upload: Callable[[str, str, bytes], str], ids: Callable[[str, int], list[str]],
                 *, clock: Callable[[], float] = time.monotonic) -> None:
        self._client_factory = client_factory
        self._upload = upload
        self._ids = ids
        self._clock = clock
        self._entries: "OrderedDict[str, dict[str, Any]]" = OrderedDict()

    def peek(self, key: str) -> dict[str, Any] | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.get("result") is not None:
            result = entry["result"]
            if result.get("state") == "failed":
                self._entries.pop(key, None)  # reported once, never cached
            else:
                self._entries.move_to_end(key)
            return result
        task = entry.get("task")
        if task is not None and task.done():
            self._entries.pop(key, None)
            return None
        return {"state": "running", "elapsed_ms": int((self._clock() - entry["started"]) * 1000)}

    def start(self, key: str, spec: dict[str, Any], picture: bytes, sha: str, size: tuple[int, int]) -> dict[str, Any]:
        known = self.peek(key)
        if known is not None:
            return known
        entry: dict[str, Any] = {"started": self._clock(), "result": None}
        self._entries[key] = entry
        entry["task"] = asyncio.get_running_loop().create_task(self._run(entry, spec, picture, sha, size))
        while len(self._entries) > CACHE_ENTRIES:
            oldest = next((k for k, e in self._entries.items() if e.get("result") is not None), None)
            if oldest is None:
                break
            self._entries.pop(oldest)
        return {"state": "running", "elapsed_ms": 0}

    async def wait(self, key: str, wait_ms: int) -> dict[str, Any] | None:
        entry = self._entries.get(key)
        if entry is not None and entry.get("result") is None and entry.get("task") is not None and wait_ms > 0:
            try:
                await asyncio.wait_for(asyncio.shield(entry["task"]), wait_ms / 1000.0)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001 - the result carries the failure
                pass
        return self.peek(key)

    async def check(self, client: Any) -> None:
        for name in REQUIRED_CLASSES:
            body = await client.object_info(name)
            if not isinstance(body, dict) or not isinstance(body.get(name), dict):
                raise FindError("find_node_missing", f"ComfyUI has no {name} node")
        spec = ((body_of(await client.object_info("CheckpointLoaderSimple")).get("input") or {}).get("required") or {}).get("ckpt_name")
        options = spec[0] if isinstance(spec, list) and spec and isinstance(spec[0], list) else (
            list((spec[1] or {}).get("options") or []) if isinstance(spec, list) and len(spec) > 1 and isinstance(spec[1], dict) else [])
        if SAM3_FILE not in options:
            raise FindError("find_model_missing", f"ComfyUI has no {SAM3_FILE}")

    async def _run(self, entry: dict[str, Any], spec: dict[str, Any], picture: bytes, sha: str, size: tuple[int, int]) -> None:
        started = self._clock()
        prompt_id = None
        try:
            client = self._client_factory()
            await self.check(client)
            name = await asyncio.to_thread(self._upload, client.origin, f"filmsuite-find-{sha[:32]}.png", picture)
            submitted = await client.submit(build_graph(spec, name), client_id=CLIENT_ID, extra_data={"studio": {"kind": "box_find"}})
            prompt_id = submitted.get("prompt_id") if isinstance(submitted, dict) else None
            if not isinstance(prompt_id, str) or not prompt_id:
                raise FindError("find_rejected", "ComfyUI refused the find job")
            outputs = None
            while self._clock() - started < TIMEOUT_SECONDS:
                history = await client.history(prompt_id)
                item = history.get(prompt_id) if isinstance(history, dict) else None
                status = (item or {}).get("status") or {}
                if status.get("status_str") == "error":
                    raise FindError("find_failed", "The detector stopped with an error")
                if isinstance(item, dict) and (status.get("completed") or status.get("status_str") == "success"):
                    outputs = item.get("outputs") or {}
                    break
                await asyncio.sleep(POLL_SECONDS)
            if outputs is None:
                raise FindError("find_timeout", "The detector did not finish in time")
            text = ((outputs.get("5") or {}).get("text") or [None])[0]
            boxes = parse_boxes(text, size[0], size[1], spec, self._ids)
            entry["result"] = {"state": "done", "prompt_id": prompt_id, "boxes": boxes,
                               "detector": {"class": "SAM3_Detect", "file": SAM3_FILE}, "ms": int((self._clock() - started) * 1000)}
        except FindError as exc:
            entry["result"] = {"state": "failed", "prompt_id": prompt_id, "error": {"code": exc.code, "message": exc.message}}
        except Exception as exc:  # noqa: BLE001 - finite answer
            unreachable = getattr(exc, "reason_code", None) == "unreachable" or isinstance(exc, OSError)
            entry["result"] = {"state": "failed", "prompt_id": prompt_id,
                               "error": {"code": "comfyui_unreachable", "message": "ComfyUI is not running."} if unreachable
                               else {"code": "find_failed", "message": "The find job failed"}}


# --- the object inside a Move box (wave 4): SAM3 instances within the box, the best one; None = use the box
SEGMENT_CLIENT_ID = "filmsuite-box-move"
SEGMENT_WAIT_SECONDS = 120.0
_SEGMENT_CACHE: "OrderedDict[str, bytes | None]" = OrderedDict()


def build_segment_graph(image_name: str, bbox: list[int], text: str) -> dict[str, Any]:
    """LoadImage + SAM3 (text + the box) -> one mask per instance (individual_masks) -> PreviewImage (node 8)."""
    graph: dict[str, Any] = {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SAM3_FILE}},
        "6": {"class_type": "PrimitiveBoundingBox", "inputs": {"x": int(bbox[0]), "y": int(bbox[1]), "width": int(bbox[2]), "height": int(bbox[3])}},
        "4": {"class_type": "SAM3_Detect", "inputs": {"model": ["2", 0], "image": ["1", 0], "bboxes": ["6", 0], "threshold": 0.3,
                                                      "refine_iterations": 2, "individual_masks": True}},
        "7": {"class_type": "MaskToImage", "inputs": {"mask": ["4", 0]}},
        "8": {"class_type": "PreviewImage", "inputs": {"images": ["7", 0]}},
    }
    if text:
        graph["3"] = {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": ["2", 1]}}
        graph["4"]["inputs"]["conditioning"] = ["3", 0]
    return graph


def pick_instance(masks: list[bytes], bbox: list[int]) -> tuple[bytes | None, list[dict[str, Any]]]:
    """The single instance mask that best matches the box: highest IoU with the box, then the larger area.

    No union of instances and no closing: the object is exactly one instance as SAM3 drew it.  Returns
    (the chosen 8-bit mask PNG or None, one ``{index, area, iou}`` fact per instance).
    """
    import io as _io
    import numpy as np  # type: ignore[import-not-found]
    from PIL import Image  # type: ignore[import-not-found]

    x, y, w, h = (int(v) for v in bbox)
    best, best_key, facts = None, None, []
    for index, data in enumerate(masks):
        try:
            with Image.open(_io.BytesIO(data)) as image:
                image.load()
                hard = np.asarray(image.convert("L"), dtype=np.uint8) >= 128
        except Exception:  # noqa: BLE001 - an unreadable instance is skipped
            continue
        box = np.zeros(hard.shape, dtype=bool)
        box[y:y + h, x:x + w] = True
        area = int(hard.sum())
        iou = float((hard & box).sum()) / float(max(1, int((hard | box).sum())))
        facts.append({"index": index, "area": area, "iou": round(iou, 4)})
        if area < 16:
            continue
        key = (round(iou, 6), area)
        if best_key is None or key > best_key:
            best_key = key
            out = _io.BytesIO()
            Image.fromarray((hard * 255).astype(np.uint8), mode="L").save(out, format="PNG", compress_level=6)
            best = out.getvalue()
    return best, facts


async def run_for_images(client: Any, graph: Mapping[str, Any], node: str, *, client_id: str, kind: str, wait_seconds: float,
                         extra: Mapping[str, Any] | None = None, clock: Callable[[], float] = time.monotonic) -> list[bytes] | None:
    """Submit one small graph (queued behind other jobs, never cancelled) and return the PNGs of ``node``,
    or None when it fails or does not finish within ``wait_seconds``."""
    submitted = await client.submit(dict(graph), client_id=client_id, extra_data={"studio": {"kind": kind, **dict(extra or {})}})
    prompt_id = submitted.get("prompt_id") if isinstance(submitted, dict) else None
    started = clock()
    while isinstance(prompt_id, str) and prompt_id and clock() - started < wait_seconds:
        history = await client.history(prompt_id)
        item = history.get(prompt_id) if isinstance(history, dict) else None
        status = (item or {}).get("status") or {}
        if status.get("status_str") == "error":
            return None
        if isinstance(item, dict) and (status.get("completed") or status.get("status_str") == "success"):
            out = []
            for image in ((item.get("outputs") or {}).get(node) or {}).get("images") or []:
                data = await client.view({k: image.get(k, "") for k in ("filename", "subfolder", "type")})
                if isinstance(data, (bytes, bytearray)) and bytes(data[:8]) == b"\x89PNG\r\n\x1a\n":
                    out.append(bytes(data))
            return out
        await asyncio.sleep(POLL_SECONDS)
    return None


async def segment_in_box(client: Any, upload: Callable[[str, str, bytes], str], picture: bytes, sha: str, bbox: list[int], text: str,
                         *, clock: Callable[[], float] = time.monotonic) -> bytes | None:
    """The object's mask (PNG, picture size): the SAM3 instance inside ``bbox`` that best matches it
    (``pick_instance``), or None when SAM3 is unavailable, finds nothing, fails or does not finish in time
    (the caller then carries the box with a feathered edge).  Queued behind other jobs like Find; never cancelled."""
    key = hashlib.sha256(json.dumps([sha, list(bbox), text, "instance"]).encode()).hexdigest()
    if key in _SEGMENT_CACHE:
        return _SEGMENT_CACHE[key]
    result = None
    try:
        for name in REQUIRED_CLASSES[:4] + ("PrimitiveBoundingBox", "MaskToImage", "PreviewImage"):
            body = await client.object_info(name)
            if not isinstance(body, dict) or not isinstance(body.get(name), dict):
                return None
        name = await asyncio.to_thread(upload, client.origin, f"filmsuite-find-{sha[:32]}.png", picture)
        images = await run_for_images(client, build_segment_graph(name, bbox, text), "8", client_id=SEGMENT_CLIENT_ID,
                                      kind="box_move_mask", wait_seconds=SEGMENT_WAIT_SECONDS, clock=clock)
        result, _facts = pick_instance(images or [], bbox)
    except Exception:  # noqa: BLE001 - no mask: the box carries
        return None
    if result is not None:
        _SEGMENT_CACHE[key] = bytes(result)
        while len(_SEGMENT_CACHE) > CACHE_ENTRIES:
            _SEGMENT_CACHE.popitem(last=False)
    return bytes(result) if result is not None else None


# --- what stands in front of / next to a Move object (wave 5): other SAM3 instances, kept by the Remove ---
OCCLUDER_TEXT = "building"  # measured on both w4 pictures: the keeper's house (roof, chimneys, gable) as one instance, the tower apart
_OCCLUDER_CACHE: "OrderedDict[str, bytes | None]" = OrderedDict()


def occluder_union(masks: list[bytes], object_png: bytes | None) -> bytes | None:
    """The union (8-bit PNG) of every instance less the object: what else SAM3 sees there.  An instance that
    holds the object and its neighbour together (the headland's tower + house) still gives the neighbour.
    None when fewer than 16 px are left."""
    import io as _io
    import numpy as np  # type: ignore[import-not-found]
    from PIL import Image  # type: ignore[import-not-found]

    obj = None
    if object_png:
        with Image.open(_io.BytesIO(object_png)) as image:
            image.load()
            obj = np.asarray(image.convert("L"), dtype=np.uint8) >= 128
    out = None
    for data in masks:
        try:
            with Image.open(_io.BytesIO(data)) as image:
                image.load()
                hard = np.asarray(image.convert("L"), dtype=np.uint8) >= 128
        except Exception:  # noqa: BLE001 - an unreadable instance is skipped
            continue
        if obj is not None and hard.shape == obj.shape:
            hard = hard & ~obj
        out = hard if out is None else (out | hard if out.shape == hard.shape else out)
    if out is None or int(out.sum()) < 16:
        return None
    buf = _io.BytesIO()
    Image.fromarray((out * 255).astype(np.uint8), mode="L").save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


async def occluders_near(client: Any, upload: Callable[[str, str, bytes], str], picture: bytes, sha: str, bbox: list[int],
                         object_png: bytes | None, *, clock: Callable[[], float] = time.monotonic) -> bytes | None:
    """The other objects next to a Move object (PNG, picture size; ``occluder_union`` of SAM3's instances for
    OCCLUDER_TEXT in ``bbox``, the Move box grown), or None (no SAM3, nothing found, failed, late).  Queued
    behind other jobs like Find; never cancelled."""
    key = hashlib.sha256(json.dumps([sha, list(bbox), OCCLUDER_TEXT, hashlib.sha256(object_png or b"").hexdigest()]).encode()).hexdigest()
    if key in _OCCLUDER_CACHE:
        return _OCCLUDER_CACHE[key]
    try:
        name = await asyncio.to_thread(upload, client.origin, f"filmsuite-find-{sha[:32]}.png", picture)
        images = await run_for_images(client, build_segment_graph(name, bbox, OCCLUDER_TEXT), "8", client_id=SEGMENT_CLIENT_ID,
                                      kind="box_move_occluders", wait_seconds=SEGMENT_WAIT_SECONDS, clock=clock)
        result = occluder_union(images or [], object_png)
    except Exception:  # noqa: BLE001 - no occluders known: the Remove pass takes the object's rings as before
        return None
    _OCCLUDER_CACHE[key] = result
    while len(_OCCLUDER_CACHE) > CACHE_ENTRIES:
        _OCCLUDER_CACHE.popitem(last=False)
    return result


# --- the Remove pass of a Move (wave 4): its own FLUX.3 Klein job on the original picture ---------------
REMOVE_CLIENT_ID = "filmsuite-box-move-remove"
REMOVE_WAIT_SECONDS = 300.0
_REMOVE_CACHE: "OrderedDict[str, bytes]" = OrderedDict()


def remove_cache_key(sha: str, mask_png: bytes, boxes: Any, seed: int, target_id: str) -> str:
    return hashlib.sha256(json.dumps([sha, hashlib.sha256(mask_png).hexdigest(), boxes, int(seed), target_id], sort_keys=True).encode()).hexdigest()


async def remove_pass(client: Any, upload: Callable[[str, str, bytes], str], key: str, graph: Mapping[str, Any], crop_png: bytes,
                      upload_name: str, node: str, target_id: str, *, clock: Callable[[], float] = time.monotonic,
                      kind: str = "box_move_remove") -> bytes | None:
    """The model output (PNG, crop work size) of the Move's Remove pass, or None (failed / not in time).
    ``graph`` already names ``upload_name`` as its picture; one result per key is kept (a replay is free).
    Wave 6: the Move's completion job (``kind`` "box_move_complete", FLUX.2 Klein) runs through here as well."""
    if key in _REMOVE_CACHE:
        _REMOVE_CACHE.move_to_end(key)
        return _REMOVE_CACHE[key]
    if await asyncio.to_thread(upload, client.origin, upload_name, crop_png) != upload_name:
        return None
    images = await run_for_images(client, graph, node, client_id=REMOVE_CLIENT_ID, kind=kind,
                                  wait_seconds=REMOVE_WAIT_SECONDS, extra={"target_id": target_id}, clock=clock)
    if not images:
        return None
    _REMOVE_CACHE[key] = images[0]
    while len(_REMOVE_CACHE) > CACHE_ENTRIES:
        _REMOVE_CACHE.popitem(last=False)
    return images[0]


def body_of(body: Any) -> dict[str, Any]:
    return body.get("CheckpointLoaderSimple") or {} if isinstance(body, dict) else {}


__all__ = ["FindError", "FindService", "SCHEMA", "build_graph", "build_segment_graph", "cache_key", "normalize", "parse_boxes", "pick_instance",
           "remove_cache_key", "remove_pass", "run_for_images", "segment_in_box", "OCCLUDER_TEXT", "occluder_union", "occluders_near"]
