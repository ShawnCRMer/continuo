"""Upscale a picture with the local SeedVR2 7B (``operation: "upscale"``), a new version, never in place.

The Suite owns the operation; the model is SeedVR2 7B int8 (convrot) through ComfyUI's
core SeedVR2 nodes (``comfy_extras/nodes_seedvr.py``), wired like the install's own
blueprint "Upscale Video (SeedVR2 3B Int8)" with the video parts removed:

    LoadImage -> ResizeImageMaskNode (lanczos, the target size) -> SeedVR2Preprocess (pad /16)
    -> VAEEncodeTiled (SeedVR2 VAE) -> SeedVR2Conditioning -> KSampler (1 step, euler/simple)
    -> VAEDecodeTiled -> SeedVR2PostProcessing (crop back, colour correction) -> PreviewImage

The source picture is uploaded to the engine's temp folder under its content name; the
result is an ordinary published ledger job in the source's work (Keep, Edit, Export and
the Media feed take it unchanged).  The snapshot holds the whole graph, so a rebuild from
its own intent gives the same canonical bytes (the ledger's rule for every operation).
Pure Python; loaded as a generation package member.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any
from .comfy_failure import failure_reason as _comfy_failure_reason  # lane w24-rest (GEM-056)

SCHEMA = "studio.image-upscale/1"
TARGET_ID = "filmsuite-seedvr2-upscale-v1"
OUTPUT_NODE = "11"
MODEL_FILE = "seedvr2_7b_int8_convrot.safetensors"
VAE_FILE = "seedvr2_ema_vae_fp16.safetensors"
MODEL_LABEL = "SeedVR2 7B"
FACTORS = (2, 4)
MODES = ("factor", "long_edge")
# The node's own options: VAEEncodeTiled/VAEDecodeTiled tile_size (min 64, step 64 / 32) and
# SeedVR2PostProcessing color_correction_method.  Offered tile sizes are the useful steps.
TILE_SIZES = (256, 512, 768, 1024)
TILE_OVERLAPS = (64, 128)
COLOR_CORRECTIONS = ("lab", "wavelet", "adain", "none")
DEFAULTS = {"tile_size": 512, "tile_overlap": 128, "color_correction": "lab"}
LONG_EDGE_MIN = 512
# 4x of a 2048 picture (8192) peaks near 16 GB on the 32 GB card measured (4x 1024: 16.2 GB);
# larger is refused rather than left to run out of memory in the artist's engine.
OUTPUT_MAX_SIDE = 8192
OUTPUT_MAX_PIXELS = 8192 * 8192
SEED_MAX = 4_294_967_295
CLASSES = ("KSampler", "LoadImage", "PreviewImage", "ResizeImageMaskNode", "SeedVR2Conditioning",
           "SeedVR2PostProcessing", "SeedVR2Preprocess", "UNETLoader", "VAEDecodeTiled", "VAEEncodeTiled",
           "VAELoader")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_JOB = re.compile(r"^imgjob_[0-9a-f]{32}$")
_UPLOAD = re.compile(r"^filmsuite-upscale-[0-9a-f]{32}\.png \[temp\]$")
_KEYS = {"schema", "source_job_id", "source_sha256", "source_size", "image", "mode", "factor", "long_edge",
         "tile_size", "tile_overlap", "color_correction", "model", "vae"}


class UpscaleError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def upload_name(picture_sha256: str) -> str:
    return f"filmsuite-upscale-{picture_sha256[:32]}.png [temp]"


def _even(n: int) -> int:
    return n - (n % 2)


def output_size(source: tuple[int, int] | list[int], mode: str, factor: int | None, long_edge: int | None) -> tuple[int, int]:
    """The picture the graph returns, by the nodes' own rules.

    ResizeImageMaskNode: ``round(side * multiplier)`` ("scale by multiplier") or the longer
    side set to ``longer_size`` and the other ``round(other / longer * longer_size)``
    ("scale longer dimension").  SeedVR2PostProcessing crops back to that size and then to
    even sides.
    """
    w, h = int(source[0]), int(source[1])
    if mode == "factor":
        tw, th = round(w * float(factor)), round(h * float(factor))
    else:
        if h > w:
            tw, th = round((w / h) * long_edge), long_edge
        elif w > h:
            tw, th = long_edge, round((h / w) * long_edge)
        else:
            tw = th = long_edge
    return _even(int(tw)), _even(int(th))


def normalize_request(raw: Mapping[str, Any], source_size: tuple[int, int] | list[int]) -> dict[str, Any]:
    """The caller's upscale intent -> canonical settings (finite refusals, never a guess)."""
    if not isinstance(raw, Mapping):
        raise UpscaleError("invalid_upscale", "The upscale request is invalid")
    factor, long_edge = raw.get("factor"), raw.get("long_edge")
    if (factor is None) == (long_edge is None):
        raise UpscaleError("invalid_upscale", "Upscale needs exactly one of a factor (2 or 4) or a target long edge")
    if factor is not None:
        if isinstance(factor, bool) or factor not in FACTORS:
            raise UpscaleError("invalid_upscale", "The factor is 2 or 4")
        mode = "factor"
    else:
        if isinstance(long_edge, bool) or not isinstance(long_edge, int) or not LONG_EDGE_MIN <= long_edge <= OUTPUT_MAX_SIDE \
                or long_edge % 2:
            raise UpscaleError("invalid_upscale", f"The long edge is an even number from {LONG_EDGE_MIN} to {OUTPUT_MAX_SIDE}")
        if long_edge <= max(int(source_size[0]), int(source_size[1])):
            raise UpscaleError("invalid_upscale", "The long edge must be larger than the picture's")
        mode = "long_edge"
    out = {"mode": mode, "factor": int(factor) if factor is not None else None,
           "long_edge": int(long_edge) if long_edge is not None else None}
    for key, allowed in (("tile_size", TILE_SIZES), ("tile_overlap", TILE_OVERLAPS), ("color_correction", COLOR_CORRECTIONS)):
        value = raw.get(key, DEFAULTS[key])
        if isinstance(value, bool) or value not in allowed:
            raise UpscaleError("invalid_upscale", f"{key.replace('_', ' ').capitalize()} must be one of {', '.join(map(str, allowed))}")
        out[key] = value
    seed = raw.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= SEED_MAX:
        raise UpscaleError("invalid_upscale", "The seed is invalid")
    out["seed"] = seed
    width, height = output_size(source_size, mode, out["factor"], out["long_edge"])
    if width > OUTPUT_MAX_SIDE or height > OUTPUT_MAX_SIDE or width * height > OUTPUT_MAX_PIXELS or min(width, height) < 16:
        raise UpscaleError("upscale_too_large", f"The upscaled picture would be {width} x {height}; at most {OUTPUT_MAX_SIDE} per side")
    out["width"], out["height"] = width, height
    return out


def graph(up: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """The whole SeedVR2 image graph for one upscale (no free branches)."""
    if up["mode"] == "factor":
        resize = {"resize_type": "scale by multiplier", "resize_type.multiplier": float(up["factor"])}
    else:
        resize = {"resize_type": "scale longer dimension", "resize_type.longer_size": int(up["long_edge"])}
    tiles = {"tile_size": up["tile_size"], "overlap": up["tile_overlap"], "temporal_size": 64, "temporal_overlap": 8}
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": up["image"]}},
        "2": {"class_type": "ResizeImageMaskNode", "inputs": {"input": ["1", 0], **resize, "scale_method": "lanczos"}},
        "3": {"class_type": "SeedVR2Preprocess", "inputs": {"resized_images": ["2", 0]}},
        "4": {"class_type": "UNETLoader", "inputs": {"unet_name": up["model"], "weight_dtype": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": up["vae"]}},
        "6": {"class_type": "VAEEncodeTiled", "inputs": {"pixels": ["3", 0], "vae": ["5", 0], **tiles}},
        "7": {"class_type": "SeedVR2Conditioning", "inputs": {"model": ["4", 0], "vae_conditioning": ["6", 0]}},
        "8": {"class_type": "KSampler", "inputs": {"model": ["4", 0], "positive": ["7", 0], "negative": ["7", 1],
                                                   "latent_image": ["6", 0], "seed": seed, "steps": 1, "cfg": 1.0,
                                                   "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["8", 0], "vae": ["5", 0], **tiles}},
        "10": {"class_type": "SeedVR2PostProcessing", "inputs": {"images": ["9", 0], "original_resized_images": ["2", 0],
                                                                 "color_correction_method": up["color_correction"]}},
        OUTPUT_NODE: {"class_type": "PreviewImage", "inputs": {"images": ["10", 0]}},
    }


MANIFEST = {"schema": SCHEMA, "target_id": TARGET_ID, "classes": list(CLASSES), "model": MODEL_FILE, "vae": VAE_FILE,
            "outputs": {"image": {"node_id": OUTPUT_NODE, "key": "images", "type": "temp"}}}
CAPABILITY_SHA256 = hashlib.sha256(_canonical(MANIFEST)).hexdigest()
# The graph's shape (the values that change per upscale are placeholders).
_SHAPE = {"image": "{image}", "mode": "factor", "factor": 2, "long_edge": None, "tile_size": 512, "tile_overlap": 128,
          "color_correction": "lab", "model": MODEL_FILE, "vae": VAE_FILE}
GRAPH_SHA256 = hashlib.sha256(_canonical(graph(_SHAPE, 0))).hexdigest()
GRAPH_LONG_EDGE_SHA256 = hashlib.sha256(_canonical(graph({**_SHAPE, "mode": "long_edge", "factor": None, "long_edge": 4096}, 0))).hexdigest()


def label(up: Mapping[str, Any]) -> str:
    return f"Upscale {up['factor']}x" if up["mode"] == "factor" else f"Upscale to {up['long_edge']} px"


def build_snapshot(up: Mapping[str, Any], seed: int, width: int, height: int, basis: dict[str, Any],
                   policy: dict[str, Any]) -> dict[str, Any]:
    """The ledger request snapshot of one upscale (validated, canonical)."""
    e = dict(up)
    if set(e) != _KEYS or e["schema"] != SCHEMA or e["mode"] not in MODES:
        raise UpscaleError("invalid_upscale", "upscale-shape")
    if not isinstance(e["source_job_id"], str) or not _JOB.match(e["source_job_id"]):
        raise UpscaleError("invalid_upscale", "upscale-source")
    if not isinstance(e["source_sha256"], str) or not _HEX64.match(e["source_sha256"]):
        raise UpscaleError("invalid_upscale", "upscale-sha256")
    if e["image"] != upload_name(e["source_sha256"]) or not _UPLOAD.match(e["image"]):
        raise UpscaleError("invalid_upscale", "upscale-image")
    size = e["source_size"]
    if not isinstance(size, list) or len(size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 16_384 for v in size):
        raise UpscaleError("invalid_upscale", "upscale-source-size")
    if e["model"] != MODEL_FILE or e["vae"] != VAE_FILE:
        raise UpscaleError("invalid_upscale", "upscale-files")
    raw = {k: e[k] for k in ("tile_size", "tile_overlap", "color_correction")}
    raw["seed"] = seed
    raw["factor" if e["mode"] == "factor" else "long_edge"] = e["factor"] if e["mode"] == "factor" else e["long_edge"]
    if (e["mode"] == "factor" and e["long_edge"] is not None) or (e["mode"] == "long_edge" and e["factor"] is not None):
        raise UpscaleError("invalid_upscale", "upscale-mode")
    settings = normalize_request(raw, size)
    if (settings["width"], settings["height"]) != (width, height):
        raise UpscaleError("invalid_upscale", "upscale-output-size")
    return {
        "operation": "upscale",
        "target_id": TARGET_ID,
        "artist_input": {"prompt": label(e), "seed": seed, "width": width, "height": height},
        "effective_input": {"upscale": copy.deepcopy(e), "graph": graph(e, seed)},
        "context_basis": copy.deepcopy(basis),
        "attach_policy": copy.deepcopy(policy),
        "capability_manifest_sha256": CAPABILITY_SHA256,
        "api_graph_sha256": GRAPH_SHA256 if e["mode"] == "factor" else GRAPH_LONG_EDGE_SHA256,
    }


def is_upscale_snapshot(snapshot: Any) -> bool:
    return isinstance(snapshot, Mapping) and snapshot.get("operation") == "upscale" and snapshot.get("target_id") == TARGET_ID


def rebuild_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild from the snapshot's own intent (the ledger compares canonical bytes)."""
    effective, artist = snapshot.get("effective_input"), snapshot.get("artist_input")
    if not isinstance(effective, Mapping) or set(effective) != {"upscale", "graph"} or not isinstance(artist, Mapping) \
            or set(artist) != {"prompt", "seed", "width", "height"}:
        raise UpscaleError("invalid_upscale", "upscale-snapshot-shape")
    return build_snapshot(effective["upscale"], artist.get("seed"), artist.get("width"), artist.get("height"),
                          snapshot["context_basis"], snapshot["attach_policy"])


def recipe(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """The settings an upscale ran with (for the export recipe and the History row)."""
    up = snapshot["effective_input"]["upscale"]
    return {"operation": "upscale", "model": MODEL_LABEL, "model_file": up["model"], "vae_file": up["vae"],
            "label": label(up), "mode": up["mode"], "factor": up["factor"], "long_edge": up["long_edge"],
            "tile_size": up["tile_size"], "tile_overlap": up["tile_overlap"], "color_correction": up["color_correction"],
            "seed": snapshot["artist_input"]["seed"], "parent_sha256": up["source_sha256"],
            "parent_job_id": up["source_job_id"], "source_size": list(up["source_size"]),
            "width": snapshot["artist_input"]["width"], "height": snapshot["artist_input"]["height"]}


def static_health(object_info: Mapping[str, Any]) -> dict[str, Any]:
    """Classes and the two files from /object_info; never runs anything."""
    missing_classes = sorted(name for name in CLASSES if not isinstance(object_info.get(name), Mapping))

    def options(class_name: str, field: str) -> list[Any]:
        spec = (((object_info.get(class_name) or {}).get("input") or {}).get("required") or {}).get(field)
        if isinstance(spec, list) and spec:
            if isinstance(spec[0], list):
                return spec[0]
            if len(spec) > 1 and isinstance(spec[1], Mapping):
                return list(spec[1].get("options") or [])
        return []

    missing_files = [name for cls, field, name in (("UNETLoader", "unet_name", MODEL_FILE), ("VAELoader", "vae_name", VAE_FILE))
                     if cls not in missing_classes and name not in options(cls, field)]
    return {"state": "missing" if missing_classes or missing_files else "ready_static", "missing_classes": missing_classes,
            "missing_files": missing_files, "inference_tested": False}


class UpscaleAdapter:
    """Provider seam for upscale jobs: submit the snapshot's graph, read its one output."""

    target_id = TARGET_ID

    def __init__(self, client: Any) -> None:
        self.client = client

    @staticmethod
    def _identity(job_id: str, client_request_id: str, snapshot_sha: str | None, graph_sha: str) -> dict[str, Any]:
        return {"provider_id": "local-comfyui", "image_producer_job_id": job_id, "client_request_id": client_request_id,
                "target_id": TARGET_ID, "request_snapshot_sha256": snapshot_sha,
                "capability_manifest_sha256": CAPABILITY_SHA256, "api_graph_sha256": graph_sha}

    async def submit(self, *, image_producer_job_id: str, request_snapshot: Mapping[str, Any], client_request_id: str,
                     request_snapshot_sha256: str | None = None) -> dict[str, Any]:
        if not is_upscale_snapshot(request_snapshot):
            return {"kind": "malformed"}
        try:
            expected = rebuild_snapshot(request_snapshot)
        except (UpscaleError, KeyError, TypeError):
            return {"kind": "malformed"}
        if _canonical(expected) != _canonical(dict(request_snapshot)):
            return {"kind": "malformed"}
        extra = {"studio": self._identity(image_producer_job_id, client_request_id, request_snapshot_sha256, expected["api_graph_sha256"])}
        receipt = await self.client.submit(expected["effective_input"]["graph"], client_id=client_request_id, extra_data=extra)
        pid = receipt.get("prompt_id") if isinstance(receipt, Mapping) else None
        if not isinstance(pid, str) or not pid or len(pid) > 256 or receipt.get("node_errors"):
            return {"kind": "malformed"}
        return {"kind": "accepted", "provider_prompt_id": pid}

    async def recover_submit(self, client_request_id: str, **kwargs: Any) -> dict[str, Any]:
        # An unknown submit stays unresolved (the coordinator quarantines it) rather than guessing a prompt.
        return {"kind": "unresolved"}

    recover = recover_submit

    async def poll(self, provider_prompt_id: str, *, expected_identity: Mapping[str, Any]) -> dict[str, Any]:
        history = await self.client.history(provider_prompt_id)
        entry = history.get(provider_prompt_id) if isinstance(history, Mapping) else None
        if entry is not None:
            prompt = entry.get("prompt")
            extra = prompt[3] if isinstance(prompt, (list, tuple)) and len(prompt) >= 4 else None
            if not isinstance(extra, Mapping) or extra.get("studio") != dict(expected_identity):
                return {"kind": "malformed"}
            status = entry.get("status") or {}
            if status.get("status_str") in {"error", "failed"}:
                return {"kind": "failed", "reason": _comfy_failure_reason(entry)}
            if status.get("status_str") == "success" or status.get("completed") is True:
                images = (entry.get("outputs", {}).get(OUTPUT_NODE) or {}).get("images")
                if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], Mapping) \
                        or images[0].get("type") != "temp" or images[0].get("subfolder") != "":
                    return {"kind": "malformed"}
                return {"kind": "ready", "output_identity": {"filename": images[0].get("filename"), "subfolder": "",
                                                             "type": "temp", "mime_type": "image/png", "extension": ".png"}}
        queue = await self.client.queue()
        for key, kind in (("queue_running", "running"), ("queue_pending", "queued")):
            for item in queue.get(key, []) if isinstance(queue, Mapping) else []:
                if isinstance(item, (list, tuple)) and len(item) >= 2 and item[1] == provider_prompt_id:
                    return {"kind": kind}
        return {"kind": "missing"}

    async def cancel(self, provider_prompt_id: str) -> dict[str, Any]:
        return {"kind": "unsupported"}


__all__ = ["CAPABILITY_SHA256", "CLASSES", "COLOR_CORRECTIONS", "DEFAULTS", "FACTORS", "GRAPH_LONG_EDGE_SHA256", "GRAPH_SHA256",
           "MODEL_FILE", "MODEL_LABEL", "OUTPUT_MAX_SIDE", "SCHEMA", "TARGET_ID", "TILE_OVERLAPS", "TILE_SIZES", "UpscaleAdapter",
           "UpscaleError", "VAE_FILE", "build_snapshot", "graph", "is_upscale_snapshot", "label", "normalize_request",
           "output_size", "rebuild_snapshot", "recipe", "static_health", "upload_name"]
