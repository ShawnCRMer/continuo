"""Relight geometry jobs: MoGe-2 surface normals and depth of a picture via ComfyUI.

Loaded unregistered by ``plugin_api`` (like ``image_mask_jobs``), so no bare
relative imports and no ``@dataclass``.  One job = one small ComfyUI graph of
native core nodes (comfy_extras.nodes_moge, no custom node pack):

  LoadImage -> LoadMoGeModel(moge_2_vitl_normal_fp16) -> MoGeInference ->
  MoGeRender(normal_opengl) -> PreviewImage
                            -> MoGeRender(depth) -> PreviewImage   (stage 1b: shadows)

The answer is an 8-bit RGB normal map at source size in the OpenGL convention
(x right, y up, z towards the viewer; channel = n * 0.5 + 0.5), which the
renderer's live-light shader reads; the depth is MoGeRender's normalised disparity (1 = nearest,
0.1/99.9 percentile clipped), one grey channel, over which the shader marches its shadows (the
same inference, so it costs one more render node).  Measured on a 1024 px portrait: about
8-14 s, once per picture, so the result is cached by the picture's sha256.
Jobs queue normally behind other work and are never cancelled or reordered
here; the mask job module's service does the ComfyUI round trip.
"""
from __future__ import annotations

import base64
import hashlib
import io
from typing import Any, Callable

SCHEMA = "studio.image-geometry-job/1"
CLIENT_ID = "filmsuite-geometry"
MOGE_FILE = "moge_2_vitl_normal_fp16.safetensors"
RESOLUTION_LEVEL = 9
REQUIRED_CLASSES = ("LoadImage", "LoadMoGeModel", "MoGeInference", "MoGeRender", "PreviewImage")


def build_graph(image_name: str) -> dict[str, Any]:
    """The API-format graph (byte-stable)."""
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "LoadMoGeModel", "inputs": {"model_name": MOGE_FILE}},
        "3": {"class_type": "MoGeInference", "inputs": {
            "moge_model": ["2", 0], "image": ["1", 0], "resolution_level": RESOLUTION_LEVEL, "fov_x_degrees": 0.0,
            "batch_size": 1, "force_projection": True, "apply_mask": False, "refine_steps": 0}},
        "4": {"class_type": "MoGeRender", "inputs": {"moge_geometry": ["3", 0], "output": "normal_opengl"}},
        "5": {"class_type": "PreviewImage", "inputs": {"images": ["4", 0]}},
        "6": {"class_type": "MoGeRender", "inputs": {"moge_geometry": ["3", 0], "output": "depth"}},
        "7": {"class_type": "PreviewImage", "inputs": {"images": ["6", 0]}},
    }


def cache_key(picture_sha256: str) -> str:
    return hashlib.sha256((picture_sha256 + "|geometry+depth|" + MOGE_FILE + f"|{RESOLUTION_LEVEL}").encode()).hexdigest()


def depth_from_png(data: bytes, width: int, height: int) -> bytes:
    """MoGeRender depth output -> one-channel (L) PNG at source size."""
    from PIL import Image  # type: ignore[import-not-found]

    with Image.open(io.BytesIO(data)) as image:
        image.load()
        depth = image.convert("L")
    if depth.size != (width, height):
        depth = depth.resize((width, height), Image.Resampling.BILINEAR)
    out = io.BytesIO()
    depth.save(out, format="PNG", optimize=False)
    return out.getvalue()


def normals_from_png(data: bytes, width: int, height: int) -> tuple[bytes, float]:
    """MoGeRender output -> RGB PNG at source size, and the mean z (a frontal-plane sanity value)."""
    from PIL import Image  # type: ignore[import-not-found]

    with Image.open(io.BytesIO(data)) as image:
        image.load()
        normal = image.convert("RGB")
    if normal.size != (width, height):
        normal = normal.resize((width, height), Image.Resampling.BILINEAR)
    blue = normal.getchannel("B").histogram()
    total = float(width * height)
    mean_z = round(sum(i * c for i, c in enumerate(blue)) / total / 255.0 * 2.0 - 1.0, 3)
    out = io.BytesIO()
    normal.save(out, format="PNG", optimize=False)
    return out.getvalue(), mean_z


def make_service(mask_jobs_module: Any, client_factory: Callable[[], Any], opener: Callable[..., Any]) -> Any:
    """A geometry service on the mask job machinery (same queueing, cache, failures, long-poll)."""
    base = mask_jobs_module.MaskJobService
    error = mask_jobs_module.MaskJobError

    class GeometryJobService(base):  # type: ignore[misc, valid-type]
        async def check_nodes(self, client: Any, mode: str = "geometry") -> None:
            for name in REQUIRED_CLASSES:
                if await self._class_info(client, name) is None:
                    raise error("geometry_node_missing", f"ComfyUI has no {name} node")
            info = await self._class_info(client, "LoadMoGeModel") or {}
            spec = ((info.get("input") or {}).get("required") or {}).get("model_name")
            options: list[Any] = []
            if isinstance(spec, list) and spec:
                if isinstance(spec[0], list):
                    options = spec[0]
                elif len(spec) > 1 and isinstance(spec[1], dict):
                    options = list(spec[1].get("options") or [])
            if MOGE_FILE not in options:
                raise error("geometry_model_missing", f"ComfyUI has no {MOGE_FILE}")

        async def _run(self, key: str, entry: Any, picture: bytes, picture_sha256: str, size: tuple[int, int]) -> None:
            import asyncio

            started = self._clock()
            prompt_id = None
            try:
                client = self._client_factory()
                await self.check_nodes(client)
                if len(picture) > mask_jobs_module.UPLOAD_MAX_BYTES:
                    raise error("geometry_picture_too_large", "The picture is too large for the light tool")
                name = await asyncio.to_thread(self._upload, client.origin, f"filmsuite-geometry-{picture_sha256[:32]}.png", picture)
                submitted = await client.submit(build_graph(name), client_id=CLIENT_ID, extra_data={"studio": {"kind": "geometry_job"}})
                prompt_id = submitted.get("prompt_id") if isinstance(submitted, dict) else None
                if not isinstance(prompt_id, str) or not prompt_id:
                    raise error("geometry_job_rejected", "ComfyUI refused the geometry job")
                outputs = None
                while self._clock() - started < mask_jobs_module.JOB_TIMEOUT_SECONDS:
                    history = await client.history(prompt_id)
                    item = history.get(prompt_id) if isinstance(history, dict) else None
                    if isinstance(item, dict):
                        status = item.get("status") or {}
                        if status.get("status_str") == "error":
                            raise error("geometry_job_failed", "ComfyUI could not read the shapes of the picture")
                        if status.get("completed") or status.get("status_str") == "success":
                            outputs = item.get("outputs") or {}
                            break
                    await asyncio.sleep(mask_jobs_module.POLL_SECONDS)
                if outputs is None:
                    raise error("geometry_job_timeout", "The geometry job did not finish in time")
                images = (outputs.get("5") or {}).get("images") or []
                if not images or not isinstance(images[0], dict):
                    raise error("geometry_job_failed", "ComfyUI made no normal map")
                ref = {k: str(images[0][k]) for k in ("filename", "subfolder", "type") if k in images[0]}
                data = await client.view(ref)
                png, mean_z = await asyncio.to_thread(normals_from_png, data, size[0], size[1])
                depth_images = (outputs.get("7") or {}).get("images") or []
                if not depth_images or not isinstance(depth_images[0], dict):
                    raise error("geometry_job_failed", "ComfyUI made no depth map")
                depth_ref = {k: str(depth_images[0][k]) for k in ("filename", "subfolder", "type") if k in depth_images[0]}
                depth_png = await asyncio.to_thread(depth_from_png, await client.view(depth_ref), size[0], size[1])
                result: dict[str, Any] = {
                    "state": "done", "prompt_id": prompt_id,
                    "normals": {"png_b64": base64.b64encode(png).decode("ascii"), "width": size[0], "height": size[1],
                                "convention": "opengl", "mean_z": mean_z, "sha256": hashlib.sha256(png).hexdigest()},
                    "depth": {"png_b64": base64.b64encode(depth_png).decode("ascii"), "width": size[0], "height": size[1],
                              "convention": "disparity", "sha256": hashlib.sha256(depth_png).hexdigest()},
                    "model": {"class": "MoGeInference", "file": MOGE_FILE},
                    "ms": int((self._clock() - started) * 1000)}
            except error as exc:
                result = {"state": "failed", "prompt_id": prompt_id, "error": {"code": exc.code, "message": exc.message}}
            except Exception as exc:  # noqa: BLE001 - finite answer
                if getattr(exc, "reason_code", None) == "unreachable" and prompt_id is None:
                    result = {"state": "failed", "prompt_id": None, "error": {"code": "comfyui_unreachable", "message": "ComfyUI is not running."}}
                else:
                    result = {"state": "failed", "prompt_id": prompt_id, "error": {"code": "geometry_job_failed", "message": "The geometry job failed"}}
            entry.result = result

    return GeometryJobService(client_factory, opener)


__all__ = ["CLIENT_ID", "MOGE_FILE", "SCHEMA", "build_graph", "cache_key", "depth_from_png", "make_service", "normals_from_png"]
