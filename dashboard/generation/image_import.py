"""A picture from disk (or the clipboard) becomes a picture of the ledger (``operation: "import"``).

Edit, Keep, compose, Use in Shoot and the Media feed all take a published
producer job, so an imported picture goes through the same ledger instead of a
second material authority.  No engine is involved: the route decodes the file
once, normalises it to the ledger's PNG (8-bit RGB or RGBA, EXIF orientation
applied, an embedded colour profile converted to sRGB), and the existing
ingestor publishes it content-addressed under its sha256 (write-once).  The
original file is never referenced again.

The snapshot is honest about what the job is: an ``imported`` provenance with
the file name and both fingerprints (source bytes and published PNG), what the
decoder did (orientation, bit depth, colour profile, alpha) and no render spec.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import re
from collections.abc import Mapping
from typing import Any

SCHEMA = "studio.image-import/1"
TARGET_ID = "filmsuite-import-v1"
FORMATS = ("PNG", "JPEG", "WEBP", "TIFF")
MAX_DIMENSION = 16_384
MAX_PIXELS = 67_108_864
MAX_PNG_BYTES = 33_554_432
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_KEYS = {"schema", "file_name", "source_sha256", "picture_sha256", "source_format", "bit_depth",
         "orientation", "icc", "alpha"}
_MANIFEST = {"schema": SCHEMA, "target_id": TARGET_ID, "classes": [], "decoder": "pillow-normalise-png8-srgb/1"}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


CAPABILITY_SHA256 = hashlib.sha256(_canonical(_MANIFEST)).hexdigest()
GRAPH_SHA256 = hashlib.sha256(_canonical({"import": "no-graph"})).hexdigest()


class ImageImportError(ValueError):
    """A plain refusal: ``code`` for machines, ``message`` for the artist."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


NOT_A_PICTURE = "The file is not a picture Continuo can open (PNG, JPEG, WebP or TIFF)"


MAX_SOURCE_BYTES = 64 * 1024 * 1024
MAX_SOURCE_B64 = (MAX_SOURCE_BYTES + 2) // 3 * 4
TOO_LARGE_FILE = "The file is too large to import (over 64 MB)"
OUTSIDE_ROOTS = "The file is outside the folders Continuo may read"
LOCKS: dict[str, Any] = {}


def read_local(path: str, approved_roots: list[str], confine: Any) -> tuple[bytes, str]:
    """Read one approved local file (confined by ``confine``, bounded); never referenced again."""
    try:
        source = confine(path, approved_roots)
    except Exception as exc:  # noqa: BLE001 - the confinement's refusal, reworded plainly
        if "does not exist" in str(exc):
            raise ImageImportError("file_missing", "The file does not exist") from exc
        raise ImageImportError("path_refused", OUTSIDE_ROOTS) from exc
    if source.stat().st_size > MAX_SOURCE_BYTES:
        raise ImageImportError("picture_too_large", TOO_LARGE_FILE)
    with source.open("rb") as handle:
        data = handle.read(MAX_SOURCE_BYTES + 1)
    if len(data) > MAX_SOURCE_BYTES:
        raise ImageImportError("picture_too_large", TOO_LARGE_FILE)
    return data, source.name


def idempotency_key(owner_scope: str, source_sha256: str) -> str:
    """Same bytes in the same scope are the same job."""
    return f"import:{owner_scope}:{source_sha256}"


def descriptor(picture_sha256: str) -> dict[str, str]:
    """The publication descriptor the ingestor validates (a PNG, never a caller's path)."""
    return {"filename": f"filmsuite-import-{picture_sha256[:32]}.png", "subfolder": "", "type": "temp",
            "mime_type": "image/png", "extension": ".png"}


def _to_8bit(arr: Any) -> Any:
    import numpy as np  # type: ignore[import-not-found]
    return np.clip(np.rint(arr.astype("float64") / 257.0), 0, 255).astype("uint8")


def source_bit_depth(image: Any) -> int:
    """Bits per channel of the file (read before decoding: Pillow reduces 16-bit RGB to 8 on load)."""
    if image.mode.startswith("I;16"):
        return 16
    if image.mode == "1":
        return 1
    tag = getattr(image, "tag_v2", None)
    if image.format == "TIFF" and tag is not None:
        bits = tag.get(258)
        bits = bits[0] if isinstance(bits, tuple) and bits else bits
        return 16 if bits == 16 else 8
    tile_args = " ".join(str(getattr(t, "args", "") or "") for t in (getattr(image, "tile", None) or ()))
    return 16 if ";16" in tile_args else 8


def _normalise_mode(image: Any, depth: int) -> tuple[Any, int]:
    """Bring every accepted mode to L/LA/RGB/RGBA/CMYK at 8 bit; returns (image, source bit depth)."""
    from PIL import Image  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]
    mode = image.mode
    if mode in {"RGB", "RGBA", "L", "LA", "CMYK"}:
        return image, depth
    if mode == "1":
        return image.convert("L"), 1
    if mode == "P":
        has_alpha = "transparency" in image.info
        return image.convert("RGBA" if has_alpha else "RGB"), 8
    if mode == "PA":
        return image.convert("RGBA"), 8
    if mode == "YCbCr":
        return image.convert("RGB"), 8
    if mode.startswith("I;16") or mode == "I":
        arr = np.asarray(image)
        if arr.dtype.kind not in "iu" or arr.size == 0 or int(arr.min()) < 0 or int(arr.max()) > 65535:
            raise ImageImportError("unsupported_picture", "This picture's pixel format is not supported")
        return Image.fromarray(_to_8bit(arr), "L"), 16
    raise ImageImportError("unsupported_picture", "This picture's pixel format is not supported")


def _to_srgb(image: Any, icc: bytes | None) -> tuple[Any, dict[str, Any] | None]:
    """Convert an embedded colour profile to sRGB (relative colorimetric + black point compensation)."""
    if not icc:
        if image.mode == "CMYK":
            return image.convert("RGB"), None
        return image, None
    from PIL import ImageCms  # type: ignore[import-not-found]
    try:
        source = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        description = (ImageCms.getProfileDescription(source) or "").strip()[:120] or "Embedded profile"
    except Exception as exc:  # noqa: BLE001 - a broken profile is a plain refusal
        raise ImageImportError("unsupported_profile", "The picture's colour profile cannot be read") from exc
    alpha = None
    base = image
    if image.mode in {"RGBA", "LA"}:
        alpha = image.getchannel("A")
        base = image.convert("RGB" if image.mode == "RGBA" else "L")
    try:
        converted = ImageCms.profileToProfile(
            base, source, ImageCms.createProfile("sRGB"),
            renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC, outputMode="RGB",
            flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
        )
    except Exception as exc:  # noqa: BLE001 - profile and pixels do not fit together
        raise ImageImportError("unsupported_profile", "The picture's colour profile cannot be applied") from exc
    if alpha is not None:
        converted = converted.convert("RGBA")
        converted.putalpha(alpha)
    return converted, {"source": description, "converted_to": "sRGB",
                       "source_sha256": hashlib.sha256(icc).hexdigest()}


def decode(data: bytes) -> tuple[bytes, dict[str, Any]]:
    """Decode once, refuse plainly, normalise to the ledger PNG.  Returns (png, facts)."""
    from PIL import Image, ImageOps, UnidentifiedImageError  # type: ignore[import-not-found]
    if type(data) is not bytes or not data:
        raise ImageImportError("not_a_picture", NOT_A_PICTURE)
    try:
        image = Image.open(io.BytesIO(data))
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ImageImportError("not_a_picture", NOT_A_PICTURE) from exc
    except Image.DecompressionBombError as exc:  # pragma: no cover - size is checked below first
        raise ImageImportError("picture_too_large", "The picture has too many pixels to import") from exc
    with image:
        if image.format not in FORMATS:
            raise ImageImportError("not_a_picture", NOT_A_PICTURE)
        source_format = image.format
        width, height = image.size
        if width > MAX_DIMENSION or height > MAX_DIMENSION or width * height > MAX_PIXELS:
            raise ImageImportError("picture_too_large", "The picture has too many pixels to import")
        if width < 1 or height < 1:
            raise ImageImportError("not_a_picture", NOT_A_PICTURE)
        depth = source_bit_depth(image)
        try:
            image.load()
        except Exception as exc:  # noqa: BLE001 - truncated or hostile file
            raise ImageImportError("not_a_picture", "The picture file is damaged") from exc
        icc = image.info.get("icc_profile") or None
        exif = image.getexif()
        orientation = exif.get(0x0112) if exif is not None else None
        orientation = orientation if type(orientation) is int and 1 <= orientation <= 8 else 1
        work, depth = _normalise_mode(image, depth)
        work = ImageOps.exif_transpose(work) if orientation != 1 else work.copy()
    work, profile = _to_srgb(work, icc)
    alpha = False
    if work.mode in {"RGBA", "LA"}:
        alpha = work.getchannel("A").getextrema()[0] < 255
    work = work.convert("RGBA" if alpha else "RGB")
    out = io.BytesIO()
    work.save(out, format="PNG", compress_level=6)
    png = out.getvalue()
    if len(png) > MAX_PNG_BYTES:
        raise ImageImportError("picture_too_large", "The picture is too large to import (over 32 MB as PNG)")
    facts = {"width": work.width, "height": work.height, "source_format": source_format, "bit_depth": depth,
             "orientation": orientation, "icc": profile, "alpha": alpha,
             "picture_sha256": hashlib.sha256(png).hexdigest(), "source_sha256": hashlib.sha256(data).hexdigest()}
    return png, facts


def clean_file_name(name: Any) -> str:
    """The file name only (never a directory), printable, at most 120 characters."""
    text = str(name or "").replace("\\", "/").rsplit("/", 1)[-1]
    text = "".join(ch for ch in text if ch.isprintable()).strip()
    return text[:120] or "Pasted picture"


def build_snapshot(provenance: Mapping[str, Any], width: int, height: int, basis: dict[str, Any],
                   policy: dict[str, Any]) -> dict[str, Any]:
    """The ledger request snapshot of one imported picture (validated, canonical)."""
    e = dict(provenance)
    if set(e) != _KEYS or e["schema"] != SCHEMA:
        raise ImageImportError("invalid_import", "import-shape")
    if not isinstance(e["file_name"], str) or e["file_name"] != clean_file_name(e["file_name"]):
        raise ImageImportError("invalid_import", "import-file-name")
    for key in ("source_sha256", "picture_sha256"):
        if not isinstance(e[key], str) or not _HEX64.match(e[key]):
            raise ImageImportError("invalid_import", f"import-{key}")
    if e["source_format"] not in FORMATS or e["bit_depth"] not in (1, 8, 16):
        raise ImageImportError("invalid_import", "import-format")
    if isinstance(e["orientation"], bool) or e["orientation"] not in range(1, 9) or type(e["alpha"]) is not bool:
        raise ImageImportError("invalid_import", "import-flags")
    icc = e["icc"]
    if icc is not None and (not isinstance(icc, dict) or set(icc) != {"source", "converted_to", "source_sha256"}
                            or icc["converted_to"] != "sRGB" or not isinstance(icc["source"], str)
                            or not 1 <= len(icc["source"]) <= 120 or not _HEX64.match(str(icc["source_sha256"]))):
        raise ImageImportError("invalid_import", "import-icc")
    for value in (width, height):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_DIMENSION:
            raise ImageImportError("invalid_import", "import-size")
    return {
        "operation": "import",
        "target_id": TARGET_ID,
        # No seed and no model settings: an imported picture has no render spec.
        "artist_input": {"prompt": e["file_name"], "width": width, "height": height},
        "effective_input": {"imported": copy.deepcopy(e)},
        "context_basis": copy.deepcopy(basis),
        "attach_policy": copy.deepcopy(policy),
        "capability_manifest_sha256": CAPABILITY_SHA256,
        "api_graph_sha256": GRAPH_SHA256,
    }


def is_import_snapshot(snapshot: Any) -> bool:
    return isinstance(snapshot, Mapping) and snapshot.get("operation") == "import" and snapshot.get("target_id") == TARGET_ID


def rebuild_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild from the snapshot's own provenance (the ledger compares canonical bytes)."""
    effective, artist = snapshot.get("effective_input"), snapshot.get("artist_input")
    if not isinstance(effective, Mapping) or set(effective) != {"imported"} or not isinstance(artist, Mapping):
        raise ImageImportError("invalid_import", "import-snapshot-shape")
    return build_snapshot(effective["imported"], artist.get("width"), artist.get("height"),
                          snapshot["context_basis"], snapshot["attach_policy"])


def provider_prompt_id(image_producer_job_id: str) -> str:
    """The ledger's provider correlation for a job no engine ran (unique per job)."""
    return f"import:{image_producer_job_id}"


def publish(jobs: Any, ingestor: Any, job: Mapping[str, Any], png: bytes) -> dict[str, Any]:
    """Drive one import job to published through the ordinary ledger states (resumable).

    accepted -> submitting -> queued (bound to its own correlation) -> running -> ready ->
    ingested -> published, then the ingestor writes the PNG once under its sha256.
    """
    job_id = job["image_producer_job_id"]
    current = jobs.get_job(job_id)
    if current["state"] == "accepted":
        current = jobs.mark_submitting(job_id)
    if current["state"] == "submitting":
        current = jobs.bind_provider_prompt(job_id, provider_prompt_id(job_id))
    if current["state"] == "queued":
        current = jobs.transition(job_id, "running")
    if current["state"] == "running":
        current = jobs.transition(job_id, "ready")
    if current["state"] not in {"ready", "ingested", "published"}:
        raise ImageImportError("import_failed", "The picture could not be imported")
    sha = hashlib.sha256(png).hexdigest()
    evidence = {
        "provider_id": current["provider"]["provider_id"], "provider_prompt_id": current["provider"]["prompt_id"],
        "target_id": TARGET_ID, "capability_manifest_sha256": current["capability_manifest_sha256"],
        "api_graph_sha256": current["api_graph_sha256"], "request_snapshot_sha256": current["request_snapshot_sha256"],
        "context_basis": dict(current["context_basis"]), "result_sha256": sha,
    }
    return ingestor.ingest_and_publish(job_id, stream=[png], descriptor_count=1,
                                       descriptor=descriptor(current["request_snapshot"]["effective_input"]["imported"]["picture_sha256"]),
                                       result_evidence=evidence)


__all__ = ["CAPABILITY_SHA256", "FORMATS", "GRAPH_SHA256", "ImageImportError", "NOT_A_PICTURE", "SCHEMA", "TARGET_ID",
           "build_snapshot", "clean_file_name", "decode", "descriptor", "idempotency_key", "is_import_snapshot",
           "provider_prompt_id", "publish", "rebuild_snapshot"]
