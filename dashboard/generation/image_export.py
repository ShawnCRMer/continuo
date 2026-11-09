"""Picture export with the Suite's recipe in the file, and reading it back (lane w1-imgout).

Formats: PNG 8/16 bit, JPEG (quality), WebP (quality or lossless), TIFF 8/16 bit.  Size:
the original, a long edge or a percent (Lanczos; 16-bit output is resampled in float, so the
extra precision is real after a resize and an exact x257 promotion without one).  Colour: the
ledger pictures are sRGB (imports are converted to sRGB), so the sRGB profile Pillow's
LittleCMS builds is embedded as is; Display P3 would need a P3 profile and a conversion
Pillow cannot build (ImageCms.createProfile knows sRGB, LAB and XYZ only), so it is refused.

The recipe (``studio.image-recipe/1``: model, prompt, seed, settings, parent sha256) rides
in the file when metadata is on: PNG iTXt ``studio.image-recipe``, JPEG/WebP/TIFF XMP
(``filmsuite:recipe`` in the Film Suite namespace).  ``read_recipe`` finds it in any of
them, so a dropped file can answer the settings it was made with.

PNG and TIFF are written by this module (Pillow cannot write 48/64-bit RGB(A)); JPEG and
WebP by Pillow.  Pure: bytes in, bytes out; no paths.
"""
from __future__ import annotations

import html
import io
import json
import re
import struct
import zlib
from typing import Any, Mapping

RECIPE_SCHEMA = "studio.image-recipe/1"
REQUEST_SCHEMA = "studio.image-export-request/1"
PNG_KEY = "studio.image-recipe"
XMP_NS = "https://hermes-agent.nousresearch.com/ns/film-suite/1.0/"
FORMATS = {"png": {"ext": "png", "bit_depths": (8, 16), "quality": False},
           "jpeg": {"ext": "jpg", "bit_depths": (8,), "quality": True},
           "webp": {"ext": "webp", "bit_depths": (8,), "quality": True},
           "tiff": {"ext": "tif", "bit_depths": (8, 16), "quality": False}}
DEFAULT_BIT_DEPTH = {"png": 8, "jpeg": 8, "webp": 8, "tiff": 16}
COLOR_PROFILES = ("srgb", "display_p3")
P3_REASON = "Display P3 needs a P3 profile and a colour conversion; Pillow's colour engine here builds sRGB only"
SIZE_MODES = ("original", "long_edge", "percent")
MAX_SIDE = 16_384
MAX_PIXELS = 16_384 * 8_192
RECIPE_MAX_BYTES = 60_000  # a JPEG APP1 XMP segment holds at most 65 502 bytes
_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_XMP_RE = re.compile(r"<filmsuite:recipe>(.*?)</filmsuite:recipe>", re.S)


class ExportError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _imports():
    import numpy as np  # type: ignore[import-not-found]
    from PIL import Image, ImageCms  # type: ignore[import-not-found]
    return np, Image, ImageCms


_SRGB: dict[str, bytes] = {}


def srgb_profile() -> bytes:
    if "icc" not in _SRGB:
        _np, _Image, ImageCms = _imports()
        _SRGB["icc"] = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    return _SRGB["icc"]


def icc_description(icc: bytes | None) -> str | None:
    if not icc:
        return None
    _np, _Image, ImageCms = _imports()
    try:
        return ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(icc))).strip() or "embedded"
    except Exception:  # noqa: BLE001 - unreadable profile
        return "unreadable"


# --- request ----------------------------------------------------------------------------------
def normalize(raw: Mapping[str, Any], source_size: tuple[int, int] | list[int]) -> dict[str, Any]:
    """The caller's export choice -> canonical settings (finite refusals)."""
    fmt = raw.get("format", "png")
    if fmt not in FORMATS:
        raise ExportError("invalid_export", "The format is png, jpeg, webp or tiff")
    spec = FORMATS[fmt]
    depth = raw.get("bit_depth", DEFAULT_BIT_DEPTH[fmt])
    if isinstance(depth, bool) or depth not in spec["bit_depths"]:
        raise ExportError("invalid_export", f"{fmt.upper()} is written at {' or '.join(map(str, spec['bit_depths']))} bit")
    quality = raw.get("quality")
    lossless = raw.get("lossless", False)
    if not isinstance(lossless, bool) or (lossless and fmt != "webp"):
        raise ExportError("invalid_export", "Lossless is a WebP choice")
    if spec["quality"]:
        quality = 92 if quality is None else quality
        if isinstance(quality, bool) or not isinstance(quality, int) or not 1 <= quality <= 100:
            raise ExportError("invalid_export", "Quality is 1 to 100")
    elif quality is not None:
        raise ExportError("invalid_export", f"{fmt.upper()} has no quality setting")
    profile = raw.get("color_profile", "srgb")
    if profile not in COLOR_PROFILES:
        raise ExportError("invalid_export", "The colour profile is srgb or display_p3")
    if profile == "display_p3":
        raise ExportError("profile_unavailable", P3_REASON)
    metadata = raw.get("metadata", True)
    if not isinstance(metadata, bool):
        raise ExportError("invalid_export", "Metadata is on or off")
    size = raw.get("size") or {"mode": "original"}
    if not isinstance(size, Mapping) or size.get("mode") not in SIZE_MODES:
        raise ExportError("invalid_export", "The size is original, long_edge or percent")
    w, h = int(source_size[0]), int(source_size[1])
    mode, value = size["mode"], size.get("value")
    if mode == "original":
        if value is not None:
            raise ExportError("invalid_export", "The original size takes no value")
        tw, th = w, h
    elif mode == "long_edge":
        if isinstance(value, bool) or not isinstance(value, int) or not 16 <= value <= MAX_SIDE:
            raise ExportError("invalid_export", f"The long edge is 16 to {MAX_SIDE} pixels")
        scale = value / max(w, h)
        tw, th = (value, max(1, round(h * scale))) if w >= h else (max(1, round(w * scale)), value)
    else:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 1 <= float(value) <= 400:
            raise ExportError("invalid_export", "The percent is 1 to 400")
        tw, th = max(1, round(w * float(value) / 100.0)), max(1, round(h * float(value) / 100.0))
    if tw > MAX_SIDE or th > MAX_SIDE or tw * th > MAX_PIXELS:
        raise ExportError("export_too_large", f"The exported picture would be {tw} x {th}")
    return {"format": fmt, "bit_depth": depth, "quality": quality if spec["quality"] and not lossless else None,
            "lossless": lossless, "color_profile": profile, "metadata": metadata,
            "size": {"mode": mode, "value": value}, "width": tw, "height": th}


# --- pixels -----------------------------------------------------------------------------------
def _pixels(source_png: bytes, width: int, height: int, depth: int) -> tuple[Any, bool]:
    """(H x W x C array, alpha?) at ``depth`` (uint8 or uint16), Lanczos-resampled when the size changes."""
    np, Image, _ = _imports()
    with Image.open(io.BytesIO(source_png)) as image:
        image.load()
        alpha = image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info)
        work = image.convert("RGBA" if alpha else "RGB")
    same = work.size == (width, height)
    if depth == 8:
        if not same:
            work = work.resize((width, height), Image.Resampling.LANCZOS)
        return np.asarray(work, dtype=np.uint8).copy(), alpha
    if same:
        return np.asarray(work, dtype=np.uint16) * 257, alpha
    planes = []
    for band in work.split():  # float per channel: no 8-bit rounding before the 16-bit result
        f = band.convert("F").resize((width, height), Image.Resampling.LANCZOS)
        planes.append(np.asarray(f, dtype=np.float32))
    arr = np.stack(planes, axis=-1) * 257.0
    return np.clip(np.rint(arr), 0, 65535).astype(np.uint16), alpha


# --- writers ----------------------------------------------------------------------------------
def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _png(arr: Any, alpha: bool, icc: bytes | None, recipe: str | None) -> bytes:
    np, _, _ = _imports()
    height, width, channels = arr.shape
    depth = 16 if arr.dtype == np.uint16 else 8
    rows = arr.astype(">u2" if depth == 16 else np.uint8).reshape(height, -1).view(np.uint8)
    raw = np.concatenate([np.zeros((height, 1), dtype=np.uint8), rows], axis=1).tobytes()
    out = [_PNG_SIG, _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, depth, 6 if alpha else 2, 0, 0, 0))]
    if icc:
        out.append(_chunk(b"iCCP", b"sRGB\x00\x00" + zlib.compress(icc, 9)))
    if recipe is not None:  # iTXt, compressed: keyword, flag 1, method 0, empty language and translated keyword
        out.append(_chunk(b"iTXt", PNG_KEY.encode("latin-1") + b"\x00\x01\x00\x00\x00" + zlib.compress(recipe.encode("utf-8"), 9)))
    out.append(_chunk(b"IDAT", zlib.compress(raw, 6)))
    out.append(_chunk(b"IEND", b""))
    return b"".join(out)


def xmp_packet(recipe: str) -> bytes:
    body = html.escape(recipe, quote=False)
    return ('<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>'
            '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            f'<rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:filmsuite="{XMP_NS}" '
            f'xmp:CreatorTool="Continuo"><filmsuite:recipe>{body}</filmsuite:recipe></rdf:Description></rdf:RDF>'
            '</x:xmpmeta><?xpacket end="w"?>').encode("utf-8")


def _tiff(arr: Any, alpha: bool, icc: bytes | None, xmp: bytes | None) -> bytes:
    """Baseline little-endian TIFF, one Deflate strip, RGB(A) at 8 or 16 bit."""
    np, _, _ = _imports()
    height, width, channels = arr.shape
    depth = 16 if arr.dtype == np.uint16 else 8
    strip = zlib.compress(arr.astype("<u2" if depth == 16 else np.uint8).tobytes(), 6)
    software = b"Continuo\x00"
    entries: list[tuple[int, int, int, Any]] = []  # tag, type, count, value (int or bytes)
    SHORT, LONG, RATIONAL, ASCII, BYTE, UNDEFINED = 3, 4, 5, 2, 1, 7
    bits = struct.pack("<" + "H" * channels, *([depth] * channels))
    entries += [(256, LONG, 1, width), (257, LONG, 1, height), (258, SHORT, channels, bits), (259, SHORT, 1, 8),
                (262, SHORT, 1, 2), (273, LONG, 1, None), (274, SHORT, 1, 1), (277, SHORT, 1, channels),
                (278, LONG, 1, height), (279, LONG, 1, len(strip)), (282, RATIONAL, 1, struct.pack("<II", 72, 1)),
                (283, RATIONAL, 1, struct.pack("<II", 72, 1)), (284, SHORT, 1, 1), (296, SHORT, 1, 2),
                (305, ASCII, len(software), software)]
    if alpha:
        entries.append((338, SHORT, 1, 2))  # unassociated alpha
    entries.append((339, SHORT, channels, struct.pack("<" + "H" * channels, *([1] * channels))))
    if xmp:
        entries.append((700, BYTE, len(xmp), xmp))
    if icc:
        entries.append((34675, UNDEFINED, len(icc), icc))
    entries.sort(key=lambda e: e[0])
    ifd_size = 2 + 12 * len(entries) + 4
    data_at = 8 + ifd_size
    blobs = b""
    packed = []
    for tag, typ, count, value in entries:
        if isinstance(value, bytes) and len(value) > 4:
            offset = data_at + len(blobs)
            blobs += value + (b"\x00" if len(value) % 2 else b"")
            packed.append((tag, typ, count, struct.pack("<I", offset)))
        elif isinstance(value, bytes):
            packed.append((tag, typ, count, value.ljust(4, b"\x00")))
        else:
            packed.append((tag, typ, count, value))
    strip_at = data_at + len(blobs)
    ifd = struct.pack("<H", len(packed))
    for tag, typ, count, value in packed:
        if value is None:
            value = strip_at
        field = value if isinstance(value, bytes) else (struct.pack("<HH", value, 0) if typ == SHORT else struct.pack("<I", value))
        ifd += struct.pack("<HHI", tag, typ, count) + field
    ifd += struct.pack("<I", 0)
    return b"II*\x00" + struct.pack("<I", 8) + ifd + blobs + strip


def encode(source_png: bytes, settings: Mapping[str, Any], recipe: Mapping[str, Any] | None) -> dict[str, Any]:
    """Write one export; returns the file bytes and its facts."""
    np, Image, _ = _imports()
    fmt, depth = settings["format"], settings["bit_depth"]
    text = None
    if settings["metadata"] and recipe is not None:
        text = json.dumps(dict(recipe), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        if len(text.encode("utf-8")) > RECIPE_MAX_BYTES:
            raise ExportError("recipe_too_large", "The recipe is too large to embed; export with metadata off")
    icc = srgb_profile()
    arr, alpha = _pixels(source_png, settings["width"], settings["height"], depth)
    warnings: list[str] = []
    if fmt == "png":
        data = _png(arr, alpha, icc, text)
    elif fmt == "tiff":
        data = _tiff(arr, alpha, icc, xmp_packet(text) if text is not None else None)
    else:
        image = Image.fromarray(arr, mode="RGBA" if alpha else "RGB")
        if fmt == "jpeg" and alpha:
            flat = Image.new("RGB", image.size, (255, 255, 255))
            flat.paste(image, mask=image.getchannel("A"))
            image = flat
            warnings.append("alpha_flattened: JPEG has no transparency; flattened on white")
        out = io.BytesIO()
        options: dict[str, Any] = {"icc_profile": icc}
        if text is not None:
            options["xmp"] = xmp_packet(text)
        if fmt == "jpeg":
            image.save(out, format="JPEG", quality=settings["quality"], subsampling=0 if settings["quality"] >= 90 else 2, **options)
        else:
            image.save(out, format="WEBP", quality=100 if settings["lossless"] else settings["quality"],
                       lossless=settings["lossless"], method=4, exif=b"", **options)
        data = out.getvalue()
    return {"data": data, "width": settings["width"], "height": settings["height"], "alpha": alpha and fmt != "jpeg",
            "bit_depth": depth, "icc": icc_description(icc), "metadata": text is not None, "warnings": warnings}


# --- reading back -----------------------------------------------------------------------------
def _png_texts(data: bytes) -> dict[str, str]:
    texts: dict[str, str] = {}
    pos = len(_PNG_SIG)
    while pos + 12 <= len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        kind, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + length]
        pos += 12 + length
        try:
            if kind == b"tEXt":
                key, _, value = body.partition(b"\x00")
                texts[key.decode("latin-1")] = value.decode("latin-1")
            elif kind == b"zTXt":
                key, _, rest = body.partition(b"\x00")
                texts[key.decode("latin-1")] = zlib.decompressobj().decompress(rest[1:], RECIPE_MAX_BYTES * 4).decode("latin-1")
            elif kind == b"iTXt":
                key, _, rest = body.partition(b"\x00")
                flag, rest = rest[0], rest[2:]
                _lang, _, rest = rest.partition(b"\x00")
                _tkey, _, value = rest.partition(b"\x00")
                if flag:
                    value = zlib.decompressobj().decompress(value, RECIPE_MAX_BYTES * 4)
                texts[key.decode("latin-1")] = value.decode("utf-8")
            elif kind == b"IDAT" and texts.get(PNG_KEY):
                break
        except (ValueError, IndexError, zlib.error, UnicodeDecodeError):
            continue
    return texts


def _from_xmp(packet: Any) -> str | None:
    if isinstance(packet, bytes):
        packet = packet.decode("utf-8", "replace")
    if not isinstance(packet, str):
        return None
    m = _XMP_RE.search(packet)
    return html.unescape(m.group(1)) if m else None


def read_recipe(data: bytes) -> dict[str, Any]:
    """The Suite recipe embedded in a picture file, or why there is none (never raises)."""
    np, Image, _ = _imports()
    found: str | None = None
    where = None
    fmt = None
    try:
        if data.startswith(_PNG_SIG):
            fmt = "png"
            found, where = _png_texts(data).get(PNG_KEY), "png-itxt"
        else:
            with Image.open(io.BytesIO(data)) as image:
                fmt = (image.format or "").lower() or None
                xmp = image.info.get("xmp")
                if xmp is None and fmt == "tiff":
                    xmp = getattr(image, "tag_v2", {}).get(700)
                found, where = _from_xmp(xmp), "xmp"
    except Exception:  # noqa: BLE001 - not a picture we can read
        return {"found": False, "format": fmt, "reason": "not_a_picture"}
    if found is None:
        return {"found": False, "format": fmt, "reason": "no_recipe"}
    try:
        recipe = json.loads(found)
    except ValueError:
        return {"found": False, "format": fmt, "reason": "recipe_unreadable"}
    if not isinstance(recipe, dict) or recipe.get("schema") != RECIPE_SCHEMA:
        return {"found": False, "format": fmt, "reason": "not_a_suite_recipe"}
    return {"found": True, "format": fmt, "where": where, "recipe": recipe}


def file_facts(data: bytes) -> dict[str, Any]:
    """Format, size, bit depth, ICC and the recipe of a written file (tests and the export answer)."""
    np, Image, _ = _imports()
    facts: dict[str, Any] = {"bytes": len(data)}
    if data.startswith(_PNG_SIG):
        w, h, depth, color = struct.unpack(">IIBB", data[16:26])
        facts.update(format="png", width=w, height=h, bit_depth=depth, alpha=color == 6)
        with Image.open(io.BytesIO(data)) as image:
            facts["icc"] = icc_description(image.info.get("icc_profile"))
            image.load()
    else:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            facts.update(format=(image.format or "").lower(), width=image.width, height=image.height,
                         icc=icc_description(image.info.get("icc_profile")), alpha="A" in image.getbands())
            bits = getattr(image, "tag_v2", {}).get(258) if facts["format"] == "tiff" else None
            facts["bit_depth"] = int(bits[0]) if bits else 8
            facts["has_exif"] = bool(image.info.get("exif"))
            facts["has_xmp"] = bool(image.info.get("xmp") or (getattr(image, "tag_v2", {}).get(700) if facts["format"] == "tiff" else None))
    read = read_recipe(data)
    facts["recipe"] = read.get("recipe")
    return facts


# --- the recipe of a ledger picture ----------------------------------------------------------
def recipe_for(job: Mapping[str, Any], *, parent_sha256: str | None = None, model_label: str | None = None) -> dict[str, Any]:
    """``studio.image-recipe/1`` of one published ledger picture (what made it, from its own snapshot)."""
    snap = job.get("request_snapshot") or {}
    artist = snap.get("artist_input") or {}
    effective = snap.get("effective_input") or {}
    op = snap.get("operation") or "create"
    artifact = job.get("artifact") or {}
    out: dict[str, Any] = {"schema": RECIPE_SCHEMA, "suite": "Continuo", "operation": op, "target_id": snap.get("target_id"),
                           "prompt": artist.get("prompt"), "seed": artist.get("seed"),
                           "width": artist.get("width"), "height": artist.get("height"),
                           "picture_sha256": artifact.get("artifact_sha256"), "parent_sha256": parent_sha256,
                           "model": model_label, "settings": {}}
    if op in ("workflow_create", "workflow_edit"):
        wf = effective.get("workflow") or {}
        out["settings"] = {"controls": wf.get("controls"), "references": [{"role": r.get("role"), "sha256": r.get("sha256")}
                                                                          for r in effective.get("owned_references") or []]}
        if job.get("sent_prompt"):
            out["settings"]["sent_prompt"] = job["sent_prompt"]
        out["apply"] = {"route": "/projects/image-producer/workflow-submit", "target_id": snap.get("target_id"),
                        "prompt": wf.get("prompt"), "controls": wf.get("controls")}
    elif op == "create":
        spec = effective.get("render_spec")
        files = (spec or {}).get("files") or {"diffusion_model": effective.get("model"), "text_encoder": effective.get("encoder"), "vae": effective.get("vae")}
        out["settings"] = {"render_spec": spec, "files": files, "steps": effective.get("steps"), "cfg": effective.get("cfg"),
                           "sampler": effective.get("sampler"), "scheduler": effective.get("scheduler")}
        out["model"] = model_label or files.get("diffusion_model")
        out["apply"] = {"route": "/projects/image-producer/submit", "prompt": artist.get("prompt"), "render_spec": spec}
    elif op == "edit":
        edit = effective.get("edit") or {}
        out["settings"] = {k: edit.get(k) for k in ("operation", "prompt", "target", "instruction_mode", "prompt_enhancer",
                                                    "color_match", "context_factor", "crop_box", "mask_sha256")}
        out["settings"].update(steps=effective.get("steps"), cfg=effective.get("cfg"), sampler=effective.get("sampler"))
        out["model"] = model_label or effective.get("model")
        out["parent_sha256"] = parent_sha256 or edit.get("source_sha256")
    elif op == "upscale":
        up = effective.get("upscale") or {}
        out["settings"] = {k: up.get(k) for k in ("mode", "factor", "long_edge", "tile_size", "tile_overlap", "color_correction")}
        out["model"] = model_label or up.get("model")
        out["parent_sha256"] = parent_sha256 or up.get("source_sha256")
    elif op == "compose":
        comp = effective.get("compose") or {}
        out["settings"] = {"kind": comp.get("kind"), "recipe": comp.get("recipe")}
    elif op == "import":
        imp = effective.get("imported") or {}
        out["settings"] = {k: imp.get(k) for k in ("file_name", "source_format", "bit_depth", "icc")}
        out["parent_sha256"] = parent_sha256 or imp.get("source_sha256")
        out["prompt"] = None
    return out


__all__ = ["COLOR_PROFILES", "ExportError", "FORMATS", "P3_REASON", "PNG_KEY", "RECIPE_SCHEMA", "REQUEST_SCHEMA", "SIZE_MODES",
           "XMP_NS", "encode", "file_facts", "normalize", "read_recipe", "recipe_for", "srgb_profile", "xmp_packet"]
