"""LoRA library (Image WP7): local metadata and on-demand Civitai info.

Loaded unregistered (``spec_from_file_location``): plain classes only, no
dataclasses, no relative imports.

Local metadata is read without any network: sidecars next to the LoRA file
(``<stem>.civitai.info`` from Civitai Helper, ``<stem>.metadata.json`` from
LoRA Manager, ``<file>.rgthree-info.json`` from rgthree, ``<stem>.json`` from
the A1111 extra-networks editor), a sidecar preview picture and the
safetensors header ``__metadata__``.  Nothing is ever written next to the
user's files: hashes, Civitai answers and thumbnails live in a plugin-data
cache directory.

Civitai is asked only on demand (one ``fetch`` call per LoRA), by SHA256 at
``/api/v1/model-versions/by-hash/<sha256>`` (plus ``/api/v1/models/<id>`` when
the version carries no description), with bounded reads, an allowlist of hosts
and redirects followed only to Civitai's own https hosts (example pictures
answer 301 to ``blobs-b2.civitai.com``).  Tests inject fake fetchers.

Base-model names are normalized to one family label (``base_family``):
trainers write ``minimax_h3``, ``MiniMax-H3/lora``, ``krea2/lora`` ... for the
same family.
"""
import base64
import hashlib
import io
import json
import os
import re
import secrets
import struct
import threading
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable

LORA_EXT = ".safetensors"
NAME_MAX = 240
_BAD_CHARS = re.compile(r'[\\:*?"<>|\x00-\x1f]')
HEADER_MAX_BYTES = 8 * 1024 * 1024
SIDECAR_MAX_BYTES = 4 * 1024 * 1024
PREVIEW_MAX_BYTES = 16 * 1024 * 1024
THUMB_PX = 320
EXAMPLE_PX = 640
THUMB_MAX_BYTES = 512 * 1024
EXAMPLE_MAX = 4
TRIGGERS_MAX = 24
CIVITAI_API = "https://civitai.com/api/v1/model-versions/by-hash/"
CIVITAI_MODEL_API = "https://civitai.com/api/v1/models/"
CIVITAI_RECORD_V = 2  # v1 records lost every example picture (redirects refused)
CIVITAI_REDIRECTS_MAX = 3
CIVITAI_MODEL_URL = "https://civitai.com/models/"
CIVITAI_HOSTS = ("civitai.com", "image.civitai.com", "imagecache.civitai.com")
CIVITAI_TIMEOUT_S = 12.0
CIVITAI_JSON_MAX = 4 * 1024 * 1024
CIVITAI_IMAGE_MAX = 12 * 1024 * 1024
_HEX64 = re.compile(r"[0-9a-f]{64}")
_PREVIEW_SUFFIXES = (".preview.png", ".preview.jpg", ".preview.jpeg", ".preview.webp", ".png", ".jpg", ".jpeg", ".webp")


class LoraLibraryError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# Names and files
# ---------------------------------------------------------------------------
def normalize_name(value: Any) -> str:
    """ComfyUI lists ``krea2\\x.safetensors`` on Windows; specs keep ``/``."""
    return str(value).replace("\\", "/") if isinstance(value, str) else ""


def name_ok(name: Any) -> bool:
    """A relative ``.safetensors`` name below the LoRA folder: no drive, no traversal."""
    if type(name) is not str or not 0 < len(name) <= NAME_MAX or not name.endswith(LORA_EXT):
        return False
    if _BAD_CHARS.search(name) or name.startswith("/"):
        return False
    parts = name.split("/")
    return all(part and part not in (".", "..") and part == part.strip() for part in parts)


def lora_file(root: Path, name: str) -> Path | None:
    """The regular file ``root/name`` (inside ``root``), else None."""
    if not name_ok(name):
        return None
    path = Path(root) / Path(*name.split("/"))
    try:
        if path.is_symlink() or not path.is_file():
            return None
        if not path.resolve().is_relative_to(Path(root).resolve()):
            return None
    except OSError:
        return None
    return path


def _stat_key(path: Path) -> tuple[str, int, int]:
    st = path.stat()
    return (str(path), st.st_mtime_ns, st.st_size)


def _read_bounded(path: Path, limit: int) -> bytes | None:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
            return None
        with open(path, "rb") as handle:
            data = handle.read(limit + 1)
    except OSError:
        return None
    return data if len(data) <= limit else None


def _read_json(path: Path) -> Any:
    data = _read_bounded(path, SIDECAR_MAX_BYTES)
    if data is None:
        return None
    try:
        return json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError):
        return None


def image_mime(data: bytes) -> str | None:
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    return None


def shrink(data: bytes, px: int) -> bytes | None:
    """A small WebP/JPEG of a picture; None when it is not a picture."""
    if image_mime(data) is None:
        return None
    try:
        from PIL import Image  # noqa: PLC0415 - optional, present in the Hermes venv
    except Exception:  # noqa: BLE001 - without Pillow only small pictures pass through
        return data if len(data) <= THUMB_MAX_BYTES else None
    try:
        with Image.open(io.BytesIO(data)) as picture:
            picture.seek(0)
            picture = picture.convert("RGB")
            picture.thumbnail((px, px))
            out = io.BytesIO()
            picture.save(out, "WEBP", quality=82)
            small = out.getvalue()
    except Exception:  # noqa: BLE001 - a broken picture is no picture
        return None
    return small if len(small) <= THUMB_MAX_BYTES else None


def data_url(data: bytes | None) -> str | None:
    mime = image_mime(data) if data else None
    return None if mime is None else f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


# ---------------------------------------------------------------------------
# Local metadata
# ---------------------------------------------------------------------------
class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in ("p", "br", "li", "div", "h1", "h2", "h3"):
            self.parts.append("\n")


def plain_text(html: Any, limit: int = 2000) -> str:
    if not isinstance(html, str) or not html.strip():
        return ""
    parser = _Text()
    try:
        parser.feed(html)
    except Exception:  # noqa: BLE001 - keep what was read
        pass
    text = re.sub(r"[ \t]+", " ", "".join(parser.parts))
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return text[:limit]


def _words(values: Any) -> list[str]:
    out: list[str] = []
    items = values if isinstance(values, list) else [values] if isinstance(values, str) else []
    for item in items:
        if not isinstance(item, str):
            continue
        for word in (item.split(",") if len(items) == 1 else [item]):
            word = re.sub(r"\s+", " ", word).strip().strip(",").strip()
            if word and len(word) <= 200 and word not in out:
                out.append(word)
    return out[:TRIGGERS_MAX]


def safetensors_metadata(path: Path) -> dict[str, str]:
    """The ``__metadata__`` strings of a safetensors header (bounded read)."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(8)
            if len(head) != 8:
                return {}
            (length,) = struct.unpack("<Q", head)
            if not 2 <= length <= HEADER_MAX_BYTES:
                return {}
            raw = handle.read(length)
    except OSError:
        return {}
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return {}
    meta = header.get("__metadata__") if isinstance(header, dict) else None
    return {str(k): v for k, v in meta.items() if isinstance(v, str)} if isinstance(meta, dict) else {}


def _top_tags(meta: dict[str, str], count: int = 8) -> list[str]:
    try:
        freq = json.loads(meta.get("ss_tag_frequency") or "{}")
    except ValueError:
        return []
    totals: dict[str, int] = {}
    if isinstance(freq, dict):
        for bucket in freq.values():
            if isinstance(bucket, dict):
                for tag, n in bucket.items():
                    if isinstance(tag, str) and isinstance(n, int):
                        tag = tag.strip()
                        if tag:
                            totals[tag] = totals.get(tag, 0) + n
    return [tag for tag, _n in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))[:count]]


# Family label per pattern, first match wins (labels as in the model catalog).
_BASE_FAMILIES = (
    (re.compile(r"minimax|mm-?h3|(?:^|[^a-z0-9])h3(?:[^a-z0-9]|$)|ref2va|fl2va", re.I), "MiniMax H3"),
    (re.compile(r"krea", re.I), "Krea 2"),
    (re.compile(r"anima", re.I), "Anima"),
    (re.compile(r"qwen.{0,12}(?<![0-9])2(?:\.\d)?(?![0-9])", re.I), "Qwen-Image 2.1"),
    (re.compile(r"qwen", re.I), "Qwen-Image"),
    (re.compile(r"flux.{0,4}2.{0,4}klein|klein", re.I), "FLUX.2 Klein"),
    (re.compile(r"flux.{0,4}2", re.I), "FLUX.2"),
    (re.compile(r"flux", re.I), "FLUX.1"),
    (re.compile(r"ideogram", re.I), "Ideogram 4"),
    (re.compile(r"wan.{0,3}2\.?2", re.I), "Wan 2.2"),
    (re.compile(r"wan", re.I), "Wan"),
    (re.compile(r"ltx", re.I), "LTX Video"),
    (re.compile(r"hunyuan", re.I), "Hunyuan Video"),
    (re.compile(r"illustrious", re.I), "Illustrious"),
    (re.compile(r"pony", re.I), "Pony"),
    (re.compile(r"sdxl|stable-diffusion-xl|sd_xl", re.I), "SDXL"),
    (re.compile(r"sd ?1\.5|stable-diffusion-v1|sd15", re.I), "SD 1.5"),
)
# Words in a file or folder name strong enough to name the family alone.
_NAME_FAMILIES = re.compile(r"minimax_h3|mm-h3|krea2|anima|qwen[-_]image", re.I)


def base_family(value: Any) -> str:
    """One family label for a raw base-model name ('' when unknown)."""
    text = str(value or "").strip()
    if not text or text.lower() in ("unknown", "other"):
        return ""
    for pattern, label in _BASE_FAMILIES:
        if pattern.search(text):
            return label
    return re.sub(r"/lora$", "", text, flags=re.I).replace("_", " ")[:80]


def _base_from_meta(meta: dict[str, str]) -> str:
    for key in ("modelspec.architecture", "base_model", "ss_base_model_version", "ss_sd_model_name"):
        value = meta.get(key)
        if value:
            return value[:80]
    return ""


def _civitai_version_fields(info: Any) -> dict[str, Any]:
    """The fields of a Civitai model-version answer (API or Civitai Helper sidecar)."""
    if not isinstance(info, dict):
        return {}
    model = info.get("model") if isinstance(info.get("model"), dict) else {}
    out: dict[str, Any] = {}
    if isinstance(info.get("modelId"), int):
        out["model_id"] = info["modelId"]
    if isinstance(info.get("id"), int):
        out["version_id"] = info["id"]
    if isinstance(model.get("name"), str):
        out["model_name"] = model["name"][:160]
    if isinstance(info.get("name"), str):
        out["version_name"] = info["name"][:120]
    if isinstance(info.get("baseModel"), str):
        out["base_model"] = info["baseModel"][:80]
    words = _words(info.get("trainedWords"))
    if words:
        out["triggers"] = words
    text = plain_text(info.get("description") or model.get("description") or "")
    if text:
        out["description"] = text
    return out


def _sidecars(path: Path) -> list[Path]:
    stem = path.with_suffix("")
    return [Path(str(stem) + ".civitai.info"), Path(str(stem) + ".metadata.json"), Path(str(path) + ".rgthree-info.json"),
            Path(str(stem) + ".json")]


def preview_file(path: Path) -> Path | None:
    stem = str(path.with_suffix(""))
    for suffix in _PREVIEW_SUFFIXES:
        candidate = Path(stem + suffix)
        try:
            if candidate.is_file() and not candidate.is_symlink():
                return candidate
        except OSError:
            continue
    return None


def local_info(path: Path) -> dict[str, Any]:
    """Everything known about one LoRA file without any network."""
    info: dict[str, Any] = {"triggers": [], "base_model": "", "sources": []}
    civitai, manager, rgthree, a1111 = (_read_json(p) for p in _sidecars(path))
    if isinstance(civitai, dict):
        info["sources"].append("civitai.info")
        info.update({k: v for k, v in _civitai_version_fields(civitai).items() if v})
    if isinstance(manager, dict):
        info["sources"].append("metadata.json")
        nested = _civitai_version_fields(manager.get("civitai"))
        for key, value in nested.items():
            info.setdefault(key, value) if key != "triggers" else None
        if not info["triggers"]:
            info["triggers"] = nested.get("triggers") or _words(manager.get("trained_words") or manager.get("trainedWords"))
        if not info["base_model"] and isinstance(manager.get("base_model"), str):
            info["base_model"] = manager["base_model"][:80]
        if "model_name" not in info and isinstance(manager.get("model_name"), str):
            info["model_name"] = manager["model_name"][:160]
    if isinstance(rgthree, dict):
        info["sources"].append("rgthree-info.json")
        words = [w.get("word") if isinstance(w, dict) else w for w in rgthree.get("trainedWords") or []]
        if not info["triggers"]:
            info["triggers"] = _words([w for w in words if isinstance(w, str)])
        if not info["base_model"] and isinstance(rgthree.get("baseModel"), str):
            info["base_model"] = rgthree["baseModel"][:80]
        if "model_name" not in info and isinstance(rgthree.get("name"), str):
            info["model_name"] = rgthree["name"][:160]
        for key, src in (("model_id", "modelId"), ("version_id", "versionId")):
            if key not in info and isinstance(rgthree.get(src), int):
                info[key] = rgthree[src]
    if isinstance(a1111, dict) and ("activation text" in a1111 or "sd version" in a1111 or "preferred weight" in a1111):
        info["sources"].append("json")
        if not info["triggers"]:
            info["triggers"] = _words(a1111.get("activation text"))
        if not info["base_model"] and isinstance(a1111.get("sd version"), str) and a1111["sd version"] not in ("Unknown", ""):
            info["base_model"] = a1111["sd version"][:80]
        weight = a1111.get("preferred weight")
        if isinstance(weight, (int, float)) and not isinstance(weight, bool) and 0 < weight <= 2:
            info["strength"] = float(weight)
    meta = safetensors_metadata(path)
    if meta:
        info["sources"].append("safetensors")
        if not info["triggers"]:
            info["triggers"] = _words(meta.get("modelspec.trigger_phrase")) or []
        if not info["triggers"]:
            info["tags"] = _top_tags(meta)
        if not info["base_model"]:
            info["base_model"] = _base_from_meta(meta)
        if "model_name" not in info and meta.get("modelspec.title"):
            info["model_name"] = meta["modelspec.title"][:160]
    info["base_model"] = base_family(info["base_model"])
    if not info["base_model"]:
        # Metadata says nothing: a strong family word in the file or folder name.
        word = _NAME_FAMILIES.search(path.parent.name + " " + path.name)
        info["base_model"] = base_family(word.group(0)) if word else ""
    info["has_preview"] = preview_file(path) is not None
    return info


# ---------------------------------------------------------------------------
# Civitai (on demand)
# ---------------------------------------------------------------------------
def _host_ok(url: str, redirected: bool = False) -> bool:
    """A request goes to an allowlisted host; a redirect only to Civitai's own https hosts."""
    from urllib.parse import urlsplit  # noqa: PLC0415
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    own = host in CIVITAI_HOSTS or (redirected and host.endswith(".civitai.com"))
    return parts.scheme == "https" and own and not parts.username and parts.port in (None, 443)


class _CivitaiRedirect(urllib.request.HTTPRedirectHandler):
    max_redirections = CIVITAI_REDIRECTS_MAX

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:  # noqa: D401 - urllib hook
        if not _host_ok(newurl, redirected=True):
            raise LoraLibraryError("civitai_refused", "Civitai redirected to a foreign address")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _get(url: str, limit: int, accept: str) -> tuple[int, bytes]:
    if not _host_ok(url):
        raise LoraLibraryError("civitai_refused", "This address is not allowed")
    opener = urllib.request.build_opener(_CivitaiRedirect)
    request = urllib.request.Request(url, headers={"User-Agent": "film-suite-lora-browser/1", "Accept": accept})
    try:
        with opener.open(request, timeout=CIVITAI_TIMEOUT_S) as response:
            status = getattr(response, "status", 200)
            data = response.read(limit + 1)
    except urllib.error.HTTPError as exc:
        return exc.code, b""
    if len(data) > limit:
        raise LoraLibraryError("civitai_unavailable", "Civitai sent too much data")
    return status, data


def fetch_json(url: str) -> tuple[int, Any]:
    status, data = _get(url, CIVITAI_JSON_MAX, "application/json")
    if status != 200:
        return status, None
    try:
        return 200, json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise LoraLibraryError("civitai_unavailable", "Civitai sent an unreadable answer") from exc


def fetch_bytes(url: str) -> bytes | None:
    status, data = _get(url, CIVITAI_IMAGE_MAX, "image/webp,image/*")
    return data if status == 200 and data else None


def _example_url(url: str) -> str:
    """Ask the Civitai image CDN for a small rendition (``/width=640/``)."""
    return re.sub(r"/(?:width|original)=[^/]+/", f"/width={EXAMPLE_PX}/", url, count=1)


class LoraLibrary:
    """Hashes, Civitai answers and thumbnails cached under one plugin-data dir."""

    def __init__(self, cache_root: Path | None, *, fetch_json: Callable[[str], tuple[int, Any]] = fetch_json,
                 fetch_bytes: Callable[[str], bytes | None] = fetch_bytes, clock: Callable[[], float] = time.time) -> None:
        self.root = Path(cache_root) if cache_root is not None else None
        self.fetch_json = fetch_json
        self.fetch_bytes = fetch_bytes
        self.clock = clock
        self._lock = threading.Lock()
        self._local: dict[tuple[str, int, int], dict[str, Any]] = {}
        self._hashes: dict[tuple[str, int, int], str] | None = None
        self._thumbs: dict[tuple[str, int, int], str | None] = {}

    # ---- cache files ----------------------------------------------------
    def _dir(self) -> Path | None:
        if self.root is None:
            return None
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            if self.root.is_symlink() or not self.root.is_dir():
                return None
        except OSError:
            return None
        return self.root

    def _write(self, name: str, data: bytes) -> None:
        root = self._dir()
        if root is None or "/" in name or "\\" in name or name.startswith("."):
            return
        temp = root / f".lora-{secrets.token_hex(8)}.tmp"
        try:
            with open(temp, "xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, root / name)
        except OSError:
            try:
                temp.unlink()
            except OSError:
                pass

    def _read(self, name: str, limit: int) -> bytes | None:
        return None if self.root is None else _read_bounded(self.root / name, limit)

    # ---- local ----------------------------------------------------------
    def local(self, path: Path) -> dict[str, Any]:
        key = _stat_key(path)
        with self._lock:
            hit = self._local.get(key)
        if hit is None:
            hit = local_info(path)
            with self._lock:
                if len(self._local) > 4096:
                    self._local.clear()
                self._local[key] = hit
        return hit

    # ---- hashes -----------------------------------------------------------
    def _hash_table(self) -> dict[tuple[str, int, int], str]:
        if self._hashes is None:
            table: dict[tuple[str, int, int], str] = {}
            raw = self._read("hashes.json", 4 * 1024 * 1024)
            try:
                rows = json.loads(raw.decode("utf-8")) if raw else []
            except (UnicodeDecodeError, ValueError):
                rows = []
            for row in rows if isinstance(rows, list) else []:
                if isinstance(row, list) and len(row) == 4 and isinstance(row[3], str) and _HEX64.fullmatch(row[3]):
                    table[(str(row[0]), int(row[1]), int(row[2]))] = row[3]
            self._hashes = table
        return self._hashes

    def sha256(self, path: Path) -> str:
        key = _stat_key(path)
        with self._lock:
            known = self._hash_table().get(key)
        if known:
            return known
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()
        with self._lock:
            table = self._hash_table()
            table[key] = value
            rows = [[k[0], k[1], k[2], v] for k, v in list(table.items())[-4096:]]
        self._write("hashes.json", json.dumps(rows).encode("utf-8"))
        return value

    def known_hash(self, path: Path) -> str | None:
        with self._lock:
            return self._hash_table().get(_stat_key(path))

    # ---- civitai ----------------------------------------------------------
    def civitai_cached(self, path: Path) -> dict[str, Any] | None:
        digest = self.known_hash(path)
        if not digest:
            return None
        raw = self._read(f"{digest}.json", 1024 * 1024)
        try:
            value = json.loads(raw.decode("utf-8")) if raw else None
        except (UnicodeDecodeError, ValueError):
            return None
        if not isinstance(value, dict) or value.get("sha256") != digest:
            return None
        if value.get("base_model"):
            value["base_model"] = base_family(value["base_model"]) or value["base_model"]
        return value

    def fetch_civitai(self, path: Path) -> dict[str, Any]:
        """One lookup by hash. Found and not-found answers are kept; failures are not."""
        digest = self.sha256(path)
        try:
            status, body = self.fetch_json(CIVITAI_API + digest)
        except LoraLibraryError:
            raise
        except Exception as exc:  # noqa: BLE001 - offline, timeout, TLS: one finite answer
            raise LoraLibraryError("civitai_unavailable", "Civitai is not reachable") from exc
        if status == 404:
            record = {"sha256": digest, "found": False, "fetched_at": int(self.clock())}
            self._write(f"{digest}.json", json.dumps(record).encode("utf-8"))
            return record
        if status != 200 or not isinstance(body, dict):
            raise LoraLibraryError("civitai_unavailable", "Civitai did not answer")
        record: dict[str, Any] = {"sha256": digest, "found": True, "v": CIVITAI_RECORD_V, "fetched_at": int(self.clock()),
                                  **_civitai_version_fields(body)}
        if record.get("base_model"):
            record["base_model"] = base_family(record["base_model"]) or record["base_model"]
        if not record.get("description") and isinstance(record.get("model_id"), int):
            # The by-hash answer carries the version's notes; the model's own
            # description lives on the model. A miss here costs only the text.
            try:
                status, model = self.fetch_json(CIVITAI_MODEL_API + str(record["model_id"]))
            except Exception:  # noqa: BLE001
                status, model = 0, None
            text = plain_text(model.get("description")) if status == 200 and isinstance(model, dict) else ""
            if text:
                record["description"] = text
        examples = []
        for image in body.get("images") or []:
            if len(examples) >= EXAMPLE_MAX:
                break
            if not isinstance(image, dict) or image.get("type", "image") != "image" or not isinstance(image.get("url"), str):
                continue
            url = _example_url(image["url"])
            if not _host_ok(url):
                continue
            try:
                data = self.fetch_bytes(url)
            except Exception:  # noqa: BLE001 - a missing example is skipped
                data = None
            small = shrink(data, EXAMPLE_PX) if data else None
            if not small:
                continue
            index = len(examples)
            self._write(f"{digest}-{index}.img", small)
            meta = image.get("meta") if isinstance(image.get("meta"), dict) else {}
            examples.append({"index": index, "prompt": str(meta.get("prompt") or "")[:2000],
                             "width": image.get("width") if isinstance(image.get("width"), int) else None,
                             "height": image.get("height") if isinstance(image.get("height"), int) else None})
        record["examples"] = examples
        self._write(f"{digest}.json", json.dumps(record).encode("utf-8"))
        with self._lock:
            self._thumbs.pop(_stat_key(path), None)
        return record

    def example_data_url(self, digest: str, index: int) -> str | None:
        if not _HEX64.fullmatch(digest) or not 0 <= index < EXAMPLE_MAX:
            return None
        return data_url(self._read(f"{digest}-{index}.img", THUMB_MAX_BYTES))

    # ---- thumbnails --------------------------------------------------------
    def thumb(self, path: Path) -> str | None:
        """Local preview first, else the first cached Civitai example."""
        key = _stat_key(path)
        with self._lock:
            if key in self._thumbs:
                return self._thumbs[key]
        url = None
        preview = preview_file(path)
        if preview is not None:
            tag = hashlib.sha1(repr(_stat_key(preview)).encode("utf-8")).hexdigest()
            small = self._read(f"thumb-{tag}.img", THUMB_MAX_BYTES)
            if small is None:
                data = _read_bounded(preview, PREVIEW_MAX_BYTES)
                small = shrink(data, THUMB_PX) if data else None
                if small:
                    self._write(f"thumb-{tag}.img", small)
            url = data_url(small)
        if url is None:
            record = self.civitai_cached(path)
            if record and record.get("examples"):
                url = self.example_data_url(record["sha256"], record["examples"][0]["index"])
        with self._lock:
            if len(self._thumbs) > 2048:
                self._thumbs.clear()
            self._thumbs[key] = url
        return url


def civitai_state(record: dict[str, Any] | None) -> str:
    """none | found | not_found. A found answer from before v2 lost its
    pictures and description, so it reads as none (asked again on open)."""
    if record is None:
        return "none"
    if not record.get("found"):
        return "not_found"
    return "found" if record.get("v") == CIVITAI_RECORD_V else "none"


def civitai_link(record: dict[str, Any] | None, local: dict[str, Any]) -> str | None:
    source = record if record and record.get("found") else local
    model_id = source.get("model_id") if isinstance(source, dict) else None
    if not isinstance(model_id, int):
        return None
    version = source.get("version_id")
    return CIVITAI_MODEL_URL + str(model_id) + (f"?modelVersionId={version}" if isinstance(version, int) else "")
