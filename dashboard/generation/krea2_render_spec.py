"""Closed, versioned Krea2 T2I run contract. No network access or inference."""
from __future__ import annotations

import json
import math
import os
import re
import threading
from pathlib import Path
from typing import Any, TypedDict

if __package__:
    from .model_catalog import find_variant as _find_variant, krea_profile, catalog_view as _catalog_view, comfy_root as _comfy_root
else:
    # The agent tool (backend/film-suite/__init__.py) executes this file as
    # a top-level module; load the sibling catalog module without registering
    # anything in sys.modules.
    import importlib.util as _importlib_util

    _catalog_spec = _importlib_util.spec_from_file_location(
        "film_suite_agent_model_catalog", Path(__file__).resolve().with_name("model_catalog.py"))
    _catalog_module = _importlib_util.module_from_spec(_catalog_spec)
    _catalog_spec.loader.exec_module(_catalog_module)
    krea_profile = _catalog_module.krea_profile
    _find_variant = _catalog_module.find_variant
    _catalog_view = _catalog_module.catalog_view
    _comfy_root = _catalog_module.comfy_root
    _availability_spec = _importlib_util.spec_from_file_location(
        "film_suite_agent_model_availability", Path(__file__).resolve().with_name("model_availability.py"))
    _availability = _importlib_util.module_from_spec(_availability_spec)
    _availability_spec.loader.exec_module(_availability)
if __package__:
    from . import model_availability as _availability


class RenderLora(TypedDict):
    file: str
    strength: float


class RenderSpec(TypedDict):
    schema: str
    profile: str
    steps: int
    cfg: float
    sampler: str
    scheduler: str
    width: int
    height: int
    seed: int
    batch_size: int
    loras: list[RenderLora]
    negative: str


SCHEMA = "studio.krea2-render-spec/1"
# Single source: file roles and the Turbo preset come from the shared model
# catalog (generation/model_catalog.v1.json, family krea2). Values and their
# JSON kinds (cfg 1.0 is a float) are pinned byte-stable by
# tests/backend/test_krea2_catalog_source.py against recorded snapshots.
_TURBO = krea_profile("turbo")
_RAW = krea_profile("raw")
MODELS = {
    "turbo": _TURBO["model"],
    "raw": _RAW["model"],
}
KREA_FILES = {"encoder": _TURBO["encoder"], "vae": _TURBO["vae"]}
# Snapshot of comfy/samplers.py:971-975,1356,1365-1375 in the local install.
SAMPLERS = ("euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2", "exp_heun_2_x0", "exp_heun_2_x0_sde", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ipndm", "ipndm_v", "deis", "cfgpp_ud10_ab", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp", "gradient_estimation", "gradient_estimation_cfg_pp", "er_sde", "seeds_2", "seeds_3", "sa_solver", "sa_solver_pece", "ddim", "uni_pc", "uni_pc_bh2")
SCHEDULERS = ("simple", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta", "normal", "linear_quadratic", "kl_optimal")
CATALOG = Path(__file__).resolve().parents[1] / "image_builder" / "krea2-builder-catalog.v1.json"
MODEL_ROOT = _comfy_root() / "models"  # FILM_SUITE_COMFY_ROOT (generation/comfy_root.py), the Suite's one ComfyUI root


def default_spec(seed: int, width: int, height: int) -> dict[str, Any]:
    preset = _TURBO["preset"]
    return {"schema": SCHEMA, "profile": "turbo", "steps": preset["steps"], "cfg": preset["cfg"],
            "sampler": preset["sampler"], "scheduler": preset["scheduler"], "seed": seed,
            "width": width, "height": height, "batch_size": 1, "negative": "", "loras": []}


def validate_spec(value: Any, *, model_root: Path | None = None, check_files: bool = True) -> dict[str, Any]:
    """Shape, bounds and (``check_files``) whether ComfyUI can load every named file.

    Lane w24-models: a STORED recipe is read with ``check_files=False``: it names files, it does not
    need them, so a moved or deleted model never makes history (or a project) unreadable.  The file
    check belongs to the moment of use (submit, compile), where a missing file is ``MissingModelFile``.
    """
    if type(value) is dict and value.get("schema") == SCHEMA_V2:
        return validate_spec_v2(value, model_root=model_root, check_files=check_files)
    return _validate_spec_v1(value, model_root=model_root, check_files=check_files)


class MissingModelFile(ValueError):
    """A well-formed recipe names a file ComfyUI cannot load now (role, file, folder)."""

    def __init__(self, role: str, name: str, folder: str) -> None:
        super().__init__(f"render_spec {role} not installed")
        self.role, self.name, self.folder = role, name, folder


def _available(kind: str, name: Any, model_root: Path | None) -> bool:
    # The live engine list counts only for the Suite's own root; an explicit root (tests, a
    # second install) is answered from its folders plus its extra_model_paths.yaml.
    return _availability.present(kind, name, MODEL_ROOT if model_root is None else model_root, live=model_root is None)


def missing_files(value: Any, *, model_root: Path | None = None) -> list[dict[str, str]]:
    """Every file a well-formed spec names that ComfyUI cannot load now: ``[{role, name, folder}]``."""
    spec = validate_spec(value, model_root=model_root, check_files=False)
    named: list[tuple[str, str, str]] = []
    if spec.get("schema") == SCHEMA_V2:
        named += [(role, spec["files"][role], _availability.folder_of(role)) for role in V2_FILE_ROLES]
    else:
        named += [("diffusion_model", MODELS[spec["profile"]], "diffusion_models"),
                  ("text_encoder", KREA_FILES["encoder"], "text_encoders"), ("vae", KREA_FILES["vae"], "vae")]
    named += [("lora", item["file"], "loras") for item in spec["loras"]]
    return [{"role": role, "name": name, "folder": folder} for role, name, folder in named
            if not _available(folder, name, model_root)]


# --- render spec v2 (additive; v1 above/below stays byte-stable) -------------
# v2 adds: an explicit model file per role (chosen from the catalog alternates
# of the Krea 2 T2I variants, so Turbo/Raw are presets, not the model), 1-8
# pictures per run, and free size within the model's bounds.  A v1 spec, a
# snapshot without render_spec and every persisted v1 job keep their bytes.
SCHEMA_V2 = "studio.krea2-render-spec/2"
# E6: LoRAs are chained one LoraLoaderModelOnly node each (ids 400..431 in
# the mutation manifests). 32 is a sanity bound against a broken request, not
# a creative limit.
LORA_MAX = 32
V2_FILE_ROLES = ("diffusion_model", "text_encoder", "vae")
V2_BATCH_MIN = 1
V2_BATCH_MAX = 8
V2_SIZE_MIN = 512
V2_SIZE_MAX = 2048
V2_SIZE_STEP = 16
V2_MAX_PIXELS = 4_194_304
ASPECT_PRESETS = ("1:1", "4:5", "3:4", "2:3", "9:16", "16:9", "3:2", "21:9")
_T2I_TARGET = "krea2-turbo-t2i-create"
# WP6 M2: besides the catalog files, any file ComfyUI's loader lists for a
# role may be chosen.  The HTTP boundary admits such a name only when it is
# an exact member of the live loader list; this contract accepts it only as a
# bare file name (no folder part, no traversal) that is installed in one of
# the role's model folders.
V2_ROLE_FOLDERS = {"diffusion_model": ("diffusion_models", "unet"), "text_encoder": ("text_encoders", "clip"), "vae": ("vae",)}
_CUSTOM_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._+()\[\]-]{0,199}\.(?:safetensors|gguf|sft|ckpt|pt|pth|bin)")


def custom_file_name_ok(name: Any) -> bool:
    """A bare model file name: no separators, no drive, no '..'."""
    return type(name) is str and _CUSTOM_NAME.fullmatch(name) is not None and ".." not in name


def custom_file_folder(role: str, name: Any, model_root: Path | None = None) -> str | None:
    """The role's folder kind when ComfyUI can load ``name`` for it, else None.

    w24-models: ComfyUI's view (live list, else models/ + extra_model_paths.yaml), never only
    ``ComfyUI/models``: a file moved to another drive through extra_model_paths still counts.
    """
    if role not in V2_ROLE_FOLDERS or not custom_file_name_ok(name):
        return None
    folder = V2_ROLE_FOLDERS[role][0]
    return folder if _available(folder, name, model_root) else None


# WP7: a LoRA outside the builder catalog may be used when it is a relative
# ``.safetensors`` name (``/`` separators, no drive, no traversal) installed as
# a regular file below ``models/loras``.  The HTTP boundary admits such a name
# only when it is an exact member of ComfyUI's live LoRA loader list.
_CUSTOM_LORA = re.compile(r"(?![/.])(?!.*(?:^|/)\.{1,2}(?:/|$))[^\\:*?\"<>|\x00-\x1f]{1,240}\.safetensors")


def custom_lora_ok(name: Any, model_root: Path | None = None) -> bool:
    """An installed LoRA file named relative to ``models/loras``."""
    return custom_lora_name_ok(name) and _available("loras", name, model_root)


def custom_lora_name_ok(name: Any) -> bool:
    """A well-formed relative LoRA name (shape only; no file-system access)."""
    return type(name) is str and _CUSTOM_LORA.fullmatch(name) is not None and all(
        part == part.strip() and part for part in name.split("/"))


_LORA_LOCK = threading.Lock()
_LORA_FILES: tuple[tuple[str, int, int], frozenset[str]] | None = None


def catalog_lora_files() -> set[str]:
    """LoRA files named by the builder catalog (a fresh set per call).

    The parsed list is cached per ``(path, mtime_ns, size)`` of ``CATALOG``,
    so validating many persisted jobs reads the file once, and an edited
    catalog is picked up on the next call.
    """
    global _LORA_FILES
    try:
        status = os.stat(CATALOG)
        key = (str(CATALOG), status.st_mtime_ns, status.st_size)
        cached = _LORA_FILES
        if cached is None or cached[0] != key:
            with _LORA_LOCK:
                cached = _LORA_FILES
                if cached is None or cached[0] != key:
                    options = json.loads(CATALOG.read_text(encoding="utf-8"))["options"]
                    files = frozenset(o["file"] for o in options if o.get("axis") == "lora" and o.get("file"))
                    cached = _LORA_FILES = (key, files)
        return set(cached[1])
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError("LoRA catalog unavailable") from exc


def v2_file_choices() -> dict[str, list[dict[str, str]]]:
    """Allowed file names per role, read from the shared model catalog.

    The union over the Krea 2 variants that run on the Suite T2I target, in
    catalog order, each tagged with the variant that lists it first.
    """
    choices: dict[str, list[dict[str, str]]] = {role: [] for role in V2_FILE_ROLES}
    catalog = _catalog_view()  # shared, read-only
    family = next(f for f in catalog["families"] if f["id"] == "krea2")
    for variant in family["variants"]:
        if _T2I_TARGET not in variant.get("suite_targets", ()):
            continue
        for role in V2_FILE_ROLES:
            spec = variant["files"].get(role)
            for name in (spec or {}).get("names", ()):
                if all(item["name"] != name for item in choices[role]):
                    choices[role].append({"name": name, "folder": spec["folder"], "variant": variant["id"]})
    return choices


def aspect_preset_sizes(megapixels: float = 1.0) -> list[dict[str, Any]]:
    """Each aspect preset at ~``megapixels``, snapped to the v2 grid."""
    result = []
    for preset in ASPECT_PRESETS:
        w_ratio, h_ratio = (int(part) for part in preset.split(":"))
        area = megapixels * 1_048_576
        width = math.sqrt(area * w_ratio / h_ratio)
        height = width * h_ratio / w_ratio
        snap = lambda n: max(V2_SIZE_MIN, min(V2_SIZE_MAX, int(round(n / V2_SIZE_STEP)) * V2_SIZE_STEP))
        result.append({"id": preset, "ratio": [w_ratio, h_ratio], "width": snap(width), "height": snap(height)})
    return result


def default_spec_v2(seed: int, width: int, height: int, profile: str = "turbo") -> dict[str, Any]:
    source = _TURBO if profile == "turbo" else _RAW
    preset = source["preset"]
    return {"schema": SCHEMA_V2, "profile": profile, "steps": preset["steps"], "cfg": preset["cfg"],
            "sampler": preset.get("sampler", _TURBO["preset"]["sampler"]),
            "scheduler": preset.get("scheduler", _TURBO["preset"]["scheduler"]), "seed": seed,
            "width": width, "height": height, "batch_size": 1, "batch_index": 0, "negative": "", "loras": [],
            "files": {"diffusion_model": source["model"], "text_encoder": source["encoder"], "vae": source["vae"]}}


def validate_spec_v2(value: Any, *, model_root: Path | None = None, check_files: bool = True) -> dict[str, Any]:
    keys = set(default_spec(0, 1024, 1024)) | {"files", "batch_index"}
    if type(value) is not dict or set(value) != keys or value.get("schema") != SCHEMA_V2:
        raise ValueError("render_spec shape invalid")
    files = value["files"]
    if type(files) is not dict or set(files) != set(V2_FILE_ROLES):
        raise ValueError("render_spec files invalid")
    choices = v2_file_choices()
    folders: dict[str, str] = {}
    for role in V2_FILE_ROLES:
        name = files[role]
        known = next((item["folder"] for item in choices[role] if item["name"] == name), None) if type(name) is str else None
        if known is None and not custom_file_name_ok(name):
            raise ValueError(f"render_spec {role} file invalid")
        if known is None and check_files and role != "diffusion_model" and custom_file_folder(role, name, model_root) is None:
            raise MissingModelFile(role, name, V2_ROLE_FOLDERS[role][0])
        folders[role] = known or V2_ROLE_FOLDERS[role][0]
    for name in ("width", "height"):
        n = value[name]
        if type(n) is not int or not V2_SIZE_MIN <= n <= V2_SIZE_MAX or n % V2_SIZE_STEP:
            raise ValueError(f"render_spec {name} invalid")
    if value["width"] * value["height"] > V2_MAX_PIXELS:
        raise ValueError("render_spec size invalid")
    batch = value["batch_size"]
    if type(batch) is not int or not V2_BATCH_MIN <= batch <= V2_BATCH_MAX:
        raise ValueError("render_spec batch_size invalid")
    # Everything else follows the v1 rules exactly; the v1 check runs on a
    # v1-shaped view with the model file of the chosen profile replaced by
    # the explicitly chosen one for the installed-file check.
    legacy_view = {key: value[key] for key in value if key not in ("files", "batch_index")}
    legacy_view.update({"schema": SCHEMA, "batch_size": 1})
    _validate_spec_v1(legacy_view, model_root=model_root, model_file=files["diffusion_model"],
                      model_folder=folders["diffusion_model"], check_files=check_files)
    for role in ("text_encoder", "vae"):
        if check_files and not _available(folders[role], files[role], model_root):
            raise MissingModelFile(role, files[role], folders[role])
    if type(value["batch_index"]) is not int or not 0 <= value["batch_index"] < batch:
        raise ValueError("render_spec batch_index invalid")
    return {**value, "files": dict(files), "loras": [dict(item) for item in value["loras"]]}


def run_member_spec(spec: dict[str, Any], index: int) -> dict[str, Any]:
    """Picture ``index`` of a v2 run: same values, its own seed and index.

    One run of N pictures is N ledger jobs on the unchanged single-image
    publication seam, each reproducible from its own seed.
    """
    if spec.get("schema") != SCHEMA_V2 or type(index) is not int or not 0 <= index < spec["batch_size"]:
        raise ValueError("render_spec run member invalid")
    return {**spec, "seed": (spec["seed"] + index) % 18_446_744_073_709_551_616, "batch_index": index,
            "files": dict(spec["files"]), "loras": [dict(item) for item in spec["loras"]]}


def _validate_spec_v1(value: Any, *, model_root: Path | None = None, model_file: str | None = None,
                      model_folder: str = "diffusion_models", check_files: bool = True) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(default_spec(0, 1024, 1024)):
        raise ValueError("render_spec shape invalid")
    spec = dict(value)
    if spec["schema"] != SCHEMA or type(spec["profile"]) is not str or spec["profile"] not in MODELS:
        raise ValueError("render_spec profile invalid")
    def integer(name: str, lo: int, hi: int, step: int = 1) -> None:
        n = spec[name]
        if type(n) is not int or not lo <= n <= hi or n % step:
            raise ValueError(f"render_spec {name} invalid")
    integer("seed", 0, 18_446_744_073_709_551_615)
    integer("width", 16, 4096, 8)
    integer("height", 16, 4096, 8)
    integer("steps", 1, 100)
    # Batch >1 is deliberately not accepted: ImageResultIngestor and
    # Krea2CreateAdapter._extract_png_descriptor require exactly one image.
    integer("batch_size", 1, 1)
    if type(spec["cfg"]) not in (int, float) or not math.isfinite(spec["cfg"]) or not 0.1 <= spec["cfg"] <= 20:
        raise ValueError("render_spec cfg invalid")
    if type(spec["sampler"]) is not str or type(spec["scheduler"]) is not str or spec["sampler"] not in SAMPLERS or spec["scheduler"] not in SCHEDULERS:
        raise ValueError("render_spec sampler/scheduler invalid")
    if type(spec["negative"]) is not str or len(spec["negative"]) > 8192 or (spec["profile"] == "turbo" and spec["negative"]):
        raise ValueError("render_spec negative invalid")
    if not isinstance(spec["loras"], list) or len(spec["loras"]) > LORA_MAX:
        raise ValueError("render_spec loras invalid")
    allowed = catalog_lora_files()
    seen = set()
    for item in spec["loras"]:
        if type(item) is not dict or set(item) != {"file", "strength"}:
            raise ValueError("render_spec lora shape invalid")
        filename = item["file"]
        strength = item["strength"]
        if type(filename) is not str or filename in seen or (filename not in allowed and not custom_lora_name_ok(filename)):
            raise ValueError("render_spec lora file invalid")
        seen.add(filename)
        if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 2:
            raise ValueError("render_spec lora strength invalid")
        if check_files and not _available("loras", filename, model_root):
            raise MissingModelFile("lora", filename, "loras")
    model_name = model_file or MODELS[spec["profile"]]
    if check_files and not _available("diffusion_models", model_name, model_root):
        raise MissingModelFile("diffusion_model", model_name, "diffusion_models")
    return spec
