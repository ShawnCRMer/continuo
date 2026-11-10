"""ComfyUI H3 provider: compile panel values and run truthful static preflight.

The graph is cloned and only the five manifest-declared parameter pointers
are mutated.  Every node ID, class, link, fixed assertion, sampler,
scheduler, VAE, encoder, and model filename is preserved.  The effective
frame count uses the manifest's named rule ``round_fps_then_align_modulo``
implemented directly — a manifest expression is never evaluated.

Static health verifies reachability, pinned runtime versions, required
classes, input schema compatibility, and model files without ever queuing
work or calling ``/prompt``.
"""

from __future__ import annotations

import asyncio
import copy
import math
import time
from dataclasses import dataclass
import os
import ntpath
import re
from typing import Any, Mapping

from .contract import COMFY_ENRICHMENT_KEYS, GenerationRequest, StaticHealth
from .bounded_download import DownloadDigest
from .comfyui_client import ComfyClientError
from .job_store import InvalidRequestError, JobNotFoundError


def _load_h3_megapixels():
    """The one H3 megapixel table (``h3_megapixels.py`` next to this file; lane p13-vid VID-005).

    Loaded by path so this module works both as a package member and when loaded alone by path."""
    import hashlib as _hashlib
    import importlib.util as _importlib_util
    import sys as _sys
    from pathlib import Path as _Path

    path = _Path(__file__).with_name("h3_megapixels.py")
    name = "film_suite_h3_megapixels_" + _hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
    module = _sys.modules.get(name)
    if module is None:
        spec = _importlib_util.spec_from_file_location(name, path)
        module = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _sys.modules[name] = module
    return module


H3_MP = _load_h3_megapixels()

STUDIO_PROMPT_SENTINEL = "__STUDIO_PROMPT__"


def _load_runtime_policy() -> Any:
    """generation/runtime_policy.py by path (not a package member: the module tree is pinned)."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("film_suite_h3_runtime_policy", Path(__file__).with_name("runtime_policy.py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("The H3 runtime policy is unavailable.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime_policy = _load_runtime_policy()

# Tracks consecutive polls where the owned Comfy prompt ID is absent from both
# history and queue, so a transient absence does not immediately kill a job.
_ABSENCE_TRACKER: dict[str, dict[str, float | int]] = {}
_CANCEL_LOCKS: dict[str, tuple[asyncio.Lock, int]] = {}


def _reset_absence_tracker() -> None:
    _ABSENCE_TRACKER.clear()


def _cancel_lock(studio_job_id: str) -> asyncio.Lock:
    lock, users = _CANCEL_LOCKS.get(studio_job_id, (asyncio.Lock(), 0))
    _CANCEL_LOCKS[studio_job_id] = (lock, users + 1)
    return lock


def _release_cancel_lock(studio_job_id: str) -> None:
    lock, users = _CANCEL_LOCKS[studio_job_id]
    if users <= 1:
        _CANCEL_LOCKS.pop(studio_job_id, None)
    else:
        _CANCEL_LOCKS[studio_job_id] = (lock, users - 1)

_POINTER_FIELDS: Mapping[str, tuple[str, ...]] = {
    "prompt": ("105:104", "inputs", "prompt"),
    "duration_seconds": ("105:111", "inputs", "value"),
    "seed": ("124", "inputs", "value"),
    "aspect_ratio": ("115", "inputs", "aspect_ratio"),
    "megapixels": ("115", "inputs", "megapixels"),
}


def _require_pointer(graph: Any, pointer: tuple[str, ...]) -> Any:
    node = graph
    for part in pointer:
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f"pinned graph is missing pointer segment {part!r}")
        node = node[part]
    return node


def _resolve_pointer_string(graph: Any, pointer: str) -> Any:
    """Resolve an absolute manifest pointer like '/105:104/inputs/prompt'."""
    if not pointer.startswith("/"):
        raise ValueError(f"manifest pointer must start with '/': {pointer!r}")
    node = graph
    for part in pointer[1:].split("/"):
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f"graph is missing pointer segment {part!r}")
        node = node[part]
    return node


def _validate_pinned_graph(graph: Any, manifest: Mapping[str, Any]) -> None:
    """Reject a source graph that drifts from the manifest's fixed contract."""
    for pointer, expected in manifest.get("fixed_assertions", {}).items():
        actual = _resolve_pointer_string(graph, pointer)
        if actual != expected:
            raise ValueError(
                f"source graph fixed assertion differs at {pointer}: "
                f"expected {expected!r}, got {actual!r}"
            )
    for model_name, model in manifest.get("models", {}).items():
        pointer = model["assertion_pointer"]
        expected = model["filename"]
        actual = _resolve_pointer_string(graph, pointer)
        if actual != expected:
            raise ValueError(
                f"source graph model {model_name!r} differs at {pointer}: "
                f"expected {expected!r}, got {actual!r}"
            )


def _round_fps_then_align_modulo(
    duration_seconds: int,
    *,
    fps: int,
    minimum_frames: int,
    modulus: int,
    remainder: int,
) -> int:
    """Named manifest rule: round duration to frames, then align to remainder."""
    base = max(minimum_frames, round(duration_seconds * fps))
    return base + ((remainder - base) % modulus)


def compile_h3_request(
    *,
    graph: dict[str, Any],
    manifest: Mapping[str, Any],
    prompt: str,
    duration_seconds: int,
    seed: int,
    aspect_ratio: str,
    megapixels: float,
) -> dict[str, Any]:
    """Clone the pinned graph and write the five declared parameter pointers.

    Returns ``{"graph": ..., "requested": {...}, "effective_frames": ...}``.
    """
    parameters = manifest["parameters"]

    # ---- reject a source graph that drifts from the manifest contract ----
    _validate_pinned_graph(graph, manifest)

    # ---- validate arguments against the manifest only ----
    if not isinstance(prompt, str):
        raise ValueError("prompt must be text")
    if prompt == STUDIO_PROMPT_SENTINEL:
        raise ValueError("prompt may not equal the internal placeholder sentinel")
    if not prompt.strip():
        raise ValueError("prompt must contain non-whitespace text")

    duration_spec = parameters["duration_seconds"]
    if isinstance(duration_seconds, bool) or not isinstance(duration_seconds, int):
        raise ValueError("duration must be a whole number of seconds")
    if not duration_spec["minimum"] <= duration_seconds <= duration_spec["maximum"]:
        raise ValueError(
            f"duration must be between {duration_spec['minimum']} and {duration_spec['maximum']} seconds"
        )

    seed_spec = parameters["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not seed_spec["minimum"] <= seed <= seed_spec["maximum"]:
        raise ValueError(
            f"seed must be between {seed_spec['minimum']} and {seed_spec['maximum']}"
        )

    aspect_spec = parameters["aspect_ratio"]
    if aspect_ratio not in aspect_spec["enum"]:
        raise ValueError("aspect ratio is not an approved preset")

    if isinstance(megapixels, bool) or not isinstance(megapixels, (int, float)):
        raise ValueError("megapixels must be a number")
    # One list (h3_megapixels): the manifest enum is built from it; the provider checks the table itself.
    H3_MP.check(str(manifest.get("id") or "minimax-h3-t2v"), megapixels)

    # ---- compute effective frames with the named rule ----
    duration_capability = manifest["capabilities"]["duration"]
    frames_rule = duration_capability["effective_frames"]
    effective_frames = _round_fps_then_align_modulo(
        duration_seconds,
        fps=duration_capability["fps"],
        minimum_frames=frames_rule["minimum_frames"],
        modulus=frames_rule["modulus"],
        remainder=frames_rule["remainder"],
    )

    # ---- clone and mutate only the declared pointers ----
    compiled = copy.deepcopy(graph)
    prompt_container = _require_pointer(compiled, _POINTER_FIELDS["prompt"][:-1])
    duration_container = _require_pointer(compiled, _POINTER_FIELDS["duration_seconds"][:-1])
    seed_container = _require_pointer(compiled, _POINTER_FIELDS["seed"][:-1])
    aspect_container = _require_pointer(compiled, _POINTER_FIELDS["aspect_ratio"][:-1])
    megapixel_container = _require_pointer(compiled, _POINTER_FIELDS["megapixels"][:-1])
    prompt_container["prompt"] = prompt
    duration_container["value"] = float(duration_seconds)
    seed_container["value"] = seed
    aspect_container["aspect_ratio"] = aspect_ratio
    megapixel_container["megapixels"] = float(megapixels)

    return {
        "graph": compiled,
        "requested": {
            "duration_seconds": duration_seconds,
            "seed": seed,
            "aspect_ratio": aspect_ratio,
            "megapixels": float(megapixels),
        },
        "effective_frames": effective_frames,
    }


def compile_h3_sampling_request(*, graph: dict[str, Any], manifest: Mapping[str, Any], args: Mapping[str, Any]) -> dict[str, Any]:
    """Compile the explicitly versioned sampling target without relaxing v1."""
    if manifest.get("id") != "minimax-h3-t2v-sampling-v2" or not isinstance(manifest.get("sampling"), Mapping):
        raise ValueError("sampling controls require the v2 H3 target contract")
    _validate_pinned_graph(graph, manifest)
    allowed = {"prompt", "duration_seconds", "seed", "aspect_ratio", "steps", "sampler_name", "scheduler", "megapixels"}
    if set(args) != allowed:
        raise ValueError("v2 args contain missing or unknown fields")
    prompt = args["prompt"]
    if type(prompt) is not str or not prompt.strip() or prompt == STUDIO_PROMPT_SENTINEL:
        raise ValueError("prompt must contain non-placeholder text")
    params = manifest["sampling"]
    steps = args["steps"]
    if type(steps) is not int or not params["steps"]["minimum"] <= steps <= params["steps"]["maximum"]:
        raise ValueError("steps must be an integer from 1 to 100")
    for key in ("sampler_name", "scheduler"):
        if type(args[key]) is not str or args[key] not in params[key]["enum"]:
            raise ValueError(f"{key} is not an approved preset")
    megapixels = args["megapixels"]
    H3_MP.check("minimax-h3-t2v-sampling-v2", megapixels)
    base_args = {key: args[key] for key in ("prompt", "duration_seconds", "seed", "aspect_ratio", "megapixels")}
    base_manifest = copy.deepcopy(dict(manifest))
    base_manifest["parameters"] = copy.deepcopy(manifest["parameters"])
    base_manifest["parameters"]["megapixels"]["enum"] = list(params["megapixels"]["enum"])
    compiled = compile_h3_request(graph=graph, manifest=base_manifest, **base_args)
    compiled_graph = compiled["graph"]
    for key in ("steps", "sampler_name", "scheduler"):
        pointer = params[key]["pointer"].strip("/").split("/")
        _resolve_pointer_string(compiled_graph, params[key]["pointer"])
        container = _require_pointer(compiled_graph, tuple(pointer[:-1]))
        container[pointer[-1]] = args[key]
    compiled["requested"] = {key: args[key] for key in ("steps", "sampler_name", "scheduler", "megapixels")}
    return compiled


def _empty_checks() -> dict[str, bool]:
    return {
        "reachable": False,
        "runtime_version": False,
        "required_classes": False,
        "input_schema": False,
        "required_models": False,
        "workflow_integrity": False,
    }


def _combo_options(value: Any) -> set[Any]:
    """Extract the option list from either ComfyUI wire format.

    V1 internal nodes serialize a combo as ``["COMBO", {"options": [...]}]``;
    legacy nodes serialize it as ``[default, [options]]``.  Any content that
    cannot be hashed into a set is treated as absent so a malformed schema
    fails closed instead of leaking a raw ``TypeError``.
    """
    if not isinstance(value, list) or len(value) < 2:
        return set()
    spec = value[1]
    candidates: Any = None
    if isinstance(spec, dict) and isinstance(spec.get("options"), list):
        candidates = spec["options"]
    elif isinstance(spec, list):
        candidates = spec
    if candidates is None:
        return set()
    try:
        return set(candidates)
    except TypeError:
        return set()


def _numeric_bounds(value: Any) -> dict[str, Any]:
    """Extract the min/max dict from either wire format for a numeric input.

    V1 internal nodes serialize ``["FLOAT", {"min":…, "max":…}]``; legacy
    nodes serialize the bare ``{"min":…, "max":…}`` dict.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and len(value) == 2 and isinstance(value[1], dict):
        return value[1]
    return {}


async def static_health(
    *,
    graph: dict[str, Any],
    manifest: Mapping[str, Any],
    client: Any,
    tested: Any = (),
) -> StaticHealth:
    """Truthful static dependency health; never queues work or calls /prompt.

    The runtime stage follows the manifest's policy (``runtime_policy``): a
    ``schema_verified`` target runs on an untested runtime when the live node
    schema still matches the recorded fingerprint (``runtime_tested: false``);
    ``tested`` are this install's records of runtimes a published take passed.
    """
    provider_id = manifest["provider_id"]
    target_id = manifest["id"]
    checks = _empty_checks()
    runtime: dict[str, Any] = {}

    def answer(state: str, reason_code: str | None, reason_detail: str | None = None) -> StaticHealth:
        return StaticHealth(
            provider_id=provider_id,
            target_id=target_id,
            state=state,
            checks=checks,
            inference_tested=False,
            reason_code=reason_code,
            reason_detail=reason_detail,
            **runtime,
        )

    # Workflow integrity first, before any network call.
    try:
        _validate_pinned_graph(graph, manifest)
    except ValueError:
        return answer("misconfigured", "workflow_drift")
    policy = manifest["runtime_compatibility"].get("policy")
    fingerprint = runtime_policy.effective_fingerprint(manifest, graph)
    if policy == runtime_policy.SCHEMA_VERIFIED and fingerprint is None:
        return answer("misconfigured", "workflow_drift")
    checks["workflow_integrity"] = True

    # All remaining stages are network calls; any client error mid-check
    # degrades to a stable unreachable state rather than leaking raw.
    try:
        # Reachability.
        stats = await client.system_stats()
        await client.features()
        checks["reachable"] = True

        # Runtime versions: tested (listed or recorded on this install),
        # untested (schema_verified: the schema below decides), or blocked
        # (exact_tested_version_required for a manifest not migrated).
        identity = runtime_policy.runtime_identity(stats)
        verdict = runtime_policy.runtime_verdict(manifest, identity, tested)
        if identity is not None:
            runtime.update(runtime=identity, runtime_tested=verdict == "tested")
        if verdict == "blocked":
            return answer("unverified_runtime", "unverified_runtime")
        checks["runtime_version"] = True

        # Required classes.
        infos: dict[str, Any] = {}
        for class_type in sorted(set(manifest["required_classes"])):
            info = await client.object_info(class_type)
            if not isinstance(info, dict) or class_type not in info:
                return answer("incompatible", "missing_node", runtime_policy.detail(class_type, None))
            infos[class_type] = info
        checks["required_classes"] = True

        # Node schema against the pinned fingerprint: class, input, type, combo
        # values, numeric ranges, autogrow members.
        if fingerprint is not None:
            merged = {name: info[name] for name, info in infos.items()}
            problem = runtime_policy.schema_problem(fingerprint, merged)
            if problem is not None:
                class_type, name, kind = problem
                code = "missing_node" if kind == "missing_node" else "incompatible_node_schema"
                return answer("incompatible", code, runtime_policy.detail(class_type, name))

        # Input schema compatibility: manifest enums/bounds vs live ResolutionSelector.
        selector_info = await client.object_info("ResolutionSelector")
        selector_node = selector_info.get("ResolutionSelector", {}) if isinstance(selector_info, dict) else {}
        selector_input = selector_node.get("input", {}) if isinstance(selector_node, dict) else {}
        if not isinstance(selector_input, dict):
            selector_input = {}
        required_inputs = selector_input.get("required", {})
        if not isinstance(required_inputs, dict):
            required_inputs = {}
        aspect_manifest = set(manifest["parameters"]["aspect_ratio"]["enum"])
        aspect_live = _combo_options(required_inputs.get("aspect_ratio"))
        megapixel_manifest = set(manifest["parameters"]["megapixels"]["enum"])
        megapixel_live_spec = _numeric_bounds(required_inputs.get("megapixels"))
        live_mp_min = megapixel_live_spec.get("min")
        live_mp_max = megapixel_live_spec.get("max")
        megapixel_in_bounds = all(
            type(bound) in (int, float) and math.isfinite(bound)
            for bound in (live_mp_min, live_mp_max)
        ) and live_mp_min <= live_mp_max and all(
            live_mp_min <= value <= live_mp_max
            for value in megapixel_manifest
        )
        if not (aspect_manifest <= aspect_live) or not megapixel_in_bounds or not required_inputs:
            return answer("incompatible", "incompatible_node_schema")
        if manifest.get("id") == "minimax-h3-t2v-sampling-v2":
            scheduler_info = await client.object_info("BasicScheduler")
            sampler_info = await client.object_info("KSamplerSelect")
            scheduler_node = scheduler_info.get("BasicScheduler", {}) if isinstance(scheduler_info, dict) else {}
            sampler_node = sampler_info.get("KSamplerSelect", {}) if isinstance(sampler_info, dict) else {}
            scheduler_inputs = scheduler_node.get("input", {}).get("required", {}) if isinstance(scheduler_node, dict) else {}
            sampler_inputs = sampler_node.get("input", {}).get("required", {}) if isinstance(sampler_node, dict) else {}
            step_bounds = _numeric_bounds(scheduler_inputs.get("steps"))
            step_min, step_max = step_bounds.get("min"), step_bounds.get("max")
            scheduler_options = _combo_options(scheduler_inputs.get("scheduler"))
            sampler_options = _combo_options(sampler_inputs.get("sampler_name"))
            sampling = manifest.get("sampling", {})
            if (
                type(step_min) not in (int, float)
                or type(step_max) not in (int, float)
                or not math.isfinite(step_min)
                or not math.isfinite(step_max)
                or step_min > sampling["steps"]["minimum"]
                or step_max < sampling["steps"]["maximum"]
                or not set(sampling["scheduler"]["enum"]) <= scheduler_options
                or not set(sampling["sampler_name"]["enum"]) <= sampler_options
            ):
                return answer("incompatible", "incompatible_node_schema")
        checks["input_schema"] = True

        # Required model files.
        for model in manifest["models"].values():
            files = await client.models(model["folder"])
            if not isinstance(files, list) or model["filename"] not in files:
                return answer("misconfigured", "missing_model")
        checks["required_models"] = True
    except ComfyClientError:
        runtime.clear()
        return answer("unreachable", "unreachable")

    return answer("ready_static", None)


async def publish_job_output(
    *,
    job_store: Any,
    client: Any,
    manifest: Mapping[str, Any],
    studio_job_id: str,
) -> Any:
    """Publish one ready Studio job's declared output via the publishing module.

    Thin wrapper so the poll route can trigger publication without importing
    the publishing submodule directly (the generation package submodule
    registry stays exactly ``{contract}`` after plugin_api loads; publishing
    is imported lazily here, inside the package, using a relative import).
    """
    from .publishing import DEFAULT_STAGING_ROOT, publish_ready_output

    return await publish_ready_output(
        job_store=job_store,
        client=client,
        manifest=manifest,
        studio_job_id=studio_job_id,
        staging_root=DEFAULT_STAGING_ROOT,
    )


def _queue_prompt_ids(queue: Any, key: str) -> set[str]:
    """Collect prompt IDs from a /queue section (item tuple index 1)."""
    items = queue.get(key, []) if isinstance(queue, dict) else []
    result: set[str] = set()
    if not isinstance(items, list):
        return result
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], str):
            result.add(item[1])
    return result


def _correlate_prompt_id(
    queue: Any,
    history: Any,
    client_request_id: str,
) -> str | None:
    """Find the Comfy prompt ID whose extra_data.studio.client_request_id matches."""
    for key in ("queue_running", "queue_pending"):
        items = queue.get(key, []) if isinstance(queue, dict) else []
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, (list, tuple)) or len(item) < 4:
                continue
            extra_data = item[3]
            if (
                isinstance(extra_data, dict)
                and isinstance(extra_data.get("studio"), dict)
                and extra_data["studio"].get("client_request_id") == client_request_id
                and isinstance(item[1], str)
            ):
                return item[1]
    if isinstance(history, dict):
        for prompt_id, entry in history.items():
            if not isinstance(entry, dict):
                continue
            prompt = entry.get("prompt")
            if not isinstance(prompt, (list, tuple)) or len(prompt) < 4:
                continue
            extra_data = prompt[3]
            if (
                isinstance(extra_data, dict)
                and isinstance(extra_data.get("studio"), dict)
                and extra_data["studio"].get("client_request_id") == client_request_id
            ):
                return str(prompt_id)
    return None


def _history_status(history: Any, prompt_id: str) -> tuple[str | None, bool]:
    """Return (status_str, completed) for an owned prompt ID in /history."""
    if not isinstance(history, dict):
        return None, False
    entry = history.get(prompt_id)
    if not isinstance(entry, dict):
        return None, False
    status = entry.get("status")
    if not isinstance(status, dict):
        return None, False
    status_str = status.get("status_str")
    completed = status.get("completed")
    return (status_str if isinstance(status_str, str) else None, completed is True)


@dataclass(frozen=True)
class ComfySubmitOutcome:
    kind: str
    provider_job_id: str | None
    reason_code: str | None = None


@dataclass(frozen=True)
class ComfyPollOutcome:
    kind: str
    status: str
    output_identity: dict[str, str] | None = None
    reason_code: str | None = None


@dataclass(frozen=True)
class ComfyCancelOutcome:
    kind: str
    reason_code: str | None = None


def _v4_valid_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _v4_refusal(reason_code: str = "invalid_request") -> ComfySubmitOutcome:
    return ComfySubmitOutcome("refused", None, reason_code)


def _v4_plain_subfolder(subfolder: str) -> bool:
    """Lane w34-sec (scan finding 20): a ComfyUI output subfolder is empty or ONE plain folder name: no NUL or control
    character, no '/' or '\\', not '.' or '..', no drive (':'). The literals before this lane were double-escaped, so
    they tested for the text "\\x00" and a double backslash and let '..' and '..\\..' through to /view."""
    if subfolder == "":
        return True
    return (subfolder not in (".", "..")
            and not any(ord(char) < 32 for char in subfolder)
            and not any(char in subfolder for char in ("/", "\\", ":")))


def _v4_output_descriptor(entry: Any, manifest: Mapping[str, Any]) -> dict[str, str]:
    """Extract one manifest-approved descriptor without importing publication code."""
    if not isinstance(entry, Mapping) or not isinstance(manifest, Mapping):
        raise ValueError("output descriptor is invalid")
    output = manifest.get("output")
    if not isinstance(output, Mapping):
        raise ValueError("output descriptor is invalid")
    node_id = output.get("node_id")
    accepted_keys = output.get("accepted_keys")
    required_type = output.get("required_folder_type", "output")
    extension = output.get("extension", ".mp4")
    if (
        not isinstance(node_id, (str, int))
        or not isinstance(accepted_keys, (list, tuple))
        or not accepted_keys
        or not all(isinstance(key, str) and key for key in accepted_keys)
        or not isinstance(required_type, str)
        or not isinstance(extension, str)
        or not extension
    ):
        raise ValueError("output descriptor is invalid")
    outputs = entry.get("outputs")
    node_outputs = outputs.get(str(node_id)) if isinstance(outputs, Mapping) else None
    legacy = output.get("legacy_output")
    if output.get("path_text_folder") is not None:
        if isinstance(node_outputs, Mapping):
            return _v4_text_descriptor(node_outputs, output, extension)
        # A job submitted on the graph before the take-pair save (lane fs-obvb) is
        # read by that graph's SaveVideo output; such a take has no sidecar.
        if not isinstance(legacy, Mapping) or not isinstance(outputs, Mapping):
            raise ValueError("output descriptor is missing")
        node_id, accepted_keys = legacy.get("node_id"), legacy.get("accepted_keys")
        if not isinstance(node_id, str) or not isinstance(accepted_keys, list) or not accepted_keys:
            raise ValueError("output descriptor is invalid")
        node_outputs = outputs.get(node_id)
    if not isinstance(node_outputs, Mapping):
        raise ValueError("output descriptor is missing")

    unique: dict[tuple[str, str, str], dict[str, str]] = {}
    for key in accepted_keys:
        descriptors = node_outputs.get(key)
        if descriptors is None:
            continue
        if not isinstance(descriptors, list):
            raise ValueError("output descriptor is invalid")
        for descriptor in descriptors:
            if not isinstance(descriptor, Mapping):
                raise ValueError("output descriptor is invalid")
            if set(descriptor) - {"filename", "subfolder", "type"} - COMFY_ENRICHMENT_KEYS:
                raise ValueError("output descriptor is invalid")
            filename = descriptor.get("filename")
            subfolder = descriptor.get("subfolder")
            folder_type = descriptor.get("type")
            if not all(isinstance(value, str) for value in (filename, subfolder, folder_type)):
                raise ValueError("output descriptor is invalid")
            if (
                not filename
                or filename != os.path.basename(filename)
                or ntpath.basename(filename) != filename
                or "\x00" in filename
                or "\\" in filename
                or "/" in filename
                or not filename.lower().endswith(extension.lower())
                or not _v4_plain_subfolder(subfolder)  # lane w34-sec (finding 20): real NUL/backslash, no ".."
                or folder_type != required_type
            ):
                raise ValueError("output descriptor is invalid")
            identity = (filename, subfolder, folder_type)
            unique[identity] = {
                "filename": filename,
                "subfolder": subfolder,
                "type": folder_type,
            }
    if not unique:
        raise ValueError("output descriptor is missing")
    if len(unique) != 1:
        raise ValueError("output descriptor is ambiguous")
    return next(iter(unique.values()))


def _v4_text_descriptor(node_outputs: Mapping[str, Any], output: Mapping[str, Any], extension: str) -> dict[str, str]:
    """A take pair's mp4 named by the path text a ``PreviewAny`` shows (lane fs-obvb).

    ``H3TrimAndSaveVideoWithMCtx`` returns the absolute path of the mp4 it
    wrote (``<output>/<folder>/<name>.mp4``; its sidecar sits beside it) and
    puts nothing in ``/history`` itself.  Only a bare basename directly in the
    manifest's fixed folder is accepted; ComfyUI's ``/view`` then serves it.
    """
    folder = output.get("path_text_folder")
    texts = node_outputs.get("text")
    if (
        not isinstance(folder, str)
        or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", folder)
        or not isinstance(texts, list)
        or len(texts) != 1
        or not isinstance(texts[0], str)
        or not texts[0].strip()
        or "\x00" in texts[0]
    ):
        raise ValueError("output descriptor is missing")
    parts = re.split(r"[\\/]+", texts[0].strip())
    filename = parts[-1]
    if (
        len(parts) < 2
        or parts[-2] != folder
        or filename in {"", ".", ".."}
        or not filename.lower().endswith(extension.lower())
        or not re.fullmatch(r"[A-Za-z0-9_.-]{1,200}", filename)
    ):
        raise ValueError("output descriptor is invalid")
    return {"filename": filename, "subfolder": folder, "type": str(output.get("required_folder_type", "output"))}


async def comfy_submit_v4(
    *,
    client: Any,
    manifest: Mapping[str, Any],
    graph: dict[str, Any],
    args: Mapping[str, Any],
    studio_job_id: str,
    client_request_id: str,
    workflow_hash: str,
    compiled_graph: dict[str, Any] | None = None,
) -> ComfySubmitOutcome:
    """Pure Contract-4 submit: compile and call ComfyUI exactly once.

    ``compiled_graph`` carries a graph the caller already compiled (pinned H3
    conditioning targets compile after their media upload); v1 and
    sampling-v2 always compile here.
    """
    if (
        not isinstance(manifest, Mapping)
        or not isinstance(graph, dict)
        or not isinstance(args, Mapping)
        or not _v4_valid_text(manifest.get("id"))
        or not _v4_valid_text(studio_job_id)
        or not _v4_valid_text(client_request_id)
        or not _v4_valid_text(workflow_hash)
        or not callable(getattr(client, "submit", None))
    ):
        return _v4_refusal()
    try:
        if compiled_graph is not None:
            if manifest.get("id") in {"minimax-h3-t2v", "minimax-h3-t2v-sampling-v2"} or not isinstance(compiled_graph, dict):
                return _v4_refusal()
            compiled = {"graph": compiled_graph}
        elif manifest.get("id") == "minimax-h3-t2v-sampling-v2":
            compiled = compile_h3_sampling_request(graph=graph, manifest=manifest, args=args)
        else:
            compiled = compile_h3_request(
                graph=graph,
                manifest=manifest,
                prompt=args["prompt"],
                duration_seconds=args["duration_seconds"],
                seed=args["seed"],
                aspect_ratio=args["aspect_ratio"],
                megapixels=args["megapixels"],
            )
    except Exception:
        return _v4_refusal()
    try:
        receipt = await client.submit(
            compiled["graph"],
            client_id=studio_job_id,
            extra_data={
                "studio": {
                    "studio_job_id": studio_job_id,
                    "client_request_id": client_request_id,
                    "target_id": manifest["id"],
                    "workflow_hash": workflow_hash,
                }
            },
        )
    except Exception:
        return ComfySubmitOutcome("transport_error", None, "submission_unknown")
    if not isinstance(receipt, Mapping):
        return ComfySubmitOutcome("malformed", None, "submission_unknown")
    try:
        if receipt.get("node_errors"):
            return _v4_refusal("invalid_request")
        prompt_id = receipt.get("prompt_id")
    except Exception:
        return ComfySubmitOutcome("malformed", None, "submission_unknown")
    if not _v4_valid_text(prompt_id):
        return ComfySubmitOutcome("malformed", None, "submission_unknown")
    return ComfySubmitOutcome("accepted", prompt_id, None)


async def comfy_recover_submit_v4(
    *, client: Any, client_request_id: str
) -> ComfySubmitOutcome:
    """Recover a lost submit response without ever issuing another submit."""
    if not _v4_valid_text(client_request_id) or not callable(getattr(client, "queue", None)):
        return ComfySubmitOutcome("malformed", None, "invalid_request")
    try:
        queue = await client.queue()
        history = await client.recent_history(max_items=100)
    except Exception:
        return ComfySubmitOutcome("transport_error", None, "submission_unknown")
    recovered = _correlate_prompt_id(queue, history, client_request_id)
    if recovered:
        return ComfySubmitOutcome("accepted", recovered, None)
    return ComfySubmitOutcome("malformed", None, "submission_unknown")


async def comfy_poll_v4(
    *, client: Any, provider_job_id: str, manifest: Mapping[str, Any]
) -> ComfyPollOutcome:
    """Pure Contract-4 poll by one opaque provider prompt ID."""
    if (
        not _v4_valid_text(provider_job_id)
        or not isinstance(manifest, Mapping)
        or not callable(getattr(client, "history", None))
        or not callable(getattr(client, "queue", None))
    ):
        return ComfyPollOutcome("malformed", "missing", reason_code="invalid_request")
    prompt_id = provider_job_id
    try:
        history = await client.history(prompt_id)
        queue = await client.queue()
    except Exception:
        return ComfyPollOutcome("transport_error", "missing", reason_code="unreachable")
    status_str, completed = _history_status(history, prompt_id)
    if status_str in {"error", "failed"}:
        return ComfyPollOutcome("failed", "failed", reason_code="execution_failed")
    if status_str == "interrupted":
        return ComfyPollOutcome("cancelled", "cancelled", reason_code="cancelled")
    if status_str == "success" or completed:
        try:
            entry = history.get(prompt_id) if isinstance(history, Mapping) else history
            descriptor = _v4_output_descriptor(entry, manifest)
        except Exception:
            return ComfyPollOutcome("failed", "failed", reason_code="output_invalid")
        return ComfyPollOutcome("ready", "ready", output_identity=descriptor)
    if prompt_id in _queue_prompt_ids(queue, "queue_running"):
        return ComfyPollOutcome("running", "running")
    if prompt_id in _queue_prompt_ids(queue, "queue_pending"):
        return ComfyPollOutcome("queued", "queued")
    return ComfyPollOutcome("missing", "missing", reason_code="job_lost")


async def comfy_stream_output_v4(
    *, client: Any, output_identity: Mapping[str, str], sink: Any
) -> DownloadDigest:
    """Revalidate and stream one opaque descriptor without publication access."""
    if not isinstance(output_identity, Mapping) or set(output_identity) != {
        "filename", "subfolder", "type"
    }:
        raise ValueError("output descriptor is invalid")
    filename = output_identity.get("filename")
    subfolder = output_identity.get("subfolder")
    folder_type = output_identity.get("type")
    if (
        not all(isinstance(value, str) for value in (filename, subfolder, folder_type))
        or not filename
        or filename != os.path.basename(filename)
        or ntpath.basename(filename) != filename
        or "\x00" in filename
        or "\\" in filename
        or "/" in filename
        or not filename.lower().endswith(".mp4")
        or not _v4_plain_subfolder(subfolder)  # lane w34-sec (finding 20)
        or folder_type != "output"
        or not callable(getattr(client, "stream_view_to_sink", None))
    ):
        raise ValueError("output descriptor is invalid")
    result = await client.stream_view_to_sink(
        {"filename": filename, "subfolder": subfolder, "type": folder_type},
        sink=sink,
        min_bytes=65536,
        max_bytes=1 << 30,
    )
    sha256 = getattr(result, "sha256", None)
    bytes_written = getattr(result, "bytes_written", None)
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256.lower())
        or type(bytes_written) is not int
        or bytes_written < 0
    ):
        raise ValueError("output stream result is invalid")
    return DownloadDigest(sha256.lower(), bytes_written)


async def comfy_cancel_v4(*, client: Any, provider_job_id: str) -> ComfyCancelOutcome:
    """Pure Contract-4 cancel; persistence is coordinator-owned."""
    if not _v4_valid_text(provider_job_id) or not callable(getattr(client, "cancel_prompt", None)):
        return ComfyCancelOutcome("refused", "invalid_request")
    try:
        response = await client.cancel_prompt(provider_job_id)
    except Exception:
        return ComfyCancelOutcome("transport_error", "unreachable")
    if isinstance(response, Mapping) and response.get("cancelled") is True:
        return ComfyCancelOutcome("accepted")
    return ComfyCancelOutcome("refused", "invalid_request")


async def comfy_submit(
    *,
    job_store: Any,
    client: Any,
    manifest: Mapping[str, Any],
    graph: dict[str, Any],
    args: Mapping[str, Any],
    client_request_id: str,
    workflow_hash: str,
    now: float | None = None,
):
    """Admit one Studio-owned Comfy job, then submit the compiled graph once.

    Admission happens in the ledger before any network I/O.  An exact retry
    returns the existing row; a different body under the same client ID is
    rejected by the ledger; a second active Comfy job raises
    ``ProviderBusyError``.  A lost ``/prompt`` response is correlated by
    ``extra_data.studio.client_request_id`` from the queue and bounded recent
    history — never by a second automatic submit.
    """
    if manifest.get("id") == "minimax-h3-t2v-sampling-v2":
        compiled = compile_h3_sampling_request(graph=graph, manifest=manifest, args=args)
    else:
        compiled = compile_h3_request(
            graph=graph,
            manifest=manifest,
            prompt=args["prompt"],
            duration_seconds=args["duration_seconds"],
            seed=args["seed"],
            aspect_ratio=args["aspect_ratio"],
            megapixels=args["megapixels"],
        )
    request = GenerationRequest(
        provider_id="comfyui-local",
        target_id=manifest["id"],
        args=dict(args),
        client_request_id=client_request_id,
    )
    try:
        existing = job_store.get_by_client_request_id(client_request_id)
        is_retry = True
    except JobNotFoundError:
        existing = None
        is_retry = False

    record = job_store.create_or_get_job(
        request,
        state="submitting",
        provider_job_id=existing.provider_job_id if existing is not None else None,
        now=now,
    )
    if record.provider_job_id is not None or record.is_terminal:
        return record

    if is_retry:
        # The original /prompt response was lost and the row never received an
        # ID.  Correlate by exact extra_data, never by a second submit.
        try:
            queue = await client.queue()
        except ComfyClientError:
            queue = {}
        try:
            history = await client.recent_history(max_items=100)
        except ComfyClientError:
            history = {}
        recovered = _correlate_prompt_id(queue, history, client_request_id)
        if recovered is not None:
            return job_store.update_provider_job(record.studio_job_id, recovered)
        # A concurrent caller may still have the first /prompt in flight while
        # the shared row shows no provider_job_id yet.  Within the just-accepted
        # grace window, return the existing row unchanged so that caller can
        # complete normally.  Only after the window, with correlation still
        # impossible, expose terminal submission_unknown and require an
        # explicit artist decision before a new client request is created.
        timestamp = now if now is not None else time.time()
        age = timestamp - record.created_at
        if age < _ABSENCE_GRACE_AFTER_ACCEPT_SECONDS:
            return record
        return job_store.transition(
            record.studio_job_id,
            "failed",
            reason_code="submission_unknown",
            now=now,
        )

    # Fresh row: submit exactly once.
    try:
        receipt = await client.submit(
            compiled["graph"],
            client_id=record.connection_id,
            extra_data={
                "studio": {
                    "studio_job_id": record.studio_job_id,
                    "client_request_id": client_request_id,
                    "target_id": manifest["id"],
                    "workflow_hash": workflow_hash,
                }
            },
        )
    except ComfyClientError:
        # Response lost; the ledger row stays 'submitting' for correlation on
        # the next submit with the same client ID.  Never resubmit here.
        return job_store.get_job(record.studio_job_id)

    node_errors = receipt.get("node_errors") if isinstance(receipt, dict) else None
    if node_errors:
        return job_store.transition(
            record.studio_job_id,
            "failed",
            reason_code="invalid_request",
            now=now,
        )
    prompt_id = receipt.get("prompt_id") if isinstance(receipt, dict) else None
    if not isinstance(prompt_id, str) or not prompt_id:
        return job_store.transition(
            record.studio_job_id,
            "failed",
            reason_code="submission_unknown",
            now=now,
        )
    job_store.update_provider_job(record.studio_job_id, prompt_id)
    return job_store.transition(record.studio_job_id, "queued", now=now)


_ABSENCE_GRACE_SECONDS = 60.0
_ABSENCE_GRACE_AFTER_ACCEPT_SECONDS = 30.0
_ABSENCE_MAX_CONSECUTIVE = 3


def _record_absence(studio_job_id: str, now: float) -> int:
    entry = _ABSENCE_TRACKER.setdefault(studio_job_id, {"count": 0, "first_absent_at": now})
    if entry["count"] == 0:
        entry["first_absent_at"] = now
    entry["count"] = int(entry["count"]) + 1
    return int(entry["count"])


def _clear_absence(studio_job_id: str) -> None:
    _ABSENCE_TRACKER.pop(studio_job_id, None)


async def comfy_poll(
    *,
    job_store: Any,
    client: Any,
    studio_job_id: str,
    now: float | None = None,
    manifest: Mapping[str, Any] | None = None,
):
    """Poll the exact owned Comfy prompt ID through history and queue.

    A terminal row returns its stored result without touching Comfy.  The
    owned ID is never inferred from another or a global queue item.  Absence
    from both sources is tolerated for at most three consecutive polls
    spanning the grace window, then the job is persisted ``failed/job_lost``.

    When ``manifest`` is supplied, a successful history result records the
    declared output identity on the ledger so a later publication can fetch
    exactly that descriptor.
    """
    record = job_store.get_job(studio_job_id)
    if record.is_terminal:
        # A ready job that somehow lacks its recorded output identity (e.g.
        # it became ready before publication wiring existed) is recoverable:
        # with a manifest available, backfill the identity from Comfy history
        # so publication can proceed.  Idempotent — this only fills a gap.
        if (
            manifest is not None
            and getattr(record, "state", None) == "ready"
            and not getattr(record, "provider_output_identity_json", None)
        ):
            prompt_id = record.provider_job_id
            if prompt_id:
                try:
                    history = await client.history(prompt_id)
                except ComfyClientError:
                    return record
                from .publishing import PublishingError, require_unique_output

                try:
                    entry = history.get(prompt_id)
                    identity = require_unique_output(entry, manifest)
                except PublishingError as exc:
                    try:
                        return job_store.fail_ready_without_output(studio_job_id, exc.reason_code)
                    except InvalidRequestError:
                        return job_store.get_job(studio_job_id)
                except Exception:
                    try:
                        return job_store.fail_ready_without_output(studio_job_id, "output_invalid")
                    except InvalidRequestError:
                        return job_store.get_job(studio_job_id)
                try:
                    return job_store.backfill_ready_output_identity(studio_job_id, identity)
                except InvalidRequestError:
                    return job_store.get_job(studio_job_id)
        return record

    prompt_id = record.provider_job_id
    timestamp = now if now is not None else time.time()

    if prompt_id is None:
        # The original /prompt response was lost and no provider_job_id was
        # ever recovered.  If the client never retries, this row would stay
        # 'submitting' forever and block all future Comfy submissions via the
        # one-active partial index.  Apply the same absence grace and
        # terminalize submission_unknown instead of wedging the provider.
        count = _record_absence(studio_job_id, timestamp)
        first_absent_at = float(_ABSENCE_TRACKER[studio_job_id]["first_absent_at"])
        age = timestamp - record.created_at
        grace = (
            _ABSENCE_GRACE_AFTER_ACCEPT_SECONDS
            if age < _ABSENCE_GRACE_SECONDS
            else _ABSENCE_GRACE_SECONDS
        )
        if count >= _ABSENCE_MAX_CONSECUTIVE and (timestamp - first_absent_at) >= grace:
            _clear_absence(studio_job_id)
            return job_store.transition(
                studio_job_id, "failed", reason_code="submission_unknown", now=now
            )
        return record

    try:
        history = await client.history(prompt_id)
        queue = await client.queue()
    except ComfyClientError:
        return record

    status_str, completed = _history_status(history, prompt_id)
    if status_str == "error":
        _clear_absence(studio_job_id)
        return job_store.transition(
            studio_job_id, "failed", reason_code="execution_failed", now=now
        )
    if status_str == "interrupted":
        _clear_absence(studio_job_id)
        return job_store.transition(
            studio_job_id, "cancelled", reason_code="cancelled", now=now
        )
    if status_str == "success" or completed:
        _clear_absence(studio_job_id)
        if manifest is not None:
            from .publishing import PublishingError, require_unique_output

            try:
                entry = history.get(prompt_id)
                identity = require_unique_output(entry, manifest)
                job_store.set_output_identity(studio_job_id, identity)
            except PublishingError as exc:
                return job_store.transition(
                    studio_job_id, "failed", reason_code=exc.reason_code, now=now
                )
            except Exception:
                return job_store.transition(
                    studio_job_id, "failed", reason_code="output_invalid", now=now
                )
        return job_store.transition(studio_job_id, "ready", now=now)

    # History is absent or nonterminal: fall through to exact queue membership.
    running = _queue_prompt_ids(queue, "queue_running")
    pending = _queue_prompt_ids(queue, "queue_pending")
    if prompt_id in running:
        _clear_absence(studio_job_id)
        return job_store.transition(studio_job_id, "running", now=now)
    if prompt_id in pending:
        _clear_absence(studio_job_id)
        return job_store.transition(studio_job_id, "queued", now=now)

    # Owned ID absent from both sources: apply the absence grace.
    count = _record_absence(studio_job_id, timestamp)
    first_absent_at = float(_ABSENCE_TRACKER[studio_job_id]["first_absent_at"])
    age = timestamp - record.created_at
    grace = (
        _ABSENCE_GRACE_AFTER_ACCEPT_SECONDS
        if age < _ABSENCE_GRACE_SECONDS
        else _ABSENCE_GRACE_SECONDS
    )
    if count >= _ABSENCE_MAX_CONSECUTIVE and (timestamp - first_absent_at) >= grace:
        _clear_absence(studio_job_id)
        return job_store.transition(
            studio_job_id, "failed", reason_code="job_lost", now=now
        )
    return record


async def comfy_cancel(
    *,
    job_store: Any,
    client: Any,
    studio_job_id: str,
    now: float | None = None,
):
    """Cancel exactly one owned queued or running Comfy prompt."""
    lock = _cancel_lock(studio_job_id)
    try:
        async with lock:
            record = job_store.get_job(studio_job_id)
            if (
                record.provider_id != "comfyui-local"
                or record.state not in {"queued", "running"}
                or not record.provider_job_id
            ):
                raise InvalidRequestError()
            response = await client.cancel_prompt(record.provider_job_id)
            if not isinstance(response, Mapping) or response.get("cancelled") is not True:
                raise InvalidRequestError()
            _clear_absence(studio_job_id)
            return job_store.transition(
                studio_job_id, "cancelled", reason_code="cancelled", now=now
            )
    finally:
        _release_cancel_lock(studio_job_id)


__all__ = [
    "STUDIO_PROMPT_SENTINEL",
    "ComfyClientError",
    "StaticHealth",
    "compile_h3_request",
    "compile_h3_sampling_request",
    "static_health",
    "comfy_submit",
    "comfy_poll",
    "comfy_cancel",
    "ComfySubmitOutcome",
    "ComfyPollOutcome",
    "ComfyCancelOutcome",
    "comfy_submit_v4",
    "comfy_recover_submit_v4",
    "comfy_poll_v4",
    "comfy_stream_output_v4",
    "comfy_cancel_v4",
    "_round_fps_then_align_modulo",
    "_reset_absence_tracker",
]
