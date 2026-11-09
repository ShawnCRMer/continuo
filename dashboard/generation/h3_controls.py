"""H3 Video inspector controls on the pinned conditioning graphs (lane w2-h3ctl).

Sampling (turbo on/off + strength, recipe, sigma shift video/audio, engine), user
LoRAs and performance (attention, sparse, cache) for ``minimax-h3-frames-v1`` and
``minimax-h3-references-v1``.  Source of every node and value: the artist's
``H3_CE_0.6_TEMPORAL_RC2.json`` (groups ENGINE, TURBO, SAMPLING RECIPE, USER LORAS,
ATTENTION BACKEND, SLA / NATIVE SPARSE, CACHE).

The controls are compiled structurally into the graph the pinned compile returns:
nodes are added, removed and rewired, never a second graph per combination.  A
request without any control key never reaches this module, so its submitted graph
is byte-identical to the one before (pinned in tests).

RC2 model order, kept here::

    UNET -> user LoRAs -> attention | sparse -> cache -> SigmaShift -> turbo LoRA
         -> AdaLN LoRA fix (only with user LoRAs) -> guider / scheduler

Sparse (SLA / native) replaces the attention patch as in RC2 (its own dense
fall-through backend follows the attention choice for SLA).  Pure: no network,
no files.  Loaded standalone (no package-relative imports).
"""

from __future__ import annotations

import copy
import math
import os
from typing import Any, Iterable, Mapping

TARGET_IDS = ("minimax-h3-frames-v1", "minimax-h3-references-v1")
CONTROL_KEYS = frozenset({"turbo", "turbo_strength", "recipe", "sigma_shift_video", "sigma_shift_audio", "loras",
                          "attention", "sparse", "sparsity", "cache", "engine"})
COUNT_MIN, COUNT_MAX = 1, 8
BATCH_KEYS = frozenset({"count", "take_index"})  # take_index is set by the route's fan-out only
SEED_MODULUS = 4294967296
LORA_MAX = 8
LORA_STRENGTH = (0.0, 2.0)
TURBO_STRENGTH = (0.0, 2.0)
SIGMA_SHIFT = (0.01, 100.0)
RECIPES = ("native", "sandwich", "fast_0903")
ATTENTIONS = ("comfy_kitchen", "sage", "pytorch")  # the catalog stage options (model_catalog.v1.json)
SPARSES = ("off", "sla", "native")
CACHES = ("none", "t8", "spectrum")
ENGINES = ("native", "shark")
SLA_SPARSITY = (0.6, 0.95, 0.85)    # min, max, RC2 value (break-even 0.60 per the node's tooltip)
NATIVE_TAU = (0.0, 4.0, 1.3)        # sol-attn threshold (RC2 value 1.3)
RECIPE_A_STEPS, RECIPE_C_STEPS = 3, 3
FAST_0903_C_SIGMAS = "0.9035, 0.6316, 0.3158, 0.0000"
LATENT_UPSCALER = "minimax_h3_latent_upscaler_3d_fp16.safetensors"
PINNED_TURBO_LORAS = frozenset({"minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors"})
H3_FAMILY = "MiniMax H3"
DEFAULTS = {"turbo": True, "turbo_strength": 0.8, "recipe": "native", "sigma_shift_video": 12.0,
            "sigma_shift_audio": 3.0, "loras": [], "attention": None, "sparse": "off", "sparsity": None,
            "cache": "none", "engine": "native"}

# Pinned node ids (identical in both graphs).
_UNET, _SHIFT, _TURBO = "1", "5", "6"
_SEED, _NOISE, _SAMPLER, _SCHEDULER, _GUIDER, _RUN = "9", "10", "11", "12", "13", "14"
_GENERATOR, _VIDEO_VAE = "8", "3"
_NEW = 300  # every added node id is >= 300 (pinned graphs use ids < 100)

# RC2 widget values (H3_CE_0.6_TEMPORAL_RC2.json), read once and frozen here.
_SLA_INPUTS = {"block_size": "64", "min_seq_len": 4096, "dense_last_steps": 1, "protect_audio": True, "enabled": True,
               "dense_steps": "0", "disable_fp16_accum": True, "stabilize_motion": False, "reference_protection": "Off",
               "tail_correction": False, "use_int8_qk": True, "engine": "comfy_kitchen"}
_SLA_DENSE = {"comfy_kitchen": "comfy_kitchen", "pytorch": "pytorch", "sage": "sage:auto", None: "comfy_kitchen"}
_NATIVE_SPARSE_INPUTS = {"start_percent": 0.2, "end_percent": 1.0, "dense_blocks": "", "min_tokens": 12288,
                         "extra_tokens": 256, "sink_conditioning": "exact_kv_and_rows", "verbose": False}
_T8_INPUTS = {"residual_diff_threshold": 0.12, "start_percent": 0.08, "end_percent": 0.95, "max_consecutive_hits": 2,
              "cache_device": "cpu", "metric_stride": 8, "verbose": False}
_SPECTRUM_INPUTS = {
    "enabled": True, "blend_weight": 0.5, "degree": 4, "ridge_lambda": 0.1, "window_size": 2.0, "flex_window": 0.75,
    "warmup_steps": 5, "tail_actual_steps": 1, "max_history": 8, "debug": False, "history_storage": "system_ram",
    "bootstrap_first_forecast": False, "anchor_residual_feedback": False, "selective_rollback_correction": False,
    "offline_smoothing_replay": True, "audio_blend_weight": 0.0, "offline_archive_storage": "system_ram",
    "model_aware_mode": "off", "model_aware_risk_threshold": 0.65, "model_aware_trust_shrinkage": False,
    "model_aware_replay_generic_correction": False, "generic_correction_mode": "coordinate_rls",
    "generic_correction_limiter": "hard_clip", "generic_correction_limit": 0.4,
    "generic_correction_attenuation": "no_attenuation", "sa_pece_forecast_policy": "balanced"}
_SHARK_OPTIONS = ("default_dtype=float32\nmultistep_fallback_sampler=ddim\nmultistep_extra_initial_steps=0\n"
                  "skip_final_model_call\nno_batch_loop\nmute_EO")

# Live classes (and combo options) each control value needs; health answers ``unavailable`` without them.
NEEDS: dict[tuple[str, Any], tuple[tuple[str, str | None, str | None], ...]] = {
    ("recipe", "sandwich"): (("SplitSigmas", None, None), ("DisableNoise", None, None),
                             ("KSamplerSelect", "sampler_name", "dpmpp_3m_sde_gpu"), ("SamplerCustomAdvanced", None, None)),
    ("recipe", "fast_0903"): (("SplitSigmas", None, None), ("DisableNoise", None, None),
                              ("KSamplerSelect", "sampler_name", "dpmpp_3m_sde_gpu"), ("SamplerCustomAdvanced", None, None),
                              ("LTXVSeparateAVLatent", None, None), ("LTXVConcatAVLatent", None, None),
                              ("ManualSigmas", None, None), ("MinimaxH3LatentUpscaler3D", "model_name", LATENT_UPSCALER)),
    ("loras", True): (("LoraLoaderModelOnly", None, None), ("H3AdaLNLoRAFix", "mode", "port")),
    ("attention", "comfy_kitchen"): (("ModelAttentionBackend", "attention", "comfy kitchen attention"),),
    ("attention", "pytorch"): (("ModelAttentionBackend", "attention", "pytorch attention"),),
    ("attention", "sage"): (("PathchSageAttentionKJ", "sage_attention", "auto"), ("MiniMaxH3MemoryEfficientSageAttentionPatch", None, None)),
    ("sparse", "sla"): (("H3SLAAttention", "dense_backend", "comfy_kitchen"),),
    ("sparse", "native"): (("BlockSparseAttention", None, None),),
    ("cache", "t8"): (("MiniMaxH3BlockCacheT8", None, None),),
    ("cache", "spectrum"): (("SpectrumApplyMiniMaxH3", None, None),),
    ("engine", "shark"): (("ClownsharKSampler_Beta", "sampler_name", "exponential/etdrk2_2s"),
                          ("ClownsharkChainsampler_Beta", "sampler_name", "multistep/res_3m"),
                          ("ClownOptions_ExtraOptions_Beta", None, None), ("ConditioningZeroOut", None, None)),
}

# Real renders through the Suite's own /submit on the local ComfyUI (lane w2-h3ctl, see the
# lane report: prompt id, seconds, ffprobe facts).  Only what is listed here is ``rendered``;
# every other control value is ``untested`` (offered, not proven).
EVIDENCE: dict[tuple[str, Any], dict[str, Any]] = {
    # Scalar controls are keyed (control, True): rendered with a value other than the pinned one.
    ("turbo", True): {'prompt_id': '698ca948-07e9-4809-a814-07f3e61ed2a9', 'date': '2026-10-22', 'seconds': 32.6, 'output': '960x544 107 frames 24/1 fps, aac 32000 Hz', 'note': 'turbo off: node 6 removed; mean frame difference to the default take 15.8 (0-255)'},
    ("recipe", "sandwich"): {'prompt_id': '9c43db86-7078-40cc-9e3e-209c2bc92039', 'date': '2026-10-22', 'seconds': 32.6, 'output': '960x544 107 frames 24/1 fps, aac 32000 Hz', 'note': '3 SamplerCustomAdvanced (A 3 | B 2 | C 3 of 8 steps); mean frame difference to the default 44.3'},
    ("loras", True): {'prompt_id': '95a142d6-d573-4b29-9ee0-6c1e48a7fb41', 'date': '2026-10-22', 'seconds': 32.5, 'output': '960x544 107 frames 24/1 fps, aac 32000 Hz', 'note': 'H3_Motion_BoosterV2 at 0.8 + H3AdaLNLoRAFix port; mean frame difference to the default 8.9'},
    ("attention", "sage"): {'prompt_id': '60784c87-d95a-432e-822c-ffd071d974d1', 'date': '2026-10-22', 'seconds': 24.6, 'output': '960x544 107 frames 24/1 fps, aac 32000 Hz', 'note': 'PathchSageAttentionKJ auto + MiniMaxH3MemoryEfficientSageAttentionPatch; 21.0 s engine time vs 21.1 s comfy_kitchen'},
    ("attention", "comfy_kitchen"): {'prompt_id': 'd3c73614-f902-4023-b1a2-715301c8c87f', 'date': '2026-10-22', 'seconds': 24.5, 'output': '960x544 107 frames 24/1 fps, aac 32000 Hz', 'note': 'ModelAttentionBackend comfy kitchen attention; mean frame difference to sage 8.2'},
    ("cache", "t8"): {'prompt_id': '740b7d78-62b7-4236-a601-c8242deac6da', 'date': '2026-10-22', 'seconds': 32.6, 'output': '960x544 107 frames 24/1 fps, aac 32000 Hz', 'note': 'ran, but the cache took 0 of 8 forwards (also 0 of 12 at 12 steps): the take is identical to the default; no speed-up measured at this size'},
    ("count", True): {'prompt_id': 'f2d206d8-14ae-4fe3-9637-237f6d126f98', 'date': '2026-10-22', 'seconds': 32.6, 'output': '960x544 107 frames 24/1 fps, aac 32000 Hz', 'note': 'count 2: two takes, seeds 4242 and 4243, one batch id; take 2 submitted when take 1 ended'},
}


class H3ControlError(ValueError):
    """Finite refusal of a control request, raised before any side effect."""

    def __init__(self, message: str, reason_code: str = "invalid_request") -> None:
        super().__init__(message)
        self.reason_code = reason_code


def _fail(message: str, reason_code: str = "invalid_request") -> H3ControlError:
    return H3ControlError(message, reason_code)


def _number(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise _fail(f"{name} must be a number from {low:g} to {high:g}")
    return round(float(value), 4)


def _choice(value: Any, name: str, options: Iterable[str]) -> str:
    options = tuple(options)
    if type(value) is not str or value not in options:
        raise _fail(f"{name} must be one of {', '.join(options)}")
    return value


def has_controls(args: Mapping[str, Any]) -> bool:
    return isinstance(args, Mapping) and any(key in args for key in CONTROL_KEYS)


def split_args(args: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """(the pinned compile's args, the control args); the batch keys belong to neither."""
    base = {k: v for k, v in args.items() if k not in CONTROL_KEYS and k not in BATCH_KEYS}
    return base, {k: v for k, v in args.items() if k in CONTROL_KEYS}


def batch_args(args: Mapping[str, Any]) -> dict[str, int] | None:
    """``{count, index}`` of one take of a Generate x N request; None for a single take (count absent or 1)."""
    if "count" not in args:
        if "take_index" in args:
            raise _fail("take_index belongs to a count request")
        return None
    count = args["count"]
    if isinstance(count, bool) or type(count) is not int or not COUNT_MIN <= count <= COUNT_MAX:
        raise _fail(f"count must be a whole number from {COUNT_MIN} to {COUNT_MAX}")
    index = args.get("take_index", 0)
    if isinstance(index, bool) or type(index) is not int or not 0 <= index < count:
        raise _fail("take_index must be a take of this count")
    return None if count == 1 else {"count": count, "index": index}


def take_seed(seed: int, index: int) -> int:
    return (seed + index) % SEED_MODULUS


def take_client_request_id(client_request_id: str, index: int) -> str:
    """Take 0 keeps the request's id (count 1 is the request before); take i gets ``:take-i``."""
    return client_request_id if index == 0 else f"{client_request_id}:take-{index}"


def batch_id(client_request_id: str) -> str:
    import hashlib

    return "h3b_" + hashlib.sha256(client_request_id.encode("utf-8")).hexdigest()[:24]


def normalize_loras(value: Any) -> list[dict[str, Any]]:
    if type(value) is not list or len(value) > LORA_MAX:
        raise _fail(f"loras must be a list of at most {LORA_MAX}")
    out, seen = [], set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != {"file", "strength"}:
            raise _fail(f"loras[{index}] must be {{file, strength}}")
        name = item["file"]
        if (type(name) is not str or not name.endswith(".safetensors") or len(name) > 240 or name.startswith(("/", "."))
                or "\\" in name or ".." in name.split("/") or any(c in name for c in ':*?"<>|') or name in seen):
            raise _fail(f"loras[{index}].file is not a LoRA file name or is listed twice")
        if name.rsplit("/", 1)[-1] in PINNED_TURBO_LORAS:
            raise _fail("the turbo LoRA is set with turbo and turbo_strength, not as a user LoRA")
        seen.add(name)
        strength = _number(item["strength"], f"loras[{index}].strength", *LORA_STRENGTH)
        out.append({"file": name, "strength": strength})
    return out


def validate_controls(target_id: str, args: Mapping[str, Any], steps: int) -> dict[str, Any]:
    """The normalized control set (every key filled from DEFAULTS); shape rules only, no live check."""
    if target_id not in TARGET_IDS:
        raise _fail("controls are offered for MiniMax H3 Frames and References only")
    given = {k: v for k, v in args.items() if k in CONTROL_KEYS}
    c = copy.deepcopy(DEFAULTS)
    if "turbo" in given:
        if type(given["turbo"]) is not bool:
            raise _fail("turbo must be true or false")
        c["turbo"] = given["turbo"]
    if "turbo_strength" in given:
        if not c["turbo"]:
            raise _fail("turbo_strength needs turbo on")
        c["turbo_strength"] = _number(given["turbo_strength"], "turbo_strength", *TURBO_STRENGTH)
    for key in ("sigma_shift_video", "sigma_shift_audio"):
        if key in given:
            c[key] = _number(given[key], key, *SIGMA_SHIFT)
    if "engine" in given:
        c["engine"] = _choice(given["engine"], "engine", ENGINES)
    if "recipe" in given:
        c["recipe"] = _choice(given["recipe"], "recipe", RECIPES)
    if c["recipe"] != "native":
        if c["engine"] != "native":
            raise _fail("a sampling recipe runs on the Native engine only")
        if steps < RECIPE_A_STEPS + RECIPE_C_STEPS + 1:
            raise _fail(f"the {c['recipe']} recipe needs at least {RECIPE_A_STEPS + RECIPE_C_STEPS + 1} steps")
    if c["engine"] == "shark" and steps < 2:
        raise _fail("the Shark engine needs at least 2 steps")
    if "loras" in given:
        c["loras"] = normalize_loras(given["loras"])
    if "attention" in given:
        c["attention"] = _choice(given["attention"], "attention", ATTENTIONS)
    if "sparse" in given:
        c["sparse"] = _choice(given["sparse"], "sparse", SPARSES)
    if "sparsity" in given:
        if c["sparse"] == "off":
            raise _fail("sparsity needs sparse sla or native")
        low, high, _ = SLA_SPARSITY if c["sparse"] == "sla" else NATIVE_TAU
        c["sparsity"] = _number(given["sparsity"], "sparsity", low, high)
    elif c["sparse"] != "off":
        c["sparsity"] = (SLA_SPARSITY if c["sparse"] == "sla" else NATIVE_TAU)[2]
    if c["sparse"] == "native" and c["attention"] is not None:
        raise _fail("native sparse runs its own attention; leave attention unset")
    if "cache" in given:
        c["cache"] = _choice(given["cache"], "cache", CACHES)
    return c


def requested(controls: Mapping[str, Any]) -> dict[str, Any]:
    """The values that differ from the pinned graph (take facts, receipts)."""
    return {k: copy.deepcopy(v) for k, v in controls.items() if v != DEFAULTS[k]}


def needs(controls: Mapping[str, Any]) -> list[tuple[str, str | None, str | None]]:
    """Every live class/option the controls need (sparse replaces the attention patch, as in RC2)."""
    out: list[tuple[str, str | None, str | None]] = []
    keys = [(k, controls[k]) for k in ("recipe", "sparse", "cache", "engine")]
    if controls["sparse"] == "off":
        keys.append(("attention", controls["attention"]))
    if controls["loras"]:
        keys.append(("loras", True))
    for key in keys:
        for need in NEEDS.get(key, ()):
            if key == ("sparse", "sla"):
                need = ("H3SLAAttention", "dense_backend", _SLA_DENSE[controls["attention"]])
            if need not in out:
                out.append(need)
    return out


# --------------------------------------------------------------------------- compile


def _consumers(graph: Mapping[str, Any], node_id: str, slot: int = 0) -> list[tuple[str, str]]:
    return [(nid, key) for nid, node in graph.items() for key, value in node["inputs"].items()
            if isinstance(value, list) and len(value) == 2 and value[0] == node_id and value[1] == slot]


def _rewire(graph: dict[str, Any], old: list[Any], new: list[Any], skip: Iterable[str] = ()) -> None:
    skip = set(skip)
    for nid, key in _consumers(graph, old[0], old[1]):
        if nid not in skip:
            graph[nid]["inputs"][key] = list(new)


class _Ids:
    def __init__(self, graph: Mapping[str, Any]) -> None:
        self.next = _NEW
        if any(k.isdigit() and int(k) >= _NEW for k in graph):
            raise _fail("graph already carries control nodes")

    def __call__(self) -> str:
        self.next += 1
        return str(self.next - 1)


def apply_controls(graph: Mapping[str, Any], controls: Mapping[str, Any], *, steps: int) -> dict[str, Any]:
    """The compiled pinned graph with the controls applied structurally (a new graph)."""
    g = copy.deepcopy(dict(graph))
    for node_id in (_UNET, _SHIFT, _TURBO, _NOISE, _SAMPLER, _SCHEDULER, _GUIDER, _RUN):
        if node_id not in g:
            raise _fail("the pinned graph lacks a node the controls need")
    new = _Ids(g)
    # Model chain between UNET and SigmaShift: user LoRAs -> attention | sparse -> cache.
    tail: list[Any] = [_UNET, 0]
    for item in controls["loras"]:
        nid = new()
        g[nid] = {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": list(tail), "lora_name": item["file"].replace("/", os.sep), "strength_model": item["strength"]}}
        tail = [nid, 0]
    if controls["sparse"] == "sla":
        nid = new()
        g[nid] = {"class_type": "H3SLAAttention", "inputs": {"model": list(tail), "sparsity_ratio": controls["sparsity"],
                                                            **_SLA_INPUTS, "dense_backend": _SLA_DENSE[controls["attention"]]}}
        tail = [nid, 0]
    elif controls["sparse"] == "native":
        nid = new()
        g[nid] = {"class_type": "BlockSparseAttention", "inputs": {"model": list(tail), "selection": "sol-attn",
                                                                  "selection.tau": controls["sparsity"], **_NATIVE_SPARSE_INPUTS}}
        tail = [nid, 0]
    elif controls["attention"] in ("comfy_kitchen", "pytorch"):
        nid = new()
        g[nid] = {"class_type": "ModelAttentionBackend", "inputs": {
            "model": list(tail), "attention": "comfy kitchen attention" if controls["attention"] == "comfy_kitchen" else "pytorch attention"}}
        tail = [nid, 0]
    elif controls["attention"] == "sage":
        kj, h3 = new(), new()
        g[kj] = {"class_type": "PathchSageAttentionKJ", "inputs": {"model": list(tail), "sage_attention": "auto", "allow_compile": False}}
        g[h3] = {"class_type": "MiniMaxH3MemoryEfficientSageAttentionPatch", "inputs": {"model": [kj, 0]}}
        tail = [h3, 0]
    if controls["cache"] == "t8":
        nid = new()
        g[nid] = {"class_type": "MiniMaxH3BlockCacheT8", "inputs": {"model": list(tail), **_T8_INPUTS}}
        tail = [nid, 0]
    elif controls["cache"] == "spectrum":
        nid = new()
        g[nid] = {"class_type": "SpectrumApplyMiniMaxH3", "inputs": {"model": list(tail), **_SPECTRUM_INPUTS}}
        tail = [nid, 0]
    g[_SHIFT]["inputs"]["model"] = list(tail)
    g[_SHIFT]["inputs"]["shift_video"] = controls["sigma_shift_video"]
    g[_SHIFT]["inputs"]["shift_audio"] = controls["sigma_shift_audio"]

    # Turbo LoRA (pinned node 6): strength, or removed with its consumers moved to SigmaShift.
    model = [_TURBO, 0]
    if controls["turbo"]:
        g[_TURBO]["inputs"]["strength_model"] = controls["turbo_strength"]
    else:
        _rewire(g, [_TURBO, 0], [_SHIFT, 0])
        del g[_TURBO]
        model = [_SHIFT, 0]
    if controls["loras"]:
        fix = new()
        _rewire(g, model, [fix, 0])
        g[fix] = {"class_type": "H3AdaLNLoRAFix", "inputs": {"model": list(model), "mode": "port"}}

    # Engine / recipe: replace the single SamplerCustomAdvanced (node 14) structurally.
    if controls["engine"] == "shark":
        _shark(g, new, steps)
    elif controls["recipe"] in ("sandwich", "fast_0903"):
        _recipe(g, new, steps, controls["recipe"])
    _check_links(g)
    return g


def _ksampler(g: dict[str, Any], new: _Ids, name: str) -> str:
    nid = new()
    g[nid] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": name}}
    return nid


def _recipe(g: dict[str, Any], new: _Ids, steps: int, recipe: str) -> None:
    """RC2 SANDWICH / FAST 0903: A (3 steps, noise) -> B (no noise) -> C (3 steps).

    The curve is the pinned BasicScheduler (12) with the artist's steps and scheduler; B uses
    the artist's sampler (11), A and C dpmpp_3m_sde_gpu.  Sandwich: C finishes the curve's last
    3 steps without noise.  Fast 0903: B's denoised video is latent-upscaled 2x (audio passes
    around), C re-noises with the validated 0903 tail at full size, the audio of B is grafted back.
    """
    run = g[_RUN]["inputs"]
    guider, latent = list(run["guider"]), list(run["latent_image"])
    out = [_RUN, 0]
    split_ab, split_bc, quiet = new(), new(), new()
    g[split_ab] = {"class_type": "SplitSigmas", "inputs": {"sigmas": [_SCHEDULER, 0], "step": RECIPE_A_STEPS}}
    g[split_bc] = {"class_type": "SplitSigmas", "inputs": {"sigmas": [split_ab, 1], "step": steps - RECIPE_A_STEPS - RECIPE_C_STEPS}}
    g[quiet] = {"class_type": "DisableNoise", "inputs": {}}
    sampler_a, sampler_c = _ksampler(g, new, "dpmpp_3m_sde_gpu"), _ksampler(g, new, "dpmpp_3m_sde_gpu")
    a, b, c = new(), new(), new()
    g[a] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": [_NOISE, 0], "guider": guider, "sampler": [sampler_a, 0],
                                                              "sigmas": [split_ab, 0], "latent_image": latent}}
    g[b] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": [quiet, 0], "guider": list(guider), "sampler": [_SAMPLER, 0],
                                                              "sigmas": [split_bc, 0], "latent_image": [a, 0]}}
    if recipe == "sandwich":
        g[c] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": [quiet, 0], "guider": list(guider), "sampler": [sampler_c, 0],
                                                                  "sigmas": [split_bc, 1], "latent_image": [b, 0]}}
        final = [c, 0]
    else:
        part, up, join, tail, part_c, graft = new(), new(), new(), new(), new(), new()
        g[part] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": [b, 1]}}
        g[up] = {"class_type": "MinimaxH3LatentUpscaler3D", "inputs": {
            "latent": [part, 0], "model_name": LATENT_UPSCALER, "mode": "scale by multiplier", "mode.scale": 2.0, "align": 32,
            "enable_temporal_chunking": True, "force_unload": True, "device": "cuda", "precision": "fp16"}}
        g[join] = {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": [up, 0], "audio_latent": [part, 1]}}
        g[tail] = {"class_type": "ManualSigmas", "inputs": {"sigmas": FAST_0903_C_SIGMAS}}
        g[c] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": [_NOISE, 0], "guider": list(guider), "sampler": [sampler_c, 0],
                                                                  "sigmas": [tail, 0], "latent_image": [join, 0]}}
        g[part_c] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": [c, 0]}}
        g[graft] = {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": [part_c, 0], "audio_latent": [part, 1]}}
        final = [graft, 0]
    _rewire(g, out, final)
    del g[_RUN]


def _shark(g: dict[str, Any], new: _Ids, steps: int) -> None:
    """RC2 SHARK: ClownsharKSampler (etdrk2_2s, all but the last step) -> Chainsampler (res_3m).

    Same patched model and conditioning as Native, zeroed negative, the artist's curve (node 12).
    """
    guider = g[_GUIDER]["inputs"]
    model, positive = list(guider["model"]), list(guider["conditioning"])
    latent = list(g[_RUN]["inputs"]["latent_image"])
    negative, options, first, second = new(), new(), new(), new()
    g[negative] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": list(positive)}}
    g[options] = {"class_type": "ClownOptions_ExtraOptions_Beta", "inputs": {"extra_options": _SHARK_OPTIONS}}
    g[first] = {"class_type": "ClownsharKSampler_Beta", "inputs": {
        "eta": 0.5, "sampler_name": "exponential/etdrk2_2s", "scheduler": "simple", "steps": 1, "steps_to_run": steps - 1,
        "denoise": 1.0, "cfg": 1.0, "seed": [_SEED, 0], "sampler_mode": "standard", "bongmath": True, "model": model,
        "positive": positive, "negative": [negative, 0], "latent_image": latent, "sigmas": [_SCHEDULER, 0],
        "options_group.options0": [options, 0]}}
    g[second] = {"class_type": "ClownsharkChainsampler_Beta", "inputs": {
        "eta": 0.15, "sampler_name": "multistep/res_3m", "steps_to_run": -1, "cfg": 1.0, "sampler_mode": "resample",
        "bongmath": True, "latent_image": [first, 0], "options_group.options0": [options, 0]}}
    _rewire(g, [_RUN, 0], [second, 0])
    for node_id in (_RUN, _GUIDER, _NOISE, _SAMPLER):
        del g[node_id]


def _check_links(g: Mapping[str, Any]) -> None:
    for node_id, node in g.items():
        for name, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and type(value[1]) is int \
                    and value[0] not in g:
                raise _fail(f"node {node_id} input {name} points at a removed node")


def canvas_after(controls: Mapping[str, Any], canvas: Mapping[str, int]) -> dict[str, int]:
    """Fast 0903 delivers twice the canvas (its C pass runs on the 2x latent)."""
    if controls.get("recipe") == "fast_0903" and controls.get("engine", "native") == "native":
        return {"width": canvas["width"] * 2, "height": canvas["height"] * 2}
    return dict(canvas)


# --------------------------------------------------------------------------- live admission / descriptor


def _options(spec: Any) -> list[Any]:
    if not isinstance(spec, list) or not spec:
        return []
    if isinstance(spec[0], list):
        return list(spec[0])
    if len(spec) > 1 and isinstance(spec[1], Mapping) and isinstance(spec[1].get("options"), list):
        return [o.get("key") if isinstance(o, Mapping) else o for o in spec[1]["options"]]
    return []


def class_problem(need: tuple[str, str | None, str | None], info: Mapping[str, Any] | None) -> str | None:
    """None when the live ``/object_info`` of the class carries what the control needs."""
    class_type, field, option = need
    node = info.get(class_type) if isinstance(info, Mapping) else None
    if not isinstance(node, Mapping):
        return f"the engine has no {class_type} node"
    if field is None:
        return None
    inputs = node.get("input") or {}
    spec = (inputs.get("required") or {}).get(field) or (inputs.get("optional") or {}).get(field)
    if option not in _options(spec):
        return f"{class_type} does not offer {option}"
    return None


def lora_family(base_model: str, name: str, base_family: Any) -> str:
    """The family of a LoRA: its metadata base model, else a strong family word in its file name."""
    family = base_family(base_model) if base_model else ""
    return family or base_family(name.rsplit("/", 1)[-1])


def admit_loras(loras: list[dict[str, Any]], live: list[str] | None, family_of: Any) -> None:
    """Only files the engine lists and only MiniMax H3 LoRAs."""
    for item in loras:
        if live is not None and item["file"] not in live:
            raise _fail(f"The engine does not list {item['file']}", "invalid_request")
        family = family_of(item["file"])
        if family != H3_FAMILY:
            raise _fail(f"{item['file']} is not a {H3_FAMILY} LoRA" + (f" ({family})" if family else ""), "invalid_request")


def evidence(key: str, value: Any) -> dict[str, Any]:
    proof = EVIDENCE.get((key, value))
    if proof is None:
        return {"state": "untested"}
    return {"state": "rendered", **copy.deepcopy(proof)}


_CONTROLS = (
    # id, section, kind, label, options or range, values that have an evidence entry
    ("turbo", "sampling", "boolean", "Turbo", None),
    ("turbo_strength", "sampling", "number", "Turbo LoRA strength", {"min": TURBO_STRENGTH[0], "max": TURBO_STRENGTH[1], "step": 0.05}),
    ("recipe", "sampling", "select", "Recipe", list(RECIPES)),
    ("sigma_shift_video", "sampling", "number", "Sigma shift video", {"min": SIGMA_SHIFT[0], "max": SIGMA_SHIFT[1], "step": 0.5}),
    ("sigma_shift_audio", "sampling", "number", "Sigma shift audio", {"min": SIGMA_SHIFT[0], "max": SIGMA_SHIFT[1], "step": 0.5}),
    ("engine", "sampling", "select", "Engine", list(ENGINES)),
    ("loras", "loras", "lora_list", "LoRAs", {"max": LORA_MAX, "strength": {"min": LORA_STRENGTH[0], "max": LORA_STRENGTH[1]}}),
    ("attention", "performance", "select", "Attention", list(ATTENTIONS)),
    ("sparse", "performance", "select", "Sparse", list(SPARSES)),
    ("sparsity", "performance", "number", "Sparsity", {"sla": {"min": SLA_SPARSITY[0], "max": SLA_SPARSITY[1], "default": SLA_SPARSITY[2]},
                                                       "native": {"min": NATIVE_TAU[0], "max": NATIVE_TAU[1], "default": NATIVE_TAU[2], "meaning": "sol-attn tau"}}),
    ("cache", "performance", "select", "Cache", list(CACHES)),
)


def descriptor(problems: Mapping[tuple[str, Any], str | None] | None, *, lora_names: list[str] | None = None) -> dict[str, Any]:
    """Per control: options, default, live availability (``problems`` None = engine not checked) and evidence.

    ``problems`` maps (control, value) to the first live problem (None = available).
    """
    controls = []
    for cid, section, kind, label, spec in _CONTROLS:
        row: dict[str, Any] = {"id": cid, "section": section, "kind": kind, "label": label, "default": DEFAULTS[cid]}
        if isinstance(spec, list):
            options = []
            for value in spec:
                problem = None if problems is None else problems.get((cid, value))
                state = "unchecked" if problems is None else ("unavailable" if problem else "available")
                options.append({"value": value, "state": state, **({"reason": problem} if problem else {}),
                                "evidence": evidence(cid, value)})
            row["options"] = options
        elif spec is not None:
            row.update(spec)
        if cid == "loras":
            problem = None if problems is None else problems.get(("loras", True))
            row["state"] = "unchecked" if problems is None else ("unavailable" if problem else "available")
            if problem:
                row["reason"] = problem
            if lora_names is not None:
                row["files"] = list(lora_names)
            row["evidence"] = evidence("loras", True)
        elif cid in ("turbo", "turbo_strength", "sigma_shift_video", "sigma_shift_audio", "sparsity"):
            row["state"] = "available" if problems is not None else "unchecked"
            row["evidence"] = evidence(cid, True)
        rendered = [o for o in row.get("options", ()) if o["evidence"]["state"] == "rendered" and o["value"] != DEFAULTS[cid]]
        row["proof"] = "rendered" if rendered or row.get("evidence", {}).get("state") == "rendered" else "untested"
        controls.append(row)
    return {"schema": "studio.h3-controls/1", "targets": list(TARGET_IDS), "controls": controls,
            "count": {"min": COUNT_MIN, "max": COUNT_MAX, "seed": "seed + index", "proof": evidence("count", True)["state"],
                      "evidence": evidence("count", True)}}


def option_needs() -> dict[tuple[str, Any], tuple[tuple[str, str | None, str | None], ...]]:
    return dict(NEEDS)


__all__ = ["BATCH_KEYS", "batch_args", "batch_id", "take_client_request_id", "take_seed", "ATTENTIONS", "CACHES", "CONTROL_KEYS", "COUNT_MAX", "COUNT_MIN", "DEFAULTS", "ENGINES", "EVIDENCE", "H3ControlError",
           "H3_FAMILY", "NEEDS", "RECIPES", "SPARSES", "TARGET_IDS", "admit_loras", "apply_controls", "canvas_after",
           "class_problem", "descriptor", "evidence", "has_controls", "lora_family", "needs", "normalize_loras",
           "option_needs", "requested", "split_args", "validate_controls"]
