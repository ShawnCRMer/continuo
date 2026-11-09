"""Small, file-backed bridge between the Image workbench pane and Hermes tools.

Mirrors ``music_bridge``: the renderer owns the truth for the pane state and
publishes bounded, path-free snapshots per window; the agent enqueues bounded
named actions which the renderer applies through the same handlers its
buttons use and then acknowledges.  A pending action is never success.

Nothing here submits work: ``generate`` and ``vary`` are only requests to the
visible pane, which decides (exactly as for a click) whether it can start.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

# Store hygiene (Weg A1): a window that has not reported for a day is gone;
# its state file and action folder are removed at first use, then hourly.
PRUNE_AGE_SECONDS = 24 * 3600.0
PRUNE_INTERVAL_SECONDS = 3600.0
# A window silent this long is gone: removed on read with its actions (suite_bridge's value).
STALE_WINDOW_SECONDS = 3600.0
_PRUNED_AT: dict[str, float] = {}
_PRUNE_LOCK = threading.Lock()


STATE_SCHEMA = "studio.image-ui-state/1"
ACTION_SCHEMA = "studio.image-ui-action/1"
ACK_SCHEMA = "studio.image-ui-ack/1"
VIEW_SCHEMA = "studio.image-ui-view/1"
ACTION_TTL_SECONDS = 10.0
KEPT_LIMIT = 20
VARY_MAX_CELLS = 4
# Same bound as krea2_render_spec.LORA_MAX (this module is loaded standalone).
LORA_MAX = 32
AXES = ("character", "shot", "outfit", "scene", "style", "lora")
VARY_AXES = ("style", "shot", "scene", "outfit")
_WINDOW_RE = re.compile(r"^win_[0-9a-f]{32}$")
_ACTION_RE = re.compile(r"^act_[0-9a-f]{32}$")
_JOB_RE = re.compile(r"^imgjob_[0-9a-f]{32}$")
_REVISION_RE = re.compile(r"^imgrev_[0-9a-f]{32}$")
_KEY_RE = re.compile(r"^[a-z0-9_]{1,64}$")
_TOKEN_RE = re.compile(r"^[a-z0-9_]{1,48}$")
# Relative model-folder names as ComfyUI lists them (spaces and brackets occur);
# never absolute, never a parent step, never a drive or backslash.
_LORA_FILE_RE = re.compile(r"^(?![/.])(?!.*(?:^|/)\.\.(?:/|$))[^\\:\x00-\x1f]{1,240}\.safetensors$")
_VARY_VARIABLE_RE = re.compile(r"^(seed|steps|cfg|sampler|scheduler|profile|lora:[a-z0-9_]{1,64}|axis:(style|shot|scene|outfit))$")
# Absolute filesystem paths never belong in a pane snapshot.
_ABSOLUTE_PATH_RE = re.compile(r"(^|[\s\"'(=])([A-Za-z]:[\\/]|\\\\[^\\]|/(Users|home|tmp|var|mnt)/)")
_MAX_SEED = 18_446_744_073_709_551_615
_JOB_STATES = frozenset({"queued", "submitting", "submitted", "running", "ready", "published", "failed", "cancelled", "quarantined", "unknown"})
_CELL_STATES = frozenset({"queued", "running", "done", "failed", "skipped"})

ALLOWED_ACTIONS = frozenset({
    "select_tile",
    "clear_axis",
    "set_intent",
    "set_manual_prompt",
    "set_mode",
    "set_sampling",
    "set_lora_strength",
    "set_seed",
    "generate",
    "keep",
    "vary",
    "vary_stop",
    "restore",
    "open_result",
    "start_fresh",
    "set_canvas",
    "set_develop",
    "reset_develop",
    "set_develop_before",
    "set_develop_group",
    "set_develop_clipping",
    "set_develop_white_balance",
    "set_develop_preset",
    "auto_tone",
    "set_develop_masks",
    "generate_edit",
    "set_edit_light",
    "stop_edit",
    "accept_edit",
    "outpaint_edit",
    "accept_outpaint",
    "discard_outpaint",
    "set_edit_layer",
    "remove_background",
    "set_layer_mask",
    "regenerate_layer",
    "set_edit_crop",
    "edit_undo",
    "edit_redo",
    "edit_history_jump",
    "view_edit",
    "set_edit_mode",
    "targeted_adjust",
    "set_seed_mode",
    "set_size",
    "set_edit_selection",
    "set_edit_tool",
    "refine_edit_selection",
    "compare",
    "survey",
    "run_command",
    "set_edit_boxes",
    "find_edit_boxes",
    "run_edit_boxes",
    "export_edit",
})
# Edit selection (desktop selection/sel-history-max.js SEL_MODIFY, edit-stage FW_TOOLS): the menu and
# Modify ops, the px bounds of each Modify dialog field, the rail tools, the Select and Mask fields.
SELECTION_OPS = ("all", "none", "invert", "grow", "shrink", "feather", "border", "smooth")
SELECTION_OPS_W23 = ("reselect", "quick_mask", "transform", "save", "load")  # w23-img (IMG-007): Photoshop Select menu
SELECTION_LOAD_MODES = ("new", "add", "subtract", "intersect")
SELECTION_PX_MAX = {"grow": 500, "shrink": 500, "feather": 250, "border": 200, "smooth": 100}
SELECTION_TOOLS = ("move", "hand", "rect", "lasso", "object", "brush", "eraser", "box", "wand")  # w23-img: wand
# Image Box (E19): the intents of a box and the bounds the pane holds (desktop image/box/box-model.js).
BOX_INTENTS = ("add", "change", "move", "keep", "remove")
BOX_MAX = 16
BOX_LIMITS = ("off", "selection")  # Limit (E19, w4-image): off, or the selection as a hard mask of the box edit
_BOX_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,31}$")
REFINE_RANGES = {"radius": (0, 250), "smooth": (0, 100), "feather": (0, 250), "contrast": (0, 100), "shift": (-100, 100)}
REFINE_OUTPUTS = ("selection", "png")
# Actions that need a picture open in Edit (state.edit present); refused before they are queued.
EDIT_PICTURE_ACTIONS = frozenset({"set_edit_selection", "refine_edit_selection"})
# Edit panel modes the pane shows (Generative and Crop follow when they ship).
EDIT_MODES = frozenset({"adjust", "masks", "generate", "crop"})
# Edit > Generative (desktop develop/layers.js): wired operations and the layer state the pane reports.
# Color > WB: what the agent may choose, and what the pane reports (custom = any other Temp/Tint).
# As Shot, Auto and the light-source presets (desktop develop/white-balance.js WB_PRESETS).
WB_MODES = ("as_shot", "auto", "daylight", "cloudy", "shade", "flash")
# Adjust > Presets (use-image-develop-presets.js): the operations, the list and name bounds.
DEVELOP_PRESET_OPS = ("copy", "paste", "save", "apply", "delete", "preview", "preview_end", "export", "import")
# A preset file (desktop develop-preset-file.js PRESET_FILE_CHARS) handed over by the agent as text.
DEVELOP_PRESET_FILE_CHARS = 524_288
DEVELOP_PRESETS_MAX = 60
DEVELOP_PRESET_NAME_MAX = 40
DEVELOP_GROUP_IDS = ("light", "color", "mixer", "grading", "curve", "detail", "effects")


def _develop_group_list(value: Any, label: str) -> list[str]:
    """A non-empty list of develop groups, each once, in panel order (Copy… / the clipboard)."""
    if not isinstance(value, list) or not value or any(not isinstance(g, str) or g not in DEVELOP_GROUP_IDS for g in value) or len(set(value)) != len(value):
        _fail(f"{label} must list develop groups once each")
    return [g for g in DEVELOP_GROUP_IDS if g in value]
# Generate panel (lane panel): seed after each run (ComfyUI control_after_generate) and the size row.
SEED_MODES = ("fixed", "increment", "decrement", "random")
_FORMAT_RE = re.compile(r"^(custom|[0-9]{1,2}:[0-9]{1,2})$")


def _preset_name(value: Any, label: str) -> str:
    name = _text(value, label, DEVELOP_PRESET_NAME_MAX, allow_empty=False, path_free=False)
    if name != " ".join(name.split()):
        _fail(f"{label} must be trimmed with single spaces")
    return name


WB_STATE_MODES = (*WB_MODES, "custom")
# Targeted adjustment (use-image-tat.js): the groups, curve channels and the drag-unit bound of amount.
TARGETED_CHANNELS = ("master", "red", "green", "blue")
TARGETED_PROPERTIES = ("hue", "saturation", "luminance")
TARGETED_AMOUNT_MAX = 255.0
GENERATIVE_OPERATIONS = frozenset({"fill", "replace", "remove", "restyle", "relight"})
EDIT_MODELS = frozenset({"qwen-image-2.1", "flux3-klein", "flux2-klein", "krea2"})  # w24-krea: Krea 2 Edit
BOX_MODELS = EDIT_MODELS - {"krea2"}  # w24-krea: Krea 2 Edit has no Box (study: the grey box stays grey)
# Remove background: a layer that is only a layer mask (desktop develop/layer-cutout.js) and its bounds.
LAYER_MASK_FEATHER_MAX = 250
# Edit > Generative > Relight (desktop develop/relight-light.js): the live light's words and controls.
LIGHT_TONES = frozenset({"warm golden", "neutral white", "cool blue", "soft daylight", "dim, moody"})
LIGHT_FROM = frozenset({"from the left", "from the right", "from the front", "from behind", "from above", "from below"})
LIGHT_CONTROLS = {"strength": (0.0, 2.0), "softness": (0.0, 1.0), "contrast": (0.5, 4.0), "fill": (0.0, 1.0)}
# Stage 1b scene controls (optional in the state, so earlier snapshots stay valid).
LIGHT_SCENE_EXTRA = {"shadows": (0.0, 1.0), "background": (0.0, 1.0)}
# Stage 1b: up to three lights; strength and softness belong to each light, contrast and fill to the scene.
LIGHTS_MAX = 3
LIGHT_OWN_CONTROLS = ("strength", "softness")
# A light's free colour: sRGB hex, or a colour temperature in kelvin; the state names it tone "custom".
LIGHT_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
LIGHT_KELVIN = (1700, 12000)
LIGHT_STATE_TONES = LIGHT_TONES | {"custom"}


def _light_color_state(row: dict[str, Any], label: str) -> None:
    """color (hex) and kelvin (integer or null) of one light in the state."""
    if "color" in row:
        _pattern(row["color"], f"{label}.color", LIGHT_HEX)
    if "kelvin" in row and row["kelvin"] is not None:
        _integer(row["kelvin"], f"{label}.kelvin", *LIGHT_KELVIN)
LIGHT_SHAPE_STATES = frozenset({"none", "reading", "ready", "failed"})
# The pane reports edit model families as the backend named them (/image/edit/capabilities).
_FAMILY_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{0,47}$")
# Develop (Edit > Light / Color / Effects): our own control ids and ranges (desktop develop/groups/*.js).
DEVELOP_LIGHT = {
    "exposure": (-5.0, 5.0),
    "contrast": (-100.0, 100.0),
    "highlights": (-100.0, 100.0),
    "shadows": (-100.0, 100.0),
    "whites": (-100.0, 100.0),
    "blacks": (-100.0, 100.0),
}
# Temp / Tint are relative to the picture as rendered (0 = as is), like Lightroom on non-raw files.
DEVELOP_COLOR = {
    "temp": (-100.0, 100.0),
    "tint": (-100.0, 100.0),
    "vibrance": (-100.0, 100.0),
    "saturation": (-100.0, 100.0),
}
DEVELOP_EFFECTS = {
    "texture": (-100.0, 100.0),
    "clarity": (-100.0, 100.0),
    "dehaze": (-100.0, 100.0),
    "vignette": (-100.0, 100.0),
    "vigMidpoint": (0.0, 100.0),
    "vigFeather": (0.0, 100.0),
    "vigRoundness": (-100.0, 100.0),
    "vigHighlights": (0.0, 100.0),
    "grain": (0.0, 100.0),
    "grainSize": (0.0, 100.0),
    "grainRoughness": (0.0, 100.0),
}
# Grading: per region (shadows, midtones, highlights, global) a tint strength, its hue in degrees and
# a lightness shift; Blending widens the overlap of the regions, Balance moves their split.
DEVELOP_GRADING = {
    **{f"{region}{field}": bounds for region in ("shadows", "midtones", "highlights", "global")
       for field, bounds in (("Sat", (0.0, 100.0)), ("Hue", (0.0, 360.0)), ("Lum", (-100.0, 100.0)))},
    "blending": (0.0, 100.0),
    "balance": (-100.0, 100.0),
}
# Color Mixer: Hue, Saturation and Luminance per colour band (Lightroom's HSL mixer).
DEVELOP_MIXER_BANDS = ("red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta")
DEVELOP_MIXER = {f"{band}{field}": (-100.0, 100.0) for field in ("Hue", "Sat", "Lum") for band in DEVELOP_MIXER_BANDS}
# Detail: Sharpening amount and its Masking (0 sharpens everything, higher spares flat areas).
DEVELOP_DETAIL = {"sharpen": (0.0, 150.0), "sharpenMasking": (0.0, 100.0), "nrLuminance": (0.0, 100.0), "nrDetail": (0.0, 100.0),
                  "nrColor": (0.0, 100.0), "nrColorDetail": (0.0, 100.0), "nrColorSmoothness": (0.0, 100.0)}
# Tone Curve: the parametric curve on display values; the splits bound its four regions.
DEVELOP_CURVE = {
    "highlights": (-100.0, 100.0),
    "lights": (-100.0, 100.0),
    "darks": (-100.0, 100.0),
    "shadows": (-100.0, 100.0),
    "splitShadows": (10.0, 70.0),
    "splitMid": (20.0, 80.0),
    "splitHighlights": (30.0, 90.0),
    "refineSaturation": (0.0, 100.0),
}
# Point curves of the Tone Curve: [[x, y], ...] in 0..1, at most 16 points; [] = identity.
DEVELOP_CURVE_POINTS = ("master", "red", "green", "blue")
DEVELOP_CURVE_POINTS_MAX = 16
DEVELOP_POINTS = {"curve": DEVELOP_CURVE_POINTS}
DEVELOP_GROUPS = {"light": DEVELOP_LIGHT, "color": DEVELOP_COLOR, "mixer": DEVELOP_MIXER, "grading": DEVELOP_GRADING, "curve": DEVELOP_CURVE, "detail": DEVELOP_DETAIL, "effects": DEVELOP_EFFECTS}
# Edit History rows the pane reports (its newest; SEL_HISTORY_MAX + 1 in the desktop).
DEVELOP_HISTORY_ROWS = 61
# Masks (develop/masks.js): vectors, positions 0..1 of the picture, radii a fraction of its long edge.
DEVELOP_MASK_MAX = 8
DEVELOP_MASK_COMPONENTS_MAX = 8
DEVELOP_MASK_LOCAL = {k: (-100.0, 100.0) for k in ("temp", "tint", "contrast", "highlights", "shadows", "whites", "blacks", "texture", "clarity", "dehaze", "hue", "saturation")}
DEVELOP_MASK_LOCAL["exposure"] = (-4.0, 4.0)
DEVELOP_MASK_OPS = frozenset({"add", "subtract", "intersect"})
DEVELOP_MASK_AI = ("subject", "background", "sky", "object", "words")
DEVELOP_MASK_KINDS = ("linear", "radial") + DEVELOP_MASK_AI
# Brush and range components (desktop develop/mask-kinds.js), appended to the kinds above.
DEVELOP_MASK_RANGES = ("luminance", "color", "depth")
DEVELOP_MASK_KINDS = DEVELOP_MASK_KINDS + ("brush",) + DEVELOP_MASK_RANGES
DEVELOP_MASK_STROKES_MAX = 100
DEVELOP_MASK_STROKE_POINTS_MAX = 1000
DEVELOP_MASK_BRUSH = {"size": (1.0, 100.0), "feather": (0.0, 100.0), "flow": (1.0, 100.0), "density": (1.0, 100.0)}
DEVELOP_MASK_COLOR_SAMPLES_MAX = 5
DEVELOP_MASK_POINTS_MAX = 16
DEVELOP_MASK_WORDS_MAX = 120
DEVELOP_HISTORY_KINDS = frozenset({"selection", "develop"})
# view_edit: the long edge of the picture handed to the agent, and the largest JPEG data URL.
VIEW_EDGE = (256, 2048)
VIEW_MAX_CHARS = 6_000_000
_VIEW_PREFIX = "data:image/jpeg;base64,"
_VIEW_B64_RE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")
# Actions that start or record work: the tool waits longer for their ack.
# export_edit (M-43): the File > Export formats (desktop image/output/use-image-output.js EXPORT_FORMATS).
EXPORT_EDIT_FORMATS = ("png8", "png16", "jpeg", "webp", "tiff8", "tiff16")
# run_command: a palette row may open a picker or start a job, so the agent waits as for keep.
SLOW_ACTIONS = frozenset({"generate", "vary", "keep", "view_edit", "accept_edit", "accept_outpaint", "run_command", "find_edit_boxes", "export_edit"})
# Outpaint (Edit > Crop): margins in pixels, each at most twice its picture side (the edit route
# checks the canvas against the picture; the bridge only bounds the numbers).
OUTPAINT_SIDES = ("left", "top", "right", "bottom")
OUTPAINT_JOB_STATES = frozenset({"submitting", "queued", "running", "stopping", "loading", "failed"})


class ImageBridgeValidationError(ValueError):
    """A public bridge envelope is malformed or contains an unsupported field."""


# --- primitive validators -------------------------------------------------

def _fail(message: str) -> None:
    raise ImageBridgeValidationError(message)


def _number(value: Any, label: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        _fail(f"{label} must be a finite number")
    number = float(value)
    if (minimum is not None and number < minimum) or (maximum is not None and number > maximum):
        _fail(f"{label} is out of range")
    return number


MASK_ID_ACCEPTS = 'a mask id from state.develop.masks (e.g. 1 or "1")'


def _mask_id(value: Any, label: str, *, all_ok: bool = False) -> int | str:
    """Lane w24-rest: a develop mask id as state shows it (a number) or written as text (\"1\"); \"all\" where allowed."""
    if all_ok and value == "all":
        return "all"
    text = value.strip() if isinstance(value, str) else None
    number = int(text) if text is not None and text.isascii() and text.isdigit() else value
    if isinstance(number, bool) or not isinstance(number, int) or not 1 <= number <= 2**31:
        _fail(f"{label} must be {MASK_ID_ACCEPTS}" + (' or "all"' if all_ok else ""))
    return number


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        _fail(f"{label} is invalid")
    return value


def _text(value: Any, label: str, maximum: int, *, allow_empty: bool = True, path_free: bool = True) -> str:
    if not isinstance(value, str) or len(value) > maximum or (not allow_empty and not value.strip()):
        _fail(f"{label} is invalid")
    if path_free and _ABSOLUTE_PATH_RE.search(value):
        _fail(f"{label} must not contain a filesystem path")
    return value


def _optional_text(value: Any, label: str, maximum: int) -> str | None:
    return None if value is None else _text(value, label, maximum)


def _bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{label} is invalid")
    return value


def _points(value: Any, label: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) > DEVELOP_CURVE_POINTS_MAX:
        _fail(f"{label} must be a list of at most {DEVELOP_CURVE_POINTS_MAX} points")
    out = []
    for index, point in enumerate(value):
        if not isinstance(point, list) or len(point) != 2:
            _fail(f"{label}[{index}] must be [x, y]")
        out.append([_number(point[0], f"{label}[{index}].x", minimum=0.0, maximum=1.0), _number(point[1], f"{label}[{index}].y", minimum=0.0, maximum=1.0)])
    return out


def _mask_point(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        _fail(f"{label} must be [x, y]")
    return [_number(value[0], f"{label}.x", minimum=-1.0, maximum=2.0), _number(value[1], f"{label}.y", minimum=-1.0, maximum=2.0)]


def _mask_ai(comp: dict[str, Any], cat: str, *, state: bool) -> dict[str, Any]:
    """An AI mask shape (desktop develop/masks.js): what the model is asked; state adds whether it is found."""
    kind = comp["kind"]
    need = {"kind"} | ({"op", "invert", "grow", "edge", "status"} if state else set())
    extra = {"op", "invert", "grow", "edge"} | ({"reason"} if state else set())
    if kind == "object":
        if ("points" in comp) == ("box" in comp):
            _fail(f"{cat} needs points or box")
        need |= {"points"} if "points" in comp else {"box"}
    elif kind == "words":
        need |= {"text"}
    comp = _keys(comp, cat, need, extra)
    shape: dict[str, Any] = {"kind": kind}
    if kind == "object" and "points" in comp:
        points = comp["points"]
        if not isinstance(points, list) or not 1 <= len(points) <= DEVELOP_MASK_POINTS_MAX:
            _fail(f"{cat}.points must be a list of 1 to {DEVELOP_MASK_POINTS_MAX} points")
        clean = []
        for n, point in enumerate(points):
            point = _keys(point, f"{cat}.points[{n}]", {"x", "y"} | ({"label"} if state else set()), set() if state else {"label"})
            p = {"x": _number(point["x"], f"{cat}.points[{n}].x", minimum=0.0, maximum=1.0), "y": _number(point["y"], f"{cat}.points[{n}].y", minimum=0.0, maximum=1.0)}
            if "label" in point:
                p["label"] = _integer(point["label"], f"{cat}.points[{n}].label", 0, 1)
            clean.append(p)
        if not any(p.get("label", 1) == 1 for p in clean):
            _fail(f"{cat}.points needs a point with label 1")
        shape["points"] = clean
    elif kind == "object":
        box = comp["box"]
        if not isinstance(box, list) or len(box) != 4:
            _fail(f"{cat}.box must be [x, y, w, h]")
        b = [_number(v, f"{cat}.box[{n}]", minimum=0.0, maximum=1.0) for n, v in enumerate(box)]
        if b[2] <= 0.002 or b[3] <= 0.002 or b[0] + b[2] > 1.000001 or b[1] + b[3] > 1.000001:
            _fail(f"{cat}.box must lie inside the picture")
        shape["box"] = b
    elif kind == "words":
        shape["text"] = _text(comp["text"], f"{cat}.text", DEVELOP_MASK_WORDS_MAX, path_free=False)
        if not shape["text"].strip():
            _fail(f"{cat}.text must not be empty")
    if "grow" in comp:
        shape["grow"] = _number(comp["grow"], f"{cat}.grow", minimum=-100.0, maximum=100.0)
    if "edge" in comp:
        shape["edge"] = _number(comp["edge"], f"{cat}.edge", minimum=-100.0, maximum=100.0)
    if state:
        if comp["status"] not in ("finding", "found", "failed"):
            _fail(f"{cat}.status must be finding, found or failed")
        shape["status"] = comp["status"]
        if "reason" in comp:
            shape["reason"] = _text(comp["reason"], f"{cat}.reason", 200, path_free=False)
    return shape


def _mask_gpu_kind(comp: dict[str, Any], cat: str, *, state: bool) -> dict[str, Any]:
    """A brush stroke list or a luminance / colour / depth range (desktop develop/mask-kinds.js)."""
    kind = comp["kind"]
    own = {"brush": {"strokes"}, "color": {"samples", "refine"}}.get(kind, {"range", "smooth"})
    need = {"kind"} | ({"op", "invert"} | own if state else ({"strokes"} if kind == "brush" else {"samples"} if kind == "color" else set()))
    extra = {"op", "invert"} | own
    # The state says whether the picture's depth is read (reading | ready | failed with the reason).
    if kind == "depth" and state:
        need |= {"status"}
        extra |= {"reason"}
    comp = _keys(comp, cat, need, extra)
    shape: dict[str, Any] = {"kind": kind}
    if kind == "brush":
        strokes = comp["strokes"]
        if not isinstance(strokes, list) or len(strokes) > DEVELOP_MASK_STROKES_MAX:
            _fail(f"{cat}.strokes must be a list of at most {DEVELOP_MASK_STROKES_MAX} strokes")
        out = []
        for n, stroke in enumerate(strokes):
            at = f"{cat}.strokes[{n}]"
            stroke = _keys(stroke, at, {"points"} | (set(DEVELOP_MASK_BRUSH) | {"mode"} if state else set()), set(DEVELOP_MASK_BRUSH) | {"mode"})
            points = stroke["points"]
            if not isinstance(points, list) or not 1 <= len(points) <= DEVELOP_MASK_STROKE_POINTS_MAX:
                _fail(f"{at}.points must be a list of 1 to {DEVELOP_MASK_STROKE_POINTS_MAX} points")
            item: dict[str, Any] = {"points": []}
            for i, point in enumerate(points):
                if not isinstance(point, list) or len(point) != 2:
                    _fail(f"{at}.points[{i}] must be [x, y]")
                item["points"].append([_number(point[0], f"{at}.points[{i}].x", minimum=-0.5, maximum=1.5), _number(point[1], f"{at}.points[{i}].y", minimum=-0.5, maximum=1.5)])
            for key, (lo, hi) in DEVELOP_MASK_BRUSH.items():
                if key in stroke:
                    item[key] = _number(stroke[key], f"{at}.{key}", minimum=lo, maximum=hi)
            if "mode" in stroke:
                if stroke["mode"] not in ("add", "erase"):
                    _fail(f"{at}.mode must be add or erase")
                item["mode"] = stroke["mode"]
            out.append(item)
        shape["strokes"] = out
    elif kind == "color":
        samples = comp["samples"]
        if not isinstance(samples, list) or len(samples) > DEVELOP_MASK_COLOR_SAMPLES_MAX or (not state and not samples):
            _fail(f"{cat}.samples must be a list of 1 to {DEVELOP_MASK_COLOR_SAMPLES_MAX} samples")
        out = []
        for n, sample in enumerate(samples):
            at = f"{cat}.samples[{n}]"
            sample = _keys(sample, at, {"x", "y"} | ({"w", "h"} if state else set()), {"w", "h"})
            s = {k: _number(sample[k], f"{at}.{k}", minimum=0.0, maximum=1.0) for k in ("x", "y", "w", "h") if k in sample}
            if s.get("x", 0) + s.get("w", 0) > 1.000001 or s.get("y", 0) + s.get("h", 0) > 1.000001:
                _fail(f"{at} must lie inside the picture")
            out.append(s)
        shape["samples"] = out
        if "refine" in comp:
            shape["refine"] = _number(comp["refine"], f"{cat}.refine", minimum=0.0, maximum=100.0)
    else:
        if "range" in comp:
            band = comp["range"]
            if not isinstance(band, list) or len(band) != 2:
                _fail(f"{cat}.range must be [lo, hi]")
            lo, hi = (_number(v, f"{cat}.range", minimum=0.0, maximum=1.0) for v in band)
            if lo > hi:
                _fail(f"{cat}.range must be [lo, hi] with lo <= hi")
            shape["range"] = [lo, hi]
        if "smooth" in comp:
            shape["smooth"] = _number(comp["smooth"], f"{cat}.smooth", minimum=0.0, maximum=100.0)
        if "status" in comp:
            if comp["status"] not in ("reading", "ready", "failed"):
                _fail(f"{cat}.status must be reading, ready or failed")
            shape["status"] = comp["status"]
            if "reason" in comp:
                shape["reason"] = _text(comp["reason"], f"{cat}.reason", 200, path_free=False)
    return shape


DEVELOP_MASK_OVERLAY_MODES = ("red", "mask", "bw", "black", "white", "edge")
DEVELOP_MASK_OVERLAY_COLOURS = ("red", "green", "blue", "cyan", "magenta")


def _mask_overlay(value: Any, label: str, *, state: bool) -> dict[str, Any]:
    """How the selected mask shows (desktop use-mask-overlay.js): show, mode, colour, opacity 0..100."""
    keys = {"show", "mode", "colour", "opacity"}
    value = _keys(value, label, keys if state else set(), set() if state else keys)
    if not value:
        _fail(f"{label} needs show, mode, colour or opacity")
    out: dict[str, Any] = {}
    if "show" in value:
        out["show"] = _bool(value["show"], f"{label}.show")
    if "mode" in value:
        if value["mode"] not in DEVELOP_MASK_OVERLAY_MODES:
            _fail(f"{label}.mode must be red, mask, bw, black, white or edge")
        out["mode"] = value["mode"]
    if "colour" in value:
        if value["colour"] not in DEVELOP_MASK_OVERLAY_COLOURS:
            _fail(f"{label}.colour must be red, green, blue, cyan or magenta")
        out["colour"] = value["colour"]
    if "opacity" in value:
        out["opacity"] = _number(value["opacity"], f"{label}.opacity", minimum=0.0, maximum=100.0)
    return out


def _masks(value: Any, label: str, *, state: bool) -> list[dict[str, Any]]:
    """A develop mask list: the agent's set_develop_masks payload, or state.develop.masks (ids required)."""
    if not isinstance(value, list) or len(value) > DEVELOP_MASK_MAX:
        _fail(f"{label} must be a list of at most {DEVELOP_MASK_MAX} masks")
    out = []
    for index, mask in enumerate(value):
        at = f"{label}[{index}]"
        fields = {"id", "name", "on", "invert", "amount", "refine", "local", "components"}
        mask = _keys(mask, at, fields if state else {"components"}, set() if state else fields - {"components"})
        item: dict[str, Any] = {}
        if "id" in mask:
            item["id"] = _mask_id(mask["id"], f"{at}.id")
        if "name" in mask:
            item["name"] = _text(mask["name"], f"{at}.name", 40)
        for key in ("on", "invert"):
            if key in mask:
                item[key] = _bool(mask[key], f"{at}.{key}")
        if "amount" in mask:
            item["amount"] = _number(mask["amount"], f"{at}.amount", minimum=0.0, maximum=200.0)
        if "refine" in mask:
            item["refine"] = _number(mask["refine"], f"{at}.refine", minimum=0.0, maximum=100.0)
        if "local" in mask:
            local = _keys(mask["local"], f"{at}.local", set(DEVELOP_MASK_LOCAL) if state else set(), set(DEVELOP_MASK_LOCAL))
            item["local"] = {k: _number(v, f"{at}.local.{k}", minimum=DEVELOP_MASK_LOCAL[k][0], maximum=DEVELOP_MASK_LOCAL[k][1]) for k, v in local.items()}
        components = mask["components"]
        if not isinstance(components, list) or not 1 <= len(components) <= DEVELOP_MASK_COMPONENTS_MAX:
            _fail(f"{at}.components must be a list of 1 to {DEVELOP_MASK_COMPONENTS_MAX} shapes")
        shapes = []
        for k, comp in enumerate(components):
            cat = f"{at}.components[{k}]"
            if not isinstance(comp, dict) or comp.get("kind") not in DEVELOP_MASK_KINDS:
                _fail(f"{cat}.kind must be linear, radial, subject, background, sky, object, words, brush, color, luminance or depth")
            if comp["kind"] == "brush" or comp["kind"] in DEVELOP_MASK_RANGES:
                shape = _mask_gpu_kind(comp, cat, state=state)
            elif comp["kind"] in DEVELOP_MASK_AI:
                shape = _mask_ai(comp, cat, state=state)
            elif comp["kind"] == "linear":
                comp = _keys(comp, cat, {"kind", "a", "b"} if not state else {"kind", "op", "invert", "a", "b"}, {"op", "invert"})
                shape = {"kind": "linear", "a": _mask_point(comp["a"], f"{cat}.a"), "b": _mask_point(comp["b"], f"{cat}.b")}
            else:
                need = {"kind", "c", "rx", "ry"} | ({"op", "invert", "angle", "feather"} if state else set())
                comp = _keys(comp, cat, need, {"op", "invert", "angle", "feather"})
                shape = {"kind": "radial", "c": _mask_point(comp["c"], f"{cat}.c"),
                         "rx": _number(comp["rx"], f"{cat}.rx", minimum=0.002, maximum=2.0), "ry": _number(comp["ry"], f"{cat}.ry", minimum=0.002, maximum=2.0)}
                if "angle" in comp:
                    shape["angle"] = _number(comp["angle"], f"{cat}.angle", minimum=-180.0, maximum=180.0)
                if "feather" in comp:
                    shape["feather"] = _number(comp["feather"], f"{cat}.feather", minimum=0.0, maximum=100.0)
            if "op" in comp:
                if comp["op"] not in DEVELOP_MASK_OPS:
                    _fail(f"{cat}.op must be add, subtract or intersect")
                shape["op"] = comp["op"]
            if "invert" in comp:
                shape["invert"] = _bool(comp["invert"], f"{cat}.invert")
            shapes.append(shape)
        item["components"] = shapes
        out.append(item)
    ids = [m["id"] for m in out if "id" in m]
    if len(ids) != len(set(ids)):
        _fail(f"{label} ids must be unique")
    return out


def _crop(value: Any, label: str, *, state: bool) -> dict[str, float]:
    """A crop rectangle in 0..1 of the (turned) picture (desktop develop/crop.js); state adds its px size and geometry."""
    item = _keys(value, label, {"x", "y", "w", "h"}, {"px", "angle", "rot", "flip"} if state else frozenset())
    out = {key: _number(item[key], f"{label}.{key}", minimum=0, maximum=1) for key in ("x", "y", "w", "h")}
    if out["w"] < 0.01 or out["h"] < 0.01 or out["x"] + out["w"] > 1.000001 or out["y"] + out["h"] > 1.000001:
        _fail(f"{label} must lie inside the picture")
    if state and "angle" in item:
        _number(item["angle"], f"{label}.angle", minimum=-45, maximum=45)
    if state and "rot" in item:
        _integer(item["rot"], f"{label}.rot", 0, 3)
    if state and "flip" in item:
        _bool(item["flip"], f"{label}.flip")
    if state and "px" in item:
        px = _keys(item["px"], f"{label}.px", {"x", "y", "w", "h"})
        for key in ("x", "y", "w", "h"):
            _integer(px[key], f"{label}.px.{key}", 0 if key in ("x", "y") else 1, 2**31)
    return out


def _outpaint_state(value: Any) -> None:
    """state.develop.outpaint: the margins, the running Outpaint and its variants (a larger picture)."""
    outpaint = _keys(value, "develop.outpaint", {"margins", "job"}, {"result", "accept_failed"})
    margins = outpaint["margins"]
    if not isinstance(margins, list) or len(margins) != 4:
        _fail("develop.outpaint.margins is invalid")
    for index, margin in enumerate(margins):
        _integer(margin, f"develop.outpaint.margins[{index}]", 0, 2**31)
    job = outpaint["job"]
    if job is not None:
        job = _keys(job, "develop.outpaint.job", {"state", "text"})
        if job["state"] not in OUTPAINT_JOB_STATES:
            _fail("develop.outpaint.job.state is invalid")
        _text(job["text"], "develop.outpaint.job.text", 240)
    if "result" in outpaint:
        result = _keys(outpaint["result"], "develop.outpaint.result", {"width", "height", "variants", "variant"}, {"shown"})
        if "shown" in result:
            _bool(result["shown"], "develop.outpaint.result.shown")
        _integer(result["width"], "develop.outpaint.result.width", 1, 16_384)
        _integer(result["height"], "develop.outpaint.result.height", 1, 16_384)
        variants = result["variants"]
        if not isinstance(variants, list) or not 1 <= len(variants) <= 4 or len(set(map(str, variants))) != len(variants):
            _fail("develop.outpaint.result.variants is invalid")
        for index, variant in enumerate(variants):
            _integer(variant, f"develop.outpaint.result.variants[{index}]", 0, 3)
        if result["variant"] not in variants:
            _fail("develop.outpaint.result.variant is invalid")
    if "accept_failed" in outpaint:
        _text(outpaint["accept_failed"], "develop.outpaint.accept_failed", 240)


def _generative_state(value: Any) -> None:
    """state.develop.generative: the picture's generative layers and the running edit."""
    generative = _keys(value, "develop.generative", {"layers", "job"}, {"area", "model", "models", "operation", "light", "strength", "warnings"})
    if "warnings" in generative:  # w24-edit2: model -> its measured reason for this operation (offered, not hidden)
        if not isinstance(generative["warnings"], dict) or len(generative["warnings"]) > 16:
            _fail("develop.generative.warnings is invalid")
        for family, reason in generative["warnings"].items():
            _pattern(family, "develop.generative.warnings key", _FAMILY_RE)
            _text(reason, f"develop.generative.warnings.{family}", 400, path_free=False)
    if "operation" in generative and generative["operation"] not in GENERATIVE_OPERATIONS:
        _fail("develop.generative.operation is invalid")
    if "light" in generative:
        # The Relight light as the pane shows it; shape: the picture's shape for the live light.
        # Stage 1b: the selected light's fields at the top, every light (1..3) in `lights`.
        light = _keys(generative["light"], "develop.generative.light", {"tone", "from", "x", "y", "behind", *LIGHT_CONTROLS, "live", "shape"}, {"on", "selected", "lights", "color", "kelvin", *LIGHT_SCENE_EXTRA})
        if light["tone"] not in LIGHT_STATE_TONES or light["from"] not in LIGHT_FROM:
            _fail("develop.generative.light tone or from is invalid")
        _light_color_state(light, "develop.generative.light")
        for key in ("x", "y"):
            _number(light[key], f"develop.generative.light.{key}", minimum=-1.0, maximum=1.0)
        for key, (low, high) in LIGHT_CONTROLS.items():
            _number(light[key], f"develop.generative.light.{key}", minimum=low, maximum=high)
        _bool(light["behind"], "develop.generative.light.behind")
        _bool(light["live"], "develop.generative.light.live")
        if light["shape"] not in LIGHT_SHAPE_STATES:
            _fail("develop.generative.light.shape is invalid")
        for key, (low, high) in LIGHT_SCENE_EXTRA.items():
            if key in light:
                _number(light[key], f"develop.generative.light.{key}", minimum=low, maximum=high)
        if "on" in light:
            _bool(light["on"], "develop.generative.light.on")
        if "lights" in light:
            rows = light["lights"]
            if not isinstance(rows, list) or not 1 <= len(rows) <= LIGHTS_MAX:
                _fail("develop.generative.light.lights must list 1 to 3 lights")
            for i, row in enumerate(rows):
                label = f"develop.generative.light.lights[{i}]"
                row = _keys(row, label, {"tone", "from", "x", "y", "behind", *LIGHT_OWN_CONTROLS, "on"}, {"color", "kelvin"})
                if row["tone"] not in LIGHT_STATE_TONES or row["from"] not in LIGHT_FROM:
                    _fail(f"{label} tone or from is invalid")
                _light_color_state(row, label)
                for key in ("x", "y"):
                    _number(row[key], f"{label}.{key}", minimum=-1.0, maximum=1.0)
                for key in LIGHT_OWN_CONTROLS:
                    low, high = LIGHT_CONTROLS[key]
                    _number(row[key], f"{label}.{key}", minimum=low, maximum=high)
                _bool(row["behind"], f"{label}.behind")
                _bool(row["on"], f"{label}.on")
            if "selected" not in light:
                _fail("develop.generative.light.selected is missing")
        if "selected" in light:
            _integer(light["selected"], "develop.generative.light.selected", 1, len(light.get("lights") or []) or 1)
    # The chosen edit model and the offered ones (families of /image/edit/capabilities).
    if "model" in generative and generative["model"] != "":
        _pattern(generative["model"], "develop.generative.model", _FAMILY_RE)
    if "models" in generative:
        models = generative["models"]
        if not isinstance(models, list) or len(models) > 16 or len(set(map(str, models))) != len(models):
            _fail("develop.generative.models is invalid")
        for index, family in enumerate(models):
            _pattern(family, f"develop.generative.models[{index}]", _FAMILY_RE)
    # Strength (charter image, owner decision 1): the panel's value 0..1 and whether the model takes it (else why).
    if "strength" in generative:
        strength = _keys(generative["strength"], "develop.generative.strength", {"value", "takes"}, {"why"})
        _number(strength["value"], "develop.generative.strength.value", minimum=0.0, maximum=1.0)
        _bool(strength["takes"], "develop.generative.strength.takes")
        if "why" in strength:
            _text(strength["why"], "develop.generative.strength.why", 200, path_free=False)
    # Area: the selection, the whole picture (Restyle only) or a mask id.
    if "area" in generative and generative["area"] not in ("selection", "all"):
        _integer(generative["area"], "develop.generative.area", 1, 2**31)
    layers = generative["layers"]
    if not isinstance(layers, list) or len(layers) > 16:
        _fail("develop.generative.layers is invalid")
    for index, layer in enumerate(layers):
        label = f"develop.generative.layers[{index}]"
        item = _keys(layer, label, {"id", "operation", "prompt", "variants", "variant", "on", "ready"}, {"target", "area", "model", "mask", "stale", "strength", "sky"})
        if "sky" in item:  # w24-edit2: a Restyle's Sky Replacement settings
            sky = _keys(item["sky"], f"{label}.sky", set(), {"shift_edge", "fade_edge", "brightness", "temperature", "lighting_mode", "lighting", "color", "haze", "reflect"})
            for key in sky:
                if key != "lighting_mode":
                    _integer(sky[key], f"{label}.sky.{key}", -100, 100)
        if "strength" in item:  # below 100 % only (charter image, owner decision 1)
            _number(item["strength"], f"{label}.strength", minimum=0.0, maximum=1.0)
        if item["operation"] == "remove_background":
            mask = _keys(item.get("mask"), f"{label}.mask", {"invert", "feather", "density"})
            _bool(mask["invert"], f"{label}.mask.invert")
            _integer(mask["feather"], f"{label}.mask.feather", 0, LAYER_MASK_FEATHER_MAX)
            _integer(mask["density"], f"{label}.mask.density", 0, 100)
        elif "mask" in item:
            _fail(f"{label}.mask is for remove_background layers only")
        if "model" in item and item["model"] != "":
            _pattern(item["model"], f"{label}.model", _FAMILY_RE)
        if "area" in item:
            _text(item["area"], f"{label}.area", 60, path_free=False)
        _integer(item["id"], f"{label}.id", 1, 2**31)
        if item["operation"] not in GENERATIVE_OPERATIONS | {"remove_background", "box"}:  # box: an Image Box edit layer
            _fail(f"{label}.operation is invalid")
        _text(item["prompt"], f"{label}.prompt", 400, path_free=False)
        if "target" in item:
            _text(item["target"], f"{label}.target", 200, path_free=False)
        variants = _integer(item["variants"], f"{label}.variants", 1, 4)
        _integer(item["variant"], f"{label}.variant", 0, variants - 1)
        _bool(item["on"], f"{label}.on")
        _bool(item["ready"], f"{label}.ready")
        if "stale" in item:  # the input below the layer changed since it was generated
            _bool(item["stale"], f"{label}.stale")
    job = generative["job"]
    if job is not None:
        job = _keys(job, "develop.generative.job", {"state", "text"})
        if job["state"] not in {"submitting", "queued", "running", "stopping", "failed"}:
            _fail("develop.generative.job.state is invalid")
        _text(job["text"], "develop.generative.job.text", 240)


def _box_list(value: Any) -> list[Any]:
    if not isinstance(value, list) or len(value) > BOX_MAX:
        _fail(f"boxes must be a list of at most {BOX_MAX}")
    return value


def _box_rect(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 4:
        _fail(f"{label} must be [x, y, width, height] in picture px")
    return [_integer(v, label, 0 if i < 2 else 2, 65536) for i, v in enumerate(value)]


def _box_item(value: Any, label: str, state: bool = False) -> dict[str, Any]:
    """One box {id?, intent, desc?, bbox [x, y, w, h], target_bbox? (move)}; ``state`` adds n and requires id."""
    item = _keys(value, label, {"intent", "bbox"} | ({"id", "n", "desc"} if state else set()), {"target_bbox"} | (set() if state else {"id", "desc"}))
    if item["intent"] not in BOX_INTENTS:
        _fail(f"{label}.intent must be add, change, move, keep or remove")
    out: dict[str, Any] = {"intent": item["intent"], "bbox": _box_rect(item["bbox"], f"{label}.bbox")}
    if "id" in item:
        out["id"] = _pattern(item["id"], f"{label}.id", _BOX_ID_RE)
    if "n" in item:
        out["n"] = _integer(item["n"], f"{label}.n", 1, BOX_MAX)
    if "desc" in item:
        out["desc"] = _text(item["desc"], f"{label}.desc", 160, path_free=False)
    if "target_bbox" in item:
        if item["intent"] != "move":
            _fail(f"{label}.target_bbox is for move only")
        out["target_bbox"] = _box_rect(item["target_bbox"], f"{label}.target_bbox")
    return out


def _box_state(value: Any) -> None:
    """state.edit.boxes: the Box tool's boxes, prompt, model, intents offered and why Apply waits."""
    boxes = _keys(value, "edit.boxes", {"on", "boxes", "selected", "intent", "intents", "prompt", "model", "models", "variants", "seed", "color_match", "run_why"}, {"find", "limit"})
    _bool(boxes["on"], "edit.boxes.on")
    for index, item in enumerate(_box_list(boxes["boxes"])):
        _box_item(item, f"edit.boxes.boxes[{index}]", state=True)
    if boxes["selected"] is not None:
        _pattern(boxes["selected"], "edit.boxes.selected", _BOX_ID_RE)
    if boxes["intent"] not in BOX_INTENTS:
        _fail("edit.boxes.intent is invalid")
    if not isinstance(boxes["intents"], list) or any(i not in BOX_INTENTS for i in boxes["intents"]):
        _fail("edit.boxes.intents is invalid")
    _text(boxes["prompt"], "edit.boxes.prompt", 2000, path_free=False)
    if boxes["model"] is not None and boxes["model"] not in BOX_MODELS:
        _fail("edit.boxes.model is invalid")
    if not isinstance(boxes["models"], list) or len(boxes["models"]) > 8:
        _fail("edit.boxes.models is invalid")
    for index, model in enumerate(boxes["models"]):
        m = _keys(model, f"edit.boxes.models[{index}]", {"id", "label", "native", "move"})
        if m["id"] not in BOX_MODELS:
            _fail(f"edit.boxes.models[{index}].id is invalid")
        _text(m["label"], f"edit.boxes.models[{index}].label", 80, path_free=False)
        _bool(m["native"], f"edit.boxes.models[{index}].native")
        _bool(m["move"], f"edit.boxes.models[{index}].move")
    _integer(boxes["variants"], "edit.boxes.variants", 1, 4)
    _integer(boxes["seed"], "edit.boxes.seed", 0, 2**31 - 1)
    _bool(boxes["color_match"], "edit.boxes.color_match")
    if boxes["run_why"] is not None:
        _text(boxes["run_why"], "edit.boxes.run_why", 240, path_free=False)
    if "limit" in boxes and boxes["limit"] not in BOX_LIMITS:
        _fail("edit.boxes.limit must be off or selection")
    if "find" in boxes:
        f = _keys(boxes["find"], "edit.boxes.find", {"state", "query"})
        if f["state"] not in ("running", "done", "empty", "failed"):
            _fail("edit.boxes.find.state is invalid")
        _text(f["query"], "edit.boxes.find.query", 120, path_free=False)


def _keys(value: Any, label: str, expected: set[str], optional: set[str] = frozenset()) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label} must be an object")
    present = set(value)
    if not expected <= present or present - expected - set(optional):
        _fail(f"{label} fields are invalid")
    return value


def _pattern(value: Any, label: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        _fail(f"{label} is invalid")
    return value


def _window_id(value: Any) -> str:
    return _pattern(str(value or ""), "window_id", _WINDOW_RE)


def _action_id(value: Any) -> str:
    return _pattern(str(value or ""), "action_id", _ACTION_RE)


def _optional_job(value: Any, label: str) -> str | None:
    return None if value is None else _pattern(value, label, _JOB_RE)


def _lora_file(value: Any, label: str) -> str:
    return _pattern(value, label, _LORA_FILE_RE)


# --- snapshot -------------------------------------------------------------

def _validate_tile(value: Any, label: str) -> None:
    if value is None:
        return
    _keys(value, label, {"key", "label"})
    _pattern(value["key"], f"{label}.key", _KEY_RE)
    _text(value["label"], f"{label}.label", 120, path_free=False)


def _validate_render_spec(spec: Any, label: str) -> None:
    _keys(spec, label, {"profile", "steps", "cfg", "sampler", "scheduler", "negative", "seed", "width", "height", "batch_size", "loras"})
    if spec["profile"] not in {"turbo", "raw"}:
        _fail(f"{label}.profile is invalid")
    _integer(spec["steps"], f"{label}.steps", 1, 100)
    _number(spec["cfg"], f"{label}.cfg", minimum=0.1, maximum=20)
    _pattern(spec["sampler"], f"{label}.sampler", _TOKEN_RE)
    _pattern(spec["scheduler"], f"{label}.scheduler", _TOKEN_RE)
    _text(spec["negative"], f"{label}.negative", 8192)
    _integer(spec["seed"], f"{label}.seed", 0, _MAX_SEED)
    _integer(spec["width"], f"{label}.width", 16, 4096)
    _integer(spec["height"], f"{label}.height", 16, 4096)
    _integer(spec["batch_size"], f"{label}.batch_size", 1, 1)
    if not isinstance(spec["loras"], list) or len(spec["loras"]) > LORA_MAX:
        _fail(f"{label}.loras is invalid")
    for index, item in enumerate(spec["loras"]):
        _keys(item, f"{label}.loras[{index}]", {"file", "strength"})
        _lora_file(item["file"], f"{label}.loras[{index}].file")
        _number(item["strength"], f"{label}.loras[{index}].strength", minimum=-4, maximum=4)


def _validate_vary_value(value: Any, label: str) -> None:
    if isinstance(value, str):
        _pattern(value, label, _KEY_RE)
    else:
        _number(value, label)


CANVAS_ZOOM_MIN = 0.05
CANVAS_ZOOM_MAX = 32


def _validate_canvas(canvas: Any) -> None:
    """state.canvas: the view of the picture on the canvas (desktop canvas-nav/canvas-view.js)."""
    _keys(canvas, "canvas", {"mode", "zoom", "viewer", "visible"})
    if canvas["mode"] not in {"fit", "fill", "scale"}:
        _fail("canvas.mode is invalid")
    _number(canvas["zoom"], "canvas.zoom", minimum=0.001, maximum=CANVAS_ZOOM_MAX)
    _bool(canvas["viewer"], "canvas.viewer")
    if not isinstance(canvas["visible"], list) or len(canvas["visible"]) != 4:
        _fail("canvas.visible is invalid")
    for index, value in enumerate(canvas["visible"]):
        _number(value, f"canvas.visible[{index}]", minimum=0, maximum=1)


# Lane compare: Compare (two-up / wipe slider) and Survey (N-up) on the canvas
# (desktop image/compare/use-image-compare.js, use-image-survey.js).
COMPARE_MODES = ("side", "slider")
COMPARE_ZOOM_MIN = 10
COMPARE_ZOOM_MAX = 3200
SURVEY_MIN = 2
SURVEY_MAX = 12
COMPARE_LABEL_MAX = 400


def _diff_fields(value: Any, label: str) -> None:
    if value is None:
        return
    if not isinstance(value, list) or len(value) > 64:
        _fail(f"{label} is invalid")
    for index, field in enumerate(value):
        _text(field, f"{label}[{index}]", 120, allow_empty=False, path_free=False)


def _validate_compare(value: Any) -> None:
    """state.compare: the pair on the canvas, its mode, link, zoom (% of A), wipe and what differs."""
    c = _keys(value, "compare", {"a", "b", "mode", "linked", "zoom", "wipe", "differs", "a_label", "b_label", "loading"})
    _pattern(c["a"], "compare.a", _JOB_RE)
    _pattern(c["b"], "compare.b", _JOB_RE)
    if c["mode"] not in COMPARE_MODES:
        _fail("compare.mode is invalid")
    _bool(c["linked"], "compare.linked")
    _bool(c["loading"], "compare.loading")
    _number(c["zoom"], "compare.zoom", minimum=0, maximum=100_000)
    _number(c["wipe"], "compare.wipe", minimum=0, maximum=1)
    _diff_fields(c["differs"], "compare.differs")
    for key in ("a_label", "b_label"):
        _text(c[key], f"compare.{key}", COMPARE_LABEL_MAX, allow_empty=False, path_free=False)


def _validate_survey(value: Any) -> None:
    """state.survey: the surveyed pictures in order, the active one and what differs."""
    sv = _keys(value, "survey", {"ids", "active", "differs", "labels", "loading"})
    ids = sv["ids"]
    if not isinstance(ids, list) or not SURVEY_MIN <= len(ids) <= SURVEY_MAX:
        _fail("survey.ids is invalid")
    for index, job in enumerate(ids):
        _pattern(job, f"survey.ids[{index}]", _JOB_RE)
    if len(set(ids)) != len(ids):
        _fail("survey.ids must be different")
    if sv["active"] not in ids:
        _fail("survey.active is invalid")
    _diff_fields(sv["differs"], "survey.differs")
    if not isinstance(sv["labels"], list) or len(sv["labels"]) != len(ids):
        _fail("survey.labels is invalid")
    for index, text in enumerate(sv["labels"]):
        _text(text, f"survey.labels[{index}]", COMPARE_LABEL_MAX, allow_empty=False, path_free=False)
    _bool(sv["loading"], "survey.loading")


# Keys & Palette: Image's rows of Hermes' command palette (desktop image/module/image-palette.js).
# Ids are the module's command ids ("generate", "ws-compare", "sel-none", "open-picture").
_COMMAND_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
COMMANDS_MAX = 64


def _command_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) > 64:
        _fail(f"{label} is invalid")
    return _pattern(value, label, _COMMAND_ID_RE)


def _validate_commands(value: Any) -> None:
    """state.commands: the rows Image has on Hermes' palette now; a disabled row says why."""
    if not isinstance(value, list) or len(value) > COMMANDS_MAX:
        _fail("commands is invalid")
    seen: set[str] = set()
    for index, row in enumerate(value):
        _keys(row, f"commands[{index}]", {"id", "label", "enabled", "why"})
        command = _command_id(row["id"], f"commands[{index}].id")
        if command in seen:
            _fail("commands ids must be different")
        seen.add(command)
        _text(row["label"], f"commands[{index}].label", 120, allow_empty=False)
        enabled = _bool(row["enabled"], f"commands[{index}].enabled")
        why = _text(row["why"], f"commands[{index}].why", 240)
        if enabled == bool(why.strip()):
            _fail(f"commands[{index}] says why exactly when it is disabled")


def _validate_state(state: Any) -> dict[str, Any]:
    _keys(state, "state", {
        "schema", "context_key", "owner_scope", "surface_visible", "catalog_ready", "mode", "selections",
        "intent", "prompt", "render_spec", "busy", "generate_enabled", "issue", "job", "result", "vary", "kept",
    }, {"develop", "canvas", "panel", "edit", "compare", "survey", "commands"})
    if state["schema"] != STATE_SCHEMA:
        _fail("state schema is invalid")
    _text(state["context_key"], "context_key", 160, allow_empty=False)
    if not isinstance(state["owner_scope"], str) or not re.fullmatch(r"film_owner:[A-Za-z0-9_\-]{1,80}", state["owner_scope"]):
        _fail("owner_scope is invalid")
    for key in ("surface_visible", "catalog_ready", "busy", "generate_enabled"):
        _bool(state[key], key)
    if state["mode"] not in {"builder", "manual"}:
        _fail("mode is invalid")

    selections = _keys(state["selections"], "selections", set(AXES))
    for axis in AXES[:-1]:
        _validate_tile(selections[axis], f"selections.{axis}")
    if not isinstance(selections["lora"], list) or len(selections["lora"]) > LORA_MAX:
        _fail("selections.lora is invalid")
    for index, item in enumerate(selections["lora"]):
        _keys(item, f"selections.lora[{index}]", {"key", "label", "file", "strength", "effective_strength"})
        _pattern(item["key"], f"selections.lora[{index}].key", _KEY_RE)
        _text(item["label"], f"selections.lora[{index}].label", 120, path_free=False)
        _lora_file(item["file"], f"selections.lora[{index}].file")
        _number(item["strength"], f"selections.lora[{index}].strength", minimum=0, maximum=2)
        _number(item["effective_strength"], f"selections.lora[{index}].effective_strength", minimum=0, maximum=2)

    _text(state["intent"], "intent", 4000, path_free=False)
    prompt = _keys(state["prompt"], "prompt", {"text", "words", "omitted", "warnings", "manual_differs"})
    _text(prompt["text"], "prompt.text", 20000, path_free=False)
    _integer(prompt["words"], "prompt.words", 0, 20000)
    _bool(prompt["manual_differs"], "prompt.manual_differs")
    if not isinstance(prompt["omitted"], list) or len(prompt["omitted"]) > 12:
        _fail("prompt.omitted is invalid")
    for index, item in enumerate(prompt["omitted"]):
        _keys(item, f"prompt.omitted[{index}]", {"axis", "option", "reason"})
        for key in ("axis", "option", "reason"):
            _text(item[key], f"prompt.omitted[{index}].{key}", 120)
    if not isinstance(prompt["warnings"], list) or len(prompt["warnings"]) > 8:
        _fail("prompt.warnings is invalid")
    for index, item in enumerate(prompt["warnings"]):
        _text(item, f"prompt.warnings[{index}]", 400)

    _validate_render_spec(state["render_spec"], "render_spec")
    _text(state["issue"], "issue", 400)

    job = state["job"]
    if job is not None:
        _keys(job, "job", {"job_id", "state", "elapsed_seconds", "error"})
        _optional_job(job["job_id"], "job.job_id")
        if job["state"] not in _JOB_STATES:
            _fail("job.state is invalid")
        if job["elapsed_seconds"] is not None:
            _number(job["elapsed_seconds"], "job.elapsed_seconds", minimum=0)
        _optional_text(job["error"], "job.error", 400)

    result = state["result"]
    if result is not None:
        _keys(result, "result", {"job_id", "render_spec", "has_picture", "kept_as"})
        _pattern(result["job_id"], "result.job_id", _JOB_RE)
        if result["render_spec"] is not None:
            _validate_render_spec(result["render_spec"], "result.render_spec")
        _bool(result["has_picture"], "result.has_picture")
        None if result["kept_as"] is None else _text(result["kept_as"], "result.kept_as", 120, path_free=False)

    vary = _keys(state["vary"], "vary", {"running", "stopping", "variable", "values", "cells"})
    _bool(vary["running"], "vary.running")
    _bool(vary["stopping"], "vary.stopping")
    if vary["variable"] is not None:
        _keys(vary["variable"], "vary.variable", {"id", "label"})
        _pattern(vary["variable"]["id"], "vary.variable.id", _VARY_VARIABLE_RE)
        _text(vary["variable"]["label"], "vary.variable.label", 120, path_free=False)
    if not isinstance(vary["values"], list) or len(vary["values"]) > 64:
        _fail("vary.values is invalid")
    for index, value in enumerate(vary["values"]):
        _validate_vary_value(value, f"vary.values[{index}]")
    if not isinstance(vary["cells"], list) or len(vary["cells"]) > VARY_MAX_CELLS:
        _fail("vary.cells is invalid")
    for index, cell in enumerate(vary["cells"]):
        _keys(cell, f"vary.cells[{index}]", {"label", "value", "state", "job_id", "kept_as", "message"})
        _text(cell["label"], f"vary.cells[{index}].label", 160, path_free=False)
        _validate_vary_value(cell["value"], f"vary.cells[{index}].value")
        if cell["state"] not in _CELL_STATES:
            _fail(f"vary.cells[{index}].state is invalid")
        _optional_job(cell["job_id"], f"vary.cells[{index}].job_id")
        None if cell["kept_as"] is None else _text(cell["kept_as"], f"vary.cells[{index}].kept_as", 120, path_free=False)
        _optional_text(cell["message"], f"vary.cells[{index}].message", 400)

    canvas = state.get("canvas")
    if canvas is not None:
        _validate_canvas(canvas)
    if state.get("compare") is not None:
        _validate_compare(state["compare"])
    if state.get("survey") is not None:
        _validate_survey(state["survey"])
    if state.get("commands") is not None:
        _validate_commands(state["commands"])
    develop = state.get("develop")
    if develop is not None:
        _keys(develop, "develop", {"light", "neutral"}, {"color", "mixer", "grading", "curve", "detail", "effects", "history", "mode", "before", "histogram", "off", "clipping", "masks", "mask_selected", "mask_component", "mask_pins", "mask_overlay", "generative", "crop", "outpaint", "white_balance", "presets"})
        if "mode" in develop and develop["mode"] not in EDIT_MODES:
            _fail("develop.mode is invalid")
        _bool(develop["neutral"], "develop.neutral")
        if "before" in develop:
            _bool(develop["before"], "develop.before")
        if "clipping" in develop:
            # The clipping view drawn over the picture (J / the histogram marks); never part of the picture.
            clipping = _keys(develop["clipping"], "develop.clipping", {"shadows", "highlights"})
            _bool(clipping["shadows"], "develop.clipping.shadows")
            _bool(clipping["highlights"], "develop.clipping.highlights")
        if "white_balance" in develop:
            # Color > WB: As Shot (0/0), Auto (grey world) or custom Temp/Tint; picking while the picker is on.
            wb = _keys(develop["white_balance"], "develop.white_balance", {"mode", "picking"})
            if wb["mode"] not in WB_STATE_MODES:
                _fail("develop.white_balance.mode is invalid")
            _bool(wb["picking"], "develop.white_balance.picking")
        if "presets" in develop:
            # Adjust > Presets: the saved names, the one the picture matches, whether settings are copied.
            pr = _keys(develop["presets"], "develop.presets", {"names", "current", "copied"}, {"clipboard", "preview"})
            if "preview" in pr:
                # The preset shown on the picture while hovered (temporary, not the document).
                _preset_name(pr["preview"], "develop.presets.preview")
            if "clipboard" in pr:
                # The groups the copied settings hold (Copy… chooses them), in panel order.
                _develop_group_list(pr["clipboard"], "develop.presets.clipboard")
            if not isinstance(pr["names"], list) or len(pr["names"]) > DEVELOP_PRESETS_MAX:
                _fail("develop.presets.names is invalid")
            for name in pr["names"]:
                _preset_name(name, "develop.presets.names")
            if pr["current"] is not None:
                _preset_name(pr["current"], "develop.presets.current")
            _bool(pr["copied"], "develop.presets.copied")
        if "masks" in develop:
            _masks(develop["masks"], "develop.masks", state=True)
        if "generative" in develop:
            _generative_state(develop["generative"])
        if "crop" in develop:
            _crop(develop["crop"], "develop.crop", state=True)
        if "outpaint" in develop:
            _outpaint_state(develop["outpaint"])
        if "mask_selected" in develop and develop["mask_selected"] is not None:
            _integer(develop["mask_selected"], "develop.mask_selected", 1, 2**31)
        if "mask_component" in develop:
            _integer(develop["mask_component"], "develop.mask_component", 0, DEVELOP_MASK_COMPONENTS_MAX - 1)
        if "mask_pins" in develop:
            _bool(develop["mask_pins"], "develop.mask_pins")
        if "mask_overlay" in develop:
            _mask_overlay(develop["mask_overlay"], "develop.mask_overlay", state=True)
        if "off" in develop:
            # Groups switched off (values kept, drawn neutral), in panel order.
            off = develop["off"]
            if not isinstance(off, list) or any(g not in DEVELOP_GROUPS for g in off) or off != [g for g in DEVELOP_GROUPS if g in off]:
                _fail("develop.off is invalid")
        if "histogram" in develop:
            # The shown picture's mean level (0..255) and the share of pixels with a channel at 0 / 255 (percent).
            histogram = _keys(develop["histogram"], "develop.histogram", {"mean", "shadows_clipped", "highlights_clipped"})
            _number(histogram["mean"], "develop.histogram.mean", minimum=0, maximum=255)
            _number(histogram["shadows_clipped"], "develop.histogram.shadows_clipped", minimum=0, maximum=100)
            _number(histogram["highlights_clipped"], "develop.histogram.highlights_clipped", minimum=0, maximum=100)
        if "history" in develop:
            history = _keys(develop["history"], "develop.history", {"at", "first", "steps"})
            steps = history["steps"]
            if not isinstance(steps, list) or not 1 <= len(steps) <= DEVELOP_HISTORY_ROWS:
                _fail("develop.history.steps is invalid")
            first = _integer(history["first"], "develop.history.first", 0, 2**31)
            _integer(history["at"], "develop.history.at", first, first + len(steps) - 1)
            for index, step in enumerate(steps):
                _keys(step, f"develop.history.steps[{index}]", {"label", "kind"})
                _text(step["label"], f"develop.history.steps[{index}].label", 80, allow_empty=True, path_free=False)
                if step["kind"] not in DEVELOP_HISTORY_KINDS:
                    _fail(f"develop.history.steps[{index}].kind is invalid")
        for group, table in DEVELOP_GROUPS.items():
            if group not in develop:
                continue
            points = DEVELOP_POINTS.get(group, ())
            values = _keys(develop[group], f"develop.{group}", set(table), set(points))
            for key, (low, high) in table.items():
                _number(values[key], f"develop.{group}.{key}", minimum=low, maximum=high)
            for key in points:
                if key in values:
                    _points(values[key], f"develop.{group}.{key}")

    panel = state.get("panel")
    if panel is not None:
        _keys(panel, "panel", {"seed_mode"}, {"size"})
        if panel["seed_mode"] not in SEED_MODES:
            _fail("panel.seed_mode is invalid")
        if "size" in panel:
            size = _keys(panel["size"], "panel.size", {"format", "width", "height", "mp", "batch", "aspect_lock"})
            _pattern(size["format"], "panel.size.format", _FORMAT_RE)
            for key in ("width", "height"):
                _integer(size[key], f"panel.size.{key}", 16, 8192)
            _number(size["mp"], "panel.size.mp", minimum=0, maximum=64)
            _integer(size["batch"], "panel.size.batch", 1, 64)
            _bool(size["aspect_lock"], "panel.size.aspect_lock")

    edit = state.get("edit")
    if edit is not None:
        # The Edit selection: whether there is one, its box and area in picture px, soft edge, the rail tool.
        _keys(edit, "edit", {"selection"}, {"boxes"})
        if "boxes" in edit:
            _box_state(edit["boxes"])
        sel = _keys(edit["selection"], "edit.selection", {"has", "bounds", "area_px", "feathered", "tool"}, {"refining", "mask_crc32"})
        _bool(sel["has"], "edit.selection.has")
        if sel["bounds"] is not None:
            box = _keys(sel["bounds"], "edit.selection.bounds", {"x", "y", "w", "h"})
            for key in ("x", "y"):
                _integer(box[key], f"edit.selection.bounds.{key}", 0, 65536)
            for key in ("w", "h"):
                _integer(box[key], f"edit.selection.bounds.{key}", 1, 65536)
        _integer(sel["area_px"], "edit.selection.area_px", 0, 2**32)
        _bool(sel["feathered"], "edit.selection.feathered")
        if sel["tool"] not in SELECTION_TOOLS:
            _fail("edit.selection.tool is invalid")
        if "refining" in sel:
            _bool(sel["refining"], "edit.selection.refining")
        if "mask_crc32" in sel and sel["mask_crc32"] is not None:
            _integer(sel["mask_crc32"], "edit.selection.mask_crc32", 0, 2**32 - 1)

    if not isinstance(state["kept"], list) or len(state["kept"]) > KEPT_LIMIT:
        _fail("kept is invalid")
    for index, row in enumerate(state["kept"]):
        _keys(row, f"kept[{index}]", {"revision_id", "label", "saved_at", "job_id"})
        _pattern(row["revision_id"], f"kept[{index}].revision_id", _REVISION_RE)
        _text(row["label"], f"kept[{index}].label", 120, path_free=False)
        _text(row["saved_at"], f"kept[{index}].saved_at", 40)
        _pattern(row["job_id"], f"kept[{index}].job_id", _JOB_RE)
    return json.loads(json.dumps(state, ensure_ascii=False, allow_nan=False))


# --- actions --------------------------------------------------------------

def validate_action_payload(action: Any, payload: Any) -> dict[str, Any]:
    """Return a canonical payload or raise; unknown actions and fields fail."""
    if action not in ALLOWED_ACTIONS:
        _fail("action is unsupported")
    if not isinstance(payload, dict):
        _fail("action payload must be an object")
    if action == "select_tile":
        _keys(payload, "select_tile payload", {"axis", "key"})
        if payload["axis"] not in AXES:
            _fail("select_tile axis is invalid")
        return {"axis": payload["axis"], "key": _pattern(payload["key"], "select_tile key", _KEY_RE)}
    if action == "clear_axis":
        _keys(payload, "clear_axis payload", {"axis"})
        if payload["axis"] not in AXES:
            _fail("clear_axis axis is invalid")
        return {"axis": payload["axis"]}
    if action == "set_intent":
        _keys(payload, "set_intent payload", {"text"})
        return {"text": _text(payload["text"], "set_intent text", 4000, path_free=False)}
    if action == "set_manual_prompt":
        _keys(payload, "set_manual_prompt payload", {"text"})
        return {"text": _text(payload["text"], "set_manual_prompt text", 20000, allow_empty=False, path_free=False)}
    if action == "set_mode":
        _keys(payload, "set_mode payload", {"mode"})
        if payload["mode"] not in {"builder", "manual"}:
            _fail("set_mode mode is invalid")
        return {"mode": payload["mode"]}
    if action == "set_sampling":
        fields = {"profile", "steps", "cfg", "sampler", "scheduler", "negative"}
        _keys(payload, "set_sampling payload", set(), fields)
        if not payload:
            _fail("set_sampling needs at least one field")
        result: dict[str, Any] = {}
        if "profile" in payload:
            if payload["profile"] not in {"turbo", "raw"}:
                _fail("set_sampling profile is invalid")
            result["profile"] = payload["profile"]
        if "steps" in payload:
            result["steps"] = _integer(payload["steps"], "set_sampling steps", 1, 100)
        if "cfg" in payload:
            result["cfg"] = _number(payload["cfg"], "set_sampling cfg", minimum=0.1, maximum=20)
        for key in ("sampler", "scheduler"):
            if key in payload:
                result[key] = _pattern(payload[key], f"set_sampling {key}", _TOKEN_RE)
        if "negative" in payload:
            result["negative"] = _text(payload["negative"], "set_sampling negative", 8192, path_free=False)
        return result
    if action == "set_lora_strength":
        _keys(payload, "set_lora_strength payload", {"file", "strength"})
        return {
            "file": _lora_file(payload["file"], "set_lora_strength file"),
            "strength": _number(payload["strength"], "set_lora_strength strength", minimum=0, maximum=2),
        }
    if action == "set_seed":
        _keys(payload, "set_seed payload", {"seed"})
        return {"seed": _integer(payload["seed"], "set_seed seed", 0, 2**53 - 1)}
    if action == "set_seed_mode":
        _keys(payload, "set_seed_mode payload", {"mode"})
        if payload["mode"] not in SEED_MODES:
            _fail("set_seed_mode mode is invalid")
        return {"mode": payload["mode"]}
    if action == "set_size":
        fields = {"format", "width", "height", "mp", "batch", "swap", "aspect_lock"}
        _keys(payload, "set_size payload", set(), fields)
        if not payload:
            _fail("set_size needs at least one field")
        result = {}
        if "format" in payload:
            result["format"] = _pattern(payload["format"], "set_size format", _FORMAT_RE)
        for key in ("width", "height"):
            if key in payload:
                result[key] = _integer(payload[key], f"set_size {key}", 16, 8192)
        if "mp" in payload:
            result["mp"] = _number(payload["mp"], "set_size mp", minimum=0.05, maximum=16)
        if "batch" in payload:
            result["batch"] = _integer(payload["batch"], "set_size batch", 1, 64)
        for key in ("swap", "aspect_lock"):
            if key in payload:
                result[key] = _bool(payload[key], f"set_size {key}")
        return result
    if action == "keep":
        _keys(payload, "keep payload", {"name"}, {"job_id"})
        result = {"name": _text(payload["name"], "keep name", 120, allow_empty=False, path_free=False).strip()}
        if payload.get("job_id") is not None:
            result["job_id"] = _pattern(payload["job_id"], "keep job_id", _JOB_RE)
        return result
    if action == "vary":
        _keys(payload, "vary payload", {"variable", "values"})
        variable = _pattern(payload["variable"], "vary variable", _VARY_VARIABLE_RE)
        values = payload["values"]
        if not isinstance(values, list) or not 2 <= len(values) <= VARY_MAX_CELLS:
            _fail(f"vary needs 2 to {VARY_MAX_CELLS} values")
        choice = variable in {"sampler", "scheduler", "profile"} or variable.startswith("axis:")
        canonical: list[Any] = []
        for index, value in enumerate(values):
            if choice:
                canonical.append(_pattern(value, f"vary values[{index}]", _KEY_RE))
            elif variable in {"seed", "steps"}:
                maximum = 2**53 - 1 if variable == "seed" else 100
                canonical.append(_integer(value, f"vary values[{index}]", 0 if variable == "seed" else 1, maximum))
            elif variable == "cfg":
                canonical.append(_number(value, f"vary values[{index}]", minimum=0.1, maximum=20))
            else:
                canonical.append(_number(value, f"vary values[{index}]", minimum=0, maximum=2))
        if len(set(map(str, canonical))) != len(canonical):
            _fail("vary values must be different")
        return {"variable": variable, "values": canonical}
    if action == "restore":
        _keys(payload, "restore payload", {"revision_id"})
        return {"revision_id": _pattern(payload["revision_id"], "restore revision_id", _REVISION_RE)}
    if action == "set_canvas":
        _keys(payload, "set_canvas payload", set(), {"zoom", "center", "viewer"})
        if not payload:
            _fail("set_canvas needs zoom, center or viewer")
        out: dict[str, Any] = {}
        if "zoom" in payload:
            zoom = payload["zoom"]
            out["zoom"] = zoom if isinstance(zoom, str) and zoom in {"fit", "fill"} else _number(zoom, "set_canvas zoom", minimum=CANVAS_ZOOM_MIN, maximum=CANVAS_ZOOM_MAX)
        if "center" in payload:
            center = payload["center"]
            if not isinstance(center, list) or len(center) != 2:
                _fail("set_canvas center must be [x, y]")
            out["center"] = [_number(value, f"set_canvas center[{index}]", minimum=0, maximum=1) for index, value in enumerate(center)]
            if out.get("zoom") in {"fit", "fill"}:
                _fail("set_canvas center needs a zoom number, not fit or fill")
        if "viewer" in payload:
            out["viewer"] = _bool(payload["viewer"], "set_canvas viewer")
        return out
    if action == "compare":
        # Open A and B (side by side or slider), change the open Compare, swap, or close it.
        _keys(payload, "compare payload", set(), {"a", "b", "mode", "swap", "linked", "wipe", "zoom", "close"})
        if not payload:
            _fail("compare needs a and b, or a change to the open Compare")
        if "close" in payload:
            if len(payload) != 1 or payload["close"] is not True:
                _fail("compare close takes no other field and must be true")
            return {"close": True}
        out: dict[str, Any] = {}
        if ("a" in payload) != ("b" in payload):
            _fail("compare needs both a and b")
        if "a" in payload:
            out["a"] = _pattern(payload["a"], "compare a", _JOB_RE)
            out["b"] = _pattern(payload["b"], "compare b", _JOB_RE)
            if out["a"] == out["b"]:
                _fail("compare a and b must be different pictures")
        if "mode" in payload:
            if payload["mode"] not in COMPARE_MODES:
                _fail("compare mode must be side or slider")
            out["mode"] = payload["mode"]
        for key in ("swap", "linked"):
            if key in payload:
                out[key] = _bool(payload[key], f"compare {key}")
        if "wipe" in payload:
            out["wipe"] = _number(payload["wipe"], "compare wipe", minimum=0, maximum=1)
        if "zoom" in payload:
            zoom = payload["zoom"]
            out["zoom"] = "fit" if zoom == "fit" else _number(zoom, "compare zoom", minimum=COMPARE_ZOOM_MIN, maximum=COMPARE_ZOOM_MAX)
        return out
    if action == "survey":
        # Open 2..12 pictures, or act on the open Survey: one field at a time.
        _keys(payload, "survey payload", set(), {"ids", "active", "remove", "pick", "close"})
        if len(payload) != 1:
            _fail("survey takes exactly one of ids, active, remove, pick or close")
        if "close" in payload:
            if payload["close"] is not True:
                _fail("survey close must be true")
            return {"close": True}
        if "ids" in payload:
            ids = payload["ids"]
            if not isinstance(ids, list) or not SURVEY_MIN <= len(ids) <= SURVEY_MAX:
                _fail(f"survey ids must list {SURVEY_MIN} to {SURVEY_MAX} pictures")
            out_ids = [_pattern(job, f"survey ids[{index}]", _JOB_RE) for index, job in enumerate(ids)]
            if len(set(out_ids)) != len(out_ids):
                _fail("survey ids must be different")
            return {"ids": out_ids}
        key = next(iter(payload))
        return {key: _pattern(payload[key], f"survey {key}", _JOB_RE)}
    if action == "open_result":
        _keys(payload, "open_result payload", {"job_id"})
        return {"job_id": _pattern(payload["job_id"], "open_result job_id", _JOB_RE)}
    if action == "set_develop":
        _keys(payload, "set_develop payload", set(), set(DEVELOP_GROUPS))
        if not payload:
            _fail("set_develop needs light, color, mixer, grading, curve, detail or effects values")
        out: dict[str, dict[str, float]] = {}
        for group, table in DEVELOP_GROUPS.items():
            if group not in payload:
                continue
            points = DEVELOP_POINTS.get(group, ())
            values = _keys(payload[group], f"set_develop {group}", set(), set(table) | set(points))
            if not values:
                _fail(f"set_develop needs at least one {group} value")
            out[group] = {
                key: _points(value, f"set_develop {group}.{key}") if key in points
                else _number(value, f"set_develop {group}.{key}", minimum=table[key][0], maximum=table[key][1])
                for key, value in values.items()
            }
        return out
    if action == "reset_develop":
        _keys(payload, "reset_develop payload", set(), {"group"})
        if "group" not in payload:
            return {}
        if payload["group"] not in DEVELOP_GROUPS:
            _fail("reset_develop group must be light, color, mixer, grading, curve, detail or effects")
        return {"group": payload["group"]}
    if action == "set_develop_group":
        _keys(payload, "set_develop_group payload", {"group", "on"})
        if payload["group"] not in DEVELOP_GROUPS:
            _fail("set_develop_group group must be light, color, mixer, grading, curve, detail or effects")
        return {"group": payload["group"], "on": _bool(payload["on"], "set_develop_group on")}
    if action == "set_develop_clipping":
        _keys(payload, "set_develop_clipping payload", set(), {"shadows", "highlights"})
        if not payload:
            _fail("set_develop_clipping needs shadows or highlights")
        return {k: _bool(v, f"set_develop_clipping {k}") for k, v in payload.items()}
    if action == "set_develop_preset":
        # copy / paste the settings, or save / apply / delete a named preset.
        _keys(payload, "set_develop_preset payload", {"op"}, {"name", "groups", "names", "file"})
        op = payload["op"]
        if op not in DEVELOP_PRESET_OPS:
            _fail("set_develop_preset op must be copy, paste, save, apply, delete, preview, preview_end, export or import")
        if ("names" in payload) != (op == "export" and "names" in payload) or ("file" in payload) != (op == "import"):
            _fail("set_develop_preset names belong to export, file to import (and import needs file)")
        if op == "export":
            if "name" in payload or "groups" in payload:
                _fail("set_develop_preset export takes names only")
            if "names" not in payload:
                return {"op": op}
            names = payload["names"]
            if not isinstance(names, list) or not 1 <= len(names) <= DEVELOP_PRESETS_MAX or len(set(map(str, names))) != len(names):
                _fail("set_develop_preset names must list 1-60 different presets")
            return {"op": op, "names": [_preset_name(n, "set_develop_preset names") for n in names]}
        if op == "import":
            if set(payload) != {"op", "file"}:
                _fail("set_develop_preset import takes file only")
            # The renderer checks the document itself (schema, version, every value); here: bounded text.
            return {"op": op, "file": _text(payload["file"], "set_develop_preset file", DEVELOP_PRESET_FILE_CHARS, allow_empty=False, path_free=False)}
        if "groups" in payload and op != "copy":
            _fail("set_develop_preset groups belong to copy")
        if op in ("copy", "paste", "preview_end"):
            if "name" in payload:
                _fail("set_develop_preset copy, paste and preview_end take no name")
            if "groups" in payload:
                return {"op": op, "groups": _develop_group_list(payload["groups"], "set_develop_preset groups")}
            return {"op": op}
        if "name" not in payload:
            _fail("set_develop_preset save, apply, delete and preview need a name")
        return {"op": op, "name": _preset_name(payload["name"], "set_develop_preset name")}
    if action == "set_develop_white_balance":
        # As Shot / Auto, the picker on one spot or an area (0..1 of the picture), or the picker on / off.
        _keys(payload, "set_develop_white_balance payload", set(), {"mode", "x", "y", "w", "h", "picking"})
        keys = set(payload)
        if keys == {"mode"}:
            if payload["mode"] not in WB_MODES:
                _fail("set_develop_white_balance mode must be as_shot, auto, daylight, cloudy, shade or flash")
            return {"mode": payload["mode"]}
        if keys == {"picking"}:
            return {"picking": _bool(payload["picking"], "set_develop_white_balance picking")}
        if keys in ({"x", "y"}, {"x", "y", "w", "h"}):
            out = {k: _number(payload[k], f"set_develop_white_balance {k}", minimum=0.0, maximum=1.0) for k in sorted(keys)}
            if "w" in out and (out["w"] <= 0 or out["h"] <= 0 or out["x"] + out["w"] > 1 + 1e-9 or out["y"] + out["h"] > 1 + 1e-9):
                _fail("set_develop_white_balance area must have a size and lie inside the picture")
            return out
        _fail("set_develop_white_balance needs mode, picking, or x and y (with w and h for an area)")
    if action == "auto_tone":
        # Light > Auto: all six Light sliders, or only the named ones (Shift+double-click on a slider).
        _keys(payload, "auto_tone payload", set(), {"sliders"})
        if "sliders" not in payload:
            return {}
        sliders = payload["sliders"]
        if (not isinstance(sliders, list) or not sliders or len(set(map(str, sliders))) != len(sliders)
                or any(not isinstance(s, str) or s not in DEVELOP_LIGHT for s in sliders)):
            _fail("auto_tone sliders must be distinct names of exposure, contrast, highlights, shadows, whites, blacks")
        return {"sliders": [s for s in DEVELOP_LIGHT if s in sliders]}
    if action == "set_develop_masks":
        _keys(payload, "set_develop_masks payload", set(), {"masks", "select", "pins", "overlay"})
        if not payload:
            _fail("set_develop_masks needs masks, select, pins or overlay")
        out: dict[str, Any] = {}
        if "masks" in payload:
            out["masks"] = _masks(payload["masks"], "set_develop_masks masks", state=False)
        if "select" in payload:
            select = _keys(payload["select"], "set_develop_masks select", {"id"}, {"component"})
            out["select"] = {"id": _mask_id(select["id"], "set_develop_masks select.id")}
            if "component" in select:
                out["select"]["component"] = _integer(select["component"], "set_develop_masks select.component", 0, DEVELOP_MASK_COMPONENTS_MAX - 1)
        if "pins" in payload:
            out["pins"] = _bool(payload["pins"], "set_develop_masks pins")
        if "overlay" in payload:
            out["overlay"] = _mask_overlay(payload["overlay"], "set_develop_masks overlay", state=False)
        return out
    if action == "generate_edit":
        # One generative edit inside the current selection or a mask (the pane submits /image/edit itself).
        _keys(payload, "generate_edit payload", {"operation"}, {"prompt", "target", "variants", "seed", "mask", "model", "live", "strength", "sky"})
        if payload["operation"] not in GENERATIVE_OPERATIONS:
            _fail("generate_edit operation must be fill, replace, remove, restyle or relight")
        out = {"operation": payload["operation"]}
        for key, limit in (("prompt", 400), ("target", 200)):
            if key in payload:
                out[key] = _text(payload[key], f"generate_edit {key}", limit, path_free=False)
        # Relight without a prompt takes the pane's light (set_edit_light); a given prompt must say something.
        relight_words = payload["operation"] == "relight" and "prompt" not in payload
        if payload["operation"] != "remove" and not relight_words and not str(out.get("prompt") or "").strip():
            _fail("generate_edit needs a prompt for fill, replace and restyle (and a non-empty one for relight)")
        if "live" in payload:
            if payload["operation"] != "relight":
                _fail("generate_edit live is for relight only")
            out["live"] = _bool(payload["live"], "generate_edit live")
        if "variants" in payload:
            out["variants"] = _integer(payload["variants"], "generate_edit variants", 1, 4)
        if "seed" in payload:
            out["seed"] = _integer(payload["seed"], "generate_edit seed", 0, 2**31 - 1)
        if "mask" in payload:
            out["mask"] = _mask_id(payload["mask"], "generate_edit mask", all_ok=True)
        if "model" in payload:
            if not isinstance(payload["model"], str) or payload["model"] not in EDIT_MODELS:
                _fail("generate_edit model must be qwen-image-2.1, flux3-klein, flux2-klein or krea2")
            out["model"] = payload["model"]
        if "strength" in payload:  # 0..1 (charter image, owner decision 1); the pane refuses it for a model without Strength
            if payload["operation"] == "relight":
                _fail("generate_edit strength is not for relight")
            out["strength"] = _number(payload["strength"], "generate_edit strength", minimum=0.0, maximum=1.0)
        if "sky" in payload:  # w24-edit2: Restyle's Sky Replacement settings (Photoshop's dialog; omitted keys = its defaults)
            if payload["operation"] != "restyle":
                _fail("generate_edit sky is for restyle only")
            sky = _keys(payload["sky"], "generate_edit sky", set(), {"shift_edge", "fade_edge", "brightness", "temperature", "lighting_mode", "lighting", "color", "haze", "reflect"})
            for key, low, high in (("shift_edge", -20, 20), ("fade_edge", 0, 20), ("brightness", -100, 100), ("temperature", -100, 100), ("lighting", 0, 100), ("color", 0, 100), ("haze", 0, 100), ("reflect", 0, 100)):
                if key in sky:
                    _integer(sky[key], f"generate_edit sky.{key}", low, high)
            if "lighting_mode" in sky and sky["lighting_mode"] not in ("multiply", "screen"):
                _fail("generate_edit sky.lighting_mode must be multiply or screen")
            out["sky"] = dict(sky)
        return out
    if action == "set_edit_light":
        # The Relight light (opens Generative > Relight); x/y place it on the direction pad.
        # Stage 1b: light 1..3 addresses (and selects) a light, add adds one, remove removes the addressed one, on switches it.
        payload = _keys(payload, "set_edit_light payload", set(), {"tone", "from", "x", "y", "behind", *LIGHT_CONTROLS, *LIGHT_SCENE_EXTRA, "live", "light", "add", "remove", "on", "color", "kelvin"})
        if not payload:
            _fail("set_edit_light needs at least one field")
        out = {}
        if "light" in payload:
            out["light"] = _integer(payload["light"], "set_edit_light light", 1, LIGHTS_MAX)
        for key in ("add", "remove"):
            if key in payload:
                if payload[key] is not True:
                    _fail(f"set_edit_light {key} must be true")
                out[key] = True
        if out.get("add") and (out.get("remove") or "light" in out):
            _fail("set_edit_light add takes no light and no remove")
        if out.get("remove") and set(payload) - {"remove", "light"}:
            _fail("set_edit_light remove takes only light")
        if "tone" in payload:
            if payload["tone"] not in LIGHT_TONES:
                _fail("set_edit_light tone must be warm golden, neutral white, cool blue, soft daylight or dim, moody")
            out["tone"] = payload["tone"]
        # The free colour: one of tone, color (#rrggbb) or kelvin (1700..12000).
        if sum(key in payload for key in ("tone", "color", "kelvin")) > 1:
            _fail("set_edit_light takes one of tone, color or kelvin")
        if "color" in payload:
            out["color"] = _pattern(payload["color"], "set_edit_light color", LIGHT_HEX).lower()
        if "kelvin" in payload:
            out["kelvin"] = _integer(payload["kelvin"], "set_edit_light kelvin", *LIGHT_KELVIN)
        if "from" in payload:
            if payload["from"] not in LIGHT_FROM:
                _fail("set_edit_light from must be from the left, from the right, from the front, from behind, from above or from below")
            out["from"] = payload["from"]
        for key in ("x", "y"):
            if key in payload:
                out[key] = _number(payload[key], f"set_edit_light {key}", minimum=-1.0, maximum=1.0)
        for key, (low, high) in {**LIGHT_CONTROLS, **LIGHT_SCENE_EXTRA}.items():
            if key in payload:
                out[key] = _number(payload[key], f"set_edit_light {key}", minimum=low, maximum=high)
        for key in ("behind", "live", "on"):
            if key in payload:
                out[key] = _bool(payload[key], f"set_edit_light {key}")
        return out
    if action == "outpaint_edit":
        # Outpaint: the picture on a larger canvas; the pane submits /image/edit operation outpaint itself.
        _keys(payload, "outpaint_edit payload", set(), {*OUTPAINT_SIDES, "prompt", "variants", "seed", "model"})
        out = {}
        for side in OUTPAINT_SIDES:
            if side in payload:
                out[side] = _integer(payload[side], f"outpaint_edit {side}", 0, 32_768)
        if "prompt" in payload:
            out["prompt"] = _text(payload["prompt"], "outpaint_edit prompt", 400, path_free=False)
        if "variants" in payload:
            out["variants"] = _integer(payload["variants"], "outpaint_edit variants", 1, 4)
        if "seed" in payload:
            out["seed"] = _integer(payload["seed"], "outpaint_edit seed", 0, 2**31 - 1)
        if "model" in payload:
            out["model"] = _pattern(payload["model"], "outpaint_edit model", _FAMILY_RE)
        return out
    if action == "accept_outpaint":
        # The picked (or named) Outpaint variant is kept as a new version and opens.
        _keys(payload, "accept_outpaint payload", set(), {"variant", "name"})
        out = {}
        if "variant" in payload:
            out["variant"] = _integer(payload["variant"], "accept_outpaint variant", 0, 3)
        if "name" in payload:
            out["name"] = _text(payload["name"], "accept_outpaint name", 160, allow_empty=False, path_free=False).strip()
        return out
    if action == "accept_edit":
        # Accept as version: the picture as Edit shows it is kept as a new version and opens.
        _keys(payload, "accept_edit payload", set(), {"name"})
        return {"name": _text(payload["name"], "accept_edit name", 160, allow_empty=False, path_free=False).strip()} if "name" in payload else {}
    if action == "set_edit_crop":
        # Edit > Crop: a rectangle, a fine angle (Straighten), a straighten line, a quarter turn, a mirror; at least one.
        _keys(payload, "set_edit_crop payload", set(), {"crop", "angle", "straighten", "rotate", "flip", "size"})
        if not payload:
            _fail("set_edit_crop needs crop, angle, straighten, rotate, flip or size")
        out: dict[str, Any] = {}
        if "crop" in payload:
            out["crop"] = None if payload["crop"] is None else _crop(payload["crop"], "set_edit_crop crop", state=False)
        if "angle" in payload:
            out["angle"] = _number(payload["angle"], "set_edit_crop angle", minimum=-45, maximum=45)
        if "straighten" in payload:
            line = _keys(payload["straighten"], "set_edit_crop straighten", {"x0", "y0", "x1", "y1"})
            out["straighten"] = {key: _number(line[key], f"set_edit_crop straighten.{key}", minimum=0, maximum=1) for key in ("x0", "y0", "x1", "y1")}
            if abs(out["straighten"]["x1"] - out["straighten"]["x0"]) + abs(out["straighten"]["y1"] - out["straighten"]["y0"]) < 0.01:
                _fail("set_edit_crop straighten needs a line, not a point")
        if "rotate" in payload:
            if payload["rotate"] not in ("left", "right"):
                _fail("set_edit_crop rotate must be left or right")
            out["rotate"] = payload["rotate"]
        if "flip" in payload:
            if payload["flip"] not in ("horizontal", "vertical"):
                _fail("set_edit_crop flip must be horizontal or vertical")
            out["flip"] = payload["flip"]
        if "size" in payload:
            size = _keys(payload["size"], "set_edit_crop size", {"w", "h"})
            out["size"] = {key: _integer(size[key], f"set_edit_crop size.{key}", 1, 65536) for key in ("w", "h")}
        return out
    if action == "remove_background":
        # Edit > Generative > Remove background: the subject mask becomes a layer mask (no generation).
        _keys(payload, "remove_background payload", set())
        return {}
    if action == "regenerate_layer":
        # Stale marker: re-run a generative layer's stored edit from what lies below it now.
        _keys(payload, "regenerate_layer payload", {"layer"})
        return {"layer": _integer(payload["layer"], "regenerate_layer layer", 1, 2**31)}
    if action == "set_layer_mask":
        # The layer mask of a Remove background layer: invert, feather (px) and density (%).
        _keys(payload, "set_layer_mask payload", {"layer"}, {"invert", "feather", "density"})
        out = {"layer": _integer(payload["layer"], "set_layer_mask layer", 1, 2**31)}
        if "invert" in payload:
            out["invert"] = _bool(payload["invert"], "set_layer_mask invert")
        if "feather" in payload:
            out["feather"] = _integer(payload["feather"], "set_layer_mask feather", 0, LAYER_MASK_FEATHER_MAX)
        if "density" in payload:
            out["density"] = _integer(payload["density"], "set_layer_mask density", 0, 100)
        if len(out) == 1:
            _fail("set_layer_mask needs invert, feather or density")
        return out
    if action == "set_edit_layer":
        _keys(payload, "set_edit_layer payload", {"layer"}, {"variant", "on", "delete"})
        out = {"layer": _integer(payload["layer"], "set_edit_layer layer", 1, 2**31)}
        if "variant" in payload:
            out["variant"] = _integer(payload["variant"], "set_edit_layer variant", 0, 3)
        if "on" in payload:
            out["on"] = _bool(payload["on"], "set_edit_layer on")
        if "delete" in payload:
            out["delete"] = _bool(payload["delete"], "set_edit_layer delete")
        if len(out) == 1:
            _fail("set_edit_layer needs variant, on or delete")
        return out
    if action == "set_develop_before":
        _keys(payload, "set_develop_before payload", {"on"})
        return {"on": _bool(payload["on"], "set_develop_before on")}
    if action == "set_edit_mode":
        _keys(payload, "set_edit_mode payload", {"mode"})
        if payload["mode"] not in EDIT_MODES:
            _fail("set_edit_mode mode must be adjust, masks, generate or crop")
        return {"mode": payload["mode"]}
    if action == "edit_history_jump":
        _keys(payload, "edit_history_jump payload", {"at"})
        return {"at": _integer(payload["at"], "edit_history_jump at", 0, 2**31)}
    if action == "targeted_adjust":
        # A drag of `amount` (CSS px up, + brightens) at x, y (0..1 of the visible picture) on the Tone Curve.
        _keys(payload, "targeted_adjust payload", {"group", "x", "y", "amount"}, {"channel", "property"})
        if payload["group"] not in ("curve", "mixer"):
            _fail("targeted_adjust group must be curve or mixer")
        if ("channel" in payload and payload["group"] != "curve") or ("property" in payload and payload["group"] != "mixer"):
            _fail("targeted_adjust takes channel only for curve and property only for mixer")
        out = {"group": payload["group"]}
        for key in ("x", "y"):
            out[key] = _number(payload[key], f"targeted_adjust {key}", minimum=0.0, maximum=1.0)
        out["amount"] = _number(payload["amount"], "targeted_adjust amount", minimum=-TARGETED_AMOUNT_MAX, maximum=TARGETED_AMOUNT_MAX)
        if "channel" in payload:
            if payload["channel"] not in TARGETED_CHANNELS:
                _fail("targeted_adjust channel must be master, red, green or blue")
            out["channel"] = payload["channel"]
        if "property" in payload:
            if payload["property"] not in TARGETED_PROPERTIES:
                _fail("targeted_adjust property must be hue, saturation or luminance")
            out["property"] = payload["property"]
        return out
    if action == "set_edit_selection":
        # Select > All / Deselect / Inverse and Select > Modify (Expand, Contract, Feather, Border, Smooth);
        # w23-img: Reselect, Quick mask, Transform selection, Save / Load selection.
        _keys(payload, "set_edit_selection payload", {"op"}, {"px", "on", "scale", "scale_x", "scale_y", "rotate", "dx", "dy", "name", "mode"})
        if payload["op"] not in SELECTION_OPS + SELECTION_OPS_W23:
            _fail("set_edit_selection op must be all, none, invert, grow, shrink, feather, border, smooth, reselect, quick_mask, transform, save or load")
        out = {"op": payload["op"]}
        allowed = {"quick_mask": {"on"}, "transform": {"scale", "scale_x", "scale_y", "rotate", "dx", "dy"}, "save": {"name"}, "load": {"name", "mode"}}
        extra = set(payload) - {"op", "px"} - allowed.get(payload["op"], set())
        if extra:
            _fail(f"set_edit_selection {payload['op']} takes no {', '.join(sorted(extra))}")
        if "px" in payload:
            if payload["op"] not in SELECTION_PX_MAX:
                _fail(f"set_edit_selection {payload['op']} takes no px")
            out["px"] = _integer(payload["px"], f"set_edit_selection px (1..{SELECTION_PX_MAX[payload['op']]})", 1, SELECTION_PX_MAX[payload["op"]])
        if "on" in payload:
            out["on"] = _bool(payload["on"], "set_edit_selection on")
        for key in ("scale", "scale_x", "scale_y"):
            if key in payload:
                out[key] = _number(payload[key], f"set_edit_selection {key} (%)", minimum=1.0, maximum=1000.0)
        if "rotate" in payload:
            out["rotate"] = _number(payload["rotate"], "set_edit_selection rotate (degrees)", minimum=-360.0, maximum=360.0)
        for key in ("dx", "dy"):
            if key in payload:
                out[key] = _integer(payload[key], f"set_edit_selection {key} (px)", -65536, 65536)
        if "name" in payload:
            out["name"] = _text(payload["name"], "set_edit_selection name", 60, path_free=False)
        if payload["op"] == "load" and "name" not in payload:
            _fail("set_edit_selection load needs the saved selection's name")
        if "mode" in payload:
            if payload["mode"] not in SELECTION_LOAD_MODES:
                _fail("set_edit_selection mode must be new, add, subtract or intersect")
            out["mode"] = payload["mode"]
        return out
    if action == "set_edit_tool":
        _keys(payload, "set_edit_tool payload", {"tool"})
        if payload["tool"] not in SELECTION_TOOLS:
            _fail("set_edit_tool tool must be move, hand, rect, lasso, object, brush, eraser, box or wand")
        return {"tool": payload["tool"]}
    if action == "refine_edit_selection":
        # Select and Mask: Radius, Smart radius, Smooth, Feather, Contrast, Shift edge, Output to; then OK or Cancel.
        _keys(payload, "refine_edit_selection payload", set(), {*REFINE_RANGES, "smart", "output", "apply", "cancel"})
        if payload.get("cancel") is True:
            if set(payload) != {"cancel"}:
                _fail("refine_edit_selection cancel takes no other field")
            return {"cancel": True}
        out = {}
        for key, (low, high) in REFINE_RANGES.items():
            if key in payload:
                out[key] = _integer(payload[key], f"refine_edit_selection {key} ({low}..{high})", low, high)
        if "smart" in payload:
            out["smart"] = _bool(payload["smart"], "refine_edit_selection smart")
        if "output" in payload:
            if payload["output"] not in REFINE_OUTPUTS:
                _fail("refine_edit_selection output must be selection or png")
            out["output"] = payload["output"]
        if "apply" in payload:
            out["apply"] = _bool(payload["apply"], "refine_edit_selection apply")
        if "cancel" in payload:
            _fail("refine_edit_selection cancel must be true")
        return out
    if action == "set_edit_boxes":
        # Image Box: the boxes (replacing the list), the box prompt with <id> references, model, Blend colour match, variants, seed.
        _keys(payload, "set_edit_boxes payload", set(), {"boxes", "prompt", "model", "color_match", "variants", "seed", "limit"})
        if not payload:
            _fail("set_edit_boxes needs at least one field")
        out: dict[str, Any] = {}
        if "boxes" in payload:
            out["boxes"] = [_box_item(item, f"set_edit_boxes boxes[{i}]") for i, item in enumerate(_box_list(payload["boxes"]))]
        if "prompt" in payload:
            out["prompt"] = _text(payload["prompt"], "set_edit_boxes prompt", 2000, path_free=False)
        if "model" in payload:
            if not isinstance(payload["model"], str) or payload["model"] not in BOX_MODELS:
                _fail("set_edit_boxes model must be qwen-image-2.1, flux3-klein or flux2-klein")
            out["model"] = payload["model"]
        if "color_match" in payload:
            out["color_match"] = _bool(payload["color_match"], "set_edit_boxes color_match")
        if "limit" in payload:
            if payload["limit"] not in BOX_LIMITS:
                _fail("set_edit_boxes limit must be off or selection")
            out["limit"] = payload["limit"]
        if "variants" in payload:
            out["variants"] = _integer(payload["variants"], "set_edit_boxes variants", 1, 4)
        if "seed" in payload:
            out["seed"] = _integer(payload["seed"], "set_edit_boxes seed", 0, 2**31 - 1)
        return out
    if action == "find_edit_boxes":
        _keys(payload, "find_edit_boxes payload", {"query"})
        query = _text(payload["query"], "find_edit_boxes query", 120, path_free=False)
        if not query.strip():
            _fail("find_edit_boxes query must say what to find")
        return {"query": query}
    if action == "run_edit_boxes":
        _keys(payload, "run_edit_boxes payload", set(), {"variants", "seed"})
        out = {}
        if "variants" in payload:
            out["variants"] = _integer(payload["variants"], "run_edit_boxes variants", 1, 4)
        if "seed" in payload:
            out["seed"] = _integer(payload["seed"], "run_edit_boxes seed", 0, 2**31 - 1)
        return out
    if action == "export_edit":
        # M-43: Export of the picture as Edit shows it (develop, masks, shown layers, crop) into the project's
        # exports/images; the same settings as the File > Export panel.
        _keys(payload, "export_edit payload", set(), {"format", "quality", "size", "metadata"})
        out: dict[str, Any] = {}
        if "format" in payload:
            if payload["format"] not in EXPORT_EDIT_FORMATS:
                _fail("export_edit format is invalid")
            out["format"] = payload["format"]
        if "quality" in payload:
            out["quality"] = _integer(payload["quality"], "export_edit quality", 1, 100)
        if "size" in payload:
            size = _keys(payload["size"], "export_edit size", {"mode"}, {"value"})
            if size["mode"] == "original" and "value" not in size:
                out["size"] = {"mode": "original"}
            elif size["mode"] == "long_edge":
                out["size"] = {"mode": "long_edge", "value": _integer(size.get("value"), "export_edit size value", 16, 16384)}
            elif size["mode"] == "percent":
                out["size"] = {"mode": "percent", "value": _integer(size.get("value"), "export_edit size value", 1, 400)}
            else:
                _fail("export_edit size is invalid")
        if "metadata" in payload:
            out["metadata"] = _bool(payload["metadata"], "export_edit metadata")
        return out
    if action == "run_command":
        _keys(payload, "run_command payload", {"id"})
        return {"id": _command_id(payload["id"], "run_command id")}
    if action == "view_edit":
        _keys(payload, "view_edit payload", set(), {"edge"})
        return {"edge": _integer(payload["edge"], "view_edit edge", *VIEW_EDGE)} if "edge" in payload else {}
    if payload:
        _fail(f"{action} does not accept a payload")
    return {}


def validate_view(data_url: Any, width: Any, height: Any) -> dict[str, Any]:
    """A view_edit picture: a bounded JPEG data URL and its size, nothing else."""
    if not isinstance(data_url, str) or len(data_url) > VIEW_MAX_CHARS or not data_url.startswith(_VIEW_PREFIX):
        _fail("view data_url is invalid")
    if not _VIEW_B64_RE.fullmatch(data_url[len(_VIEW_PREFIX):]):
        _fail("view data_url is invalid")
    return {
        "data_url": data_url,
        "width": _integer(width, "view width", 1, VIEW_EDGE[1]),
        "height": _integer(height, "view height", 1, VIEW_EDGE[1]),
    }


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _semantic_state(value: Any) -> Any:
    """Elapsed time ticks must not advance the action revision."""
    if not isinstance(value, dict):
        return value
    projection = json.loads(_canonical(value))
    if isinstance(projection.get("job"), dict):
        projection["job"]["elapsed_seconds"] = 0
    return projection


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _atomic_write(path: Path, value: dict[str, Any], *, durable: bool = True) -> None:
    """Replace ``path`` atomically; ``durable=False`` skips fsync (heartbeat-only rewrites)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.flush()
            if durable:
                os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _validate_ack(ack: Any) -> dict[str, Any]:
    _keys(ack, "ack", {"ok", "reason_code", "message", "applied", "revision", "context_key"})
    _bool(ack["ok"], "ack.ok")
    if ack["reason_code"] is not None:
        _pattern(ack["reason_code"], "ack.reason_code", re.compile(r"^[a-z0-9_]{1,64}$"))
    _optional_text(ack["message"], "ack.message", 240)
    if ack["applied"] is not None:
        if not isinstance(ack["applied"], dict) or len(_canonical(ack["applied"])) > 4096:
            _fail("ack applied state is invalid")
        if _ABSOLUTE_PATH_RE.search(_canonical(ack["applied"])):
            _fail("ack applied state must not contain a filesystem path")
    _integer(ack["revision"], "ack.revision", 0, 2**53 - 1)
    _text(ack["context_key"], "ack.context_key", 160)
    return json.loads(_canonical(ack))


class ImageBridgeStore:
    """Atomic path-per-envelope storage for Image pane observations and actions."""

    def __init__(self, root: Path, clock: Callable[[], float] | None = None) -> None:
        self.root = Path(root)
        self.clock = clock or time.time

    def prune_if_due(self) -> list[str]:
        """Delete window files and action dirs idle for 24 h: first use after start, then hourly."""
        key = str(self.root)
        now = float(self.clock())
        with _PRUNE_LOCK:
            last = _PRUNED_AT.get(key)
            if last is not None and 0.0 <= now - last < PRUNE_INTERVAL_SECONDS:
                return []
            _PRUNED_AT[key] = now
        return self.prune(now)

    def prune(self, now: float | None = None) -> list[str]:
        """Remove idle windows' state files and action dirs; returns the removed window ids."""
        cutoff = (float(self.clock()) if now is None else now) - PRUNE_AGE_SECONDS
        removed: list[str] = []
        try:
            states = list(self.state_dir.glob("win_*.json")) if self.state_dir.is_dir() else []
        except OSError:
            states = []
        for path in states:
            try:
                if _WINDOW_RE.match(path.stem) and path.stat().st_mtime < cutoff:
                    # Gone with its actions (HER-048): a pending file newer than the state must not keep the folder.
                    self.forget(path.stem)
                    removed.append(path.stem)
            except OSError:
                continue
        try:
            folders = list(self.action_dir.glob("win_*")) if self.action_dir.is_dir() else []
        except OSError:
            folders = []
        for folder in folders:
            try:
                if not _WINDOW_RE.match(folder.name) or folder.is_symlink() or not folder.is_dir():
                    continue
                if (self.state_dir / f"{folder.name}.json").exists():
                    continue  # the window still reports; its actions are its own business
                newest = max([folder.stat().st_mtime] + [item.stat().st_mtime for item in folder.rglob("*")])
                if newest < cutoff:
                    shutil.rmtree(folder, ignore_errors=True)
                    if folder.name not in removed:
                        removed.append(folder.name)
            except OSError:
                continue
        return removed

    @property
    def state_dir(self) -> Path:
        return self.root / "states"

    @property
    def action_dir(self) -> Path:
        return self.root / "actions"

    def _state_path(self, window_id: str) -> Path:
        return self.state_dir / f"{_window_id(window_id)}.json"

    def _pending_path(self, window_id: str, action_id: str) -> Path:
        return self.action_dir / _window_id(window_id) / "pending" / f"{_action_id(action_id)}.json"

    def _ack_path(self, window_id: str, action_id: str) -> Path:
        return self.action_dir / _window_id(window_id) / "acks" / f"{_action_id(action_id)}.json"

    def write_snapshot(self, window_id: str, state: dict[str, Any]) -> dict[str, Any]:
        window_id = _window_id(window_id)
        state = _validate_state(state)
        path = self._state_path(window_id)
        previous = _read_json(path) if path.exists() else None
        revision = int(previous.get("revision", 0)) if previous else 0
        changed = previous is None or _semantic_state(previous.get("state")) != _semantic_state(state)
        if changed:
            revision += 1
        record = {
            "schema": STATE_SCHEMA,
            "window_id": window_id,
            "revision": revision,
            "observed_at": float(self.clock()),
            "state": state,
        }
        # Same semantic state: only the heartbeat (observed_at) moves. Still
        # written - freshness is how the agent sees a connected window - but
        # without fsync; losing it in a crash costs one heartbeat.
        _atomic_write(path, record, durable=changed)
        self.prune_if_due()
        return record

    def forget(self, window_id: str) -> None:
        """Remove a gone window: its state file and its pending actions, acks and views."""
        window_id = _window_id(window_id)
        self._state_path(window_id).unlink(missing_ok=True)
        shutil.rmtree(self.action_dir / window_id, ignore_errors=True)

    def list_snapshots(
        self,
        window_id: str | None = None,
        max_age_seconds: float = 5.0,
        forget_after_seconds: float | None = None,
    ) -> list[dict[str, Any]]:
        """Rows for reporting windows; with ``forget_after_seconds`` a window silent that long is removed on read."""
        if window_id is not None:
            paths = [self._state_path(window_id)]
        elif self.state_dir.is_dir():
            paths = sorted(self.state_dir.glob("win_*.json"))
        else:
            paths = []
        now = float(self.clock())
        if window_id is None:
            self.prune_if_due()
            paths = [path for path in paths if path.exists()]
        rows: list[dict[str, Any]] = []
        for path in paths:
            record = _read_json(path)
            if forget_after_seconds is not None and path.exists() and _WINDOW_RE.match(path.stem):
                try:
                    observed = float(record["observed_at"]) if record else path.stat().st_mtime
                except (OSError, KeyError, TypeError, ValueError):
                    observed = now
                if now - observed > forget_after_seconds:
                    self.forget(path.stem)
                    continue
            if not record or record.get("schema") != STATE_SCHEMA:
                continue
            try:
                wid = _window_id(record.get("window_id"))
                revision = int(record.get("revision"))
                observed_at = float(record.get("observed_at"))
                state = _validate_state(record.get("state"))
            except (ImageBridgeValidationError, TypeError, ValueError):
                continue
            age = max(0.0, now - observed_at)
            fresh = age <= max(0.0, float(max_age_seconds))
            rows.append({
                "schema": STATE_SCHEMA,
                "window_id": wid,
                "revision": revision,
                "observed_at": observed_at,
                "age_seconds": round(age, 3),
                "freshness": "fresh" if fresh else "stale",
                "connected": fresh,
                "state": state,
            })
        rows.sort(key=lambda row: (row["connected"], row["observed_at"]), reverse=True)
        return rows

    def enqueue_action(
        self,
        window_id: str,
        action: str,
        payload: dict[str, Any],
        expected_revision: int | None = None,
        context_key: str | None = None,
    ) -> dict[str, Any]:
        window_id = _window_id(window_id)
        payload = validate_action_payload(action, payload)
        if expected_revision is not None:
            _integer(expected_revision, "expected_revision", 1, 2**53 - 1)
        if context_key is not None:
            _text(context_key, "context_key", 160, allow_empty=False)
        created_at = float(self.clock())
        action_id = "act_" + uuid.uuid4().hex
        record = {
            "schema": ACTION_SCHEMA,
            "action_id": action_id,
            "window_id": window_id,
            "action": action,
            "payload": payload,
            "expected_revision": expected_revision,
            "context_key": context_key,
            "created_at": created_at,
            "expires_at": created_at + ACTION_TTL_SECONDS,
        }
        _atomic_write(self._pending_path(window_id, action_id), record)
        return record

    def pending_actions(self, window_id: str, since: float = 0.0) -> list[dict[str, Any]]:
        window_id = _window_id(window_id)
        pending = self.action_dir / window_id / "pending"
        if not pending.is_dir():
            return []
        now = float(self.clock())
        rows: list[dict[str, Any]] = []
        for path in sorted(pending.glob("act_*.json")):
            record = _read_json(path)
            if not record:
                continue
            try:
                _action_id(record.get("action_id"))
                if record.get("schema") != ACTION_SCHEMA or record.get("window_id") != window_id or float(record.get("expires_at")) <= now:
                    path.unlink(missing_ok=True)
                    continue
                if float(record.get("created_at")) < float(since):
                    continue
                validate_action_payload(record.get("action"), record.get("payload"))
            except (ImageBridgeValidationError, TypeError, ValueError):
                continue
            rows.append(record)
        rows.sort(key=lambda row: (float(row["created_at"]), row["action_id"]))
        return rows

    def ack_action(self, window_id: str, action_id: str, ack: dict[str, Any]) -> dict[str, Any]:
        window_id = _window_id(window_id)
        action_id = _action_id(action_id)
        record = {"schema": ACK_SCHEMA, "action_id": action_id, "window_id": window_id, **_validate_ack(ack)}
        _atomic_write(self._ack_path(window_id, action_id), record)
        self._pending_path(window_id, action_id).unlink(missing_ok=True)
        return record

    def consume_ack(self, window_id: str, action_id: str) -> dict[str, Any] | None:
        path = self._ack_path(window_id, action_id)
        record = _read_json(path)
        if record is None:
            return None
        path.unlink(missing_ok=True)
        return record

    def _view_path(self, window_id: str, action_id: str) -> Path:
        return self.action_dir / _window_id(window_id) / "views" / f"{_action_id(action_id)}.json"

    def put_view(self, window_id: str, action_id: str, data_url: Any, width: Any, height: Any) -> dict[str, Any]:
        """Store the picture a view_edit action produced; only for an action still pending."""
        window_id = _window_id(window_id)
        action_id = _action_id(action_id)
        record = _read_json(self._pending_path(window_id, action_id))
        if not record or record.get("action") != "view_edit":
            _fail("no pending view_edit action")
        view = {"schema": VIEW_SCHEMA, "action_id": action_id, "window_id": window_id, **validate_view(data_url, width, height)}
        _atomic_write(self._view_path(window_id, action_id), view, durable=False)
        return view

    def consume_view(self, window_id: str, action_id: str) -> dict[str, Any] | None:
        path = self._view_path(window_id, action_id)
        record = _read_json(path)
        path.unlink(missing_ok=True)
        return record

    def withdraw_action(self, window_id: str, action_id: str) -> bool:
        """Remove a still-pending action so a late renderer never applies it."""
        path = self._pending_path(window_id, action_id)
        existed = path.exists()
        path.unlink(missing_ok=True)
        self._view_path(window_id, action_id).unlink(missing_ok=True)
        return existed



# Lane w19-g7hermes item 4 (HER-052; reference HER-051 music_bridge.ACTION_PAYLOADS): ONE table of every Image action's
# payload. validate_action_payload checks key sets against it before the per-action checks, and the
# continuo_image_action description Hermes reads is built from it (payload_description), so they cannot disagree.
# (key, required, value) per key; () = no payload.
ACTION_PAYLOADS: dict[str, tuple[tuple[str, bool, str], ...]] = {
    "select_tile": (("axis", True, "|".join(AXES)), ("key", True, "tile key from continuo_image_catalog")),
    "clear_axis": (("axis", True, "|".join(AXES)),),
    "set_intent": (("text", True, "words"),),
    "set_manual_prompt": (("text", True, "prompt"),),
    "set_mode": (("mode", True, "builder|manual"),),
    "set_sampling": (("profile", False, "turbo|raw"), ("steps", False, "1-100"), ("cfg", False, "0.1-20"),
                     ("sampler", False, "name"), ("scheduler", False, "name"), ("negative", False, "text")),
    "set_lora_strength": (("file", True, "LoRA file"), ("strength", True, "0-2")),
    "set_seed": (("seed", True, "integer >= 0"),),
    "set_seed_mode": (("mode", True, "|".join(SEED_MODES)),),
    "set_size": (("format", False, "custom|W:H"), ("width", False, "16-8192"), ("height", False, "16-8192"),
                 ("mp", False, "0.05-16"), ("batch", False, "1-64"), ("swap", False, "bool"), ("aspect_lock", False, "bool")),
    "keep": (("name", True, "text"), ("job_id", False, "imgjob_...")),
    "vary": (("variable", True, "seed|steps|cfg|sampler|scheduler|profile|lora:<id>|axis:<style|shot|scene|outfit>"),
             ("values", True, "list")),
    "restore": (("revision_id", True, "imgrev_..."),),
    "open_result": (("job_id", True, "imgjob_..."),),
    "set_canvas": (("zoom", False, "fit|fill|number"), ("center", False, "[x y] 0-1"), ("viewer", False, "bool")),
    "compare": (("a", False, "imgjob_..."), ("b", False, "imgjob_..."), ("mode", False, "|".join(COMPARE_MODES)),
                ("swap", False, "bool"), ("linked", False, "bool"), ("wipe", False, "0-1"), ("zoom", False, "fit|number"),
                ("close", False, "bool")),
    "survey": (("ids", False, "[imgjob_...]"), ("active", False, "imgjob_..."), ("remove", False, "imgjob_..."),
               ("pick", False, "imgjob_..."), ("close", False, "bool")),
    "set_develop": tuple((group, False, "(slider: value)") for group in DEVELOP_GROUPS),
    "reset_develop": (("group", False, "|".join(DEVELOP_GROUPS)),),
    "set_develop_before": (("on", True, "bool"),),
    "set_develop_group": (("group", True, "|".join(DEVELOP_GROUPS)), ("on", True, "bool")),
    "set_develop_clipping": (("shadows", False, "bool"), ("highlights", False, "bool")),
    "set_develop_white_balance": (("mode", False, "|".join(WB_MODES)), ("x", False, "0-1"), ("y", False, "0-1"),
                                  ("w", False, "0-1"), ("h", False, "0-1"), ("picking", False, "bool")),
    "set_develop_preset": (("op", True, "|".join(DEVELOP_PRESET_OPS)), ("name", False, "text"), ("groups", False, "[group]"),
                           ("names", False, "[name]"), ("file", False, "preset text")),
    "auto_tone": (("sliders", False, "[light slider]"),),
    "set_develop_masks": (("masks", False, "mask list"), ("select", False, "(id; component?)"), ("pins", False, "bool"),
                          ("overlay", False, "overlay")),
    "generate_edit": (("operation", True, "|".join(sorted(GENERATIVE_OPERATIONS))), ("prompt", False, "text"),
                      ("target", False, "text"), ("variants", False, "1-4"), ("seed", False, "integer"),
                      ("mask", False, "all|mask id"), ("model", False, "|".join(sorted(EDIT_MODELS))), ("live", False, "bool"),
                      ("strength", False, "0-1")),
    "set_edit_light": (("tone", False, "|".join(f'"{tone}"' for tone in sorted(LIGHT_TONES))), ("from", False, "|".join(sorted(LIGHT_FROM))),
                       ("x", False, "-1..1"), ("y", False, "-1..1"), ("behind", False, "bool"),
                       *((key, False, "number") for key in (*LIGHT_CONTROLS, *LIGHT_SCENE_EXTRA)),
                       ("live", False, "bool"), ("light", False, "light id"), ("add", False, "bool"), ("remove", False, "bool"),
                       ("on", False, "bool"), ("color", False, "#rrggbb"), ("kelvin", False, "kelvin")),
    "stop_edit": (),
    "accept_edit": (("name", False, "text"),),
    "outpaint_edit": (*((side, False, "px") for side in OUTPAINT_SIDES), ("prompt", False, "text"), ("variants", False, "1-4"),
                      ("seed", False, "integer"), ("model", False, "family")),
    "accept_outpaint": (("variant", False, "0-3"), ("name", False, "text")),
    "set_edit_layer": (("layer", True, "layer id"), ("variant", False, "0-3"), ("on", False, "bool"), ("delete", False, "bool")),
    "set_layer_mask": (("layer", True, "layer id"), ("invert", False, "bool"), ("feather", False, "px"), ("density", False, "0-100")),
    "regenerate_layer": (("layer", True, "layer id"),),
    "set_edit_crop": (("crop", False, "(x y w h) 0-1"), ("angle", False, "-45..45"), ("straighten", False, "(x0 y0 x1 y1)"),
                      ("rotate", False, "left|right"), ("flip", False, "horizontal|vertical"), ("size", False, "(w h)")),
    "edit_history_jump": (("at", True, "history row"),),
    "view_edit": (("edge", False, f"{VIEW_EDGE[0]}-{VIEW_EDGE[1]} px"),),
    "set_edit_mode": (("mode", True, "|".join(sorted(EDIT_MODES))),),
    "targeted_adjust": (("group", True, "curve|mixer"), ("x", True, "0-1"), ("y", True, "0-1"), ("amount", True, "number"),
                        ("channel", False, "|".join(TARGETED_CHANNELS)), ("property", False, "|".join(TARGETED_PROPERTIES))),
    "set_edit_selection": (("op", True, "|".join(SELECTION_OPS + SELECTION_OPS_W23)), ("px", False, "px for grow|shrink|feather|border|smooth"),
                           ("on", False, "bool for quick_mask"), ("scale", False, "% for transform"), ("scale_x", False, "%"), ("scale_y", False, "%"),
                           ("rotate", False, "degrees"), ("dx", False, "px"), ("dy", False, "px"), ("name", False, "text for save|load"),
                           ("mode", False, "new|add|subtract|intersect for load")),
    "set_edit_tool": (("tool", True, "|".join(SELECTION_TOOLS)),),
    "refine_edit_selection": (*((key, False, "number") for key in REFINE_RANGES), ("smart", False, "bool"),
                              ("output", False, "|".join(REFINE_OUTPUTS)), ("apply", False, "bool"), ("cancel", False, "bool")),
    "run_command": (("id", True, "command id"),),
    "set_edit_boxes": (("boxes", False, "box list"), ("prompt", False, "text"), ("model", False, "|".join(sorted(BOX_MODELS))),
                       ("color_match", False, "bool"), ("variants", False, "1-4"), ("seed", False, "integer"),
                       ("limit", False, "|".join(BOX_LIMITS))),
    "find_edit_boxes": (("query", True, "words"),),
    "run_edit_boxes": (("variants", False, "1-4"), ("seed", False, "integer")),
    "export_edit": (("format", False, "|".join(EXPORT_EDIT_FORMATS)), ("quality", False, "1-100"),
                    ("size", False, "(mode original|long_edge|percent; value?)"), ("metadata", False, "bool")),
}


def payload_spec(action: str) -> str:
    keys = ", ".join(f"{key}{'' if required else '?'}: {value}" for key, required, value in ACTION_PAYLOADS.get(action, ()))
    return f"{action} {{{keys}}}"


def payload_description() -> str:
    """The payload of every Image action, for the tool description (actions in name order)."""
    with_payload = "; ".join(payload_spec(action) for action in sorted(ALLOWED_ACTIONS) if ACTION_PAYLOADS.get(action))
    without = ", ".join(sorted(action for action in ALLOWED_ACTIONS if not ACTION_PAYLOADS.get(action)))
    return f"Payload per action (? = optional): {with_payload}. No payload ({{}}): {without}."


_validate_action_payload_before_w19 = validate_action_payload


def validate_action_payload(action: Any, payload: Any) -> dict[str, Any]:  # noqa: F811 - append-only extension
    if action in ALLOWED_ACTIONS and isinstance(payload, dict):
        spec = ACTION_PAYLOADS.get(action, ())
        keys, required = set(payload), {key for key, needed, _value in spec if needed}
        if not required <= keys or keys - {key for key, _needed, _value in spec}:
            _fail(f"{action} payload is invalid: {payload_spec(action)}")
    return _validate_action_payload_before_w19(action, payload)


# Lane w24-rest HER-016: Hermes adds and removes references of the Image Create strip (and the Box edit's, when its
# model takes any) with the hand's verbs and limits (desktop image/components/reference-strip.js); the page publishes
# the strip as state.references. Append-only: the action set, payload table, validator and state validator grow here.
REFERENCE_ACTIONS = frozenset({"add_reference", "remove_reference"})
ALLOWED_ACTIONS = ALLOWED_ACTIONS | REFERENCE_ACTIONS
ACTION_PAYLOADS["add_reference"] = (("job_id", True, "imgjob_... of a kept picture (state.kept)"), ("role", False, "a role the slot offers"),
                                    ("slot", False, "slot id (state.references[].slot)"), ("surface", False, "create|box"))
ACTION_PAYLOADS["remove_reference"] = (("index", True, "1-based position in the strip"), ("slot", False, "slot id"), ("surface", False, "create|box"))
_REF_SLOT_RE = re.compile(r"[a-z][a-z0-9_]{0,40}")
_validate_action_payload_before_w24_rest = validate_action_payload


def validate_action_payload(action: Any, payload: Any) -> dict[str, Any]:  # noqa: F811 - append-only extension
    if action not in REFERENCE_ACTIONS:
        return _validate_action_payload_before_w24_rest(action, payload)
    if not isinstance(payload, dict):
        _fail("action payload must be an object")
    spec = ACTION_PAYLOADS[action]
    keys, required = set(payload), {key for key, needed, _value in spec if needed}
    if not required <= keys or keys - {key for key, _needed, _value in spec}:
        _fail(f"{action} payload is invalid: {payload_spec(action)}")
    out: dict[str, Any] = {}
    if action == "add_reference":
        out["job_id"] = _pattern(payload["job_id"], "add_reference job_id", _JOB_RE)
        if "role" in payload:
            out["role"] = _pattern(payload["role"], "add_reference role", _REF_SLOT_RE)
    else:
        out["index"] = _integer(payload["index"], "remove_reference index", 1, 64)
    if "slot" in payload:
        out["slot"] = _pattern(payload["slot"], f"{action} slot", _REF_SLOT_RE)
    if "surface" in payload:
        if payload["surface"] not in ("create", "box"):
            _fail(f"{action} surface is invalid")
        out["surface"] = payload["surface"]
    return out


_validate_state_before_w24_rest = _validate_state


def _validate_state(state: Any) -> dict[str, Any]:  # noqa: F811 - append-only extension
    refs = state.get("references") if isinstance(state, dict) else None
    if isinstance(state, dict) and "references" in state:
        if not isinstance(refs, list) or len(refs) > 8:
            _fail("references is invalid")
        for index, slot in enumerate(refs):
            _keys(slot, f"references[{index}]", {"surface", "slot", "max", "roles", "items"})
            if slot["surface"] not in ("create", "box"):
                _fail(f"references[{index}].surface is invalid")
            _pattern(slot["slot"], f"references[{index}].slot", _REF_SLOT_RE)
            _integer(slot["max"], f"references[{index}].max", 0, 64)
            if not isinstance(slot["roles"], list) or len(slot["roles"]) > 16:
                _fail(f"references[{index}].roles is invalid")
            for role in slot["roles"]:
                _pattern(role, f"references[{index}].roles", _REF_SLOT_RE)
            if not isinstance(slot["items"], list) or len(slot["items"]) > slot["max"]:
                _fail(f"references[{index}].items is invalid")
            for k, item in enumerate(slot["items"]):
                _keys(item, f"references[{index}].items[{k}]", {"job_id", "role"})
                _pattern(item["job_id"], f"references[{index}].items[{k}].job_id", _JOB_RE)
                _pattern(item["role"], f"references[{index}].items[{k}].role", _REF_SLOT_RE)
        rest = {k: v for k, v in state.items() if k != "references"}
        out = _validate_state_before_w24_rest(rest)
        return {**out, "references": refs}
    return _validate_state_before_w24_rest(state)

# Lane w26-player (HER-089): the page publishes the model the artist sees (state.model: family, label, variant,
# target_id, uses_render_spec, and every visible field with its value), so Hermes reads the hand-picked model instead of
# the Krea render_spec. Append-only: the state validator grows here.
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9_.:\-]{0,120}$")
_validate_state_before_w26_player = _validate_state


def _validate_model_state(model: Any) -> None:
    _keys(model, "model", {"family", "label", "variant", "target_id", "uses_render_spec", "fields"})
    for key in ("family", "variant", "target_id"):
        _pattern(model[key], f"model.{key}", _MODEL_ID_RE)
    _text(model["label"], "model.label", 120)
    _bool(model["uses_render_spec"], "model.uses_render_spec")
    if not isinstance(model["fields"], list) or len(model["fields"]) > 32:
        _fail("model.fields is invalid")
    for index, field in enumerate(model["fields"]):
        _keys(field, f"model.fields[{index}]", {"id", "label", "kind", "value"}, {"min", "max", "options"})
        _pattern(field["id"], f"model.fields[{index}].id", _REF_SLOT_RE)
        _text(field["label"], f"model.fields[{index}].label", 120)
        _pattern(field["kind"], f"model.fields[{index}].kind", re.compile(r"^[a-z]{1,16}$"))
        value = field["value"]
        if isinstance(value, str):
            _text(value, f"model.fields[{index}].value", 400, path_free=False)
        elif not isinstance(value, (int, float, bool)) or (isinstance(value, float) and not math.isfinite(value)):
            _fail(f"model.fields[{index}].value is invalid")
        for key in ("min", "max"):
            if key in field and (isinstance(field[key], bool) or not isinstance(field[key], (int, float)) or not math.isfinite(field[key])):
                _fail(f"model.fields[{index}].{key} is invalid")
        if "options" in field:
            if not isinstance(field["options"], list) or len(field["options"]) > 64:
                _fail(f"model.fields[{index}].options is invalid")
            for option in field["options"]:
                _text(option, f"model.fields[{index}].options", 120, path_free=False)


def _validate_state(state: Any) -> dict[str, Any]:  # noqa: F811 - append-only extension
    if isinstance(state, dict) and "model" in state:
        model = state["model"]
        _validate_model_state(model)
        out = _validate_state_before_w26_player({k: v for k, v in state.items() if k != "model"})
        return {**out, "model": model}
    return _validate_state_before_w26_player(state)


# Lane w27-model (HER-013): Hermes chooses the Image model the way the hand does in the picker (Create: the family
# button's list and its presets, Krea 2's Model files; Edit: the operation's Model select). The page runs the picker's
# own code (src/image/panel/model-bridge.js): a "Not available" model is refused with the row's reason, a measured-weak
# Edit model is chosen and answers applied.warning. Reference: Photoshop/Lightroom scripting set the control the panel
# shows; a choice the panel refuses is an error naming why. Append-only: actions, payload table, validator.
MODEL_ACTIONS = frozenset({"set_model"})
ALLOWED_ACTIONS = ALLOWED_ACTIONS | MODEL_ACTIONS
ACTION_PAYLOADS["set_model"] = (("family", True, "family id (continuo_models_catalog; Create picker or state.develop.generative.models)"),
                                ("variant", False, "preset id of that family (e.g. turbo|raw|create)"),
                                ("file", False, "Krea 2 model file as the Model files list names it"),
                                ("surface", False, "create|edit"))
_SET_MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_SET_MODEL_FILE_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_. ()+\[\]-]{0,119}(?:/[A-Za-z0-9_][A-Za-z0-9_. ()+\[\]-]{0,119}){0,3}$")
_validate_action_payload_before_w27_model = validate_action_payload


def validate_action_payload(action: Any, payload: Any) -> dict[str, Any]:  # noqa: F811 - append-only extension
    if action not in MODEL_ACTIONS:
        return _validate_action_payload_before_w27_model(action, payload)
    if not isinstance(payload, dict):
        _fail("action payload must be an object")
    spec = ACTION_PAYLOADS[action]
    keys, required = set(payload), {key for key, needed, _value in spec if needed}
    if not required <= keys or keys - {key for key, _needed, _value in spec}:
        _fail(f"{action} payload is invalid: {payload_spec(action)}")
    out: dict[str, Any] = {"family": _pattern(payload["family"], "set_model family", _SET_MODEL_ID_RE)}
    if "variant" in payload:
        out["variant"] = _pattern(payload["variant"], "set_model variant", _SET_MODEL_ID_RE)
    if "file" in payload:
        file = _pattern(payload["file"], "set_model file", _SET_MODEL_FILE_RE)
        if ".." in file.split("/"):
            _fail("set_model file is invalid")
        out["file"] = file
    if "surface" in payload:
        if payload["surface"] not in ("create", "edit"):
            _fail("set_model surface is invalid")
        out["surface"] = payload["surface"]
    return out
