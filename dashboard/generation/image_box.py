"""Image Box: boxes with an intent, compiled for FLUX.3 Klein (native) or into a mask (every other model).

Pure Python (PIL only in ``box_mask``), loaded both as a package member and by path,
so no bare relative imports and no ``@dataclass``.

A box is ``{id, intent, desc, bbox: [x, y, w, h], target_bbox?: [x, y, w, h]}`` in
source pixels; intents are Add, Change, Move, Keep and Remove (the NKD Klein Tools
kinds new / move / keep / remove; Add and Change are both "new content at a box",
their description says what to do).  The intent is a field of the box: every box is
wired the same way.

FLUX.3 Klein reads boxes itself (Black Forest Labs' edit layout, as NKD
``box_prompt._emit`` writes it):

* the prompt gets one sentence per box whose ``<id>`` it does not name itself
  (Remove / Keep / Move pre-written, Add / Change = the box description), the first
  one prefixed "In <ref_image_0>, " and "Keep everything else in the image exactly
  unchanged." appended - each only when the prompt does not say it already;
* then the element list ``[{id, from, src_bbox, tgt_bbox, desc}]`` with boxes as
  ``[top, left, bottom, right]`` on a 0..1000 grid of the picture the model sees.

Every other edit model takes a mask: ``box_mask`` rasterises the changing boxes
(feathered like the Suite's measured edit masks), cuts Keep boxes out hard and
refuses Move (a mask cannot carry a thing to a new place).
"""
from __future__ import annotations

import io
import json
import re
from typing import Any, Mapping

INTENTS = ("add", "change", "move", "keep", "remove")
MAX_BOXES = 16
MAX_DESC = 160
ID_MAX = 32
_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,31}$")
# Box -> mask feather: a Gaussian blur of 0.6 % of the long edge (6 px at 1024, the soft edge
# of the Suite's measured Remove bench mask, render_targets.mask_png GaussianBlur(6)).
FEATHER_FRACTION = 0.006
# Box -> mask: which edit operation a changing intent becomes for a mask-taking model.
MASK_OPERATION = {"remove": "remove", "add": "fill", "change": "replace"}
KEEP_ALL = "Keep everything else in the image exactly unchanged."
REGION_GROW_FRACTION = 0.04


class BoxError(ValueError):
    """A finite, user-facing refusal with a stable code (HTTP 422 at the route)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _rect(value: Any, name: str, size: tuple[int, int] | None) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4 or any(isinstance(v, bool) or not isinstance(v, int) for v in value):
        raise BoxError("invalid_box", f"{name} must be [x, y, width, height] in whole pixels")
    x, y, w, h = (int(v) for v in value)
    if x < 0 or y < 0 or w < 2 or h < 2:
        raise BoxError("invalid_box", f"{name} must lie on the picture and be at least 2 x 2 pixels")
    if size is not None and (x + w > size[0] or y + h > size[1]):
        raise BoxError("box_outside_picture", f"{name} reaches outside the {size[0]} x {size[1]} picture")
    return [x, y, w, h]


def normalize_boxes(raw: Any, size: tuple[int, int] | None = None) -> list[dict[str, Any]]:
    """Validate the caller's boxes; ``size`` (picture w, h) checks they lie on the picture."""
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_BOXES:
        raise BoxError("invalid_box", f"Boxes must be a list of 1 to {MAX_BOXES}")
    out, seen = [], set()
    for index, item in enumerate(raw, 1):
        if not isinstance(item, Mapping):
            raise BoxError("invalid_box", f"Box {index} is invalid")
        box_id, intent, desc = item.get("id"), item.get("intent"), item.get("desc", "")
        if not isinstance(box_id, str) or not _ID.match(box_id):
            raise BoxError("invalid_box", f"Box {index}: the id must be letters, digits and underscores (up to {ID_MAX}), not starting with a digit")
        if box_id in seen:
            raise BoxError("invalid_box", f"Box id {box_id} is used twice")
        seen.add(box_id)
        if intent not in INTENTS:
            raise BoxError("invalid_box", f"{box_id}: the intent must be add, change, move, keep or remove")
        if desc is None:
            desc = ""
        if not isinstance(desc, str) or len(desc) > MAX_DESC or any(ord(c) < 32 for c in desc):
            raise BoxError("invalid_box", f"{box_id}: the description is invalid (one short line, up to {MAX_DESC} characters)")
        desc = " ".join(desc.split())
        if intent in ("add", "change") and not desc:
            raise BoxError("invalid_box", f"{box_id}: {intent.capitalize()} needs a description of what goes there")
        box = {"id": box_id, "intent": intent, "desc": desc, "bbox": _rect(item.get("bbox"), f"{box_id}: the box", size)}
        target = item.get("target_bbox")
        if intent == "move":
            if target is None:
                raise BoxError("invalid_box", f"{box_id}: Move needs a target box")
            box["target_bbox"] = _rect(target, f"{box_id}: the target box", size)
        elif target is not None:
            raise BoxError("invalid_box", f"{box_id}: only Move takes a target box")
        out.append(box)
    return out


def to_grid(bbox: list[int], frame: list[int]) -> list[int]:
    """[x, y, w, h] in source pixels -> [top, left, bottom, right] on 0..1000 of ``frame`` [fx, fy, fw, fh].

    The rounding is NKD's ``box_prompt._px_box`` (clipped to the frame, Python ``round``).
    """
    fx, fy, fw, fh = (int(v) for v in frame)
    fw, fh = max(1, fw), max(1, fh)
    x1, y1 = bbox[0] - fx, bbox[1] - fy
    x2, y2 = x1 + bbox[2], y1 + bbox[3]
    t, l = round(max(0.0, y1) / fh * 1000), round(max(0.0, x1) / fw * 1000)
    b, r = round(min(fh, y2) / fh * 1000), round(min(fw, x2) / fw * 1000)
    if b - t < 1 or r - l < 1:
        raise BoxError("box_outside_picture", "A box lies outside the edited area")
    return [t, l, b, r]


def bfl_rows(boxes: list[Mapping[str, Any]], frame: list[int]) -> list[dict[str, Any]]:
    """BFL's edit element list: ``{id, from, src_bbox, tgt_bbox, desc}`` (reference = the picture, ref_image_0)."""
    rows = []
    for box in boxes:
        grid = to_grid(box["bbox"], frame)
        intent = box["intent"]
        if intent in ("add", "change"):  # no source: new content there, or an instruction for what is there
            row = {"from": None, "src_bbox": None, "tgt_bbox": grid}
        elif intent == "remove":
            row = {"from": "ref_image_0", "src_bbox": grid, "tgt_bbox": None}
        elif intent == "keep":  # BFL: src == tgt is an anchor
            row = {"from": "ref_image_0", "src_bbox": grid, "tgt_bbox": list(grid)}
        else:
            row = {"from": "ref_image_0", "src_bbox": grid, "tgt_bbox": to_grid(box["target_bbox"], frame)}
        rows.append({"id": box["id"], **row, "desc": box["desc"]})
    return rows


def sentence(box: Mapping[str, Any]) -> str:
    """The caption sentence for a box the prompt does not name (NKD ``_mention``, edit shape)."""
    box_id, desc, intent = box["id"], box["desc"], box["intent"]
    what = f"the {desc} <{box_id}>" if desc else f"<{box_id}>"
    if intent in ("add", "change"):
        return f"{desc.rstrip('.')} <{box_id}>." if desc else f"<{box_id}>."
    if intent == "remove":
        return f"Remove {what} completely and fill its place with the surrounding background."
    if intent == "keep":
        return f"Keep {what} unchanged."
    return f"Move {what} from <ref_image_0>."


def compile_prompt(prompt: str, boxes: list[Mapping[str, Any]], frame: list[int]) -> dict[str, Any]:
    """The final prompt FLUX.3 Klein receives: caption + sentences + the BFL list (NKD ``_emit``).

    Returns ``{"prompt", "sentences", "rows"}``; ``sentences`` are the ones added (boxes the
    prompt names as ``<id>`` get none).
    """
    caption = (prompt or "").rstrip()
    rows = bfl_rows(boxes, frame)
    missing = [sentence(box) for box in boxes if f"<{box['id']}>" not in caption]
    if missing:
        if "<ref_image_" not in caption:
            missing[0] = "In <ref_image_0>, " + missing[0][0].lower() + missing[0][1:]
        if "everything else" not in caption.lower():
            missing.append(KEEP_ALL)
    text = " ".join([caption] + missing).strip()
    return {"prompt": f"{text} {json.dumps(rows, ensure_ascii=False)}".strip(), "sentences": missing, "rows": rows}


def mask_plan(boxes: list[Mapping[str, Any]], prompt: str = "", label: str = "This model") -> dict[str, Any]:
    """Box -> mask: the one edit operation the changing boxes become, and its prompt.

    A mask edit is one operation: the changing boxes must share one intent (Keep boxes are
    cut out of the mask); Move is refused, a mask cannot carry a thing to a new place.
    """
    if any(b["intent"] == "move" for b in boxes):
        raise BoxError("box_move_unavailable", f"Move is not available for {label}: it edits through a mask, which cannot carry a thing to a new place (FLUX.3 Klein can)")
    changing = [b for b in boxes if b["intent"] != "keep"]
    if not changing:
        raise BoxError("invalid_box", "Keep boxes alone change nothing: no Add, Change or Remove box")
    intents = {b["intent"] for b in changing}
    if len(intents) > 1:
        raise BoxError("box_mixed_intents", f"{label} makes one kind of edit at a time through a mask: one of Add, Change or Remove (plus Keep); FLUX.3 Klein takes mixed boxes")
    intent = intents.pop()
    text = " ".join((prompt or "").split())
    if not text and intent != "remove":
        text = " and ".join(dict.fromkeys(b["desc"] for b in changing))
    if not text and intent == "remove":
        text = " and ".join(dict.fromkeys(b["desc"] for b in changing if b["desc"]))
    return {"operation": MASK_OPERATION[intent], "intent": intent, "prompt": text,
            "changing": [b["id"] for b in changing], "kept": [b["id"] for b in boxes if b["intent"] == "keep"]}


def box_mask(boxes: list[Mapping[str, Any]], size: tuple[int, int]) -> dict[str, Any]:
    """8-bit mask PNG at picture size: changing boxes white with a feathered edge, Keep boxes 0 (hard)."""
    import numpy as np  # type: ignore[import-not-found]
    from PIL import Image, ImageDraw, ImageFilter  # type: ignore[import-not-found]

    width, height = int(size[0]), int(size[1])
    if any(b["intent"] == "move" for b in boxes):
        raise BoxError("box_move_unavailable", "Move cannot be made into a mask")
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for b in boxes:
        if b["intent"] != "keep":
            x, y, w, h = b["bbox"]
            draw.rectangle((x, y, x + w - 1, y + h - 1), fill=255)
    feather = max(1.0, max(width, height) * FEATHER_FRACTION)
    image = image.filter(ImageFilter.GaussianBlur(feather))
    mask = np.asarray(image, dtype=np.uint8).copy()
    for b in boxes:
        if b["intent"] == "keep":
            x, y, w, h = b["bbox"]
            mask[y:y + h, x:x + w] = 0
    out = io.BytesIO()
    Image.fromarray(mask, mode="L").save(out, format="PNG", optimize=False, compress_level=6)
    ys, xs = np.nonzero(mask >= 128)
    bbox = None if xs.size == 0 else [int(xs.min()), int(ys.min()), int(xs.max()) - int(xs.min()) + 1, int(ys.max()) - int(ys.min()) + 1]
    if bbox is None:
        raise BoxError("empty_mask", "The boxes leave nothing to change")
    return {"png": out.getvalue(), "bbox": bbox, "feather_px": round(feather, 2),
            "coverage_pct": round(float((mask >= 128).mean() * 100.0), 3)}


def cover(boxes: list[Mapping[str, Any]]) -> list[int]:
    """(x0, y0, x1, y1) around every box and Move target: the crop must show all of them."""
    rects = [b["bbox"] for b in boxes] + [b["target_bbox"] for b in boxes if b.get("target_bbox")]
    return [min(r[0] for r in rects), min(r[1] for r in rects), max(r[0] + r[2] for r in rects), max(r[1] + r[3] for r in rects)]


def region_mask(boxes: list[Mapping[str, Any]], size: tuple[int, int]) -> dict[str, Any]:
    """FLUX.3 Klein: where the result is pasted back - every changing box (and Move target) grown, feathered.

    A box is not a hard mask (the model redraws around it: the tower's rim, its shadow); the grow is
    the Suite's measured Remove margin (image_edit_service.REMOVE_GROW_FRACTION, 4 % of the long edge).
    Keep boxes only guide the model; outside the pasted area the picture stays byte for byte.
    """
    import numpy as np  # type: ignore[import-not-found]
    from PIL import Image, ImageDraw, ImageFilter  # type: ignore[import-not-found]

    width, height = int(size[0]), int(size[1])
    grow = max(2, int(round(max(width, height) * REGION_GROW_FRACTION)))
    changing = [b for b in boxes if b["intent"] != "keep"]
    if not changing:
        raise BoxError("invalid_box", "Keep boxes alone change nothing: no Add, Change, Move or Remove box")
    grown, inner = Image.new("L", (width, height), 0), Image.new("L", (width, height), 0)
    for b in changing:
        for x, y, w, h in [b["bbox"]] + ([b["target_bbox"]] if b.get("target_bbox") else []):
            ImageDraw.Draw(grown).rectangle((x - grow, y - grow, x + w - 1 + grow, y + h - 1 + grow), fill=255)
            ImageDraw.Draw(inner).rectangle((x, y, x + w - 1, y + h - 1), fill=255)
    # the boxes grown by ``grow``, feathered inwards (nothing past the grown rectangle changes), the boxes themselves full
    soft = np.minimum(np.asarray(grown.filter(ImageFilter.GaussianBlur(grow / 2.0)), dtype=np.uint8), np.asarray(grown, dtype=np.uint8))
    mask = np.maximum(soft, np.asarray(inner, dtype=np.uint8))
    out = io.BytesIO()
    Image.fromarray(mask, mode="L").save(out, format="PNG", optimize=False, compress_level=6)
    return {"png": out.getvalue(), "grow_px": grow, "cover": cover(boxes)}


# --- Move that carries the object (wave 4, owner decision) ------------------------------------------
# FLUX.3 Klein redrew a moved object at the target (a shorter tower, a seam); the wave-3 carry moved a
# slab of the attached house along (a union mask widened by notch closing) and smeared the source.  A
# Move now runs in three steps:
#   1. the object = the single SAM3 instance that best matches its box (IoU with the box, then area;
#      no union, no notch closing), else the box with a feathered edge;
#   2. a real Remove pass of the object (its own FLUX.3 Klein job on the ORIGINAL picture: Remove at
#      the source box, Keep boxes kept), pasted back only over the object grown by two rings inside
#      its source box (``move_remove_plan``);
#   3. the object's own pixels pasted onto that result at the target (scaled only for another size)
#      and one generative ring pass that takes only a narrow ring around the carried object and, when
#      the object stood on the ground (its mask reaches the source box's bottom), a ground strip with
#      a contact shadow under its base inside the target box (``move_carry``).
# The object's core is the source's own pixels; outside both boxes nothing changes.
MOVE_RING_FRACTION = 0.006  # the blended ring: 0.6 % of the long edge (6 px at 1024), at least 3 px
MOVE_REMOVE_RINGS = 2  # the Remove pass takes the object grown by two rings (the antialiased rim, a thin rod, a shadow line)
MOVE_TOUCH_FRACTION = 0.02  # a mask ending within 2 % of its box height above the box bottom stands on the ground
MOVE_GROUND_FRACTION = 0.1  # the ground under a grounded object: from 10 % of its landed height above its foot (room for a slope up to it), at least two rings
MOVE_EDGE_SIGMA = 1.0  # the carried edge: SAM3's hard mask softened like an antialiased edge (Gaussian, px)
MOVE_OCCLUDER_GROW_FRACTION = 0.1  # SAM3 looks for occluders in the Move box grown by 10 % of its long side
MOVE_HIDDEN_BAND_FRACTION = 0.05  # (w5, kept for callers) the object's width just above an occluder
MOVE_HIDDEN_ROWS = 8  # wave 6: a hidden side's edge = the outermost edge of the nearest 8 rows above with a free edge
MOVE_TIP_FRACTION = 0.03  # wave 6: the instance's top 3 % rows give the columns of a tip above it (a lightning rod)
MOVE_UNGHOST_STEP = 4.0  # wave 6: a plain-area pixel next to the removed area: luminance step to its neighbours <= 4 levels
MOVE_UNGHOST_REACH = 12  # wave 6: plain pixels per side for a row's reference line
# Wave 6 (owner decision c): the hidden part and the ground are one dedicated completion job after the carry,
# FLUX.2 Klein's Add path (the local flux2-klein-9b-kv-edit-v1 workflow) on a crop around the target box with
# 1.6x context (>= 1.5x), its mask = hidden part + ground zone; the ring pass then only blends the ring.
MOVE_COMPLETE_TARGET_ID = "flux2-klein-9b-kv-edit-v1"
MOVE_COMPLETE_CONTEXT = 1.6


def move_complete_prompt(desc: str) -> str:
    """The completion job's prompt (FLUX.2 Klein Add path: "Add <prompt> ..."), from the object's name only."""
    what = (desc or "object").strip()
    return (f"the missing part of the {what}, completing its shape exactly like its visible part, and the ground under "
            f"the {what}'s base continuing the nearby ground, so the {what} stands on the ground")


def move_ring_px(size: tuple[int, int]) -> int:
    return max(3, int(round(max(int(size[0]), int(size[1])) * MOVE_RING_FRACTION)))


def _inward_feather(width: int, height: int, rect: list[int], px: int) -> Any:
    """0..1 float array: 1 deep inside ``rect``, ramping to 0 at its border over ``px`` pixels, 0 outside."""
    import numpy as np  # type: ignore[import-not-found]

    x, y, w, h = rect
    out = np.zeros((height, width), dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    dx = np.minimum(xs + 1, w - xs)
    dy = np.minimum(ys + 1, h - ys)
    out[y:y + h, x:x + w] = np.clip(np.minimum(dy[:, None], dx[None, :]) / float(max(1, px)), 0.0, 1.0)
    return out


def _morph(mask: Any, px: int, grow: bool) -> Any:
    """A bool mask grown (``grow``) or shrunk by ``px`` pixels (square window)."""
    import numpy as np  # type: ignore[import-not-found]
    from PIL import Image, ImageFilter  # type: ignore[import-not-found]

    if px <= 0:
        return mask.copy()
    image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    image = image.filter((ImageFilter.MaxFilter if grow else ImageFilter.MinFilter)(2 * px + 1))
    return np.asarray(image, dtype=np.uint8) > 0


def _blur(mask: Any, radius: float) -> Any:
    """A 0..1 float mask blurred (Gaussian ``radius``)."""
    import numpy as np  # type: ignore[import-not-found]
    from PIL import Image, ImageFilter  # type: ignore[import-not-found]

    image = Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="L")
    return np.asarray(image.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255.0


def _png_l(mask: Any) -> bytes:
    import numpy as np  # type: ignore[import-not-found]
    from PIL import Image  # type: ignore[import-not-found]

    out = io.BytesIO()
    Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="L").save(out, format="PNG", compress_level=6)
    return out.getvalue()


def _rgb(png: bytes) -> Any:
    import numpy as np  # type: ignore[import-not-found]
    from PIL import Image  # type: ignore[import-not-found]

    with Image.open(io.BytesIO(png)) as image:
        image.load()
        return np.asarray(image.convert("RGB"), dtype=np.float32)


def _unique_id(wanted: str, taken: set[str]) -> str:
    out = wanted[:ID_MAX]
    while out in taken:
        out = out[:ID_MAX - 1] + "x"
    taken.add(out)
    return out


def move_fit(bbox: list[int], target: list[int]) -> list[int]:
    """Where the object lands: the source box scaled to fit the target (aspect kept), centred in it."""
    sw, sh = bbox[2], bbox[3]
    tx, ty, tw, th = target
    if (sw, sh) == (tw, th):
        return list(target)
    scale = min(tw / sw, th / sh)
    w, h = max(2, int(round(sw * scale))), max(2, int(round(sh * scale)))
    return [tx + (tw - w) // 2, ty + (th - h) // 2, w, h]


def move_object(size: tuple[int, int], box: Mapping[str, Any], object_png: bytes | None) -> tuple[Any, str]:
    """The Move box's object as a 0..1 float mask at picture size: the SAM3 instance (8-bit PNG at picture
    size) inside the box, as it is (no union, no notch closing), else the box with a feathered edge."""
    import numpy as np  # type: ignore[import-not-found]
    from PIL import Image  # type: ignore[import-not-found]

    width, height = int(size[0]), int(size[1])
    feathered = _inward_feather(width, height, box["bbox"], move_ring_px(size))
    if object_png:
        with Image.open(io.BytesIO(object_png)) as m:
            m.load()
            obj = np.asarray(m.convert("L").resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        obj = obj * (feathered > 0)
        if float((obj >= 0.5).sum()) >= 16:
            # SAM3 draws a hard edge; the picture's own edge is antialiased over about two pixels
            return np.minimum(_blur(obj, MOVE_EDGE_SIGMA), (feathered > 0).astype(np.float32)), "sam3"
    return feathered, "box"


def _hard_l(png: bytes | None, width: int, height: int) -> Any:
    """An 8-bit mask PNG as a bool array at picture size (None: all False)."""
    import numpy as np  # type: ignore[import-not-found]
    from PIL import Image  # type: ignore[import-not-found]

    if not png:
        return np.zeros((height, width), dtype=bool)
    with Image.open(io.BytesIO(png)) as m:
        m.load()
        return np.asarray(m.convert("L").resize((width, height), Image.Resampling.NEAREST), dtype=np.uint8) >= 128


def move_occluder_box(bbox: list[int], size: tuple[int, int]) -> list[int]:
    """Where SAM3 looks for occluders of a Move object: its box grown by 10 % of the long side, inside the picture."""
    width, height = int(size[0]), int(size[1])
    g = int(round(max(bbox[2], bbox[3]) * MOVE_OCCLUDER_GROW_FRACTION))
    x0, y0 = max(0, bbox[0] - g), max(0, bbox[1] - g)
    x1, y1 = min(width, bbox[0] + bbox[2] + g), min(height, bbox[1] + bbox[3] + g)
    return [x0, y0, x1 - x0, y1 - y0]


def move_occluder(size: tuple[int, int], obj: Any, occluder_png: bytes | None) -> Any:
    """The other SAM3 instances next to a Move object (bool, picture size): their union less the object grown
    by 2 px (its antialiased rim), without slivers thinner than two rings (SAM3 draws a neighbour's instance
    a few px over the object's outline: measured on the headland, a fringe along the whole tower).  They
    stay where they are (owner decision w5)."""
    width, height = int(size[0]), int(size[1])
    ring = move_ring_px(size)
    occ = _hard_l(occluder_png, width, height) & ~_morph(obj >= 0.5, 2, True)
    body = _morph(_morph(occ, ring, False), ring, True)  # opened by a ring: only the neighbour's body is left
    return occ & _morph(body, 2, True)


def move_hidden(obj: Any, occ: Any, bbox: list[int], ring: int) -> Any:
    """The part of a Move object an occluder hides (bool, picture size), following the object's own outline:
    in each row where a side of the object's outline meets the occluder, out to that side's edge as the rows
    above it had it (the nearest ``MOVE_HIDDEN_ROWS`` rows with a free edge); under the object's foot, rows the
    occluder fills (a wall in front of the base) continue the last free span, down to the box bottom.  Only on the
    occluder's pixels.  Nothing is carried there: the completion job generates it (owner decisions w5, w6 c).
    Wave 6: the w5 zone was a rectangle cut by the occluder, so the generated part took the house's outline
    (measured on the headland: a stepped annex instead of the tower's wall)."""
    import numpy as np  # type: ignore[import-not-found]

    hard = obj >= 0.5
    zone = np.zeros(hard.shape, dtype=bool)
    if not hard.any() or not (_morph(hard, ring, True) & occ).any():
        return zone
    near = _morph(occ, 3, True)
    height, width = hard.shape
    bx, by, bw, bh = bbox
    rows = np.nonzero(hard.any(axis=1))[0]
    top, bottom = int(rows.min()), int(rows.max())
    free_l: list[int] = []
    free_r: list[int] = []
    for y in range(top, bottom + 1):
        cols = np.nonzero(hard[y])[0]
        if cols.size == 0:
            continue
        lo, hi = int(cols.min()), int(cols.max())
        if near[y, max(0, lo - 1)] and free_l:
            zone[y, max(bx, min(free_l[-MOVE_HIDDEN_ROWS:])):lo] = True
        elif not near[y, max(0, lo - 1)]:
            free_l.append(lo)
        if near[y, min(width - 1, hi + 1)] and free_r:
            zone[y, hi + 1:min(bx + bw, max(free_r[-MOVE_HIDDEN_ROWS:]) + 1)] = True
        elif not near[y, min(width - 1, hi + 1)]:
            free_r.append(hi)
    if free_l and free_r:
        lo, hi = max(bx, int(np.median(free_l[-MOVE_HIDDEN_ROWS:]))), min(bx + bw - 1, int(np.median(free_r[-MOVE_HIDDEN_ROWS:])))
        for y in range(bottom + 1, by + bh):
            if occ[y, lo:hi + 1].mean() < 0.5:
                break
            zone[y, lo:hi + 1] = True
    return zone & occ & ~hard


def move_tip(hard: Any, bbox: list[int], ring: int) -> Any:
    """What stands above SAM3's instance inside the artist's box (bool): the columns of the instance's top rows
    (MOVE_TIP_FRACTION of its height) grown by a ring, from the box top down to the instance top.  Measured on
    the lighthouse: the lightning rod's tip (y 91-99) lies above the instance (y 113) and stayed at the source as
    a dark speck in every w4 / w5 render."""
    import numpy as np  # type: ignore[import-not-found]

    out = np.zeros(hard.shape, dtype=bool)
    rows = np.nonzero(hard.any(axis=1))[0]
    bx, by, bw, bh = bbox
    if rows.size == 0 or int(rows.min()) <= by:
        return out
    top = int(rows.min())
    cols = np.nonzero(hard[top:top + max(2, int(round((rows.max() - top + 1) * MOVE_TIP_FRACTION)))].any(axis=0))[0]
    out[by:top + ring, max(bx, int(cols.min()) - ring):min(bx + bw, int(cols.max()) + 1 + ring)] = True
    return out


def _row_reference(y: Any, place: Any, known: Any, reach: int) -> Any:
    """Per row over ``place``: a line between the median of the nearest ``reach`` known pixels left and right (NaN
    where a side has fewer than reach/2) - what a plain area (a sky) holds there without the object."""
    import numpy as np  # type: ignore[import-not-found]

    ref = np.full(y.shape, np.nan, dtype=np.float32)
    width = y.shape[1]
    for r in np.nonzero(place.any(axis=1))[0]:
        cols = np.nonzero(place[r])[0]
        lo, hi = int(cols.min()), int(cols.max())
        left = [c for c in range(lo - 1, max(-1, lo - 1 - reach * 3), -1) if known[r, c]][:reach]
        right = [c for c in range(hi + 1, min(width, hi + 1 + reach * 3)) if known[r, c]][:reach]
        ok_l, ok_r = len(left) >= reach // 2, len(right) >= reach // 2
        if not ok_l and not ok_r:
            continue
        if not (ok_l and ok_r):  # one side is a kept occluder (a roof): the plain side's level runs across (render round 2)
            ref[r] = float(np.median(y[r, left if ok_l else right]))
            continue
        vl, vr, xl, xr = float(np.median(y[r, left])), float(np.median(y[r, right])), float(np.mean(left)), float(np.mean(right))
        ref[r] = vl + (vr - vl) * np.clip((np.arange(width, dtype=np.float32) - xl) / max(1.0, xr - xl), 0.0, 1.0)
    return ref


def move_unghost(source_png: bytes, mask_png: bytes, composite_png: bytes, keep_out_png: bytes | None = None) -> dict[str, Any]:
    """The Remove pass result without a low-frequency ghost: over the removed area, in rows where the picture is
    plain on both sides (a sky: smooth pixels left and right, kept occluders not counted), the output's smooth
    drift from the row's reference line (``_row_reference`` per channel) is taken out; its fine texture stays.
    Measured on the headland (w5 rounds 2-3, w6 round 1): FLUX.3 Klein left a dark tower-shaped smear (-12 levels
    over 4600 px) in plain sky; the prefill could not serve as the reference there (the wall below pulled it dark).
    Outside the area nothing changes.  Returns ``composite_png`` and ``unghost_px`` (pixels moved by more than a level)."""
    import numpy as np  # type: ignore[import-not-found]
    from PIL import Image  # type: ignore[import-not-found]

    out = _rgb(composite_png)
    height, width = out.shape[:2]
    with Image.open(io.BytesIO(mask_png)) as m:
        m.load()
        alpha = np.asarray(m.convert("L"), dtype=np.float32) / 255.0
    hole = alpha > 0
    if not hole.any():
        return {"composite_png": composite_png, "unghost_px": 0}
    ring = move_ring_px((width, height))
    occ = _hard_l(keep_out_png, width, height) if keep_out_png else np.zeros((height, width), dtype=bool)
    lum = out.mean(axis=2)
    gy = np.abs(np.diff(lum, axis=0, append=lum[-1:]))
    gx = np.abs(np.diff(lum, axis=1, append=lum[:, -1:]))
    smooth = np.maximum(gx, gy) <= MOVE_UNGHOST_STEP
    known = ~_morph(hole, 2, True) & ~_morph(occ, 2, True) & smooth
    place = hole & ~occ
    fixed = out.copy()
    h = place.astype(np.float32)
    sig = float(ring * 2)
    hsum = np.maximum(_blur_f(h, sig), 1e-6)
    have = None
    for c in range(3):
        ref = _row_reference(out[..., c], place, known, MOVE_UNGHOST_REACH)
        ok = place & ~np.isnan(ref)
        have = ok if have is None else have & ok
        diff = np.where(ok, out[..., c] - np.nan_to_num(ref), 0.0).astype(np.float32)
        drift = _blur_f(diff, sig) / hsum
        fixed[..., c] = out[..., c] - np.where(ok, drift, 0.0) * alpha
    fixed[~(have & place)] = out[~(have & place)]
    result = np.clip(fixed + 0.5, 0, 255).astype(np.uint8)
    moved = int((np.abs(result.astype(np.int32) - np.clip(out + 0.5, 0, 255).astype(np.int32)).max(axis=2) > 1).sum())
    buf = io.BytesIO()
    Image.fromarray(result, mode="RGB").save(buf, format="PNG", compress_level=6)
    return {"composite_png": buf.getvalue(), "unghost_px": moved}


def _move_seed_hidden(base: Any, hidden: Any, hard: Any) -> Any:
    """The completion job's start inside the hidden part: per row, the carried object's pixels mirrored across the
    nearest object edge (its own wall and shading), blurred a little.  Measured w6 render round 2: from sky the
    FLUX.2 Klein Add pass kept sky inside the tower (headland: 1213 sky pixels inside the expected outline)."""
    import numpy as np  # type: ignore[import-not-found]

    out = base.copy()
    seed = np.zeros_like(base)
    have = np.zeros(hidden.shape, dtype=bool)
    for r in np.nonzero(hidden.any(axis=1))[0]:
        obj_cols = np.nonzero(hard[r])[0]
        if obj_cols.size == 0:
            rows = np.nonzero(hard.any(axis=1))[0]
            above = rows[rows < r]
            if above.size == 0:
                continue
            src_r = int(above.max())
            obj_cols = np.nonzero(hard[src_r])[0]
        else:
            src_r = int(r)
        lo, hi = int(obj_cols.min()), int(obj_cols.max())
        for c in np.nonzero(hidden[r])[0]:
            m = 2 * hi - c if c > hi else (2 * lo - c if c < lo else c)
            m = int(min(max(m, lo), hi))
            seed[r, c] = base[src_r, m]
            have[r, c] = True
    if have.any():
        soft = np.stack([_blur_f(seed[..., k] * have, 1.5) / np.maximum(_blur_f(have.astype(np.float32), 1.5), 1e-6) for k in range(3)], axis=-1)
        out[have] = soft[have]
    return out


def move_grounded(obj: Any, bbox: list[int]) -> bool:
    """The object stands on something: its mask reaches the source box's bottom (within 2 % of the box height)."""
    import numpy as np  # type: ignore[import-not-found]

    rows = np.nonzero((obj >= 0.5).any(axis=1))[0]
    if rows.size == 0:
        return False
    bottom = bbox[1] + bbox[3] - 1
    return int(rows.max()) >= bottom - max(2, int(round(bbox[3] * MOVE_TOUCH_FRACTION)))


def move_remove_plan(source_png: bytes, boxes: list[Mapping[str, Any]], object_masks: Mapping[str, Any] | None = None,
                     occluder_masks: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The Remove pass at the source, its own FLUX.3 Klein job on the original picture.

    Returns ``mask_png`` (where that job's output is pasted back: each Move object grown by two rings,
    soft outwards, inside its source box; a Move without a mask: its box feathered inwards), ``boxes``
    (each Move -> Remove at its source box, Keep boxes kept; Add / Change / Remove boxes wait for the
    ring pass) and per-move facts.  Other SAM3 instances next to the object (``occluder_masks``, e.g. the
    keeper's house in front of the tower) are never part of it: they stay as they are (owner decision w5).
    """
    import numpy as np  # type: ignore[import-not-found]

    source = _rgb(source_png)
    height, width = source.shape[:2]
    ring = move_ring_px((width, height))
    paste = np.zeros((height, width), dtype=np.float32)
    keep_out = np.zeros((height, width), dtype=bool)
    plan_boxes: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []
    masks = dict(object_masks or {})
    for box in boxes:
        if box["intent"] == "keep":
            plan_boxes.append(dict(box))
        if box["intent"] != "move":
            continue
        obj, source_kind = move_object((width, height), box, masks.get(box["id"]))
        occ = move_occluder((width, height), obj, dict(occluder_masks or {}).get(box["id"])) if source_kind == "sam3" else None
        if source_kind == "sam3":
            hard = obj >= 0.5
            grown = _morph(hard, MOVE_REMOVE_RINGS * ring, True)
            removal = np.maximum(_blur(grown.astype(np.float32), ring / 2.0), _morph(hard, ring, True).astype(np.float32))
            removal = np.maximum(removal, move_tip(hard, box["bbox"], ring).astype(np.float32))
            removal = np.minimum(removal, _inward_feather(width, height, box["bbox"], 2))
            removal[occ] = 0.0
        else:
            removal = obj
        paste = np.maximum(paste, removal)
        if occ is not None:
            keep_out |= _morph(occ, 2, True)
        plan_boxes.append({"id": box["id"], "intent": "remove", "desc": box["desc"], "bbox": list(box["bbox"])})
        hidden = move_hidden(obj, occ, box["bbox"], ring) if occ is not None else None
        stands = np.maximum(obj, hidden.astype(np.float32)) if hidden is not None else obj
        facts.append({"id": box["id"], "mask_source": source_kind, "remove_px": int((removal > 0).sum()),
                      "protected_px": int(occ.sum()) if occ is not None else 0,
                      "grounded": bool(source_kind == "sam3" and move_grounded(stands, box["bbox"]))})
    if not facts:
        raise BoxError("invalid_box", "A Move plan needs a Move box")
    return {"mask_png": _png_l(paste), "boxes": plan_boxes, "moves": facts, "ring_px": ring,
            "keep_out_png": _png_l(keep_out.astype(np.float32)) if keep_out.any() else None}


def move_blend(source_png: bytes, mask_png: bytes, output_png: bytes, crop: list[int]) -> dict[str, Any]:
    """Paste a model output (crop work size) back over the picture inside ``mask_png`` without a seam.

    The output is scaled to the crop, then a smooth offset field is added: on the band just outside the
    mask it equals (picture - output), inside it is that difference spread inwards (normalised Gaussian
    convolution, a membrane in effect).  Klein shifts the crop's light (+13 RGB in a sky); a global colour
    match leaves a step where the paste meets the untouched picture, the offset field removes it.  Outside
    the mask the picture stays byte for byte.  Returns ``composite_png`` and ``changed_outside_mask``.
    """
    import numpy as np  # type: ignore[import-not-found]
    from PIL import Image  # type: ignore[import-not-found]

    source = _rgb(source_png)
    height, width = source.shape[:2]
    with Image.open(io.BytesIO(mask_png)) as m:
        m.load()
        mask = np.asarray(m.convert("L"), dtype=np.float32) / 255.0
    x, y, w, h = crop
    with Image.open(io.BytesIO(output_png)) as image:
        image.load()
        piece = image.convert("RGB")
        if piece.size != (w, h):
            piece = piece.resize((w, h), Image.Resampling.LANCZOS)
    model = source.copy()
    model[y:y + h, x:x + w] = np.asarray(piece, dtype=np.float32)
    inside = mask > 0
    ring = move_ring_px((width, height))
    band = _morph(inside, ring, True) & ~inside
    band[:y] = band[y + h:] = False
    band[:, :x] = band[:, x + w:] = False
    sigma = float(max(ring * 4, int(round(max(w, h) * 0.03))))
    out = source.copy()
    if band.sum() >= 16:
        weight = band.astype(np.float32)
        wsum = _blur_f(weight, sigma)
        offset = np.zeros_like(source)
        for c in range(3):
            offset[..., c] = _blur_f((source[..., c] - model[..., c]) * weight, sigma) / np.maximum(wsum, 1e-6)
        model = model + offset
    a = mask[..., None]
    out = np.where(inside[..., None], model * a + source * (1.0 - a), source)
    composite = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    changed = np.any(composite != np.clip(source + 0.5, 0, 255).astype(np.uint8), axis=2)
    buf = io.BytesIO()
    Image.fromarray(composite, mode="RGB").save(buf, format="PNG", compress_level=6)
    return {"composite_png": buf.getvalue(), "changed_outside_mask": int((changed & ~inside).sum()), "changed_pixels": int(changed.sum()),
            "offset_sigma_px": sigma}


def move_prefill(source_png: bytes, mask_png: bytes, keep_out_png: bytes | None = None) -> bytes:
    """The picture the Remove pass sees: the area under ``mask_png`` filled with a smooth guess from its
    surroundings (normalised Gaussian convolution, coarse to fine), never row or column streaks.  FLUX.3
    Klein left a pale ghost of a tower it could still see; with the object gone it only refines the fill.
    ``keep_out_png`` (8-bit, picture size): pixels that never feed the guess (a kept occluder: the dark
    gable next to the tower pulled a dark smudge into the sky, measured on the headland)."""
    import numpy as np  # type: ignore[import-not-found]
    from PIL import Image  # type: ignore[import-not-found]

    source = _rgb(source_png)
    with Image.open(io.BytesIO(mask_png)) as m:
        m.load()
        hole = np.asarray(m.convert("L"), dtype=np.uint8) > 0
    known = (~hole).astype(np.float32)
    if keep_out_png:
        known *= (~_hard_l(keep_out_png, hole.shape[1], hole.shape[0])).astype(np.float32)
    ys, xs = np.nonzero(hole)
    out = source.copy()
    if xs.size == 0:
        return source_png
    span = float(max(4, min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)))
    fill = np.zeros_like(source)
    have = np.zeros(hole.shape, dtype=np.float32)
    sigma = span
    while sigma >= 2.0:  # coarse to fine: each finer scale takes over where it has enough known pixels nearby
        w = _blur_f(known, sigma)
        level = np.stack([_blur_f(source[..., c] * known, sigma) for c in range(3)], axis=-1) / np.maximum(w, 1e-6)[..., None]
        take = np.clip((w - 0.02) / 0.2, 0.0, 1.0) if sigma < span else np.ones_like(w)
        fill = fill * (1.0 - take[..., None]) + level * take[..., None]
        have = np.maximum(have, take)
        sigma /= 2.0
    out[hole] = fill[hole]
    buf = io.BytesIO()
    Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8), mode="RGB").save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def _blur_f(a: Any, sigma: float) -> Any:
    """Gaussian blur of a float array (any range), separable, edges clamped."""
    import numpy as np  # type: ignore[import-not-found]
    from PIL import Image  # type: ignore[import-not-found]

    step = int(sigma // 4)
    if step >= 2:  # a wide blur on a reduced grid (same result within a fraction of a level, far faster)
        h, w = a.shape
        small = Image.fromarray(a.astype(np.float32), mode="F").resize((max(1, w // step), max(1, h // step)), Image.Resampling.BOX)
        blurred = _blur_f(np.asarray(small, dtype=np.float32), sigma / step)
        return np.asarray(Image.fromarray(blurred, mode="F").resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32)
    r = max(1, int(round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1, dtype=np.float32) / float(sigma)) ** 2)
    k /= k.sum()
    p = np.pad(a, ((r, r), (0, 0)), mode="edge")
    v = sum(k[i] * p[i:i + a.shape[0]] for i in range(2 * r + 1))
    p = np.pad(v, ((0, 0), (r, r)), mode="edge")
    return sum(k[i] * p[:, i:i + a.shape[1]] for i in range(2 * r + 1)).astype(np.float32)


def move_carry(source_png: bytes, boxes: list[Mapping[str, Any]], object_masks: Mapping[str, Any] | None = None,
               removed_png: bytes | None = None, occluder_masks: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Carry every Move box's object onto ``removed_png`` (the Remove pass result; else the source).

    Returns ``base_png`` (the picture the ring pass sees: the removed picture with each object's own
    pixels at its target), ``mask_png`` (where the ring pass output is pasted back: a narrow ring around
    each carried object and, for a grounded object, a ground strip under its base, all inside the target
    box; 0 on the carried core, at the source and outside every target), ``core_png`` (the carried cores),
    ``model_boxes`` (Move -> Keep ``<id>_moved`` at the landing box, plus Change ``<id>_moved_ground``
    for the ground strip; the other boxes unchanged) and per-move facts.

    Wave 5 (owner decisions): the landing is exactly the artist's target box (no snapping).  The part of
    the object an occluder hid at the source (``move_hidden``) is carried nowhere.  Wave 6 (owner decision c):
    that part and the ground are generated by one completion job (``completion_png``: its mask, 0 on every
    carried core; ``completion``: one job per move with its target box and prompt), FLUX.2 Klein's Add path on
    a crop around the target box; the ring pass's ``mask_png`` is the ring only, never the completion area.  A grounded object (its mask, or its hidden part, reaches its box bottom)
    gets ground across the whole target box width from just above its foot down to the box bottom.  Each
    move's ``generated_png`` (8-bit, picture size) marks the pixels made by the model, not carried: the
    hidden part and the ground.
    """
    import numpy as np  # type: ignore[import-not-found]
    from PIL import Image  # type: ignore[import-not-found]

    source = _rgb(source_png)
    height, width = source.shape[:2]
    base = _rgb(removed_png) if removed_png else source.copy()
    if base.shape != source.shape:
        raise BoxError("invalid_box", "The removed picture must be the size of the picture")
    ring = move_ring_px((width, height))
    paste = np.zeros((height, width), dtype=np.float32)
    completion = np.zeros((height, width), dtype=np.float32)
    jobs: list[dict[str, Any]] = []
    core_all = np.zeros((height, width), dtype=bool)
    model_boxes: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []
    masks = dict(object_masks or {})
    taken = {b["id"] for b in boxes}
    for box in boxes:
        if box["intent"] != "move":
            model_boxes.append(dict(box))
            continue
        sx, sy, sw, sh = box["bbox"]
        tx, ty, tw, th = box["target_bbox"]
        landing = move_fit(box["bbox"], box["target_bbox"])
        lx, ly, lw, lh = landing
        obj, source_kind = move_object((width, height), box, masks.get(box["id"]))
        occ = move_occluder((width, height), obj, dict(occluder_masks or {}).get(box["id"])) if source_kind == "sam3" else None
        hidden_src = move_hidden(obj, occ, box["bbox"], ring) if occ is not None else np.zeros((height, width), dtype=bool)
        piece = Image.fromarray(np.clip(source[sy:sy + sh, sx:sx + sw] + 0.5, 0, 255).astype(np.uint8), mode="RGB")
        piece_alpha = Image.fromarray((obj[sy:sy + sh, sx:sx + sw] * 255.0 + 0.5).astype(np.uint8), mode="L")
        if (lw, lh) != (sw, sh):
            piece = piece.resize((lw, lh), Image.Resampling.LANCZOS)
            piece_alpha = piece_alpha.resize((lw, lh), Image.Resampling.BILINEAR)
        alpha = np.zeros((height, width), dtype=np.float32)
        alpha[ly:ly + lh, lx:lx + lw] = np.asarray(piece_alpha, dtype=np.float32) / 255.0
        a = alpha[ly:ly + lh, lx:lx + lw, None]
        base[ly:ly + lh, lx:lx + lw] = np.asarray(piece, dtype=np.float32) * a + base[ly:ly + lh, lx:lx + lw] * (1.0 - a)
        hard = alpha >= 0.5
        core = _morph(hard, ring, False)
        inside_target = np.zeros((height, width), dtype=bool)
        inside_target[ty:ty + th, tx:tx + tw] = True
        target_feather = _inward_feather(width, height, box["target_bbox"], 2)
        # the hidden part at the target: where the occluder covered the object, moved like the object
        hidden = np.zeros((height, width), dtype=bool)
        if hidden_src.any():
            h_piece = Image.fromarray((hidden_src[sy:sy + sh, sx:sx + sw] * 255).astype(np.uint8), mode="L")
            if (lw, lh) != (sw, sh):
                h_piece = h_piece.resize((lw, lh), Image.Resampling.NEAREST)
            hidden[ly:ly + lh, lx:lx + lw] = np.asarray(h_piece, dtype=np.uint8) >= 128
            hidden &= inside_target & ~core
            if occ is not None:
                hidden &= ~occ  # an occluder that stays where it is (a short or identity move) keeps covering
        ring_zone = _morph(hard, ring, True) & ~core & inside_target
        ring_alpha = np.where(ring_zone, np.maximum(_blur(ring_zone.astype(np.float32), ring / 3.0), 0.5), 0.0) * target_feather
        moved_id = _unique_id(f"{box['id']}_moved", taken)
        model_boxes.append({"id": moved_id, "intent": "keep", "desc": box["desc"], "bbox": list(landing)})
        hidden_rect = None
        hidden_alpha = np.zeros((height, width), dtype=np.float32)
        if hidden.any():
            zone = _morph(hidden, ring, True) & inside_target & ~core
            if occ is not None:
                zone &= ~occ
            hidden_alpha = np.where(zone, np.maximum(_blur(zone.astype(np.float32), ring / 3.0), hidden.astype(np.float32)), 0.0) * target_feather
            hys, hxs = np.nonzero(zone)
            hidden_rect = [int(hxs.min()), int(hys.min()), int(hxs.max() - hxs.min() + 1), int(hys.max() - hys.min() + 1)]
        if hidden.any():
            base = _move_seed_hidden(base, hidden, hard)
        stands = np.maximum(obj, hidden_src.astype(np.float32))
        grounded = source_kind == "sam3" and move_grounded(stands, box["bbox"])
        ground_rect = None
        ground_alpha = np.zeros((height, width), dtype=np.float32)
        body = hard | hidden
        if grounded and body.any():
            rows = np.nonzero(body.any(axis=1))[0]
            bottom = int(rows.max())
            g = max(2 * ring, int(round(lh * MOVE_GROUND_FRACTION)))
            # across the whole target box (the artist's box, no snapping): the ground meets the terrain at its sides
            x0, x1 = tx, tx + tw
            y0, y1 = max(ty, bottom - g + 1), ty + th  # down to the target box bottom: the ground runs on below the base
            if x1 - x0 >= 2 and y1 - y0 >= 2:
                ground_rect = [x0, y0, x1 - x0, y1 - y0]
                zone = np.zeros((height, width), dtype=bool)
                zone[y0:y1, x0:x1] = True
                zone &= ~core
                if occ is not None:
                    zone &= ~_morph(occ, 2, True)  # a kept occluder inside the target box (short or identity move) stays
                ground_alpha = np.maximum(_blur(zone.astype(np.float32), ring / 2.0), _morph(zone, ring // 2, False).astype(np.float32))
                ground_alpha = np.where(zone, ground_alpha, 0.0) * target_feather
                # the model's start under the foot: the ground the object stood on at its source (the Remove
                # result, moved like the object; kept occluders never travel), so the ring pass refines ground
                # instead of keeping sky (measured w5 round 2: a Change box over sky stayed sky).
                seed_rows = np.zeros((height, width), dtype=bool)
                seed_rows[bottom - g // 2:y1, x0:x1] = True
                seed = seed_rows & zone & ~body
                sys_, sxs = np.nonzero(seed)
                fy, fx = sys_ - (ly - sy), sxs - (lx - sx)
                ok = (fy >= 0) & (fy < height) & (fx >= 0) & (fx < width)
                if occ is not None:
                    ok[ok] &= ~occ[fy[ok], fx[ok]]
                removed_src = _rgb(removed_png) if removed_png else source
                fill = _blur(seed.astype(np.float32), ring / 2.0)[sys_[ok], sxs[ok], None]
                base[sys_[ok], sxs[ok]] = removed_src[fy[ok], fx[ok]] * fill + base[sys_[ok], sxs[ok]] * (1.0 - fill)
        # wave 6 (owner decision c): hidden part + ground go to the completion job, the ring pass blends the ring only
        complete_alpha = np.maximum(hidden_alpha, ground_alpha)
        complete_alpha[core] = 0.0
        completion = np.maximum(completion, complete_alpha)
        paste = np.maximum(paste, np.where(complete_alpha > 0, 0.0, ring_alpha))
        core_all |= core
        generated = (hidden | (ground_alpha >= 0.5)) & ~core
        if complete_alpha.any():
            jobs.append({"id": box["id"], "target_bbox": list(box["target_bbox"]), "desc": box["desc"] or "object",
                         "prompt": move_complete_prompt(box["desc"])})
        facts.append({"id": box["id"], "bbox": list(box["bbox"]), "target_bbox": list(box["target_bbox"]), "landing": landing,
                      "scale": round(lw / sw, 4), "mask_source": source_kind, "ring_px": ring, "core_px": int(core.sum()),
                      "ring_zone_px": int(ring_zone.sum()), "grounded": bool(grounded), "ground": ground_rect, "moved_id": moved_id,
                      "occluder_px": int(occ.sum()) if occ is not None else 0, "hidden_px": int(hidden.sum()), "hidden": hidden_rect,
                      "generated_px": int(generated.sum()), "generated_png": _png_l(generated.astype(np.float32))})
    paste[core_all] = 0.0
    completion[core_all] = 0.0
    out_base = io.BytesIO()
    Image.fromarray(np.clip(base + 0.5, 0, 255).astype(np.uint8), mode="RGB").save(out_base, format="PNG", compress_level=6)
    return {"base_png": out_base.getvalue(), "mask_png": _png_l(paste), "core_png": _png_l(core_all.astype(np.float32)),
            "model_boxes": model_boxes, "moves": facts, "ring_px": ring,
            "completion_png": _png_l(completion) if jobs else None, "completion": jobs}


def detection_ids(label: str, count: int, taken: set[str] | None = None) -> list[str]:
    """Ids for detected boxes, numbered left to right as NKD does: lighthouse -> lighthouse_1, lighthouse_2 ..."""
    prefix = re.sub(r"\W+", "_", (label or "").strip().lower()).strip("_")[:ID_MAX - 3] or "obj"
    if prefix[0].isdigit():
        prefix = ("obj_" + prefix)[:ID_MAX - 3]
    taken = set() if taken is None else taken
    out, n = [], 1
    while len(out) < count:
        candidate = f"{prefix}_{n}"
        if candidate not in taken:
            taken.add(candidate)
            out.append(candidate)
        n += 1
    return out


__all__ = ["BoxError", "INTENTS", "KEEP_ALL", "MASK_OPERATION", "MAX_BOXES", "MOVE_RING_FRACTION", "bfl_rows", "box_mask", "compile_prompt", "cover",
           "detection_ids", "move_blend", "move_carry", "move_fit", "move_prefill", "move_grounded", "move_hidden", "move_object", "move_occluder",
           "move_occluder_box", "move_remove_plan", "move_ring_px", "move_complete_prompt", "move_tip", "move_unghost", "MOVE_COMPLETE_TARGET_ID", "MOVE_COMPLETE_CONTEXT", "region_mask", "mask_plan",
           "normalize_boxes", "sentence", "to_grid"]
