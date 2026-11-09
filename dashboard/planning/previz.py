"""studio.previz/1: one previz set per scene (lane w20-plan2, Previz stage 1; planning board round 2 panels 6a/6b/6c).

Contract 2 (owner steer, architecture review): the set holds shared geometry and figure INSTANCES plus sparse per-shot
overrides of a figure's position, turn and pose. It never holds a camera: the shot record (shot_plan.py) is the only
camera authority. Units metres, y up, right-handed; a figure stands on the floor at (x, z), ``yaw`` in degrees
(0 = facing +z, toward the default camera), pose ``stand`` | ``sit``.

Document: {schema, revision, scenes {s-xxxxxxxx: {floor {w, d}, boxes [{id bx-, name, pos [x, y, z] (centre),
size [w, h, d], yaw}], figures [{id fig-, name, element el-?|null, version v-?|null, pos [x, z], yaw, pose}],
overrides {sh-: {fig-: {pos?, yaw?, pose?}}}, kept {sh-: {item, sig}}}}, proposals [{id pr-, by, scene, label, at,
state open|undone|dismissed, before {set, cameras {sh-: camera|null}}, after {...}}]}.

Contract 3: a Hermes change is ONE proposal with its before/after delta (the scene's set and every shot camera it
touched); ``undo_check`` restores only while every touched value still equals ``after`` (else proposal_conflict).
Pure; the route unit owns the files, the lock and the shot record.
"""
from __future__ import annotations

import copy
import math
import re
import secrets
from typing import Any

SCHEMA = "studio.previz/1"
SCENE_ID_RE = re.compile(r"^s-[0-9a-f]{8}$")
SHOT_ID_RE = re.compile(r"^sh-[0-9a-f]{8}$")
FIG_RE = re.compile(r"^fig-[0-9a-f]{8}$")
BOX_RE = re.compile(r"^bx-[0-9a-f]{8}$")
ELEMENT_ID_RE = re.compile(r"^el-[0-9a-f]{8}$")
VERSION_RE = re.compile(r"^v-[0-9a-f]{8}$")
ITEM_RE = re.compile(r"^(imgjob_[0-9a-f]{32}|file:[0-9a-f]{64})$")
KEPT_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,3})?Z$")
POSES = ("stand", "sit", "walk")  # w23-pvlook: walk (FrameForge / Cine Tracer posable actors; a stride for blocking)
# w23-pvlook (study: FrameForge sun/key, Cine Tracer time of day, Blender passepartout): the set's light (auto = from the
# scene heading's INT/EXT and time of day; or a fixed time) and the passepartout's opacity outside the frame (0-1).
LIGHTS = ("auto", "day", "dawn", "dusk", "night")
WORLD = 100.0
MAX_BOXES, MAX_FIGURES, MAX_PROPOSALS = 200, 40, 20
CAMERA_WORDS = {"camera", "cameras", "lens", "lens_mm", "start", "end", "at"}


class PrevizRefusal(Exception):
    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code, self.message = reason_code, message


def empty() -> dict[str, Any]:
    return {"schema": SCHEMA, "revision": 0, "scenes": {}, "proposals": []}


def valid_document(doc: Any) -> bool:
    return isinstance(doc, dict) and doc.get("schema") == SCHEMA and isinstance(doc.get("scenes"), dict)


def default_set() -> dict[str, Any]:
    return {"floor": {"w": 8.0, "d": 8.0}, "boxes": [], "figures": [], "overrides": {}, "kept": {}}


def scene_of(doc: dict[str, Any], scene: str) -> dict[str, Any]:
    return copy.deepcopy(doc["scenes"].get(scene) or default_set())


def _num(v: Any, lo: float, hi: float, where: str) -> float:
    if type(v) not in (int, float) or not math.isfinite(v) or not lo <= v <= hi:
        raise PrevizRefusal("set_invalid", f"{where} is a number from {lo} to {hi}.")
    return float(v)


def _vec(v: Any, n: int, where: str, lo: float = -WORLD, hi: float = WORLD) -> list[float]:
    if not isinstance(v, list) or len(v) != n:
        raise PrevizRefusal("set_invalid", f"{where} is a list of {n} numbers (metres).")
    return [_num(x, lo, hi, where) for x in v]


def _name(v: Any, where: str) -> str:
    if not isinstance(v, str) or len(v) > 60 or any(ord(c) < 32 for c in v):
        raise PrevizRefusal("set_invalid", f"{where} is one line of at most 60 characters.")
    return " ".join(v.split())


def _no_camera(raw: dict[str, Any], where: str) -> None:
    hit = CAMERA_WORDS & set(raw)
    if hit:
        raise PrevizRefusal("camera_not_here", f"{where} holds no camera ({sorted(hit)[0]}): a shot's camera lives in its shot record (continuo_previz_camera).")


def _figure(raw: Any, old: dict[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PrevizRefusal("figure_invalid", "A figure is an object.")
    _no_camera(raw, "A figure")
    if set(raw) - {"id", "name", "element", "version", "pos", "yaw", "pose"}:
        raise PrevizRefusal("figure_invalid", "A figure is {id?, name, element?, version?, pos [x, z], yaw, pose}.")
    fig = dict(old or {"id": "fig-" + secrets.token_hex(4), "name": "", "element": None, "version": None, "pos": [0.0, 0.0], "yaw": 0.0, "pose": "stand"})
    if "id" in raw and raw["id"] is not None:
        if not isinstance(raw["id"], str) or not FIG_RE.match(raw["id"]):
            raise PrevizRefusal("figure_invalid", "A figure id is fig-<8 hex>.")
        fig["id"] = raw["id"]
    if "name" in raw:
        fig["name"] = _name(raw["name"], "figure.name")
    for key, rx in (("element", ELEMENT_ID_RE), ("version", VERSION_RE)):
        if key in raw:
            if raw[key] is not None and (not isinstance(raw[key], str) or not rx.match(raw[key])):
                raise PrevizRefusal("figure_invalid", f"figure.{key} is an id from continuo_elements_read or null.")
            fig[key] = raw[key]
    if "pos" in raw:
        fig["pos"] = _vec(raw["pos"], 2, "figure.pos [x, z]")
    if "yaw" in raw:
        fig["yaw"] = _num(raw["yaw"], -360, 360, "figure.yaw (degrees)")
    if "pose" in raw:
        if raw["pose"] not in POSES:
            raise PrevizRefusal("figure_invalid", "figure.pose is stand, sit or walk.")
        fig["pose"] = raw["pose"]
    return fig


def _box(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PrevizRefusal("box_invalid", "A box is an object.")
    _no_camera(raw, "A box")
    if set(raw) - {"id", "name", "pos", "size", "yaw"}:
        raise PrevizRefusal("box_invalid", "A box is {id?, name, pos [x, y, z], size [w, h, d], yaw}.")
    bid = raw.get("id") or "bx-" + secrets.token_hex(4)
    if not isinstance(bid, str) or not BOX_RE.match(bid):
        raise PrevizRefusal("box_invalid", "A box id is bx-<8 hex>.")
    return {"id": bid, "name": _name(raw.get("name", "Box"), "box.name"), "pos": _vec(raw.get("pos", [0, 0.5, 0]), 3, "box.pos"),
            "size": _vec(raw.get("size", [1, 1, 1]), 3, "box.size", 0.01, 100), "yaw": _num(raw.get("yaw", 0), -360, 360, "box.yaw")}


def _override(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) - {"pos", "yaw", "pose"}:
        raise PrevizRefusal("override_invalid", "A per-shot override is {pos?, yaw?, pose?}.")
    out: dict[str, Any] = {}
    if "pos" in raw:
        out["pos"] = _vec(raw["pos"], 2, "override.pos [x, z]")
    if "yaw" in raw:
        out["yaw"] = _num(raw["yaw"], -360, 360, "override.yaw")
    if "pose" in raw:
        if raw["pose"] not in POSES:
            raise PrevizRefusal("override_invalid", "override.pose is stand, sit or walk.")
        out["pose"] = raw["pose"]
    return out


def apply_set(doc: dict[str, Any], scene: str, patch: dict[str, Any]) -> dict[str, Any]:
    """Validate and apply a set patch {floor?, boxes? (full list), figures? (full list), overrides? {shot: {fig: o|null}},
    kept? {shot: {item, sig}|null}} to one scene; returns the new set (nothing stored on a refusal)."""
    if not isinstance(scene, str) or not SCENE_ID_RE.match(scene):
        raise PrevizRefusal("scene_invalid", "A scene id is s-<8 hex> (continuo_script_read).")
    if not isinstance(patch, dict):
        raise PrevizRefusal("set_invalid", "The set patch is an object.")
    _no_camera(patch, "The set")
    if set(patch) - {"floor", "boxes", "figures", "overrides", "kept", "light", "matte"}:
        raise PrevizRefusal("set_invalid", "The set is {floor?, boxes?, figures?, overrides?, kept?, light?, matte?}.")
    s = scene_of(doc, scene)
    if "light" in patch:
        if patch["light"] not in LIGHTS:
            raise PrevizRefusal("set_invalid", "light is auto (from the scene heading), day, dawn, dusk or night.")
        s["light"] = patch["light"]
    if "matte" in patch:
        s["matte"] = round(_num(patch["matte"], 0, 1, "matte (the passepartout's opacity)"), 3)
    if "floor" in patch:
        f = patch["floor"]
        if not isinstance(f, dict) or set(f) != {"w", "d"}:
            raise PrevizRefusal("set_invalid", "floor is {w, d} in metres.")
        s["floor"] = {"w": _num(f["w"], 1, 200, "floor.w"), "d": _num(f["d"], 1, 200, "floor.d")}
    if "boxes" in patch:
        if not isinstance(patch["boxes"], list) or len(patch["boxes"]) > MAX_BOXES:
            raise PrevizRefusal("box_invalid", f"boxes is a list of at most {MAX_BOXES}.")
        s["boxes"] = [_box(b) for b in patch["boxes"]]
    if "figures" in patch:
        if not isinstance(patch["figures"], list) or len(patch["figures"]) > MAX_FIGURES:
            raise PrevizRefusal("figure_invalid", f"figures is a list of at most {MAX_FIGURES}.")
        old = {f["id"]: f for f in s["figures"]}
        s["figures"] = [_figure(f, old.get(f.get("id")) if isinstance(f, dict) else None) for f in patch["figures"]]
    ids = [f["id"] for f in s["figures"]] + [b["id"] for b in s["boxes"]]
    if len(set(ids)) != len(ids):
        raise PrevizRefusal("set_invalid", "Every figure and box appears once.")
    if "overrides" in patch:
        if not isinstance(patch["overrides"], dict):
            raise PrevizRefusal("override_invalid", "overrides is {shot: {figure: override|null}}.")
        for shot, per in patch["overrides"].items():
            if not SHOT_ID_RE.match(str(shot)) or not isinstance(per, dict):
                raise PrevizRefusal("override_invalid", "overrides is {sh-…: {fig-…: override|null}}.")
            slot = s["overrides"].setdefault(shot, {})
            for fig, o in per.items():
                if not FIG_RE.match(str(fig)):
                    raise PrevizRefusal("override_invalid", "An override names a figure id fig-<8 hex>.")
                if o is None:
                    slot.pop(fig, None)
                else:
                    slot[fig] = _override(o)
            if not slot:
                s["overrides"].pop(shot, None)
    if "kept" in patch:
        if not isinstance(patch["kept"], dict):
            raise PrevizRefusal("set_invalid", "kept is {shot: {item, sig}|null}.")
        for shot, k in patch["kept"].items():
            if not SHOT_ID_RE.match(str(shot)):
                raise PrevizRefusal("set_invalid", "kept names shot ids.")
            if k is None:
                s["kept"].pop(shot, None)
            elif not isinstance(k, dict) or set(k) - {"item", "sig", "at"} or not ITEM_RE.match(str(k.get("item")))                     or not re.fullmatch(r"[0-9a-f]{8,64}", str(k.get("sig"))) or ("at" in k and not KEPT_AT_RE.match(str(k["at"]))):
                raise PrevizRefusal("set_invalid", "kept is {item: a Media item id, sig: hex, at?: UTC time}.")
            else:
                # w21-pv item 6: ``at`` (when the frame was kept) feeds the footer's "Previz frame of 1.2 · 11:42".
                s["kept"][shot] = {"item": k["item"], "sig": k["sig"], **({"at": k["at"]} if "at" in k else {})}
    known = {f["id"] for f in s["figures"]}
    for shot in list(s["overrides"]):
        s["overrides"][shot] = {f: o for f, o in s["overrides"][shot].items() if f in known}
        if not s["overrides"][shot]:
            del s["overrides"][shot]
    return s


# --- blocking (Hermes' continuo_previz_block_scene): a starting set and cameras from the script -----------------------
DISTANCE = {"ECU": 0.45, "CU": 0.8, "MCU": 1.15, "MCS": 1.5, "MS": 1.9, "MWS": 2.8, "FS": 3.4, "WS": 5.2, "EWS": 9.0, "INSERT": 0.55}
# Lane w22-pv item 3: a close size on a named figure is framed like the page's defaultCamera (previz-eval.js SPAN, the
# usual size chart): the span in metres from the top of the head down, shown by the lens' vertical field of view on a
# 36 mm 16:9 sensor; CU = head 0.62 of the frame, chin above the bottom edge (the distance table read a CU as an MCU).
SPAN = {"ECU": 0.18, "CU": 0.37, "MCU": 0.56, "INSERT": 0.34, "MCS": 0.72, "MS": 0.9, "MWS": 1.3}
_TOP = {"stand": 1.735, "sit": 1.415, "walk": 1.735}
# Lane w23-pvlook (d), the same table as previz-eval.js HEADROOM (owner read: the 85 mm CU had no headroom): the air above
# the head as a share of the frame height per size.
HEADROOM = {"CU": 0.08, "MCU": 0.1, "MCS": 0.1, "MS": 0.1, "MWS": 0.1}


def _framed(size: str | None, lens: float, figure: dict[str, Any]) -> tuple[float, float] | None:
    """(distance, aim height) that frames ``figure`` at ``size`` with ``lens``; None for sizes the table does not fix."""
    if size not in SPAN:
        return None
    top = _TOP.get(figure.get("pose") or "stand", _TOP["stand"])
    span = min(SPAN[size], top + 0.15)
    vfov = 2 * math.atan((36 / (16 / 9)) / 2 / lens)
    if size in HEADROOM:  # w23-pvlook (d): the top of the head sits HEADROOM of the frame height below its top edge
        return span / (2 * math.tan(vfov / 2)), max(top * 0.45, top - span * (0.5 - HEADROOM[size]))
    return span / (2 * math.tan(vfov / 2)), max(top * 0.45, top + span * 0.06 - span / 2)


def _fig_pos(i: int, n: int) -> list[float]:
    if n == 1:
        return [0.0, 0.0]
    span = min(3.0, 1.2 * (n - 1))
    x = -span / 2 + span * i / (n - 1)
    return [round(x, 3), round(-0.3 * abs(x), 3)]


def _facing(fig: dict[str, Any], figures: list[dict[str, Any]]) -> float:
    """Yaw (degrees, 0 faces +z) toward the other figures' centre, cheated 30 degrees toward the room's front."""
    others = [f for f in figures if f is not fig]
    cx = sum(f["pos"][0] for f in others) / len(others) - fig["pos"][0]
    cz = sum(f["pos"][1] for f in others) / len(others) - fig["pos"][1]
    a = math.degrees(math.atan2(cx, cz))
    a -= math.copysign(min(30.0, abs(a)), a)
    return round(a, 1)


def _lead(target: dict[str, Any], figures: list[dict[str, Any]], size: str | None, lens: float, dist: float) -> float:
    """Sideways aim offset (metres along +x) for a medium single on ``target`` with a partner within 4 m: the aim moves
    toward the partner by 0.18 of the frame width, so the subject sits on the third away from the partner (it looks
    into the frame) and the partner's near shoulder enters the frame edge. Close sizes and lone figures: 0."""
    if size not in ("MCS", "MS", "MWS", "FS"):
        return 0.0
    near = sorted((abs(f["pos"][0] - target["pos"][0]), f["pos"][0] - target["pos"][0]) for f in figures
                  if f is not target and math.hypot(f["pos"][0] - target["pos"][0], f["pos"][1] - target["pos"][1]) <= 4.0)
    if not near or near[0][0] < 0.2:
        return 0.0
    return math.copysign(0.18 * dist * 36 / lens, near[0][1])


def block(current: dict[str, Any], scene_heading: str, people: list[dict[str, Any]], shots: list[dict[str, Any]],
          cameras: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """A starting blocking: figures for the scene's people that have none yet (facing the room's front), a floor and a
    back wall when the set is empty, and a camera for every shot without one: placed by its Size at a distance in front
    of the figure the shot names (else the group), its Angle sets the height, a Movement gives an End. Returns
    (new set, {shot: camera}) for the shots it gives a camera; existing figures, boxes and cameras stay."""
    s = copy.deepcopy(current)
    if not s["boxes"]:
        s["boxes"].append(_box({"name": "Back wall", "pos": [0, 1.4, -3.0], "size": [8, 2.8, 0.15], "yaw": 0}))
        if scene_heading.upper().startswith(("INT", "I/E")):
            s["boxes"].append(_box({"name": "Table", "pos": [1.6, 0.37, -1.4], "size": [1.2, 0.74, 0.7], "yaw": 0}))
    have = {f["name"].casefold() for f in s["figures"]}
    new = [p for p in people if p["name"].casefold() not in have]
    total = len(s["figures"]) + len(new)
    for i, p in enumerate(new, start=len(s["figures"])):
        s["figures"].append(_figure({"name": p["name"], "element": p.get("element"), "version": p.get("version"), "pos": _fig_pos(i, total), "yaw": 0, "pose": "stand"}))
    # w23-pvlook2 item 4 (owner read: "both figures face the lens: a blocked two-hander faces each other"): the figures
    # this block places turn toward the others (their centre), cheated 30 degrees toward the room's front (+z) so the
    # cameras there still see three-quarter faces (the usual stage cheat); figures the artist placed keep their turn.
    placed = {id(f) for f in s["figures"][len(s["figures"]) - len(new):]}
    for f in s["figures"]:
        if id(f) in placed and len(s["figures"]) > 1:
            f["yaw"] = _facing(f, s["figures"])
    out: dict[str, Any] = {}
    for shot in shots:
        if cameras.get(shot["id"]):
            continue
        d = shot.get("details") or {}
        text = shot.get("text", "").upper()
        target = next((f for f in s["figures"] if f["name"] and f["name"].upper() in text), None)
        if target is not None:
            tx, tz, ty = target["pos"][0], target["pos"][1], 1.55 if d.get("size") in ("ECU", "CU", "MCU", "MCS") else 1.2
        elif s["figures"]:
            tx = sum(f["pos"][0] for f in s["figures"]) / len(s["figures"])
            tz, ty = sum(f["pos"][1] for f in s["figures"]) / len(s["figures"]), 1.2
        else:
            tx, tz, ty = 0.0, 0.0, 1.0
        dist = DISTANCE.get(d.get("size"), DISTANCE["MS"]) * (float(d.get("lens") or 35) / 35)
        framed = _framed(d.get("size"), float(d.get("lens") or 35), target) if target is not None else None
        if framed is not None:
            dist, ty = max(framed[0], 0.52), framed[1]
        h = {"high": ty + 1.2, "low": max(0.35, ty - 0.8), "overhead": ty + 3.0, "ground": 0.25}.get(d.get("angle"), ty)
        if framed is not None and d.get("size") in HEADROOM and d.get("angle") in ("high", "low", "ground"):
            # w23-pvlook2 item 6 (owner read: "the Low angle barely reads"): a tilted camera aims so the head keeps the
            # size's headroom (as the page's aimAt), so a Low looks up at the figure instead of at its chest
            top = _TOP.get(target.get("pose") or "stand", _TOP["stand"])
            vh = math.atan((36 / (16 / 9)) / 2 / float(d.get("lens") or 35))
            up = math.atan2(top - h, dist) - math.atan((0.5 - HEADROOM[d["size"]]) * 2 * math.tan(vh))
            ty = h + math.tan(up) * dist
        # w23-pvlook2 item 4 (owner read: "Ines vanishes from the 1.3 MS"): a medium single on one figure of several
        # leads toward its partner (_lead), so the partner stays in the frame's edge and the subject looks into the frame
        lead = _lead(target, s["figures"], d.get("size"), float(d.get("lens") or 35), dist) if target is not None else 0.0
        start = {"pos": [round(tx, 3), round(h, 3), round(tz + dist, 3)], "at": [round(tx + lead, 3), round(ty, 3), round(tz, 3)]}
        cam: dict[str, Any] = {"lens_mm": d.get("lens") or 35, "start": start}
        move = d.get("move")
        if move in ("dolly_in", "dolly_out", "truck", "pedestal", "crane", "arc"):
            px, py, pz = start["pos"]
            end = {"dolly_in": [px, py, tz + dist * 0.55], "dolly_out": [px, py, tz + dist * 1.6], "truck": [px + 1.2, py, pz],
                   "pedestal": [px, py + 0.6, pz], "crane": [px, py + 1.8, pz + 0.6],
                   "arc": [round(tx + math.sin(0.6) * dist, 3), py, round(tz + math.cos(0.6) * dist, 3)]}[move]
            cam["end"] = {"pos": [round(v, 3) for v in end], "at": list(start["at"])}
        out[shot["id"]] = cam
    return s, out


# --- proposals (contract 3) -------------------------------------------------------------------------------------------
def propose(doc: dict[str, Any], scene: str, by: str, label: str, at: str, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    for p in doc.setdefault("proposals", []):
        if p["state"] == "open" and p["scene"] == scene:
            p["state"] = "accepted"  # a newer change on the same scene closes the older one (it stays as history)
    proposal = {"id": "pr-" + secrets.token_hex(4), "by": by, "scene": scene, "label": label[:120], "at": at, "state": "open",
                "before": copy.deepcopy(before), "after": copy.deepcopy(after)}
    doc["proposals"] = (doc["proposals"] + [proposal])[-MAX_PROPOSALS:]
    return proposal


def find_proposal(doc: dict[str, Any], pid: str | None) -> dict[str, Any]:
    open_ = [p for p in doc.get("proposals") or [] if p["state"] == "open"]
    hit = next((p for p in open_ if p["id"] == pid), None) if pid else (open_[-1] if open_ else None)
    if hit is None:
        raise PrevizRefusal("proposal_unknown", "No open Hermes change with that id (continuo_previz_read proposals).")
    return hit


def undo_check(proposal: dict[str, Any], current_set: dict[str, Any], current_cameras: dict[str, Any]) -> None:
    """Undo restores only what is still exactly as the proposal left it; anything changed since is a conflict."""
    changed = []
    if proposal["after"].get("set") is not None and _strip(current_set) != _strip(proposal["after"]["set"]):
        changed.append("the set")
    for shot, cam in (proposal["after"].get("cameras") or {}).items():
        if current_cameras.get(shot) != cam:
            changed.append(shot)
    if changed:
        raise PrevizRefusal("proposal_conflict", f"Changed since Hermes' change: {', '.join(changed)}. Nothing was undone.")


def _strip(s: dict[str, Any] | None) -> dict[str, Any] | None:
    return None if s is None else {k: v for k, v in s.items() if k != "kept"}


def check_document(doc: Any) -> str | None:
    """None when ``doc`` is a well-formed studio.previz/1 document, else the problem (architecture review finding 2):
    every scene is re-validated through the same rules a write uses, so a stored set equals what apply_set would keep."""
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        return "not a studio.previz/1 document"
    if set(doc) - {"schema", "revision", "scenes", "proposals"}:
        return "unknown top-level field"
    if type(doc.get("revision")) is not int or doc["revision"] < 0:
        return "revision is not an integer >= 0"
    scenes, props = doc.get("scenes"), doc.get("proposals", [])
    if not isinstance(scenes, dict) or len(scenes) > 2000 or not isinstance(props, list) or len(props) > MAX_PROPOSALS:
        return "scenes / proposals are not bounded"
    try:
        for scene, s in scenes.items():
            err = _check_set(scene, s)
            if err:
                return err
        for p in props:
            if not isinstance(p, dict) or set(p) != {"id", "by", "scene", "label", "at", "state", "before", "after"}:
                return "a proposal has the wrong fields"
            if not re.fullmatch(r"pr-[0-9a-f]{8}", str(p["id"])) or p["by"] not in ("artist", "hermes") or not SCENE_ID_RE.match(str(p["scene"])) \
                    or p["state"] not in ("open", "accepted", "undone", "dismissed") or not isinstance(p["label"], str) or len(p["label"]) > 120:
                return "a proposal is invalid"
            for side in (p["before"], p["after"]):
                if not isinstance(side, dict) or set(side) - {"set", "cameras"}:
                    return "a proposal delta is invalid"
                if side.get("set") is not None and _check_set(p["scene"], side["set"]):
                    return "a proposal set is invalid"
                cams = side.get("cameras") or {}
                if not isinstance(cams, dict) or not all(SHOT_ID_RE.match(str(k)) and (v is None or isinstance(v, dict)) for k, v in cams.items()):
                    return "a proposal camera delta is invalid"
    except PrevizRefusal as exc:
        return exc.message
    return None


def _check_set(scene: Any, s: Any) -> str | None:
    # w23-pvlook: light and matte are optional (a set written before them has neither: auto light, the default matte)
    if not isinstance(scene, str) or not SCENE_ID_RE.match(scene) or not isinstance(s, dict) or not {"floor", "boxes", "figures", "overrides", "kept"} <= set(s) <= {"floor", "boxes", "figures", "overrides", "kept", "light", "matte"}:
        return f"scene {str(scene)[:20]} is not a set"
    probe = {"scenes": {}}
    again = apply_set(probe, scene, {k: copy.deepcopy(s[k]) for k in ("floor", "boxes", "figures", "overrides", "kept", "light", "matte") if k in s})
    if again != s:
        return f"scene {scene} holds values a write would not keep"
    return None


def _proposal_shots(p: dict[str, Any]) -> list[str]:
    """The shots a proposal changes: its cameras and (lane w24-rest) the shots whose per-shot figure override changed."""
    shots = set((p["after"].get("cameras") or {}).keys())
    before, after = (p["before"].get("set") or {}).get("overrides") or {}, (p["after"].get("set") or {}).get("overrides") or {}
    shots |= {shot for shot in set(before) | set(after) if before.get(shot) != after.get(shot)}
    return sorted(shots)


def public(doc: dict[str, Any], scene: str | None = None) -> dict[str, Any]:
    scenes = {k: v for k, v in doc["scenes"].items() if scene is None or k == scene}
    if scene is not None and scene not in scenes:
        scenes[scene] = default_set()
    props = [{k: p[k] for k in ("id", "by", "scene", "label", "at", "state")} | {"shots": _proposal_shots(p)}
             for p in doc.get("proposals") or [] if scene is None or p["scene"] == scene]
    return {"schema": SCHEMA, "revision": int(doc.get("revision") or 0), "scenes": copy.deepcopy(scenes), "proposals": props,
            "conventions": {"units": "metres", "up": "+y", "handedness": "right", "yaw": "degrees, 0 faces +z"}}
