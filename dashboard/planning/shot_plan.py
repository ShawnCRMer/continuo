"""studio.shot-plan/1: one shot record per project, keyed by the script's shot id (lane w20-plan item 2).

Pure functions over the stored document; the route unit (routes/planning/plan_1.py) owns the file, the lock and the
Media marks. One record feeds every view (shot list beside the script, Board, shot details, later Previz): the
OpenShotDesigner idea, written here from scratch. Vocabulary after StudioBinder's shot list and ShotDeck; the owner's
fixes: MCS = Medium Close Shot, Full Shot exists, Drone is equipment (not a movement), lens choices 18-135 + custom.

Document: {schema, revision, shots: {sh-xxxxxxxx: {details, camera, pictures [{item, role}], elements [{key, version}],
joined [item], by, at}}, board_order: {scene id: [shot id]}}. ``joined`` lists the Media-marked pictures (shot:<id>)
already taken into the record once, so a picture the artist takes off never comes back from its mark.
"""
from __future__ import annotations

import math
import re
from typing import Any

SCHEMA = "studio.shot-plan/1"
SHOT_ID_RE = re.compile(r"^sh-[0-9a-f]{8}$")
SCENE_ID_RE = re.compile(r"^s-[0-9a-f]{8}$")
ITEM_RE = re.compile(r"^(imgjob_[0-9a-f]{32}|file:[0-9a-f]{64}|folder:[0-9a-f]{32}|[a-z][a-z0-9_]{1,30}_[0-9a-f]{32})$")
ELEMENT_KEY_MAX = 60
VERSION_RE = re.compile(r"^v-[0-9a-f]{8}$")
MAX_PICTURES = 60
MAX_ELEMENTS = 40
MAX_SHOTS = 5000
NOTES_MAX = 2000

VOCABULARY: dict[str, list[dict[str, str]]] = {
    "size": [{"code": "ECU", "name": "Extreme Close-up"}, {"code": "CU", "name": "Close-up"},
             {"code": "MCU", "name": "Medium Close-up"}, {"code": "MCS", "name": "Medium Close Shot"},
             {"code": "MS", "name": "Medium Shot"}, {"code": "MWS", "name": "Medium Wide Shot"},
             {"code": "FS", "name": "Full Shot"}, {"code": "WS", "name": "Wide Shot"},
             {"code": "EWS", "name": "Extreme Wide Shot"}, {"code": "INSERT", "name": "Insert"}],
    # Lane w20-plan2 A3 (owner review): the field reads "Framing" on the page; Cutaway replaces Establishing.
    "type": [{"code": "single", "name": "Single"}, {"code": "two_shot", "name": "Two shot"},
             {"code": "ots", "name": "Over the shoulder"}, {"code": "pov", "name": "POV"},
             {"code": "insert", "name": "Insert"}, {"code": "cutaway", "name": "Cutaway"}, {"code": "group", "name": "Group"}],
    "angle": [{"code": "eye", "name": "Eye level"}, {"code": "high", "name": "High"}, {"code": "low", "name": "Low"},
              {"code": "overhead", "name": "Overhead"}, {"code": "dutch", "name": "Dutch"}, {"code": "ground", "name": "Ground"}],
    "move": [{"code": "static", "name": "Static"}, {"code": "pan", "name": "Pan"}, {"code": "tilt", "name": "Tilt"},
             {"code": "dolly_in", "name": "Dolly in"}, {"code": "dolly_out", "name": "Dolly out"}, {"code": "truck", "name": "Truck"},
             {"code": "pedestal", "name": "Pedestal"}, {"code": "crane", "name": "Crane"}, {"code": "arc", "name": "Arc"},
             {"code": "zoom", "name": "Zoom"}, {"code": "handheld", "name": "Handheld"}],
    "equipment": [{"code": "tripod", "name": "Tripod"}, {"code": "dolly", "name": "Dolly"}, {"code": "slider", "name": "Slider"},
                  {"code": "crane", "name": "Crane/Jib"}, {"code": "steadicam", "name": "Steadicam"}, {"code": "gimbal", "name": "Gimbal"},
                  {"code": "drone", "name": "Drone"}, {"code": "handheld_rig", "name": "Handheld rig"}],
}
LENSES = [18, 24, 35, 50, 85, 135]
ROLES = ("storyboard", "keyframe", "reference", "previz")
DETAIL_KEYS = ("size", "type", "angle", "move", "equipment", "lens", "dur", "notes")
CAMERA_KEYS = {"lens_mm": (8, 1200), "sensor_w_mm": (1, 100)}


class PlanRefusal(Exception):
    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code, self.message = reason_code, message


def empty() -> dict[str, Any]:
    return {"schema": SCHEMA, "revision": 0, "shots": {}, "board_order": {}}


def vocabulary() -> dict[str, Any]:
    return {**{k: [dict(v) for v in rows] for k, rows in VOCABULARY.items()}, "lens": list(LENSES), "roles": list(ROLES)}


def valid_document(doc: Any) -> bool:
    return isinstance(doc, dict) and doc.get("schema") == SCHEMA and isinstance(doc.get("shots"), dict)


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _detail(key: str, value: Any) -> Any:
    if key in VOCABULARY:
        codes = [row["code"] for row in VOCABULARY[key]]
        if value not in codes:
            raise PlanRefusal(f"{key}_unknown", f"{key} is one of {', '.join(codes)}.")
        return value
    if key == "lens":
        if not _number(value) or not 8 <= value <= 1200:
            raise PlanRefusal("lens_out_of_range", "A lens is 8 to 1200 mm.")
        return int(value) if float(value).is_integer() else round(float(value), 1)
    if key == "dur":
        if not _number(value) or not 0 < value <= 3600:
            raise PlanRefusal("dur_out_of_range", "A duration is more than 0 and at most 3600 s.")
        return int(value) if float(value).is_integer() else round(float(value), 3)
    if key == "notes":
        if not isinstance(value, str) or len(value) > NOTES_MAX:
            raise PlanRefusal("notes_invalid", f"Notes are text of at most {NOTES_MAX} characters.")
        return value
    raise PlanRefusal("detail_unknown", f"{key} is not a shot detail.")


def _pictures(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > MAX_PICTURES:
        raise PlanRefusal("pictures_invalid", f"A shot holds at most {MAX_PICTURES} pictures.")
    out, seen = [], set()
    for row in value:
        if not isinstance(row, dict) or set(row) - {"item", "role"} or not isinstance(row.get("item"), str) or not ITEM_RE.match(row["item"]):
            raise PlanRefusal("picture_invalid", "A picture is {item, role?} with a Media item id.")
        role = row.get("role")
        if role is not None and role not in ROLES:
            raise PlanRefusal("role_unknown", f"A role is one of {', '.join(ROLES)}.")
        if row["item"] in seen:
            continue
        seen.add(row["item"])
        out.append({"item": row["item"], "role": role})
    return out


ELEMENT_ID_RE = re.compile(r"^el-[0-9a-f]{8}$")


def _elements(value: Any, resolve: Any = None) -> list[dict[str, Any]]:
    """A shot's elements. Lane w20-plan2 contract 1: with ``resolve`` (the route's element catalogue) every entry is
    resolved to the element's immutable id and an existing version (an omitted version = the element's first one, fixed
    here, never left open); unknown ones are refused. Without it (pure checks) the entry is stored as given."""
    if not isinstance(value, list) or len(value) > MAX_ELEMENTS:
        raise PlanRefusal("elements_invalid", f"A shot names at most {MAX_ELEMENTS} elements.")
    out, seen = [], set()
    for row in value:
        if not isinstance(row, dict) or set(row) - {"id", "key", "version"} or not isinstance(row.get("key", row.get("id")), str):
            raise PlanRefusal("element_invalid", "An element is {id | key, version?}.")
        ref = row.get("id") if isinstance(row.get("id"), str) else " ".join(row["key"].split()).casefold()
        if not ref or len(ref) > ELEMENT_KEY_MAX or any(ord(c) < 32 for c in ref):
            raise PlanRefusal("element_invalid", "An element key is a name of at most 60 characters.")
        version = row.get("version")
        if version is not None and (not isinstance(version, str) or not VERSION_RE.match(version)):
            raise PlanRefusal("version_invalid", "A version id is v-<8 hex>.")
        entry = {"key": ref, "version": version}
        if resolve is not None:
            found = resolve(ref)
            if found is None:
                raise PlanRefusal("element_unknown", f"No element {ref} (continuo_elements_read).")
            if version is not None and version not in found["versions"]:
                raise PlanRefusal("version_unknown", f"{found['key']} has no version {version}.")
            entry = {"id": found["id"], "key": found["key"], "version": version or found["versions"][0]}
        elif ELEMENT_ID_RE.match(ref):
            entry = {"id": ref, "key": ref, "version": version}
        if entry.get("id", entry["key"]) in seen:
            continue
        seen.add(entry.get("id", entry["key"]))
        out.append(entry)
    return out


def elements_in_use(doc: dict[str, Any]) -> dict[str, set[str]]:
    """{element id: {version ids}} the shots reference (lane w20-plan2 contract 1: those are not removable)."""
    used: dict[str, set[str]] = {}
    for record in (doc.get("shots") or {}).values():
        for row in record.get("elements") or ():
            if isinstance(row.get("id"), str):
                used.setdefault(row["id"], set()).add(row.get("version"))
    return used


def resolver(elements_doc: dict[str, Any]) -> Any:
    """Look an element up by id, key or alias -> {id, key, versions [ids]} (the route's catalogue for _elements)."""
    rows = elements_doc.get("elements") or []

    def resolve(ref: str) -> dict[str, Any] | None:
        hit = next((e for e in rows if e.get("id") == ref), None) or next((e for e in rows if e.get("key") == ref), None) \
            or next((e for e in rows if ref in (e.get("aliases") or ())), None)
        return None if hit is None or not hit.get("id") else {"id": hit["id"], "key": hit["key"], "versions": [v["id"] for v in hit["versions"]]}
    return resolve


# Lane w20-plan2 contract 2 (owner steer, architecture review): the shot record is the ONLY camera authority. One shape:
# {lens_mm, sensor_w_mm (36), aspect "w:h" (16:9), roll_deg (0), start {pos, at}, end? {pos, at}} in metres, y up,
# right-handed (x right, z toward the default camera). The height is start.pos[1] (derived on read, never stored); the
# lens is ONE field (details.lens mirrors camera.lens_mm). The previz scene document never holds a camera.
CAMERA_SHAPE = ("lens_mm", "sensor_w_mm", "aspect", "roll_deg", "start", "end")
ASPECT_RE = re.compile(r"^([1-9][0-9]{0,3}):([1-9][0-9]{0,3})$")
WORLD_MAX_M = 1000


def _point(raw: Any, where: str) -> list[float]:
    if not isinstance(raw, list) or len(raw) != 3 or not all(_number(v) and abs(v) <= WORLD_MAX_M for v in raw):
        raise PlanRefusal("camera_invalid", f"{where} is [x, y, z] in metres (|v| <= {WORLD_MAX_M}).")
    return [float(v) for v in raw]


def _key(raw: Any, where: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {"pos", "at"}:
        raise PlanRefusal("camera_invalid", f"{where} is {{pos, at}}.")
    pos, at = _point(raw["pos"], where + ".pos"), _point(raw["at"], where + ".at")
    if pos == at:
        raise PlanRefusal("camera_invalid", f"{where}: pos and at are the same point.")
    return {"pos": pos, "at": at}


def _camera(value: Any) -> dict[str, Any] | None:
    """Validate a camera in the one shape; the w20-plan seam keys (height_m, pos, at) are accepted and folded into
    start, and a height that contradicts start.pos[1] is refused (camera_contradiction)."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise PlanRefusal("camera_invalid", "A camera is an object.")
    unknown = set(value) - set(CAMERA_SHAPE) - {"height_m", "pos", "at"}
    if unknown:
        raise PlanRefusal("camera_invalid", f"camera.{sorted(unknown)[0]} is not a camera field.")
    out: dict[str, Any] = {"lens_mm": 35, "sensor_w_mm": 36, "aspect": "16:9", "roll_deg": 0}
    for key in ("lens_mm", "sensor_w_mm"):
        if key in value:
            lo, hi = CAMERA_KEYS[key]
            if not _number(value[key]) or not lo <= value[key] <= hi:
                raise PlanRefusal("camera_invalid", f"camera.{key} is {lo} to {hi}.")
            out[key] = value[key]
    if "aspect" in value:
        if not isinstance(value["aspect"], str) or not ASPECT_RE.match(value["aspect"]):
            raise PlanRefusal("camera_invalid", "camera.aspect is \"w:h\" (16:9).")
        out["aspect"] = value["aspect"]
    if "roll_deg" in value:
        if not _number(value["roll_deg"]) or not -180 <= value["roll_deg"] <= 180:
            raise PlanRefusal("camera_invalid", "camera.roll_deg is -180 to 180.")
        out["roll_deg"] = value["roll_deg"]
    if "start" in value:
        if "pos" in value or "at" in value:
            raise PlanRefusal("camera_contradiction", "Give start {pos, at} or the old pos/at, not both.")
        out["start"] = _key(value["start"], "camera.start")
    else:
        h = value.get("height_m", 1.6)
        if not _number(h) or not 0 <= h <= 500:
            raise PlanRefusal("camera_invalid", "camera.height_m is 0 to 500.")
        pos = _point(value["pos"], "camera.pos") if "pos" in value else [0.0, float(h), 4.0]
        at = _point(value["at"], "camera.at") if "at" in value else [0.0, float(h), 0.0]
        out["start"] = _key({"pos": pos, "at": at}, "camera.start")
    if "height_m" in value and (not _number(value["height_m"]) or abs(out["start"]["pos"][1] - value["height_m"]) > 1e-6):
        raise PlanRefusal("camera_contradiction", "camera.height_m differs from start.pos[1] (the height is the camera's y).")
    if value.get("end") is not None:
        out["end"] = _key(value["end"], "camera.end")
    return out


def migrate(doc: dict[str, Any]) -> bool:
    """Lane w20-plan2: stored w20-plan cameras ({lens_mm, height_m, pos?, at?}) take the one shape; never resets."""
    changed = False
    for record in (doc.get("shots") or {}).values():
        cam = record.get("camera")
        if isinstance(cam, dict) and "start" not in cam:
            cam = {("sensor_w_mm" if k == "sensor_mm" else k): v for k, v in cam.items()}
            try:
                record["camera"] = _camera(cam)
            except PlanRefusal:
                record["camera"] = _camera({k: v for k, v in cam.items() if k in ("lens_mm", "sensor_w_mm")})
            changed = True
    return changed


def camera_public(cam: dict[str, Any] | None) -> dict[str, Any] | None:
    if not cam:
        return None
    out = {k: (dict(v, pos=list(v["pos"]), at=list(v["at"])) if k in ("start", "end") else v) for k, v in cam.items()}
    out["height_m"] = round(cam["start"]["pos"][1], 4)  # derived, read-only
    return out


def apply(doc: dict[str, Any], changes: list[dict[str, Any]], board: dict[str, Any] | None, by: str, at: str, resolve: Any = None) -> list[str]:
    """Apply per-shot patches in place (all validated first: a refused change stores nothing). Returns touched ids."""
    migrate(doc)
    staged: list[tuple[str, dict[str, Any]]] = []
    for change in changes:
        shot = change.get("shot")
        if not isinstance(shot, str) or not SHOT_ID_RE.match(shot):
            raise PlanRefusal("shot_invalid", "A shot id is sh-<8 hex>.")
        patch: dict[str, Any] = {}
        details = change.get("details")
        if details is not None:
            if not isinstance(details, dict):
                raise PlanRefusal("details_invalid", "details is an object.")
            patch["details"] = {k: (None if v is None else _detail(k, v)) for k, v in details.items()}
            for k in patch["details"]:
                if k not in DETAIL_KEYS:
                    raise PlanRefusal("detail_unknown", f"{k} is not a shot detail.")
        if "pictures" in change:
            patch["pictures"] = _pictures(change["pictures"])
        if "elements" in change:
            patch["elements"] = _elements(change["elements"], resolve)
        if "pictures_ops" in change:
            if "pictures" in change:
                raise PlanRefusal("change_invalid", "Give pictures or pictures_ops, not both.")
            patch["pictures_ops"] = _picture_ops(change["pictures_ops"])
        if "elements_ops" in change:
            if "elements" in change:
                raise PlanRefusal("change_invalid", "Give elements or elements_ops, not both.")
            patch["elements_ops"] = _element_ops(change["elements_ops"], resolve)
        if "camera" in change:
            patch["camera"] = _camera(change["camera"])
            lens = (patch.get("details") or {}).get("lens")
            cam_lens = isinstance(change["camera"], dict) and "lens_mm" in change["camera"]
            if patch["camera"] is not None and lens is not None and cam_lens and lens != patch["camera"]["lens_mm"]:
                raise PlanRefusal("lens_contradiction", "details.lens and camera.lens_mm are one field: give one value.")
            patch["camera_lens"] = cam_lens
        staged.append((shot, patch))
    order = None
    if board is not None:
        scene, ids = board.get("scene"), board.get("order")
        if not isinstance(scene, str) or not SCENE_ID_RE.match(scene) or not isinstance(ids, list) or len(ids) > 500 \
                or not all(isinstance(i, str) and SHOT_ID_RE.match(i) for i in ids) or len(set(ids)) != len(ids):
            raise PlanRefusal("board_invalid", "board is {scene: s-<8 hex>, order: [distinct shot ids]}.")
        order = (scene, list(ids))
    shots = doc.setdefault("shots", {})
    if len(set(shots) | {s for s, _ in staged}) > MAX_SHOTS:
        raise PlanRefusal("too_many_shots", f"A project holds at most {MAX_SHOTS} shot records.")
    touched = []
    for shot, patch in staged:
        record = shots.setdefault(shot, {"details": {}, "camera": None, "pictures": [], "elements": [], "joined": []})
        if "details" in patch:
            for k, v in patch["details"].items():
                if v is None:
                    record["details"].pop(k, None)
                else:
                    record["details"][k] = v
        for key in ("pictures", "elements", "camera"):
            if key in patch:
                record[key] = patch[key]
        if "pictures_ops" in patch:
            record["pictures"] = _run_picture_ops(record.get("pictures") or [], patch["pictures_ops"])
        if "elements_ops" in patch:
            record["elements"] = _run_element_ops(record.get("elements") or [], patch["elements_ops"])
        # One lens field (contract 2): the camera's lens and the detail mirror each other, the last written wins both.
        cam = record.get("camera")
        if cam:
            if patch.get("camera_lens"):
                record["details"]["lens"] = cam["lens_mm"]
            elif record["details"].get("lens") is not None:
                cam["lens_mm"] = record["details"]["lens"]
            else:
                record["details"]["lens"] = cam["lens_mm"]
        record["by"], record["at"] = by, at
        touched.append(shot)
    if order is not None:
        doc.setdefault("board_order", {})[order[0]] = order[1]
    return touched


# --- Semantic list changes (lane w20-plan2, architecture review finding 3) --------------------------------------------
# ``pictures`` / ``elements`` replace the whole list: right for Hermes' deliberate rewrite, wrong for a page retry after
# plan_conflict (it drops what another writer added meanwhile). The ops name one item each, so they apply to whatever the
# list is now: add (no-op when present; ``at`` = index, default end), remove (no-op when absent), role / version (no-op
# when absent), move (to an index). A page retries ops after a conflict without losing anyone's work.
PICTURE_OPS = ("add", "remove", "role", "move")
ELEMENT_OPS = ("add", "remove", "version")


def _index(raw: Any, where: str) -> int | None:
    if raw is None:
        return None
    if type(raw) is not int or not 0 <= raw <= MAX_PICTURES:
        raise PlanRefusal("op_invalid", f"{where} is an index 0 to {MAX_PICTURES}.")
    return raw


def _picture_ops(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw or len(raw) > MAX_PICTURES:
        raise PlanRefusal("op_invalid", f"pictures_ops is a list of 1 to {MAX_PICTURES} operations.")
    out = []
    for op in raw:
        if not isinstance(op, dict) or op.get("op") not in PICTURE_OPS or set(op) - {"op", "item", "role", "at"}:
            raise PlanRefusal("op_invalid", "A picture op is {op: add|remove|role|move, item, role?, at?}.")
        if not isinstance(op.get("item"), str) or not ITEM_RE.match(op["item"]):
            raise PlanRefusal("picture_invalid", "A picture op names a Media item id.")
        role = op.get("role")
        if role is not None and role not in ROLES:
            raise PlanRefusal("role_unknown", f"A role is one of {', '.join(ROLES)}.")
        out.append({"op": op["op"], "item": op["item"], "role": role, "at": _index(op.get("at"), "pictures_ops.at")})
    return out


def _run_picture_ops(current: list[dict[str, Any]], ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pics = [dict(p) for p in current]
    for op in ops:
        at = next((i for i, p in enumerate(pics) if p["item"] == op["item"]), None)
        if op["op"] == "add" and at is None:
            if len(pics) >= MAX_PICTURES:
                raise PlanRefusal("pictures_invalid", f"A shot holds at most {MAX_PICTURES} pictures.")
            pics.insert(len(pics) if op["at"] is None else min(op["at"], len(pics)), {"item": op["item"], "role": op["role"]})
        elif op["op"] == "remove" and at is not None:
            pics.pop(at)
        elif op["op"] == "role" and at is not None:
            pics[at]["role"] = op["role"]
        elif op["op"] == "move" and at is not None:
            row = pics.pop(at)
            pics.insert(len(pics) if op["at"] is None else min(op["at"], len(pics)), row)
    return pics


def _element_ops(raw: Any, resolve: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw or len(raw) > MAX_ELEMENTS:
        raise PlanRefusal("op_invalid", f"elements_ops is a list of 1 to {MAX_ELEMENTS} operations.")
    out = []
    for op in raw:
        if not isinstance(op, dict) or op.get("op") not in ELEMENT_OPS or set(op) - {"op", "id", "key", "version"}:
            raise PlanRefusal("op_invalid", "An element op is {op: add|remove|version, id | key, version?}.")
        if op["op"] == "version" and op.get("version") is None:
            raise PlanRefusal("op_invalid", "A version op names the version.")
        entry = _elements([{k: op[k] for k in ("id", "key", "version") if k in op}], resolve if op["op"] != "remove" else None)[0]
        if op["op"] == "remove" and resolve is not None:
            found = resolve(entry["key"])
            entry = {"id": found["id"], "key": found["key"], "version": None} if found else entry
        out.append({"op": op["op"], **entry})
    return out


def _run_element_ops(current: list[dict[str, Any]], ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(e) for e in current]
    for op in ops:
        handle = op.get("id", op["key"])
        at = next((i for i, e in enumerate(rows) if e.get("id", e["key"]) == handle or e["key"] == op["key"]), None)
        if op["op"] == "add" and at is None:
            if len(rows) >= MAX_ELEMENTS:
                raise PlanRefusal("elements_invalid", f"A shot names at most {MAX_ELEMENTS} elements.")
            rows.append({k: v for k, v in op.items() if k != "op"})
        elif op["op"] == "remove" and at is not None:
            rows.pop(at)
        elif op["op"] == "version" and at is not None:
            rows[at]["version"] = op["version"]
    return rows


# Codes earlier lanes stored that the vocabulary no longer offers (A3 removed Establishing): kept as stored, never refused
# on read (the page shows them as they are); a write of them is refused.
LEGACY_CODES = {"type": ("establishing",)}


# --- Full document check (architecture review finding 2): what the store loads is checked to the leaf ----------------
def check_document(doc: Any) -> str | None:
    """None when ``doc`` is a well-formed studio.shot-plan/1 document (w20-plan cameras included), else the problem."""
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        return "not a studio.shot-plan/1 document"
    if set(doc) - {"schema", "revision", "shots", "board_order"}:
        return f"unknown field {sorted(set(doc) - {'schema', 'revision', 'shots', 'board_order'})[0]}"
    if type(doc.get("revision")) is not int or doc["revision"] < 0:
        return "revision is not an integer >= 0"
    shots, order = doc.get("shots"), doc.get("board_order", {})
    if not isinstance(shots, dict) or len(shots) > MAX_SHOTS or not isinstance(order, dict):
        return "shots / board_order are not objects"
    try:
        for shot, rec in shots.items():
            if not isinstance(shot, str) or not SHOT_ID_RE.match(shot) or not isinstance(rec, dict):
                return f"shot {str(shot)[:20]} is not sh-<8 hex> with a record"
            if set(rec) - {"details", "camera", "pictures", "elements", "joined", "by", "at"}:
                return f"shot {shot} has an unknown field"
            details = rec.get("details", {})
            if not isinstance(details, dict) or set(details) - set(DETAIL_KEYS):
                return f"shot {shot} details are invalid"
            for k, v in details.items():
                if v not in LEGACY_CODES.get(k, ()):
                    _detail(k, v)
            if _pictures(rec.get("pictures", [])) != [{"item": p["item"], "role": p.get("role")} for p in rec.get("pictures", [])]:
                return f"shot {shot} pictures repeat an item"
            for row in rec.get("elements", []) or []:
                if not isinstance(row, dict) or set(row) - {"id", "key", "version"}:
                    return f"shot {shot} elements are invalid"
            _elements(rec.get("elements", []))
            cam = rec.get("camera")
            if isinstance(cam, dict) and "start" not in cam:  # a w20-plan camera: migrate() folds it on read
                cam = None if not all(k in ("lens_mm", "sensor_mm", "sensor_w_mm", "height_m", "pos", "at") for k in cam) else {}
                if cam is None:
                    return f"shot {shot} camera has an unknown field"
            else:
                _camera(cam)
            joined = rec.get("joined", [])
            if not isinstance(joined, list) or not all(isinstance(i, str) and ITEM_RE.match(i) for i in joined):
                return f"shot {shot} joined is invalid"
            if rec.get("by") not in (None, "artist", "hermes") or not (rec.get("at") is None or isinstance(rec["at"], str) and len(rec["at"]) <= 40):
                return f"shot {shot} by/at are invalid"
        for scene, ids in order.items():
            if not isinstance(scene, str) or not SCENE_ID_RE.match(scene) or not isinstance(ids, list) or len(ids) > 500 \
                    or not all(isinstance(i, str) and SHOT_ID_RE.match(i) for i in ids):
                return f"board_order of {str(scene)[:20]} is invalid"
    except PlanRefusal as exc:
        return exc.message
    return None


def join_marks(doc: dict[str, Any], marked: dict[str, list[str]]) -> int:
    """Take Media-marked pictures (shot:<id>, the old ShotMargin data) into the record once, role storyboard."""
    count = 0
    shots = doc.setdefault("shots", {})
    for shot, items in marked.items():
        if not SHOT_ID_RE.match(shot):
            continue
        record = shots.get(shot)
        for item in items:
            if not isinstance(item, str) or not ITEM_RE.match(item):
                continue
            if record is not None and item in record.get("joined", ()):
                continue
            if record is None:
                record = shots.setdefault(shot, {"details": {}, "camera": None, "pictures": [], "elements": [], "joined": []})
            record.setdefault("joined", []).append(item)
            if not any(p["item"] == item for p in record["pictures"]) and len(record["pictures"]) < MAX_PICTURES:
                record["pictures"].append({"item": item, "role": "storyboard"})
            count += 1
    return count


def public(doc: dict[str, Any]) -> dict[str, Any]:
    shots = {}
    for shot, record in (doc.get("shots") or {}).items():
        shots[shot] = {"details": dict(record.get("details") or {}), "camera": camera_public(record.get("camera")),
                       "pictures": [dict(p) for p in record.get("pictures") or []],
                       "elements": [dict(e) for e in record.get("elements") or []],
                       "by": record.get("by"), "at": record.get("at")}
    return {"schema": SCHEMA, "revision": int(doc.get("revision") or 0), "shots": shots,
            "board_order": {k: list(v) for k, v in (doc.get("board_order") or {}).items()}, "vocabulary": vocabulary()}
