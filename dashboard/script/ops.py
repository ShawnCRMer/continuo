"""One function per hand action on ``studio.script/1``, with the per-part conflict rule.

Every verb takes ``expected_revision``: the document revision the caller last read. The
verb conflicts only when the part it changes was changed after that revision:

* scene verbs (write, omit, delete): the scene's ``revision`` (the document revision of
  its last change) must be <= expected_revision; else ``scene_conflict`` with the
  current and the proposed version. Writing scene 12 never conflicts with scene 3.
* order verbs (add, move, aside, act breaks, numbers, title page): ``order_revision``
  must be <= expected_revision; else ``order_conflict`` with the current order.
* card place: the card's own ``revision``; text edits never conflict with a card move.
* full-script write and import: per scene and per order as above; import over an
  existing script needs expected_revision == the document revision.

``apply(state, verb, args, expected_revision, revision, by)`` returns the new state
(a copy), the result for the caller and whether anything changed (no change, no new
revision). ``revision`` is the revision the write will get.
"""
from __future__ import annotations

import copy
from typing import Any, Callable

from . import fountain
from . import shots as shot_ids
from .schema import (
    ACT_RE,
    BREAK_ID_RE,
    MAX_BREAKS,
    MAX_SCENES,
    NUMBER_RE,
    SCENE_ID_RE,
    ScriptError,
    aside_ids,
    assign_locked_numbers,
    heading_of,
    check_synopsis,
    check_text,
    check_title,
    default_state,
    fail,
    new_id,
    numbers,
    scene_ids_in_order,
    validate_state,
)


_SCENE_ID = SCENE_ID_RE


class ScriptConflict(ScriptError):
    """A stale write to a part someone changed since: never applied, both versions shown."""

    def __init__(self, code: str, message: str, details: dict[str, Any]) -> None:
        super().__init__(code, message)
        self.details = details


def normalize_text(text: Any, where: str) -> str:
    if type(text) is not str:
        fail("invalid_op", f"{where}: must be text")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text, _notes = shot_ids.split_notes(text)  # [[shot sh-...]] notes carry ids, never text
    lines = [fountain.TWO_SPACES if fountain.is_kept_blank(line) else line.rstrip() for line in text.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return check_text("\n".join(lines), where)


def public_scene(sid: str, scene: dict[str, Any]) -> dict[str, Any]:
    return {"id": sid, **copy.deepcopy(scene)}


def _scene(state: dict[str, Any], sid: Any) -> dict[str, Any]:
    if type(sid) is not str or sid not in state["scenes"]:
        fail("not_found", f"scene {sid!r} is not in this script")
    return state["scenes"][sid]


def _check_scene_fresh(state: dict[str, Any], sid: str, expected: int, proposed: dict[str, Any]) -> None:
    scene = state["scenes"][sid]
    if scene["revision"] > expected:
        raise ScriptConflict(
            "scene_conflict",
            f"scene {sid} changed at revision {scene['revision']} (by {scene['by']}) after revision {expected}",
            {"scene": sid, "current": public_scene(sid, scene), "proposed": proposed},
        )


def _check_order_fresh(state: dict[str, Any], expected: int) -> None:
    if state["order_revision"] > expected:
        raise ScriptConflict(
            "order_conflict",
            f"the order changed at revision {state['order_revision']} after revision {expected}",
            {"current": {"order": copy.deepcopy(state["order"]), "order_revision": state["order_revision"]}},
        )


def _key(item: Any) -> str:
    return item if type(item) is str else item["id"]


def _insert_at(order: list[Any], items: list[Any], at: Any) -> list[Any]:
    """``at``: None (end), {index}, {before: id} or {after: id} (ids of the order)."""
    if at is None:
        return order + items
    if type(at) is not dict or len(at) != 1 or next(iter(at)) not in ("index", "before", "after"):
        fail("invalid_op", "at: must be {index}, {before: id} or {after: id}")
    (how, value), = at.items()
    if how == "index":
        if type(value) is not int or not 0 <= value <= len(order):
            fail("invalid_op", f"at.index: must be 0..{len(order)}")
        index = value
    else:
        keys = [_key(item) for item in order]
        if value not in keys:
            fail("not_found", f"at.{how}: {value!r} is not in the order (or is one of the items moved)")
        index = keys.index(value) + (1 if how == "after" else 0)
    return order[:index] + items + order[index:]


def _new_scene(text: str, synopsis: str, revision: int, by: str) -> dict[str, Any]:
    return {"text": text, "synopsis": synopsis, "card": None, "number": None, "numbered": False,
            "omitted": False, "revision": revision, "by": by}


def _taken_shots(state: dict[str, Any], *, but: str | None = None) -> set[str]:
    """Shot ids used by the scenes of ``state`` other than ``but``."""
    out: set[str] = set()
    for sid, scene in state["scenes"].items():
        if sid != but:
            out.update(shot_ids.ids_of(sid, scene))
    return out


def _given_shots(value: Any, raw_text: Any) -> list[Any] | None:
    """The caller's shot ids: ``shots`` (the editor), else ``[[shot ...]]`` notes in the text."""
    if value is not None:
        if type(value) is not list or len(value) > shot_ids.MAX_SHOTS or any(v is not None and type(v) is not str for v in value):
            fail("invalid_op", "shots: a list of shot ids (sh-1a2b3c4d) or null, one per Shot line")
        return value
    if type(raw_text) is str:
        clean, notes = shot_ids.split_notes(raw_text.replace("\r\n", "\n").replace("\r", "\n"))
        return shot_ids.ids_from_notes(clean, notes)
    return None


def _set_shots(state: dict[str, Any], sid: str, old_text: str, old_ids: list[str], given: list[Any] | None) -> None:
    scene = state["scenes"][sid]
    scene["shots"] = shot_ids.reconcile(old_text, old_ids, scene["text"], given=given, taken=_taken_shots(state, but=sid))


def _mark_numbered(state: dict[str, Any]) -> None:
    for scene in state["scenes"].values():
        if scene["number"]:
            scene["numbered"] = True


def _order_changed(state: dict[str, Any], revision: int) -> None:
    state["order_revision"] = revision
    assign_locked_numbers(state)
    _mark_numbered(state)


# --- verbs -------------------------------------------------------------------------


def scene_add(state, args, expected, revision, by):
    _check_order_fresh(state, expected) if not args.get("aside") else None
    if len(state["scenes"]) >= MAX_SCENES:
        fail("script_too_large", f"a script holds at most {MAX_SCENES} scenes")
    text = normalize_text(args.get("text", ""), "text")
    synopsis = check_synopsis(args.get("synopsis", ""), "synopsis")
    sid = new_id("s", set(state["scenes"]))
    state["scenes"][sid] = _new_scene(text, synopsis, revision, by)
    _set_shots(state, sid, "", [], _given_shots(args.get("shots"), args.get("text")))
    if not args.get("aside"):
        state["order"] = _insert_at(state["order"], [sid], args.get("at"))
        _order_changed(state, revision)
    return {"scene": sid}


def scene_write(state, args, expected, revision, by):
    sid = args.get("scene")
    scene = _scene(state, sid)
    proposed: dict[str, Any] = {}
    if "text" in args:
        proposed["text"] = normalize_text(args["text"], "text")
    if "synopsis" in args:
        proposed["synopsis"] = check_synopsis(args["synopsis"], "synopsis")
    if not proposed:
        fail("invalid_op", "neither text nor synopsis given")
    old_text, old_ids = scene["text"], shot_ids.ids_of(sid, scene)
    given = _given_shots(args.get("shots"), args.get("text")) if "text" in proposed else None
    if all(scene[key] == value for key, value in proposed.items()):
        return {"scene": sid, "unchanged": True}  # same text: the shots keep their ids
    _check_scene_fresh(state, sid, expected, proposed)
    scene.update(proposed)
    if "text" in proposed:
        _set_shots(state, sid, old_text, old_ids, given)
    scene["revision"], scene["by"] = revision, by
    return {"scene": sid}


def scene_omit(state, args, expected, revision, by):
    sid = args.get("scene")
    scene = _scene(state, sid)
    omitted = args.get("omitted", True)
    if type(omitted) is not bool:
        fail("invalid_op", "omitted: must be true or false")
    if scene["omitted"] == omitted:
        return {"scene": sid, "unchanged": True}
    _check_scene_fresh(state, sid, expected, {"omitted": omitted})
    scene["omitted"], scene["revision"], scene["by"] = omitted, revision, by
    return {"scene": sid, "number": numbers(state).get(sid)}


def scene_delete(state, args, expected, revision, by):
    sid = args.get("scene")
    scene = _scene(state, sid)
    force = args.get("force", False)
    if type(force) is not bool:
        fail("invalid_op", "force: must be true or false")
    if scene["numbered"] and not force:
        fail("scene_numbered", f"scene {sid} has had a locked number; omitting keeps it, deleting needs force: true")
    _check_scene_fresh(state, sid, expected, {"deleted": True})
    if sid in state["order"]:
        _check_order_fresh(state, expected)
        state["order"] = [item for item in state["order"] if item != sid]
        state["order_revision"] = revision
    del state["scenes"][sid]
    return {"scene": sid, "deleted": True}


def _take_items(state: dict[str, Any], ids: Any) -> tuple[list[Any], list[Any]]:
    if type(ids) is not list or not ids or len(ids) > MAX_SCENES + MAX_BREAKS:
        fail("invalid_op", "items: a non-empty list of scene and act break ids")
    wanted = set()
    for value in ids:
        if type(value) is not str or value in wanted:
            fail("invalid_op", f"items: {value!r} is not an id or is given twice")
        wanted.add(value)
    keys = [_key(item) for item in state["order"]]
    missing = [value for value in ids if value not in keys]
    if missing:
        aside = [value for value in missing if value in state["scenes"]]
        if aside:
            fail("not_in_order", f"{', '.join(aside)} is aside (script_aside with aside: false brings it back)")
        fail("not_found", f"{', '.join(missing)} is not in the order")
    taken = [item for item in state["order"] if _key(item) in wanted]  # their order in the script
    rest = [item for item in state["order"] if _key(item) not in wanted]
    return taken, rest


def _act_tail(text: str) -> int:
    """Where the act line(s) at the end of a scene's text begin (charter decision 14: a centered act word alone after
    the scene, e.g. ``> AKT 2 <``), or len(text) when the scene ends without one. Only whole blocks after a blank line
    count, and never the whole text (a scene keeps its own words)."""
    lines = text.split("\n")
    end = len(lines)
    cut = None
    k = end - 1
    while k >= 0:
        line = lines[k].strip()
        if not line:
            k -= 1
            continue
        if line.startswith(">") and line.endswith("<") and len(line) > 2 and ACT_RE.match(" ".join(line[1:-1].split())):
            if k == 0 or lines[k - 1].strip():
                break  # part of a longer block, or the scene's only line
            cut = k
            k -= 1
            continue
        break
    if cut is None:
        return len(text)
    body = "\n".join(lines[:cut]).rstrip("\n")
    return len(body)


def _keep_act_lines(state: dict[str, Any], old_order: list[Any], taken: list[Any], at: Any, revision: int, by: str) -> None:
    """Act breaks stay where they were (Final Draft Navigator / Scrite structure: a scene dropped next to a scene
    joins that scene's act; the writer moves act breaks explicitly). An act line is text at the end of the scene
    before the act, so a move must carry it the other way: a moved scene leaves its act line to the scene that stays
    before its gap, and scenes dropped AFTER the last scene of an act take that act's line behind them."""
    moved = {item for item in taken if type(item) is str}
    if not moved:
        return
    changed: dict[str, str] = {}
    text = lambda sid: changed.get(sid, state["scenes"][sid]["text"])
    # 1. A moved scene's act line stays at its gap: to the nearest scene before it that stays.
    for k, item in enumerate(old_order):
        if type(item) is not str or item not in moved:
            continue  # stored act breaks ({break: act}) are structure items, never text
        cut = _act_tail(text(item))
        if cut == len(text(item)):
            continue
        keeper = next((old_order[j] for j in range(k - 1, -1, -1) if type(old_order[j]) is str and old_order[j] not in moved), None)
        if keeper is None:
            continue
        tail = text(item)[cut:]
        changed[item] = text(item)[:cut]
        changed[keeper] = text(keeper).rstrip("\n") + tail
    # 2. Dropped after the last scene of an act: the act's line goes behind the dropped scenes (they stay in the act).
    if type(at) is dict and set(at) == {"after"} and type(at["after"]) is str and at["after"] in state["scenes"] and at["after"] not in moved:
        anchor, last = at["after"], [item for item in taken if type(item) is str][-1]
        cut = _act_tail(text(anchor))
        if cut < len(text(anchor)):
            tail = text(anchor)[cut:]
            changed[anchor] = text(anchor)[:cut]
            changed[last] = text(last).rstrip("\n") + tail
    for sid, new in changed.items():
        scene = state["scenes"][sid]
        if new == scene["text"]:
            continue
        old_text, old_ids = scene["text"], shot_ids.ids_of(sid, scene)
        scene["text"], scene["revision"], scene["by"] = new, revision, by
        _set_shots(state, sid, old_text, old_ids, None)


def move(state, args, expected, revision, by):
    _check_order_fresh(state, expected)
    taken, rest = _take_items(state, args.get("items"))
    order = _insert_at(rest, taken, args.get("at"))
    if order == state["order"]:
        return {"items": [_key(item) for item in taken], "unchanged": True}
    old_order = state["order"]
    state["order"] = order
    _keep_act_lines(state, old_order, taken, args.get("at"), revision, by)
    _order_changed(state, revision)
    return {"items": [_key(item) for item in taken]}


def aside(state, args, expected, revision, by):
    _check_order_fresh(state, expected)
    ids = args.get("scenes")
    put_aside = args.get("aside", True)
    if type(put_aside) is not bool:
        fail("invalid_op", "aside: must be true or false")
    if type(ids) is not list or not ids or len(ids) > MAX_SCENES:
        fail("invalid_op", "scenes: a non-empty list of scene ids")
    for sid in ids:
        _scene(state, sid)
    if len(set(ids)) != len(ids):
        fail("invalid_op", "scenes: an id is given twice")
    if put_aside:
        if args.get("at") is not None:
            fail("invalid_op", "at: only when putting scenes back (aside: false)")
        placed = set(scene_ids_in_order(state))
        if not any(sid in placed for sid in ids):
            return {"scenes": ids, "unchanged": True}
        state["order"] = [item for item in state["order"] if not (type(item) is str and item in ids)]
    else:
        placed = set(scene_ids_in_order(state))
        back = [sid for sid in ids if sid not in placed]
        if len(back) != len(ids):
            fail("invalid_op", f"{', '.join(sid for sid in ids if sid in placed)} is already in the order (script_move moves it)")
        state["order"] = _insert_at(state["order"], back, args.get("at"))
    _order_changed(state, revision)
    return {"scenes": ids, "aside": aside_ids(state)}


def _break(state: dict[str, Any], bid: Any) -> dict[str, Any]:
    for item in state["order"]:
        if type(item) is dict and item["id"] == bid:
            return item
    fail("not_found", f"act break {bid!r} is not in the order")
    raise AssertionError


def break_add(state, args, expected, revision, by):
    _check_order_fresh(state, expected)
    title = check_title(args.get("title"), "title")
    if sum(1 for item in state["order"] if type(item) is dict) >= MAX_BREAKS:
        fail("script_too_large", f"a script holds at most {MAX_BREAKS} act breaks")
    bid = new_id("a", {item["id"] for item in state["order"] if type(item) is dict})
    state["order"] = _insert_at(state["order"], [{"break": "act", "id": bid, "title": title}], args.get("at"))
    _order_changed(state, revision)
    return {"break": bid}


def break_rename(state, args, expected, revision, by):
    item = _break(state, args.get("break"))
    title = check_title(args.get("title"), "title")
    if item["title"] == title:
        return {"break": item["id"], "unchanged": True}
    _check_order_fresh(state, expected)
    item["title"] = title
    state["order_revision"] = revision
    return {"break": item["id"]}


def break_remove(state, args, expected, revision, by):
    item = _break(state, args.get("break"))
    _check_order_fresh(state, expected)
    state["order"] = [entry for entry in state["order"] if entry is not item]
    state["order_revision"] = revision
    return {"break": item["id"], "removed": True}


def card_place(state, args, expected, revision, by):
    sid = args.get("scene")
    scene = _scene(state, sid)
    if args.get("card", "missing") is None:
        card = None
    else:
        x, y = args.get("x"), args.get("y")
        for name, value in (("x", x), ("y", y)):
            if type(value) not in (int, float) or value != value or abs(value) > 1_000_000:
                fail("invalid_op", f"{name}: a finite number within +-1000000")
        card = {"x": x, "y": y}
    current = None if scene["card"] is None else {"x": scene["card"]["x"], "y": scene["card"]["y"]}
    if current == card:
        return {"scene": sid, "unchanged": True}
    if scene["card"] is not None and scene["card"]["revision"] > expected:
        raise ScriptConflict("card_conflict", f"the card of {sid} moved at revision {scene['card']['revision']} after revision {expected}",
                             {"scene": sid, "current": copy.deepcopy(scene["card"]), "proposed": card})
    scene["card"] = None if card is None else {**card, "revision": revision}
    return {"scene": sid, "card": scene["card"]}


def numbers_lock(state, args, expected, revision, by):
    locked = args.get("locked")
    if type(locked) is not bool:
        fail("invalid_op", "locked: must be true or false")
    if state["numbers_locked"] == locked:
        return {"locked": locked, "unchanged": True}
    _check_order_fresh(state, expected)
    if locked:
        shown = numbers(state)
        for sid, number in shown.items():
            state["scenes"][sid]["number"] = number
        state["numbers_locked"] = True
        _mark_numbered(state)
    else:
        for scene in state["scenes"].values():
            scene["number"] = None
        state["numbers_locked"] = False
    state["order_revision"] = revision
    return {"locked": locked, "numbers": numbers(state)}


def title_page(state, args, expected, revision, by):
    entries = args.get("entries")
    candidate = {**default_state(), "title_page": entries}
    validate_state(candidate)
    if entries == state["title_page"]:
        return {"unchanged": True}
    _check_order_fresh(state, expected)
    state["title_page"] = copy.deepcopy(entries)
    state["order_revision"] = revision
    return {"title_page": state["title_page"]}


def page_size(state, args, expected, revision, by):
    """The page size of the document (charter decision 4): letter (default) or a4. Changes the
    page breaks of the editor, the PDF and derived.pages together."""
    size = args.get("size")
    if size not in ("letter", "a4"):
        fail("invalid_op", "size: letter or a4")
    if state.get("page_size", "letter") == size:
        return {"page_size": size, "unchanged": True}
    _check_order_fresh(state, expected)
    if size == "letter":
        state.pop("page_size", None)
    else:
        state["page_size"] = size
    state["order_revision"] = revision
    return {"page_size": size}


# --- whole script --------------------------------------------------------------------


def _unshift_preamble_numbers(parsed: dict[str, Any], report: dict[str, Any]) -> None:
    """Scripts exported before charter decision 11 numbered the preamble (the text before the
    first heading) as scene 1, so their first real heading reads #2#. When the blocks before the
    first heading carry no number and the headings carry exactly P+1, P+2, ... (P = those
    blocks), the numbers were only that offset: they become 1, 2, ... (report numbers_shifted)."""
    scenes = [item for item in parsed["items"] if item["kind"] == "scene"]
    lead = 0
    for item in scenes:
        if item["omitted"] or heading_of(item["text"]) is not None:
            break
        if item.get("number"):
            return
        lead += 1
    rest = scenes[lead:]
    if not lead or not rest or [item.get("number") for item in rest] != [str(lead + 1 + k) for k in range(len(rest))]:
        return
    for k, item in enumerate(rest):
        item["number"] = str(k + 1)
    report["numbers_shifted"] = lead


def _build_from_items(parsed: dict[str, Any], revision: int, by: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """A fresh document from parsed items (import): every scene new, numbers locked when any
    scene carried one."""
    state = default_state()
    state["title_page"] = parsed["title_page"]
    report = {"scenes": 0, "breaks": 0, "aside": 0, "omitted": 0, "numbers_dropped": [], "dropped": parsed["dropped"]}
    any_number = any(item.get("number") for item in parsed["items"] if item["kind"] == "scene")
    _unshift_preamble_numbers(parsed, report)
    used: set[str] = set()
    for item in parsed["items"]:
        if item["kind"] == "break":
            state["order"].append({"break": "act", "id": new_id("a", {_key(i) for i in state["order"]}), "title": item["title"],
                                   **({"level": item["level"]} if item.get("level") else {})})
            report["breaks"] += 1
            continue
        sid = new_id("s", set(state["scenes"]))
        scene = _new_scene(normalize_text(item["text"], f"scene {len(state['scenes']) + 1}"), check_synopsis(item["synopsis"], "synopsis"), revision, by)
        scene["omitted"] = item["omitted"]
        state["scenes"][sid] = scene
        _set_shots(state, sid, "", [], _given_shots(item.get("shots"), item["text"]))
        number = item.get("number")
        if any_number and number:
            if number in used:
                report["numbers_dropped"].append(number)
            else:
                scene["number"], scene["numbered"] = number, True
                used.add(number)
        state["scenes"][sid] = scene
        state["order"].append(sid)
        report["scenes"] += 1
        report["omitted"] += item["omitted"]
    for item in parsed["aside"]:
        sid = new_id("s", set(state["scenes"]))
        scene = _new_scene(normalize_text(item["text"], "aside scene"), check_synopsis(item["synopsis"], "synopsis"), revision, by)
        scene["omitted"] = item["omitted"]
        state["scenes"][sid] = scene
        _set_shots(state, sid, "", [], _given_shots(item.get("shots"), item["text"]))
        report["aside"] += 1
    if len(state["scenes"]) > MAX_SCENES:
        fail("script_too_large", f"a script holds at most {MAX_SCENES} scenes")
    state["numbers_locked"] = any_number
    state["order_revision"] = revision
    assign_locked_numbers(state)
    _mark_numbered(state)
    return validate_state(state), report


def import_state(parsed: dict[str, Any], revision: int, by: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return _build_from_items(parsed, revision, by)


def write_fountain(state, args, expected, revision, by):
    """The full script back from Fountain with id notes: scenes map by their [[s-...]] note;
    a scene without a known note is new; a scene missing from the text is set aside.

    ``base`` (optional) is the document state at ``expected`` (a kept revision). A part that
    changed after ``expected`` and comes back exactly as it was at ``expected`` was not
    touched by this writer: the current version stays and nothing conflicts. Without the
    base (revision no longer kept) every stale difference is a conflict."""
    base = args.get("base")
    text = args.get("fountain")
    if type(text) is not str or len(text.encode("utf-8")) > 8 * 1024 * 1024:
        fail("invalid_op", "fountain: the whole script as text (at most 8 MiB)")
    parsed = fountain.parse(text)
    report: dict[str, Any] = {"created": [], "changed": [], "unknown_ids": [], "set_aside": [], "dropped": parsed["dropped"]}
    conflicts: list[dict[str, Any]] = []
    seen: set[str] = set()
    order: list[Any] = []
    breaks = {item["id"]: item for item in state["order"] if type(item) is dict}
    new_scenes = copy.deepcopy(state["scenes"])

    def take_scene(item: dict[str, Any], placed: bool) -> str:
        sid = item.get("id")
        if sid is not None and sid in seen:
            fail("invalid_op", f"[[{sid}]] appears twice; a scene has one place")
        if sid is None or sid not in state["scenes"]:
            if sid is not None:
                report["unknown_ids"].append(sid)
            sid = new_id("s", set(new_scenes) | seen)
            scene = _new_scene(normalize_text(item["text"], f"new scene {len(report['created']) + 1}"), check_synopsis(item["synopsis"], "synopsis"), revision, by)
            scene["omitted"] = item["omitted"]
            new_scenes[sid] = scene
            scene["shots"] = shot_ids.reconcile("", [], scene["text"], given=_given_shots(None, item["text"]),
                                                taken=_taken_shots({"scenes": new_scenes}, but=sid))
            report["created"].append(sid)
        else:
            current = state["scenes"][sid]
            proposed = {"omitted": item["omitted"]}
            if not item["omitted"]:
                proposed["text"] = normalize_text(item["text"], f"[[{sid}]]")
                proposed["synopsis"] = check_synopsis(item["synopsis"], f"[[{sid}]] synopsis")
            if any(current[key] != value for key, value in proposed.items()):
                was = (base or {}).get("scenes", {}).get(sid)
                if current["revision"] > expected and was is not None and all(was[key] == value for key, value in proposed.items()):
                    pass  # untouched by this writer: the newer version stays
                elif current["revision"] > expected:
                    conflicts.append({"scene": sid, "current": public_scene(sid, current), "proposed": proposed})
                else:
                    old_text, old_ids = current["text"], shot_ids.ids_of(sid, current)
                    new_scenes[sid].update(proposed)
                    if "text" in proposed:
                        new_scenes[sid]["shots"] = shot_ids.reconcile(old_text, old_ids, proposed["text"], given=_given_shots(None, item["text"]),
                                                                      taken=_taken_shots({"scenes": new_scenes}, but=sid))
                    new_scenes[sid]["revision"], new_scenes[sid]["by"] = revision, by
                    report["changed"].append(sid)
        seen.add(sid)
        return sid

    for item in parsed["items"]:
        if item["kind"] == "break":
            bid = item.get("id")
            if bid in breaks and bid not in seen:
                entry = {**{k: v for k, v in breaks[bid].items() if k != "level"}, "title": check_title(item["title"], f"[[{bid}]]")}
            else:
                bid = new_id("a", set(breaks) | seen)
                entry = {"break": "act", "id": bid, "title": check_title(item["title"], "act break")}
            if item.get("level"):
                entry["level"] = item["level"]
            seen.add(bid)
            order.append(entry)
            continue
        order.append(take_scene(item, True))
    for item in parsed["aside"]:
        take_scene(item, False)
    for sid in state["scenes"]:
        if sid not in seen and sid in state["order"]:
            report["set_aside"].append(sid)
    if len(new_scenes) > MAX_SCENES:
        fail("script_too_large", f"a script holds at most {MAX_SCENES} scenes")
    structure_changed = order != state["order"] or parsed["title_page"] != state["title_page"]
    if structure_changed and base is not None and state["order_revision"] > expected:
        base_ids = set(base["scenes"])
        same_order = [item for item in order if not (type(item) is str and item not in base_ids)] == base["order"]
        if same_order and parsed["title_page"] == base["title_page"] and not report["created"]:
            structure_changed = False  # the order and title page came back as they were: keep the newer ones
    if structure_changed and state["order_revision"] > expected:
        conflicts.append({"order": True, "current": {"order": copy.deepcopy(state["order"]), "title_page": copy.deepcopy(state["title_page"]),
                                                     "order_revision": state["order_revision"]}})
    if conflicts:
        raise ScriptConflict("scene_conflict" if all("scene" in c for c in conflicts) else "order_conflict",
                             f"{len(conflicts)} part(s) changed after revision {expected}; nothing was saved", {"conflicts": conflicts})
    state["scenes"] = new_scenes
    if structure_changed:
        state["order"] = order
        state["title_page"] = parsed["title_page"]
        _order_changed(state, revision)
    report["structure_changed"] = structure_changed
    if not (report["created"] or report["changed"] or structure_changed):
        report["unchanged"] = True
    return report


def restore(state, args, expected, revision, by):
    """Bring a kept revision back as a new revision (the History's Undo / Restore, board E21):
    ``target`` is the state of that revision, ``target_revision`` its number. Replaces the whole
    document, so ``expected`` must be the current revision (``current``); a scene, the order or
    a card that comes back different gets this revision and author (later stale writes to it
    conflict as usual), an untouched one keeps its own."""
    target, current = args.get("target"), args.get("current")
    if type(target) is not dict:
        fail("invalid_op", "target: the state of a kept revision")
    if expected != current:
        raise ScriptError("script_conflict", f"restoring replaces the script and needs its current revision ({current})")
    back = copy.deepcopy(validate_state(copy.deepcopy(target)))
    content = ("text", "synopsis", "omitted", "number", "numbered", "shots")
    for sid, scene in back["scenes"].items():
        now = state["scenes"].get(sid)
        if now is not None and all(now.get(k) == scene.get(k) for k in content):
            scene["revision"], scene["by"] = now["revision"], now["by"]
        else:
            scene["revision"], scene["by"] = revision, by
        if now is not None and now["card"] == scene["card"]:
            continue
        if scene["card"] is not None:
            scene["card"]["revision"] = revision
    order_same = back["order"] == state["order"] and back["numbers_locked"] == state["numbers_locked"]
    back["order_revision"] = state["order_revision"] if order_same else revision
    removed = [sid for sid in state["scenes"] if sid not in back["scenes"]]
    state.clear()
    state.update(back)
    return {"restored": args.get("target_revision"), "scenes": [sid for sid in back["scenes"]] + removed}


VERBS: dict[str, Callable[..., dict[str, Any]]] = {
    "scene_add": scene_add,
    "scene_write": scene_write,
    "scene_omit": scene_omit,
    "scene_delete": scene_delete,
    "move": move,
    "aside": aside,
    "break_add": break_add,
    "break_rename": break_rename,
    "break_remove": break_remove,
    "card_place": card_place,
    "numbers_lock": numbers_lock,
    "title_page": title_page,
    "page_size": page_size,
    "write_fountain": write_fountain,
    "restore": restore,
}


def apply(state: dict[str, Any] | None, verb: str, args: dict[str, Any], *, expected_revision: int, revision: int,
          by: str) -> tuple[dict[str, Any], dict[str, Any], bool]:
    if verb not in VERBS:
        fail("unknown_op", f"unknown verb {verb!r}")
    if type(expected_revision) is not int or expected_revision < 0 or expected_revision >= revision:
        fail("invalid_op", f"expected_revision: 0..{revision - 1}")
    before = default_state() if state is None else state
    after = copy.deepcopy(before)
    result = VERBS[verb](after, dict(args), expected_revision, revision, by)
    validate_state(after)
    changed = after != before
    result.pop("unchanged", None)
    if not changed:
        result["unchanged"] = True
    return after, result, changed


def touched(result: dict[str, Any]) -> list[str]:
    """The scene ids a saved verb touched, for the script.changed event (empty: only the
    order, act breaks, numbering or the title page changed)."""
    found: list[Any] = []
    if isinstance(result.get("scene"), str):
        found.append(result["scene"])
    for key in ("items", "scenes", "created", "changed", "set_aside"):
        if isinstance(result.get(key), list):
            found.extend(result[key])
    out: list[str] = []
    for sid in found:
        if isinstance(sid, str) and _SCENE_ID.fullmatch(sid) and sid not in out:
            out.append(sid)
    return out


__all__ = ["ScriptConflict", "VERBS", "apply", "import_state", "normalize_text", "public_scene", "touched"]
del BREAK_ID_RE, NUMBER_RE, SCENE_ID_RE
