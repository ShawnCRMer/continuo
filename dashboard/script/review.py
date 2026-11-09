"""Hermes' changes under review (lane w14-scr, SCR-071, board D-1 Tafel 1 A): the record's rules, no I/O.

``script/review.json`` = {hermes_revision, at, label, scenes: {scene id: the scene's text before Hermes}}. A Hermes save
adds the before-text of every scene it changed (a scene already under review keeps its first baseline); every save
drops a scene whose text is back at its baseline, omitted or gone. The page diffs baseline -> current text into
changes: Accept moves the baseline ({scene: text}), Accept all clears the record, Undo writes the old run back.
Reference: Final Draft 13 Revision Mode (revised text marked) + VS Code Keep/Undo (per change, against a snapshot).
Used by store.py (under the project lock) and by checks/lib/script-backend.py (the page-live fake), so both agree.
"""
from __future__ import annotations

from typing import Any

from .ops import ScriptConflict, normalize_text
from .schema import fail

KEYS = {"hermes_revision", "at", "label", "scenes"}
UNDO_LABEL = "Undo Hermes change"


def empty() -> dict[str, Any]:
    return {"hermes_revision": 0, "at": None, "label": "", "scenes": {}}


def check(value: Any) -> dict[str, Any]:
    if (type(value) is not dict or set(value) != KEYS or type(value["scenes"]) is not dict
            or type(value["hermes_revision"]) is not int or not all(type(v) is str for v in value["scenes"].values())):
        fail("invalid_script", "review.json: not a review record")
    return value


def prune(record: dict[str, Any], state: dict[str, Any] | None) -> dict[str, Any]:
    scenes = (state or {}).get("scenes") or {}
    keep = {sid: text for sid, text in record["scenes"].items()
            if sid in scenes and not scenes[sid].get("omitted") and scenes[sid]["text"] != text}
    return {**record, "scenes": keep}


def after_save(record: dict[str, Any], old: dict[str, Any] | None, new: dict[str, Any], by: str,
               revision: int, stamp: str | None, label: str, wholesale: bool = False) -> dict[str, Any]:
    """``wholesale``: the save replaced scene text as a whole (restore, import, write_fountain). By the artist it drops
    every scene it changed: they are no longer Hermes' pending changes (w14-fix, S5; Final Draft 13: restoring a
    revision clears its revision marks). By Hermes it records baselines as any Hermes save does."""
    scenes = dict(record["scenes"])
    before = (old or {}).get("scenes") or {}
    if wholesale and by != "hermes":
        for sid in list(scenes):
            was, now = before.get(sid), new["scenes"].get(sid)
            if (None if was is None else (was.get("omitted"), was["text"])) != (None if now is None else (now.get("omitted"), now["text"])):
                del scenes[sid]
    if by == "hermes":
        for sid, scene in new["scenes"].items():
            was = before.get(sid)
            was_text = "" if was is None or was.get("omitted") else was["text"]
            if scene["text"] != was_text and sid not in scenes:
                scenes[sid] = was_text
        record = {**record, "hermes_revision": revision, "at": stamp, "label": label[:120]}
    return prune({**record, "scenes": scenes}, new)


def admit_undo(record: dict[str, Any], state: dict[str, Any] | None, scenes: Any) -> None:
    """An undo writes back only scenes that are in the record now (w14-fix, S1); read under the lock of the write."""
    open_scenes = prune(record, state)["scenes"]
    for sid in scenes:
        if sid not in open_scenes:
            fail("review_unknown_scene", f"undo: scene {sid} is not among Hermes' changes")


def admit_decision(record: dict[str, Any], expected_hermes_revision: Any, current: dict[str, Any]) -> None:
    """Accept / Accept all name the record they were decided on (w14-fix, S2 + S3); ``current`` = the record now."""
    # Why this guard and not the script revision: an artist hand edit moves the script revision but not Hermes' changes;
    # the page's hand-edit baseline posts must still land while the artist types. Only a NEW Hermes save makes an older
    # page decision stale.
    if expected_hermes_revision is None:
        fail("invalid_op", "baselines and clear need expected_hermes_revision (the record's hermes_revision)")
    if type(expected_hermes_revision) is not int or expected_hermes_revision < 0:
        fail("invalid_op", "expected_hermes_revision must be an integer >= 0")
    if expected_hermes_revision != record["hermes_revision"]:
        raise ScriptConflict("review_conflict", f"Hermes changed the script after hermes_revision {expected_hermes_revision}"
                             f" (now {record['hermes_revision']}); nothing was accepted", {"current": current})


def admit_base(state: dict[str, Any] | None, baselines: Any, base_revision: Any, current: dict[str, Any]) -> None:
    """SCR-073 (lane w16-scr; w14 cold probe S2b): a baseline post names the script revision its texts were computed on
    (``base_revision``, the page's editor revision). A scene it touches that the artist saved after that revision holds
    words the posting window never saw; its baseline would draw them as Hermes' change, so the post is refused
    (review_stale_base, details.current = the record now). Any later save counts (the scene's last writer alone
    cannot tell whether the artist wrote in between). A post without base_revision (the agent tool) is judged by
    expected_hermes_revision alone, as before. Reference: Final Draft 13 collaboration
    and Google Docs refuse an edit made on an older copy instead of applying it."""
    if base_revision is None or baselines is None:
        return
    if type(base_revision) is not int or base_revision < 0:
        fail("invalid_op", "base_revision must be an integer >= 0")
    scenes = (state or {}).get("scenes") or {}
    stale = sorted(sid for sid in baselines if sid in scenes and scenes[sid]["revision"] > base_revision)
    if stale:
        raise ScriptConflict("review_stale_base", f"scene {stale[0]} changed at revision {scenes[stale[0]]['revision']} "
                             f"after base_revision {base_revision}; nothing was accepted", {"current": current, "scenes": stale})


def apply(record: dict[str, Any], state: dict[str, Any] | None, baselines: Any, clear: bool) -> dict[str, Any]:
    """Accept: ``baselines`` {scene id: text | None (accept the whole scene)}; ``clear`` = Accept all."""
    if baselines is not None and (type(baselines) is not dict or len(baselines) > 2000):
        fail("invalid_op", "baselines: an object {scene id: text or null}")
    for value in (baselines or {}).values():
        if value is not None and (type(value) is not str or len(value) > 65536):
            fail("invalid_op", "baselines: each value is the scene text (at most 64 KiB) or null")
    scenes = (state or {}).get("scenes") or {}
    for sid in baselines or {}:
        if sid not in scenes:
            fail("review_unknown_scene", f"baselines: no scene {sid} in the script")
    merged = {} if clear else dict(record["scenes"])
    for sid, text in (baselines or {}).items():
        if text is None:
            merged.pop(sid, None)
        else:
            merged[sid] = normalize_text(text, "baselines")
    return prune({**record, "scenes": merged}, state)
