"""Nested cuts at export (lane w23-bins, CUT-180): a nested clip is replaced by what it shows.

Reference: Resolve / Premiere render a nested timeline as the inner timeline's picture and sound inside the clip's
range (w23-bins-STUDY.md). ``flatten_document`` rewrites the exported cut so every nested clip becomes the inner
cut's items for that range (recursively), before ``build_plan``. The film, its mix and every writer then work on
ordinary clips and stay frame-exact with the same verifiers; the EDL has no nesting anyway (it gets a warning).

Rules (Resolve's own where it has one):
- Picture: the inner V1 goes where the nested clip is and the inner V2 to the outer V2 (so the film composites
  them exactly as the inner film does), cut to the nested clip's ``[source_in, source_in + duration)``. Frames past
  the inner cut's end are black on V1 and empty on V2 (Resolve: "those areas ... simply go black"). A nest on V2
  whose range holds inner V2 clips, or inner V2 clips over the outer V2 clips, refuse (``nested_overlay``).
- Sound: a picture clip's own sound follows its piece; the inner A1-A3 clips go to the outer sound track of the
  same role (made when the cut has none and has room) and the inner song lane to the outer song lane when that is
  empty; anything that would overlap the outer cut's own sound is refused (``nested_sound_overlap``), never dropped.
- A retimed inner clip may be cut only at its tail by the nested clip's range (``nested_retime_cut`` otherwise).
- Transitions whose two clips stay adjacent and whole are kept; others are dropped and named in the warnings.
- A disabled nested clip shows nothing and sounds nothing.
Piece ids are ``<nest id>~<inner id>`` (unique, and a writer can tell which nest a piece came from).
"""
from __future__ import annotations

import copy
from fractions import Fraction
from typing import Any, Mapping

from .errors import ExportError

SEP = "~"
KIND = "cut"
MAX_DEPTH = 16


def _rate(value: Any) -> Fraction:
    return Fraction(value["num"], value["den"])


def _is_nest(item: Mapping[str, Any]) -> bool:
    return item.get("type") == "clip" and (item.get("source") or {}).get("kind") == KIND


def has_nests(timeline: Mapping[str, Any]) -> bool:
    return any(_is_nest(item) for track in timeline.get("tracks") or () if track.get("kind") == "video" for item in track.get("items") or ())


def _length(timeline: Mapping[str, Any]) -> int:
    from .model import timeline_end
    return timeline_end(timeline)


def _source_frames(item: Mapping[str, Any], frames: int, rate: Fraction) -> Fraction:
    """Source frames ``frames`` record frames of a clip read (constant speed; stills read none)."""
    speed = Fraction(1)
    retime = item.get("retime")
    if retime:
        speed = Fraction(retime["speed"]["num"], retime["speed"]["den"])
    return Fraction(frames) * (_rate(item["source_rate"]) if item.get("source_rate") else rate) * speed / rate


def _piece(item: Mapping[str, Any], head: int, length: int, rate: Fraction, nest_id: str) -> dict[str, Any]:
    """The part of clip ``item`` that starts ``head`` record frames into it and runs ``length`` frames."""
    piece = {k: copy.deepcopy(v) for k, v in item.items() if not k.startswith("__") and k not in ("record_start_frame", "attach")}
    piece["id"] = f"{nest_id}{SEP}{item['id']}"
    piece["duration_frames"] = length
    if head and (item.get("source") or {}).get("kind") != "picture":  # a still ignores source_in (model._clip)
        if item.get("retime"):
            raise ExportError("nested_retime_cut", f"The nested clip starts inside the retimed clip {item.get('label') or item['id']}; "
                                                   "a retimed clip is exported whole or from its first frame.")
        exact = _source_frames(item, head, rate)
        piece["source_in_frame"] = int(item["source_in_frame"]) + (exact.numerator // exact.denominator)
    if item.get("retime") and length < item["duration_frames"] - head and (item["retime"].get("curve")):
        raise ExportError("nested_retime_cut", f"The nested cut ends inside the speed ramp of {item.get('label') or item['id']}.")
    return piece


def _inner(state: Mapping[str, Any], ref: str, outer: Mapping[str, Any]) -> dict[str, Any]:
    inner = next((t for t in state.get("timelines") or () if t.get("id") == ref), None)
    if inner is None:
        raise ExportError("invalid_timeline", f"The nested cut {ref} is not in the document.")
    if inner["rate"] != outer["rate"]:
        raise ExportError("invalid_timeline", f"The nested cut {inner['name']} runs at another frame rate.")
    return inner


def _expanded(state: Mapping[str, Any], timeline: Mapping[str, Any], depth: int) -> dict[str, Any]:
    """``timeline`` with its own nests already flattened (inner cuts first, recursively)."""
    if depth > MAX_DEPTH:
        raise ExportError("invalid_timeline", "Nested cuts are nested too deep (or in a cycle).")
    if not has_nests(timeline):
        return copy.deepcopy(dict(timeline))
    return _flatten_timeline(state, timeline, depth, [])


def _track_pieces(inner: Mapping[str, Any], role: str, first: int, last: int) -> list[tuple[int, int, dict[str, Any] | None]]:
    """``[(start, end, item | None)]`` of the inner cut's ``role`` picture track inside ``[first, last)``, in inner record
    frames (None: nothing shown there). Each item carries ``__start``, its own record start."""
    track = next((t for t in inner["tracks"] if t.get("kind") == "video" and t.get("role") == role), None)
    if track is None or (role == "overlay" and track.get("enabled", True) is not True):
        return []
    out, at = [], 0
    for item in track.get("items") or ():
        start = at if role == "storyline" else item["record_start_frame"]
        end = start + item["duration_frames"]
        at = end if role == "storyline" else at
        lo, hi = max(start, first), min(end, last)
        if hi > lo:
            shown = item.get("type") == "clip" and item.get("enabled", True)
            out.append((lo, hi, {**item, "__start": start} if shown else None))
    return out


def _flatten_timeline(state: Mapping[str, Any], timeline: Mapping[str, Any], depth: int, warnings: list[str]) -> dict[str, Any]:
    out = copy.deepcopy(dict(timeline))
    nest_ids = {item["id"] for track in out["tracks"] if track.get("kind") == "video" for item in track.get("items") or () if _is_nest(item)}
    for tr in out.get("transitions") or ():
        if tr.get("left") in nest_ids or tr.get("right") in nest_ids:
            raise ExportError("nested_transition", f"Transition {tr.get('id')} joins a nested cut; the export blends clips inside one cut only, "
                                                   "and a transition into or out of a nested cut is not exported yet.")
    rate = _rate(out["rate"])
    extra_sounds: list[tuple[str, dict[str, Any]]] = []
    extra_music: list[dict[str, Any]] = []
    extra_overlay: list[dict[str, Any]] = []
    kept_transitions: list[dict[str, Any]] = []
    for track in out["tracks"]:
        if track.get("kind") != "video":
            continue
        storyline = track.get("role") == "storyline"
        items, at = [], 0
        for item in track.get("items") or ():
            start = at if storyline else item.get("record_start_frame", 0)
            at += item["duration_frames"] if storyline else 0
            if not _is_nest(item):
                items.append(item)
                continue
            inner = _expanded(state, _inner(state, item["source"]["ref"], timeline), depth + 1)
            first, length = int(item["source_in_frame"]), int(item["duration_frames"])
            last = first + length
            on = item.get("enabled", True) is True
            # The inner V1 goes where the nest is (gaps stay gaps on V1, nothing on V2); the inner V2 goes to V2.
            pieces: list[dict[str, Any]] = []
            for lo, hi, it in _track_pieces(inner, "storyline", first, last):
                if it is None or not on:
                    if storyline:
                        pieces.append({"id": f"{item['id']}{SEP}g{lo}", "type": "gap", "duration_frames": hi - lo})
                    continue
                piece = _piece(it, lo - it["__start"], hi - lo, rate, item["id"])
                if not storyline:
                    piece["record_start_frame"] = start + (lo - first)
                pieces.append(piece)
            covered = sum(p["duration_frames"] for p in pieces) if storyline else 0
            if storyline and covered < length:  # past the inner cut's end: black (Resolve)
                pieces.append({"id": f"{item['id']}{SEP}black", "type": "gap", "duration_frames": length - covered})
            items.extend(pieces)
            if not on:
                continue
            for lo, hi, it in _track_pieces(inner, "overlay", first, last):
                if it is None:
                    continue
                if not storyline:
                    raise ExportError("nested_overlay", f"The nested cut {inner['name']} on V2 has a V2 clip of its own "
                                                        f"({it.get('label') or it['id']}); the film has one picture track above V1.")
                piece = _piece(it, lo - it["__start"], hi - lo, rate, item["id"])
                piece["record_start_frame"] = start + (lo - first)
                extra_overlay.append(piece)
            whole = {p["id"] for p in pieces if p.get("type") == "clip"}
            for tr in inner.get("transitions") or ():
                left, right = f"{item['id']}{SEP}{tr['left']}", f"{item['id']}{SEP}{tr['right']}"
                if storyline and left in whole and right in whole:
                    kept_transitions.append({**copy.deepcopy(tr), "id": f"{item['id']}{SEP}{tr['id']}", "left": left, "right": right})
                else:
                    warnings.append(f"The nested cut {inner['name']}: transition {tr['id']} lies outside the nested clip's range and is left out.")
            for inner_track in inner["tracks"]:
                if inner_track.get("kind") == "audio" and inner_track.get("mute") is not True:
                    for sound in inner_track.get("items") or ():
                        s, e = sound["record_start_frame"], sound["record_start_frame"] + sound["duration_frames"]
                        lo, hi = max(s, first), min(e, last)
                        if hi <= lo or sound.get("enabled", True) is not True:
                            continue
                        piece = {k: copy.deepcopy(v) for k, v in sound.items() if k not in ("attach", "link")}
                        piece["id"] = f"{item['id']}{SEP}{sound['id']}"
                        piece["record_start_frame"] = start + (lo - first)
                        piece["duration_frames"] = hi - lo
                        if lo > s:
                            exact = Fraction(lo - s) * (_rate(sound["source_rate"]) if sound.get("source_rate") else rate) / rate
                            piece["source_in_frame"] = int(sound["source_in_frame"]) + (exact.numerator // exact.denominator)
                            piece["fade_in_frames"] = max(0, int(piece.get("fade_in_frames") or 0) - (lo - s))
                        if hi < e:
                            piece["fade_out_frames"] = max(0, int(piece.get("fade_out_frames") or 0) - (e - hi))
                        extra_sounds.append((inner_track["role"], piece))
                elif inner_track.get("kind") == "music" and isinstance(inner_track.get("music"), dict) and inner_track.get("enabled", True):
                    music = inner_track["music"]
                    s, e = music["record_start_frame"], music["record_start_frame"] + music["duration_frames"]
                    lo, hi = max(s, first), min(e, last)
                    if hi > lo:
                        moved = copy.deepcopy(music)
                        moved["record_start_frame"] = start + (lo - first)
                        moved["source_in_frame"] = int(music.get("source_in_frame") or 0) + (lo - s)
                        moved["duration_frames"] = hi - lo
                        extra_music.append(moved)
        track["items"] = items
    _place_overlay(out, extra_overlay)
    out["transitions"] = [*(out.get("transitions") or ()), *kept_transitions]
    _place_sounds(out, extra_sounds)
    _place_music(out, extra_music)
    return out


def _place_overlay(timeline: dict[str, Any], pieces: list[dict[str, Any]]) -> None:
    if not pieces:
        return
    track = next((t for t in timeline["tracks"] if t.get("kind") == "video" and t.get("role") == "overlay"), None)
    if track is None:
        raise ExportError("nested_overlay", "The nested cut has V2 clips; this cut has no V2 track.")
    for piece in pieces:
        s, e = piece["record_start_frame"], piece["record_start_frame"] + piece["duration_frames"]
        for other in track["items"]:
            if s < other["record_start_frame"] + other["duration_frames"] and other["record_start_frame"] < e:
                raise ExportError("nested_overlay", f"The nested V2 clip {piece.get('label') or piece['id']} overlaps "
                                                    f"{other.get('label') or other['id']} on V2.")
        track["items"].append(piece)
    track["items"].sort(key=lambda i: i["record_start_frame"])


def _place_sounds(timeline: dict[str, Any], sounds: list[tuple[str, dict[str, Any]]]) -> None:
    for role, sound in sounds:
        track = next((t for t in timeline["tracks"] if t.get("kind") == "audio" and t.get("role") == role), None)
        if track is None:
            ids = {t["id"] for t in timeline["tracks"]}
            free = next((f"A{n}" for n in (1, 2, 3) if f"A{n}" not in ids), None)
            if free is None or sum(t.get("kind") == "audio" for t in timeline["tracks"]) >= 3:
                raise ExportError("nested_sound_overlap", f"The nested sound {sound.get('label') or sound['id']} needs a {role} track; the cut has three sound tracks.")
            # The document's empty sound track (timeline/schema.py new_audio_track).
            track = {"id": free, "kind": "audio", "role": role, "gain_db": 0, "mute": False, "solo": False, "locked": False, "items": []}
            timeline["tracks"].append(track)
        s, e = sound["record_start_frame"], sound["record_start_frame"] + sound["duration_frames"]
        for other in track["items"]:
            if s < other["record_start_frame"] + other["duration_frames"] and other["record_start_frame"] < e:
                raise ExportError("nested_sound_overlap", f"The nested sound {sound.get('label') or sound['id']} overlaps "
                                                          f"{other.get('label') or other['id']} on {track['id']}.")
        track["items"].append(sound)
        track["items"].sort(key=lambda i: i["record_start_frame"])


def _place_music(timeline: dict[str, Any], music: list[dict[str, Any]]) -> None:
    if not music:
        return
    lane = next((t for t in timeline["tracks"] if t.get("kind") == "music"), None)
    if lane is None or lane.get("music") is not None or len(music) > 1:
        raise ExportError("nested_sound_overlap", "The nested cut's song needs the song lane, and this cut already has a song there.")
    lane["music"] = music[0]


def flatten_document(document: Mapping[str, Any], timeline_id: str | None = None) -> tuple[dict[str, Any], list[str]]:
    """``(document with the exported cut's nests replaced by their pieces, warnings)``; unchanged without nests."""
    envelope = copy.deepcopy(dict(document))
    state = envelope["state"] if "state" in envelope else envelope
    if not isinstance(state, dict) or not isinstance(state.get("timelines"), list):
        return envelope, []
    wanted = timeline_id or state.get("active")
    index = next((i for i, t in enumerate(state["timelines"]) if isinstance(t, dict) and t.get("id") == wanted), None)
    if index is None or not has_nests(state["timelines"][index]):
        return envelope, []
    warnings: list[str] = []
    names = {t.get("id"): t.get("name") for t in state["timelines"] if isinstance(t, dict)}
    nests = [names.get(item["source"]["ref"]) or item["source"]["ref"] for track in state["timelines"][index]["tracks"]
             if track.get("kind") == "video" for item in track.get("items") or () if _is_nest(item)]
    state_before = copy.deepcopy(state["timelines"][index]["tracks"])
    state["timelines"][index] = _flatten_timeline(state, state["timelines"][index], 0, warnings)
    envelope["__nests"] = list(dict.fromkeys(nests))
    # The nested clips of the exported cut by id -> the nested cut's name (interchange writers group their pieces).
    envelope["__nest_ids"] = {item["id"]: names.get(item["source"]["ref"]) or item["source"]["ref"] for track in state_before
                              if track.get("kind") == "video" for item in track.get("items") or () if _is_nest(item)}
    return envelope, warnings


def storyline_runs(storyline: list[Any], nest_ids: Mapping[str, str]) -> list[tuple[str | None, str, list[Any]]]:
    """The storyline as ``[(nest id | None, nest name, items)]``: each nested clip's pieces (ids ``<nest id>~...``) in one
    run, every other item on its own. Interchange writers write a run as one nested sequence (compound clip)."""
    runs: list[tuple[str | None, str, list[Any]]] = []
    for item in storyline:
        owner = item.id.split(SEP, 1)[0] if SEP in item.id else None
        if owner is not None and owner in nest_ids:
            if runs and runs[-1][0] == owner:
                runs[-1][2].append(item)
            else:
                runs.append((owner, nest_ids[owner], [item]))
        else:
            runs.append((None, "", [item]))
    return runs


def nest_names(plan_ids: list[str]) -> set[str]:
    """The nest ids a plan's item ids came from (``<nest id>~...``)."""
    return {ident.split(SEP, 1)[0] for ident in plan_ids if SEP in ident}
