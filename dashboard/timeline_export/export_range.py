"""Export a frame range of a timeline (the artist's IN-OUT): the document cut to ``[in, out)``.

Pure and copy-only: the storyline keeps the items that overlap the range, the first and last cut
at the range edges (a clip's source in moves by the head it loses, at its own source rate); V2
clips, the sound tracks A1-A3, the music lane and the markers move by ``-in`` and are cut to the range too. Frames are
record frames of the timeline. The result is a normal studio.timeline/1 document, so every
export kind (film, FCPXML, FCP7, EDL, OTIO) handles a range the same way as a whole cut.
"""
from __future__ import annotations

import copy
from fractions import Fraction
from typing import Any, Mapping

from .errors import ExportError
from .model import timeline_end


def _rate(value: Any) -> Fraction | None:
    if isinstance(value, Mapping) and type(value.get("num")) is int and type(value.get("den")) is int and value["num"] > 0 and value["den"] > 0:
        return Fraction(value["num"], value["den"])
    return None


def _source_head(item: Mapping[str, Any], lost: int, rate: Fraction) -> int:
    """Source frames a clip loses when ``lost`` record frames are cut from its head."""
    source_rate = _rate(item.get("source_rate")) or rate
    exact = Fraction(lost) * source_rate / rate
    return int(exact) if exact.denominator == 1 else int(exact + Fraction(1, 2))


def _crop_sound(items: list[Any], in_frame: int, out_frame: int, rate: Fraction) -> list[dict[str, Any]]:
    """A1-A3 (lane p13-cut, CUT-116): every sound clip is cut like a V2 clip (Resolve Deliver In/Out renders
    every track; FCPXML gives every audio asset-clip its own offset/start). Volume keys sit in source frames, so
    they stay on the same sound; a fade the clip loses with its head or tail shrinks by the lost frames."""
    kept = []
    for item in items:
        if not isinstance(item, dict) or type(item.get("record_start_frame")) is not int:
            continue
        start = item["record_start_frame"]
        end = start + int(item.get("duration_frames") or 0)
        a, b = max(start, in_frame), min(end, out_frame)
        if b <= a:
            continue
        piece = dict(item)
        piece["record_start_frame"] = a - in_frame
        piece["duration_frames"] = b - a
        if a > start and "source_in_frame" in piece:
            piece["source_in_frame"] = int(piece["source_in_frame"]) + _source_head(item, a - start, rate)
        if "fade_in_frames" in piece:
            piece["fade_in_frames"] = max(0, int(piece["fade_in_frames"] or 0) - (a - start))
        if "fade_out_frames" in piece:
            piece["fade_out_frames"] = max(0, int(piece["fade_out_frames"] or 0) - (end - b))
        kept.append(piece)
    return kept


def crop_document(document: Mapping[str, Any], in_frame: int, out_frame: int, *, timeline_id: str | None = None) -> dict[str, Any]:
    if type(in_frame) is not int or type(out_frame) is not int or in_frame < 0 or out_frame <= in_frame:
        raise ExportError("invalid_range", "The range needs whole frames with in < out.")
    envelope = copy.deepcopy(dict(document))
    state = envelope["state"] if "state" in envelope else envelope
    if not isinstance(state, dict) or not isinstance(state.get("timelines"), list):
        raise ExportError("invalid_timeline", "The timeline document has no timelines.")
    wanted = timeline_id or state.get("active")
    timeline = next((t for t in state["timelines"] if isinstance(t, dict) and t.get("id") == wanted), None)
    if timeline is None:
        raise ExportError("timeline_not_found", "The requested timeline is not in the document.")
    rate = _rate(timeline.get("rate")) or Fraction(24)
    # w22-dlv (CUT-178): a range is rendered as asked or refused, never shortened (Resolve: In/Out stay inside the
    # timeline). Past the picture the storyline continues in black (filled below), so the range keeps its length.
    end = timeline_end(timeline)
    if out_frame > end:
        raise ExportError("range_past_end", f"The range ends at {out_frame} f, the timeline at {end} f.",
                          {"out_frame": out_frame, "timeline_end": end})
    storyline_length = 0
    for track in timeline.get("tracks") or []:
        if not isinstance(track, dict):
            continue
        if track.get("kind") == "video" and track.get("role") == "storyline":
            kept, at = [], 0
            for item in track.get("items") or []:
                length = item.get("duration_frames") if isinstance(item, dict) else None
                if type(length) is not int:
                    raise ExportError("invalid_timeline", "Every storyline item needs duration_frames.")
                start, end = at, at + length
                at = end
                a, b = max(start, in_frame), min(end, out_frame)
                if b <= a:
                    continue
                piece = dict(item)
                piece["duration_frames"] = b - a
                if a > start and piece.get("type") == "clip" and "source_in_frame" in piece:
                    piece["source_in_frame"] = int(piece["source_in_frame"]) + _source_head(item, a - start, rate)
                kept.append(piece)
            storyline_length = sum(i["duration_frames"] for i in kept)
            if storyline_length < out_frame - in_frame:
                kept.append({"id": "range-black", "type": "gap", "duration_frames": out_frame - in_frame - storyline_length,
                             "label": "Black"})
                storyline_length = out_frame - in_frame
            track["items"] = kept
        elif track.get("kind") == "video":
            kept = []
            for item in track.get("items") or []:
                if not isinstance(item, dict) or type(item.get("record_start_frame")) is not int:
                    continue
                start = item["record_start_frame"]
                end = start + int(item.get("duration_frames") or 0)
                a, b = max(start, in_frame), min(end, out_frame)
                if b <= a:
                    continue
                piece = dict(item)
                piece["record_start_frame"] = a - in_frame
                piece["duration_frames"] = b - a
                if a > start and "source_in_frame" in piece:
                    piece["source_in_frame"] = int(piece["source_in_frame"]) + _source_head(item, a - start, rate)
                kept.append(piece)
            track["items"] = kept
        elif track.get("kind") == "audio":
            track["items"] = _crop_sound(track.get("items") or [], in_frame, out_frame, rate)
        elif track.get("kind") == "music" and isinstance(track.get("music"), dict):
            music = dict(track["music"])
            start = int(music.get("record_start_frame") or 0)
            end = start + int(music.get("duration_frames") or 0)
            a, b = max(start, in_frame), min(end, out_frame)
            if b <= a:
                track["music"] = None
                continue
            music["source_in_frame"] = int(music.get("source_in_frame") or 0) + (a - start)
            music["record_start_frame"] = a - in_frame
            music["duration_frames"] = b - a
            track["music"] = music
    if storyline_length <= 0:
        raise ExportError("invalid_range", "The range holds no part of the storyline.")
    timeline["markers"] = [
        {**m, "frame": m["frame"] - in_frame} for m in (timeline.get("markers") or [])
        if isinstance(m, dict) and type(m.get("frame")) is int and in_frame <= m["frame"] < out_frame
    ]
    timeline["start_tc_frame"] = int(timeline.get("start_tc_frame") or 0) + in_frame
    return envelope
