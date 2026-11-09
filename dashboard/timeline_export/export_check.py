"""Check before render (lane w2-cutdel): what would stop or change an export, before it starts.

``check_document(document, resolver, timeline_id=None, rng=None)`` reads one cut the way
``build_plan`` does and answers every finding at once (``build_plan`` stops at the first):

* ``offline`` (block): a picture clip, sound clip or the song lane whose file is not there. Each item
  names the clip ids and the fix ``locate`` (relink the file); the export would refuse with
  ``source_offline``.
* ``invalid`` (block): the cut itself cannot be exported (the plan's own refusal, e.g. a clip that
  reads past the end of its file, a source rate that is not the file's measured rate).
* ``offline_outside`` (warn, lane w25-cutfix CUT-182): offline clips outside the range; the range renders, but
  the check never reads "all online" while a source of the cut is gone.
* ``rate_mismatch`` (warn): moving sources at another rate than the cut, grouped by rate (a 48 fps
  take on a 24 fps cut shows every second frame: plays at speed, not slow motion).
* ``variable_rate`` (warn): sources with a variable frame rate (a phone clip): conformed by frame
  index at their nominal rate, every recorded frame once.

``ok`` is False while any item blocks (the Deliver page keeps Render disabled and offers Locate...).
Only clips inside the optional range ``{in_frame, out_frame}`` count.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Mapping

from .errors import ExportError
from .model import build_plan, load_timeline, timeline_end
from .timebase import ffmpeg_rate

SCHEMA = "studio.timeline-export-check/1"


def _rate(value: Any) -> Fraction | None:
    if isinstance(value, Mapping) and type(value.get("num")) is int and type(value.get("den")) is int and value["num"] > 0 and value["den"] > 0:
        return Fraction(value["num"], value["den"])
    return None


def _label(rate: Fraction) -> str:
    text = f"{float(rate):.3f}".rstrip("0").rstrip(".")
    return f"{text} fps"


def _inside(start: int, duration: int, rng: tuple[int, int] | None) -> bool:
    return rng is None or (start < rng[1] and rng[0] < start + duration)


def check_document(document: Mapping[str, Any], resolver: Any, *, timeline_id: str | None = None,
                   rng: tuple[int, int] | None = None) -> dict[str, Any]:
    _envelope, timeline = load_timeline(document, timeline_id)
    cut_rate = _rate(timeline.get("rate")) or Fraction(24)
    offline: dict[str, list[str]] = {"picture": [], "sound": [], "music": []}
    outside: dict[str, list[str]] = {"picture": [], "sound": []}
    by_rate: dict[Fraction, list[str]] = {}
    vfr: dict[Fraction, list[str]] = {}
    resolve_sound = getattr(resolver, "resolve_sound", None)
    for track in timeline.get("tracks") or ():
        if not isinstance(track, Mapping):
            continue
        kind = track.get("kind")
        if kind == "music":
            music = track.get("music")
            if isinstance(music, Mapping) and track.get("enabled", True) is True and resolver.resolve_music(music) is None:
                offline["music"].append(str(track.get("id")))
            continue
        cursor = 0
        for item in track.get("items") or ():
            if not isinstance(item, Mapping):
                continue
            duration = item.get("duration_frames") if type(item.get("duration_frames")) is int else 0
            start = item.get("record_start_frame", cursor) if kind == "audio" or track.get("role") == "overlay" else cursor
            cursor = start + duration
            if item.get("type") != "clip" or not isinstance(item.get("source"), Mapping):
                continue
            ident = str(item.get("id"))
            if not _inside(start, duration, rng):
                # w25-cutfix (CUT-182; Resolve keeps listing a timeline's offline media whatever the render range): a
                # clip outside the range does not block (CUT-105), but its offline file is named, never "all online".
                if kind == "audio":
                    if item.get("enabled", True) is True and (resolve_sound is None or resolve_sound(item["source"]) is None):
                        outside["sound"].append(ident)
                elif resolver.resolve(item["source"]) is None:
                    outside["picture"].append(ident)
                continue
            if kind == "audio":
                if item.get("enabled", True) is True and (resolve_sound is None or resolve_sound(item["source"]) is None):
                    offline["sound"].append(ident)
                continue
            info = resolver.resolve(item["source"])
            if info is None:
                offline["picture"].append(ident)
                continue
            if info.frames is None:
                continue  # a still has no rate
            source_rate = _rate(item.get("source_rate")) or info.rate
            if source_rate is not None and source_rate != cut_rate:
                by_rate.setdefault(source_rate, []).append(ident)
            if info.vfr:
                vfr.setdefault(info.rate, []).append(ident)
    items: list[dict[str, Any]] = []
    gone = [*offline["picture"], *offline["sound"], *offline["music"]]
    if gone:
        parts = [f"{len(v)} {k}" + ("" if k == "music" else " clip" + ("s" if len(v) > 1 else "")) for k, v in offline.items() if v]
        items.append({"code": "offline", "severity": "block", "clips": gone, "fix": "locate",
                      "message": "Offline: " + ", ".join(parts) + ".",
                      "hint": "The files moved (own footage and sound come back by adding the file again from its new folder); the export refuses offline sources."})
    away = [*outside["picture"], *outside["sound"]]
    if away:
        parts = [f"{len(v)} {k} clip" + ("s" if len(v) > 1 else "") for k, v in outside.items() if v]
        items.append({"code": "offline_outside", "severity": "warn", "clips": away, "fix": "locate",
                      "message": "Offline: " + ", ".join(parts) + ", outside the range.",
                      "hint": "The range renders without them; the entire timeline is refused until the files are back."})
    for rate, clips in sorted(by_rate.items()):
        speed = rate / cut_rate
        what = (f"every {speed} source frames show as one" if speed.denominator == 1 and speed > 1 else
                "frames repeat" if speed < 1 else "frames are picked by nearest time")
        items.append({"code": "rate_mismatch", "severity": "warn", "clips": clips, "fix": "retime",
                      "source_rate": {"num": rate.numerator, "den": rate.denominator},
                      "message": f"{len(clips)} clip{'s' if len(clips) > 1 else ''} at {_label(rate)} on a {_label(cut_rate)} cut: plays at speed, {what}.",
                      "hint": (f"For slow motion set the clip's speed to {float(cut_rate / rate * 100):.4g} % (set_retime); "
                               "for real time leave it; a cut takes its rate while it is empty (set_format rate).")})
    for rate, clips in sorted(vfr.items(), key=lambda pair: pair[0] or 0):
        items.append({"code": "variable_rate", "severity": "warn", "clips": clips, "fix": "transcode",
                      "message": f"{len(clips)} clip{'s' if len(clips) > 1 else ''} with a variable frame rate, conformed by frame at {_label(rate)}.",
                      "hint": f"Every recorded frame plays once at {ffmpeg_rate(rate)} fps; for exact sound sync transcode the file to a constant rate and relink it."})
    if not gone:
        try:
            build_plan(document, resolver, timeline_id=timeline_id)
        except ExportError as exc:
            if exc.code != "source_offline":
                items.insert(0, {"code": "invalid", "severity": "block", "clips": [], "fix": None, "message": exc.message,
                                 "hint": "The cut is not exportable as it is (the Cut page or continuo_timeline_apply edits it)."})
    # w22-dlv (CUT-178): the preflight refuses a range past the timeline end exactly like the export (export_range).
    end = timeline_end(timeline)
    if rng is not None and rng[1] > end:
        items.insert(0, {"code": "range_past_end", "severity": "block", "clips": [], "fix": None,
                         "message": f"The range ends at {rng[1]} f, the timeline at {end} f.",
                         "hint": "Deliver renders a range inside the timeline only; OUT is past the timeline end."})
    blocking = sum(1 for item in items if item["severity"] == "block")
    return {"schema": SCHEMA, "ok": blocking == 0, "blocking": blocking, "warnings": len(items) - blocking,
            "rate": {"num": cut_rate.numerator, "den": cut_rate.denominator}, "items": items}
