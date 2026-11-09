"""Write one export package (film | fcpxml | edl) into its own folder, atomically.

Everything is built in ``.<name>.partial-<hex>`` next to the final folder; the
FCPXML verifier and the frame-count checks run there; only a green package is
renamed into place.  A failure removes the partial folder (fail closed).

Package (archived interchange skill "strong package"):

- film:   ``<name>.mp4`` + ``manifest.json``
- fcpxml: ``<name>.fcpxml`` + ``<name>.edl`` + ``<name>-clips.csv`` +
          ``<name>-markers.csv`` + ``media/`` (video-only remuxes) +
          ``README.txt`` + ``manifest.json`` (verifier report inside)
- fcp7:   ``<name>.xml`` (FCP7 xmeml 5, Premiere Pro) + the same EDL, CSVs, media, README, manifest
- otio:   ``<name>.otio`` (OpenTimelineIO Timeline.1 JSON) + the same EDL, CSVs, media, README, manifest
- edl:    ``<name>.edl`` + both CSVs + ``media/`` + ``README.txt`` + ``manifest.json``

Every interchange file is checked by its verifier in the staging folder and,
without decoding, again after the rename; any red gate removes the package.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from .errors import ExportError
from .export_edl import build_edl, build_marker_edl
from .export_fcp7 import XMEML_VERSION, build_fcp7
from .export_fcpxml import FCPXML_VERSION, build_fcpxml
from .export_otio import build_otio, dumps as otio_dumps, validate_otio
from .export_verify import verify
from .export_verify_fcp7 import verify_fcp7
from .ffmpeg_run import CancelFn, ffmpeg_version, run_ffmpeg
from .model import Plan
from .render_film import RESOLVE_FREE_DECODES, clip_table, film_suffix, render_film
from .resolver import count_frames
from .timebase import timecode

MANIFEST_SCHEMA = "studio.timeline-export/1"
KINDS = ("film", "fcpxml", "fcp7", "edl", "otio")
PRIMARY_SUFFIX = {"fcpxml": ".fcpxml", "fcp7": ".xml", "edl": ".edl", "otio": ".otio"}
APP_LABEL = {"fcpxml": "DaVinci Resolve", "fcp7": "Premiere Pro (or Resolve)", "edl": "any NLE", "otio": "an OpenTimelineIO reader"}
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")

ProgressFn = Callable[[float, str], None]


@dataclass
class ExportResult:
    folder: Path
    manifest: dict[str, Any]


def safe_name(text: str, fallback: str = "timeline") -> str:
    cleaned = _SAFE.sub("-", text).strip("-._")[:60]
    return cleaned or fallback


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def remux_video_only(plan: Plan, media_dir: Path, *, ffmpeg: Path, ffprobe: Path, should_cancel: CancelFn | None) -> dict[str, Path]:
    """Full-length video-only stream copies, one per unique moving source; collision-safe names."""
    media_dir.mkdir(parents=True, exist_ok=True)
    remuxes: dict[str, Path] = {}
    for clip in [*plan.storyline, *plan.overlays]:
        if clip.kind != "clip" or clip.info is None or clip.info.is_still:
            continue
        ref = clip.source["ref"]
        if ref in remuxes:
            continue
        tag = hashlib.sha256(ref.encode("utf-8")).hexdigest()[:8]
        target = media_dir / f"{len(remuxes) + 1:03d}-{safe_name(clip.info.path.stem, 'clip')}-{tag}.mp4"
        if clip.info.vfr:
            conform_vfr(clip.info, target, ffmpeg=ffmpeg, should_cancel=should_cancel)
        else:
            run_ffmpeg([str(ffmpeg), "-y", "-i", str(clip.info.path), "-map", "0:v:0", "-c:v", "copy", "-an",
                        "-movflags", "+faststart", "-map_metadata", "-1", str(target)],
                       should_cancel=should_cancel, what=f"Remux {clip.info.path.name}")
        frames = count_frames(ffprobe, target)
        if frames != clip.info.frames:
            raise ExportError("remux_mismatch", f"The remux of {clip.info.path.name} has {frames} frames, the source {clip.info.frames}.")
        remuxes[ref] = target
    return remuxes


# A variable-frame-rate source cannot be stream-copied for an NLE: Resolve and Premiere place its
# frames by time and the cut would land on other frames. It is re-encoded once (x264 CRF 10, every
# frame kept, none added) at its nominal rate, frame N at N / rate: the frame-index conform the film
# render uses too (render_film._conform).
VFR_NOTE = "A clip with a variable frame rate (phone) is conformed by frame: every recorded frame once at {rate} fps; its sound may drift."


def conform_vfr(info: Any, target: Path, *, ffmpeg: Path, should_cancel: CancelFn | None) -> None:
    rate = f"{info.rate.numerator}/{info.rate.denominator}"
    run_ffmpeg([str(ffmpeg), "-y", "-i", str(info.path), "-map", "0:v:0", "-vf", f"setpts=N/({rate})/TB",
                "-fps_mode", "cfr", "-r", rate, "-c:v", "libx264", "-preset", "medium", "-crf", "10",
                "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", "-map_metadata", "-1", str(target)],
               should_cancel=should_cancel, what=f"Conform {info.path.name}")


def vfr_warnings(plan: Plan) -> list[str]:
    rates: list[str] = []
    for clip in [*plan.storyline, *plan.overlays]:
        if clip.kind == "clip" and clip.info is not None and clip.info.vfr and clip.info.rate is not None:
            text = f"{float(clip.info.rate):.3f}".rstrip("0").rstrip(".")
            if text not in rates:
                rates.append(text)
    return [VFR_NOTE.format(rate=rate) for rate in rates]


CURVE_NOTE = ("{format} cannot carry a speed ramp ({why}): {count} clip(s) are written at their average speed "
              "({names}). FCPXML and FCP7 XML carry the ramp exactly; the reference movie shows it exactly.")
CURVE_WHY = {
    "OTIO": "its core schema has one LinearTimeWarp speed per clip; the keys are in metadata film_suite.speed_curve",
    "EDL": "CMX 3600 M2 has one constant speed per event",
}
# Interchange files per package kind (every non-film package also holds the EDLs).
KIND_FORMATS = {"film": (), "fcpxml": ("EDL",), "fcp7": ("EDL",), "otio": ("OTIO", "EDL"), "edl": ("EDL",)}


def curve_warnings(plan: Plan, kind: str | None = None) -> list[str]:
    """Speed curves (lanes w6-tempo, w7-export): one warning per format that cannot carry a clip's
    ramp, naming the format, why, and the clips. FCPXML (timeMap) and FCP7 XML (Time Remap graph)
    carry every curve exactly; OTIO and EDL carry one speed per clip, so a curve that is one speed
    over the clip (or a pure hold) is exact there and only a real ramp is named. ``kind``: only the
    formats that package writes (None = every format, for the agent's check)."""
    from .export_curve import CONSTANT_ONLY_FORMATS, ramped

    names = [str(c.label or c.id or "clip") for c in [*plan.storyline, *plan.overlays] if ramped(c, plan.rate)]
    if not names:
        return []
    formats = CONSTANT_ONLY_FORMATS if kind is None else [f for f in CONSTANT_ONLY_FORMATS if f in KIND_FORMATS.get(kind, ())]
    listed = ", ".join(names[:5]) + (" …" if len(names) > 5 else "")
    return [CURVE_NOTE.format(format=f, why=CURVE_WHY[f], count=len(names), names=listed) for f in formats]


def film_note(quality: str) -> str:
    """What the film preset means for Resolve (empty when Resolve free plays it)."""
    if RESOLVE_FREE_DECODES.get(quality, True):
        return ""
    return "H.264 High 4:4:4 Predictive: does not decode in DaVinci Resolve (free)."


def still_name(index: int, path: Path, ref: str) -> str:
    """``004-cabin-1a2b3c4d-still.png``: never ends in a digit before the extension, because
    Resolve reads ``still_4x3.png`` beside ``still_4x2.png`` as one image sequence
    (EXPORT-PROOF-RESOLVE.md finding 4); no spaces (OTIO and EDL readers mangle ``%20``)."""
    tag = hashlib.sha256(ref.encode("utf-8")).hexdigest()[:8]
    return f"{index:03d}-{safe_name(path.stem, 'still')}-{tag}-still{path.suffix.lower()}"


def copy_stills(plan: Plan, media_dir: Path, final_media: Path) -> tuple[Plan, dict[str, Path]]:
    """Copy every still into media/ under a safe name and point the plan at the copy's final path.
    Answers (plan for the interchange writers, {ref: staged copy})."""
    media_dir.mkdir(parents=True, exist_ok=True)
    copies: dict[str, Path] = {}
    named: dict[str, Path] = {}
    for clip in [*plan.storyline, *plan.overlays]:
        if clip.kind != "clip" or clip.info is None or not clip.info.is_still:
            continue
        ref = clip.source["ref"]
        if ref in copies:
            continue
        target = media_dir / still_name(len(copies) + 1, clip.info.path, ref)
        shutil.copyfile(clip.info.path, target)
        copies[ref] = target
        named[ref] = final_media / target.name

    def moved(clip: Any) -> Any:
        if clip.kind != "clip" or clip.info is None or not clip.info.is_still:
            return clip
        return replace(clip, info=replace(clip.info, path=named[clip.source["ref"]]))

    return replace(plan, storyline=[moved(c) for c in plan.storyline], overlays=[moved(c) for c in plan.overlays],
                   warnings=list(plan.warnings)), copies


def sound_tracks_manifest(plan: Plan) -> list[dict[str, Any]]:
    """A1-A3 as the package states them: role, gain, audibility and every clip's frames."""
    return [{"id": t.id, "role": t.role, "gain_db": t.gain_db, "audible": t.audible, "clips": [
        {"id": c.id, "record_in": c.record_start, "record_out": c.record_end, "source": dict(c.source),
         "source_in": c.source_in, "source_rate": {"num": c.source_rate.numerator, "den": c.source_rate.denominator},
         "gain_db": c.clip_gain_db, "fade_in_frames": c.fade_in, "fade_out_frames": c.fade_out, "file_name": c.info.path.name,
         **({"volume_keys": [{"source_frame": f, "gain_db": g} for f, g in c.volume_keys]} if c.volume_keys else {})}
        for c in t.clips]} for t in plan.sound_tracks]


def render_mix_samples(plan: Plan) -> int:
    from .render_mix import samples
    return samples(plan.total_frames, plan.rate)


def _csv(rows: list[list[Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerows(rows)
    return "\ufeff" + buffer.getvalue()


def clips_csv(plan: Plan, remuxes: dict[str, Path]) -> str:
    rows: list[list[Any]] = [["id", "track", "type", "record_in", "record_out", "record_in_tc", "record_out_tc", "frames",
                              "source_ref", "source_file", "source_in", "source_out", "source_rate", "still", "audio", "enabled", "label"]]
    for row in clip_table(plan):
        rate = row.get("source_rate")
        source = row.get("source") or {}
        ref = source.get("ref", "")
        rows.append([
            row["id"], row["track"], row["type"], row["record_in"], row["record_out"],
            timecode(plan.start_tc_frame + row["record_in"], plan.rate, plan.drop_frame), timecode(plan.start_tc_frame + row["record_out"], plan.rate, plan.drop_frame),
            row["duration_frames"], ref, remuxes[ref].name if ref in remuxes else row.get("file_name", ""),
            "" if row.get("source_in") is None else row["source_in"], "" if row.get("source_out") is None else row["source_out"],
            "" if not rate else f"{rate['num']}/{rate['den']}", row.get("still", ""), row.get("audio", ""), row["enabled"],
            next((c.label for c in [*plan.storyline, *plan.overlays] if c.id == row["id"]), ""),
        ])
    return _csv(rows)


def markers_csv(plan: Plan) -> str:
    rows: list[list[Any]] = [["id", "frame", "timecode", "seconds", "duration_frames", "name", "note"]]
    for marker in plan.markers:
        rows.append([marker.id, marker.frame, timecode(plan.start_tc_frame + marker.frame, plan.rate, plan.drop_frame),
                     f"{float(marker.frame / plan.rate):.3f}", marker.duration, marker.name, marker.note])
    return _csv(rows)


def readme(plan: Plan, name: str, kind: str, warnings: list[str]) -> str:
    rate = f"{plan.rate.numerator}/{plan.rate.denominator}" if plan.rate.denominator != 1 else str(plan.rate.numerator)
    primary = f"{name}{PRIMARY_SUFFIX[kind]}"
    if kind == "fcp7":
        steps = [f"1. Premiere Pro: File > Import... and choose {primary} (Resolve: File > Import > Timeline...)."]
    elif kind == "otio":
        steps = [f"1. In an OpenTimelineIO-capable application, import {primary} (Resolve: File > Import > Timeline...)."]
    else:
        steps = [f"1. File > Import > Timeline... and choose {primary}"]
    lines = [
        f"{plan.name} - timeline export for {APP_LABEL[kind]}",
        "",
        "Import",
        *steps,
        f"2. Keep the imported settings: {plan.width}x{plan.height}, {rate} fps, start {timecode(plan.start_tc_frame, plan.rate, plan.drop_frame)}.",
        {"fit": "3. Mismatched resolution: Scale entire image to fit (do not crop).",
         "fill": "3. Mismatched resolution: Scale full frame with crop (the cut fills the frame).",
         "stretch": "3. Mismatched resolution: Stretch frame to all corners."}[plan.fit],
        "4. If clips are offline, relink to the media folder next to this file.",
        "",
        "Contents",
        "- media/: full-length video-only copies (stream copy, no re-encode) of every moving source; all handles stay.",
        "- Stills are copied into media/ as <name>-still.<ext> (a name ending in a digit is read as an image sequence).",
        "- Clip audio is not in these copies; clips set to audio on get a separate audio clip from the original file." if kind != "edl"
        else "- The EDLs carry picture cuts only: no music, no audio.",
        f"- EDL (CMX3600, one picture track per file): {name}.edl = V1 with its markers"
        + (f", {name}-V2.edl = V2 at its record times" if plan.overlays else "") + ".",
        "  Resolve: EDL stills may relink to another clip in the media pool; use the FCP7 XML for cuts with stills.",
        f"- {name}-markers.edl: the timeline markers in Resolve's marker EDL (Resolve: right-click the imported timeline >",
        "  Timelines > Import > Timeline Markers from EDL). Resolve does not read FCPXML markers.",
        f"- {name}-clips.csv and {name}-markers.csv list every cut and marker with frames and timecode.",
        "",
        "Not checked here: whether the editing application imports and plays this file. Check it once (NLE-CHECK).",
    ]
    retimed = [c for c in plan.storyline + plan.overlays if c.kind == "clip" and c.retime is not None]
    if retimed or plan.transitions:
        lines += ["", "Speed and transitions"]
        if retimed:
            lines.append(f"- {len(retimed)} clip(s) change speed (constant speed, reverse or a freeze frame); their own sound is not in the cut.")
        if plan.transitions:
            dips = sum(1 for t in plan.transitions if t.kind != "dissolve")
            lines.append(f"- {len(plan.transitions)} transition(s) on V1 (the media copies keep the handles they need).")
            if dips and kind == "fcpxml":
                lines.append("- FCPXML has no dip: a dip to black or white is a Cross Dissolve named after the dip; change it in the NLE.")
            if dips and kind == "edl":
                lines.append("- EDL: a dip is two dissolves through black (BL); CMX has no white.")
    if warnings:
        lines += ["", "Warnings", *[f"- {w}" for w in warnings]]
    return "\r\n".join(lines) + "\r\n"


_SUMMARY_KEYS = {
    # verifier report key -> summary key, per kind; a key the verifier does not write stays None.
    "fcpxml": {"storyline_clips": "primary_spine_clips", "music_clips": "music_clips", "moving_media_checked": "moving_assets_checked"},
    "fcp7": {"storyline_clips": "storyline_clips", "music_clips": "music_clips", "moving_media_checked": "moving_files_checked"},
    "otio": {"storyline_clips": "storyline_clips", "music_clips": "music_clips", "moving_media_checked": None},
}


def verification_summary(kind: str, report: dict[str, Any] | None) -> dict[str, Any] | None:
    """One verification answer for every interchange kind: ``{ok, storyline_clips, music_clips,
    moving_media_checked}``.  The full verifier report stays in the manifest unchanged."""
    if report is None or kind not in _SUMMARY_KEYS:
        return None
    summary: dict[str, Any] = {"ok": bool(report.get("ok"))}
    for key, source in _SUMMARY_KEYS[kind].items():
        summary[key] = None if source is None else report.get(source)
    return summary



def require_sources_online(plan: Plan) -> None:
    """Lane w25-cutfix (CUT-182; Resolve refuses to render offline media): every file the plan reads is checked again
    right before the render, not only when the job was queued (a picture renamed while the job waited). An offline
    source refuses with ``source_offline`` and the check's words, before any folder is made."""
    from .model import offline_message

    def gone(info: Any) -> bool:
        path = getattr(info, "path", None)
        return path is None or not Path(path).is_file()

    picture = [c.id for c in [*plan.storyline, *plan.overlays] if c.kind == "clip" and c.info is not None and gone(c.info)]
    sound = [c.id for track in plan.sound_tracks for c in track.clips if gone(c.info)]
    music = ["music"] if plan.music is not None and gone(plan.music.info) else []
    if picture or sound or music:
        raise ExportError("source_offline", offline_message([*picture, *music], sound, music), {"items": [*picture, *sound, *music]})

def export_package(
    plan: Plan,
    export_root: Path,
    kind: str,
    *,
    ffmpeg: Path,
    ffprobe: Path,
    quality: str = "review",
    progress: ProgressFn | None = None,
    should_cancel: CancelFn | None = None,
    name: str | None = None,
    verify_hook: Callable[[Path], None] | None = None,
    music_dry: bool = False,
    loudness: dict[str, Any] | None = None,
    folder_name: str | None = None,
    subtitles: dict[str, Any] | None = None,
    burn_subtitles: bool = False,
) -> ExportResult:
    """``subtitles`` (lane w23-subs): the cut's subtitle entry (studio.subtitles/1, already cropped to the plan's
    range): every package then carries <name>.srt and <name>.vtt (frame-exact against the plan), FCPXML carries them
    as iTT captions too, and a film with ``burn_subtitles`` has them burned in (Resolve: Deliver > Subtitle Settings)."""
    if kind not in KINDS:
        raise ExportError("invalid_request", "Kind must be film, fcpxml, fcp7, edl or otio.")
    require_sources_online(plan)
    export_root = Path(export_root)
    export_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    base = safe_name(name or plan.name)
    # The artist reads this name (Deliver sheet, file manager): name, kind, time; the revision is in the manifest.
    folder_name = f"{base}-{kind}-{stamp}" if not folder_name else safe_name(folder_name, base)
    # Lane p7-cut item 4 (PRJ-015): Deliver's default package is named by the project rule (folder_name, already
    # unique in export_root); a typed name keeps <name>-<kind>-<date-time>.
    final = export_root / folder_name
    n = 2
    while final.exists():
        final = export_root / f"{folder_name}-{n}"
        n += 1
    partial = export_root / f".{final.name}.partial-{uuid.uuid4().hex[:8]}"
    partial.mkdir()

    def step(fraction: float, stage: str) -> None:
        if progress is not None:
            progress(fraction, stage)

    files: dict[str, Path] = {}
    master: dict[str, Any] | None = None
    delivery = loudness
    loudness = None
    report: dict[str, Any] | None = None
    check: Any = None
    try:
        warnings = list(plan.warnings) + vfr_warnings(plan) + curve_warnings(plan, kind)
        if kind == "film":
            film = partial / f"{base}{film_suffix(quality)}"
            rendered = render_film(plan, film, ffmpeg=ffmpeg, ffprobe=ffprobe, quality=quality, work_dir=partial / ".work",
                                   progress=lambda f, s: step(0.98 * f, s), should_cancel=should_cancel,
                                   **({"subtitles": subtitles} if (burn_subtitles and subtitles is not None) else {}),
                                   **({"audio_dir": partial / "audio", "music_dry": music_dry} if plan.sound_tracks else {}),
                                   **({"loudness": delivery} if delivery is not None else {}))
            files["film"] = film
            if rendered.loudness is not None:
                loudness = dict(rendered.loudness)
                for name, path in loudness.pop("stems").items():
                    files[f"audio:{name}"] = path
                warnings.extend(loudness.pop("warnings"))
            if film_note(quality):
                warnings.append(film_note(quality))
            master = rendered.master
        else:
            step(0.0, "remux")
            staged = remux_video_only(plan, partial / "media", ffmpeg=ffmpeg, ffprobe=ffprobe, should_cancel=should_cancel)
            # The XML and EDL name the files where they will live after the rename.
            remuxes = {ref: final / "media" / path.name for ref, path in staged.items()}
            plan, stills = copy_stills(plan, partial / "media", final / "media")
            staged = {**staged, **stills}
            step(0.6, "write")
            edl = partial / f"{base}.edl"
            edl.write_bytes(build_edl(plan, remuxes, title=plan.name).encode("utf-8"))
            files["edl"] = edl
            if plan.overlays:
                edl_v2 = partial / f"{base}-V2.edl"
                edl_v2.write_bytes(build_edl(plan, remuxes, title=f"{plan.name} V2", track="V2").encode("utf-8"))
                files["edl_v2"] = edl_v2
            if plan.markers:
                marker_edl = partial / f"{base}-markers.edl"
                marker_edl.write_bytes(build_marker_edl(plan, title=plan.name).encode("utf-8"))
                files["markers_edl"] = marker_edl
            if plan.nests:  # lane w23-bins (CUT-180): an EDL has no nesting; the nest is written as its clips there
                kept = {"otio": " The OTIO keeps each nested cut as a nested stack.", "fcp7": " The FCP7 XML keeps each nested cut as a nested sequence.",
                        "fcpxml": " The FCPXML keeps each nested cut as a compound clip."}.get(kind, "")
                warnings.append("nested_flattened: " + ", ".join(plan.nests) + (" is" if len(plan.nests) == 1 else " are")
                                + " written as " + ("its" if len(plan.nests) == 1 else "their") + " own clips in the EDLs (the film is exact)." + kept)
            if plan.music is not None and kind == "edl":
                warnings.append("The EDLs carry picture only; the music is not in them.")
            (partial / f"{base}-clips.csv").write_bytes(clips_csv(plan, remuxes).encode("utf-8"))
            (partial / f"{base}-markers.csv").write_bytes(markers_csv(plan).encode("utf-8"))
            files["clips_csv"] = partial / f"{base}-clips.csv"
            files["markers_csv"] = partial / f"{base}-markers.csv"
            if kind == "fcpxml":
                document = build_fcpxml(plan, remuxes, project_name=plan.name, event_name=plan.name,
                                        **({"captions": subtitles} if subtitles is not None else {}))
                primary = partial / f"{base}.fcpxml"
                primary.write_bytes(document.text.encode("utf-8"))

                def check(where: Path, *, decode: bool, rebase: tuple[Path, Path] | None) -> dict[str, Any]:
                    extra = {} if document.expected_sound_clips is None else {"expected_sound_clips": document.expected_sound_clips}
                    if document.expected_transitions or document.expected_retimed:
                        extra.update(expected_transitions=document.expected_transitions, expected_retimed=document.expected_retimed)
                    if plan.nest_ids:  # lane w23-bins (CUT-180)
                        extra["expected_compounds"] = document.expected_compounds
                    return verify(where, ffprobe=ffprobe, expected_spine_clips=document.expected_spine_clips,
                                  expected_music_clips=document.expected_music_clips, decode=decode, rebase=rebase, **extra)
            elif kind == "fcp7":
                document = build_fcp7(plan, remuxes, name=plan.name)
                primary = partial / f"{base}.xml"
                primary.write_bytes(document.text.encode("utf-8"))

                def check(where: Path, *, decode: bool, rebase: tuple[Path, Path] | None) -> dict[str, Any]:
                    extra = {} if document.expected_sound_clips is None else {"expected_sound_clips": document.expected_sound_clips}
                    if document.expected_transitions or document.expected_retimed:
                        extra.update(expected_transitions=document.expected_transitions, expected_retimed=document.expected_retimed)
                    if plan.nest_ids:  # lane w23-bins (CUT-180)
                        extra["expected_nested"] = document.expected_nested
                    return verify_fcp7(where, ffprobe=ffprobe, rate=plan.rate, expected_storyline_clips=document.expected_storyline_clips,
                                       expected_music_clips=document.expected_music_clips, decode=decode, rebase=rebase, **extra)
            elif kind == "otio":
                primary = partial / f"{base}.otio"
                primary.write_bytes(otio_dumps(build_otio(plan, remuxes, name=plan.name)).encode("utf-8"))

                sound_expected = sum(1 for t in plan.sound_tracks for c in t.clips if c.record_start < plan.total_frames)
                from .export_nest import storyline_runs
                nested_on_v1 = sum(1 for owner, _name, _run in storyline_runs(plan.storyline, plan.nest_ids) if owner is not None)

                pro = {"expected_transitions": len(plan.transitions),
                       "expected_retimed": sum(1 for c in [*plan.storyline, *plan.overlays] if c.kind == "clip" and c.retime is not None and not c.is_still)}

                def check(where: Path, *, decode: bool, rebase: tuple[Path, Path] | None) -> dict[str, Any]:
                    extra = {"expected_sound_clips": sound_expected} if plan.sound_tracks else {}
                    if pro["expected_transitions"] or pro["expected_retimed"]:
                        extra.update(pro)
                    if plan.nest_ids:  # lane w23-bins (CUT-180): one nested Stack per nested clip on V1
                        extra["expected_nested"] = nested_on_v1
                    return validate_otio(where, rate=plan.rate, expected_storyline_clips=len(plan.storyline_clips),
                                         expected_total_frames=plan.total_frames, rebase=rebase, **extra)
            elif plan.transitions or any(c.retime is not None for c in plan.storyline):
                # The EDL is the primary file: read it back (1.1 cuts only; a 1.0 EDL is unchanged).
                primary = edl
                from .export_edl import verify_edl

                def check(where: Path, *, decode: bool, rebase: tuple[Path, Path] | None) -> dict[str, Any]:
                    del decode, rebase
                    return verify_edl(where.read_text(encoding="utf-8"), plan)
            else:
                primary, check = None, None
            if primary is not None and kind != "edl":
                files[kind] = primary
            if primary is not None:
                if verify_hook is not None:
                    verify_hook(primary)
                step(0.7, "verify")
                report = check(primary, decode=True, rebase=(final, partial))
                if not report["ok"]:
                    raise ExportError("verify_failed", "The exported timeline failed verification; nothing was exported.", report)
                report["summary"] = verification_summary(kind, report)
        if subtitles is not None and subtitles.get("cues"):
            from . import subtitles as _subs
            rate_doc = {"num": plan.rate.numerator, "den": plan.rate.denominator}
            in_plan = [c for c in subtitles["cues"] if c["start"] < plan.total_frames]
            for suffix, writer in (("srt", _subs.srt), ("vtt", _subs.vtt)):
                path = partial / f"{base}.{suffix}"
                path.write_bytes(writer(in_plan, rate_doc, subtitles["track"]["rules"], subtitles["track"]["style"]).encode("utf-8"))
                files[f"subtitles_{suffix}"] = path
            subtitle_facts = {"cues": len(in_plan), "burned_in": bool(kind == "film" and burn_subtitles), "fcpxml_captions": kind == "fcpxml",
                              "preset": subtitles["track"]["style"]["preset"]}
        else:
            subtitle_facts = None
        if kind != "film":
            readme_path = partial / "README.txt"
            readme_path.write_bytes(readme(plan, base, kind, warnings).encode("utf-8"))
            files["readme"] = readme_path
            for ref, path in staged.items():
                files[f"media:{ref}"] = path
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "kind": kind,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "project_id": plan.project_id,
            "timeline": {"id": plan.timeline_id, "name": plan.name, "revision": plan.revision},
            "rate": {"num": plan.rate.numerator, "den": plan.rate.denominator},
            "format": {"width": plan.width, "height": plan.height, "fit": plan.fit},
            "start_tc_frame": plan.start_tc_frame,
            "start_tc": timecode(plan.start_tc_frame, plan.rate, plan.drop_frame),
            "tc_format": "DF" if plan.drop_frame else "NDF",
            "color": plan.color.manifest(),
            "total_frames": plan.total_frames,
            "format_version": {"fcpxml": FCPXML_VERSION, "fcp7": f"xmeml {XMEML_VERSION}", "otio": "Timeline.1"}.get(kind),
            "quality": quality if kind == "film" else None,
            "clips": clip_table(plan),
            "music": None if plan.music is None else {
                "track_sha256": plan.music.track_sha256, "map_id": plan.music.map_id,
                "record_start_frame": plan.music.record_start, "source_in_frame": plan.music.source_in,
                "duration_frames": plan.music.duration, "gain_db": plan.music.gain_db, "file_name": plan.music.info.path.name,
            },
            "markers": [{"id": m.id, "frame": m.frame, "name": m.name} for m in plan.markers],
            "files": {
                key: {"name": path.relative_to(partial).as_posix(), "sha256": _sha256(path), "bytes": path.stat().st_size}
                for key, path in sorted(files.items())
            },
            "verification": report,
            "ffmpeg": ffmpeg_version(ffmpeg),
            "warnings": warnings,
        }
        if subtitle_facts is not None:
            manifest["subtitles"] = subtitle_facts  # lane w23-subs
        if kind == "film":
            manifest["files"]["film"]["frames"] = plan.total_frames
            if master is not None:
                manifest["master"] = master
            if loudness is not None and not plan.sound_tracks:
                manifest["audio"] = {"loudness": loudness, "sample_rate": 48000}
        if plan.sound_tracks:
            manifest["sound_tracks"] = sound_tracks_manifest(plan)
            if loudness is not None:
                manifest["audio"] = {"loudness": loudness, "sample_rate": 48000, "bit_depth": 24,
                                     "samples": render_mix_samples(plan)}
        (partial / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        shutil.rmtree(partial / ".work", ignore_errors=True)
        partial.rename(final)
        if report is not None:
            # After the rename every path must resolve as written (no rebase); decoding was done above.
            settled = check(final / files[kind].name, decode=False, rebase=None)
            if not settled["ok"]:
                shutil.rmtree(final, ignore_errors=True)
                raise ExportError("verify_failed", "The exported timeline failed verification after it was put in place; nothing was exported.", settled)
        step(1.0, "done")
        manifest["folder"] = final.name
        return ExportResult(folder=final, manifest=manifest)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
