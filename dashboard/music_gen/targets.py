"""Music targets: pinned ComfyUI graphs + manifest per local music model (lane w16-mus).

The pattern is the voice targets (dashboard/voice/targets.py): data only, the Suite writes declared
pointers and nothing else, fixed values are re-checked after every write, health fails closed.
Music adds three rules:

* AUD-071 headroom: every audio graph ends ``AudioAdjustVolume`` (-6 dB, fixed) -> ``PreviewAudio``.
  A graph whose output is not fed by the -6 dB node is refused.  ``PreviewAudio`` writes the FLAC to
  ComfyUI ``temp/`` only; the take runner copies it into the project, never ``ComfyUI/output/``.
* A target may have more than one graph (YuE2: ``notes`` then ``song``; Foundation-1: ``loop`` or
  ``keybed``); ``stage_plan`` names which graphs one take runs and in which order.
* Every refusal carries a reason code (``MusicTargetError.code``) and a readable message.

The prompt of each engine is assembled here in the authors' format (SA3 prompt assistant,
Foundation-1 tag order); the graph never sees a free-form override of a fixed sampler value.
Standalone module (stdlib only, no package-relative imports).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "studio.music-workflow/1"
_HERE = Path(__file__).resolve()
# Repository layout first (workflows/ at the root), else the installed plugin's copy (dashboard/workflows).
WORKFLOW_ROOT = _HERE.parents[4] / "workflows" if (_HERE.parents[4] / "workflows" / "sa3-small-music-v1").is_dir() else _HERE.parents[1] / "workflows"
TARGET_IDS = ("sa3-small-music-v1", "yue2-song-v1", "foundation1-loop-v1")
ENGINES = frozenset({"sa3", "yue2", "foundation1"})
HEADROOM_DB = -6
CONTROL_KINDS = frozenset({"integer", "number", "boolean", "text", "choice"})
_MANIFEST_KEYS = frozenset({"schema", "id", "version", "kind", "provider_id", "engine", "engine_label", "label",
                            "graphs", "controls", "models", "source", "policy"})
_GRAPH_KEYS = frozenset({"api_file", "api_sha256", "label", "output", "writes", "fixed", "required_classes"})
_POLICY = {"graph_mutation": "declared_pointers_only", "custom_node_install": "never", "model_download": "never_automatic",
           "output_folder": "temp_only", "submission_requires_explicit_user_intent": True}
_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_PLACEHOLDER = re.compile(r"__[A-Z]+__")
_NOTE = re.compile(r"^[A-G](#|b)?[0-8]$")
# AUD-072 / charta audio decision 12: Foundation-1.2-Keybeds sounds one octave below the note name (W-16 mus, pyin:
# C4..F4 requested sound C3..F3 at -1199 cents per note, the author's M-1 file the same, a synthetic control through
# the same code +1 cent).  The target writes every keybed note this many octaves higher, so "C4" sounds as C4 (261.6 Hz).
KEYBED_OCTAVE_OFFSET = 1
KEYS = tuple(f"{root} {mode}" for mode in ("major", "minor") for root in ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"))


class MusicTargetError(ValueError):
    """A music manifest, graph or request is invalid.  ``code`` is the reason code, str() the message."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _fail(code: str, message: str) -> None:
    raise MusicTargetError(code, message)


def _pointer_parts(pointer: str) -> list[str]:
    if type(pointer) is not str or not pointer.startswith("/") or len(pointer) > 200:
        _fail("manifest_invalid", "invalid pointer")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def read_pointer(graph: Mapping[str, Any], pointer: str) -> Any:
    node: Any = graph
    for part in _pointer_parts(pointer):
        if not isinstance(node, Mapping) or part not in node:
            _fail("manifest_invalid", f"pointer {pointer} is not in the graph")
        node = node[part]
    return node


def write_pointer(graph: dict[str, Any], pointer: str, value: Any) -> None:
    parts = _pointer_parts(pointer)
    node: Any = graph
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            _fail("manifest_invalid", f"pointer {pointer} is not in the graph")
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node or isinstance(node[parts[-1]], list):
        _fail("manifest_invalid", f"pointer {pointer} is not a writable input")
    node[parts[-1]] = value


def _check_control(control: Any, ids: set[str]) -> None:
    keys = {"id", "label", "kind", "default"}
    extra = {"integer": {"min", "max", "step"}, "number": {"min", "max", "step"}, "boolean": set(),
             "text": {"max_chars", "required"}, "choice": {"options"}}
    if type(control) is not dict or not keys <= set(control) or control.get("kind") not in CONTROL_KINDS:
        _fail("manifest_invalid", "invalid control")
    if set(control) - keys - extra[control["kind"]] or control["id"] in ids or not _ID.fullmatch(control["id"].replace("_", "-")):
        _fail("manifest_invalid", "invalid control")
    ids.add(control["id"])
    kind = control["kind"]
    if kind == "boolean" and type(control["default"]) is not bool:
        _fail("manifest_invalid", "invalid control")
    if kind in ("integer", "number"):
        for key in ("default", "min", "max", "step"):
            value = control.get(key)
            if type(value) not in ((int,) if kind == "integer" else (int, float)) or not math.isfinite(value):
                _fail("manifest_invalid", "invalid control")
        if not control["min"] <= control["default"] <= control["max"] or control["step"] <= 0:
            _fail("manifest_invalid", "invalid control")
    if kind == "text" and (type(control["default"]) is not str or type(control.get("max_chars")) is not int):
        _fail("manifest_invalid", "invalid control")
    if kind == "choice" and (type(control.get("options")) is not list or control["default"] not in control["options"]):
        _fail("manifest_invalid", "invalid control")


def validate(manifest: Mapping[str, Any], graphs: Mapping[str, Mapping[str, Any]], api_bytes: Mapping[str, bytes]) -> None:
    """Every rule the pinned set must hold; raises MusicTargetError (code manifest_invalid)."""
    if type(manifest) is not dict or set(manifest) != _MANIFEST_KEYS:
        _fail("manifest_invalid", "manifest keys are not the music schema")
    if manifest["schema"] != SCHEMA or manifest["kind"] != "music" or manifest["provider_id"] != "comfyui-local":
        _fail("manifest_invalid", "manifest is not a local music workflow")
    if not _ID.fullmatch(str(manifest["id"])) or manifest["policy"] != _POLICY or manifest["engine"] not in ENGINES:
        _fail("manifest_invalid", "manifest id, engine or policy is invalid")
    ids: set[str] = set()
    for control in manifest["controls"]:
        _check_control(control, ids)
    if type(manifest["graphs"]) is not dict or not manifest["graphs"] or set(manifest["graphs"]) != set(graphs):
        _fail("manifest_invalid", "manifest graphs do not match the files")
    for gid, entry in manifest["graphs"].items():
        graph = graphs[gid]
        if type(entry) is not dict or set(entry) != _GRAPH_KEYS:
            _fail("manifest_invalid", f"graph {gid} entry is not the music schema")
        if hashlib.sha256(api_bytes[gid]).hexdigest() != entry["api_sha256"]:
            _fail("manifest_invalid", f"pinned graph {gid} does not match its sha256")
        for slot, pointers in entry["writes"].items():
            if type(pointers) is not list or not pointers:
                _fail("manifest_invalid", f"slot {slot} has no pointer")
            for pointer in pointers:
                if isinstance(read_pointer(graph, pointer), list):
                    _fail("manifest_invalid", f"slot {slot} points at a graph link")
        for pointer, expected in entry["fixed"].items():
            value = read_pointer(graph, pointer)
            if value != expected or type(value) is not type(expected):
                _fail("manifest_invalid", f"graph {gid} does not hold its fixed value at {pointer}")
        classes = {node.get("class_type") for node in graph.values() if isinstance(node, Mapping)}
        if classes != set(entry["required_classes"]):
            _fail("manifest_invalid", f"required classes do not match graph {gid}")
        if any(node.get("class_type") == "SaveAudio" for node in graph.values()):
            _fail("manifest_invalid", "a music graph never writes ComfyUI/output (SaveAudio)")
        output = entry["output"]
        out = graph.get(output.get("node_id"), {})
        if output.get("kind") == "text":
            if out.get("class_type") != "PreviewAny":
                _fail("manifest_invalid", f"graph {gid}: a text output must be PreviewAny")
            continue
        if output.get("kind") != "audio" or out.get("class_type") != "PreviewAudio" or output.get("extension") != ".flac":
            _fail("manifest_invalid", f"graph {gid}: the output must be PreviewAudio (flac, temp)")
        source = out["inputs"].get("audio")
        gain = graph.get(source[0]) if isinstance(source, list) and source else None
        gain_ptr = f"/{source[0]}/inputs/volume" if gain else ""
        if not gain or gain.get("class_type") != "AudioAdjustVolume" or gain["inputs"].get("volume") != HEADROOM_DB \
                or entry["fixed"].get(gain_ptr) != HEADROOM_DB:
            _fail("manifest_invalid", f"graph {gid}: the output needs -6 dB headroom (AudioAdjustVolume) right before it")


def load_target(target_id: str, root: Path | None = None) -> dict[str, Any]:
    """The verified manifest and graphs of one target (fresh copies)."""
    if target_id not in TARGET_IDS:
        _fail("unknown_target", "unknown music target")
    folder = (root or WORKFLOW_ROOT) / target_id
    try:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        api_bytes = {gid: (folder / str(entry.get("api_file", ""))).read_bytes() for gid, entry in manifest["graphs"].items()}
        graphs = {gid: json.loads(data.decode("utf-8")) for gid, data in api_bytes.items()}
    except (OSError, ValueError, AttributeError, KeyError, TypeError) as exc:
        raise MusicTargetError("manifest_invalid", "music workflow is unreadable") from exc
    validate(manifest, graphs, api_bytes)
    if manifest["id"] != target_id:
        _fail("manifest_invalid", "manifest id does not match its folder")
    return {"manifest": manifest, "graphs": graphs}


def public_target(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """What the page and the agent see: no pointers, files or graphs."""
    return {
        "id": manifest["id"], "label": manifest["label"], "engine": manifest["engine"], "engine_label": manifest["engine_label"],
        "stages": [{"id": gid, "label": entry["label"], "output": entry["output"]["kind"]} for gid, entry in manifest["graphs"].items()],
        "controls": copy.deepcopy(manifest["controls"]), "headroom_db": HEADROOM_DB, "media_label": manifest["engine_label"],
    }


def resolve_values(manifest: Mapping[str, Any], values: Mapping[str, Any]) -> dict[str, Any]:
    """Every declared control with its checked value (defaults filled).  Refusals carry a code."""
    if not isinstance(values, Mapping):
        _fail("not_an_object", "values must be an object")
    controls = {c["id"]: c for c in manifest["controls"]}
    unknown = sorted(set(values) - set(controls))
    if unknown:
        _fail("unknown_setting", f"{manifest['engine_label']} has no setting {unknown[0]}")
    out: dict[str, Any] = {}
    for cid, control in controls.items():
        value = values[cid] if cid in values else control["default"]
        kind, label = control["kind"], control["label"]
        if kind == "boolean":
            if type(value) is not bool:
                _fail("not_a_boolean", f"{label} must be on or off")
        elif kind in ("integer", "number"):
            if type(value) is bool or type(value) not in (int, float) or not math.isfinite(value):
                _fail("not_a_number", f"{label} must be a number")
            if kind == "integer" and value != int(value):
                _fail("length_whole_seconds" if cid == "length" else "not_whole",
                      f"{label} must be whole seconds" if cid == "length" else f"{label} must be a whole number")
            value = int(value) if kind == "integer" else float(value)
            if not control["min"] <= value <= control["max"]:
                _fail("out_of_range", f"{label} must be {control['min']} to {control['max']}")
        elif kind == "text":
            if type(value) is not str:
                _fail("not_text", f"{label} must be text")
            value = value.strip()
            if len(value) > control["max_chars"]:
                _fail("text_too_long", f"{label} is longer than {control['max_chars']} characters")
            if control.get("required") and not value:
                _fail("text_required", f"{label} is empty")
        else:
            if value not in control["options"] or type(value) is bool:
                _fail("choice_unknown", f"{label} has no value {value!r}")
        out[cid] = value
    return out


def stage_plan(target: Mapping[str, Any], values: Mapping[str, Any]) -> list[str]:
    """The graphs one take runs, in order."""
    manifest = target["manifest"]
    picked = {c["id"]: c for c in manifest["controls"] if c["id"] in ("stages", "keybed")}
    resolved = resolve_values({**manifest, "controls": list(picked.values())}, {k: values[k] for k in picked if k in values})
    if manifest["engine"] == "yue2":
        return {"both": ["notes", "song"], "notes": ["notes"], "song": ["song"]}[resolved["stages"]]
    if manifest["engine"] == "foundation1":
        return ["keybed"] if resolved["keybed"] else ["loop"]
    return ["main"]


def _bar_seconds(bars: int, bpm: int) -> float:
    return round(bars * 4 * 60.0 / bpm, 6)


def expected_seconds(target: Mapping[str, Any], graph_id: str, values: Mapping[str, Any]) -> float | None:
    """The exact length the stored file must have, or None when the model decides (YuE2)."""
    manifest = target["manifest"]
    resolved = resolve_values(manifest, values)
    if manifest["engine"] == "sa3":
        return float(resolved["length"])
    if manifest["engine"] == "yue2" and graph_id == "song" and resolved.get("length"):
        return float(resolved["length"])
    if manifest["engine"] == "foundation1":
        if graph_id == "keybed":
            return float(target["graphs"]["keybed"]["11"]["inputs"]["duration"])
        return _bar_seconds(int(resolved["bars"]), resolved["bpm"])
    return None


def keybed_prompt_note(note: str) -> str:
    """The note name the Keybeds model is asked for, so that it sounds as ``note`` (decision 12: +12 semitones)."""
    return f"{note[:-1]}{int(note[-1]) + KEYBED_OCTAVE_OFFSET}"


def _join(*parts: str) -> str:
    return ", ".join(p.strip().strip(",").strip() for p in parts if p and p.strip().strip(","))


def _slots(manifest: Mapping[str, Any], graph_id: str, v: Mapping[str, Any]) -> dict[str, Any]:
    engine = manifest["engine"]
    if engine == "sa3":
        sentence = v["prompt"].rstrip(" .")
        if not sentence:
            _fail("text_required", "Prompt is empty")
        body = _join("TrackType: Music, VocalType: Instrumental", sentence, v["key"])
        return {"text": f"{body}. BPM: {v['bpm']}.", "seconds": float(v["length"]), "seed": v["seed"]}
    if engine == "yue2":
        if not v["prompt"] or not v["lyrics"]:
            _fail("text_required", "Style and lyrics are needed")
        slots = {"style": v["prompt"], "lyrics": v["lyrics"], "seed": v["seed"]}
        if graph_id == "song":
            if not v["notes"]:
                _fail("notes_required", "The song needs its notes")
            # Decision 13: an optional Length (whole seconds) trims the song with native TrimAudioDuration; 0 keeps the
            # model's own length (the trim stands at the generation limit, which the song never exceeds).
            length = int(v.get("length") or 0)
            limit = max(float(v["max_length"]), float(length))
            slots.update({"abc": v["notes"], "max_duration": limit, "trim": float(length) if length else limit})
        return slots
    # foundation1
    if not v["instrument"]:
        _fail("text_required", "Instrument is empty")
    if graph_id == "keybed":
        notes = v["notes"].replace(",", " ").split()
        if not notes or len(notes) > 16 or not all(_NOTE.fullmatch(n) for n in notes):
            _fail("notes_invalid", "Notes must be 1 to 16 names like C4 or F#3")
        text = _join("Keybed, Sequence, Timbre Profile", v["instrument"], v["timbre"], v["wet"], v["fx"],
                     "Chromatic Chunk, Note Sequence", ", ".join(keybed_prompt_note(n) for n in notes))
        return {"text": text, "seed": v["seed"]}
    seconds = _bar_seconds(int(v["bars"]), v["bpm"])
    text = _join(v["instrument"], v["timbre"], v["wet"], v["fx"], v["key"], f"{v['bars']} Bars", f"{v['bpm']} BPM")
    return {"text": text, "seed": v["seed"], "seconds_total": seconds, "trim": seconds, "latent_seconds": round(seconds + 0.05, 6)}


def compile_graph(target: Mapping[str, Any], graph_id: str, values: Mapping[str, Any]) -> dict[str, Any]:
    """One pinned graph with only declared pointers written; fixed values re-checked."""
    manifest = target["manifest"]
    if graph_id not in manifest["graphs"]:
        _fail("unknown_stage", f"{manifest['engine_label']} has no stage {graph_id}")
    entry = manifest["graphs"][graph_id]
    resolved = resolve_values(manifest, values)
    slots = _slots(manifest, graph_id, resolved)
    graph = copy.deepcopy(target["graphs"][graph_id])
    if set(entry["writes"]) - set(slots):
        _fail("manifest_invalid", "a declared slot has no value")
    for slot, pointers in entry["writes"].items():
        for pointer in pointers:
            write_pointer(graph, pointer, slots[slot])
    for pointer, expected in entry["fixed"].items():
        if read_pointer(graph, pointer) != expected:
            _fail("manifest_invalid", "a fixed value changed")
    if _PLACEHOLDER.search(json.dumps(graph)):
        _fail("manifest_invalid", "the graph still holds a placeholder")
    return graph


def recipe(target: Mapping[str, Any], graph_id: str, values: Mapping[str, Any]) -> dict[str, Any]:
    """What is stored with the take: the resolved values and the fixed sampler settings that ran."""
    graph = compile_graph(target, graph_id, values)
    sampler = next((n["inputs"] for n in graph.values() if n["class_type"] == "KSampler"), None)
    out: dict[str, Any] = {"target": target["manifest"]["id"], "stage": graph_id, "values": resolve_values(target["manifest"], values),
                           "headroom_db": HEADROOM_DB}
    if sampler:
        out["sampler"] = {k: sampler[k] for k in ("sampler_name", "scheduler", "steps", "cfg", "seed")}
    for node in graph.values():
        if node["class_type"] in ("CLIPTextEncode",) and node["inputs"].get("text"):
            out["prompt"] = node["inputs"]["text"]
        if node["class_type"] == "CheckpointLoaderSimple":
            out["model"] = node["inputs"]["ckpt_name"]
    return out


def health(manifest: Mapping[str, Any], object_info: Mapping[str, Any] | None, model_names: Mapping[str, Any] | None) -> dict[str, Any]:
    """Fail-closed readiness from live /object_info and /models/<folder> answers."""
    if object_info is None:
        return {"ready": False, "reasons": ["The local engine is not running"]}
    reasons: list[str] = []
    graphs = load_target(manifest["id"])["graphs"]
    classes = sorted({name for entry in manifest["graphs"].values() for name in entry["required_classes"]})
    for name in classes:
        if name not in object_info:
            reasons.append(f"Node {name} is not installed")
    for graph in graphs.values():
        for node in graph.values():
            info = object_info.get(node["class_type"])
            if not isinstance(info, Mapping):
                continue
            declared = {**(info.get("input", {}).get("required") or {}), **(info.get("input", {}).get("optional") or {})}
            for key in node["inputs"]:
                reason = f"Node {node['class_type']} no longer has the input {key}"
                if key not in declared and reason not in reasons:
                    reasons.append(reason)
    for model in manifest["models"]:
        names = (model_names or {}).get(model["folder"])
        names = None if names is None else {str(name).replace("\\", "/") for name in names}
        if names is None:
            reasons.append(f"Model folder {model['folder']} could not be read")
        elif model["name"] not in names:
            reasons.append(f"Model file {model['name']} is missing")
    return {"ready": not reasons, "reasons": reasons}


__all__ = ["HEADROOM_DB", "KEYBED_OCTAVE_OFFSET", "KEYS", "keybed_prompt_note", "MusicTargetError", "SCHEMA", "TARGET_IDS", "compile_graph", "expected_seconds", "health",
           "load_target", "public_target", "read_pointer", "recipe", "resolve_values", "stage_plan", "validate", "write_pointer"]
