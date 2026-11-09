"""The one naming rule of a project (Project charter decisions 3 + 10; PRJ-002, VID-060, GEM-068).

Every write site calls ``name_for`` at submit time, so a job knows its file name before it runs. The template has the
tokens ``{project} {page} {scene} {take} {date}``: scene = the shot's scene number or ``free``; take = a per-project,
per-page counter (no width, grows past 999); date = ``YYYYMMDD-HHMMSS`` local. A name typed by the artist is used as
is, with `` (2)``, `` (3)`` ... only on collision. The template and the counters are project state
(``<project>/naming.json``), so they travel with the folder. Pure stdlib: no route, no store import.
"""
from __future__ import annotations

import datetime as _datetime
import json
import os
import re
import secrets
import threading
from pathlib import Path
from typing import Any

DEFAULT_TEMPLATE = "{project}_{page}_{scene}_{take}_{date}"
TOKENS = ("project", "page", "scene", "take", "date")
STATE_FILE = "naming.json"
TEMPLATE_MAX = 200
_TOKEN_RE = re.compile(r"\{([^{}]*)\}")
_UNSAFE = set('<>:"/\\|?*')
_UNSAFE_RUN = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')  # a run of unsafe characters becomes ONE '-'
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
_LOCK = threading.Lock()


class NamingError(ValueError):
    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.message = message


def _clean(text: Any) -> str:
    """One path segment: unsafe and control characters become '-', no leading/trailing dots or blanks."""
    value = _UNSAFE_RUN.sub("-", str(text or ""))
    return value.strip().strip(".").strip()


def _token_value(text: Any, fallback: str) -> str:
    value = re.sub(r"\s+", "-", _clean(text))
    return value[:60] or fallback


def validate_template(template: Any) -> str:
    if not isinstance(template, str) or not template.strip() or len(template) > TEMPLATE_MAX:
        raise NamingError("name_template_invalid", "The file name template is empty or too long.")
    template = template.strip()
    if any(ch in _UNSAFE or ord(ch) < 32 for ch in _TOKEN_RE.sub("", template)) or "{" in _TOKEN_RE.sub("", template) \
            or "}" in _TOKEN_RE.sub("", template):
        raise NamingError("name_template_invalid", "The file name template has a character a file name cannot hold.")
    unknown = [token for token in _TOKEN_RE.findall(template) if token not in TOKENS]
    if unknown:
        raise NamingError("name_template_invalid", f"Unknown token {{{unknown[0]}}}.")
    if "{take}" not in template and "{date}" not in template:
        raise NamingError("name_template_not_unique", "The template needs {take} or {date} to keep names unique.")
    return template


def result_name(template: str, project: Any, page: str, scene: Any, take_counter: int, now: _datetime.datetime,
                suffix: str = "") -> str:
    """The rule applied once: no counter state, no collision check (see name_for)."""
    template = validate_template(template)
    scene_text = _token_value(scene, "free") if scene not in (None, "") else "free"
    values = {
        "project": _token_value(project, "Project"),
        "page": _token_value(page, "result"),
        "scene": scene_text,
        "take": str(int(take_counter)),
        "date": now.strftime("%Y%m%d-%H%M%S"),
    }
    stem = _clean(_TOKEN_RE.sub(lambda m: values[m.group(1)], template))[:200] or "result"
    if stem.upper() in _RESERVED:
        stem = f"{stem}-1"
    return stem + suffix


def typed_name(typed: Any, suffix: str = "") -> str:
    """A name the artist typed, made safe; its own suffix is replaced by the result's."""
    stem = _clean(typed)
    if suffix and stem.lower().endswith(suffix.lower()):
        stem = _clean(stem[: -len(suffix)])
    stem = stem[:200] or "result"
    if stem.upper() in _RESERVED:
        stem = f"{stem}-1"
    return stem + suffix


def _state_path(project_dir: str | Path) -> Path:
    return Path(project_dir) / STATE_FILE


def _read_state(project_dir: str | Path) -> dict[str, Any]:
    path = _state_path(project_dir)
    try:
        if not path.is_file() or path.stat().st_size > 256 * 1024:
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_state(project_dir: str | Path, state: dict[str, Any]) -> None:
    path = _state_path(project_dir)
    temp = path.with_name(f".{STATE_FILE}.{secrets.token_hex(8)}.tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(temp, path)


def read_template(project_dir: str | Path) -> str:
    template = _read_state(project_dir).get("template")
    try:
        return validate_template(template) if template else DEFAULT_TEMPLATE
    except NamingError:
        return DEFAULT_TEMPLATE


def write_template(project_dir: str | Path, template: str | None) -> str:
    """Set the project's template (None = the default). Refuses a template that could not stay unique."""
    clean = DEFAULT_TEMPLATE if template is None else validate_template(template)
    with _LOCK:
        state = _read_state(project_dir)
        if clean == DEFAULT_TEMPLATE:
            state.pop("template", None)
        else:
            state["template"] = clean
        _write_state(project_dir, state)
    return clean


def _existing_takes(folder: str | Path | None, template: str, project: Any, page: str) -> int:
    """The highest take already on disk under this template (a folder made before the counter was stored)."""
    if folder is None or "{take}" not in template:
        return 0
    fixed = {"project": re.escape(_token_value(project, "Project")), "page": re.escape(_token_value(page, "result"))}
    parts = []
    for piece in re.split(r"(\{[a-z]+\})", template):
        token = piece[1:-1] if piece.startswith("{") and piece.endswith("}") else None
        if token == "take":
            parts.append(r"(?P<take>\d+)") if "(?P<take>" not in "".join(parts) else parts.append(r"\d+")
        elif token in fixed:
            parts.append(fixed[token])
        elif token in ("scene", "date"):
            parts.append(r".+?")
        else:
            parts.append(re.escape(_clean(piece)) if piece else "")
    pattern = re.compile("^" + "".join(parts) + r"(?: \(\d+\))?(?:\.[A-Za-z0-9]+)?$")
    best = 0
    try:
        entries = os.scandir(folder)
    except OSError:
        return 0
    with entries:
        for entry in entries:
            match = pattern.match(entry.name)
            if match:
                best = max(best, int(match.group("take")))
    return best


def _free(folder: str | Path | None, name: str, suffix: str) -> str:
    if folder is None or not (Path(folder) / name).exists():
        return name
    stem = name[: -len(suffix)] if suffix and name.endswith(suffix) else name
    counter = 2
    while (Path(folder) / f"{stem} ({counter}){suffix}").exists():
        counter += 1
    return f"{stem} ({counter}){suffix}"


def _next(project_dir: str | Path, folder: str | Path | None, project: Any, page: str, template: str,
          state: dict[str, Any]) -> int:
    counters = state.get("takes") if isinstance(state.get("takes"), dict) else {}
    stored = counters.get(page)
    stored = stored if isinstance(stored, int) and not isinstance(stored, bool) and stored >= 0 else 0
    return max(stored, _existing_takes(folder, template, project, page)) + 1


def name_for(project_dir: str | Path, folder: str | Path | None, project: Any, page: str, *, scene: Any = None,
             typed: Any = None, now: _datetime.datetime | None = None, suffix: str = "") -> str:
    """The result's file name at submit time; the rule's counter moves (a typed name never moves it)."""
    now = now or _datetime.datetime.now()
    if isinstance(typed, str) and typed.strip():
        return _free(folder, typed_name(typed, suffix), suffix)
    with _LOCK:
        state = _read_state(project_dir)
        template = read_template(project_dir)
        take = _next(project_dir, folder, project, page, template, state)
        counters = dict(state.get("takes") or {}) if isinstance(state.get("takes"), dict) else {}
        counters[page] = take
        state["takes"] = counters
        try:
            _write_state(project_dir, state)
        except OSError:
            pass  # a read-only project still names its result; the folder scan keeps the order
        return _free(folder, result_name(template, project, page, scene, take, now, suffix), suffix)


def preview(project_dir: str | Path, folder: str | Path | None, project: Any, page: str, *, scene: Any = None,
            now: _datetime.datetime | None = None, suffix: str = "", template: str | None = None) -> str:
    """The next real name, reserving nothing (Project settings' live example)."""
    now = now or _datetime.datetime.now()
    state = _read_state(project_dir)
    template = validate_template(template) if template is not None else read_template(project_dir)
    take = _next(project_dir, folder, project, page, template, state)
    return _free(folder, result_name(template, project, page, scene, take, now, suffix), suffix)


JOB_NAMES_KEEP = 500


def record_job(project_dir: str | Path, job_id: str, name: str) -> None:
    """Remember the name a job was given at submit (Jobs shows it while running; keep writes it)."""
    with _LOCK:
        state = _read_state(project_dir)
        jobs = dict(state.get("jobs") or {}) if isinstance(state.get("jobs"), dict) else {}
        jobs.pop(job_id, None)
        jobs[job_id] = name
        while len(jobs) > JOB_NAMES_KEEP:
            jobs.pop(next(iter(jobs)))
        state["jobs"] = jobs
        _write_state(project_dir, state)


def job_names(project_dir: str | Path) -> dict[str, str]:
    jobs = _read_state(project_dir).get("jobs")
    return {k: v for k, v in jobs.items() if isinstance(k, str) and isinstance(v, str)} if isinstance(jobs, dict) else {}


def public(project_dir: str | Path, folder: str | Path | None, project: Any) -> dict[str, Any]:
    """What Project settings and continuo_project_state answer."""
    template = read_template(project_dir)
    now = _datetime.datetime.now()
    take = _next(project_dir, folder, project, "image", template, _read_state(project_dir))
    return {
        "name_template": template,
        "name_template_default": template == DEFAULT_TEMPLATE,
        "name_example": preview(project_dir, folder, project, "image", now=now, suffix=".png"),
        # The values the example uses, so Project settings can show a live example while the template is typed.
        "name_tokens": {"project": _token_value(project, "Project"), "page": "image", "scene": "free", "take": str(take),
                        "date": now.strftime("%Y%m%d-%H%M%S")},
    }
