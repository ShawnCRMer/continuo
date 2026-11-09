from __future__ import annotations

import threading

# --- Lane p13-med item 3 (MED-023/029, Media charter owner decision 5) ----------------------------------------------
# "Use as reference…" is the only naming act: a NAME on a picture or sound (MARA, HARBOUR). The same name is the same
# reference; there is no character library and no form. Reference behaviour: keywords in Resolve / Premiere / Lightroom
# (a free word on material; typing an existing word reuses it, letter case and spaces do not make a second one).
# Stored server-side in the project (state/references.json, one document with a revision), read by Media now and by
# Image / Video / Script later through POST /projects/references/read. By hand (the Media item menu) and by Hermes
# (continuo_references_read / continuo_reference_use) through the same routes; refusals carry a reason_code.
# An item is named by its public id only (imgjob_…, studio:…, file:<sha256>, folder:<32 hex>); answers carry no path.

_REFERENCES_SCHEMA = "studio.references/1"
_REFERENCES_FILE = "references.json"
_REFERENCES_MAX = 500          # references per project
_REFERENCE_ITEMS_MAX = 200     # items under one name
_REFERENCE_NAME_MAX = 60
_REFERENCE_ITEM_RE = (r"^(imgjob_[0-9a-f]{32}|file:[0-9a-f]{64}|folder:[0-9a-f]{32}|"
                      + _take_ids_module.STUDIO_JOB_ID_BODY + ")$")
_REFERENCES_LOCK = threading.Lock()


class ReferenceItem(_ProjectRequest):
    id: str = Field(pattern=_REFERENCE_ITEM_RE, max_length=80)
    kind: Literal["image", "audio", "video"]
    label: str = Field(default="", max_length=200)


class ReferencesReadRequest(_ProjectRequest):
    project_ref: ProjectReference


class ReferenceUseRequest(_ProjectRequest):
    project_ref: ProjectReference
    name: str = Field(max_length=400)
    item: ReferenceItem


class _ReferenceRefusal(Exception):
    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code, self.message = reason_code, message


def _reference_name(raw: str) -> tuple[str, str]:
    """(display name, key): spaces collapsed; key = casefolded, so MARA, Mara and ' mara ' are one reference."""
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in raw):
        raise _ReferenceRefusal("name_invalid", "A name is one line without control characters.")
    name = " ".join(raw.split())
    if not name:
        raise _ReferenceRefusal("name_empty", "A reference needs a name.")
    if len(name) > _REFERENCE_NAME_MAX:
        raise _ReferenceRefusal("name_too_long", f"A name has at most {_REFERENCE_NAME_MAX} characters.")
    return name, name.casefold()


def _references_file(directory: str) -> Path:
    return Path(directory) / "state" / _REFERENCES_FILE


def _references_document(directory: str) -> dict[str, Any]:
    target = _references_file(directory)
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"schema": _REFERENCES_SCHEMA, "revision": 0, "references": []}
    except (OSError, ValueError):
        raise _ReferenceRefusal("references_unreadable", "The project's references could not be read.") from None
    if not isinstance(document, dict) or document.get("schema") != _REFERENCES_SCHEMA or not isinstance(document.get("references"), list):
        raise _ReferenceRefusal("references_unreadable", "The project's references could not be read.")
    return document


def _references_project(reference: ProjectReference) -> str:
    directory = _project_reference_directory(reference)
    _project_store_factory().read_project(directory, expected_project_id=reference.project_id)
    return directory


def _references_public(document: dict[str, Any]) -> dict[str, Any]:
    return {"schema": _REFERENCES_SCHEMA, "revision": int(document.get("revision") or 0),
            "references": [{"name": r["name"], "key": r["key"], "items": list(r.get("items") or [])} for r in document["references"]]}


def _reference_use(request: ReferenceUseRequest) -> dict[str, Any]:
    name, key = _reference_name(request.name)
    directory = _references_project(request.project_ref)
    with _REFERENCES_LOCK:
        document = _references_document(directory)
        found = next((r for r in document["references"] if isinstance(r, dict) and r.get("key") == key), None)
        created = found is None
        if created:
            if len(document["references"]) >= _REFERENCES_MAX:
                raise _ReferenceRefusal("too_many_references", f"A project holds at most {_REFERENCES_MAX} names.")
            found = {"name": name, "key": key, "items": []}
            document["references"].append(found)
        items = found.setdefault("items", [])
        already = any(isinstance(i, dict) and i.get("id") == request.item.id for i in items)
        if not already:
            if len(items) >= _REFERENCE_ITEMS_MAX:
                raise _ReferenceRefusal("too_many_items", f"One name holds at most {_REFERENCE_ITEMS_MAX} items.")
            items.append({"id": request.item.id, "kind": request.item.kind, "label": request.item.label.strip()[:200],
                          "by": _film_caller(), "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
            document["revision"] = int(document.get("revision") or 0) + 1
            target = _references_file(directory)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
            try:
                temporary.write_text(json.dumps(document, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        public = _references_public(document)
    return {"ok": True, "reason_code": None, "message": None, "created": created, "added": not already,
            "reference": next(r for r in public["references"] if r["key"] == key), **public}


def _references_answer(work: Any) -> dict[str, Any]:
    try:
        return work()
    except _ReferenceRefusal as exc:
        return {"ok": False, "reason_code": exc.reason_code, "message": exc.message}
    except studio_core.ProjectError as exc:
        return {"ok": False, **_project_error_response(exc)}


@project_router.post("/projects/references/read")
async def projects_references_read(request: ReferencesReadRequest) -> dict[str, Any]:
    """The project's named references: [{name, key, items [{id, kind, label, by, at}]}] with the revision."""
    def work() -> dict[str, Any]:
        return {"ok": True, "reason_code": None, "message": None, **_references_public(_references_document(_references_project(request.project_ref)))}
    return await asyncio.to_thread(_references_answer, work)


@project_router.post("/projects/references/use")
async def projects_references_use(request: ReferenceUseRequest) -> dict[str, Any]:
    """Use as reference…: put the name on the item; the same name (any letter case or spacing) is the same reference."""
    answer = await asyncio.to_thread(_references_answer, lambda: _reference_use(request))
    if answer.get("ok") and answer.get("added"):
        try:
            film_events.media_changed("film_owner:" + request.project_ref.project_id)
        except Exception:  # noqa: BLE001 - an event never fails a route
            pass
    return answer
