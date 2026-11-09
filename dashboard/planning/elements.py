"""studio.elements/1: characters, places and props with freely named versions (lane w20-plan item 6).

Planning board round 2 panels 4/5/10: an element is ONE record with a list of versions; each version has a free name and
its own ordered pictures (the first = primary). Nothing is predefined and nothing is required. Seeds, merged on read and
recorded once in ``joined`` so a picture the artist takes off never comes back:
* the script's characters and places (``derived.characters`` / ``derived.places``) -> an element with one unnamed version;
* the project's Media names (studio.references/1, \"Use as reference…\") -> their pictures join the FIRST version.
Props are added by hand or by Hermes. Pure; the route unit owns the file, the lock and the reads.

Lane w20-plan2 contract 1 (owner steer, architecture review): every element has an immutable ``id`` (``el-<8 hex>``);
its names are lookup aliases (``key`` stays the unique handle the page and Hermes address; ``aliases`` keep old keys).
Seeding matches by (kind, name), so a character and a place with the same name stay two elements (the second gets the
key ``<name> (<kind>)``). A version or element a shot still references cannot be removed (``version_in_use`` /
``element_in_use``; the route passes the references in). Contract 4: removing a script-derived element leaves a
tombstone (kind, name) so the next read does not seed it again. ``migrate`` gives old documents ids, never resets.
"""
from __future__ import annotations

import re
import secrets
from typing import Any

SCHEMA = "studio.elements/1"
KINDS = ("character", "place", "prop")
VERSION_RE = re.compile(r"^v-[0-9a-f]{8}$")
ITEM_RE = re.compile(r"^(imgjob_[0-9a-f]{32}|file:[0-9a-f]{64}|folder:[0-9a-f]{32}|[a-z][a-z0-9_]{1,30}_[0-9a-f]{32})$")
NAME_MAX = 60
VERSION_NAME_MAX = 60
MAX_ELEMENTS = 500
MAX_VERSIONS = 40
MAX_PICTURES = 60


class ElementsRefusal(Exception):
    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code, self.message = reason_code, message


def empty() -> dict[str, Any]:
    return {"schema": SCHEMA, "revision": 0, "elements": []}


def valid_document(doc: Any) -> bool:
    return isinstance(doc, dict) and doc.get("schema") == SCHEMA and isinstance(doc.get("elements"), list)


def key_of(raw: Any) -> str:
    if not isinstance(raw, str) or any(ord(c) < 32 or ord(c) == 127 for c in raw):
        raise ElementsRefusal("key_invalid", "An element key is a one-line name.")
    name = " ".join(raw.split())
    if not name or len(name) > NAME_MAX:
        raise ElementsRefusal("key_invalid", f"An element key is a name of 1 to {NAME_MAX} characters.")
    return name.casefold()


ELEMENT_ID_RE = re.compile(r"^el-[0-9a-f]{8}$")


def _new_id() -> str:
    return "el-" + secrets.token_hex(4)


def migrate(doc: dict[str, Any]) -> bool:
    """Old documents (lane w20-plan) gain ids, aliases and tombstones in place; True when anything changed."""
    changed = False
    for element in doc["elements"]:
        if not isinstance(element.get("id"), str) or not ELEMENT_ID_RE.match(element["id"]):
            element["id"] = _new_id()
            changed = True
        if not isinstance(element.get("aliases"), list):
            element["aliases"] = [element["key"]]
            changed = True
    if not isinstance(doc.get("tombstones"), list):
        doc["tombstones"] = []  # an empty list is no change worth a revision
    return changed


def _by_ref(doc: dict[str, Any], ref: str) -> dict[str, Any] | None:
    """An element by id, key or alias (id first)."""
    if ELEMENT_ID_RE.match(ref):
        return next((e for e in doc["elements"] if e.get("id") == ref), None)
    key = key_of(ref)
    return _find(doc, key) or next((e for e in doc["elements"] if key in (e.get("aliases") or ())), None)


def _seed_match(doc: dict[str, Any], kind: str, name: str) -> dict[str, Any] | None:
    folded = name.casefold()
    return next((e for e in doc["elements"] if e["kind"] == kind and (e["name"].casefold() == folded or folded in (e.get("aliases") or ()))), None)


def _free_key(doc: dict[str, Any], name: str, kind: str) -> str:
    key = name.casefold()
    if _find(doc, key) is None:
        return key
    key = f"{name} ({kind})".casefold()
    n = 2
    while _find(doc, key) is not None:
        key, n = f"{name} ({kind} {n})".casefold(), n + 1
    return key


def _new_version(name: str = "") -> dict[str, Any]:
    return {"id": "v-" + secrets.token_hex(4), "name": name, "pictures": []}


def _find(doc: dict[str, Any], key: str) -> dict[str, Any] | None:
    return next((e for e in doc["elements"] if e.get("key") == key), None)


def _versions(raw: Any, old: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or len(raw) > MAX_VERSIONS:
        raise ElementsRefusal("versions_invalid", f"An element holds at most {MAX_VERSIONS} versions.")
    if not raw:
        raise ElementsRefusal("versions_empty", "An element keeps at least one version.")
    known = {v["id"] for v in old}
    out, seen = [], set()
    for row in raw:
        if not isinstance(row, dict) or set(row) - {"id", "name", "pictures"}:
            raise ElementsRefusal("version_invalid", "A version is {id?, name, pictures}.")
        vid = row.get("id")
        if vid is not None and (not isinstance(vid, str) or not VERSION_RE.match(vid) or vid not in known):
            raise ElementsRefusal("version_invalid", "A version id is one this element already has (v-<8 hex>); a new version has none.")
        name = row.get("name", "")
        if not isinstance(name, str) or len(name) > VERSION_NAME_MAX or any(ord(c) < 32 for c in name):
            raise ElementsRefusal("version_name_invalid", f"A version name is one line of at most {VERSION_NAME_MAX} characters.")
        pics = row.get("pictures", [])
        if not isinstance(pics, list) or len(pics) > MAX_PICTURES or not all(isinstance(p, str) and ITEM_RE.match(p) for p in pics):
            raise ElementsRefusal("picture_invalid", "Pictures are Media item ids (imgjob_…, file:<sha256>, folder:<32 hex>).")
        version = {"id": vid or "v-" + secrets.token_hex(4), "name": " ".join(name.split()), "pictures": list(dict.fromkeys(pics))}
        if version["id"] in seen:
            raise ElementsRefusal("version_invalid", "A version appears once.")
        seen.add(version["id"])
        out.append(version)
    return out


def apply(doc: dict[str, Any], changes: list[dict[str, Any]], remove: list[str], by: str, at: str,
          in_use: dict[str, set[str]] | None = None) -> list[str]:
    """Upsert elements by id or key (name, kind, versions) and remove ids/keys; all validated before anything changes.
    ``in_use`` = {element id: {version ids shots reference}} (w20-plan2 contract 1): those cannot be removed."""
    migrate(doc)
    in_use = in_use or {}
    work = [dict(e, versions=[dict(v, pictures=list(v["pictures"])) for v in e["versions"]], joined=list(e.get("joined") or []), aliases=list(e.get("aliases") or [])) for e in doc["elements"]]
    scratch = {"elements": work}
    tombstones = [dict(t) for t in doc.get("tombstones") or []]
    touched = []
    for change in changes:
        if not isinstance(change, dict) or set(change) - {"id", "key", "name", "kind", "versions"}:
            raise ElementsRefusal("change_invalid", "A change is {id? | key, name?, kind?, versions?}.")
        if change.get("id") is not None:
            if not isinstance(change["id"], str) or not ELEMENT_ID_RE.match(change["id"]):
                raise ElementsRefusal("element_unknown", "An element id is el-<8 hex>.")
            found = _by_ref(scratch, change["id"])
            if found is None:
                raise ElementsRefusal("element_unknown", "No element with that id (continuo_elements_read).")
            key = found["key"]
        else:
            key = key_of(change.get("key"))
            found = _find(scratch, key) or next((e for e in work if key in e["aliases"]), None)
        kind = change.get("kind", found["kind"] if found else None)
        if kind not in KINDS:
            raise ElementsRefusal("kind_unknown", "kind is character, place or prop.")
        name = change.get("name", found["name"] if found else " ".join(str(change.get("key")).split()))
        if not isinstance(name, str) or not name.strip() or len(name) > NAME_MAX:
            raise ElementsRefusal("name_invalid", f"A name has 1 to {NAME_MAX} characters.")
        if found is None:
            if len(work) >= MAX_ELEMENTS:
                raise ElementsRefusal("too_many_elements", f"A project holds at most {MAX_ELEMENTS} elements.")
            found = {"id": _new_id(), "key": key, "aliases": [key], "name": name.strip(), "kind": kind, "versions": [_new_version()], "joined": []}
            work.append(found)
        elif found.get("by") == "script" and kind != found["kind"]:
            raise ElementsRefusal("kind_fixed", "A character or place from the script keeps its kind.")
        new_name = " ".join(name.split())
        if new_name.casefold() != found["name"].casefold() and new_name.casefold() not in found["aliases"]:
            found["aliases"].append(new_name.casefold())
        found["name"], found["kind"] = new_name, kind
        if "versions" in change:
            versions = _versions(change["versions"], found["versions"])
            gone = (in_use.get(found["id"]) or set()) - {v["id"] for v in versions}
            if gone:
                raise ElementsRefusal("version_in_use", f"Shots still use {', '.join(sorted(gone))}; give those shots another version first.")
            found["versions"] = versions
        found["by"] = by if found.get("by") != "script" else "script"
        found["at"] = at
        touched.append(key)
    for raw in remove or []:
        if not isinstance(raw, str):
            raise ElementsRefusal("key_invalid", "remove lists element ids or keys.")
        target = _by_ref(scratch, raw)
        if target is None:
            continue
        if in_use.get(target["id"]):
            raise ElementsRefusal("element_in_use", f"Shots still use {target['name']}; take it off those shots first.")
        work.remove(target)
        if target.get("by") == "script":  # contract 4: a deleted script-derived element stays deleted
            tombstones.append({"kind": target["kind"], "name": target["name"].casefold()})
        touched.append(target["key"])
    doc["elements"] = work
    doc["tombstones"] = tombstones
    return touched


def seed(doc: dict[str, Any], derived: dict[str, Any] | None, references: list[dict[str, Any]]) -> int:
    """Join the script's characters/places and the Media names; returns the number of changes (0 = nothing new)."""
    count = 1 if migrate(doc) else 0
    dead = {(t.get("kind"), t.get("name")) for t in doc["tombstones"]}
    for kind, rows in (("character", (derived or {}).get("characters") or []), ("place", (derived or {}).get("places") or [])):
        for row in rows:
            name = " ".join(str(row.get("name") or "").split())
            if not name or len(name) > NAME_MAX or (kind, name.casefold()) in dead:
                continue
            if _seed_match(doc, kind, name) is None and len(doc["elements"]) < MAX_ELEMENTS:
                key = _free_key(doc, name, kind)
                doc["elements"].append({"id": _new_id(), "key": key, "aliases": [key], "name": name, "kind": kind, "versions": [_new_version()], "joined": [], "by": "script", "at": None})
                count += 1
    for ref in references or []:
        key = ref.get("key")
        items = [i.get("id") for i in ref.get("items") or [] if isinstance(i, dict) and i.get("kind") == "image" and isinstance(i.get("id"), str) and ITEM_RE.match(i["id"])]
        if not isinstance(key, str) or not items:
            continue
        found = _find(doc, key)
        if found is None:
            if len(doc["elements"]) >= MAX_ELEMENTS:
                continue
            found = {"id": _new_id(), "key": key, "aliases": [key], "name": ref.get("name") or key, "kind": "character", "versions": [_new_version()], "joined": [], "by": "media", "at": None}
            doc["elements"].append(found)
            count += 1
        joined = found.setdefault("joined", [])
        first = found["versions"][0]
        for item in items:
            if item in joined:
                continue
            joined.append(item)
            if item not in first["pictures"] and len(first["pictures"]) < MAX_PICTURES:
                first["pictures"].append(item)
            count += 1
    return count


def check_document(doc: Any) -> str | None:
    """None when ``doc`` is a well-formed studio.elements/1 document (w20-plan documents without ids included), else the
    problem (architecture review finding 2: the store loads nothing it has not checked to the leaf)."""
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        return "not a studio.elements/1 document"
    if set(doc) - {"schema", "revision", "elements", "tombstones"}:
        return "unknown top-level field"
    if type(doc.get("revision")) is not int or doc["revision"] < 0:
        return "revision is not an integer >= 0"
    rows, dead = doc.get("elements"), doc.get("tombstones", [])
    if not isinstance(rows, list) or len(rows) > MAX_ELEMENTS or not isinstance(dead, list) or len(dead) > 5000:
        return "elements / tombstones are not bounded lists"
    for t in dead:
        if not isinstance(t, dict) or set(t) != {"kind", "name"} or t["kind"] not in KINDS or not isinstance(t["name"], str) or len(t["name"]) > NAME_MAX:
            return "a tombstone is not {kind, name}"
    ids, keys = set(), set()
    for e in rows:
        if not isinstance(e, dict) or set(e) - {"id", "key", "aliases", "name", "kind", "versions", "joined", "by", "at"}:
            return "an element has an unknown field"
        if "id" in e and (not isinstance(e["id"], str) or not ELEMENT_ID_RE.match(e["id"]) or e["id"] in ids):
            return "an element id is invalid or repeated"
        ids.add(e.get("id"))
        if not isinstance(e.get("key"), str) or not e["key"] or len(e["key"]) > NAME_MAX + 20 or e["key"] in keys:
            return "an element key is invalid or repeated"
        keys.add(e["key"])
        if not isinstance(e.get("name"), str) or not e["name"] or len(e["name"]) > NAME_MAX or e.get("kind") not in KINDS:
            return f"element {e['key']} has an invalid name or kind"
        aliases = e.get("aliases", [])
        if not isinstance(aliases, list) or len(aliases) > 200 or not all(isinstance(a, str) and len(a) <= NAME_MAX + 20 for a in aliases):
            return f"element {e['key']} aliases are invalid"
        joined = e.get("joined", [])
        if not isinstance(joined, list) or len(joined) > 5000 or not all(isinstance(i, str) and ITEM_RE.match(i) for i in joined):
            return f"element {e['key']} joined is invalid"
        if e.get("by") not in (None, "artist", "hermes", "script", "media") or not (e.get("at") is None or isinstance(e["at"], str) and len(e["at"]) <= 40):
            return f"element {e['key']} by/at are invalid"
        vs = e.get("versions")
        if not isinstance(vs, list) or not vs or len(vs) > MAX_VERSIONS:
            return f"element {e['key']} versions are invalid"
        vids = set()
        for v in vs:
            if not isinstance(v, dict) or set(v) != {"id", "name", "pictures"} or not isinstance(v["id"], str) or not VERSION_RE.match(v["id"]) \
                    or v["id"] in vids or not isinstance(v["name"], str) or len(v["name"]) > VERSION_NAME_MAX:
                return f"element {e['key']} has an invalid version"
            vids.add(v["id"])
            if not isinstance(v["pictures"], list) or len(v["pictures"]) > MAX_PICTURES or not all(isinstance(p, str) and ITEM_RE.match(p) for p in v["pictures"]):
                return f"element {e['key']} has invalid version pictures"
    return None


def public(doc: dict[str, Any]) -> dict[str, Any]:
    return {"schema": SCHEMA, "revision": int(doc.get("revision") or 0), "kinds": list(KINDS),
            "elements": [{"id": e.get("id"), "key": e["key"], "aliases": list(e.get("aliases") or [e["key"]]), "name": e["name"], "kind": e["kind"], "by": e.get("by"), "at": e.get("at"),
                          "versions": [{"id": v["id"], "name": v["name"], "pictures": list(v["pictures"])} for v in e["versions"]]}
                         for e in doc["elements"]]}
