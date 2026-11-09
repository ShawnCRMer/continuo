"""MiniMax H3 take sidecars: the open ``mctx_v1`` header, take pairs and seams (lane fs-obvb).

Every local H3 take is a pair with one basename::

    <take>.mp4                  the delivered clip (byte-exact, never re-encoded)
    <take>.mctx.safetensors     raw AV latents + an open JSON header

The sidecar is a standard safetensors file: 8 bytes little-endian header
length, then that many bytes of JSON whose ``__metadata__`` object holds
string values (``format`` = ``mctx_v1``, ``self_id``, ``parent_id``,
``relation``, ``parent_join_frame``, frame counts, ``pins`` as JSON text).
This module reads that header itself (no tensor is loaded, no code of the
GPL node pack is used; the pack is driven over HTTP only).  ``self_id`` is
the SHA-256 of the mp4: any re-encode, trim or edit breaks the pair, and a
broken pair offers no continuation.

Seams follow from the header numbers alone (fs-obvpm RESULT §3):

* extend (one ``before`` pin): the parent plays to ``source_start +
  pinned_head_frames`` (its ``parent_join_frame``), the take from its frame 0;
* prepend (one ``after`` pin): the take plays whole, the target enters at
  ``source_start + source_frames - pinned_tail_frames`` (frame 5 for a
  39-frame window);
* bridge (a ``before`` and an ``after`` pin): exit and entry as above.

Standalone module (no package-relative imports, no dataclasses).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from pathlib import Path
from typing import Any, Mapping

FORMAT = "mctx_v1"
SIDECAR_SUFFIX = ".mctx.safetensors"
MAX_HEADER_BYTES = 16 * 1024 * 1024
BARELY_MOVES_RELATIVE = 0.5
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_INT_KEYS = ("parent_join_frame", "width", "height", "fps", "raw_frames", "pinned_head_frames",
             "pinned_tail_frames", "delivered_frames")


class MctxError(ValueError):
    """A sidecar that is missing, unreadable or does not pair with its take."""

    reason_code = "no_sidecar"


def sidecar_path(video_path: str | os.PathLike[str]) -> Path:
    """``take.mp4`` -> ``take.mctx.safetensors`` in the same folder."""
    path = Path(video_path)
    return path.with_name(path.stem + SIDECAR_SUFFIX)


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_header(raw: bytes) -> dict[str, str]:
    """``__metadata__`` of a safetensors prefix (``raw`` holds at least the header)."""
    if len(raw) < 8:
        raise MctxError("the sidecar is too short")
    (length,) = struct.unpack("<Q", raw[:8])
    if length <= 1 or length > MAX_HEADER_BYTES or len(raw) < 8 + length:
        raise MctxError("the sidecar header is not readable")
    try:
        header = json.loads(raw[8:8 + length].decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise MctxError("the sidecar header is not readable") from None
    meta = header.get("__metadata__") if isinstance(header, dict) else None
    if not isinstance(meta, dict) or meta.get("format") != FORMAT:
        raise MctxError("the file is not an mctx_v1 sidecar")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in meta.items()):
        raise MctxError("the sidecar header is not readable")
    return dict(meta)


def read_header(path: str | os.PathLike[str]) -> dict[str, str]:
    """The sidecar's string metadata, without reading any tensor."""
    try:
        with open(path, "rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise MctxError("the sidecar is too short")
            (length,) = struct.unpack("<Q", prefix)
            if length <= 1 or length > MAX_HEADER_BYTES:
                raise MctxError("the sidecar header is not readable")
            body = handle.read(length)
    except OSError:
        raise MctxError("the sidecar cannot be read") from None
    return parse_header(prefix + body)


def _int(meta: Mapping[str, str], key: str) -> int:
    try:
        return int(meta.get(key) or 0)
    except (TypeError, ValueError):
        raise MctxError(f"the sidecar header has no whole number {key}") from None


def _pins(meta: Mapping[str, str]) -> list[dict[str, Any]]:
    try:
        raw = json.loads(meta.get("pins") or "[]")
    except ValueError:
        raise MctxError("the sidecar pins are not readable") from None
    if not isinstance(raw, list):
        raise MctxError("the sidecar pins are not readable")
    pins = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise MctxError("the sidecar pins are not readable")
        source_id = entry.get("source_id")
        place = entry.get("place")
        if not isinstance(source_id, str) or place not in ("before", "after", "at_frame"):
            raise MctxError("the sidecar pins are not readable")
        try:
            start, frames = int(entry.get("source_start", 0)), int(entry.get("source_frames", 0))
        except (TypeError, ValueError):
            raise MctxError("the sidecar pins are not readable") from None
        pins.append({"source_id": source_id, "source_kind": str(entry.get("source_kind") or "clip"),
                     "source_start": start, "source_frames": frames, "place": place,
                     "mode": str(entry.get("mode") or "guide")})
    return pins


def relation_of(meta: Mapping[str, str], pins: list[Mapping[str, Any]]) -> str:
    """``root`` | ``extends`` | ``prepends`` | ``bridges`` | ``other``."""
    if not pins:
        return "root"
    if meta.get("relation") in ("extends", "prepends"):
        return str(meta["relation"])
    places = sorted(pin["place"] for pin in pins)
    if places == ["after", "before"] and all(pin["source_kind"] == "clip" for pin in pins):
        return "bridges"
    return "other"


def take_facts(meta: Mapping[str, str]) -> dict[str, Any]:
    """Path-free facts of one take: identity, lineage, frame counts and seams."""
    self_id = meta.get("self_id", "")
    if not _HEX64.match(self_id):
        raise MctxError("the sidecar has no take identity")
    numbers = {key: _int(meta, key) for key in _INT_KEYS}
    pins = _pins(meta)
    relation = relation_of(meta, pins)
    parents: list[str] = []
    for pin in pins:
        if pin["source_id"] and pin["source_id"] not in parents:
            parents.append(pin["source_id"])
    facts: dict[str, Any] = {
        "self_id": self_id,
        "parent_id": meta.get("parent_id") or (parents[0] if parents else ""),
        "parents": parents,
        "relation": relation,
        "pins": pins,
        **numbers,
        "seams": seams(pins, numbers["pinned_head_frames"], numbers["pinned_tail_frames"]),
    }
    return facts


def seams(pins: list[Mapping[str, Any]], pinned_head: int, pinned_tail: int) -> dict[str, Any]:
    """Where each source hands over: ``exit`` (the source plays to it) / ``entry`` (it resumes there)."""
    result: dict[str, Any] = {}
    before = [pin for pin in pins if pin["place"] == "before"]
    after = [pin for pin in pins if pin["place"] == "after"]
    if len(before) == 1:
        result["exit"] = {"source_id": before[0]["source_id"], "frame": before[0]["source_start"] + pinned_head}
    if len(after) == 1:
        pin = after[0]
        result["entry"] = {"source_id": pin["source_id"],
                           "frame": pin["source_start"] + pin["source_frames"] - pinned_tail}
    return result


def verify_pair(video_path: str | os.PathLike[str], sidecar: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Facts of a take whose sidecar pairs with its mp4 (SHA-256 == ``self_id``)."""
    side = Path(sidecar) if sidecar is not None else sidecar_path(video_path)
    if not side.is_file():
        raise MctxError("This take was made before continuation existed")
    facts = take_facts(read_header(side))
    try:
        actual = sha256_file(video_path)
    except OSError:
        raise MctxError("the take cannot be read") from None
    if actual != facts["self_id"]:
        error = MctxError("This take was changed after it was made; it cannot be continued")
        error.reason_code = "pair_broken"
        raise error
    return facts


def public_mctx(facts: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The take-facts projection the UI and the agent read (``mctx`` key)."""
    if not facts:
        return None
    keys = ("self_id", "parent_id", "parents", "relation", "pins", "parent_join_frame", "delivered_frames",
            "pinned_head_frames", "pinned_tail_frames", "raw_frames", "width", "height", "seams")
    return {key: facts[key] for key in keys if key in facts}


def relative_motion(take: float | None, source: float | None) -> dict[str, Any] | None:
    """``motion_activity`` of a continuation relative to its source; below 0.5 it barely moves."""
    if type(take) not in (int, float) or type(source) not in (int, float) or source <= 0:
        return None
    ratio = round(float(take) / float(source), 3)
    return {"motion_relative": ratio, "barely_moves": ratio < BARELY_MOVES_RELATIVE}


__all__ = [
    "BARELY_MOVES_RELATIVE",
    "FORMAT",
    "MctxError",
    "SIDECAR_SUFFIX",
    "parse_header",
    "public_mctx",
    "read_header",
    "relation_of",
    "relative_motion",
    "seams",
    "sha256_file",
    "sidecar_path",
    "take_facts",
    "verify_pair",
]
