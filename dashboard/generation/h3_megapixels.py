"""One megapixel list per local H3 target (lane p13-vid, VID-005).

Reference: the official MiniMax H3 sizes Draft ~0.4 MP and Native 0.98 MP (1344x768 at 16:9)
[PL docs.comfy.org minimax-h3; ComfyUI comfy_extras/nodes_minimax_h3.py:32-34,54-67]. Owner VID-005: only
catalog values. The catalog (registry), the workflow manifest enum, the durable request identity, the ComfyUI
compilers and the H3 conditioning canvas all read THIS table; a value outside it is refused everywhere with the
reason code ``megapixels_not_offered``.

The stored values stay 0.5 / 1.0 (saved shots, chains and recipes carry them; the UI names them Draft / Native,
``src/shoot/parts/size-tier-label.js``). The conditioning targets compute their canvas from the size tier and
also accept the official numbers 0.4 / 0.98 as the same two tiers. ``minimax-h3-t2v`` is the byte-stable pinned
v1 graph (its manifest declares 1.0 only).

This module has no imports from the package so every loader (package import or a by-path load of a single
module) can read it: consumers load it by path next to themselves (``_load_h3_megapixels``).
"""

from __future__ import annotations

from typing import Any

CODE = "megapixels_not_offered"
DRAFT = 0.5
NATIVE = 1.0
# Official H3 numbers read as the same two tiers by the conditioning targets (canvas from the tier).
OFFICIAL_ALIASES = {0.4: DRAFT, 0.98: NATIVE}

H3_MEGAPIXELS: dict[str, dict[str, Any]] = {
    "minimax-h3-t2v": {"options": (NATIVE,), "default": NATIVE, "aliases": False},
    "minimax-h3-t2v-sampling-v2": {"options": (DRAFT, NATIVE), "default": NATIVE, "aliases": False},
    "minimax-h3-frames-v1": {"options": (DRAFT, NATIVE), "default": DRAFT, "aliases": True},
    "minimax-h3-references-v1": {"options": (DRAFT, NATIVE), "default": DRAFT, "aliases": True},
}


class MegapixelsNotOffered(ValueError):
    """A megapixel value that the target's catalog list does not offer."""

    reason_code = CODE
    code = CODE

    def __init__(self, target_id: str, value: Any) -> None:
        offered = " / ".join(f"{v:g}" for v in accepted(target_id)) if target_id in H3_MEGAPIXELS else "none"
        super().__init__(f"{CODE}: megapixels {value!r} is not offered by {target_id} (offered: {offered})")


def options(target_id: str) -> tuple[float, ...]:
    """The catalog options of ``target_id`` (the list the menu shows)."""
    return tuple(H3_MEGAPIXELS[target_id]["options"])


def default(target_id: str) -> float:
    return float(H3_MEGAPIXELS[target_id]["default"])


def accepted(target_id: str) -> tuple[float, ...]:
    """Every value a request may carry: the options, plus the official aliases where the canvas reads the tier."""
    entry = H3_MEGAPIXELS[target_id]
    return tuple(entry["options"]) + (tuple(OFFICIAL_ALIASES) if entry["aliases"] else ())


def control(target_id: str) -> dict[str, Any]:
    """The registry control for the megapixels field of ``target_id``."""
    return {"id": "megapixels", "kind": "select", "options": list(options(target_id)), "default": default(target_id)}


def check(target_id: str, value: Any) -> float:
    """Return ``value`` as float when ``target_id`` offers it, else raise :class:`MegapixelsNotOffered`."""
    if target_id not in H3_MEGAPIXELS or type(value) not in (int, float) or float(value) not in accepted(target_id):
        raise MegapixelsNotOffered(target_id, value)
    return float(value)


def any_tier(value: Any) -> str:
    """The size tier of ``value`` for the conditioning canvas ("draft" / "native"), else refuse with the code."""
    if type(value) in (int, float):
        number = OFFICIAL_ALIASES.get(float(value), float(value))
        if number == DRAFT:
            return "draft"
        if number == NATIVE:
            return "native"
    raise MegapixelsNotOffered("minimax-h3-frames-v1", value)
