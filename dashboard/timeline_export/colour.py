"""Colour space of a timeline, read from its sources (CUT-061, BCK-037).

Reference: Apple's FCPXML 1.9 DTD ``format colorSpace "<cp>-<tc>-<mc> (<name>)"`` (ISO/IEC 23001-8
code points; named spaces Rec. 709, Rec. 2020, Rec. 2020 PQ, Rec. 2020 HLG) and ffmpeg's stream tags
``-color_primaries / -color_trc / -colorspace``. One dominant space per timeline: Rec. 2020 (with the
sources' transfer) only when every moving source says bt2020, Rec. 709 when they say bt709 or nothing
(the export never converts colour, it only names what the pictures are).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ColourSpace:
    name: str
    code: str  # "<cp>-<tc>-<mc>"
    primaries: str  # ffmpeg names
    transfer: str
    matrix: str

    @property
    def fcpxml(self) -> str:
        return f"{self.code} ({self.name})"

    def ffmpeg_args(self) -> list[str]:
        return ["-color_primaries", self.primaries, "-color_trc", self.transfer, "-colorspace", self.matrix]

    def setparams(self) -> str:
        """The filter that tags the frames themselves (ffmpeg 7 encoders take the frame's tags; the
        output options alone leave primaries/transfer ``unknown``)."""
        return f"setparams=color_primaries={self.primaries}:color_trc={self.transfer}:colorspace={self.matrix}"

    def manifest(self) -> dict[str, Any]:
        return {"name": self.name, "fcpxml": self.fcpxml, "primaries": self.primaries, "transfer": self.transfer, "matrix": self.matrix}


REC709 = ColourSpace("Rec. 709", "1-1-1", "bt709", "bt709", "bt709")
REC2020 = ColourSpace("Rec. 2020", "9-1-9", "bt2020", "bt709", "bt2020nc")
REC2020_PQ = ColourSpace("Rec. 2020 PQ", "9-16-9", "bt2020", "smpte2084", "bt2020nc")
REC2020_HLG = ColourSpace("Rec. 2020 HLG", "9-18-9", "bt2020", "arib-std-b67", "bt2020nc")
_TRANSFER_2020 = {"smpte2084": REC2020_PQ, "arib-std-b67": REC2020_HLG}


def of_source(info: Any) -> ColourSpace:
    """The named space of one source; anything not tagged bt2020 counts as Rec. 709."""
    if info is None or getattr(info, "color_primaries", None) != "bt2020":
        return REC709
    return _TRANSFER_2020.get(getattr(info, "color_transfer", None) or "", REC2020)


def dominant(infos: Iterable[Any]) -> ColourSpace:
    """One space per timeline: the sources' common bt2020 space when ALL moving sources share it,
    else Rec. 709 (a mixed cut is delivered as Rec. 709, as a Rec. 709 project in Resolve)."""
    spaces = {of_source(info) for info in infos if info is not None and not info.is_still}
    return spaces.pop() if len(spaces) == 1 else REC709
