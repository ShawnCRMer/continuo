"""One subtitle rasteriser for the Cut viewer and the burned-in film (lane w23-subs, CUT-011).

A state (one cue, and in the Social style the active word) is drawn at the cut's format as a transparent RGBA PNG the
size of the frame. The viewer lays exactly these PNGs over the program monitor (routes/projects/subtitles_1.py,
/projects/subtitles/frame), and the film's burn-in overlays the same PNGs frame by frame (burn_inputs), so the
viewer shows what is exported by construction (Resolve: the subtitle track is rendered into the picture on
"Burn into video"; Premiere: "Burn captions into video").

Fonts: the system's own (Windows Arial / Georgia / Consolas, macOS Helvetica / Georgia / Menlo, Linux DejaVu), else
Pillow's bundled font. The font file is part of the state key, so a cached PNG is never drawn with another font.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from . import subtitles as S

RASTER_VERSION = "w23-subs/1"

_FONT_CANDIDATES = {
    "sans": ["arial.ttf", "Arial.ttf", "Helvetica.ttc", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"],
    "sans_bold": ["arialbd.ttf", "Arial Bold.ttf", "HelveticaNeue.ttc", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"],
    "serif": ["georgia.ttf", "Georgia.ttf", "DejaVuSerif.ttf", "LiberationSerif-Regular.ttf"],
    "mono": ["consola.ttf", "Menlo.ttc", "DejaVuSansMono.ttf", "LiberationMono-Regular.ttf"],
}
_FONT_DIRS = [Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts", Path("/System/Library/Fonts"), Path("/System/Library/Fonts/Supplemental"),
              Path("/Library/Fonts"), Path("/usr/share/fonts/truetype/dejavu"), Path("/usr/share/fonts/truetype/liberation"), Path("/usr/share/fonts")]
_FONT_CACHE: dict[str, str | None] = {}


def font_file(name: str) -> str | None:
    """The file of a style font on this machine (None: Pillow's bundled font)."""
    if name not in _FONT_CACHE:
        found = None
        for candidate in _FONT_CANDIDATES.get(name, []):
            for directory in _FONT_DIRS:
                path = directory / candidate
                if path.is_file():
                    found = str(path)
                    break
            if found:
                break
        _FONT_CACHE[name] = found
    return _FONT_CACHE[name]


def _font(name: str, px: int) -> ImageFont.FreeTypeFont:
    path = font_file(name)
    if path is not None:
        return ImageFont.truetype(path, px)
    return ImageFont.load_default(px)


def _rgba(hex_color: str, alpha: float = 1.0) -> tuple[int, int, int, int]:
    return (int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16), int(round(255 * alpha)))


def state_key(track: dict[str, Any], cue: dict[str, Any], active: int | None, width: int, height: int) -> str:
    """Content key of one drawn state (style, rules, text, active word, size, font file, rasteriser version)."""
    payload = [RASTER_VERSION, width, height, track["style"], track["rules"], cue["text"], active, font_file(track["style"]["font"])]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:32]


def render_state(track: dict[str, Any], cue: dict[str, Any], active: int | None, width: int, height: int) -> Image.Image:
    """The frame-sized RGBA picture of one cue (and its active word)."""
    style, rules = track["style"], track["rules"]
    lines = S.layout_lines(cue["text"], rules)
    if style["uppercase"]:
        lines = [line.upper() for line in lines]
    px = max(8, int(round(height * style["size"] / 100.0)))
    font = _font(style["font"], px)
    stroke = int(round(px * style["outline"] / 100.0))
    space = font.getlength(" ")
    line_h = int(round(px * 1.22))
    pad_x, pad_y = int(round(px * 0.35)), int(round(px * 0.18))
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    widths = [[font.getlength(w) for w in line.split()] for line in lines]
    line_w = [sum(ws) + space * max(0, len(ws) - 1) for ws in widths]
    block_h = line_h * len(lines)
    cx = width * style["x"] / 100.0
    bottom = height * style["y"] / 100.0
    top = bottom - block_h
    top = max(0.0, min(top, height - block_h))
    color, highlight = _rgba(style["color"]), _rgba(style["highlight"])
    outline = _rgba(style["outline_color"])
    if style["background_opacity"] > 0:
        box = _rgba(style["background"], style["background_opacity"])
        for k, lw in enumerate(line_w):
            x0 = cx - lw / 2 - pad_x
            y0 = top + k * line_h - pad_y + line_h * 0.08
            draw.rectangle([round(x0), round(y0), round(x0 + lw + 2 * pad_x), round(y0 + line_h + pad_y * 2 - line_h * 0.16)], fill=box)
    ascent = font.getmetrics()[0]
    index = 0
    for k, line in enumerate(lines):
        x = cx - line_w[k] / 2
        y = top + k * line_h + (line_h - px) / 2 + (px - ascent) * 0.0
        for w, ww in zip(line.split(), widths[k]):
            fill = highlight if (style["active_word"] and active is not None and index == active) else color
            draw.text((round(x), round(y)), w, font=font, fill=fill, stroke_width=stroke, stroke_fill=outline if stroke else None)
            x += ww + space
            index += 1
    return image


def png(track: dict[str, Any], cue: dict[str, Any], active: int | None, width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    render_state(track, cue, active, width, height).save(buffer, format="PNG", optimize=False, compress_level=3)
    return buffer.getvalue()


def write_states(entry: dict[str, Any], width: int, height: int, folder: Path) -> list[dict[str, Any]]:
    """Every state of the cut's subtitles as a PNG in ``folder`` (one file per distinct state): [{start, end, file}]."""
    folder.mkdir(parents=True, exist_ok=True)
    cues = {c["id"]: c for c in entry["cues"]}
    out = []
    for st in S.states(entry):
        cue = cues[st["cue"]]
        key = state_key(entry["track"], cue, st["active"], width, height)
        path = folder / f"{key}.png"
        if not path.is_file():
            render_state(entry["track"], cue, st["active"], width, height).save(path, format="PNG", compress_level=3)
        out.append({"start": st["start"], "end": st["end"], "file": path})
    return out


def burn_inputs(entry: dict[str, Any], width: int, height: int, total_frames: int, rate_text: str, folder: Path) -> list[str] | None:
    """ffmpeg input arguments for one RGBA picture stream of the whole film: a concat list with ONE entry per film
    frame (the state's PNG, or a transparent PNG where no subtitle shows) and no durations, so frame n of the stream
    is entry n; burn_filter numbers the frames at the cut's rate. Durations would be stored in microseconds by the
    concat demuxer and could put a state one frame early or late. None when the cut has no subtitle state."""
    states = [s for s in write_states(entry, width, height, folder) if s["start"] < total_frames]
    if not states:
        return None
    blank = folder / "blank.png"
    if not blank.is_file():
        Image.new("RGBA", (width, height), (0, 0, 0, 0)).save(blank, format="PNG")
    shown: list[Path] = [blank] * total_frames
    for st in states:
        for f in range(max(0, st["start"]), min(st["end"], total_frames)):
            shown[f] = st["file"]
    lines = ["ffconcat version 1.0", *(f"file '{path.name}'" for path in shown)]
    listing = folder / "burn.ffconcat"
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ["-f", "concat", "-safe", "0", "-i", str(listing)]


# ffmpeg colorspace names (colour.py ``matrix``) -> swscale out_color_matrix names.
_SWS_MATRIX = {"bt709": "bt709", "bt2020nc": "bt2020", "bt2020c": "bt2020", "smpte170m": "smpte170m", "bt470bg": "bt601"}


def to_film_yuv(matrix: str = "bt709") -> str:
    """The one RGBA -> 4:2:0 conversion of a subtitle picture: the film's own matrix, limited range, exact rounding.
    Explicit, so it never depends on what the other overlay input happens to be tagged with."""
    return (f"scale=out_color_matrix={_SWS_MATRIX.get(matrix, 'bt709')}:out_range=tv:flags=bilinear+accurate_rnd+full_chroma_int,"
            "format=yuva420p")


def burn_filter(index: int, rate_text: str, out_label: str = "subs", matrix: str | None = None) -> str:
    """Input ``index`` (burn_inputs' stream) as one picture per film frame: frame n at n / rate, in a time base of
    exactly one frame (a coarser base gives two frames one timestamp and the muxer drops one of them). With
    ``matrix`` the pictures are converted to the film's 4:2:0 (to_film_yuv), else they stay RGBA."""
    num, den = (int(x) for x in rate_text.split("/")) if "/" in rate_text else (int(rate_text), 1)
    convert = "format=rgba" if matrix is None else f"format=rgba,{to_film_yuv(matrix)}"
    return f"[{index}:v:0]{convert},settb={den}/{num},setpts=N,fps={num}/{den}[{out_label}]"


def frame_table(entry: dict[str, Any], total_frames: int) -> list[tuple[int, int, int, int | None]]:
    """(start, end, cue index, active) per state, for tests: which state shows on which frame."""
    ids = {c["id"]: i for i, c in enumerate(entry["cues"])}
    return [(s["start"], min(s["end"], total_frames), ids[s["cue"]], s["active"]) for s in S.states(entry) if s["start"] < total_frames]
