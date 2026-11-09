"""Camera move of a clip in words: a subject-masked camera solve and its camera sentence.

Ported from Imagination Studio's Motion Previs (``app/motion_worker.py`` ``solve_step``,
``true_roll``, ``analyze_camera``, ``describe_camera``; Apache-2.0, (c) Sam Wasserman / drowzeys,
itself a Python port of Sam Wasserman's Motion Previs Studio). The original runs OpenCV
(Shi-Tomasi corners, pyramidal Lucas-Kanade, RANSAC similarity). The Film Suite backend has numpy
but no OpenCV, so the same steps are written here in numpy:

* corners: Shi-Tomasi minimum eigenvalue of the 3x3 structure tensor, quality 0.01, a minimum
  spacing, at most 150, seeded only where the subject mask allows (or away from the frame's middle
  when there is no subject track, as the original does);
* tracking: pyramidal Lucas-Kanade, window 21, 3 levels, forward and backward, a point is kept when
  it comes back within 1 px;
* fit: RANSAC similarity (2-point samples, fixed seed: the same clip gives the same solve), inlier
  2 px at the 384 px analysis size, refit on the inliers; roll kept only when both axes turn alike.

Signs as in the original: pan is the camera turning right (content slides left), tilt up is the
content sliding down. A flat solve cannot tell a pan from a sideways track, a tilt from a crane, or
a zoom from a dolly: the words name what it looks like on screen.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np

CORNER_TARGET = 150
LK_WINDOW = 21
PYRAMID_LEVELS = 3
ANALYSIS_EDGE = 384
RANSAC_INLIER_PX = 2.0
RANSAC_ITERS = 400
SHAKE = 0.004  # of the frame per sample: above this the shot reads as handheld
# A pan or tilt under 5 % of the frame over the shot is not named (the original: 3 %). Measured: the residual
# drift of a locked-off shot on a featureless wall (DancerInRed) is 3.5 % (fs-ctrl CHOICE §5).
PAN_FLOOR = 0.05
# A zoom under 6 % (log 0.058) is not named (the original: 4 %). Measured: the same locked-off shot drifts to
# 1.034-1.041 through the live route (fs-ctrl CHOICE §5).
ZOOM_FLOOR = 0.058


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def analysis_size(width: int, height: int) -> tuple[int, int]:
    if not width or not height:
        return ANALYSIS_EDGE, round(ANALYSIS_EDGE * 0.5625)
    if width >= height:
        return ANALYSIS_EDGE, max(48, round(height / width * ANALYSIS_EDGE))
    return max(48, round(width / height * ANALYSIS_EDGE)), ANALYSIS_EDGE


# ---------------------------------------------------------------- image helpers

def _box(image: np.ndarray, radius: int) -> np.ndarray:
    """Mean over a (2r+1)^2 window, edges replicated (an integral image)."""
    pad = np.pad(image, radius + 1, mode="edge").astype(np.float64)
    total = pad.cumsum(0).cumsum(1)
    k = 2 * radius + 1
    h, w = image.shape
    out = total[k:k + h, k:k + w] - total[0:h, k:k + w] - total[k:k + h, 0:w] + total[0:h, 0:w]
    return out / (k * k)


def _gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gy, gx = np.gradient(image.astype(np.float64))
    return gx, gy


def _down(image: np.ndarray) -> np.ndarray:
    """Half size with a 2x2 mean (the pyramid step)."""
    h, w = image.shape
    h2, w2 = h // 2 * 2, w // 2 * 2
    cut = image[:h2, :w2].astype(np.float64)
    return (cut[0::2, 0::2] + cut[1::2, 0::2] + cut[0::2, 1::2] + cut[1::2, 1::2]) / 4.0


def _sample(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear samples at float coordinates (clamped to the frame)."""
    h, w = image.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx, fy = x - x0, y - y0
    a = image[y0, x0]
    b = image[y0, x0 + 1]
    c = image[y0 + 1, x0]
    d = image[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def seed_corners(gray: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Up to 150 Shi-Tomasi corners (N x 2, x/y) where ``mask`` is 255; with no mask the middle of the
    frame (where a subject usually is) is the last place corners are taken from."""
    image = gray.astype(np.float64)
    gx, gy = _gradients(image)
    sxx, syy, sxy = _box(gx * gx, 1), _box(gy * gy, 1), _box(gx * gy, 1)
    half_trace = (sxx + syy) / 2
    score = half_trace - np.sqrt(np.maximum(((sxx - syy) / 2) ** 2 + sxy ** 2, 0))
    h, w = image.shape
    border = 3
    score[:border, :] = score[-border:, :] = 0
    score[:, :border] = score[:, -border:] = 0
    spacing = max(3, round(min(w, h) / 18))

    def pick(limit: int, allowed: np.ndarray) -> list[tuple[float, float]]:
        values = np.where(allowed, score, 0)
        top = float(values.max()) if values.size else 0.0
        if top <= 0 or limit <= 0:
            return []
        ys, xs = np.nonzero(values >= top * 0.01)
        order = np.argsort(-values[ys, xs], kind="stable")
        taken = np.zeros((h // spacing + 2, w // spacing + 2), bool)
        found: list[tuple[float, float]] = []
        chosen: list[tuple[int, int]] = []
        for i in order:
            x, y = int(xs[i]), int(ys[i])
            cx, cy = x // spacing, y // spacing
            if taken[cy, cx]:
                continue
            if any((x - px) ** 2 + (y - py) ** 2 < spacing * spacing for px, py in chosen
                   if abs(px // spacing - cx) <= 1 and abs(py // spacing - cy) <= 1):
                continue
            chosen.append((x, y))
            taken[cy, cx] = True
            found.append((float(x), float(y)))
            if len(found) >= limit:
                break
        return found

    if mask is not None:
        points = pick(CORNER_TARGET, mask > 0)
    else:
        edge = np.ones((h, w), bool)
        edge[round(h * 0.275):round(h * 0.725), round(w * 0.275):round(w * 0.725)] = False
        points = pick(CORNER_TARGET, edge)
        if len(points) < CORNER_TARGET * 0.6:
            points += pick(CORNER_TARGET - len(points), ~edge)
    return np.array(points, np.float64).reshape(-1, 2)


def _lk(previous: list[np.ndarray], current: list[np.ndarray], points: np.ndarray, guess: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Pyramidal Lucas-Kanade: where ``points`` of ``previous`` moved to in ``current``."""
    half = LK_WINDOW // 2
    offs = np.arange(-half, half + 1, dtype=np.float64)
    wx, wy = np.meshgrid(offs, offs)
    wx, wy = wx.ravel(), wy.ravel()
    levels = len(previous)
    flow = np.zeros_like(points) if guess is None else (guess - points) / (2 ** (levels - 1))
    ok = np.ones(len(points), bool)
    for level in range(levels - 1, -1, -1):
        scale = 2 ** level
        a, b = previous[level], current[level]
        gx, gy = _gradients(a)
        px = points[:, 0:1] / scale + wx
        py = points[:, 1:2] / scale + wy
        ix, iy = _sample(gx, px, py), _sample(gy, px, py)
        ia = _sample(a, px, py)
        gxx, gyy, gxy = (ix * ix).sum(1), (iy * iy).sum(1), (ix * iy).sum(1)
        det = gxx * gyy - gxy * gxy
        ok &= det > 1e-6 * LK_WINDOW ** 2
        safe = np.where(np.abs(det) > 1e-12, det, 1.0)
        for _ in range(20):
            ib = _sample(b, px + flow[:, 0:1], py + flow[:, 1:2])
            diff = ia - ib
            bx, by = (diff * ix).sum(1), (diff * iy).sum(1)
            dx = (gyy * bx - gxy * by) / safe
            dy = (gxx * by - gxy * bx) / safe
            flow[:, 0] += dx
            flow[:, 1] += dy
            if float(np.max(np.abs(np.concatenate([dx, dy])), initial=0.0)) < 0.01:
                break
        if level:
            flow *= 2
    return points + flow, ok


def _pyramid(gray: np.ndarray) -> list[np.ndarray]:
    levels = [gray.astype(np.float64)]
    for _ in range(PYRAMID_LEVELS - 1):
        levels.append(_down(levels[-1]))
    return levels


def _similarity(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Least-squares similarity [a -b tx; b a ty] carrying start onto end."""
    n = len(start)
    design = np.zeros((2 * n, 4))
    design[0::2] = np.column_stack([start[:, 0], -start[:, 1], np.ones(n), np.zeros(n)])
    design[1::2] = np.column_stack([start[:, 1], start[:, 0], np.zeros(n), np.ones(n)])
    target = end.reshape(-1)
    a, b, tx, ty = np.linalg.lstsq(design, target, rcond=None)[0]
    return np.array([[a, -b, tx], [b, a, ty]])


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:, :2].T + matrix[:, 2]


def true_roll(start: np.ndarray, end: np.ndarray, fitted: float) -> float:
    """The roll in a step, kept only where the picture turns as a whole (both axes alike)."""
    if len(start) < 6:
        return fitted
    design = np.hstack([start, np.ones((len(start), 1))])
    solved = np.linalg.lstsq(design, end, rcond=None)[0]
    across, down = math.atan2(solved[0, 1], solved[0, 0]), math.atan2(-solved[1, 0], solved[1, 1])
    if across * down <= 0:
        return 0.0
    return math.copysign(min(abs(across), abs(down), abs(fitted)), across)


def solve_step(previous: np.ndarray, current: np.ndarray, mask: np.ndarray | None) -> dict[str, float]:
    """The similarity (shift, zoom, roll) that carries the background of ``previous`` onto ``current``."""
    height, width = previous.shape
    identity = {"tx": 0.0, "ty": 0.0, "scale": 1.0, "rotation": 0.0, "inlierRatio": 0.0}
    corners = seed_corners(previous, mask)
    if len(corners) == 0:
        return identity
    pa, pb = _pyramid(previous), _pyramid(current)
    moved, ok_f = _lk(pa, pb, corners)
    back, ok_b = _lk(pb, pa, moved)
    flow = moved - corners
    keep = (ok_f & ok_b & (np.abs(back - corners).max(1) < 1.0)
            & (moved[:, 0] >= 0) & (moved[:, 1] >= 0) & (moved[:, 0] < width) & (moved[:, 1] < height)
            & (np.abs(flow).max(1) <= min(width, height) * 0.5))
    start, end = corners[keep], moved[keep]
    if len(start) < 3:
        return {**identity, "inlierRatio": 0.15 if len(start) else 0.0}
    rng = np.random.default_rng(0x1E3779B9)
    best: np.ndarray | None = None
    best_count = 0
    for _ in range(RANSAC_ITERS):
        i, j = rng.choice(len(start), 2, replace=False)
        if np.hypot(*(start[i] - start[j])) < 4:
            continue
        model = _similarity(start[[i, j]], end[[i, j]])
        inliers = np.hypot(*(_apply(model, start) - end).T) < RANSAC_INLIER_PX
        count = int(inliers.sum())
        if count > best_count:
            best, best_count = inliers, count
    if best is None or best_count < 3:
        shift = np.median(end - start, axis=0)  # a slow shot that starves RANSAC still gets its translation
        return {"tx": float(shift[0]), "ty": float(shift[1]), "scale": 1.0, "rotation": 0.0, "inlierRatio": 0.2}
    for _ in range(2):  # refine on the inliers
        matrix = _similarity(start[best], end[best])
        refined = np.hypot(*(_apply(matrix, start) - end).T) < RANSAC_INLIER_PX
        if refined.sum() < 3:
            break
        best = refined
    matrix = _similarity(start[best], end[best])
    a, b = float(matrix[0, 0]), float(matrix[1, 0])
    cx, cy = (width - 1) / 2, (height - 1) / 2
    # Shift taken at the centre of the frame (a push-in on the middle has no shift).
    tx = a * cx - b * cy + float(matrix[0, 2]) - cx
    ty = b * cx + a * cy + float(matrix[1, 2]) - cy
    limit = max(width, height) * 0.6
    return {"tx": clamp(tx, -limit, limit), "ty": clamp(ty, -limit, limit), "scale": clamp(math.hypot(a, b), 0.8, 1.25),
            "rotation": clamp(true_roll(start[best], end[best], math.atan2(b, a)), -0.2, 0.2),
            "inlierRatio": int(best.sum()) / len(start)}


# ---------------------------------------------------------------- subject mask

def subject_mask(width: int, height: int, people: Sequence[Sequence[Sequence[float]]] | None, pad: float = 0.25) -> np.ndarray | None:
    """255 where a corner may be seeded: everything but a padded box round each person, open to the bottom
    edge (feet, shadow and floor reflection move with the subject).

    ``people``: per person a list of OpenPose-18 points (x, y, confidence) in 0..1 frame units. The original
    carves thin capsules round the limbs; measured on DancerInRed (a static camera) those let the corners on
    the dancer's clothes and floor reflection through and the solve drifted into a tilt (fs-ctrl CHOICE §5)."""
    if not people:
        return None
    mask = np.full((height, width), 255, np.uint8)
    drawn = False
    for marks in people:
        seen = [(p[0], p[1]) for p in marks if len(p) >= 2 and (len(p) < 3 or p[2] > 0.3) and (p[0] > 0 or p[1] > 0)]
        if not seen:
            continue
        xs, ys = [x for x, _ in seen], [y for _, y in seen]
        bw, bh = max(xs) - min(xs), max(ys) - min(ys)
        grow = pad * max(bw, bh * 0.4, 0.05)
        x0, x1 = max(0, int((min(xs) - grow) * width)), min(width, int(math.ceil((max(xs) + grow) * width)))
        y0 = max(0, int((min(ys) - pad * max(bh, 0.1)) * height))
        mask[y0:, x0:x1] = 0
        drawn = True
    if not drawn or mask.mean() < 255 * 0.08:
        return None if not drawn else mask
    return mask


# ---------------------------------------------------------------- the solve over a clip

REFERENCE_MAX_SHIFT = 16.0   # analysis px: a new reference frame is taken once the picture moved this far
REFERENCE_MAX_ZOOM = 0.05    # |log scale| since the reference
REFERENCE_MAX_FRAMES = 24
REFERENCE_MIN_INLIERS = 0.6


def analyze_camera(frames: Iterable[np.ndarray], *, sample_fps: float, width: int, height: int,
                   people_at: Any = None, progress: Any = None, total: int = 0) -> dict[str, Any]:
    """Accumulated camera move over gray analysis frames.

    Each frame is solved against a REFERENCE frame, not against the one before: a still camera then has
    no drift at all (the original chains frame-to-frame steps, and 0.1 px of bias per step summed to a
    tilt of a fifth of the frame over 8 s of a locked-off shot). A new reference is taken when the
    picture has moved 16 px, zoomed 5 %, 24 frames passed, or fewer than 60 % of the points agree.

    ``width``/``height``: the clip's own size (pixels are reported in it). ``people_at(index)`` gives the
    subject's OpenPose-18 points for analysis frame ``index`` (or None)."""
    track: list[dict[str, Any]] = []
    ref = None
    ref_index = 0
    base = (0.0, 0.0, 1.0, 0.0)  # x, y, zoom, roll at the reference frame
    last = base
    sx = sy = 1.0
    last_gray = None
    for index, gray in enumerate(frames):
        gray = np.asarray(gray, np.float64)
        confidence = 1.0
        if ref is None:
            aw, ah = gray.shape[1], gray.shape[0]
            sx, sy = width / aw, height / ah
            ref, ref_index, current = gray, index, base
        else:
            mask = subject_mask(ref.shape[1], ref.shape[0], people_at(ref_index) if people_at else None)
            step = solve_step(ref, gray, mask)
            moved = max(abs(step["tx"]), abs(step["ty"])) > REFERENCE_MAX_SHIFT or abs(math.log(step["scale"])) > REFERENCE_MAX_ZOOM
            if step["inlierRatio"] < REFERENCE_MIN_INLIERS and last_gray is not None and ref_index != index - 1:
                # The reference no longer matches (occlusion, a big change): restart from the previous frame.
                base, ref, ref_index = last, last_gray, index - 1
                mask = subject_mask(ref.shape[1], ref.shape[0], people_at(ref_index) if people_at else None)
                step = solve_step(ref, gray, mask)
            confidence = clamp(step["inlierRatio"], 0, 1)
            current = (base[0] + step["tx"] * sx, base[1] + step["ty"] * sy, base[2] * step["scale"], base[3] + step["rotation"])
            if moved or index - ref_index >= REFERENCE_MAX_FRAMES:
                base, ref, ref_index = current, gray, index
        motion = {"xPixels": current[0] - last[0], "yPixels": current[1] - last[1],
                  "scale": current[2] / last[2] if last[2] else 1.0, "rollRadians": current[3] - last[3]}
        x, y, zoom, roll = current
        track.append({"time": index / sample_fps, "frameIndex": index, "imageMotion": motion,
                      "cameraMove": {"pan": -x / max(width, 1), "tilt": -y / max(height, 1), "dollyZoom": zoom, "roll": -roll},
                      "confidence": confidence})
        last, last_gray = current, gray
        if progress:
            progress(index + 1, max(total, index + 1))
    end = track[-1]["cameraMove"] if track else None
    duration = (len(track) - 1) / sample_fps if len(track) > 1 else 0.0
    return {"fps": sample_fps, "duration": duration, "width": width, "height": height, "frames": track,
            "summary": {"panPixels": -end["pan"] * width if end else 0, "tiltPixels": -end["tilt"] * height if end else 0,
                        "zoomRatio": (end["dollyZoom"] if end else 1) or 1, "rollDegrees": (end["roll"] if end else 0) * 180 / math.pi,
                        "averageConfidence": sum(f["confidence"] for f in track) / max(len(track), 1)}}


def describe_camera(camera: dict[str, Any] | None, seconds: float | None = None) -> dict[str, Any]:
    """The solve as a sentence or two a cinematographer would say, for a video model's prompt.

    Returns {"sentence", "moves", "confidence", "reliable", "shake"} (the original's wording)."""
    frames = (camera or {}).get("frames") or []
    if len(frames) < 2:
        return {"sentence": "", "moves": [], "confidence": 0.0, "reliable": False, "shake": 0.0}
    length = float(seconds or camera.get("duration") or frames[-1]["time"] or 1.0)
    pan = [f["cameraMove"]["pan"] for f in frames]
    tilt = [-f["cameraMove"]["tilt"] for f in frames]
    zoom = [math.log(max(f["cameraMove"]["dollyZoom"], 1e-3)) for f in frames]
    roll = [math.degrees(f["cameraMove"]["roll"]) for f in frames]
    confidence = float(camera.get("summary", {}).get("averageConfidence", 0))

    def amount(track: list[float], floor: float):
        end, peak = track[-1], max(track, key=abs)
        if abs(peak) >= floor and abs(end) < 0.4 * abs(peak):
            return peak, True, None
        if abs(end) < floor:
            return 0.0, False, None
        last = max(frames[-1]["time"], 1e-6)

        def passed(part: float) -> float:
            return next((f["time"] for f, v in zip(frames, track) if abs(v) >= part * abs(end) and v * end > 0), 0.0)
        return end, False, (passed(0.2) / last, passed(0.8) / last)

    def pace(per_second: float, slow: float, fast: float) -> str:
        return "slowly " if per_second < slow else "quickly " if per_second > fast else ""

    def span(value: float) -> str:
        value = abs(value)
        return ("a little" if value < 0.08 else "across about a tenth of the frame" if value < 0.15
                else "across about a quarter of the frame" if value < 0.3 else "across about a third of the frame" if value < 0.42
                else "across about half the frame" if value < 0.65 else "across most of the frame" if value < 0.9
                else "across more than a full frame")

    moves, parts, timing = [], [], []
    for name, track, words in (("pan", pan, ("pans left", "pans right")), ("tilt", tilt, ("tilts down", "tilts up"))):
        value, returns, when = amount(track, PAN_FLOOR)
        if not value:
            continue
        verb = words[value > 0]
        if returns:
            text = f"{verb} {span(value)} and comes back"
        elif abs(value) / length > 0.7:
            text = f"whip-{verb} {span(value)}"
        else:
            text = f"{pace(abs(value) / length, 0.05, 0.22)}{verb} {span(value)}"
        moves.append({"move": name, "direction": verb.split()[1], "amount": round(value, 4), "returns": returns})
        parts.append((abs(value), text))
        timing.append(when)
    value, returns, when = amount(zoom, ZOOM_FLOOR)
    if value:
        ratio = math.exp(abs(value))
        verb = "pushes in" if value > 0 else "pulls back"
        size = " slightly" if ratio < 1.12 else "" if ratio < 1.4 else " a long way"
        text = f"{verb} and returns" if returns else f"{pace(abs(value) / length, 0.03, 0.15)}{verb}{size}"
        moves.append({"move": "zoom", "direction": "in" if value > 0 else "out", "amount": round(math.exp(value), 4), "returns": returns})
        parts.append((abs(value) * 1.5, text))
        timing.append(when)
    value, returns, when = amount(roll, 4.0)
    if value:
        text = f"rolls {'clockwise' if value > 0 else 'counter-clockwise'} about {round(abs(value))} degrees" + (" and back" if returns else "")
        moves.append({"move": "roll", "direction": "clockwise" if value > 0 else "counter-clockwise", "amount": round(value, 2), "returns": returns})
        parts.append((abs(value) / 60, text))
        timing.append(when)

    width, height = max(camera.get("width", 1), 1), max(camera.get("height", 1), 1)
    dx = [f["imageMotion"]["xPixels"] / width for f in frames[1:]]
    dy = [f["imageMotion"]["yPixels"] / height for f in frames[1:]]
    near = max(1, round(camera.get("fps", 12) * 0.25))

    def rough(track: list[float]) -> float:
        if not track:
            return 0.0
        smooth = [sum(track[max(0, i - near):i + near + 1]) / len(track[max(0, i - near):i + near + 1]) for i in range(len(track))]
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(track, smooth)) / len(track))

    shake = math.hypot(rough(dx), rough(dy))
    parts.sort(key=lambda item: -item[0])
    if not parts:
        sentence = ("Handheld shot that holds its framing: the camera stays on one spot with a slight natural shake." if shake > SHAKE
                    else "Locked-off static shot on a still camera.")
    else:
        words = [text for _, text in parts]
        sentence = "The camera " + (words[0] if len(words) == 1 else ", ".join(words[:-1]) + " while it " + words[-1]) + "."
        when = next((w for w in timing if w), None)
        if shake > SHAKE:
            sentence += " The move is handheld, with a slight natural shake."
        elif when and when[0] > 0.4:
            sentence += " The move starts gently and gathers pace toward the end of the shot."
        elif when and when[1] < 0.6:
            sentence += " The move is quickest at the start and then eases to a stop."
        else:
            sentence += " The move is smooth and continuous through the whole shot."
    return {"sentence": sentence, "moves": moves, "confidence": round(confidence, 4), "reliable": confidence >= 0.3, "shake": round(shake, 5)}


__all__ = ["analysis_size", "analyze_camera", "describe_camera", "seed_corners", "solve_step", "subject_mask", "true_roll"]
