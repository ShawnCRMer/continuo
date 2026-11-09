"""Sync evaluation layer (lane w6-scene item 1): offset, drift, frozen mouth and foreign audio from
per-frame SyncNet v2 scores plus MediaPipe mouth landmarks. No model here (numpy only); the engine
runner produces the inputs:

* ``scores``    (n_frames, 2*25+1) SyncNet score (negative embedding distance) of video frame i vs
                audio frame i+d, d = -25..25 frames (+d = sound late).
* ``scores_q``  (n_frames, 2*100+1) the same at quarter-frame audio lags (one MFCC hop = 1/4 frame),
                used only for drift.
* ``gray``      (n_frames, H, W) luma frames and ``landmarks`` (n_frames, 6, 2) = FaceMesh points
                13, 14 (inner lips), 61, 291 (mouth corners), 33, 263 (eye corners), same pixel scale.
* ``audio``     mono float samples, for the speech / music decision.

Every constant below was set on gt g1 (eval/bench RESULTS.md section A, w6-scene measurements in
eval/scene/); change one only with a new measurement.  Numbers per constant:

FOREIGN_Z_MIN 2.4      global curve peak z (peak minus median over std of the lag curve), round 2. Genuine
                       speech >= 2.596 (h3 frozen), 2.873 (h3 shift-4), 3.536 (tp drift); foreign 2.20 (h3)
                       / 1.69 (tp). Margin 0.2 on h3: the narrowest number in this layer.
MUSIC_LOW_ENERGY_MAX   share of 20 ms frames below half the 1 s mean RMS. Speech 0.26-0.29, sung band
  0.10                 mix 0.011.  Music -> "Offset only": the bench showed confidence is unusable on
                       songs (genuine z 1.5-2.2 = foreign level), so drift and foreign are never claimed.
DRIFT_WINDOW_S 2.0,    windows of 2 s every 0.25 s on the quarter-lag matrix (round 2); a window counts when
DRIFT_Z_MIN 1.5        its peak z >= 1.5 and lies within 3 frames of the global lag; Theil-Sen slope x length.
DRIFT_COVERAGE_MIN     drift is only reported when >= 25 % of the windows count (tp 0.83, h3 0.29: h3 talks
  0.25                 in 40 % of the take). Generated takes carry their own drift (h3 steady copy: -2.98 f);
                       the drift is reported as measured, the g1 score is drift(copy) - drift(steady).
DRIFT_MIN_FRAMES 1.0   |drift| below one frame over the take is no drift.
FROZEN_*               still-mouth score over 0.5 s blocks: log((median temporal std inside the mouth
                       box + 0.3) / (same over the face box + 0.3)); a span seeds below -3.0 and grows
                       while < -1.0.  Normal takes min -1.84 (h3), -1.05 (tp), -2.50 (sung).
"""
from __future__ import annotations

from typing import Any

import numpy as np

SCHEMA = "studio.sync-eval/1"
LAGS = 25
LAGS_Q = 100
FOREIGN_Z_MIN = 2.4
MUSIC_LOW_ENERGY_MAX = 0.10
DRIFT_WINDOW_S = 2.0
DRIFT_HOP_S = 0.25
DRIFT_Z_MIN = 1.5
DRIFT_BAND_FRAMES = 3.0
DRIFT_COVERAGE_MIN = 0.25
DRIFT_MIN_FRAMES = 1.0
FROZEN_BLOCK_S = 0.5
FROZEN_SEED = -3.0
FROZEN_GROW = -1.0
FROZEN_FLOOR = 0.3

PARAMETERS = {k: v for k, v in globals().items() if k.isupper() and k not in {"SCHEMA"}}


def _residual(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    return s - np.nanmedian(s, axis=1, keepdims=True)


def _peak(curve: np.ndarray) -> tuple[int, float, float]:
    """Index of the peak, sub-sample peak and z of the peak."""
    j = int(np.nanargmax(curve))
    sub = float(j)
    if 0 < j < len(curve) - 1 and np.isfinite(curve[j - 1]) and np.isfinite(curve[j + 1]):
        den = curve[j - 1] - 2 * curve[j] + curve[j + 1]
        if den < 0:
            sub += 0.5 * (curve[j - 1] - curve[j + 1]) / den
    z = float((curve[j] - np.nanmedian(curve)) / (np.nanstd(curve) + 1e-9))
    return j, sub, z


def global_offset(scores: np.ndarray) -> dict[str, Any]:
    """Offset in whole frames (+ = sound late) and its peak z."""
    curve = np.nanmean(_residual(scores), axis=0)
    j, sub, z = _peak(curve)
    return {"frames": j - LAGS, "frames_sub": round(sub - LAGS, 3), "z": round(z, 3)}


def music_like(audio: np.ndarray, sr: int) -> tuple[bool, float]:
    """Low-energy ratio (Scheirer-Slaney): speech has pauses, a band mix has none."""
    a = np.asarray(audio, dtype=np.float64)
    hop = max(1, sr // 50)
    if len(a) < hop * 60:
        return False, float("nan")
    e = np.sqrt(np.add.reduceat(a[: len(a) // hop * hop] ** 2, np.arange(0, len(a) // hop * hop, hop)) / hop + 1e-12)
    ratios = [float(np.mean(e[i:i + 50] < 0.5 * e[i:i + 50].mean())) for i in range(0, len(e) - 50, 50)]
    ler = float(np.mean(ratios)) if ratios else float("nan")
    return bool(ler < MUSIC_LOW_ENERGY_MAX), round(ler, 4)


def drift(scores_q: np.ndarray, fps: float) -> dict[str, Any]:
    """Linear drift over the take in frames (end minus start), or measurable False."""
    from scipy.stats import theilslopes

    r = _residual(scores_q)
    n = len(r)
    g = int(np.nanargmax(np.nanmean(r, axis=0)))
    w = max(3, int(round(DRIFT_WINDOW_S * fps)))
    h = max(1, int(round(DRIFT_HOP_S * fps)))
    pts, total = [], 0
    for lo in range(0, n - w + 1, h):
        total += 1
        j, sub, z = _peak(np.nanmean(r[lo:lo + w], axis=0))
        if z >= DRIFT_Z_MIN and abs(j - g) <= DRIFT_BAND_FRAMES * 4:
            pts.append(((lo + w / 2) / fps, (sub - LAGS_Q) / 4.0))
    coverage = len(pts) / total if total else 0.0
    out: dict[str, Any] = {"windows": total, "counted": len(pts), "coverage": round(coverage, 3)}
    if coverage < DRIFT_COVERAGE_MIN or len(pts) < 3:
        return {**out, "measurable": False, "frames": None, "slope_frames_per_s": None}
    p = np.asarray(pts)
    slope = float(theilslopes(p[:, 1], p[:, 0])[0])
    frames = slope * n / fps
    return {**out, "measurable": True, "frames": round(frames, 2), "slope_frames_per_s": round(slope, 4),
            "found": bool(abs(frames) >= DRIFT_MIN_FRAMES)}


def still_mouth_series(gray: np.ndarray, landmarks: np.ndarray, fps: float) -> np.ndarray:
    """Per block start: log ratio of mouth-box vs face-box temporal std over FROZEN_BLOCK_S."""
    g = np.asarray(gray, dtype=np.float32)
    lm = np.asarray(landmarks, dtype=np.float64)
    n, hgt, wid = g.shape
    k = max(3, int(round(FROZEN_BLOCK_S * fps)))
    out = np.full(n, np.nan)
    for i in range(0, n - k + 1):
        pt = np.nanmedian(lm[i:i + k], axis=0)
        if not np.all(np.isfinite(pt)):
            continue
        cx, cy = (pt[2] + pt[3]) / 2
        mw = max(4.0, float(np.linalg.norm(pt[2] - pt[3])))
        ew = max(4.0, float(np.linalg.norm(pt[4] - pt[5])))
        x0, x1 = int(max(0, cx - 0.45 * mw)), int(min(wid, cx + 0.45 * mw))
        y0, y1 = int(max(0, cy - 0.35 * mw)), int(min(hgt, cy + 0.35 * mw))
        X0, X1 = int(max(0, cx - 1.3 * ew)), int(min(wid, cx + 1.3 * ew))
        Y0, Y1 = int(max(0, cy - 2.3 * ew)), int(min(hgt, cy + 0.9 * ew))
        if x1 - x0 < 2 or y1 - y0 < 2 or X1 - X0 < 2 or Y1 - Y0 < 2:
            continue
        blk = g[i:i + k]
        sm = float(np.median(blk[:, y0:y1, x0:x1].std(0)))
        sf = float(np.median(blk[:, Y0:Y1, X0:X1].std(0)))
        out[i] = np.log((sm + FROZEN_FLOOR) / (sf + FROZEN_FLOOR))
    return out


def frozen_windows(series: np.ndarray, fps: float) -> list[dict[str, Any]]:
    """Spans where the mouth holds still while the face moves: [{start_frame, end_frame, start_s, end_s}]."""
    x = np.asarray(series, dtype=np.float64)
    n = len(x)
    k = max(3, int(round(FROZEN_BLOCK_S * fps)))
    spans: list[list[int]] = []
    i = 0
    while i < n:
        if np.isfinite(x[i]) and x[i] < FROZEN_SEED:
            a = i
            while a > 0 and np.isfinite(x[a - 1]) and x[a - 1] < FROZEN_GROW:
                a -= 1
            b = i
            while b + 1 < n and np.isfinite(x[b + 1]) and x[b + 1] < FROZEN_GROW:
                b += 1
            end = min(n, b + k)
            if spans and a <= spans[-1][1]:
                spans[-1][1] = max(spans[-1][1], end)
            else:
                spans.append([a, end])
            i = b + 1
        else:
            i += 1
    return [{"start_frame": a, "end_frame": b, "start_s": round(a / fps, 3), "end_s": round(b / fps, 3)} for a, b in spans]


FROZEN_FIXED_STILL = 1.0  # round 3: grey levels; a frame-to-frame median change below this in the seed's mouth box = still


def refine_fixed(spans: list[dict[str, Any]], gray: np.ndarray, landmarks: np.ndarray, fps: float) -> list[dict[str, Any]]:
    """Grow/trim each span on the mouth box of its first frame held fixed in the picture: a held mouth patch
    stays put while the head (and the landmarks) move on, so the moving box ends the span too early (h3: 6.125 s
    instead of 6.5 s). Measured on g1: the fixed box changes by <= 0.5 grey inside the held second, >= 40 outside."""
    g = np.asarray(gray, dtype=np.float32)
    n, hgt, wid = g.shape
    out = []
    for sp in spans:
        a = min(n - 2, sp["start_frame"] + 1)
        pt = np.asarray(landmarks[a], dtype=np.float64)
        cx, cy = (pt[2] + pt[3]) / 2
        mw = max(4.0, float(np.linalg.norm(pt[2] - pt[3])))
        x0, x1 = int(max(0, cx - 0.4 * mw)), int(min(wid, cx + 0.4 * mw))
        y0, y1 = int(max(0, cy - 0.3 * mw)), int(min(hgt, cy + 0.3 * mw))
        if x1 - x0 < 2 or y1 - y0 < 2:
            out.append(sp)
            continue
        still = lambda i: float(np.median(np.abs(g[i, y0:y1, x0:x1] - g[i - 1, y0:y1, x0:x1]))) < FROZEN_FIXED_STILL
        s0 = a
        while s0 - 1 >= 1 and still(s0 - 1 + 1) and still(s0 - 1):
            s0 -= 1
        e0 = a
        while e0 + 1 < n and still(e0 + 1):
            e0 += 1
        start, end = s0 - 1 if s0 > 0 else 0, e0 + 1
        out.append({"start_frame": start, "end_frame": end, "start_s": round(start / fps, 3), "end_s": round(end / fps, 3)})
    return out


def evaluate(*, scores: np.ndarray, fps: float, scores_q: np.ndarray | None = None, audio: np.ndarray | None = None,
             sr: int | None = None, gray: np.ndarray | None = None, landmarks: np.ndarray | None = None) -> dict[str, Any]:
    """The sync block of studio.scene-map/1."""
    off = global_offset(scores)
    music, ler = music_like(audio, sr) if audio is not None and sr else (False, None)
    frames = int(off["frames"])
    result: dict[str, Any] = {
        "schema": SCHEMA, "offset_frames": frames, "offset_ms": round(frames * 1000.0 / fps), "offset_z": off["z"],
        "material": "music" if music else "speech", "low_energy_ratio": ler,
        "offset_only": False, "foreign": None, "drift": None, "frozen": [], "parameters": dict(PARAMETERS),
    }
    if music:
        result["offset_only"] = True
        result["offset_only_reason"] = "music"
        return result
    if gray is not None and landmarks is not None:
        # The picture alone decides a frozen mouth; it is measured before (and independent of) the sound check.
        result["frozen"] = refine_fixed(frozen_windows(still_mouth_series(gray, landmarks, fps), fps), gray, landmarks, fps)
    result["foreign"] = bool(off["z"] < FOREIGN_Z_MIN)
    if result["foreign"]:
        result["offset_only"] = True
        result["offset_only_reason"] = "foreign"
        return result
    if scores_q is not None:
        result["drift"] = drift(scores_q, fps)
        if not result["drift"]["measurable"]:
            result["offset_only"] = True
            result["offset_only_reason"] = "drift_not_measurable"
    return result
