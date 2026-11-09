"""Film Suite Beat This! runner: one job, JSON in (stdin) / JSON out (stdout), then exit.

Runs ONLY inside the Film-Suite-owned engine venv created by
``scripts/install-beat-engine.py`` (torch + beat_this pinned there; never in the Hermes
venv, never in ComfyUI's python).  The backend starts it per analysis and reads the last
stdout line.  Exiting releases every byte of GPU memory the job used.

Output: beats / downbeats from Beat This! final0 with the minimal decoder (=
``File2Beats(dbn=False)``, chosen by the fs-beat bake-off), plus the per-beat
probabilities the backend needs for honest status: beat probability at each beat, at
each midpoint between beats (double-time rival) and downbeat probability at each beat.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _at(prob, times, fps):
    import numpy as np
    out = []
    n = len(prob)
    for t in times:
        f = int(round(float(t) * fps))
        lo, hi = max(0, f - 2), min(n, f + 3)
        out.append(float(prob[lo:hi].max()) if hi > lo else 0.0)
    return [round(v, 4) for v in out] if out else list(np.zeros(0))


def main() -> int:
    req = json.loads(sys.stdin.read())
    import numpy as np
    import torch
    import beat_this
    from beat_this.inference import Audio2Frames
    from beat_this.model.postprocessor import Postprocessor
    from beat_this.preprocessing import load_audio

    ckpt = req["checkpoint"]
    if req.get("checkpoint_sha256") and _sha256(ckpt) != req["checkpoint_sha256"]:
        print(json.dumps({"schema": "studio.beat-runner-result/1", "ok": False, "reason": "checkpoint_sha_mismatch"}))
        return 2
    device = req.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if device == "cpu":
        torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    t0 = time.perf_counter()
    frames = Audio2Frames(checkpoint_path=ckpt, device=device, float16=False)
    t1 = time.perf_counter()
    signal, sr = load_audio(req["audio_path"])
    if signal.ndim > 1:
        signal = signal.mean(1)
    with torch.inference_mode():
        beat_logits, down_logits = frames(signal, sr)
        post = Postprocessor(type="minimal", fps=50)
        beats, downs = post(beat_logits, down_logits)
    if device == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    bp = torch.sigmoid(beat_logits.float()).cpu().numpy()
    dp = torch.sigmoid(down_logits.float()).cpu().numpy()
    beats = np.asarray(beats, dtype=float)
    downs = np.asarray(downs, dtype=float)
    mids = (beats[:-1] + beats[1:]) / 2 if len(beats) > 1 else np.zeros(0)
    out = {
        "schema": "studio.beat-runner-result/1",
        "ok": True,
        "device": device,
        "beats": [round(float(x), 4) for x in beats],
        "downbeats": [round(float(x), 4) for x in downs],
        "beat_prob": _at(bp, beats, 50.0),
        "mid_prob": _at(bp, mids, 50.0),
        "down_prob": _at(dp, beats, 50.0),
        "fps": 50.0,
        "versions": {"beat_this": getattr(beat_this, "__version__", None) or "1.1.0", "torch": torch.__version__,
                     "python": sys.version.split()[0]},
        "timing": {"load_s": round(t1 - t0, 3), "infer_s": round(t2 - t1, 3)},
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device == "cuda" else 0.0,
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
