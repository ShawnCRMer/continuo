"""Film Suite scene-analysis runner (lane w6-scene): one job, JSON in (stdin) / JSON out (stdout), then exit.

Runs ONLY inside the engine venv made by ``scripts/install-scene-engine.py`` (torch, torchvision,
transformers pinned there). Parts, each optional per request:

* cuts      OmniShotCut (MIT, uva-cv-lab rev 7b59cc5), vendored by the installer with ``pretrained=False`` in its
            backbone: upstream asks torchvision for ImageNet weights, which the checkpoint overwrites anyway.
            The HF download counter of ``omnishotcut.load`` is never called (``engine.load_model`` directly).
* speakers  Nemotron-3-Diarization (OpenMDW-1.1, rev f667ed7) from the local snapshot, offline.
* sync      SyncNet v2 per-frame scores (whole and quarter-frame lags) on a MediaPipe face track
            (ComfyUI's torch port, read in place), then ``sync_eval.evaluate`` (the backend's own file).
* motion    mean absolute luma difference between frames (no model).

Network is closed for the whole job: HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE, and torch.hub's download is
replaced by a function that raises, so a library that tries to fetch weights fails loudly instead.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
HERE = Path(__file__).resolve().parent
RESULT = "studio.scene-runner-result/1"
VS, VQ = 25, 100


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _no_download(*_a, **_k):
    raise RuntimeError("scene engine: downloads are disabled")


def _close_network():
    import torch.hub
    torch.hub.download_url_to_file = _no_download
    torch.hub.load_state_dict_from_url = _no_download
    try:
        import torchvision.models._api as tv_api
        tv_api.load_state_dict_from_url = _no_download
    except Exception:  # noqa: BLE001
        pass


def _frames(path, *, size=None, fmt="rgb24"):
    import av
    import numpy as np
    c = av.open(str(path))
    st = c.streams.video[0]
    fps = float(st.average_rate or st.guessed_rate or 24)
    out = []
    for f in c.decode(video=0):
        if size:
            f = f.reformat(width=size[0], height=size[1])
        out.append(f.to_ndarray(format=fmt))
    c.close()
    return np.stack(out), fps


def _audio(path, sr=16000):
    import av
    import numpy as np
    c = av.open(str(path))
    if not c.streams.audio:
        c.close()
        return None
    rs = av.AudioResampler(format="flt", layout="mono", rate=sr)
    parts = []
    for f in c.decode(audio=0):
        for g in rs.resample(f):
            parts.append(g.to_ndarray().reshape(-1))
    for g in rs.resample(None):
        parts.append(g.to_ndarray().reshape(-1))
    c.close()
    return np.concatenate(parts).astype("float32") if parts else None


# ------------------------------------------------------------------ cuts
def run_cuts(req, device):
    import torch
    os.environ["FILM_SUITE_SCENE_TORCH_DEVICE"] = device
    sys.path.insert(0, str(HERE / "vendor"))
    from omnishotcut.engine import _run_on_numpy, load_model  # noqa: E402  (vendored, pretrained=False patched in)
    from omnishotcut.datasets.utils import _resize_video  # noqa: E402
    from omnishotcut.label_correspondence import inter_int2string, intra_int2string  # noqa: E402
    model, args = load_model(req["weights"]["omnishotcut"])
    model.to(device)
    frames, fps = _frames(req["video_path"], size=(args.process_width, args.process_height))
    frames = _resize_video(frames, args.process_width, args.process_height)
    with torch.no_grad():
        ranges, intra, inter = _run_on_numpy(frames, model, args, 10)
    intra = [str(intra_int2string.get(x, x)) for x in intra]
    inter = [str(inter_int2string.get(x, x)) for x in inter]
    events = []
    gen = [s.lower() == "general" for s in intra]
    for k in range(1, len(ranges)):
        a, b = int(ranges[k][0]), int(ranges[k][1])
        if not gen[k]:
            events.append({"kind": "dissolve" if intra[k].lower() == "dissolve" else "transition", "label": intra[k],
                           "start_frame": a, "end_frame": b - 1})
        elif gen[k - 1]:
            events.append({"kind": "hard", "label": inter[k], "start_frame": a, "end_frame": a})
    return {"fps": fps, "n_frames": int(len(frames)), "events": events}


# ------------------------------------------------------------------ speakers
def run_speakers(req, device):
    import torch
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor
    snap = req["weights"]["nemotron"]
    proc = AutoProcessor.from_pretrained(snap, local_files_only=True)
    model = AutoModelForAudioFrameClassification.from_pretrained(snap, local_files_only=True).to(device).eval()
    y = _audio(req.get("audio_path") or req["video_path"])
    if y is None:
        return {"turns": [], "count": 0}
    inp = proc(y, sampling_rate=16000).to(device, dtype=model.dtype)
    with torch.no_grad():
        logits = model(**inp).logits
    segs = proc.extract_speaker_dict(logits, inp.get("attention_mask"))[0]
    turns = sorted(({"speaker": str(s["Speaker"]), "start_s": round(float(s["Start"]), 3), "end_s": round(float(s["End"]), 3)}
                    for s in segs), key=lambda t: t["start_s"])
    return {"turns": turns, "count": len({t["speaker"] for t in turns})}


# ------------------------------------------------------------------ sync
def _mediapipe(req, device):
    import torch
    from safetensors.torch import load_file
    sys.path.insert(0, req["comfy_root"])
    from comfy_extras.mediapipe.face_landmarker import FaceLandmarker
    sd = load_file(req["weights"]["mediapipe"])
    sub = {k: v for k, v in sd.items() if k.startswith(("mesh.", "blendshapes."))}
    sub.update({f"detector.{k[len('detector_full.'):]}": v for k, v in sd.items() if k.startswith("detector_full.")})
    fl = FaceLandmarker(device=device, dtype=torch.float32, operations=None, detector_variant="full").eval()
    fl.load_state_dict(sub, strict=False)
    fl.to(device)
    if hasattr(fl, "set_canonical"):
        fl.set_canonical({k: v.numpy() for k, v in sd.items() if k.startswith("canonical")})
    return fl


def run_sync(req, device):
    import cv2
    import numpy as np
    import torch
    from scipy.signal import medfilt, resample_poly
    import python_speech_features
    sys.path.insert(0, str(HERE / "vendor"))
    from SyncNetModel import S
    import sync_eval
    rgb, fps = _frames(req["video_path"])
    a = _audio(req["video_path"], sr=48000)
    if a is None:
        return {"available": False, "reason": "no_sound"}
    fl = _mediapipe(req, device)
    faces = []
    with torch.no_grad():
        for i in range(0, len(rgb), 32):
            for f in fl.detect_batch(list(rgb[i:i + 32]), num_faces=1, score_thresh=0.4):
                faces.append(f[0] if f else None)
    ok = np.array([f is not None for f in faces])
    if ok.mean() < 0.2:
        return {"available": False, "reason": "no_face", "face_rate": round(float(ok.mean()), 3)}
    n = len(rgb)
    idx = np.where(ok)[0]
    box = np.array([faces[i]["bbox_xyxy"] for i in idx], np.float64)
    allb = np.stack([medfilt(np.interp(np.arange(n), idx, box[:, k]), 13) for k in range(4)], 1)
    lm = np.full((n, 6, 2), np.nan)
    for i in idx:
        lm[i] = faces[i]["landmarks_xy"][[13, 14, 61, 291, 33, 263]]
    for p in range(6):
        for c in range(2):
            lm[:, p, c] = np.interp(np.arange(n), idx, lm[idx, p, c])
    del fl
    sn = S(num_layers_in_fc_layers=1024).to(device).eval()
    sd = torch.load(req["weights"]["syncnet"], map_location="cpu", weights_only=False)
    sn.load_state_dict(sd if isinstance(sd, dict) else sd.state_dict())
    crops = []
    for f, b in zip(rgb, allb):
        f = f[..., ::-1]
        x0, y0, x1, y1 = b
        bs = max(x1 - x0, y1 - y0) / 2
        bsi = int(bs * 1.8)
        fp = np.pad(f, ((bsi, bsi), (bsi, bsi), (0, 0)), constant_values=110)
        my, mx = (y0 + y1) / 2 + bsi, (x0 + x1) / 2 + bsi
        crops.append(cv2.resize(fp[int(my - bs):int(my + bs * 1.8), int(mx - bs * 1.4):int(mx + bs * 1.4)], (224, 224)))
    im = torch.from_numpy(np.stack(crops).astype(np.float32)).permute(3, 0, 1, 2)[None]
    target = int(round(16000 * fps / 25))  # feature hops stay locked to the picture at any frame rate
    from math import gcd
    g = gcd(48000, target)
    a16 = resample_poly(a, target // g, 48000 // g).astype(np.float32)
    mf = python_speech_features.mfcc((np.clip(a16, -1, 1) * 32767).astype(np.int16), 16000)
    cc = torch.from_numpy(mf.T[None, None].astype(np.float32))
    nn_ = min(n, cc.shape[-1] // 4) - 5
    m = cc.shape[-1] - 20
    with torch.no_grad():
        imf = torch.cat([sn.forward_lip(torch.cat([im[:, :, i:i + 5] for i in range(s, min(nn_, s + 64))]).to(device)).cpu() for s in range(0, nn_, 64)])
        ccf = torch.cat([sn.forward_aud(torch.cat([cc[:, :, :, i:i + 20] for i in range(s, min(m, s + 256))]).to(device)).cpu() for s in range(0, m, 256)])
    Sq = np.full((nn_, 2 * VQ + 1), np.nan, np.float32)
    for j, d in enumerate(range(-VQ, VQ + 1)):
        ix = np.arange(nn_) * 4 + d
        good = (ix >= 0) & (ix < m)
        Sq[good, j] = -torch.nn.functional.pairwise_distance(imf[good], ccf[ix[good]]).numpy()
    Sw = Sq[:, ::4][:, (VQ - 4 * VS) // 4:(VQ + 4 * VS) // 4 + 1]
    gray = np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (f.shape[1] // 2, f.shape[0] // 2)) for f in rgb])
    a_eval = resample_poly(a, 1, 3).astype(np.float32)
    out = sync_eval.evaluate(scores=Sw, fps=fps, scores_q=Sq, audio=a_eval, sr=16000, gray=gray, landmarks=lm / 2.0)
    out["face_rate"] = round(float(ok.mean()), 3)
    out["available"] = True
    return out


# ------------------------------------------------------------------ motion
def run_motion(req):
    import numpy as np
    g, fps = _frames(req["video_path"], size=(160, 90), fmt="gray")
    g = g.astype(np.float32)
    e = np.r_[0.0, np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))] / 255.0
    return {"fps": fps, "energy": [round(float(x), 4) for x in e]}


def main() -> int:
    req = json.loads(sys.stdin.read())
    for key, want in (req.get("weights_sha256") or {}).items():
        path = (req.get("weights") or {}).get(key)
        if path and Path(path).is_file() and _sha256(path) != want:
            print(json.dumps({"schema": RESULT, "ok": False, "reason": "weights_sha_mismatch", "which": key}))
            return 2
    import torch
    _close_network()
    device = req.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if device == "cpu":
        torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    out = {"schema": RESULT, "ok": True, "device": device, "parts": {}, "timing": {}, "errors": {}}
    for part in req.get("parts") or ["cuts", "speakers", "sync", "motion"]:
        t0 = time.perf_counter()
        try:
            if part == "cuts":
                out["parts"]["cuts"] = run_cuts(req, device)
            elif part == "speakers":
                out["parts"]["speakers"] = run_speakers(req, device)
            elif part == "sync":
                out["parts"]["sync"] = run_sync(req, device)
            elif part == "motion":
                out["parts"]["motion"] = run_motion(req)
        except Exception as exc:  # noqa: BLE001 - one part failing is named, the others still answer
            out["errors"][part] = f"{type(exc).__name__}: {str(exc)[:300]}"
        out["timing"][part] = round(time.perf_counter() - t0, 3)
    if device == "cuda":
        out["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    out["versions"] = {"torch": torch.__version__, "python": sys.version.split()[0]}
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
