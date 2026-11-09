"""Film Suite song-analysis runner: one job, JSON in (stdin) / JSON out (stdout), then exit.

Runs ONLY inside the engine venv made by ``scripts/install-song-engine.py`` (torch,
transformers, muq, onnxruntime, librosa pinned there; never in the Hermes venv, never in
ComfyUI's python).  The backend (``music_core/song_map.py``) starts it once per song-map
job and reads the last stdout line.  Exiting releases every byte of GPU memory it used.

Tasks (each reports ok / reason on its own; one failing task never hides another):
  drums     ADTOF Frame_RNN (xavriley/ADTOF-pytorch) on the mix: kick / snare / hat / crash
            onsets with the activation at each peak as confidence.         (B2 winner)
  melody    RMVPE onnx on the vocal stem: f0 on a 10 ms grid + voiced flag, notes and the
            peak note of every phrase.                                       (B3 winner)
  words     Qwen3-ForcedAligner-0.6B with the given lyrics, on the vocal stem.  (B4 winner)
            Carries the three fixes of eval/bench/RESULTS.md B4.
  sections  SongFormer (HX-8class head).                                     (B1 winner)
            Carries the four offline / transformers-5 fixes of RESULTS.md B1.
  novelty   Foote checkerboard novelty on the given beats (librosa chroma + MFCC).
  loudness  RMS dBFS per 100 ms of the mix and of the instrumental stem.
  events    silence (RMS < -50 dBFS for >= 0.5 s) and risers (RMS rising >= 3 dB/s over >= 2 s,
            linear fit R^2 >= 0.8).
Nothing is ever downloaded: HF_HUB_OFFLINE is forced before any import, and the only hub id
in the stack (MusicFM's Wav2Vec2ConformerConfig.from_pretrained) is answered from the
config shipped with SongFormer.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

RESULT_SCHEMA = "studio.song-runner-result/1"
REQUEST_SCHEMA = "studio.song-runner-request/1"
TASKS = ("drums", "melody", "words", "sections", "novelty", "loudness", "events")
ADTOF_CLASSES = {"kick": 0, "snare": 1, "tom": 2, "hat": 3, "crash": 4}  # LABELS_5 = 35, 38, 47, 42, 49
ADTOF_THRESHOLDS = (0.22, 0.24, 0.32, 0.22, 0.30)  # FRAME_RNN_THRESHOLDS of ADTOF


def _r(x, n=3):
    return round(float(x), n)


# ------------------------------------------------------------------ audio helpers
def _load(path, sr, mono=True):
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=mono)
    return y


def _rms_db(y, sr, hop_s=0.1):
    import numpy as np
    hop = int(sr * hop_s)
    n = max(1, len(y) // hop)
    frames = y[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


# ------------------------------------------------------------------ drums (ADTOF)
def task_drums(req, dev):
    import numpy as np
    import torch
    from adtof_pytorch import create_frame_rnn_model, calculate_n_bins, load_audio_for_model, load_pytorch_weights
    from adtof_pytorch.post_processing import NotePeakPickingProcessor
    model = create_frame_rnn_model(calculate_n_bins())
    model = load_pytorch_weights(model, req["weights"]["adtof"], strict=False).to(dev).eval()
    x = load_audio_for_model(req["audio_path"]).to(dev)
    with torch.no_grad():
        act = model(x).float().cpu().numpy()[0]  # T, 5 at 100 fps
    out = {}
    confs = []
    for name in ("kick", "snare", "hat", "crash"):
        i = ADTOF_CLASSES[name]
        picker = NotePeakPickingProcessor(threshold=ADTOF_THRESHOLDS[i], pre_avg=0.1, post_avg=0.01, pre_max=0.02,
                                          post_max=0.01, combine=0.02, fps=100)
        hits = []
        for t, _p in picker.process(act[:, i]):
            f = int(round(t * 100))
            c = float(act[max(0, f - 1): f + 2, i].max())
            hits.append({"t": _r(t), "confidence": _r(c)})
            confs.append(c)
        out[name] = hits
    return {"classes": out, "fps": 100,
            "confidence": _r(np.mean(confs)) if confs else 0.0,
            "confidence_basis": "mean ADTOF activation at the picked onsets"}


# ------------------------------------------------------------------ melody (RMVPE)
def task_melody(req, dev):
    import librosa
    import numpy as np
    import onnxruntime as ort
    vocal = (req.get("stems") or {}).get("vocals")
    if not vocal:
        return {"ok": False, "reason": "no_vocal_stem"}
    y16 = _load(vocal, 16000)
    sess = ort.InferenceSession(req["weights"]["rmvpe"], providers=["CPUExecutionProvider"])
    mel_fb = librosa.filters.mel(sr=16000, n_fft=1024, n_mels=128, fmin=30, fmax=8000, htk=True)
    S = np.abs(librosa.stft(y16, n_fft=1024, hop_length=160, win_length=1024, window="hann", center=True))
    mel = np.log(np.clip(mel_fb @ S, 1e-5, None)).astype(np.float32)
    T = mel.shape[1]
    pad = 32 * ((T - 1) // 32 + 1) - T
    mel = np.pad(mel, ((0, 0), (0, pad)), mode="constant")
    hid = sess.run(None, {"input": mel[None]})[0][0][:T]
    cents_map = 20 * np.arange(360) + 1997.3794084376191
    idx = hid.argmax(1)
    cents = np.zeros(T)
    for t in range(T):
        lo, hi = max(0, idx[t] - 4), min(360, idx[t] + 5)
        w = hid[t, lo:hi]
        cents[t] = (w * cents_map[lo:hi]).sum() / (w.sum() + 1e-9)
    conf = hid.max(1)
    f0 = 10 * 2 ** (cents / 1200)
    voiced = conf > 0.03
    f0[~voiced] = 0
    midi = np.where(voiced, 69 + 12 * np.log2(np.maximum(f0, 1e-6) / 440.0), 0)
    # notes: voiced runs split where the rounded pitch changes for >= 50 ms
    notes = []
    i = 0
    while i < T:
        if not voiced[i]:
            i += 1
            continue
        j = i
        while j < T and voiced[j] and abs(round(midi[j]) - round(midi[i])) == 0:
            j += 1
        if j == i:
            j = i + 1
        if j - i >= 5:
            seg = midi[i:j]
            notes.append({"start": _r(i * 0.01, 2), "end": _r(j * 0.01, 2), "midi": int(round(float(np.median(seg)))),
                          "hz": _r(float(np.median(f0[i:j])), 1), "confidence": _r(float(conf[i:j].mean()))})
        i = j
    # phrases: notes separated by < 0.3 s; the peak note is the highest of each phrase
    peaks, phrase = [], []
    for n in notes + [None]:
        if n is not None and (not phrase or n["start"] - phrase[-1]["end"] < 0.3):
            phrase.append(n)
            continue
        if phrase:
            top = max(phrase, key=lambda x: (x["midi"], x["end"] - x["start"]))
            peaks.append(dict(top))
        phrase = [n] if n is not None else []
    return {"hop_s": 0.01, "f0_hz": [_r(v, 1) for v in f0], "voiced": [bool(v) for v in voiced],
            "notes": notes, "peaks": peaks, "device": "cpu",
            "confidence": _r(float(conf[voiced].mean())) if voiced.any() else 0.0,
            "confidence_basis": "mean RMVPE salience over voiced frames"}


# ------------------------------------------------------------------ words (Qwen3-ForcedAligner)
FA_TEMPLATE = ("{%- for m in messages -%}{%- for c in m.content -%}"
               "{%- if c.type == 'audio' or ('audio' in c) -%}<|audio_start|><|audio_pad|><|audio_end|>"
               "{%- elif c.type == 'text' -%}{{ c.text }}<timestamp><timestamp>{%- endif -%}"
               "{%- endfor -%}{%- endfor -%}")


def task_words(req, dev):
    import numpy as np
    import torch
    lyrics = (req.get("lyrics") or "").strip()
    if not lyrics:
        return {"ok": False, "reason": "no_lyrics"}
    audio = (req.get("stems") or {}).get("vocals") or req["audio_path"]
    from safetensors.torch import load_file
    from transformers import AutoProcessor, Qwen3ASRConfig, Qwen3ASRForTokenClassification
    p = req["weights"]["qwen3_aligner_dir"]
    cfgj = json.loads(open(os.path.join(p, "config.json"), encoding="utf-8").read())
    d = cfgj["thinker_config"]
    ac, tc = dict(d["audio_config"]), dict(d["text_config"])
    ac["model_type"], tc["model_type"] = "qwen3_asr_encoder", "qwen3"
    # fix 1: from_pretrained builds the wrong projector; build from thinker_config + remap keys
    cfg = Qwen3ASRConfig(audio_config=ac, text_config=tc, audio_token_id=d["audio_token_id"], num_labels=d["classify_num"])
    m = Qwen3ASRForTokenClassification(cfg)
    sd = {}
    for k, v in load_file(os.path.join(p, "model.safetensors")).items():
        k = k.replace("thinker.audio_tower.proj1.", "model.multi_modal_projector.linear_1.") \
             .replace("thinker.audio_tower.proj2.", "model.multi_modal_projector.linear_2.") \
             .replace("thinker.audio_tower.", "model.audio_tower.") \
             .replace("thinker.model.", "model.language_model.").replace("thinker.lm_head.", "score.")
        sd[k] = v
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        return {"ok": False, "reason": "aligner_key_mismatch"}
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    m = m.to(dtype).to(dev).eval()
    proc = AutoProcessor.from_pretrained(p, local_files_only=True)
    y = _load(audio, 16000)
    # fix 3: mel length must be a multiple of 100 -> pad trailing silence to whole seconds
    n = int(np.ceil(len(y) / 16000)) * 16000
    y = np.pad(y, (0, max(0, n - len(y))))
    # fix 2: the snapshot's chat template is the ASR one; pass the aligner template
    inputs, wl = proc.prepare_forced_aligner_inputs(y, lyrics, language=req.get("language") or "en",
                                                     chat_template=FA_TEMPLATE)
    inputs = inputs.to(dev)
    for k, v in inputs.items():
        if torch.is_floating_point(v):
            inputs[k] = v.to(dtype)
    with torch.no_grad():
        lg = m(**inputs).logits
    ts_id = cfgj["timestamp_token_id"]
    out = proc.decode_forced_alignment(lg, inputs["input_ids"], wl, ts_id)[0]
    probs = torch.softmax(lg.float(), -1).max(-1).values[0]
    mask = inputs["input_ids"][0] == ts_id
    slot_conf = probs[mask].cpu().numpy()
    words = []
    for i, o in enumerate(out):
        c = float(slot_conf[2 * i: 2 * i + 2].mean()) if len(slot_conf) >= 2 * i + 2 else 0.0
        words.append({"word": o["text"], "start": _r(o["start_time"]), "end": _r(o["end_time"]), "confidence": _r(c)})
    return {"words": words, "audio": "vocals" if (req.get("stems") or {}).get("vocals") else "mix",
            "confidence": _r(float(np.mean([w["confidence"] for w in words]))) if words else 0.0,
            "confidence_basis": "mean aligner class probability at the timestamp slots"}


# ------------------------------------------------------------------ sections (SongFormer)
_HX_LABELS = {0: "intro", 1: "verse", 2: "chorus", 3: "bridge", 4: "inst", 5: "outro", 6: "silence", 26: "pre-chorus"}


def _songformer_shims(sfdir):
    """Training-only imports of the snapshot, answered in memory (fix 4)."""
    import types
    lab = types.ModuleType("dataset.label2id")
    names = ["intro", "verse", "chorus", "bridge", "inst", "outro", "silence"] + [f"_unused{i}" for i in range(7, 26)] + \
            ["pre-chorus"] + [f"_unused{i}" for i in range(27, 69)]
    lab.LABEL_TO_ID = {n: i for i, n in enumerate(names)}
    lab.ID_TO_LABEL = {i: n for i, n in enumerate(names)}
    lab.DATASET_LABEL_TO_DATASET_ID = {"SongForm-HX-8Class": 5}
    lab.DATASET_ID_ALLOWED_LABEL_IDS = {5: [0, 1, 2, 3, 4, 5, 6, 26]}
    ct = types.ModuleType("dataset.custom_types")
    ct.MsaInfo = list
    pkg = types.ModuleType("dataset")
    pkg.__path__ = []
    msaf, ev = types.ModuleType("msaf"), types.ModuleType("msaf.eval")
    msaf.__path__ = []

    def _no(*a, **k):
        raise NotImplementedError("training only")
    ev.compute_results = _no
    sys.modules.update({"dataset": pkg, "dataset.label2id": lab, "dataset.custom_types": ct, "msaf": msaf, "msaf.eval": ev})
    sys.path[:0] = [sfdir]
    os.environ["SONGFORMER_LOCAL_DIR"] = sfdir


def task_sections(req, dev):
    import numpy as np
    import torch
    sfdir = req["weights"]["songformer_dir"]
    _songformer_shims(sfdir)
    from omegaconf import OmegaConf
    from transformers.modeling_outputs import BaseModelOutput
    from transformers.models.wav2vec2_conformer import configuration_wav2vec2_conformer as w2c
    from transformers.models.wav2vec2_conformer import modeling_wav2vec2_conformer as w2m
    w2v2 = OmegaConf.to_container(OmegaConf.load(os.path.join(sfdir, "muq_config2.json")))["w2v2_config"]

    def _offline(cls, name, *a, **k):  # fix 1: MusicFM asks the hub for this config
        if name != "facebook/wav2vec2-conformer-rope-large-960h-ft":
            raise RuntimeError("unexpected hub id " + str(name))
        return cls(**w2v2)
    w2c.Wav2Vec2ConformerConfig.from_pretrained = classmethod(_offline)
    import muq.muq.models.muq_model as mm  # fix 2: a real config object instead of EasyDict
    mm.EasyDict = lambda d: w2c.Wav2Vec2ConformerConfig(**(OmegaConf.to_container(d) if hasattr(d, "_metadata") else dict(d)))

    def _enc_forward(self, hidden_states, attention_mask=None, output_hidden_states=False, **kw):  # fix 3
        if attention_mask is not None:
            hidden_states[~attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])] = 0.0
        mask = w2m.create_bidirectional_mask(config=self.config, inputs_embeds=hidden_states, attention_mask=attention_mask)
        hidden_states = self.dropout(hidden_states)
        rel = self.embed_positions(hidden_states) if self.embed_positions is not None else None
        states = []
        for layer in self.layers:
            states.append(hidden_states)
            hidden_states = layer(hidden_states, attention_mask=mask, relative_position_embeddings=rel)
        hidden_states = self.layer_norm(hidden_states)
        states.append(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=tuple(states))
    w2m.Wav2Vec2ConformerEncoder.forward = _enc_forward
    from safetensors.torch import load_file
    from configuration_songformer import SongFormerConfig
    import modeling_songformer as msf
    seen = {}
    real_pp = msf.postprocess_functional_structure

    def _pp(logits, config):  # keep the boundary probabilities for a measured confidence
        seen["prob"] = torch.sigmoid(logits["boundary_logits"][0].float()).numpy()
        seen["fr"] = config.frame_rates
        return real_pp(logits, config)
    msf.postprocess_functional_structure = _pp
    model = msf.SongFormerModel(SongFormerConfig())
    missing, unexpected = model.load_state_dict(load_file(os.path.join(sfdir, "model.safetensors")), strict=False)
    if missing or unexpected:
        return {"ok": False, "reason": "songformer_key_mismatch"}
    model = model.to(dev).eval()
    y = _load(req["audio_path"], 24000)
    segs = model(y)
    prob, fr = seen.get("prob"), seen.get("fr", 8.333)
    out, confs = [], []
    for s in segs:
        c = None
        if prob is not None and s["start"] > 0:
            f = int(round(s["start"] * fr))
            c = float(prob[max(0, f - 2): f + 3].max()) if len(prob) else None
            confs.append(c)
        out.append({"start": _r(s["start"]), "end": _r(s["end"]), "label": s["label"],
                    "confidence": None if c is None else _r(c)})
    return {"sections": out, "confidence": _r(np.mean(confs)) if confs else 0.0,
            "confidence_basis": "mean SongFormer boundary probability at the boundaries"}


# ------------------------------------------------------------------ novelty (Foote on beats)
def task_novelty(req, dev):
    import librosa
    import numpy as np
    beats = np.asarray(req.get("beats") or [], dtype=float)
    if len(beats) < 8:
        return {"ok": False, "reason": "no_beat_grid"}
    y = _load(req["audio_path"], 22050)
    sr, hop, half = 22050, 512, 16
    bf = librosa.time_to_frames(beats, sr=sr, hop_length=hop)
    feats = []
    for f in (librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop), librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)):
        s = librosa.util.sync(f, bf, aggregate=np.median)
        feats.append((s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-8))
    x = np.vstack(feats)
    x = x / (np.linalg.norm(x, axis=0, keepdims=True) + 1e-8)
    ssm = x.T @ x
    n = ssm.shape[0]
    g = np.exp(-0.5 * (np.linspace(-2, 2, 2 * half) ** 2))
    sign = np.r_[-np.ones(half), np.ones(half)]
    ker = np.outer(g, g) * np.outer(sign, sign)
    pad = np.pad(ssm, half, mode="constant")
    nov = np.maximum(np.array([(pad[i:i + 2 * half, i:i + 2 * half] * ker).sum() for i in range(n)]), 0)
    nov = nov / (nov.max() + 1e-9)
    starts = np.r_[0.0, beats]
    thr = float(nov.mean() + 0.5 * nov.std())
    peaks = [i for i in range(1, n - 1) if nov[i] >= nov[i - 1] and nov[i] > nov[i + 1] and nov[i] > thr]
    peaks.sort(key=lambda i: -nov[i])
    keep = []
    for p in peaks:
        if all(abs(p - q) >= 8 for q in keep):
            keep.append(p)
    keep.sort()
    curve = [{"t": _r(starts[i]), "value": _r(nov[i])} for i in range(min(n, len(starts)))]
    pk = [{"t": _r(starts[i]), "value": _r(nov[i])} for i in keep if i < len(starts)]
    conf = float(np.mean([(nov[i] - thr) / max(1e-6, 1 - thr) for i in keep])) if keep else 0.0
    return {"curve": curve, "peaks": pk, "threshold": _r(thr), "confidence": _r(conf),
            "confidence_basis": "mean peak height above the threshold (0..1)"}


# ------------------------------------------------------------------ loudness, silence, risers
def task_loudness(req, dev):
    out = {"hop_s": 0.1, "unit": "dBFS RMS"}
    y = _load(req["audio_path"], 22050)
    out["mix"] = [_r(v, 1) for v in _rms_db(y, 22050)]
    inst = (req.get("stems") or {}).get("instrumental")
    out["instrumental"] = [_r(v, 1) for v in _rms_db(_load(inst, 22050), 22050)] if inst else None
    out["confidence"] = 1.0
    out["confidence_basis"] = "direct measurement"
    return out


def task_events(req, dev):
    import numpy as np
    y = _load(req["audio_path"], 22050)
    db = _rms_db(y, 22050)  # 100 ms
    silence = []
    quiet = db < -50.0
    i = 0
    while i < len(db):
        if quiet[i]:
            j = i
            while j < len(db) and quiet[j]:
                j += 1
            if (j - i) * 0.1 >= 0.5:
                silence.append({"start": _r(i * 0.1, 2), "end": _r(j * 0.1, 2), "confidence": 1.0})
            i = j
        else:
            i += 1
    sm = np.convolve(db, np.ones(5) / 5, mode="same")
    win, step = 20, 2  # 2 s windows, 0.2 s step
    good = []
    for s in range(0, max(0, len(sm) - win), step):
        seg = sm[s:s + win]
        t = np.arange(win) * 0.1
        slope, icpt = np.polyfit(t, seg, 1)
        r2 = 1 - ((seg - (slope * t + icpt)) ** 2).sum() / (((seg - seg.mean()) ** 2).sum() + 1e-9)
        if slope >= 3.0 and r2 >= 0.8:
            good.append((s, s + win, r2))
    risers = []
    for s, e, r2 in good:
        if risers and s * 0.1 <= risers[-1]["end"]:
            risers[-1]["end"] = _r(e * 0.1, 2)
            risers[-1]["_r2"].append(r2)
        else:
            risers.append({"start": _r(s * 0.1, 2), "end": _r(e * 0.1, 2), "_r2": [r2]})
    for r in risers:
        r["confidence"] = _r(np.mean(r.pop("_r2")))
    conf = float(np.mean([r["confidence"] for r in risers])) if risers else 1.0
    return {"silence": silence, "risers": risers, "confidence": _r(conf),
            "confidence_basis": "risers: mean R^2 of the rising loudness fit; silence: direct measurement"}


FUNCS = {"drums": task_drums, "melody": task_melody, "words": task_words, "sections": task_sections,
         "novelty": task_novelty, "loudness": task_loudness, "events": task_events}


def main() -> int:
    req = json.loads(sys.stdin.read())
    if req.get("schema") != REQUEST_SCHEMA:
        print(json.dumps({"schema": RESULT_SCHEMA, "ok": False, "reason": "bad_request"}))
        return 2
    import torch
    import transformers
    device = req.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if device == "cpu":
        torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    results = {}
    for task in req.get("tasks") or TASKS:
        if task not in FUNCS:
            results[task] = {"ok": False, "reason": "unknown_task"}
            continue
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            res = FUNCS[task](req, device)
            res.setdefault("ok", True)
        except Exception as exc:  # noqa: BLE001 - every task reports its own failure
            res = {"ok": False, "reason": "task_failed", "detail": f"{type(exc).__name__}: {str(exc)[:200]}",
                   "trace_tail": traceback.format_exc()[-600:]}
        res["wall_s"] = round(time.perf_counter() - t0, 3)
        res.setdefault("device", device)
        if device == "cuda":
            res["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            torch.cuda.empty_cache()
        results[task] = res
    import importlib.metadata as md
    versions = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__}
    for pkg in ("librosa", "onnxruntime", "muq", "adtof-pytorch"):
        try:
            versions[pkg] = md.version(pkg)
        except Exception:  # noqa: BLE001
            versions[pkg] = None
    print(json.dumps({"schema": RESULT_SCHEMA, "ok": True, "device": device, "tasks": results, "versions": versions}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
