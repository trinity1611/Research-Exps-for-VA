"""Condition-space extraction (framework D4 / D6 / D10 / D15 inputs): what the DiT actually sees from the source.

    python extract.py --manifest m.csv --out OUT --seedvc-root R --profile configs/pipelines/combined.env [--cond-dir ASPECT1_RUN] [--no-asr] [--limit N]

For every manifest utterance (original source clip):
  H       the content condition as the engine computes it: eng.semantic(w16) = XLS-R 300m, layer 12, (L x 1024),
          50 frames/s; saved float16 as OUT/H/<utt_id>.npy (the frame-level record for later diagnostics)
  hbar    utterance mean of H (1024)          hstd  std over frames (1024)       hdelta  mean |h_t - h_{t-1}| (1024)
  mu      the length-regulated condition the DiT consumes (384 per mel frame), source part only, from the Aspect 1
          run's cond/<utt_id>.npz when --cond-dir is given: mu_mean, mu_std (384 each)
  rate    duration s, n_frames L, pause fraction (energy VAD), and with ASR (whisper-small, Hindi): characters per
          second and the transcript (OUT/asr.jsonl, resumable)
Writes OUT/features.npz (arrays, one row per utterance in manifest order) and OUT/table.csv (scalars + text).
"""
import argparse, csv, io, json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next((d / "pipeline" for d in HERE.parents if (d / "pipeline" / "src" / "anonymize.py").exists()), None) or (Path(os.environ["ANON_ROOT"]) if os.environ.get("ANON_ROOT") else next(d for d in HERE.parents if (d / "src" / "anonymize.py").exists()))   # pipeline copy in this repo, else ANON_ROOT, else a parent folder
sys.path.insert(0, str(ROOT / "src"))
SR = 16000


def read_csv(p):
    rows = list(csv.DictReader(open(p, newline="", encoding="utf-8")))
    return [{k.strip(): (v or "").strip() for k, v in r.items()} for r in rows]


def load_wav16(path):
    import soundfile as sf, torch, torchaudio
    x, sr = sf.read(io.BytesIO(Path(path).read_bytes()), dtype="float32", always_2d=True)   # one read call (slow NFS)
    x = x.mean(1)
    if sr != SR:
        x = torchaudio.functional.resample(torch.from_numpy(x)[None], sr, SR)[0].numpy()
    return x


def pause_fraction(x, win=400, hop=160, rel_db=-35.0):
    """Share of 10 ms frames whose RMS is below rel_db relative to the 95th-percentile frame RMS."""
    n = max(1, (len(x) - win) // hop + 1)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt((x[idx] ** 2).mean(1) + 1e-12)
    ref = np.percentile(rms, 95)
    return float(np.mean(20 * np.log10(rms / (ref + 1e-12) + 1e-12) < rel_db))


class Asr:
    def __init__(self, device):
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        self.proc = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", torch_dtype=torch.float16).to(device).eval()
        self.device = device
        self.ids = self.proc.get_decoder_prompt_ids(language="hi", task="transcribe")

    def __call__(self, x):
        import torch
        feats = self.proc(x, sampling_rate=SR, return_tensors="pt").input_features.to(self.device, torch.float16)
        with torch.no_grad():
            out = self.model.generate(feats, forced_decoder_ids=self.ids, max_new_tokens=220)
        return self.proc.batch_decode(out, skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seedvc-root", required=True); ap.add_argument("--profile", required=True); ap.add_argument("--models-dir", default=None)
    ap.add_argument("--cond-dir", default=None, help="Aspect 1 run dir holding cond/<utt_id>.npz (mu); omitted -> mu stats are NaN")
    ap.add_argument("--no-asr", action="store_true"); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    import torch
    import anonymize
    out = Path(a.out); (out / "H").mkdir(parents=True, exist_ok=True)
    man = read_csv(a.manifest)[: a.limit]
    sys.argv = ["anonymize.py", "--profile", a.profile, "--seedvc-root", a.seedvc_root, "--wav-dir", str(out), "--out-dir", str(out / "_unused"),
                "--log-dir", str(out / "logs"), "--tag", "audit", "--mode", "whole", "--no-warmup"] + (["--models-dir", a.models_dir] if a.models_dir else [])
    args = anonymize.parse_args(); anonymize.apply_profile(args)
    svc = anonymize.bootstrap_seedvc(args.seedvc_root)
    eng = anonymize.SeedVCEngine(svc, args)
    torch.set_grad_enabled(False)
    dev = eng.device
    asr = None if a.no_asr else Asr(dev)
    # resumable ASR record
    asr_path = out / "asr.jsonl"; texts = {}
    if asr_path.exists():
        for line in asr_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line); texts[d["utt_id"]] = d["text"]
    asr_fh = open(asr_path, "a", encoding="utf-8")
    print(f"{len(man)} utterances; engine semantic = XLS-R layer {getattr(eng, 'output_layer', '?')}; asr={'off' if a.no_asr else 'whisper-small hi'}; "
          f"{len(texts)} transcripts resumed; device {dev}", flush=True)
    n = len(man)
    hbar = np.zeros((n, 1024), np.float32); hstd = np.zeros_like(hbar); hdelta = np.zeros_like(hbar)
    mu_mean = np.full((n, 384), np.nan, np.float32); mu_std = np.full((n, 384), np.nan, np.float32)
    scal = []
    t0 = time.time()
    for i, r in enumerate(man):
        u = r["utt_id"]
        hp = out / "H" / f"{u}.npy"
        x = load_wav16(r["wav"])
        if hp.exists():
            H = np.load(io.BytesIO(hp.read_bytes())).astype(np.float32)
        else:
            H = eng.semantic(torch.from_numpy(x)[None].to(dev))[0].float().cpu().numpy()   # (L, 1024)
            np.save(hp, H.astype(np.float16))
        hbar[i] = H.mean(0); hstd[i] = H.std(0); hdelta[i] = np.abs(np.diff(H, axis=0)).mean(0) if len(H) > 1 else 0
        if a.cond_dir:
            cp = Path(a.cond_dir) / "cond" / f"{u}.npz"
            if cp.exists():
                z = np.load(io.BytesIO(cp.read_bytes()))
                mu = z["mu"][int(z["prompt_frames"]):].astype(np.float32)   # (frames, 384), source part only
                mu_mean[i] = mu.mean(0); mu_std[i] = mu.std(0)
        if asr is not None and u not in texts:
            texts[u] = asr(x)
            asr_fh.write(json.dumps({"utt_id": u, "text": texts[u]}, ensure_ascii=False) + "\n"); asr_fh.flush()
        txt = texts.get(u, "")
        dur = len(x) / SR
        n_chars = len(txt.replace(" ", ""))
        scal.append(dict(utt_id=u, src_spk=r["src_spk"], sentence_id=r.get("sentence_id", u), dur_s=f"{dur:.3f}", n_frames=len(H),
                         pause_frac=f"{pause_fraction(x):.4f}", n_chars=n_chars, chars_per_s=f"{n_chars / dur:.3f}", text=txt))
        if (i + 1) % 100 == 0 or i + 1 == n:
            el = time.time() - t0
            print(f"[{i + 1}/{n}] {el / 60:.1f} min, left ~{el / (i + 1) * (n - i - 1) / 60:.0f} min", flush=True)
    asr_fh.close()
    np.savez(out / "features.npz", hbar=hbar, hstd=hstd, hdelta=hdelta, mu_mean=mu_mean, mu_std=mu_std,
             utt_id=np.array([r["utt_id"] for r in man]), src_spk=np.array([r["src_spk"] for r in man]))
    with open(out / "table.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(scal[0].keys())); w.writeheader(); w.writerows(scal)
    print(f"written: {out / 'features.npz'}, {out / 'table.csv'}, {out / 'H'} ({n} files), {asr_path if asr else ''}")


if __name__ == "__main__":
    main()
