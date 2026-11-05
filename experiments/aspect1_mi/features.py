"""Feature extraction, independent of Seed-VC (spec section 4), for every generated step output and for the originals.

Speaker features f_S:  wavlm   microsoft/wavlm-base-plus-sv x-vector (512)      primary ASV family
                       campplus Seed-VC's CAM++ style encoder (192)              second view only
Content features f_C:  whisper openai/whisper-small encoder, mean over frames (768)
                       xlsr    facebook/wav2vec2-xls-r-300m last hidden, mean (1024)
Raw state (Y_t^A):     melstat mean and std over time of the saved mel x_t (160)  (t = 0 sanity check only)

Rows: every (utt_id, seed, k) of <out>/metadata.csv with kind = 'gen' (file gen/<utt>__r<r>__k<kk>.wav), plus every
manifest utterance with kind = 'orig', k = -1 (the ceiling) and, if --noise-control, kind = 'noise' (pure noise of the
same length, negative control). Output: <out>/features/<encoder>.npz with `emb` (rows x dim) and `rows.csv`.

    python features.py --out <out> --manifest m.csv [--encoders wavlm,campplus,whisper,xlsr,melstat] [--seedvc-root R]
"""
import argparse, csv, io, sys, time
import os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next((d / "pipeline" for d in HERE.parents if (d / "pipeline" / "src" / "anonymize.py").exists()), None) or (Path(os.environ["ANON_ROOT"]) if os.environ.get("ANON_ROOT") else next(d for d in HERE.parents if (d / "src" / "anonymize.py").exists()))   # pipeline copy in this repo, else ANON_ROOT, else a parent folder
sys.path.insert(0, str(ROOT / "src"))


def read_csv(p):
    rows = list(csv.DictReader(open(p, newline="", encoding="utf-8")))
    return [{k.strip(): (v or "").strip() for k, v in r.items()} for r in rows]


class Wavlm:
    dim = 512
    def __init__(self, device):
        from evaluate import SpeakerVerifier
        self.sv = SpeakerVerifier("microsoft/wavlm-base-plus-sv", device)
    def __call__(self, x):
        import torch
        inp = self.sv.fe(x, sampling_rate=16000, return_tensors="pt", padding=True).to(self.sv.device)
        with torch.no_grad():
            e = self.sv.model(**inp).embeddings[0]
        return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()


class Campplus:
    dim = 192
    def __init__(self, device, seedvc_root):
        import torch
        sys.path.insert(0, str(seedvc_root))
        from modules.campplus.DTDNN import CAMPPlus
        self.m = CAMPPlus(feat_dim=80, embedding_size=192)
        self.m.load_state_dict(torch.load(str(Path(seedvc_root) / "campplus_cn_common.bin"), map_location="cpu"))
        self.m = self.m.to(device).eval(); self.device = device
    def __call__(self, x):
        import torch, torchaudio
        w = torch.from_numpy(x)[None].to(self.device)
        f = torchaudio.compliance.kaldi.fbank(w, num_mel_bins=80, dither=0, sample_frequency=16000)
        f = f - f.mean(dim=0, keepdim=True)
        with torch.no_grad():
            e = self.m(f.unsqueeze(0))[0]
        return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()


class Whisper:
    dim = 768
    def __init__(self, device):
        from transformers import WhisperFeatureExtractor, WhisperModel
        self.fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        self.enc = WhisperModel.from_pretrained("openai/whisper-small").encoder.to(device).eval(); self.device = device
    def __call__(self, x):
        import torch
        inp = self.fe(x, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)   # 30 s padded
        with torch.no_grad():
            h = self.enc(inp).last_hidden_state[0]                                                   # (1500, 768), 50/s
        n = max(1, min(1500, int(round(len(x) / 16000 * 50))))
        return h[:n].mean(0).cpu().numpy()


class Xlsr:
    dim = 1024
    def __init__(self, device):
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
        self.m = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m").to(device).eval(); self.device = device
    def __call__(self, x):
        import torch
        inp = self.fe(x, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
        with torch.no_grad():
            h = self.m(inp).last_hidden_state[0]
        return h.mean(0).cpu().numpy()


def melstat(npz_path, k):
    z = np.load(io.BytesIO(Path(npz_path).read_bytes()))   # one read call: NFS latency
    x = z[f"x_{k:02d}"].astype("float32")
    return np.concatenate([x.mean(1), x.std(1)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--encoders", default="wavlm,campplus,whisper,xlsr,melstat")
    ap.add_argument("--seedvc-root", default=str(ROOT / "repos" / "seed-vc"))
    ap.add_argument("--noise-control", action="store_true", help="add pure-noise rows (negative control)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    import torch
    from evaluate import load_16k
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); fdir = out / "features"; fdir.mkdir(exist_ok=True)
    meta = read_csv(out / "metadata.csv")[: a.limit]
    man = {r["utt_id"]: r for r in read_csv(a.manifest)}
    rows = []                                                           # (kind, utt_id, src_spk, sentence_id, target_spk, seed, k, wav or npz)
    for r in meta:
        for k in map(int, r["steps_saved"].split()):
            rows.append(dict(kind="gen", utt_id=r["utt_id"], src_spk=r["src_spk"], sentence_id=r["sentence_id"], target_spk=r["target_spk"],
                             seed=r["seed"], k=k, split=r["split"], wav=str(out / "gen" / f"{r['utt_id']}__r{r['seed']}__k{k:02d}.wav"), npz=str(out / r["path"])))
    for u in sorted({r["utt_id"] for r in meta}):
        m = man[u]
        rows.append(dict(kind="orig", utt_id=u, src_spk=m["src_spk"], sentence_id=m["sentence_id"], target_spk="", seed="", k=-1, split=m.get("split", ""), wav=m["wav"], npz=""))
        if a.noise_control:
            rows.append(dict(kind="noise", utt_id=u, src_spk=m["src_spk"], sentence_id=m["sentence_id"], target_spk="", seed="", k=-1, split=m.get("split", ""), wav=m["wav"], npz=""))
    with open(fdir / "rows.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} rows: {sum(r['kind'] == 'gen' for r in rows)} generated, {sum(r['kind'] == 'orig' for r in rows)} originals, device {dev}")
    rng = np.random.default_rng(0)
    cache = {}
    def audio(r):
        key = (r["kind"], r["wav"])
        if key not in cache:
            x = load_16k(r["wav"])
            if r["kind"] == "noise":
                x = (0.05 * rng.standard_normal(len(x))).astype("float32")
            cache.clear(); cache[key] = x
        return cache[key]
    for name in a.encoders.split(","):
        t0 = time.time()
        if name == "melstat":
            emb = np.full((len(rows), 160), np.nan, np.float32)
            for i, r in enumerate(rows):
                if r["kind"] == "gen":
                    emb[i] = melstat(r["npz"], r["k"])
        else:
            enc = {"wavlm": lambda: Wavlm(dev), "campplus": lambda: Campplus(dev, a.seedvc_root), "whisper": lambda: Whisper(dev), "xlsr": lambda: Xlsr(dev)}[name]()
            emb = np.zeros((len(rows), enc.dim), np.float32)
            for i, r in enumerate(rows):
                emb[i] = enc(audio(r))
                if (i + 1) % 500 == 0:
                    el = time.time() - t0
                    print(f"  {name} [{i + 1}/{len(rows)}] {el / 60:.1f} min, left ~{el / (i + 1) * (len(rows) - i - 1) / 60:.0f} min", flush=True)
            del enc; torch.cuda.empty_cache() if dev == "cuda" else None
        np.savez(fdir / f"{name}.npz", emb=emb)
        print(f"{name}: {emb.shape} in {(time.time() - t0) / 60:.1f} min -> {fdir / (name + '.npz')}")


if __name__ == "__main__":
    main()
