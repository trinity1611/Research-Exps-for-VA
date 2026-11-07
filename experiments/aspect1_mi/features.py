"""Feature extraction, independent of Seed-VC (spec section 4), for every generated step output and for the originals.

Speaker features f_S:  wavlm   microsoft/wavlm-base-plus-sv x-vector (512)      primary ASV family
                       campplus Seed-VC's CAM++ style encoder (192)              second view only
Content features f_C:  whisper openai/whisper-small encoder, mean over frames (768)
                       xlsr    facebook/wav2vec2-xls-r-300m last hidden, mean (1024)
Raw state (Y_t^A):     melstat mean and std over time of the saved mel x_t (160)  (t = 0 sanity check only)

Rows: every (utt_id, seed, k) of <out>/metadata.csv with kind = 'gen' (file gen/<utt>__r<r>__k<kk>.wav), plus every
manifest utterance with kind = 'orig', k = -1 (the ceiling) and, if --noise-control, kind = 'noise' (pure noise of the
same length, negative control). Output: <out>/features/<encoder>.npz with `emb` (rows x dim) and `rows.csv`.

    python features.py --out <out> --manifest m.csv [--encoders wavlm,campplus,whisper,xlsr,melstat] [--seedvc-root R] [--batch 32] [--verify]
Clips are batched per utterance (all its step outputs have the source's length), so no padding: batched numbers equal single-clip numbers.
"""
import argparse, csv, sys, time
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
    """Batched: xs = list of equal-length 16 kHz float32 arrays -> (B, 512)."""
    dim = 512
    def __init__(self, device):
        from evaluate import SpeakerVerifier
        self.sv = SpeakerVerifier("microsoft/wavlm-base-plus-sv", device)
    def __call__(self, xs):
        import torch
        inp = self.sv.fe(list(xs), sampling_rate=16000, return_tensors="pt", padding=True).to(self.sv.device)
        with torch.no_grad():
            e = self.sv.model(**inp).embeddings
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
    def __call__(self, xs):
        import torch, torchaudio
        feats = []
        for x in xs:
            w = torch.from_numpy(x)[None].to(self.device)
            f = torchaudio.compliance.kaldi.fbank(w, num_mel_bins=80, dither=0, sample_frequency=16000)
            feats.append(f - f.mean(dim=0, keepdim=True))
        with torch.no_grad():
            e = self.m(torch.stack(feats))                       # equal lengths -> equal frame counts, no padding
        return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()


class Whisper:
    dim = 768
    def __init__(self, device):
        from transformers import WhisperFeatureExtractor, WhisperModel
        self.fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        self.enc = WhisperModel.from_pretrained("openai/whisper-small").encoder.to(device).eval(); self.device = device
    def __call__(self, xs):
        import torch
        inp = self.fe(list(xs), sampling_rate=16000, return_tensors="pt").input_features.to(self.device)   # 30 s padded
        with torch.no_grad():
            h = self.enc(inp).last_hidden_state                                                           # (B, 1500, 768), 50/s
        out = []
        for i, x in enumerate(xs):
            n = max(1, min(1500, int(round(len(x) / 16000 * 50))))
            out.append(h[i, :n].mean(0))
        return torch.stack(out).cpu().numpy()


class Xlsr:
    dim = 1024
    def __init__(self, device):
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
        self.m = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m").to(device).eval(); self.device = device
    def __call__(self, xs):
        import torch
        inp = self.fe(list(xs), sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(self.device)
        with torch.no_grad():
            h = self.m(inp).last_hidden_state                    # equal lengths -> no padding
        return h.mean(1).cpu().numpy()


def melstat(npz_path, k):
    z = np.load(npz_path)
    x = z[f"x_{k:02d}"].astype("float32")
    return np.concatenate([x.mean(1), x.std(1)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--encoders", default="wavlm,campplus,whisper,xlsr,melstat")
    ap.add_argument("--seedvc-root", default=str(ROOT / "repos" / "seed-vc"))
    ap.add_argument("--noise-control", action="store_true", help="add pure-noise rows (negative control)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32, help="clips per encoder pass (equal-length clips of one utterance)")
    ap.add_argument("--verify", action="store_true", help="first batch: compare batched with single-clip embeddings")
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
    def audio(r):
        x = load_16k(r["wav"])
        if r["kind"] == "noise":
            x = (0.05 * rng.standard_normal(len(x))).astype("float32")
        return x
    # batches: the clips of one utterance (its step outputs, original and noise control) all have the source's length,
    # so they go through an encoder together without any padding -> identical numbers to one clip at a time
    by_utt = {}
    for i, r in enumerate(rows):
        by_utt.setdefault(r["utt_id"], []).append(i)
    batches = []
    for u, idx in by_utt.items():
        for j in range(0, len(idx), a.batch):
            batches.append(idx[j:j + a.batch])
    print(f"{len(batches)} batches of <= {a.batch} equal-length clips")
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
            n_done = 0
            for bi, idx in enumerate(batches):
                xs = [audio(rows[i]) for i in idx]
                groups = {}
                for i, x in zip(idx, xs):                       # guard: split by exact length if anything differs
                    groups.setdefault(len(x), []).append((i, x))
                for g in groups.values():
                    e = enc([x for _, x in g])
                    for (i, _), v in zip(g, e):
                        emb[i] = v
                if bi == 0 and a.verify:                        # batched vs one clip at a time, first batch
                    single = np.stack([enc([x])[0] for x in xs[:4]])
                    bat = emb[idx[:4]]
                    cos = [(single[j] @ bat[j]) / (np.linalg.norm(single[j]) * np.linalg.norm(bat[j]) + 1e-9) for j in range(len(single))]
                    print(f"  {name} verify: batched vs single, cosine {min(cos):.6f}..{max(cos):.6f}, max |diff| {np.abs(single - bat).max():.2e}")
                n_done += len(idx)
                if (bi + 1) % 100 == 0 or bi + 1 == len(batches):
                    el = time.time() - t0
                    print(f"  {name} [{n_done}/{len(rows)}] {el / 60:.1f} min, left ~{el / n_done * (len(rows) - n_done) / 60:.0f} min"
                          + (f"  GPU peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB" if dev == "cuda" else ""), flush=True)
            del enc; torch.cuda.empty_cache() if dev == "cuda" else None
        np.savez(fdir / f"{name}.npz", emb=emb)
        print(f"{name}: {emb.shape} in {(time.time() - t0) / 60:.1f} min -> {fdir / (name + '.npz')}")


if __name__ == "__main__":
    main()
