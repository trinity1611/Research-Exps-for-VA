"""Manifest + target pool from the DISPLACE-M conversations (the trajectory experiment's data), using the reference
diarization: every diarized speaker is a source speaker or a pool (target) speaker, never both.

  sources: recordings 1..N_src of the sorted id list -> for each of its two main speakers, up to --per-speaker
           single-speaker stretches of --min-sec .. --max-sec s (other speaker's overlaps removed, pauses joined as in
           trajectory/pair_experiment.py); utt_id = <rec>_<spk>_uNNN, sentence_id = utt_id (no two speakers read the
           same text in this corpus: content is continuous, see README)
  pool:    the remaining recordings -> one reference clip per speaker (--ref-sec s of their stretches)

    python make_manifest_displace.py --wav-dir data/wav --rttm-dir data/rttm --out data/aspect1_displace
                                     [--n-src 60] [--per-speaker 15] [--min-sec 4] [--max-sec 8] [--ref-sec 20] [--seed 0]
Writes <out>/sources/<spk>/uNNN.wav, <out>/pool/<spk>.wav, <out>/manifest.csv, <out>/pool.csv, <out>/manifest.log
"""
import argparse, csv, io, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "trajectory"))
from pair_experiment import read_rttm, speaker_turns, build_segments, cut_pieces, SR     # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav-dir", required=True); ap.add_argument("--rttm-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-src", type=int, default=60, help="recordings used as sources (the rest form the target pool)")
    ap.add_argument("--per-speaker", type=int, default=15); ap.add_argument("--min-sec", type=float, default=4.0); ap.add_argument("--max-sec", type=float, default=8.0)
    ap.add_argument("--ref-sec", type=float, default=20.0); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import soundfile as sf
    wav_dir, rttm_dir, out = Path(a.wav_dir), Path(a.rttm_dir), Path(a.out)
    (out / "sources").mkdir(parents=True, exist_ok=True); (out / "pool").mkdir(exist_ok=True)
    ids = [p.stem for p in sorted(wav_dir.glob("*.wav"))]
    rng = np.random.default_rng(a.seed)
    man, pool, log = [], [], []

    def load(fid):
        x, sr = sf.read(io.BytesIO((wav_dir / f"{fid}.wav").read_bytes()), dtype="float32", always_2d=True)   # one read call: NFS latency
        x = x.mean(axis=1)
        if sr != SR:
            import torch, torchaudio
            x = torchaudio.functional.resample(torch.from_numpy(x)[None], sr, SR)[0].numpy()
        return x

    for i, fid in enumerate(ids):
        rp = next(iter(sorted(rttm_dir.glob(f"{fid}*.rttm"))), None)
        if rp is None:
            continue
        rttm = read_rttm(rp)
        per = {}
        for s0, s1, l in rttm:
            per[l] = per.get(l, 0.0) + s1 - s0
        labels = sorted(per, key=per.get, reverse=True)[:2]
        x = None
        for l in labels:
            lab = l if l.lower().startswith("spk") else f"spk{l}"
            spk = f"{fid}_{lab}"
            turns = speaker_turns(rttm, l)
            if i < a.n_src:
                segs = build_segments(turns, a.min_sec, a.max_sec)
                segs = [s for s in segs if sum(b - a_ for a_, b in s) >= a.min_sec]
                if len(segs) < 3:
                    log.append(f"{spk}: only {len(segs)} segments, skipped"); continue
                pick = sorted(rng.choice(len(segs), min(a.per_speaker, len(segs)), replace=False))
                x = load(fid) if x is None else x
                (out / "sources" / spk).mkdir(exist_ok=True)
                for n, j in enumerate(pick, 1):
                    seg = segs[j]
                    p = out / "sources" / spk / f"u{n:03d}.wav"
                    sf.write(str(p), cut_pieces(x, seg), SR, subtype="PCM_16")
                    man.append([f"{spk}_u{n:03d}", spk, f"{spk}_u{n:03d}", str(p), ""])
                log.append(f"{spk}: {len(pick)} of {len(segs)} segments ({a.min_sec}-{a.max_sec} s)")
            else:
                ref, tot = [], 0.0
                for s0, s1 in turns:
                    ref.append((s0, s1)); tot += s1 - s0
                    if tot >= a.ref_sec:
                        break
                if tot < 5.0:
                    log.append(f"{spk}: only {tot:.1f} s of clean speech, not in the pool"); continue
                x = load(fid) if x is None else x
                p = out / "pool" / f"{spk}.wav"
                sf.write(str(p), cut_pieces(x, ref), SR, subtype="PCM_16")
                pool.append([spk, str(p), ""])
                log.append(f"{spk}: pool reference {tot:.1f} s")
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["utt_id", "src_spk", "sentence_id", "wav", "split"]); w.writerows(man)
    with open(out / "pool.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["tgt_spk", "wav", "gender"]); w.writerows(pool)
    (out / "manifest.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    n_spk = len({r[1] for r in man})
    print(f"{len(man)} source utterances from {n_spk} speakers ({a.n_src} recordings); pool {len(pool)} speakers ({len(ids) - a.n_src} recordings)")
    print(f"-> {out / 'manifest.csv'}, {out / 'pool.csv'}, {out / 'manifest.log'}")


if __name__ == "__main__":
    main()
