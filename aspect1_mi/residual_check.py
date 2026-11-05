"""Is the speaker information left in the outputs the voice, or something a content encoder also sees?

    python aspect1_mi/residual_check.py --out <run dir with features/>  [--steps 0,6,30]

For the original audio and for the one-step estimates at the given steps: linear speaker probe (closed-set split as
in analyze.py) on CAM++ and WavLM-SV, and on the same embeddings after removing everything a ridge regression on the
Whisper embedding of the SAME clip explains (fitted on training rows only). The residual holds the speaker
information the content encoder does not carry. Prints I_hat (nats), accuracy and the R^2 of the regression.
Result on DISPLACE (2026-11-05): originals keep 1.86 nats (51 %) in the CAM++ residual; outputs keep 0.0 at every step.
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mi.estimators import classifier_bound, split_by_group  # noqa: E402


def z(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-6)


def resid(Y, X, trmask, lam=10.0):
    """Y - X W with W = ridge fitted on the training rows; also the training R^2."""
    Xa = np.concatenate([X, np.ones((len(X), 1))], 1)
    A = Xa[trmask]
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y[trmask])
    R = Y - Xa @ W
    return R, 1 - (R[trmask] ** 2).sum() / ((Y[trmask] - Y[trmask].mean(0)) ** 2).sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True); ap.add_argument("--steps", default="0,6,30")
    ap.add_argument("--encoders", default="campplus,wavlm"); ap.add_argument("--split-seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    rows = list(csv.DictReader(open(out / "features" / "rows.csv", newline="", encoding="utf-8")))
    kind = np.array([r["kind"] for r in rows]); k_of = np.array([int(r["k"]) for r in rows]); utt = np.array([r["utt_id"] for r in rows])
    spk = sorted({r["src_spk"] for r in rows}); S = np.array([spk.index(r["src_spk"]) for r in rows])
    encs = a.encoders.split(",")
    F = {e: np.load(out / "features" / f"{e}.npz")["emb"] for e in encs + ["whisper"]}
    orig_idx = np.flatnonzero(kind == "orig")
    _, te_u = split_by_group(utt[orig_idx], 0.3, seed=a.split_seed)
    te_set = set(utt[orig_idx][te_u].tolist()); closed_te = np.array([u in te_set for u in utt])
    conds = [("orig", kind == "orig")] + [(f"gen k={k}", (kind == "gen") & (k_of == int(k))) for k in a.steps.split(",")]
    print(f"{'rows':>10s} {'features':>26s} {'I_hat':>6s} {'acc':>6s}  R2(whisper -> enc)")
    for name, sel in conds:
        tr = sel & ~closed_te; te = sel & closed_te; trm = ~closed_te[sel]
        W = z(F["whisper"][sel])
        for enc in encs:
            Y = z(F[enc][sel]); R, r2 = resid(Y, W, trm)
            for label, X, rr in ((enc, Y, float("nan")), (f"{enc} | whisper removed", R, r2)):
                e = classifier_bound(X[trm].astype(np.float32), S[tr], X[~trm].astype(np.float32), S[te], probe="linear", epochs=200, groups_tr=utt[tr])
                print(f"{name:>10s} {label:>26s} {e.value:6.2f} {e.extra['acc'] * 100:5.1f}%  {rr:5.2f}", flush=True)


if __name__ == "__main__":
    main()
