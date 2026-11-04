"""Per-step MI lower bounds (spec sections 5, 8): I(S;Y_t), I(S;Y_t|T), I(S;Y_t|C), I(C;Y_t) for every encoder and
saved step, with ceilings on the original audio, permutation nulls, speaker-level bootstrap CIs, and plots.

    python analyze.py --out <out> [--encoders wavlm,campplus,whisper,xlsr,melstat] [--n-null 100] [--n-boot 200] [--quick]

Splits (fixed by --split-seed, whole utterances with all their seeds on one side):
  closed-set (S classifier):   utterance-disjoint, every speaker on both sides
  open-set  (InfoNCE, KSG, C): speaker-disjoint (30 % of the speakers held out)
Estimators: classifier bound (linear and MLP probes), InfoNCE (separable critic, N = --batch), KSG on PCA-10.
Continuous s for InfoNCE/KSG = WavLM-SV embedding of the ORIGINAL utterance.
Output: <out>/results/results.csv (+ .parquet): t, k, kind, encoder, estimator, probe, target, cond, I_hat, ci_lo, ci_hi,
        null_mean, null_sd, null_p, ceiling, frac, H, CE, N, acc;  plots mi_vs_t_<target>.png, leak_vs_util.png
"""
import argparse, csv, json, math, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mi.estimators import (classifier_bound, infonce_bound, ksg, pca, permutation_null, p_value, speaker_bootstrap,   # noqa: E402
                           split_by_group, assert_no_group_overlap, entropy_of_labels)

SPEAKER_ENC = ("wavlm", "campplus")
CONTENT_ENC = ("whisper", "xlsr")


def read_csv(p):
    rows = list(csv.DictReader(open(p, newline="", encoding="utf-8")))
    return [{k.strip(): (v or "").strip() for k, v in r.items()} for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True); ap.add_argument("--encoders", default="wavlm,campplus,whisper,xlsr,melstat")
    ap.add_argument("--n-null", type=int, default=100); ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--batch", type=int, default=512, help="InfoNCE N"); ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--test-frac", type=float, default=0.3); ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--quick", action="store_true", help="few epochs / nulls, for plumbing tests")
    a = ap.parse_args()
    if a.quick:
        a.n_null, a.n_boot, a.epochs = 5, 20, 30
    out = Path(a.out); rdir = out / "results"; rdir.mkdir(exist_ok=True)
    rows = read_csv(out / "features" / "rows.csv")
    feats = {e: np.load(out / "features" / f"{e}.npz")["emb"] for e in a.encoders.split(",")}
    kind = np.array([r["kind"] for r in rows]); k_of = np.array([int(r["k"]) for r in rows])
    utt = np.array([r["utt_id"] for r in rows])
    spk_names = sorted({r["src_spk"] for r in rows}); sent_names = sorted({r["sentence_id"] for r in rows})
    tgt_names = sorted({r["target_spk"] for r in rows if r["target_spk"]})
    S = np.array([spk_names.index(r["src_spk"]) for r in rows]); C = np.array([sent_names.index(r["sentence_id"]) for r in rows])
    T = np.array([tgt_names.index(r["target_spk"]) if r["target_spk"] else -1 for r in rows])
    steps = sorted(set(k_of[kind == "gen"].tolist()))
    tmax = max(steps)
    print(f"{len(rows)} rows; {len(spk_names)} speakers (H(S) = {math.log(len(spk_names)):.3f} nats), {len(sent_names)} sentences "
          f"(H(C) = {math.log(len(sent_names)):.3f}), {len(tgt_names)} targets; steps {steps}")

    # the continuous speaker variable s = WavLM-SV embedding of the ORIGINAL utterance
    if "wavlm" not in feats:
        feats["wavlm"] = np.load(out / "features" / "wavlm.npz")["emb"]
    s_orig = {u: feats["wavlm"][i] for i, u in enumerate(utt) if kind[i] == "orig"}
    s_cont = np.stack([s_orig[u] for u in utt])
    # continuous content variable c = Whisper-encoder embedding of the ORIGINAL utterance (used when no two speakers
    # read the same sentence, i.e. sentence_id is unique per utterance, as in conversational corpora)
    if "whisper" not in feats:
        feats["whisper"] = np.load(out / "features" / "whisper.npz")["emb"]
    c_orig = {u: feats["whisper"][i] for i, u in enumerate(utt) if kind[i] == "orig"}
    c_cont = np.stack([c_orig[u] for u in utt])
    discrete_c = len(sent_names) < len(set(utt.tolist()))
    print("content C:", "discrete sentence ids (classifier bound)" if discrete_c else "unique per utterance -> continuous (InfoNCE / KSG on Whisper features of the original)")

    # splits on the ORIGINAL rows' utterances / speakers, applied to every kind by utt_id
    orig_idx = np.flatnonzero(kind == "orig")
    utt_u = utt[orig_idx]; spk_u = S[orig_idx]
    tr_u, te_u = split_by_group(utt_u, a.test_frac, seed=a.split_seed)              # utterance-disjoint
    test_utts_closed = set(utt_u[te_u].tolist())
    tr_s, te_s = split_by_group(spk_u, a.test_frac, seed=a.split_seed)              # speaker-disjoint
    test_spk_open = set(spk_u[te_s].tolist())
    assert_no_group_overlap(spk_u[tr_s], spk_u[te_s])
    closed_te = np.array([u in test_utts_closed for u in utt]); open_te = np.array([s in test_spk_open for s in S])
    print(f"closed-set split: {len(test_utts_closed)} test utterances; open-set split: {len(test_spk_open)} test speakers")

    results = []
    def add(**kw):
        results.append(kw); print("   ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in kw.items() if k in ("kind", "k", "encoder", "estimator", "probe", "target", "cond", "I_hat", "ci_lo", "ci_hi", "null_p", "frac")})

    def subsets(kd, k):
        return np.flatnonzero((kind == kd) & (k_of == k)) if kd == "gen" else np.flatnonzero(kind == kd)

    t0 = time.time()
    conditions = [("orig", -1), ("noise", -1)] + [("gen", k) for k in steps]
    conditions = [c for c in conditions if (kind == c[0]).any()]
    for enc, F in feats.items():
        for kd, k in conditions:
            idx = subsets(kd, k)
            if enc == "melstat" and kd != "gen":
                continue
            if len(idx) == 0 or np.isnan(F[idx]).any():
                continue
            t = (k / tmax) if kd == "gen" else float("nan")
            base = dict(kind=kd, k=k, t=t, encoder=enc)
            # ---- discrete S, closed-set classifier, cond None / T / C
            tr = idx[~closed_te[idx]]; te = idx[closed_te[idx]]
            for cond in ("None", "T", "C"):
                if cond == "T" and kd != "gen":
                    continue
                if cond == "C" and not discrete_c:
                    continue
                ct = ce = None
                if cond == "T":
                    oh = np.eye(len(tgt_names), dtype=np.float32)[T]; ct, ce = oh[tr], oh[te]
                elif cond == "C":
                    oh = np.eye(len(sent_names), dtype=np.float32)[C]; ct, ce = oh[tr], oh[te]
                for probe in ("linear", "mlp"):
                    e = classifier_bound(F[tr], S[tr], F[te], S[te], probe=probe, cond_tr=ct, cond_te=ce, epochs=a.epochs)
                    lo, hi, _ = speaker_bootstrap(e.extra["ce_each"], S[te], n=a.n_boot)
                    rec = dict(base, estimator="classifier", probe=probe, target="S", cond=cond, I_hat=e.value, ci_lo=e.h - hi, ci_hi=e.h - lo,
                               H=e.h, CE=e.ce, N=len(te), acc=e.extra["acc"], null_mean=float("nan"), null_sd=float("nan"), null_p=float("nan"))
                    if probe == "linear" and cond == "None" and a.n_null > 0:
                        nl = permutation_null(lambda l1, l2: classifier_bound(F[tr], l1, F[te], l2, probe="linear", epochs=max(10, a.epochs // 4)), S[tr], S[te], n=a.n_null)
                        rec.update(null_mean=nl["null_mean"], null_sd=nl["null_sd"], null_p=p_value(e.value, nl["null"]))
                    add(**rec)
            # ---- discrete C, speaker-disjoint classifier
            tr = idx[~open_te[idx]]; te = idx[open_te[idx]]
            for probe in (("linear", "mlp") if discrete_c else ()):
                e = classifier_bound(F[tr], C[tr], F[te], C[te], probe=probe, epochs=a.epochs)
                lo, hi, _ = speaker_bootstrap(e.extra["ce_each"], S[te], n=a.n_boot)
                rec = dict(base, estimator="classifier", probe=probe, target="C", cond="None", I_hat=e.value, ci_lo=e.h - hi, ci_hi=e.h - lo,
                           H=e.h, CE=e.ce, N=len(te), acc=e.extra["acc"], null_mean=float("nan"), null_sd=float("nan"), null_p=float("nan"))
                if probe == "linear" and a.n_null > 0:
                    nl = permutation_null(lambda l1, l2: classifier_bound(F[tr], l1, F[te], l2, probe="linear", epochs=max(10, a.epochs // 4)), C[tr], C[te], n=a.n_null)
                    rec.update(null_mean=nl["null_mean"], null_sd=nl["null_sd"], null_p=p_value(e.value, nl["null"]))
                add(**rec)
            # ---- continuous S (and continuous C when sentences are unique), open-set InfoNCE and KSG(PCA-10)
            if len(te) >= 4 and len(tr) >= 4:                                      # (real runs have thousands; 4 keeps plumbing tests alive)
                for target, V in (("S", s_cont),) + ((("C", c_cont),) if not discrete_c else ()):
                    e = infonce_bound(V[tr], F[tr], V[te], F[te], batch=min(a.batch, len(te)), epochs=max(5, a.epochs // 4))
                    rec = dict(base, estimator="infonce", probe="separable", target=target, cond="None", I_hat=e.value, ci_lo=float("nan"), ci_hi=float("nan"),
                               H=e.h, CE=e.ce, N=e.extra["N"], acc=float("nan"), null_mean=float("nan"), null_sd=float("nan"), null_p=float("nan"))
                    if a.n_null > 0:
                        nn_ = max(3, a.n_null // 10)
                        rng = np.random.default_rng(1)
                        null = [infonce_bound(V[tr][rng.permutation(len(tr))], F[tr], V[te][rng.permutation(len(te))], F[te], batch=min(a.batch, len(te)), epochs=max(3, a.epochs // 10)).value for _ in range(nn_)]
                        rec.update(null_mean=float(np.mean(null)), null_sd=float(np.std(null)), null_p=p_value(e.value, null))
                    add(**rec)
                    ptr, pte = pca(F[tr], F[te], dims=10); vtr, vte = pca(V[tr], V[te], dims=10)
                    e = ksg(vte, pte, k=3)
                    add(**dict(base, estimator="ksg", probe="pca10", target=target, cond="None", I_hat=e.value, ci_lo=float("nan"), ci_hi=float("nan"),
                               H=float("nan"), CE=float("nan"), N=len(te), acc=float("nan"), null_mean=float("nan"), null_sd=float("nan"), null_p=float("nan")))
        print(f"  {enc} done, {(time.time() - t0) / 60:.1f} min", flush=True)

    # ---- ceilings (same encoder / estimator / probe / target / cond on the ORIGINAL audio) and fractions
    ceil = {(r["encoder"], r["estimator"], r["probe"], r["target"], r["cond"]): r["I_hat"] for r in results if r["kind"] == "orig"}
    for r in results:
        key = (r["encoder"], r["estimator"], r["probe"], r["target"], "None" if r["cond"] == "T" else r["cond"])
        c = ceil.get(key, float("nan"))
        r["ceiling"] = c; r["frac"] = r["I_hat"] / c if c and c > 0 else float("nan")
        r["exceeds_ceiling"] = bool(r["kind"] == "gen" and c == c and r["I_hat"] > c + max(0.05, (r["ci_hi"] - r["I_hat"]) if r["ci_hi"] == r["ci_hi"] else 0.05))
    cols = ["t", "k", "kind", "encoder", "estimator", "probe", "target", "cond", "I_hat", "ci_lo", "ci_hi", "null_mean", "null_sd", "null_p", "ceiling", "frac", "exceeds_ceiling", "H", "CE", "N", "acc"]
    with open(rdir / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in results:
            w.writerow({c: r.get(c, "") for c in cols})
    try:
        import pandas as pd
        pd.DataFrame(results)[cols].to_parquet(rdir / "results.parquet", index=False)
    except Exception as e:
        print(f"(no parquet: {e})")
    flagged = [r for r in results if r.get("exceeds_ceiling")]
    print(f"\n{len(results)} estimates -> {rdir / 'results.csv'};  {len(flagged)} exceed their ceiling (estimator inconsistency to inspect)")
    json.dump({"speakers": spk_names, "sentences": sent_names, "targets": tgt_names, "steps": steps, "closed_test_utts": sorted(test_utts_closed),
               "open_test_speakers": [spk_names[i] for i in sorted(test_spk_open)], "args": vars(a),
               "Y_t": "B: xhat1_t -> vocoder -> waveform -> encoders (primary); A: melstat of raw x_t (t = 0 sanity check)",
               "note": "all values are LOWER BOUNDS in nats; InfoNCE capped at log N"}, open(rdir / "design.json", "w"), indent=1)
    plots(results, steps, tmax, rdir)


def plots(results, steps, tmax, rdir):
    try:
        import matplotlib
        matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(no plots: {e})"); return
    cols = {"wavlm": "#2a78d6", "campplus": "#eb6834", "whisper": "#2e9e6b", "xlsr": "#8b5cf6", "melstat": "#7a7a73"}
    ts = [k / tmax for k in steps]
    for target in ("S", "C"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=120); fig.patch.set_facecolor("white")
        for ax, probe in zip(axes, ("linear", "mlp")):
            for enc in cols:
                rs = {r["k"]: r for r in results if r["kind"] == "gen" and r["encoder"] == enc and r["estimator"] == "classifier" and r["probe"] == probe and r["target"] == target and r["cond"] == "None"}
                if not rs and probe == "linear":
                    rs = {r["k"]: r for r in results if r["kind"] == "gen" and r["encoder"] == enc and r["estimator"] == "infonce" and r["target"] == target}
                    ax.set_title("InfoNCE bound (no shared sentences: C continuous)", loc="left", fontsize=10)
                elif not rs and probe == "mlp":
                    rs = {r["k"]: r for r in results if r["kind"] == "gen" and r["encoder"] == enc and r["estimator"] == "ksg" and r["target"] == target}
                    ax.set_title("KSG on PCA-10 (not a bound)", loc="left", fontsize=10)
                if not rs:
                    continue
                y = [rs[k]["I_hat"] if k in rs else np.nan for k in steps]
                lo = [rs[k]["ci_lo"] if k in rs else np.nan for k in steps]; hi = [rs[k]["ci_hi"] if k in rs else np.nan for k in steps]
                ax.fill_between(ts, lo, hi, color=cols[enc], alpha=0.15, linewidth=0)
                ax.plot(ts, y, color=cols[enc], marker="o", markersize=3.5, linewidth=1.8, label=enc)
                c = next((r["ceiling"] for r in rs.values() if r["ceiling"] == r["ceiling"]), None)
                if c is not None:
                    ax.axhline(c, color=cols[enc], linestyle=":", linewidth=1)
                nm = [rs[k]["null_mean"] for k in steps if k in rs and rs[k]["null_mean"] == rs[k]["null_mean"]]
                if nm:
                    ax.axhspan(min(nm) - 0.02, max(nm) + 0.02, color="#c3c2b7", alpha=0.3, linewidth=0)
            ax.set_xlabel("flow time t  (0 = noise, 1 = output)"); ax.set_ylabel(f"I_hat({target}; Y_t)  [nats, lower bound]")
            if not ax.get_title(loc="left"):
                ax.set_title(f"{probe} probe; dotted = ceiling on original audio; grey = permutation null", loc="left", fontsize=10)
            ax.grid(True, color="#e6e5e0", linewidth=0.8); ax.set_axisbelow(True)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(frameon=False, fontsize=9)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        fig.suptitle(f"{'Source-speaker' if target == 'S' else 'Content'} information in the one-step estimate at each denoising step", x=0.01, ha="left", fontsize=12)
        fig.tight_layout(); fig.savefig(rdir / f"mi_vs_t_{target}.png"); plt.close(fig)
    # leakage vs utility across t (linear probe, wavlm vs whisper fractions)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120); fig.patch.set_facecolor("white")
    xs, ys, labels = [], [], []
    for k in steps:
        rS = next((r for r in results if r["kind"] == "gen" and r["k"] == k and r["encoder"] == "wavlm" and r["estimator"] == "classifier" and r["probe"] == "linear" and r["target"] == "S" and r["cond"] == "None"), None)
        rC = next((r for r in results if r["kind"] == "gen" and r["k"] == k and r["encoder"] == "whisper" and r["estimator"] in ("classifier", "infonce") and r["probe"] in ("linear", "separable") and r["target"] == "C" and r["cond"] == "None"), None)
        if rS and rC:
            xs.append(rS["frac"]); ys.append(rC["frac"]); labels.append(f"t={k / tmax:.2f}")
    if xs:
        ax.plot(xs, ys, color="#2a78d6", marker="o", linewidth=1.5)
        for x, y, l in zip(xs, ys, labels):
            ax.annotate(l, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("leak_frac_S = I(S;Y_t) / I(S;X)   (WavLM-SV, linear)"); ax.set_ylabel("util_frac_C = I(C;Y_t) / I(C;X)   (Whisper, linear)")
    ax.set_title("Leakage vs utility across denoising steps", loc="left", fontsize=11)
    ax.grid(True, color="#e6e5e0", linewidth=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(rdir / "leak_vs_util.png"); plt.close(fig)
    print(f"plots: {rdir / 'mi_vs_t_S.png'}, mi_vs_t_C.png, leak_vs_util.png")


if __name__ == "__main__":
    main()
