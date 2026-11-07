"""Condition-space audit, analysis stage (framework D4, D6, D10-context, D15, D8). CPU only; training-free apart
from the calibrated probes of aspect1_mi/mi (same estimators as Aspect 1, lower bounds in nats).

    python analyze.py --out OUT [--aspect1 ASPECT1_RUN] [--n-boot 200] [--quick]

D4  identity in the condition: calibrated speaker probes (linear, mlp; closed set, utterance-disjoint) on hbar, hstd,
    hdelta, their concatenation, and on the DiT condition mu (mean, std). Then the identity coordinate: LDA fitted on
    the TRAINING speakers of a speaker-disjoint split (84 of 120), evaluated on the held-out speakers (36): probes on
    U_S^T (hbar - mu) for k = 1..83 and on the residual hbar - P_k hbar. PCA directions as the comparison.
D6  identity in duration / rate only: probes on (dur_s, n_frames, pause_frac, chars_per_s).
D10 context-only floor: TF-IDF (char 1-3-grams and word unigrams) of the ASR transcript -> SVD-256 -> probe.
D15 intrinsic dimension (TwoNN, Levina-Bickel MLE), global PCA dims for 90/95 % variance, local PCA at anchors,
    tangent rotation between neighbouring anchors and between local tangents and the global LDA / PCA subspaces,
    within/between-speaker ratio R.
D8  attacker baselines from the Aspect 1 features (step 30 outputs): ignorant and lazy-informed cosine ASV with
    WavLM-SV and CAM++, EER and top-1 identification, plus the original-vs-original reference.
Writes OUT/results/{d4_probes.csv, d4_identity_vs_k.csv, d6_rate.csv, d10_context.csv, d15_geometry.csv, d8_asv.csv,
d4_identity_vs_k.png, summary.md}.
"""
import argparse, csv, json, math, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "aspect1_mi"))
from mi.estimators import classifier_bound, split_by_group, speaker_bootstrap, entropy_of_labels   # noqa: E402


def read_csv(p):
    rows = list(csv.DictReader(open(p, newline="", encoding="utf-8")))
    return [{k.strip(): (v or "").strip() for k, v in r.items()} for r in rows]


def probe(X, S, tr, te, utt, probe="linear", epochs=200, n_boot=200):
    """Calibrated classifier bound on rows tr -> te; returns dict with I_hat, ci, acc, H, CE."""
    X = np.asarray(X, np.float32)
    S = np.asarray(S)
    labels = {s: i for i, s in enumerate(sorted(set(S[tr]) | set(S[te])))}
    s_tr = np.array([labels[s] for s in S[tr]]); s_te = np.array([labels[s] for s in S[te]])
    e = classifier_bound(X[tr], s_tr, X[te], s_te, probe=probe, epochs=epochs, groups_tr=utt[tr])
    lo, hi, _ = speaker_bootstrap(e.extra["ce_each"], S[te], n=n_boot)
    return dict(I_hat=float(e.value), ci_lo=float(e.h - hi), ci_hi=float(e.h - lo), acc=float(e.extra["acc"]), H=float(e.h), CE=float(e.ce),
                K=len(labels), n_tr=int(len(tr)), n_te=int(len(te)))


def orthonormal(A):
    Q, _ = np.linalg.qr(A)
    return Q


def lda_directions(X, S):
    """LDA scalings (between/within generalized eigenvectors), shrinkage within-class covariance; columns ordered by eigenvalue."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(X, S)
    return orthonormal(lda.scalings_[:, : len(set(S)) - 1])


def principal_angles_deg(A, B):
    """Principal angles between the column spaces of A and B (orthonormal columns), in degrees."""
    s = np.linalg.svd(A.T @ B, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1)))


def twonn(X, frac=0.9):
    from sklearn.neighbors import NearestNeighbors
    d, _ = NearestNeighbors(n_neighbors=3).fit(X).kneighbors(X)
    mu = d[:, 2] / np.maximum(d[:, 1], 1e-12)
    mu = np.sort(mu[np.isfinite(mu) & (mu > 1)])
    n = len(mu); keep = int(frac * n)
    F = np.arange(1, n + 1) / n
    x = np.log(mu[:keep]); y = -np.log(1 - F[:keep])
    return float((x @ y) / (x @ x))


def mle_id(X, k1=10, k2=20):
    from sklearn.neighbors import NearestNeighbors
    d, _ = NearestNeighbors(n_neighbors=k2 + 1).fit(X).kneighbors(X)
    d = np.maximum(d[:, 1:], 1e-12)
    est = []
    for k in range(k1, k2 + 1):
        est.append(1.0 / np.mean(np.mean(np.log(d[:, k - 1:k] / d[:, : k - 1]), axis=1)))
    return float(np.mean(est))


def eer(scores_t, scores_n):
    """Equal error rate of target vs non-target score lists (higher = more similar)."""
    s = np.concatenate([scores_t, scores_n]); lab = np.concatenate([np.ones(len(scores_t)), np.zeros(len(scores_n))])
    order = np.argsort(-s); lab = lab[order]
    tp = np.cumsum(lab); fp = np.cumsum(1 - lab)
    frr = 1 - tp / len(scores_t); far = fp / len(scores_n)
    i = int(np.argmin(np.abs(frr - far)))
    return float((frr[i] + far[i]) / 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True); ap.add_argument("--aspect1", default=None, help="Aspect 1 run dir with features/ (for D8)")
    ap.add_argument("--n-boot", type=int, default=200); ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--split-seed", type=int, default=0); ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--ks", default="1,2,4,8,16,32,64,83"); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.n_boot, a.epochs = 20, 30
    out = Path(a.out); rdir = out / "results"; rdir.mkdir(exist_ok=True)
    F = np.load(out / "features.npz"); tab = read_csv(out / "table.csv")
    utt = np.array([r["utt_id"] for r in tab]); S = np.array([r["src_spk"] for r in tab])
    assert (F["utt_id"] == utt).all()
    hbar, hstd, hdelta = F["hbar"], F["hstd"], F["hdelta"]
    have_mu = np.isfinite(F["mu_mean"]).all(1)
    n = len(utt); spk = sorted(set(S)); K = len(spk)
    print(f"{n} utterances, {K} speakers (H(S) = {math.log(K):.3f} nats); mu available for {have_mu.sum()}; "
          f"transcripts for {sum(bool(r['text']) for r in tab)}", flush=True)
    tr, te = split_by_group(utt, a.test_frac, seed=a.split_seed)                     # closed set: utterance-disjoint
    summary = []
    def say(s):
        print(s, flush=True); summary.append(s)

    # ------------------------------------------------------------------ D4a: identity in the condition, closed set
    say("## D4 identity in the condition (closed set, 120 speakers)")
    d4 = []
    feats = {"hbar": hbar, "hstd": hstd, "hdelta": hdelta, "hbar+hstd+hdelta": np.concatenate([hbar, hstd, hdelta], 1)}
    if have_mu.all():
        feats["mu_mean"] = F["mu_mean"]; feats["mu_std"] = F["mu_std"]; feats["mu_mean+mu_std"] = np.concatenate([F["mu_mean"], F["mu_std"]], 1)
    for name, X in feats.items():
        for pr in ("linear", "mlp"):
            t0 = time.time()
            r = probe(X, S, tr, te, utt, probe=pr, epochs=a.epochs, n_boot=a.n_boot)
            d4.append(dict(features=name, dim=X.shape[1], probe=pr, **r))
            say(f"  {name:>18s} {pr:6s} dim {X.shape[1]:5d}: I_hat {r['I_hat']:5.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]  acc {r['acc'] * 100:5.1f} %  ({time.time() - t0:.0f} s)")
    with open(rdir / "d4_probes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(d4[0].keys())); w.writeheader(); w.writerows(d4)

    # ------------------------------------------------------------------ D4b: identity coordinate, speaker-disjoint LDA
    say("## D4 identity coordinate theta = U_S^T (hbar - mu): LDA on 84 training speakers, probes on the 36 held-out speakers")
    tr_s, te_s = split_by_group(S, a.test_frac, seed=a.split_seed)                   # speaker-disjoint
    mu0 = hbar[tr_s].mean(0)
    U_lda = lda_directions(hbar[tr_s] - mu0, S[tr_s])                                 # (1024, 83)
    Xc = hbar[te_s] - mu0; S_te = S[te_s]; utt_te = utt[te_s]
    tr2, te2 = split_by_group(utt_te, a.test_frac, seed=a.split_seed)                 # closed set inside the held-out speakers
    pca_U = np.linalg.svd(hbar[tr_s] - mu0, full_matrices=False)[2].T                 # (1024, min)
    rows = []
    base = probe(Xc, S_te, tr2, te2, utt_te, probe="linear", epochs=a.epochs, n_boot=a.n_boot)
    say(f"  full hbar on the held-out speakers (H = {base['H']:.2f}): I_hat {base['I_hat']:.2f} acc {base['acc'] * 100:.1f} %")
    rows.append(dict(basis="full", k=1024, part="all", **base))
    ks = [int(k) for k in a.ks.split(",") if int(k) <= U_lda.shape[1]]
    for basis, U in (("lda", U_lda), ("pca", pca_U[:, : U_lda.shape[1]])):
        for k in ks:
            Q = U[:, :k]
            coord = Xc @ Q; resid = Xc - coord @ Q.T
            r1 = probe(coord, S_te, tr2, te2, utt_te, probe="linear", epochs=a.epochs, n_boot=a.n_boot)
            r2 = probe(resid, S_te, tr2, te2, utt_te, probe="linear", epochs=a.epochs, n_boot=a.n_boot)
            rows.append(dict(basis=basis, k=k, part="coordinate", **r1)); rows.append(dict(basis=basis, k=k, part="residual", **r2))
            say(f"  {basis} k={k:3d}: coordinate I_hat {r1['I_hat']:5.2f} (acc {r1['acc'] * 100:4.1f} %)   residual I_hat {r2['I_hat']:5.2f} (acc {r2['acc'] * 100:4.1f} %)")
    with open(rdir / "d4_identity_vs_k.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ------------------------------------------------------------------ D6: duration / rate only
    say("## D6 identity from duration and rate only (closed set)")
    rate = np.array([[float(r["dur_s"]), float(r["n_frames"]), float(r["pause_frac"]), float(r["chars_per_s"])] for r in tab], np.float32)
    d6 = []
    for name, cols in (("dur_s", [0]), ("pause_frac", [2]), ("chars_per_s", [3]), ("all4", [0, 1, 2, 3])):
        for pr in ("linear", "mlp"):
            r = probe(rate[:, cols], S, tr, te, utt, probe=pr, epochs=a.epochs, n_boot=a.n_boot)
            d6.append(dict(features=name, probe=pr, **r))
            say(f"  {name:>12s} {pr:6s}: I_hat {r['I_hat']:5.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]  acc {r['acc'] * 100:5.1f} %")
    log_rate = np.log(np.maximum(rate[:, 3], 1e-3))
    say(f"  log chars/s: mean {log_rate.mean():.3f}, var {log_rate.var():.4f} (Var rho for the rate branch); pause frac mean {rate[:, 2].mean():.3f}")
    with open(rdir / "d6_rate.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(d6[0].keys())); w.writeheader(); w.writerows(d6)

    # ------------------------------------------------------------------ D10: context-only floor from the transcript
    texts = [r["text"] for r in tab]
    d10 = []
    if sum(bool(t) for t in texts) > 0.9 * n:
        say("## D10 context-only floor: speaker from the ASR transcript (TF-IDF -> SVD-256 -> probe, closed set)")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        for name, vec in (("char1-3", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), sublinear_tf=True, min_df=2)),
                          ("word1", TfidfVectorizer(analyzer="word", ngram_range=(1, 1), sublinear_tf=True, min_df=2))):
            Xtr = vec.fit_transform([texts[i] for i in tr]); Xall = vec.transform(texts)
            svd = TruncatedSVD(n_components=min(256, Xtr.shape[1] - 1), random_state=0).fit(Xtr)
            Z = svd.transform(Xall).astype(np.float32)
            for pr in ("linear", "mlp"):
                r = probe(Z, S, tr, te, utt, probe=pr, epochs=a.epochs, n_boot=a.n_boot)
                d10.append(dict(features=name, vocab=Xtr.shape[1], probe=pr, **r))
                say(f"  {name:>8s} {pr:6s} (vocab {Xtr.shape[1]}): I_hat {r['I_hat']:5.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]  acc {r['acc'] * 100:5.1f} %")
        with open(rdir / "d10_context.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(d10[0].keys())); w.writeheader(); w.writerows(d10)
    else:
        say("## D10 skipped: no transcripts (run extract.py without --no-asr)")

    # ------------------------------------------------------------------ D15: intrinsic dimension and curvature of hbar
    say("## D15 intrinsic dimension and curvature of the XLS-R utterance means")
    Xs = (hbar - hbar.mean(0)) / (hbar.std(0) + 1e-6)
    sv = np.linalg.svd(Xs - Xs.mean(0), compute_uv=False); ev = sv ** 2 / (sv ** 2).sum(); cum = np.cumsum(ev)
    d90, d95 = int(np.searchsorted(cum, 0.90) + 1), int(np.searchsorted(cum, 0.95) + 1)
    id_two, id_mle = twonn(Xs), mle_id(Xs)
    means = np.stack([Xs[S == s].mean(0) for s in spk])
    id_two_spk = twonn(means) if K >= 30 else float("nan")
    # within / between speaker ratio on cosine distance
    Xn = Xs / np.linalg.norm(Xs, axis=1, keepdims=True)
    G = 1 - Xn @ Xn.T; same = S[:, None] == S[None, :]; np.fill_diagonal(same, False)
    off = ~np.eye(n, dtype=bool)
    Dw = G[same].mean(); Db = G[off & ~same].mean(); R = Db / Dw
    # local PCA at anchors
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(0)
    n_anchor = min(150, n); kn = min(50, n - 1); d_loc = 10
    anchors = rng.choice(n, n_anchor, replace=False)
    nn = NearestNeighbors(n_neighbors=kn + 1).fit(Xs)
    _, nb = nn.kneighbors(Xs[anchors])
    tangents, loc_dims = [], []
    for j in range(n_anchor):
        P = Xs[nb[j, 1:]]; P = P - P.mean(0)
        U, s, _ = np.linalg.svd(P, full_matrices=False)
        e = s ** 2 / (s ** 2).sum(); loc_dims.append(int(np.searchsorted(np.cumsum(e), 0.90) + 1))
        tangents.append(np.linalg.svd(P, full_matrices=False)[2][:d_loc].T)   # (1024, d_loc)
    # rotation between neighbouring anchors (nearest anchor in the sample)
    A = Xs[anchors]; da = ((A[:, None, :] - A[None, :, :]) ** 2).sum(-1); np.fill_diagonal(da, np.inf)
    nbr = da.argmin(1)
    rot = np.array([principal_angles_deg(tangents[j], tangents[nbr[j]]).mean() for j in range(n_anchor)])
    # local tangent vs global subspaces (standardized space: refit on Xs for comparability)
    U_lda_s = lda_directions(Xs[tr_s], S[tr_s])[:, :d_loc]
    U_pca_s = np.linalg.svd(Xs - Xs.mean(0), full_matrices=False)[2][:d_loc].T
    ang_lda = np.array([principal_angles_deg(tangents[j], U_lda_s).mean() for j in range(n_anchor)])
    ang_pca = np.array([principal_angles_deg(tangents[j], U_pca_s).mean() for j in range(n_anchor)])
    rand_ref = np.mean([principal_angles_deg(orthonormal(rng.standard_normal((1024, d_loc))), U_pca_s).mean() for _ in range(20)])
    d15 = dict(n=n, dim=1024, pca_dims_90=d90, pca_dims_95=d95, twonn_id=id_two, mle_id_k10_20=id_mle, twonn_id_speaker_means=id_two_spk,
               within_between_ratio_R=float(R), local_dim90_median=float(np.median(loc_dims)), local_dim90_p10=float(np.percentile(loc_dims, 10)),
               local_dim90_p90=float(np.percentile(loc_dims, 90)), tangent_rotation_neighbours_deg_mean=float(rot.mean()),
               tangent_rotation_neighbours_deg_p90=float(np.percentile(rot, 90)), tangent_vs_global_lda_deg_mean=float(ang_lda.mean()),
               tangent_vs_global_pca_deg_mean=float(ang_pca.mean()), random_subspace_vs_pca_deg=float(rand_ref), d_loc=d_loc, k_neighbours=kn, n_anchors=n_anchor)
    for k, v in d15.items():
        say(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    with open(rdir / "d15_geometry.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(d15.keys())); w.writeheader(); w.writerow(d15)

    # ------------------------------------------------------------------ D8: attacker baselines from the Aspect 1 features
    d8 = []
    if a.aspect1 and (Path(a.aspect1) / "features" / "rows.csv").exists():
        say("## D8 cosine ASV on the step-30 outputs (closed-set split of Aspect 1; enrol = speaker centroid of the training utterances)")
        rows_a = read_csv(Path(a.aspect1) / "features" / "rows.csv")
        kind = np.array([r["kind"] for r in rows_a]); k_of = np.array([int(r["k"]) for r in rows_a])
        utt_a = np.array([r["utt_id"] for r in rows_a]); S_a = np.array([r["src_spk"] for r in rows_a])
        te_utts = set(utt[te].tolist())
        is_te = np.array([u in te_utts for u in utt_a])
        for enc in ("wavlm", "campplus"):
            fp = Path(a.aspect1) / "features" / f"{enc}.npz"
            if not fp.exists():
                continue
            E = np.load(fp)["emb"]; E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
            orig, gen30 = kind == "orig", (kind == "gen") & (k_of == 30)
            spk_list = sorted(set(S_a))
            def centroids(mask):
                C = np.stack([E[mask & (S_a == s)].mean(0) for s in spk_list]); return C / np.linalg.norm(C, axis=1, keepdims=True)
            for name, enrol_mask, trial_mask in (("orig->orig (reference)", orig & ~is_te, orig & is_te),
                                                 ("ignorant: enrol orig, trial anon", orig & ~is_te, gen30 & is_te),
                                                 ("lazy-informed: enrol anon, trial anon", gen30 & ~is_te, gen30 & is_te)):
                C = centroids(enrol_mask); T = E[trial_mask]; lab = np.array([spk_list.index(s) for s in S_a[trial_mask]])
                sc = T @ C.T
                is_tgt = np.eye(len(spk_list), dtype=bool)[lab]                 # (trials, speakers)
                tgt, non = sc[is_tgt], sc[~is_tgt]
                e = eer(tgt, non); top1 = float((sc.argmax(1) == lab).mean())
                d8.append(dict(encoder=enc, attack=name, n_trials=int(len(lab)), n_speakers=len(spk_list), eer=e, top1=top1))
                say(f"  {enc:8s} {name:38s}: EER {e * 100:5.1f} %  top-1 {top1 * 100:5.1f} %  ({len(lab)} trials, {len(spk_list)} speakers)")
        if d8:
            with open(rdir / "d8_asv.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(d8[0].keys())); w.writeheader(); w.writerows(d8)
    else:
        say("## D8 skipped: no Aspect 1 features (--aspect1)")

    # ------------------------------------------------------------------ plot + summary
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.4), dpi=120); fig.patch.set_facecolor("white")
        for basis, col in (("lda", "#2a78d6"), ("pca", "#eb6834")):
            for part, ls in (("coordinate", "-"), ("residual", "--")):
                rr = [r for r in rows if r["basis"] == basis and r["part"] == part]
                ax.plot([r["k"] for r in rr], [r["I_hat"] for r in rr], ls, color=col, marker="o", markersize=3.5, label=f"{basis} {part}")
        ax.axhline(base["I_hat"], color="#7a7a73", linestyle=":", label="full hbar (1024)")
        ax.set_xscale("log", base=2); ax.set_xlabel("k (dimension of the identity coordinate)"); ax.set_ylabel("I_hat(S; .)  [nats, 36 held-out speakers]")
        ax.set_title("Identity in the coordinate vs in the residual", loc="left", fontsize=11); ax.grid(True, color="#e6e5e0"); ax.legend(frameon=False, fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout(); fig.savefig(rdir / "d4_identity_vs_k.png")
    except Exception as e:
        print(f"(no plot: {e})")
    (rdir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"written: {rdir}")


if __name__ == "__main__":
    main()
