"""D7 analysis: per configuration, identity and content left in the output, next to the certificate term.

    python analyze.py --out OUT [--aspect1 ASPECT1_RUN] [--n-boot 200] [--epochs 200] [--quick]

With --aspect1, the probes are ALSO trained on the Aspect 1 step-30 outputs of the unchanged pipeline (all seeds,
every utterance except the sweep's test utterances: ~4,800 rows instead of 420) and applied to the sweep's test
outputs of each configuration (columns *_a1). This is the attacker who learned on normal anonymized speech; the bound
stays a held-out lower bound, with the sensitivity of Aspect 1 (the baseline row must reproduce its 1.2 nats / 19 %).

Per configuration (and on the originals): calibrated linear speaker probe on CAM++ and WavLM-SV embeddings (closed
set, utterance-disjoint split of the sweep's utterances; same estimators as Aspect 1), InfoNCE content bound on the
Whisper embedding against the Whisper embedding of the original, mean cosine similarity of the Whisper embeddings
(utility proxy), mean cosine similarity of the CAM++ embedding to the ORIGINAL (how much of the source voice is
left, attacker-free) and the certificate term from configs.json.
Writes OUT/results/sweep.csv, OUT/results/leak_vs_utility.png, OUT/results/summary.md.
"""
import argparse, csv, json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "aspect1_mi"))
from mi.estimators import classifier_bound, infonce_bound, split_by_group, speaker_bootstrap   # noqa: E402


def read_csv(p):
    rows = list(csv.DictReader(open(p, newline="", encoding="utf-8")))
    return [{k.strip(): (v or "").strip() for k, v in r.items()} for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True); ap.add_argument("--aspect1", default=None, help="Aspect 1 run dir with features/ (probes trained there)")
    ap.add_argument("--n-boot", type=int, default=200); ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--split-seed", type=int, default=0); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.n_boot, a.epochs = 20, 30
    out = Path(a.out); rdir = out / "results"; rdir.mkdir(exist_ok=True)
    rows = read_csv(out / "features" / "rows.csv")
    cfgs = json.load(open(out / "configs.json", encoding="utf-8"))
    cert = {c["name"]: c for c in cfgs["configs"]}
    E = {e: np.load(out / "features" / f"{e}.npz")["emb"] for e in ("campplus", "wavlm", "whisper")}
    config = np.array([r["config"] for r in rows]); utt = np.array([r["utt_id"] for r in rows]); S = np.array([r["src_spk"] for r in rows])
    spk = sorted(set(S)); sid = np.array([spk.index(s) for s in S])
    utts = sorted(set(utt)); tr_u, te_u = split_by_group(np.array(utts), 0.3, seed=a.split_seed)
    te_set = set(np.array(utts)[te_u].tolist()); is_te = np.array([u in te_set for u in utt])
    orig_idx = {u: i for i, u in enumerate(utt) if config[i] == "orig"}
    names = ["orig"] + [c["name"] for c in cfgs["configs"]]
    # probes trained on the Aspect 1 outputs (unchanged pipeline, step 30, all seeds), excluding the sweep's test utterances
    A1 = None
    if a.aspect1 and (Path(a.aspect1) / "features" / "rows.csv").exists():
        ra = read_csv(Path(a.aspect1) / "features" / "rows.csv")
        kind_a = np.array([r["kind"] for r in ra]); k_a = np.array([int(r["k"]) for r in ra]); utt_a = np.array([r["utt_id"] for r in ra]); S_a = np.array([r["src_spk"] for r in ra])
        Ea = {e: np.load(Path(a.aspect1) / "features" / f"{e}.npz")["emb"] for e in ("campplus", "wavlm", "whisper")}
        tr_a = np.flatnonzero((kind_a == "gen") & (k_a == 30) & np.array([u not in te_set for u in utt_a]))
        oi = {u: i for i, u in enumerate(utt_a) if kind_a[i] == "orig"}
        A1 = dict(tr=tr_a, sid=np.array([spk.index(x) if x in spk else -1 for x in S_a]), utt=utt_a, E=Ea, c=np.stack([Ea["whisper"][oi[u]] for u in utt_a[tr_a]]))
        assert (A1["sid"][tr_a] >= 0).all(), "Aspect 1 speakers differ from the sweep's"
        print(f"probes also trained on {len(tr_a)} Aspect 1 step-30 rows ({len(set(utt_a[tr_a]))} utterances, excluding the {len(te_set)} sweep test utterances)", flush=True)
    print(f"{len(utts)} utterances, {len(spk)} speakers (H(S) = {math.log(len(spk)):.3f}), {len(te_set)} test utterances; configurations: {names}", flush=True)
    res = []; summary = []
    def say(s):
        print(s, flush=True); summary.append(s)
    if A1:
        say("| config | gamma | sigma/tau | certificate (nats) | I(S;Y) CAM++ (A1-trained) | acc | I(S;Y) WavLM (A1-trained) | acc | I(C;Y) Whisper (A1-trained) | I(S;Y) CAM++ (self) | acc | whisper cos | CAM++ cos to source |")
        say("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    else:
        say(f"| config | gamma | sigma/tau | certificate (nats) | I(S;Y) CAM++ | acc | I(S;Y) WavLM | acc | I(C;Y) Whisper | whisper cos | CAM++ cos to source |")
        say("|---|---|---|---|---|---|---|---|---|---|---|")
    for name in names:
        m = config == name
        tr = np.flatnonzero(m & ~is_te); te = np.flatnonzero(m & is_te)
        r = dict(config=name, gamma=cert[name]["gamma"] if name in cert else 1.0, sigma_rel=cert[name]["sigma_rel"] if name in cert else 0.0,
                 basis=cert[name]["basis"] if name in cert else "orig", certificate_nats=cert[name]["certificate_nats"] if name in cert else float("nan"))
        for enc in ("campplus", "wavlm"):
            e = classifier_bound(E[enc][tr], sid[tr], E[enc][te], sid[te], probe="linear", epochs=a.epochs, groups_tr=utt[tr])
            lo, hi, _ = speaker_bootstrap(e.extra["ce_each"], S[te], n=a.n_boot)
            r[f"S_{enc}"] = float(e.value); r[f"S_{enc}_lo"] = float(e.h - hi); r[f"S_{enc}_hi"] = float(e.h - lo); r[f"acc_{enc}"] = float(e.extra["acc"])
        # content: InfoNCE between the Whisper embedding of the output and of the original of the same utterance
        c_orig = np.stack([E["whisper"][orig_idx[u]] for u in utt[m]])
        y = E["whisper"][m]; grp = utt[m]; te_m = is_te[m]
        e = infonce_bound(c_orig[~te_m], y[~te_m], c_orig[te_m], y[te_m], batch=512, epochs=50 if not a.quick else 10, groups_tr=grp[~te_m], groups_te=grp[te_m])
        r["C_whisper"] = float(e.value); r["C_cap"] = float(e.h)
        def cos(A, B):
            return float(np.mean(np.sum(A * B, 1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-9)))
        r["whisper_cos"] = cos(y, c_orig)
        r["campplus_cos_source"] = cos(E["campplus"][m], np.stack([E["campplus"][orig_idx[u]] for u in utt[m]]))
        r["n_tr"] = int(len(tr)); r["n_te"] = int(len(te))
        if A1:
            tra = A1["tr"]
            for enc in ("campplus", "wavlm"):
                e = classifier_bound(A1["E"][enc][tra], A1["sid"][tra], E[enc][te], sid[te], probe="linear", epochs=a.epochs, groups_tr=A1["utt"][tra])
                lo, hi, _ = speaker_bootstrap(e.extra["ce_each"], S[te], n=a.n_boot)
                r[f"S_{enc}_a1"] = float(e.value); r[f"S_{enc}_a1_lo"] = float(e.h - hi); r[f"S_{enc}_a1_hi"] = float(e.h - lo); r[f"acc_{enc}_a1"] = float(e.extra["acc"])
            e = infonce_bound(A1["c"], A1["E"]["whisper"][tra], c_orig[te_m], y[te_m], batch=512, epochs=50 if not a.quick else 10, groups_tr=A1["utt"][tra], groups_te=grp[te_m])
            r["C_whisper_a1"] = float(e.value); r["C_cap_a1"] = float(e.h)
            say(f"| {name} | {r['gamma']:g} | {r['sigma_rel']:g} | {r['certificate_nats']:.2f} | {r['S_campplus_a1']:.2f} [{r['S_campplus_a1_lo']:.2f}, {r['S_campplus_a1_hi']:.2f}] | {r['acc_campplus_a1'] * 100:.1f} % | "
                f"{r['S_wavlm_a1']:.2f} | {r['acc_wavlm_a1'] * 100:.1f} % | {r['C_whisper_a1']:.2f} / {r['C_cap_a1']:.2f} | {r['S_campplus']:.2f} | {r['acc_campplus'] * 100:.1f} % | {r['whisper_cos']:.3f} | {r['campplus_cos_source']:.3f} |")
        else:
            say(f"| {name} | {r['gamma']:g} | {r['sigma_rel']:g} | {r['certificate_nats']:.2f} | {r['S_campplus']:.2f} [{r['S_campplus_lo']:.2f}, {r['S_campplus_hi']:.2f}] | {r['acc_campplus'] * 100:.1f} % | "
                f"{r['S_wavlm']:.2f} | {r['acc_wavlm'] * 100:.1f} % | {r['C_whisper']:.2f} / {r['C_cap']:.2f} | {r['whisper_cos']:.3f} | {r['campplus_cos_source']:.3f} |")
        res.append(r)
    with open(rdir / "sweep.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0].keys())); w.writeheader(); w.writerows(res)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=120); fig.patch.set_facecolor("white")
        ax = axes[0]
        kS = "S_campplus_a1" if A1 else "S_campplus"; kC = "C_whisper_a1" if A1 else "C_whisper"
        for r in res:
            col = {"orig": "#1a1a19", "all": "#2a78d6", "lda": "#eb6834"}[r["basis"]]
            ax.scatter(r[kS], r[kC], color=col, s=28, zorder=3)
            ax.annotate(r["config"], (r[kS], r[kC]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("identity left: I(S; Y) CAM++ linear probe [nats]"); ax.set_ylabel("content kept: I(C; Y) Whisper InfoNCE [nats]")
        ax.set_title("Leakage vs utility per configuration", loc="left", fontsize=11); ax.grid(True, color="#e6e5e0")
        ax = axes[1]
        gam = sorted({r["gamma"] for r in res if r["basis"] == "all"}, reverse=True)
        for g, col in zip(gam, ["#2a78d6", "#2e9e6b", "#8b5cf6", "#c9a227"]):
            rr = sorted([r for r in res if r["basis"] == "all" and r["gamma"] == g], key=lambda r: r["sigma_rel"])
            ax.plot([r["sigma_rel"] for r in rr], [r[kS] for r in rr], "-o", color=col, markersize=4, label=f"gamma = {g:g}")
        o = next(r for r in res if r["config"] == "orig"); ax.axhline(o[kS], color="#1a1a19", linestyle=":", label="original audio")
        ax.set_xlabel("sigma / tau"); ax.set_ylabel("I(S; Y) CAM++ [nats]"); ax.set_title("Identity left vs noise, per contraction", loc="left", fontsize=11)
        ax.grid(True, color="#e6e5e0"); ax.legend(frameon=False, fontsize=8)
        for ax in axes:
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        fig.tight_layout(); fig.savefig(rdir / "leak_vs_utility.png")
    except Exception as e:
        print(f"(no plot: {e})")
    (rdir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"written: {rdir / 'sweep.csv'}, leak_vs_utility.png, summary.md")


if __name__ == "__main__":
    main()
