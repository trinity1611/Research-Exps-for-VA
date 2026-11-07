"""D7 analysis: per configuration, identity and content left in the output, next to the certificate term.

    python analyze.py --out OUT [--n-boot 200] [--epochs 200] [--quick]

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
    ap.add_argument("--out", required=True); ap.add_argument("--n-boot", type=int, default=200); ap.add_argument("--epochs", type=int, default=200)
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
    print(f"{len(utts)} utterances, {len(spk)} speakers (H(S) = {math.log(len(spk)):.3f}), {len(te_set)} test utterances; configurations: {names}", flush=True)
    res = []; summary = []
    def say(s):
        print(s, flush=True); summary.append(s)
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
        res.append(r)
        say(f"| {name} | {r['gamma']:g} | {r['sigma_rel']:g} | {r['certificate_nats']:.2f} | {r['S_campplus']:.2f} [{r['S_campplus_lo']:.2f}, {r['S_campplus_hi']:.2f}] | {r['acc_campplus'] * 100:.1f} % | "
            f"{r['S_wavlm']:.2f} | {r['acc_wavlm'] * 100:.1f} % | {r['C_whisper']:.2f} / {r['C_cap']:.2f} | {r['whisper_cos']:.3f} | {r['campplus_cos_source']:.3f} |")
    with open(rdir / "sweep.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0].keys())); w.writeheader(); w.writerows(res)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=120); fig.patch.set_facecolor("white")
        ax = axes[0]
        for r in res:
            col = {"orig": "#1a1a19", "all": "#2a78d6", "lda": "#eb6834"}[r["basis"]]
            ax.scatter(r["S_campplus"], r["C_whisper"], color=col, s=28, zorder=3)
            ax.annotate(r["config"], (r["S_campplus"], r["C_whisper"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("identity left: I(S; Y) CAM++ linear probe [nats]"); ax.set_ylabel("content kept: I(C; Y) Whisper InfoNCE [nats]")
        ax.set_title("Leakage vs utility per configuration", loc="left", fontsize=11); ax.grid(True, color="#e6e5e0")
        ax = axes[1]
        gam = sorted({r["gamma"] for r in res if r["basis"] == "all"}, reverse=True)
        for g, col in zip(gam, ["#2a78d6", "#2e9e6b", "#8b5cf6", "#c9a227"]):
            rr = sorted([r for r in res if r["basis"] == "all" and r["gamma"] == g], key=lambda r: r["sigma_rel"])
            ax.plot([r["sigma_rel"] for r in rr], [r["S_campplus"] for r in rr], "-o", color=col, markersize=4, label=f"gamma = {g:g}")
        o = next(r for r in res if r["config"] == "orig"); ax.axhline(o["S_campplus"], color="#1a1a19", linestyle=":", label="original audio")
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
