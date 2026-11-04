"""MSE between the last denoising step and every other step, in mel space, for a speaker-pair run.

    python trajectory/pair_mse.py <run_dir> [--out <run_dir>/mse] [--limit N]

For every conversion in index.csv the states x_0 .. x_S are rebuilt from noises.npz and the saved velocities
(x_0 = z_j[:, prompt_frames : prompt_frames + frames], x_{k+1} = x_k + dt_k * velocity_k; x_S = the saved final mel),
and MSE_k = mean over all mel bins of (x_k - x_S)^2 is computed for k = 0 .. S (MSE_S = 0). Conversions split into
windows are pooled over their windows.

Writes  mse_per_conversion.csv   one row per conversion: source, reference, segment, noise, frames, energy, mse_00 .. mse_S
        mse_summary.csv          per step: t, mean, std, median, p10, p90, min, max over all conversions
        mse_by_noise.csv         per step: mean MSE for each noise z1 .. zN
        mse_vs_step.png          the curves (log scale)
"""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np


def states_mse(z, npz):
    """Pooled squared error per step against the final state, over all windows of one conversion."""
    meta = json.loads(str(npz["meta"]))
    pf = meta["prompt_frames"]
    keys = ["velocity"] if "velocity" in npz.files else sorted(k for k in npz.files if k.endswith("_velocity"))
    t = npz["t"]
    sq = None
    n = 0
    energy = 0.0
    for key in keys:
        v = npz[key].astype("float32")                                  # (steps, n_mels, frames)
        fin = npz[key.replace("velocity", "final")].astype("float32")   # (n_mels, frames)
        S, _, T = v.shape
        x = z[:, pf:pf + T].astype("float32").copy()
        s = np.zeros(S + 1)
        s[0] = ((x - fin) ** 2).sum()
        for k in range(S):
            x += (t[k + 1] - t[k]) * v[k]
            s[k + 1] = ((x - fin) ** 2).sum()
        sq = s if sq is None else sq + s
        n += fin.size
        energy += (fin ** 2).sum()
    return sq / n, energy / n, meta, t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run"); ap.add_argument("--out", default=None); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run = Path(a.run)
    out = Path(a.out) if a.out else run / "mse"
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(run / "index.csv", newline="", encoding="utf-8")))
    rows = [{k.strip(): v.strip() for k, v in r.items()} for r in rows]
    if a.limit:
        rows = rows[: a.limit]
    noises = np.load(run / "noises.npz")
    t0 = time.time()
    per, energies, t_span = [], [], None
    with open(out / "mse_per_conversion.csv", "w", newline="", encoding="utf-8") as fh:
        w = None
        for i, r in enumerate(rows, 1):
            npz = np.load(run / r["path"])
            mse, energy, meta, t = states_mse(noises[r["noise"]], npz)
            t_span = t
            if w is None:
                w = csv.writer(fh)
                w.writerow(["source", "reference", "segment", "noise", "frames", "energy"] + [f"mse_{k:02d}" for k in range(len(mse))])
            w.writerow([r["source"], r["reference"], r["segment"], r["noise"], r["frames"], f"{energy:.5f}"] + [f"{m:.6g}" for m in mse])
            per.append((r["noise"], mse))
            energies.append(energy)
            if i % 250 == 0 or i == len(rows):
                el = time.time() - t0
                print(f"[{i}/{len(rows)}] {el / 60:.1f} min, left ~{el / i * (len(rows) - i) / 60:.0f} min", flush=True)
    M = np.stack([m for _, m in per])                                  # (conversions, steps+1)
    S = M.shape[1] - 1
    energy_mean = float(np.mean(energies))
    with open(out / "mse_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "t", "mean", "std", "median", "p10", "p90", "min", "max", "n"])
        for k in range(S + 1):
            c = M[:, k]
            w.writerow([k, f"{t_span[k]:.4f}", f"{c.mean():.6g}", f"{c.std():.6g}", f"{np.median(c):.6g}", f"{np.percentile(c, 10):.6g}",
                        f"{np.percentile(c, 90):.6g}", f"{c.min():.6g}", f"{c.max():.6g}", len(c)])
    zs = sorted({z for z, _ in per}, key=lambda s: int(s[1:]))
    by_noise = {z: np.stack([m for zz, m in per if zz == z]).mean(axis=0) for z in zs}
    with open(out / "mse_by_noise.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "t"] + zs)
        for k in range(S + 1):
            w.writerow([k, f"{t_span[k]:.4f}"] + [f"{by_noise[z][k]:.6g}" for z in zs])

    print(f"\nMSE to step {S} in mel space, {len(per)} conversions, {len(zs)} noises  (for scale: mean of x_S^2 over the mel = {energy_mean:.3f})")
    print(f"{'step':>4s} {'t':>5s} {'mean':>10s} {'std':>9s} {'median':>10s} {'p10':>10s} {'p90':>10s}   " + "  ".join(f"{z:>9s}" for z in zs))
    for k in range(S + 1):
        c = M[:, k]
        print(f"{k:4d} {t_span[k]:5.2f} {c.mean():10.4g} {c.std():9.3g} {np.median(c):10.4g} {np.percentile(c, 10):10.4g} {np.percentile(c, 90):10.4g}   "
              + "  ".join(f"{by_noise[z][k]:9.4g}" for z in zs))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ks = np.arange(S + 1)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=120)
        fig.patch.set_facecolor("white")
        ax = axes[0]
        mean, p10, p90 = M.mean(axis=0), np.percentile(M, 10, axis=0), np.percentile(M, 90, axis=0)
        ax.fill_between(ks[:-1], p10[:-1], p90[:-1], color="#86b6ef", alpha=0.35, linewidth=0, label="10th to 90th percentile")
        ax.plot(ks[:-1], mean[:-1], color="#2a78d6", linewidth=2, marker="o", markersize=3.5, label="mean over conversions")
        ax.set_yscale("log"); ax.set_xlabel("denoising step k"); ax.set_ylabel(f"MSE(x_k, x_{S})  [mel units squared]")
        ax.set_title(f"Distance to the final state, {len(per)} conversions", loc="left", fontsize=11)
        ax.grid(True, color="#e6e5e0", linewidth=0.8); ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9)
        ax = axes[1]
        cols = ["#2a78d6", "#eb6834", "#2e9e6b", "#8b5cf6", "#c9a227", "#d63a8a", "#1a1a19", "#7a7a73"]
        for z, col in zip(zs, cols):
            ax.plot(ks[:-1], by_noise[z][:-1], color=col, linewidth=1.8, marker="o", markersize=3, label=z)
        ax.set_yscale("log"); ax.set_xlabel("denoising step k"); ax.set_ylabel(f"mean MSE(x_k, x_{S})")
        ax.set_title("Per starting noise", loc="left", fontsize=11)
        ax.grid(True, color="#e6e5e0", linewidth=0.8); ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9, title="noise")
        for ax in axes:
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "mse_vs_step.png")
        print(f"\nplot: {out / 'mse_vs_step.png'}")
    except Exception as e:                                   # the CSVs are the result; the picture is a convenience
        print(f"(no plot: {e})")
    print(f"written: {out / 'mse_per_conversion.csv'}, mse_summary.csv, mse_by_noise.csv")


if __name__ == "__main__":
    main()
