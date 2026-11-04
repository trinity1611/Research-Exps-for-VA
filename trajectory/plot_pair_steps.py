"""Plot one pair-experiment trajectory as a grid of mel spectrograms, one panel per Euler step (5 per row, 6 rows).
    python trajectory/plot_pair_steps.py <run_dir> <npz relative to run_dir> [more npz ...] [--out-dir DIR]
The states are rebuilt from noises.npz + the saved velocities, exactly as a reader of the dataset would do."""
import argparse, json, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

def states_of(run, rel):
    z = np.load(run / rel); m = json.loads(str(z["meta"])); n = np.load(run / "noises.npz")
    v = z["velocity"].astype("float32"); t = z["t"]; T = v.shape[-1]; pf = m["prompt_frames"]
    x = n[m["noise"]][:, pf:pf + T].copy(); st = [x.copy()]
    for k in range(v.shape[0]): x = x + (t[k + 1] - t[k]) * v[k]; st.append(x.copy())
    fin = z["final"].astype("float32"); err = np.abs(st[-1] - fin).mean() / np.abs(fin).mean() * 100
    return st, t, m, T, fin, err

def plot(run, rel, out_dir):
    st, t, m, T, fin, err = states_of(run, rel); fps = m["frames_per_s"]
    vmin, vmax = np.percentile(fin, 1), np.percentile(fin, 99.5)
    fig, axes = plt.subplots(6, 5, figsize=(20, 14.5), dpi=110, sharex=True, sharey=True); fig.patch.set_facecolor("white")
    for k, ax in enumerate(axes.ravel(), start=1):
        im = ax.imshow(st[k], origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax, extent=[0, T / fps, 0, 80], interpolation="nearest")
        ax.set_title(f"step {k}/{len(t) - 1}   t = {t[k]:.2f}", fontsize=10, loc="left", pad=3); ax.tick_params(labelsize=7, length=2)
        for s in ax.spines.values(): s.set_color("#999")
    for ax in axes[-1]: ax.set_xlabel("time (s)", fontsize=8)
    for ax in axes[:, 0]: ax.set_ylabel("mel band", fontsize=8)
    fig.subplots_adjust(left=0.04, right=0.9, top=0.94, bottom=0.05, hspace=0.3, wspace=0.12)
    cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01); cb.set_label("log-mel energy", fontsize=9); cb.ax.tick_params(labelsize=7)
    fig.suptitle(f"Flow-matching trajectory: {m['source']} spoken in the voice of {m['reference']}  (segment {m['segment']}, noise {m['noise']}, "
                 f"{T} frames = {T / fps:.1f} s, 80 mel bands; state after each of the {len(t) - 1} Euler steps)", fontsize=12, y=0.975)
    out = out_dir / f"steps_{m['source']}__to__{m['reference']}_seg{m['segment']}_{m['noise']}.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"{out.name}: {T} frames = {T / fps:.1f} s, reconstruction error {err:.3f} %")

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("run"); p.add_argument("npz", nargs="+"); p.add_argument("--out-dir", default=".")
    a = p.parse_args(); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for rel in a.npz: plot(Path(a.run), rel, out)
