#!/usr/bin/env python
"""
Look at saved trajectories (from save_trajectories.py).

  python trajectory/inspect_trajectory.py <traj dir>                     # summary of index.csv and one example file
  python trajectory/inspect_trajectory.py <traj dir>/<file>/seg0001_1.npz --png out.png
        prints per step: t, distance to the final state, step size, and (with --velocities runs) the size of the
        conditional and unconditional directions; --png draws the mel spectrogram at a few steps.

Loading in your own code:
    z = np.load(path); meta = json.loads(str(z["meta"])); states = z["w0_states"]   # (steps+1, n_mels, frames)
"""
import sys
import csv
import json
import argparse
from pathlib import Path

import numpy as np


def show_file(p, png=None):
    z = np.load(p)
    meta = json.loads(str(z["meta"]))
    print(json.dumps({k: meta[k] for k in ("file", "segment", "speaker", "target_voice", "start_s", "len_s", "steps",
                                           "cfg_rate", "frames_per_s", "window_offsets_frames")}, indent=1))
    nw = len([k for k in z.files if k.endswith("_states")])
    for j in range(nw):
        s = z[f"w{j}_states"].astype("float32")
        t = z[f"w{j}_t"]
        final = s[-1]
        print(f"\nwindow {j}: states {s.shape} (steps+1, n_mels, frames)")
        print(" step      t   |x_k - x_final|   |x_k - x_k-1|" + ("   |v_cond|  |v_uncond|" if f"w{j}_v_cond" in z.files else ""))
        for k in range(len(s)):
            line = f" {k:4d}  {t[k]:5.3f}   {np.sqrt(np.mean((s[k] - final) ** 2)):12.4f}   " + \
                   (f"{np.sqrt(np.mean((s[k] - s[k - 1]) ** 2)):12.4f}" if k else f"{'':12s}")
            if f"w{j}_v_cond" in z.files and k:
                vc, vu = z[f"w{j}_v_cond"][k - 1].astype("float32"), z[f"w{j}_v_uncond"][k - 1].astype("float32")
                line += f"   {np.sqrt(np.mean(vc ** 2)):8.3f}  {np.sqrt(np.mean(vu ** 2)):8.3f}"
            print(line)
    if png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        s = z["w0_states"].astype("float32")
        picks = sorted({0, len(s) // 4, len(s) // 2, 3 * len(s) // 4, len(s) - 1})
        fig, ax = plt.subplots(len(picks), 1, figsize=(10, 2.1 * len(picks)), sharex=True)
        for a_, k in zip(ax, picks):
            a_.imshow(s[k], origin="lower", aspect="auto", cmap="magma", vmin=np.percentile(s[-1], 1), vmax=np.percentile(s[-1], 99))
            a_.set_ylabel(f"step {k}\nt={z['w0_t'][k]:.2f}")
        ax[-1].set_xlabel("frame")
        fig.suptitle(f"{meta['file']} segment {meta['segment']} speaker {meta['speaker']} -> {meta['target_voice']}")
        fig.tight_layout(); fig.savefig(png, dpi=110)
        print(f"\npicture: {png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--png", default=None)
    a = ap.parse_args()
    p = Path(a.path)
    if p.is_dir():
        rows = list(csv.DictReader((p / "index.csv").open(encoding="utf-8")))
        files = sorted({r["file"] for r in rows})
        frames = sum(int(r["frames"]) for r in rows)
        size = sum(f.stat().st_size for f in p.rglob("*.npz"))
        print(f"{len(rows)} segments of {len(files)} files, {frames} frames in total, {size / 1e6:.0f} MB on disk")
        for f in files:
            rs = [r for r in rows if r["file"] == f]
            print(f"  {f}: {len(rs)} segments, speakers {sorted({r['speaker'] for r in rs})}, "
                  f"targets {sorted({r['target_voice'] for r in rs})}")
        if rows:
            print("\nexample:"); show_file(p / rows[0]["path"], a.png)
    else:
        show_file(p, a.png)


if __name__ == "__main__":
    main()
