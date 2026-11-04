#!/usr/bin/env python
"""
Run the anonymization pipeline (combined or L) exactly as usual and, on top, save the ENTIRE sampling trajectory of the
Seed-VC voice-conversion model for every piece of speech it converts.

How Seed-VC generates speech: it starts from random noise shaped like a mel spectrogram (n_mels x frames) and moves it
in STEPS small Euler steps (30 for combined, 25 for L) along the direction the diffusion transformer predicts, steered by
the content of the original speech and the target voice. The trajectory is that sequence of mel spectrograms:

    x_0 (pure noise)  ->  x_1  ->  ...  ->  x_STEPS (the converted speech, which the vocoder turns into audio)

Saved per converted segment (one diarized turn, or the whole file with --mode whole), one .npz:
    w<j>_states      (STEPS+1, n_mels, frames)  x_0 .. x_STEPS of DiT window j (long segments use several windows)
    w<j>_t           (STEPS+1,)                 the time t of each state (0 = noise, 1 = speech)
    w<j>_v_cond      (STEPS, n_mels, frames)    --velocities: the model's direction WITH the target voice / content
    w<j>_v_uncond    (STEPS, n_mels, frames)    --velocities: the same WITHOUT them (guidance branch)
    w<j>_style       (192,)                     the target voice's speaker embedding fed to the model
    meta             json: file, speaker, target voice, segment start/length, context, window offsets, steps, cfg ...
The guided step actually taken is (1 + cfg) * v_cond - cfg * v_uncond = (x_{k+1} - x_k) / dt, so the states alone
already contain it; --velocities adds the two branches separately. The reference prompt mel of every target voice is
saved once in refs/<voice>.npy. Only the frames of the speech being converted are kept (not the prompt part).

    python trajectory/save_trajectories.py --traj-out <dir> [--velocities] [--seed 0] -- <anonymize.py arguments>

Everything after "--" goes to src/anonymize.py unchanged (profile, wav dir, rttm dir, files, ...). Output:
    <dir>/<file_id>/seg<NNNN>_<speaker>.npz, <dir>/refs/<voice>.npy, <dir>/index.csv (one row per segment)
Size: about 0.43 MB per second of converted audio (float16 states), about 1.3 MB/s with --velocities. Each turn is
converted with 3 s of context on both sides, so the converted audio is about 1.5x the speech (measured: 173 s of
speech -> 262 s converted -> 331 MB with --velocities, about 110 MB without).
"""
import os
import sys
import csv
import json
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = (Path(os.environ["ANON_ROOT"]) if os.environ.get("ANON_ROOT") else next(d for d in HERE.parents if (d / "src" / "anonymize.py").exists())) / "src"   # pipeline root: ANON_ROOT, or a parent folder
sys.path.insert(0, str(SRC))


class Recorder:
    def __init__(self, out, velocities, dtype):
        self.out, self.velocities, self.dtype = Path(out), velocities, dtype
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "refs").mkdir(exist_ok=True)
        self.windows, self.engine, self.ref_name = [], None, None
        self.seg_count, self.saved_refs = {}, set()
        self.index_path = self.out / "index.csv"
        self.cfg = {}
        new = not self.index_path.exists()
        self.index = self.index_path.open("a", newline="", encoding="utf-8")
        self.iw = csv.writer(self.index)
        if new:
            self.iw.writerow(["file", "segment", "speaker", "target_voice", "start_s", "len_s", "padded_start_s",
                              "n_windows", "frames", "steps", "path"])

    # --- the sampler: a copy of seed-vc modules/flow_matching.py BASECFM.solve_euler that keeps every state ---------------
    def make_solver(self):
        rec = self

        def solve_euler(cfm, x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate=0.5):
            import torch
            t = t_span[0]
            prompt_len = prompt.size(-1)
            prompt_x = torch.zeros_like(x)
            prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
            x[..., :prompt_len] = 0
            if cfm.zero_prompt_speech_token:
                mu[..., :prompt_len] = 0
            keep = lambda a: a[0, :, prompt_len:].detach().float().cpu().numpy().astype(rec.dtype)
            states, ts, vc, vu = [keep(x)], [float(t)], [], []
            sol = []
            for step in range(1, len(t_span)):
                dt = t_span[step] - t_span[step - 1]
                if inference_cfg_rate > 0:
                    stacked = cfm.estimator(torch.cat([x, x], 0), torch.cat([prompt_x, torch.zeros_like(prompt_x)], 0), x_lens,
                                            torch.cat([t.unsqueeze(0), t.unsqueeze(0)], 0), torch.cat([style, torch.zeros_like(style)], 0),
                                            torch.cat([mu, torch.zeros_like(mu)], 0))
                    d_cond, d_unc = stacked.chunk(2, dim=0)
                    dphi_dt = (1.0 + inference_cfg_rate) * d_cond - inference_cfg_rate * d_unc
                else:
                    d_cond = d_unc = dphi_dt = cfm.estimator(x, prompt_x, x_lens, t.unsqueeze(0), style, mu)
                if rec.velocities:
                    vc.append(keep(d_cond)); vu.append(keep(d_unc))
                x = x + dt * dphi_dt
                t = t + dt
                sol.append(x)
                states.append(keep(x)); ts.append(float(t))
                x[:, :, :prompt_len] = 0
            import numpy as np
            w = {"states": np.stack(states), "t": np.asarray(ts, dtype="float32"),
                 "style": style[0].detach().float().cpu().numpy(), "cfg": float(inference_cfg_rate), "prompt_len": int(prompt_len)}
            if rec.velocities:
                w["v_cond"], w["v_uncond"] = np.stack(vc), np.stack(vu)
            rec.windows.append(w)
            if rec.ref_name and rec.ref_name not in rec.saved_refs:
                np.save(rec.out / "refs" / f"{rec.ref_name}.npy", prompt[0].detach().float().cpu().numpy().astype("float32"))
                rec.saved_refs.add(rec.ref_name)
            return sol[-1]
        return solve_euler

    # --- one segment record from anonymize.py = the windows sampled since the segment's convert() began ------------------
    def flush(self, seg):
        import numpy as np
        if not self.windows:
            return
        fid, spk = seg["file"], str(seg["spk"])
        k = self.seg_count.get(fid, 0) + 1
        self.seg_count[fid] = k
        d = self.out / fid
        d.mkdir(exist_ok=True)
        e = self.engine
        ovl = getattr(e, "overlap_frame_len", None)
        offsets, off = [], 0
        for w in self.windows:                       # source frame where each window starts (windows overlap by ovl frames)
            offsets.append(off)
            off += w["states"].shape[-1] - (ovl or 0)
        ctx = float(self.cfg.get("context_sec", 0) or 0)
        start = seg.get("start_s")
        meta = {"file": fid, "segment": k, "speaker": spk, "target_voice": seg.get("ref"), "start_s": start,
                "len_s": seg.get("len_s"), "context_sec": ctx,
                "padded_start_s": None if start is None else max(0.0, start - ctx),
                "note": "frames cover [padded_start_s, start_s + len_s + context_sec]: the context around the turn is converted too",
                "sr": getattr(e, "sr", None), "hop": getattr(e, "hop", None),
                "frames_per_s": (e.sr / e.hop) if e is not None and getattr(e, "hop", None) else None,
                "window_offsets_frames": offsets, "window_overlap_frames": ovl, "steps": int(self.windows[0]["states"].shape[0] - 1),
                "cfg_rate": self.windows[0]["cfg"], "prompt_frames": self.windows[0]["prompt_len"], "seed": self.cfg.get("seed"),
                "pipeline_profile": self.cfg.get("profile"), "solver": "Euler, linear t from 0 to 1 (seed-vc BASECFM.solve_euler)"}
        arrays = {"meta": np.asarray(json.dumps(meta))}
        for j, w in enumerate(self.windows):
            arrays[f"w{j}_states"], arrays[f"w{j}_t"], arrays[f"w{j}_style"] = w["states"], w["t"], w["style"]
            if "v_cond" in w:
                arrays[f"w{j}_v_cond"], arrays[f"w{j}_v_uncond"] = w["v_cond"], w["v_uncond"]
        p = d / f"seg{k:04d}_{spk}.npz"
        np.savez(p, **arrays)
        frames = sum(w["states"].shape[-1] for w in self.windows) - (ovl or 0) * (len(self.windows) - 1)
        self.iw.writerow([fid, k, spk, seg.get("ref"), start, seg.get("len_s"), meta["padded_start_s"], len(self.windows),
                          frames, meta["steps"], str(p.relative_to(self.out))])
        self.index.flush()
        self.windows = []


def main():
    if "--" not in sys.argv:
        sys.exit(__doc__)
    cut = sys.argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj-out", required=True, help="folder for the trajectories")
    ap.add_argument("--velocities", action="store_true", help="also save the model's two velocity branches per step")
    ap.add_argument("--float32", action="store_true", help="store float32 instead of float16 (twice the size)")
    ap.add_argument("--seed", type=int, default=0, help="random seed of the starting noise (makes the run reproducible)")
    a = ap.parse_args(sys.argv[1:cut])
    rest = sys.argv[cut + 1:]
    if "--no-warmup" not in rest:
        rest.append("--no-warmup")

    import torch
    import anonymize
    rec = Recorder(a.traj_out, a.velocities, "float32" if a.float32 else "float16")
    rec.cfg = {"seed": a.seed}

    # 1) the engine: remember it, the target voice of the current call, and install the recording sampler once
    orig_convert = anonymize.SeedVCEngine.convert
    patched = {"done": False}

    def convert(self, w16, ref, t):
        if not patched["done"]:
            type(self.model.cfm).solve_euler = rec.make_solver()
            patched["done"] = True
        rec.engine, rec.ref_name, rec.windows = self, ref.get("name"), []
        return orig_convert(self, w16, ref, t)
    anonymize.SeedVCEngine.convert = convert

    # 2) the segment log: every segment record closes one trajectory file
    orig_write = anonymize.JsonlWriter.write

    def write(self, record):
        if "len_s" in record and "spk" in record and "file" in record:
            rec.flush(record)
        return orig_write(self, record)
    anonymize.JsonlWriter.write = write

    # 3) the run settings the files need (context, profile), then the pipeline itself
    orig_parse = anonymize.parse_args

    def parse_args():
        args = orig_parse()
        if args.model == "v2":
            sys.exit("trajectories are implemented for the Seed-VC V1 engine (pipelines combined and L)")
        rec.cfg.update({"profile": getattr(args, "profile", None)})
        return args
    anonymize.parse_args = parse_args
    orig_apply = anonymize.apply_profile

    def apply_profile(args):
        r = orig_apply(args)
        rec.cfg["context_sec"] = getattr(args, "context_sec", 0)
        return r
    anonymize.apply_profile = apply_profile

    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    sys.argv = [str(SRC / "anonymize.py")] + rest
    try:
        anonymize.main()
    finally:
        rec.index.close()
    print(f"trajectories: {sum(rec.seg_count.values())} segments of {len(rec.seg_count)} files -> {rec.out}  (index.csv)")


if __name__ == "__main__":
    main()
