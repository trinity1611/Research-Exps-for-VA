"""Generation stage (spec sections 1-3): run Seed-VC (the pipeline's engine, unmodified) on a manifest of source
utterances, with the TARGET drawn at random independently of the source speaker, R noise seeds per utterance, and save
at a grid of denoising steps t_k:
    x_t      the mel state                                   (Y_t^A, raw state)
    v        v_theta(x_t, t, u), the predicted velocity
    xhat1_t  x_t + (1 - t) * v, the one-step clean estimate  -> vocoder -> waveform  (Y_t^B, the primary analysis)
and the conditioning u once per (utterance, target): content condition mu, the target's prompt mel and style vector.

[VERIFY] items of the spec, checked in repos/seed-vc/modules/flow_matching.py and src/anonymize.py:
  - training interpolant  y_t = (1 - (1 - sigma_min) t) z + t x1,  u = x1 - (1 - sigma_min) z,  sigma_min = 1e-6
    -> x_t + (1 - t) v = x1 + sigma_min z: the one-step clean estimate of the spec, exact up to sigma_min.
  - sampler: Euler, t_k = k / STEPS linear from 0 (noise) to 1 (data), STEPS = 30 for profile `combined`.
  - classifier-free guidance: v = (1 + cfg) v_cond - cfg v_uncond, cfg = 1.0 (profile); the uncond branch zeroes
    mu, the prompt and the style.
  - conditioning path: mu = length-regulated XLS-R content features (384 ch per mel frame) with the target's
    prompt content in front; prompt = the target's mel (first 10 s); style = CAM++ embedding (192) of the target.
  - the prompt region of x_t is held at the prompt mel and is not generated; all saved arrays drop it.

Manifest CSV (one row per source utterance): utt_id, src_spk, sentence_id, wav [, split]
Target pool CSV (one row per reference clip):  tgt_spk, wav [, gender]
Output:  <out>/gen/<utt_id>__r<r>.npz, <out>/gen/<utt_id>__r<r>__k<kk>.wav (one per saved step),
         <out>/cond/<utt_id>__r<r>.npz (u: mu), <out>/targets/<tgt_spk>.npz (prompt mel, style),
         <out>/metadata.csv (utt_id, src_spk, sentence_id, target_spk, target_ref_path, seed, split, steps_saved, frames, path)

    python generate.py --manifest m.csv --pool pool.csv --out <out> --seedvc-root <seed-vc> --profile <combined.env>
                       [--models-dir D] [--seeds 3] [--steps-grid 0,3,6,10,15,20,25,30] [--batch 3] [--limit N]
"""
import argparse, csv, hashlib, json, sys, time
import os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ["ANON_ROOT"]) if os.environ.get("ANON_ROOT") else next(d for d in HERE.parents if (d / "src" / "anonymize.py").exists())   # pipeline root: ANON_ROOT, or a parent folder
sys.path.insert(0, str(ROOT / "src"))
SR = 16000


def read_csv(p):
    rows = list(csv.DictReader(open(p, newline="", encoding="utf-8")))
    return [{k.strip(): (v or "").strip() for k, v in r.items()} for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True); ap.add_argument("--pool", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seedvc-root", required=True); ap.add_argument("--profile", required=True); ap.add_argument("--models-dir", default=None)
    ap.add_argument("--ref-max-sec", type=float, default=None)
    ap.add_argument("--seeds", type=int, default=3, help="R noise seeds per utterance")
    ap.add_argument("--steps-grid", default="0,3,6,10,15,20,25,30", help="Euler steps k at which x_t, v, xhat1 are saved (t = k / STEPS)")
    ap.add_argument("--batch", type=int, default=3, help="seeds denoised together in one DiT pass")
    ap.add_argument("--target-seed", type=int, default=12345, help="seed of the independent target draw")
    ap.add_argument("--noise-base", type=int, default=50000)
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--no-vocode", action="store_true")
    a = ap.parse_args()

    import torch, torchaudio, soundfile as sf
    import anonymize
    out = Path(a.out); (out / "gen").mkdir(parents=True, exist_ok=True); (out / "cond").mkdir(exist_ok=True); (out / "targets").mkdir(exist_ok=True)
    man = read_csv(a.manifest)[: a.limit]
    pool = read_csv(a.pool)
    src_spks = {r["src_spk"] for r in man}
    pool = [r for r in pool if r["tgt_spk"] not in src_spks]                 # T never equals S (pool disjoint from sources)
    assert pool, "target pool is empty after removing source speakers"
    grid = sorted({int(k) for k in a.steps_grid.split(",")})

    sys.argv = ["anonymize.py", "--profile", a.profile, "--seedvc-root", a.seedvc_root, "--wav-dir", str(out), "--out-dir", str(out / "_unused"),
                "--log-dir", str(out / "logs"), "--tag", "mi", "--mode", "whole", "--no-warmup"] + (["--models-dir", a.models_dir] if a.models_dir else []) \
               + (["--ref-max-sec", str(a.ref_max_sec)] if a.ref_max_sec else [])
    args = anonymize.parse_args(); anonymize.apply_profile(args)
    svc = anonymize.bootstrap_seedvc(args.seedvc_root)
    eng = anonymize.SeedVCEngine(svc, args)
    cfm = eng.model.cfm
    STEPS, cfg, n_mels, T_max = eng.steps, eng.cfg, cfm.in_channels, eng.max_context_window
    assert max(grid) <= STEPS, f"steps grid exceeds STEPS={STEPS}"
    sigma_min = float(getattr(cfm, "sigma_min", 0.0))
    print(f"engine: steps={STEPS} cfg={cfg} n_mels={n_mels} sigma_min={sigma_min} prompt<= {eng.ref_max_sec}s  grid k={grid}  seeds={a.seeds} batch={a.batch}")
    torch.set_grad_enabled(False)
    dev = eng.device

    state = {"noises": None, "rec": {}}

    def inference(self, mu, x_lens, prompt, style, f0, n_timesteps, temperature=1.0, inference_cfg_rate=0.5):
        zs = state["noises"]; nb = len(zs); T = mu.size(1)
        z = torch.stack([z_[:, :T] for z_ in zs], 0) * temperature
        mu, prompt = mu.expand(nb, -1, -1).clone(), prompt.expand(nb, -1, -1).clone()
        style, x_lens = style.expand(nb, -1).clone(), x_lens.expand(nb).clone()
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate)

    def solve_euler(self, x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate=0.5):
        t = t_span[0]
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x); prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        if self.zero_prompt_speech_token:
            mu[..., :prompt_len] = 0
        keep = lambda v: v[:, :, prompt_len:].detach().float().cpu().numpy().astype("float16")
        rec = {"x": {}, "v": {}, "xhat1": {}, "t": t_span.detach().float().cpu().numpy(), "prompt_frames": int(prompt_len)}
        state["rec"] = rec
        for step in range(1, len(t_span)):
            k = step - 1
            dt = t_span[step] - t_span[step - 1]
            if inference_cfg_rate > 0:
                stacked = self.estimator(torch.cat([x, x], 0), torch.cat([prompt_x, torch.zeros_like(prompt_x)], 0), torch.cat([x_lens, x_lens], 0),
                                         t.reshape(1).expand(2 * x.size(0)), torch.cat([style, torch.zeros_like(style)], 0),
                                         torch.cat([mu, torch.zeros_like(mu)], 0))
                d_cond, d_unc = stacked.chunk(2, dim=0)
                dphi_dt = (1.0 + inference_cfg_rate) * d_cond - inference_cfg_rate * d_unc
            else:
                dphi_dt = self.estimator(x, prompt_x, x_lens, t.reshape(1).expand(x.size(0)), style, mu)
            if k in grid:
                rec["x"][k] = keep(x); rec["v"][k] = keep(dphi_dt)
                rec["xhat1"][k] = keep(x + (1 - t) * dphi_dt)                 # one-step clean estimate
            x = x + dt * dphi_dt
            t = t + dt
            x[:, :, :prompt_len] = 0
        if STEPS in grid:
            rec["x"][STEPS] = keep(x); rec["xhat1"][STEPS] = keep(x)          # t = 1: the result itself
        return x
    type(cfm).inference = inference; type(cfm).solve_euler = solve_euler

    def vocode(mel16):
        """(nb, n_mels, frames) float16 numpy mel -> list of 16 kHz float32 signals."""
        m = torch.from_numpy(mel16.astype("float32")).to(dev)
        w = eng.vocoder_fn(m).reshape(m.size(0), -1)
        w16 = torchaudio.functional.resample(w, eng.sr, SR).cpu().numpy()
        return [w16[i] for i in range(w16.shape[0])]

    # references: prepare every pool clip once (prompt mel, style, prompt content)
    refs = {}
    for r in pool:
        if r["tgt_spk"] in refs:
            continue
        refs[r["tgt_spk"]] = (r, eng.prepare_ref(Path(r["wav"])))
        ref = refs[r["tgt_spk"]][1]
        np.savez(out / "targets" / f"{r['tgt_spk']}.npz", prompt_mel=ref["mel2"][0].float().cpu().numpy().astype("float16"),
                 style=ref["style2"][0].float().cpu().numpy(), prompt_frames=ref["prompt_frames"], ref_path=str(r["wav"]))
    tgt_ids = sorted(refs)
    print(f"{len(man)} source utterances, {len(src_spks)} source speakers; target pool {len(tgt_ids)} speakers (disjoint)")
    rng_t = np.random.default_rng(a.target_seed)

    meta_p = out / "metadata.csv"
    new = not meta_p.exists()
    done_keys = set()
    if not new:
        done_keys = {(r["utt_id"], r["seed"]) for r in read_csv(meta_p)}
    fh = open(meta_p, "a", newline="", encoding="utf-8"); mw = csv.writer(fh)
    if new:
        mw.writerow(["utt_id", "src_spk", "sentence_id", "target_spk", "target_ref_path", "seed", "split", "steps_saved", "frames", "path"])
    groups = [list(range(a.seeds))[i:i + a.batch] for i in range(0, a.seeds, a.batch)]
    t0 = time.time(); n_done = 0; skipped = 0
    for ui, row in enumerate(man):
        # the target: drawn uniformly from the pool, independent of the source speaker (a function of utt_id and the draw seed only)
        tgt = tgt_ids[int(rng_t.integers(len(tgt_ids)))]
        if all((row["utt_id"], str(r)) in done_keys for r in range(a.seeds)):
            continue
        ref = refs[tgt][1]
        x, sr = sf.read(row["wav"], dtype="float32", always_2d=True); x = x.mean(1)
        if sr != SR:
            x = torchaudio.functional.resample(torch.from_numpy(x)[None], sr, SR)[0].numpy()
        w16 = torch.from_numpy(x)[None].to(dev)
        # content condition (as SeedVCEngine.convert)
        src = torchaudio.functional.resample(w16, SR, eng.sr)
        S_alt = eng.semantic(w16); mel = eng.to_mel(src.float())
        target_lengths = torch.LongTensor([int(mel.size(2) * eng.length_adjust)]).to(dev)
        cond = eng.model.length_regulator(S_alt, ylens=target_lengths, n_quantizers=3, f0=None)[0]
        if ref["mel2"].size(2) + cond.size(1) > T_max:
            skipped += 1; print(f"    skip {row['utt_id']}: {cond.size(1)} frames + prompt exceed one window"); continue
        cat_condition = torch.cat([ref["prompt_condition"], cond], dim=1)
        np.savez(out / "cond" / f"{row['utt_id']}.npz", mu=cat_condition[0].float().cpu().numpy().astype("float16"),
                 prompt_frames=ref["prompt_frames"], target_spk=tgt)
        for js in groups:
            # noise per (utterance, seed): deterministic, independent of the speaker
            gens = [torch.Generator(device="cpu").manual_seed(a.noise_base + 1000 * r + ui) for r in js]
            state["noises"] = [torch.randn(n_mels, T_max, generator=g).to(dev) for g in gens]
            with torch.autocast(device_type=dev.type, dtype=torch.float16 if eng.fp16 else torch.float32):
                cfm.inference(cat_condition, torch.LongTensor([cat_condition.size(1)]).to(dev), ref["mel2"], ref["style2"], None, STEPS, inference_cfg_rate=cfg)
            rec = state["rec"]
            frames = rec["x"][grid[0]].shape[-1]
            for i, r in enumerate(js):
                p = out / "gen" / f"{row['utt_id']}__r{r}.npz"
                arrays = {"t": rec["t"], "steps_grid": np.asarray(grid),
                          "meta": np.asarray(json.dumps({"utt_id": row["utt_id"], "src_spk": row["src_spk"], "sentence_id": row["sentence_id"],
                                                         "target_spk": tgt, "seed": r, "noise_seed": a.noise_base + 1000 * r + ui, "steps": STEPS, "cfg": cfg,
                                                         "prompt_frames": rec["prompt_frames"], "frames": frames, "frames_per_s": eng.sr / eng.hop,
                                                         "xhat1": "x_t + (1 - t) * v  (sigma_min %g)" % sigma_min,
                                                         "Y_A": "x_k (raw state)", "Y_B": "xhat1_k -> vocoder -> *_k<kk>.wav"}))}
                for k in grid:
                    arrays[f"x_{k:02d}"] = rec["x"][k][i]; arrays[f"xhat1_{k:02d}"] = rec["xhat1"][k][i]
                    if k in rec["v"]:
                        arrays[f"v_{k:02d}"] = rec["v"][k][i]
                np.savez(p, **arrays)
            if not a.no_vocode:
                for k in grid:
                    ys = vocode(rec["xhat1"][k])
                    for i, r in enumerate(js):
                        sf.write(str(out / "gen" / f"{row['utt_id']}__r{r}__k{k:02d}.wav"), anonymize.fit_length(ys[i], w16.size(-1)), SR, subtype="PCM_16")
            for r in js:
                mw.writerow([row["utt_id"], row["src_spk"], row["sentence_id"], tgt, refs[tgt][0]["wav"], r, row.get("split", ""),
                             " ".join(map(str, grid)), frames, f"gen/{row['utt_id']}__r{r}.npz"])
            fh.flush(); n_done += len(js)
        if (ui + 1) % 25 == 0 or ui + 1 == len(man):
            el = time.time() - t0
            print(f"[{ui + 1}/{len(man)}] {n_done} outputs, {el / 60:.1f} min, left ~{el / (ui + 1) * (len(man) - ui - 1) / 60:.0f} min"
                  + (f"  GPU peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB" if dev.type == "cuda" else ""), flush=True)
    fh.close()
    print(f"done: {n_done} outputs ({skipped} utterances skipped); metadata {meta_p}")


if __name__ == "__main__":
    main()
