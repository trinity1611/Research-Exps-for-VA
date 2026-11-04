#!/usr/bin/env python
"""
Speaker-pair trajectory experiment on diarized conversations (combined or L pipeline, Seed-VC V1).

Design (as specified):
  * 10 two-speaker conversations -> 20 speakers (the two main diarized speakers of each).
  * per speaker, 5 SOURCE segments of 12-20 s (a turn longer than 20 s is cut to its first 20 s); the speaker's
    remaining speech (any duration, capped by the pipeline's prompt length) is the REFERENCE voice.
  * speaker pairs: 20C2 = 190 unordered pairs (A, B) with A < B: A's 5 segments are converted into B's voice
    (--pairs ordered gives both directions, 380 pairs).
  * 4 noise vectors z1..z4 from a standard Gaussian, FIXED for every conversion: each z_j is drawn once at the longest
    length the model ever uses (30 s of mel frames) and cut to the length needed, so the same values start every
    trajectory of the same length. Seeds: --noise-seed + j.
  * saved per conversion: the velocity the model predicted at each of the STEPS Euler steps (the trajectory in the
    model's representation space, the mel spectrogram), float16, plus the converted audio.

  python trajectory/pair_experiment.py select --wav-dir data/wav --rttm-dir data/rttm --out trajectory/pairs_exp
  python trajectory/pair_experiment.py run --out trajectory/pairs_exp --seedvc-root <seed-vc> --models-dir <models> \
         --profile configs/pipelines/combined.env

Output <out>/
  selection.json           conversations, speakers, the 5 source segments and the reference stretches of each speaker
  sources/<spk>/seg<k>.wav source segments (16 kHz)        refs/<spk>.wav   reference audio of each speaker
  noises.npz               z1..z4 (full length) + seeds     index.csv       one row per conversion
  pairs/<A>__to__<B>/seg<k>_z<j>.npz   velocity (STEPS, n_mels, frames) float16, t (STEPS+1,), meta (json)
  pairs/<A>__to__<B>/seg<k>_z<j>.wav   the converted audio (16 kHz)
The states are not stored: x_k = z[:frames] + sum_{i<k} dt_i * velocity_i reproduces them (x_0 = the noise).
With --branches the two guidance branches v_cond / v_uncond are stored as well (guided = (1+cfg) v_cond - cfg v_uncond).
"""
import sys
import os
import csv
import json
import time
import argparse
import itertools
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ["ANON_ROOT"]) if os.environ.get("ANON_ROOT") else next(d for d in HERE.parents if (d / "src" / "anonymize.py").exists())   # pipeline root: ANON_ROOT, or a parent folder (cluster: ~/anon-bench)
sys.path.insert(0, str(ROOT / "src"))
SR = 16000


# ---------------------------------------------------------------- turns from an RTTM -------------------------------------
def read_rttm(path):
    segs = []
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            segs.append((float(p[3]), float(p[3]) + float(p[4]), p[7]))
    return sorted(segs)


def speaker_turns(rttm, label, merge_gap=2.0, min_len=0.5):
    """Stretches of one diarized speaker: everything overlapped by other speakers is removed, and two pieces are joined
    across a gap shorter than merge_gap only when nobody else speaks in that gap (a pause, not a reply). So a stretch
    cut from the original audio contains this speaker and silence only."""
    others = [(a, b) for a, b, l in rttm if l != label]
    silent = lambda g0, g1: all(o1 <= g0 or o0 >= g1 for o0, o1 in others)
    turns = []
    for a, b, l in rttm:
        if l != label:
            continue
        cuts = [(a, b)]
        for o0, o1 in others:
            nxt = []
            for c0, c1 in cuts:
                if o1 <= c0 or o0 >= c1:
                    nxt.append((c0, c1))
                else:
                    if o0 > c0:
                        nxt.append((c0, o0))
                    if o1 < c1:
                        nxt.append((o1, c1))
            cuts = nxt
        turns.extend(cuts)
    turns.sort()
    merged = []
    for a, b in turns:
        if merged and a - merged[-1][1] < merge_gap and silent(merged[-1][1], a):
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return [(a, b) for a, b in merged if b - a >= min_len]


def build_segments(turns, lo=12.0, hi=20.0, pad=0.05):
    """Source segments of lo..hi seconds: a speaker's clean stretches are taken in time order and joined (the other
    speaker's interjections between them are left out) until a segment holds >= lo s; a stretch that would push it
    past hi is cut there and its remainder opens the next segment. Returns a list of segments, each a list of
    (start, end) pieces."""
    items = [(a + pad, b - pad) for a, b in turns if b - a > 2 * pad + 0.3]
    segs, cur, cur_len = [], [], 0.0
    while items:
        a, b = items.pop(0)
        room = hi - cur_len
        if room < 0.05:
            items.insert(0, (a, b)); segs.append(cur); cur, cur_len = [], 0.0
            continue
        if b - a > room:
            items.insert(0, (a + room, b)); b = a + room
        cur.append((a, b)); cur_len += b - a
        if cur_len >= lo - 1e-6:
            segs.append(cur); cur, cur_len = [], 0.0
    return segs


def cut_pieces(x, pieces, fade=0.02, pause=0.15):
    """Concatenate pieces of a 16 kHz signal with short fades and a short pause between them."""
    nf = int(fade * SR)
    ramp = np.linspace(0, 1, nf, dtype=np.float32)
    out = []
    for a, b in pieces:
        seg = x[int(a * SR):int(b * SR)].copy()
        if len(seg) > 2 * nf:
            seg[:nf] *= ramp; seg[-nf:] *= ramp[::-1]
        out.append(seg); out.append(np.zeros(int(pause * SR), dtype=np.float32))
    return np.concatenate(out[:-1]) if out else np.zeros(0, dtype=np.float32)


def spread(items, n):
    """n items spread evenly over the list (first, last and evenly in between)."""
    if len(items) <= n:
        return list(items)
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in idx]


# ---------------------------------------------------------------- select ------------------------------------------------
def select(a):
    import soundfile as sf
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wav_dir, rttm_dir = Path(a.wav_dir), Path(a.rttm_dir)
    ids = [p.stem for p in sorted(wav_dir.glob("*.wav"))]
    if a.candidates:
        want = [l.strip() for l in Path(a.candidates).read_text().splitlines() if l.strip()]
        ids = [i for i in want if i in ids] + [i for i in ids if i not in want]       # preferred ones first
    excl = set(a.exclude)
    for sp in a.exclude_selection:
        excl |= {c["file"] for c in json.loads(Path(sp).read_text(encoding="utf-8"))["conversations"]}
    if excl:
        ids = [i for i in ids if i not in excl]
        print(f"excluded {len(excl)} recordings of earlier runs; {len(ids)} candidates left")
    convs = []
    for fid in ids:
        rp = next(iter(sorted(rttm_dir.glob(f"{fid}*.rttm"))), None)      # <id>.rttm or <id>_SPEAKER.rttm
        if rp is None:
            continue
        rttm = read_rttm(rp)
        per = {}
        for s0, s1, l in rttm:
            per[l] = per.get(l, 0.0) + s1 - s0
        if len(per) < 2:
            continue
        labels = sorted(per, key=per.get, reverse=True)[:2]
        speakers = []
        for l in labels:
            turns = speaker_turns(rttm, l)
            all_segs = build_segments(turns, a.min_sec, a.max_sec)
            if len(all_segs) < a.n_seg + 1:                       # n_seg sources + at least one segment's worth of reference speech
                break
            chosen = spread(all_segs, a.n_seg)
            chosen_ids = {id(s) for s in chosen}
            rest = [s for s in all_segs if id(s) not in chosen_ids]
            ref, tot = [], 0.0
            for s in rest:                                         # reference: other segments of the same speaker, in time order
                for s0, s1 in s:
                    ref.append((s0, s1)); tot += s1 - s0
                if tot >= a.ref_sec:
                    break
            speakers.append({"label": l, "speech_s": round(per[l], 1), "clean_stretches": len(turns), "segments_available": len(all_segs),
                             "segments": [{"k": k + 1, "start_s": round(s[0][0], 3), "end_s": round(s[-1][1], 3),
                                           "len_s": round(sum(b - a_ for a_, b in s), 3), "pieces": [[round(a_, 3), round(b, 3)] for a_, b in s]}
                                          for k, s in enumerate(chosen)],
                             "reference": [{"start_s": round(s0, 3), "end_s": round(s1, 3)} for s0, s1 in ref], "reference_s": round(tot, 1)})
        if len(speakers) == 2:
            convs.append({"file": fid, "speakers": speakers})
        if len(convs) == a.n_conv:
            break
    if len(convs) < a.n_conv:
        print(f"WARNING: only {len(convs)} conversations have two speakers with {a.n_seg} turns of >= {a.min_sec} s")
    # write the source segments and the reference audio of every speaker
    (out / "sources").mkdir(exist_ok=True); (out / "refs").mkdir(exist_ok=True)
    spk_rows = []
    for c in convs:
        x, sr = sf.read(str(wav_dir / f"{c['file']}.wav"), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr != SR:
            import torch, torchaudio
            x = torchaudio.functional.resample(torch.from_numpy(x)[None], sr, SR)[0].numpy()
        for s in c["speakers"]:
            lab = s["label"] if s["label"].lower().startswith("spk") else f"spk{s['label']}"
            spk = f"{c['file']}_{lab}"
            s["id"] = spk
            d = out / "sources" / spk
            d.mkdir(exist_ok=True)
            for seg in s["segments"]:
                sf.write(str(d / f"seg{seg['k']}.wav"), cut_pieces(x, seg["pieces"]), SR, subtype="PCM_16")
            sf.write(str(out / "refs" / f"{spk}.wav"), cut_pieces(x, [(r["start_s"], r["end_s"]) for r in s["reference"]]), SR, subtype="PCM_16")
            spk_rows.append((spk, s["speech_s"], s["segments_available"], s["reference_s"],
                             " ".join(f"{g['len_s']:.1f}({len(g['pieces'])})" for g in s["segments"])))
    sel = {"wav_dir": str(wav_dir), "rttm_dir": str(rttm_dir), "n_conv": len(convs), "n_seg": a.n_seg, "min_sec": a.min_sec,
           "max_sec": a.max_sec, "ref_sec": a.ref_sec, "conversations": convs}
    (out / "selection.json").write_text(json.dumps(sel, indent=1), encoding="utf-8")
    print(f"{len(convs)} conversations, {2 * len(convs)} speakers, {a.n_seg} source segments each -> {out}")
    print(f"{'speaker':18s} {'speech':>7s} {'avail':>5s} {'ref s':>6s}  segment lengths in s (pieces joined)")
    for r in spk_rows:
        print(f"{r[0]:18s} {r[1]:7.0f} {r[2]:5d} {r[3]:6.1f}  {r[4]}")


# ---------------------------------------------------------------- run ---------------------------------------------------
def run(a):
    import torch
    import torchaudio
    import soundfile as sf
    import anonymize
    out = Path(a.out)
    sel = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    speakers = [s for c in sel["conversations"] for s in c["speakers"]]
    if a.limit_speakers:
        speakers = speakers[: a.limit_speakers]
    n_seg = a.limit_segments or sel["n_seg"]
    ids = [s["id"] for s in speakers]
    pairs = list(itertools.permutations(range(len(ids)), 2)) if a.pairs == "ordered" else list(itertools.combinations(range(len(ids)), 2))
    batch = max(1, min(a.batch, a.n_noise))
    groups = [list(range(a.n_noise))[i:i + batch] for i in range(0, a.n_noise, batch)]    # noises denoised together in one DiT pass

    # the engine exactly as the pipeline builds it (profile = checkpoint, config, steps, guidance, prompt length)
    sys.argv = ["anonymize.py", "--profile", a.profile, "--seedvc-root", a.seedvc_root, "--wav-dir", str(out), "--out-dir", str(out / "_unused"),
                "--log-dir", str(out / "logs"), "--tag", "pairs", "--mode", "whole", "--no-warmup"] + (["--models-dir", a.models_dir] if a.models_dir else []) \
               + (["--ref-max-sec", str(a.ref_max_sec)] if a.ref_max_sec else [])
    args = anonymize.parse_args()
    anonymize.apply_profile(args)
    if args.model == "v2":
        sys.exit("this experiment is for the Seed-VC V1 engine (pipelines combined and L)")
    svc = anonymize.bootstrap_seedvc(args.seedvc_root)
    eng = anonymize.SeedVCEngine(svc, args)
    cfm = eng.model.cfm
    steps, cfg = eng.steps, eng.cfg
    n_mels = cfm.in_channels
    T_max = eng.max_context_window
    print(f"engine: sr={eng.sr} hop={eng.hop} steps={steps} cfg={cfg} n_mels={n_mels} prompt<= {eng.ref_max_sec}s  max window {T_max} frames; "
          f"batch {batch} noise(s) per DiT pass ({2 * batch} sequences with guidance)")

    # the fixed noises (full length, cut to each conversion's length)
    noises, seeds = [], []
    for j in range(a.n_noise):
        g = torch.Generator(device="cpu").manual_seed(a.noise_seed + j)
        noises.append(torch.randn(1, n_mels, T_max, generator=g)); seeds.append(a.noise_seed + j)
    np.savez(out / "noises.npz", **{f"z{j + 1}": z[0].numpy().astype("float32") for j, z in enumerate(noises)},
             seeds=np.asarray(seeds), n_mels=n_mels, frames=T_max,
             note=np.asarray("x_0 of a conversion = z_j[:, prompt_frames : prompt_frames + source_frames] (the model sees z_j[:, :prompt_frames+frames])"))
    noises = [z.to(eng.device) for z in noises]
    state = {"js": [0], "windows": []}

    def inference(self, mu, x_lens, prompt, style, f0, n_timesteps, temperature=1.0, inference_cfg_rate=0.5):
        js, T = state["js"], mu.size(1)
        nb = len(js)
        z = torch.cat([noises[j][:, :, :T] for j in js], 0) * temperature            # the fixed noises, one per batch element
        mu, prompt = mu.expand(nb, -1, -1).clone(), prompt.expand(nb, -1, -1).clone()  # same content, prompt and voice for all
        style, x_lens = style.expand(nb, -1).clone(), x_lens.expand(nb).clone()
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate)

    def solve_euler(self, x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate=0.5):
        t = t_span[0]
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        if self.zero_prompt_speech_token:
            mu[..., :prompt_len] = 0
        keep = lambda v: v[:, :, prompt_len:].detach().float().cpu().numpy().astype("float16")     # (batch, n_mels, source frames)
        vel, vc, vu = [], [], []
        for step in range(1, len(t_span)):
            dt = t_span[step] - t_span[step - 1]
            if inference_cfg_rate > 0:
                stacked = self.estimator(torch.cat([x, x], 0), torch.cat([prompt_x, torch.zeros_like(prompt_x)], 0), torch.cat([x_lens, x_lens], 0),
                                         t.reshape(1).expand(2 * x.size(0)), torch.cat([style, torch.zeros_like(style)], 0),
                                         torch.cat([mu, torch.zeros_like(mu)], 0))
                d_cond, d_unc = stacked.chunk(2, dim=0)
                dphi_dt = (1.0 + inference_cfg_rate) * d_cond - inference_cfg_rate * d_unc
            else:
                d_cond = d_unc = dphi_dt = self.estimator(x, prompt_x, x_lens, t.reshape(1).expand(x.size(0)), style, mu)
            vel.append(keep(dphi_dt))
            if a.branches:
                vc.append(keep(d_cond)); vu.append(keep(d_unc))
            x = x + dt * dphi_dt
            t = t + dt
            x[:, :, :prompt_len] = 0
        w = {"velocity": np.stack(vel), "t": t_span.detach().float().cpu().numpy(), "prompt_frames": int(prompt_len),
             "final": keep(x)}                                                                   # velocity (steps, batch, n_mels, frames)
        if a.branches:
            w["v_cond"], w["v_uncond"] = np.stack(vc), np.stack(vu)
        state["windows"].append(w)
        return x
    type(cfm).inference = inference
    type(cfm).solve_euler = solve_euler

    def convert_batch(w16, ref, js):
        """SeedVCEngine.convert with the DiT run on len(js) noises at once (same content, prompt and voice); returns one
        16 kHz numpy signal per noise, assembled from the windows exactly as the engine does it."""
        state["js"], state["windows"] = js, []
        nb = len(js)
        src = torchaudio.functional.resample(w16, 16000, eng.sr)
        S_alt = eng.semantic(w16)
        mel = eng.to_mel(src.float())
        target_lengths = torch.LongTensor([int(mel.size(2) * eng.length_adjust)]).to(eng.device)
        cond = eng.model.length_regulator(S_alt, ylens=target_lengths, n_quantizers=3, f0=None)[0]
        mel2, style2, prompt_condition = ref["mel2"], ref["style2"], ref["prompt_condition"]
        max_source_window = eng.max_context_window - mel2.size(2)
        processed, chunks, previous = 0, [[] for _ in js], [None] * nb
        while processed < cond.size(1):
            chunk_cond = cond[:, processed:processed + max_source_window]
            is_last = processed + max_source_window >= cond.size(1)
            cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
            with torch.autocast(device_type=eng.device.type, dtype=torch.float16 if eng.fp16 else torch.float32):
                vc_target = eng.model.cfm.inference(cat_condition, torch.LongTensor([cat_condition.size(1)]).to(eng.device),
                                                    mel2, style2, None, eng.steps, inference_cfg_rate=eng.cfg)
                vc_target = vc_target[:, :, mel2.size(-1):]
            vc_wave = eng.vocoder_fn(vc_target.float()).reshape(nb, -1)
            ovl = eng.overlap_wave_len
            for i in range(nb):
                w = vc_wave[i]
                if processed == 0:
                    if is_last:
                        chunks[i].append(w.cpu().numpy())
                    else:
                        chunks[i].append(w[:-ovl].cpu().numpy()); previous[i] = w[-ovl:]
                elif is_last:
                    chunks[i].append(eng.svc.crossfade(previous[i].cpu().numpy(), w.cpu().numpy(), ovl))
                else:
                    chunks[i].append(eng.svc.crossfade(previous[i].cpu().numpy(), w[:-ovl].cpu().numpy(), ovl)); previous[i] = w[-ovl:]
            processed += vc_target.size(2) - eng.overlap_frame_len
            if is_last:
                break
        outs = []
        for i in range(nb):
            o = np.concatenate(chunks[i]).astype("float32")
            o16 = torchaudio.functional.resample(torch.from_numpy(o)[None].to(eng.device), eng.sr, 16000)[0].cpu().numpy()
            outs.append(anonymize.fit_length(o16, w16.size(-1)))
        return outs

    torch.set_grad_enabled(False)                     # as the pipeline runs: no autograd graphs over the 30 steps
    cuda = eng.device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    refs = {s["id"]: eng.prepare_ref(out / "refs" / f"{s['id']}.wav") for s in speakers}
    print(f"{len(refs)} reference voices encoded (prompt {refs[ids[0]]['prompt_s']} s = {refs[ids[0]]['prompt_frames']} frames)")

    if a.verify:                                       # the batched loop against the engine's own convert, on the first pair
        A, B = speakers[pairs[0][0]], speakers[pairs[0][1]]
        x, _ = sf.read(str(out / "sources" / A["id"] / "seg1.wav"), dtype="float32")
        w16 = torch.from_numpy(x)[None].to(eng.device)
        state["js"], state["windows"] = [0], []
        y_eng = eng.convert_to_16k(w16, refs[B["id"]], {})
        v_eng = [w["velocity"][:, 0] for w in state["windows"]]
        y_b1 = convert_batch(w16, refs[B["id"]], [0])[0]
        v_b1 = [w["velocity"][:, 0] for w in state["windows"]]
        y_bn = convert_batch(w16, refs[B["id"]], list(range(min(2, a.n_noise))))
        v_bn = [w["velocity"] for w in state["windows"]]
        d1 = float(np.abs(y_eng - y_b1).max()); d2 = float(np.abs(y_eng - y_bn[0]).max())
        dv1 = max(float(np.abs(p.astype("float32") - q.astype("float32")).max()) for p, q in zip(v_eng, v_b1))
        dv2 = max(float(np.abs(p.astype("float32") - q[:, 0].astype("float32")).max()) for p, q in zip(v_eng, v_bn))
        dz = float(np.abs(y_bn[0] - y_bn[-1]).max()) if len(y_bn) > 1 else float("nan")
        rel = lambda p, q: float(np.abs(p.astype("float32") - q.astype("float32")).mean() / np.abs(p.astype("float32")).mean() * 100)
        rv2 = max(rel(p, q[:, 0]) for p, q in zip(v_eng, v_bn))
        rf2 = rel(y_eng, y_bn[0])
        print(f"verify {A['id']} -> {B['id']} seg1: engine vs batched loop (batch 1): max |dy| = {d1:.2e}, max |dv| = {dv1:.2e}; "
              f"batch {len(y_bn)}: max |dy| = {d2:.2e} (mean rel {rf2:.2f} %), max |dv| = {dv2:.2e} (mean rel {rv2:.2f} %); "
              f"z1 vs z2 audio max |dy| = {dz:.3f}  (|y| <= 1)")

    (out / "pairs").mkdir(exist_ok=True)
    index = out / "index.csv"
    new = not index.exists()
    fh = open(index, "a", newline="", encoding="utf-8")
    iw = csv.writer(fh)
    if new:
        iw.writerow(["source", "reference", "segment", "noise", "src_len_s", "frames", "steps", "windows", "batch", "wall_s", "path"])
    total = len(pairs) * n_seg * a.n_noise
    done = 0
    t_start = time.time()
    for ia, ib in pairs:
        A, B = speakers[ia], speakers[ib]
        d = out / "pairs" / f"{A['id']}__to__{B['id']}"
        d.mkdir(exist_ok=True)
        for seg in A["segments"][:n_seg]:
            x = None
            for js in groups:
                paths = [d / f"seg{seg['k']}_z{j + 1}.npz" for j in js]
                done += len(js)
                if all(p.exists() for p in paths):
                    continue
                if x is None:
                    x, _ = sf.read(str(out / "sources" / A["id"] / f"seg{seg['k']}.wav"), dtype="float32")
                    w16 = torch.from_numpy(x)[None].to(eng.device)
                t1 = time.time()
                ys = convert_batch(w16, refs[B["id"]], js)
                wall = time.time() - t1
                for i, (j, p) in enumerate(zip(js, paths)):
                    sf.write(str(p.with_suffix(".wav")), ys[i], SR, subtype="PCM_16")
                    meta = {"source": A["id"], "reference": B["id"], "segment": seg["k"], "source_file": A["id"].rsplit("_", 1)[0],
                            "source_start_s": seg["start_s"], "source_len_s": seg["len_s"], "noise": f"z{j + 1}", "noise_seed": seeds[j],
                            "steps": steps, "cfg_rate": cfg, "n_mels": n_mels, "frames_per_s": eng.sr / eng.hop,
                            "prompt_frames": state["windows"][0]["prompt_frames"], "reference_prompt_s": refs[B["id"]]["prompt_s"],
                            "pipeline_profile": a.profile, "solver": "Euler, linear t 0->1 (seed-vc BASECFM.solve_euler)", "batch": len(js),
                            "windows": len(state["windows"]), "window_overlap_frames": eng.overlap_frame_len if len(state["windows"]) > 1 else 0,
                            "reconstruct": "x_0 = noises.npz z_j[:, prompt_frames : prompt_frames + frames]; x_{k+1} = x_k + (t[k+1]-t[k]) * velocity[k]; final = x_steps"}
                    arrays = {"meta": np.asarray(json.dumps(meta)), "t": state["windows"][0]["t"]}
                    if len(state["windows"]) == 1:
                        w = state["windows"][0]
                        arrays["velocity"], arrays["final"] = w["velocity"][:, i], w["final"][i]
                        if a.branches:
                            arrays["v_cond"], arrays["v_uncond"] = w["v_cond"][:, i], w["v_uncond"][:, i]
                    else:
                        for k, w in enumerate(state["windows"]):
                            arrays[f"w{k}_velocity"], arrays[f"w{k}_final"] = w["velocity"][:, i], w["final"][i]
                            if a.branches:
                                arrays[f"w{k}_v_cond"], arrays[f"w{k}_v_uncond"] = w["v_cond"][:, i], w["v_uncond"][:, i]
                    np.savez(p, **arrays)
                    frames = sum(w["velocity"].shape[-1] for w in state["windows"])
                    iw.writerow([A["id"], B["id"], seg["k"], f"z{j + 1}", seg["len_s"], frames, steps, len(state["windows"]), len(js),
                                 round(wall / len(js), 3), str(p.relative_to(out))])
                fh.flush()
                if done % 50 < len(js) or done == total:
                    el = time.time() - t_start
                    mem = f"  GPU peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB" if cuda else ""
                    print(f"[{done}/{total}] {A['id']} -> {B['id']} seg{seg['k']} z{'+'.join(str(j + 1) for j in js)}  {wall:.2f}s  "
                          f"elapsed {el / 60:.1f} min, left ~{el / done * (total - done) / 60:.0f} min{mem}", flush=True)
    fh.close()
    size = sum(p.stat().st_size for p in (out / "pairs").rglob("*.npz")) / 1e9
    mem = f"; GPU peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB" if cuda else ""
    print(f"done: {total} conversions, {len(pairs)} pairs x {n_seg} segments x {a.n_noise} noises (batch {batch}); "
          f"velocities {size:.1f} GB in {out / 'pairs'}; index.csv{mem}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--wav-dir", required=True); s.add_argument("--rttm-dir", required=True); s.add_argument("--out", required=True)
    s.add_argument("--candidates", default=None, help="file with recording ids to prefer (one per line)")
    s.add_argument("--exclude", nargs="*", default=[], help="recording ids never to use")
    s.add_argument("--exclude-selection", nargs="*", default=[], help="selection.json files of earlier runs whose recordings are never used")
    s.add_argument("--n-conv", type=int, default=10); s.add_argument("--n-seg", type=int, default=5)
    s.add_argument("--min-sec", type=float, default=12.0); s.add_argument("--max-sec", type=float, default=20.0)
    s.add_argument("--ref-sec", type=float, default=25.0, help="reference audio per speaker (the pipeline's prompt length caps what the model uses)")
    r = sub.add_parser("run")
    r.add_argument("--out", required=True); r.add_argument("--seedvc-root", required=True); r.add_argument("--profile", required=True)
    r.add_argument("--models-dir", default=None)
    r.add_argument("--ref-max-sec", type=float, default=None, help="prompt length in s (default: the profile's; 10 fits an 8 GB GPU)")
    r.add_argument("--pairs", choices=["unordered", "ordered"], default="unordered", help="20C2 (A<B, A into B's voice) or 20P2 (both directions)")
    r.add_argument("--n-noise", type=int, default=4); r.add_argument("--noise-seed", type=int, default=1000)
    r.add_argument("--batch", type=int, default=1, help="noises denoised together in one DiT pass (1, 2 or 4)")
    r.add_argument("--branches", action="store_true", help="also store v_cond and v_uncond (3x the size)")
    r.add_argument("--verify", action="store_true", help="first pair: compare the batched loop with the engine's own convert")
    r.add_argument("--limit-speakers", type=int, default=None, help="test: only the first N speakers")
    r.add_argument("--limit-segments", type=int, default=None, help="test: only the first N segments per speaker")
    a = ap.parse_args()
    select(a) if a.cmd == "select" else run(a)


if __name__ == "__main__":
    main()
