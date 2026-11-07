"""D7 / first mechanism test: contract-and-noise the content condition, run the frozen generator, embed the outputs.

    python sweep.py gen  --manifest m.csv --metadata ASPECT1/metadata.csv --audit-features COND_AUDIT/features.npz --out OUT \
                         --seedvc-root R --profile configs/pipelines/combined.env [--per-speaker 5] [--gammas 1,0.5,0] [--sigmas 0,0.5,1,2] [--lda-k 32]
    python sweep.py feat --out OUT --manifest m.csv --seedvc-root R

Mechanism (framework Section 5.2, Step 2, with the removable basis = the WHOLE space, as the condition audit showed
that a k-dimensional identity subspace leaves the identity in its complement):
    h'_t = h_t - (1 - gamma) (hbar - mu) + sigma * eps,     eps ~ N(0, I_1024), one draw per utterance (shared by all
                                                             frames, so the frame deviations h_t - hbar are untouched)
with hbar the utterance mean of the engine's XLS-R condition H, mu the population mean of hbar over the TRAINING
speakers of the speaker-disjoint split (condition audit), sigma = sigma_rel * tau, tau = sqrt(tr Sigma_psi / 1024)
(the RMS per-dimension spread of the utterance means). Control configuration `lda<k>`: the same contraction and noise
restricted to the k leading LDA identity directions (what a low-dimensional mechanism would do).
Baseline gamma = 1, sigma = 0 is the unchanged pipeline (checked against the Aspect 1 step-30 output of the same
utterance, same target, same noise seed).
For every (utterance, configuration) the final mel is vocoded to OUT/gen/<config>/<utt_id>.wav; all configurations of
one utterance share the starting noise and go through the DiT in one batch. The certificate term of Proposition 5.1,
0.5 * sum_i log(1 + gamma^2 lambda_i / sigma^2) over the eigenvalues of Sigma_psi, is written per configuration.
Stage feat: CAM++, WavLM-SV and Whisper-encoder embeddings of every output and of the originals -> OUT/features/.
"""
import argparse, csv, io, json, math, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next((d / "pipeline" for d in HERE.parents if (d / "pipeline" / "src" / "anonymize.py").exists()), None) or (Path(os.environ["ANON_ROOT"]) if os.environ.get("ANON_ROOT") else next(d for d in HERE.parents if (d / "src" / "anonymize.py").exists()))   # pipeline copy in this repo, else ANON_ROOT, else a parent folder
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(HERE.parent / "aspect1_mi"))
SR = 16000


def read_csv(p):
    rows = list(csv.DictReader(open(p, newline="", encoding="utf-8")))
    return [{k.strip(): (v or "").strip() for k, v in r.items()} for r in rows]


def load_wav16(path):
    import soundfile as sf, torch, torchaudio
    x, sr = sf.read(io.BytesIO(Path(path).read_bytes()), dtype="float32", always_2d=True); x = x.mean(1)
    if sr != SR:
        x = torchaudio.functional.resample(torch.from_numpy(x)[None], sr, SR)[0].numpy()
    return x


def select_rows(man, per_speaker):
    by = {}
    for r in man:
        by.setdefault(r["src_spk"], []).append(r)
    return [r for s in sorted(by) for r in by[s][:per_speaker]]


def configs(gammas, sigmas, lda_k):
    cf = []
    for g in gammas:
        for s in sigmas:
            cf.append(dict(name=f"all_g{g:g}_s{s:g}", basis="all", gamma=g, sigma_rel=s))
    if lda_k:
        cf.append(dict(name=f"lda{lda_k}_g0_s0", basis="lda", gamma=0.0, sigma_rel=0.0))
        cf.append(dict(name=f"lda{lda_k}_g0_s1", basis="lda", gamma=0.0, sigma_rel=1.0))
    # the baseline must be present and first
    cf.sort(key=lambda c: (c["name"] != "all_g1_s0",))
    assert cf[0]["name"] == "all_g1_s0", "gammas must contain 1 and sigmas must contain 0 (the baseline)"
    return cf


def mechanism_stats(audit_features, split_seed, lda_k):
    """mu, eigenvalues of Sigma_psi, tau and the LDA directions, all from the TRAINING speakers of the speaker-disjoint split."""
    from mi.estimators import split_by_group
    F = np.load(audit_features)
    hbar, S = F["hbar"].astype(np.float64), F["src_spk"]
    tr_s, te_s = split_by_group(S, 0.3, seed=split_seed)
    mu = hbar[tr_s].mean(0)
    X = hbar[tr_s] - mu
    sv = np.linalg.svd(X, compute_uv=False)
    lam = np.zeros(hbar.shape[1]); lam[: len(sv)] = sv ** 2 / (len(X) - 1)        # eigenvalues of Sigma_psi (1024, zero-padded)
    tau = math.sqrt(lam.sum() / hbar.shape[1])
    Q = None; lam_q = None
    if lda_k:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(X, S[tr_s])
        Q, _ = np.linalg.qr(lda.scalings_[:, : lda_k])
        lam_q = np.linalg.eigvalsh(Q.T @ (X.T @ X / (len(X) - 1)) @ Q)
    return dict(mu=mu.astype(np.float32), lam=lam, tau=tau, Q=None if Q is None else Q.astype(np.float32), lam_q=lam_q,
                n_train_speakers=len(set(S[tr_s])), n_train_utts=len(tr_s))


def certificate(c, st):
    """Controlled term of Proposition 5.1 in nats for configuration c (inf when gamma > 0 and sigma = 0)."""
    g, s = c["gamma"], c["sigma_rel"] * st["tau"]
    lam = st["lam"] if c["basis"] == "all" else st["lam_q"]
    if g == 0:
        return 0.0
    if s == 0:
        return float("inf")
    return float(0.5 * np.sum(np.log1p(g * g * lam / (s * s))))


def stage_gen(a):
    import torch, torchaudio, soundfile as sf
    import anonymize
    out = Path(a.out); (out / "gen").mkdir(parents=True, exist_ok=True)
    man = read_csv(a.manifest)
    ui_of = {r["utt_id"]: i for i, r in enumerate(man)}                      # index in the FULL manifest = Aspect 1 noise index
    sel = select_rows(man, a.per_speaker)[: a.limit]
    meta_a1 = {r["utt_id"]: r for r in read_csv(a.metadata) if r["seed"] == "0"}
    gammas = [float(g) for g in a.gammas.split(",")]; sigmas = [float(s) for s in a.sigmas.split(",")]
    cf = configs(gammas, sigmas, a.lda_k)
    st = mechanism_stats(a.audit_features, a.split_seed, a.lda_k)
    for c in cf:
        c["sigma_abs"] = c["sigma_rel"] * st["tau"]; c["certificate_nats"] = certificate(c, st)
    np.savez(out / "mechanism.npz", mu=st["mu"], lam=st["lam"], tau=st["tau"], Q=st["Q"] if st["Q"] is not None else np.zeros(0), lam_q=st["lam_q"] if st["lam_q"] is not None else np.zeros(0))
    json.dump(dict(configs=cf, tau=st["tau"], n_train_speakers=st["n_train_speakers"], n_train_utts=st["n_train_utts"], per_speaker=a.per_speaker,
                   n_utts=len(sel), split_seed=a.split_seed, noise_base=a.noise_base, mechanism="h'_t = h_t - (1-gamma)(hbar-mu) + sigma eps, eps ~ N(0, I) per utterance"),
              open(out / "configs.json", "w", encoding="utf-8"), indent=1)
    for c in cf:
        (out / "gen" / c["name"]).mkdir(exist_ok=True)
    print(f"{len(sel)} utterances ({a.per_speaker} per speaker), {len(cf)} configurations; tau = {st['tau']:.4f} (RMS spread of hbar over {st['n_train_utts']} training utterances of {st['n_train_speakers']} speakers)")
    for c in cf:
        print(f"    {c['name']:14s} basis={c['basis']:3s} gamma={c['gamma']:<4g} sigma={c['sigma_abs']:.4f} ({c['sigma_rel']:g} tau)  certificate {c['certificate_nats']:.2f} nats")

    sys.argv = ["anonymize.py", "--profile", a.profile, "--seedvc-root", a.seedvc_root, "--wav-dir", str(out), "--out-dir", str(out / "_unused"),
                "--log-dir", str(out / "logs"), "--tag", "mech", "--mode", "whole", "--no-warmup"] + (["--models-dir", a.models_dir] if a.models_dir else [])
    args = anonymize.parse_args(); anonymize.apply_profile(args)
    svc = anonymize.bootstrap_seedvc(args.seedvc_root)
    eng = anonymize.SeedVCEngine(svc, args)
    cfm = eng.model.cfm
    STEPS, cfg, n_mels, T_max = eng.steps, eng.cfg, cfm.in_channels, eng.max_context_window
    torch.set_grad_enabled(False)
    dev = eng.device
    state = {}

    def inference(self, mu, x_lens, prompt, style, f0, n_timesteps, temperature=1.0, inference_cfg_rate=0.5):
        """Stock inference with the starting noise injected (same noise for every item of the batch)."""
        B, T = mu.size(0), mu.size(1)
        z = state["noise"][None, :, :T].expand(B, -1, -1).clone()
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate)

    def solve_euler(self, x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate=0.5):
        """Stock Euler solver made batch-safe (the time step and x_lens expanded to the batch), as in Aspect 1's generate.py."""
        t = t_span[0]
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x); prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        if self.zero_prompt_speech_token:
            mu[..., :prompt_len] = 0
        for step in range(1, len(t_span)):
            dt = t_span[step] - t_span[step - 1]
            if inference_cfg_rate > 0:
                stacked = self.estimator(torch.cat([x, x], 0), torch.cat([prompt_x, torch.zeros_like(prompt_x)], 0), torch.cat([x_lens, x_lens], 0),
                                         t.reshape(1).expand(2 * x.size(0)), torch.cat([style, torch.zeros_like(style)], 0),
                                         torch.cat([mu, torch.zeros_like(mu)], 0))
                d_cond, d_unc = stacked.chunk(2, dim=0)
                dphi_dt = (1.0 + inference_cfg_rate) * d_cond - inference_cfg_rate * d_unc
            else:
                dphi_dt = self.estimator(x, prompt_x, x_lens, t.reshape(1).expand(x.size(0)), style, mu)
            x = x + dt * dphi_dt
            t = t + dt
            x[:, :, :prompt_len] = 0
        return x
    type(cfm).inference = inference; type(cfm).solve_euler = solve_euler

    mu_t = torch.from_numpy(st["mu"]).to(dev)
    Q_t = torch.from_numpy(st["Q"]).to(dev) if st["Q"] is not None else None
    refs = {}
    meta_p = out / "metadata.csv"; new = not meta_p.exists()
    done = set()
    if not new:
        done = {(r["config"], r["utt_id"]) for r in read_csv(meta_p)}
    fh = open(meta_p, "a", newline="", encoding="utf-8"); mw = csv.writer(fh)
    if new:
        mw.writerow(["config", "utt_id", "src_spk", "target_spk", "gamma", "sigma_rel", "sigma_abs", "basis", "wav", "noise_seed", "shift_norm"])
    t0 = time.time(); n_done = 0; skipped = 0; base_diff = []
    for j, row in enumerate(sel):
        u = row["utt_id"]
        if all((c["name"], u) in done for c in cf):
            continue
        m = meta_a1.get(u)
        if m is None:
            skipped += 1; print(f"    skip {u}: not in the Aspect 1 metadata"); continue
        ref_path = m["target_ref_path"]
        if ref_path not in refs:
            refs[ref_path] = eng.prepare_ref(Path(ref_path))
        ref = refs[ref_path]
        x = load_wav16(row["wav"]); w16 = torch.from_numpy(x)[None].to(dev)
        src = torchaudio.functional.resample(w16, SR, eng.sr)
        H = eng.semantic(w16)                                                   # (1, L, 1024)
        hbar = H.mean(1, keepdim=True)                                          # (1, 1, 1024)
        mel = eng.to_mel(src.float())
        target_lengths = torch.LongTensor([int(mel.size(2) * eng.length_adjust)]).to(dev)
        ui = ui_of[u]
        g_eps = torch.Generator(device="cpu").manual_seed(a.eps_base + ui)
        eps = torch.randn(1, 1, H.size(-1), generator=g_eps).to(dev)
        conds, shifts = [], []
        for c in cf:
            d = (1.0 - c["gamma"]) * (hbar - mu_t)
            if c["basis"] == "all":
                Hp = H - d + c["sigma_abs"] * eps
            else:
                d_q = (d @ Q_t) @ Q_t.T; e_q = (eps @ Q_t) @ Q_t.T          # contraction and noise inside the k-dim subspace only
                Hp = H - d_q + c["sigma_abs"] * e_q
            shifts.append(float((Hp.mean(1) - hbar[:, 0]).norm()))
            conds.append(eng.model.length_regulator(Hp, ylens=target_lengths, n_quantizers=3, f0=None)[0])
        if ref["mel2"].size(2) + conds[0].size(1) > T_max:
            skipped += 1; print(f"    skip {u}: {conds[0].size(1)} frames + prompt exceed one window"); continue
        B = len(cf)
        cat = torch.cat([torch.cat([ref["prompt_condition"], c_], dim=1) for c_ in conds], dim=0)   # (B, P+T, 384)
        g = torch.Generator(device="cpu").manual_seed(a.noise_base + ui)      # = Aspect 1 seed r = 0 of this utterance
        state["noise"] = torch.randn(n_mels, T_max, generator=g).to(dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16 if eng.fp16 else torch.float32):
            vc = cfm.inference(cat, torch.LongTensor([cat.size(1)] * B).to(dev), ref["mel2"].expand(B, -1, -1).clone(), ref["style2"].expand(B, -1).clone(), None, STEPS, inference_cfg_rate=cfg)
            vc = vc[:, :, ref["mel2"].size(-1):]
        wav = eng.vocoder_fn(vc.float()).reshape(B, -1)
        w16o = torchaudio.functional.resample(wav, eng.sr, SR).cpu().numpy()
        for i, c in enumerate(cf):
            p = out / "gen" / c["name"] / f"{u}.wav"
            y = anonymize.fit_length(w16o[i], w16.size(-1))
            sf.write(str(p), y, SR, subtype="PCM_16")
            mw.writerow([c["name"], u, row["src_spk"], m["target_spk"], c["gamma"], c["sigma_rel"], f"{c['sigma_abs']:.5f}", c["basis"], str(p), a.noise_base + ui, f"{shifts[i]:.4f}"])
            if i == 0 and a.aspect1_gen:
                q = Path(a.aspect1_gen) / f"{u}__r0__k30.wav"
                if q.exists():
                    ya = load_wav16(q); n_ = min(len(ya), len(y))
                    base_diff.append((float(np.abs(ya[:n_] - y[:n_]).max()), float(np.corrcoef(ya[:n_], y[:n_])[0, 1])))
        fh.flush(); n_done += 1
        if n_done % 25 == 0 or j + 1 == len(sel):
            el = time.time() - t0
            print(f"[{j + 1}/{len(sel)}] {el / 60:.1f} min ({el / n_done:.2f} s per utterance x {B} configs), left ~{el / n_done * (len(sel) - j - 1) / 60:.0f} min", flush=True)
    fh.close()
    if base_diff:
        bd = np.array(base_diff)
        print(f"baseline vs Aspect 1 r0 step-30 output on {len(bd)} utterances: max |dy| median {np.median(bd[:, 0]):.3g} (max {bd[:, 0].max():.3g}), waveform corr median {np.median(bd[:, 1]):.4f}")
    print(f"done: {n_done} utterances generated, {skipped} skipped; GPU peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB" if dev.type == "cuda" else f"done: {n_done} generated, {skipped} skipped")


def stage_feat(a):
    import torch
    from features import Wavlm, Campplus, Whisper
    out = Path(a.out); fdir = out / "features"; fdir.mkdir(exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = read_csv(out / "metadata.csv")
    man = {r["utt_id"]: r for r in read_csv(a.manifest)}
    rows = [dict(kind="gen", config=r["config"], utt_id=r["utt_id"], src_spk=r["src_spk"], wav=r["wav"]) for r in meta]
    for u in sorted({r["utt_id"] for r in meta}):
        rows.append(dict(kind="orig", config="orig", utt_id=u, src_spk=man[u]["src_spk"], wav=man[u]["wav"]))
    with open(fdir / "rows.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} rows ({len(meta)} outputs + {len(rows) - len(meta)} originals), device {dev}", flush=True)
    for name in a.encoders.split(","):
        t0 = time.time()
        enc = {"wavlm": lambda: Wavlm(dev), "campplus": lambda: Campplus(dev, a.seedvc_root), "whisper": lambda: Whisper(dev)}[name]()
        emb = np.zeros((len(rows), enc.dim), np.float32)
        for i, r in enumerate(rows):
            emb[i] = enc(load_wav16(r["wav"]))
            if (i + 1) % 2000 == 0:
                print(f"    {name} [{i + 1}/{len(rows)}] {(time.time() - t0) / 60:.1f} min", flush=True)
        np.savez(fdir / f"{name}.npz", emb=emb)
        print(f"{name}: {emb.shape} in {(time.time() - t0) / 60:.1f} min", flush=True)
        del enc; torch.cuda.empty_cache() if dev == "cuda" else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["gen", "feat"])
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True); ap.add_argument("--seedvc-root", required=True)
    ap.add_argument("--metadata", default=None, help="Aspect 1 metadata.csv (targets)"); ap.add_argument("--audit-features", default=None, help="condition_audit features.npz")
    ap.add_argument("--profile", default=None); ap.add_argument("--models-dir", default=None)
    ap.add_argument("--aspect1-gen", default=None, help="Aspect 1 gen/ dir, to check the baseline against its step-30 outputs")
    ap.add_argument("--per-speaker", type=int, default=5); ap.add_argument("--gammas", default="1,0.5,0"); ap.add_argument("--sigmas", default="0,0.5,1,2")
    ap.add_argument("--lda-k", type=int, default=32, help="0 disables the subspace control configurations")
    ap.add_argument("--split-seed", type=int, default=0); ap.add_argument("--noise-base", type=int, default=50000); ap.add_argument("--eps-base", type=int, default=777000)
    ap.add_argument("--encoders", default="campplus,wavlm,whisper"); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.stage == "gen":
        assert a.metadata and a.audit_features and a.profile, "gen needs --metadata, --audit-features, --profile"
        stage_gen(a)
    else:
        stage_feat(a)


if __name__ == "__main__":
    main()
