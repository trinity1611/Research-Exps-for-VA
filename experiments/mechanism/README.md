# Mechanism sweep (framework D7 and the first test of the mechanism, Section 5.2)

Contract-and-noise the content condition of the frozen Seed-VC generator over a (gamma, sigma) grid, run the
generator, and measure how much source identity and how much content the outputs keep, next to the certificate
term of Proposition 5.1. Follows the condition audit (`../condition_audit/`), which showed that identity is spread
over the whole 1024-d condition mean, so the removable basis here is the whole space; a k-dim subspace control is
included to show the difference at the output.

| file | what |
|---|---|
| `sweep.py gen` | GPU: for 5 utterances per speaker (600 of the Aspect 1 manifest), same target and same starting noise as Aspect 1 seed 0, apply `h'_t = h_t - (1 - gamma)(hbar - mu) + sigma eps` to the engine's XLS-R condition (one Gaussian eps per utterance, shared by all frames) for every configuration, run all configurations of one utterance through the DiT in one batch, vocode, write `gen/<config>/<utt>.wav`; `mu`, the eigenvalues of Sigma_psi and the LDA directions come from the condition audit's training speakers; the baseline (gamma 1, sigma 0) is compared with the Aspect 1 step-30 output |
| `sweep.py feat` | GPU: CAM++, WavLM-SV, Whisper-encoder embeddings of every output and of the originals |
| `analyze.py` | CPU: per configuration, calibrated linear speaker probes (CAM++, WavLM-SV; closed set, utterance-disjoint), InfoNCE content bound (Whisper of the output vs Whisper of the original), Whisper cosine, CAM++ cosine to the source, certificate term; `results/sweep.csv`, `leak_vs_utility.png`, `summary.md` |
| `mechanism.sbatch` | `cd ~/anon-bench && sbatch mechanism/mechanism.sbatch`; variables `TAG`, `PER_SPEAKER`, `GAMMAS`, `SIGMAS`, `LDA_K`, `STAGES`, `LIMIT`, `N_BOOT`, `EPOCHS` |

Default grid: gamma in {1, 0.5, 0} x sigma/tau in {0, 0.5, 1, 2} (12 configurations, `all_g<gamma>_s<sigma>`), plus
`lda32_g0_s0` and `lda32_g0_s1` (contraction and noise restricted to the 32 leading LDA identity directions).
tau = sqrt(tr Sigma_psi / 1024) is the RMS per-dimension spread of the utterance means, so sigma/tau = 1 adds noise
of the size of the natural between-utterance variation. Certificate term per configuration:
0.5 sum_i log(1 + gamma^2 lambda_i / sigma^2) (inf at sigma = 0 with gamma > 0; 0 at gamma = 0).

Needs on the cluster: `aspect1_mi/runs/displace/` (manifest, metadata.csv with the targets, gen/ for the baseline
check) and `condition_audit/runs/displace/features.npz` (hbar of every utterance). Cost: 600 utterances x 14
configurations = 8,400 conversions, batched 14 per DiT pass; estimated 1-1.5 h generation, 40 min features, 15 min
analysis; about 6 GB of wav.

What to read in `results/summary.md`: the row `all_g1_s0` must match Aspect 1 (CAM++ probe about 1.2 nats, 19 %);
`orig` is the ceiling (4.5 nats). Then, per row: identity left (CAM++ and WavLM probes), content kept (Whisper
InfoNCE, cap log N), and the certificate. The frontier question (D7) is the largest sigma and smallest gamma at
which content is still at the baseline; the mechanism question is whether identity drops below the baseline at all
when the condition mean is contracted, given that the generator already removes most of it.

## Results on DISPLACE (jobs 16434 gen + feat, 16455 analysis; 2026-11-07)

Files: `results/displace/` (`results/sweep.csv`, `summary.md`, `leak_vs_utility.png`, `configs.json`, `metadata.csv`,
logs). 600 utterances (5 per speaker, 120 speakers) x 14 configurations = 8,400 conversions, 3.6 s per utterance
for all 14 (one DiT batch), GPU peak 3.5 GB; baseline vs the Aspect 1 step-30 output of the same utterance: waveform
correlation 1.000 (median), max |dy| 0.002. tau = 0.0278; mu, Sigma_psi and the LDA directions from the 84 training
speakers (1,256 utterances) of the condition audit.

Probes: trained on the Aspect 1 step-30 outputs (4,848 rows, all seeds, every utterance except the sweep's 180 test
utterances) and applied to the sweep's test outputs per configuration; the sweep-only probes (420 training rows)
were too weak to see a 1-nat signal and are kept in `sweep.csv` as `*_self`. Linear classifier bound, nats, with
speaker-bootstrap CI; InfoNCE content bound against the Whisper embedding of the original (cap log 180 = 5.19).

| config | gamma | sigma/tau | certificate (nats) | I(S;Y) CAM++ | acc | I(S;Y) WavLM | acc | I(C;Y) Whisper | Whisper cos | CAM++ cos to source |
|---|---|---|---|---|---|---|---|---|---|---|
| original audio | - | - | - | 2.65 [2.33, 3.03] | 37.8 % | 1.07 | 11.7 % | 5.14 | 1.000 | 1.000 |
| all_g1_s0 (unchanged pipeline) | 1 | 0 | inf | 1.46 [1.13, 1.85] | 25.0 % | 0.38 | 3.9 % | 4.93 | 0.986 | 0.410 |
| all_g1_s1 | 1 | 1 | 100.8 | 1.41 | 22.8 % | 0.35 | 4.4 % | 4.88 | 0.986 | 0.410 |
| all_g1_s2 | 1 | 2 | 43.4 | 1.32 | 22.8 % | 0.32 | 5.0 % | 4.83 | 0.986 | 0.409 |
| all_g0.5_s0 | 0.5 | 0 | inf | 1.37 | 25.0 % | 0.35 | 4.4 % | 4.97 | 0.986 | 0.408 |
| all_g0.5_s2 | 0.5 | 2 | 17.2 | 1.25 | 20.0 % | 0.29 | 3.9 % | 4.78 | 0.986 | 0.406 |
| all_g0_s0 (mean replaced) | 0 | 0 | 0 | 1.27 [0.97, 1.61] | 23.9 % | 0.27 | 2.8 % | 4.96 | 0.986 | 0.405 |
| all_g0_s2 | 0 | 2 | 0 | 1.16 [0.83, 1.47] | 18.3 % | 0.28 | 5.6 % | 4.89 | 0.986 | 0.404 |
| lda32_g0_s0 (32-d subspace control) | 0 | 0 | 0 | 1.40 | 25.0 % | 0.36 | 3.9 % | 4.95 | 0.986 | 0.409 |
| lda32_g0_s1 | 0 | 1 | 0 | 1.36 | 25.0 % | 0.36 | 4.4 % | 4.93 | 0.986 | 0.410 |

(Rows with sigma/tau = 0.5 are in `sweep.csv`; they sit between their neighbours. The original-audio row uses the
anonymized-trained probe, so it is not the ceiling; the ceiling with a probe trained on originals is 4.50 nats.)

Reading:

1. **Contracting the condition mean does almost nothing to the leak.** The unchanged pipeline leaves 1.46 nats
   (25 %); replacing the mean by the population mean and adding noise of twice the natural spread leaves 1.16 nats
   (18 %). The CIs overlap. The 32-direction subspace control changes nothing (1.40, 25 %), as the condition audit
   predicted at the input.
2. **The generator does not use the mean either.** Content is unchanged in every row (InfoNCE 4.8-5.0 of 5.19,
   Whisper cosine 0.986) and the output's CAM++ similarity to the source is 0.41 in every row. D7's question, how
   much perturbation of the mean the frozen DiT tolerates, has a trivial answer: all of it. The condition mean
   carries 3.9 nats of identity on the way in (condition audit) and the DiT ignores it on the way out.
3. **The leak is in the frame-level structure of the condition**, h_t - hbar, which this mechanism keeps by design
   (it is where the words are). Consistent with D3 (content-side residue) and with the condition audit (frame std
   and delta carry 3.0-3.3 nats on their own). The certificate of Section 5.2 bounds the mean coordinate, which
   contributes nothing to the leak: Lambda_res is essentially the whole leak. The certificate values themselves
   (17-218 nats) are vacuous in 1,024 dimensions.
4. **Next (D7b):** a mechanism on the frame dynamics, trading some content: vector quantization of the frames to K
   centroids (k-means on disjoint speakers, closed form; sweep K), with temporal smoothing and independent per-frame
   noise as comparison arms, measured the same way.
