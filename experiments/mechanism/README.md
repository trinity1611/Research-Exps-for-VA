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
