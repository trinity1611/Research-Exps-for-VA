# Condition-space audit (framework D4, D6, D10-context, D15, D8)

How much source-speaker identity is in the **content condition**, the only source-derived input the frozen Seed-VC
generator receives, and where in it. Companion of Aspect 1 (`../aspect1_mi/`), which measured the output; this
measures the input. Framework: `docs/framework/voice_anonymization_framework.pdf`, Section 6.

| file | what |
|---|---|
| `extract.py` | GPU stage: for every source utterance of the Aspect 1 manifest, the engine's own XLS-R content features H (layer 12, 1024-d, 50/s, saved per utterance as float16), their utterance mean `hbar`, std over frames `hstd`, mean frame-to-frame change `hdelta`; the DiT's length-regulated condition `mu` (384-d, from the Aspect 1 `cond/` files), mean and std; duration, pause fraction (energy VAD), a Whisper-small Hindi transcript and characters per second |
| `analyze.py` | CPU stage, same calibrated probes as Aspect 1 (`../aspect1_mi/mi/`): D4 speaker probes on every view of the condition, LDA identity coordinate vs k on held-out speakers with the residual; D6 duration/rate-only probes; D10 transcript-only probe (TF-IDF, SVD-256); D15 intrinsic dimension, local tangents, curvature, within/between ratio; D8 cosine ASV (ignorant, lazy-informed) on the Aspect 1 step-30 outputs |
| `condition_audit.sbatch` | `cd ~/anon-bench && sbatch condition_audit/condition_audit.sbatch`; variables `TAG`, `ASPECT1_TAG`, `STAGES`, `LIMIT`, `N_BOOT`, `EPOCHS`, `NO_ASR` |
| `results/displace/` | results of jobs 16428 (smoke, 60 utterances) and 16429 (full run, 2026-11-07): `results/*.csv`, `summary.md`, `d4_identity_vs_k.png`, `table.csv` (per-utterance scalars and transcripts), logs |

Run on DISPLACE (job 16429, n2, extract 29 min for 1,796 utterances incl. ASR, analyze 8 min; `H/` = 1.1 GB on the
cluster). Splits are the Aspect 1 ones (closed set: utterance-disjoint, seed 0, 539 test utterances; speaker-disjoint:
36 test speakers), so every number is directly comparable with `../aspect1_mi/results/displace/`.

## Results (2026-11-07)

**D4a, identity in the condition (closed set, 120 speakers, H(S) = 4.79 nats, chance 0.8 %), linear probe:**

| view of the condition | dim | I_hat (nats) | CI | accuracy |
|---|---|---|---|---|
| `hbar` utterance mean of H | 1024 | 3.89 | 3.67-4.10 | 78.3 % |
| `hstd` std over frames | 1024 | 3.05 | 2.73-3.37 | 59.6 % |
| `hdelta` mean frame-to-frame change | 1024 | 3.30 | 3.02-3.59 | 63.8 % |
| `mu_mean` DiT condition, mean | 384 | 3.84 | 3.65-4.03 | 74.6 % |
| `mu_mean + mu_std` | 768 | 4.01 | 3.80-4.23 | 79.0 % |
| for comparison: CAM++ on the original audio (Aspect 1) | 192 | 4.50 | | 92.6 % |
| for comparison: CAM++ on the anonymized output, step 30 (Aspect 1) | 192 | 1.16 | 1.00-1.33 | 19.2 % |

MLP probes are 0.1-0.2 nats lower everywhere. The condition carries nearly the whole identity into the generator;
the frozen DiT discards about 2.7 of the 3.9 nats by itself.

**D4b, identity coordinate (LDA on 84 speakers, probes on the 36 held-out speakers, H = 3.58; full `hbar` there: 3.29
nats, 90.7 %):**

| k | LDA coordinate | residual after removing it | PCA coordinate | PCA residual |
|---|---|---|---|---|
| 1 | 0.03 (3.7 %) | 3.29 (90.7 %) | -0.04 | 3.30 |
| 4 | 0.72 (18.5 %) | 3.24 (88.9 %) | 0.42 | 3.33 |
| 8 | 1.76 (37.0 %) | 3.25 (88.3 %) | 1.23 | 3.30 |
| 16 | 2.72 (74.7 %) | 3.20 (88.9 %) | 2.21 | 3.30 |
| 32 | 3.26 (92.6 %) | 3.24 (89.5 %) | 3.10 | 3.34 |
| 83 | 3.21 (86.4 %) | 3.31 (92.0 %) | 3.32 | 3.17 |

A 32-dimensional LDA coordinate fitted on other speakers captures the identity of unseen speakers almost completely,
but the orthogonal complement captures it just as well at every k: identity is **redundant across the 1024
dimensions, not concentrated in a subspace**. `d4_identity_vs_k.png` shows both curves.

**D6, duration and rate only:** nothing (duration -0.02, chars/s -0.02, pause fraction 0.02 linear / 0.17 MLP, all
four together 0.03 nats; accuracies 0.2-1.7 %). log chars/s: mean 2.57, variance 0.31. The rate proxy depends on
the transcripts (see D10).

**D10, transcript-only floor:** 0.02 nats (2.4 %) with character 1-3-grams, 0 with word unigrams. **Not usable yet**:
Whisper-small in Hindi fell into repetition loops on 34.5 % of the clips (620 of 1,796 transcripts are a single
word repeated, e.g. "तो तो तो ..."), so the transcript carries far less content than the audio. The context floor
must be redone with a decoding guard (`no_repeat_ngram_size`, repetition penalty) or a larger Whisper, or with the
Whisper audio embedding as the framework specifies.

**D15, geometry of the utterance means (standardized, 1796 x 1024):**

| quantity | value |
|---|---|
| intrinsic dimension, TwoNN / Levina-Bickel MLE (k 10-20) | 27.3 / 21.1 |
| intrinsic dimension of the 120 speaker means (TwoNN) | 12.2 |
| PCA components for 90 % / 95 % of the variance | 209 / 346 |
| local PCA dimension for 90 % at 150 anchors (50 neighbours), median (p10-p90) | 35 (34-36) |
| rotation of the 10-d local tangent between neighbouring anchors, mean (p90) | 53° (61°) |
| local tangent vs global LDA-10 / global PCA-10 subspace, mean | 73° / 56° |
| random 10-d subspace vs global PCA-10 (reference for "unrelated") | 85° |
| within/between-speaker cosine-distance ratio R | 1.40 |

Low intrinsic dimension (about 20-27) but 209 linear components for 90 % of the variance, and tangents that rotate
by 53° between neighbours: the set is low-dimensional and **curved**, not a flat subspace. Caveat: the noise floor
of the local-tangent estimate (two random halves of one neighbourhood) is not computed yet; add it before quoting
the 53°.

**D8, cosine ASV on the step-30 outputs (enrolment = speaker centroid of the training utterances; 1,617 trials, 120
speakers):**

| encoder | attack | EER | top-1 |
|---|---|---|---|
| WavLM-SV | original vs original (reference, 539 trials) | 8.5 % | 57.7 % |
| WavLM-SV | ignorant (enrol original, trial anonymized) | 48.2 % | 1.6 % |
| WavLM-SV | lazy-informed (enrol anonymized, trial anonymized) | 48.5 % | 1.5 % |
| CAM++ | original vs original (reference) | 1.8 % | 91.3 % |
| CAM++ | ignorant | 42.7 % | 2.9 % |
| CAM++ | lazy-informed | 41.1 % | 3.8 % |

WavLM-SV (independent) is at chance on the outputs; CAM++ (the pipeline's own style encoder) is slightly below
chance. The trained linear probe of Aspect 1 (19 %) is a stronger attacker than cosine scoring (3-4 %).

## What it means for the mechanism (framework Section 5.2)

1. A mechanism that contracts a k-dimensional identity coordinate psi of `hbar` removes essentially nothing: the
   residual keeps 3.2 of 3.3 nats. The removable basis R_tau must be the whole complement of the protected span, so
   the contraction acts on all of `hbar - mu` minus the preserved directions (and possibly on `hstd` / `hdelta`, which
   carry 3.0-3.3 nats on their own). Proposition 5.1 then has Sigma_psi of dimension about 1,000; the bound still
   holds, but Assumption A6's "low-dimensional theta" does not describe the data.
2. The generator already removes about 2.7 nats of the condition's identity. The question for D7 is which part of
   the condition the DiT passes through, i.e. how output identity (CAM++ probe) and content (Whisper InfoNCE) change
   when `hbar` is contracted toward the population mean with noise, over a (gamma, sigma) grid. This needs no
   subspace from D4 and is the first direct test of the mechanism.
3. The rate branch (Step 3) is not needed on this data (D6), subject to better transcripts.
4. By the Section 5.3 rule, the curvature in D15 triggers D16 (local vs global coordinates).
