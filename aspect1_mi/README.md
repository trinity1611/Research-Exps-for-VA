# Aspect 1: per-denoising-step mutual information (speaker S, content C) in Seed-VC

Implementation of `Research Exps/Aspect 1.md`. Everything here is a LOWER BOUND of the mutual information in nats,
evaluated on held-out data; see "Caveats" at the end. Status 2026-11-05: the DISPLACE run is done (job 15902,
generation + features; analysis rerun with calibrated estimators as job 16117) and read in section 7.

## 1. Pipeline

| stage | script | what it does |
|---|---|---|
| generation | `generate.py` | Seed-VC (the pipeline's engine, unmodified) on a manifest of source utterances; target drawn uniformly from a disjoint pool, independent of the source speaker; R noise seeds; saves `x_t`, `v`, `xhat1_t = x_t + (1 - t) v` at a step grid and vocodes every `xhat1_t` to a waveform |
| features | `features.py` | independent encoders on every step waveform and on the originals: WavLM-SV x-vector (512), CAM++ (192, Seed-VC's style encoder, second view only), Whisper-small encoder mean (768), XLS-R 300m mean (1024); `melstat` = mean/std of the raw mel state (Y^A) |
| estimators | `mi/estimators.py` | classifier bound (linear and MLP probes, same capacity at every t), InfoNCE (separable critic, cap log N), KSG on PCA-10, permutation null, speaker-level bootstrap, group splits |
| analysis | `analyze.py` | I(S;Y_t), I(S;Y_t\|T), I(S;Y_t\|C), I(C;Y_t) per encoder and step, ceilings on the original audio, fractions, nulls, CIs, `results.csv/.parquet`, plots |
| tests | `tests/test_estimators.py` | spec section 7: Gaussian pair with known MI, independent pair, exact discrete case, confound (fixed vs independent target), split-leak assertion, PCA+KSG. All 17 checks pass (2026-11-04) |
| cluster | `aspect1.sbatch` | gen -> feat -> analyze with `MANIFEST`, `POOL`, `TAG` variables |

## 2. Data design (section 1 of the spec) - DECIDED 2026-11-04: DISPLACE-M, the trajectory experiment's data

The user chose the data of the trajectory experiment (`data/wav`, 78 two-speaker Hindi conversations, reference
RTTMs). `make_manifest_displace.py` builds the manifest on the cluster (stage `manifest` of the job):
- 60 recordings -> 120 source speakers, up to 15 single-speaker stretches of 4-8 s each (other speaker's overlaps
  removed, pauses joined): 1,796 utterances, 189 min; 18 recordings -> 36 target-pool speakers (reference clips of
  14-30 s). Sources and pool are disjoint; the target is drawn uniformly per utterance.
- No two speakers read the same sentence, so C cannot be a shared sentence id. C is treated as CONTINUOUS: the
  Whisper-encoder embedding of the original utterance, with I(C;Y_t) from InfoNCE and KSG (speaker-disjoint split),
  exactly as the continuous S. The discrete C classifier and the C-conditioning switch themselves on automatically
  when a manifest has shared sentence ids (e.g. VCTK).
- Consequence for S: S and C are not independent by design (each speaker has their own topics), so part of a
  speaker probe's score could come from content; the S|T conditioning is available, the S|C one is not. Report this.
- Scale: 1,796 x 3 seeds = 5,388 outputs x 8 steps = 43k step waveforms of ~6 s. Estimated on the RTX PRO 6000:
  generation ~30-40 min, features ~2-3 h (Whisper pads every clip to 30 s), analysis ~1 h; disk ~6 GB npz + 8 GB wav.

Alternatives considered (a crossed grid as the spec asks) remain possible with the same code:

| option | speakers x shared sentences | where | notes |
|---|---|---|---|
| VCTK (English, CSTR, 109 speakers) | 109 x ~24 shared sentences (the elicitation paragraph + Rainbow Passage, ids 001-024) | needs download (~11 GB, login node has internet) | the standard crossed grid; the model is language-agnostic (XLS-R content) |
| Kathbath test-known (Indic) | 20 speakers / language (hi, kn, te), ~100-170 utterances each; sentence sharing across speakers UNKNOWN (check `transcription_n2w.txt` on the cluster) | on the cluster: `~/anon-bench/testkn_audio/kb_data_clean_m4a/<lang>/test_known`, transcripts `~/anon-bench/transcripts_n2w/...` | natural Indic speech, K = 20 only (H(S) = 3.0 nats); a target pool from another split is needed |
| DISPLACE-M | conversational, no shared sentences | on the cluster | not a crossed design; used only for the plumbing test |

Manifest format (any dataset): `utt_id, src_spk, sentence_id, wav[, split]`; pool: `tgt_spk, wav[, gender]`.
Target pool speakers must be disjoint from the source speakers (`generate.py` removes any overlap).

## 3. Definitions fixed in the code ([VERIFY] items of the spec, checked in the repos)

- Interpolant (`repos/seed-vc/modules/flow_matching.py`): `y_t = (1 - (1 - s) t) z + t x1`, `u = x1 - (1 - s) z`,
  `s = sigma_min = 1e-6`. Hence `x_t + (1 - t) v = x1 + s z`: the one-step clean estimate, exact up to 1e-6.
- Sampler: Euler, `t_k = k / 30`, k = 0 noise ... 30 data; guidance `v = 2 v_cond - v_uncond` (cfg 1.0, profile combined).
- Conditioning u = (`mu` content condition: XLS-R features length-regulated to mel frames, 384 ch, with the target's
  prompt content in front; the target's prompt mel 80 x 861; CAM++ style 192). Saved once per (utterance, target)
  in `cond/<utt>.npz` and `targets/<tgt>.npz`.
- Y_t^B (primary): `xhat1_t -> BigVGAN -> 16 kHz waveform -> encoders`. Y_t^A (sanity): `melstat` of raw `x_t`.
  Y_t^C (x_t, u) is NOT estimated: the conditioning contains the target voice, and the ceiling I(S; x_t, u) is
  I(S; u) + ... which the independent-target design makes equal to I(S; mu); it can be added by concatenating `mu`
  statistics to the features if wanted.
- Default step grid: k = 0, 3, 6, 10, 15, 20, 25, 30 (t = 0, .10, .20, .33, .50, .67, .83, 1). Note that
  `xhat1_0` is NOT noise: at t = 0 the model's one-step estimate is already a blurred prediction of the output
  (MSE to the final mel 5 vs 57 for the raw state), so I(S; Y_0^B) need not be ~0; the ~0 check is Y^A.
- Noise: per (utterance, seed) from `torch.Generator(seed = 50000 + 1000 r + utterance index)`; logged.
- Target draw: `numpy.default_rng(12345).integers(pool size)` per utterance in manifest order; logged in `metadata.csv`.
- Splits (`analyze.py`, seed 0, 30 % test): closed-set S classifier = utterance-disjoint (every speaker on both
  sides); open-set InfoNCE/KSG and the C classifier = speaker-disjoint. All seeds of an utterance stay together.
- Continuous s for InfoNCE/KSG = WavLM-SV embedding of the ORIGINAL utterance.
- Probes: linear softmax and 1-hidden-layer MLP (256), Adam 1e-3, weight decay 1e-4, up to 200 epochs, batch 256, at
  every t. InfoNCE: two 256-64 MLP encoders + scaled dot product, N = 512, up to 50 epochs; held-out CE averaged over
  50 random batches. KSG: k = 3 on PCA-10 of both sides (standardized), one noise seed at a time (mean over seeds).
- Calibration (added 2026-11-05 after the first DISPLACE run): 15 % of the TRAINING rows (whole utterances, all
  seeds together) are a validation set; the epoch with the lowest validation CE is kept and a softmax / score
  temperature is fitted on it. Without this the probes were over-confident on held-out data: CE > H(S) and negative
  "bounds" at 27 % closed-set accuracy (chance 0.8 %), InfoNCE at -15 nats, 52 of 368 estimates above their ceiling.
  The test rows are never used for either choice, so the values remain held-out lower bounds.
- Duplicates: the three seeds of one utterance share the source embedding s exactly. InfoNCE batches are therefore
  drawn over distinct utterances (one random seed each, `_GroupBatches`), and KSG is evaluated per seed; otherwise
  the negatives / k-NN counts contain exact copies of the positive and the estimates are wrong in both directions.
- Nulls: labels / pairings shuffled on train and test, 100 shuffles (linear probe, cond None); CIs: 200 speaker
  bootstraps of the held-out CE (the probe is not retrained per resample).
- Ceilings: the same estimator on the original audio; `frac = I_hat(Y_t) / I_hat(X)`; rows with `I_hat` above
  the ceiling by more than the CI are flagged `exceeds_ceiling` (estimator inconsistency to inspect, spec 5.3).

## 4. Running

```bash
python "Research Exps/aspect1_mi/tests/test_estimators.py"                      # synthetic checks (CPU, ~3 min)
python generate.py --manifest m.csv --pool pool.csv --out OUT --seedvc-root repos/seed-vc --profile configs/pipelines/combined.env --seeds 3 --batch 3
python features.py --out OUT --manifest m.csv --noise-control
python analyze.py --out OUT                                                      # --quick for a plumbing test
sbatch --export=ALL,MANIFEST=...,POOL=...,TAG=vctk trajectory-style job: Research\ Exps/aspect1_mi/aspect1.sbatch
```

Local plumbing test (2026-11-04, laptop GPU 1.8 GB peak): 6 utterances x 2 seeds x 8 steps = 96 step waveforms in
1.9 min; all four encoders ~0.1 s per 15 s clip; `analyze.py --quick` produces 368 estimates and the three plots.

Cost estimate for VCTK (109 speakers x 24 sentences x 3 seeds = 7,848 outputs, 8 steps each): generation ~1 h on
the RTX PRO 6000 (DiT 30 steps + 8 vocoder calls per output), 63k step waveforms of ~3 s (~6 GB), features ~1-2 h,
analysis ~1 h. Disk for the npz (x, v, xhat1 at 8 steps, float16): ~1 MB per output, ~8 GB.

## 5. Outputs

`OUT/metadata.csv`, `OUT/gen/<utt>__r<r>.npz` (+ `__k<kk>.wav`), `OUT/cond/`, `OUT/targets/`, `OUT/features/<enc>.npz`
+ `rows.csv`, `OUT/results/results.csv|parquet` (t, k, kind, encoder, estimator, probe, target, cond, I_hat, ci_lo,
ci_hi, null_mean, null_sd, null_p, ceiling, frac, exceeds_ceiling, H, CE, N, acc), `design.json`,
`mi_vs_t_S.png`, `mi_vs_t_C.png`, `leak_vs_util.png`.

## 6. Caveats to record with the results (spec section 9)

- Lower bounds: a low value means little information OR a weak probe; two speaker encoders and two content
  encoders, two probe capacities and InfoNCE/KSG are reported side by side.
- From N held-out samples nothing above ~log N can be certified (InfoNCE cap reported in `H`).
- The estimates describe what these evaluators extract; ASV EER against a stated threat model stays the privacy
  ground truth. The independent-target draw is part of the threat model (a pairing-aware attacker is another one).
- ECAPA-TDNN (speechbrain) is not installed anywhere; the second ASV view is CAM++ for now. Installing speechbrain
  on the cluster login node would add it (`features.py` gets a fourth class).

## 7. Results on DISPLACE-M (jobs 15902 + 16117, read 2026-11-05)

Files: `results/displace/results/` (`results.csv`, `design.json`, the three plots, `residual_check.txt`);
the first, uncalibrated analysis is kept in `results/displace/results_v1/` for the before/after comparison.
Run: 120 source speakers, 1,796 utterances x 3 seeds = 5,388 outputs, steps k = 0, 3, 6, 10, 15, 20, 25, 30;
closed-set split 539 test utterances (1,617 test rows), open-set split 36 test speakers; 368 estimates,
0 above their ceiling (52 in the first analysis), permutation nulls -0.017 +- 0.004 nats (p = 0.01) for every
generated row of the four waveform encoders, pure-noise rows 0.0.

Source speaker S, classifier bound, linear probe, closed set, nats (accuracy among 120 speakers, chance 0.8 %):

| Y | CAM++ | WavLM-SV | XLS-R | Whisper | melstat (raw x_t) |
|---|---|---|---|---|---|
| original audio (ceiling) | 4.50 (92.6 %) | 3.54 (66.8 %) | 3.14 (61.4 %) | 2.17 (38.4 %) | - |
| k = 0, t = 0 | 0.65 (13.0 %) | 0.12 (3.5 %) | 0.31 (5.2 %) | 0.97 (15.4 %) | -0.02 (0.9 %) |
| k = 3, t = 0.10 | 0.97 (16.8 %) | 0.24 (3.5 %) | 0.47 (8.5 %) | 1.10 (18.4 %) | -0.02 |
| k = 6, t = 0.20 | 1.16 (19.5 %) | 0.37 (5.8 %) | 0.51 (8.9 %) | 1.16 (19.1 %) | -0.02 |
| k = 10, t = 0.33 | 1.18 (19.7 %) | 0.38 (6.1 %) | 0.60 (12.2 %) | 1.16 (19.0 %) | -0.02 |
| k = 15 .. 30 | 1.14-1.17 (19-20 %) | 0.38-0.39 (6-7 %) | 0.60-0.61 (10-12 %) | 1.15-1.16 (18-19 %) | -0.02 (0.2-1 %) |
| pure noise vocoded | -0.02 (0.9 %) | -0.02 (1.5 %) | -0.03 (1.1 %) | -0.01 (1.1 %) | - |

Bootstrap CI (200 speaker resamples) at k = 30: CAM++ 1.00-1.33, WavLM 0.31-0.47. Open-set InfoNCE (N = 188 distinct
utterances per batch, cap 5.24): CAM++ 0.21 (k 0) -> 0.47 (k 6) -> 0.41 (k 30), original 2.22; WavLM 0.18 -> 0.48
-> 0.54, original 5.11. KSG on PCA-10 is ~0 for the generated outputs of the speaker encoders (not sensitive enough
at ~540 points per seed) and 2.4 / 5.0 on the originals.

Content C (Whisper embedding of the original; InfoNCE over N = 188, cap 5.24; KSG not a bound):

| Y | Whisper InfoNCE | Whisper KSG | XLS-R InfoNCE | melstat InfoNCE |
|---|---|---|---|---|
| original audio | 5.21 | 5.04 | 1.55 | - |
| k = 0 | 4.88 | 2.23 | 1.01 | 0.06 |
| k = 3 | 4.94 | 2.46 | 1.34 | 0.09 |
| k = 6 | 4.95 | 2.57 | 1.10 | 0.19 |
| k = 10 .. 30 | 4.94-4.99 | 2.65-2.71 | 1.21-1.38 | 0.23-0.25 |
| pure noise vocoded | 0.94 | 0.58 | 0.28 | - |

Reading (plain words):

1. **The voice is removed; what remains is a speaker-correlated signal that a content encoder sees just as well.**
   On the original audio the speaker encoder CAM++ identifies 93 % of the speakers (4.5 nats) and Whisper only 38 %
   (2.2 nats). On every generated output, from step 0 to step 30, both land at the same 19 % (1.16 nats). The
   residual check (`residual_check.py`, output in `results/residual_check.txt`) makes this explicit: after removing
   from the CAM++ embedding everything a ridge regression on the Whisper embedding of the same clip explains, the
   originals keep 1.86 nats (51 % accuracy) of speaker information and the outputs keep 0.00 at every step
   (1-2 % accuracy, chance 0.8 %). So the part of the speaker identity that only a speaker encoder reads (timbre)
   is gone from the first step on; the 1.2 nats that are left are readable through the content channel. On
   DISPLACE that channel includes the topic (every speaker talks about their own things), speaking rate and
   prosody carried by the XLS-R content condition. Which of these it is needs the crossed design (section 2, VCTK).
2. **Whatever leaks, leaks through the conditioning and is in place by step 6.** The one-step estimate at t = 0 is
   the model's prediction from the condition alone (content features of the source + the target's prompt), and it
   already carries 0.65 nats (CAM++); steps 3-6 add the rest (1.16) and steps 6-30 add nothing. The later steps,
   which the step grids show adding harmonic/timbre detail, do not add source-speaker information: they render the
   TARGET's timbre. This matches the MSE result (straight path: each step adds the same kind of detail).
3. **Content is complete from step 0.** Whisper InfoNCE is at 4.88 of the original's 5.21 nats at t = 0 and
   4.94-4.99 afterwards; the KSG estimate rises until k = 10 (2.23 -> 2.71) and is flat afterwards. 0.94 nats of the
   content score is clip duration (the pure-noise rows, vocoded at the utterance's length, reach 0.94), so the true
   noise floor for C is 0.9, not 0.
4. **The raw mel state x_t carries nothing readable by first and second moments** (melstat -0.02 at every step for
   S, 0.06 -> 0.25 for C, the latter again mostly duration). I(S; x_t) itself is not estimated this way; the
   one-step estimate is the usable view.
5. **Leakage/utility plot** (`leak_vs_util.png`, WavLM leak fraction vs Whisper content fraction): all steps sit at
   leak 0.03-0.11 of the original, utility 0.94-0.96; there is no step with less leakage at equal utility, so an
   early stop of the sampler is not a privacy knob.

Estimator notes for the write-up:
- `cond = T` rows (I(S; Y | T)): the accuracies are unchanged by the target label (CAM++ 19.9 % vs 19.5 % at k = 6),
  i.e. the target voice tells nothing about the source, as the independent draw guarantees. Their `I_hat` values are
  shifted by a constant -1.4 nats because the plug-in H(S | T) from 3,770 training rows over 120 x 36 cells is biased
  low (3.38 vs H(S) = 4.77); use the accuracy or `H(S) - CE`, not that column. (Fix if needed: H(S|T) = H(S) under
  the independent draw, or a bias-corrected entropy.)
- MLP probes are weaker than linear ones on the outputs (CAM++ 0.75 vs 1.16; WavLM and XLS-R ~0): with 3.8k training
  rows of a weak signal the validation early stop halts the MLP before it learns. The reported bound is the maximum
  over probes, i.e. the linear one. On the originals both probes agree (4.50 / 4.50).
- InfoNCE nulls use 10 shuffles, so their smallest p is 1/11 = 0.09; the classifier nulls use 100 (p = 0.01).
- Before/after calibration (v1 -> v2): 52 -> 0 ceiling violations, CAM++ k = 30 linear 0.63 -> 1.16 nats at the
  same ~19 % accuracy (v1 under-counted because the probe was over-confident), Whisper InfoNCE C -15 -> 4.9.
