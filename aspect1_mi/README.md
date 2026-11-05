# Aspect 1: per-denoising-step mutual information (speaker S, content C) in Seed-VC

Implementation of `Research Exps/Aspect 1.md`. Everything here is a LOWER BOUND of the mutual information in nats,
evaluated on held-out data; see "Caveats" at the end. Status 2026-11-04: code complete and tested on synthetic data
and on a 12-output local run; the real run waits for the dataset decision (section 2).

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
