# Results so far (2026-10-29 to 2026-11-05)

What has been done on the research side of the voice-anonymization work and what came out of it. Written for
someone who has not followed the sessions. `docs/HANDOVER.md` is the operational handover (paths, jobs, practicalities);
the per-folder READMEs hold the details. All numbers below are in the committed result files.

## The question

The anonymization pipeline converts speech with Seed-VC: a flow-matching diffusion transformer that starts from
Gaussian noise shaped like an 80-band mel spectrogram and moves it in 30 Euler steps to the converted mel, steered by
the source content (XLS-R features) and the target voice (10 s mel prompt + CAM++ speaker embedding); BigVGAN turns
the final mel into audio. We want to know what happens along those 30 steps: how the state moves, and at which step
the source speaker's identity and the spoken content become readable in the output.

## Experiment 1: saving the trajectories (done 2026-10-29)

`experiments/trajectory/save_trajectories.py` runs the pipeline unchanged on whole DISPLACE-M recordings and saves every
intermediate mel state (31 states x 80 bands x frames, float16) of every diarized turn, optionally with the two
guidance branches of the velocity. Verified: saving changes nothing (audio bit-identical) and the saved velocities
reproduce the states to 0.5 %.

| run (cluster `~/anon-bench/trajectory/runs/`) | content | size |
|---|---|---|
| `combined_seed0` | states of 4,671 turns from 20 recordings (682 min of audio) | 17 GB |
| `combined_seed0_vel` | the same plus v_cond / v_uncond | 48 GB |

Cost on the RTX PRO 6000: 0.62 s GPU per 20 s segment, 8.5 MB of velocities per segment.

## Experiment 2: speaker-pair trajectories and MSE per step (done 2026-11-04)

The colleague's protocol: pick conversations, diarize them into speakers, take 5 source segments of 12-20 s per
speaker and a reference clip, convert every segment of speaker A into the voice of speaker B for all unordered
pairs, with the same 4 fixed Gaussian noises starting every conversion, and save the per-step velocities.
`experiments/trajectory/pair_experiment.py` does the selection and the run, `experiments/trajectory/pair_mse.py` the analysis.

| set | conversations | speakers | pairs | conversions | stored |
|---|---|---|---|---|---|
| `pairs_combined` | 10 | 20 | 190 | 3,800 | 26 GB |
| `pairs_combined_2` | 5 new ones | 10 | 45 | 900 | 6.9 GB |

MSE_k = mean over the mel of (x_k - x_30)^2 for k = 0..30, x_k rebuilt from the saved velocities
(`experiments/trajectory/results/README.md`, `experiments/trajectory/results/<set>/mse/`):

| step k | 0 | 5 | 10 | 15 | 20 | 25 | 29 | 30 |
|---|---|---|---|---|---|---|---|---|
| set 1 mean (3,800) | 46.05 | 31.61 | 20.18 | 11.34 | 5.04 | 1.26 | 0.050 | 5e-6 |
| set 2 mean (900) | 54.05 | 37.06 | 23.67 | 13.30 | 5.91 | 1.48 | 0.059 | 6e-6 |
| (1 - t)^2 | 1 | 0.694 | 0.444 | 0.250 | 0.111 | 0.028 | 0.0011 | 0 |

Findings:

1. **The sampling path is a straight line in mel space.** MSE_k = 0.985 (1 - t)^2 MSE_0; per conversion
   MSE_15 / MSE_0 = 0.246 +- 0.001 for all 4,700 conversions (exactly straight = 0.250). Every Euler step moves the
   state the same distance in the same direction; nothing wanders and nothing is corrected late.
2. **A trace of the starting noise survives.** MSE_0 - mean(x_30^2) = 0.3 instead of the 1.0 that independence
   would give: the final mel is correlated (about 0.05) with its own starting noise. This is the audible difference
   between the z1 and z2 outputs of one source.
3. **The starting noise does not change the curve** (z1..z4 within 0.7 %), and **the spread between conversions is
   loudness** (corr 1.000 between MSE_0 and the energy of the final mel). Normalizing by MSE_0 collapses both sets
   onto one line.
4. **Batching two noises per DiT pass changes nothing** (within 0.07 %); GPU peak 1.9 GB.
5. Step grids (`experiments/trajectory/figures/step_grids/`): steps 1-5 noise, 6-15 coarse layout (pauses, word boundaries), 16-25
   harmonic striations (pitch and timbre of the target), 26-30 refinement.

## Experiment 3: at which step does speaker identity / content appear? (Aspect 1, done 2026-11-05)

`experiments/aspect1_mi/` (spec `docs/specs/aspect1_mi.md`). For 120 DISPLACE-M source speakers (1,796 utterances of 4-8 s), each utterance
is converted into a target voice drawn independently from a disjoint pool of 36 speakers, with 3 noise seeds
(5,388 outputs). At steps k = 0, 3, 6, 10, 15, 20, 25, 30 the model's one-step estimate of the clean result,
xhat = x_t + (1 - t) v, is vocoded and embedded with four independent encoders (CAM++ and WavLM-SV for speaker,
Whisper-small and XLS-R for content). On each embedding a probe is trained to identify the source speaker (or, for
content, to match the output to its original among 188 candidates) on held-out utterances; its score, converted to
nats, is a lower bound on the mutual information between the step and the speaker (or content). Controls: the same
probes on the original recordings (ceiling), on vocoded pure noise (floor), with shuffled labels (null), and
speaker bootstraps (CI). Details and all tables: `experiments/aspect1_mi/README.md` section 7; files
`experiments/aspect1_mi/results/displace/results/`.

Speaker identification among 120 speakers (linear probe, held-out utterances; chance 0.8 %):

| audio | CAM++ (speaker encoder) | WavLM-SV | Whisper (content encoder) |
|---|---|---|---|
| original recordings | 92.6 % (4.50 nats) | 66.8 % (3.54) | 38.4 % (2.17) |
| output, step 0 | 13.0 % (0.65) | 3.5 % (0.12) | 15.4 % (0.97) |
| output, step 3 | 16.8 % (0.97) | 3.5 % (0.24) | 18.4 % (1.10) |
| output, step 6 | 19.5 % (1.16) | 5.8 % (0.37) | 19.1 % (1.16) |
| output, steps 10-30 | 19-20 % (1.14-1.18) | 6-7 % (0.38-0.39) | 18-19 % (1.15-1.16) |
| vocoded pure noise | 0.9 % (0.00) | 1.5 % (0.00) | 1.1 % (0.00) |

Findings:

1. **The voice is removed from the first step on; what remains is speaker-correlated information in the content
   channel.** On real recordings the speaker encoder is far ahead of the content encoder (93 % vs 38 %). On every
   output they land at the same 19 %. Removing from the CAM++ embedding everything the Whisper embedding of the same
   clip explains (`experiments/aspect1_mi/residual_check.py`) leaves 51 % accuracy on the originals and chance (1-2 %) on the
   outputs at every step: the part of identity only a speaker encoder reads, the timbre, is gone. The 1.2 nats
   that survive are things a content encoder also sees: on DISPLACE, the topic (each speaker talks about their own
   things), speaking rate and prosody carried by the XLS-R content condition. Telling topic from prosody needs a
   crossed design (same sentences for every speaker, e.g. VCTK); the code supports it unchanged.
2. **The residue enters through the conditioning and is in place by step 6.** The step-0 estimate is the model's
   prediction from the condition alone and already carries half of it (0.65 nats); steps 3-6 add the rest; steps
   6-30 add nothing. The later steps render the target's timbre, consistent with the straight path of experiment 2.
3. **Content is complete from step 0.** Whisper InfoNCE 4.88 of the original's 5.21 nats at step 0, 4.94-4.99
   afterwards (0.9 nats of this is clip duration: vocoded noise of the right length scores 0.94).
4. **The raw mel state's first and second moments carry nothing** at any step (speaker -0.02 nats; content 0.06-0.25,
   mostly duration).
5. **No step has less leakage at equal utility** (`leak_vs_util.png`): stopping the sampler early is not a privacy knob.

Estimator history: the first analysis (kept as `results_v1`) had over-confident probes (52 of 368 estimates above
their ceiling, negative bounds). Fixed on 2026-11-05 with a validation split for early stopping and temperature
calibration, InfoNCE batches over distinct utterances, KSG per seed; the rerun has 0 violations and nulls at zero.
Remaining footnotes: the target-conditioned rows carry a constant -1.4 nats offset from a biased plug-in entropy
(their accuracies are unchanged, so the target voice tells nothing about the source); MLP probes under-train on the
outputs and sit below the linear ones; InfoNCE nulls use 10 shuffles (p >= 0.09).

## What this means for the anonymization product

- Seed-VC removes the source timbre at the very first denoising step; the speaker-identification residue in the
  output (about a quarter of what the speaker encoder reads on the original, in nats) is not timbre but
  content-side information that comes in through the content condition. Reducing it is a matter of what the
  content features carry (prosody, rate), not of the sampler.
- The number of Euler steps is not a privacy lever: the path is straight and every step adds target detail only.
- Noise seeds change the output audibly but not its information content; the seed is a free parameter.

## Where things are

| what | repo | cluster / laptop |
|---|---|---|
| trajectory code and results | `experiments/trajectory/`, `experiments/trajectory/results/` | `~/anon-bench/trajectory/{runs,pairs_combined,pairs_combined_2}` (data) |
| Aspect 1 code and results | `experiments/aspect1_mi/`, `experiments/aspect1_mi/results/displace/` | `~/anon-bench/aspect1_mi/runs/displace` (gen, features); laptop `C:\anon-local\aspect1_displace` (features) |
| pipeline snapshot the experiments run on | `pipeline/` (Voice-Anonymization 370b5c9) | |
| handover, jobs, practicalities | `docs/HANDOVER.md` | |

Audio, npz, models and datasets are never committed.
