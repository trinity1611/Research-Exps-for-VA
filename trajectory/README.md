# Sampling trajectories of the voice-conversion model

The anonymization pipelines (combined and L) convert speech with Seed-VC, a diffusion transformer trained with flow
matching. For every piece of speech it converts, it starts from random noise shaped like a mel spectrogram and moves it
in a fixed number of Euler steps (30 for combined, 25 for L) to the converted speech, steered by the content of the
original speech and by the target voice. A **trajectory** is that sequence of mel spectrograms, from step 0 (noise) to
the last step (the speech the vocoder turns into audio). `example_trajectory.png` shows steps 0, 7, 15, 23 and 30 of one
turn.

## Files

| file | what it does |
|---|---|
| `save_trajectories.py` | runs `src/anonymize.py` unchanged and saves the full trajectory of every converted segment |
| `save_trajectories.sbatch` | the same on the cluster for a set of recordings |
| `inspect_trajectory.py` | summary of a run, per-step numbers of one segment, picture of a few steps |

Saving the trajectory does not change the anonymization: with the same seed the audio is bit-for-bit identical.

## Run

```bash
cd ~/anon-bench && sbatch trajectory/save_trajectories.sbatch                    # combined, seed 0
sbatch --export=ALL,PIPELINE=L trajectory/save_trajectories.sbatch               # pipeline L
sbatch --export=ALL,VELOCITIES=1 trajectory/save_trajectories.sbatch             # + both velocity branches
python trajectory/inspect_trajectory.py trajectory/runs/combined_seed0/traj --png example.png
```

## Format

One `.npz` per converted segment (one diarized turn): `traj/<recording>/seg<NNNN>_<speaker>.npz`

| key | shape | meaning |
|---|---|---|
| `w<j>_states` | (steps+1, n_mels, frames) | x_0 (noise) ... x_steps (converted speech) of window j; float16 |
| `w<j>_t` | (steps+1,) | the flow time of each state, 0 to 1 |
| `w<j>_v_cond` | (steps, n_mels, frames) | with `VELOCITIES=1`: the predicted direction with content and target voice |
| `w<j>_v_uncond` | (steps, n_mels, frames) | with `VELOCITIES=1`: the same without them (guidance branch) |
| `w<j>_style` | (192,) | the target voice's speaker embedding given to the model |
| `meta` | json string | recording, speaker, target voice, start and length of the turn, context, frames per second, window offsets, steps, guidance |

- The step actually taken is `(1 + cfg) * v_cond - cfg * v_uncond = (x_{k+1} - x_k) / dt`, so the states alone already
  contain the guided direction; the two branches are only needed to separate content/voice guidance from the rest.
- A turn is converted together with 3 s of context on each side: the frames cover `padded_start_s` to
  `start_s + len_s + 3 s` at `frames_per_s` (86.1). Long turns are split into windows (`w0`, `w1`, ...) that overlap
  by `window_overlap_frames`, starting at `window_offsets_frames`.
- The reference clip of each target voice (the prompt the model is conditioned on) is in `refs/<voice>.npy`.
- `index.csv` lists every segment with its recording, speaker, target voice, times, frames and path.
- The starting noise is fixed by the seed; another seed gives another trajectory to the same kind of result.

Size: about 0.43 MB per second of converted audio, about 1.3 MB/s with velocities (a 3-minute recording: about 110 MB,
or 330 MB with velocities).

```python
import json, numpy as np
z = np.load("traj/2508371/seg0001_1.npz")
meta = json.loads(str(z["meta"]))
states = z["w0_states"].astype("float32")      # (31, 80, frames) for combined
```

## Speaker-pair experiment (`pair_experiment.py`, `pair_experiment.sbatch`)

The protocol requested for the publication: 10 two-speaker conversations -> 20 speakers; per speaker 5 source segments
of 12-20 s (a speaker's consecutive stretches joined, with the other speaker's interjections left out, since nobody
holds the floor for 12 s in these dialogues); the speaker's remaining speech is the reference voice (the pipeline
uses the first 10 s as the prompt); 20C2 = 190 pairs, A's segments converted into B's voice (`--pairs ordered` for
both directions); 4 Gaussian noises z1..z4 drawn once at the model's longest length and cut to each conversion, so the
same values start every trajectory; saved per conversion: the velocity at every Euler step, float16, plus the audio.

```bash
sbatch trajectory/pair_experiment.sbatch                  # select + run, ~3800 conversions, ~45 min, ~25 GB
python trajectory/pair_experiment.py select --wav-dir data/wav --rttm-dir data/rttm --out trajectory/pairs_combined --candidates trajectory/colleague_ids.txt
python trajectory/pair_experiment.py run --out trajectory/pairs_combined --seedvc-root repos/seed-vc --profile configs/pipelines/combined.env
```

`pairs/<A>__to__<B>/seg<k>_z<j>.npz`: `velocity` (steps, n_mels, frames), `t` (steps+1), `final` (the converted mel),
`meta` (json: source, reference, segment, noise, seed, prompt frames, ...). `noises.npz` holds z1..z4 (n_mels x 2580
frames) and their seeds. Reconstruction: `x_0 = z_j[:, prompt_frames : prompt_frames + frames]`, then
`x_{k+1} = x_k + (t[k+1] - t[k]) * velocity[k]`; `x_steps` equals `final` (checked: 0.03 % error from float16).
Cross-gender pairs are part of the design (A and B of different gender); this is an analysis run, not the anonymization
product, which always keeps gender.

### Second set, batching and the MSE-per-step analysis

```bash
sbatch --export=ALL,TAG=pairs_combined_2,N_CONV=5,EXCLUDE_SELECTION=$HOME/anon-bench/trajectory/pairs_combined/selection.json trajectory/pair_experiment.sbatch
sbatch --export=ALL,TAG=pairs_combined,STAGES=mse trajectory/pair_experiment.sbatch      # MSE analysis of the first run
python trajectory/pair_mse.py trajectory/pairs_combined_2                                # the same by hand
```

- `select --exclude-selection <selection.json>` never reuses a recording of an earlier run; the second set takes the next
  5 qualifying conversations (10 speakers, 10C2 = 45 pairs x 5 segments x 4 noises = 900 conversions).
- `run --batch N` denoises N of the noises in one DiT pass (same content, prompt and voice; 2N sequences with guidance).
  Checked on the laptop: batch 1 reproduces the engine bit for bit; batch 2 differs from batch 1 by 0.06 % (velocities)
  and 0.2 % (audio), the float16 kernel difference. `--verify` prints that check for the first pair of a run; the
  `batch` column of `index.csv` and `meta["batch"]` record it. The job default is `BATCH=2`.
- `pair_mse.py <run>` rebuilds every trajectory and writes `mse/`: `mse_per_conversion.csv` (one row per conversion,
  `mse_00` .. `mse_30` = mean over the mel of (x_k - x_30)^2), `mse_summary.csv` (mean, std, median, p10, p90 per step),
  `mse_by_noise.csv` (per noise z1..z4) and `mse_vs_step.png`. `mse_30` is ~1e-5, the float16 storage error, not 0.
