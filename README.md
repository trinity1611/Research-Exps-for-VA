# Research experiments: reference for a new chat

Research work for the voice-anonymization publication, separate from the pipeline/benchmark work (repo
trinity1611/Voice-Anonymization). This repo: trinity1611/Research-Exps-for-VA. Large data stays outside the repo and
outside OneDrive: laptop `C:\anon-local`, cluster `~/anon-bench`. Setup and cluster usage: `CLAUDE.md`.
Written 2026-11-04, last updated 2026-11-04.

**Starting a new Claude Code session**: clone this repo anywhere (the pipeline code it needs is in `pipeline/`), open
Claude Code in the clone, and say "read README.md and continue with <task>". Section 0 lists what is pending.
`CLAUDE.md` lists what must exist on the machine besides the repo: the Seed-VC checkout with its model cache, the
stock DiT checkpoint, the Python environment, and the DISPLACE-M data.

## 0. Pending at the time of writing (2026-11-04)

| cluster job | what | output | how to read |
|---|---|---|---|
| 15892 (running) | trajectory set 2: 5 new conversations, 900 conversions, batch 2, then MSE | `~/anon-bench/trajectory/pairs_combined_2/` | `tail slurm_pairs_traj_15892.out`; `cat trajectory/pairs_combined_2/mse/mse_summary.csv` |
| 15893 (running) | MSE per step of trajectory set 1 (3,800 conversions) | `~/anon-bench/trajectory/pairs_combined/mse/` | `tail -n 40 slurm_pairs_traj_15893.out` |
| 15896 (queued) | Aspect 1 MI run on DISPLACE (manifest, 5,388 outputs, features, analysis), ~5 h | `~/anon-bench/aspect1_mi/runs/displace/results/` | `tail slurm_aspect1_mi_15896.out`; `results.csv`, `mi_vs_t_S.png`, `mi_vs_t_C.png`, `leak_vs_util.png` |

When they finish: fetch the small result files (`mse/*.csv`, `mse_vs_step.png`, `results/*`) into this repo under the
matching folder (`trajectory/results/<run>/`, `aspect1_mi/results/<tag>/`) and commit them; interpret the curves.
The listening test (anon-fuzzy.pages.dev, deadline Fri 30 Oct 2026) is pipeline work, not part of this repo.

## 1. Context in one paragraph

The anonymization pipeline (`src/anonymize.py`, profile `configs/pipelines/combined.env`) converts speech with
Seed-VC V1: a diffusion transformer trained with flow matching. For each piece of speech it starts from Gaussian noise
shaped like a mel spectrogram and moves it in 30 Euler steps (25 for pipeline L) to the converted mel, steered by the
source content (XLS-R features) and by the target voice (10 s mel prompt + CAM++ speaker embedding); the BigVGAN
vocoder then turns the final mel into audio once. A **trajectory** is that sequence of mel states x_0 (noise) ...
x_30 (result), or equivalently the 30 velocities the model predicted, since x_{k+1} = x_k + dt * v_k. The denoising
happens in **mel space** (80 bands, 22.05 kHz, hop 256 -> 86.13 frames/s), not on the waveform and not in a learned
latent. The colleague who asked for the trajectories (Kiran MK) had none in his own folder
(`/home/kiran.mk/AudioAnon/evaluation/reconstruction/ood_displace/conversation_anon/samples`, wav/json/txt only).

## 2. Folder `Research Exps/trajectory/`

| file | purpose |
|---|---|
| `README.md` | format of every output, reconstruction rule, run commands (read this first) |
| `save_trajectories.py`, `save_trajectories.sbatch` | run the pipeline unchanged on whole recordings and save the full state trajectory of every diarized turn |
| `inspect_trajectory.py` | summary / per-step table / picture of a run of `save_trajectories` |
| `pair_experiment.py`, `pair_experiment.sbatch` | the speaker-pair protocol (section 4): `select` + `run` (+ `mse` stage in the sbatch) |
| `pair_mse.py` | MSE between step 30 and every step, in mel space, for a pair run |
| `plot_pair_steps.py` | 6 x 5 grid of mel spectrograms, one panel per Euler step, for chosen conversions |
| `colleague_ids.txt` | the 20 DISPLACE-M recording ids the colleague works with |
| `example_trajectory.png`, `example_cluster_trajectory.png` | steps 0/7/15/23/30 of one turn (laptop run, cluster run) |
| `pair_trajectory_steps.png`, `step_grids/*.png` | 30-step grids of 6 conversions of the pair experiment (noise z1, segment 1) |

The scripts locate `src/anonymize.py` by walking up the parents, so they run from here and from
`~/anon-bench/trajectory/` on the cluster (cluster layout unchanged: upload them there).
None of this is in the GitHub repo (trinity1611/Voice-Anonymization) and it must not be pushed without asking.

## 3. Whole-recording trajectories (done 2026-10-29)

Cluster `udit.tatia@10.16.63.40`, `~/anon-bench/trajectory/runs/`:

| run | content | size |
|---|---|---|
| `combined_seed0/traj` | states x_0..x_30 of every turn, 20 recordings, 4671 turns, 682 min of converted audio | 17 GB |
| `combined_seed0_vel/traj` | the same + v_cond / v_uncond (guidance branches) | 48 GB |

Format: `traj/<recording>/seg<NNNN>_<speaker>.npz` with `w<j>_states (31, 80, frames) float16`, `w<j>_t`,
`w<j>_style (192,)`, optional `w<j>_v_cond/v_uncond (30, 80, frames)`, `meta` json; `refs/<voice>.npy`; `index.csv`.
Turns are converted with 3 s of context on each side; long turns are split into windows overlapping by 16 frames.
Verified: saving changes nothing (audio bit-identical), velocities reproduce the steps to 0.5 %.
Laptop sample (outside OneDrive): `C:\anon-local\trajectory_runs\{combined_seed0,combined_seed0_vel}\traj\2508371\` (2.8 GB).

Cost figures (RTX PRO 6000): 20 s segment = 0.62 s GPU (DiT 0.39, vocoder 0.20), peak 5.4 GB GPU memory;
disk per 20 s segment: states 8.5 MB (11 MB with context), velocities only 8.5 MB, both guidance branches 16.7 MB.
Estimates: 32,000 x 20 s = 270 GB (velocities) / 535 GB (branches), ~9 h one GPU stream; 8,000 x 20 s = 68 / 135 GB, ~2.3 h.

## 4. Speaker-pair experiment (colleague's protocol)

Protocol as given: 10 conversations -> 20 speakers after diarization; per speaker 5 source segments of 12-20 s
(content preserved) and a reference of any duration; 20C2 pairs x 5 segments; 4 Gaussian noises z1..z4 fixed for
every conversion; save the intermediate velocity vectors (the trajectory).

Decisions taken (state them to the colleague):
- Data: DISPLACE-M dev set, `data/wav/<id>.wav`, reference RTTMs `data/rttm/<id>_SPEAKER.rttm` (78 recordings);
  the colleague's 20 ids are preferred candidates.
- Nobody holds the floor for 12 s in these dialogues (3-4 of 78 recordings only), so a segment is the speaker's
  consecutive stretches joined, with the other speaker's interjections cut out and overlapped speech removed.
- Reference = the speaker's remaining speech (25-40 s); the pipeline uses its first 10 s as the prompt (861 frames).
- Pairs unordered, A < B, A's segments converted into B's voice (`PAIRS=ordered` gives both directions).
  Cross-gender pairs are included (analysis run; the product always keeps gender).
- Noises: seeds 1000-1003, drawn once at the longest window (80 x 2580 frames = 30 s) and cut per conversion, so
  the same values start every trajectory. x_0 of a conversion = `z_j[:, 861 : 861 + frames]`.
- Saved per conversion: `velocity (30, 80, frames) float16`, `t (31,)`, `final (80, frames)`, `meta`, and the wav.
  The 861 prompt frames are dropped (the model holds them fixed). 20.0 s sources (prompt + source > 2580 frames)
  are split in 2 windows: keys `w0_velocity`, `w0_final`, `w1_velocity`, `w1_final` (412 of 3800 in the first run).
- Reconstruction: `x_{k+1} = x_k + (t[k+1] - t[k]) * velocity[k]`, `x_30 = final` (checked: 0.027 % float16 error).

### Run 1, done 2026-10-30 (job 15285, n1, 36 min, ~0.3 s per conversion)
`~/anon-bench/trajectory/pairs_combined/` (world-readable: home and `trajectory` are o+x, `pairs_combined` o+rX;
the listening-test folders were closed with o-rwx first): `index.csv` (3800 rows: source, reference, segment,
noise, src_len_s, frames, steps, windows, wall_s, path), `selection.json`, `noises.npz`, `refs/`, `sources/`,
`pairs/<A>__to__<B>/seg<k>_z<j>.npz` + `.wav` (190 pair folders, 26 GB).
Conversations: 2190472 2215699 2392867 2508371 2577003 2751425 3017365 3228806 3229783 3303284.
Laptop sample: `C:\anon-local\trajectory_runs\pairs_combined\` (index, selection, noises, refs, sources, pair
2190472_spk0__to__2215699_spk0 complete, plus seg1_z1 of five other pairs; 220 MB).

### Run 2, submitted 2026-11-04 (jobs 15892 new set, 15893 MSE of run 1), both PENDING in the queue
Second protocol message: 5 NEW conversations (none of run 1) -> 10 speakers, 10C2 = 45 pairs x 5 segments x 4 noises
= 900 conversions; "if you are using less than 45 GB increase the batch by 2 times" (read as GPU batch: run 1 used
~5 GB at batch 1, so `--batch 2` noises per DiT pass; the alternative reading, disk < 45 GB, would be `N_CONV=10`);
then MSE between step 30 and every other step in mel space.
Selection (verified locally with the same RTTMs): 3334594 3340696 3385730 3413937 3433042.
Batching verified on the laptop: batch 1 reproduces the engine bit for bit; batch 2 differs by 0.06 % (velocities)
and 0.2 % (audio), float16 kernel differences; the job prints this check (`--verify`) and the GPU peak.
Output: `~/anon-bench/trajectory/pairs_combined_2/` (same layout, `index.csv` has a `batch` column, plus `mse/`).
Commands used:
```bash
cd ~/anon-bench
sbatch --export=ALL,TAG=pairs_combined_2,N_CONV=5,EXCLUDE_SELECTION=$HOME/anon-bench/trajectory/pairs_combined/selection.json trajectory/pair_experiment.sbatch
sbatch --export=ALL,TAG=pairs_combined,STAGES=mse trajectory/pair_experiment.sbatch     # MSE of run 1 -> trajectory/pairs_combined/mse/
```
When they finish: `tail slurm_pairs_traj_15892.out slurm_pairs_traj_15893.out`, `cat trajectory/pairs_combined_2/mse/mse_summary.csv`.

### MSE analysis (`pair_mse.py`)
Outputs in `<run>/mse/`: `mse_per_conversion.csv` (mse_00..mse_30 per conversion = mean over the mel of (x_k - x_30)^2,
plus `energy` = mean of x_30^2 for scale), `mse_summary.csv` (mean, std, median, p10, p90, min, max per step),
`mse_by_noise.csv`, `mse_vs_step.png` (log scale). On 12 local test conversions: 56 at step 0, 14 at step 15, 1.0 at
step 26, 0.06 at step 29, 6e-6 at step 30 (float16 storage, not exactly 0); the curve follows (1 - t)^2 x MSE_0
closely, i.e. the sampling path is nearly straight in mel space; the four noises give indistinguishable curves.

### What the step grids show
Steps 1-5 nearly uniform noise; steps 6-15 the coarse layout (pauses as dark vertical stripes, word boundaries,
low-to-high energy tilt) = the content condition; steps 16-25 the harmonic striations in the low bands = pitch and
timbre of the target voice; steps 26-30 refinement only.

## 5. Shapes to quote

| array | shape | meaning |
|---|---|---|
| source wav | (16000 x s,) | 12-20 s segment, 16 kHz |
| XLS-R content | (1, 50 x s, 1024) | wav2vec2 XLS-R 300m features, 50/s |
| `mu` content condition | (1, 861 + frames, 384) | resampled to mel rate, projected to 384 channels, prompt first |
| style | (1, 192) | CAM++ speaker embedding of the reference |
| prompt mel | (1, 80, 861) | the reference's first 10 s |
| noise z_j | (80, 2580) | one standard Gaussian for the longest window, cut per conversion |
| state x_k | (80, 861 + frames) | prompt region fixed, source region moves |
| velocity (saved) | (30, 80, frames) | per step, source frames only, float16 |
| final | (80, frames) | converted mel -> vocoder |

Frames = seconds x 86.13 (e.g. 14.83 s -> 1277 frames). Model: `configs/models/xlsr_tiny_bigvgan_v2_8k.yml`
(DiT hidden 384, 9 layers, 6 heads; content_dim 384; in_channels 80; vocoder nvidia/bigvgan_v2_22khz_80band_fmax8k_256x).

## 6. Practicalities

- Cluster: `udit.tatia@10.16.63.40`, project `~/anon-bench`, env `envs/seedvc/bin/python` (no system pip; nvitop is
  installed in that env, run it inside a job with `srun --jobid=<id> --overlap --pty ...`); GPU nodes n1 (RTX PRO
  6000), n2 (H200), n3; login node has no GPU. Jobs queue with reason Priority/Resources when the nodes are full.
- Files written on Windows must be CRLF-stripped before sbatch: `sed -i 's/\r$//' trajectory/*.py trajectory/*.sbatch`.
- Windows scp does not expand `{a,b}`; tar on the cluster first, then scp the archive.
- Laptop test environment: `C:\Users\Hp\miniconda3\envs\seedvc\python.exe`, `--seedvc-root C:/anon-local/seed-vc`,
  `--models-dir C:/anon-local/exp6_9`, `HF_HUB_CACHE=C:/anon-local/seed-vc/checkpoints/hf_cache`, offline flags on;
  8 GB GPU: direct engine calls need `torch.set_grad_enabled(False)` (done in the scripts) or the DiT OOMs.
- Standing rules: ask before any GitHub push; evaluation/experiment code stays out of the repo; no fine-tuning
  proposals of any kind; gender preservation is compulsory in the product; ask at the start of a task whether it is
  research-based (then it belongs here).

## 7. Aspect 1: per-denoising-step mutual information (`aspect1_mi/`, spec `Aspect 1.md`)

Implemented and tested 2026-11-04; full description in `aspect1_mi/README.md`. In one paragraph: for every source
utterance the pipeline's engine is run with a target drawn independently of the speaker and 3 noise seeds; at steps
0, 3, 6, 10, 15, 20, 25, 30 the state x_t, the velocity v and the one-step clean estimate xhat1 = x_t + (1 - t) v are
saved and xhat1 is vocoded; independent encoders (WavLM-SV, CAM++, Whisper, XLS-R) embed every step waveform and the
originals; `analyze.py` computes lower bounds I(S;Y_t), I(S;Y_t|T), I(C;Y_t) with classifier (linear/MLP), InfoNCE
and KSG estimators, ceilings on the original audio, permutation nulls and speaker-bootstrap CIs. Data: DISPLACE-M
(120 source speakers x up to 15 stretches of 4-8 s, 36 pool speakers); since no two speakers read the same sentence,
content C is continuous (Whisper embedding of the original). Synthetic tests: `aspect1_mi/tests/test_estimators.py`
(17 checks pass). Cluster job: `aspect1_mi/aspect1.sbatch` (job 15896 queued 2026-11-04).

## 8. Open points

- Read the results of jobs 15892 / 15893 (MSE tables for both sets) and 15896 (MI curves); plot grids for z2-z4 if wanted.
- The analysis goal beyond the MSE curve has not been stated by the colleague; ask before adding features.
- `PAIRS=ordered` (both directions) and `BRANCHES=1` (v_cond / v_uncond, 3x size) are available if asked.
- ECAPA-TDNN as a second ASV family needs speechbrain installed on the cluster login node (not done).
