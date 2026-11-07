# Research experiments for the voice-anonymization publication (Claude Code instructions)

This repo holds the RESEARCH side of the "Voice to EMR" Phase A voice-anonymization work: analyses of the Seed-VC
flow-matching model used by the anonymization pipeline. The pipeline itself (the product) is a separate repo,
github.com/trinity1611/Voice-Anonymization, and this code depends on it (see Setup). Start every session by reading
`docs/HANDOVER.md` (what was done, where the data is, which jobs are running), `docs/RESULTS.md` (the findings) and
the README of the experiment folder you work in.

## Standing rules from the user
- Never push to GitHub on your own; ask first, every time. Never commit audio, npz, models or datasets (`.gitignore`).
- Do not propose fine-tuning or training of any model (ASR, generator, anything); do not build a gender classifier.
- Gender preservation in the anonymization product is compulsory, never an option.
- Experiments run on the cluster, not on the laptop; the laptop is for plumbing tests only (8 GB GPU).
- Large data stays outside OneDrive: laptop `C:\anon-local`, cluster `~/anon-bench`.
- Research work goes in this repo; pipeline/benchmark work goes in the pipeline repo. Ask which one a new task is.
- The listening-test audio (mental-health study) is sensitive: it is never copied here or published.
- Reply style: plain, short, answer first; the user often asks yes/no questions and wants the yes/no first.

## Setup the scripts need
- `pipeline/` IN THIS REPO is a snapshot of the pipeline code the experiments run on (Voice-Anonymization commit
  370b5c9, 2026-10-30): `src/anonymize.py` (the engine), `src/common.py`, `src/evaluate.py` (WavLM-SV loader),
  `configs/pipelines/combined.env` (profile: XLS-R tiny DiT, BigVGAN v2 22 kHz fmax-8k, 30 steps, cfg 1.0, 10 s prompt,
  3 s context) and `L.env`, `configs/models/*.yml`, `refs/hi` (the 12 reference voices), `env/conda_env.sh`,
  `env/setup_envs.sh`, `env/stage_models_seedvc.py` (downloads Seed-VC models). Experiments use the combined profile only.
  If the pipeline repo changes, re-copy these files deliberately and note the commit in `pipeline/README.md`.
- How the scripts find the pipeline: `<repo>/pipeline` if it holds `src/anonymize.py` (the normal case), else
  `ANON_ROOT=<pipeline root>`, else the nearest parent folder containing `src/anonymize.py` (the cluster layout, where
  the folders of this repo are copied into `~/anon-bench` next to its own `src/`).
- Still needed on each machine, NOT in the repo: the Seed-VC checkout (`repos/seed-vc` on the cluster,
  `C:\anon-local\seed-vc` on the laptop) with `campplus_cn_common.bin` and its Hugging Face cache
  `checkpoints/hf_cache` (XLS-R 300m, whisper-small, wavlm-base-plus-sv, BigVGAN); the model folder with
  `seedvc_stock/DiT_uvit_tat_xlsr_ema.pth` (`--models-dir`; cluster `~/anon-bench/models`, laptop `C:\anon-local\exp6_9`);
  the Python env (`pipeline/env/setup_envs.sh` builds it: torch 2.7 cu128, torchaudio, transformers 4.46, soundfile,
  scipy, scikit-learn, pandas, pyarrow, matplotlib; laptop `C:\Users\Hp\miniconda3\envs\seedvc\python.exe`, cluster
  `~/anon-bench/envs/seedvc`); and the data (below). A machine with internet downloads the public models with
  `pipeline/env/stage_models_seedvc.py`; the cluster compute nodes are offline, so stage on the login node.
- Environment for a run: `HF_HUB_CACHE=<seed-vc>/checkpoints/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
- Data: DISPLACE-M dev set, `data/wav/<id>.wav` (78 two-speaker Hindi conversations, 16 kHz) and reference RTTMs
  `data/rttm/<id>_SPEAKER.rttm`; on the cluster under `~/anon-bench/data`, on the laptop in the pipeline project `data/`.

## Cluster usage
- Login `ssh udit.tatia@10.16.63.40` (password; Claude cannot run it, the user runs the commands). Project `~/anon-bench`.
  Slurm partition `normal`; GPU nodes n1 (RTX PRO 6000, 96 GB), n2 (H200), n3; the login node has no GPU and no `pip`
  on PATH (use `envs/seedvc/bin/python -m pip`). Compute nodes have no internet (everything offline from caches).
- Submit: `cd ~/anon-bench && sbatch <folder>/<job>.sbatch`; watch `squeue --me`; log `slurm_<jobname>_<id>.out`;
  `sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,NodeList`; GPU of a running job: `srun --jobid=<id> --overlap --pty nvidia-smi`.
- Files written on Windows must be CRLF-stripped on the cluster before use: `sed -i 's/\r$//' <files>`.
- Windows scp does not expand `{a,b}`; pack with tar on the cluster, then scp the archive. Many-GB outputs stay on the
  cluster; fetch samples only, into `C:\anon-local`.
- Job files read variables via `sbatch --export=ALL,VAR=value ...`; every job prints its settings in the first log line.
- Uploading this code: `scp -r experiments/<folder> udit.tatia@10.16.63.40:~/anon-bench/` so that
  `~/anon-bench/trajectory` and `~/anon-bench/aspect1_mi` mirror `experiments/trajectory` and `experiments/aspect1_mi`
  (the cluster layout is flat, next to its own `src/`; the sbatch files assume it).

## Layout
- `README.md`: short: what the repo is, folder map, how to start a session.
- `docs/HANDOVER.md`: handover for a new session (status, jobs, cluster paths, practicalities, numbers).
- `docs/RESULTS.md`: consolidated findings of all experiments, for a new reader.
- `docs/specs/`: experiment specs (`aspect1_mi.md` = the Aspect 1 spec).
- `pipeline/`: snapshot of the pipeline code (Voice-Anonymization 370b5c9), unchanged.
- `experiments/trajectory/`: experiments 1-2: saving Seed-VC sampling trajectories; the speaker-pair protocol
  (`pair_experiment.py`), MSE per step (`pair_mse.py`), step-grid plots; `figures/`; `results/` (both pair sets).
  Data on the cluster: `~/anon-bench/trajectory/{runs,pairs_combined,pairs_combined_2}`.
- `experiments/aspect1_mi/`: experiment 3: per-denoising-step mutual information (speaker, content); estimator
  library `mi/`, `tests/`, `results/displace/` (`results` calibrated, `results_v1` first analysis).
- `experiments/condition_audit/`: experiment 4 (framework D4, D6, D10, D15, D8): identity in the content condition.
- `experiments/mechanism/`: D7 / mechanism sweep: contract-and-noise the condition over a (gamma, sigma) grid, output
  identity and content vs the certificate term.
- Reserved for later work, create only when it starts: `experiments/entanglement_vctk/` (crossed design, kappa_ent), `experiments/estimation/`
  (verification / Fano floors vs attackers), `theory/` (proof notes, synthetic E1), `docs/framework/` (the proposal).
- Result files (csv, json, png, md, txt) are committed; everything heavy is ignored.
