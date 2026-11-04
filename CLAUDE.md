# Research experiments for the voice-anonymization publication (Claude Code instructions)

This repo holds the RESEARCH side of the "Voice to EMR" Phase A voice-anonymization work: analyses of the Seed-VC
flow-matching model used by the anonymization pipeline. The pipeline itself (the product) is a separate repo,
github.com/trinity1611/Voice-Anonymization, and this code depends on it (see Setup). Start every session by reading
`README.md` (the handover: what was done, where the data is, which jobs are running) and the README of the folder
you work in.

## Standing rules from the user
- Never push to GitHub on your own; ask first, every time. Never commit audio, npz, models or datasets (`.gitignore`).
- Do not propose fine-tuning or training of any model (ASR, generator, anything); do not build a gender classifier.
- Gender preservation in the anonymization product is compulsory, never an option.
- Experiments run on the cluster, not on the laptop; the laptop is for plumbing tests only (8 GB GPU).
- Large data stays outside OneDrive: laptop `C:\anon-local`, cluster `~/anon-bench`.
- Research work goes in this repo; pipeline/benchmark work goes in the pipeline repo. Ask which one a new task is.
- The listening-test audio (mental-health study) is sensitive: it is never copied here or published.
- Reply style: plain, short, answer first; the user often asks yes/no questions and wants the yes/no first.

## Setup the scripts need (not in this repo)
- The pipeline repo checked out, with `src/anonymize.py`, `configs/pipelines/combined.env` (profile: XLS-R tiny DiT,
  BigVGAN v2 22 kHz fmax-8k, 30 steps, cfg 1.0, 10 s prompt, 3 s context) and `configs/models/`.
- Seed-VC checkout (`repos/seed-vc` on the cluster, `C:\anon-local\seed-vc` on the laptop) with `campplus_cn_common.bin`
  and the Hugging Face cache `checkpoints/hf_cache` (XLS-R 300m, whisper-small, wavlm-base-plus-sv, BigVGAN).
- Model folder with `seedvc_stock/DiT_uvit_tat_xlsr_ema.pth` (`--models-dir`; cluster `~/anon-bench/models`, laptop `C:\anon-local\exp6_9`).
- Python env: the Seed-VC conda env (torch 2.7 cu128, torchaudio, transformers 4.46, soundfile, scipy, scikit-learn,
  pandas, pyarrow, matplotlib). Laptop: `C:\Users\Hp\miniconda3\envs\seedvc\python.exe`; cluster: `~/anon-bench/envs/seedvc`.
- How the scripts find the pipeline: `ANON_ROOT=<pipeline root>` if set, else the nearest parent folder containing
  `src/anonymize.py` (the laptop clone sits inside the pipeline project folder; on the cluster the code is copied into
  `~/anon-bench/<folder>` so the parent is the root).
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
- Uploading this code: `scp -r <folder> udit.tatia@10.16.63.40:~/anon-bench/` so that `~/anon-bench/trajectory` and
  `~/anon-bench/aspect1_mi` mirror the folders here.

## Layout
- `README.md`: handover for a new session (status, paths, pending jobs, numbers).
- `trajectory/`: saving Seed-VC sampling trajectories; the speaker-pair protocol (`pair_experiment.py`), MSE per step
  (`pair_mse.py`), step-grid plots. Results on the cluster: `~/anon-bench/trajectory/pairs_combined{,_2}`.
- `aspect1_mi/`: per-denoising-step mutual information (speaker, content), spec in `Aspect 1.md`; tests in `tests/`.
- Result files (csv, json, png, md) are committed; everything heavy is ignored.
