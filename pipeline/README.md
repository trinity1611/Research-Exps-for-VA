# Pipeline snapshot

A copy of the parts of the anonymization pipeline (github.com/trinity1611/Voice-Anonymization, commit 370b5c9 of
2026-10-30, "Pitch estimate: level-normalize the clip") that the research scripts import or read. Copied 2026-11-04.
Do not edit here; if the pipeline changes, re-copy and update this note.

| path | role |
|---|---|
| `src/anonymize.py` | the engine: `parse_args`, `apply_profile`, `bootstrap_seedvc`, `SeedVCEngine` (prepare_ref, convert, convert_to_16k), `fit_length` |
| `src/common.py` | helpers used by anonymize.py (timers, jsonl writer, rttm reading) |
| `src/evaluate.py` | `SpeakerVerifier` (WavLM-SV x-vectors) and `load_16k`, used by `aspect1_mi/features.py` |
| `configs/pipelines/combined.env` | the profile the experiments run: XLS-R tiny DiT, BigVGAN v2 22 kHz fmax-8k, 30 steps, cfg 1.0, 10 s prompt, 3 s context, timbre_far refs |
| `configs/pipelines/L.env` | the earlier F0-conditioned profile (25 steps), selectable with `PIPELINE=L` |
| `configs/models/*.yml` | Seed-VC model configs (`xlsr_tiny_bigvgan_v2_8k.yml` for combined) |
| `configs/run.env` | cluster/run defaults read by the profiles |
| `refs/hi` | the 12 reference voices (6 f, 6 m) of the pipeline, needed only by `save_trajectories.sbatch` (`--ref-dir`) |
| `env/conda_env.sh`, `env/setup_envs.sh` | conda activation used by the sbatch files; environment build |
| `env/stage_models_seedvc.py` | downloads the Seed-VC models into `repos/seed-vc/checkpoints` on a machine with internet |

Not copied: the Seed-VC checkout itself, model weights, Hugging Face caches, data, results, the demo UI, the
listening-test and review sites, the dataset-evaluation scripts. The research scripts locate this folder first
(`<repo>/pipeline`), then `ANON_ROOT`, then a parent folder with `src/anonymize.py` (the cluster layout).
