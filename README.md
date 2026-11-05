# Research-Exps-for-VA

Research experiments for the voice-anonymization publication ("Voice to EMR", Phase A): analyses of the Seed-VC
flow-matching model that the anonymization pipeline (github.com/trinity1611/Voice-Anonymization) uses. The product
code is in that repo; this one holds the experiments, their results and the write-ups. Status 2026-11-05: three
experiments done (trajectory saving, speaker-pair MSE per step, per-step mutual information); findings in
`docs/RESULTS.md`.

## Folder map

| folder | what |
|---|---|
| `docs/HANDOVER.md` | handover for a new session: what was done, cluster jobs and paths, practicalities, numbers to quote |
| `docs/RESULTS.md` | the findings of all experiments so far, written for a new reader |
| `docs/specs/` | experiment specs (`aspect1_mi.md`) |
| `pipeline/` | snapshot of the pipeline code the experiments run on (Voice-Anonymization commit 370b5c9); do not edit, re-copy deliberately |
| `experiments/trajectory/` | experiments 1-2: saving Seed-VC sampling trajectories, the speaker-pair protocol, MSE per step; `figures/`, `results/` |
| `experiments/aspect1_mi/` | experiment 3: at which denoising step speaker identity and content become readable (mutual-information lower bounds); `mi/` estimators, `tests/`, `results/` |
| `CLAUDE.md` | standing rules and setup notes for Claude Code sessions |

Reserved for later work (created when it starts): `experiments/mechanism/`, `experiments/entanglement_vctk/`,
`experiments/estimation/`, `theory/`, `docs/framework/`.

## How to start a session

1. Read `docs/HANDOVER.md` (section 0 lists pending jobs), then `docs/RESULTS.md`, then the README of the
   experiment folder you will work in.
2. Code runs on the cluster (`udit.tatia@10.16.63.40`, project `~/anon-bench`, flat layout: `experiments/<folder>` is
   uploaded as `~/anon-bench/<folder>`); the laptop is for plumbing tests only. Setup and environment: `CLAUDE.md`.
3. Large data (audio, trajectories, features, models) never enters this repo: laptop `C:\anon-local`, cluster
   `~/anon-bench`. Result files (csv, json, png, md) are committed.
4. Never push without asking.
