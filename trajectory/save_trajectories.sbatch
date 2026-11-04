#!/bin/bash
#SBATCH --job-name=anon_traj
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --output=slurm_%x_%j.out
#
# THE ENTIRE SAMPLING TRAJECTORY of the voice-conversion model, for a set of recordings, with the normal pipeline output.
#   cd ~/anon-bench && sbatch trajectory/save_trajectories.sbatch                                  # combined, default files
#   sbatch --export=ALL,PIPELINE=L trajectory/save_trajectories.sbatch                             # pipeline L
#   sbatch --export=ALL,VELOCITIES=1 trajectory/save_trajectories.sbatch                           # + both velocity branches (3x size)
#   sbatch --export=ALL,FILES=my_ids.txt,WAV_DIR=$HOME/anon-bench/data/wav trajectory/save_trajectories.sbatch
# Inputs : WAV_DIR (16 kHz wavs), RTTM_DIR (diarization, <id>.rttm), FILES (one id per line; default: the 20 recordings
#          of the colleague's samples folder that exist in WAV_DIR)
# Output : $BENCH_ROOT/trajectory/runs/<TAG>/traj/<id>/seg*.npz, index.csv, refs/   anonymized audio in .../anon/
set -uo pipefail
export BENCH_ROOT="${BENCH_ROOT:-$HOME/anon-bench}"
PIPE="$BENCH_ROOT"
export PIPE MODELS_DIR="${MODELS_DIR:-$BENCH_ROOT/models}"
PIPELINE="${PIPELINE:-combined}"
PROFILE="$PIPE/configs/pipelines/$PIPELINE.env"
[ -f "$PROFILE" ] || { echo "ERROR: no profile $PROFILE"; exit 1; }
WAV_DIR="${WAV_DIR:-$BENCH_ROOT/data/wav}"
SEED="${SEED:-0}"
VELOCITIES="${VELOCITIES:-0}"
TAG="${TAG:-${PIPELINE}_seed${SEED}$([ "$VELOCITIES" = 1 ] && echo _vel)}"
OUT="$BENCH_ROOT/trajectory/runs/$TAG"; mkdir -p "$OUT/logs"
SEEDVC="$BENCH_ROOT/repos/seed-vc"; ENV_SVC="$BENCH_ROOT/envs/seedvc"
# diarization: the first folder that exists, unless RTTM_DIR is given
if [ -z "${RTTM_DIR:-}" ]; then
  for d in "$BENCH_ROOT/runs/full_$PIPELINE/rttm_hyp" "$BENCH_ROOT/runs/full_combined/rttm_hyp" "$BENCH_ROOT/data/rttm"; do
    [ -n "$(ls -A "$d" 2>/dev/null)" ] && { RTTM_DIR="$d"; break; }
  done
fi
[ -d "${RTTM_DIR:-}" ] || { echo "ERROR: no RTTM folder; set RTTM_DIR"; exit 1; }
# files: given list, or the colleague's 20 recordings
if [ -z "${FILES:-}" ]; then
  FILES="$OUT/files.txt"
  printf "%s\n" 2190472 2215699 2392867 2508371 2577003 2751425 3017365 3228806 3229783 3303284 \
                3334594 3340696 3385730 3413937 3433042 3519376 3563865 3785595 3801058 3900331 > "$OUT/files_wanted.txt"
  : > "$FILES"
  while read -r id; do [ -f "$WAV_DIR/$id.wav" ] && echo "$id" >> "$FILES" || echo "    not in $WAV_DIR: $id"; done < "$OUT/files_wanted.txt"
fi
[ -s "$FILES" ] || { echo "ERROR: no files to run ($FILES is empty)"; exit 1; }

source "$PIPE/env/conda_env.sh" || exit 1
export HF_HOME="$BENCH_ROOT/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_CACHE="$SEEDVC/checkpoints/hf_cache"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
activate_env "$ENV_SVC"
echo "=== trajectories PIPELINE=$PIPELINE seed=$SEED velocities=$VELOCITIES files=$(wc -l < "$FILES") rttm=$RTTM_DIR out=$OUT host=$(hostname) job=${SLURM_JOB_ID:-local} $(date -Is)"
df -h "$BENCH_ROOT" | tail -1 | awk '{print "    free space: " $4}'

python "$PIPE/trajectory/save_trajectories.py" --traj-out "$OUT/traj" --seed "$SEED" $([ "$VELOCITIES" = 1 ] && echo --velocities) -- \
    --seedvc-root "$SEEDVC" --profile "$PROFILE" --wav-dir "$WAV_DIR" --files "$FILES" --rttm-dir "$RTTM_DIR" \
    --mode diarized --out-dir "$OUT/anon" --log-dir "$OUT/logs" --tag "traj_$TAG" \
    --ref-dir "$BENCH_ROOT/refs" --lang auto --ref-exclude "" --granularity segment --loudness match \
    --non-speech attenuate --attenuate-db 20 \
    2>&1 | tee "$OUT/logs/run.log" | grep -E '^\[.*\] \[[0-9]+/|trajectories:|SUMMARY|FAILED|Error|Traceback' | cut -c1-200
echo
python "$PIPE/trajectory/inspect_trajectory.py" "$OUT/traj" > "$OUT/traj_summary.txt" 2>&1; head -n 25 "$OUT/traj_summary.txt"
du -sh "$OUT/traj"
echo "=== done $(date -Is).  trajectories: $OUT/traj   anonymized audio: $OUT/anon"
