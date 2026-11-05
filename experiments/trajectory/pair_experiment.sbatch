#!/bin/bash
#SBATCH --job-name=pairs_traj
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --output=slurm_%x_%j.out
#
# SPEAKER-PAIR TRAJECTORY EXPERIMENT: 10 conversations -> 20 speakers, 5 source segments (12-20 s) per speaker,
# 20C2 = 190 pairs (A's segments into B's voice), 4 fixed Gaussian noises z1..z4 -> 3800 conversions; the velocity
# of every Euler step is saved (trajectory/pair_experiment.py).
#   cd ~/anon-bench && sbatch trajectory/pair_experiment.sbatch
#   sbatch --export=ALL,TAG=pairs_combined_2,N_CONV=5,EXCLUDE_SELECTION=$HOME/anon-bench/trajectory/pairs_combined/selection.json \
#          trajectory/pair_experiment.sbatch                                       # 2nd set: 5 new conversations, 10C2 pairs, batch 2
#   sbatch --export=ALL,PAIRS=ordered trajectory/pair_experiment.sbatch          # both directions, 380 pairs
#   sbatch --export=ALL,BRANCHES=1 trajectory/pair_experiment.sbatch             # + v_cond / v_uncond (3x the size)
#   sbatch --export=ALL,PIPELINE=L,TAG=pairs_L trajectory/pair_experiment.sbatch
#   sbatch --export=ALL,STAGES=run trajectory/pair_experiment.sbatch              # re-run / resume with the existing selection
#   sbatch --export=ALL,TAG=pairs_combined,STAGES=mse trajectory/pair_experiment.sbatch   # only the MSE-per-step analysis of a run
# BATCH = noises denoised together in one DiT pass (2: twice the batch of the first run; 4 = all noises at once).
# Output: $BENCH_ROOT/trajectory/<TAG>/  (selection.json, sources/, refs/, noises.npz, pairs/, index.csv, mse/)
set -uo pipefail
export BENCH_ROOT="${BENCH_ROOT:-$HOME/anon-bench}"
PIPE="$BENCH_ROOT"
export PIPE MODELS_DIR="${MODELS_DIR:-$BENCH_ROOT/models}"
PIPELINE="${PIPELINE:-combined}"
PROFILE="$PIPE/configs/pipelines/$PIPELINE.env"
[ -f "$PROFILE" ] || { echo "ERROR: no profile $PROFILE"; exit 1; }
WAV_DIR="${WAV_DIR:-$BENCH_ROOT/data/wav}"
RTTM_DIR="${RTTM_DIR:-$BENCH_ROOT/data/rttm}"            # reference RTTMs; or a run's rttm_hyp for the pipeline's own diarization
CANDIDATES="${CANDIDATES:-$PIPE/trajectory/colleague_ids.txt}"
TAG="${TAG:-pairs_$PIPELINE}"
OUT="$BENCH_ROOT/trajectory/$TAG"
STAGES="${STAGES:-select run mse}"
PAIRS="${PAIRS:-unordered}"; N_NOISE="${N_NOISE:-4}"; NOISE_SEED="${NOISE_SEED:-1000}"; BRANCHES="${BRANCHES:-0}"; BATCH="${BATCH:-2}"
N_CONV="${N_CONV:-10}"; N_SEG="${N_SEG:-5}"; MIN_SEC="${MIN_SEC:-12}"; MAX_SEC="${MAX_SEC:-20}"
EXCLUDE_SELECTION="${EXCLUDE_SELECTION:-}"                # selection.json of earlier runs whose recordings must not be reused
SEEDVC="$BENCH_ROOT/repos/seed-vc"; ENV_SVC="$BENCH_ROOT/envs/seedvc"
mkdir -p "$OUT"

source "$PIPE/env/conda_env.sh" || exit 1
export HF_HOME="$BENCH_ROOT/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_CACHE="$SEEDVC/checkpoints/hf_cache"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
activate_env "$ENV_SVC"
has() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }
echo "=== pair experiment PIPELINE=$PIPELINE pairs=$PAIRS noises=$N_NOISE seed=$NOISE_SEED batch=$BATCH branches=$BRANCHES out=$OUT host=$(hostname) job=${SLURM_JOB_ID:-local} $(date -Is)"
df -h "$BENCH_ROOT" | tail -1 | awk '{print "    free space: " $4}'

if has select; then
  echo; echo "=== select  $(date -Is)"
  python "$PIPE/trajectory/pair_experiment.py" select --wav-dir "$WAV_DIR" --rttm-dir "$RTTM_DIR" --out "$OUT" \
      $([ -f "$CANDIDATES" ] && echo --candidates "$CANDIDATES") $([ -n "$EXCLUDE_SELECTION" ] && echo --exclude-selection $EXCLUDE_SELECTION) \
      --n-conv "$N_CONV" --n-seg "$N_SEG" --min-sec "$MIN_SEC" --max-sec "$MAX_SEC" \
      2>&1 | tee "$OUT/select.log"
fi
if has run; then
  echo; echo "=== run  $(date -Is)"
  python "$PIPE/trajectory/pair_experiment.py" run --out "$OUT" --seedvc-root "$SEEDVC" --profile "$PROFILE" \
      --pairs "$PAIRS" --n-noise "$N_NOISE" --noise-seed "$NOISE_SEED" --batch "$BATCH" --verify $([ "$BRANCHES" = 1 ] && echo --branches) \
      2>&1 | tee "$OUT/run.log" | grep -vE 'Warning|warn|it/s'
fi
if has mse; then
  echo; echo "=== mse  $(date -Is)"
  python "$PIPE/trajectory/pair_mse.py" "$OUT" 2>&1 | tee "$OUT/mse.log"
fi
echo; du -sh "$OUT/pairs" 2>/dev/null
echo "=== done $(date -Is).  $OUT  (index.csv, selection.json, noises.npz, pairs/, mse/)"
