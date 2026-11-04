#!/bin/bash
#SBATCH --job-name=aspect1_mi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --output=slurm_%x_%j.out
#
# ASPECT 1: per-denoising-step mutual information (speaker S, content C) of Seed-VC.  manifest -> gen -> feat -> analyze.
#   cd ~/anon-bench && sbatch aspect1_mi/aspect1.sbatch                              # DISPLACE-M (data/wav + data/rttm), tag displace
#   sbatch --export=ALL,TAG=displace,STAGES="feat analyze" aspect1_mi/aspect1.sbatch  # re-run later stages only
#   sbatch --export=ALL,MANIFEST=my/manifest.csv,POOL=my/pool.csv,TAG=other,STAGES="gen feat analyze" aspect1_mi/aspect1.sbatch
# Variables: TAG (displace), STAGES ("manifest gen feat analyze"), WAV_DIR, RTTM_DIR, N_SRC (60 recordings as sources, the rest = pool),
#            PER_SPEAKER (15), MIN_SEC (4), MAX_SEC (8), MANIFEST/POOL (csv; default: built by the manifest stage), SEEDS (3), BATCH (3),
#            GRID (0,3,6,10,15,20,25,30), ENCODERS (wavlm,campplus,whisper,xlsr,melstat), N_NULL (100), N_BOOT (200), PIPELINE (combined), LIMIT (debug)
# Output: $BENCH_ROOT/aspect1_mi/runs/$TAG/  (data/ manifest + clips, metadata.csv, gen/, cond/, targets/, features/, results/)
set -uo pipefail
export BENCH_ROOT="${BENCH_ROOT:-$HOME/anon-bench}"
PIPE="$BENCH_ROOT"
export PIPE MODELS_DIR="${MODELS_DIR:-$BENCH_ROOT/models}"
PIPELINE="${PIPELINE:-combined}"
PROFILE="$PIPE/configs/pipelines/$PIPELINE.env"
[ -f "$PROFILE" ] || { echo "ERROR: no profile $PROFILE"; exit 1; }
CODE="$PIPE/aspect1_mi"
TAG="${TAG:-displace}"; OUT="$BENCH_ROOT/aspect1_mi/runs/$TAG"
WAV_DIR="${WAV_DIR:-$BENCH_ROOT/data/wav}"; RTTM_DIR="${RTTM_DIR:-$BENCH_ROOT/data/rttm}"
N_SRC="${N_SRC:-60}"; PER_SPEAKER="${PER_SPEAKER:-15}"; MIN_SEC="${MIN_SEC:-4}"; MAX_SEC="${MAX_SEC:-8}"
MANIFEST="${MANIFEST:-$OUT/data/manifest.csv}"; POOL="${POOL:-$OUT/data/pool.csv}"
SEEDS="${SEEDS:-3}"; BATCH="${BATCH:-3}"; GRID="${GRID:-0,3,6,10,15,20,25,30}"; STAGES="${STAGES:-manifest gen feat analyze}"
ENCODERS="${ENCODERS:-wavlm,campplus,whisper,xlsr,melstat}"; N_NULL="${N_NULL:-100}"; N_BOOT="${N_BOOT:-200}"; LIMIT="${LIMIT:-}"
SEEDVC="$BENCH_ROOT/repos/seed-vc"; ENV_SVC="$BENCH_ROOT/envs/seedvc"
mkdir -p "$OUT"

source "$PIPE/env/conda_env.sh" || exit 1
export HF_HOME="$BENCH_ROOT/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_CACHE="$SEEDVC/checkpoints/hf_cache"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
activate_env "$ENV_SVC"
has() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }
echo "=== aspect 1 MI  tag=$TAG manifest=$MANIFEST pool=$POOL seeds=$SEEDS batch=$BATCH grid=$GRID stages=$STAGES host=$(hostname) job=${SLURM_JOB_ID:-local} $(date -Is)"
df -h "$BENCH_ROOT" | tail -1 | awk '{print "    free space: " $4}'

if has manifest; then
  echo; echo "=== manifest  $(date -Is)"
  python "$CODE/make_manifest_displace.py" --wav-dir "$WAV_DIR" --rttm-dir "$RTTM_DIR" --out "$OUT/data" \
      --n-src "$N_SRC" --per-speaker "$PER_SPEAKER" --min-sec "$MIN_SEC" --max-sec "$MAX_SEC" 2>&1 | tee "$OUT/manifest.log"
fi
[ -f "$MANIFEST" ] && [ -f "$POOL" ] || { echo "ERROR: manifest $MANIFEST / pool $POOL missing"; exit 1; }
if has gen; then
  echo; echo "=== generate  $(date -Is)"
  python "$CODE/generate.py" --manifest "$MANIFEST" --pool "$POOL" --out "$OUT" --seedvc-root "$SEEDVC" --profile "$PROFILE" \
      --seeds "$SEEDS" --batch "$BATCH" --steps-grid "$GRID" $([ -n "$LIMIT" ] && echo --limit "$LIMIT") \
      2>&1 | tee "$OUT/generate.log" | grep -vE 'Warning|warn|it/s'
fi
if has feat; then
  echo; echo "=== features  $(date -Is)"
  python "$CODE/features.py" --out "$OUT" --manifest "$MANIFEST" --encoders "$ENCODERS" --seedvc-root "$SEEDVC" --noise-control \
      2>&1 | tee "$OUT/features.log" | grep -vE 'Warning|warn|it/s|not used|newly initialized|should probably'
fi
if has analyze; then
  echo; echo "=== analyze  $(date -Is)"
  python "$CODE/analyze.py" --out "$OUT" --encoders "$ENCODERS" --n-null "$N_NULL" --n-boot "$N_BOOT" \
      2>&1 | tee "$OUT/analyze.log" | grep -vE '^    \{'
fi
echo; du -sh "$OUT/gen" "$OUT/features" 2>/dev/null
echo "=== done $(date -Is).  $OUT/results/results.csv, mi_vs_t_S.png, mi_vs_t_C.png, leak_vs_util.png"
