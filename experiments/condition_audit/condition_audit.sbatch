#!/bin/bash
#SBATCH --job-name=cond_audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --output=slurm_%x_%j.out
#
# CONDITION-SPACE AUDIT (framework D4, D6, D10-context, D15, D8) on the Aspect 1 source utterances.  extract (GPU) -> analyze (CPU).
#   cd ~/anon-bench && sbatch condition_audit/condition_audit.sbatch
#   sbatch --export=ALL,STAGES=analyze condition_audit/condition_audit.sbatch        # re-run the analysis only
# Variables: TAG (displace), ASPECT1_TAG (displace: manifest, cond/ and features/ come from aspect1_mi/runs/$ASPECT1_TAG),
#            STAGES ("extract analyze"), N_BOOT (200), EPOCHS (200), LIMIT (debug), NO_ASR (set to 1 to skip whisper transcripts)
# Output: $BENCH_ROOT/condition_audit/runs/$TAG/  (H/ per-utterance XLS-R frames, features.npz, table.csv, asr.jsonl, results/)
set -uo pipefail
export BENCH_ROOT="${BENCH_ROOT:-$HOME/anon-bench}"
PIPE="$BENCH_ROOT"
export PIPE MODELS_DIR="${MODELS_DIR:-$BENCH_ROOT/models}"
PIPELINE="${PIPELINE:-combined}"
PROFILE="$PIPE/configs/pipelines/$PIPELINE.env"
[ -f "$PROFILE" ] || { echo "ERROR: no profile $PROFILE"; exit 1; }
CODE="$PIPE/condition_audit"
TAG="${TAG:-displace}"; ASPECT1_TAG="${ASPECT1_TAG:-displace}"
A1="$BENCH_ROOT/aspect1_mi/runs/$ASPECT1_TAG"; OUT="$BENCH_ROOT/condition_audit/runs/$TAG"
MANIFEST="${MANIFEST:-$A1/data/manifest.csv}"
STAGES="${STAGES:-extract analyze}"; N_BOOT="${N_BOOT:-200}"; EPOCHS="${EPOCHS:-200}"; LIMIT="${LIMIT:-}"; NO_ASR="${NO_ASR:-}"
SEEDVC="$BENCH_ROOT/repos/seed-vc"; ENV_SVC="$BENCH_ROOT/envs/seedvc"
mkdir -p "$OUT"

source "$PIPE/env/conda_env.sh" || exit 1
export HF_HOME="$BENCH_ROOT/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_CACHE="$SEEDVC/checkpoints/hf_cache"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
activate_env "$ENV_SVC"
has() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }
echo "=== condition audit  tag=$TAG manifest=$MANIFEST aspect1=$A1 stages=$STAGES n_boot=$N_BOOT epochs=$EPOCHS limit=${LIMIT:-none} no_asr=${NO_ASR:-0} host=$(hostname) job=${SLURM_JOB_ID:-local} $(date -Is)"
[ -f "$MANIFEST" ] || { echo "ERROR: manifest $MANIFEST missing"; exit 1; }

if has extract; then
  echo; echo "=== extract  $(date -Is)"
  python "$CODE/extract.py" --manifest "$MANIFEST" --out "$OUT" --seedvc-root "$SEEDVC" --profile "$PROFILE" --cond-dir "$A1" \
      $([ -n "$LIMIT" ] && echo --limit "$LIMIT") $([ -n "$NO_ASR" ] && echo --no-asr) \
      2>&1 | tee "$OUT/extract.log" | grep --line-buffered -vE 'Warning|warn|it/s|not used|newly initialized|should probably'
fi
if has analyze; then
  echo; echo "=== analyze  $(date -Is)"
  python "$CODE/analyze.py" --out "$OUT" --aspect1 "$A1" --n-boot "$N_BOOT" --epochs "$EPOCHS" 2>&1 | tee "$OUT/analyze.log"
fi
echo; du -sh "$OUT/H" 2>/dev/null
echo "=== done $(date -Is).  $OUT/results/summary.md"
