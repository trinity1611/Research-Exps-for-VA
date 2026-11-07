#!/bin/bash
#SBATCH --job-name=mech_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --output=slurm_%x_%j.out
#
# D7 / MECHANISM SWEEP: contract-and-noise the content condition over a (gamma, sigma) grid, run the frozen generator,
# measure identity and content left in the outputs next to the certificate term.  gen (GPU) -> feat (GPU) -> analyze (CPU).
#   cd ~/anon-bench && sbatch mechanism/mechanism.sbatch
#   sbatch --export=ALL,TAG=smoke,LIMIT=24,N_BOOT=20,EPOCHS=30 mechanism/mechanism.sbatch     # smoke test
#   sbatch --export=ALL,STAGES=analyze mechanism/mechanism.sbatch                              # re-run the analysis only
# Variables: TAG (displace), ASPECT1_TAG (displace), AUDIT_TAG (displace), STAGES ("gen feat analyze"), PER_SPEAKER (5),
#            GAMMAS (1,0.5,0), SIGMAS (0,0.5,1,2), LDA_K (32; 0 = no subspace control), N_BOOT (200), EPOCHS (200), LIMIT (debug)
# Output: $BENCH_ROOT/mechanism/runs/$TAG/  (configs.json, mechanism.npz, metadata.csv, gen/<config>/<utt>.wav, features/, results/)
set -uo pipefail
export BENCH_ROOT="${BENCH_ROOT:-$HOME/anon-bench}"
PIPE="$BENCH_ROOT"
export PIPE MODELS_DIR="${MODELS_DIR:-$BENCH_ROOT/models}"
PIPELINE="${PIPELINE:-combined}"
PROFILE="$PIPE/configs/pipelines/$PIPELINE.env"
[ -f "$PROFILE" ] || { echo "ERROR: no profile $PROFILE"; exit 1; }
CODE="$PIPE/mechanism"
TAG="${TAG:-displace}"; ASPECT1_TAG="${ASPECT1_TAG:-displace}"; AUDIT_TAG="${AUDIT_TAG:-displace}"
A1="$BENCH_ROOT/aspect1_mi/runs/$ASPECT1_TAG"; CA="$BENCH_ROOT/condition_audit/runs/$AUDIT_TAG"; OUT="$BENCH_ROOT/mechanism/runs/$TAG"
MANIFEST="${MANIFEST:-$A1/data/manifest.csv}"
STAGES="${STAGES:-gen feat analyze}"; PER_SPEAKER="${PER_SPEAKER:-5}"; GAMMAS="${GAMMAS:-1,0.5,0}"; SIGMAS="${SIGMAS:-0,0.5,1,2}"; LDA_K="${LDA_K:-32}"
N_BOOT="${N_BOOT:-200}"; EPOCHS="${EPOCHS:-200}"; LIMIT="${LIMIT:-}"
SEEDVC="$BENCH_ROOT/repos/seed-vc"; ENV_SVC="$BENCH_ROOT/envs/seedvc"
mkdir -p "$OUT"

source "$PIPE/env/conda_env.sh" || exit 1
export HF_HOME="$BENCH_ROOT/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_CACHE="$SEEDVC/checkpoints/hf_cache"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
activate_env "$ENV_SVC"
has() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }
echo "=== mechanism sweep  tag=$TAG per_speaker=$PER_SPEAKER gammas=$GAMMAS sigmas=$SIGMAS lda_k=$LDA_K stages=$STAGES limit=${LIMIT:-none} host=$(hostname) job=${SLURM_JOB_ID:-local} $(date -Is)"
[ -f "$MANIFEST" ] && [ -f "$A1/metadata.csv" ] && [ -f "$CA/features.npz" ] || { echo "ERROR: need $MANIFEST, $A1/metadata.csv, $CA/features.npz"; exit 1; }
df -h "$BENCH_ROOT" | tail -1 | awk '{print "    free space: " $4}'

if has gen; then
  echo; echo "=== gen  $(date -Is)"
  python "$CODE/sweep.py" gen --manifest "$MANIFEST" --metadata "$A1/metadata.csv" --audit-features "$CA/features.npz" --aspect1-gen "$A1/gen" \
      --out "$OUT" --seedvc-root "$SEEDVC" --profile "$PROFILE" --per-speaker "$PER_SPEAKER" --gammas "$GAMMAS" --sigmas "$SIGMAS" --lda-k "$LDA_K" \
      $([ -n "$LIMIT" ] && echo --limit "$LIMIT") 2>&1 | tee "$OUT/gen.log" | grep -vE 'Warning|warn|it/s'
fi
if has feat; then
  echo; echo "=== feat  $(date -Is)"
  python "$CODE/sweep.py" feat --manifest "$MANIFEST" --out "$OUT" --seedvc-root "$SEEDVC" \
      2>&1 | tee "$OUT/feat.log" | grep -vE 'Warning|warn|it/s|not used|newly initialized|should probably'
fi
if has analyze; then
  echo; echo "=== analyze  $(date -Is)"
  python "$CODE/analyze.py" --out "$OUT" --n-boot "$N_BOOT" --epochs "$EPOCHS" 2>&1 | tee "$OUT/analyze.log"
fi
echo; du -sh "$OUT/gen" 2>/dev/null
echo "=== done $(date -Is).  $OUT/results/summary.md, sweep.csv, leak_vs_utility.png"
