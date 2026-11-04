#!/bin/bash
# ---------------------------------------------------------------------------
# One-time environment setup. RUN ON THE MASTER NODE (10.16.63.40) - it is the
# only node with internet access. Creates:
#   $BENCH_ROOT/repos/DiariZen   (+ its vendored pyannote-audio fork)
#   $BENCH_ROOT/repos/seed-vc
#   $BENCH_ROOT/envs/diarizen    conda env for stage 1
#   $BENCH_ROOT/envs/seedvc      conda env for stage 2
#
# Torch: the RTX PRO 6000 on n1 is Blackwell (sm_120) and needs CUDA >= 12.8,
# i.e. torch >= 2.7. The same cu128 wheels run on the H200 (sm_90) on n2, so
# one env spec serves both nodes. (Upstream pins torch 2.1 / 2.4 - too old.)
#
# Usage:  bash env/setup_envs.sh            (from $BENCH_ROOT)
# Env:    BENCH_ROOT (default $HOME/anon-bench), TORCH_INDEX, TORCH_SPEC
# ---------------------------------------------------------------------------
set -euo pipefail

BENCH_ROOT="${BENCH_ROOT:-$HOME/anon-bench}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1}"
PY="${PY_VERSION:-3.10}"

mkdir -p "$BENCH_ROOT"/{envs,repos,hf_cache,runs,data}
cd "$BENCH_ROOT"

source "$BENCH_ROOT/env/conda_env.sh" || exit 1

echo "== cloning repos"
[ -d repos/DiariZen ] || git clone --recurse-submodules https://github.com/BUTSpeechFIT/DiariZen.git repos/DiariZen
[ -d repos/seed-vc ]  || git clone https://github.com/Plachtaa/seed-vc.git repos/seed-vc
( cd repos/DiariZen && git submodule update --init --recursive || true )
[ -f repos/DiariZen/pyannote-audio/setup.py ] || [ -f repos/DiariZen/pyannote-audio/pyproject.toml ] || {
  echo "repos/DiariZen/pyannote-audio is empty - the vendored pyannote fork is required"; exit 1; }

# ---------------------------------------------------------------------------
echo "== env 1/2: diarizen"
[ -d envs/diarizen ] || conda create -y -p "$BENCH_ROOT/envs/diarizen" "python=$PY"
(
  activate_env "$BENCH_ROOT/envs/diarizen"
  python -m pip install --upgrade pip
  pip install $TORCH_SPEC --index-url "$TORCH_INDEX"
  cd repos/DiariZen
  pip install -r requirements.txt
  pip install -e .
  # the fork's constraints file may pin an old torch - strip torch pins, keep the rest
  if [ -f constraints.txt ]; then
    grep -viE '^(torch|torchaudio|torchvision)([=<>!~ ]|$)' constraints.txt > "$BENCH_ROOT/constraints_notorch.txt" || true
    ( cd pyannote-audio && pip install -e . -c "$BENCH_ROOT/constraints_notorch.txt" )
  else
    ( cd pyannote-audio && pip install -e . )
  fi
  pip install toml soundfile
  # re-assert the CUDA 12.8 torch in case a dependency downgraded it
  pip install $TORCH_SPEC --index-url "$TORCH_INDEX"
  python - <<'PY'
import torch, torchaudio, pyannote.audio, diarizen
print("diarizen env OK: torch", torch.__version__, "cuda", torch.version.cuda, "torchaudio", torchaudio.__version__)
assert torch.__version__.startswith("2.7"), torch.__version__
PY
)

# ---------------------------------------------------------------------------
echo "== env 2/2: seedvc"
[ -d envs/seedvc ] || conda create -y -p "$BENCH_ROOT/envs/seedvc" "python=$PY"
(
  activate_env "$BENCH_ROOT/envs/seedvc"
  python -m pip install --upgrade pip
  pip install $TORCH_SPEC --index-url "$TORCH_INDEX"
  # drop torch pins (we install cu128 ourselves), GUI / evaluation-only packages
  grep -viE '^(torch|torchvision|torchaudio|gradio|FreeSimpleGUI|sounddevice|resemblyzer|jiwer|--)' \
      repos/seed-vc/requirements.txt > "$BENCH_ROOT/seedvc_requirements_filtered.txt"
  pip install -r "$BENCH_ROOT/seedvc_requirements_filtered.txt"
  pip install soundfile onnxruntime      # onnxruntime: DNSMOS in src/evaluate.py
  pip install $TORCH_SPEC --index-url "$TORCH_INDEX"
  python - <<'PY'
import torch, torchaudio, transformers, librosa, munch, einops, yaml
print("seedvc env OK: torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__)
assert torch.__version__.startswith("2.7"), torch.__version__
PY
)

echo
echo "== done. Next: export HF_TOKEN=<your token> ; bash env/stage_models.sh"
