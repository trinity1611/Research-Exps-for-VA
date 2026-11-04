#!/bin/bash
# Sourced by every script: makes `conda` and `source activate` available.
# Order: $CONDA_MODULE if set -> common module names -> conda binaries under /opt/ohpc/pub/apps -> conda already on PATH.
_conda_hook() {
  # define the `conda` shell function for this session only (no `conda init`, which the cluster guide forbids)
  command -v conda >/dev/null 2>&1 || return 0
  eval "$(conda shell.bash hook 2>/dev/null)" 2>/dev/null || true
}
_load_conda() {
  if command -v module >/dev/null 2>&1; then
    module purge >/dev/null 2>&1 || true
    for m in ${CONDA_MODULE:-} Miniforge3 miniforge3 Miniconda3 miniconda miniconda3 miniforge conda Anaconda3 anaconda anaconda3 mambaforge; do
      [ -n "$m" ] || continue
      if module load "$m" >/dev/null 2>&1; then echo "conda via module $m"; _conda_hook; return 0; fi
    done
    # search the module tree for anything conda-like (e.g. Miniforge3/26.1.1-3)
    for m in $(module -t avail 2>&1 | grep -iE 'conda|forge' | head -3); do
      if module load "$m" >/dev/null 2>&1; then echo "conda via module $m"; _conda_hook; return 0; fi
    done
  fi
  for c in ${CONDA_HOME:-} /opt/ohpc/pub/apps/miniforge3 /opt/ohpc/pub/apps/miniconda3 /opt/ohpc/pub/apps/miniconda \
           /opt/ohpc/pub/apps/anaconda3 /opt/ohpc/pub/apps/*conda* /opt/ohpc/pub/apps/*forge* /opt/*conda* /opt/*forge* $HOME/miniforge3 $HOME/miniconda3; do
    if [ -f "$c/etc/profile.d/conda.sh" ]; then
      # shellcheck disable=SC1090
      source "$c/etc/profile.d/conda.sh"; export PATH="$c/bin:$PATH"; echo "conda via $c"; return 0
    fi
  done
  command -v conda >/dev/null 2>&1 && { echo "conda already on PATH: $(command -v conda)"; _conda_hook; return 0; }
  echo "ERROR: conda not found. Run 'module avail 2>&1 | grep -i conda' or 'ls /opt/ohpc/pub/apps' and set CONDA_MODULE=<name> or CONDA_HOME=<dir>." >&2
  return 1
}
_load_conda
# `source activate` is deprecated in new conda; provide a shim that works with both
activate_env() {
  if conda activate "$1" 2>/dev/null; then return 0; fi
  if source activate "$1" 2>/dev/null; then return 0; fi
  # last resort: put the env's bin first on PATH (works for python/pip, which is all we need)
  export PATH="$1/bin:$PATH"; export CONDA_PREFIX="$1"
  [ -x "$1/bin/python" ] || { echo "ERROR: cannot activate env $1" >&2; return 1; }
}
