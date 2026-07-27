#!/usr/bin/env bash

# Shared environment setup for continuous_graph_emulator jobs.
# Prefer the current Python when it already has the required training
# dependencies. Otherwise, activate a local conda env before setting PYTHON_BIN.

cge_resolve_python() {
  local candidate="$1"
  if [[ -x "${candidate}" ]]; then
    printf '%s\n' "${candidate}"
    return 0
  fi
  command -v "${candidate}" 2>/dev/null
}

cge_python_has_deps() {
  local candidate="$1"
  local python_bin
  python_bin="$(cge_resolve_python "${candidate}")" || return 1
  "${python_bin}" -c "import torch, torch_geometric, torchdiffeq, torchcde, yaml, numpy, scipy" >/dev/null 2>&1
}

cge_use_python() {
  local candidate="$1"
  local python_bin
  python_bin="$(cge_resolve_python "${candidate}")" || return 1
  cge_python_has_deps "${python_bin}" || return 1
  export PYTHON_BIN="${python_bin}"
  return 0
}

if [[ -n "${PYTHON_BIN:-}" ]] && cge_use_python "${PYTHON_BIN}"; then
  return 0 2>/dev/null || exit 0
fi

if cge_use_python python; then
  return 0 2>/dev/null || exit 0
fi

conda_sh_candidates=()
if [[ -n "${CONDA_SH:-}" ]]; then
  conda_sh_candidates+=("${CONDA_SH}")
fi
if [[ -n "${CONDA_EXE:-}" ]]; then
  conda_sh_candidates+=("$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)/etc/profile.d/conda.sh")
fi
if command -v conda >/dev/null 2>&1; then
  conda_bin="$(command -v conda)"
  conda_sh_candidates+=("$(cd "$(dirname "${conda_bin}")/.." && pwd)/etc/profile.d/conda.sh")
fi
conda_sh_candidates+=(
  "/software/u22/anaconda/python3.9/etc/profile.d/conda.sh"
  "${HOME}/miniconda3/etc/profile.d/conda.sh"
  "${HOME}/anaconda3/etc/profile.d/conda.sh"
  "/work/09575/${USER}/ls6/etc/profile.d/conda.sh"
)

for conda_sh_candidate in "${conda_sh_candidates[@]}"; do
  if [[ -f "${conda_sh_candidate}" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "${conda_sh_candidate}"
    set -u
    break
  fi
done

conda_env_candidates=()
if [[ -n "${CONDA_ENV_2:-}" ]]; then
  conda_env_candidates+=("${CONDA_ENV_2}")
fi
if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
  conda_env_candidates+=("${CONDA_DEFAULT_ENV}")
fi
conda_env_candidates+=(
  "cge"
  "${HOME}/.conda/envs/cge"
  "torchpyg-cu124"
  "${HOME}/.conda/envs/torchpyg-cu124"
  "issm-gnn"
  "${HOME}/.conda/envs/issm-gnn"
  "/work/09575/${USER}/conda_envs/torchpyg-cu128-cge"
)

if command -v conda >/dev/null 2>&1; then
  for conda_env_candidate in "${conda_env_candidates[@]}"; do
    set +u
    conda activate "${conda_env_candidate}" >/dev/null 2>&1
    activate_status="$?"
    set -u
    if [[ "${activate_status}" -eq 0 ]] && cge_use_python python; then
      return 0 2>/dev/null || exit 0
    fi
  done
fi

for conda_env_candidate in "${conda_env_candidates[@]}"; do
  if [[ -x "${conda_env_candidate}/bin/python" ]] && cge_use_python "${conda_env_candidate}/bin/python"; then
    export PATH="${conda_env_candidate}/bin:${PATH}"
    return 0 2>/dev/null || exit 0
  fi
done

echo "[FATAL] Could not find a Python environment with torch/PyG dependencies." >&2
echo "[FATAL] Tried PYTHON_BIN, the active python, and conda env candidates including ${HOME}/.conda/envs/cge." >&2
echo "[FATAL] Activate your env first or set PYTHON_BIN=/path/to/python or CONDA_ENV_2=/path/to/env." >&2
exit 2
