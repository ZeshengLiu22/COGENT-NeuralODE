#!/usr/bin/env bash

# Shared environment setup for continuous_graph_emulator Slurm jobs.
# The login/default Python on LS6 can resolve to Intel Python, which does not
# include torch. Activate the CGE PyTorch/PyG env before choosing PYTHON_BIN.

: "${CONDA_SH:=/work/09575/${USER}/ls6/etc/profile.d/conda.sh}"
: "${CONDA_ENV_1:=base}"
: "${CONDA_ENV_2:=/work/09575/${USER}/conda_envs/torchpyg-cu128-cge}"

if [[ -f "${CONDA_SH}" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  export CONDA_PKGS_DIRS="/work/09575/${USER}/conda_pkgs"
  conda activate "${CONDA_ENV_1}" >/dev/null 2>&1 || true
  conda activate "${CONDA_ENV_2}"
  set -u
elif [[ -x "${CONDA_ENV_2}/bin/python" ]]; then
  export PATH="${CONDA_ENV_2}/bin:${PATH}"
else
  echo "[FATAL] Could not activate torch/PyG env: ${CONDA_ENV_2}" >&2
  echo "[FATAL] Missing conda.sh (${CONDA_SH}) and env python." >&2
  exit 2
fi

: "${PYTHON_BIN:=$(command -v python)}"
