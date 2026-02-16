#!/bin/bash
set -euo pipefail

ENV_NAME="${1:-CS396_pp}"
PYTHON_VERSION="${2:-3.12}"

if [ -z "${CONDA_EXE:-}" ]; then
  if [ -n "${CONDA_PREFIX:-}" ] && [ -f "${CONDA_PREFIX}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_PREFIX}/etc/profile.d/conda.sh"
  elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
  else
    echo "Conda not found. Please install Miniconda/Anaconda first." >&2
    exit 1
  fi
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env '${ENV_NAME}' already exists. Activating."
else
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

# Avoid MKL activation/deactivation errors under set -u
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-}"
export CONDA_MKL_INTERFACE_LAYER_BACKUP="${CONDA_MKL_INTERFACE_LAYER_BACKUP:-}"

set +u
conda activate "${ENV_NAME}"
set -u

# Install PyTorch (CUDA 12.1). Adjust if your cluster uses a different CUDA.
set +u
conda install -y -c pytorch -c nvidia pytorch pytorch-cuda=12.1
set -u

pip install -U \
  transformers \
  datasets \
  trl \
  peft \
  bitsandbytes \
  accelerate \
  tqdm \
  pandas \
  sentencepiece \
  protobuf \
  safetensors

echo "Environment setup complete: ${ENV_NAME}"
