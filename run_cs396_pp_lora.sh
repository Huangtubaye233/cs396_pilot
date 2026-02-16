#!/bin/bash
#SBATCH --job-name=cs396-pp
#SBATCH --output=output/cs396_pp_%j.out
#SBATCH --error=output/cs396_pp_%j.err
#SBATCH --account=p31502
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kefanyu2026@u.northwestern.edu
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00

set -euo pipefail

# Avoid MKL activation error under set -u
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-}"

module purge

export HF_HOME="/projects/b1170/users/kyx8046/hf-cache"
mkdir -p "${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_HUB_CACHE="${HF_HOME}"

mkdir -p output

source /projects/b1170/users/kyx8046/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate CS396

python /projects/b1170/users/kyx8046/CS396/CS396_PP_lora.py "$@"
