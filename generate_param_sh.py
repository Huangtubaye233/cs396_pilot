import itertools
import os
from pathlib import Path


GRID = {
    "mode": ["lora", "pt"],
    "num_train_epochs": [1, 3],
    "learning_rate": [5e-5, 1e-5],
    "test_n_shot": [3, 5, 8],
}

SH_DIR = Path("param_jobs")
SH_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=output/{job_name}.out
#SBATCH --error=output/{job_name}.err
#SBATCH --account={account}
#SBATCH --mail-type=ALL
#SBATCH --mail-user={mail_user}
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time={time_limit}

set -euo pipefail

# Avoid MKL activation/deactivation errors under set -u
export MKL_INTERFACE_LAYER="${{MKL_INTERFACE_LAYER:-}}"
export CONDA_MKL_INTERFACE_LAYER_BACKUP="${{CONDA_MKL_INTERFACE_LAYER_BACKUP:-}}"

module purge

export HF_HOME="${{HF_HOME:-$PWD/hf-cache}}"
mkdir -p "${{HF_HOME}}"
export TRANSFORMERS_CACHE="${{TRANSFORMERS_CACHE:-${{HF_HOME}}}}"
export HF_HUB_CACHE="${{HF_HUB_CACHE:-${{HF_HOME}}}}"

mkdir -p output

if [ -z "${{CONDA_EXE:-}}" ]; then
  if [ -n "${{CONDA_PREFIX:-}}" ] && [ -f "${{CONDA_PREFIX}}/etc/profile.d/conda.sh" ]; then
    source "${{CONDA_PREFIX}}/etc/profile.d/conda.sh"
  elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    source "${{CONDA_BASE}}/etc/profile.d/conda.sh"
  else
    echo "Conda not found. Please install Miniconda/Anaconda first." >&2
    exit 1
  fi
fi
eval "$(conda shell.bash hook)"
set +u
export ENV_NAME="{env_name}"
conda activate "${{ENV_NAME}}"
set -u

export EXP_ID="{exp_id}"
export NUM_EPOCHS="{num_train_epochs}"
export LEARNING_RATE="{learning_rate}"
export TEST_N_SHOT="{test_n_shot}"

python "{project_root}/{script}" "$@"
"""


def main():
    project_root = Path(__file__).resolve().parent
    env_name = os.getenv("ENV_NAME", "CS396_pp")
    account = os.getenv("SBATCH_ACCOUNT", "123456") # TODO: change to actual account
    mail_user = os.getenv("SBATCH_MAIL_USER", "your_email@example.com") # TODO: change to actual email
    partition = os.getenv("SBATCH_PARTITION", "abcde") # TODO: change to actual partition
    gres = os.getenv("SBATCH_GRES", "gpu:a100:1") # TODO: change to actual gres
    time_limit = os.getenv("SBATCH_TIME", "03:00:00") # TODO: change to actual time limit
    keys = ["mode", "num_train_epochs", "learning_rate", "test_n_shot"]
    values = [GRID[k] for k in keys]

    for idx, combo in enumerate(itertools.product(*values), 1):
        params = dict(zip(keys, combo))
        exp_id = f"exp{idx:03d}"
        mode = params["mode"]
        job_name = f"cs396_{mode}_{exp_id}"
        script = "CS396_PP_lora.py" if mode == "lora" else "CS396_PP_pt.py"

        sh_content = TEMPLATE.format(
            job_name=job_name,
            exp_id=exp_id,
            num_train_epochs=params["num_train_epochs"],
            learning_rate=params["learning_rate"],
            test_n_shot=params["test_n_shot"],
            script=script,
            project_root=project_root,
            env_name=env_name,
            account=account,
            mail_user=mail_user,
            partition=partition,
            gres=gres,
            time_limit=time_limit,
        )
        sh_path = SH_DIR / f"{job_name}.sh"
        sh_path.write_text(sh_content)
        sh_path.chmod(0o755)

    print(f"Generated {len(list(SH_DIR.glob('*.sh')))} scripts in {SH_DIR}/")


if __name__ == "__main__":
    main()
