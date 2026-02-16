# CS396 Pilot Workflow

## Quick Start

1. **Create the conda environment**
   ```bash
   bash setup_conda_env.sh
   ```

2. **Adjust the grid search parameters (optional)**
   - Edit `generate_param_sh.py` and update `GRID` as needed.
   - You may also need to set cluster-specific environment variables (e.g., `ENV_NAME`, `SBATCH_ACCOUNT`, `SBATCH_PARTITION`) before generating jobs.

3. **Generate Slurm job scripts**
   ```bash
   python generate_param_sh.py
   ```

4. **Submit the jobs**
   ```bash
   bash submit_param_jobs.sh
   ```

## Notes
- Output and debug files are written under `results/` by default.
