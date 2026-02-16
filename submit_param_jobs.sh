#!/bin/bash
set -euo pipefail

JOB_DIR="${1:-param_jobs_new}"

if [ ! -d "${JOB_DIR}" ]; then
  echo "Job directory not found: ${JOB_DIR}" >&2
  exit 1
fi

count=0
for job in "${JOB_DIR}"/*.sh; do
  if [ ! -f "${job}" ]; then
    continue
  fi
  sbatch "${job}"
  count=$((count + 1))
done

echo "Submitted ${count} jobs from ${JOB_DIR}"
