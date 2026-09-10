#!/bin/bash
#SBATCH --job-name=ciam04
#SBATCH --partition=caslake
#SBATCH --array=0-499
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/04_%A_%a.out

# Submit from this directory, after the template exists:
#   mkdir -p logs
#   python -u ../04_calc_cases.py --scenario glocal --template-only
#   sbatch --export=ALL,SCENARIO=glocal 04_array.sbatch
# then verify and, if needed, gap-fill:
#   python -u ../04_calc_cases.py --scenario glocal --probe-only
#   sbatch --export=ALL,SCENARIO=glocal 04_array.sbatch   (runs only the probe's list)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source rcc_env.sh
cd ..
N=$((SLURM_ARRAY_TASK_MAX + 1))
python -u 04_calc_cases.py --scenario "${SCENARIO}" --shard "${SLURM_ARRAY_TASK_ID}/${N}"
