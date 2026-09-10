#!/bin/bash
#SBATCH --job-name=ciam05
#SBATCH --partition=caslake
#SBATCH --array=0-499
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/05_%A_%a.out

# Run only after stage 4's probe reports zero unfinished groups:
#   python -u ../04_calc_cases.py --scenario glocal --probe-only
#   sbatch --export=ALL,SCENARIO=glocal 05_array.sbatch
# then:
#   python -u ../05_optimize.py --scenario glocal --probe-only
# Each element also refuses any segment whose stage-4 npv has nulls, so a
# wrong-order run fails loudly instead of skewing segment choices.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source rcc_env.sh
cd ..
N=$((SLURM_ARRAY_TASK_MAX + 1))
python -u 05_optimize.py --scenario "${SCENARIO}" --shard "${SLURM_ARRAY_TASK_ID}/${N}"
