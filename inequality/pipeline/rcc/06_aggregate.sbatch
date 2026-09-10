#!/bin/bash
#SBATCH --job-name=ciam06
#SBATCH --partition=caslake
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --output=logs/06_%A.out

# Single-node aggregation on a LocalCluster:
#   sbatch --export=ALL,SCENARIO=glocal 06_aggregate.sbatch
# The completeness gate runs first and refuses an incomplete store.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source rcc_env.sh
cd ..
python -u 06_aggregate.py --scenario "${SCENARIO}"
