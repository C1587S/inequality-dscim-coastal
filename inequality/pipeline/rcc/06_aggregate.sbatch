#!/bin/bash
#SBATCH --job-name=ciam06
#SBATCH --partition=caslake
# billing account; switch to cil once that account covers this work
#SBATCH --account=pi-mgreenst
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=180G
#SBATCH --time=02:00:00
# literal path: #SBATCH can't expand env vars; sync with PIPELINE_LOGS
#SBATCH --output=/scratch/midway3/cadavidsanchez/pyciam-logs/06_%A.out

# Single-node aggregation on a LocalCluster:
#   sbatch --export=ALL,SCENARIO=glocal 06_aggregate.sbatch
# The completeness gate runs first and refuses an incomplete store.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source rcc_env.sh
cd ..
run 06_aggregate.py --scenario "${SCENARIO}"
