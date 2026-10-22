#!/bin/bash
#SBATCH --job-name=ciam05
#SBATCH --partition=caslake
# billing account; switch to cil once that account covers this work
#SBATCH --account=pi-mgreenst
#SBATCH --array=0-499
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# measured ~2G physical per element (node-level RSS; sacct MaxRSS is
# broken here); 3G keeps cores, not memory, the binding resource on
# both partitions, and an OOM-killed element fails visibly and
# resubmits cheaply. 8G reserved half of cil for nothing.
#SBATCH --mem=3G
#SBATCH --time=03:00:00
# literal path: #SBATCH can't expand env vars; sync with PIPELINE_LOGS
#SBATCH --output=/scratch/midway3/cadavidsanchez/pyciam-logs/05_%A_%a.out

# Run only after stage 4's probe reports zero unfinished groups. Each
# element also refuses any segment whose stage-4 npv has nulls, so a
# wrong-order run fails loudly instead of skewing segment choices.
# N_SHARDS comes from rcc_env.sh; array size can be any subset.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source rcc_env.sh
cd ..
run 05_optimize.py --scenario "${SCENARIO}" --shard "${SLURM_ARRAY_TASK_ID}/${N_SHARDS}" ${EXTRA:-}
