#!/bin/bash
#SBATCH --job-name=ciam04
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
#SBATCH --time=06:00:00
# literal path: #SBATCH can't expand env vars; sync with PIPELINE_LOGS
#SBATCH --output=/scratch/midway3/cadavidsanchez/pyciam-logs/04_%A_%a.out

# Submit from this directory after the template exists (see README.md).
# The shard count comes from N_SHARDS in rcc_env.sh, not from the array
# size, so a trial (--array=0-9) or a resubmission of failed elements
# (--array=17,203) addresses the same partition of the work.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source rcc_env.sh
cd ..
run 04_calc_cases.py --scenario "${SCENARIO}" --shard "${SLURM_ARRAY_TASK_ID}/${N_SHARDS}" ${EXTRA:-}
