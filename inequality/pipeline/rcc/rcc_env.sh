# Environment for running pipeline stages on RCC. The sbatch scripts source
# this file; source it yourself for login-node commands, then use run():
#   source rcc_env.sh
#   run ../04_calc_cases.py --scenario glocal --template-only

module load apptainer 2>/dev/null || true

export CONTAINER=/project/cil/home_dirs/rcc/containers/pyciam.sif
export APPTAINER_BIND=/project,/scratch/midway3
# node-local disk, where SLURM_TMPDIR lives: bind it when the machine has
# it (compute nodes), so the env var names a real path inside the
# container - an unbound SLURM_TMPDIR is just a string and broke a whole
# gap-fill wave. _tmp_base() validates either way.
[ -d /scratch/local ] && export APPTAINER_BIND="$APPTAINER_BIND,/scratch/local"

# shard count for the stage 4/5 arrays. Fixed here rather than derived from
# the array size, so submitting a subset of elements (a trial, a resubmit of
# failures) still addresses the same partition of the work.
export N_SHARDS=${N_SHARDS:-500}

# stage 6 and the probes use a LocalCluster instead of Dask Gateway
export PIPELINE_EXECUTOR=local

# nothing writes to /project mid-run: a full group fileset there killed a
# whole array once (SLURM couldn't create log files; elements died in
# seconds). Logs and probe-state files live on scratch, and the container
# must not scatter __pycache__ into the repo from 500 elements at once.
# The #SBATCH --output lines can't expand variables, so they hardcode the
# log dir; keep them in sync with PIPELINE_LOGS.
export PIPELINE_LOGS=/scratch/midway3/cadavidsanchez/pyciam-logs
export PIPELINE_STATE_DIR=/scratch/midway3/cadavidsanchez/pyciam-state
mkdir -p "$PIPELINE_LOGS" "$PIPELINE_STATE_DIR"
export PYTHONDONTWRITEBYTECODE=1

# tmp stores and staged inputs live on scratch (110 TB free, no inode
# pressure, purge-tolerant: everything under them is re-stageable or
# regenerable). /project/cil receives only the final outputs.
export PIPELINE_SCRATCH_ROOT=/scratch/midway3/cadavidsanchez/pyciam-scratch
export PIPELINE_INPUTS_ROOT=/scratch/midway3/cadavidsanchez/pyciam-inputs
export PIPELINE_OUTPUTS_ROOT=/project/cil/gcp/inequality/coastal

run() { apptainer exec "$CONTAINER" python -u "$@"; }
