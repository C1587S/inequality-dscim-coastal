# Environment for running pipeline stages on RCC. Edit the paths for your
# layout, then the sbatch scripts source this file.

source activate /project/cil/home_dirs/rcc/envs/pyciam

# pyCIAM is pre-installed in the env (see environment.yml); a thousand
# concurrent pip installs into a shared env would race
export PIPELINE_SKIP_PYCIAM_INSTALL=1

# stage 6 and the probes use a LocalCluster instead of Dask Gateway
export PIPELINE_EXECUTOR=local

# the three data roots, mirroring the GCS layout (see ../config.py)
export PIPELINE_SCRATCH_ROOT=/project/cil/gcp/inequality/pyciam-scratch
export PIPELINE_INPUTS_ROOT=/project/cil/gcp/inequality/pyciam-inputs
export PIPELINE_OUTPUTS_ROOT=/project/cil/gcp/inequality/pyciam-outputs
