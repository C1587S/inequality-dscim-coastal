# Running stages 4-6 on RCC

Stages 4 and 5 go out as SLURM arrays on caslake: each element runs its
segments serially in its own process, so there is no scheduler to lose and
no chunk two elements could both write. Stage 6 runs on one node with a
local dask cluster. Stages 1-3 stay on the hub; their outputs are inputs
here.

At the measured ~3.5 s per calc task, stage 4 is about 96 core-hours: a
500-element array finishes in roughly 15 minutes of queue-wide work, and a
whole scenario (4 through 6) is well under an hour of machine time.

## One-time setup

Build the env and install the vendored pyCIAM from a clone of this repo:

```bash
mamba env create -f environment.yml -p /project/cil/home_dirs/rcc/envs/pyciam
source activate /project/cil/home_dirs/rcc/envs/pyciam
pip install --no-deps -e /path/to/inequality-dscim-coastal/pyciam
```

Edit `rcc_env.sh`: the env path and the three data roots. Then stage the
inputs under those roots, mirroring the GCS layout (about 18 GB total):

```
$PIPELINE_INPUTS_ROOT/surge-lookup/surge-lookup-seg-ir.zarr
$PIPELINE_SCRATCH_ROOT/slr/ar6-tlim-1000samples.zarr                  (9.9 GB)
$PIPELINE_SCRATCH_ROOT/slr/ar6-tlim-1000samples-globalclimate.zarr    (7.8 GB)
$PIPELINE_SCRATCH_ROOT/sliiders/local-rho-ir.zarr                     (0.2 GB)
$PIPELINE_SCRATCH_ROOT/sliiders/global-rho-ir.zarr                    (0.2 GB)
$PIPELINE_SCRATCH_ROOT/refa/refa.zarr                                 (0.1 GB)
$PIPELINE_OUTPUTS_ROOT/pyCIAM_outputs_inequality_1000_ssp234.zarr     (stage 6's
                                                          coastal reference)
```

## Per scenario

```bash
cd inequality/pipeline/rcc && mkdir -p logs && source rcc_env.sh

python -u ../04_calc_cases.py --scenario glocal --template-only
sbatch --export=ALL,SCENARIO=glocal 04_array.sbatch
python -u ../04_calc_cases.py --scenario glocal --probe-only

sbatch --export=ALL,SCENARIO=glocal 05_array.sbatch
python -u ../05_optimize.py --scenario glocal --probe-only

sbatch --export=ALL,SCENARIO=glocal 06_aggregate.sbatch
```

A probe that reports unfinished groups saves their list and exits nonzero;
resubmit the same array and only those tasks run. Repeat until the probe
exits clean, then move on. Stage 5 won't run over stage-4 holes (each
element checks its own segments' npv first), and stage 6's gate re-checks
everything before aggregating.

Finals land under `$PIPELINE_OUTPUTS_ROOT`; copy them back to
`gs://impactlab-data/gcp/outputs/coastal/` when done.

## Disk and inodes

A stage-4/5 tmp store is on the order of half a million chunk files. With
/project/cil tight on inodes, run one scenario at a time and delete
`$PIPELINE_SCRATCH_ROOT/runs/<scenario>/` after stage 6 passes and the
finals are copied off.
