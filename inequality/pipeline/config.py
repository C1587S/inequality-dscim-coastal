"""
Paths and parameters for the inequality pyCIAM pipeline.

Three scenarios, keyed by name throughout the pipeline:

- fulladapt: local income (rho) and local sea level. One run produces both the
  optimalfixed (full adapt) and noAdaptation cases.
- glocal: population-weighted global average rho (2000-2014 mean), local sea
  level. Global income, local climate. This is the run previously published as
  "globaladapt".
- global: global average rho, and globally uniform climate-driven sea level.
  The SLR input replaces local sea level with global-mean sea level plus each
  site's local vertical land motion, so the climate signal is homogenised while
  local geophysics (uplift, subsidence) is kept.

Sample caveat, to be carried into any description of these scenarios: the
1,000 SLR samples are rank quantiles computed independently for the local,
global, and VLM series (see 01_process_slr.py), so sample n is rank-matched
across series, not a single physical FACTS draw.
"""

from pathlib import Path

from cloudpathlib import AnyPath

SCENARIOS = ("fulladapt", "glocal", "global")

# =============================================================================
# SLR PROCESSING
# =============================================================================
TLIM_SCENARIOS = ["tlim1.5", "tlim2.0", "tlim3.0", "tlim4.0", "tlim5.0"]
WORKFLOWS = ["wf_1f", "wf_2f"]
N_SAMPLES_PER_WORKFLOW = 500
N_SAMPLES_TOTAL = N_SAMPLES_PER_WORKFLOW * len(WORKFLOWS)

# 20,000 AR6 FACTS draws per scenario x workflow, reduced by seeded
# rank-quantile draw. The seed reproduces the store used in the 2026 runs.
N_DRAWS = 20000
SEED = 11222023

DIR_SLR_AR6_GRIDDED_PUBLIC = (
    "gs://ar6-lsl-simulations-public-standard/gridded/full_sample_workflows"
)
DIR_SLR_AR6_RAW = AnyPath(
    "gs://impactlab-data/coastal/data/raw/slr/ar6/ar6/global/full_sample_workflows"
)
PATH_VLM_REQUESTER_PAYS = (
    "gs://ar6-lsl-simulations-requesterpays-standard/gridded/full_sample_components/"
    "verticallandmotion-kopp14-verticallandmotion_localsl.zarr"
)

# =============================================================================
# INPUTS SURVIVING FROM THE REGIONAL SCC WORKFLOW (read-only)
# =============================================================================
PATH_SLIIDERS = AnyPath(
    "gs://impactlab-data/coastal/local-scc-model/data/int/sliiders-ir.zarr"
)
PATHS_SURGE_LOOKUP = {
    "seg": AnyPath(
        "gs://impactlab-data/coastal/local-scc-model/data/int/surge-lookup/surge-lookup-seg.zarr"
    ),
    "seg_ir": AnyPath(
        "gs://impactlab-data/coastal/local-scc-model/data/int/surge-lookup/surge-lookup-seg-ir.zarr"
    ),
}
PATH_PARAMS = Path(__file__).parents[2] / "params.json"

# =============================================================================
# SCRATCH INTERMEDIATES
# =============================================================================
DIR_SCRATCH = AnyPath("gs://impactlab-data-scratch/inequality-pyciam")

PATH_SLR = DIR_SCRATCH / "slr" / "ar6-tlim-1000samples.zarr"
# Same layout as PATH_SLR but lsl_msl05 := gsl_msl05 + lsl_ncc_msl05
# (global-mean climate-driven SLR on top of local VLM). Fed to pyCIAM
# unchanged in place of the base store for the "global" scenario.
PATH_SLR_GLOBALCLIMATE = DIR_SCRATCH / "slr" / "ar6-tlim-1000samples-globalclimate.zarr"

# SLIIDERS variants: base data with K/pop renamed to the 2019 labels pyCIAM's
# collapse expects, and (for glocal/global) rho replaced by its global average.
_SLIIDERS_LOCAL_RHO = DIR_SCRATCH / "sliiders" / "local-rho-ir.zarr"
_SLIIDERS_GLOBAL_RHO = DIR_SCRATCH / "sliiders" / "global-rho-ir.zarr"

SLIIDERS_IR = {
    "fulladapt": _SLIIDERS_LOCAL_RHO,
    "glocal": _SLIIDERS_GLOBAL_RHO,
    "global": _SLIIDERS_GLOBAL_RHO,
}
# One seg store and one refA serve all three scenarios. The collapse
# recomputes rho from ypcc (pyCIAM/utils.py:268) and discards the store's
# rho, and our scenarios never touch ypcc, so every variant collapses to the
# same seg store and the same refA. refA is optimised under no climate
# change, where both SLR stores are identical anyway. This matches Ian's
# original design: present-day adaptation reflects local income. Rho reaches
# the model only through the main cost calculation, which reads the IR store.
#
# Checked against the published outputs (Aug 2026). They disagree cell for
# cell, but the cause is missing data in the published stores, not refA
# responding to rho (see README). Final confirmation comes with the rerun:
# v3 fulladapt should match published glocal on non-storm noAdaptation costs
# wherever glocal has no holes.
PATH_SLIIDERS_SEG = DIR_SCRATCH / "sliiders" / "seg.zarr"
PATH_REFA = DIR_SCRATCH / "refa" / "refa.zarr"
SLR = {
    "fulladapt": PATH_SLR,
    "glocal": PATH_SLR,
    "global": PATH_SLR_GLOBALCLIMATE,
}
PATH_TMP = {s: DIR_SCRATCH / "runs" / s / "costs-by-case.zarr" for s in SCENARIOS}
PATH_INTERMEDIATE = {
    s: DIR_SCRATCH / "runs" / s / "costs-filtered.zarr" for s in SCENARIOS
}

# =============================================================================
# FINAL OUTPUTS
# =============================================================================
# v3 outputs. The published v2 stores have incomplete noAdaptation cases
# (see README) and stay in place so nothing changes under their readers; the
# corrected outputs get new names.
_DIR_OUTPUTS = AnyPath("gs://impactlab-data/gcp/outputs/coastal")
PATH_FINAL = {
    "fulladapt": _DIR_OUTPUTS / "pyCIAM_outputs_inequality_1000_ssp234_v3.zarr",
    "glocal": _DIR_OUTPUTS / "pyCIAM_outputs_inequality_1000_ssp234_v3_glocal.zarr",
    "global": _DIR_OUTPUTS
    / "pyCIAM_outputs_inequality_1000_ssp234_v3_global_income_climate.zarr",
}

# Views of the final store: gadmid-labelled, then subset to the original
# run's regions, which drops ~1,500 inland-water ones. The published glocal
# store never got the coastal subset; stage 6 applies the full chain to
# every scenario.
def _derived(path, suffix):
    return path.parent / path.name.replace(".zarr", f"_{suffix}.zarr")


PATH_FINAL_GADMID = {s: _derived(p, "gadmid") for s, p in PATH_FINAL.items()}
PATH_FINAL_COASTAL = {s: _derived(p, "gadmid_coastal") for s, p in PATH_FINAL.items()}

# Reference for the coastal-gadmid subset: the original run's output, whose
# gadmid set excludes the ~1,500 inland-water regions.
PATH_OLD_OUTPUT = _DIR_OUTPUTS / "pyCIAM_outputs_inequality_1000_ssp234.zarr"

IAM_RENAME = {"IIASA": "IIASA GDP", "OECD": "OECD Env-Growth"}

# =============================================================================
# RUN PARAMETERS
# =============================================================================
SEG_VAR = "seg_ir"
AGG_VAR = "impact_region"
MC_DIM = "sample"

SEG_CHUNKSIZE = 2
SAMPLE_CHUNKSIZE = 100
REFA_SEG_CHUNKSIZE = 15
OPT_SAMPLE_CHUNKSIZE = 250

OUTPUT_YEARS = [2050, 2090]
OUTPUT_SSPS = ["SSP2", "SSP3", "SSP4"]
OUTPUT_CASES = ["noAdaptation", "optimalfixed"]

N_WORKERS = 600
SLR_N_WORKERS = 40
AGG_N_WORKERS = 200
BATCH_SIZE = 300
REFA_BATCH_SIZE = 100

# --test runs use these everywhere, so the stages stay consistent with each
# other (the test SLR store has TEST_N_SAMPLES samples, and refA/calc/optimize
# must use the same segments and sample count)
TEST_TLIM_SCENARIOS = ["tlim2.0"]
TEST_WORKFLOWS = ["wf_1f"]
TEST_N_SAMPLES = 5
TEST_N_SEGS = 50
TEST_N_WORKERS = 10


def test_path(path):
    """Sibling store path used by --test runs."""
    return path.parent / path.name.replace(".zarr", "-test.zarr")
