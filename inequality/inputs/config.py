"""
Configuration for inequality pyCIAM workflow.

This config uses updated paths from the regional SCC workflow where possible,
and defines the inequality-specific paths for tlim SLR scenarios.
"""

from pathlib import Path

import geopandas as gpd
import xarray as xr
from cloudpathlib import AnyPath, GSClient
from gcsfs import GCSFileSystem

# =============================================================================
# CLOUD STORAGE SETUP
# =============================================================================
FS = GCSFileSystem()
gsclient = GSClient()

STORAGE_OPTIONS = {}

def _to_fuse(path):
    """Convert gs:// path to /gcs/ fuse mount path."""
    return Path(str(path).replace("gs://", "/gcs/"))


# =============================================================================
# VERSION INFO
# =============================================================================
SLIIDERS_VERS = "v1.2"  # Updated from regional SCC workflow
RES_VERS = "inequality-v0.2"

# Output dataset attrs
HISTORY = f"""inequality-v0.2: Updated with regional SCC data (SLIIDERS {SLIIDERS_VERS}, Feb 2026)."""
AUTHOR = "Sebastian Cadavid / Ian Bolliger"
CONTACT = "ian@reask.earth"


# =============================================================================
# ROOT DIRECTORIES
# =============================================================================
# Regional SCC workflow paths (reusable data)
DIR_REGIONAL_SCC = AnyPath("gs://impactlab-data/coastal/local-scc-model")
DIR_REGIONAL_SCC_DATA = DIR_REGIONAL_SCC / "data"
DIR_REGIONAL_SCC_INT = DIR_REGIONAL_SCC_DATA / "int"
DIR_REGIONAL_SCC_RES = DIR_REGIONAL_SCC / "results"

# Scratch directory for intermediate outputs
DIR_SCRATCH = AnyPath("gs://impactlab-data-scratch/inequality-pyciam")

# Raw AR6 SLR data (for tlim scenarios)
DIR_SLR_AR6_RAW = AnyPath("gs://impactlab-data/coastal/data/raw/slr/ar6/ar6/global/full_sample_workflows")

# Public AR6 gridded SLR (for local/gridded tlim projections)
DIR_SLR_AR6_GRIDDED_PUBLIC = "gs://ar6-lsl-simulations-public-standard/gridded/full_sample_workflows"

# VLM data (requester-pays bucket - needs to be accessed from compute cluster)
PATH_VLM_REQUESTER_PAYS = "gs://ar6-lsl-simulations-requesterpays-standard/gridded/full_sample_components/verticallandmotion-kopp14-verticallandmotion_localsl.zarr"


# =============================================================================
# MODEL PARAMS
# =============================================================================
# Use params.json from repo root
PATH_PARAMS = Path(__file__).parents[2] / "params.json"


# =============================================================================
# REUSABLE INPUTS FROM REGIONAL SCC WORKFLOW
# =============================================================================
# SLIIDERS (socioeconomic data) - reuse from regional SCC
PATH_SLIIDERS = DIR_REGIONAL_SCC_INT / "sliiders-ir.zarr"
PATH_SLIIDERS_SEG = DIR_SCRATCH / "sliiders-seg-inequality.zarr"

# Surge lookup tables - reuse from regional SCC
PATHS_SURGE_LOOKUP = {
    "seg": DIR_REGIONAL_SCC_INT / "surge-lookup" / "surge-lookup-seg.zarr",
    "seg_ir": DIR_REGIONAL_SCC_INT / "surge-lookup" / "surge-lookup-seg-ir.zarr",
}

# Reference adaptation heights - need to regenerate for 1000 samples
PATH_REFA = DIR_REGIONAL_SCC_INT / "refa.zarr"  # Can use as reference
PATH_REFA_INEQUALITY = DIR_SCRATCH / "refa-inequality-1000samples.zarr"


# =============================================================================
# INEQUALITY-SPECIFIC SLR INPUTS
# =============================================================================
# Temperature limit scenarios
TLIM_SCENARIOS = ['tlim1.5', 'tlim2.0', 'tlim3.0', 'tlim4.0', 'tlim5.0']
WORKFLOWS = ['wf_1f', 'wf_2f']

# Number of samples (500 per workflow × 2 workflows = 1000)
N_SAMPLES_PER_WORKFLOW = 500
N_SAMPLES_TOTAL = N_SAMPLES_PER_WORKFLOW * len(WORKFLOWS)  # 1000

# Processed SLR output path
PATH_SLR_INEQUALITY = DIR_SCRATCH / "ar6-tlim-slr-1000samples.zarr"


# =============================================================================
# PYCIAM OUTPUTS
# =============================================================================
# Temporary output during processing
PATH_OUTPUT_TMP = DIR_SCRATCH / "pyciam-inequality-tmp.zarr"

# Final output (matches format of pyCIAM_outputs_inequality_1000_ssp234.zarr)
PATH_OUTPUT_INEQUALITY = DIR_SCRATCH / "pyCIAM_outputs_inequality_1000_ssp234_v2.zarr"

# Final output location (production)
PATH_OUTPUT_FINAL = AnyPath("gs://impactlab-data/gcp/outputs/coastal/pyCIAM_outputs_inequality_1000_ssp234_v2.zarr")


# =============================================================================
# PYCIAM RUN PARAMETERS
# =============================================================================
# Segment variable and admin variable for aggregation
SEG_VAR = "seg_ir"
ADM_VAR = "gadmid"

# Monte Carlo dimension name
MC_DIM = "sample"

# Chunking parameters for memory management
SEG_CHUNKSIZE = 3
REFA_SEG_CHUNKSIZE = 15
SAMPLE_CHUNKSIZE = 100

# Years to include in final output (matching original inequality output)
OUTPUT_YEARS = [2050, 2090]

# SSPs to include (matching original: SSP2, SSP3, SSP4)
OUTPUT_SSPS = ['SSP2', 'SSP3', 'SSP4']

# Cases to include in final output
OUTPUT_CASES = ['noAdaptation', 'optimalfixed']


# =============================================================================
# DASK CLUSTER SETTINGS
# =============================================================================
N_WORKERS_MIN = 40
N_WORKERS_MAX = 800


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def open_zarr(path, **kwargs):
    """Open a zarr store with storage options."""
    return xr.open_zarr(str(path), storage_options=STORAGE_OPTIONS, **kwargs)


def open_dataset(path, **kwargs):
    """Open a netCDF dataset via fuse mount."""
    _path = str(_to_fuse(path))
    return xr.open_dataset(_path, **kwargs)


def save_zarr(ds, path, **kwargs):
    """Save dataset to zarr with storage options."""
    for v in ds.data_vars:
        ds[v].encoding.clear()
    for k, v in ds.coords.items():
        if v.dtype == object:
            ds[k] = v.astype("unicode")
    return ds.to_zarr(str(path), storage_options=STORAGE_OPTIONS, **kwargs)
