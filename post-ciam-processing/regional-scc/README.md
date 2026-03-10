# Post-CIAM Coastal Damages Processing

This directory contains scripts to process pyCIAM coastal damage model outputs into integration-ready datasets for the Regional SCC pipeline.

These scripts convert raw CIAM outputs (total damages per coastal segment) into per-capita damages binned by GMSL, year, SSP, and growth model.

**Key features:**
- Batch processing for memory handling (500MB per batch instead)
- Does NOT include VSL (Value of Statistical Life)
- Simplified adaptation of the original [notebook](https://gitlab.com/ClimateImpactLab/coastal/projects/glo-co/-/blob/master/code/pipeline/data_processing/pyCIAM/post-CIAM/process-projected-pyCIAM-damages.ipynb)

## Files

| File | Description |
|------|-------------|
| `process_coastal_damages.py` | Main processing script - converts pyCIAM outputs to per-capita damages |
| `combine_scenarios.py` | Combines separate scenario files into single zarr with `adapt_type` dimension |
| `config.py` | Configuration settings and GCS paths |
| `README.md` | This file |

## Dependencies

```
numpy
pandas
xarray
dask
gcsfs
tqdm
cloudpathlib
pyarrow
```

Install with:
```bash
pip install numpy pandas xarray dask gcsfs tqdm cloudpathlib pyarrow
```

## Usage

### 1. Process all scenarios

```bash
# Full run (all 3 adaptation scenarios)
python process_coastal_damages.py --scenario all

# Test mode (50 regions only)
python process_coastal_damages.py --test

# Single scenario
python process_coastal_damages.py --scenario optimal
```

### 2. Combine scenarios into single file

```bash
python combine_scenarios.py --force
```

This creates `coastal-proj-combined.zarr` with an `adapt_type` dimension containing all 3 scenarios.

## Runtime

Processing was run on https://notebooks.cilresearch.org/ with:
- CPU: Intel Xeon @ 2.20GHz (8 cores)
- Memory: 50GB RAM
- Runtime: approx 3.5 hours for all 3 scenarios

## Output Format

### Dimensions
- `adapt_type`: 3 (`fixedadapt`, `globaladapt`, `optimal`)
- `region`: 7,430 impact regions (HIERID format)
- `year`: 80 (2020-2099)
- `batch`: 15 Monte Carlo batches
- `slr`: 10 GMSL bins
- `model`: 2 (`IIASA GDP`, `OECD Env-Growth`)
- `ssp`: 3 (`SSP2`, `SSP3`, `SSP4`)

### Variables
- `histclim`: Historical climate (counterfactual) per-capita damages
- `delta`: Climate change impact (climate - histclim) per-capita damages

### Output Location (GCS)
```
gs://impactlab-data/coastal/local-scc-model/results/formatted_regional_scc/
├── fixedadapt/
│   ├── coastal-proj.zarr
│   └── coastal-proj-gmsl.zarr
├── globaladapt/
│   ├── coastal-proj.zarr
│   └── coastal-proj-gmsl.zarr
├── optimal/
│   ├── coastal-proj.zarr
│   └── coastal-proj-gmsl.zarr
└── coastal-proj-combined.zarr # after combine scenarios step
```

## Post-Processing for Integration

After generating the combined zarr file, additional steps are needed to prepare data for the Regional SCC integration pipeline.

### Convert to zarr v2 format

The processing scripts may generate zarr v3 format depending on the zarr version installed. For compatibility with downstream tools, convert to zarr v2:

1. Copy data to server:
```bash
gsutil -m cp -r gs://impactlab-data/coastal/local-scc-model/results/formatted_regional_scc/coastal-proj-combined.zarr /project/cil/gcp/integration/gmsl_update_feb2026/
```

2. Convert zarr v3 to NetCDF (requires zarr v3 environment):
```python
import xarray as xr
ds = xr.open_zarr('/project/cil/gcp/integration/gmsl_update_feb2026/coastal-proj-combined.zarr')
ds.load().to_netcdf('/project/cil/gcp/integration/gmsl_update_feb2026/coastal-proj-combined.nc', engine='h5netcdf')
```

3. Convert NetCDF to zarr v2 (requires zarr v2 environment):
```python
import xarray as xr
ds = xr.open_dataset('/project/cil/gcp/integration/gmsl_update_feb2026/coastal-proj-combined.nc')
ds.to_zarr('/project/cil/gcp/integration/gmsl_update_feb2026/coastal-proj-optimal.zarr', mode='w')
```

### Filter to optimal scenario and fix chunking

For integration, filter to optimal scenario and ensure correct chunking (on RCC):

```python
import xarray as xr

NC_PATH = '/project/cil/gcp/integration/gmsl_update_feb2026/coastal-proj-combined.nc'
ZARR_V2_PATH = '/project/cil/gcp/integration/float32/sectoral_ir_damages_replication/coastal_data/coastal_damages_v0.23-optimal-iso.zarr'

# Load and filter to optimal
ds = xr.open_dataset(NC_PATH)
ds_optimal = ds.sel(adapt_type='optimal').drop_vars('adapt_type')

# Filter to common regions with reference (drops 11 all-NaN regions)
REFERENCE = '/project/cil/gcp/integration/float32/sectoral_ir_damages_replication/coastal_data/coastal_damages_v0.22-optimal-iso.zarr'
ds_ref = xr.open_zarr(REFERENCE)
common_regions = sorted(set(ds_optimal.region.values) & set(ds_ref.region.values))
ds_filtered = ds_optimal.sel(region=common_regions)

# Add metadata
ds_filtered.attrs = {
    # verify with Ian
}

ds_filtered.to_zarr(ZARR_V2_PATH, mode='w')
```

### Fix batch chunking

After saving, ensure the `batch` dimension is properly chunked:

```python
import xarray as xr

ZARR_V2_PATH = '/project/cil/gcp/integration/float32/sectoral_ir_damages_replication/coastal_data/coastal_damages_v0.23-optimal-iso.zarr'

ds = xr.open_zarr(ZARR_V2_PATH).chunk({'batch': 15})

# Clear encoding to avoid conflicts
for var in ds.data_vars:
    ds[var].encoding.clear()
for coord in ds.coords:
    ds[coord].encoding.clear()

ds.to_zarr(ZARR_V2_PATH, mode='w')

# Verify
print(xr.open_zarr(ZARR_V2_PATH).chunks['batch'])
```

## Final Output Locations

| File | Location |
|------|----------|
| GCS (all scenarios, zarr v3) | `gs://impactlab-data/coastal/local-scc-model/results/formatted_regional_scc/coastal-proj-combined.zarr` |
| Server (NetCDF) | `/project/cil/gcp/integration/gmsl_update_feb2026/coastal-proj-combined.nc` |
| Server (optimal, zarr v2) | `/project/cil/gcp/integration/float32/sectoral_ir_damages_replication/coastal_data/coastal_damages_v0.23-optimal-iso.zarr` |

## Comparison with Reference

The output `v0.23` file matches the structure of the reference `v0.22` file:

| Dimension | v0.23 | v0.22 |
|-----------|-------|-------|
| region | 7,419 | 7,419 |
| year | 80 | 82 |
| batch | 15 | 15 |
| slr | 10 | 10 |
| model | 2 | 2 |
| ssp | 3 | 5 |

Key differences:
- SSPs: v0.23 has SSP2, SSP3, SSP4 only (v0.22 has all 5)
- Years: v0.23 is 2020-2099 (v0.22 includes 2018-2019)
- No VSL dimension in v0.23
