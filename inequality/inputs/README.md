# Inequality pyCIAM Inputs

This folder contains the notebooks and configuration needed to produce pyCIAM coastal damage estimates for the inequality analysis using temperature-limit (GWL) scenarios.

## Output Format

The final output matches the structure of `pyCIAM_outputs_inequality_1000_ssp234.zarr`:

| Dimension | Values |
|-----------|--------|
| `case` | 2: `noAdaptation`, `optimalfixed` |
| `costtype` | 6: `wetland`, `inundation`, `relocation`, `protection`, `stormCapital`, `stormPopulation` |
| `gadmid` | ~6,000 integer region IDs |
| `scenario` | 6: `ncc_ar6`, `tlim1.5`, `tlim2.0`, `tlim3.0`, `tlim4.0`, `tlim5.0` |
| `sample` | 1,000 Monte Carlo samples |
| `year` | 2: `2050`, `2090` |
| `ssp` | 3: `SSP2`, `SSP3`, `SSP4` |
| `iam` | 2: `IIASA GDP`, `OECD Env-Growth` |

**Variable:** `costs` (float32) — total damages in dollars

## Workflow

### Step 1: Process SLR Inputs
**Notebook:** `01_process_slr_inputs.ipynb`

Processes AR6 FACTS temperature-limit SLR projections into pyCIAM-ready format:
- Loads local (gridded) SLR from public AR6 bucket
- Loads global SLR from impactlab bucket
- Loads VLM (vertical land motion) from requester-pays bucket
- Downsamples 20,000 MC draws to 1,000 quantiles (500 per workflow × 2 workflows)
- Outputs zarr with dimensions: `(scenario[5], year[9], sample[1000], site_id[~50000])`

**Must run on compute cluster** (requires requester-pays bucket access)

### Step 2: Run pyCIAM
**Notebook:** `02_run_pyciam_inequality.ipynb`

Runs pyCIAM to calculate coastal damages:
1. Collapses SLIIDERS to segment level (if needed)
2. Calculates reference adaptation heights (refA) under no-climate-change
3. Runs cost calculations for all adaptation cases (noAdaptation + 4 protect + 4 retreat)
4. Optimizes to select best adaptation strategy per segment
5. Aggregates from segment-region to gadmid level
6. Filters to final output format (cases, years, SSPs)

**Must run on compute cluster** (requires Dask Gateway with 40-800 workers)

## Configuration

All paths and parameters are defined in `config.py`:

### Reused from Regional SCC Workflow
- **SLIIDERS** (`sliiders-ir.zarr`): Updated socioeconomic data
- **Surge lookup tables**: Pre-computed storm surge damage functions
- **Model parameters** (`params.json`): Same as regional SCC

### Inequality-Specific
- **SLR inputs**: Temperature-limit scenarios (tlim1.5–5.0) instead of SSP-RCP
- **Reference adaptation**: Regenerated for 1,000 samples
- **Output format**: gadmid integers, 1,000 samples, 2 years (2050, 2090)

## Data Sources

| Data | Source |
|------|--------|
| Local SLR (tlim) | `gs://ar6-lsl-simulations-public-standard/gridded/full_sample_workflows/` |
| Global SLR (tlim) | `gs://impactlab-data/coastal/data/raw/slr/ar6/ar6/global/full_sample_workflows/` |
| VLM | `gs://ar6-lsl-simulations-requesterpays-standard/gridded/full_sample_components/` |
| SLIIDERS | `gs://impactlab-data/coastal/local-scc-model/data/int/sliiders-ir.zarr` |
| Surge lookup | `gs://impactlab-data/coastal/local-scc-model/data/int/surge-lookup/` |

## Runtime

On the CIL notebooks compute cluster with 40-800 Dask workers:
- Step 1 (SLR processing): ~30 minutes
- Step 2 (pyCIAM): ~4-6 hours

## Dependencies

```
numpy
pandas
xarray
dask
dask-gateway
distributed
gcsfs
cloudpathlib
python-CIAM  # pip install python-CIAM
pint-xarray
```
