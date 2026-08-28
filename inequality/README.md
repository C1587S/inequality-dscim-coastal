# Inequality pyCIAM

Coastal damage estimates for the [inequality
analysis](https://gitlab.com/ClimateImpactLab/Impacts/inequality), using
temperature-limit (GWL) SLR scenarios.

- **`pipeline/`** — the current pipeline: one script per stage, numbered in
  run order, parameterised by scenario (`fulladapt`, `glocal`, `global`).
  Start with its README.
- **`legacy/`** — the scripts that produced the currently published stores,
  kept for provenance. Its README maps each published zarr to the script that
  made it. Do not run these.

## Output format

Final stores (per scenario, gadmid-coastal view) match the structure of the
original `pyCIAM_outputs_inequality_1000_ssp234.zarr`:

| Dimension | Values |
|-----------|--------|
| `case` | `noAdaptation`, `optimalfixed` |
| `costtype` | `wetland`, `inundation`, `relocation`, `protection`, `stormCapital`, `stormPopulation` |
| `gadmid` | 5,903 integer region IDs (coastal subset) |
| `scenario` | `ncc_ar6`, `tlim1.5`, `tlim2.0`, `tlim3.0`, `tlim4.0`, `tlim5.0` |
| `sample` | 1,000 |
| `year` | 2050, 2090 |
| `ssp` | SSP2, SSP3, SSP4 |
| `iam` | `IIASA GDP`, `OECD Env-Growth` |

**Variable:** `costs` (float32) — total damages in dollars.
