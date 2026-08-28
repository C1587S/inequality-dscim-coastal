# Legacy run scripts — provenance of the published stores

These files are kept because they are the only record of how the existing
zarrs in `gs://impactlab-data/gcp/outputs/coastal/` were produced. They are
superseded by `../pipeline/` and should not be run again.

## Which store came from which script

`pyCIAM_outputs_inequality_1000_ssp234_v2.zarr` (fulladapt, impact_region
level) was produced across four scripts, in order:

1. `00_full_run.py` — completed the SLR store, SLIIDERS collapse, and refA,
   then crashed mid-calc (all ~10k tasks submitted at once killed the
   scheduler; hence no `full_run_report.json`).
2. `00_resume_run.py` — added batching; scheduler still died.
3. `00_finish_run.py` — added auto-reconnect and check=True gap-fill;
   completed (20.2 h) but with only 5 scenarios (no ncc_ar6). Superseded.
4. `00_finish_v2.py` — full re-run of calc/optimize/aggregate against a
   6-scenario template with the IAM rename; this produced the store
   (25.1 h, `finish_v2_report.json`).

**KNOWN DEFECT (confirmed Aug 2026): the noAdaptation case of this store is
incomplete** — failed refA groups in `00_full_run.py` left NaN noAdaptation
costs that the aggregation's skipna sum turned into zeros (3,056 of 7,430
regions are entirely zero; USA storm costs read as zero). optimalfixed has no
holes but ran its case selection without the noAdaptation candidate in the
hole tiles, overstating costs where no-adaptation would have won. The defect
propagates to every store derived from v2 (`_gadmid`, `_gadmid_coastal`,
`_gadmid_shared`, the cluster copy `_c0.23.zarr`). The glocal store below
carries the same tile-structured defect at roughly a tenth the size (38M
cells across 2,858 regions), inherited by its derivatives
(`_globaladapt_gadmid`, `_c0.23_global`). Do not use either published
noAdaptation case; see `../pipeline/README.md` for the full mechanism and
measurements. Corrected outputs are published under `_v3` names.

`..._v2_gadmid.zarr` — `03_map_to_gadmid.py` from the v2 store.
`..._v2_gadmid_coastal.zarr` — `04_subset_coastal_gadmids.py` from _gadmid,
dropping ~1,500 inland-water gadmids (7,430 -> 5,903).

`..._v2_globaladapt.zarr` (glocal: global average rho, local SLR;
impact_region level) — `05_globaladapt.py` end-to-end (21.7 h,
`globaladapt_report.json`). Note: despite its step-3 docstring, its refA is
identical to the fulladapt refA — the seg-level collapse recomputes rho from
ypcc, discarding the modification. See `../pipeline/README.md`.

`..._v2_globaladapt_gadmid.zarr`, `..._v2_globaladapt_only.zarr` (single
case relabelled "globaladapt"), and `..._v2_gadmid_shared.zarr` were produced
**off-repo**; no script here generates them. The globaladapt gadmid stores
never received the coastal subset (still 7,430 regions), so they do not match
the 5,903-region fulladapt coastal store.

The cluster copies `..._c0.23.zarr` and `..._c0.23_global.zarr` are renames
of v2 and v2_globaladapt: the file named "global" is the glocal scenario.

## Inputs

The SLR store (`ar6-tlim-slr-1000samples.zarr`, since deleted from scratch)
was built by `01_process_slr_inputs.ipynb` / the SLR section of
`00_full_run.py`, with seeded quantile draws — `../pipeline/01_process_slr.py`
reproduces it.

## Other files

- `config.py` — the paths these scripts import (`from config import *`)
- `00a_validate_access.ipynb` — preflight checks against the old paths
- `00b_test_run.py`, `00b_test_run_and_estimate_costs.ipynb` — one-off cost
  estimation (`test_run_report.json`)
- `02_run_pyciam_inequality*.ipynb` — notebook precursors of `00_full_run.py`
- `adapt_scenarios.ipynb`, `tests.ipynb` — scratch exploration (the rho/income
  analysis and bucket forensics)
