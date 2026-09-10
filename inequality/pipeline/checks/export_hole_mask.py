"""
Export the detectable hole tiles of the published v2 stores as a parquet
lookup, so anyone who used them can check whether their results touched the
holes rather than take our word for it.

One row per affected (region, sample-block) pair per store, detected as
noAdaptation cells (any cost type) that are zero in that store and nonzero in
the other.

Two mechanisms produced the holes. Failed refA groups left NaN noAdaptation
costs that the aggregation's skipna sum wrote out as zeros. And the run
stores chunked samples at 1000 while tasks wrote 100-sample slices, so the
ten tasks sharing each chunk raced on zarr's read-modify-write and the
losers' slices silently reverted to NaN. The refA mechanism only touches
noAdaptation; the races hit every case.

Two limits follow for auditors. The counts are FLOORS: a tile that failed in
both runs reads zero-equals-zero and is invisible to any comparison between
the stores. And the mask says nothing about optimalfixed: it probes the
noAdaptation case only.

The optimalfixed damage has since been measured, and it dwarfs what this
mask shows. Against the race-free v3 global store at optimalfixed, tlim3.0,
SSP2, IIASA, 2090, on the shared coastal gadmids: 82% of (gadmid, sample)
cost totals are exactly zero in BOTH published v2 stores, against 15% in v3
- the legitimate no-cost share. Roughly 67 points of every published
optimalfixed slice is holes read as zeros. Sums and means are understated
several-fold, and per-gadmid sample medians mostly collapse to zero
outright. Even on cells both stores wrote, glocal v2 sits ~2x below v3
uniformly across scenarios including no-climate-change, and fulladapt v2
sits 10-16x below with per-gadmid quartiles reaching 17,000 - far beyond
any income effect, matching its larger hole counts. Do not compute anything
from the published optimalfixed; use the v3 stores.

Writes:
  gs://impactlab-data/gcp/outputs/coastal/
      pyCIAM_outputs_inequality_1000_ssp234_v2_hole_mask.parquet

Columns:
  store         zarr name of the defective store the row refers to
  impact_region region id as it appears in that store
  sample_block  0-9, blocks of 100 samples (the task write granularity)
  sample_start, sample_end  the 1-based sample range of the block
  n_zero_cells  detected asymmetric-zero cells in the tile (lower bound)

Run on the hub: python -u checks/export_hole_mask.py
"""

import pandas as pd
import xarray as xr

DIR = "gs://impactlab-data/gcp/outputs/coastal"
STORES = {
    "pyCIAM_outputs_inequality_1000_ssp234_v2.zarr": None,
    "pyCIAM_outputs_inequality_1000_ssp234_v2_globaladapt.zarr": None,
}
PATH_OUT = f"{DIR}/pyCIAM_outputs_inequality_1000_ssp234_v2_hole_mask.parquet"
SAMPLE_BLOCK = 100


def pair_counts(mask):
    hits = mask.sum(
        [d for d in mask.dims if d not in ("impact_region", "sample")]
    ).compute()
    hits = hits.transpose("impact_region", "sample")
    blocks = hits.values.reshape(hits.sizes["impact_region"], -1, SAMPLE_BLOCK).sum(
        axis=2
    )
    return hits.impact_region.values, blocks


def rows_for(store_name, mask):
    regions, blocks = pair_counts(mask)
    rows = []
    for r, counts in zip(regions, blocks):
        for block, n in enumerate(counts):
            if n > 0:
                rows.append(
                    {
                        "store": store_name,
                        "impact_region": str(r),
                        "sample_block": block,
                        "sample_start": block * SAMPLE_BLOCK + 1,
                        "sample_end": (block + 1) * SAMPLE_BLOCK,
                        "n_zero_cells": int(n),
                    }
                )
    return rows


def main():
    names = list(STORES)
    full = xr.open_zarr(f"{DIR}/{names[0]}").costs.sel(case="noAdaptation")
    gloc = xr.open_zarr(f"{DIR}/{names[1]}").costs.sel(case="noAdaptation")
    full, gloc = xr.align(full, gloc, join="exact")
    ok = full.notnull() & gloc.notnull()

    rows = rows_for(names[0], ok & (full == 0) & (gloc != 0))
    rows += rows_for(names[1], ok & (gloc == 0) & (full != 0))

    df = pd.DataFrame(rows).sort_values(
        ["store", "impact_region", "sample_block"], ignore_index=True
    )
    df.to_parquet(PATH_OUT, index=False)
    print(f"wrote {len(df)} rows to {PATH_OUT}")
    for store, grp in df.groupby("store"):
        print(
            f"  {store}: {len(grp)} affected (region, block) pairs, "
            f"{grp.impact_region.nunique()} regions, "
            f"{grp.n_zero_cells.sum()} cells (floor)"
        )


if __name__ == "__main__":
    main()
