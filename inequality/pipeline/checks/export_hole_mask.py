"""
Export the detectable hole tiles of the published v2 stores as a parquet
lookup, so anyone who used them can check whether their results touched the
holes rather than take our word for it.

One row per affected (region, sample-block) pair per store, detected as
noAdaptation cells (any cost type) that are zero in that store and nonzero in
the other. The counts are FLOORS: a tile that failed in both runs reads
zero-equals-zero and is invisible to any comparison between the stores.

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
