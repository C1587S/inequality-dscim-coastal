"""
Are the glocal-zero cells tiled like the fulladapt ones?

Confirmed so far: the published fulladapt noAdaptation case has holes that
read as zeros (349M asymmetric zero cells, tile-structured). The reverse
direction — 38M non-storm cells where glocal is zero and fulladapt is not —
would mean glocal has its own tenfold-smaller holes IF those cells sit in
(region x 100-sample-block) tiles, the write granularity of the refA and calc
tasks. If they are scattered instead, it is a different phenomenon and glocal
should not be called defective on this evidence.

Method: count asymmetric-zero cells per (region, sample-block) pair. A hole
zeroes a region's cells across every costtype, scenario, year, ssp, and iam
for the affected samples, so holes produce few pairs with hits in the
thousands (tile ceiling: 4 costtypes x 6 scenarios x 2 years x 3 ssps x
2 iams x 100 samples = 28,800 cells). Scatter produces many pairs with hits
in the single digits. The fulladapt-zero cells are reported with identical
statistics as the known-holes fingerprint: same shape at ~10x scale => same
mechanism.

Run on the hub: python -u checks/tile_check_glocal.py
"""

import numpy as np
import xarray as xr

DIR = "gs://impactlab-data/gcp/outputs/coastal"
PATH_FULL = f"{DIR}/pyCIAM_outputs_inequality_1000_ssp234_v2.zarr"
PATH_GLOC = f"{DIR}/pyCIAM_outputs_inequality_1000_ssp234_v2_globaladapt.zarr"

NONSTORM = ["wetland", "inundation", "relocation", "protection"]
SAMPLE_BLOCK = 100


def per_pair_hits(mask):
    """Asymmetric-zero cells per (region, sample-block) pair."""
    hits = mask.sum(
        [d for d in mask.dims if d not in ("impact_region", "sample")]
    ).compute()
    hits = hits.transpose("impact_region", "sample")
    return hits.values.reshape(hits.sizes["impact_region"], -1, SAMPLE_BLOCK).sum(
        axis=2
    )


def report(blocks, name):
    nz = blocks[blocks > 0]
    total = int(blocks.sum())
    n_pairs = int((blocks > 0).sum())
    print(f"{name}: {total} cells in {n_pairs} (region x block) pairs")
    if not n_pairs:
        return
    q = np.quantile(nz, [0.05, 0.5, 0.95]).astype(int)
    print(f"  hits per affected pair: p5={q[0]} median={q[1]} p95={q[2]} max={int(nz.max())}")
    in_big = int(nz[nz >= 1000].sum())
    in_small = int(nz[nz <= 10].sum())
    print(
        f"  share of cells in pairs with >=1000 hits: {in_big / total:.2f} "
        f"(tiled); in pairs with <=10 hits: {in_small / total:.2f} (scattered)"
    )
    per_region = (blocks > 0).sum(axis=1)
    affected = per_region[per_region > 0]
    print(
        f"  affected regions: {len(affected)}, blocks per affected region: "
        f"median={int(np.median(affected))} max={int(affected.max())} of "
        f"{blocks.shape[1]}"
    )


def main():
    full = xr.open_zarr(PATH_FULL).costs.sel(case="noAdaptation", costtype=NONSTORM)
    gloc = xr.open_zarr(PATH_GLOC).costs.sel(case="noAdaptation", costtype=NONSTORM)
    full, gloc = xr.align(full, gloc, join="exact")
    ok = full.notnull() & gloc.notnull()

    gloc_zero = ok & (gloc == 0) & (full != 0)
    full_zero = ok & (full == 0) & (gloc != 0)

    report(per_pair_hits(gloc_zero), "glocal-zero cells (the question)")
    print()
    report(per_pair_hits(full_zero), "fulladapt-zero cells (known-holes fingerprint)")
    print()
    print(
        "verdict guide: if the two distributions have the same shape at ~10x\n"
        "scale (cells concentrated in >=1000-hit pairs), glocal has the same\n"
        "defect smaller; if glocal-zero cells sit mostly in <=10-hit pairs,\n"
        "it is a different phenomenon."
    )


if __name__ == "__main__":
    main()
