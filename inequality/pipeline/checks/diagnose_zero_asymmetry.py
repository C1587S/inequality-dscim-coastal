"""
Separate "refA values differed" from "a store's noAdaptation case is
incomplete", via asymmetry.

Context: diagnose_refa_mismatch.py showed the two published outputs differ
by >10% relative on essentially every nonzero non-storm noAdaptation cell,
including 1.83M cells inside storm-free regions where rho cannot act. The
difference is real and is not rho. Two readings fit: refA genuinely differed
(threshold effects flip inundation on/off, in both directions), or one
store's noAdaptation case is incomplete.

Why incompleteness would be invisible: refA feeds only the noAdaptation case.
00_full_run.py's refA stage logged errors and continued, leaving NaN refA for
failed (seg x 100-sample) groups; calc then wrote NaN noAdaptation costs for
those tiles while every other case computed fine. The aggregation's skipna
sum turned NaN seg contributions into zeros at impact_region level, and the
legacy verification assert (sum over costtype, skipna, then notnull) passes
over all-NaN cells. Holes masquerade as zeros, and only in noAdaptation.

Discriminators, sharpest first:
1. Storm zero-asymmetry: (1 - rho) is never zero, so a cell with zero storm
   cost in exactly one store cannot be a rho effect under any reading.
2. Non-storm zero-asymmetry: fulladapt==0 & glocal!=0 vs the reverse.
   One-sided => incompleteness in the zero-heavy store; refA threshold
   effects should produce both directions.
3. Sign among both-nonzero differing cells: missing seg_irs deflate a
   region's sum, making the incomplete store systematically smaller.
4. Tile structure: refA and calc both wrote 100-sample blocks, so holes
   should form (region x sample-block) tiles. Reported per block, and as a
   region-by-block table for the most affected regions.

Run on the hub: python -u checks/diagnose_zero_asymmetry.py
"""

import numpy as np
import pandas as pd
import xarray as xr

DIR = "gs://impactlab-data/gcp/outputs/coastal"
PATH_FULL = f"{DIR}/pyCIAM_outputs_inequality_1000_ssp234_v2.zarr"
PATH_GLOC = f"{DIR}/pyCIAM_outputs_inequality_1000_ssp234_v2_globaladapt.zarr"
PATH_SLIIDERS = "gs://impactlab-data/coastal/local-scc-model/data/int/sliiders-ir.zarr"

NONSTORM = ["wetland", "inundation", "relocation", "protection"]
STORM = ["stormCapital", "stormPopulation"]
RICH = ["USA", "NOR", "QAT"]
POOR = ["COD", "BGD", "MOZ"]
SAMPLE_BLOCK = 100
TOP_N = 10


def zero_asymmetry(full, gloc, kinds, label):
    a = full.sel(costtype=kinds)
    b = gloc.sel(costtype=kinds)
    ok = a.notnull() & b.notnull()
    a_zero = ok & (a == 0) & (b != 0)
    b_zero = ok & (b == 0) & (a != 0)
    nz_diff = ok & (a != 0) & (b != 0) & (a != b)
    stats = xr.Dataset(
        {
            "full_zero_gloc_not": a_zero.sum(),
            "gloc_zero_full_not": b_zero.sum(),
            "both_nonzero_diff": nz_diff.sum(),
            "full_smaller": (nz_diff & (a < b)).sum(),
            "full_larger": (nz_diff & (a > b)).sum(),
        }
    ).compute()
    print(f"{label}:")
    print(f"  fulladapt==0, glocal!=0: {int(stats['full_zero_gloc_not'])}")
    print(f"  glocal==0, fulladapt!=0: {int(stats['gloc_zero_full_not'])}")
    print(f"  both nonzero, differing: {int(stats['both_nonzero_diff'])}")
    print(
        f"    of which fulladapt smaller: {int(stats['full_smaller'])}, "
        f"larger: {int(stats['full_larger'])}"
    )
    return a_zero, b_zero


def tile_structure(mask, name):
    """Where the asymmetric zeros sit: per region, per sample block, and the
    joint region-by-block table for the most affected regions."""
    per_region = mask.sum([d for d in mask.dims if d != "impact_region"]).compute()
    hit = per_region.where(per_region > 0, drop=True)
    cells_per_region = int(mask.size / mask.sizes["impact_region"])
    print(f"{name}: {int(per_region.sum())} cells in "
          f"{hit.sizes['impact_region']}/{mask.sizes['impact_region']} regions")
    if not hit.sizes["impact_region"]:
        return

    per_sample = mask.sum([d for d in mask.dims if d != "sample"]).compute()
    blocks = per_sample.values.reshape(-1, SAMPLE_BLOCK).sum(axis=1)
    print(f"  by sample block of {SAMPLE_BLOCK}: {list(blocks.astype(int))}")

    frac = hit.values / cells_per_region
    q = np.quantile(frac, [0.05, 0.5, 0.95])
    print(
        f"  affected regions, fraction of region's cells: "
        f"p5={q[0]:.3f} median={q[1]:.3f} p95={q[2]:.3f}"
    )

    top = hit.sortby(hit, ascending=False).isel(impact_region=slice(TOP_N))
    joint = (
        mask.sel(impact_region=top.impact_region.values)
        .sum([d for d in mask.dims if d not in ("impact_region", "sample")])
        .compute()
        .transpose("impact_region", "sample")
    )
    per_block = joint.values.reshape(joint.sizes["impact_region"], -1, SAMPLE_BLOCK).sum(
        axis=2
    )
    print(f"  top regions x sample-block counts (blocks of {SAMPLE_BLOCK}):")
    for region, row in zip(joint.impact_region.values, per_block.astype(int)):
        print(f"    {region}: {list(row)}")


def fully_zero_regions(full, gloc, kinds):
    dims_a = [d for d in full.dims if d != "impact_region"]
    az = (full.sel(costtype=kinds).fillna(0) == 0).all(dims_a).compute()
    bz = (gloc.sel(costtype=kinds).fillna(0) == 0).all(dims_a).compute()
    print("regions with all-zero non-storm noAdaptation costs:")
    print(f"  fulladapt only: {int((az & ~bz).sum())}")
    print(f"  glocal only: {int((bz & ~az).sum())}")
    print(f"  both: {int((az & bz).sum())}")


def storm_direction(full, gloc):
    sl = xr.open_zarr(PATH_SLIIDERS, chunks=None)
    ir_country = (
        pd.DataFrame(
            {"ir": sl.impact_region.values, "country": sl.seg_country.values}
        )
        .groupby("ir")
        .country.first()
    )
    sel = dict(costtype=STORM, scenario="tlim2.0", year=2090, ssp="SSP2")
    f = full.sel(**sel).isel(iam=0, drop=True).sum("costtype")
    g = gloc.sel(**sel).isel(iam=0, drop=True).sum("costtype")
    store_regions = set(str(r) for r in f.impact_region.values)
    print("storm cost by country (tlim2.0, 2090, SSP2, median over samples):")
    for country in RICH + POOR:
        irs = [str(r) for r in ir_country.index[ir_country == country]]
        matched = [r for r in irs if r in store_regions]
        if not matched:
            print(f"  {country}: mapped {len(irs)} regions, 0 matched the store")
            continue
        fm = float(f.sel(impact_region=matched).sum("impact_region").compute().median("sample"))
        gm = float(g.sel(impact_region=matched).sum("impact_region").compute().median("sample"))
        ratio = f"{gm / fm:.3f}" if fm > 0 else "undefined (fulladapt zero => holes?)"
        print(
            f"  {country}: {len(matched)}/{len(irs)} regions matched | "
            f"fulladapt {fm:.4g}, glocal {gm:.4g} | ratio {ratio}"
        )
    if any(
        not [r for r in ir_country.index[ir_country == c] if str(r) in store_regions]
        for c in RICH + POOR
    ):
        print(f"  seg_country sample values: {sorted(set(ir_country.values))[:15]}")


def main():
    full = xr.open_zarr(PATH_FULL).costs.sel(case="noAdaptation")
    gloc = xr.open_zarr(PATH_GLOC).costs.sel(case="noAdaptation")
    full, gloc = xr.align(full, gloc, join="exact")

    ns_a_zero, ns_b_zero = zero_asymmetry(
        full, gloc, NONSTORM, "non-storm noAdaptation"
    )
    print()
    zero_asymmetry(
        full, gloc, STORM, "storm noAdaptation (one-sided zeros here cannot be rho)"
    )
    print()
    tile_structure(ns_a_zero, "fulladapt-zero cells")
    print()
    tile_structure(ns_b_zero, "glocal-zero cells")
    print()
    fully_zero_regions(full, gloc, NONSTORM)
    print()
    storm_direction(full, gloc)


if __name__ == "__main__":
    main()
