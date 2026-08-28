"""
Diagnose the fulladapt-vs-glocal noAdaptation mismatch found by
verify_shared_refa.py (78.75% exact-equal non-storm, expected ~100%).

Coordinate identity is already established: the first script aligned with
join="exact", which raises unless every dimension has identical values in
identical order. This script works through the remaining explanations:

1. NaN masks — cells present in one store and missing in the other are
   structural (incomplete runs), not physical.
2. Zeros — cells that are zero in both stores count as "equal" but carry no
   information. protection is identically zero under noAdaptation, and
   wetland/inundation are zero-heavy, so the interesting number is the
   equal fraction among NONZERO cells, per cost type.
3. Noise vs signal — differing cells bucketed by relative difference
   (>1e-6, >1e-3, >10%). Float noise from run nondeterminism looks like
   tiny relative differences; refA or input changes look like large ones.
4. Concentration — differing cells broken down by costtype, scenario, year,
   sample block (tasks wrote 100-sample groups; block-shaped patterns point
   at run history, not physics), and region (scattered vs concentrated).
   Plus the coordinates and values of the single worst-differing cell.
5. Storm-free regions — regions with zero storm cost in both stores are
   unreachable by rho through any channel, so any non-storm difference
   there is unexplained by the rho modification.

Also reruns the rich/poor storm-ratio direction check, computing before the
median so dask's nanmedian limitation is not hit.

Reads ~80 GB from the stores; run on the hub.
  python -u checks/diagnose_refa_mismatch.py
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


def per_costtype_stats(full, gloc):
    print("per-costtype decomposition (noAdaptation):")
    print(
        f"  {'costtype':<16} {'nan-mismatch':>12} {'both-zero':>12} "
        f"{'equal-nonzero':>14} {'diff':>12} {'rel>1e-6':>10} {'rel>1e-3':>10} "
        f"{'rel>10%':>10} {'max|diff|':>12}"
    )
    for ct in NONSTORM + STORM:
        a = full.sel(costtype=ct)
        b = gloc.sel(costtype=ct)
        both_nan = a.isnull() & b.isnull()
        nan_mismatch = a.isnull() != b.isnull()
        equal = (a == b) | both_nan
        differs = ~equal & ~nan_mismatch
        both_zero = (a == 0) & (b == 0)
        absdiff = abs(a - b)
        # NaN denominator where both values are zero, so the division neither
        # warns nor counts in the rel buckets (such cells are equal anyway)
        denom = np.maximum(abs(a), abs(b))
        rel = absdiff / denom.where(denom > 0)
        stats = xr.Dataset(
            {
                "n_nan_mismatch": nan_mismatch.sum(),
                "n_both_zero": both_zero.sum(),
                "n_equal_nonzero": (equal & ~both_zero & ~both_nan).sum(),
                "n_diff": differs.sum(),
                "n_rel_1e6": (differs & (rel > 1e-6)).sum(),
                "n_rel_1e3": (differs & (rel > 1e-3)).sum(),
                "n_rel_10pct": (differs & (rel > 0.1)).sum(),
                "max_absdiff": absdiff.max(),
            }
        ).compute()
        print(
            f"  {ct:<16} {int(stats['n_nan_mismatch']):>12} "
            f"{int(stats['n_both_zero']):>12} {int(stats['n_equal_nonzero']):>14} "
            f"{int(stats['n_diff']):>12} {int(stats['n_rel_1e6']):>10} "
            f"{int(stats['n_rel_1e3']):>10} {int(stats['n_rel_10pct']):>10} "
            f"{float(stats['max_absdiff']):>12.4g}"
        )


def diff_mask(full, gloc, kinds):
    a = full.sel(costtype=kinds)
    b = gloc.sel(costtype=kinds)
    mask = ((a != b) & a.notnull() & b.notnull()) | (a.isnull() != b.isnull())
    return mask.compute()


def concentration(mask):
    total = int(mask.sum())
    print(f"non-storm differing cells: {total}")
    for dim in ["costtype", "scenario", "year", "ssp", "iam"]:
        counts = mask.sum([d for d in mask.dims if d != dim])
        parts = ", ".join(
            f"{str(v)}: {int(c)}" for v, c in zip(mask[dim].values, counts.values)
        )
        print(f"  by {dim}: {parts}")

    per_sample = mask.sum([d for d in mask.dims if d != "sample"]).values
    blocks = per_sample.reshape(-1, SAMPLE_BLOCK).sum(axis=1)
    print(f"  by sample block of {SAMPLE_BLOCK}: {list(blocks.astype(int))}")

    per_region = mask.sum([d for d in mask.dims if d != "impact_region"])
    hit = per_region.where(per_region > 0, drop=True)
    cells_per_region = int(mask.size / mask.sizes["impact_region"])
    print(
        f"  regions with any diff: {hit.sizes['impact_region']}"
        f"/{mask.sizes['impact_region']}"
    )
    if hit.sizes["impact_region"]:
        q = np.quantile(hit.values, [0.05, 0.5, 0.95]) / cells_per_region
        print(
            f"  among those, fraction of the region's cells differing: "
            f"p5={q[0]:.3f} median={q[1]:.3f} p95={q[2]:.3f}"
        )
        top = hit.sortby(hit, ascending=False).isel(impact_region=slice(10))
        print("  top regions by differing cells:")
        for r, c in zip(top.impact_region.values, top.values):
            print(f"    {r}: {int(c)} ({int(c) / cells_per_region:.2f} of region)")
    return per_region


def worst_cell(full, gloc):
    a = full.sel(costtype=NONSTORM)
    b = gloc.sel(costtype=NONSTORM)
    absdiff = abs(a - b).fillna(0)
    idx = {k: int(v) for k, v in absdiff.argmax(dim=absdiff.dims).items()}
    cell = absdiff.isel(idx)
    coords = {d: cell[d].item() for d in cell.coords if cell[d].ndim == 0}
    va = float(a.isel(idx))
    vb = float(b.isel(idx))
    print(f"worst non-storm cell: {coords}")
    print(f"  fulladapt={va:.6g}, glocal={vb:.6g}, diff={va - vb:.6g}")


def stormfree_regions(full, gloc, per_region_diffs):
    dims = [d for d in full.dims if d != "impact_region"]
    sf = full.sel(costtype=STORM).fillna(0).sum(dims).compute()
    sg = gloc.sel(costtype=STORM).fillna(0).sum(dims).compute()
    stormfree = (sf == 0) & (sg == 0)
    n_free = int(stormfree.sum())
    diffs_in_free = int(per_region_diffs.where(stormfree, 0).sum())
    print(f"storm-free regions (zero storm cost in both stores): {n_free}")
    print(f"  non-storm differing cells within them: {diffs_in_free}")
    print("  (rho cannot reach these regions; any diff here is not explained by rho)")


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
    print("storm cost ratio glocal/fulladapt (tlim2.0, 2090, SSP2, median sample):")
    for country in RICH + POOR:
        irs = list(ir_country.index[ir_country == country])
        fc = f.sel(impact_region=irs).sum("impact_region").compute()
        gc = g.sel(impact_region=irs).sum("impact_region").compute()
        ratio = float(gc.median("sample")) / float(fc.median("sample"))
        expect = ">1" if country in RICH else "<1"
        print(f"  {country}: {ratio:.3f} (expect {expect})")


def main():
    full = xr.open_zarr(PATH_FULL).costs.sel(case="noAdaptation")
    gloc = xr.open_zarr(PATH_GLOC).costs.sel(case="noAdaptation")
    full, gloc = xr.align(full, gloc, join="exact")
    print("coordinates identical on every dimension (align join='exact' passed)\n")

    per_costtype_stats(full, gloc)
    print()
    mask = diff_mask(full, gloc, NONSTORM)
    per_region = concentration(mask)
    print()
    worst_cell(full, gloc)
    print()
    stormfree_regions(full, gloc, per_region)
    print()
    storm_direction(full, gloc)


if __name__ == "__main__":
    main()
