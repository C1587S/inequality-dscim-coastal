"""
RESOLVED (Sep 2026): the jump was an artifact of v2 incompleteness, not a
scenario effect. diagnose_uniform_gap_rcc.py has the closing evidence; see
the README note.

Verify the 50x jump of global (v3) over glocal (v2) at optimalfixed,
tlim3.0, SSP2, IIASA, 2090.

Hypothesis: sites protected by land uplift lose that protection under the
homogenised climate signal, cross the adapt-or-not threshold, and generate
protection and relocation costs where they had none. If true, the excess
cost must concentrate in uplift countries (Scandinavia, the Baltic, Canada,
Alaska/USA, Russia), and be dominated by protection and relocation there.
If the excess is spread evenly across countries regardless of uplift, the
mechanism is wrong and the variant store deserves suspicion instead.

Also measures how much of the ratio is the v2 stores being understated:
their optimalfixed carries race holes that aggregation turned into zeros,
so their totals are biased low. Two probes: the exact-zero rate per store
at this slice (v3 is race-free, so its rate is the legitimate baseline),
and the ratio recomputed on cells nonzero in both stores.

Run on the hub: python -u checks/verify_global_jump.py
"""

import time

import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

DIR = "gs://impactlab-data/gcp/outputs/coastal"
STORES = {
    "fulladapt_v2": f"{DIR}/pyCIAM_outputs_inequality_1000_ssp234_v2.zarr",
    "glocal_v2": f"{DIR}/pyCIAM_outputs_inequality_1000_ssp234_v2_globaladapt.zarr",
    "global_v3": f"{DIR}/pyCIAM_outputs_inequality_1000_ssp234_v3_global_income_climate.zarr",
}
PATH_SLIIDERS = "gs://impactlab-data/coastal/local-scc-model/data/int/sliiders-ir.zarr"
SEL = dict(case="optimalfixed", scenario="tlim3.0", year=2090, ssp="SSP2")
UPLIFT = ["NOR", "SWE", "FIN", "DNK", "ISL", "CAN", "USA", "RUS", "EST", "LVA", "LTU", "GBR"]


def load_slice(path, name):
    ds = xr.open_zarr(path)
    da = ds.costs.sel(**SEL, drop=True)
    iams = [v for v in da.iam.values.astype(str) if "IIASA" in v]
    da = da.sel(iam=iams[0] if iams else da.iam.values[0], drop=True)
    # store chunks span everything except region, so this reads the whole
    # optimalfixed half of the store, not just the 178 MB logical slice
    gb = ds.costs.nbytes / 2 / 1e9
    print(
        f"{name}: reading ~{gb:.0f} GB uncompressed through hub threads, "
        f"typically a few minutes:",
        flush=True,
    )
    t0 = time.time()
    with ProgressBar():
        da = da.load()
    print(f"{name}: loaded in {(time.time() - t0) / 60:.1f} min", flush=True)
    return da


def main():
    slices = {name: load_slice(path, name) for name, path in STORES.items()}
    regions = None
    for da in slices.values():
        r = set(str(x) for x in da.impact_region.values)
        regions = r if regions is None else regions & r
    regions = sorted(regions)
    print(f"common regions: {len(regions)}")
    totals = {
        name: da.sel(impact_region=regions).sum("costtype")
        for name, da in slices.items()
    }

    print("\ngrand totals (sum regions, median samples) at the slice:")
    for name, t in totals.items():
        print(f"  {name:14s} {float(t.sum('impact_region').median('sample')):.3e}")

    print("\nexact-zero rate of (region, sample) totals:")
    print("  (v3 is race-free, so its rate is the legitimate share of no-cost")
    print("   cells; anything above it in the v2 stores is holes-as-zeros)")
    for name, t in totals.items():
        print(f"  {name:14s} {float((t == 0).mean()):.3f}")

    mask = (totals["glocal_v2"] > 0) & (totals["global_v3"] > 0)
    g3 = float(totals["global_v3"].where(mask, 0.0).sum("impact_region").median("sample"))
    gl = float(totals["glocal_v2"].where(mask, 0.0).sum("impact_region").median("sample"))
    raw3 = float(totals["global_v3"].sum("impact_region").median("sample"))
    rawl = float(totals["glocal_v2"].sum("impact_region").median("sample"))
    print(f"\nglobal/glocal ratio: raw {raw3 / rawl:.1f}x, "
          f"on cells nonzero in both {g3 / gl:.1f}x")

    sl = xr.open_zarr(PATH_SLIIDERS, chunks=None)
    ir_country = (
        pd.DataFrame({"ir": sl.impact_region.values, "country": sl.seg_country.values})
        .groupby("ir")
        .country.first()
    )
    med3 = totals["global_v3"].median("sample").to_series()
    medl = totals["glocal_v2"].median("sample").to_series()
    excess = (med3 - medl).rename("excess").to_frame()
    excess["country"] = ir_country.reindex(excess.index.astype(str)).values
    by_country = excess.groupby("country").excess.sum().sort_values(ascending=False)
    total_excess = by_country.sum()

    print(f"\ntotal excess (global - glocal, per-region sample medians): {total_excess:.3e}")
    print("top 15 countries by excess, with cumulative share:")
    cum = 0.0
    for country, v in by_country.head(15).items():
        cum += v
        tag = " <- uplift" if country in UPLIFT else ""
        print(f"  {country}: {v:.3e} ({cum / total_excess:.0%} cum){tag}")
    uplift_share = by_country.reindex(UPLIFT).fillna(0).sum() / total_excess
    print(f"uplift-country share of excess: {uplift_share:.0%}")

    top5 = list(by_country.head(5).index)
    top5_regions = [r for r in regions if ir_country.get(r) in top5]
    comp = (
        slices["global_v3"]
        .sel(impact_region=top5_regions)
        .median("sample")
        .sum("impact_region")
    )
    comp = comp / comp.sum()
    print("\ncosttype shares of global_v3 cost in the top-5 excess countries:")
    for ct, share in zip(comp.costtype.values, comp.values):
        print(f"  {str(ct)}: {share:.0%}")


if __name__ == "__main__":
    main()
