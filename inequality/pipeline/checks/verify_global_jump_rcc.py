"""
RCC-local version of verify_global_jump.py

Run:
  source activate /project/cil/home_dirs/rcc/envs/python_r_general/
  python -u verify_global_jump_rcc.py
"""

from pathlib import Path

# ---- EDIT HERE to run elsewhere: store paths and the slice under test -----
DIR = "/project/cil/gcp/inequality/coastal"
STORES = {
    "fulladapt_v2": f"{DIR}/pyCIAM_outputs_inequality_1000_ssp234_v2_gadmid_coastal.zarr",
    "glocal_v2": f"{DIR}/pyCIAM_outputs_inequality_1000_ssp234_c0.23_global.zarr",
    "global_v3": f"{DIR}/pyCIAM_outputs_inequality_1000_ssp234_v3_global_income_climate_gadmid_coastal.zarr",
}
PATH_GADMID_COUNTRY = Path(__file__).parent / "gadmid_country.csv"
SEL = dict(scenario="tlim3.0", year=2090, ssp="SSP2")
DISK_MB_PER_S = 300
# ---------------------------------------------------------------------------

import time

import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

UPLIFT = ["NOR", "SWE", "FIN", "DNK", "ISL", "CAN", "USA", "RUS", "EST", "LVA", "LTU", "GBR"]
FULL_DIMS = ("gadmid", "sample", "costtype")


def pick_case(ds):
    cases = [str(c) for c in ds.case.values] if "case" in ds.dims else []
    for want in ("optimalfixed", "globaladapt"):
        if want in cases:
            return want
    if len(cases) == 1:
        return cases[0]
    raise SystemExit(f"no optimalfixed-equivalent case among {cases}")


def pick_iam(da):
    iams = [str(v) for v in da.iam.values]
    for match in ("IIASA", "low"):
        hit = [v for v in iams if match in v]
        if hit:
            return hit[0]
    return iams[0]


def read_gb(var):
    """Bytes actually read for one slice: chunks intersecting the selection
    load whole, so point-selected dims cost a full chunk each."""
    total = var.dtype.itemsize
    for dim, chunk, size in zip(var.dims, var.data.chunksize, var.shape):
        total *= size if dim in FULL_DIMS else min(chunk, size)
    return total / 1e9


def main():
    opened = {name: xr.open_zarr(path) for name, path in STORES.items()}
    estimates = {name: read_gb(ds.costs) for name, ds in opened.items()}
    total_gb = sum(estimates.values())
    print(
        f"total read ~{total_gb:.0f} GB from disk; at ~{DISK_MB_PER_S} MB/s "
        f"expect ~{total_gb * 1000 / DISK_MB_PER_S / 60:.0f} min\n",
        flush=True,
    )

    slices = {}
    for name, ds in opened.items():
        case = pick_case(ds)
        da = ds.costs.sel(case=case, drop=True).sel(**SEL, drop=True)
        iam = pick_iam(da)
        da = da.sel(iam=iam, drop=True)
        print(
            f"{name}: case='{case}', iam='{iam}', reading ~{estimates[name]:.1f} GB:",
            flush=True,
        )
        t0 = time.time()
        with ProgressBar():
            slices[name] = da.load()
        print(f"{name}: loaded in {(time.time() - t0) / 60:.1f} min\n", flush=True)

    common = None
    for da in slices.values():
        g = set(int(x) for x in da.gadmid.values)
        common = g if common is None else common & g
    common = sorted(common)
    print(f"shared gadmids: {len(common)} "
          f"(store sizes: {[da.sizes['gadmid'] for da in slices.values()]})")
    totals = {
        name: da.sel(gadmid=common).sum("costtype") for name, da in slices.items()
    }

    print("\ngrand totals (sum gadmids, median samples) at the slice:")
    for name, t in totals.items():
        print(f"  {name:14s} {float(t.sum('gadmid').median('sample')):.3e}")

    print("\nexact-zero rate of (gadmid, sample) totals:")
    print("  (v3 is race-free, so its rate is the legitimate share of no-cost")
    print("   cells; anything above it in the v2 stores is holes-as-zeros)")
    for name, t in totals.items():
        print(f"  {name:14s} {float((t == 0).mean()):.3f}")

    mask = (totals["glocal_v2"] > 0) & (totals["global_v3"] > 0)
    g3 = float(totals["global_v3"].where(mask, 0.0).sum("gadmid").median("sample"))
    gl = float(totals["glocal_v2"].where(mask, 0.0).sum("gadmid").median("sample"))
    raw3 = float(totals["global_v3"].sum("gadmid").median("sample"))
    rawl = float(totals["glocal_v2"].sum("gadmid").median("sample"))
    print(f"\nglobal/glocal ratio: raw {raw3 / rawl:.1f}x, "
          f"on cells nonzero in both {g3 / gl:.1f}x")

    if not PATH_GADMID_COUNTRY.exists():
        print(f"\n{PATH_GADMID_COUNTRY} not found; skipping the country "
              "concentration test (copy it from the repo next to this script)")
        return
    gadmid_country = pd.read_csv(PATH_GADMID_COUNTRY).set_index("gadmid").country

    med3 = totals["global_v3"].median("sample").to_series()
    medl = totals["glocal_v2"].median("sample").to_series()
    excess = (med3 - medl).rename("excess").to_frame()
    excess["country"] = gadmid_country.reindex(excess.index).values
    by_country = excess.groupby("country").excess.sum().sort_values(ascending=False)
    total_excess = by_country.sum()

    print(f"\ntotal excess (global - glocal, per-gadmid sample medians): {total_excess:.3e}")
    print("top 15 countries by excess, with cumulative share:")
    cum = 0.0
    for country, v in by_country.head(15).items():
        cum += v
        tag = " <- uplift" if country in UPLIFT else ""
        print(f"  {country}: {v:.3e} ({cum / total_excess:.0%} cum){tag}")
    uplift_share = by_country.reindex(UPLIFT).fillna(0).sum() / total_excess
    print(f"uplift-country share of excess: {uplift_share:.0%}")

    top5 = list(by_country.head(5).index)
    top5_gadmids = [g for g in common if gadmid_country.get(g) in top5]
    comp = (
        slices["global_v3"]
        .sel(gadmid=top5_gadmids)
        .median("sample")
        .sum("gadmid")
    )
    comp = comp / comp.sum()
    print("\ncosttype shares of global_v3 cost in the top-5 excess countries:")
    for ct, share in zip(comp.costtype.values, comp.values):
        print(f"  {str(ct)}: {share:.0%}")


if __name__ == "__main__":
    main()
