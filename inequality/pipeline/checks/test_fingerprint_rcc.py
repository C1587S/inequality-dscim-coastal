"""
Test what drives the global-vs-glocal cost excess, on RCC, with statistics
that survive the v2 holes.

The naive excess (per-gadmid sample medians, global minus glocal) is
contaminated: 82% of glocal's (gadmid, sample) totals are zeros from holes,
so its per-gadmid medians collapse to zero and the "excess" ranking
degenerates into a ranking of global costs, which is an exposure map. Here
every per-gadmid statistic is a mean over the samples where BOTH stores are
nonzero, so holes drop out instead of counting as zero cost.

The mechanism question: does the excess track the sea-level change the
homogenisation actually imposed? gadmid_dslr.csv carries, per gadmid, the
median-SLR difference between the global variant and the local store
(tlim3.0, 2090, pyCIAM's own seg-to-site mapping). If the excess and the
cost ratio concentrate where dslr > 0 and vanish where dslr <= 0, the jump
is the sea-level swap working as designed, and the dslr>0 geography (not a
guessed mechanism) is what the scenario description should say. If the
excess is as large where dslr <= 0, the jump is not an SLR effect and the
variant store needs different scrutiny.

Needs gadmid_dslr.csv and gadmid_country.csv next to this script. Announces
read sizes with a progress bar; single machine, no cluster.

Run:
  source activate /project/cil/home_dirs/rcc/envs/python_r_general/
  python -u test_fingerprint_rcc.py
"""

from pathlib import Path

# ---- EDIT HERE to run elsewhere: store paths and the slice under test -----
DIR = "/project/cil/gcp/inequality/coastal"
STORES = {
    "glocal_v2": f"{DIR}/pyCIAM_outputs_inequality_1000_ssp234_c0.23_global.zarr",
    "global_v3": f"{DIR}/pyCIAM_outputs_inequality_1000_ssp234_v3_global_income_climate_gadmid_coastal.zarr",
}
PATH_DSLR = Path(__file__).parent / "gadmid_dslr.csv"
PATH_COUNTRY = Path(__file__).parent / "gadmid_country.csv"
SEL = dict(scenario="tlim3.0", year=2090, ssp="SSP2")
MIN_COMMON_SAMPLES = 50
DISK_MB_PER_S = 300
# ---------------------------------------------------------------------------

import time

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

UPLIFT = ["NOR", "SWE", "FIN", "DNK", "ISL", "CAN", "USA", "RUS", "EST", "LVA", "LTU", "GBR"]
FULL_DIMS = ("gadmid", "sample", "costtype")
BIN_EDGES = [-np.inf, -0.05, 0.0, 0.05, 0.20, np.inf]
BIN_LABELS = ["< -5cm", "-5..0cm", "0..5cm", "5..20cm", "> 20cm"]


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
    total = var.dtype.itemsize
    for dim, chunk, size in zip(var.dims, var.data.chunksize, var.shape):
        total *= size if dim in FULL_DIMS else min(chunk, size)
    return total / 1e9


def main():
    opened = {name: xr.open_zarr(path) for name, path in STORES.items()}
    total_gb = sum(read_gb(ds.costs) for ds in opened.values())
    print(
        f"total read ~{total_gb:.0f} GB from disk; at ~{DISK_MB_PER_S} MB/s "
        f"expect ~{total_gb * 1000 / DISK_MB_PER_S / 60:.0f} min\n",
        flush=True,
    )

    totals = {}
    for name, ds in opened.items():
        case = pick_case(ds)
        da = ds.costs.sel(case=case, drop=True).sel(**SEL, drop=True)
        iam = pick_iam(da)
        print(f"{name}: case='{case}', iam='{iam}', reading ~{read_gb(ds.costs):.1f} GB:", flush=True)
        t0 = time.time()
        with ProgressBar():
            totals[name] = da.sel(iam=iam, drop=True).load().sum("costtype")
        print(f"{name}: loaded in {(time.time() - t0) / 60:.1f} min\n", flush=True)

    common = sorted(
        set(int(x) for x in totals["glocal_v2"].gadmid.values)
        & set(int(x) for x in totals["global_v3"].gadmid.values)
    )
    gl = totals["glocal_v2"].sel(gadmid=common)
    g3 = totals["global_v3"].sel(gadmid=common)
    print(f"shared gadmids: {len(common)}")

    # per-gadmid means over samples nonzero in BOTH stores; holes drop out
    m = (gl > 0) & (g3 > 0)
    n_common = m.sum("sample")
    mean_l = gl.where(m).mean("sample")
    mean_g = g3.where(m).mean("sample")
    df = pd.DataFrame(
        {
            "n_common": n_common.values,
            "mean_l": mean_l.values,
            "mean_g": mean_g.values,
        },
        index=pd.Index(common, name="gadmid"),
    )
    kept = df[df.n_common >= MIN_COMMON_SAMPLES].copy()
    print(
        f"gadmids with >= {MIN_COMMON_SAMPLES} commonly-nonzero samples: "
        f"{len(kept)} of {len(df)}"
    )
    kept["excess"] = kept.mean_g - kept.mean_l

    dslr = pd.read_csv(PATH_DSLR).set_index("gadmid")
    country = pd.read_csv(PATH_COUNTRY).set_index("gadmid").country
    kept = kept.join(dslr[["dslr_med"]], how="inner")
    kept["country"] = country.reindex(kept.index).values
    print(f"joined with dslr: {len(kept)} gadmids")

    rho = kept.dslr_med.corr(np.log(kept.mean_g / kept.mean_l), method="spearman")
    print(f"\nSpearman(dslr, log cost ratio) across gadmids: {rho:.2f}")

    kept["bin"] = pd.cut(kept.dslr_med, BIN_EDGES, labels=BIN_LABELS)
    total_excess = kept.excess.sum()
    print(f"\ntotal excess on common support: {total_excess:.3e}")
    print(f"{'dslr bin':>10} {'gadmids':>8} {'excess':>11} {'share':>7} {'g/l ratio':>10}")
    for label in BIN_LABELS:
        b = kept[kept.bin == label]
        if not len(b):
            continue
        ratio = b.mean_g.sum() / b.mean_l.sum()
        print(
            f"{label:>10} {len(b):>8} {b.excess.sum():>11.3e} "
            f"{b.excess.sum() / total_excess:>7.0%} {ratio:>10.1f}x"
        )
    print(
        "mechanism holds if excess and ratio concentrate in dslr > 0 bins\n"
        "and the dslr <= 0 bins sit near ratio 1"
    )

    by_country = kept.groupby("country").excess.sum().sort_values(ascending=False)
    print("\ntop 10 countries by common-support excess:")
    cum = 0.0
    for c, v in by_country.head(10).items():
        sub = kept[kept.country == c]
        wmean_dslr = (sub.dslr_med * sub.mean_l).sum() / sub.mean_l.sum()
        pos_share = sub[sub.dslr_med > 0].excess.sum() / sub.excess.sum() if sub.excess.sum() else np.nan
        cum += v
        tag = " <- uplift" if c in UPLIFT else ""
        print(
            f"  {c}: {v:.3e} ({cum / total_excess:.0%} cum) | "
            f"cost-weighted dslr {wmean_dslr * 100:.0f}cm | "
            f"{pos_share:.0%} of excess in dslr>0 gadmids{tag}"
        )


if __name__ == "__main__":
    main()
