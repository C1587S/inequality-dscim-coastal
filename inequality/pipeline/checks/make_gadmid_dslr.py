"""
Build gadmid_dslr.csv: how much the global-variant sea level differs from
the local one per gadmid, at tlim3.0 in 2090.

For each seg_ir, the nearest SLR site comes from the same haversine
BallTree lookup pyCIAM uses (spherical_nearest_neighbor, copied verbatim
from pyCIAM/utils.py:114 since importing pyCIAM.io needs pint_xarray),
so the mapping matches what the runs used. At that site
    dslr = median(gsl) + median(lsl_ncc) - median(lsl)
in meters. The sample dimension of these stores is rank-sorted, so the
median of the variant's comonotonic sum is the sum of the medians and only
the base store is needed. Per-gadmid values average over the gadmid's
seg_irs; lsl_med and var_med columns carry the two absolute levels.

Needs GCS auth (gcloud google_default) and scikit-learn. Reads ~4 GB;
announces sizes and shows progress.

Run from pipeline/: python3 checks/make_gadmid_dslr.py
"""

import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from sklearn.neighbors import BallTree


def spherical_nearest_neighbor(df1, df2, x1="lon", y1="lat", x2="lon", y2="lat"):
    ball = BallTree(np.deg2rad(df2[[y2, x2]]), metric="haversine")
    _, ixs = ball.query(np.deg2rad(df1[[y1, x1]]))
    return pd.Series(df2.index[ixs[:, 0]], index=df1.index)

PATH_SLR = "gs://impactlab-data-scratch/inequality-pyciam/slr/ar6-tlim-1000samples.zarr"
PATH_SLIIDERS = "gs://impactlab-data/coastal/local-scc-model/data/int/sliiders-ir.zarr"
SO = {"token": "google_default"}
SCEN, YEAR = "tlim3.0", 2090
OUT = Path(__file__).parent / "gadmid_dslr.csv"


def main():
    slr = xr.open_zarr(PATH_SLR, storage_options=SO)
    print(
        "reading ~4 GB of SLR chunks (lsl and lsl_ncc are chunked across all "
        "years, so one year costs the full year axis):",
        flush=True,
    )
    t0 = time.time()
    with ProgressBar():
        lsl = slr.lsl_msl05.sel(scenario=SCEN, year=YEAR).load()
        ncc = slr.lsl_ncc_msl05.sel(year=YEAR).load()
        gsl = slr.gsl_msl05.sel(scenario=SCEN, year=YEAR).load()
    print(f"loaded in {(time.time() - t0) / 60:.1f} min", flush=True)

    dsite = (
        float(gsl.median("sample"))
        + ncc.median("sample")
        - lsl.median("sample")
    )

    print("reading sliiders seg coords (~few MB)", flush=True)
    sl = xr.open_zarr(PATH_SLIIDERS, chunks=None, storage_options=SO)
    lonlats = pd.DataFrame(
        {
            "seg_lon": sl.seg_lon.values,
            "seg_lat": sl.seg_lat.values,
            "ir": [str(x) for x in sl.impact_region.values],
        }
    )
    slr_lonlat = slr[["lon", "lat"]].to_dataframe()
    sites = spherical_nearest_neighbor(
        lonlats, slr_lonlat, x1="seg_lon", y1="seg_lat"
    )

    df = pd.DataFrame(
        {
            "gadmid": lonlats.ir.str.split("_").str[1].astype(int),
            "dslr_med": dsite.sel(site_id=sites.values).values,
            "lsl_med": lsl.median("sample").sel(site_id=sites.values).values,
        }
    )
    out = df.groupby("gadmid").agg(
        dslr_med=("dslr_med", "mean"),
        lsl_med=("lsl_med", "mean"),
        n_seg_ir=("dslr_med", "size"),
    )
    out["var_med"] = out.lsl_med + out.dslr_med
    out.reset_index().to_csv(OUT, index=False)
    print(
        f"wrote {OUT}: {len(out)} gadmids | dslr_med (m) "
        f"p5={out.dslr_med.quantile(0.05):.3f} med={out.dslr_med.median():.3f} "
        f"p95={out.dslr_med.quantile(0.95):.3f} | "
        f"{(out.dslr_med > 0).mean():.0%} of gadmids have dslr > 0"
    )


if __name__ == "__main__":
    main()
