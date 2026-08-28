"""
Stage 1: build the tlim SLR stores.

Reduces the 20,000 AR6 FACTS draws per scenario x workflow to 1,000
rank-quantile samples, 500 per workflow, and writes two stores.

The base store (PATH_SLR) has three variables:
    lsl_msl05      local sea level for the total workflow: fingerprinted
                   climate components plus vertical land motion
    lsl_ncc_msl05  the Kopp14 vertical-land-motion component alone, which is
                   the no-climate-change counterfactual
    gsl_msl05      global-mean sea level from the AR6 global workflows; no
                   VLM, no site dimension

The globalclimate variant (PATH_SLR_GLOBALCLIMATE) has the same layout with
lsl_msl05 = gsl_msl05 + lsl_ncc_msl05: a uniform climate signal on top of
each site's own land motion. gsl has no VLM and lsl_ncc has nothing else, so
nothing is double-counted. The "global" scenario runs against this store and
pyCIAM needs no changes.

One caveat. The local, global, and VLM series are quantiled independently,
so sample n is the rank-n quantile of each series, not a single FACTS draw.
Carry that into any description of scenarios built from these stores.

The draws are seeded (config.SEED), so a rebuild matches the store used in
the 2026 production runs.

Run on the hub:
  test:  python -u 01_process_slr.py --test
  full:  nohup python -u 01_process_slr.py > 01_process_slr.log 2>&1 &
"""

import argparse
import time
from itertools import product

import dask
import numpy as np
import pint_xarray  # noqa: F401  registers the .pint accessor
import xarray as xr
from gcsfs import GCSFileSystem

from config import (
    DIR_SLR_AR6_GRIDDED_PUBLIC,
    DIR_SLR_AR6_RAW,
    N_DRAWS,
    N_SAMPLES_PER_WORKFLOW,
    PATH_SLR,
    PATH_SLR_GLOBALCLIMATE,
    PATH_VLM_REQUESTER_PAYS,
    SEED,
    SLR_N_WORKERS,
    TEST_N_SAMPLES,
    TEST_N_WORKERS,
    TEST_TLIM_SCENARIOS,
    TEST_WORKFLOWS,
    TLIM_SCENARIOS,
    WORKFLOWS,
    test_path,
)
from runner import Cluster, clean_for_zarr, write_report, zarr_exists

dask.config.set({"array.rechunk.method": "tasks"})

STORE_CHUNKS = {"site_id": -1, "scenario": 1, "year": -1, "sample": 100}


def draw_quantiles(n):
    """Seeded rank-quantile draw: one draw from each of n equal bins of the
    20,000 FACTS samples. Reseeds on every call, matching the original runs."""
    np.random.seed(SEED)
    low = np.arange(0, N_DRAWS, step=N_DRAWS / n)
    return np.random.randint(low=low, high=low + N_DRAWS / n, size=None) / N_DRAWS


def to_meters(ds):
    ds["sea_level_change"] = (
        ds.sea_level_change.pint.quantify().pint.to("meters").pint.dequantify()
    )
    return ds


def quantile_draws(ds, quants, wf, tlim):
    return (
        ds.sel(years=slice(2020, 2100), drop=True)
        .sea_level_change.chunk(dict(samples=-1))
        .quantile(quants, dim="samples")
        .assign_coords(workflow=wf, tlim=tlim)
        .expand_dims(["workflow", "tlim"])
    )


def stack_workflows(ds, n_samples):
    """Concatenate per-workflow quantiles into a single 1..n sample index."""
    ds = (
        ds.stack(samples=["workflow", "quantile"])
        .reset_index("samples")
        .drop_vars(["workflow", "quantile"])
    )
    ds["samples"] = np.arange(1, n_samples + 1)
    return ds


def load_local_and_global(scenarios, workflows, n_per_wf):
    quants = draw_quantiles(n_per_wf)
    local, glob = [], []
    for tlim, wf in product(scenarios, workflows):
        print(f"  {tlim}/{wf}")
        lsl = to_meters(
            xr.open_zarr(
                f"{DIR_SLR_AR6_GRIDDED_PUBLIC}/{wf}/{tlim}win0.25/total-workflow.zarr"
            )
        )
        local.append(quantile_draws(lsl, quants, wf, tlim))
        gsl = to_meters(
            xr.open_dataset(
                str(DIR_SLR_AR6_RAW / wf / f"{tlim}win0.25" / "total-workflow.nc")
                .replace("gs://", "/gcs/")
            )
        )
        glob.append(quantile_draws(gsl, quants, wf, tlim))
    return local, glob


def load_vlm(n_samples):
    """The VLM component, quantiled to n_samples. Requester-pays bucket, so
    this must run where the cluster credentials allow it."""
    fs = GCSFileSystem(requester_pays=True)
    vlm = to_meters(xr.open_zarr(fs.get_mapper(PATH_VLM_REQUESTER_PAYS)))
    return (
        vlm.sel(years=slice(2020, 2100), drop=True)
        .sea_level_change.chunk(dict(samples=-1))
        .quantile(draw_quantiles(n_samples), dim="samples")
        .rename({"quantile": "samples"})
        .to_dataset()
    )


def combine(local, glob, vlm, n_samples):
    lsl = stack_workflows(xr.combine_by_coords(local), n_samples)
    gsl = stack_workflows(xr.combine_by_coords(glob), n_samples).squeeze(drop=True)
    vlm["samples"] = np.arange(1, n_samples + 1)

    ds = xr.Dataset(
        {
            "lsl_msl05": lsl.sea_level_change,
            "lsl_ncc_msl05": vlm.sea_level_change,
            "gsl_msl05": gsl.sea_level_change,
            "lon": vlm.lon,
            "lat": lsl.lat,
        }
    )
    ds["lon"] = ds.lon.where(ds.lon != -180, 180)
    ds = ds.stack(locations=["lat", "lon"]).persist()

    # keep only sites with complete local series through 2100
    reduce_dims = [d for d in ["tlim", "samples", "years"] if d in ds.dims]
    valid = (
        ds[["lsl_msl05", "lsl_ncc_msl05"]]
        .sel(years=slice(2100))
        .notnull()
        .all(reduce_dims)
        .to_array("tmp")
        .all("tmp")
    ).compute()
    ds = ds.sel(locations=valid.where(valid, drop=True).locations)

    ds = ds.rename(
        {"years": "year", "samples": "sample", "locations": "site_id", "tlim": "scenario"}
    )
    ds = ds.chunk(STORE_CHUNKS)

    # replace the (lat, lon) MultiIndex with integer site ids
    lat, lon = ds.lat.values, ds.lon.values
    ds = ds.drop_vars(["lat", "lon"])
    ds["site_id"] = np.arange(len(ds.site_id))
    ds["lat"] = ("site_id", lat)
    ds["lon"] = ("site_id", lon)
    return ds


def build_base_store(path, scenarios, workflows, n_per_wf):
    n_samples = n_per_wf * len(workflows)
    local, glob = load_local_and_global(scenarios, workflows, n_per_wf)
    vlm = load_vlm(n_samples)
    ds = combine(local, glob, vlm, n_samples)
    clean_for_zarr(ds).to_zarr(str(path), mode="w", zarr_format=2)
    print(f"  saved {path} ({len(ds.site_id)} sites)")


def build_globalclimate_variant(base_path, out_path):
    """Derive the globalclimate store from the saved base store."""
    base = xr.open_zarr(str(base_path))
    ds = base.copy()
    lsl = (
        (base.gsl_msl05 + base.lsl_ncc_msl05)
        .transpose(*base.lsl_msl05.dims)
        .astype(base.lsl_msl05.dtype)
    )
    lsl.attrs = dict(base.lsl_msl05.attrs)
    ds["lsl_msl05"] = lsl
    ds = ds.chunk(STORE_CHUNKS)
    clean_for_zarr(ds).to_zarr(str(out_path), mode="w", zarr_format=2)
    print(f"  saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test", action="store_true",
        help="1 scenario, 1 workflow, 5 samples, -test store paths",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="rebuild stores that already exist"
    )
    args = parser.parse_args()

    scenarios, workflows, n_per_wf = TLIM_SCENARIOS, WORKFLOWS, N_SAMPLES_PER_WORKFLOW
    base_path, variant_path = PATH_SLR, PATH_SLR_GLOBALCLIMATE
    n_workers = SLR_N_WORKERS
    if args.test:
        scenarios, workflows = TEST_TLIM_SCENARIOS, TEST_WORKFLOWS
        n_per_wf = TEST_N_SAMPLES // len(TEST_WORKFLOWS)
        n_workers = TEST_N_WORKERS
        base_path, variant_path = test_path(base_path), test_path(variant_path)
        print("=== TEST MODE ===")

    build_base = args.overwrite or not zarr_exists(base_path)
    build_variant = build_base or args.overwrite or not zarr_exists(variant_path)
    if not build_variant:
        print("both stores exist, nothing to do (--overwrite to rebuild)")
        return

    cluster = Cluster(n_workers, min_workers=1, ship_pyciam=False)
    cluster.start()
    timings = {}

    if build_base:
        print("--- base store ---")
        t0 = time.time()
        build_base_store(base_path, scenarios, workflows, n_per_wf)
        timings["base_store"] = time.time() - t0
    else:
        print(f"base store exists: {base_path}")

    print("--- globalclimate variant ---")
    t0 = time.time()
    build_globalclimate_variant(base_path, variant_path)
    timings["globalclimate_variant"] = time.time() - t0

    write_report(
        "01_process_slr" + ("_test" if args.test else ""),
        timings,
        base_store=str(base_path),
        globalclimate_store=str(variant_path),
    )
    cluster.close()
    print("done.")


if __name__ == "__main__":
    main()
