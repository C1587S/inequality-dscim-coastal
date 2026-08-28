"""
Stage 2: prepare the SLIIDERS economic inputs.

Writes three stores, each skipped if it already exists:

- local-rho IR store: the base SLIIDERS with variable names normalised
  (K_2014 -> K_2019, pop_2014 -> pop_2019, the labels pyCIAM's collapse
  expects). Feeds the fulladapt run.
- global-rho IR store: the same, with rho replaced by its population-weighted
  global average over 2000-2014, expanded to all countries and years. Feeds
  the glocal and global runs.
- seg-level store: the IR store collapsed to segments, used only by refA.
  Built once and shared by all scenarios: the collapse recomputes rho from
  ypcc (pyCIAM/utils.py:268), discarding the rho variable of whichever IR
  store it is fed, so both variants collapse to the same thing. This
  reproduces the published runs, where refA always reflects local income.

Everything loads into hub memory; no cluster needed. Runs in minutes.

Run on the hub:
  python -u 02_prepare_inputs.py
"""

import argparse
import time

import xarray as xr

from runner import clean_for_zarr, install_vendored_pyciam, write_report, zarr_exists

install_vendored_pyciam()

from pyCIAM.utils import collapse_econ_inputs_to_seg  # noqa: E402

from config import (  # noqa: E402
    PATH_REFA,
    PATH_SLIIDERS,
    PATH_SLIIDERS_SEG,
    SEG_VAR,
    SLIIDERS_IR,
)


def load_base_sliiders():
    ds = xr.open_zarr(str(PATH_SLIIDERS), chunks=None)
    renames = {
        old: new
        for old, new in {"K_2014": "K_2019", "pop_2014": "pop_2019"}.items()
        if old in ds.data_vars
    }
    return ds.rename(renames)


def global_average_rho(ds):
    """Population-weighted global mean of rho over 2000-2014, matching the
    published glocal (formerly "globaladapt") run."""
    pop = ds.pop_2019.sum("elev").groupby(ds.seg_country).sum().load()
    rho = ds.rho.sel(year=slice(2000, 2014)).mean("year").load()
    countries = sorted(set(rho.country.values) & set(pop.seg_country.values))
    pop = pop.sel(seg_country=countries).rename({"seg_country": "country"})
    rho = rho.sel(country=countries)
    return rho.weighted(pop).mean("country")


def build_ir_store(path, global_rho):
    ds = load_base_sliiders()
    if global_rho:
        avg = global_average_rho(ds)
        print(f"  global rho (SSP2, IIASA): {float(avg.isel(ssp=0, iam=0)):.4f}")
        print(
            f"  replacing rho range {float(ds.rho.min()):.3f}-{float(ds.rho.max()):.3f}"
        )
        ds["rho"] = avg.expand_dims(country=ds.country, year=ds.year)
    clean_for_zarr(ds).to_zarr(str(path), mode="w", zarr_format=2)
    print(f"  saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite", action="store_true", help="rebuild stores that already exist"
    )
    args = parser.parse_args()

    timings = {}
    stores = [
        ("local_rho_ir", SLIIDERS_IR["fulladapt"], dict(global_rho=False)),
        ("global_rho_ir", SLIIDERS_IR["glocal"], dict(global_rho=True)),
    ]
    for name, path, kwargs in stores:
        print(f"--- {name} ---")
        if not args.overwrite and zarr_exists(path):
            print(f"  exists: {path}")
            continue
        t0 = time.time()
        build_ir_store(path, **kwargs)
        timings[name] = time.time() - t0

    print("--- seg store (shared, feeds refA) ---")
    if not args.overwrite and zarr_exists(PATH_SLIIDERS_SEG):
        print(f"  exists: {PATH_SLIIDERS_SEG}")
    else:
        t0 = time.time()
        collapse_econ_inputs_to_seg(
            str(SLIIDERS_IR["fulladapt"]),
            str(PATH_SLIIDERS_SEG),
            seg_var_subset=None,
            output_chunksize=100,
            seg_var=SEG_VAR,
        )
        timings["collapse_to_seg"] = time.time() - t0
        print(f"  saved {PATH_SLIIDERS_SEG}")

    write_report(
        "02_prepare_inputs",
        timings,
        local_rho_ir=str(SLIIDERS_IR["fulladapt"]),
        global_rho_ir=str(SLIIDERS_IR["glocal"]),
        seg_store=str(PATH_SLIIDERS_SEG),
        refa_store_consumer=str(PATH_REFA),
    )
    print("done.")


if __name__ == "__main__":
    main()
