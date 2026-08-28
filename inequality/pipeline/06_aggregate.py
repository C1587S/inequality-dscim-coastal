"""
Stage 6: aggregate to the final outputs, per --scenario.

Filters the stage-4/5 store to the output cases, years and SSPs, sums seg_ir
to impact_region, renames the IAMs, then derives two views: gadmid (integer
region ids) and gadmid_coastal (subset to the original run's regions, which
drops ~1,500 inland-water ones). The impact_region store is the canonical
output.

Before any of that it counts nulls in the seg_ir store and refuses to run if
there are any, because the skipna aggregation would turn them into zeros.

Existing targets are skipped, so published stores are never clobbered.
--overwrite rebuilds them deliberately.

Run on the hub:
  test:  python -u 06_aggregate.py --scenario glocal --test
  full:  python -u 06_aggregate.py --scenario glocal
"""

import argparse
import time

import numpy as np
import xarray as xr

from runner import Cluster, clean_for_zarr, write_report, zarr_exists

from config import (
    AGG_N_WORKERS,
    AGG_VAR,
    IAM_RENAME,
    OUTPUT_CASES,
    OUTPUT_SSPS,
    OUTPUT_YEARS,
    PATH_FINAL,
    PATH_FINAL_COASTAL,
    PATH_FINAL_GADMID,
    PATH_INTERMEDIATE,
    PATH_OLD_OUTPUT,
    PATH_TMP,
    SCENARIOS,
    SEG_VAR,
    TEST_N_WORKERS,
    test_path,
)
from inputs import load_ciam_in

AGG_CHUNKSIZE = 2


def check_complete(tmp_path):
    """Refuse to aggregate an incomplete store.

    The groupby sum is skipna, so null seg cells become zeros at
    impact_region level. That is how the published v2 stores went wrong, and
    no post-aggregation check can see it, so this runs first.
    """
    t = xr.open_zarr(str(tmp_path)).costs.sel(
        case=OUTPUT_CASES, ssp=OUTPUT_SSPS, year=OUTPUT_YEARS
    )
    n_null = int(t.isnull().sum())
    if n_null:
        per_case = t.isnull().sum([d for d in t.dims if d != "case"]).compute()
        detail = ", ".join(
            f"{str(c)}: {int(n)}" for c, n in zip(per_case.case.values, per_case.values)
        )
        raise SystemExit(
            f"{n_null} null cells in {tmp_path} for the output cases/years "
            f"({detail}); rerun stages 4/5 to gap-fill before aggregating"
        )
    print("  completeness gate passed: no null cells in output cases/years")


def aggregate_to_impact_region(tmp_path, intermediate_path, final_path, ciam_in):
    t = xr.open_zarr(str(tmp_path))
    t = t.sel(case=OUTPUT_CASES, ssp=OUTPUT_SSPS, year=OUTPUT_YEARS)[["costs"]]
    clean_for_zarr(t).to_zarr(str(intermediate_path), mode="w", zarr_format=2)

    out = xr.open_zarr(
        str(intermediate_path), chunks={"case": -1, SEG_VAR: AGG_CHUNKSIZE}
    )
    out["costs"] = (
        out.costs.groupby(ciam_in[AGG_VAR]).sum().chunk({AGG_VAR: AGG_CHUNKSIZE}).persist()
    )
    out = out.drop_vars(SEG_VAR).unify_chunks()
    out["iam"] = [IAM_RENAME.get(str(i), str(i)) for i in out.iam.values]
    out = clean_for_zarr(out).persist()
    out.to_zarr(str(final_path), mode="w", zarr_format=2)
    print(f"  saved {final_path}")


def to_gadmid(final_path, gadmid_path):
    ds = xr.open_zarr(str(final_path))
    gadmids = np.array([int(str(ir).split("_")[1]) for ir in ds[AGG_VAR].values])
    ds = ds.rename({AGG_VAR: "gadmid"})
    ds["gadmid"] = gadmids
    clean_for_zarr(ds).to_zarr(str(gadmid_path), mode="w", zarr_format=2)
    print(f"  saved {gadmid_path}")


def coastal_subset(gadmid_path, coastal_path):
    ds = xr.open_zarr(str(gadmid_path))
    old = set(int(g) for g in xr.open_zarr(str(PATH_OLD_OUTPUT)).gadmid.values)
    keep = np.array([int(g) in old for g in ds.gadmid.values])
    ds = ds.isel(gadmid=keep)
    print(f"  dropped {int((~keep).sum())} inland-water gadmids, kept {int(keep.sum())}")
    clean_for_zarr(ds).to_zarr(str(coastal_path), mode="w", zarr_format=2)
    print(f"  saved {coastal_path}")


def verify(final_path):
    ds = xr.open_zarr(str(final_path))
    print(f"  dims: {dict(ds.sizes)}")
    print(f"  scenarios: {list(ds.scenario.values)}")
    print(f"  iams: {list(ds.iam.values)}")
    n_nonnull = int(ds.costs.notnull().sum())
    print(f"  non-null: {n_nonnull}/{ds.costs.size} ({100 * n_nonnull / ds.costs.size:.1f}%)")
    assert "ncc_ar6" in ds.scenario.values, "ncc_ar6 missing"
    # every cell, every case — the legacy assert (sum over costtype, skipna,
    # then notnull) passes over all-NaN cells and proved nothing
    assert n_nonnull == ds.costs.size, "null cells in final store"
    print("  checks passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--overwrite", action="store_true", help="rebuild outputs that already exist"
    )
    args = parser.parse_args()

    s = args.scenario
    n_workers = AGG_N_WORKERS
    paths = {
        "tmp": PATH_TMP[s],
        "intermediate": PATH_INTERMEDIATE[s],
        "final": PATH_FINAL[s],
        "gadmid": PATH_FINAL_GADMID[s],
        "coastal": PATH_FINAL_COASTAL[s],
    }
    if args.test:
        n_workers = TEST_N_WORKERS
        paths = {k: test_path(p) for k, p in paths.items()}
        print("=== TEST MODE ===")

    todo = {
        k: args.overwrite or not zarr_exists(paths[k])
        for k in ("final", "gadmid", "coastal")
    }
    if not any(todo.values()):
        print("all outputs exist, nothing to do (--overwrite to rebuild)")
        return

    ciam_in = load_ciam_in(s, test=args.test)
    cluster = Cluster(n_workers, ship_pyciam=False)
    cluster.start()
    timings = {}

    if todo["final"]:
        print("--- completeness gate ---")
        check_complete(paths["tmp"])
        print("--- aggregate to impact_region ---")
        t0 = time.time()
        aggregate_to_impact_region(
            paths["tmp"], paths["intermediate"], paths["final"], ciam_in
        )
        timings["aggregate"] = time.time() - t0
    else:
        print(f"final store exists: {paths['final']}")

    if todo["gadmid"]:
        print("--- gadmid view ---")
        t0 = time.time()
        to_gadmid(paths["final"], paths["gadmid"])
        timings["gadmid"] = time.time() - t0

    if todo["coastal"]:
        print("--- coastal subset ---")
        t0 = time.time()
        coastal_subset(paths["gadmid"], paths["coastal"])
        timings["coastal_subset"] = time.time() - t0

    print("--- verify ---")
    verify(paths["final"])

    write_report(
        f"06_aggregate_{s}" + ("_test" if args.test else ""),
        timings,
        final=str(paths["final"]),
        gadmid=str(paths["gadmid"]),
        coastal=str(paths["coastal"]),
    )
    cluster.close()
    print("done.")


if __name__ == "__main__":
    main()
