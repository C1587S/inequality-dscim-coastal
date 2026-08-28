"""
Stage 3: reference adaptation heights (refA).

refA is the present-day adaptation height per segment, optimised under no
climate change. It comes from the seg store, which is the same for every
scenario (see config.py), so this runs once and has no --scenario flag.

A rerun recomputes only the groups with null cells, so filling holes after a
crash takes minutes. The completion gate is the store's exact null count;
task errors that a retry already fixed don't fire it.

Run on the hub:
  test:  python -u 03_refa.py --test    (needs the 01 --test store)
  full:  nohup python -u 03_refa.py > 03_refa.log 2>&1 &
"""

import argparse
import time
from itertools import product

import numpy as np
import pandas as pd
import xarray as xr

from runner import (
    Cluster,
    install_vendored_pyciam,
    run_batched,
    write_report,
    zarr_exists,
)

install_vendored_pyciam()

from config import (  # noqa: E402
    MC_DIM,
    N_SAMPLES_TOTAL,
    N_WORKERS,
    PATH_PARAMS,
    PATH_REFA,
    PATH_SLIIDERS_SEG,
    PATH_SLR,
    PATHS_SURGE_LOOKUP,
    REFA_BATCH_SIZE,
    REFA_SEG_CHUNKSIZE,
    SAMPLE_CHUNKSIZE,
    TEST_N_SAMPLES,
    TEST_N_WORKERS,
    test_path,
)
from inputs import chunked, load_ciam_in  # noqa: E402


def run_get_refA(grp, **kwargs):
    from pyCIAM.run import get_refA

    return get_refA(grp, **kwargs)


def write_template(path, segs, samples):
    template = (
        xr.DataArray(
            coords=dict(
                sample=samples,
                seg=segs,
                case=(
                    ["sample", "seg"],
                    np.ones((len(samples), len(segs)), dtype=np.int8),
                ),
            ),
            dims=("sample", "seg"),
        )
        .to_dataset(name="refA")
        .chunk({"seg": REFA_SEG_CHUNKSIZE, "sample": SAMPLE_CHUNKSIZE})
    )
    template.to_zarr(str(path), mode="w", zarr_format=2)
    print(f"template: sample={len(samples)}, seg={len(segs)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--overwrite", action="store_true", help="recreate the refA store from scratch"
    )
    args = parser.parse_args()

    n_samples, n_workers = N_SAMPLES_TOTAL, N_WORKERS
    slr_path, refa_path = PATH_SLR, PATH_REFA
    if args.test:
        n_samples, n_workers = TEST_N_SAMPLES, TEST_N_WORKERS
        slr_path, refa_path = test_path(PATH_SLR), test_path(PATH_REFA)
        print("=== TEST MODE ===")

    samples = np.arange(1, n_samples + 1)
    params = pd.read_json(PATH_PARAMS)["values"]

    # the fulladapt IR store is only used for the segment list, which is the
    # same in every variant
    ciam_in = load_ciam_in("fulladapt", test=args.test)
    segs = np.unique(ciam_in.seg)

    resuming = not args.overwrite and zarr_exists(refa_path)
    if not resuming:
        write_template(refa_path, segs, samples)

    groups = list(product(chunked(segs, REFA_SEG_CHUNKSIZE), chunked(samples, SAMPLE_CHUNKSIZE)))
    if resuming:
        # fill only the holes: get_refA has no completed-work check, so
        # restrict the task list to groups whose cells contain nulls
        isnull = xr.open_zarr(str(refa_path)).refA.isnull().compute()
        groups = [
            g for g in groups if bool(isnull.sel(seg=g[0], sample=g[1]).any())
        ]
        print(f"resuming: {len(groups)} groups with null cells to fill")
    print(f"refA groups: {len(groups)}")

    cluster = Cluster(n_workers)
    cluster.start()

    def make_futures(client, batch):
        return client.map(
            run_get_refA,
            batch,
            output_path=str(refa_path),
            econ_input_path=str(PATH_SLIIDERS_SEG),
            slr_input_path=str(slr_path),
            params=params,
            surge_input_path=str(PATHS_SURGE_LOOKUP["seg"]),
            mc_dim=MC_DIM,
            diaz_inputs=False,
            eps=1,
        )

    t0 = time.time()
    n_ok, n_err = run_batched(cluster, make_futures, groups, REFA_BATCH_SIZE, "refA")
    timings = {"refA": time.time() - t0}

    # gate on the store, not on task attempts: n_err counts failed attempts,
    # some of which are superseded by batch retries; the exact null count is
    # the ground truth
    refa = xr.open_zarr(str(refa_path))
    n_null = int(refa.refA.isnull().sum())
    print(f"refA null cells: {n_null} of {refa.refA.size} ({n_err} task errors)")
    if n_err and not n_null:
        print(f"note: all {n_err} task errors were superseded by retries")

    write_report(
        "03_refa" + ("_test" if args.test else ""),
        timings,
        n_ok=n_ok,
        n_err=n_err,
        n_null=n_null,
        refa_store=str(refa_path),
    )
    cluster.close()
    # NaN refA holes propagate as NaN noAdaptation costs that the aggregation
    # silently zeros — the confirmed flaw in the published v2 stores. Never
    # hand an incomplete refA to stage 4.
    if n_null:
        raise SystemExit(
            f"refA incomplete: {n_null} null cells; rerun this stage "
            "(only unfilled groups are recomputed) before running stage 4"
        )
    print("done.")


if __name__ == "__main__":
    main()
