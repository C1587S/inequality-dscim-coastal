"""
Stage 3: reference adaptation heights (refA), shared by all scenarios.

refA is the present-day adaptation height per segment, found by optimising
pyCIAM under the no-climate-change scenario (local VLM only). It is computed
from the seg-level store, which is identical for every rho treatment (see
config.py), so this stage runs once and has no --scenario flag; every
scenario's calc stage reads the same store.

Rerunning after a crash recomputes all groups (get_refA has no completed-work
check); at roughly 1.3h for the full run that is acceptable.

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

    if args.overwrite or not zarr_exists(refa_path):
        write_template(refa_path, segs, samples)
    else:
        print(f"resuming into existing store {refa_path} (all groups recomputed)")

    groups = list(product(chunked(segs, REFA_SEG_CHUNKSIZE), chunked(samples, SAMPLE_CHUNKSIZE)))
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

    refa = xr.open_zarr(str(refa_path))
    pct = float(refa.refA.notnull().mean()) * 100
    print(f"refA non-null: {pct:.1f}%")

    write_report(
        "03_refa" + ("_test" if args.test else ""),
        timings,
        n_ok=n_ok,
        n_err=n_err,
        pct_nonnull=pct,
        refa_store=str(refa_path),
    )
    cluster.close()
    # NaN refA holes propagate as NaN noAdaptation costs that the aggregation
    # silently zeros — the suspected flaw in the published v2 store. Never
    # hand an incomplete refA to stage 4.
    if n_err or pct < 100:
        raise SystemExit(
            f"refA incomplete ({n_err} failed groups, {pct:.1f}% non-null); "
            "rerun this stage until it is 100% before running stage 4"
        )
    print("done.")


if __name__ == "__main__":
    main()
