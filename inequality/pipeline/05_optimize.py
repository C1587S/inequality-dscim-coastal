"""
Stage 5: pick the optimal adaptation case per segment, per --scenario.
About an hour at 600 workers.

Compares the NPV of the cases stage 4 wrote and fills the optimalfixed slot,
in sample groups of 250. Tasks are plain (seg_ir, sample group) pairs:
optimize_case ignores its wait-futures arguments (pyCIAM/run.py:1589) and
finds sibling segments itself, so the legacy group bookkeeping is gone.

Refuses to start unless stage 4 is complete. The case selection sums npv
across a segment's sibling regions with skipna (pyCIAM/run.py:842), so a
missing region would silently skew the choice for the whole segment.

After a run with failures, the failed tasks are saved next to this script
and a rerun retries only those. The store itself can't say which optimize
tasks ran: optimal_case is uint8, and its unwritten cells read as 0, which
is a valid case index.

Run on the hub:
  test:  python -u 05_optimize.py --scenario glocal --test
  full:  nohup python -u 05_optimize.py --scenario glocal > 05_optimize_glocal.log 2>&1 &
"""

import argparse
import time
from itertools import product

import numpy as np
import xarray as xr

from runner import (
    Cluster,
    clear_failed_tasks,
    install_vendored_pyciam,
    load_failed_tasks,
    run_batched,
    save_failed_tasks,
    write_report,
)

install_vendored_pyciam()

from config import (  # noqa: E402
    BATCH_SIZE,
    N_SAMPLES_TOTAL,
    N_WORKERS,
    OPT_SAMPLE_CHUNKSIZE,
    PATH_TMP,
    SCENARIOS,
    SEG_VAR,
    SLIIDERS_IR,
    TEST_N_SAMPLES,
    TEST_N_WORKERS,
    test_path,
)
from inputs import chunked, load_ciam_in  # noqa: E402


def run_optimize_case(seg_ir, **kwargs):
    from pyCIAM.run import optimize_case

    return optimize_case(seg_ir, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    n_samples, n_workers = N_SAMPLES_TOTAL, N_WORKERS
    tmp_path = PATH_TMP[args.scenario]
    if args.test:
        n_samples, n_workers = TEST_N_SAMPLES, TEST_N_WORKERS
        tmp_path = test_path(tmp_path)
        print("=== TEST MODE ===")

    samples = np.arange(1, n_samples + 1)
    ciam_in = load_ciam_in(args.scenario, test=args.test)
    stage = f"05_optimize_{args.scenario}" + ("_test" if args.test else "")

    print("checking the calc store is complete")
    n_null = int(
        xr.open_zarr(str(tmp_path))
        .npv.isel(case=0, scenario=0, ssp=0, iam=0, drop=True)
        .isnull()
        .sum()
    )
    if n_null:
        raise SystemExit(
            f"{n_null} (seg_ir, sample) cells in {tmp_path} have no calc "
            "results; rerun stage 4 first. The case selection sums sibling "
            "npv with skipna, so holes would silently skew whole segments."
        )

    prev = load_failed_tasks(stage)
    if prev is not None:
        tasks = prev
        print(f"retrying {len(tasks)} tasks that failed last run")
    else:
        tasks = list(
            product(ciam_in[SEG_VAR].values, chunked(samples, OPT_SAMPLE_CHUNKSIZE))
        )
    print(f"scenario: {args.scenario}, tasks: {len(tasks)}")

    cluster = Cluster(n_workers)
    cluster.start()

    def make_futures(client, batch):
        return [
            client.submit(
                run_optimize_case,
                seg_ir,
                quantiles=sample_grp,
                econ_input_path=str(SLIIDERS_IR[args.scenario]),
                output_path=str(tmp_path),
                seg_var=SEG_VAR,
                eps=1,
                check=True,
            )
            for seg_ir, sample_grp in batch
        ]

    t0 = time.time()
    n_ok, n_err, failed = run_batched(
        cluster, make_futures, tasks, BATCH_SIZE, f"optimize[{args.scenario}]"
    )
    timings = {"optimization": time.time() - t0}

    write_report(stage, timings, n_ok=n_ok, n_err=n_err, output_store=str(tmp_path))
    cluster.close()
    if n_err:
        save_failed_tasks(stage, failed)
        raise SystemExit(
            f"{n_err} optimize tasks failed; rerun this stage before "
            "aggregating. The rerun retries only these tasks."
        )
    clear_failed_tasks(stage)
    print("done.")


if __name__ == "__main__":
    main()
