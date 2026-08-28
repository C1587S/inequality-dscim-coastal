"""
Stage 5: pick the optimal adaptation case per segment, per --scenario.
About an hour at 600 workers.

Compares the NPV of the cases stage 4 wrote and fills the optimalfixed slot,
in sample groups of 250. Tasks are plain (seg_ir, sample group) pairs:
optimize_case ignores its wait-futures arguments (pyCIAM/run.py:1589) and
finds sibling segments itself, so the legacy group bookkeeping is gone.
A rerun skips completed segments.

Run on the hub:
  test:  python -u 05_optimize.py --scenario glocal --test
  full:  nohup python -u 05_optimize.py --scenario glocal > 05_optimize_glocal.log 2>&1 &
"""

import argparse
import time
from itertools import product

import numpy as np

from runner import Cluster, install_vendored_pyciam, run_batched, write_report

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
    n_ok, n_err = run_batched(
        cluster, make_futures, tasks, BATCH_SIZE, f"optimize[{args.scenario}]"
    )
    timings = {"optimization": time.time() - t0}

    write_report(
        f"05_optimize_{args.scenario}" + ("_test" if args.test else ""),
        timings,
        n_ok=n_ok,
        n_err=n_err,
        output_store=str(tmp_path),
    )
    cluster.close()
    if n_err:
        raise SystemExit(
            f"{n_err} optimize tasks failed; rerun this stage to gap-fill "
            "(check=True skips completed segments) before aggregating"
        )
    print("done.")


if __name__ == "__main__":
    main()
