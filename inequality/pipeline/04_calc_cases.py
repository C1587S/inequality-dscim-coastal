"""
Stage 4: calc_all_cases — the long stage (~19-24h at 600 workers).

Computes costs for every adaptation case, per --scenario. The output template
covers six SLR scenarios: the five tlim scenarios from the SLR store plus
ncc_ar6, which pyCIAM generates internally from the no-climate-change series.

Tasks run with check=True, so completed (segment, sample) groups are skipped:
rerunning this script after a crash gap-fills the store. Pass --overwrite to
discard an existing store and start fresh.

Run on the hub:
  test:  python -u 04_calc_cases.py --scenario glocal --test
  full:  nohup python -u 04_calc_cases.py --scenario glocal > 04_calc_glocal.log 2>&1 &
"""

import argparse
import time
from collections import OrderedDict
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

from pyCIAM.constants import CASES, COSTTYPES  # noqa: E402
from pyCIAM.io import create_template_dataarray  # noqa: E402
from pyCIAM.utils import add_attrs_to_result  # noqa: E402

from config import (  # noqa: E402
    BATCH_SIZE,
    MC_DIM,
    N_SAMPLES_TOTAL,
    N_WORKERS,
    PATH_PARAMS,
    PATH_REFA,
    PATH_TMP,
    PATHS_SURGE_LOOKUP,
    SAMPLE_CHUNKSIZE,
    SCENARIOS,
    SEG_CHUNKSIZE,
    SEG_VAR,
    SLIIDERS_IR,
    SLR,
    TEST_N_SAMPLES,
    TEST_N_WORKERS,
    test_path,
)
from inputs import chunked, load_ciam_in  # noqa: E402


def run_calc_all_cases(grp, **kwargs):
    from pyCIAM.run import calc_all_cases

    return calc_all_cases(grp, **kwargs)


def write_template(path, ciam_in, slr_path, samples, params):
    slr = xr.open_zarr(str(slr_path), chunks=None)
    scenarios = ["ncc_ar6"] + list(slr.scenario.values)
    print(f"template scenarios: {scenarios}")

    coords = OrderedDict(
        {
            "case": CASES,
            "costtype": COSTTYPES,
            SEG_VAR: ciam_in[SEG_VAR].values,
            "scenario": scenarios,
            "sample": samples,
            "year": np.arange(params.model_start, ciam_in.year.max().item() + 1),
            **{d: ciam_in[d].values for d in ["ssp", "iam"] if d in ciam_in.dims},
        }
    )
    chunk_spec = {SEG_VAR: 1, "case": len(CASES) - 1}
    chunks = {k: chunk_spec.get(k, len(v)) for k, v in coords.items()}
    out = create_template_dataarray(coords.keys(), coords, chunks).to_dataset(
        name="costs"
    )
    out["npv"] = out.costs.isel(year=0, costtype=0, drop=True).astype("float64")
    out["optimal_case"] = out.npv.isel(case=0, drop=True).astype("uint8")
    out = add_attrs_to_result(out)
    out.to_zarr(str(path), compute=False, mode="w", zarr_format=2)
    print(f"template: {dict(out.sizes)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--overwrite", action="store_true", help="discard the output store, start fresh"
    )
    args = parser.parse_args()

    n_samples, n_workers = N_SAMPLES_TOTAL, N_WORKERS
    slr_path, refa_path, tmp_path = SLR[args.scenario], PATH_REFA, PATH_TMP[args.scenario]
    if args.test:
        n_samples, n_workers = TEST_N_SAMPLES, TEST_N_WORKERS
        slr_path, refa_path = test_path(slr_path), test_path(refa_path)
        tmp_path = test_path(tmp_path)
        print("=== TEST MODE ===")

    samples = np.arange(1, n_samples + 1)
    params = pd.read_json(PATH_PARAMS)["values"]
    ciam_in = load_ciam_in(args.scenario, test=args.test)
    print(f"scenario: {args.scenario}, seg_ir: {len(ciam_in[SEG_VAR])}")

    if args.overwrite or not zarr_exists(tmp_path):
        write_template(tmp_path, ciam_in, slr_path, samples, params)
    else:
        print(f"resuming into existing store {tmp_path} (check=True skips done work)")

    tasks = list(
        product(
            chunked(ciam_in[SEG_VAR].values, SEG_CHUNKSIZE),
            chunked(samples, SAMPLE_CHUNKSIZE),
        )
    )
    print(f"tasks: {len(tasks)}")

    cluster = Cluster(n_workers)
    cluster.start()

    def make_futures(client, batch):
        return client.map(
            run_calc_all_cases,
            batch,
            params=params,
            econ_input_path=str(SLIIDERS_IR[args.scenario]),
            slr_input_paths=[str(slr_path)],
            slr_names=["ar6"],
            output_path=str(tmp_path),
            refA_path=str(refa_path),
            surge_input_path=str(PATHS_SURGE_LOOKUP[SEG_VAR]),
            seg_var=SEG_VAR,
            mc_dim=MC_DIM,
            diaz_inputs=False,
            check=True,
        )

    t0 = time.time()
    n_ok, n_err = run_batched(
        cluster, make_futures, tasks, BATCH_SIZE, f"calc[{args.scenario}]"
    )
    timings = {"calc_all_cases": time.time() - t0}

    write_report(
        f"04_calc_{args.scenario}" + ("_test" if args.test else ""),
        timings,
        n_ok=n_ok,
        n_err=n_err,
        output_store=str(tmp_path),
    )
    cluster.close()
    if n_err:
        raise SystemExit(
            f"{n_err} calc tasks failed; rerun this stage to gap-fill "
            "(check=True skips completed work) before running stage 5"
        )
    print("done.")


if __name__ == "__main__":
    main()
