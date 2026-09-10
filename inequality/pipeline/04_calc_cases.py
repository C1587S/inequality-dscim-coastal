"""
Stage 4: costs for every adaptation case, per --scenario. The long stage,
roughly a day at 600 workers.

The output covers six SLR scenarios: the five tlims from the SLR store plus
ncc_ar6, which pyCIAM builds from the no-climate-change series.

Templates chunk the sample dimension at the task write size, so each task
owns its chunks. The global store predates this and has 1000-sample chunks
shared by ten tasks each; concurrent writers there raced and reverted each
other's slices to NaN, which is where most of its holes came from. Resume
handles both layouts: it probes npv and costs for null points, reruns only
the owning tasks, in waves that never write the same chunk concurrently.

The script cycles run -> probe until the store is complete, so scheduler
trouble costs a re-probe rather than a restart, and it only exits nonzero
if a whole round makes no progress. For a cluster that keeps dropping,
wrap it so process death restarts too:
  until python -u 04_calc_cases.py --scenario global; do sleep 120; done
--overwrite starts the store fresh.

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
    # sample chunks must match the task write size: tasks write 100-sample
    # slices, and a chunk shared by several tasks gets read-modify-written
    # concurrently, silently reverting the losers' slices to NaN
    chunk_spec = {SEG_VAR: 1, "case": len(CASES) - 1, "sample": SAMPLE_CHUNKSIZE}
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

    resuming = not args.overwrite and zarr_exists(tmp_path)
    if not resuming:
        write_template(tmp_path, ciam_in, slr_path, samples, params)

    seg_groups = chunked(ciam_in[SEG_VAR].values, SEG_CHUNKSIZE)
    samp_groups = chunked(samples, SAMPLE_CHUNKSIZE)
    tasks = list(product(seg_groups, samp_groups))
    n_total = len(tasks)

    cluster = Cluster(n_workers)
    cluster.start()

    def unfinished(as_tasks=True):
        """One null probe point of npv or costs per (seg_ir, sample) flags
        the owning task; tasks write both variables, and a task can lose a
        chunk race in one variable but not the other, so probe both."""
        ds = xr.open_zarr(str(tmp_path))
        npv = ds.npv.isel(case=0, scenario=0, ssp=0, iam=0, drop=True)
        costs = ds.costs.isel(
            case=0, costtype=0, scenario=0, year=-1, ssp=0, iam=0, drop=True
        )
        gb = (ds.npv.nbytes + ds.costs.nbytes) * 10 / 11 / 1e9
        print(
            f"probe: ~{gb:.0f} GB of solve chunks on the cluster "
            "(live view on the dashboard)",
            flush=True,
        )
        tp = time.time()
        isnull = (
            (npv.isnull() | costs.isnull())
            .transpose(SEG_VAR, "sample")
            .compute()
            .values
        )
        print(f"probe: done in {(time.time() - tp) / 60:.1f} min", flush=True)
        if not as_tasks:
            return int(isnull.sum())
        seg_off = np.cumsum([0] + [len(g) for g in seg_groups])
        samp_off = np.cumsum([0] + [len(g) for g in samp_groups])
        return [
            (sg, qg)
            for i, sg in enumerate(seg_groups)
            for j, qg in enumerate(samp_groups)
            if isnull[seg_off[i] : seg_off[i + 1], samp_off[j] : samp_off[j + 1]].any()
        ]

    def build_waves(task_list, probe_driven):
        """On the misaligned global store, tasks that share a seg group also
        share chunks; wave them so no two ever write the same chunk at once."""
        if not probe_driven:
            return [task_list]
        by_seg = {}
        for sg, qg in task_list:
            by_seg.setdefault(sg[0], []).append((sg, qg))
        n_waves = max(len(v) for v in by_seg.values())
        return [[v[k] for v in by_seg.values() if len(v) > k] for k in range(n_waves)]

    probe_driven = resuming

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
            # the probe already decided what to rerun; the store-side check
            # only reads costs, so it would wrongly skip npv-only holes
            check=not probe_driven,
        )

    t0 = time.time()
    n_ok = n_err = 0
    if resuming:
        print(f"resuming into {tmp_path}; probing for unfinished groups")
        tasks = unfinished()
        print(f"{len(tasks)} of {n_total} groups unfinished")

    # keep cycling run -> probe until the store is complete, so scheduler
    # trouble mid-run costs a re-probe, not a restart
    while tasks:
        waves = build_waves(tasks, probe_driven)
        for w, wave in enumerate(waves):
            if len(waves) > 1:
                print(f"wave {w + 1}/{len(waves)}: {len(wave)} tasks")
            ok, err, _ = run_batched(
                cluster, make_futures, wave, BATCH_SIZE, f"calc[{args.scenario}]"
            )
            n_ok += ok
            n_err += err
        print("probing the store")
        n_before = len(tasks)
        tasks = unfinished()
        print(f"{len(tasks)} groups still unfinished")
        if tasks and probe_driven and len(tasks) >= n_before:
            print("no progress this round; stopping")
            break
        probe_driven = True
    timings = {"calc_all_cases": time.time() - t0}

    write_report(
        f"04_calc_{args.scenario}" + ("_test" if args.test else ""),
        timings,
        n_ok=n_ok,
        n_err=n_err,
        n_unfinished_groups=len(tasks),
        output_store=str(tmp_path),
    )
    cluster.close()
    if tasks:
        raise SystemExit(
            f"{len(tasks)} groups still unfinished and the last round made "
            "no progress; check the log, then rerun this stage"
        )
    print("done.")


if __name__ == "__main__":
    main()
