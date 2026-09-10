"""
Stage 5: pick the optimal adaptation case per segment, per --scenario.

Compares the NPV of the cases stage 4 wrote and fills the optimalfixed slot,
in sample groups of 250. Tasks are plain (seg_ir, sample group) pairs:
optimize_case ignores its wait-futures arguments (pyCIAM/run.py:1589) and
finds sibling segments itself.

Refuses to start unless stage 4 is complete. The case selection sums npv
across a segment's sibling regions with skipna (pyCIAM/run.py:842), so a
missing region would silently skew the choice for the whole segment.

The optimalfixed writes race exactly like stage 4's: several sample-group
tasks share each store chunk, and concurrent read-modify-writes revert each
other's slices to NaN (the global run lost 58% of optimalfixed this way).
So tasks run in waves, never two on the same segment at once, and the script
cycles run -> probe until every optimalfixed cell is written, exiting
nonzero only if a round makes no progress. The probe reads one npv point
and one costs point per (seg_ir, sample) at case=optimalfixed; the store's
optimal_case variable can't be probed, being uint8 whose unwritten cells
read as 0, a valid case index. For a cluster that keeps dropping, wrap it:
  until python -u 05_optimize.py --scenario global; do sleep 120; done

On RCC this runs as a SLURM array instead: --shard K/N elements each own
whole segments, and refuse any segment whose stage-4 npv has nulls;
--probe-only verifies and stages a gap-fill list. See rcc/README.md.

Run on the hub:
  test:  python -u 05_optimize.py --scenario glocal --test
  full:  nohup python -u 05_optimize.py --scenario glocal > 05_glocal.log 2>&1 &
"""

import argparse
import time

import numpy as np
import xarray as xr

from runner import (
    Cluster,
    install_vendored_pyciam,
    load_unfinished,
    run_batched,
    save_unfinished,
    unfinished_tasks_file,
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


def run_shard(k, n, segs, samp_groups, tmp_path, unfinished_file, task_kwargs):
    """One SLURM array element: segments k::n with all their sample groups,
    run serially, so elements never share a chunk. Before optimizing a
    segment, its own stage-4 npv is checked for nulls: the case selection
    sums sibling npv with skipna (pyCIAM/run.py:842), so optimizing over
    holes would silently skew the whole segment. A prior --probe-only run
    leaves an unfinished list; when present, only those tasks run."""
    ds = xr.open_zarr(str(tmp_path))
    todo = load_unfinished(unfinished_file)
    if todo is not None:
        mine = [(i, j) for i, j in todo if i % n == k]
        print(f"shard {k}/{n}: {len(mine)} unfinished tasks from the probe")
    else:
        mine = [(i, j) for i in range(k, len(segs), n) for j in range(len(samp_groups))]
        print(f"shard {k}/{n}: {len(mine)} tasks, full sweep with store check")
    n_err = 0
    t0 = time.time()
    for count, (i, j) in enumerate(mine):
        seg = segs[i]
        if count == 0 or mine[count - 1][0] != i:
            n_null = int(
                ds.npv.sel({SEG_VAR: seg})
                .drop_sel(case="optimalfixed")
                .isnull()
                .sum()
            )
            if n_null:
                n_err += 1
                print(f"{seg}: {n_null} null calc cells, refusing to optimize")
                continue
        try:
            run_optimize_case(
                seg, quantiles=samp_groups[j], check=todo is None, **task_kwargs
            )
        except Exception as e:
            n_err += 1
            print(f"task ({seg}, group {j}) failed: {str(e)[:200]}")
        if (count + 1) % 40 == 0 or count + 1 == len(mine):
            print(f"{count + 1}/{len(mine)} | {(time.time() - t0) / 60:.1f} min", flush=True)
    if n_err:
        raise SystemExit(f"shard {k}/{n}: {n_err} failures; fix stage 4 holes "
                         "if reported, then resubmit this element")
    print(f"shard {k}/{n} done in {(time.time() - t0) / 60:.1f} min")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--shard", metavar="K/N",
        help="SLURM array mode: run segments K::N serially, no cluster",
    )
    parser.add_argument(
        "--probe-only", action="store_true",
        help="count unfinished groups, save them for shard resume, and exit "
        "nonzero if any",
    )
    args = parser.parse_args()

    n_samples, n_workers = N_SAMPLES_TOTAL, N_WORKERS
    tmp_path = PATH_TMP[args.scenario]
    if args.test:
        n_samples, n_workers = TEST_N_SAMPLES, TEST_N_WORKERS
        tmp_path = test_path(tmp_path)
        print("=== TEST MODE ===")

    samples = np.arange(1, n_samples + 1)
    ciam_in = load_ciam_in(args.scenario, test=args.test)
    segs = ciam_in[SEG_VAR].values
    samp_groups = chunked(samples, OPT_SAMPLE_CHUNKSIZE)
    samp_off = np.cumsum([0] + [len(g) for g in samp_groups])
    n_total = len(segs) * len(samp_groups)
    stage = f"05_optimize_{args.scenario}" + ("_test" if args.test else "")
    unfinished_file = unfinished_tasks_file(stage)
    task_kwargs = dict(
        econ_input_path=str(SLIIDERS_IR[args.scenario]),
        output_path=str(tmp_path),
        seg_var=SEG_VAR,
        eps=1,
    )

    def probe_indices():
        """(seg index, group index) pairs whose optimalfixed npv or costs
        probe point is null; both probed since either can be missing."""
        ds = xr.open_zarr(str(tmp_path))
        npv = ds.npv.sel(case="optimalfixed", drop=True).isel(
            scenario=0, ssp=0, iam=0, drop=True
        )
        costs = ds.costs.sel(case="optimalfixed", drop=True).isel(
            costtype=0, scenario=0, year=-1, ssp=0, iam=0, drop=True
        )
        isnull = (
            (npv.isnull() | costs.isnull())
            .transpose(SEG_VAR, "sample")
            .compute()
            .values
        )
        return [
            (i, j)
            for i in range(len(segs))
            for j in range(len(samp_groups))
            if isnull[i, samp_off[j] : samp_off[j + 1]].any()
        ]

    if args.probe_only:
        print("probing optimalfixed for unfinished groups")
        hit = probe_indices()
        print(f"{len(hit)} of {n_total} groups unfinished")
        save_unfinished(unfinished_file, hit)
        raise SystemExit(1 if hit else 0)

    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        run_shard(k, n, segs, samp_groups, tmp_path, unfinished_file, task_kwargs)
        return

    cluster = Cluster(n_workers)
    cluster.start()

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

    def unfinished(as_tasks=True):
        """One null probe point of optimalfixed npv or costs per
        (seg_ir, sample) flags the owning task; a task can lose the chunk
        race in one variable but not the other, so probe both."""
        ds = xr.open_zarr(str(tmp_path))
        npv = ds.npv.sel(case="optimalfixed", drop=True).isel(
            scenario=0, ssp=0, iam=0, drop=True
        )
        costs = ds.costs.sel(case="optimalfixed", drop=True).isel(
            costtype=0, scenario=0, year=-1, ssp=0, iam=0, drop=True
        )
        isnull = (
            (npv.isnull() | costs.isnull())
            .transpose(SEG_VAR, "sample")
            .compute()
            .values
        )
        if not as_tasks:
            return int(isnull.sum())
        return [
            (segs[i], qg)
            for i in range(len(segs))
            for j, qg in enumerate(samp_groups)
            if isnull[i, samp_off[j] : samp_off[j + 1]].any()
        ]

    def build_waves(task_list):
        by_seg = {}
        for seg, qg in task_list:
            by_seg.setdefault(seg, []).append((seg, qg))
        n_waves = max(len(v) for v in by_seg.values())
        return [[v[k] for v in by_seg.values() if len(v) > k] for k in range(n_waves)]

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
                # the probe decides what runs; the store-side check reads the
                # whole segment across all samples, the wrong granularity
                check=False,
            )
            for seg_ir, sample_grp in batch
        ]

    t0 = time.time()
    n_ok = n_err = 0
    print("probing optimalfixed (reads every optimalfixed chunk; slow on a small cluster)")
    tasks = unfinished()
    print(f"{len(tasks)} of {n_total} groups unfinished")

    while tasks:
        waves = build_waves(tasks)
        for w, wave in enumerate(waves):
            if len(waves) > 1:
                print(f"wave {w + 1}/{len(waves)}: {len(wave)} tasks")
            ok, err, _ = run_batched(
                cluster, make_futures, wave, BATCH_SIZE, f"optimize[{args.scenario}]"
            )
            n_ok += ok
            n_err += err
        print("probing optimalfixed")
        n_before = len(tasks)
        tasks = unfinished()
        print(f"{len(tasks)} groups still unfinished")
        if tasks and len(tasks) >= n_before:
            print("no progress this round; stopping")
            break
    timings = {"optimization": time.time() - t0}

    write_report(
        stage,
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
