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

On RCC this runs as a SLURM array instead, with no cluster at all:
--template-only once, an array of --shard K/N elements each owning its
segments' chunks outright, then --probe-only to verify and to stage a
gap-fill list for resubmission. See rcc/README.md.

Run on the hub:
  test:  python -u 04_calc_cases.py --scenario glocal --test
  full:  nohup python -u 04_calc_cases.py --scenario glocal > 04_calc_glocal.log 2>&1 &
"""

import argparse
import resource
import traceback
import os
import time
import warnings
from collections import deque

# the pinned pyCIAM triggers xarray FutureWarnings we can't act on, and in
# a shard log they bury the progress lines
warnings.filterwarnings("ignore", category=FutureWarning)
from collections import OrderedDict
from itertools import product

import numpy as np
import pandas as pd
import xarray as xr

from runner import (
    Cluster,
    install_vendored_pyciam,
    load_unfinished,
    run_batched,
    save_unfinished,
    unfinished_tasks_file,
    write_report,
    zarr_exists,
)

install_vendored_pyciam()

from pyCIAM.constants import CASES, COSTTYPES  # noqa: E402
from pyCIAM.io import create_template_dataarray, save_to_zarr_region  # noqa: E402
from pyCIAM.utils import add_attrs_to_result  # noqa: E402

from config import (  # noqa: E402
    BATCH_SIZE,
    MC_DIM,
    N_SAMPLES_TOTAL,
    N_WORKERS,
    OUTPUT_YEARS,
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


def _tmp_base():
    """A writable node-local base for the mini stores, validated rather
    than trusted: inside the container SLURM_TMPDIR can be an env string
    naming a host path that was never bound, so each candidate must prove
    it exists and is writable. Fallback is a job-scoped dir under /tmp,
    which epilog cleaners for other jobs leave alone."""
    import tempfile

    candidates = []
    d = os.environ.get("SLURM_TMPDIR")
    if d:
        candidates.append(d)
    jid = os.environ.get("SLURM_JOB_ID")
    if jid:
        candidates.append(os.path.join(tempfile.gettempdir(), "ciam-job-" + jid))
    for d in candidates:
        try:
            os.makedirs(d, exist_ok=True)
            if os.access(d, os.W_OK):
                return d
        except OSError:
            continue
    return None


def run_calc_all_cases(grp, **kwargs):
    """Run one calc task via a throwaway local store, keeping the real
    store small.

    pyCIAM writes costs for every model year, but downstream reads only
    the output years: stage 5 copies the winning case (whatever years the
    store has) and stage 6 selects 2050 and 2090, so ~98% of a full-year
    store is write-only data - 3.4 TB per scenario, which does not fit
    scratch. Each task therefore gives pyCIAM a full-year mini store on
    node-local disk, then region-writes the output-year slice of costs
    plus npv (all solve cases; the optimizer's argmin needs them) into the
    real 2-year store. ~100 GB per scenario, and the bulk of the write
    traffic never touches GPFS."""
    import shutil
    import tempfile

    from pyCIAM.run import calc_all_cases

    mini_spec = kwargs.pop("_mini")
    check = kwargs.pop("check", False)
    seg_var = kwargs["seg_var"]
    real = kwargs["output_path"]
    segs, samps = grp[0], grp[1]

    if check and task_done(real, seg_var, segs, samps):
        return

    tmpdir = None
    mini = None
    phase = "tmp dir creation"
    try:
        tmpdir = tempfile.mkdtemp(prefix="ciam04-", dir=_tmp_base())
        mini = os.path.join(tmpdir, "mini.zarr")
        phase = "mini template write"
        write_mini_template(mini, seg_var, segs, samps, mini_spec)
        phase = "pyCIAM calc into the mini"
        calc_all_cases(grp, **{**kwargs, "output_path": mini, "check": False})
        phase = "mini readback"
        ds = xr.open_zarr(mini, chunks=None)
        solve = list(CASES[:-1])
        out = xr.Dataset(
            {
                "costs": ds.costs.sel(
                    year=list(mini_spec["output_years"]), case=solve
                ),
                "npv": ds.npv.sel(case=solve),
            }
        ).load()
        for v in out.variables:
            out[v].encoding.clear()
        phase = "landing write to the real store"
        save_to_zarr_region(out, str(real), storage_options=None)
    except Exception as e:
        # a failure must say what failed and in what state, not just its
        # consequence: which phase, whether the mini still exists (a
        # vanished mini points at a /tmp janitor, not at us), and how much
        # tmp space was free (ENOSPC points at the node, not the code)
        loc = tmpdir or _tmp_base() or "/tmp"
        try:
            free = shutil.disk_usage(loc).free / 1e9
        except OSError:
            free = float("nan")
        raise RuntimeError(
            f"{phase} failed; mini exists: "
            f"{mini is not None and os.path.exists(mini)}, "
            f"tmp free: {free:.1f} GB, dir: {tmpdir or loc}"
        ) from e
    finally:
        # verification hook: keep the full-year mini store (the "95-year
        # store" for this task) so verify_two_year.py can compare what
        # landed in the real store against it, bit for bit
        keep = os.environ.get("CIAM04_KEEP_MINI")
        if keep and mini and os.path.exists(mini):
            os.makedirs(keep, exist_ok=True)
            dest = os.path.join(keep, f"mini-{segs[0]}-s{samps[0]}.zarr")
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.move(mini, dest)
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def task_done(real, seg_var, segs, samps):
    """The task's own npv and costs probe points in the real store."""
    ds = xr.open_zarr(str(real), chunks=None)
    sel = {seg_var: list(segs), "sample": list(samps)}
    npv = ds.npv.isel(case=0, scenario=0, ssp=0, iam=0, drop=True).sel(sel)
    costs = ds.costs.isel(
        case=0, costtype=0, scenario=0, year=-1, ssp=0, iam=0, drop=True
    ).sel(sel)
    return not bool(npv.isnull().any() or costs.isnull().any())


def write_mini_template(path, seg_var, segs, samps, spec):
    """A per-task, full-year template for pyCIAM to write into."""
    coords = OrderedDict(
        {
            "case": CASES,
            "costtype": COSTTYPES,
            seg_var: np.asarray(segs),
            "scenario": spec["scenarios"],
            "sample": np.asarray(samps),
            "year": spec["model_years"],
            **spec["extra_coords"],
        }
    )
    chunk_spec = {seg_var: len(segs), "case": len(CASES) - 1}
    chunks = {k: chunk_spec.get(k, len(v)) for k, v in coords.items()}
    out = create_template_dataarray(coords.keys(), coords, chunks).to_dataset(
        name="costs"
    )
    out["npv"] = out.costs.isel(year=0, costtype=0, drop=True).astype("float64")
    out.to_zarr(path, compute=False, mode="w", zarr_format=2)


def run_shard(k, n, seg_groups, samp_groups, unfinished_file, task_kwargs):
    """One SLURM array element: seg groups k::n with all their sample groups,
    run serially. An element owns its seg groups' chunks outright, so there
    is nothing to race and no scheduler to lose. A prior --probe-only run
    leaves an unfinished list behind; when present, only those tasks run.
    Otherwise a full sweep with pyCIAM's store check, which makes
    resubmitting an element a cheap gap-fill."""
    todo = load_unfinished(unfinished_file, store=task_kwargs["output_path"])
    if todo is not None:
        pairs = [
            (seg_groups[i], samp_groups[j]) for i, j in todo if i % n == k
        ]
        check = False
        print(f"shard {k}/{n}: {len(pairs)} unfinished tasks from the probe")
    else:
        pairs = [(sg, qg) for sg in seg_groups[k::n] for qg in samp_groups]
        check = True
        print(f"shard {k}/{n}: {len(pairs)} tasks, full sweep with store check")
    print(f"mini store base: {_tmp_base() or 'system default tmp'}")
    n_err = 0
    t0 = time.time()
    # rate over a trailing window: a skip-assisted start (cheap re-checks of
    # already-written work) otherwise drags the projection far off the
    # fresh-compute rate
    recent = deque([t0], maxlen=21)
    n_tb = 0
    for i, (sg, qg) in enumerate(pairs):
        for attempt in (1, 2):
            try:
                run_calc_all_cases((sg, qg), check=check, **task_kwargs)
                break
            except Exception as e:
                # a vanished mini is environmental (tmp janitor), not
                # deterministic: one retry recovers it transparently
                if attempt == 1 and "mini exists: False" in str(e):
                    print(f"task ({sg[0]}, samples {qg[0]}..{qg[-1]}): mini "
                          "store vanished mid-task; retrying once")
                    continue
                n_err += 1
                cause = f" <- {str(e.__cause__)[:200]}" if e.__cause__ else ""
                print(f"task ({sg[0]}, samples {qg[0]}..{qg[-1]}) failed: "
                      f"{str(e)[:200]}{cause}")
                if n_tb < 2:
                    n_tb += 1
                    traceback.print_exc()
                break
        recent.append(time.time())
        done = i + 1
        elapsed = time.time() - t0
        rate = (recent[-1] - recent[0]) / (len(recent) - 1)
        ru = resource.getrusage(resource.RUSAGE_SELF)
        cpu = 100 * (ru.ru_utime + ru.ru_stime) / elapsed
        eta = rate * (len(pairs) - done)
        print(
            f"{done}/{len(pairs)} ({sg[0]}, samples {qg[0]}..{qg[-1]}) | "
            f"{elapsed / 60:.1f} min | {rate:.0f} s/task | cpu {cpu:.0f}% | "
            f"~{eta / 60:.0f} min left",
            flush=True,
        )
    if n_err:
        raise SystemExit(f"shard {k}/{n}: {n_err} tasks failed; resubmit this element")
    print(f"shard {k}/{n} done in {(time.time() - t0) / 60:.1f} min")


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
            # only the output years: tasks compute all model years in local
            # mini stores and land just this slice here (see
            # run_calc_all_cases), keeping the store ~100 GB, not ~3.4 TB
            "year": np.asarray(OUTPUT_YEARS),
            **{d: ciam_in[d].values for d in ["ssp", "iam"] if d in ciam_in.dims},
        }
    )
    # chunks must match the task write shape: tasks write SEG_CHUNKSIZE
    # segments x SAMPLE_CHUNKSIZE samples, and a chunk shared by several
    # tasks gets read-modify-written concurrently, silently reverting the
    # losers' slices to NaN. Seg chunks at the task size also halve the
    # store's file count, which matters on inode-limited filesystems.
    chunk_spec = {
        SEG_VAR: SEG_CHUNKSIZE,
        "case": len(CASES) - 1,
        "sample": SAMPLE_CHUNKSIZE,
    }
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
    parser.add_argument(
        "--template-only", action="store_true",
        help="write the template and exit (run once before a SLURM array)",
    )
    parser.add_argument(
        "--shard", metavar="K/N",
        help="SLURM array mode: run seg groups K::N serially, no cluster",
    )
    parser.add_argument(
        "--probe-only", action="store_true",
        help="count unfinished groups, save them for shard resume, and exit "
        "nonzero if any",
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

    unfinished_file = unfinished_tasks_file(f"04_{args.scenario}")
    resuming = not args.overwrite and zarr_exists(tmp_path)
    if not resuming:
        write_template(tmp_path, ciam_in, slr_path, samples, params)
        # a probe list describing the old store must not outlive it: a
        # stale anchor silently turns shards into no-ops
        save_unfinished(unfinished_file, [])
    if args.template_only:
        print("template ready.")
        return

    seg_groups = chunked(ciam_in[SEG_VAR].values, SEG_CHUNKSIZE)
    samp_groups = chunked(samples, SAMPLE_CHUNKSIZE)
    tasks = list(product(seg_groups, samp_groups))
    n_total = len(tasks)

    task_kwargs = dict(
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
        # pyCIAM defaults this to {}, which zarr rejects for local paths;
        # None works for local and gs:// alike, and pyCIAM forwards it to
        # every zarr op, so no platform conditional is needed
        storage_options=None,
        # everything the wrapper needs to build per-task mini templates
        _mini=dict(
            model_years=np.arange(
                params.model_start, ciam_in.year.max().item() + 1
            ),
            output_years=np.asarray(OUTPUT_YEARS),
            scenarios=["ncc_ar6"]
            + list(xr.open_zarr(str(slr_path), chunks=None).scenario.values),
            extra_coords={
                d: ciam_in[d].values for d in ["ssp", "iam"] if d in ciam_in.dims
            },
        ),
    )

    def unfinished(as_tasks=True, as_indices=False):
        """One null probe point of npv or costs per (seg_ir, sample) flags
        the owning task; tasks write both variables, and a task can lose a
        chunk race in one variable but not the other, so probe both."""
        ds = xr.open_zarr(str(tmp_path))
        npv = ds.npv.isel(case=0, scenario=0, ssp=0, iam=0, drop=True)
        costs = ds.costs.isel(
            case=0, costtype=0, scenario=0, year=-1, ssp=0, iam=0, drop=True
        )
        isnull = (
            (npv.isnull() | costs.isnull())
            .transpose(SEG_VAR, "sample")
            .compute()
            .values
        )
        if not as_tasks:
            return int(isnull.sum())
        seg_off = np.cumsum([0] + [len(g) for g in seg_groups])
        samp_off = np.cumsum([0] + [len(g) for g in samp_groups])
        hit = [
            (i, j)
            for i in range(len(seg_groups))
            for j in range(len(samp_groups))
            if isnull[seg_off[i] : seg_off[i + 1], samp_off[j] : samp_off[j + 1]].any()
        ]
        if as_indices:
            return hit
        return [(seg_groups[i], samp_groups[j]) for i, j in hit]

    if args.probe_only:
        print("probing npv and costs for unfinished groups")
        hit = unfinished(as_indices=True)
        print(f"{len(hit)} of {n_total} groups unfinished")
        save_unfinished(unfinished_file, hit)
        raise SystemExit(1 if hit else 0)

    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        run_shard(k, n, seg_groups, samp_groups, unfinished_file, task_kwargs)
        return

    cluster = Cluster(n_workers)
    cluster.start()

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
        # the probe already decided what to rerun; the store-side check
        # only reads costs, so it would wrongly skip npv-only holes
        return client.map(
            run_calc_all_cases, batch, check=not probe_driven, **task_kwargs
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
