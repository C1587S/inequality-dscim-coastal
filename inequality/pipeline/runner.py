"""
Shared machinery for the pipeline stages: the vendored pyCIAM install, Dask
Gateway cluster lifecycle, batched submission that survives scheduler death,
and run reports.

A stage script calls install_vendored_pyciam() before importing pyCIAM, runs
its heavy work through a Cluster and run_batched, and ends with write_report.
Stages write through pyCIAM's check=True machinery, so rerunning a stage after
a crash skips completed work.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = Path(__file__).resolve().parent


def install_vendored_pyciam():
    """Install the pyciam/ checkout at the repo root so it wins over any
    release installed in the environment. Call before importing pyCIAM."""
    import importlib

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
         "--no-deps", str(REPO_ROOT / "pyciam")]
    )
    importlib.invalidate_caches()


class Cluster:
    """A Dask Gateway cluster that can be recreated when the scheduler dies."""

    def __init__(self, n_workers, min_workers=5, idle_timeout=7200, ship_pyciam=True):
        self.n_workers = n_workers
        self.min_workers = min(min_workers, n_workers)
        self.idle_timeout = idle_timeout
        self.ship_pyciam = ship_pyciam
        self.client = None
        self._cluster = None

    def start(self):
        from dask_gateway import Gateway

        gateway = Gateway()
        for c in gateway.list_clusters():
            try:
                gateway.stop_cluster(c.name)
            except Exception:
                pass
        time.sleep(10)

        img = os.environ.get("JUPYTER_IMAGE")
        self._cluster = gateway.new_cluster(
            idle_timeout=self.idle_timeout,
            profile="micro",
            **(dict(worker_image=img, scheduler_image=img) if img else {}),
        )
        self.client = self._cluster.get_client()
        self._cluster.scale(self.n_workers)

        deadline = time.time() + 600
        while self._n_alive() < self.min_workers and time.time() < deadline:
            time.sleep(5)
        n = self._n_alive()
        if n == 0:
            raise RuntimeError("no workers came up within 10 minutes")
        print(f"cluster: {n} workers, dashboard: {self.client.dashboard_link}")

        if self.ship_pyciam:
            self._push_pyciam()
        return self.client

    def ensure(self):
        """Return a live client, recreating the cluster if the scheduler died."""
        try:
            self.client.scheduler_info()
        except Exception:
            print("scheduler unreachable, recreating cluster...")
            self.start()
        return self.client

    def close(self):
        try:
            self.client.close()
            self._cluster.close()
        except Exception:
            pass

    def _n_alive(self):
        return len(self.client.scheduler_info()["workers"])

    def _push_pyciam(self):
        """Ship the locally installed pyCIAM to workers, whose image carries an
        older release, and install cloudpathlib which the image lacks."""
        from distributed.diagnostics.plugin import UploadDirectory

        import pyCIAM

        self.client.register_plugin(
            UploadDirectory(
                str(Path(pyCIAM.__file__).parent),
                update_path=True,
                restart_workers=False,
            ),
            name="pyciam-upload",
        )
        self.client.run(
            lambda: __import__("subprocess").check_call(
                ["pip", "install", "-q", "cloudpathlib"]
            )
        )


def run_batched(cluster, make_futures, tasks, batch_size, label):
    """Run tasks through make_futures(client, batch) one batch at a time.

    A batch interrupted by scheduler death is retried on a fresh cluster.
    Combined with check=True in the pyCIAM entry points, that makes each stage
    restartable: completed tasks are skipped on the next pass.

    Returns (n_ok, n_err).
    """
    from distributed import wait

    total = len(tasks)
    n_batches = (total + batch_size - 1) // batch_size
    n_ok = n_err = 0
    t0 = time.time()
    ix = 0
    while ix < n_batches:
        batch = tasks[ix * batch_size : (ix + 1) * batch_size]
        try:
            client = cluster.ensure()
            futs = list(make_futures(client, batch))
            wait(futs)
        except Exception as e:
            print(f"{label} batch {ix + 1}/{n_batches} failed: {str(e)[:200]}")
            time.sleep(30)
            continue
        ok = sum(f.status == "finished" for f in futs)
        err = len(futs) - ok
        n_ok += ok
        n_err += err
        _print_progress(label, ix + 1, n_batches, ok, err, n_ok, total, t0)
        if err:
            _print_first_error(futs)
        ix += 1
    return n_ok, n_err


def _print_progress(label, batch, n_batches, ok, err, done, total, t0):
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0
    eta = f"~{(total - done) / rate / 3600:.1f}h left" if rate > 0 else ""
    print(
        f"{label} batch {batch}/{n_batches}: ok={ok} err={err}"
        f" | {done}/{total} | {elapsed / 60:.0f}min {eta}"
    )


def _print_first_error(futs):
    for f in futs:
        if f.status == "error":
            try:
                f.result()
            except Exception as e:
                print(f"  first error: {str(e)[:200]}")
            return


def zarr_exists(path):
    """True if a readable zarr store exists at path."""
    import xarray as xr

    try:
        xr.open_zarr(str(path), chunks=None).close()
        return True
    except Exception:
        return False


def clean_for_zarr(ds):
    """Clear encodings and coerce object coords so stores write as zarr v2."""
    for v in ds.data_vars:
        ds[v].encoding.clear()
    for k, v in ds.coords.items():
        v.encoding.clear()
        if v.dtype == object:
            ds[k] = v.astype("unicode")
    return ds


def write_report(stage, timings, **extra):
    """Print a timing summary and write a json report beside the scripts."""
    total = sum(timings.values())
    for step, secs in timings.items():
        print(f"  {step:<25} {secs:>10.1f}s  {secs / 3600:>6.2f}h")
    print(f"  {'total':<25} {total:>10.1f}s  {total / 3600:>6.2f}h")
    path = PIPELINE_DIR / f"{stage}_report.json"
    report = {"timings_seconds": timings, "total_hours": total / 3600, **extra}
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"report: {path}")
