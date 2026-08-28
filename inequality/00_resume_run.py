"""
resume production run from calc_all_cases onwards.
skips: slr, sliiders, template, refa (all already done)
fix: submits tasks in batches to avoid crashing the scheduler

run: nohup python -u 00_resume_run.py > resume_run.log 2>&1 &
check: tail -f resume_run.log
"""

import time
import json
import subprocess
import sys
import importlib

subprocess.check_call([
    sys.executable, '-m', 'pip', 'install', '-q', '--force-reinstall', '--no-deps',
    '/home/jovyan/inequality-dscim-coastal/pyciam'
])
importlib.invalidate_caches()

import numpy as np
import pandas as pd
import xarray as xr
import dask
from collections import OrderedDict
from itertools import product
from pathlib import Path
from cloudpathlib import AnyPath
from gcsfs import GCSFileSystem

dask.config.set({"array.rechunk.method": "tasks"})

from config import *

from pyCIAM.constants import CASES, COSTTYPES, SOLVCASES
from pyCIAM.io import create_template_dataarray, save_to_zarr_region
from pyCIAM.run import calc_all_cases, get_refA, optimize_case
from pyCIAM.utils import add_attrs_to_result, collapse_econ_inputs_to_seg, subset_econ_inputs

timings = {}

N_SAMP = N_SAMPLES_TOTAL  # 1000
PROD_SEG_CHUNKSIZE = 2
PROD_SAMPLE_CHUNKSIZE = 100
PROD_N_WORKERS = 600
BATCH_SIZE = 500  # submit this many tasks at a time to avoid killing scheduler

# paths (all already exist from previous run)
PROD_SLR = PATH_SLR_INEQUALITY
PROD_SLIIDERS_SEG = PATH_SLIIDERS_SEG
PROD_REFA = PATH_REFA_INEQUALITY
PROD_TMP = PATH_OUTPUT_TMP
PROD_INTERMEDIATE = DIR_SCRATCH / 'pyciam-inequality-intermediate.zarr'
PROD_FINAL = PATH_OUTPUT_FINAL

# verify everything we need exists
print('--- checking existing data ---')
for name, path in [
    ('slr', PROD_SLR), ('sliiders_seg', PROD_SLIIDERS_SEG),
    ('refa', PROD_REFA), ('output_template', PROD_TMP),
]:
    try:
        ds = xr.open_zarr(str(path), chunks=None)
        print(f'  {name}: ok ({dict(ds.sizes)})')
        ds.close()
    except Exception as e:
        print(f'  {name}: MISSING ({e})')
        print(f'  cannot resume, run 00_full_run.py first')
        sys.exit(1)


# cluster
print('\n--- starting cluster ---')
import os
from dask_gateway import Gateway
from distributed import wait
from distributed.diagnostics.plugin import UploadDirectory

img = os.environ.get('JUPYTER_IMAGE', None)
gateway = Gateway()

for c in gateway.list_clusters():
    try:
        gateway.stop_cluster(c.name)
    except:
        pass
time.sleep(10)

cluster = gateway.new_cluster(
    idle_timeout=7200,
    profile='micro',
    **(dict(worker_image=img, scheduler_image=img) if img else {})
)
client = cluster.get_client()
cluster.scale(PROD_N_WORKERS)

# wait for at least 40 workers before starting heavy computation
print('waiting for workers (need at least 40)...')
start = time.time()
deadline = start + 600
while len(client.scheduler_info()['workers']) < 40 and time.time() < deadline:
    elapsed = int(time.time() - start)
    n = len(client.scheduler_info()['workers'])
    print(f'  {elapsed}s: {n} workers')
    time.sleep(10)

n = len(client.scheduler_info()['workers'])
print(f'workers: {n}')
if n < 5:
    print('not enough workers, exiting.')
    sys.exit(1)

import pyCIAM
client.register_plugin(
    UploadDirectory(str(Path(pyCIAM.__file__).parent), update_path=True, restart_workers=False),
    name='pyciam-upload',
)
client.run(lambda: __import__('subprocess').check_call(['pip', 'install', '-q', 'cloudpathlib']))
print(f'pyciam installed on {n} workers')
print(f'dashboard: {client.dashboard_link}')


def run_calc_all_cases(grp, **kwargs):
    from pyCIAM.run import calc_all_cases
    return calc_all_cases(grp, **kwargs)

def run_optimize_case(*args, **kwargs):
    from pyCIAM.run import optimize_case
    return optimize_case(*args, **kwargs)

def ensure_packages():
    client.register_plugin(
        UploadDirectory(str(Path(pyCIAM.__file__).parent), update_path=True, restart_workers=False),
        name='pyciam-upload',
    )
    client.run(lambda: __import__('subprocess').check_call(['pip', 'install', '-q', 'cloudpathlib']))


# load econ inputs
params = pd.read_json(PATH_PARAMS)['values']
quantiles = np.arange(1, N_SAMP + 1)
econ_input_path = str(PATH_SLIIDERS)
slr_input_paths = [PROD_SLR]
slr_names = ['ar6']
surge_input_paths = {k: AnyPath(v) for k, v in PATHS_SURGE_LOOKUP.items()}
mc_dim = MC_DIM
storage_options = STORAGE_OPTIONS

ciam_in = subset_econ_inputs(
    xr.open_zarr(econ_input_path, chunks=None, storage_options=storage_options),
    SEG_VAR, seg_var_subset=None,
)
ciam_in = ciam_in.sel(year=np.concatenate((np.arange(2040, 2060), np.arange(2080, 2100))))
print(f'seg_ir: {len(ciam_in[SEG_VAR])}')


# calc_all_cases (in batches)
print('\n--- calc_all_cases ---')
ensure_packages()
t0 = time.time()

groups = [
    ciam_in[SEG_VAR].isel({SEG_VAR: slice(i, i + PROD_SEG_CHUNKSIZE)}).values
    for i in np.arange(0, len(ciam_in[SEG_VAR]), PROD_SEG_CHUNKSIZE)
]
groups_ser = (
    pd.Series(groups).explode().reset_index()
    .rename(columns={'index': 'group_id', 0: SEG_VAR})
    .set_index(SEG_VAR).group_id
)
samps = np.arange(1, N_SAMP + 1)
samp_grps = [samps[i:i + PROD_SAMPLE_CHUNKSIZE] for i in range(0, len(samps), PROD_SAMPLE_CHUNKSIZE)]
all_grps = list(product(groups, samp_grps))
total_tasks = len(all_grps)
print(f'  total tasks: {total_tasks}')

# submit in batches
all_ciam_futs = []
n_ok_total = 0
n_err_total = 0

for batch_start in range(0, total_tasks, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, total_tasks)
    batch = all_grps[batch_start:batch_end]
    batch_num = batch_start // BATCH_SIZE + 1
    total_batches = (total_tasks + BATCH_SIZE - 1) // BATCH_SIZE
    print(f'  batch {batch_num}/{total_batches}: tasks {batch_start}-{batch_end}...')

    # reinstall packages in case workers got recycled between batches
    if batch_start > 0 and batch_start % (BATCH_SIZE * 5) == 0:
        ensure_packages()

    futs = list(client.map(
        run_calc_all_cases, batch,
        params=params, econ_input_path=econ_input_path,
        slr_input_paths=slr_input_paths, slr_names=slr_names,
        output_path=str(PROD_TMP), refA_path=str(PROD_REFA),
        surge_input_path=surge_input_paths[SEG_VAR],
        seg_var=SEG_VAR, mc_dim=mc_dim, quantiles=quantiles,
        storage_options=storage_options, diaz_inputs=False, check=False,
    ))
    wait(futs)

    n_ok = sum(1 for f in futs if f.status == 'finished')
    n_err = sum(1 for f in futs if f.status == 'error')
    n_ok_total += n_ok
    n_err_total += n_err
    elapsed = time.time() - t0
    rate = n_ok_total / elapsed if elapsed > 0 else 0
    remaining = (total_tasks - batch_end) / rate / 3600 if rate > 0 else 0
    print(f'    ok: {n_ok}, err: {n_err} | total: {n_ok_total}/{total_tasks} | {elapsed/3600:.1f}h elapsed | est remaining: {remaining:.1f}h')

    if n_err > 0:
        for f in futs:
            if f.status == 'error':
                try: f.result()
                except Exception as e: print(f'    error sample: {e}')
                break

    all_ciam_futs.extend(futs)

timings['calc_all_cases'] = time.time() - t0
print(f'  done: {timings["calc_all_cases"]:.1f}s ({timings["calc_all_cases"]/3600:.1f}h), ok: {n_ok_total}, err: {n_err_total}')


# optimization (in batches too)
print('\n--- optimization ---')
ensure_packages()
t0 = time.time()

n_opt_groups = 4
samples_per_group = N_SAMP // n_opt_groups  # 250
sample_ids = {}
for i in range(n_opt_groups):
    sample_ids[i] = np.arange(samples_per_group * i + 1, samples_per_group * (i + 1) + 1)
print(f'  {n_opt_groups} groups of {samples_per_group} samples')

seg_adm_ser = pd.Series(ciam_in[SEG_VAR].values)
seg_adm_ser.index = ciam_in.seg.values
seg_grps_opt = seg_adm_ser.groupby(seg_adm_ser.index).apply(list)
precurser_futs = seg_adm_ser.to_frame(SEG_VAR).join(seg_grps_opt.rename('seg_group'))
precurser_futs.loc[:, 'samples'] = [np.arange(0, n_opt_groups)] * len(precurser_futs)
precurser_futs = (
    precurser_futs.explode('samples')
    .set_index([SEG_VAR, 'samples']).seg_group.explode().to_frame()
    .join(groups_ser, on='seg_group')
    .groupby([SEG_VAR, 'samples']).group_id.apply(set).apply(list)
)

# build all optimization tasks as a list
opt_task_list = []
for _, row in precurser_futs.reset_index(drop=False).iterrows():
    opt_task_list.append(row)
total_opt = len(opt_task_list)
print(f'  total optimization tasks: {total_opt}')

n_ok_total = 0
n_err_total = 0
for batch_start in range(0, total_opt, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, total_opt)
    batch_rows = opt_task_list[batch_start:batch_end]
    batch_num = batch_start // BATCH_SIZE + 1
    total_batches = (total_opt + BATCH_SIZE - 1) // BATCH_SIZE
    print(f'  batch {batch_num}/{total_batches}: tasks {batch_start}-{batch_end}...')

    if batch_start > 0 and batch_start % (BATCH_SIZE * 5) == 0:
        ensure_packages()

    futs = []
    for row in batch_rows:
        f = client.submit(
            run_optimize_case, row[SEG_VAR], *row.group_id,
            quantiles=sample_ids[row['samples']],
            econ_input_path=econ_input_path,
            output_path=str(PROD_TMP), seg_var=SEG_VAR,
            eps=1, check=False, storage_options=storage_options,
        )
        futs.append(f)
    wait(futs)

    n_ok = sum(1 for f in futs if f.status == 'finished')
    n_err = sum(1 for f in futs if f.status == 'error')
    n_ok_total += n_ok
    n_err_total += n_err
    elapsed = time.time() - t0
    rate = n_ok_total / elapsed if elapsed > 0 else 0
    remaining = (total_opt - batch_end) / rate / 3600 if rate > 0 else 0
    print(f'    ok: {n_ok}, err: {n_err} | total: {n_ok_total}/{total_opt} | {elapsed/3600:.1f}h elapsed | est remaining: {remaining:.1f}h')

    if n_err > 0:
        for f in futs:
            if f.status == 'error':
                try: f.result()
                except Exception as e: print(f'    error sample: {e}')
                break

timings['optimization'] = time.time() - t0
print(f'  done: {timings["optimization"]:.1f}s ({timings["optimization"]/3600:.1f}h), ok: {n_ok_total}, err: {n_err_total}')


# aggregation
print('\n--- aggregation ---')
t0 = time.time()
AGG_VAR = 'impact_region'
t = xr.open_zarr(str(PROD_TMP))
t = t.sel(case=OUTPUT_CASES, ssp=OUTPUT_SSPS, year=OUTPUT_YEARS)[['costs']]
for v in t.data_vars: t[v].encoding.clear()
for k, v in t.coords.items():
    v.encoding.clear()
    if v.dtype == object: t[k] = v.astype('unicode')
t.to_zarr(str(PROD_INTERMEDIATE), mode='w', zarr_format=2)

this_chunksize = 2
out = xr.open_zarr(str(PROD_INTERMEDIATE), chunks={'case': -1, SEG_VAR: this_chunksize})
out['costs'] = out.costs.groupby(ciam_in[AGG_VAR]).sum().chunk({AGG_VAR: this_chunksize}).persist()
out = out.drop_vars(SEG_VAR).unify_chunks()
for v in out.data_vars: out[v].encoding.clear()
for k, v in out.coords.items():
    v.encoding.clear()
    if v.dtype == object: out[k] = v.astype('unicode')
out = out.persist()
out.to_zarr(str(PROD_FINAL), storage_options=storage_options, mode='w', zarr_format=2)
timings['aggregation'] = time.time() - t0
print(f'  done: {timings["aggregation"]:.1f}s')


# verify
print('\n--- verify ---')
verify = xr.open_zarr(str(PROD_FINAL))
print(f'  dims: {dict(verify.sizes)}')
n_nonnull = int(verify.costs.notnull().sum())
n_total = int(verify.costs.size)
pct = 100 * n_nonnull / n_total if n_total > 0 else 0
print(f'  non-null: {n_nonnull}/{n_total} ({pct:.1f}%)')

assert (
    verify.sel(case='optimalfixed', drop=True).sum(dim='costtype')
    .costs.notnull().all()
), "optimalfixed has null values!"
print('  final check passed.')


# results
n_workers_final = max(len(client.scheduler_info()['workers']), 1)
print('\n' + '=' * 50)
print('timings')
print('=' * 50)
total = sum(timings.values())
for step, secs in timings.items():
    print(f'  {step:<25} {secs:>10.1f}s  {secs/3600:>6.2f}h')
print(f'  {"total":<25} {total:>10.1f}s  {total/3600:>6.2f}h')
print(f'  workers: {n_workers_final}')
print(f'  worker-hours: {total * n_workers_final / 3600:.1f}')

report = {
    'total_seconds': total,
    'total_hours': total / 3600,
    'n_workers': n_workers_final,
    'worker_hours': total * n_workers_final / 3600,
    'timings_seconds': timings,
    'output_path': str(PROD_FINAL),
    'output_dims': dict(verify.sizes),
}
report_path = Path('/home/jovyan/inequality-dscim-coastal/inequality/resume_run_report.json')
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2, default=str)
print(f'\nreport: {report_path}')
print(f'output: {PROD_FINAL}')

client.close()
cluster.close()
print('\ndone.')
