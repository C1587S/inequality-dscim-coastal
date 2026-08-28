"""
finish the production run: fill calc_all_cases gaps, run optimization, aggregate.
auto-reconnects if scheduler dies.
run: nohup python -u 00_finish_run.py > finish_run.log 2>&1 &
"""

import time
import json
import subprocess
import sys
import importlib
import traceback

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
import os

dask.config.set({"array.rechunk.method": "tasks"})

from config import *
from pyCIAM.constants import CASES, COSTTYPES, SOLVCASES
from pyCIAM.io import create_template_dataarray, save_to_zarr_region
from pyCIAM.run import calc_all_cases, get_refA, optimize_case
from pyCIAM.utils import add_attrs_to_result, collapse_econ_inputs_to_seg, subset_econ_inputs

N_SAMP = N_SAMPLES_TOTAL
PROD_SEG_CHUNKSIZE = 2
PROD_SAMPLE_CHUNKSIZE = 100
PROD_N_WORKERS = 600
BATCH_SIZE = 300  # smaller batches = less risk of scheduler crash

PROD_SLR = PATH_SLR_INEQUALITY
PROD_SLIIDERS_SEG = PATH_SLIIDERS_SEG
PROD_REFA = PATH_REFA_INEQUALITY
PROD_TMP = PATH_OUTPUT_TMP
PROD_INTERMEDIATE = DIR_SCRATCH / 'pyciam-inequality-intermediate.zarr'
PROD_FINAL = PATH_OUTPUT_FINAL


# cluster management

img = os.environ.get('JUPYTER_IMAGE', None)
client = None
cluster = None

def create_cluster():
    global client, cluster
    from dask_gateway import Gateway
    from distributed.diagnostics.plugin import UploadDirectory

    gateway = Gateway()
    for c in gateway.list_clusters():
        try: gateway.stop_cluster(c.name)
        except: pass
    time.sleep(10)

    cluster = gateway.new_cluster(
        idle_timeout=7200, profile='micro',
        **(dict(worker_image=img, scheduler_image=img) if img else {})
    )
    client = cluster.get_client()
    cluster.scale(PROD_N_WORKERS)

    # wait for some workers
    start = time.time()
    while len(client.scheduler_info()['workers']) < 5 and time.time() - start < 600:
        time.sleep(5)
    n = len(client.scheduler_info()['workers'])
    print(f'  cluster ready: {n} workers, dashboard: {client.dashboard_link}')

    import pyCIAM
    client.register_plugin(
        UploadDirectory(str(Path(pyCIAM.__file__).parent), update_path=True, restart_workers=False),
        name='pyciam-upload',
    )
    client.run(lambda: __import__('subprocess').check_call(['pip', 'install', '-q', 'cloudpathlib']))
    return client, cluster

def ensure_cluster():
    global client, cluster
    try:
        client.scheduler_info()
    except:
        print('  reconnecting cluster...')
        client, cluster = create_cluster()
    return client

def run_calc_all_cases(grp, **kwargs):
    from pyCIAM.run import calc_all_cases
    return calc_all_cases(grp, **kwargs)

def run_optimize_case(*args, **kwargs):
    from pyCIAM.run import optimize_case
    return optimize_case(*args, **kwargs)


# setup

print('--- starting cluster ---')
client, cluster = create_cluster()

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
print(f'  seg_ir: {len(ciam_in[SEG_VAR])}')

timings = {}


# calc_all_cases gap fill (check=True skips completed tasks)

print('\n--- calc_all_cases (gap fill, check=True) ---')
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
print(f'  total tasks: {total_tasks} (most will be skipped by check=True)')

n_ok_total = 0
n_err_total = 0
n_skipped = 0
batch_idx = 0

while batch_idx * BATCH_SIZE < total_tasks:
    batch_start = batch_idx * BATCH_SIZE
    batch_end = min(batch_start + BATCH_SIZE, total_tasks)
    batch = all_grps[batch_start:batch_end]
    batch_num = batch_idx + 1
    total_batches = (total_tasks + BATCH_SIZE - 1) // BATCH_SIZE

    try:
        client = ensure_cluster()
        from distributed import wait

        futs = list(client.map(
            run_calc_all_cases, batch,
            params=params, econ_input_path=econ_input_path,
            slr_input_paths=slr_input_paths, slr_names=slr_names,
            output_path=str(PROD_TMP), refA_path=str(PROD_REFA),
            surge_input_path=surge_input_paths[SEG_VAR],
            seg_var=SEG_VAR, mc_dim=mc_dim, quantiles=quantiles,
            storage_options=storage_options, diaz_inputs=False, check=True,
        ))
        wait(futs)

        n_ok = sum(1 for f in futs if f.status == 'finished')
        n_err = sum(1 for f in futs if f.status == 'error')
        n_ok_total += n_ok
        n_err_total += n_err
        elapsed = time.time() - t0
        print(f'  batch {batch_num}/{total_batches}: ok={n_ok} err={n_err} | total={n_ok_total}/{total_tasks} | {elapsed/60:.0f}min')

        if n_err > 0:
            for f in futs:
                if f.status == 'error':
                    try: f.result()
                    except Exception as e: print(f'    err: {str(e)[:200]}')
                    break

        batch_idx += 1

    except Exception as e:
        print(f'  batch {batch_num} failed: {str(e)[:200]}')
        print(f'  reconnecting and retrying batch {batch_num}...')
        time.sleep(30)
        try:
            client, cluster = create_cluster()
        except:
            print('  cluster creation failed, waiting 60s...')
            time.sleep(60)
            client, cluster = create_cluster()

timings['calc_all_cases_gapfill'] = time.time() - t0
print(f'  done: {timings["calc_all_cases_gapfill"]/60:.0f}min, ok={n_ok_total}, err={n_err_total}')


# optimization

print('\n--- optimization ---')
client = ensure_cluster()
t0 = time.time()

n_opt_groups = 4
samples_per_group = N_SAMP // n_opt_groups
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
opt_task_list = list(precurser_futs.reset_index(drop=False).itertuples(index=False))
total_opt = len(opt_task_list)
print(f'  total tasks: {total_opt}')

n_ok_total = 0
n_err_total = 0
batch_idx = 0

while batch_idx * BATCH_SIZE < total_opt:
    batch_start = batch_idx * BATCH_SIZE
    batch_end = min(batch_start + BATCH_SIZE, total_opt)
    batch_rows = opt_task_list[batch_start:batch_end]
    batch_num = batch_idx + 1
    total_batches = (total_opt + BATCH_SIZE - 1) // BATCH_SIZE

    try:
        client = ensure_cluster()
        from distributed import wait

        futs = []
        for row in batch_rows:
            seg_var_val = getattr(row, SEG_VAR)
            group_ids = row.group_id
            samp_idx = row.samples
            f = client.submit(
                run_optimize_case, seg_var_val, *group_ids,
                quantiles=sample_ids[samp_idx],
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
        remaining = (total_opt - batch_end) / rate / 60 if rate > 0 else 0
        print(f'  batch {batch_num}/{total_batches}: ok={n_ok} err={n_err} | total={n_ok_total}/{total_opt} | {elapsed/60:.0f}min | est left: {remaining:.0f}min')

        if n_err > 0:
            for f in futs:
                if f.status == 'error':
                    try: f.result()
                    except Exception as e: print(f'    err: {str(e)[:200]}')
                    break

        batch_idx += 1

    except Exception as e:
        print(f'  batch {batch_num} failed: {str(e)[:200]}')
        print(f'  reconnecting and retrying...')
        time.sleep(30)
        try:
            client, cluster = create_cluster()
        except:
            time.sleep(60)
            client, cluster = create_cluster()

timings['optimization'] = time.time() - t0
print(f'  done: {timings["optimization"]/60:.0f}min, ok={n_ok_total}, err={n_err_total}')


# aggregation

print('\n--- aggregation ---')
client = ensure_cluster()
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
print(f'  done: {timings["aggregation"]/60:.0f}min')


# verify

print('\n--- verify ---')
verify = xr.open_zarr(str(PROD_FINAL))
print(f'  dims: {dict(verify.sizes)}')
n_nonnull = int(verify.costs.notnull().sum())
n_total = int(verify.costs.size)
print(f'  non-null: {n_nonnull}/{n_total} ({100*n_nonnull/n_total:.1f}%)')

assert (
    verify.sel(case='optimalfixed', drop=True).sum(dim='costtype')
    .costs.notnull().all()
), "optimalfixed has null values!"
print('  passed.')


# results

n_workers = max(len(client.scheduler_info()['workers']), 1)
total = sum(timings.values())
print(f'\ntotal: {total/3600:.1f}h, workers: {n_workers}')
print(f'output: {PROD_FINAL}')

report = {'timings': timings, 'total_hours': total / 3600, 'output': str(PROD_FINAL)}
with open('/home/jovyan/inequality-dscim-coastal/inequality/finish_run_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)

client.close()
cluster.close()
print('done.')
