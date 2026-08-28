"""
test run and cost estimation for pyciam inequality pipeline.
run: nohup python -u 00b_test_run.py > test_run.log 2>&1 &
check: tail -f test_run.log
"""

import time
import json
import subprocess
import sys
import importlib

# install pyciam from the inequality branch before anything else
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

import inspect
print(f"pyciam: {Path(inspect.getfile(get_refA)).parent}")
print(f"get_refA sig: {inspect.signature(get_refA)}")

timings = {}

T_SCENARIOS = ['tlim2.0']
T_WORKFLOWS = ['wf_1f']
T_N_SAMP_PER_WF = 5
T_N_SAMP = 5
T_N_SEGMENTS = 50
T_SEG_CHUNKSIZE = 1

T_DIR = DIR_SCRATCH / 'test-run-slr-inputs'
T_SLR = T_DIR / 'costtest-slr.zarr'
T_SLIIDERS_SEG = T_DIR / 'costtest-sliiders-seg.zarr'
T_REFA = T_DIR / 'costtest-refa.zarr'
T_TMP = T_DIR / 'costtest-pyciam-tmp.zarr'
T_INTERMEDIATE = T_DIR / 'costtest-pyciam-intermediate.zarr'
T_FINAL = T_DIR / 'costtest-pyciam-final.zarr'

F_N_SCENARIOS = 5
F_N_WORKFLOWS = 2
F_N_SAMPLES = 1000
F_N_WORKERS = 800

print('parameters set.')

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
    idle_timeout=3600,
    profile='micro',
    **(dict(worker_image=img, scheduler_image=img) if img else {})
)
client = cluster.get_client()
cluster.scale(3)

start = time.time()
deadline = start + 300
while len(client.scheduler_info()['workers']) < 1 and time.time() < deadline:
    elapsed = int(time.time() - start)
    n = len(client.scheduler_info()['workers'])
    print(f'  waiting... {elapsed}s, workers: {n}')
    time.sleep(5)

n = len(client.scheduler_info()['workers'])
print(f'workers ready: {n}')
if n == 0:
    print('no workers, exiting.')
    sys.exit(1)

import pyCIAM
client.register_plugin(
    UploadDirectory(
        str(Path(pyCIAM.__file__).parent),
        update_path=True,
        restart_workers=False,
    ),
    name='pyciam-upload',
)
client.run(lambda: __import__('subprocess').check_call(['pip', 'install', '-q', 'cloudpathlib']))
result = client.submit(lambda: str(__import__('pyCIAM.run', fromlist=['get_refA']))).result(timeout=60)
print(f'pyciam installed on {n} workers')
print(f'dashboard: {client.dashboard_link}')


def run_get_refA(grp, **kwargs):
    from pyCIAM.run import get_refA
    return get_refA(grp, **kwargs)

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


import pint_xarray

def open_and_convert_zarr(ds_path):
    out = xr.open_zarr(ds_path)
    out['sea_level_change'] = out.sea_level_change.pint.quantify().pint.to('meters').pint.dequantify()
    return out

def open_and_convert_nc(ds_path):
    _path = str(ds_path).replace('gs://', '/gcs/')
    out = xr.open_dataset(_path)
    out['sea_level_change'] = out.sea_level_change.pint.quantify().pint.to('meters').pint.dequantify()
    return out


print('\n--- slr local + global ---')
t0 = time.time()
np.random.seed(11222023)
nsamps = 20000
low = np.arange(0, nsamps, step=nsamps / T_N_SAMP_PER_WF)
high = low + nsamps / T_N_SAMP_PER_WF
quants = np.random.randint(low=low, high=high, size=None) / nsamps

local_dfs, global_dfs = [], []
for tlim, wf in product(T_SCENARIOS, T_WORKFLOWS):
    print(f'  {tlim}/{wf}...')
    lp = f'{DIR_SLR_AR6_GRIDDED_PUBLIC}/{wf}/{tlim}win0.25/total-workflow.zarr'
    df = open_and_convert_zarr(lp)
    df = (df.sel(years=slice(2020, 2100), drop=True)
          .sea_level_change.chunk(dict(samples=-1))
          .quantile(quants, dim='samples')
          .assign_coords({'workflow': wf, 'tlim': tlim})
          .expand_dims(['workflow', 'tlim']))
    local_dfs.append(df)
    gp = DIR_SLR_AR6_RAW / wf / f'{tlim}win0.25' / 'total-workflow.nc'
    gdf = open_and_convert_nc(gp)
    gdf = (gdf.sel(years=slice(2020, 2100), drop=True)
           .sea_level_change.chunk(dict(samples=-1))
           .quantile(quants, dim='samples')
           .assign_coords({'workflow': wf, 'tlim': tlim})
           .expand_dims(['workflow', 'tlim']))
    global_dfs.append(gdf)
timings['slr_local_global'] = time.time() - t0
print(f'  done: {timings["slr_local_global"]:.1f}s')


print('\n--- vlm ---')
t0 = time.time()
fs = GCSFileSystem(requester_pays=True)
mapping = fs.get_mapper(PATH_VLM_REQUESTER_PAYS)
np.random.seed(11222023)
low_v = np.arange(0, nsamps, step=nsamps / T_N_SAMP)
high_v = low_v + nsamps / T_N_SAMP
quants_v = np.random.randint(low=low_v, high=high_v, size=None) / nsamps
vlm_df = open_and_convert_zarr(mapping)
vlm_df = (vlm_df.sel(years=slice(2020, 2100), drop=True)
          .sea_level_change.chunk(dict(samples=-1))
          .quantile(quants_v, dim='samples')
          .rename({'quantile': 'samples'})
          .assign_coords({'samples': np.arange(1, T_N_SAMP + 1)})
          .to_dataset())
timings['slr_vlm'] = time.time() - t0
print(f'  done: {timings["slr_vlm"]:.1f}s')


print('\n--- combine + save slr ---')
t0 = time.time()
df_full = xr.combine_by_coords(local_dfs)
df_full = df_full.stack(samples=['workflow', 'quantile'])
df_full = df_full.reset_index('samples').drop_vars(['workflow', 'quantile'])
df_full['samples'] = np.arange(1, T_N_SAMP + 1)

global_ds = xr.combine_by_coords(global_dfs)
global_ds = global_ds.stack(samples=['workflow', 'quantile'])
global_ds = global_ds.reset_index('samples').drop_vars(['workflow', 'quantile'])
global_ds['samples'] = np.arange(1, T_N_SAMP + 1)
global_ds = global_ds.squeeze(drop=True).sea_level_change

vlm_df['samples'] = np.arange(1, T_N_SAMP + 1)

all_ds = xr.Dataset({
    'lsl_msl05': df_full.sea_level_change,
    'lsl_ncc_msl05': vlm_df.sea_level_change,
    'gsl_msl05': global_ds,
    'lon': vlm_df.lon, 'lat': df_full.lat,
})
all_ds['lon'] = all_ds.lon.where(all_ds.lon != -180, 180)
all_ds = all_ds.stack(locations=['lat', 'lon'])
all_ds = all_ds.compute(scheduler='synchronous')

reduce_dims = [d for d in ['tlim', 'samples', 'years'] if d in all_ds.dims]
valid = (all_ds[['lsl_msl05', 'lsl_ncc_msl05']]
         .sel(years=slice(2100)).notnull().all(reduce_dims)
         .to_array('tmp').all('tmp'))
all_ds = all_ds.sel(locations=valid.where(valid, drop=True).locations)

rename_map = {}
if 'years' in all_ds.dims: rename_map['years'] = 'year'
if 'samples' in all_ds.dims: rename_map['samples'] = 'sample'
if 'locations' in all_ds.dims: rename_map['locations'] = 'site_id'
if 'tlim' in all_ds.dims: rename_map['tlim'] = 'scenario'
all_ds = all_ds.rename(rename_map)

n_sites = len(all_ds.site_id)
all_ds = all_ds.chunk({'site_id': -1, 'scenario': 1, 'year': -1, 'sample': T_N_SAMP})

lat_vals = all_ds.coords['lat'].values
lon_vals = all_ds.coords['lon'].values
all_ds = all_ds.drop_vars(['lat', 'lon'])
all_ds['site_id'] = np.arange(len(all_ds.site_id))
all_ds['lat'] = ('site_id', lat_vals)
all_ds['lon'] = ('site_id', lon_vals)

for v in all_ds.data_vars: all_ds[v].encoding.clear()
for k, v in all_ds.coords.items():
    v.encoding.clear()
    if v.dtype == object: all_ds[k] = v.astype("unicode")
all_ds.to_zarr(str(T_SLR), mode='w', zarr_format=2)
timings['slr_combine_save'] = time.time() - t0
print(f'  done: {timings["slr_combine_save"]:.1f}s, {n_sites} sites')


# pyciam setup
params = pd.read_json(PATH_PARAMS)['values']
quantiles = np.arange(1, T_N_SAMP + 1)
econ_input_path = str(PATH_SLIIDERS)
slr_input_paths = [T_SLR]
slr_names = ['ar6']
surge_input_paths = {k: AnyPath(v) for k, v in PATHS_SURGE_LOOKUP.items()}
mc_dim = MC_DIM
storage_options = STORAGE_OPTIONS
diaz_inputs = False
eps = 1
model_kwargs = {}


# patch sliiders if needed (v1.2 has K_2014, pyciam expects K_2019)
print('\n--- checking sliiders ---')
sliiders_check = xr.open_zarr(econ_input_path, chunks=None)
sliiders_vars = list(sliiders_check.data_vars)
print(f'  vars: {sliiders_vars[:12]}')
needs_rename = 'K_2019' not in sliiders_vars and 'K_2014' in sliiders_vars
if needs_rename:
    print('  patching K_2014 -> K_2019')
    rename_dict = {}
    if 'K_2014' in sliiders_vars: rename_dict['K_2014'] = 'K_2019'
    if 'pop_2014' in sliiders_vars: rename_dict['pop_2014'] = 'pop_2019'
    patched = sliiders_check.rename(rename_dict)
    patched_path = T_DIR / 'sliiders-patched.zarr'
    for v in patched.data_vars: patched[v].encoding.clear()
    for k, v in patched.coords.items(): v.encoding.clear()
    patched.to_zarr(str(patched_path), mode='w', zarr_format=2)
    econ_input_path_for_collapse = str(patched_path)
else:
    econ_input_path_for_collapse = econ_input_path
    print('  no patch needed')
sliiders_check.close()


print('\n--- collapse sliiders ---')
t0 = time.time()
collapse_econ_inputs_to_seg(
    econ_input_path_for_collapse, T_SLIIDERS_SEG,
    seg_var_subset=None, output_chunksize=100,
    storage_options=storage_options, seg_var=SEG_VAR,
)
timings['collapse_sliiders'] = time.time() - t0
print(f'  done: {timings["collapse_sliiders"]:.1f}s')


ciam_in_full = subset_econ_inputs(
    xr.open_zarr(econ_input_path, chunks=None, storage_options=storage_options),
    SEG_VAR, seg_var_subset=None,
)
total_seg_ir = len(ciam_in_full[SEG_VAR])
test_seg_ir = ciam_in_full[SEG_VAR].values[:T_N_SEGMENTS]
ciam_in = ciam_in_full.sel({SEG_VAR: test_seg_ir})
ciam_in = ciam_in.sel(year=np.concatenate((np.arange(2040, 2060), np.arange(2080, 2100))))
print(f'  total seg_ir: {total_seg_ir}, test: {len(test_seg_ir)}')


# output template
slr_test = xr.open_zarr(str(T_SLR), chunks=None)
scenarios = slr_test.scenario.values
print(f'  scenarios: {scenarios}')

coords = OrderedDict({
    'case': CASES, 'costtype': COSTTYPES,
    SEG_VAR: ciam_in[SEG_VAR].values, 'scenario': scenarios,
    'sample': quantiles,
    'year': np.arange(params.model_start, ciam_in.year.max().item() + 1),
    **{d: ciam_in[d].values for d in ['ssp', 'iam'] if d in ciam_in.dims},
})
chunk_spec = {SEG_VAR: 1, 'case': len(coords['case']) - 1}
resolved_chunks = {k: chunk_spec.get(k, len(v) if hasattr(v, '__len__') else v) for k, v in coords.items()}
out_ds = create_template_dataarray(coords.keys(), coords, resolved_chunks).to_dataset(name='costs')
out_ds['npv'] = out_ds.costs.isel(year=0, costtype=0, drop=True).astype('float64')
out_ds['optimal_case'] = out_ds.npv.isel(case=0, drop=True).astype('uint8')
out_ds = add_attrs_to_result(out_ds)
out_ds.to_zarr(str(T_TMP), compute=False, mode='w', storage_options=storage_options, zarr_format=2)
print(f'  template: {dict(out_ds.sizes)}')


# refa
print('\n--- refa ---')
segs = np.unique(ciam_in.seg)
samps = np.arange(1, T_N_SAMP + 1)
n_segs = len(segs)

# create refa zarr template (save_to_zarr_region needs it to exist)
# taken from original notebook cell 28
xr.DataArray(
    coords=dict(
        sample=samps,
        seg=segs,
        case=(['sample', 'seg'], np.ones((len(samps), n_segs), dtype=np.int8))
    ),
    dims=('sample', 'seg')
).to_dataset(name='refA').chunk(
    {'seg': REFA_SEG_CHUNKSIZE, 'sample': T_N_SAMP}
).to_zarr(str(T_REFA), storage_options=storage_options, mode='w', zarr_format=2)
print(f'  refa template: (sample={len(samps)}, seg={n_segs})')

ensure_packages()
t0 = time.time()
seg_grps_refa = [segs[i:i + REFA_SEG_CHUNKSIZE] for i in range(0, len(segs), REFA_SEG_CHUNKSIZE)]
grps_refa = list(product(seg_grps_refa, [samps]))
print(f'  refa groups: {len(grps_refa)}')

refa_futs = client.map(
    run_get_refA, grps_refa,
    output_path=str(T_REFA), econ_input_path=str(T_SLIIDERS_SEG),
    slr_input_path=slr_input_paths[0], params=params,
    surge_input_path=surge_input_paths['seg'],
    mc_dim=mc_dim, storage_options=storage_options,
    quantiles=quantiles, diaz_inputs=diaz_inputs, eps=eps,
    **model_kwargs,
)
wait(refa_futs)
timings['refa'] = time.time() - t0
n_ok = sum(1 for f in refa_futs if f.status == 'finished')
n_err = sum(1 for f in refa_futs if f.status == 'error')
print(f'  done: {timings["refa"]:.1f}s, ok: {n_ok}, errors: {n_err}')
if n_err > 0:
    for f in refa_futs:
        if f.status == 'error':
            try: f.result()
            except Exception as e: print(f'  error: {e}')
            break


# calc_all_cases
print('\n--- calc_all_cases ---')
ensure_packages()
t0 = time.time()
groups = [
    ciam_in[SEG_VAR].isel({SEG_VAR: slice(i, i + T_SEG_CHUNKSIZE)}).values
    for i in np.arange(0, len(ciam_in[SEG_VAR]), T_SEG_CHUNKSIZE)
]
groups_ser = (
    pd.Series(groups).explode().reset_index()
    .rename(columns={'index': 'group_id', 0: SEG_VAR})
    .set_index(SEG_VAR).group_id
)
grps_ciam = list(product(groups, [samps]))
print(f'  groups: {len(grps_ciam)}')

ciam_futs = list(client.map(
    run_calc_all_cases, grps_ciam,
    params=params, econ_input_path=econ_input_path,
    slr_input_paths=slr_input_paths, slr_names=slr_names,
    output_path=str(T_TMP), refA_path=str(T_REFA),
    surge_input_path=surge_input_paths[SEG_VAR],
    seg_var=SEG_VAR, mc_dim=mc_dim, quantiles=quantiles,
    storage_options=storage_options, diaz_inputs=diaz_inputs, check=False,
    **model_kwargs,
))
wait(ciam_futs)
timings['calc_all_cases'] = time.time() - t0
n_ok = sum(1 for f in ciam_futs if f.status == 'finished')
n_err = sum(1 for f in ciam_futs if f.status == 'error')
print(f'  done: {timings["calc_all_cases"]:.1f}s, ok: {n_ok}, err: {n_err}')
if n_err > 0:
    for f in ciam_futs:
        if f.status == 'error':
            try: f.result()
            except Exception as e: print(f'  error: {e}')
            break


# optimization
print('\n--- optimization ---')
ensure_packages()
t0 = time.time()
sample_ids = {0: np.arange(1, T_N_SAMP + 1)}
n_opt_groups = 1

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
opt_futs = precurser_futs.reset_index(drop=False).apply(
    lambda row: client.submit(
        run_optimize_case, row[SEG_VAR], *row.group_id,
        quantiles=sample_ids[row['samples']],
        econ_input_path=econ_input_path,
        output_path=str(T_TMP), seg_var=SEG_VAR,
        eps=eps, check=False, storage_options=storage_options,
    ), axis=1,
)
wait(list(opt_futs))
timings['optimization'] = time.time() - t0
n_ok = sum(1 for f in opt_futs if f.status == 'finished')
n_err = sum(1 for f in opt_futs if f.status == 'error')
print(f'  done: {timings["optimization"]:.1f}s, ok: {n_ok}, err: {n_err}')
if n_err > 0:
    for f in opt_futs:
        if f.status == 'error':
            try: f.result()
            except Exception as e: print(f'  error: {e}')
            break


# aggregation
print('\n--- aggregation ---')
t0 = time.time()
AGG_VAR = 'impact_region'
t = xr.open_zarr(str(T_TMP))
t = t.sel(case=OUTPUT_CASES, ssp=OUTPUT_SSPS, year=OUTPUT_YEARS)[['costs']]
for v in t.data_vars: t[v].encoding.clear()
for k, v in t.coords.items():
    v.encoding.clear()
    if v.dtype == object: t[k] = v.astype('unicode')
t.to_zarr(str(T_INTERMEDIATE), mode='w', zarr_format=2)

out = xr.open_zarr(str(T_INTERMEDIATE), chunks={'case': -1, SEG_VAR: 2})
out['costs'] = out.costs.groupby(ciam_in[AGG_VAR]).sum().chunk({AGG_VAR: 2}).persist()
out = out.drop_vars(SEG_VAR).unify_chunks()
for v in out.data_vars: out[v].encoding.clear()
for k, v in out.coords.items():
    v.encoding.clear()
    if v.dtype == object: out[k] = v.astype('unicode')
out = out.persist()
out.to_zarr(str(T_FINAL), mode='w', zarr_format=2)
timings['aggregation'] = time.time() - t0
print(f'  done: {timings["aggregation"]:.1f}s')


# verify
print('\n--- verify ---')
verify = xr.open_zarr(str(T_FINAL))
print(f'  dims: {dict(verify.sizes)}')
n_nonnull = int(verify.costs.notnull().sum())
n_total = int(verify.costs.size)
pct = 100 * n_nonnull / n_total if n_total > 0 else 0
print(f'  non-null: {n_nonnull}/{n_total} ({pct:.1f}%)')

refa_check = xr.open_zarr(str(T_REFA))
refa_nondefault = int((refa_check.refA != 0).sum())
print(f'  refa non-zero values: {refa_nondefault} (should be > 0)')

if n_nonnull > 0 and refa_nondefault > 0:
    print('  pipeline ran end-to-end with real data.')
elif n_nonnull > 0 and refa_nondefault == 0:
    print('  WARNING: output has values but refa was all zeros. pyciam may have failed silently.')
else:
    print('  output is empty, check errors above.')


# results
n_workers_test = max(len(client.scheduler_info()['workers']), 1)
print('\n' + '=' * 50)
print('timings')
print('=' * 50)
total_test = sum(timings.values())
for step, secs in timings.items():
    pct = secs / total_test * 100 if total_test > 0 else 0
    print(f'  {step:<25} {secs:>8.1f}s  ({pct:>4.0f}%)')
print(f'  {"total":<25} {total_test:>8.1f}s')
print(f'  workers: {n_workers_test}')

scale_scen_wf = (F_N_SCENARIOS * F_N_WORKFLOWS) / (len(T_SCENARIOS) * len(T_WORKFLOWS))
scale_seg = total_seg_ir / T_N_SEGMENTS
scale_samp = F_N_SAMPLES / T_N_SAMP
parallel_multiplier = scale_seg * scale_samp
test_worker_hours = (total_test * n_workers_test) / 3600
slr_time = timings.get('slr_local_global', 0) + timings.get('slr_vlm', 0) + timings.get('slr_combine_save', 0)
pyciam_time = total_test - slr_time
est_slr_wh = (slr_time * scale_scen_wf * min(200, F_N_WORKERS)) / 3600
est_pyciam_wh = (pyciam_time * n_workers_test * parallel_multiplier) / 3600
est_total_wh = est_slr_wh + est_pyciam_wh

print('\n' + '=' * 50)
print('scaling estimate')
print('=' * 50)
print(f'  slr scales: {scale_scen_wf:.0f}x')
print(f'  pyciam scales: {parallel_multiplier:.0f}x')
print(f'  test used: {test_worker_hours:.1f} worker-hours')
print(f'  full run: {est_total_wh:.0f} worker-hours')
print(f'  cost multiplier: {est_total_wh / max(test_worker_hours, 0.01):.0f}x')
print(f'  wall time with 40 workers:  {est_total_wh / 40:.1f}h')
print(f'  wall time with 200 workers: {est_total_wh / 200:.1f}h')
print(f'  wall time with 800 workers: {est_total_wh / 800:.1f}h')

report = {
    'test': {
        'scenarios': T_SCENARIOS, 'workflows': T_WORKFLOWS,
        'n_samples': T_N_SAMP, 'n_segments': T_N_SEGMENTS,
        'total_seg_ir': total_seg_ir, 'n_workers': n_workers_test,
        'timings_seconds': timings, 'total_seconds': total_test,
        'worker_hours': test_worker_hours,
    },
    'full_run_estimate': {
        'worker_hours': est_total_wh,
        'cost_multiplier': est_total_wh / max(test_worker_hours, 0.01),
        'wall_hours_40w': est_total_wh / 40,
        'wall_hours_200w': est_total_wh / 200,
        'wall_hours_800w': est_total_wh / 800,
    }
}
report_path = Path('/home/jovyan/inequality-dscim-coastal/inequality/test_run_report.json')
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2, default=str)
print(f'\nreport saved to {report_path}')

client.close()
cluster.close()
print('\ndone.')