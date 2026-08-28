"""
full production run of pyciam inequality pipeline.
run: nohup python -u 00_full_run.py > full_run.log 2>&1 &
check: tail -f full_run.log
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

import inspect
print(f"pyciam: {Path(inspect.getfile(get_refA)).parent}")
print(f"get_refA sig: {inspect.signature(get_refA)}")

timings = {}


# production parameters (matching original notebook)

SCENARIOS = TLIM_SCENARIOS          # ['tlim1.5', 'tlim2.0', 'tlim3.0', 'tlim4.0', 'tlim5.0']
PROD_WORKFLOWS = WORKFLOWS          # ['wf_1f', 'wf_2f']
N_SAMP_PER_WF = N_SAMPLES_PER_WORKFLOW  # 500
N_SAMP = N_SAMPLES_TOTAL            # 1000
PROD_SEG_CHUNKSIZE = 2              # from original notebook
PROD_SAMPLE_CHUNKSIZE = 100         # from original notebook
PROD_REFA_SEG_CHUNKSIZE = 15        # from original notebook
PROD_N_WORKERS = 800
PROD_N_WORKERS_MIN = 40

# output paths (production)
PROD_SLR = PATH_SLR_INEQUALITY
PROD_SLIIDERS_SEG = PATH_SLIIDERS_SEG
PROD_REFA = PATH_REFA_INEQUALITY
PROD_TMP = PATH_OUTPUT_TMP
PROD_INTERMEDIATE = DIR_SCRATCH / 'pyciam-inequality-intermediate.zarr'
PROD_FINAL = PATH_OUTPUT_FINAL

print(f'scenarios: {SCENARIOS}')
print(f'workflows: {PROD_WORKFLOWS}')
print(f'samples: {N_SAMP} ({N_SAMP_PER_WF} per workflow)')
print(f'output: {PROD_FINAL}')


# cluster

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
    idle_timeout=7200,  # 2 hours idle timeout for production
    profile='micro',
    **(dict(worker_image=img, scheduler_image=img) if img else {})
)
client = cluster.get_client()
cluster.scale(PROD_N_WORKERS_MIN)  # start with 40, scale up later

start = time.time()
deadline = start + 600  # 10 min timeout for production
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
    UploadDirectory(str(Path(pyCIAM.__file__).parent), update_path=True, restart_workers=False),
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


# slr processing

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
low = np.arange(0, nsamps, step=nsamps / N_SAMP_PER_WF)
high = low + nsamps / N_SAMP_PER_WF
quants = np.random.randint(low=low, high=high, size=None) / nsamps

local_dfs, global_dfs = [], []
for tlim, wf in product(SCENARIOS, PROD_WORKFLOWS):
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
low_v = np.arange(0, nsamps, step=nsamps / N_SAMP)
high_v = low_v + nsamps / N_SAMP
quants_v = np.random.randint(low=low_v, high=high_v, size=None) / nsamps
vlm_df = open_and_convert_zarr(mapping)
vlm_df = (vlm_df.sel(years=slice(2020, 2100), drop=True)
          .sea_level_change.chunk(dict(samples=-1))
          .quantile(quants_v, dim='samples')
          .rename({'quantile': 'samples'})
          .assign_coords({'samples': np.arange(1, N_SAMP + 1)})
          .to_dataset())
timings['slr_vlm'] = time.time() - t0
print(f'  done: {timings["slr_vlm"]:.1f}s')


print('\n--- combine + save slr ---')
t0 = time.time()
df_full = xr.combine_by_coords(local_dfs)
df_full = df_full.stack(samples=['workflow', 'quantile'])
df_full = df_full.reset_index('samples').drop_vars(['workflow', 'quantile'])
df_full['samples'] = np.arange(1, N_SAMP + 1)

global_ds = xr.combine_by_coords(global_dfs)
global_ds = global_ds.stack(samples=['workflow', 'quantile'])
global_ds = global_ds.reset_index('samples').drop_vars(['workflow', 'quantile'])
global_ds['samples'] = np.arange(1, N_SAMP + 1)
global_ds = global_ds.squeeze(drop=True).sea_level_change

vlm_df['samples'] = np.arange(1, N_SAMP + 1)

all_ds = xr.Dataset({
    'lsl_msl05': df_full.sea_level_change,
    'lsl_ncc_msl05': vlm_df.sea_level_change,
    'gsl_msl05': global_ds,
    'lon': vlm_df.lon, 'lat': df_full.lat,
})
all_ds['lon'] = all_ds.lon.where(all_ds.lon != -180, 180)
all_ds = all_ds.stack(locations=['lat', 'lon'])

# for production this is big, use distributed scheduler
# (test used synchronous because it was small)
all_ds = all_ds.persist()

reduce_dims = [d for d in ['tlim', 'samples', 'years'] if d in all_ds.dims]
valid = (all_ds[['lsl_msl05', 'lsl_ncc_msl05']]
         .sel(years=slice(2100)).notnull().all(reduce_dims)
         .to_array('tmp').all('tmp')).compute()
all_ds = all_ds.sel(locations=valid.where(valid, drop=True).locations)

rename_map = {}
if 'years' in all_ds.dims: rename_map['years'] = 'year'
if 'samples' in all_ds.dims: rename_map['samples'] = 'sample'
if 'locations' in all_ds.dims: rename_map['locations'] = 'site_id'
if 'tlim' in all_ds.dims: rename_map['tlim'] = 'scenario'
all_ds = all_ds.rename(rename_map)

n_sites = len(all_ds.site_id)
all_ds = all_ds.chunk({'site_id': -1, 'scenario': 1, 'year': -1, 'sample': 100})

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
all_ds.to_zarr(str(PROD_SLR), mode='w', zarr_format=2)
timings['slr_combine_save'] = time.time() - t0
print(f'  done: {timings["slr_combine_save"]:.1f}s, {n_sites} sites')


# pyciam setup
params = pd.read_json(PATH_PARAMS)['values']
quantiles = np.arange(1, N_SAMP + 1)
econ_input_path = str(PATH_SLIIDERS)
slr_input_paths = [PROD_SLR]
slr_names = ['ar6']
surge_input_paths = {k: AnyPath(v) for k, v in PATHS_SURGE_LOOKUP.items()}
mc_dim = MC_DIM
storage_options = STORAGE_OPTIONS
diaz_inputs = False
eps = 1
model_kwargs = {}


# patch sliiders if needed
print('\n--- checking sliiders ---')
sliiders_check = xr.open_zarr(econ_input_path, chunks=None)
sliiders_vars = list(sliiders_check.data_vars)
needs_rename = 'K_2019' not in sliiders_vars and 'K_2014' in sliiders_vars
if needs_rename:
    print('  patching K_2014 -> K_2019')
    rename_dict = {}
    if 'K_2014' in sliiders_vars: rename_dict['K_2014'] = 'K_2019'
    if 'pop_2014' in sliiders_vars: rename_dict['pop_2014'] = 'pop_2019'
    patched = sliiders_check.rename(rename_dict)
    patched_path = DIR_SCRATCH / 'sliiders-patched-prod.zarr'
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
    econ_input_path_for_collapse, PROD_SLIIDERS_SEG,
    seg_var_subset=None, output_chunksize=100,
    storage_options=storage_options, seg_var=SEG_VAR,
)
timings['collapse_sliiders'] = time.time() - t0
print(f'  done: {timings["collapse_sliiders"]:.1f}s')


# load all econ inputs (no subsetting for production)
ciam_in = subset_econ_inputs(
    xr.open_zarr(econ_input_path, chunks=None, storage_options=storage_options),
    SEG_VAR, seg_var_subset=None,
)
ciam_in = ciam_in.sel(year=np.concatenate((np.arange(2040, 2060), np.arange(2080, 2100))))
total_seg_ir = len(ciam_in[SEG_VAR])
print(f'  seg_ir: {total_seg_ir}')


# output template
slr_test = xr.open_zarr(str(PROD_SLR), chunks=None)
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
out_ds.to_zarr(str(PROD_TMP), compute=False, mode='w', storage_options=storage_options, zarr_format=2)
print(f'  template: {dict(out_ds.sizes)}')


# scale up cluster before heavy computation
print('\n--- scaling cluster ---')
cluster.scale(PROD_N_WORKERS)
time.sleep(30)
n_workers = len(client.scheduler_info()['workers'])
print(f'  workers: {n_workers}')

# reinstall on new workers
ensure_packages()
time.sleep(10)
n_workers = len(client.scheduler_info()['workers'])
print(f'  after ensure_packages: {n_workers} workers')


# refa
print('\n--- refa ---')
segs = np.unique(ciam_in.seg)
samps = np.arange(1, N_SAMP + 1)
n_segs = len(segs)

# refa zarr template
xr.DataArray(
    coords=dict(
        sample=samps,
        seg=segs,
        case=(['sample', 'seg'], np.ones((len(samps), n_segs), dtype=np.int8))
    ),
    dims=('sample', 'seg')
).to_dataset(name='refA').chunk(
    {'seg': PROD_REFA_SEG_CHUNKSIZE, 'sample': 200}
).to_zarr(str(PROD_REFA), storage_options=storage_options, mode='w', zarr_format=2)
print(f'  refa template: (sample={len(samps)}, seg={n_segs})')

ensure_packages()
t0 = time.time()
seg_grps_refa = [segs[i:i + PROD_REFA_SEG_CHUNKSIZE] for i in range(0, len(segs), PROD_REFA_SEG_CHUNKSIZE)]
samp_grps = [samps[i:i + PROD_SAMPLE_CHUNKSIZE] for i in range(0, len(samps), PROD_SAMPLE_CHUNKSIZE)]
grps_refa = list(product(seg_grps_refa, samp_grps))
print(f'  refa groups: {len(grps_refa)}')

refa_futs = client.map(
    run_get_refA, grps_refa,
    output_path=str(PROD_REFA), econ_input_path=str(PROD_SLIIDERS_SEG),
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
    print(f'  WARNING: {n_err} refa errors. continuing anyway.')


# calc_all_cases
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
grps_ciam = list(product(groups, samp_grps))
print(f'  groups: {len(grps_ciam)}')

ciam_futs = list(client.map(
    run_calc_all_cases, grps_ciam,
    params=params, econ_input_path=econ_input_path,
    slr_input_paths=slr_input_paths, slr_names=slr_names,
    output_path=str(PROD_TMP), refA_path=str(PROD_REFA),
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
    print(f'  WARNING: {n_err} calc errors. continuing anyway.')


# optimization (4 groups of 250 samples, matching original)
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
print(f'  optimization tasks: {len(precurser_futs)}')

opt_futs = precurser_futs.reset_index(drop=False).apply(
    lambda row: client.submit(
        run_optimize_case, row[SEG_VAR], *row.group_id,
        quantiles=sample_ids[row['samples']],
        econ_input_path=econ_input_path,
        output_path=str(PROD_TMP), seg_var=SEG_VAR,
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

refa_check = xr.open_zarr(str(PROD_REFA))
refa_nondefault = int((refa_check.refA != 0).sum())
print(f'  refa non-zero: {refa_nondefault}')

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
    hrs = secs / 3600
    pct = secs / total * 100 if total > 0 else 0
    print(f'  {step:<25} {secs:>10.1f}s  {hrs:>6.2f}h  ({pct:>4.0f}%)')
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
report_path = Path('/home/jovyan/inequality-dscim-coastal/inequality/full_run_report.json')
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2, default=str)
print(f'\nreport: {report_path}')
print(f'output: {PROD_FINAL}')

client.close()
cluster.close()
print('\ndone.')
