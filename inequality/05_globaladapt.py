"""
Run pyCIAM globaladapt for inequality analysis.

Globaladapt replaces rho (resilience to storm impacts) with a population-weighted
global average fixed at 2000-2014 levels. This means all segments have the same
storm resilience, removing local adaptation in storm vulnerability.

Steps:
1. Create modified SLIIDERS with global average rho
2. Collapse modified SLIIDERS to seg level
3. Generate new refA with modified SLIIDERS
4. Run calc_all_cases (~25h)
5. Optimization + aggregation
6. Save final output

Based on Ian's run-all-adaptation-scens.ipynb approach.

Test mode:  python -u 05_globaladapt.py --test
Full run:   nohup python -u 05_globaladapt.py > globaladapt.log 2>&1 &
Monitor:    tail -f globaladapt.log
"""

import argparse
import time, json, subprocess, sys, importlib, os
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--force-reinstall', '--no-deps',
    '/home/jovyan/inequality-dscim-coastal/pyciam'])
importlib.invalidate_caches()

import numpy as np, pandas as pd, xarray as xr, dask
from collections import OrderedDict
from itertools import product
from pathlib import Path
from cloudpathlib import AnyPath

dask.config.set({"array.rechunk.method": "tasks"})
from config import *
from pyCIAM.constants import CASES, COSTTYPES
from pyCIAM.io import create_template_dataarray
from pyCIAM.run import calc_all_cases, get_refA, optimize_case
from pyCIAM.utils import add_attrs_to_result, collapse_econ_inputs_to_seg, subset_econ_inputs

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', help='test mode: 50 segments, 5 samples')
args = parser.parse_args()

TEST_MODE = args.test

if TEST_MODE:
    N_SAMP = 5
    N_TEST_SEGS = 50
    PROD_N_WORKERS = 10
    BATCH_SIZE = 50
    PROD_SEG_CHUNKSIZE = 2
    PROD_SAMPLE_CHUNKSIZE = 5
    print('=== TEST MODE: 50 segments, 5 samples ===')
else:
    N_SAMP = N_SAMPLES_TOTAL
    N_TEST_SEGS = None
    PROD_N_WORKERS = 600
    BATCH_SIZE = 300
    PROD_SEG_CHUNKSIZE = 2
    PROD_SAMPLE_CHUNKSIZE = 100

# paths for globaladapt (separate from optimalfixed run)
SLIIDERS_GLOBALADAPT = DIR_SCRATCH / 'sliiders-ir-globaladapt.zarr'
SLIIDERS_SEG_GLOBALADAPT = DIR_SCRATCH / 'sliiders-seg-globaladapt.zarr'
REFA_GLOBALADAPT = DIR_SCRATCH / 'refA-inequality-globaladapt.zarr'
PROD_SLR = PATH_SLR_INEQUALITY
PROD_TMP = DIR_SCRATCH / 'pyciam-inequality-tmp-globaladapt.zarr'
PROD_INTERMEDIATE = DIR_SCRATCH / 'pyciam-inequality-intermediate-globaladapt.zarr'

if TEST_MODE:
    PROD_FINAL = AnyPath('gs://impactlab-data/gcp/outputs/coastal/pyCIAM_outputs_inequality_globaladapt_TEST.zarr')
else:
    PROD_FINAL = AnyPath('gs://impactlab-data/gcp/outputs/coastal/pyCIAM_outputs_inequality_1000_ssp234_v2_globaladapt.zarr')

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
    cluster = gateway.new_cluster(idle_timeout=7200, profile='micro',
        **(dict(worker_image=img, scheduler_image=img) if img else {}))
    client = cluster.get_client()
    cluster.scale(PROD_N_WORKERS)
    start = time.time()
    while len(client.scheduler_info()['workers']) < 3 and time.time() - start < 600:
        time.sleep(5)
    n = len(client.scheduler_info()['workers'])
    print(f'  cluster: {n} workers, dashboard: {client.dashboard_link}')
    import pyCIAM
    client.register_plugin(
        UploadDirectory(str(Path(pyCIAM.__file__).parent), update_path=True, restart_workers=False),
        name='pyciam-upload')
    client.run(lambda: __import__('subprocess').check_call(['pip', 'install', '-q', 'cloudpathlib']))
    return client, cluster

def ensure_cluster():
    global client, cluster
    try: client.scheduler_info()
    except:
        print('  reconnecting...')
        client, cluster = create_cluster()
    return client

def run_calc_all_cases(grp, **kwargs):
    from pyCIAM.run import calc_all_cases
    return calc_all_cases(grp, **kwargs)

def run_optimize_case(*args, **kwargs):
    from pyCIAM.run import optimize_case
    return optimize_case(*args, **kwargs)


timings = {}

# ============================================================
# STEP 1: Create modified SLIIDERS with global average rho
# ============================================================
print('\n--- step 1: modify SLIIDERS (global rho) ---')
t0 = time.time()

# skip if already done
try:
    _check = xr.open_zarr(str(SLIIDERS_GLOBALADAPT), chunks=None)
    if 'K_2019' in _check.data_vars:
        print(f'  already exists at {SLIIDERS_GLOBALADAPT}, skipping')
        timings['modify_sliiders'] = 0
    else:
        raise FileNotFoundError
except:
    sliiders = xr.open_zarr(str(PATH_SLIIDERS), chunks=None)

    # population by country for weighting
    pop_by_country = sliiders.pop_2014.sum('elev').groupby(sliiders.seg_country).sum().load()

    # compute global population-weighted average rho (2000-2014 mean)
    rho_mean = sliiders.rho.sel(year=slice(2000, 2014)).mean('year').load()

    # match countries
    common_countries = sorted(set(rho_mean.country.values) & set(pop_by_country.seg_country.values))
    pop_matched = pop_by_country.sel(seg_country=common_countries).rename({'seg_country': 'country'})
    rho_matched = rho_mean.sel(country=common_countries)

    global_rho = rho_matched.weighted(pop_matched).mean('country')
    print(f'  global rho (SSP2, IIASA): {global_rho.isel(ssp=0, iam=0).values:.4f}')
    print(f'  original rho range: {float(sliiders.rho.min()):.3f} - {float(sliiders.rho.max()):.3f}')

    # replace rho with global average expanded to all countries and years
    sliiders['rho'] = global_rho.expand_dims(country=sliiders.country, year=sliiders.year)
    print(f'  new rho shape: {sliiders.rho.shape}')

    # rename K_2014 -> K_2019, pop_2014 -> pop_2019 (pyCIAM expects 2019 names)
    if 'K_2014' in sliiders.data_vars and 'K_2019' not in sliiders.data_vars:
        sliiders = sliiders.rename({'K_2014': 'K_2019'})
        print('  renamed K_2014 -> K_2019')
    if 'pop_2014' in sliiders.data_vars and 'pop_2019' not in sliiders.data_vars:
        sliiders = sliiders.rename({'pop_2014': 'pop_2019'})
        print('  renamed pop_2014 -> pop_2019')

    # save modified SLIIDERS
    for v in sliiders.data_vars: sliiders[v].encoding.clear()
    for k, v in sliiders.coords.items():
        v.encoding.clear()
        if v.dtype == object or 'str' in str(v.dtype).lower() or 'string' in str(v.dtype).lower():
            sliiders[k] = xr.DataArray(np.array([str(x) for x in v.values], dtype='U'), dims=v.dims)
    sliiders.to_zarr(str(SLIIDERS_GLOBALADAPT), mode='w', storage_options=STORAGE_OPTIONS, zarr_format=2)
    print(f'  saved to {SLIIDERS_GLOBALADAPT}')

    timings['modify_sliiders'] = time.time() - t0
    print(f'  done: {timings["modify_sliiders"]/60:.0f}min')


# ============================================================
# STEP 2: Collapse modified SLIIDERS to seg level
# ============================================================
print('\n--- step 2: collapse SLIIDERS to seg ---')
t0 = time.time()

try:
    _check2 = xr.open_zarr(str(SLIIDERS_SEG_GLOBALADAPT), chunks=None)
    print(f'  already exists at {SLIIDERS_SEG_GLOBALADAPT}, skipping')
    timings['collapse_sliiders'] = 0
except:
    collapse_econ_inputs_to_seg(
        str(SLIIDERS_GLOBALADAPT),
        str(SLIIDERS_SEG_GLOBALADAPT),
        seg_var_subset=None,
        output_chunksize=100,
        seg_var=SEG_VAR,
        storage_options=STORAGE_OPTIONS,
    )
    print(f'  saved to {SLIIDERS_SEG_GLOBALADAPT}')
    timings['collapse_sliiders'] = time.time() - t0
    print(f'  done: {timings["collapse_sliiders"]/60:.0f}min')


# ============================================================
# STEP 3: Setup cluster and load inputs
# ============================================================
print('\n--- step 3: setup ---')
client, cluster = create_cluster()

params = pd.read_json(PATH_PARAMS)['values']
quantiles = np.arange(1, N_SAMP + 1)
econ_input_path = str(SLIIDERS_GLOBALADAPT)  # use MODIFIED SLIIDERS
slr_input_paths = [PROD_SLR]
slr_names = ['ar6']
surge_input_paths = {k: AnyPath(v) for k, v in PATHS_SURGE_LOOKUP.items()}

ciam_in = subset_econ_inputs(
    xr.open_zarr(econ_input_path, chunks=None, storage_options=STORAGE_OPTIONS),
    SEG_VAR, seg_var_subset=None,
)
ciam_in = ciam_in.sel(year=np.concatenate((np.arange(2040, 2060), np.arange(2080, 2100))))

if TEST_MODE:
    test_seg_ir = ciam_in[SEG_VAR].values[:N_TEST_SEGS]
    ciam_in = ciam_in.sel({SEG_VAR: test_seg_ir})
    print(f'  TEST: subset to {len(test_seg_ir)} seg_ir')

print(f'  seg_ir: {len(ciam_in[SEG_VAR])}')


# ============================================================
# STEP 4: Generate new refA with modified SLIIDERS
# ============================================================
print('\n--- step 4: generate refA ---')
t0 = time.time()

segs = np.unique(ciam_in.seg)
refa_seg_chunksize = 15
seg_grps = [segs[i:i + refa_seg_chunksize] for i in range(0, len(segs), refa_seg_chunksize)]
samp_grps_refa = [quantiles[i:i + PROD_SAMPLE_CHUNKSIZE] for i in range(0, len(quantiles), PROD_SAMPLE_CHUNKSIZE)]
refa_grps = list(product(seg_grps, samp_grps_refa))
print(f'  refA groups: {len(refa_grps)}')

from distributed import wait

def run_get_refA(grp, **kwargs):
    from pyCIAM.run import get_refA
    return get_refA(grp, **kwargs)

# create refA template (only if not already partially done)
try:
    refa_check = xr.open_zarr(str(REFA_GLOBALADAPT), chunks=None)
    refa_pct = float(refa_check.refA.notnull().mean().values) * 100
except:
    refa_pct = 0
if refa_pct < 1:
    refa_ds = xr.DataArray(
        coords=dict(
            sample=quantiles,
            seg=segs,
            case=(['sample', 'seg'], np.ones((len(quantiles), len(segs)), dtype=np.int8))
        ), dims=("sample", "seg")
    ).to_dataset(name="refA").chunk({"seg": 15, "sample": PROD_SAMPLE_CHUNKSIZE})
    refa_ds.to_zarr(str(REFA_GLOBALADAPT), storage_options=STORAGE_OPTIONS, mode="w", zarr_format=2)
    print(f'  refA template created')
else:
    print(f'  refA already {refa_pct:.1f}% done, resuming')

REFA_BATCH_SIZE = 100
total_refa = len(refa_grps)
n_ok_total = 0
n_err_total = 0
batch_idx = 0

while batch_idx * REFA_BATCH_SIZE < total_refa:
    batch_start = batch_idx * REFA_BATCH_SIZE
    batch_end = min(batch_start + REFA_BATCH_SIZE, total_refa)
    batch = refa_grps[batch_start:batch_end]
    batch_num = batch_idx + 1
    total_batches = (total_refa + REFA_BATCH_SIZE - 1) // REFA_BATCH_SIZE

    try:
        client = ensure_cluster()
        from distributed import wait
        futs = list(client.map(
            run_get_refA, batch,
            output_path=str(REFA_GLOBALADAPT),
            econ_input_path=str(SLIIDERS_SEG_GLOBALADAPT),
            slr_input_path=slr_input_paths[0],
            params=params,
            surge_input_path=surge_input_paths["seg"],
            mc_dim=MC_DIM,
            storage_options=STORAGE_OPTIONS,
            quantiles=quantiles,
            diaz_inputs=False,
            eps=1,
        ))
        wait(futs)
        n_ok = sum(1 for f in futs if f.status == 'finished')
        n_err = sum(1 for f in futs if f.status == 'error')
        n_ok_total += n_ok
        n_err_total += n_err
        elapsed = time.time() - t0
        print(f'  refA batch {batch_num}/{total_batches}: ok={n_ok} err={n_err} | {n_ok_total}/{total_refa} | {elapsed/60:.0f}min')
        if n_err > 0:
            for f in futs:
                if f.status == 'error':
                    try: f.result()
                    except Exception as e: print(f'    err: {str(e)[:200]}')
                    break
        batch_idx += 1
    except Exception as e:
        print(f'  refA batch {batch_num} failed: {str(e)[:150]}')
        print(f'  reconnecting...')
        time.sleep(30)
        try: client, cluster = create_cluster()
        except:
            time.sleep(60)
            client, cluster = create_cluster()

timings['refA'] = time.time() - t0
print(f'  done: {timings["refA"]/60:.0f}min')


# ============================================================
# STEP 5: Create template and run calc_all_cases
# ============================================================
print('\n--- step 5: create template ---')
slr_ds = xr.open_zarr(str(PROD_SLR), chunks=None)
all_scenarios = ['ncc_ar6'] + list(slr_ds.scenario.values)
print(f'  scenarios: {all_scenarios}')

coords = OrderedDict({
    'case': CASES, 'costtype': COSTTYPES,
    SEG_VAR: ciam_in[SEG_VAR].values,
    'scenario': all_scenarios,
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
out_ds.to_zarr(str(PROD_TMP), compute=False, mode='w', storage_options=STORAGE_OPTIONS, zarr_format=2)
print(f'  template: {dict(out_ds.sizes)}')

print('\n--- step 5b: calc_all_cases ---')
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

n_ok_total = 0
n_err_total = 0
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
            output_path=str(PROD_TMP), refA_path=str(REFA_GLOBALADAPT),
            surge_input_path=surge_input_paths[SEG_VAR],
            seg_var=SEG_VAR, mc_dim=MC_DIM, quantiles=quantiles,
            storage_options=STORAGE_OPTIONS, diaz_inputs=False, check=True,
        ))
        wait(futs)
        n_ok = sum(1 for f in futs if f.status == 'finished')
        n_err = sum(1 for f in futs if f.status == 'error')
        n_ok_total += n_ok
        n_err_total += n_err
        elapsed = time.time() - t0
        rate = n_ok_total / elapsed if elapsed > 0 else 0
        remaining = (total_tasks - batch_end) / rate / 3600 if rate > 0 else 0
        print(f'  batch {batch_num}/{total_batches}: ok={n_ok} err={n_err} | {n_ok_total}/{total_tasks} | {elapsed/60:.0f}min | ~{remaining:.1f}h left')
        if n_err > 0:
            for f in futs:
                if f.status == 'error':
                    try: f.result()
                    except Exception as e: print(f'    err: {str(e)[:200]}')
                    break
        batch_idx += 1
    except Exception as e:
        print(f'  batch {batch_num} failed: {str(e)[:150]}')
        print(f'  reconnecting...')
        time.sleep(30)
        try: client, cluster = create_cluster()
        except:
            time.sleep(60)
            client, cluster = create_cluster()

timings['calc_all_cases'] = time.time() - t0
print(f'  done: {timings["calc_all_cases"]/3600:.1f}h, ok={n_ok_total}, err={n_err_total}')


# ============================================================
# STEP 6: Optimization
# ============================================================
print('\n--- step 6: optimization ---')
client = ensure_cluster()
t0 = time.time()

n_opt_groups = 4
samples_per_group = N_SAMP // n_opt_groups
sample_ids = {i: np.arange(samples_per_group * i + 1, samples_per_group * (i + 1) + 1) for i in range(n_opt_groups)}
print(f'  {n_opt_groups} x {samples_per_group} samples')

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
print(f'  tasks: {total_opt}')

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
            f = client.submit(run_optimize_case, seg_var_val, *row.group_id,
                quantiles=sample_ids[row.samples],
                econ_input_path=econ_input_path,
                output_path=str(PROD_TMP), seg_var=SEG_VAR,
                eps=1, check=True, storage_options=STORAGE_OPTIONS)
            futs.append(f)
        wait(futs)
        n_ok = sum(1 for f in futs if f.status == 'finished')
        n_err = sum(1 for f in futs if f.status == 'error')
        n_ok_total += n_ok
        n_err_total += n_err
        elapsed = time.time() - t0
        rate = n_ok_total / elapsed if elapsed > 0 else 0
        remaining = (total_opt - batch_end) / rate / 3600 if rate > 0 else 0
        print(f'  batch {batch_num}/{total_batches}: ok={n_ok} err={n_err} | {n_ok_total}/{total_opt} | {elapsed/60:.0f}min | ~{remaining:.1f}h left')
        if n_err > 0:
            for f in futs:
                if f.status == 'error':
                    try: f.result()
                    except Exception as e: print(f'    err: {str(e)[:200]}')
                    break
        batch_idx += 1
    except Exception as e:
        print(f'  batch {batch_num} failed: {str(e)[:150]}')
        print(f'  reconnecting...')
        time.sleep(30)
        try: client, cluster = create_cluster()
        except:
            time.sleep(60)
            client, cluster = create_cluster()

timings['optimization'] = time.time() - t0
print(f'  done: {timings["optimization"]/3600:.1f}h, ok={n_ok_total}, err={n_err_total}')


# ============================================================
# STEP 7: Aggregation + rename IAMs
# ============================================================
print('\n--- step 7: aggregation ---')
client = ensure_cluster()
t0 = time.time()

AGG_VAR = 'impact_region'
t = xr.open_zarr(str(PROD_TMP))
t = t.sel(case=OUTPUT_CASES, ssp=OUTPUT_SSPS, year=OUTPUT_YEARS)[['costs']]
for v in t.data_vars: t[v].encoding.clear()
for k, v in t.coords.items():
    v.encoding.clear()
    if v.dtype == object or 'str' in str(v.dtype).lower() or 'string' in str(v.dtype).lower():
        t[k] = xr.DataArray(np.array([str(x) for x in v.values], dtype='U'), dims=v.dims)
t.to_zarr(str(PROD_INTERMEDIATE), mode='w', zarr_format=2)

this_chunksize = 2
out = xr.open_zarr(str(PROD_INTERMEDIATE), chunks={'case': -1, SEG_VAR: this_chunksize})
out['costs'] = out.costs.groupby(ciam_in[AGG_VAR]).sum().chunk({AGG_VAR: this_chunksize}).persist()
out = out.drop_vars(SEG_VAR).unify_chunks()

# rename IAMs
iam_rename = {'IIASA': 'IIASA GDP', 'OECD': 'OECD Env-Growth'}
current_iams = list(out.iam.values)
new_iams = [iam_rename.get(str(i), str(i)) for i in current_iams]
out['iam'] = new_iams
print(f'  iams renamed: {current_iams} -> {new_iams}')

for v in out.data_vars: out[v].encoding.clear()
for k, v in out.coords.items():
    v.encoding.clear()
    if v.dtype == object or 'str' in str(v.dtype).lower() or 'string' in str(v.dtype).lower():
        out[k] = xr.DataArray(np.array([str(x) for x in v.values], dtype='U'), dims=v.dims)
out = out.persist()
out.to_zarr(str(PROD_FINAL), storage_options=STORAGE_OPTIONS, mode='w', zarr_format=2)

timings['aggregation'] = time.time() - t0
print(f'  done: {timings["aggregation"]/60:.0f}min')


# ============================================================
# VERIFY
# ============================================================
print('\n--- verify ---')
verify = xr.open_zarr(str(PROD_FINAL))
print(f'  dims: {dict(verify.sizes)}')
print(f'  scenarios: {list(verify.scenario.values)}')
print(f'  iams: {list(verify.iam.values)}')
n_nonnull = int(verify.costs.notnull().sum())
n_total = int(verify.costs.size)
print(f'  non-null: {n_nonnull}/{n_total} ({100*n_nonnull/n_total:.1f}%)')

assert 'ncc_ar6' in verify.scenario.values, "ncc_ar6 missing!"
assert (verify.sel(case='optimalfixed', drop=True).sum(dim='costtype').costs.notnull().all()), "nulls found!"
print('  all checks passed.')

total = sum(timings.values())
print(f'\ntotal: {total/3600:.1f}h')
print(f'output: {PROD_FINAL}')

report = {'timings': timings, 'total_hours': total / 3600, 'output': str(PROD_FINAL)}
report_path = '/home/jovyan/inequality-dscim-coastal/inequality/globaladapt_report.json'
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2, default=str)
print(f'report: {report_path}')

client.close()
cluster.close()
print('done.')