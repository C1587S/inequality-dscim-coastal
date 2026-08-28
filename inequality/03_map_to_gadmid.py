"""
Map pyCIAM output from impact_region (IR_XXXXX) to gadmid (integer) format.
Also renames IAMs to match the old output format.

Usage:
    python 03_map_to_gadmid.py

Input:  gs://impactlab-data/gcp/outputs/coastal/pyCIAM_outputs_inequality_1000_ssp234_v2.zarr
Output: gs://impactlab-data/gcp/outputs/coastal/pyCIAM_outputs_inequality_1000_ssp234_v2_gadmid.zarr
"""

import xarray as xr
import numpy as np

INPUT = 'gs://impactlab-data/gcp/outputs/coastal/pyCIAM_outputs_inequality_1000_ssp234_v2.zarr'
OUTPUT = 'gs://impactlab-data/gcp/outputs/coastal/pyCIAM_outputs_inequality_1000_ssp234_v2_gadmid.zarr'

IAM_RENAME = {
    'IIASA': 'IIASA GDP',
    'OECD': 'OECD Env-Growth',
}

print(f'reading {INPUT}')
ds = xr.open_zarr(INPUT)
print(f'  dims: {dict(ds.sizes)}')

# impact_region (IR_XXXXX) -> gadmid (integer)
# the number after IR_ is the GADM admin boundary ID
gadmid_vals = np.array([int(str(ir).split('_')[1]) for ir in ds.impact_region.values])
ds_gadm = ds.rename({'impact_region': 'gadmid'})
ds_gadm['gadmid'] = gadmid_vals
print(f'  mapped {len(gadmid_vals)} impact_regions to gadmid integers')

# rename IAMs to match old format
ds_gadm['iam'] = [IAM_RENAME.get(str(i), str(i)) for i in ds_gadm.iam.values]
print(f'  iams: {list(ds_gadm.iam.values)}')

# clean encodings for zarr v2 compatibility
for v in ds_gadm.data_vars:
    ds_gadm[v].encoding.clear()
for k, v in ds_gadm.coords.items():
    v.encoding.clear()
    if v.dtype == object:
        ds_gadm[k] = v.astype('unicode')

# save as zarr v2 (compatible with older zarr/xarray on RCC)
ds_gadm.to_zarr(OUTPUT, mode='w', zarr_format=2)
print(f'saved to {OUTPUT}')
print(f'  dims: {dict(ds_gadm.sizes)}')
print(f'  scenarios: {list(ds_gadm.scenario.values)}')
print('done.')
