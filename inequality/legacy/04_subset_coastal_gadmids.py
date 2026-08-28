"""
Subset pyCIAM output to only gadmids present in both old and new outputs.

The updated SLIIDERS maps coastal segments to ~1,500 additional admin regions
that touch inland water bodies (lakes, rivers) but not the ocean. These regions
have zero or negligible coastal damages (91% have costs=0, total = 1.5% of global).

This script removes those inland regions by keeping only the 5,903 gadmids that
appear in both the original (Ian's) and updated output.

Input:  gs://impactlab-data/gcp/outputs/coastal/pyCIAM_outputs_inequality_1000_ssp234_v2_gadmid.zarr
Output: gs://impactlab-data/gcp/outputs/coastal/pyCIAM_outputs_inequality_1000_ssp234_v2_gadmid_coastal.zarr

Also copy to RCC:
    gsutil -m cp -r gs://...coastal.zarr /project/cil/gcp/inequality/coastal/
"""

import xarray as xr
import numpy as np

INPUT = 'gs://impactlab-data/gcp/outputs/coastal/pyCIAM_outputs_inequality_1000_ssp234_v2_gadmid.zarr'
OLD_REF = 'gs://impactlab-data/gcp/outputs/coastal/pyCIAM_outputs_inequality_1000_ssp234.zarr'
OUTPUT = 'gs://impactlab-data/gcp/outputs/coastal/pyCIAM_outputs_inequality_1000_ssp234_v2_gadmid_coastal.zarr'

print(f'loading new: {INPUT}')
new_ds = xr.open_zarr(INPUT)

print(f'loading old ref: {OLD_REF}')
old_ds = xr.open_zarr(OLD_REF)
old_gadmids = set(int(g) for g in old_ds.gadmid.values)

# subset to gadmids present in old output (removes inland water body regions)
shared_mask = np.array([int(g) in old_gadmids for g in new_ds.gadmid.values])
ds_coastal = new_ds.isel(gadmid=shared_mask)

print(f'  original: {len(new_ds.gadmid)} gadmids')
print(f'  after filter: {len(ds_coastal.gadmid)} gadmids')
print(f'  removed: {len(new_ds.gadmid) - len(ds_coastal.gadmid)} inland water body regions')

# clean encodings for zarr v2
for v in ds_coastal.data_vars:
    ds_coastal[v].encoding.clear()
for k, v in ds_coastal.coords.items():
    v.encoding.clear()
    if v.dtype == object:
        ds_coastal[k] = v.astype('unicode')

ds_coastal.to_zarr(OUTPUT, mode='w', zarr_format=2)
print(f'saved to {OUTPUT}')
print(f'dims: {dict(ds_coastal.sizes)}')
print('done.')
