"""
Input loading shared by the run stages (03-05).
"""

import numpy as np
import xarray as xr

from config import SEG_VAR, SLIIDERS_IR, TEST_N_SEGS


def load_ciam_in(scenario, test=False):
    """Economic inputs at IR level, subset to the model years around the
    output years. Used for segment lists and output-template coordinates."""
    ds = xr.open_zarr(str(SLIIDERS_IR[scenario]), chunks=None)
    ds = ds.sel(year=np.concatenate((np.arange(2040, 2060), np.arange(2080, 2100))))
    if test:
        ds = ds.sel({SEG_VAR: ds[SEG_VAR].values[:TEST_N_SEGS]})
    return ds


def chunked(values, size):
    return [values[i : i + size] for i in range(0, len(values), size)]
