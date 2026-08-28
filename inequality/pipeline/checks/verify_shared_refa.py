"""
Test, against the two published outputs, that the glocal ("globaladapt") run
used the same refA as the fulladapt run.

Logic: rho enters pyCIAM in exactly one place, surge * (1 - rho), so it can
only touch the two storm cost types. In the noAdaptation case every other
cost type is driven by SLR, the socioeconomic inputs, and refA — all of which
the two runs shared, IF the claim is right that the seg-level collapse
discarded the glocal rho modification and reproduced the fulladapt refA.

Expected if the claim holds:
- noAdaptation non-storm costs: identical between the two stores (exact, or
  within float noise)
- noAdaptation storm costs: different, glocal > fulladapt in rich countries
  (local rho above the global average of ~0.161 means (1 - rho) rises when
  replaced) and glocal < fulladapt in poor ones

If non-storm costs diverge, the glocal refA responded to global rho after all
and the shared-refA design in config.py is wrong.

Reads ~35 GB from the two stores; run on the hub.
  python -u checks/verify_shared_refa.py
"""

import pandas as pd
import xarray as xr

DIR = "gs://impactlab-data/gcp/outputs/coastal"
PATH_FULL = f"{DIR}/pyCIAM_outputs_inequality_1000_ssp234_v2.zarr"
PATH_GLOC = f"{DIR}/pyCIAM_outputs_inequality_1000_ssp234_v2_globaladapt.zarr"
PATH_SLIIDERS = "gs://impactlab-data/coastal/local-scc-model/data/int/sliiders-ir.zarr"

NONSTORM = ["wetland", "inundation", "relocation", "protection"]
STORM = ["stormCapital", "stormPopulation"]
RICH = ["USA", "NOR", "QAT"]
POOR = ["COD", "BGD", "MOZ"]


def compare(full, gloc, kinds, label):
    a, b = xr.align(full.sel(costtype=kinds), gloc.sel(costtype=kinds), join="exact")
    equal = ((a == b) | (a.isnull() & b.isnull())).sum().compute()
    maxdiff = abs(a - b).max().compute()
    n = a.size
    print(f"{label}:")
    print(f"  exact-equal: {int(equal)}/{n} ({100 * int(equal) / n:.2f}%)")
    print(f"  max |diff|: {float(maxdiff):.6g}")


def storm_direction(full, gloc):
    """Country-level ratio of glocal to fulladapt storm costs."""
    sl = xr.open_zarr(PATH_SLIIDERS, chunks=None)
    ir_country = (
        pd.DataFrame(
            {"ir": sl.impact_region.values, "country": sl.seg_country.values}
        )
        .groupby("ir")
        .country.first()
    )
    f = full.sel(costtype=STORM, scenario="tlim2.0", year=2090, ssp="SSP2").isel(
        iam=0, drop=True
    ).sum("costtype")
    g = gloc.sel(costtype=STORM, scenario="tlim2.0", year=2090, ssp="SSP2").isel(
        iam=0, drop=True
    ).sum("costtype")
    print("storm cost ratio glocal/fulladapt (tlim2.0, 2090, SSP2, median sample):")
    for country in RICH + POOR:
        irs = list(ir_country.index[ir_country == country])
        fc = f.sel(impact_region=irs).sum("impact_region").median("sample").compute()
        gc = g.sel(impact_region=irs).sum("impact_region").median("sample").compute()
        expect = ">1" if country in RICH else "<1"
        print(f"  {country}: {float(gc) / float(fc):.3f} (expect {expect})")


def main():
    full = xr.open_zarr(PATH_FULL).costs.sel(case="noAdaptation")
    gloc = xr.open_zarr(PATH_GLOC).costs.sel(case="noAdaptation")
    compare(full, gloc, NONSTORM, "non-storm noAdaptation costs (refA-driven)")
    compare(full, gloc, STORM, "storm noAdaptation costs (rho-driven)")
    storm_direction(full, gloc)


if __name__ == "__main__":
    main()
