"""
CONCLUDED (Sep 2026). Full record of the investigation this script closed:

An apparent 50x cost jump of the global scenario (v3) over the published
glocal store (v2) at optimalfixed, tlim3.0, SSP2, IIASA, 2090 was an
artifact of comparing a complete store against incomplete ones. The v2
stores lost most of their optimalfixed cells to write races and refA holes:
82% of (gadmid, sample) cost totals are exactly zero in both v2 stores
against 15% legitimate zeros in the race-free v3, and the raw ratio fell
from 11x to 2.5x on cells nonzero in both. Sea level was then ruled out
(test_fingerprint_rcc.py: Spearman 0.02 against the imposed SLR change,
ratio 1.9x where sea level falls), income was never in play (both stores
use global rho), and this script ruled on what remained: the ratio is
1.82x on ncc_ar6, which contains no climate signal at all, against
1.90-2.02x on the warming scenarios, with no warming gradient and matching
costtype shares (protection 33% against 31%, so case selection is clean).
A ratio that survives the removal of all climate content measures store
completeness, not scenario physics. fulladapt v2 is far worse off than
glocal v2: 10-16x below v3 on common support with per-gadmid quartiles
reaching 17,000, far beyond any income effect, matching its larger hole
counts.

Therefore no v2-vs-v3 comparison informs scenario effects, and scenario
numbers exist once fulladapt and glocal have v3 reruns, compared within v3.

Find where the uniform ~2x gap between global_v3 and the v2 stores comes
from, given that sea level doesn't explain it (Spearman 0.02 against dslr,
ratio 1.9x where sea level falls).

Three discriminating measurements, one pass over the stores:

1. ncc_ar6. The no-climate-change scenario uses only the VLM series, which
   is identical in the base and variant SLR stores, and both stores use
   global rho. If the common-support ratio on ncc_ar6 is also ~2x, the gap
   is v2-vs-v3 machinery (case selection, refA, code paths), not the
   climate swap. If it is ~1, the gap enters through the tlim climate
   draws, pointing at the rebuilt SLR store or the variant's rank-summed
   construction.

2. The tlim gradient. Ratios for every scenario in the stores. Machinery
   bias is flat across warming levels; an SLR-linked effect grows with
   them.

3. Costtype shares. glocal_v2's case selection ran with noAdaptation's
   segment npv deflated by skipna over holes, so it likely over-chose
   noAdaptation. An over-chooser shows a depressed protection share in its
   optimalfixed mix. If shares match between stores and only the level
   differs, selection is innocent and the gap is a level effect.

Also compares fulladapt_v2 against global_v3 the same way: if both v2
stores sit the same distance below v3 on every scenario, the story is v2
versus v3, not scenario physics.

All per-gadmid statistics are means over samples nonzero in both stores of
the pair, so the v2 holes drop out. Announces read sizes with progress.

Run:
  source activate /project/cil/home_dirs/rcc/envs/python_r_general/
  python -u diagnose_uniform_gap_rcc.py
"""

from pathlib import Path

# ---- EDIT HERE to run elsewhere: store paths and the slice under test -----
DIR = "/project/cil/gcp/inequality/coastal"
STORES = {
    "fulladapt_v2": f"{DIR}/pyCIAM_outputs_inequality_1000_ssp234_v2_gadmid_coastal.zarr",
    "glocal_v2": f"{DIR}/pyCIAM_outputs_inequality_1000_ssp234_c0.23_global.zarr",
    "global_v3": f"{DIR}/pyCIAM_outputs_inequality_1000_ssp234_v3_global_income_climate_gadmid_coastal.zarr",
}
SEL = dict(year=2090, ssp="SSP2")
MIN_COMMON_SAMPLES = 50
DISK_MB_PER_S = 300
# ---------------------------------------------------------------------------

import time

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

FULL_DIMS = ("gadmid", "sample", "costtype", "scenario")


def pick_case(ds):
    cases = [str(c) for c in ds.case.values] if "case" in ds.dims else []
    for want in ("optimalfixed", "globaladapt"):
        if want in cases:
            return want
    if len(cases) == 1:
        return cases[0]
    raise SystemExit(f"no optimalfixed-equivalent case among {cases}")


def pick_iam(da):
    iams = [str(v) for v in da.iam.values]
    for match in ("IIASA", "low"):
        hit = [v for v in iams if match in v]
        if hit:
            return hit[0]
    return iams[0]


def read_gb(var):
    total = var.dtype.itemsize
    for dim, chunk, size in zip(var.dims, var.data.chunksize, var.shape):
        total *= size if dim in FULL_DIMS else min(chunk, size)
    return total / 1e9


def pair_stats(ta, tb):
    """Common-support ratio sum(mean_b)/sum(mean_a) and per-gadmid log-ratio
    spread, means over samples nonzero in both."""
    m = (ta > 0) & (tb > 0)
    enough = m.sum("sample") >= MIN_COMMON_SAMPLES
    mean_a = ta.where(m).mean("sample").where(enough)
    mean_b = tb.where(m).mean("sample").where(enough)
    ratio = float(mean_b.sum()) / float(mean_a.sum())
    logr = np.log(mean_b / mean_a).values
    logr = logr[np.isfinite(logr)]
    return ratio, int(enough.sum()), np.exp(np.percentile(logr, [25, 50, 75]))


def main():
    opened = {name: xr.open_zarr(path) for name, path in STORES.items()}
    total_gb = sum(read_gb(ds.costs) for ds in opened.values())
    print(
        f"total read ~{total_gb:.0f} GB from disk; at ~{DISK_MB_PER_S} MB/s "
        f"expect ~{total_gb * 1000 / DISK_MB_PER_S / 60:.0f} min\n",
        flush=True,
    )

    slices = {}
    for name, ds in opened.items():
        case = pick_case(ds)
        da = ds.costs.sel(case=case, drop=True).sel(**SEL, drop=True)
        iam = pick_iam(da)
        print(f"{name}: case='{case}', iam='{iam}', reading ~{read_gb(ds.costs):.1f} GB:", flush=True)
        t0 = time.time()
        with ProgressBar():
            slices[name] = da.sel(iam=iam, drop=True).load()
        print(f"{name}: loaded in {(time.time() - t0) / 60:.1f} min\n", flush=True)

    gadmids = None
    scens = None
    for da in slices.values():
        g = set(int(x) for x in da.gadmid.values)
        s = set(str(x) for x in da.scenario.values)
        gadmids = g if gadmids is None else gadmids & g
        scens = s if scens is None else scens & s
    gadmids = sorted(gadmids)
    scens = sorted(scens)
    print(f"shared gadmids: {len(gadmids)}, shared scenarios: {scens}")
    slices = {k: v.sel(gadmid=gadmids, scenario=scens) for k, v in slices.items()}
    totals = {k: v.sum("costtype") for k, v in slices.items()}

    pairs = [("glocal_v2", "global_v3"), ("fulladapt_v2", "global_v3"),
             ("fulladapt_v2", "glocal_v2")]
    print(f"\ncommon-support ratios (b/a), with per-gadmid ratio quartiles:")
    print(f"{'scenario':>10} | " + " | ".join(f"{b}/{a}" for a, b in pairs))
    for scen in scens:
        cells = []
        for a, b in pairs:
            r, n, q = pair_stats(
                totals[a].sel(scenario=scen), totals[b].sel(scenario=scen)
            )
            cells.append(f"{r:.2f}x [{q[0]:.2f},{q[1]:.2f},{q[2]:.2f}] n={n}")
        print(f"{scen:>10} | " + " | ".join(cells))
    print(
        "read: ncc_ar6 ratio ~= tlim ratios => v2-vs-v3 machinery, scenario "
        "innocent;\nncc ~1 with tlim ~2 => the gap enters through the "
        "climate draws;\nratios growing with warming => SLR-amplitude-linked"
    )

    print("\ncosttype shares of optimalfixed on the pair's common support:")
    for scen in ("tlim3.0", "ncc_ar6"):
        if scen not in scens:
            continue
        ta = totals["glocal_v2"].sel(scenario=scen)
        tb = totals["global_v3"].sel(scenario=scen)
        m = (ta > 0) & (tb > 0)
        print(f"  {scen}:")
        for name in ("glocal_v2", "global_v3"):
            comp = slices[name].sel(scenario=scen).where(m).mean("sample").sum("gadmid")
            comp = comp / comp.sum()
            parts = ", ".join(
                f"{str(ct)} {float(v):.0%}" for ct, v in zip(comp.costtype.values, comp.values)
            )
            print(f"    {name:14s} {parts}")
    print(
        "read: depressed protection share in glocal_v2 => its case selection "
        "over-chose\nnoAdaptation (skipna-deflated candidate); matching "
        "shares => selection innocent,\nthe gap is a level effect"
    )


if __name__ == "__main__":
    main()
