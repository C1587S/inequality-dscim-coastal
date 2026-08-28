# Inequality pyCIAM pipeline

Produces the coastal damage outputs for the inequality decomposition, under
three scenarios keyed by name throughout (`--scenario` on stages 4-6):

| scenario | income (rho) | sea level | published as |
|---|---|---|---|
| `fulladapt` | local | local | `..._v2.zarr` (cases `optimalfixed` + `noAdaptation`) |
| `glocal` | global average (0.161) | local | `..._v2_globaladapt.zarr` — historically mislabelled "global" |
| `global` | global average | global-mean climate signal + local VLM | `..._v2_global_income_climate.zarr` (new) |

The `global` scenario needs no pyCIAM changes: stage 1 writes a variant SLR
store in which `lsl_msl05 := gsl_msl05 + lsl_ncc_msl05`, and stage 4 simply
reads that store. `gsl` carries no vertical land motion and `lsl_ncc` carries
nothing but vertical land motion, so the sum homogenises the climate signal
while keeping local geophysics, with no double counting.

## Stages

Each stage is one script, restartable (completed work is skipped on rerun),
with `--test` for a small end-to-end check (5 samples, 50 segments, one tlim
scenario, consistent across stages).

1. `01_process_slr.py` — AR6 draws to 1,000 rank-quantile samples; writes the
   base SLR store and the globalclimate variant. Seeded, so a rebuild
   reproduces the store used in the published runs.
2. `02_prepare_inputs.py` — SLIIDERS variants (names normalised to `K_2019` /
   `pop_2019`; global-rho version for glocal/global) and the shared seg-level
   collapse. Minutes, no cluster.
3. `03_refa.py` — present-day adaptation heights. Runs once, no `--scenario`:
   see "shared refA" below. ~1.3 h.
4. `04_calc_cases.py --scenario X` — costs for all adaptation cases. The long
   stage, ~19-24 h at 600 workers. Rerun to gap-fill after a crash.
5. `05_optimize.py --scenario X` — optimal case per segment. ~1 h.
6. `06_aggregate.py --scenario X` — filter, aggregate to impact_region, IAM
   rename, then the gadmid and gadmid-coastal views. Never overwrites an
   existing store unless `--overwrite`.

Paths and parameters live in `config.py`; cluster machinery and batched
submission in `runner.py`.

## Why refA is shared by all scenarios — UNDER VERIFICATION

`collapse_econ_inputs_to_seg` recomputes rho from ypcc at the segment level
(`pyCIAM/utils.py:268`), discarding the rho variable of whichever IR store it
is fed. Since the rho treatments leave ypcc untouched, every variant collapses
to an identical seg store — and refA, computed from it, should be identical
too. If so, this reproduces the published runs (the glocal run's "regenerate
refA with modified SLIIDERS" step silently regenerated the fulladapt refA) and
matches the original design intent that present-day adaptation reflects local
income. The rho variants reach the model only through the main cost
calculation, which reads rho from the IR store via `prep_sliiders`.

**Status: the code says this, but the data has not yet confirmed it.** A
comparison of the two published outputs (`checks/verify_shared_refa.py`) found
non-storm noAdaptation costs only ~79% exact-equal where the shared-refA
hypothesis predicts ~100%. Until `checks/diagnose_refa_mismatch.py` explains
the discrepancy, do not rely on refA being shared, and treat the single
`PATH_REFA` design in `config.py` as provisional.

## Findings about pyCIAM's input pipeline

These are findings about pyCIAM's data flow, not about our scenarios. They
matter to anyone building a scenario by modifying SLIIDERS variables.

**The collapse silently drops and regenerates variables.**
`collapse_econ_inputs_to_seg` carries only an explicit list of variables to
the seg level (sums, firsts, weighted averages) and passes through variables
without a `seg_ir` dimension. Anything else — any `seg_ir`-dimensioned
variable not on the list — is dropped, and `prep_sliiders` regenerates it
downstream from ypcc. In the current SLIIDERS store, three variables fall in
this trap:

- `rho` — recomputed from ypcc in the collapse itself (`utils.py:268`)
- `vsl` — dims `(year, ssp, seg_ir, iam)`: dropped by the collapse,
  recomputed by `prep_sliiders` from ypcc via the elasticity formula
  (`io.py:100-110`)
- `ypc` — dims `(ssp, iam, year, seg_ir)`: dropped, recomputed from ypcc and
  population density

Consequence: a modification to any of these in the IR store reaches the main
cost calculation (which loads the IR store directly) but never reaches refA
(which loads the seg store). ypcc is the root regenerator — modify it and
rho, `ref_income`, vsl, and ypc all regenerate from it at the seg level.
Modifying rho, as the glocal scenario does, is the safe case precisely
because it only needed to reach the main calculation.

**Every published run carries two different VSL series.** The IR store's
`vsl` was hand-curated (`create-inputs.ipynb`, "Update VSL"), and the main
cost calculation uses it. But because the collapse drops it, refA was always
computed with the formula-derived VSL instead — the curated values never
reached the refA optimisation. This is identical across scenarios, so it
cancels in the decomposition, but it is unlikely to be what the curation
intended, and it belongs on any list of pyCIAM issues to fix upstream.

**The 1,000 SLR samples are rank-matched, not draw-matched.** The local,
global, and VLM series are quantiled independently (same seed, same bins), so
sample n is the rank-n quantile of each series, not one physical FACTS draw.
This applies to the published stores and the rebuilt ones equally; carry the
caveat into any description of the scenarios.

## Provenance of the published stores

The legacy scripts that produced the existing zarrs are preserved in
`../legacy/` — see the README there for the script-to-store mapping.
