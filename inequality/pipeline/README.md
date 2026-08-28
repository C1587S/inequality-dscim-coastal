# Inequality pyCIAM pipeline

Coastal damages for the inequality decomposition, in three scenarios.

| scenario | income (rho) | sea level |
|---|---|---|
| `fulladapt` | local | local |
| `glocal` | global average | local |
| `global` | global average | global climate signal plus local land motion |

Income enters pyCIAM as `rho`, which SLIIDERS builds from income per capita
normalised against US income in 2000. It scales storm damages through
`surge * (1 - rho)`. Replacing it with the population-weighted global average
gives everyone the same wealth while leaving the sea level each segment faces
alone.

For `global` the sea level is homogenised too. Stage 1 writes an SLR variant
where `lsl_msl05 = gsl_msl05 + lsl_ncc_msl05`: the global mean climate signal
on top of each site's own vertical land motion. `gsl` has no land motion in it
and `lsl_ncc` is land motion and nothing else, so the sum swaps out the local
climate signal without double counting. Stage 4 reads that store like any
other, so no pyCIAM change is needed.

## Running it

One script per stage. Stages 1 to 3 run once and serve all three scenarios;
stages 4 to 6 take `--scenario`. Every stage takes `--test` for a small
end-to-end check at 5 samples and 50 segments.

Run one stage at a time. Starting a stage stops any cluster already running,
including another stage's.

```bash
# once, shared by all three scenarios
nohup python -u 01_process_slr.py > 01.log 2>&1 &
python -u 02_prepare_inputs.py
nohup python -u 03_refa.py > 03.log 2>&1 &

# then repeat this block for fulladapt, glocal and global
S=fulladapt
nohup python -u 04_calc_cases.py --scenario $S > 04_$S.log 2>&1 &
nohup python -u 05_optimize.py   --scenario $S > 05_$S.log 2>&1 &
python -u 06_aggregate.py --scenario $S
```

| stage | what it does | time |
|---|---|---|
| `01_process_slr.py` | AR6 draws to 1,000 rank-quantile samples; writes the base and global-climate SLR stores | 2-6 h |
| `02_prepare_inputs.py` | SLIIDERS variants and the segment-level collapse | minutes |
| `03_refa.py` | reference adaptation (refA): the protection height each segment already has | 1.3 h |
| `04_calc_cases.py` | costs for every adaptation case | 19-24 h |
| `05_optimize.py` | optimal case per segment | 1 h |
| `06_aggregate.py` | impact_region store plus the gadmid and coastal views | 30 min |

Every stage is restartable. Rerun one after a crash and it fills the gaps
rather than starting over. Stage 3 won't finish while any refA cell is null,
and stage 6 won't aggregate over an incomplete input, so a partial run can't
reach the output as zeros.

Paths and parameters live in `config.py`, cluster machinery in `runner.py`.

## Output

Each scenario produces a store at impact_region resolution, plus two views:
one mapped to gadmid, and one subset to the coastal gadmids only.

Dimensions are case, costtype, impact_region, scenario, sample, year, ssp and
iam. Cases are `noAdaptation` and `optimalfixed`. Cost types are wetland,
inundation, relocation, protection, stormCapital and stormPopulation.
Scenarios are `ncc_ar6` plus the five temperature limits.

## Notes

- **One refA serves all three scenarios.** The segment-level collapse recomputes
  rho from ypcc rather than carrying it through, so every SLIIDERS variant
  collapses to the same segment store and refA comes out the same. Rho reaches
  the model only through the cost calculation, which reads the impact-region
  store directly. This also matches the intent: present-day adaptation heights
  reflect the income history a place actually had.

- **The collapse drops variables it doesn't explicitly handle.** Any `seg_ir`
  variable not on its list is dropped, and `prep_sliiders` rebuilds it
  downstream from ypcc. `rho`, `vsl` and `ypc` all work this way. A scenario
  built by modifying any of them reaches the cost calculation but not refA; one
  built by modifying ypcc changes all of them at once.

- **The 1,000 SLR samples are rank-matched, not draw-matched.** Each series is
  quantiled independently with the same seed and bins, so sample n is the
  rank-n quantile of that series rather than a single FACTS draw. Worth
  carrying into any description of the scenarios.