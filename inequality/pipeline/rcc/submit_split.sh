#!/bin/bash
# Submit one stage's array split across cil and caslake with non-overlapping
# element ranges, so nobody does range arithmetic by hand and no two
# elements can ever address the same chunks.
#
#   ./submit_split.sh SCENARIO TOTAL [STAGE]
#   ./submit_split.sh glocal 500        # stage 04, elements 0-499
#   ./submit_split.sh glocal 500 05     # same split for stage 05
#
# cil gets the first CIL_FRAC of the range, throttled to CIL_CONC concurrent
# elements (80% of its 128 cores; at 3G each that also stays under 80% of
# its memory). caslake takes the rest under its own account. N_SHARDS is
# set to TOTAL for both, so the partition of the work is the same one
# regardless of who runs which element. Ranges only decide who does what;
# a wrong CIL_FRAC just means one side finishes first and the probe's
# gap-fill list covers the difference on the next wave.

set -euo pipefail
SCENARIO=${1:?usage: submit_split.sh SCENARIO TOTAL [STAGE]}
TOTAL=${2:?usage: submit_split.sh SCENARIO TOTAL [STAGE]}
STAGE=${3:-04}

source rcc_env.sh
CIL_FRAC=${CIL_FRAC:-0.30}
CIL_CONC=${CIL_CONC:-102}
CIL_ACCOUNT=${CIL_ACCOUNT:-cil}
CASLAKE_ACCOUNT=${CASLAKE_ACCOUNT:-pi-mgreenst}

CIL_N=$(python3 -c "print(max(1, min($TOTAL - 1, round($TOTAL * $CIL_FRAC))))")
LAST=$((TOTAL - 1))

echo "cil:     elements 0-$((CIL_N - 1)) of $TOTAL, throttle %$CIL_CONC, account $CIL_ACCOUNT"
echo "caslake: elements $CIL_N-$LAST of $TOTAL, account $CASLAKE_ACCOUNT"

sbatch --partition=cil --account="$CIL_ACCOUNT" \
    --array="0-$((CIL_N - 1))%$CIL_CONC" \
    --export=ALL,SCENARIO="$SCENARIO",N_SHARDS="$TOTAL" \
    "${STAGE}_array.sbatch"

sbatch --account="$CASLAKE_ACCOUNT" \
    --array="$CIL_N-$LAST" \
    --export=ALL,SCENARIO="$SCENARIO",N_SHARDS="$TOTAL" \
    "${STAGE}_array.sbatch"
