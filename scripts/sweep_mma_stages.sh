#!/usr/bin/env bash
# Pipeline-depth sweep for the mma.sync kernels (roadmap Phase 2 "3/4-stage pipeline
# sweep" read-out). Runs mma_pipe and mma_warptile at 2/3/4 cp.async stages and
# records every row into results/mma_stage_sweep.csv with the stage count appended
# to the kernel name (e.g. mma_warptile@s2), so the winner is auditable raw data.
#   SIZE=8192 bash scripts/sweep_mma_stages.sh
set -euo pipefail
SIZE="${SIZE:-8192}"
OUT=results/mma_stage_sweep.csv
TMP=results/_sweep_tmp.csv
mkdir -p results
rm -f "$OUT" "$TMP"
for S in 2 3 4; do
  echo ">> MMA_STAGES=$S  M=N=K=$SIZE"
  rm -f "$TMP"
  MMA_STAGES=$S ./build/gemm_bench "$SIZE" "$SIZE" "$SIZE" "$TMP" "mma_pipe,mma_warptile"
  if [[ ! -f "$OUT" ]]; then head -1 "$TMP" > "$OUT"; fi
  tail -n +2 "$TMP" | sed "s/,mma_pipe,/,mma_pipe@s${S},/; s/,mma_warptile,/,mma_warptile@s${S},/" >> "$OUT"
done
rm -f "$TMP"
echo ">> wrote $OUT"
