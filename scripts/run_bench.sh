#!/usr/bin/env bash
# Build (if needed) and sweep GEMM sizes, appending to results/bench.csv.
# Run this ONCE per GPU you want to capture — the CSV records device+sm, so a
# run on the H100 box and a run on the Blackwell box accumulate into one file.
#   ARCH=90      bash scripts/run_bench.sh   # build only for the local GPU (faster)
#   SIZES="2048 4096 8192" bash scripts/run_bench.sh
set -euo pipefail
ARCH="${ARCH:-90;120}"
SIZES="${SIZES:-512 1024 2048 4096 8192}"
mkdir -p results
if [[ ! -x build/gemm_bench ]]; then
  cmake -B build -DCMAKE_CUDA_ARCHITECTURES="$ARCH" >/dev/null
  cmake --build build -j >/dev/null
fi
for S in $SIZES; do
  echo ">> M=N=K=$S"
  ./build/gemm_bench "$S" "$S" "$S" results/bench.csv
done
echo ">> appended to results/bench.csv"
