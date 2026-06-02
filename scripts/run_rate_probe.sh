#!/usr/bin/env bash
# FP32-accumulate tensor rate probe (roadmap Phase 4): is FP16-in/FP32-acc full-rate
# or half-rate vs FP16-in/FP16-acc on this card?
#
# Runs the register-resident mma.sync microbenchmark with a background clock/power
# log (same convention as the GEMM sessions). Outputs:
#   results/mma_rate_probe.csv          per-rep TFLOP/s + clocks + ratio summary
#   results/clock_state_rate_probe.txt  500 ms clock/power/throttle samples
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results

if [[ ! -x build/mma_rate_probe ]]; then
  cmake -B build -DCMAKE_CUDA_ARCHITECTURES="${ARCH:-120a}" >/dev/null
  cmake --build build -j --target mma_rate_probe >/dev/null
fi

nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active \
  --format=csv,noheader -lms 500 > results/clock_state_rate_probe.txt &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT
sleep 1

./build/mma_rate_probe results/mma_rate_probe.csv

sleep 1
echo ">> results/mma_rate_probe.csv + results/clock_state_rate_probe.txt"
