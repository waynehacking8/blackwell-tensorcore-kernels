#!/usr/bin/env bash
# Nsight Compute (kernel-level) + Nsight Systems (timeline) capture for the WMMA
# Tensor Core kernel. Produces results/ncu_wmma_<M>.ncu-rep and results/nsys_<M>.nsys-rep.
#   bash scripts/profile.sh 4096
set -euo pipefail
M="${1:-4096}"
mkdir -p results
TMP=results/_profile_tmp.csv
echo ">> Nsight Compute (gemm_wmma)"
ncu --set full -k "regex:gemm_wmma" -c 1 -f -o "results/ncu_wmma_${M}" \
    ./build/gemm_bench "$M" "$M" "$M" "$TMP" || echo "   (ncu unavailable — skipped)"
echo ">> Nsight Systems (full timeline)"
nsys profile -f true -o "results/nsys_${M}" \
    ./build/gemm_bench "$M" "$M" "$M" "$TMP" || echo "   (nsys unavailable — skipped)"
rm -f "$TMP"
echo ">> results/ncu_wmma_${M}.ncu-rep , results/nsys_${M}.nsys-rep"
