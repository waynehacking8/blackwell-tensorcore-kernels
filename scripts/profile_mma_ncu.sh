#!/usr/bin/env bash
# Nsight Compute side-by-side capture for the Phase 2 mma.sync analysis (sm_120).
#
# This box restricts GPU performance counters to root (RmProfilingAdminOnly=1), so the
# capture runs inside a CUDA container with --cap-add=SYS_ADMIN (same approach as the
# Phase 1 H100 captures). Needs docker + nvidia runtime; no sudo required for members
# of the docker group.
#   bash scripts/profile_mma_ncu.sh
#
# Captures, at M=N=K=8192, one kernel instance each of:
#   * gemm_wmma_t    (Phase 1 kernel — expected top stall: MIO queue full / smem feed)
#   * gemm_mma_t     (Phase 2 final mma_warptile kernel)
#   * cutlass_80_... (the kernel cuBLAS-TC dispatches on sm_120)
# and exports the full details page (throughput, stall reasons, occupancy) to text.
set -euo pipefail
cd "$(dirname "$0")/.."
SIZE="${SIZE:-8192}"
IMAGE="${IMAGE:-nvidia/cuda:12.9.1-devel-ubuntu24.04}"
mkdir -p results

# NOTE: the container must run as root — CAP_SYS_ADMIN (which is what unlocks the
# perf counters under RmProfilingAdminOnly=1) only applies to root processes. Files
# are chown'd back to the host user at the end.
docker run --rm --gpus all --cap-add=SYS_ADMIN \
  -v "$(pwd)":/work -w /work \
  -e HOST_UIDGID="$(id -u):$(id -g)" \
  "$IMAGE" bash -c '
    set -euo pipefail
    export HOME=/tmp
    NCU=$(ls /opt/nvidia/nsight-compute/*/ncu /usr/local/cuda/bin/ncu 2>/dev/null | head -1)
    echo ">> using $NCU"
    run_capture() {  # $1=kernel regex  $2=bench filter  $3=output basename
      echo ">> capturing $3 (kernel regex: $1, M=N=K='"$SIZE"')"
      # --kernel-name-base demangled: lets the regex match the full template name
      # (needed for cuBLAS, whose kernel is cutlass::Kernel2<cutlass_80_tensorop_...> —
      # the bare function name is just "Kernel2", which would also match the FP32
      # reference kernel cutlass::Kernel2<cutlass_80_simt_sgemm_...> that runs first)
      "$NCU" --set full --kernel-name-base demangled -k "regex:$1" -c 1 -f -o "results/$3" \
          ./build/gemm_bench '"$SIZE $SIZE $SIZE"' /tmp/_ncu_bench.csv "$2"
      "$NCU" --import "results/$3.ncu-rep" --page details > "results/$3.txt"
      echo "   -> results/$3.ncu-rep + results/$3.txt"
    }
    run_capture "gemm_wmma"  "wmma"         "ncu_sm120_wmma_'"$SIZE"'"
    run_capture "gemm_mma_t" "mma_warptile" "ncu_sm120_mma_warptile_'"$SIZE"'"
    run_capture "tensorop"   "cublas_tc"    "ncu_sm120_cublastc_'"$SIZE"'"
    chown "$HOST_UIDGID" results/ncu_sm120_*
  '
echo ">> done. Text summaries: results/ncu_sm120_{wmma,mma_warptile,cublastc}_${SIZE}.txt"
