#!/usr/bin/env bash
# Third-party reference baselines on this box (roadmap Phase 4).
#
# Runs each project's OWN benchmark, unmodified, and captures raw output + clock state:
#   deepgemm  DeepSeek DeepGEMM      - FP8 GEMM, JIT     (published ~1358 TFLOPS on H800)
#   fa3       FlashAttention-3       - FP16 attention    (published 740 TFLOPS = 75% of peak)
#   tk        ThunderKittens         - H100 GEMM kernels (published FP8 ~1500 TFLOPS)
#
# These are REFERENCE rows: measured-on-our-box vs published. The gap between those two
# columns (clock state, shared box, version drift) is itself a finding. They are NOT
# this repo's kernels and are clearly attributed as third-party.
#
# Usage (on the GPU box):  GPU=6 bash run_reference_benches.sh [deepgemm|fa3|tk|all]
set -euo pipefail

IMG=nvcr.io/nvidia/pytorch:26.02-py3
GPU="${GPU:-6}"
WORK="${WORK:-/home/user/sa-portfolio/blackwell-ref}"
OUT="$WORK/results"
WHAT="${1:-all}"
mkdir -p "$WORK" "$OUT"

clock_state() { # tag
  nvidia-smi --query-gpu=index,name,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,power.limit \
    --format=csv -i "$GPU" > "$OUT/$1.clocks.csv"
}

# ---------------------------------------------------------------- DeepGEMM (FP8, JIT)
run_deepgemm() {
  echo "== DeepGEMM =="
  [ -d "$WORK/DeepGEMM" ] || git clone --recursive --depth 1 https://github.com/deepseek-ai/DeepGEMM "$WORK/DeepGEMM"
  clock_state deepgemm
  docker run --rm --gpus "\"device=$GPU\"" --shm-size=8g -v "$WORK:/work" "$IMG" bash -c '
    set -e
    cd /work/DeepGEMM
    # DeepGEMM is JIT-based: install registers the python package; kernels compile at run time.
    ./install.sh > /work/results/deepgemm.install.log 2>&1 || pip install -e . >> /work/results/deepgemm.install.log 2>&1
    ls tests/
    # their own test/benchmark entry points (print us + TFLOPS per shape);
    # test_fp8_fp4.py carries the FP8 rows = the published ~1358 TFLOPS reference
    for t in tests/test_core.py tests/test_bf16.py tests/test_fp8_fp4.py tests/test_fp8.py; do
      [ -f "$t" ] && { echo "--- $t ---"; python3 "$t" 2>&1; }
    done
    chown -R 1000:1000 /work/results /work/DeepGEMM 2>/dev/null || true
  ' > "$OUT/deepgemm.txt" 2>&1 || echo "DEEPGEMM FAILED (see results/deepgemm.txt)"
  grep -iE "TFLOPS|GB/s" "$OUT/deepgemm.txt" | tail -20 || true
}

# ----------------------------------------------------- FlashAttention-3 (FP16, hopper/)
run_fa3() {
  echo "== FlashAttention-3 =="
  [ -d "$WORK/flash-attention" ] || git clone --depth 1 https://github.com/Dao-AILab/flash-attention "$WORK/flash-attention"
  # FA3 compiles against the cutlass submodule (csrc/cutlass) - a depth-1 clone leaves it empty
  git -C "$WORK/flash-attention" submodule update --init --depth 1 csrc/cutlass
  clock_state fa3
  docker run --rm --gpus "\"device=$GPU\"" --shm-size=16g -v "$WORK:/work" "$IMG" bash -c '
    set -e
    cd /work/flash-attention/hopper
    # forward pass only + sm90a only keeps the build tractable; cap parallel jobs (shared box)
    export FLASH_ATTENTION_DISABLE_BACKWARD=TRUE NVCC_THREADS=4 MAX_JOBS=16
    pip install -e . --no-build-isolation > /work/results/fa3.build.log 2>&1
    # their own benchmark (prints TFLOPS per seqlen/headdim config)
    python3 benchmark_attn.py 2>&1 | tail -120
    chown -R 1000:1000 /work/results /work/flash-attention 2>/dev/null || true
  ' > "$OUT/fa3.txt" 2>&1 || echo "FA3 FAILED (see results/fa3.txt)"
  grep -iE "TFLOPS|it/s|ms" "$OUT/fa3.txt" | tail -20 || true
}

# -------------------------------------------------------------- ThunderKittens (GEMM)
run_tk() {
  echo "== ThunderKittens =="
  [ -d "$WORK/ThunderKittens" ] || git clone --depth 1 https://github.com/HazyResearch/ThunderKittens "$WORK/ThunderKittens"
  clock_state tk
  docker run --rm --gpus "\"device=$GPU\"" --shm-size=8g -v "$WORK:/work" "$IMG" bash -c '
    cd /work/ThunderKittens
    export THUNDERKITTENS_ROOT=/work/ThunderKittens
    # each H100 GEMM kernel ships its own Makefile + self-benchmarking harness
    for d in kernels/gemm/bf16_h100 kernels/gemm/fp8_h100 kernels/gemm/fp8_h100_scaled; do
      [ -d "$d" ] || continue
      echo "--- $d ---"; cd "/work/ThunderKittens/$d"
      make -j16 2>&1 | tail -8
      # the harness binary name varies; run whatever the Makefile produced
      for exe in $(find . -maxdepth 1 -type f -executable -not -name "*.cu" -not -name Makefile); do
        echo "run $exe"; "$exe" 2>&1 | tail -60
      done
      cd /work/ThunderKittens
    done
    chown -R 1000:1000 /work/results /work/ThunderKittens 2>/dev/null || true
  ' > "$OUT/tk.txt" 2>&1 || echo "TK FAILED (see results/tk.txt)"
  grep -iE "TFLOPS|GFLOPS|efficiency" "$OUT/tk.txt" | tail -20 || true
}

case "$WHAT" in
  deepgemm) run_deepgemm ;;
  fa3)      run_fa3 ;;
  tk)       run_tk ;;
  all)      run_deepgemm; run_tk; run_fa3 ;;
  *) echo "unknown: $WHAT"; exit 1 ;;
esac
echo "DONE -> $OUT"
