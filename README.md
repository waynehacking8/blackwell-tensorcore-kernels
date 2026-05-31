# Blackwell Tensor Core Kernels

Hand-written CUDA GEMM kernels targeting **Tensor Cores**, benchmarked on both **Hopper
(H100, sm_90)** and **Blackwell (RTX Pro 6000, sm_120)**, with a path toward Blackwell's
FP8/FP4 microscaling formats.

Extends my earlier kernel work (tiled GEMM, WMMA Tensor Core matmul, Nsight profiling) onto
the newest architecture. The point is to connect kernel-level choices — tiling, Tensor Core
fragment shapes, occupancy — to measured TFLOP/s as a fraction of the cuBLAS ceiling, on real
Hopper and Blackwell silicon.

## What this is
- A naive baseline, a shared-memory tiled GEMM, and a **WMMA Tensor Core** GEMM (FP16 accumulate FP32).
- A harness that checks correctness against cuBLAS and reports **TFLOP/s and % of the cuBLAS ceiling**.
- Builds for **sm_90 (H100)** and **sm_120 (Blackwell RTX Pro 6000)** so the same kernels are
  profiled across two generations.

## What this is NOT
- Not a cuBLAS replacement — cuBLAS is the ceiling we measure against, honestly.
- Not yet FP4 — Blackwell 5th-gen Tensor Core FP4/MXFP8 (tcgen05) is the documented frontier in
  the roadmap, built on top of the working FP16 WMMA kernel.

## Hardware
- NVIDIA RTX Pro 6000 (Blackwell, sm_120) and/or H100 (Hopper, sm_90). CUDA 12.x+.

## Layout
```
src/gemm_naive.cu      # one-thread-per-output baseline (correctness anchor)
src/gemm_tiled.cu      # shared-memory tiled + register-blocked SIMT GEMM
src/gemm_wmma.cu       # WMMA 16x16x16 Tensor Core GEMM (FP16 in, FP32 acc)
src/reference.cu       # cuBLAS GEMM — the performance ceiling
src/main.cu            # correctness + benchmark driver -> results/bench.csv
include/util.cuh       # timing, init, max-abs-error check
CMakeLists.txt         # builds for sm_90 and sm_120
docs/design-decisions.md
docs/roadmap.md
```

## Quick start (on the Blackwell / H100 box)

One command does the full capture — size sweep + Nsight + plots + report:

```bash
make capture            # build -> sweep sizes -> ncu/nsys -> results/report.md + PNGs
```

The benchmark CSV records the **device and SM** of each run, so to compare both
generations you run the sweep **once on each GPU** and the rows accumulate:

```bash
# on the H100 box:           ARCH=90  make bench
# on the Blackwell box:      ARCH=120 make bench
make analyze                 # merge both -> results/report.md + results/tflops_sm*.png
```

Single run / one kernel, by hand:

```bash
make build
./build/gemm_bench 4096 4096 4096 results/bench.csv   # M N K [out_csv]
bash scripts/profile.sh 4096                           # ncu + nsys for gemm_wmma
```

## Results
`make capture` produces, in `results/`:

- `bench.csv` — per kernel × size × GPU: ms, TFLOP/s, **% of cuBLAS ceiling**, max abs err.
- `tflops_sm90.png` / `tflops_sm120.png` — throughput vs size (Hopper vs Blackwell).
- `ncu_wmma_*.ncu-rep`, `nsys_*.nsys-rep` — Nsight Compute / Systems captures.
- `report.md` — the summary table + charts.

**(populated after running on the GPUs)**
