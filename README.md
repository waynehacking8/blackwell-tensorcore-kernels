# Blackwell Tensor Core Kernels

Hand-written CUDA GEMM kernels targeting **Tensor Cores**, benchmarked on both **Hopper
(H100, sm_90)** and **Blackwell (RTX Pro 6000, sm_120)**, with a path toward Blackwell's
FP8/FP4 microscaling formats.

The point is to connect kernel-level choices — tiling, Tensor Core
fragment shapes, occupancy — to measured TFLOP/s as a fraction of the cuBLAS ceiling, on real
Hopper and Blackwell silicon.

## What this is
- A naive baseline, a shared-memory tiled GEMM, and a **WMMA Tensor Core** GEMM (FP16 accumulate FP32).
- A harness that checks correctness against cuBLAS and reports **TFLOP/s and % of the cuBLAS ceiling**.
  Two cuBLAS baselines: **`cublas`** = `cublasSgemm` (FP32, CUDA cores) and **`cublas_tc`** =
  `cublasGemmEx` (FP16 in / FP32 accumulate, Tensor Cores — the same precision as the WMMA kernel).
- Builds for **sm_90 (H100)** and **sm_120 (Blackwell RTX Pro 6000)** so the same kernels are
  profiled across two generations.

## What this is NOT
- Not a cuBLAS replacement — cuBLAS is the ceiling measured against, honestly.
- Not yet FP4 — Blackwell 5th-gen Tensor Core FP4/MXFP8 (tcgen05) is the documented frontier in
  the roadmap, built on top of the working FP16 WMMA kernel.

## Hardware
- NVIDIA RTX Pro 6000 (Blackwell, sm_120) and/or H100 (Hopper, sm_90). CUDA 12.x+.

## Layout
```
src/gemm_naive.cu      # one-thread-per-output baseline (correctness anchor)
src/gemm_tiled.cu      # shared-memory tiled + register-blocked SIMT GEMM
src/gemm_wmma.cu       # WMMA 16x16x16 Tensor Core GEMM (FP16 in, FP32 acc)
src/reference.cu       # cuBLAS FP32 ceiling (cublasSgemm, CUDA cores)
src/cublas_tc.cu       # cuBLAS Tensor Core ceiling (cublasGemmEx, FP16 in / FP32 acc) — same precision as wmma
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

- `bench.csv` — per kernel × size × GPU: ms, TFLOP/s, **% of FP32 cuBLAS** (`pct_of_cublas`,
  vs `cublasSgemm`), max abs err. Now includes a `cublas_tc` row per size.
- `tflops_sm90.png` / `tflops_sm120.png` — throughput vs size (Hopper vs Blackwell).
- `ncu_wmma_*.ncu-rep`, `nsys_*.nsys-rep` — Nsight Compute / Systems captures.
- `report.md` — the summary table (now with both **% of FP32 cuBLAS** and **% of cuBLAS-TC**) + charts.

### Measured on RTX PRO 6000 Blackwell Max-Q (sm_120, CUDA 12.8)

The `gemm_wmma` kernel is shared-memory tiled with per-warp 2×2 register tiling and a 3-stage
`cp.async` pipeline, **size-dispatched**: 64×64 tile for N<1536 (better occupancy), 128×128 tile
for N≥1536 (more reuse). TFLOP/s and its fraction of **both** baselines:

| size | wmma | tile | cublas_tc (TC) | cublas (FP32) | wmma % of FP32 cuBLAS | **wmma % of cuBLAS-TC** | cublas_tc speedup |
|---|---|---|---|---|---|---|---|
| 512  | 16.08 | 64×64   | 33.80  | 1.49  | 1097% | 47.6% | 21.3× |
| 1024 | 38.72 | 64×64   | 123.27 | 9.80  | 371%  | 31.4% | 14.3× |
| 2048 | 68.64 | 128×128 | 215.66 | 37.57 | 151%  | 32.0% | 5.7× |
| 4096 | 96.57 | 128×128 | 238.95 | 52.89 | 118%  | 40.7% | 4.5× |
| 8192 | 102.7 | 128×128 | 228.40 | 54.42 | 188%  | **45.0%** | 4.2× |

Read the **% of cuBLAS-TC** column — the honest same-precision (FP16-in/FP32-acc, Tensor Core)
ceiling. Across two optimization passes (shared-mem + cp.async, then register tiling + deeper
pipeline + size dispatch) the WMMA kernel went from a naive **17.3%** to **45.0%** of cuBLAS-TC
at 8192 (1.64× the single-buffer version), and no longer decays at scale. Pipeline depth is
tuned (3 stages > 4 > 5) and **warp specialization was tried but did not beat the multi-stage
pipeline** — see `results/nsys_profile.md` for the full before/after and the WS experiment.
The **% of FP32 cuBLAS** column is precision-mismatched (kept for continuity); its `>100%` rows
are FP16-TC vs FP32-CUDA-core, not the kernel beating cuBLAS. `cublas_tc` is 4.2–21× faster than
`cublasSgemm`, confirming the Tensor Core path.

> ### On the two cuBLAS baselines (read before quoting any "% of cuBLAS")
> **% of FP32 cuBLAS** (e.g. 1097% @ 512, 114% @ 8192) compares **FP16-on-Tensor-Cores WMMA** against
> **`cublasSgemm` FP32 on CUDA cores** — precision-mismatched, **not** a Tensor Core ceiling; a `>100%`
> row reflects that mismatch (plus small-size launch overhead), not a kernel beating cuBLAS.
> **% of cuBLAS-TC** compares against **`cublas_tc` (`cublasGemmEx`, FP16 in / FP32 accumulate)** in
> `src/cublas_tc.cu` — same precision, same timing methodology (FP16 cast staged once outside the timed
> loop; cuBLAS handle created once). Against this honest ceiling the optimized WMMA lands at ~45% at
> large sizes; the remaining gap is cuBLAS's larger 128×64 CTA tile and warp-specialized/register-level
> scheduling that a WMMA-fragment kernel can't fully match.

## References
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) — the production reference for Tensor Core GEMM.
- [WMMA API (CUDA C++ Programming Guide)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma) — the API used by `gemm_wmma.cu`.

## Disclaimer
Personal project for learning and benchmarking. Views and results are my own and do not represent any employer.
