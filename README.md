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

**(populated after running on the GPUs)**

> ### On the two cuBLAS baselines (read before quoting any "% of cuBLAS")
> The originally reported percentages (e.g. **907.5% at 512**, and **"WMMA ≈ 72.6% of cuBLAS"**
> at 8192) are **% of FP32 cuBLAS** — the WMMA kernel runs **FP16 on Tensor Cores** while
> `cublasSgemm` runs **FP32 on CUDA cores**. That is a **precision-mismatched** comparison and is
> **not** the Tensor Core ceiling. A `>100%` row does **not** mean a hand-written kernel beats
> cuBLAS; it reflects the FP16-TC vs FP32-CUDA-core mismatch (plus launch overhead at small sizes).
>
> A same-precision Tensor Core baseline — **`cublas_tc` (`cublasGemmEx`, FP16 in / FP32 accumulate)**,
> in `src/cublas_tc.cu` — has been added to the harness with identical timing methodology (FP16 cast
> staged once outside the timed loop, same as `gemm_wmma.cu`). The report now also computes
> **% of cuBLAS-TC**. **Those numbers are pending a re-run on the sm_120 box**; against this honest
> ceiling the WMMA `%` is expected to drop substantially, because `cublasGemmEx` also uses Tensor Cores.

## References
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) — the production reference for Tensor Core GEMM.
- [WMMA API (CUDA C++ Programming Guide)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma) — the API used by `gemm_wmma.cu`.

## Disclaimer
Personal project for learning and benchmarking. Views and results are my own and do not represent any employer.
