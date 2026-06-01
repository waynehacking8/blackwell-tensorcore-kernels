# Blackwell Tensor Core Kernels

Hand-written CUDA GEMM kernels targeting **Tensor Cores**, benchmarked on both **Hopper
(H100, sm_90)** and **Blackwell (RTX Pro 6000, sm_120)**, with a path toward Blackwell's
FP8/FP4 microscaling formats.

The point is to connect kernel-level choices — tiling, Tensor Core
fragment shapes, occupancy — to measured TFLOP/s as a fraction of the cuBLAS ceiling, on real
Hopper and Blackwell silicon.

## What this is
- A naive baseline, a shared-memory tiled GEMM, and a **WMMA Tensor Core** GEMM (FP16 in / FP32 accumulate).
- A harness that checks correctness against cuBLAS and reports **TFLOP/s and % of the cuBLAS ceiling**,
  across a **precision ladder** of cuBLAS baselines on the same card:
  **`cublas`** = `cublasSgemm` (FP32, CUDA cores) → **`cublas_tf32`** = `cublasGemmEx` (FP32 in,
  TF32 compute, Tensor Cores) → **`cublas_tc`** = `cublasGemmEx` (FP16 in / FP32 acc, Tensor Cores —
  the same precision as the WMMA kernel, hence the honest ceiling).
- Builds for **sm_90 (H100)** and **sm_120 (Blackwell RTX Pro 6000)** so the same kernels *can* be
  profiled across two generations.

> **Measurement status:** both generations are now populated in `results/bench.csv` —
> **Blackwell RTX PRO 6000 Max-Q (sm_120)** and **H100 80GB SXM5 (sm_90)** (run on a separate
> 8×H100 box, pinned to a single idle GPU). The cross-generation comparison below is the
> headline finding of this repo: **the same WMMA kernel that reaches 45% of cuBLAS-TC on
> sm_120 reaches only 8% on H100** — not because the kernel runs slower per-SM, but because
> Hopper's Tensor Core ceiling is only reachable through `wgmma` warpgroup instructions that
> the WMMA API cannot emit.

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
- `tflops_sm120.png`, `pct_tc_sm120.png`, `roofline_sm120.png` — the three charts below.
- `ncu_wmma_*.ncu-rep`, `nsys_*.nsys-rep` — Nsight Compute / Systems captures.
- `report.md` — the summary table (both **% of FP32 cuBLAS** and **% of cuBLAS-TC**) + charts.

### Measured on RTX PRO 6000 Blackwell Max-Q (sm_120, CUDA 12.8)

**Throughput across the precision ladder (FP32 → TF32 → FP16, all real, one card):**

![GEMM throughput vs size](results/tflops_sm120.png)

**Each kernel as a fraction of the honest same-precision ceiling (cuBLAS FP16-TC = 100%):**

![% of cuBLAS-TC](results/pct_tc_sm120.png)

**Throughput at the largest (most steady-state) size, M=N=K=8192:**

![throughput bar at 8192](results/roofline_sm120.png)

The `gemm_wmma` kernel is shared-memory tiled with per-warp 2×2 register tiling and a 3-stage
`cp.async` pipeline, **size-dispatched**: 64×64 tile for N<1536 (better occupancy), 128×128 tile
for N≥1536 (more reuse). Numbers at each size (TFLOP/s and fraction of the **same-precision**
cuBLAS FP16-TC ceiling):

| size | wmma | tf32-TC | cublas_tc (FP16-TC) | **wmma % of cuBLAS-TC** | tf32 % of cuBLAS-TC |
|---|---|---|---|---|---|
| 512  | 16.2 | 28.0  | 33.3  | 48.7% | 84.0% |
| 1024 | 38.7 | 89.7  | 137.1 | 28.2% | 65.5% |
| 2048 | 68.7 | 131.3 | 215.7 | 31.8% | 60.9% |
| 4096 | 96.3 | 146.7 | 238.2 | 40.4% | 61.6% |
| 8192 | 103.5 | 152.7 | 229.2 | **45.2%** | 66.6% |

Read the **% of cuBLAS-TC** column — the honest same-precision (FP16-in/FP32-acc, Tensor Core)
ceiling. Across two optimization passes (shared-mem + cp.async, then register tiling + deeper
pipeline + size dispatch) the WMMA kernel went from a naive **17.3%** to **45.2%** of cuBLAS-TC
at 8192 (1.64× the single-buffer version), and no longer decays at scale. Pipeline depth is
tuned (3 stages > 4 > 5) and **warp specialization was tried but did not beat the multi-stage
pipeline** — the expected outcome per CudaDMA (Bauer et al., SC'11), since warp specialization
needs large tiles + async-transfer hardware + register reallocation as prerequisites, none of
which a 512-thread WMMA tile supplies on its own; see `results/nsys_profile.md` for the full
before/after and the WS experiment.
The **% of FP32 cuBLAS** column is precision-mismatched (kept for continuity); its `>100%` rows
are FP16-TC vs FP32-CUDA-core, not the kernel beating cuBLAS. `cublas_tc` is 4.2–23× faster than
`cublasSgemm`, confirming the Tensor Core path.

### Measured on H100 80GB SXM5 (sm_90, CUDA 13.1) — the cross-generation result

The same source, built with `ARCH=90`, run on one idle GPU of an 8×H100 box
(`nvcr.io/nvidia/pytorch:26.02-py3` container; the box's busy production GPU was never touched):

![GEMM throughput vs size, H100](results/tflops_sm90.png)

![% of cuBLAS-TC, H100](results/pct_tc_sm90.png)

| size | wmma (TFLOP/s) | cublas_tc (TFLOP/s) | **wmma % of cuBLAS-TC** | same kernel on sm_120 |
|---|---|---|---|---|
| 2048 | 56.6 | 555.0 | 10.2% | 31.8% |
| 4096 | 59.4 | 732.6 | 8.1% | 40.4% |
| 8192 | 60.9 | 761.7 | **8.0%** | **45.2%** |

Two things happened at once, and `nsys`/`ncu` separate them cleanly
(see `results/nsys_profile.md` for the full profiles):

1. **The ceiling moved.** On H100, `cublas_tc` dispatches **`nvjet_sm90_hss_320x128`** — a
   Hopper-native warpgroup (`wgmma`) kernel that reaches **762 TFLOP/s ≈ 77% of H100's 989
   TFLOP/s FP16 dense peak**. On the RTX Pro 6000, cuBLAS-TC tops out at 229 TFLOP/s (an
   sm_80-style `cutlass_80_tensorop` kernel). The H100 ceiling is **3.3× higher** in absolute
   terms.
2. **Our kernel cannot follow it.** The WMMA API lowers to per-warp `mma.sync` instructions.
   On Hopper, `mma.sync` cannot reach the Tensor Core peak — that requires `wgmma.mma_async`
   (warpgroup MMA, 128 threads cooperating, operands fed from shared memory). This is a
   documented architectural property, not a tuning gap: Hopper microbenchmark studies
   (arXiv:2402.13499, arXiv:2501.12084) measure the `mma`-path far below the `wgmma`-path,
   and worked H100 GEMM examples show WMMA-only kernels plateau near ~10% of peak regardless
   of tiling effort.

`ncu --set full` on both kernels at 8192³ quantifies the gap (full table in
`results/nsys_profile.md`):

| metric (ncu, 8192³) | `gemm_wmma_t<128,128,3>` (ours) | `nvjet_sm90` (cuBLAS-TC) |
|---|---|---|
| Duration | 22.9 ms | 1.56 ms |
| SM compute throughput | 26.7% | **91.9%** |
| DRAM throughput | 6.2% | 28.9% |
| Achieved occupancy | 49.4% | **14.8%** |
| Registers / thread | 64 | 168 |
| Top stall reason | MIO queue full (shared-mem traffic) | WARPGROUP.ARRIVES (wgmma sync) |

The signature is unmistakable: the winning kernel runs at **low occupancy with huge register
state and near-peak tensor utilization** (the warpgroup model), while the WMMA kernel burns its
issue slots on shared-memory `ld`/`st` (MIO stalls) feeding fragments to `mma.sync` — high
occupancy, low utilization.

**What transfers across generations and what doesn't:** the optimization *story* (tiling,
`cp.async` pipelining, register tiling) transfers — the kernel's absolute TFLOP/s is nearly
identical on both cards (103 vs 61 TFLOP/s, the difference tracking SM count × clock for
`mma.sync`-issue-bound code). What does **not** transfer is the *fraction of the ceiling*:
each architecture generation moves the ceiling behind a new instruction (Volta `mma.sync` →
Ampere `mma.sync`+`cp.async` → Hopper `wgmma`+TMA → Blackwell `tcgen05`), and a kernel written
against the previous generation's abstraction silently keeps its absolute speed while losing
its relative one. That is the actual lesson an SA needs when a partner asks "why is my custom
kernel slow on the new GPUs?"

> ### On the two cuBLAS baselines (read before quoting any "% of cuBLAS")
> **% of FP32 cuBLAS** (e.g. 1125% @ 512, 189% @ 8192) compares **FP16-on-Tensor-Cores WMMA** against
> **`cublasSgemm` FP32 on CUDA cores** — precision-mismatched, **not** a Tensor Core ceiling; a `>100%`
> row reflects that mismatch (plus small-size launch overhead), not a kernel beating cuBLAS.
> **% of cuBLAS-TC** compares against **`cublas_tc` (`cublasGemmEx`, FP16 in / FP32 accumulate)** in
> `src/cublas_tc.cu` — same precision, same timing methodology (FP16 cast staged once outside the timed
> loop; cuBLAS handle created once). Against this honest ceiling the optimized WMMA lands at ~45% at
> large sizes. **The remaining gap is *not* TMA or warp specialization** — nsys shows cuBLAS
> dispatches `cutlass_80_tensorop_s16816gemm_f16_128x64`, an **Ampere-generation (sm_80) kernel**
> that uses `cp.async` multistage pipelining, *not* TMA and *not* warp specialization (both are
> sm_90/Hopper+ features). The gap comes from cuBLAS's larger 128×64 CTA tile, deeper
> register-level warp/thread tiling, vectorized loads, swizzle/rasterization, and more aggressive
> multistage `cp.async` pipelining — the kernel-name string itself (`cutlass_80_*`) is the
> evidence. This is consistent with Boehm's published GEMM ablation, where 2D blocktiling +
> vectorized loads + warptiling reaches ~94% of cuBLAS *without* TMA or warp specialization.

## References
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) — the production reference for Tensor Core GEMM.
- [WMMA API (CUDA C++ Programming Guide)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma) — the API used by `gemm_wmma.cu`.
- [Simon Boehm, "How to Optimize a CUDA Matmul Kernel"](https://siboehm.com/articles/22/CUDA-MMM) — quantified step-by-step ablation: 2D blocktiling alone reaches 68.7% of cuBLAS, +vectorized loads 78.4%, +warptiling 93.7%, all without TMA or warp specialization.
- [CudaDMA: Optimizing GPU Memory Bandwidth via Warp Specialization (Bauer et al., SC'11)](https://research.nvidia.com/publication/2011-11_cudadma-optimizing-gpu-memory-bandwidth-warp-specialization) — origin of warp specialization; it pays off only with large tiles + async memory hardware + register reallocation.
- [CUTLASS Efficient GEMM docs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html) — CTA/warp/thread tiling, multistage pipelining, and kernel naming.

## Disclaimer
Personal project for learning and benchmarking. Views and results are my own and do not represent any employer.
