# Validation & Cross-Reference

Cross-checking the measured GEMM numbers (`results/bench.csv`) against vendor specs
and expected kernel behaviour. GPU: **NVIDIA RTX PRO 6000 Blackwell Max-Q
Workstation Edition** (`sm_120`, 300 W Max-Q), CUDA 12.8, FP32 inputs, cuBLAS as ceiling.

## Hardware spec cross-check

- **Spec:** RTX PRO 6000 Blackwell **FP32 = 125 TFLOPS** dense (non-tensor), @ **600 W**
  (nvidia.com / flopper.io).
- **Measured:** cuBLAS `cublasSgemm` peaks at **~54.7 TFLOP/s** at M=N=K=8192.
- **Reconciliation:** this board is the **Max-Q (300 W)** variant — roughly half the
  power budget of the 600 W spec part. ~54.7 / 125 ≈ **44%**, i.e. about half the
  rated FP32 throughput, consistent with a halved power/clock envelope. ✓ Reasonable.

## Kernel-vs-ceiling sanity

At M=N=K=8192 (the most timing-stable point):

| kernel | TFLOP/s | % of FP32 cuBLAS | % of cuBLAS-TC | max abs err | reading |
|---|---|---|---|---|---|
| naive | 4.8 | 8.7% | 2.1% | 0 | one-thread-per-output, no reuse — expected floor |
| tiled | 7.3 | 13.4% | 3.2% | 0 | shared-mem + register block, ~1.5× naive |
| **wmma** | 102.7 | 188% † | **45.0%** | 0.011 | 128×128 reg-tiled + 3-stage cp.async, FP16-in/FP32-acc TC |
| cublas | 54.7 | 100% | 23.9% | 0 | FP32 baseline (cublasSgemm, CUDA cores) |
| **cublas_tc** | 228.4 | 418% † | **100%** | 0.011 | FP16/FP32-acc Tensor Core ceiling (cublasGemmEx) |

> **% of cuBLAS-TC** is the honest same-precision ceiling (vs `cublasGemmEx`, FP16 in / FP32
> acc, Tensor Cores). **% of FP32 cuBLAS** is vs `cublasSgemm` (FP32, CUDA cores) and is
> precision-mismatched (**†**): WMMA and cublas_tc run FP16 on Tensor Cores, so their `>100%`
> rows there (e.g. wmma 1097% @ 512, cublas_tc 418% @ 8192) are **not** kernels beating cuBLAS —
> just FP16-TC vs FP32-CUDA-core. Against the same-precision cuBLAS-TC ceiling the optimized WMMA
> kernel (size-dispatched: 64×64 for N<1536, 128×128 + 2×2 register tiling + 3-stage cp.async for
> N≥1536) reaches **45.0% @ 8192** (40.7% @ 4096, 32.0% @ 2048; 31.4% @ 1024, 47.6% @ 512 on the
> small-tile path) — see the naive→optimized→warp-spec progression in `results/nsys_profile.md`.

- **WMMA against the honest same-precision ceiling.** Vs cuBLAS-TC (same FP16-in/FP32-acc
  Tensor Core path), the optimized WMMA kernel reaches **45.0% @ 8192**. Three stages of work
  got it there from the naive **17.3%**: (1) shared-mem tiling + cp.async double-buffering fixed
  the naive memory-bound decay (47→63 TFLOP/s); (2) a 128×128 tile with per-warp 2×2 register
  tiling + 3-stage pipeline raised reuse (→103 TFLOP/s, 1.64×); (3) size dispatch keeps the
  64×64 tile for N<1536 so small matrices don't lose occupancy. Pipeline depth is tuned by
  measurement (3 stages 103 > 4 stages 95 > 5 stages 88 TFLOP/s @ 8192). **Warp specialization
  was implemented and benchmarked but did *not* beat the multi-stage pipeline** (95.4 vs 103.2 @
  8192) — at this 512-thread tile the cp.async pipeline already saturates latency-hiding, so
  dedicating warps to production costs more mma throughput than it saves. The remaining gap to
  cuBLAS is its larger CTA tile and warp-specialized scheduling on top of TMA-class transfers.
  The "% of FP32 cuBLAS" figure remains precision-mismatched and is kept only for continuity.
- **cuBLAS-TC confirms the Tensor Core path.** cublas_tc reaches 229 TFLOP/s @ 8192 — **4.2×**
  the FP32 cublasSgemm (54.4) and up to **22×** at 512 — which is only possible on the 5th-gen
  Tensor Cores, so the baseline is doing what it claims.
- **Precision check:** naive/tiled (FP32) max-abs-err ~1e-4→0; wmma and cublas_tc (FP16 inputs)
  ~0.01 over large K — exactly the expected FP16 rounding, confirming correctness.

## Honest caveat — now resolved by the same-precision baseline

The original "% of FP32 cuBLAS" was apples-to-oranges (FP16-WMMA vs FP32-`cublasSgemm`); its
`>100%` small-N rows reflect the precision difference, not WMMA beating a same-precision kernel.
That prediction has now been **measured**: the harness includes **`cublas_tc`
(`cublasGemmEx`, FP16 in / FP32 accumulate)** in `src/cublas_tc.cu`, with identical timing
methodology to the WMMA kernel — the FP32→FP16 cast is staged once outside the timed loop, and
(after a fix) the cuBLAS handle is created once rather than per timed call, which had dominated
small-N timing.

Result on sm_120, vs this honest ceiling (full sweep in `bench.csv`; FP32-only sweep preserved
in `bench_fp32only.csv`). WMMA here is the **size-dispatched, register-tiled, 3-stage cp.async**
kernel:

| size | wmma % of FP32 cuBLAS | **wmma % of cuBLAS-TC** | cublas_tc speedup vs FP32 |
|---|---|---|---|
| 512  | 1097% | 47.6% | 21.3× |
| 1024 | 371%  | 31.4% | 14.3× |
| 2048 | 151%  | 32.0% | 5.7× |
| 4096 | 118%  | 40.7% | 4.5× |
| 8192 | 188%  | **45.0%** | 4.2× |

The WMMA kernel reaches **45% @ 8192** against the same-precision Tensor Core ceiling (naive was
17–22% and decayed at 8192; shared-mem+cp.async, then register tiling + deeper pipeline + size
dispatch closed most of the gap; warp specialization was tried and rejected — see
`results/nsys_profile.md`). No size exceeds 100% of cuBLAS-TC. The FP32-cuBLAS column is retained
only for continuity.

## Sources
- [RTX PRO 6000 Blackwell — NVIDIA](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/)
- [RTX PRO 6000 Blackwell spec sheet — flopper.io](https://flopper.io/gpu/nvidia-rtx-pro-6000-blackwell-workstation-edition)
- [NVIDIA RTX Blackwell PRO GPU Architecture (v1.0 PDF)](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf)
