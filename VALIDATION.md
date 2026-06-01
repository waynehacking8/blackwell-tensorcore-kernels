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
| **wmma** | 39.8 | 73.1% † | **17.4%** | 0.011 | FP16-input Tensor Core, FP32 accumulate |
| cublas | 54.4 | 100% | 23.7% | 0 | FP32 baseline (cublasSgemm, CUDA cores) |
| **cublas_tc** | 229.1 | 421% † | **100%** | 0.011 | FP16/FP32-acc Tensor Core ceiling (cublasGemmEx) |

> **% of cuBLAS-TC** is the honest same-precision ceiling (vs `cublasGemmEx`, FP16 in / FP32
> acc, Tensor Cores). **% of FP32 cuBLAS** is vs `cublasSgemm` (FP32, CUDA cores) and is
> precision-mismatched (**†**): WMMA and cublas_tc run FP16 on Tensor Cores, so their `>100%`
> rows there (e.g. wmma 932% @ 512, cublas_tc 421% @ 8192) are **not** kernels beating cuBLAS —
> just FP16-TC vs FP32-CUDA-core. Against the same-precision cuBLAS-TC ceiling the naive WMMA
> kernel sits at **~17–22%** at large sizes (17.4% @ 8192, 20.0% @ 4096, 20.7% @ 2048, 22.5% @
> 1024; 42.2% @ 512 is small-matrix launch overhead) — the expected naive-WMMA range.

- **WMMA against the honest same-precision ceiling.** Vs cuBLAS-TC (same FP16-in/FP32-acc
  Tensor Core path), the naive WMMA kernel lands at **~17–22%** across large sizes (17.4% @
  8192 … 22.5% @ 1024), exactly the expected naive-WMMA range — it has no shared-memory
  double-buffering, so at large N the Tensor Cores starve on global-memory traffic while
  cuBLAS-TC stays fed. The old "73% of FP32 cuBLAS" figure was an artifact of the
  precision-mismatched baseline (FP16-TC vs FP32-CUDA-core), now superseded by the cuBLAS-TC
  column.
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
in `bench_fp32only.csv`):

| size | wmma % of FP32 cuBLAS | **wmma % of cuBLAS-TC** | cublas_tc speedup vs FP32 |
|---|---|---|---|
| 512  | 932% | 42.2% | 22.1× |
| 1024 | 323% | 22.5% | 14.4× |
| 2048 | 118% | 20.7% | 5.7× |
| 4096 | 90%  | 20.0% | 4.5× |
| 8192 | 73%  | **17.4%** | 4.2× |

Exactly as predicted, the WMMA figure drops to the **~17–22%** naive-WMMA range against the
same-precision Tensor Core ceiling, and no size exceeds 100% of cuBLAS-TC (512's 42% is
small-matrix launch overhead). The FP32-cuBLAS column is retained only for continuity.

## Sources
- [RTX PRO 6000 Blackwell — NVIDIA](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/)
- [RTX PRO 6000 Blackwell spec sheet — flopper.io](https://flopper.io/gpu/nvidia-rtx-pro-6000-blackwell-workstation-edition)
- [NVIDIA RTX Blackwell PRO GPU Architecture (v1.0 PDF)](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf)
