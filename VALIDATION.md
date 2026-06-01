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
| **wmma** | 62.6 | 114% † | **27.3%** | 0.011 | shared-mem tiled + cp.async, FP16-in/FP32-acc TC |
| cublas | 54.7 | 100% | 23.9% | 0 | FP32 baseline (cublasSgemm, CUDA cores) |
| **cublas_tc** | 229.3 | 419% † | **100%** | 0.011 | FP16/FP32-acc Tensor Core ceiling (cublasGemmEx) |

> **% of cuBLAS-TC** is the honest same-precision ceiling (vs `cublasGemmEx`, FP16 in / FP32
> acc, Tensor Cores). **% of FP32 cuBLAS** is vs `cublasSgemm` (FP32, CUDA cores) and is
> precision-mismatched (**†**): WMMA and cublas_tc run FP16 on Tensor Cores, so their `>100%`
> rows there (e.g. wmma 1097% @ 512, cublas_tc 419% @ 8192) are **not** kernels beating cuBLAS —
> just FP16-TC vs FP32-CUDA-core. Against the same-precision cuBLAS-TC ceiling the (now
> shared-mem + cp.async pipelined) WMMA kernel holds **~26–27%** at large sizes (27.3% @ 8192,
> 26.4% @ 4096, 26.2% @ 2048, 28.6% @ 1024; 49.5% @ 512 is small-matrix launch overhead) and no
> longer decays at 8192 — see the naive→optimized before/after in `results/nsys_profile.md`.

- **WMMA against the honest same-precision ceiling.** Vs cuBLAS-TC (same FP16-in/FP32-acc
  Tensor Core path), the WMMA kernel — after adding shared-memory tiling + cp.async double
  buffering — holds **~26–27%** across large sizes (27.3% @ 8192 … 28.6% @ 1024). The earlier
  *naive* version (no shared-mem reuse) sat at ~17–22% and decayed to 17% at 8192 as the Tensor
  Cores starved on global-memory traffic; pipelining the global→shared loads fixed that
  (1.58× @ 8192, holds ~63 TFLOP/s). The remaining gap to 100% is cuBLAS's deeper 3+-stage,
  128×64 register-tiled pipeline. The "% of FP32 cuBLAS" figure remains precision-mismatched
  and is kept only for continuity.
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
in `bench_fp32only.csv`). WMMA here is the **shared-mem + cp.async pipelined** kernel:

| size | wmma % of FP32 cuBLAS | **wmma % of cuBLAS-TC** | cublas_tc speedup vs FP32 |
|---|---|---|---|
| 512  | 1097% | 49.5% | 21.3× |
| 1024 | 371%  | 28.6% | 14.3× |
| 2048 | 151%  | 26.2% | 5.7× |
| 4096 | 118%  | 26.4% | 4.5× |
| 8192 | 114%  | **27.3%** | 4.2× |

The WMMA figure sits at the **~26–27%** range against the same-precision Tensor Core ceiling
(naive was ~17–22% and decayed at 8192; the cp.async pipeline closed that — see
`results/nsys_profile.md`). No size exceeds 100% of cuBLAS-TC (512's 49.5% is small-matrix
launch overhead). The FP32-cuBLAS column is retained only for continuity.

## Sources
- [RTX PRO 6000 Blackwell — NVIDIA](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/)
- [RTX PRO 6000 Blackwell spec sheet — flopper.io](https://flopper.io/gpu/nvidia-rtx-pro-6000-blackwell-workstation-edition)
- [NVIDIA RTX Blackwell PRO GPU Architecture (v1.0 PDF)](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf)
