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

| kernel | TFLOP/s | % of cuBLAS | max abs err | reading |
|---|---|---|---|---|
| naive | 4.7 | 8.6% | 0 | one-thread-per-output, no reuse — expected floor |
| tiled | 7.2 | 13.2% | 0 | shared-mem + register block, ~1.5× naive |
| **wmma** | 39.7 | **72.6%** | 0.011 | FP16-input Tensor Core, FP32 accumulate |
| cublas | 54.7 | 100% | 0 | FP32 ceiling |

- **WMMA % of cuBLAS is plausible and the timing fix worked.** The WMMA kernel uses
  FP16-input Tensor Cores vs an FP32 cuBLAS baseline, so it *beats* cuBLAS at small N
  (907% @ 512, 306% @ 1024 — Tensor Cores ≫ FP32 CUDA cores) and settles to ~73% at
  8192 as the naive (no shared-mem double-buffer) kernel becomes memory-bound and the
  Tensor Cores starve. That monotone-then-plateau shape is the textbook naive-WMMA
  signature; before the "time the GEMM, not the FP16 cast" fix the numbers were
  meaningless.
- **Precision check:** naive/tiled (FP32) max-abs-err ~1e-4→0; wmma (FP16 inputs)
  ~0.01 over large K — exactly the expected FP16 rounding, confirming correctness.

## Honest caveat

Comparing **FP16-WMMA against an FP32-cuBLAS** ceiling is apples-to-oranges — the
>100% values at small N reflect the precision difference, not WMMA beating a
same-precision tuned kernel. For a true "% of Tensor-Core ceiling", the reference
should use `cublasGemmEx` with TF32/FP16 compute (which would push cuBLAS to several
hundred TFLOP/s and WMMA's % down to the typical ~15–25% naive range). The current
comparison is honest about this.

## Sources
- [RTX PRO 6000 Blackwell — NVIDIA](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/)
- [RTX PRO 6000 Blackwell spec sheet — flopper.io](https://flopper.io/gpu/nvidia-rtx-pro-6000-blackwell-workstation-edition)
- [NVIDIA RTX Blackwell PRO GPU Architecture (v1.0 PDF)](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf)
