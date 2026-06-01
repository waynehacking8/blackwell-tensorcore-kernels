# Blackwell vs Hopper Tensor Core GEMM — results

## NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (sm_120)

![throughput](tflops_sm120.png)

At M=N=K=8192:

| kernel | TFLOP/s | % of FP32 cuBLAS | % of cuBLAS-TC | max abs err |
|---|---|---|---|---|
| naive | 4.8 | 8.7% | 2.1% | 0 |
| tiled | 7.3 | 13.3% | 3.2% | 0 |
| wmma | 62.6 | 114.5% | 27.3% | 0.0112 |
| cublas | 54.7 | 100.0% | 23.9% | 0 |
| cublas_tc | 229.3 | 419.1% | 100.0% | 0.0112 |

> **% of FP32 cuBLAS** is vs `cublasSgemm` (FP32, CUDA cores) — precision-mismatched, **not** a Tensor Core ceiling; >100% reflects FP16-TC vs FP32-CUDA-core, not a kernel beating cuBLAS. **% of cuBLAS-TC** is vs `cublasGemmEx` (FP16 in / FP32 acc, Tensor Cores) — the honest same-precision ceiling. `n/a` if no `cublas_tc` row is present (re-run the sweep to populate it).

