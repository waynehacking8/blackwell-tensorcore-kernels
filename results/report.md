# Blackwell vs Hopper Tensor Core GEMM — results

## NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (sm_120)

![throughput vs size](tflops_sm120.png)

![% of cuBLAS-TC](pct_tc_sm120.png)

![throughput bar at 8192](roofline_sm120.png)

At M=N=K=8192:

| kernel | TFLOP/s | % of FP32 cuBLAS | % of cuBLAS-TC | max abs err |
|---|---|---|---|---|
| naive | 4.8 | 8.7% | 2.1% | 0 |
| tiled | 7.3 | 13.3% | 3.2% | 0 |
| wmma | 103.5 | 188.7% | 45.2% | 0.0112 |
| cublas | 54.8 | 100.0% | 23.9% | 0 |
| cublas_tf32 | 152.7 | 278.4% | 66.6% | 0.0113 |
| cublas_tc | 229.2 | 417.9% | 100.0% | 0.0112 |

> Precision ladder, all on the **same card**: **cublas** = `cublasSgemm` (FP32, CUDA cores); **cublas_tf32** = `cublasGemmEx` (FP32 in, TF32 compute, Tensor Cores); **cublas_tc** = `cublasGemmEx` (FP16 in / FP32 acc, Tensor Cores) — the honest same-precision ceiling for `wmma`. **% of FP32 cuBLAS** is precision-mismatched (FP16/TF32-TC vs FP32-CUDA-core), so its `>100%` rows are **not** a kernel beating cuBLAS. Quote **% of cuBLAS-TC**.

