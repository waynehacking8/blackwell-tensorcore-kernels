# Blackwell vs Hopper Tensor Core GEMM — results

## NVIDIA H100 80GB HBM3 (sm_90)

![throughput vs size](tflops_sm90.png)

![% of cuBLAS-TC](pct_tc_sm90.png)

![throughput bar at 8192](roofline_sm90.png)

At M=N=K=8192:

| kernel | TFLOP/s | % of FP32 cuBLAS | % of cuBLAS-TC | max abs err |
|---|---|---|---|---|
| naive | 5.2 | 10.9% | 0.7% | 0 |
| tiled | 9.2 | 19.4% | 1.2% | 0 |
| wmma | 60.9 | 128.7% | 8.0% | 0.0112 |
| cublas | 47.3 | 100.0% | 6.2% | 0 |
| cublas_tf32 | 417.6 | 882.3% | 54.8% | 0.0113 |
| cublas_tc | 761.7 | 1609.2% | 100.0% | 0.0112 |

> Precision ladder, all on the **same card**: **cublas** = `cublasSgemm` (FP32, CUDA cores); **cublas_tf32** = `cublasGemmEx` (FP32 in, TF32 compute, Tensor Cores); **cublas_tc** = `cublasGemmEx` (FP16 in / FP32 acc, Tensor Cores) — the honest same-precision ceiling for `wmma`. **% of FP32 cuBLAS** is precision-mismatched (FP16/TF32-TC vs FP32-CUDA-core), so its `>100%` rows are **not** a kernel beating cuBLAS. Quote **% of cuBLAS-TC**.

## NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (sm_120)

![throughput vs size](tflops_sm120.png)

![% of cuBLAS-TC](pct_tc_sm120.png)

![throughput bar at 8192](roofline_sm120.png)

![mma.sync ablation ladder](mma_ablation_sm120.png)

![precision Pareto](precision_pareto_sm120.png)

At M=N=K=8192:

| kernel | TFLOP/s | % of FP32 cuBLAS | % of cuBLAS-TC | max abs err |
|---|---|---|---|---|
| naive | 4.6 | 8.6% | 2.1% | 0 |
| tiled | 7.0 | 12.9% | 3.1% | 0 |
| wmma | 100.2 | 185.0% | 44.4% | 0.0112 |
| mma_base | 47.4 | 87.4% | 21.0% | 0.0112 |
| mma_swizzle | 58.3 | 107.6% | 25.8% | 0.0112 |
| mma_vec | 163.0 | 300.9% | 72.2% | 0.0112 |
| mma_pipe | 175.4 | 323.9% | 77.7% | 0.0112 |
| mma_warptile | 238.5 | 440.3% | 105.6% | 0.0112 |
| mma_fp8 | 502.0 | 926.8% | 222.4% | 1.4 |
| mma_fp4 | 517.6 | 955.6% | 229.3% | 5.97 |
| mma_mxfp4 | 966.7 | 1784.7% | 428.2% | 5.97 |
| cublas | 54.2 | 100.0% | 24.0% | 0 |
| cublas_tf32 | 150.5 | 277.9% | 66.7% | 0.0113 |
| cublas_tc | 225.8 | 416.8% | 100.0% | 0.0112 |
| cublaslt_fp8 | 552.0 | 1019.1% | 244.5% | 1.4 |

> Precision ladder, all on the **same card**: **cublas** = `cublasSgemm` (FP32, CUDA cores); **cublas_tf32** = `cublasGemmEx` (FP32 in, TF32 compute, Tensor Cores); **cublas_tc** = `cublasGemmEx` (FP16 in / FP32 acc, Tensor Cores) — the honest same-precision ceiling for `wmma`. **% of FP32 cuBLAS** is precision-mismatched (FP16/TF32-TC vs FP32-CUDA-core), so its `>100%` rows are **not** a kernel beating cuBLAS. Quote **% of cuBLAS-TC**.

