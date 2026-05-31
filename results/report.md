# Blackwell vs Hopper Tensor Core GEMM — results

## NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (sm_120)

![throughput](tflops_sm120.png)

At M=N=K=8192:

| kernel | TFLOP/s | % of cuBLAS | max abs err |
|---|---|---|---|
| naive | 4.7 | 8.6% | 0 |
| tiled | 7.2 | 13.2% | 0 |
| wmma | 39.7 | 72.6% | 0.0112 |
| cublas | 54.7 | 100.0% | 0 |

