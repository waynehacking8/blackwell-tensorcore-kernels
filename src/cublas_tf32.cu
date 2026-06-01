// cuBLAS TF32 Tensor Core baseline: cublasGemmEx, FP32 inputs routed to the
// TF32 Tensor Core path (CUBLAS_COMPUTE_32F_FAST_TF32). This is the *middle*
// rung of the precision ladder between the FP32 CUDA-core baseline (reference.cu,
// cublasSgemm) and the FP16 Tensor Core ceiling (cublas_tc.cu): same FP32 inputs
// as Sgemm — no FP16 cast — but executed on Tensor Cores at TF32 (19-bit, 10-bit
// mantissa) precision. Expected: throughput between Sgemm and FP16-TC; max abs
// error between FP32 (~1e-4) and FP16 (~1e-2), i.e. ~1e-3.
#include <cublas_v2.h>
#include "util.cuh"

void launch_cublas_tf32(const float*A,const float*B,float*C,int M,int N,int K){
  static cublasHandle_t h=nullptr;
  if(!h) cublasCreate(&h);
  float al=1.f,be=0.f;
  // Column-major: compute C^T = B^T A^T by swapping args, matching reference.cu.
  // FP32 in/out (CUDA_R_32F) but TF32 compute -> cuBLAS dispatches to the Tensor
  // Core TF32 path automatically. No FP16 staging: TF32 truncates FP32 in-hardware.
  cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&al,
               B,CUDA_R_32F,N,
               A,CUDA_R_32F,K,
               &be,
               C,CUDA_R_32F,N,
               CUBLAS_COMPUTE_32F_FAST_TF32,CUBLAS_GEMM_DEFAULT);
}
