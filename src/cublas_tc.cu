// cublas_tc.cu — cuBLAS Tensor Core GEMM baseline (FP16 in / FP32 accumulate).
// Same-precision honest ceiling for the WMMA kernel: cublasGemmEx with
// CUDA_R_16F inputs and CUDA_R_32F output. With CUBLAS_COMPUTE_32F and FP16
// inputs, modern cuBLAS dispatches to the Tensor Core path automatically.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "util.cuh"

// Times cublasGemmEx (FP16 in, FP32 out, Tensor Core path) over `iters`
// iterations using CUDA events. Returns average milliseconds per call.
// FP32->FP16 cast is done outside this function (in main.cu), matching the
// WMMA timing methodology — the cast is not part of the timed region.
float time_cublas_tc(cublasHandle_t handle,
                     const half* A, const half* B, float* C,
                     int M, int N, int K, int iters) {
    const float alpha = 1.0f, beta = 0.0f;

    // warmup
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              M, N, K, &alpha,
                              A, CUDA_R_16F, M,
                              B, CUDA_R_16F, K,
                              &beta,
                              C, CUDA_R_32F, M,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));
    for (int it = 0; it < iters; ++it) {
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  M, N, K, &alpha,
                                  A, CUDA_R_16F, M,
                                  B, CUDA_R_16F, K,
                                  &beta,
                                  C, CUDA_R_32F, M,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    return ms / iters;
}
