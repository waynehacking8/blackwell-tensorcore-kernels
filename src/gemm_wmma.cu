// WMMA Tensor Core GEMM: 16x16x16 fragments, FP16 inputs, FP32 accumulate.
// Runs on Volta+; this repo targets sm_90 (H100) and sm_120 (Blackwell RTX Pro 6000).
// A,B,C are FP32 on host; we convert to half for the Tensor Core path.
#include <mma.h>
#include <cuda_fp16.h>
#include "util.cuh"
using namespace nvcuda;
#define WM 16
#define WN 16
#define WK 16

__global__ void gemm_wmma(const half* A,const half* B,float* C,int M,int N,int K){
  int warpM = (blockIdx.y*blockDim.y + threadIdx.y);
  int warpN = (blockIdx.x*blockDim.x + threadIdx.x)/warpSize;
  wmma::fragment<wmma::matrix_a,WM,WN,WK,half,wmma::row_major> a;
  wmma::fragment<wmma::matrix_b,WM,WN,WK,half,wmma::row_major> b;
  wmma::fragment<wmma::accumulator,WM,WN,WK,float> acc;
  wmma::fill_fragment(acc,0.f);
  int aRow=warpM*WM, bCol=warpN*WN;
  for(int k=0;k<K;k+=WK){
    if(aRow<M && bCol<N && k<K){
      wmma::load_matrix_sync(a, A+aRow*K+k, K);
      wmma::load_matrix_sync(b, B+k*N+bCol, N);
      wmma::mma_sync(acc,a,b,acc);
    }
  }
  if(aRow<M && bCol<N)
    wmma::store_matrix_sync(C+aRow*N+bCol, acc, N, wmma::mem_row_major);
}

__global__ void f2h(const float* in, half* out, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=__float2half(in[i]);
}

void launch_wmma(const float*A,const float*B,float*C,int M,int N,int K){
  half *Ah,*Bh; CUDA_CHECK(cudaMalloc(&Ah,sizeof(half)*M*K)); CUDA_CHECK(cudaMalloc(&Bh,sizeof(half)*K*N));
  int n=M*K; f2h<<<(n+255)/256,256>>>(A,Ah,n);
  n=K*N;     f2h<<<(n+255)/256,256>>>(B,Bh,n);
  dim3 t(128,4), g((N+WN-1)/WN, (M+ (WM*4) -1)/(WM*4));
  gemm_wmma<<<g,t>>>(Ah,Bh,C,M,N,K);
  cudaFree(Ah); cudaFree(Bh);
}
