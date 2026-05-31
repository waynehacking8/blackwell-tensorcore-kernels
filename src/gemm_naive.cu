// Naive GEMM: one thread per output element. Correctness anchor, not for speed.
#include "util.cuh"
__global__ void gemm_naive(const float* A,const float* B,float* C,int M,int N,int K){
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  if(row<M && col<N){
    float acc=0.f;
    for(int k=0;k<K;k++) acc += A[row*K+k]*B[k*N+col];
    C[row*N+col]=acc;
  }
}
void launch_naive(const float*A,const float*B,float*C,int M,int N,int K){
  dim3 t(16,16), g((N+15)/16,(M+15)/16);
  gemm_naive<<<g,t>>>(A,B,C,M,N,K);
}
