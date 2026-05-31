// Shared-memory tiled GEMM (32x32 tiles). Classic SIMT optimization; the FP32 SIMT
// reference that the Tensor Core kernel must beat.
#include "util.cuh"
#define T 32
__global__ void gemm_tiled(const float* A,const float* B,float* C,int M,int N,int K){
  __shared__ float As[T][T], Bs[T][T];
  int row=blockIdx.y*T+threadIdx.y, col=blockIdx.x*T+threadIdx.x;
  float acc=0.f;
  for(int t=0;t<(K+T-1)/T;t++){
    int ak=t*T+threadIdx.x, bk=t*T+threadIdx.y;
    As[threadIdx.y][threadIdx.x] = (row<M&&ak<K)? A[row*K+ak]:0.f;
    Bs[threadIdx.y][threadIdx.x] = (col<N&&bk<K)? B[bk*N+col]:0.f;
    __syncthreads();
    for(int k=0;k<T;k++) acc += As[threadIdx.y][k]*Bs[k][threadIdx.x];
    __syncthreads();
  }
  if(row<M&&col<N) C[row*N+col]=acc;
}
void launch_tiled(const float*A,const float*B,float*C,int M,int N,int K){
  dim3 t(T,T), g((N+T-1)/T,(M+T-1)/T);
  gemm_tiled<<<g,t>>>(A,B,C,M,N,K);
}
