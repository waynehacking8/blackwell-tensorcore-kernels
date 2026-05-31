// cuBLAS SGEMM — the performance ceiling and a correctness reference.
#include <cublas_v2.h>
#include "util.cuh"
void launch_cublas(const float*A,const float*B,float*C,int M,int N,int K){
  cublasHandle_t h; cublasCreate(&h);
  float al=1.f,be=0.f;
  // cuBLAS is column-major: compute C^T = B^T A^T by swapping args.
  cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&al,B,N,A,K,&be,C,N);
  cublasDestroy(h);
}
