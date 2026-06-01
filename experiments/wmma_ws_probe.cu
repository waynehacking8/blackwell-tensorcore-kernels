// EXPERIMENT (not in the build): warp-specialized WMMA — 16 consumer warps do
// mma, 4 producer warps do all cp.async, synced via named barriers. Standalone
// probe with its own main(); compares against the dispatch kernel's 128x128 path.
// Built manually:  nvcc -O3 -arch=sm_120 src/wmma_ws_probe.cu -lcublas -o /tmp/ws
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cstdio>
#include <vector>
#include <cmath>
using namespace nvcuda;
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %d %s\n",__LINE__,cudaGetErrorString(e));return 1;}}while(0)
#define WM 16
#define WN 16
#define WK 16
#define BM 128
#define BN 128
#define BK 32
#define CONS_WARPS 16          // 4x4 consumer warps
#define PROD_WARPS 4
#define TOTWARP (CONS_WARPS+PROD_WARPS)
#define NTHREAD (TOTWARP*32)   // 640
#define STAGES 3
#define SLAB (BM*BK + BK*BN)
#define WITER_M (BM/4/WM)      // 2
#define WITER_N (BN/4/WN)      // 2

// named barriers: producers signal "stage filled", consumers signal "stage freed"
__device__ __forceinline__ void bar_arrive(int bar,int cnt){ asm volatile("bar.arrive %0,%1;"::"r"(bar),"r"(cnt)); }
__device__ __forceinline__ void bar_sync(int bar,int cnt){ asm volatile("bar.sync %0,%1;"::"r"(bar),"r"(cnt)); }

__global__ void __launch_bounds__(NTHREAD)
gemm_ws(const half* __restrict__ A,const half* __restrict__ B,float* __restrict__ C,int M,int N,int K){
  extern __shared__ half smem[];
  half* As[STAGES]; half* Bs[STAGES];
  #pragma unroll
  for(int s=0;s<STAGES;s++){ As[s]=smem+s*SLAB; Bs[s]=smem+s*SLAB+BM*BK; }
  const int tid=threadIdx.x, warp=tid>>5;
  const int blockRow=blockIdx.y*BM, blockCol=blockIdx.x*BN;
  const int numTiles=(K+BK-1)/BK;
  const bool producer = warp>=CONS_WARPS;

  // barriers 1..STAGES = "filled[s]" (prod->cons), STAGES+1..2*STAGES = "freed[s]" (cons->prod)
  // each uses all NTHREAD threads so arrive/sync counts match.
  if(producer){
    const int ptid = (warp-CONS_WARPS)*32 + (tid&31);
    const int PCH=(BM*BK)/8, QCH=(BK*BN)/8;
    for(int t=0;t<numTiles;t++){
      int s=t%STAGES;
      if(t>=STAGES) bar_sync(STAGES+1+s, NTHREAD);   // wait until consumers freed stage s
      int k0=t*BK;
      for(int i=ptid;i<PCH;i+=PROD_WARPS*32){
        int row=(i*8)/BK, col=(i*8)%BK; int gRow=blockRow+row;
        half* d=&As[s][row*BK+col];
        if(gRow<M) __pipeline_memcpy_async(d,&A[gRow*K+(k0+col)],16); else *reinterpret_cast<float4*>(d)=make_float4(0,0,0,0);
      }
      for(int i=ptid;i<QCH;i+=PROD_WARPS*32){
        int row=(i*8)/BN, col=(i*8)%BN; int gCol=blockCol+col;
        half* d=&Bs[s][row*BN+col];
        if(gCol<N) __pipeline_memcpy_async(d,&B[(k0+row)*N+gCol],16); else *reinterpret_cast<float4*>(d)=make_float4(0,0,0,0);
      }
      __pipeline_commit(); __pipeline_wait_prior(0);
      bar_arrive(1+s, NTHREAD);                        // signal stage s filled
    }
  } else {
    const int wm=warp/4, wn=warp%4;
    wmma::fragment<wmma::accumulator,WM,WN,WK,float> acc[WITER_M][WITER_N];
    #pragma unroll
    for(int i=0;i<WITER_M;i++)for(int j=0;j<WITER_N;j++) wmma::fill_fragment(acc[i][j],0.f);
    wmma::fragment<wmma::matrix_a,WM,WN,WK,half,wmma::row_major> a[WITER_M];
    wmma::fragment<wmma::matrix_b,WM,WN,WK,half,wmma::row_major> b[WITER_N];
    for(int t=0;t<numTiles;t++){
      int s=t%STAGES;
      bar_sync(1+s, NTHREAD);                          // wait until stage s filled
      const half* As_b=As[s]; const half* Bs_b=Bs[s];
      #pragma unroll
      for(int kk=0;kk<BK;kk+=WK){
        #pragma unroll
        for(int im=0;im<WITER_M;im++) wmma::load_matrix_sync(a[im],As_b+((wm*WITER_M+im)*WM)*BK+kk,BK);
        #pragma unroll
        for(int in=0;in<WITER_N;in++) wmma::load_matrix_sync(b[in],Bs_b+kk*BN+((wn*WITER_N+in)*WN),BN);
        #pragma unroll
        for(int im=0;im<WITER_M;im++)
          #pragma unroll
          for(int in=0;in<WITER_N;in++) wmma::mma_sync(acc[im][in],a[im],b[in],acc[im][in]);
      }
      bar_arrive(STAGES+1+s, NTHREAD);                 // signal stage s freed
    }
    #pragma unroll
    for(int im=0;im<WITER_M;im++)
      #pragma unroll
      for(int in=0;in<WITER_N;in++){
        int cR=blockRow+(wm*WITER_M+im)*WM, cC=blockCol+(wn*WITER_N+in)*WN;
        if(cR<M&&cC<N) wmma::store_matrix_sync(C+cR*N+cC,acc[im][in],N,wmma::mem_row_major);
      }
  }
}
__global__ void f2h(const float*in,half*out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=__float2half(in[i]);}

#include <cublas_v2.h>
int main(int argc,char**argv){
  int S=argc>1?atoi(argv[1]):8192; int M=S,N=S,K=S;
  std::vector<float> hA(M*K),hB(K*N); for(auto&x:hA)x=(rand()/(float)RAND_MAX)-.5f; for(auto&x:hB)x=(rand()/(float)RAND_MAX)-.5f;
  float *dA,*dB,*dC,*dRef; half*Ah,*Bh;
  CK(cudaMalloc(&dA,4llu*M*K));CK(cudaMalloc(&dB,4llu*K*N));CK(cudaMalloc(&dC,4llu*M*N));CK(cudaMalloc(&dRef,4llu*M*N));
  CK(cudaMalloc(&Ah,2llu*M*K));CK(cudaMalloc(&Bh,2llu*K*N));
  CK(cudaMemcpy(dA,hA.data(),4llu*M*K,cudaMemcpyHostToDevice));CK(cudaMemcpy(dB,hB.data(),4llu*K*N,cudaMemcpyHostToDevice));
  f2h<<<(M*K+255)/256,256>>>(dA,Ah,M*K); f2h<<<(K*N+255)/256,256>>>(dB,Bh,K*N);
  // cublas_tc reference
  cublasHandle_t h; cublasCreate(&h); float al=1,be=0;
  cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&al,Bh,CUDA_R_16F,N,Ah,CUDA_R_16F,K,&be,dRef,CUDA_R_32F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
  CK(cudaDeviceSynchronize());
  size_t sh=(size_t)STAGES*SLAB*sizeof(half);
  CK(cudaFuncSetAttribute(gemm_ws,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh));
  dim3 t(NTHREAD),g((N+BN-1)/BN,(M+BM-1)/BM);
  gemm_ws<<<g,t,sh>>>(Ah,Bh,dC,M,N,K); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  std::vector<float> hC(M*N),hR(M*N);
  CK(cudaMemcpy(hC.data(),dC,4llu*M*N,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hR.data(),dRef,4llu*M*N,cudaMemcpyDeviceToHost));
  double mx=0; for(size_t i=0;i<(size_t)M*N;i++) mx=fmax(mx,fabs((double)hC[i]-hR[i]));
  cudaEvent_t a,b; cudaEventCreate(&a);cudaEventCreate(&b);
  gemm_ws<<<g,t,sh>>>(Ah,Bh,dC,M,N,K); cudaDeviceSynchronize();
  cudaEventRecord(a); for(int i=0;i<10;i++) gemm_ws<<<g,t,sh>>>(Ah,Bh,dC,M,N,K); cudaEventRecord(b); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms,a,b); ms/=10;
  double tf=2.0*M*N*K/(ms*1e-3)/1e12;
  printf("WS  S=%d  %.3f ms  %.2f TFLOP/s  maxerr=%.4g\n",S,ms,tf,mx);
  return 0;
}
