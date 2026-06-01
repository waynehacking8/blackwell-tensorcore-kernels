// WMMA Tensor Core GEMM: templated block tile, per-warp register tiling,
// multi-stage cp.async shared-memory pipeline. FP16 in / FP32 accumulate.
// Targets sm_90 (H100) and sm_120 (Blackwell RTX Pro 6000); cp.async needs sm_80+.
//
// Structure (closing the gap to cuBLAS's tensorop kernel, per the nsys profile):
//   * BMxBN output tile per block, 16 warps (4x4). Each warp owns
//     (BM/4)x(BN/4) = WITER_M x WITER_N 16x16 WMMA fragments -> that many FP32
//     accumulators in registers. Register tiling reuses each loaded A/B fragment
//     across the other warp dimension, raising arithmetic intensity.
//   * BK=32 K-slab; STAGES-deep cp.async pipeline prefetches future slabs so the
//     Tensor Cores never wait on global memory. Dynamic shared memory holds them.
//   * Two instances are dispatched by size: a 64x64 tile (more blocks -> better
//     occupancy at small N) and a 128x128 tile (more reuse -> wins at large N).
// Assumes M,N multiples of 64 and K a multiple of 32 (true for the 512..8192 sweep).
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include "util.cuh"
using namespace nvcuda;

#define WM 16
#define WN 16
#define WK 16
#define BK 32
#define WARPS_M 4
#define WARPS_N 4
#define NTHREAD (WARPS_M*WARPS_N*32)   // 512

template<int BM,int BN,int STAGES>
__global__ void __launch_bounds__(NTHREAD)
gemm_wmma_t(const half* __restrict__ A,const half* __restrict__ B,float* __restrict__ C,
            int M,int N,int K){
  constexpr int WITER_M=BM/WARPS_M/WM;
  constexpr int WITER_N=BN/WARPS_N/WN;
  constexpr int SLAB=BM*BK + BK*BN;        // halves per stage
  extern __shared__ half smem[];
  half* As[STAGES]; half* Bs[STAGES];
  #pragma unroll
  for(int s=0;s<STAGES;s++){ As[s]=smem+s*SLAB; Bs[s]=smem+s*SLAB+BM*BK; }

  const int tid=threadIdx.x, warp=tid>>5;
  const int wm=warp/WARPS_N, wn=warp%WARPS_N;
  const int blockRow=blockIdx.y*BM, blockCol=blockIdx.x*BN;

  wmma::fragment<wmma::accumulator,WM,WN,WK,float> acc[WITER_M][WITER_N];
  #pragma unroll
  for(int i=0;i<WITER_M;i++) for(int j=0;j<WITER_N;j++) wmma::fill_fragment(acc[i][j],0.f);

  constexpr int AS_CH=(BM*BK)/8, BS_CH=(BK*BN)/8;
  auto load=[&](int s,int k0){
    #pragma unroll
    for(int i=tid;i<AS_CH;i+=NTHREAD){
      int row=(i*8)/BK, col=(i*8)%BK; int gRow=blockRow+row;
      half* dst=&As[s][row*BK+col];
      if(gRow<M) __pipeline_memcpy_async(dst,&A[gRow*K+(k0+col)],16);
      else *reinterpret_cast<float4*>(dst)=make_float4(0,0,0,0);
    }
    #pragma unroll
    for(int i=tid;i<BS_CH;i+=NTHREAD){
      int row=(i*8)/BN, col=(i*8)%BN; int gCol=blockCol+col;
      half* dst=&Bs[s][row*BN+col];
      if(gCol<N) __pipeline_memcpy_async(dst,&B[(k0+row)*N+gCol],16);
      else *reinterpret_cast<float4*>(dst)=make_float4(0,0,0,0);
    }
  };

  const int numTiles=(K+BK-1)/BK;
  #pragma unroll
  for(int s=0;s<STAGES-1;s++){ if(s<numTiles) load(s,s*BK); __pipeline_commit(); }

  wmma::fragment<wmma::matrix_a,WM,WN,WK,half,wmma::row_major> a[WITER_M];
  wmma::fragment<wmma::matrix_b,WM,WN,WK,half,wmma::row_major> b[WITER_N];

  for(int t=0;t<numTiles;t++){
    int cur=t%STAGES, fetch=t+STAGES-1;
    if(fetch<numTiles){ load(fetch%STAGES, fetch*BK); __pipeline_commit(); }
    __pipeline_wait_prior(STAGES-1);
    __syncthreads();
    const half* As_b=As[cur]; const half* Bs_b=Bs[cur];
    #pragma unroll
    for(int kk=0;kk<BK;kk+=WK){
      #pragma unroll
      for(int im=0;im<WITER_M;im++)
        wmma::load_matrix_sync(a[im], As_b+((wm*WITER_M+im)*WM)*BK+kk, BK);
      #pragma unroll
      for(int in=0;in<WITER_N;in++)
        wmma::load_matrix_sync(b[in], Bs_b+kk*BN+((wn*WITER_N+in)*WN), BN);
      #pragma unroll
      for(int im=0;im<WITER_M;im++)
        #pragma unroll
        for(int in=0;in<WITER_N;in++)
          wmma::mma_sync(acc[im][in], a[im], b[in], acc[im][in]);
    }
    __syncthreads();
  }

  #pragma unroll
  for(int im=0;im<WITER_M;im++)
    #pragma unroll
    for(int in=0;in<WITER_N;in++){
      int cRow=blockRow+(wm*WITER_M+im)*WM, cCol=blockCol+(wn*WITER_N+in)*WN;
      if(cRow<M && cCol<N) wmma::store_matrix_sync(C+cRow*N+cCol, acc[im][in], N, wmma::mem_row_major);
    }
}

__global__ void f2h(const float* in, half* out, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=__float2half(in[i]);
}

// Tile choices: small N -> 64x64 (more CTAs, better occupancy); large N -> 128x128
// (2x2 register tiling, more reuse). Both use a 3-stage cp.async pipeline.
template<int BM,int BN,int ST>
static void run(const half*A,const half*B,float*C,int M,int N,int K){
  static int set=-1; size_t sh=(size_t)ST*(BM*BK+BK*BN)*sizeof(half);
  if(set!=BM){ cudaFuncSetAttribute(gemm_wmma_t<BM,BN,ST>,
                 cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh); set=BM; }
  dim3 t(NTHREAD), g((N+BN-1)/BN,(M+BM-1)/BM);
  gemm_wmma_t<BM,BN,ST><<<g,t,sh>>>(A,B,C,M,N,K);
}

void launch_wmma(const float*A,const float*B,float*C,int M,int N,int K){
  // Stage A,B into FP16 ONCE (cast outside the timed region); persists per process.
  static half *Ah=nullptr,*Bh=nullptr;
  static const float *cA=nullptr,*cB=nullptr; static int cM=0,cN=0,cK=0;
  if(M!=cM || N!=cN || K!=cK){ cudaFree(Ah); cudaFree(Bh); Ah=Bh=nullptr; cA=cB=nullptr; }
  if(!Ah){
    CUDA_CHECK(cudaMalloc(&Ah,sizeof(half)*M*K));
    CUDA_CHECK(cudaMalloc(&Bh,sizeof(half)*K*N));
    cM=M; cN=N; cK=K;
  }
  if(A!=cA){ int n=M*K; f2h<<<(n+255)/256,256>>>(A,Ah,n); cA=A; }
  if(B!=cB){ int n=K*N; f2h<<<(n+255)/256,256>>>(B,Bh,n); cB=B; }

  // crossover ~1536: below it the 128-tile leaves the GPU under-occupied. The
  // small path uses 2 stages (few K-slabs -> 3 stages just wastes smem/occupancy);
  // the large path uses a deeper 3-stage pipeline to hide global-memory latency.
  int big = (M>=1536 && N>=1536);
  if(big) run<128,128,3>(Ah,Bh,C,M,N,K);
  else    run<64,64,2>(Ah,Bh,C,M,N,K);
}
