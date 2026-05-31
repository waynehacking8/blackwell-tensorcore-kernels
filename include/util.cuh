#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

struct Timer {
  cudaEvent_t a, b;
  Timer(){ cudaEventCreate(&a); cudaEventCreate(&b); }
  void start(){ cudaEventRecord(a); }
  float stop(){ cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms,a,b); return ms; }
};

inline void fill_rand(float* p, int n){ for(int i=0;i<n;i++) p[i] = (rand()/(float)RAND_MAX)-0.5f; }

// returns max abs error between C and a reference
inline double max_abs_err(const float* C, const float* ref, int n){
  double m=0; for(int i=0;i<n;i++) m = fmax(m, fabs((double)C[i]-ref[i])); return m;
}

inline double tflops(long M,long N,long K,float ms){ return (2.0*M*N*K)/(ms*1e-3)/1e12; }
