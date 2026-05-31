// Driver: correctness vs cuBLAS + TFLOP/s for each kernel -> results/bench.csv
#include "util.cuh"
#include <vector>
#include <cstdio>
void launch_naive(const float*,const float*,float*,int,int,int);
void launch_tiled(const float*,const float*,float*,int,int,int);
void launch_wmma (const float*,const float*,float*,int,int,int);
void launch_cublas(const float*,const float*,float*,int,int,int);

struct Kern{const char*name; void(*fn)(const float*,const float*,float*,int,int,int);};

int main(int argc,char**argv){
  int M=argc>1?atoi(argv[1]):4096, N=argc>2?atoi(argv[2]):4096, K=argc>3?atoi(argv[3]):4096;
  size_t szA=sizeof(float)*M*K, szB=sizeof(float)*K*N, szC=sizeof(float)*M*N;
  std::vector<float> hA(M*K),hB(K*N),hRef(M*N);
  fill_rand(hA.data(),M*K); fill_rand(hB.data(),K*N);
  float *dA,*dB,*dC; CUDA_CHECK(cudaMalloc(&dA,szA));CUDA_CHECK(cudaMalloc(&dB,szB));CUDA_CHECK(cudaMalloc(&dC,szC));
  CUDA_CHECK(cudaMemcpy(dA,hA.data(),szA,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB,hB.data(),szB,cudaMemcpyHostToDevice));

  launch_cublas(dA,dB,dC,M,N,K); CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hRef.data(),dC,szC,cudaMemcpyDeviceToHost));

  Kern kerns[]={{"naive",launch_naive},{"tiled",launch_tiled},{"wmma",launch_wmma},{"cublas",launch_cublas}};
  FILE* f=fopen("results/bench.csv","w"); fprintf(f,"kernel,ms,tflops,max_abs_err\n");
  printf("M=%d N=%d K=%d\n",M,N,K);
  std::vector<float> hC(M*N);
  for(auto&k:kerns){
    Timer t; k.fn(dA,dB,dC,M,N,K); CUDA_CHECK(cudaDeviceSynchronize()); // warmup
    t.start(); for(int i=0;i<10;i++) k.fn(dA,dB,dC,M,N,K); float ms=t.stop()/10.f;
    CUDA_CHECK(cudaMemcpy(hC.data(),dC,szC,cudaMemcpyDeviceToHost));
    double err=max_abs_err(hC.data(),hRef.data(),M*N);
    printf("%-8s %8.3f ms  %7.2f TFLOP/s  maxerr=%.3g\n",k.name,ms,tflops(M,N,K,ms),err);
    fprintf(f,"%s,%.3f,%.2f,%.3g\n",k.name,ms,tflops(M,N,K,ms),err);
  }
  fclose(f);
  cudaFree(dA);cudaFree(dB);cudaFree(dC);
  return 0;
}
