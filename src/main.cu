// Driver: correctness vs cuBLAS + TFLOP/s (and % of the cuBLAS ceiling) per kernel.
//
// Appends one row per kernel to a self-describing CSV (device, sm, M, N, K, ...),
// so a size sweep — and separate runs on different GPUs (H100 sm_90 / Blackwell
// sm_120) — accumulate into one file that analysis/plot.py can chart directly.
//
//   ./gemm_bench [M] [N] [K] [out_csv] [kernels]
//     out_csv defaults to results/bench.csv
//     kernels: optional comma-separated filter (e.g. "mma_pipe,cublas_tc");
//              default runs every kernel
#include "util.cuh"
#include <vector>
#include <string>
#include <cstdio>
#include <cstring>
#include <unistd.h>

void launch_naive    (const float*,const float*,float*,int,int,int);
void launch_tiled    (const float*,const float*,float*,int,int,int);
void launch_wmma     (const float*,const float*,float*,int,int,int);
void launch_cublas    (const float*,const float*,float*,int,int,int); // cublasSgemm  (FP32, CUDA cores)
void launch_cublas_tf32(const float*,const float*,float*,int,int,int);// cublasGemmEx (FP32 in, TF32 compute, Tensor Cores)
void launch_cublas_tc (const float*,const float*,float*,int,int,int); // cublasGemmEx (FP16 in / FP32 acc, Tensor Cores)
// hand-written mma.sync ablation ladder (gemm_mma.cu) — Phase 2 roadmap item
void launch_mma_base    (const float*,const float*,float*,int,int,int);
void launch_mma_swizzle (const float*,const float*,float*,int,int,int);
void launch_mma_vec     (const float*,const float*,float*,int,int,int);
void launch_mma_pipe    (const float*,const float*,float*,int,int,int);
void launch_mma_warptile(const float*,const float*,float*,int,int,int);

struct Kern{ const char* name; void(*fn)(const float*,const float*,float*,int,int,int); };

int main(int argc,char**argv){
  int M=argc>1?atoi(argv[1]):4096, N=argc>2?atoi(argv[2]):4096, K=argc>3?atoi(argv[3]):4096;
  const char* out = argc>4?argv[4]:"results/bench.csv";
  const char* filter = argc>5?argv[5]:nullptr;   // comma-separated kernel names, or all
  size_t szA=sizeof(float)*M*K, szB=sizeof(float)*K*N, szC=sizeof(float)*M*N;
  std::vector<float> hA(M*K),hB(K*N),hRef(M*N);
  fill_rand(hA.data(),M*K); fill_rand(hB.data(),K*N);
  float *dA,*dB,*dC; CUDA_CHECK(cudaMalloc(&dA,szA));CUDA_CHECK(cudaMalloc(&dB,szB));CUDA_CHECK(cudaMalloc(&dC,szC));
  CUDA_CHECK(cudaMemcpy(dA,hA.data(),szA,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB,hB.data(),szB,cudaMemcpyHostToDevice));

  cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
  int sm = prop.major*10 + prop.minor;

  // cuBLAS reference output for the correctness check.
  launch_cublas(dA,dB,dC,M,N,K); CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hRef.data(),dC,szC,cudaMemcpyDeviceToHost));

  // precision ladder, all on the same card:
  //   "cublas"      = cublasSgemm  (FP32, CUDA cores)            — non-TC baseline
  //   "cublas_tf32" = cublasGemmEx (FP32 in, TF32 compute, TC)   — TF32 Tensor Core rung
  //   "cublas_tc"   = cublasGemmEx (FP16 in / FP32 acc, TC)      — FP16 Tensor Core ceiling
  Kern kerns[]={{"naive",launch_naive},{"tiled",launch_tiled},{"wmma",launch_wmma},
                {"mma_base",launch_mma_base},{"mma_swizzle",launch_mma_swizzle},
                {"mma_vec",launch_mma_vec},{"mma_pipe",launch_mma_pipe},
                {"mma_warptile",launch_mma_warptile},
                {"cublas",launch_cublas},{"cublas_tf32",launch_cublas_tf32},{"cublas_tc",launch_cublas_tc}};
  struct Row{ const char* name; float ms, tf; double err; };
  std::vector<Row> rows;
  std::vector<float> hC(M*N);
  double cublas_tf=0;
  // kernel filter: ",name," substring match against ",<filter>," so names can't
  // partially match each other (e.g. "mma_pipe" vs "mma_pipe2")
  auto selected=[&](const char* name){
    if(!filter) return true;
    std::string hay=std::string(",")+filter+",", needle=std::string(",")+name+",";
    return hay.find(needle)!=std::string::npos;
  };
  for(auto&k:kerns){
    if(!selected(k.name)) continue;
    Timer t; k.fn(dA,dB,dC,M,N,K); CUDA_CHECK(cudaDeviceSynchronize());          // warmup
    t.start(); for(int i=0;i<10;i++) k.fn(dA,dB,dC,M,N,K); float ms=t.stop()/10.f;
    CUDA_CHECK(cudaMemcpy(hC.data(),dC,szC,cudaMemcpyDeviceToHost));
    double err=max_abs_err(hC.data(),hRef.data(),M*N);
    float tf=(float)tflops(M,N,K,ms);
    if(strcmp(k.name,"cublas")==0) cublas_tf=tf;
    rows.push_back({k.name,ms,tf,err});
  }

  bool exists = (access(out,F_OK)==0);
  FILE* f=fopen(out,"a");
  if(!f){ fprintf(stderr,"cannot open %s for writing\n",out); return 1; }
  if(!exists) fprintf(f,"device,sm,M,N,K,kernel,ms,tflops,pct_of_cublas,max_abs_err\n");
  printf("M=%d N=%d K=%d on %s (sm_%d)\n",M,N,K,prop.name,sm);
  for(auto&r:rows){
    double pct = cublas_tf>0 ? 100.0*r.tf/cublas_tf : 0.0;
    printf("%-8s %8.3f ms  %7.2f TFLOP/s  %6.1f%% cuBLAS  maxerr=%.3g\n",r.name,r.ms,r.tf,pct,r.err);
    fprintf(f,"\"%s\",%d,%d,%d,%d,%s,%.3f,%.2f,%.1f,%.3g\n",prop.name,sm,M,N,K,r.name,r.ms,r.tf,pct,r.err);
  }
  fclose(f);
  cudaFree(dA);cudaFree(dB);cudaFree(dC);
  return 0;
}
