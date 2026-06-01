cmake_minimum_required(VERSION 3.24)
project(blackwell_tensorcore_kernels LANGUAGES CXX CUDA)
set(CMAKE_CUDA_STANDARD 17)
# Hopper (H100) + Blackwell (RTX Pro 6000). Override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90 120)
endif()
add_executable(gemm_bench
  src/main.cu src/gemm_naive.cu src/gemm_tiled.cu src/gemm_wmma.cu
  src/reference.cu src/cublas_tc.cu)
target_include_directories(gemm_bench PRIVATE include)
target_link_libraries(gemm_bench PRIVATE cublas)
