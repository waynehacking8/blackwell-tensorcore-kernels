cmake_minimum_required(VERSION 3.24)
project(blackwell_tensorcore_kernels LANGUAGES CXX CUDA)
set(CMAKE_CUDA_STANDARD 17)
# Hopper (H100) + Blackwell (RTX Pro 6000). Override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90 120)
endif()

set(BENCH_SOURCES
  src/main.cu src/gemm_naive.cu src/gemm_tiled.cu src/gemm_wmma.cu src/gemm_mma.cu
  src/gemm_mma_fp8.cu src/cublaslt_fp8.cu
  src/reference.cu src/cublas_tc.cu src/cublas_tf32.cu)

# Phase 3: FP4 mma kinds (kind::f8f6f4 / kind::mxf4) need the sm_120a feature target.
# The FP8 kernel works on any sm_89+ build; FP4/MXFP4 rows register only on 120a.
#   cmake -DCMAKE_CUDA_ARCHITECTURES=120a ..
if(CMAKE_CUDA_ARCHITECTURES MATCHES "120a")
  add_compile_definitions(BUILD_SM120A)
  message(STATUS "FP4/MXFP4 kernel rows ENABLED (sm_120a)")
else()
  message(STATUS "FP4/MXFP4 kernel rows disabled (build with -DCMAKE_CUDA_ARCHITECTURES=120a)")
endif()

# Phase 2.5: CUTLASS 3.x wgmma kernel — Hopper only (sm_90a), needs CUTLASS headers.
#   cmake -DCMAKE_CUDA_ARCHITECTURES=90a -DCUTLASS_DIR=/path/to/cutlass ..
# wgmma requires the 90a feature target; the kernel is skipped on plain 90 / 120 builds.
set(CUTLASS_DIR "" CACHE PATH "Path to a CUTLASS >= 3.4 checkout (enables the cutlass_sm90 kernel row)")
if(CUTLASS_DIR AND CMAKE_CUDA_ARCHITECTURES MATCHES "90a")
  list(APPEND BENCH_SOURCES src/cutlass_sm90.cu)
  add_compile_definitions(HAVE_CUTLASS_SM90)
  message(STATUS "cutlass_sm90 kernel ENABLED (CUTLASS at ${CUTLASS_DIR})")
else()
  message(STATUS "cutlass_sm90 kernel disabled (set -DCUTLASS_DIR=... and -DCMAKE_CUDA_ARCHITECTURES=90a)")
endif()

add_executable(gemm_bench ${BENCH_SOURCES})
target_include_directories(gemm_bench PRIVATE include)
if(CUTLASS_DIR AND CMAKE_CUDA_ARCHITECTURES MATCHES "90a")
  target_include_directories(gemm_bench PRIVATE
    ${CUTLASS_DIR}/include ${CUTLASS_DIR}/tools/util/include)
endif()
target_link_libraries(gemm_bench PRIVATE cublas cublasLt)
