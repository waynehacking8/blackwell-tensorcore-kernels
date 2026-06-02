// CUTLASS 3.x Hopper GEMM: TMA + wgmma (warpgroup MMA), warp-specialized cooperative
// schedule. FP16 in / FP32 accumulate — same precision contract as gemm_wmma.cu and
// cublas_tc.cu, so all three are directly comparable rows in bench.csv.
//
// Phase 2.5 (roadmap): the H100 result established that the WMMA API cannot exceed ~65%
// of peak on Hopper because it cannot emit wgmma. This kernel is the constructive proof:
// the same operation built with CUTLASS's CollectiveBuilder (which *does* emit wgmma via
// warpgroup-wide MMA + TMA loads) should approach nvjet's ~77%-of-peak.
//
// Build: requires CUTLASS 3.x headers (-DCUTLASS_DIR=...) and sm_90a
// (CMAKE_CUDA_ARCHITECTURES=90a). Excluded from sm_120-only builds — wgmma is
// Hopper-only; Blackwell replaces it with tcgen05.
#include <cuda_fp16.h>
#include "util.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace {

// A: M×K row-major (our layout). B: our K×N row-major buffer. In CUTLASS 3.x stride
// conventions the B operand is indexed (n, k); RowMajor here maps B(n,k) -> ptr[k*N + n]
// == our row-major K×N buffer (verified against the cuBLAS reference: ColumnMajor reads
// the transposed product). C/D: M×N row-major FP32.
using ElementA = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;
using ElementB = cutlass::half_t;
using LayoutB = cutlass::layout::RowMajor;
using ElementC = float;
using LayoutC = cutlass::layout::RowMajor;
using ElementAccumulator = float;

// 128x256 CTA tile, 64-deep K slab, 2x1 thread-block cluster: the shape CUTLASS's own
// Hopper FP16 profiler picks for large square GEMMs.
using TileShape = Shape<_128, _256, _64>;
using ClusterShape = Shape<_2, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, 4,
    ElementC, LayoutC, 4,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 8,
    ElementB, LayoutB, 8,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

}  // namespace

// At file scope (not in the anonymous namespace): nvcc's kernel-registration stub cannot
// reference __global__ functions declared inside anonymous namespaces.
__global__ void f2h_cutlass(const float* in, cutlass::half_t* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = cutlass::half_t(in[i]);
}

void launch_cutlass_sm90(const float* A, const float* B, float* C, int M, int N, int K) {
  // Stage A,B into FP16 once and reuse across repeated calls (same methodology as
  // cublas_tc.cu / gemm_wmma.cu: the cast is outside the timed region).
  static cutlass::half_t *Ah = nullptr, *Bh = nullptr;
  static const float *cA = nullptr, *cB = nullptr;
  static int cM = 0, cN = 0, cK = 0;
  static Gemm* gemm = nullptr;
  static uint8_t* workspace = nullptr;

  if (M != cM || N != cN || K != cK) {
    cudaFree(Ah); cudaFree(Bh); cudaFree(workspace);
    delete gemm;
    Ah = Bh = nullptr; workspace = nullptr; gemm = nullptr;
    cA = cB = nullptr;
  }
  if (!Ah) {
    CUDA_CHECK(cudaMalloc(&Ah, sizeof(cutlass::half_t) * (size_t)M * K));
    CUDA_CHECK(cudaMalloc(&Bh, sizeof(cutlass::half_t) * (size_t)K * N));
    cM = M; cN = N; cK = K;
  }
  if (A != cA) { int n = M * K; f2h_cutlass<<<(n + 255) / 256, 256>>>(A, Ah, n); cA = A; }
  if (B != cB) { int n = K * N; f2h_cutlass<<<(n + 255) / 256, 256>>>(B, Bh, n); cB = B; }

  // Build the GEMM object + workspace once per problem size; the timed region is
  // gemm->run() only (cuBLAS gets the same treatment via its cached handle).
  if (!gemm) {
    gemm = new Gemm;
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {Ah, stride_A, Bh, stride_B},
        {{1.0f, 0.0f}, C, stride_C, C, stride_D}};
    size_t ws_size = Gemm::get_workspace_size(args);
    if (ws_size) CUDA_CHECK(cudaMalloc(&workspace, ws_size));
    auto status = gemm->initialize(args, workspace);
    if (status != cutlass::Status::kSuccess) {
      fprintf(stderr, "CUTLASS initialize failed: %s\n", cutlassGetStatusString(status));
      return;
    }
  }
  auto status = gemm->run();
  if (status != cutlass::Status::kSuccess)
    fprintf(stderr, "CUTLASS run failed: %s\n", cutlassGetStatusString(status));
}
