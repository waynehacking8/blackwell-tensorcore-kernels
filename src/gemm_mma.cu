// Hand-written mma.sync GEMM (roadmap Phase 2, "Hand-written mma.sync" item).
// FP16 in / FP32 accumulate, raw PTX mma.sync.aligned.m16n8k16 + ldmatrix —
// no WMMA wrapper, so register layout, shared-memory addressing (swizzle), and
// the smem->register pipeline are all under explicit control.
//
// One templated kernel generates the whole ablation ladder; each launcher below
// is one bench.csv row, adding exactly one optimization (Boehm-style discipline):
//   mma_base      128x128 CTA, 2x2 per-warp register tiling (16 warps x 32x32),
//                 naive row-major smem, scalar loads, single-stage
//   mma_swizzle   + XOR-swizzled (bank-conflict-free) smem layout for ldmatrix
//   mma_vec       + 16-byte vectorized cp.async global->shared loads
//   mma_pipe      + multi-stage cp.async pipeline (depth via MMA_STAGES; default 2,
//                 the sweep winner on sm_120 — see results/mma_stage_sweep.csv)
//   mma_warptile  + 64x64 per-warp register tile (4 warps; Boehm's "warptiling"
//                 analog — raises mma:ldmatrix ratio from 2:1 to 4:1, attacking
//                 the MIO-queue-full stall the ncu profile identified)
// Assumes M,N multiples of BN/BM and K a multiple of BK (true for the 512..8192 sweep).
#include <cuda_fp16.h>
#include <cstdlib>
#include "util.cuh"
#include "mma_ptx.cuh"

template <int BM, int BN, int BK, int WM, int WN, int STAGES, bool VEC, bool SWZ>
__global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32)
gemm_mma_t(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C,
           int M, int N, int K) {
  constexpr int WARPS_N = BN / WN;
  constexpr int NTHREADS = (BM / WM) * WARPS_N * 32;
  constexpr int MITER = WM / 16;       // m16n8k16 tiles per warp, M direction
  constexpr int NITER = WN / 8;        //                          N direction
  constexpr int A_CPR = BK / 8;        // 16-byte chunks per A smem row
  constexpr int B_CPR = BN / 8;        //                       B smem row
  constexpr int SLAB = BM * BK + BK * BN;  // halves per pipeline stage

  extern __shared__ half smem[];
  half* As[STAGES];
  half* Bs[STAGES];
#pragma unroll
  for (int s = 0; s < STAGES; s++) { As[s] = smem + s * SLAB; Bs[s] = smem + s * SLAB + BM * BK; }

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / WARPS_N, wn = warp % WARPS_N;
  const int blockRow = blockIdx.y * BM, blockCol = blockIdx.x * BN;

  float acc[MITER][NITER][4] = {};

  // ---- global -> shared: one BMxBK A slab + one BKxBN B slab ----
  auto load_slab = [&](int s, int k0) {
    if (VEC) {
      constexpr int A_CH = BM * BK / 8, B_CH = BK * BN / 8;
#pragma unroll
      for (int i = tid; i < A_CH; i += NTHREADS) {
        int row = i / A_CPR, chunk = i % A_CPR, gRow = blockRow + row;
        half* dst = &As[s][smem_off<A_CPR, SWZ>(row, chunk * 8)];
        if (gRow < M) cp_async_16(dst, &A[(size_t)gRow * K + k0 + chunk * 8]);
        else          *reinterpret_cast<float4*>(dst) = make_float4(0, 0, 0, 0);
      }
#pragma unroll
      for (int i = tid; i < B_CH; i += NTHREADS) {
        int row = i / B_CPR, chunk = i % B_CPR, gCol = blockCol + chunk * 8;
        half* dst = &Bs[s][smem_off<B_CPR, SWZ>(row, chunk * 8)];
        if (gCol < N) cp_async_16(dst, &B[(size_t)(k0 + row) * N + gCol]);
        else          *reinterpret_cast<float4*>(dst) = make_float4(0, 0, 0, 0);
      }
    } else {
      // scalar half-at-a-time loads (the ablation baseline cp.async replaces)
#pragma unroll 1
      for (int i = tid; i < BM * BK; i += NTHREADS) {
        int row = i / BK, col = i % BK, gRow = blockRow + row;
        As[s][smem_off<A_CPR, SWZ>(row, col)] = gRow < M ? A[(size_t)gRow * K + k0 + col] : __float2half(0.f);
      }
#pragma unroll 1
      for (int i = tid; i < BK * BN; i += NTHREADS) {
        int row = i / BN, col = i % BN, gCol = blockCol + col;
        Bs[s][smem_off<B_CPR, SWZ>(row, col)] = gCol < N ? B[(size_t)(k0 + row) * N + gCol] : __float2half(0.f);
      }
    }
  };

  const int numSlabs = K / BK;

  // prefetch STAGES-1 slabs (no-op at STAGES==1); commit per stage to keep the
  // cp.async group count aligned with the wait in the main loop
#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if (s < numSlabs) load_slab(s, s * BK);
    if (VEC) cp_async_commit();
  }

  int lr, lc;
  ldmatrix_lane_rc(lane, lr, lc);

  for (int t = 0; t < numSlabs; t++) {
    int cur = t % STAGES, fetch = t + STAGES - 1;
    if (fetch < numSlabs) load_slab(fetch % STAGES, fetch * BK);
    if (VEC) { cp_async_commit(); cp_async_wait<STAGES - 1>(); }
    __syncthreads();

    const half* As_c = As[cur];
    const half* Bs_c = Bs[cur];

#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      // A fragments: one ldmatrix.x4 per m16 tile
      unsigned afrag[MITER][4];
#pragma unroll
      for (int im = 0; im < MITER; im++)
        ldmatrix_x4(afrag[im], &As_c[smem_off<A_CPR, SWZ>(wm * WM + im * 16 + lr, kk + lc)]);

      // B fragments: one ldmatrix.x4.trans per PAIR of n8 tiles
      unsigned bfrag[NITER][2];
#pragma unroll
      for (int in = 0; in < NITER; in += 2) {
        unsigned r4[4];
        ldmatrix_x4_trans(r4, &Bs_c[smem_off<B_CPR, SWZ>(kk + lr, wn * WN + in * 8 + lc)]);
        bfrag[in][0] = r4[0]; bfrag[in][1] = r4[1];
        bfrag[in + 1][0] = r4[2]; bfrag[in + 1][1] = r4[3];
      }

#pragma unroll
      for (int im = 0; im < MITER; im++)
#pragma unroll
        for (int in = 0; in < NITER; in++)
          mma_m16n8k16(acc[im][in], afrag[im], bfrag[in]);
    }
    __syncthreads();
  }

  // ---- epilogue: accumulator registers -> C (FP32, row-major) ----
#pragma unroll
  for (int im = 0; im < MITER; im++)
#pragma unroll
    for (int in = 0; in < NITER; in++) {
      int row = blockRow + wm * WM + im * 16 + lane / 4;
      int col = blockCol + wn * WN + in * 8 + 2 * (lane % 4);
      if (row < M && col < N)
        *reinterpret_cast<float2*>(&C[(size_t)row * N + col]) = make_float2(acc[im][in][0], acc[im][in][1]);
      if (row + 8 < M && col < N)
        *reinterpret_cast<float2*>(&C[(size_t)(row + 8) * N + col]) = make_float2(acc[im][in][2], acc[im][in][3]);
    }
}

// ---------------------------------------------------------------------------
// FP16 staging: cast A,B to FP16 once, outside the timed region — identical
// methodology to launch_wmma / launch_cublas_tc so the comparison stays fair.
// ---------------------------------------------------------------------------
__global__ void f2h_mma(const float* in, half* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2half(in[i]);
}

static half *g_Ah = nullptr, *g_Bh = nullptr;
static const float *g_cA = nullptr, *g_cB = nullptr;
static int g_M = 0, g_N = 0, g_K = 0;

static void stage_fp16(const float* A, const float* B, int M, int N, int K) {
  if (M != g_M || N != g_N || K != g_K) {
    cudaFree(g_Ah); cudaFree(g_Bh);
    g_Ah = g_Bh = nullptr; g_cA = g_cB = nullptr;
  }
  if (!g_Ah) {
    CUDA_CHECK(cudaMalloc(&g_Ah, sizeof(half) * (size_t)M * K));
    CUDA_CHECK(cudaMalloc(&g_Bh, sizeof(half) * (size_t)K * N));
    g_M = M; g_N = N; g_K = K;
  }
  if (A != g_cA) { size_t n = (size_t)M * K; f2h_mma<<<((int)((n + 255) / 256)), 256>>>(A, g_Ah, (int)n); g_cA = A; }  // size_t cast prevents int overflow at large M*K
  if (B != g_cB) { size_t n = (size_t)K * N; f2h_mma<<<((int)((n + 255) / 256)), 256>>>(B, g_Bh, (int)n); g_cB = B; }
}

template <int BM, int BN, int BK, int WM, int WN, int ST, bool VEC, bool SWZ>
static void run_mma(const half* A, const half* B, float* C, int M, int N, int K) {
  size_t sh = (size_t)ST * (BM * BK + BK * BN) * sizeof(half);
  static bool attr_set = false;
  if (!attr_set && sh > 48 * 1024) {
    CUDA_CHECK(cudaFuncSetAttribute(gemm_mma_t<BM, BN, BK, WM, WN, ST, VEC, SWZ>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
    attr_set = true;
  }
  dim3 t((BM / WM) * (BN / WN) * 32), g((N + BN - 1) / BN, (M + BM - 1) / BM);
  gemm_mma_t<BM, BN, BK, WM, WN, ST, VEC, SWZ><<<g, t, sh>>>(A, B, C, M, N, K);
}

// Pipeline depth for the pipe/warptile rows; MMA_STAGES=2|3|4 overrides (sweep).
// Default 2: the measured sweep winner on sm_120 (2 > 3 > 4; the wmma kernel
// preferred 3 — with raw mma.sync the math is fast enough that deeper pipelines
// just cost occupancy). Raw sweep: results/mma_stage_sweep.csv.
static int env_stages(int dflt) {
  const char* s = getenv("MMA_STAGES");
  if (!s) return dflt;
  int v = atoi(s);
  return (v >= 2 && v <= 4) ? v : dflt;
}

// ---- the ablation ladder: one launcher per bench.csv row ----
void launch_mma_base(const float* A, const float* B, float* C, int M, int N, int K) {
  stage_fp16(A, B, M, N, K);
  run_mma<128, 128, 32, 32, 32, 1, false, false>(g_Ah, g_Bh, C, M, N, K);
}

void launch_mma_swizzle(const float* A, const float* B, float* C, int M, int N, int K) {
  stage_fp16(A, B, M, N, K);
  run_mma<128, 128, 32, 32, 32, 1, false, true>(g_Ah, g_Bh, C, M, N, K);
}

void launch_mma_vec(const float* A, const float* B, float* C, int M, int N, int K) {
  stage_fp16(A, B, M, N, K);
  run_mma<128, 128, 32, 32, 32, 1, true, true>(g_Ah, g_Bh, C, M, N, K);
}

void launch_mma_pipe(const float* A, const float* B, float* C, int M, int N, int K) {
  stage_fp16(A, B, M, N, K);
  switch (env_stages(2)) {
    case 3:  run_mma<128, 128, 32, 32, 32, 3, true, true>(g_Ah, g_Bh, C, M, N, K); break;
    case 4:  run_mma<128, 128, 32, 32, 32, 4, true, true>(g_Ah, g_Bh, C, M, N, K); break;
    default: run_mma<128, 128, 32, 32, 32, 2, true, true>(g_Ah, g_Bh, C, M, N, K); break;
  }
}

void launch_mma_warptile(const float* A, const float* B, float* C, int M, int N, int K) {
  stage_fp16(A, B, M, N, K);
  switch (env_stages(2)) {
    case 3:  run_mma<128, 128, 32, 64, 64, 3, true, true>(g_Ah, g_Bh, C, M, N, K); break;
    case 4:  run_mma<128, 128, 32, 64, 64, 4, true, true>(g_Ah, g_Bh, C, M, N, K); break;
    default: run_mma<128, 128, 32, 64, 64, 2, true, true>(g_Ah, g_Bh, C, M, N, K); break;
  }
}
