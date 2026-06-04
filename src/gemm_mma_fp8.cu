// Phase 3 — Blackwell low-precision GEMM on the sm_120 mma path (FP8 / FP4 / MXFP4).
// Built directly on the Phase 2 mma.sync kernel (gemm_mma.cu): same 128x128 CTA tile,
// 64x64 warp tile, XOR-swizzled smem, 16-byte cp.async, 2-stage pipeline — only the
// instruction (and operand width) changes — plus a register-pipelined mainloop
// (fragments for k-step i+1 load while the mma for k-step i runs):
//
//   mma_fp8     mma.m16n8k32 e4m3   QMMA.16832 SASS   1 byte/elem    sm_120
//   mma_fp4     mma.m16n8k32 e2m1   QMMA.16832 SASS   1 byte/elem    sm_120a (kind::f8f6f4)
//   mma_mxfp4   mma.m16n8k64 e2m1   OMMA.SF.16864     0.5 byte/elem  sm_120a (kind::mxf4)
//
// Operand layouts (the FP8/FP4 hardware path requires "TN"): A row-major (M x K),
// B *transposed* (N x K). Both are quantized once outside the timed region (same
// staging methodology as the FP16 kernels / cuBLAS-TC baseline). B^T also matches
// cuBLASLt's FP8 layout requirement, so the comparison is layout-for-layout fair.
//
// Accuracy: inputs are uniform [-0.5, 0.5]. FP8 e4m3 covers that range natively.
// FP4 e2m1 has only levels {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}: without scaling, all
// inputs collapse onto {0, ±0.5} - so A and B are pre-scaled by 8 (into [-4, 4]),
// and the epilogue divides by 64. This is per-tensor scaling: the documented accuracy
// is what FP4 can do WITHOUT per-block (MX) scale calibration; with uniform input
// data per-block scales would be identical anyway.
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include "util.cuh"
#include "mma_ptx.cuh"

namespace lowprec {

enum class Fmt { FP8, FP4, MXFP4 };

// ---------------------------------------------------------------------------
// Quantization kernels: FP32 -> low precision (staging, outside the timed loop)
// ---------------------------------------------------------------------------
// FP32 -> FP8 (E4M3). [-0.5,0.5] fits e4m3 range; no scaling needed.
__global__ void q_fp8(const float* in, uint8_t* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __nv_fp8_e4m3(in[i]).__x;
}
// FP32 -> FP8 transposed: out[n*K + k] = in[k*N + n]
__global__ void q_fp8_t(const float* in, uint8_t* out, int K, int N) {
  int k = blockIdx.x * blockDim.x + threadIdx.x, n = blockIdx.y * blockDim.y + threadIdx.y;
  if (k < K && n < N) out[(size_t)n * K + k] = __nv_fp8_e4m3(in[(size_t)k * N + n]).__x;
}
// FP32 -> FP4 (E2M1) in an 8-bit container (low 4 bits), pre-scaled by SCALE.
__device__ __forceinline__ uint8_t f32_to_e2m1(float v) {
  // E2M1 levels: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (+ sign)
  const float lv[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float a = fabsf(v);
  int best = 0;
  float bd = fabsf(a - lv[0]);
  for (int i = 1; i < 8; i++) {
    float d = fabsf(a - lv[i]);
    if (d < bd) { bd = d; best = i; }
  }
  return (uint8_t)((v < 0.f ? 8 : 0) | best);
}
// kind::f8f6f4 stores FP4/FP6 values in bits [5:2] of the 8-bit container (FP6 alignment);
// plain low-nibble packing reads as garbage. MXFP4 (packed, 2/byte) is NOT shifted.
__global__ void q_fp4(const float* in, uint8_t* out, int n, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = (uint8_t)(f32_to_e2m1(in[i] * scale) << 2);
}
__global__ void q_fp4_t(const float* in, uint8_t* out, int K, int N, float scale) {
  int k = blockIdx.x * blockDim.x + threadIdx.x, n = blockIdx.y * blockDim.y + threadIdx.y;
  if (k < K && n < N) out[(size_t)n * K + k] = (uint8_t)(f32_to_e2m1(in[(size_t)k * N + n] * scale) << 2);
}
// packed variants for MXFP4: two e2m1 per byte (element i in low nibble, i+1 in high)
__global__ void q_mxfp4(const float* in, uint8_t* out, int n, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // output byte index; covers elements 2i, 2i+1
  if (i < n / 2)
    out[i] = (uint8_t)(f32_to_e2m1(in[2 * i] * scale) | (f32_to_e2m1(in[2 * i + 1] * scale) << 4));
}
__global__ void q_mxfp4_t(const float* in, uint8_t* out, int K, int N, float scale) {
  int k2 = blockIdx.x * blockDim.x + threadIdx.x, n = blockIdx.y * blockDim.y + threadIdx.y;
  if (k2 < K / 2 && n < N)
    out[(size_t)n * (K / 2) + k2] = (uint8_t)(f32_to_e2m1(in[(size_t)(2 * k2) * N + n] * scale) |
                                              (f32_to_e2m1(in[(size_t)(2 * k2 + 1) * N + n] * scale) << 4));
}

// ---------------------------------------------------------------------------
// The kernel. Template on format; layout constants are in BYTES so FP8 (1 B/elem)
// and packed MXFP4 (0.5 B/elem) share all addressing.
//   BM=128, BN=128, KB (bytes per slab row) = 64 -> 4 chunks/row -> CPR=4 swizzle.
//   K-step: FP8/FP4 = 32 elems (32 B); MXFP4 = 64 elems (32 B) -> always 32 B.
//   2 k-steps per slab, both A and B^T 128x64 B = 16 KB/stage.
// ---------------------------------------------------------------------------
template <Fmt FMT, int STAGES, int BM = 128, int BN = 128>
__global__ void __launch_bounds__((BM / 64) * (BN / 64) * 32) gemm_lowprec_t(
    const uint8_t* __restrict__ A,   // M x K, row-major (packed for MXFP4)
    const uint8_t* __restrict__ Bt,  // N x K, row-major = B transposed (packed for MXFP4)
    float* __restrict__ C, int M, int N, int K) {
  constexpr int KB = 64;                                      // bytes per smem row
  constexpr int EPB = (FMT == Fmt::MXFP4) ? 2 : 1;            // elements per byte
  constexpr int KSTEP_B = 32;                                 // bytes per k-step (all formats)
  constexpr int WM = 64, WN = 64;                             // warp tile
  constexpr int NWARP = (BM / WM) * (BN / WN);                // 4
  constexpr int NTHREADS = NWARP * 32;
  constexpr int MITER = WM / 16, NITER = WN / 8;
  constexpr int CPR = KB / 16;                                // 16-byte chunks per row = 4
  constexpr int SLAB = (BM + BN) * KB;                        // bytes per stage

  extern __shared__ uint8_t smem[];
  uint8_t* As[STAGES];
  uint8_t* Bs[STAGES];
#pragma unroll
  for (int s = 0; s < STAGES; s++) { As[s] = smem + s * SLAB; Bs[s] = smem + s * SLAB + BM * KB; }

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / (BN / WN), wn = warp % (BN / WN);
  const int blockRow = blockIdx.y * BM, blockCol = blockIdx.x * BN;
  const int Kb = K / EPB;  // K extent in bytes

  float acc[MITER][NITER][4] = {};

  // global -> shared: BM rows of A and BN rows of B^T, KB bytes per row per slab
  auto load_slab = [&](int s, int kb0) {
#pragma unroll
    for (int i = tid; i < BM * CPR; i += NTHREADS) {
      int row = i / CPR, chunk = i % CPR;
      cp_async_16(&As[s][smem_off_b<CPR, true>(row, chunk * 16)],
                  &A[(size_t)(blockRow + row) * Kb + kb0 + chunk * 16]);
    }
#pragma unroll
    for (int i = tid; i < BN * CPR; i += NTHREADS) {
      int row = i / CPR, chunk = i % CPR;
      cp_async_16(&Bs[s][smem_off_b<CPR, true>(row, chunk * 16)],
                  &Bt[(size_t)(blockCol + row) * Kb + kb0 + chunk * 16]);
    }
  };

  const int numSlabs = Kb / KB;

  int lr, lc;
  ldmatrix_lane_rc(lane, lr, lc);
  const int lcb = lc * 2;  // ldmatrix lane column offset in BYTES (b16 units -> bytes)

  // Register-pipelined mainloop (xmma/CUTLASS style): fragments for k-step i+1 are
  // loaded from smem while the mma for k-step i runs, so ldmatrix latency hides
  // behind QMMA/OMMA latency. Double-buffered fragments (~64 extra regs/thread,
  // still <=255 -> occupancy unchanged at 2 CTAs/SM).
  unsigned afrag[2][MITER][4], bfrag[2][NITER][2];

  auto load_frags = [&](int buf, const uint8_t* As_c, const uint8_t* Bs_c, int kk) {
#pragma unroll
    for (int im = 0; im < MITER; im++)
      ldmatrix_x4(afrag[buf][im], (const half*)&As_c[smem_off_b<CPR, true>(wm * WM + im * 16 + lr, kk + lcb)]);
#pragma unroll
    for (int in = 0; in < NITER; in += 2) {
      unsigned r4[4];
      ldmatrix_x4(r4, (const half*)&Bs_c[smem_off_b<CPR, true>(wn * WN + in * 8 + lr, kk + lcb)]);
      // B^T stored N-major in smem. ldmatrix.x4 (non-trans) loads 4 tiles:
      // tiles {0,2} cover n-tile 0, tiles {1,3} cover n-tile 1.
      // Re-interleave to match mma.m16n8k32.row.col B-operand layout per PTX ISA §9.7.13.4.
      bfrag[buf][in][0] = r4[0];     bfrag[buf][in][1] = r4[2];
      bfrag[buf][in + 1][0] = r4[1]; bfrag[buf][in + 1][1] = r4[3];
    }
  };
  auto mma_step = [&](int buf) {
#pragma unroll
    for (int im = 0; im < MITER; im++)
#pragma unroll
      for (int in = 0; in < NITER; in++) {
        if (FMT == Fmt::FP8)        mma_m16n8k32_e4m3(acc[im][in], afrag[buf][im], bfrag[buf][in]);
        else if (FMT == Fmt::FP4)   mma_m16n8k32_e2m1(acc[im][in], afrag[buf][im], bfrag[buf][in]);
        else                        mma_m16n8k64_mxf4(acc[im][in], afrag[buf][im], bfrag[buf][in]);
      }
  };

  // prologue: stage the first STAGES slabs, then preload k-step 0 fragments of slab 0
#pragma unroll
  for (int s = 0; s < STAGES; s++) {
    if (s < numSlabs) load_slab(s, (size_t)s * KB);
    cp_async_commit();
  }
  cp_async_wait<STAGES - 1>();
  __syncthreads();
  load_frags(0, As[0], Bs[0], 0);

  int buf = 0;
  for (int t = 0; t < numSlabs; t++) {
    const uint8_t* As_c = As[t % STAGES];
    const uint8_t* Bs_c = Bs[t % STAGES];

    // first k-step: prefetch this slab's second k-step, then run mma on the current one
    load_frags(buf ^ 1, As_c, Bs_c, KSTEP_B);
    mma_step(buf);
    buf ^= 1;

    // second k-step: cross-slab boundary
    // Terminal iteration: skip async copy, mma_step uses fragments loaded in previous iteration
    if (t + 1 < numSlabs) {
      cp_async_wait<STAGES - 2>();  // slab t+1 resident (STAGES-2 of the in-flight groups may stay pending)
      __syncthreads();              // all warps done reading slab t's smem
      if (t + STAGES < numSlabs) {  // overwrite slab t's buffer with slab t+STAGES
        load_slab(t % STAGES, (size_t)(t + STAGES) * KB);
      }
      cp_async_commit();
      // prefetch first k-step fragments of slab t+1 while computing this slab's last step
      load_frags(buf ^ 1, As[(t + 1) % STAGES], Bs[(t + 1) % STAGES], 0);
    }
    mma_step(buf);
    buf ^= 1;
  }

  // FP4 inputs were pre-scaled by 8 -> accumulators carry 8*8 = 64x
  const float unscale = (FMT == Fmt::FP8) ? 1.f : (1.f / 64.f);

#pragma unroll
  for (int im = 0; im < MITER; im++)
#pragma unroll
    for (int in = 0; in < NITER; in++) {
      int row = blockRow + wm * WM + im * 16 + lane / 4;
      int col = blockCol + wn * WN + in * 8 + 2 * (lane % 4);
      if (row < M && col < N)
        *reinterpret_cast<float2*>(&C[(size_t)row * N + col]) =
            make_float2(acc[im][in][0] * unscale, acc[im][in][1] * unscale);
      if (row + 8 < M && col < N)
        *reinterpret_cast<float2*>(&C[(size_t)(row + 8) * N + col]) =
            make_float2(acc[im][in][2] * unscale, acc[im][in][3] * unscale);
    }
}

// ---------------------------------------------------------------------------
// Staging: quantize A (row-major) and B (transposed) once per (input, format).
// ---------------------------------------------------------------------------
constexpr float FP4_SCALE = 8.f;  // [-0.5,0.5] -> [-4,4]: use the e2m1 grid

struct Staged {
  uint8_t *A = nullptr, *Bt = nullptr;
  const float *srcA = nullptr, *srcB = nullptr;
  int M = 0, N = 0, K = 0;
};

static Staged g_stage[3];  // one per Fmt

static void stage(Fmt fmt, const float* A, const float* B, int M, int N, int K) {
  Staged& s = g_stage[(int)fmt];
  int epb = (fmt == Fmt::MXFP4) ? 2 : 1;
  if (M != s.M || N != s.N || K != s.K) {
    cudaFree(s.A); cudaFree(s.Bt);
    s.A = s.Bt = nullptr; s.srcA = s.srcB = nullptr;
  }
  if (!s.A) {
    CUDA_CHECK(cudaMalloc(&s.A, (size_t)M * K / epb));
    CUDA_CHECK(cudaMalloc(&s.Bt, (size_t)N * K / epb));
    s.M = M; s.N = N; s.K = K;
  }
  dim3 t2(16, 16), g2((K + 15) / 16, (N + 15) / 16);
  if (A != s.srcA) {
    size_t n = (size_t)M * K;  // size_t cast prevents int overflow at large M*K
    if (fmt == Fmt::FP8)        q_fp8<<<((int)((n + 255) / 256)), 256>>>(A, s.A, (int)n);
    else if (fmt == Fmt::FP4)   q_fp4<<<((int)((n + 255) / 256)), 256>>>(A, s.A, (int)n, FP4_SCALE);
    else                        q_mxfp4<<<((int)((n / 2 + 255) / 256)), 256>>>(A, s.A, (int)n, FP4_SCALE);
    s.srcA = A;
  }
  if (B != s.srcB) {
    if (fmt == Fmt::FP8)        q_fp8_t<<<g2, t2>>>(B, s.Bt, K, N);
    else if (fmt == Fmt::FP4)   q_fp4_t<<<g2, t2>>>(B, s.Bt, K, N, FP4_SCALE);
    else                        q_mxfp4_t<<<dim3((K / 2 + 15) / 16, (N + 15) / 16), t2>>>(B, s.Bt, K, N, FP4_SCALE);
    s.srcB = B;
  }
}

template <Fmt FMT, int STAGES, int BM = 128, int BN = 128>
static void run(const uint8_t* A, const uint8_t* Bt, float* C, int M, int N, int K) {
  static_assert(STAGES >= 2, "pipeline requires at least 2 stages");
  constexpr int SLAB = (BM + BN) * 64;
  size_t sh = (size_t)STAGES * SLAB;
  static bool attr_set = false;
  if (!attr_set && sh > 48 * 1024) {
    CUDA_CHECK(cudaFuncSetAttribute(gemm_lowprec_t<FMT, STAGES, BM, BN>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
    attr_set = true;
  }
  dim3 t((BM / 64) * (BN / 64) * 32), g((N + BN - 1) / BN, (M + BM - 1) / BM);
  gemm_lowprec_t<FMT, STAGES, BM, BN><<<g, t, sh>>>(A, Bt, C, M, N, K);
}

}  // namespace lowprec

// ---- bench.csv rows ----
void launch_mma_fp8(const float* A, const float* B, float* C, int M, int N, int K) {
  using namespace lowprec;
  stage(Fmt::FP8, A, B, M, N, K);
  Staged& s = g_stage[(int)Fmt::FP8];
  // Tuning history (8192^3, register-pipelined mainloop):
  //   CTA tile:  128x128 -> 500 | 256x128 -> 470 | 128x256 -> 463 | 256x256 -> 123 (reg spill)
  //              -> larger tiles LOSE: occupancy drops to 1 CTA/SM and the 128 MB L2 already
  //                 feeds the 128x128 tile; reuse is not the binding constraint.
  //   Stages:    2 -> 500.1 | 3 -> 503.8 (winner) | 4 -> 430
  // MMA_FP8_TILE / MMA_FP8_STAGES env vars re-run those sweep points.
  const char* t = getenv("MMA_FP8_TILE");
  const char* st = getenv("MMA_FP8_STAGES");
  int stages = st ? atoi(st) : 3;
  if (t && strcmp(t, "256x128") == 0)      run<Fmt::FP8, 2, 256, 128>(s.A, s.Bt, C, M, N, K);
  else if (t && strcmp(t, "128x256") == 0) run<Fmt::FP8, 2, 128, 256>(s.A, s.Bt, C, M, N, K);
  else if (t && strcmp(t, "256x256") == 0) run<Fmt::FP8, 2, 256, 256>(s.A, s.Bt, C, M, N, K);
  else if (stages == 2)                    run<Fmt::FP8, 2, 128, 128>(s.A, s.Bt, C, M, N, K);
  else if (stages == 4)                    run<Fmt::FP8, 4, 128, 128>(s.A, s.Bt, C, M, N, K);
  else                                     run<Fmt::FP8, 3, 128, 128>(s.A, s.Bt, C, M, N, K);
}

void launch_mma_fp4(const float* A, const float* B, float* C, int M, int N, int K) {
  using namespace lowprec;
  stage(Fmt::FP4, A, B, M, N, K);
  Staged& s = g_stage[(int)Fmt::FP4];
  run<Fmt::FP4, 2>(s.A, s.Bt, C, M, N, K);
}

void launch_mma_mxfp4(const float* A, const float* B, float* C, int M, int N, int K) {
  using namespace lowprec;
  stage(Fmt::MXFP4, A, B, M, N, K);
  Staged& s = g_stage[(int)Fmt::MXFP4];
  run<Fmt::MXFP4, 2>(s.A, s.Bt, C, M, N, K);
}
