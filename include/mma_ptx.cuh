// Raw PTX building blocks for the hand-written mma.sync GEMM (roadmap Phase 2,
// "Hand-written mma.sync" item). These are exactly the pieces the WMMA API hides:
//   * mma.sync.aligned.m16n8k16  — the Tensor Core instruction itself, operands in
//     registers whose layout WE control (WMMA keeps fragment layout opaque).
//   * ldmatrix                    — shared-mem -> register fragment loads, 4x 8x8
//     tiles per instruction, addresses WE compute (so they can be swizzled).
//   * XOR swizzle                 — bank-conflict-free shared-memory addressing,
//     impossible through wmma::load_matrix_sync's fixed row-major contract.
#pragma once
#include <cuda_fp16.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
//   D(16x8,f32; 4 regs) += A(16x16,f16; 4 regs) x B(16x8,f16 col-major; 2 regs)
// Register layout (per PTX ISA): thread t of the warp holds
//   A: a0a1=(r=t/4,    c=2(t%4)+{0,1})  a2a3=(r=t/4+8, c=..)   [k-quad +8 in a4..a7 -> regs 2,3]
//   B: b0b1=(k=2(t%4)+{0,1}, n=t/4)     b2b3=(k+8, n=t/4)
//   D: d0d1=(r=t/4, c=2(t%4)+{0,1})     d2d3=(r=t/4+8, c=..)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mma_m16n8k16(float d[4], const unsigned a[4], const unsigned b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---------------------------------------------------------------------------
// ldmatrix: each instruction loads four 8x8 b16 tiles from shared memory.
// Thread t supplies the address of one 16-byte tile row:
//   tiles are filled in order; tile i uses the addresses of threads 8i..8i+7.
// Non-trans: A operand (row-major in smem). Trans: B operand (k-major in smem,
// transposed on the fly into the col-major fragment mma.sync expects).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void ldmatrix_x4(unsigned r[4], const half* smem_ptr) {
  unsigned addr = (unsigned)__cvta_generic_to_shared(smem_ptr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_trans(unsigned r[4], const half* smem_ptr) {
  unsigned addr = (unsigned)__cvta_generic_to_shared(smem_ptr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

// ---------------------------------------------------------------------------
// Both A (non-trans) and B (trans) ldmatrix.x4 use the same lane -> tile-row map:
//   tile-row index r = lane%8 + 8*((lane/8)%2),  tile-col half-offset c = 8*(lane/16)
// For A:  address = &As[m0 + r][k0 + c]    (covers one m16 x k16 A fragment)
// For B:  address = &Bs[k0 + r][n0 + c]    (covers TWO n8 x k16 B fragments: n0, n0+8)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void ldmatrix_lane_rc(int lane, int& r, int& c) {
  r = (lane & 7) + ((lane >> 3) & 1) * 8;
  c = (lane >> 4) * 8;
}

// ---------------------------------------------------------------------------
// XOR swizzle: shared memory is viewed as rows of 16-byte chunks (8 halves).
// A logical (row, chunk) is stored at physical chunk  chunk ^ f(row)  so that the
// 8 rows touched by one ldmatrix tile land in 8 distinct 16-byte bank groups.
//   CPR = chunks per row (BK or BN in halves, divided by 8)
//   CPR==4  (64 B rows, e.g. BK=32):  f(row) = (row >> 1) & 3
//   CPR>=8  (128+ B rows):            f(row) = row & 7
// SWIZZLE=false gives the identity map (the bank-conflicted ablation baseline).
// ---------------------------------------------------------------------------
template <int CPR, bool SWIZZLE>
__device__ __forceinline__ int smem_off(int row, int col_half) {
  int chunk = col_half >> 3, in = col_half & 7;
  if (SWIZZLE) chunk ^= (CPR == 4) ? ((row >> 1) & 3) : (row & 7);
  return row * CPR * 8 + chunk * 8 + in;
}

// byte-addressed variant (FP8/FP4 kernels): col and result in bytes, 16-byte chunks
template <int CPR, bool SWIZZLE>
__device__ __forceinline__ int smem_off_b(int row, int col_byte) {
  int chunk = col_byte >> 4, in = col_byte & 15;
  if (SWIZZLE) chunk ^= (CPR == 4) ? ((row >> 1) & 3) : (row & 7);
  return row * CPR * 16 + chunk * 16 + in;
}

// ---------------------------------------------------------------------------
// cp.async: 16-byte global -> shared copy that bypasses registers (Ampere+).
// Thin PTX wrappers (same instruction __pipeline_memcpy_async lowers to, but kept
// explicit so this file is the single place all PTX lives).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void cp_async_16(half* smem_dst, const half* gmem_src) {
  unsigned daddr = (unsigned)__cvta_generic_to_shared(smem_dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(daddr), "l"(gmem_src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// byte-typed overload for the FP8/FP4 kernels (Phase 3): same instruction, byte pointers
__device__ __forceinline__ void cp_async_16(uint8_t* smem_dst, const uint8_t* gmem_src) {
  unsigned daddr = (unsigned)__cvta_generic_to_shared(smem_dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(daddr), "l"(gmem_src));
}

// ---------------------------------------------------------------------------
// Phase 3 — Blackwell low-precision formats through the sm_120 mma path.
//
//   FP8 (E4M3):  mma.sync.aligned.m16n8k32.f32.e4m3.e4m3.f32   (sm_89+; QMMA.16832 SASS)
//   FP4 (E2M1):  mma...kind::f8f6f4 m16n8k32, e2m1 in 8-bit containers (sm_120a; QMMA.16832)
//   MXFP4:       mma...kind::mxf4.block_scale m16n8k64, packed e2m1 + UE8M0 scales
//                (sm_120a; OMMA.SF.16864 — the only path with packed-FP4 throughput)
//
// All three keep the m16n8k32-byte operand layout: A row-major (1 byte/element,
// MXFP4: 2 elements/byte), B *N-major* (transposed). The same ldmatrix.x4 + lane
// mapping used for FP16 works unchanged (each "b16" element = 2 FP8s / 4 FP4s).
// FP4 kinds need the sm_120a target; guarded with __CUDA_ARCH_FEAT_SM120_ALL.
// ---------------------------------------------------------------------------
#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FEAT_SM120_ALL)
#define MMA_PTX_HAS_SM120A 1
#else
#define MMA_PTX_HAS_SM120A 0
#endif

// D(16x8,f32) += A(16x32, e4m3, row-major) x B(32x8, e4m3, col-major)
__device__ __forceinline__ void mma_m16n8k32_e4m3(float d[4], const unsigned a[4], const unsigned b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// D(16x8,f32) += A(16x32, e2m1-in-8bit, row-major) x B(32x8, e2m1-in-8bit, col-major)
// FP4 values in the LOW 4 bits of each byte. Same QMMA.16832 rate as FP8 on sm_120.
__device__ __forceinline__ void mma_m16n8k32_e2m1(float d[4], const unsigned a[4], const unsigned b[2]) {
#if MMA_PTX_HAS_SM120A
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

// D(16x8,f32) += A(16x64, packed e2m1, row-major) x B(64x8, packed e2m1, col-major)
// with UE8M0 block scales (scale_vec::2X = one scale per 32 elements).
// Scales are passed as 0x7F = 2^0 = 1.0 (per-tensor scaling handled outside the kernel),
// which keeps the math exact while exercising the full OMMA.SF.16864 datapath.
__device__ __forceinline__ void mma_m16n8k64_mxf4(float d[4], const unsigned a[4], const unsigned b[2]) {
#if MMA_PTX_HAS_SM120A
  const unsigned sfa = 0x7f7f7f7f, sfb = 0x7f7f7f7f;
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0,0}, {%11}, {0,0};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sfa), "r"(sfb));
#endif
}
