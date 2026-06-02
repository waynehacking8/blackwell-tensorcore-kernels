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
