# Phase 3 — FP8 / FP4 GEMM on the sm_120 mma path: full analysis

**Roadmap items:** FP8 (E4M3) GEMM · FP4 / MXFP4 GEMM · FP4 vs FP8 vs FP16 Pareto.

**Hardware scoping:** tcgen05 (tensor memory, CTA pairs) is sm_100-only. Everything here uses
the **sm_120 `mma` instruction path**: `QMMA.16832` for FP8/FP4-unpacked, `OMMA.SF.16864` for
block-scaled packed FP4 (SASS verified). Build target `sm_120a` (the FP4 kinds require it).

## Headline (M=N=K=8192, one session, raw rows in `bench.csv`)

| kernel | format | instruction | TFLOP/s | vs FP16 | max_abs_err |
|---|---|---|---|---|---|
| `mma_warptile` | FP16 | `mma.m16n8k16` (HMMA) | 241.2 | 1.00× | 0.0112 |
| `mma_fp8` | FP8 E4M3 | `mma.m16n8k32` (QMMA) | 493.0 | **2.04×** | 1.4 |
| `mma_fp4` | FP4 E2M1, 8-bit containers | `mma.m16n8k32.kind::f8f6f4` (QMMA) | 519.0 | 2.15× | 5.97 |
| `mma_mxfp4` | FP4 E2M1, packed + block scale | `mma.m16n8k64.kind::mxf4` (OMMA.SF) | **951.9** | **3.95×** | 5.97 |
| `cublas_tc` | FP16 | cuBLAS (cutlass_80) | 226.9 | 0.94× | 0.0112 |
| `cublaslt_fp8` | FP8 E4M3 | cuBLASLt | 555.5 | 2.30× | 1.4 |

Chart: `precision_pareto_sm120.png`. Session CSV: `bench_phase3_session.csv`.
Clock/power: 300 W Max-Q cap reached, avg SM clock 2.06 GHz (`clock_state_phase3_session.txt`).

## What the ladder shows

1. **FP8 delivers exactly the spec'd 2×.** 493.0 vs 241.2 TFLOP/s = 2.04×. The kernel is the
   Phase 2 winner with the instruction swapped (`m16n8k16` FP16 → `m16n8k32` FP8) and the
   operand path narrowed to 1 byte/element — half the bytes feed twice the FLOPs.
2. **FP4 without packing buys ~nothing over FP8** (519 vs 493 = +5%, same error class).
   `kind::f8f6f4` stores E2M1 values in 8-bit containers → same QMMA pipeline rate, same memory
   traffic as FP8. Accuracy drops from FP8's max_abs_err 1.4 to 5.97 for ~nothing in return.
3. **The 4× lives in the packed, block-scaled path.** MXFP4 (`kind::mxf4`) packs 2 values/byte
   and runs the OMMA.SF pipeline: 951.9 TFLOP/s = 3.95× FP16, 1.93× FP8 — at the *same*
   accuracy as unpacked FP4.
4. **The accuracy axis is the price.** FP16 0.0112 → FP8 1.4 → FP4 5.97 max_abs_err
   (vs FP32 cuBLAS reference at K=8192). FP8 ≈ 2 decimal digits; FP4 ≈ 1.

## Honest comparisons

- **Our FP8 = 88.8% of cuBLASLt FP8** (493.0 vs 555.5). The Phase 2 kernel structure beat
  cuBLAS at FP16 (106%), but at FP8 the same structure trails the library kernel: at 2× math
  throughput, the feed path (16 KB smem slabs, 4:1 mma:ldmatrix) is no longer generous —
  exactly the MIO bind Phase 2 fixed reappears one precision later. Closing it needs a larger
  CTA tile (more reuse per byte) — left as documented future work.
- **cuBLAS has no FP4 GEMM path on sm_120** (no FP4 cuBLASLt type, no NVFP4 cuBLAS API at
  CUDA 12.8); our MXFP4 number is the only measured FP4 GEMM data point on this card.
- **MXFP4 throughput is real, but the scaling is per-tensor.** The block-scale factors are
  fed as 1.0 (UE8M0 = 127) and quantization uses one scale for the whole matrix. This is
  throughput-identical to real per-32-block scaling (the hardware does the same work), and for
  the uniform [-0.5, 0.5] test matrices, per-block scales would be ≈ identical anyway. With
  real-world activation outliers the accuracy gap FP4 vs FP8 would widen.

## Validation checks

| check | evidence | verdict |
|---|---|---|
| FP8 math correct | max_abs_err 1.4 identical to cuBLASLt FP8 (same quantized inputs, same K) | ✓ |
| FP4 math correct | mma_fp4 (QMMA) and mma_mxfp4 (OMMA.SF) produce identical max_abs_err (5.97) through two different instructions | ✓ |
| Throughput plausibility | 2.04× / 3.95× vs FP16 = hardware spec 2× and ~4×; ours stays ~54% of peak at all precisions (cuBLAS-TC: 51%; cuBLASLt FP8: 62%) | ✓ |
| Bit alignment of `kind::f8f6f4` | bug caught & fixed: E2M1 in bits [5:2], not [3:0] — wrong packing gave max_abs_err 82.6 (~20×) | ✓ documented |
| Baseline continuity | same-session FP16 rows reproduce committed values within −0.8% (mma_warptile) / −1.0% (cublas_tc) | ✓ |
| Clock state | 300 W Max-Q cap reached during sweep; avg SM clock 2.06 GHz | recorded |

## Limitations / next steps

- **FP8 tile shape**: 128×128 CTA / 64×64 warp tile is tuned for FP16 feed rates. At FP8/FP4
  rates a 128×256 or 256×128 CTA (more reuse per loaded byte) is the obvious next step to
  close the 11% gap to cuBLASLt.
- **Per-block MX scaling**: the harness feeds unit block scales; real MXFP4 (per-32-element
  UE8M0 scales) costs no extra math but needs a scale-aware quantizer + scale loading in the
  kernel. Accuracy work, not throughput work.
- **FP6 (E2M3 / E3M2)**: `kind::f8f6f4` also supports FP6 — same QMMA rate, intermediate
  accuracy. Not benched; expected between FP8 and FP4 on the error axis at FP8 throughput.
