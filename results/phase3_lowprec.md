# Phase 3 — FP8 / FP4 GEMM on the sm_120 mma path: full analysis

**Roadmap items:** FP8 (E4M3) GEMM · FP4 / MXFP4 GEMM · FP4 vs FP8 vs FP16 Pareto.

**Hardware scoping:** tcgen05 (tensor memory, CTA pairs) is sm_100-only. Everything here uses
the **sm_120 `mma` instruction path**: `QMMA.16832` for FP8/FP4-unpacked, `OMMA.SF.16864` for
block-scaled packed FP4 (SASS verified). Build target `sm_120a` (the FP4 kinds require it).

## Headline (M=N=K=8192, one session, raw rows in `bench.csv`)

| kernel | format | instruction | TFLOP/s | vs FP16 | max_abs_err |
|---|---|---|---|---|---|
| `mma_warptile` | FP16 | `mma.m16n8k16` (HMMA) | 239.2 | 1.00× | 0.0112 |
| `mma_fp8` | FP8 E4M3 | `mma.m16n8k32` (QMMA) | 501.6 | **2.10×** | 1.4 |
| `mma_fp4` | FP4 E2M1, 8-bit containers | `mma.m16n8k32.kind::f8f6f4` (QMMA) | 517.8 | 2.16× | 5.97 |
| `mma_mxfp4` | FP4 E2M1, packed + block scale | `mma.m16n8k64.kind::mxf4` (OMMA.SF) | **988.3** | **4.13×** | 5.97 |
| `cublas_tc` | FP16 | cuBLAS (cutlass_80) | 225.3 | 0.94× | 0.0112 |
| `cublaslt_fp8` | FP8 E4M3 | cuBLASLt | 552.4 | 2.31× | 1.4 |

Chart: `precision_pareto_sm120.png`. Session CSV: `bench_phase3_session.csv`.
Clock/power: 300 W Max-Q cap reached, avg SM clock 2.06 GHz (`clock_state_phase3_session.txt`).

## What the ladder shows

1. **FP8 delivers the spec'd 2×.** 501.6 vs 239.2 TFLOP/s = 2.10×. The kernel is the Phase 2
   winner with the instruction swapped (`m16n8k16` FP16 → `m16n8k32` FP8), the operand path
   narrowed to 1 byte/element, and a register-pipelined mainloop (fragments for k-step i+1
   load while the mma for k-step i runs).
2. **FP4 without packing buys ~nothing over FP8** (518 vs 502 = +3%, same error class).
   `kind::f8f6f4` stores E2M1 values in 8-bit containers → same QMMA pipeline rate, same memory
   traffic as FP8. Accuracy drops from FP8's max_abs_err 1.4 to 5.97 for ~nothing in return.
3. **The 4× lives in the packed, block-scaled path.** MXFP4 (`kind::mxf4`) packs 2 values/byte
   and runs the OMMA.SF pipeline: 988.3 TFLOP/s = 4.13× FP16, 1.97× FP8 — at the *same*
   accuracy as unpacked FP4.
4. **The accuracy axis is the price.** FP16 0.0112 → FP8 1.4 → FP4 5.97 max_abs_err
   (vs FP32 cuBLAS reference at K=8192). FP8 ≈ 2 decimal digits; FP4 ≈ 1.

## Honest comparisons

- **Our FP8 = 90.8% of cuBLASLt FP8** (501.6 vs 552.4). cuBLASLt dispatches
  `sm89_xmma_gemm_e4m3..._tilesize128x128x64_stage3_warpsize2x2x1_tensor16x8x32` — the *same*
  CTA tile, warp layout and instruction as ours. The tuning ladder measured here: larger CTA
  tiles LOSE (256×128 → 470, 128×256 → 463, 256×256 → 123 — occupancy, then spills),
  smem-stage sweep peaks at 3 (504) with register pipelining, 2 stages without (500 vs 493
  baseline). The remaining ~9% is xmma's instruction-level interleave of loads and QMMA —
  a scheduling gap, not a structural one.
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
| Throughput plausibility | 2.10× / 4.13× vs FP16 = hardware spec 2× and ~4×; ours stays 54–57% of the measured peak at all precisions (cuBLAS-TC: 51%; cuBLASLt FP8: 63%) — peak measured directly by the Phase 4 rate probe (`mma_rate_probe.csv`: full-rate FP32-acc, 440.3 TFLOP/s × format multiplier) | ✓ |
| Bit alignment of `kind::f8f6f4` | bug caught & fixed: E2M1 in bits [5:2], not [3:0] — wrong packing gave max_abs_err 82.6 (~20×) | ✓ documented |
| Baseline continuity | same-session FP16 rows reproduce committed values within −0.8% (mma_warptile) / −1.0% (cublas_tc) | ✓ |
| Clock state | 300 W Max-Q cap reached during sweep; avg SM clock 2.06 GHz | recorded |

## Limitations / next steps

- **FP8 vs cuBLASLt, last 9%**: bigger CTA tiles and deeper smem pipelines are both measured
  dead ends (above); what remains is xmma-style interleaving of cp.async/ldmatrix/QMMA at
  instruction granularity inside the k-loop.
- **Per-block MX scaling**: the harness feeds unit block scales; real MXFP4 (per-32-element
  UE8M0 scales) costs no extra math but needs a scale-aware quantizer + scale loading in the
  kernel. Accuracy work, not throughput work.
- **FP6 (E2M3 / E3M2)**: `kind::f8f6f4` also supports FP6 — same QMMA rate, intermediate
  accuracy. Not benched; expected between FP8 and FP4 on the error axis at FP8 throughput.
