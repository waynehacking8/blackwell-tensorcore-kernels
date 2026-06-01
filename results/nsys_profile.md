# Nsight Systems — wmma vs cublas_tc kernel comparison

Capture: `nsys profile ./build/gemm_bench 4096 4096 4096` on **RTX PRO 6000 Blackwell
Max-Q (sm_120)**, CUDA 12.8. Raw `nsys_4096.nsys-rep` is kept local (gitignored, ~285 KB);
this is the shareable summary. Reproduce with `bash scripts/profile.sh 4096` then
`nsys stats --report cuda_gpu_kern_sum results/nsys_4096.nsys-rep`.

## GPU kernel summary (per-instance average, M=N=K=4096)

| kernel (as nsys names it) | avg µs | TFLOP/s | role |
|---|---|---|---|
| `gemm_naive` | 23137 | 5.9 | one-thread-per-output baseline |
| `gemm_tiled` | 16657 | 8.3 | shared-mem + register block |
| **`gemm_wmma`** (hand-written, `__half` in) | **2903** | **47.3** | our WMMA `m16n16k16` Tensor Core kernel |
| `cutlass_80_simt_sgemm_256x128` | 2404 | 57.2 | what FP32 `cublasSgemm` dispatches to — **SIMT / CUDA cores**, not TC |
| **`cutlass_80_tensorop_s16816gemm_f16_128x64`** | **573** | **240.1** | what `cublas_tc` (`cublasGemmEx`) dispatches to — **Tensor Core** |
| `f2h_tc` / `f2h` (FP32→FP16 cast) | ~60 / ~57 | — | staged once, outside the timed GEMM loop |

(Instances = 11 per kernel = 1 warmup + 10 timed iters; the cast kernels fire only ~2×,
confirming the FP16 staging is **not** in the timed region.)

## What the profile reveals

1. **Both "Tensor Core" things are real, but they are different kernels.** nsys names give it
   away: `cublas_tc` runs **`...tensorop_s16816gemm_f16...`** — `tensorop` + `16816` is the
   `mma.m16n8k16` 5th-gen Tensor Core instruction. The FP32 `cublas` baseline runs
   **`...simt_sgemm...`** — SIMT, i.e. ordinary CUDA cores. So `cublas_tc` is the correct
   same-precision ceiling; `cublas` (FP32) was the precision-mismatched one.

2. **Our WMMA kernel is ~5× slower than cuBLAS's Tensor Core kernel** (2903 µs vs 573 µs →
   **19.7% kernel-only**), which matches the end-to-end 20.0% from `bench.csv` — the timing
   harness is clean (no cast/handle overhead leaking in).

3. **Why the 5× gap (the interesting part):** both issue Tensor Core MMAs, so the difference is
   the memory pipeline around them, not the math units:
   - cuBLAS's `128x64` CTA tile uses **multi-stage shared-memory pipelining** (the `_64x3`
     ≈ 3 pipeline stages) and `cp.async` global→shared prefetch to keep the Tensor Cores fed.
   - Our `gemm_wmma` has **no shared-memory double-buffering / async copy** — each warp loads
     fragments straight from global memory, so at 4096 the Tensor Cores stall on memory traffic.
     That is the textbook naive-WMMA bottleneck and exactly why we land at the ~17–22% range
     rather than near 100%.

4. **Honest baseline confirmed:** `cublas_tc` (240 TFLOP/s) is **4.2×** the FP32 `simt_sgemm`
   (57 TFLOP/s) — only reachable on Tensor Cores, so the ceiling is doing what it claims, and
   "% of cuBLAS-TC" is the number to quote.

**Next optimization, directly indicated by this profile:** add shared-memory tiling with
`cp.async` double-buffering to `gemm_wmma` (mirror cuBLAS's multi-stage `128x64` structure) —
that's the single change expected to move WMMA off the ~20% memory-bound plateau toward the
Tensor Core ceiling.

## Optimization result — shared-memory + cp.async double-buffering

The WMMA kernel was rewritten from naive (per-warp 16×16, fragments loaded straight from
global memory) to a **64×64 block tile, BK=32, with a 2-stage `cp.async`
(`__pipeline_memcpy_async`) shared-memory pipeline** — the next K-slab is prefetched while the
current one feeds the Tensor Cores. Correctness is unchanged (maxerr 0.00769 @4096, 0.0112
@8192, identical to before).

nsys kernel time at 4096 (`gemm_wmma`, per-instance avg): **2903 µs → 2180 µs (−25%)**, i.e.
**19.7% → 26.3% of the cuBLAS tensorop kernel** (574 µs) — and the kernel-only ratio matches
the end-to-end 26.4% from `bench.csv`, so the harness stays clean.

| size | wmma naive | wmma opt | speedup | naive % of cuBLAS-TC | opt % of cuBLAS-TC |
|---|---|---|---|---|---|
| 512  | 13.89 | 16.12 | 1.16× | 42.6% | 49.5% |
| 1024 | 31.68 | 40.05 | 1.26× | 22.6% | 28.6% |
| 2048 | 44.47 | 56.03 | 1.26× | 20.8% | 26.2% |
| 4096 | 47.61 | 62.99 | 1.32× | 20.0% | 26.4% |
| 8192 | 39.77 | 62.65 | **1.58×** | 17.3% | **27.3%** |

The biggest gain is at the **largest** size (1.58× @ 8192), exactly where the naive kernel was
memory-bound: naive WMMA *fell back* to 39.8 TFLOP/s at 8192, while the pipelined version
**holds ~63 TFLOP/s** and no longer decays — confirming the bottleneck was global-memory
traffic, not the Tensor Core math, as the profile predicted.

## Optimization v2 → v3: register tiling + deeper pipeline + size dispatch

Next, the kernel was templated on tile size and given **per-warp 2×2 register tiling** (each
warp now owns a 32×32 region = four 16×16 accumulators, reusing every loaded fragment across
the other warp axis) plus a **3-stage** pipeline, with a **128×128** tile for large N and the
64×64 tile kept for small N (size dispatch at N≥1536, since the 128 tile under-occupies the GPU
below that).

| size | v2 (64×64, 2-stage) | **v3 (dispatch)** | tile used | speedup | **% of cuBLAS-TC** |
|---|---|---|---|---|---|
| 512  | 16.12 | 16.08 | 64×64  | 1.00× | 47.6% |
| 1024 | 40.05 | 38.72 | 64×64  | 0.97× | 31.4% |
| 2048 | 56.03 | 68.64 | 128×128 | 1.23× | 32.0% |
| 4096 | 62.99 | 96.57 | 128×128 | 1.53× | 40.7% |
| 8192 | 62.65 | 102.7 | 128×128 | **1.64×** | **45.0%** |

nsys @4096: `gemm_wmma_t<128,128,3>` = **1425 µs** (vs v2's 2180 µs, vs cuBLAS tensorop 575 µs)
→ **40.3% of cuBLAS-TC kernel-only**, matching end-to-end 40.7%. Overall the WMMA kernel went
**17.3% → 45.0%** of the same-precision Tensor Core ceiling at 8192 across the two optimizations.

**Pipeline depth is tuned, not arbitrary** — measured at 8192: 3 stages 103 TFLOP/s, 4 stages
95, 5 stages 88. Past 3, the larger shared-memory footprint costs more occupancy than the extra
prefetch buys, so 3 is the sweet spot.

## Warp specialization — tried, did NOT help (kept honest)

A warp-specialized variant (`experiments/wmma_ws_probe.cu`: 16 consumer warps doing mma + 4
dedicated producer warps doing all `cp.async`, synced via `bar.arrive/bar.sync` named barriers)
was implemented and benchmarked head-to-head:

| size | dispatch (multi-stage) | warp-specialized | result |
|---|---|---|---|
| 4096 | 96.6 | 97.4 | ~tie (+0.8%) |
| 8192 | 103.2 | 95.4 | **WS 7.6% slower** |

It did not beat the plain multi-stage pipeline, consistent with the depth sweep above: at this
512-thread / 128×128 tile the `cp.async` pipeline already saturates latency-hiding, so peeling
4 warps off for production just removes them from mma throughput (and adds barrier cost). Warp
specialization pays off on Hopper-style persistent/TMA kernels with much larger tiles and
async-transaction barriers — not here. The probe is kept under `experiments/` (own `main()`,
not in the build) for reproducibility; the shipped kernel is the multi-stage dispatch one.
