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
