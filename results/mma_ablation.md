# Hand-written mma.sync ablation — full analysis (sm_120)

**Roadmap item:** Phase 2, "Hand-written mma.sync (m16n8k16) replacing the WMMA wrapper — test
whether Boehm's '~94% with standard optimizations' transfers to the Tensor Core regime."

**Result: it transfers — and overshoots. The final kernel reaches 243.2 TFLOP/s @ 8192³ =
106.1% of cuBLAS-TC** (the `cutlass_80_tensorop_s16816gemm_f16_128x64_64x3` kernel cuBLAS
dispatches on this card). Boehm's FP32/SIMT ablation peaked at 93.7% of cuBLAS; the same
optimization stack applied to raw `mma.sync` Tensor Core code lands above 100% here.
The WMMA wrapper — not the optimization recipe — was the blocker.

## The ablation ladder (M=N=K=8192, all numbers measured in one session)

Raw data: `bench_mma_session.csv` (full session, all 11 kernels × 5 sizes),
`mma_stage_sweep.csv` (pipeline-depth sweep), `bench.csv` (canonical rows).
Chart: `mma_ablation_sm120.png`.

| step | kernel row | what it adds | TFLOP/s | % of cuBLAS-TC | step gain |
|---|---|---|---|---|---|
| – | `wmma` | previous best (WMMA wrapper, same tiling ideas) | 103.1 | 45.0% | – |
| 1 | `mma_base` | raw `mma.sync.m16n8k16` + `ldmatrix`, 128×128 CTA, 2×2 per-warp register tiling (16 warps × 32×32), naive row-major smem, **scalar** loads, single-stage | 47.4 | 20.7% | – |
| 2 | `mma_swizzle` | + XOR-swizzled (bank-conflict-free) smem layout | 58.7 | 25.6% | +24% |
| 3 | `mma_vec` | + 16-byte vectorized `cp.async` global→shared loads | 165.3 | 72.2% | +2.82× |
| 4 | `mma_pipe` | + multi-stage `cp.async` pipeline (2 stages = sweep winner) | 178.0 | 77.7% | +7.7% |
| 5 | `mma_warptile` | + 64×64 per-warp register tile (4 warps/CTA, 196 reg/thread) | **243.2** | **106.1%** | +37% |
| ceiling | `cublas_tc` | `cublasGemmEx` FP16-in/FP32-acc (dispatches `cutlass_80_tensorop_s16816gemm_f16_128x64_64x3`) | 229.0 | 100% | – |

Session baselines reproduce the committed Phase 1 numbers: `wmma` 103.1 vs 103.5 (−0.3%),
`cublas_tc` 229.0 vs 229.2 (−0.1%) — same clock regime, comparisons valid.

### Reading the ladder

- **Step 1 (mma_base) starts *below* WMMA — that is the point of the ablation.** The baseline
  strips every memory-path optimization so each one can be re-added and priced. With scalar
  loads and a conflicted smem layout the Tensor Cores are 80% idle: raw `mma.sync` alone buys
  nothing.
- **Steps 2–3 are the memory story.** Swizzle alone (+24%) is muted while scalar loads dominate;
  vectorized `cp.async` on top of the swizzled layout is the single biggest jump (2.8×). Order
  matters for attribution, not for the destination — these are the "vectorize" step of Boehm's
  ladder, amplified because Tensor Cores consume operands ~8× faster than CUDA cores.
- **Step 4 (pipeline) is small and peaks at 2 stages** (178.0 vs 172.7 @ 3, 151.7 @ 4 — raw
  sweep in `mma_stage_sweep.csv`). The WMMA kernel preferred 3 stages; with `mma.sync`+`ldmatrix`
  the per-slab math finishes sooner, so deeper pipelines just spend shared memory (and occupancy)
  buffering data the kernel doesn't yet need.
- **Step 5 (warptiling) is the decisive step, exactly as in Boehm's FP32 ablation** (his
  warptiling step: 78.4% → 93.7%). Going from 32×32 to 64×64 per-warp tiles doubles the
  arithmetic intensity per `ldmatrix`: each warp issues 8 `ldmatrix.x4` per k16-step feeding
  32 `mma.sync` (ratio 4:1) instead of 4 feeding 8 (ratio 2:1). This is the direct fix for the
  **MIO-queue-full** stall that the Phase 1 ncu profile identified as the WMMA kernel's
  bottleneck — fewer, fatter shared-memory operations per unit of math.

## Why "beats cuBLAS" is credible — and what it does (not) mean

An above-100% claim needs more than one number. Checks performed:

1. **Correctness.** `max_abs_err` vs the FP32 cuBLAS reference is 0.0112 @ 8192 — identical to
   `wmma` and `cublas_tc` (all three run FP16 inputs / FP32 accumulate). Verified additionally
   by running `mma_warptile` alone (first FP16 kernel in the process) so it cannot inherit a
   previous kernel's output buffer.
2. **Reproducibility.** Five separate process invocations: `mma_warptile` 243.2–244.0 TFLOP/s,
   `cublas_tc` 229.0–229.5 TFLOP/s. Spread <0.5%, gap ~6%.
3. **Order/thermal fairness.** Alternating cold/hot runs in separate processes
   (cublas → mma → cublas → mma) leave the gap unchanged. Peak power during the session hit
   the 300 W Max-Q cap (see `clock_state_mma_session.txt`) — both kernels run against the same
   power-governed clock.
4. **Kernel-level evidence, not harness artifact.** nsys (`nsys_kern_sum_mma_8192.txt`) shows
   per-instance GPU durations: `gemm_mma_t<128,128,32,64,64,2>` **4.52 ms** vs
   `cutlass_80_tensorop_s16816gemm_f16_128x64_64x3` **4.79 ms**. The gap exists inside the GPU
   timeline, independent of our CPU-side timing.
5. **Same instruction class.** The cuBLAS kernel's name says it: `s16816gemm` — it is built on
   the *same* `mma.sync.m16n8k16` instruction our kernel uses. This is a like-for-like contest
   of tiling/scheduling, not of instruction sets.

**What it means:** our kernel beats *the kernel cuBLAS chooses to run* on sm_120 — an
Ampere-generation (sm_80) CUTLASS kernel with a 128×64 CTA tile and 3-stage pipeline, selected
by cuBLAS's heuristics that (as of CUDA 12.8) have no Blackwell-native FP16 GEMM specialization
for this shape. Our 128×128 tile with a 2-stage pipeline is simply better matched to this card.

**What it does NOT mean:** we are *not* at 106% of the hardware. Whether GB202's
FP16→FP32-accumulate Tensor Core rate is full-rate (512 FMA/SM/clk, like datacenter parts) or
half-rate (256, like GeForce parts) is not public for the RTX PRO 6000; at the governed
~2.3 GHz those correspond to ~440 or ~220 TFLOP/s dense peaks. Our 243 TFLOP/s exceeding the
half-rate figure indicates the workstation part is **not** half-rate, and against a full-rate
peak both our kernel (~55%) and cuBLAS (~52%) have ample headroom. The honest claim stays
relative: **106% of the cuBLAS-TC ceiling on this card** — the same ceiling every other
number in this repo is quoted against.

## What this says about the Phase 1 result

Phase 1 ended at WMMA = 45% with the ncu diagnosis "MIO-queue-full, shared-memory-feed-bound."
The Phase 2 ladder confirms that diagnosis constructively:

- The feed path, not the math, was the problem: fixing only the feed
  (swizzle + cp.async + warptiling) recovered 2.36× over WMMA with zero change to the math
  instructions' theoretical throughput (`wmma::mma_sync` lowers to the same HMMA operations).
- The WMMA API blocks the two fixes that mattered most: it owns the smem→register layout
  (so no swizzle control, only padding) and its 16×16×16 fragment granularity caps the
  achievable register-tiling ratio. Raw `mma.sync` + `ldmatrix` unlocks both.

## ncu side-by-side profile — the stall-level proof (sm_120)

`ncu --set full` on all three kernels at 8192³, run via `scripts/profile_mma_ncu.sh` (inside a
`--cap-add=SYS_ADMIN` CUDA container — this box restricts GPU counters to root). One kernel
instance each; full text exports committed as `ncu_sm120_{wmma,mma_warptile,cublastc}_8192.txt`.

| metric (ncu, 8192³) | `gemm_wmma_t<128,128,3>` | `gemm_mma_t<…,64,64,2>` (ours, final) | `cutlass_80_tensorop_s16816` (cuBLAS-TC) |
|---|---|---|---|
| **Tensor pipe utilization** | **24.7%** | **85.0%** | 74.0% |
| Compute (SM) throughput | 48.7% | **83.4%** | 72.9% |
| L1/TEX (shared-mem) throughput | **88.1% (saturated)** | 42.6% | 55.5% |
| Warp cycles per issued instruction | 49.6 | **5.8** | 6.6 |
| **Top stall reason** | **MIO queue full** (17.0 cyc = 34.2%) | fixed-latency exec dependency (2.9 cyc) | fixed-latency exec dependency (4.0 cyc) |
| Achieved occupancy | 65.3% | 16.6% | 8.3% |
| Registers / thread | 64 | 186 | 154 |
| CTA / threads | 4096 × 512 | 4096 × 128 | 8192 × 128 |

Three things the table proves:

1. **The Phase 1 diagnosis was right, and the fix worked.** The WMMA kernel stalls on
   MIO-queue-full (its shared-memory pipe runs at 88% while its Tensor pipe idles at 24.7%).
   In the mma.sync kernel that stall is *gone* — the top stall becomes a 2.9-cycle math
   dependency, which ncu's own guidance describes as a stall that "only shows up as a top
   contributor in **already highly optimized kernels**."
2. **The winning signature transferred.** Phase 1 observed (on H100) that winning kernels run
   at *low occupancy with large register state and near-peak tensor utilization*. Our final
   kernel now has exactly that profile (16.6% occupancy, 186 reg/thread, 85% Tensor pipe) —
   the same shape as cuBLAS's kernel (8.3%, 154, 74%), and the opposite of the WMMA kernel's
   high-occupancy/low-utilization profile.
3. **Why we win the head-to-head:** our Tensor pipe runs at 85.0% vs cuBLAS's 74.0% (and
   compute throughput 83.4% vs 72.9%). cuBLAS's 128×64 CTA tile needs 2× the CTAs (8192 vs
   4096) and re-reads operands more; the 128×128 tile amortizes better on this card. Same
   instruction, better feed schedule — that 11-point utilization gap is the 6% wall-clock win.

(Durations under ncu replay — 22.9 / 6.8 / 7.7 ms — are inflated vs wall-clock but preserve
the ranking; the wall-clock numbers in `bench.csv` are the quoted results.)

## Configuration of the final kernel (`mma_warptile`)

```
CTA tile        128 × 128 × 32 (BM × BN × BK)
Warp tile       64 × 64  →  4 warps / 128 threads per CTA
Per k16-step    4× ldmatrix.x4 (A) + 4× ldmatrix.x4.trans (B) → 32× mma.sync.m16n8k16
Smem            2-stage cp.async pipeline, 16 KB/stage, XOR-swizzled
Registers       196/thread, zero spills (ptxas)
Occupancy       2 CTAs/SM (register-limited) = 8 warps/SM — low occupancy, high reuse,
                the same signature as cuBLAS's winning kernels
```

Measurement conditions: RTX PRO 6000 Blackwell Max-Q (300 W cap), driver-governed clocks
(avg ~1.8 GHz / peak 2.3 GHz under load, peak power 300.1 W — the cap is reached), CUDA 12.8.
Absolute TFLOP/s are therefore lower than a 600 W board would show; all conclusions are ratios
measured under identical conditions. Raw clock log: `clock_state_mma_session.txt`.

## Limitations / next steps

- **Small sizes are not dispatched.** The mma kernels run the 128×128 tile at every size, so
  they lose to cuBLAS below 2048 (occupancy). The WMMA kernel's size-dispatch trick (64×64 tile
  for N<1536) would transfer directly; not implemented because the roadmap question is about
  the steady-state ceiling.
- **Hopper.** This result is sm_120-specific by design. On H100 the same kernel would still cap
  at the ~63–65%-of-peak `mma.sync` instruction ceiling (arXiv:2501.12084) — beating cuBLAS
  there requires `wgmma` (roadmap Phase 2.5).
