# Validation & Cross-Reference

Cross-checking the measured GEMM numbers (`results/bench.csv`) against vendor specs
and expected kernel behaviour. GPU: **NVIDIA RTX PRO 6000 Blackwell Max-Q
Workstation Edition** (`sm_120`, 300 W Max-Q), CUDA 12.8, FP32 inputs, cuBLAS as ceiling.

## Hardware spec cross-check

- **Spec:** RTX PRO 6000 Blackwell **FP32 = 125 TFLOPS** dense (non-tensor), @ **600 W**
  (nvidia.com / flopper.io).
- **Measured:** cuBLAS `cublasSgemm` peaks at **~54.7 TFLOP/s** at M=N=K=8192.
- **Reconciliation:** this board is the **Max-Q (300 W)** variant — roughly half the
  power budget of the 600 W spec part. ~54.7 / 125 ≈ **44%**, i.e. about half the
  rated FP32 throughput, consistent with a halved power/clock envelope. ✓ Reasonable.

## Clock state — why absolute TFLOP/s sit below the hardware peak

The absolute numbers here (cuBLAS-TC ~228, our WMMA ~103 TFLOP/s) are **well below** the
Blackwell FP16/FP32-acc hardware peak (≳1 PFLOP/s class with 2:4 sparsity; roughly half that dense). We verified this is **clock
governance, not a measurement or kernel bug**, by direct sampling during the GEMM:

| metric | value during GEMM | ceiling | note |
|---|---|---|---|
| SM clock | **2325 MHz** | 3090 MHz | running at ~75% of max |
| Power draw | **174 W** | 300 W | only ~58% of the budget used |
| Throttle reason | **`0x0`** | — | **no thermal/power throttle active** |
| Temp | 59 °C | — | cool |

So the GPU simply isn't boosting to full clocks (Max-Q idle/curve governance), and it is
**not** being throttled — there is headroom it doesn't use. Under sustained back-to-back GEMMs
the clock does climb (2145–2325 MHz, ~274 W), confirming the part can go higher but settles low
between kernels. We could not pin clocks (`nvidia-smi -lgc 3090` needs root: *"current user does
not have permission to change clocks"*), so all numbers are at the governed ~2.3 GHz.

**Two consequences, stated honestly:**
1. **Measurement is sound, not noisy.** Five repeats of cuBLAS-TC @8192 gave 227.0–229.0
   TFLOP/s (<1 % spread) — the timing harness is stable; the low absolute value is the clock,
   not jitter.
2. **Relative percentages stay valid.** Our headline metric is **wmma ÷ cublas_tc**, and both
   are measured in the same process, back-to-back, at the same governed clock — so the ratio
   (e.g. 45 % @ 8192) is unaffected by the down-clocking; only the **absolute** TFLOP/s of
   *both* kernels scale down together. "45 % of cuBLAS-TC" is **not** "45 % of the hardware
   FP16 peak" — it is 45 % of the (also down-clocked) cuBLAS Tensor Core kernel, which is the
   honest, like-for-like baseline.

## Kernel-vs-ceiling sanity

At M=N=K=8192 (the most timing-stable point):

| kernel | TFLOP/s | % of FP32 cuBLAS | % of cuBLAS-TC | max abs err | reading |
|---|---|---|---|---|---|
| naive | 4.8 | 8.7% | 2.1% | 0 | one-thread-per-output, no reuse — expected floor |
| tiled | 7.3 | 13.3% | 3.2% | 0 | shared-mem + register block, ~1.5× naive |
| **wmma** | 103.5 | 189% † | **45.2%** | 0.011 | 128×128 reg-tiled + 3-stage cp.async, FP16-in/FP32-acc TC |
| cublas | 54.8 | 100% | 23.9% | 0 | FP32 baseline (cublasSgemm, CUDA cores) |
| cublas_tf32 | 152.7 | 278% † | 66.6% | 0.0113 | TF32 Tensor Core rung (cublasGemmEx, FP32 in / TF32 compute) |
| **cublas_tc** | 229.2 | 418% † | **100%** | 0.011 | FP16/FP32-acc Tensor Core ceiling (cublasGemmEx) |

> **Precision-ladder sanity (FP32→TF32→FP16, all same card):** throughput is strictly
> monotone at every size — FP32 54.8 < TF32 152.7 < FP16 229.2 TFLOP/s @8192 — and the
> intermediate TF32 rung lands cleanly between the CUDA-core baseline and the FP16 ceiling,
> exactly as the hardware predicts. **Notable (verified) detail:** TF32 and FP16 have the
> *same* max-abs-err (~0.011) **and** the same mean-abs-err (0.001111 vs 0.001111 @4096) — not
> a bug: TF32 and FP16 both carry a **10-bit mantissa**, so on well-conditioned [-0.5,0.5]
> inputs with FP32 accumulation the rounding error is identical; TF32's advantage is its wider
> 8-bit exponent (range), which this benign test doesn't exercise.

> **% of cuBLAS-TC** is the honest same-precision ceiling (vs `cublasGemmEx`, FP16 in / FP32
> acc, Tensor Cores). **% of FP32 cuBLAS** is vs `cublasSgemm` (FP32, CUDA cores) and is
> precision-mismatched (**†**): WMMA and cublas_tc run FP16 on Tensor Cores, so their `>100%`
> rows there (e.g. wmma 1125% @ 512, cublas_tc 418% @ 8192) are **not** kernels beating cuBLAS —
> just FP16-TC vs FP32-CUDA-core. Against the same-precision cuBLAS-TC ceiling the optimized WMMA
> kernel (size-dispatched: 64×64 for N<1536, 128×128 + 2×2 register tiling + 3-stage cp.async for
> N≥1536) reaches **45.2% @ 8192** (40.4% @ 4096, 31.8% @ 2048; 28.2% @ 1024, 48.7% @ 512 on the
> small-tile path) — see the naive→optimized→warp-spec progression in `results/nsys_profile.md`.

- **WMMA against the honest same-precision ceiling.** Vs cuBLAS-TC (same FP16-in/FP32-acc
  Tensor Core path), the optimized WMMA kernel reaches **45.2% @ 8192**. Three stages of work
  got it there from the naive **17.3%**: (1) shared-mem tiling + cp.async double-buffering fixed
  the naive memory-bound decay (47→63 TFLOP/s); (2) a 128×128 tile with per-warp 2×2 register
  tiling + 3-stage pipeline raised reuse (→103 TFLOP/s, 1.64×); (3) size dispatch keeps the
  64×64 tile for N<1536 so small matrices don't lose occupancy. Pipeline depth is tuned by
  measurement (3 stages 103 > 4 stages 95 > 5 stages 88 TFLOP/s @ 8192). **Warp specialization
  was implemented and benchmarked but did *not* beat the multi-stage pipeline** (95.4 vs 103.2 @
  8192) — at this 512-thread tile the cp.async pipeline already saturates latency-hiding, so
  dedicating warps to production costs more mma throughput than it saves. This is the **expected
  outcome per CudaDMA (Bauer et al., SC'11)**: warp specialization only pays off when combined
  with large tiles + async-transfer hardware (TMA) + register reallocation; added alone it just
  removes compute warps and adds barrier sync overhead. (Colfax's CUTLASS tutorial finds
  warp-specialized vs plain multistage differ by only ~1.7% even on Hopper where it applies.)
  **The remaining gap to cuBLAS is *not* TMA or warp specialization.** nsys shows cuBLAS dispatches
  `cutlass_80_tensorop_s16816gemm_f16_128x64` — an **Ampere-generation (sm_80) kernel** that uses
  `cp.async` multistage pipelining; TMA and warp specialization are sm_90/Hopper+ features the
  baseline does not use (the `cutlass_80_*` name string is the evidence). The gap comes from the
  larger 128×64 CTA tile, deeper register-level warp/thread tiling, vectorized loads,
  swizzle/rasterization, and more aggressive multistage cp.async pipelining — exactly the
  ingredients Boehm's published ablation rides to ~94% of cuBLAS without TMA or warp spec.
  The "% of FP32 cuBLAS" figure remains precision-mismatched and is kept only for continuity.
- **cuBLAS-TC confirms the Tensor Core path.** cublas_tc reaches 229 TFLOP/s @ 8192 — **4.2×**
  the FP32 cublasSgemm (54.8) and up to **23×** at 512 — which is only possible on the Tensor
  Core path, so the baseline is doing what it claims.
- **Observed-but-not-overclaimed:** on this Blackwell (sm_120) card, cuBLAS *chose* an sm_80-style
  CUTLASS kernel (`cutlass_80_tensorop_s16816gemm_f16_128x64`) for this problem size — an
  interesting fact in itself. We note it without overclaiming the cause: this is cuBLAS's internal
  heuristic selection; native sm_120 kernels may well exist but were not selected for these shapes.
  The takeaway for the gap analysis is only that the baseline runs Ampere-style cp.async
  multistage code, not TMA/warp-spec.
- **Precision check:** naive/tiled (FP32) max-abs-err ~1e-4→0; wmma and cublas_tc (FP16 inputs)
  ~0.01 over large K — exactly the expected FP16 rounding, confirming correctness.

## Honest caveat — now resolved by the same-precision baseline

The original "% of FP32 cuBLAS" was apples-to-oranges (FP16-WMMA vs FP32-`cublasSgemm`); its
`>100%` small-N rows reflect the precision difference, not WMMA beating a same-precision kernel.
That prediction has now been **measured**: the harness includes **`cublas_tc`
(`cublasGemmEx`, FP16 in / FP32 accumulate)** in `src/cublas_tc.cu`, with identical timing
methodology to the WMMA kernel — the FP32→FP16 cast is staged once outside the timed loop, and
(after a fix) the cuBLAS handle is created once rather than per timed call, which had dominated
small-N timing.

Result on sm_120, vs this honest ceiling (full sweep in `bench.csv`; FP32-only sweep preserved
in `bench_fp32only.csv`). WMMA here is the **size-dispatched, register-tiled, 3-stage cp.async**
kernel:

| size | wmma % of FP32 cuBLAS | **wmma % of cuBLAS-TC** | cublas_tc speedup vs FP32 |
|---|---|---|---|
| 512  | 1125% | 48.7% | 23.2× |
| 1024 | 377%  | 28.2% | 13.3× |
| 2048 | 193%  | 31.8% | 6.1× |
| 4096 | 202%  | 40.4% | 5.0× |
| 8192 | 189%  | **45.2%** | 4.2× |

> All derived percentages in this document are computed from the committed `results/bench.csv`
> (the single source of truth). cuBLAS absolutes vary a few percent run-to-run under the clock
> governance described above, so percentages quoted from earlier runs may differ slightly.

The WMMA kernel reaches **45% @ 8192** against the same-precision Tensor Core ceiling (naive was
17–22% and decayed at 8192; shared-mem+cp.async, then register tiling + deeper pipeline + size
dispatch closed most of the gap; warp specialization was tried and rejected — see
`results/nsys_profile.md`). No size exceeds 100% of cuBLAS-TC **for the WMMA kernel** (the Phase 2
raw-`mma.sync` kernel does exceed it — validated separately below). The FP32-cuBLAS column is
retained only for continuity.

**On the 1024 dip in % of cuBLAS-TC (28.2 %, below its neighbours).** Verified to be a
*denominator* effect, not a kernel bug: forcing M=N=K=1024 through the 128×128 tile gives 30.3
TFLOP/s vs the dispatched 64×64 tile's **38.7** — i.e. the dispatch correctly picks the faster
tile, and our WMMA is *not* slow at 1024. The lower percentage comes from cuBLAS-TC itself being
unusually fast there (~140 TFLOP/s), so the ratio dips even though our absolute throughput is
healthy. The size-dispatch crossover (N≥1536 → 128×128) is therefore correct as set.

## Phase 2 mma.sync kernel — validating an above-the-ceiling claim

The Phase 2 hand-written `mma.sync` kernel (`mma_warptile`, `src/gemm_mma.cu`) measures
**243.2 TFLOP/s @ 8192³ = 106.1% of cuBLAS-TC** on sm_120. A >100% number gets extra scrutiny:

| check | evidence | verdict |
|---|---|---|
| Correctness | `max_abs_err` = 0.0112 @ 8192, bit-identical to `wmma` and `cublas_tc` (same FP16-in/FP32-acc math); verified standalone so it cannot inherit another kernel's output buffer | ✓ |
| Reproducibility | 5 separate process invocations: 243.2–244.0 TFLOP/s (spread <0.5%); `cublas_tc` 229.0–229.5 in the same runs | ✓ |
| Thermal/order fairness | alternating cublas→mma→cublas→mma in separate processes; gap unchanged; both run under the same 300 W Max-Q governor (cap reached — `results/clock_state_mma_session.txt`) | ✓ |
| GPU-timeline (not harness) | nsys per-instance durations: ours 4.52 ms vs `cutlass_80_tensorop_s16816gemm_f16_128x64_64x3` 4.79 ms (`results/nsys_kern_sum_mma_8192.txt`) | ✓ |
| Same instruction class | cuBLAS kernel name contains `s16816` = `mma.sync.m16n8k16`, the same instruction ours uses — a tiling/scheduling contest, not an instruction-set advantage | ✓ |
| Stall-level mechanism (ncu) | `ncu --set full` side-by-side (`results/ncu_sm120_*_8192.txt`): ours runs Tensor pipe at 85.0% vs cuBLAS 74.0%; both stall only on fixed-latency math dependencies (2.9 vs 4.0 cyc/warp); the WMMA kernel's MIO-queue-full stall (34.2%) is eliminated | ✓ |
| Baseline continuity | session re-measured `wmma` 103.1 (committed: 103.5) and `cublas_tc` 229.0 (committed: 229.2) — same clock regime as Phase 1 | ✓ |

**Scope of the claim (kept honest):** the kernel beats *the kernel cuBLAS chooses to dispatch*
on sm_120 (an sm_80-generation CUTLASS kernel, 128×64 tile) — it does **not** beat the hardware
peak. GB202's FP16→FP32-acc dense peak at the governed ~2.3 GHz is ~440 TFLOP/s if the
workstation part runs full-rate Tensor Cores (our 243 exceeding the half-rate figure of ~220
indicates it is full-rate); both our kernel (~55% of that) and cuBLAS (~52%) leave headroom.
Full analysis: `results/mma_ablation.md`.

## Phase 3 FP8/FP4 kernels — throughput ratios vs hardware spec

The Phase 3 kernels (`src/gemm_mma_fp8.cu`, sm_120a build) claim **2.09× (FP8)** and
**4.11× (MXFP4)** over the FP16 kernel. Cross-checks:

| check | evidence | verdict |
|---|---|---|
| FP8 ratio vs spec | 5th-gen TC spec: FP8 = 2× FP16. Measured: 503.7 / 241.5 = **2.09×** | ✓ |
| FP4 ratio vs spec | packed FP4 = 4× FP16. Measured: 992.6 / 241.5 = **4.11×** | ✓ |
| Constant fraction of peak | ours = ~54% of (estimated full-rate) peak at FP16, FP8 and FP4; cuBLAS-TC = 51%, cuBLASLt FP8 = 62% — no precision is anomalous | ✓ |
| FP8 math correctness | max_abs_err = 1.4 bit-identical to cuBLASLt FP8 (same E4M3 quantized inputs, same K) | ✓ |
| FP4 math correctness | `mma_fp4` (QMMA, unpacked) and `mma_mxfp4` (OMMA.SF, packed + block-scale) agree exactly (max_abs_err 5.97) through two different instruction paths | ✓ |
| Library baseline | cuBLASLt FP8 (E4M3, TN layout) = 553.5 TFLOP/s; ours = 91.0% of it — reported as-is, not hidden | ✓ |
| Error scaling | max_abs_err 0.011 (FP16) → 1.4 (FP8) → 6.0 (FP4) tracks 2^-10 / 2^-3 / 2^-1 mantissa widths at K=8192 | ✓ |
| Session continuity | same-session FP16 rows reproduce committed values within −0.8% / −1.0% (clock-cap variance) | ✓ |

Caveats stated in `results/phase3_lowprec.md`: MXFP4 block-scale factors are fed as 1.0 with
per-tensor (not per-32-block) input scaling — throughput-identical; accuracy generalizes only
to data without per-block outliers. cuBLAS has no FP4 GEMM on sm_120, so the MXFP4 number has
no library ceiling to compare against.

## H100 (sm_90) cross-check — does any of this transfer?

The roadmap question: does the 45.2%-of-cuBLAS-TC result (and the tile/pipeline choices behind
it) transfer from sm_120 to H100? Measured (`ARCH=90 make bench` on one idle GPU of an 8×H100
box):

| check | sm_120 | sm_90 (H100) | verdict |
|---|---|---|---|
| WMMA absolute TFLOP/s @8192 | 103.5 | 60.9 | ratio ≈ SM count × clock (188 SM @ ~2.6 GHz vs 132 SM @ ~1.98 GHz) — `mma.sync`-issue-bound code scales with issue slots, as expected |
| cuBLAS-TC ceiling @8192 | 229.2 (`cutlass_80_tensorop`) | 761.7 (`nvjet_sm90`, wgmma) | H100's ceiling is 3.3× higher *and* uses a different instruction class |
| **WMMA % of cuBLAS-TC** | **45.2%** | **8.0%** | **does NOT transfer** |
| cuBLAS-TC % of card's FP16 peak | ~92% of ~250 | ~77% of 989 | both ceilings are honest |
| FP32 `cublasSgemm` dispatches sm_80 FFMA kernel | yes | yes | the "baseline is an sm_80 kernel" statement holds on both — no scoping needed |
| max abs err (wmma, 8192) | 0.0112 | 0.0112 | bit-identical rounding behavior — same code, same math |

**Why the % collapses (cross-checked against literature):** two stacked effects, and the
literature lets us separate them.

1. *Architectural ceiling for the instruction class.* The WMMA API lowers to per-warp
   `mma.sync` and cannot emit `wgmma.mma_async`. ["Dissecting the NVIDIA Hopper Architecture
   through Microbenchmarking" (arXiv:2501.12084)](https://arxiv.org/abs/2501.12084) measures
   the split precisely on H800: dense FP16 `mma` (m16n8k16) reaches **494.4 TFLOPS = 65.3% of
   the 756.5 peak** ("mma instructions on Hopper can only attain an average of 62.9% of the
   theoretical peak"), while `wgmma` (m64n256k16) reaches **729.3 TFLOPS = 96.4%**.
   ["Benchmarking and Dissecting the Nvidia Hopper GPU Architecture"
   (arXiv:2402.13499)](https://arxiv.org/abs/2402.13499) reports the same qualitative split.
   So even a *perfectly fed* WMMA/`mma.sync` kernel tops out near **~63–65% of peak** on
   Hopper — the last ~35% belongs to `wgmma` alone.
2. *Our kernel is additionally feed-bound, well below that ceiling.* 60.9 TFLOPS = **6.2% of
   peak** — an order of magnitude below the mma.sync ceiling. The ncu profile
   (`results/nsys_profile.md`, H100 section) shows why: 26.7% SM compute with 90.5% L1/shared
   throughput and MIO-queue-full as the top stall — the shared-memory operand pipeline (tile
   size and 3-stage `cp.async` depth tuned on sm_120, whose Tensor Cores are 3.3× slower)
   cannot feed Hopper's TCs. The kernel kept the *same absolute* operand-feed rate on both
   cards, which was 45% of the ceiling on sm_120 and is 8% of the 3.3×-higher ceiling on H100.

The honest summary: the 45.2% → 8.0% collapse is **(ceiling moved 3.3× up) × (our feed
pipeline did not move)**. Hopper-tuned operand staging could in principle recover to roughly
the mma.sync ceiling (~63–65% of peak ≈ ~85% of cuBLAS-TC); the remainder is unreachable
without `wgmma` — i.e., without abandoning the WMMA API this repo is about.

## Sources
- [Benchmarking and Dissecting the Nvidia Hopper GPU Architecture (arXiv:2402.13499)](https://arxiv.org/abs/2402.13499) — Hopper `wgmma` vs `mma` Tensor Core paths.
- [Dissecting the NVIDIA Hopper Architecture through Microbenchmarking (arXiv:2501.12084)](https://arxiv.org/abs/2501.12084) — instruction-level Hopper TC analysis.
- [RTX PRO 6000 Blackwell — NVIDIA](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/)
- [RTX PRO 6000 Blackwell spec sheet — flopper.io](https://flopper.io/gpu/nvidia-rtx-pro-6000-blackwell-workstation-edition)
- [NVIDIA RTX Blackwell PRO GPU Architecture (v1.0 PDF)](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf)
- [Simon Boehm, "How to Optimize a CUDA Matmul Kernel"](https://siboehm.com/articles/22/CUDA-MMM) — quantified ablation: tiling + vectorized loads + warptiling reach ~94% of cuBLAS without TMA or warp specialization.
- [CudaDMA: Optimizing GPU Memory Bandwidth via Warp Specialization (Bauer et al., SC'11)](https://research.nvidia.com/publication/2011-11_cudadma-optimizing-gpu-memory-bandwidth-warp-specialization) — warp specialization requires large tiles + async memory hardware + register reallocation as prerequisites.
- [CUTLASS Efficient GEMM docs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html) — CTA/warp/thread tiling, multistage pipelining, and kernel naming conventions.
