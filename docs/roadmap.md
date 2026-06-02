# Roadmap

## Phase 1 — FP16 Tensor Core baseline (this scaffold)
- [x] naive / tiled / WMMA build for sm_90 and sm_120; correctness vs cuBLAS.
- [x] TFLOP/s table + % of cuBLAS on H100 and on RTX Pro 6000.
  - **Question:** does the 45.2%-of-cuBLAS-TC result (and the tile-size / pipeline-depth choices
    behind it) transfer from sm_120 to H100, or is it architecture-specific?
  - **Method (as run):** on the H100 box (one idle GPU of an 8×H100, in
    `nvcr.io/nvidia/pytorch:26.02-py3`): `ARCH=90 bash scripts/run_bench.sh` → rows appended to
    `results/bench.csv`; `make analyze` → `results/report.md` + sm_90 charts;
    `bash scripts/profile.sh 8192` → nsys capture of which kernel H100's cuBLAS dispatches.
  - **Result: it does NOT transfer — 45.2% (sm_120) → 8.0% (sm_90).** The kernel's *absolute*
    speed transfers (103 vs 61 TFLOP/s, tracking SM count × clock); the *fraction of ceiling*
    collapses because H100's cuBLAS-TC dispatches `nvjet_sm90` (a `wgmma` warpgroup kernel,
    762 TFLOP/s = 77% of the 989 peak) and the WMMA API cannot emit `wgmma`. H100's FP32 cuBLAS
    still dispatches an sm_80 FFMA kernel, so the VALIDATION.md baseline statement needs no
    scoping. Cross-checked against arXiv:2402.13499 / arXiv:2501.12084. See README (H100
    section), `VALIDATION.md` (H100 cross-check), `results/nsys_profile.md` (H100 profile).
- [x] Nsight Compute side-by-side profile of `gemm_wmma` vs the cuBLAS FP16-TC kernel.
  - **Method (as run):** `ncu --set full` on `gemm_wmma_t<128,128,3>` and on `nvjet_sm90` (the
    kernel H100 actually dispatches for FP16-TC; the planned `cutlass_80` regex matched the TF32
    kernel instead, also captured) at 8192³, with `--cap-add=SYS_ADMIN` for counter access.
  - **Result (per-factor breakdown written to `results/nsys_profile.md`):** ours = 26.7% SM
    compute, 90.5% L1/shared throughput, 6.2% DRAM, 49.4% occupancy, top stall MIO-queue-full →
    shared-memory-feed-bound. cuBLAS nvjet = 91.9% compute at 14.8% occupancy, 168 reg/thread,
    stalls only on WARPGROUP.ARRIVES → wgmma reads operands from shared memory asynchronously,
    so the feeding work that saturates our MIO pipe doesn't exist as warp instructions. On
    Hopper the gap is *architectural* (instruction class), not a tiling-quality gap.

## Phase 2 — Better SIMT + Tensor Core
- [ ] Register-blocked tiled kernel (e.g. 128x128 block, 8x8 per thread).
- [ ] Double-buffered shared memory; bank-conflict-free layout.
- [ ] **Hand-written mma.sync (m16n8k16) replacing the WMMA wrapper — test whether Boehm's
  "~94% with standard optimizations" transfers to the Tensor Core regime.**
  - **Question:** Boehm's ablation (2D blocktiling 68.7% → vectorize 78.4% → warptiling 93.7%
    of cuBLAS) is FP32 SIMT data — CUDA cores. Tensor Cores consume operands ~8× faster, so
    every feed inefficiency is amplified ~8×; and the WMMA wrapper hides fragment register
    layout, blocking register-level pipelining. Is ≥90% of the sm_80-style cuBLAS-TC ceiling
    (229 TFLOP/s on the Pro 6000) still reachable when the compute units are Tensor Cores?
  - **Method:** replace WMMA with raw `mma.sync.aligned.m16n8k16` PTX (full register control),
    then apply the standard stack step by step, benching each as a new kernel row (same
    ablation discipline as the existing wmma progression): 128×128 CTA tile with 2×2 per-warp
    register tiling → swizzled (bank-conflict-free) shared-memory layout → 16-byte vectorized
    `cp.async` loads → 3/4-stage pipeline sweep.
  - **Read-out:** the per-step ablation table vs cublas_tc. Final kernel ≥85–90% → Boehm's
    claim transfers to the TC regime (and the WMMA wrapper was the blocker). Stalls at
    ~60–70% → the remaining gap quantifies what CUTLASS's deeper instruction scheduling buys —
    either result is a finding.

## Phase 2.5 — Hopper wgmma (new: follows from the Phase 1 H100 result)
- [ ] **CUTLASS 3.x / wgmma GEMM on H100 — break the WMMA ceiling with a constructive proof.**
  - **Question:** the H100 run established that WMMA cannot exceed ~65% of peak on Hopper
    (cannot emit `wgmma`); that conclusion currently rests on literature + ncu evidence.
    Does a from-scratch CUTLASS 3.x kernel (CuTe, warpgroup MMA + TMA) actually recover the
    gap — i.e., approach nvjet's 77%-of-peak?
  - **Method:** implement the GEMM with CUTLASS 3.x `CollectiveBuilder` (sm_90, TMA + wgmma);
    add as kernel row `cutlass_sm90`; bench at 8192³ on the H100 box (`ARCH=90`); ncu-profile
    to confirm the kernel actually issues wgmma (top stall should become WARPGROUP.ARRIVES,
    matching nvjet's signature).
  - **Read-out:** % of cuBLAS-TC (nvjet, 761.7 TFLOP/s). ≥90% → the ceiling conclusion closes
    with a constructive proof. 70–90% → quantifies the remaining CUTLASS-vs-nvjet tuning gap.
    Also document the reversal: warp specialization (producer/consumer warpgroups) is
    *required* in this regime — the same technique that was a measured negative result on
    sm_120.

## Phase 3 — Blackwell formats (the frontier)
- [ ] FP8 (E4M3) Tensor Core GEMM; quality vs FP16; throughput delta.
- [ ] Blackwell tcgen05 / MXFP4 microscaling GEMM; tensor-memory accelerator (TMA) loads.
- [ ] FP4 vs FP8 vs FP16 throughput/accuracy Pareto on RTX Pro 6000.
  - **Question:** how much throughput do Blackwell-generation formats buy, and at what accuracy
    cost?
  - **Method:** implement the FP8 / tcgen05 kernels on top of the working FP16 WMMA kernel; add
    them as new kernel rows to `make bench`; plot the throughput (TFLOP/s) vs accuracy
    (max_abs_err) Pareto.
  - **Read-out:** the Pareto frontier per precision; compare against cuBLAS's FP8 path where
    available.
