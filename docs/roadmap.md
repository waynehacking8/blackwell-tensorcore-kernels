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
- [x] Register-blocked tiled kernel (e.g. 128x128 block, 8x8 per thread).
  *(Subsumed by the mma.sync item below: `mma_base`..`mma_warptile` are 128×128-block,
  register-tiled kernels; the per-warp register tile is the ablation's decisive step.)*
- [x] Double-buffered shared memory; bank-conflict-free layout.
  *(Subsumed by the mma.sync item below: `mma_swizzle` adds the bank-conflict-free XOR-swizzled
  layout, `mma_pipe` the multi-stage — sweep winner: double-buffered — cp.async pipeline.)*
- [x] **Hand-written mma.sync (m16n8k16) replacing the WMMA wrapper — test whether Boehm's
  "~94% with standard optimizations" transfers to the Tensor Core regime.**
  - **Question:** Boehm's ablation (2D blocktiling 68.7% → vectorize 78.4% → warptiling 93.7%
    of cuBLAS) is FP32 SIMT data — CUDA cores. Tensor Cores consume operands ~8× faster, so
    every feed inefficiency is amplified ~8×; and the WMMA wrapper hides fragment register
    layout, blocking register-level pipelining. Is ≥90% of the sm_80-style cuBLAS-TC ceiling
    (229 TFLOP/s on the Pro 6000) still reachable when the compute units are Tensor Cores?
  - **Method (as run):** replace WMMA with raw `mma.sync.aligned.m16n8k16` PTX + `ldmatrix`
    (full register control), then apply the standard stack step by step, benching each as a new
    kernel row (same ablation discipline as the existing wmma progression): 128×128 CTA tile
    with 2×2 per-warp register tiling (`mma_base`) → swizzled bank-conflict-free shared-memory
    layout (`mma_swizzle`) → 16-byte vectorized `cp.async` loads (`mma_vec`) → 2/3/4-stage
    pipeline sweep (`mma_pipe`) → 64×64 per-warp register tile (`mma_warptile`, the analog of
    Boehm's warptiling step). Code: `src/gemm_mma.cu` + `include/mma_ptx.cuh`;
    sweep: `scripts/sweep_mma_stages.sh`.
  - **Result: Boehm's claim transfers — and overshoots. 20.7% → 25.6% → 72.2% → 77.7% →
    106.1% of cuBLAS-TC** (243.2 vs 229.0 TFLOP/s @ 8192³). The final kernel *beats* the
    `cutlass_80_tensorop_s16816gemm_f16_128x64` kernel cuBLAS dispatches on sm_120 — same
    `m16n8k16` instruction class, better tile fit (128×128 CTA / 64×64 warp tile / 2-stage
    pipeline, tuned on-card). The WMMA wrapper was conclusively the blocker: it owns the
    smem→register layout (blocking the swizzle) and caps register tiling at 16×16 fragments
    (blocking the 4:1 mma:ldmatrix ratio that fixes the MIO-queue-full stall from Phase 1).
    ncu confirms the mechanism at the stall level (`scripts/profile_mma_ncu.sh`, run in a
    SYS_ADMIN container): Tensor-pipe utilization 24.7% (wmma) → **85.0%** (ours) vs 74.0%
    (cuBLAS); the MIO-queue-full stall (34.2% of warp cycles) is eliminated — the top stall
    becomes the fixed-latency math dependency that ncu attributes to "already highly optimized
    kernels". Caveat kept honest: this beats *the kernel cuBLAS chose*, not the hardware peak —
    see `results/mma_ablation.md` for the full analysis, validation checks, and raw data pointers.

## Phase 2.5 — Hopper wgmma (new: follows from the Phase 1 H100 result)
- [x] **CUTLASS 3.x / wgmma GEMM on H100 — break the WMMA ceiling with a constructive proof.**
  **DONE — `src/cutlass_sm90.cu` / README Phase 2.5 section / `results/bench_sm90a.csv` /
  `results/ncu_cutlass_8192.txt`.** Result: **640.9 TFLOP/s at 8192³ = 85.5% of cuBLAS-TC
  (nvjet 749.9)** vs WMMA's 8.1% — a 10.5× recovery from the instruction class alone, with
  identical numerical error. ncu signature matches nvjet's class exactly: 168 reg/thread,
  14.1% occupancy, top stall = warpgroup CTA barrier (not WMMA's MIO-feed-bound profile) —
  the kernel verifiably crossed into the wgmma regime. Landed in the 70–90% read-out band:
  the remaining 14.5% is nvjet's tile-shape/cluster autotuning (320×128 vs our fixed
  128×256), a tuning gap, not an instruction-class gap. The warp-specialization reversal is
  documented: required on Hopper, measured-negative on sm_120.
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
**Hardware scoping (corrected 2026-06-02):** `tcgen05` instructions (tensor memory, CTA pairs,
`tcgen05.mma`) are **sm_100 (B200 / datacenter Blackwell) only** — they do not exist on sm_120
(GB202 / RTX Pro 6000). What sm_120 *does* have: 5th-gen Tensor Core FP8 (E4M3/E5M2) and FP4
support through the regular `mma` instruction path. The items below are scoped to what this
repo's hardware can actually run; tcgen05 work is parked as B200-only (see the literature-
ceilings note for the B200 references).

- [x] FP8 (E4M3) Tensor Core GEMM via the sm_120 `mma` path; quality vs FP16; throughput delta.
- [x] FP4 (and MXFP4 block-scaling where the sm_120 mma path supports it) GEMM.
- [x] FP4 vs FP8 vs FP16 throughput/accuracy Pareto on RTX Pro 6000.
  - **Question:** how much throughput do Blackwell-generation formats buy on a workstation
    Blackwell card (sm_120), and at what accuracy cost?
  - **Method (as run):** three kernels on top of Phase 2's `gemm_mma.cu` structure (128×128 CTA,
    64×64 warp tile, swizzle, 2-stage cp.async), in `src/gemm_mma_fp8.cu`: `mma_fp8`
    (`mma.m16n8k32` E4M3), `mma_fp4` (`kind::f8f6f4`, E2M1 in 8-bit containers), `mma_mxfp4`
    (`kind::mxf4.block_scale`, packed E2M1 + UE8M0 scales). cuBLASLt FP8 baseline in
    `src/cublaslt_fp8.cu`. Build target sm_120a (the FP4 kinds need it). Both operands TN layout
    (B transposed at staging, same for our kernels and cuBLASLt).
  - **Result (8192³, raw rows in `bench.csv`, full analysis `results/phase3_lowprec.md`):**
    **FP16 242 → FP8 504 (2.09×) → MXFP4 993 TFLOP/s (4.11×)** — the spec'd 2×/4× of the 5th-gen
    Tensor Cores delivered through the plain mma path. (The MXFP4 4.11× is a *throughput* result:
    block-scale factors are fed as 1.0 / per-tensor, not per-32-element-block — identical hardware
    work, but the numerics do not exercise real per-block MXFP4 scaling; see
    `results/phase3_lowprec.md`.) Accuracy: max_abs_err 0.0112 → 1.4 → 6.0
    (the Pareto's price axis, chart `precision_pareto_sm120.png`). Two findings:
    (1) **unpacked FP4 (kind::f8f6f4) is pointless** — 520 TFLOP/s ≈ FP8 speed at 4× worse error
    (it shares the QMMA pipe; only packed mxf4 reaches OMMA.SF and 2× FP8);
    (2) **our FP8 = 91.0% of cuBLASLt FP8** (504 vs 554) — v2 tuning tested and *rejected* the
    larger-CTA-tile hypothesis (256×128 / 128×256 / 256×256 all measured slower: 469/463/123);
    the gain that worked was register-level fragment double-buffering (+2.2% FP8, +4.3% MXFP4).
    nsys shows cuBLASLt's kernel uses the *same* tile/warp/instruction config — the remaining
    ~9% is xmma-style instruction-level interleaving inside the k-loop (see
    `results/phase3_lowprec.md` Limitations). cuBLAS has no FP4
    path on sm_120, so the MXFP4 number is the card's only measured FP4 GEMM datapoint.
- ~~Blackwell tcgen05 / tensor-memory GEMM~~ — **out of scope on this hardware** (sm_100 only).

## Phase 4 — Literature-ceiling reproductions on available hardware (specified)

Goal: reproduce published ceiling numbers with the same harness and methodology used for the
repo's own kernels, so every external claim becomes a measured row in `results/bench.csv`.

- [ ] **Tawa warp-specialization compiler vs cuBLAS vs this repo's kernels (arXiv:2510.14719, CGO'26).**
  Published target: 79% SM utilization, 1.1× over cuBLAS on H100.
  **Scoping note (2026-06-02):** Tawa is not a standalone tool — the implementation lives in a
  Triton development branch (`triton-lang/triton@aref_auto_ws`); reproducing it requires
  building that Triton fork from source (LLVM build, hours). Deferred. The architectural
  question it poses — does compiler-generated warp specialization reach hand-tuned
  performance — is answered constructively on this box by Phase 2.5: CUTLASS's
  warp-specialized cooperative schedule reaches 85.5% of nvjet with the same ncu stall
  signature.
  - **Question:** can a warp-specialization compiler actually beat cuBLAS on this H100 box, and
    where does it land relative to (a) our WMMA kernel (8.0% of nvjet) and (b) nvjet itself?
  - **Method:** build Tawa on the H100 box; compile GEMM for the repo's benchmark shapes
    (2048/4096/8192); add results as kernel rows; ncu-profile the generated kernel and compare
    its stall profile against ours (MIO-queue-full) and nvjet's (WARPGROUP.ARRIVES).
  - **Read-out:** a three-line TFLOPS-vs-shape chart (WMMA / Tawa / cuBLAS-TC). If Tawa ≥ 1.0×
    cuBLAS, the "compilers can match hand-tuned libraries" claim is reproduced on our hardware;
    its ncu profile shows which async-overlap mechanisms our hand-written kernel lacks.

- [x] **Stream-K work decomposition (arXiv:2301.03598) — fix the wave-quantization tail.**
  Published target: up to 6.7× over data-parallel tiling on quantization-unfriendly shapes.
  **DONE (published number does NOT reproduce) — `results/reference/streamk.txt`.** CUTLASS
  example 47 (sm_80 kernel) on one idle H100: Stream-K vs basic data-parallel lands at
  **0.94×–1.05×** across the default sweep AND three deliberately wave-quantization-unfriendly
  shapes (640×5120×8192 has a 48%-idle final wave → theory predicts ~+24%, measured 0.945×).
  Why the 6.7× doesn't transfer: (a) it was measured on A100 (108 SMs) against specifically
  constructed worst cases; (b) H100's 132 SMs shrink relative tail losses; (c) the sm_80
  kernel's Stream-K reduction/fixup overhead on H100 eats the recovered tail. Honest
  conclusion: Stream-K is a real technique for pathological shapes on the architecture it was
  tuned for, but it is not a free win — and modern cuBLAS (nvjet) already embeds tile-raster
  heuristics that make the baseline hard to beat.
  - **Question:** our WMMA kernel uses classic data-parallel tiling; on shapes where
    (M/tile × N/tile) is not a multiple of SM count, how much of the loss can Stream-K's
    K-loop splitting + atomic fixup recover?
  - **Method:** run CUTLASS example 47 (stream-k) on a sweep including non-SM-divisible shapes;
    optionally implement Stream-K scheduling in our own kernel; measure both on H100 and sm_120.
  - **Read-out:** speedup vs data-parallel tiling per shape; worst-case shape variance reduction.

- [x] **Committed third-party baselines: DeepGEMM / ThunderKittens / FlashAttention-3.**
  Published targets: DeepGEMM FP8 ~1358 TFLOPS (~78% of H100 FP8 peak); ThunderKittens FP8
  ~1500 TFLOPS; FlashAttention-3 FP16 740 TFLOPS (75% of peak).
  **DONE — `scripts/run_reference_benches.sh` / `results/reference/` / README "Third-party
  reference baselines".** Results on one idle H100 of this box (each project's own benchmark,
  unmodified): DeepGEMM FP8 **1523 TFLOPS (77% of peak, 112% of the published H800 number)**,
  DeepGEMM BF16 830 (84%); ThunderKittens FP8 **1465 (74%, 98% of published)**, BF16 775
  (78%), FP8-scaled 985; FlashAttention-3 FP16 fwd **757 TFLOPS (77% of peak, 102% of the
  published 740)** — with free FA2 (388) and cuDNN-attention (689) context rows from the same
  benchmark. Build note: FA3's hopper/setup.py editable install (`pip install -e .`) fails at
  packaging on the CUDA 13.1 container; non-editable `pip install .` works. Key reading:
  published numbers reproduce within ~±10% on this shared box → the repo's own kernel gaps are
  real, not environmental. These rows are the honest ceiling for Phase 3's FP8 work.
  - **Question:** what do the leading open-source kernels actually achieve on *this* H100 box
    (shared, no root, Max-Q-class power limits do not apply here but clock state does)?
  - **Method:** run each project's own benchmark (pip/JIT installs, no root needed); record
    numbers + clock state into `results/`; treat them as reference rows, clearly attributed.
  - **Read-out:** measured-on-our-box vs published table. The gap between the two columns is
    itself a finding (clock governance, shared-box interference, version drift). These rows
    become the honest ceiling for Phase 3's own FP8 work.
