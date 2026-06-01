# Roadmap

## Phase 1 — FP16 Tensor Core baseline (this scaffold)
- [ ] naive / tiled / WMMA build for sm_90 and sm_120; correctness vs cuBLAS.
- [ ] TFLOP/s table + % of cuBLAS on H100 and on RTX Pro 6000.
- [ ] Nsight Compute roofline for the WMMA kernel — use ncu to **quantify the contribution of CTA
      tile size / register-level warp/thread tiling / multistage pipelining depth** to closing the
      gap vs the Ampere-style (sm_80) cuBLAS baseline (not TMA/warp-spec, which the baseline doesn't use).

## Phase 2 — Better SIMT + Tensor Core
- [ ] Register-blocked tiled kernel (e.g. 128x128 block, 8x8 per thread).
- [ ] Double-buffered shared memory; bank-conflict-free layout.
- [ ] Hand-written mma.sync (m16n8k16) replacing the WMMA wrapper.

## Phase 3 — Blackwell formats (the frontier)
- [ ] FP8 (E4M3) Tensor Core GEMM; quality vs FP16; throughput delta.
- [ ] Blackwell tcgen05 / MXFP4 microscaling GEMM; tensor-memory accelerator (TMA) loads.
- [ ] FP4 vs FP8 vs FP16 throughput/accuracy Pareto on RTX Pro 6000.
