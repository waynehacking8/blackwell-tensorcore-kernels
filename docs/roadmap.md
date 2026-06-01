# Roadmap

## Phase 1 — FP16 Tensor Core baseline (this scaffold)
- [ ] naive / tiled / WMMA build for sm_90 and sm_120; correctness vs cuBLAS.
- [ ] TFLOP/s table + % of cuBLAS on H100 and on RTX Pro 6000.
  - **Question:** does the 45.2%-of-cuBLAS-TC result (and the tile-size / pipeline-depth choices
    behind it) transfer from sm_120 to H100, or is it architecture-specific?
  - **Method:** on the H100 box: `ARCH=90 make bench` (rows append into the same
    `results/bench.csv`, keyed by the device column), then `make analyze`. Also run
    `nsys profile ./build/gemm_bench 4096 4096 4096 /tmp/b.csv` to capture which kernel H100's
    cuBLAS dispatches to.
  - **Read-out:** % of cuBLAS-TC still ~40–50% → the optimization strategy generalizes across
    generations; significantly different → explain via SM count / shared-memory size / clocks.
    If H100's cuBLAS dispatches an sm_90 kernel (TMA / warp-spec), the "baseline is an sm_80
    kernel" statement in VALIDATION.md must be scoped to RTX Pro 6000.
- [ ] Nsight Compute side-by-side profile of `gemm_wmma` vs the cuBLAS `cutlass_80` kernel — use
      ncu to **quantify the contribution of CTA tile size / register-level warp/thread tiling /
      multistage pipelining depth** to closing the gap vs the Ampere-style (sm_80) cuBLAS baseline
      (not TMA/warp-spec, which the baseline doesn't use).
  - **Question:** the remaining 45.2% → 100% gap is qualitatively attributed (larger CTA tile +
    register tiling + vectorized loads + deeper pipeline, per the dispatched kernel name and
    Boehm's ablation); what is each factor's quantified share?
  - **Method:** profile both kernels with identical settings:
    ```bash
    ncu --set full -k "regex:gemm_wmma" -c 1 -f -o results/ncu_wmma_8192 \
        ./build/gemm_bench 8192 8192 8192 /tmp/b.csv
    ncu --set full -k "regex:cutlass" -c 1 -f -o results/ncu_cublas_8192 \
        ./build/gemm_bench 8192 8192 8192 /tmp/b.csv
    ```
  - **Read-out:** `sm__pipe_tensor_cycles_active` ratio ≈ the "Tensor Core idle" share of the gap;
    `smsp__warp_issue_stalled_long_scoreboard` high on our side → pipeline/tile still insufficient;
    `dram__bytes` higher than cuBLAS → data-reuse gap (direct consequence of tile size); achieved
    occupancy → cross-check the 3-stage > 4/5-stage pipeline result. Write the per-factor breakdown
    back into `results/nsys_profile.md`.

## Phase 2 — Better SIMT + Tensor Core
- [ ] Register-blocked tiled kernel (e.g. 128x128 block, 8x8 per thread).
- [ ] Double-buffered shared memory; bank-conflict-free layout.
- [ ] Hand-written mma.sync (m16n8k16) replacing the WMMA wrapper.

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
