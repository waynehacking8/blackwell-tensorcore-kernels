# Design decisions

**D1 — Three kernels + cuBLAS, on purpose.** naive → tiled → WMMA is the optimization ladder;
each step's speedup is measured, and cuBLAS is the honest ceiling (we report % of it, not a
naked TFLOP/s).

**D2 — WMMA before raw mma.sync / tcgen05.** The WMMA C++ API is portable across Volta→Blackwell
and gets the Tensor Core story working end-to-end. Hand-written `mma.sync` and Blackwell's
`tcgen05` (5th-gen Tensor Cores, FP4/MXFP8) are the next rung, built on this baseline.

**D3 — Same kernels on sm_90 and sm_120.** Building for both H100 and the Blackwell RTX Pro 6000
lets the repo show how the *same* kernel scales across generations — exactly the cross-arch
reasoning a DevTech/SA conversation needs.

**D4 — Profile, don't guess.** Every claim is backed by Nsight Compute (roofline, Tensor Core
utilization, memory throughput), committed under results/.
