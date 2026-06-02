# Third-party reference baselines on this H100 (measured vs published)

Each row is the project's OWN benchmark, unmodified, run on one idle GPU of this
shared 8x H100 SXM5 box (raw logs + clock state committed alongside). Published
numbers come from each project's paper/README - different boxes, different clocks,
different shapes - so the comparison is a sanity band, not a controlled experiment.

| kernel | measured TFLOPS | % of H100 peak | published TFLOPS | measured/published |
|---|---|---|---|---|
| DeepGEMM BF16 | 830 | 84% | - | - |
| DeepGEMM FP8 | 1523 | 77% | 1358 (H800) | 112% |
| ThunderKittens BF16 | 775 | 78% | ~760 | 102% |
| ThunderKittens FP8 | 1465 | 74% | ~1500 | 98% |
| ThunderKittens FP8 (scaled) | 985 | 50% | - | - |
| FlashAttention-3 FP16 fwd | 757 | 77% | 740 | 102% |
| FlashAttention-2 FP16 fwd | 388 | 39% | - | - |
| cuDNN attention FP16 fwd | 689 | 70% | - | - |

Context (this repo's own committed rows, same box):

| kernel | TFLOPS | % of FP16 peak |
|---|---|---|
| cuBLAS FP16-TC (nvjet_sm90) | 761.7 | 77% |
| this repo's CUTLASS wgmma kernel | 640.9 | 65% |
| this repo's WMMA kernel | 60.9 | 6% |

