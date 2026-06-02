#!/usr/bin/env python3
"""Third-party reference baselines: parse raw benchmark logs -> summary table + chart.

Inputs (committed raw logs from scripts/run_reference_benches.sh):
  results/reference/deepgemm.txt   DeepSeek DeepGEMM tests (BF16 + FP8 "Perf ..." lines)
  results/reference/tk.txt         ThunderKittens H100 GEMM harnesses
  results/reference/fa3.txt        FlashAttention-3 hopper/benchmark_attn.py
  results/reference/*.clocks.csv   clock state captured at launch

Outputs:
  results/reference/summary.md     measured-on-this-box vs published table
  results/reference_benches.png    chart (% of H100 peak per project, measured vs published)

Published references (papers / project READMEs, H100 or H800):
  DeepGEMM FP8   ~1358 TFLOPS (DeepSeek README, H800)         H100 FP8 peak 1979 TFLOPS
  ThunderKittens FP8 ~1500 TFLOPS (HazyResearch blog)         H100 FP8 peak 1979
  ThunderKittens BF16 ~760 TFLOPS                             H100 BF16 peak 989
  FlashAttention-3 FP16 fwd 740 TFLOPS (arXiv:2407.08608)     H100 FP16 peak 989
"""
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
REF = os.path.join(REPO, "results", "reference")

PEAK_FP8 = 1979.0   # H100 SXM dense FP8 TFLOPS
PEAK_FP16 = 989.0   # H100 SXM dense FP16/BF16 TFLOPS

NVIDIA_GREEN = "#76b900"
GREY = "#9aa0a6"


def read(name):
    p = os.path.join(REF, name)
    return open(p, errors="replace").read() if os.path.exists(p) else ""


def parse_deepgemm():
    """Max TFLOPS from the 'normal GEMM' sections of DeepGEMM's own tests.
    FP8 rows live under test_fp8_fp4.py, BF16 rows under test_bf16.py."""
    txt = read("deepgemm.txt")
    if not txt:
        return {}
    parts = txt.split("--- tests/test_fp8_fp4.py ---")
    bf16 = [int(m) for m in re.findall(r"(\d+) TFLOPS", parts[0])]
    fp8 = [int(m) for m in re.findall(r"(\d+) TFLOPS", parts[1])] if len(parts) > 1 else []
    out = {}
    if bf16:
        out["DeepGEMM BF16"] = {"tflops": max(bf16), "peak": PEAK_FP16, "published": None,
                                "pub_label": "-"}
    if fp8:
        out["DeepGEMM FP8"] = {"tflops": max(fp8), "peak": PEAK_FP8, "published": 1358,
                               "pub_label": "1358 (H800)"}
    return out


def parse_tk():
    """One 'Achieved performance' line per kernel harness section."""
    txt = read("tk.txt")
    out = {}
    section = None
    for line in txt.splitlines():
        m = re.match(r"--- kernels/gemm/(\S+) ---", line)
        if m:
            section = m.group(1)
        m = re.search(r"Achieved performance: ([\d.]+) TFLOPs", line)
        if m and section:
            tf = float(m.group(1))
            if section == "bf16_h100":
                out["ThunderKittens BF16"] = {"tflops": tf, "peak": PEAK_FP16,
                                              "published": 760, "pub_label": "~760"}
            elif section == "fp8_h100":
                out["ThunderKittens FP8"] = {"tflops": tf, "peak": PEAK_FP8,
                                             "published": 1500, "pub_label": "~1500"}
            elif section == "fp8_h100_scaled":
                out["ThunderKittens FP8 (scaled)"] = {"tflops": tf, "peak": PEAK_FP8,
                                                      "published": None, "pub_label": "-"}
    return out


def parse_fa3():
    """Max FA3 forward TFLOPS from benchmark_attn.py output (the file also reports FA2 and
    cuDNN baselines - match the Fav3 lines only)."""
    txt = read("fa3.txt")
    vals = [float(m) for m in re.findall(r"Fav3 fwd:[^,]+, ([\d.]+) TFLOPS", txt)]
    if not vals:
        return {}
    out = {"FlashAttention-3 FP16 fwd": {"tflops": max(vals), "peak": PEAK_FP16,
                                         "published": 740, "pub_label": "740"}}
    # the same file gives an in-container FA2 + cuDNN attention reference for free
    fa2 = [float(m) for m in re.findall(r"Fav2 fwd:[^,]+, ([\d.]+) TFLOPS", txt)]
    cudnn = [float(m) for m in re.findall(r"CuDNN fwd:[^,]+, ([\d.]+) TFLOPS", txt)]
    if fa2:
        out["FlashAttention-2 FP16 fwd"] = {"tflops": max(fa2), "peak": PEAK_FP16,
                                            "published": None, "pub_label": "-"}
    if cudnn:
        out["cuDNN attention FP16 fwd"] = {"tflops": max(cudnn), "peak": PEAK_FP16,
                                           "published": None, "pub_label": "-"}
    return out


def main():
    rows = {}
    rows.update(parse_deepgemm())
    rows.update(parse_tk())
    rows.update(parse_fa3())
    if not rows:
        sys.exit("no reference results found under results/reference/")

    # repo context rows (from the committed bench data: bench.csv + bench_sm90a.csv, 8192^3)
    context = {"cuBLAS FP16-TC (nvjet_sm90)": {"tflops": 761.7, "peak": PEAK_FP16},
               "this repo's CUTLASS wgmma kernel": {"tflops": 640.9, "peak": PEAK_FP16},
               "this repo's WMMA kernel": {"tflops": 60.9, "peak": PEAK_FP16}}

    # ---- summary.md ----
    L = ["# Third-party reference baselines on this H100 (measured vs published)\n",
         "Each row is the project's OWN benchmark, unmodified, run on one idle GPU of this",
         "shared 8x H100 SXM5 box (raw logs + clock state committed alongside). Published",
         "numbers come from each project's paper/README - different boxes, different clocks,",
         "different shapes - so the comparison is a sanity band, not a controlled experiment.\n",
         "| kernel | measured TFLOPS | % of H100 peak | published TFLOPS | measured/published |",
         "|---|---|---|---|---|"]
    for name, r in rows.items():
        pct = r["tflops"] / r["peak"] * 100
        ratio = f"{r['tflops'] / r['published'] * 100:.0f}%" if r["published"] else "-"
        L.append(f"| {name} | {r['tflops']:.0f} | {pct:.0f}% | {r['pub_label']} | {ratio} |")
    L.append("")
    L.append("Context (this repo's own committed rows, same box):\n")
    L.append("| kernel | TFLOPS | % of FP16 peak |")
    L.append("|---|---|---|")
    for name, r in context.items():
        L.append(f"| {name} | {r['tflops']:.1f} | {r['tflops'] / r['peak'] * 100:.0f}% |")
    L.append("")
    out_md = os.path.join(REF, "summary.md")
    with open(out_md, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"wrote {out_md}")

    # ---- chart: % of peak, measured vs published ----
    fig, ax = plt.subplots(figsize=(11, 5.5))
    names = list(rows.keys()) + list(context.keys())
    measured_pct = [r["tflops"] / r["peak"] * 100 for r in rows.values()] + \
                   [r["tflops"] / r["peak"] * 100 for r in context.values()]
    published_pct = [(r["published"] / r["peak"] * 100 if r["published"] else 0)
                     for r in rows.values()] + [0] * len(context)
    xpos = range(len(names))
    width = 0.38
    b1 = ax.bar([x - width / 2 for x in xpos], measured_pct, width,
                label="measured on this box", color=NVIDIA_GREEN, edgecolor="black", linewidth=0.6)
    b2 = ax.bar([x + width / 2 for x in xpos], published_pct, width,
                label="published", color=GREY, edgecolor="black", linewidth=0.6)
    for bars in (b1, b2):
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9,
                        fontweight="bold")
    def wrap(name, width=16):
        """Wrap on spaces into lines of <= width chars (first-space-only breaks collide)."""
        words, lines, cur = name.split(), [], ""
        for w in words:
            if cur and len(cur) + 1 + len(w) > width:
                lines.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        lines.append(cur)
        return "\n".join(lines)

    ax.set_xticks(list(xpos))
    ax.set_xticklabels([wrap(n) for n in names], fontsize=8)
    ax.set_ylabel("% of H100 dense peak (per precision)")
    ax.set_ylim(0, 100)
    ax.set_title("Leading open-source kernels: measured on this box vs published\n"
                 "(the WMMA bar is why Phase 2.5 exists)", fontsize=12, pad=12)
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out_png = os.path.join(REPO, "results", "reference_benches.png")
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
