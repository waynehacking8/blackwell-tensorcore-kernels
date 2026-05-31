#!/usr/bin/env python3
"""Aggregate results/bench.csv into per-GPU TFLOP/s curves and a results/report.md.

No pandas dependency — stdlib csv + matplotlib. One chart per device (TFLOP/s vs
matrix size, per kernel) plus a summary table at the largest size, including each
hand-written kernel's percentage of the cuBLAS ceiling.
"""
import csv, os, collections

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV  = os.path.join(HERE, "results", "bench.csv")
KORDER = ["naive", "tiled", "wmma", "cublas"]


def load(path):
    if not os.path.exists(path):
        raise SystemExit(f"no {path} — run `make bench` on the GPU box first")
    with open(path) as f:
        return [r for r in csv.DictReader(f)]


def main():
    rows = load(CSV)
    # rows[device][kernel] = list of (size, tflops, pct, err)
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by[r["device"]][r["kernel"]].append(
            (int(r["M"]), float(r["tflops"]), float(r["pct_of_cublas"]), float(r["max_abs_err"]))
        )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        have_plt = True
    except ImportError:
        have_plt = False
        print("matplotlib not installed — writing report.md only (no PNGs)")

    lines = ["# Blackwell vs Hopper Tensor Core GEMM — results\n"]
    for device in sorted(by):
        kerns = by[device]
        sizes = sorted({s for k in kerns.values() for (s, *_ ) in k})
        sm = next((r["sm"] for r in rows if r["device"] == device), "?")
        lines.append(f"## {device} (sm_{sm})\n")

        if have_plt:
            plt.figure(figsize=(7, 4.5))
            for k in KORDER:
                if k in kerns:
                    pts = sorted(kerns[k])
                    plt.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=k)
            plt.xlabel("matrix size (M=N=K)"); plt.ylabel("TFLOP/s")
            plt.title(f"GEMM throughput — {device}"); plt.legend(); plt.grid(True, alpha=0.3)
            png = os.path.join(HERE, "results", f"tflops_sm{sm}.png")
            plt.tight_layout(); plt.savefig(png, dpi=130); plt.close()
            lines.append(f"![throughput](tflops_sm{sm}.png)\n")

        big = max(sizes)
        lines.append(f"At M=N=K={big}:\n")
        lines.append("| kernel | TFLOP/s | % of cuBLAS | max abs err |")
        lines.append("|---|---|---|---|")
        for k in KORDER:
            if k in kerns:
                row = next((r for r in sorted(kerns[k]) if r[0] == big), None)
                if row:
                    _, tf, pct, err = row
                    lines.append(f"| {k} | {tf:.1f} | {pct:.1f}% | {err:.3g} |")
        lines.append("")

    out = os.path.join(HERE, "results", "report.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f">> wrote {out}" + ("" if have_plt else " (no PNGs — install matplotlib)"))


if __name__ == "__main__":
    main()
