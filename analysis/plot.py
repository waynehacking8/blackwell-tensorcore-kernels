#!/usr/bin/env python3
"""Aggregate results/bench.csv into per-GPU charts + results/report.md.

Stdlib csv + matplotlib (no pandas). Per device it emits:
  * tflops_sm<sm>.png      — throughput vs size, precision ladder
  * pct_tc_sm<sm>.png      — each kernel as % of the same-precision cuBLAS-TC ceiling
  * roofline_sm<sm>.png     — TFLOP/s bar at the largest size (precision ladder)
and a summary table (% of FP32 cuBLAS and % of cuBLAS-TC) in report.md.
"""
import csv, os, collections

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV  = os.path.join(HERE, "results", "bench.csv")
# precision ladder: cublas=Sgemm(FP32,CUDA cores); cublas_tf32=GemmEx(FP32 in/TF32 compute,TC);
# cublas_tc=GemmEx(FP16 in/FP32 acc,TC) — the honest same-precision ceiling for wmma.
KORDER = ["naive", "tiled", "wmma", "cublas", "cublas_tf32", "cublas_tc"]
# consistent colour + style per kernel across all charts
STYLE = {
    "naive":       ("#9aa0a6", "o", "naive (CUDA core)"),
    "tiled":       ("#5f6368", "s", "tiled (CUDA core)"),
    "wmma":        ("#1a73e8", "D", "wmma (our kernel, FP16-TC)"),
    "cublas":      ("#b31412", "^", "cuBLAS FP32 (cublasSgemm)"),
    "cublas_tf32": ("#e8710a", "v", "cuBLAS TF32-TC (cublasGemmEx)"),
    "cublas_tc":   ("#188038", "*", "cuBLAS FP16-TC (cublasGemmEx) — ceiling"),
}


def load(path):
    if not os.path.exists(path):
        raise SystemExit(f"no {path} — run `make bench` on the GPU box first")
    with open(path) as f:
        return [r for r in csv.DictReader(f)]


def main():
    rows = load(CSV)
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by[r["device"]][r["kernel"]].append(
            (int(r["M"]), float(r["tflops"]), float(r["pct_of_cublas"]), float(r["max_abs_err"]))
        )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try: plt.style.use("seaborn-v0_8-whitegrid")
        except Exception:
            try: plt.style.use("seaborn-whitegrid")
            except Exception: pass
        have_plt = True
    except ImportError:
        have_plt = False
        print("matplotlib not installed — writing report.md only (no PNGs)")

    lines = ["# Blackwell vs Hopper Tensor Core GEMM — results\n"]
    for device in sorted(by):
        kerns = by[device]
        sizes = sorted({s for k in kerns.values() for (s, *_ ) in k})
        sm = next((r["sm"] for r in rows if r["device"] == device), "?")
        big = max(sizes)
        short = device.replace("NVIDIA ", "").replace(" Workstation Edition", "")
        lines.append(f"## {device} (sm_{sm})\n")

        if have_plt:
            # ---- Chart 1: throughput vs size (log-x), precision ladder ----
            fig, ax = plt.subplots(figsize=(8, 5))
            for k in KORDER:
                if k in kerns:
                    c, m, lab = STYLE[k]
                    pts = sorted(kerns[k])
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=m, color=c,
                            lw=2, ms=8, label=lab)
            ax.set_xscale("log", base=2); ax.set_xticks(sizes)
            ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
            ax.set_xlabel("matrix size  M=N=K"); ax.set_ylabel("throughput (TFLOP/s)")
            ax.set_title(f"GEMM throughput — {short} (sm_{sm})\nFP32 → TF32 → FP16 precision ladder, all on one card")
            # legend below the axes so it can never overlap the data lines
            leg = ax.legend(fontsize=8, framealpha=1.0, loc="upper center",
                            bbox_to_anchor=(0.5, -0.13), ncol=3)
            ax.grid(True, alpha=.3)
            p1 = f"tflops_sm{sm}.png"
            fig.savefig(os.path.join(HERE,"results",p1), dpi=140,
                        bbox_inches="tight", bbox_extra_artists=[leg]); plt.close(fig)

            # ---- Chart 2: % of cuBLAS-TC (same-precision ceiling) ----
            tc = {s: tf for (s, tf, *_ ) in sorted(kerns.get("cublas_tc", []))}
            fig, ax = plt.subplots(figsize=(8, 5))
            for k in ["naive","tiled","wmma","cublas","cublas_tf32"]:
                if k in kerns and tc:
                    c, m, lab = STYLE[k]
                    pts = [(s, 100*tf/tc[s]) for (s,tf,*_ ) in sorted(kerns[k]) if s in tc]
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=m, color=c, lw=2, ms=8, label=lab)
            ax.axhline(100, color=STYLE["cublas_tc"][0], ls="--", lw=1.5, label="cuBLAS FP16-TC ceiling (100%)")
            ax.set_xscale("log", base=2); ax.set_xticks(sizes)
            ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
            ax.set_xlabel("matrix size  M=N=K"); ax.set_ylabel("% of cuBLAS FP16-TC (same precision)")
            ax.set_title(f"Fraction of the honest Tensor Core ceiling — {short} (sm_{sm})")
            ax.set_ylim(0, 112)  # headroom so the 100% ceiling line sits clear of the frame
            # legend below the axes so the 100% dashed line never crosses it
            leg = ax.legend(fontsize=8, framealpha=1.0, loc="upper center",
                            bbox_to_anchor=(0.5, -0.13), ncol=3)
            ax.grid(True, alpha=.3)
            p2 = f"pct_tc_sm{sm}.png"
            fig.savefig(os.path.join(HERE,"results",p2), dpi=140,
                        bbox_inches="tight", bbox_extra_artists=[leg]); plt.close(fig)

            # ---- Chart 3: TFLOP/s bar at largest size (precision ladder) ----
            fig, ax = plt.subplots(figsize=(8, 5))
            present = [k for k in KORDER if k in kerns]
            vals = [next((tf for (s,tf,*_ ) in sorted(kerns[k]) if s==big), 0) for k in present]
            cols = [STYLE[k][0] for k in present]
            labs = [STYLE[k][2].split(" (")[0] for k in present]
            bars = ax.bar(range(len(present)), vals, color=cols)
            for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
            ax.set_xticks(range(len(present))); ax.set_xticklabels(labs, rotation=20, ha="right", fontsize=8)
            ax.set_ylabel("TFLOP/s"); ax.set_title(f"Throughput at M=N=K={big} — {short} (sm_{sm})")
            ax.grid(True, axis="y", alpha=.3)
            p3 = f"roofline_sm{sm}.png"
            fig.tight_layout(); fig.savefig(os.path.join(HERE,"results",p3), dpi=140); plt.close(fig)

            lines.append(f"![throughput vs size]({p1})\n")
            lines.append(f"![% of cuBLAS-TC]({p2})\n")
            lines.append(f"![throughput bar at {big}]({p3})\n")

        tc_tf = next((r[1] for r in sorted(kerns.get("cublas_tc", [])) if r[0] == big), None)
        lines.append(f"At M=N=K={big}:\n")
        lines.append("| kernel | TFLOP/s | % of FP32 cuBLAS | % of cuBLAS-TC | max abs err |")
        lines.append("|---|---|---|---|---|")
        for k in KORDER:
            if k in kerns:
                row = next((r for r in sorted(kerns[k]) if r[0] == big), None)
                if row:
                    _, tf, pct, err = row
                    pct_tc = f"{100.0 * tf / tc_tf:.1f}%" if tc_tf else "n/a"
                    lines.append(f"| {k} | {tf:.1f} | {pct:.1f}% | {pct_tc} | {err:.3g} |")
        lines.append("")
        lines.append(
            "> Precision ladder, all on the **same card**: **cublas** = `cublasSgemm` "
            "(FP32, CUDA cores); **cublas_tf32** = `cublasGemmEx` (FP32 in, TF32 compute, "
            "Tensor Cores); **cublas_tc** = `cublasGemmEx` (FP16 in / FP32 acc, Tensor Cores) "
            "— the honest same-precision ceiling for `wmma`. **% of FP32 cuBLAS** is "
            "precision-mismatched (FP16/TF32-TC vs FP32-CUDA-core), so its `>100%` rows are "
            "**not** a kernel beating cuBLAS. Quote **% of cuBLAS-TC**.\n"
        )

    out = os.path.join(HERE, "results", "report.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f">> wrote {out}" + ("" if have_plt else " (no PNGs — install matplotlib)"))


if __name__ == "__main__":
    main()
