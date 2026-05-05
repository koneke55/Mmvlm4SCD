"""Render a Q1-style graphical abstract for the Mmvlm4SCD paper.

The figure summarises the architecture (4 encoders -> fusion -> 2 heads)
and the headline results pulled from the live JSON artefacts. Output:

    paper/figures/graphical_abstract.png   (and .pdf)
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _meanstd(vals):
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(mean(vals)), float(pstdev(vals))


def _load_results():
    base = ROOT / "experiments" / "results" / "mmvlm4scd_default"
    summary = json.load((base / "summary.json").open())
    bdir = ROOT / "experiments" / "results" / "baselines"
    logreg = json.load((bdir / "logreg_severity.json").open())
    cox = json.load((bdir / "cox_survival.json").open())
    return summary, logreg, cox


def _box(ax, x, y, w, h, label, color, fc="white", text_color="black",
         fontsize=9.5, lw=1.4):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.04",
                         linewidth=lw, edgecolor=color, facecolor=fc)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight="bold")


def _arrow(ax, x1, y1, x2, y2, color="#1a3d63"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>",
                                 mutation_scale=12,
                                 color=color, linewidth=1.2))


def main():
    summary, logreg, cox = _load_results()
    aurocs = [r["auroc_ovr"] for r in summary["test_per_seed"]]
    cidxs = [r["c_index"] for r in summary["test_per_seed"]]
    auroc_m, auroc_s = _meanstd(aurocs)
    c_m, c_s = _meanstd(cidxs)

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=200)
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.5)
    ax.axis("off")

    accent = "#1a3d63"
    accent2 = "#7f3d63"

    # Title strip
    ax.text(5.0, 4.25,
            "Mmvlm4SCD: multimodal severity & survival prediction in "
            "Sickle Cell Disease",
            ha="center", va="center", fontsize=12.5, fontweight="bold",
            color=accent)
    ax.text(5.0, 3.95,
            "Synthetic-cohort benchmark (n=2,000) - jointly trained "
            "cross-entropy + Cox partial likelihood",
            ha="center", va="center", fontsize=9.0, style="italic",
            color="#444")

    # Modalities (left column)
    mods = [("Clinical", "#1f78b4"),
            ("Genomic", "#33a02c"),
            ("Imaging", "#ff7f00"),
            ("Temporal", "#6a3d9a")]
    for i, (name, c) in enumerate(mods):
        y = 3.1 - 0.7 * i
        _box(ax, 0.3, y, 1.5, 0.5, name, c, fontsize=9.5)

    # Encoders (column 2)
    for i, (name, c) in enumerate(mods):
        y = 3.1 - 0.7 * i
        _box(ax, 2.2, y, 1.6, 0.5, f"{name} encoder", c,
             fc="#f4f8fc", fontsize=9, text_color=c)
        _arrow(ax, 1.8, y + 0.25, 2.2, y + 0.25, color=c)

    # Fusion box (column 3)
    _box(ax, 4.4, 1.1, 1.7, 1.6,
         "Cross-attention\n+ Transformer\nfusion",
         accent, fc="#eaf0f7", fontsize=9.0, text_color=accent)
    for i in range(4):
        y = 3.1 - 0.7 * i + 0.25
        _arrow(ax, 3.8, y, 4.4, 1.9, color="#888")

    # Heads (column 4)
    _box(ax, 6.4, 2.05, 1.8, 0.55, "Severity head\n(3-way ordinal)",
         accent, fc="#ffffff", fontsize=8.6)
    _box(ax, 6.4, 1.30, 1.8, 0.55, "Survival head\n(Cox risk score)",
         accent2, fc="#ffffff", fontsize=8.6)
    _arrow(ax, 6.1, 2.05, 6.4, 2.32, color=accent)
    _arrow(ax, 6.1, 1.85, 6.4, 1.57, color=accent2)

    # Results panel (column 5)
    _box(ax, 8.45, 1.95, 1.45, 0.65,
         f"AUROC\n{auroc_m:.3f} \u00b1 {auroc_s:.3f}",
         accent, fc="#eaf0f7", fontsize=10, text_color=accent)
    _box(ax, 8.45, 1.20, 1.45, 0.65,
         f"C-index\n{c_m:.3f} \u00b1 {c_s:.3f}",
         accent2, fc="#fbeef6", fontsize=10, text_color=accent2)
    _arrow(ax, 8.2, 2.32, 8.45, 2.27, color=accent)
    _arrow(ax, 8.2, 1.57, 8.45, 1.52, color=accent2)

    # Lower strip: comparison + bullets
    ax.text(0.3, 0.6, "Strong clinical-only baselines:",
            fontsize=9.2, fontweight="bold", color="#222")
    ax.text(0.3, 0.30,
            f"  - Logistic regression  AUROC = {logreg['auroc_ovr']:.3f}     "
            f"   - Cox PH  C-index = {cox['c_index']:.3f}",
            fontsize=8.8, color="#222")

    ax.text(5.0, 0.6, "Findings (Tables 2-5, Figures 6-8):",
            fontsize=9.2, fontweight="bold", color="#222", ha="left")
    ax.text(5.0, 0.32,
            "  - Clinical features dominate decision-time importance\n"
            "  - Fusion strategy differences fall within seed variance\n"
            "  - Subgroup gaps motivate ancestry/age-aware modelling",
            fontsize=8.6, color="#222", ha="left", va="top")

    out_dir = ROOT / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "graphical_abstract.png"
    pdf = out_dir / "graphical_abstract.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
