#!/usr/bin/env python3
"""Regenerate Paper II's equation figure FROM the paper's corrected text.

The April 2026 image highlighted "alpha > 1: Compounding returns (sequential
recursion)" in gold as THE KEY INSIGHT: a verdict, and it moved. The corrected
figure keeps the equation and the regime DEFINITIONS (which are timeless) and
adds the measured state as the paper now reports it, extracted from the text
at generation time, so the figure can never again outrun the paper.

Usage: python3 tools/generate_paper_ii_fig8.py <paper-ii.html> <out.png>
"""
import sys, re, html
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def text_of(path):
    t = open(path, encoding="utf-8").read()
    p = html.unescape(re.sub(r"<[^>]+>", " ", t))
    return re.sub(r"\s+", " ", p)

def main(src, dst):
    p = text_of(src)
    m = re.search(r"alpha_\{\\text\{seq\}\} \\approx (0\.\d+)|narrowed the defensible claim to \$?\\?alpha[^0-9]{0,30}(0\.\d+)", p)
    if not m:
        raise SystemExit("corrected alpha_seq not found; refusing to invent")
    a = m.group(1) or m.group(2)
    assert "directional" in p.lower() and "revised" in p.lower(), "revision statement absent"

    fig, ax = plt.subplots(figsize=(14, 8), dpi=200)
    ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis("off")
    ax.add_patch(FancyBboxPatch((3.4, 7.9), 7.2, 1.5, boxstyle="round,pad=0.12",
                                fc="#f7f0e0", ec="#d9c9a3", lw=1.2))
    ax.text(7, 8.65, r"$E(R) = E_0 \times R^{-\alpha}$", ha="center", va="center",
            fontsize=30, color="#14202e")
    cards = [
        ("E(R)", "Error rate\nat depth R", "#8a6d1a", "#f5edd8"),
        (r"$E_0$", "Baseline\nerror rate", "#555555", "#ececec"),
        ("R", "Recursive\ndepth", "#1a5f8a", "#e2eef7"),
        (r"$\alpha$", "Scaling\nexponent", "#b02540", "#fbe4ea"),
    ]
    for k, (sym, lab, col, bg) in enumerate(cards):
        x = 1.1 + k * 3.3
        ax.add_patch(FancyBboxPatch((x, 4.6), 2.4, 2.3, boxstyle="round,pad=0.08",
                                    fc=bg, ec="#cccccc", lw=1))
        ax.text(x + 1.2, 6.25, sym, ha="center", fontsize=19, fontweight="bold", color=col)
        ax.text(x + 1.2, 5.35, lab, ha="center", fontsize=11.5, color="#333333")
    ax.text(7, 3.6, "THE REGIMES, BY DEFINITION", ha="center", fontsize=13,
            fontweight="bold", family="serif", color="#14202e")
    ax.text(7, 3.0, "alpha < 1: diminishing returns (parallel recursion)     "
                    "alpha > 1: compounding returns (sequential recursion)",
            ha="center", fontsize=12, color="#444444")
    ax.add_patch(FancyBboxPatch((1.6, 0.7), 10.8, 1.6, boxstyle="round,pad=0.12",
                                fc="#f2f0e6", ec="#cccccc", lw=1))
    ax.text(7, 1.85, "MEASURED STATE (six models, corrected)", ha="center",
            fontsize=11, fontweight="bold", color="#14202e")
    ax.text(7, 1.15,
            f"Directional finding holds: sequential > parallel wherever both were measurable.\n"
            f"The quantitative alpha > 1 claim is revised: best clean fit alpha_seq = {a} (sub-linear); behaviour is architecture-dependent.",
            ha="center", fontsize=10.5, color="#333333")
    ax.text(7, 0.15, "Generated from the paper: definitions are timeless; the measured state is "
                     "extracted from the text at generation time.",
            ha="center", fontsize=9, style="italic", color="#57574e")
    fig.savefig(dst, bbox_inches="tight", facecolor="white")
    print(f"wrote {dst}: definitions + measured state (alpha_seq = {a})")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
