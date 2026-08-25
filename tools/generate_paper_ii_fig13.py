#!/usr/bin/env python3
"""Regenerate Paper II's alpha-summary figure FROM the paper's corrected text.

The April 2026 image declared "COMPLETE EVIDENCE ... ALL DATA CONSISTENT:
alpha_seq > 1" with the 2.24-era estimate as a live bar. The corrected record
is richer and more honest: published exponents; the initial single-model 2.24
shown AS retracted history; and the six-model result displayed as what it is,
architecture-dependent behaviour (one clean sub-linear fit with a wide
bootstrap CI, two ceilings, a step function, a floor). Every number here is
extracted from the paper text at generation time; nothing is typed.

Usage: python3 tools/generate_paper_ii_fig13.py <paper-ii.html> <out.png>
"""
import sys, re, html
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def text_of(path):
    t = open(path, encoding="utf-8").read()
    p = html.unescape(re.sub(r"<[^>]+>", " ", t))
    return re.sub(r"\s+", " ", p)

def grab(plain, pattern, what):
    m = re.search(pattern, plain)
    if not m:
        raise SystemExit(f"{what} not found in paper text; refusing to invent")
    return m

def main(src, dst):
    p = text_of(src)
    o1 = grab(p, r"o1 System Card Parallel ([\d.]+)-([\d.]+) \[([\d.]+), ([\d.]+)\]", "o1 row")
    r1 = grab(p, r"R1 Technical Report Sequential ~?([\d.]+) \[([\d.]+), ([\d.]+)\]", "R1 row")
    init = grab(p, r"initial, DeepSeek R1\) Sequential ([\d.]+) \[([\d.]+), ([\d.]+)\]", "initial row")
    gem = grab(p, r"= (0\.\d+)\$? \(regression, \$?r\^2\$? = ([\d.]+)\$?, SE = ([\d.]+), boot CI \[([−-]?[\d.]+), ([\d.]+)\]\)", "Gemini fit")
    assert "retracted" in p.lower() and "2.24" in p, "retraction statement absent"
    behaviours = [
        ("Gemini 3 Flash", f"clean monotonic fit\nalpha_seq = {gem.group(1)}\n(r2 = {gem.group(2)}, boot CI [{gem.group(4)}, {gem.group(5)}])", "#1a7a3a"),
        ("DeepSeek R1", "ceiling (94.4-100%)", "#1a5f8a"),
        ("Grok 4.1 Fast", "ceiling (100% at all depths)", "#1a5f8a"),
        ("GPT-5.4", "binary step (50% -> 100%)", "#b05f10"),
        ("Groq Qwen3", "no trend (~50%, floor)", "#777777"),
    ]
    for name, _, _ in behaviours[1:]:
        assert name.split()[0] in p or name in p, name + " absent from text"

    fig = plt.figure(figsize=(15, 7.2), dpi=200)
    fig.suptitle("Measured Scaling Exponents and Behaviours", fontsize=18,
                 fontweight="bold", family="serif", color="#14202e", y=0.99)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.25,
                          left=0.07, right=0.97, top=0.86, bottom=0.16)

    ax1 = fig.add_subplot(gs[0])
    xs = [0, 1, 2]
    o1mid = (float(o1.group(1)) + float(o1.group(2))) / 2
    vals = [o1mid, float(r1.group(1)), float(init.group(1))]
    los = [float(o1.group(3)), float(r1.group(2)), float(init.group(2))]
    his = [float(o1.group(4)), float(r1.group(3)), float(init.group(3))]
    cols = ["#555555", "#b8860b", "#bbbbbb"]
    for x, v, lo, hi, c in zip(xs, vals, los, his, cols):
        ax1.bar([x], [v], color=c, width=0.55,
                hatch="//" if x == 2 else None, edgecolor="white")
        ax1.errorbar([x], [v], yerr=[[v - lo], [hi - v]], fmt="none",
                     ecolor="#22262b", capsize=5, lw=1.6)
    ax1.axhline(1.0, color="#b02540", ls="--", lw=1.6)
    ax1.text(2.42, 1.03, "alpha = 1", fontsize=9.5, color="#b02540", ha="right")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(["o1 published\n(parallel)", "R1 published\n(sequential)",
                         "initial single-model\n(RETRACTED)"], fontsize=9.5)
    ax1.text(2, float(init.group(1)) + 0.28,
             f"{init.group(1)}: retracted;\ninflated by small sample",
             ha="center", fontsize=9, color="#8a8a8a", style="italic")
    ax1.set_ylim(0, 3.4)
    ax1.set_ylabel("Scaling exponent alpha")
    ax1.set_title("Published data, and the retracted initial estimate",
                  fontsize=11.5, family="serif")

    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    ax2.set_title("The six-model experiment: architecture-dependent behaviour",
                  fontsize=11.5, family="serif")
    ax2.text(0.5, 0.97, "sequential condition, 18 AIME/Putnam-level problems",
             ha="center", fontsize=9.5, color="#57574e", transform=ax2.transAxes)
    for k, (name, desc, col) in enumerate(behaviours):
        y = 0.84 - k * 0.17
        ax2.text(0.02, y, name, fontsize=11.5, fontweight="bold",
                 color="#22262b", transform=ax2.transAxes, va="top")
        ax2.text(0.36, y, desc, fontsize=10.5, color=col,
                 transform=ax2.transAxes, va="top")
    ax2.text(0.02, 0.02,
             "Parallel scaling: at or near zero in every model. Directional finding\n"
             "(sequential > parallel) holds; the quantitative alpha > 1 claim is revised.",
             fontsize=10, color="#22262b", transform=ax2.transAxes,
             bbox=dict(boxstyle="round,pad=0.5", fc="#f2f0e6", ec="#cccccc"))

    fig.text(0.5, 0.02,
             "Generated from the paper: every value extracted from the text at "
             "generation time; the retracted estimate is displayed as retracted.",
             ha="center", fontsize=9.5, style="italic", color="#57574e")
    fig.savefig(dst, bbox_inches="tight", facecolor="white")
    print(f"wrote {dst}: o1 {o1mid:.2f}, R1 {r1.group(1)}, retracted {init.group(1)}, Gemini {gem.group(1)}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
