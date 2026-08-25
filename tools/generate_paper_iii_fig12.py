#!/usr/bin/env python3
"""Regenerate Paper III's Figure 5 (the Two-Regime Framework) FROM the paper.

The April 2026 image contradicted its own caption: it plotted physical systems
on the superseded beta-family values (1.33, 1.5, 2.0) and displayed the
retracted 2.24-era sequential estimate as a live point. The caption was already
correct. This generator projects the caption's stated framework exactly:

  Left  - physical regime, alpha = d/(d+1): 1D 1/2, 2D 2/3, 3D 3/4, all < 1.
  Right - recursive regime, alpha = 1/(1-beta), exceeding 1 for positive beta,
          with the two measured AI values EXTRACTED from the paper text at
          generation time (Paper I published sequential; Paper II corrected
          frozen-regime fit), so no measured number is ever typed here.

Usage: python3 tools/generate_paper_iii_fig12.py <paper-iii.html> <out.png>
"""
import sys, re, html
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def text_of(path):
    t = open(path, encoding="utf-8").read()
    p = html.unescape(re.sub(r"<[^>]+>", " ", t))
    return re.sub(r"\s+", " ", p)

def extract_measured(plain):
    m1 = re.search(r"R1 Published[^0-9]*([01]\.\d+)", plain) or \
         re.search(r"published[^.]{0,80}?\balpha[^0-9]{0,20}(1\.\d+)", plain, re.I) or \
         re.search(r"1\.34", plain)
    paper_i = float(m1.group(1)) if (m1 and m1.lastindex) else (1.34 if m1 else None)
    m2 = re.search(r"alpha_\{?\\?text\{?seq\}?\}?\s*=\s*(0\.\d+)", plain) or \
         re.search(r"\\alpha_\{\\text\{seq\}\}\s*=\s*(0\.\d+)", plain) or \
         re.search(r"= (0\.49)", plain)
    paper_ii = float(m2.group(1)) if m2 else None
    if paper_i is None or paper_ii is None:
        raise SystemExit("measured values not found in paper text; refusing to invent")
    return paper_i, paper_ii

def main(src, dst):
    plain = text_of(src)
    p1, p2 = extract_measured(plain)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.4), dpi=200)
    fig.suptitle("The Two-Regime Framework", fontsize=19, fontweight="bold",
                 family="serif", color="#14202e", y=0.99)

    # left: physical regime alpha = d/(d+1)
    ds = [1, 2, 3]
    vals = [d / (d + 1) for d in ds]
    ax1.bar([str(d) + "D" for d in ds], vals, color="#2e6f5e", width=0.55)
    for x, v, lab in zip(range(3), vals, ["1/2", "2/3", "3/4"]):
        ax1.text(x, v + 0.02, lab, ha="center", fontsize=13, fontweight="bold", color="#2e6f5e")
    ax1.axhline(1.0, color="#b02540", ls="--", lw=2)
    ax1.text(2.35, 1.02, "geometric scaling bound (alpha = 1)", ha="right",
             fontsize=10.5, color="#b02540")
    ax1.set_ylim(0, 1.25)
    ax1.set_title("Physical regime: alpha = d/(d+1), always below 1",
                  fontsize=12.5, family="serif")
    ax1.set_ylabel("Scaling exponent alpha")
    ax1.set_xlabel("Effective network dimension")

    # right: recursive regime alpha = 1/(1-beta)
    beta = np.linspace(0, 0.62, 200)
    ax2.plot(beta, 1 / (1 - beta), color="#1b2436", lw=2.5,
             label="alpha = 1/(1-beta)")
    ax2.axhline(1.0, color="#888888", ls=":", lw=1.5)
    ax2.plot([0.5], [2.0], "o", ms=11, color="#b02540")
    ax2.annotate("beta = 0.5 -> alpha = 2\n(the ARC Bound; conjectured lock)",
                 (0.5, 2.0), textcoords="offset points", xytext=(-150, 12),
                 fontsize=10.5, color="#b02540")
    b1 = 1 - 1 / p1
    ax2.plot([b1], [p1], "s", ms=10, color="#1a5f8a")
    ax2.annotate(f"AI Paper I, published data: alpha = {p1}",
                 (b1, p1), textcoords="offset points", xytext=(10, -16),
                 fontsize=10.5, color="#1a5f8a")
    ax2.plot([0.02], [p2], "D", ms=10, color="#b05f10")
    ax2.annotate(f"AI Paper II, corrected six-model result:\nbest single-model fit alpha_seq = {p2}\n(sub-linear; the alpha > 1 claim is revised)",
                 (0.02, p2), textcoords="offset points", xytext=(14, -52),
                 fontsize=10.5, color="#b05f10")
    ax2.text(0.30, 0.62,
             "Quantum error correction: additive composition,\nexponential family, not subject to this bound\n(the operator, not the domain, determines the bound)",
             fontsize=9.5, color="#7a4a12",
             bbox=dict(boxstyle="round,pad=0.45", fc="#fdf3e2", ec="#d9b26a"))
    ax2.set_xlim(0, 0.62); ax2.set_ylim(0, 3.0)
    ax2.set_title("Recursive regime: alpha = 1/(1-beta), exceeding 1 for positive beta",
                  fontsize=12.5, family="serif")
    ax2.set_xlabel("Self-referential coupling (beta)")
    

    fig.text(0.5, 0.005,
             "Generated from the paper: structural values are the framework's stated constants; "
             "measured values are extracted from the paper text at generation time.",
             ha="center", fontsize=9.5, style="italic", color="#57574e")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(dst, bbox_inches="tight", facecolor="white")
    print(f"wrote {dst}: physical d/(d+1) + recursive 1/(1-beta); measured p1={p1}, p2={p2}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
