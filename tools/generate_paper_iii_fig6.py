#!/usr/bin/env python3
"""Regenerate Paper III's falsification-criteria figure FROM the paper's own table.

The moving-number lesson (25 August 2026): a status baked into artwork always
drifts, because the artwork cannot know the status moved. This generator makes
the figure a projection of the paper's falsification table, which is the single
authoring. Statuses are never typed here; they are read from the table at
generation time, and the chip shows a compressed label while the table carries
the full note.

Usage: python3 tools/generate_paper_iii_fig6.py <paper-iii.html> <out.png>
"""
import sys, re, html
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def extract_table(path):
    t = open(path, encoding="utf-8").read()
    i = t.find(">F1<")
    a = t.rfind("<table", 0, i); b = t.find("</table>", i)
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", t[a:b], flags=re.S)
    out = []
    for r in rows:
        cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, flags=re.S)]
        if cells and cells[0].startswith("F"):
            out.append(cells)
    return out

def strip_tex(s):
    s = s.replace("$", "")
    s = re.sub(r"\\alpha(_\{?\\?text\{?(\w+)\}?\}?)?", lambda m: "alpha" + ("_" + m.group(2) if m.group(2) else ""), s)
    s = s.replace("\\leq", "<=").replace("\\geq", ">=").replace("\\beta", "beta")
    s = re.sub(r"\\(text|mathrm)\{([^}]*)\}", r"\2", s)
    s = re.sub(r"[{}\\]", "", s)
    return re.sub(r"\s+", " ", s).strip()

def chip(status):
    s = status.strip()
    first = re.split(r"[.:]", s, 1)[0].strip()
    low = first.lower()
    # a leading data descriptor (no status keyword) must not become the chip:
    # scan the whole note for the verdict phrase instead
    KEYS = ("confirmed", "unconfirmed", "untested", "open", "mixed", "partial",
            "directional", "tested", "supported", "retracted", "median")
    if not any(k in low for k in KEYS):
        m = re.search(r"median[^.,;]*", s, flags=re.I)
        if m:
            first = "Tested: " + m.group(0).strip()
        else:
            for k in KEYS:
                mm = re.search(k + r"[^.,;]*", s, flags=re.I)
                if mm:
                    first = mm.group(0).strip()
                    break
        low = first.lower()
    if low.startswith("confirmed"):
        col, bg = "#1a7a3a", "#e7f3ea"
    elif "unconfirmed" in low or "partial" in low or "mixed" in low or "directional" in low:
        col, bg = "#b05f10", "#fdf3e2"
    elif low.startswith("untested"):
        col, bg = "#555555", "#efefef"
    elif low.startswith("open") or "conjecture" in low:
        col, bg = "#b02540", "#fbe8ee"
    elif "tested" in low or "measured" in low:
        col, bg = "#1a5f8a", "#e7f0f7"
    else:
        col, bg = "#555555", "#efefef"
    label = first if len(first) <= 34 else first[:31] + "..."
    return label, col, bg

def main(src, dst):
    rows = extract_table(src)
    n = len(rows)
    fig, ax = plt.subplots(figsize=(14, 1.05 * n + 2.6), dpi=200)
    ax.set_xlim(0, 14); ax.set_ylim(0, n + 2.2); ax.axis("off")
    ax.text(7, n + 1.7, f"{'Thirteen' if n == 13 else n} Falsification Criteria",
            ha="center", fontsize=22, fontweight="bold", family="serif", color="#14202e")
    ax.text(7, n + 1.1,
            "Generated from the paper's falsification table, which is the single authoring; "
            "chips compress the status, the table carries the full note.",
            ha="center", fontsize=10.5, color="#57574e", style="italic")
    for k, cells in enumerate(rows):
        y = n - k
        fid, hyp = cells[0], strip_tex(cells[1])
        label, col, bg = chip(strip_tex(cells[4]))
        ax.add_patch(FancyBboxPatch((0.4, y - 0.32), 1.1, 0.64,
                     boxstyle="round,pad=0.02", fc="#1b2436", ec="none"))
        ax.text(0.95, y, fid, ha="center", va="center", fontsize=12,
                fontweight="bold", color="white")
        ax.text(1.9, y, hyp if len(hyp) <= 72 else hyp[:69] + "...",
                ha="left", va="center", fontsize=12.5, color="#22262b", family="serif")
        ax.add_patch(FancyBboxPatch((9.9, y - 0.3), 3.6, 0.6,
                     boxstyle="round,pad=0.02", fc=bg, ec="#cccccc", lw=0.8))
        ax.text(11.7, y, label, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=col)
    ax.text(7, 0.15, "We welcome falsification. Either outcome advances science.",
            ha="center", fontsize=12, color="#1a6b5a", style="italic", family="serif")
    fig.savefig(dst, bbox_inches="tight", facecolor="white")
    print(f"wrote {dst}: {n} criteria projected from the table")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
