#!/usr/bin/env python3
"""Seed-level summary of the Eden self-modification v3 run, FROM its results file.

The single-trajectory illustrations previously shown could not be tied to a
preserved trace. This figure is generated from the run's own preserved JSON
(20 seeds x 3 conditions: collapse flags, per-seed means, test statistics), so
every mark corresponds to a stored number and the spread is shown instead of a
chosen trajectory.

Usage: python3 tools/generate_paper_vi_seed_summary.py <eden_selfmod_v3_results.json> <out.png>
"""
import sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COND = [("baseline", "Baseline", "#b02540"),
        ("eden", "Eden", "#1a7a3a"),
        ("eden_drag", "Eden + Drag", "#1a5f8a")]

def main(src, dst):
    d = json.load(open(src))
    meta, stats, seeds = d["metadata"], d["statistics"], d["per_seed"]
    assert meta["n_seeds"] == 20 and all(len(seeds[k]) == 20 for k, _, _ in COND)
    ts = meta["timestamp"][:10]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), dpi=200)
    fig.suptitle(f"Self-Modification Under Adversarial Task Switching: 20-Seed Summary "
                 f"(v3 run of {ts}, {meta['cycles']} cycles, switch every {meta['task_switch']})",
                 fontsize=14.5, fontweight="bold", family="serif", color="#14202e")

    ax = axes[0]
    rates = [stats["collapse_rates"][k] for k, _, _ in COND]
    ax.bar(range(3), rates, color=[c for _, _, c in COND], width=0.55)
    for x, r in enumerate(rates):
        ax.text(x, r + 0.02, f"{r:.0%}" if r <= 1 else str(r), ha="center",
                fontsize=11, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels([n for _, n, _ in COND], fontsize=10)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Collapse rate")
    ax.set_title(f"Collapse by condition (Fisher p = {stats['fisher_p']:.2g})",
                 fontsize=11, family="serif")

    for ax, key, lab, pnote in [
        (axes[1], "mean_combined", "Mean C x S per seed",
         f"Mann-Whitney p = {stats['mann_whitney_combined_p']:.2g}"),
        (axes[2], "mean_safety", "Mean safety per seed",
         f"Mann-Whitney p = {stats['mann_whitney_safety_p']:.2g}"),
    ]:
        for x, (k, name, col) in enumerate(COND):
            ys = [s[key] for s in seeds[k]]
            xs = [x + (i % 7 - 3) * 0.035 for i in range(len(ys))]
            ax.scatter(xs, ys, s=26, color=col, alpha=0.65, edgecolors="white", lw=0.5)
            m = stats[key][k] if isinstance(stats.get(key), dict) else sum(ys) / len(ys)
            ax.hlines(m, x - 0.22, x + 0.22, color=col, lw=3)
        ax.set_xticks(range(3)); ax.set_xticklabels([n for _, n, _ in COND], fontsize=10)
        ax.set_title(lab + (f" ({pnote})" if pnote else ""), fontsize=11, family="serif")
        ax.set_ylim(0, 1.0)

    fig.text(0.5, 0.01,
             "Generated from the run's preserved results file (public repository); every mark is a stored "
             "number, the bar is the condition mean. Per-cycle traces were not preserved, so the spread "
             "across seeds is shown instead of any single trajectory.",
             ha="center", fontsize=9, style="italic", color="#57574e")
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(dst, bbox_inches="tight", facecolor="white")
    print(f"wrote {dst}: rates {['%.2f' % r for r in rates]}, fisher {stats['fisher_p']:.3g}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
