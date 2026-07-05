# Reproduce — Paper I (ARC Principle, Foundational Toolkit)

> **Provenance note (T2-research, 2026-07-05):** Assembled from this dir's committed script. This dir is a **figure/analysis toolkit** (no result-JSON artefacts — it generates the paper's figures from the framework equations), so there is no per-run parameter file to extract; the toolkit itself is the reproducible unit. HOLD for T5.

## What this is
`arc_principle_research_toolkit.py` — the foundational toolkit that generates Paper I's figures and illustrative analyses of the ARC Principle (U = I × R^α scaling family, geometric speed limit, cross-domain forms).

## Entry-point
- **Script:** `arc_principle_research_toolkit.py` (has `__main__`).
- **Dependencies (from imports):** `matplotlib`, `numpy`, `scipy`. Version pins TODO(author).

## To reproduce
1. **Env:** Python 3 + `matplotlib`, `numpy`, `scipy`.
2. **Run:** `python3 arc_principle_research_toolkit.py` — regenerates the figures/analyses.
3. **Outputs:** figures written to the dir (per the script's save paths — see `savefig` calls).
4. **Note:** Paper I's empirical claims are validated in the sibling experiment dirs (alignment-scaling, paper-ii-compute); this toolkit is the illustrative/figure layer. For the measured exponents, see those dirs' REPRODUCE.md.
