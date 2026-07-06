# Reproduce - Cross-Programme Analysis Tools

> **Provenance note (T2-research, 2026-07-05):** Assembled from this dir's committed scripts. This dir holds **post-processing / analysis tools** that consume the OTHER experiment dirs' result JSONs (it has no primary result artefacts of its own). HOLD for T5.

## What this is
Cross-programme analysis utilities that aggregate and re-analyse the raw results produced by the primary experiment dirs (chiefly `alignment-scaling__Papers-IV-a-b-c-d`).

## Entry-points
- `analyze_alpha_align_v5.py` - re-analyses the alignment-scaling `alpha_align` results (uses `glob` to sweep the raw result JSONs, then `numpy`/`scipy` for the fit). Takes CLI args.
- `per_scorer_check.py` - per-scorer consistency check (blinded-scoring integrity).
- **Dependencies:** `numpy`, `scipy`, `glob`. Version pins TODO(author).

## To reproduce
1. **Env:** Python 3 + `numpy`, `scipy`.
2. **Prerequisite:** the primary experiment results must exist (run `alignment-scaling` first - see its REPRODUCE.md).
3. **Run:** `python3 analyze_alpha_align_v5.py <path-to-results>` to re-aggregate `alpha_align` across models; `python3 per_scorer_check.py` for scorer consistency.
4. **Outputs:** aggregated exponent tables / scorer-consistency reports (per the scripts' print/save paths).
5. **Note:** This dir does not itself generate primary data - it verifies and aggregates the primary dirs' outputs. It is the "checks the checkers" layer.
