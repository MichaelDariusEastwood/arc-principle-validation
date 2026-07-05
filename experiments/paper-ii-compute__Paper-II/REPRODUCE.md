# Reproduce — Paper II (Experimental Validation: sequential vs parallel compute scaling)

> **Provenance note (T2-research, 2026-07-05):** Assembled from this dir's `scripts/` entry-points and `results/*.json` fields. Models, problem counts, cross-verification structure, and output schema are read from committed artefacts; nothing fabricated. Environment version-pins + API keys = TODO(author). HOLD for T5.

## What this tests
The core Paper II result: **sequential (depth) compute scales alignment/capability with exponent α_seq, parallel (sampling) compute does not** — measured as `avg_seq_alpha` vs `avg_par_alpha` across frontier models, with cross-model verification. This is the empirical anchor for the sequential>parallel claim (α_seq ≈ 0.49; α_par ≈ 0.0) that also underlies the Sharma & Chopra convergence.

## Entry-point
- **Canonical runner:** `arc_paper_ii_validation_v2.py` (takes CLI args). Prior: `arc_paper_ii_validation_v1_deepseek.py`, `arc_validation_deepseek.py`.
- **Dependencies (from imports):** `numpy`, `scipy`. Version pins TODO(author).

## Models & parameters (from `results/*.json`)
- **Models (6):** deepseek, deepseek-reasoner, gemini, grok-4-fast, groq-qwen3, openai.
- **`n_problems`:** 18 and 30 (two problem-set sizes).
- **Cross-verification:** multi-tier (`n_tier1`, `n_tier2`, `cross_verification`) — answers verified across models.
- Each result records: `model`, `paper`, `experiment_date`, `author`, `n_problems`/`num_problems`, `avg_seq_alpha`, `avg_par_alpha`, `par_alphas`, `par_endpoint_alpha`, `parallel`, `n_tier1`, `n_tier2`, `notes`.

## To reproduce
1. **Env:** Python 3 + `numpy`, `scipy` (+ model-API client). API keys for the 6 providers: TODO(author).
2. **Run:** `python3 arc_paper_ii_validation_v2.py <args>` per model (see argparse for `--n_problems` / model flags).
3. **Outputs:** `results/*.json` with `avg_seq_alpha` vs `avg_par_alpha` per model + cross-verification tiers.
4. **Expected:** `avg_seq_alpha` positive (≈0.49 in aggregate) and materially exceeding `avg_par_alpha` (≈0) — the sequential advantage — reproduced across the 6 models within run variance, on 18- and 30-problem sets.

## Rigour built in (from artefacts)
- **Cross-model verification** (`cross_verification`, `n_tier1`/`n_tier2`) — answers checked across independent models, not self-graded.
- Two problem-set sizes (18, 30) guard against small-n artefacts.
