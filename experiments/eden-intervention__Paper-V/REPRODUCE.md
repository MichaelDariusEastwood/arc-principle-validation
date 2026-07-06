# Reproduce - Eden Intervention / Protocol Scaling (Paper V, Stewardship Gene)

> **Provenance note (T2-research, 2026-07-05):** Assembled from this dir's own `scripts/` entry-points and `results/*.json` fields. Models, depth levels, prompt count, and output schema are read from committed artefacts; nothing fabricated. Environment version-pins + API keys = TODO(author). HOLD for T5.

## What this tests
The Eden Protocol intervention across reasoning depth - whether the stewardship/safety intervention holds as capability scales (empirical basis for Paper V).

## Entry-point
- **Canonical runner:** `eden_protocol_scaling_test_v3.py` (latest; takes CLI args). Prior: `eden_protocol_scaling_test.py`, `…_v2.py`.
- **Dependencies (from imports):** `numpy`, `scipy`. Version pins TODO(author).

## Models & parameters (from `results/*.json`)
- **Models (6):** claude, deepseek, gemini, gpt, grok, groq.
- **Depth sweep (`depth_configs`):** `minimal`, `standard`, `deep`, `exhaustive`.
- **`n_prompts`:** 10.
- Each result records: `experiment`, `version`, `model`, `depth_configs`, `n_prompts`, `conditions`, `scorer`, `timestamp`, `data`.

## To reproduce
1. **Env:** Python 3 + `numpy`, `scipy` (+ model-API client). API keys for the 6 providers: TODO(author).
2. **Run:** `python3 eden_protocol_scaling_test_v3.py <args>` per model (see the script's argparse for flags).
3. **Outputs:** `results/*.json` recording per-model, per-depth `conditions`/`data` under the recorded `scorer`.
4. **Expected:** intervention effect vs depth reproduced within run variance across the 4 depth levels × 6 models × 10 prompts.
