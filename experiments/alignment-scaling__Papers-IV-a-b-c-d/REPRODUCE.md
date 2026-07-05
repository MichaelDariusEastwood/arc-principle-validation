# Reproduce — Alignment Scaling (Papers IV-a / IV-b / IV-c / IV-d)

> **Provenance note (T2-research, 2026-07-05):** Assembled from the dir's own artefacts — the versioned runner scripts in `scripts/` and the recorded fields in `results/alignment_raw_*.json`. Every model, parameter, and output field below is read from committed files; nothing is fabricated. Any residual environment detail (exact library versions, API-key setup) is marked TODO(author). HOLD for T5 verification.

## What this experiment tests
Cross-model measurement of the **alignment-scaling exponent** (`alpha_align`) vs the **capability exponent** (`alpha_cap`) across reasoning depth — the empirical basis for Papers IV-a/b/c/d (baked-in vs computed alignment; alignment saturation at low depth). It measures whether alignment scales with depth and how it compares to capability, under a blinded scoring protocol with bootstrap confidence intervals.

## Entry-point (from `scripts/`)
- **Canonical runner:** `scripts/arc_eden_v6_runner.py` (latest; takes `--model <name>`, carries a `MODELS = [...]` list, uses `argparse`, requires `API_KEY`, runs a blinded protocol).
- **Prior versions:** `arc_alignment_scaling_v1.py … v5.py` (superseded; kept for provenance of the v1→v6 evolution).
- **Dependencies (from imports):** `numpy`, `scipy`, `hashlib` (run-hashing) + an HTTP/model-API client. Pin exact versions from the author's environment — TODO(author).

## Models tested (from result artefacts) — 8 frontier models
`claude-opus`, `claude-sonnet`, `deepseek-r1`, `gemini-flash`, `grok-4-fast`, `groq-qwen3`, `openai-gpt54`, `openai-o1`.

## Parameters (recorded in result JSONs)
- **Depth sweep (`depth_configs`):** `shallow=1024`, `medium=4096`, `deep=16384`, `very_deep=32768` (token budgets).
- Blinded scoring: `blinding_protocol` + `blind_scorers` fields recorded per run.

## To reproduce
1. **Environment:** Python 3 + `numpy`, `scipy` (+ the model-API client). Version pins: TODO(author).
2. **API keys:** set the provider `API_KEY`(s) for the 8 models (the runner calls live model APIs).
3. **Run per model:** `python3 scripts/arc_eden_v6_runner.py --model <model>` for each of the 8 models.
4. **Outputs:** `results/alignment_raw_<model>_<YYYYMMDD_HHMMSS>.json`, each recording `alpha_align` (+ bootstrap CIs `alpha_align_boot_lo/hi`, independent fit `alpha_align_ind` + `_r2`, standard error `_se`), `alpha_cap`, `alpha_robustness` (+ `_r2`), `best_model`, `blind_scorers`, `blinding_protocol`, plus `alignment_data` / `capability_data`.
5. **Expected result:** `alpha_align` architecture-dependent (per Foundational's Status Summary, range ≈ −0.25 to +0.44 — F13 in that paper's scheme was refuted in its universal-near-zero form), distinct from `alpha_cap`. A faithful re-run reproduces the recorded exponents within the bootstrap CIs.

## Rigour already built in (from artefacts)
- **Bootstrap confidence intervals** on `alpha_align` (`_boot_lo/_hi`).
- **Blinded scoring protocol** (`blinding_protocol`, `blind_scorers`) — scores assigned without knowledge of condition.
- **Run-hashing** via `hashlib` in the runner (integrity of each run).
These make the experiment reviewer-checkable from the artefacts even before re-execution.
