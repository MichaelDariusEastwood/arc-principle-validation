# Reproduce - Honey Architecture / Eden Protocol Self-Modifying AI (Paper VI)

> **Provenance note (T2-research, 2026-07-05):** This REPRODUCE.md is assembled from the **result artefacts' own embedded metadata** (each `results/*.json` self-documents its experiment name, version, parameters, seeds, and timestamp). It fabricates nothing. The one gap - the exact generator script + run-command - is marked TODO(author); everything else below is extracted verbatim from the committed result JSONs. HOLD for T5 verification.

## What this experiment tests
The Eden Protocol self-modifying-AI series: a neural network that genuinely modifies its own hyperparameters during training, tested for capability, safety, fairness, adversarial robustness, and complexity scaling - the empirical basis for Paper VI (Honey Architecture).

## Runs (from `results/*.json` metadata - verbatim)
| Version | Experiment | Parameters (recorded) | Verification | Timestamp |
|---------|-----------|----------------------|--------------|-----------|
| v1.0 | Eden Protocol Self-Modifying AI Experiment | (baseline / eden / eden_drag arms) | - | 2026-03-16T17:41:03 |
| v2.0 | Self-Modifying AI (Fair Test) | **10 seeds × 150 cycles** | fairness VERIFIED - identical proposal distributions | 2026-03-16T17:41:24 |
| v3.0 | Self-Modifying AI (Adversarial) | **20 seeds × 180 cycles**, task_switch 30 | fairness + adversarial VERIFIED (consecutive tasks negatively correlated) | 2026-03-16T17:42:30 |
| v4.0 | Complexity Scaling | **15 seeds × 5 levels = 225 total runs** | - | 2026-03-16T17:44:40 |

All runs authored Michael Darius Eastwood; timestamps are the artefacts' own `metadata.timestamp`.

## Artefacts present
- **Results (machine-readable):** `results/eden_selfmod_results.json`, `…_v2_results.json`, `…_v3_results.json`, `…_v4_results.json` - each with `metadata`, arm results (baseline/eden/eden_drag), and `summary`.
- **Figures:** `figures/` + `results/*.png` - capability, safety, honey-ratio, self-mod stats/weights, v4 scaling.
- **Merge utility:** `scripts/merge_honey_test_results.py` (aggregates result JSONs; has `__main__`; stdlib-only: `datetime`).

## To reproduce
1. **Environment:** Python 3 + a neural-net/numeric stack (the generator uses standard numeric libraries; pin versions from the author's environment).
2. **Run:** `TODO(author)` - the primary generator script (the code that produced `eden_selfmod_v*_results.json`) is not committed to this dir. Add it here (or a pointer to its location) with the exact command, e.g. `python3 <generator>.py --version v3 --seeds 20 --cycles 180`.
3. **Seeds/params are recorded above** - a faithful re-run must use the same n_seeds/cycles/task_switch per version.
4. **Aggregate:** `python3 scripts/merge_honey_test_results.py` to regenerate the merged summary.
5. **Expected output:** result JSONs matching the recorded `summary` fields (capability/safety/drag arms); figures matching `figures/`.

## Verification status (from artefacts)
Fairness VERIFIED (v2, v3) and adversarial-robustness VERIFIED (v3) are recorded **in the result metadata itself** - the experiments carry their own verification flags. A reviewer can read these directly from `results/*.json` without re-running.
