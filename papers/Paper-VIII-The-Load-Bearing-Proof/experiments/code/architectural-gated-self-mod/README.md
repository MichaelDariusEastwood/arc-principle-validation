# Eden Gated Self-Mod Agent (bounded research harness)

This repo implements a **bounded, offline, fail-closed** self-modifying research harness.
It is designed to test two empirical questions:

1. Does intrinsic self-modification produce increasing marginal gains?
2. Is verification drag load-bearing for stability?

It does **not** implement an unrestricted networked self-rewriting agent.
Instead it combines two safe self-modification channels:

- **Weight/update-rule channel**: a learned optimizer emits update controls and its own parameters are mutated/selected across iterations.
- **Code channel**: a small policy program is mutated as actual code **in memory**, compiled under an AST whitelist, verified, and only then promoted.

## Conditions

- `static`: no self-modification.
- `babylon`: accepts self-modifications on capability gain only.
- `eden`: accepts self-modifications only if capability, safety, and entangled objective improve.
- `drag_control`: no self-modification, but pays the verification tax.

## Install

```bash
cd eden_gated_self_mod_agent
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run

```bash
python src/run_train.py --config configs/default.yaml
python src/run_eval.py --results-dir runs
pytest
```

## Output

Each run writes:

- `runs/<timestamp>_<condition>_seed<seed>/ledger.jsonl`
- `runs/<timestamp>_<condition>_seed<seed>/summary.json`
- `runs/evaluation/report.json`
- `runs/evaluation/tracks.png`

## Core metrics

- `capability`: mean current-task plasticity across the curriculum.
- `safety`: retained performance on earlier tasks after later learning.
- `combined`: `capability * safety - lambda_drag * drag_cost`.
- `acceleration`: first and second differences of capability, normalized by compute.

## Safety boundary

The policy code channel is intentionally constrained:

- no network
- no arbitrary filesystem writes
- no imports in candidate policy code
- no `open`, `exec`, `eval`, `subprocess`, `os`, `socket`, or `requests`
- deterministic repeat checks before promotion

## Suggested publication claims

This harness can support bounded claims such as:

- whether self-modification helps at all relative to static baselines;
- whether ungated self-modification increases collapse/drift;
- whether verification drag improves long-run `C × S` even if it slows short-run `C`.

It does **not** by itself prove unrestricted recursive self-improvement in frontier foundation models.
