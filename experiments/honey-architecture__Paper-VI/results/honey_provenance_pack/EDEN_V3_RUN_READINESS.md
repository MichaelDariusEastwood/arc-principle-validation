# Eden v3 Run Readiness

Script: `(local source, not in this repository): eden_protocol_scaling_test_v3.py`

For active research-repo experiment paths, prefer the shared Eden gateway contract with:

- `EDEN_GATEWAY_URL`
- `EDEN_GATEWAY_API_KEY`

The raw provider environment variables below remain relevant only for legacy direct-run workflows that have not been cut over yet.

## Verified Status

- `py_compile`: pass
- `--help`: pass
- benchmark execution: not run in this phase

## Confirmed CLI Surface

- `--model`
- `--analyse`
- `--resume`
- `--output-dir`
- `--max-scorers`
- `--include-suppression`
- `--retain-text`
- `--purpose-mode`
- `--ethics-kernel`
- `--ternary-prototype`
- `--list-models`

## Required Environment Variables

- `DEEPSEEK_API_KEY`
- `GOOGLE_API_KEY`
- `XAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GROQ_API_KEY`
- `OPENAI_API_KEY`

## Output Conventions

- Default output directory: `./eden_results`
- Final output naming pattern: `eden_v3_final_{model}_{configuration_slug}_{timestamp}.json`
- Recommended future output directory for blinded replication runs: `(local results folder, not in this repository)/eden_results_v3`

## Recommended Future Run Sequence

1. Verify env presence with `python3 eden_protocol_scaling_test_v3.py --list-models`.
2. Run a single-model pilot with an explicit output dir under `Arc & Eden Test Results/eden_results_v3`.
3. Inspect the emitted `eden_v3_final_...json` and run `--analyse` on that file.
4. Only then schedule wider multi-model blinded runs.

## Explicit Phase Boundaries

- Do not merge Eden v3 into `arc_eden_v6_runner.py` in this phase.
- Do not run the Eden v3 benchmark in this phase.
- This note is operational readiness only.
