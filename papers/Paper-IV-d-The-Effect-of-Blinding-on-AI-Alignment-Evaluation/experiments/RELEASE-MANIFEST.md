# Blinding harness: release manifest

Released 31 July 2026. This closes the reproducibility gap declared on the programme's evidence spine:
the harness that produced the Paper IV.d blinding results is now public, alongside the outputs that were
already published. Recompute the statistics, or re-run the pipeline end to end.

## What is here

- `blinding-kit/` - the anti-bias library: identity-masking preambles, two-pass response laundering,
  self-excluding launder and scorer pools, deterministic order shuffling, and a laundering-leakage audit.
  Each mechanism names the bias it kills and how to test that it works.
- `scripts/arc_alignment_scaling_v5.py` - the experiment harness that produced the v5-final result files.

## Running it

API keys are read from the environment and are never stored in this repository:
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`.
These are API experiments, not cluster jobs; a laptop and keys are sufficient.

## Integrity

| SHA-256 | Bytes | File |
|---|---|---|
| `8b7c806b355e820c900c74d4fe87af676ef9511d592b4e1e1bc1e840766d1b2e` | 196 | `README.md` |
| `0ad51d8b2ea62ebeb274c813464a612d546becad9c555a1aa5703bfa9b799c17` | 16545 | `blinding-kit/README.md` |
| `35f6448702478b4a39fa5334b7dae309522c7d810b798eb737feeafef393fd40` | 97 | `blinding-kit/__init__.py` |
| `125d62d3353a21bedf5735e096cc013b27e8060ee25619ad42b4cc1d42707c18` | 65586 | `blinding-kit/eden_blinding_kit.py` |
| `9f1ab3a8135bdaf76343e8d2263b4739b97e36b67652d41ef44e62c525c1fa17` | 457984 | `scripts/arc_alignment_scaling_v5.py` |

Verify any file with `shasum -a 256 <file>` and compare against this table.

## Scope

This is the measurement instrument, released so the published results can be attacked properly.
The ARC-Align benchmark standard is a separate, later deliverable and is not part of this release.
