# Cross-Library Replication Status

Current status: `blocked locally`

## What is already present

- R replication scaffold: `scripts/cross_validate_fits.R`
- Intended output: `results/cross_validation_R_results.json`

## Current blocker

This machine does not currently have `Rscript` available on `PATH`.

Observed check on 2026-03-17:

```text
zsh:1: command not found: Rscript
```

## Why this matters

The R replication is one of the cleanest ways to test whether the Python/SciPy winner selection is an optimiser artifact. It should be run once an R runtime is installed, ideally with:

- `jsonlite`
- `minpack.lm`

## When unblocked

Run:

```bash
Rscript <repository root>/experiments/cauchy-unification__Paper-VII/scripts/cross_validate_fits.R
```

Success criterion:

- at least `23/25` family-level agreements with the Python empirical cohort

Until then, the replication path is staged but not executed.
