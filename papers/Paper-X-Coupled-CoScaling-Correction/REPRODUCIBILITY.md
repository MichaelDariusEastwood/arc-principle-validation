# Reproducibility - Paper X

Everything in this paper except the live-model experiment is deterministic and reproducible
offline in under a minute. This file says exactly which command reproduces which claim, and
- as honestly - which claim **cannot** yet be reproduced because the experiment has not been run.

## Environment

- Python 3.11 (tested 3.11.15). Dependencies and tested versions: `requirements.txt`.
- `pip install -r requirements.txt`  (numpy, scipy, matplotlib; pytest for tests).
- Or use the container: `docker build -t paperx . && docker run --rm paperx`.

## One command

```bash
make all        # install + verify (harness) + test-all (both suites)
```

## What reproduces what

| Claim in the paper | Command | Artefact produced | What it establishes |
|---|---|---|---|
| Theorems 1-6 numerically consistent with their closed forms (E1-E9) | `make verify` (`code/experiment_coscaling.py`) | `results/verdicts.json`, `results/report.txt`, `figures/*.png` | code matches the maths (10/10 internal-consistency + integrator checks) |
| Regression: harness output is stable | `make test` (`code/test_coscaling.py`) | - | 12 assertions on the harness |
| **Independent** re-derivation of the corrected theorems | `make test-independent` (`code/test_theorems_independent.py`) | - | 14 from-scratch checks (no harness import): Thm 2 time-varying limits, Thm 4 equality = linear divergence, Thm 5 non-normal transient + null-axis floor, Thm 6 OU variance, γ₃=1 boundary |
| β/k estimator recovers known exponents | `make estimator` | stdout | synthetic identifiability only |
| Real-model harness plumbing | `make selftest` | `results/realmodel/*_selftest.json` | deterministic stub, **not data** |

The two test suites are deliberately independent: `test_coscaling.py` shares the harness
integrator (regression), while `test_theorems_independent.py` re-implements the dynamics
from scratch with `scipy.solve_ivp` + matrix exponentials, so a shared bug cannot hide.

## Determinism

Random seed is fixed (`7`) in the harness; figures and verdicts are byte-reproducible on a
fixed library stack. `code/experiment_coscaling.py` writes to the current directory by
default (`COSCALING_OUTDIR=.`); run it from the paper root to regenerate the canonical
`results/` and `figures/`.

## What is NOT reproducible here (stated plainly)

- The **`β > k` threshold on a real system.** Not run. The decisive experiment is
  pre-registered in `experiments/PROTOCOL.md`; reproducing it needs live API keys for
  several model families and a network-isolated sandbox (`SECURITY.md`).
- The **first Claude run** is a pilot: `n = 1`, one task, same-family scorer,
  IV.d non-compliant, H1/H2 false. It corroborates the corrector *mechanism* only and is
  **not** evidence for the threshold (see `NEGATIVE_RESULTS.md`).

## Continuous integration (optional)

CI is not wired up as a live GitHub Actions workflow here (the repository's Actions
policy must first allow the GitHub-authored `actions/checkout` and `actions/setup-python`).
The exact verification commands are the `Makefile` targets, so a one-job workflow is just:

```yaml
# .github/workflows/paperx-ci.yml  (enable once Actions allows GitHub-authored actions)
on: [push, pull_request]
jobs:
  verify:
    runs-on: ubuntu-latest
    defaults: { run: { working-directory: papers/Paper-X-Coupled-CoScaling-Correction } }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - run: cd code && python experiment_coscaling.py && python -m pytest -q
      - run: cd experiments/scripts && python estimate_exponents.py --selftest
```

Until then, `make all` (or the `Dockerfile`) is the canonical, environment-independent
verification path.

## Honesty ladder

The status of every load-bearing claim - proved-in-model / internally-verified /
synthetically-validated / real-model-pilot / open - is in `CLAIMS.md`. Nothing in this
repository should be read one rung higher than it is placed there.
