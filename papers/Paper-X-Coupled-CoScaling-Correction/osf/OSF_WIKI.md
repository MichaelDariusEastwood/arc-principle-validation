# The Coupled Co-Scaling Law

A falsifiable threshold criterion for the stability of recursive self-improvement:
**a misalignment-correcting force must out-scale drift-acceleration, i.e. `beta > k`.**

**Author:** Michael Darius Eastwood (independent; ARC/Eden research programme)
**Working paper dated:** 26 June 2026 &middot; **Deposited to OSF:** 29 June 2026
**Code (full commit history):** https://github.com/MichaelDariusEastwood/arc-principle-validation

## The claim, precisely
From a minimal dynamical model, stability under recursive self-improvement is set
not by the growth rate but by an inequality between two scaling exponents:
correction strength scales as capability^beta, drift acceleration as capability^k,
and the system is alignment-stable iff `beta > k`. A finite-time intelligence
explosion drives the modelled misalignment fraction to zero iff `beta > k`, and the
explosion's speed does not change that asymptotic verdict.

## What is original (priority)
1. the dimensionless control parameter `rho = gamma1 r / A`;
2. the sharpened stability criterion `beta > k` under accelerating self-improvement;
3. the Hard-Takeoff Depth-Regularity Theorem;
4. identification of the compounding drift channel as the locus of genuine divergence.

The underlying conceptual thesis was first set out in the book *Infinite Architects*
(copyright deposited 8 December 2024; published 2 January 2026); the formal,
measurable results here are the 2026 form of that thesis.

## What is NOT claimed (honesty)
- Not a proof of AI alignment.
- Not an empirically confirmed law: the decisive test, whether real self-improving
  systems satisfy the criterion, is the open problem. A pre-registered protocol and
  a runnable real-model harness are provided to make that test possible.
- The quantum-error-correction correspondence is offered as a falsifiable hypothesis
  (power-law suppression, not exponential).

## Standing
The proofs are independently re-derived in a standalone test suite, and the paper
has been hardened against three independent adversarial reviews.

## Contents of this project
- `Paper-X-Coupled-CoScaling-Correction.pdf` / `.html` — the paper
- `CLAIMS.md` — itemised claim ledger
- `code/` — verification harness + theorem test suites
- `experiments/` — real-model protocol, harness, and the zero-context launch runbook
- `results/` — verdict tables and run records
- `MANIFEST.json` — SHA-256 of every file (integrity)
- `NEGATIVE_RESULTS.md`, `SECURITY.md`, `REPRODUCIBILITY.md` — honesty and reproducibility

## How to cite
> Eastwood, M. D. (2026). *The Coupled Co-Scaling Law: A Falsifiable Threshold
> Criterion for the Stability of Recursive Self-Improvement.* ARC/Eden research
> programme. OSF. https://doi.org/10.17605/OSF.IO/6C5XB
