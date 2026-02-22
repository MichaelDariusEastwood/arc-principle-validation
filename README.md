# The ARC Principle

**A mathematical framework for measuring alignment scaling, validated across four independent physical domains**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the complete document suite, experimental code, raw data, figures, and research toolkits for the ARC Principle framework and the Eden Protocol alignment architecture, first proposed in *Infinite Architects: Intelligence, Recursion, and the Creation of Everything* (Eastwood, 2026).

**Author:** Michael Darius Eastwood
**London, United Kingdom**

## The Document Suite

All authoritative documents are in `paper/FINAL-SUITE/`:

| # | Document | Version | Description |
|---|----------|---------|-------------|
| 01 | **White Paper III: The Alignment Scaling Problem** | v9.0 | Primary document. Alignment scaling exponent, ARC Bound, cross-domain validation, 13 falsification criteria. |
| 02 | **Foundational Paper: The ARC Principle** | v2.1 | Condensed academic treatment. Axiomatic derivation, computational validation, five universal properties. |
| 03 | **Eden Protocol: Engineering Specification** | v4.0 | Alignment architecture. Three Ethical Loops, Six Questions, Ternary Logic, Monitoring Removal Test, Caretaker Doping. |
| 04 | **Eden Protocol: Philosophical Vision** | v1.0 | Companion piece. The Grande Purpose, the Eternal Architect, the Cosmic Fork, the Infinite Covenant. |
| 05 | **Executive Summary** | v1.0 | One-page overview for grant reviewers (£150k/£500k/£1.1M). |

See `CANONICAL-VERSIONS.md` for version history. Do not use older versions in `drafts-not-published/` — they contain mathematical errors corrected during February 2026 review.

## The ARC Principle

```
U = I × R^α     where α = 1/(1−β)
```

- **U** = Effective capability
- **I** = Base potential (structured asymmetry)
- **R** = Recursive depth
- **β** = Self-referential coupling (how much each step builds on the previous)
- **α** = Scaling exponent (derived, not fitted)

**Core prediction:** α_sequential > 1 > α_parallel

**Computational validation:** α = 1/(1−β) validated to R² = 1.00000000 (machine precision)

**The ARC Bound:** U_max = I × R² (an information-theoretic upper bound on classical sequential computation, analogous to the Shannon Limit)

## The Alignment Argument

AI capability scales through recursive self-correction. External alignment constraints (RLHF, constitutional rules, output filters) do not participate in the recursive loop and therefore cannot compound. We define the *alignment scaling exponent* α_align as a measurable quantity and predict:

- **External constraints:** α_align ≈ 0 (safety degrades with depth)
- **Embedded values (Eden Protocol):** α_align ≈ α_cap (safety scales with capability)

If α_cap > α_align, the safety ratio S ∝ R^(α_align − α_cap) → 0 as R → ∞. **This prediction has not been tested.** We provide the measurement protocol.

## Cross-Domain Evidence

| Domain | System | Finding |
|--------|--------|---------|
| AI | DeepSeek R1 | Sequential α = 1.3–2.2; parallel α ≈ 0 |
| Quantum | Google Willow | Exponential error suppression (Λ = 2.14) |
| Physics | NYU Time Crystals | Frozen disorder + feedback → temporal order |
| Neuroscience | COGITATE | Recurrent processing required for consciousness |

## Key Experimental Results

| Source | Recursion Type | α | Method |
|--------|----------------|---|--------|
| Paper I | Parallel (o1) | 0.1–0.3 | Published data analysis |
| Paper I | Sequential (R1) | ~1.34 | Published data analysis |
| Paper II | Sequential | **2.2** | Direct experiment |
| Paper II | Parallel | **0.0** | Direct experiment |
| Validation Suite | Bernoulli ODE | **R²=1.0** | Computational |

**Core result:** Sequential recursion with 412 tokens outperformed parallel recursion with 1,101 tokens by 25 percentage points.

**The form of recursion matters more than the amount of compute.**

## Repository Structure

```
arc-principle-validation/
├── README.md                           # You are here
├── CANONICAL-VERSIONS.md               # Current authoritative versions
├── LICENCE                             # MIT Licence
│
├── paper/FINAL-SUITE/                  # AUTHORITATIVE DOCUMENT SUITE
│   ├── README.md                       # Reading order and cross-references
│   ├── Paper-III-White-Paper-v9.0.html        # Primary document (+ PDF)
│   ├── Foundational-v2.1.html       # Foundational paper (+ PDF)
│   ├── Eden-Engineering-v4.0.html   # Eden Engineering (+ PDF)
│   ├── Eden-Vision-v1.0.html        # Eden Vision (+ PDF)
│   ├── Executive-Summary-v1.0.html  # Grant summary (+ PDF)
│   ├── figures/                        # 12 figures for White Paper III
│   └── PAPER-I-REFERENCE/             # Paper I (published PDF)
│
├── research-toolkits/                  # REPLICATION TOOLKITS
│   ├── paper-i/                        # Paper I: public data analysis
│   │   ├── code/                       # Analysis scripts
│   │   ├── results/                    # Pre-computed results (JSON)
│   │   └── figures/                    # 3 visualisations
│   └── paper-ii/                       # Paper II: DeepSeek R1 experiments
│       ├── code/                       # Experiment scripts
│       ├── results/                    # Raw experimental data (JSON)
│       └── figures/                    # 15 visualisations
│
├── validation/                         # COMPUTATIONAL VALIDATION
│   ├── prove_IxR_equals_complexity_v2.py
│   ├── arc_definitive_test.py
│   └── arc_unified_paradigm_test.py
│
└── blind-test/                         # BLIND PREDICTION TEST
    ├── BLIND_PREDICTION_TEST.py
    ├── BLIND_TEST_FORENSIC_ANALYSIS.md # Confound identification
    └── BLIND_TEST_FORENSIC_ANALYSIS.png
```

## Blind Prediction Test

A blind prediction test was conducted to validate α = 1/(1−β):

| System | β measured | α predicted | α measured | Result |
|--------|------------|-------------|------------|--------|
| BA Network | 0.70 | 3.33 | 0.34 | FAIL |
| Gradient Descent | 0.95 | 20.0 | 0.87 | FAIL |
| Kuramoto | 0.55 | 2.24 | 0.55 | PASS* |

**Forensic analysis** identified two independent confounds:
1. **Measurement bias:** Numerical-derivative β estimation gives β ≈ 0.95 regardless of true β
2. **Axiom violation:** None of the tested systems satisfy the framework's three axioms

When proper linearisation is applied to axiom-satisfying systems, R² = 0.9999. See `blind-test/BLIND_TEST_FORENSIC_ANALYSIS.md`.

## Falsification

Thirteen explicit falsification criteria are specified across the ARC framework and Eden Protocol, each independently sufficient to refute the relevant claims. See White Paper III Section 4 and Eden Protocol Section 11.

## Priority Timeline

| Date | Event |
|------|-------|
| 8 Dec 2024 | Original manuscript (DKIM-verified) |
| 9 Dec 2024 | Google Willow announced |
| 20 Jan 2025 | DeepSeek R1 released |
| 17 Jan 2026 | Paper I published |
| 22 Jan 2026 | Paper II published |
| 6 Feb 2026 | NYU time crystal paper published |
| 9 Feb 2026 | Paper III published |
| 20 Feb 2026 | Paper III v9.0 (current version) |
| 22 Feb 2026 | Complete document suite finalised |

The theoretical prediction preceded all experimental confirmations.

## Citation

```bibtex
@article{eastwood2026arc,
  title={The ARC Principle: The Alignment Scaling Problem},
  author={Eastwood, Michael Darius},
  year={2026},
  month={February},
  note={White Paper III, Version 9.0},
  doi={10.17605/OSF.IO/6C5XB}
}
```

## OSF DOIs

| Deposit | DOI |
|---------|-----|
| Complete Suite (Parent) | [10.17605/OSF.IO/6C5XB](https://doi.org/10.17605/OSF.IO/6C5XB) |
| Paper II (Experimental) | [10.17605/OSF.IO/8FJMA](https://doi.org/10.17605/OSF.IO/8FJMA) |
| Paper III (Cross-Domain) | [10.17605/OSF.IO/HQCGF](https://doi.org/10.17605/OSF.IO/HQCGF) |

## Related

- **Book:** Eastwood, M.D. (2026). *Infinite Architects: Intelligence, Recursion, and the Creation of Everything*. ISBN 978-1806056200.

## Licence

MIT License — See [LICENCE](LICENCE) for details.

## Contributing

All contributions welcome, **including falsifications**.

If you find evidence that contradicts the ARC Principle, please open an issue or submit a pull request. Good science requires rigorous testing.

---

**Priority Established:** 8 December 2024 (DKIM-verified manuscript submission)

**Last Updated:** 22 February 2026

**Copyright 2026 Michael Darius Eastwood**
