# The ARC Principle

**A mathematical framework for recursive intelligence scaling, validated across 20 scientific domains by blind prediction**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Headline Result (March 2026)

**Paper VII: The Cauchy Unification** tested whether Cauchy's four functional equations determine the form of every scaling law in nature. Using a blind prediction protocol - the composition operator was classified before data was examined - the framework was tested across 20 scientific domains spanning biology, physics, economics, epidemiology, linguistics, computer science, and ecology.

- **20/20 domains confirmed** at the R² level (p = 2.87 x 10⁻¹⁰)
- **14/20 confirmed** under strict AIC model selection (p = 8.79 x 10⁻⁴)
- Less than **one in three billion** probability by chance - exceeding the 5σ discovery threshold used in particle physics

[Read Paper VII](paper/FINAL-SUITE/v-major/Paper-VII-Cauchy-Unification-v1.html) | [Experiment code](experiments/cauchy-unification__Paper-VII/)

## Overview

This repository contains the complete document suite, experimental code, raw data, figures, and research toolkits for the ARC Principle framework and the Eden Protocol alignment architecture, first proposed in *Infinite Architects: Intelligence, Recursion, and the Creation of Everything* (Eastwood, 2026).

**Author:** Michael Darius Eastwood
**London, United Kingdom**
**Web:** [michaeldariuseastwood.com](https://www.michaeldariuseastwood.com) | **Research hub:** [michaeldariuseastwood.com/research](https://www.michaeldariuseastwood.com/research/)

## The Document Suite

All authoritative documents are in `paper/FINAL-SUITE/v-major/`:

| # | Document | Version | Description |
|---|----------|---------|-------------|
| 01 | **Paper I: The ARC Principle** | v1.1 | Original published paper. U = I x R^α, January 2026. |
| 02 | **Foundational: Cauchy Framework** | v4 | Axiomatic derivation from Cauchy functional equations, d/(d+1) prediction, five universal properties. |
| 03 | **On the Origin of Scaling Laws** | v2 | Cross-domain d/(d+1) evidence catalogue, 8 independent derivations, 6-level evidence hierarchy. |
| 04 | **Paper II: Experimental Validation** | v12 | 6-model direct experiment. Sequential vs parallel recursion. |
| 05 | **Paper III: White Paper** | v11 | Alignment scaling problem, ARC Bound, 13 falsification criteria. |
| 06 | **Paper IV.a: Response Classes** | v1 | Architecture-dependent alignment response to depth. |
| 07 | **Paper IV.b: Saturation** | v1 | Alignment saturation at low depth. |
| 08 | **Paper IV.c: ARC-Align Benchmark** | v1 | Reproducible benchmark specification. |
| 09 | **Paper IV.d: Blinding** | v1 | Unblinded scoring can reverse alignment results. Flagship methods paper. |
| 10 | **Paper V: The Stewardship Gene** | v2 | Eden Protocol intervention pilot. |
| 11 | **Paper VI: Honey Architecture** | v1 | Entangled loss functions for self-modifying AI safety. |
| 12 | **Paper VII: The Cauchy Unification** | v1 | 20-domain blind prediction test. Headline empirical result. |
| 13 | **Eden Engineering** | v6 | Protocol architecture specification. |
| 14 | **Eden Vision** | v3 | Public coordination document. |
| 15 | **Executive Summary** | v6 | Programme overview with all current results. |
| 16 | **Master Table of Contents** | v1 | Suite navigation. |

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
├── LICENCE                             # MIT Licence
│
├── paper/FINAL-SUITE/v-major/          # AUTHORITATIVE DOCUMENT SUITE (16 papers)
│
├── experiments/                        # ALL EXPERIMENT CODE AND RESULTS
│   ├── cauchy-unification__Paper-VII/  # 20-domain blind prediction (HEADLINE)
│   ├── alignment-scaling__Papers-IV/   # v1-v5 alignment benchmark + v6 runner
│   ├── eden-intervention__Paper-V/     # Eden Protocol pilot tests
│   ├── honey-architecture__Paper-VI/   # Simulations + live API battery
│   ├── domain-validation__Foundational/# 13 cross-domain validation scripts
│   ├── blind-prediction-test/          # Forensic analysis of blind test
│   ├── paper-i-foundational__Paper-I/  # Original ARC toolkit
│   ├── paper-ii-compute__Paper-II/     # 6-model direct experiments
│   └── analysis-tools__Cross-Programme/# Shared analysis utilities
│
├── research-toolkits/                  # LEGACY REPLICATION TOOLKITS
│   ├── paper-i/                        # Paper I: public data analysis
│   └── paper-ii/                       # Paper II: DeepSeek R1 experiments
│
└── validation/                         # COMPUTATIONAL VALIDATION
    ├── prove_IxR_equals_complexity_v2.py
    ├── arc_definitive_test.py
    └── arc_unified_paradigm_test.py
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
| 20 Feb 2026 | Paper III v9.0 |
| 22 Feb 2026 | Document suite finalised |
| Mar 2026 | Papers IV.a-d: 6-model blind ARC-Align benchmark |
| Mar 2026 | Paper V: Eden Protocol intervention pilot |
| Mar 2026 | Paper VI: Honey Architecture simulations + live API |
| 16 Mar 2026 | Paper VII: Cauchy Unification - 20-domain blind prediction (p = 2.87e-10) |

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
