# ARC Principle Validation

**Experimental validation of Eastwood's ARC Principle**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the complete experimental code, raw data, figures, and research toolkits for validating the ARC Principle (Artificial Recursive Creation).

**Papers:**
- **Paper I:** Preliminary Evidence for Super-Linear Capability Amplification (17 January 2026)
- **Paper II:** Experimental Validation of Super-Linear Error Suppression (22 January 2026)
- **Paper III v6.5:** Cross-Domain Unification Across AI, Quantum Computing, and Physics (11 February 2026)
- **Foundational Paper v1.1:** Condensed Theoretical Treatment (11 February 2026) **NEW**

**Author:** Michael Darius Eastwood

## Key Findings

| Condition | Accuracy | Error Rate | Tokens | alpha |
|-----------|----------|------------|--------|-------|
| Sequential (best) | 91.7% | 0.083 | 412 | **2.2** |
| Parallel (best) | 66.7% | 0.333 | 1,101 | **0.0** |

**Core result:** Sequential recursion with 412 tokens outperformed parallel recursion with 1,101 tokens by 25 percentage points.

**The form of recursion matters more than the amount of compute.**

## Repository Structure

```
arc-principle-validation/
├── README.md                       # You are here
├── LICENCE                         # MIT Licence
├── OSF-PAPER-III-UPLOAD.md         # OSF upload guide
│
├── paper/                          # Published Papers
│   ├── EASTWOOD-ARC-PRINCIPLE-PAPER-I-v1.1.pdf
│   ├── EASTWOOD-ARC-PRINCIPLE-PAPER-II-v11.pdf
│   ├── WHITE-PAPER-III-ARC-PRINCIPLE-v6.5.pdf      ← UPDATED
│   ├── WHITE-PAPER-III-ARC-PRINCIPLE-v6.5.html
│   ├── ARC-PRINCIPLE-FOUNDATIONAL-v1.1.pdf         ← NEW
│   ├── ARC-PRINCIPLE-FOUNDATIONAL-v1.1.html        ← NEW
│   └── figures/                    # Paper III figures (11 PNGs)
│
├── validation/                     # Computational Validation Suite
│   ├── prove_IxR_equals_complexity_v2.py   # I×R proof with honest caveats
│   ├── arc_definitive_test.py              # Definitive validation suite
│   └── arc_unified_paradigm_test.py        # Unified paradigm tests
│
├── blind-test/                     # Blind Prediction Test & Analysis
│   ├── BLIND_PREDICTION_TEST.py            # Blind test protocol
│   ├── BLIND_TEST_FORENSIC_ANALYSIS.md     # Forensic analysis of results
│   └── BLIND_TEST_FORENSIC_ANALYSIS.png    # Visual evidence
│
└── research-toolkits/              # REPLICATION TOOLKITS
    ├── README.md
    ├── paper-i/                    # Paper I toolkit
    └── paper-ii/                   # Paper II toolkit
```

## Quick Start

### Validation Suite

```bash
cd validation
pip install numpy scipy matplotlib
python prove_IxR_equals_complexity_v2.py
```

### Blind Prediction Test

```bash
cd blind-test
python BLIND_PREDICTION_TEST.py
```

**Note:** The blind test identified methodological confounds. See `BLIND_TEST_FORENSIC_ANALYSIS.md` for full analysis.

## The ARC Principle

**Generalised Equation:**
```
U(R) = I × f(R, β)
```

**Key Derivation:**
```
α = 1/(1-β)
```

Where:
- U = Effective capability
- I = Base potential (structured asymmetry)
- R = Recursive depth
- β = Self-referential coupling parameter
- α = Scaling exponent (derived, not fitted)

**Core prediction:** α_sequential > 1 > α_parallel

## Paper III v6.5: Cross-Domain Unification

Paper III synthesises convergent evidence from **four independent research programmes**:

| Domain | System | Key Finding |
|--------|--------|-------------|
| AI | DeepSeek R1 | Sequential α ≈ 1.3-2.2, parallel α ≈ 0 |
| Quantum | Google Willow | Exponential error suppression (Λ = 2.14) |
| Physics | NYU Time Crystals | Frozen disorder + feedback = temporal order |
| Neuroscience | COGITATE | Recurrent processing required for consciousness |

**New in v6.5:**
- Composition operator formalism (Theorem 5)
- Blind prediction test results with forensic analysis
- Updated falsification criteria (F4 now "Mixed")
- 11 publication-quality figures

## Foundational Paper v1.1: Condensed Theory

Rigorous axiomatic treatment deriving α = 1/(1-β) from three axioms:
- **Axiom 1:** Dimensional consistency (U = I × g(R))
- **Axiom 2:** Recursive coupling (dQ/dr = aQ^β)
- **Axiom 3:** Compositional self-similarity

**Computational Validation:** R² = 1.00000000 (machine precision)

**Includes:** Full mathematical proofs, Appendix C validation suite, honest limitations.

## Blind Prediction Test

A blind prediction test was conducted to validate α = 1/(1-β):

| System | β measured | α predicted | α measured | Result |
|--------|------------|-------------|------------|--------|
| BA Network | 0.70 | 3.33 | 0.34 | FAIL |
| Gradient Descent | 0.95 | 20.0 | 0.87 | FAIL |
| Kuramoto | 0.55 | 2.24 | 0.55 | PASS* |

**Forensic Analysis:** Two confounds identified:
1. **Measurement bias:** Numerical-derivative β estimation gives β ≈ 0.95 regardless of true β
2. **Axiom violation:** None of the tested systems satisfy constant coupling coefficient (Axiom 2)

**Verdict:** Not valid falsification due to confounds. Proper linearisation method recovers R² = 0.9999.

See `blind-test/BLIND_TEST_FORENSIC_ANALYSIS.md` for full analysis.

## Summary of Results

| Source | Recursion Type | alpha | Method |
|--------|----------------|-------|--------|
| Paper I | Parallel (o1) | 0.1-0.3 | Published data analysis |
| Paper I | Sequential (R1) | ~1.34 | Published data analysis |
| Paper II | Sequential | **2.2** | Direct experiment |
| Paper II | Parallel | **0.0** | Direct experiment |
| Validation Suite | Bernoulli ODE | **R²=1.0** | Computational |

**Confirmed:** α_sequential > 1 > α_parallel

## Citation

```bibtex
@article{eastwood2026arc3,
  title={Eastwood's ARC Principle: Cross-Domain Unification of Recursive Amplification Across AI, Quantum Computing, and Physics},
  author={Eastwood, Michael Darius},
  year={2026},
  month={February},
  day={11},
  note={White Paper III, Version 6.5}
}

@article{eastwood2026arcfoundational,
  title={The ARC Principle: Recursive Amplification as a Cross-Domain Structural Principle},
  author={Eastwood, Michael Darius},
  year={2026},
  month={February},
  day={11},
  note={Foundational Paper, Version 1.1}
}
```

## OSF DOIs

| Paper | DOI |
|-------|-----|
| Paper I | [10.17605/OSF.IO/6C5XB](https://doi.org/10.17605/OSF.IO/6C5XB) |
| Paper II | [10.17605/OSF.IO/8FJMA](https://doi.org/10.17605/OSF.IO/8FJMA) |
| Paper III + Foundational | [10.17605/OSF.IO/HQCGF](https://doi.org/10.17605/OSF.IO/HQCGF) |

## Related Resources

- **Book:** Eastwood, M.D. (2026). *Infinite Architects: Intelligence, Recursion, and the Creation of Everything*. ISBN: 978-1806056200.

## Licence

MIT License - See [LICENCE](LICENCE) for details.

## Contributing

All contributions welcome, **including falsifications**.

If you find evidence that contradicts the ARC Principle, please open an issue or submit a pull request. Good science requires rigorous testing.

---

**Priority Established:** 8 December 2024 (DKIM-verified manuscript submission)

**Last Updated:** 11 February 2026

**Copyright 2026 Michael Darius Eastwood**
