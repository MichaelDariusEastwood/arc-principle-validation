# ARC Principle Validation

**Experimental validation of Eastwood's ARC Principle**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the complete experimental code, raw data, and figures for validating the ARC Principle (Artificial Recursive Creation).

**Paper:** Eastwood's ARC Principle: Experimental Validation of Super-Linear Error Suppression Through Sequential Recursive Processing

**Author:** Michael Darius Eastwood

**Date:** 22 January 2026

## Key Findings

| Condition | Accuracy | Error Rate | Tokens | α (scaling exponent) |
|-----------|----------|------------|--------|---------------------|
| Sequential (best) | 91.7% | 0.083 | 412 | **2.24** |
| Parallel (best) | 66.7% | 0.333 | 1,101 | **0.0** |

**Core result:** Sequential recursion with 412 tokens outperformed parallel recursion with 1,101 tokens by 25 percentage points.

**The form of recursion matters more than the amount of compute.**

## Repository Structure

```
eastwoods-arc-principle/
├── README.md                 ← You are here
├── code/
│   └── arc_validation_deepseek.py    ← Complete experiment script
├── data/
│   └── arc_deepseek_results_20260121_175028.json    ← Raw experimental data
├── figures/
│   ├── figure_1_raw_data.png
│   ├── figure_2_scaling_loglog.png
│   ├── ... (15 figures total)
│   └── figure_15_complete_summary.png
└── paper/
    ├── WHITEPAPER-I-PRELIMINARY-EVIDENCE.pdf       ← Paper I (17 Jan 2026)
    ├── WHITEPAPER-II-EXPERIMENTAL-VALIDATION.md    ← Paper II source
    └── WHITEPAPER-II-EXPERIMENTAL-VALIDATION.pdf   ← Paper II (22 Jan 2026)
```

## Quick Start

### Prerequisites

```bash
pip install openai numpy scipy matplotlib pandas
```

### Set API Key

```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key"
```

### Run Experiment

```bash
cd code
python arc_validation_deepseek.py
```

## The ARC Principle

**Equation:**
```
E(R) = E₀ × R^(−α)
```

Where:
- E(R) = Error rate at recursive depth R
- E₀ = Baseline error rate
- R = Recursive depth (tokens or samples)
- α = Scaling exponent

**Core prediction:** α_sequential > 1 > α_parallel

## Replication

This experiment used:
- **Model:** DeepSeek R1 (deepseek-reasoner)
- **API:** DeepSeek API with visible reasoning tokens
- **Problems:** 12 AIME-level mathematics problems
- **Sequential budgets:** 512, 1024, 2048, 4096 tokens
- **Parallel samples:** N = 1, 2, 4

## Citation

```bibtex
@article{eastwood2026arc,
  title={Eastwood's ARC Principle: Experimental Validation of Super-Linear Error Suppression Through Sequential Recursive Processing},
  author={Eastwood, Michael Darius},
  year={2026},
  month={January},
  note={Paper II in the ARC Principle series}
}
```

## Related Work

- **Paper I:** Eastwood, M.D. (2026). Eastwood's ARC Principle: Preliminary Evidence for Super-Linear Capability Amplification Through Sequential Self-Reference. Published 17 January 2026.

- **Book:** Eastwood, M.D. (2026). *Infinite Architects: Intelligence, Recursion, and the Creation of Everything*. ISBN: 978-1806056200.

## Licence

MIT License - See [LICENCE](LICENCE) for details.

## Contributing

All contributions welcome, **including falsifications**.

If you find evidence that contradicts the ARC Principle, please open an issue or submit a pull request. Good science requires rigorous testing.

---

**Priority Established:** 8 December 2024 (DKIM-verified manuscript submission)

**Copyright © 2026 Michael Darius Eastwood**
