# ARC Principle Validation

**Experimental validation of Eastwood's ARC Principle**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the complete experimental code, raw data, figures, and research toolkits for validating the ARC Principle (Artificial Recursive Creation).

**Papers:**
- **Paper I:** Preliminary Evidence for Super-Linear Capability Amplification (17 January 2026)
- **Paper II:** Experimental Validation of Super-Linear Error Suppression (22 January 2026)

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
│
├── paper/                          # Published PDFs
│   ├── EASTWOOD-ARC-PRINCIPLE-PAPER-I-v1.1.pdf
│   └── EASTWOOD-ARC-PRINCIPLE-PAPER-II-v11.pdf
│
└── research-toolkits/              # REPLICATION TOOLKITS
    ├── README.md
    │
    ├── paper-i/                    # Paper I: Preliminary Evidence
    │   ├── README.md
    │   ├── requirements.txt
    │   ├── LICENCE
    │   ├── code/
    │   │   └── arc_principle_research_toolkit.py
    │   ├── results/
    │   │   └── arc_principle_results.json
    │   └── figures/
    │       └── *.png (3 visualisations)
    │
    └── paper-ii/                   # Paper II: Experimental Validation
        ├── README.md
        ├── requirements.txt
        ├── LICENCE
        ├── code/
        │   └── arc_validation_deepseek.py
        ├── results/
        │   └── arc_deepseek_results_*.json
        └── figures/
            └── *.png (15 figures)
```

## Quick Start

### Paper I Toolkit (Published Data Analysis)

```bash
cd research-toolkits/paper-i
pip install -r requirements.txt
python code/arc_principle_research_toolkit.py
```

### Paper II Toolkit (Direct Experiment)

```bash
cd research-toolkits/paper-ii
pip install -r requirements.txt
export DEEPSEEK_API_KEY="your-deepseek-api-key"
python code/arc_validation_deepseek.py
```

## The ARC Principle

**Equation:**
```
E(R) = E_0 x R^(-alpha)
```

Where:
- E(R) = Error rate at recursive depth R
- E_0 = Baseline error rate
- R = Recursive depth (tokens or samples)
- alpha = Scaling exponent

**Core prediction:** alpha_sequential > 1 > alpha_parallel

## Summary of Results

| Source | Recursion Type | alpha | Method |
|--------|----------------|-------|--------|
| Paper I | Parallel (o1) | 0.1-0.3 | Published data analysis |
| Paper I | Sequential (R1) | ~1.34 | Published data analysis |
| Paper II | Sequential | **2.2** | Direct experiment |
| Paper II | Parallel | **0.0** | Direct experiment |

**Confirmed:** alpha_sequential > 1 > alpha_parallel

## Citation

```bibtex
@article{eastwood2026arc1,
  title={Eastwood's ARC Principle: Preliminary Evidence for Super-Linear Capability Amplification Through Sequential Self-Reference},
  author={Eastwood, Michael Darius},
  year={2026},
  note={Paper I}
}

@article{eastwood2026arc2,
  title={Eastwood's ARC Principle: Experimental Validation of Super-Linear Error Suppression Through Sequential Recursive Processing},
  author={Eastwood, Michael Darius},
  year={2026},
  note={Paper II}
}
```

## Related Work

- **Book:** Eastwood, M.D. (2026). *Infinite Architects: Intelligence, Recursion, and the Creation of Everything*. ISBN: 978-1806056200.

## Licence

MIT License - See [LICENCE](LICENCE) for details.

## Contributing

All contributions welcome, **including falsifications**.

If you find evidence that contradicts the ARC Principle, please open an issue or submit a pull request. Good science requires rigorous testing.

---

**Priority Established:** 8 December 2024 (DKIM-verified manuscript submission)

**Copyright 2026 Michael Darius Eastwood**
