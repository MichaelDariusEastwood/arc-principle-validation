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
├── README.md                 # You are here
├── LICENCE                   # MIT Licence
├── requirements.txt          # Python dependencies
│
├── paper/                    # Published papers
│   ├── EASTWOOD-ARC-PRINCIPLE-PAPER-I-v1.1.pdf
│   └── EASTWOOD-ARC-PRINCIPLE-PAPER-II-v11.pdf
│
├── code/                     # Experiment scripts (legacy location)
│   └── arc_validation_deepseek.py
│
├── data/                     # Raw experimental data (legacy location)
│   └── arc_deepseek_results_20260121_175028.json
│
├── figures/                  # Publication figures (legacy location)
│   └── figure_1_raw_data.png ... figure_15_complete_summary.png
│
└── research-toolkits/        # REPLICATION TOOLKITS
    ├── README.md             # Toolkit overview
    ├── paper-i/              # Paper I toolkit
    │   ├── README.md
    │   ├── arc_principle_research_toolkit.py
    │   ├── arc_principle_results.json
    │   ├── arc_scaling_comparison.png
    │   ├── arc_sensitivity_analysis.png
    │   ├── arc_falsification_regions.png
    │   ├── requirements.txt
    │   └── LICENCE
    │
    └── paper-ii/             # Paper II toolkit
        ├── README.md
        ├── arc_validation_deepseek.py
        ├── arc_deepseek_results_20260121_175028.json
        ├── figure_1_raw_data.png ... (15 figures)
        ├── requirements.txt
        └── LICENCE
```

## Quick Start

### Paper I Toolkit (Published Data Analysis)

```bash
cd research-toolkits/paper-i
pip install -r requirements.txt
python arc_principle_research_toolkit.py
```

### Paper II Toolkit (Direct Experiment)

```bash
cd research-toolkits/paper-ii
pip install -r requirements.txt
export DEEPSEEK_API_KEY="your-deepseek-api-key"
python arc_validation_deepseek.py
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

## Replication

Paper II experiment used:
- **Model:** DeepSeek R1 (deepseek-reasoner)
- **API:** DeepSeek API with visible reasoning tokens
- **Problems:** 12 AIME-level mathematics problems
- **Sequential budgets:** 512, 1024, 2048, 4096 tokens
- **Parallel samples:** N = 1, 2, 4

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
