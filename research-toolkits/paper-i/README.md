# Paper I Research Toolkit: Preliminary Evidence Analysis

**Eastwood's ARC Principle - Paper I**

*Test the equation yourself. Verify the data. Challenge the findings.*

---

## Overview

This toolkit analyses publicly available data from OpenAI o1 and DeepSeek R1 technical reports to calculate scaling exponents for the ARC Principle.

**Paper:** Eastwood's ARC Principle: Preliminary Evidence for Super-Linear Capability Amplification Through Sequential Self-Reference

**Author:** Michael Darius Eastwood

**Date:** 17 January 2026

---

## Quick Start

```bash
cd paper-i
pip install -r requirements.txt
python code/arc_principle_research_toolkit.py
```

---

## What It Does

- Calculates scaling exponents for parallel recursion (OpenAI o1)
- Calculates scaling exponents for sequential recursion (DeepSeek R1)
- Runs sensitivity analysis across token ratio assumptions
- Generates publication-quality visualisations
- Exports results to JSON

---

## Folder Structure

```
paper-i/
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── LICENCE                 # MIT Licence
├── code/
│   └── arc_principle_research_toolkit.py
├── results/
│   └── arc_principle_results.json
└── figures/
    ├── arc_scaling_comparison.png
    ├── arc_sensitivity_analysis.png
    └── arc_falsification_regions.png
```

---

## Key Functions

```python
from code.arc_principle_research_toolkit import calculate_alpha_two_points

alpha = calculate_alpha_two_points(
    R1=12000,   # Recursive depth point 1
    R2=23000,   # Recursive depth point 2
    E1=0.30,    # Error rate at point 1 (30%)
    E2=0.125    # Error rate at point 2 (12.5%)
)
print(f"alpha = {alpha:.3f}")  # alpha = 1.35
```

---

## Output Files

| File | Location | Description |
|------|----------|-------------|
| `arc_principle_research_toolkit.py` | `code/` | Main analysis script |
| `arc_principle_results.json` | `results/` | Pre-computed results |
| `arc_scaling_comparison.png` | `figures/` | Parallel vs Sequential comparison |
| `arc_sensitivity_analysis.png` | `figures/` | Alpha sensitivity to assumptions |
| `arc_falsification_regions.png` | `figures/` | Falsification criteria visualisation |

---

## The Equation

```
E(R) = E_0 x R^(-alpha)
```

Where:
- **E(R)** = Error rate at recursive depth R
- **E_0** = Baseline error rate
- **R** = Recursive depth (tokens)
- **alpha** = Scaling exponent

**Core Prediction:** alpha_sequential > 1 > alpha_parallel

---

## Data Sources

| Source | Data |
|--------|------|
| OpenAI o1 System Card | AIME 2024 accuracy at 1, 64, 1000 samples |
| DeepSeek R1 Report | AIME 2024 accuracy at ~12K and ~23K tokens |

---

## Falsification Criteria

| alpha Range | Interpretation |
|-------------|----------------|
| alpha < 1.0 | Sub-linear (diminishing returns) |
| 1.0 <= alpha < 1.5 | Super-linear but weak |
| 1.5 <= alpha < 2.3 | Super-linear approaching quadratic |
| alpha > 2.5 | Exponential (different relationship) |

---

## Citation

```bibtex
@article{eastwood2026arc1,
  title={Eastwood's ARC Principle: Preliminary Evidence for Super-Linear Capability Amplification Through Sequential Self-Reference},
  author={Eastwood, Michael Darius},
  year={2026},
  note={Paper I}
}
```

---

## Contribute

Found issues? Want to replicate? Challenge the findings?

1. Fork this repository
2. Add your analysis
3. Submit a pull request with your findings

**All contributions welcome, including falsifications.**

---

**Copyright 2026 Michael Darius Eastwood**
