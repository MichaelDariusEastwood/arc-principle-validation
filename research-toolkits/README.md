# ARC Principle Research Toolkits

**Test the equation yourself. Verify the data. Challenge the findings.**

This folder contains experimental validation scripts for both ARC Principle papers.

---

## Paper I Toolkit: Preliminary Evidence Analysis

**File:** `arc_principle_research_toolkit.py`

Analyses publicly available data from OpenAI o1 and DeepSeek R1 technical reports to calculate scaling exponents.

### Quick Start

```bash
cd research-toolkits
pip install numpy scipy matplotlib pandas seaborn
python arc_principle_research_toolkit.py
```

### What It Does

- Calculates scaling exponents for parallel recursion (OpenAI o1)
- Calculates scaling exponents for sequential recursion (DeepSeek R1)
- Runs sensitivity analysis across token ratio assumptions
- Generates publication-quality visualisations
- Exports results to JSON

### Key Functions

```python
from arc_principle_research_toolkit import calculate_alpha_two_points

alpha = calculate_alpha_two_points(
    R1=12000,   # Recursive depth point 1
    R2=23000,   # Recursive depth point 2
    E1=0.30,    # Error rate at point 1 (30%)
    E2=0.125    # Error rate at point 2 (12.5%)
)
print(f"α = {alpha:.3f}")  # α ≈ 1.35
```

### Output Files

- `arc_principle_results.json` - Pre-computed results
- `arc_scaling_comparison.png` - Parallel vs Sequential comparison
- `arc_sensitivity_analysis.png` - α sensitivity to assumptions
- `arc_falsification_regions.png` - Falsification criteria

---

## Paper II Toolkit: Direct Experimental Validation

**File:** `../code/arc_validation_deepseek.py`

Runs controlled experiments using DeepSeek R1 API with visible reasoning tokens.

### Quick Start

```bash
cd code
pip install openai
export DEEPSEEK_API_KEY="your-api-key"
python arc_validation_deepseek.py
```

### What It Does

- Tests 12 AIME-level mathematics problems
- Compares sequential vs parallel recursion at matched compute
- Directly measures reasoning token counts
- Calculates α ≈ 2.2 (sequential) vs α ≈ 0.0 (parallel)
- Generates 15 publication-quality figures

### Requirements

- Python 3.8+
- `openai >= 1.0.0`
- DeepSeek API key (get from platform.deepseek.com)

### Output

- Console: Per-problem results and summary statistics
- JSON: Complete experimental data with all token counts
- Figures: 15 visualisations in `../figures/`

---

## The Equation

```
E(R) = E₀ × R⁻ᵅ
```

Where:
- **E(R)** = Error rate at recursive depth R
- **E₀** = Baseline error rate
- **R** = Recursive depth (tokens)
- **α** = Scaling exponent

**Core Prediction:** α_sequential > 1 > α_parallel

---

## Data Sources

| Paper | Source | Data |
|-------|--------|------|
| Paper I | OpenAI o1 System Card | AIME 2024 accuracy at 1, 64, 1000 samples |
| Paper I | DeepSeek R1 Report | AIME 2024 accuracy at ~12K and ~23K tokens |
| Paper II | Direct experiment | 12 problems × 4 token budgets × 2 recursion types |

---

## Falsification Criteria

| α Range | Interpretation |
|---------|----------------|
| α < 1.0 | Sub-linear (diminishing returns) |
| 1.0 ≤ α < 1.5 | Super-linear but weak |
| 1.5 ≤ α < 2.3 | Super-linear approaching quadratic |
| α > 2.5 | Exponential (different relationship) |

---

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

---

## Contribute

Found new data? Run replication studies? Challenge the findings?

1. Fork this repository
2. Add your analysis
3. Submit a pull request with your findings

**All contributions welcome, including falsifications.**

---

**Copyright 2026 Michael Darius Eastwood**
