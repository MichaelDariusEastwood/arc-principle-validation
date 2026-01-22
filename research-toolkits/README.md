# ARC Principle Research Toolkits

**Test the equation yourself. Verify the data. Challenge the findings.**

This folder contains experimental validation scripts for both ARC Principle papers.

---

## Toolkit Structure

```
research-toolkits/
├── paper-i/                    # Paper I: Preliminary Evidence
│   ├── arc_principle_research_toolkit.py
│   ├── arc_principle_results.json
│   ├── arc_scaling_comparison.png
│   ├── arc_sensitivity_analysis.png
│   ├── arc_falsification_regions.png
│   ├── requirements.txt
│   ├── LICENCE
│   └── README.md
│
├── paper-ii/                   # Paper II: Experimental Validation
│   ├── arc_validation_deepseek.py
│   ├── arc_deepseek_results_20260121_175028.json
│   ├── figure_1_raw_data.png
│   ├── figure_2_scaling_loglog.png
│   ├── ... (15 figures total)
│   ├── requirements.txt
│   ├── LICENCE
│   └── README.md
│
└── README.md                   # This file
```

---

## Paper I Toolkit: Preliminary Evidence Analysis

**Location:** `paper-i/`

Analyses publicly available data from OpenAI o1 and DeepSeek R1 technical reports to calculate scaling exponents.

```bash
cd paper-i
pip install -r requirements.txt
python arc_principle_research_toolkit.py
```

**Key outputs:**
- Scaling exponent calculations
- Sensitivity analysis across token ratio assumptions
- Publication-quality visualisations

See [paper-i/README.md](paper-i/README.md) for full documentation.

---

## Paper II Toolkit: Direct Experimental Validation

**Location:** `paper-ii/`

Runs controlled experiments using DeepSeek R1 API with visible reasoning tokens.

```bash
cd paper-ii
pip install -r requirements.txt
export DEEPSEEK_API_KEY="your-api-key"
python arc_validation_deepseek.py
```

**Key outputs:**
- Direct measurement of alpha = 2.2 (sequential) vs 0.0 (parallel)
- 15 publication-quality figures
- Complete experimental data in JSON

See [paper-ii/README.md](paper-ii/README.md) for full documentation.

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

## Summary of Results

| Paper | Recursion Type | alpha | Method |
|-------|----------------|-------|--------|
| Paper I | Parallel (o1) | 0.1-0.3 | Published data analysis |
| Paper I | Sequential (R1) | ~1.34 | Published data analysis |
| Paper II | Sequential | **2.2** | Direct experiment |
| Paper II | Parallel | **0.0** | Direct experiment |

**Confirmed:** alpha_sequential > 1 > alpha_parallel

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
