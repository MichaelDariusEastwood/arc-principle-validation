# Paper II Research Toolkit: Direct Experimental Validation

**Eastwood's ARC Principle - Paper II**

*Run controlled experiments. Measure directly. Verify the findings.*

---

## Overview

This toolkit runs controlled experiments using DeepSeek R1 API with visible reasoning tokens, enabling direct measurement of recursive depth rather than estimation.

**Paper:** Eastwood's ARC Principle: Experimental Validation of Super-Linear Error Suppression Through Sequential Recursive Processing

**Author:** Michael Darius Eastwood

**Date:** 22 January 2026

---

## Quick Start

```bash
cd paper-ii
pip install -r requirements.txt
export DEEPSEEK_API_KEY="your-api-key"
python arc_validation_deepseek.py
```

**Note:** Requires a DeepSeek API key from [platform.deepseek.com](https://platform.deepseek.com)

---

## What It Does

- Tests 12 AIME-level mathematics problems
- Compares sequential vs parallel recursion at matched compute
- Directly measures reasoning token counts
- Calculates alpha = 2.2 (sequential) vs alpha = 0.0 (parallel)
- Generates 15 publication-quality figures

---

## Experimental Design

| Parameter | Value |
|-----------|-------|
| Model | DeepSeek R1 (deepseek-reasoner) |
| Problems | 12 AIME-level mathematics |
| Sequential budgets | 512, 1024, 2048, 4096 tokens |
| Parallel samples | N = 1, 2, 4 with majority voting |
| Date | 21 January 2026 |

---

## Key Results

| Metric | Sequential | Parallel |
|--------|------------|----------|
| Best accuracy | 91.7% | 66.7% |
| Tokens used | 412 | 1,101 |
| alpha | 2.2 | 0.0 |
| Error reduction | 5x | 0x |

**Key finding:** Sequential with 412 tokens outperformed parallel with 1,101 tokens by 25 percentage points.

---

## Files

| File | Description |
|------|-------------|
| `arc_validation_deepseek.py` | Main experiment script |
| `arc_deepseek_results_20260121_175028.json` | Raw experimental data |
| `figure_1_raw_data.png` | Raw accuracy data |
| `figure_2_scaling_loglog.png` | Log-log scaling plot |
| `figure_4_alpha_comparison.png` | Alpha value comparison |
| `figure_5_error_reduction.png` | Error reduction visualisation |
| `figure_10_summary.png` | Summary dashboard |
| `figure_12_combined_scaling.png` | Combined scaling comparison |
| `figure_13_alpha_summary.png` | Alpha summary across sources |
| `figure_14_form_vs_amount.png` | Form vs amount comparison |
| `figure_15_complete_summary.png` | Complete research summary |
| *(and more...)* | All 15 figures included |
| `requirements.txt` | Python dependencies |
| `LICENCE` | MIT Licence |

---

## The Equation

```
E(R) = E_0 x R^(-alpha)
```

**Measured values:**
- Sequential: alpha = 2.2 (95% CI: 1.5-3.0)
- Parallel: alpha = 0.0

**Core prediction confirmed:** alpha_sequential > 1 > alpha_parallel

---

## Replication Instructions

1. **Get API key:** Sign up at platform.deepseek.com
2. **Set environment variable:** `export DEEPSEEK_API_KEY="your-key"`
3. **Run experiment:** `python arc_validation_deepseek.py`
4. **Compare results:** Check output against `arc_deepseek_results_20260121_175028.json`

**Note:** API costs approximately $2-5 for full replication.

---

## Figures Generated

The script generates 15 publication-quality figures:

1. Raw experimental data
2. Log-log scaling plot
3. Sensitivity analysis
4. Alpha comparison
5. Error reduction
6. Alignment taxonomy
7. Cross-domain evidence
8. Equation visualisation
9. Divergence plot
10. Summary dashboard
11. Experimental data
12. Combined scaling
13. Alpha summary
14. Form vs amount
15. Complete summary

---

## Citation

```bibtex
@article{eastwood2026arc2,
  title={Eastwood's ARC Principle: Experimental Validation of Super-Linear Error Suppression Through Sequential Recursive Processing},
  author={Eastwood, Michael Darius},
  year={2026},
  note={Paper II}
}
```

---

## Contribute

Run your own replication? Found different results? Challenge the findings?

1. Fork this repository
2. Run the experiment with your own API key
3. Submit a pull request with your findings

**All contributions welcome, including falsifications.**

---

**Copyright 2026 Michael Darius Eastwood**
