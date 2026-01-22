# Paper III Research Toolkit: Eden Protocol Validation

**Eastwood's ARC Principle - Paper III**

*Test whether alignment scales with capability. Validate the core safety claim.*

---

## Status: EXPERIMENTAL DESIGN PHASE

This toolkit is under development. The experiment has been designed but not yet executed.

---

## Overview

Papers I and II established preliminary support for capability scaling (α ≈ 2.2 for sequential recursion). **But the Eden Protocol's core claim remains untested:**

> Values embedded at the reasoning level will scale alongside capability, while external constraints will not.

This experiment directly tests that prediction.

---

## The Core Predictions

```
If Eden Protocol is correct:

A_rules(R) ≈ O(1)        → Alignment via constraints stays flat
A_values(R) ≈ O(R^β)     → Alignment via values scales with depth
β_values ≈ α_capability   → Values scale alongside capability
```

---

## Experimental Design

| Component | Description |
|-----------|-------------|
| **Conditions** | Rules-as-Constraints vs Values-as-Reasoning |
| **Depths** | 512, 1024, 2048, 4096, 8192 tokens |
| **Test cases** | 150 adversarial and ethical scenarios |
| **Metrics** | Jailbreak resistance, consistency, generalisation, reasoning quality |
| **Model** | DeepSeek R1 (same as Paper II) |

---

## Folder Structure

```
paper-iii/
├── README.md                     # This file
├── EXPERIMENTAL-DESIGN.md        # Full experimental protocol
├── test_suite_template.json      # Adversarial prompt test suite
├── code/                         # (To be implemented)
│   ├── run_experiment.py
│   ├── evaluate_alignment.py
│   └── analysis.py
├── results/                      # (After experiment)
└── figures/                      # (After experiment)
```

---

## Current Files

| File | Status | Description |
|------|--------|-------------|
| `EXPERIMENTAL-DESIGN.md` | Complete | Full experimental protocol |
| `test_suite_template.json` | Draft | Example test cases (needs expansion) |
| `code/` | Not started | Experiment implementation |
| `results/` | Not started | Will contain experimental data |
| `figures/` | Not started | Will contain visualisations |

---

## Next Steps

### Immediate (This Week)

- [ ] Expand test suite to 150 cases
- [ ] Implement basic experiment runner
- [ ] Run pilot with N=30

### Short-term (2-4 Weeks)

- [ ] Complete full test suite
- [ ] Develop human evaluation interface
- [ ] Run main experiment

### Medium-term (1-2 Months)

- [ ] Complete analysis
- [ ] Write Paper III

---

## Resource Requirements

| Component | Estimated Cost |
|-----------|---------------|
| API calls | $1,000-2,500 |
| Human evaluation | $2,000-3,500 |
| **Total** | **$3,000-6,000** |

---

## What Success Looks Like

**If Eden Protocol is supported:**

```
β_values >> β_rules
β_values ≈ α_capability

Values-as-Reasoning shows super-linear alignment scaling
Rules-as-Constraints shows flat alignment scaling
```

**If Eden Protocol is falsified:**

```
β_values ≈ β_rules
or
β_rules > β_values

Integration depth does not matter for alignment scaling
```

Either result advances the field.

---

## Citation

```bibtex
@article{eastwood2026arc3,
  title={Eastwood's ARC Principle: Empirical Validation of Values-as-Reasoning Scaling},
  author={Eastwood, Michael Darius},
  year={2026},
  note={Paper III - Eden Protocol Validation}
}
```

---

## The Honest Framing

**What Papers I & II established:**
- Preliminary support for E(R) = E₀ × R^(-α) with α ≈ 2.2

**What Paper III will test:**
- Whether alignment properties participate in this scaling
- Whether values-as-reasoning scales differently than rules-as-constraints
- The central prediction of the Eden Protocol

**What Paper III cannot establish alone:**
- That the Eden Protocol is "proven"
- Generalisation to all models
- That fine-tuned values behave identically to prompt-induced values

---

**Status:** Experimental design complete. Implementation pending.

**Copyright 2026 Michael Darius Eastwood**
