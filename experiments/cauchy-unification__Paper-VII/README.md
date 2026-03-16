# Cauchy Unification -- Paper VII

Experimental scripts and results supporting **Paper VII: The Cauchy Unification (v1)**.

## Core claim

The form of every scaling law is determined by the composition operator of the underlying recursive process. Cauchy's four functional equations constrain which forms are possible. Tested blindly across 20 domains: 20/20 confirmed at R^2 level (p = 2.87e-10); 14/20 under strict AIC model selection (p = 8.79e-4).

## Scripts

All scripts are in `scripts/`. They require Python 3 with NumPy and SciPy.

| Script | Description | Role |
|--------|-------------|------|
| `arc_20_domain_universal_test.py` | **Primary.** Blind prediction test across 20 domains (7 multiplicative, 3 additive, 10 bounded). Classifies composition operator from physics, predicts scaling form, loads real data, fits independently, scores. | Canonical experiment cited in Paper VII |
| `arc_complete_test_suite.py` | Test 3: Cauchy no-go theorem verification (exactly three scaling forms). Also includes other ARC tests. | Supporting -- Cauchy theorem numerics |
| `arc_unified_paradigm_test.py` | Cauchy classification across 15 test cases (3 regimes). Unified paradigm validation including Cauchy Theorem 1. | Supporting -- regime classification |
| `arc_rigorous_validation.py` | Tier 1 mathematical foundation: numerical verification of Cauchy's three functional equations. | Supporting -- mathematical foundation |
| `arc_universal_proof.py` | Universal proof of ARC Principle from Cauchy's functional equations and maximum entropy. | Supporting -- theoretical derivation |

## Results

Pre-computed outputs are in `results/`. Each corresponds to a script above:

| Results file | Source script |
|--------------|---------------|
| `results_20_domain_validation.txt` | `arc_20_domain_universal_test.py` |
| `results_complete_suite.txt` | `arc_complete_test_suite.py` |
| `results_arc_unified_paradigm_test.txt` | `arc_unified_paradigm_test.py` |
| `results_arc_rigorous_validation.txt` | `arc_rigorous_validation.py` |
| `results_arc_universal_proof.txt` | `arc_universal_proof.py` |

## How to run

```bash
# Primary 20-domain blind test (the canonical Paper VII experiment)
python3 scripts/arc_20_domain_universal_test.py

# Supporting tests
python3 scripts/arc_complete_test_suite.py
python3 scripts/arc_unified_paradigm_test.py
python3 scripts/arc_rigorous_validation.py
python3 scripts/arc_universal_proof.py
```

## Provenance

These scripts were originally developed in `domain-validation__Foundational-and-Origin/` and `validation/` as part of the Foundational and Origin paper validation programme. They are collected here because Paper VII (Cauchy Unification) directly cites the 20-domain blind test as its primary evidence and the Cauchy theorem verification as its mathematical foundation.

## Evidence tier

- **Primary (20-domain test):** Supporting (mathematical/computational validation, synthetic + published data)
- **Supporting scripts:** Supporting (mathematical validation, numerical verification of proven theorems)

## Paper

Paper VII: The Cauchy Unification (v1), located at:
`paper/FINAL-SUITE/v-major/Paper-VII-Cauchy-Unification-v1.html`
