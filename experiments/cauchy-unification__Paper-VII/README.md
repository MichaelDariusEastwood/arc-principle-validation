# Cauchy Unification -- Paper VII

Experimental scripts and results supporting **Paper VII: The Cauchy Unification (v1)**.

## Core claim

The form of a scaling law is constrained by the composition operator of the
underlying recursive process. The original Paper VII experiment tested this with
an exploratory 20-domain structured prediction comparison. That legacy script is
still present for provenance, but it uses a permissive `R² within 0.05`
confirmation rule and should no longer be treated as the clean canonical
headline.

The stricter canonical suite in this folder is now:

- `scripts/arc_50_domain_universal_test.py`
- `data/canonical_50_domain_manifest.json`

Its design changes are:

- manifest-backed predictions and datasets
- strict family matching with no tolerance rescue
- actual hyperbolic fitting for the muscle domain
- explicit evidence tiers instead of one flat blended score

Current stricter results:

- legacy baseline-20 under the new runner: `15/20` empirical family matches
  (`p = 1.67e-4`)
- expanded empirical cohort: `19/25` empirical family matches
  (`p = 1.56e-5`)
- full 50-domain suite: tiered only, not one single blended p-value

Secondary tiers in the 50-domain suite:

- published exponents, direct transport cases: `13/13`
- published exponents, provisional cases: `3/6`
- analytic identities: `6/6`

## Scripts

All scripts are in `scripts/`. They require Python 3 with NumPy and SciPy.

| Script | Description | Role |
|--------|-------------|------|
| `arc_50_domain_universal_test.py` | **Primary.** Tiered 50-domain validation harness with manifest-backed inputs, strict empirical endpoint, and explicit evidence tiers. | Current canonical experiment |
| `arc_20_domain_universal_test.py` | **Primary.** Blind prediction test across 20 domains (7 multiplicative, 3 additive, 10 bounded). Classifies composition operator from physics, predicts scaling form, loads real data, fits independently, scores. | Canonical experiment cited in Paper VII |
| `arc_complete_test_suite.py` | Test 3: Cauchy no-go theorem verification (exactly three scaling forms). Also includes other ARC tests. | Supporting -- Cauchy theorem numerics |
| `arc_unified_paradigm_test.py` | Cauchy classification across 15 test cases (3 regimes). Unified paradigm validation including Cauchy Theorem 1. | Supporting -- regime classification |
| `arc_rigorous_validation.py` | Tier 1 mathematical foundation: numerical verification of Cauchy's three functional equations. | Supporting -- mathematical foundation |
| `arc_universal_proof.py` | Universal proof of ARC Principle from Cauchy's functional equations and maximum entropy. | Supporting -- theoretical derivation |

## Results

Pre-computed outputs are in `results/`. Each corresponds to a script above:

| Results file | Source script |
|--------------|---------------|
| `results_50_domain_validation.txt` | `arc_50_domain_universal_test.py` |
| `results_50_domain_validation.json` | `arc_50_domain_universal_test.py` |
| `results_20_domain_validation.txt` | `arc_20_domain_universal_test.py` |
| `results_complete_suite.txt` | `arc_complete_test_suite.py` |
| `results_arc_unified_paradigm_test.txt` | `arc_unified_paradigm_test.py` |
| `results_arc_rigorous_validation.txt` | `arc_rigorous_validation.py` |
| `results_arc_universal_proof.txt` | `arc_universal_proof.py` |

## How to run

```bash
# Current canonical tiered suite
python3 scripts/arc_50_domain_universal_test.py

# Legacy 20-domain exploratory comparison retained for provenance
python3 scripts/arc_20_domain_universal_test.py

# Supporting tests
python3 scripts/arc_complete_test_suite.py
python3 scripts/arc_unified_paradigm_test.py
python3 scripts/arc_rigorous_validation.py
python3 scripts/arc_universal_proof.py
```

## Provenance

These scripts were originally developed in `domain-validation__Foundational-and-Origin/`
and `validation/` as part of the Foundational and Origin paper validation
programme. They are collected here because Paper VII (Cauchy Unification)
directly cites the 20-domain comparison as its original empirical evidence and
the Cauchy theorem verification as its mathematical foundation.

## Evidence tier

- **Primary (`arc_50_domain_universal_test.py`):** Supporting
  (manifest-driven mathematical/computational validation with tiered evidence)
- **Legacy (`arc_20_domain_universal_test.py`):** Supporting exploratory
  comparison (mixed provenance, permissive legacy scoring)
- **Supporting scripts:** Supporting (mathematical validation, numerical verification of proven theorems)

## Paper

Paper VII: The Cauchy Unification (v1), located at:
`paper/FINAL-SUITE/v-major/Paper-VII-Cauchy-Unification-v1.html`
