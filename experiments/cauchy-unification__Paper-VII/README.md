# Cauchy Unification -- Paper VII

Experimental scripts and results supporting **Paper VII: The Cauchy Unification (v2)**.

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

## Follow-up hardening assets

This folder now also contains the next-step strengthening package for the
canonical 25-domain empirical cohort:

- blinded operator-classification packet and response template for an external
  assessor
- archived 12-domain locked extension packet plus its local dry-run execution
- null/negative-control harness for shuffled-data checks
- conservative miss-analysis outputs that keep the current six misses as misses
- cross-library replication scaffold in R, plus a status note documenting the
  current local blocker (`Rscript` not installed)

These hardening assets are meant to support the **next genuinely preregistered
replication**. The first 12-domain packet in this folder was already exercised
locally before any OSF timestamp, so it must be treated as a locked pilot
extension rather than a preregistered confirmation batch.

The currently saved null-control run is still a lightweight first pass:

- family-label null: `10,000` trials
- shuffled-y null: `20` iterations

That is enough to show the broken-structure controls do not reproduce
`19/25`, but not enough to call the null surface fully converged.

## Scripts

All scripts are in `scripts/`. They require Python 3 with NumPy and SciPy.

| Script | Description | Role |
|--------|-------------|------|
| `arc_50_domain_universal_test.py` | **Primary.** Tiered 50-domain validation harness with manifest-backed inputs, strict empirical endpoint, and explicit evidence tiers. | Current canonical experiment |
| `arc_20_domain_universal_test.py` | **Legacy.** 20-domain structured prediction comparison retained for provenance. Uses permissive legacy scoring and should not be treated as the clean canonical headline. | Original Paper VII empirical comparison |
| `build_operator_classification_packet.py` | Generates a neutralized operator-classification packet, response template, and instructions that remove model-name hints from the 25 empirical domains. | Follow-up hardening |
| `analyze_empirical_misses.py` | Generates conservative machine-readable miss-analysis outputs for the six current empirical misses. | Follow-up hardening |
| `run_null_controls.py` | Runs family-label and shuffled-data null controls against the canonical 25-domain empirical cohort. | Follow-up hardening |
| `cross_validate_fits.R` | Repeats the empirical curve-fitting step in R to check the Python/SciPy winner selections once an R runtime is available. | Follow-up hardening |
| `run_preregistered_extension.py` | Replays the first locked 12-domain extension packet against locally extracted data. Useful for audit and sourcing, but noncanonical because execution preceded any OSF timestamp. | Locked pilot dry run |
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
| `independent_operator_classification.md` | Existing narrative independent-classification result |
| `blinded_operator_classification_instructions.md` | Instructions for a genuinely blinded external classifier |
| `empirical_miss_analysis.md` | Conservative miss-analysis report |
| `empirical_miss_analysis.json` | Conservative miss-analysis structured output |
| `null_control_results.md` | Family-label and shuffled-data null-control summary |
| `null_control_results.json` | Family-label and shuffled-data null-control structured output |
| `cross_library_replication_status.md` | R replication blocker / readiness note |
| `results_preregistered_extension.txt` | First locked 12-domain extension dry run (pilot-only; not preregistered) |
| `results_preregistered_extension.json` | Structured output for the same pilot-only dry run |
| `results_complete_suite.txt` | `arc_complete_test_suite.py` |
| `results_arc_unified_paradigm_test.txt` | `arc_unified_paradigm_test.py` |
| `results_arc_rigorous_validation.txt` | `arc_rigorous_validation.py` |
| `results_arc_universal_proof.txt` | `arc_universal_proof.py` |

## Data and preregistration assets

| File | Purpose |
|------|---------|
| `data/canonical_50_domain_manifest.json` | Canonical manifest for the 50-domain tiered suite |
| `data/blinded_operator_classification_packet.json` | Neutralized 25-domain packet for an external operator-classification reviewer |
| `data/blinded_operator_classification_template.json` | Response template for the blinded packet |
| `preregistration/next_extension_protocol.md` | Archived draft for the first 12-domain extension packet; now marked as exercised locally and not valid as a prospective preregistration |
| `preregistration/next_extension_manifest.json` | Locked local candidate list used by that exercised 12-domain extension dry run |
| `preregistration/osf_component_registration.md` | Archived OSF wrapper for the first packet, now marked not for unchanged upload as a preregistration |
| `preregistration/osf_attachment_manifest.json` | Attachment inventory for the archived first-packet bundle |
| `preregistration/file_checksums.txt` | SHA-256 checksums for the archived packet files |
| `preregistration/extracted_data/README.md` | Notes for the extracted-data folder, which now contains the exercised local extension datasets |
| `preregistration/extracted_data/extension_data.json` | Extracted datasets used by the exercised local 12-domain extension dry run |

## How to run

```bash
# Current canonical tiered suite
python3 scripts/arc_50_domain_universal_test.py

# Legacy 20-domain exploratory comparison retained for provenance
python3 scripts/arc_20_domain_universal_test.py

# Generate the blinded operator-classification packet
python3 scripts/build_operator_classification_packet.py

# Generate conservative miss-analysis outputs
python3 scripts/analyze_empirical_misses.py

# Run negative controls for the 25 empirical domains
python3 scripts/run_null_controls.py

# Replay the first locked 12-domain extension dry run (pilot-only)
python3 scripts/run_preregistered_extension.py

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

Paper VII: The Cauchy Unification (v2), located at:
`papers/Paper-VII-Cauchy-Unification/`
