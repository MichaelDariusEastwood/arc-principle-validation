# Cauchy unification: the experiment home of Paper VII

The scripts, data and recorded outputs behind **Paper VII: Cauchy Unification**, as they were run in March 2026.

## Read this first: this folder is under correction

The runs here are recorded as they were made, and their recorded outputs stand. The reading first placed on them does not: the paper has corrected it. The functional-equation grid has four cells rather than three, and the primary statistic is a permutation test conditioned on both marginals rather than the binomial reported at the time.

Several recorded outputs, script docstrings and file names in this folder still carry the earlier counts, probabilities and labels. They are kept unchanged as the record of what was run. A file name is the script's own, as first published, and is not a claim about the result.

**The paper's current edition states the result and its status:** [`papers/Paper-VII-Cauchy-Unification/`](../../papers/Paper-VII-Cauchy-Unification/). Where anything in this folder differs from it, the paper is right. This README states no result. The [corrections notice](https://www.michaeldariuseastwood.com/research/corrections/) is authoritative where a result has been corrected or withdrawn.

The paper's abstract gives the standing of the comparison in these words (version 3.19, 19 November 2026): "The predictions were not timestamped in advance of their own fits and the operators were classified by the author, so the result is not proven; a replication with independent classification and a four-family test are preregistered by timestamp and await submission."

## What the experiment asks

Does the composition operator of a process constrain the form of its scaling law? The original comparison covered 20 domains. That legacy script is still present for provenance; it uses a permissive `R² within 0.05` confirmation rule and is not the primary comparison.

The primary suite is:

- `scripts/arc_50_domain_universal_test.py`
- `data/canonical_50_domain_manifest.json`

Its design changes are:

- manifest-backed predictions and datasets
- strict family matching with no tolerance rescue
- actual hyperbolic fitting for the muscle domain
- explicit evidence tiers instead of one flat blended score

## Follow-up assets

This folder also holds the material prepared for a later replication of the 25-domain empirical cohort:

- a neutralised operator-classification packet and response template for an assessor outside the programme
- the archived 12-domain locked extension packet and its local dry-run execution
- null-control and negative-control scripts with their recorded outputs
- miss-analysis outputs that keep every miss as a miss
- a cross-library replication scaffold in R, with a status note recording that it is blocked locally (`Rscript` not installed)

The first 12-domain packet in this folder was exercised locally before any OSF timestamp, so it is a locked pilot extension and not a preregistered confirmation batch.

Two written records in `results/` were made inside the programme and are internal checks, not independent ones: `independent_operator_classification.md` and `miss_analysis.md`. Each carries a note at its head that says so. No assessor outside the programme has classified these domains.

## Scripts

All scripts are in `scripts/`. They require Python 3 with NumPy and SciPy.

| Script | Description | Role |
|--------|-------------|------|
| `arc_50_domain_universal_test.py` | **Primary.** Tiered 50-domain harness with manifest-backed inputs, a strict empirical endpoint and separate evidence tiers. | Primary comparison |
| `arc_20_domain_universal_test.py` | **Legacy.** 20-domain structured prediction comparison retained for provenance. Uses permissive legacy scoring and is not the primary comparison. | Original Paper VII empirical comparison |
| `build_operator_classification_packet.py` | Generates a neutralised operator-classification packet, response template and instructions that remove model-name hints from the 25 empirical domains. | Follow-up |
| `analyze_empirical_misses.py` | Generates machine-readable miss-analysis outputs that keep each miss as a miss. | Follow-up |
| `run_null_controls.py` | Runs family-label and shuffled-data null controls against the 25-domain empirical cohort. | Follow-up |
| `negative_controls.py` | Runs scrambled-data, random monotone, axiom-violating and bootstrap-stability controls. | Follow-up |
| `temporal_out_of_sample.py` | Fits candidate forms on an earlier data window and checks whether the same functional family is selected on a later window, in five domains. | Follow-up |
| `cross_validate_fits.R` | Repeats the empirical curve-fitting step in R to check the Python/SciPy winner selections once an R runtime is available. | Follow-up |
| `run_preregistered_extension.py` | Replays the first locked 12-domain extension packet against locally extracted data. Useful for audit and sourcing, but noncanonical because execution preceded any OSF timestamp. | Locked pilot dry run |
| `arc_complete_test_suite.py` | Six tests in one script, among them a numerical check of Cauchy's functional equations. Its docstring counts three scaling forms, the earlier reading; the paper gives four families. | Supporting: numerical checks |
| `arc_unified_paradigm_test.py` | Classifies fifteen test cases into regimes by Cauchy's functional equations. | Supporting: regime classification |
| `arc_rigorous_validation.py` | Checks Cauchy's functional equations numerically, then fits eighteen domains with three two-parameter functions under strict matching. | Supporting: numerical checks |
| `arc_universal_proof.py` | Works through an argument from Cauchy's functional equations and maximum entropy to an exponent formula. The file name is the script's own; the paper states what the argument does and does not establish. | Supporting: theoretical derivation |

## Results

Recorded outputs are in `results/`. Each corresponds to a script above unless the table says otherwise:

| Results file | Source |
|--------------|--------|
| `results_50_domain_validation.txt` | `arc_50_domain_universal_test.py` |
| `results_50_domain_validation.json` | `arc_50_domain_universal_test.py` |
| `results_20_domain_validation.txt` | `arc_20_domain_universal_test.py` |
| `independent_operator_classification.md` | A second classification of the 25 empirical domains, made inside the programme on 17 March 2026: an internal check, not an independent assessment (see the note at its head) |
| `blinded_operator_classification_instructions.md` | Instructions for a classifier outside the programme, written by `build_operator_classification_packet.py` |
| `empirical_miss_analysis.md` | Miss-analysis report, from `analyze_empirical_misses.py` |
| `empirical_miss_analysis.json` | Miss-analysis structured output |
| `miss_analysis.md` | A written review of each miss, made inside the programme: an internal review (see the note at its head) |
| `null_control_results.md` | Family-label and shuffled-data null-control summary |
| `null_control_results.json` | Family-label and shuffled-data null-control structured output |
| `negative_control_results.json`, `negative_control_results.txt` | `negative_controls.py` |
| `results_temporal_out_of_sample.json` | `temporal_out_of_sample.py` |
| `cross_library_replication_status.md` | R replication blocker / readiness note |
| `results_preregistered_extension.txt` | First locked 12-domain extension dry run (pilot-only; not preregistered) |
| `results_preregistered_extension.json` | Structured output for the same pilot-only dry run |
| `results_complete_suite.txt` | `arc_complete_test_suite.py` |
| `results_arc_unified_paradigm_test.txt` | `arc_unified_paradigm_test.py` |
| `results_arc_rigorous_validation.txt` | `arc_rigorous_validation.py` |
| `results_arc_universal_proof.txt` | `arc_universal_proof.py` |

`results/README.md` explains the reader copies of the two archived result tables. The repository's [`EXPERIMENTS-INDEX.md`](../../EXPERIMENTS-INDEX.md) lists every file of this folder and says what each one is.

## Data and preregistration assets

| File | Purpose |
|------|---------|
| `data/canonical_50_domain_manifest.json` | The manifest of the 50-domain tiered suite |
| `data/blinded_operator_classification_packet.json` | Neutralised 25-domain packet for an operator-classification reviewer outside the programme |
| `data/blinded_operator_classification_template.json` | Response template for that packet |
| `preregistration/next_extension_protocol.md` | Archived draft for the first 12-domain extension packet; now marked as exercised locally and not valid as a prospective preregistration |
| `preregistration/next_extension_manifest.json` | Locked local candidate list used by that exercised 12-domain extension dry run |
| `preregistration/osf_component_registration.md` | Archived OSF wrapper for the first packet, now marked not for unchanged upload as a preregistration |
| `preregistration/osf_attachment_manifest.json` | Attachment inventory for the archived first-packet bundle |
| `preregistration/file_checksums.txt` | SHA-256 checksums for the archived packet files |
| `preregistration/extracted_data/README.md` | Notes for the extracted-data folder, which holds the exercised local extension datasets |
| `preregistration/extracted_data/extension_data.json` | Extracted datasets used by the exercised local 12-domain extension dry run |

## How to run

```bash
# The 50-domain tiered suite (primary)
python3 scripts/arc_50_domain_universal_test.py

# Legacy 20-domain comparison retained for provenance
python3 scripts/arc_20_domain_universal_test.py

# Generate the neutralised operator-classification packet
python3 scripts/build_operator_classification_packet.py

# Generate the miss-analysis outputs
python3 scripts/analyze_empirical_misses.py

# Run null controls for the 25 empirical domains
python3 scripts/run_null_controls.py

# Replay the first locked 12-domain extension dry run (pilot-only)
python3 scripts/run_preregistered_extension.py

# Supporting scripts
python3 scripts/arc_complete_test_suite.py
python3 scripts/arc_unified_paradigm_test.py
python3 scripts/arc_rigorous_validation.py
python3 scripts/arc_universal_proof.py
```

## Provenance

Five of these scripts are also kept, byte for byte, in `domain-validation__Foundational-and-Origin/scripts/`: `arc_20_domain_universal_test.py`, `arc_complete_test_suite.py`, `arc_rigorous_validation.py`, `arc_unified_paradigm_test.py` and `arc_universal_proof.py`. They are collected here with the rest of the paper's scripts.

## Evidence tier

Under correction, as [`experiments/README.md`](../README.md) gives it: the recorded outputs stand; the reading placed on them does not. No tier in that file's table means validated, replicated or independently confirmed.

## Paper

Paper VII: Cauchy Unification, located at:
`papers/Paper-VII-Cauchy-Unification/`
