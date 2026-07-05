# Objective D — experiment reproducibility coverage (uneven; Paper-VII is the template)

**Lane:** T2-research · 2026-07-05 · Factual inventory (what exists), not invented run-steps. Route: T4-research + operator (author owns experiment run-knowledge). Ladder #6 (methodology not canon-clean).

## Coverage (10 experiment dirs — verified)
| Experiment dir | code | data | dir-level write-up |
|---|---|---|---|
| cauchy-unification__Paper-VII | ✓ | ✓ | **GOLD** — preregistration/ + README.md + results/ (8 analyses: null-control, miss-analysis, cross-library replication, blinded operator classification) |
| blind-prediction-test__Foundational-and-Origin | ✓ | - | BLIND_TEST_FORENSIC_ANALYSIS.md |
| blind-prediction-test__Paper-III | ✓ | - | BLIND_TEST_FORENSIC_ANALYSIS.md |
| domain-validation__Foundational-and-Origin | ✓ | ✓ | phase0_evidence_pack/ (canonical tables + paper_writer_memo) |
| alignment-scaling__Papers-IV-a-b-c-d | ✓ | ✓ | **none** |
| analysis-tools__Cross-Programme | ✓ | - | **none** |
| eden-intervention__Paper-V | ✓ | ✓ | **none** |
| honey-architecture__Paper-VI | ✓ | ✓ | **none** |
| paper-i-foundational__Paper-I | ✓ | - | **none** |
| paper-ii-compute__Paper-II | ✓ | ✓ | **none** |

Central `OPEN-CORE-REPRODUCIBILITY.md` = 38 lines, references only alignment-scaling + eden-intervention (2/10).

## Honest severity (LAW 1)
NOT "9/10 irreproducible". Every dir has code; methodology also lives in each paper's Methods section; several dirs have rich write-ups. The gap is a UNIFORM dir-level "how to reproduce THIS result" doc (data path → entry-point → seed → expected output) — present for Paper-VII, absent for 6. For a reviewer who wants to RUN it, this unevenness is a real friction and a discount vector. MEDIUM.

## Fix (author owns run-knowledge — flag not fabricate)
1. Adopt Paper-VII's structure as the template for the 6 without a write-up: a short REPRODUCE.md per dir (data → command → seed → expected output/figure).
2. Expand OPEN-CORE-REPRODUCIBILITY.md (38 lines) to index all 10 experiments with a one-line run-pointer each.
3. Cross-links to the paper Data & Code Availability sections (only Papers II/VI/VII have one — see STATISTICAL_METHODOLOGY_FINDINGS_2026-07-05 §data-availability). Standardise both together.
I do NOT author the run-steps here — I cannot verify a seed/entry-point/expected-output I did not run (would violate LAW 1). The author/experiment-owner supplies them; I have scoped exactly which 6 dirs + the template.
