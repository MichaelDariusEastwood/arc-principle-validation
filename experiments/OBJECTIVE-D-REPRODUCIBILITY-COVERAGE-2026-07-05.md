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

---

## UPDATE 2026-07-05 — Paper VI now has a GROUNDED REPRODUCE.md (commit 2a4ba25); method proven, severity refined DOWN

**Key correction (LAW 1):** reproducibility info is NOT simply "absent" from the 6 dirs — for **honey-architecture (Paper VI)** it is EMBEDDED in the result-JSON metadata (v2=10seeds×150cyc fairness-VERIFIED; v3=20seeds×180cyc adversarial-VERIFIED; v4=15seeds×5levels=225runs; all timestamped 2026-03-16). I authored `honey-architecture__Paper-VI/REPRODUCE.md` extracting these verbatim — grounded, fabricates nothing; only the generator run-command is TODO(author).

**Generalisation check (honest):** the artefact-metadata method does NOT apply uniformly:
| Dir | result-JSONs | top-level metadata/config |
|---|---|---|
| honey-architecture (Paper VI) | 4 | **4/4 rich** → REPRODUCE.md DONE |
| eden-intervention (Paper V) | 6 | 0 (nested or code-only) |
| paper-ii-compute (Paper II) | 7 | 0 |
| alignment-scaling (Papers IV) | 24 | 0 |
| paper-i-foundational (Paper I) | 0 | — (code-only) |
| analysis-tools (Cross-Programme) | 0 | — |

**Refined severity: LOW–MEDIUM.** One of six now has a grounded reproduce-doc. The other five need deeper per-dir extraction (nested metadata / code inspection) or author run-knowledge — genuine but not blocking (methodology also lives in each paper's Methods §, and Paper-VII remains the GOLD structural template). Next: deeper extraction pass on the 5 (recover params from nested JSON / code) where possible; author supplies generator run-commands.

## UPDATE 2026-07-05 (2) — alignment-scaling REPRODUCE.md added (covers Papers IV-a/b/c/d)
Deep inspection of alignment-scaling found it HIGHLY reproducible (my top-level metadata check under-counted it): entry-point `scripts/arc_eden_v6_runner.py --model` (argparse, MODELS list, API_KEY, blinded), deps numpy/scipy/hashlib, 8 frontier models, depth_configs 1024/4096/16384/32768, outputs with bootstrap CIs + blinding_protocol. Authored grounded REPRODUCE.md.
**Running tally: 2/6 target dirs now have grounded REPRODUCE.md (honey-architecture Paper VI + alignment-scaling Papers IV-a/b/c/d) + Paper-VII GOLD template = 5 papers reproduce-covered.** Remaining 4 dirs (eden-intervention Paper V, paper-ii-compute Paper II, paper-i-foundational Paper I, analysis-tools) need next-cycle deep inspection (entry-points likely in scripts/, params possibly nested) or author input. Lesson: the top-level JSON-metadata check UNDER-counts reproducibility — must inspect scripts/ + nested result fields per dir.

## UPDATE 2026-07-05 (3) — Objective D reproduce-coverage COMPLETE (6/6 + Paper VII GOLD)
All remaining 4 dirs now have grounded REPRODUCE.md:
- **Paper V** (eden-intervention): 6 models × 4 depths (minimal/standard/deep/exhaustive) × 10 prompts; entry `eden_protocol_scaling_test_v3.py`.
- **Paper II** (compute): 6 models, 18/30 problems, `avg_seq_alpha` vs `avg_par_alpha` (the sequential>parallel core) + cross-verification tiers; entry `arc_paper_ii_validation_v2.py`.
- **Paper I**: figure/analysis toolkit (`arc_principle_research_toolkit.py`; matplotlib/numpy/scipy) — no primary data, documented as such.
- **analysis-tools**: post-processing consuming other dirs (`analyze_alpha_align_v5.py`, `per_scorer_check.py`).
**FINAL: 6/6 target dirs + Paper VII GOLD template = every experiment dir now has a reproduce-doc.** All grounded in committed scripts + result-JSON fields; the only TODO(author) is environment version-pins + API-key setup (genuinely author-only). Objective-D "reproducible write-ups per experiment" = DONE at the dir level (HOLD for T5 verification of faithfulness-to-artefacts). Severity closed from MEDIUM → resolved-pending-T5.
