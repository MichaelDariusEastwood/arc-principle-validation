# Collaboration Handoff

**Date:** 16 March 2026

This note is a cross-session context file for coordinated work between Claude and Codex. It is not a canonical paper, not a public report, and not a research claim in its own right. Its job is to keep the next session from re-litigating settled points.

## 1. Current Research Position

- Papers IV.a-d already exist and cover the v5 blind alignment benchmark.
- Paper V already exists and covers the Eden intervention results written so far.
- The actual unwritten paper gap is the honey/self-modifying simulation line:
  - honey architecture
  - self-modifying AI simulations
  - load-bearing wall safety/speed trade-off
  - weight-dynamics / collapse-prevention figures
- Eden v3 blind replication has **not** been run yet.
- The existing Eden intervention results are from earlier single-scorer, nonblind runs and should remain labelled `pilot` until a blinded replication exists.

## 2. Phase 0 Deliverables Completed By Codex

The bounded evidence-hardening pass is complete. The output lives in:

- [phase0_evidence_pack](/Users/michaeleastwood/arc-principle-validation/validation/phase0_evidence_pack)

Deliverables:

- [claim_evidence_ledger.json](/Users/michaeleastwood/arc-principle-validation/validation/phase0_evidence_pack/claim_evidence_ledger.json)
- [canonical_v5_alignment_table.md](/Users/michaeleastwood/arc-principle-validation/validation/phase0_evidence_pack/canonical_v5_alignment_table.md)
- [canonical_eden_intervention_table.md](/Users/michaeleastwood/arc-principle-validation/validation/phase0_evidence_pack/canonical_eden_intervention_table.md)
- [paper_writer_memo.md](/Users/michaeleastwood/arc-principle-validation/validation/phase0_evidence_pack/paper_writer_memo.md)

Engineering unblockers completed:

- [arc_paper_ii_validation_v2.py](/Users/michaeleastwood/Downloads/arc_paper_ii_validation_v2.py): syntax error fixed
- [analyze_alpha_align_v5.py](/Users/michaeleastwood/Downloads/analyze_alpha_align_v5.py): hardcoded paths removed, explicit path-driven analysis supported
- [arc_eden_v6_runner.py](/Users/michaeleastwood/Downloads/arc_eden_v6_runner.py): legacy result discovery made configurable without merging runners
- [generate_phase0_evidence_pack.py](/Users/michaeleastwood/arc-principle-validation/validation/generate_phase0_evidence_pack.py): reproducible Phase 0 pack generator

Validation completed:

- `py_compile` passes on the patched scripts and generator
- `--help` works for the patched scripts
- final Phase 0 folder contains only the four requested deliverables

## 3. Canonical Research Findings For The Next Session

These are the working truths established for collaboration:

- The v5 final JSONs contain valid blind alignment consensus scores. The dataset is not “all dead `-1`s”.
- The safest v5 headline is: alignment scaling is architecture-dependent and mixed, not universally positive.
- The Eden intervention line is still pilot-grade because it is nonblind and single-scorer.
- GPT-5.4 intervention remains an operational failure and should stay out of inferential claims.
- Honey/self-modifying work currently sits at artifact/provenance level:
  - PDFs and PNGs were located
  - standalone source scripts were recovered
  - a canonical raw-results directory now exists at [raw_results_generated](/Users/michaeleastwood/arc-principle-validation/validation/honey_provenance_pack/raw_results_generated)
  - the honey API benchmark now has a merged 6-model JSON in [eden_honey_test_results.json](/Users/michaeleastwood/arc-principle-validation/validation/honey_provenance_pack/raw_results_generated/eden_honey_test_results.json)

## 4. Agreed Division Of Labour

Use this as the default split unless the user explicitly asks otherwise.

### Claude

- Paper drafting
- voice-sensitive framing
- report updates
- broader architecture/refactoring decisions across the site or paper suite

### Codex

- evidence extraction
- engineering unblockers
- reproducibility and path/config cleanup
- canonical tables, ledgers, and supporting tooling

## 5. Agreed Boundaries

- Do **not** rewrite the published paper suite during evidence hardening.
- Do **not** merge `eden_protocol_scaling_test_v3.py` into `arc_eden_v6_runner.py` before publication work is complete.
- Do **not** redesign the website from zero.
- Do **not** treat simulation/toy results as if they are frontier-model evidence.

## 6. Website Context From Claude Session

This website context comes from the parallel Claude session and is recorded here for continuity. It was not the focus of the Phase 0 evidence pass.

Working position:

- Keep the current homepage redesign foundation.
- Do not bin it and do not rebuild from scratch.
- Limit further changes to factual hardening and maintainability:
  - align structured data dates with visible dates
  - remove or qualify risky unsourced claims
  - remove stale rating/review counts unless properly wired
  - reduce repeated inline styling where it is clearly duplicated
  - prefer tightening over “cinematic” scope creep

In short:

- Claude’s role on the website: architecture, coherence, voice, restrained editing
- Codex’s safe role on the website: factual hardening, metadata consistency, maintainability cleanup, targeted engineering fixes

## 7. Recommended Next Sequence

1. Claude uses [paper_writer_memo.md](/Users/michaeleastwood/arc-principle-validation/validation/phase0_evidence_pack/paper_writer_memo.md) and the canonical tables to harden the claims already made in Papers IV.a-d and V.
2. The next genuinely missing paper is the honey/self-modifying simulation paper.
3. A separate pass should locate or reconstruct the raw data/scripts behind the honey/self-modifying figures before those results are treated as stronger than artifact-level evidence.
4. Eden v3 blind replication should be run later as a credibility upgrade for Paper V, not retroactively merged into the existing evidence line.
5. Only after paper work is stabilised should runner consolidation be considered.

## 7A. Honey Provenance Pack

- Provenance/readiness pack created at [honey_provenance_pack](/Users/michaeleastwood/arc-principle-validation/validation/honey_provenance_pack).
- Recovered source fragments now exist for honey simulation, honey tests, honey dashboard, and self-modifying v1-v4.
- Artifacts were mapped to `text.txt` line ranges.
- Standalone honey/self-mod source files were recovered in Downloads.
- Generated JSON outputs now exist for honey simulation, self-modifying v1-v4, and the honey API benchmark in [raw_results_generated](/Users/michaeleastwood/arc-principle-validation/validation/honey_provenance_pack/raw_results_generated).
- The canonical honey API benchmark file is [eden_honey_test_results.json](/Users/michaeleastwood/arc-principle-validation/validation/honey_provenance_pack/raw_results_generated/eden_honey_test_results.json), merged across `claude`, `deepseek`, `qwen3`, `gpt54`, `gemini`, and `grok`.
- Preserved batch-specific files also exist:
  - [eden_honey_test_results_main4.json](/Users/michaeleastwood/arc-principle-validation/validation/honey_provenance_pack/raw_results_generated/eden_honey_test_results_main4.json)
  - [eden_honey_test_results_gemini_grok.json](/Users/michaeleastwood/arc-principle-validation/validation/honey_provenance_pack/raw_results_generated/eden_honey_test_results_gemini_grok.json)

## 8. One-Line Summary

The v5 and Eden paper lines already exist; Codex has now hardened the evidence pack underneath them and recovered a real 6-model honey API results file. The next real writing gap is the honey/self-modifying simulation paper, and the next real experimental gap is a blinded Eden v3 replication.
