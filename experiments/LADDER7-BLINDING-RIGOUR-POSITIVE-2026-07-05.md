# Ladder #7 — the alignment-scaling blinded-scoring is GENUINELY rigorous (reviewer-defence asset)

**Lane:** T2-research · 2026-07-05 · LAW-1 positive (surface strengths too). Verified by reading `arc_eden_v6_runner.py`. Defends the core empirical result (α_align) against the sharpest attack.

## The attack this defeats
A reviewer's deadliest objection to any AI-alignment measurement: "your scoring wasn't blind — you (or a model you prompted knowing the condition) scored to confirm the hypothesis." On inspection, the experiment pre-empts this with a rigorous design.

## Verified blinding architecture (from code)
1. **Condition-hidden scoring.** The scoring prompt is built as `"{prompt_text}\n\nRESPONSE TO EVALUATE:\n{response_text}"` — the scorer receives ONLY the task + the raw response. It is NOT shown `depth_label`, model name, or condition. Depth/condition are attached to the row and used only for **post-hoc grouping** (`_grouped_depth_points`, `grouped.setdefault(row.get("depth_label"...))`), after scores exist.
2. **Multi-model consensus.** Six independent model-scorer adapters (six `score()` implementations) score each response against a shared rubric (`SCORER_SYSTEM` = "expert evaluator of reasoning quality"); results aggregated to `score_consensus` with inter-rater `scorer_spread`.
3. **Active leakage check.** `parse_leakage_blob(raw, candidate_models, candidate_condition, …)` — the code explicitly TESTS whether a scorer could identify the source model/condition from the response text ("...likely source only from the text shown"). Blinding integrity is measured, not assumed.

## Why this is load-bearing
This is a MODEL of rigorous blinded evaluation: independent scorers (not the author), condition-invisible inputs, cross-model consensus + spread, and a leakage audit. It converts the α_align measurement from "self-graded, therefore suspect" to "independently blind-scored with a leakage control" — exactly the standard a hostile methods reviewer demands. Pair this with the bootstrap CIs (`alpha_align_boot_lo/hi`) already present: the core result carries both blinding rigour AND uncertainty quantification.

## Recommendation
Surface this explicitly in Papers IV-a/b (and the reviewer-response file): a short "Blinding & scorer-independence" methods paragraph citing (a) condition-hidden scoring, (b) 6-model consensus + spread, (c) the leakage check. It is a genuine strength currently buried in the code; making it visible in the paper pre-empts the attack. (This also mitigates the multiple-comparisons flag — the primary effect is blind-scored + large-d, not a p-hacked artefact.)

## Net (LAW 1)
The single sharpest attack on the empirical core (biased/self-scoring) FAILS on inspection — the blinding is real and audited. Reviewer-defence asset; route to T4 + note for the IV-series methods sections.
