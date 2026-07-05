# T2-research — SESSION MASTER FINDINGS INDEX (2026-07-05)

**One prioritised, deduplicated action list for all findings surfaced this session.** Supersedes the individual inbox files as the single tracking surface. All finding docs live in `arc/experiments/`; all paper edits are on branch `pr/t2-research/paper-vi-falsifiability-20260705T0331Z` (HOLD for T5 GREEN). Guard: canonical convergence count held at **19** throughout; `restore-29` never entered main.

## HIGH — decision required (owner: T4-research / PM / operator)
| # | Finding | Objective | Status | Owner | Action | Doc/commit |
|---|---|---|---|---|---|---|
| H1 | **Convergence-19 internal independence** — row 13 (Pope encyclical) & row 14 (Olah co-presents SAME encyclical) = double-count risk (19→18); 2/19 are class=PREDICTION not realised convergences | B | surfaced, NOT changed (guarded) | T4/PM/operator | Justify row-14 independence explicitly OR merge+re-baseline; footnote predictions OR report "17 convergences + 2 predictions" | CONVERGENCE-19-INDEPENDENCE… / 4ab9c53 |

## MEDIUM — ratify/fix (owner mixed)
| # | Finding | Obj | Status | Owner | Action | Doc/commit |
|---|---|---|---|---|---|---|
| M1 | **MToC count 18 vs canonical 22** — MToC uses stale IV-as-1 taxonomy; canon says 22 (IV-as-4 + 2 meta). Full enumeration provided | A | reconciled, ratifiable | T4/operator | Ratify taxonomy → MToC total 22 + explicit composition line; label "11 empirical/8 papers" as subset | MTOC-COUNT-RECONCILIATION / 66721a9 |
| M2 | **Licence contradiction** — LICENSE-PAPERS.md=CC BY-NC-ND but 9 papers say "All Rights Reserved"; 0/22 carry CC notice inline | G | flagged (rights decision) | operator/T4 | Confirm CC BY-NC-ND intent → replace ARR + add uniform CC notice to all 22 (post-merge) | LICENCE-CONTRADICTION… / 187fd93 |
| M3 | **LTFF interview-prep** states retracted α≈2.2 as current, no retraction in doc | C | flagged | grants lane/operator | Reframe directional + add α=2.24→0.49 note (LTFF SEND-PACKET already has it) | OBJECTIVE-C-GRANT… / 7f29f4b |
| M4 | **Reproducibility uneven** — 6/10 experiment dirs lack a dir-level reproduce-doc; Paper-VII is the template | D | flagged (author owns run-knowledge) | operator/author | Add REPRODUCE.md (data→cmd→seed→output) to the 6; expand OPEN-CORE-REPRODUCIBILITY to index all 10 | OBJECTIVE-D-REPRODUCIBILITY… / 61d2470 |

| L5 | **Paper XI falsifiability** — was MISSING (central convergence paper, apophenia-exposed); §4.1 ADDED grounded in §4 criteria as defeat conditions | A/#4 | **FIXED (edit applied, hold-for-T5)** | T5 verify | Confirm reads clean + matches §4; finalise WITH H1 | LADDER4-PAPER-XI… / see commit |

## LOW / paper-edit HOLD-for-T5 (owner: T5 verify, then merge)
| # | Finding | Obj | Status | Owner | Action | Doc/commit |
|---|---|---|---|---|---|---|
| L1 | **R²=1.00000000** framing — Paper III EDIT applied (now "exact analytical identity, not empirical fit", matching Foundational) | A/#7 | EDIT made, hold-for-T5 | T5 | Verify edit + identity claim; then merge | LADDER7-R2-IDENTITY… / effbd0d |
| L2 | **ARC-Bound retirement xref** — On-the-Origin lists α≤2 as current w/o retirement note (only lag of 9) | A | **FIXED (edit applied, hold-for-T5)** | T5 verify | Paper-III-style qualifier ADDED to On-the-Origin | ARC-BOUND-RETIREMENT-XREF / cba8f9c |
| L3 | **F23 metabolic** — "2.4 vs 2.5" NOT a contradiction (different sets); full table-mean recompute needed | A/#2 | refined, routed | T5 | Parse full tables from source; confirm prose-mean=table-mean; add set-scope line | F23-METABOLIC-REFINED / cd6c4e5 |
| L4 | **Citation real-existence** spot-check (year-consistency already CLEAN) | A/#7 | deferred (LAW 5) | T5 online | Online-verify top ~10 authorities exist as cited | LADDER7-CITATION… / 14581a3 |

## Objective B truth-gate → zero (owner: PM/T4 coordinated)
| # | Finding | Status | Action |
|---|---|---|---|
| B1 | **Product is CLEAN** — arc papers 0 violations; public served pages effectively 0; the ~290 is internal working-docs | verified | Reframe the "290" honestly — product surfaces pass NOW |
| B2 | Gate tuning (biggest lever) | specified | Exclude/allowlist defect-DESCRIBING docs (finding docs quote forbidden phrasings to fix them) |
| B3 | Gate CHECKS patch | specified, uncommitted (needs clean T2 apps-script checkout) | Add "25" + "confirmations" to count-drift regex; broaden unqualified-78 to 40-char RL window |
| B4 | Count-drift relabel ~25 internal docs | post-merge | Context-sensitive (blind 28→19 corrupts quoted-as-wrong) |
| B5 | 2 loose-dkim working docs | minor | "DKIM-verified" → "…in transit" |

## VERIFIED CLEAN this session (positives — LAW 1)
- **Every paper has PDF+DOI (22/22)**; SHA-256 manifest now published + self-verified (23/23 OK) — G1 / b541f66.
- **DOI scheme sound** (6C5XB=project/book DOI, papers=components; Paper II 8FJMA anomaly benign).
- **221 public blog articles clean** on all 4 number vectors (count-drift/super-linear/78%/19).
- **Grant estate (~25 funders) clean** on count-drift + super-linear framing + £624k, bar M3.
- **Trilogy (HRIH/Vision/Engineering+PNP) publish-grade** + consistent + provenance-complete.
- **Citations year-consistent** (apparent inconsistencies = prolific authors + DOI false-match).
- **Equations-by-artifact consistent** — R (Dec-8 thesis) / R^α (general, α=1/(1−β)) / R² (α=2 info-theoretic CEILING, ARC Bound); Foundational explicitly frames R² as "an information-theoretic upper bound" not a rival to R^α; NO paper poses "R² or R^α". Canon rule (never present R²/R as competing) HOLDS.
- **ISBN estate clean** — one programme ISBN 9781806056200 (checksum-valid) consistent across 17 papers; PNP two extra ISBNs = legit external citations (Pattern Seekers 9781541647145, Jessica Kingsley 9781843103882), all checksum-valid.

## Guard status
origin/main convergence count = **19** (held all session). `restore-29` NOT in main ancestry. Papers truth-gate-clean. Re-flag immediately if either changes.
