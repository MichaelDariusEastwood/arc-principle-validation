# T2-research — A–G objective scorecard + decision/verification queue (2026-07-05)

**Purpose:** one actionable view of where every objective stands after this session, what T2 completed, and the single next action per item (with owner). Turns ~15 cycles of routed findings into one queue so nothing is forgotten. Weekly coordination deliverable for T4/operator.

## Scorecard
| Obj | State | T2 did (verified) | Open item → OWNER |
|-----|-------|-------------------|-------------------|
| **A** Papers peer-review grade | STRONG | Falsifiability 22/22 (full corpus, incl charter VII/IV-a/c/d/XI); PDF 23/23; SHA-256 23/23 verified; equation family clean; α=2.24 retraction exemplary; overstatement pass; R²=1 identity reframe | (1) held edits → T5 verify+merge; (2) per-paper DOI → OSF/operator |
| **B** Truth-gate ZERO | IN PROGRESS | Accurate map (290 viols/92 files); **executable remediation tool validated (129/44% clearable in one --apply)**; 2 gate-regex fixes specified | Run tool on clean checkout + reframe ~117 non-mechanical → clean-checkout owner; count-drift 10 after H1 |
| **C** Grants batch B | OVER-SATISFIED | Verified **25 funder apps draft-complete** (target 5); £624k ask consistent across all 25; 78% qualified in all 50 mentions | Fundability judgment + submission → operator (LAW 5) |
| **D** Book + methodology | METHODOLOGY DONE; BOOK flagged | 6/6 REPRODUCE.md grounded; **found book↔papers equation inconsistency (book U=I×R² vs papers R^α ceiling)** | (1) confirm book wording + errata → T3/operator; (2) corpus equation bridge → T4 ratify |
| **E** Trilogy + credential | TRILOGY CLEAN; CREDENTIAL flagged | Trilogy canon-consistent (all anchors); PNP clean | credential.html 19-vs-29 (F15) → gated on H1 → operator/clean-checkout |
| **F** Articles/promotion | PARTIAL | Grants + papers factually clean vs canon (verified) | public-surface sweep → T6 (SEO/nav) + factual cross-check |
| **G** Provenance + licensing | PROVENANCE VERIFIED; LICENCE flagged | verify-yourself works (23/23 SHA); LAW-3 merge-sequencing catch | 3-way licence contradiction (MIT vs All-Rights-Reserved vs charter CC-BY-NC-ND) → operator IP decision |

## The operator/T4 decision queue (each unblocks ≥1 objective)
1. **H1 — 19-vs-29 count** (governs register + credential.html/F15 + count-drift + merge guard + E). Consolidated brief delivered. → T4/operator.
2. **Book errata** — equation R² bridge + α=2.24 + development timeline (D). T3 audit + T2 equation finding delivered. → operator/T3.
3. **Licence** — 3-way contradiction (G). → operator IP call.
4. **F10** — HRIH priority framing (A). Triangulated (3 sources say concurrent). → operator.
5. **Per-paper DOI** — OSF registration (A/G). → operator/OSF.
6. **Grant fundability + submission** (C). → operator (LAW 5).

## T5 verification queue (held on branch pr/t2-research/paper-vi-falsifiability-20260705T0331Z)
8 paper edits (R²=1, ARC-Bound, overstatement, falsifiability ×6: I/II/VII/IV-a/IV-c/IV-d/XI/VIII/VI/IV-b) + 6 REPRODUCE.md. Manifest + criteria delivered. → T5 GREEN then T4-os merge (respect sync pause).

## Clean-checkout owner queue
Run `truthgate_remediate.py --apply` (129 fixes) + reframe ~117 non-mechanical + 2 gate-regex fixes (B). credential.html reconcile after H1 (E).

## Guard (standing)
origin/main `headline_count_verified` = 19 · `restore-29` OUT of main ancestry. Held every cycle.
