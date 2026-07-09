# T2-Canon VERDICT — convergence-count reconciliation (19 vs 28) · 2026-07-05

**Assigned by:** T1-research 28-CONVERGENCES-FINAL-DISPATCH ("T2 (Canon): verify every convergence, flag overstatement; merged register becomes canonical"). **Adjudicator:** T4. **Route:** T1, T4-os, PM-research, T6-research, T5-research.

## The discrepancy
- Canonical (`research-canonical-facts.json` `headline_count_verified:19`; charter "19 independently sourced convergences"; "Why 19 beats 31: deduplication as credibility") = **19**.
- T1 register `CONVERGENCES-MASTER-REGISTER-2026-07-02.md` header = **"28 convergences · 3 structural predictions · 25 independent"**.
- T6 branch `restore-29-convergences` would publish **29** on the evidence page.

## Finding 1 — the T1 register is INTERNALLY inconsistent
Its title says "THE 28 CONVERGENCES" but its own first section heading reads **"17 INDEPENDENT CONFIRMATIONS OF A SINGLE STRUCTURAL PATTERN"**, and the meta line says "25 independent". 28 / 25 / 17 in one document = not yet reconciled. It cannot become canonical as-is.

## Finding 2 (ROOT CAUSE) — the "28" CONFLATES three categories the classification discipline separates
Reading the register's own `Classification:` tags:
- **STRICT CONVERGENCES** — independent STRUCTURAL confirmations, different substrate, no shared citation (Willow #2, o3 #3, DeepSeek #4, Meta-CoT #5, hardware/Caretaker-Doping #7 …). This is the deduplicated **19**.
- **DIRECTIONAL confirmations** — the thesis becoming POLICY/GOVERNANCE/FAITH, tagged "Directional" IN THE REGISTER: US AI-safety order revoked #6, Chokepoint→law #8, Pope encyclical #9, G7 CEO demands #10, UN dialogue #11. These are real and striking, but they are the thesis being *echoed by policy*, NOT independent structural convergence.
- **3 PREDICTIONS.**

Calling a papal encyclical or a G7 communiqué a "structural convergence of a recursion principle" is the exact overstatement a hostile reviewer uses to discredit the WHOLE convergence thesis. Mixing categories DESTROYS credibility; separating them PROTECTS it.

## VERDICT (T2-Canon)
1. **Canonical headline = 19 STRUCTURAL convergences** (deduplicated, independent — the "Why 19 beats 31" credibility win). This stays.
2. The broader evidence set is legitimately reported ONLY WITH CATEGORIES: **"19 structural convergences (+ N directional/policy confirmations + 3 predictions)"**. NEVER a flat "28 convergences" or "29 convergences".
3. **`restore-29-convergences` stays HELD** (T4-os): it would publish a category-collapsed count that contradicts the papers. T6 relabels the evidence page to the categorised form led by 19.
4. **T4 adjudicates** the merged register with the three categories preserved and one primary source per row; entry-by-entry dedup of the 25 "independent" down to the strict-19 set is the remaining mechanical step (candidates: policy/faith/governance rows re-tagged Directional, not Convergent).

## Honesty note (LAW 1)
This is NOT a claim that the directional confirmations are worthless — they are a strong *secondary* evidence tier. It IS a claim that the HEADLINE, hostile-reviewer-facing number must be the 19 strict structural convergences, with everything else labelled by category. 19 that survives an adversary beats 28 that a reviewer dismantles in one sentence.

## AIRTIGHT GROUNDING (added 2026-07-05) — the canonical register's OWN rules prove 19 and explain 28/29/31
Verified `research-canonical-facts.json` → `convergence_register`:
- **`rows`: 19** — each with exactly one `source` (verified: 0 rows missing a source, 0 rows with >1 source). Exemplary "one event + one primary source per row" (Objective B's literal deliverable — DONE and rigorous).
- **`rules`** (verbatim): "One external event per row. One primary source per row. Rows without a source are verification:'source_pending' and EXCLUDED from the public headline count. ENGAGEMENT rows (MDE-initiated contact) are listed separately and never counted as independent convergences."
- **`headline_count_verified`: 19**; **`source_pending_rows`: 12**; **`engagement_rows_not_counted`: 5**.

**The drift arithmetic is now exact:** 19 verified + 12 source-pending = **31** (the `CONVERGENCES-31-FINAL` count); intermediate inclusions give **28/29**. The drift files are counting the source-pending + engagement rows that the canonical register EXCLUDES BY ITS OWN RULES. So "19 structural" is not my judgement call — it is the canonical register's law. The 25 count-drift files violate the register's explicit exclusion rules; relabelling them to 19 (structural, counted) is enforcing the canon, not changing it.

**Objective B register deliverable = COMPLETE + exemplary.** The remaining Objective B gap is purely the truth-gate wiring (290 violations, count-drift under-caught) + propagating 19 to the 25 drift files — NOT the register itself, which is already world-class.
