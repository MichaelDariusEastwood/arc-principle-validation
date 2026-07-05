# Objective G finding (MEDIUM) — licence contradiction: LICENSE-PAPERS.md says CC BY-NC-ND, 9 papers say "All Rights Reserved"

**Lane:** T2-research · **Date:** 2026-07-05 · **Route:** T4-os + operator/author (rights decision). Post-merge fix (editing papers now risks mid-merge conflict).

## The contradiction (confirmed)
- **`LICENSE-PAPERS.md`** (f28 branch) declares: "Creative Commons Attribution–NonCommercial–NoDerivatives 4.0 International (CC BY-NC-ND 4.0)" — papers are meant to be shareable (attribution, non-commercial, no-derivatives).
- **9 papers' inline notices say the OPPOSITE:** "© 2026 Michael Darius Eastwood. All Rights Reserved" — no sharing permitted. Papers: Eden-Engineering, Eden-Vision, Executive-Summary, Foundational, Master-Table-of-Contents, On-the-Origin-of-Scaling-Laws, Paper-II, Paper-III, Paper-V.
- **0/22 papers carry the CC BY-NC-ND notice inline** — so a standalone PDF conveys either just CDPA-1988 copyright (22/22) or an actively contradictory "All Rights Reserved".

## Why it matters (Objective G + LAW 3)
Objective G = "dual licence (papers CC BY-NC-ND / code proprietary) EVERYWHERE". A repo-level LICENSE file does not travel with an individual PDF; a reader who downloads one paper sees "All Rights Reserved" and concludes they cannot share it — contradicting the open-licence intent and undermining discoverability/reuse. A reviewer sees the corpus can't state its own licence consistently.

## Recommendation (operator/author confirms intent; then post-merge fix)
1. Confirm the intent is CC BY-NC-ND for the papers (Objective G says yes).
2. Then reconcile inline notices to match: replace the 9 "All Rights Reserved" with the CC BY-NC-ND notice, and ADD a uniform "Licensed under CC BY-NC-ND 4.0 — see LICENSE-PAPERS.md" line + link to all 22 papers (so standalone PDFs convey the licence). Keep the CDPA-1988 copyright line (compatible — copyright holder grants CC BY-NC-ND).
3. Code stays proprietary (LICENSE-CODE.md) — the dual-licence split.
This is a rights change (All-Rights-Reserved → CC BY-NC-ND grants more permissions), so it needs author confirmation, NOT a unilateral edit — flag not fix.

## Credit (LAW 1)
Every paper DOES carry a proper CDPA-1988 copyright notice (22/22) — authorship/rights are asserted. The gap is purely the CC BY-NC-ND grant not being expressed inline (and 9 papers asserting the contradictory "All Rights Reserved").
