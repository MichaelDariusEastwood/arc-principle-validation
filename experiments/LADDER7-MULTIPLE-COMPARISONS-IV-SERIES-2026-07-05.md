# Ladder #7 — multiple-comparison correction absent in the p-value-heavy IV-series (headline robust, marginals vulnerable)

**Lane:** T2-research · 2026-07-05 · Reviewer-facing statistical-rigour hardening. Flag + specific fix → T5/author (substantive stats — do not unilaterally recompute corrections without the raw test details).

## Finding (LAW 1, bounded)
**Paper IV-a** reports **24 p-values** across multiple models/tests with **no explicit multiple-comparison correction** (Bonferroni / Holm / FDR / Benjamini-Hochberg all absent) and **no primary/confirmatory-vs-exploratory pre-specification** ("secondary" appears 2× but no primary-hypothesis framing; no "uncorrected/nominal-p/exploratory" caveat). Same pattern in **Paper IV-b** (p=0.000001, 0.006, 0.007…). A statistically-literate reviewer attacks this: 24 tests at α=0.05 → ~1.2 expected false positives uncorrected.

## Why the CORE claim survives (don't overstate the problem)
The headline results are ROBUST to correction: e.g. "Grok 4.1 Fast, **d = +1.38**, p < 0.000001; Claude Opus 4.6, **d = +1.27**, p = 0.000001" — **large effect sizes** (Cohen's d > 1.2) with p ≤ 1e-6. Bonferroni-corrected (×24) these are still ≪ 0.05. The central result is not at risk.

## What IS vulnerable
The **marginal** p-values (p = 0.02, 0.037, 0.065, 0.08) — several already ≥ 0.05, and the p=0.02/0.037 would fail Bonferroni (threshold 0.05/24 ≈ 0.002). Presented without a correction caveat, they read as "significant" when they would not survive family-wise correction.

## Recommended fix (T5/author — modest, pre-empts the attack)
1. Add a **multiple-comparison statement**: report that the headline effects (d > 1.2, p ≤ 1e-6) survive Bonferroni/FDR across the N tests; give the corrected threshold.
2. **Label the marginal results** (p = 0.02–0.08) as exploratory, OR apply FDR (Benjamini-Hochberg) and report which survive.
3. If there was a **pre-registered primary hypothesis**, state it explicitly — a single confirmatory test needs no correction; the rest are then honestly secondary/exploratory.
This is presentation rigour, not a result change. The paper's strength (large effects, tiny primary p) is real; the fix removes a free shot for a reviewer.

## Scope
Paper IV-a (24 p-values), Paper IV-b (same pattern); spot-check IV-c/IV-d + Paper VI (p=0.037 present) for the same. Route T5/author.
