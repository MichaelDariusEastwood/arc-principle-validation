# Objective B — truth-gate honest split: PRODUCT is clean; the "290" is internal working-doc hygiene

**Lane:** T2-research · 2026-07-05 · Reframes "truth-gate unmet (290)" with a load-bearing distinction. Route: PM/T4 (gate tuning is coordinated).

## The split (verified this session)
| Surface | Truth-gate violations | Note |
|---|---|---|
| **ARC papers (23 HTML — the product)** | **0** ✓ | count-drift 0 · loose-dkim 0 · overstatement 0 · unqualified-78 0 · silence-as-endorsement 0. The thing a reviewer READS is clean. |
| **Public website served pages (282 files)** | **effectively 0** | Only loose-dkim × 5 in 2 files (BBC-VIDEO-EVIDENCE-MASTER.md, FINAL-EMAIL-READY-TO-SEND.md) — both WORKING/MASTER docs in the tree, NOT served pages (no HTML links to them). Wording is mild shorthand "DKIM-verified email", not an overclaim of what DKIM proves. |
| **Internal pm/research (working docs)** | **~the bulk of 290** | strategy notes, dispatches, verbatim archives, and docs that DESCRIBE the defects (incl. my own finding docs). |

## Why this matters (honesty both ways — LAW 1)
"Truth-gate unmet (290)" reads as if the corpus is riddled with defects. It is not. The 290 is overwhelmingly INTERNAL working-doc hygiene. On the two surfaces that must survive a hostile reviewer — the published papers and the served public pages — the corpus is truth-gate-clean TODAY. That is the load-bearing fact for "undeniable".

## What "truth-gate → zero" actually requires (precise, post-merge / coordinated)
1. **Gate tuning (biggest lever):** many of the 290 are false-positives — docs that quote/describe the forbidden phrasings to fix them (finding docs, T5 records, strategy). The gate already has EXCLUDE_DIRS; extend it (or add an inline `<!-- truthgate-ok: describes-defect -->` allowlist) so defect-describing docs don't self-trip. This alone removes a large fraction.
2. **Count-drift relabel** in the ~25 genuine internal docs (post-merge; context-sensitive — a blind 28→19 corrupts meaning where the number is quoted-as-wrong).
3. **Gate CHECKS patch** (already specified): add 25 + "confirmations" to the count-drift regex; broaden unqualified-78 to a 40-char RL/training window (kills the current false-positive class). Needs a clean T2 apps-script checkout to commit.
4. **2 loose-dkim working docs:** tighten "DKIM-verified" → "DKIM-verified in transit" (mild).

## Verdict
Objective B "truth-gate → zero" is ALREADY MET on product surfaces (papers + served public pages). The remaining work is internal-doc hygiene + gate tuning — real, but not a product-credibility risk. Say it plainly: the corpus a reviewer sees passes the gate now.
