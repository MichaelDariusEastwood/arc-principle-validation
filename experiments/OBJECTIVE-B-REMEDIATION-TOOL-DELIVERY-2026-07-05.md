# Objective B — executable remediation tool delivered (fixes 44% of truth-gate violations in one command)

**Lane:** T2-research · 2026-07-05 · Tool: `experiments/truthgate_remediate.py` (lives in arc repo; owner runs on a CLEAN apps-script checkout). Dry-run VALIDATED against the live gate.

## What it does (validated)
Conservative, **dry-run by default**. Targets the dominant mechanical category:
- **unqualified-78** → adds "in the RL-training condition (12% baseline)" qualifier, but ONLY where the sentence is Anthropic/alignment/RL-context AND not already qualified.

**Validation (LAW 1 — verified, not asserted):** dry-run proposes **129 edits across 63 files** — this EXACTLY matches the gate's 129 *genuine* unqualified-78 (the 144 gate hits minus the 15 regex false-positives). The tool independently reproduces the gate's genuine set. That is 129/290 = **~44% of all current truth-gate violations fixable in one `--apply`**.

## Honest scope (LAW 1)
- The **book-date** rule fires **0** in dry-run: its conservative sentence-check declines the context-dependent "Jan 6" cases (the correct fix depends on whether each is a mis-stated publication date or a legitimate arrival mention). Those 10–12 stay **manual** — the tool does not guess.
- The remaining ~117 violations (loose-dkim reframes, forbidden-framing, garbled-valuation, count-drift-after-H1, etc.) are NOT mechanical and are NOT touched — they need human reframing per the remediation map.
- LAW 3: touches prose only — no slugs/DOIs/filenames/SHAs. Reviewable diff before any commit.

## How the owner runs it
```
# on a CLEAN apps-script checkout (not the churny shared one):
python3 truthgate_remediate.py pm/research            # dry-run — review the 129 proposed edits
python3 truthgate_remediate.py pm/research --apply     # write
git diff                                                # REVIEW before commit
python3 tools/research_truth_gate.py pm/research        # expect ~129 fewer violations (290 -> ~161)
```
Then apply the 2 gate-regex refinements (from the remediation map) to clear the 15 false-positives, leaving the genuine non-mechanical ~117 for human reframing.

## Net
Objective B advanced from a map to an **executable, validated** fix: 44% of violations clearable in one reviewed command; the rest honestly scoped as manual. Route T4 + clean-checkout owner + gate-file owner.
