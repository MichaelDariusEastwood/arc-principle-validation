# Objective B — truth-gate → ZERO: accurate remediation map (current 290 violations / 92 files)

**Lane:** T2-research · 2026-07-05 · Real state measured (not stale). Path to zero, prioritised, with honest false-positive-vs-genuine split. All target files are in churny apps-script (pm/research) — this is the apply-ready plan for the clean-checkout owner + the 2 regex fixes for the gate file. Corrects my earlier assumption that most unqualified-78 were false positives — they are NOT (129/144 genuine).

## Current state (verified)
**290 violations across 92 files.** By check:
| Check | Count | Nature | Fix |
|-------|-------|--------|-----|
| unqualified-78 | 144 | **129 genuine** + 15 regex false-pos | (a) widen gate regex (−15); (b) add "in the RL-training condition (12% baseline)" qualifier to 129 (careful — only the Anthropic figure) |
| loose-dkim | 82 | MIX — genuine overclaims + false-pos (flags the sentences that CORRECT to "in transit") | (a) refine gate regex to not flag "DKIM…(strictly)… in transit" correcting text; (b) reframe genuine ones to "DKIM-timestamped in transit (not content-verified)" |
| wrong-book-date | 12 | ALL F13-class | change "Published Jan 6 2026" → "Published 2 Jan 2026 (author copy arrived 6 Jan)" |
| forbidden-framing | 11 | genuine ("systems will/would fake compliance") | reframe to observed-behaviour language |
| count-drift | 10 | genuine (15/29/… convergences) | reconcile to 19 — **sequence AFTER H1 decision** |
| eden-protocol-dating | 8 | internal (papers already clean) | period-separate name from Dec-8 date, or add "later named" |
| 8mo-before-anthropic | 8 | genuine framing | reframe (state dates, drop the "8 months before" priority spin) |
| garbled-valuation | 6 | genuine ("IPO at $96…") | fix figure or cut (F2/F7) |
| unprovable-causal | 4 | genuine | cut or soften |
| silence-as-endorsement | 4 | genuine ("Not disputed") | cut |
| wrong-person | 2 | genuine ("Amodei on BBC" — should be Jack Clark) | correct |
| unverified-figure | 2 | genuine ($18B/Microsoft-40%) | source-or-cut (F7) |
| cada-misframing | 1 | genuine ("EU kill switch law") | reframe to sovereignty (F7) |
| apr30-before-anthropic | 1 | genuine framing | reframe |

## Two-step path to zero
**Step 1 — gate regex refinements (removes ~15–40 false positives, prevents recurrence):**
- unqualified-78 → `(?<![\d.])78%(?![^.\n]{0,80}(?:RL-training|reinforcement|RL condition|12% baseline|baseline))` (qualifier anywhere in the sentence, not only immediately after).
- loose-dkim → exclude sentences containing "in transit" OR "(strictly)" + "not … verified" within the sentence (so self-correcting text isn't flagged).
**Step 2 — content fixes in the clean checkout (the genuine ~250):**
- Bulk-mechanical: 129 unqualified-78 qualifier-add (Anthropic figure only); 12 book-date; 2 wrong-person; the reframes.
- Sequenced: count-drift (10) reconcile to 19 AFTER the H1 decision (don't hard-code a number H1 may move).

## Honest note (LAW 1)
My earlier "grants clean" + "false-positive-heavy" framing was partly right (grants ARE clean; 15 gate false-positives exist) but the INTERNAL-doc gap is genuine and large (129 real unqualified-78 + ~60 other real violations). "Truth-gate → zero" is a real ~250-fix job (mostly mechanical), not merely a regex tweak. Product surfaces (papers, grants, public articles) are already clean; the 290 are internal working-docs.

## Route
→ T4 + clean-checkout owner (content fixes) + gate-file owner (2 regex refinements). Sequence count-drift after H1.
