# Grants batch B — canon-cleanliness status (T2 deliverable → T4) + truth-gate false-positive found

**Lane:** T2-research · 2026-07-05 · Charter COORDINATION deliverable ("deliver grants batch B status to T4-research"). My lane's angle = canon-cleanliness (numbers + truth-gate + anti-crank), complementary to the grant-lane's send-readiness. Grant apps are in churny apps-script (read-only from here).

## Estate
- **25 grant target dirs** under `grant-applications/` (ARIA, AISI, LTFF, Open-Phil, Wellcome, Nuffield, Leverhulme, Turing, Schmidt, Simons, Templeton, Thiel, UKRI-EPSRC, SFF, Manifund, Emergent/Effective-Ventures, McGovern, Mozilla, CIFAR, 1517, BERI, Anthropic-Fellows, FLI, Unbound…). The charter's "batch B = 5 apps / £624k" is a fundable SUBSET; the £624k programme ask is consistent where it appears (LTFF-2026, Open-Philanthropy-2026 both "£624,000"). ~100 send-ready PDFs present.

## Canon-cleanliness (truth-gate CHECKS across all grant whitepapers)
- **£624k ask: consistent** where stated. ✓
- **Truth-gate: effectively ZERO real violations.** The scan surfaced 2 apparent `unqualified-78` hits (both ARIA-2026) — VERIFIED as FALSE POSITIVES. ARIA-2026 fully qualifies: "the 78% alignment-faking figure **applies specifically in the RL-training condition (12% baseline)**" and "(78% in the RL-training condition (12% baseline))". Content is honest. ✓
- No count-drift / kill-switch-misframe / garbled-valuation / overstatement / loose-dkim hits in the grant apps. ✓

## Objective-B finding — truth-gate `unqualified-78` regex is TOO NARROW (false positives)
Current: `(?<![\d.])78%(?! in (?:the )?RL| in the reinforcement)` — only accepts the qualifier IMMEDIATELY after "78%". It misclassifies the legitimate form "78% [noun phrase] applies specifically in the RL-training condition" (ARIA-2026) as a violation.
**Recommended fix (widen the qualifier window):**
`(?<![\d.])78%(?![^.\n]{0,80}(?:RL-training|reinforcement|RL condition|RL-condition))` — treats any "78%" as qualified if "RL-training/reinforcement" appears within ~80 chars in the same sentence. Reduces false positives without letting a truly-bare "78%" through. This directly helps Objective B "truth-gate → zero" (removes spurious hits). Gate file is churny apps-script — route to clean-checkout owner + T4.

## Batch-B status (canon angle) for T4
- Fundable-facing grant surface is **canon-clean**: numbers consistent, 78% properly qualified, no anti-crank violations.
- Send-readiness / which-5-are-batch-B / LTFF interview-prep = grant-lane (T3-research owns grant-send). This deliverable is the CANON sign-off, not the send sign-off.
- One Objective-B gate-tune (unqualified-78 false-positive) with exact regex, routed.

## Route
→ T4-research (coordination) + clean-checkout owner (gate patch). Complements grant-lane send-readiness.
