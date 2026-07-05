# Objective B — truth-gate honest state + count-drift endemic (NOT zero) · 2026-07-05

**Lane:** T2-research (owns truth-gate wiring + canonical-facts). **Route:** PM-research, T4-os, T5-research.

## LAW-1 CORRECTION of my own earlier report
Earlier this session I reported "TRUTH GATE: clean". **That was WRONG** — a wrong-CWD false negative. `tools/research_truth_gate.py`
defaults to scanning `pm/research` **relative to the current directory**; I ran it from `tools/`, so it scanned the nonexistent
`tools/pm/research`, found 0 files, and printed "clean". Run correctly from the repo root it was **never clean**.

## True current state (reverted/live gate, run from repo root)
**290 violations.** Breakdown: `unqualified-78` 144 · `loose-dkim` 79 · `count-drift` 10 · `forbidden-framing` 11 ·
`wrong-book-date` 10 · `eden-protocol-dating` 8 · `8mo-before-anthropic` 8 · others ≤6. **Objective B "truth-gate → zero" is UNMET.**

## Finding 1 (usability bug) — the gate false-passes from the wrong CWD
Fix: anchor the default path to the repo root (e.g. `os.path.join(REPO_ROOT, 'pm/research')`) so it cannot silently scan nothing.
Until then, ALWAYS run from repo root. A truth gate that false-passes on a CWD mistake is dangerous (it manufactured MY false "clean").

## Finding 2 (HIGH, LAW 2) — convergence count-drift is ENDEMIC and mostly INVISIBLE to the current check
The current regex requires the exact phrase "N independent convergences" and catches only 10. The REAL drift is **46 hits across
25 files** claiming **15 / 28 / 29 / 31** convergences — even in FILENAMES (`CONVERGENCES-28-FINAL`, `-29-FINAL`, `-31-FINAL`,
`CONVERGENCE-29-*`). Canonical is **19** (T2-Canon verdict 2026-07-05: 19 STRUCTURAL; 28/29/31 conflate convergent+directional+prediction).
Every one of these 25 files needs relabelling to "19 structural (+ N directional + M predictions)".

**Patch (apply on a T2 branch — I could not, the shared checkout was on a pm-research branch):**
```python
# replace the single count-drift line with:
(r'\b(?:15|17|18|20|21|22|25|28|29|31|32) independent(?:ly sourced)? (?:convergences|confirmations)\b', 'count-drift'),
(r'\b(?:15|17|18|20|21|22|25|28|29|31|32) convergences\b', 'count-drift'),   # 19 is NEVER listed
```
This raises count-drift 10→46 (surfaces the register header "28 convergences", the restore-29 target "29 convergences", etc.).

## Finding 3 (MEDIUM) — unqualified-78 + loose-dkim over-fire on audit/discussion docs (223 of 290)
Many hits are docs that MENTION a forbidden pattern to WARN against it — e.g. loose-dkim flags a line literally saying
"we do NOT claim DKIM-verified", and unqualified-78 flags audit docs that DO qualify "78% ... during RL training" but not in the
exact "78% in the RL" phrasing. These need either better EXCLUDE_DIRS coverage (add the audit/priority-claims docs) or
context-aware negative lookaheads ("do not claim", "cannot claim", "NOT"). Triage required before "zero" is meaningful.

## Path to a MEANINGFUL zero (Objective B)
1. Fix the CWD bug (anchor path to repo root).
2. Apply the count-drift patch (above).
3. Relabel the 25 count-drift files to 19 per the Canon verdict (coordinated; T4 adjudicates, owning lanes apply).
4. Triage unqualified-78/loose-dkim: exclude audit docs OR add "do-not-claim" negative lookaheads.
Only then does "TRUTH GATE: clean" MEAN the corpus is contradiction-free. Today it does not.
