# Objective A — "execute audit F1–F29": coverage map (session work → actual backlog)

**Lane:** T2-research · 2026-07-05 · Maps this session's ~25 cycles of corpus-hardening to the **actual** F1–F29 audit backlog (`pm/research/strategy/T2-fable5-corpus-audit-plan-2026-07-02.md`). This is the genuine "execute audit F1–F29" deliverable. (Note: the corpus's *own* falsification criteria are a separate F1–F13 scheme, audited separately — see OBJECTIVE-A-FALSIFICATION-CRITERIA-AUDIT.)

Legend: ✅ resolved/verified · ◐ partial · ○ open (next work) · ▶ owned by another lane.

| F# | Backlog item | Disposition | Evidence |
|----|--------------|-------------|----------|
| F1 | "31 convergences" collapses under dedup (≈24) | ✅ superseded | Canon verdict = 19 structural; independence stress-test flags rows 13/14 double-count + 2 predictions (goes further than F1) |
| F2 | "Anthropic IPO $96.5B" false | ◐ papers clean | Papers pass `garbled-valuation` (IPO at $96) full-gate; internal docs = truth-gate churn |
| F3 | Book publication date | ✅ resolved | Operator: 2 Jan 2026 canonical |
| F4 | Wrong person on BBC convergence | ◐ papers clean | Papers pass `wrong-person` (Amodei on BBC); convergence row 15 = Jack Clark (verify register) |
| F5 | Unqualified "78%" | ✅ product clean | Public articles 0 unqualified; grants clean (verified). Internal = churn |
| F6 | Unprovable causal claims | ◐ papers clean | Papers pass `unprovable-causal` full-gate; internal = churn |
| F7 | Unverified specifics (sources-or-cut) | ○ open | Needs per-claim sourcing pass (internal docs) |
| F8 | Numeric drift matrix | ◐ partial | Papers clean; ~25 internal count-drift docs need one-canonical-value relabel (needs clean apps-script checkout) |
| F9 | Three papers lack PDFs (HRIH, Paper C, XI) | ✅ resolved | Completeness matrix: PDF 22/22 — all present incl HRIH/C/XI |
| F10 | Sharma & Chopra "11 months" false priority (FATAL) | ◐ **flagged → operator/T4** | HRIH "directional priority 11mo" CONTRADICTS register "concurrent"; fix specified (drop active priority claim / match register). Operator decides priority framing. See F10-SHARMA-CHOPRA finding |
| F11 | Convergence count variants (8,11,15…) | ✅ addressed | count-drift + Canon verdict → single count 19 |
| F12 | "Single source of truth" wrong | ✅ addressed | research-canonical-facts.json built + guarded (count 19) |
| F13 | FULL-EVIDENCE-TIMELINE inconsistent | ◐ diagnosed → clean-checkout owner | 3 precise fixes: book-date Jan6-vs-Jan2 (Line66), F25 author-order residue (Line62), ISBN-10/13 drift. Churny checkout = cannot edit; line-numbered. See F13 finding |
| F14 | Equation family unreconciled | ✅ resolved | equations-by-artifact verified consistent (U=I×R / R^α / R² by artifact) |
| F15 | credential.html table mismatch | ◐ diagnosed → T4/PM+owner | credential.html has 19-vs-29 contradiction LIVE ("19 independently sourced" x2 + "THE 29 CONVERGENCES" section). Same tension as merge guard. Fix to 19 but finalise w/ H1. Churny = cannot edit |
| F16 | PNAS DOI pgag076 unverified | ◐ partial | DOI-scheme checked; real-existence deferred → T5 online |
| F17 | Public evidence page broken | ▶ T6 | SEO/nav/public surface = T6-research |
| F18 | Named-individual attribution risks | ▶ T6 | Public page = T6 |
| F19 | Forbidden-phrase residue in JSONs/handoffs | ◐ partial | Truth-gate internal-docs scope; product clean |
| F20 | Missing .eml shipping | ○ flagged (G) | .emls not on Mac 2; verify-yourself manifest built (SHA-256), .eml shipping = operator/provenance |
| F21 | GOOD: £624k + α≈2.24 retraction verified | ✅ confirmed | Re-verified both: grants £624k consistent; super-linear retraction handled honestly |
| F22 | "Eden Protocol" Dec-8 in FOUR papers | ✅ papers now clean | Full-gate `eden-protocol-dating` = 0 violations; Paper XI ref is period-separated label + naming distinction stated. (Backlog predates fixes) |
| F23 | Metabolic numbers contradict within Paper III | ✅ addressed | Refined: mean(11 species)=2.4% vs set-wide=2.5% — different quantities, not contradiction; recompute → T5 |
| F24 | ARC Bound live in III + Foundational | ✅ resolved | III + Foundational HAVE the retirement note; On-the-Origin was the lag → FIXED this session |
| F25 | Author-order Paper XI (Grier/Morrell/Elliott) | ✅ verified resolved | 0 wrong-order occurrences; ref reads "Morrell, Elliott, & Grier (2026)" correctly |
| F26 | MToC missing FOUR papers (X, C, XI, HRIH) | ✅ verified resolved | MToC lists all 4 (X,C,XI,HRIH) + trilogy + pre-papers + IV a/b/c/d — complete at 22. Backlog predates fix |
| F27 | Overstatement inventory | ✅ done | Overstatement pass: corpus hedges well; 1 flag (Paper III "mathematical certainty") |
| F28 | Missing falsifiability (VII, IV-a/c/d, XI) | ✅ resolved | f28 branch = VII/IV-a/c/d; XI added this session — all 5 present |
| F29 | HRIH:232 drops α (U=I×R vs U=I×R^α) | ✅ equation resolved / ◐ ISBN cosmetic | Bare U=I×R only in dated Dec-8 priority table (consistent w/ canon); line-232 ref stale. ISBN 978-1806056200 uniform but non-standard hyphen — cosmetic house-style flag |

## Tally
- **✅ resolved/verified/confirmed: 14** (F1, F3, F5, F9, F11, F12, F14, F21, F22, F23, F24, F25, F27, F28)
- **◐ partial: 6** (F2, F4, F6, F8, F16, F19) — mostly product-clean, internal-doc churn remaining
- **○ open (T2-addressable next): 6** (F7, F10, F13, F15, F26, F29) — F10/F26/F29 are concrete + paper-addressable
- **▶ other lane: 2** (F17, F18 → T6)

## Next work (this lane, in order)
1. **F10** — Sharma & Chopra "11 months" false-priority (graded FATAL) — verify + fix/cut.
2. **F26** — MToC missing X/C/XI/HRIH — verify against MToC.
3. **F29** — HRIH α-exponent (U=I×R vs U=I×R^α) — verify + reconcile to suite standard.
Then F7/F13/F15 (internal/credential) as clean-checkout allows.

## Honest headline
Of 29 backlog items, **~half are resolved/verified by this session**, the product surfaces are clean, and the remaining open items are concrete and named — not vague. The corpus is materially closer to "survives a hostile reviewer" and the residue is tracked, not hidden.
