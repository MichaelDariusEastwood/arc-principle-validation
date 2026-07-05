# T5 verification manifest — T2-research held edits (branch `pr/t2-research/paper-vi-falsifiability-20260705T0331Z`)

**Purpose:** single actionable list of every PAPER-CONTENT edit + REPRODUCE.md held on this branch for T5 GREEN before merge. Each row: file · commit · change · **exact verification criterion**. (Analysis/finding docs in `experiments/*.md` are routed separately and are not paper content — not listed here.) All are additive or wording-precision edits; none touch data, slugs, DOIs, or SHAs (LAW 3 preserved).

## A. Paper HTML content edits (6)
| # | File | Commit | Change | Verify (GREEN if…) |
|---|------|--------|--------|--------------------|
| 1 | `papers/Paper-III-Alignment-Scaling-Problem.html` | effbd0d | R²=1.00000000 reframed: "validated empirically" → "exact analytical identity (α=1/(1−β)), not an empirical fit … across 30 exact Bernoulli ODE solutions" | reads as an identity not a fit; matches Foundational's framing; no new numeric claim introduced |
| 2 | `papers/On-the-Origin-of-Scaling-Laws.html` | 9a12ce5 | ARC-Bound: added "(a conjecture later refined — Paper X retires the capability-growth law as a binding law, retaining the β>k co-scaling condition)" | note matches Paper III's existing ARC-Bound retirement note; no contradiction with Paper X |
| 3 | `papers/Paper-IV-b-Alignment-Saturation-at-Low-Depth.html` | 3f6cf26 | Overstatement: "Eden Protocol **proves** the ceiling can be broken" → "**demonstrates**" | wording only; claim strength reduced; no meaning change |
| 4 | `papers/Paper-IV-b-Alignment-Saturation-at-Low-Depth.html` | 0a99468, 47050f4 | Added §8.7 Falsifiability (5 defeat conditions for architecture-dependent saturation); renumbered to avoid §8.6 clash | section numbering unique; 5 conditions are genuine defeat conditions; consistent with paper's claims |
| 5 | `papers/Paper-VI-Honey-Architecture.html` | e2b0383 | Added §7.3 Falsifiability (5 defeat conditions for the honey-architecture claim) | conditions grounded in the paper's own claims; numbering clean |
| 6 | `papers/Paper-VIII-The-Load-Bearing-Proof.html` | 6b6de29 | Added §7.7 Falsifiability — what would overturn these nulls | consistent with Paper VIII's null-result framing |
| 7 | `papers/Paper-XI-Convergence.html` | 22be5cd | Added §4.1 Falsifiability grounded in §4 Method's 3 criteria as defeat conditions (independence-collapse / elastic-matching / temporal-gap + base-rate) | defeat conditions match §4 criteria verbatim; consistent with H1 independence concern; no overclaim |

## B. REPRODUCE.md (6 — grounded in artefacts, TODO(author)=env pins+API keys only)
| # | File | Commit | Verify (GREEN if…) |
|---|------|--------|--------------------|
| 8 | `experiments/honey-architecture__Paper-VI/REPRODUCE.md` | 2a4ba25 | every param (v2 10×150, v3 20×180 ts30, v4 15×5=225) matches the result-JSON metadata verbatim |
| 9 | `experiments/alignment-scaling__Papers-IV-a-b-c-d/REPRODUCE.md` | 5ff2ce8 | 8 models + depth 1024/4096/16384/32768 + entry arc_eden_v6_runner.py match scripts+JSONs |
| 10 | `experiments/eden-intervention__Paper-V/REPRODUCE.md` | b3e6d8b | 6 models × 4 depths × 10 prompts match results; entry eden_protocol_scaling_test_v3.py |
| 11 | `experiments/paper-ii-compute__Paper-II/REPRODUCE.md` | b3e6d8b | 6 models, 18/30 problems, avg_seq/par_alpha + cross-verification match results |
| 12 | `experiments/paper-i-foundational__Paper-I/REPRODUCE.md` | b3e6d8b | toolkit description accurate (matplotlib/numpy/scipy; no primary data) |
| 13 | `experiments/analysis-tools__Cross-Programme/REPRODUCE.md` | b3e6d8b | correctly described as post-processing consuming other dirs |

## C. Flags for author (NOT edits — do not need T5 GREEN, need author action)
- **F10** HRIH "directional priority 11mo" vs register "concurrent" → operator priority-framing decision.
- **Paper III "mathematical certainty"** metabolic universalisation → author scope-tighten.
- **IV-series multiple-comparison** correction statement → author.
- **Blinding methods paragraph** (surface the rigorous blinding in IV-a/b) → author.
- **alpha_align statistic labelling** (pooled vs independent vs bootstrap) → author.

## Merge sequencing note
Section A/B are safe to verify + merge independently of the operator decisions in C. The credential 19-vs-29 (F15) and the licence realignment (Objective G) are NOT on this branch — they are separate operator/clean-checkout items. Merging A/B does not touch the H1 count or the licence.

**Guard reminder:** origin/main `headline_count_verified` must remain 19 and `restore-29` must stay out of main ancestry through any merge.
