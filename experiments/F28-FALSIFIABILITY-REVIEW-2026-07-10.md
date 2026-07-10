# F28 falsifiability review — 2026-07-10

**Lane:** T6-research (independent review, no edits to underlying claim files).
**Scope:** F28 in the F1–F29 corpus audit backlog: "Missing falsifiability in Papers VII, IV-a, IV-c, IV-d, and XI." The audit's own disposition marks F28 as "resolved" because falsifiability sections have now been drafted into all five papers. This review checks whether the drafted sections are genuinely falsifiable and whether the test each paper offers can actually discriminate signal from noise.

## F28 location (as found)

- Backlog entry: `experiments/OBJECTIVE-A-F1-F29-COVERAGE-MAP-2026-07-05.md`, row F28.
- Context notes: `experiments/LADDER4-PAPER-XI-FALSIFIABILITY-ADDED-2026-07-05.md`, `experiments/OBJECTIVE-A-FALSIFICATION-CRITERIA-AUDIT-2026-07-05.md`, `experiments/FALSIFICATION-STATUS-SUMMARY-ALREADY-EXISTS-2026-07-05.md`.
- Load-bearing text lives in the paper HTMLs themselves:
  - `papers/Paper-VII-Cauchy-Unification/Paper-VII-Cauchy-Unification.html` §12.1 "Falsifiability: what would defeat this claim".
  - `papers/Paper-IV-a-Baked-In-vs-Computed-Alignment/Paper-IV-a-Baked-In-vs-Computed-Alignment.html` §6.9.
  - `papers/Paper-IV-c-ARC-Align-Benchmark/Paper-IV-c-ARC-Align-Benchmark.html` §12.5.
  - `papers/Paper-IV-d-The-Effect-of-Blinding-on-AI-Alignment-Evaluation/Paper-IV-d-The-Effect-of-Blinding-on-AI-Alignment-Evaluation.html` §8.1.
  - `papers/Paper-XI-Convergence/Paper-XI-Convergent-Evidence.html` §5.

F28 is not itself a single scientific claim; it is a meta-audit item over five papers. The review therefore covers each paper's drafted falsifiability section and then returns an overall verdict for F28.

## Verdict

**FALSIFIABLE-AS-STATED, with two working-scientist caveats.**

- All five sections identify the correct vulnerabilities and state defeaters that could in principle strike the load-bearing claim. In classical Popperian terms the sections pass: an outcome is specified in advance whose observation would refute the claim.
- Caveat 1 (evidential status): under two papers' own defeat conditions the *current* evidence in those same papers is not yet on the "confirmed" side of the line. This is disclosed honestly (Paper VII calls itself "exploratory"; Paper IV-a concedes n=6; Paper IV-d concedes multiple simultaneous v4/v5 changes; Paper XI's H1 independence stress-test already flags rows 13/14). That is a maturity issue, not a falsifiability failure. Reviewers should nonetheless read the falsifiability sections *with* the accompanying limitations sections, not as a substitute for them.
- Caveat 2 (headline discipline): a couple of section-external headline phrasings (see per-paper notes) risk overstating what the falsifiability section actually promises. Concrete rewording suggestions below; no rewrite of the falsifiability sections themselves is needed.

## Per-paper analysis

### Paper VII (Cauchy-Unification) §12.1

- **Claim under test.** A domain's composition operator, classified from independently known physics *before* fitting, predicts its scaling-law family. Primary result: 19/25 hit rate against a one-in-three-to-four chance baseline, one-sided binomial p = 1.56 × 10⁻⁵.
- **Stated defeaters.** Hit rate at chance under pre-registered independent-classifier replication; post-hoc classification (blind assessors cannot reproduce operator labels from physics alone); fit degeneracy (families empirically indistinguishable on the observed range/noise); miss rate rising when post-hoc exclusions are removed; the "exploratory ceiling" clause (until a locked pre-registration exists, presenting the result as established would itself falsify the stated evidential status).
- **Power / confound at a working-scientist level.** The binomial power calculation is sound *if* the operator labels are truly blind. The dominant risk is exactly the one the paper names — post-hoc operator freedom. The mitigation (an independent blind classifier) is the right test and would discriminate. The "unrationalised misses" clause is important: six misses on 25 gives investigators substantial degrees of freedom to explain misses after the fact.
- **Overstatement risk.** Low. The abstract-level status "exploratory" is preserved in §12.1's fifth bullet. The p-value should always be quoted alongside the exploratory-ceiling caveat; if any downstream summary quotes the p-value without it, that summary is what would need correcting, not §12.1.
- **Verdict.** Falsifiable as stated. Discriminating test is nameable (blind classifier + pre-registered replication + fixed miss-exclusion rule).

### Paper IV-a (Baked-In vs Computed Alignment) §6.9

- **Claim under test.** Alignment response to inference-time depth is architecture-dependent; models sort into three tiers (positive / flat / negative); capability scaling does not predict alignment scaling.
- **Stated defeaters.** Tier instability across pre-registered replication; scorer-panel dependence; blinding leakage (subject identifiable despite the four-layer blind); coupling reappearing at larger n; depth confound (depth manipulation covaries with length/format).
- **Power / confound.** With six subject models and seven scorers, the tier three-way partition is inevitably fragile. §6.9 correctly refuses to treat the "baked-in vs computed" mechanistic labels as load-bearing; the load-bearing claim is the *blinded behavioural* result. That is a legitimate move — an existence-style demonstration on n=6 can survive a small-sample honesty caveat, but a universality claim cannot. The v5.4.0 seven-scorer note (7-way vote, spurious-unanimity ≈0.8%) meaningfully strengthens the internal signal.
- **Overstatement risk.** Small but present. Any section that phrases the "capability does not predict alignment" finding as universal would go beyond what n=6 can carry. §6.9's own fourth bullet correctly names this. The headline finding should everywhere read "does not predict in this sample" until the replication under bullet 4 is done.
- **Verdict.** Falsifiable as stated. Discriminating test is nameable (larger-n pre-registered replication with fresh blind scorer panel and depth-only manipulation).

### Paper IV-c (ARC-Align benchmark) §12.5

- **Claim under test.** ARC-Align is a valid *instrument* for measuring alignment as a function of reasoning depth.
- **Stated defeaters.** Construct-validity failure (scores don't track any independent alignment/safety measure); low inter-scorer reliability; non-reproducibility from the open-source pipeline; depth-manipulation invalidity (models ignore the instruction, or depth covaries with length/content); Goodhart failure (adversarial gaming beats it).
- **Power / confound.** The right frame — benchmarks are judged by construct validity, reliability, reproducibility, and gameability. The five bullets cover all four axes and add Goodhart. The construct-validity bullet requires an independent alignment measure to correlate against; naming a specific yardstick (Anthropic's HH, Meta's harmlessness set, expert judgement, or otherwise) would make the test sharper.
- **Overstatement risk.** Low. §12.5's opening sentence explicitly frames ARC-Align as a "candidate benchmark", which is the right hedge.
- **Verdict.** Falsifiable as stated. Discriminating test is nameable but would be sharper if the specific external validity criterion were named.

### Paper IV-d (Effect of Blinding) §8.1

- **Claim under test.** Unblinded AI alignment evaluation can be wrong about the *direction* of an effect, so blinding is methodologically necessary.
- **Stated defeaters.** No blind/unblind divergence on a clean re-run holding scorers, prompts, model versions and consensus rule constant; divergence explained fully by a concomitant change (scorer set, problem set, model versions), not the blinding manipulation; blinding leakage; self-preference under blinding; direction failing to survive proper power.
- **Power / confound.** This is the most challenging of the five to test because §8 concedes multiple protocol components changed between v4 and v5. §8.1 handles this by restricting the load-bearing claim to an *existence* statement ("*can* flip the sign"). An existence claim needs only one clean, confound-controlled case; that is a legitimate scoping move. The tests as written would discriminate: the "clean re-run holding everything constant" bullet is exactly the required experiment. Until it is run, the load-bearing existence claim rests on an evidentially exploratory case, which the paper acknowledges.
- **Overstatement risk.** Real but named. If anywhere the paper (or a summary of it) infers that "unblinded evaluation is unsafe" as a general operational rule, that infers a universal claim from an existence claim. §8.1 warns against this by restricting the claim to the sign-flip existence. Any section-external phrasing that reads as universal-methodological should be softened to "at least in this case" or "at least in this evaluation programme".
- **Verdict.** Falsifiable as stated *for the existence claim*. Not falsifiable in the strong "unblinded evaluation is universally unsafe" reading, which the paper does not make. The paper is protecting itself correctly.

### Paper XI (Convergent Evidence) §5

- **Claim under test.** Nineteen structurally independent convergent developments support the ARC principle.
- **Stated defeaters.** Count dropping below chance-distinguishable level under strict one-event-per-row dedup and primary-source verification; missing denominator (no systematic disconfirming search); look-elsewhere effect (observed rate not exceeding false-positive rate expected from pattern-matching across the surveyed number of domains); predictive failure on pre-registered forward predictions.
- **Power / confound.** The four defeaters correctly identify the entire attack surface for a convergence-style claim (apophenia, base-rate neglect, multiple comparisons, unfalsifiable retrofit). The paper's own H1 independence stress-test already flags rows 13 and 14 as sharing sources (Pope-encyclical + Olah-co-presents-it), which under bullet 1 collapses the pair into a single row — so the count itself is already contested internally, honestly. The denominator bullet requires an equally systematic disconfirming search; per the audit trail no such search has yet been published, which means under §5's own bullet 2 the claim is currently exploratory as evidence, even though it is fully falsifiable as a hypothesis.
- **Overstatement risk.** The word "convergences" carries interpretive weight. Where the paper (or its abstract, or the credential page) states a specific count (19), that count is provisional under §5's own bullet 1. Everywhere the count appears the language should read "nineteen candidate structurally independent convergent developments *under the current dedup rule*" and cross-refer to §5. This is the same tension the F1/F11/F15 audit rows identified, and §5 pre-concedes it.
- **Verdict.** Falsifiable as stated. Discriminating tests are nameable (locked-dedup re-audit; systematic disconfirming search across the same domain population; pre-registered forward predictions). Under two of its own defeaters (independence, denominator) the paper is currently on the exploratory side — and §5 says so.

## Cross-cutting observations

1. **All five sections avoid the classic unfalsifiable-corpus failure mode.** None reads as post-hoc rationalisation of everything the framework predicts. Each names outcomes that would strike the load-bearing claim, and none uses "explains anything" language.
2. **The exploratory-ceiling framing in Paper VII §12.1 is the strongest single sentence of the five.** It converts "no locked pre-registration exists yet" from a weakness into an internally enforced ceiling on claim strength. That construction is exportable: Paper XI in particular would benefit from an explicit sentence of the same shape ("until a locked disconfirming-search denominator exists, the count cannot rise above exploratory").
3. **Paper VII and Paper IV-a share the small-sample discrimination problem.** Both have honest defeat conditions (bullet 4/5 respectively) that identify what a pre-registered larger-n replication would show. Neither claim is yet on the confirmed side of that test. The falsifiability sections do not overstate this.
4. **Paper IV-d's existence-vs-universal distinction is important and easy to miss in downstream summaries.** Any executive summary, abstract, or slide that says "unblinded evaluation is unsafe" without qualifier is overstating. §8.1 protects against this internally; downstream artefacts should be spot-checked.
5. **Paper XI's §5 is the most exposed and the best-drafted.** It correctly targets the four attack surfaces a hostile reviewer would use. It is also the paper whose current evidence sits closest to the exploratory line under its own defeat conditions — the denominator is missing and rows 13/14 flag under bullet 1.
6. **The Foundational paper's "Falsification Status Summary" model is not yet mirrored in the five F28 papers.** Adopting Foundational's "confirmed / weakened / refuted / untested" per-criterion column into each of the five would upgrade the sections from "criteria stated" to "criteria stated *and* their current empirical status reported". Not required to close F28, but a clear next-round improvement.

## Concrete rewording suggestions (proposals only; not applied)

These are optional wording tweaks for whoever holds the pen next. They do not change the falsifiability sections themselves; they close remaining overstatement risk in headline or abstract language elsewhere.

- **Paper XI.** Everywhere a specific count appears ("19 convergences", "nineteen convergences"), add ", under the current dedup rule (see §5)" on first mention per section. This carries the H1 stress-test caveat into the reader's eye without weakening the section.
- **Paper XI §5.** Add one Paper-VII-style sentence: "Until a locked, pre-registered disconfirming-search denominator has been reported, the count and rate cannot rise above exploratory as evidence, however rigorous the independence audit of surviving rows."
- **Paper IV-a.** Where the "capability does not predict alignment" finding is stated in abstract or headline form, add "in this six-model sample" until bullet 4 of §6.9 has been executed.
- **Paper IV-d.** Where the paper's implication reads "blinding is methodologically necessary", ensure it is qualified by "for programmes of this kind" or paired with "on at least one confound-controlled case" until bullet 1 of §8.1 has been executed. §8.1 itself is fine; this is about downstream language.
- **Paper IV-c.** Add a named external-alignment yardstick under bullet 1 of §12.5 (Anthropic HH-RLHF, Meta harmlessness set, expert-panel score, or another specific instrument). Naming the yardstick makes construct validity sharply testable rather than in principle testable.
- **Paper VII.** No wording change required. §12.1 is the model.

## Bottom line

F28 as a backlog item is legitimately marked "resolved" *in the sense* that all five papers now carry falsifiability sections that identify the correct vulnerabilities and name discriminating tests. What the disposition line does not say, and what any hostile reviewer will want to see clearly, is that two of the five papers (IV-d, XI) currently sit on the exploratory side of their own defeat conditions and one (VII) says so in the section itself. That is honest science, not a falsifiability failure. Adopting the concrete rewording suggestions above, and porting Foundational's per-criterion status column into the five, would move the sections from "falsifiable in principle" to "falsifiable in principle and status-transparent in practice", which is what a peer-reviewer at a serious venue would look for.

**Verdict: FALSIFIABLE-AS-STATED.** No rewrite required to close F28. Suggested wording tweaks are quality-of-life for headline language, not falsifiability repair.

---

Review artefact only. Underlying claim files were not modified.
