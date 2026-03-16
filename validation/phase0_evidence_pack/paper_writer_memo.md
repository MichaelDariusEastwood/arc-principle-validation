# Paper-Writer Memo

## 1. Claims Safe To Lead With

- The blind v5 alignment benchmark is real, file-backed, and usable: six final subject-model runs contain valid alignment consensus scores in `consensus_weighted_mean`.
- The safest empirical headline is architecture dependence: the blind benchmark produces mixed alignment-scaling behavior rather than one universal positive curve.
- Groq Qwen3 is the clearest positive signal in the current pack (grouped α=+0.118), with Grok 4 Fast positive but less decisive (grouped α=+0.401).
- The benchmark methodology itself is a contribution: response laundering, non-participant scoring, and cross-model blind evaluation are already implemented and backed by result files.
- Papers IV.a-d already cover the v5 blind benchmark. Use this pack to harden those claims, not to pretend the work is unwritten.

## 2. Claims That Must Be Qualified

- Eden intervention results are pilot evidence, not canonical proof. They are nonblind and single-scorer.
- DeepSeek, Gemini, and Groq show positive pilot intervention deltas, but those deltas should be framed as promising rather than decisive.
- Claude intervention evidence is exploratory/partial and the delta is negligible after excluded failures.
- Grok intervention evidence is mixed-quality because valid Eden rows are missing and the positive delta depends on exclusions.
- Positive blind alignment signals are not universal across models. Claude and DeepSeek are near-flat; Gemini and GPT-5.4 do not support a positive blind scaling claim in the current pack.
- Paper V already covers the Eden intervention results written so far. This table is an evidence-hardening pass for that paper, not proof that the Eden topic lacks a paper.

## 3. Claims To Omit For Now

- Omit any claim that alignment scaling is universally positive across frontier models.
- Omit GPT-5.4 intervention results from inferential sections; keep them only as an operational failure note.
- Omit the toy-scale claim that Eden advantage scales with complexity as a lead result.
- Omit cross-domain universality and ARC Bound language from the empirical headline of the next paper.

## 4. Live-Model Evidence vs Simulation/Toy Evidence

- Live-model evidence: the v5 blind alignment benchmark and the Eden intervention JSON result packs.
- Simulation/toy evidence: the honey architecture and self-modifying PDFs/artifacts.
- Write them as different evidence tiers. The live-model paper should not pretend the toy/simulation results are frontier-model proof.
- The toy/simulation material is still useful as mechanistic support and as the bridge into the companion honey paper.
- The actual unwritten gap is the honey/self-modifying simulation paper. The v5/Eden topics already have written papers; this pack makes their evidence base cleaner.
- Phase 0 located the honey/self-modifying artifacts as PDFs and PNGs, but did not identify a canonical raw-result directory or source script in this pass. The companion paper should state that provenance level honestly unless additional source files are recovered.

## 5. Recommended Paper Order

1. Use this pack to harden the claims already made in Papers IV.a-d and V.
2. The next genuinely missing paper is the honey/self-modifying simulation paper, explicitly framed as simulation + toy-system evidence.
3. Treat Eden v3 blind replication as future replication work, not as a prerequisite for Phase 0.
4. Keep the anti-sycophancy notebook material as a later methods note or appendix, not as the next flagship paper.
5. Defer any v3/v6 tool consolidation until after the paper. Do not merge the runners before publication work.

Phase 0 stop condition: ledger complete, two tables complete, this memo complete. No paper drafting in this phase.
