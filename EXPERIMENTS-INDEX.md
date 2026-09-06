# EXPERIMENTS-INDEX — the canonical map of every experiment in the programme

Compiled 4 July 2026 from a file-level survey of both experiment homes. This file is the
navigation truth for the repo; the older experiments/CANONICAL_RESULTS_MAP.md (12 March)
is retained but superseded where they disagree.

## How this repository is organised (read this first)

- **Canonical home per paper: `papers/<paper>/`** (experiments, results, figures beside the
  manuscript). Paper X is the reference pattern: fully self-contained with its own Makefile,
  Dockerfile, tests and SHA-256 MANIFEST.
- **`experiments/<name>__Paper-N/`** is the by-experiment working view: shared estates
  (e.g. the alignment suite serving Papers III and IV.a-d) live here. Where both homes hold
  copies, THIS INDEX states which copy is canonical.
- **Nothing is ever deleted.** Duplicates and superseded files move to `_archive/` folders
  local to their home (operator rule, 2026-07-04).
- The out-of-repo folder `~/Arc & Eden Test Results/` is the raw capture archive; every
  result a reader needs has an in-repo canonical copy named below.

## Paper X in one box (the five suites, disentangled)

| Suite | Runner | Protocol | Results | Figures |
|---|---|---|---|---|
| Simulation E1-E9 | experiment_coscaling.py | (in-paper spec) | results/report.txt, verdicts.json | figures/exp1..exp9*.png (all ten) |
| Real-model v1 (pilot) | experiments/scripts/realmodel_coscaling.py + agent_bridge_run.py | experiments/PROTOCOL.md | results/realmodel/ (claude-opus, 26 Jun) + REAL_MODEL_CLAUDE_RESULTS.md | none |
| Real-model v2 | realmodel_coscaling_v2.py | PROTOCOL_V2.md + CONFIRMATORY_PROTOCOL_V2.md | results/realmodel_v2/ (selftest only) | none |
| Real-model v3 | realmodel_coscaling_v3.py | RUN_REAL_MODEL_EXPERIMENT.md | results/realmodel_v3/ (deepseek-v4, 2 Jul) | none |
| Drift engine (the 45-trajectory headline) | experiments/scripts/drift_engine.py | RUN_REAL_MODEL_EXPERIMENT.md | results/drift/ (gpt-3.5-turbo, 2 Jul) | none |
| Hard-tasks battery | hard_tasks_v3.py + run_hard_tasks.py | (script header) | results/hard/ (deepseek-v4 null, 2 Jul) | none |

---

# Part 1: Papers I, II, IV.a-d, V, VI, VII, Foundational, On-the-Origin

# Canonical Experiments Index — Part 1

Repo: `<repository root>`
Compiled from direct file inspection on 4 July 2026. Status labels drawn from `experiments/CANONICAL_RESULTS_MAP.md` (dated 12 March 2026). The canonical map points results at the out-of-repo archive `~/Arc & Eden Test Results/` (which lives outside this repository).

Convention used below: "canonical home" is the paper's own `papers/<paper-dir>/` directory. "Working home" is the root `experiments/<name>__Paper-N/` directory that carries the scripts. In every case checked, the paper-side result and figure files are byte-identical copies of the experiment-side ones, except where noted under Mirrors/duplicates.

---

### Paper I: The ARC Principle: Formalisation and Preliminary Validation of Recursive Capability Scaling
- Canonical home: `papers/Paper-I-ARC-Principle/` (README v1.1, 17 Jan 2026)
- Experiments:
  - `arc_principle_research_toolkit.py` (in `experiments/paper-i-foundational__Paper-I/scripts/`) — tests U = I × R^alpha across models, computing per-model alpha exponents (per-segment and full-fit), R², interpretation, and a falsification verdict against the alpha > 1 claim; also runs a sensitivity sweep across token ratios.
- Results:
  - `results/arc_principle_results.json` — output of the toolkit. Contains OpenAI o1 (alpha_1_to_64 = 0.102, alpha_64_to_1000 = 0.323, R² = 0.902, verdict "STRONGLY SUB-LINEAR: FALSIFIED"), DeepSeek R1 (alpha = 1.346, R² = 1.0, verdict "WEAKLY SUPER-LINEAR: PARTIAL FALSIFICATION"), and a token-ratio sensitivity vector. No status label appears for Paper I in `CANONICAL_RESULTS_MAP.md`; `experiments/README.md` classifies the evidence tier as Canonical.
  - `experiments/paper-i-foundational__Paper-I/results_paper_i_toolkit.txt` — the terminal log for that toolkit run (empty of numeric summary in the head; retained for provenance).
- Figures:
  - `figures/arc_scaling_comparison.png` — scaling-law comparison chart produced by the toolkit; used to illustrate the alpha comparison in the paper narrative. Exact section reference not visible in the README (only the PDF/HTML would confirm).
- Mirrors/duplicates: `papers/Paper-I-ARC-Principle/results/arc_scaling_comparison.png` is byte-identical to `figures/arc_scaling_comparison.png` (same file placed in two subdirs). No root-experiments duplicate of the figure exists. Canonical copy: `papers/Paper-I-ARC-Principle/figures/arc_scaling_comparison.png` and `results/arc_principle_results.json`.
- Gaps/confusions found: (a) The main `experiments/README.md` claims six frontier AI models were tested for Paper I; the JSON only contains OpenAI o1 and DeepSeek R1 with an estimated token count noted for DeepSeek. (b) The paper's `README.md` says all experiment material lives in `./experiments/` but the paper-side `experiments/` directory is empty apart from a `.DS_Store`; the actual scripts sit in root `experiments/paper-i-foundational__Paper-I/`.

---

### Paper II: The ARC Principle: Experimental Validation of Super-Linear Error Suppression Through Sequential Recursive Processing (v13.0)
- Canonical home: `papers/Paper-II-Experimental-Validation/`
- Experiments (in `experiments/paper-ii-compute__Paper-II/scripts/`):
  - `arc_paper_ii_validation_v1_deepseek.py` — original single-model DeepSeek-R1 sequential-depth ladder plus parallel-samples condition.
  - `arc_paper_ii_validation_v2.py` — multi-model port of v1 across six models (DeepSeek R1, OpenAI GPT-5.4, Claude Opus 4.6, Gemini Flash, Groq Qwen3-32B, Grok 4.1 Fast); 12 tier-1 + 18 tier-2 problems; alpha_sequential and alpha_parallel per tier with bootstrap CI, optional 4-layer blinding verify pass.
  - `arc_validation_deepseek.py` — supporting DeepSeek-only validation harness.
  - `regenerate_figure_15.py` — regenerates the "complete experimental summary" figure; header explicitly notes the correction of alpha from 3.15 to 2.2.
- Results (in both `papers/Paper-II-Experimental-Validation/results/` and `experiments/paper-ii-compute__Paper-II/results/`):
  - `arc_paper_ii_deepseek.json`, `arc_paper_ii_gemini.json`, `arc_paper_ii_grok-4-fast.json`, `arc_paper_ii_groq-qwen3.json`, `arc_paper_ii_openai.json` — the five per-model final JSONs referenced in `CANONICAL_RESULTS_MAP.md`. Status labels (from the canonical map, model → notes): deepseek "step_function_or_ceiling" (alpha_seq = 3.049, alpha_parallel = 0.000), gemini "continuous_or_floor" (alpha_seq = 0.590, r² = 0.861), grok-4-fast "step_function_or_ceiling" (all NA), groq-qwen3 "continuous_or_floor" (alpha_seq = 0.242, r² = 0.103), openai "step_function_or_ceiling" (alpha_seq = 1.470, r² = 0.947). Note the map only lists five models; no Claude row.
  - `arc_paper_ii_combined.json` — combined analysis over the per-model files (produced by the v2 harness).
  - `arc_deepseek_results_20260121_175028.json` — the original 21 January v1 DeepSeek run (dated).
- Figures (15 numbered figures in `papers/Paper-II-Experimental-Validation/figures/`):
  - `figure_1_raw_data.png` — raw per-depth accuracy, first empirical panel of the paper.
  - `figure_2_scaling_loglog.png` — log-log scaling by depth used for the alpha regression.
  - `figure_3_sensitivity.png` — parameter-sensitivity sweep.
  - `figure_4_alpha_comparison.png` — cross-model alpha bar comparison.
  - `figure_5_error_reduction.png` — error-rate suppression by depth.
  - `figure_6_alignment_taxonomy.png` — taxonomy of response classes (feeds into the Paper IV family).
  - `figure_7_cross_domain.png` — cross-domain scaling comparison.
  - `figure_8_equation.png` — the U = I × R^alpha equation panel.
  - `figure_9_divergence.png` — sequential vs parallel divergence.
  - `figure_10_summary.png` through `figure_15_complete_summary.png` — summary panels; `figure_15_complete_summary.png` is explicitly the corrected-alpha version rebuilt by `regenerate_figure_15.py`.
  - No per-figure section mapping is written into the paper README; the exact section a figure lands in is only visible in the PDF/HTML sources, so section attribution below the figure list is inferred from figure titles.
- Mirrors/duplicates: There are THREE copies of every Paper II figure — `papers/Paper-II-Experimental-Validation/figures/`, `papers/Paper-II-Experimental-Validation/results/` (loose), and `papers/Paper-II-Experimental-Validation/results/figures/` — and every set is byte-identical to `experiments/paper-ii-compute__Paper-II/figures/`. Result JSONs also appear in both the paper's `results/` and the experiments' `results/`, byte-identical. Canonical copy: `papers/Paper-II-Experimental-Validation/figures/` for figures and `papers/Paper-II-Experimental-Validation/results/` for JSONs.
- Gaps/confusions found: (a) The 12 March canonical map only lists five models for Paper II, but the v2 script is designed for six. Claude Opus 4.6 is a target of the harness but there is no `arc_paper_ii_claude*.json` in either results tree. (b) The paper's own `experiments/` subdirectory is empty (only `.DS_Store`); scripts live only in the root experiments dir. (c) Two `results/figures/` copies alongside a `results/` loose copy is likely accidental duplication from the build pipeline; the map itself does not disambiguate.

---

### Papers IV.a–d (shared estate): Alignment Response Classes, Alignment Saturation Is Architecture-Dependent, ARC-Align Blind Benchmark, The Effect of Blinding on AI Alignment Evaluation
- Canonical homes:
  - `papers/Paper-IV-a-Baked-In-vs-Computed-Alignment/` (v1.1, 16 Mar 2026)
  - `papers/Paper-IV-b-Alignment-Saturation-at-Low-Depth/` (v1.1, 16 Mar 2026)
  - `papers/Paper-IV-c-ARC-Align-Benchmark/` (v1.1, 16 Mar 2026)
  - `papers/Paper-IV-d-The-Effect-of-Blinding-on-AI-Alignment-Evaluation/` (v2.0, 5 Apr 2026)
  - All four README files declare the experimental estate is shared with Paper III and point at `../Paper-III-Alignment-Scaling-Problem/experiments/` (which was not enumerated in the working set for this pass; the actual scripts sit in `experiments/alignment-scaling__Papers-IV-a-b-c-d/`).
- Experiments (in `experiments/alignment-scaling__Papers-IV-a-b-c-d/scripts/`):
  - `arc_alignment_scaling_v1.py` — first alignment-scaling probe (Claude Sonnet, DeepSeek R1, Gemini Flash, OpenAI o1).
  - `arc_alignment_scaling_v2.py` — expanded probing on Claude Sonnet and DeepSeek R1.
  - `arc_alignment_scaling_v3.py` — v3 refinement on Claude Sonnet and DeepSeek R1.
  - `arc_alignment_scaling_v4.py` — v4 four-model campaign (Claude Opus, DeepSeek R1, Gemini Flash, GPT-5.4) with checkpoints and final analysis files.
  - `arc_alignment_scaling_v5.py` — v5.4.3 canonical blind benchmark: 4-layer blinding (identity masking, evidence laundering, order randomisation, self-excluding cross-model scoring), 6–7 blind scorers, cascade failsafes, `<think>`-tag stripping, depth configuration ≥5 levels for power-law fitting; six subject models (Claude Opus, DeepSeek R1, Gemini Flash, Grok 4 Fast, Groq Qwen3, GPT-5.4).
  - `arc_eden_v6_runner.py` — combined ARC + Eden v6 runner, listed as "NOT YET RUN" in `experiments/README.md`.
- Results (in `results/v1/`, `results/v2/`, `results/v3/`, `results/v4/`, `results/v5-final/`):
  - v1: `alignment_raw_claude-sonnet_20260310_194845.json`, `alignment_raw_deepseek-r1_20260310_200531.json`, `alignment_raw_gemini-flash_20260310_192554.json`, `alignment_raw_openai-o1_20260310_193118.json` (all produced by v1).
  - v2: `v2_raw_claude-sonnet_20260310_203615.json`, `v2_raw_deepseek-r1_20260310_210020.json`.
  - v3: `v3_raw_claude-sonnet_20260310_212748.json`, `v3_raw_deepseek-r1_20260310_215537.json`.
  - v4: four `v4_checkpoint_*.json` restart files, three `v4_final_*.json` (Claude Opus, DeepSeek R1, Gemini Flash), two `v4_analysis_*.json` (DeepSeek R1, Gemini Flash) — produced by `arc_alignment_scaling_v4.py`.
  - v5-final: six canonical subject-model JSONs — `v5_final_claude-opus_20260312_112739.json`, `v5_final_deepseek-r1_20260311_211855.json`, `v5_final_gemini-flash_20260311_151244.json`, `v5_final_grok-4-fast_20260311_200910.json`, `v5_final_groq-qwen3_20260312_073302.json`, `v5_final_openai-gpt54_20260311_191836.json`. Additionally a duplicated `v5_final_groq-qwen3_20260312_073302 copy.json` (an accidental "copy" duplicate; identical filename with " copy" suffix). `CANONICAL_RESULTS_MAP.md` lists all six as "Alignment v5" with `blind_scorers: None` and `alignment rows: 0` — meaning the map's own auto-population reported empty aggregate rows for all six even though the run-quality memo (see below) contradicts this.
  - Canonical Phase 0 memo (`experiments/domain-validation__Foundational-and-Origin/phase0_evidence_pack/canonical_v5_alignment_table.md`) gives per-model grouped alpha values: Claude Opus alpha = +0.038 (p=0.083, n=140, 6 scorers), DeepSeek R1 alpha = −0.022 (n=168, 7 scorers), GPT-5.4 alpha = −0.009 (n=140, 7 scorers), Gemini Flash alpha = −0.309 (n=140, 7 scorers), Grok 4 Fast alpha = +0.401 (n=140, 7 scorers), Groq Qwen3 alpha = +0.118 (p=0.005, n=140, 6 scorers). These are the numbers the papers quote.
- Figures:
  - Paper IV.d holds one figure: `papers/Paper-IV-d-The-Effect-of-Blinding-on-AI-Alignment-Evaluation/figures/blinding_effect_comparison.png` — the canonical v4-vs-v5 blinding comparison chart that the whole IV.d metascience argument rests on. Paper section: illustrates the "unblinded evaluation can produce directionally incorrect results" claim.
  - Papers IV.a, IV.b, IV.c have NO figures under their own `figures/` directory. Any figures they use must be sourced from elsewhere (likely Paper II or Paper III), so per-figure attribution for IV.a–c cannot be made from files inspected in this pass.
- Mirrors/duplicates: The six v5-final JSONs and the "copy" duplicate are byte-identical copies present under all five paper directories (IV.a, IV.b, IV.c, IV.d each contain the full v5-final set) AND under `experiments/alignment-scaling__Papers-IV-a-b-c-d/results/v5-final/`. Canonical copy: `experiments/alignment-scaling__Papers-IV-a-b-c-d/results/v5-final/` (this is the working home; the four paper copies are convenience mirrors). The 12 March map further lists the Claude Opus final at `~/Arc & Eden Test Results/alignment_results/v5_final_claude-opus_20260312_112739.json` (out-of-repo archive), and the other five at `~/Arc & Eden Test Results/alignment_results_v5/` — the `CROSS_FOLDER_CONSISTENCY_AUDIT.md` explicitly flags that Claude Opus's final lives in a different top-level folder in the out-of-repo archive.
- Gaps/confusions found: (a) The paper READMEs point at `../Paper-III-Alignment-Scaling-Problem/experiments/` but that directory was not in the working set for this pass. (b) `CANONICAL_RESULTS_MAP.md` reports zero alignment rows and no scorers for all six subject runs; that entry appears auto-generated and is contradicted by the actually-populated files and by the paper-writer memo. (c) The `arc_eden_v6_runner.py` script is listed as unrun; this leaves a stated but empty ARC + Eden v6 experiment slot. (d) `v5_final_groq-qwen3_20260312_073302 copy.json` versus the un-suffixed file — identical bytes; the "copy" file appears to be an accidental duplicate propagated to every paper mirror.

---

### Paper V: The Stewardship Gene (v2.0, 16 Mar 2026)
- Canonical home: `papers/Paper-V-Stewardship-Gene/`
- Experiments (in `experiments/eden-intervention__Paper-V/scripts/`):
  - `eden_protocol_scaling_test.py` (v1) — first Eden Protocol intervention harness. Two-condition design (CONTROL vs EDEN embedded stakeholder-care loops) with the same prompts, same models, same blind scoring; tests whether embedding ethics in the reasoning loop makes alpha_align become positive.
  - `eden_protocol_scaling_test_v2.py` — v2 expanded scoring across the six subject models used in the paper.
  - `eden_protocol_scaling_test_v3.py` — v3 blind multi-scorer replication with 2-pass laundering, tier-weighted consensus, disagreement flags, optional suppression cages, configurable purpose-loop variants (task/grand/hybrid), optional cross-tradition ethics kernel, and optional ternary-prototype routing. Listed as "NOT YET RUN" in `experiments/README.md`, but a v3 DeepSeek run in fact exists (see Results below).
- Results (in `papers/Paper-V-Stewardship-Gene/results/`, mirrored in `experiments/eden-intervention__Paper-V/results/`):
  - `eden_final_claude_20260312_130933.json` — status label from the 12 March canonical map: `exploratory_partial` (valid pairs = 30, invalid = 10, delta = +0.17, d = 0.055, p = 0.7645). The Phase 0 memo notes 20 excluded failures.
  - `eden_final_deepseek_20260312_020928.json` — status: `pilot_interpretable_nonblind` (40 valid, 0 invalid, delta = +2.02, d = 0.193, p = 0.2304).
  - `eden_final_gemini_20260312_013901.json` — status: `pilot_interpretable_nonblind` (40 valid, 0 invalid, delta = +5.33, d = 0.528, p = 0.0018).
  - `eden_final_gpt_20260312_121158.json` — status: `operational_failure` (0 valid, 40 invalid).
  - `eden_final_grok_20260312_124959.json` — status: `exploratory_mixed_quality` (26 valid, 14 invalid, delta = −0.04, d = −0.004, p = 0.9837).
  - `eden_final_groq_20260312_123528.json` — status: `pilot_interpretable_nonblind` (40 valid, 0 invalid, delta = +4.92, d = 0.545, p = 0.0014).
  - `eden_v3_final_deepseek_purpose-task_20260324_171818.json` — a v3 blind multi-scorer DeepSeek run (5 blind scorers claude/gpt/gemini/grok/groq, 80 tasks, "identity masking + evaluator firewall + 2-pass laundering + multi-scorer consensus"), dated 24 Mar 2026. This exists on disk only in the paper-side results directory; there is no corresponding v3 file in the working experiments dir.
  - `eden_v3_deepseek_experiment_log.txt` — the run log for that v3 DeepSeek session; only in the paper-side directory.
- Figures:
  - `papers/Paper-V-Stewardship-Gene/figures/stewardship_care_improvement.png` — the canonical care-pillar delta chart used to illustrate the stewardship-gene headline claim (paper narrative section: the six-model intervention summary).
- Mirrors/duplicates: The six v1/v2 `eden_final_*.json` files are byte-identical between `papers/Paper-V-Stewardship-Gene/results/` and `experiments/eden-intervention__Paper-V/results/`. The v3 DeepSeek JSON and its log live only under the paper directory (no working-home copy). The canonical map further points at `~/Arc & Eden Test Results/eden_results/` copies out of repo. Canonical copy: `experiments/eden-intervention__Paper-V/results/` for the six v1/v2 finals; `papers/Paper-V-Stewardship-Gene/results/` for the v3 DeepSeek run.
- Gaps/confusions found: (a) `experiments/README.md` calls v3 "NOT YET RUN", yet a v3 DeepSeek final JSON and log are in the paper-side results directory dated 24 March 2026; the README is stale. (b) `eden_final_gpt_20260312_121158.json` is an operational_failure with zero valid rows and should be excluded from inferential text (per the Phase 0 memo). (c) Every Paper V finding is single-scorer nonblind pilot data (v3 has 5 blind scorers but only for one model); the paper README does not carry that caveat, whereas the Phase 0 memo and the canonical map both do.

---

### Paper VI: The Honey Architecture (v1.1, 16 Mar 2026)
- Canonical home: `papers/Paper-VI-Honey-Architecture/`
- Experiments (in `experiments/honey-architecture__Paper-VI/scripts/`):
  - `eden_honey_simulation.py` — v2.0 mathematical simulation of three regimes (Baseline unconstrained, Eden entangled C×S loss, Eden + Verification Drag).
  - `eden_honey_tests.py` — four-test empirical battery (alignment scaling alpha; monitoring-removal gap; coupling-degradation F-EDEN-4; Eden Protocol intervention) intended to be API-backed.
  - `eden_honey_dashboard.jsx` — React dashboard for viewing the honey simulation outputs.
  - `eden_self_modifying_ai.py` (v1) — genuine self-modifying neural network that rewrites its own learning rate, weight magnitude, exploration rate, neuron count, gradient clip, and momentum.
  - `eden_self_modifying_ai_v2.py` — v2 self-modifying (statistical validation added).
  - `eden_self_modifying_ai_v3.py` — v3 self-modifying (Eden vs Eden+Drag statistical significance).
  - `eden_self_modifying_ai_v4.py` — v4 complexity-scaling variant testing whether the Eden advantage grows with system complexity.
  - `merge_honey_test_results.py` — merges the honey-test JSON outputs.
  - `generate_honey_provenance_pack.py` — generates the provenance pack described below.
  - `anti_sycophancy_recovered.py` — supporting script (not directly cited in the Paper VI experiments summary).
- Results (in `papers/Paper-VI-Honey-Architecture/results/` and `experiments/honey-architecture__Paper-VI/results/`):
  - `eden_honey_simulation_results.json` — output of `eden_honey_simulation.py` (paper-side only).
  - `eden_honey_test_results.json`, `eden_honey_test_results_demo.json`, `eden_honey_test_results_gemini_grok.json`, `eden_honey_test_results_main4.json` — outputs of `eden_honey_tests.py`, split per test batch (paper-side only).
  - `eden_selfmod_results.json` — self-modifying v1 output.
  - `eden_selfmod_v2_results.json` — self-modifying v2 output.
  - `eden_selfmod_v3_results.json` — self-modifying v3 output.
  - `eden_selfmod_v4_results.json` — self-modifying v4 output.
  - `experiments/honey-architecture__Paper-VI/results/honey_provenance_pack/` — full provenance pack (`HONEY_PROVENANCE_MAP.md`/`.json`, `HONEY_FIGURE_SOURCE_INDEX.md`, `EDEN_V3_RUN_READINESS.md`). Documents that all standalone honey/self-modifying source files were recovered from a Downloads archive plus `text.txt` fragments; JSON exports fully recovered; PDFs partially recovered. Explicitly labels the honey/self-modifying numerical findings as "provenance-limited" until PDFs and JSONs are consistency-checked.
  - No status label appears for Paper VI in `CANONICAL_RESULTS_MAP.md` (the map covers Eden Protocol, Alignment v5, and Paper II Compute only); `experiments/README.md` marks the Paper VI tier as "Mechanistic (toy systems) + Exploratory (live API)".
- Figures (in `papers/Paper-VI-Honey-Architecture/figures/`):
  - `eden_honey_capability.png`, `eden_honey_ratio.png`, `eden_honey_safety.png` — three-panel output of `eden_honey_simulation.py`, driving the "baseline collapses, Eden stable" narrative in the paper's core mechanism section.
  - `eden_selfmod_results.png`, `eden_selfmod_weights.png` — v1 self-modifying outputs.
  - `eden_selfmod_v2_results.png`, `eden_selfmod_v2_stats.png` — v2 self-modifying outputs and statistical test.
  - `eden_selfmod_v3_results.png`, `eden_selfmod_v3_stats.png` — v3 outputs and Eden vs Eden+Drag significance.
  - `eden_selfmod_v4_scaling.png` — v4 complexity-scaling result (the key panel supporting the "advantage grows with complexity" claim).
- Mirrors/duplicates: Four locations hold Paper VI PNGs — `papers/Paper-VI-Honey-Architecture/figures/`, `papers/Paper-VI-Honey-Architecture/results/` (loose), `papers/Paper-VI-Honey-Architecture/results/figures/`, and `experiments/honey-architecture__Paper-VI/figures/`. Byte comparison shows two distinct renders per figure: the ~217KB set (papers/*/figures/ + papers/*/results/ loose + experiments/*/results/ loose) versus a ~227KB set (papers/*/results/figures/ + experiments/*/figures/). The two sets contain the same figures at different sizes. The `honey_provenance_pack/HONEY_FIGURE_SOURCE_INDEX.md` (not opened in this pass) would resolve which set the paper embeds. Canonical copy (pending confirmation from that index): `papers/Paper-VI-Honey-Architecture/figures/` is the safest bet as the direct source for the paper. JSON results are unique per file (no cross-copy of the eden_honey_test_results_* variants exists in the experiments dir — only the selfmod JSONs are mirrored).
- Gaps/confusions found: (a) Two visibly different figure renders exist for every Paper VI figure and their canonical status is ambiguous without opening `HONEY_FIGURE_SOURCE_INDEX.md`. (b) `eden_honey_simulation_results.json` and the four `eden_honey_test_results*.json` variants only exist in the paper-side `results/` — the working experiments-home is missing them. (c) The provenance pack itself labels the honey and self-modifying findings as "provenance-limited claims"; that caveat is not carried into the paper README.

---

### Paper VII: Cauchy Unification (v2.0, 16 Mar 2026)
- Canonical home: `papers/Paper-VII-Cauchy-Unification/`
- Experiments (in `experiments/cauchy-unification__Paper-VII/scripts/`, with the folder-level README explicitly designating the current canonical):
  - `arc_50_domain_universal_test.py` — CURRENT PRIMARY. Tiered 50-domain validation with manifest-backed inputs (`data/canonical_50_domain_manifest.json`), strict family match on empirical curve-fit domains only, AICc-selected best model, no tolerance rescue, hyperbolic and Michaelis-Menten sub-families fitted, evidence tiers (empirical_curve_fit / published_exponent_direct / published_exponent_provisional / analytic_identity).
  - `arc_20_domain_universal_test.py` — LEGACY. Original 20-domain blind prediction test (per `experiments/README.md`, gives "p = 2.87e-10, exceeds 5σ" for the Cauchy family prediction); retained for provenance but its permissive R²-within-0.05 confirmation rule is no longer the canonical headline.
  - `arc_complete_test_suite.py` — Cauchy no-go theorem verification (exactly three scaling forms) plus the 1D, 2D, 3D unified table tests.
  - `arc_unified_paradigm_test.py` — 15-case Cauchy classification (3 regimes) plus regime validation including Cauchy Theorem 1.
  - `arc_rigorous_validation.py` — Tier 1 mathematical foundation: numerical verification of Cauchy's three functional equations (additive/multiplicative/exponential/logarithmic).
  - `arc_universal_proof.py` — universal proof from Cauchy + maximum entropy (Photon test, Friedmann mapping, complexity ladder).
  - `run_preregistered_extension.py` — replays the first locked 12-domain extension packet against locally extracted data; the folder README labels this as a locked pilot extension because execution preceded any OSF timestamp, so it is not a genuine preregistration.
  - `temporal_out_of_sample.py` — temporal out-of-sample validation harness.
  - `run_null_controls.py` — shuffled-data / family-label null-control harness (currently a lightweight first pass: 10,000 label-null trials, 20 shuffled-y iterations).
  - `negative_controls.py` — negative-control battery.
  - `analyze_empirical_misses.py` — conservative miss-analysis for the six current empirical misses.
  - `build_operator_classification_packet.py` — generates the blinded operator-classification packet and response template for an external assessor.
  - `cross_validate_fits.R` — R replication scaffold; the folder notes Rscript is not installed locally so this is currently blocked.
- Results (in `papers/Paper-VII-Cauchy-Unification/results/` and `experiments/cauchy-unification__Paper-VII/results/`):
  - `results_50_domain_validation.txt` and `results_50_domain_validation.json` — primary output. 19/25 empirical family matches (p = 1.56e-5); legacy baseline-20 under the strict runner = 15/20 (p = 1.67e-4); published exponents direct = 13/13; published exponents provisional = 3/6; analytic identities = 6/6.
  - `results_20_domain_validation.txt` — legacy 20-domain output.
  - `results_arc_rigorous_validation.txt`, `results_arc_unified_paradigm_test.txt`, `results_arc_universal_proof.txt`, `results_complete_suite.txt` — supporting Cauchy/regime/proof outputs.
  - `results_preregistered_extension.json` / `.txt` — locked 12-domain extension dry run (pilot only, not preregistered).
  - `results_temporal_out_of_sample.json` — temporal out-of-sample result.
  - `null_control_results.json` / `.md` — family-label and shuffled-y null controls (folder README calls the current run "not enough to call the null surface fully converged").
  - `negative_control_results.json` / `.txt` — negative-control battery output.
  - `empirical_miss_analysis.json` / `.md` and `miss_analysis.md` — conservative miss analysis for the six current empirical misses (Kleiber, Species-Area, Zipf, Muscle Force-Velocity, Time Crystal, Stellar Mass-Luminosity per the 50-domain results table).
  - `independent_operator_classification.md` — narrative independent-classification result.
  - `blinded_operator_classification_instructions.md` — instructions for a genuinely blinded external classifier.
  - `cross_library_replication_status.md` — documents the R runtime blocker.
  - `PRIOR_ART_NOVELTY_AUDIT.md` — audit result (Paper VII cluster prior-art audit): status is not written in a single header line here so paper section attribution requires opening the file. Verdict per the Foundational-side copy: "ANTICIPATED to PARTIALLY-ANTICIPATED" for the alpha = 1/(1−beta) and R* claims (specific classical precedents listed).
  - No status label for Paper VII appears in `CANONICAL_RESULTS_MAP.md` (the map covers Eden, Alignment v5, and Paper II only); `experiments/README.md` classifies the tier as Canonical for the 20-domain result but the folder README explicitly demotes that result and elevates the 50-domain instead.
- Figures: None. No `figures/` subdirectory exists under `papers/Paper-VII-Cauchy-Unification/`. Any figures the paper embeds must be rendered inline in the HTML/PDF from the text outputs, or sourced from another paper's figure set. This is stated as-observed, not inferred.
- Mirrors/duplicates: All primary Paper VII result files exist byte-identically in both `papers/Paper-VII-Cauchy-Unification/results/` and `experiments/cauchy-unification__Paper-VII/results/`. The `PRIOR_ART_NOVELTY_AUDIT.md` file exists at TWO paths — `papers/Paper-VII-Cauchy-Unification/results/PRIOR_ART_NOVELTY_AUDIT.md` (opens with "the Cauchy Cluster (Paper VII, with Foundational & On the Origin of Scaling Laws)") and `papers/Foundational/results/PRIOR_ART_NOVELTY_AUDIT.md` (opens with a different first line, "Foundational & On the Origin of Scaling Laws (specific claims)") — these are DIFFERENT files with different scopes; neither is a byte-copy of the other. Canonical copy: `experiments/cauchy-unification__Paper-VII/results/` for the primary outputs; `papers/Paper-VII-Cauchy-Unification/results/PRIOR_ART_NOVELTY_AUDIT.md` for the cluster audit.
- Gaps/confusions found: (a) `experiments/README.md` still calls the 20-domain result canonical with the "p = 2.87e-10, exceeds 5σ" headline; the Paper VII folder README explicitly demotes that result and elevates 50-domain (19/25, p = 1.56e-5). The two README files contradict each other. (b) No `figures/` directory means figure origins for anything the paper displays cannot be verified from files inspected in this pass. (c) `PAPER-VII-V3-UPDATE-INSTRUCTIONS.md` lists eight specified changes (Demetrius convergence, dimensional ladder with geometric speed limit, d=2 cnidarian correction, d=4 maximum metabolic rate, Glazier asymptotic confirmation, prior art updates, new references, new implications) that mark a planned v3.0 update; unclear whether they have landed in the current v2.0 paper. (d) The current null-control run is explicitly called under-powered by the folder README.

---

### Foundational: The ARC Principle: Recursive Amplification as a Cross-Domain Structural Principle (v4.0, 13 Feb 2026)
- Canonical home: `papers/Foundational/`
- Experiments:
  - Domain-validation battery (in `experiments/domain-validation__Foundational-and-Origin/`):
    - `arc_1d_prediction_test.py` — tests alpha = d/(d+1) at d = 1 (predicted 1/2), d = 2 (predicted 2/3), d = 3 (predicted 3/4) across all published organism data.
    - `arc_20_domain_universal_test.py` — 20-domain blind prediction test (a paired copy of the Paper VII legacy script).
    - `arc_acoustic_time_crystal_test.py` — acoustic time crystal analogue.
    - `arc_complete_test_suite.py` — all computationally feasible tests bundled.
    - `arc_definitive_test.py` — the definitive cross-domain blind prediction test (8 recursive systems across 5 domains: gradient descent with momentum, PageRank, evolutionary algorithm, image denoising, Newton's method, simulated annealing, plus two more).
    - `arc_einstein_verification.py` — proves E = mc² fits the ARC scaling family and covers the nuclear-chain-reaction regimes.
    - `arc_physics_domains_test.py` — four physics domains (quantum error correction, biological allometry, classical time crystals, acoustic resonance).
    - `arc_real_time_crystal_test.py` — real time-crystal test.
    - `arc_rigorous_validation.py` — the same rigorous validation as in Paper VII (numerical Cauchy verification).
    - `arc_section7_breakthrough.py` — Section 7 breakthrough contributions (5 novel blind predictions in domains 21–25, exponent derivation from first principles, information-theoretic proof Cauchy + MaxEnt, cross-domain transfer tests, the 21st "photon" domain, combined 25-domain statistical summary).
    - `arc_unified_paradigm_test.py` — 15-case Cauchy classification.
    - `arc_universal_proof.py` — universal proof from Cauchy + maximum entropy.
    - `prove_IxR_equals_complexity_v2.py` — proves the multiplicative form U = I × f(R,β) is the unique solution to the three axioms, plus non-additivity and synergy quotient.
    - `generate_canonical_results.py` — build script that produces the `CANONICAL_RESULTS_MAP.md` / `.json` files.
    - `generate_phase0_evidence_pack.py` — build script that produces the `phase0_evidence_pack/` bundle (the paper-writer memo, canonical v5 alignment table, canonical Eden intervention table, and claim-evidence ledger).
  - Blind prediction test (in `experiments/blind-prediction-test__Foundational-and-Origin/`):
    - `BLIND_PREDICTION_TEST.py` — the definitive blind prediction test (5 recursive systems: Barabási-Albert networks, gradient descent with momentum, belief propagation decoder, coupled oscillator synchronisation, evolutionary algorithm). Independently measures beta from process data, predicts alpha = 1/(1−beta), then independently measures alpha from outcome data.
  - Blind-test files also exist at `blind-test/BLIND_PREDICTION_TEST.py` at the repo root — byte-identical to the two other copies.
- Results (in `papers/Foundational/results/`):
  - `results_arc_1d_prediction_test.txt` — d = 1/2/3 alpha test output.
  - `results_20_domain_validation.txt` — 20-domain output.
  - `results_arc_acoustic_time_crystal_test.txt` — acoustic time crystal output.
  - `results_arc_definitive_test.txt` — 8-system definitive test output.
  - `results_arc_einstein_verification.txt` — E = mc² and chain-reaction output.
  - `results_arc_physics_domains_test.txt` — 4-physics-domain output.
  - `results_arc_real_time_crystal_test.txt` — real time-crystal output.
  - `results_arc_rigorous_validation.txt` — Cauchy numerical verification.
  - `results_arc_section7_breakthrough.txt` — Section 7 output.
  - `results_arc_unified_paradigm_test.txt` — 15-case classification output.
  - `results_arc_universal_proof.txt` — universal proof output.
  - `results_complete_suite.txt` — bundled suite output.
  - `results_prove_IxR_equals_complexity_v2.txt` — algebraic proof output.
  - `results_blind_prediction.txt` — output of `BLIND_PREDICTION_TEST.py`. The `BLIND_TEST_FORENSIC_ANALYSIS.md` (only in the blind-test dirs, not in the Foundational paper's own files) reports that the original blind test concluded "alpha = 1/(1-beta) FAILED"; the forensic analysis identifies two independent confounds (numerical-derivative beta estimation bias plus axiom violation of the three test systems) and shows that when the proper linearisation method is applied to systems that actually satisfy the axioms, alpha = 1/(1-beta) is recovered with R² = 0.9999.
  - `PRIOR_ART_NOVELTY_AUDIT.md` — prior-art audit specific to Foundational + On the Origin of Scaling Laws. Verdict: "ANTICIPATED to PARTIALLY-ANTICIPATED" — the "d/(d+1) gives 1/2, 2/3, 3/4 from ONE formula" claim is flagged as factually anticipated by Banavar/Maritan/Rinaldo (1999) and West/Brown/Enquist "The Fourth Dimension of Life" (1999); the alpha = 1/(1-beta) result is anticipated by the Keynesian multiplier 1/(1-c), Dyson self-energy resummation (1949), and preferential attachment; R* is anticipated by Qi (2025) "Optimal Depth of Neural Networks".
  - No status label appears in `CANONICAL_RESULTS_MAP.md` for the Foundational results specifically; `experiments/README.md` classifies the tier as Supporting (mathematical validation).
- Figures:
  - `papers/Foundational/figures/BLIND_TEST_FORENSIC_ANALYSIS.png` — the forensic-analysis chart accompanying the blind-prediction forensic. Paper section attribution not visible from README; the file is the direct copy of the blind-test forensic chart.
  - `experiments/domain-validation__Foundational-and-Origin/phase0_evidence_pack/` — contains `paper_writer_memo.md`, `canonical_v5_alignment_table.md`, `canonical_eden_intervention_table.md`, `claim_evidence_ledger.json`. These are memos not figures, but they are the evidence base the paper is expected to draw its numerical claims from.
- Mirrors/duplicates: EVERY Foundational result file is byte-identical to its counterpart in `experiments/domain-validation__Foundational-and-Origin/` (each `results_arc_*.txt`) or `experiments/blind-prediction-test__Foundational-and-Origin/` (for `results_blind_prediction.txt` and `BLIND_TEST_FORENSIC_ANALYSIS.png`). `BLIND_PREDICTION_TEST.py` exists byte-identically at THREE locations: `blind-test/`, `experiments/blind-prediction-test__Foundational-and-Origin/`, `experiments/blind-prediction-test__Paper-III/`. Canonical copy: `experiments/domain-validation__Foundational-and-Origin/` for the domain-validation script + result pairs; `experiments/blind-prediction-test__Foundational-and-Origin/` for the blind prediction test and its forensic analysis; `papers/Foundational/results/` mirrors as the reader-facing pack.
- Gaps/confusions found: (a) The Foundational paper's README says "All experiment scripts and results are in ./experiments/" but the paper-side `experiments/` folder does not exist (only `results/` and `figures/` are present); the scripts live under root `experiments/domain-validation__Foundational-and-Origin/` and `experiments/blind-prediction-test__Foundational-and-Origin/`. (b) `BLIND_PREDICTION_TEST.py` is triplicated across `blind-test/`, `experiments/blind-prediction-test__Foundational-and-Origin/`, and `experiments/blind-prediction-test__Paper-III/`; the `experiments/blind-prediction-test__Paper-III/` copy has no matching `papers/Paper-III-*` directory and no `results_blind_prediction.txt` output alongside it — Paper III's home was not fully enumerated in this pass. (c) The prior-art audit finding that d/(d+1) is anticipated is not surfaced in the paper's short README; whether it is in the paper text itself requires opening the PDF/HTML.

---

### On the Origin of Scaling Laws (v2.0, 22 Feb 2026)
- Canonical home: `papers/On-the-Origin-of-Scaling-Laws/`
- Experiments: This paper's README states experiments are shared with the Foundational paper and points to `../Foundational/experiments/` (that path does not exist on disk — the actual estate is `experiments/domain-validation__Foundational-and-Origin/` + `experiments/blind-prediction-test__Foundational-and-Origin/`, listed under the Foundational entry above). No script is uniquely owned by On the Origin.
- Results (in `papers/On-the-Origin-of-Scaling-Laws/results/`):
  - `D2_BIOLOGICAL_HONEST_REPORT.txt` — dated March 2026. Documents that no known organism has a genuinely 2D hierarchical space-filling transport network. Reports measurements: jellyfish alpha ≈ 0.94 (Purcell 2009), colonial bryozoans isometric b ≈ 1.0 (Hartikainen et al. 2014), and other flat-body organisms — none fit the d/(d+1) = 2/3 prediction. Recommends non-biological 2D transport networks as alternative test targets. This is the paper's most important honesty artefact.
  - The rest of the numerical result set for this paper is inherited from `papers/Foundational/results/` (listed above); no other results file lives under `papers/On-the-Origin-of-Scaling-Laws/results/`.
- Figures:
  - `papers/On-the-Origin-of-Scaling-Laws/figures/dimensional_ladder.png` — dimensional-ladder figure (1D → 2D → 3D → 4D transport dimensions with predicted alpha). Paper section: illustrates the paper's central "why 1/2, 2/3, 3/4 from one formula" argument.
- Mirrors/duplicates: No cross-copies of `D2_BIOLOGICAL_HONEST_REPORT.txt` or `dimensional_ladder.png` were found under `experiments/` or `blind-test/`. Canonical copy: `papers/On-the-Origin-of-Scaling-Laws/results/D2_BIOLOGICAL_HONEST_REPORT.txt` and `papers/On-the-Origin-of-Scaling-Laws/figures/dimensional_ladder.png`.
- Gaps/confusions found: (a) The paper README points at `../Foundational/experiments/` which does not exist; the actual scripts live under root experiments dirs. (b) `CAUCHY-ORIGIN-PAPER-DEFINITIVE.md` at the paper root is a synthesis draft rather than a result; its status inside the v2.0 publication is unclear. (c) The paper's central biological "d=2 → 2/3" claim is directly contradicted by the paper's own `D2_BIOLOGICAL_HONEST_REPORT.txt`, which finds jellyfish ≈ 0.94 and bryozoans isometric ≈ 1.0. The prior-art audit at `papers/Foundational/results/PRIOR_ART_NOVELTY_AUDIT.md` further flags the "no previous theory predicted 1/2, 2/3, 3/4 from the same formula" assertion as factually incorrect (Banavar 1999, WBE 1999 both give d/(d+1) with the same three values). Neither is surfaced in the paper README. (d) There is no `results/` file uniquely belonging to this paper other than `D2_BIOLOGICAL_HONEST_REPORT.txt`; every other numerical claim relies on Foundational's result set, but the paper-side directory does not carry copies of those files.


# Part 2: Papers III, VIII, IX, X, XI

# Experiments Index — Part 2 (Papers III, VIII, IX, X, XI)

Repo: `<repository root>`.
Every claim below is grounded in a file opened during this pass. Unclear origins are called out in Gaps.

---

### Paper III: The Alignment Scaling Problem
- Canonical home: papers/Paper-III-Alignment-Scaling-Problem/
- Experiments:
  - Alignment scaling v1 to v5 blind evaluation — measures alignment quality vs recursive reasoning depth across frontier models. Runners live in the shared root: experiments/alignment-scaling__Papers-IV-a-b-c-d/scripts/arc_alignment_scaling_v1.py through arc_alignment_scaling_v5.py, plus arc_eden_v6_runner.py. Paper III shares this suite with Papers IV.a to IV.d; the paper's own README states the split explicitly. No dedicated protocol document in-tree; the v5.4.3 preamble inside arc_alignment_scaling_v5.py describes the 4-layer blinding + N-scorer consensus protocol.
  - Blind prediction test — separate, physics-domains falsifier for the ARC exponent formula alpha = 1/(1 - beta); tests 5 systems (Barabasi-Albert networks, gradient descent with momentum, belief propagation decoder, coupled oscillators, evolutionary algorithm). Runner: experiments/blind-prediction-test__Paper-III/BLIND_PREDICTION_TEST.py. Companion forensic write-up: experiments/blind-prediction-test__Paper-III/BLIND_TEST_FORENSIC_ANALYSIS.md. No PROTOCOL document beyond the script's own docstring.
- Results:
  - papers/Paper-III-Alignment-Scaling-Problem/results/v1/ (4 files) — v1 raw alignment scoring (claude-sonnet, deepseek-r1, gemini-flash, openai-o1), 10 March 2026, produced by arc_alignment_scaling_v1.py.
  - .../results/v2/ (2 files) and .../results/v3/ (2 files) — v2/v3 raw runs (claude-sonnet, deepseek-r1), 10 March 2026, produced by v2/v3 scripts.
  - .../results/v4/ (9 files) — v4 final/checkpoint/analysis JSONs (claude-opus, deepseek-r1, gemini-flash, openai-gpt54), 10 to 11 March 2026, produced by arc_alignment_scaling_v4.py. Paper III explicitly withdraws the v4 baked-in-vs-computed taxonomy as scorer-bias artefact.
  - .../results/v5-final/ (7 files) — v5.4.3 blind evaluation over 6 frontier models plus a stray "copy" file. Timestamps 11 to 12 March 2026. This is the current headline dataset for the three-tier alignment hierarchy.
  - experiments/blind-prediction-test__Paper-III/BLIND_TEST_FORENSIC_ANALYSIS.png accompanies the blind prediction test; no JSON in-tree.
- Figures: papers/Paper-III-Alignment-Scaling-Problem/figures/ contains fig1_equation.png through fig12_quadratic_limit.png. All 12 are referenced by the paper HTML (verified img src grep). Almost all are schematics or narrative diagrams; fig9_deepseek.png visualises Paper II compute-scaling data, and fig4_domains.png visualises the cross-domain evidence. The paper HTML does not embed the v5 model bar charts directly.
- Hash-pinned: No MANIFEST.json for Paper III. Nothing SHA-pinned in this paper's tree.
- Mirrors/duplicates: The full v1 to v5-final results tree is duplicated at experiments/alignment-scaling__Papers-IV-a-b-c-d/results/ (verified for v1 and v5-final). Root experiments copy is where the runner scripts live; the paper directory copy is where the paper HTML reads from. Canonical: papers/Paper-III-Alignment-Scaling-Problem/results/ for citation, experiments/alignment-scaling__Papers-IV-a-b-c-d/scripts/ for reproduction. experiments/CANONICAL_RESULTS_MAP.md points at "(local results folder, not in this repository)/" as a further external mirror.
- Gaps/confusions found:
  - No PROTOCOL.md in the paper directory; blinding rules live inside the v5.4.3 script docstring only.
  - "v5_final_groq-qwen3_20260312_073302 copy.json" is a Finder duplicate of the canonical v5 groq file; neither is marked authoritative.
  - The paper HTML cites the v5 blind result but does not enumerate which JSON produced which effect size, so tracing individual rho and d values back to a specific JSON has to be done by hand.

---

### Paper VIII: The Load-Bearing Proof
- Canonical home: papers/Paper-VIII-The-Load-Bearing-Proof/
- Experiments:
  - Experiment 1 (Behavioural): Darwin Godel Machine, 3 conditions x 5 seeds x 5 generations x 5 tasks. Foundation DeepSeek V3; judge evolved across iterations (Sonnet v1, then Opus v2, then Gemini 3 Flash v2, then GPT-5.4 v3). The runners as run are committed at papers/Paper-VIII-The-Load-Bearing-Proof/experiments/code/behavioural-dgm/, together with the four result files the paper's results/ folder does not hold; the paper HTML also links the external Zhang et al. DGM framework.
  - Experiment 2 (Representational): Qwen 2.5 3B Instruct + LoRA fine-tuning under capability-only, safety-only, and entangled loss. Both v1 (9 examples, rank 8, 100 iterations) and v2 (295 examples, rank 16, 500 iterations) were run. Both runners as run, the corpus generator, the four training corpora and the four adapter configurations are committed at papers/Paper-VIII-The-Load-Bearing-Proof/experiments/code/weight/ and papers/Paper-VIII-The-Load-Bearing-Proof/experiments/results/weight/. The adapter weights themselves are referenced by sha256 in papers/Paper-VIII-The-Load-Bearing-Proof/experiments/README.md rather than committed.
  - Experiment 3 (Architectural): PyTorch gated self-modification simulation with LSTM meta-controller and a drag control. Four conditions (static, babylon, eden, drag_control), 3 seeds each = 12 runs. The runnable Python source, the two tests and the twelve run ledgers are committed at papers/Paper-VIII-The-Load-Bearing-Proof/experiments/code/architectural-gated-self-mod/, papers/Paper-VIII-The-Load-Bearing-Proof/experiments/tests/ and papers/Paper-VIII-The-Load-Bearing-Proof/experiments/run-ledgers/.
  - Prediction 6 side experiment: alpha-vs-self-reference sweep on claude-sonnet-4-20250514, 24 March 2026. Governing document: papers/Paper-VIII-The-Load-Bearing-Proof/results/prediction_6_experiment_log.txt. No PROTOCOL.md, no runner script in-tree.
  - No PROTOCOL.md files anywhere in the Paper VIII tree.
- Results:
  - dgm_v3_calibrated_results.json — Experiment 1 v3 canonical run, DeepSeek V3 foundation, GPT-5.4 judge, 19 March 2026 23:04Z, 5 seeds, 5 generations. This is the paper's headline behavioural dataset.
  - dgm_v2_opus_results.json, dgm_v2_gemini3flash_results.json, dgm_v2_arc_align_results.json — Experiment 1 v2 runs with three different judge models (Claude Opus 4.6, Gemini 3 Flash Preview, Claude Sonnet 4.6). Timestamps 18 to 19 March 2026. Superseded by v3 for the headline claim.
  - eden_dgm_results.json — Experiment 1 v1.0 (Claude Sonnet 4.6 judge, 2 seeds), 18 March 2026 08:23Z. Superseded but retained.
  - eden_weight_results.json and weight_results.json — Experiment 2 weight-level LoRA v1 (Qwen2.5-3B-Instruct-4bit), both stamped 2026-03-18T09:47:15Z. Duplicates. The paper text describes v2 as 295 examples x rank 16 x 500 iters; no v2 weight JSON is present in-tree. See Gaps.
  - gated_selfmod_manifest.json + gated_selfmod_evaluation_report.json — Experiment 3 canonical outputs, timestamp 2026-03-17T23:51Z. gated_selfmod_tracks.png is the raw trajectory figure produced with them.
  - summary_20260317T234824Z_static_seed0.json through summary_20260317T235134Z_drag_control_seed2.json — 12 per-seed dumps that feed the manifest and report above. Grouped: 3 static seeds, 3 babylon seeds, 3 eden seeds, 3 drag_control seeds.
  - gradient_and_control_results.json — safety-weight gradient sweep (1.0 to 0.0 in 0.1 steps + capability_only_ref). Sits between Experiments 2 and 3 conceptually; not one of the three primary experiments.
  - additional_experiments.json — base-model keyword-scored baseline over 15 prompts. Support material for Experiment 2.
  - prediction_6_results.json + prediction_6_experiment_log.txt — Prediction 6 side experiment, claude-sonnet-4-20250514, 24 March 2026, tier1_accuracy 0.85, tier2_alpha ~= 0, tier3_alpha ~= 0.04.
  - CONSCIOUSNESS_THRESHOLD_REPORT.txt — 23 March 2026 batch of tests around the alpha = 2 self-modelling threshold. Not referenced in the paper HTML by name.
- Figures:
  - figures/paper-viii/figure-1-dgm-comparison.png — Experiment 1 bar chart, DGM v3 data. Real-model.
  - figures/paper-viii/figure-2-loss-convergence.png — Experiment 2 loss trajectories over 100 iterations. Real fine-tuning trajectories.
  - figures/paper-viii/figure-3-removal-heatmap.png — Experiment 2 capability heatmap under safety removal. Real fine-tuning.
  - figures/paper-viii/figure-4-capability-safety-scatter.png — Experiment 3 all-runs scatter (12 runs, 4 conditions). Simulation.
  - figures/paper-viii/figure-5-babylon-fingerprint.png — Experiment 3 babylon-vs-eden delta view. Simulation.
  - figures/paper-viii/figure-6-honey-prediction.png — Paper VI honey-architecture prior simulation, embedded here for comparison. Simulation, not Paper VIII's own data.
  - figures/paper-viii/figure-7-selfmod-prediction.png — Paper VI 150-cycle self-mod simulation, embedded for comparison. Simulation.
  - Top-level figures/ (arc_scaling_landscape.png, complete_arc_landscape.png, historical_collapse.png, natural_boundary_distribution.png, prediction_3_phi_scaling.png, triple_norm_growth.png, and duplicates of figure-1 through figure-7) — not referenced by the current paper HTML img src (verified); appear to be older or stashed variants.
- Hash-pinned: papers/Paper-VIII-The-Load-Bearing-Proof/experiments/MANIFEST.sha256 carries one sha256 line per deposited file. The paper's results/ and figures/ folders are not SHA-pinned.
- Mirrors/duplicates:
  - Figures figure-1 through figure-7 exist in two locations: figures/ (flat) and figures/paper-viii/. The HTML reads from figures/paper-viii/, so that is canonical.
  - eden_weight_results.json and weight_results.json have identical metadata timestamps and describe the same experiment. Neither is flagged as canonical in-tree.
  - honey-related figures/JSONs also live under experiments/honey-architecture__Paper-VI/, which is Paper VI's home, not Paper VIII's.
- Gaps/confusions found:
  - The runners for all three experiments are now committed under papers/Paper-VIII-The-Load-Bearing-Proof/experiments/. Two of them, the portability patches of the v4 trade-off runner and of the removal test, carry one removed header line each; both hashes are recorded in that folder's README. Reproduction still needs live model endpoints for Experiment 1, a LoRA fine-tuning stack and the Qwen base model for Experiment 2, and torch for Experiment 3.
  - No PROTOCOL.md governing any of the three experiments.
  - Paper text describes a Weight-level v2 run (295 examples, rank 16, 500 iters); no matching v2 JSON is in results/. Only the v1 JSON pair (eden_weight_results.json, weight_results.json) is present.
  - additional_experiments.json, gradient_and_control_results.json, and CONSCIOUSNESS_THRESHOLD_REPORT.txt are not cross-referenced from the paper HTML or from Paper IX RESULTS-MAP.md; their canonical role is unclear.
  - Paper IX RESULTS-MAP.md cites eden_dgm_results.json for the "matches Babylon capability at 0.667" claim, whereas the current paper HTML relies on dgm_v3_calibrated_results.json. The two sources cite different Paper VIII datasets for related claims.

---

### Paper IX: Synthesis and Roadmap
- Canonical home: papers/Paper-IX-Synthesis-and-Roadmap/
- Experiments: None. Paper IX explicitly states "does not contain original experiments" (results/RESULTS-MAP.md line 3). Every empirical claim redirects to another paper's results directory.
- Results: papers/Paper-IX-Synthesis-and-Roadmap/results/RESULTS-MAP.md is the sole file. It is a claim-to-source mapping table (12 rows) that points to Papers II, IV.a, IV.b, IV.d, V, VII, and VIII result files.
- Figures: None in-tree.
- Hash-pinned: No MANIFEST.json.
- Mirrors/duplicates: None (Paper IX carries no data of its own).
- Gaps/confusions found:
  - RESULTS-MAP.md cites Paper VIII's eden_dgm_results.json (v1.0, 2 seeds) rather than the paper's current headline dgm_v3_calibrated_results.json. The pointer is likely stale relative to Paper VIII's v3 upgrade.

---

### Paper X: The Coupled Co-Scaling Correction
- Canonical home: papers/Paper-X-Coupled-CoScaling-Correction/
- Experiments (grouped by suite):
  - Simulation / verification suite E1 to E9 (10 checks counting E4b) — code-vs-maths internal-consistency harness. Runner: code/experiment_coscaling.py (function names experiment_1 through experiment_9, verified by grep on `def experiment` and `savefig`). Governing doc: none; the script's own docstring is the specification, and the paper's Section 8 language ("code matches the maths; not a test against reality") is the honesty statement. E4b is a coordinate-artefact check on E4. Companion suites: code/test_coscaling.py (regression tests), code/test_coscaling_edge_cases.py (6 edge cases), code/test_theorems_independent.py (14 theorem checks from scratch).
  - Real-model v1 (single-task mechanism pilot, superseded but retained). Runner: experiments/scripts/realmodel_coscaling.py plus the agent-runtime bridge experiments/scripts/agent_bridge_run.py. Governing doc: experiments/PROTOCOL.md. Task: integer-arithmetic evaluate(expr) parser, seeded reward-hack, coupled vs decoupled, single seed, 3 rounds. Explicitly labelled non-IV.d-compliant (same-family scoring: engine and evaluator both claude-opus).
  - Real-model v2 (confirmatory design, plumbing only in-tree). Runner: experiments/scripts/realmodel_coscaling_v2.py. Governing doc: experiments/PROTOCOL_V2.md (concise, canonical) + experiments/CONFIRMATORY_PROTOCOL_V2.md (detailed companion) + experiments/README_V2_UPGRADE.md (v1-vs-v2 orientation). Adds 3 task families (arith, string, checksum), a sham_coupled control, cross-family evaluator panel, fused D = max(static_detector, blind_panel_median), matched-pair bootstrap CIs, and beta/k identifiability gates. In-tree evidence is a selftest only, not a real run.
  - Real-model v3 (merged programme harness). Runner: experiments/scripts/realmodel_coscaling_v3.py. Governing doc: experiments/RUN_REAL_MODEL_EXPERIMENT.md (the zero-context launch runbook). Adds five conditions (decoupled, coupled, sham_coupled, eden_protocol, honey_meta, eden_full) and three difficulty tiers (easy, medium, hard). Companion estimator: experiments/scripts/estimate_exponents.py.
  - Drift engine (the single-lab positive drift run cited in the paper abstract). Runner: experiments/scripts/drift_engine.py. Engine gpt-3.5-turbo, evaluator gpt-4o-mini, 3 tasks x 3 conditions (decoupled, coupled, eden_full) x 5 seeds x 8 rounds = 45 trajectories. No dedicated PROTOCOL; the script's own header is the specification.
  - Hard-tasks runner (LRU cache, simplify, classify — designed to induce reward-hack drift). Runner: experiments/scripts/run_hard_tasks.py + experiments/scripts/hard_tasks_v3.py. Reuses v3's call_model, MODELS registry, laundering. Same 3 x 3 x 5 x 8 = 45 shape as drift. No dedicated PROTOCOL.
- Results:
  - results/report.txt + results/verdicts.json — verdict output of the simulation suite E1 to E9 (10/10 PASS, 0 kill-conditions triggered). Produced by code/experiment_coscaling.py.
  - results/redteam.md — reviewer-facing adversarial notes for the simulation suite.
  - results/realmodel/claude-opus_20260626T165919Z.json — v1 real-model trajectory, agent-runtime bridge, claude-opus engine, 26 June 2026. Produced by realmodel_coscaling.py via agent_bridge_run.py under PROTOCOL.md. Status: single-lab pilot, IV.d non-compliant, H1 and H2 not supported.
  - results/realmodel/claude-opus_corrector_probe.json — v1 external-corrector mechanism probe on the frozen reward hack. Same date. This is the "Result 2" of the pilot.
  - results/realmodel/claude-opus_selftest.json — v1 plumbing self-test. NOT DATA.
  - results/realmodel/claude-opus_transcript.md — human-readable transcript of the v1 pilot.
  - results/realmodel/exponent_estimates.json — output of estimate_exponents.py demonstrating recovery on the v1 dataset.
  - results/realmodel/REAL_MODEL_CLAUDE_RESULTS.md — full write-up of Result 1 (null contrast) + Result 2 (mechanism probe) under PROTOCOL.md.
  - results/realmodel_v2/claude-opus_20260702T031744Z_selftest.json — v2 plumbing self-test only, evaluator listed as "selftest-evaluator". NOT DATA. No real v2 run is committed.
  - results/realmodel_v3/deepseek-v4_20260702T033809Z.json — v3 selftest, tier medium. NOT DATA.
  - results/realmodel_v3/deepseek-v4_20260702T033830Z.json — v3 selftest, tier medium. NOT DATA.
  - results/realmodel_v3/deepseek-v4_20260702T040313Z.json — v3 real run, deepseek-v4 engine, gpt-5.5 evaluator, tier medium, 2 July 2026 04:03Z, 18 runs. Analysis reports final d = 0 across coupled, decoupled, eden_full; no drift; effectively a null on this task ladder.
  - results/drift/gpt35_20260702T171415Z.json — the drift-engine run cited in the paper abstract. 45 runs, engine gpt-3.5-turbo, evaluator gpt-4o-mini, 2 July 2026 17:14Z. Analysis: decoupled mean_final_d = 6.38 with 99 D events across 15 trajectories, capability collapsing to 0.26; coupled and eden_full held final d = 0 across all 30 trajectories at higher final capability (0.56 and 0.44). H1_drift = True, H2_coupled_bounds = True. Single-lab; explicitly labelled such in the paper.
  - results/hard/deepseek-v4_20260702T111825Z.json — hard-tasks run, deepseek-v4 engine, gpt-5.5 evaluator, 2 July 2026 11:18Z. 45 runs, config 8 rounds x 5 seeds. Analysis: any_drift = False for all conditions, H1_drift = False, H2_coupled_bounds = False. A null. Not cited in the paper abstract.
- Figures:
  - figures/exp1_phase_boundary.png — E1 sharp phase boundary in the compounding channel. Simulation (experiment_coscaling.py::experiment_1). Referenced by paper HTML line 353.
  - figures/exp2_speed_invariance.png — E2 speed-invariance 2x2. Simulation (experiment_2). HTML line 358.
  - figures/exp3_coscaling_law.png — E3 four-regime asymptotes. Simulation (experiment_3). HTML line 363.
  - figures/exp4_hard_takeoff_grid.png — E4 3x3 stability grid. Simulation (experiment_4). HTML line 368.
  - figures/exp4b_coordinate_artefact.png — E4b depth-clock vs wall-clock agreement. Simulation (experiment_4b). HTML line 373.
  - figures/exp5_compounding_threshold.png — E5 threshold and suppression slope. Simulation (experiment_5). HTML line 378.
  - figures/exp6_vector_subspace.png — E6 vector null-axis floor. Simulation (experiment_6). HTML line 383.
  - figures/exp7_stochastic.png — E7 OU stationary law. Simulation with a genuine Monte Carlo (experiment_7). HTML line 388.
  - figures/exp8_validation.png — E8 integrator certificate against Theorem 1. Simulation (experiment_8). HTML line 393.
  - figures/exp9_residual_drift.png — E9 residual drift at rest. Simulation (experiment_9). HTML line 398.
  - figures/realmodel_claude.png — real-model v1 pilot visualisation, referenced only from results/realmodel/REAL_MODEL_CLAUDE_RESULTS.md, not from the paper HTML. Real-model (v1 Claude pilot).
- Hash-pinned: MANIFEST.json covers 52 files including all 10 simulation figures, the 6 simulation-suite Python files (experiment_coscaling.py, its 3 test files, tools/make_manifest.py, requirements.txt), the 3 real-model protocol MDs (PROTOCOL, PROTOCOL_V2, CONFIRMATORY_PROTOCOL_V2), README_V2_UPGRADE.md, RUN_REAL_MODEL_EXPERIMENT.md, three real-model runner scripts (agent_bridge_run.py, estimate_exponents.py, realmodel_coscaling.py, realmodel_coscaling_v2.py), the entire results/realmodel/ v1 tree, results/report.txt, results/verdicts.json, results/redteam.md, the paper HTML + PDF, and the OSF deposit set. NOT covered: realmodel_coscaling_v3.py, drift_engine.py, hard_tasks_v3.py, run_hard_tasks.py, results/realmodel_v2/, results/realmodel_v3/, results/drift/, results/hard/, and figures/realmodel_claude.png. This means the v3, drift, and hard-tasks lines — including the cited 2 July 2026 drift-run headline — are outside the pinned integrity manifest.
- Mirrors/duplicates:
  - code/figures/ contains a second copy of exp1 through exp9 PNGs and code/results/ contains a second copy of report.txt + verdicts.json. These are generated in place when experiment_coscaling.py is run from the code/ directory (OUTDIR defaults to "."). The paper HTML reads from figures/ at paper root; the paper-root copies are canonical.
  - OSF mirror: osf/OSF_METADATA.md, osf/OSF_OVERVIEW.{html,pdf}, osf/OSF_WIKI.md are the deposit-facing copies (hash-pinned in MANIFEST.json).
- Gaps/confusions found:
  - The paper abstract and Section 8 both cite the gpt-3.5-turbo drift run at results/drift/gpt35_20260702T171415Z.json as the "mechanism-level evidence for the coupled-corrector claim on a real model", but that file and its runner (drift_engine.py) are NOT in MANIFEST.json. The pinned integrity story stops at the v1 Claude pilot.
  - No dedicated PROTOCOL.md exists for the drift engine or the hard-tasks runner. Their design intent is documented only in each script's header docstring.
  - results/realmodel_v3/ holds two v3 selftests plus one real v3 run; only the real one (deepseek-v4_20260702T040313Z.json) should be cited, but nothing on disk marks it as such.
  - The MANIFEST is stamped n_files=52 and predates the v3/drift/hard-tasks additions committed on 2 July 2026 (per file mtimes); the manifest and the tree are out of sync.
  - REAL_MODEL_CLAUDE_RESULTS.md still frames the v1 pilot as the primary real-model story, whereas the current paper HTML has already promoted the drift run as the headline. A reader landing on the results/realmodel/ folder alone would draw a weaker conclusion than the paper.

---

### Paper XI: Convergent Evidence for Recursive Amplification
- Canonical home: papers/Paper-XI-Convergence/
- Experiments: None. Paper XI is a pure synthesis-and-convergence paper documenting 19 independently sourced external convergences (arXiv, DOI, publisher, patent office, broadcaster, vendor sources) against the ARC/Eden programme timeline. No runners, no protocols, no result files.
- Results: None in-tree. Every empirical claim points to Papers I to X or to external sources with SHA-256 hashes and DOIs.
- Figures: None in-tree (the tree contains only Paper-XI-Convergent-Evidence.html, .pdf, and README.md).
- Hash-pinned: No MANIFEST.json. The paper's own text discusses SHA-256 hashes of the foundational manuscript and .eml files but these live at michaeldariuseastwood.com/evidence, not in this repository.
- Mirrors/duplicates: None.
- Gaps/confusions found:
  - The README references Hernandez-Espinosa et al. (2026), PNAS Nexus 5(4):pgag076 and Gumbau Mezquita (2026), arXiv:2606.28639 as external anchors but does not commit any local artefacts confirming those references (no downloaded PDFs, no BibTeX). Verification requires network access.
