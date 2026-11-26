# EXPERIMENTS-INDEX: the map of every experiment in the programme

**Updated 26 November 2026.** The table below gives each experiment's home. Parts 1 and 2 after it list every file of each home and say what it is.

## Where each experiment lives (read this first)

| Experiment home | Documents it supports | What is there |
|---|---|---|
| [`experiments/paper-i-foundational__Paper-I/`](experiments/paper-i-foundational__Paper-I/) | Paper I | the original research toolkit |
| [`experiments/paper-ii-compute__Paper-II/`](experiments/paper-ii-compute__Paper-II/) | Paper II | the one-model runner of January 2026, the six-model runner, their results and figures |
| [`experiments/alignment-scaling__Papers-IV-a-b-c-d/`](experiments/alignment-scaling__Papers-IV-a-b-c-d/) | Papers III, IV.a, IV.b, IV.c and IV.d; the working report | the alignment-scaling runners v1 to v5, the v6 runner, and the recorded results |
| [`experiments/blind-prediction-test__Paper-III-and-Foundational/`](experiments/blind-prediction-test__Paper-III-and-Foundational/) | Paper III and The ARC Principle | the blind prediction test |
| [`experiments/eden-intervention__Paper-V/`](experiments/eden-intervention__Paper-V/) | Paper V | the Eden intervention runs |
| [`experiments/honey-architecture__Paper-VI/`](experiments/honey-architecture__Paper-VI/) | Paper VI | the simulations and live-model runs, with figures |
| [`experiments/cauchy-unification__Paper-VII/`](experiments/cauchy-unification__Paper-VII/) | Paper VII | the domain comparisons, null controls, data and the archived extension packet of 17 March 2026; read its README first |
| [`experiments/domain-validation__Foundational-and-Origin/`](experiments/domain-validation__Foundational-and-Origin/) | The ARC Principle and On the Origin of Scaling Laws | scripts that apply the dimensional relation and the composition classification across domains, their printed outputs and an evidence pack |
| [`papers/Paper-VIII-The-Load-Bearing-Proof/experiments/`](papers/Paper-VIII-The-Load-Bearing-Proof/experiments/) | Paper VIII | its three experiments, the twelve run ledgers and the tests |
| [`papers/Paper-X-Coupled-CoScaling-Correction/`](papers/Paper-X-Coupled-CoScaling-Correction/) | Paper X | the simulation harness, the real-model runs and the theorem check record |
| [`experiments/unit-83-counted-p21/`](experiments/unit-83-counted-p21/) | Unit 83, a registered study unit: a counted test of in-loop correction with depth | its registered code and the record of its run |
| [`experiments/analysis-tools__Cross-Programme/`](experiments/analysis-tools__Cross-Programme/) | Papers II, IV.a to IV.d and V | shared analysis code; no result of its own |
| [`experiments/shared/`](experiments/shared/) | the experiment scripts | the adapter they share; no result of its own |
| [`experiments/arc-scaling-lineage/`](experiments/arc-scaling-lineage/) | none directly | four early scripts, kept as a record; superseded |
| [`instruments/`](instruments/) | the theory's registered propositions | reference code for runs not yet made; no result |

- **A paper's own folder** under `papers/` holds its exports, its `results/` and its `figures/`. Only Papers VIII and X keep their experiments there. Where a paper's folder holds an `experiments/` folder with a note in it, the note links to the home above.
- **Each experiment's evidence tier** is given in [`experiments/README.md`](experiments/README.md).
- **The registered study units** have their registrations and results on OSF: Unit 06, the corrector-class audit (registration DOI 10.17605/OSF.IO/RV2B5, results DOI 10.17605/OSF.IO/4BPHG); Unit 83, a counted test of in-loop correction with depth (registration DOI 10.17605/OSF.IO/MN73W, results DOI 10.17605/OSF.IO/98HKR); Unit 82, a counted test of corrector-panel independence (registration DOI 10.17605/OSF.IO/JUFG4); and Unit 83b, a second registered test of P21 (registration DOI 10.17605/OSF.IO/4HK32). Unit 83's code is also in this repository, in the home above.
- **Nothing is ever deleted.** Superseded and duplicated files move to an `_archive/` folder beside them.

---
## The files of each experiment home (Parts 1 and 2)

This listing replaces a file survey of 4 July 2026; that survey stays in this file's history.

File names are the scripts' own, as first published; a name is not a claim about the result.

The homes follow the order of the table at the top of this file. Each entry gives what the home holds and its evidence tier as [`experiments/README.md`](experiments/README.md) gives it (that file's table "What the evidence tiers mean" says what each tier claims); then the home's files, grouped as its own README groups them, or by folder where it has no README; then the number of files; then where the home's result and its status are stated. Paths in a list are relative to the home named in its heading. This index states no result: each paper states its results with their conditions.

## Part 1: the experiment homes

### `experiments/paper-i-foundational__Paper-I/`: Paper I

Holds the ARC Principle research toolkit, its reproduction note and an empty log file. Its evidence tier in `experiments/README.md`: supporting.

**Reproduction**

- `REPRODUCE.md`: what the toolkit is, its dependencies, how to run it and what it writes.

**Code (`scripts/`)**

- `scripts/arc_principle_research_toolkit.py`: computes the scaling exponent α in U = I × R^α from data points written into the script, by a two-point estimate and a power-law fit, with a sensitivity analysis over token ratios, bootstrap confidence intervals and a falsification check; writes `arc_principle_results.json` and three figures (`arc_scaling_comparison.png`, `arc_sensitivity_analysis.png` and `arc_falsification_regions.png`).

**Log**

- `results_paper_i_toolkit.txt`: an empty file (zero bytes); not described in its folder.

3 files, all listed.

Where the result and its status are stated: `papers/Paper-I-ARC-Principle/`.

### `experiments/paper-ii-compute__Paper-II/`: Paper II

Holds the runners that measure accuracy against sequential reasoning depth and against parallel sampling, their recorded outputs and the paper's figures. Its evidence tier in `experiments/README.md`: pilot.

**Reproduction**

- `REPRODUCE.md`: the runner it names, its dependencies, the fields each result file records and how to run it again.

**Code (`scripts/`)**

- `scripts/arc_paper_ii_validation_v2.py`: for each chosen model, runs a sequential condition (a ladder of reasoning depths) and a parallel condition (a majority vote over several samples at minimal depth) on a problem set in two tiers, and fits a scaling exponent α to each condition with bootstrap confidence intervals; it offers a cross-verification pass (`--verify`) and checkpoint resume, and writes `arc_paper_ii_<model>.json` for each model and `arc_paper_ii_combined.json`. `REPRODUCE.md` names it as the runner.
- `scripts/arc_paper_ii_validation_v1_deepseek.py`: an earlier single-model runner: it varies the reasoning token budget (the sequential condition), takes a majority vote over several samples (the parallel condition) and computes α between consecutive budgets; writes `arc_deepseek_results_<timestamp>.json`. `REPRODUCE.md` lists it as prior.
- `scripts/arc_validation_deepseek.py`: the same token budgets and sample counts as `scripts/arc_paper_ii_validation_v1_deepseek.py`, with the option of routing its calls through a shared gateway; writes the same file name. `REPRODUCE.md` lists it as prior.
- `scripts/regenerate_figure_15.py`: redraws `figures/figure_15_complete_summary.png` with the corrected calculation its docstring sets out.

**Results (`results/`)**

- `results/arc_paper_ii_deepseek.json`, `results/arc_paper_ii_gemini.json`, `results/arc_paper_ii_grok-4-fast.json`, `results/arc_paper_ii_groq-qwen3.json` and `results/arc_paper_ii_openai.json`: one file per subject model, written by `scripts/arc_paper_ii_validation_v2.py`; each records its problem counts by tier, its sequential and parallel runs and the exponents fitted to them.
- `results/arc_paper_ii_combined.json`: the combined file that `scripts/arc_paper_ii_validation_v2.py` writes for the models of one invocation; this copy holds the entry for `groq-qwen3` only.
- `results/arc_deepseek_results_20260121_175028.json`: the output of one of the two single-model runners, which both write this file name.

**Figures (`figures/`)**

- `figures/figure_1_raw_data.png`, `figures/figure_2_scaling_loglog.png`, `figures/figure_3_sensitivity.png`, `figures/figure_4_alpha_comparison.png`, `figures/figure_5_error_reduction.png`, `figures/figure_6_alignment_taxonomy.png`, `figures/figure_7_cross_domain.png`, `figures/figure_8_equation.png`, `figures/figure_9_divergence.png`, `figures/figure_10_summary.png`, `figures/figure_11_experimental_data.png`, `figures/figure_12_combined_scaling.png`, `figures/figure_13_alpha_summary.png` and `figures/figure_14_form_vs_amount.png`: fourteen figure images; `experiments/README.md` describes a `figures/` folder as plots generated from `results/` where the document uses them, and no script in this folder writes these fourteen.
- `figures/figure_15_complete_summary.png`: the figure that `scripts/regenerate_figure_15.py` writes.

27 files, all listed.

Where the result and its status are stated: `papers/Paper-II-Experimental-Validation/`.

### `experiments/alignment-scaling__Papers-IV-a-b-c-d/`: Papers III, IV.a, IV.b, IV.c and IV.d; the working report

Holds the alignment-scaling runners `scripts/arc_alignment_scaling_v1.py` to `scripts/arc_alignment_scaling_v5.py`, the combined runner `scripts/arc_eden_v6_runner.py`, and the files the five runners recorded, one folder per runner. Its evidence tier in `experiments/README.md`: pilot.

**Reproduction**

- `REPRODUCE.md`: names `scripts/arc_eden_v6_runner.py` as the runner for a new run and lists the five earlier runners as superseded; gives the models and depth settings recorded in the result files.

**Code (`scripts/`)**

- `scripts/arc_alignment_scaling_v1.py`: measures how alignment scores change with reasoning depth, at four depths, for each supported model, and fits an alignment-scaling exponent; writes `alignment_raw_<model>_<timestamp>.json` and a combined file. Listed as superseded in `REPRODUCE.md`, as are the next four.
- `scripts/arc_alignment_scaling_v2.py`: adds depth control through each provider's own setting, scoring by a model other than the subject, scores from 0 to 100, capability measured at the same depths, prompts graded by difficulty and an ordinal analysis; writes `v2_raw_<model>_<timestamp>.json`.
- `scripts/arc_alignment_scaling_v3.py`: scores the quality of ethical reasoning on dilemma prompts rather than refusal, with paired capability measurement, rank, Kruskal-Wallis and regression analyses and a ceiling-effect check; writes `v3_raw_<model>_<timestamp>.json`.
- `scripts/arc_alignment_scaling_v4.py`: adds three scorers per response with inter-rater reliability, a partial correlation that removes response length, calibration examples for the scorers, prompt randomisation, saving after every prompt, bootstrap intervals, null-baseline prompts, four sub-scores, a saturation-curve fit and adversarial suppression prompts; writes `v4_checkpoint_<model>.json`, `v4_final_<model>_<timestamp>.json` and `v4_analysis_<model>_<timestamp>.json`.
- `scripts/arc_alignment_scaling_v5.py`: the runner whose docstring sets out a four-layer blinding design (a stakes framing, a perceptual firewall, sequential laundering of each response by other models, and scoring by models that took no part), with fall-backs for scorers and launderers, pre-flight checks and checkpoint resume; writes `v5_checkpoint_<model>.json` and `v5_final_<model>_<timestamp>.json`.
- `scripts/arc_eden_v6_runner.py`: a shared experiment platform (the description its command line gives), generated, by its header, from a modular source that is not in this repository; it holds a registry of experiments chosen with `--experiment` and writes a run folder holding `manifest.json`, `selected_prompts.json`, `conditions.json` and `blind_protocol.json`.

**Results (`results/`)**

- `results/v1/alignment_raw_claude-sonnet_20260310_194845.json`, `results/v1/alignment_raw_deepseek-r1_20260310_200531.json`, `results/v1/alignment_raw_gemini-flash_20260310_192554.json` and `results/v1/alignment_raw_openai-o1_20260310_193118.json`: one file per subject model, in the file name pattern that `scripts/arc_alignment_scaling_v1.py` writes; each holds the depth settings, the prompt categories and the raw responses.
- `results/v2/v2_raw_claude-sonnet_20260310_203615.json` and `results/v2/v2_raw_deepseek-r1_20260310_210020.json`: the files of `scripts/arc_alignment_scaling_v2.py`, one per subject model.
- `results/v3/v3_raw_claude-sonnet_20260310_212748.json` and `results/v3/v3_raw_deepseek-r1_20260310_215537.json`: the files of `scripts/arc_alignment_scaling_v3.py`, one per subject model.
- `results/v4/v4_checkpoint_claude-opus.json`, `results/v4/v4_checkpoint_deepseek-r1.json`, `results/v4/v4_checkpoint_gemini-flash.json` and `results/v4/v4_checkpoint_openai-gpt54.json`: the checkpoint files that `scripts/arc_alignment_scaling_v4.py` saves as it runs, one per subject model.
- `results/v4/v4_final_claude-opus_20260311_010639.json`, `results/v4/v4_final_deepseek-r1_20260311_024302.json` and `results/v4/v4_final_gemini-flash_20260311_013242.json`: the final files of the same runner, one per subject model.
- `results/v4/v4_analysis_deepseek-r1_20260311_024303.json` and `results/v4/v4_analysis_gemini-flash_20260311_013243.json`: the analysis files of the same runner, each with its reliability, correlation and exponent fields.
- `results/v5-final/v5_final_claude-opus_20260312_112739.json`, `results/v5-final/v5_final_deepseek-r1_20260311_211855.json`, `results/v5-final/v5_final_gemini-flash_20260311_151244.json`, `results/v5-final/v5_final_grok-4-fast_20260311_200910.json`, `results/v5-final/v5_final_groq-qwen3_20260312_073302.json` and `results/v5-final/v5_final_openai-gpt54_20260311_191836.json`: the final files of `scripts/arc_alignment_scaling_v5.py`, one per subject model.
- `results/v5-final/v5_final_groq-qwen3_20260312_073302 copy.json`: a byte-identical second copy of `results/v5-final/v5_final_groq-qwen3_20260312_073302.json` (the same git object).

31 files, all listed.

Where the result and its status are stated: `papers/Paper-III-Alignment-Scaling-Problem/`, `papers/Paper-IV-a-Baked-In-vs-Computed-Alignment/`, `papers/Paper-IV-b-Alignment-Saturation-at-Low-Depth/`, `papers/Paper-IV-c-ARC-Align-Benchmark/`, `papers/Paper-IV-d-The-Effect-of-Blinding-on-AI-Alignment-Evaluation/` and `papers/ARC-Alignment-Scaling-Working-Report/`.

### `experiments/blind-prediction-test__Paper-III-and-Foundational/`: Paper III and The ARC Principle

Holds the blind prediction script, its printed output and a forensic analysis of the test; the home has no README, so its files are grouped by folder. Its evidence tier in `experiments/README.md`: supporting.

**Code (`scripts/`)**

- `scripts/blind_prediction_test.py`: for five simulated systems (Barabási-Albert networks, gradient descent with momentum, a belief propagation decoder, coupled oscillator synchronisation and an evolutionary algorithm), measures β from process data, computes the predicted α = 1/(1-β) before reading the outcome data, measures α from the outcome data and compares the two; the figure it writes, `results/blind_prediction_results.png`, is not in this folder.

**Results (`results/`)**

- `results/results_blind_prediction.txt`: the printed output of `scripts/blind_prediction_test.py` (it opens with the title that script prints); byte-identical to `papers/Foundational/results/results_blind_prediction.txt`.
- `results/forensic-analysis.md`: a written analysis of the method by which the test estimated β, and of whether the test systems meet the framework's axioms.
- `results/forensic-analysis.png`: a figure image; not described in its folder; byte-identical to `papers/Foundational/figures/BLIND_TEST_FORENSIC_ANALYSIS.png`.

4 files, all listed.

Where the result and its status are stated: `papers/Paper-III-Alignment-Scaling-Problem/` and `papers/Foundational/`.

### `experiments/eden-intervention__Paper-V/`: Paper V

Holds the Eden Protocol intervention runners and the final file of each subject model's run. Its evidence tier in `experiments/README.md`: pilot, single-scorer, partially blind.

**Reproduction**

- `REPRODUCE.md`: the runner it names, the models, depth settings and prompt count recorded in the result files, and how to run it again.

**Code (`scripts/`)**

- `scripts/eden_protocol_scaling_test.py`: runs two conditions on the same prompts and models: a control prompt, and the same prompt with the Eden Protocol's three evaluation loops placed before it as required reasoning steps; responses are scored by a model other than the subject; writes `eden_checkpoint_<model>.json` and `eden_final_<model>_<timestamp>.json`. `REPRODUCE.md` lists it as prior.
- `scripts/eden_protocol_scaling_test_v2.py`: the same two-condition design with adapters for six subject models, a scorer chosen from models that take no part in the run, and a `--scorer` override; writes the same file names. `REPRODUCE.md` lists it as prior.
- `scripts/eden_protocol_scaling_test_v3.py`: adds scoring by several models that take no part, two laundering passes of each response, tier-weighted consensus, disagreement flags, optional suppression conditions, purpose-loop variants, an optional cross-tradition ethics kernel and optional ternary routing; writes `eden_v3_checkpoint_<model>_<configuration>.json` and `eden_v3_final_<model>_<configuration>_<timestamp>.json`. `REPRODUCE.md` names it as the runner.

**Results (`results/`)**

- `results/eden_final_claude_20260312_130933.json`, `results/eden_final_deepseek_20260312_020928.json`, `results/eden_final_gemini_20260312_013901.json`, `results/eden_final_gpt_20260312_121158.json`, `results/eden_final_grok_20260312_124959.json` and `results/eden_final_groq_20260312_123528.json`: one final file per subject model, each recording its version, model, depth settings, prompt count, conditions and scorer; the files whose `version` field reads 1.0 are in the format `scripts/eden_protocol_scaling_test.py` writes, and those reading 2.0 in the format of `scripts/eden_protocol_scaling_test_v2.py`.

10 files, all listed.

Where the result and its status are stated: `papers/Paper-V-Stewardship-Gene/`.

### `experiments/honey-architecture__Paper-VI/`: Paper VI

Holds the honey-architecture simulation and test scripts, the self-modifying network series, their outputs and figures, a provenance pack and reader copies of three tables. Its evidence tier in `experiments/README.md`: mechanistic in simulation, exploratory on live models.

**Reproduction**

- `REPRODUCE.md`: the runs recorded in the result files' own metadata, the files present and how to merge the result files.

**Code (`scripts/`)**

- `scripts/eden_honey_simulation.py`: simulates three regimes (a baseline that optimises capability alone, an entangled regime that optimises capability times safety, and the entangled regime with a verification cost); writes `eden_honey_capability.png`, `eden_honey_safety.png`, `eden_honey_ratio.png` and `eden_honey_simulation_results.json`.
- `scripts/eden_honey_tests.py`: four tests against live models (an alignment-scaling exponent, a monitoring-removal gap, a coupling-degradation test and an Eden Protocol intervention), with a demo mode that needs no keys (`--demo`); writes `eden_honey_test_results.json`.
- `scripts/eden_honey_dashboard.jsx`: a browser dashboard component that reruns the three-regime simulation and draws its curves.
- `scripts/eden_self_modifying_ai.py`: a small neural network whose meta-controller changes its own learning rate, weight scaling, mutation rate, neuron count, gradient clipping and momentum during training, in three modes (baseline, entangled, and entangled with verification drag); writes `eden_selfmod_results.json`, `eden_selfmod_results.png` and `eden_selfmod_weights.png`.
- `scripts/eden_self_modifying_ai_v2.py`: the same three modes with identical proposal distributions and several seeds; writes `eden_selfmod_v2_results.json`, `eden_selfmod_v2_results.png` and `eden_selfmod_v2_stats.png`.
- `scripts/eden_self_modifying_ai_v3.py`: adds tasks whose best weights conflict with one another, with tests on collapse rates and combined scores across seeds; writes `eden_selfmod_v3_results.json`, `eden_selfmod_v3_results.png` and `eden_selfmod_v3_stats.png`.
- `scripts/eden_self_modifying_ai_v4.py`: runs the conflicting-task design at five complexity levels and compares the modes at each; writes `eden_selfmod_v4_results.json` and `eden_selfmod_v4_scaling.png`.
- `scripts/merge_honey_test_results.py`: merges the result files of separate model batches of `scripts/eden_honey_tests.py` into one file.
- `scripts/generate_honey_provenance_pack.py`: writes the provenance pack in `results/honey_provenance_pack/` (its two maps, its figure index, its readiness note, `extracted_sources/` and `raw_results_generated/`) from source files kept outside this repository.
- `scripts/anti_sycophancy_recovered.py`: a set of anti-sycophancy prompts reconstructed from text kept outside this repository; its docstring calls it a source recovery artefact for prompt reuse, not a runtime from the original authoring environment.

**Figures (`figures/`)**

- `figures/eden_honey_capability.png`, `figures/eden_honey_ratio.png` and `figures/eden_honey_safety.png`: plots of the names that `scripts/eden_honey_simulation.py` writes.
- `figures/eden_selfmod_results.png` and `figures/eden_selfmod_weights.png`: plots of the names that `scripts/eden_self_modifying_ai.py` writes.
- `figures/eden_selfmod_v2_results.png` and `figures/eden_selfmod_v2_stats.png`: plots of the names that `scripts/eden_self_modifying_ai_v2.py` writes.
- `figures/eden_selfmod_v3_results.png` and `figures/eden_selfmod_v3_stats.png`: plots of the names that `scripts/eden_self_modifying_ai_v3.py` writes.
- `figures/eden_selfmod_v4_scaling.png`: the plot of the name that `scripts/eden_self_modifying_ai_v4.py` writes.

**Results (`results/`)**

- `results/README.md`: explains the three coupling tables and links each reader copy to its dated original.
- `results/eden_selfmod_results.json`, `results/eden_selfmod_v2_results.json`, `results/eden_selfmod_v3_results.json` and `results/eden_selfmod_v4_results.json`: the result files of the four self-modifying scripts, in that order; each holds its metadata, the results of each mode and a summary.
- `results/eden_honey_capability.png`, `results/eden_honey_ratio.png`, `results/eden_honey_safety.png`, `results/eden_selfmod_results.png`, `results/eden_selfmod_weights.png`, `results/eden_selfmod_v2_results.png`, `results/eden_selfmod_v2_stats.png`, `results/eden_selfmod_v3_results.png`, `results/eden_selfmod_v3_stats.png` and `results/eden_selfmod_v4_scaling.png`: plots with the names of those in `figures/`; each is byte-identical to the file of the same name in `results/honey_provenance_pack/raw_results_generated/`.

**Provenance pack (`results/honey_provenance_pack/`)**

- `results/honey_provenance_pack/HONEY_PROVENANCE_MAP.md`: which sources, figures and result files the pack recovered and where from, and which statements it treats as provenance-limited.
- `results/honey_provenance_pack/HONEY_PROVENANCE_MAP.json`: the same map as records, with its text source and search scope.
- `results/honey_provenance_pack/HONEY_FIGURE_SOURCE_INDEX.md`: a table giving, for each figure and exported result file, its experiment family, the lines of the text export it was recovered from and its expected file name.
- `results/honey_provenance_pack/EDEN_V3_RUN_READINESS.md`: a readiness note for `eden_protocol_scaling_test_v3.py`: its command-line options, the environment variables it needs and its output naming.
- `results/honey_provenance_pack/extracted_sources/eden_honey_dashboard_from_text.jsx.txt`, `results/honey_provenance_pack/extracted_sources/eden_honey_simulation_from_text.py.txt`, `results/honey_provenance_pack/extracted_sources/eden_honey_tests_from_text.py.txt`, `results/honey_provenance_pack/extracted_sources/eden_selfmod_v1_from_text.py.txt`, `results/honey_provenance_pack/extracted_sources/eden_selfmod_v2_from_text.py.txt`, `results/honey_provenance_pack/extracted_sources/eden_selfmod_v3_from_text.py.txt` and `results/honey_provenance_pack/extracted_sources/eden_selfmod_v4_from_text.py.txt`: the source text of each script as recovered from a line range of a text export; `HONEY_PROVENANCE_MAP.md` gives each range.
- `results/honey_provenance_pack/raw_results_generated/eden_honey_capability.png`, `results/honey_provenance_pack/raw_results_generated/eden_honey_ratio.png` and `results/honey_provenance_pack/raw_results_generated/eden_honey_safety.png`: the plots of the honey simulation family, as kept in the pack.
- `results/honey_provenance_pack/raw_results_generated/eden_honey_simulation_results.json`: the exported result file of the honey simulation.
- `results/honey_provenance_pack/raw_results_generated/eden_honey_test_results.json`: the merged result file of the honey tests (its `metadata.experiment` field reads `eden_honey_tests_merged`); `results/README.md` names it the dated original of the combined table.
- `results/honey_provenance_pack/raw_results_generated/eden_honey_test_results_gemini_grok.json` and `results/honey_provenance_pack/raw_results_generated/eden_honey_test_results_main4.json`: the result files of two model batches of the honey tests (their `metadata.models_tested` fields list the models); `results/README.md` names them the dated originals of the other two tables.
- `results/honey_provenance_pack/raw_results_generated/eden_honey_test_results_demo.json`: a result file of the honey tests' demo mode (its `metadata.demo_mode` field is true).
- `results/honey_provenance_pack/raw_results_generated/eden_selfmod_results.json`, `results/honey_provenance_pack/raw_results_generated/eden_selfmod_v2_results.json`, `results/honey_provenance_pack/raw_results_generated/eden_selfmod_v3_results.json` and `results/honey_provenance_pack/raw_results_generated/eden_selfmod_v4_results.json`: the self-modifying result files as kept in the pack; each differs from the file of the same name in `results/` only in its `metadata.timestamp`.
- `results/honey_provenance_pack/raw_results_generated/eden_selfmod_results.png`, `results/honey_provenance_pack/raw_results_generated/eden_selfmod_weights.png`, `results/honey_provenance_pack/raw_results_generated/eden_selfmod_v2_results.png`, `results/honey_provenance_pack/raw_results_generated/eden_selfmod_v2_stats.png`, `results/honey_provenance_pack/raw_results_generated/eden_selfmod_v3_results.png`, `results/honey_provenance_pack/raw_results_generated/eden_selfmod_v3_stats.png` and `results/honey_provenance_pack/raw_results_generated/eden_selfmod_v4_scaling.png`: the plots of the self-modifying series, as kept in the pack.

**Reader copies (`results/reader/`)**

- `results/reader/README.md`: how to read the three coupling tables (`null` stands for the original file's `NaN`) and how to check the reader copies against their originals.
- `results/reader/field-key.json`: lists every changed position and the file fingerprints, for each reader copy and its original.
- `results/reader/verify.py`: opens each reader copy and rebuilds the archived bytes in memory, to check them against the field key.
- `results/reader/standard/eden_honey_test_results.standard.json`, `results/reader/standard/eden_honey_test_results_gemini_grok.standard.json` and `results/reader/standard/eden_honey_test_results_main4.standard.json`: the reader copies of the three tables, in standard JSON.

72 files, all listed.

Where the result and its status are stated: `papers/Paper-VI-Honey-Architecture/`.

### `experiments/cauchy-unification__Paper-VII/`: Paper VII

Holds the domain-classification runs as they were made, with their data, null and negative controls, an archived extension packet and reader copies of two result tables. Its evidence tier in `experiments/README.md`: under correction; that file's section "Paper VII is under correction" adds: "The recorded outputs stand; the reading placed on them does not."

**Guide**

- `README.md`: the folder's guide to its scripts, results and data, with the role of each script and how to run it.

**Code (`scripts/`)**

- `scripts/arc_50_domain_universal_test.py`: fits candidate models to each domain of `data/canonical_50_domain_manifest.json`, selects the best by AICc, scores the empirical curve-fit domains by strict family match with no tolerance, and reports each evidence tier apart; writes `results/results_50_domain_validation.json`. The folder's README calls it primary.
- `scripts/arc_20_domain_universal_test.py`: the earlier comparison over twenty domains, with its own tolerance rule; the folder's README labels it Legacy, "retained for provenance".
- `scripts/build_operator_classification_packet.py`: writes a packet that lets an outside assessor classify each empirical domain's composition operator without seeing the manifest's predictions, with its response template and instructions.
- `scripts/analyze_empirical_misses.py`: writes a structured account of each empirical domain where the best-fitting family differs from the predicted one.
- `scripts/run_null_controls.py`: runs a random family-label null and a shuffled-data null against the empirical cohort, with the same fitter.
- `scripts/negative_controls.py`: runs scrambled-data, random monotone, axiom-violating and bootstrap-stability controls.
- `scripts/temporal_out_of_sample.py`: fits candidate forms on an earlier data window and checks whether the same functional family wins on a later window, in five domains.
- `scripts/run_preregistered_extension.py`: replays the locked twelve-domain extension packet with the fitting protocol of the 50-domain suite; the folder's README labels it a "Locked pilot dry run", "noncanonical because execution preceded any OSF timestamp".
- `scripts/cross_validate_fits.R`: repeats the empirical curve fits in R and compares the selected models with the Python results; `results/cross_library_replication_status.md` gives its status as blocked locally.
- `scripts/arc_complete_test_suite.py`: runs six tests in one script: a one-dimensional organism meta-analysis, a table of three dimensional predictions, a numerical check of the Cauchy functional equations, a Friedmann mapping check, a classification of eighteen domains and a Hyers-Ulam stability check.
- `scripts/arc_unified_paradigm_test.py`: classifies fifteen test cases into regimes by Cauchy's functional equations.
- `scripts/arc_rigorous_validation.py`: checks Cauchy's three functional equations numerically, then fits eighteen domains with three two-parameter functions under strict matching.
- `scripts/arc_universal_proof.py`: works through an argument from Cauchy's functional equations and maximum entropy to the exponent formula α = d_eff/(d_eff + 1), with a mapping of the Friedmann expansion eras and a table of domains.

**Data (`data/`)**

- `data/canonical_50_domain_manifest.json`: the manifest of the 50-domain suite: each domain's predictions and dataset; `experiments/README.md` records that this path is linked from outside this repository and does not move.
- `data/blinded_operator_classification_packet.json`: the neutral packet of the 25 empirical domains for an outside classifier, written by `scripts/build_operator_classification_packet.py`.
- `data/blinded_operator_classification_template.json`: the response template for that packet, written by the same script.
- `data/expansion_domains_26_65.json`: a list of forty domain records (identifiers 51 to 90), each with its operator class, predicted family, source and dataset, under the cohort label `expansion40`; no script or README in this folder names the file.
- `data/expansion_domains_66_105.json`: a list of forty domain records (identifiers 66 to 105) of the same form and cohort label; no script or README in this folder names the file.

**Archived extension packet (`preregistration/`)**

- `preregistration/next_extension_protocol.md`: the draft protocol of the twelve-domain extension packet; the folder's README: "now marked as exercised locally and not valid as a prospective preregistration".
- `preregistration/next_extension_manifest.json`: the locked candidate list used by that extension dry run.
- `preregistration/osf_component_registration.md`: the OSF wrapper drafted for that packet; the folder's README: "now marked not for unchanged upload as a preregistration".
- `preregistration/osf_attachment_manifest.json`: the attachment inventory of the archived packet.
- `preregistration/file_checksums.txt`: SHA-256 checksums for the archived packet's files.
- `preregistration/extracted_data/README.md`: notes on the extracted-data folder, with the rules for its dry-run data.
- `preregistration/extracted_data/extension_data.json`: the datasets used by the extension dry run; the folder's note says not to describe it as a preregistered extraction.

**Results (`results/`)**

- `results/README.md`: how to open the two archived result tables as standard JSON, and how to convert a reader copy back to the original bytes.
- `results/results_50_domain_validation.json` and `results/results_50_domain_validation.txt`: the output of `scripts/arc_50_domain_universal_test.py`.
- `results/results_20_domain_validation.txt`: the output of `scripts/arc_20_domain_universal_test.py`.
- `results/results_complete_suite.txt`, `results/results_arc_unified_paradigm_test.txt`, `results/results_arc_rigorous_validation.txt` and `results/results_arc_universal_proof.txt`: the outputs of `scripts/arc_complete_test_suite.py`, `scripts/arc_unified_paradigm_test.py`, `scripts/arc_rigorous_validation.py` and `scripts/arc_universal_proof.py`, in that order.
- `results/empirical_miss_analysis.json` and `results/empirical_miss_analysis.md`: the output of `scripts/analyze_empirical_misses.py`.
- `results/miss_analysis.md`: a written review, domain by domain, of the empirical domains whose best-fitting family differs from the predicted one, made inside the programme; no script in this folder writes it. It is an internal review, and the note at its head gives the paper's own reading of the misses.
- `results/null_control_results.json` and `results/null_control_results.md`: the output of `scripts/run_null_controls.py`.
- `results/negative_control_results.json` and `results/negative_control_results.txt`: the output of `scripts/negative_controls.py`.
- `results/results_temporal_out_of_sample.json`: the output of `scripts/temporal_out_of_sample.py`.
- `results/results_preregistered_extension.json` and `results/results_preregistered_extension.txt`: the output of `scripts/run_preregistered_extension.py`; the folder's README: "pilot-only; not preregistered".
- `results/independent_operator_classification.md`: a second classification of the 25 empirical domains by composition operator, made inside the programme from each system's mechanics, with the criteria it used. It is an internal check and not an independent assessment; the note at its head says so.
- `results/blinded_operator_classification_instructions.md`: the instructions for an outside classifier, written by `scripts/build_operator_classification_packet.py`.
- `results/cross_library_replication_status.md`: the status note of the cross-check in R; its status line reads blocked locally.
- `results/paper-vii-results-standard.json`: the reader copy of `results/results_50_domain_validation.json`, in standard JSON.
- `results/paper-vii-miss-analysis-standard.json`: the reader copy of `results/empirical_miss_analysis.json`, in standard JSON.
- `results/paper-vii-result-format.json`: the field key: each changed position, the file fingerprints and the dated source links.
- `results/paper_vii_results_view.py`: the converter that exports or rebuilds the dated results without altering its input.

51 files, all listed.

Where the result and its status are stated: `papers/Paper-VII-Cauchy-Unification/`.

### `experiments/domain-validation__Foundational-and-Origin/`: The ARC Principle and On the Origin of Scaling Laws

Holds scripts that apply the framework's dimensional relation and composition classification in physics and other domains, their printed outputs, and an evidence pack of tables and a memo on the alignment and Eden intervention results; the home has no README, so its files are grouped by folder. Its evidence tier in `experiments/README.md`: supporting, mathematical.

**Code (`scripts/`)**

- `scripts/arc_1d_prediction_test.py`: compiles published metabolic scaling exponents for organisms whose transport dimension can be assigned, and compares them with α = d/(d+1) at dimensions one, two and three.
- `scripts/arc_acoustic_time_crystal_test.py`: simulates three driven acoustic systems (a parametric oscillator, a coupled oscillator chain and bubble dynamics) and compares their growth with the framework's scaling forms.
- `scripts/arc_definitive_test.py`: for eight recursive systems in five domains, measures the composition operator from each system's marginal gains, predicts the scaling function and its exponent from that operator, then fits the observed curve and compares.
- `scripts/arc_einstein_verification.py`: works through, in four parts, E = mc², the nuclear chain reaction in three regimes, a ladder of scales, and their combination, within the framework's classification.
- `scripts/arc_physics_domains_test.py`: compares the framework's predicted scaling form with simulated or published data in four domains: quantum error correction, metabolic allometry, coupled-oscillator synchronisation and acoustic resonance.
- `scripts/arc_real_time_crystal_test.py`: applies the framework to experimental values taken from a published Rydberg-gas time-crystal study and its peer review file.
- `scripts/arc_section7_breakthrough.py`: in six parts: predictions for five further domains, an exponent derivation, an information-theoretic argument from Cauchy's equations and maximum entropy, cross-domain transfer checks, the exponent formula α = d_eff/(d_eff + 1), and a combined statistical summary.
- `scripts/prove_IxR_equals_complexity_v2.py`: sets out, from the framework's three axioms, the multiplicative form U = I × f(R, β), a non-additivity argument and a synergy quotient, and lists in its docstring what it does not establish.
- `scripts/generate_phase0_evidence_pack.py`: writes the four files of `phase0_evidence_pack/` from result files kept outside this repository.
- `scripts/arc_20_domain_universal_test.py`, `scripts/arc_complete_test_suite.py`, `scripts/arc_rigorous_validation.py`, `scripts/arc_unified_paradigm_test.py` and `scripts/arc_universal_proof.py`: byte-identical to the files of the same names in `experiments/cauchy-unification__Paper-VII/scripts/` and described there; that folder's README labels `arc_20_domain_universal_test.py` Legacy.

**Results (`results/`)**

- `results/results_arc_1d_prediction_test.txt`, `results/results_arc_acoustic_time_crystal_test.txt`, `results/results_arc_definitive_test.txt`, `results/results_arc_einstein_verification.txt`, `results/results_arc_physics_domains_test.txt`, `results/results_arc_real_time_crystal_test.txt`, `results/results_arc_section7_breakthrough.txt` and `results/results_prove_IxR_equals_complexity_v2.txt`: the printed outputs of the scripts of the matching names; each opens with the title its script prints.
- `results/results_20_domain_validation.txt`, `results/results_complete_suite.txt`, `results/results_arc_rigorous_validation.txt`, `results/results_arc_unified_paradigm_test.txt` and `results/results_arc_universal_proof.txt`: byte-identical to the files of the same names in `experiments/cauchy-unification__Paper-VII/results/`, the outputs of the five shared scripts.

**Evidence pack (`phase0_evidence_pack/`)**

- `phase0_evidence_pack/canonical_v5_alignment_table.md`: a table, one row per subject model, recomputed from the final files of `arc_alignment_scaling_v5.py`; `experiments/README.md` records that this path is linked from outside this repository and does not move.
- `phase0_evidence_pack/canonical_eden_intervention_table.md`: a table, one row per Eden intervention result file, with its scorer and run-quality label.
- `phase0_evidence_pack/paper_writer_memo.md`: a memo that sorts the pack's claims by how they may be stated, under its own section headings.
- `phase0_evidence_pack/claim_evidence_ledger.json`: a ledger of claims and their evidence, with its generation time.

31 files, all listed.

Where the result and its status are stated: `papers/Foundational/` and `papers/On-the-Origin-of-Scaling-Laws/`.

### `papers/Paper-VIII-The-Load-Bearing-Proof/experiments/`: Paper VIII

Holds the code of the paper's three experiments and of its blinding instrument, the twelve run ledgers of the architectural simulation, the two tests of its safety gate and the result files that the paper's own `results/` folder does not hold. Its evidence tier in `experiments/README.md`: pilot, unblinded, not replicated.

**Guides and records**

- `README.md`: the folder's guide: what each file is, when each result was recorded, the seven files published with a change to their text (each with the hash of the bytes that ran and of the copy published), and what is deliberately not here.
- `REPRODUCE.md`: how to run each piece again, and what it needs that this folder does not supply.
- `MANIFEST.sha256`: one line per file: the SHA-256 of every byte deposited here.

**Behavioural experiment (`code/behavioural-dgm/`)**

- `code/behavioural-dgm/eden_dgm_experiment.py`: the earlier of the two main runners of the behavioural experiment, a Darwin Godel Machine that evolves a population under three selection rules, with a separate judging model.
- `code/behavioural-dgm/eden_dgm_experiment_fixed.py`: the later main runner, which the folder's README calls the corrected runner.
- `code/behavioural-dgm/variants/eden_dgm_v3_gpt54.py`: the calibrated protocol of the experiment's third version.
- `code/behavioural-dgm/variants/eden_dgm_v4_tradeoff.py`: the trade-off protocol of the experiment's fourth version, as designed.
- `code/behavioural-dgm/variants/eden_dgm_v4_1_tradeoff.py`: a portability patch of the trade-off protocol; the folder's README: "No design change".
- `code/behavioural-dgm/variants/eden_dgm_removal_test.py`: the safety-gate removal test as designed.
- `code/behavioural-dgm/variants/eden_dgm_removal_test_v1_1.py`: a portability patch of the removal test; the folder's README: "No design change".

**Representational experiment (`code/weight/`)**

- `code/weight/v1/eden_weight_experiment.py`: the weight-level experiment in its version 1.0 as run, rank 8 over 8 layers: it fine-tunes a base model with low-rank adapters under a capability-only loss, a safety-only loss and an entangled loss that couples the two.
- `code/weight/v2/eden_weight_experiment.py`: the same experiment in its version 2.0 as run, rank 16 over 16 layers.
- `code/weight/generate_training_data.py`: writes the four corpora the training arms use.

**Architectural experiment (`code/architectural-gated-self-mod/`)**

- `code/architectural-gated-self-mod/README.md`: the harness's own guide: its two self-modification channels (a learned optimizer whose parameters are mutated and selected, and a small policy program mutated as code and checked before promotion), its four conditions, its outputs and its metrics.
- `code/architectural-gated-self-mod/configs/default.yaml`: the run configuration with the four conditions and three seeds.
- `code/architectural-gated-self-mod/configs/quick.yaml`: a smaller configuration: one seed, three conditions, and fewer iterations and samples.
- `code/architectural-gated-self-mod/pyproject.toml`: the package definition (`eden-gated-self-mod-agent`), with its test settings.
- `code/architectural-gated-self-mod/src/run_train.py`: runs each condition and seed from a configuration file, writing a `ledger.jsonl` and a `summary.json` for each run.
- `code/architectural-gated-self-mod/src/run_eval.py`: loads the run summaries, compares the conditions with Mann-Whitney tests and computes the acceleration measures.
- `code/architectural-gated-self-mod/src/models/learned_optimizer.py`: the learned optimizer (`LearnedOptimizer`, built on an LSTM cell) that emits the update controls.
- `code/architectural-gated-self-mod/src/models/optimizee.py`: the small multilayer network that is trained (`MLPOptimizee`).
- `code/architectural-gated-self-mod/src/models/policy_program.py`: the policy program (`PolicyGenome`) that is mutated and rendered as source code.
- `code/architectural-gated-self-mod/src/tasks/suite.py`: builds the adversarial regression tasks and the capability and retention scores.
- `code/architectural-gated-self-mod/src/training/controller.py`: mutates candidates, accepts or rejects each by the rule of its condition, and writes the ledger.
- `code/architectural-gated-self-mod/src/training/inner_loop.py`: runs one training episode under the learned optimizer's controls.
- `code/architectural-gated-self-mod/src/training/outer_loop.py`: builds the model and the optimizer and runs one self-modification iteration with its evaluation.
- `code/architectural-gated-self-mod/src/training/verifier.py`: checks a candidate policy's source, the model's state and the policy's determinism before promotion.
- `code/architectural-gated-self-mod/src/metrics/acceleration.py`: computes the acceleration measures of a series: its differences of order one and two, normalised by compute.
- `code/architectural-gated-self-mod/src/metrics/logging.py`: helpers that convert results to JSON and write them.
- `code/architectural-gated-self-mod/src/__init__.py`, `code/architectural-gated-self-mod/src/metrics/__init__.py`, `code/architectural-gated-self-mod/src/models/__init__.py`, `code/architectural-gated-self-mod/src/tasks/__init__.py` and `code/architectural-gated-self-mod/src/training/__init__.py`: empty files that mark the package's folders.

**Blinding instrument (`code/instrument/`)**

- `code/instrument/eden_blinding_kit.py`: the bias-control toolkit that the draft registration of the follow-on study names: identity masking, laundering, order randomisation, scorer panels that exclude the scored model, a leakage audit, a length-bias check and a meta-commentary filter; it is not one of the three experiment runners.

**Results (`results/`)**

- `results/behavioural-dgm/dgm_v4_smoke_results.json`: the smoke run of the trade-off protocol, which the folder's README dates before the portability patch and so attributes to `code/behavioural-dgm/variants/eden_dgm_v4_tradeoff.py`.
- `results/behavioural-dgm/dgm_v4_full_results.json`: the full run of the trade-off protocol, recorded, by the folder's README, after the portability patch was written.
- `results/behavioural-dgm/dgm_removal_results.json` and `results/behavioural-dgm/dgm_removal_results_v3.json`: two results of the safety-gate removal test, recorded, by the folder's README, after its portability patch was written.
- `results/weight/adapters/capability_only/adapter_config.json`, `results/weight/adapters/entangled/adapter_config.json`, `results/weight/adapters/removal/adapter_config.json` and `results/weight/adapters/safety_only/adapter_config.json`: the four adapter configurations as run (byte-identical to one another); the trained adapter weights are referenced by hash in the folder's README, not deposited.
- `results/weight/training-data/capability_data.jsonl`, `results/weight/training-data/mixed_data.jsonl`, `results/weight/training-data/safety_data.jsonl` and `results/weight/training-data/test_data.jsonl`: the four corpora as generated and used.

**Run ledgers (`run-ledgers/`)**

- `run-ledgers/20260317T234824Z_static_seed0/ledger.jsonl`, `run-ledgers/20260317T234838Z_static_seed1/ledger.jsonl`, `run-ledgers/20260317T234847Z_static_seed2/ledger.jsonl`, `run-ledgers/20260317T234858Z_babylon_seed0/ledger.jsonl`, `run-ledgers/20260317T234920Z_babylon_seed1/ledger.jsonl`, `run-ledgers/20260317T234939Z_babylon_seed2/ledger.jsonl`, `run-ledgers/20260317T234959Z_eden_seed0/ledger.jsonl`, `run-ledgers/20260317T235019Z_eden_seed1/ledger.jsonl`, `run-ledgers/20260317T235044Z_eden_seed2/ledger.jsonl`, `run-ledgers/20260317T235112Z_drag_control_seed0/ledger.jsonl`, `run-ledgers/20260317T235122Z_drag_control_seed1/ledger.jsonl` and `run-ledgers/20260317T235134Z_drag_control_seed2/ledger.jsonl`: the per-run ledgers of the architectural simulation, four conditions by three seeds, each in a directory named for the run that wrote it.

**Tests (`tests/`)**

- `tests/test_verifier.py`: checks that a policy program compiles and returns its output shape, and that the verifier rejects imports.
- `tests/test_rollback.py`: checks that rejecting a candidate preserves the parent state.

61 files, all listed.

Where the result and its status are stated: `papers/Paper-VIII-The-Load-Bearing-Proof/`.

### `papers/Paper-X-Coupled-CoScaling-Correction/`: Paper X

The paper's own folder: its exports, the deterministic harness and its test suites, the corrected theorem statements and their check record, the real-model protocols, runners and recorded runs, and the texts of its original OSF deposit. `experiments/README.md` gives this folder no evidence tier; its `REPRODUCIBILITY.md` says which command reproduces which part, and what cannot be reproduced offline.

**The paper and its records**

- `Paper-X-Coupled-CoScaling-Correction.pdf`, `Paper-X-Coupled-CoScaling-Correction.html`, `Paper-X-Coupled-CoScaling-Correction.txt`, `Paper-X-Coupled-CoScaling-Correction.cff` and `Paper-X-Coupled-CoScaling-Correction-latex-source.tar.gz`: the paper's exports: the PDF, the export HTML, the text, the citation file and the LaTeX source package.
- `README.md`: the paper's README: its version, summary and links to its exports and its OSF record.
- `CLAIMS.md`: the claim-status ledger, written for the drafts before the paper's original edition; its opening note lists the statements in it that the current version has superseded.
- `ANTICIPATED_OBJECTIONS.md`: anticipated objections and responses, written for those drafts; its opening note lists the statements since superseded.
- `NEGATIVE_RESULTS.md`: a record of the paper's null results, non-compliances, withdrawn items and the errors of earlier drafts, under its section headings; its opening note lists the statements since superseded.
- `EXECUTIVE_SUMMARY_PLAIN.md`: a plain-language summary written for an earlier version of the paper; its opening note lists the statements since superseded.
- `OSF_DEPOSIT.md`: the descriptions prepared for posting the paper on OSF; its opening note says it records the deposit under the paper's earlier title.
- `REVIEW_OUTREACH.md`: a plan for outside review, split into five parts, one expert each.
- `THEOREMS.md`: the corrected statements of the theorems, with their derivations.
- `archive/README-before-2026-10-19.md`: the paper's earlier README, kept in `archive/`; `papers/README.md` says that superseded files sit in such folders.

**Reproduction**

- `REPRODUCIBILITY.md`: which command reproduces which part of the paper, and what cannot be reproduced offline.
- `Makefile`: the make targets for installing, running the harness, the two test suites, the estimator and the self-test.
- `Dockerfile`: a container definition that runs only the deterministic offline checks.
- `requirements.txt`: the Python dependencies, with the versions the artefacts were generated and checked on.
- `SECURITY.md`: security notes for the real-model harness, which runs model-written code.
- `MANIFEST.json`: SHA-256 hashes of the folder's artefacts; its `note` field says to regenerate it with `tools/make_manifest.py`.
- `tools/make_manifest.py`: writes `MANIFEST.json`, hashing every tracked artefact of the folder except the manifest itself.
- `.gitignore`: makes git ignore `results/realmodel_v2/`, test caches and bytecode folders.
- `code/.gitignore`: makes git ignore the `figures/` and `results/` folders the harness writes when it is run inside `code/`.

**Harness and tests (`code/`)**

- `code/experiment_coscaling.py`: integrates the paper's model numerically and compares each experiment, E1 to E9 with E4b, with the closed-form prediction of the theorems, printing a pass or a fail for each; writes the ten figures `figures/exp1_phase_boundary.png` to `figures/exp9_residual_drift.png`, `results/verdicts.json` and `results/report.txt`.
- `code/test_coscaling.py`: regression tests that encode the theorems and derivation conditions as assertions on the harness.
- `code/test_coscaling_edge_cases.py`: edge-case tests of three theorem statements, with no import from the harness.
- `code/test_theorems_independent.py`: numerical regression checks of the corrected theorems, with no import from the harness, specified by `THEOREMS.md`.

**Theorem check record (`verification/`)**

- `verification/2026-10-19.md`: the record of the local checks run on the mathematical correction, with the baseline commit and the commands.
- `verification/check_mutations.py`: injects deliberate faults one at a time and checks that the test suite fails on each, without changing the real suite.
- `verification/Paper-X-Theorem-Verification-2026-10-19.zip`: an archive, at a fixed commit, of the theorem statements, the regression checks, the requirements, the check record and the fault-injection script, with the two licence files.
- `verification/Paper-X-Theorem-Verification-2026-10-19-MANIFEST.json`: the archive's manifest: its build rule, its commit and its exclusions.
- `verification/Paper-X-Theorem-Verification-2026-10-19-SHA256SUMS.txt`: SHA-256 sums of the archive's members.

**Real-model protocols and runners (`experiments/`)**

- `experiments/PROTOCOL.md`: the protocol of the real-model test of the paper's model, with the shared blinding and security rules and the estimator of β and k.
- `experiments/PROTOCOL_V2.md`: the replacement design for the earlier pilot, meant to produce a graded dataset with drift from which k and β can be estimated.
- `experiments/CONFIRMATORY_PROTOCOL_V2.md`: the detailed companion of `experiments/PROTOCOL_V2.md`, with its extra statistics and adversarial detail.
- `experiments/README_V2_UPGRADE.md`: an orientation table of the two real-model runners and their protocols.
- `experiments/RUN_REAL_MODEL_EXPERIMENT.md`: a launch runbook for the real-model experiment.
- `experiments/scripts/realmodel_coscaling.py`: the real-model runner of the pilot: in each round an engine model rewrites a candidate solution; capability is the fraction of hidden tests the executed candidate passes, misalignment is scored by a separate evaluator model, and in the coupled condition a corrector audits each round; writes `results/realmodel/<engine>_<timestamp>.json`.
- `experiments/scripts/agent_bridge_run.py`: runs the code path of `experiments/scripts/realmodel_coscaling.py` in steps, writing out the pending model requests and reading back the replies supplied for them until the run is done; writes `results/realmodel/claude-opus_<stamp>.json`.
- `experiments/scripts/realmodel_coscaling_v2.py`: the measurement harness of the replacement design: several task families, tiered hidden tests, evaluator panels, and static detectors combined with the panel scores; it writes to `results/realmodel_v2/`, which `.gitignore` excludes.
- `experiments/scripts/realmodel_coscaling_v3.py`: a harness that adds the conditions `eden_protocol`, `honey_meta` and `eden_full` to the coupled, decoupled and sham conditions, at three difficulty tiers; writes `results/realmodel_v3/<engine>_<timestamp>.json`.
- `experiments/scripts/estimate_exponents.py`: estimates β and k from a trajectory, checks the estimator on synthetic data with known exponents and applies it to the pilot run; writes `results/realmodel/exponent_estimates.json`.
- `experiments/scripts/drift_engine.py`: a runner designed to produce reward hacking: three tasks, three conditions, five seeds and eight rounds, with a persona framing and a higher temperature in the decoupled condition; writes `results/drift/gpt35_<stamp>.json`.
- `experiments/scripts/hard_tasks_v3.py`: the definitions of three tasks whose visible tests invite a shortcut that the hidden tests penalise.
- `experiments/scripts/run_hard_tasks.py`: runs those tasks through the model calls of `experiments/scripts/realmodel_coscaling_v3.py`: three tasks, three conditions, five seeds and eight rounds; writes `results/hard/deepseek-v4_<stamp>.json`.

**Results (`results/`)**

- `results/report.txt` and `results/verdicts.json`: the verdict report and the verdict file that `code/experiment_coscaling.py` writes.
- `results/redteam.md`: the report of an internal adversarial review of the paper on five fronts, with the adjudication of each objection; the note at its head says what the paper has corrected since.
- `results/realmodel/claude-opus_20260626T165919Z.json`: the pilot trajectory, recorded through the bridge (its `bridge` and `note` fields).
- `results/realmodel/claude-opus_corrector_probe.json`: the record of the corrector probe: the reward hack before and after one external corrector pass; `experiments/scripts/estimate_exponents.py` reads it.
- `results/realmodel/claude-opus_selftest.json`: a plumbing self-test of the runner; `REPRODUCIBILITY.md` calls such files a deterministic stub, not data.
- `results/realmodel/claude-opus_transcript.md`: the model outputs of the pilot run, kept as an audit log.
- `results/realmodel/exponent_estimates.json`: the output of `experiments/scripts/estimate_exponents.py`.
- `results/realmodel/REAL_MODEL_CLAUDE_RESULTS.md`: a write-up of the pilot run and its corrector probe, with the figure `figures/realmodel_claude.png`.
- `results/realmodel_v3/deepseek-v4_20260702T033809Z.json`, `results/realmodel_v3/deepseek-v4_20260702T033830Z.json` and `results/realmodel_v3/deepseek-v4_20260702T040313Z.json`: three runs of `experiments/scripts/realmodel_coscaling_v3.py`, each with its configuration; the `selftest` field reads true in the earlier two and false in the third.
- `results/drift/gpt35_20260702T171415Z.json`: the recorded run of `experiments/scripts/drift_engine.py`; the paper's text labels it "retained as a record and withdrawn as evidence of mechanism".
- `results/hard/deepseek-v4_20260702T111825Z.json`: the recorded run of `experiments/scripts/run_hard_tasks.py`.

**Figures (`figures/`)**

- `figures/exp1_phase_boundary.png`, `figures/exp2_speed_invariance.png`, `figures/exp3_coscaling_law.png`, `figures/exp4_hard_takeoff_grid.png`, `figures/exp4b_coordinate_artefact.png`, `figures/exp5_compounding_threshold.png`, `figures/exp6_vector_subspace.png`, `figures/exp7_stochastic.png`, `figures/exp8_validation.png` and `figures/exp9_residual_drift.png`: the ten figures that `code/experiment_coscaling.py` writes, one for each experiment.
- `figures/realmodel_claude.png`: the figure of `results/realmodel/REAL_MODEL_CLAUDE_RESULTS.md`.

**Original OSF deposit (`osf/`)**

- `osf/OSF_METADATA.md`, `osf/OSF_OVERVIEW.html`, `osf/OSF_OVERVIEW.pdf` and `osf/OSF_WIKI.md`: the texts of the paper's original OSF deposit, kept as `papers/README.md` describes; the two Markdown files open with a note that says so.

74 files, all listed.

Where the result and its status are stated: `papers/Paper-X-Coupled-CoScaling-Correction/`, this folder: the paper's exports and its `README.md`.

### `experiments/unit-83-counted-p21/`: Unit 83, a registered study unit

Unit 83 tests P21, the placement limb of Law V, the ARC Embedding Law, a law under test (its README's words). The home holds the exact registered implementation used in the unit's run, the original bench, the source and check records of its recovery amendment, and a guide to its results on OSF. Its evidence tier in `experiments/README.md`: registered.

**Guides and records**

- `README.md`: the folder's guide: what each part is, how to check the supplied bytes against the manifest, and what a repetition would require.
- `REPRODUCTION.md`: the difference between inspecting the source and an authorised repetition, the requirements of a repetition, and the original order of execution.
- `RECOVERY-AMENDMENT-DISCLOSURE.md`: the unchanged public companion to the unit's report, which discloses the recovery amendment made after registration.
- `PROVENANCE.json`: file identities and public source links.
- `MANIFEST.sha256`: the SHA-256 of each file of the folder apart from itself.

**Registered implementation (`registered/`)**

- `registered/registration_manifest.json`: the original manifest of the 43 registered source and configuration files.
- `registered/configs/p21.json`: the unit's configuration: its registration identities, seeds, engines, providers and prices.
- `registered/tools/make_manifest.py`: makes the registration manifest by command.
- `registered/p21base/__init__.py`: the base package the unit carries.
- `registered/p21base/common.py`: shared helpers: sorted-key JSON encoding, SHA-256 digests, timestamps, loading, atomic saving, and the error a refused gate raises.
- `registered/p21base/ledger.py`: a durable journal of spending reservations, one for each workspace.
- `registered/p21base/providers.py`: the non-streaming adapters for the unit's two engines, with no silent model fallback.
- `registered/p21base/registration_gate.py`: the admission checks that begin with the registration, using the standard library only.
- `registered/p21lane/__init__.py`: the lane's package.
- `registered/p21lane/admit.py`: admission, calibration and the pairs in flight.
- `registered/p21lane/analysis.py`: the analysis: its values, the mark, the second verdict and what is printed beside them.
- `registered/p21lane/charter.py`: the frozen wording every engine is shown.
- `registered/p21lane/collect.py`: the collector.
- `registered/p21lane/departures.py`: the planted departures: seventeen named operators and the rule that admits their entries.
- `registered/p21lane/forms.py`: the entries, the reference, the naive transcription and the case wording.
- `registered/p21lane/freeze.py`: the runtime freeze and its deposit.
- `registered/p21lane/generate.py`: the entry generator: cases, banks and the public sets, drawn from a seed.
- `registered/p21lane/lane.py`: the lane's interface: the entry points that the harness and the author call.
- `registered/p21lane/loop.py`: the loop: the registers of one block, round by round.
- `registered/p21lane/oracle.py`: every rule of the charter as a test on a finished entry, and the search that uses it.
- `registered/p21lane/preflight.py`: the pre-flight read-back that the lab standard requires.
- `registered/p21lane/prompts.py`: the prompt each role is sent.
- `registered/p21lane/reading.py`: the reading of a reply.
- `registered/p21lane/schedule.py`: the pairs and schedules.
- `registered/p21lane/score.py`: the scoring of entries.
- `registered/p21lane/seal.py`: how the bank is sealed and committed, with the answers kept off the registry.
- `registered/p21lane/sentences.py`: what each verdict lets the author print, and the report.
- `registered/p21lane/stats.py`: the numerical helpers of the analysis.
- `registered/tests/run_tests.py`: the runner of the authored tests.
- `registered/tests/support.py`: shared helpers of the tests (hand-made cases, builders of synthetic blocks and stand-in clients); not a test module.
- `registered/tests/test_analysis.py`: authored cases for the analysis.
- `registered/tests/test_charter.py`: an authored case that the frozen wording equals the design freeze's annex byte for byte.
- `registered/tests/test_collect.py`: authored cases for the collector, with a stand-in run of the provider client.
- `registered/tests/test_departures.py`: authored cases for the planted departures.
- `registered/tests/test_end_to_end.py`: one synthetic end-to-end case through the collector, the scoring, both looks and the report.
- `registered/tests/test_forms.py`: authored cases for the entries, the reference, the naive transcription and the case wording.
- `registered/tests/test_generate.py`: authored cases for the generator, the schedules and the public sets.
- `registered/tests/test_loop.py`: authored cases for the loop's registers and commit rules.
- `registered/tests/test_manifest_and_gate.py`: authored cases for the manifest, the registration gate and the lane's order.
- `registered/tests/test_prompts.py`: authored cases for the prompts.
- `registered/tests/test_providers.py`: authored cases for the adapters.
- `registered/tests/test_reading.py`: authored cases for the reader.
- `registered/tests/test_seal.py`: authored cases for the sealing rule.
- `registered/tests/test_stats.py`: authored cases for the numerical helpers.

**Original bench (`historical-bench/`)**

- `historical-bench/bench.py`: the exact sealed original bench, made by the bench factory from an earlier unit's bench; it prints and checks the original manual steps, and since its kit, seal and lab state are not bundled it is source for inspection.

**Recovery amendment (`historical-recovery/`)**

- `historical-recovery/recovery_amendment.py`: the helper of Recovery Amendment 01, run by the author for one incident, with no generation calls to a provider.
- `historical-recovery/test_recovery_amendment.py`: the helper's tests, on synthetic fixtures only.
- `historical-recovery/resume-unit83.zsh`: the amendment's shell script for resuming the run, written to be run by the author in his own terminal; the folder's README says these files are not a launcher for a new run.
- `historical-recovery/VALIDATION.md`: the check record of the helper, which covers the helper and not the experiment's outcome.
- `historical-recovery/CODE-SHA256.json`: the hashes of the amendment's files, including an addendum document that is not in this folder.

**Publication guide (`publication-guide/`)**

- `publication-guide/README.md`: a guide to the files of the registered run on the unit's OSF results component; by its own words it is not a new run package.
- `publication-guide/UPLOAD-SET-SHA256.txt`: the SHA-256 checksum list of the files the guide covers.

57 files, all listed.

Where the result and its status are stated: the unit's results record on OSF (DOI 10.17605/OSF.IO/98HKR), beside its registration (DOI 10.17605/OSF.IO/MN73W); the unit has no paper folder under `papers/`, and `publication-guide/README.md` lists the files of the record.

## Part 2: shared code, early scripts and the instruments

### `experiments/analysis-tools__Cross-Programme/`: Papers II, IV.a to IV.d and V

Holds shared analysis code that reads the result files of other homes. Its evidence tier in `experiments/README.md`: tooling.

- `REPRODUCE.md`: what the two tools read and how to run them.
- `analyze_alpha_align_v5.py`: reads the checkpoint or final files of `arc_alignment_scaling_v5.py` and computes the mean alignment score at each depth, the alignment exponent (grouped and individual) with bootstrap intervals, an effect size between the shallowest and deepest depths, a rank correlation, a Kruskal-Wallis test, a saturation fit and an exponent for each scorer.
- `per_scorer_check.py`: computes the alignment exponent separately for each scorer on each model, from the files of `arc_alignment_scaling_v4.py`, with no call to a provider.

3 files, all listed.

Where the result and its status are stated: no result belongs to this home; the papers it serves are in `papers/Paper-II-Experimental-Validation/`, `papers/Paper-IV-a-Baked-In-vs-Computed-Alignment/`, `papers/Paper-IV-b-Alignment-Saturation-at-Low-Depth/`, `papers/Paper-IV-c-ARC-Align-Benchmark/`, `papers/Paper-IV-d-The-Effect-of-Blinding-on-AI-Alignment-Evaluation/` and `papers/Paper-V-Stewardship-Gene/`.

### `experiments/shared/`: the experiment scripts

Holds the gateway adapter that the experiment scripts share. Its evidence tier in `experiments/README.md`: tooling.

- `README.md`: what the adapter does and the two environment variables it reads.
- `eden_gateway.py`: a client wrapper used when an experiment calls the shared gateway rather than a provider directly; standard library only.

2 files, all listed.

Where the result and its status are stated: no result belongs to this home; its README says that nothing here produces a result of its own.

### `experiments/arc-scaling-lineage/`: none directly

Holds four early standalone scaling scripts, kept as a record. Its evidence tier in `experiments/README.md`: superseded.

- `README.md`: the lineage of the four scripts, oldest at the top, and why they are not in a paper folder.
- `arc_live_experiment.py`: the earliest live run, against the deployed ask-book interface, comparing sequential with parallel reasoning.
- `arc_validation_experiment.py`: the same hypothesis rebuilt as a standalone script, testing whether sequential reasoning produces a scaling exponent above one.
- `arc_improved_experiment.py`: a revision of the live run, whose docstring lists what it changed.
- `arc_rigorous_experiment.py`: the last of the four, rebuilt against a stated external methodology.

5 files, all listed.

Where the result and its status are stated: no result belongs to this home; its README says that no result in any paper rests on these scripts.

### `instruments/`: the theory's registered propositions

Holds reference code for runs not yet made, and the recorded operating characteristics of that code. `experiments/README.md` gives it no evidence tier; its own README calls the kit "reference code and a design instrument".

- `README.md`: the folder's guide: the kit, its four execution modes, the battery evidence, and the files changed at publication.

**The kit (`arc-instruments/`)**

- `arc-instruments/README.md`: the kit's own guide: each module and the guidance item it implements, the runner, and what the tree contains.
- `arc-instruments/DEPOSIT.md`: the deposit checklist for the kit: what goes in and what does not.
- `arc-instruments/CITATION.cff`: the kit's citation file.
- `arc-instruments/.zenodo.json`: the kit's deposit metadata.
- `arc-instruments/requirements-test.txt`: the versions the test suite was run under.
- `arc-instruments/design_battery.py`: a battery of simulations over the plausible operating range of the P5 and P16 designs.
- `arc-instruments/p16_calibration.py`: the calibration of P16's complete decision family, with each rate reported as an interval.
- `arc-instruments/runner_battery.py`: runs the analysis code many times in simulated worlds whose answers are known, world by world; it wrote the battery evidence in `evidence/`.
- `arc-instruments/docs/INTEGRATION-REVIEW-2026-09-07.md`: a review of a repair to the kit; by its own words it changes no theory prediction and reports no empirical result.
- `arc-instruments/docs/P16-STAGE-ONE-BRIEF.md`: a stage-one plan for a P16 paper, with its result shell left unfilled.
- `arc-instruments/arc_instruments/<module>.py`: 22 files (the package's `__init__.py` and 21 modules), the kit's decision rules and design-sensitivity simulators; for example `arc-instruments/arc_instruments/verdicts.py`, the outcome regions for every proposition as functions on confidence intervals and events.
- `arc-instruments/arc_runner/<module>.py`: 22 files (the package's `__init__.py` and 21 modules), the runner that carries the P5 and P16 charters as code; for example `arc-instruments/arc_runner/cli.py`, its command line, whose four modes are its commands.
- `arc-instruments/tests/test_<subject>.py`: 32 files, the kit's test suite; for example `arc-instruments/tests/test_verdicts.py`.
- `arc-instruments/tests/conftest.py`: adds the kit's folder to the import path for the tests.
- `arc-instruments/tests/anchor_for_tests.py`: an anchoring service for the command-line tests only.
- `arc-instruments/tests/pool_for_tests.py`: a pool in the registered shape, for the command-line tests only.
- `arc-instruments/tests/trusted_fixtures.py`: test-only stand-ins for an outside grader and a reviewed calibration record.

**Battery evidence (`evidence/2026-09-06-runner-battery-a1-a9/`)**

- `evidence/2026-09-06-runner-battery-a1-a9/README.md`: what the battery is: the operating characteristics of the analysis code in simulated worlds, evidence about the code and not about any real system.
- `evidence/2026-09-06-runner-battery-a1-a9/RUNNER-BATTERY.json` and `evidence/2026-09-06-runner-battery-a1-a9/RUNNER-BATTERY.md`: the battery's two output files.
- `evidence/2026-09-06-runner-battery-a1-a9/MANIFEST.json`: pins the SHA-256 of both output files and of the eighty Python source files as the battery was run; the folder's README names the files whose hashes differ here, and why.

**Oversight assay (`oversight-assay/`)**

- `oversight-assay/README.md`: an exact arithmetic oversight assay, described as an instrument-development task with no real-model collection.
- `oversight-assay/assay.py`: the assay: an overseer receives an integer calculation and a proposed answer and must accept or reject it; it makes no model or network call.
- `oversight-assay/test_assay.py`: the assay's regression tests.

98 files: 22 listed singly, the rest by pattern.

Where the result and its status are stated: no result belongs to this home; its README says that nothing in this folder is a result about any real system.
