# Reproduce: the honey-architecture simulations, tests and self-modifying networks (Paper VI)

The scripts simulate three optimisation regimes, train small self-modifying networks under three acceptance rules, and pose four tests to language models. The result and its status are stated in the paper's folder, `papers/Paper-VI-Honey-Architecture/`.

## What is here

- `scripts/eden_honey_simulation.py`: the three-regime simulation; it writes `eden_honey_simulation_results.json` and three figures.
- `scripts/eden_self_modifying_ai.py`, `scripts/eden_self_modifying_ai_v2.py`, `scripts/eden_self_modifying_ai_v3.py` and `scripts/eden_self_modifying_ai_v4.py`: the self-modifying series. In that order they write `eden_selfmod_results.json`, `eden_selfmod_v2_results.json`, `eden_selfmod_v3_results.json` and `eden_selfmod_v4_results.json`, the names of the four files in `results/`, and the figures of the same names in `figures/`.
- `scripts/eden_honey_tests.py`: four tests against language models; it writes `eden_honey_test_results.json`. The batch files in `results/honey_provenance_pack/raw_results_generated/` carry the `metadata.experiment` value this script writes, under other names.
- `scripts/merge_honey_test_results.py`: merges the result files of separate model batches; it wrote `results/honey_provenance_pack/raw_results_generated/eden_honey_test_results.json`, whose `metadata.experiment` is the `eden_honey_tests_merged` it writes.
- `scripts/eden_honey_dashboard.jsx`: a browser component that reruns the simulation. `scripts/anti_sycophancy_recovered.py`: a module of prompts with a `render_prompt` function and no entry point.
- `scripts/generate_honey_provenance_pack.py`: built the provenance pack from files outside this repository. It writes to `validation/honey_provenance_pack/` under this folder; the pack is kept in `results/honey_provenance_pack/`.
- `results/reader/verify.py`: rebuilds each archived coupling table from its reader copy.

## What a run needs

- Python 3 with `numpy` and `matplotlib` (simulation and self-modifying series), `scipy` (the series from `scripts/eden_self_modifying_ai_v2.py` on, and the tests), and, for the tests' providers, `anthropic`, `openai` and `google.generativeai`. The merge script and `results/reader/verify.py` use the standard library only.
- Environment variables, by name, read by the tests only: `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`.
- `scripts/eden_honey_tests.py` calls model providers unless `--demo` is given. A run makes paid calls to model providers. The other scripts call none.

## How to run it

- `python3 scripts/eden_honey_simulation.py` and `python3 scripts/generate_honey_provenance_pack.py` take no arguments.
- `scripts/eden_self_modifying_ai.py`: `--cycles` (help `Number of self-modification cycles (default: 150)`), `--task-switch` (help `Cycles between task switches (default: 30)`), `--seed` (help `Random seed for reproducibility`), `--verbose` (help `Show every modification proposal`), `--output` (help `Output directory for plots and data`).
- `scripts/eden_self_modifying_ai_v2.py`: `--cycles`, `--task-switch`, `--seeds` (help `Number of random seeds (default: 10)`), `--output`, `--verbose`. `scripts/eden_self_modifying_ai_v3.py`: `--cycles`, `--task-switch`, `--seeds` (help `Number of random seeds (default: 20)`), `--output`. `scripts/eden_self_modifying_ai_v4.py`: `--seeds`, `--output`. Options shown here without help text have none.
- `scripts/eden_honey_tests.py`: `--test` (choices `all`, `alignment_scaling`, `monitoring_removal`, `coupling_degradation`, `eden_intervention`), `--model` (help `Single model to test`; a key of its `MODELS` table), `--demo` (help `Demo mode (no API keys needed)`), `--scorer` (help `Model to use for scoring`), `--output` (help `Output directory`).
- `scripts/merge_honey_test_results.py`: the input files (help `Input result JSON files to merge`) and `--output`, which is required (help `Destination for the merged JSON`).
- `results/reader/verify.py`: `python3 verify.py`, run from `results/reader/`, as `results/reader/README.md` says.
- The default `--output` of the self-modifying scripts and of the tests, and the output directory of `scripts/eden_honey_simulation.py`, is a fixed path written into the script, not a path in this folder.

## What it writes

- Simulation: `eden_honey_simulation_results.json`, `eden_honey_capability.png`, `eden_honey_safety.png`, `eden_honey_ratio.png`.
- Self-modifying series: `eden_selfmod_results.json`, `eden_selfmod_results.png` and `eden_selfmod_weights.png`; `eden_selfmod_v2_results.json`, `eden_selfmod_v2_results.png` and `eden_selfmod_v2_stats.png`; `eden_selfmod_v3_results.json`, `eden_selfmod_v3_results.png` and `eden_selfmod_v3_stats.png`; `eden_selfmod_v4_results.json` and `eden_selfmod_v4_scaling.png`; each in its `--output`.
- Tests: `eden_honey_test_results.json` in `--output`; the merge script writes to the `--output` it is given.

## What the recorded files hold

- `results/eden_selfmod_v2_results.json`, `results/eden_selfmod_v3_results.json` and `results/eden_selfmod_v4_results.json`: `metadata.n_seeds` `10`, `20` and `15`; `metadata.cycles` `150` and `180` (v2, v3); `metadata.task_switch` `30` (v3); `metadata.n_levels` `5` and `metadata.total_runs` `225` (v4); `metadata.timestamp` `2026-03-16`.
- `results/eden_selfmod_results.json`: `metadata.timestamp` `2026-03-16`; its seed and cycle count are not recorded in this folder.
- The plots: the ten in `figures/` and the ten of the same names directly under `results/` differ in their bytes; those under `results/` are byte-identical to the copies in `results/honey_provenance_pack/raw_results_generated/`. Nothing in this folder says which render the paper prints.
- The test files in `results/honey_provenance_pack/raw_results_generated/`: `metadata.models_tested` (`claude`, `deepseek`, `qwen3`, `gpt54` in `eden_honey_test_results_main4.json`; `gemini`, `grok` in `eden_honey_test_results_gemini_grok.json`), `metadata.scorer` `claude`, `metadata.demo_mode` (`true` in `eden_honey_test_results_demo.json` only), `metadata.timestamp` `2026-03-16`. The provider model identifiers are not recorded in this folder.

## Checking the record without a new run

`python3 verify.py`, run from `results/reader/`, rebuilds the three archived coupling tables in memory from their reader copies and checks them against the fingerprints in `field-key.json`; it calls no provider.

## What a new run is, and is not

For the tests: a new run is a new collection. Model outputs are not deterministic and the providers' models change, so a new run is not expected to return the recorded numbers. The recorded files are kept unchanged beside any new ones. The simulation draws no random numbers: its output is fixed by the values written into its `main()` (`steps = 80`, `R_MAX = 15.0`, `DRAG=0.15`). The self-modifying scripts set their seeds: `--seed` (default `42`) in `scripts/eden_self_modifying_ai.py`, and in the other three the seeds from `42` upward, one per run (`seed_list = list(range(42, 42 + ...))`); `scripts/eden_self_modifying_ai_v3.py` also sets seed `0` before it checks its task environment. The tests' `--demo` mode sets seed `42`.
