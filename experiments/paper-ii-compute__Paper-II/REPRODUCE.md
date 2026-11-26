# Reproduce: the sequential and parallel accuracy runners (Paper II)

The runners pose problems to language models at increasing reasoning depth (the sequential condition) and as a majority vote over several samples (the parallel condition), and fit a scaling exponent to each condition. The result and its status are stated in the paper's folder, `papers/Paper-II-Experimental-Validation/`.

## What is here

- `scripts/arc_paper_ii_validation_v2.py`: the multi-model runner. It wrote `results/arc_paper_ii_<model>.json` (five files) and `results/arc_paper_ii_combined.json`: it writes those names, and its `VERSION`, `2.1`, is the `version` the files record.
- `scripts/arc_paper_ii_validation_v1_deepseek.py`: a single-model runner with fixed token budgets and sample counts.
- `scripts/arc_validation_deepseek.py`: the same budgets and counts, with a gateway option. It imports its gateway helper from `archive.validation.eden_gateway`, a path that is not in this repository, and stops if that import fails; `experiments/shared/eden_gateway.py` defines the names it imports.
- `results/arc_deepseek_results_20260121_175028.json` has the file name that both single-model runners write; the recorded files do not say which of the two wrote it.
- `scripts/regenerate_figure_15.py`: redraws `figures/figure_15_complete_summary.png` from values written into the script.

## What a run needs

- Python 3 with `numpy` and `scipy`; for the providers a run uses, `openai`, `anthropic` and `google.genai` (the v2 runner's import lines). The single-model runners import `openai`; the figure script imports `matplotlib` and `numpy`.
- Environment variables, by name: `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY` and `XAI_API_KEY` (v2 runner); `DEEPSEEK_API_KEY` (both single-model runners; the gateway runner's messages also name `EDEN_GATEWAY_URL` and `EDEN_GATEWAY_API_KEY`).
- A run makes paid calls to model providers.

## How to run it

`python3 scripts/arc_paper_ii_validation_v2.py`, with these options:

- `--model`: choices `deepseek`, `openai`, `anthropic`, `gemini`, `groq-qwen3`, `grok-4-fast`, `all`; help `Model to test (default: all)`
- `--mode`: choices `sequential`, `parallel`, `both`; help `Test condition (default: both)`
- `--problems`: choices `all`, `tier1`, `tier2`, `tier3`; help `Problem set (default: all)`
- `--repeats`: help `Repeats per problem per depth (default: 3)`
- `--parallel-counts`: help `Comma-separated parallel sample counts (default: 1,3,5,9)`
- `--output-dir`: help `Output directory (default: ./arc_paper_ii_results)`
- `--verify`: help `Cross-verify answers with a different model (4-layer blinding)`
- `--resume`: help `Resume from checkpoint if available`

`scripts/arc_paper_ii_validation_v1_deepseek.py`, `scripts/arc_validation_deepseek.py` and `scripts/regenerate_figure_15.py` take no arguments.

## What it writes

- v2 runner: `arc_paper_ii_<model>.json` for each model and `arc_paper_ii_combined.json`, in `--output-dir`.
- Single-model runners: `arc_deepseek_results_<timestamp>.json`, in the directory they are run from.
- Figure script: `../figures/figure_15_complete_summary.png`, a path relative to the directory it is run from.

## What the recorded files hold

- `results/arc_paper_ii_<model>.json`: `model` `deepseek`, `gemini`, `grok-4-fast`, `groq-qwen3` or `openai`; `n_problems` `18` (`deepseek`, `gemini`, `grok-4-fast`) or `30` (`groq-qwen3`, `openai`), split into `n_tier1` and `n_tier2`; the depth labels are the keys of `sequential` (`minimal`, `standard`, `deep`, `exhaustive`, `extreme`; for `deepseek`, `thorough` in place of `deep` and `maximum` added; for `openai`, `low` added and no `extreme`); the sample counts are the keys of `parallel` (`1`, `3`, `5`, `9`); `timestamp` `2026-03-11` to `2026-03-12`. `results/arc_paper_ii_combined.json` holds the `groq-qwen3` entry only.
- `results/arc_deepseek_results_20260121_175028.json`: `model` `deepseek-reasoner`; `num_problems` `12`; `experiment_date` `2026-01-21`; token budgets `512`, `1024`, `2048` and `4096` (the `budget` of each `sequential` entry); sample counts `1`, `2` and `4` (the `N` of each `parallel` entry).
- Each depth entry records its trial count in `n`. The `--repeats` value, the seeds and the provider model identifiers behind the `model` labels are not recorded in this folder.

## What a new run is, and is not

A new run is a new collection. Model outputs are not deterministic and the providers' models change, so a new run is not expected to return the recorded numbers. The recorded files are kept unchanged beside any new ones. `scripts/regenerate_figure_15.py` makes no provider call: its figure is drawn from values written into the script, which sets no seed.
