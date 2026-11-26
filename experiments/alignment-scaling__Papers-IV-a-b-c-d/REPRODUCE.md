# Reproduce: the alignment-scaling runners (Papers III, IV.a, IV.b, IV.c and IV.d; the working report)

The runners pose prompts to language models at several reasoning depths, have other models score the responses, and fit an alignment-scaling exponent across depth. The result and its status are stated in the papers' folders: `papers/Paper-III-Alignment-Scaling-Problem/`, `papers/Paper-IV-a-Baked-In-vs-Computed-Alignment/`, `papers/Paper-IV-b-Alignment-Saturation-at-Low-Depth/`, `papers/Paper-IV-c-ARC-Align-Benchmark/`, `papers/Paper-IV-d-The-Effect-of-Blinding-on-AI-Alignment-Evaluation/` and `papers/ARC-Alignment-Scaling-Working-Report/`.

## What is here

Six runners. The recorded files were written by the five numbered `arc_alignment_scaling_` runners, as their file name patterns show; `scripts/arc_eden_v6_runner.py` wrote none of the files in this folder.

- `scripts/arc_alignment_scaling_v1.py`: four depths per model and one scoring model; wrote `results/v1/alignment_raw_<model>_<timestamp>.json`.
- `scripts/arc_alignment_scaling_v2.py`: depth set through each provider's own setting, a scorer other than the subject, scores from 0 to 100 and capability prompts; wrote `results/v2/v2_raw_<model>_<timestamp>.json`.
- `scripts/arc_alignment_scaling_v3.py`: dilemma prompts scored for the quality of reasoning; wrote `results/v3/v3_raw_<model>_<timestamp>.json`.
- `scripts/arc_alignment_scaling_v4.py`: three scorers per response and a save after every prompt; wrote the `v4_checkpoint_`, `v4_final_` and `v4_analysis_` files in `results/v4/`.
- `scripts/arc_alignment_scaling_v5.py`: each response reworded by other models before scoring by several models, with pre-flight checks; wrote `results/v5-final/v5_final_<model>_<timestamp>.json`. `results/v5-final/v5_final_groq-qwen3_20260312_073302 copy.json` is a byte-identical second copy of one of them.
- `scripts/arc_eden_v6_runner.py`: a shared platform of twelve named experiments; it writes files beginning `v6_`, and none is here.

## What a run needs

- Python 3 with `numpy` and `scipy`; for the providers a run uses, `openai`, `anthropic`, and `google.generativeai` (v1 to v3) or `google.genai` (v4 to v6).
- Environment variables, by name: `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` (every runner); `GROQ_API_KEY` and `XAI_API_KEY` (v5, v6); `ARC_EDEN_RESULTS_DIR` (v6, where it looks for earlier result files).
- A run makes paid calls to model providers.

## How to run it

- `python3 scripts/arc_alignment_scaling_v1.py`: `--model` (choices `deepseek`, `openai`, `anthropic`, `gemini`, `all`, `dry-run`; help `Which model to test (default: dry-run)`), `--repeats` (help `Number of repeats per prompt per depth (default: 1, use 3 for publication)`), `--output-dir` (help `Output directory for results`).
- `scripts/arc_alignment_scaling_v2.py`: `--model` (the v1 choices; help `Model to test`), `--scorer` (choices `deepseek`, `openai`, `anthropic`, `gemini`; help `Model to use for scoring (default: auto-select different model)`), `--repeats` (help `Repeats per prompt per depth (3 for publication)`), `--output-dir` (help `Output directory`).
- `scripts/arc_alignment_scaling_v3.py`: `--model`, `--scorer`, `--repeats`, `--output-dir`, with the v2 choices and no help text.
- `scripts/arc_alignment_scaling_v4.py`: `--model`, `--scorer`, `--scorer2`, `--scorer3`, `--repeats`, `--resume` and `--output-dir`, with no help text, and `--skip-preflight` (help `Skip API health checks (use if you know APIs are working)`).
- `scripts/arc_alignment_scaling_v5.py`: `--mode` (choices `fresh`, `rescore-v4`, `laundering-control`, `dry-run`; help `Operating mode (default: dry-run)`), `--model` (choices `deepseek`, `openai`, `anthropic`, `gemini`, `groq-qwen3`, `grok-4-fast`, `all`; help `Subject model (required for --mode fresh)`), `--v4-dir` (help `Directory containing v4 checkpoint files (for rescore/laundering modes)`), `--repeats` (help `Number of repeats per prompt (default: 1)`), `--resume` (help `Resume from checkpoint`), `--skip-preflight` (help `Skip API health checks`), `--no-laundering` (help `Skip response laundering (for comparison)`), `--prefill` (choices `none`, `4k`, `8k`, `both`, `eden`, `eden-4k`, `eden-8k`, `full`; help begins `Context pre-fill condition.`), `--output-dir` (help `Output directory for results`); it also accepts `--scorer`, `--scorer2` and `--scorer3`, hidden from its help.
- `scripts/arc_eden_v6_runner.py`: `--experiment` (choices `arc_compute_scaling`, `baseline_alignment`, `capability_control`, `deception_hawthorne`, `eden_intervention`, `laundering_control`, `laundering_leakage`, `loop_ablation`, `null_baseline`, `purpose_kernel`, `rescore_legacy`, `suppression_residual`) and `--model` (choices `deepseek`, `gemini`, `grok`, `claude`, `groq`, `gpt`, `all`), with `--output-dir`, `--resume`, `--max-scorers`, `--include-suppression`, `--retain-text`, `--holdout-fraction`, `--split` (choices `all`, `train`, `holdout`), `--repeats`, `--seed`, `--pre-registered-id`, `--export-replication-pack`, `--legacy-path`, `--legacy-glob`, `--dry-run`, `--show-preflight`, `--analyse`, `--list-models` and `--list-experiments`; none of its options carries help text.

## What it writes

- v1: `alignment_raw_<model>_<timestamp>.json` and `alignment_scaling_combined_<timestamp>.json`; v2: `v2_raw_<model>_<timestamp>.json`; v3: `v3_raw_<model>_<timestamp>.json`; v4: `v4_checkpoint_<model>.json`, `v4_final_<model>_<timestamp>.json`, `v4_analysis_<model>_<timestamp>.json`; v5: `v5_checkpoint_<model>.json`, `v5_final_<model>_<timestamp>.json`, `v5_analysis_<model>_<timestamp>.json`, `v5_rescore_<model>.json`, `v5_laundering_control.json`, `v5_ethics_report_<timestamp>.json`. Each goes to the runner's `--output-dir`, and `<model>` is the name of the runner's adapter for that model.
- v6: `v6_final_<slug>_<timestamp>.json`, `v6_checkpoint_<slug>.json`, `v6_publication_card_<slug>.json` and `.md`, `v6_deployment_risks_<slug>.json` and `.md`; a folder `v6_replication_pack_<slug>` when `--export-replication-pack` is given, and a folder `v6_human_audit_pack_<slug>_<timestamp>` when its audit queue is not empty.

## What the recorded files hold

- `model`: `claude-sonnet`, `deepseek-r1`, `gemini-flash`, `openai-o1` (v1); `claude-sonnet`, `deepseek-r1` (v2, v3); `claude-opus`, `deepseek-r1`, `gemini-flash`, `openai-gpt54` (v4); `claude-opus`, `deepseek-r1`, `gemini-flash`, `grok-4-fast`, `groq-qwen3`, `openai-gpt54` (v5).
- `depth_configs`: in v1, label and token-budget pairs that differ by model (`["shallow", 256]` in `alignment_raw_claude-sonnet_20260310_194845.json`, `["shallow", 1024]` in `alignment_raw_openai-o1_20260310_193118.json`); in v2 to v5, labels from `minimal` to `exhaustive`, with `extreme` and `maximum` in some v5 files.
- Scorers: `scorer` (v2, v3); `scorer1` to `scorer3` and `n_scorers` (v4); `blind_scorers` and `n_scorers`, with `blinding_protocol`, `laundering` and `prefill_conditions` (v5). The papers state what the scoring was.
- `repeats`: `1` (v4, v5); not recorded in this folder for v1 to v3.
- Provider model identifiers: `model_api_ids` and `scorer_api_ids` in the v5 files of `claude-opus`, `grok-4-fast` and `groq-qwen3`; not recorded in this folder for the other runs.
- Prompt counts and seeds: not recorded in this folder as fields; each response is an entry of `raw_responses` (v1), `alignment_data` and `capability_data` (v2, v3) or `data` (v4, v5).
- Dates: `timestamp` `2026-03-10` (v1 to v3), `2026-03-11` (v4), `2026-03-11` to `2026-03-12` (v5).

## Checking the record without a new run

`experiments/analysis-tools__Cross-Programme/analyze_alpha_align_v5.py` reads `v5_final_*.json` files (here, `results/v5-final/`), and `experiments/analysis-tools__Cross-Programme/per_scorer_check.py` reads `v4_final` or `v4_checkpoint` files (here, `results/v4/`); neither calls a provider.

## What a new run is, and is not

A new run is a new collection. Model outputs are not deterministic and the providers' models change, so a new run is not expected to return the recorded numbers. The recorded files are kept unchanged beside any new ones.
