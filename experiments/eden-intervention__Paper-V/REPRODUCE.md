# Reproduce: the Eden Protocol intervention runners (Paper V)

The runners pose each prompt to a language model in two conditions, a control and the same prompt preceded by the Eden Protocol's three evaluation loops, at several reasoning depths, and have another model score each response. The result and its status are stated in the paper's folder, `papers/Paper-V-Stewardship-Gene/`.

## What is here

- `scripts/eden_protocol_scaling_test.py`: two subject models, `deepseek` and `gemini`, each scored by the other. It wrote the recorded files whose `version` is `1.0`: it writes their file name and that `version`.
- `scripts/eden_protocol_scaling_test_v2.py`: six subject models, each scored by a model other than itself. It wrote the recorded files whose `version` is `2.0`, by the same two marks.
- `scripts/eden_protocol_scaling_test_v3.py`: adds scoring by several models, two rewording passes before scoring, consensus scores, suppression conditions and purpose-loop variants. It writes files beginning `eden_v3_`, and none is in this folder.
- `results/eden_final_<model>_<timestamp>.json`: six files, one for each subject model.

## What a run needs

- Python 3 with `numpy` and `scipy`; for the providers a run uses, `openai`, `google.genai` and, in v2 and v3, `anthropic`.
- Environment variables, by name: `DEEPSEEK_API_KEY` and `GOOGLE_API_KEY` (all three runners); `XAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY` and `OPENAI_API_KEY` (v2 and v3).
- A run makes paid calls to model providers.

## How to run it

- `python3 scripts/eden_protocol_scaling_test.py`: `--model` (choices `deepseek`, `gemini`, `all`; help `Subject model to test`), `--analyse` (help `Analyse existing results file instead of running`), `--resume` (help `Resume from checkpoint`), `--output-dir` (help `Output directory`).
- `scripts/eden_protocol_scaling_test_v2.py`: the same four options, with `--model` choices `deepseek`, `gemini`, `grok`, `claude`, `groq`, `gpt`, `all`; and `--scorer` (choices the six models; help `Override the blind scorer model (default: auto-select)`) and `--list-models` (help `List all supported models and exit`).
- `scripts/eden_protocol_scaling_test_v3.py`: `--model` (the v2 choices), `--analyse`, `--resume`, `--output-dir`, `--max-scorers`, `--include-suppression`, `--retain-text`, `--purpose-mode` (choices `task`, `grand`, `hybrid`), `--ethics-kernel` (choices `none`, `cross_tradition`), `--ternary-prototype` and `--list-models`; none of its options carries help text.

## What it writes

- v1 and v2: `eden_checkpoint_<model>.json` and `eden_final_<model>_<timestamp>.json`, in `--output-dir` (default `./eden_results`).
- v3: `eden_v3_checkpoint_<model>_<configuration>.json` and `eden_v3_final_<model>_<configuration>_<timestamp>.json`, in the same default directory.

## What the recorded files hold

- `model`: `claude`, `deepseek`, `gemini`, `gpt`, `grok`, `groq`; `version`: `1.0` (`deepseek`, `gemini`) or `2.0` (the other four).
- `scorer`: `gemini` for each subject except `gemini`, whose scorer is `deepseek`.
- `depth_configs`: `minimal`, `standard`, `thorough`, `exhaustive`; `deep` in place of `thorough` for `gemini`.
- `n_prompts`: `10`; `conditions`: `control` and `eden`.
- `timestamp`: `2026-03-12`.
- Seeds and the provider model identifiers behind the `model` labels: not recorded in this folder.

## Checking the record without a new run

`--analyse <file>` of `scripts/eden_protocol_scaling_test.py` and of `scripts/eden_protocol_scaling_test_v2.py` reads a recorded results file of their format and prints its analysis, with no provider call.

## What a new run is, and is not

A new run is a new collection. Model outputs are not deterministic and the providers' models change, so a new run is not expected to return the recorded numbers. The recorded files are kept unchanged beside any new ones.
