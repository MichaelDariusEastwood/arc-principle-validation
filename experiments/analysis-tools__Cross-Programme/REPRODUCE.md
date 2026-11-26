# Reproduce: the shared analysis tools (Papers II, IV.a to IV.d and V)

Two scripts re-analyse recorded alignment-scaling result files: one fits the alignment exponent across depth for each model, the other fits it for each scorer apart. They record no result of their own; the results they serve are stated in the papers' folders: `papers/Paper-II-Experimental-Validation/`, `papers/Paper-IV-a-Baked-In-vs-Computed-Alignment/`, `papers/Paper-IV-b-Alignment-Saturation-at-Low-Depth/`, `papers/Paper-IV-c-ARC-Align-Benchmark/`, `papers/Paper-IV-d-The-Effect-of-Blinding-on-AI-Alignment-Evaluation/` and `papers/Paper-V-Stewardship-Gene/`.

## What is here

- `analyze_alpha_align_v5.py`: reads the checkpoint or final files of `arc_alignment_scaling_v5.py` and computes, for each model, the mean alignment score at each depth, the alignment exponent with bootstrap intervals, an effect size between the shallowest and deepest depths, a rank correlation, a Kruskal-Wallis test, a saturation fit and an exponent for each scorer.
- `per_scorer_check.py`: computes the alignment exponent separately for each scorer on each model, from the checkpoint or final files of `arc_alignment_scaling_v4.py`.

## What a run needs

- Python 3 with `numpy` and `scipy` (the two scripts' import lines).
- `analyze_alpha_align_v5.py` reads `ARC_ALIGNMENT_RESULTS_GLOB` and `ARC_ALIGNMENT_RESULTS_DIR`, for its default search when no input is named; `per_scorer_check.py` reads no environment variable.
- Neither script calls a model provider.

## How to run it

- `python3 analyze_alpha_align_v5.py`: input paths (help `Checkpoint/final-result JSON files or directories containing them.`), `--glob` (help `Additional glob pattern(s) for input JSON discovery.`), `--output` (help `Optional output JSON path. Defaults to alpha_align_analysis.json next to the first input file.`).
- `python3 per_scorer_check.py` takes one or more file paths and no options; with none, it prints its usage line and stops.

## What it writes

- `analyze_alpha_align_v5.py`: `alpha_align_analysis.json` beside the leading input file, or at the `--output` path.
- `per_scorer_check.py`: no file; it prints its results.

## What the recorded files hold

The tools hold no recorded file. Their inputs are in this repository: the `v5_final_*.json` files in `experiments/alignment-scaling__Papers-IV-a-b-c-d/results/v5-final/` for `analyze_alpha_align_v5.py`, and the `v4_final_*.json` and `v4_checkpoint_*.json` files in `experiments/alignment-scaling__Papers-IV-a-b-c-d/results/v4/` for `per_scorer_check.py`. Those recorded files are described in `experiments/alignment-scaling__Papers-IV-a-b-c-d/REPRODUCE.md`.

## What a new run is, and is not

Neither tool calls a provider: a run reads the recorded files named above and computes from them. `analyze_alpha_align_v5.py` draws its bootstrap samples from a generator it seeds with `42` (`np.random.default_rng(42)`); `per_scorer_check.py` draws no random numbers.
