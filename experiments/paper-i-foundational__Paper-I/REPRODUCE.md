# Reproduce: the ARC Principle research toolkit (Paper I)

The toolkit computes the scaling exponent α in U = I × R^α from data points written into the script, and draws three figures. The result and its status are stated in the paper's folder, `papers/Paper-I-ARC-Principle/`.

## What is here

- `scripts/arc_principle_research_toolkit.py`: estimates α by a two-point calculation and a power-law fit, runs a sensitivity analysis over token ratios and a falsification check, and writes a results file and three figures.
- `results_paper_i_toolkit.txt`: an empty file; nothing in this folder says what wrote it.

## What a run needs

- Python 3 with `numpy`, `scipy` and `matplotlib` (the script's import lines).
- No environment variable: the script reads none.
- No model provider: the script calls none.

## How to run it

`python3 scripts/arc_principle_research_toolkit.py` takes no arguments. It writes into the directory it is run from.

## What it writes

- `arc_principle_results.json`, from its function `export_results_json`.
- `arc_scaling_comparison.png`, `arc_sensitivity_analysis.png` and `arc_falsification_regions.png`, from its `main()`; where the figures cannot be drawn, it prints the error and goes on.
- It also prints a LaTeX table of the results.

## What the recorded files hold

No file the toolkit writes is in this folder. Files with two of the names it writes are in `papers/Paper-I-ARC-Principle/results/`: `arc_principle_results.json`, whose fields are `openai_o1`, `deepseek_r1`, `sensitivity` and `summary`, and `arc_scaling_comparison.png`. `arc_principle_results.json` records no date of its run.

## What a new run is, and is not

The toolkit makes no provider call. Its numbers come from the data points written into the script (its `DataPoint` entries). The script sets no seed; its function `bootstrap_confidence_interval`, which draws at random, is not called by `main()`.
