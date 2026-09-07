# Reproducing Paper VIII

What each piece needs, and the two places where this folder's layout differs from the layout
the code expects. Read those two notes before running anything, because in both cases the
code will run and do the wrong thing rather than stop.

## Requirements

| Piece | Needs |
|-------|-------|
| architectural simulation and its tests | Python 3.11 or later, `torch>=2.2`, `numpy`, `scipy`, `pyyaml`, `matplotlib`, `pytest` |
| behavioural runners | Python 3.11 or later, network access to the foundation and judge model endpoints, and a key for each. `certifi` is optional in the patched copies and required in the originals |
| weight experiment | Python 3.11 or later, an MLX or equivalent LoRA fine-tuning stack, and the Qwen 2.5 3B Instruct base model |

No key is stored anywhere in this repository. Every runner reads its credentials from the
environment.

## The two layout notes

**1. The tests and the package are in separate folders here.** The harness ships a
`pyproject.toml` that puts `src` on the path and points pytest at a sibling `tests`
directory. This deposit places the tests under `tests/` at the top of this folder instead, so
that a reader finds them without opening a package. Run them by naming the source tree:

```bash
cd papers/Paper-VIII-The-Load-Bearing-Proof/experiments
PYTHONPATH=code/architectural-gated-self-mod/src python3 -m pytest tests/
```

To use the package's own configuration instead, install it and run pytest from inside it,
after copying `tests/` back beside `pyproject.toml`:

```bash
cd papers/Paper-VIII-The-Load-Bearing-Proof/experiments/code/architectural-gated-self-mod
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp -r ../../tests ./tests
pytest
```

**2. `run_eval.py` looks for run summaries this folder does not hold.** The evaluation step
globs `<results-dir>/*/summary.json` and exits with `No summary.json files found` when it
finds none. The twelve summaries are not deposited here because they are already public,
under different names, in the paper's results folder. Copy them back into the shape the
script expects before running it:

```bash
cd papers/Paper-VIII-The-Load-Bearing-Proof/experiments
for f in ../results/summary_*.json; do
  run=$(basename "$f" .json); run=${run#summary_}
  mkdir -p "runs/$run" && cp "$f" "runs/$run/summary.json"
  cp "run-ledgers/$run/ledger.jsonl" "runs/$run/ledger.jsonl"
done
PYTHONPATH=code/architectural-gated-self-mod/src \
  python3 code/architectural-gated-self-mod/src/run_eval.py --results-dir runs
```

That reconstructs the twelve run directories exactly as the harness wrote them, ledger and
summary together. `runs/` is working output; it is not part of the deposit.

## Running the architectural simulation from scratch

```bash
cd papers/Paper-VIII-The-Load-Bearing-Proof/experiments/code/architectural-gated-self-mod
python3 src/run_train.py --config configs/default.yaml
```

`configs/quick.yaml` is the smoke configuration. A fresh run writes a new dated directory; it
does not overwrite a recorded one.

## Regenerating the weight-experiment corpora

`code/weight/generate_training_data.py` writes its four files into a `training-data`
directory beside itself. The corpora as generated and used are deposited at
`results/weight/training-data/`. Running the generator therefore writes a fresh set to
`code/weight/training-data/` and leaves the recorded set untouched. Compare the two rather
than replacing one with the other; a recorded corpus is evidence and is never overwritten.

## Running a behavioural evolution

Each runner takes its parameters on the command line. The patched copies write a
`<output>.error.json` and exit non-zero on an uncaught exception rather than finishing
silently, which is the only behavioural difference between them and the originals.

```bash
cd papers/Paper-VIII-The-Load-Bearing-Proof/experiments/code/behavioural-dgm
python3 variants/eden_dgm_v3_gpt54.py --help
```

Results depend on live model endpoints and on the judge model named in the run. A rerun today
will not reproduce a recorded run byte for byte, and it is not meant to. The recorded outputs
are the artefact; the code is deposited so that the procedure behind them can be read and
criticised.

## What a rerun does and does not settle

These are three pilot experiments. Rerunning one tests whether the procedure behaves as
described. It does not replicate the finding, because a replication needs a design fixed in
advance, a scorer that cannot see the condition, and a pre-stated analysis. That study has
been written and dated as a draft registration and awaits the author's own submission.
