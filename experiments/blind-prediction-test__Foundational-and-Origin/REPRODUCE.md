# Reproduce - Blind Prediction Test (Foundational & Origin)

> **Provenance note (T2-research, 2026-07-05):** Assembled from this directory's
> committed script and outputs. Written to close a reproducibility gap (the dir
> held the test + a forensic analysis but no run doc). HOLD for T5 verification.

## What this is
The gold-standard **blind prediction test** for the ARC Principle: the ARC form
α = d/(d+1) is fixed *before* looking at each domain's measured exponent, then
scored against held-out data. Its stated purpose (script docstring): *"the test
that determines whether the ARC Principle is a discovery."*

## Environment
Python 3 with **NumPy**, **SciPy**, **matplotlib**.

## To reproduce
1. **Env:** `python3 -m pip install numpy scipy matplotlib`
2. **Run:** `python3 BLIND_PREDICTION_TEST.py`
3. **Compare** the printed/saved output against the committed `results_blind_prediction.txt`.
4. **Interpretation:** see `BLIND_TEST_FORENSIC_ANALYSIS.md` in this directory - the honest post-hoc analysis of what the blind test does and does not establish (read this before citing the result).

## Honest scope (LAW 1)
Read `BLIND_TEST_FORENSIC_ANALYSIS.md` first: the "blind" protocol's strength depends on the prediction genuinely preceding data inspection. Treat the result at the evidence tier the forensic analysis assigns it, not as standalone proof.

## Outputs
`results_blind_prediction.txt` (+ any figures matplotlib emits).
