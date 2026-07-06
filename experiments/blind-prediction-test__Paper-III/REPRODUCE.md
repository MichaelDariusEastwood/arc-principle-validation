# Reproduce - Blind Prediction Test (Paper III)

> **Provenance note (T2-research, 2026-07-05):** Assembled from this directory's
> committed script and forensic analysis. Written to close a reproducibility gap
> (the dir held the test + a forensic analysis but no run doc). HOLD for T5 verification.

## What this is
The gold-standard **blind prediction test** as applied in the Paper III
(alignment-scaling) context: the ARC form α = d/(d+1) is fixed before inspecting
each domain's measured exponent, then scored against held-out data. Its stated
purpose (script docstring): *"the test that determines whether the ARC Principle
is a discovery."*

## Environment
Python 3 with **NumPy**, **SciPy**, **matplotlib**.

## To reproduce
1. **Env:** `python3 -m pip install numpy scipy matplotlib`
2. **Run:** `python3 BLIND_PREDICTION_TEST.py`
3. **Interpretation:** read `BLIND_TEST_FORENSIC_ANALYSIS.md` in this directory - the honest analysis of what this blind test establishes for Paper III (read before citing).

## Honest scope (LAW 1)
Read `BLIND_TEST_FORENSIC_ANALYSIS.md` first. The blindness guarantee rests on the prediction preceding data inspection; cite the result at the tier the forensic analysis assigns, not as standalone proof. Note Paper III's own metabolic/species-group figures are separately flagged for recompute (see the 2026-07-05 metabolic reconciliation note) - do not conflate the two.

## Outputs
Printed/saved test output (+ any figures matplotlib emits).
