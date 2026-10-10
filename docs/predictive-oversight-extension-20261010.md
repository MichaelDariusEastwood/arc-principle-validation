# Predictive oversight: an instrument-development extension

10 October 2026. Proposed measurement support, not a registration or empirical result.

This extension adds a small offline forecasting and diagnostic module beside the
existing [exact-arithmetic assay](../instruments/oversight-assay/README.md). The
assay, public task bank and original scoring rules remain unchanged.

## Contribution and scope

The module scores existing response envelopes through the assay's exact oracle,
then estimates two development transition frequencies: a correct proposal becoming
unsuccessful under the strict output contract, and an incorrect proposal being
successfully repaired. Missing or malformed responses count as unsuccessful
outputs. They are separately reported and must not be described as observed
harmful edits to the original artefact.

For declared transition probabilities a and b and an initial error proportion e,
the adopted baseline iterates:

`e_next = (1 - e) * a + e * (1 - b)`.

The two-state feedback-control approach is a serious prior model, including Liu
and Meng's [Self-Correction as Feedback Control](https://arxiv.org/abs/2604.22273)
(2026 preprint). The recurrence is credited as an adopted rival, not claimed as
a new ARC law. The proposed scientific question is whether independently
calibrated models predict held-out trajectories and intervention effects.

The present output is a plug-in forecast. It does not propagate parameter
uncertainty, prove that a one-pass panel represents iterative states, or define
how a later iteration continues after an invalid response. These matters require
a separately fixed trajectory protocol. Every forecast retains
`confirmatory_eligible: false` and `proposition_verdict: null`.

## Offline commands

From `instruments/oversight-assay/`:

```bash
python3 -m unittest -v test_assay test_predictive
python3 predictive.py forecast --panel pilot-panel.json --responses pilot-responses.jsonl --initial-error 0.5 --depth 4 --out development-forecast.json
python3 predictive.py range --values capability-values.json
python3 predictive.py zero-events --n 100
```

The panel and response paths are inputs collected or generated using the existing
assay instructions. The command's `0.5` and `4` are illustrative development
settings, not a frozen study design or claimed measurement. No result file is
overwritten. A capability-values file contains a JSON list such as `[1, 1, 1]`;
constant observations yield zero range and cannot identify a scaling exponent.
Variation alone is also not a sufficient identification test.

The zero-events utility gives the one-sided exact binomial upper bound under
fixed sample size, independent identically distributed trials, complete detection
and a defined population. It is not automatically valid under optional stopping,
correlated failures or deployment shift. A run with zero observed failures cannot
certify arbitrary future AI safety.

## Interpretation boundaries

The full arithmetic specification already determines the correct answer. Giving
a computationally bounded model a diagnostic result can improve its performance
without adding conditional Shannon information for an ideal observer. A literal
missing-information experiment would need an independently sampled hidden state,
a declared observation channel and a separate analysis.

Neither the transition frequencies nor the range diagnostic estimates the ARC
correction-strength exponent, drift exponent, correction leverage, unused service
capacity, energy efficiency or value persistence. The public development bank
cannot be reclassified as an unseen confirmatory bank. No P1-P22 entry is edited
or newly designated as decided.

## Verification

The new tests cover identity and alternating recurrences, correction that helps
or harms, perfect and useless policies, malformed and missing replies, hash
binding, mismatched panels, constant-capability non-identification, exact
zero-event arithmetic, invalid inputs and preservation of existing output files.
They verify implementation behaviour. They are not model experiments.

No provider call, training job, payment, registration or empirical output was
created by this extension. See the separate
[pilot scope correction](paper-x-pilot-scope-correction-20261010.md) for why the
existing flat Paper X pilot does not supply a favourable exponent margin.
