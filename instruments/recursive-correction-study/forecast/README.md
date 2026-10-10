# Frozen transition baselines for a later prediction study

**Status: instrument development only.** The included demonstration contains synthetic fixtures, not observations from an AI model. This module implements adopted comparison models. It does not introduce a new ARC law, register an experiment or establish model safety.

The implementation uses Python's standard library and makes no network or model calls. Its purpose is to turn the later Stage B comparison into something checkable: forecasts are frozen from calibration, then separately collected outcomes are scored against both pooled and predeclared-stratum rivals.

## Quick start

From the parent continuation directory:

```bash
python3 -m unittest discover -s forecast -p 'test_*.py' -v
python3 forecast/frozen_baselines.py demo --out-dir forecast-demo-new
```

The destination must not already exist. The demonstration writes `plan.json`, `calibration.json`, `ledger.json`, `outcomes.json`, `evaluation.json` and `mixture-counterexample.json`. A second run needs another destination. The main console result shows the exact theoretical mixture counterexample, not a new empirical finding.

To use separately prepared development records:

```bash
python3 forecast/frozen_baselines.py freeze \
  --plan plan.json --calibration calibration.json --out ledger.json

python3 forecast/frozen_baselines.py score \
  --ledger ledger.json \
  --expected-ledger-sha256 <digest-retained-before-outcomes> \
  --outcomes outcomes.json --out evaluation.json
```

The expected digest is deliberately required as a separate argument. Retain it and the ledger before revealing prediction outcomes. A genuine confirmatory project additionally needs an authentic external registration or custody record and its own reviewed analysis. The local checker cannot manufacture that evidence.

## Why two baselines are necessary

With error probability `e`, a correct-to-unsuccessful transition rate `a` and an error-to-correct repair rate `b`, the adopted recurrence is:

\[
e_{t+1}=(1-e_t)a+e_t(1-b).
\]

This is an established two-state transition model. Yang and colleagues published a self-correction scaling analysis in EMNLP 2025 in which a first correction round supplies parameters for later accuracy forecasts. Liu and Meng's 2026 preprint also treats self-correction through feedback-control dynamics. These are required comparators, not discoveries originating in this module. [1,2]

Pooling tasks can introduce a misleading apparent change in correction strength. Consider two equally weighted strata, initially all wrong, with no regressions and repair probabilities 0.8 and 0.2. Each stratum has stationary transitions.

\[
e_1=\tfrac12(0.2+0.8)=0.5,
\qquad
e_2=\tfrac12(0.2^2+0.8^2)=0.34.
\]

Pooling the first-round repair probability into `b=0.5` predicts `e_2=0.25`. The remaining errors increasingly belong to the hard stratum. No physical resource crisis, capability exponent change or nonstationary within-stratum corrector is needed to produce that discrepancy.

The module therefore fits both a pooled rival and a rival that predicts within frozen strata before aggregation. The example is a mathematical counterexample to naive pooling, not evidence that this mechanism explains Eastwood's current data.

## Plan and data contracts

`demo/plan.json`, when the demonstration is generated, provides a complete schema example. All keys are checked exactly. The plan binds:

- The target population and whether records are synthetic or collected.
- Calibration completion time, a stratum-rule hash and a scorer hash.
- Named configurations with a fixed model revision label and policy hash.
- Sorted horizons and fixed target stratum weights summing to one.
- Explicit transition smoothing parameters and the missing-output rule.
- Calibration and prediction task IDs, content hashes, cluster IDs, strata and initial-state basis.
- Any control/treatment configuration pairs whose effect forecasts will be reported.

Calibration initial states must be checked binary values. Prediction initial error may be a declared probability or a known starting-state indicator. Its basis must precede the predicted outcomes. This allows conditional forecasts for known defective artefacts; it does not permit using final holdout correctness to choose the starting state or stratum.

Calibration and prediction must be disjoint in task ID, content hash and cluster ID. Multiple task instances may belong to a single cluster. A cluster belongs to exactly one predeclared stratum. Every stratum must occur in both partitions. A string label and hash are declarations, not independent verification that the task distribution or scoring construct is valid.

Calibration supplies one row per task and configuration. Outcomes supply one row per prediction task, configuration and horizon. Each row repeats the expected task hash. Duplicate, foreign or incomplete panels are rejected. A missing model response must have an explicit `final_status: missing` row; it must not disappear from the panel.

The accepted final statuses are `correct`, `incorrect`, `invalid` and `missing`. The latter three are unsuccessful under this output contract. They remain separately counted. This matches a delivered-output endpoint and must not be equated with a harmful edit or general semantic misalignment.

## Fitting and honest limitations

For every configuration and stratum, fit `a` from initially correct cases and `b` from initially incorrect cases. The pooled comparator uses the corresponding combined calibration exposures. Raw event numerators, state exposures, invalid counts and missing counts are retained.

With zero pseudocounts, the estimator is the empirical proportion. With success and failure pseudocounts both one, it is `(events + 1)/(exposure + 2)`. This is a prespecified smoothed plug-in rate. The module does not claim a posterior credible interval, validate independent Bayesian sampling assumptions or propagate calibration uncertainty through the multi-step forecast.

Both starting-state exposures must be positive in every required configuration/stratum. Smoothing cannot quietly replace an unobserved transition class with a prior-only prediction. An absent class causes an explicit failure. Stratum construction, smoothing and target weights must be settled on generator structure or calibration before holdout outcomes, not selected because they make a favoured model win.

Time-homogeneous two-state transport remains an empirical assumption. Unobserved history, changing prompts, task selection, repeated failed repairs or resource changes can violate it. The stratified rival addresses one mixture mechanism; it is not an exhaustive model of recursive correction.

Each calibration row represents exactly one correction opportunity. Calling an invalid or missing reply an unsuccessful output does not specify how a later revision starts from it. A genuine multi-step collector must declare whether it carries forward a previous valid artefact, uses a sentinel state, retries under a fixed rule or terminates. It must then check that its calibration transition definition adequately represents that continuation policy.

## Frozen ledger and chronology

The ledger contains the plan, calibration records, their hashes, counts, derived rates and every task/configuration/horizon forecast. Its digest binds the entire ledger. Verification recomputes the rates and forecasts from the frozen inputs, so simply editing a prediction and recomputing the ledger hash cannot pass the derivation check.

Calibration completion must precede the local forecast freeze. Outcome rows must declare collection strictly after that freeze and no later than evaluation time. JSON duplicate keys and non-finite numeric values are rejected. Writing uses exclusive creation, protecting previous outputs from overwrite.

These are documentary checks. A researcher who fabricates a whole plan, calibration file and timestamp can still fabricate a story. External custody, actual collection records, code review and an authentic registry receipt are required to support genuine prospective claims. The software expressly remains `confirmatory_eligible: false` with no proposition verdict.

The ledger binds plan and data bytes and recomputes the stated derivation. It does not itself contain the implementation source hash. The delivery's `verification.json` records the exact module and test hashes; the complete package manifest and any reviewed Git commit should be retained alongside the frozen ledger. A future real collection must bind those software identities in its registration record as well.

## Paired Brier evaluation

For each task outcome, the unsuccessful-output indicator is `z`, and a model's predicted error probability is `p`. The Brier loss is `(p-z)^2`. The same task is scored under both frozen rivals. Their paired loss difference is pooled minus stratified, so a positive difference favours the stratified rival on that score.

The implementation first averages instances within a task cluster, then averages clusters within each stratum, then uses the predeclared target stratum weights. Configurations and horizons remain paired within clusters. The overall summary weights configurations and horizons equally; separate configuration/horizon scores are also returned. A different target estimand requires a disclosed plan revision before outcomes.

The exploratory 95% percentile interval resamples entire clusters within strata. It conditions on the frozen fitted forecasts and does not propagate calibration-rate uncertainty. It is not a simultaneous interval or a validated confirmatory decision procedure. Fewer than two clusters in any stratum produce no interval. The independent primary-analysis module and a future Stage B registration must specify any confirmatory multiplicity and coverage requirements.

For predeclared intervention pairs, the report compares predicted and observed treatment-minus-control error differences at each horizon, using the same cluster and stratum weights. Absolute effect-forecast error is descriptive. Pairing the outputs alone does not establish a causal effect: the actual intervention assignment and collection protocol must justify that interpretation.

## What the tests check

The test suite checks the 0.34 versus 0.25 counterexample, state exposures, smoothing, absent-state refusal, shared-cluster and duplicate-content rejection, fixed versions, frozen derivation, external digest pinning, local chronology, complete paired panels, invalid/missing denominators, proper squared-probability loss, target weights, cluster aggregation, intervention-pair arithmetic, order-invariant paired bootstrap and exclusive writes.

Passing these tests verifies implementation examples and failure handling. It supplies no new model performance, forecast generalisation, registration acceptance or established ARC prediction.

## References

[1] Yang, Z., Zhang, Y., Wang, Y., Xu, Z., Lin, J. and Sui, Z. (2025). *A Probabilistic Inference Scaling Theory for LLM Self-Correction*. Proceedings of EMNLP, 13573-13587. https://aclanthology.org/2025.emnlp-main.685/ . DOI: https://doi.org/10.18653/v1/2025.emnlp-main.685 . Primary publication record checked 10 October 2026.

[2] Liu and Meng (2026). *Self-Correction as Feedback Control*. Preprint. https://arxiv.org/abs/2604.22273 . Its presence in the comparison set does not establish priority over Yang et al. or validate the present empirical mapping.
