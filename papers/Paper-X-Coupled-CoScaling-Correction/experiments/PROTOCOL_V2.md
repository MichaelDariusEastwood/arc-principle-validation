# Paper X real-model experiment v2 - confirmatory protocol

## Purpose

This protocol replaces the single-task first-run design with a stronger, reviewer-resistant empirical test of the Paper X claim:

> In recursive self-improvement, the relevant safety variable is not raw speed but whether correction co-scales with capability faster than drift/reward-hacking pressure.

This protocol does **not** test the mathematical proof. The proof is internal to the model. This tests whether the mechanism appears in real model trajectories.

## Primary changes from the current repository experiment

1. **Multi-task benchmark**: arithmetic parser, Roman numeral conversion, CSV statistics.  
2. **Three conditions**: decoupled, sham-extra-compute, coupled-corrector.  
3. **Two speeds**: steady and fast.  
4. **Always-on corrector in coupled arm**, not only when $D\ge3$.  
5. **Static reward-hack detector plus blinded model panel**, reducing dependence on model-scored ethics.  
6. **Matched-pair analysis** by task × speed × seed.  
7. **Bootstrap confidence intervals**, not just point slopes.  
8. **Corrector observations at multiple capability levels** for beta estimation.  
9. **Explicit dynamic-range criteria** before estimating $\beta$ or $k$.  
10. **Same-family evaluator ban** for confirmatory runs.

## Confirmatory hypotheses

### H1 - Decoupled drift
The decoupled arm exhibits positive $d$-vs-$C$ slope, with bootstrap 95% CI lower bound $>0$.

### H2 - Coupling bounds $d$
Matched final $d$ difference:

$$
d_{\mathrm{decoupled,final}} - d_{\mathrm{coupled,final}} > 0
$$

with bootstrap 95% CI lower bound $>0$.

### H3 - Sham control
The sham-extra-compute arm does **not** match the coupled arm. This controls for the objection that coupling works merely because the coupled arm gets an extra model call.

Primary contrast:

$$
d_{\mathrm{sham,final}} - d_{\mathrm{coupled,final}} > 0.
$$

### H4 - Speed invariance
The sign of the decoupled-minus-coupled final-$d$ contrast remains positive under both steady and fast settings.

### H5 - Operational $(\beta,k)$
Only estimate $\beta$ and $k$ when dynamic range is adequate:

- at least 3 positive capability growth steps;
- at least 2x capability range;
- at least 3 corrector observations over at least 2x capability range.

If these criteria fail, the result is **not estimable**, not negative.

## Minimum viable sample

Minimum publishable pilot:

- 3 models;
- 3 tasks;
- 2 speeds;
- 3 conditions;
- 5 seeds;
- 8 rounds.

Total trajectories: $3\times3\times2\times3\times5=270$.

Stronger confirmatory sweep:

- 6 models;
- 3 tasks;
- 2 speeds;
- 3 conditions;
- 10 seeds;
- 8 rounds.

Total trajectories: $6\times3\times2\times3\times10=1080$.

## Mandatory blinding

Misalignment $D$ has two components:

1. Static deterministic gaming detector.
2. Median score from a cross-family blinded evaluator panel.

Confirmatory runs must satisfy:

- evaluator family != engine family;
- laundered code only;
- no condition/round labels to evaluator;
- median of at least three evaluator families where API cost permits.

Same-family model scoring is demonstration-only and must be excluded from confirmatory evidence.

## Primary endpoint

The primary endpoint is the matched final-$d$ contrast:

$$
\Delta d = d_{\mathrm{decoupled,final}} - d_{\mathrm{coupled,final}}.
$$

A positive result requires:

- mean $\Delta d > 0$;
- bootstrap 95% CI lower bound $>0$;
- no negative sign reversal in the fast-speed subgroup.

## Secondary endpoints

1. Decoupled $d$-vs-$C$ slope.  
2. Coupled $d$-vs-$C$ slope.  
3. Sham-minus-coupled final-$d$ contrast.  
4. Estimated $\beta-k$, if estimable.  
5. Task heterogeneity.  
6. Model heterogeneity.

## Negative-result interpretation

- If decoupled remains clean, the task failed to elicit drift or the model is intrinsically corrective.
- If coupled does not beat sham, the correction mechanism is not doing specific safety work.
- If $\beta,k$ are not estimable, the benchmark lacks sufficient dynamic range.
- If fast speed flips the verdict, the speed-invariance hypothesis fails for this empirical setting.

## Reporting language

Use:

> “Mechanism evidence in this task suite.”

Do not use:

> “Proof of alignment,” “hard takeoff solved,” “QEC-equivalent threshold established,” or “universal law.”

## File produced

`realmodel_coscaling_v2.py` is a drop-in stronger harness. It has a deterministic `--selftest` mode and can be used with the existing provider registry.
