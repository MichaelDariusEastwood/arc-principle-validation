"""arc_runner: the code that runs the P5 and P16 experiments.

The instrument kit (arc_instruments) decides verdicts and simulates designs. It never touched a system.
This package is the other half: a model adapter, a frozen hidden capability ladder, the revision loop
with retention control and checkpoints, sealing with a pilot flag, the P5 crossed bank with
seal-then-generate, the P16 driven titration, and a command line.

Two adapters ship. The mock simulates a system whose revision follows the framework's own growth
equation with a coupling the test knows, so the entire pipeline can be run end to end and checked
against a truth before a penny is spent. The real adapter speaks to any OpenAI-compatible endpoint
through environment variables and never prints a key. Swapping one for the other changes nothing
else, which is the point.

Every run states its execution mode, and the mode resolves fail closed: silence is a demonstration
and never the deciding path. A demonstration keeps its verdicts and says in its manifest and in every
printed summary that they are simulated recoveries; a smoke test and a pilot are refused at
proposition level; a confirmatory run refuses to begin unless the domain ladder, the checkpoint store,
the resolved configuration and the spending allowance are all present before the first provider call.
See arc_runner.mode.

Every run also carries its custody: a portable code identity that the same bytes reproduce in any
directory, a seal whose digest is handed to the operator's anchoring service and whose receipt is
recorded, a named party's statement on that seal that the material deciding the predictions was
unseen when they were fixed, and an evidence bundle holding the bank, every read, every replicate and
margin series, the seal, the receipt, the attestation and the provider metadata. See
arc_runner.custody.

What P16 observes is typed at the callback boundary: a source declares its quantity, its units, its
clock, its estimator and its smoothing window, and the declaration decides the observation model, so
the balance elasticity is tested on its level and a log ratio on its slope in the logarithm of the
recursive coordinate. An undeclared source is the unregistered per-round surplus rate and never the
registered elasticity, the sealed slope is not compared with a quantity in another coordinate, and a
deciding run refuses a source that returns the assumed balance instead of the correction service and
the offered burden it is a ratio of. See arc_runner.observation.

The absence of a P16 alarm is read rather than assumed. A silent arm is in one of four states: a
demonstrated sign, a demonstrated practical absence, low information, or an event censored by the
horizon. Only demonstrated positivity in a majority of the above-boundary arms, each beyond a
registered informative horizon, may reach `REFUTED (no reversal)`; silence alone reaches nothing. The
sequential rule the arms are read under, being the growing-window look schedule, the size of its
first look, its threshold, its autocorrelation-robust standard error, the terminal reading, the
control aggregation and the support and refutation rules, is frozen as one object, sealed with the
line and calibrated as a whole by `p16_calibration.py`, which reports every rate as an interval. See
arc_runner.p16_sequential.

P16's support branch is built from the nine components the design promises, each with its own
uncertainty: delivery, realised exposure, the event, the timing against the sealed change-point
window, the location of the fitted zero with both the fit's and the boundary calibration's
uncertainty in the comparison, the slope magnitude inside a registered equivalence, the controls,
discrimination and the fresh-data repetition. A wrapper combines them and refuses a proposition-level
result while any contract-required component is unsupplied, and labels every single run provisional.
Several alarms in several arms are not support. See arc_runner.p16_components.

What a repeated ladder reading resamples is declared by the ladder and decides the arithmetic. A
deterministic score over a fixed item set is exact conditional on that set, so repeating it reports
the same error however many times it is read; a fresh item form drawn from a frozen pool carries the
finite-population sampling error of that draw, so a whole-pool read reports none; only a genuinely
fresh draw earns the square root of the repeat count. Silence fails closed to the deterministic
reading, and a deciding run refuses an undeclared sampling unit and a substring smoke verifier. See
arc_runner.sampling.

P5's coupling is fitted to the readings it took and not to two numbers derived from them. The before
reading enters the available capability and is subtracted inside the increment, so their errors are
correlated by construction and no single ratio of error variances can express that; the fit is a
maximum likelihood fit of the observation model to the paired readings, with the read covariance in
the increment's conditional mean, the process variability in its conditional variance and the exact
retention fraction handled as exact. No cell is removed for having a nonpositive increment, because
that removal selects on the sign of the read noise and does it hardest where the growth is smallest; a
zero increment and a negative one are observations and are fitted as such, and a bank whose increment
is not distinguishable from zero is reported in those words rather than as a coupling recovered from
the cells that happened to grow. See arc_runner.p5_observation.

P5's two routes are two regression directions through one bank and are reported as such. Under the
general process the state direction estimates the elasticity of the increment in the CAPABILITY
state and the retention direction estimates its elasticity in the RETENTION fraction, so their
agreement is a test of whether those are one number and their gap is an estimate of the difference.
The gap is estimated once, by the crossed fit, with its own standard error, and compared with the
margin as an interval equivalence, and the label it carries is CONSISTENT, INCONSISTENT or
UNRESOLVED and never IDENTIFIED. Identification is a separate judgement resting on a separate thing:
an independent capability manipulation supplied by the domain, being a second way of placing the
capability state that does not run through retention, carrying a written exclusion restriction and
its own estimate. With none supplied the judgement is NOT ESTABLISHED and says what is missing; a
nuisance rate that scales with the available capability makes the two directions agree exactly while
neither is measuring the coupling, and a retention elasticity that genuinely differs from the
capability elasticity makes them disagree while the capability elasticity is still estimated. Both
equivalences read the REGISTERED interval convention rather than a second copy of it, so the boundary
is strict there as everywhere else; the manipulation's documentation is re-derived from its own
description at the moment the word is produced rather than read off a key in its record; a second
channel that places with the bank's own loader is refused as the same intervention twice; and the set
of manipulations is inside the sealed specification and inside the deciding gate, being the only
route to that word. See arc_runner.p5_identification.

P5's final comparison carries the whole of its uncertainty and the whole of its assigned panel. The
sealed prediction's interval is propagated jointly from the coupling, every calibration reading and
the starting state, which is one of those readings and moves with them; the permitted coupling
domain is enforced rather than clipped into, and a prediction whose interval reaches it says so and
is not scored against; the observed-minus-predicted interval combines the prediction's uncertainty
with the observed replicate spread, at the larger of the two multipliers, so that adding a source can
only widen it. Every assigned held-out system keeps its weight: a system that reached the ladder
ceiling stays in the denominator as NOT EVALUABLE rather than being dropped from it, the panel is
sealed inside the specification hash before the run, and the aggregate rule is stated over the
assigned panel. The interval level and the boundary convention are in one configurable place, and
the registered choice is the ninety per cent two-sided interval with STRICT clearance: an interval
that lands exactly on the margin has not cleared it. See arc_runner.p5_prediction.

Nothing here submits, anchors or publishes anything: the anchoring service is supplied by the
operator, and the only anchor shipped here is a mock that says so and is refused on the deciding
path. A run that is flagged as a pilot cannot be scored.
"""
from . import (adapters, custody, ladder, observation, sampling, trajectory, manifest, mode, p5,
               p5_identification, p5_observation, p5_prediction, p16, p16_components, p16_sequential)

__all__ = ["adapters", "custody", "ladder", "observation", "sampling", "trajectory", "manifest", "mode",
           "p5", "p5_identification", "p5_observation", "p5_prediction", "p16", "p16_components",
           "p16_sequential"]
