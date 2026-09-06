"""The typed observation at the P16 callback boundary: what is measured, in what units, on what clock.

WHY THIS FILE EXISTS. Finding A1: the reference runner took a callback returning one float per round,
called it a correction margin, regressed it against ordinary round indices and tested the fitted
slope. Nothing said which quantity the float was, so nothing could say what its slope meant. The
registration's Delta is already the logarithmic slope of the service to burden ratio, Delta =
d log(Q/W) / d log R, so a series of Delta values whose slope is negative is not the registered
event: the registered event is Delta itself crossing below zero, and a falling but still positive
Delta satisfies nothing. A series of log(Q/W) values is a different case, whose registered coordinate
is log R rather than the round number. A series of Q/W is a third. The reference runner could not
tell the three apart, because the boundary carried no type.

THE THREE ADMISSIBLE QUANTITIES, AND THE OBSERVATION MODEL EACH ONE IMPLIES.

  balance-elasticity   the series IS Delta. Units: dimensionless, being d log(Q/W) / d log R. The
                       clock orders the readings and no more, because the elasticity has already
                       been taken with respect to log R. The estimator is the window MEAN and the
                       event is the mean lying wholly below zero at the registered threshold. A
                       falling positive Delta does not fire, which is the whole of finding A1.

  log-service-ratio    the series is log(Q/W). Units: natural log units of a dimensionless ratio.
                       The registered clock is log R, the logarithm of the recursive coordinate, and
                       never the unqualified round number. The estimator is the OLS slope of the
                       series on log R, which IS Delta, and the event is that slope lying wholly
                       below zero.

  service-ratio        the series is Q/W. Units: dimensionless. Identical to the previous case after
                       the logarithm is taken at intake, which is why a non-positive reading is
                       excluded and counted rather than silently made to disappear inside a log.

AND THE FOURTH, WHICH IS NOT ONE OF THEM. The reference runner's margin, and the margin the four
balance objects produce in code_domain.BalanceTracker, is (fixes minus events minus backlog growth)
divided by the level: a per-round SURPLUS RATE. Its sign is the level object Q - W < 0, which
arc_instruments.balance separates from the trend object by two worked counterexamples. It is carried
here as `surplus-rate` so that a run may state that it measured one, and it is never treated as the
registered elasticity: it is not comparable with the sealed line's slope, and it cannot be scored at
proposition level.

THE COORDINATE RECONCILIATION, STATED ONCE. The sealed line has slope minus (1 minus chi), being
d Delta / d alpha across arms, which is a derivative of an elasticity with respect to a growth
exponent. The reference mock's trend is 0.01 per unit dose per ROUND, which is a rate of change of a
per-round surplus rate. There is no conversion between them without Q, W and R, so this module
refuses the comparison rather than reporting a ratio of two numbers that measure different things.
Where Q, W and R are present the conversion is not needed, because the quantity is then Delta by
construction and the two slopes are already in one coordinate.

WHAT A SOURCE MUST SUPPLY ON THE DECIDING PATH. Q and W separately, or an approved set of independent
sufficient observations, approved by somebody at some recorded time. A source that returns the assumed
balance itself is refused there: a callback handed back Delta has assumed the answer the arm exists to
measure, which is the difference between an assay and a restatement. A source that declares itself a
simulation is refused there as well, for the reason the deciding path refuses the mock anchor and a
simulated ladder.

AND A DECLARATION IS CHECKED AGAINST THE READINGS, NOT ONLY AGAINST ITSELF. Every field of the
declaration is written by the source's author. `require_assay` reads it before the first paid call,
which is the only moment a run can still be refused for nothing, and it can read nothing else because
no round exists yet. `require_reading_matches_declaration` reads each round as it arrives and refuses
the round that does not carry what was promised, and `reconcile_readings` makes the same comparison
over saved rounds so that a bundle re-scored without the source reaches the same judgement. Without
the second and third, a source could declare Q and W, hand back a bare float, clear every gate, and be
regressed against a round number standing in for log R while flagged as the registered coordinate.

AND THE STANDARD ERROR IS PART OF THE MODEL, NOT OF THE FIT. Finding A3: an arm's rounds are a time
series, so an independence standard error computed from its residuals is too small and the growing
window declares the event on noise. The registered estimator is Newey and West's with Bartlett
weights, a moving-block bootstrap of the residuals is implemented beside it as the stated
alternative, and the rule takes the conservative maximum of the robust and the independence figures
so that a variance repair can never manufacture an alarm. See the variance block below.

FAIL CLOSED ON SILENCE, as everywhere else in this runner. An undeclared callback resolves to the
unregistered surplus rate, never to Delta: silence promotes nothing to the registered coordinate, and
a confirmatory run refuses an undeclared source outright.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from arc_instruments import balance as BAL

# --------------------------------------------------------------------------------------------------
# The vocabulary
# --------------------------------------------------------------------------------------------------

BALANCE_ELASTICITY = "balance-elasticity"
LOG_SERVICE_RATIO = "log-service-ratio"
SERVICE_RATIO = "service-ratio"
SURPLUS_RATE = "surplus-rate"
QUANTITIES = (BALANCE_ELASTICITY, LOG_SERVICE_RATIO, SERVICE_RATIO, SURPLUS_RATE)
REGISTERED_QUANTITIES = (BALANCE_ELASTICITY, LOG_SERVICE_RATIO, SERVICE_RATIO)

ROUND_INDEX = "round-index"
LOG_RECURSIVE_COORDINATE = "log-recursive-coordinate"
CLOCKS = (ROUND_INDEX, LOG_RECURSIVE_COORDINATE)

WINDOW_MEAN = "window-mean-of-the-elasticity"
SLOPE_ON_LOG_R = "ols-slope-of-the-series-on-log-recursive-coordinate"
SLOPE_OF_LOG_VALUE_ON_LOG_R = "ols-slope-of-the-log-series-on-log-recursive-coordinate"
SLOPE_ON_ROUND = "ols-slope-of-the-series-on-the-round-index"

UNITS = {
    BALANCE_ELASTICITY: "dimensionless, being d log(Q/W) / d log R",
    LOG_SERVICE_RATIO: "natural log units of the dimensionless ratio Q/W",
    SERVICE_RATIO: "dimensionless, the ratio of correction service to offered burden",
    SURPLUS_RATE: "items per round per item passing: a per-round surplus rate and not an elasticity",
}
REQUIRED_CLOCK = {
    BALANCE_ELASTICITY: ROUND_INDEX,
    LOG_SERVICE_RATIO: LOG_RECURSIVE_COORDINATE,
    SERVICE_RATIO: LOG_RECURSIVE_COORDINATE,
    SURPLUS_RATE: ROUND_INDEX,
}
ESTIMATORS = {
    BALANCE_ELASTICITY: WINDOW_MEAN,
    LOG_SERVICE_RATIO: SLOPE_ON_LOG_R,
    SERVICE_RATIO: SLOPE_OF_LOG_VALUE_ON_LOG_R,
    SURPLUS_RATE: SLOPE_ON_ROUND,
}
EVENTS = {
    BALANCE_ELASTICITY: "Delta crosses below zero",
    LOG_SERVICE_RATIO: "Delta, the slope of log(Q/W) in log R, crosses below zero",
    SERVICE_RATIO: "Delta, the slope of log(Q/W) in log R, crosses below zero",
    SURPLUS_RATE: "the per-round surplus rate falls, which is a level object and not the registered "
                  "balance event",
}

MIN_POINTS = 4          # as in the reference rule: fewer than four points is not a fitted window

# --------------------------------------------------------------------------------------------------
# The variance rule, which is part of the observation model and not a detail of the fit (finding A3)
# --------------------------------------------------------------------------------------------------
#
# WHY A SERIES NEEDS ONE. The rounds of an arm are a time series and its residuals are not
# independent: a system that is behind at round 40 is behind at round 41. An independence standard
# error computed from such residuals is too small by roughly the square root of the sum of the
# autocorrelations, so the growing-window rule that declares the event at delta + z se < 0 declares
# it on noise. Finding A3 requires the sequential inference to be frozen for the actual schedule
# INCLUDING its temporal correlation, and requires the runner to state which robust estimator it uses.
#
# WHICH ONE IS USED, STATED. The registered estimator is Newey and West's autocorrelation-consistent
# standard error with Bartlett weights, at Andrews' automatic bandwidth with an autoregressive
# plug-in of order one. A moving-block bootstrap of the residuals is implemented beside it as the
# stated alternative, so that a run may say it used one; it is never chosen silently.
#
# WHY THE BANDWIDTH IS AUTOMATIC AND NOT THE RULE OF THUMB. The fixed rule L = floor(4 (n/100)^(2/9))
# gives L = 3 across this runner's window lengths, and three lags of a Bartlett kernel recover about
# a fifth of the long-run variance of a series whose round-to-round correlation is 0.9. Measured on
# this runner: an autocorrelated flat null at 0.9 produced the same alarms under that rule as under
# the independence assumption, which is a correction that does not correct. Andrews' bandwidth is a
# FIXED RULE whose bandwidth varies with the persistence the residuals show, which is a different
# thing from choosing a lag per series to taste; at a residual correlation of zero it returns a
# bandwidth of zero, so the estimate collapses to the heteroskedasticity-only form and the rule costs
# no power in the independent worlds it is not needed in.
#
# AND WHAT THE RULE ACTUALLY TAKES. The default is the CONSERVATIVE MAXIMUM of the independence
# standard error and the robust one. The reason is that a repair to a variance estimate must not be
# able to manufacture an alarm: taking the maximum can only widen an interval, so this rule declares
# the event on no series that the independence rule would not also declare it on, while removing the
# alarms that only the independence assumption supported. The cost is power against a genuinely
# negatively autocorrelated series, where the robust estimate is the smaller and the independence one
# is then used. That trade is named here rather than buried: if the author registers the robust
# estimate alone, the constant below is the one line that changes.
INDEPENDENCE = "independence"
NEWEY_WEST = "newey-west-bartlett-autocorrelation-robust"
BLOCK_BOOTSTRAP = "moving-block-bootstrap-of-the-residuals"
CONSERVATIVE_MAX = "maximum-of-independence-and-newey-west"
VARIANCE_ESTIMATORS = (INDEPENDENCE, NEWEY_WEST, BLOCK_BOOTSTRAP, CONSERVATIVE_MAX)
DEFAULT_VARIANCE = CONSERVATIVE_MAX
BLOCK_BOOTSTRAP_DRAWS = 400
BLOCK_BOOTSTRAP_SEED = 20260905   # fixed so that a bundle re-scores to the same standard error


def ar1_correlation(resid: np.ndarray) -> float:
    """The lag-one autocorrelation of the residuals, clipped away from the unit circle.

    This is the plug-in the bandwidth is computed from. It is the residuals' own correlation and not
    the series', because the dependence the standard error must survive is the dependence the fitted
    model leaves behind.
    """
    e = np.asarray(resid, float)
    n = len(e)
    if n < 3:
        return 0.0
    d0 = float(np.sum(e * e))
    if d0 <= 0:
        return 0.0
    r = float(np.sum(e[1:] * e[:-1]) / d0)
    return float(min(max(r, -0.97), 0.97))


def bartlett_lag(resid: np.ndarray, n: Optional[int] = None) -> int:
    """Andrews' automatic bandwidth for the Bartlett kernel with an order-one plug-in.

    S = 1.1447 (alpha n) ^ (1/3) with alpha = 4 rho^2 / ((1 - rho)^2 (1 + rho)^2). At rho = 0 the
    bandwidth is zero and the estimate is the heteroskedasticity-only one; at rho = 0.9 over fifty
    rounds it is about eighteen, which is the length of dependence such a series actually carries.
    The bandwidth is truncated at n - 2 so that a short window keeps a fitted segment.
    """
    e = np.asarray(resid, float)
    n = int(n if n is not None else len(e))
    if n < 3:
        return 0
    rho = ar1_correlation(e)
    denom = ((1.0 - rho) ** 2) * ((1.0 + rho) ** 2)
    if denom <= 0:
        return int(n - 2)
    alpha = 4.0 * (rho ** 2) / denom
    if alpha <= 0:
        return 0
    return int(max(0, min(int(1.1447 * (alpha * n) ** (1.0 / 3.0)), n - 2)))


def _bartlett(u: np.ndarray, lag: int) -> float:
    """The long-run variance of the score series u, with Bartlett weights w(l) = 1 - l / (L + 1)."""
    s = float(np.sum(u * u))
    for l in range(1, int(lag) + 1):
        w = 1.0 - l / (lag + 1.0)
        s += 2.0 * w * float(np.sum(u[l:] * u[:-l]))
    return max(s, 0.0)


class ObservationRefusal(ValueError):
    """The observation cannot be interpreted, so no event and no line may be read from it."""


# --------------------------------------------------------------------------------------------------
# The declaration
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ObservationSpec:
    """What the callback returns, said in full: quantity, units, clock, estimator, smoothing window.

    `supplies_q_and_w` is the declaration a deciding run checks. `sufficient_observations` is the
    escape hatch the registration allows, being an approved set of independent observations from
    which Q and W follow; it is recorded by name and never inferred, because a set nobody approved is
    not an approved set.

    AND A DECLARATION IS NOT A MEASUREMENT. Every field here is written by the source's author, so
    each one is a promise about what will arrive and none of them is evidence that it did. Two
    separate checks enforce the promise, and they are separate because they can be made at different
    moments: `require_assay` reads the declaration before the first paid call, which is the only
    moment at which a run can still be refused for nothing; `require_reading_matches_declaration`
    reads each round as it arrives and refuses the round that does not carry what was declared. A
    gate that only ever read the first of the two would accept a source claiming Q and W, handing
    back a bare float, and being regressed against a round number standing in for log R, which is
    the substitution this module exists to forbid.

    `observations_approved_by` and `observations_approved_utc` are what makes an approved set
    approved: a person and a time recorded against the exact set of observations, in the same shape
    as `mode.ConfirmatoryInputs.config_resolution`. They are two scalars rather than a mapping so
    that the specification stays hashable and compares by value across a bundle round trip.

    `simulated` is the source's own statement that it models a world rather than reading one. It is
    refused on the deciding path for the reason `mode._anchor_refusals` refuses the mock anchor and
    `mode._ladder_refusals` refuses a simulated ladder: a simulation attests nothing about the
    world, and a marker the deciding gate never reads is not a marker.
    """

    quantity: str
    clock: str
    estimator: str
    units: str
    smoothing_window: int = 1
    supplies_q_and_w: bool = False
    sufficient_observations: Tuple[str, ...] = ()
    observations_approved_by: str = ""
    observations_approved_utc: str = ""
    simulated: bool = False
    chi_hat: Optional[float] = None
    curvature_correction: bool = False
    source: str = "unspecified"
    note: str = ""

    def __post_init__(self) -> None:
        if self.quantity not in QUANTITIES:
            raise ObservationRefusal("unknown observed quantity %r; one of %s" % (self.quantity, ", ".join(QUANTITIES)))
        if self.clock != REQUIRED_CLOCK[self.quantity]:
            raise ObservationRefusal(
                "quantity %r is read on the %s clock and not on %r: the registered coordinate belongs "
                "to the quantity and cannot be chosen per run" % (self.quantity, REQUIRED_CLOCK[self.quantity], self.clock))
        if self.estimator != ESTIMATORS[self.quantity]:
            raise ObservationRefusal(
                "quantity %r is estimated by %s and not by %r" % (self.quantity, ESTIMATORS[self.quantity], self.estimator))
        if int(self.smoothing_window) != 1:
            # CONSERVATIVE READING, NAMED AS OPEN. The contract requires the smoothing window to be
            # stated; it does not state one, and no standard error for a smoothed series is
            # registered. A moving average correlates the residuals, so the interval that decides the
            # event would be too narrow by a factor nobody has approved. Refusing is the reading that
            # cannot decide anything by accident; if the author registers a window, its variance rule
            # belongs here beside it.
            raise ObservationRefusal(
                "a smoothing window of %r was declared. No standard error for a smoothed series is "
                "registered, and a moving average narrows the interval that decides the event by a "
                "factor nobody has approved. Declare 1, or register the window with its variance rule."
                % self.smoothing_window)

    @property
    def registered(self) -> bool:
        """True when this quantity is one of the three the registration admits."""
        return self.quantity in REGISTERED_QUANTITIES

    @property
    def approved_observations(self) -> bool:
        """True when a named set of sufficient observations carries a recorded approval.

        A set of names with nobody's name and no time against it is a list somebody typed. The
        docstring above this class has said since finding A1 that an approved set is approved by
        somebody; until this property existed nothing read that sentence, so the escape hatch cleared
        the deciding path on a tuple of strings alone.
        """
        return bool(self.sufficient_observations) and bool(self.observations_approved_by) \
            and bool(self.observations_approved_utc)

    @property
    def returns_the_assumed_balance(self) -> bool:
        """True when the callback hands back Delta rather than the observations it is computed from."""
        return self.quantity == BALANCE_ELASTICITY

    @property
    def event(self) -> str:
        return EVENTS[self.quantity]

    def as_record(self) -> Dict[str, Any]:
        """What the seal, the manifest and the bundle carry. Plain JSON, so a later reader needs no code."""
        return {"quantity": self.quantity, "units": self.units, "clock": self.clock,
                "estimator": self.estimator, "smoothing_window": int(self.smoothing_window),
                "supplies_q_and_w": bool(self.supplies_q_and_w),
                "sufficient_observations": list(self.sufficient_observations),
                "observations_approved_by": self.observations_approved_by,
                "observations_approved_utc": self.observations_approved_utc,
                "simulated": bool(self.simulated),
                "chi_hat": self.chi_hat, "curvature_correction": bool(self.curvature_correction),
                "registered_coordinate": self.registered, "event": self.event,
                "source": self.source, "note": self.note}

    @classmethod
    def from_record(cls, record: Optional[Dict[str, Any]]) -> "ObservationSpec":
        """Rebuild a spec from a bundle. An absent record is the undeclared one, fail closed."""
        if not record:
            return UNDECLARED
        return cls(quantity=record["quantity"], clock=record["clock"], estimator=record["estimator"],
                   units=record.get("units", UNITS.get(record["quantity"], "")),
                   smoothing_window=int(record.get("smoothing_window", 1)),
                   supplies_q_and_w=bool(record.get("supplies_q_and_w", False)),
                   sufficient_observations=tuple(record.get("sufficient_observations") or ()),
                   observations_approved_by=record.get("observations_approved_by", "") or "",
                   observations_approved_utc=record.get("observations_approved_utc", "") or "",
                   simulated=bool(record.get("simulated", False)),
                   chi_hat=record.get("chi_hat"),
                   curvature_correction=bool(record.get("curvature_correction", False)),
                   source=record.get("source", "unspecified"), note=record.get("note", ""))


def _spec(quantity: str, **kw) -> ObservationSpec:
    return ObservationSpec(quantity=quantity, clock=REQUIRED_CLOCK[quantity],
                           estimator=ESTIMATORS[quantity], units=UNITS[quantity], **kw)


def balance_elasticity_observation(**kw) -> ObservationSpec:
    """The series is Delta itself. Admissible, and refused on the deciding path: see `require_assay`."""
    return _spec(BALANCE_ELASTICITY, **kw)


def log_service_ratio_observation(**kw) -> ObservationSpec:
    """The series is log(Q/W), read against log R."""
    return _spec(LOG_SERVICE_RATIO, **kw)


def service_ratio_observation(**kw) -> ObservationSpec:
    """The series is Q/W, read against log R after the logarithm is taken at intake."""
    return _spec(SERVICE_RATIO, **kw)


def surplus_rate_observation(**kw) -> ObservationSpec:
    """The per-round surplus rate the four balance objects produce. Not the registered elasticity."""
    return _spec(SURPLUS_RATE, **kw)


UNDECLARED = surplus_rate_observation(
    source="undeclared",
    note="This callback declared no quantity, so it resolves to the unregistered per-round surplus "
         "rate: silence never promotes a series to the registered coordinate. Declare the quantity "
         "with arc_runner.observation.declare before a run that must be read.")


def declare(fn: Callable, spec: ObservationSpec) -> Callable:
    """Attach a spec to a margin source. The source is returned so that a call site reads as one line."""
    setattr(fn, "observation", spec)
    return fn


def spec_of(source: Any) -> ObservationSpec:
    """The source's declaration, or the undeclared one. Never a guess from the values it returns."""
    spec = getattr(source, "observation", None)
    return spec if isinstance(spec, ObservationSpec) else UNDECLARED


def require_assay(spec: ObservationSpec) -> None:
    """The deciding path's requirement: Q and W separately, or approved sufficient observations.

    Six refusals, and each is a different failure. A source that declared nothing has said nothing
    about what it measured. A source that declares itself a simulation has modelled a world and not
    read one. A source that hands back Delta has assumed the balance the arm exists to measure. A
    source on the per-round surplus rate has measured a level object, which may be reported and may
    not decide the registered event. A source naming sufficient observations that nobody approved has
    not reached the escape hatch the registration allows. A source with neither Q and W nor such an
    approved set has a balance nothing can be checked against.

    THIS READS THE DECLARATION AND NOTHING ELSE, WHICH IS ALL IT CAN READ. It runs before the first
    paid call, so no reading exists yet to compare the declaration with. That is why it is one half
    of the requirement and not the whole of it: `require_reading_matches_declaration` is the other
    half and runs on every round `read` normalises. Removing either half restores the defect, because
    a promise checked against nothing is a promise.
    """
    if spec is UNDECLARED or spec.source == "undeclared":
        raise ObservationRefusal(
            "the margin source declares no observed quantity. A float per round has no settled "
            "meaning, so no event can be tested in the correct coordinate; declare the quantity, its "
            "units, its clock and its estimator with arc_runner.observation.declare")
    # CONSERVATIVE READING, NAMED AS OPEN. The contract requires the real assay to supply Q and W or
    # the approved sufficient observations, and does not say in those words that the source must not
    # be a simulation. The reading taken here is the one the rest of the deciding path already takes:
    # `mode._anchor_refusals` refuses the mock anchor by identity and `mode._ladder_refusals` refuses
    # a ladder that measures a simulated latent capability, and a simulated margin is the same kind of
    # object. It is also the reading that cannot decide anything by accident. If the author settles
    # that a simulated source may decide a proposition, this refusal is the one line that changes.
    if spec.simulated:
        raise ObservationRefusal(
            "this source (%r) declares itself a simulation, so its Q and W are drawn from a model of "
            "the world and are not a reading of one. A simulated margin may demonstrate the "
            "coordinate, the estimator and the whole apparatus, and it may not decide a proposition; "
            "the deciding path refuses it for the reason it refuses the mock anchor and a simulated "
            "ladder" % spec.source)
    if spec.returns_the_assumed_balance:
        raise ObservationRefusal(
            "this source returns the balance elasticity itself. The real assay must supply the "
            "correction service Q and the offered burden W separately, or the approved independent "
            "sufficient observations; a callback that returns the assumed balance restates the "
            "answer the arm exists to measure")
    if not spec.registered:
        raise ObservationRefusal(
            "the observed quantity is %r, being %s. That is a level object and not the registered "
            "balance elasticity, so an event read from it is not the registered event and its trend "
            "is not the sealed line's slope; it may be measured and reported, and it may not decide "
            "a proposition" % (spec.quantity, UNITS[spec.quantity]))
    # CONSERVATIVE READING, NAMED AS OPEN. The contract says "the approved independent sufficient
    # observations" and does not say what makes a set approved. The reading taken here is the one the
    # runner already uses for a resolved configuration in `mode._config_refusals`: a person and a time
    # recorded against the exact set, which cannot be satisfied by accident. The set must then also
    # arrive, which `require_reading_matches_declaration` enforces round by round. If the author
    # settles a stricter rule, a countersigned record of the observations for instance, it belongs
    # here beside this one.
    if spec.sufficient_observations and not spec.approved_observations:
        raise ObservationRefusal(
            "this source names %d sufficient observation(s) and records no approval against them. An "
            "approved set is approved by somebody at some time, so name the approver and the time in "
            "observations_approved_by and observations_approved_utc; a set nobody approved is not an "
            "approved set and cannot stand in for Q and W"
            % len(spec.sufficient_observations))
    if not (spec.supplies_q_and_w or spec.approved_observations):
        raise ObservationRefusal(
            "this source supplies neither Q and W nor any approved sufficient observations, so the "
            "balance it reports cannot be checked against a measurement (quantity %r)" % spec.quantity)


# --------------------------------------------------------------------------------------------------
# The other half of the requirement: what ARRIVED, against what was DECLARED
# --------------------------------------------------------------------------------------------------

def _carries(rd: "Reading", name: str) -> bool:
    """Whether a reading carries an observation of that name, as a field or in what it was computed
    from. Both are looked at because a domain names its own observations and the runner does not know
    them: `extra` is where a source puts what only it can name."""
    if getattr(rd, name, None) is not None:
        return True
    return isinstance(rd.extra, dict) and rd.extra.get(name) is not None


def require_reading_matches_declaration(rd: "Reading", spec: ObservationSpec) -> None:
    """Refuse the round that does not carry what the declaration promised, at the round it arrives.

    WHY THIS EXISTS AND WHY IT IS NOT INSIDE `require_assay`. The declaration is written by the
    source's author and, until this function, nothing anywhere compared it with the readings: a
    source could declare `supplies_q_and_w=True`, return a bare float per round, clear the deciding
    path's gate and be estimated as the registered quantity. The reading it produced then carried no
    recursive coordinate either, and the estimator's fallback put the round number in place of log R,
    so a per-round series was regressed on log(round + 1) and reported as the balance elasticity. The
    fallback is gone (see `_points_full`) and this is the check that stops the same reading arriving
    at all. `require_assay` cannot make it, because it runs before any round exists.

    THREE REFUSALS, ONE PER PROMISE. Q and W when they were declared. The recursive coordinate
    whenever the declared clock is log R, because R belongs to the artefact's depth and no other
    number substitutes for it. And every named sufficient observation, because a set that was
    approved and then did not arrive is not the set that was approved.
    """
    if spec.supplies_q_and_w and (rd.Q is None or rd.W is None):
        raise ObservationRefusal(
            "round %d declared that it supplies the correction service Q and the offered burden W "
            "separately and supplied %s. A declaration is not a measurement: the ratio under test "
            "cannot be re-checked against readings that were never made"
            % (rd.round_index,
               "neither" if (rd.Q is None and rd.W is None) else ("no W" if rd.W is None else "no Q")))
    if spec.clock == LOG_RECURSIVE_COORDINATE and rd.R is None:
        raise ObservationRefusal(
            "round %d is read on the %s clock and carried no recursive coordinate R. The round number "
            "is not R: an arm that spends more revision passes per round travels further in R than "
            "the rounds it took, so substituting the round index reads a different quantity in a "
            "different coordinate and calls it the registered one"
            % (rd.round_index, LOG_RECURSIVE_COORDINATE))
    if spec.sufficient_observations:
        missing = [n for n in spec.sufficient_observations if not _carries(rd, n)]
        if missing:
            raise ObservationRefusal(
                "round %d declared the approved sufficient observation(s) %s and carried none of %s. "
                "An approved set stands in for Q and W only where it actually arrives"
                % (rd.round_index, ", ".join(spec.sufficient_observations), ", ".join(missing)))


def reconcile_readings(readings: Sequence["Reading"], spec: ObservationSpec) -> Dict[str, Any]:
    """The same comparison made over a whole arm, for the moment `read` is not in the picture.

    A bundle re-scored by `custody.recompute_verdicts` reads saved rounds rather than calling a
    source, so the per-round refusal above never fires there; and a bundle written before that
    refusal existed may carry exactly the rounds it would have refused. This reports, from the saved
    readings alone, whether the declaration and the readings agree, so that a re-scoring reaches the
    same judgement as the run and an old bundle is read for what it holds rather than for what its
    declaration claims.

    A summary that saved no readings is NOT reconciled and is not refused either: it is reported as
    unchecked, because absent evidence is not contrary evidence.
    """
    rows = list(readings)
    if not rows:
        return {"state": "not-checked", "n_readings": 0, "n_with_q_and_w": 0, "n_with_r": 0,
                "reason": "no readings were supplied with these arms, so the declaration could not be "
                          "reconciled against what arrived"}
    n_qw = sum(1 for rd in rows if rd.Q is not None and rd.W is not None)
    n_r = sum(1 for rd in rows if rd.R is not None)
    failures: List[str] = []
    if spec.supplies_q_and_w and n_qw < len(rows):
        failures.append("%d of %d rounds carry no Q and W, and the declaration says every round does"
                        % (len(rows) - n_qw, len(rows)))
    if spec.clock == LOG_RECURSIVE_COORDINATE and n_r < len(rows):
        failures.append("%d of %d rounds carry no recursive coordinate R, and the declared clock is "
                        "log R" % (len(rows) - n_r, len(rows)))
    if spec.sufficient_observations:
        absent = sorted({n for rd in rows for n in spec.sufficient_observations if not _carries(rd, n)})
        if absent:
            failures.append("the approved sufficient observation(s) %s did not arrive" % ", ".join(absent))
    return {"state": ("refused" if failures else "reconciled"), "n_readings": len(rows),
            "n_with_q_and_w": n_qw, "n_with_r": n_r, "reason": "; ".join(failures)}


# --------------------------------------------------------------------------------------------------
# One reading
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Reading:
    """One round's observation, with the parts it was computed from kept beside it.

    Q and W are kept separately and never only as their ratio, because the ratio is the thing under
    test and a bundle that saved only the ratio cannot be re-checked. R is the recursive coordinate,
    being the depth of recursion the artefact has reached, which is the clock the registered balance
    is read against. U is the capability, present only when the run measures it, and is what the
    curvature correction needs.
    """

    round_index: int
    value: Optional[float] = None
    Q: Optional[float] = None
    W: Optional[float] = None
    R: Optional[float] = None
    U: Optional[float] = None
    level: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None      # whatever the source computed the reading from

    def as_dict(self) -> Dict[str, Any]:
        return {"round": int(self.round_index), "value": _f(self.value), "Q": _f(self.Q), "W": _f(self.W),
                "R": _f(self.R), "U": _f(self.U), "level": _f(self.level), "extra": self.extra}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Reading":
        return cls(round_index=int(d["round"]), value=d.get("value"), Q=d.get("Q"), W=d.get("W"),
                   R=d.get("R"), U=d.get("U"), level=d.get("level"), extra=d.get("extra"))


def _f(v: Any) -> Optional[float]:
    return None if v is None else float(v)


def read(source: Callable, arm: str, alpha_arm: float, r: int, rng, spec: ObservationSpec) -> Reading:
    """Call the source for one round and normalise what comes back into a typed reading.

    A bare float is taken as the declared quantity's value. A mapping may give Q and W instead of the
    value, in which case the value is DERIVED here rather than trusted: the derivation is the
    observation model and it belongs in one place.

    AND THE ROUND IS CHECKED AGAINST THE DECLARATION BEFORE IT IS RETURNED. This is the one place
    every reading passes through, so it is the one place at which a source that promised Q, W or the
    recursive coordinate can be caught not supplying them. It fires on the first round of the first
    arm, which is the cheapest moment a run can fail.
    """
    out = source(arm, alpha_arm, r, rng)
    if isinstance(out, Reading):
        rd = out if out.round_index == r else replace(out, round_index=r)
    elif isinstance(out, dict):
        d = dict(out)
        d.setdefault("round", r)
        rd = Reading.from_dict(d)
    else:
        rd = Reading(round_index=r, value=float(out))
    if rd.value is None:
        rd = replace(rd, value=_derive(rd, spec))
    require_reading_matches_declaration(rd, spec)
    return rd


def readings_from_values(values: Sequence[float], first_round: int = 0) -> List[Reading]:
    """A bare series of numbers as typed readings, one per round, with nothing claimed about them.

    The legacy entry points take a list of floats. They keep working, and what they keep working AS
    is the undeclared quantity: a value with no Q, no W and no recursive coordinate beside it can
    only be read on the round clock, which is the reading `UNDECLARED` names.
    """
    return [Reading(round_index=first_round + i, value=float(v)) for i, v in enumerate(values)]


def _derive(rd: Reading, spec: ObservationSpec) -> Optional[float]:
    """The declared quantity from Q and W. Delta is never derived from one round: it is a slope."""
    if rd.Q is None or rd.W is None:
        raise ObservationRefusal(
            "round %d supplied neither a value nor both of Q and W, so nothing was observed" % rd.round_index)
    if spec.quantity == SERVICE_RATIO:
        return float(rd.Q) / float(rd.W) if rd.W != 0 else float("nan")
    if spec.quantity == LOG_SERVICE_RATIO:
        return math.log(rd.Q / rd.W) if (rd.Q > 0 and rd.W > 0) else float("nan")
    if spec.quantity == SURPLUS_RATE:
        lvl = float(rd.level) if rd.level else 1.0
        return (float(rd.Q) - float(rd.W)) / lvl
    raise ObservationRefusal(
        "the balance elasticity is a slope over a window and cannot be derived from one round's Q and W")


# --------------------------------------------------------------------------------------------------
# The estimator
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class DeltaEstimate:
    """The window's estimate of the quantity the event is tested on, with what it cost to get it.

    `delta` is the balance elasticity for every registered quantity, so that one number means one
    thing across the three admissible cases and the sealed line is comparable with all of them. For
    the unregistered surplus rate it is the per-round trend, `registered` is False, and no comparison
    with the sealed line is offered.
    """

    delta: float
    se: float
    n_used: int
    n_excluded: int
    registered: bool
    coordinate: str
    reason: str = ""
    model_check: Optional[Dict[str, Any]] = None
    # THE TWO STANDARD ERRORS, KEPT APART (finding A3). `se` is the one the sequential rule uses;
    # these two are what it was computed from, so that a reader can see how much of the interval the
    # temporal correlation is responsible for and an analyst can re-derive either.
    se_independent: float = float("nan")
    se_robust: Optional[float] = None
    variance_estimator: str = DEFAULT_VARIANCE

    def as_dict(self) -> Dict[str, Any]:
        return {"delta": float(self.delta), "se": float(self.se), "n_used": int(self.n_used),
                "n_excluded": int(self.n_excluded), "registered_coordinate": bool(self.registered),
                "coordinate": self.coordinate, "reason": self.reason, "model_check": self.model_check,
                "se_independent": float(self.se_independent), "se_robust": self.se_robust,
                "variance_estimator": self.variance_estimator}


def _unusable_value_reason(rd: Reading) -> str:
    """Why this round has no usable value, named rather than left as a bare count.

    A round excluded without a reason is a round that disappeared: the count says how many, and only
    the reason says whether the arm met a regime the ratio does not describe or whether nothing was
    observed at all. A zero offered burden is the first of those, and it is the one the acceptance
    case is about, so it is named first.
    """
    if rd.W is not None and float(rd.W) == 0.0:
        return "no offered burden (W = 0), a regime in which the service ratio does not exist"
    if rd.value is None:
        return "no value was observed and none could be derived"
    return "a non-finite reading"


def _points(readings: Sequence[Reading], spec: ObservationSpec) -> Tuple[np.ndarray, np.ndarray, int, str]:
    """(x, y, excluded, reason) in the declared coordinate, with every unusable round counted.

    A thin wrapper over `_points_full`, which keeps each usable round's index beside its coordinates.
    The estimator does not need the round index; the change-point estimator does, because the timing
    the registration seals is a round and not a position in a filtered array.

    A round is unusable when its value is not finite, when the log clock's recursive coordinate is
    not positive, or when a ratio quantity is not positive so its logarithm does not exist. They are
    counted and reported rather than dropped in silence: a zero-burden round is a regime the ratio
    does not describe, which arc_instruments.balance.service_ratio refuses outright, and an arm that
    is mostly such rounds has not been measured.
    """
    rounds, x, y, excluded, reason = _points_full(readings, spec)
    return x, y, excluded, reason


def _points_full(readings: Sequence[Reading], spec: ObservationSpec):
    """(rounds, x, y, excluded, reason): the same points, with the round each one came from."""
    rs: List[int] = []
    xs: List[float] = []
    ys: List[float] = []
    excluded = 0
    reasons: List[str] = []
    for rd in readings:
        v = rd.value
        if v is None or not np.isfinite(v):
            excluded += 1
            reasons.append(_unusable_value_reason(rd))
            continue
        if spec.clock == LOG_RECURSIVE_COORDINATE:
            # THE ROUND NUMBER IS NOT R, AND SILENCE ABOUT R DOES NOT PROMOTE THE ROUND NUMBER INTO
            # IT. This line read `rd.R if rd.R is not None else float(rd.round_index + 1)`, which
            # fired only in the case this module forbids: a series declared on the registered log R
            # clock and handed back without a recursive coordinate was regressed against
            # log(round + 1) and reported as the balance elasticity, in the coordinate flagged as the
            # registered one. A missing R makes the round unusable, and it is excluded and counted
            # like every other unusable round rather than repaired with a number that means something
            # else. Every source in this runner that declares the log R clock supplies R, so nothing
            # legitimate depended on the fallback; the bare legacy series reaches here only as the
            # undeclared quantity, whose clock is the round index.
            if rd.R is None:
                excluded += 1
                reasons.append("no recursive coordinate R was supplied, and the round number is not R")
                continue
            R = float(rd.R)
            if R <= 0:
                excluded += 1
                reasons.append("non-positive recursive coordinate")
                continue
            x = math.log(R)
        else:
            x = float(rd.round_index)
        if spec.quantity == SERVICE_RATIO:
            if v <= 0:
                excluded += 1
                reasons.append("non-positive service ratio")
                continue
            y = math.log(v)
        else:
            y = float(v)
        rs.append(int(rd.round_index))
        xs.append(x)
        ys.append(y)
    return (np.asarray(rs, int), np.asarray(xs, float), np.asarray(ys, float), excluded,
            "; ".join(sorted(set(reasons))))


def _ols_full(x: np.ndarray, y: np.ndarray, variance: str = DEFAULT_VARIANCE
              ) -> Tuple[float, float, float, Optional[float]]:
    """Slope, the registered standard error, the independence one and the robust one.

    A zero-residual window is a perfectly measured one: its interval is a point, and the point is
    either below zero or it is not. The reference rule required a strictly positive standard error
    before it would declare, so noiseless data never reached its alarm branch and a constant negative
    Delta could not fire the event it defines. That is finding A3's smaller half and it holds for
    every estimator here: a robust variance of a zero-residual window is zero as well.
    """
    n = len(x)
    A = np.vstack([x, np.ones_like(x)]).T
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = np.asarray(y - A @ coef, float)
    xc = np.asarray(x, float) - float(np.mean(x))
    sxx = float(np.sum(xc ** 2))
    sigma2 = float(np.sum(resid ** 2)) / (n - 2) if n > 2 else 0.0
    se_ind = float(np.sqrt(sigma2 / sxx)) if sigma2 > 0 and sxx > 0 else 0.0
    se_rob: Optional[float] = None
    if sxx > 0 and n > 2 and float(np.sum(resid ** 2)) > 0:
        if variance in (NEWEY_WEST, CONSERVATIVE_MAX):
            # the bandwidth is read from the RESIDUALS' persistence and the meat is the score series
            # x_centred * residual; the small-sample factor n / (n - 2) is the same degrees-of-freedom
            # correction the independence estimate already carries
            lag = bartlett_lag(resid, n)
            s = _bartlett(xc * resid, lag) * (n / float(n - 2))
            se_rob = float(np.sqrt(s / (sxx ** 2)))
        elif variance == BLOCK_BOOTSTRAP:
            se_rob = _block_bootstrap_slope_se(A, coef, resid, n)
    elif variance != INDEPENDENCE:
        se_rob = 0.0                       # a perfectly fitted window is perfectly fitted robustly too
    return float(coef[0]), _combine_se(se_ind, se_rob, variance), se_ind, se_rob


def _block_bootstrap_slope_se(A: np.ndarray, coef: np.ndarray, resid: np.ndarray, n: int) -> float:
    """The moving-block bootstrap of the residuals: blocks of consecutive residuals are resampled and
    the line refitted, so the dependence inside a block survives the resampling.

    The block length is the Bartlett bandwidth plus one, which keeps the two estimators reading the
    same dependence length, and the stream is seeded from a fixed constant so that the same saved
    series re-scores to the same standard error in another analyst's hands.
    """
    b = max(1, min(bartlett_lag(resid, n) + 1, max(n // 2, 1)))
    rng = np.random.default_rng(BLOCK_BOOTSTRAP_SEED)
    fitted = A @ coef
    starts_max = n - b
    slopes = []
    for _ in range(BLOCK_BOOTSTRAP_DRAWS):
        picks = rng.integers(0, starts_max + 1, size=int(np.ceil(n / b)))
        e = np.concatenate([resid[s:s + b] for s in picks])[:n]
        c = np.linalg.lstsq(A, fitted + e, rcond=None)[0]
        slopes.append(float(c[0]))
    return float(np.std(np.asarray(slopes, float), ddof=1))


def _combine_se(se_ind: float, se_rob: Optional[float], variance: str) -> float:
    if variance == INDEPENDENCE or se_rob is None or not np.isfinite(se_rob):
        return float(se_ind)
    if variance == CONSERVATIVE_MAX:
        return float(max(se_ind, se_rob))
    return float(se_rob)


def _ols(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Slope and its independence standard error, for the callers that measure something other than
    the balance (the realised growth exponent, whose own uncertainty is not the sequential rule's)."""
    slope, _, se_ind, _ = _ols_full(x, y, INDEPENDENCE)
    return slope, se_ind


def _mean(y: np.ndarray, variance: str = DEFAULT_VARIANCE) -> Tuple[float, float, float, Optional[float]]:
    """The level and the standard error of the level. This is the estimator for a Delta series, and
    the reason is the whole of finding A1: the event is where Delta IS, not where it is going.

    The robust form is the long-run variance of the same series divided by n, which is the mean's
    Newey and West standard error: a level read from a series whose deviations persist is known less
    well than the independence formula says.
    """
    n = len(y)
    m = float(np.mean(y))
    se_ind = float(np.std(y, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    se_rob: Optional[float] = None
    dev = np.asarray(y, float) - m
    if n > 2 and float(np.sum(dev ** 2)) > 0:
        if variance in (NEWEY_WEST, CONSERVATIVE_MAX):
            lag = bartlett_lag(dev, n)
            s = _bartlett(dev, lag) * (n / float(n - 1))
            se_rob = float(np.sqrt(s / (n ** 2)))
        elif variance == BLOCK_BOOTSTRAP:
            b = max(1, min(bartlett_lag(dev, n) + 1, max(n // 2, 1)))
            rng = np.random.default_rng(BLOCK_BOOTSTRAP_SEED)
            means = []
            for _ in range(BLOCK_BOOTSTRAP_DRAWS):
                picks = rng.integers(0, n - b + 1, size=int(np.ceil(n / b)))
                means.append(float(np.mean(np.concatenate([dev[s:s + b] for s in picks])[:n])))
            se_rob = float(np.std(np.asarray(means, float), ddof=1))
    elif variance != INDEPENDENCE:
        se_rob = 0.0
    return m, _combine_se(se_ind, se_rob, variance), se_ind, se_rob


def _curvature_check(readings: Sequence[Reading], spec: ObservationSpec) -> Optional[Dict[str, Any]]:
    """The constant-exponent reading against the curvature-corrected one, where U and R were measured.

    On a shifted trajectory (dU/dR = a U^beta_L) the local elasticity alpha_R changes with depth, and
    inserting a finite-window alpha_R into the constant-exponent formula Delta = 1 - alpha (1 - chi)
    reports a surplus that is not there: arc_instruments.balance.shifted_trajectory_balance gives the
    truth, alpha_R (chi - beta_L), and local_balance recovers it by subtracting d log alpha_R / d log R.
    The measured slope of log(Q/W) on log R needs no correction, because it is that quantity already.
    This block exists so the two can be compared and the discrepancy seen rather than inferred.
    """
    if spec.chi_hat is None:
        return None
    # The same rule as `_points_full`: a round with no recursive coordinate does not contribute one,
    # because the local exponent is a derivative with respect to log R and the round number is not R.
    pts = [(rd.R, rd.U) for rd in readings if rd.U is not None and rd.R is not None and rd.R > 0]
    pts = [(float(R), float(U)) for R, U in pts if U is not None and U > 0 and R > 0]
    if len(pts) < 4:
        return None
    R = np.asarray([p[0] for p in pts], float)
    U = np.asarray([p[1] for p in pts], float)
    try:
        local = BAL.local_exponent_and_curvature(R, U)
        alpha_R = np.asarray(local["alpha_R"], float)
        curv = np.asarray(local["dlog_alpha_dlogR"], float)
        ok = np.isfinite(alpha_R) & np.isfinite(curv)
        if not ok.any():
            return None
        naive = np.asarray([BAL.balance_elasticity(a, spec.chi_hat) for a in alpha_R[ok]], float)
        corrected = np.asarray([BAL.local_balance(a, c, spec.chi_hat)
                                for a, c in zip(alpha_R[ok], curv[ok])], float)
    except Exception:
        return None
    return {"alpha_R_mean": float(np.mean(alpha_R[ok])),
            "dlog_alpha_dlogR_mean": float(np.mean(curv[ok])),
            "delta_constant_exponent": float(np.mean(naive)),
            "delta_curvature_corrected": float(np.mean(corrected)),
            "note": "the constant-exponent value is the one that misreads a shifted trajectory; the "
                    "corrected value is what the measured slope should agree with"}


def estimate(readings: Sequence[Reading], spec: ObservationSpec,
             variance: str = DEFAULT_VARIANCE) -> DeltaEstimate:
    """The window's Delta (or, for the unregistered surplus rate, its per-round trend) and its error.

    `variance` names the standard error the sequential rule will read; it belongs to the registered
    rule and not to the fit, which is why it is passed in rather than chosen here. See the variance
    block at the top of this module for what each name means and why the default is the conservative
    maximum.
    """
    if variance not in VARIANCE_ESTIMATORS:
        raise ObservationRefusal(
            "unknown variance estimator %r; one of %s" % (variance, ", ".join(VARIANCE_ESTIMATORS)))
    x, y, excluded, reason = _points(readings, spec)
    n = len(y)
    if n < MIN_POINTS:
        return DeltaEstimate(float("nan"), float("nan"), n, excluded, spec.registered,
                             _coordinate(spec),
                             reason="fewer than %d usable rounds in the window%s"
                                    % (MIN_POINTS, ("; " + reason) if reason else ""),
                             variance_estimator=variance)
    if spec.quantity == BALANCE_ELASTICITY:
        delta, se, se_ind, se_rob = _mean(y, variance)
    else:
        delta, se, se_ind, se_rob = _ols_full(x, y, variance)
    return DeltaEstimate(delta, se, n, excluded, spec.registered, _coordinate(spec), reason,
                         _curvature_check(readings, spec) if spec.curvature_correction else None,
                         se_independent=se_ind, se_robust=se_rob, variance_estimator=variance)


def _coordinate(spec: ObservationSpec) -> str:
    if spec.quantity == BALANCE_ELASTICITY:
        return "the balance elasticity Delta, read as a level over the post-settling window"
    if spec.quantity in (LOG_SERVICE_RATIO, SERVICE_RATIO):
        return "the balance elasticity Delta, being the slope of log(Q/W) on log R"
    return ("a per-round surplus rate against the round index, which is a level object and is not "
            "the balance elasticity")


# --------------------------------------------------------------------------------------------------
# The event
# --------------------------------------------------------------------------------------------------

def detect_event(readings: Sequence[Reading], spec: ObservationSpec, start: int,
                 z_threshold: float, variance: str = DEFAULT_VARIANCE,
                 min_points: Optional[int] = None) -> Dict[str, Any]:
    """The registered rule, in the coordinate the quantity settles.

    The window starts after the settling period and grows to the horizon; the event is declared at
    the first round where the estimate's interval lies wholly below zero at the registered threshold.
    What the estimate IS depends on the quantity, and that is the repair: for a Delta series the
    estimate is the level, so a falling but positive Delta never declares; for a ratio series it is
    the fitted slope on log R, which is Delta.
    """
    # SHORT-WINDOW BEHAVIOUR IS PART OF THE FROZEN RULE (finding A3). The first looks of a growing
    # window are the ones a persistent series wins on: at four points no variance estimator can see a
    # dependence length of eighteen, so the earliest looks fall back on something close to the
    # independence figure whatever estimator is registered. `min_points` is the size of the first
    # look, registered with the rest of the rule; the reference rule's four is the default so that
    # this repair changes no run by itself, and `p16_calibration.py` measures what the choice costs.
    first = max(int(min_points if min_points is not None else MIN_POINTS), MIN_POINTS)
    window = [rd for rd in readings if rd.round_index >= start]
    looks = 0
    for end in range(first, len(window) + 1):
        looks += 1
        est = estimate(window[:end], spec, variance)
        if np.isfinite(est.delta) and est.delta + z_threshold * est.se < 0:
            return {"declared_round": int(window[end - 1].round_index), "delta": est.delta,
                    "se": est.se, "event": spec.event, "falling": True,
                    "estimate": est.as_dict(), "looks_taken": looks,
                    "looks_available": max(len(window) - first + 1, 0),
                    "min_points": first}
    est = estimate(window, spec, variance)
    # THE LOOKS ARE COUNTED (finding A3): the family whose error rate is calibrated is the whole
    # schedule of looks across every arm, and a family whose size is not recorded cannot be calibrated
    # afterwards from the bundle.
    return {"declared_round": None, "delta": est.delta, "se": est.se, "event": spec.event,
            "falling": bool(np.isfinite(est.delta) and est.delta < 0), "estimate": est.as_dict(),
            "looks_taken": looks, "looks_available": max(len(window) - first + 1, 0),
            "min_points": first}


def arm_estimate(readings: Sequence[Reading], spec: ObservationSpec, start: int,
                 variance: str = DEFAULT_VARIANCE) -> DeltaEstimate:
    """The arm's own Delta, fitted once over the whole post-settling window.

    The declaration round's estimate is a selected value and therefore biased, which is why the line
    is fitted from this one and never from a mixture of the two: the first runner battery false
    refuted a true boundary on location twenty-eight times in a hundred by mixing them.

    This is also the TERMINAL estimate finding A3 reads a non-alarm from: the whole post-settling
    window, taken once at the horizon, which is the only look in the schedule that is not selected.
    """
    return estimate([rd for rd in readings if rd.round_index >= start], spec, variance)


def window_segments(readings: Sequence[Reading], spec: ObservationSpec, start: int,
                    n_segments: Optional[int], variance: str = DEFAULT_VARIANCE
                    ) -> Optional[List[Dict[str, Any]]]:
    """The same window read in consecutive equal segments, which is what ACROSS the window means.

    WHY THIS EXISTS. The registered refutation predicate is
    `arc_instruments.verdicts.p16_prohibition`'s `margin_lower_bound_positive_across_window`: a margin
    whose lower bound stays above zero ACROSS the window fixed before the run. One interval on the
    window MEAN is not that predicate and cannot be, because a mean is deaf to when the margin was
    where it was: an arm measured at plus 0.30 for forty-two rounds and then NEGATIVE for the last
    eight, the reversal having plainly happened, still has a whole-window interval wholly above the
    band. Reading that arm as demonstrated positivity is finding A3's defect relocated from the alarm
    counter to the terminal reading, so the terminal reading is taken in segments and every one of
    them has to clear.

    THE RESOLUTION IS REGISTERED AND IS NOT CHOSEN HERE, and the reason is a trade with no
    contract-given answer. A reversal confined to fewer rounds than one segment can hide inside it,
    which argues for many short segments; and a short segment cannot measure anything, which argues
    for few long ones. On this runner's coordinate the second edge binds hard: the registered
    estimator for a log-ratio series is its slope on log R, and log R barely moves late in a run, so
    a four-round segment near the horizon has a standard error several times the margin it is meant
    to measure. Measured on the calibration world whose margin is firmly positive everywhere (level
    0.30, noise 0.02): in halves every one of ninety arms sustains the margin above the band, in
    thirds forty-seven of ninety do, and at four segments the fourth segment's lower bound goes
    negative on a world with no reversal in it. In the same halves the flat null sustains nothing in
    ninety arms and every late-reversal adversary tried is caught. `n_segments` of None returns None,
    which is the fail-closed reading: an unregistered resolution leaves the across-window predicate
    unevaluated, and an unevaluated predicate refutes nothing.
    """
    if n_segments is None:
        return None
    k = int(n_segments)
    if k < 2:
        # ONE SEGMENT IS THE WINDOW MEAN, WHICH IS THE READING THIS FUNCTION EXISTS TO REPLACE. An
        # author who registers it has registered the defect under the name of the repair, and the
        # rest of the module would then report a predicate as evaluated when nothing was read across
        # anything. Refusing here is the one place that can say so; the fail-closed state for an
        # author who has not chosen a resolution is None, which leaves the predicate unevaluated.
        raise ObservationRefusal(
            "an across-window resolution of %r reads the window in one piece, which is its mean and "
            "not a reading across it: the registered predicate is a lower bound that STAYS above the "
            "band, and one segment cannot show that anything stayed anywhere. Register two or more, "
            "or register none and leave the predicate unevaluated" % (n_segments,))
    window = [rd for rd in readings if rd.round_index >= start]
    n = len(window)
    if n < k * MIN_POINTS:
        # NOT an empty list, which a caller could read as "no segment failed". A window too short to
        # be read at the registered resolution has not been read at it.
        return None
    edges = [int(round(i * n / float(k))) for i in range(k + 1)]
    out: List[Dict[str, Any]] = []
    for i in range(k):
        seg = window[edges[i]:edges[i + 1]]
        est = estimate(seg, spec, variance)
        out.append({"first_round": int(seg[0].round_index), "last_round": int(seg[-1].round_index),
                    "delta": float(est.delta), "se": float(est.se),
                    "se_independent": float(est.se_independent), "se_robust": est.se_robust,
                    "n_used": int(est.n_used), "n_excluded": int(est.n_excluded)})
    return out


# --------------------------------------------------------------------------------------------------
# The two measurements the support branch needs beside the event: what the arm was exposed to, and
# when the exposure changed
# --------------------------------------------------------------------------------------------------

DELIVERY_KEY = "delivery"


def delivery_summary(readings: Sequence[Reading]) -> Dict[str, Any]:
    """What the apparatus attested, round by round, about administering this arm's dose.

    The attestation lives at `reading.extra["delivery"]` and is a mapping with `applied`, a boolean
    saying whether this round ran under the arm's own dose rather than the baseline, and optionally
    `lever`, whatever the apparatus moved to do it. It is an attestation by the apparatus and not an
    inference from the numbers, which is exactly why it is a separate component from realised
    exposure: delivery is the procedural record that the dose was administered, and realised exposure
    is the independent measurement of what it produced. A source that attests nothing has supplied no
    delivery evidence, and `attested` is False rather than a defaulted True: an unattested arm cannot
    be told apart from an arm the apparatus silently failed to dose.
    """
    readings = list(readings)
    n = 0
    applied: List[int] = []
    levers: List[Any] = []
    for rd in readings:
        rec = (rd.extra or {}).get(DELIVERY_KEY) if isinstance(rd.extra, dict) else None
        if not isinstance(rec, dict):
            continue
        n += 1
        if bool(rec.get("applied")):
            applied.append(int(rd.round_index))
            if rec.get("lever") is not None and rec.get("lever") not in levers:
                levers.append(rec.get("lever"))
    return {"attested": n > 0 and n == len(readings),
            "n_attested": n, "n_rounds": len(readings),
            "n_applied": len(applied),
            "first_applied_round": (min(applied) if applied else None),
            "levers": levers}


def realised_exposure(readings: Sequence[Reading], start: int) -> Dict[str, Any]:
    """The growth exponent the arm ACTUALLY realised, from the capability it reached against the depth
    it reached it at: the OLS slope of log U on log R over the post-settling window.

    WHY THIS IS MEASURED AND NOT READ OFF THE ASSIGNMENT. Finding A2: the reference runner handed each
    arm an assigned `alpha_arm` and then used that same assigned number as the x coordinate of the
    fitted balance line. A titration whose dose lever fails, saturates or overshoots then reports a
    line drawn against the doses somebody intended rather than the exponents the systems reached, and
    an arm assigned above the boundary whose realised exponent is below it counts as an
    above-boundary arm throughout. The registered design locates the boundary from realised
    exponents (see `code_domain.locate_boundary`, whose x is the realised exponent), so the arms must
    be read the same way.

    U is the capability the ladder measured and R the recursive coordinate; an arm whose source
    supplies neither has not measured its exposure, and `measured` is False rather than a number.
    """
    # AND HERE TOO THE ROUND NUMBER IS NOT R. This measurement is a slope of log U on log R, so a
    # round that carried no recursive coordinate carries no point on that plane; substituting the
    # round index would report an exponent measured against a different abscissa and call it the
    # realised exposure, which is the same fail-open the estimator had.
    pts = [(rd.R, rd.U) for rd in readings
           if rd.round_index >= start and rd.U is not None and rd.R is not None]
    pts = [(float(R), float(U)) for R, U in pts if R is not None and R > 0 and U is not None and U > 0]
    if len(pts) < MIN_POINTS:
        return {"measured": False, "alpha": float("nan"), "se": float("nan"), "n": len(pts),
                "reason": "fewer than %d rounds carried both a positive capability U and a positive "
                          "recursive coordinate R, so no realised growth exponent was measured" % MIN_POINTS}
    x = np.log(np.asarray([p[0] for p in pts], float))
    y = np.log(np.asarray([p[1] for p in pts], float))
    if float(np.sum((x - x.mean()) ** 2)) <= 0:
        return {"measured": False, "alpha": float("nan"), "se": float("nan"), "n": len(pts),
                "reason": "the recursive coordinate did not move over the window, so no exponent is "
                          "identified from it"}
    slope, se = _ols(x, y)
    return {"measured": True, "alpha": float(slope), "se": float(se), "n": len(pts),
            "estimator": "ols-slope-of-log-capability-on-log-recursive-coordinate",
            "reason": ""}


def change_point(readings: Sequence[Reading], spec: ObservationSpec,
                 min_segment: int = MIN_POINTS) -> Dict[str, Any]:
    """The round at which the observation model changed, by least squares over every candidate split.

    THIS IS NOT THE DECLARATION ROUND, and the difference is the whole of the timing component. The
    declaration round is when the growing-window rule has accumulated enough precision to put the
    interval wholly below zero, so it moves with the noise, the horizon and the threshold: on the
    reference configuration a true switch at round 8 is declared somewhere in the high twenties. The
    registered timing is the change point, being where the series' shape actually changes, and it is
    the quantity the sealed tolerance is a tolerance on.

    The model fitted on each side is the one the quantity settles: segment means for a Delta series,
    because its estimator is the level, and a segment OLS in the declared coordinate otherwise. The
    two segments are fitted independently, so a shape change is found whether or not the path is
    continuous through it; a continuous break simply costs one free parameter that buys nothing.
    """
    rounds, x, y, excluded, reason = _points_full(readings, spec)
    n = len(y)
    if n < 2 * min_segment:
        return {"measured": False, "round": None, "n": n,
                "reason": "fewer than %d usable rounds, so no split leaves a segment on each side"
                          % (2 * min_segment)}
    level = spec.quantity == BALANCE_ELASTICITY

    def ssr(xi: np.ndarray, yi: np.ndarray) -> float:
        if level or len(yi) < 3 or float(np.sum((xi - xi.mean()) ** 2)) <= 0:
            return float(np.sum((yi - yi.mean()) ** 2))
        A = np.vstack([xi, np.ones_like(xi)]).T
        coef = np.linalg.lstsq(A, yi, rcond=None)[0]
        return float(np.sum((yi - A @ coef) ** 2))

    best_i, best = None, float("inf")
    for i in range(min_segment, n - min_segment + 1):
        total = ssr(x[:i], y[:i]) + ssr(x[i:], y[i:])
        if total < best:
            best_i, best = i, total
    if best_i is None:
        return {"measured": False, "round": None, "n": n, "reason": "no admissible split"}
    return {"measured": True, "round": int(rounds[best_i]), "n": n, "ssr": float(best),
            "estimator": "least-squares split over candidate rounds, %s on each side"
                         % ("segment means" if level else "segment slopes in the declared coordinate"),
            "n_excluded": int(excluded), "reason": reason}


# --------------------------------------------------------------------------------------------------
# The sealed line, and whether the run may be compared with it
# --------------------------------------------------------------------------------------------------

def line_comparison(spec: ObservationSpec, fitted_slope: float, sealed_slope: float) -> Dict[str, Any]:
    """Whether the fitted line's slope may be compared with the sealed one, and the comparison itself.

    Finding A1's second half. The sealed slope is d Delta / d alpha across arms, dimensionless. A
    per-round surplus rate's trend is a rate of change of a level per round. The reference runner
    reported the sealed minus one half beside a fitted minus 0.01 as though the two were candidates
    for the same number. There is no conversion between them without Q, W and R, so none is offered:
    the comparison is refused by name and the run says which coordinate it was in.
    """
    if not spec.registered:
        return {"comparable": False, "comparison_made": False,
                "fitted_slope": _f(fitted_slope), "sealed_slope": _f(sealed_slope),
                "units": UNITS[spec.quantity],
                "reason": "the observed quantity is %r, a per-round %s. The sealed slope is d Delta / "
                          "d alpha, dimensionless. They are in different coordinates and no conversion "
                          "exists without Q, W and the recursive coordinate R, so no comparison is made."
                          % (spec.quantity, "surplus rate")}
    # COMPARABLE AND MEASURED ARE TWO DIFFERENT THINGS, AND THEY ARE REPORTED SEPARATELY. `comparable`
    # is about the coordinate: a registered quantity's fitted line and the sealed line are the same
    # object. `comparison_made` is about the run: a declaration is not a measurement, so a registered
    # quantity whose rounds were all unusable has a fitted slope that is not a number, and a reader
    # shown a difference of None beside no reason would take the coordinate for the measurement.
    if fitted_slope is None or not np.isfinite(fitted_slope):
        return {"comparable": True, "comparison_made": False, "fitted_slope": _f(fitted_slope),
                "sealed_slope": _f(sealed_slope), "difference": None,
                "units": "dimensionless: d Delta / d alpha, both sides",
                "reason": "the observed quantity is the balance elasticity in the registered "
                          "coordinate, and no finite slope was fitted in it, so there is nothing to "
                          "compare with the sealed line"}
    d = None
    if sealed_slope not in (None, 0):
        d = float(fitted_slope) - float(sealed_slope)
    return {"comparable": True, "comparison_made": True,
            "fitted_slope": _f(fitted_slope), "sealed_slope": _f(sealed_slope),
            "difference": d, "units": "dimensionless: d Delta / d alpha, both sides",
            "reason": "the observed quantity is the balance elasticity in the registered coordinate, "
                      "so the fitted line and the sealed line are the same object"}
