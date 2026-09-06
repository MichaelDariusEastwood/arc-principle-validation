"""Canonical outcome regions for the twenty-two propositions of registration v1.93.

One scoring function per proposition, on confidence intervals and recorded events, so that every prose
summary in the registration can be generated from, or checked against, a single source. The
registration governs: the numbers below are the registered margins where the registration fixes them
(P8 and P19 plus or minus 0.10; P12 0.10 on the correction-leverage scale; P17 0.20 on the log scale;
P10 ten per cent; P3 tolerance 0.10; P15 one half; P22 delta 0.10) and the reconciliation this module
carries is the one the reviews asked for: the interval rule dominates every short form.

Conventions, second edition (5 September 2026, afternoon). An interval is a pair (lo, hi) with lo < hi.
"Wholly above x" means lo > x, strictly. "Wholly inside (a, b)" means a < lo and hi < b, strictly: a
boundary contact is not a clearance and is scored INCONCLUSIVE, in every direction, so that support,
refutation and equivalence all need the same strict clearance. A zero-width interval is not an
experimental result: an exactly known equality is a logical statement, so every region function
returns NOT EVALUABLE for it rather than a verdict. The first edition of this module scored contacts
as inside a closed region; that reading is kept behind `closed=True` for comparison only. The operator
ruled on 5 September 2026 (ruling 17): strict clearance everywhere, one rule for every proposition. The
fixed numbers follow ruling 10 of the same day: P3 tolerance 0.10, P17 margin 0.20 on the log ratio, P5
and P11 bands 0.10, and the registration's other numbers as v1.92 fixed them. The regions module exports
the table with every axis open.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Interval = Tuple[float, float]

# Verdict vocabulary. Strings rather than an enum so that they can be written straight into a report.
SUPPORTED = "SUPPORTED"
REFUTED = "REFUTED"
INCONCLUSIVE = "INCONCLUSIVE"
NOT_EVALUABLE = "NOT EVALUABLE"
NOT_DISCRIMINATING = "NOT DISCRIMINATING"
INSTRUMENT_FAILED = "INSTRUMENT FAILED"
AWAITING_REPLICATION = "AWAITING FRESH-DATA REPLICATION"

ABOVE_NULL = "ABOVE NULL"
BELOW_NULL = "BELOW NULL"
EQUIVALENT_TO_NULL = "EQUIVALENT TO NULL"
CLEARS_UNITY = "CLEARS UNITY"
DOES_NOT_CLEAR_UNITY = "DOES NOT CLEAR UNITY"

CHECKABLE_HIGHER = "CHECKABLE HIGHER"
JUDGED_HIGHER = "JUDGED HIGHER"
PRACTICALLY_EQUAL = "PRACTICALLY EQUAL"

EQUIVALENT_TO_ZERO = "EQUIVALENT TO ZERO"
MATERIALLY_POSITIVE = "MATERIALLY POSITIVE"
MATERIALLY_NEGATIVE = "MATERIALLY NEGATIVE"
WHOLLY_ABOVE_ONE_HALF = "WHOLLY ABOVE ONE HALF"
WHOLLY_BELOW_ONE_HALF = "WHOLLY BELOW ONE HALF"
REFUTES_WORDING_NOT_SUPPORT = "REFUTES THE WORDING, NOT SUPPORT"
OUT_SCALES_DRIFT = "OUT-SCALES DRIFT"
LAGS_DRIFT = "LAGS DRIFT"

INDEPENDENT = "INDEPENDENT"
CORRELATED = "CORRELATED"
ANTI_CORRELATED = "ANTI-CORRELATED"
INSUFFICIENT_PRECISION = "INSUFFICIENT PRECISION"

CONSISTENT_NOT_SUPPORTIVE = "CONSISTENT, NOT SUPPORTIVE"
NO_COUNTEREXAMPLE = "NO COUNTEREXAMPLE FOUND AMONG N"
CORRECTION_ADVANTAGE = "CORRECTION HOLDS THE ASYMPTOTIC ADVANTAGE"
BURDEN_ADVANTAGE = "BURDEN HOLDS THE ASYMPTOTIC ADVANTAGE"
EQUALITY_UNDECIDED = "EQUALITY: UNDECIDED BY EXPONENTS"
SUPPORTED_WITH_MECHANISM = "SUPPORTED, MECHANISM CONSISTENT"
SUPPORTED_MECHANISM_UNEXPLAINED = "SUPPORTED, MECHANISM UNEXPLAINED"
INTERIOR_MAXIMUM = "INTERIOR MAXIMUM"
MONOTONE = "MONOTONE"
ENDPOINT_MAXIMUM = "ENDPOINT MAXIMUM"
FLAT = "FLAT"
MULTIMODAL = "MULTIMODAL"
UNRESOLVED = "UNRESOLVED"
DISTINCT = "DISTINCT"
IDENTICAL_WITHIN_MARGIN = "IDENTICAL WITHIN MARGIN (REFUTES DISTINCTNESS)"
ABOVE_ONE = "ABOVE ONE"
BELOW_ONE = "BELOW ONE"
NEAR_ONE = "NEAR ONE"
NO_SURVEY = "NO SURVEY"
FIT_FAILURE = "FIT FAILURE"
PERFORMANCE_SUPPORTED_MECHANISM_UNRESOLVED = "PERFORMANCE SUPPORTED, MECHANISM UNRESOLVED"
PERFORMANCE_SUPPORTED_MECHANISM_REFUTED = "PERFORMANCE SUPPORTED, MECHANISM REFUTED"

# Permitted cell states for P1's FORM adjudication: the power label, a registered rival family, or an
# unresolved state. Anything else is a typing error and is refused, never counted.
UNRESOLVED_CELL_LABELS = frozenset({INCONCLUSIVE, NOT_DISCRIMINATING, "TIE", "UNRESOLVED", FIT_FAILURE, ""})
RIVAL_FAMILIES = frozenset({"logarithmic", "exponential", "saturating", "broken", "geometric", "shifted"})

# P16 endpoints (see balance.py): only a margin reversal is read from the balance elasticity alone.
MARGIN_REVERSAL = "margin-reversal"
SERVICE_DEFICIT = "service-deficit"
BACKLOG_THRESHOLD = "backlog-threshold"
CONFORMANCE_FAILURE = "conformance-failure"
ENDPOINTS = (MARGIN_REVERSAL, SERVICE_DEFICIT, BACKLOG_THRESHOLD, CONFORMANCE_FAILURE)


def _check(iv: Interval) -> Interval:
    lo, hi = float(iv[0]), float(iv[1])
    if lo > hi:
        raise ValueError("interval lower bound exceeds upper bound: %r" % (iv,))
    return lo, hi


def is_point(iv: Interval) -> bool:
    """A zero-width interval: an exact equality, which is a logical result and not a measurement."""
    lo, hi = _check(iv)
    return lo == hi


def wholly_above(iv: Interval, x: float) -> bool:
    lo, _ = _check(iv)
    return lo > x


def wholly_below(iv: Interval, x: float) -> bool:
    _, hi = _check(iv)
    return hi < x


def wholly_inside(iv: Interval, a: float, b: float, closed: bool = False) -> bool:
    """Strict containment by default: a boundary contact is not inside."""
    lo, hi = _check(iv)
    if closed:
        return a <= lo and hi <= b
    return a < lo and hi < b


def region_verdict(iv: Interval, centre: float, margin: float,
                   above: str, below: str, equivalent: str, closed: bool = False) -> str:
    """Three-region rule with an equivalence band (centre - margin, centre + margin), strict clearance.

    A zero-width interval returns NOT EVALUABLE. Under `closed=True` the first-edition reading applies:
    contacts count as inside the closed equivalence region."""
    if is_point(iv):
        return NOT_EVALUABLE
    if wholly_inside(iv, centre - margin, centre + margin, closed=closed):
        return equivalent
    if wholly_above(iv, centre + margin):
        return above
    if wholly_below(iv, centre - margin):
        return below
    return INCONCLUSIVE


# ---------------------------------------------------------------------------------------------- P1
def p1_form(cells: Sequence[str], false_selection_rate: float, registered_max_rate: float,
            power_label: str = "power") -> str:
    """FORM verdict: the power family against the registered rivals across estimable cells.

    A FORM verdict may not be read as support until the deciding unit has demonstrated its own
    false-selection rate and registered the maximum; where the demonstrated rate exceeds the
    registered maximum the outcome is NOT DISCRIMINATING regardless of the cell count. A cell won by the
    shifted finite-window solution of the registered growth equation is not a power-family cell: it is
    credited to P5's mechanism, so the caller labels such cells with a different string.
    """
    if false_selection_rate > registered_max_rate:
        return NOT_DISCRIMINATING
    cells = list(cells)
    if not cells:
        return INCONCLUSIVE
    wins = losses = 0
    for c in cells:
        if c == power_label:
            wins += 1
        elif c in RIVAL_FAMILIES:
            losses += 1
        elif c is None or c in UNRESOLVED_CELL_LABELS:
            continue                      # an unresolved cell is neither a win nor a rival victory
        else:
            raise ValueError("unknown cell label %r; permitted: the power label, a registered rival family, or an unresolved state" % (c,))
    total = len(cells)                    # unresolved cells stay in the denominator: they cannot manufacture a majority
    if wins * 2 > total:
        return SUPPORTED
    if losses * 2 > total:
        return REFUTED
    return INCONCLUSIVE


def p1_recursion(depth_minus_breadth: Interval, margin: float) -> str:
    """RECURSION verdict: the depth exponent exceeds the compute-matched breadth exponent by the
    registered margin. A linear fit (exponent one) is inside the power family; whether the exponent
    clears one is P8's question and not this one."""
    if is_point(depth_minus_breadth):
        return NOT_EVALUABLE
    if wholly_above(depth_minus_breadth, margin):
        return SUPPORTED
    if wholly_below(depth_minus_breadth, margin):
        return REFUTED
    return INCONCLUSIVE


# ---------------------------------------------------------------------------------------------- P2
def p2_shape(betas: Sequence[float], values: Sequence[float], ses: Sequence[float],
             k: float = 1.96, flat_margin: Optional[float] = None) -> str:
    """Shape of the sustained-improvement profile over the reinvestment share.

    INTERIOR MAXIMUM: the best cell is interior and beats both endpoints by more than k standard
    errors of the difference. MONOTONE: every successive step is resolved in the same direction.
    ENDPOINT MAXIMUM: the best cell is an endpoint and beats the other endpoint, but the profile is not
    resolved monotone; it still refutes an interior-maximum prediction, for the right reason. FLAT: a
    resolved equivalence, every pairwise difference lying within the registered flatness margin with its
    uncertainty; without a registered margin, or with imprecise cells, flatness is never returned, because
    failing to resolve a difference is not evidence of flatness. MULTIMODAL: two interior cells beat their
    neighbours. UNRESOLVED otherwise. Both endpoints are excluded from the interior by construction.
    """
    b = list(map(float, betas)); v = list(map(float, values)); s = list(map(float, ses))
    n = len(b)
    if n < 3 or not (len(v) == len(s) == n):
        return UNRESOLVED
    order = sorted(range(n), key=lambda i: b[i])
    b = [b[i] for i in order]; v = [v[i] for i in order]; s = [s[i] for i in order]

    def se_diff(i: int, j: int) -> float:
        return (s[i] ** 2 + s[j] ** 2) ** 0.5

    def beats(i: int, j: int) -> bool:
        return (v[i] - v[j]) > k * se_diff(i, j)

    def equivalent(i: int, j: int) -> bool:
        return flat_margin is not None and abs(v[i] - v[j]) + k * se_diff(i, j) < flat_margin

    peaks = [i for i in range(1, n - 1) if beats(i, i - 1) and beats(i, i + 1)]
    if len(peaks) >= 2:
        return MULTIMODAL
    best = max(range(n), key=lambda i: v[i])
    if 0 < best < n - 1 and beats(best, 0) and beats(best, n - 1):
        return INTERIOR_MAXIMUM
    if best in (0, n - 1) and beats(best, n - 1 - best):
        increasing = all(beats(i + 1, i) for i in range(n - 1))
        decreasing = all(beats(i, i + 1) for i in range(n - 1))
        return MONOTONE if (increasing or decreasing) else ENDPOINT_MAXIMUM
    if all(equivalent(i, j) for i in range(n) for j in range(i + 1, n)):
        return FLAT
    return UNRESOLVED


# ---------------------------------------------------------------------------------------------- P3
def p3_frontier(paired_difference: Optional[Interval], correctability_positive: bool,
                p20_passed: bool, gamma_measured: bool, exceedance_replicated: bool,
                tolerance: float = 0.10) -> str:
    """The corrected relation upper-bounds the relative scaling frontier. Four outcomes.

    REFUTED: a replicated exceedance of the returned value while the balance stays independently
    positive in that regime. SUPPORTED: only where P20's held-out form test passed and the paired
    difference between the measured frontier and the value the relation returns from the same system's
    service elasticity lies wholly within the registered tolerance, with joint uncertainty. CONSISTENT,
    NOT SUPPORTIVE: no violation found but the frontier not located within tolerance. NOT EVALUABLE:
    the service elasticity unmeasured. The object scored is the trend frontier (balance.py), never a
    finite-slack window: an above-bound exponent with ample service for a finite interval is not a
    violation of this proposition.
    """
    if not gamma_measured:
        return NOT_EVALUABLE
    if exceedance_replicated and correctability_positive:
        return REFUTED
    if paired_difference is None:
        return CONSISTENT_NOT_SUPPORTIVE
    if is_point(paired_difference):
        return NOT_EVALUABLE
    if p20_passed and wholly_inside(paired_difference, -tolerance, tolerance):
        return SUPPORTED
    return CONSISTENT_NOT_SUPPORTIVE


# ---------------------------------------------------------------------------------------------- P4
def p4_regime(correction_exponent: Interval, drift_exponent: Interval, margin: float = 0.0) -> str:
    """Three regimes of the minimal model, decided on the exponents alone. At equality the finite-run
    outcome is decided by coefficients, delay and backlog, which this function does not see. The
    regime is a trend statement; the response P4 scores is the separately calibrated model's held-out
    prediction, which the caller compares outside this function."""
    g = _check(correction_exponent); k = _check(drift_exponent)
    if g[0] > k[1] + margin:
        return CORRECTION_ADVANTAGE
    if g[1] < k[0] - margin:
        return BURDEN_ADVANTAGE
    if abs((g[0] + g[1]) / 2 - (k[0] + k[1]) / 2) <= margin and (g[1] - g[0]) <= 2 * margin and (k[1] - k[0]) <= 2 * margin:
        return EQUALITY_UNDECIDED
    return INCONCLUSIVE


# ---------------------------------------------------------------------------------------------- P5
def p5_agreement(predicted_minus_fitted: Interval, margin: float = 0.10) -> str:
    """The sealed finite-window exponent predicted from the separately measured coupling agrees with the
    fitted exponent on the validation trajectories within the registered margin. Supported by
    agreement, so the instrument's sensitivity must be demonstrated first (Condition D)."""
    if is_point(predicted_minus_fitted):
        return NOT_EVALUABLE
    if wholly_inside(predicted_minus_fitted, -margin, margin):
        return SUPPORTED
    if wholly_above(predicted_minus_fitted, margin) or wholly_below(predicted_minus_fitted, -margin):
        return REFUTED
    return INCONCLUSIVE


# ---------------------------------------------------------------------------------------------- P6
def p6_survey(exponent_intervals: Sequence[Interval], bound: float = 0.5, replicated: bool = False) -> str:
    """A universal same-class bound has no supported verdict. REFUTED on a replicated interval wholly
    above the bound, and AWAITING FRESH-DATA REPLICATION until the repetition is done (the default is
    unreplicated: no caller inherits a refutation by omission); NO COUNTEREXAMPLE FOUND AMONG N where
    every interval lies wholly below the bound; INCONCLUSIVE where any interval straddles or touches the
    bound; NO SURVEY for an empty roster. Precision near the bound is a reported operating characteristic
    (precision.exceedance_power), never a verdict switch."""
    ivs = list(exponent_intervals)
    if not ivs:
        return NO_SURVEY                  # an empty roster is not a completed survey
    if any(is_point(iv) for iv in ivs):
        return NOT_EVALUABLE
    if any(wholly_above(iv, bound) for iv in ivs):
        return REFUTED if replicated else AWAITING_REPLICATION
    if all(wholly_below(iv, bound) for iv in ivs):
        return NO_COUNTEREXAMPLE
    return INCONCLUSIVE


# ---------------------------------------------------------------------------------------------- P8
def p8_departure(exponent: Interval, null: float = 0.5, margin: float = 0.10) -> str:
    """Departure from the one-half reference, four outcomes on the region (null - margin, null + margin).
    The interval [0.35, 0.45] is INCONCLUSIVE here, not REFUTED: it straddles 0.40. The interval
    [0.48, 0.52] is EQUIVALENT TO NULL. Every short form in the registration must agree with this
    function. The reference is a named model (the variance of a mean of independent readings), not a
    theorem-free null; that is a matter for the text, not for the region."""
    return region_verdict(exponent, null, margin, ABOVE_NULL, BELOW_NULL, EQUIVALENT_TO_NULL)


def p8_clears_unity(exponent: Interval) -> str:
    """The separate second verdict: the super-linear content of the conversion claim."""
    if is_point(exponent):
        return NOT_EVALUABLE
    if wholly_above(exponent, 1.0):
        return CLEARS_UNITY
    if wholly_below(exponent, 1.0):
        return DOES_NOT_CLEAR_UNITY
    return INCONCLUSIVE


# ---------------------------------------------------------------------------------------------- P9
def p9_dimension(predicted_exponent: float, measured: Interval, margin: float = 0.10) -> str:
    if is_point(measured):
        return NOT_EVALUABLE
    lo, hi = _check(measured)
    if wholly_inside((lo - predicted_exponent, hi - predicted_exponent), -margin, margin):
        return SUPPORTED
    if wholly_above(measured, predicted_exponent + margin) or wholly_below(measured, predicted_exponent - margin):
        return REFUTED
    return INCONCLUSIVE


# --------------------------------------------------------------------------------------------- P10
def p10_material_fraction(fraction_not_surviving: Interval, threshold: float = 0.10) -> str:
    """Fraction of earlier unblinded same-family results that fail cross-family re-scoring. SUPPORTED
    where the interval lies wholly above ten per cent; REFUTED wholly below; INCONCLUSIVE otherwise,
    including a contact with the threshold. Refutation does not restore the earlier scores' standing;
    it means the re-scoring found no material loss. Support challenges the affected earlier scores; it
    validates nothing else."""
    if is_point(fraction_not_surviving):
        return NOT_EVALUABLE
    if wholly_above(fraction_not_surviving, threshold):
        return SUPPORTED
    if wholly_below(fraction_not_surviving, threshold):
        return REFUTED
    return INCONCLUSIVE


# --------------------------------------------------------------------------------------------- P11
def p11_pair(difference: Interval, margin: float = 0.10) -> str:
    """Residual-decay exponent against correction-leverage exponent. Refuted by agreement (identical
    within the registered margin, 0.10 on the exponent scale by ruling 10 of 5 September 2026); distinct
    where the difference lies wholly outside it. Agreement at the tested resolution is
    interchangeability there, never construct identity."""
    if is_point(difference):
        return NOT_EVALUABLE
    if wholly_inside(difference, -margin, margin):
        return IDENTICAL_WITHIN_MARGIN
    if wholly_above(difference, margin) or wholly_below(difference, -margin):
        return DISTINCT
    return INCONCLUSIVE


# --------------------------------------------------------------------------------------------- P12
def p12_build_order(change_in_leverage_exponent: Interval, manipulation_delivered: bool,
                    sesoi: float = 0.10) -> str:
    """Build order changes the correction-leverage exponent by at least the SESOI, in either direction
    (an undirected material-change wager). Validity asks whether the planned ordering was administered,
    never whether it produced the predicted effect; an adequately delivered null is a result."""
    if not manipulation_delivered:
        return INSTRUMENT_FAILED
    if is_point(change_in_leverage_exponent):
        return NOT_EVALUABLE
    if wholly_above(change_in_leverage_exponent, sesoi) or wholly_below(change_in_leverage_exponent, -sesoi):
        return SUPPORTED
    if wholly_inside(change_in_leverage_exponent, -sesoi, sesoi):
        return REFUTED
    return INCONCLUSIVE


# --------------------------------------------------------------------------------------------- P13
def p13_performance(cross_minus_same: Interval) -> str:
    """The performance component: cross-family minus same-family held-out misalignment reduction.
    Wholly above zero supports it; wholly below refutes it; a contact with zero is INCONCLUSIVE."""
    if is_point(cross_minus_same):
        return NOT_EVALUABLE
    if wholly_below(cross_minus_same, 0.0):
        return REFUTED
    if wholly_above(cross_minus_same, 0.0):
        return SUPPORTED
    return INCONCLUSIVE


def p13_conjunction(cross_minus_same: Interval, dependence_verdict: str) -> str:
    """P13 is one conjunction of two components: the performance axis and the dependence contrast
    (same-family minus cross-family residual dependence under the charter's material-effect rule, whose
    verdict the caller supplies as SUPPORTED, REFUTED, INCONCLUSIVE or NOT EVALUABLE). The aggregate
    follows the conjunction rule: a refuted component refutes; a missing one makes the whole not
    evaluable; an unresolved one makes it inconclusive; only both supported supports. A performance
    advantage with an unresolved mechanism is reported as that cell (p13_components) and is never full
    support; evidence against the mechanism and absence of evidence about it are different cells."""
    if dependence_verdict not in (SUPPORTED, REFUTED, INCONCLUSIVE, NOT_EVALUABLE):
        raise ValueError("dependence verdict must be one of SUPPORTED, REFUTED, INCONCLUSIVE, NOT EVALUABLE")
    return conjunction([p13_performance(cross_minus_same), dependence_verdict])


def p13_components(cross_minus_same: Interval, dependence_verdict: str) -> Dict[str, str]:
    perf = p13_performance(cross_minus_same)
    agg = p13_conjunction(cross_minus_same, dependence_verdict)
    if perf == SUPPORTED and dependence_verdict == SUPPORTED:
        cell = SUPPORTED_WITH_MECHANISM
    elif perf == SUPPORTED and dependence_verdict == REFUTED:
        cell = PERFORMANCE_SUPPORTED_MECHANISM_REFUTED
    elif perf == SUPPORTED:
        cell = PERFORMANCE_SUPPORTED_MECHANISM_UNRESOLVED
    else:
        cell = agg
    return {"performance": perf, "dependence": dependence_verdict, "aggregate": agg, "cell": cell}


# --------------------------------------------------------------------------------------------- P14
def p14_blinded_shrink(blinded_minus_unblinded_gain: Interval) -> str:
    """Gains shrink under blinded cross-family scoring. REFUTED where the interval on the blinded minus
    unblinded gain sits wholly above zero; SUPPORTED wholly below; INCONCLUSIVE otherwise, including a
    contact with zero. The sign is part of what is measured: a judge that disfavours its own family
    refutes this too. Refutation leaves the earlier unblinded scores standing; support challenges the
    affected scores only."""
    if is_point(blinded_minus_unblinded_gain):
        return NOT_EVALUABLE
    if wholly_above(blinded_minus_unblinded_gain, 0.0):
        return REFUTED
    if wholly_below(blinded_minus_unblinded_gain, 0.0):
        return SUPPORTED
    return INCONCLUSIVE


# --------------------------------------------------------------------------------------------- P15
def p15_decay(fraction_retained: Interval, threshold: float = 0.5) -> str:
    """Externally installed gains decay under subsequent capability training: more than half lost. The
    threshold is a convention and is described as such wherever it appears. A denominator that is not
    identified as positive makes the ratio NOT EVALUABLE upstream (see eclipse_ratio)."""
    if is_point(fraction_retained):
        return NOT_EVALUABLE
    if wholly_below(fraction_retained, threshold):
        return SUPPORTED
    if wholly_above(fraction_retained, threshold):
        return REFUTED
    return INCONCLUSIVE


# --------------------------------------------------------------------------------------------- P16
def p16_prohibition(excursion_observed: bool, margin_negative_after: Optional[bool],
                    margin_lower_bound_positive_across_window: Optional[bool],
                    replicated: bool, censored: bool = False) -> str:
    """The first-edition prohibition form, kept for the historical rule: not both an exponent above the
    held-out boundary and a correction margin whose lower bound stays above zero across the window fixed
    before the run. p16_endpoint is the typed form the second edition scores."""
    if censored or not excursion_observed:
        return INCONCLUSIVE
    if margin_negative_after is True and replicated:
        return SUPPORTED
    if margin_lower_bound_positive_across_window is True and replicated:
        return REFUTED
    return INCONCLUSIVE


def p16_endpoint(endpoint: str, entered_region: bool, event_depth: Optional[float],
                 sealed_window: Optional[Interval], horizon: float, controls_distinguish: Optional[bool],
                 replicated: bool, response_model_registered: bool = False) -> str:
    """A pre-estimated boundary predicts a later event, with the event typed.

    The only endpoint the balance elasticity alone can predict is a margin reversal (the first
    persistent negative measured balance). A service deficit, a backlog threshold or a conformance
    failure needs a registered response model with coefficients, initial state, delays, horizon and
    uncertainty; without one those endpoints are NOT EVALUABLE. SUPPORTED: the trajectory entered the
    forecast above-boundary region, the event fell inside the sealed timing window, the non-crossing
    controls distinguish the boundary account from generic deterioration, and a fresh-data repetition
    met the same rule. REFUTED: the informative horizon passed with no event, or the event fell outside
    the sealed window, replicated. No entry into the region, censoring before the horizon, or an
    unreplicated result is INCONCLUSIVE or AWAITING FRESH-DATA REPLICATION.
    """
    if endpoint not in ENDPOINTS:
        raise ValueError("unknown endpoint %r" % (endpoint,))
    if endpoint != MARGIN_REVERSAL and not response_model_registered:
        return NOT_EVALUABLE
    if not entered_region or sealed_window is None:
        return INCONCLUSIVE
    lo, hi = _check(sealed_window)
    if event_depth is None:
        if horizon > hi:
            return REFUTED if replicated else AWAITING_REPLICATION
        return INCONCLUSIVE
    inside = lo <= event_depth <= hi
    if inside:
        if controls_distinguish is not True:
            return INCONCLUSIVE
        return SUPPORTED if replicated else AWAITING_REPLICATION
    return REFUTED if replicated else AWAITING_REPLICATION


# --------------------------------------------------------------------------------------------- P17
def p17_pairwise(log_ratio: Interval, margin: float = 0.20) -> str:
    """Pairwise joint misses against the conditional-independence baseline on the frozen panel and fault
    set, on the log scale. INDEPENDENT wholly inside the margin; CORRELATED wholly above; ANTI-CORRELATED
    wholly below; INSUFFICIENT PRECISION otherwise. Both departures contradict practical independence.
    Nothing here locates a ceiling, estimates an exponent, or speaks to temporal accumulation."""
    if is_point(log_ratio):
        return NOT_EVALUABLE
    v = region_verdict(log_ratio, 0.0, margin, CORRELATED, ANTI_CORRELATED, INDEPENDENT)
    return INSUFFICIENT_PRECISION if v == INCONCLUSIVE else v


def p17_panel(pair_verdicts: Iterable[str]) -> str:
    """Panel-level reading: SUPPORTED only where every primary pair is INDEPENDENT; REFUTED where any
    pair is CORRELATED or ANTI-CORRELATED; INSUFFICIENT PRECISION otherwise. Pairwise independence is not
    joint independence, and the panel verdict is read no higher than its estimator."""
    vs = list(pair_verdicts)
    if not vs:
        return INSUFFICIENT_PRECISION
    if any(v in (CORRELATED, ANTI_CORRELATED) for v in vs):
        return REFUTED
    if all(v == INDEPENDENT for v in vs):
        return SUPPORTED
    return INSUFFICIENT_PRECISION


# --------------------------------------------------------------------------------------------- P18
def p18_classification(correct: int, total: int, registered_min_fraction: float,
                       rule_frozen_before_curves: bool) -> str:
    if not rule_frozen_before_curves:
        return NOT_EVALUABLE
    if total <= 0:
        return INCONCLUSIVE
    lo, hi = wilson_interval(correct, total)
    if lo > registered_min_fraction:
        return SUPPORTED
    if hi < registered_min_fraction:
        return REFUTED
    return INCONCLUSIVE


# --------------------------------------------------------------------------------------------- P19
def p19_zero_scaling(alignment_exponent: Interval, margin: float = 0.10) -> str:
    return region_verdict(alignment_exponent, 0.0, margin, MATERIALLY_POSITIVE, MATERIALLY_NEGATIVE, EQUIVALENT_TO_ZERO)


def p19_keep_pace(alignment_exponent: Interval, threshold: float = 0.5) -> str:
    """The historical one-half threshold, kept as a labelled legacy reference. It is not a keep-pace
    verdict: keeping pace is decided against the measured drift exponent (p19_keep_pace_relative)."""
    if is_point(alignment_exponent):
        return NOT_EVALUABLE
    if wholly_above(alignment_exponent, threshold):
        return WHOLLY_ABOVE_ONE_HALF
    if wholly_below(alignment_exponent, threshold):
        return WHOLLY_BELOW_ONE_HALF
    return INCONCLUSIVE


def p19_keep_pace_relative(correction_minus_drift: Interval, margin: float = 0.0) -> str:
    """Keeping pace, commensurably: the correction exponent against the independently measured drift
    exponent on the same scale and clock. A correction exponent of 0.3 out-scales drift at 0.1; one of
    0.6 lags drift at 0.8; the one-half threshold answers a different question."""
    if is_point(correction_minus_drift):
        return NOT_EVALUABLE
    if wholly_above(correction_minus_drift, margin):
        return OUT_SCALES_DRIFT
    if wholly_below(correction_minus_drift, -margin):
        return LAGS_DRIFT
    return INCONCLUSIVE


def p19_joint(alignment_exponent: Interval, margin: float = 0.10, threshold: float = 0.5) -> str:
    """The joint cell: materially positive but wholly below the legacy threshold refutes the flatness
    wording and is not support for it; whether the mechanism keeps pace is a separate, commensurable
    comparison."""
    z = p19_zero_scaling(alignment_exponent, margin)
    if z == NOT_EVALUABLE:
        return NOT_EVALUABLE
    k = p19_keep_pace(alignment_exponent, threshold)
    if z == EQUIVALENT_TO_ZERO:
        return SUPPORTED
    if z == MATERIALLY_POSITIVE and k == WHOLLY_BELOW_ONE_HALF:
        return REFUTES_WORDING_NOT_SUPPORT
    if z in (MATERIALLY_POSITIVE, MATERIALLY_NEGATIVE):
        return REFUTED
    return INCONCLUSIVE


# --------------------------------------------------------------------------------------------- P20
def p20_form(gamma_values: Sequence[float], held_out_advantage: Dict[str, Interval],
             margin: float, admissibility_half_width: float = 0.10, centre: float = 0.5,
             discriminating_power: Optional[float] = None, registered_power: float = 0.80,
             nested: Optional[Set[str]] = None) -> str:
    """The frontier takes the restricted reciprocal form against its rivals.

    The gate is prospective discriminating power: where the caller supplies the calibrated probability
    that the rival predictions separate relative to uncertainty over the realised cells
    (precision.form_discriminating_power), the outcome is NOT DISCRIMINATING below the registered
    power, whatever the cells' position relative to one half (values wholly below one half can
    suffice: at 0.2 the two reciprocals predict 1.25 and 5). Where no power is supplied, the
    first-edition geometric gate applies as the fallback: cells beyond the admissibility band on both
    sides of one half. Then, for a non-nested rival, the restricted form must beat it by the margin;
    a non-nested rival that beats the restricted form refutes. For a nested generalisation (its name
    in `nested`) the rule is practical restriction: the restricted form needs only predictive
    non-inferiority (its advantage not wholly below minus the margin), because when the extra terms
    vanish the two predict identically; a nested rival wholly better than the restricted form by the
    margin refutes, since its extra terms are then identified as non-zero. A tie among nested models
    is INCONCLUSIVE, never a loss for the correct restriction.
    """
    if discriminating_power is None:
        return NOT_EVALUABLE            # the gate is calibrated discriminating power; there is no geometric substitute
    if discriminating_power < registered_power:
        return NOT_DISCRIMINATING
    if not held_out_advantage:
        return INCONCLUSIVE
    nested = set(nested or ())
    if any(is_point(iv) for iv in held_out_advantage.values()):
        return NOT_EVALUABLE
    for name, iv in held_out_advantage.items():
        if name in nested:
            if wholly_below(iv, -margin):
                return REFUTED
        elif wholly_below(iv, 0.0):
            return REFUTED
    ok = True
    for name, iv in held_out_advantage.items():
        if name in nested:
            if wholly_below(iv, -margin):
                ok = False
        elif not wholly_above(iv, margin):
            ok = False
    return SUPPORTED if ok else INCONCLUSIVE


# --------------------------------------------------------------------------------------------- P21
def p21_interaction(interaction: Interval, tolerance: float, sensitivity_demonstrated: bool,
                    level_difference_only: bool = False, high_depth_effect: Optional[Interval] = None) -> str:
    """In-loop correction pulls away with depth: the placement-by-depth interaction, not a constant
    advantage, together with a positive high-depth simple effect (the in-loop regime must be better at
    high depth, not merely deteriorate less from an inferior level). The registered-size sensitivity is a
    gate on every direction: without it the instrument failed, whatever sign the estimate shows. A
    constant advantage is a level difference and is scored against. A change in the path elasticity does
    not identify the direct-depth term (conversion.path_elasticity)."""
    if not sensitivity_demonstrated:
        return INSTRUMENT_FAILED
    if level_difference_only:
        return REFUTED
    if is_point(interaction):
        return NOT_EVALUABLE
    if wholly_below(interaction, -tolerance) or wholly_inside(interaction, -tolerance, tolerance):
        return REFUTED
    if wholly_above(interaction, tolerance):
        if high_depth_effect is None:
            return INCONCLUSIVE           # the positive high-depth simple effect is a required component
        if is_point(high_depth_effect):
            return NOT_EVALUABLE
        if wholly_below(high_depth_effect, 0.0):
            return REFUTED
        if wholly_above(high_depth_effect, 0.0):
            return SUPPORTED
        return INCONCLUSIVE
    return INCONCLUSIVE


# ---------------------------------------------------------------------------------------------- P7
def p7_census(fixed_fraction_by_technique: Interval, fixed_fraction_by_deployment: Interval) -> str:
    """Fixed-strength mechanisms predominate in the dated census, on both registered counts (distinct
    techniques; deployment-weighted), with mixed and ambiguous cases kept in the full denominator.
    SUPPORTED where the fixed class's fraction lies wholly above one half on both counts; REFUTED where
    it lies wholly below one half on both (the scaling class predominates); INCONCLUSIVE otherwise,
    including a contact, a split between the counts, or dominant mixed cases."""
    a, b = fixed_fraction_by_technique, fixed_fraction_by_deployment
    if is_point(a) or is_point(b):
        return NOT_EVALUABLE
    if wholly_above(a, 0.5) and wholly_above(b, 0.5):
        return SUPPORTED
    if wholly_below(a, 0.5) and wholly_below(b, 0.5):
        return REFUTED
    return INCONCLUSIVE


# --------------------------------------------------------------------------------------------- P22
def p22_typed_ordering(checkable_minus_judged: Interval, delta: float = 0.10) -> str:
    """The typed ordering with a material margin. CHECKABLE HIGHER wholly above +delta; JUDGED HIGHER
    wholly below -delta; PRACTICALLY EQUAL wholly inside (-delta, +delta); INCONCLUSIVE otherwise. The
    interval [-0.15, -0.05] at delta 0.10 is INCONCLUSIVE, not a reversal: it straddles -delta. Both
    reversal and practical equality contradict the strict ordering and are reported separately; both
    need the fresh-data repetition before the proposition-level label is issued."""
    return region_verdict(checkable_minus_judged, 0.0, delta, CHECKABLE_HIGHER, JUDGED_HIGHER, PRACTICALLY_EQUAL)


# ---------------------------------------------------------------------------------- ratio objects
def eclipse_ratio(numerator: Interval, denominator: Interval, margin: float = 0.10) -> Tuple[str, Optional[Interval]]:
    """A ratio of correction exponents (the corrector-class ratio) with a positive-denominator
    identifiability rule and interval propagation. Where the denominator interval is not wholly above
    zero the ratio is NOT EVALUABLE: near a zero denominator its uncertainty is unbounded and a large
    headline ratio is an accounting artefact. Otherwise the ratio interval follows from the endpoint
    quotients and is read against one with the registered margin: NEAR ONE, ABOVE ONE, BELOW ONE or
    INCONCLUSIVE. A ratio near one says the exponents are similar on the tested axis; it does not say
    the correction levels are equal or that class is irrelevant."""
    nlo, nhi = _check(numerator); dlo, dhi = _check(denominator)
    if not wholly_above(denominator, 0.0):
        return NOT_EVALUABLE, None
    cands = [nlo / dlo, nlo / dhi, nhi / dlo, nhi / dhi]
    iv = (min(cands), max(cands))
    if iv[0] == iv[1]:
        return NOT_EVALUABLE, iv
    return region_verdict(iv, 1.0, margin, ABOVE_ONE, BELOW_ONE, NEAR_ONE), iv


# ------------------------------------------------------------------------------------ conjunctions
def conjunction(components: Sequence[str], shared_admissibility: bool = True) -> str:
    """The aggregate of a registered conjunction from its component verdicts.

    A validly refuted component refutes the conjunction whatever the others show. Otherwise a missing
    or unevaluable required component makes the whole NOT EVALUABLE (the registration's UNTESTED with
    the qualifier), an inconclusive one makes it INCONCLUSIVE, and only all-supported is SUPPORTED.
    Component results are reported beside the aggregate, never suppressed because the aggregate loses.
    Where the components do not share their admissibility gates the aggregate is NOT EVALUABLE."""
    vs = list(components)
    if not shared_admissibility or not vs:
        return NOT_EVALUABLE
    if any(v == REFUTED for v in vs):
        return REFUTED
    if all(v == SUPPORTED for v in vs):
        return SUPPORTED
    if any(v in (NOT_EVALUABLE, INSTRUMENT_FAILED, AWAITING_REPLICATION) for v in vs):
        return NOT_EVALUABLE
    return INCONCLUSIVE


# ---------------------------------------------------------------------------------- shared helpers
def wilson_interval(successes: int, n: int, z: float = 1.96) -> Interval:
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def scoring_table() -> List[Dict[str, str]]:
    """The outcome regions as documentation rows, one per proposition, for generating or checking the
    registration's scoring sheet and every prose summary. Every region is strict: a boundary contact is
    INCONCLUSIVE and a zero-width interval is NOT EVALUABLE."""
    return [
        {"proposition": "P1", "axes": "FORM; RECURSION", "rule": "FORM: power family wins a majority of estimable cells, only where the demonstrated false-selection rate is at or below the registered maximum, else NOT DISCRIMINATING; a cell won by the shifted finite-window solution is credited to P5, never to the power family. RECURSION: depth exponent minus breadth exponent wholly above the registered margin."},
        {"proposition": "P2", "axes": "shape", "rule": "INTERIOR MAXIMUM beats both endpoints beyond k standard errors; MONOTONE, FLAT, MULTIMODAL and UNRESOLVED are distinct outcomes; endpoints are never interior; a failed unimodality test proves no rival shape."},
        {"proposition": "P3", "axes": "trend frontier", "rule": "REFUTED on replicated exceedance with the balance independently positive; SUPPORTED only with P20 passed and the paired difference wholly within 0.10; CONSISTENT, NOT SUPPORTIVE where no violation and no location; NOT EVALUABLE without the service elasticity. The object is the trend frontier, never a finite-slack window."},
        {"proposition": "P4", "axes": "regime; response", "rule": "correction exponent wholly above drift exponent: correction advantage; wholly below: burden advantage; equality undecided by exponents; the scored response is the separately calibrated model's held-out prediction."},
        {"proposition": "P5", "axes": "agreement", "rule": "predicted minus fitted finite-window exponent wholly within 0.10 supports; wholly outside refutes; sensitivity demonstrated first; identification argument before the unit leaves draft."},
        {"proposition": "P6", "axes": "survey", "rule": "REFUTED on a replicated interval wholly above one half (AWAITING FRESH-DATA REPLICATION before then); NO COUNTEREXAMPLE FOUND AMONG N where every interval is wholly below; INCONCLUSIVE where any straddles or touches; NO SURVEY for an empty roster; no supported verdict exists; detection probability reported."},
        {"proposition": "P7", "axes": "census", "rule": "fixed class wholly above one half on both counts supports; wholly below on both refutes; a contact, a split between the counts or dominant mixed cases is INCONCLUSIVE; mixed cases stay in the full denominator."},
        {"proposition": "P9", "axes": "dimension", "rule": "the measured exponent wholly within the margin of the prospectively assigned dimensional prediction supports; wholly outside refutes; an unwritten assignment rule is NOT EVALUABLE."},
        {"proposition": "P8", "axes": "departure; clears unity", "rule": "region (0.40, 0.60): EQUIVALENT TO NULL strictly inside, ABOVE NULL wholly above 0.60, BELOW NULL wholly below 0.40, INCONCLUSIVE straddling or touching. CLEARS UNITY wholly above 1.0."},
        {"proposition": "P10", "axes": "material fraction", "rule": "fraction not surviving wholly above ten per cent supports; wholly below refutes; refutation does not restore the earlier scores; support challenges only the affected scores."},
        {"proposition": "P11", "axes": "pair", "rule": "difference wholly within the margin refutes distinctness (interchangeable at that resolution, not identical); wholly outside supports it."},
        {"proposition": "P12", "axes": "endpoint", "rule": "change wholly above +0.10 or wholly below -0.10 supports (undirected); wholly within refutes when the ordering was delivered; not delivered is INSTRUMENT FAILED."},
        {"proposition": "P13", "axes": "performance; dependence (one conjunction)", "rule": "performance wholly above zero and the dependence contrast supported: SUPPORTED; either component refuted: REFUTED; a missing component: NOT EVALUABLE; an unresolved one: INCONCLUSIVE, with the cell named (performance supported, mechanism unresolved is never full support)."},
        {"proposition": "P14", "axes": "blinded shrink", "rule": "blinded minus unblinded gain wholly below zero supports; wholly above zero refutes; the sign is measured, not assumed; refutation leaves the earlier scores standing."},
        {"proposition": "P15", "axes": "decay", "rule": "fraction retained wholly below one half supports; wholly above refutes; the threshold is a convention; an unidentified denominator is NOT EVALUABLE."},
        {"proposition": "P16", "axes": "typed endpoint", "rule": "the endpoint is named: a margin reversal is read from the balance elasticity; a service deficit, backlog threshold or conformance failure needs a registered response model or is NOT EVALUABLE; SUPPORTED on entry into the forecast region, the event inside the sealed window, controls distinguishing, replicated; REFUTED on no event past the informative horizon or an event outside the window, replicated."},
        {"proposition": "P17", "axes": "pairwise dependence", "rule": "log ratio wholly within 0.20 is INDEPENDENT; wholly above CORRELATED; wholly below ANTI-CORRELATED; else INSUFFICIENT PRECISION; the panel is read no higher than its pairs; no ceiling or exponent is located."},
        {"proposition": "P18", "axes": "classification", "rule": "Wilson interval on correct classifications wholly above the registered fraction supports; rule frozen before any curve or NOT EVALUABLE."},
        {"proposition": "P19", "axes": "zero-scaling; legacy half; relative keep-pace", "rule": "region (-0.10, 0.10) on the alignment elasticity; materially positive but wholly below one half refutes the wording and is not support; the legacy half threshold is labelled; keeping pace is decided against the measured drift exponent."},
        {"proposition": "P20", "axes": "form", "rule": "NOT DISCRIMINATING below the registered discriminating power over the realised cells (values wholly below one half can suffice); SUPPORTED where the restricted form beats every non-nested rival by the margin and is non-inferior to every nested generalisation; a non-nested rival win, or a nested generalisation better by the margin, refutes; a nested tie is INCONCLUSIVE."},
        {"proposition": "P21", "axes": "interaction; high-depth simple effect", "rule": "the registered-size sensitivity is a gate on every direction (INSTRUMENT FAILED without it); interaction wholly above the tolerance and a positive high-depth simple effect: SUPPORTED; interaction wholly within or below the tolerance, or a resolved negative high-depth effect: REFUTED; a constant advantage is scored against; the direct-depth term is not identified by the path elasticity."},
        {"proposition": "P22", "axes": "typed ordering", "rule": "region (-0.10, 0.10): CHECKABLE HIGHER wholly above, JUDGED HIGHER wholly below, PRACTICALLY EQUAL wholly inside, INCONCLUSIVE straddling or touching; both contradictions need the fresh-data repetition."},
        {"proposition": "ratio", "axes": "corrector-class ratio", "rule": "NOT EVALUABLE unless the denominator interval is wholly above zero; then the endpoint-quotient interval read against one with the registered margin."},
    ]
