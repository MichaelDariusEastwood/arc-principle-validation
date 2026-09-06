"""P16's support branch, built from the components the design promises, each carrying its uncertainty.

WHY THIS FILE EXISTS. Finding A2: the reference support branch counted reversals in the
above-boundary arms, counted non-reversals in the controls, checked that the fitted line was falling
and that its zero sat within a tolerance of the sealed zero, and declared support. Four things the
design registers were sealed, reported or promised and never enforced.

  timing            `timing_tolerance` was written into the sealed object and no line of code ever
                    read it back. A reversal at any round in the horizon counted the same as one at
                    the round the switch predicts.
  slope magnitude   the sealed slope was printed beside the fitted slope and compared with nothing.
                    The branch asked only whether the line was FALLING, so a line an order of
                    magnitude too shallow or too steep supported the proposition exactly as well as
                    the registered one.
  location          the fitted zero was compared with the sealed zero by a point distance. Neither
                    the uncertainty of the fitted zero nor the calibration uncertainty of the
                    boundary the seal names entered that comparison, so a line fitted from three
                    noisy arms and a line fitted from thirty clean ones were read identically.
  exposure          the arm's assigned `alpha_arm` was both the dose the apparatus was asked to
                    deliver and the x coordinate the line was fitted against. A titration whose lever
                    fails, saturates or overshoots then reports a line drawn against intentions.

And two the design promises that were absent altogether: DELIVERY, the apparatus's own record that
the dose was administered, and REPETITION, the fresh-data replication the registered rule
(`arc_instruments.verdicts.p16_endpoint`) requires before any single run reads as support.

WHAT A COMPONENT IS. One named requirement, in one of four states, with the reason it is in that
state and the numbers it was decided from. SATISFIED and FAILED are decisions; UNRESOLVED means the
measurement was made and its uncertainty does not decide; NOT SUPPLIED means the run carries no
evidence on that requirement at all, which is a different statement and must never be read as either
decision. Every state carries an interval wherever the requirement is a comparison of quantities,
because the whole of this finding is that a comparison without its uncertainty is not a check.

WHAT THE WRAPPER DOES. It combines them and refuses. A proposition result is withheld while any
contract-required component is NOT SUPPLIED; a component that FAILED withholds support whatever else
holds; and a single run is PROVISIONAL in every direction, reported as AWAITING FRESH-DATA
REPLICATION, which is the registered label for everything met but not yet repeated on fresh data.
Several alarms in several arms never make complete support on their own, which is the sentence
finding A2 ends on.

THE STRICT-CLEARANCE CONVENTION IS THE KIT'S. Ruling 17: a boundary contact is not a clearance, in
every direction. The interval comparisons here go through `arc_instruments.verdicts.wholly_inside`,
`wholly_above` and `wholly_below` so that this module cannot drift from the rest of the kit.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from arc_instruments import verdicts as V

from . import p16_sequential as SEQ
from .p5 import t95

# --------------------------------------------------------------------------------------------------
# The vocabulary
# --------------------------------------------------------------------------------------------------

SATISFIED = "SATISFIED"
FAILED = "FAILED"
UNRESOLVED = "UNRESOLVED"
NOT_SUPPLIED = "NOT SUPPLIED"
STATES = (SATISFIED, FAILED, UNRESOLVED, NOT_SUPPLIED)

DELIVERY = "delivery"
REALISED_EXPOSURE = "realised-exposure"
EVENT = "event"
TIMING = "timing"
LOCATION = "location"
SLOPE_MAGNITUDE = "slope-magnitude"
CONTROLS = "controls"
DISCRIMINATION = "discrimination"
REPETITION = "repetition"

# The order is the order the design note states them in, and it is the order they are reported in.
CONTRACT_COMPONENTS = (DELIVERY, REALISED_EXPOSURE, EVENT, TIMING, LOCATION, SLOPE_MAGNITUDE,
                       CONTROLS, DISCRIMINATION, REPETITION)

PROVISIONAL_LABEL = ("PROVISIONAL: this is a single run. The registered rule reads a result at "
                     "proposition level only after a fresh-data repetition has met the same rule, so "
                     "nothing here is a completed P16 finding.")


@dataclass(frozen=True)
class Component:
    """One named requirement of the support branch, its state, its reason and its numbers."""

    name: str
    state: str
    reason: str
    detail: Dict[str, Any] = field(default_factory=dict)
    required: bool = True

    def __post_init__(self) -> None:
        if self.state not in STATES:
            raise ValueError("unknown component state %r" % (self.state,))

    def as_dict(self) -> Dict[str, Any]:
        return {"component": self.name, "state": self.state, "reason": self.reason,
                "required": bool(self.required), "detail": self.detail}


# --------------------------------------------------------------------------------------------------
# The line, fitted with its covariance
# --------------------------------------------------------------------------------------------------

def fit_line(xs: Sequence[float], ys: Sequence[float]) -> Dict[str, Any]:
    """The balance line across arms, with the covariance the zero's uncertainty needs.

    `np.polyfit` returns the two coefficients and nothing else, which is why the reference branch
    could only ever compare points. The zero is a ratio of the two coefficients, so its variance
    needs their covariance and not merely their variances: the delta method term in the covariance
    is the one that dominates when the slope is small, which is exactly the case where a fitted zero
    is least trustworthy and the reference branch treated it as exact.
    """
    x = np.asarray(list(xs), float)
    y = np.asarray(list(ys), float)
    n = len(x)
    out: Dict[str, Any] = {"n": int(n), "slope": float("nan"), "intercept": float("nan"),
                           "se_slope": float("nan"), "se_intercept": float("nan"), "cov": float("nan"),
                           "zero": float("nan"), "se_zero": float("nan"), "df": max(n - 2, 0)}
    if n < 3 or float(np.sum((x - x.mean()) ** 2)) <= 0:
        out["reason"] = "fewer than three arms with distinct exposures, so no line is fitted"
        return out
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = float(coef[0]), float(coef[1])
    resid = y - A @ coef
    dof = n - 2
    sigma2 = float(np.sum(resid ** 2)) / dof if dof > 0 else 0.0
    try:
        xtx_inv = np.linalg.inv(A.T @ A)
    except np.linalg.LinAlgError:
        out["reason"] = "the arm exposures are collinear, so the line is not identified"
        return out
    cov = sigma2 * xtx_inv
    var_m, var_b, cov_mb = float(cov[0, 0]), float(cov[1, 1]), float(cov[0, 1])
    out.update({"slope": slope, "intercept": intercept, "se_slope": float(np.sqrt(max(var_m, 0.0))),
                "se_intercept": float(np.sqrt(max(var_b, 0.0))), "cov": cov_mb, "df": int(dof)})
    if slope == 0:
        out["reason"] = "the fitted line is flat, so it has no zero"
        return out
    zero = -intercept / slope
    # z = -b/m: dz/db = -1/m, dz/dm = b/m^2
    dz_db = -1.0 / slope
    dz_dm = intercept / (slope ** 2)
    var_z = dz_db ** 2 * var_b + dz_dm ** 2 * var_m + 2.0 * dz_db * dz_dm * cov_mb
    out.update({"zero": float(zero), "se_zero": float(np.sqrt(max(var_z, 0.0))), "reason": ""})
    return out


def _interval(estimate: float, se: float, df: int) -> Optional[Tuple[float, float]]:
    """The two-sided interval the kit's equivalence rules are read on, or None when there is none.

    A zero-width interval is refused upstream by every region function in `arc_instruments.verdicts`
    (an exact equality is a logical statement and not a measurement), so an estimate with no
    uncertainty returns None here and its component is UNRESOLVED rather than decided.
    """
    if not np.isfinite(estimate) or not np.isfinite(se) or se < 0:
        return None
    if se == 0:
        # A ZERO STANDARD ERROR IS A PERFECT MEASUREMENT, NOT A MISSING ONE, and this package already
        # says so where it matters: `observation._ols` records that a zero-residual window is a
        # perfectly measured one whose interval is a point, and finding A1's repair was precisely to
        # stop the detection rule requiring a strictly positive standard error before it would
        # declare. The same reading is taken here. It bites only where a quantity is measured without
        # error at all, which in practice means a simulated world; every comparison against a sealed
        # quantity here adds that quantity's calibration uncertainty in quadrature, so a registered
        # calibration keeps those intervals wide whatever the fit does.
        return (float(estimate), float(estimate))
    half = t95(max(int(df), 1)) * float(se)
    if not np.isfinite(half) or half <= 0:
        return None
    return (float(estimate) - half, float(estimate) + half)


def _equivalence(diff: float, se: float, df: int, margin: float, what: str,
                 detail: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    """A difference against a registered equivalence band, with strict clearance in both directions."""
    iv = _interval(diff, se, df)
    detail = dict(detail)
    detail.update({"difference": float(diff) if np.isfinite(diff) else None,
                   "standard_error": float(se) if np.isfinite(se) else None,
                   "interval": list(iv) if iv else None, "equivalence_margin": float(margin),
                   "degrees_of_freedom": int(df)})
    if iv is None:
        return (UNRESOLVED,
                "%s could not be given an interval (its standard error is zero or not finite), and a "
                "point comparison is the defect this component exists to remove" % what, detail)
    if V.wholly_inside(iv, -margin, margin):
        return SATISFIED, "%s lies wholly inside the registered equivalence band" % what, detail
    if V.wholly_above(iv, margin) or V.wholly_below(iv, -margin):
        return FAILED, "%s lies wholly outside the registered equivalence band" % what, detail
    return UNRESOLVED, "%s neither clears nor falls outside the band; the comparison does not decide" % what, detail


# --------------------------------------------------------------------------------------------------
# The nine components
# --------------------------------------------------------------------------------------------------

def _is_dose(arm: Dict[str, Any]) -> bool:
    return str(arm.get("arm", "")).startswith("dose")


def _is_control(arm: Dict[str, Any]) -> bool:
    return arm.get("arm") in ("sham", "baseline")


def delivery_component(arms: Sequence[Dict[str, Any]], switch_round: int) -> Component:
    """Was the dose administered, on the apparatus's own record, and at the sealed switch?

    Not an inference from the numbers: an arm whose dose never landed and an arm whose dose landed
    and did nothing produce the same series, and only the apparatus can tell them apart. An
    unattested arm is NOT SUPPLIED, never a defaulted yes.
    """
    detail: Dict[str, Any] = {"switch_round_sealed": int(switch_round), "arms": []}
    unattested, wrong = [], []
    for a in arms:
        d = a.get("delivery")
        row = {"arm": a.get("arm"), "alpha_assigned": a.get("alpha"), "record": d}
        detail["arms"].append(row)
        if not isinstance(d, dict) or not d.get("attested"):
            unattested.append(a.get("arm"))
            continue
        first = d.get("first_applied_round")
        if _is_control(a):
            # a control that was dosed is not a control
            if d.get("n_applied"):
                wrong.append("%s was dosed at round %s and is a control" % (a.get("arm"), first))
        else:
            if not d.get("n_applied"):
                wrong.append("%s records no round at which its dose was administered" % a.get("arm"))
            elif first is None or int(first) != int(switch_round):
                wrong.append("%s was first dosed at round %s and the sealed switch is round %d"
                             % (a.get("arm"), first, switch_round))
    if unattested:
        return Component(DELIVERY, NOT_SUPPLIED,
                         "no delivery record from %d of %d arms (%s): the apparatus did not attest "
                         "that the assigned dose was administered, and an unattested arm cannot be "
                         "told apart from an arm whose dose silently failed"
                         % (len(unattested), len(arms), ", ".join(sorted(set(map(str, unattested))))),
                         detail)
    if wrong:
        return Component(DELIVERY, FAILED, "; ".join(wrong), detail)
    return Component(DELIVERY, SATISFIED,
                     "every arm attests that its assigned manipulation was administered at the "
                     "sealed switch round, and no control was dosed", detail)


def realised_exposure_component(arms: Sequence[Dict[str, Any]], sealed_zero: float) -> Component:
    """Did each dose arm actually reach the side of the boundary its assignment put it on?

    The acceptance case finding A2 names: an arm assigned above the boundary whose realised growth
    exponent is measured below it. The reference branch could not see that, because the assigned dose
    was the only exposure it ever held.
    """
    rows: List[Dict[str, Any]] = []
    unmeasured, wrong, straddling = [], [], []
    for a in arms:
        if not _is_dose(a):
            continue
        re_ = a.get("realised_exposure")
        assigned = float(a.get("alpha", float("nan")))
        side = "above" if assigned > sealed_zero else "below"
        row: Dict[str, Any] = {"arm": a.get("arm"), "alpha_assigned": assigned,
                               "assigned_side": side, "realised": re_}
        rows.append(row)
        if not isinstance(re_, dict) or not re_.get("measured"):
            unmeasured.append(a.get("arm"))
            continue
        iv = _interval(float(re_.get("alpha", float("nan"))), float(re_.get("se", float("nan"))),
                       int(re_.get("n", 2)) - 2)
        row["interval"] = list(iv) if iv else None
        if iv is None:
            straddling.append("%s has no interval on its realised exponent" % a.get("arm"))
            continue
        if side == "above":
            if V.wholly_above(iv, sealed_zero):
                row["verdict"] = "above the boundary, as assigned"
            elif V.wholly_below(iv, sealed_zero):
                wrong.append("%s was assigned above the boundary and realised %.3f, wholly below it"
                             % (a.get("arm"), float(re_["alpha"])))
            else:
                straddling.append("%s straddles the boundary" % a.get("arm"))
        else:
            if V.wholly_below(iv, sealed_zero):
                row["verdict"] = "below the boundary, as assigned"
            elif V.wholly_above(iv, sealed_zero):
                wrong.append("%s was assigned below the boundary and realised %.3f, wholly above it"
                             % (a.get("arm"), float(re_["alpha"])))
            else:
                straddling.append("%s straddles the boundary" % a.get("arm"))
    detail = {"sealed_zero": float(sealed_zero), "arms": rows}
    if unmeasured:
        return Component(REALISED_EXPOSURE, NOT_SUPPLIED,
                         "%d of %d dose arms measured no realised growth exponent (%s). The assigned "
                         "dose is not a measurement of exposure and this runner will not substitute "
                         "it for one" % (len(unmeasured), len(rows),
                                         ", ".join(sorted(set(map(str, unmeasured))))),
                         detail)
    if wrong:
        return Component(REALISED_EXPOSURE, FAILED, "; ".join(wrong), detail)
    if straddling:
        return Component(REALISED_EXPOSURE, UNRESOLVED, "; ".join(straddling), detail)
    return Component(REALISED_EXPOSURE, SATISFIED,
                     "every dose arm's measured realised exponent lies wholly on the side of the "
                     "sealed boundary its assignment put it on", detail)


def _wilson(k: int, n: int) -> Tuple[float, float]:
    return V.wilson_interval(int(k), int(n))


def event_component(arms: Sequence[Dict[str, Any]], sealed_zero: float,
                    alarm_rate_null: float,
                    refutation: Optional[Dict[str, Any]] = None) -> Component:
    """Did the registered event occur in the arms predicted to show it, more often than the null rate?

    The rate is read against the calibrated per-arm false-alarm rate of the detection rule rather
    than against a bare majority, so that the component says how far above chance the alarms are and
    not merely that there were several of them.

    AND AN ABSENCE OF ALARMS FAILS THIS COMPONENT ONLY WHERE THE ABSENCE WAS MEASURED. Finding A3:
    this component used to reach FAILED from a low alarm rate alone, and a FAILED component is a
    refutation in the wrapper, so a set of arms too noisy, too short or too few to decide anything
    refuted the proposition exactly as a set of arms whose margins were measured firmly positive.
    They are different findings. FAILED now needs the refutation record's demonstrated positivity in
    a majority of these arms, each beyond the registered informative horizon; a low rate without it
    is UNRESOLVED and says which silence it was, and the record is passed in rather than recomputed
    here so that one rule decides it for the run, the components and the bundle alike.
    """
    above = [a for a in arms if _is_dose(a) and float(a.get("alpha", 0.0)) > sealed_zero]
    n = len(above)
    k = sum(1 for a in above if a.get("declared_round") is not None)
    iv = _wilson(k, n) if n else (0.0, 1.0)
    detail = {"n_above_boundary_arms": n, "n_declared": k, "rate_interval": list(iv),
              "null_alarm_rate": float(alarm_rate_null),
              "non_alarm": (refutation or {}).get("counts"),
              "n_demonstrated_positive": (refutation or {}).get("n_demonstrated_positive"),
              "demonstrated_refutation": bool((refutation or {}).get("refutes")),
              "arms": [{"arm": a.get("arm"), "alpha": a.get("alpha"),
                        "declared_round": a.get("declared_round")} for a in above]}
    if n == 0:
        return Component(EVENT, NOT_SUPPLIED, "no arm was placed above the sealed boundary", detail)
    if V.wholly_above(iv, alarm_rate_null):
        return Component(EVENT, SATISFIED,
                         "the reversal rate in the above-boundary arms (%d of %d) clears the "
                         "calibrated per-arm false-alarm rate of %.3f" % (k, n, alarm_rate_null), detail)
    if refutation is not None and refutation.get("refutes"):
        return Component(EVENT, FAILED,
                         "the event did not occur in the above-boundary arms (%d of %d declared) and "
                         "its absence is demonstrated rather than merely observed: %s"
                         % (k, n, refutation.get("reason", "")), detail)
    if V.wholly_below(iv, alarm_rate_null):
        return Component(EVENT, UNRESOLVED,
                         "the reversal rate in the above-boundary arms (%d of %d) lies wholly below "
                         "the calibrated false-alarm rate, and the absence is not demonstrated: %s"
                         % (k, n, (refutation or {}).get("reason",
                                                         "no non-alarm record accompanies these arms")),
                         detail)
    return Component(EVENT, UNRESOLVED,
                     "the reversal rate in the above-boundary arms (%d of %d) is consistent with the "
                     "calibrated false-alarm rate of %.3f, so the arms do not establish the event"
                     % (k, n, alarm_rate_null), detail)


def timing_component(arms: Sequence[Dict[str, Any]], window: Optional[Sequence[float]],
                     sealed_zero: float) -> Component:
    """Did the change happen when the seal says it would, within the sealed tolerance?

    The quantity compared is the CHANGE POINT and not the declaration round. The declaration round is
    a detection latency: it moves with the noise, the horizon and the threshold, so on the reference
    configuration a true switch at round 8 is declared somewhere in the high twenties and no
    tolerance around the switch could ever contain it. The change point is where the series' shape
    changes, which is the quantity a tolerance on the switch is a tolerance on.
    """
    above = [a for a in arms if _is_dose(a) and float(a.get("alpha", 0.0)) > sealed_zero]
    rows, inside, outside, unmeasured = [], [], [], []
    for a in above:
        cp = a.get("change_point")
        row = {"arm": a.get("arm"), "change_point": cp, "declared_round": a.get("declared_round")}
        rows.append(row)
        if not isinstance(cp, dict) or not cp.get("measured") or cp.get("round") is None:
            unmeasured.append(a.get("arm"))
            continue
        r = int(cp["round"])
        (inside if window and window[0] <= r <= window[1] else outside).append((a.get("arm"), r))
    detail = {"sealed_window": list(window) if window else None, "arms": rows,
              "n_inside": len(inside), "n_outside": len(outside),
              "outside": [{"arm": n_, "change_point": r} for n_, r in outside],
              "note": "the change point is compared, never the declaration round, which is a "
                      "detection latency and not a change point"}
    if window is None:
        return Component(TIMING, NOT_SUPPLIED,
                         "the seal carries no timing window, so the change point has nothing "
                         "registered to be compared with", detail)
    if unmeasured:
        return Component(TIMING, NOT_SUPPLIED,
                         "%d of %d above-boundary arms measured no change point (%s)"
                         % (len(unmeasured), len(above),
                            ", ".join(sorted(set(map(str, unmeasured))))), detail)
    if not above:
        return Component(TIMING, NOT_SUPPLIED, "no arm was placed above the sealed boundary", detail)
    if outside and not inside:
        return Component(TIMING, FAILED,
                         "every above-boundary arm changed outside the sealed window %s (%s)"
                         % (list(window), ", ".join("%s at round %d" % (n_, r) for n_, r in outside)),
                         detail)
    if outside:
        return Component(TIMING, UNRESOLVED,
                         "%d of %d above-boundary arms changed outside the sealed window %s"
                         % (len(outside), len(above), list(window)), detail)
    return Component(TIMING, SATISFIED,
                     "every above-boundary arm's change point falls inside the sealed window %s"
                     % (list(window),), detail)


def location_component(line: Dict[str, Any], sealed_zero: float, tolerance: float,
                       chi_hat: float, chi_hat_se: Optional[float]) -> Component:
    """The fitted zero against the sealed zero, with BOTH uncertainties in the comparison.

    The reference branch took `abs(fitted - sealed) <= tolerance`, which treats a line fitted from
    three noisy arms exactly as it treats one fitted from thirty clean arms, and treats the sealed
    boundary as exactly known. The sealed zero is 1 / (1 - chi) at the calibrated chi, so its
    uncertainty is the calibration's, propagated by the delta method: d(1/(1-chi))/dchi = 1/(1-chi)^2.

    An unregistered calibration uncertainty is NOT SUPPLIED and not a zero. A run that does not know
    how well its boundary was located cannot check its location, and defaulting the unknown to
    perfect knowledge is the point-comparison defect wearing a different name.
    """
    detail: Dict[str, Any] = {"sealed_zero": float(sealed_zero), "location_tolerance": float(tolerance),
                              "chi_hat": float(chi_hat), "chi_hat_se": chi_hat_se,
                              "fitted_zero": line.get("zero"), "se_fitted_zero": line.get("se_zero"),
                              "fitted_on": line.get("fitted_on")}
    if line.get("fitted_on") != "realised-exposure":
        return Component(LOCATION, NOT_SUPPLIED,
                         "the line was fitted against %r rather than against measured realised "
                         "exposures, so its zero is a zero in the assigned doses and not a located "
                         "boundary" % line.get("fitted_on"), detail)
    if not np.isfinite(line.get("zero", float("nan"))):
        return Component(LOCATION, NOT_SUPPLIED,
                         "no line zero was fitted: %s" % (line.get("reason") or "the fit failed"), detail)
    if chi_hat_se is None:
        return Component(LOCATION, NOT_SUPPLIED,
                         "the boundary's calibration uncertainty is not registered (chi_hat_se is "
                         "unset), so the sealed zero has no interval and the only comparison "
                         "available is the point distance this component exists to replace", detail)
    denom = (1.0 - float(chi_hat))
    se_sealed = float(chi_hat_se) / (denom ** 2) if denom != 0 else float("inf")
    se_fit = float(line.get("se_zero", float("nan")))
    detail["se_sealed_zero_from_calibration"] = se_sealed
    diff = float(line["zero"]) - float(sealed_zero)
    se = float(np.sqrt(se_fit ** 2 + se_sealed ** 2)) if np.isfinite(se_fit) else float("nan")
    state, reason, detail = _equivalence(diff, se, int(line.get("df", 1)), float(tolerance),
                                         "the fitted zero's distance from the sealed zero", detail)
    return Component(LOCATION, state, reason, detail)


def slope_component(line: Dict[str, Any], sealed_slope: float, equivalence: Optional[float],
                    chi_hat_se: Optional[float], comparable: bool) -> Component:
    """The fitted slope against the sealed slope, inside a registered equivalence band.

    The reference branch asked only whether the line was FALLING. A line at minus 0.05 against a
    sealed minus 0.5 is falling, and supported the proposition exactly as well as the registered one.
    The magnitude is the prediction, so it is tested as one.

    No equivalence band is registered by the contract for this comparison. An unset band is NOT
    SUPPLIED rather than a number invented here: choosing the width of the band that decides the
    proposition is the author's act, not this module's.
    """
    detail: Dict[str, Any] = {"sealed_slope": float(sealed_slope), "fitted_slope": line.get("slope"),
                              "se_fitted_slope": line.get("se_slope"),
                              "equivalence_margin": equivalence, "chi_hat_se": chi_hat_se,
                              "fitted_on": line.get("fitted_on"), "comparable": bool(comparable)}
    if not comparable:
        return Component(SLOPE_MAGNITUDE, NOT_SUPPLIED,
                         "the observed quantity is not the registered balance elasticity, so the "
                         "fitted slope and the sealed slope are not candidates for the same number "
                         "and no magnitude comparison exists", detail)
    if line.get("fitted_on") != "realised-exposure":
        return Component(SLOPE_MAGNITUDE, NOT_SUPPLIED,
                         "the line was fitted against %r rather than against measured realised "
                         "exposures, so its slope is d Delta / d assigned dose and not d Delta / "
                         "d alpha" % line.get("fitted_on"), detail)
    if not np.isfinite(line.get("slope", float("nan"))):
        return Component(SLOPE_MAGNITUDE, NOT_SUPPLIED,
                         "no line slope was fitted: %s" % (line.get("reason") or "the fit failed"), detail)
    if equivalence is None:
        return Component(SLOPE_MAGNITUDE, NOT_SUPPLIED,
                         "no equivalence margin is registered for the sealed slope, so there is no "
                         "band the fitted slope can be shown to lie inside; the reference branch's "
                         "only test was that the line fell, which any magnitude satisfies", detail)
    if chi_hat_se is None:
        return Component(SLOPE_MAGNITUDE, NOT_SUPPLIED,
                         "the sealed slope is minus (1 minus chi) at the calibrated chi and its "
                         "calibration uncertainty is not registered, so the comparison would treat "
                         "the prediction as exactly known", detail)
    se_fit = float(line.get("se_slope", float("nan")))
    # d(-(1 - chi))/dchi = 1, so the sealed slope's uncertainty IS the calibration's
    se = float(np.sqrt(se_fit ** 2 + float(chi_hat_se) ** 2)) if np.isfinite(se_fit) else float("nan")
    detail["se_sealed_slope_from_calibration"] = float(chi_hat_se)
    diff = float(line["slope"]) - float(sealed_slope)
    state, reason, detail = _equivalence(diff, se, int(line.get("df", 1)), float(equivalence),
                                         "the fitted slope's difference from the sealed slope", detail)
    return Component(SLOPE_MAGNITUDE, state, reason, detail)


def controls_component(arms: Sequence[Dict[str, Any]], alarm_rate_null: float,
                       readings: Optional[Sequence[Dict[str, Any]]] = None) -> Component:
    """The coefficient-only sham and the untouched baseline: no shift.

    SATISFIED needs no control to have declared, which is what "no shift" means, AND each control's
    own silence to have demonstrated it.

    WHY THE SECOND HALF IS THERE. Finding A3's governing rule is that an unobserved event is not an
    absent one, and it does not stop at the dose arms. This component used to reach SATISFIED from
    `k == 0` alone, so a run with no registered informative horizon reported that "neither the sham
    nor the baseline arms shifted" while every one of those same silences classified as CENSORED: the
    event was not observed by the horizon. One run, two contradictory readings of one silence, and
    the generous one decided. The readings are passed in from `arc_runner.p16_sequential` so that one
    rule decides for the run, the components and the bundle alike, exactly as the refutation record
    is passed to the event component.

    `readings` of None keeps the older behaviour, and says so in the reason. A bundle written before
    the sequential rule existed carries no terminal records to read, and a bundle that can no longer
    be re-scored is not evidence; what changes is that such a run no longer claims the controls were
    demonstrated, because nothing in it demonstrated them.

    A minority of control alarms is UNRESOLVED and not a specificity failure, because the detection
    rule's own calibrated per-arm false-alarm rate predicts some: reading one alarm in six controls as
    generic deterioration would refute the instrument with its own registered error rate.
    """
    controls = [a for a in arms if _is_control(a)]
    n = len(controls)
    k = sum(1 for a in controls if a.get("declared_round") is not None)
    iv = _wilson(k, n) if n else (0.0, 1.0)
    rows = list(readings) if readings is not None else None
    silent = [r for r in (rows or []) if r.get("state") != SEQ.EVENT_DECLARED]
    undemonstrated = [r for r in silent if not r.get("demonstrated")]
    detail = {"n_controls": n, "n_declared": k, "rate_interval": list(iv),
              "null_alarm_rate": float(alarm_rate_null),
              "non_alarm": (None if rows is None else
                            {"census": dict(collections.Counter(r.get("state") for r in silent)),
                             "n_demonstrated": len(silent) - len(undemonstrated),
                             "n_not_demonstrated": len(undemonstrated)}),
              "arms": [{"arm": a.get("arm"), "declared_round": a.get("declared_round")} for a in controls]}
    if n == 0:
        return Component(CONTROLS, NOT_SUPPLIED, "the run carries no sham and no baseline arm", detail)
    if k == 0:
        if rows is None:
            return Component(CONTROLS, SATISFIED,
                             "neither the sham nor the baseline arms shifted (%d controls, no "
                             "alarm), read from the alarm count alone because this run carries no "
                             "four-way reading of their silences" % n, detail)
        if undemonstrated:
            return Component(CONTROLS, UNRESOLVED,
                             "no control declared, but %d of %d control silences demonstrate nothing "
                             "(%s), and an unobserved event is not an absent one: the controls were "
                             "watched and their steadiness was not measured"
                             % (len(undemonstrated), len(silent),
                                "; ".join(sorted(set(str(r.get("state")) for r in undemonstrated)))),
                             detail)
        return Component(CONTROLS, SATISFIED,
                         "neither the sham nor the baseline arms shifted (%d controls, no alarm) and "
                         "every control's silence demonstrates it: their margins were measured past "
                         "the informative horizon" % n, detail)
    if V.wholly_above(iv, alarm_rate_null):
        return Component(CONTROLS, FAILED,
                         "%d of %d control arms shifted, a rate wholly above the calibrated per-arm "
                         "false-alarm rate: the shift is not specific to the dose" % (k, n), detail)
    return Component(CONTROLS, UNRESOLVED,
                     "%d of %d control arms shifted, a rate the calibrated false-alarm rate of %.3f "
                     "does not exclude" % (k, n, alarm_rate_null), detail)


def discrimination_component(arms: Sequence[Dict[str, Any]], sealed_zero: float) -> Component:
    """Does the pattern separate the boundary account from generic deterioration?

    The two rates compared are the above-boundary arms against everything else, being the
    below-boundary arms and both controls. Strict non-overlap of the two Wilson intervals is the
    clearance, so a run in which every arm reverses (deterioration everywhere) cannot discriminate
    however many alarms it produced.
    """
    above = [a for a in arms if _is_dose(a) and float(a.get("alpha", 0.0)) > sealed_zero]
    rest = [a for a in arms if not any(a is b for b in above)]
    if not above or not rest:
        return Component(DISCRIMINATION, NOT_SUPPLIED,
                         "the run does not carry both above-boundary arms and comparison arms",
                         {"n_above": len(above), "n_rest": len(rest)})
    ka = sum(1 for a in above if a.get("declared_round") is not None)
    kr = sum(1 for a in rest if a.get("declared_round") is not None)
    ia, ir = _wilson(ka, len(above)), _wilson(kr, len(rest))
    detail = {"above": {"declared": ka, "n": len(above), "interval": list(ia)},
              "not_above": {"declared": kr, "n": len(rest), "interval": list(ir)}}
    if ia[0] > ir[1]:
        return Component(DISCRIMINATION, SATISFIED,
                         "the above-boundary reversal rate (%d of %d) clears the rate everywhere "
                         "else (%d of %d) with no overlap of intervals" % (ka, len(above), kr, len(rest)),
                         detail)
    if ir[0] > ia[1]:
        return Component(DISCRIMINATION, FAILED,
                         "the reversal rate away from the boundary (%d of %d) clears the "
                         "above-boundary rate (%d of %d): the pattern runs the wrong way"
                         % (kr, len(rest), ka, len(above)), detail)
    return Component(DISCRIMINATION, UNRESOLVED,
                     "the above-boundary reversal rate (%d of %d) does not separate from the rate "
                     "everywhere else (%d of %d), so this run does not distinguish the boundary "
                     "account from generic deterioration" % (ka, len(above), kr, len(rest)), detail)


def repetition_component(replication: Optional[Dict[str, Any]]) -> Component:
    """Fresh-data replication, which the registered rule requires before support is complete.

    `arc_instruments.verdicts.p16_endpoint` returns AWAITING FRESH-DATA REPLICATION for an otherwise
    complete single run, so a single run is provisional by the contract and not by preference. The
    record must say both that the data were fresh and that the same rule was met on them; either
    alone is not the requirement.
    """
    if not isinstance(replication, dict) or not replication:
        return Component(REPETITION, NOT_SUPPLIED,
                         "this is a single run and no fresh-data repetition is recorded", {})
    fresh = bool(replication.get("fresh_data"))
    same = bool(replication.get("same_rule_met"))
    detail = {"record": replication}
    if fresh and same:
        return Component(REPETITION, SATISFIED,
                         "a fresh-data repetition met the same registered rule (%s)"
                         % replication.get("source", "source unstated"), detail)
    if not fresh:
        return Component(REPETITION, NOT_SUPPLIED,
                         "the recorded repetition does not state that its data were fresh, so it is "
                         "a re-analysis and not a repetition", detail)
    return Component(REPETITION, FAILED,
                     "the fresh-data repetition did not meet the same registered rule", detail)


# --------------------------------------------------------------------------------------------------
# The wrapper
# --------------------------------------------------------------------------------------------------

def combine(components: Sequence[Component]) -> Dict[str, Any]:
    """Combine the component verdicts, and refuse an overall result while the contract is unmet.

    The order of the rules is the order of the refusals, and it is deliberate.

      1. Any contract-required component other than repetition NOT SUPPLIED: NOT EVALUABLE. The
         instrument did not produce the evidence, so there is nothing to weigh, and a result computed
         from the components that happen to be present is exactly the aggregation finding A2 refuses.
      2. Any component FAILED: never support. REFUTED where a fresh-data repetition met the same
         rule, and AWAITING FRESH-DATA REPLICATION where it did not.
      3. Any component UNRESOLVED: INCONCLUSIVE. Uncertainty that does not decide never decides.
      4. Everything satisfied but no repetition: AWAITING FRESH-DATA REPLICATION, which is the
         registered label and the one `p16_endpoint` returns in the same state.
      5. Everything satisfied and repeated on fresh data: SUPPORTED.

    Every result carries `provisional`, which is true for as long as the fresh-data repetition is
    absent, together with the sentence a reader must carry with it.
    """
    by_name = {c.name: c for c in components}
    missing_names = [n for n in CONTRACT_COMPONENTS if n not in by_name]
    required = [c for c in components if c.required]
    not_supplied = [c.name for c in required if c.state == NOT_SUPPLIED]
    failed = [c.name for c in required if c.state == FAILED]
    unresolved = [c.name for c in required if c.state == UNRESOLVED]
    replicated = by_name.get(REPETITION) is not None and by_name[REPETITION].state == SATISFIED
    blocking = [n for n in not_supplied if n != REPETITION] + missing_names

    reasons: List[str] = []
    if blocking:
        verdict = V.NOT_EVALUABLE
        reasons.append("no proposition-level result: the contract-required evidence is not supplied "
                       "for %s" % ", ".join(blocking))
    elif failed:
        verdict = V.REFUTED if replicated else V.AWAITING_REPLICATION
        reasons.append("component(s) failed: %s" % ", ".join(failed))
    elif unresolved:
        verdict = V.INCONCLUSIVE
        reasons.append("component(s) unresolved: %s" % ", ".join(unresolved))
    elif not replicated:
        verdict = V.AWAITING_REPLICATION
        reasons.append("every component is satisfied on this run and no fresh-data repetition is "
                       "recorded")
    else:
        verdict = V.SUPPORTED
        reasons.append("every contract-required component is satisfied and a fresh-data repetition "
                       "met the same rule")
    if failed and blocking:
        reasons.append("component(s) also failed: %s" % ", ".join(failed))
    return {"P16": verdict, "provisional": not replicated,
            "provisional_label": PROVISIONAL_LABEL if not replicated else "",
            "reason": "; ".join(reasons),
            "components": [c.as_dict() for c in components],
            "component_states": {c.name: c.state for c in components},
            "not_supplied": not_supplied, "failed": failed, "unresolved": unresolved,
            "contract_components": list(CONTRACT_COMPONENTS)}
