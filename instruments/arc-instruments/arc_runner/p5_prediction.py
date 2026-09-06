"""P5's final comparison: the whole prediction uncertainty, the coupling's domain, and one boundary.

WHY THIS FILE EXISTS (finding A7). The sealed prediction and the held-out observation were compared
by an arithmetic that dropped most of the uncertainty on one side of the comparison and all of the
population on the other. Three separate defects, and they are three separate repairs.

THE FIRST IS THE PREDICTION'S OWN UNCERTAINTY. The predicted exponent is a function of three
measured things: the coupling estimated by the bank, the rate calibrated from the system's own
calibration window, and the starting state read at the last calibration checkpoint. The old width
moved ONLY the coupling, to the two ends of its interval, and re-calibrated the rate at each end from
THE SAME calibration readings. The calibration readings therefore entered the width as though they
were exact, and the starting state, which is one of those readings, never moved at all. That is not a
conservative simplification: on a short window the calibrated rate and the starting state are read
from a handful of checkpoints, their read error is the binding precision of the whole comparison, and
an interval that holds them fixed is narrower than the truth by whatever they contribute. The
propagation here resamples all three JOINTLY, so that the part of the calibration error that is
common to the rate and to the starting state moves them together, which is how it moves in the
world: a calibration window read high gives a larger starting state AND a larger fitted rate, and
those two errors partly cancel inside the predicted exponent. Resampling them separately, or holding
one fixed, gets that cancellation wrong in both directions.

THE SECOND IS THE COUPLING'S DOMAIN. The old width moved the coupling to the ends of its interval
and then silently clipped each end into [0, 0.95]. A clip is a statement that the coupling is at the
boundary, which is a stronger claim than the bank made and which reports a narrower interval than the
bank supports: the draws beyond the boundary are exactly the ones the interval could not otherwise
represent, and clipping them onto the boundary hides that the interval reached it at all. Here the
domain is named, the draws that fall outside it are counted rather than moved, and a prediction whose
interval reaches the boundary says so in words. Under strict clearance an interval that touches the
boundary has not cleared it, so such a system is reported unresolved rather than scored against a
prediction the runner cannot represent.

THE THIRD IS THE DIFFERENCE. The observed-minus-predicted interval used the observed replicate
standard error alone, so a system with three tight continuations and a prediction the bank could
barely pin down was compared as though the prediction were exact. A narrow prediction gate does not
make the remaining prediction uncertainty zero: it makes it smaller than the margin, which is a
different statement. The two variances are combined here, with an explicit correlation term that is
zero by default and configurable, and the multiplier applied to the combined standard error is the
LARGER of the replicate t quantile and the equivalence multiplier, so that combining two sources can
only widen the interval and never narrow it.

AND THE BOUNDARY CONVENTION, IN ONE PLACE. The agreement rule read `<=`, so an interval whose end
landed exactly on the margin was called agreement. The candidate's convention is strict clearance:
the interval must lie STRICTLY inside the margin. Exact contact is therefore not clearance, and the
same rule governs the coupling domain. The convention object below is the one place the level, the
multiplier, the strictness and the domain are written down, so that the prediction interval, the
per-system comparison and any later equivalence in P5 cannot drift apart. The registered choice
today is the ninety per cent two-sided interval, being the interval a two one-sided tests procedure
at five per cent reads, and it is named here rather than inferred from whichever helper a caller
happened to reach for. `arc_runner.p5_identification` reads its level, its multiplier and its
boundary rule FROM this object rather than declaring a second copy of them: two files each declaring
the convention and each citing the other is not one place, and the drift it permits is silent. That
is the shape the first repair of finding A6 left behind, and it had already produced one divergence:
the identification equivalence read a closed boundary while this convention read a strict one, so an
interval landing exactly on the margin was agreement in one file and not in the other.

WHAT IT BUYS, MEASURED, AND AT WHICH SEEDS. Over twenty seeds of the runner battery's own worlds, at
the registered configuration. TWENTY RUNS PIN A RATE TO ABOUT A TENTH EITHER WAY, so each count below
is given at BOTH seed bases: 2000 to 2019, where these were first measured, and 1000 to 1019, which
is the base `runner_battery.run_p5_world` uses and therefore the one a reader reproduces. A figure
quoted without its seeds reads as a property of the repair when part of it is the sampling error of
twenty runs. The old arithmetic is reconstructed from the endpoint half width every run still carries
as a diagnostic, scored as the finding describes it: the survivors alone, the observed replicate
spread alone, and a closed boundary. On the TRUE world the repair costs nothing: twenty runs of
twenty reach SUPPORTED under both arithmetics at both bases, because a design adequate to the
question is adequate to the wider interval too. On the REGIME CHANGE world, where the bank measures a
coupling of 0.50 and the held-out panel evolves at 0.35, so that P5 must NOT support the prediction,
the old arithmetic reported SUPPORTED in twenty runs of twenty at both bases and the new one in four
of twenty at the first base and seven of twenty at the second: the false support rate falls from one
to between a fifth and a third, and the runs that changed are those whose predicted interval, once
the calibration window was allowed to move in it, no longer cleared the margin. On the CHEAP LADDER
world, where the ladder is coarse enough that the calibration readings carry real error, the old
arithmetic reported SUPPORTED in fifteen runs of twenty at the first base and sixteen at the second,
and the new one in none at either: a ladder too coarse to pin the prediction down cannot support it,
which the old width could not see because it held the calibration readings fixed. The battery's P5
rows carry the panel's own bookkeeping beside every rate for the same reason, since a panel result
cannot be read without the rate at which systems left the comparison.

OPEN DECISIONS, NAMED RATHER THAN TAKEN. Five choices belong to the author and the conservative
reading is implemented and reported in every record.

  1. The permitted coupling domain is [0, 0.95]. The lower end is the framework's own requirement
     that the coupling not be negative. The upper end is the runner's working ceiling, inherited from
     the clip this file replaces, and it is a numerical guard rather than a statement of theory: the
     complete solution's exponent carries 1 / (1 - beta), which is finite up to one and unusable
     near it. If the registered domain is [0, 1) with 0.95 only a guard, it is one field here.
  2. A prediction whose coupling interval reaches the domain boundary refuses the comparison for
     that system rather than annotating it. The alternative, scoring it with a note, would let a
     prediction the runner cannot represent decide a proposition.
  3. A prediction whose uncertainty cannot be computed at all is refused, not treated as exact. The
     old code took a non-finite half width as no width and compared the point.
  4. The prediction and the observation are combined as independent. They share the sealed
     checkpoint's artefact and nothing else: the continuation replicates are generated after the
     seal and are read at depths the calibration window does not contain, so no reading is common to
     both sides. The correlation is a field, so a domain that knows its two sides share more can say
     so, and the sign is handled correctly in the variance rather than assumed away.
  5. A panel whose assigned list could only be INFERRED after the run, from the systems that produced
     a fit, keeps its counts and loses the proposition-level word: `aggregate` reports the reading and
     returns INCONCLUSIVE where it would have returned SUPPORTED or REFUTED. Counting every system on
     a list read off the survivors is the survivors' denominator under another name, and the finding's
     acceptance case is that complete support must depend on the frozen denominator. The alternative,
     refusing to re-score such a bundle at all, would discard a reading somebody paid for; the
     alternative in the other direction, scoring it as though the panel had been committed to, is the
     defect. Nothing on the `run_p5` path reaches this case: that path writes the panel into the
     manifest's configuration and into the held-out record before any verdict is read.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

AGREES = "AGREES"
DISAGREES = "DISAGREES"
UNRESOLVED = "UNRESOLVED"
NOT_EVALUABLE = "NOT EVALUABLE"

# Student's t at the ninety-fifth percentile, which is the two-sided ninety per cent interval a two
# one-sided tests procedure at five per cent reads. It lives here beside the convention that names
# that level, and `arc_runner.p5` and `arc_runner.p16_components` import it from here through p5, so
# that there is one table and not three.
# THE TWO ONE-SIDED TESTS MULTIPLIER, DECLARED HERE AND NOWHERE ELSE. A TOST at five per cent reads
# the ninety per cent two-sided interval, and this is the large-sample multiplier for it. It sits
# beside the convention that names the level because this file is the one place the level, the
# multiplier, the strictness and the domain are written down; `arc_runner.p5_identification` imports
# the convention from here, and `arc_runner.p5.P5Config` takes its defaults from `REGISTERED` below.
TOST_Z = 1.645

_T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833,
        10: 1.812, 15: 1.753, 20: 1.725, 30: 1.697}


def t95(df: int) -> float:
    """The 95th percentile of Student's t for `df` degrees of freedom. Nearest lower tabulated df;
    1.645 beyond 30, which is the normal multiplier the table is converging to."""
    if df < 1:
        return float("inf")
    keys = [k for k in _T95 if k <= df]
    return _T95[max(keys)] if df <= 30 else 1.645


@dataclass(frozen=True)
class IntervalConvention:
    """The interval level and the boundary rule, in one place, for the whole of P5's comparison.

    `strict_clearance` is the candidate's convention and the one later unison records use: an
    interval agrees with a margin when it lies STRICTLY inside it. The rule this replaces read `<=`,
    so an interval whose end landed exactly on the margin was reported as agreement, which is a
    decision taken at the one point where the evidence decides nothing.

    `equivalence_z` is the multiplier for the interval the equivalence rule reads and for the
    prediction's reported half width, which must be the same level: a difference interval that adds a
    ninety-five per cent half width to a ninety per cent one is an interval at no level at all.

    `coupling_domain` is the permitted domain of the coupling, and it is enforced rather than clipped
    into. `domain_excursion_tolerance` is the fraction of resampled couplings that may fall outside
    it before the comparison is refused; zero is the conservative reading and any excursion at all
    then refuses.

    `prediction_observation_correlation` is the correlation between the prediction's error and the
    observation's error, carried through the variance of their difference with its sign. Zero is the
    default and the reason is in the module docstring: the two sides share no reading.
    """

    equivalence_z: float = TOST_Z
    level: str = ("90 per cent two-sided, being the interval a two one-sided tests procedure at five "
                  "per cent reads")
    strict_clearance: bool = True
    coupling_domain: Tuple[float, float] = (0.0, 0.95)
    domain_excursion_tolerance: float = 0.0
    prediction_draws: int = 800
    prediction_observation_correlation: float = 0.0
    shared_calibration_sd: float = 0.0

    # ---------------------------------------------------------------- the boundary rule, once

    def inside(self, magnitude: float, margin: float) -> bool:
        """Does an interval end at `magnitude` lie inside `margin`? Strictly, under the convention."""
        return magnitude < margin if self.strict_clearance else magnitude <= margin

    def strictly_within_domain(self, value: float) -> bool:
        lo, hi = self.coupling_domain
        if not np.isfinite(value):
            return False
        return (lo < value < hi) if self.strict_clearance else (lo <= value <= hi)

    def clearance(self, difference: float, half_width: float, margin: float) -> Dict[str, Any]:
        """The per-system reading of one difference against the margin.

        AGREES when the interval lies wholly inside the margin, DISAGREES when it lies wholly
        outside, UNRESOLVED otherwise, which includes the case where there is no interval. Both
        outer comparisons are strict under the registered convention, so exact contact with the
        margin decides nothing in either direction and says so.
        """
        m = float(margin)
        if not np.isfinite(difference):
            return {"verdict": UNRESOLVED, "reason": "the difference has no estimate", "margin": m}
        if not np.isfinite(half_width):
            return {"verdict": UNRESOLVED, "margin": m,
                    "reason": "the difference has a point estimate and no interval, so no comparison "
                              "is read; a point comparison is refused here rather than substituted"}
        d = abs(float(difference))
        h = float(half_width)
        if self.inside(d + h, m):
            v, why = AGREES, "the interval on the difference lies wholly inside the margin"
        elif d - h > m:
            v, why = DISAGREES, "the interval on the difference lies wholly outside the margin"
        elif self.strict_clearance and abs((d + h) - m) <= 1e-12:
            v, why = UNRESOLVED, ("the interval on the difference reaches the margin exactly; under "
                                  "strict clearance exact contact is not clearance")
        elif self.strict_clearance and abs((d - h) - m) <= 1e-12:
            v, why = UNRESOLVED, ("the interval on the difference reaches the margin exactly from "
                                  "outside; under strict clearance exact contact does not establish "
                                  "disagreement either")
        else:
            v, why = UNRESOLVED, ("the interval on the difference spans the margin, so this system "
                                  "does not decide whether the prediction was met")
        return {"verdict": v, "reason": why, "margin": m, "difference": float(difference),
                "half_width": h, "interval": [float(difference) - h, float(difference) + h],
                "boundary": "strict" if self.strict_clearance else "closed", "level": self.level}

    def as_record(self) -> Dict[str, Any]:
        return {"equivalence_z": float(self.equivalence_z), "level": self.level,
                "boundary": "strict clearance: the interval must lie strictly inside the margin"
                            if self.strict_clearance else
                            "closed: an interval touching the margin is inside it",
                "strict_clearance": bool(self.strict_clearance),
                "coupling_domain": [float(self.coupling_domain[0]), float(self.coupling_domain[1])],
                "domain_excursion_tolerance": float(self.domain_excursion_tolerance),
                "prediction_draws": int(self.prediction_draws),
                "prediction_observation_correlation": float(self.prediction_observation_correlation),
                "shared_calibration_sd": float(self.shared_calibration_sd),
                "registered_choice": "the 90 per cent two-sided interval with strict clearance"}


# The registered convention. Every default in `arc_runner.p5.P5Config` reads its value from here, so
# that this object remains the one place the convention is written down and the configuration is the
# place a run overrides it.
REGISTERED = IntervalConvention()


def convention_from_config(cfg: Any) -> IntervalConvention:
    """The convention this run is under, taken from the configuration where it states one.

    Every field falls back to the registered value, so a configuration written before these fields
    existed, or a bundle re-scored from one, is read under the registered convention rather than
    under whatever a missing attribute would have defaulted to.
    """
    if cfg is None:
        return REGISTERED
    def _get(name, default):
        v = getattr(cfg, name, None)
        if v is None and isinstance(cfg, dict):
            v = cfg.get(name)
        return default if v is None else v
    domain = _get("coupling_domain", REGISTERED.coupling_domain)
    try:
        domain = (float(domain[0]), float(domain[1]))
    except (TypeError, ValueError, IndexError):
        domain = REGISTERED.coupling_domain
    return IntervalConvention(
        equivalence_z=float(_get("equivalence_z", REGISTERED.equivalence_z)),
        level=str(_get("equivalence_level", REGISTERED.level)),
        strict_clearance=bool(_get("strict_clearance", REGISTERED.strict_clearance)),
        coupling_domain=domain,
        domain_excursion_tolerance=float(_get("domain_excursion_tolerance",
                                              REGISTERED.domain_excursion_tolerance)),
        prediction_draws=int(_get("prediction_draws", REGISTERED.prediction_draws)),
        prediction_observation_correlation=float(_get("prediction_observation_correlation",
                                                     REGISTERED.prediction_observation_correlation)),
        shared_calibration_sd=float(_get("shared_calibration_sd", REGISTERED.shared_calibration_sd)))


# ------------------------------------------------------------------------------------------------
# The joint propagation


def _domain_reading(convention: IntervalConvention, beta_hat: float, beta_se: float,
                    drawn: Optional[np.ndarray]) -> Dict[str, Any]:
    """What the coupling's interval did against the permitted domain, in words and in numbers.

    The analytic endpoints are reported as well as the draws, because a Monte Carlo with a finite
    number of draws can miss a boundary its interval genuinely reaches, and because an author reading
    the record wants the two ends of the interval and not only a fraction.
    """
    lo, hi = convention.coupling_domain
    z = convention.equivalence_z
    ends = ([float(beta_hat - z * beta_se), float(beta_hat + z * beta_se)]
            if np.isfinite(beta_hat) and np.isfinite(beta_se) else
            [float(beta_hat), float(beta_hat)] if np.isfinite(beta_hat) else [float("nan")] * 2)
    if drawn is not None and drawn.size:
        outside = int(np.sum((drawn < lo) | (drawn > hi)))
        touching = int(np.sum((drawn == lo) | (drawn == hi)))
        fraction = outside / float(drawn.size)
    else:
        outside = touching = 0
        fraction = 0.0
    ends_clear = all(convention.strictly_within_domain(e) for e in ends)
    within = ends_clear and fraction <= convention.domain_excursion_tolerance and touching == 0
    if not np.isfinite(ends[0]):
        reason = "the coupling has no estimate, so its interval cannot be placed in the domain"
    elif within:
        reason = ("the coupling's interval [%.4f, %.4f] lies strictly inside the permitted domain "
                  "[%.2f, %.2f]" % (ends[0], ends[1], lo, hi))
    else:
        reason = ("the coupling's interval [%.4f, %.4f] reaches the permitted domain [%.2f, %.2f]: "
                  "%.1f per cent of the resampled couplings fall outside it and are counted rather "
                  "than clipped onto the boundary, so the interval reported here is a LOWER BOUND on "
                  "the prediction's uncertainty" % (ends[0], ends[1], lo, hi, 100.0 * fraction))
    return {"domain": [float(lo), float(hi)], "coupling_interval": ends,
            "excursion_fraction": float(fraction), "n_outside": outside, "n_on_boundary": touching,
            "tolerance": float(convention.domain_excursion_tolerance),
            "within_domain": bool(within), "boundary_reached": not bool(within),
            "reason": reason}


def propagate_prediction(*, beta_hat: float, beta_se: float, depths: Sequence[float],
                         scores: Sequence[float], score_sds: Sequence[float], r1: float,
                         window_end: float, checkpoints: Sequence[float],
                         calibrate: Callable[[Sequence[float], Sequence[float], float], float],
                         predict: Callable[..., float],
                         convention: IntervalConvention = REGISTERED,
                         seed: int = 0,
                         shared_calibration_sd: Optional[float] = None) -> Dict[str, Any]:
    """The predicted exponent with the whole of its uncertainty, propagated jointly.

    `calibrate(depths, scores, beta) -> rate` and `predict(beta, rate, U0, r1, r2, checkpoints)` are
    handed in rather than imported, so that this module carries the propagation and P5 carries the
    solution it is propagating through. One draw resamples the coupling and the WHOLE calibration
    series together, and the starting state is taken from that same resampled series, because the
    starting state IS the last calibration reading: resampling it separately would treat one reading
    as two independent measurements and would lose the cancellation between a rate read high and a
    start read high.

    `shared_calibration_sd` is an error common to every reading of this system, being a calibration
    bias rather than read noise. It is zero by default because the runner measures no such thing;
    where a domain can state one it enters here, and it does NOT cancel out of the prediction, since
    it shifts the starting state while leaving the fitted rate almost unmoved.

    The Monte Carlo uses its own generator seeded from `seed`, so that it consumes nothing from the
    run's stream (the continuations after the seal must be the ones the run would have generated
    without this calculation) and so that the whole propagation is reproducible from the bundle.
    """
    depths = np.asarray(list(depths), float)
    scores = np.asarray(list(scores), float)
    sds = np.asarray(list(score_sds), float) if score_sds is not None else np.zeros_like(scores)
    if sds.shape != scores.shape:
        sds = np.zeros_like(scores)
    sds = np.where(np.isfinite(sds), sds, 0.0)
    shared = float(convention.shared_calibration_sd if shared_calibration_sd is None
                   else shared_calibration_sd)
    draws = int(max(0, convention.prediction_draws))
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)

    def _one(b: float, s: np.ndarray) -> float:
        """One resampled prediction, or a not-a-number where the draw has none.

        The numpy error state is silenced here on purpose and nowhere else: the solution takes a
        fractional power, a draw whose calibrated rate carries the trajectory below zero has no
        prediction at all, and that draw is counted and discarded by the caller. A warning printed
        once per discarded draw would bury the run's own output under the arithmetic of a Monte
        Carlo that is behaving correctly.
        """
        if not np.isfinite(b):
            return float("nan")
        u0 = float(s[-1])
        if u0 <= 0.0:
            return float("nan")
        try:
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                a = float(calibrate(depths.tolist(), s.tolist(), b))
                return float(predict(b, a, u0, r1, window_end, checkpoints))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            return float("nan")

    point = _one(float(beta_hat), scores)
    a_hat = float("nan")
    try:
        a_hat = float(calibrate(depths.tolist(), scores.tolist(), float(beta_hat)))
    except (ValueError, ZeroDivisionError, FloatingPointError):
        pass

    have_beta_se = bool(np.isfinite(beta_se))
    any_read_error = bool(np.any(sds > 0.0) or shared > 0.0)
    if draws <= 0 or not (have_beta_se or any_read_error):
        # No source of uncertainty is measurable here. The old code took that as an exact prediction
        # and compared the point; this reports no interval, and the comparison refuses rather than
        # treating an unknown width as a zero one.
        reading = _domain_reading(convention, float(beta_hat),
                                  float(beta_se) if have_beta_se else float("nan"), None)
        return {"predicted_exponent": point, "se": float("nan"), "half_width": float("nan"),
                "interval": [float("nan"), float("nan")], "percentile_interval": None,
                "usable": False, "n_draws": 0, "n_admissible": 0,
                "reason": ("no uncertainty could be propagated: the coupling carries no standard "
                           "error and the calibration readings carry no read error, so the "
                           "prediction's width is unknown and is not taken to be zero"),
                "beta_hat": float(beta_hat), "beta_se": float(beta_se), "a_hat": a_hat,
                "start_state": float(scores[-1]) if scores.size else float("nan"),
                "shared_calibration_sd": shared, "seed": int(seed),
                "coupling_domain": reading, "components": {}, "convention": convention.as_record()}

    b_draws = (float(beta_hat) + float(beta_se) * rng.standard_normal(draws)) if have_beta_se \
        else np.full(draws, float(beta_hat))
    per_read = rng.standard_normal((draws, scores.size)) * sds
    common = rng.standard_normal(draws) * shared
    s_draws = scores[None, :] + per_read + common[:, None]

    lo, hi = convention.coupling_domain
    in_domain = np.isfinite(b_draws) & (b_draws >= lo) & (b_draws <= hi)
    # A resampled calibration reading at or below zero is outside the support of a pass count, and
    # the solution takes a fractional power of it, so such a draw has no prediction rather than a
    # missing one. It is counted and skipped, never evaluated: evaluating it produced a not-a-number
    # that vanished silently into the discarded draws and told nobody the calibration window was
    # within a read error of zero.
    degenerate = np.any(s_draws <= 0.0, axis=1)
    admissible = in_domain & ~degenerate
    values = []
    for i in range(draws):
        if not admissible[i]:
            continue                      # counted in the domain reading, never clipped onto an end
        v = _one(float(b_draws[i]), s_draws[i])
        if np.isfinite(v):
            values.append(v)
    arr = np.asarray(values, float)
    n_admissible_draws = int(np.sum(admissible))
    degenerate_fraction = float(np.mean(degenerate)) if draws else 0.0

    # The two single-source spreads, so that a reader of the record can see which part of the width
    # is the bank's and which is this system's own calibration window. They are diagnostics and no
    # decision reads them: the joint spread is the one that governs, because the two sources move the
    # prediction together and their contributions do not add on the exponent scale.
    components: Dict[str, Any] = {}
    if have_beta_se:
        only_beta = [_one(float(b), scores) for b in b_draws[admissible]]
        only_beta = np.asarray([v for v in only_beta if np.isfinite(v)], float)
        components["coupling_only_sd"] = float(only_beta.std(ddof=1)) if only_beta.size >= 2 else float("nan")
    if any_read_error:
        only_cal = [_one(float(beta_hat), s_draws[i]) for i in range(draws) if not degenerate[i]]
        only_cal = np.asarray([v for v in only_cal if np.isfinite(v)], float)
        components["calibration_only_sd"] = float(only_cal.std(ddof=1)) if only_cal.size >= 2 else float("nan")

    reading = _domain_reading(convention, float(beta_hat),
                             float(beta_se) if have_beta_se else float("nan"), b_draws)
    reading["degenerate_fraction"] = degenerate_fraction
    if degenerate_fraction > convention.domain_excursion_tolerance:
        reading["within_domain"] = False
        reading["boundary_reached"] = True
        reading["reason"] = ("%.1f per cent of the resampled calibration windows contain a reading at "
                             "or below zero, which is outside the support of a pass count: this "
                             "system's calibration window is within a read error of the bottom of "
                             "the ladder, and the prediction's interval cannot be read there"
                             % (100.0 * degenerate_fraction)) + ". " + reading["reason"]
    if arr.size < 2:
        return {"predicted_exponent": point, "se": float("nan"), "half_width": float("nan"),
                "interval": [float("nan"), float("nan")], "percentile_interval": None,
                "usable": False, "n_draws": draws, "n_admissible": int(arr.size),
                "reason": ("fewer than two of %d resampled predictions could be evaluated, so no "
                           "interval is available" % draws),
                "beta_hat": float(beta_hat), "beta_se": float(beta_se), "a_hat": a_hat,
                "start_state": float(scores[-1]) if scores.size else float("nan"),
                "shared_calibration_sd": shared, "seed": int(seed),
                "coupling_domain": reading, "components": components,
                "convention": convention.as_record()}

    se = float(arr.std(ddof=1))
    half = float(convention.equivalence_z * se)
    tail = 100.0 * (1.0 - _two_sided_coverage(convention.equivalence_z)) / 2.0
    pct = [float(np.percentile(arr, tail)), float(np.percentile(arr, 100.0 - tail))]
    # A percentile interval and a symmetric one differ when the resampled predictions are skewed,
    # which they are when the coupling's interval runs towards the top of its domain: 1 / (1 - beta)
    # is convex there. The symmetric interval is the one the difference uses, because two standard
    # errors combine and two percentile intervals do not, and the gap between the two is reported so
    # that a skew large enough to matter is visible rather than silently averaged away.
    skew_gap = float(abs((pct[1] - pct[0]) / 2.0 - half))
    # THE INTERVAL'S OWN SIMULATION UNCERTAINTY, REPORTED (the handoff's Monte Carlo requirement).
    # This width is estimated from a finite number of draws, so it has a standard error of its own,
    # which for a standard deviation is approximately se / sqrt(2(n - 1)). Nothing decides on it: it
    # is reported because a half width quoted to three decimal places from a draw count that only
    # supports one is a precision the calculation does not have, and a reader comparing the width
    # with the margin is entitled to know how much of the gap between them is simulation noise.
    se_mc = float(se / np.sqrt(2.0 * max(arr.size - 1, 1)))
    # AND THE CENTRE, WHICH IS NOT THE MEAN OF THE DRAWS. The difference is read against the point
    # prediction, being the solution evaluated at the estimates, while the draws average to something
    # else wherever the solution is curved in the coupling: 1 / (1 - beta) is convex, so a coupling
    # interval running towards the top of its domain pushes the draws above the point. The gap is
    # reported rather than corrected, because correcting the centre would move the sealed prediction
    # after the seal, and a gap material beside the margin is a statement that the prediction is not
    # linear enough over its own interval for a symmetric comparison to mean what it appears to.
    centre_gap = float(arr.mean() - point) if np.isfinite(point) else float("nan")
    return {"predicted_exponent": point, "mean_of_draws": float(arr.mean()), "se": se,
            "se_monte_carlo_error": se_mc,
            "half_width_monte_carlo_error": float(convention.equivalence_z * se_mc),
            "mean_of_draws_minus_point": centre_gap,
            "half_width": half, "interval": [point - half, point + half], "percentile_interval": pct,
            "percentile_vs_symmetric_half_width_gap": skew_gap, "usable": True,
            "n_draws": draws, "n_admissible": int(arr.size),
            "n_admissible_draws": n_admissible_draws,
            "n_without_a_prediction": int(n_admissible_draws - arr.size),
            "beta_hat": float(beta_hat), "beta_se": float(beta_se), "a_hat": a_hat,
            "start_state": float(scores[-1]) if scores.size else float("nan"),
            "shared_calibration_sd": shared, "seed": int(seed),
            "resampled": "the coupling, every calibration reading and the starting state jointly; "
                         "the starting state is the last calibration reading and moves with it",
            "coupling_domain": reading, "components": components,
            "convention": convention.as_record()}


def _two_sided_coverage(z: float) -> float:
    """The two-sided normal coverage of a multiplier, for naming the percentile interval at the same
    level as the symmetric one. An error function rather than a table, because the multiplier is
    configurable and a table would only cover the values somebody happened to write down."""
    from math import erf, sqrt
    return float(erf(abs(float(z)) / sqrt(2.0)))


def difference_interval(*, observed: float, observed_se: float, predicted: float,
                        predicted_se: float, n_replicates: int,
                        convention: IntervalConvention = REGISTERED) -> Dict[str, Any]:
    """The observed-minus-predicted difference with BOTH uncertainties in it.

    The variance carries the correlation term with its sign, so a domain that declares its two sides
    positively correlated gets a narrower difference and one that declares them negatively
    correlated gets a wider one, which is the arithmetic rather than an assumption. The multiplier is
    the larger of the replicate t quantile and the equivalence multiplier: the observed side has few
    degrees of freedom and deserves t, the prediction side is a Monte Carlo of an asymptotically
    normal estimate, and taking the larger of the two means that adding the prediction's uncertainty
    can only widen the interval.
    """
    d = float(observed) - float(predicted)
    so, sp = float(observed_se), float(predicted_se)
    rho = float(convention.prediction_observation_correlation)
    if not np.isfinite(so) or not np.isfinite(sp):
        missing = ("the observed replicate spread" if not np.isfinite(so)
                   else "the prediction's propagated uncertainty")
        return {"difference": d, "se": float("nan"), "half_width": float("inf"),
                "multiplier": float("nan"), "observed_se": so, "predicted_se": sp,
                "correlation": rho,
                "reason": "%s is not available, so no difference interval can be read" % missing}
    var = so * so + sp * sp - 2.0 * rho * so * sp
    se = float(np.sqrt(max(var, 0.0)))
    mult = float(max(t95(int(n_replicates) - 1), convention.equivalence_z))
    return {"difference": d, "se": se, "half_width": float(mult * se), "multiplier": mult,
            "observed_se": so, "predicted_se": sp, "correlation": rho,
            "share_of_variance_from_the_prediction": (float(sp * sp / var) if var > 0 else float("nan")),
            "reason": "the observed replicate spread and the prediction's propagated uncertainty, "
                      "combined, at the larger of the replicate t quantile and the equivalence "
                      "multiplier"}


# ------------------------------------------------------------------------------------------------
# The aggregate over the assigned panel


AGGREGATE_RULE = (
    "the denominator is the ASSIGNED held-out panel, frozen in the sealed configuration. A system "
    "that could not be evaluated keeps its place in that denominator as NOT EVALUABLE and is never "
    "removed from it: removing it would decide the proposition on the systems that survived, which "
    "is a different population from the one that was assigned. SUPPORTED needs agreement on more "
    "than half of the assigned panel and a premise that is not refuted; REFUTED needs disagreement "
    "on more than half of it; a panel in which no assigned system could be evaluated at all is NOT "
    "EVALUABLE; everything else is INCONCLUSIVE. A denominator that was not frozen before the run, "
    "but inferred afterwards from the systems that produced a fit, decides nothing at proposition "
    "level: the counts are reported and the panel result is INCONCLUSIVE, because support read off "
    "an inferred denominator is support read off the survivors under another name."
)


def aggregate(per_system: Dict[str, Dict[str, Any]], assigned: Sequence[str],
              premise: str = "NOT REFUTED", denominator_frozen: bool = True,
              denominator_source: str = "") -> Dict[str, Any]:
    """The panel result and the denominator it was decided on, reported together.

    Finding A7: a system that failed the headroom rule was labelled NOT EVALUABLE and then dropped
    before the denominator was counted, so a panel of five systems of which three reached the ladder
    ceiling could be declared supported by the two that did not. The candidate's weight on the
    primary system panel is frozen before the run, and the arithmetic here keeps it: every assigned
    system is counted, whatever happened to it.

    `denominator_frozen` is whether the list handed in was a PRE-RUN COMMITMENT. Keeping every
    assigned system repairs the arithmetic and leaves one door open: where the assigned list itself
    was reconstructed after the fact from the systems that produced a fit, every one of them is
    counted and the denominator is still the survivors, which is the population the finding forbids
    deciding on. The counts are reported either way, because a reader is entitled to them; the
    proposition-level words SUPPORTED and REFUTED are withheld, because the finding's acceptance case
    is that complete P5 support must depend on the frozen denominator. `denominator_source` is where
    the list came from, recorded so that a reader can tell a frozen denominator from an inferred one
    without comparing two files by eye. The default is the frozen reading, so a caller that supplies
    a panel it committed to is unaffected.
    """
    assigned = [s for s in assigned]
    # AND THE POPULATION CANNOT GROW EITHER. A system present in the results and absent from the
    # assigned panel is the same defect from the other side: the finding's title is that the final
    # comparison can CHANGE the population, and a system added after the seal changes it upwards. It
    # is never counted, and it is named here so that a reader of the panel can see it happened rather
    # than having to compare two lists by eye.
    unassigned = [s for s in per_system if s not in assigned]
    counts = {AGREES: 0, DISAGREES: 0, UNRESOLVED: 0, NOT_EVALUABLE: 0, "MISSING": 0}
    for s in assigned:
        rec = per_system.get(s)
        if rec is None:
            counts["MISSING"] += 1
            continue
        v = rec.get("verdict")
        counts[v] = counts.get(v, 0) + 1
    n = len(assigned)
    evaluated = counts[AGREES] + counts[DISAGREES] + counts[UNRESOLVED]
    not_evaluable = counts[NOT_EVALUABLE] + counts["MISSING"]
    if n == 0 or evaluated == 0:
        result = NOT_EVALUABLE
    elif counts[DISAGREES] > n / 2.0:
        result = "REFUTED"
    elif counts[AGREES] > n / 2.0 and premise in ("HOLDS", "NOT REFUTED"):
        result = "SUPPORTED"
    else:
        result = "INCONCLUSIVE"
    # AND THE DENOMINATOR ITSELF HAS TO HAVE BEEN COMMITTED TO. Counting every system on a list that
    # was itself read off the systems which produced a fit is the survivors' denominator wearing the
    # word "assigned": the arithmetic is right and the population is still the one the run selected
    # after seeing the data. The counts stand and the proposition-level word is withheld.
    inferred_note = ""
    if not denominator_frozen and result in ("SUPPORTED", "REFUTED"):
        inferred_note = ("the panel would have read %s on these counts, and the denominator was not "
                         "frozen before the run but inferred afterwards (%s), so the proposition is "
                         "not decided on it: complete support must depend on a denominator that was "
                         "committed to" % (result, denominator_source or "source not recorded"))
        result = "INCONCLUSIVE"
    return {"result": result, "assigned": n, "agrees": counts[AGREES],
            # WHERE THE DENOMINATOR CAME FROM, in the record beside the result. A reader cannot tell a
            # frozen panel from an inferred one by looking at a list of names, and the difference is
            # the whole of finding A7's population question.
            "denominator_frozen": bool(denominator_frozen),
            "denominator_source": denominator_source,
            "denominator_note": inferred_note,
            "disagrees": counts[DISAGREES], "unresolved": counts[UNRESOLVED],
            "not_evaluable": not_evaluable, "evaluated": evaluated,
            # A system that never reported at all, whether it is absent from the table or carries a
            # row saying so. Both routes reach the same count, so a caller that builds the table
            # itself and a caller that goes through `arc_runner.p5` agree.
            "never_reported": counts["MISSING"] + sum(
                1 for s in assigned if (per_system.get(s) or {}).get("never_reported")),
            "assigned_systems": list(assigned), "premise": premise, "rule": AGGREGATE_RULE,
            "unassigned_systems_present": list(unassigned),
            "n_unassigned_systems_present": len(unassigned),
            "unassigned_note": ("%d system(s) were scored that the sealed panel does not contain; "
                                "they are reported and never counted, because the denominator is the "
                                "panel that was assigned" % len(unassigned)) if unassigned else "",
            "majority_needed": float(n) / 2.0}
