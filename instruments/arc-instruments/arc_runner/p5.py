"""P5, run: the crossed bank, two routes, errors in variables, seal-then-generate, the typed pair.

This is the P5 registration's charter as code. The bank crosses capability states with retention fractions and
replicates each cell. The coupling is estimated by two routes that share no estimator: the response to
the state at fixed retention, and the response to retention at fixed state. The held-out panel runs a
calibration window, is checkpointed, the predictions are sealed, and only then is the continuation
generated. The result is a typed pair: PREDICTION and IDENTIFICATION, never one word.

WHAT EACH ROUTE IS NOW FITTED TO, AND WHY IT CHANGED (finding A5). Both routes used to be Deming
slopes of the log increment on the log available capability, with the nonpositive increments removed
before the transform. Two things were wrong with that and both are in `arc_runner.p5_observation`. The
before reading enters the available capability and is subtracted inside the increment, so the errors
on the two axes are correlated by construction, and Deming's single ratio of two error variances
cannot carry a covariance. And removing the nonpositive increments removes exactly the cells where the
read noise ran downwards, which is a selection on the noise and falls hardest where the increment is
smallest beside that noise, so it kept the upward errors and deleted the genuine regressions, and then
reported them as cells the ladder could not resolve. On a bank whose increment does not depend on the
available capability at all, what survived carried a fitted exponent of plus 0.12 against a truth of
zero. Both routes are now maximum likelihood fits of the observation model to the
ORIGINAL PAIRED READINGS, with the read covariance in the conditional mean of the increment, the
process variability in its conditional variance, the retention fraction handled as the exact quantity
it is, and no row dropped for any reason. `_deming` and `_log_error_ratio` remain because the log
scale diagnostic still uses them and because they are the arithmetic the repair is measured against.

AND NO REGISTERED LEVEL DROPPED EITHER, WHICH IS THE OTHER HALF. A route was then built as the mean
of separate fits to its registered levels, over the levels that produced a usable exponent. A level
produces none exactly when its own increment is not distinguishable from zero, so that mean is the
row filter's own selection moved up to the level of the registered design: a bank at the registered
replicate count whose round could not use the smallest retained fraction reported 0.5274 from four of
its five retention levels, with the precision condition passing and nothing saying a level was
absent. The registration forbids it in two places, "Every run at a registered seed and retention
level is included. No run is excluded on its outcome" and "No retention level is added, dropped or
reweighted after any increment is seen", so each route is now ONE fit over every cell with one common
exponent and one rate per registered level. The per-level fits are still computed and still printed,
as diagnostics that decide nothing.

WHAT THE REGISTRATION NAMES AS ITS PRIMARY IS COMPUTED AND REPORTED IN EVERY RUN, under
`routes["registered_estimator"]`, together with the declaration that superseding it is an amendment
the author has not made. See `arc_runner.p5_observation.registered_log_scale_estimate`.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from arc_instruments import capacity, coupling_identification as ci
from . import (custody as CUSTODY, manifest as M, mode as MODE, p5_identification as PI,
               p5_observation as PO, p5_prediction as PRED, sampling as SAMPLING)
from .adapters import ModelAdapter
from .ladder import Ladder
from .trajectory import loglog_slope, run_round, run_trajectory


@dataclass
class P5Config:
    # The bank's states span the capability range the held-out trajectory will traverse, in the
    # ladder's own units, because a coupling measured only at the bottom of the ladder predicts the
    # bottom and a pass rate near zero carries almost no information per item. The first end-to-end
    # run placed every state below a quarter of the ladder and recovered 0.24 for a true 0.50; the
    # attenuation was precision, not the estimator, which is the two per cent requirement of the P5 registration.
    states: Sequence[float] = (30.0, 60.0, 100.0, 160.0, 250.0)
    fractions: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0)
    reps: int = 5
    calibration_depths: Sequence[int] = (1, 2, 3, 4)
    window_end: int = 128
    checkpoints: Sequence[int] = (4, 8, 16, 32, 64, 128)
    margin: float = 0.10                # ruling 10: exponent-scale margin
    route_margin: float = 0.10          # the two regression directions are CONSISTENT when the
                                        # interval on the gap between them lies wholly inside this.
                                        # It is a consistency margin and never an identification:
                                        # see arc_runner.p5_identification and finding A6.
    route_equivalence_z: float = PRED.REGISTERED.equivalence_z
                                        # the multiplier for the interval the equivalence test reads,
                                        # taken from the REGISTERED interval convention rather than
                                        # declared here, so that the route gap, the capability
                                        # manipulation's agreement and the final per-system
                                        # comparison are all read at one level with one boundary
                                        # rule. The boundary itself is `strict_clearance` below and
                                        # is not a second field: an interval landing exactly on the
                                        # margin has not cleared it, in the identification
                                        # equivalence exactly as in the final comparison.
    eiv_delta: Optional[float] = None   # ratio of error variances; None = estimate from the bank's repeated reads, which is the registered rule
    reads: int = 4                      # repeated ladder reads per bank cell, averaged
    cal_reads: int = 16                 # reads per calibration checkpoint: the rate is the binding precision
    heldout_reads: int = 4              # reads per held-out checkpoint
    max_unresolved_fraction: float = 0.2  # H2: more unresolved cells than this and the bank is inconclusive
    # WHAT A NEGATIVE CONTROL THAT PRODUCED NO INCREMENT AT ALL COUNTS AS. A control cell retains
    # material the round cannot use, so the world the design is built for is one in which the control
    # increment is zero. There is no elasticity of an increment that is not there, so such a control
    # returns no readable exponent, and reading that as UNRESOLVED makes the registered IDENTIFIED
    # unreachable for a PERFECT control while a control that still grew a little could pass. That
    # inversion is a defect, and the reading implemented here is that a control whose increment is
    # bounded below the bank's own carries no material coupling, with the bound measured and reported.
    # The alternative reading, that no readable exponent is always UNRESOLVED, is the author's to take
    # and is this switch.
    control_no_increment_passes: bool = True
    replicates: int = 4               # replicate continuations per held-out system from the one sealed checkpoint;
                                      # the exponent reported per system is their mean and its standard error is
                                      # their spread, which carries process noise as well as read noise. The first
                                      # runner battery scored a single continuation against the margin with no
                                      # uncertainty at all and refuted a linear-growth world 47 times in 100.
    # Per-system agreement is a two-one-sided-tests equivalence rule: AGREES when the 90 per cent interval on the
    # difference lies wholly inside the margin, DISAGREES when it lies wholly outside, UNRESOLVED otherwise. The
    # interval uses the t quantile for the registered replicate count, never a normal quantile borrowed from a
    # larger sample.
    task: str = "Improve the artefact."
    start_capability: float = 20.0
    control_fraction_of_bank: float = 0.2   # negative-control cells, run in the same bank and frozen with it, as a
                                            # registered fraction of the main bank's replicates per cell
    control_reads_multiplier: int = 32      # a control cell's increment is about a tenth of a main cell's, so the
                                            # same read count leaves it under the ladder's noise; reads cost no
                                            # generation (for a checkable ladder a read is a run of the hidden
                                            # suite, so the control is read with the whole item pool rather than a
                                            # subset). The first battery read the control at interval width 0.8
                                            # against a margin of 0.10 with the bank's read count, and at 0.16 with
                                            # eight times it; the equivalence rule needs about 0.08.
    bootstrap: int = 200                    # row-resampling draws for the log-scale diagnostic's interval. The two
                                            # routes and the negative control no longer resample: each is a
                                            # maximum likelihood fit of the observation model and carries its
                                            # own standard error, which is both cheaper and the reading that
                                            # does not have to drop a cell in order to be computed at all.
    z_interval: float = 1.96                # the two-sided 95 per cent interval on the bank's estimates
    # THE INTERVAL AND BOUNDARY CONVENTION OF THE FINAL COMPARISON, IN ONE PLACE (finding A7). Every
    # default below is read from `arc_runner.p5_prediction.REGISTERED`, which is where the convention
    # is written down; these fields are the place a run OVERRIDES it, so the two cannot drift apart.
    # The registered choice is the ninety per cent two-sided interval with STRICT clearance: an
    # interval that lands exactly on the margin has not cleared it. The rule this replaces read `<=`
    # and called exact contact agreement.
    equivalence_z: float = PRED.REGISTERED.equivalence_z
    equivalence_level: str = PRED.REGISTERED.level
    strict_clearance: bool = PRED.REGISTERED.strict_clearance
    # The permitted domain of the coupling. It is ENFORCED and never clipped into: the previous
    # prediction width moved the coupling to the ends of its interval and then silently clipped each
    # end into this range, which reports a narrower interval than the bank supports and hides that
    # the interval reached the boundary at all.
    coupling_domain: tuple = PRED.REGISTERED.coupling_domain
    domain_excursion_tolerance: float = PRED.REGISTERED.domain_excursion_tolerance
    # The resampling that carries the calibration, the bank and the starting state into the sealed
    # prediction's interval jointly. Its generator is seeded from the run's seed and consumes nothing
    # from the run's own stream, so the continuations after the seal are the ones the run would have
    # generated without this calculation.
    prediction_draws: int = PRED.REGISTERED.prediction_draws
    prediction_observation_correlation: float = PRED.REGISTERED.prediction_observation_correlation
    shared_calibration_sd: float = PRED.REGISTERED.shared_calibration_sd
    # H3, the constant-elasticity check. The dose-response of log increment on log available capability carries
    # a registered quadratic term; the change of local elasticity it implies across the titrated range is
    # compared with the registered margin. Material curvature refutes the premise and is reported in those words.


def _deming(x, y, delta):
    """The errors-in-variables slope, and ordinary least squares where there is no error to correct.

    Deming's delta is var(err_y) / var(err_x), and delta tending to infinity recovers OLS of y on x.
    An exact ladder reading gives var(err_x) = 0 exactly, so the ratio is not a finite number and the
    closed form cannot be evaluated at it: the correct estimator in that case is the one the limit
    names, and it is taken here rather than approached with a large finite delta that only introduces
    a rounding error. Finding A4: a deterministic whole-pool read has no measurement error at all, so
    an attenuation correction has nothing to correct and must not invent a ratio to correct it by.
    """
    if not np.isfinite(delta):
        x = np.asarray(list(x), float); y = np.asarray(list(y), float)
        if x.size < 2 or float(np.var(x)) == 0.0:
            return float("nan")
        return float(np.polyfit(x, y, 1)[0])
    return float(capacity.deming_slope(list(x), list(y), delta=delta))


def _log_error_ratio(rows) -> float:
    """The registered errors-in-variables ratio, estimated from the bank's own repeated reads rather than
    assumed. Deming's delta is var(err_y) / var(err_x); in log space each error is the read sd over the
    quantity. Assuming delta = 1 when the increment's error dwarfs the state's inflates the slope, which
    is the mirror image of the attenuation the correction exists to remove."""
    ex = [ (r["read_sd"] / np.sqrt(2)) / max(r["available"], 1e-9) for r in rows ]   # before-read error on x
    ey = [ r["read_sd"] / max(r["increment"], 1e-9) for r in rows ]                     # difference error on y
    vx, vy = float(np.mean(np.square(ex))), float(np.mean(np.square(ey)))
    if vx <= 0.0:
        # An exact ladder read: the state on the x axis carries no measurement error, so there is no
        # attenuation to correct and the limit delta -> infinity, which `_deming` takes as ordinary
        # least squares, is the estimator. Finding A4: before the sampling unit was declared, a
        # deterministic whole-pool read reported a fabricated binomial error here, and the correction
        # was applied to a quantity that did not have the error it was correcting for.
        return float("inf")
    return vy / vx


def run_bank(adapter: ModelAdapter, ladder: Ladder, cfg: P5Config, rng: np.random.Generator,
             place_at_state) -> Dict[str, Any]:
    """The crossed bank. `place_at_state(state)` returns an artefact at that capability state; for the
    mock it sets the latent capability, for a real system it loads a checkpoint measured at that state."""
    from .ladder import read_mean
    rows, nonpositive = [], 0
    for s in cfg.states:
        for f in cfg.fractions:
            for r in range(cfg.reps):
                art = place_at_state(s)
                ctx = "bank state=%s fraction=%s rep=%d" % (s, f, r)
                before, sd_b, _ = read_mean(ladder, art, rng, cfg.reads, ctx + " before")
                # The revised artefact is BOUND rather than passed through, because the bundle keeps
                # the thing that was read as well as the number read off it: a saved row carrying a
                # count and no artefact cannot be recounted by anybody.
                after_art = run_round(adapter, art, f, cfg.task, rng)
                after, sd_a, ceiling_a = read_mean(ladder, after_art, rng, cfg.reads, ctx + " after")
                inc = after - before
                if inc <= 0:
                    nonpositive += 1          # counted and KEPT: finding A5. A nonpositive increment is
                                              # an observation of this cell, not a failure of the ladder,
                                              # and it is fitted with every other row.
                # THE TWO READINGS ARE THE MEASUREMENT AND THEY ARE RECORDED AS SUCH. `available` and
                # `increment` are functions of them and are kept for the diagnostics and for every
                # bundle written before this change; the estimator reads `before`, `after` and their
                # own variances, because the correlation between the two derived numbers is exactly
                # the thing the derived numbers cannot show.
                rows.append({"state": s, "fraction": f, "rep": r, "available": f * before,
                             "increment": inc, "read_sd": float(np.hypot(sd_a, sd_b)), "control": False,
                             "before": before, "after": after,
                             "before_sd": sd_b, "after_sd": sd_a,
                             "var_before": float(sd_b) ** 2, "var_after": float(sd_a) ** 2,
                             # The two artefacts and the ceiling flag of the second, so that a saved
                             # bank can be recounted from what was read rather than believed on the
                             # strength of the count it reported.
                             "before_artefact": art, "after_artefact": after_art,
                             "after_at_ceiling": ceiling_a})
    # NEGATIVE CONTROLS, run in the same bank and frozen with it: retained material the round cannot use.
    # The framework predicts no coupling in the retained fraction here; a material coupling in these cells
    # says the estimator responds to something other than the mechanism, and the identification fails.
    n_ctrl = max(1, int(round(cfg.reps * cfg.control_fraction_of_bank)))
    c_reads = cfg.reads * max(1, cfg.control_reads_multiplier)
    for s in cfg.states:
        for f in cfg.fractions:
            for r in range(n_ctrl):
                art = place_at_state(s)
                ctx = "control state=%s fraction=%s rep=%d" % (s, f, r)
                before, sd_b, _ = read_mean(ladder, art, rng, c_reads, ctx + " before")
                after_art = run_round(adapter, art, f, cfg.task, rng, control="unusable_retention")
                after, sd_a, ceiling_a = read_mean(ladder, after_art, rng, c_reads, ctx + " after")
                rows.append({"state": s, "fraction": f, "rep": r, "available": f * before,
                             "increment": after - before, "read_sd": float(np.hypot(sd_a, sd_b)),
                             "control": True, "before": before, "after": after,
                             "before_sd": sd_b, "after_sd": sd_a,
                             "var_before": float(sd_b) ** 2, "var_after": float(sd_a) ** 2,
                             "before_artefact": art, "after_artefact": after_art,
                             "after_at_ceiling": ceiling_a})
    # AND EVERY ROW CARRIES ITS OWN IDENTITY AND THE READING OF ITS INCREMENT. A replay that cannot
    # name a row cannot say which row failed to reproduce, and an increment of zero or less is an
    # observation of this cell that is fitted with the rest: the status names it rather than hiding
    # it in the sign of a number.
    for i, row in enumerate(rows):
        row["observation_id"] = "bank-%06d" % i
        row["increment_status"] = "POSITIVE" if row["increment"] > 0 else "ZERO OR REGRESSION"
    main = [r for r in rows if not r["control"]]
    # WHAT EVERY read_sd IN THESE ROWS IS THE SAMPLING ERROR OF, recorded once with the rows rather
    # than left to be guessed from the numbers. A bank of pass counts whose sampling unit is not
    # stated cannot be re-analysed by anybody, including its own authors a month later. Finding A4.
    # WHAT THE OBSERVATION MODEL OF THESE ROWS IS, recorded with them for the same reason the read
    # model is: whether the before and after reads of a cell share their item form decides the sign
    # and the size of the covariance between the two derived quantities, and a later analyst cannot
    # recover that from the numbers. Finding A5.
    return {"rows": rows, "nonpositive": nonpositive, "cells": len(main),
            "control_cells": len(rows) - len(main),
            "read_model": SAMPLING.read_model_record(ladder),
            "observation_model": PO.model_for(ladder).as_record()}


def _observation_model(bank: Dict[str, Any]) -> PO.ObservationModel:
    """The observation model the bank was recorded under, or the fail-closed default for a bank
    written before it was recorded. Zero read correlation is the wider reading, so a bank that never
    declared one is fitted under the more cautious weights and not the more flattering ones."""
    rec = bank.get("observation_model") or {}
    return PO.ObservationModel(read_correlation=float(rec.get("read_correlation", 0.0) or 0.0),
                               state_is_shared=bool(rec.get("state_is_shared", True)))


def _curvature(rows, cfg: P5Config, model: PO.ObservationModel) -> Dict[str, Any]:
    """H3, the constant-elasticity check: the local elasticity of increment on available capability over
    the upper half of the titrated range against the lower half, each fitted to the paired readings of
    its own half, and the difference carrying the interval the two independent fits imply.

    A plain quadratic on these data is biased: the state read enters the available capability and the
    increment with opposite signs and its relative error is largest at the low end, which shows as
    curvature that is not there. The first battery's quadratic refuted a true premise in ten runs of a
    hundred for that reason. The halves are disjoint sets of cells, so their fits are independent and
    the variance of the difference is the sum of the two variances; the previous version resampled the
    two Deming slopes instead, and each of those had already dropped the half's nonpositive cells,
    which is the same selection twice over on the half where the increment is smallest.

    REFUTED (curvature) when the interval on the change lies wholly beyond the margin; HOLDS when it
    lies wholly inside (equivalence established); NOT REFUTED otherwise. H3 is registered as a
    refutation test, so SUPPORTED for P5 needs NOT REFUTED or HOLDS, and the label says which."""
    if len(rows) < 12:
        return {"premise": "NOT REFUTED", "elasticity_change": float("nan"), "half_width": float("nan"),
                "reason": "too few cells"}
    med = float(np.median([r["available"] for r in rows]))
    lo = [r for r in rows if r["available"] < med]
    hi = [r for r in rows if r["available"] >= med]
    fl = PO.fit_paired(lo, model, cfg.z_interval)
    fh = PO.fit_paired(hi, model, cfg.z_interval)
    if not (fl.usable and fh.usable):
        return {"premise": "NOT REFUTED", "elasticity_change": float("nan"), "half_width": float("nan"),
                "split_at_available": med,
                "reason": "one half of the titrated range carries no readable elasticity (%s; %s)"
                          % (fl.adequacy, fh.adequacy),
                "lower_half": fl.as_record(), "upper_half": fh.as_record()}
    point = float(fh.beta - fl.beta)
    if np.isfinite(fl.beta_se) and np.isfinite(fh.beta_se):
        half = float(cfg.z_interval * np.hypot(fl.beta_se, fh.beta_se))
    else:
        half = float("inf")
    if abs(point) - half > cfg.margin:
        premise = "REFUTED (curvature)"
    elif abs(point) + half <= cfg.margin:
        premise = "HOLDS"
    else:
        premise = "NOT REFUTED"
    return {"premise": premise, "elasticity_change": point, "half_width": half,
            "slope_lower": float(fl.beta), "slope_upper": float(fh.beta), "split_at_available": med,
            "lower_half": fl.as_record(), "upper_half": fh.as_record()}


def _control_coupling(rows, cfg: P5Config, model: PO.ObservationModel,
                      bank_fit: Optional[PO.PairedFit] = None) -> Dict[str, Any]:
    """The retention route estimated on the negative-control cells alone.

    EVERY CONTROL CELL IS KEPT, AND SO IS EVERY CONTROL STATE. The first version fitted
    `r["control"] and r["increment"] > 0`, so the control's own nonpositive cells were dropped before
    the check that is supposed to establish that there is NO coupling here; a control whose increment
    is small is precisely the control whose cells that filter removes, and removing them can only
    make a control look cleaner than it is. The second version kept the cells and then averaged only
    the STATES that carried a readable exponent, which is the same selection one level up. The
    control is now one fit over all of its cells with one common exponent and one rate per state, so
    nothing is left out of it.

    WHAT A CONTROL WITH NO INCREMENT AT ALL MEANS, WHICH IS THE WORLD THIS CONTROL IS BUILT FOR. The
    control retains material the round cannot use, so a clean control produces no increment. An
    elasticity is the elasticity OF an increment, and where there is none there is no exponent to
    read: a fit that reported one there would be describing read noise. Treating that as UNRESOLVED,
    which is what the fit's own adequacy says, makes the registered IDENTIFIED unreachable for a
    PERFECT control while a control that still grew slightly could pass, and the cleaner the control
    the more certainly the run could not identify. That inversion is a defect and it is repaired
    here on a quantity that still exists when the rate is zero: the fitted MEAN INCREMENT of the
    control cells, on the same capability scale as the bank's. Where the control's increment is
    bounded above by less than the bank's own increment is bounded below, at the run's interval, the
    round produced nothing in these cells for a coupling to be an elasticity of, and the control
    carries no material coupling. Where the bound is not that tight the control is UNRESOLVED and
    says which of the two it failed, because a control too imprecise to bound its own increment has
    not established anything either.
    """
    ctrl = [r for r in rows if r.get("control")]
    if len(ctrl) < 6:
        return {"beta_control": float("nan"), "half_width": float("nan"), "status": "UNRESOLVED",
                "cells": len(ctrl), "reason": "fewer than six control cells"}
    fit = PO.fit_paired(ctrl, model, cfg.z_interval, group_by="state")
    per_state = [PO.fit_paired([r for r in ctrl if r["state"] == s], model, cfg.z_interval)
                 for s in cfg.states]
    out: Dict[str, Any] = {"cells": len(ctrl), "fit": fit.as_record(),
                           "n_state_sets_in_the_fit": len(fit.group_labels),
                           "state_sets_dropped": 0,
                           "per_state": [f.as_record() for f in per_state],
                           "per_state_is_a_diagnostic": True}
    if not fit.usable and fit.adequacy == PO.NO_MEASURABLE_GROWTH:
        c_hi = float(fit.mean_increment + cfg.z_interval * fit.mean_increment_se)
        b_lo = (float(bank_fit.mean_increment - cfg.z_interval * bank_fit.mean_increment_se)
                if bank_fit is not None else float("nan"))
        out.update({"beta_control": float("nan"), "half_width": float("nan"),
                    "control_mean_increment": float(fit.mean_increment),
                    "control_mean_increment_upper": c_hi,
                    "bank_mean_increment_lower": b_lo,
                    "basis": "the increment, because no elasticity of a missing increment exists"})
        if (cfg.control_no_increment_passes and np.isfinite(c_hi) and np.isfinite(b_lo)
                and c_hi < b_lo):
            out.update({"status": "NO MATERIAL COUPLING",
                        "reason": "the control cells produced no increment distinguishable from "
                                  "zero and the largest increment consistent with them (%.4g) lies "
                                  "below the smallest consistent with the bank (%.4g), so there is "
                                  "no increment here for a coupling to be an elasticity of"
                                  % (c_hi, b_lo)})
        else:
            out.update({"status": "UNRESOLVED",
                        "reason": "the control cells carry no elasticity (%s) and their own "
                                  "increment is not bounded below the bank's, so this control has "
                                  "not established that the estimator responds to nothing here"
                                  % fit.adequacy})
        out["open_decision"] = ("whether a negative control that produced no increment at all "
                                "satisfies the registered condition that the controls return no "
                                "material coupling, or is instead unresolved, is the author's; the "
                                "first reading is implemented because the second makes IDENTIFIED "
                                "unreachable for a perfect control, and the switch is "
                                "P5Config.control_no_increment_passes")
        return out
    if not fit.usable:
        out.update({"beta_control": float("nan"), "half_width": float("nan"), "status": "UNRESOLVED",
                    "reason": "the control cells carry no readable elasticity (%s: %s)"
                              % (fit.adequacy, fit.adequacy_reason)})
        return out
    point, se = float(fit.beta), float(fit.beta_se)
    half = float(cfg.z_interval * se) if np.isfinite(se) else float("inf")
    if abs(point) + half <= cfg.margin:
        status = "NO MATERIAL COUPLING"
    elif abs(point) - half > cfg.margin:
        status = "MATERIAL COUPLING"
    else:
        status = "UNRESOLVED"
    out.update({"beta_control": point, "half_width": half, "status": status,
                "basis": "the exponent, the control cells having a readable increment"})
    return out


def _route_record(fit: PO.PairedFit, level_key: str, expected: Sequence[Any]) -> Dict[str, Any]:
    """One route, as one fit over every cell with one rate per registered level.

    `levels_dropped` is zero by construction and is printed anyway, because the defect this replaces
    was invisible: a route reported as the mean of the levels that happened to grow looked exactly
    like a route reported from all of them. `levels_missing` is the separate question of whether the
    bank actually contains every level the configuration registered, which is a fact about the data
    and not about the estimator.
    """
    present = list(fit.group_labels)
    missing = [v for v in expected if v not in present]
    return {"beta": float(fit.beta) if fit.usable else float("nan"),
            "se": float(fit.beta_se) if fit.usable else float("nan"),
            "levels": present, "n_levels": len(present), "levels_dropped": 0,
            "levels_registered": list(expected), "levels_missing": missing,
            "levels_without_measurable_growth": list(fit.groups_without_measurable_growth),
            "n_rows": int(fit.n_rows), "adequacy": fit.adequacy,
            "adequacy_reason": fit.adequacy_reason, "usable": bool(fit.usable),
            "fit": fit.as_record(),
            "estimator": "one maximum likelihood fit over every cell of the bank, with one common "
                         "exponent and one rate for each registered %s, so that a level whose round "
                         "produced nothing keeps its cells and its place in the fit rather than "
                         "being left out of a mean" % level_key}


def _log_scale_diagnostic(rows, cfg: P5Config, rho: float, rng: np.random.Generator) -> Dict[str, Any]:
    """The old estimator and the covariance-corrected one beside it, with the truncation named.

    This is not a route and decides nothing. It exists so that the size of the repair is visible in
    every run's own record rather than only in the test that measured it once: the log scale slope
    with the covariance term, the same slope without it, the Deming slope the runner used to report,
    and the fraction of the bank the transform had to throw away to produce any of them.
    """
    d = PO.log_scale_corrected_slope(rows, rho)
    boots = []
    for _ in range(max(0, int(cfg.bootstrap))):
        idx = rng.integers(0, len(rows), len(rows))
        boots.append(PO.log_scale_corrected_slope([rows[i] for i in idx], rho)["slope"])
    boots = np.asarray([b for b in boots if np.isfinite(b)], float)
    d["half_width"] = (cfg.z_interval * float(boots.std(ddof=1))) if boots.size >= 2 else float("inf")
    kept = [r for r in rows if r["increment"] > 0]
    if len(kept) >= 4:
        delta = _log_error_ratio(kept) if cfg.eiv_delta is None else cfg.eiv_delta
        d["deming_slope_as_previously_reported"] = _deming(
            np.log([r["available"] for r in kept]), np.log([r["increment"] for r in kept]), delta)
        d["eiv_delta"] = float(delta)
    else:
        d["deming_slope_as_previously_reported"] = float("nan")
        d["eiv_delta"] = float("nan")
    return d


def _manipulation_summary(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The manipulation records as they travel inside `routes`, without their own bank rows.

    The full record, rows and all, goes into the evidence bundle, which is where a reanalyst looks
    for it. `routes` is printed by the command line and read by every caller, and a route record
    carrying two hundred bank rows per manipulation is a summary nobody reads. Everything the
    judgement uses is kept.
    """
    keep = ("name", "manipulation", "documentation_failures", "measured", "source", "beta",
            "beta_se", "crossed", "fit", "reason")
    return [{k: r[k] for k in keep if k in r} for r in records]


def _no_identification(consistency: Dict[str, Any], control: Optional[Dict[str, Any]],
                       cfg: P5Config, reason: str,
                       manipulations: Sequence[Dict[str, Any]] = ()) -> Dict[str, Any]:
    """The identification judgement for a bank that produced no readable elasticity at all.

    The manipulations still travel in, so that a run which supplied one and then measured a bank it
    could not read reports what it supplied rather than reporting nothing. The label is INCONCLUSIVE
    either way, because there is no bank elasticity for a second channel to agree with.
    """
    return PI.judge(consistency=consistency, control=control, bank_elasticity=float("nan"),
                    bank_elasticity_se=float("nan"), manipulations=list(manipulations),
                    margin=cfg.route_margin, z=cfg.route_equivalence_z, bank_usable=False,
                    convention=PRED.convention_from_config(cfg)) | {"reason": reason}


def estimate_routes(bank: Dict[str, Any], cfg: P5Config, rng: Optional[np.random.Generator] = None,
                    manipulation_estimates: Sequence[Dict[str, Any]] = ()) -> Dict[str, Any]:
    """The two routes, each one fit to the paired readings of the whole bank in its own direction.

    NO ROW IS REMOVED HERE AND NO REGISTERED LEVEL IS EITHER, and that is the change of finding A5.
    Route one is the response to the capability state at fixed retention: one fit with one rate for
    each retention fraction and one common exponent, so the exponent is identified by the movement of
    the state inside each fraction. Route two is the response to retention at fixed state: one fit
    with one rate for each capability state, so its exponent is identified by the movement of the
    fraction inside each state. The two remain two different estimators of two different directions,
    they are not the same fit twice, and the standard error of each is its own sandwich rather than
    an arithmetic over subsets.

    THE ROUTES WERE PREVIOUSLY THE MEAN OF SEPARATE FITS TO THEIR LEVELS, over the levels that
    produced a usable exponent. A level produces none exactly when its own increment is not
    distinguishable from zero, so the mean was the row filter's own selection moved up to the level
    of the registered design, and it was invisible: a route reported from four of five levels looked
    exactly like a route reported from all five. The registration forbids it twice, in "Every run at
    a registered seed and retention level is included. No run is excluded on its outcome" and in "No
    retention level is added, dropped or reweighted after any increment is seen". The per-level fits
    are still computed and still reported, as diagnostics that decide nothing.

    WHAT THE TWO ROUTES ARE, SAID EXACTLY, AND WHAT THEY ARE NOT (finding A6). They are two
    regression directions through one bank, not two interventions. Under the general process
    a * f ** theta * U ** beta the state direction estimates the CAPABILITY elasticity beta and the
    retention direction estimates the RETENTION elasticity theta, so their agreement is a test of
    theta = beta and their gap is an estimate of theta - beta. Both are now reported as such: the
    crossed fit estimates the two elasticities together, the gap is its excess retention exponent
    with its own standard error, and the comparison with the margin is an interval equivalence test
    labelled CONSISTENT, INCONSISTENT or UNRESOLVED. It is never labelled IDENTIFIED. The
    identification judgement is a separate object, built by `arc_runner.p5_identification.judge`,
    and it reaches IDENTIFIED only on an independent capability manipulation supplied by the domain,
    which is why `manipulation_estimates` is an argument here: a reanalysis from a saved bundle
    passes back the manipulations that run measured, so the bundle re-scores to the same judgement.

    WHICH ELASTICITY THE RUN THEN USES is a separate question again and is settled in `run_p5`.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    model = _observation_model(bank)
    rows = [r for r in bank["rows"] if not r.get("control")]
    pooled_fit = PO.fit_paired(rows, model, cfg.z_interval)
    # H2's precision condition, UNCHANGED: the registered quantity is the fraction of cells whose
    # increment came back nonpositive, and a repair to the estimator is not a licence to restate a
    # registered threshold. What has changed is what happens to those cells afterwards: they are
    # counted here and then fitted with every other cell, where before they were counted and then
    # deleted. The model's own expectation of the same fraction is reported beside it, because
    # twelve cells of read noise can move the realised count either side of the ceiling and a reader
    # is entitled to see both; which of the two the ceiling should be read against is named as an
    # open decision in `arc_runner.p5_observation.expected_nonpositive_fraction`.
    unresolved = float(np.mean([r["increment"] <= 0 for r in rows])) if rows else float("nan")
    diagnostic = _log_scale_diagnostic(rows, cfg, model.read_correlation, rng)
    exact = all(float(r.get("read_sd", 0.0)) == 0.0 for r in rows) if rows else False
    correction = ("none: the ladder read is exact on both readings, so there is no read error to "
                  "carry into the increment's conditional mean and no attenuation to correct"
                  if exact else
                  "the paired observation model: the before read's error enters the increment's "
                  "conditional mean with the coefficient the subtraction gives it, and the after "
                  "read's error and the process variability enter its conditional variance")
    base = {"unresolved_fraction": unresolved, "nonpositive_fraction": unresolved,
            "expected_nonpositive_fraction": float(pooled_fit.expected_nonpositive_fraction),
            "pooled_fit": pooled_fit.as_record(), "observation_model": model.as_record(),
            "log_scale_diagnostic": diagnostic,
            # THE ESTIMATOR THE REGISTRATION NAMES, computed on these cells and reported beside the
            # one in use, with the amendment its demotion needs declared as unratified. A run record
            # that shows only the estimator actually used does not let a reader see that the
            # registered primary was superseded, and superseding it is the author's decision and not
            # this code's. See arc_runner.p5_observation.registered_log_scale_estimate.
            "registered_estimator": PO.registered_log_scale_estimate(rows, model.read_correlation),
            # `eiv_delta_used` is kept as the diagnostic's own ratio, because callers and saved
            # bundles have read the key since the first run. It no longer parameterises any route.
            "eiv_delta_used": diagnostic["eiv_delta"], "eiv_correction": correction}
    empty_consistency = {"label": PI.UNRESOLVED, "capability_elasticity": float("nan"),
                         "retention_elasticity": float("nan"),
                         "test": {"label": PI.UNRESOLVED, "reason": "the bank carries no readable "
                                                                   "elasticity to compare"}}
    if np.isfinite(unresolved) and unresolved > cfg.max_unresolved_fraction:
        why = ("%.0f%% of cells returned an increment the ladder could not resolve; the precision "
               "condition H2 fails" % (100 * unresolved))
        return dict(base, beta_state_route=float("nan"), beta_retention_route=float("nan"),
                    beta_pooled=float("nan"), route_gap=float("nan"), identification="INCONCLUSIVE",
                    route_agreement=PI.UNRESOLVED, route_consistency=empty_consistency,
                    identification_judgement=_no_identification(empty_consistency, None, cfg, why,
                                                                manipulation_estimates),
                    capability_manipulations=_manipulation_summary(manipulation_estimates),
                    premise="NOT REFUTED", reason=why)
    if not pooled_fit.usable:
        # The bank was measured and it carries no readable elasticity. This is reported as what it is,
        # with the reason the fit gives, and never as a coupling recovered from the cells that
        # happened to grow. Finding A5: a true zero-growth bank previously reported a coupling near
        # 0.28 fitted to the 57 per cent of cells whose noise ran upwards.
        why = pooled_fit.adequacy + ": " + pooled_fit.adequacy_reason
        return dict(base, beta_state_route=float("nan"), beta_retention_route=float("nan"),
                    beta_pooled=float("nan"), route_gap=float("nan"), identification="INCONCLUSIVE",
                    route_agreement=PI.UNRESOLVED, route_consistency=empty_consistency,
                    identification_judgement=_no_identification(empty_consistency, None, cfg, why,
                                                                manipulation_estimates),
                    capability_manipulations=_manipulation_summary(manipulation_estimates),
                    premise="NOT REFUTED", reason=why)
    # EACH ROUTE IS ONE FIT OVER EVERY CELL OF THE BANK. The state route puts one rate on each
    # retention fraction and one common exponent across them, so the exponent is identified by the
    # movement of the capability state inside each fraction, which is what the state route is. The
    # retention route puts one rate on each capability state, so its exponent is identified by the
    # movement of the retention fraction inside each state. The two remain two different estimators
    # of two different directions, and neither drops a registered level.
    #
    # WHAT THIS REPLACES AND WHY (finding A5, second refutation). A route used to be the MEAN of
    # separate fits to its levels, and a level whose own increment was not distinguishable from zero
    # produced no usable exponent and was left out of that mean. That is the row filter this finding
    # removed, moved up one level: a set removed exactly because its increment was smallest beside
    # its read noise, on the axis whose slope is being measured. Measured on a bank at the registered
    # replicate count whose round could not use the smallest retained fraction, the mean over the
    # four levels that grew reported 0.5274 with the precision condition passing and nothing saying a
    # level was absent. The registration forbids it twice: "Every run at a registered seed and
    # retention level is included. No run is excluded on its outcome", and "No retention level is
    # added, dropped or reweighted after any increment is seen".
    state_route_fit = PO.fit_paired(rows, model, cfg.z_interval, group_by="fraction")
    retention_route_fit = PO.fit_paired(rows, model, cfg.z_interval, group_by="state")
    route_state = _route_record(state_route_fit, "retention fraction", list(cfg.fractions))
    route_retention = _route_record(retention_route_fit, "capability state", list(cfg.states))
    # The per-level fits are kept as DIAGNOSTICS and decide nothing. They are what a reader looks at
    # to see which level carried what, and they are the estimates the route used to be a selected
    # mean of, so keeping them is how the difference stays visible in the run's own record.
    state_fits = [PO.fit_paired([r for r in rows if r["fraction"] == f], model, cfg.z_interval)
                  for f in cfg.fractions]
    retention_fits = [PO.fit_paired([r for r in rows if r["state"] == s], model, cfg.z_interval)
                      for s in cfg.states]
    b1, b2 = route_state["beta"], route_retention["beta"]
    se_state, se_pooled = route_state["se"], pooled_fit.beta_se
    control = _control_coupling(bank["rows"], cfg, model, pooled_fit)
    curvature = _curvature(rows, cfg, model)
    # THE CROSSED FIT, which is what makes the capability elasticity readable when the two directions
    # differ. The single-exponent pooled fit above assumes they do not: it puts one exponent on the
    # available capability, so a bank whose retention elasticity is 0.9 and whose capability
    # elasticity is 0.5 returns 0.60 from it and 0.60 is neither of them. The crossed fit returns
    # both, and its excess retention exponent is the gap the consistency check tests.
    crossed = PO.fit_paired(rows, model, cfg.z_interval, crossed=True)
    # THE INTERVAL CONVENTION THIS RUN IS UNDER, read once and handed to both equivalences. The
    # route gap and the manipulation's agreement are read at the same level and under the same
    # boundary rule as the final per-system comparison, because they are all the same convention and
    # a file that writes its own comparison out again is a file that can drift from it.
    convention = PRED.convention_from_config(cfg)
    consistency = PI.route_consistency(crossed, route_state, route_retention, cfg.route_margin,
                                       cfg.route_equivalence_z, convention)
    beta_cap = float(crossed.beta) if crossed.usable else float(pooled_fit.beta)
    se_cap = float(crossed.beta_se) if crossed.usable else float(se_pooled)
    judgement = PI.judge(consistency=consistency, control=control, bank_elasticity=beta_cap,
                         bank_elasticity_se=se_cap, manipulations=list(manipulation_estimates),
                         margin=cfg.route_margin, z=cfg.route_equivalence_z, bank_usable=True,
                         convention=convention)
    return dict(base, beta_state_route=b1, beta_retention_route=b2, beta_pooled=float(pooled_fit.beta),
                se_state_route=se_state, se_retention_route=route_retention["se"], se_pooled=se_pooled,
                negative_control=control, curvature=curvature, premise=curvature["premise"],
                route_gap=(abs(b1 - b2) if np.isfinite(b1) and np.isfinite(b2) else float("nan")),
                # The numerical consistency check and the identification judgement, separately, and
                # the second is never read off the first. Finding A6.
                route_agreement=consistency["label"], route_consistency=consistency,
                identification=judgement["label"], identification_judgement=judgement,
                capability_manipulations=_manipulation_summary(manipulation_estimates),
                crossed_fit=crossed.as_record(), beta_capability=beta_cap, se_capability=se_cap,
                retention_elasticity=float(crossed.retention_elasticity) if crossed.usable else float("nan"),
                state_route_fit=route_state, retention_route_fit=route_retention,
                # The per-level fits, labelled as the diagnostics they now are. A reader who compares
                # them with the route above sees the size of the selection the route no longer makes.
                state_route_fits=[f.as_record() for f in state_fits],
                retention_route_fits=[f.as_record() for f in retention_fits],
                per_level_fits_are_diagnostics=True,
                route_subsets_used={"state_route": route_state, "retention_route": route_retention,
                                    "note": "each route is one fit over every cell with one rate per "
                                            "registered level; no level is dropped, and the "
                                            "per-level fits beside it decide nothing"})


def measure_manipulation(m: PI.CapabilityManipulation, adapter: ModelAdapter, ladder: Ladder,
                         cfg: P5Config, rng: np.random.Generator,
                         bank_place_at_state: Any = None) -> Dict[str, Any]:
    """Measure the capability elasticity under a second, independent placement channel.

    THE ORDER HERE IS THE POINT. The documentation is checked BEFORE anything is run, so a
    manipulation that has not written down its exclusion restriction, or that has not been asked
    whether it shares a nuisance pathway with the bank, costs nothing and is reported as
    inadmissible. An inadmissible manipulation that had already been paid for would be a temptation
    to admit it.

    AND THE BANK'S OWN LOADER IS ONE OF THE THINGS CHECKED THERE (a defect found in review). A
    manipulation carrying no adapter falls back to the bank's adapter below, and a manipulation whose
    `place_at_state` IS the bank's `place_at_state` places its cells exactly as the bank placed its
    own. Such an object differs from the bank in nothing but two declared booleans, and it reached
    IDENTIFIED. Whether a second channel truly escapes the bank's nuisance is a domain declaration
    and is treated as one; whether it is literally the same callable is a fact about this run, so
    `bank_place_at_state` is passed here and the comparison travels into the record. Where the caller
    does not pass it the record says the comparison was not performed, and
    `arc_runner.p5_identification.failures_in_record` reads that as a failure, because an
    identification whose independence was never checked has not been checked.

    A manipulation may carry its own estimate instead of a loader, for a domain that measured the
    elasticity under the second channel elsewhere. It is used as given, with its own standard error,
    and the record says which of the two it was, and the estimate must carry a written provenance:
    without one, a number typed into a mapping and a number measured elsewhere are the same object.

    Which fit the manipulation's own bank gets is decided by what its design varies. Where it crosses
    states with retention fractions the crossed fit is used, so its capability elasticity is the
    exponent in the state and not a mixture. Where it varies capability alone, at one retention
    fraction, the single-exponent fit already IS the capability elasticity, because with the fraction
    held constant the two exponents cannot be separated and do not need to be.
    """
    description = m.as_record(bank_place_at_state, adapter)
    record: Dict[str, Any] = {"name": m.name, "manipulation": description,
                              "documentation_failures": PI.failures_in_record(description)}
    if record["documentation_failures"]:
        record["measured"] = False
        record["reason"] = ("refused before any call was made: %s"
                            % "; ".join(record["documentation_failures"]))
        return record
    if m.estimate:
        record.update({"measured": True, "source": "supplied with the manipulation",
                       "beta": float(m.estimate.get("beta", float("nan"))),
                       "beta_se": float(m.estimate.get("beta_se", float("nan"))),
                       "fit": dict(m.estimate)})
        return record
    sub = replace(cfg, states=tuple(m.states) if m.states else tuple(cfg.states))
    bank = run_bank(m.adapter or adapter, ladder, sub, rng, m.place_at_state)
    rows = [r for r in bank["rows"] if not r.get("control")]
    model = _observation_model(bank)
    crossed = len({float(r["fraction"]) for r in rows}) >= 2 and len({r["state"] for r in rows}) >= 2
    fit = PO.fit_paired(rows, model, cfg.z_interval, crossed=crossed)
    record.update({"measured": True,
                   "source": "its own bank, placed by this manipulation's loader",
                   "beta": float(fit.beta) if fit.usable else float("nan"),
                   "beta_se": float(fit.beta_se) if fit.usable else float("nan"),
                   "crossed": bool(crossed), "fit": fit.as_record(), "bank": bank})
    if not fit.usable:
        record["reason"] = fit.adequacy + ": " + fit.adequacy_reason
    return record


def predicted_exponent(beta: float, a: float, U0: float, r1: float, r2: float,
                       checkpoints: Optional[Sequence[int]] = None) -> float:
    """The finite-window exponent the complete solution predicts, never the asymptote, and computed
    with THE SAME ESTIMATOR the observed side uses. The first runner battery refuted a true world in
    33 runs of 100 because the prediction was a two-point endpoint slope while the observation was a
    log-log fit over four checkpoints; for a shifted power law those differ by more than the margin.
    The prediction now generates the complete solution at the same checkpoints and applies the same fit."""
    if checkpoints is None:
        return float(ci.endpoint_slope(U0, a, beta, r1, r2))
    R = np.asarray([d for d in checkpoints if d >= r1], float)
    # the complete solution anchored at the calibration depth: U(R) = (U0^(1-beta) + a(1-beta)(R - r1))^(1/(1-beta))
    if not all(np.isfinite(v) for v in (U0,a,beta,r1,r2)) or U0 <= 0:
        return float("nan")
    with np.errstate(over="ignore",invalid="ignore"):
        if abs(1.0-beta) > 1e-9:
            base=U0**(1.0-beta)+a*(1.0-beta)*(R-r1)
            if not np.all(base>0): return float("nan")
            U=base**(1.0/(1.0-beta))
        else:
            U=U0*np.exp(a*(R-r1))
    return loglog_slope(R, U)


def calibrate_rate(depths: Sequence[int], scores: Sequence[float], beta: float) -> float:
    """Under the framework's solution, score ** (1 - beta) is linear in depth with slope a (1 - beta)."""
    values = np.asarray(scores, float)
    if not np.all(np.isfinite(values) & (values > 0)) or not np.isfinite(beta):
        return float("nan")
    if abs(1.0 - beta) <= 1e-9:
        return float(np.polyfit(np.asarray(depths, float), np.log(values), 1)[0])
    y = values ** (1.0 - beta)
    slope = np.polyfit(np.asarray(depths, float), y, 1)[0]
    return float(slope / (1.0 - beta))


def _endpoint_half_width(t, beta_hat: float, beta_se: float, U0: float, r1: float,
                         cfg: P5Config) -> float:
    """The width the runner used to report: the coupling at the ends of its interval, the rate
    re-calibrated at each end from the same readings, and each end clipped into the domain.

    It decides nothing and is kept as a diagnostic, so that the difference between it and the joint
    propagation is visible in every run's own record rather than only in the test that measured it
    once. It is the arithmetic finding A7 names, reproduced exactly, clip included.
    """
    if not np.isfinite(beta_se):
        return float("nan")
    lo, hi = cfg.coupling_domain
    ends = []
    for b in (beta_hat - cfg.z_interval * beta_se, beta_hat + cfg.z_interval * beta_se):
        b = min(max(b, lo), hi)
        ends.append(predicted_exponent(b, calibrate_rate(t.depths(), t.scores(), b), U0, r1,
                                       cfg.window_end, cfg.checkpoints))
    return float(abs(ends[1] - ends[0]) / 2.0)


def run_heldout(adapter: ModelAdapter, ladder: Ladder, cfg: P5Config, rng: np.random.Generator,
                systems: Sequence[str], start_for, beta_hat: float, man: Dict[str, Any],
                sealed_by: str = "arc_runner", beta_se: float = float("nan"),
                anchor=None, bundle=None, attestation: Optional[Dict[str, Any]] = None
                ) -> Dict[str, Any]:
    """Calibrate, checkpoint, SEAL, ANCHOR, WRITE, then generate. The order is the content of ruling
    27, extended by finding A8: the seal and its receipt reach the disk before the first continuation
    exists, so that a run which stops between the seal and the last checkpoint still leaves a
    commitment somebody else can check. A seal that lives only in memory until the run finishes is a
    commitment only if the run finishes.

    THE SEALED PREDICTION CARRIES THE WHOLE OF ITS UNCERTAINTY (finding A7). H2's precision condition
    needs a width, and the width this used to compute moved the COUPLING to the ends of its interval,
    re-calibrated the rate at each end from THE SAME calibration readings, and clipped each end
    silently into [0, 0.95]. Three things were wrong with that. The calibration readings entered as
    though they were exact, when they are read from four checkpoints and are the binding precision of
    the whole comparison. The starting state IS one of those readings and never moved at all. And the
    clip reported a narrower interval than the bank supported while hiding that the interval had
    reached the domain boundary. `arc_runner.p5_prediction.propagate_prediction` resamples the
    coupling, every calibration reading and the starting state JOINTLY, so that a calibration window
    read high raises the starting state and the fitted rate together and the cancellation between
    them is the one the world has; it counts the draws that fall outside the permitted coupling
    domain rather than moving them onto its edge; and it reports the interval at the same level the
    comparison reads, because a difference interval that adds a ninety-five per cent half width to a
    ninety per cent one is an interval at no level at all. The old endpoint width is kept beside it
    under its own name, as a diagnostic, so that the size of the repair is visible in every run.

    The panel is recorded as ASSIGNED, and it is the denominator the aggregate rule is stated over."""
    cal, preds, checkpoints = {}, {}, {}
    d_cal = list(cfg.calibration_depths)
    convention = PRED.convention_from_config(cfg)
    run_seed = int((man.get("config") or {}).get("seed", 0) or 0)
    for i, sysname in enumerate(systems):
        t = run_trajectory(adapter, ladder, start_for(sysname), cfg.task, d_cal, 1.0, rng, system=sysname,
                           start_depth=0, reads=cfg.cal_reads)
        a_hat = calibrate_rate(t.depths(), t.scores(), beta_hat)
        U4 = t.at(d_cal[-1]).reading.score
        read_sds = t.read_sds()
        cal[sysname] = {"a_hat": a_hat, "U_at_cal": U4, "scores": t.scores(), "depths": t.depths(),
                        # The read error of each calibration score, recorded with the scores. Without
                        # it a later analyst cannot rebuild the prediction's interval from the bundle,
                        # and this run could not build it either.
                        "read_sds": read_sds,
                        "start_state_read_sd": read_sds[-1] if read_sds else float("nan"),
                        "read_model": SAMPLING.read_model_record(ladder),
                        # EVERY CALIBRATION CHECKPOINT, WITH THE ARTEFACT IT READ. A saved
                        # calibration holding only the fitted summaries cannot be recounted, which
                        # is the whole reason an evidence bundle exists.
                        "checkpoints": [{"depth": c.depth, "artefact": c.artefact,
                                         "reading": dict(c.reading.__dict__)} for c in t.checkpoints]}
        prediction = PRED.propagate_prediction(
            beta_hat=beta_hat, beta_se=beta_se, depths=t.depths(), scores=t.scores(),
            score_sds=read_sds, r1=d_cal[-1], window_end=cfg.window_end,
            checkpoints=cfg.checkpoints, calibrate=calibrate_rate, predict=predicted_exponent,
            convention=convention, seed=run_seed * 1000003 + i)
        point = prediction["predicted_exponent"]
        half = prediction["half_width"]
        preds[sysname] = {"predicted_exponent": point, "predicted_half_width": half, "beta_se": beta_se,
                          "beta_hat": beta_hat, "a_hat": a_hat, "window": [d_cal[-1], cfg.window_end],
                          "estimator": "log-log OLS over the registered checkpoints, identical on both sides",
                          # The whole propagation travels inside the SEALED object, so that the
                          # interval a verdict is read against is the one that was committed to and
                          # not one recomputed afterwards from a configuration that may have moved.
                          "prediction": prediction,
                          "endpoint_half_width_diagnostic": _endpoint_half_width(
                              t, beta_hat, beta_se, U4, d_cal[-1], cfg)}
        checkpoints[sysname] = t.at(d_cal[-1]).artefact
        if bundle is not None:
            # The calibration window is paid for before the seal exists, so it is recorded before the
            # seal too. A run that stops inside the calibration has spent money and must leave what it
            # bought, which the seal file cannot carry because there is no seal yet.
            bundle.record_progress("calibration", {"system": sysname, "calibration": cal[sysname]})
    # The seal, the attestation of what was unseen, and the anchor receipt, all before any
    # continuation exists, and then to disk.
    M.seal_predictions(man, preds, sealed_by, anchor=anchor, attestation=attestation)
    if bundle is not None:
        bundle.write_seal(man)
    fitted = {}
    for sysname in systems:
        runs, head_ok = [], True
        for replicate_id in range(max(1, cfg.replicates)):
            cont = run_trajectory(adapter, ladder, checkpoints[sysname], cfg.task, list(cfg.checkpoints), 1.0, rng,
                                  system=sysname, start_depth=d_cal[-1], reads=cfg.heldout_reads)
            ok = all(ladder.headroom_ok(c.reading) for c in cont.checkpoints)
            head_ok = head_ok and ok
            runs.append({"replicate_id": replicate_id, "depths": cont.depths(), "scores": cont.scores(),
                         "checkpoints": [{"depth": c.depth, "artefact": c.artefact,
                                          "reading": c.reading.__dict__} for c in cont.checkpoints],
                         "fitted_exponent": loglog_slope(cont.depths(), cont.scores()), "headroom_ok": ok})
        arr = np.asarray([r["fitted_exponent"] for r in runs if np.isfinite(r["fitted_exponent"])], float)
        mean = float(arr.mean()) if arr.size else float("nan")
        se = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size >= 2 else float("nan")
        fitted[sysname] = {"replicates": runs, "n_replicates": int(arr.size), "fitted_exponent": mean,
                           "fitted_se": se, "headroom_ok": head_ok}
        if bundle is not None:
            # ONE LINE PER SYSTEM, AS THE CONTINUATION PROCEEDS. This is the acceptance case that the
            # seal file does not cover: a stop part-way through the held-out panel leaves the systems
            # already continued, with their replicate series and their readings, rather than a
            # commitment with nothing behind it.
            bundle.record_progress("heldout", {"system": sysname, "fitted": fitted[sysname]})
    # THE ASSIGNED PANEL, RECORDED (finding A7). The aggregate rule is stated over the systems that
    # were assigned, not over the ones that survived, so the list of assigned systems has to reach
    # the verdict. It is also written into the manifest's configuration by `run_p5`, which puts it
    # inside the sealed specification hash: the denominator is a pre-run commitment and a panel that
    # shrank after the seal is a custody failure rather than a smaller denominator.
    return {"calibration": cal, "sealed_predictions": preds, "fitted": fitted,
            "assigned_systems": list(systems),
            "interval_convention": convention.as_record()}


# The t table moved to `arc_runner.p5_prediction`, beside the convention that names the level it is
# the quantile of, and is re-exported here because `arc_runner.p16_components` and every caller since
# the first run have imported it from this module. One table, not three.
t95 = PRED.t95
_T95 = PRED._T95


def _assigned_panel(man: Dict[str, Any], heldout: Dict[str, Any]) -> Tuple[List[str], str, bool]:
    """The held-out systems this run was assigned, WHERE THAT LIST CAME FROM, and whether it was
    frozen before the run.

    The sealed configuration is asked first: `run_p5` writes the panel into the manifest's
    configuration, which is inside the sealed specification hash, so a panel that changed after the
    seal is a custody failure rather than a smaller denominator. The run's own record and then the
    systems that produced a fit are the fallbacks, so that a bundle written before the panel was
    sealed still re-scores, on the widest panel it can evidence.

    THE THIRD SOURCE IS THE SURVIVORS, AND IT IS SAID SO (finding A7). Reading the panel off the
    systems that produced a fit is exactly the population the finding forbids deciding on: every one
    of them is then counted, the arithmetic looks correct, and the denominator is still the set the
    run selected after seeing the data. It is kept, because a bundle written before the panel was
    sealed should still re-score and a reader should still see its counts, and it is returned with
    the provenance and with `frozen` false, so that `arc_runner.p5_prediction.aggregate` withholds
    the proposition-level word rather than the whole reading. Nothing on the `run_p5` path reaches
    it: that path writes the panel into the manifest's configuration and into the held-out record
    before any verdict is read.
    """
    sealed = (man.get("config") or {}).get("assigned_heldout_systems")
    if sealed:
        return ([str(x) for x in sealed],
                "the sealed configuration, inside the specification hash", True)
    if heldout.get("assigned_systems"):
        return ([str(x) for x in heldout["assigned_systems"]],
                "the run's own held-out record, which names the panel it was given", True)
    return ([str(x) for x in heldout.get("fitted", {})],
            "INFERRED AFTER THE FACT from the systems that produced a fit, because neither the "
            "sealed configuration nor the held-out record names a panel", False)


def _per_system_verdicts(heldout: Dict[str, Any], cfg: P5Config,
                         convention: PRED.IntervalConvention,
                         assigned: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, Any]]:
    """One reading per system, with the whole of both uncertainties in the comparison.

    The order of the refusals is the order of the questions. A system that reached the ladder ceiling
    was never measured on the window the prediction is about. A prediction whose coupling interval
    reaches the permitted domain is one this runner cannot represent, so it is not scored against.
    A prediction whose uncertainty could not be propagated at all is not treated as exact, which is
    what a non-finite half width used to mean here. H2's precision condition then asks whether the
    prediction is sharp enough to be worth comparing. Only after all four does the difference get
    read, and it is read with the prediction's uncertainty in it as well as the observation's.

    THE TABLE IS THE PANEL, NOT THE SURVIVORS (finding A7). The loop below runs over the systems that
    produced a fit, so an assigned system that never reported at all appeared nowhere in it. The
    aggregate counted such a system in the denominator, which is the arithmetic the finding asks for,
    but the per-system table is the only place a reader sees a system's STATUS, and a system missing
    from that table has been dropped wherever anybody actually looks. Every assigned system is
    therefore given a row here, and a system that was scored without being on the sealed panel is
    given one too, marked as not counted: the population may not shrink and it may not grow.
    """
    per: Dict[str, Dict[str, Any]] = {}
    for s, f in heldout["fitted"].items():
        sp = heldout["sealed_predictions"].get(s, {})
        pred_rec = sp.get("prediction") or {}
        if not f.get("headroom_ok", False):
            per[s] = {"verdict": PRED.NOT_EVALUABLE,
                      "reason": "ladder ceiling reached before the final checkpoint",
                      "counted_in_denominator": True}
            continue
        domain = pred_rec.get("coupling_domain") or {}
        if domain and not domain.get("within_domain", True):
            per[s] = {"verdict": PRED.UNRESOLVED, "reason": domain.get("reason", ""),
                      "coupling_domain": domain, "counted_in_denominator": True}
            continue
        pw = sp.get("predicted_half_width", float("nan"))
        # A NEGATIVE WIDTH IS NOT A NARROW ONE. It was compared with the margin as though it were,
        # so a half width of minus one passed the precision condition more easily than any real
        # interval and the system was scored on a prediction whose uncertainty is not a number.
        if not np.isfinite(pw) or pw < 0:
            per[s] = {"verdict": PRED.UNRESOLVED, "counted_in_denominator": True,
                      "predicted_half_width": pw,
                      "reason": ("the prediction's uncertainty could not be propagated (%s), and an "
                                 "unknown width is not read as a zero one, and neither is a negative "
                                 "one" % (pred_rec.get("reason") or "no interval available")),
                      "prediction": pred_rec}
            continue
        if not convention.inside(2.0 * pw, cfg.margin):
            # H2 in precision: the interval on the predicted exponent must be narrower than the
            # margin, else no comparison is read for this system. The width is now the joint one and
            # the comparison is strict, so a predicted interval exactly as wide as the margin does
            # not pass it.
            per[s] = {"verdict": PRED.UNRESOLVED,
                      "reason": "predicted exponent's interval (width %.3f) is not narrower than the margin"
                                % (2.0 * pw),
                      "predicted_half_width": pw, "prediction": pred_rec,
                      "counted_in_denominator": True}
            continue
        if not np.isfinite(pred_rec.get("se", float("nan"))):
            # A BUNDLE SEALED BEFORE THE JOINT PROPAGATION EXISTED. Its sealed prediction carries the
            # endpoint half width and nothing else, and the calibration window's read error was never
            # recorded, so the prediction's uncertainty cannot be reconstructed from it: the endpoint
            # width is a LOWER BOUND on it and using a lower bound as the value is the defect finding
            # A7 names, arriving through a different door. Such a system is unresolved and says so,
            # rather than being compared as though the prediction were exact.
            per[s] = {"verdict": PRED.UNRESOLVED, "counted_in_denominator": True,
                      "predicted_half_width": pw, "prediction": pred_rec,
                      "reason": ("this sealed prediction carries no propagated uncertainty, only the "
                                 "endpoint half width, which is a lower bound on it: the comparison "
                                 "is refused rather than read against a prediction treated as exact")}
            continue
        diff = PRED.difference_interval(observed=f["fitted_exponent"],
                                        observed_se=f.get("fitted_se", float("nan")),
                                        predicted=sp["predicted_exponent"],
                                        predicted_se=pred_rec.get("se", float("nan")),
                                        n_replicates=int(f.get("n_replicates", 1)),
                                        convention=convention)
        reading = convention.clearance(diff["difference"], diff["half_width"], cfg.margin)
        per[s] = {"verdict": reading["verdict"], "difference": diff["difference"],
                  "interval_half_width": diff["half_width"],
                  "n_replicates": f.get("n_replicates", 1), "reason": reading["reason"],
                  "difference_interval": diff, "prediction": pred_rec,
                  "predicted_half_width": pw, "counted_in_denominator": True}
    if assigned is None:
        return per
    panel = [str(s) for s in assigned]
    for s in panel:
        if s not in per:
            # ASSIGNED AND NEVER REPORTED. Not absent: a system that produced no fit at all is a
            # system the run could not evaluate, and it keeps its place in the denominator under that
            # name. The reason says what happened rather than leaving a reader to infer it from a
            # gap between two lists.
            per[s] = {"verdict": PRED.NOT_EVALUABLE, "counted_in_denominator": True, "assigned": True,
                      "never_reported": True,
                      "reason": ("this system was assigned to the sealed panel and produced no fitted "
                                 "exponent at all, so it could not be evaluated; it keeps its place "
                                 "in the denominator as NOT EVALUABLE and is never dropped from it")}
    for s, rec in per.items():
        if s in panel:
            rec["assigned"] = True
            continue
        rec["assigned"] = False
        rec["counted_in_denominator"] = False
        rec["reason"] = ("this system is not on the sealed panel, so it is reported and never "
                         "counted: the denominator is the panel that was assigned. Its reading was: "
                         + str(rec.get("reason", "")))
    return per


def verdicts(man: Dict[str, Any], routes: Dict[str, Any], heldout: Dict[str, Any], cfg: P5Config,
             ladder: Optional[Ladder] = None) -> Dict[str, Any]:
    """`ladder` is the LIVE ladder where the caller has one, and it goes into the custody check on the
    deciding path so that the recorded verifier binding is recomputed rather than read back. A
    re-scoring from a saved bundle passes none, and the custody report says the check was not
    performed rather than saying it passed."""
    # `require_reportable`, not `require_scoreable`: a demonstration run keeps its verdicts, because
    # recovering a known coupling is the proof the pipeline works, and carries the sentence saying
    # they are simulated recoveries. Reading them at proposition level goes through
    # `M.require_scoreable`, which refuses every mode but the deciding one.
    label = M.require_reportable(man)
    # ON THE DECIDING PATH THE CUSTODY CHECK RUNS HERE, WITH THE LIVE OBJECTS IN HAND (finding A8).
    # This is the only place that holds the predictions actually being scored and the configuration
    # actually used, so it is the only place that can tell whether either moved after the seal. A
    # confirmatory run whose predictions were edited refuses to produce a verdict at all rather than
    # producing one nobody can trace to a commitment.
    custody_report = None
    if M.mode_of(man) is M.ExecutionMode.CONFIRMATORY:
        custody_report = M.require_scoreable(man, predictions=heldout.get("sealed_predictions"),
                                             config=cfg, ladder=ladder)
    # AND THE RECORD ITSELF IS CHECKED ON EVERY PATH, not only the deciding one. The reference gate
    # read a flag and the presence of a seal, so it would print readings from a run whose sealed
    # predictions had been edited afterwards and say nothing about it. A demonstration whose record
    # has moved is not a demonstration of anything. It runs AFTER the custody chain because the
    # custody chain is the more specific diagnosis wherever it applies: a deciding run should be told
    # which link failed, not merely that the record no longer agrees with itself.
    M.require_diagnostic(man, heldout.get("sealed_predictions"))
    convention = PRED.convention_from_config(cfg)
    # THE DENOMINATOR IS THE ASSIGNED PANEL, NOT THE SURVIVORS (finding A7). A system that failed the
    # headroom rule used to be labelled NOT EVALUABLE and then skipped before the aggregate was
    # counted, so a panel of five of which three reached the ladder ceiling could be declared
    # supported by the two that did not. That decides the proposition on a population the run
    # selected after seeing the data, and the candidate's weight is on the panel that was assigned.
    # The assigned list is read from the sealed configuration first, because that is the frozen
    # commitment; the run's own record and finally the systems that produced a fit are the fallbacks
    # for a bundle written before the panel was sealed. The last of those three is the SURVIVORS, so
    # the source and whether it was frozen travel with the list: a denominator inferred after the run
    # is counted and reported, and the proposition-level word is withheld from it.
    assigned, panel_source, panel_frozen = _assigned_panel(man, heldout)
    # The panel is read BEFORE the table is built, because the table is written over the panel: an
    # assigned system that never reported gets its row as NOT EVALUABLE, and a system scored without
    # being on the panel gets one saying it is not counted.
    per = _per_system_verdicts(heldout, cfg, convention, assigned)
    premise = routes.get("premise", "NOT REFUTED")
    agg = PRED.aggregate(per, assigned, premise, denominator_frozen=panel_frozen,
                         denominator_source=panel_source)
    pred = agg["result"]
    # HOW MANY OF THE PANEL WERE LOST TO THE LADDER CEILING, counted from the table and reported as
    # its own figure. The aggregate already refuses to decide a panel with nothing evaluable in it,
    # which is the arithmetic; this is the reason, and a reader who sees NOT EVALUABLE is entitled to
    # know whether the window was unmeasurable or the systems never reported at all.
    headroom_failures = sum(1 for rec in per.values()
                            if rec.get("counted_in_denominator")
                            and "ceiling" in str(rec.get("reason", "")))
    return {"PREDICTION": pred, "IDENTIFICATION": routes["identification"], "PREMISE": premise, "per_system": per,
            # The denominator the panel result was decided on, printed beside it, with every assigned
            # system accounted for. A reader of the table can add the four counts and get the panel.
            "panel": agg,
            "interval_convention": convention.as_record(),
            # THE NUMERICAL CONSISTENCY OF THE TWO REGRESSION DIRECTIONS, reported beside the
            # identification and never as it. A reader of the table who sees only one word cannot
            # tell a bank whose two slopes agree from a study that intervened on capability twice,
            # and those are different claims. Finding A6.
            "ROUTE_AGREEMENT": routes.get("route_agreement"),
            "route_consistency": routes.get("route_consistency"),
            "identification_judgement": routes.get("identification_judgement"),
            "capability_elasticity": routes.get("beta_capability"),
            "retention_elasticity": routes.get("retention_elasticity"),
            "execution_mode": man.get("execution_mode"), "interpretation": label,
            "custody": custody_report,
            "scoreable_at_proposition_level": bool(man.get("scoreable_at_proposition_level")),
            "negative_control": routes.get("negative_control"), "curvature": routes.get("curvature"),
            "assigned_systems": agg["assigned"], "headroom_failures": headroom_failures,
            # WHETHER THE OBSERVATION MODEL WAS ADEQUATE TO THE BANK, in the verdict block and not
            # only in the routes, because a reader of the table is entitled to see it there. A bank
            # whose increment is not distinguishable from zero, or whose rate is negative, has not
            # produced a coupling, and a table that shows only the pair does not say so. Finding A5.
            "observation": {k: (routes.get("pooled_fit") or {}).get(k)
                            for k in ("adequacy", "adequacy_reason", "n_rows", "n_nonpositive",
                                      "expected_nonpositive_fraction", "process_sd", "dispersion",
                                      "state_homogeneity", "usable")},
            "nonpositive_cells_kept": (routes.get("pooled_fit") or {}).get("n_nonpositive"),
            # WHETHER EVERY REGISTERED LEVEL IS IN EVERY ROUTE, in the verdict block and not only
            # buried in the routes (finding A5). A route reported as the mean of the levels that
            # happened to grow looked exactly like a route reported from all of them, and no reader
            # of the table could tell. `levels_dropped` is zero by construction now and is printed
            # so that it is a thing a reader can check rather than a thing they have to trust;
            # `levels_without_measurable_growth` names the levels whose own rate came back
            # indistinguishable from zero, which is a fact about the bank and never a reason to
            # leave a level out.
            "route_levels": {
                name: {k: (routes.get(key) or {}).get(k)
                       for k in ("n_levels", "levels_dropped", "levels_missing",
                                 "levels_without_measurable_growth", "n_rows")}
                for name, key in (("state_route", "state_route_fit"),
                                  ("retention_route", "retention_route_fit"))},
            # The estimator the registration names, and the declaration that superseding it has not
            # been ratified by an amendment. A reader of the verdict is entitled to that on the same
            # page as the verdict.
            "registered_estimator": routes.get("registered_estimator"),
            "note": "the pair is the result; a route disagreement never converts a correct prediction into inconclusive; "
                    "a refuted premise (H3) is reported in those words and withholds SUPPORTED. "
                    "The panel result is decided on the ASSIGNED held-out systems: a system that "
                    "could not be evaluated keeps its place in the denominator as NOT EVALUABLE and "
                    "is never dropped from it. Each system's difference interval carries the "
                    "prediction's propagated uncertainty as well as the observed replicate spread, "
                    "and the boundary is strict clearance. "
                    "ROUTE_AGREEMENT is a numerical consistency check between two regression "
                    "directions through one bank and is never an identification; IDENTIFICATION "
                    "reaches IDENTIFIED only on an independent capability manipulation."}


def run_p5(adapter: ModelAdapter, ladder: Ladder, cfg: P5Config, seed: int, place_at_state, start_for,
           heldout_systems: Sequence[str], pilot: bool = False, heldout_adapter: Optional[ModelAdapter] = None,
           mode: Any = None, confirmatory_inputs: Optional[MODE.ConfirmatoryInputs] = None,
           bundle: Any = None, anchor=None, attestation: Optional[Dict[str, Any]] = None,
           manipulations: Sequence[PI.CapabilityManipulation] = ()) -> Dict[str, Any]:
    """`heldout_adapter` lets a battery place the held-out panel in a world the bank's coupling does
    not govern, which is the charter's REGIME CHANGE rival and the world in which P5 must refute.

    `mode` states which kind of run this is and resolves fail closed: silence is a demonstration,
    never the deciding path. A CONFIRMATORY run is gated here, at the top, because the next statement
    after the gate is `run_bank`, and `run_bank` is the first paid call. The inputs the runner already
    holds (the ladder, both loaders, the states the bank will place, the configuration) are written
    into the gate's copy rather than read from the caller's, so the object cannot describe a setup
    other than the one about to run.

    `manipulations` are the domain's independent capability interventions, being second ways of
    placing the capability state that do not run through retention. They are what the identification
    judgement needs and what the bank cannot contain: two regression directions through one bank are
    two slopes through the same cells, however closely they agree. With none supplied the judgement
    is NOT ESTABLISHED and says so; it never reads IDENTIFIED off the numerical agreement of the two
    routes. Finding A6.

    `attestation` is the named party's statement that the material deciding these predictions was
    unseen when they were fixed. It is a record and not a callable: the runner asks nobody for it,
    composes none, and attaches its own labelled placeholder where none is supplied, which the
    deciding path then refuses at proposition level. Finding A8's third requirement.

    `bundle` is a directory (or an `arc_runner.custody.EvidenceBundle`) that receives the complete
    evidence: the manifest before the bank runs, the seal and its receipt at the moment of sealing,
    and the full record at the end. `anchor` is the operator's anchoring service; on the deciding path
    the gate has already refused a run that has none, and off it an absent service means a receipt
    labelled a mock. Finding A8: the previous `--out` wrote a manifest, so the bank rows, the reads,
    the replicate series and the provider metadata existed only inside one process.

    THE CONFIGURATION IS CHECKED BEFORE ANY OF THIS, because every failure below belongs to a run that
    has already been paid for. A duplicated bank cell, a checkpoint grid that does not reach the
    declared horizon, a calibration that does not end where the held-out window begins, a panel with a
    repeat in it: each produces something that looks like a result, and each is free to detect here. A
    remote endpoint is refused in the same place, at the library boundary rather than at the command
    line, because this package has no released assay and no released negative-control implementation
    to point a real system at, and a second caller can walk around a command line.
    """
    if not heldout_systems or len(set(heldout_systems)) != len(heldout_systems):
        raise ValueError("held-out systems must be nonempty and unique")
    if cfg.reps < 1 or cfg.replicates < 1 or cfg.bootstrap < 2:
        raise ValueError("replication and bootstrap counts are invalid")
    if len(set(cfg.states)) != len(cfg.states) or len(set(cfg.fractions)) != len(cfg.fractions):
        raise ValueError("bank cells are duplicated")
    if not cfg.checkpoints or not cfg.calibration_depths:
        raise ValueError("calibration and held-out checkpoints are required")
    if list(cfg.checkpoints) != sorted(set(cfg.checkpoints)) or cfg.checkpoints[-1] != cfg.window_end:
        raise ValueError("checkpoint grid must be unique, ordered and reach the declared horizon")
    if cfg.calibration_depths[-1] != cfg.checkpoints[0]:
        raise ValueError("calibration end must equal first held-out checkpoint")
    if any(getattr(a, "uses_remote_endpoint", False) for a in (adapter, heldout_adapter)):
        raise M.InstrumentNotReleased(
            "real P5 collection lacks a released assay and negative-control implementation")
    m = MODE.resolve(mode, pilot)
    pilot = bool(pilot or m is MODE.ExecutionMode.PILOT)
    inputs_record = None
    if m is MODE.ExecutionMode.CONFIRMATORY:
        checked = confirmatory_inputs if confirmatory_inputs is not None else MODE.ConfirmatoryInputs()
        checked = replace(checked, ladder=ladder, place_at_state=place_at_state, start_for=start_for,
                          states=tuple(cfg.states), config=cfg,
                          # AND THE SYSTEMS THEMSELVES, which the gate had never been shown. The
                          # remote-endpoint refusal above catches a real endpoint, on the ground that
                          # no released assay exists to point one at; nothing caught a LOCAL object
                          # that models the world, so the only thing this gate could ever accept was
                          # a simulation, and the manifest then wrote `simulated: false` because that
                          # field is derived from the mode word. Both adapters go in, because the
                          # held-out panel decides the verdict as much as the bank does.
                          observing_systems=tuple(a for a in (adapter, heldout_adapter)
                                                  if a is not None),
                          anchor=anchor if anchor is not None else checked.anchor,
                          attestation=(attestation if attestation is not None
                                       else checked.attestation))
        # AND THE STORE THE LOADERS ACTUALLY READ (finding A9). The gate validates whichever store the
        # inputs name; the bank places from whichever store the loaders close over. When the caller
        # named no store the loaders' own store is written in here, so that the object cannot describe
        # a bank other than the one about to run. When the caller named a DIFFERENT store the
        # disagreement is deliberately left standing, because the gate refuses it in its own words:
        # silently preferring one of the two would hide a setup whose author believed something false
        # about it.
        if checked.checkpoint_store is None:
            checked = replace(checked, checkpoint_store=MODE.loader_store(place_at_state, start_for))
        # AND THE SECOND CHANNELS GO THROUGH THE GATE WITH THE FIRST ONE (a defect found in review).
        # A manipulation is now the ONLY route to IDENTIFIED, and nothing in the deciding gate had
        # ever seen one: its loader escaped the placeholder-loader refusal that finding A9 exists
        # for, so a deciding run could reach IDENTIFIED from a channel that manufactured its
        # artefacts from a number. The manipulations are written into the gate's copy rather than
        # read from the caller's, for the same reason the loaders and the states are.
        checked = replace(checked, manipulations=tuple(manipulations))
        MODE.require_confirmatory_inputs(checked)
        inputs_record = checked.as_record()
        anchor = checked.anchor
        attestation = checked.attestation
    bundle = CUSTODY.as_bundle(bundle)
    reads = CUSTODY.attach_read_log(ladder) if bundle is not None else None
    if bundle is not None:
        # The live read log is handed to the bundle so that every progress line carries the readings
        # taken since the previous one. Without it the reads accumulate in memory until the final
        # write and a run that stops loses every reading it paid for.
        bundle.attach_reads(reads)
    rng = np.random.default_rng(seed)
    # THE ASSIGNED PANEL GOES INTO THE MANIFEST'S CONFIGURATION, AND THEREFORE INSIDE THE SEALED
    # SPECIFICATION HASH (finding A7). The aggregate rule is stated over the assigned systems, so the
    # denominator has to be a pre-run commitment: a panel that shrank between the seal and the
    # verdict then refuses as a custody failure instead of quietly becoming a smaller denominator
    # that a majority is easier to reach in.
    # AND THE SECOND CHANNELS GO INSIDE THE SEALED SPECIFICATION HASH (a defect found in review).
    # The manipulations are the only route to IDENTIFIED, so the set of them is a pre-run commitment
    # exactly as the assigned panel is: a channel added after the bank was read, or one quietly
    # dropped because it disagreed, would otherwise leave no trace. Their DESCRIPTIONS travel, never
    # the callables, and the description is what the judgement re-derives admissibility from.
    manipulation_records = [mp.as_record(place_at_state, adapter) for mp in manipulations]
    man = M.new_manifest("P5", pilot, ladder.sha256,
                         cfg.__dict__ | {"seed": seed,
                                         "assigned_heldout_systems": list(heldout_systems),
                                         "capability_manipulations": manipulation_records},
                         adapter.name, mode=m, confirmatory_inputs=inputs_record, ladder=ladder)
    if bundle is not None:
        bundle.write_manifest(man)      # on disk before the first paid call, not after the last one
    bank = run_bank(adapter, ladder, cfg, rng, place_at_state)
    if bundle is not None:
        # THE BANK REACHES DISK AS SOON AS IT EXISTS (finding A8's second crash case). The bank is the
        # most expensive object in the run and it used to live in memory until the final write, so a
        # stop anywhere after it lost every row that had been paid for. A milestone line here, and one
        # per manipulation and one per held-out system below, means the collected evidence is on disk
        # at every point at which a stop is possible.
        bundle.record_progress("bank", {"bank": bank})
    measured = []
    for mp in manipulations:
        measured.append(measure_manipulation(mp, adapter, ladder, cfg, rng,
                                             bank_place_at_state=place_at_state))
        if bundle is not None:
            bundle.record_progress("manipulation", {"manipulation": measured[-1]})
    # THE ANALYSIS STREAM'S STATE, CAPTURED WHERE THE ANALYSIS BEGINS. Every draw the estimator makes
    # comes from here, so a replay that restores this state and re-runs the estimator over the saved
    # bank reaches the same numbers; without it a saved run's route estimates can be read back and
    # never re-derived.
    analysis_rng_state = copy.deepcopy(rng.bit_generator.state)
    routes = estimate_routes(bank, cfg, rng, manipulation_estimates=measured)
    # The sealed prediction uses the STATE route. A held-out trajectory evolves along the capability
    # state, so whatever governs the response to state, the mechanism alone or the mechanism with a
    # nuisance rate riding on it, governs the path too. That is why a correct prediction and a wrong
    # mechanism reading can coexist, and why ruling 25 reports them as a pair: the retention route is
    # the identification check, and when the two disagree the state route still predicts and the
    # identification says NOT IDENTIFIED. The first end-to-end run in the rate-confound world showed
    # 0.72 by state against 0.48 by retention, which is the design doing what it was built to do.
    # When the routes agree they estimate one quantity and pooling them is the precise estimate; when
    # they disagree the state route is the one that governs the path. The rule is registered here
    # because the first clean-world run with the state route alone put two identical systems at
    # differences of 0.096 and 0.107 against a margin of 0.10, which is precision at the boundary.
    #
    # WHICH ELASTICITY THE PREDICTION USES IS A PRECISION QUESTION, NOT AN IDENTIFICATION ONE, and
    # since finding A6 it is keyed on the numerical CONSISTENCY check rather than on the word
    # IDENTIFIED, which now depends on whether the domain supplied a second capability intervention
    # and can no longer be produced by the bank alone. That re-keying is forced by the finding: the
    # word the rule used to read has changed its meaning. Nothing else about the rule has changed.
    #
    # THE ESTIMATOR ON THE DISAGREEMENT BRANCH IS THE REGISTERED ONE, WHICH IS THE STATE ROUTE (a
    # defect found in review). The first repair of finding A6 swapped it for the crossed fit's
    # capability exponent. The two estimate the same quantity in every world this mock can produce,
    # since the state route carries one rate per retention fraction and therefore reads the exponent
    # in the state exactly as the crossed fit's beta does, and measured at eight replicates on the
    # differing-retention world and on the shared-rate world they agree to about one ten-thousandth.
    # So the swap was not a large number; it was an unregistered one. The finding did not ask for it,
    # the rule above names the state route, and the estimator a sealed prediction is built from is
    # the author's amendment and not this repair's. The registered reading is restored and the
    # crossed fit's exponent is reported beside it under its own name, so an author who wants it can
    # see what it would have been in every run's own record. Where the state route carries no
    # readable exponent the crossed fit stands behind it, and the pooled fit behind that, and the
    # record says which of the three was used and why.
    if routes.get("route_agreement") == PI.CONSISTENT:
        beta_for_prediction, beta_se = routes["beta_pooled"], routes.get("se_pooled", float("nan"))
        source = ("the pooled fit: the two regression directions are consistent, so they estimate "
                  "one quantity and pooling them is the precise estimate")
    else:
        beta_for_prediction = routes.get("beta_state_route", float("nan"))
        beta_se = routes.get("se_state_route", float("nan"))
        source = ("the state route: the two directions are not consistent, and a continuation runs "
                  "at full retention and therefore travels along the capability state, so whatever "
                  "governs the response to state governs the path. This is the registered rule and "
                  "it is not keyed on the identification judgement")
        if not np.isfinite(beta_for_prediction):
            beta_for_prediction = routes.get("beta_capability", float("nan"))
            beta_se = routes.get("se_capability", float("nan"))
            source = ("the crossed fit's capability exponent: the state route carried no readable "
                      "exponent on this bank, so the registered estimator was unavailable")
    if not np.isfinite(beta_for_prediction):
        beta_for_prediction, beta_se = routes["beta_pooled"], routes.get("se_pooled", float("nan"))
        source = ("the pooled fit: neither the registered estimator nor the crossed fit carried a "
                  "readable exponent on this bank")
    routes["beta_used_for_prediction"] = float(beta_for_prediction)
    routes["beta_se_used_for_prediction"] = float(beta_se) if np.isfinite(beta_se) else None
    routes["beta_for_prediction_source"] = source
    # The estimator NOT used, printed beside the one that was, so that the size of the choice is
    # visible in every run's own record rather than only in the review that found it.
    routes["beta_for_prediction_alternatives"] = {
        "state_route": routes.get("beta_state_route"), "pooled": routes.get("beta_pooled"),
        "crossed_capability_exponent": routes.get("beta_capability"),
        "open_decision": "the registered rule takes the pooled fit when the two directions are "
                         "consistent and the state route when they are not. The crossed fit's "
                         "capability exponent estimates the same quantity from every cell at once "
                         "and would be a defensible amendment; it is reported and not used, because "
                         "the estimator a sealed prediction is built from is the author's to change."}
    heldout = run_heldout(heldout_adapter or adapter, ladder, cfg, rng, heldout_systems, start_for, beta_for_prediction, man,
                          beta_se=beta_se, anchor=anchor, attestation=attestation, bundle=bundle)
    # THE BANK AND THE ANALYSIS STREAM TRAVEL WITH THE RESULT. A returned run holding only its fitted
    # summaries cannot be re-scored from its own observations, which is what an evidence replay does.
    out = {"manifest": man, "routes": routes, "heldout": heldout, "bank": bank,
           "analysis_rng_state": analysis_rng_state}
    if not pilot:
        # The LIVE ladder goes into the verdict, so the sealed verifier binding is recomputed from the
        # object that actually did the scoring rather than read back out of the manifest that recorded
        # it. A ladder whose checking rule was replaced after the seal refuses here.
        out["verdicts"] = verdicts(man, routes, heldout, cfg, ladder=ladder)
    if bundle is not None:
        # The provider metadata is read from whichever adapters actually ran: a held-out panel placed
        # by a second adapter is a second account to keep, and a bundle that reports only the bank's
        # is not the run's account.
        provider = {"bank": CUSTODY.adapter_metadata(adapter)}
        if heldout_adapter is not None and heldout_adapter is not adapter:
            provider["heldout"] = CUSTODY.adapter_metadata(heldout_adapter)
        out["bundle_path"] = bundle.write_bundle(
            CUSTODY.build_bundle(man, "P5", cfg, sealed=heldout.get("sealed_predictions"), routes=routes,
                                 bank=bank, heldout=heldout, verdicts_=out.get("verdicts"), reads=reads,
                                 provider=provider, ladder=ladder, manipulations=measured))
    # The readings are exported under both names and the evidence status is written beside them, so
    # that a simulated recovery cannot be quoted as a verdict on the strength of a key name.
    return M.label_result(out)
