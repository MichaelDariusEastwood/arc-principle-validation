"""P16's sequential inference, frozen for the schedule this runner actually runs, and the four
readings of a non-alarm.

WHY THIS FILE EXISTS. Finding A3: the reference branch treated the absence of an alarm in an arm
predicted not to reverse as "AS PREDICTED", and a majority of above-boundary arms without alarms as
`REFUTED (no reversal)`. Neither is a measurement. An arm that produced no alarm may have produced

  a DEMONSTRATED SIGN          its terminal interval lies wholly on one side of zero, so the margin
                               was measured and it is positive (or negative) beyond the registered
                               band, AND it stayed there across the window rather than only on
                               average over it. Only this reading, and only positive, is evidence
                               against the predicted reversal.
  NOT SUSTAINED                its whole-window interval says one thing and a segment of the window
                               says another: the margin was above the band on average and below it
                               for part of the window, or nil on average and demonstrably signed for
                               part of it. The registered predicate is about the whole window and
                               this arm does not meet it.
  a DEMONSTRATED PRACTICAL     its terminal interval lies wholly inside a registered band about
  ABSENCE                      zero, so the margin is measured and is practically nil: the arm did
                               not reverse and did not stay positive either.
  LOW INFORMATION              its terminal interval spans both signs. The arm was watched and
                               nothing was learned from it, which is not the same statement as
                               either of the two above and must never be read as either.
  a CENSORED OBSERVATION       the event was not observed BY THE HORIZON: the arm did not run far
                               enough past the switch for absence to mean anything, or no informative
                               horizon is registered at all, in which case no length of silence is
                               informative and the fail-closed reading is that the arm is censored.

THE REFUTATION RULE, WHICH IS THE POINT. `REFUTED (no reversal)` may follow only from DEMONSTRATED
POSITIVITY in a MAJORITY of the above-boundary arms, each beyond the informative horizon. A mere
non-alarm never refutes anything. This is the registered rule and not a preference:
`arc_instruments.verdicts.p16_prohibition` reaches REFUTED from
`margin_lower_bound_positive_across_window`, and `p16_endpoint` reaches it from an event absent past
an INFORMATIVE horizon; neither reaches it from silence. A run that cannot demonstrate positivity
says INCONCLUSIVE, which is a statement about the run rather than about the world.

AND THE PREDICATE IS THE ONE THE CONTRACT NAMES, ACROSS THE WINDOW AND NOT ON ITS MEAN. The
registered evidence is a margin whose lower bound STAYS ABOVE ZERO ACROSS the window fixed before the
run. One interval on the window mean is a different predicate and a weaker one: an arm at plus 0.30
for forty-two rounds and then negative for the last eight, the reversal having plainly happened, has
a whole-window interval wholly above the band, and reading it as demonstrated positivity is this
finding's defect moved from the alarm counter to the terminal reading. So the window is also read in
consecutive equal segments at the registered resolution, and every segment has to clear the band for
the positivity to be demonstrated. The resolution is registered by the author and not chosen here,
because it is a trade the contract does not settle: a reversal shorter than one segment can hide
inside it, and a segment too short measures nothing. An unregistered resolution leaves the predicate
unevaluated, and an unevaluated predicate demonstrates nothing and refutes nothing.

AND A PRACTICALLY NIL MARGIN IS NOT POSITIVITY, SO A REGISTERED BAND IS REQUIRED FOR REFUTATION. With
no band registered the module can only compare the sign with zero, and a margin of plus 0.001
measured with no noise then clears zero and would refute the proposition on a measurement that says
the margin is nil. That is the reading this module refuses two paragraphs above, so refutation
requires the band as well as the horizon: the sign against zero is still reported, and it decides
nothing.

WHAT IS FROZEN HERE. `SequentialRule` is the complete decision family written down in one object so
that it can be sealed with the line, carried into the evidence bundle, and calibrated as a whole:

  the look schedule        one look per round, on a window that starts at the switch plus the
                           settling period and grows to the horizon, the first look being the fourth
                           usable round. The number of looks available is recorded per arm, because a
                           family whose size is not recorded cannot be calibrated afterwards.
  the threshold            delta + z se < 0 at every look, with z registered for THAT schedule. A
                           rule that looks at every growing window needs a higher threshold than one
                           that looks once, and the number belongs to the rule and travels with it.
  temporal correlation     the standard error is autocorrelation-robust: Newey and West with
                           Bartlett weights, taken at the conservative maximum with the independence
                           figure so that no variance repair can manufacture an alarm. The
                           alternative implemented beside it is a moving-block bootstrap of the
                           residuals. Which one a run used is recorded and never inferred. See
                           `arc_runner.observation`.
  the terminal reading     the whole post-settling window taken once, which is the only look in the
                           schedule that is not selected, at the same threshold as the sequential
                           rule, TOGETHER WITH the same window read in the registered number of
                           consecutive segments, which is what across the window means. Reusing the
                           sequential threshold is the conservative choice: a narrower single-look
                           interval would demonstrate positivity, and so refute, more readily than
                           the rule that declares the event.
  the control aggregation  the controls are satisfied only when no control declared AND every
                           control's silence was itself demonstrated, being a measured sign or a
                           measured practical absence past the informative horizon. A censored
                           control has not shown that it did not shift, and reading it as though it
                           had is the same defect in the controls that the refutation rule removes
                           from the dose arms. A minority of control alarms is unresolved rather
                           than a specificity failure, because the rule's own calibrated per-arm
                           false-alarm rate predicts some.
  support and refutation   support needs every contract component (see `arc_runner.p16_components`);
                           refutation needs the demonstrated positivity above.

THE ERROR RATE IS CALIBRATED, NOT ASSERTED. `p16_calibration.py` runs the complete decision family on
the registered configuration under null worlds and reports every rate as an INTERVAL with its Monte
Carlo uncertainty. A point estimate of 0.052 from 400 series has an exact upper bound near 0.075 and
is not demonstrated control at 0.05, which is what the reference comment claimed for it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from arc_instruments import mc_interval as MC, verdicts as V

from . import observation as OBS

# --------------------------------------------------------------------------------------------------
# The four readings of a non-alarm, and the fifth state that is not one
# --------------------------------------------------------------------------------------------------

EVENT_DECLARED = "EVENT DECLARED"
DEMONSTRATED_SIGN = "DEMONSTRATED SIGN"
PRACTICAL_ABSENCE = "DEMONSTRATED PRACTICAL ABSENCE"
LOW_INFORMATION = "LOW INFORMATION"
NOT_SUSTAINED = "NOT SUSTAINED ACROSS THE WINDOW"
CENSORED = "CENSORED: THE EVENT WAS NOT OBSERVED BY THE HORIZON"
NOT_MEASURED = "NOT MEASURED"          # the arm carries no terminal record at all

NON_ALARM_STATES = (DEMONSTRATED_SIGN, PRACTICAL_ABSENCE, LOW_INFORMATION, NOT_SUSTAINED, CENSORED)
STATES = (EVENT_DECLARED,) + NON_ALARM_STATES + (NOT_MEASURED,)

POSITIVE = "positive"
NEGATIVE = "negative"


@dataclass(frozen=True)
class SequentialRule:
    """The complete decision family, frozen as one object so that it can be sealed and calibrated."""

    switch_round: int
    settling: int
    start_round: int
    horizon: int
    min_points: int
    z_threshold: float
    terminal_z: float
    variance_estimator: str
    informative_horizon: Optional[int]
    practical_absence_band: Optional[float]
    across_window_segments: Optional[int]
    alarm_rate_null: float
    alarm_rate_null_calibration: Optional[Dict[str, Any]] = None

    @property
    def looks_available(self) -> int:
        """One look per round from the fourth usable round in the window to the horizon."""
        return max(int(self.horizon) - int(self.start_round) - int(self.min_points) + 1, 0)

    def as_record(self) -> Dict[str, Any]:
        """Plain JSON for the seal and the bundle: a later reader needs no code to read the rule."""
        return {
            "look_schedule": ("one look per round on a growing window from round %d, the first look "
                              "at the %dth usable round, to the horizon at round %d"
                              % (self.start_round, self.min_points, self.horizon)),
            "looks_available_per_arm": self.looks_available,
            "switch_round": int(self.switch_round), "settling": int(self.settling),
            "start_round": int(self.start_round), "horizon": int(self.horizon),
            "min_points": int(self.min_points),
            "threshold_z": float(self.z_threshold), "terminal_z": float(self.terminal_z),
            "variance_estimator": self.variance_estimator,
            "informative_horizon_rounds_after_switch": self.informative_horizon,
            "practical_absence_band": self.practical_absence_band,
            "across_window_segments": self.across_window_segments,
            "across_window_predicate": (
                "margin_lower_bound_positive_across_window, being the registered evidence in "
                "arc_instruments.verdicts.p16_prohibition: the lower bound clears the band on the "
                "whole post-settling window AND on each of the %s consecutive segments it is read "
                "in. An unregistered resolution leaves the predicate unevaluated and nothing is "
                "demonstrated from it" % (self.across_window_segments
                                          if self.across_window_segments is not None
                                          else "(unregistered number of)")),
            "alarm_rate_null": float(self.alarm_rate_null),
            "alarm_rate_null_calibration": self.alarm_rate_null_calibration,
            "control_aggregation": ("controls are satisfied only when no control declared AND every "
                                    "control's own silence was demonstrated, being a measured sign "
                                    "or a measured practical absence past the informative horizon: "
                                    "a censored control has not shown that it did not shift. A "
                                    "minority of control alarms is unresolved at the rule's own "
                                    "calibrated per-arm false-alarm rate and is not a specificity "
                                    "failure"),
            "support_rule": ("every contract component of arc_runner.p16_components satisfied, and a "
                             "fresh-data repetition meeting the same rule"),
            "refutation_rule": ("demonstrated positivity, being a lower bound above the registered "
                                "band on the whole window AND on every segment of it, in a majority "
                                "of the above-boundary arms, each beyond the informative horizon. A "
                                "non-alarm alone never refutes, a window mean alone is not the "
                                "across-window predicate, and with no band registered nothing is "
                                "admissible because a practically nil margin is not positivity"),
            "non_alarm_states": list(NON_ALARM_STATES),
        }


def rule_from_config(cfg: Any) -> SequentialRule:
    """The rule this configuration runs. Every field is read from the configuration and none defaults
    to a value invented here: an unregistered informative horizon stays None, and None means no
    length of silence is informative."""
    return SequentialRule(
        switch_round=int(cfg.switch_round), settling=int(cfg.settling),
        start_round=int(cfg.switch_round) + int(cfg.settling), horizon=int(cfg.horizon),
        min_points=int(max(int(getattr(cfg, "min_look_points", OBS.MIN_POINTS)), OBS.MIN_POINTS)),
        z_threshold=float(cfg.z_threshold),
        terminal_z=float(cfg.terminal_z if getattr(cfg, "terminal_z", None) is not None else cfg.z_threshold),
        variance_estimator=str(getattr(cfg, "variance_estimator", OBS.DEFAULT_VARIANCE)),
        informative_horizon=(None if getattr(cfg, "informative_horizon", None) is None
                             else int(cfg.informative_horizon)),
        practical_absence_band=(None if getattr(cfg, "practical_absence_band", None) is None
                                else float(cfg.practical_absence_band)),
        across_window_segments=(None if getattr(cfg, "across_window_segments", None) is None
                                else int(cfg.across_window_segments)),
        alarm_rate_null=float(cfg.alarm_rate_null),
        alarm_rate_null_calibration=getattr(cfg, "alarm_rate_null_calibration", None))


# --------------------------------------------------------------------------------------------------
# The terminal record, written at run time and read at scoring time
# --------------------------------------------------------------------------------------------------

def terminal_record(est: OBS.DeltaEstimate, first_round: int, last_round: int,
                    switch_round: int, declared_round: Optional[int],
                    looks_taken: Optional[int] = None,
                    looks_available: Optional[int] = None,
                    segments: Optional[Sequence[Dict[str, Any]]] = None,
                    n_segments: Optional[int] = None) -> Dict[str, Any]:
    """What a non-alarm has to be read from: the terminal estimate, its error, and how long it looked.

    It is a plain mapping on the arm rather than a classification because the classification depends
    on the registered band and the informative horizon, which are configuration and belong to the
    scoring moment: an analyst re-scoring the bundle under the sealed configuration must reach the
    same reading from these numbers alone.

    AND THE WINDOW READ IN SEGMENTS, WHICH IS THE ACROSS-WINDOW PREDICATE'S EVIDENCE. `segments` is
    the same post-settling window read in `n_segments` consecutive equal pieces, each with its own
    estimate and standard error, so that a reader can see whether the margin STAYED above the band or
    only averaged above it. The resolution is part of the sealed configuration, so it is applied when
    the arm runs and recorded with the count it was taken at; `classify` refuses to read segments
    taken at one resolution under a rule that registers another, which is the one way this record
    could otherwise drift from the rule it is scored under. `segments` of None is the fail-closed
    state and leaves the predicate unevaluated.
    """
    rec = {"delta": float(est.delta), "se": float(est.se),
           "se_independent": float(est.se_independent), "se_robust": est.se_robust,
           "variance_estimator": est.variance_estimator,
           "n_used": int(est.n_used), "n_excluded": int(est.n_excluded),
           "first_round": int(first_round), "last_round": int(last_round),
           "rounds_after_switch": int(last_round) - int(switch_round),
           "declared_round": (None if declared_round is None else int(declared_round)),
           "looks_taken": looks_taken, "looks_available": looks_available}
    rec["across_window"] = (None if segments is None else
                            {"n_segments": int(n_segments if n_segments is not None else len(segments)),
                             "segments": [dict(seg) for seg in segments]})
    return rec


def _interval(delta: Any, se: Any, z: float) -> Optional[Tuple[float, float]]:
    """An estimate's interval at the registered threshold, or None where it was not estimable.

    A zero standard error is admissible and gives a point interval: a perfectly measured window is
    perfectly measured, and the reference rule's demand for a strictly positive error is the smaller
    half of finding A3.
    """
    try:
        d, e = float(delta), float(se)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(d) and np.isfinite(e)) or e < 0:
        return None
    half = float(z) * e
    return (d - half, d + half)


def _across_window(terminal: Dict[str, Any], rule: SequentialRule,
                   band: Optional[float]) -> Dict[str, Any]:
    """Did the margin STAY on one side of the band across the window, or only average there?

    This is the registered refutation predicate,
    `arc_instruments.verdicts.p16_prohibition`'s `margin_lower_bound_positive_across_window`, and the
    reason it is not the whole-window interval is the arm that sits at plus 0.30 for forty-two rounds
    and then goes negative for the last eight: its whole-window interval is wholly above the band
    while the reversal it was watched for has plainly happened. A mean cannot see when the margin was
    where it was, so the same window is read in segments and every one of them must clear.

    Three states, and only the first decides anything. EVALUATED, where the registered resolution and
    the arm's own segmented reading agree and every segment was estimable. NOT EVALUATED, where no
    resolution is registered, where the arm carries no segmented reading, or where the two
    resolutions differ: the predicate is then unevaluated, which is not the same as met, and the
    window mean alone demonstrates nothing. A segment that was not estimable also leaves it
    unevaluated rather than passing on the segments that were, because a window read in part is not
    the window.

    `edge` is the registered band, or zero where no band is registered. Nothing is admissible for
    refutation without a band in any case (see `classify`); the zero comparison is kept so that the
    sign is still reported.
    """
    edge = 0.0 if band is None else float(band)
    rec: Dict[str, Any] = {"registered_segments": rule.across_window_segments, "evaluated": False,
                           "segments": None, "sustained_above": None, "sustained_below": None,
                           "contradicts_practical_absence": None, "edge": edge, "reason": ""}
    if rule.across_window_segments is None:
        rec["reason"] = ("no across-window resolution is registered, so the predicate the contract "
                         "names was not evaluated: a lower bound above the band on the window MEAN "
                         "is a different and weaker statement, and it demonstrates nothing here")
        return rec
    aw = terminal.get("across_window")
    if not isinstance(aw, dict) or not aw.get("segments"):
        rec["reason"] = ("this arm carries no segmented reading of its post-settling window, so the "
                         "registered across-window predicate cannot be evaluated from it and its "
                         "silence demonstrates nothing")
        return rec
    if int(aw.get("n_segments", 0)) != int(rule.across_window_segments):
        rec["reason"] = ("this arm's window was read in %s segments and the registered resolution is "
                         "%d, so the predicate would be evaluated at a resolution the run did not "
                         "use" % (aw.get("n_segments"), int(rule.across_window_segments)))
        return rec
    rows: List[Dict[str, Any]] = []
    ivs: List[Tuple[float, float]] = []
    for seg in aw["segments"]:
        iv = _interval(seg.get("delta"), seg.get("se"), rule.terminal_z)
        if iv is None:
            rec["reason"] = ("the segment covering rounds %s to %s was not estimable, so the window "
                             "was read only in part and the predicate is unevaluated"
                             % (seg.get("first_round"), seg.get("last_round")))
            return rec
        ivs.append(iv)
        rows.append({"first_round": seg.get("first_round"), "last_round": seg.get("last_round"),
                     "delta": seg.get("delta"), "se": seg.get("se"),
                     "n_used": seg.get("n_used"), "interval": [iv[0], iv[1]]})
    rec.update(evaluated=True, segments=rows,
               sustained_above=all(V.wholly_above(iv, edge) for iv in ivs),
               sustained_below=all(V.wholly_below(iv, -edge) for iv in ivs),
               contradicts_practical_absence=any(V.wholly_above(iv, edge) or V.wholly_below(iv, -edge)
                                                 for iv in ivs),
               reason="the window was read in %d segments at the registered resolution" % len(rows))
    return rec


def _reading(detail: Dict[str, Any], state: str, reason: str, sign: Optional[str] = None,
             interval: Optional[Sequence[float]] = None, censored: Optional[bool] = None,
             demonstrated: bool = False, admissible: bool = False,
             **extra: Any) -> Dict[str, Any]:
    """One arm's reading, with the two flags every consumer decides on rather than re-deriving.

    `demonstrated` says this silence measured something under the complete registered rule, which is
    what the controls need before "no shift" means anything. `refutation_admissible` says it is the
    one reading that may refute: a demonstrated POSITIVE margin, across the window, past the
    informative horizon, against a REGISTERED band. Two consumers deriving these from the state
    string apart from each other is how the controls came to report SATISFIED on the same silences
    the refutation rule was calling censored.
    """
    out = dict(detail)
    out.update(state=state, sign=sign, interval=(list(interval) if interval is not None else None),
               censored=censored, demonstrated=bool(demonstrated),
               refutation_admissible=bool(admissible), reason=reason)
    out.update(extra)
    return out


def classify(terminal: Optional[Dict[str, Any]], rule: SequentialRule) -> Dict[str, Any]:
    """The reading of one arm's silence, with the interval and the segments it was read from.

    The order of the tests is the order of the refusals and is deliberate.

      1. No terminal record: NOT MEASURED. A bundle written before this rule existed carries none,
         and inventing one from the summary would be the defect this component exists to remove.
      2. An alarm: EVENT DECLARED. There is no silence to read.
      3. Censoring, which dominates every interval reading: an arm that did not reach the informative
         horizon may still reverse, so whatever its terminal interval says about the margin so far,
         the EVENT is unobserved rather than absent. An unregistered informative horizon censors every
         arm, because with no registered horizon no length of silence has been said to be informative.
      4. The interval, against the registered band, AND the same window in segments. With a band
         registered, practical absence is tested FIRST and the sign has to CLEAR the band: a margin
         of plus 0.001 inside a band of 0.05 is practically nil and is not evidence that the reversal
         did not happen. A sign that clears the band on the whole window but not on every segment of
         it is NOT SUSTAINED, which is neither a demonstrated sign nor low information. A practical
         absence contradicted by a segment that demonstrates a sign is NOT SUSTAINED as well: a
         margin that was firmly positive for half the window and firmly negative for the other half
         is not a margin that was measured to be nil.
      5. Without a band, practical absence is unreachable and the sign is tested against zero. That
         reading is reported and is NEVER admissible for refutation: with no band the module cannot
         tell a practically nil margin from a real one, and a noiseless plus 0.001 would otherwise
         refute the proposition on a measurement that says the margin is zero. Registering the width
         of a band that decides a proposition is the author's act, and until it is registered the
         fail-closed reading is that no arm's positivity is admissible.
    """
    if not isinstance(terminal, dict) or not terminal:
        return {"state": NOT_MEASURED, "sign": None, "interval": None, "censored": None,
                "demonstrated": False, "refutation_admissible": False,
                "reason": "this arm carries no terminal record, so its silence cannot be read: the "
                          "run predates the frozen sequential rule or did not measure the arm"}
    if terminal.get("declared_round") is not None:
        return {"state": EVENT_DECLARED, "sign": NEGATIVE, "interval": None, "censored": False,
                "demonstrated": False, "refutation_admissible": False,
                "declared_round": int(terminal["declared_round"]),
                "reason": "the event was declared at round %d, so there is no non-alarm to read"
                          % int(terminal["declared_round"])}
    delta, se = float(terminal.get("delta", float("nan"))), float(terminal.get("se", float("nan")))
    n = int(terminal.get("n_used", 0))
    iv = _interval(delta, se, rule.terminal_z)
    band = rule.practical_absence_band
    detail = {"delta": delta if np.isfinite(delta) else None,
              "se": se if np.isfinite(se) else None,
              "se_independent": terminal.get("se_independent"),
              "se_robust": terminal.get("se_robust"),
              "variance_estimator": terminal.get("variance_estimator"),
              "terminal_z": float(rule.terminal_z),
              "practical_absence_band": band,
              "informative_horizon": rule.informative_horizon,
              "rounds_after_switch": terminal.get("rounds_after_switch"),
              "n_used": n, "looks_available": terminal.get("looks_available")}
    if n < rule.min_points or iv is None:
        return _reading(detail, CENSORED, censored=True,
                        reason="the arm's post-settling window carried %d usable rounds, fewer than "
                               "the %d the estimator needs, so nothing was measured to be absent"
                               % (n, rule.min_points))
    if rule.informative_horizon is None:
        return _reading(detail, CENSORED, interval=iv, censored=True,
                        reason="no informative horizon is registered, so no length of silence has "
                               "been said to be informative and the absence of the event is censored "
                               "rather than absent. The terminal interval is reported beside this "
                               "and decides nothing on its own")
    observed = int(terminal.get("rounds_after_switch", -1))
    if observed < int(rule.informative_horizon):
        return _reading(detail, CENSORED, interval=iv, censored=True,
                        reason="the arm was observed for %d rounds after the switch and the "
                               "registered informative horizon is %d, so the event is unobserved "
                               "rather than absent" % (observed, int(rule.informative_horizon)))
    aw = _across_window(terminal, rule, band)
    detail["across_window"] = aw
    edge = 0.0 if band is None else float(band)
    band_note = ("" if band is not None else
                 ". No practical-absence band is registered, so the comparison is with zero and this "
                 "reading is not admissible for refutation: a practically nil margin is not "
                 "positivity")
    if band is not None and V.wholly_inside(iv, -float(band), float(band)):
        if aw["evaluated"] and aw["contradicts_practical_absence"]:
            return _reading(detail, NOT_SUSTAINED, interval=iv, censored=False,
                            reason="the terminal interval lies wholly inside the registered band, "
                                   "but a segment of the window demonstrates a sign outside it: the "
                                   "margin averaged nil and was not nil across the window")
        return _reading(detail, PRACTICAL_ABSENCE, interval=iv, censored=False,
                        demonstrated=bool(aw["evaluated"]),
                        reason="the terminal interval lies wholly inside the registered band of plus "
                               "or minus %g about zero: the margin is measured and practically nil%s"
                               % (float(band), "" if aw["evaluated"] else ", though " + aw["reason"]))
    if V.wholly_above(iv, edge):
        if not aw["evaluated"]:
            return _reading(detail, DEMONSTRATED_SIGN, sign=POSITIVE, interval=iv, censored=False,
                            reason="the terminal interval lies wholly above the band on the window "
                                   "MEAN, which is not the registered across-window predicate: %s%s"
                                   % (aw["reason"], band_note))
        if not aw["sustained_above"]:
            return _reading(detail, NOT_SUSTAINED, sign=POSITIVE, interval=iv, censored=False,
                            reason="the terminal interval lies wholly above the band on the window "
                                   "mean, and at least one segment of the window does not: the "
                                   "margin did not STAY above the band, which is what the registered "
                                   "predicate asks")
        return _reading(detail, DEMONSTRATED_SIGN, sign=POSITIVE, interval=iv, censored=False,
                        demonstrated=True, admissible=band is not None,
                        reason="the margin's lower bound stays above the registered band across the "
                               "window, on the whole window and on each of its %d segments, past the "
                               "informative horizon%s" % (len(aw["segments"]), band_note))
    if V.wholly_below(iv, -edge):
        if aw["evaluated"] and not aw["sustained_below"]:
            return _reading(detail, NOT_SUSTAINED, sign=NEGATIVE, interval=iv, censored=False,
                            reason="the terminal interval lies wholly below the band on the window "
                                   "mean and at least one segment of the window does not, without "
                                   "the sequential rule having declared the event")
        return _reading(detail, DEMONSTRATED_SIGN, sign=NEGATIVE, interval=iv, censored=False,
                        demonstrated=bool(aw["evaluated"]),
                        reason="the terminal interval lies wholly below the registered band without "
                               "the sequential rule having declared the event%s"
                               % ("" if aw["evaluated"] else "; " + aw["reason"]))
    return _reading(detail, LOW_INFORMATION, interval=iv, censored=False,
                    reason=("the terminal interval neither clears the registered band nor lies "
                            "inside it, so this arm's silence carries no information about the "
                            "margin") if band is not None else
                           ("the terminal interval spans both signs, so this arm's silence carries "
                            "no information about the margin"))


def classify_arm(arm: Dict[str, Any], rule: SequentialRule) -> Dict[str, Any]:
    """The reading of one arm as it is saved, keyed by the arm's own name."""
    out = classify(arm.get("terminal"), rule)
    out["arm"] = arm.get("arm")
    out["alpha"] = arm.get("alpha")
    return out


# --------------------------------------------------------------------------------------------------
# The refutation rule
# --------------------------------------------------------------------------------------------------

def refutation(arms: Sequence[Dict[str, Any]], sealed_zero: float, rule: SequentialRule) -> Dict[str, Any]:
    """May this run say `REFUTED (no reversal)`, and on what evidence?

    Only `margin_lower_bound_positive_across_window` in a majority of the above-boundary arms, each
    beyond the informative horizon: a lower bound above the REGISTERED band on the whole window and
    on every segment of it. Everything else, including every arm being silent for the whole horizon,
    is a run that did not decide: the reference branch reached a refutation from exactly that
    silence, which is the defect finding A3 names.

    THREE THINGS DO NOT COUNT, AND EACH ONE WAS ONCE COUNTED HERE. A silence with no measurement
    behind it, which is the original defect. A window MEAN above the band whose segments were never
    read or do not sustain it, which is the same defect wearing the terminal reading's clothes: an
    arm positive for forty-two rounds and negative for the last eight has a mean above the band and
    has plainly reversed. And a sign measured against zero because no band is registered, since the
    module cannot then tell a practically nil margin from a real one and a noiseless plus 0.001 would
    refute the proposition on a measurement that says the margin is zero.

    THE RESOLUTION TRAVELS WITH THE VERDICT, because it is what the run could have seen: positivity
    demonstrated on segments of a hundred rounds says nothing about a reversal ten rounds long, and a
    reader who is not told the segment length cannot know which reversals this refutation excludes.
    """
    above = [a for a in arms if str(a.get("arm", "")).startswith("dose")
             and float(a.get("alpha", float("nan"))) > float(sealed_zero)]
    rows = [classify_arm(a, rule) for a in above]
    positive = [r for r in rows if r.get("refutation_admissible")]
    # a demonstrated NEGATIVE sign without an alarm is its own bucket rather than an unaccounted row:
    # the terminal interval put the margin below the band while the sequential rule never declared,
    # which is a disagreement between the two readings and belongs in the census that reports them
    negative = [r for r in rows if r["state"] == DEMONSTRATED_SIGN and r.get("sign") == NEGATIVE]
    # and a positive sign that is NOT admissible is its own bucket too, because "the mean was above
    # the band and the predicate was not met" is a different refusal from "nothing was measured".
    # Every row lands in exactly one bucket below: a census with an unaccounted remainder is how a
    # reader comes to assume the remainder was the reading they expected.
    # a positive sign the rule cannot admit: the across-window predicate was not evaluated, or no
    # band is registered to clear. The bucket is named for the refusal and not for one of its two
    # causes, because each row's own reason says which of them applied to it
    inadmissible = [r for r in rows if r["state"] == DEMONSTRATED_SIGN and r.get("sign") == POSITIVE
                    and not r.get("refutation_admissible")]
    unsustained_positive = [r for r in rows if r.get("sign") == POSITIVE
                            and not r.get("refutation_admissible")
                            and r["state"] in (DEMONSTRATED_SIGN, NOT_SUSTAINED)]
    censored = [r for r in rows if r["state"] == CENSORED]
    low = [r for r in rows if r["state"] == LOW_INFORMATION]
    absent = [r for r in rows if r["state"] == PRACTICAL_ABSENCE]
    unsustained = [r for r in rows if r["state"] == NOT_SUSTAINED]
    unmeasured = [r for r in rows if r["state"] == NOT_MEASURED]
    declared = [r for r in rows if r["state"] == EVENT_DECLARED]
    n = len(rows)
    refutes = bool(n) and len(positive) > n / 2.0
    counts = {DEMONSTRATED_SIGN + " (positive, across the window)": len(positive),
              DEMONSTRATED_SIGN + " (positive, not admissible as refutation evidence)": len(inadmissible),
              DEMONSTRATED_SIGN + " (negative, undeclared)": len(negative),
              NOT_SUSTAINED: len(unsustained),
              PRACTICAL_ABSENCE: len(absent),
              LOW_INFORMATION: len(low), CENSORED: len(censored), NOT_MEASURED: len(unmeasured),
              EVENT_DECLARED: len(declared)}
    spans = [s_["last_round"] - s_["first_round"] + 1
             for r in positive for s_ in (r.get("across_window") or {}).get("segments") or []]
    resolution = {"across_window_segments": rule.across_window_segments,
                  "practical_absence_band": rule.practical_absence_band,
                  "shortest_segment_rounds": (min(spans) if spans else None),
                  "note": "a reversal confined to fewer rounds than one segment could not have been "
                          "seen by this predicate, so the segment length is part of what the "
                          "refutation says"}
    if not n:
        reason = "no arm was placed above the sealed boundary, so there is nothing to refute from"
    elif refutes:
        reason = ("%d of %d above-boundary arms hold a lower bound above the registered band across "
                  "the window, past the informative horizon, which is the majority the registered "
                  "rule requires" % (len(positive), n))
    else:
        rest = ", ".join("%d %s" % (v, k.lower()) for k, v in counts.items()
                         if v and k != DEMONSTRATED_SIGN + " (positive, across the window)")
        reason = ("%d of %d above-boundary arms hold a lower bound above the registered band across "
                  "the window, past the informative horizon, which is not a majority; the rest are "
                  "%s. Absence of an alarm is neither a positive margin nor a refutation"
                  % (len(positive), n, rest or "unaccounted"))
        if unsustained_positive and not rule.across_window_segments:
            reason += (". %d of them clear the band on the window mean, which is not the registered "
                       "across-window predicate" % len(unsustained_positive))
        if rule.practical_absence_band is None:
            reason += (". No practical-absence band is registered, so no arm's positivity is "
                       "admissible: a practically nil margin is not positivity")
    return {"refutes": refutes, "n_above_boundary_arms": n, "n_demonstrated_positive": len(positive),
            "n_positive_mean_not_admissible": len(unsustained_positive),
            "counts": counts, "reason": reason, "arms": rows,
            "predicate": "margin_lower_bound_positive_across_window",
            "resolution": resolution,
            "rule": {"informative_horizon_rounds_after_switch": rule.informative_horizon,
                     "practical_absence_band": rule.practical_absence_band,
                     "across_window_segments": rule.across_window_segments,
                     "terminal_z": float(rule.terminal_z)}}


def control_readings(arms: Sequence[Dict[str, Any]], rule: SequentialRule) -> List[Dict[str, Any]]:
    """The same four-way reading, applied to the sham and the baseline.

    WHY THE CONTROLS NEED IT TOO. The controls component reached SATISFIED from a bare alarm count,
    so one run could report that "neither the sham nor the baseline arms shifted" while every one of
    those same silences classified as CENSORED: the event was not observed by the horizon. Two
    contradictory readings of one silence, and the generous one decided. An unobserved event is not
    an absent one in a control arm any more than in a dose arm, which is the rule this whole module
    exists to hold, so the controls are read from these rows and not from the count alone.
    """
    return [classify_arm(a, rule) for a in arms
            if str(a.get("arm", "")) in ("sham", "baseline")]


# --------------------------------------------------------------------------------------------------
# The family, and its calibrated error as an interval
# --------------------------------------------------------------------------------------------------

def family_size(rule: SequentialRule, n_arms: int) -> Dict[str, Any]:
    """How many decisions the complete family makes, which is what its error rate is a rate over."""
    looks = rule.looks_available
    return {"arms": int(n_arms), "looks_per_arm": int(looks),
            "arm_look_tests": int(n_arms) * int(looks),
            "run_level_decisions": 5,     # support, refutation, mislocation, specificity, inconclusive
            "note": "the family is every look in every arm plus the run-level rules that read them; a "
                    "per-arm rate is not the family's rate and the two are never reported as one"}


def rate_with_interval(successes: int, repetitions: int, inner: Optional[int] = None,
                       conf: float = 0.95) -> Dict[str, Any]:
    """A rate the way this kit reports one: the count, the rate and the exact binomial interval.

    Finding A3's second half. The reference threshold comment reported "approximately 0.052 from 400
    series" as though it were demonstrated control at 0.05. Twenty in 400 is an observed 5 per cent
    whose exact one-sided upper bound is about 7.2 per cent, so the observation does not certify the
    rate it estimates, and zero in 400 is not zero risk either.
    """
    return MC.RateWithUncertainty(int(successes), int(repetitions), inner, conf).as_dict()
