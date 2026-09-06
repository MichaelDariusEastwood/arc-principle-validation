"""P16, run: the driven titration across a located boundary, with the registered detection rule.

This is the P16 registration's STEP 9 as code. The boundary is located from calibration systems, dose arms sit
either side of it, a coefficient-only sham and an untouched baseline run alongside, and the balance
line is sealed as a line before the switch. After the switch the balance is read from a window that
starts after a settling period and grows to the horizon; the reversal is declared at the first round
where the estimate's interval lies wholly below zero at the registered threshold, the estimate being
the level for a Delta series and the fitted slope for a ratio series; the timing is a
change-point estimate with a tolerance; and no location is claimed from a line that is not falling.

WHAT THE SOURCE RETURNS IS TYPED, AND THE TYPE DECIDES THE COORDINATE. Finding A1: this file used to
take a callback returning one float per round, call it a correction margin, regress it against
ordinary round indices and test the fitted slope, with nothing anywhere saying which quantity the
float was. The registration's Delta is already the logarithmic slope of the service to burden ratio,
so a falling series of Delta values is not the registered event and a falling but positive Delta is
not an event at all. Every source now declares its quantity, its units, its clock, its estimator and
its smoothing window through `arc_runner.observation`, and the declaration decides what is estimated
and what the event is: the level for a Delta series, the slope on log R for a log-ratio series. An
undeclared source resolves to the unregistered per-round surplus rate, which is what the reference
mock actually produced, and a confirmatory run refuses it.

AND THE ABSENCE OF AN ALARM IS READ, NEVER ASSUMED. Finding A3: a silent arm used to count as "AS
PREDICTED" wherever no reversal was predicted, and a majority of silent above-boundary arms used to
produce `REFUTED (no reversal)` outright. Silence is now read in four ways, in
`arc_runner.p16_sequential`: a demonstrated sign, a demonstrated practical absence, low information,
or an event censored by the horizon. Only demonstrated positivity in a majority of the
above-boundary arms, each beyond a registered informative horizon, may refute; every other silence
is a statement about the run and not about the world. The sequential rule the arms are read under,
being the growing-window look schedule, its threshold, its autocorrelation-robust standard error, the
terminal reading, the control aggregation and the support and refutation rules, is frozen as one
object and sealed with the line.

Two demonstration sources ship. `mock_margin_source` is the reference one, unchanged in its numbers
and now declared for what it is: a per-round surplus rate whose trend is (alpha_crit - alpha_arm)
times a slope, in a coordinate the sealed line does not live in, so no comparison with the sealed
slope is offered. `mock_balance_source` supplies Q and W separately from the marginal-burden model,
so its arms sit on the registered coordinate and the fitted line IS the sealed line. Both declare
themselves simulations and neither may decide a proposition: they demonstrate the coordinate, the
estimator and the whole apparatus, and a simulated margin attests nothing about the world. The
command line's demonstration reads the second of them, because the first supplies no delivery record,
no capability and no recursive coordinate, so four of the nine components are NOT SUPPLIED in it
whatever numbers are registered and no run on it can reach a result at all.

AND A DEMONSTRATION IS GIVEN CANDIDATE NUMBERS, LABELLED AS SUCH. The five registered quantities are
unset on the shipped defaults and stay unset on every paid path, which is the fail-closed reading and
is not changed here. A demonstration that carries none of them withholds every result, so it shows
nothing of the branch it exists to show; it is therefore built by `demonstration_config` from the
candidate numbers `p16_calibration.py` measures the decision family under, and every one of them is
named as a candidate in the sealed configuration, in the verdict block and in the printed summary.
The deciding gate refuses a titration carrying the label, so the numbers cannot travel from a
rehearsal to a run that decides anything.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import custody as CUSTODY, manifest as M, mode as MODE, observation as OBS
from . import p16_components as C, p16_sequential as SEQ


@dataclass
class P16Config:
    dose_offsets: Sequence[float] = (-0.6, -0.3, 0.3, 0.6, 0.9)
    systems_per_arm: int = 3
    horizon: int = 96
    switch_round: int = 8
    settling: int = 6                 # ruling 30
    z_threshold: float = 4.0          # ruling 30, registered FOR THIS WINDOW RULE. A rule that looks at every
                                      # growing window needs a higher threshold than one that looks once, so
                                      # the number belongs to the rule and travels with it. What the rule's
                                      # error rate IS, is calibrated and not asserted: see `alarm_rate_null`
                                      # below and `p16_calibration.py`.
    timing_tolerance: int = 4
    chi_hat: float = 0.5
    alpha_crit_hat: float = 2.0       # located from calibration systems, never from the arms
    margin_noise: float = 0.04
    location_tolerance: float = 0.15  # the fitted line's zero must sit within this of the sealed zero; the first
                                      # runner battery supported a boundary mislocated by 0.3 in every run because
                                      # the verdict checked signs and never the line's zero, which is the location
                                      # test the design registers and the reason the line is sealed as a line
    # THE THREE NUMBERS THE COMPONENT BRANCH NEEDS (finding A2). Two of them are unset by default and
    # the components they belong to are NOT SUPPLIED until they are set, which is the fail-closed
    # reading: choosing the width of a band that decides a proposition is the author's act.
    chi_hat_se: Optional[float] = None    # the calibration uncertainty on chi. The sealed zero is
                                          # 1/(1-chi) and the sealed slope is -(1-chi), so without
                                          # this the location and slope comparisons treat the
                                          # prediction as exactly known, which is the point-distance
                                          # defect finding A2 names.
    slope_equivalence: Optional[float] = None  # the registered equivalence band on the sealed slope.
                                          # The reference branch tested only that the line was
                                          # falling, which a line an order of magnitude too shallow
                                          # satisfies. No band is registered by the contract.
    alarm_rate_null: float = 0.052        # THE WORKING PER-ARM FALSE-ALARM RATE, AND IT IS A POINT
                                          # ESTIMATE. Finding A3: the reference comment reported "an
                                          # approximately 0.052 flat-null alarm rate from 400 series
                                          # at z = 4.0" as though it were demonstrated control at or
                                          # below 0.05. It is not. Twenty-one alarms in 400 series is
                                          # an observed 0.0525 whose exact 95 per cent interval is
                                          # [0.033, 0.079] and whose one-sided upper bound is 0.075,
                                          # so the observation is consistent with a true rate half
                                          # again above 0.05; and 400 series of one stipulated window
                                          # and noise setup say nothing about the other worlds or
                                          # about the COMPLETE decision family, which is every look
                                          # in every arm plus the run-level rules that read them.
                                          # `p16_calibration.py` calibrates that family on the
                                          # registered configuration and reports every rate as an
                                          # interval; put its record in the field below so that the
                                          # run carries the calibration it was scored against.
    alarm_rate_null_calibration: Optional[Dict[str, Any]] = None
    # THE THREE NUMBERS FINDING A3 NEEDS, TWO OF THEM UNSET AND FAIL-CLOSED FOR THE SAME REASON AS
    # THE TWO ABOVE: a band or a horizon that decides a proposition is registered by the author.
    informative_horizon: Optional[int] = None   # rounds after the switch beyond which the ABSENCE of
                                          # the event is informative. Unset means no length of silence
                                          # has been said to be informative, so every non-alarm is
                                          # CENSORED and no run may refute from silence. The reference
                                          # branch had no such notion and refuted from silence alone.
    practical_absence_band: Optional[float] = None  # the band about zero inside which a measured
                                          # margin is practically nil. Unset means practical absence
                                          # cannot be demonstrated, a sign is read against zero, and
                                          # NO arm's positivity is admissible for refutation: the
                                          # module cannot then tell a practically nil margin from a
                                          # real one, and a noiseless plus 0.001 would otherwise
                                          # refute the proposition on a measurement that says the
                                          # margin is zero.
    across_window_segments: Optional[int] = None    # the number of consecutive equal segments the
                                          # post-settling window is read in for the registered
                                          # predicate `margin_lower_bound_positive_across_window`: a
                                          # lower bound ABOVE the band ACROSS the window, and not on
                                          # its mean. A mean cannot see when the margin was where it
                                          # was, so an arm at plus 0.30 for forty-two rounds and
                                          # negative for the last eight passes a mean test with the
                                          # reversal already in it. Unset means the predicate is not
                                          # evaluated and nothing refutes. The resolution is a trade
                                          # the contract does not settle and is therefore the
                                          # author's: a reversal shorter than one segment hides
                                          # inside it, and a segment too short measures nothing. On
                                          # the calibration world whose margin is firmly positive
                                          # everywhere, halves sustain in 90 of 90 arms, thirds in 47
                                          # of 90, and at four segments a world with no reversal in
                                          # it fails; in the same halves the flat null sustains
                                          # nothing in 90 arms. See arc_runner.observation.
    terminal_z: Optional[float] = None    # the threshold of the single terminal interval a non-alarm
                                          # is read from. Unset means the sequential threshold, which
                                          # is the conservative choice: a narrower single-look
                                          # interval would demonstrate positivity, and so refute,
                                          # more readily than the rule that declares the event.
    variance_estimator: str = OBS.DEFAULT_VARIANCE   # which standard error the sequential rule reads;
                                          # see arc_runner.observation. Recorded, never inferred.
    min_look_points: int = OBS.MIN_POINTS  # the size of the FIRST look of the growing window. Short
                                          # windows are where a persistent series wins: at four
                                          # points no variance estimator can see a dependence length
                                          # of eighteen. The reference rule's four is kept as the
                                          # default so that this repair changes no run by itself, and
                                          # p16_calibration.py measures what the choice costs.
    # WHICH OF THE FIVE REGISTERED QUANTITIES ABOVE CARRY A CANDIDATE VALUE AND NOT A REGISTERED ONE.
    # A candidate is a number this package measures its own decision family under so that the family
    # can be exercised at all; it is not the author's registered choice and it may not decide a
    # proposition. The names sit in the configuration rather than beside it, so they travel inside the
    # sealed specification hash and cannot be attached or removed after the seal, and the deciding gate
    # refuses a titration carrying any of them. Empty is the shipped state, and a configuration whose
    # numbers came from the registration says nothing here.
    candidate_quantities: Tuple[str, ...] = ()


# THE CANDIDATE NUMBERS, DECLARED IN ONE PLACE. `p16_calibration.py` measures the complete decision
# family under these and states in its own prose that they are candidates and not registrations. They
# live here so that the calibration script, the demonstration and the drift test read one dictionary
# rather than three copies of it that would eventually differ. Nothing on a paid path applies them:
# `demonstration_config` is the only constructor that does, it labels every one it applies, and
# `arc_runner.mode` refuses a deciding run that carries a label.
CANDIDATE_QUANTITIES: Dict[str, Any] = {"chi_hat_se": 0.02, "slope_equivalence": 0.15,
                                        "informative_horizon": 40, "practical_absence_band": 0.05,
                                        "across_window_segments": 2}

CANDIDATE_LABEL = ("CANDIDATE NUMBERS: this run was read with %s, which are candidate values this "
                   "package measures its own decision family under and not registered ones. Nothing "
                   "read with them decides a proposition, and the deciding path refuses a titration "
                   "that carries any of them.")


def candidate_record(cfg: Any) -> Dict[str, Any]:
    """Which of a configuration's registered quantities are candidates, their values, and the sentence
    that must travel with anything read from them.

    One reader for one label, used by the manifest, by the verdict block and by the command line, so
    that a run cannot say one thing in its record and another in its summary.
    """
    names = [str(n) for n in (getattr(cfg, "candidate_quantities", ()) or ())]
    values = {n: getattr(cfg, n, None) for n in names}
    label = "" if not names else CANDIDATE_LABEL % ", ".join("%s=%s" % (n, values[n]) for n in names)
    return {"names": names, "values": values, "label": label}


def demonstration_config(**supplied) -> P16Config:
    """The configuration a demonstration runs under: the candidate numbers, each labelled a candidate.

    WHY A DEMONSTRATION IS GIVEN NUMBERS AT ALL. On the shipped defaults all five registered
    quantities are unset, so the components that read them are NOT SUPPLIED and the wrapper withholds
    any result: `demonstrate p16` reported NOT EVALUABLE and demonstrated nothing about the branch it
    exists to show. Refusing to invent a registered number is right and is unchanged on every paid
    path; what a demonstration needs is a number it may exercise the machinery with, and the
    calibration script already declares five of those. They are applied here and nowhere else.

    ANYTHING TYPED WINS AND IS NOT LABELLED. A quantity supplied by the operator is the operator's
    number and stops being a candidate, so it replaces the candidate and drops out of the label; the
    remaining candidates are named. That keeps the label a statement about where each value came
    from rather than about which command was typed.
    """
    typed = {k: v for k, v in supplied.items() if v is not None}
    values = dict(CANDIDATE_QUANTITIES)
    values.update(typed)
    names = tuple(n for n in CANDIDATE_QUANTITIES if n not in typed)
    return P16Config(candidate_quantities=names, **values)


def predicted_line(cfg: P16Config) -> Dict[str, float]:
    """The balance line, sealed as a line: zero at the located boundary, slope minus (1 - chi)."""
    return {"zero": cfg.alpha_crit_hat, "slope": -(1.0 - cfg.chi_hat)}


def predicted_timing(cfg: P16Config) -> Dict[str, Any]:
    """The sealed change point and its tolerance: the switch round, plus or minus the tolerance.

    Finding A2: `timing_tolerance` was written into the sealed object and never read back by any
    decision, so a reversal at any round in the horizon counted exactly as one at the round the
    switch predicts. A tolerance needs something to be a tolerance ON, and this is it.

    CONSERVATIVE READING, NAMED AS OPEN. The contract seals the tolerance without stating the round
    it surrounds. The reading taken here is the only one the apparatus fixes: the manipulation is
    applied at the switch round, so the change point the design predicts is the switch round, and the
    window is symmetric about it. The quantity compared with this window is the CHANGE POINT and
    never the declaration round: the declaration round is a detection latency that moves with the
    noise, the horizon and the threshold, and on the reference configuration a true switch at round 8
    is declared somewhere in the high twenties. If the author registers a different expected round,
    or an asymmetric window, it belongs here.
    """
    return {"expected_round": int(cfg.switch_round), "tolerance": int(cfg.timing_tolerance),
            "window": [int(cfg.switch_round) - int(cfg.timing_tolerance),
                       int(cfg.switch_round) + int(cfg.timing_tolerance)],
            "quantity": "change point, estimated by least squares over candidate split rounds"}


def predicted_signs(cfg: P16Config) -> Dict[str, str]:
    out = {"dose%+.1f" % o: ("negative" if o > 0 else "positive") for o in cfg.dose_offsets}
    out["sham"] = "no shift"; out["baseline"] = "no shift"
    return out


def detect_reversal(margin: Sequence[float], cfg: P16Config, spec: Optional[OBS.ObservationSpec] = None) -> Dict[str, Any]:
    """The registered rule for a bare series of numbers, kept for the callers that hold one.

    The rule itself now lives in `observation.detect_event`, because the rule depends on the quantity:
    a Delta series is tested on its level and a log-ratio series on its slope in log R, and only the
    declaration says which. An absent declaration is the undeclared per-round quantity, which is what
    this function's callers have always passed and is exactly how it used to behave. One behaviour
    does change: the old loop required a strictly positive standard error before it would declare, so
    a noiseless declining series never reached its alarm branch and the acceptance case of a constant
    negative Delta could not fire. A zero-residual window is a perfectly measured one and its interval
    is a point.
    """
    spec = spec if spec is not None else OBS.UNDECLARED
    det = OBS.detect_event(OBS.readings_from_values(margin), spec, cfg.switch_round + cfg.settling,
                           cfg.z_threshold, cfg.variance_estimator, cfg.min_look_points)
    return {"declared_round": det["declared_round"], "slope": det["delta"], "se": det["se"],
            "falling": det["falling"], "event": det["event"], "estimate": det["estimate"],
            "looks_taken": det.get("looks_taken"), "looks_available": det.get("looks_available")}


def window_slope(margin: Sequence[float], cfg: P16Config, spec: Optional[OBS.ObservationSpec] = None) -> float:
    """The arm's own estimate over the whole post-settling window, in the declared coordinate.
    The reversal rule's value is taken at the declaration round, which is a selected and therefore
    biased value; the first runner battery fitted the line from a mixture of those and full-window
    values and false-refuted a true boundary 28 times in 100 on location. The line is fitted from
    this alone. For a declared Delta series this is the window mean and not a slope, which is the
    point of finding A1 and the reason the returned arm carries `arm_delta` beside this name."""
    spec = spec if spec is not None else OBS.UNDECLARED
    return OBS.arm_estimate(OBS.readings_from_values(margin), spec, cfg.switch_round + cfg.settling,
                            cfg.variance_estimator).delta


def run_arm(margin_source, arm: str, alpha_arm: float, cfg: P16Config, rng: np.random.Generator,
            spec: Optional[OBS.ObservationSpec] = None) -> Dict[str, Any]:
    """One arm, read through the typed boundary and estimated in the coordinate its quantity settles.

    The readings are kept whole. Q and W are never reduced to their ratio here: the ratio is the
    quantity under test, and a record that saved only the ratio could not be re-checked against the
    measurements it came from, which is what an independent analyst has to be able to do.
    """
    spec = spec if spec is not None else OBS.spec_of(margin_source)
    readings = [OBS.read(margin_source, arm, alpha_arm, r, rng, spec) for r in range(cfg.horizon)]
    start = cfg.switch_round + cfg.settling
    det = OBS.detect_event(readings, spec, start, cfg.z_threshold, cfg.variance_estimator,
                           cfg.min_look_points)
    est = OBS.arm_estimate(readings, spec, start, cfg.variance_estimator)
    # AND THE SAME WINDOW IN SEGMENTS (finding A3). The registered refutation evidence is a lower
    # bound above the band ACROSS the window, which a single interval on the window mean is not: an
    # arm positive for forty-two rounds and negative for the last eight clears a mean test with its
    # reversal already in it. The resolution is registered configuration and is sealed with the rule,
    # so it is applied here and recorded with the count it was taken at; an unregistered resolution
    # yields None and leaves the predicate unevaluated, which refutes nothing.
    segments = OBS.window_segments(readings, spec, start, cfg.across_window_segments,
                                   cfg.variance_estimator)
    window = [rd for rd in readings if rd.round_index >= start]
    # THE TERMINAL RECORD, which is what a silence is read from (finding A3). It carries the estimate
    # taken once over the whole post-settling window, both standard errors, how far past the switch
    # the arm was actually observed and how many looks the schedule took, and it is written on the
    # arm rather than classified here: the four-way reading depends on the registered band and the
    # informative horizon, which belong to the scoring moment and travel in the sealed configuration.
    terminal = SEQ.terminal_record(
        est, first_round=(window[0].round_index if window else start),
        last_round=(window[-1].round_index if window else start - 1),
        switch_round=cfg.switch_round, declared_round=det["declared_round"],
        looks_taken=det.get("looks_taken"), looks_available=det.get("looks_available"),
        segments=segments, n_segments=cfg.across_window_segments)
    # WHETHER THE ARM WAS OBSERVED AT ALL, typed and carried rather than left to be inferred from a
    # NaN. A source that returned nothing usable for some round produces an arm whose estimates are
    # not numbers, and a decision rule reading those as measurements is deciding on an arm nobody
    # measured. The status says which, and the rule below refuses a titration containing one.
    complete = bool(np.all(np.isfinite([r.value for r in readings])))
    return {"arm": arm, "alpha": alpha_arm, "terminal": terminal,
            "observation_status": "COMPLETE" if complete else "MISSING OR INVALID",
            "margin": [r.value for r in readings],          # the declared quantity, one value per round
            "readings": [r.as_dict() for r in readings],    # and the parts each value was computed from
            "observation": spec.as_record(),
            "arm_delta": est.delta, "window_slope": est.delta,   # one number, its registered name and its old one
            "arm_estimate": est.as_dict(),
            "declared_round": det["declared_round"], "slope": det["delta"], "se": det["se"],
            "falling": det["falling"], "event": det["event"],
            # THE THREE MEASUREMENTS THE SUPPORT BRANCH NEEDS BESIDE THE EVENT (finding A2). They are
            # computed here, from the readings, and carried as scalars on the arm, so that the
            # printed summary and a saved bundle hold the same component inputs: a component that
            # could only be computed while the series was in memory would not survive to the
            # re-scoring an evidence bundle exists for.
            "realised_exposure": OBS.realised_exposure(readings, start),
            "change_point": OBS.change_point(readings, spec),
            "delivery": OBS.delivery_summary(readings)}


def arm_verdict(arm: Dict[str, Any], predicted: str, cfg: P16Config,
                classification: Optional[Dict[str, Any]] = None) -> str:
    """One arm's reading, with a silence qualified by what the silence actually showed.

    Finding A3: an arm predicted not to reverse, and which did not reverse, used to read "AS
    PREDICTED" whether its margin had been measured positive past the horizon or whether nothing had
    been measured at all. The label now carries which of the four readings the silence was, and the
    unqualified "AS PREDICTED" is reserved for a silence that demonstrated something. The five
    labels themselves are unchanged as prefixes, because the operating-characteristic battery and
    every bundle written before this change read them.
    """
    declared = arm["declared_round"] is not None
    if predicted == "negative":
        if declared:
            return "AS PREDICTED"
        return "NO REVERSAL" + _silence(classification)
    head = "REVERSED AGAINST PREDICTION" if predicted == "positive" else "SHIFT WHERE NONE PREDICTED"
    if declared:
        return head
    return "AS PREDICTED" + _silence(classification)


def _silence(classification: Optional[Dict[str, Any]]) -> str:
    """The parenthetical that says which of the four readings a non-alarm was."""
    if not classification:
        return ""
    state, sign = classification.get("state"), classification.get("sign")
    if state == SEQ.DEMONSTRATED_SIGN:
        if not classification.get("demonstrated"):
            return " (%s margin on the window mean only: the across-window predicate was not " \
                   "evaluated)" % (sign or "signed")
        return " (%s margin demonstrated)" % (sign or "signed")
    if state == SEQ.PRACTICAL_ABSENCE:
        return " (practical absence demonstrated)"
    if state == SEQ.LOW_INFORMATION:
        return " (low information: nothing demonstrated)"
    if state == SEQ.NOT_SUSTAINED:
        return " (not sustained across the window: the mean says one thing and a segment another)"
    if state == SEQ.CENSORED:
        return " (censored: the event is unobserved, not absent)"
    if state == SEQ.NOT_MEASURED:
        return " (no terminal record)"
    return ""


def _line(arms: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The balance line across the dose arms, fitted against the exposure each arm actually realised.

    Finding A2: the reference branch fitted the line against the arm's ASSIGNED `alpha_arm`, which is
    the dose the apparatus was asked to deliver and not a measurement of anything. A titration whose
    lever fails, saturates or overshoots then reports a line drawn against intentions, and the
    registered locating stage does the opposite (`code_domain.locate_boundary` takes realised
    exponents as its x). So the line is fitted against the realised exponents where every contributing
    arm measured one, and the fit records WHICH exposure it used.

    Where they were not measured the assigned doses are still fitted, because the run pattern below
    and every bundle written before this change are read from that line and a bundle that can no
    longer be re-scored is not evidence. What changes is that a line fitted on assigned doses decides
    nothing: the location and slope-magnitude components refuse it by name.
    """
    dose = [a for a in arms if str(a.get("arm", "")).startswith("dose")]
    rows = [a for a in dose if np.isfinite(_arm_delta(a))]
    realised = []
    for a in rows:
        re_ = a.get("realised_exposure")
        realised.append(float(re_["alpha"]) if isinstance(re_, dict) and re_.get("measured") else None)
    on_realised = bool(rows) and all(r is not None and np.isfinite(r) for r in realised)
    xs = [r for r in realised] if on_realised else [float(a["alpha"]) for a in rows]
    ys = [_arm_delta(a) for a in rows]
    fit = C.fit_line(xs, ys)
    fit["fitted_on"] = "realised-exposure" if on_realised else "assigned-dose"
    fit["arms"] = [{"arm": a.get("arm"), "alpha_assigned": a.get("alpha"), "x": x, "delta": y}
                   for a, x, y in zip(rows, xs, ys)]
    if not on_realised:
        fit["reason"] = (fit.get("reason") or "") + (
            "; " if fit.get("reason") else "") + (
            "fitted against the assigned doses because not every contributing arm measured a "
            "realised growth exponent: this line describes the run and decides nothing")
    return fit


def _timing_window(preds: Dict[str, Any], cfg: P16Config) -> Tuple[Optional[Sequence[float]], str]:
    """The sealed timing window, or the one reconstructed from the sealed configuration.

    A run sealed after finding A2 carries the window in `preds["timing"]`. A bundle sealed before it
    carries only `timing_tolerance`, and the switch round it surrounds is in the configuration, which
    is itself inside the sealed specification hash. Reconstructing it there is therefore reading a
    commitment and not inventing one, and the reconstruction says so in the component's reason.
    """
    t = preds.get("timing")
    if isinstance(t, dict) and t.get("window"):
        return list(t["window"]), "sealed"
    tol = preds.get("timing_tolerance")
    if tol is None:
        return None, "absent"
    return ([int(cfg.switch_round) - int(tol), int(cfg.switch_round) + int(tol)],
            "reconstructed from the sealed tolerance and the sealed switch round")


def components(preds: Dict[str, Any], arms: Sequence[Dict[str, Any]], cfg: P16Config,
               spec: OBS.ObservationSpec, line: Dict[str, Any],
               replication: Optional[Dict[str, Any]] = None,
               refutation: Optional[Dict[str, Any]] = None) -> List[C.Component]:
    """The nine contract components, each decided on its own evidence and its own uncertainty.

    `refutation` is finding A3's record for the event component: a low alarm rate is a failure of the
    event only where the silence behind it demonstrated a positive margin past the informative
    horizon, and is otherwise unresolved.
    """
    sealed_zero = float(preds["line"]["zero"])
    window, window_source = _timing_window(preds, cfg)
    timing = C.timing_component(arms, window, sealed_zero)
    if window_source not in ("sealed", "absent"):
        timing = C.Component(timing.name, timing.state,
                             timing.reason + " (window %s)" % window_source, timing.detail)
    if refutation is None:
        refutation = SEQ.refutation(arms, sealed_zero, SEQ.rule_from_config(cfg))
    return [
        C.delivery_component(arms, cfg.switch_round),
        C.realised_exposure_component(arms, sealed_zero),
        C.event_component(arms, sealed_zero, cfg.alarm_rate_null, refutation),
        timing,
        C.location_component(line, sealed_zero, cfg.location_tolerance, cfg.chi_hat, cfg.chi_hat_se),
        C.slope_component(line, float(preds["line"]["slope"]), cfg.slope_equivalence, cfg.chi_hat_se,
                          spec.registered),
        C.controls_component(arms, cfg.alarm_rate_null,
                             SEQ.control_readings(arms, SEQ.rule_from_config(cfg))),
        C.discrimination_component(arms, sealed_zero),
        C.repetition_component(replication),
    ]


def verdicts(man: Dict[str, Any], preds: Dict[str, Any], arms: Sequence[Dict[str, Any]],
             cfg: P16Config, replication: Optional[Dict[str, Any]] = None,
             ladder: Any = None) -> Dict[str, Any]:
    """The complete support branch, as a function of the sealed line and the arms.

    It is a function rather than a block inside `run_p16` for one reason (finding A8): an evidence
    bundle that cannot be re-scored is not evidence. Given the saved arms and the saved seal, this
    reaches the same table without the run, the provider or the terminal summary.

    `P16` IS THE COMPONENT WRAPPER'S RESULT (finding A2). The reference branch reached SUPPORTED from
    an arm pattern, a falling line and a point distance, with the sealed timing unread and the sealed
    slope magnitude compared with nothing, so several alarms in several arms were enough. The nine
    components of `arc_runner.p16_components` are each decided on their own evidence and their own
    uncertainty, and the wrapper refuses a proposition-level result while any of them is unsupplied
    and labels every single run provisional. The old branch's own reading is kept verbatim under
    `run_pattern`, with its five labels unchanged, because it is the run's description of its arms
    and because the operating-characteristic battery measures the DETECTION RULE through it; it is
    not, and is no longer named as, the proposition's result.

    `ladder` is the LIVE ladder where the caller holds one, and it goes into the custody check so that
    the sealed verifier binding is recomputed from the object that did the scoring rather than read
    back out of the manifest. A re-scoring from a saved bundle has none, and the custody report then
    says the check was not performed, which is not the same statement as saying it passed.
    """
    label = M.require_reportable(man)          # see p5.verdicts: reportable here, scoreable below
    custody_report = None
    if M.mode_of(man) is M.ExecutionMode.CONFIRMATORY:
        # The live sealed line against the sealed hash, on the deciding path only: a line edited
        # between the seal and the verdict is exactly what the seal exists to catch.
        custody_report = M.require_scoreable(man, predictions=preds, config=cfg, ladder=ladder)
    # And the record, on every path: a sealed line edited after the seal is caught here rather than
    # only where the deciding gate runs, which a demonstration never reaches. After the custody chain
    # for the reason P5 orders them that way.
    M.require_diagnostic(man, preds)
    # WHAT WAS MEASURED, taken from the seal and never from the arms. The quantity is a pre-run
    # commitment like the line and the signs, so it is sealed with them; reading it back from the
    # seal is what lets a saved bundle be re-scored in the coordinate it was collected in, and what
    # makes a quantity changed between the seal and the verdict a custody failure rather than a
    # silent recoordination.
    spec = OBS.ObservationSpec.from_record(preds.get("observation"))
    # AND WHETHER THE ROUNDS THE ARMS CARRY ARE THE ROUNDS THAT DECLARATION PROMISED. `read` refuses a
    # round that does not carry what was declared, but `read` is not in the picture here: a bundle is
    # re-scored from saved rounds, and a bundle written before that refusal existed may hold exactly
    # the rounds it would have refused. A declaration checked only against itself is a promise, so it
    # is checked here against what is on disk as well, and a run whose readings contradict it is
    # reported and refused at proposition level rather than scored on the strength of its own label.
    reconciliation = _reconcile(arms, spec)
    # THE SEQUENTIAL RULE, and the four-way reading of every silence under it (finding A3). The rule
    # is rebuilt from the configuration, which is inside the sealed specification hash, so a bundle
    # re-scores under the rule it was collected under and a rule changed after the seal is a custody
    # failure rather than a silent reinterpretation of the arms.
    rule = SEQ.rule_from_config(cfg)
    silences = {}
    per = {}
    for a in arms:
        cl = SEQ.classify_arm(a, rule)
        silences.setdefault(a["arm"], []).append(cl)
        per.setdefault(a["arm"], []).append(arm_verdict(a, preds["signs"][a["arm"]], cfg, cl))
    above = [a for a in arms if a["arm"].startswith("dose+")]
    below = [a for a in arms if a["arm"].startswith("dose-")]
    controls = [a for a in arms if a["arm"] in ("sham", "baseline")]
    above_ok = sum(a["declared_round"] is not None for a in above) > len(above) / 2
    below_ok = sum(a["declared_round"] is None for a in below) > len(below) / 2
    ctrl_ok = sum(a["declared_round"] is None for a in controls) > len(controls) / 2
    # the balance line: the arms' own estimates against the exposure each arm realised
    line = _line(arms)
    line_slope, line_zero = line["slope"], line["zero"]
    falling = line_slope < 0
    # THE LOCATION TEST: the fitted line's zero against the sealed zero. A mislocated boundary is
    # invisible to the sign pattern across arms and visible here, which is why the line is sealed.
    # This point distance is the RUN PATTERN's gate and is kept unchanged; the location COMPONENT
    # below is the one that carries both uncertainties, which is finding A2's requirement.
    located_ok = np.isfinite(line_zero) and abs(line_zero - preds["line"]["zero"]) <= cfg.location_tolerance
    # WHETHER A NON-REVERSAL MAY REFUTE AT ALL (finding A3). The reference branch reached
    # `REFUTED (no reversal)` from a majority of above-boundary arms simply not alarming, which is a
    # statement about the arms' silence and not about their margins: an arm too noisy to decide, an
    # arm stopped before the event could be seen, and an arm whose margin was measured firmly
    # positive all produced the same label. Refutation now needs demonstrated positivity in a
    # majority of those arms, each beyond a registered informative horizon, which is the rule
    # `arc_instruments.verdicts.p16_prohibition` and `p16_endpoint` both state. Everything else that
    # fails to reverse is inconclusive.
    refute = SEQ.refutation(arms, preds["line"]["zero"], rule)
    # AND AN ARM NOBODY MEASURED DECIDES NOTHING. A titration holding a round that came back missing
    # or not a number has no reading of that arm, and every branch below would have read its silence
    # as a measurement: an unobserved arm counted as "did not reverse" is the cheapest way to reach
    # a refutation. The whole pattern is inconclusive while one is present, and it says why.
    incomplete = [a["arm"] for a in arms if a.get("observation_status") != "COMPLETE"]
    if incomplete:
        pattern = "INCONCLUSIVE (incomplete observations)"
    elif above_ok and below_ok and ctrl_ok and falling and located_ok:
        pattern = "SUPPORTED"
    elif not ctrl_ok:
        pattern = "NOT SPECIFIC (INCONCLUSIVE)"
    elif not above_ok:
        pattern = "REFUTED (no reversal)" if refute["refutes"] else \
            "INCONCLUSIVE (no reversal, and none demonstrated absent)"
    elif falling and not located_ok:
        pattern = "REFUTED (boundary mislocated)"
    else:
        pattern = "INCONCLUSIVE"
    # THE NUMERICAL LINE, and whether the two slopes are candidates for the same number at all. The
    # sealed slope is d Delta / d alpha, dimensionless. A per-round surplus rate's trend is a rate of
    # change of a level per round: the reference runner printed the sealed minus one half beside a
    # fitted minus 0.01 as though a reader could compare them, and no conversion between the two
    # exists without Q, W and the recursive coordinate R. Where the quantity is unregistered the
    # comparison is refused by name rather than reported as a difference.
    line_cmp = OBS.line_comparison(spec, line_slope, preds["line"]["slope"])
    comps = components(preds, arms, cfg, spec, line, replication, refute)
    combined = C.combine(comps)
    # AND WHETHER THE NUMBERS THIS RESULT WAS READ WITH WERE REGISTERED OR MERELY CANDIDATES. It sits
    # beside `provisional` for the same reason `provisional` is there: a reader who has to reach into
    # the configuration to learn that a band was a candidate will quote the result without it. Empty
    # on every run whose quantities came from the registration.
    candidates = candidate_record(cfg)
    out = {"P16": combined["P16"], "run_pattern": pattern,
           "incomplete_arms": incomplete,
           "provisional": combined["provisional"],
           "provisional_label": combined["provisional_label"],
           "candidate_quantities": candidates["values"],
           "candidate_label": candidates["label"],
           "component_reason": combined["reason"],
           "components": combined["components"],
           "component_states": combined["component_states"],
           "components_not_supplied": combined["not_supplied"],
           "components_failed": combined["failed"],
           "components_unresolved": combined["unresolved"],
           "execution_mode": man.get("execution_mode"),
           "interpretation": label, "custody": custody_report,
           "scoreable_at_proposition_level": bool(man.get("scoreable_at_proposition_level")),
           "sequential_rule": rule.as_record(),
           "family_size": SEQ.family_size(rule, len(arms)),
           # the two flags travel with the state, because they are what every consumer decides on:
           # `demonstrated` is what the controls need before "no shift" means anything, and
           # `refutation_admissible` is the one reading that may refute. A verdict block that
           # reported the state alone would leave the next reader to re-derive them, which is how the
           # controls came to report SATISFIED on silences the refutation rule was calling censored.
           "non_alarm": {k: [{"state": c["state"], "sign": c.get("sign"),
                              "interval": c.get("interval"), "censored": c.get("censored"),
                              "demonstrated": c.get("demonstrated"),
                              "refutation_admissible": c.get("refutation_admissible"),
                              "across_window": c.get("across_window"),
                              "reason": c["reason"]} for c in v] for k, v in silences.items()},
           "refutation": refute,
           "per_arm": per, "line_slope_fitted": line_slope,
           "line_slope_sealed": preds["line"]["slope"], "line_zero_fitted": line_zero,
           "line_zero_sealed": preds["line"]["zero"], "falling_line_gate": falling,
           "location_gate": bool(located_ok),
           "line_fitted_on": line["fitted_on"], "line_fit": line,
           "observation": spec.as_record(), "observed_quantity": spec.quantity,
           "event_tested": spec.event, "registered_coordinate": spec.registered,
           "observation_reconciliation": reconciliation,
           "line_comparison": line_cmp}
    if not spec.registered:
        # Said in the verdict itself, because a verdict block that names the coordinate only in a
        # nested record is a verdict a reader will quote without it.
        out["proposition_level"] = (
            "REFUSED: the observed quantity is %r, which is not the registered balance elasticity. "
            "The arms were read and the pattern is reported; the proposition is not decided by it."
            % spec.quantity)
    elif reconciliation["state"] == "refused":
        out["proposition_level"] = (
            "REFUSED: the sealed declaration says %r and the saved rounds do not carry it (%s). A "
            "quantity nothing supplied is not a quantity that was measured, whatever the declaration "
            "is labelled; the arms are reported and the proposition is not decided by them."
            % (spec.quantity, reconciliation["reason"]))
    return out


def diagnostic_verdicts(arms: Sequence[Dict[str, Any]], preds: Dict[str, Any], cfg: P16Config,
                        man: Optional[Dict[str, Any]] = None,
                        replication: Optional[Dict[str, Any]] = None,
                        ladder: Any = None) -> Dict[str, Any]:
    """The registered P16 decision rule, reachable without the run, so that a saved titration can be
    re-scored from its stored observations with no provider call and no trajectory regenerated. This
    is the name an evidence replay reaches for.

    THE MANIFEST IS REQUIRED AND NOT OPTIONAL. The rule's result depends on which kind of run
    produced the arms: the wrapper labels a single run provisional, refuses a proposition-level
    reading while a component is unsupplied, and prints the mode with the table. A rule applied
    without the manifest would therefore be a DIFFERENT rule from the one the run itself applied, and
    a replay comparing the two would report a difference that is an artefact of how it asked. Anyone
    holding a saved run holds its manifest, so nothing is lost by asking for it.

    `ladder` is the live ladder where a caller has one. A replay has none, and the custody report then
    says the verifier binding was not checked rather than saying it passed.
    """
    if man is None:
        raise ValueError(
            "the registered decision rule needs the run's manifest: its result depends on the mode "
            "the arms were collected in, so scoring them without it would apply a different rule "
            "from the one the run applied")
    return verdicts(man, preds, arms, cfg, replication, ladder=ladder)


def _reconcile(arms: Sequence[Dict[str, Any]], spec: OBS.ObservationSpec) -> Dict[str, Any]:
    """The sealed declaration against every saved round of every arm.

    Arms that saved no rounds, which is what the printed summary carries, reconcile to `not-checked`:
    absent evidence is not contrary evidence, and a summary is a reading of the measurement rather
    than the measurement.
    """
    rows: List[OBS.Reading] = []
    for a in arms:
        for d in (a.get("readings") or []):
            try:
                rows.append(OBS.Reading.from_dict(d))
            except (KeyError, TypeError, ValueError):
                continue
    return OBS.reconcile_readings(rows, spec)


def _arm_delta(arm: Dict[str, Any]) -> float:
    """The arm's estimate of the quantity the line is fitted from, under either name.

    `arm_delta` is what a run written after finding A1 records. `window_slope` is what every bundle
    written before it recorded, and a bundle that can no longer be re-scored is not evidence, so both
    are read here and neither is renamed on disk.
    """
    v = arm.get("arm_delta", arm.get("window_slope"))
    return float("nan") if v is None else float(v)


def run_p16(margin_source, cfg: P16Config, seed: int, ladder_sha256: str, adapter_name: str,
            pilot: bool = False, sealed_by: str = "arc_runner", mode: Any = None,
            confirmatory_inputs: Optional[MODE.ConfirmatoryInputs] = None,
            bundle: Any = None, anchor=None, attestation: Optional[Dict[str, Any]] = None,
            replication: Optional[Dict[str, Any]] = None,
            ladder: Any = None, adapter: Any = None) -> Dict[str, Any]:
    """`mode` resolves fail closed exactly as in P5: silence is a demonstration. A CONFIRMATORY run is
    gated before the arms loop, because the first arm's first round is the first paid call. P16 places
    no bank states, so the store requirement here is the seed artefact every arm starts from; the
    states list is empty by construction rather than by omission. The gate also asks this
    configuration for the five registered quantities its verdict is read with, since a titration
    carrying none of them can pay for every arm and still return NOT EVALUABLE, and it asks that the
    approved figure be held in a controller rather than be a figure nothing reserves against.

    `bundle`, `anchor` and `attestation` are as in P5. The returned arms keep their summaries without
    the margin series, which is what a printed summary can carry; the bundle keeps every arm's full
    series, because the series is the measurement and the summary is a reading of it.

    `ladder` and `adapter` are the objects a real titration holds, and the reason they are here is
    the half of finding A8's bundle that P16 did not have. P5 turned the ladder's read log on and
    passed the readings and the provider's own account into the bundle; this run took neither, so
    every whole-pool read a real arm performs through `code_domain.suite_margin_source` was counted
    into the four balance objects and then discarded, and the model identifier, the usage figures and
    the finish reasons the adapter records were thrown away on the only path where a P16 run has a
    provider. An analyst regenerating a P16 result from the saved evidence needs both. Neither is
    required: a caller holding only a hash and a name passes those, as before, and the bundle then
    records what was supplied rather than inventing what was not. Where the ladder object is given,
    it is also the identity the manifest seals, so the verifier binding is bound rather than recorded
    as never bound.

    AND `ladder` IS A CLAIM, WHICH IS WHY THREE THINGS NOW CHECK IT. P5 holds the ladder it scores
    with, so its argument and its readings cannot come apart; a P16 margin source is built by the
    caller over its own ladder and this run is only TOLD which one that is. First, a run handed two
    different ladders, one in the deciding inputs and another here, is refused rather than resolved:
    a setup whose two hands name different pools is one whose author believed something false about
    it, and silently preferring either would seal that belief. Second, the object this run binds is
    the object the deciding gate is given, so a ladder that reaches the seal has been through the
    domain rules that refuse a simulated pool, the reference smoke pool and an unattested verifier;
    it used to be possible to pass the gate on one ladder and seal another. Third, a read log
    attached to the named ladder that records nothing while arms were collected is the claim failing
    in the only way anything here can observe, which no argument can be checked for in advance and
    which the bundle states in words and the deciding path refuses.

    `replication` is the fresh-data repetition record the registered rule requires before a result
    reads at proposition level: a mapping with `fresh_data`, `same_rule_met` and a `source`. Absent,
    which is the default and what a first run has, the repetition component is NOT SUPPLIED and the
    wrapper labels the finding provisional. It travels into the bundle so that a re-scoring reaches
    the same table.

    THE TITRATION'S OWN SHAPE IS CHECKED FIRST, because each of these failures is a run that has been
    paid for before it can be seen. Two dose offsets that round to the same label at the declared
    precision collapse into one arm and the second silently overwrites the first's prediction; a
    horizon that does not clear the switch, the settling and a few looks leaves nothing to detect in;
    a non-finite offset places an arm nowhere. A remote endpoint is refused here for the reason it is
    refused in P5, at the library boundary: this package has no released service and burden
    measurements to point a real system at.
    """
    if cfg.systems_per_arm < 1 or cfg.horizon <= cfg.switch_round + cfg.settling + 4:
        raise ValueError("invalid arm count or measurement horizon")
    if len(set(cfg.dose_offsets)) != len(cfg.dose_offsets) or not all(np.isfinite(cfg.dose_offsets)):
        raise ValueError("dose offsets must be finite and unique")
    names = ["dose%+.1f" % o for o in cfg.dose_offsets]
    if len(set(names)) != len(names):
        raise ValueError("dose labels collide at the declared precision")
    if getattr(margin_source, "uses_remote_endpoint", False):
        raise M.InstrumentNotReleased(
            "real P16 collection lacks released service and burden measurements")
    m = MODE.resolve(mode, pilot)
    pilot = bool(pilot or m is MODE.ExecutionMode.PILOT)
    spec = OBS.spec_of(margin_source)
    inputs_record = None
    # THE LIVE LADDER IS NAMED BEFORE THE GATE, NOT AFTER IT. It was resolved below the gate, so the
    # object the manifest sealed and the read log recorded had never been through the domain rules
    # that refuse a simulated pool, the reference smoke pool or an unattested verifier: a deciding
    # run could be approved on the ladder its inputs named and bind the one its argument named, and
    # a simulated ladder the gate refuses on its own passed that way and was sealed with
    # `all_bound` false and no failure anywhere. One name for it, because the manifest identity, the
    # read log, the gate and the verdict must all be about the same object.
    named = confirmatory_inputs.ladder if confirmatory_inputs is not None else None
    if ladder is not None and named is not None and not CUSTODY.same_ladder(ladder, named):
        raise MODE.ModeRefusal(
            "this titration was handed two different ladders: its inputs name the pool %s and the "
            "run was given the pool %s. This run does not read the ladder itself, it is told which "
            "ladder its margin source reads, so a disagreement between the two hands is refused "
            "here rather than resolved: preferring either would seal a setup whose author believed "
            "something false about it"
            % (str(getattr(named, "sha256", None))[:16], str(getattr(ladder, "sha256", None))[:16]),
            ("domain-ladder: the inputs and the run were given different ladders",))
    live_ladder = ladder if ladder is not None else named
    if m is MODE.ExecutionMode.CONFIRMATORY:
        checked = confirmatory_inputs if confirmatory_inputs is not None else MODE.ConfirmatoryInputs()
        # The configuration is the one this run holds and never the caller's copy of it, for the same
        # reason P5 overwrites its own: an inputs object that can describe a different titration from
        # the one about to run is a record of an intention rather than of a setup. The ladder is
        # written in for that same reason and was not: it is the object this run binds and logs, so
        # it is the object the gate must be shown.
        # AND THE ANCHOR AND THE ATTESTATION GO IN WITH IT, for the same reason and against the same
        # defect. Both were read from the caller's copy at the gate and taken from the run's own
        # arguments afterwards, where the argument wins: a run could therefore be approved on the
        # inputs' anchoring service and seal with the mock anchor it was handed as an argument, and
        # a record supplied only as an argument was refused although the run held one. P5 writes
        # both into the gate's copy before the gate; this now does the same, so the objects checked
        # are the objects used.
        checked = replace(checked, states=(), config=cfg, ladder=live_ladder,
                          # The margin source is this titration's observing system, and it goes into
                          # the gate's copy with everything else so that the manifest records what it
                          # said it was. `OBS.require_assay` below asks the same object a narrower
                          # question about the quantity it returns; this records the answer to the
                          # broader one, so that a bundle re-scored later reaches it too.
                          observing_systems=(margin_source,),
                          anchor=anchor if anchor is not None else checked.anchor,
                          attestation=(attestation if attestation is not None
                                       else checked.attestation))
        if checked.checkpoint_store is None:
            checked = replace(checked, checkpoint_store=MODE.loader_store(checked.place_at_state,
                                                                          checked.start_for))
        if checked.checkpoint_store is not None and (checked.place_at_state is None or checked.start_for is None):
            # P16 reads its start artefact from the same store P5's bank places from, so the loaders
            # are derived from the store that was supplied rather than required a second time. They
            # are derived and never accepted from the caller here, so a placeholder cannot slip past.
            from .code_domain import place_at_state_factory, start_for_factory
            checked = replace(checked, place_at_state=place_at_state_factory(checked.checkpoint_store),
                              start_for=start_for_factory(checked.checkpoint_store))
        # THE ASSAY FIRST, THEN THE SETUP GATE, AND THE ORDER IS THE REPAIR. Both refusals stand
        # before the arms loop and therefore before the first paid call, so ordering costs nothing
        # and decides only which failure a run is told about. It was the other way round, and behind
        # `require_confirmatory_inputs` the assay was unreachable: `mode._CONFIG_RULES` asks a
        # configuration for bank states, retention fractions, replicates and a positive margin, which
        # are the P5 bank's fields and which no P16Config carries, so every P16 confirmatory run was
        # refused by the mode gate before this line was reached and no run could ever be refused on
        # assay grounds. That defect belongs to another finding's file and is not repaired here; the
        # ordering means the assay does not depend on its repair. The two questions remain different:
        # the mode gate asks whether the apparatus is present, and this asks whether what the source
        # will hand back is a measurement. A source that returns the assumed balance has restated the
        # answer the arm exists to measure; an undeclared one has said nothing about what it measured
        # at all; and neither is rescued by a complete apparatus, which is why this one speaks first.
        OBS.require_assay(spec)
        MODE.require_confirmatory_inputs(checked)
        inputs_record = checked.as_record()
    bundle = CUSTODY.as_bundle(bundle)
    # The readings, on for a run that is saving evidence and off for one that is not, exactly as in
    # P5. A real arm reads the whole frozen pool once per round through `code_domain.SuiteLadder`,
    # and until this line those readings had nowhere to go.
    reads = CUSTODY.attach_read_log(live_ladder) if bundle is not None else None
    if bundle is not None:
        # The live log goes to the bundle so that each arm's progress line carries the readings taken
        # since the previous one; without it they accumulate in memory and a run that stops loses
        # every reading it paid for.
        bundle.attach_reads(reads)
    rng = np.random.default_rng(seed)
    # THE CANDIDATE LABEL GOES INTO THE SEALED CONFIGURATION, not beside it. The configuration is
    # inside `custody.spec_hash_of`, so a run cannot have the label removed afterwards and be read as
    # though its bands had been registered; the names are a field of the configuration itself and the
    # values and the sentence travel with them so that the manifest names each candidate in words.
    man = M.new_manifest("P16", pilot, ladder_sha256,
                         cfg.__dict__ | {"seed": seed,
                                         "candidate_quantities_declared": candidate_record(cfg)},
                         adapter_name,
                         mode=m, confirmatory_inputs=inputs_record,
                         ladder=(live_ladder if live_ladder is not None else ladder_sha256))
    preds = {"line": predicted_line(cfg), "signs": predicted_signs(cfg),
             "timing_tolerance": cfg.timing_tolerance,
             # the timing the tolerance is a tolerance ON, sealed with the line it is part of. Before
             # finding A2 the tolerance was sealed alone and never read back by any decision.
             "timing": predicted_timing(cfg),
             # the quantity is a commitment like the line and the signs, so it is sealed with them
             "observation": spec.as_record(),
             # AND SO IS THE RULE THE ARMS WILL BE READ UNDER (finding A3): the look schedule, the
             # threshold, the variance estimator, the terminal reading, the informative horizon, the
             # practical-absence band and the support and refutation rules. A decision family chosen
             # after the arms are in hand is not a pre-run commitment, and a family that is not
             # written down cannot have its error rate calibrated.
             "sequential": SEQ.rule_from_config(cfg).as_record()}
    if m is MODE.ExecutionMode.CONFIRMATORY:
        anchor = checked.anchor if anchor is None else anchor
        attestation = checked.attestation if attestation is None else attestation
    # The seal, the attestation of what was unseen, and the receipt: before the switch, before any
    # arm runs, and on disk before the first arm runs.
    M.seal_predictions(man, preds, sealed_by, anchor=anchor, attestation=attestation)
    if bundle is not None:
        bundle.write_seal(man)
    arms = []

    def _collect(arm: Dict[str, Any], replicate_id: int) -> None:
        """One arm, kept and recorded. ONE PROGRESS LINE PER ARM (finding A8's second crash case): the
        arms used to reach disk only in the final write, so a run that stopped in the middle of the
        titration left its commitment and none of the series it had already paid for.

        The replicate identity travels with the arm because a re-scoring has to be able to say which
        arms it was given: a list of arms carrying only their labels cannot be checked against the
        universe the configuration says was collected."""
        arm["replicate_id"] = replicate_id
        arms.append(arm)
        if bundle is not None:
            bundle.record_progress("arm", {"arm": arm})

    for o in cfg.dose_offsets:
        for k in range(cfg.systems_per_arm):
            _collect(run_arm(margin_source, "dose%+.1f" % o, cfg.alpha_crit_hat + o, cfg, rng, spec), k)
    for k in range(cfg.systems_per_arm):
        _collect(run_arm(margin_source, "sham", cfg.alpha_crit_hat, cfg, rng, spec), k)
        _collect(run_arm(margin_source, "baseline", cfg.alpha_crit_hat, cfg, rng, spec), k)
    # THE CLAIM, CHECKED THE ONLY WAY IT CAN BE. A log was attached to the ladder this run was told
    # about; if the source reads that ladder, every arm's every round put a reading in it. An empty
    # log after the arms have been collected says the object named here is not the object being
    # read, and the identity sealed above it then describes a pool nothing in this run ever touched.
    # It cannot be asked earlier: nothing can show which ladder a source reads until the source has
    # read one, and a probing read before the arms would cost a call and change the measurement. So
    # the evidence is written first and the refusal comes after it, which is this file's rule for
    # everything a stop can interrupt.
    ladder_unread = bool(reads is not None and arms and not reads)
    series_keys = ("margin", "readings")
    out = {"manifest": man, "sealed": preds, "replication": replication,
           "arms": [{k: v for k, v in a.items() if k not in series_keys} for a in arms],
           # AND THE SERIES THEMSELVES, BESIDE THE SUMMARIES RATHER THAN INSIDE THEM. The arms carry
           # what a printed summary can carry, which is why the series were taken off them; a local
           # evidence bundle preserves what the run returned, which is why the series may not simply
           # be dropped from it. They travel here, keyed to the arm and the replicate they belong to,
           # so a re-scoring can join them back and recompute the readings from the measurements.
           "arm_series": [{"arm": a["arm"], "replicate_id": a.get("replicate_id"),
                           **{k: a[k] for k in series_keys if k in a}} for a in arms]}
    if not pilot:
        # The live ladder goes into the verdict for the reason it does in P5: the sealed verifier
        # binding is recomputed from the object in hand rather than read back out of the manifest that
        # recorded it.
        out["verdicts"] = verdicts(man, preds, arms, cfg, replication, ladder=live_ladder)
    if bundle is not None:
        # WHAT THE PROVIDER SAID IT SERVED, kept rather than discarded (finding A8). A run given the
        # adapter records the adapter's own account; a run given only a name records the name and
        # says that is all it was given, which is a statement a reader can act on and is not a guess
        # about a provider nobody handed over.
        provider = {"arms": (CUSTODY.adapter_metadata(adapter) if adapter is not None else
                             {"adapter": adapter_name,
                              "note": "this run was given the adapter's name and not the adapter, so "
                                      "what it served is recorded by name alone"})}
        out["bundle_path"] = bundle.write_bundle(
            CUSTODY.build_bundle(man, "P16", cfg, sealed=preds, arms=arms, verdicts_=out.get("verdicts"),
                                 replication=replication, reads=reads, provider=provider,
                                 ladder=(live_ladder if live_ladder is not None else ladder_sha256)))
    if ladder_unread and m is MODE.ExecutionMode.CONFIRMATORY:
        raise CUSTODY.CustodyRefusal(
            "this run was given the ladder %s and turned its read log on, and %d arm(s) later the "
            "log is empty: the margin source does not read the ladder this run named, so the "
            "verifier binding sealed with these predictions is about another pool. The evidence "
            "collected has been written; the result is not readable at proposition level"
            % (str(getattr(live_ladder, "sha256", None))[:16], len(arms)), ("domain-ladder",))
    # Both names for the readings, and the evidence status beside them, for the reason P5 exports
    # both: a key called `verdicts` on a simulated titration is quoted as one.
    return M.label_result(out)


def mock_margin_source(cfg: P16Config, true_alpha_crit: float, jump: float = 0.15, slope_scale: float = 0.01):
    """A simulated margin: trend (true_alpha_crit - alpha_arm) * slope_scale per round after the switch,
    a level jump at the switch (real or sham), noise throughout.

    Its numbers are unchanged and its declaration is new. What it produces is a per-round SURPLUS
    RATE, which is the quantity the four balance objects in code_domain.BalanceTracker report and is a
    level object: its trend of 0.01 per unit dose per round is a rate of change of a rate, and the
    sealed line's minus (1 minus chi) is d Delta / d alpha. There is no conversion between them
    without Q, W and the recursive coordinate R, so the verdict refuses to compare them and this
    source cannot decide a proposition. `mock_balance_source` is the one that demonstrates the
    registered coordinate; it decides no proposition either, because it declares itself a simulation
    and the deciding path refuses a simulation exactly as it refuses the mock anchor.
    """
    def src(arm, alpha_arm, r, rng):
        base = 0.5
        if r < cfg.switch_round:
            return base + rng.normal(0, cfg.margin_noise)
        trend = 0.0 if arm in ("sham", "baseline") else (true_alpha_crit - alpha_arm) * slope_scale * (r - cfg.switch_round)
        return base + jump + trend + rng.normal(0, cfg.margin_noise)
    return OBS.declare(src, OBS.surplus_rate_observation(
        source="mock-surplus-rate", simulated=True,
        note="the reference simulated margin: a per-round surplus rate with an imposed linear trend "
             "of %g per unit dose per round. It is not the balance elasticity and its trend is not "
             "the sealed slope." % slope_scale))


def mock_balance_source(cfg: P16Config, true_alpha_crit: float = 2.0, chi: Optional[float] = None,
                        baseline_offset: float = -0.4, coefficient_b: float = 1.0,
                        coefficient_q: float = 1.0, noise: Optional[float] = None):
    """A simulated system on the REGISTERED coordinate: Q and W separately, from the marginal-burden model.

    U = I R^alpha, W = b dU/dR and Q = q U^chi, so log(Q/W) = constant + (1 - alpha (1 - chi)) log R
    and the slope of the log ratio in log R is Delta exactly. An arm's Delta is therefore
    1 - alpha_arm (1 - chi), the fitted line across arms has slope minus (1 minus chi) and zero at
    1 / (1 - chi), and those are the sealed line's two numbers rather than numbers in some other
    coordinate that happen to share a zero. This is the reconciliation finding A1 asks for: where Q,
    W and R are present no conversion is needed, because the measured quantity is Delta already.

    The world's boundary is its chi and not a free parameter: chi is derived from `true_alpha_crit`
    through the crossover 1 / (1 - chi) unless it is given, so a world whose boundary sits away from
    the sealed one is built by moving the boundary and not by moving the seal.

    Before the switch, and in the sham and baseline arms throughout, the exponent is the baseline one.
    The path is continuous at the switch, so the post-switch window is a pure power in R with the
    arm's own exponent and the level jump the settling period exists to exclude is a jump and not a
    change of shape. The noise is multiplicative on Q and on W separately, which is where measurement
    error in a ratio actually enters.
    """
    chi = (1.0 - 1.0 / float(true_alpha_crit)) if chi is None else float(chi)
    noise = cfg.margin_noise if noise is None else float(noise)
    alpha_base = float(true_alpha_crit) + float(baseline_offset)
    R_switch = float(cfg.switch_round + 1)

    def src(arm, alpha_arm, r, rng):
        R = float(r + 1)
        dosed = not (r < cfg.switch_round or arm in ("sham", "baseline"))
        a = float(alpha_arm) if dosed else alpha_base
        log_U = alpha_base * math.log(min(R, R_switch))
        if R > R_switch:
            log_U += a * math.log(R / R_switch)
        U = math.exp(log_U)
        W = coefficient_b * a * U / R                      # b dU/dR along the power path
        Q = coefficient_q * U ** chi
        if noise > 0:
            W *= math.exp(rng.normal(0, noise))
            Q *= math.exp(rng.normal(0, noise))
        # THE APPARATUS'S OWN RECORD THAT THE DOSE WAS ADMINISTERED (finding A2). It is an
        # attestation and not an inference: an arm whose dose never landed and an arm whose dose
        # landed and did nothing produce the same series, and only the apparatus can tell them apart.
        return {"round": r, "Q": Q, "W": W, "R": R, "U": U,
                "extra": {OBS.DELIVERY_KEY: {"applied": bool(dosed),
                                             "lever": ("growth exponent %.4f" % a) if dosed else None}}}

    return OBS.declare(src, OBS.log_service_ratio_observation(
        supplies_q_and_w=True, chi_hat=chi, source="mock-marginal-burden-model", simulated=True,
        note="U = I R^alpha, W = b dU/dR, Q = q U^chi, with chi = %g and a crossover at %g. The "
             "series is log(Q/W) read against log R, so its slope is the balance elasticity. It is a "
             "simulation and is refused on the deciding path."
             % (chi, 1.0 / (1.0 - chi) if chi < 1.0 else float("inf"))))
