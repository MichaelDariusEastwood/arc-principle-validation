"""Finding A3: the absence of a P16 alarm is neither demonstrated positivity nor a resolved refutation.

THE DEFECT THESE CASES PIN. The reference runner read a silent arm as "AS PREDICTED" wherever no
reversal was predicted, and read a majority of silent above-boundary arms as `REFUTED (no reversal)`.
Both readings turn the absence of a measurement into a measurement. An arm can be silent because its
margin was measured firmly positive past a horizon at which the event would have been seen, which is
evidence; because its margin was measured and is practically nil, which is a different piece of
evidence; because it is too noisy or too short to decide anything, which is no evidence at all; or
because it was stopped before the event could have appeared, which is censoring. The reference
runner produced one label for all four. Its threshold comment reported an approximately 0.052
flat-null alarm rate from 400 series as though it were demonstrated control at 0.05, and its
detection branch required a strictly positive standard error, so a deterministic declining series
never reached the alarm branch at all.

EVERY ACCEPTANCE CASE THE FINDING NAMES IS A SECTION BELOW: pure noise, highly autocorrelated flat
data, an informative persistently positive crossing arm, an uninformative non-alarm, and the
noiseless positive and negative cases. The required result is correct classification and a
calibrated error, not an alarm in every mock.
"""
import numpy as np
import pytest

from arc_instruments import mc_interval as MC, verdicts as V
from arc_runner import observation as OBS, p16, p16_components as C, p16_sequential as SEQ

import p16_calibration as CAL

INCONCLUSIVE_SILENT = "INCONCLUSIVE (no reversal, and none demonstrated absent)"


def _cfg(**kw):
    """The candidate registered configuration: the bands, the horizon and the across-window
    resolution a decision needs.

    `across_window_segments` is the resolution the registered predicate is read at: a lower bound
    above the band on the whole window AND on each segment of it. Two is the candidate number and it
    is measured rather than chosen: on the world whose margin is firmly positive everywhere the
    margin sustains in ninety of ninety arms in halves and forty-seven of ninety in thirds, while the
    flat null sustains nothing in ninety arms in either.
    """
    base = dict(systems_per_arm=2, horizon=64, chi_hat_se=0.02, slope_equivalence=0.15,
                informative_horizon=40, practical_absence_band=0.05, across_window_segments=2)
    base.update(kw)
    return p16.P16Config(**base)


def _run(cfg, line_slope=0.0, zero=2.0, level=0.0, rho=0.0, noise=0.02, seed=11):
    """One world through the runner, built on the registered coordinate by the calibration script.

    The worlds come from `p16_calibration` rather than from a copy here, so that the script whose job
    is to calibrate this rule is exercised by the same cases that fix the rule's behaviour.
    """
    w = CAL.world("case", line_slope=line_slope, zero=zero, level=level, rho=rho, noise=noise)
    return p16.run_p16(CAL.source_for(w, cfg), cfg, seed, "none", "mock")


def _states(v, arm_prefix="dose+"):
    return [c["state"] for name, rows in v["non_alarm"].items() if name.startswith(arm_prefix)
            for c in rows]


# --------------------------------------------------------------------------------------------------
# Acceptance case 1: pure noise
# --------------------------------------------------------------------------------------------------

def test_pure_noise_is_a_practical_absence_when_measured_and_low_information_when_not():
    """The same silence, twice, and the classification follows the measurement rather than the silence.

    A flat world whose margin is exactly zero and whose noise is small enough for the terminal
    interval to sit inside the registered band has DEMONSTRATED a practical absence: the margin is
    measured and it is nil. The same world at five times the noise has demonstrated nothing, and the
    difference between the two is invisible to a rule that only counts alarms.
    """
    tight = _run(_cfg(), noise=0.02)["verdicts"]
    assert set(_states(tight, "dose")) == {SEQ.PRACTICAL_ABSENCE}, tight["non_alarm"]
    loose = _run(_cfg(), noise=0.10)["verdicts"]
    assert set(_states(loose, "dose")) == {SEQ.LOW_INFORMATION}, loose["non_alarm"]
    for v in (tight, loose):
        assert v["refutation"]["refutes"] is False
        assert v["refutation"]["n_demonstrated_positive"] == 0
        assert v["run_pattern"] == INCONCLUSIVE_SILENT
        assert v["P16"] != V.REFUTED and v["P16"] != V.SUPPORTED


def test_pure_noise_never_reaches_a_refutation_however_many_arms_are_silent():
    """The sentence the finding ends on, at the run level: the arms are silent for the whole horizon
    and the run says inconclusive, because no arm's margin was shown to be anything.

    One of these nine arms does alarm, which is the detection rule's own calibrated false-alarm rate
    appearing where it is expected to: at the flat null a few arms in a hundred declare, and a run
    with twenty-one arms in it meets that regularly. It changes nothing here. The majority needed for
    a reversal is not there, and neither is the demonstrated positivity needed for a refutation, so
    the run decides in neither direction.
    """
    v = _run(_cfg(systems_per_arm=3), noise=0.10)["verdicts"]
    above = [a for name, rows in v["non_alarm"].items() if name.startswith("dose+") for a in rows]
    assert len(above) == 9
    silent = [a for a in above if a["state"] != SEQ.EVENT_DECLARED]
    assert len(silent) > 4 and all(a["state"] == SEQ.LOW_INFORMATION for a in silent)
    assert v["refutation"]["refutes"] is False
    assert v["run_pattern"] not in ("REFUTED (no reversal)", "SUPPORTED")
    assert "Absence of an alarm is neither a positive margin nor a refutation" in v["refutation"]["reason"]


# --------------------------------------------------------------------------------------------------
# Acceptance case 2: highly autocorrelated flat data
# --------------------------------------------------------------------------------------------------

def test_highly_autocorrelated_flat_data_is_widened_by_the_robust_standard_error():
    """The temporal-correlation half of the frozen rule, measured on the arms themselves.

    A flat null whose rounds are correlated at 0.9 carries excursions that an independence standard
    error reads as precision. The registered estimator is Newey and West's at Andrews' automatic
    bandwidth, and on these arms it is several times the independence figure; the rule takes the
    larger of the two, so the interval can only widen.
    """
    res = _run(_cfg(), rho=0.9, noise=0.05)
    ratios = []
    for a in res["arms"]:
        t = a["terminal"]
        assert t["variance_estimator"] == OBS.CONSERVATIVE_MAX
        assert t["se"] >= t["se_independent"] - 1e-12, "the rule's error is never the narrower one"
        if t["se_independent"] > 0 and t["se_robust"] is not None:
            ratios.append(t["se_robust"] / t["se_independent"])
    assert np.median(ratios) > 1.5, "a correlated series is not measured as well as an independent one"


def test_highly_autocorrelated_flat_data_is_not_read_as_a_demonstrated_absence():
    """And what the classification does with it: nothing is demonstrated, which is the correct
    reading of a series whose own dependence is the largest thing in it."""
    v = _run(_cfg(), rho=0.9, noise=0.05)["verdicts"]
    silences = [c for rows in v["non_alarm"].values() for c in rows if c["state"] != SEQ.EVENT_DECLARED]
    assert silences, "this world does leave silent arms"
    assert sum(c["state"] == SEQ.LOW_INFORMATION for c in silences) > len(silences) / 2
    assert v["refutation"]["refutes"] is False
    assert v["P16"] != V.SUPPORTED


def test_the_short_window_clause_of_the_frozen_rule_is_measurable_and_registered():
    """Short-window behaviour, which the finding requires the frozen rule to include.

    The earliest looks of a growing window are the ones a persistent series wins on: at four points
    no variance estimator can see a dependence length of eighteen, so those looks fall back on
    something close to the independence figure whatever is registered. The size of the first look is
    therefore part of the rule, it is sealed with it, and what it costs is measured rather than
    assumed. The reference rule's four is kept as the default so that the repair changes no run on
    its own.
    """
    alarms = {}
    for first in (4, 24):
        cfg = _cfg(min_look_points=first)
        assert cfg.min_look_points == first
        res = _run(cfg, rho=0.9, noise=0.05, seed=101)
        alarms[first] = sum(a["declared_round"] is not None for a in res["arms"])
        assert res["sealed"]["sequential"]["min_points"] == first
    assert alarms[24] < alarms[4], alarms


# --------------------------------------------------------------------------------------------------
# Acceptance case 3: an informative persistently positive crossing arm
# --------------------------------------------------------------------------------------------------

def test_a_persistently_positive_crossing_arm_past_the_horizon_is_what_refutes():
    """The only evidence the registered rule accepts for `REFUTED (no reversal)`.

    `arc_instruments.verdicts.p16_prohibition` reaches REFUTED from a margin whose LOWER BOUND stays
    above zero across the window, and `p16_endpoint` from an event absent past an INFORMATIVE
    horizon. This world is both: no arm reverses, every above-boundary arm's terminal interval sits
    wholly above the registered band, and every arm ran forty rounds past the switch.
    """
    v = _run(_cfg(), level=0.30)["verdicts"]
    above = [c for name, rows in v["non_alarm"].items() if name.startswith("dose+") for c in rows]
    assert all(c["state"] == SEQ.DEMONSTRATED_SIGN and c["sign"] == SEQ.POSITIVE for c in above)
    assert all(c["interval"][0] > _cfg().practical_absence_band for c in above)
    assert v["refutation"]["refutes"] is True
    assert v["refutation"]["n_demonstrated_positive"] == len(above)
    assert v["run_pattern"] == "REFUTED (no reversal)"
    assert "majority the registered rule requires" in v["refutation"]["reason"]


def test_a_demonstrated_absence_of_the_event_is_what_fails_the_event_component():
    """And the component says the same thing in the wrapper's vocabulary: the event component fails
    on a demonstrated absence and is unresolved on a bare one."""
    demonstrated = _run(_cfg(), level=0.30)["verdicts"]
    assert demonstrated["component_states"][C.EVENT] == C.FAILED
    assert "demonstrated rather than merely observed" in \
        [c for c in demonstrated["components"] if c["component"] == C.EVENT][0]["reason"]
    bare = _run(_cfg(), noise=0.10)["verdicts"]
    assert bare["component_states"][C.EVENT] == C.UNRESOLVED
    assert bare["P16"] != V.REFUTED


# --------------------------------------------------------------------------------------------------
# Acceptance case 4: an uninformative non-alarm
# --------------------------------------------------------------------------------------------------

def test_an_unregistered_informative_horizon_censors_every_silence_and_refutes_nothing():
    """Fail closed. With no horizon registered, no length of silence has been said to be informative,
    so the same firmly positive world that refutes above cannot refute here."""
    v = _run(_cfg(informative_horizon=None), level=0.30)["verdicts"]
    silences = [c for rows in v["non_alarm"].values() for c in rows]
    assert set(c["state"] for c in silences) == {SEQ.CENSORED}
    assert all("no informative horizon is registered" in c["reason"] for c in silences)
    assert v["refutation"]["refutes"] is False
    assert v["run_pattern"] == INCONCLUSIVE_SILENT
    assert v["component_states"][C.EVENT] == C.UNRESOLVED


def test_an_arm_stopped_before_the_horizon_is_censored_and_not_an_absence():
    """The same world watched for fewer rounds than the registered horizon requires. The margin is
    just as positive; the EVENT is unobserved rather than absent, and censoring dominates."""
    v = _run(_cfg(horizon=40, informative_horizon=40), level=0.30)["verdicts"]
    silences = [c for rows in v["non_alarm"].values() for c in rows]
    assert set(c["state"] for c in silences) == {SEQ.CENSORED}
    assert all("unobserved rather than absent" in c["reason"] for c in silences)
    assert all(c["interval"] is not None for c in silences), \
        "the terminal interval is still reported: censoring hides no measurement"
    assert v["refutation"]["refutes"] is False


def test_a_silent_arm_says_which_silence_it_was_in_its_own_label():
    """The per-arm label the reference runner printed as a bare AS PREDICTED."""
    v = _run(_cfg(), level=0.30)["verdicts"]
    assert any("positive margin demonstrated" in s for rows in v["per_arm"].values() for s in rows)
    censored = _run(_cfg(informative_horizon=None), level=0.30)["verdicts"]
    assert all("censored: the event is unobserved, not absent" in s
               for rows in censored["per_arm"].values() for s in rows)


# --------------------------------------------------------------------------------------------------
# Acceptance case 5: the noiseless positive and negative cases
# --------------------------------------------------------------------------------------------------

DELTA = OBS.balance_elasticity_observation(source="acceptance-case")


def _constant(value):
    def src(arm, alpha_arm, r, rng):
        return value
    return OBS.declare(src, DELTA)


def test_a_noiseless_declining_series_reaches_the_alarm_branch_that_required_a_positive_error():
    """The reference detection branch required `se > 0` before it would declare, so a deterministic
    series with zero residual never reached it and the acceptance case of a constant negative Delta
    could not fire the event it defines. A zero-residual window is a perfectly measured one, and its
    interval is a point that is either below zero or is not."""
    cfg = _cfg(informative_horizon=20)
    arm = p16.run_arm(_constant(-0.20), "dose+0.6", 2.6, cfg, np.random.default_rng(3))
    assert arm["declared_round"] == cfg.switch_round + cfg.settling + cfg.min_look_points - 1
    assert arm["terminal"]["se"] < 1e-12 and arm["terminal"]["se_robust"] < 1e-12
    got = SEQ.classify(arm["terminal"], SEQ.rule_from_config(cfg))
    assert got["state"] == SEQ.EVENT_DECLARED


def test_a_noiseless_positive_series_demonstrates_its_sign_rather_than_going_unread():
    """The same series with the opposite sign: no alarm, and a silence that has measured something.
    A point interval wholly above the registered band is the strongest demonstration there is."""
    cfg = _cfg(informative_horizon=20)
    arm = p16.run_arm(_constant(0.20), "dose+0.6", 2.6, cfg, np.random.default_rng(3))
    assert arm["declared_round"] is None
    got = SEQ.classify(arm["terminal"], SEQ.rule_from_config(cfg))
    assert got["state"] == SEQ.DEMONSTRATED_SIGN and got["sign"] == SEQ.POSITIVE
    assert got["interval"] == [pytest.approx(0.20), pytest.approx(0.20)]


def test_a_noiseless_series_inside_the_band_is_a_practical_absence_and_not_a_positive_margin():
    """Strict clearance, in this direction too (ruling 17): a margin of plus 0.01 inside a band of
    0.05 is practically nil, and reading it as a demonstrated positive margin would refute the
    proposition on a measurement that says the margin is zero."""
    cfg = _cfg(informative_horizon=20, practical_absence_band=0.05)
    arm = p16.run_arm(_constant(0.01), "dose+0.6", 2.6, cfg, np.random.default_rng(3))
    got = SEQ.classify(arm["terminal"], SEQ.rule_from_config(cfg))
    assert got["state"] == SEQ.PRACTICAL_ABSENCE
    assert SEQ.refutation([dict(arm, alpha=2.6)], 2.0, SEQ.rule_from_config(cfg))["refutes"] is False


# --------------------------------------------------------------------------------------------------
# The refutation rule itself
# --------------------------------------------------------------------------------------------------

def _arm(name, alpha, delta, se, rounds_after=50, declared=None, n=40, segments=None,
         n_segments=2):
    """One arm's saved terminal record, at the registered across-window resolution.

    `segments` defaults to the same estimate twice, which is the record a constant arm writes: the
    margin the whole window measured is the margin each half of it measured. A case that needs the
    two halves to disagree passes them, which is the only way this helper can build the arm that used
    to refute wrongly: a mean above the band with a reversal inside the window.
    """
    seg = segments if segments is not None else [
        {"first_round": 14, "last_round": 14 + rounds_after // 2, "delta": delta, "se": se,
         "se_independent": se, "se_robust": se, "n_used": max(n // 2, 4), "n_excluded": 0},
        {"first_round": 15 + rounds_after // 2, "last_round": 14 + rounds_after, "delta": delta,
         "se": se, "se_independent": se, "se_robust": se, "n_used": max(n // 2, 4), "n_excluded": 0}]
    return {"arm": name, "alpha": alpha,
            "terminal": {"delta": delta, "se": se, "se_independent": se, "se_robust": se,
                         "variance_estimator": OBS.CONSERVATIVE_MAX, "n_used": n, "n_excluded": 0,
                         "first_round": 14, "last_round": 14 + rounds_after,
                         "rounds_after_switch": rounds_after, "declared_round": declared,
                         "looks_taken": 1, "looks_available": 1,
                         "across_window": (None if seg is None else
                                           {"n_segments": int(n_segments), "segments": list(seg)})}}


def test_refutation_needs_a_majority_of_demonstrated_positives_and_counts_nothing_else():
    rule = SEQ.rule_from_config(_cfg())
    positive = lambda i: _arm("dose+0.6", 2.6, 0.30, 0.01)
    low = _arm("dose+0.6", 2.6, 0.02, 0.20)                  # interval spans zero
    censored = _arm("dose+0.6", 2.6, 0.30, 0.01, rounds_after=5)
    assert SEQ.refutation([positive(0), positive(1), low, low, low], 2.0, rule)["refutes"] is False
    assert SEQ.refutation([positive(0), positive(1), positive(2), low, low], 2.0, rule)["refutes"] is True
    # the same three positives, two of them stopped early: censoring is not evidence of absence
    got = SEQ.refutation([positive(0), censored, censored, low, low], 2.0, rule)
    assert got["refutes"] is False and got["counts"][SEQ.CENSORED] == 2


def test_only_arms_above_the_sealed_boundary_enter_the_refutation():
    rule = SEQ.rule_from_config(_cfg())
    below = _arm("dose-0.6", 1.4, 0.30, 0.01)
    got = SEQ.refutation([below, below, below], 2.0, rule)
    assert got["n_above_boundary_arms"] == 0 and got["refutes"] is False
    assert "nothing to refute from" in got["reason"]


def test_an_arm_from_a_bundle_written_before_this_rule_is_not_measured_and_cannot_refute():
    """A bundle with no terminal record classifies as NOT MEASURED. Reconstructing a silence from a
    summary would be the defect this rule exists to remove, so it is refused instead."""
    rule = SEQ.rule_from_config(_cfg())
    old = {"arm": "dose+0.6", "alpha": 2.6, "declared_round": None, "window_slope": 0.3}
    got = SEQ.classify_arm(old, rule)
    assert got["state"] == SEQ.NOT_MEASURED
    assert SEQ.refutation([old, old, old], 2.0, rule)["refutes"] is False


def test_the_event_component_no_longer_fails_from_a_low_alarm_rate_alone():
    """Directly on the component. A FAILED component is a refutation in the wrapper, so a set of arms
    too noisy to decide anything used to refute exactly as a set of arms measured firmly positive."""
    arms = [{"arm": "dose+0.6", "alpha": 2.6, "declared_round": None} for _ in range(9)]
    assert C.event_component(arms, 2.0, 0.052).state == C.UNRESOLVED
    assert C.event_component(arms, 2.0, 0.052, {"refutes": False, "reason": "low information"}).state \
        == C.UNRESOLVED
    got = C.event_component(arms, 2.0, 0.052,
                            {"refutes": True, "n_demonstrated_positive": 9, "reason": "9 of 9"})
    assert got.state == C.FAILED and got.detail["demonstrated_refutation"] is True


# --------------------------------------------------------------------------------------------------
# The frozen rule, sealed and re-scored
# --------------------------------------------------------------------------------------------------

def test_the_sequential_rule_is_sealed_with_the_line_and_names_every_clause():
    res = _run(_cfg(), level=0.30)
    rule = res["sealed"]["sequential"]
    assert rule["look_schedule"].startswith("one look per round on a growing window")
    assert rule["threshold_z"] == 4.0 and rule["terminal_z"] == 4.0
    assert rule["variance_estimator"] == OBS.CONSERVATIVE_MAX
    assert rule["informative_horizon_rounds_after_switch"] == 40
    assert rule["practical_absence_band"] == 0.05
    assert "no control declared" in rule["control_aggregation"]
    assert "A non-alarm alone never refutes" in rule["refutation_rule"]
    assert set(rule["non_alarm_states"]) == set(SEQ.NON_ALARM_STATES)
    assert res["verdicts"]["family_size"]["arm_look_tests"] > 0


def test_a_bundle_re_scores_to_the_same_reading_of_every_silence():
    cfg = _cfg(systems_per_arm=1)
    res = _run(cfg, level=0.30)
    again = p16.verdicts(res["manifest"], res["sealed"], res["arms"], cfg)
    assert again["run_pattern"] == res["verdicts"]["run_pattern"] == "REFUTED (no reversal)"
    assert again["non_alarm"] == res["verdicts"]["non_alarm"]
    assert again["refutation"]["counts"] == res["verdicts"]["refutation"]["counts"]


def test_the_terminal_reading_uses_the_sequential_threshold_unless_one_is_registered():
    """The conservative reading, named as open in `P16Config`: a narrower single-look interval would
    demonstrate positivity, and so refute, more readily than the rule that declares the event."""
    assert SEQ.rule_from_config(_cfg()).terminal_z == _cfg().z_threshold
    assert SEQ.rule_from_config(_cfg(terminal_z=2.0)).terminal_z == 2.0
    cfg = _cfg(terminal_z=1.0, informative_horizon=20)
    arm = p16.run_arm(_constant(0.02), "dose+0.6", 2.6, cfg, np.random.default_rng(3))
    wide = SEQ.classify(arm["terminal"], SEQ.rule_from_config(_cfg(informative_horizon=20)))
    narrow = SEQ.classify(arm["terminal"], SEQ.rule_from_config(cfg))
    assert wide["interval"][1] - wide["interval"][0] >= narrow["interval"][1] - narrow["interval"][0]


# --------------------------------------------------------------------------------------------------
# The calibrated error, reported as an interval
# --------------------------------------------------------------------------------------------------

def test_the_reference_flat_null_figure_does_not_certify_control_at_five_per_cent():
    """The arithmetic the config comment now carries. Twenty-one alarms in 400 series is an observed
    0.0525 whose exact interval reaches 0.079, so the observation is consistent with a true rate half
    again above 0.05; and a rate of zero in 400 is not zero risk either."""
    got = SEQ.rate_with_interval(21, 400)
    assert got["rate"] == pytest.approx(0.0525)
    assert got["exact_one_sided_upper"] > 0.05
    assert got["exact_two_sided"][1] == pytest.approx(0.079, abs=0.002)
    assert MC.exact_upper(0, 400) > 0, "zero in four hundred is not zero risk"


def test_the_calibration_script_reports_every_rate_as_an_interval_and_never_as_a_point():
    """The script finding A3 asks for, exercised on a handful of seeds. What matters here is not
    which rates come back but that no rate comes back alone: a family-wise error is reported with
    its Monte Carlo uncertainty or it is not reported."""
    cfg = CAL.config(systems_per_arm=1, horizon=48, dose_offsets=(-0.6, 0.6))
    rows = CAL.calibrate(4, CAL.WORLDS[:3], cfg)
    assert len(rows) == 3
    for r in rows:
        for key in ("family_wise_error", "correct_claim_rate", "refutation_on_demonstrated_positivity"):
            d = r[key]
            lo, hi = d["exact_two_sided"]
            assert hi > lo, "%s came back as a point" % key
            assert lo <= d["rate"] <= hi
            assert d["outer_repetitions"] == 4
        assert r["family_size"]["arm_look_tests"] == \
            r["family_size"]["arms"] * r["family_size"]["looks_per_arm"]
        assert r["sequential_rule"]["variance_estimator"] == OBS.CONSERVATIVE_MAX
    report = CAL.render_report(rows, 4, "2026-09-06T00:00:00Z")
    assert "exact 95 per cent interval" in report
    assert "is not zero risk" in report
    for r in rows:
        assert r["world"] in report


def test_the_calibration_separates_the_per_arm_rate_from_the_family_rate():
    """The second half of the finding's complaint about 0.052: a per-arm rate is not the family's
    rate, and the two are never reported as one number."""
    cfg = CAL.config(systems_per_arm=2, horizon=48, dose_offsets=(-0.6, 0.6))
    row = CAL.calibrate_world(CAL.WORLDS[0], 4, cfg)
    assert set(row["per_arm_alarm_rate"]) == {"above", "below", "control"}
    assert row["per_arm_alarm_rate"]["above"]["outer_repetitions"] > row["family_wise_error"]["outer_repetitions"]
    assert "a per-arm rate is not the family's rate" in row["family_size"]["note"]


# --------------------------------------------------------------------------------------------------
# The predicate the contract names: a lower bound above the band ACROSS the window, not on its mean
# --------------------------------------------------------------------------------------------------

def _reversing(hi, lo, reversal_round):
    """A margin held at `hi` and then reversed to `lo` for the rest of the horizon.

    This is the arm the whole-window mean cannot see: the reversal the run was watching for has
    happened, and the mean of the window it happened in is still above the band.
    """
    def src(arm, alpha_arm, r, rng):
        return hi if r < reversal_round else lo
    return OBS.declare(src, DELTA)


def test_a_window_mean_above_the_band_is_not_the_across_window_predicate():
    """The arm that used to refute wrongly, run through the runner rather than hand-built.

    Plus 0.30 for forty-two rounds of the post-settling window and MINUS 0.05 for the last eight, so
    the margin is negative at the horizon and the reversal has plainly happened. Its whole-window
    interval is wholly above the registered band, which is what the terminal reading used to be; the
    registered predicate is `margin_lower_bound_positive_across_window`, and this arm does not meet
    it, so it demonstrates nothing and refutes nothing.
    """
    cfg = _cfg(informative_horizon=40)
    rule = SEQ.rule_from_config(cfg)
    arm = p16.run_arm(_reversing(0.30, -0.05, 56), "dose+0.6", 2.6, cfg, np.random.default_rng(3))
    assert arm["declared_round"] is None, "the sequential rule does not declare on this arm"
    got = SEQ.classify(arm["terminal"], rule)
    assert got["interval"][0] > cfg.practical_absence_band, \
        "the window MEAN clears the band, which is what made this arm refute"
    assert got["state"] == SEQ.NOT_SUSTAINED
    assert got["demonstrated"] is False and got["refutation_admissible"] is False
    assert "did not STAY above the band" in got["reason"]
    arms = [dict(arm, alpha=2.6) for _ in range(3)]
    ref = SEQ.refutation(arms, 2.0, rule)
    assert ref["refutes"] is False
    assert ref["n_demonstrated_positive"] == 0 and ref["counts"][SEQ.NOT_SUSTAINED] == 3
    assert ref["predicate"] == "margin_lower_bound_positive_across_window"
    assert C.event_component(arms, 2.0, cfg.alarm_rate_null, ref).state == C.UNRESOLVED


def test_the_two_readings_of_the_same_arm_are_reported_side_by_side():
    """The record says both things: the mean cleared the band and the window did not sustain it.
    A reading that hid one of them would leave the next reader with the mean alone again."""
    cfg = _cfg(informative_horizon=40)
    arm = p16.run_arm(_reversing(0.30, -0.05, 56), "dose+0.6", 2.6, cfg, np.random.default_rng(3))
    aw = SEQ.classify(arm["terminal"], SEQ.rule_from_config(cfg))["across_window"]
    assert aw["evaluated"] is True and aw["registered_segments"] == 2
    assert len(aw["segments"]) == 2
    assert aw["segments"][0]["interval"][0] > 0.05, "the first half is firmly above the band"
    assert aw["segments"][1]["interval"][0] < 0.05, "the second half is not, which is the reversal"
    assert aw["sustained_above"] is False


def test_a_hand_built_arm_whose_halves_disagree_cannot_refute_however_many_of_them_there_are():
    """The same statement directly on the rule, with the segments supplied rather than measured."""
    rule = SEQ.rule_from_config(_cfg())
    swinging = _arm("dose+0.6", 2.6, 0.30, 0.01, segments=[
        {"first_round": 14, "last_round": 38, "delta": 0.60, "se": 0.01, "se_independent": 0.01,
         "se_robust": 0.01, "n_used": 25, "n_excluded": 0},
        {"first_round": 39, "last_round": 64, "delta": 0.00, "se": 0.01, "se_independent": 0.01,
         "se_robust": 0.01, "n_used": 25, "n_excluded": 0}])
    got = SEQ.refutation([swinging] * 5, 2.0, rule)
    assert got["refutes"] is False and got["counts"][SEQ.NOT_SUSTAINED] == 5
    sustained = _arm("dose+0.6", 2.6, 0.30, 0.01)
    assert SEQ.refutation([sustained] * 5, 2.0, rule)["refutes"] is True


def test_an_unregistered_across_window_resolution_demonstrates_nothing():
    """Fail closed, as the horizon and the band already do. With no resolution registered the
    predicate the contract names is not evaluated, and the window mean alone is a weaker statement
    that no run may refute from."""
    v = _run(_cfg(across_window_segments=None), level=0.30)["verdicts"]
    above = [c for name, rows in v["non_alarm"].items() if name.startswith("dose+") for c in rows]
    assert above and all(c["state"] == SEQ.DEMONSTRATED_SIGN and c["sign"] == SEQ.POSITIVE
                         for c in above)
    assert all(c["demonstrated"] is False and c["refutation_admissible"] is False for c in above)
    assert all("no across-window resolution is registered" in c["reason"] for c in above)
    assert v["refutation"]["refutes"] is False
    assert v["run_pattern"] == INCONCLUSIVE_SILENT
    assert any("window mean only" in s for rows in v["per_arm"].values() for s in rows)


def test_the_refutation_carries_the_resolution_it_was_read_at():
    """What the refutation can and cannot have seen. Positivity demonstrated on segments of twenty-five
    rounds says nothing about a reversal ten rounds long, and a reader who is not told the segment
    length cannot know which reversals the refutation excludes."""
    v = _run(_cfg(), level=0.30)["verdicts"]
    res = v["refutation"]["resolution"]
    assert v["refutation"]["refutes"] is True
    assert res["across_window_segments"] == 2 and res["practical_absence_band"] == 0.05
    assert res["shortest_segment_rounds"] > 0
    assert "could not have been seen" in res["note"]
    assert v["sequential_rule"]["across_window_segments"] == 2
    assert "margin_lower_bound_positive_across_window" in \
        v["sequential_rule"]["across_window_predicate"]


# --------------------------------------------------------------------------------------------------
# A practically nil margin is not positivity, so a band is required before anything may refute
# --------------------------------------------------------------------------------------------------

def test_an_unregistered_band_makes_no_positivity_admissible():
    """The reading this module refuses in its own docstring, refused in the rule as well.

    With no band registered the module can only compare the sign with zero, so a noiseless margin of
    plus 0.001 has a point interval wholly above zero and would refute the proposition on a
    measurement that says the margin is nil. The sign is still reported; it decides nothing.
    """
    cfg = _cfg(informative_horizon=20, practical_absence_band=None)
    rule = SEQ.rule_from_config(cfg)
    arm = p16.run_arm(_constant(0.001), "dose+0.6", 2.6, cfg, np.random.default_rng(3))
    got = SEQ.classify(arm["terminal"], rule)
    assert got["state"] == SEQ.DEMONSTRATED_SIGN and got["sign"] == SEQ.POSITIVE
    assert got["interval"] == [pytest.approx(0.001), pytest.approx(0.001)]
    assert got["refutation_admissible"] is False
    assert "not admissible for refutation" in got["reason"]
    arms = [dict(arm, alpha=2.6) for _ in range(3)]
    ref = SEQ.refutation(arms, 2.0, rule)
    assert ref["refutes"] is False
    assert "practically nil margin is not positivity" in ref["reason"]
    assert C.event_component(arms, 2.0, cfg.alarm_rate_null, ref).state == C.UNRESOLVED


def test_an_unregistered_band_refuses_a_firmly_positive_arm_too():
    """And it is the missing registration that refuses it, not the size of the margin: the same arm
    at plus 0.20 is equally inadmissible, because nothing has said how large a margin must be before
    it is evidence that the reversal did not happen."""
    cfg = _cfg(informative_horizon=20, practical_absence_band=None)
    arm = p16.run_arm(_constant(0.20), "dose+0.6", 2.6, cfg, np.random.default_rng(3))
    got = SEQ.classify(arm["terminal"], SEQ.rule_from_config(cfg))
    assert got["state"] == SEQ.DEMONSTRATED_SIGN and got["refutation_admissible"] is False
    with_band = SEQ.classify(arm["terminal"], SEQ.rule_from_config(_cfg(informative_horizon=20)))
    assert with_band["refutation_admissible"] is True, "the band is the difference and nothing else"


# --------------------------------------------------------------------------------------------------
# The control aggregation: one silence, one reading
# --------------------------------------------------------------------------------------------------

def test_the_controls_are_not_satisfied_by_a_silence_they_never_measured():
    """The contradiction this clause removes. A run with no registered informative horizon classified
    every control silence as CENSORED, the event unobserved by the horizon, while the controls
    component simultaneously reported SATISFIED, neither the sham nor the baseline arms shifted. One
    run, two readings of the same silence, and the generous one decided."""
    v = _run(_cfg(informative_horizon=None), level=0.30)["verdicts"]
    controls = [c for name, rows in v["non_alarm"].items() if name in ("sham", "baseline")
                for c in rows]
    assert controls and all(c["state"] == SEQ.CENSORED for c in controls)
    assert v["component_states"][C.CONTROLS] == C.UNRESOLVED
    reason = [c for c in v["components"] if c["component"] == C.CONTROLS][0]["reason"]
    assert "an unobserved event is not an absent one" in reason


def test_the_controls_are_satisfied_where_their_own_silences_demonstrate_it():
    """And the clause is not a refusal to ever be satisfied: the same world with the horizon
    registered measures its controls past it and the component says so."""
    v = _run(_cfg(), level=0.30)["verdicts"]
    controls = [c for name, rows in v["non_alarm"].items() if name in ("sham", "baseline")
                for c in rows]
    assert all(c["demonstrated"] for c in controls)
    assert v["component_states"][C.CONTROLS] == C.SATISFIED
    detail = [c for c in v["components"] if c["component"] == C.CONTROLS][0]["detail"]
    assert detail["non_alarm"]["n_not_demonstrated"] == 0


def test_a_bundle_with_no_four_way_reading_says_which_evidence_the_controls_rest_on():
    """A bundle written before the sequential rule carries no terminal records to read, and a bundle
    that can no longer be re-scored is not evidence. Such a run keeps the older reading and says in
    its own reason that the alarm count is all it rests on."""
    old = [{"arm": "sham", "declared_round": None}, {"arm": "baseline", "declared_round": None}]
    got = C.controls_component(old, 0.052)
    assert got.state == C.SATISFIED and "no four-way reading" in got.reason
    assert C.controls_component(old, 0.052, []).state == C.SATISFIED, \
        "no silences to read is not an unmeasured silence"


# --------------------------------------------------------------------------------------------------
# The calibration: a world in which the mean reading and the across-window reading disagree
# --------------------------------------------------------------------------------------------------

def test_the_calibration_carries_a_world_whose_margin_is_not_stationary():
    """Every other world is constant in Delta within an arm, so the whole-window mean equals the
    terminal margin in all of them and a calibration built only from those worlds cannot see the
    difference the across-window predicate exists to see."""
    reversing = [w for w in CAL.WORLDS if w.get("reversal_round") is not None]
    assert reversing == [CAL.REVERSAL_SEEN, CAL.REVERSAL_INSIDE_SEGMENT], \
        "the family is calibrated in a world that reverses where the resolution sees it and in one " \
        "where it does not"
    for w in reversing:
        assert w["level"] > 0 and w["reversal_level"] < 0
        assert CAL.REFUTED_NO_REVERSAL in w["error_labels"]
    stationary = [w for w in CAL.WORLDS if w.get("reversal_round") is None]
    assert all(w["reversal_level"] == 0.0 for w in stationary), \
        "every other world is constant in Delta within an arm, which is what these two are not"


def test_the_family_never_refutes_in_the_world_that_reverses_late():
    """And the rate is measured rather than argued. The world's margin is firmly positive for most of
    the window and negative at the end, so a refutation there is a wrong claim; the mean alone cannot
    tell it from the world where refutation is warranted, and the segmented reading can."""
    cfg = CAL.config(systems_per_arm=1, dose_offsets=(-0.6, 0.6))
    row = CAL.calibrate_world(CAL.REVERSAL_SEEN, 4, cfg)
    assert row["refutation_on_demonstrated_positivity"]["rate"] == 0.0
    assert row["family_wise_error"]["rate"] == 0.0
    assert row["family_wise_error"]["exact_two_sided"][1] > 0.0, "zero in four runs is not zero risk"
    assert row["non_alarm_census"].get(SEQ.NOT_SUSTAINED, 0) > 0, \
        "the arms are read as not sustained, which is what this world is for"


def test_the_price_of_the_registered_resolution_is_measured_and_not_hidden():
    """The other side of the same clause, and it is a cost rather than a demonstration.

    A reversal confined to the last segment is averaged away by a two-segment reading, so the family
    refutes in a world where the reversal has happened. That is the resolution's price; it is
    calibrated in its own world rather than left for a reader to discover, and the segment length
    travels with every refutation so that a reader knows which reversals it excludes.
    """
    cfg = CAL.config(systems_per_arm=1, dose_offsets=(-0.6, 0.6))
    row = CAL.calibrate_world(CAL.REVERSAL_INSIDE_SEGMENT, 4, cfg)
    assert row["family_wise_error"]["rate"] > 0.0, \
        "the world exists because the registered resolution cannot see this reversal"
    assert "price of reading the window in two pieces" in row["truth"]
    seen = CAL.calibrate_world(CAL.REVERSAL_SEEN, 4, cfg)
    assert seen["family_wise_error"]["rate"] < row["family_wise_error"]["rate"], \
        "and the same rule sees the same reversal when it is not confined to one segment"


def test_the_same_family_still_refutes_in_the_world_where_refutation_is_warranted():
    """The control on the case above: the segmented reading is not simply refusing everything."""
    w = [w for w in CAL.WORLDS if w.get("reversal_round") is None and w["level"] > 0][0]
    cfg = CAL.config(systems_per_arm=1, dose_offsets=(-0.6, 0.6))
    row = CAL.calibrate_world(w, 4, cfg)
    assert row["refutation_on_demonstrated_positivity"]["rate"] == 1.0
    assert row["correct_claim_rate"]["rate"] == 1.0


# --------------------------------------------------------------------------------------------------
# The variance rule, which the frozen rule names and which was until now untested
# --------------------------------------------------------------------------------------------------

def test_every_registered_variance_estimator_is_reachable_and_recorded_by_name():
    """The moving-block bootstrap is the stated alternative to Newey and West, and a stated
    alternative nothing exercises is a claim rather than an implementation. Each estimator produces
    its own standard error, each is recorded on the estimate under its own name, and a name outside
    the registered set is refused rather than silently replaced by a default."""
    rng = np.random.default_rng(5)
    e = np.zeros(60)
    for i in range(1, 60):
        e[i] = 0.9 * e[i - 1] + rng.normal(0, 0.05)
    rd = OBS.readings_from_values(0.3 + e)
    spec = OBS.balance_elasticity_observation(source="variance-case")
    got = {}
    for name in OBS.VARIANCE_ESTIMATORS:
        est = OBS.estimate(rd, spec, name)
        assert est.variance_estimator == name
        got[name] = est.se
    assert got[OBS.INDEPENDENCE] < got[OBS.NEWEY_WEST], "a correlated series is not measured as well"
    assert got[OBS.BLOCK_BOOTSTRAP] > got[OBS.INDEPENDENCE], \
        "the block bootstrap sees the same dependence the kernel does"
    assert got[OBS.CONSERVATIVE_MAX] == pytest.approx(max(got[OBS.INDEPENDENCE], got[OBS.NEWEY_WEST]))
    with pytest.raises(OBS.ObservationRefusal):
        OBS.estimate(rd, spec, "some-estimator-nobody-registered")


def test_the_fixed_rule_of_thumb_recovers_about_a_fifth_of_the_long_run_variance():
    """The measurement behind the comment in `arc_runner.observation`, made executable.

    The fixed rule L = floor(4 (n/100)^(2/9)) gives three lags across this runner's window lengths.
    A Bartlett kernel truncated at three lags recovers 1 + 2 sum (1 - l/4) rho^l of a series whose
    long-run variance is (1 + rho)/(1 - rho): at rho = 0.9 that is 3.52 against 19, which is about a
    fifth, and a correction that recovers a fifth of what is missing is a correction that does not
    correct. Andrews' bandwidth on the same residuals is several times three.
    """
    rho, n = 0.9, 4000
    rng = np.random.default_rng(11)
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = rho * e[i - 1] + rng.normal(0, 1.0)
    e = e - e.mean()
    truth = float(np.var(e, ddof=1)) * (1 + rho) / (1 - rho)
    fixed = OBS._bartlett(e, 3) / n
    assert fixed / truth == pytest.approx(0.19, abs=0.04)
    assert OBS.bartlett_lag(e, n) > 3 * 3, "Andrews' bandwidth reads the dependence the series has"
    assert OBS.ar1_correlation(e) == pytest.approx(rho, abs=0.05)
    assert OBS.bartlett_lag(rng.normal(0, 1.0, n), n) < 3, \
        "and it costs nothing where there is no dependence to see"


def test_the_robust_standard_error_removes_alarms_the_independence_assumption_supported():
    """Not merely that the interval widens: that the widening changes decisions.

    The same autocorrelated flat null read under the independence assumption and under the registered
    estimator. The registered rule takes the larger of the two, so it can only remove alarms, and on
    these arms it removes some: an alarm that survives only the assumption of independence is the
    alarm this clause exists to remove.
    """
    counts = {}
    for name in (OBS.INDEPENDENCE, OBS.CONSERVATIVE_MAX):
        res = _run(_cfg(variance_estimator=name), rho=0.9, noise=0.05, seed=101)
        counts[name] = sum(a["declared_round"] is not None for a in res["arms"])
        assert all(a["terminal"]["variance_estimator"] == name for a in res["arms"])
    assert counts[OBS.CONSERVATIVE_MAX] < counts[OBS.INDEPENDENCE], counts


def test_the_census_accounts_for_every_above_boundary_arm():
    """No unaccounted remainder. A census that leaves rows out is how a reader comes to assume the
    rows left out were the reading they expected, which is the whole shape of this finding."""
    rule = SEQ.rule_from_config(_cfg())
    unread = SEQ.rule_from_config(_cfg(across_window_segments=None))
    rows = [_arm("dose+0.6", 2.6, 0.30, 0.01),                       # sustained above the band
            _arm("dose+0.6", 2.6, 0.02, 0.20),                       # spans zero
            _arm("dose+0.6", 2.6, 0.30, 0.01, rounds_after=5),       # stopped before the horizon
            _arm("dose+0.6", 2.6, 0.00, 0.001),                      # practically nil
            _arm("dose+0.6", 2.6, -0.30, 0.01)]                      # below the band, undeclared
    for r in (rule, unread):
        got = SEQ.refutation(rows, 2.0, r)
        assert sum(got["counts"].values()) == got["n_above_boundary_arms"] == len(rows)
        assert "unaccounted" not in got["reason"]
    mean_only = SEQ.refutation([_arm("dose+0.6", 2.6, 0.30, 0.01)] * 3, 2.0, unread)
    assert mean_only["refutes"] is False
    assert mean_only["counts"][
        SEQ.DEMONSTRATED_SIGN + " (positive, not admissible as refutation evidence)"] == 3
    assert mean_only["n_positive_mean_not_admissible"] == 3


def test_a_resolution_of_one_segment_is_refused_because_it_is_the_mean_again():
    """The one registration that would reinstate the defect under the name of the repair. A window
    read in one piece is its mean, and a mean cannot show that anything stayed anywhere."""
    cfg = _cfg(across_window_segments=1)
    with pytest.raises(OBS.ObservationRefusal) as exc:
        p16.run_arm(_constant(0.20), "dose+0.6", 2.6, cfg, np.random.default_rng(3))
    assert "not a reading across it" in str(exc.value)
    assert SEQ.rule_from_config(_cfg(across_window_segments=None)).across_window_segments is None
