"""Finding A2: P16's support branch enforces every component the design promises, with its uncertainty.

THE DEFECT THESE CASES PIN. The reference support branch counted reversals in the above-boundary
arms, counted non-reversals in the controls, checked that the fitted line was falling and that its
zero sat within a tolerance of the sealed zero, and declared support. Four registered requirements
were sealed, reported or promised and never enforced: the timing tolerance was written into the seal
and no decision ever read it back; the sealed slope magnitude was printed beside the fitted slope and
compared with nothing, so a line an order of magnitude too shallow supported the proposition as well
as the registered one; the location was a point distance with neither the fit's uncertainty nor the
boundary's calibration uncertainty in it; and the arm's ASSIGNED dose was used as the exposure the
line was fitted against, so a titration whose lever failed reported a line drawn against intentions.
Two more were absent altogether: the apparatus's record that the dose was administered, and the
fresh-data repetition the registered rule requires before any single run reads as support.

EVERY CASE BELOW RUNS A WORLD THROUGH `p16.run_p16`, OR A PAIR OF WORLDS DIFFERING IN ONE
REGISTERED QUANTITY, AND ASKS ONE QUESTION OF THE WRAPPER. The five acceptance cases the finding
names are the five sections, and the sentence they exist to enforce is the last one in it: none of
them may become complete P16 support merely because several alarms occurred. Each case therefore
asserts both which component caught the defect and that the overall result is not support.
"""
import math

import numpy as np
import pytest

from arc_instruments import verdicts as V
from arc_runner import custody, manifest as M, observation as OBS, p16, p16_components as C


# The numbers the contract does not register and the components refuse to invent: supplied here so
# that the comparisons they gate can be exercised at all. See P16Config.
CHI_SE = 0.02              # a well calibrated boundary
SLOPE_BAND = 0.15          # a registered equivalence band on d Delta / d alpha
# and the three finding A3 adds, for the same reason. The controls component reads each control's
# own silence rather than the alarm count alone, so a run with no registered informative horizon has
# censored controls and cannot say they did not shift; a registered world therefore registers these
# as well, exactly as it registers the two above.
HORIZON = 40               # rounds after the switch beyond which an absence is informative
BAND = 0.05                # the band about zero inside which a margin is practically nil
SEGMENTS = 2               # the resolution the across-window predicate is read at
REPLICATED = {"fresh_data": True, "same_rule_met": True, "source": "a second panel, seed 11"}


def _cfg(**kw):
    base = dict(systems_per_arm=3, horizon=96, chi_hat_se=CHI_SE, slope_equivalence=SLOPE_BAND,
                informative_horizon=HORIZON, practical_absence_band=BAND,
                across_window_segments=SEGMENTS)
    base.update(kw)
    return p16.P16Config(**base)


def acceptance_source(cfg, zero=2.0, line_slope=-0.5, exposure=None, change_at=None,
                      attest_delivery=True, noise=0.01, baseline_offset=-0.4):
    """A world built to the sealed line's two numbers, with each of them separately movable.

    log(Q/W) travels as Delta log R with Delta = line_slope * (alpha - zero), so the line fitted
    across the arms has slope `line_slope` and its zero at `zero`, and the two can be moved
    independently. `mock_balance_source` cannot do that: there both numbers come from one chi, so a
    world with a wrong slope necessarily has a wrong zero and the case the finding names, a correct
    zero with a wrong slope magnitude, cannot be built from it.

    U travels as R to the power of the REALISED exponent, which `exposure` may make differ from the
    assigned dose. `change_at` is the round the arm's exponent actually changes, which `attest`
    keeps separate from the round the apparatus says it dosed: the whole of the timing component is
    that the two can disagree.
    """
    exposure = exposure or (lambda arm, alpha_arm: float(alpha_arm))
    change_at = cfg.switch_round if change_at is None else int(change_at)
    alpha_base = float(zero) + float(baseline_offset)
    R_change = float(change_at + 1)

    def src(arm, alpha_arm, r, rng):
        R = float(r + 1)
        control = arm in ("sham", "baseline")
        changed = (r >= change_at) and not control
        a = exposure(arm, alpha_arm) if changed else alpha_base
        delta_base = line_slope * (alpha_base - zero)
        delta_arm = line_slope * ((float(alpha_arm) if changed else alpha_base) - zero)
        log_R0 = math.log(min(R, R_change))
        log_R1 = math.log(R / R_change) if R > R_change else 0.0
        log_U = alpha_base * log_R0 + a * log_R1
        log_ratio = delta_base * log_R0 + delta_arm * log_R1
        U = math.exp(log_U)
        W = a * U / R
        Q = W * math.exp(log_ratio)
        if noise > 0:
            W *= math.exp(rng.normal(0, noise))
            Q *= math.exp(rng.normal(0, noise))
        out = {"round": r, "Q": Q, "W": W, "R": R, "U": U}
        if attest_delivery:
            # the apparatus attests dosing at the SEALED switch round, whatever round the response
            # actually arrives at, which is what lets delivery pass while timing fails
            out["extra"] = {OBS.DELIVERY_KEY: {
                "applied": bool(r >= cfg.switch_round and not control),
                "lever": ("assigned growth exponent %.4f" % float(alpha_arm)) if not control else None}}
        return out

    return OBS.declare(src, OBS.log_service_ratio_observation(
        supplies_q_and_w=True, chi_hat=cfg.chi_hat, source="acceptance-case",
        note="a world whose balance line has slope %g and zero %g by construction"
             % (line_slope, zero)))


def _states(v):
    return v["component_states"]


# --------------------------------------------------------------------------------------------------
# The world every case is a departure from: every component satisfied, and still not support
# --------------------------------------------------------------------------------------------------

def test_the_registered_world_satisfies_every_component_and_is_support_only_once_repeated():
    cfg = _cfg()
    src = acceptance_source(cfg)
    v = p16.run_p16(src, cfg, 5, "none", "mock")["verdicts"]
    assert all(s == C.SATISFIED for n, s in _states(v).items() if n != C.REPETITION), _states(v)
    assert _states(v)[C.REPETITION] == C.NOT_SUPPLIED
    assert v["P16"] == V.AWAITING_REPLICATION
    assert v["provisional"] is True and "single run" in v["provisional_label"]
    # and the same world with a fresh-data repetition recorded
    v2 = p16.run_p16(src, cfg, 5, "none", "mock", replication=REPLICATED)["verdicts"]
    assert v2["P16"] == V.SUPPORTED and v2["provisional"] is False


# --------------------------------------------------------------------------------------------------
# Acceptance case 1: correct zero with wrong slope magnitude
# --------------------------------------------------------------------------------------------------

def test_a_correct_zero_with_the_wrong_slope_magnitude_is_not_support():
    """The reference branch asked only whether the line was FALLING, so a line five times too shallow
    with its zero in exactly the right place was complete support."""
    cfg = _cfg()
    v = p16.run_p16(acceptance_source(cfg, zero=2.0, line_slope=-0.1), cfg, 5, "none", "mock",
                    replication=REPLICATED)["verdicts"]
    assert v["line_zero_fitted"] == pytest.approx(2.0, abs=0.1)
    assert v["line_slope_fitted"] == pytest.approx(-0.1, abs=0.03)
    assert v["falling_line_gate"] is True, "the reference gate is satisfied by this line"
    assert v["location_gate"] is True, "and so is the reference point-distance location gate"
    assert _states(v)[C.LOCATION] == C.SATISFIED
    assert _states(v)[C.SLOPE_MAGNITUDE] == C.FAILED
    assert v["P16"] == V.REFUTED
    assert "slope-magnitude" in v["component_reason"]


def test_the_slope_magnitude_is_not_checked_at_all_without_a_registered_equivalence():
    """No band is registered by the contract, and this module refuses to invent one: the component is
    NOT SUPPLIED and the wrapper withholds a proposition-level result rather than passing the line."""
    cfg = _cfg(slope_equivalence=None)
    v = p16.run_p16(acceptance_source(cfg, zero=2.0, line_slope=-0.1), cfg, 5, "none", "mock",
                    replication=REPLICATED)["verdicts"]
    assert _states(v)[C.SLOPE_MAGNITUDE] == C.NOT_SUPPLIED
    assert v["P16"] == V.NOT_EVALUABLE
    assert "slope-magnitude" in v["component_reason"]


# --------------------------------------------------------------------------------------------------
# Acceptance case 2: the reversal falls outside the timing band
#
# TWO HALVES, AND THE FIRST ALONE IS NOT THE CASE. A reversal falls outside the sealed band
# either by never being declared at all, which is what the first test holds and what finding
# A3's variance rule produces on a window straddling its own change point, or by being declared
# late, which is the case finding A2 actually names. Both are here. The second was missing
# while the first stood in for it, so the case's own precondition, a reversal that happened,
# went unexercised.
# --------------------------------------------------------------------------------------------------

def test_a_reversal_outside_the_sealed_timing_band_is_not_support():
    """`timing_tolerance` was sealed and never read back, so a response arriving thirty rounds after
    the manipulation counted exactly as one arriving at the switch.

    THE EVENT COMPONENT'S STATE HERE CHANGED WITH FINDING A3, and for a reason that belongs to this
    world. The sequential rule's standard error is now autocorrelation-robust, and the window this
    world is read over straddles its own change point: a single line fitted from round 14 across a
    series that is flat until round 40 has strongly autocorrelated residuals, so the robust interval
    is two to three times the independence one and the diluted window slope no longer clears the
    threshold. Under the independence standard error these arms declared at a ratio of about 4.9,
    which is exactly the margin a dependence correction removes. The timing component still fails on
    the measured change points, which is what this case exists to pin, and the wrapper still refutes.
    """
    cfg = _cfg()
    v = p16.run_p16(acceptance_source(cfg, change_at=40), cfg, 5, "none", "mock",
                    replication=REPLICATED)["verdicts"]
    assert _states(v)[C.DELIVERY] == C.SATISFIED, "the apparatus did dose at the sealed switch"
    assert _states(v)[C.EVENT] == C.UNRESOLVED, \
        "a window straddling the change point does not declare under the robust standard error"
    assert v["refutation"]["refutes"] is False, "and its silence is not a demonstrated absence either"
    assert _states(v)[C.TIMING] == C.FAILED
    timing = [c for c in v["components"] if c["component"] == C.TIMING][0]
    assert timing["detail"]["sealed_window"] == [4, 12]
    assert all(row["change_point"]["round"] > 12 for row in timing["detail"]["arms"])
    assert v["P16"] == V.REFUTED


def test_the_change_point_and_not_the_declaration_round_is_what_the_window_holds():
    """A detection latency is not a change point. On this world the switch is at round 8 and the
    change point is recovered there, while the reversal is not declared until the growing window has
    accumulated the precision to put its interval wholly below zero, which happens well outside the
    sealed tolerance of four rounds. A rule that compared the declaration round with that window
    would refute every true world."""
    cfg = _cfg()
    arms = p16.run_p16(acceptance_source(cfg), cfg, 5, "none", "mock")["arms"]
    above = [a for a in arms if a["arm"].startswith("dose+")]
    assert all(a["change_point"]["round"] == pytest.approx(8, abs=2) for a in above)
    assert all(a["declared_round"] > 12 for a in above), \
        "the declaration round is a latency and would fail a timing window it is not measured against"


# THE BAND A DECLARED-BUT-LATE WORLD HAS TO SIT IN, MEASURED RATHER THAN GUESSED. The growing
# window opens at the switch plus the settling period, which is round 14 here, so where the change
# point falls decides two things at once, and both edges of the usable band were measured over
# sixty seeds rather than picked. Below about round 16 the change-point estimator's own scatter
# puts an arm or two back inside the sealed window and the component reads UNRESOLVED instead of
# FAILED: at round 14 that happened in five of sixty runs. Above about round 24 the window
# straddles its own change point far enough that the residuals carry the dependence finding A3's
# conservative-maximum standard error exists to charge for, and the arms begin to fall silent: at
# round 26 six or seven arms of nine declare, at round 30 four to six, and at round 40 none, which
# is the world the silent case in this section holds. Between the two edges the arms declare UNDER
# the robust rule and still change late, which is the half this section was missing. At round 20:
# nine of nine above-boundary arms declare in all sixty runs, the earliest change point measured
# is 17 against a window that closes at 12, and fifty-nine of the sixty put every component except
# timing at SATISFIED. The round is a fixture's choice and not a registered quantity, and it
# pre-empts no registration: the case reaches the same states at a threshold of 3.0 as at the
# shipped 4.0, so restoring it settles nothing about which of those two governs.
LATE_CHANGE = 20


def test_a_declared_reversal_whose_change_point_falls_outside_the_sealed_window_is_not_support():
    """The other half of this acceptance case: a response that ARRIVES, and arrives late.

    The case above holds a world whose reversal is never declared at all, so its timing component
    fails on change points that no alarm accompanies. That is a true reading of that world and it is
    the one finding A3's variance rule produces on a window straddling its own change point, but it
    is not the case finding A2 names. The sentence there is that a reversal falling outside the
    sealed band is not support, and a reversal that never happened cannot test it: the precondition
    of the case, a reversal that happened, went unexercised while the silent world stood in for it.
    This world supplies it. The dose lands at the sealed switch, the response arrives twelve rounds
    later against a sealed tolerance of four, every other component is satisfied, and the run is
    still refused, on the timing tolerance alone.

    NONE OF THAT IS BOUGHT BY WEAKENING THE VARIANCE RULE. The arms are read under the same
    conservative maximum of the independence and the autocorrelation-robust standard errors; that
    maximum is the robust figure on every above-boundary arm and between two and three and a third
    times the independence one, so the correction is charged here exactly as it is charged in the
    case above; and each arm still puts its unselected terminal reading seven to twenty standard
    errors below zero against a registered threshold of four. The declaration is neither a
    look-schedule artefact nor the independence assumption's.

    AND THE LATENESS DOES TOUCH ONE OTHER MEASUREMENT, WHICH IS WHY IT IS CHECKED AND NOT ASSUMED.
    The window opens six rounds before the exponent changes, so the realised exposure measured over
    it is pulled a little towards the baseline: the arm nearest the boundary is assigned 2.3 and
    measures 2.2583. That is a true reading of an arm that spent part of its window undosed, and it
    is still wholly above the sealed boundary, so the assignment's side survives the lateness and
    the component that would otherwise mask this case stays satisfied.
    """
    cfg = _cfg()
    res = p16.run_p16(acceptance_source(cfg, change_at=LATE_CHANGE), cfg, 5, "none", "mock",
                      replication=REPLICATED)
    v = res["verdicts"]
    above = [a for a in res["arms"] if a["arm"].startswith("dose+")]

    # the reversal happened, and the components say so
    assert len(above) == 9 and all(a["declared_round"] is not None for a in above), \
        "the precondition of this acceptance case is a reversal that was declared"
    assert _states(v)[C.EVENT] == C.SATISFIED
    assert _states(v)[C.DELIVERY] == C.SATISFIED, "the apparatus dosed at the sealed switch"

    # and it was declared under the robust rule, not despite it
    for a in above:
        est = a["arm_estimate"]
        assert est["variance_estimator"] == OBS.CONSERVATIVE_MAX
        assert est["se"] == est["se_robust"] > est["se_independent"], \
            "the standard error the rule read is the widened one, so finding A3 is in force here"
        assert -est["delta"] / est["se"] > cfg.z_threshold, \
            "and the arm clears the threshold on the one look in the schedule that is not selected"

    # the change point is late, and it is the change point that the sealed window holds
    assert _states(v)[C.TIMING] == C.FAILED
    timing = [c for c in v["components"] if c["component"] == C.TIMING][0]
    assert timing["detail"]["sealed_window"] == [4, 12]
    assert timing["detail"]["n_inside"] == 0 and timing["detail"]["n_outside"] == len(above)
    assert all(row["change_point"]["round"] > 12 for row in timing["detail"]["arms"])
    assert "changed outside the sealed window" in timing["reason"]

    # nothing else in this run is wrong, and it is still not support
    assert v["components_failed"] == [C.TIMING]
    assert v["components_unresolved"] == [] and v["components_not_supplied"] == []
    assert v["P16"] == V.REFUTED and v["P16"] != V.SUPPORTED
    assert v["run_pattern"] == "SUPPORTED", \
        "the reference branch reads this run as complete support, which is the defect A2 names"


def test_lateness_alone_is_the_difference_between_support_and_refusal():
    """One registered titration, read twice, with the response on time and then twelve rounds late.

    The controlled form of the sentence above. Both runs carry the same sealed line, the same doses,
    the same attested delivery, the same fresh-data repetition and the same seed, and the reference
    branch's own reading calls both of them complete support. Exactly one component moves between
    them, and the proposition-level result moves with it: a tolerance that no decision reads back is
    a tolerance that changes nothing, and this is what it looks like when one does.
    """
    cfg = _cfg()
    on_time = p16.run_p16(acceptance_source(cfg), cfg, 5, "none", "mock",
                          replication=REPLICATED)["verdicts"]
    late = p16.run_p16(acceptance_source(cfg, change_at=LATE_CHANGE), cfg, 5, "none", "mock",
                       replication=REPLICATED)["verdicts"]
    assert on_time["run_pattern"] == late["run_pattern"] == "SUPPORTED"
    assert [n for n in C.CONTRACT_COMPONENTS
            if _states(on_time)[n] != _states(late)[n]] == [C.TIMING]
    assert on_time["P16"] == V.SUPPORTED
    assert late["P16"] == V.REFUTED and late["P16"] != V.SUPPORTED


# --------------------------------------------------------------------------------------------------
# Acceptance case 3: assigned dose above the boundary, realised exposure below it
# --------------------------------------------------------------------------------------------------

def test_an_arm_assigned_above_the_boundary_whose_realised_exposure_is_below_it_is_not_support():
    """The reference runner used the assigned `alpha_arm` both as the dose and as the exposure the
    line was fitted against, so a lever that fails cannot be seen at all."""
    cfg = _cfg()

    def failed_lever(arm, alpha_arm):
        # the apparatus was asked for 2.6 and the system realised 1.4: the dose did not land
        return 1.4 if arm == "dose+0.6" else float(alpha_arm)

    v = p16.run_p16(acceptance_source(cfg, exposure=failed_lever), cfg, 5, "none", "mock",
                    replication=REPLICATED)["verdicts"]
    assert _states(v)[C.DELIVERY] == C.SATISFIED, "the apparatus attests it applied the dose"
    assert _states(v)[C.REALISED_EXPOSURE] == C.FAILED
    got = [c for c in v["components"] if c["component"] == C.REALISED_EXPOSURE][0]
    assert "assigned above the boundary" in got["reason"]
    assert v["P16"] != V.SUPPORTED and v["P16"] == V.REFUTED
    # and the line is fitted against what the arms realised, not against what they were assigned
    assert v["line_fitted_on"] == "realised-exposure"
    xs = {row["arm"]: row["x"] for row in v["line_fit"]["arms"]}
    assert xs["dose+0.6"] == pytest.approx(1.4, abs=0.05)


def test_a_source_that_measures_no_realised_exposure_supplies_none_and_the_line_decides_nothing():
    """The reference per-round surplus-rate source carries no capability reading, so there is no
    realised exponent to fit against. The assigned doses are still fitted, because the run pattern
    and every earlier bundle are read from that line, and the components refuse it by name."""
    cfg = _cfg(systems_per_arm=2, horizon=60)
    v = p16.run_p16(p16.mock_margin_source(cfg, true_alpha_crit=2.0), cfg, 5, "none",
                    "mock")["verdicts"]
    assert v["line_fitted_on"] == "assigned-dose"
    assert _states(v)[C.REALISED_EXPOSURE] == C.NOT_SUPPLIED
    assert _states(v)[C.LOCATION] == C.NOT_SUPPLIED
    assert _states(v)[C.SLOPE_MAGNITUDE] == C.NOT_SUPPLIED
    assert _states(v)[C.DELIVERY] == C.NOT_SUPPLIED
    assert v["run_pattern"] == "SUPPORTED", "the arm pattern is unchanged and still says this"
    assert v["P16"] == V.NOT_EVALUABLE, "and the proposition is not decided by it"


# --------------------------------------------------------------------------------------------------
# Acceptance case 4: unresolved boundary uncertainty
# --------------------------------------------------------------------------------------------------

def test_a_boundary_whose_calibration_uncertainty_is_wide_leaves_the_location_unresolved():
    """The sealed zero is 1 / (1 - chi) at the calibrated chi, so a poorly calibrated chi is a poorly
    located boundary. The reference point distance could not see that: it compared the fitted zero
    with the sealed zero as though the second were exactly known."""
    cfg = _cfg(chi_hat_se=0.2)
    v = p16.run_p16(acceptance_source(cfg), cfg, 5, "none", "mock",
                    replication=REPLICATED)["verdicts"]
    assert v["location_gate"] is True, "the reference point-distance gate passes this world"
    assert _states(v)[C.LOCATION] == C.UNRESOLVED
    loc = [c for c in v["components"] if c["component"] == C.LOCATION][0]
    assert loc["detail"]["se_sealed_zero_from_calibration"] > cfg.location_tolerance
    assert v["P16"] == V.INCONCLUSIVE


def test_an_unregistered_calibration_uncertainty_is_not_a_zero():
    """Defaulting the unknown to perfect knowledge is the point comparison wearing another name."""
    cfg = _cfg(chi_hat_se=None)
    v = p16.run_p16(acceptance_source(cfg), cfg, 5, "none", "mock",
                    replication=REPLICATED)["verdicts"]
    assert _states(v)[C.LOCATION] == C.NOT_SUPPLIED
    assert "calibration uncertainty is not registered" in \
        [c for c in v["components"] if c["component"] == C.LOCATION][0]["reason"]
    assert v["P16"] == V.NOT_EVALUABLE


# --------------------------------------------------------------------------------------------------
# Acceptance case 5: replication pending
# --------------------------------------------------------------------------------------------------

def test_replication_pending_is_never_support_however_many_alarms_occurred():
    """The sentence finding A2 ends on. Nine of nine above-boundary arms reversed, no control moved,
    the line is the sealed line and its zero is the sealed zero, and this is still a single run."""
    cfg = _cfg()
    v = p16.run_p16(acceptance_source(cfg), cfg, 5, "none", "mock")["verdicts"]
    event = [c for c in v["components"] if c["component"] == C.EVENT][0]
    assert event["detail"]["n_declared"] == event["detail"]["n_above_boundary_arms"] == 9
    assert _states(v)[C.EVENT] == C.SATISFIED and _states(v)[C.CONTROLS] == C.SATISFIED
    assert _states(v)[C.DISCRIMINATION] == C.SATISFIED
    assert v["run_pattern"] == "SUPPORTED"
    assert v["P16"] == V.AWAITING_REPLICATION and v["P16"] != V.SUPPORTED


def test_a_repetition_that_is_not_on_fresh_data_is_not_a_repetition():
    cfg = _cfg()
    for record in ({"fresh_data": False, "same_rule_met": True, "source": "re-analysis"},
                   {"same_rule_met": True}):
        v = p16.run_p16(acceptance_source(cfg), cfg, 5, "none", "mock", replication=record)["verdicts"]
        assert _states(v)[C.REPETITION] == C.NOT_SUPPLIED
        assert v["P16"] == V.AWAITING_REPLICATION
    v = p16.run_p16(acceptance_source(cfg), cfg, 5, "none", "mock",
                    replication={"fresh_data": True, "same_rule_met": False})["verdicts"]
    assert _states(v)[C.REPETITION] == C.FAILED
    assert v["P16"] == V.AWAITING_REPLICATION, "a failed repetition on one run is still provisional"


# --------------------------------------------------------------------------------------------------
# Delivery and the controls: the two the reference branch could not distinguish
# --------------------------------------------------------------------------------------------------

def test_an_unattested_arm_is_unsupplied_evidence_and_never_a_defaulted_yes():
    cfg = _cfg()
    v = p16.run_p16(acceptance_source(cfg, attest_delivery=False), cfg, 5, "none", "mock",
                    replication=REPLICATED)["verdicts"]
    assert _states(v)[C.DELIVERY] == C.NOT_SUPPLIED
    assert v["P16"] == V.NOT_EVALUABLE


def test_a_control_that_was_dosed_is_a_delivery_failure_and_not_a_control():
    arms = [{"arm": "dose+0.6", "alpha": 2.6,
             "delivery": {"attested": True, "n_applied": 80, "first_applied_round": 8}},
            {"arm": "sham", "alpha": 2.0,
             "delivery": {"attested": True, "n_applied": 80, "first_applied_round": 8}}]
    got = C.delivery_component(arms, switch_round=8)
    assert got.state == C.FAILED and "is a control" in got.reason


def test_a_dose_that_landed_at_the_wrong_round_is_a_delivery_failure():
    arms = [{"arm": "dose+0.6", "alpha": 2.6,
             "delivery": {"attested": True, "n_applied": 80, "first_applied_round": 30}}]
    got = C.delivery_component(arms, switch_round=8)
    assert got.state == C.FAILED and "sealed switch is round 8" in got.reason


def test_generic_deterioration_in_every_arm_fails_the_controls_and_never_supports():
    """Controls that move are not a specificity failure at the detection rule's own error rate, and
    are one when every control moves. The reference branch's control gate was a bare majority."""
    controls = [{"arm": "sham", "declared_round": 20}, {"arm": "baseline", "declared_round": 22},
                {"arm": "sham", "declared_round": 24}, {"arm": "baseline", "declared_round": 25},
                {"arm": "sham", "declared_round": 26}, {"arm": "baseline", "declared_round": 28}]
    assert C.controls_component(controls, 0.052).state == C.FAILED
    one = list(controls)
    for a in one[1:]:
        a["declared_round"] = None
    assert C.controls_component(one, 0.052).state == C.UNRESOLVED, \
        "one alarm in six controls is the rule's own calibrated error rate, not deterioration"
    for a in one:
        a["declared_round"] = None
    assert C.controls_component(one, 0.052).state == C.SATISFIED


# --------------------------------------------------------------------------------------------------
# The wrapper's own rules
# --------------------------------------------------------------------------------------------------

def _comp(name, state):
    return C.Component(name, state, "for the test")


def test_the_wrapper_refuses_a_result_while_any_required_component_is_unsupplied():
    comps = [_comp(n, C.SATISFIED) for n in C.CONTRACT_COMPONENTS]
    comps[3] = _comp(C.TIMING, C.NOT_SUPPLIED)
    got = C.combine(comps)
    assert got["P16"] == V.NOT_EVALUABLE and C.TIMING in got["reason"]


def test_the_wrapper_never_reaches_support_from_a_failure_or_an_unresolved_component():
    for state, expected in ((C.FAILED, V.REFUTED), (C.UNRESOLVED, V.INCONCLUSIVE)):
        comps = [_comp(n, C.SATISFIED) for n in C.CONTRACT_COMPONENTS]
        comps[5] = _comp(C.SLOPE_MAGNITUDE, state)
        assert C.combine(comps)["P16"] == expected


def test_the_wrapper_refuses_a_component_the_run_did_not_report_at_all():
    comps = [_comp(n, C.SATISFIED) for n in C.CONTRACT_COMPONENTS if n != C.DISCRIMINATION]
    got = C.combine(comps)
    assert got["P16"] == V.NOT_EVALUABLE and C.DISCRIMINATION in got["reason"]


def test_an_unknown_component_state_is_refused_rather_than_counted():
    with pytest.raises(ValueError):
        C.Component(C.EVENT, "PROBABLY FINE", "")


# --------------------------------------------------------------------------------------------------
# The line, and the uncertainty the reference fit never carried
# --------------------------------------------------------------------------------------------------

def test_the_fitted_zero_carries_a_standard_error_that_grows_with_the_scatter():
    xs = [1.4, 1.7, 2.3, 2.6, 2.9]
    clean = C.fit_line(xs, [-0.5 * (x - 2.0) for x in xs])
    noisy = C.fit_line(xs, [-0.5 * (x - 2.0) + d for x, d in zip(xs, [0.2, -0.2, 0.2, -0.2, 0.2])])
    assert clean["zero"] == pytest.approx(2.0) and noisy["zero"] == pytest.approx(2.0, abs=0.4)
    assert clean["se_zero"] == pytest.approx(0.0, abs=1e-9)
    assert noisy["se_zero"] > 0.1, "the same point estimate from scattered arms is not the same result"


def test_a_line_from_fewer_than_three_arms_is_not_fitted():
    got = C.fit_line([1.5, 2.5], [0.25, -0.25])
    assert not np.isfinite(got["slope"]) and "fewer than three" in got["reason"]


# --------------------------------------------------------------------------------------------------
# The bundle re-scores to the same table, components and all
# --------------------------------------------------------------------------------------------------

def test_a_bundle_re_scores_to_the_same_components_and_the_same_wrapper_result(tmp_path):
    cfg = _cfg(systems_per_arm=2, horizon=60)
    res = p16.run_p16(acceptance_source(cfg), cfg, 5, "none", "mock", replication=REPLICATED,
                      bundle=str(tmp_path / "evidence"))
    bundle = custody.load_bundle(str(tmp_path / "evidence"))
    assert bundle["replication"] == REPLICATED
    again = custody.recompute_verdicts(bundle)
    assert again["P16"] == res["verdicts"]["P16"]
    assert again["component_states"] == res["verdicts"]["component_states"]
    assert bundle["sealed_predictions"]["timing"]["window"] == [4, 12]


def test_a_bundle_sealed_before_the_timing_window_existed_still_re_scores():
    """Only the tolerance was sealed then, and the switch round it surrounds is in the configuration,
    which is itself inside the sealed specification hash. Reconstructing the window there reads a
    commitment rather than inventing one, and the component says that it did."""
    cfg = _cfg(systems_per_arm=2, horizon=60)
    res = p16.run_p16(acceptance_source(cfg), cfg, 5, "none", "mock", replication=REPLICATED)
    old_preds = {k: v for k, v in res["sealed"].items() if k != "timing"}
    # THE OLD RECORD IS SEALED OVER THE OLD PAYLOAD, which is what makes it an old record rather than
    # a new one handed a shortened argument. A manifest keeps the payload it committed to, and a
    # scoring run whose predictions differ from that payload is refused: that refusal is the point of
    # keeping the payload, so this case has to produce a record whose commitment IS the old shape.
    old_man = M.new_manifest("P16", False, "none", cfg.__dict__ | {"seed": 5}, "mock")
    M.seal_predictions(old_man, old_preds, "the test")
    again = p16.verdicts(old_man, old_preds, res["arms"], cfg, REPLICATED)
    timing = [c for c in again["components"] if c["component"] == C.TIMING][0]
    assert timing["state"] == C.SATISFIED and "reconstructed" in timing["reason"]
    assert again["P16"] == res["verdicts"]["P16"]
