"""P5's final comparison: the whole uncertainty, the whole assigned panel, one boundary convention.

The acceptance cases of finding A7, each written so that the OLD arithmetic fails it and the new one
passes: precise observations with uncertain predictions; a calibration error shared across a
system's readings; many systems reaching the ladder ceiling; uncertainty crossing the permitted
coupling domain; and exact contact with a boundary.

The per-system cases build the held-out record by hand rather than running a bank, because the
quantity under test is the comparison and not the measurement: a case that has to run a bank to
reach a difference of exactly the margin cannot reach it.
"""
import numpy as np
import pytest

from arc_runner import manifest as M, p5
from arc_runner import p5_prediction as PRED


# ------------------------------------------------------------------------------------------------
# Fixtures: a reportable manifest and a held-out record with the numbers the case needs


def _manifest(assigned, seed=1, predictions=None):
    """The manifest seals THE PREDICTIONS IT WILL BE SCORED AGAINST, as a run does.

    It used to seal a placeholder, so every case below scored one object against a commitment to
    another and the integrity check had nothing to compare. A fixture that cannot fail the check it
    is standing in for is not standing in for it."""
    man = M.new_manifest("P5", False, "ladder-sha", {"seed": seed, "assigned_heldout_systems": list(assigned)},
                         "test-adapter")
    M.seal_predictions(man, predictions if predictions is not None else {"sealed": True}, "the test")
    return man


def _prediction_record(se, *, within_domain=True, reason="", usable=True):
    """The propagation record a sealed prediction carries, with only the fields the verdict reads."""
    return {"se": se, "usable": usable, "reason": reason,
            "coupling_domain": {"within_domain": within_domain, "boundary_reached": not within_domain,
                                "reason": reason or "inside the domain", "excursion_fraction":
                                    0.0 if within_domain else 0.3}}


def _heldout(systems):
    """`systems` maps a name to (predicted, predicted_se, observed, observed_se, n_reps, headroom_ok)."""
    preds, fitted = {}, {}
    for name, (pred, pse, obs, ose, n, head) in systems.items():
        half = PRED.REGISTERED.equivalence_z * pse if np.isfinite(pse) else float("nan")
        preds[name] = {"predicted_exponent": pred, "predicted_half_width": half,
                       "prediction": _prediction_record(pse)}
        fitted[name] = {"fitted_exponent": obs, "fitted_se": ose, "n_replicates": n,
                        "headroom_ok": head, "replicates": []}
    return {"sealed_predictions": preds, "fitted": fitted, "assigned_systems": list(systems),
            "calibration": {}}


def _verdicts(heldout, cfg=None, assigned=None, premise="NOT REFUTED"):
    cfg = cfg or p5.P5Config()
    man = _manifest(assigned if assigned is not None else heldout["assigned_systems"],
                    predictions=heldout["sealed_predictions"])
    routes = {"identification": "NOT ESTABLISHED", "premise": premise}
    return p5.verdicts(man, routes, heldout, cfg)


# ------------------------------------------------------------------------------------------------
# Acceptance case: precise observations with uncertain predictions


def test_a_precise_observation_does_not_rescue_an_uncertain_prediction():
    """The difference interval carries BOTH uncertainties, so a tight continuation series compared
    with a loosely pinned prediction is not agreement.

    The old rule used the observed replicate standard error alone. Here the observation is exact to
    six decimal places and the prediction carries a standard error of 0.02, and the difference is
    0.07 against a margin of 0.10. On the old arithmetic the interval was the point, 0.07 was inside
    0.10, and the system agreed. With the prediction's uncertainty in it the interval reaches 0.113
    and the system decides nothing, which is what a bank that could not pin the prediction down has
    established.
    """
    h = _heldout({"S1": (1.50, 0.02, 1.57, 1e-6, 4, True)})
    v = _verdicts(h)
    rec = v["per_system"]["S1"]
    assert rec["verdict"] == "UNRESOLVED", rec
    assert rec["difference_interval"]["predicted_se"] == 0.02
    # and the prediction supplies almost the whole of the difference's variance, which the record says
    assert rec["difference_interval"]["share_of_variance_from_the_prediction"] > 0.99
    # the same numbers with the observation alone, which is the arithmetic being replaced
    old_half = p5.t95(3) * 1e-6
    assert abs(0.07) + old_half <= 0.10


def test_a_prediction_whose_uncertainty_cannot_be_propagated_is_not_treated_as_exact():
    """A missing width is not a zero width. The old code took a non-finite half width as no gate at
    all and compared the points, which is the strongest possible claim made from the least possible
    evidence."""
    h = _heldout({"S1": (1.50, float("nan"), 1.51, 0.001, 4, True)})
    rec = _verdicts(h)["per_system"]["S1"]
    assert rec["verdict"] == "UNRESOLVED"
    assert "unknown width is not read as a zero one" in rec["reason"]


def test_a_negative_predicted_half_width_is_refused_and_never_read_as_a_narrow_one():
    """A width below zero passed the precision condition more easily than any real interval.

    The condition asks whether twice the half width is under the margin, so minus one cleared it by a
    wider margin than any honest interval could, and the system was then compared as though its
    prediction were sharper than the instrument. A width that is not a width is not a narrow width.
    """
    h = _heldout({"S1": (1.5, 0.01, 1.50, 0.005, 4, True)})
    h["sealed_predictions"]["S1"]["predicted_half_width"] = -0.5
    v = _verdicts(h)
    rec = v["per_system"]["S1"]
    assert rec["verdict"] == "UNRESOLVED"
    assert rec["counted_in_denominator"] is True
    assert "neither is a negative" in rec["reason"]
    # UNRESOLVED is a reading and keeps its place in the denominator, so the panel is inconclusive
    # rather than unevaluable: the system was measured and the comparison was refused, which is not
    # the same statement as the system never having been measurable.
    assert v["PREDICTION"] == "INCONCLUSIVE"


def test_an_imprecise_prediction_fails_the_precision_condition_before_it_is_compared():
    """H2's precision condition, now read on the joint width and at the same level as the comparison."""
    h = _heldout({"S1": (1.50, 0.06, 1.50, 0.001, 4, True)})
    rec = _verdicts(h)["per_system"]["S1"]
    assert rec["verdict"] == "UNRESOLVED"
    assert "not narrower than the margin" in rec["reason"]


# ------------------------------------------------------------------------------------------------
# Acceptance case: shared calibration error


def _propagate(**kw):
    base = dict(beta_hat=0.5, beta_se=0.0, depths=[1, 2, 3, 4], scores=[100.0, 130.0, 152.0, 170.0],
                score_sds=[0.0, 0.0, 0.0, 0.0], r1=4, window_end=32, checkpoints=(4, 8, 16, 32),
                calibrate=p5.calibrate_rate, predict=p5.predicted_exponent,
                convention=PRED.REGISTERED, seed=11)
    base.update(kw)
    return PRED.propagate_prediction(**base)


def test_a_calibration_error_shared_across_a_systems_readings_reaches_the_prediction_interval():
    """A bias common to every calibration reading of one system moves the starting state, and the
    old width could not see it at all: it moved only the coupling and re-calibrated at each end from
    the same readings, so with an exactly known coupling it reported a width of zero.

    A shared error does not cancel. It shifts the starting state while leaving the fitted rate almost
    unmoved, and the predicted exponent is a function of both.
    """
    exact = _propagate(beta_se=0.0)
    # an exactly known coupling and exact readings: there is nothing to propagate and the interval
    # is a point, which is the width the endpoint arithmetic reported for EVERY case of this shape
    assert exact["half_width"] == pytest.approx(0.0, abs=1e-12)
    shared = _propagate(shared_calibration_sd=4.0)
    assert shared["usable"]
    assert shared["half_width"] > 0.001
    assert shared["components"]["calibration_only_sd"] > 0.001
    # and the coupling contributes nothing here, because the coupling is exact in this case
    assert shared["components"]["coupling_only_sd"] == pytest.approx(0.0, abs=1e-12)


def test_independent_read_error_and_shared_error_both_reach_the_interval_and_the_record_says_which():
    per_read = _propagate(score_sds=[3.0, 3.0, 3.0, 3.0])
    both = _propagate(beta_se=0.01, score_sds=[3.0, 3.0, 3.0, 3.0])
    assert per_read["usable"] and per_read["half_width"] > 0
    assert both["half_width"] > per_read["half_width"] * 0.99
    assert both["components"]["coupling_only_sd"] > 0
    assert both["components"]["calibration_only_sd"] > 0
    assert "starting state" in both["resampled"]


def test_the_joint_width_exceeds_the_endpoint_width_it_replaces_on_a_real_run():
    """The size of the repair, measured on the runner's own clean world rather than asserted.

    The joint width is read at the ninety per cent level and the endpoint width at ninety-five, so
    the joint one carries the SMALLER multiplier and is still wider: the difference is the
    calibration window's read error, which the endpoint width held fixed.
    """
    from arc_runner import adapters, ladder as L
    ad = adapters.MockCouplingAdapter(beta=0.5, theta=0.0, noise=0.03)
    lad = L.MockLadder(n_items=20000, scale=400.0)
    place = lambda s: {"kind": "mock", "capability": float(s), "rounds": 0}
    start = lambda n: {"kind": "mock", "capability": 20.0, "rounds": 0, "system": n}
    cfg = p5.P5Config(reps=8, window_end=32, checkpoints=(4, 8, 16, 32))
    res = p5.run_p5(ad, lad, cfg, 7, place, start, ["S1", "S2", "S3"])
    for name, sp in res["heldout"]["sealed_predictions"].items():
        assert sp["predicted_half_width"] > sp["endpoint_half_width_diagnostic"], name
        comp = sp["prediction"]["components"]
        assert comp["calibration_only_sd"] > comp["coupling_only_sd"], (name, comp)
    assert res["verdicts"]["panel"]["assigned"] == 3
    assert res["verdicts"]["PREDICTION"] == "SUPPORTED", res["verdicts"]["panel"]


# ------------------------------------------------------------------------------------------------
# Acceptance case: many systems reaching the ladder ceiling


def test_systems_that_reached_the_ceiling_keep_their_weight_in_the_denominator():
    """Three of five systems reach the ladder ceiling and two agree. The old rule dropped the three
    before counting and reported support from a majority of two out of two. The panel is the assigned
    five, the three keep their places as NOT EVALUABLE, and two of five is not a majority."""
    h = _heldout({"S1": (1.5, 0.01, 1.50, 0.005, 4, True),
                  "S2": (1.5, 0.01, 1.50, 0.005, 4, True),
                  "S3": (1.5, 0.01, 1.50, 0.005, 4, False),
                  "S4": (1.5, 0.01, 1.50, 0.005, 4, False),
                  "S5": (1.5, 0.01, 1.50, 0.005, 4, False)})
    v = _verdicts(h)
    panel = v["panel"]
    assert panel["assigned"] == 5
    assert panel["agrees"] == 2 and panel["not_evaluable"] == 3
    assert v["PREDICTION"] == "INCONCLUSIVE", panel
    # every assigned system is accounted for, and the counts add up to the panel
    assert panel["agrees"] + panel["disagrees"] + panel["unresolved"] + panel["not_evaluable"] == 5
    assert all(r.get("counted_in_denominator") for r in v["per_system"].values())


def test_a_majority_of_the_assigned_panel_still_reaches_support():
    h = _heldout({"S1": (1.5, 0.01, 1.50, 0.005, 4, True),
                  "S2": (1.5, 0.01, 1.50, 0.005, 4, True),
                  "S3": (1.5, 0.01, 1.50, 0.005, 4, True),
                  "S4": (1.5, 0.01, 1.50, 0.005, 4, False),
                  "S5": (1.5, 0.01, 1.50, 0.005, 4, False)})
    v = _verdicts(h)
    assert v["PREDICTION"] == "SUPPORTED", v["panel"]
    assert v["panel"]["agrees"] == 3 and v["panel"]["not_evaluable"] == 2


def test_a_panel_none_of_whose_systems_could_be_evaluated_is_not_evaluable():
    h = _heldout({"S1": (1.5, 0.01, 1.5, 0.005, 4, False), "S2": (1.5, 0.01, 1.5, 0.005, 4, False)})
    v = _verdicts(h)
    assert v["PREDICTION"] == "NOT EVALUABLE"
    assert v["panel"]["not_evaluable"] == 2


def test_the_denominator_is_the_sealed_panel_and_a_system_that_vanished_is_still_counted():
    """A system assigned before the run and absent from the results is NOT EVALUABLE, not absent.

    The assigned panel is read from the manifest's configuration, which is inside the sealed
    specification hash, so the denominator is a pre-run commitment: a run that lost a system between
    the seal and the verdict cannot become a run with a smaller denominator.
    """
    h = _heldout({"S1": (1.5, 0.01, 1.50, 0.005, 4, True), "S2": (1.5, 0.01, 1.50, 0.005, 4, True)})
    man = _manifest(["S1", "S2", "S3", "S4", "S5"], predictions=h["sealed_predictions"])
    v = p5.verdicts(man, {"identification": "NOT ESTABLISHED", "premise": "NOT REFUTED"}, h,
                    p5.P5Config())
    assert v["panel"]["assigned"] == 5
    assert v["panel"]["not_evaluable"] == 3          # the three that never reported
    assert v["PREDICTION"] == "INCONCLUSIVE"


def test_a_refuted_premise_still_withholds_support_over_the_assigned_panel():
    h = _heldout({"S1": (1.5, 0.01, 1.50, 0.005, 4, True), "S2": (1.5, 0.01, 1.50, 0.005, 4, True)})
    v = _verdicts(h, premise="REFUTED (curvature)")
    assert v["PREDICTION"] == "INCONCLUSIVE"
    assert v["PREMISE"] == "REFUTED (curvature)"


# ------------------------------------------------------------------------------------------------
# Acceptance case: uncertainty crossing the permitted coupling domain


def test_a_coupling_interval_reaching_the_domain_is_reported_and_never_clipped():
    """The old width clipped each end of the coupling's interval into [0, 0.95] and reported the
    clipped width as the prediction's uncertainty, which is narrower than the bank supports and says
    nothing about having reached the boundary at all."""
    out = _propagate(beta_hat=0.90, beta_se=0.05, score_sds=[1.0, 1.0, 1.0, 1.0])
    dom = out["coupling_domain"]
    assert dom["within_domain"] is False
    assert dom["boundary_reached"] is True
    assert dom["n_outside"] > 0
    assert dom["excursion_fraction"] > 0.0
    assert "LOWER BOUND" in dom["reason"]
    assert dom["coupling_interval"][1] > 0.95


def test_a_system_whose_prediction_reaches_the_coupling_domain_is_not_scored_against_it():
    h = _heldout({"S1": (1.50, 0.01, 1.50, 0.001, 4, True)})
    h["sealed_predictions"]["S1"]["prediction"] = _prediction_record(
        0.01, within_domain=False, reason="the coupling's interval reaches the permitted domain")
    rec = _verdicts(h)["per_system"]["S1"]
    assert rec["verdict"] == "UNRESOLVED"
    assert "permitted domain" in rec["reason"]
    assert rec["counted_in_denominator"] is True


def test_a_coupling_interval_inside_the_domain_says_so_in_words():
    out = _propagate(beta_hat=0.50, beta_se=0.02, score_sds=[1.0, 1.0, 1.0, 1.0])
    dom = out["coupling_domain"]
    assert dom["within_domain"] is True and dom["n_outside"] == 0
    assert "strictly inside the permitted domain" in dom["reason"]


def test_a_calibration_window_within_a_read_error_of_zero_is_refused_not_averaged():
    """A resampled pass count at or below zero is outside the support of the reading. Such a draw has
    no prediction rather than a missing one, and it used to disappear into the discarded draws
    without telling anybody the calibration window was that close to the bottom of the ladder."""
    out = _propagate(scores=[2.0, 3.0, 4.0, 5.0], score_sds=[4.0, 4.0, 4.0, 4.0], beta_se=0.01)
    dom = out["coupling_domain"]
    assert dom["degenerate_fraction"] > 0.0
    assert dom["within_domain"] is False
    assert "outside the support of a pass count" in dom["reason"]


# ------------------------------------------------------------------------------------------------
# Acceptance case: exact boundary contact


def test_exact_contact_with_the_margin_is_not_clearance():
    """Strict clearance, which is the candidate's convention and the one the later unison record
    uses. The rule this replaces read `<=` and called exact contact agreement, which decides the one
    case where the evidence decides nothing."""
    strict = PRED.REGISTERED
    closed = PRED.IntervalConvention(strict_clearance=False)
    assert 0.05 + 0.05 == 0.10                 # the contact is exact in the arithmetic, not near it
    assert strict.clearance(0.05, 0.05, 0.10)["verdict"] == "UNRESOLVED"
    assert "exact contact is not clearance" in strict.clearance(0.05, 0.05, 0.10)["reason"]
    assert closed.clearance(0.05, 0.05, 0.10)["verdict"] == "AGREES"


def test_exact_contact_from_outside_does_not_establish_disagreement_either():
    assert 0.2 - 0.1 == 0.1
    r = PRED.REGISTERED.clearance(0.2, 0.1, 0.1)
    assert r["verdict"] == "UNRESOLVED"
    assert "from outside" in r["reason"]


def test_exact_contact_with_the_coupling_domain_is_not_clearance():
    """The same convention governs the domain: an interval whose end lands exactly on 0.95 has not
    cleared it. The comparison is on the analytic endpoints and not on the draws, because a finite
    Monte Carlo can miss a boundary its interval genuinely reaches."""
    hi = PRED.REGISTERED.coupling_domain[1]
    z = PRED.REGISTERED.equivalence_z
    out = _propagate(beta_hat=hi - z * 0.01, beta_se=0.01, score_sds=[1.0, 1.0, 1.0, 1.0])
    assert out["coupling_domain"]["coupling_interval"][1] == pytest.approx(hi, abs=1e-12)
    assert out["coupling_domain"]["within_domain"] is False


def test_a_clearly_separated_difference_still_disagrees():
    r = PRED.REGISTERED.clearance(0.5, 0.02, 0.10)
    assert r["verdict"] == "DISAGREES"


# ------------------------------------------------------------------------------------------------
# The convention lives in one place


def test_the_configuration_defaults_are_the_registered_convention():
    cfg = p5.P5Config()
    reg = PRED.REGISTERED
    assert cfg.equivalence_z == reg.equivalence_z
    assert cfg.strict_clearance is reg.strict_clearance
    assert tuple(cfg.coupling_domain) == tuple(reg.coupling_domain)
    assert cfg.prediction_draws == reg.prediction_draws
    assert PRED.convention_from_config(cfg) == reg
    assert "two one-sided tests" in reg.as_record()["level"]
    assert reg.as_record()["registered_choice"].startswith("the 90 per cent")


def test_a_configuration_may_override_the_convention_in_that_one_place():
    cfg = p5.P5Config(strict_clearance=False, equivalence_z=1.96, coupling_domain=(0.0, 0.99))
    conv = PRED.convention_from_config(cfg)
    assert conv.strict_clearance is False and conv.equivalence_z == 1.96
    assert conv.coupling_domain == (0.0, 0.99)
    # and the override reaches the comparison rather than only the record
    h = _heldout({"S1": (1.50, 0.01, 1.55, 0.001, 4, True)})
    assert _verdicts(h, cfg=cfg)["per_system"]["S1"]["verdict"] == "AGREES"
    assert _verdicts(h)["per_system"]["S1"]["verdict"] == "AGREES"


def test_a_configuration_written_before_these_fields_reads_as_the_registered_convention():
    """A bundle re-scored from an older configuration is read under the registered convention, not
    under whatever a missing attribute would have defaulted to."""
    class Old:
        margin = 0.10
    assert PRED.convention_from_config(Old()) == PRED.REGISTERED
    assert PRED.convention_from_config(None) == PRED.REGISTERED
    assert PRED.convention_from_config({}) == PRED.REGISTERED


def test_the_verdict_block_carries_the_convention_it_was_read_under():
    h = _heldout({"S1": (1.5, 0.01, 1.5, 0.005, 4, True)})
    rec = _verdicts(h)["interval_convention"]
    assert rec["strict_clearance"] is True
    assert rec["coupling_domain"] == [0.0, 0.95]


# ------------------------------------------------------------------------------------------------
# The difference interval, and the multiplier that cannot narrow it


def test_combining_the_two_uncertainties_can_only_widen_the_interval():
    obs_only = PRED.difference_interval(observed=1.5, observed_se=0.01, predicted=1.4,
                                        predicted_se=0.0, n_replicates=4)
    both = PRED.difference_interval(observed=1.5, observed_se=0.01, predicted=1.4,
                                    predicted_se=0.02, n_replicates=4)
    assert both["half_width"] > obs_only["half_width"]
    assert both["multiplier"] == pytest.approx(p5.t95(3))


def test_a_declared_correlation_enters_the_variance_with_its_sign():
    """Zero is the default and the reason is that the two sides share no reading. It is a field so
    that a domain that knows better can say so, and the arithmetic handles the sign rather than
    assuming the answer."""
    up = PRED.IntervalConvention(prediction_observation_correlation=0.8)
    down = PRED.IntervalConvention(prediction_observation_correlation=-0.8)
    base = dict(observed=1.5, observed_se=0.02, predicted=1.4, predicted_se=0.02, n_replicates=4)
    assert PRED.difference_interval(convention=up, **base)["se"] < \
           PRED.difference_interval(**base)["se"] < \
           PRED.difference_interval(convention=down, **base)["se"]


def test_a_missing_observed_spread_refuses_the_comparison_rather_than_assuming_it():
    d = PRED.difference_interval(observed=1.5, observed_se=float("nan"), predicted=1.4,
                                 predicted_se=0.01, n_replicates=1)
    assert not np.isfinite(d["half_width"])
    assert PRED.REGISTERED.clearance(d["difference"], d["half_width"], 0.10)["verdict"] == "UNRESOLVED"


# ------------------------------------------------------------------------------------------------
# The propagation consumes nothing from the run's own stream, and reproduces from the bundle


def test_the_propagation_is_reproducible_from_its_own_seed():
    a = _propagate(beta_se=0.02, score_sds=[2.0, 2.0, 2.0, 2.0], seed=5)
    b = _propagate(beta_se=0.02, score_sds=[2.0, 2.0, 2.0, 2.0], seed=5)
    c = _propagate(beta_se=0.02, score_sds=[2.0, 2.0, 2.0, 2.0], seed=6)
    assert a["se"] == b["se"]
    assert a["se"] != c["se"]


def test_a_saved_bundle_recomputes_the_same_panel(tmp_path):
    from arc_runner import adapters, custody, ladder as L
    ad = adapters.MockCouplingAdapter(beta=0.5, theta=0.0, noise=0.03)
    lad = L.MockLadder(n_items=20000, scale=400.0)
    place = lambda s: {"kind": "mock", "capability": float(s), "rounds": 0}
    start = lambda n: {"kind": "mock", "capability": 20.0, "rounds": 0, "system": n}
    cfg = p5.P5Config(reps=4, window_end=32, checkpoints=(4, 8, 16, 32), replicates=3)
    res = p5.run_p5(ad, lad, cfg, 3, place, start, ["S1", "S2"], bundle=str(tmp_path / "run"))
    again = custody.recompute_verdicts(custody.load_bundle(str(tmp_path / "run")))
    assert again["panel"]["assigned"] == 2
    assert again["panel"] == res["verdicts"]["panel"]
    assert again["PREDICTION"] == res["verdicts"]["PREDICTION"]


# ------------------------------------------------------------------------------------------------
# The panel is the TABLE as well as the count: it may not shrink, and it may not grow


def test_an_assigned_system_that_never_reported_has_a_row_saying_so():
    """The aggregate counted such a system in the denominator, which is the arithmetic. The table is
    where a reader sees a system's status, and a system missing from the table has been dropped
    wherever anybody actually looks."""
    h = _heldout({"S1": (1.5, 0.01, 1.50, 0.005, 4, True), "S2": (1.5, 0.01, 1.50, 0.005, 4, True)})
    man = _manifest(["S1", "S2", "S3"], predictions=h["sealed_predictions"])
    v = p5.verdicts(man, {"identification": "NOT ESTABLISHED", "premise": "NOT REFUTED"}, h,
                    p5.P5Config())
    assert set(v["per_system"]) == {"S1", "S2", "S3"}
    rec = v["per_system"]["S3"]
    assert rec["verdict"] == "NOT EVALUABLE"
    assert rec["counted_in_denominator"] is True and rec["assigned"] is True
    assert "produced no fitted exponent" in rec["reason"]
    assert v["panel"]["assigned"] == 3 and v["panel"]["not_evaluable"] == 1
    assert v["panel"]["never_reported"] == 1
    # and the counts still add up to the panel, from the table alone
    assert len([r for r in v["per_system"].values() if r["counted_in_denominator"]]) == 3


def test_a_system_scored_without_being_assigned_is_reported_and_never_counted():
    """The finding's title is that the final comparison can CHANGE the population, and a system added
    after the seal changes it upwards. It keeps its reading and loses its weight."""
    h = _heldout({"S1": (1.5, 0.01, 1.50, 0.005, 4, True), "EXTRA": (1.5, 0.01, 1.50, 0.005, 4, True)})
    man = _manifest(["S1"], predictions=h["sealed_predictions"])
    v = p5.verdicts(man, {"identification": "NOT ESTABLISHED", "premise": "NOT REFUTED"}, h,
                    p5.P5Config())
    extra = v["per_system"]["EXTRA"]
    assert extra["counted_in_denominator"] is False and extra["assigned"] is False
    assert "not on the sealed panel" in extra["reason"]
    assert v["panel"]["assigned"] == 1 and v["panel"]["agrees"] == 1
    assert v["panel"]["unassigned_systems_present"] == ["EXTRA"]
    assert "never counted" in v["panel"]["unassigned_note"]


# ------------------------------------------------------------------------------------------------
# The propagation reports its own simulation uncertainty


def test_the_propagated_width_reports_the_monte_carlo_error_of_its_own_estimate():
    """A half width quoted from a finite number of draws has a standard error of its own, and a
    reader comparing that width with the margin is entitled to know how much of the gap between them
    is simulation noise. Nothing decides on it; it is reported."""
    few = _propagate(beta_se=0.02, score_sds=[2.0, 2.0, 2.0, 2.0],
                     convention=PRED.IntervalConvention(prediction_draws=50))
    many = _propagate(beta_se=0.02, score_sds=[2.0, 2.0, 2.0, 2.0],
                      convention=PRED.IntervalConvention(prediction_draws=1600))
    for out in (few, many):
        assert out["se_monte_carlo_error"] > 0
        assert out["half_width_monte_carlo_error"] == pytest.approx(
            PRED.REGISTERED.equivalence_z * out["se_monte_carlo_error"])
        # the simulation error is a small fraction of the width it qualifies, not a rival to it
        assert out["half_width_monte_carlo_error"] < out["half_width"]
    # and it falls with the draw count, as the square root of it
    assert few["se_monte_carlo_error"] / few["se"] > many["se_monte_carlo_error"] / many["se"]


def test_the_gap_between_the_point_prediction_and_the_mean_of_the_draws_is_reported():
    """The difference is read against the point prediction while the draws average to something else
    wherever the solution is curved in the coupling. The gap is reported and never corrected:
    correcting the centre would move the prediction after it was sealed."""
    out = _propagate(beta_hat=0.80, beta_se=0.05, score_sds=[1.0, 1.0, 1.0, 1.0])
    assert "mean_of_draws_minus_point" in out
    assert out["mean_of_draws_minus_point"] == pytest.approx(
        out["mean_of_draws"] - out["predicted_exponent"], abs=1e-12)
    # THE COMPARISON HOLDS EVERYTHING BUT THE CURVATURE FIXED. An earlier version of this case
    # compared a coupling of 0.80 with a standard error of 0.05 against a coupling of 0.20 with a
    # standard error of 0.005, and the second gap is smaller for the width of the interval alone:
    # the assertion would have passed on a flat solution. Here only the coupling moves, so the gap
    # that shrinks is the curvature of 1 / (1 - beta), which is what the field is reporting.
    flat = _propagate(beta_hat=0.20, beta_se=0.05, score_sds=[1.0, 1.0, 1.0, 1.0])
    assert flat["beta_se"] == out["beta_se"]
    # where the solution is nearly linear over the interval the two centres agree
    assert abs(flat["mean_of_draws_minus_point"]) < abs(out["mean_of_draws_minus_point"])


# ------------------------------------------------------------------------------------------------
# A deciding run may not soften the convention on the way past


def _convention_refusals(cfg):
    from arc_runner import mode as MODE
    return [r for r in MODE.missing_confirmatory_inputs(MODE.ConfirmatoryInputs(config=cfg))
            if r.startswith("interval-convention")]


def test_a_deciding_run_refuses_every_weakening_of_the_registered_convention():
    """Each of these makes agreement easier to reach than the registered convention makes it, and
    each is named separately rather than the first one found."""
    assert any("strict clearance is off" in r
               for r in _convention_refusals(p5.P5Config(strict_clearance=False)))
    assert any("below the registered" in r
               for r in _convention_refusals(p5.P5Config(equivalence_z=1.0)))
    assert any("wider than the registered" in r
               for r in _convention_refusals(p5.P5Config(coupling_domain=(0.0, 0.999))))
    assert any("excursion tolerance" in r
               for r in _convention_refusals(p5.P5Config(domain_excursion_tolerance=0.2)))
    assert any("cannot produce a prediction interval" in r
               for r in _convention_refusals(p5.P5Config(prediction_draws=1)))
    # AND EVERY DRAW COUNT BELOW THE REGISTERED ONE, not only a count too small to form an interval
    # at all. A gate written against well-formedness refuses one draw and lets fifty through, and
    # fifty is the weakening: see the measured test below for what a below-registered count reports.
    for n in (2, 3, 50, 400, int(PRED.REGISTERED.prediction_draws) - 1):
        refusals = _convention_refusals(p5.P5Config(prediction_draws=n))
        assert any("below the registered" in r for r in refusals), (n, refusals)
    assert any("subtracts from the variance" in r
               for r in _convention_refusals(p5.P5Config(prediction_observation_correlation=0.5)))
    # and a setup that softens two things at once is told about both
    assert len(_convention_refusals(p5.P5Config(strict_clearance=False, equivalence_z=1.0))) == 2


def test_a_stricter_convention_passes_and_a_configuration_without_the_fields_is_untouched():
    """A gate that refuses every setup is an outage. A run may hold itself to more than the
    registered convention, and a configuration that carries none of these fields is under the
    registered convention already, which is the value being checked for."""
    assert _convention_refusals(p5.P5Config()) == []
    assert _convention_refusals(p5.P5Config(equivalence_z=2.5, coupling_domain=(0.0, 0.9),
                                            prediction_observation_correlation=-0.3)) == []
    # the registered draw count passes, and so does a larger one: more draws can only sharpen the
    # estimate of a width, and a gate that refused them would be refusing a stricter run
    assert _convention_refusals(
        p5.P5Config(prediction_draws=int(PRED.REGISTERED.prediction_draws))) == []
    assert _convention_refusals(
        p5.P5Config(prediction_draws=int(PRED.REGISTERED.prediction_draws) * 5)) == []

    class Old:                                     # a configuration written before these fields, and
        margin = 0.10                              # a P16 configuration, which has none of them
    assert _convention_refusals(Old()) == []
    from arc_runner import p16
    assert _convention_refusals(p16.P16Config()) == []


def test_a_draw_count_below_the_registered_one_reports_a_narrower_width_and_can_decide():
    """WHY the draw count belongs in the gate, measured rather than asserted.

    The half width is the resampled standard deviation times the multiplier. A standard deviation
    estimated from a handful of draws is biased low and scattered wide, so a run that lowers the
    count reports a width that is narrower on average and, on the seeds where it is much narrower,
    clears a margin the registered count does not clear. Twelve seeds of the same propagation, the
    registered eight hundred draws against two: the mean half width falls from about 0.091 to about
    0.054 and is narrower on eleven of the twelve. The decision consequence is shown on one of them
    rather than described, and the margin is taken from the two widths themselves so that the case
    is about their ORDER and not about a number written down here.
    """
    def half(draws, seed):
        return _propagate(beta_hat=0.5, beta_se=0.03, score_sds=[1.0, 1.0, 1.0, 1.0],
                          window_end=128, checkpoints=(4, 8, 16, 32, 64, 128), seed=seed,
                          convention=PRED.IntervalConvention(prediction_draws=draws))["half_width"]
    registered = [half(int(PRED.REGISTERED.prediction_draws), s) for s in range(12)]
    reduced = [half(2, s) for s in range(12)]
    assert np.mean(reduced) < 0.75 * np.mean(registered)
    assert sum(r < g for r, g in zip(reduced, registered)) >= 9
    # and on one seed the two widths fall either side of a margin between them, so the reduced count
    # reads agreement where the registered count reads nothing
    r_few, r_reg = reduced[6], registered[6]
    assert r_few < r_reg
    margin = 0.5 * (r_few + r_reg)
    assert PRED.REGISTERED.clearance(0.0, r_few, margin)["verdict"] == "AGREES"
    assert PRED.REGISTERED.clearance(0.0, r_reg, margin)["verdict"] == "UNRESOLVED"


# ------------------------------------------------------------------------------------------------
# The denominator itself has to have been committed to


def _manifest_without_a_panel(seed=1, predictions=None):
    """A manifest that names no assigned panel, being a bundle sealed before the panel was written
    into the configuration. `run_p5` never produces one: it writes the panel to both surfaces."""
    man = M.new_manifest("P5", False, "ladder-sha", {"seed": seed}, "test-adapter")
    M.seal_predictions(man, predictions if predictions is not None else {"sealed": True}, "the test")
    return man


def test_a_denominator_inferred_from_the_survivors_does_not_support_the_proposition():
    """Counting every system on a list that was itself read off the systems which produced a fit is
    the survivors' denominator under another name: the arithmetic is right and the population is
    still the one the run chose after seeing the data. The counts are reported and the
    proposition-level word is withheld, because complete support must depend on a frozen
    denominator."""
    h = _heldout({"S1": (1.5, 0.01, 1.50, 0.005, 4, True), "S2": (1.5, 0.01, 1.50, 0.005, 4, True)})
    h.pop("assigned_systems")
    v = p5.verdicts(_manifest_without_a_panel(predictions=h["sealed_predictions"]),
                    {"identification": "NOT ESTABLISHED", "premise": "NOT REFUTED"}, h, p5.P5Config())
    panel = v["panel"]
    assert panel["assigned"] == 2 and panel["agrees"] == 2      # the reading is kept
    assert panel["denominator_frozen"] is False
    assert "INFERRED AFTER THE FACT" in panel["denominator_source"]
    assert "SUPPORTED" in panel["denominator_note"]             # what it would have read, said plainly
    assert v["PREDICTION"] == "INCONCLUSIVE", panel


def test_a_panel_named_by_the_run_is_frozen_and_still_decides():
    """A gate that refused every panel it did not find in the sealed configuration would be an
    outage: the run's own held-out record names the panel it was given, and that is a commitment made
    before the systems were scored."""
    h = _heldout({"S1": (1.5, 0.01, 1.50, 0.005, 4, True), "S2": (1.5, 0.01, 1.50, 0.005, 4, True)})
    v = p5.verdicts(_manifest_without_a_panel(predictions=h["sealed_predictions"]),
                    {"identification": "NOT ESTABLISHED", "premise": "NOT REFUTED"}, h, p5.P5Config())
    assert v["panel"]["denominator_frozen"] is True
    assert "held-out record" in v["panel"]["denominator_source"]
    assert v["PREDICTION"] == "SUPPORTED", v["panel"]
    # and the sealed configuration is preferred to it, with the provenance saying so
    sealed = p5.verdicts(_manifest(["S1", "S2"], predictions=h["sealed_predictions"]),
                         {"identification": "NOT ESTABLISHED",
                          "premise": "NOT REFUTED"}, h, p5.P5Config())
    assert "sealed configuration" in sealed["panel"]["denominator_source"]
    assert sealed["panel"]["denominator_frozen"] is True


def test_an_inferred_denominator_withholds_refutation_as_well_as_support():
    """The rule is about the population and not about which way the panel fell, so a majority
    disagreeing on an inferred denominator is withheld in the same words."""
    h = _heldout({"S1": (1.5, 0.001, 1.90, 0.001, 4, True), "S2": (1.5, 0.001, 1.90, 0.001, 4, True)})
    h.pop("assigned_systems")
    v = p5.verdicts(_manifest_without_a_panel(predictions=h["sealed_predictions"]),
                    {"identification": "NOT ESTABLISHED", "premise": "NOT REFUTED"}, h, p5.P5Config())
    assert v["panel"]["disagrees"] == 2
    assert "REFUTED" in v["panel"]["denominator_note"]
    assert v["PREDICTION"] == "INCONCLUSIVE"


def test_the_registered_multiplier_is_declared_once_and_the_identification_helper_reads_it():
    """The finding says the ninety per cent level cannot be inferred from a convenient helper. The
    multiplier is declared in `p5_prediction`, beside the convention that names the level it belongs
    to, and `p5_identification.TOST_Z` is an alias of it rather than a second declaration: a number
    written down twice is a number that can drift."""
    from arc_runner import p5_identification as PI
    assert PRED.TOST_Z == 1.645
    assert PRED.REGISTERED.equivalence_z == PRED.TOST_Z
    assert PI.TOST_Z == PRED.REGISTERED.equivalence_z
    assert p5.P5Config().equivalence_z == PRED.REGISTERED.equivalence_z
    assert p5.P5Config().route_equivalence_z == PRED.REGISTERED.equivalence_z
