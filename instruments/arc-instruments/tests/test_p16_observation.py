"""Finding A1: what P16 observes is typed at the callback boundary, and the type decides the event.

THE DEFECT THESE CASES PIN. The reference runner took a callback returning one float per round,
called it a correction margin, regressed it against ordinary round indices and declared a reversal
when the fitted slope was negative. Nothing said which quantity the float was. The registration's
Delta is already the logarithmic slope of the service to burden ratio, so on a series of Delta values
that rule tests the wrong thing in both directions: a constant negative Delta, which IS the event,
has no slope and never fires, and a falling but still positive Delta, which is not the event at all,
fires every time. Both halves are exercised below on the same numbers, read once undeclared and once
declared, so that what changes between them is the type and nothing else.

ONE ENTRY POINT, ONE INPUT NAME. Every case goes through `_arm`, which is `p16.run_arm`, with the
source passed as `source`. The finding requires that each case receive the contract's event result
and numerical line without the input being renamed between them: the declaration changes, the call
does not.
"""
import math

import numpy as np
import pytest

from arc_instruments import balance as BAL
from arc_runner import code_domain as CD, custody, observation as OBS, p16


CFG = p16.P16Config(horizon=60)


def _arm(source, cfg=CFG, arm="dose+0.6", alpha=2.6, seed=0):
    """The one entry point every acceptance case uses."""
    return p16.run_arm(source, arm, alpha, cfg, np.random.default_rng(seed))


def _declared(values, spec):
    """The series behind a declaration."""
    def source(arm, alpha_arm, r, rng):
        return values(arm, alpha_arm, r, rng)
    return OBS.declare(source, spec)


def _undeclared(values):
    """The same series behind no declaration at all, which is the reference runner's boundary."""
    def source(arm, alpha_arm, r, rng):
        return values(arm, alpha_arm, r, rng)
    return source


def constant_negative_delta(arm, alpha_arm, r, rng):
    return -0.20


def falling_positive_delta(arm, alpha_arm, r, rng):
    # 0.90 down to 0.43: falling, positive throughout. The small deterministic wobble is there so
    # that the residuals of a fitted line are not identically zero, which keeps the case below about
    # ONE repair: the reference loop also required a strictly positive standard error before it would
    # declare, so a noiseless declining series would have failed to fire for that second reason and
    # the case would pass without the coordinate repair it is named for. The variance half is pinned
    # on its own, in test_a_window_with_no_residual_at_all_still_reaches_the_alarm_branch below.
    return 0.90 - 0.008 * r + 0.002 * math.sin(r)


DELTA = OBS.balance_elasticity_observation(source="acceptance-case")


# --------------------------------------------------------------------------------------------------
# Acceptance case 1: constant negative Delta
# --------------------------------------------------------------------------------------------------

def test_a_constant_negative_delta_is_the_event_and_the_undeclared_reading_misses_it():
    got = _arm(_declared(constant_negative_delta, DELTA))
    assert got["declared_round"] is not None, "Delta below zero throughout is the registered event"
    assert got["event"] == "Delta crosses below zero"
    assert got["arm_delta"] == pytest.approx(-0.20)
    assert got["observation"]["estimator"] == OBS.WINDOW_MEAN
    # the same numbers with no declaration: the reference rule regresses them, finds no slope, and
    # reports no event on a series that never leaves the negative half of the line
    reference = _arm(_undeclared(constant_negative_delta))
    assert reference["declared_round"] is None
    assert reference["observation"]["quantity"] == OBS.SURPLUS_RATE


# --------------------------------------------------------------------------------------------------
# Acceptance case 2: decreasing but positive Delta throughout the horizon
# --------------------------------------------------------------------------------------------------

def test_a_falling_but_positive_delta_never_fires_and_the_undeclared_reading_fires_every_time():
    got = _arm(_declared(falling_positive_delta, DELTA))
    assert got["declared_round"] is None, "a falling positive Delta is not a negative Delta"
    assert got["arm_delta"] > 0
    reference = _arm(_undeclared(falling_positive_delta))
    assert reference["declared_round"] is not None, \
        "the defect this case pins: the untyped rule declares the event on a series that never had it"


def test_a_window_with_no_residual_at_all_still_reaches_the_alarm_branch():
    """The smaller half of the finding, pinned on its own rather than only inside another case.

    The reference loop required a STRICTLY POSITIVE standard error before it would declare, so a
    window with no residual at all never reached its alarm branch: a perfectly measured window was
    treated as an unmeasured one. A zero-residual window is a perfectly measured one and its interval
    is a point, which is either below zero or it is not.

    The case is built on the level estimator over an exactly representable constant, because that is
    where an exactly zero standard error actually arises: a line fitted through floating-point data
    leaves residuals of order 1e-16, which the old guard would have accepted, and even the mean of
    forty-six copies of -0.20 carries a deviation of 9e-17. A quarter is a power of two, so its mean
    and its deviations are exact. And the rule is not "declare whenever the interval is a point": the
    same constant with the opposite sign is a point above zero and declares nothing.
    """
    below = OBS.readings_from_values([-0.25] * 60)
    est = OBS.estimate([rd for rd in below if rd.round_index >= 14], DELTA)
    assert est.se == 0.0, "the deviations of a constant series are exactly zero"
    det = OBS.detect_event(below, DELTA, 14, CFG.z_threshold)
    assert det["declared_round"] is not None, \
        "the defect this case pins: a strictly positive standard error was required to declare"
    above = OBS.readings_from_values([0.25] * 60)
    assert OBS.estimate([rd for rd in above if rd.round_index >= 14], DELTA).se == 0.0
    assert OBS.detect_event(above, DELTA, 14, CFG.z_threshold)["declared_round"] is None, \
        "a point interval above zero is not the event either"


# --------------------------------------------------------------------------------------------------
# Acceptance case 3: a known log-ratio power trajectory recovers the registered line
# --------------------------------------------------------------------------------------------------

def test_a_known_log_ratio_power_trajectory_recovers_the_registered_line():
    """U = I R^alpha, W = b dU/dR, Q = q U^chi, so the slope of log(Q/W) on log R is Delta exactly.

    Each arm's estimate must be 1 - alpha (1 - chi), the line fitted across arms must have slope
    minus (1 minus chi) and its zero at 1 / (1 - chi), and those are the two numbers in the seal.
    """
    cfg = p16.P16Config(systems_per_arm=3, horizon=96)
    source = p16.mock_balance_source(cfg, true_alpha_crit=2.0)
    chi = OBS.spec_of(source).chi_hat
    assert chi == pytest.approx(0.5)

    got = _arm(source, cfg=cfg, arm="dose+0.6", alpha=2.6)
    assert got["arm_delta"] == pytest.approx(BAL.balance_elasticity(2.6, chi), abs=0.05)
    assert got["declared_round"] is not None
    assert _arm(source, cfg=cfg, arm="dose-0.6", alpha=1.4)["declared_round"] is None

    res = p16.run_p16(source, cfg, 5, "none", "mock")
    v = res["verdicts"]
    assert v["registered_coordinate"] is True
    assert v["observed_quantity"] == OBS.LOG_SERVICE_RATIO
    assert v["line_slope_fitted"] == pytest.approx(-(1.0 - chi), abs=0.02)
    assert v["line_zero_fitted"] == pytest.approx(BAL.trend_crossover(chi), abs=0.05)
    assert v["line_comparison"]["comparable"] is True
    assert abs(v["line_comparison"]["difference"]) < 0.02
    # the run's reading of its own arms. Finding A2 moved the `P16` key to the component wrapper, so
    # the arm pattern is read under its own name here; the wrapper's own result on this run is
    # exercised in tests/test_p16_components.py, where the registered bands are supplied.
    assert v["run_pattern"] == "SUPPORTED", v
    assert v["P16"] != "SUPPORTED", "a single run without a fresh-data repetition is never support"


def test_a_world_whose_boundary_sits_away_from_the_seal_is_refuted_on_location():
    """The same source with a different chi puts the crossover at 2.3 while the seal says 2.0. The
    sign pattern across arms cannot see that; the line's zero can, which is why it is sealed."""
    cfg = p16.P16Config(systems_per_arm=3, horizon=96)
    res = p16.run_p16(p16.mock_balance_source(cfg, true_alpha_crit=2.3), cfg, 7, "none", "mock")
    v = res["verdicts"]
    assert v["line_zero_fitted"] == pytest.approx(2.3, abs=0.1)
    assert v["run_pattern"] == "REFUTED (boundary mislocated)", v


# --------------------------------------------------------------------------------------------------
# Acceptance case 4: a shifted trajectory needing the existing curvature correction
# --------------------------------------------------------------------------------------------------

def shifted_trajectory(chi=0.5, beta_L=0.9, a=0.03, U0=1.0):
    """dU/dR = a U^beta_L with W = b dU/dR and Q = q U^chi, so log(Q/W) moves as (chi - beta_L) log U
    and the balance is alpha_R (chi - beta_L). The constant-exponent formula, handed the same
    finite-window alpha_R, reports 1 - alpha_R (1 - chi), which here has the opposite sign."""
    def values(arm, alpha_arm, r, rng):
        R = float(r + 1)
        U = float(BAL.shifted_local_terms([R], U0, a, beta_L)["U"][0])
        return {"round": r, "Q": U ** chi, "W": a * U ** beta_L, "R": R, "U": U}
    return values


def test_a_shifted_trajectory_agrees_with_the_curvature_correction_and_not_with_the_naive_formula():
    chi, beta_L = 0.5, 0.9
    spec = OBS.log_service_ratio_observation(supplies_q_and_w=True, chi_hat=chi,
                                             curvature_correction=True, source="acceptance-case")
    got = _arm(_declared(shifted_trajectory(chi=chi, beta_L=beta_L), spec))
    check = got["arm_estimate"]["model_check"]
    measured = got["arm_delta"]
    assert check is not None, "a declared curvature correction must report both readings"
    # the truth on this trajectory: alpha_R (chi - beta_L), negative, and the event fires
    assert measured == pytest.approx(check["alpha_R_mean"] * (chi - beta_L), abs=0.05)
    assert measured == pytest.approx(check["delta_curvature_corrected"], abs=0.05)
    assert got["declared_round"] is not None
    # and the constant-exponent reading of the same path reports a surplus that is not there
    assert check["delta_constant_exponent"] > 0 > measured, check


# --------------------------------------------------------------------------------------------------
# The declaration itself: quantity, units, clock, estimator, smoothing window
# --------------------------------------------------------------------------------------------------

def test_the_clock_and_the_estimator_belong_to_the_quantity_and_are_not_chosen_per_run():
    with pytest.raises(OBS.ObservationRefusal):
        OBS.ObservationSpec(quantity=OBS.LOG_SERVICE_RATIO, clock=OBS.ROUND_INDEX,
                            estimator=OBS.SLOPE_ON_LOG_R, units="")
    with pytest.raises(OBS.ObservationRefusal):
        OBS.ObservationSpec(quantity=OBS.BALANCE_ELASTICITY, clock=OBS.ROUND_INDEX,
                            estimator=OBS.SLOPE_ON_ROUND, units="")
    with pytest.raises(OBS.ObservationRefusal):
        OBS.ObservationSpec(quantity="something-else", clock=OBS.ROUND_INDEX,
                            estimator=OBS.SLOPE_ON_ROUND, units="")


def test_an_unregistered_smoothing_window_is_refused_rather_than_given_an_invented_variance():
    with pytest.raises(OBS.ObservationRefusal) as exc:
        OBS.log_service_ratio_observation(smoothing_window=5)
    assert "variance rule" in str(exc.value)


def test_a_specification_survives_the_round_trip_a_bundle_puts_it_through():
    spec = OBS.log_service_ratio_observation(supplies_q_and_w=True, chi_hat=0.5, source="round-trip")
    back = OBS.ObservationSpec.from_record(spec.as_record())
    assert back == spec
    assert OBS.ObservationSpec.from_record(None) is OBS.UNDECLARED, "silence stays undeclared"


# --------------------------------------------------------------------------------------------------
# What the deciding path requires of a source
# --------------------------------------------------------------------------------------------------

def test_the_deciding_path_refuses_an_undeclared_source():
    with pytest.raises(OBS.ObservationRefusal) as exc:
        OBS.require_assay(OBS.spec_of(_undeclared(constant_negative_delta)))
    assert "declares no observed quantity" in str(exc.value)


def test_the_deciding_path_refuses_a_source_that_returns_the_assumed_balance():
    with pytest.raises(OBS.ObservationRefusal) as exc:
        OBS.require_assay(DELTA)
    assert "returns the balance elasticity itself" in str(exc.value)
    assert "supply the correction service Q and the offered burden W separately" in str(exc.value)


def test_the_deciding_path_refuses_the_unregistered_surplus_rate_and_a_source_with_no_q_and_w():
    with pytest.raises(OBS.ObservationRefusal):
        OBS.require_assay(OBS.surplus_rate_observation(supplies_q_and_w=True, source="declared"))
    with pytest.raises(OBS.ObservationRefusal) as exc:
        OBS.require_assay(OBS.log_service_ratio_observation(source="declared"))
    assert "neither Q and W nor any approved sufficient observations" in str(exc.value)
    # and the one that passes: Q and W separately, in the registered coordinate
    OBS.require_assay(OBS.log_service_ratio_observation(supplies_q_and_w=True, source="declared"))


def test_a_confirmatory_run_refuses_an_undeclared_source_before_the_first_paid_call(tmp_path):
    """Selecting the deciding path with a source that has declared nothing must fail before any call,
    ON THE ASSAY, and this case is about which refusal speaks.

    It used to accept either refusal, and that made it pass on the other one: `mode._CONFIG_RULES`
    asks a configuration for bank states, retention fractions, replicates and a positive margin,
    which no P16Config carries, so behind the mode gate the assay was unreachable and this case was
    green on a bank-shaped complaint about a P16 run. The assay now runs first, because a source that
    has said nothing about what it measures is not rescued by a complete apparatus, and the case
    names the exception it expects. The companion case below keeps the ordering honest in the other
    direction: a declared source still meets the mode gate.
    """
    from arc_runner import budget as B, mode as MODE

    calls = []

    def must_not_be_called(arm, alpha_arm, r, rng):
        calls.append((arm, r))
        raise AssertionError("a provider call was made after a confirmatory run should have refused")

    cfg = p16.P16Config(systems_per_arm=1, horizon=20)
    store = CD.CheckpointStore(str(tmp_path))
    store.save("seed", CD.new_artefact(_library({0, 1})))
    lad = CD.SuiteLadder(_pool(), batch_runner=CD.inprocess_batch_runner())
    inputs = MODE.ConfirmatoryInputs(ladder=lad, checkpoint_store=store,
                                     allowance=B.Allowance(limit_gbp=5.0), anchor=_anchor,
                                     config_resolution={"resolved_by": "the operator",
                                                        "resolved_utc": "2026-09-05T00:00:00Z"})
    with pytest.raises(OBS.ObservationRefusal) as exc:
        p16.run_p16(must_not_be_called, cfg, 5, lad.sha256, "adapter", mode="confirmatory",
                    confirmatory_inputs=inputs)
    assert "declares no observed quantity" in str(exc.value)
    assert not calls, "the source was read after the deciding path should have refused"


def test_the_assay_gate_does_not_swallow_the_mode_gate(tmp_path):
    """A declared, real-shaped source with an incomplete apparatus is still refused by the mode gate.

    The assay speaking first would be worth nothing if it spoke for every refusal: this is the case
    that shows the two gates still ask different questions and that the second one is still reached.
    """
    from arc_runner import mode as MODE

    def declared(arm, alpha_arm, r, rng):
        raise AssertionError("a provider call was made after a confirmatory run should have refused")

    OBS.declare(declared, OBS.log_service_ratio_observation(
        supplies_q_and_w=True, chi_hat=0.5, source="a-declared-source"))
    cfg = p16.P16Config(systems_per_arm=1, horizon=20)
    with pytest.raises(MODE.ModeRefusal) as exc:
        p16.run_p16(declared, cfg, 5, "sha", "adapter", mode="confirmatory")
    assert any(r.startswith("checkpoint-store") for r in exc.value.requirements), exc.value.requirements


def _anchor(digest):
    return custody.receipt("stub-anchor:%s" % digest[:12], digest, service="test-stub")


_anchor.anchor_service = "test-stub"


# --------------------------------------------------------------------------------------------------
# A declaration is not a measurement: what arrives is checked against what was declared
# --------------------------------------------------------------------------------------------------

def test_a_source_that_declares_q_and_w_and_hands_back_a_bare_float_is_refused_at_the_boundary():
    """The gap the pre-run gate cannot close, closed at the round the reading arrives.

    `require_assay` reads the declaration and nothing else, because it runs before any round exists.
    A source could therefore declare that it supplies Q and W, hand back one float per round, clear
    that gate, and be estimated as the registered quantity: the reading carried no recursive
    coordinate either, so the round number stood in for log R and a falling positive series was
    reported as a negative balance elasticity IN THE REGISTERED COORDINATE, which is strictly worse
    than the undeclared reading this file's second case pins.
    """
    spec = OBS.log_service_ratio_observation(supplies_q_and_w=True, chi_hat=0.5, source="relabelled")
    OBS.require_assay(spec)                    # the declaration alone clears the pre-run gate
    with pytest.raises(OBS.ObservationRefusal) as exc:
        _arm(_declared(falling_positive_delta, spec))
    assert "and supplied neither" in str(exc.value)


def test_a_source_that_supplies_q_and_w_but_no_recursive_coordinate_is_refused_at_the_boundary():
    """R belongs to the artefact's depth and no other number substitutes for it, least of all the
    round: an arm spending more revision passes per round travels further in R than rounds it took."""
    spec = OBS.log_service_ratio_observation(supplies_q_and_w=True, chi_hat=0.5, source="no-depth")

    def values(arm, alpha_arm, r, rng):
        return {"round": r, "Q": 2.0, "W": 1.0}

    with pytest.raises(OBS.ObservationRefusal) as exc:
        _arm(_declared(values, spec))
    assert "carried no recursive coordinate R" in str(exc.value)
    assert "The round number is not R" in str(exc.value)


def test_a_bare_series_on_the_log_clock_is_excluded_and_never_regressed_on_the_round_number():
    """The same law where no source is in the picture: a saved or legacy series of bare numbers.

    The estimator defaulted a missing R to round + 1, and that default fired ONLY for a series
    declared on the registered log R clock without one, which is the single case the module forbids.
    Relabelling the reference mock's per-round surplus rate as the registered quantity then regressed
    it on log(round + 1) and printed a fitted -0.457 beside the sealed -0.5, manufacturing an
    agreement to within 0.043 out of two quantities in different coordinates. Nothing is fitted now.
    """
    values = [0.90 - 0.008 * r for r in range(60)]
    spec = OBS.log_service_ratio_observation(supplies_q_and_w=True, chi_hat=0.5, source="relabelled")
    est = OBS.estimate(OBS.readings_from_values(values), spec)
    assert est.n_used == 0 and est.n_excluded == 60
    assert "the round number is not R" in est.reason, est.reason
    assert not np.isfinite(est.delta)
    det = p16.detect_reversal(values, CFG, spec)
    assert det["declared_round"] is None
    # and the line comparison says the coordinate is right and the measurement is absent, which are
    # two different statements and were one before
    cmp_ = OBS.line_comparison(spec, det["slope"], -0.5)
    assert cmp_["comparable"] is True and cmp_["comparison_made"] is False
    assert cmp_["difference"] is None


def test_a_saved_run_whose_rounds_do_not_carry_the_sealed_quantity_is_refused_at_proposition_level(tmp_path):
    """The same comparison at the moment `read` is not in the picture.

    A bundle is re-scored from saved rounds and calls no source, so the per-round refusal never fires
    there, and a bundle written before that refusal existed may hold exactly the rounds it would have
    refused. The arms' saved numbers are untouched here and still fit a line; what is missing is the
    coordinate the seal says they were read in, and that is enough to refuse the proposition.
    """
    cfg = p16.P16Config(systems_per_arm=2, horizon=60)
    p16.run_p16(p16.mock_balance_source(cfg, true_alpha_crit=2.0), cfg, 5, "none", "mock",
                bundle=str(tmp_path / "evidence"))
    bundle = custody.load_bundle(str(tmp_path / "evidence"))
    for a in bundle["arms"]:
        for rd in a["readings"]:
            rd["R"] = None
    again = p16.verdicts(bundle["manifest"], bundle["sealed_predictions"], bundle["arms"], cfg)
    assert again["observation_reconciliation"]["state"] == "refused"
    assert "recursive coordinate R" in again["observation_reconciliation"]["reason"]
    assert again["proposition_level"].startswith("REFUSED")


def test_an_unapproved_set_of_sufficient_observations_does_not_stand_in_for_q_and_w():
    """The escape hatch the registration allows is an APPROVED set. Nothing read the word `approved`,
    so a tuple of strings cleared the deciding path with no Q, no W and nobody's name against it."""
    with pytest.raises(OBS.ObservationRefusal) as exc:
        OBS.require_assay(OBS.log_service_ratio_observation(
            sufficient_observations=("anything",), source="declared"))
    assert "records no approval" in str(exc.value)
    OBS.require_assay(OBS.log_service_ratio_observation(
        sufficient_observations=("cleared_faults",), observations_approved_by="the operator",
        observations_approved_utc="2026-09-05T00:00:00Z", source="declared"))


def test_an_approved_observation_set_that_never_arrives_is_refused_at_the_round_it_is_absent():
    """A set that was approved and then did not arrive is not the set that was approved."""
    spec = OBS.log_service_ratio_observation(
        sufficient_observations=("cleared_faults",), observations_approved_by="the operator",
        observations_approved_utc="2026-09-05T00:00:00Z", chi_hat=0.5, source="declared")
    rng = np.random.default_rng(0)

    def absent(arm, alpha_arm, r, rng_):
        return {"round": r, "value": -0.2, "R": float(r + 1)}

    with pytest.raises(OBS.ObservationRefusal) as exc:
        OBS.read(OBS.declare(absent, spec), "dose+0.6", 2.6, 0, rng, spec)
    assert "carried none of" in str(exc.value)

    def present(arm, alpha_arm, r, rng_):
        return {"round": r, "value": -0.2, "R": float(r + 1), "extra": {"cleared_faults": 3.0}}

    rd = OBS.read(OBS.declare(present, spec), "dose+0.6", 2.6, 0, rng, spec)
    assert rd.extra["cleared_faults"] == 3.0


def test_a_source_that_declares_itself_a_simulation_cannot_clear_the_deciding_path():
    """Both demonstration sources are simulations, and one of them declares Q, W and the registered
    coordinate, so nothing else in the assay would have stopped it. `mode` already refuses the mock
    anchor by identity and a simulated ladder by its flag; the observation carried no such marker, so
    a wholly simulated margin satisfied the gate that decides whether a measurement is a measurement.
    """
    cfg = p16.P16Config(systems_per_arm=1, horizon=30)
    spec = OBS.spec_of(p16.mock_balance_source(cfg, true_alpha_crit=2.0))
    assert spec.supplies_q_and_w is True and spec.registered is True
    with pytest.raises(OBS.ObservationRefusal) as exc:
        OBS.require_assay(spec)
    assert "declares itself a simulation" in str(exc.value)
    with pytest.raises(OBS.ObservationRefusal):
        OBS.require_assay(OBS.spec_of(p16.mock_margin_source(cfg, true_alpha_crit=2.0)))


# --------------------------------------------------------------------------------------------------
# The two slopes, and the refusal to compare them across coordinates
# --------------------------------------------------------------------------------------------------

def test_the_sealed_slope_is_never_compared_with_a_per_round_surplus_rate():
    """The seal's slope is minus (1 minus chi), being d Delta / d alpha. The reference mock's trend is
    0.01 per unit dose per round, being a rate of change of a rate. The reference runner printed the
    two side by side; there is no conversion between them without Q, W and R, so none is offered."""
    cfg = p16.P16Config(systems_per_arm=2, horizon=60)
    res = p16.run_p16(p16.mock_margin_source(cfg, true_alpha_crit=2.0), cfg, 5, "none", "mock")
    v = res["verdicts"]
    assert v["observed_quantity"] == OBS.SURPLUS_RATE
    assert v["registered_coordinate"] is False
    assert v["line_comparison"]["comparable"] is False
    assert "different coordinates" in v["line_comparison"]["reason"]
    assert v["proposition_level"].startswith("REFUSED")
    assert v["line_slope_sealed"] == pytest.approx(-0.5)
    assert abs(v["line_slope_fitted"]) < 0.05, "the fitted trend is per round and is not that slope"


def test_the_declared_quantity_is_sealed_and_the_bundle_re_scores_in_the_same_coordinate(tmp_path):
    cfg = p16.P16Config(systems_per_arm=2, horizon=60)
    res = p16.run_p16(p16.mock_balance_source(cfg, true_alpha_crit=2.0), cfg, 5, "none", "mock",
                      bundle=str(tmp_path / "evidence"))
    assert res["sealed"]["observation"]["quantity"] == OBS.LOG_SERVICE_RATIO
    bundle = custody.load_bundle(str(tmp_path / "evidence"))
    again = custody.recompute_verdicts(bundle)
    assert again["P16"] == res["verdicts"]["P16"]
    assert again["observed_quantity"] == OBS.LOG_SERVICE_RATIO
    # Q and W reached the bundle separately, unreduced, so the ratio can be re-checked
    first = bundle["arms"][0]["readings"][0]
    assert first["Q"] is not None and first["W"] is not None and first["R"] is not None
    # and the re-scoring checks the sealed declaration against those saved rounds rather than
    # believing the label on them
    assert again["observation_reconciliation"]["state"] == "reconciled"
    assert "proposition_level" not in again


# --------------------------------------------------------------------------------------------------
# The real assay: the code domain supplies Q and W separately
# --------------------------------------------------------------------------------------------------

def _library(passing, n_total=6):
    lines = []
    for i in range(n_total):
        body = "return x + y" if i in passing else "return None"
        lines.append("def add_%d(x, y):\n    %s" % (i, body))
    return "\n".join(lines)


def _pool(n=6):
    tasks = []
    for i in range(n):
        name = "add_%d" % i
        checks = ("assert %s(3, 4) == 7" % name, "assert %s(0, 0) == 0" % name,
                  "assert %s(-1, 1) == 0" % name)
        tasks.append(CD.Task(id=name, statement="Define %s(x, y) returning the sum." % name,
                             signature="def %s(x, y): ..." % name, shown_examples=(checks[0],),
                             checks=checks))
    return CD.TaskPool(tasks, name="observation-test-pool")


class ScriptedSwapSystem:
    """A system that fixes one item and breaks another in the same round, so the round has both a
    correction service and an offered burden and their ratio exists."""
    name = "scripted-swap"

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def revise(self, artefact, retained, task, rng):
        passing = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        out = dict(artefact)
        out["text"] = _library(passing)
        out["rounds"] = int(artefact.get("rounds", 0)) + 1
        return out


def test_the_code_domain_source_supplies_q_and_w_separately_on_the_recursive_coordinate(tmp_path):
    pool = _pool()
    lad = CD.SuiteLadder(pool, batch_runner=CD.inprocess_batch_runner())
    store = CD.CheckpointStore(str(tmp_path))
    store.save("seed", CD.new_artefact(_library({0, 1, 2})))
    adapter = ScriptedSwapSystem([{0, 1, 2}, {0, 1, 3}, {0, 1, 3, 4}])
    source = CD.suite_margin_source(adapter, lad, store, "Improve the solutions library.",
                                    CD.DoseSchedule().dose_for, switch_round=8)

    spec = OBS.spec_of(source)
    assert spec.quantity == OBS.SERVICE_RATIO and spec.clock == OBS.LOG_RECURSIVE_COORDINATE
    assert spec.supplies_q_and_w is True
    OBS.require_assay(spec)                      # a real assay: Q and W separately, registered coordinate

    rng = np.random.default_rng(0)
    readings = [OBS.read(source, "baseline", 2.0, r, rng, spec) for r in range(3)]
    # the baseline arm spends one revision pass per round, so its depth and its round count agree and
    # this assertion cannot tell them apart. The case that can is the next one.
    assert [r.R for r in readings] == [1.0, 2.0, 3.0], "R is the accumulated revision depth"
    swap = readings[1]                            # one item fixed, one item broken
    assert swap.Q == 1.0 and swap.W == 1.0
    assert swap.value == pytest.approx(1.0), "the value is derived from Q and W and never trusted"
    assert swap.extra["margin"] == pytest.approx(readings[1].extra["margin"]), \
        "the four balance objects travel with the reading rather than being replaced by it"
    assert math.isclose(readings[1].extra["fixes"], 1.0)


def test_a_round_with_no_offered_burden_is_excluded_and_named_rather_than_made_to_disappear():
    """W = 0 is a regime the ratio does not describe; arc_instruments.balance.service_ratio refuses it
    outright. An arm that is mostly such rounds has not been measured, so they are counted AND NAMED:
    a count says how many rounds went, and only the reason says whether the arm met a regime the
    ratio does not describe or whether nothing was observed at all.

    THE ROUNDS TAKE THE PATH A REAL SOURCE TAKES. This case used to hand-build readings whose value
    was already None, which exercised the estimator's missing-value branch and never the derivation's:
    a real zero-burden round arrives with Q and W present and its value is derived here, so the
    exclusion has to come from the derivation and the reason has to survive it.
    """
    spec = OBS.service_ratio_observation(supplies_q_and_w=True, source="acceptance-case")

    def src(arm, alpha_arm, r, rng):
        return {"round": r, "Q": 1.0, "W": (0.0 if r % 2 else 1.0), "R": float(r + 1)}

    source = OBS.declare(src, spec)
    rng = np.random.default_rng(0)
    readings = [OBS.read(source, "baseline", 2.0, r, rng, spec) for r in range(20)]
    assert all(rd.W is not None for rd in readings), "the zero burden is recorded and not dropped"
    assert math.isnan(readings[1].value), "the ratio does not exist at W = 0 and is not invented"
    est = OBS.estimate(readings, spec)
    assert est.n_used == 10 and est.n_excluded == 10
    assert "no offered burden (W = 0)" in est.reason, est.reason


def test_the_recursive_coordinate_is_the_revision_depth_and_parts_from_the_round_number(tmp_path):
    """A dose arm above the baseline spends two revision passes per round, and there R and the round
    number part: R runs 2, 4, 6 while round + 1 runs 1, 2, 3.

    THIS IS THE CASE THAT DISCRIMINATES. The baseline arm above spends exactly one pass per round, so
    an implementation returning `float(r + 1)` passes that assertion verbatim, and the conflation the
    source's own docstring forbids would not be visible in any test. The dose is applied from round
    zero here so that the divergence is present in the rounds the case reads.
    """
    lad = CD.SuiteLadder(_pool(), batch_runner=CD.inprocess_batch_runner())
    store = CD.CheckpointStore(str(tmp_path))
    store.save("seed", CD.new_artefact(_library({0, 1, 2})))
    source = CD.suite_margin_source(ScriptedSwapSystem([{0, 1, 2}, {0, 1, 3}, {0, 1, 3, 4}]),
                                    lad, store, "Improve the solutions library.",
                                    CD.DoseSchedule().dose_for, switch_round=0)
    spec = OBS.spec_of(source)
    rng = np.random.default_rng(0)
    readings = [OBS.read(source, "dose+1.0", 3.0, r, rng, spec) for r in range(3)]
    assert [r.R for r in readings] == [2.0, 4.0, 6.0], \
        "two revision passes a round travel twice as far in R as the rounds they took"
    assert [r.R for r in readings] != [1.0, 2.0, 3.0], "and the round number is not that coordinate"


def test_an_artefact_that_does_not_track_its_revision_depth_supplies_no_coordinate_and_is_refused(tmp_path):
    """The same fail-open the estimator had, in the source that feeds it: the depth was
    `art.get("rounds", 0) or float(r + 1)`, so an artefact that tracks nothing became the round
    number silently. An untracked artefact has no recursive coordinate, and a reading without one is
    refused rather than given a substitute that means something else."""
    class UntrackedSystem:
        name = "untracked"

        def revise(self, artefact, retained, task, rng):
            out = dict(artefact)
            out["text"] = _library({0, 1, 2})
            return out                            # no revision depth is recorded

    lad = CD.SuiteLadder(_pool(), batch_runner=CD.inprocess_batch_runner())
    store = CD.CheckpointStore(str(tmp_path))
    store.save("seed", CD.new_artefact(_library({0, 1, 2})))
    source = CD.suite_margin_source(UntrackedSystem(), lad, store, "Improve the solutions library.",
                                    CD.DoseSchedule().dose_for, switch_round=8)
    spec = OBS.spec_of(source)
    with pytest.raises(OBS.ObservationRefusal) as exc:
        OBS.read(source, "baseline", 2.0, 0, np.random.default_rng(0), spec)
    assert "carried no recursive coordinate R" in str(exc.value)
