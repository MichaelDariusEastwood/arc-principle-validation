"""End-to-end tests of the runner against the simulated system with a known coupling.

These are the tests that prove the pipeline before a real system is touched: the crossed bank recovers
the true coupling by two routes, the sealed prediction agrees with the held-out fit, the seal is written
before the continuation exists, a pilot cannot be scored, and the rate-confound world produces the
correct-prediction-wrong-mechanism pair that ruling 25 exists to report.
"""
import numpy as np
import pytest

from arc_runner import adapters, ladder as L, manifest as M, p5, p16


def _mock(beta=0.5, theta=0.0, noise=0.05, n_items=20000, scale=400.0):
    # The precision here is what the design needs, not what is cheap: the first runs at 2,000 and
    # 4,000 items recovered 0.24 and 0.66 for a true 0.50 and predicted exponents that varied by 0.3
    # across identical systems, because the calibrated rate is the binding precision. That is the
    # pilot's finding, made in code before any pilot ran.
    ad = adapters.MockCouplingAdapter(beta=beta, theta=theta, noise=noise)
    lad = L.MockLadder(n_items=n_items, scale=scale)
    place = lambda s: {"kind": "mock", "capability": float(s), "rounds": 0}
    start = lambda name: {"kind": "mock", "capability": 20.0, "rounds": 0, "system": name}
    return ad, lad, place, start


def test_crossed_bank_recovers_the_coupling_by_both_routes():
    """The two regression directions recover the coupling, and that is CONSISTENCY, not identification.

    THIS CASE'S LAST ASSERTION CHANGED WITH FINDING A6, AND THE CHANGE IS THE FINDING. It used to
    assert that this run said IDENTIFIED. It said that from two slopes taken through one bank in two
    directions: nothing was manipulated twice, the two directions share every cell, every artefact
    and every read, and a nuisance rate that scales with the available capability moves both of them
    by the same amount. What the agreement of those two numbers establishes is that the elasticity in
    the retention fraction and the elasticity in the capability state are one number, which is worth
    knowing and is not identification. So the run reports CONSISTENT for the numbers and NOT
    ESTABLISHED for the identification, the second naming what is missing: an independent capability
    manipulation. Identification reached on one is exercised in tests/test_p5_identification.py.
    """
    ad, lad, place, start = _mock(beta=0.5)
    cfg = p5.P5Config(reps=8)   # eight replicates per cell: the negative control needs two control cells per (state, fraction) to resolve
    bank = p5.run_bank(ad, lad, cfg, np.random.default_rng(1), place)
    r = p5.estimate_routes(bank, cfg)
    assert abs(r["beta_state_route"] - 0.5) < 0.12
    assert abs(r["beta_retention_route"] - 0.5) < 0.12
    assert r["route_agreement"] == "CONSISTENT", r["route_consistency"]
    assert r["identification"] == "NOT ESTABLISHED", r["identification_judgement"]
    assert r["identification_judgement"]["n_manipulations_supplied"] == 0


def test_a_ladder_without_headroom_is_refused_not_scored():
    # ruling 29: a system reaching the top rung before the final checkpoint is NOT EVALUABLE, never a ceiling
    ad, lad, place, start = _mock(beta=0.5, noise=0.03)
    lad = L.MockLadder(n_items=20000, scale=100.0)         # far too little headroom for depth 64
    cfg = p5.P5Config(reps=3, window_end=64, checkpoints=(4, 8, 16, 32, 64))
    res = p5.run_p5(ad, lad, cfg, 7, place, start, ["S1", "S2"])
    assert res["diagnostics"]["PREDICTION"] == "NOT EVALUABLE"
    assert all(v["verdict"] == "NOT EVALUABLE" for v in res["diagnostics"]["per_system"].values())


def test_seal_precedes_generation_and_prediction_agrees():
    ad, lad, place, start = _mock(beta=0.5, noise=0.03)
    cfg = p5.P5Config(reps=8, window_end=32, checkpoints=(4, 8, 16, 32))
    res = p5.run_p5(ad, lad, cfg, 7, place, start, ["S1", "S2", "S3"])
    man = res["manifest"]
    assert man["seal"] is not None and man["seal"]["predictions_sha256"]
    assert man["code_sha256"] and man["ladder_sha256"] == lad.sha256
    v = res["diagnostics"]
    assert v["PREDICTION"] == "SUPPORTED", v
    # Finding A6: the routes agree numerically, the run supplied no second capability intervention,
    # and the two facts are reported as the two different things they are.
    assert v["ROUTE_AGREEMENT"] == "CONSISTENT", v["route_consistency"]
    assert v["IDENTIFICATION"] == "NOT ESTABLISHED", v["identification_judgement"]
    # Finding A5: the verdict block says of itself whether every registered level is in every route,
    # because a route reported as the mean of the levels that happened to grow looked exactly like a
    # route reported from all of them, and it declares that the estimator the registration names as
    # primary has been superseded without an amendment the author has made.
    assert set(v["route_levels"]) == {"state_route", "retention_route"}
    for name, level in v["route_levels"].items():
        assert level["levels_dropped"] == 0 and level["levels_missing"] == [], (name, level)
        assert level["n_levels"] == len(cfg.fractions if name == "state_route" else cfg.states)
    assert v["registered_estimator"]["amendment_required"] is True
    assert v["registered_estimator"]["amendment_status"].startswith("NOT RATIFIED")


def test_pilot_can_run_but_never_be_scored():
    ad, lad, place, start = _mock()
    cfg = p5.P5Config(reps=2, window_end=16, checkpoints=(4, 8, 16))
    res = p5.run_p5(ad, lad, cfg, 3, place, start, ["S1"], pilot=True)
    assert "verdicts" not in res
    with pytest.raises(M.PilotNotScoreable):
        p5.verdicts(res["manifest"], res["routes"], res["heldout"], cfg)


def test_rate_confound_world_gives_correct_prediction_with_wrong_mechanism():
    # a nuisance rate growing with capability: the bank measures a larger coupling that still predicts
    # the path, because the same confound governs both; ruling 25's typed pair is what reports it
    ad, lad, place, start = _mock(beta=0.5, theta=0.2, noise=0.03, scale=2000.0)   # headroom for the faster world
    cfg = p5.P5Config(reps=64, window_end=32, checkpoints=(4, 8, 16, 32))
    # sixty-four replicates per cell: at an exponent near 1.6 the prediction moves about 2.4 exponent units per
    # unit of coupling, so H2 in precision (predicted interval narrower than the margin) needs a coupling
    # standard error near 0.008, four times the bank of the clean world; the runner battery measured this
    res = p5.run_p5(ad, lad, cfg, 11, place, start, ["S1", "S2", "S3"])
    r = res["routes"]
    # the state route carries the confound, the retention route recovers the true coupling, and the
    # bank says so: this is the separation a single ladder cannot make
    assert r["beta_state_route"] > r["beta_retention_route"] + 0.1, r
    assert abs(r["beta_retention_route"] - 0.5) < 0.12, r
    assert r["identification"] == "NOT IDENTIFIED", r
    # AND THE STATE-ROUTE COUPLING STILL PREDICTS THE PATH: no held-out system disagrees with its
    # prediction, and the panel is never refuted. That is the pair ruling 25 reports here, and it is
    # what this world can carry.
    #
    # WHY THE PANEL'S WORD IS NOT ASSERTED. It used to read SUPPORTED, on a panel of three of which
    # one cleared the margin by eight ten-thousandths: the difference interval reached 0.0992 against
    # a margin of 0.100. The reading therefore turned on the fourth decimal of the held-out scores,
    # and it moved to INCONCLUSIVE the moment an averaged read stopped being rounded back to a whole
    # count, which is a repair to the measurement and not to this world. Measured across bank sizes
    # the word is not stable either: 64 and 128 replicates per cell read INCONCLUSIVE and 96 reads
    # SUPPORTED, at a bank cost that changes nothing about the mechanism. A claim that turns on a
    # thousandth is not a claim, so what is asserted is the part that holds at every size: the
    # prediction is never contradicted, and the mechanism is not identified.
    d = res["diagnostics"]
    assert d["PREDICTION"] in ("SUPPORTED", "INCONCLUSIVE"), d
    assert d["panel"]["disagrees"] == 0, d["per_system"]
    assert d["IDENTIFICATION"] == "NOT IDENTIFIED"


def test_p16_titration_supports_a_true_boundary_and_does_not_refute_from_silence():
    """The arm pattern, which is what this end-to-end case has always exercised.

    Finding A2 moved the `P16` key to the component wrapper's result, so the run's reading of its own
    arms is read here under its own name, `run_pattern`, with its five labels unchanged. The
    wrapper's refusal on the same run is asserted beside it: this source is the unregistered
    per-round surplus rate and attests no delivery, so the proposition is not decided by it.

    THE SECOND HALF OF THIS CASE CHANGED WITH FINDING A3, AND THE CHANGE IS THE FINDING. The flat
    world below has no boundary and no arm reverses, and this case used to assert that the runner
    said `REFUTED (no reversal)` for it. It said that from silence alone: no arm's margin was
    measured positive, no informative horizon was registered, and a world too noisy to decide would
    have produced the same label. The flat world's terminal intervals span zero, so the arms carry no
    information about the margin at all, and the honest label is inconclusive. Refutation from a
    measured positive margin past a registered horizon is exercised in tests/test_p16_non_alarm.py.
    """
    cfg = p16.P16Config(systems_per_arm=3, horizon=96)
    true_src = p16.mock_margin_source(cfg, true_alpha_crit=2.0)
    res = p16.run_p16(true_src, cfg, 5, "none", "mock")
    assert res["manifest"]["seal"]["sealed_at_utc"]
    # Both names address one object, so the export is checked here as well as the reading: a run
    # whose diagnostics key went missing would otherwise fail somewhere far from the change.
    assert res["diagnostics"] is res["verdicts"]
    assert res["evidence_status"] == "SIMULATION/DEVELOPMENT ONLY"
    assert res["empirical_verdict"] == "NOT TESTED"
    assert res["diagnostics"]["run_pattern"] == "SUPPORTED", res["diagnostics"]
    assert res["diagnostics"]["P16"] == "NOT EVALUABLE", res["diagnostics"]["component_reason"]
    assert res["diagnostics"]["falling_line_gate"]
    # no boundary: no arm's trend depends on alpha
    def flat(arm, alpha_arm, r, rng):
        return 0.5 + (0.15 if r >= cfg.switch_round else 0.0) + rng.normal(0, cfg.margin_noise)
    res0 = p16.run_p16(flat, cfg, 6, "none", "mock")
    assert res0["diagnostics"]["run_pattern"] == \
        "INCONCLUSIVE (no reversal, and none demonstrated absent)", res0["diagnostics"]
    assert res0["diagnostics"]["refutation"]["refutes"] is False
    assert res0["diagnostics"]["refutation"]["n_demonstrated_positive"] == 0


def test_settling_period_keeps_the_switch_jump_out_of_the_slope():
    # a jump at the switch and a flat trend afterwards must not be read as a reversal more often than
    # the registered per-arm false-alarm rate at the flat null (about five per cent at z = 3.0)
    cfg = p16.P16Config(systems_per_arm=1, horizon=60, settling=6)
    alarms = 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        ser = [0.5 + (0.3 if r >= cfg.switch_round else 0.0) + rng.normal(0, 0.02) for r in range(60)]
        alarms += p16.detect_reversal(ser, cfg)["declared_round"] is not None
    assert alarms / 200 <= 0.10, "false alarms at the flat null: %d of 200" % alarms
