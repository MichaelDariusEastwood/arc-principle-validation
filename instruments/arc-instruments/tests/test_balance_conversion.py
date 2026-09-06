import math

import numpy as np

from arc_instruments import balance as bl
from arc_instruments import conversion as cv
from arc_instruments import verdicts as V


def test_trend_crossover_and_domain():
    assert bl.trend_crossover(0.5) == 2.0
    assert abs(bl.balance_elasticity(2.0, 0.5)) < 1e-12
    assert bl.balance_elasticity(3.0, 0.5) < 0
    for bad in (1.0, 1.2, -0.1):
        try:
            bl.trend_crossover(bad)
        except ValueError:
            continue
        raise AssertionError("chi outside [0, 1) must not return a ceiling")


def test_counterexample_above_bound_has_ample_service():
    c = bl.counterexample_above_bound(32.0)
    assert c["alpha"] > c["trend_crossover"]
    assert abs(c["ratio_at_R"] - 17.6776695) < 1e-5
    assert abs(c["ratio_crossing_depth"] - 10000.0) < 1e-6
    d = bl.describe(*(lambda R: (R, 300.0 * R ** 1.5, 3.0 * R ** 2))(np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])))
    assert d["trend"] < 0 and d["level_above_one_throughout"] and d["first_depth_ratio_below_one"] is None


def test_counterexample_favourable_trend_has_inadequate_service():
    c = bl.counterexample_favourable_trend(32.0)
    assert c["balance_elasticity"] > 0 and c["ratio_at_R"] < 1.0
    assert abs(c["ratio_crossing_depth"] - 10000.0) < 1e-6


def test_ratio_crossing_depth_cases():
    assert math.isinf(bl.ratio_crossing_depth(2.0, 0.1))
    assert math.isinf(bl.ratio_crossing_depth(0.5, -0.1))
    assert abs(bl.ratio_crossing_depth(100.0, -0.5) - 10000.0) < 1e-6
    assert bl.ratio_crossing_depth(1.0, -0.3) == 1.0


def test_model_paths_recover_the_balance_elasticity():
    R = np.logspace(0, 2, 20)
    p = bl.model_paths(R, alpha=1.5, chi=0.4)
    assert abs(bl.fitted_trend(R, p["M"]) - bl.balance_elasticity(1.5, 0.4)) < 1e-9


def test_endpoint_typing():
    assert bl.endpoint_requires_response_model(bl.MARGIN_REVERSAL) is False
    assert all(bl.endpoint_requires_response_model(e) for e in (bl.SERVICE_DEFICIT, bl.BACKLOG_THRESHOLD, bl.CONFORMANCE_FAILURE))
    assert V.p16_endpoint(V.SERVICE_DEFICIT, True, 20.0, (15.0, 25.0), 40.0, True, True) == V.NOT_EVALUABLE
    assert V.p16_endpoint(V.SERVICE_DEFICIT, True, 20.0, (15.0, 25.0), 40.0, True, True, response_model_registered=True) == V.SUPPORTED
    assert V.p16_endpoint(V.MARGIN_REVERSAL, True, 20.0, (15.0, 25.0), 40.0, True, True) == V.SUPPORTED
    assert V.p16_endpoint(V.MARGIN_REVERSAL, True, 20.0, (15.0, 25.0), 40.0, True, False) == V.AWAITING_REPLICATION
    assert V.p16_endpoint(V.MARGIN_REVERSAL, True, 20.0, (15.0, 25.0), 40.0, False, True) == V.INCONCLUSIVE
    assert V.p16_endpoint(V.MARGIN_REVERSAL, True, 30.0, (15.0, 25.0), 40.0, True, True) == V.REFUTED
    assert V.p16_endpoint(V.MARGIN_REVERSAL, True, None, (15.0, 25.0), 40.0, True, True) == V.REFUTED
    assert V.p16_endpoint(V.MARGIN_REVERSAL, True, None, (15.0, 25.0), 20.0, True, True) == V.INCONCLUSIVE
    assert V.p16_endpoint(V.MARGIN_REVERSAL, False, None, (15.0, 25.0), 40.0, True, True) == V.INCONCLUSIVE


def test_chain_rule_worked_example():
    w = cv.worked_example()
    assert abs(w["target_axis"] - 0.4) < 1e-12 and abs(w["product_only"] - 0.8) < 1e-12
    assert abs(w["ceiling_target_axis"] - 1.6666666667) < 1e-9 and abs(w["ceiling_product_only"] - 5.0) < 1e-9


def test_crossed_design_identifies_both_partials_and_one_path_does_not():
    c = cv.crossed_design(-0.4, 0.8, noise=0.02, seed=1)
    assert abs(c["direct_hat"] + 0.4) < 0.03 and abs(c["cross_hat"] - 0.8) < 0.03 and c["identified"]
    o = cv.one_path_design(-0.4, 0.8, allocation=1.0, noise=0.02, seed=1)
    assert abs(o["total_hat"] - 0.4) < 0.03 and o["identified"] is False


def test_path_elasticity_is_not_chi_plus_eta():
    assert abs(cv.path_elasticity(0.5, 0.2, 2.0) - 0.6) < 1e-12
    try:
        cv.path_elasticity(0.5, 0.2, 0.0)
    except ValueError:
        return
    raise AssertionError("zero growth exponent must be refused")
