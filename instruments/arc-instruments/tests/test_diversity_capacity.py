import numpy as np

from arc_instruments import capacity as cp
from arc_instruments import diversity as dv


def test_copy_rate_flags_verbatim_repeats():
    revising = ["first attempt at the proof", "second attempt with a corrected lemma", "third attempt tightening the bound"]
    copying = ["the same output every round"] * 4
    assert dv.copy_rate(revising) == 0.0
    assert dv.copy_rate(copying) == 1.0
    diag = dv.diagnostics({"sequential": [copying, copying], "revising": [revising]})
    assert diag["sequential"]["mean_copy_rate"] == 1.0 and diag["revising"]["mean_copy_rate"] == 0.0
    gate = dv.delivery_gate({"sequential": [True, True], "revising": [True]})
    assert gate["sequential"]["valid"] is True and gate["revising"]["valid"] is True   # copying never invalidates a delivered run


def test_similarity_bounds():
    s = dv.successive_similarity(["abcd efgh", "abcd efgh", "zzzz yyyy"])
    assert s[0] == 1.0 and 0.0 <= s[1] < 0.2


def test_net_correction_can_be_negative():
    assert cp.net_correction(100, 30, 45) == -15.0


def test_supply_regime_classification():
    assert cp.supply_regime([10, 20, 40, 80], [10, 20, 39, 78])["regime"] == "supply-limited"
    r = cp.supply_regime([10, 20, 40, 80], [10, 19, 25, 26])
    assert r["regime"] == "capacity-limited" and 24 < r["capacity_estimate"] < 27
    assert cp.supply_regime([10, 20, 40], [10, 18, 30])["regime"] == "transition"


def test_deming_recovers_slope_where_ols_attenuates():
    rng = np.random.default_rng(3)
    true = 0.5
    x_true = rng.uniform(0, 4, 400)
    y = true * x_true + rng.normal(0, 0.3, 400)
    x_obs = x_true + rng.normal(0, 0.3, 400)
    ols = cp.ols_slope(x_obs, y); dem = cp.deming_slope(x_obs, y, delta=1.0)
    assert ols < true - 0.02
    assert abs(dem - true) < 0.05


def test_capacity_elasticity_interval_contains_truth():
    # 400 cells: at 60 cells the fixed-seed sample can sit two standard errors from the truth, which is
    # a statement about the sample and not about the estimator (checked across seeds at 400).
    rng = np.random.default_rng(4)
    cap = np.exp(rng.uniform(0, 3, 400))
    capacity = cap ** 0.5 * np.exp(rng.normal(0, 0.05, 400))
    cap_obs = cap * np.exp(rng.normal(0, 0.05, 400))
    r = cp.capacity_elasticity(cap_obs, capacity, delta=1.0, n_boot=300)
    lo, hi = r["interval"]
    assert lo < 0.5 < hi and abs(r["elasticity"] - 0.5) < 0.03
    assert (hi - lo) < 0.05
