import numpy as np

from arc_instruments import coupling_identification as ci
from arc_instruments import dependence as dp


def test_exact_file_audit_witnesses():
    assert abs(ci.endpoint_slope(1.0, 0.1, 0.5, 1.0, 32.0) - 0.540198899) < 1e-6
    seven = np.array([1, 2, 4, 8, 16, 24, 32], float)
    assert abs(ci.loglog_fit_slope(1.0, 0.1, 0.5, seven) - 0.538792977) < 1e-6


def test_retention_only_estimates_theta_not_beta():
    r = ci.recovery_study(beta=0.5, theta=0.6, noise=0.05, reps=12)
    assert abs(r["retention_only"]["slope_mean"] - 0.6) < 0.05
    assert abs(r["crossed"]["beta_mean"] - 0.5) < 0.05
    assert abs(r["crossed"]["theta_mean"] - 0.6) < 0.05


def test_negative_control_no_coupling():
    r = ci.recovery_study(beta=0.0, theta=0.6, noise=0.05, reps=12)
    assert abs(r["retention_only"]["slope_mean"] - 0.6) < 0.05      # would be misread as a coupling
    assert abs(r["crossed"]["beta_mean"]) < 0.05                    # the crossed design says none


def test_sealed_prediction_check():
    R = np.array([1, 2, 4, 8, 16, 32], float)
    observed = ci.trajectory(1.0, 0.1, 0.5, R)
    res = ci.sealed_prediction_check(0.5, 1.0, 0.1, R, observed)
    assert res["within_margin"] is True and abs(res["difference"]) < 1e-9


def test_shared_difficulty_manufactures_marginal_correlation():
    misses, d = dp.simulate_panel(n_faults=800, n_correctors=3, difficulty_sd=1.5, seed=9)
    un = dp.pairwise_dependence(misses, None, n_boot=150, seed=1)
    st = dp.pairwise_dependence(misses, dp.difficulty_strata(d, 5), n_boot=150, seed=1)
    un_v = [p["verdict"] for p in un["pairs"].values()]
    st_v = [p["verdict"] for p in st["pairs"].values()]
    assert any(v == dp.CORRELATED for v in un_v)
    assert all(v in (dp.INDEPENDENT, dp.INSUFFICIENT_PRECISION) for v in st_v)
    assert all(p["log_ratio"] < q["log_ratio"] for p, q in zip(st["pairs"].values(), un["pairs"].values()))


def test_common_shock_survives_stratification():
    misses, d = dp.simulate_panel(n_faults=800, n_correctors=3, difficulty_sd=1.0, common_shock_sd=2.0, seed=21)
    st = dp.pairwise_dependence(misses, dp.difficulty_strata(d, 5), n_boot=150, seed=2)
    assert any(p["verdict"] == dp.CORRELATED for p in st["pairs"].values())
