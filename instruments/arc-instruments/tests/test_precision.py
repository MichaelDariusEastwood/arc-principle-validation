import math

from arc_instruments import precision as pr


def test_ceiling_and_sensitivity_at_one_half():
    assert pr.implied_ceiling(0.5) == 2.0
    assert pr.ceiling_sensitivity(0.5) == 4.0
    assert abs(pr.gamma_halfwidth_required(0.5, 0.10) - 0.025) < 1e-12


def test_exact_file_audit_witness_interval():
    lo, hi = pr.ceiling_interval((0.42, 0.58))
    assert abs(lo - 1.724137931) < 1e-6 and abs(hi - 2.380952381) < 1e-6


def test_sample_scale_factor_is_inverse_square():
    assert abs(pr.sample_scale_factor(0.07, 0.025) - (0.07 / 0.025) ** 2) < 1e-12
    r = pr.p3_reachability(0.5, 0.10, 0.07)
    assert r["reachable_as_registered"] is False and r["sample_scale_factor_needed"] > 7


def test_exponent_se_matches_regression_formula():
    se = pr.exponent_se(7, 1.5, 0.02)
    assert 0.005 < se < 0.010
    assert pr.exponent_se(12, 2.5, 0.02) < se


def test_sign_power_monotone_in_margin_and_precision():
    p1 = pr.sign_power(0.05, 0.02, 0.07, reps=4000)
    p2 = pr.sign_power(0.20, 0.02, 0.07, reps=4000)
    p3 = pr.sign_power(0.20, 0.02, 0.02, reps=4000)
    assert p1 < p2 <= p3 and p3 > 0.99


def test_gamma_domain_enforced():
    try:
        pr.implied_ceiling(1.0)
    except ValueError:
        return
    raise AssertionError("gamma of one must not return a finite ceiling")
