"""The consolidated package's eighteen witnesses (5 September 2026, late afternoon) and its two model
findings, pinned against the engine's second edition. Where a witness targeted the first edition and
the second edition had already fixed it, the case is kept so that it cannot regress."""
import numpy as np

from arc_instruments import balance as bl
from arc_instruments import blinding as bd
from arc_instruments import dependence as dp
from arc_instruments import diversity as dv
from arc_instruments import form_discrimination as fd
from arc_instruments import precision as pr
from arc_instruments import sealing as sl
from arc_instruments import verdicts as V


def test_a01_p12_negative_material_change_supports():
    assert V.p12_build_order((-0.30, -0.20), True) == V.SUPPORTED


def test_a02_p13_is_one_conjunction_with_named_cells():
    assert V.p13_conjunction((0.05, 0.20), V.SUPPORTED) == V.SUPPORTED
    assert V.p13_conjunction((0.05, 0.20), V.REFUTED) == V.REFUTED
    assert V.p13_conjunction((0.05, 0.20), V.INCONCLUSIVE) == V.INCONCLUSIVE
    assert V.p13_conjunction((0.05, 0.20), V.NOT_EVALUABLE) == V.NOT_EVALUABLE
    assert V.p13_conjunction((-0.20, -0.05), V.SUPPORTED) == V.REFUTED
    c = V.p13_components((0.05, 0.20), V.INCONCLUSIVE)
    assert c["cell"] == V.PERFORMANCE_SUPPORTED_MECHANISM_UNRESOLVED and c["aggregate"] == V.INCONCLUSIVE
    assert V.p13_components((0.05, 0.20), V.REFUTED)["cell"] == V.PERFORMANCE_SUPPORTED_MECHANISM_REFUTED
    try:
        V.p13_conjunction((0.05, 0.20), "maybe")
    except ValueError:
        pass
    else:
        raise AssertionError("an untyped dependence verdict must be refused")


def test_a03_a04_p21_sensitivity_gate_is_symmetric_and_needs_the_high_depth_effect():
    assert V.p21_interaction((0.12, 0.30), 0.10, False) == V.INSTRUMENT_FAILED
    assert V.p21_interaction((-0.05, 0.05), 0.10, False) == V.INSTRUMENT_FAILED
    assert V.p21_interaction((0.12, 0.30), 0.10, True) == V.INCONCLUSIVE                      # no high-depth effect supplied
    assert V.p21_interaction((0.12, 0.30), 0.10, True, high_depth_effect=(0.05, 0.30)) == V.SUPPORTED
    assert V.p21_interaction((0.12, 0.30), 0.10, True, high_depth_effect=(-0.30, -0.05)) == V.REFUTED
    assert V.p21_interaction((-0.05, 0.05), 0.10, True) == V.REFUTED


def test_a05_p1_unresolved_cells_are_not_rival_wins():
    assert V.p1_form([V.INCONCLUSIVE] * 8, 0.05, 0.10) == V.INCONCLUSIVE
    assert V.p1_form(["power"] * 5 + [V.INCONCLUSIVE] * 3, 0.05, 0.10) == V.SUPPORTED
    assert V.p1_form(["power"] * 4 + [V.INCONCLUSIVE] * 4, 0.05, 0.10) == V.INCONCLUSIVE   # unresolved cells stay in the denominator
    assert V.p1_form(["saturating"] * 5 + ["power"] * 2 + ["shifted"], 0.05, 0.10) == V.REFUTED
    try:
        V.p1_form(["power", "banana"], 0.05, 0.10)
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown cell label must be refused")


def test_a06_a07_p2_imprecision_is_not_flat_and_endpoint_maximum_is_not_monotone():
    betas = [0.1, 0.5, 0.9]
    assert V.p2_shape(betas, [0.0, 3.0, 0.0], [100.0] * 3) == V.UNRESOLVED
    assert V.p2_shape(betas, [0.0, 3.0, 0.0], [100.0] * 3, flat_margin=0.5) == V.UNRESOLVED
    assert V.p2_shape([0.1, 0.3, 0.6, 0.9], [4.0, 0.0, 2.0, 1.0], [0.01] * 4) == V.ENDPOINT_MAXIMUM
    assert V.p2_shape([0.1, 0.3, 0.6, 0.9], [1.0, 2.0, 3.0, 4.0], [0.01] * 4) == V.MONOTONE
    assert V.p2_shape([0.1, 0.3, 0.6, 0.9, 1.0], [2.0, 2.02, 1.99, 2.01, 2.0], [0.01] * 5, flat_margin=0.10) == V.FLAT
    assert V.p2_shape([0.1, 0.3, 0.6, 0.9, 1.0], [2.0, 2.02, 1.99, 2.01, 2.0], [0.01] * 5) == V.UNRESOLVED


def test_a08_a09_p6_pending_counterexample_and_empty_roster():
    assert V.p6_survey([(0.52, 0.70)], replicated=False) == V.AWAITING_REPLICATION
    assert V.p6_survey([]) == V.NO_SURVEY


def test_a10_a12_contacts_inconclusive():
    assert V.p22_typed_ordering((-0.10, 0.10)) == V.INCONCLUSIVE
    assert V.p14_blinded_shrink((0.0, 0.20)) == V.INCONCLUSIVE
    assert V.p15_decay((0.50, 0.80)) == V.INCONCLUSIVE


def test_a13_p20_has_no_geometric_gate():
    adv = {"reciprocal": (0.15, 0.40), "constant": (0.20, 0.50)}
    assert V.p20_form([0.2, 0.4], adv, margin=0.10) == V.NOT_EVALUABLE                  # no calibrated power supplied
    assert V.p20_form([0.2, 0.4], adv, margin=0.10, discriminating_power=0.95) == V.SUPPORTED


def test_a14_all_fits_failed_is_a_fit_failure():
    r = fd.ladder(3, 1.0)
    y = np.array([1.0, 2.0, 3.0])
    assert fd.select_family(r, y, candidates=("broken",)) == fd.FIT_FAILURE   # four parameters on three points: no valid AICc


def test_a15_table_covers_all_twenty_two():
    ids = {row["proposition"] for row in V.scoring_table()}
    assert {"P%d" % i for i in range(1, 23)} <= ids


def test_a16_james_index_is_chance_adjusted():
    actual = ["A"] * 90 + ["B"] * 10
    assert abs(bd.james_index(actual, ["A"] * 100) - 0.5) < 1e-12
    balanced = ["A", "B"] * 50
    assert abs(bd.james_index(balanced, ["A"] * 100) - 0.5) < 1e-12
    assert bd.james_index(balanced, balanced) == 0.0
    assert bd.james_index(balanced, [bd.DONT_KNOW] * 100) == 1.0


def test_a17_a18_copying_is_a_diagnostic_and_delivery_is_the_gate():
    d = dv.diagnostics({"sequential": [["same"] * 4, ["same"] * 3], "empty": []})
    assert d["sequential"]["mean_copy_rate"] == 1.0 and d["empty"]["state"] == dv.NO_OBSERVATIONS
    g = dv.delivery_gate({"sequential": [True, True], "broken": [True, False], "empty": []})
    assert g["sequential"]["valid"] is True and g["broken"]["valid"] is False and g["empty"]["valid"] is None
    try:
        dv.validity_gate({})
    except RuntimeError:
        pass
    else:
        raise AssertionError("the retired gate must refuse")


def test_p7_census_both_counts():
    assert V.p7_census((0.55, 0.80), (0.60, 0.85)) == V.SUPPORTED
    assert V.p7_census((0.10, 0.40), (0.20, 0.45)) == V.REFUTED
    assert V.p7_census((0.55, 0.80), (0.30, 0.45)) == V.INCONCLUSIVE
    assert V.p7_census((0.50, 0.80), (0.60, 0.85)) == V.INCONCLUSIVE


def test_s1_shifted_trajectory_balance_is_zero_when_beta_equals_chi():
    R = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 24.0, 32.0])
    t = bl.shifted_local_terms(R, U0=1.0, a=0.1, beta_L=0.5)
    assert np.allclose(t["U"], (0.95 + 0.05 * R) ** 2)
    corrected = [bl.local_balance(a, c, 0.5) for a, c in zip(t["alpha_R"], t["dlog_alpha_dlogR"])]
    naive = [bl.balance_elasticity(a, 0.5) for a in t["alpha_R"]]
    assert all(abs(bl.shifted_trajectory_balance(0.5, 0.5, a)) < 1e-12 for a in t["alpha_R"])
    assert max(abs(x) for x in corrected) < 1e-12 and min(naive) > 0.3      # the curvature term cancels the false surplus exactly
    dense = np.linspace(1.0, 32.0, 4000)
    n = bl.local_exponent_and_curvature(dense, (0.95 + 0.05 * dense) ** 2)
    assert max(abs(bl.local_balance(a, c, 0.5)) for a, c in zip(n["alpha_R"][5:-5], n["dlog_alpha_dlogR"][5:-5])) < 0.01
    assert abs(bl.local_balance(1.0, 1.0, 0.5) + 0.5) < 1e-12


def test_s4_s5_chain_power_and_interval_clearance():
    assert abs(pr.conjunction_power([0.8] * 5) - 0.32768) < 1e-12
    assert abs(pr.conjunction_power([0.8] * 10) - 0.1073741824) < 1e-12
    assert abs(pr.component_power_for_joint(0.8, 5) - 0.9563525) < 1e-6
    r = pr.interval_clearance_power(0.10, 0.02, 0.07 / 1.96, gamma=0.5)
    assert abs(r["sd_pressure"] - 0.0792) < 2e-5
    assert abs(r["point_sign_probability"] - 0.897) < 0.005
    assert abs(r["interval_clearance_power"] - 0.244) < 0.005


def test_s6_s7_exact_gamma_band_and_coefficient_intervention():
    e = pr.exact_gamma_interval_for_ceiling_band(1.9, 2.1)
    assert abs(e["gamma_lo"] - 0.4736842) < 1e-6 and abs(e["gamma_hi"] - 0.5238095) < 1e-6
    assert abs(e["max_symmetric_halfwidth_about_half"] - 0.0238095) < 1e-6
    c = bl.coefficient_intervention(10.0, -0.5, 2.0)
    assert abs(c["crossing_before"] - 100.0) < 1e-9 and abs(c["crossing_after"] - 400.0) < 1e-9
    corr = bl.author_review_correspondence(0.2, 0.0)
    assert corr["kappa"] == 1.2 and corr["zeta_total"] == -1.0


def test_p17_excess_scale_keeps_negative_dependence_informative():
    rng = np.random.default_rng(3)
    n = 1200
    a = rng.random(n) < 0.3
    b = np.where(a, rng.random(n) < 0.05, rng.random(n) < 0.45)   # strong negative dependence
    misses = np.column_stack([a, b]).astype(int)
    ex = dp.pairwise_dependence(misses, None, margin=0.03, n_boot=200, scale="excess")
    assert ex["pairs"]["0-1"]["verdict"] == dp.ANTI_CORRELATED
    assert ex["pairs"]["0-1"]["bootstrap_draws_dropped"] == 0


def test_form_families_include_logarithmic_and_shifted_and_report_false_support():
    truths = {"logarithmic b=0.8": ("logarithmic", lambda r: fd.logarithmic(r, 1.0, 0.8)),
              "shifted s=19 a=2": ("shifted", lambda r: fd.shifted(r, 0.0025, 19.0, 2.0))}
    m = fd.confusion_matrix(points=9, decades=2.0, noise=0.02, reps=8, truths=truths, seed=2)
    fs = fd.false_support_rate(m)
    assert set(fs) == {"logarithmic b=0.8", "shifted s=19 a=2", "_worst_case"}
    assert 0.0 <= fs["_worst_case"] <= 1.0


def test_sealing_round_trip_and_holdout():
    payload = {"proposition": "P5", "beta_L": 0.5, "a": 0.1, "U0": 1.0, "window": [1, 32], "predicted_exponent": 0.5402}
    rec = sl.seal(payload, spec_sha256="deadbeef", sealed_by="a test")
    assert sl.verify(rec, payload, "deadbeef")["verified"] is True
    assert sl.verify(rec, dict(payload, predicted_exponent=0.6), "deadbeef")["verified"] is False
    part = sl.holdout_partition(["s%02d" % i for i in range(20)], 0.25, seed=7)
    assert part["n_holdout"] == 5 and set(part["holdout"]).isdisjoint(part["calibration"])
    assert part["holdout_sha256"] == sl.sha256_of(part["holdout"])
