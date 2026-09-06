from arc_instruments import designs as ds


def test_p5_design_has_power_under_truth_and_controls_false_affirmatives():
    r = ds.p5_design_power(sims=40, seed=3)
    assert r["support_rate_under_truth"] >= 0.80
    assert r["refutation_rate_under_truth"] <= 0.05
    assert r["false_affirmative_rate_under_shifted_coupling"] <= 0.05
    assert r["detection_rate_of_shifted_coupling"] >= 0.80
    assert r["non_identification_declared_under_nuisance_rate"] >= 0.80
    assert 0.2 < r["coupling_shift_resolvable_by_band"] < 0.35          # a 0.10 band resolves shifts of about 0.26 here


def test_p5_band_cannot_see_a_small_coupling_shift():
    r = ds.p5_design_power(sims=30, seed=5, alt_shift=0.10)
    assert r["detection_rate_of_shifted_coupling"] < 0.20              # the charter must name a larger alternative or a longer window


def test_p5_measurement_error_attenuates_the_coupling_the_band_can_hide():
    noisy = ds.p5_design_power(sims=20, seed=5, cap_noise=0.5)
    assert noisy["mean_calibration_beta_hat"] < 0.45                    # attenuation from 0.5; errors in variables is required
    assert noisy["support_rate_under_truth"] > 0.5                       # and the band would still call it agreement


def test_p16_design_transports_under_truth_and_not_under_the_null():
    r = ds.p16_design_power(sims=60, seed=4)
    assert r["support_rate_under_truth"] >= 0.80
    assert r["support_rate_under_null_no_boundary"] <= 0.05
    assert r["non_crossing_control_false_alarm_rate"] <= 0.10
