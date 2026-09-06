from arc_instruments import precision as pr


def test_exceedance_power_matches_the_audit_figure():
    p = pr.exceedance_power(0.55, 0.04, 0.5)
    assert abs(p - 0.239) < 0.01
    assert pr.exceedance_power(0.70, 0.04, 0.5) > 0.95


def test_form_discriminating_power_is_chance_at_one_half_and_high_below_it():
    at_half = pr.form_discriminating_power([0.5, 0.5, 0.5], gamma_se=0.03, frontier_se=0.3, reps=600)
    assert 0.35 < at_half["discriminating_power"] < 0.65
    below = pr.form_discriminating_power([0.2, 0.3, 0.4], gamma_se=0.03, frontier_se=0.3, reps=600)
    assert below["discriminating_power"] > 0.95
    try:
        pr.form_discriminating_power([0.0, 0.5], 0.03, 0.3, reps=10)
    except ValueError:
        return
    raise AssertionError("cells outside (0, 1) must be refused")
