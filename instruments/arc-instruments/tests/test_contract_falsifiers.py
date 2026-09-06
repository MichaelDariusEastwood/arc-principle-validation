"""Twenty contract-derived falsifiers (5 September 2026): each probe is a way the engine
could silently disagree with the frozen registration text. They pass against the engine as carried on
main; the twenty-first row of the scoring table is the labelled ratio extension, which is why the
table has twenty-three rows for twenty-two propositions."""
import pytest

from arc_instruments import blinding as bl, precision as pr, verdicts as v


def test_01_p1_eight_inconclusive_cells_are_not_refuted():
    assert v.p1_form([v.INCONCLUSIVE] * 8, 0.02, 0.05) != v.REFUTED


def test_02_p1_unknown_cell_label_is_refused():
    with pytest.raises(ValueError):
        v.p1_form(["POWER WINS?"] * 3, 0.02, 0.05)


def test_03_p1_demonstrated_false_selection_above_the_registered_maximum_is_not_discriminating():
    assert v.p1_form([v.SUPPORTED] * 8, 0.08, 0.05) == v.NOT_DISCRIMINATING


def test_04_p13_failed_dependence_is_never_supported():
    assert v.p13_conjunction((0.10, 0.30), v.REFUTED) != v.SUPPORTED


def test_05_p13_unresolved_dependence_is_never_supported():
    assert v.p13_conjunction((0.10, 0.30), v.INCONCLUSIVE) != v.SUPPORTED


def test_06_p13_missing_dependence_makes_the_whole_not_evaluable():
    assert v.p13_conjunction((0.10, 0.30), v.NOT_EVALUABLE) == v.NOT_EVALUABLE


def test_07_p21_uncalibrated_positive_direction_is_instrument_failed():
    assert v.p21_interaction((0.20, 0.40), 0.10, sensitivity_demonstrated=False) == v.INSTRUMENT_FAILED


def test_08_p21_uncalibrated_negative_direction_is_instrument_failed():
    assert v.p21_interaction((-0.40, -0.20), 0.10, sensitivity_demonstrated=False) == v.INSTRUMENT_FAILED


def test_09_p21_calibrated_without_a_high_depth_effect_is_not_supported():
    assert v.p21_interaction((0.20, 0.40), 0.10, sensitivity_demonstrated=True) != v.SUPPORTED


def test_10_p21_calibrated_with_a_positive_high_depth_effect_is_supported():
    assert v.p21_interaction((0.20, 0.40), 0.10, sensitivity_demonstrated=True, high_depth_effect=(0.10, 0.30)) == v.SUPPORTED


def test_11_p2_flat_values_with_huge_errors_are_not_flat():
    assert v.p2_shape([0.1, 0.3, 0.5, 0.7, 0.9], [1.0] * 5, [5.0] * 5, flat_margin=0.05) != v.FLAT


def test_12_p6_default_is_awaiting_fresh_data_replication():
    assert v.p6_survey([(0.60, 0.70)]) == v.AWAITING_REPLICATION


def test_13_p6_empty_roster_is_no_survey():
    assert v.p6_survey([]) == v.NO_SURVEY


def test_14_scoring_table_is_twenty_two_propositions_plus_the_labelled_ratio_row():
    rows = v.scoring_table()
    assert len(rows) == 23
    assert sum(1 for r in rows if r["proposition"].startswith("P")) == 22
    assert rows[-1]["proposition"] == "ratio"


def test_15_james_index_is_one_half_for_the_allocation_exploiting_guesser():
    actual = ["A"] * 90 + ["B"] * 10
    assert abs(bl.james_index(actual, ["A"] * 100) - 0.50) < 1e-9


def test_16_p8_zero_width_is_not_evaluable():
    assert v.p8_departure((0.60, 0.60)) == v.NOT_EVALUABLE


def test_17_p8_boundary_contact_is_inconclusive():
    assert v.p8_departure((0.40, 0.55)) == v.INCONCLUSIVE


def test_18_p8_the_review_counterexample_is_inconclusive_not_refuted():
    assert v.p8_departure((0.35, 0.45)) == v.INCONCLUSIVE


def test_19_sign_power_is_never_labelled_decision_power():
    assert "decision power" not in (pr.sign_power.__doc__ or "").split("must never be called")[0]
    assert hasattr(pr, "interval_clearance_power")


def test_20_interval_clearance_power_is_far_below_sign_power_at_the_registered_precision():
    out = pr.interval_clearance_power(0.10, 0.02, 0.07 / 1.96, gamma=0.5)
    assert out["interval_clearance_power"] < out["point_sign_probability"] - 0.30
