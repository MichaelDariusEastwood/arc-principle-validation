"""The counterexample intervals the reviews named, pinned as tests. If the registration's prose gives a
different verdict for any of these, the prose is wrong and this file says which sentence."""
from arc_instruments import verdicts as v


def test_p8_region_dominates_short_form():
    assert v.p8_departure((0.35, 0.45)) == v.INCONCLUSIVE          # straddles 0.40; "wholly below 0.50 refutes" is the contradiction
    assert v.p8_departure((0.48, 0.52)) == v.EQUIVALENT_TO_NULL
    assert v.p8_departure((0.61, 0.90)) == v.ABOVE_NULL
    assert v.p8_departure((0.10, 0.39)) == v.BELOW_NULL
    assert v.p8_departure((0.60, 0.60)) == v.NOT_EVALUABLE          # a zero-width interval is an equality, not a measurement
    assert v.p8_departure((0.45, 0.60)) == v.INCONCLUSIVE           # a boundary contact is not a clearance
    assert v.p8_departure((0.55, 0.70)) == v.INCONCLUSIVE
    assert v.p8_clears_unity((1.05, 1.30)) == v.CLEARS_UNITY
    assert v.p8_clears_unity((0.95, 1.05)) == v.INCONCLUSIVE
    assert v.p8_clears_unity((0.40, 0.60)) == v.DOES_NOT_CLEAR_UNITY


def test_p22_margin_dominates_reversal_sentence():
    assert v.p22_typed_ordering((-0.15, -0.05)) == v.INCONCLUSIVE    # negative but straddles -delta
    assert v.p22_typed_ordering((-0.30, -0.12)) == v.JUDGED_HIGHER
    assert v.p22_typed_ordering((0.12, 0.30)) == v.CHECKABLE_HIGHER
    assert v.p22_typed_ordering((-0.05, 0.05)) == v.PRACTICALLY_EQUAL
    assert v.p22_typed_ordering((-0.10, 0.10)) == v.INCONCLUSIVE     # contact with both edges: no clearance
    assert v.p22_typed_ordering((-0.09, 0.09)) == v.PRACTICALLY_EQUAL


def test_p19_joint_cell():
    assert v.p19_zero_scaling((0.02, 0.08)) == v.EQUIVALENT_TO_ZERO
    assert v.p19_joint((0.02, 0.08)) == v.SUPPORTED
    assert v.p19_zero_scaling((0.15, 0.30)) == v.MATERIALLY_POSITIVE
    assert v.p19_keep_pace((0.15, 0.30)) == v.WHOLLY_BELOW_ONE_HALF
    assert v.p19_joint((0.15, 0.30)) == v.REFUTES_WORDING_NOT_SUPPORT
    assert v.p19_joint((0.55, 0.80)) == v.REFUTED
    assert v.p19_joint((-0.05, 0.15)) == v.INCONCLUSIVE


def test_p17_two_sided():
    assert v.p17_pairwise((-0.10, 0.10)) == v.INDEPENDENT
    assert v.p17_pairwise((0.25, 0.60)) == v.CORRELATED
    assert v.p17_pairwise((-0.50, -0.25)) == v.ANTI_CORRELATED
    assert v.p17_pairwise((-0.30, 0.30)) == v.INSUFFICIENT_PRECISION
    assert v.p17_panel([v.INDEPENDENT, v.INDEPENDENT]) == v.SUPPORTED
    assert v.p17_panel([v.INDEPENDENT, v.ANTI_CORRELATED]) == v.REFUTED
    assert v.p17_panel([v.INDEPENDENT, v.INSUFFICIENT_PRECISION]) == v.INSUFFICIENT_PRECISION


def test_p10_and_p14_consequences_do_not_run_backwards():
    assert v.p10_material_fraction((0.12, 0.30)) == v.SUPPORTED
    assert v.p10_material_fraction((0.01, 0.05)) == v.REFUTED
    assert v.p10_material_fraction((0.05, 0.15)) == v.INCONCLUSIVE
    assert v.p14_blinded_shrink((-0.30, -0.10)) == v.SUPPORTED
    assert v.p14_blinded_shrink((0.00, 0.20)) == v.INCONCLUSIVE      # contact with zero is not a clearance
    assert v.p14_blinded_shrink((0.01, 0.20)) == v.REFUTED           # wholly above zero
    assert v.p14_blinded_shrink((-0.10, 0.10)) == v.INCONCLUSIVE


def test_p3_four_outcomes():
    assert v.p3_frontier(None, True, False, False, False) == v.NOT_EVALUABLE
    assert v.p3_frontier(None, True, True, True, True) == v.REFUTED
    assert v.p3_frontier((-0.05, 0.05), True, True, True, False) == v.SUPPORTED
    assert v.p3_frontier((-0.05, 0.05), True, False, True, False) == v.CONSISTENT_NOT_SUPPORTIVE
    assert v.p3_frontier((-0.30, 0.30), True, True, True, False) == v.CONSISTENT_NOT_SUPPORTIVE


def test_p20_gated_on_calibrated_power():
    adv = {"reciprocal": (0.12, 0.30), "constant": (0.15, 0.40), "burden": (0.11, 0.25), "smooth": (0.14, 0.33)}
    assert v.p20_form([0.45, 0.50, 0.55], adv, margin=0.10, discriminating_power=0.45) == v.NOT_DISCRIMINATING
    assert v.p20_form([0.35, 0.50, 0.65], adv, margin=0.10, discriminating_power=0.9) == v.SUPPORTED
    adv_tie = dict(adv); adv_tie["burden"] = (-0.02, 0.03)
    assert v.p20_form([0.35, 0.50, 0.65], adv_tie, margin=0.10, discriminating_power=0.9) == v.INCONCLUSIVE
    adv_loss = dict(adv); adv_loss["smooth"] = (-0.40, -0.10)
    assert v.p20_form([0.35, 0.50, 0.65], adv_loss, margin=0.10, discriminating_power=0.9) == v.REFUTED


def test_p16_order_of_events_and_censoring():
    assert v.p16_prohibition(True, True, None, True) == v.SUPPORTED
    assert v.p16_prohibition(True, None, True, True) == v.REFUTED
    assert v.p16_prohibition(True, True, None, False) == v.INCONCLUSIVE
    assert v.p16_prohibition(True, True, None, True, censored=True) == v.INCONCLUSIVE
    assert v.p16_prohibition(False, None, None, True) == v.INCONCLUSIVE


def test_p1_form_and_recursion():
    cells = ["power"] * 5 + ["saturating"] * 2
    assert v.p1_form(cells, 0.05, 0.10) == v.SUPPORTED
    assert v.p1_form(cells, 0.15, 0.10) == v.NOT_DISCRIMINATING
    assert v.p1_form(["saturating"] * 4 + ["power"] * 3, 0.05, 0.10) == v.REFUTED
    assert v.p1_recursion((0.15, 0.30), 0.10) == v.SUPPORTED
    assert v.p1_recursion((-0.10, 0.05), 0.10) == v.REFUTED
    assert v.p1_recursion((0.05, 0.15), 0.10) == v.INCONCLUSIVE


def test_p21_level_difference_scored_against():
    assert v.p21_interaction((0.12, 0.30), 0.10, True, high_depth_effect=(0.05, 0.30)) == v.SUPPORTED
    assert v.p21_interaction((0.12, 0.30), 0.10, True) == v.INCONCLUSIVE
    assert v.p21_interaction((-0.05, 0.05), 0.10, True) == v.REFUTED
    assert v.p21_interaction((-0.05, 0.05), 0.10, False) == v.INSTRUMENT_FAILED
    assert v.p21_interaction((0.12, 0.30), 0.10, True, level_difference_only=True) == v.REFUTED


def test_p12_validity_does_not_require_benefit():
    assert v.p12_build_order((-0.05, 0.05), True) == v.REFUTED
    assert v.p12_build_order((-0.05, 0.05), False) == v.INSTRUMENT_FAILED
    assert v.p12_build_order((0.12, 0.30), True) == v.SUPPORTED


def test_p13_two_axes():
    assert v.p13_components((0.05, 0.20), v.SUPPORTED)["cell"] == v.SUPPORTED_WITH_MECHANISM
    assert v.p13_components((0.05, 0.20), v.INCONCLUSIVE)["cell"] == v.PERFORMANCE_SUPPORTED_MECHANISM_UNRESOLVED
    assert v.p13_conjunction((-0.20, -0.01), v.SUPPORTED) == v.REFUTED
    assert v.p13_conjunction((-0.20, 0.00), v.SUPPORTED) == v.INCONCLUSIVE
    assert v.p13_conjunction((-0.05, 0.10), v.SUPPORTED) == v.INCONCLUSIVE


def test_p15_p11_p5_p6_p4():
    assert v.p15_decay((0.20, 0.45)) == v.SUPPORTED
    assert v.p15_decay((0.50, 0.80)) == v.INCONCLUSIVE
    assert v.p15_decay((0.51, 0.80)) == v.REFUTED
    assert v.p11_pair((-0.05, 0.05), 0.10) == v.IDENTICAL_WITHIN_MARGIN
    assert v.p11_pair((0.15, 0.40), 0.10) == v.DISTINCT
    assert v.p5_agreement((-0.05, 0.05)) == v.SUPPORTED
    assert v.p5_agreement((0.15, 0.30)) == v.REFUTED
    assert v.p6_survey([(0.30, 0.45), (0.40, 0.49)]) == v.NO_COUNTEREXAMPLE
    assert v.p6_survey([(0.30, 0.45), (0.52, 0.70)]) == v.AWAITING_REPLICATION
    assert v.p6_survey([(0.30, 0.45), (0.52, 0.70)], replicated=True) == v.REFUTED
    assert v.p6_survey([(0.30, 0.45), (0.35, 0.65)]) == v.INCONCLUSIVE
    assert v.p4_regime((0.55, 0.65), (0.30, 0.45)) == v.CORRECTION_ADVANTAGE
    assert v.p4_regime((0.30, 0.40), (0.50, 0.60)) == v.BURDEN_ADVANTAGE


def test_p2_shape_endpoints_never_interior():
    betas = [0.1, 0.3, 0.5, 0.7, 0.9]
    assert v.p2_shape(betas, [1.0, 2.0, 3.0, 2.0, 1.0], [0.1] * 5) == v.INTERIOR_MAXIMUM
    assert v.p2_shape(betas, [1.0, 2.0, 3.0, 4.0, 5.0], [0.1] * 5) == v.MONOTONE
    assert v.p2_shape(betas, [2.0, 2.05, 2.0, 1.95, 2.0], [0.5] * 5) == v.UNRESOLVED     # imprecise cells are never flat
    assert v.p2_shape(betas, [2.0, 2.02, 2.0, 1.99, 2.0], [0.01] * 5, flat_margin=0.10) == v.FLAT
    assert v.p2_shape(betas, [1.0, 3.0, 1.0, 3.0, 1.0], [0.1] * 5) == v.MULTIMODAL


def test_scoring_table_covers_every_scored_proposition():
    ids = {row["proposition"] for row in v.scoring_table()}
    assert ids == {"P%d" % i for i in range(1, 23)} | {"ratio"}


def test_wilson_interval_bounds():
    lo, hi = v.wilson_interval(9, 10)
    assert 0.55 < lo < 0.75 and 0.95 < hi <= 1.0
    assert v.wilson_interval(0, 0) == (0.0, 1.0)
