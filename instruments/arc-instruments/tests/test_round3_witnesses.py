"""Witnesses from the implementation brief of 5 September 2026, pinned as tests."""
import numpy as np

from arc_instruments import balance as bl
from arc_instruments import blinding as bd
from arc_instruments import burden_identification as bi
from arc_instruments import identification_adversaries as ia
from arc_instruments import resources as rs
from arc_instruments import verdicts as V


def test_resource_axis_witness_manufactures_no_advantage_on_the_paired_schedule():
    w = rs.resource_axis_witness()
    assert abs(w["naive_contrast"] - 0.6) < 1e-12
    assert abs(w["matched_schedule_contrast"]) < 1e-12
    assert abs(w["breadth_slope_on_paired_schedule"] - 1.2) < 1e-9
    assert w["max_paired_capability_difference"] < 1e-9


def test_delivered_slope_favours_the_worse_corrector_when_supply_limited():
    w = rs.delivered_vs_available_witness()
    assert abs(w["good_delivered_slope"]) < 1e-9 and abs(w["weak_delivered_slope"] - 0.5) < 1e-9
    assert w["good_available_slope"] > w["weak_available_slope"] and w["supply_limited_for_good_corrector"]


def test_generalised_frontier_and_the_compensating_example():
    assert abs(bl.generalised_frontier(0.5) - 2.0) < 1e-12
    assert abs(bl.generalised_frontier(0.5, mu_B=0.2, eta=0.4) - 2.0) < 1e-12   # same number, different mechanism
    try:
        bl.generalised_frontier(0.5, mu_B=-0.6)
    except ValueError:
        pass
    else:
        raise AssertionError("a non-positive denominator must not return a ceiling")


def test_local_curvature_term_changes_the_sign():
    # C = exp(0.2 R) at R = 5: local elasticity one, curvature term one, chi one half.
    assert abs(bl.local_balance(1.0, 1.0, 0.5) + 0.5) < 1e-12
    assert abs(bl.local_balance(1.0, 0.0, 0.5) - 0.5) < 1e-12


def test_queue_witness_identical_exponents_different_event_times():
    cases = bl.queue_witness()
    assert [c["first_backlog_event_R"] for c in cases] == [6, 101, None]
    assert [c["ratio_crossing_depth"] for c in cases] == [1.0, 100.0, 1000000.0]
    assert cases[2]["backlog_at_end"] == 0.0


def test_burden_identification_single_path_rank_two_crossed_rank_three():
    w = bi.witness(noise=0.05, seed=3)
    assert w["single_path_rank"] == 2 and w["crossed_rank"] == 3
    assert abs(w["derivative_truth"]["gain_rate_coefficient"] - 1.0) < 0.03 and abs(w["derivative_truth"]["capability_coefficient"]) < 0.03
    assert abs(w["stock_truth"]["gain_rate_coefficient"]) < 0.03 and abs(w["stock_truth"]["capability_coefficient"] - 0.7) < 0.03


def test_p5_acceptance_recovers_declares_and_rejects():
    a = ia.acceptance(beta=0.5, theta=0.8, lam=0.3, reps=60, seed=11)
    assert abs(a["favourable_crossed_beta_mean"] - 0.5) < 0.02
    assert abs(a["retention_only_slope_mean"] - 0.8) < 0.02          # theta, misreadable as the coupling
    assert abs(a["confounded_crossed_beta_mean"] - 0.8) < 0.02       # beta plus lambda under the nuisance-rate confound
    assert a["confounded_coverage_of_structural_beta"] < 0.05


def test_allocation_confounding_and_genuine_effect_classifier():
    fp = bd.allocation_confounding(reps=400, seed=5)
    assert fp["confounded"] > 0.95
    assert fp["balanced_adjusted"] < 0.10
    assert abs(bd.genuine_effect_classifier_accuracy(0.4, 0.4) - 0.6915) < 0.001


def test_conjunction_table():
    S, R, I, N = V.SUPPORTED, V.REFUTED, V.INCONCLUSIVE, V.NOT_EVALUABLE
    assert V.conjunction([S, S]) == S
    assert V.conjunction([S, R]) == R and V.conjunction([N, R]) == R and V.conjunction([I, R]) == R
    assert V.conjunction([S, I]) == I and V.conjunction([I, I]) == I
    assert V.conjunction([S, N]) == N and V.conjunction([I, N]) == N and V.conjunction([N, N]) == N
    assert V.conjunction([S, S], shared_admissibility=False) == N
    assert V.conjunction([S, V.AWAITING_REPLICATION]) == N
