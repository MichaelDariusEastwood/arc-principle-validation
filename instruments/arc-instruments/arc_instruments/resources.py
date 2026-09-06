"""Resource coordinates for the depth-against-breadth comparison (P1 RECURSION), and delivered against
available correction (capacity).

A slope in log depth and a slope in log sample count are not comparable. The witness from the
implementation brief of 5 September 2026: depth performance C_D(R) = R^1.2 with resource cost
B_D(R) = R^2, and breadth performance C_B(B) = B^0.6. Subtracting the displayed exponents gives an
apparent depth advantage of 0.6; on the paired budget schedule C_B(B_D(R)) = R^1.2 = C_D(R), no advantage
at all. P1's comparator is therefore the breadth curve evaluated on the depth arm's paired budget
schedule, or both arms on one common resource coordinate, and the converted breadth exponent is
(d log C_B / d log B) (d log B_D / d log R).

The second witness is capacity against delivery: a strong corrector offered only three faults delivers
three at every capability level (delivered slope zero) while a weak corrector with capacity 0.5 sqrt(C)
delivers less at every level and shows slope one half. Ranking delivered slopes as correction strength
favours the worse corrector; delivered repairs and available service are reported separately.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def breadth_exponent_on_depth_schedule(cb_exponent: float, depth_cost_exponent: float) -> float:
    """(d log C_B / d log B) times (d log B_D / d log R): the breadth exponent in the depth coordinate."""
    return cb_exponent * depth_cost_exponent


def naive_contrast(depth_exponent: float, cb_exponent: float) -> float:
    """The mismatched-coordinate subtraction, kept only to show what it manufactures."""
    return depth_exponent - cb_exponent


def matched_schedule_contrast(depth_exponent: float, cb_exponent: float, depth_cost_exponent: float) -> float:
    """The contrast P1 scores: the depth exponent minus the breadth exponent on the paired budget schedule."""
    return depth_exponent - breadth_exponent_on_depth_schedule(cb_exponent, depth_cost_exponent)


def paired_budget_curve(R: Sequence[float], depth_cost_exponent: float, cb_exponent: float) -> np.ndarray:
    """C_B evaluated at the depth arm's realised budget B_D(R) = R^cost."""
    R = np.asarray(R, float)
    return (R ** depth_cost_exponent) ** cb_exponent


def resource_axis_witness() -> Dict[str, float]:
    R = np.geomspace(1.0, 32.0, 20)
    depth = R ** 1.2
    breadth = paired_budget_curve(R, 2.0, 0.6)
    return {
        "naive_contrast": naive_contrast(1.2, 0.6),
        "matched_schedule_contrast": matched_schedule_contrast(1.2, 0.6, 2.0),
        "breadth_slope_on_paired_schedule": float(np.polyfit(np.log(R), np.log(breadth), 1)[0]),
        "max_paired_capability_difference": float(np.max(np.abs(depth - breadth))),
    }


def delivered_vs_available_witness() -> Dict[str, object]:
    C = np.array([1.0, 2.0, 4.0, 8.0])
    offered = np.full(4, 3.0)
    good_available = 10.0 * C
    weak_available = 0.5 * np.sqrt(C)
    good_delivered = np.minimum(good_available, offered)
    weak_delivered = np.minimum(weak_available, offered)
    slope = lambda y: float(np.polyfit(np.log(C), np.log(y), 1)[0])
    return {
        "good_delivered_slope": slope(good_delivered), "weak_delivered_slope": slope(weak_delivered),
        "good_available_slope": slope(good_available), "weak_available_slope": slope(weak_available),
        "supply_limited_for_good_corrector": bool(np.all(offered < good_available)),
    }
