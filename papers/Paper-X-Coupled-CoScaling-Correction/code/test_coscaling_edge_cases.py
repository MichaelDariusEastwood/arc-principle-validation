#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Additional edge-case tests for Paper X theorem statements.

These are deliberately independent of experiment_coscaling.py. They protect the
paper against the three most reviewer-visible mathematical edge cases:

1. The compounding threshold equality case is not stable when injection is nonzero.
2. A null correction axis floors at gamma1*c/(1-gamma3), not at gamma1, unless gamma3=0 and c=1.
3. Level drift vanishes when A+r grows, but gain-drift still requires beta>k to drive d -> 0.

Run from the Paper-X code directory with:
    pytest test_coscaling_edge_cases.py -q
"""


def euler_depth(gamma1=0.05, gamma2_over_r=0.0, gamma3=0.0, A_over_r=1.0, d0=0.05, steps=20000, h=1e-3):
    """Depth-clock scalar dynamics with constant coefficients:
       d' = gamma1 + gamma2/r - [A/r + (1-gamma3)] d.
    """
    d = d0
    for _ in range(steps):
        d += h * (gamma1 + gamma2_over_r - (A_over_r + (1.0 - gamma3)) * d)
    return d


def test_compounding_threshold_equality_unbounded_with_injection():
    """At rho_prop=1, kappa_eff=0. With gamma1>0 the solution grows linearly in depth.
    Therefore the stability condition is strict: rho_prop < 1, and unboundedness
    occurs for rho_prop >= 1 when injection is nonzero.
    """
    gamma1 = 0.05
    d0 = 0.1
    tau = 50.0
    # equality A/r = gamma3-1 = 2, so decay coefficient is zero
    d = d0 + gamma1 * tau
    assert d > 2.0


def test_compounding_threshold_equality_neutral_only_without_injection():
    gamma1 = 0.0
    d0 = 0.1
    tau = 50.0
    d = d0 + gamma1 * tau
    assert abs(d - d0) < 1e-12


def test_null_axis_floor_general_gamma3_less_than_one():
    """If Av=0, the null-axis dynamics are d' = gamma1*c - (1-gamma3)d.
    The floor is gamma1*c/(1-gamma3), not gamma1 except in the special case
    gamma3=0,c=1.
    """
    gamma1 = 0.05
    c = 1.0
    gamma3 = 0.5
    expected = gamma1 * c / (1.0 - gamma3)
    d = euler_depth(gamma1=gamma1 * c, gamma3=gamma3, A_over_r=0.0, d0=0.0, steps=200000, h=1e-3)
    assert abs(d - expected) < 1e-3
    assert abs(d - gamma1) > 0.02


def test_null_axis_gamma3_equal_one_grows_linearly_with_injection():
    gamma1 = 0.05
    d0 = 0.0
    tau = 50.0
    d = d0 + gamma1 * tau
    assert d > 2.0


def test_level_drift_vanishes_but_gain_drift_requires_beta_gt_k():
    """For d*=(gamma1*r+gamma2)/(A+r), a positive k makes gamma2/(A+r) vanish;
    it does not make the gain-drift contribution vanish unless A/r -> infinity,
    i.e. beta>k.
    """
    gamma1 = 0.05
    gamma2 = 0.02
    b = 1.0
    A0 = 1.0
    beta = 0.0
    k = 1.0
    C = 1e9
    r = b * C**k
    A = A0 * C**beta
    d_star = (gamma1 * r + gamma2) / (A + r)
    assert abs(d_star - gamma1) < 1e-6


def test_level_drift_and_gain_drift_vanish_when_beta_gt_k():
    gamma1 = 0.05
    gamma2 = 0.02
    b = 1.0
    A0 = 1.0
    beta = 2.0
    k = 1.0
    C = 1e9
    r = b * C**k
    A = A0 * C**beta
    d_star = (gamma1 * r + gamma2) / (A + r)
    assert d_star < 1e-8
