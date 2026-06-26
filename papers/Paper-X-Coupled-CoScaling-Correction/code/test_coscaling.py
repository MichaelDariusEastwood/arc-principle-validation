#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Internal-consistency + integrator test suite for the Coupled Co-Scaling Law (Paper X).

Each test encodes a theorem or a derivation-consistency condition as a hard
assertion. `pytest` returns 0 iff the closed-form predictions are correctly
derived from the model and correctly reproduced numerically. These are
derivation-consistency and integrator-accuracy checks -- they verify the code
matches the maths. They do NOT test whether the model describes any real AI
system; that is the open empirical problem stated in the paper's Limitations.

Run:  pytest test_coscaling.py -q
"""
import numpy as np
from experiment_coscaling import P, integrate_depth, asymptote, log_slope, threshold_crossing


# ---- Theorem 1: exact transient -------------------------------------------- #
def test_theorem1_exact_transient():
    """Integrator reproduces d(t)=d*+(d0-d*)e^{-(A+r)t} (closed form)."""
    g1, A0, b, d0 = 0.05, 0.08, 0.5, 0.05
    r = b
    dstar = g1 * r / (A0 + r)
    p = P(gamma1=g1, A0=A0, beta=0.0, k=0.0, b=b, d0=d0)
    tau, C, d = integrate_depth(p, tau_max=r * 30, n=2000)
    t = tau / r
    exact = dstar + (d0 - dstar) * np.exp(-(A0 + r) * t)
    assert np.max(np.abs(d - exact)) < 1e-4


# ---- Theorem 2 / F3: corrected co-scaling regimes -------------------------- #
def test_theorem2_beta_positive_converges():
    p = P(gamma1=0.05, A0=0.08, beta=0.5, k=0.0, b=0.5)
    tau, C, d = integrate_depth(p, tau_max=16)
    assert asymptote(tau, d) < 8e-3            # d* -> 0

def test_theorem2_beta_zero_permanent_gap():
    g1, A0, b = 0.05, 0.08, 0.5
    p = P(gamma1=g1, A0=A0, beta=0.0, k=0.0, b=b)
    tau, C, d = integrate_depth(p, tau_max=16)
    assert abs(asymptote(tau, d) - g1 * b / (A0 + b)) < 5e-3

def test_theorem2_beta_negative_SATURATES_not_diverges():
    """The key correction to the v3 draft: beta<0 saturates at gamma1, it does
    NOT diverge to infinity."""
    g1 = 0.05
    p = P(gamma1=g1, A0=0.08, beta=-0.5, k=0.0, b=0.5)
    tau, C, d = integrate_depth(p, tau_max=20)
    a = asymptote(tau, d)
    assert a < 1.0                             # bounded (would be inf if v3 were right)
    assert abs(a - g1) < 1e-2                  # saturates at the drift coefficient


# ---- Theorem 3 / F3': hard-takeoff boundary at beta=k ---------------------- #
def test_theorem3_boundary_at_beta_equals_k():
    """Under accelerating growth the boundary is at beta=k, not beta=0."""
    g1, A0, b = 0.05, 0.08, 0.02
    # beta=0.75 is STABLE when k=0.5 (beta>k) ...
    p1 = P(gamma1=g1, A0=A0, beta=0.75, k=0.5, b=b)
    t1, C1, d1 = integrate_depth(p1, tau_max=14)
    assert asymptote(t1, d1) < 8e-3
    # ... but the SAME beta=0.75 SATURATES when k=1.0 (beta<k), bounded at ~gamma1
    p2 = P(gamma1=g1, A0=A0, beta=0.75, k=1.0, b=b)
    t2, C2, d2 = integrate_depth(p2, tau_max=14)
    a2 = asymptote(t2, d2)
    assert a2 > 0.02 and a2 < 1.0              # saturates, bounded -- not convergent, not infinite

def test_theorem3_finite_time_singularity_controlled():
    """d stays controlled through a genuine finite-time singularity (k=1)."""
    p = P(gamma1=0.05, A0=0.08, beta=1.5, k=1.0, b=0.02)
    tau, C, d = integrate_depth(p, tau_max=12)
    assert C[-1] > 1e4                          # capability has exploded
    assert np.max(d) < 0.06 and d[-1] < d[0]    # misalignment never lost control


# ---- Theorem 4 / P5: compounding threshold (true divergence) --------------- #
def test_theorem4_compounding_threshold():
    """With gamma3>1, divergence iff A0<(gamma3-1)b; threshold is sharp."""
    b, g3 = 1.0, 4.0
    thr = (g3 - 1.0) * b
    A0s = np.linspace(1.0, 6.0, 60)
    vals = []
    for A0 in A0s:
        p = P(gamma1=0.05, gamma3=g3, A0=A0, beta=0.0, k=0.0, b=b)
        tau, C, d = integrate_depth(p, tau_max=40)
        vals.append(min(asymptote(tau, d), 10.0))
    meas = threshold_crossing(A0s, vals, level=5.0)
    assert abs(meas - thr) < 0.25

def test_theorem4_gamma3_below_one_is_bounded():
    """gamma3<=1 cannot overcome dilution: always bounded, never diverges."""
    p = P(gamma1=0.05, gamma3=0.9, A0=0.01, beta=-0.5, k=0.0, b=1.0)
    tau, C, d = integrate_depth(p, tau_max=30)
    assert asymptote(tau, d) < 1.0


# ---- P8 / F4: power-law suppression signature ------------------------------ #
def test_p8_suppression_slope():
    """log d* linear in log C with slope -(beta-k)."""
    for beta in (0.5, 1.0):
        p = P(gamma1=0.05, A0=0.08, beta=beta, k=0.0, b=0.3)
        tau, C, d = integrate_depth(p, tau_max=18)
        assert abs(log_slope(tau, C, d) - (-beta)) < 0.08


# ---- Theorem 5 / F6: vector spectral threshold ----------------------------- #
def test_theorem5_blind_axis_persists():
    """Co-scaling monitored axis -> 0; null (unmonitored) axis floors at gamma1."""
    g1 = 0.05
    mon = P(gamma1=g1, A0=0.30, beta=1.0, k=0.0, b=1.0)
    blind = P(gamma1=g1, A0=0.0, beta=0.0, k=0.0, b=1.0)
    tm, Cm, dm = integrate_depth(mon, tau_max=14)
    tb, Cb, db = integrate_depth(blind, tau_max=14)
    a_mon, a_blind = asymptote(tm, dm), asymptote(tb, db)
    assert a_mon < 5e-3
    assert abs(a_blind - g1) < 5e-3
    assert a_blind > 5 * a_mon


# ---- P2 / F2: speed-invariance of the verdict ------------------------------ #
def test_p2_speed_does_not_decide():
    """Coupling decides; raw speed does not. Decoupled diverges at BOTH speeds;
    coupled stays bounded at BOTH speeds."""
    g3 = 3.0
    def fate(ratio, b):
        p = P(gamma1=0.05, gamma3=g3, A0=ratio * b, beta=0.0, k=0.0, b=b)
        tau, C, d = integrate_depth(p, tau_max=12)
        return asymptote(tau, d)
    # decoupled (ratio<2) diverges regardless of speed
    assert fate(1.0, 0.5) > 1.0 and fate(1.0, 5.0) > 1.0
    # coupled (ratio>2) bounded regardless of speed
    assert fate(3.0, 0.5) < 1.0 and fate(3.0, 5.0) < 1.0


# ---- §3.6 / F5: residual drift at rest ------------------------------------- #
def test_f5_residual_drift_at_rest():
    """A frozen (r=0) capable system relaxes to d* = gamma2/A0 when gamma2>0
    (pausing growth is not correction), and to 0 when gamma2=0."""
    A0, g2, dt = 0.1, 0.02, 0.01
    d = 0.05
    for _ in range(50000):                 # 500 time units; fully relaxed
        d += (g2 - A0 * d) * dt
    assert abs(d - g2 / A0) < 1e-3          # residual floor gamma2/A0 > 0
    d = 0.05
    for _ in range(50000):
        d += (0.0 - A0 * d) * dt
    assert d < 1e-3                          # gamma2=0 -> d -> 0
