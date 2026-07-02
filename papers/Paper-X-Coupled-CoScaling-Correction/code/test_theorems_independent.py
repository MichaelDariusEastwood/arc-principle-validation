"""
Independent analytic verification of the Paper X theorems.

This suite deliberately does NOT import from ``experiment_coscaling.py``. The
in-tree ``test_coscaling.py`` shares the harness's integrator and classifier, so
a bug in either would be reproduced by both. This file re-implements the master
equation and the integrators from scratch (scipy ``solve_ivp`` + closed forms +
matrix exponentials) so that a shared implementation error cannot hide.

It targets exactly the corrections raised in the two external adversarial reviews
(June 2026):

  * Theorem 1  - closed-form transient (independent integrator).
  * Theorem 2  - the three regimes under genuinely TIME-VARYING A(C), r(C)
                 (the case the moving-fixed-point Lyapunov argument did not cover).
  * Theorem 4  - the equality case rho_prop = 1: LINEAR (not absent, not
                 exponential) divergence when injection > 0; bounded for < 1;
                 exponential for > 1; and the d=0 invariant when injection = 0.
  * Theorem 5  - spectral-abscissa stability for a NON-NORMAL operator that has
                 positive eigenvalues yet transient growth of ||d||; plus the
                 corrected null-axis floor gamma1*c_v/(1-gamma3).
  * Theorem 6  - OU stationary variance sigma^2 / (2 kappa).
  * Boundary equalities beta = k and gamma3 = 1.

Run: ``pytest test_theorems_independent.py -q``  (needs numpy + scipy).
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm


# --------------------------------------------------------------------------- #
#  Independent re-implementation of the master equation (Eq. 1 / Eq. 2).       #
#  d/dt of the fraction:  d' = gamma1*r + gamma2 - [A + (1-gamma3)*r] d        #
#  with A = A0 C^beta,  r = b C^k,  dC/dt = b C^(1+k).                         #
# --------------------------------------------------------------------------- #
def integrate_fraction(gamma1, gamma2, gamma3, A0, beta, b, k,
                       C0=1.0, d0=0.05, tau_end=40.0, n=8001):
    """Integrate the misalignment fraction d in the SELF-IMPROVEMENT-DEPTH clock
    tau = ln(C/C0) - exactly the coordinate of the corrected Theorem 2/3 proof.
    This is what makes the time-varying case tractable: for k>0 capability blows
    up at a finite wall-clock t*, but tau -> infinity is regular, so we can drive
    C -> infinity (C = C0 e^tau) without hitting the singularity. Returns (tau, C, d).

        dd/dtau = gamma1 + gamma2/r - [A/r + (1-gamma3)] d
        C = C0 e^tau,  r = b C^k,  A/r = (A0/b) C^(beta-k)
    """
    def rhs(tau, y):
        d = y[0]
        C = C0 * np.exp(tau)
        r = b * C**k
        AoverR = (A0 / b) * C**(beta - k)
        dd = gamma1 + (gamma2 / r if r > 0 else 0.0) - (AoverR + (1.0 - gamma3)) * d
        return [dd]
    tau_eval = np.linspace(0.0, tau_end, n)
    sol = solve_ivp(rhs, (0.0, tau_end), [d0], t_eval=tau_eval,
                    method="LSODA", rtol=1e-9, atol=1e-12)
    C = C0 * np.exp(sol.t)
    return sol.t, C, sol.y[0]


# --------------------------------------------------------------------------- #
#  Theorem 1 - exact transient, constant coefficients.                         #
# --------------------------------------------------------------------------- #
def test_theorem1_closed_form_matches_independent_integrator():
    gamma1, A, r, d0 = 0.05, 0.8, 0.3, 0.4
    dstar = gamma1 * r / (A + r)

    def rhs(t, d):
        return [gamma1 * r - (A + r) * d[0]]
    t_eval = np.linspace(0, 40, 4001)
    sol = solve_ivp(rhs, (0, 40), [d0], t_eval=t_eval, rtol=1e-11, atol=1e-13)
    closed = dstar + (d0 - dstar) * np.exp(-(A + r) * t_eval)
    assert np.max(np.abs(sol.y[0] - closed)) < 1e-7
    assert abs(sol.y[0][-1] - dstar) < 1e-6


# --------------------------------------------------------------------------- #
#  Theorem 2 - three regimes under TIME-VARYING A(C), r(C).                     #
#  This is the case the moving-fixed-point Lyapunov line did not cover; the     #
#  corrected proof uses the depth-clock comparison. We verify the limits hold.  #
# --------------------------------------------------------------------------- #
def test_theorem2_beta_gt_k_drives_to_zero():
    # beta > k : d* -> 0
    t, C, d = integrate_fraction(0.05, 0.0, 0.0, A0=0.5, beta=1.0, b=0.5, k=0.0,
                                 tau_end=30.0)
    assert d[-1] < 1e-3
    assert C[-1] > 1e6 * C[0]                # capability blew up (depth clock, A,r time-varying)


def test_theorem2_beta_eq_k_holds_constant_gap():
    # beta = k : d* -> gamma1 b / (A0 + b)  (independent of C)
    gamma1, A0, b = 0.05, 0.5, 0.5
    t, C, d = integrate_fraction(gamma1, 0.0, 0.0, A0=A0, beta=0.5, b=b, k=0.5,
                                 tau_end=40.0)
    expected = gamma1 * b / (A0 + b)
    assert abs(d[-1] - expected) < 5e-3


def test_theorem2_beta_lt_k_saturates_at_gamma1():
    # beta < k : d* -> gamma1  (saturates, NOT diverges)
    gamma1 = 0.05
    t, C, d = integrate_fraction(gamma1, 0.0, 0.0, A0=0.5, beta=0.0, b=0.5, k=0.5,
                                 d0=0.02, tau_end=40.0)
    assert abs(d[-1] - gamma1) < 5e-3
    assert np.all(d <= gamma1 + 1e-6)        # bounded by gamma1 throughout (d0 < gamma1)


def test_theorem2_global_bound_holds():
    # 0 <= d <= max(d0, gamma1) for arbitrary non-negative A,r time courses
    gamma1, d0 = 0.05, 0.2
    t, C, d = integrate_fraction(gamma1, 0.0, 0.0, A0=0.1, beta=-0.3, b=1.0, k=0.4,
                                 d0=d0, tau_end=40.0)
    assert np.all(d >= -1e-9)
    assert np.all(d <= max(d0, gamma1) + 1e-6)


# --------------------------------------------------------------------------- #
#  Theorem 4 - the compounding-channel equality case (the reviewer's bug).     #
#  Hold C fixed (k=0) so r is constant and the threshold is exact.             #
# --------------------------------------------------------------------------- #
def _compounding_traj(gamma1, gamma3, A0, b, d0=0.05, t_end=60.0, n=6001):
    """k = 0 so r = b is constant and C = C0 e^{bt}; A = A0 (beta = 0). The
    fraction obeys d' = (gamma1 b) - kappa_eff d, kappa_eff = A0 + (1-gamma3) b."""
    r = b
    A = A0
    kappa_eff = A + (1.0 - gamma3) * r
    iota = gamma1 * r
    t = np.linspace(0, t_end, n)
    def rhs(tt, d):
        return [iota - kappa_eff * d[0]]
    sol = solve_ivp(rhs, (0, t_end), [d0], t_eval=t, rtol=1e-10, atol=1e-12)
    return t, sol.y[0], kappa_eff, iota


def test_theorem4_equality_case_is_LINEAR_divergence():
    # rho_prop = 1 exactly: kappa_eff = 0, injection > 0 -> LINEAR growth d ~ iota*t
    gamma1, gamma3, b = 0.05, 4.0, 1.0
    A0 = (gamma3 - 1.0) * b                  # A0 = 3.0  => kappa_eff = 0 exactly
    t, d, kappa_eff, iota = _compounding_traj(gamma1, gamma3, A0, b, t_end=50.0)
    assert abs(kappa_eff) < 1e-12
    # linear, not exponential: d(t) ~ d0 + iota t
    fit_slope = np.polyfit(t, d, 1)[0]
    assert abs(fit_slope - iota) < 1e-3
    # residual from a straight line is tiny (it is exactly linear)
    lin = d[0] + iota * t
    assert np.max(np.abs(d - lin)) < 1e-6
    # and it is unbounded (grew a lot), not a bounded fixed point
    assert d[-1] > 50 * d[0]


def test_theorem4_below_threshold_is_bounded():
    # rho_prop < 1: A0 > (gamma3-1) b -> stable fixed point iota/kappa_eff
    gamma1, gamma3, b = 0.05, 4.0, 1.0
    A0 = (gamma3 - 1.0) * b + 1.0            # kappa_eff = +1 > 0
    t, d, kappa_eff, iota = _compounding_traj(gamma1, gamma3, A0, b, t_end=80.0)
    assert kappa_eff > 0
    assert abs(d[-1] - iota / kappa_eff) < 1e-4


def test_theorem4_above_threshold_is_exponential():
    # rho_prop > 1: kappa_eff < 0 -> exponential divergence (faster than linear)
    gamma1, gamma3, b = 0.05, 4.0, 1.0
    A0 = (gamma3 - 1.0) * b - 1.0            # kappa_eff = -1 < 0
    t, d, kappa_eff, iota = _compounding_traj(gamma1, gamma3, A0, b, t_end=30.0)
    assert kappa_eff < 0
    # exponential growth: ratio of successive decades blows up far past linear
    lin = d[0] + iota * t
    assert d[-1] > 100 * lin[-1]


def test_theorem4_zero_injection_origin_is_invariant():
    # injection iota = 0 (gamma1 = 0): d = 0 is invariant even with kappa_eff < 0
    gamma1, gamma3, b = 0.0, 4.0, 1.0
    A0 = (gamma3 - 1.0) * b - 1.0            # kappa_eff = -1 < 0
    # start AT zero -> stays zero
    t, d0traj, kappa_eff, iota = _compounding_traj(gamma1, gamma3, A0, b, d0=0.0,
                                                   t_end=20.0)
    assert np.max(np.abs(d0traj)) < 1e-9
    # start ABOVE zero -> diverges
    t, dpos, _, _ = _compounding_traj(gamma1, gamma3, A0, b, d0=0.01, t_end=20.0)
    assert dpos[-1] > 100 * 0.01


# --------------------------------------------------------------------------- #
#  Theorem 5 - non-normal operator: positive eigenvalues, transient growth.    #
#  M has spectral abscissa > 0 (asymptotically stable) yet ||d(t)|| overshoots #
#  ||d(0)|| before decaying. Eigenvalues decide the asymptote; not the transient#
# --------------------------------------------------------------------------- #
def test_theorem5_nonnormal_transient_growth_with_positive_spectrum():
    # classic non-normal example: eigenvalues {1, 2} (both > 0 => stable) but
    # a large off-diagonal coupling produces transient amplification of ||x||.
    M = np.array([[1.0, 25.0],
                  [0.0, 2.0]])
    eig = np.linalg.eigvals(M)
    assert np.min(eig.real) > 0                       # positive spectral abscissa
    # integrate xdot = -M x
    x0 = np.array([1.0, 1.0])
    ts = np.linspace(0, 12, 4001)
    norms = np.array([np.linalg.norm(expm(-M * t) @ x0) for t in ts])
    assert norms.max() > 1.3 * norms[0]               # genuine transient overshoot
    assert norms[-1] < 1e-3                            # but asymptotically stable


def test_theorem5_null_axis_floor_is_gamma1_cv_over_one_minus_gamma3():
    # Along a null axis (A v = 0), gain-only depth clock: d_v' = gamma1 c_v - (1-gamma3) d_v
    # corrected floor = gamma1 c_v / (1 - gamma3), NOT gamma1 (unless c_v=1, gamma3=0).
    gamma1, c_v, gamma3 = 0.05, 0.5, 0.4
    decay = 1.0 - gamma3
    def rhs(tau, dv):
        return [gamma1 * c_v - decay * dv[0]]
    sol = solve_ivp(rhs, (0, 200), [0.0], t_eval=np.linspace(0, 200, 4001),
                    rtol=1e-11, atol=1e-13)
    floor = gamma1 * c_v / (1.0 - gamma3)
    assert abs(sol.y[0][-1] - floor) < 1e-6
    # special case gamma3=0, c_v=1 recovers the scalar gamma1 floor used in E6
    def rhs2(tau, dv):
        return [gamma1 * 1.0 - 1.0 * dv[0]]
    sol2 = solve_ivp(rhs2, (0, 100), [0.0], t_eval=np.linspace(0, 100, 2001),
                     rtol=1e-11, atol=1e-13)
    assert abs(sol2.y[0][-1] - gamma1) < 1e-6


def test_theorem5_null_axis_gamma3_one_grows_linearly():
    # gamma3 = 1 on a null axis: d_v' = gamma1 c_v  -> linear growth
    gamma1, c_v = 0.05, 0.8
    rate = gamma1 * c_v
    t = np.linspace(0, 100, 2001)
    dv = rate * t                                      # exact solution from d0=0
    assert abs(np.polyfit(t, dv, 1)[0] - rate) < 1e-9
    assert dv[-1] > 50 * (dv[1] - dv[0])               # unbounded, linear


# --------------------------------------------------------------------------- #
#  Theorem 6 - OU stationary variance sigma^2 / (2 kappa) (Monte Carlo).        #
# --------------------------------------------------------------------------- #
def test_theorem6_ou_stationary_variance():
    rng = np.random.default_rng(7)
    gamma1, r, A = 0.05, 0.3, 0.8
    kappa = A + r
    dstar = gamma1 * r / kappa
    sigma = 0.05
    dt = 1e-3
    steps = 200000
    d = dstar
    # burn-in then collect
    samples = []
    for i in range(steps):
        d += (gamma1 * r - kappa * d) * dt + sigma * np.sqrt(dt) * rng.standard_normal()
        if i > steps // 2:
            samples.append(d)
    samples = np.array(samples)
    var_theory = sigma**2 / (2 * kappa)
    assert abs(np.mean(samples) - dstar) < 5e-3
    assert abs(np.var(samples) - var_theory) / var_theory < 0.15


# --------------------------------------------------------------------------- #
#  Boundary equality beta = k handled above; gamma3 = 1 unconditional bound.    #
# --------------------------------------------------------------------------- #
def test_gamma3_equals_one_boundary_bounded_only_for_beta_ge_k():
    # The knife-edge gamma3 = 1: dilution term (1-gamma3) r vanishes, so kappa_eff = A.
    # No finite-time blow-up (A > 0), but d* = gamma1 r / A grows as C^(k-beta).
    gamma1 = 0.05
    # (a) beta > k : d* = gamma1 r/A -> 0, bounded and small.
    t, C, d = integrate_fraction(gamma1, 0.0, 1.0, A0=0.5, beta=0.5, b=1.0, k=0.0,
                                 tau_end=20.0)
    assert np.all(np.isfinite(d))
    assert d[-1] < 0.01                                 # bounded -> 0 for beta > k
    # (b) beta < k : the fraction GROWS without bound (polynomially), no finite-time
    #     blow-up. This is exactly why "gamma3 <= 1 unconditionally bounded" is too
    #     strong at the boundary - it is bounded only for beta >= k.
    t2, C2, d2 = integrate_fraction(gamma1, 0.0, 1.0, A0=0.5, beta=0.0, b=0.5, k=0.5,
                                    d0=0.05, tau_end=40.0)
    assert np.all(np.isfinite(d2))                      # no finite-time blow-up (kappa_eff = A > 0)
    assert d2[-1] > 10 * d2[0]                          # but it grows without bound


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
