"""Independent finite-domain regression checks for Paper X's corrected theorems.

Specification: ../THEOREMS.md (19 October 2026). No harness imports. These
checks compare independently integrated equations, exact solutions, quadrature,
and matrix exponentials. They are numerical regression tests, NOT proofs of
infinite-time limits, model validation, or an AI safety certificate.

Run from the Paper X directory: python -m pytest -q code/test_theorems_independent.py
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from scipy.stats import norm


def _solve(rhs, grid, y0, *, method="DOP853", **options):
    """Fail closed on failed, truncated, event-terminated or non-finite solves."""
    grid = np.asarray(grid, dtype=float)
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    if (grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid))
            or not np.all(np.diff(grid) > 0) or not np.all(np.isfinite(y0))):
        raise ValueError("Expected a finite increasing grid and finite initial state")
    sol = solve_ivp(rhs, (grid[0], grid[-1]), y0, t_eval=grid,
                    method=method, rtol=2e-10, atol=2e-13, **options)
    if not sol.success or sol.status != 0:
        raise RuntimeError(f"Integration did not complete: {sol.message}")
    if sol.y.shape != (y0.size, grid.size) or not np.array_equal(sol.t, grid):
        raise RuntimeError("Integration returned an incomplete output grid")
    if not np.all(np.isfinite(sol.y)):
        raise RuntimeError("Integration returned non-finite state values")
    return sol.y


def integrate_fraction(gamma1, gamma2, gamma3, A0, beta, b, k,
                       C0=1.0, d0=0.05, tau_end=20.0, n=801):
    """Independently solve Eq. 2 in log-capability depth, not wall-clock time.

    Finite depth is finite capability. Never claim to integrate to infinity.
    A0,b,C0 > 0; drift coefficients and d0 >= 0. Invalid/overflowing coefficients
    raise rather than silently removing a forcing term or returning partial data.
    """
    values = (gamma1, gamma2, gamma3, A0, beta, b, k, C0, d0, tau_end)
    if not np.all(np.isfinite(values)):
        raise ValueError("All parameters must be finite")
    if min(A0, b, C0, tau_end) <= 0 or min(gamma1, gamma2, gamma3, d0) < 0:
        raise ValueError("Invalid coefficient, initial state or depth")
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 2:
        raise ValueError("n must be an integer >= 2")
    log_C0 = np.log(C0)

    def rhs(tau, y):
        log_C = log_C0 + tau
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            q = np.exp(np.log(A0) - np.log(b) + (beta - k) * log_C)
            level = 0.0 if gamma2 == 0 else np.exp(np.log(gamma2 / b) - k * log_C)
            return [gamma1 + level - (q + 1.0 - gamma3) * y[0]]

    tau = np.linspace(0, tau_end, n)
    d = _solve(rhs, tau, [d0], method="Radau")[0]
    with np.errstate(over="raise", invalid="raise"):
        C = np.exp(log_C0 + tau)
    return tau, C, d


def _affine_exact(t, injection, decay, initial):
    """Exact autonomous scalar solution, including the zero-decay case."""
    t = np.asarray(t)
    if decay == 0:
        return initial + injection * t
    return initial * np.exp(-decay * t) - injection * np.expm1(-decay * t) / decay


# Theorems 1 and 2: the ODE itself, not a fitted classifier, is the test target.
@pytest.mark.parametrize("gamma1,A,r,d0", [
    (0.05, 0.8, 0.3, 0.4), (0.0, 0.8, 0.3, 0.4),
    (0.05, 0.0, 0.3, 0.0), (0.05, 0.8, 0.0, 0.4),
])
def test_theorem1_closed_form_matches_independent_integrator(gamma1, A, r, d0):
    t = np.linspace(0, 25, 501)
    d = _solve(lambda t, x: [gamma1 * r - (A + r) * x[0]], t, [d0])[0]
    assert_allclose(d, _affine_exact(t, gamma1 * r, A + r, d0), rtol=3e-8, atol=2e-11)


@pytest.mark.parametrize("k", [-0.5, 0.0, 0.5, 1.0])
@pytest.mark.parametrize("margin", [-0.5, 0.0, 0.5])
@pytest.mark.parametrize("C0", [0.5, 2.0])
def test_theorem2_three_regimes_in_actual_time_varying_trajectory(k, margin, C0):
    g, A0, b, d0 = 0.05, 0.7, 0.4, 0.2
    tau, C, d = integrate_fraction(g, 0, 0, A0, k + margin, b, k, C0, d0,
                                   tau_end=24, n=401)
    assert d.min() >= -1e-11
    assert d.max() <= max(g, d0) + 1e-11
    if margin == 0:
        expected = _affine_exact(tau, g, 1 + A0 / b, d0)
        assert_allclose(d, expected, rtol=2e-8, atol=2e-11)
    elif margin < 0:
        assert abs(d[-1] - g) < 5e-6
    else:
        # Leading-order tracking is proved in THEOREMS.md; test the relative error
        # while d is still resolvable, not a "less than 0.001" near-zero shortcut.
        leading = (g * b / A0) * C[-1] ** (-margin)
        assert abs(d[-1] / leading - 1) < 1e-4


@pytest.mark.parametrize("margin", [-0.5, 0.0, 0.5])
def test_theorem2_zero_gain_drift_exact_homogeneous_solution(margin):
    tau, C, d = integrate_fraction(0, 0, 0, 0.7, margin, 0.4, 0, d0=0.2,
                                   tau_end=6, n=301)
    q0 = 0.7 / 0.4
    integral_q = q0 * tau if margin == 0 else q0 * np.expm1(margin * tau) / margin
    expected = 0.2 * np.exp(-tau - integral_q)
    assert_allclose(d, expected, rtol=3e-7, atol=3e-12)
    # gamma1=0 is deliberately NOT subjected to the positive-drift necessity iff.


def test_theorem2_bound_for_non_power_law_coefficients():
    t = np.linspace(0, 30, 1001)
    A = lambda t: (1 + np.sin(2 * t)) ** 2
    r = lambda t: 0.1 + (1 + np.cos(3 * t)) ** 2
    for d0 in (0.0, 0.02, 0.2):
        d = _solve(lambda t, x: [0.05 * r(t) - (A(t) + r(t)) * x[0]], t, [d0],
                   max_step=0.03)[0]
        assert d.min() >= -1e-11
        assert d.max() <= max(0.05, d0) + 1e-11


# Theorem 3: integrate ORIGINAL (C,D) in t, separately from Eq. 2 in tau.
@pytest.mark.parametrize("k,beta,b", [(0.5, 1.0, 0.2), (1.0, 1.0, 1.0), (1.0, 0.5, 0.7)])
def test_theorem3_original_master_equation_matches_depth_clock(k, beta, b):
    C0, d0, A0, g1, g2, g3 = 1.3, 0.07, 0.6, 0.05, 0.02, 0.4
    tstar = 1 / (k * b * C0**k)
    t = np.linspace(0, 0.94 * tstar, 401)

    def original(t, state):
        C, D = state
        Cdot = b * C ** (1 + k)
        return [Cdot, g1 * Cdot + g2 * C + g3 * (Cdot / C) * D - A0 * C**beta * D]

    state = _solve(original, t, [C0, C0 * d0])
    C_exact = C0 * (1 - t / tstar) ** (-1 / k)
    tau = -np.log1p(-t / tstar) / k
    depth = _solve(lambda tau, d: [g1 + g2 / (b * (C0 * np.exp(tau))**k)
                        - ((A0 / b) * (C0 * np.exp(tau))**(beta - k) + 1 - g3) * d[0]],
                   tau, [d0], method="Radau")[0]
    assert_allclose(state[0], C_exact, rtol=2e-8, atol=1e-10)
    assert_allclose(state[1] / state[0], depth, rtol=3e-8, atol=1e-10)


def test_theorem3_zero_gain_nonzero_level_counterexample_to_unqualified_iff():
    tau, C, d = integrate_fraction(0, 1, 0, 1, 1, 1, 1, d0=0.05, tau_end=18)
    expected = np.exp(-tau) - 0.95 * np.exp(-2 * tau)
    assert_allclose(d, expected, rtol=2e-6, atol=2e-11)
    assert d[-1] < 2e-8  # beta=k, but d tends to zero: exact formula proves the limit.


# Theorem 4: autonomous threshold, then the DISTINCT non-autonomous regimes.
@pytest.mark.parametrize("decay", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("g1,d0", [(0.05, 0.05), (0.0, 0.0), (0.0, 0.03)])
def test_theorem4_constant_compounding_all_threshold_and_forcing_cases(decay, g1, d0):
    g3, b = 4.0, 1.0
    A0 = (g3 - 1) * b + decay
    t = np.linspace(0, 12, 401)
    d = _solve(lambda t, x: [g1 * b - (A0 + (1 - g3) * b) * x[0]], t, [d0])[0]
    assert_allclose(d, _affine_exact(t, g1 * b, decay, d0), rtol=4e-8, atol=2e-11)


def test_theorem4_temporary_threshold_crossing_does_not_imply_divergence():
    # gamma3=2, q=0.25 exp(tau): initially unstable; subsequently corrective.
    tau, C, d = integrate_fraction(0.05, 0, 2, 0.25, 2, 1, 1,
                                   d0=0.05, tau_end=9)
    assert d.max() > d[0]
    assert d[-1] < d[0] / 100
    # Independent integrating-factor quadrature at moderate depths.
    H = lambda s: 0.25 * np.expm1(s) - s
    for j in (40, 160, 300):
        t = tau[j]
        expected = 0.05 * np.exp(-H(t)) + 0.05 * quad(
            lambda s: np.exp(-(H(t) - H(s))), 0, t, epsabs=1e-12)[0]
        assert_allclose(d[j], expected, rtol=3e-8, atol=1e-10)


@pytest.mark.parametrize("h", [-0.5, 0.0, 0.5])
def test_theorem4_beta_equal_k_rates_are_in_depth_not_wall_clock(h):
    g, g3, q0 = 0.05, 2.0, 1.0 + h
    tau, C, d = integrate_fraction(g, 0, g3, q0, 1, 1, 1, tau_end=12)
    assert_allclose(d, _affine_exact(tau, g, h, 0.05), rtol=4e-8, atol=2e-10)
    if h == 0:
        wall_t = -np.expm1(-tau)
        assert not np.allclose(d, 0.05 + g * wall_t)  # NOT linear in wall time.


@pytest.mark.parametrize("margin", [-0.5, 0.0, 0.5])
def test_theorem4_subcritical_compounding_limit_is_not_always_gamma_floor(margin):
    g, g3, q0 = 0.05, 0.4, 1.4
    tau, C, d = integrate_fraction(g, 0, g3, q0, 0.5 + margin, 1, 0.5,
                                   tau_end=24)
    if margin < 0:
        assert abs(d[-1] - g / (1 - g3)) < 2e-5
    elif margin == 0:
        assert_allclose(d[-1], g / (q0 + 1 - g3), atol=1e-10)
    else:
        assert d[-1] < 1e-6


def test_theorem4_gamma3_one_logarithmic_not_polynomial_trajectory():
    tau, C, d = integrate_fraction(0.05, 0, 1, 0.5, 0, 0.5, 0.5,
                                   tau_end=40, n=801)
    tail = tau > 30
    assert abs(np.polyfit(tau[tail], d[tail], 1)[0] - 0.05) < 1e-6
    instantaneous_equilibrium = 0.05 * C**0.5
    assert d[-1] / instantaneous_equilibrium[-1] < 1e-5


def test_theorem4_gamma3_one_zero_forcing_positive_limit():
    tau, C, d = integrate_fraction(0, 0, 1, 0.5, 0, 0.5, 0.5, d0=0.2, tau_end=30)
    expected = 0.2 * np.exp(-2 * (1 - np.exp(-0.5 * tau)))
    assert_allclose(d, expected, rtol=2e-8, atol=2e-11)
    assert abs(d[-1] - 0.2 * np.exp(-2)) < 1e-7


def test_theorem4_positive_decay_even_with_infinite_integral_can_have_unbounded_forcing_response():
    # x'=1-x/(1+t), x(0)=0; homogeneous response decays but forced x grows.
    t = np.linspace(0, 50, 401)
    d = _solve(lambda t, x: [1 - x[0] / (1 + t)], t, [0])[0]
    assert_allclose(d, (t + 0.5 * t**2) / (1 + t), rtol=2e-8, atol=2e-10)


# Theorem 5: full vector dynamics, independent projection, and matrix exponential.
@pytest.mark.parametrize("g3", [0.0, 0.4, 1.0, 1.4])
def test_theorem5_full_nonnormal_system_closes_on_left_null_vector(g3):
    A = np.array([[0.0, 1.0], [0.0, 2.0]])
    v = np.array([2.0, -1.0]) / np.sqrt(5)
    c, x0, g = np.array([1.0, 0.2]), np.array([0.03, 0.01]), 0.05
    assert_allclose(v @ A, 0, atol=1e-15)
    assert not np.allclose(A @ v, 0)
    tau = np.linspace(0, 10, 401)
    x = _solve(lambda t, x: g * c - (A + (1 - g3) * np.eye(2)) @ x, tau, x0)
    expected = _affine_exact(tau, g * (v @ c), 1 - g3, v @ x0)
    assert_allclose(v @ x, expected, rtol=5e-8, atol=1e-10)


def test_theorem5_right_null_projection_regression_detects_missing_coupling():
    A = np.array([[0.0, 1.0], [0.0, 2.0]])
    v, c = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    x0, g = np.array([0.1, 0.1]), 0.05
    tau = np.linspace(0, 4, 201)
    assert_allclose(A @ v, 0)
    assert not np.allclose(v @ A, 0)
    x = _solve(lambda t, x: g * c - (A + np.eye(2)) @ x, tau, x0)
    incorrectly_dropped_coupling = _affine_exact(tau, g * (v @ c), 1, v @ x0)
    assert np.max(np.abs(v @ x - incorrectly_dropped_coupling)) > 0.01
    projected_rhs = g * (v @ c) - v @ x - v @ A @ x
    full_rhs_projection = v @ (g * c[:, None] - (A + np.eye(2)) @ x)
    assert_allclose(projected_rhs, full_rhs_projection, atol=1e-14)


def test_theorem5_symmetric_left_and_right_null_conditions_agree():
    A = np.array([[1.0, -1.0], [-1.0, 1.0]])
    v = np.ones(2) / np.sqrt(2)
    assert_allclose(A @ v, 0, atol=1e-15)
    assert_allclose(v @ A, 0, atol=1e-15)


def test_theorem5_nonnormal_transient_growth_with_positive_spectrum():
    M, x0 = np.array([[1.0, 25.0], [0.0, 2.0]]), np.ones(2)
    assert np.max(np.linalg.eigvals(-M).real) < 0  # conventional spectral abscissa
    t = np.linspace(0, 12, 501)
    numerical = _solve(lambda t, x: -M @ x, t, x0)
    exact = np.column_stack([expm(-M * s) @ x0 for s in t])
    assert_allclose(numerical, exact, rtol=2e-7, atol=2e-10)
    norms = np.linalg.norm(numerical, axis=0)
    assert norms.max() > 1.3 * norms[0]
    assert norms[-1] < 1e-3


def test_theorem5_hermitian_margin_controls_transient():
    M, x0 = np.array([[2.0, 1.0], [-1.0, 3.0]]), np.ones(2)
    mu = np.linalg.eigvalsh((M + M.T) / 2).min()
    t = np.linspace(0, 8, 301)
    x = _solve(lambda t, x: -M @ x, t, x0)
    assert np.all(np.linalg.norm(x, axis=0) <= np.linalg.norm(x0) * np.exp(-mu * t) + 1e-10)


def test_theorem5_time_varying_pointwise_spectrum_does_not_certify_stability():
    # Both frozen matrices are positive stable; their switched flow expands.
    M1 = np.array([[1.0, -4.0], [0.0, 1.0]])
    M2 = M1.T
    for M in (M1, M2):
        assert np.linalg.eigvals(M).real.min() > 0
    flow = expm(-M2) @ expm(-M1)
    assert max(abs(np.linalg.eigvals(flow))) > 2
    x = np.ones(2)
    for M in (M1, M2):
        x = _solve(lambda t, y: -M @ y, np.linspace(0, 1, 101), x)[:, -1]
    assert_allclose(x, flow @ np.ones(2), rtol=2e-8, atol=1e-10)


# Theorem 6: deterministic moment equations + independent exact-transition ensemble.
@pytest.mark.parametrize("kappa,sigma", [(0.5, 0.02), (1.1, 0.05), (3.0, 0.1)])
def test_theorem6_finite_time_and_stationary_moments(kappa, sigma):
    injection, d0, v0 = 0.015, 0.2, 0.01
    mu = injection / kappa
    t = np.linspace(0, 15, 301)
    moment = _solve(lambda t, y: [injection - kappa * y[0], sigma**2 - 2 * kappa * y[1]],
                    t, [d0, v0])
    assert_allclose(moment[0], mu + (d0 - mu) * np.exp(-kappa * t), rtol=2e-8, atol=1e-10)
    expected_v = v0 * np.exp(-2 * kappa * t) - sigma**2 * np.expm1(-2 * kappa * t) / (2 * kappa)
    assert_allclose(moment[1], expected_v, rtol=2e-8, atol=1e-10)


@pytest.mark.parametrize("seed", [7, 23, 101])
def test_theorem6_ou_stationary_variance(seed):
    # Independent paths, not treating one autocorrelated time series as iid.
    rng, size, kappa, sigma, mu = np.random.default_rng(seed), 40000, 1.1, 0.05, 0.015 / 1.1
    x, dt = np.full(size, 0.2), 0.25
    decay = np.exp(-kappa * dt)
    innovation = sigma * np.sqrt(-np.expm1(-2 * kappa * dt) / (2 * kappa))
    for _ in range(60):
        x = mu + decay * (x - mu) + innovation * rng.standard_normal(size)
    variance = sigma**2 / (2 * kappa)
    # Six-standard-error envelopes, specified before running; no tolerance retuning.
    assert abs(x.mean() - mu) < 6 * np.sqrt(variance / size)
    assert abs(x.var(ddof=1) - variance) < 6 * variance * np.sqrt(2 / (size - 1))


def test_theorem6_stationary_tail_is_single_time_not_pathwise_certificate():
    mu, sd, critical = 0.01, 0.03, 0.08
    integrated_tail = quad(lambda x: norm.pdf(x, loc=mu, scale=sd), critical, np.inf)[0]
    assert_allclose(integrated_tail, norm.sf((critical - mu) / sd), rtol=1e-9)
    assert norm.cdf(-mu / sd) > 0.3  # unrestricted OU is NOT a bounded fraction.


def test_theorem6_brownian_time_change_preserves_constant_coefficient_variance():
    kappa, r, sigma = 1.1, 0.3, 0.05
    depth_kappa, depth_sigma = kappa / r, sigma / np.sqrt(r)
    assert_allclose(depth_sigma**2 / (2 * depth_kappa), sigma**2 / (2 * kappa))


# Theorem 7: finite-capacity equilibrium, actual trajectories, and capacity lift.
def _hill_q(C, *, Amax=1.0, Cs=3.0, beta=1.5, b=0.2, k=0.5):
    C = np.asarray(C)
    return (Amax / (1 + (Cs / C)**beta)) / (b * C**k)


@pytest.mark.parametrize("beta,k", [(1.0, 0.5), (1.5, 0.5), (2.0, 1.5)])
def test_theorem7_capacity_optimum_matches_independent_optimisation(beta, k):
    Amax, Cs, b = 2.0, 3.0, 0.2
    q = lambda C: _hill_q(C, Amax=Amax, Cs=Cs, beta=beta, b=b, k=k)
    Copt = Cs * ((beta - k) / k)**(1 / beta)
    qmax = (Amax / (b * Cs**k)) * (k / beta) * ((beta - k) / k)**((beta - k) / beta)
    result = minimize_scalar(lambda logC: -q(np.exp(logC)), bounds=(-8, 8), method="bounded")
    assert result.success
    assert_allclose(np.exp(result.x), Copt, rtol=2e-5)
    assert_allclose(-result.fun, qmax, rtol=1e-9)


@pytest.mark.parametrize("d0", [0.0, 0.1])
def test_theorem7_equilibrium_minimum_is_not_every_trajectory_minimum(d0):
    g, beta, k, Cs, Amax, b = 0.05, 1.5, 0.5, 3.0, 1.0, 0.2
    Copt = Cs * ((beta - k) / k)**(1 / beta)
    qmax = _hill_q(Copt)
    m = g / (1 + qmax)
    tau = np.linspace(0, 20, 601)
    d = _solve(lambda t, x: [g - (1 + _hill_q(Copt * np.exp(t))) * x[0]], tau, [d0],
               method="Radau")[0]
    lower = m + (d0 - m) * np.exp(-(1 + qmax) * tau)
    assert np.all(d >= lower - 1e-10)
    if d0 == 0:
        assert d[0] < m  # old universal-floor interpretation must never return.
    else:
        assert d.min() >= m - 1e-10
    assert abs(d[-1] - g) < 1e-4


@pytest.mark.parametrize("cap_margin", [-0.5, 0.0, 0.5])
def test_theorem7_capacity_lift_changes_final_exponent(cap_margin):
    g, k, a0, b = 0.05, 0.5, 0.7, 0.4
    tau = np.linspace(0, 22, 601)
    def rhs(t, x):
        C = np.exp(t)
        q = _hill_q(C, Amax=a0 * C**(k + cap_margin), b=b, k=k)
        return [g - (1 + q) * x[0]]
    d = _solve(rhs, tau, [0.1], method="Radau")[0]
    expected = 0.0 if cap_margin > 0 else (g if cap_margin < 0 else g / (1 + a0 / b))
    assert abs(d[-1] - expected) < 1e-5


def test_theorem7_threshold_invariance_requires_safe_initial_state():
    g, critical = 0.05, 0.02
    q_required = g / critical - 1
    tau = np.linspace(0, 10, 401)
    d = _solve(lambda t, x: [g - (1 + q_required + np.sin(t)**2) * x[0]], tau, [critical])[0]
    assert d.max() <= critical + 1e-11


# Model implications and safe generalisation of time-varying exponents.
@pytest.mark.parametrize("margin", [0.5, 1.0, 1.5])
def test_relative_and_absolute_misalignment_have_distinct_asymptotic_exponents(margin):
    # Positive gain drift; avoid evaluating d below numerical absolute resolution.
    tau, C, d = integrate_fraction(0.05, 0, 0, 1, 0.5 + margin, 1, 0.5, tau_end=8)
    relative_leading = 0.05 * C[-1]**(-margin)
    assert abs(d[-1] / relative_leading - 1) < 0.02
    assert_allclose(C[-1] * d[-1] / (0.05 * C[-1]**(1 - margin)),
                    d[-1] / relative_leading, rtol=1e-12)


def test_time_varying_ratio_exact_kernel_instead_of_instantaneous_iff():
    # q=2+sin(tau)>0, H=int(1+q)=3tau+1-cos(tau).
    tau = np.linspace(0, 6, 121)
    H = lambda t: 3 * t + 1 - np.cos(t)
    d = _solve(lambda t, x: [0.05 - (3 + np.sin(t)) * x[0]], tau, [0.2])[0]
    expected = [0.2 * np.exp(-H(t)) + 0.05 * quad(
        lambda s: np.exp(-(H(t) - H(s))), 0, t, epsabs=1e-12)[0] for t in tau]
    assert_allclose(d, expected, rtol=2e-8, atol=1e-11)


# Regressions for the TEST INSTRUMENT: a failed solve cannot count as a pass.
@pytest.mark.parametrize("bad", [
    {"b": 0}, {"C0": 0}, {"gamma1": -1}, {"gamma2": np.nan},
    {"beta": np.inf}, {"tau_end": 0}, {"n": 1}, {"n": 2.5},
])
def test_integrate_fraction_rejects_invalid_inputs(bad):
    params = dict(gamma1=0.05, gamma2=0, gamma3=0, A0=1, beta=1, b=1, k=0)
    params.update(bad)
    with pytest.raises(ValueError):
        integrate_fraction(**params)


@pytest.mark.parametrize("kind", ["failed", "event", "truncated", "nan"])
def test_solver_guard_rejects_inadmissible_results(monkeypatch, kind):
    grid = np.array([0.0, 1.0])
    fake = SimpleNamespace(success=kind != "failed", status=1 if kind == "event" else 0,
                           message="deliberate regression fixture", t=grid.copy(), y=np.zeros((1, 2)))
    if kind == "failed":
        fake.status = -1
    if kind == "truncated":
        fake.t, fake.y = grid[:1], np.zeros((1, 1))
    if kind == "nan":
        fake.y[0, -1] = np.nan
    monkeypatch.setitem(_solve.__globals__, "solve_ivp", lambda *a, **kw: fake)
    with pytest.raises(RuntimeError):
        _solve(lambda t, y: -y, grid, [1])


def test_theorem1_zero_decay_is_neutral_not_attracting():
    t = np.linspace(0, 10, 101)
    d = _solve(lambda t, x: [0.0], t, [0.2])[0]
    assert_allclose(d, 0.2, atol=1e-14)


@pytest.mark.parametrize("g3", [1.2, 2.0])
def test_theorem4_beta_below_k_supercritical_compounding_rate(g3):
    tau, C, d = integrate_fraction(0.05, 0, g3, 0.5, 0, 0.5, 0.5,
                                   tau_end=40, n=801)
    tail = tau > 35
    assert abs(np.polyfit(tau[tail], np.log(d[tail]), 1)[0] - (g3 - 1)) < 5e-4


def test_theorem5_common_left_null_projection_for_time_varying_corrector():
    base = np.array([[0.0, 1.0], [0.0, 2.0]])
    v = np.array([2.0, -1.0]) / np.sqrt(5)
    c, x0 = np.array([1.0, 0.2]), np.array([0.03, 0.01])
    tau = np.linspace(0, 10, 301)
    x = _solve(lambda t, x: 0.05 * c - 0.6 * x - (1 + np.sin(t)**2) * base @ x,
               tau, x0)
    assert_allclose(v @ x, _affine_exact(tau, 0.05 * (v @ c), 0.6, v @ x0),
                    rtol=3e-8, atol=1e-10)


def test_theorem6_positive_margin_does_not_control_scaling_noise():
    C = np.logspace(1, 6, 31)
    A, r, sigma = C**1.5, C**0.5, C**0.75
    mu, variance = 0.05 * r / (A + r), sigma**2 / (2 * (A + r))
    assert mu[-1] < 1e-7
    assert variance[-1] > 0.49  # beta>k, but noise variance does not vanish.


def test_theorem7_exponential_growth_has_post_saturation_gap():
    tau = np.linspace(0, 20, 401)
    d = _solve(lambda t, x: [0.05 - (1 + _hill_q(np.exp(t), k=0)) * x[0]],
               tau, [0.1], method="Radau")[0]
    assert_allclose(d[-1], 0.05 / (1 + 1 / 0.2), atol=1e-10)


def test_theorem7_zero_gain_drift_does_not_require_scaling_capacity():
    tau = np.linspace(0, 20, 401)
    d = _solve(lambda t, x: [-(1 + _hill_q(np.exp(t))) * x[0]],
               tau, [0.1], method="Radau")[0]
    assert d.min() >= -1e-11
    assert d[-1] < 1e-9


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
