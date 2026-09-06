"""Balance objects for the frontier propositions: trend, level, surplus and event are four objects.

Under the registered marginal-burden model (U = I R^alpha, W = b dU/dR, Q = q U^chi) the balance
elasticity is Delta = 1 - alpha (1 - chi) and its zero is the trend crossover alpha_crit = 1/(1 - chi).
That zero says where the service ratio M = Q/W stops improving. It says nothing about whether M is
above one, whether the surplus S = Q - W is positive, whether a backlog accumulates, or whether
behaviour fails. Each of those is its own object with its own rule, and P3, P4 and P16 must each name
the one they score. The two worked counterexamples from the exact-file audit of 5 September 2026 are
pinned as tests: a system above the trend boundary with ample service for ten thousand rounds, and a
system with a favourable trend and inadequate service for ten thousand rounds.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

MARGIN_REVERSAL = "margin-reversal"
SERVICE_DEFICIT = "service-deficit"
BACKLOG_THRESHOLD = "backlog-threshold"
CONFORMANCE_FAILURE = "conformance-failure"
ENDPOINTS = (MARGIN_REVERSAL, SERVICE_DEFICIT, BACKLOG_THRESHOLD, CONFORMANCE_FAILURE)


def balance_elasticity(alpha: float, chi: float) -> float:
    """Delta = 1 - alpha (1 - chi): the slope of log(Q/W) in log R under the restricted model."""
    return 1.0 - alpha * (1.0 - chi)


def trend_crossover(chi: float) -> float:
    """The zero of the balance elasticity in alpha; defined for chi in [0, 1) only."""
    if not (0.0 <= chi < 1.0):
        raise ValueError("the restricted frontier is defined for chi in [0, 1); no negative ceiling is reported")
    return 1.0 / (1.0 - chi)


def service_ratio(Q: Sequence[float], W: Sequence[float]) -> np.ndarray:
    Q = np.asarray(Q, float)
    W = np.asarray(W, float)
    if np.any(W <= 0):
        raise ValueError("the service ratio needs positive offered burden; a zero-burden regime is handled without a ratio")
    return Q / W


def surplus(Q: Sequence[float], W: Sequence[float]) -> np.ndarray:
    return np.asarray(Q, float) - np.asarray(W, float)


def model_paths(R: Sequence[float], alpha: float, chi: float, I: float = 1.0, b: float = 1.0, q: float = 1.0) -> Dict[str, np.ndarray]:
    """U, W, Q, M and S along the restricted model for given coefficients."""
    R = np.asarray(R, float)
    U = I * R ** alpha
    W = b * I * alpha * R ** (alpha - 1.0)
    Q = q * U ** chi
    return {"R": R, "U": U, "W": W, "Q": Q, "M": Q / W, "S": Q - W}


def ratio_crossing_depth(M0: float, delta: float, R0: float = 1.0) -> float:
    """Depth at which M = M0 (R/R0)^delta reaches one; infinite when the path never reaches one.

    This is the service-ratio crossing, a level event. It is not the trend crossover, which is a slope
    event, and near delta = 0 its uncertainty is enormous.
    """
    if M0 <= 0:
        raise ValueError("M0 must be positive")
    if M0 == 1.0:
        return R0
    if (M0 > 1.0 and delta >= 0) or (M0 < 1.0 and delta <= 0):
        return math.inf
    return R0 * M0 ** (-1.0 / delta)


def fitted_trend(R: Sequence[float], M: Sequence[float]) -> float:
    """Slope of log M on log R over the supplied window (a finite-window trend, not an asymptote)."""
    lr = np.log(np.asarray(R, float))
    lm = np.log(np.asarray(M, float))
    return float(np.polyfit(lr, lm, 1)[0])


def describe(R: Sequence[float], Q: Sequence[float], W: Sequence[float]) -> Dict[str, object]:
    """Report the four objects separately for a measured path: trend, level, surplus and first crossing."""
    R = np.asarray(R, float)
    M = service_ratio(Q, W)
    S = surplus(Q, W)
    below = np.where(M < 1.0)[0]
    neg = np.where(S < 0.0)[0]
    return {
        "trend": fitted_trend(R, M),
        "level_above_one_throughout": bool(np.all(M > 1.0)),
        "first_depth_ratio_below_one": (float(R[below[0]]) if below.size else None),
        "first_depth_surplus_negative": (float(R[neg[0]]) if neg.size else None),
        "ratio_at_end": float(M[-1]),
    }


def counterexample_above_bound(R: float = 32.0) -> Dict[str, float]:
    """U = R^3, W = 3 R^2, Q = 300 R^(3/2): alpha = 3 above the trend boundary of two (chi = 1/2),
    yet M = 100 / sqrt(R) exceeds one until R = 10,000 (about 17.68 at R = 32)."""
    M = 300.0 * R ** 1.5 / (3.0 * R ** 2)
    return {"alpha": 3.0, "chi": 0.5, "trend_crossover": trend_crossover(0.5), "balance_elasticity": balance_elasticity(3.0, 0.5),
            "ratio_at_R": M, "ratio_crossing_depth": ratio_crossing_depth(100.0, -0.5)}


def counterexample_favourable_trend(R: float = 32.0) -> Dict[str, float]:
    """U = R, W = 1, Q = 0.01 sqrt(R): a favourable trend (Delta = 1/2) with inadequate service until R = 10,000."""
    M = 0.01 * math.sqrt(R)
    return {"balance_elasticity": 0.5, "ratio_at_R": M, "ratio_crossing_depth": ratio_crossing_depth(0.01, 0.5)}


def generalised_frontier(chi: float, mu_B: float = 0.0, eta: float = 0.0, zeta: float = 0.0) -> float:
    """alpha_crit = (1 + eta - zeta) / (1 + mu_B - chi), the zero of the generalised balance on a
    pure-power path, defined only where numerator and denominator are positive; the restricted relation
    is the case mu_B = eta = zeta = 0. The adversarial example from the implementation brief: chi 0.5,
    mu_B 0.2 and eta 0.4 also return two, so a measured boundary of two cannot by itself establish the
    restricted mechanism; P20's combined rival stays."""
    num = 1.0 + eta - zeta
    den = 1.0 + mu_B - chi
    if num <= 0.0 or den <= 0.0:
        raise ValueError("outside the positive-numerator, positive-denominator regime; read the balance elasticity instead")
    return num / den


def local_balance(alpha_R: float, dlog_alpha_dlogR: float, chi: float, mu_B: float = 0.0, eta: float = 0.0,
                  zeta: float = 0.0) -> float:
    """Delta for smooth capability with a local elasticity alpha_R: the constant-exponent formula plus the
    curvature term minus d log alpha_R / d log R. For C = exp(0.2 R) at R = 5 the local elasticity is one
    and the curvature term is one; omitting it turns the balance from minus one half into plus one half."""
    return 1.0 + eta - zeta - alpha_R * (1.0 + mu_B - chi) - dlog_alpha_dlogR


def queue_backlog(W: Sequence[float], Q: Sequence[float], threshold: float) -> Dict[str, object]:
    """Toy first-in-first-out aggregate backlog B[r+1] = max(0, B[r] + W[r] - Q[r]); reports the first index
    (one-based) at which the backlog exceeds the threshold, or None. Illustrative only: an event threshold
    in this queue is not an alignment endpoint."""
    B = 0.0
    first = None
    for i, (w, q) in enumerate(zip(np.asarray(W, float), np.asarray(Q, float)), start=1):
        B = max(0.0, B + w - q)
        if first is None and B > threshold:
            first = i
    return {"first_event_index": first, "backlog_at_end": B}


def queue_witness(coefficients: Sequence[float] = (3.0, 30.0, 3000.0), R_max: int = 1024, threshold: float = 100.0) -> List[Dict[str, object]]:
    """Three systems with identical exponents (U = R^3, W = 3 R^2, Q = c R^1.5, Delta = minus one half) and
    different correction coefficients: the service-ratio crossing sits at R = (c/3)^2 and the first toy
    backlog event at R = 6, 101 and never (through 1,024)."""
    R = np.arange(1, R_max + 1, dtype=float)
    W = 3.0 * R ** 2
    out = []
    for c in coefficients:
        Q = c * R ** 1.5
        q = queue_backlog(W, Q, threshold)
        out.append({"coefficient": c, "ratio_crossing_depth": (c / 3.0) ** 2, "first_backlog_event_R": q["first_event_index"],
                    "backlog_at_end": q["backlog_at_end"]})
    return out


def shifted_trajectory_balance(beta_L: float, chi: float, alpha_R: float) -> float:
    """Along the shifted growth solution (dU/dR = a U^beta_L) with W = b dU/dR and Q = q U^chi, the
    service ratio is proportional to U^(chi - beta_L), so the balance is alpha_R (chi - beta_L): exactly
    zero throughout the trajectory when beta_L equals chi, whatever the finite-window slope. Inserting the
    finite-window slope into the constant-exponent formula reports a false surplus; the curvature term
    cancels it (see local_balance)."""
    return alpha_R * (chi - beta_L)


def local_exponent_and_curvature(R: Sequence[float], U: Sequence[float]) -> Dict[str, np.ndarray]:
    """Numerical local elasticity alpha_R = d log U / d log R and its log-derivative on a sampled path.
    Finite differences on a seven-point ladder are coarse; use a dense path or the analytic form
    (shifted_local_terms) where the trajectory's equation is known."""
    lr = np.log(np.asarray(R, float))
    lu = np.log(np.asarray(U, float))
    alpha_R = np.gradient(lu, lr)
    curvature = np.gradient(np.log(alpha_R), lr)
    return {"alpha_R": alpha_R, "dlog_alpha_dlogR": curvature}


def shifted_local_terms(R: Sequence[float], U0: float, a: float, beta_L: float) -> Dict[str, np.ndarray]:
    """Exact local elasticity and curvature along the shifted solution U = [U0^(1-b) + (1-b) a (R-1)]^(1/(1-b)):
    with D = U0^(1-b) + (1-b) a (R-1), alpha_R = a R / D and d log alpha_R / d log R = 1 - (1-b) a R / D,
    so the curvature-corrected balance 1 - alpha_R (1 - chi) - curvature equals alpha_R (chi - b) exactly."""
    R = np.asarray(R, float)
    D = U0 ** (1.0 - beta_L) + (1.0 - beta_L) * a * (R - 1.0)
    alpha_R = a * R / D
    curvature = 1.0 - (1.0 - beta_L) * a * R / D
    return {"alpha_R": alpha_R, "dlog_alpha_dlogR": curvature, "U": D ** (1.0 / (1.0 - beta_L))}


def coefficient_intervention(M0: float, delta: float, factor: float, R0: float = 1.0) -> Dict[str, float]:
    """Multiplying the correction coefficient by `factor` without changing its elasticity moves the
    service-ratio crossing (from 100 to 400 for M = 10 R^(-1/2) and a doubling) and leaves the exponent
    frontier where it was. A level intervention is not a ceiling shift."""
    return {"crossing_before": ratio_crossing_depth(M0, delta, R0), "crossing_after": ratio_crossing_depth(M0 * factor, delta, R0),
            "frontier_unchanged": True}


def author_review_correspondence(mu_B: float, zeta_extra: float) -> Dict[str, float]:
    """On a constant-exponent power path the author-review parameterisation W = W0 U^kappa R^zeta_total
    corresponds to the candidate's W = b U^mu_B (dU/dR) R^zeta_extra by kappa = 1 + mu_B and zeta_total =
    zeta_extra - 1 (coefficients adjusted). Not a general off-path identity; it exists so that the older
    minus one is never imported into the new extra exponent and the derivative counted twice."""
    return {"kappa": 1.0 + mu_B, "zeta_total": zeta_extra - 1.0}


def endpoint_requires_response_model(endpoint: str) -> bool:
    """A margin reversal is read from the balance elasticity alone; every other endpoint needs a
    registered response model with coefficients, initial state, delays, horizon and uncertainty."""
    if endpoint not in ENDPOINTS:
        raise ValueError("unknown endpoint %r; one of %s" % (endpoint, ", ".join(ENDPOINTS)))
    return endpoint != MARGIN_REVERSAL
