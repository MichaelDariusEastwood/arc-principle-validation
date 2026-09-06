"""Burden identification: a derivative-generated workload against a stock-generated one.

On a single pure-power trajectory, log capability and log capability-gain rate are collinear (the design
matrix with an intercept, log C and log dC/dR has rank two), so the marginal-burden model W proportional
to dC/dR and a stock model W proportional to C^kappa are observationally equivalent there. Under a
crossed design that varies capability state and gain rate independently the matrix has rank three, and
the two models recover different coefficients: derivative truth gives a gain-rate coefficient near one
and a capability coefficient near zero; stock truth gives a gain-rate coefficient near zero and a
capability coefficient near its true value. This defeats "no obtainable data can distinguish the
models"; it does not establish that a real intervention on the gain rate leaves every other cause of
workload untouched, which is the exclusion restriction the deciding instrument must state.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def single_path_design(n: int = 50, alpha: float = 2.0) -> np.ndarray:
    """Intercept, log C and log dC/dR along C = R^alpha: rank two."""
    lr = np.log(np.geomspace(1.0, 64.0, n))
    lc = alpha * lr
    lg = np.log(alpha) + (alpha - 1.0) * lr
    return np.column_stack([np.ones(n), lc, lg])


def crossed_design(levels: int = 5, reps: int = 5) -> np.ndarray:
    """Intercept, log C and log gain rate on an independent grid: rank three."""
    lc, lg = np.meshgrid(np.log(np.geomspace(1.0, 16.0, levels)), np.log(np.geomspace(0.1, 4.0, levels)), indexing="ij")
    lc = np.repeat(lc.ravel(), reps)
    lg = np.repeat(lg.ravel(), reps)
    return np.column_stack([np.ones(lc.size), lc, lg])


def design_rank(X: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(X))


def recover(X: np.ndarray, log_w: np.ndarray) -> Dict[str, float]:
    coef, *_ = np.linalg.lstsq(X, log_w, rcond=None)
    return {"intercept": float(coef[0]), "capability_coefficient": float(coef[1]), "gain_rate_coefficient": float(coef[2])}


def witness(noise: float = 0.15, seed: int = 0) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    X = crossed_design()
    derivative = -1.0 + 1.0 * X[:, 2] + rng.normal(0.0, noise, X.shape[0])
    stock = -1.0 + 0.7 * X[:, 1] + rng.normal(0.0, noise, X.shape[0])
    return {
        "single_path_rank": design_rank(single_path_design()),
        "crossed_rank": design_rank(X),
        "derivative_truth": recover(X, derivative),
        "stock_truth": recover(X, stock),
    }
