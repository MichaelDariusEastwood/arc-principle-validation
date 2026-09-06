"""The checker-to-target conversion needs the complete chain rule.

For correction service Q(U, K), with U the target capability and K the checker capability, the derivative
along a declared allocation path K(U) is

    d log Q / d log U = (partial log Q / partial log U at fixed K)
                      + (partial log Q / partial log K at fixed U) * d log K / d log U.

The product-only conversion keeps the second term alone. It is valid only where the direct term is
independently constrained to zero. The worked example from the exact-file audit of 5 September 2026:
Q = K^0.8 U^-0.4 with K proportional to U gives a target-axis elasticity of 0.4, not 0.8, and the
implied restricted frontiers are about 1.67 and 5. A study that varies the checker while holding the
target fixed identifies neither the direct term nor the allocation elasticity; a crossed design does.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .precision import implied_ceiling


def target_axis_elasticity(direct: float, cross: float, allocation: float) -> float:
    """direct + cross * allocation, the complete chain rule along the declared path."""
    return direct + cross * allocation


def product_only(cross: float, allocation: float) -> float:
    """The second term alone; the special case where the direct term is zero."""
    return cross * allocation


def path_elasticity(chi: float, eta: float, alpha: float) -> float:
    """For Q = q U^chi R^eta along U proportional to R^alpha, the slope of log Q on log U is chi + eta/alpha.
    It cannot be inserted as chi and then have eta added again."""
    if alpha == 0:
        raise ValueError("a path elasticity in capability needs a non-zero growth exponent")
    return chi + eta / alpha


def worked_example() -> Dict[str, float]:
    direct, cross, allocation = -0.4, 0.8, 1.0
    total = target_axis_elasticity(direct, cross, allocation)
    only = product_only(cross, allocation)
    return {"direct": direct, "cross": cross, "allocation": allocation, "target_axis": total, "product_only": only,
            "ceiling_target_axis": implied_ceiling(total), "ceiling_product_only": implied_ceiling(only)}


def crossed_design(direct: float, cross: float, n_u: int = 6, n_k: int = 6, reps: int = 3, noise: float = 0.05,
                   seed: int = 0) -> Dict[str, float]:
    """Vary target and checker capability on a crossed grid and recover both partial elasticities by least squares."""
    rng = np.random.default_rng(seed)
    lu = np.repeat(np.linspace(0.0, 2.0, n_u), n_k)
    lk = np.tile(np.linspace(0.0, 2.0, n_k), n_u)
    lu = np.tile(lu, reps)
    lk = np.tile(lk, reps)
    ld = direct * lu + cross * lk + rng.normal(0.0, noise, lu.size)
    X = np.column_stack([np.ones_like(lu), lu, lk])
    coef, *_ = np.linalg.lstsq(X, ld, rcond=None)
    return {"direct_hat": float(coef[1]), "cross_hat": float(coef[2]), "identified": True}


def one_path_design(direct: float, cross: float, allocation: float = 1.0, n: int = 24, noise: float = 0.05,
                    seed: int = 0) -> Dict[str, float]:
    """Vary the target with the checker allocated along one path K proportional to U^allocation: only the total
    target-axis elasticity is recoverable; the two partials are not separately identified."""
    rng = np.random.default_rng(seed)
    lu = np.linspace(0.0, 2.0, n)
    lk = allocation * lu
    ld = direct * lu + cross * lk + rng.normal(0.0, noise, n)
    slope = float(np.polyfit(lu, ld, 1)[0])
    return {"total_hat": slope, "identified": False, "truth_total": target_axis_elasticity(direct, cross, allocation)}
