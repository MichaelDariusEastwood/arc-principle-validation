"""Correction capacity against throughput, and slopes with error on both axes.

Observed corrections completed equal the minimum of available capacity and offered correctable work.
A corrector fed too few faults returns burden-tracking throughput; a fixed finite fault bank returns
an artificial low elasticity. This module gives the accounting (net correction with introduced faults
counted), the supply-sufficiency regime test on a load titration, and Deming regression for the
elasticity of one measured quantity on another, because ordinary least squares on a noisy regressor
attenuates the slope towards zero and can falsely refute or falsely support.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


def net_correction(offered: float, corrected_valid: float, introduced: float) -> float:
    """Net valid clearance: valid faults removed minus new faults introduced. May be zero or negative,
    and is reported as such rather than logged away."""
    return float(corrected_valid - introduced)


def supply_regime(offered_levels: Sequence[float], completed: Sequence[float],
                  tracking_ratio: float = 0.90, plateau_tolerance: float = 0.10) -> Dict[str, object]:
    """Classify a load titration at one capability level.

    supply-limited: completed tracks offered at the highest load (ratio at or above tracking_ratio),
    so the corrector was never the bottleneck and its capacity is unmeasured here.
    capacity-limited: completed changes by less than plateau_tolerance (relative) across the top two
    loads while offered rises, so the plateau is the capacity estimate.
    transition: neither; report and add loads.
    """
    o = np.asarray(offered_levels, dtype=float); c = np.asarray(completed, dtype=float)
    if len(o) < 2 or len(o) != len(c):
        return {"regime": "unresolved", "capacity_estimate": float("nan")}
    order = np.argsort(o); o = o[order]; c = c[order]
    if c[-1] / o[-1] >= tracking_ratio:
        return {"regime": "supply-limited", "capacity_estimate": float("nan"), "top_ratio": float(c[-1] / o[-1])}
    rel = abs(c[-1] - c[-2]) / max(abs(c[-2]), 1e-12)
    if rel <= plateau_tolerance and o[-1] > o[-2]:
        return {"regime": "capacity-limited", "capacity_estimate": float(np.mean(c[-2:])), "plateau_change": float(rel)}
    return {"regime": "transition", "capacity_estimate": float("nan"), "plateau_change": float(rel)}


def ols_slope(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    return float(np.cov(x, y, bias=True)[0, 1] / np.var(x))


def deming_slope(x: Sequence[float], y: Sequence[float], delta: float = 1.0) -> float:
    """Deming regression slope with error-variance ratio delta = var(err_y) / var(err_x).

    delta = 1 is orthogonal regression; delta -> infinity recovers ordinary least squares of y on x.
    Closed form: (s_yy - delta s_xx + sqrt((s_yy - delta s_xx)^2 + 4 delta s_xy^2)) / (2 s_xy).
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    sxx = np.var(x); syy = np.var(y); sxy = np.cov(x, y, bias=True)[0, 1]
    if sxy == 0:
        return 0.0
    return float((syy - delta * sxx + np.sqrt((syy - delta * sxx) ** 2 + 4 * delta * sxy ** 2)) / (2 * sxy))


def capacity_elasticity(capability: Sequence[float], capacity: Sequence[float], delta: float = 1.0,
                        n_boot: int = 2000, seed: int = 20260905) -> Dict[str, object]:
    """Elasticity of correction capacity with capability on log scales, with error on both axes, and a
    bootstrap interval over cells. Only capacity-limited cells should enter."""
    lx = np.log(np.asarray(capability, float)); ly = np.log(np.asarray(capacity, float))
    est = deming_slope(lx, ly, delta)
    rng = np.random.default_rng(seed); n = len(lx); boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(idx.tolist())) < 3:
            continue
        boots.append(deming_slope(lx[idx], ly[idx], delta))
    return {"elasticity": est, "ols_elasticity": ols_slope(lx, ly),
            "interval": (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))) if boots else (float("nan"), float("nan")),
            "n_cells": n}
