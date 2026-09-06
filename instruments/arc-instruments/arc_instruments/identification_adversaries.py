"""Adversarial generators for P5's identification contract.

The toy rate model is log G = log a + beta_L log C + theta log f + noise, with C the capability state and f
the retained fraction. A retention-only experiment at fixed capability estimates theta and can be
misread as the coupling; a crossed capability-by-retention design recovers beta_L and theta separately;
but where the nuisance rate itself depends on capability (a proportional to C^lambda, unmodelled) the
crossed design returns beta_L plus lambda with nominal intervals that never cover the structural
coupling. The acceptance requirement this module encodes: recovery under favourable generators, the
retention-only misreading rejected, and non-identification declared, not hidden, under a
capability-dependent nuisance rate.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def crossed_regression(beta: float, theta: float, lam: float = 0.0, levels: int = 5, reps: int = 5,
                       noise: float = 0.15, seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    lc, lf = np.meshgrid(np.log(np.geomspace(1.0, 16.0, levels)), np.log(np.geomspace(0.125, 1.0, levels)), indexing="ij")
    lc = np.repeat(lc.ravel(), reps)
    lf = np.repeat(lf.ravel(), reps)
    X = np.column_stack([np.ones(lc.size), lc, lf])
    y = -2.0 + (beta + lam) * lc + theta * lf + rng.normal(0.0, noise, lc.size)
    coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    df = lc.size - 3
    sigma2 = float(res[0]) / df if len(res) else float(np.sum((y - X @ coef) ** 2)) / df
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = float(np.sqrt(cov[1, 1]))
    return {"beta_hat": float(coef[1]), "theta_hat": float(coef[2]), "beta_se": se,
            "covers_structural_beta": bool(abs(coef[1] - beta) <= 1.96 * se)}


def retention_only(beta: float, theta: float, levels: int = 5, reps: int = 25, noise: float = 0.15,
                   seed: int = 0, fixed_capability: float = 4.0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    lf = np.repeat(np.log(np.geomspace(0.125, 1.0, levels)), reps)
    X = np.column_stack([np.ones(lf.size), lf])
    y = -2.0 + beta * np.log(fixed_capability) + theta * lf + rng.normal(0.0, noise, lf.size)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {"slope": float(coef[1]), "estimates": "the retention sensitivity theta, not the coupling"}


def acceptance(beta: float = 0.5, theta: float = 0.8, lam: float = 0.3, reps: int = 200, seed: int = 0) -> Dict[str, object]:
    """Run the three generators over replicates and report what an instrument must show."""
    fav = np.array([crossed_regression(beta, theta, 0.0, seed=seed + i)["beta_hat"] for i in range(reps)])
    ret = np.array([retention_only(beta, theta, seed=seed + i)["slope"] for i in range(reps)])
    conf = [crossed_regression(beta, theta, lam, seed=seed + i) for i in range(reps)]
    return {
        "favourable_crossed_beta_mean": float(fav.mean()),
        "retention_only_slope_mean": float(ret.mean()),
        "confounded_crossed_beta_mean": float(np.mean([c["beta_hat"] for c in conf])),
        "confounded_coverage_of_structural_beta": float(np.mean([c["covers_structural_beta"] for c in conf])),
        "declaration": "non-identification under a capability-dependent nuisance rate must be declared",
    }
