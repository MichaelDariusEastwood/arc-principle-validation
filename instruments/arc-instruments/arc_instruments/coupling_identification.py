"""P5 identification: does the instrument estimate the coupling, or a retention effect?

The programme's coupling equation is dC/dR = a C^beta, with the finite-window solution
C(R) = [C0^(1-beta) + (1-beta) a (R - 1)]^(1/(1-beta)) for beta < 1. Its asymptotic exponent is
1/(1-beta); its finite-window slope depends on the baseline, the rate and the window (for beta 0.5,
C0 = 1, a = 0.1 the endpoint slope over depths 1 to 32 is 0.540, not 2).

The reviews' countermodel: dC/dR = a f^theta C^beta, where f is the retained fraction of prior
material. Varying f at fixed capability estimates theta, the retention exponent, and says nothing
about beta. Only a design that varies capability and retention on a crossed grid, with the structural
model fitted jointly, identifies both. This module simulates both designs, with measurement noise, and
reports what each recovers, including the negative control (beta = 0 with theta > 0), in which a
retention-only design would report a "coupling" that does not exist.
"""
from __future__ import annotations

import argparse
from typing import Dict, Sequence, Tuple

import numpy as np


def trajectory(C0: float, a: float, beta: float, R: np.ndarray, f: float = 1.0, theta: float = 0.0) -> np.ndarray:
    """Closed-form solution of dC/dR = a f^theta C^beta from C(1) = C0, for beta < 1."""
    if beta >= 1.0:
        raise ValueError("beta must be below one for the finite-window solution used here")
    rate = a * f ** theta
    return (C0 ** (1 - beta) + (1 - beta) * rate * (R - 1.0)) ** (1.0 / (1.0 - beta))


def endpoint_slope(C0: float, a: float, beta: float, r1: float, r2: float, f: float = 1.0, theta: float = 0.0) -> float:
    """Finite-window log-log secant slope between depths r1 and r2."""
    c = trajectory(C0, a, beta, np.array([r1, r2], float), f, theta)
    return float((np.log(c[1]) - np.log(c[0])) / (np.log(r2) - np.log(r1)))


def loglog_fit_slope(C0: float, a: float, beta: float, R: np.ndarray, f: float = 1.0, theta: float = 0.0) -> float:
    c = trajectory(C0, a, beta, R, f, theta)
    x = np.log(R); y = np.log(c)
    return float(np.polyfit(x, y, 1)[0])


def improvement_rate(C: float, a: float, beta: float, f: float, theta: float, rng: np.random.Generator,
                     noise: float) -> float:
    """One noisy observation of the improvement rate dC/dR at capability C and retention f."""
    return a * f ** theta * C ** beta * np.exp(rng.normal(0.0, noise))


def retention_only_design(C_fixed: float, f_grid: Sequence[float], a: float, beta: float, theta: float,
                          noise: float, reps_per_cell: int, rng: np.random.Generator) -> Dict[str, float]:
    """Vary retention at fixed capability; regress log rate on log f. The slope estimates theta."""
    xs, ys = [], []
    for f in f_grid:
        for _ in range(reps_per_cell):
            xs.append(np.log(f)); ys.append(np.log(improvement_rate(C_fixed, a, beta, f, theta, rng, noise)))
    slope = float(np.polyfit(xs, ys, 1)[0])
    return {"slope_on_log_f": slope, "estimates": "theta (the retention exponent); beta is not identified"}


def crossed_design(C_grid: Sequence[float], f_grid: Sequence[float], a: float, beta: float, theta: float,
                   noise: float, reps_per_cell: int, rng: np.random.Generator) -> Dict[str, float]:
    """Vary capability and retention on a crossed grid; fit log rate = const + theta log f + beta log C."""
    X, y = [], []
    for C in C_grid:
        for f in f_grid:
            for _ in range(reps_per_cell):
                X.append([1.0, np.log(f), np.log(C)])
                y.append(np.log(improvement_rate(C, a, beta, f, theta, rng, noise)))
    coef = np.linalg.lstsq(np.asarray(X), np.asarray(y), rcond=None)[0]
    return {"theta_hat": float(coef[1]), "beta_hat": float(coef[2])}


def recovery_study(beta: float, theta: float, noise: float = 0.05, reps: int = 50,
                   C_grid: Sequence[float] = (1.0, 2.0, 4.0, 8.0, 16.0), f_grid: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
                   reps_per_cell: int = 4, a: float = 0.1, seed: int = 20260905) -> Dict[str, Dict[str, float]]:
    """Repeat both designs and report the mean and standard error of what each recovers."""
    rng = np.random.default_rng(seed)
    ro, cb, ct = [], [], []
    for _ in range(reps):
        ro.append(retention_only_design(4.0, f_grid, a, beta, theta, noise, reps_per_cell, rng)["slope_on_log_f"])
        c = crossed_design(C_grid, f_grid, a, beta, theta, noise, reps_per_cell, rng)
        cb.append(c["beta_hat"]); ct.append(c["theta_hat"])
    return {
        "truth": {"beta": beta, "theta": theta},
        "retention_only": {"slope_mean": float(np.mean(ro)), "slope_se": float(np.std(ro)), "reads_as": "theta, not beta"},
        "crossed": {"beta_mean": float(np.mean(cb)), "beta_se": float(np.std(cb)), "theta_mean": float(np.mean(ct)), "theta_se": float(np.std(ct))},
    }


def sealed_prediction_check(beta_hat: float, C0: float, a_hat: float, R: np.ndarray, observed: np.ndarray,
                            margin: float = 0.10) -> Dict[str, object]:
    """Given a coupling and rate estimated on disjoint records, compare the predicted finite-window slope
    with the slope fitted on the validation trajectory (the registration's P5 agreement test)."""
    predicted = loglog_fit_slope(C0, a_hat, beta_hat, R)
    fitted = float(np.polyfit(np.log(R), np.log(observed), 1)[0])
    return {"predicted_slope": predicted, "fitted_slope": fitted, "difference": predicted - fitted,
            "within_margin": bool(abs(predicted - fitted) <= margin)}


def main(argv: Sequence[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--theta", type=float, default=0.6)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args(argv)
    print("finite-window witness (beta 0.5, C0 1, a 0.1, depths 1 to 32): endpoint slope %.6f, seven-point fit %.6f"
          % (endpoint_slope(1.0, 0.1, 0.5, 1.0, 32.0), loglog_fit_slope(1.0, 0.1, 0.5, np.array([1, 2, 4, 8, 16, 24, 32], float))))
    for label, b, t in (("as registered", a.beta, a.theta), ("negative control: no coupling, retention effect only", 0.0, a.theta)):
        r = recovery_study(b, t, a.noise, a.reps)
        print(label, r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
