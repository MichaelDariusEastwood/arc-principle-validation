"""P1 form discrimination: the full pre-data confusion matrix for a registered ladder.

The registration requires the deciding unit to demonstrate, on its own ladder, point count and
dispersion, the rate at which its model comparison selects a rival when the truth is the power law,
and to register that rate in advance; the reviews require the whole matrix P(selected family | true
family), so that an instrument that never rejects the power law is not mistaken for one that can
detect saturation. This module produces both.

Families: power (c r^a), exponential (c e^{b r}), saturating (c_max r / (K + r)), geometric approach
to a fixed point (c_inf - (c_inf - c_0) lambda^r, the Markov fixed-point shape), and a continuous
broken power law with one change of regime. Fits are least squares on the log scale; selection is by
AICc. Noise is multiplicative log-normal per observation, which is the registration's capability
precision expressed as a dispersion.
"""
from __future__ import annotations

import argparse
import json
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit


def power(r, c, a):
    return c * r ** a


def expo(r, c, b):
    return c * np.exp(b * r)


def satur(r, cmax, k):
    return cmax * r / (k + r)


def geom(r, cinf, c0, lam):
    return cinf - (cinf - c0) * lam ** r


def broken(r, c, a1, a2, rb):
    return np.where(r < rb, c * r ** a1, c * rb ** (a1 - a2) * r ** a2)


def logarithmic(r, c, b):
    """c (1 + b log r): the logarithmic rival the candidate's B2 requires."""
    return c * (1.0 + b * np.log(r))


def shifted(r, c, s, a):
    """c (r + s)^a: the finite-window solution of the registered growth equation, a named candidate
    credited to P5's mechanism and never scored as a power-family win."""
    return c * (r + s) ** a


FAMILIES: Dict[str, Tuple[Callable, Tuple[Sequence[float], Sequence[float]]]] = {
    "power": (power, ([1e-6, -5.0], [1e6, 5.0])),
    "logarithmic": (logarithmic, ([1e-6, 1e-4], [1e6, 1e3])),
    "exponential": (expo, ([1e-6, -2.0], [1e6, 2.0])),
    "saturating": (satur, ([1e-6, 1e-3], [1e9, 1e6])),
    "geometric": (geom, ([1e-6, 0.0, 0.01], [1e9, 1e9, 0.9999])),
    "broken": (broken, ([1e-6, -5.0, -5.0, 1.01], [1e6, 5.0, 5.0, 1e4])),
    "shifted": (shifted, ([1e-6, 0.0, -5.0], [1e6, 1e3, 5.0])),
}

FIT_FAILURE = "FIT FAILURE"


def ladder(points: int, decades: float, r_min: float = 1.0) -> np.ndarray:
    """A log-spaced depth ladder of `points` rounds spanning `decades` decades from r_min."""
    return np.logspace(np.log10(r_min), np.log10(r_min) + decades, points)


def _start(name: str, r: np.ndarray, y: np.ndarray) -> List[float]:
    if name == "power":
        b = np.linalg.lstsq(np.vstack([np.ones_like(r), np.log(r)]).T, np.log(y), rcond=None)[0]
        return [float(np.exp(b[0])), float(b[1])]
    if name == "exponential":
        b = np.linalg.lstsq(np.vstack([np.ones_like(r), r]).T, np.log(y), rcond=None)[0]
        return [float(np.exp(b[0])), float(b[1])]
    if name == "saturating":
        return [float(y.max() * 2), float(np.median(r))]
    if name == "geometric":
        return [float(y.max() * 1.2), float(max(y.min(), 1e-6)), 0.9]
    if name == "logarithmic":
        return [float(y[0]), 0.5]
    if name == "shifted":
        b = np.linalg.lstsq(np.vstack([np.ones_like(r), np.log(r)]).T, np.log(y), rcond=None)[0]
        return [float(np.exp(b[0])), 1.0, float(b[1])]
    return [float(y[0]), 1.0, 0.5, float(np.sqrt(r.min() * r.max()))]


def fit_family(name: str, r: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    """Return (AICc, parameters) for one family on log-scale residuals; inf on failure, and inf where
    the small-sample correction is undefined (n - k - 1 at or below zero), never a substituted
    denominator."""
    f, bounds = FAMILIES[name]
    n = len(r)
    try:
        popt, _ = curve_fit(lambda rr, *p: np.log(f(rr, *p)), r, np.log(y), p0=_start(name, r, y),
                            bounds=bounds, maxfev=20000)
        k = len(popt)
        if n - k - 1 <= 0:
            return float("inf"), popt
        resid = np.log(y) - np.log(f(r, *popt))
        rss = float(np.sum(resid ** 2))
        aicc = n * np.log(rss / n + 1e-300) + 2 * k + (2 * k * (k + 1)) / (n - k - 1)
        return aicc, popt
    except Exception:
        return float("inf"), np.array([])


def select_family(r: np.ndarray, y: np.ndarray, candidates: Sequence[str] = tuple(FAMILIES)) -> str:
    """The family with the smallest AICc; FIT FAILURE where no candidate fitted (never a default winner)."""
    scores = {name: fit_family(name, r, y)[0] for name in candidates}
    best = min(scores, key=scores.get)
    if not np.isfinite(scores[best]):
        return FIT_FAILURE
    return best


DEFAULT_TRUTHS: Dict[str, Tuple[str, Callable[[np.ndarray], np.ndarray]]] = {
    "power a=0.5": ("power", lambda r: power(r, 1.0, 0.5)),
    "power a=1.0": ("power", lambda r: power(r, 1.0, 1.0)),
    "power a=1.5": ("power", lambda r: power(r, 1.0, 1.5)),
    "exponential b=0.06": ("exponential", lambda r: expo(r, 1.0, 0.06)),
    "saturating K=15": ("saturating", lambda r: satur(r, 40.0, 15.0)),
    "geometric lam=0.88": ("geometric", lambda r: geom(r, 20.0, 1.0, 0.88)),
    "broken 1.4 to 0.5 at 8": ("broken", lambda r: broken(r, 1.0, 1.4, 0.5, 8.0)),
    "logarithmic b=0.8": ("logarithmic", lambda r: logarithmic(r, 1.0, 0.8)),
    "shifted s=19 a=2": ("shifted", lambda r: shifted(r, 0.0025, 19.0, 2.0)),
}


def confusion_matrix(points: int = 7, decades: float = 1.5, noise: float = 0.02, reps: int = 200,
                     truths: Dict[str, Tuple[str, Callable]] = None, seed: int = 20260905) -> Dict[str, Dict[str, float]]:
    """P(selected family | generating model) for the design, as nested dicts truth -> family -> share."""
    rng = np.random.default_rng(seed)
    r = ladder(points, decades)
    truths = truths or DEFAULT_TRUTHS
    out: Dict[str, Dict[str, float]] = {}
    for tname, (fam, gen) in truths.items():
        counts = {name: 0 for name in list(FAMILIES) + [FIT_FAILURE]}
        for _ in range(reps):
            y = gen(r) * np.exp(rng.normal(0.0, noise, size=r.shape))
            counts[select_family(r, y)] += 1
        out[tname] = {name: counts[name] / reps for name in list(FAMILIES) + [FIT_FAILURE]}
        out[tname]["_true_family"] = fam  # type: ignore[assignment]
    return out


def false_support_rate(matrix: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """The readiness number the registration needs and the first edition did not report: how often the
    comparison selects the power family when each serious rival is true (false support), per rival
    truth, with the worst case. The false-selection rate under power truth is the other direction; a
    readiness decision uses both cells and their Monte Carlo uncertainty."""
    rows = {t: row["power"] for t, row in matrix.items() if row["_true_family"] != "power"}
    rows["_worst_case"] = max(rows.values()) if rows else float("nan")
    return rows


def recovery_rates(matrix: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Share of replications selecting the true family, per generating model."""
    return {t: row[row["_true_family"]] for t, row in matrix.items()}


def false_selection_rate(matrix: Dict[str, Dict[str, float]], power_key_prefix: str = "power") -> float:
    """The registration's number: how often the comparison prefers a rival when the truth is a power
    law, averaged over the power-law truths in the matrix."""
    rows = [row for t, row in matrix.items() if row["_true_family"] == "power"]
    if not rows:
        return float("nan")
    return float(np.mean([1.0 - row["power"] for row in rows]))


def rival_detection_rates(matrix: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Probability of rejecting the power family when each serious rival is true (the second half of the
    reviews' requirement)."""
    return {t: 1.0 - row["power"] for t, row in matrix.items() if row["_true_family"] != "power"}


def main(argv: Sequence[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--points", type=int, default=7)
    ap.add_argument("--decades", type=float, default=1.5)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--registered-max-false-selection", type=float, default=0.10)
    ap.add_argument("--json", default=None, help="write the matrix and summary here")
    a = ap.parse_args(argv)
    m = confusion_matrix(a.points, a.decades, a.noise, a.reps)
    fsr = false_selection_rate(m)
    fsup = false_support_rate(m)
    summary = {
        "design": {"points": a.points, "decades": a.decades, "noise": a.noise, "reps": a.reps},
        "false_selection_rate_under_power_truth": fsr,
        "false_support_rate_under_each_rival_truth": fsup,
        "registered_max": a.registered_max_false_selection,
        "demonstrably_small": bool(fsr <= a.registered_max_false_selection and fsup["_worst_case"] <= a.registered_max_false_selection),
        "recovery_rates": recovery_rates(m),
        "rival_detection_rates": rival_detection_rates(m),
        "matrix": m,
    }
    print(json.dumps(summary, indent=1))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
