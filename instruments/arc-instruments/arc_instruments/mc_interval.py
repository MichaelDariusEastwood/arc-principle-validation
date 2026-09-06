"""Monte Carlo uncertainty for a reported operating characteristic, and the two counts a reader must
be able to tell apart.

WHY. A rate such as "REFUTED in 0 of 40 runs" is itself an estimate from a finite simulation, and the
programme's registrations quote such rates as if they were the instrument's property. A reader outside
the programme put the arithmetic plainly on 5 September 2026: zero false decisions in 400 independent
runs of one fixed world gives an exact one-sided 95 per cent upper bound of about 0.75 per cent, which
is encouraging and is not zero risk and says nothing about untested worlds; and twenty in 400 is an
observed 5 per cent whose upper bound is about 7.2 per cent, so observing 5 per cent does not certify a
true rate at or below 5 per cent. Every rate the battery reports carries its interval from here.

THE TWO COUNTS. A battery has OUTER repetitions (independent simulated experiments, one seed each)
and, inside each, INNER resamples (the bootstrap draws the analysis itself uses for its intervals).
More outer repetitions sharpen the rate; they do not by themselves make an inner bootstrap of few draws
give stable tail quantiles. The report names both and never lets one stand in for the other.

The bounds are exact binomial (Clopper and Pearson), computed by bisection on the regularised
incomplete beta so that this module needs nothing beyond the standard library. Wilson's score interval
is included for the two-sided case because it is what the estate's registrations already quote.
"""
from __future__ import annotations

import math
from statistics import NormalDist
from dataclasses import dataclass
from typing import Dict, Optional


def _betacf(a: float, b: float, x: float) -> float:
    # continued fraction for the incomplete beta function (Numerical Recipes form)
    MAXIT, EPS, FPMIN = 300, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN: d = FPMIN
    d = 1.0 / d; h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; d = FPMIN if abs(d) < FPMIN else d
        c = 1.0 + aa / c; c = FPMIN if abs(c) < FPMIN else c
        d = 1.0 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; d = FPMIN if abs(d) < FPMIN else d
        c = 1.0 + aa / c; c = FPMIN if abs(c) < FPMIN else c
        d = 1.0 / d; de = d * c; h *= de
        if abs(de - 1.0) < EPS: break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    bt = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0: return 0.0
    if k >= n: return 1.0
    return betainc(n - k, k + 1, 1.0 - p)


def _solve(f, lo: float, hi: float, target: float, iters: int = 200) -> float:
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) < target: hi = mid
        else: lo = mid
    return 0.5 * (lo + hi)


def _validate(k, n, conf):
    if any(isinstance(v,bool) or not isinstance(v,int) for v in (k,n)) or not 0 <= k <= n:
        raise ValueError("counts must be integers with 0 <= successes <= repetitions")
    if not math.isfinite(conf) or not 0 < conf < 1:
        raise ValueError("confidence must lie strictly between zero and one")


def exact_upper(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided exact upper bound on p given k successes in n: the largest p with P(X <= k) >= 1 - conf."""
    _validate(k,n,conf)
    if n <= 0: return float("nan")
    if k >= n: return 1.0
    return _solve(lambda p: binom_cdf(k, n, p), 0.0, 1.0, 1.0 - conf)


def exact_lower(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided exact lower bound on p: the smallest p with P(X >= k) >= 1 - conf."""
    _validate(k,n,conf)
    if n <= 0: return float("nan")
    if k <= 0: return 0.0
    return _solve_lower(k, n, conf)


def _solve_lower(k: int, n: int, conf: float) -> float:
    # the smallest p such that P(X >= k | p) >= 1 - conf; P(X >= k) rises with p
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 1.0 - binom_cdf(k - 1, n, mid) < 1.0 - conf: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson(k: int, n: int, conf: float = 0.95):
    """Two-sided exact interval at `conf` (each tail at (1 - conf) / 2)."""
    tail = 1.0 - (1.0 - conf) / 2.0
    return exact_lower(k, n, tail), exact_upper(k, n, tail)


def wilson(k: int, n: int, z: float = 1.959963984540054):
    _validate(k,n,.95)
    if not math.isfinite(z) or z<=0: raise ValueError("z must be positive and finite")
    if n <= 0: return float("nan"), float("nan")
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class RateWithUncertainty:
    successes: int
    outer_repetitions: int
    inner_resamples: Optional[int]
    conf: float = 0.95

    def __post_init__(self):
        _validate(self.successes,self.outer_repetitions,self.conf)
        if self.inner_resamples is not None and (isinstance(self.inner_resamples,bool)
                or not isinstance(self.inner_resamples,int) or self.inner_resamples<1):
            raise ValueError("inner resamples must be a positive integer or None")

    @property
    def rate(self) -> float:
        return self.successes / self.outer_repetitions if self.outer_repetitions else float("nan")

    def as_dict(self) -> Dict[str, object]:
        lo, hi = clopper_pearson(self.successes, self.outer_repetitions, self.conf)
        wlo, whi = wilson(self.successes, self.outer_repetitions,NormalDist().inv_cdf((1+self.conf)/2))
        return {"rate": self.rate, "successes": self.successes, "outer_repetitions": self.outer_repetitions,
                "inner_resamples_per_run": self.inner_resamples,
                "exact_two_sided": [lo, hi], "exact_one_sided_upper": exact_upper(self.successes, self.outer_repetitions, self.conf),
                "exact_one_sided_lower": exact_lower(self.successes, self.outer_repetitions, self.conf),
                "wilson_two_sided": [wlo, whi], "conf": self.conf,
                "note": "the rate is an estimate from the outer repetitions; the inner resamples belong to the "
                        "analysis inside each run and do not narrow this interval"}

    def render(self) -> str:
        d = self.as_dict()
        return "%d of %d (%.3f), exact %.1f%% [%.3f, %.3f], one-sided upper %.4f; inner resamples per run: %s" % (
            self.successes, self.outer_repetitions, self.rate, self.conf*100, d["exact_two_sided"][0], d["exact_two_sided"][1],
            d["exact_one_sided_upper"], self.inner_resamples if self.inner_resamples is not None else "n/a")
