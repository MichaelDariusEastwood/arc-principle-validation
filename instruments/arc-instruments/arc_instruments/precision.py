"""Precision arithmetic for the ceiling chain (P3, P8, P16, P20).

Every number here is a property of a registered design, computed before any data. The ceiling
relation alpha_crit = 1 / (1 - gamma) amplifies uncertainty in the correction-leverage exponent by
1 / (1 - gamma)^2, which is 4 at one half; P3's tolerance of 0.10 on the paired difference therefore
needs the gamma half-width at or below 0.025 there, and the sample scales with the inverse square of
the half-width. The growth exponent's own precision on a log-spaced ladder is the ordinary regression
standard error, sigma / sqrt(sum (x - mean x)^2) in log depth, and is rarely the binding constraint.
"""
from __future__ import annotations

import argparse
from typing import Dict, Sequence, Tuple

import numpy as np

Interval = Tuple[float, float]

# Calibration targets for every deciding instrument's complete decision procedure, over the charter's
# declared grid of the target model and every named rival (ruling 10 of 5 September 2026): the
# one-sided 95 per cent Monte Carlo upper bound on false affirmative verdicts must not exceed
# FALSE_AFFIRMATIVE_MAX, and the lower bound on detection probability at the charter's materially
# separated alternatives must reach DETECTION_PROBABILITY_MIN. Operating targets, never derived values.
FALSE_AFFIRMATIVE_MAX = 0.05
DETECTION_PROBABILITY_MIN = 0.80


def implied_ceiling(gamma: float) -> float:
    """alpha_crit = 1 / (1 - gamma), defined for gamma in [0, 1)."""
    if not (0.0 <= gamma < 1.0):
        raise ValueError("gamma must lie in [0, 1); no finite ceiling otherwise")
    return 1.0 / (1.0 - gamma)


def ceiling_interval(gamma_interval: Interval) -> Interval:
    """Exact image of a gamma interval under the relation (monotone increasing)."""
    lo, hi = gamma_interval
    return implied_ceiling(lo), implied_ceiling(hi)


def ceiling_sensitivity(gamma: float) -> float:
    """d alpha_crit / d gamma = 1 / (1 - gamma)^2."""
    return 1.0 / (1.0 - gamma) ** 2


def gamma_halfwidth_required(gamma: float, tolerance: float) -> float:
    """The largest gamma half-width whose delta-method image stays within the tolerance."""
    return tolerance / ceiling_sensitivity(gamma)


def sample_scale_factor(current_halfwidth: float, required_halfwidth: float) -> float:
    """Multiply the current sample by this to reach the required half-width (half-width scales as
    the inverse square root of the sample)."""
    return (current_halfwidth / required_halfwidth) ** 2


def exponent_se(points: int, decades: float, noise: float, r_min: float = 1.0) -> float:
    """Standard error of a log-log least-squares slope on a log-spaced ladder with multiplicative
    log-normal noise of dispersion `noise` per observation."""
    x = np.log(np.logspace(np.log10(r_min), np.log10(r_min) + decades, points))
    return float(noise / np.sqrt(np.sum((x - x.mean()) ** 2)))


def sign_power(margin: float, se_alpha: float, se_gamma: float, gamma: float = 0.5,
               reps: int = 20000, seed: int = 20260905) -> float:
    """P16: probability of correctly signing the correction-pressure slope alpha (1 - gamma) - 1 when the
    true slope is +margin, with independent normal errors on alpha and gamma. Cells near the boundary
    return INCONCLUSIVE by design; this reports how near is too near."""
    rng = np.random.default_rng(seed)
    alpha_true = (1.0 + margin) / (1.0 - gamma)
    a = alpha_true + rng.normal(0.0, se_alpha, reps)
    g = gamma + rng.normal(0.0, se_gamma, reps)
    return float(np.mean(a * (1.0 - g) - 1.0 > 0.0))


def p3_reachability(gamma: float, tolerance: float, registered_gamma_halfwidth: float) -> Dict[str, float]:
    req = gamma_halfwidth_required(gamma, tolerance)
    return {
        "gamma": gamma,
        "alpha_crit": implied_ceiling(gamma),
        "sensitivity": ceiling_sensitivity(gamma),
        "tolerance": tolerance,
        "gamma_halfwidth_required": req,
        "registered_gamma_halfwidth": registered_gamma_halfwidth,
        "implied_ceiling_halfwidth_at_registered": registered_gamma_halfwidth * ceiling_sensitivity(gamma),
        "sample_scale_factor_needed": sample_scale_factor(registered_gamma_halfwidth, req),
        "reachable_as_registered": bool(registered_gamma_halfwidth <= req),
    }


def interval_clearance_power(margin: float, se_alpha: float, se_gamma: float, gamma: float = 0.5,
                             z: float = 1.96) -> Dict[str, float]:
    """P16's decision is an interval clearing zero, not a point estimate carrying the right sign.
    Under independent normal errors on alpha and gamma with true pressure alpha (1 - gamma) - 1 equal to
    the margin, the pressure estimate has standard deviation sqrt(((1 - gamma) se_alpha)^2 +
    (alpha se_gamma)^2); the point-sign probability is Phi(margin / sd) and the interval-clearance power is
    Phi(margin / sd - z). At alpha 2.2, gamma one half, se_alpha 0.02 and se_gamma 0.07/1.96 the two are
    about 0.90 and 0.24. sign_power reports the first kind of number and must never be called decision
    power."""
    from scipy.stats import norm
    alpha_true = (1.0 + margin) / (1.0 - gamma)
    sd = ((1.0 - gamma) * se_alpha) ** 2 + (alpha_true * se_gamma) ** 2
    sd = sd ** 0.5
    return {"sd_pressure": sd, "point_sign_probability": float(norm.cdf(margin / sd)),
            "interval_clearance_power": float(norm.cdf(margin / sd - z))}


def exact_gamma_interval_for_ceiling_band(lo: float, hi: float) -> Dict[str, float]:
    """The exact set of gamma values whose image under 1/(1 - gamma) lies within [lo, hi], and the largest
    symmetric half-width about one half inside it: for [1.9, 2.1] that is 0.0238, not the first-order
    0.025, and it still ignores every other uncertainty source."""
    g_lo, g_hi = 1.0 - 1.0 / lo, 1.0 - 1.0 / hi
    return {"gamma_lo": g_lo, "gamma_hi": g_hi, "max_symmetric_halfwidth_about_half": min(0.5 - g_lo, g_hi - 0.5)}


def conjunction_power(component_powers: Sequence[float]) -> float:
    """Joint power of a registered conjunction whose component decisions are independent and each
    component hypothesis is true at its alternative: the product. Five components at 0.80 give 0.33; ten
    give 0.11. A design operating characteristic, never a probability that the theory is true."""
    p = 1.0
    for x in component_powers:
        p *= float(x)
    return p


def component_power_for_joint(joint_target: float, n_components: int) -> float:
    """Equal component power needed for a joint target under independence: target^(1/n). About 0.956
    for five components at 0.80."""
    return float(joint_target) ** (1.0 / n_components)


def exceedance_power(true_value: float, se: float, bound: float, z: float = 1.96) -> float:
    """Probability that a normal interval of half-width z * se lies wholly above the bound when the
    true value is `true_value`. For P6, a true 0.55 with standard error 0.04 clears one half about 24
    per cent of the time: low power, not unrefutability."""
    from scipy.stats import norm
    return float(norm.cdf((true_value - bound) / se - z))


def form_discriminating_power(gamma_cells: Sequence[float], gamma_se: float, frontier_se: float,
                              reps: int = 4000, seed: int = 20260905) -> Dict[str, float]:
    """P20's gate as a calibrated quantity rather than a geometric rule.

    Simulate measured pairs (service elasticity, frontier) at the realised cells with the registered
    form 1/(1 - g) true, and again with the withdrawn reciprocal 1/g true, and select the form with the
    smaller residual sum of squares. Reports the probability of selecting the true form under each truth
    and the worst case, which is the discriminating power the registration gates on. Cells at one half
    give chance; cells wholly below one half can give near-certain discrimination (at 0.2 the forms
    predict 1.25 and 5).
    """
    rng = np.random.default_rng(seed)
    g = np.asarray(list(gamma_cells), float)
    if np.any(g <= 0.0) or np.any(g >= 1.0):
        raise ValueError("cells must lie in (0, 1) for both forms to be defined")
    reg = 1.0 / (1.0 - g)
    riv = 1.0 / g

    def select_rate(truth: np.ndarray, want_registered: bool) -> float:
        hits = 0
        for _ in range(reps):
            g_hat = np.clip(g + rng.normal(0.0, gamma_se, g.size), 1e-3, 1 - 1e-3)
            f_hat = truth + rng.normal(0.0, frontier_se, g.size)
            rss_reg = float(np.sum((f_hat - 1.0 / (1.0 - g_hat)) ** 2))
            rss_riv = float(np.sum((f_hat - 1.0 / g_hat) ** 2))
            picked_registered = rss_reg < rss_riv
            hits += int(picked_registered == want_registered)
        return hits / reps

    p_reg = select_rate(reg, True)
    p_riv = select_rate(riv, False)
    return {"power_registered_true": p_reg, "power_rival_true": p_riv, "discriminating_power": min(p_reg, p_riv)}


def main(argv: Sequence[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--tolerance", type=float, default=0.10)
    ap.add_argument("--gamma-halfwidth", type=float, default=0.07)
    ap.add_argument("--points", type=int, default=7)
    ap.add_argument("--decades", type=float, default=1.5)
    ap.add_argument("--noise", type=float, default=0.02)
    a = ap.parse_args(argv)
    r = p3_reachability(a.gamma, a.tolerance, a.gamma_halfwidth)
    for k, v in r.items():
        print("%-42s %s" % (k, v))
    print("%-42s %.4f" % ("exponent_se_on_ladder", exponent_se(a.points, a.decades, a.noise)))
    for m in (0.05, 0.10, 0.20):
        print("%-42s %.3f" % ("p16_sign_power_at_margin_%.2f" % m, sign_power(m, 0.02, a.gamma_halfwidth / 1.96, a.gamma)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
