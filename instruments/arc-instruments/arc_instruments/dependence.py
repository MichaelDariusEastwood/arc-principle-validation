"""P17: pairwise joint misses against a conditional-independence baseline on a frozen fault set.

The operative claim is pairwise: within a corrector panel named in advance on a typed, frozen fault
population, pairwise joint misses are practically equivalent to the conditional-independence baseline
under the registered equivalence margin (0.20 on the log scale), Bonferroni over pairs, with the
sensitivity demonstrated. Two refinements the reviews required are built in. Fault difficulty is a
conditioning variable frozen before analysis: under conditional independence given difficulty, the
marginal joint-miss rate exceeds the product of marginal miss rates because E[p(d)^2] is at least
E[p(d)]^2, so an unstratified baseline manufactures "correlated blind spots" out of shared item
difficulty. And a material departure on either side contradicts practical independence: negative
dependence has its own branch. Nothing here locates a ceiling.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .verdicts import ANTI_CORRELATED, CORRELATED, INDEPENDENT, INSUFFICIENT_PRECISION, p17_pairwise


def simulate_panel(n_faults: int = 600, n_correctors: int = 4, difficulty_sd: float = 1.5,
                   base_logit: float = -1.0, difficulty_slope: float = 1.2, common_shock_sd: float = 0.0,
                   seed: int = 20260905) -> Tuple[np.ndarray, np.ndarray]:
    """Miss matrix (faults x correctors) and the latent difficulty per fault. Misses are conditionally
    independent given difficulty when common_shock_sd is zero; a positive common shock adds genuine
    dependence beyond difficulty."""
    rng = np.random.default_rng(seed)
    d = rng.normal(0.0, difficulty_sd, n_faults)
    offsets = rng.normal(0.0, 0.3, n_correctors)
    shock = rng.normal(0.0, common_shock_sd, n_faults) if common_shock_sd > 0 else np.zeros(n_faults)
    logits = base_logit + difficulty_slope * d[:, None] + offsets[None, :] + shock[:, None]
    p = 1.0 / (1.0 + np.exp(-logits))
    misses = (rng.random((n_faults, n_correctors)) < p).astype(int)
    return misses, d


def _baseline_joint(mi: np.ndarray, mj: np.ndarray, strata: Optional[np.ndarray]) -> float:
    """Expected joint-miss rate under conditional independence: within each stratum the product of the
    two marginal miss rates, averaged over faults."""
    if strata is None:
        return float(mi.mean() * mj.mean())
    total = 0.0
    for s in np.unique(strata):
        idx = strata == s
        total += idx.sum() * mi[idx].mean() * mj[idx].mean()
    return float(total / len(mi))


def pairwise_dependence(misses: np.ndarray, strata: Optional[np.ndarray] = None, margin: float = 0.20,
                        n_boot: int = 1000, alpha: float = 0.05, bonferroni: bool = True,
                        seed: int = 20260905, scale: str = "log_ratio") -> Dict[str, object]:
    """Per pair: observed joint-miss rate, the conditional-independence baseline, the departure on the
    chosen scale with a bootstrap interval over faults (Bonferroni-adjusted across pairs where
    requested), and the P17 verdict. Also the panel reading, which is read no higher than its pairs.

    Two scales, one primary per registration. `scale="excess"` is the candidate's estimand: the excess
    joint-miss probability e_ij = P(both miss) minus the conditional-independence baseline, in
    probability units, with the margin in the same units; it needs no logarithm and keeps strong negative
    dependence informative. `scale="log_ratio"` is v1.92's 0.20 on the log scale, which discards
    bootstrap draws with zero joint misses or a zero baseline and so can make strong negative-dependence
    cases uninformative by construction. Both are reported; the registration's frozen text says which
    decides, and the other is a named secondary diagnostic. The margin passed in must be on the chosen
    scale.
    """
    if scale not in ("log_ratio", "excess"):
        raise ValueError("scale must be 'log_ratio' or 'excess'")
    n_faults, n_corr = misses.shape
    pairs = list(combinations(range(n_corr), 2))
    a = alpha / len(pairs) if bonferroni and pairs else alpha
    rng = np.random.default_rng(seed)
    out: Dict[str, object] = {"pairs": {}, "margin": margin, "alpha_per_pair": a, "stratified": strata is not None, "scale": scale}
    for i, j in pairs:
        mi, mj = misses[:, i], misses[:, j]
        obs = float(np.mean(mi * mj)); base = _baseline_joint(mi, mj, strata)
        boots = []
        dropped = 0
        for _ in range(n_boot):
            idx = rng.integers(0, n_faults, n_faults)
            bi, bj = mi[idx], mj[idx]
            bs = strata[idx] if strata is not None else None
            o = np.mean(bi * bj); b = _baseline_joint(bi, bj, bs)
            if scale == "excess":
                boots.append(o - b)
            elif o > 0 and b > 0:
                boots.append(np.log(o / b))
            else:
                dropped += 1
        if len(boots) < 10:
            verdict = INSUFFICIENT_PRECISION; iv = (float("nan"), float("nan"))
        else:
            iv = (float(np.quantile(boots, a / 2)), float(np.quantile(boots, 1 - a / 2)))
            verdict = p17_pairwise(iv, margin)
        out["pairs"]["%d-%d" % (i, j)] = {"observed_joint": obs, "baseline_joint": base,
                                          "excess": obs - base,
                                          "log_ratio": float(np.log(obs / base)) if obs > 0 and base > 0 else float("nan"),
                                          "interval": iv, "verdict": verdict, "bootstrap_draws_dropped": dropped}
    vs = [p["verdict"] for p in out["pairs"].values()]
    if any(v in (CORRELATED, ANTI_CORRELATED) for v in vs):
        out["panel"] = "REFUTED (a pair departs from practical independence)"
    elif vs and all(v == INDEPENDENT for v in vs):
        out["panel"] = "SUPPORTED (every primary pair practically independent)"
    else:
        out["panel"] = INSUFFICIENT_PRECISION
    return out


def difficulty_strata(d: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Frozen difficulty bins by quantile; in a real unit the bins come from the fault set's
    pre-registered difficulty labels, never from the misses."""
    q = np.quantile(d, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(q, d)


def main(argv: Sequence[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--n-faults", type=int, default=600)
    a = ap.parse_args(argv)
    misses, d = simulate_panel(n_faults=a.n_faults)
    un = pairwise_dependence(misses, None, n_boot=300)
    st = pairwise_dependence(misses, difficulty_strata(d), n_boot=300)
    print("conditionally independent given difficulty, unstratified baseline:", un["panel"])
    for k, v in un["pairs"].items():
        print("  pair", k, "log ratio %.3f" % v["log_ratio"], v["verdict"])
    print("same misses, baseline stratified by frozen difficulty:", st["panel"])
    for k, v in st["pairs"].items():
        print("  pair", k, "log ratio %.3f" % v["log_ratio"], v["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
