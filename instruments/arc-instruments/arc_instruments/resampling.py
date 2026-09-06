"""Cluster and paired-history resampling.

A bootstrap that resamples rows treats every row as its own laboratory. Tasks repeated across depths are
paired histories, trajectories sit inside systems, and judges sit inside families; resampling rows breaks
that structure and understates uncertainty. The released v5 analysis resampled rows (exact-file audit of
5 September 2026, 7.3); the deciding instruments resample the unit that was actually sampled.
"""
from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np


def cluster_bootstrap_indices(cluster_ids: Sequence, rng: np.random.Generator) -> np.ndarray:
    """Resample whole clusters with replacement and return the concatenated row indices."""
    ids = np.asarray(cluster_ids)
    uniq = np.unique(ids)
    members = {c: np.where(ids == c)[0] for c in uniq}
    draw = rng.choice(uniq, size=uniq.size, replace=True)
    return np.concatenate([members[c] for c in draw])


def cluster_bootstrap(stat_fn: Callable[[np.ndarray], float], values: Sequence[float], cluster_ids: Sequence,
                      n_boot: int = 1000, seed: int = 0, level: float = 0.95) -> Tuple[float, float]:
    """Percentile interval for stat_fn(values) resampling clusters, not rows."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = cluster_bootstrap_indices(cluster_ids, rng)
        stats[i] = stat_fn(v[idx])
    a = (1.0 - level) / 2.0
    return float(np.quantile(stats, a)), float(np.quantile(stats, 1.0 - a))


def row_bootstrap(stat_fn: Callable[[np.ndarray], float], values: Sequence[float], n_boot: int = 1000, seed: int = 0,
                  level: float = 0.95) -> Tuple[float, float]:
    """The naive row bootstrap, kept for comparison; never the primary procedure where rows are clustered."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, v.size, v.size)
        stats[i] = stat_fn(v[idx])
    a = (1.0 - level) / 2.0
    return float(np.quantile(stats, a)), float(np.quantile(stats, 1.0 - a))


def paired_history_bootstrap(stat_fn: Callable[[np.ndarray], float], values: Sequence[float], item_ids: Sequence,
                             n_boot: int = 1000, seed: int = 0, level: float = 0.95) -> Tuple[float, float]:
    """Resample items with all their depth checkpoints together (the paired history is the unit)."""
    return cluster_bootstrap(stat_fn, values, item_ids, n_boot=n_boot, seed=seed, level=level)
