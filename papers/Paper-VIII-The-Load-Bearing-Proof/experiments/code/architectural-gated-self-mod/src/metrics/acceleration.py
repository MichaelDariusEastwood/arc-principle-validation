from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import numpy as np


@dataclass
class SeriesAcceleration:
    deltas: List[float]
    second_deltas: List[float]
    mean_delta: float
    mean_second_delta: float
    normalized_mean_second_delta: float
    auc: float


def compute_series_acceleration(values: List[float], compute: List[float]) -> SeriesAcceleration:
    arr = np.asarray(values, dtype=float)
    comp = np.asarray(compute, dtype=float)
    if arr.size < 2:
        return SeriesAcceleration([], [], 0.0, 0.0, 0.0, float(np.trapz(arr)) if arr.size else 0.0)
    deltas = np.diff(arr)
    second = np.diff(deltas) if deltas.size >= 2 else np.asarray([], dtype=float)
    compute_norm = np.diff(comp) if comp.size >= 2 else np.ones_like(deltas)
    compute_norm = np.where(compute_norm <= 0, 1.0, compute_norm)
    norm_second = second / np.where(np.diff(comp[1:]) <= 0, 1.0, np.diff(comp[1:])) if second.size else np.asarray([], dtype=float)
    return SeriesAcceleration(
        deltas=deltas.tolist(),
        second_deltas=second.tolist(),
        mean_delta=float(np.mean(deltas)) if deltas.size else 0.0,
        mean_second_delta=float(np.mean(second)) if second.size else 0.0,
        normalized_mean_second_delta=float(np.mean(norm_second)) if norm_second.size else 0.0,
        auc=float(np.trapz(arr)),
    )
