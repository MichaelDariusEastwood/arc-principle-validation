#!/usr/bin/env python3
"""
================================================================================
ARC PRINCIPLE: 12-DOMAIN PRE-REGISTERED EXTENSION
Cauchy Unification Test (Paper VII)
================================================================================

Purpose:
  Run the locked 12-domain pre-registered extension for the Cauchy unification
  test. Uses EXACTLY the same fitting protocol as the canonical 50-domain suite
  (arc_50_domain_universal_test.py): same candidate models, same AICc selection,
  same family labels, same saturation guard.

Primary endpoint:
  Strict family match -- AICc-best model's family vs pre-registered predicted
  family -- with NO tolerance rescue.

Design:
  - Predictions locked in next_extension_manifest.json BEFORE data extraction.
  - Empirical data sourced from published literature and stored in
    extension_data.json.
  - Misses reported honestly; no retroactive prediction changes.

Statistical test:
  One-sided binomial test against chance baseline of 1/3 (three families:
  power_law, exponential, bounded).

================================================================================
Michael Darius Eastwood | March 2026
================================================================================
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import optimize, stats


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "preregistration" / "next_extension_manifest.json"
DATA_PATH = REPO_ROOT / "preregistration" / "extracted_data" / "extension_data.json"
RESULTS_DIR = REPO_ROOT / "results"
JSON_OUT = RESULTS_DIR / "results_preregistered_extension.json"
TEXT_OUT = RESULTS_DIR / "results_preregistered_extension.txt"


# ---------------------------------------------------------------------------
# Family labels (identical to arc_50_domain_universal_test.py)
# ---------------------------------------------------------------------------
FAMILY_LABELS = {
    "power_law": "power_law",
    "exponential": "exponential",
    "saturation_exp": "bounded",
    "michaelis_menten": "bounded",
    "logistic": "bounded",
    "hill": "bounded",
    "hyperbolic_decay": "bounded",
}


# ---------------------------------------------------------------------------
# Helpers (replicated verbatim from arc_50_domain_universal_test.py)
# ---------------------------------------------------------------------------
def safe_rss_r2(y: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    rss = float(np.sum((y - y_pred) ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    if tss <= 0:
        r2 = 1.0 if rss <= 1e-12 else 0.0
    else:
        r2 = 1.0 - rss / tss
    return rss, r2


def information_criteria(n: int, k: int, rss: float) -> tuple[float, float]:
    if n <= k + 1 or not np.isfinite(rss):
        return float("inf"), float("inf")
    rss_term = max(rss / n, 1e-12)
    aic = n * math.log(rss_term) + 2 * k
    aicc = aic + (2 * k * (k + 1)) / max(n - k - 1, 1)
    bic = n * math.log(rss_term) + k * math.log(n)
    return aicc, bic


def has_meaningful_saturation(y: np.ndarray, asymptote: float, threshold: float = 0.5) -> bool:
    if not np.isfinite(asymptote) or asymptote <= 0:
        return False
    return (float(np.max(y)) / asymptote) >= threshold


def invalid_fit(name: str, family: str, reason: str) -> dict[str, Any]:
    return {
        "name": name,
        "family": family,
        "valid": False,
        "reason": reason,
        "rss": float("inf"),
        "r2": float("-inf"),
        "aicc": float("inf"),
        "bic": float("inf"),
        "params": {},
    }


def valid_fit(
    name: str,
    family: str,
    x: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    params: dict[str, float],
    k: int,
) -> dict[str, Any]:
    rss, r2 = safe_rss_r2(y, y_pred)
    aicc, bic = information_criteria(len(x), k, rss)
    return {
        "name": name,
        "family": family,
        "valid": True,
        "rss": rss,
        "r2": r2,
        "aicc": aicc,
        "bic": bic,
        "params": params,
    }


# ---------------------------------------------------------------------------
# Fitters (identical to arc_50_domain_universal_test.py)
# ---------------------------------------------------------------------------
def fit_power_law(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    mask = (x > 0) & (y > 0)
    if int(mask.sum()) < 3:
        return invalid_fit("power_law", "power_law", "need >= 3 positive points")
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    slope, intercept, _, _, _ = stats.linregress(lx, ly)
    a = float(np.exp(intercept))
    b = float(slope)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        y_pred = a * np.power(x, b)
    if not np.all(np.isfinite(y_pred)):
        return invalid_fit("power_law", "power_law", "non-finite prediction")
    return valid_fit("power_law", "power_law", x, y, y_pred, {"a": a, "b": b}, 2)


def fit_exponential(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    mask = y > 0
    if int(mask.sum()) < 3:
        return invalid_fit("exponential", "exponential", "need >= 3 positive y points")
    ly = np.log(y[mask])
    slope, intercept, _, _, _ = stats.linregress(x[mask], ly)
    a = float(np.exp(intercept))
    b = float(slope)
    y_pred = a * np.exp(b * x)
    return valid_fit("exponential", "exponential", x, y, y_pred, {"a": a, "b": b}, 2)


def fit_saturation_exp(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    try:
        def fn(xv, y_max, k):
            return y_max * (1.0 - np.exp(-k * xv))

        p0 = [max(float(np.max(y)) * 1.05, 1e-6), max(1.0 / (np.mean(np.abs(x)) + 1e-6), 1e-6)]
        popt, _ = optimize.curve_fit(
            fn, x, y, p0=p0, bounds=([0.0, 0.0], [np.inf, np.inf]), maxfev=20000,
        )
        y_pred = fn(x, *popt)
        if not has_meaningful_saturation(y, float(popt[0])):
            return invalid_fit("saturation_exp", "bounded", "data do not approach fitted asymptote")
        return valid_fit(
            "saturation_exp", "bounded", x, y, y_pred,
            {"y_max": float(popt[0]), "k": float(popt[1])}, 2,
        )
    except Exception as exc:
        return invalid_fit("saturation_exp", "bounded", str(exc))


def fit_michaelis_menten(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    try:
        def fn(xv, L, K):
            return L * xv / (K + xv)

        p0 = [max(float(np.max(y)) * 1.05, 1e-6), max(float(np.median(np.abs(x))), 1e-6)]
        popt, _ = optimize.curve_fit(
            fn, x, y, p0=p0, bounds=([0.0, 0.0], [np.inf, np.inf]), maxfev=20000,
        )
        y_pred = fn(x, *popt)
        if not has_meaningful_saturation(y, float(popt[0])):
            return invalid_fit("michaelis_menten", "bounded", "data do not approach fitted asymptote")
        return valid_fit(
            "michaelis_menten", "bounded", x, y, y_pred,
            {"L": float(popt[0]), "K": float(popt[1])}, 2,
        )
    except Exception as exc:
        return invalid_fit("michaelis_menten", "bounded", str(exc))


def fit_logistic(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    try:
        def fn(xv, K, r, x0):
            return K / (1.0 + np.exp(-r * (xv - x0)))

        p0 = [max(float(np.max(y)) * 1.05, 1e-6), 0.1, float(np.median(x))]
        popt, _ = optimize.curve_fit(
            fn, x, y, p0=p0, bounds=([0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=30000,
        )
        y_pred = fn(x, *popt)
        if not has_meaningful_saturation(y, float(popt[0])):
            return invalid_fit("logistic", "bounded", "data do not approach fitted asymptote")
        return valid_fit(
            "logistic", "bounded", x, y, y_pred,
            {"K": float(popt[0]), "r": float(popt[1]), "x0": float(popt[2])}, 3,
        )
    except Exception as exc:
        return invalid_fit("logistic", "bounded", str(exc))


def fit_hill(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    try:
        def fn(xv, y_max, K, n):
            return y_max * np.power(xv, n) / (np.power(K, n) + np.power(xv, n))

        safe_x = np.maximum(x, 0.0)
        p0 = [max(float(np.max(y)) * 1.05, 1e-6), max(float(np.median(np.abs(safe_x))), 1e-6), 2.0]
        popt, _ = optimize.curve_fit(
            fn, safe_x, y, p0=p0, bounds=([0.0, 0.0, 0.1], [np.inf, np.inf, 20.0]),
            maxfev=30000,
        )
        y_pred = fn(safe_x, *popt)
        if not has_meaningful_saturation(y, float(popt[0])):
            return invalid_fit("hill", "bounded", "data do not approach fitted asymptote")
        return valid_fit(
            "hill", "bounded", x, y, y_pred,
            {"y_max": float(popt[0]), "K": float(popt[1]), "n": float(popt[2])}, 3,
        )
    except Exception as exc:
        return invalid_fit("hill", "bounded", str(exc))


def fit_hyperbolic_decay(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    try:
        def fn(xv, a, b):
            return a / (b + xv)

        positive_x = x[x >= 0]
        p0 = [
            max(float(np.max(y)) * (float(np.min(positive_x)) + 1.0), 1e-6),
            1.0,
        ]
        popt, _ = optimize.curve_fit(
            fn, x, y, p0=p0, bounds=([0.0, 0.0], [np.inf, np.inf]), maxfev=20000,
        )
        y_pred = fn(x, *popt)
        return valid_fit(
            "hyperbolic_decay", "bounded", x, y, y_pred,
            {"a": float(popt[0]), "b": float(popt[1])}, 2,
        )
    except Exception as exc:
        return invalid_fit("hyperbolic_decay", "bounded", str(exc))


FITTERS = [
    fit_power_law,
    fit_exponential,
    fit_saturation_exp,
    fit_michaelis_menten,
    fit_logistic,
    fit_hill,
    fit_hyperbolic_decay,
]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_domain(domain_data: dict[str, Any], manifest_entry: dict[str, Any]) -> dict[str, Any]:
    x = np.array(domain_data["x"], dtype=float)
    y = np.array(domain_data["y"], dtype=float)

    predicted_family = manifest_entry["predicted_family"]
    predicted_model = manifest_entry["predicted_model"]

    # Run all fitters (include hyperbolic_decay for completeness)
    fits = [fitter(x, y) for fitter in FITTERS]
    valid_fits = [f for f in fits if f["valid"]]

    if not valid_fits:
        return {
            "extension_id": domain_data["extension_id"],
            "domain_name": domain_data["domain_name"],
            "status": "no_valid_fits",
            "predicted_family": predicted_family,
            "predicted_model": predicted_model,
            "family_match": False,
            "model_match": False,
            "n_points": len(x),
            "source": domain_data["source"],
        }

    ranked = sorted(valid_fits, key=lambda f: (f["aicc"], f["bic"], -f["r2"]))
    best = ranked[0]
    family_match = best["family"] == predicted_family
    model_match = best["name"] == predicted_model

    return {
        "extension_id": domain_data["extension_id"],
        "domain_name": domain_data["domain_name"],
        "operator_class": domain_data["operator_class"],
        "predicted_family": predicted_family,
        "predicted_model": predicted_model,
        "best_model": best["name"],
        "best_family": best["family"],
        "family_match": family_match,
        "model_match": model_match,
        "best_r2": best["r2"],
        "best_aicc": best["aicc"],
        "n_points": len(x),
        "source": domain_data["source"],
        "top_fits": [
            {
                "name": f["name"],
                "family": f["family"],
                "r2": f["r2"],
                "aicc": f["aicc"],
                "bic": f["bic"],
                "params": f["params"],
            }
            for f in ranked[:4]
        ],
    }


def binomial_pvalue(k: int, n: int, p: float = 1 / 3) -> float:
    if n <= 0:
        return 1.0
    return float(stats.binomtest(k, n, p=p, alternative="greater").pvalue)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load manifest (locked predictions)
    manifest = json.loads(MANIFEST_PATH.read_text())
    manifest_domains = {d["extension_id"]: d for d in manifest["domains"]}

    # Load extracted empirical data
    data_file = json.loads(DATA_PATH.read_text())
    data_domains = data_file["domains"]

    # Evaluate each domain
    results = []
    for dd in data_domains:
        eid = dd["extension_id"]
        me = manifest_domains[eid]
        result = evaluate_domain(dd, me)
        results.append(result)

    # Tally
    total = len(results)
    family_matches = sum(1 for r in results if r.get("family_match"))
    model_matches = sum(1 for r in results if r.get("model_match"))
    p_value = binomial_pvalue(family_matches, total)

    # Build output lines
    lines = []
    lines.append("=" * 100)
    lines.append("  ARC PRINCIPLE: 12-DOMAIN PRE-REGISTERED EXTENSION")
    lines.append("  Cauchy Unification Test (Paper VII)")
    lines.append("=" * 100)
    lines.append(f"  Date:     {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"  Manifest: {MANIFEST_PATH}")
    lines.append(f"  Data:     {DATA_PATH}")
    lines.append(f"  Domains:  {total}")
    lines.append(f"  Primary endpoint: Strict family match (AICc best), no tolerance rescue")
    lines.append(f"  Baseline: 1/3 (three families: power_law, exponential, bounded)")
    lines.append("")
    lines.append("=" * 100)
    lines.append(f"  {'ID':<8} {'Domain':<52} {'Predicted':<12} {'Best model':<18} {'Family':<12} {'Match'}")
    lines.append("-" * 100)

    for r in results:
        match_str = "YES" if r.get("family_match") else "NO"
        best_model = r.get("best_model", "n/a")
        best_family = r.get("best_family", "n/a")
        lines.append(
            f"  {r['extension_id']:<8} {r['domain_name']:<52} "
            f"{r['predicted_family']:<12} {best_model:<18} {best_family:<12} {match_str}"
        )

    lines.append("-" * 100)
    lines.append("")
    lines.append("=" * 100)
    lines.append("  RESULTS")
    lines.append("=" * 100)
    lines.append(f"  Strict family match:  {family_matches}/{total}")
    lines.append(f"  Exact model match:    {model_matches}/{total}")
    lines.append(f"  Binomial p-value:     {p_value:.6e}")
    lines.append(f"    (one-sided, H0: p <= 1/3, H1: p > 1/3)")
    lines.append("")

    if family_matches == total:
        lines.append(f"  PASS: All {total} domains match predicted family.")
    elif p_value < 0.05:
        lines.append(f"  PASS: {family_matches}/{total} family matches, p = {p_value:.3e} < 0.05.")
    else:
        lines.append(f"  INCONCLUSIVE or FAIL: {family_matches}/{total} family matches, p = {p_value:.3e}.")

    lines.append("")

    # Misses detail
    misses = [r for r in results if not r.get("family_match")]
    if misses:
        lines.append("  MISSES:")
        for m in misses:
            lines.append(
                f"    {m['extension_id']} {m['domain_name']}: "
                f"predicted {m['predicted_family']}, got {m.get('best_family', 'n/a')} "
                f"(best model: {m.get('best_model', 'n/a')})"
            )
        lines.append("")

    # Print to stdout
    output_text = "\n".join(lines)
    print(output_text)

    # Write text output
    TEXT_OUT.write_text(output_text + "\n")
    print(f"  Text results written to {TEXT_OUT}")

    # Write JSON output
    payload = {
        "metadata": {
            "test_name": "12-domain pre-registered extension",
            "paper": "Paper VII (Cauchy unification)",
            "date": datetime.now(timezone.utc).isoformat(),
            "manifest_path": str(MANIFEST_PATH),
            "data_path": str(DATA_PATH),
            "primary_endpoint": "strict_family_match",
            "baseline_rate": 1 / 3,
            "n_families": 3,
            "fitting_protocol": "Same as arc_50_domain_universal_test.py: AICc-ranked, 7 candidate models, saturation guard",
        },
        "summary": {
            "total_domains": total,
            "family_matches": family_matches,
            "model_matches": model_matches,
            "family_match_rate": family_matches / total if total > 0 else 0,
            "p_value_binomial": p_value,
            "pass": p_value < 0.05,
        },
        "results": results,
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  JSON results written to {JSON_OUT}")


if __name__ == "__main__":
    main()
