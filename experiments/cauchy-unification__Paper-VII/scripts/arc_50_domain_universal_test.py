#!/usr/bin/env python3
"""
================================================================================
ARC PRINCIPLE: TIERED 50-DOMAIN VALIDATION SUITE
================================================================================

Purpose:
  Replace the permissive headline logic of the legacy 20-domain script with a
  stricter, manifest-driven validation harness and expand the suite to 50
  domains using the best local material currently available.

Key design changes:
  1. Predictions and datasets live in a JSON manifest, not inline in code.
  2. The primary endpoint is only the empirical curve-fit cohort.
  3. Empirical domains are scored by strict family match with no tolerance
     rescue, using AICc-selected best model.
  4. Bounded sub-families are actually fitted, including hyperbolic decay and
     Michaelis-Menten style saturation.
  5. Broader domains are retained, but split into explicit evidence tiers.

Evidence tiers:
  - empirical_curve_fit
  - published_exponent_direct
  - published_exponent_provisional
  - analytic_identity

Primary endpoint:
  Strict family match on empirical_curve_fit domains only.

================================================================================
Michael Darius Eastwood | Codex hardening pass | March 2026
================================================================================
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import optimize, stats


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "canonical_50_domain_manifest.json"
DEFAULT_JSON_OUT = REPO_ROOT / "results" / "results_50_domain_validation.json"


FAMILY_LABELS = {
    "power_law": "power_law",
    "exponential": "exponential",
    "saturation_exp": "bounded",
    "michaelis_menten": "bounded",
    "logistic": "bounded",
    "hill": "bounded",
    "hyperbolic_decay": "bounded",
}


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
            fn,
            x,
            y,
            p0=p0,
            bounds=([0.0, 0.0], [np.inf, np.inf]),
            maxfev=20000,
        )
        y_pred = fn(x, *popt)
        if not has_meaningful_saturation(y, float(popt[0])):
            return invalid_fit("saturation_exp", "bounded", "data do not approach fitted asymptote")
        return valid_fit(
            "saturation_exp",
            "bounded",
            x,
            y,
            y_pred,
            {"y_max": float(popt[0]), "k": float(popt[1])},
            2,
        )
    except Exception as exc:
        return invalid_fit("saturation_exp", "bounded", str(exc))


def fit_michaelis_menten(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    try:
        def fn(xv, L, K):
            return L * xv / (K + xv)

        p0 = [max(float(np.max(y)) * 1.05, 1e-6), max(float(np.median(np.abs(x))), 1e-6)]
        popt, _ = optimize.curve_fit(
            fn,
            x,
            y,
            p0=p0,
            bounds=([0.0, 0.0], [np.inf, np.inf]),
            maxfev=20000,
        )
        y_pred = fn(x, *popt)
        if not has_meaningful_saturation(y, float(popt[0])):
            return invalid_fit("michaelis_menten", "bounded", "data do not approach fitted asymptote")
        return valid_fit(
            "michaelis_menten",
            "bounded",
            x,
            y,
            y_pred,
            {"L": float(popt[0]), "K": float(popt[1])},
            2,
        )
    except Exception as exc:
        return invalid_fit("michaelis_menten", "bounded", str(exc))


def fit_logistic(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    try:
        def fn(xv, K, r, x0):
            return K / (1.0 + np.exp(-r * (xv - x0)))

        p0 = [max(float(np.max(y)) * 1.05, 1e-6), 0.1, float(np.median(x))]
        popt, _ = optimize.curve_fit(
            fn,
            x,
            y,
            p0=p0,
            bounds=([0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=30000,
        )
        y_pred = fn(x, *popt)
        if not has_meaningful_saturation(y, float(popt[0])):
            return invalid_fit("logistic", "bounded", "data do not approach fitted asymptote")
        return valid_fit(
            "logistic",
            "bounded",
            x,
            y,
            y_pred,
            {"K": float(popt[0]), "r": float(popt[1]), "x0": float(popt[2])},
            3,
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
            fn,
            safe_x,
            y,
            p0=p0,
            bounds=([0.0, 0.0, 0.1], [np.inf, np.inf, 20.0]),
            maxfev=30000,
        )
        y_pred = fn(safe_x, *popt)
        if not has_meaningful_saturation(y, float(popt[0])):
            return invalid_fit("hill", "bounded", "data do not approach fitted asymptote")
        return valid_fit(
            "hill",
            "bounded",
            x,
            y,
            y_pred,
            {"y_max": float(popt[0]), "K": float(popt[1]), "n": float(popt[2])},
            3,
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
            fn,
            x,
            y,
            p0=p0,
            bounds=([0.0, 0.0], [np.inf, np.inf]),
            maxfev=20000,
        )
        y_pred = fn(x, *popt)
        return valid_fit(
            "hyperbolic_decay",
            "bounded",
            x,
            y,
            y_pred,
            {"a": float(popt[0]), "b": float(popt[1])},
            2,
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


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def build_analytic_dataset(spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    generator = spec["generator"]
    params = spec.get("params", {})

    if generator == "emc2":
        x = np.logspace(params.get("min_exp", -3), params.get("max_exp", 3), params.get("num", 30))
        c = params.get("c", 3.0e8)
        y = x * (c**2)
        return x, y

    if generator == "hydrogen_levels":
        x = np.arange(1, params.get("max_n", 20) + 1, dtype=float)
        scale = params.get("scale", 13.6)
        exponent = params.get("exponent", 2.0)
        y = scale / np.power(x, exponent)
        return x, y

    if generator == "arrhenius":
        T = np.linspace(params.get("t_min", 300.0), params.get("t_max", 800.0), params.get("num", 50))
        ea = params.get("ea", 50000.0)
        gas_constant = params.get("gas_constant", 8.314)
        prefactor = params.get("prefactor", 1.0e10)
        x = 1000.0 / T
        y = prefactor * np.exp(-ea / (gas_constant * T))
        return x, y

    if generator == "michaelis_menten":
        x = np.linspace(params.get("s_min", 0.1), params.get("s_max", 50.0), params.get("num", 50))
        vmax = params.get("vmax", 100.0)
        km = params.get("km", 5.0)
        y = vmax * x / (km + x)
        return x, y

    if generator == "matter_era":
        x = np.linspace(params.get("t_min", 0.01), params.get("t_max", 1.0), params.get("num", 100))
        exponent = params.get("exponent", 2.0 / 3.0)
        y = np.power(x, exponent)
        return x, y

    if generator == "dark_energy":
        x = np.linspace(params.get("t_min", 0.01), params.get("t_max", 1.0), params.get("num", 100))
        H = params.get("H", 1.0)
        y = np.exp(H * x)
        return x, y

    raise ValueError(f"Unsupported analytic generator: {generator}")


def dataset_for_domain(domain: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if domain["evidence_tier"] == "analytic_identity":
        return build_analytic_dataset(domain["dataset"])

    dataset = domain["dataset"]
    return np.array(dataset["x"], dtype=float), np.array(dataset["y"], dtype=float)


def evaluate_curve_domain(domain: dict[str, Any]) -> dict[str, Any]:
    x, y = dataset_for_domain(domain)
    active_fitters = []
    allow_hyperbolic = domain.get("predicted_model") == "hyperbolic_decay" or domain.get("allow_hyperbolic", False)
    for fitter in FITTERS:
        if fitter.__name__ == "fit_hyperbolic_decay" and not allow_hyperbolic:
            continue
        active_fitters.append(fitter)

    fits = [fitter(x, y) for fitter in active_fitters]
    valid_fits = [fit for fit in fits if fit["valid"]]
    if not valid_fits:
        return {
            "domain_id": domain["id"],
            "name": domain["name"],
            "tier": domain["evidence_tier"],
            "status": "invalid",
            "reason": "no valid fits",
        }

    ranked = sorted(valid_fits, key=lambda item: (item["aicc"], item["bic"], -item["r2"]))
    best = ranked[0]
    family_match = best["family"] == domain["predicted_family"]

    predicted_model = domain.get("predicted_model")
    model_match = best["name"] == predicted_model if predicted_model else None

    return {
        "domain_id": domain["id"],
        "name": domain["name"],
        "tier": domain["evidence_tier"],
        "operator_class": domain["operator_class"],
        "predicted_family": domain["predicted_family"],
        "predicted_model": predicted_model,
        "best_model": best["name"],
        "best_family": best["family"],
        "family_match": family_match,
        "model_match": model_match,
        "n_points": int(len(x)),
        "source": domain["source"],
        "top_fits": [
            {
                "name": fit["name"],
                "family": fit["family"],
                "r2": fit["r2"],
                "aicc": fit["aicc"],
                "bic": fit["bic"],
                "params": fit["params"],
            }
            for fit in ranked[:4]
        ],
    }


def evaluate_published_exponent_domain(domain: dict[str, Any]) -> dict[str, Any]:
    observed = float(domain["published_exponent"])
    predicted = float(domain["predicted_exponent"])
    comparators = domain["comparison_exponents"]
    ranked = sorted(
        [
            {
                "label": item["label"],
                "value": float(item["value"]),
                "abs_error": abs(observed - float(item["value"])),
            }
            for item in comparators
        ],
        key=lambda item: item["abs_error"],
    )
    nearest = ranked[0]
    predicted_label = domain["predicted_label"]
    nearest_match = nearest["label"] == predicted_label

    se = domain.get("published_se")
    ci_includes_prediction = None
    ci_low = None
    ci_high = None
    if se is not None:
        ci_low = observed - 1.96 * float(se)
        ci_high = observed + 1.96 * float(se)
        ci_includes_prediction = ci_low <= predicted <= ci_high

    return {
        "domain_id": domain["id"],
        "name": domain["name"],
        "tier": domain["evidence_tier"],
        "operator_class": domain["operator_class"],
        "predicted_family": domain["predicted_family"],
        "predicted_label": predicted_label,
        "predicted_exponent": predicted,
        "published_exponent": observed,
        "published_se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_includes_prediction": ci_includes_prediction,
        "nearest_comparator": nearest,
        "nearest_match": nearest_match,
        "transport_status": domain.get("transport_status"),
        "source": domain["source"],
        "ranked_comparators": ranked,
    }


def evaluate_domain(domain: dict[str, Any]) -> dict[str, Any]:
    if domain["evidence_tier"] in {"empirical_curve_fit", "analytic_identity"}:
        return evaluate_curve_domain(domain)
    if domain["evidence_tier"] in {"published_exponent_direct", "published_exponent_provisional"}:
        return evaluate_published_exponent_domain(domain)
    raise ValueError(f"Unsupported evidence tier: {domain['evidence_tier']}")


def count_if(results: list[dict[str, Any]], key: str, value: Any = True) -> tuple[int, int]:
    subset = [item for item in results if key in item and item[key] is not None]
    hits = sum(1 for item in subset if item[key] == value)
    return hits, len(subset)


def binomial_pvalue(k: int, n: int, p: float = 1 / 3) -> float | None:
    if n <= 0:
        return None
    return stats.binomtest(k, n, p=p, alternative="greater").pvalue


def print_curve_table(title: str, results: list[dict[str, Any]]) -> None:
    print()
    print("=" * 100)
    print(f"  {title}")
    print("=" * 100)
    print(f"{'ID':>3} {'Domain':<42} {'Predicted':<12} {'Best model':<18} {'Best family':<12} {'Match':<5}")
    print("-" * 100)
    for item in results:
        match = "YES" if item.get("family_match") else "NO"
        print(
            f"{item['domain_id']:>3} {item['name']:<42} "
            f"{item['predicted_family']:<12} {item['best_model']:<18} "
            f"{item['best_family']:<12} {match:<5}"
        )


def print_exponent_table(title: str, results: list[dict[str, Any]]) -> None:
    print()
    print("=" * 120)
    print(f"  {title}")
    print("=" * 120)
    print(
        f"{'ID':>3} {'Domain':<34} {'Pred':>7} {'Obs':>7} {'Nearest':<10} "
        f"{'Nearest?':<8} {'CI has pred?':<12} {'Status'}"
    )
    print("-" * 120)
    for item in results:
        ci_value = (
            "n/a"
            if item["ci_includes_prediction"] is None
            else ("yes" if item["ci_includes_prediction"] else "no")
        )
        status = item.get("transport_status", "")
        print(
            f"{item['domain_id']:>3} {item['name']:<34} {item['predicted_exponent']:>7.3f} "
            f"{item['published_exponent']:>7.3f} {item['nearest_comparator']['label']:<10} "
            f"{'yes' if item['nearest_match'] else 'no':<8} {ci_value:<12} {status}"
        )


def print_summary(summary: dict[str, Any]) -> None:
    print()
    print("=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    print()
    print("  Primary endpoint")
    print(
        f"    Empirical curve-fit family match: {summary['empirical_curve_fit']['matches']}/"
        f"{summary['empirical_curve_fit']['total']} "
        f"(p = {summary['empirical_curve_fit']['p_value']:.3e})"
    )
    print(
        f"    Fixed baseline-20 family match: {summary['baseline20_curve_fit']['matches']}/"
        f"{summary['baseline20_curve_fit']['total']} "
        f"(p = {summary['baseline20_curve_fit']['p_value']:.3e})"
    )
    print(
        f"    Expanded 25-domain empirical family match: {summary['expanded25_curve_fit']['matches']}/"
        f"{summary['expanded25_curve_fit']['total']} "
        f"(p = {summary['expanded25_curve_fit']['p_value']:.3e})"
    )
    print()
    print("  Secondary tiers")
    print(
        f"    Published exponents (direct): nearest comparator matches "
        f"{summary['published_exponent_direct']['nearest_matches']}/"
        f"{summary['published_exponent_direct']['total']}"
    )
    print(
        f"    Published exponents (direct): CI includes prediction "
        f"{summary['published_exponent_direct']['ci_matches']}/"
        f"{summary['published_exponent_direct']['ci_total']}"
    )
    print(
        f"    Published exponents (provisional): nearest comparator matches "
        f"{summary['published_exponent_provisional']['nearest_matches']}/"
        f"{summary['published_exponent_provisional']['total']}"
    )
    print(
        f"    Published exponents (provisional): CI includes prediction "
        f"{summary['published_exponent_provisional']['ci_matches']}/"
        f"{summary['published_exponent_provisional']['ci_total']}"
    )
    print(
        f"    Analytic identities family match: {summary['analytic_identity']['matches']}/"
        f"{summary['analytic_identity']['total']}"
    )
    print()
    print("  Important guardrail")
    print("    No single blended p-value is reported for the 50-domain suite.")
    print("    The 50-domain expansion mixes evidence tiers and is reported as a")
    print("    tiered validation set, not as one flat blind empirical count.")


def build_summary(domains: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    domain_map = {domain["id"]: domain for domain in domains}

    empirical = [item for item in results if item["tier"] == "empirical_curve_fit"]
    baseline20 = [item for item in empirical if domain_map[item["domain_id"]]["cohort"] == "baseline20"]
    expanded25 = [item for item in empirical if domain_map[item["domain_id"]]["cohort"] in {"baseline20", "expansion25"}]

    direct = [item for item in results if item["tier"] == "published_exponent_direct"]
    provisional = [item for item in results if item["tier"] == "published_exponent_provisional"]
    analytic = [item for item in results if item["tier"] == "analytic_identity"]

    empirical_matches, empirical_total = count_if(empirical, "family_match")
    baseline_matches, baseline_total = count_if(baseline20, "family_match")
    expanded_matches, expanded_total = count_if(expanded25, "family_match")
    direct_nearest, direct_total = count_if(direct, "nearest_match")
    direct_ci, direct_ci_total = count_if(direct, "ci_includes_prediction")
    provisional_nearest, provisional_total = count_if(provisional, "nearest_match")
    provisional_ci, provisional_ci_total = count_if(provisional, "ci_includes_prediction")
    analytic_matches, analytic_total = count_if(analytic, "family_match")

    return {
        "empirical_curve_fit": {
            "matches": empirical_matches,
            "total": empirical_total,
            "p_value": binomial_pvalue(empirical_matches, empirical_total),
        },
        "baseline20_curve_fit": {
            "matches": baseline_matches,
            "total": baseline_total,
            "p_value": binomial_pvalue(baseline_matches, baseline_total),
        },
        "expanded25_curve_fit": {
            "matches": expanded_matches,
            "total": expanded_total,
            "p_value": binomial_pvalue(expanded_matches, expanded_total),
        },
        "published_exponent_direct": {
            "nearest_matches": direct_nearest,
            "total": direct_total,
            "ci_matches": direct_ci,
            "ci_total": direct_ci_total,
        },
        "published_exponent_provisional": {
            "nearest_matches": provisional_nearest,
            "total": provisional_total,
            "ci_matches": provisional_ci,
            "ci_total": provisional_ci_total,
        },
        "analytic_identity": {
            "matches": analytic_matches,
            "total": analytic_total,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUT)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    domains = manifest["domains"]
    results = [evaluate_domain(domain) for domain in domains]
    summary = build_summary(domains, results)

    empirical = [item for item in results if item["tier"] == "empirical_curve_fit"]
    direct = [item for item in results if item["tier"] == "published_exponent_direct"]
    provisional = [item for item in results if item["tier"] == "published_exponent_provisional"]
    analytic = [item for item in results if item["tier"] == "analytic_identity"]

    print("=" * 100)
    print("  ARC PRINCIPLE: TIERED 50-DOMAIN VALIDATION SUITE")
    print("=" * 100)
    print(f"  Manifest: {args.manifest}")
    print(f"  Domain count: {len(domains)}")
    print(f"  Primary endpoint: {manifest['metadata']['primary_endpoint']}")
    print()
    print("  Evidence tiers:")
    for tier, count in manifest["metadata"]["tier_counts"].items():
        print(f"    {tier}: {count}")

    print_curve_table("EMPIRICAL CURVE-FIT DOMAINS", empirical)
    print_exponent_table("PUBLISHED EXPONENT DOMAINS (DIRECT)", direct)
    print_exponent_table("PUBLISHED EXPONENT DOMAINS (PROVISIONAL)", provisional)
    print_curve_table("ANALYTIC IDENTITY DOMAINS", analytic)
    print_summary(summary)

    payload = {
        "metadata": manifest["metadata"],
        "summary": summary,
        "results": results,
    }
    args.json_output.write_text(json.dumps(payload, indent=2))
    print()
    print(f"  JSON results written to {args.json_output}")


if __name__ == "__main__":
    main()
