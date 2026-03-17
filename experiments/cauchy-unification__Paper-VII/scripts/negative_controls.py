#!/usr/bin/env python3
"""
================================================================================
NEGATIVE CONTROL TEST SUITE — CAUCHY UNIFICATION FRAMEWORK
================================================================================

Purpose:
  Demonstrate that the Cauchy unification framework has genuine discriminative
  power. If the framework simply 'confirms everything monotone', it would
  match scrambled data and axiom-violating systems at the same rate as real
  data. This suite proves it does not.

Controls:
  1. Scrambled data (10 domains) — shuffle y values, destroy structure
  2. Random monotone curves (10 tests) — monotone but structureless
  3. Axiom-violating domains (5+ tests) — systems that violate Cauchy axioms
  4. Bootstrap stability (25 domains) — check best-fit family stability

Key insight:
  If the framework gets ~19/25 on real data but only ~8/25 on scrambled data,
  that is powerful evidence the signal is real.

================================================================================
Michael Darius Eastwood | March 2026
================================================================================
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy import optimize, stats

warnings.filterwarnings("ignore")


# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "arc_50_domain_universal_test.py"
RESULTS_PATH = REPO_ROOT / "results" / "results_50_domain_validation.json"
JSON_OUT = REPO_ROOT / "results" / "negative_control_results.json"
TXT_OUT = REPO_ROOT / "results" / "negative_control_results.txt"

FAMILIES = ["power_law", "exponential", "bounded"]
CHANCE_RATE = 1.0 / len(FAMILIES)


# ── Terminal formatting ──────────────────────────────────────────────────────

BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"


def divider(char: str = "=", width: int = 80) -> str:
    return char * width


def header(text: str, level: int = 1) -> str:
    if level == 1:
        return f"\n{divider('=')}\n  {text}\n{divider('=')}"
    return f"\n  {text}\n{divider('-', 70)}"


# ── Load runner module ───────────────────────────────────────────────────────

def load_runner():
    spec = importlib.util.spec_from_file_location("arc50_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_empirical_domains(runner) -> list[dict]:
    manifest = runner.load_manifest(runner.DEFAULT_MANIFEST)
    return [d for d in manifest["domains"] if d["evidence_tier"] == "empirical_curve_fit"]


def load_observed_summary() -> dict:
    return json.loads(RESULTS_PATH.read_text())


# ══════════════════════════════════════════════════════════════════════════════
# CONTROL 1: SCRAMBLED DATA (10 DOMAINS)
# ══════════════════════════════════════════════════════════════════════════════

def run_scrambled_controls(
    runner,
    domains: list[dict],
    rng: np.random.Generator,
    n_select: int = 10,
    iterations: int = 50,
) -> dict[str, Any]:
    """
    Take n_select domains, shuffle y values while keeping x fixed.
    Run the prediction protocol. The framework should NOT match at
    a rate above chance (1/3).
    """
    selected = rng.choice(domains, size=min(n_select, len(domains)), replace=False).tolist()

    per_domain_results = []
    all_match_counts = []

    for iteration in range(iterations):
        match_count = 0
        for domain in selected:
            shuffled = copy.deepcopy(domain)
            y_arr = np.array(domain["dataset"]["y"], dtype=float)
            shuffled["dataset"]["y"] = rng.permutation(y_arr).tolist()

            result = runner.evaluate_curve_domain(shuffled)
            if result.get("family_match", False):
                match_count += 1

        all_match_counts.append(match_count)

    match_arr = np.array(all_match_counts)
    mean_rate = float(match_arr.mean()) / len(selected)

    # Per-domain stability: run once more, tracking per domain
    per_domain_hits = {d["id"]: 0 for d in selected}
    for _ in range(iterations):
        for domain in selected:
            shuffled = copy.deepcopy(domain)
            y_arr = np.array(domain["dataset"]["y"], dtype=float)
            shuffled["dataset"]["y"] = rng.permutation(y_arr).tolist()
            result = runner.evaluate_curve_domain(shuffled)
            if result.get("family_match", False):
                per_domain_hits[domain["id"]] += 1

    per_domain_rates = [
        {
            "domain_id": d["id"],
            "name": d["name"],
            "predicted_family": d["predicted_family"],
            "scrambled_match_rate": per_domain_hits[d["id"]] / iterations,
        }
        for d in selected
    ]

    return {
        "test": "scrambled_data",
        "n_domains": len(selected),
        "iterations": iterations,
        "domain_ids": [d["id"] for d in selected],
        "mean_match_rate": mean_rate,
        "mean_matches_per_iter": float(match_arr.mean()),
        "std_matches_per_iter": float(match_arr.std(ddof=1)) if iterations > 1 else 0.0,
        "max_matches_in_any_iter": int(match_arr.max()),
        "chance_rate": CHANCE_RATE,
        "above_chance": mean_rate > CHANCE_RATE + 0.05,
        "pass": mean_rate <= CHANCE_RATE + 0.10,
        "per_domain_rates": sorted(per_domain_rates, key=lambda x: x["scrambled_match_rate"], reverse=True),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CONTROL 2: RANDOM MONOTONE CURVES (10 TESTS)
# ══════════════════════════════════════════════════════════════════════════════

def generate_random_monotone(rng: np.random.Generator, n_points: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Generate a random monotone curve that is NOT necessarily from any of the
    three canonical families."""
    x = np.sort(rng.uniform(0.1, 100.0, size=n_points))

    # Choose a random non-canonical generating function
    choice = rng.integers(0, 6)
    if choice == 0:
        # Square root with additive noise
        y = np.sqrt(x) * rng.uniform(0.5, 5.0) + rng.normal(0, 0.1, n_points)
    elif choice == 1:
        # Logarithmic
        y = rng.uniform(1.0, 10.0) * np.log(x + 1) + rng.normal(0, 0.2, n_points)
    elif choice == 2:
        # Linear with noise
        y = rng.uniform(0.1, 5.0) * x + rng.uniform(-10, 10) + rng.normal(0, 1.0, n_points)
    elif choice == 3:
        # Polynomial blend
        a, b, c = rng.uniform(-1, 1, 3)
        y = a * x**2 + b * x + c + rng.normal(0, 0.5, n_points)
    elif choice == 4:
        # Arctan (bounded-ish but not Michaelis-Menten)
        y = rng.uniform(1, 20) * np.arctan(x / rng.uniform(1, 50)) + rng.normal(0, 0.1, n_points)
    else:
        # Cube root
        y = rng.uniform(0.5, 10.0) * np.cbrt(x) + rng.normal(0, 0.05, n_points)

    # Ensure monotonicity by sorting
    y = np.sort(np.abs(y) + 0.01)
    return x, y


def run_random_monotone_controls(
    runner,
    rng: np.random.Generator,
    n_tests: int = 10,
) -> dict[str, Any]:
    """
    Generate random monotone curves, assign random operator classes,
    run the prediction protocol. Should NOT match at above-chance rate.
    """
    results = []
    match_count = 0

    for i in range(n_tests):
        x, y = generate_random_monotone(rng)
        random_family = rng.choice(FAMILIES)
        random_operator = rng.choice(["multiplicative", "additive", "bounded"])

        # Build a synthetic domain
        domain = {
            "id": 1000 + i,
            "cohort": "negative_control",
            "evidence_tier": "empirical_curve_fit",
            "name": f"Random Monotone #{i+1}",
            "operator_class": random_operator,
            "predicted_family": random_family,
            "predicted_model": None,
            "source": "synthetic",
            "dataset": {"x": x.tolist(), "y": y.tolist()},
        }

        result = runner.evaluate_curve_domain(domain)
        matched = result.get("family_match", False)
        if matched:
            match_count += 1

        results.append({
            "test_id": i + 1,
            "assigned_family": random_family,
            "best_family": result.get("best_family", "unknown"),
            "matched": matched,
            "n_points": len(x),
        })

    match_rate = match_count / n_tests

    return {
        "test": "random_monotone",
        "n_tests": n_tests,
        "matches": match_count,
        "match_rate": match_rate,
        "chance_rate": CHANCE_RATE,
        "above_chance": match_rate > CHANCE_RATE + 0.10,
        "pass": match_rate <= CHANCE_RATE + 0.15,
        "per_test": results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CONTROL 3: AXIOM-VIOLATING DOMAINS (5+ TESTS)
# ══════════════════════════════════════════════════════════════════════════════

def build_axiom_violating_domains(rng: np.random.Generator) -> list[dict]:
    """
    Build domains where the Cauchy axioms are explicitly violated.
    The framework SHOULD fail on these.
    """
    domains = []

    # 1. Non-associative composition: rock-paper-scissors dynamics
    #    The composition operator is non-associative, violating the
    #    semigroup axiom underlying Cauchy's functional equations.
    t = np.linspace(0, 50, 100)
    # Three-species Lotka-Volterra with cyclic dominance
    x1 = 1.0 + 0.8 * np.sin(2 * np.pi * t / 10)
    x2 = 1.0 + 0.8 * np.sin(2 * np.pi * t / 10 + 2 * np.pi / 3)
    x3 = 1.0 + 0.8 * np.sin(2 * np.pi * t / 10 + 4 * np.pi / 3)
    y_rps = x1 + x2 + x3 + rng.normal(0, 0.05, len(t))
    domains.append({
        "id": 2001,
        "cohort": "negative_control",
        "evidence_tier": "empirical_curve_fit",
        "name": "Non-Associative: Rock-Paper-Scissors Dynamics",
        "operator_class": "multiplicative",
        "predicted_family": "power_law",
        "predicted_model": None,
        "source": "synthetic — cyclic Lotka-Volterra",
        "violation": "non_associative_composition",
        "dataset": {"x": t.tolist(), "y": y_rps.tolist()},
    })

    # 2. Discontinuous system: phase transition with sharp jump
    t2 = np.linspace(0, 20, 100)
    y_phase = np.where(t2 < 10, 0.5 * t2, 5.0 + 2.0 * (t2 - 10)) + rng.normal(0, 0.1, len(t2))
    domains.append({
        "id": 2002,
        "cohort": "negative_control",
        "evidence_tier": "empirical_curve_fit",
        "name": "Discontinuous: Phase Transition (Sharp Jump)",
        "operator_class": "additive",
        "predicted_family": "exponential",
        "predicted_model": None,
        "source": "synthetic — piecewise linear with discontinuity",
        "violation": "discontinuous",
        "dataset": {"x": t2.tolist(), "y": y_phase.tolist()},
    })

    # 3. Chaotic system: logistic map in chaotic regime (r = 3.9)
    n_iter = 200
    x_logistic = np.arange(n_iter, dtype=float)
    y_logistic = np.zeros(n_iter)
    y_logistic[0] = 0.5
    r = 3.9
    for i in range(1, n_iter):
        y_logistic[i] = r * y_logistic[i - 1] * (1.0 - y_logistic[i - 1])
    # Take running mean to make it plottable but still chaotic
    window = 5
    y_running = np.convolve(y_logistic, np.ones(window) / window, mode="valid")
    x_running = x_logistic[: len(y_running)]
    domains.append({
        "id": 2003,
        "cohort": "negative_control",
        "evidence_tier": "empirical_curve_fit",
        "name": "Chaotic: Logistic Map (r=3.9)",
        "operator_class": "bounded",
        "predicted_family": "bounded",
        "predicted_model": None,
        "source": "synthetic — logistic map, chaotic regime",
        "violation": "chaotic",
        "dataset": {"x": x_running.tolist(), "y": y_running.tolist()},
    })

    # 4. Oscillatory system: predator-prey cycles
    t4 = np.linspace(0, 60, 150)
    prey = 100 * (1.0 + 0.6 * np.sin(2 * np.pi * t4 / 12))
    predator = 30 * (1.0 + 0.5 * np.sin(2 * np.pi * t4 / 12 + np.pi / 2))
    y_osc = prey + rng.normal(0, 2.0, len(t4))
    domains.append({
        "id": 2004,
        "cohort": "negative_control",
        "evidence_tier": "empirical_curve_fit",
        "name": "Oscillatory: Predator-Prey Cycles",
        "operator_class": "multiplicative",
        "predicted_family": "power_law",
        "predicted_model": None,
        "source": "synthetic — sinusoidal predator-prey",
        "violation": "oscillatory",
        "dataset": {"x": t4.tolist(), "y": y_osc.tolist()},
    })

    # 5. Multi-modal distribution: mixture of Gaussians
    t5 = np.linspace(-10, 10, 200)
    y_multi = (
        3.0 * np.exp(-0.5 * (t5 + 3) ** 2)
        + 5.0 * np.exp(-0.5 * (t5 - 2) ** 2 / 0.5)
        + 1.5 * np.exp(-0.5 * (t5 - 6) ** 2 / 2.0)
        + rng.normal(0, 0.05, len(t5))
    )
    domains.append({
        "id": 2005,
        "cohort": "negative_control",
        "evidence_tier": "empirical_curve_fit",
        "name": "Multi-Modal: Gaussian Mixture (3 peaks)",
        "operator_class": "additive",
        "predicted_family": "exponential",
        "predicted_model": None,
        "source": "synthetic — mixture of 3 Gaussians",
        "violation": "multi_modal",
        "dataset": {"x": t5.tolist(), "y": y_multi.tolist()},
    })

    # 6. Non-monotone: damped oscillation (Cauchy requires monotonicity for bounded)
    t6 = np.linspace(0, 30, 120)
    y_damped = np.exp(-0.1 * t6) * np.cos(2 * np.pi * t6 / 5) + rng.normal(0, 0.01, len(t6))
    domains.append({
        "id": 2006,
        "cohort": "negative_control",
        "evidence_tier": "empirical_curve_fit",
        "name": "Non-Monotone: Damped Oscillation",
        "operator_class": "additive",
        "predicted_family": "exponential",
        "predicted_model": None,
        "source": "synthetic — damped cosine",
        "violation": "non_monotone",
        "dataset": {"x": t6.tolist(), "y": y_damped.tolist()},
    })

    # 7. White noise: no structure at all
    t7 = np.linspace(0, 50, 100)
    y_noise = rng.normal(5.0, 2.0, len(t7))
    domains.append({
        "id": 2007,
        "cohort": "negative_control",
        "evidence_tier": "empirical_curve_fit",
        "name": "Pure White Noise",
        "operator_class": "multiplicative",
        "predicted_family": "power_law",
        "predicted_model": None,
        "source": "synthetic — Gaussian noise",
        "violation": "no_structure",
        "dataset": {"x": t7.tolist(), "y": y_noise.tolist()},
    })

    return domains


def run_axiom_violating_controls(
    runner,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """
    Run the prediction protocol on axiom-violating domains.
    The framework SHOULD fail on these (low match rate).
    """
    domains = build_axiom_violating_domains(rng)
    results = []
    match_count = 0

    for domain in domains:
        result = runner.evaluate_curve_domain(domain)
        matched = result.get("family_match", False)
        if matched:
            match_count += 1

        results.append({
            "domain_id": domain["id"],
            "name": domain["name"],
            "violation": domain["violation"],
            "predicted_family": domain["predicted_family"],
            "best_family": result.get("best_family", "unknown"),
            "best_model": result.get("best_model", "unknown"),
            "matched": matched,
            "r2_top": result.get("top_fits", [{}])[0].get("r2", None) if result.get("top_fits") else None,
        })

    match_rate = match_count / len(domains) if domains else 0.0

    return {
        "test": "axiom_violating",
        "n_domains": len(domains),
        "matches": match_count,
        "match_rate": match_rate,
        "chance_rate": CHANCE_RATE,
        "pass": match_rate <= CHANCE_RATE + 0.10,
        "per_domain": results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CONTROL 4: BOOTSTRAP STABILITY (OVERFITTING DETECTION)
# ══════════════════════════════════════════════════════════════════════════════

def run_bootstrap_stability(
    runner,
    domains: list[dict],
    rng: np.random.Generator,
    n_resamples: int = 100,
) -> dict[str, Any]:
    """
    For each of the 25 real domains, bootstrap resample n_resamples times.
    Check whether the best-fit family is stable across resamples.
    If a domain's winning family flips frequently, that domain is unreliable.
    """
    stability_results = []

    for domain in domains:
        x_orig = np.array(domain["dataset"]["x"], dtype=float)
        y_orig = np.array(domain["dataset"]["y"], dtype=float)
        n = len(x_orig)

        family_counts: dict[str, int] = {}
        model_counts: dict[str, int] = {}

        for _ in range(n_resamples):
            indices = rng.integers(0, n, size=n)
            x_boot = x_orig[indices]
            y_boot = y_orig[indices]

            # De-duplicate x values by adding tiny jitter to avoid singular fits
            x_boot = x_boot + rng.normal(0, 1e-10 * (np.max(np.abs(x_orig)) + 1e-15), size=n)

            boot_domain = copy.deepcopy(domain)
            boot_domain["dataset"]["x"] = x_boot.tolist()
            boot_domain["dataset"]["y"] = y_boot.tolist()

            result = runner.evaluate_curve_domain(boot_domain)
            best_family = result.get("best_family", "unknown")
            best_model = result.get("best_model", "unknown")
            family_counts[best_family] = family_counts.get(best_family, 0) + 1
            model_counts[best_model] = model_counts.get(best_model, 0) + 1

        # Stability = fraction of resamples that agree with the plurality winner
        if family_counts:
            dominant_family = max(family_counts, key=family_counts.get)
            family_stability = family_counts[dominant_family] / n_resamples
        else:
            dominant_family = "unknown"
            family_stability = 0.0

        if model_counts:
            dominant_model = max(model_counts, key=model_counts.get)
            model_stability = model_counts[dominant_model] / n_resamples
        else:
            dominant_model = "unknown"
            model_stability = 0.0

        predicted_match = dominant_family == domain["predicted_family"]

        stability_results.append({
            "domain_id": domain["id"],
            "name": domain["name"],
            "predicted_family": domain["predicted_family"],
            "dominant_family": dominant_family,
            "family_stability": family_stability,
            "dominant_model": dominant_model,
            "model_stability": model_stability,
            "family_counts": family_counts,
            "predicted_matches_dominant": predicted_match,
            "stable": family_stability >= 0.80,
        })

    n_stable = sum(1 for r in stability_results if r["stable"])
    n_predicted_match = sum(1 for r in stability_results if r["predicted_matches_dominant"])
    mean_stability = float(np.mean([r["family_stability"] for r in stability_results]))

    return {
        "test": "bootstrap_stability",
        "n_domains": len(domains),
        "n_resamples": n_resamples,
        "n_stable": n_stable,
        "n_unstable": len(domains) - n_stable,
        "n_predicted_match_dominant": n_predicted_match,
        "mean_family_stability": mean_stability,
        "stability_threshold": 0.80,
        "per_domain": sorted(stability_results, key=lambda r: r["family_stability"]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY & OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def build_text_report(payload: dict[str, Any]) -> str:
    """Build human-readable text report."""
    lines = []

    lines.append(divider("="))
    lines.append("  NEGATIVE CONTROL TEST SUITE — CAUCHY UNIFICATION FRAMEWORK")
    lines.append(divider("="))
    lines.append("")
    lines.append(f"  Date: {payload['date']}")
    lines.append(f"  Seed: {payload['seed']}")
    lines.append(f"  Observed real-data matches: {payload['observed_matches']}/{payload['observed_total']}")
    lines.append(f"  Observed real-data match rate: {payload['observed_matches']/payload['observed_total']:.1%}")
    lines.append(f"  Chance rate (3-family random): {CHANCE_RATE:.1%}")
    lines.append("")

    # --- Scrambled data ---
    sc = payload["scrambled_data"]
    lines.append(header("CONTROL 1: SCRAMBLED DATA", 1))
    lines.append(f"  Domains tested: {sc['n_domains']}")
    lines.append(f"  Iterations: {sc['iterations']}")
    lines.append(f"  Mean match rate: {sc['mean_match_rate']:.3f} (chance = {CHANCE_RATE:.3f})")
    lines.append(f"  Mean matches per iteration: {sc['mean_matches_per_iter']:.1f}/{sc['n_domains']}")
    lines.append(f"  Max matches in any iteration: {sc['max_matches_in_any_iter']}/{sc['n_domains']}")
    lines.append(f"  PASS: {'YES' if sc['pass'] else 'NO'} — {'at or near chance' if sc['pass'] else 'ABOVE CHANCE — investigate'}")
    lines.append("")
    lines.append("  Per-domain scrambled match rates (highest first):")
    for entry in sc["per_domain_rates"][:5]:
        lines.append(
            f"    Domain {entry['domain_id']:>2} {entry['name']:<40} "
            f"rate={entry['scrambled_match_rate']:.3f}  predicted={entry['predicted_family']}"
        )

    # --- Random monotone ---
    rm = payload["random_monotone"]
    lines.append(header("CONTROL 2: RANDOM MONOTONE CURVES", 1))
    lines.append(f"  Tests: {rm['n_tests']}")
    lines.append(f"  Matches: {rm['matches']}/{rm['n_tests']}")
    lines.append(f"  Match rate: {rm['match_rate']:.3f} (chance = {CHANCE_RATE:.3f})")
    lines.append(f"  PASS: {'YES' if rm['pass'] else 'NO'}")
    lines.append("")
    lines.append("  Per-test detail:")
    for entry in rm["per_test"]:
        status = "MATCH" if entry["matched"] else "no match"
        lines.append(
            f"    Test {entry['test_id']:>2}: assigned={entry['assigned_family']:<12} "
            f"best_fit={entry['best_family']:<12} {status}"
        )

    # --- Axiom-violating ---
    av = payload["axiom_violating"]
    lines.append(header("CONTROL 3: AXIOM-VIOLATING DOMAINS", 1))
    lines.append(f"  Domains: {av['n_domains']}")
    lines.append(f"  Matches: {av['matches']}/{av['n_domains']}")
    lines.append(f"  Match rate: {av['match_rate']:.3f} (chance = {CHANCE_RATE:.3f})")
    lines.append(f"  PASS: {'YES' if av['pass'] else 'NO'}")
    lines.append("")
    lines.append("  Per-domain detail:")
    for entry in av["per_domain"]:
        status = "MATCH (unexpected)" if entry["matched"] else "no match (expected)"
        r2_str = f"R2={entry['r2_top']:.4f}" if entry["r2_top"] is not None else "R2=n/a"
        lines.append(
            f"    [{entry['violation']:<24}] {entry['name']:<44} "
            f"pred={entry['predicted_family']:<12} best={entry['best_family']:<12} "
            f"{r2_str}  {status}"
        )

    # --- Bootstrap stability ---
    bs = payload["bootstrap_stability"]
    lines.append(header("CONTROL 4: BOOTSTRAP STABILITY (OVERFITTING DETECTION)", 1))
    lines.append(f"  Domains: {bs['n_domains']}")
    lines.append(f"  Resamples per domain: {bs['n_resamples']}")
    lines.append(f"  Stability threshold: {bs['stability_threshold']:.0%}")
    lines.append(f"  Stable domains: {bs['n_stable']}/{bs['n_domains']}")
    lines.append(f"  Unstable domains: {bs['n_unstable']}/{bs['n_domains']}")
    lines.append(f"  Mean family stability: {bs['mean_family_stability']:.3f}")
    lines.append(f"  Domains where dominant family = predicted: {bs['n_predicted_match_dominant']}/{bs['n_domains']}")
    lines.append("")

    # Show unstable domains first, then stable
    lines.append("  Per-domain stability (sorted by stability, lowest first):")
    for entry in bs["per_domain"]:
        marker = "STABLE" if entry["stable"] else "UNSTABLE"
        pred_match = "pred=dominant" if entry["predicted_matches_dominant"] else "pred!=dominant"
        lines.append(
            f"    Domain {entry['domain_id']:>2} {entry['name']:<40} "
            f"stability={entry['family_stability']:.2f}  dominant={entry['dominant_family']:<12} "
            f"{marker:<8}  {pred_match}"
        )

    # --- Grand summary ---
    lines.append(header("GRAND SUMMARY", 1))
    lines.append("")
    lines.append(f"  Real data match rate:            {payload['observed_matches']}/{payload['observed_total']} "
                 f"= {payload['observed_matches']/payload['observed_total']:.1%}")
    lines.append(f"  Scrambled data match rate:        {sc['mean_match_rate']:.1%}")
    lines.append(f"  Random monotone match rate:       {rm['match_rate']:.1%}")
    lines.append(f"  Axiom-violating match rate:       {av['match_rate']:.1%}")
    lines.append(f"  Chance baseline:                  {CHANCE_RATE:.1%}")
    lines.append("")

    real_rate = payload["observed_matches"] / payload["observed_total"]
    scrambled_rate = sc["mean_match_rate"]
    if scrambled_rate > 0:
        discrimination_ratio = real_rate / scrambled_rate
    else:
        discrimination_ratio = float("inf")

    lines.append(f"  Discrimination ratio (real / scrambled): {discrimination_ratio:.1f}x")
    lines.append("")

    all_pass = sc["pass"] and rm["pass"] and av["pass"]
    if all_pass:
        lines.append("  VERDICT: ALL NEGATIVE CONTROLS PASS.")
        lines.append("  The framework has genuine discriminative power.")
        lines.append("  It does NOT simply confirm everything monotone.")
    else:
        lines.append("  VERDICT: SOME NEGATIVE CONTROLS FAILED — investigate.")
        if not sc["pass"]:
            lines.append("    - Scrambled data matched above chance.")
        if not rm["pass"]:
            lines.append("    - Random monotone curves matched above chance.")
        if not av["pass"]:
            lines.append("    - Axiom-violating domains matched above chance.")

    lines.append("")
    lines.append(f"  Bootstrap stability: {bs['n_stable']}/{bs['n_domains']} domains are stable across resamples.")
    if bs["n_unstable"] > 0:
        unstable_names = [e["name"] for e in bs["per_domain"] if not e["stable"]]
        lines.append(f"  Unstable domains: {', '.join(unstable_names)}")
    lines.append("")
    lines.append(divider("="))
    lines.append("")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Negative control test suite for the Cauchy unification framework."
    )
    parser.add_argument("--seed", type=int, default=20260317)
    parser.add_argument("--scrambled-domains", type=int, default=10)
    parser.add_argument("--scrambled-iterations", type=int, default=50)
    parser.add_argument("--monotone-tests", type=int, default=10)
    parser.add_argument("--bootstrap-resamples", type=int, default=100)
    args = parser.parse_args()

    start = time.time()
    rng = np.random.default_rng(args.seed)

    print(f"{BOLD}Loading runner and domains...{RESET}")
    runner = load_runner()
    domains = load_empirical_domains(runner)
    observed = load_observed_summary()
    observed_matches = observed["summary"]["empirical_curve_fit"]["matches"]
    observed_total = observed["summary"]["empirical_curve_fit"]["total"]

    print(f"  Observed real-data result: {observed_matches}/{observed_total}")
    print(f"  Running negative controls with seed={args.seed}")
    print()

    # --- Control 1: Scrambled data ---
    print(f"{BOLD}Control 1: Scrambled data ({args.scrambled_domains} domains, {args.scrambled_iterations} iterations)...{RESET}")
    scrambled = run_scrambled_controls(
        runner, domains, rng,
        n_select=args.scrambled_domains,
        iterations=args.scrambled_iterations,
    )
    print(f"  Match rate: {scrambled['mean_match_rate']:.3f} (chance={CHANCE_RATE:.3f})  {'PASS' if scrambled['pass'] else 'FAIL'}")
    print()

    # --- Control 2: Random monotone ---
    print(f"{BOLD}Control 2: Random monotone curves ({args.monotone_tests} tests)...{RESET}")
    monotone = run_random_monotone_controls(runner, rng, n_tests=args.monotone_tests)
    print(f"  Match rate: {monotone['match_rate']:.3f} (chance={CHANCE_RATE:.3f})  {'PASS' if monotone['pass'] else 'FAIL'}")
    print()

    # --- Control 3: Axiom-violating ---
    print(f"{BOLD}Control 3: Axiom-violating domains...{RESET}")
    axiom = run_axiom_violating_controls(runner, rng)
    print(f"  Match rate: {axiom['match_rate']:.3f} (chance={CHANCE_RATE:.3f})  {'PASS' if axiom['pass'] else 'FAIL'}")
    print()

    # --- Control 4: Bootstrap stability ---
    print(f"{BOLD}Control 4: Bootstrap stability ({args.bootstrap_resamples} resamples per domain)...{RESET}")
    bootstrap = run_bootstrap_stability(
        runner, domains, rng,
        n_resamples=args.bootstrap_resamples,
    )
    print(f"  Stable domains: {bootstrap['n_stable']}/{bootstrap['n_domains']} (threshold={bootstrap['stability_threshold']:.0%})")
    print(f"  Mean family stability: {bootstrap['mean_family_stability']:.3f}")
    print()

    # --- Assemble output ---
    payload = {
        "date": "2026-03-17",
        "seed": args.seed,
        "observed_matches": observed_matches,
        "observed_total": observed_total,
        "scrambled_data": scrambled,
        "random_monotone": monotone,
        "axiom_violating": axiom,
        "bootstrap_stability": bootstrap,
    }

    # Write JSON
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote JSON: {JSON_OUT}")

    # Write human-readable text
    report = build_text_report(payload)
    TXT_OUT.write_text(report)
    print(f"Wrote text: {TXT_OUT}")

    # Also print the report
    print()
    print(report)

    elapsed = time.time() - start
    print(f"{DIM}Completed in {elapsed:.1f} seconds{RESET}")


if __name__ == "__main__":
    main()
