#!/usr/bin/env python3
"""
Run null/negative controls for the canonical 25-domain empirical cohort.

Controls included:
  1. Monte Carlo family-label null: random family guesses across 25 domains.
  2. Y-shuffle null: destroy within-domain structure by permuting y values and
     re-running the same fitter selection.

The point is to show that the stricter harness does not trivially confirm the
predicted family when structure is broken.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "arc_50_domain_universal_test.py"
RESULTS_PATH = REPO_ROOT / "results" / "results_50_domain_validation.json"
JSON_OUT = REPO_ROOT / "results" / "null_control_results.json"
MD_OUT = REPO_ROOT / "results" / "null_control_results.md"


def load_runner():
    spec = importlib.util.spec_from_file_location("arc50_runner", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_empirical_domains(runner) -> list[dict]:
    manifest = runner.load_manifest(runner.DEFAULT_MANIFEST)
    return [domain for domain in manifest["domains"] if domain["evidence_tier"] == "empirical_curve_fit"]


def load_observed_summary() -> dict:
    return json.loads(RESULTS_PATH.read_text())


def monte_carlo_family_null(rng: np.random.Generator, trials: int, observed_matches: int, n_domains: int) -> dict:
    families = np.array(["power_law", "exponential", "bounded"], dtype=object)
    predicted = families[rng.integers(0, len(families), size=(trials, n_domains))]
    actual = families[rng.integers(0, len(families), size=(trials, n_domains))]
    matches = (predicted == actual).sum(axis=1)
    return {
        "trials": trials,
        "n_domains": n_domains,
        "observed_matches": observed_matches,
        "mean_matches": float(matches.mean()),
        "std_matches": float(matches.std(ddof=1)),
        "max_matches": int(matches.max()),
        "p_ge_observed": float(np.mean(matches >= observed_matches)),
        "quantiles": {
            "q95": float(np.quantile(matches, 0.95)),
            "q99": float(np.quantile(matches, 0.99)),
            "q999": float(np.quantile(matches, 0.999)),
        },
    }


def shuffled_y_null(
    runner,
    domains: list[dict],
    rng: np.random.Generator,
    iterations: int,
    observed_matches: int,
) -> dict:
    per_domain_hits = {domain["id"]: 0 for domain in domains}
    total_matches = []
    family_win_counts = {}

    for _ in range(iterations):
        matches_this_iter = 0
        for domain in domains:
            shuffled = copy.deepcopy(domain)
            shuffled["dataset"]["y"] = rng.permutation(np.array(domain["dataset"]["y"], dtype=float)).tolist()
            result = runner.evaluate_curve_domain(shuffled)
            best_family = result["best_family"]
            family_win_counts[best_family] = family_win_counts.get(best_family, 0) + 1
            if result["family_match"]:
                per_domain_hits[domain["id"]] += 1
                matches_this_iter += 1
        total_matches.append(matches_this_iter)

    total_matches_array = np.array(total_matches)
    per_domain_rates = [
        {
            "domain_id": domain["id"],
            "name": domain["name"],
            "predicted_family": domain["predicted_family"],
            "shuffle_match_rate": per_domain_hits[domain["id"]] / iterations,
        }
        for domain in domains
    ]

    per_domain_rates.sort(key=lambda item: item["shuffle_match_rate"], reverse=True)
    return {
        "iterations": iterations,
        "n_domains": len(domains),
        "observed_matches": observed_matches,
        "mean_total_matches": float(total_matches_array.mean()),
        "std_total_matches": float(total_matches_array.std(ddof=1)),
        "max_total_matches": int(total_matches_array.max()),
        "p_ge_observed": float(np.mean(total_matches_array >= observed_matches)),
        "quantiles": {
            "q95": float(np.quantile(total_matches_array, 0.95)),
            "q99": float(np.quantile(total_matches_array, 0.99)),
        },
        "winner_family_share": {
            family: count / (iterations * len(domains)) for family, count in sorted(family_win_counts.items())
        },
        "per_domain_rates": per_domain_rates,
    }


def build_markdown(payload: dict) -> str:
    family_null = payload["family_label_null"]
    shuffle_null = payload["shuffled_y_null"]
    lines = [
        "# Null-Control Results",
        "",
        "These controls test whether the stricter 25-domain empirical result can be reproduced after breaking the predictive structure.",
        "",
        f"- Observed empirical headline: `{payload['observed_matches']}/{payload['n_domains']}`",
        f"- Family-label null `p(match >= observed)` over `{family_null['trials']}` trials: `{family_null['p_ge_observed']:.6f}`",
        f"- Shuffled-y null `p(match >= observed)` over `{shuffle_null['iterations']}` iterations: `{shuffle_null['p_ge_observed']:.6f}`",
        "",
        "## Family-label null",
        "",
        f"- Mean matches: `{family_null['mean_matches']:.3f}`",
        f"- Std dev: `{family_null['std_matches']:.3f}`",
        f"- 99th percentile: `{family_null['quantiles']['q99']:.3f}`",
        f"- Max observed in simulation: `{family_null['max_matches']}`",
        "",
        "## Shuffled-y null",
        "",
        f"- Mean total matches: `{shuffle_null['mean_total_matches']:.3f}`",
        f"- Std dev: `{shuffle_null['std_total_matches']:.3f}`",
        f"- 99th percentile: `{shuffle_null['quantiles']['q99']:.3f}`",
        f"- Max observed in simulation: `{shuffle_null['max_total_matches']}`",
        "",
        "Winner-family share under shuffled-y null:",
        "",
    ]
    for family, share in shuffle_null["winner_family_share"].items():
        lines.append(f"- `{family}`: `{share:.4f}`")
    lines.extend(["", "Highest per-domain false-confirmation rates under shuffled-y null:", ""])
    for entry in shuffle_null["per_domain_rates"][:8]:
        lines.append(
            f"- `{entry['domain_id']:02d}` {entry['name']}: `{entry['shuffle_match_rate']:.3f}` "
            f"for predicted family `{entry['predicted_family']}`"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family-trials", type=int, default=200000)
    parser.add_argument("--shuffle-iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260317)
    args = parser.parse_args()

    runner = load_runner()
    domains = load_empirical_domains(runner)
    observed = load_observed_summary()
    observed_matches = observed["summary"]["empirical_curve_fit"]["matches"]
    rng = np.random.default_rng(args.seed)

    family_null = monte_carlo_family_null(rng, args.family_trials, observed_matches, len(domains))
    shuffle_null = shuffled_y_null(runner, domains, rng, args.shuffle_iterations, observed_matches)

    payload = {
        "version": "2026-03-17",
        "seed": args.seed,
        "observed_matches": observed_matches,
        "n_domains": len(domains),
        "family_label_null": family_null,
        "shuffled_y_null": shuffle_null,
    }

    JSON_OUT.write_text(json.dumps(payload, indent=2))
    MD_OUT.write_text(build_markdown(payload))
    print(f"Wrote {JSON_OUT}")
    print(f"Wrote {MD_OUT}")


if __name__ == "__main__":
    main()
