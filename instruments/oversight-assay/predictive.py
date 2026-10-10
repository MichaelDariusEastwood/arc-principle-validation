#!/usr/bin/env python3
"""Development forecasts from the existing exact-output oversight assay.

No network, model calls, code execution, empirical result or proposition verdict.
The adopted two-state Markov model is a rival baseline, not an ARC law. See
Liu and Meng (2026), https://arxiv.org/abs/2604.22273 . Transport from a
single-pass panel to an iterative trajectory is an untested assumption.
"""
import argparse
import math
import sys
from pathlib import Path

import assay


SCHEMA = "arc-oversight-predictive-development/0.1"


def probability(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise assay.AssayError(f"{name} must be a finite probability")
    if not 0 <= value <= 1 or not math.isfinite(value):
        raise assay.AssayError(f"{name} must be a finite probability")
    return float(value)


def positive_count(value, name, maximum=1_000_000):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise assay.AssayError(f"{name} must be an integer in [1, {maximum}]")
    return value


def error_forecast(error_introduction, error_repair, initial_error, depth):
    """Exact recursion for specified transition probabilities, not fitted evidence."""
    a = probability(error_introduction, "error_introduction")
    b = probability(error_repair, "error_repair")
    e = probability(initial_error, "initial_error")
    positive_count(depth, "depth", 10_000)
    series = [e]
    for _ in range(depth):
        e = (1 - e) * a + e * (1 - b)
        series.append(e)
    return series


def transition_counts(panel, records):
    """Score raw responses through the existing oracle and retain denominators."""
    scored = assay.score(panel, records)
    obs = scored["observations"]
    good = [r for r in obs if r["proposal_correct"]]
    faulty = [r for r in obs if not r["proposal_correct"]]
    introduced = sum(not r["final_correct"] for r in good)
    repaired = sum(r["fault_repaired"] for r in faulty)
    return {
        "initially_correct": len(good), "initially_incorrect": len(faulty),
        "correct_to_unsuccessful_output": introduced, "incorrect_to_successful_output": repaired,
        "correct_inputs_with_invalid_or_missing_output": sum(r["response_status"] != "valid" for r in good),
        "correct_inputs_with_valid_but_incorrect_output": sum(r["response_status"] == "valid" and not r["final_correct"] for r in good),
        "a_hat": introduced / len(good), "b_hat": repaired / len(faulty),
        "coverage": scored["coverage"],
        "panel_sha256": scored["panel_sha256"],
        "responses_sha256": scored["responses_sha256"],
    }


def forecast_report(panel, records, initial_error, depth):
    counts = transition_counts(panel, records)
    series = error_forecast(counts["a_hat"], counts["b_hat"], initial_error, depth)
    report = {
        "schema": SCHEMA,
        "analysis_status": "instrument-development-only",
        "confirmatory_eligible": False,
        "proposition_verdict": None,
        "source": counts,
        "model": "adopted two-state time-homogeneous Markov baseline",
        "outcome_definition": "Success means a correct final answer under the existing strict output contract. "
                              "Failure includes a missing or malformed reply as well as an incorrect valid reply. "
                              "It does not necessarily mean a harmful edit to the initial artefact.",
        "initial_error": float(initial_error),
        "depth": depth,
        "predicted_error": series,
        "parameter_uncertainty_propagated": False,
        "assumptions": [
            "Calibration proposals represent the later correct and incorrect states.",
            "State-conditional transition probabilities stay constant over depth.",
            "The declared state is adequate; dependence on history does not alter transitions.",
            "Missing and malformed replies remain failures under the existing output contract.",
        ],
        "non_implications": [
            "This is a plug-in forecast, not an uncertainty interval or an empirical test.",
            "No beta, k, gamma, capacity, energy, value alignment or general safety is measured.",
            "Public development panels cannot become a secret confirmatory holdout.",
            "A fitted or successful one-step baseline does not establish multi-step transport.",
            "Continuation after an absent or malformed output needs a separately defined state and retry policy.",
        ],
    }
    report["forecast_sha256"] = assay.digest(report)
    return report


def zero_event_upper_bound(n, alpha=0.05):
    """One-sided exact upper bound after zero events in n IID Bernoulli trials.

    Fixed sample size, complete detection, specified population and no selection.
    This is not valid automatically under optional stopping or distribution shift.
    """
    positive_count(n, "n", 1_000_000_000)
    alpha = probability(alpha, "alpha")
    if not 0 < alpha < 1:
        raise assay.AssayError("alpha must be strictly between zero and one")
    return -math.expm1(math.log(alpha) / n)


def log_range(values):
    """A necessary range diagnostic; sufficient range would not identify a model."""
    if not isinstance(values, list) or len(values) < 2:
        raise assay.AssayError("at least two capability observations are required")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise assay.AssayError("capability must contain finite positive numbers")
        if value <= 0 or value > sys.float_info.max or not math.isfinite(value):
            raise assay.AssayError("capability must contain finite positive numbers")
    distinct = len(set(values))
    return {
        "distinct_capability_values": distinct,
        "range_dex": math.log10(max(values)) - math.log10(min(values)),
        "constant_axis": distinct == 1,
        "exponent_identified": False,
        "interpretation": (
            "Constant capability cannot identify a capability scaling exponent."
            if distinct == 1 else
            "Variation exists; scale validity, independent interventions, uncertainty and model fit remain untested."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    forecast = sub.add_parser("forecast")
    forecast.add_argument("--panel", type=Path, required=True)
    forecast.add_argument("--responses", type=Path, required=True)
    forecast.add_argument("--initial-error", type=float, required=True)
    forecast.add_argument("--depth", type=int, required=True)
    forecast.add_argument("--out", type=Path, required=True)
    risk = sub.add_parser("zero-events")
    risk.add_argument("--n", type=int, required=True)
    risk.add_argument("--alpha", type=float, default=0.05)
    axis = sub.add_parser("range")
    axis.add_argument("--values", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "forecast":
            panel = assay.parse_json(args.panel.read_text(encoding="utf-8"))
            records = [assay.parse_json(s) for s in args.responses.read_text(encoding="utf-8").splitlines() if s.strip()]
            result = forecast_report(panel, records, args.initial_error, args.depth)
            assay.write_new(args.out, assay.canonical(result) + "\n")
            print(str(args.out))
        elif args.command == "zero-events":
            print(assay.canonical({"upper_bound": zero_event_upper_bound(args.n, args.alpha),
                                  "assumptions": "zero events; fixed n; IID Bernoulli; complete detection; no shift",
                                  "global_safety_guarantee": False}))
        else:
            print(assay.canonical(log_range(assay.parse_json(args.values.read_text(encoding="utf-8")))))
    except (ValueError, TypeError, KeyError, OSError, RecursionError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
