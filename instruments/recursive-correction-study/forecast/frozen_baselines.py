#!/usr/bin/env python3
"""Frozen, adopted Markov rivals for development of a later prediction study.

Standard library only. No network, model dispatch, fitted ARC theory, registration
or safety verdict. A local hash binds bytes; it is not an external timestamp.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics


PLAN = "rd-adopted-transition-plan/0.1"
LEDGER = "rd-adopted-transition-ledger/0.1"
REPORT = "rd-adopted-transition-evaluation/0.1"
STATUSES = {"correct", "incorrect", "invalid", "missing"}
BASELINES = ("pooled", "stratified")
HASH = re.compile(r"[0-9a-f]{64}")


class ForecastError(ValueError):
    pass


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def digest(obj):
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ForecastError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    def nonfinite(value):
        raise ForecastError(f"Non-finite JSON value: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=_unique, parse_constant=nonfinite)


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def exact(obj, fields, name):
    if not isinstance(obj, dict) or set(obj) != set(fields):
        raise ForecastError(f"Unexpected fields in {name}")


def text(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 500:
        raise ForecastError(f"Invalid {name}")
    return value


def sha(value, name):
    if not isinstance(value, str) or not HASH.fullmatch(value):
        raise ForecastError(f"Invalid SHA-256 for {name}")
    return value


def number(value, name, lower=0.0, upper=1.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ForecastError(f"Invalid number for {name}")
    if value < lower or value > upper or not math.isfinite(value):
        raise ForecastError(f"Out-of-range number for {name}")
    return float(value)


def positive_int(value, name, maximum=100000):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ForecastError(f"Invalid positive integer for {name}")
    return value


def stamp(value, name):
    text(value, name)
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None:
            raise ValueError("timezone absent")
        return result.astimezone(timezone.utc)
    except (ValueError, OverflowError) as exc:
        raise ForecastError(f"Invalid timezone-aware timestamp: {name}") from exc


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _tasks(tasks, name, strata, calibration=False):
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 100000:
        raise ForecastError(f"Invalid task list: {name}")
    ids, hashes, clusters = set(), set(), {}
    for task in tasks:
        exact(task, ("task_id", "cluster_id", "task_sha256", "stratum",
                     "initial_error", "initial_state_basis"), name)
        for key in ("task_id", "cluster_id", "stratum", "initial_state_basis"):
            text(task[key], key)
        sha(task["task_sha256"], "task")
        if task["task_id"] in ids or task["task_sha256"] in hashes:
            raise ForecastError(f"Duplicate task identity or content hash in {name}")
        ids.add(task["task_id"])
        hashes.add(task["task_sha256"])
        if task["stratum"] not in strata:
            raise ForecastError("Undeclared stratum")
        old = clusters.setdefault(task["cluster_id"], task["stratum"])
        if old != task["stratum"]:
            raise ForecastError("A cluster cannot span strata")
        initial = number(task["initial_error"], "initial_error")
        if calibration and initial not in (0, 1):
            raise ForecastError("Calibration initial state must be a checked binary state")
    if set(clusters.values()) != set(strata):
        raise ForecastError(f"Every declared stratum needs tasks in {name}")
    return ids, hashes, set(clusters)


def validate_plan(plan):
    exact(plan, ("schema", "purpose", "evidence_origin", "population",
                 "calibration_completed_at", "stratum_rule_sha256", "scorer_sha256",
                 "strata_weights", "configurations", "horizons", "smoothing",
                 "calibration_tasks", "prediction_tasks", "missing_rule",
                 "intervention_pairs"), "plan")
    if plan["schema"] != PLAN or plan["purpose"] != "development-only":
        raise ForecastError("This module is for development-only plans")
    if plan["evidence_origin"] not in ("synthetic", "collected"):
        raise ForecastError("Declare synthetic or collected evidence origin")
    text(plan["population"], "population")
    stamp(plan["calibration_completed_at"], "calibration_completed_at")
    sha(plan["stratum_rule_sha256"], "stratum rule")
    sha(plan["scorer_sha256"], "scorer")
    if plan["missing_rule"] != "explicit_missing_or_invalid_is_unsuccessful":
        raise ForecastError("Missing and invalid final outputs must remain unsuccessful")
    weights = plan["strata_weights"]
    if not isinstance(weights, dict) or not weights:
        raise ForecastError("Declare target stratum weights before prediction")
    for key, value in weights.items():
        text(key, "stratum")
        if key == "*":
            raise ForecastError("The pooled-group marker is not a stratum name")
        if number(value, "stratum weight") <= 0:
            raise ForecastError("Stratum weights must be positive")
    if not math.isclose(sum(weights.values()), 1, abs_tol=1e-12, rel_tol=0):
        raise ForecastError("Target stratum weights must sum to one")
    configs = plan["configurations"]
    if not isinstance(configs, list) or not 1 <= len(configs) <= 100:
        raise ForecastError("Invalid configurations")
    seen = set()
    for config in configs:
        exact(config, ("configuration_id", "model_revision", "policy_sha256"), "configuration")
        cid = text(config["configuration_id"], "configuration_id")
        revision = text(config["model_revision"], "model_revision")
        if revision.lower() in {"latest", "main", "current", "auto"} or revision.lower().endswith("-latest"):
            raise ForecastError("A moving model alias is not a revision")
        sha(config["policy_sha256"], "policy")
        if cid in seen:
            raise ForecastError("Duplicate configuration")
        seen.add(cid)
    horizons = plan["horizons"]
    if not isinstance(horizons, list) or not horizons or len(horizons) > 100:
        raise ForecastError("Invalid horizons")
    for horizon in horizons:
        positive_int(horizon, "horizon", 10000)
    if len(set(horizons)) != len(horizons) or horizons != sorted(horizons):
        raise ForecastError("Horizons must be unique and increasing")
    exact(plan["smoothing"], ("success_pseudocount", "failure_pseudocount"), "smoothing")
    for key, value in plan["smoothing"].items():
        number(value, key, 0, 100000)
    calibration = _tasks(plan["calibration_tasks"], "calibration", weights, True)
    prediction = _tasks(plan["prediction_tasks"], "prediction", weights)
    if any(a & b for a, b in zip(calibration, prediction)):
        raise ForecastError("Calibration and prediction overlap in task, content or cluster identity")
    pairs = plan["intervention_pairs"]
    if not isinstance(pairs, list):
        raise ForecastError("Invalid intervention pairs")
    used = set()
    for pair in pairs:
        exact(pair, ("control", "treatment"), "intervention pair")
        key = (pair["control"], pair["treatment"])
        if any(c not in seen for c in key) or key[0] == key[1] or key in used:
            raise ForecastError("Invalid or duplicate intervention pair")
        used.add(key)
    return plan


def transition_forecast(a, b, initial_error, horizon):
    a, b = number(a, "regression"), number(b, "repair")
    error = number(initial_error, "initial_error")
    positive_int(horizon, "horizon", 10000)
    for _ in range(horizon):
        error = (1 - error) * a + error * (1 - b)
    return error


def fit(plan, calibration_records):
    validate_plan(plan)
    if not isinstance(calibration_records, list):
        raise ForecastError("Calibration records must be a list")
    tasks = {t["task_id"]: t for t in plan["calibration_tasks"]}
    configs = [c["configuration_id"] for c in plan["configurations"]]
    expected = {(tid, cid) for tid in tasks for cid in configs}
    observed = set()
    counts = defaultdict(Counter)
    for row in calibration_records:
        exact(row, ("task_id", "task_sha256", "configuration_id", "final_status"), "calibration row")
        key = (row["task_id"], row["configuration_id"])
        if key not in expected or key in observed:
            raise ForecastError("Unknown or duplicate calibration row")
        observed.add(key)
        task = tasks[key[0]]
        if row["task_sha256"] != task["task_sha256"] or row["final_status"] not in STATUSES:
            raise ForecastError("Calibration hash or status mismatch")
        for group in ((key[1], "*"), (key[1], task["stratum"])):
            count = counts[group]
            count["missing"] += row["final_status"] == "missing"
            count["invalid"] += row["final_status"] == "invalid"
            if task["initial_error"] == 0:
                count["correct_exposure"] += 1
                count["regressions"] += row["final_status"] != "correct"
            else:
                count["error_exposure"] += 1
                count["repairs"] += row["final_status"] == "correct"
    if observed != expected:
        raise ForecastError("Missing calibration records; record missing outputs explicitly")
    smoothing = plan["smoothing"]
    success, failure = smoothing["success_pseudocount"], smoothing["failure_pseudocount"]
    result = {}
    for cid in configs:
        result[cid] = {}
        for group in ["*"] + list(plan["strata_weights"]):
            c = counts[(cid, group)]
            if not c["correct_exposure"] or not c["error_exposure"]:
                raise ForecastError(f"No required state exposure in {cid}/{group}; prior-only fallback is forbidden")
            result[cid][group] = {
                **{k: c[k] for k in ("correct_exposure", "error_exposure", "regressions", "repairs", "missing", "invalid")},
                "a": (c["regressions"] + success) / (c["correct_exposure"] + success + failure),
                "b": (c["repairs"] + success) / (c["error_exposure"] + success + failure),
                "estimator": "MLE plug-in" if success == failure == 0 else "prespecified smoothed plug-in",
            }
    return result


def forecasts(plan, rates):
    out = []
    for task in sorted(plan["prediction_tasks"], key=lambda x: x["task_id"]):
        for config in plan["configurations"]:
            cid = config["configuration_id"]
            for horizon in plan["horizons"]:
                row = {"task_id": task["task_id"], "cluster_id": task["cluster_id"],
                       "task_sha256": task["task_sha256"], "stratum": task["stratum"],
                       "configuration_id": cid, "horizon": horizon, "predicted_error": {}}
                for name, group in (("pooled", "*"), ("stratified", task["stratum"])):
                    r = rates[cid][group]
                    row["predicted_error"][name] = transition_forecast(r["a"], r["b"], task["initial_error"], horizon)
                out.append(row)
    return out


def freeze(plan, calibration_records):
    rates = fit(plan, calibration_records)
    frozen = now_iso()
    if stamp(plan["calibration_completed_at"], "calibration completion") > stamp(frozen, "freeze"):
        raise ForecastError("Calibration completion cannot follow forecast freeze")
    result = {"schema": LEDGER, "status": "development-only", "confirmatory_eligible": False,
              "proposition_verdict": None, "frozen_at": frozen,
              "plan": json.loads(canonical(plan)), "plan_sha256": digest(plan),
              "calibration_records": json.loads(canonical(calibration_records)),
              "calibration_sha256": digest(calibration_records), "rates": rates,
              "forecasts": forecasts(plan, rates), "parameter_uncertainty_propagated": False}
    result["ledger_sha256"] = digest(result)
    return result


def verify_ledger(ledger, expected_sha256):
    exact(ledger, ("schema", "status", "confirmatory_eligible", "proposition_verdict", "frozen_at",
                   "plan", "plan_sha256", "calibration_records", "calibration_sha256",
                   "rates", "forecasts", "parameter_uncertainty_propagated", "ledger_sha256"), "ledger")
    sha(expected_sha256, "externally retained ledger digest")
    body = {k: v for k, v in ledger.items() if k != "ledger_sha256"}
    if ledger["ledger_sha256"] != digest(body) or ledger["ledger_sha256"] != expected_sha256:
        raise ForecastError("Frozen ledger hash mismatch")
    if ledger["schema"] != LEDGER or ledger["status"] != "development-only" or ledger["confirmatory_eligible"] is not False or ledger["proposition_verdict"] is not None or ledger["parameter_uncertainty_propagated"] is not False:
        raise ForecastError("The development forecast scope has changed")
    plan = ledger["plan"]
    if ledger["plan_sha256"] != digest(plan) or ledger["calibration_sha256"] != digest(ledger["calibration_records"]):
        raise ForecastError("Plan or calibration digest mismatch")
    rates = fit(plan, ledger["calibration_records"])
    if canonical(rates) != canonical(ledger["rates"]) or canonical(forecasts(plan, rates)) != canonical(ledger["forecasts"]):
        raise ForecastError("Forecast or rate differs from its frozen calibration derivation")
    frozen = stamp(ledger["frozen_at"], "freeze")
    if frozen < stamp(plan["calibration_completed_at"], "calibration completion") or frozen > datetime.now(timezone.utc):
        raise ForecastError("Invalid local forecast chronology")
    return ledger


def _weighted(values, weights):
    return sum(weights[s] * statistics.mean(values[s]) for s in weights)


def _quantile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (index - lo) * (values[hi] - values[lo])


def evaluate(ledger, outcomes, expected_sha256, bootstrap=2000, seed=20261010):
    verify_ledger(ledger, expected_sha256)
    if not isinstance(outcomes, list):
        raise ForecastError("Outcomes must be a list")
    positive_int(bootstrap, "bootstrap replicates", 100000)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ForecastError("Invalid bootstrap seed")
    plan, frozen = ledger["plan"], stamp(ledger["frozen_at"], "freeze")
    lookup = {(r["task_id"], r["configuration_id"], r["horizon"]): r for r in ledger["forecasts"]}
    observed, rows, status_counts = set(), [], Counter()
    collection_time = datetime.now(timezone.utc)
    for row in outcomes:
        exact(row, ("task_id", "task_sha256", "configuration_id", "horizon", "final_status", "collected_at"), "outcome")
        positive_int(row["horizon"], "outcome horizon", 10000)
        key = (row["task_id"], row["configuration_id"], row["horizon"])
        if key not in lookup or key in observed:
            raise ForecastError("Unknown or duplicate outcome")
        observed.add(key)
        predicted = lookup[key]
        if row["task_sha256"] != predicted["task_sha256"] or row["final_status"] not in STATUSES:
            raise ForecastError("Outcome hash or status mismatch")
        recorded = stamp(row["collected_at"], "outcome collection")
        if recorded <= frozen or recorded > collection_time:
            raise ForecastError("Outcome collection must be after freeze and not in the future")
        error = int(row["final_status"] != "correct")
        losses = {b: (predicted["predicted_error"][b] - error)**2 for b in BASELINES}
        status_counts[row["final_status"]] += 1
        rows.append({**predicted, "observed_error": error, "final_status": row["final_status"],
                     "brier": losses, "paired_loss_difference": losses["pooled"] - losses["stratified"]})
    if observed != set(lookup):
        raise ForecastError("Incomplete scheduled holdout; record missing replies explicitly")
    # Average instances within a task cluster, then clusters within each stratum.
    # Model/configuration and horizon panels remain paired inside a cluster.
    cluster_rows = defaultdict(list)
    cell_rows = defaultdict(lambda: defaultdict(list))
    for row in rows:
        cluster_rows[(row["stratum"], row["cluster_id"])].append(row)
        cell_rows[(row["configuration_id"], row["horizon"])][(row["stratum"], row["cluster_id"])].append(row)

    def summary(grouped):
        by_stratum = {s: {b: [] for b in BASELINES} for s in plan["strata_weights"]}
        deltas = {s: [] for s in plan["strata_weights"]}
        for (stratum, _), rr in sorted(grouped.items()):
            for b in BASELINES:
                by_stratum[stratum][b].append(statistics.mean(x["brier"][b] for x in rr))
            deltas[stratum].append(statistics.mean(x["paired_loss_difference"] for x in rr))
        mean_loss = {b: sum(w * statistics.mean(by_stratum[s][b]) for s, w in plan["strata_weights"].items()) for b in BASELINES}
        return {"mean_brier": mean_loss, "pooled_minus_stratified": _weighted(deltas, plan["strata_weights"])}, deltas

    overall, differences = summary(cluster_rows)
    ci = None
    if all(len(x) >= 2 for x in differences.values()):
        rng = random.Random(seed)
        sample_differences = []
        for _ in range(bootstrap):
            draw = {s: rng.choices(v, k=len(v)) for s, v in differences.items()}
            sample_differences.append(_weighted(draw, plan["strata_weights"]))
        ci = [_quantile(sample_differences, .025), _quantile(sample_differences, .975)]

    interventions = []
    by_task = {(r["task_id"], r["configuration_id"], r["horizon"]): r for r in rows}
    for pair in plan["intervention_pairs"]:
        for horizon in plan["horizons"]:
            groups = defaultdict(list)
            for task in plan["prediction_tasks"]:
                control = by_task[(task["task_id"], pair["control"], horizon)]
                treatment = by_task[(task["task_id"], pair["treatment"], horizon)]
                actual = treatment["observed_error"] - control["observed_error"]
                predicted = {b: treatment["predicted_error"][b] - control["predicted_error"][b] for b in BASELINES}
                groups[(task["stratum"], task["cluster_id"])].append((actual, predicted))
            actual_by_s, predicted_by_b = {s: [] for s in plan["strata_weights"]}, {b: {s: [] for s in plan["strata_weights"]} for b in BASELINES}
            for (stratum, _), rr in groups.items():
                actual_by_s[stratum].append(statistics.mean(r[0] for r in rr))
                for b in BASELINES:
                    predicted_by_b[b][stratum].append(statistics.mean(r[1][b] for r in rr))
            actual = _weighted(actual_by_s, plan["strata_weights"])
            predicted = {b: _weighted(predicted_by_b[b], plan["strata_weights"]) for b in BASELINES}
            interventions.append({**pair, "horizon": horizon, "observed_error_difference": actual,
                                  "predicted_error_difference": predicted,
                                  "absolute_effect_forecast_error": {b: abs(predicted[b] - actual) for b in BASELINES}})

    result = {"schema": REPORT, "status": "development-only", "confirmatory_eligible": False,
              "proposition_verdict": None, "evidence_origin": plan["evidence_origin"],
              "ledger_sha256": expected_sha256, "outcomes_sha256": digest(outcomes),
              "scheduled_rows": len(lookup), "scored_rows": len(rows),
              "status_counts": {k: status_counts[k] for k in sorted(STATUSES)},
              "clusters_by_stratum": {s: len(v) for s, v in differences.items()},
              "target_strata_weights": plan["strata_weights"], "overall": overall,
              "by_configuration_horizon": [{"configuration_id": key[0], "horizon": key[1], **summary(value)[0]} for key, value in sorted(cell_rows.items())],
              "intervention_effect_forecasts": interventions,
              "paired_cluster_bootstrap": {"level": .95, "replicates": bootstrap, "seed": seed,
                  "pooled_minus_stratified_interval": ci,
                  "interpretation": "Development percentile interval; clusters resampled within predeclared strata; not multiplicity-adjusted or a validated confirmatory procedure."},
              "aggregation": "Equal instance weight within each task cluster; equal clusters within each stratum; fixed target stratum weights. Configurations and horizons have equal weight in the overall summary.",
              "limitations": ["The locally retained hash and declared timestamps do not authenticate external preregistration or model execution.",
                  "Plug-in forecasts do not propagate calibration-rate uncertainty.",
                  "Brier scores concern the specified unsuccessful-output contract, not general alignment.",
                  "No ARC-specific prediction model is implemented or established.",
                  "Strata and target weights must precede holdout outcomes; metadata declarations require independent audit."],
              "rows": sorted(rows, key=lambda r: (r["task_id"], r["configuration_id"], r["horizon"]))}
    result["report_sha256"] = digest(result)
    return result


def mixture_counterexample():
    e1 = .5 * transition_forecast(0, .8, 1, 1) + .5 * transition_forecast(0, .2, 1, 1)
    e2 = .5 * transition_forecast(0, .8, 1, 2) + .5 * transition_forecast(0, .2, 1, 2)
    pooled = transition_forecast(0, .5, 1, 2)
    return {"status": "exact synthetic counterexample, not empirical evidence", "initial_error": 1,
            "stratum_repair_probabilities": [.8, .2], "stratum_regression_probabilities": [0, 0],
            "weights": [.5, .5], "mixture_error_depth_1": e1,
            "mixture_error_depth_2": e2, "pooled_prediction_depth_2": pooled,
            "explanation": "The remaining errors become concentrated in the harder stratum although each stratum has stationary transitions."}


def demo_inputs():
    tasks, prediction, records = [], [], []
    for stratum, repairs in (("easier", 8), ("harder", 2)):
        for initial in (0, 1):
            for i in range(10):
                tid = f"cal-{stratum}-{initial}-{i}"
                task = {"task_id": tid, "cluster_id": tid, "task_sha256": digest({"synthetic-task": tid}),
                        "stratum": stratum, "initial_error": initial, "initial_state_basis": "known synthetic fixture"}
                tasks.append(task)
                records.append({"task_id": tid, "task_sha256": task["task_sha256"], "configuration_id": "mock-policy",
                                "final_status": "correct" if initial == 0 or i < repairs else "incorrect"})
        for i in range(25):
            tid = f"pred-{stratum}-{i}"
            prediction.append({"task_id": tid, "cluster_id": tid, "task_sha256": digest({"synthetic-task": tid}),
                               "stratum": stratum, "initial_error": 1, "initial_state_basis": "known synthetic fixture"})
    plan = {"schema": PLAN, "purpose": "development-only", "evidence_origin": "synthetic",
            "population": "Declared two-stratum synthetic finite fixture; not sampled empirical evidence",
            "calibration_completed_at": now_iso(), "stratum_rule_sha256": digest("fixed synthetic generator label"),
            "scorer_sha256": digest("synthetic binary output contract"), "strata_weights": {"easier": .5, "harder": .5},
            "configurations": [{"configuration_id": "mock-policy", "model_revision": "synthetic-fixture-v1", "policy_sha256": digest("mock-policy")}],
            "horizons": [1, 2], "smoothing": {"success_pseudocount": 0, "failure_pseudocount": 0},
            "calibration_tasks": tasks, "prediction_tasks": prediction,
            "missing_rule": "explicit_missing_or_invalid_is_unsuccessful", "intervention_pairs": []}
    return plan, records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    frozen = commands.add_parser("freeze")
    frozen.add_argument("--plan", type=Path, required=True)
    frozen.add_argument("--calibration", type=Path, required=True)
    frozen.add_argument("--out", type=Path, required=True)
    score = commands.add_parser("score")
    score.add_argument("--ledger", type=Path, required=True)
    score.add_argument("--expected-ledger-sha256", required=True)
    score.add_argument("--outcomes", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    score.add_argument("--bootstrap", type=int, default=2000)
    score.add_argument("--seed", type=int, default=20261010)
    demo = commands.add_parser("demo")
    demo.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            ledger = freeze(read_json(args.plan), read_json(args.calibration))
            write_new(args.out, ledger)
            print(canonical({"ledger_sha256": ledger["ledger_sha256"], "status": ledger["status"]}))
        elif args.command == "score":
            report = evaluate(read_json(args.ledger), read_json(args.outcomes), args.expected_ledger_sha256, args.bootstrap, args.seed)
            write_new(args.out, report)
            print(canonical({"status": report["status"], "evidence_origin": report["evidence_origin"], "overall": report["overall"]}))
        else:
            args.out_dir.mkdir(parents=True, exist_ok=False)
            plan, records = demo_inputs()
            ledger = freeze(plan, records)
            outcomes = []
            for predicted in ledger["forecasts"]:
                i = int(predicted["task_id"].rsplit("-", 1)[1])
                count = {("easier", 1): 5, ("harder", 1): 20, ("easier", 2): 1, ("harder", 2): 16}[(predicted["stratum"], predicted["horizon"])]
                outcomes.append({k: predicted[k] for k in ("task_id", "task_sha256", "configuration_id", "horizon")}
                                | {"final_status": "incorrect" if i < count else "correct", "collected_at": now_iso()})
            report = evaluate(ledger, outcomes, ledger["ledger_sha256"])
            for name, value in (("plan.json", plan), ("calibration.json", records), ("ledger.json", ledger),
                                ("outcomes.json", outcomes), ("evaluation.json", report), ("mixture-counterexample.json", mixture_counterexample())):
                write_new(args.out_dir / name, value)
            print(canonical({"status": "synthetic development demonstration only", **mixture_counterexample(), "ledger_sha256": ledger["ledger_sha256"]}))
    except (ValueError, TypeError, KeyError, OSError, RecursionError, OverflowError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
