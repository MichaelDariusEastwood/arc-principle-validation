#!/usr/bin/env python3
"""Offline paired-cluster analysis candidate. No model calls or registration.

This module accepts complete binary endpoint records for the Stage A grid.
Synthetic and pilot data remain development data. Neither a small p-value nor
a label in a JSON file makes this a registered or confirmatory analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import random
import re
import statistics
import sys


PLAN_SCHEMA = "rd-stage-a-analysis-plan/0.1"
RECORD_SCHEMA = "rd-stage-a-endpoint/0.1"
REPORT_SCHEMA = "rd-stage-a-analysis-report/0.1"
STUDY_ID = "RD-EXT-20261010-DIAGNOSTIC-PLACEMENT-CANDIDATE"
CONTRASTS = (
    "diagnostic_benefit_d4",
    "placement_advantage_d4",
    "placement_widening_d4_minus_d2",
)
FEEDBACK = ("diagnostic", "neutral_masked")
PLACEMENT = ("during_revision", "terminal_panel")
DEPTHS = (2, 4)
RANGES = ((-1.0, 1.0), (-1.0, 1.0), (-2.0, 2.0))
STATUS = ("valid", "invalid", "missing", "infrastructure_failure")
PLAN_FIELDS = {
    "schema", "study_id", "suite_id", "analysis_status", "dataset_kind",
    "task_cluster_ids", "model_block_ids", "replicate_ids", "independence_basis",
    "source_manifest_sha256", "scorer_sha256", "analysis", "plan_sha256",
}
ANALYSIS_FIELDS = {"resamples", "seed", "familywise_alpha", "minimum_effect", "primary_contrasts"}
RECORD_FIELDS = {
    "schema", "study_id", "dataset_kind", "task_cluster_id", "model_block_id",
    "replicate_id", "feedback", "placement", "depth", "initial_artefact_sha256",
    "endpoint_status", "initial_success", "final_success", "trace_sha256", "scorer_sha256",
}
MAX_RECORDS = 500_000
MAX_BOOTSTRAP_DRAWS = 10_000_000
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class AnalysisError(ValueError):
    """An input cannot be analysed under the declared design."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AnalysisError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise AnalysisError(f"non-finite JSON constant: {value}")


def parse_json(value):
    return json.loads(value, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def _fields(value, expected, name):
    if not isinstance(value, dict) or set(value) != expected:
        raise AnalysisError(f"{name} must contain exactly {sorted(expected)}")


def _integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise AnalysisError(f"{name} must be an integer from {low} to {high}")
    return value


def _number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"{name} must be a finite number")
    try:
        valid = math.isfinite(value) and low <= value <= high
    except OverflowError:
        valid = False
    if not valid:
        raise AnalysisError(f"{name} must be finite and in [{low}, {high}]")
    return float(value)


def _identifier(value, name):
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise AnalysisError(f"{name} must be a nonempty ASCII identifier of at most 128 characters")
    return value


def _hash(value, name):
    if not isinstance(value, str) or not HASH_PATTERN.fullmatch(value):
        raise AnalysisError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _id_list(value, name, low, high):
    if not isinstance(value, list) or not low <= len(value) <= high:
        raise AnalysisError(f"{name} must have {low} to {high} entries")
    for item in value:
        _identifier(item, name)
    if len(value) != len(set(value)):
        raise AnalysisError(f"duplicate identifier in {name}")
    return value


def make_plan(task_cluster_ids, model_block_ids, replicate_ids, scorer_sha256,
              source_manifest_sha256, *, dataset_kind="synthetic",
              suite_id="arithmetic_development", study_id=STUDY_ID,
              independence_basis="not_established", resamples=10_000,
              seed=20261010, alpha=0.05, minimum_effect=0.10):
    """Make a development analysis plan; hashes bind identity, not registration.

    Lists are copied. Two model blocks are fixed by this candidate. A cluster
    ID is one task unit; related templates must be grouped before using this
    interface. Relabelling related tasks does not create independence.
    """
    plan = {
        "schema": PLAN_SCHEMA, "study_id": study_id, "suite_id": suite_id,
        "analysis_status": "instrument_development",
        "dataset_kind": dataset_kind,
        "task_cluster_ids": list(task_cluster_ids),
        "model_block_ids": list(model_block_ids),
        "replicate_ids": list(replicate_ids),
        "independence_basis": independence_basis,
        "source_manifest_sha256": source_manifest_sha256,
        "scorer_sha256": scorer_sha256,
        "analysis": {
            "resamples": resamples, "seed": seed,
            "familywise_alpha": alpha, "minimum_effect": minimum_effect,
            "primary_contrasts": list(CONTRASTS),
        },
    }
    plan["plan_sha256"] = digest(plan)
    return validate_plan(plan)


def validate_plan(plan):
    _fields(plan, PLAN_FIELDS, "plan")
    if plan["schema"] != PLAN_SCHEMA or plan["analysis_status"] != "instrument_development":
        raise AnalysisError("only this instrument-development plan schema is supported")
    _identifier(plan["study_id"], "study_id")
    _identifier(plan["suite_id"], "suite_id")
    if plan["dataset_kind"] not in ("synthetic", "pilot"):
        raise AnalysisError("dataset_kind must be synthetic or pilot; confirmation is unsupported")
    basis = plan["independence_basis"]
    if basis not in ("synthetic_iid", "not_established", "declared_independent"):
        raise AnalysisError("unknown independence_basis")
    if basis == "synthetic_iid" and plan["dataset_kind"] != "synthetic":
        raise AnalysisError("synthetic_iid cannot qualify model-response pilot data")
    if basis == "declared_independent" and plan["dataset_kind"] != "pilot":
        raise AnalysisError("synthetic records cannot be relabelled as reviewed empirical clusters")
    _id_list(plan["task_cluster_ids"], "task_cluster_ids", 2, 10_000)
    _id_list(plan["model_block_ids"], "model_block_ids", 2, 2)
    _id_list(plan["replicate_ids"], "replicate_ids", 1, 50)
    _hash(plan["scorer_sha256"], "scorer_sha256")
    _hash(plan["source_manifest_sha256"], "source_manifest_sha256")
    _fields(plan["analysis"], ANALYSIS_FIELDS, "analysis")
    analysis = plan["analysis"]
    if analysis["primary_contrasts"] != list(CONTRASTS):
        raise AnalysisError("the fixed three-contrast family cannot be added to, reordered or narrowed")
    _integer(analysis["resamples"], "resamples", 1000, 50_000)
    _integer(analysis["seed"], "seed", 0, 2**63 - 1)
    _number(analysis["familywise_alpha"], "familywise_alpha", 0.001, 0.10)
    _number(analysis["minimum_effect"], "minimum_effect", 0.001, 1.0)
    n = len(plan["task_cluster_ids"])
    if n * analysis["resamples"] > MAX_BOOTSTRAP_DRAWS:
        raise AnalysisError("requested bootstrap exceeds this development implementation's draw cap")
    if n * len(plan["model_block_ids"]) * len(plan["replicate_ids"]) * 8 > MAX_RECORDS:
        raise AnalysisError("declared endpoint count exceeds development input cap")
    _hash(plan["plan_sha256"], "plan_sha256")
    if plan["plan_sha256"] != digest({k: v for k, v in plan.items() if k != "plan_sha256"}):
        raise AnalysisError("analysis plan hash mismatch")
    return plan


def _key(row):
    return (row["task_cluster_id"], row["model_block_id"], row["replicate_id"],
            row["feedback"], row["placement"], row["depth"])


def validate_records(plan, records):
    """Require the complete Cartesian grid, explicit absence and shared starts.

    The scorer and trace hashes are identity references. This function cannot
    establish the truth of outcomes or fetch evidence for those references.
    """
    validate_plan(plan)
    if not isinstance(records, list) or len(records) > MAX_RECORDS:
        raise AnalysisError("records must be a bounded list")
    clusters = set(plan["task_cluster_ids"])
    models = set(plan["model_block_ids"])
    replicates = set(plan["replicate_ids"])
    by_key, starts = {}, {}
    for row in records:
        _fields(row, RECORD_FIELDS, "endpoint record")
        if row["schema"] != RECORD_SCHEMA:
            raise AnalysisError("unexpected endpoint schema")
        if row["study_id"] != plan["study_id"] or row["dataset_kind"] != plan["dataset_kind"]:
            raise AnalysisError("endpoint study or data kind differs from plan")
        for field, allowed in (("task_cluster_id", clusters), ("model_block_id", models),
                               ("replicate_id", replicates), ("feedback", FEEDBACK),
                               ("placement", PLACEMENT), ("endpoint_status", STATUS)):
            _identifier(row[field], field)
            if row[field] not in allowed:
                raise AnalysisError(f"undeclared {field}: {row[field]}")
        _integer(row["depth"], "depth", 2, 4)
        if row["depth"] not in DEPTHS:
            raise AnalysisError("depth must be 2 or 4")
        for field in ("initial_success", "final_success"):
            if not isinstance(row[field], bool):
                raise AnalysisError(f"{field} must be a JSON boolean, never null or a self-rating")
        if row["endpoint_status"] != "valid" and row["final_success"]:
            raise AnalysisError("invalid, missing and infrastructure-failure endpoints must score false")
        for field in ("initial_artefact_sha256", "trace_sha256", "scorer_sha256"):
            _hash(row[field], field)
        if row["scorer_sha256"] != plan["scorer_sha256"]:
            raise AnalysisError("scorer digest differs from the declared instrument")
        key = _key(row)
        if key in by_key:
            raise AnalysisError("duplicate scheduled endpoint")
        by_key[key] = row
        block = key[:3]
        start = (row["initial_artefact_sha256"], row["initial_success"])
        if block in starts and starts[block] != start:
            raise AnalysisError("all policy arms must share the same initial artefact and initial success")
        starts[block] = start
    expected = len(clusters) * len(models) * len(replicates) * 8
    if len(by_key) != expected:
        raise AnalysisError(f"incomplete endpoint grid: expected {expected}, received {len(by_key)}; "
                            "record absent outputs explicitly instead of deleting scheduled outcomes")
    return by_key


def _model_task_contrasts(by_key, cluster, model, replicates):
    def score(feedback, placement, depth):
        return statistics.fmean(float(by_key[(cluster, model, r, feedback, placement, depth)]
                                      ["final_success"]) for r in replicates)
    diagnostic = statistics.fmean(score("diagnostic", p, 4) - score("neutral_masked", p, 4)
                                 for p in PLACEMENT)
    advantage4 = score("diagnostic", "during_revision", 4) - score("diagnostic", "terminal_panel", 4)
    advantage2 = score("diagnostic", "during_revision", 2) - score("diagnostic", "terminal_panel", 2)
    return (diagnostic, advantage4, advantage4 - advantage2)


def cluster_contrasts(plan, records):
    by_key = validate_records(plan, records)
    models = plan["model_block_ids"]
    reps = plan["replicate_ids"]
    result, by_model = [], {m: [] for m in models}
    for cluster in plan["task_cluster_ids"]:
        local = [_model_task_contrasts(by_key, cluster, model, reps) for model in models]
        for model, values in zip(models, local):
            by_model[model].append(values)
        result.append(tuple(statistics.fmean(values[j] for values in local) for j in range(3)))
    return result, by_model


def quantile_sorted(values, probability):
    """Linear interpolation between ordered bootstrap values (type-7 quantile)."""
    if not values:
        raise AnalysisError("quantile requires observations")
    probability = _number(probability, "quantile probability", 0, 1)
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def holm_adjust(p_values):
    """Holm adjusted p-values; dependence between contrasts is permitted.

    If marginal p-values are approximate, error control remains approximate.
    An unavailable p-value is kept in the family as 1 for adjustment and then
    restored to null; the family is never shrunk after observing outcomes.
    """
    if len(p_values) != 3:
        raise AnalysisError("Holm family must contain exactly three entries")
    values = [1.0 if p is None else _number(p, "p-value", 0, 1) for p in p_values]
    order = sorted(range(3), key=lambda i: (values[i], i))
    adjusted = [None] * 3
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (3 - rank) * values[index]))
        adjusted[index] = None if p_values[index] is None else running
    return adjusted


def hoeffding_interval(mean, count, lower, upper, alpha, family_size=3):
    """Conditional simultaneous bound for independent bounded task contrasts.

    Covers the mean of the declared expectations; it does not certify sampling,
    population representativeness, temporal independence or score validity.
    """
    width = (upper - lower) * math.sqrt(math.log(2 * family_size / alpha) / (2 * count))
    return [max(lower, mean - width), min(upper, mean + width)]


def bootstrap_cluster_means(vectors, resamples, seed):
    """One shared cluster-index draw retains all three contrasts and model arms."""
    rng = random.Random(seed)
    n = len(vectors)
    samples = [[], [], []]
    for _ in range(resamples):
        totals = [0.0, 0.0, 0.0]
        for _ in range(n):
            vector = vectors[rng.randrange(n)]
            totals[0] += vector[0]
            totals[1] += vector[1]
            totals[2] += vector[2]
        for j in range(3):
            samples[j].append(totals[j] / n)
    return samples


def _arm_summary(plan, records):
    rows = []
    for model, feedback, placement, depth in itertools.product(
            plan["model_block_ids"], FEEDBACK, PLACEMENT, DEPTHS):
        selected = [r for r in records if r["model_block_id"] == model and
                    r["feedback"] == feedback and r["placement"] == placement and r["depth"] == depth]
        good = [r for r in selected if r["initial_success"]]
        faulty = [r for r in selected if not r["initial_success"]]
        rows.append({
            "model_block_id": model, "feedback": feedback, "placement": placement, "depth": depth,
            "scheduled_endpoints": len(selected),
            "successful_endpoints": sum(r["final_success"] for r in selected),
            "mean_success": statistics.fmean(float(r["final_success"]) for r in selected),
            "initially_correct": len(good), "initially_incorrect": len(faulty),
            "correct_to_successful": sum(r["final_success"] for r in good),
            "correct_to_unsuccessful": sum(not r["final_success"] for r in good),
            "incorrect_to_successful": sum(r["final_success"] for r in faulty),
            "incorrect_to_unsuccessful": sum(not r["final_success"] for r in faulty),
            "endpoint_status_counts": {s: sum(r["endpoint_status"] == s for r in selected) for s in STATUS},
        })
    return rows


def analyse(plan, records):
    """Compute descriptive and approximate uncertainty outputs, never a verdict."""
    vectors, by_model = cluster_contrasts(plan, records)
    n = len(vectors)
    settings = plan["analysis"]
    alpha = settings["familywise_alpha"]
    resamples = settings["resamples"]
    means = [statistics.fmean(v[j] for v in vectors) for j in range(3)]
    sds = [statistics.stdev(v[j] for v in vectors) for j in range(3)]
    bootstrap = bootstrap_cluster_means(vectors, resamples, settings["seed"])
    tail = alpha / (2 * 3)
    p_values, estimates = [], []
    for j, identifier in enumerate(CONTRASTS):
        degenerate = sds[j] == 0.0
        ordered = sorted(bootstrap[j])
        # This is a centred bootstrap approximation to a zero-mean null. The
        # plus-one convention avoids a reported zero, but does not make this
        # an exact randomisation or permutation test.
        p = None if degenerate else (1 + sum(abs(value - means[j]) >= abs(means[j])
                                             for value in bootstrap[j])) / (resamples + 1)
        p_values.append(p)
        estimates.append({
            "id": identifier, "estimate": means[j], "task_clusters": n,
            "cluster_sd": sds[j], "cluster_standard_error": sds[j] / math.sqrt(n),
            "contrast_range": list(RANGES[j]),
            "bootstrap_degenerate": degenerate,
            "bootstrap_percentile_interval_bonferroni_approx": None if degenerate else [
                quantile_sorted(ordered, tail), quantile_sorted(ordered, 1 - tail)],
            "centred_bootstrap_two_sided_p_approx": p,
            "holm_adjusted_p_approx": None,
            "hoeffding_interval_bonferroni_conditional": hoeffding_interval(
                means[j], n, *RANGES[j], alpha),
            "minimum_effect_for_later_design_review": settings["minimum_effect"],
            "directional_or_practical_effect_verdict": None,
        })
    for estimate, adjusted in zip(estimates, holm_adjust(p_values)):
        estimate["holm_adjusted_p_approx"] = adjusted
    warnings = [
        "This implementation analyses development data only and cannot decide any registered proposition.",
        "Percentile intervals and centred-bootstrap p-values are approximate; Holm cannot repair invalid marginal p-values.",
        "The paired nonparametric bootstrap requires independently sampled, exchangeable task clusters appropriate to the target estimand.",
        "Bonferroni gives simultaneous coverage only to the extent that the marginal interval procedure is calibrated.",
        "The Hoeffding intervals are a separate conservative sensitivity calculation conditional on independent bounded task clusters.",
        "Pointwise, fixed-sample analysis does not cover optional stopping, adaptive task selection or distribution shift.",
        "Two fixed model blocks do not constitute a random sample of model families; replicates do not increase the cluster count.",
        "Scores and source hashes require an independently validated scorer and custody audit; this module cannot verify their truth.",
        "No equivalence test, exponent estimate, power certification, Stage B forecast test or general safety conclusion is implemented.",
    ]
    blocked = ["development_data_no_confirmatory_verdict"]
    if plan["dataset_kind"] == "synthetic":
        blocked.append("synthetic_records_are_not_empirical_evidence")
    if plan["independence_basis"] == "not_established":
        blocked.append("independent_task_clusters_not_established")
        warnings.append("Declared item identifiers do not establish independent template clusters; use all uncertainty calculations descriptively.")
    if n < 20:
        blocked.append("fewer_than_20_task_clusters")
        warnings.append("Fewer than 20 clusters: no bootstrap significance interpretation. Twenty is a guard, not a sufficiency theorem.")
    if any(e["bootstrap_degenerate"] for e in estimates):
        blocked.append("one_or_more_degenerate_bootstrap_contrasts")
        warnings.append("Constant observed contrasts do not prove zero population uncertainty; their bootstrap intervals and p-values are withheld.")
    if resamples * tail < 100:
        warnings.append("Fewer than 100 expected bootstrap draws per extreme CI tail; Monte Carlo quantile resolution needs review.")
    model_points = {
        model: {identifier: statistics.fmean(v[j] for v in values)
                for j, identifier in enumerate(CONTRASTS)}
        for model, values in by_model.items()
    }
    ordered_records = sorted(records, key=_key)
    report = {
        "schema": REPORT_SCHEMA, "study_id": plan["study_id"], "suite_id": plan["suite_id"],
        "analysis_status": "instrument_development", "dataset_kind": plan["dataset_kind"],
        "confirmatory_eligible": False, "registered": False, "proposition_verdict": None,
        "scope": "binary Stage A endpoints only; fixed model blocks; equal task and replicate weighting",
        "plan_sha256": plan["plan_sha256"],
        "source_manifest_sha256": plan["source_manifest_sha256"],
        "scorer_sha256": plan["scorer_sha256"],
        "endpoints_canonical_sha256": digest(ordered_records),
        "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_runtime": platform.python_version(),
        "scheduled_endpoints": len(records), "task_clusters": n,
        "independence_basis": plan["independence_basis"],
        "fixed_model_blocks": list(plan["model_block_ids"]),
        "replicates_per_task_model_arm": len(plan["replicate_ids"]),
        "bootstrap": {"unit": "paired_task_cluster", "resamples": resamples,
                      "seed": settings["seed"], "familywise_alpha": alpha,
                      "two_sided_tail_probability_per_contrast": tail,
                      "interval_method": "type-7 percentile with Bonferroni across fixed three contrasts",
                      "p_value_method": "centred bootstrap approximation; Holm across fixed three contrasts",
                      "all_model_arm_replicate_observations_retained_together": True,
                      "exact_finite_sample_error_control_claimed": False},
        "primary_contrasts": estimates,
        "model_specific_points_descriptive_only": model_points,
        "arm_summaries": _arm_summary(plan, records),
        "inferential_interpretation_blocks": blocked,
        "warnings": warnings,
    }
    report["report_sha256"] = digest(report)
    return report


def write_new(path, text):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        plan = parse_json(args.plan.read_text(encoding="utf-8"))
        lines = args.records.read_text(encoding="utf-8").splitlines()
        records = [parse_json(line) for line in lines if line.strip()]
        result = analyse(plan, records)
        result["raw_input_file_sha256"] = hashlib.sha256(args.records.read_bytes()).hexdigest()
        result["report_sha256"] = digest({k: v for k, v in result.items() if k != "report_sha256"})
        write_new(args.out, json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    except (ValueError, TypeError, KeyError, OSError, RecursionError, OverflowError) as exc:
        parser.error(str(exc))
    print(str(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
