#!/usr/bin/env python3
"""Create conspicuously synthetic, balanced fixtures for software verification.

No output here is an empirical model response. The fixtures are deliberately
balanced, not IID samples, and therefore do not validate interval coverage.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import analyze_stage_a as analysis


def make_fixture(kind="known_positive", clusters=32, resamples=1000):
    if kind not in ("balanced_null", "known_positive", "known_negative"):
        raise analysis.AnalysisError("unknown synthetic fixture")
    analysis._integer(clusters, "clusters", 8, 10_000)
    if clusters % 8:
        raise analysis.AnalysisError("balanced fixtures require a multiple of eight clusters")
    spec = {
        "status": "SYNTHETIC_SOFTWARE_FIXTURE_NOT_EMPIRICAL_DATA",
        "kind": kind, "clusters": clusters,
        "sampling": "deterministic balanced patterns, not IID",
        "generator": "synthetic_fixture.py/0.1",
        "expected_contrasts": {
            "balanced_null": [0.0, 0.0, 0.0],
            "known_positive": [0.375, 0.25, 0.25],
            "known_negative": [-0.375, -0.25, -0.25],
        }[kind],
    }
    # The generator itself assigns the synthetic score; it is not an
    # independent empirical checker. Bind its exact source bytes openly.
    scorer = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    spec_bytes = (analysis.canonical(spec) + "\n").encode("utf-8")
    plan = analysis.make_plan(
        [f"SYNTHETIC-{i:04d}" for i in range(clusters)],
        ["SYNTHETIC-A", "SYNTHETIC-B"], ["SYNTHETIC-REPLICATE-0"],
        scorer, hashlib.sha256(spec_bytes).hexdigest(), dataset_kind="synthetic",
        suite_id="synthetic_analysis_verification", independence_basis="not_established",
        resamples=resamples,
    )
    records = []
    for i, cluster in enumerate(plan["task_cluster_ids"]):
        for model_index, model in enumerate(plan["model_block_ids"]):
            start = analysis.digest({"synthetic_initial": i, "model": model})
            for arm_index, (feedback, placement, depth) in enumerate(itertools.product(
                    analysis.FEEDBACK, analysis.PLACEMENT, analysis.DEPTHS)):
                if kind == "balanced_null":
                    # Every arm has four successes per eight clusters. Adjacent
                    # complement patterns give zero contrasts with variation.
                    bit_index = (i + model_index + 3 * arm_index) % 8
                    result = bool((0b10010110 >> bit_index) & 1)
                else:
                    threshold = 0.25
                    if feedback == "diagnostic":
                        threshold = 0.75 if placement == "during_revision" and depth == 4 else 0.50
                    u = ((i + model_index) % 8 + 0.5) / 8
                    result = u < threshold
                    if kind == "known_negative":
                        result = not result
                row = {
                    "schema": analysis.RECORD_SCHEMA,
                    "study_id": plan["study_id"], "dataset_kind": "synthetic",
                    "task_cluster_id": cluster, "model_block_id": model,
                    "replicate_id": "SYNTHETIC-REPLICATE-0",
                    "feedback": feedback, "placement": placement, "depth": depth,
                    "initial_artefact_sha256": start,
                    "endpoint_status": "valid", "initial_success": bool(i % 2),
                    "final_success": result, "scorer_sha256": scorer,
                }
                row["trace_sha256"] = analysis.digest({"synthetic_endpoint": row})
                records.append(row)
    return spec, plan, records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["balanced_null", "known_positive", "known_negative"],
                        default="known_positive")
    parser.add_argument("--clusters", type=int, default=32)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        spec, plan, records = make_fixture(args.kind, args.clusters, args.resamples)
        args.out_dir.mkdir(parents=True, exist_ok=False)
        analysis.write_new(args.out_dir / "SYNTHETIC_SPEC.json", analysis.canonical(spec) + "\n")
        analysis.write_new(args.out_dir / "plan.json", json.dumps(plan, indent=2) + "\n")
        analysis.write_new(args.out_dir / "endpoints.jsonl",
                           "".join(analysis.canonical(row) + "\n" for row in records))
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(str(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
