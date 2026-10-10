#!/usr/bin/env python3
"""Run the complete OFFLINE SYNTHETIC instrumentation demonstration."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "analysis"))
import stage_a
from analyze_stage_a import make_plan, analyse


def run(out, pairs):
    root = Path(out)
    root.mkdir(parents=True, exist_ok=False)
    study, ledger = stage_a.demo_study(pairs), []
    stage_a.write_new(root / "study.json", study)
    plan = make_plan([i["id"] for i in study["panel"]["items"]], [m["block_id"] for m in study["models"]],
                     study["replicate_ids"], study["scorer_sha256"], study["study_sha256"],
                     study_id=study["study_id"], dataset_kind="synthetic", independence_basis="not_established", resamples=2000)
    # Save analysis choices before producing the synthetic observations too.
    stage_a.write_new(root / "analysis-plan.json", plan)
    wave = 0
    while True:
        ready = stage_a.pending(study, ledger)
        if not ready:
            break
        wave += 1
        stage_a.write_new(root / f"requests-{wave:02d}.jsonl", ready, True)
        incoming = [stage_a.synthetic_response(study, req) for req in ready]
        ledger = stage_a.import_responses(study, ledger, incoming)
    endpoints = stage_a.endpoint_records(study, ledger)
    report = analyse(plan, endpoints)
    stage_a.write_new(root / "responses.jsonl", ledger, True)
    stage_a.write_new(root / "endpoints.jsonl", endpoints, True)
    stage_a.write_new(root / "analysis-report.json", report)
    stage_a.write_new(root / "quality.json", stage_a.quality_report(study, ledger, endpoints))
    return {"status": "SYNTHETIC_INSTRUMENT_DEVELOPMENT_ONLY", "directory": str(root),
            "scheduled_calls": study["planned_model_calls"], "recorded_calls": len(ledger),
            "dependency_waves": wave, "endpoints": len(endpoints), "registered": False,
            "confirmatory_eligible": False, "inference_calls_executed": 0}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--pairs", type=int, default=2)
    a = p.parse_args()
    try:
        print(stage_a.canonical(run(a.out, a.pairs)))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        p.exit(2, f"ERROR: {exc}\n")
