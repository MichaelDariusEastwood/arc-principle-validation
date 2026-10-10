# Offline Stage A analysis candidate

This directory supplies a strict endpoint schema, paired task-cluster contrasts, approximate bootstrap uncertainty with a fixed three-comparison family, conservative conditional bounds, synthetic verification fixtures, and a CLI. It performs no API calls, purchases, code execution from models, or registration.

**All output is instrument development.** Both `synthetic` and `pilot` inputs remain nonconfirmatory. A `confirmation` label is rejected. No result decides P21, estimates ARC exponents or establishes general alignment.

## Files

| File | Purpose |
|---|---|
| `analyze_stage_a.py` | Strict plan/endpoint validation, paired estimation and uncertainty calculations |
| `synthetic_fixture.py` | Conspicuously synthetic balanced null, positive and negative examples |
| `test_analysis.py` | Meaningful checks of pairing, contrast arithmetic, failure denominators and integrity |
| `METHODS.md` | Estimands, assumptions, multiplicity, limitations and prior statistical sources |

Python 3.9 or newer, standard library only. Tests run without network access or API credentials.

## Run the verified synthetic example

From this directory:

```bash
python3 -m unittest discover -s . -p 'test_*.py' -v
python3 synthetic_fixture.py --kind known_positive --clusters 32 --out-dir example-SYNTHETIC
python3 analyze_stage_a.py --plan example-SYNTHETIC/plan.json --records example-SYNTHETIC/endpoints.jsonl --out example-SYNTHETIC/report.json
```

The generating rule gives exact sample contrasts 0.375, 0.25 and 0.25. These values are deliberately manufactured **software-fixture values**. They are not measurements of any model, estimated evidence for a theory, or calibration of bootstrap coverage. The negative fixture reverses their signs; the balanced null has three zero means. The generator refuses an existing output directory, and the analyser refuses to overwrite a report.

## Integration interface

```python
import analyze_stage_a as analysis

plan = analysis.make_plan(
    task_cluster_ids=["cluster-001", "cluster-002"],
    model_block_ids=["fixed-block-A", "fixed-block-B"],
    replicate_ids=["seed-0"],
    scorer_sha256=scorer_file_hash,
    source_manifest_sha256=manifest_file_hash,
    dataset_kind="pilot",
    suite_id="arithmetic_development",
    independence_basis="not_established",
    resamples=10000,
    seed=20261010,
)
report = analysis.analyse(plan, complete_endpoint_records)
```

The small two-cluster example demonstrates the interface only. It cannot support bootstrap significance. An arithmetic item ID is not evidence of an independent program-repair template cluster.

Required endpoint fields, with no extra keys:

| Field | Required value or meaning |
|---|---|
| `schema` | `rd-stage-a-endpoint/0.1` |
| `study_id` | Exact plan study ID |
| `dataset_kind` | Exact plan data kind: `synthetic` or `pilot` |
| `task_cluster_id` | One declared task cluster |
| `model_block_id` | One of the two declared fixed model blocks |
| `replicate_id` | A declared execution replicate identifier |
| `feedback` | `diagnostic` or `neutral_masked` |
| `placement` | `during_revision` or `terminal_panel` |
| `depth` | Integer 2 or 4 |
| `initial_artefact_sha256` | Digest shared across all eight policies in this task/model/replicate block |
| `endpoint_status` | `valid`, `invalid`, `missing`, or `infrastructure_failure` |
| `initial_success` | Independently scored JSON boolean |
| `final_success` | Independently scored JSON boolean; false for every nonvalid endpoint |
| `trace_sha256` | Digest of the immutable execution/absence record |
| `scorer_sha256` | Exact declared scorer digest |

Every task/model/replicate block must contain all eight policies. Missing responses require explicit absence records with false success, never deleted rows. Digests identify the submitted records and code; this module does not authenticate the underlying execution.

## Interpretation

Read `METHODS.md` before interpreting uncertainty. Bootstrap intervals and centred-bootstrap p-values are approximations. Holm correction cannot repair inadequate marginal calibration, and Bonferroni percentile intervals are not exact merely because their tails are adjusted. Hoeffding intervals offer a separately labelled, often broad bound conditional on independent bounded clusters. Synthetic records, uncertain cluster independence, small samples and degenerate contrasts are made visible in each report. All automatic directional and practical-effect verdicts remain null.

This is analysis infrastructure for review. The live model snapshots, valid richer tasks, held-out oracle, real pilot, statistical simulation, resource cap, scientific freeze and actual preregistration still have to be supplied before a deciding empirical study.
