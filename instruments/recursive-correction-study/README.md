# Recursive correction study: development instrument

Status: instrument development, 10 October 2026. All included generated examples are synthetic. No live model result, accepted registration or established ARC prediction is supplied here.

## Purpose

Make correction studies inspectable: preserve the initial artefact, compare eight declared feedback/scheduling policies, retain unsuccessful outcomes, analyse paired endpoints and freeze adopted transition forecasts before separately collected outcomes.

The code uses Python's standard library. It makes no network calls and requires no API key. From this directory:

```bash
python3 run_demo.py --pairs 2 --out demo-local
python3 -m unittest discover -s harness -p 'test_*.py'
python3 -m unittest discover -s analysis -p 'test_*.py'
python3 -m unittest discover -s forecast -p 'test_*.py'
python3 forecast/frozen_baselines.py demo --out-dir forecast-demo-local
```

Use new output directories. The integrated demonstration generates four arithmetic items, two synthetic model blocks, 456 artificial responses and 64 endpoints across ten dependency waves. It saves the analysis choices first. These outputs test wiring; they do not measure model performance.

## Components

| Directory | Contribution | Scope |
|---|---|---|
| `harness/` | Shared initial artefacts, eight policies, response custody, exact checker and offline Responses Batch adapter | Arithmetic development only |
| `analysis/` | Three paired task-cluster contrasts, approximate bootstrap calculations, multiplicity and conditional conservative bounds | Synthetic or pilot records only; no confirmatory verdict |
| `forecast/` | Frozen pooled and predeclared-stratum transition rivals, paired Brier scores | Adopted models; no ARC-specific fitted predictor |

Read each component's README and `analysis/METHODS.md` before collecting or interpreting responses. Python 3.11 and 3.12 are the CI targets. The development code has no third-party Python dependency.

## Scientific boundaries

The current arithmetic bank is not the proposed representative program-repair main study. Task IDs alone do not establish independent clusters. Two depths give a finite contrast, not an ARC exponent. The code preserves token exhaustion, refusal and invalid output as unsuccessful model outcomes, separately from infrastructure failures.

The forecast recurrence is established mathematics. Yang et al., *A Probabilistic Inference Scaling Theory for LLM Self-Correction*, EMNLP 2025, already uses one-round calibration to predict later self-correction accuracy: https://aclanthology.org/2025.emnlp-main.685/ . Liu and Meng's 2026 *Self-Correction as Feedback Control* preprint is another comparator: https://arxiv.org/abs/2604.22273 . The module's task-mixture counterexample is illustrative mathematics, not new empirical evidence.

Bootstrap calculations are approximate. The forecast intervals condition on fitted rates and omit calibration-parameter uncertainty. Local hashes document bytes; they do not authenticate a registration, independent task sampling or actual execution. All results remain nonconfirmatory.

Before a deciding study, bind real task and hidden-test banks, independent units, exact model revisions, measured pilot precision, a reviewed sample-size and cost plan, and an authentic prospective record. Preserve all existing P1-P22 claims. This instrument introduces no new numbered law.

## External response collection

`harness/batch_adapter.py` serialises ready requests and interprets returned files. It does not submit batches or spend funds. The operator must verify actual model availability, revision evidence, sampling support, resource limits and collection scope. API results are joined by opaque `custom_id`, never row order. See the full instructions in `harness/README.md`.

## Provenance and reuse

`UPSTREAM_PROVENANCE.json` identifies the unchanged arithmetic checker and the source hashes of this kit. Repository licence terms apply. No private manuscript, holdout answer key or correspondence is included in this instrument directory. Paper X manuscript changes are reviewed separately.
