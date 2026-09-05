# Open-Core Reproducibility & Re-Publish Plan — ARC / Eden Programme

**Repo:** arc-principle-validation · **Companion:** arc-scaling-challenge (toolkit) · **DOI:** 10.17605/OSF.IO/6C5XB
**Prepared:** 2026-06-15 · Staged for review — **NOT yet pushed to GitHub.**

## Principle
Open the science; keep the product closed. Re-publishing the reproducibility materials *protects* priority (timestamped, citable) and drives credibility + AI ingestion. The commercial app stays private. This is the open-core / dual-licensing model.

## PUBLISH (this repo + arc-scaling-challenge)
- Papers (18) + results JSONs — incl. `papers/Paper-IV-d-.../results/v5-final/*.json` (reproduce the blinding sign-flip) and `papers/Paper-V-Stewardship-Gene/results/*` (care result, Fisher p≈6.3×10⁻²¹).
- Experiment and validation code: `experiments/`, mapped folder by folder in `experiments/README.md`, and the per-paper experiments under `papers/<Paper>/experiments/`.
- Scaling toolkit: `arc-scaling-challenge` (α/β estimators, falsification criteria, protocols) — already public-ready.
- `CITATION.cff` (added today → gives a "Cite this repository" button), the dual `LICENCE` (present), OSF DOI link (present in README badges).

## How to reproduce the headline findings
1. **Blinding sign-flip (strongest, most defensible claim):** Paper IV-d v5-final per-model JSONs → DeepSeek V3.2 and Gemini 3 Flash reverse the sign of alignment-vs-depth vs unblinded scoring (recompute Spearman on `consensus_weighted_mean`).
2. **Care result:** Paper V matched-pair JSONs → stakeholder_care (eden − control), Fisher-combined p ≈ 6.3×10⁻²¹.
3. **Scaling form:** arc-scaling-challenge α-estimator on sequential vs parallel data.

## DO NOT PUBLISH (keep private)
- Eden Legal AI app (apps-script-project react-app), production pipeline, API keys, gateway internals.
- `sdk/python/legal_ai/eden.py` production Love-Loop implementation.
- Full raw datasets containing any third-party or personal data.
- Any hardware / "caretaker doping" / substrate implementation that may be patentable — **UK patentability is lost on public disclosure; get IP advice first.**

## Gaps to close before pushing
- **Blind-harness code (Papers IV) is NOT in any connected folder — only its outputs are.** Locate the v5 alignment-scaling + eden-intervention harness, redact secrets, add under `experiments/alignment/`. This is the single most-requested reproducibility artefact.
- **Prompts & scoring rubric** are embedded in the papers, not separate files. Extract a clean `prompts/` + `rubric/` (redact any secrets/PII) before publishing.
- Add ORCID + Google Scholar IDs to `CITATION.cff` once created.

## Prior work and honest attribution (this is what protects credibility)
- **Sequential > parallel:** independently reported by **Sharma & Chopra, arXiv:2511.02309 (4 Nov 2025)** — cite; do NOT claim priority.
- **α≈2.24** was a single-model (DeepSeek R1) estimate that did not replicate cross-architecturally (cross-arch ≈0.49). Robust claim: recursion *form* governs the exponent.
- **Creation theory / recursion-as-creation / fine-tuning** have substantial prior work (Teilhard, Wheeler, Smolin, complexity theory) — frame ARC as a named *synthesis*, not a first.
- **Genuinely distinct (claim these):** the ARC structural/power-law synthesis; architecture-dependent alignment scaling; the **blinding metascience finding**; the **Eden care-loop** result.

## Licence
Superseded: the code is proprietary and the papers are CC BY-NC-ND 4.0; see LICENCE. (This line recorded an earlier plan of MIT for code and CC-BY for data, which the dual licence replaced.) README carries the Sharma citation.
