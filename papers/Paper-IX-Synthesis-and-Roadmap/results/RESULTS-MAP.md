# Paper IX Results Map

Paper IX (Synthesis and Roadmap) does not contain original experiments. Every empirical claim in Paper IX traces to source data in another paper's results directory. This file maps each claim to its source.

## Claim-to-Source Mapping

| Claim in Paper IX | Source Paper | Source Data Path |
|-------------------|-------------|------------------|
| Three-tier alignment hierarchy (Embedded > Hybrid > External) | Paper IV.a | `papers/Paper-IV-a-Baked-In-vs-Computed-Alignment/results/v5-final/` |
| Stakeholder care improvement (Fisher p = 6.3 x 10^-21) **[withdrawn, see Corrections]** | Paper V | `papers/Paper-V-Stewardship-Gene/results/` |
| 19/25 empirical domains preferred Cauchy family (p = 1.56 x 10^-5) **[superseded, see Corrections]** | Paper VII | `papers/Paper-VII-Cauchy-Unification/results/results_50_domain_validation.json` |
| DGM Eden matches Babylon capability at 0.667 | Paper VIII | `papers/Paper-VIII-The-Load-Bearing-Proof/results/eden_dgm_results.json` |
| Gated simulation confirms Babylon reward-hacking fingerprint | Paper VIII | `papers/Paper-VIII-The-Load-Bearing-Proof/results/gated_selfmod_evaluation_report.json` |
| Weight-level entanglement inconclusive at current scale | Paper VIII | `papers/Paper-VIII-The-Load-Bearing-Proof/results/eden_weight_results.json` |
| Removal of safety collapses all capability (p = 0.04) | Paper VIII | `papers/Paper-VIII-The-Load-Bearing-Proof/results/eden_dgm_results.json` |
| Blinding reverses alignment measurements | Paper IV.d | `papers/Paper-IV-d-The-Effect-of-Blinding-on-AI-Alignment-Evaluation/` |
| Alignment saturation at low depth | Paper IV.b | `papers/Paper-IV-b-Alignment-Saturation-at-Low-Depth/` |
| Sequential alpha > 1, parallel alpha approx 0 **[not established, see Corrections]** | Paper II | `papers/Paper-II-Experimental-Validation/` |
| 50-domain negative controls (0% axiom-violating match) | Paper VII | `papers/Paper-VII-Cauchy-Unification/results/negative_control_results.json` |
| Entangled loss co-descends smoothly | Paper VIII | `papers/Paper-VIII-The-Load-Bearing-Proof/results/eden_weight_results.json` |

## Corrections

Three rows above are marked. Each is quoted here in the wording the repository already carries, transliterated to this file's ASCII convention.

- **Stakeholder care improvement (Fisher p = 6.3 x 10^-21), Paper V.** The Fisher-combined figure of 6.3 x 10^-21 is withdrawn under AQ-017, because independence among the five tests was never established; the five per-model results stand on their own. (README.md, the Paper V row of the claim table; docs/release-notes-v1.0.md.)
- **19/25 empirical domains preferred Cauchy family (p = 1.56 x 10^-5), Paper VII.** Paper VII is under correction as of 11 August 2026. The grid has four cells, not three; the three-family framing and the derived-from-Cauchy treatment of bounded curves are withdrawn; the primary statistic is a permutation test conditioned on both marginals, not the binomial reported previously. The statement previously given, 19/25 empirical domains prefer the predicted family under strict AICc (p = 1.56 x 10^-5), is superseded. (README.md, the Paper VII row; docs/release-notes-v1.0.md.)
- **Sequential alpha > 1, parallel alpha approx 0, Paper II.** Alpha > 1 is not established cross-architecturally. The robust cross-architecture estimate is alpha approximately 0.49, sub-linear. Do not cite alpha > 1 as a result. (docs/release-notes-v1.0.md, under "What is explicitly not established".)

The source paths are unchanged and every row is kept. A marked row records what the source data was read to show; it is not a standing claim of the programme.

## Notes

All paths are relative to the repository root (`arc-principle-validation/`).

Paper IX synthesises findings from across the programme. No results file in this directory should be treated as primary evidence. Always cite the source paper.
