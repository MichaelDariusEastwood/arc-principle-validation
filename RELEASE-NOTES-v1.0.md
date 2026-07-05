# Release Notes — `arc-principle-validation` v1.0 (First Public Release)

**Repository:** [github.com/MichaelDariusEastwood/arc-principle-validation](https://github.com/MichaelDariusEastwood/arc-principle-validation)
**Canonical DOI:** [10.17605/OSF.IO/6C5XB](https://doi.org/10.17605/OSF.IO/6C5XB)
**Author:** Michael Darius Eastwood — Independent researcher, London — ORCID [0009-0003-8483-8512](https://orcid.org/0009-0003-8483-8512)
**Licence:** MIT
**Companion toolkit:** [arc-scaling-challenge](https://github.com/MichaelDariusEastwood/arc-scaling-challenge)

This is the first open-access release of the ARC / Eden research programme: the full document suite, the experiment results, and the supporting code, published under an open licence so that anyone can read, reproduce, or attempt to falsify the work.

---

## What this release is — and is not

This is an **open research programme**, not a finished, peer-reviewed result. None of the papers has undergone formal external human peer review; every empirical finding comes from a single research programme; and no result is claimed to be proven at frontier scale. The programme is published precisely so that independent groups can replicate or refute it. **The author does not claim to have solved AI alignment.**

The honest claims summary below states what is defensible, what needs replication, and what is speculative. Read it before citing anything.

---

## Honest claims summary

### Lead findings (the two strongest, defensible results)

1. **Unblinded LLM-as-judge evaluation can be directionally wrong.** Under a four-layer blind evaluation protocol, the measured alignment-vs-reasoning-depth effect *reversed sign* for two frontier model families versus unblinded scoring: DeepSeek (ρ = +0.354 → −0.135) and Gemini (ρ = +0.311 → −0.246). In other words, 2 of 4 models reversed or collapsed once the scorers could no longer see model identity, reasoning depth, or ordering. This is the programme's strongest, most portable contribution because it stands even if the rest of the framework is wrong: it is a methodological warning about how AI-safety evaluation should be conducted (the clinical-trial blinding analogue applied to LLM-as-judge). It still needs independent-lab confirmation. (Papers IV.d, IX.)

2. **A stakeholder-care intervention robustly improves measured ethical reasoning.** Having a model identify *who is affected and how* before answering — the measurable output of the "Stewardship Gene" / Love Loop — improved measured stakeholder care across five analysable frontier models (Fisher-combined p ≈ 6.3×10⁻²¹). This is the single most universal alignment improvement observed in the suite. **Caveat (important):** this experiment was **not** run under the full four-layer ARC-Align blinding (cross-model scoring only), so part of the effect could reflect scorer bias; the programme itself rates it as sitting "between proven and supported", and blind replication is the number-one next priority. (Paper V.)

### Methodology asserted as the author's original contributions

- **The four-layer / double-blinding evaluation methodology** (identity laundering, depth laundering, order randomisation, evaluator bias-suppression, self-excluding cross-model scoring) is asserted as an original contribution of this programme.
- **The cross-domain Cauchy unification** — the claim that the three observed scaling families (power law, exponential, saturation) all follow from Cauchy's four functional equations (1821), and that this has a physically testable cross-domain consequence — is asserted as an original contribution. (Papers VII, Foundational, On the Origin of Scaling Laws.)

These are *asserted* originals, not adjudicated priority. They are offered for the community to assess.

### Where there is no priority claim

- **Sequential recursion outperforms parallel recursion** (αₛₑ𝑞 > αₚₐᵣ; parallel ≈ 0 across all six models tested) is a defensible *directional* finding here, but **no priority is claimed for it.** It is independently corroborated by — and this programme *cites* — Sharma & Chopra, *The Sequential Edge: Inverse-Entropy Voting Beats Parallel Self-Consistency at Matched Compute*, [arXiv:2511.02309](https://arxiv.org/abs/2511.02309) (4 November 2025), who report sequential beating parallel in 95.6% of tested configurations at matched compute. Credit for that specific finding belongs to its authors.

### Acknowledged prior art

- **The bare geometric exponent α = d/(d+1)** is acknowledged **prior art**. It has been independently derived by multiple groups (West–Brown–Enquist 1997; Banavar et al.; Demetrius; and others) over the past three decades. This programme does **not** claim to have originated the formula; the claimed contribution is only the *Cauchy unification* of these derivations (see above).

### What is explicitly *not* established

- **Super-linear sequential scaling (α > 1, "compounding").** An early single-model estimate (α ≈ 2.24, DeepSeek R1, 12 problems) **did not replicate across architectures.** The robust cross-architecture estimate is α ≈ 0.49 (sub-linear; Gemini 3 Flash, r² = 0.86), and only 1 of the multi-model set fell in a cleanly measurable scaling range. **α > 1 is not established cross-architecturally** and α = 2.24 is treated by the programme as an inflated single-model artefact. Do not cite α > 1 as a result.
- **Weight-level structural entanglement of safety and capability** is **inconclusive** at the training scale tested (catastrophic forgetting in the LoRA experiment; the removal test did not produce a clean phase transition). The one consistent finding is the narrower claim that embedded safety imposes *zero measurable capability cost*; whether it produces measurable *benefit* remains open. (Paper VIII: 1 positive, 2 null/inconclusive of 3 experiments.)
- **Cosmological / "recursion-as-creation" (HRIH) framings** are speculative and untested; the empirical programme does not rest on them.

---

## What's included

The complete **18-document suite** lives under `papers/`, each in its own folder with `README.md`, the canonical HTML/PDF, and (where applicable) `experiments/`, `results/`, and `figures/`.

**12 research papers**

| # | Paper | One-line finding |
|---|-------|------------------|
| I | ARC Principle | Formalisation: U = I × R^α. Preliminary, two-data-point sequential estimate. |
| II | Experimental Validation | Sequential beats parallel across six models; cross-architecture α ≈ 0.49 (not super-linear). |
| III | Alignment Scaling Problem | Three-tier, architecture-dependent alignment hierarchy; 13 falsification criteria. |
| IV.a | Baked-In vs Computed Alignment | Positive / flat / negative alignment response classes under blinded depth. |
| IV.b | Alignment Saturation at Low Depth | Saturation shape is architecture-dependent, not universal. |
| IV.c | ARC-Align Benchmark | A *candidate* blind benchmark for depth-variable alignment (not yet a field standard). |
| IV.d | Effect of Blinding | **Unblinded scoring can reverse the sign of an alignment result** (lead finding 1). |
| V | Stewardship Gene | **Stakeholder care is the most robust intervention** (lead finding 2); blinding gap noted. |
| VI | Honey Architecture | In toy self-modifying nets, an entangled C×S loss prevents capability-only collapse. |
| VII | Cauchy Unification | 19/25 empirical domains prefer the predicted family under strict AICc (p = 1.56×10⁻⁵). |
| VIII | Load-Bearing Test | 1 positive, 2 null/inconclusive; embedded safety = zero capability cost (benefit open). |
| IX | Synthesis and Roadmap | Five-tier evidence hierarchy; honest tally; four-phase replication roadmap. |

**6 supporting documents:** Foundational (axioms; α = 1/(1−β), ARC bound α ≤ 2), On the Origin of Scaling Laws (cross-domain d/(d+1) catalogue), Eden Engineering (protocol spec, v6.1), Eden Vision (philosophical position paper), Executive Summary (grant-facing overview), Master Table of Contents (navigation).

**Code, results and data**
- Per-paper `experiments/` scripts and raw `results/` JSON/TXT for the validation, alignment, Cauchy, Honey, and Load-Bearing experiments.
- The v5 alignment blind-evaluation **outputs** (`papers/Paper-IV-c-ARC-Align-Benchmark/results/v5-final/*.json`, mirrored under Papers III and IV.a–d).
- Experiments read the shared Eden gateway (`EDEN_GATEWAY_URL` / `EDEN_GATEWAY_API_KEY`) or provider env vars. **No API keys are shipped.**

---

## Known gap — the blind-evaluation harness code

The repository ships the **outputs** of the v5 four-layer blind evaluation but, as of this release, **not the harness code that produced them.** A reviewer reading Papers IV.c / IV.d can see the result JSONs (which self-document the schema: `version 5.0`, `blinding_protocol "4-layer"`, `laundering true`, 6–7 blind scorers, the depth configs, `prompt_id`, `score1..scoreN`, `response_hash`) but cannot yet re-run the pipeline end-to-end from source.

This is disclosed honestly rather than papered over. The harness, plus standalone `prompts/` and `rubric/` artefacts, is being prepared for a follow-up release; the output schema and the four-layer blinding protocol are fully documented in the result files in the meantime. The programme's position is explicit: honesty over completeness — a fabricated harness would be worse than an admitted gap. See `PUBLISH-CHECKLIST.md` §3–§4 for the exact contract the released harness must satisfy.

---

## How to cite

This repository ships a machine-readable [`CITATION.cff`](CITATION.cff). On GitHub, use the **"Cite this repository"** button (top-right of the repo page) to export APA or BibTeX.

**Preferred citation:**

> Eastwood, M.D. (2026). *The ARC Principle and Eden Protocol: Recursive Intelligence Scaling and Embedded AI Alignment.* OSF. https://doi.org/10.17605/OSF.IO/6C5XB

```bibtex
@misc{eastwood2026arc,
  author    = {Eastwood, Michael Darius},
  title     = {The ARC Principle and Eden Protocol: Recursive Intelligence
               Scaling and Embedded AI Alignment},
  year      = {2026},
  publisher = {OSF},
  doi       = {10.17605/OSF.IO/6C5XB},
  url       = {https://doi.org/10.17605/OSF.IO/6C5XB}
}
```

The canonical project DOI is **10.17605/OSF.IO/6C5XB**. If a Zenodo software DOI is minted for this GitHub release (see `ZENODO-OSF-RELEASE-RUNBOOK.md`), it will be added to `CITATION.cff` as a secondary identifier; the OSF DOI remains the primary, authoritative citation to avoid fragmenting references.

**When citing the sequential-vs-parallel result, also cite the corroborating prior/parallel art:**

> Sharma, A. & Chopra, P. (2025). *The Sequential Edge: Inverse-Entropy Voting Beats Parallel Self-Consistency at Matched Compute.* arXiv:2511.02309. https://arxiv.org/abs/2511.02309

**Related:** Eastwood, M.D. (2026). *Infinite Architects: Intelligence, Recursion, and the Creation of Everything* (ISBN 978-1806056200).

---

## Falsification welcome

Thirteen explicit falsification criteria are specified across the framework (Paper III §4; Eden Engineering §11), each independently sufficient to refute the relevant claim. The companion [arc-scaling-challenge](https://github.com/MichaelDariusEastwood/arc-scaling-challenge) toolkit exists specifically so that others can measure α in their own systems and try to break the framework. If you find evidence that contradicts the ARC Principle, please open an issue or submit a pull request. Either outcome advances the science.

---

*First public release. Canonical DOI 10.17605/OSF.IO/6C5XB. MIT licence. British English throughout.*
