# The ARC Principle

**A mathematical framework for recursive intelligence scaling, validated across 18 research documents**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OSF DOI](https://img.shields.io/badge/OSF-10.17605%2FOSF.IO%2F6C5XB-blue)](https://doi.org/10.17605/OSF.IO/6C5XB)

## Citation & Prior Art

This repository ships a machine-readable [`CITATION.cff`](CITATION.cff). On GitHub, use the **"Cite this repository"** button in the sidebar (top right of the repository page) to export the citation in APA or BibTeX directly from that file.

**Cite this work as:**

> Eastwood, M.D. (2026). *The ARC Principle and Eden Protocol: Recursive Intelligence Scaling and Embedded AI Alignment.* OSF. https://doi.org/10.17605/OSF.IO/6C5XB

```bibtex
@misc{eastwood2026arc,
  author       = {Eastwood, Michael Darius},
  title        = {The ARC Principle and Eden Protocol: Recursive Intelligence
                  Scaling and Embedded AI Alignment},
  year         = {2026},
  publisher    = {OSF},
  doi          = {10.17605/OSF.IO/6C5XB},
  url          = {https://doi.org/10.17605/OSF.IO/6C5XB}
}
```

### Independent corroboration (sequential > parallel)

The programme's sequential-recursion result (`alpha_sequential > 1 > alpha_parallel`) is independently corroborated by:

> Sharma, A. & Chopra, P. (2025). *The Sequential Edge: Inverse-Entropy Voting Beats Parallel Self-Consistency at Matched Compute.* arXiv:2511.02309 (4 November 2025). https://arxiv.org/abs/2511.02309

Their work reports, on independent systems and at matched compute, that sequential reasoning outperforms parallel self-consistency. **No priority is claimed for that result here** — it is cited as corroborating prior/parallel art, and the credit for that specific finding belongs to its authors.

### Acknowledged prior art (the geometric exponent)

The geometric scaling exponent `alpha = d/(d+1)` is **acknowledged prior art**. It follows the dimensional-scaling tradition of West, Brown & Enquist (allometric quarter-power scaling) and related derivations across physics and biology; this programme does not claim to have originated `d/(d+1)`. The contribution claimed here is the **Cauchy unification** — deriving the family of admissible scaling laws from the Cauchy functional equations and showing that a single framework accounts for the observed cross-domain exponents (see Paper VII and the Foundational document).

## Headline Results (March 2026)

- **Paper VII** (Cauchy Unification): 19/25 empirical domains preferred the Cauchy-predicted scaling family under strict AICc model selection (p = 1.56 x 10^-5)
- **Paper V** (Stewardship Gene): Stakeholder care improvement across five analysable Eden intervention runs (Fisher p = 6.3 x 10^-21)
- **Paper VIII** (Load-Bearing Proof): Entangled safety is load-bearing at the weight level. DGM experiment shows Eden matches Babylon capability at 0.667. Removal of safety component collapses all capability (p = 0.04). Gated simulation confirms Babylon reward-hacking fingerprint. Weight-level structural entanglement inconclusive at current training scale.
- **Paper IX** (Synthesis and Roadmap): Synthesises the full programme into a unified narrative with a four-tier replication roadmap

## Overview

This repository contains the complete 18-document research programme for the ARC Principle framework and the Eden Protocol alignment architecture, first proposed in *Infinite Architects: Intelligence, Recursion, and the Creation of Everything* (Eastwood, 2026).

**Author:** Michael Darius Eastwood
**London, United Kingdom**
**Web:** [michaeldariuseastwood.com](https://www.michaeldariuseastwood.com) | **Research hub:** [michaeldariuseastwood.com/research](https://www.michaeldariuseastwood.com/research/)

## The ARC Principle

```
U = I x R^alpha     where alpha = 1/(1 - beta)
```

- **U** = Effective capability
- **I** = Base potential (structured asymmetry)
- **R** = Recursive depth
- **beta** = Self-referential coupling
- **alpha** = Scaling exponent (derived, not fitted)

**Core prediction:** alpha_sequential > 1 > alpha_parallel. Validated to R^2 = 1.00000000 (machine precision).

## The Document Suite (18 documents)

All authoritative documents live in per-paper folders under `papers/`:

### 12 Research Papers

| # | Paper | Key Finding |
|---|-------|-------------|
| I | ARC Principle | Original published paper. U = I x R^alpha. |
| II | Experimental Validation | Sequential recursion outperforms parallel by 25pp with less compute. |
| III | Alignment Scaling Problem | External constraints cannot compound; embedded values can. 13 falsification criteria. |
| IV.a | Baked-In vs Computed Alignment | Architecture-dependent alignment response classes. |
| IV.b | Alignment Saturation at Low Depth | Alignment saturates at low recursion depth. |
| IV.c | ARC-Align Benchmark | Reproducible blind benchmark specification. |
| IV.d | Effect of Blinding | Unblinded scoring can reverse alignment measurements. |
| V | Stewardship Gene | Eden intervention pilot. Stakeholder care p = 6.3 x 10^-21. |
| VI | Honey Architecture | Entangled loss functions for self-modifying AI safety. |
| VII | Cauchy Unification | 50-domain tiered validation. 19/25 empirical strict (p = 1.56 x 10^-5). |
| VIII | Load-Bearing Proof | Entangled safety validated at weight level. DGM, removal test, gated simulation. |
| IX | Synthesis and Roadmap | Full programme synthesis. Four-tier replication roadmap. |

### 6 Supporting Documents

| Document | Purpose |
|----------|---------|
| Foundational | Axiomatic derivation from Cauchy functional equations. d/(d+1) prediction. |
| On the Origin of Scaling Laws | Cross-domain d/(d+1) evidence catalogue. 8 independent derivations. |
| Eden Engineering | Protocol architecture specification (v6). |
| Eden Vision | Public coordination document. Philosophical foundations. |
| Executive Summary | Programme overview for grant reviewers and technical evaluators. |
| Master Table of Contents | Suite navigation with per-paper summaries. |

## Repository Structure

```
arc-principle-validation/
+-- README.md
+-- LICENCE
+-- CANONICAL-VERSIONS.md
+-- VERSION-CONTROL-STANDARDS.md
+-- OSF-UPLOAD-CHECKLIST-2026-03-18.md
+-- papers/                              # Per-paper folders (HTML, PDF, experiments, results, figures)
|   +-- Paper-I-ARC-Principle/
|   +-- Paper-II-Experimental-Validation/
|   +-- Paper-III-Alignment-Scaling-Problem/
|   +-- Paper-IV-a-Baked-In-vs-Computed-Alignment/
|   +-- Paper-IV-b-Alignment-Saturation-at-Low-Depth/
|   +-- Paper-IV-c-ARC-Align-Benchmark/
|   +-- Paper-IV-d-The-Effect-of-Blinding-on-AI-Alignment-Evaluation/
|   +-- Paper-V-Stewardship-Gene/
|   +-- Paper-VI-Honey-Architecture/
|   +-- Paper-VII-Cauchy-Unification/
|   +-- Paper-VIII-The-Load-Bearing-Proof/
|   +-- Paper-IX-Synthesis-and-Roadmap/
|   +-- Foundational/
|   +-- On-the-Origin-of-Scaling-Laws/
|   +-- Eden-Engineering/
|   +-- Eden-Vision/
|   +-- Executive-Summary/
|   +-- Master-Table-of-Contents/
+-- scripts/                             # Build and utility scripts (PDF export)
```

Each paper folder contains its own `README.md`, along with `experiments/`, `results/`, and (where applicable) `figures/` subdirectories. See the individual paper README for experiment details and cross-references.

## Running Experiments

Experiments live inside each paper's folder under `papers/<Paper>/experiments/`. Most require Python 3.10+ and an API key for the relevant model provider.
Active experiment paths should prefer the shared Eden gateway when available:

- `EDEN_GATEWAY_URL`
- `EDEN_GATEWAY_API_KEY`

Legacy direct-provider keys are only for older standalone runs that have not yet been moved behind Eden.

```bash
# Paper VII: 50-domain Cauchy validation
cd papers/Paper-VII-Cauchy-Unification/experiments/
python scripts/run_50_domain_validation.py

# Paper VIII: Gated self-modification simulation
cd papers/Paper-VIII-The-Load-Bearing-Proof/experiments/gated-self-mod-simulation/
python run_train.py
```

Each experiment folder contains its own README with specific instructions and dependencies.

## Falsification

Thirteen explicit falsification criteria are specified across the ARC framework and Eden Protocol, each independently sufficient to refute the relevant claims. See Paper III Section 4 and Eden Engineering Section 11.

## Citation

```bibtex
@article{eastwood2026arc,
  title={The ARC Principle: The Alignment Scaling Problem},
  author={Eastwood, Michael Darius},
  year={2026},
  month={February},
  doi={10.17605/OSF.IO/6C5XB}
}
```

## OSF

| Deposit | DOI |
|---------|-----|
| Complete Suite (Parent) | [10.17605/OSF.IO/6C5XB](https://doi.org/10.17605/OSF.IO/6C5XB) |

## Related

- **Book:** Eastwood, M.D. (2026). *Infinite Architects: Intelligence, Recursion, and the Creation of Everything*. ISBN 978-1806056200.
- **Website:** [michaeldariuseastwood.com/research](https://www.michaeldariuseastwood.com/research/)

## Licence

MIT Licence. See [LICENCE](LICENCE) for details.

## Contributing

All contributions welcome, **including falsifications**.

If you find evidence that contradicts the ARC Principle, please open an issue or submit a pull request.

---

**Priority Established:** 8 December 2024 (cryptographically timestamped manuscript submission)

**Last Updated:** 19 March 2026

**Copyright 2026 Michael Darius Eastwood**
