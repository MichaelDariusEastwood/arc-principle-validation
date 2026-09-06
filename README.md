# The ARC Principle and the Eden Protocol

**The code, data and recorded outputs behind a research programme on recursive intelligence
scaling and embedded AI alignment.** Twenty-three documents, nine experiments, every result
file the papers cite.

[![OSF DOI](https://img.shields.io/badge/OSF-10.17605%2FOSF.IO%2F6C5XB-blue)](https://doi.org/10.17605/OSF.IO/6C5XB)

## What this repository claims, and what it does not

It ships the material a reader needs to check the programme's results: the code that was
run, the outputs that code recorded, and a mirror of each paper.

**Nothing here has been independently replicated, and nothing here has been through peer
review.** Two results have been withdrawn by the author and are marked as withdrawn where
they appear. One paper is under correction. Those are stated in the results table below
rather than left for a reader to discover.

## Repository structure

```
arc-principle-validation/
├── README.md                  this file
├── LICENCE                    which licence governs which material
├── LICENCE-CODE.md            code, experiments, data and results
├── LICENCE-PAPERS.md          papers, text, figures and PDFs
├── CITATION.cff               machine-readable citation
├── EXPERIMENTS-INDEX.md       every experiment, indexed by claim
├── papers/                    23 document folders, each with its own README
├── experiments/               9 experiments plus shared code, mapped in experiments/README.md
├── instruments/               reference code for runs not yet made, mapped in instruments/README.md
├── priority-claims/           the dated provenance record
└── docs/                      reproducibility notes, release notes, PDF checksums
```

Every folder has a `README.md` saying what is in it and what it is for. Start with
[`experiments/README.md`](experiments/README.md) if you came to check a result.

## Rules this repository follows

**Every paper has its own registry component.** Each document is deposited as its own
component under the parent record rather than folded into a single suite deposit, so that a
paper can be cited, versioned and corrected on its own identifier. The parent record in the
deposit table below is the suite, and it is not a substitute for the per-paper component.

**Nothing is ever deleted.** Superseded and duplicated material moves to an `_archive/` folder
local to its own home rather than leaving the repository.

**A recorded result is never overwritten.** A fresh collection writes a new dated file beside
the existing one.

**Code that has produced no result is not filed as an experiment.** `experiments/` holds runs
that produced results; `instruments/` holds instruments for runs that have not been made.

## The ARC Principle

```
U = I x R^alpha     where alpha = 1/(1 - beta)
```

| Symbol | Meaning |
|--------|---------|
| `U` | effective capability |
| `I` | base potential, structured asymmetry |
| `R` | recursive depth |
| `beta` | self-referential coupling |
| `alpha` | scaling exponent, derived rather than fitted |

**The core prediction is `alpha_sequential > 1 > alpha_parallel`, and only half of it is
supported.** The ordering `alpha_sequential > alpha_parallel` is supported by Paper II. The
early super-linearity estimate of approximately 2.24 was retracted; the robust v13
cross-architecture estimate is approximately 0.49, which is sub-linear. Super-linearity
remains an open prediction and is not confirmed. The `R^2 = 1.00000000` fit reported in an
earlier version was an algebraic identity check and never an empirical validation.

## Results, with their current status

| Document | What it reports | Status |
|----------|-----------------|--------|
| Paper I | the original statement, `U = I x R^alpha` | published |
| Paper II | sequential recursion outperforms parallel at lower compute | supported for the ordering only, see above |
| Paper III | external constraints cannot compound where embedded values can; 13 falsification criteria | published |
| Paper IV.a | architecture-dependent alignment response classes | pilot |
| Paper IV.b | alignment saturates at low recursion depth | pilot |
| Paper IV.c | a reproducible blind benchmark specification | specification, not a result |
| Paper IV.d | unblinded scoring can reverse an alignment measurement | pilot, and the programme's own strongest caution about its other results |
| Paper V | the Eden intervention on stakeholder care | five per-model results stand; the Fisher-combined figure of 6.3 x 10^-21 is **withdrawn** under AQ-017, because independence among the five tests was never established |
| Paper VI | entangled loss functions for self-modifying AI safety | mechanistic in simulation, exploratory against live models |
| Paper VII | the cross-domain functional-equation classification | **under correction as of 11 August 2026.** The grid has four cells, not three; the three-family framing and the derived-from-Cauchy treatment of bounded curves are withdrawn; the primary statistic is a permutation test conditioned on both marginals, not the binomial reported previously |
| Paper VIII | whether entangled safety is load-bearing | the removal test collapses capability (p = 0.04) and the gated simulation reproduces the reward-hacking fingerprint; **weight-level structural entanglement is inconclusive at the training scale tested** |
| Paper IX | synthesis of the programme and a four-tier replication roadmap | synthesis |
| Paper X | coupled co-scaling of capability and correction | published |
| Paper XI | convergence across independent traditions | published |
| Paper C | a complexity-theoretic treatment | published |
| HRIH Paper | a cosmological hypothesis | **speculative, outside the empirical core.** No result in this repository bears on it and it is not part of the evidence base |
| The ARC Theory | the theory-level statement | published |
| Foundational | the axiomatic derivation and the `d/(d+1)` prediction | published |
| On the Origin of Scaling Laws | the cross-domain evidence catalogue | published |
| Eden Engineering | the protocol architecture specification | specification |
| Eden Vision | the philosophical foundations | position paper |
| Executive Summary | the programme overview | overview |
| Master Table of Contents | suite navigation | index |

Where a result is under correction or withdrawn, the
[corrections notice](https://www.michaeldariuseastwood.com/research/corrections/) is
authoritative and this page is not.

## Where the master copy lives

As of 11 August 2026 the master copy of each paper is its page on
[michaeldariuseastwood.com](https://www.michaeldariuseastwood.com/research/papers/), and this
repository receives each version after it is published there. The published page carries
structured metadata, figures and mathematics that a plain-text copy drops silently, so
editing anywhere else loses information. The canon previously said the reverse; the canon
was wrong, and every revision for two months went to the site.

**This repository is authoritative for reproduction.** The code, data and experiment runners
here are the ones that produced the results. If a paper mirror here disagrees with the
published page, the published page is correct and this copy is behind. Please
[open an issue](https://github.com/MichaelDariusEastwood/arc-principle-validation/issues) if
you find that, because a lagging mirror is a defect rather than a variant.

## Two ways to check this work

**Without credentials, from a clone.** Every paper's full text, the experiment code and the
recorded run outputs are committed here. The experimental papers ship their dated per-model
result files. A reviewer with a clone and Python can read the code that produced the data,
re-run the analysis over the recorded outputs, and check any document against its committed
checksums. No API key, account or spend is needed for any of that.

**By re-collecting from live models.** Reproducing a collection from scratch needs Python
3.10 or later and access to the relevant providers, plus the associated cost. Each
experiment's `REPRODUCE.md` states its own requirements. A fresh collection writes new dated
result files beside the committed ones and never overwrites them.

## Running an experiment

Experiments live in two places, and both are real. The per-paper folders under
`papers/<Paper>/experiments/` hold the experiments belonging to a single paper. The
top-level `experiments/` tree holds the nine experiments that span papers or predate the
split; [`experiments/README.md`](experiments/README.md) maps each one to the documents it
supports.

Where an experiment supports it, prefer the shared gateway:

```bash
export EDEN_GATEWAY_URL=...
export EDEN_GATEWAY_API_KEY=...
```

Direct provider keys remain in use for older standalone runs that have not been moved behind
the gateway.

```bash
# Paper VII: 50-domain validation
cd experiments/cauchy-unification__Paper-VII
python scripts/run_50_domain_validation.py

# Paper VIII: gated self-modification simulation
cd papers/Paper-VIII-The-Load-Bearing-Proof/experiments/gated-self-mod-simulation
python run_train.py
```

## Falsification

Thirteen falsification criteria are specified across the ARC framework and the Eden
Protocol, each independently sufficient to refute the claim it attaches to. See Paper III
section 4 and Eden Engineering section 11.

## Prior and parallel work this programme does not claim

**Sequential over parallel.** The ordering result is independently corroborated by Sharma, A.
and Chopra, P. (2025), *The Sequential Edge: Inverse-Entropy Voting Beats Parallel
Self-Consistency at Matched Compute*, [arXiv:2511.02309](https://arxiv.org/abs/2511.02309),
4 November 2025. They report, on independent systems at matched compute, that sequential
reasoning outperforms parallel self-consistency. **No priority is claimed for that result
here**; the credit belongs to its authors.

**The geometric exponent.** `alpha = d/(d+1)` is **not claimed as original here**. It follows
the dimensional-scaling tradition of West, Brown and Enquist and related derivations across
physics and biology. What is claimed is the Cauchy unification: deriving the family of
admissible scaling laws from the Cauchy functional equations, and showing that one framework
accounts for the observed cross-domain exponents. That claim is the one currently under
correction, and Paper VII's row above says so.

## Citation

This repository ships a machine-readable [`CITATION.cff`](CITATION.cff). On GitHub, use the
**Cite this repository** button in the sidebar to export APA or BibTeX from it.

> Eastwood, M.D. (2026). *The ARC Principle and Eden Protocol: Recursive Intelligence Scaling
> and Embedded AI Alignment.* OSF. https://doi.org/10.17605/OSF.IO/6C5XB

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

| Deposit | DOI |
|---------|-----|
| Complete suite, parent record | [10.17605/OSF.IO/6C5XB](https://doi.org/10.17605/OSF.IO/6C5XB) |

## Related

- **Book.** Eastwood, M.D. (2026). *Infinite Architects: Intelligence, Recursion, and the
  Creation of Everything.* ISBN 978-1806056200.
- **Research hub.** [michaeldariuseastwood.com/research](https://www.michaeldariuseastwood.com/research/)

## Licence

This repository is **dual-licensed**, and which licence applies depends on the material.

| Material | Licence |
|----------|---------|
| Papers, text, figures and PDFs | Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International, see [LICENCE-PAPERS.md](LICENCE-PAPERS.md) |
| Code, experiments, scripts, data and results | proprietary, all rights reserved, see [LICENCE-CODE.md](LICENCE-CODE.md) |

[LICENCE](LICENCE) states the split. Reading, checking and re-analysing the committed
results is expressly what this repository exists for; redistribution of the code is not
granted by either licence.

## Contributing

Contributions are welcome, **falsifications most of all**. If you find evidence that
contradicts the ARC Principle, or a result here that does not reproduce, please
[open an issue](https://github.com/MichaelDariusEastwood/arc-principle-validation/issues).
A failure to reproduce is the most useful thing anyone can send.

---

**Author.** Michael Darius Eastwood, London, United Kingdom.
[michaeldariuseastwood.com](https://www.michaeldariuseastwood.com)

**Earliest dated statement.** 8 December 2024, a manuscript self-emailed on that date; the
sender copy carries the date in its header and a SHA-256 authenticates the bytes. The
priority record is in [`priority-claims/`](priority-claims/).

**Copyright 2026 Michael Darius Eastwood.**
