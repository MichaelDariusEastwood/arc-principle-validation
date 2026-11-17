# The ARC Principle and the Eden Protocol

**The code, data and recorded outputs behind a research programme on recursive intelligence
scaling and embedded AI alignment.** 35 document folders, the code and recorded outputs of every published experiment, and every result
file the papers cite.

[![OSF DOI](https://img.shields.io/badge/OSF-10.17605%2FOSF.IO%2F6C5XB-blue)](https://doi.org/10.17605/OSF.IO/6C5XB)

## What this repository claims, and what it does not

It ships the material a reader needs to check the programme's results: the code that was
run, the outputs that code recorded, and the canonical released documents under papers/.

**Nothing here has been independently replicated, and nothing here has been through peer
review.** Two results have been withdrawn by the author and are marked as withdrawn where
they appear. One paper is under correction. Those are stated in the results table below
rather than left for a reader to discover. A claim of priority for the theory's five components together was withdrawn on 12 November 2026; [`priority-claims/README.md`](priority-claims/README.md) states it.

## Repository structure

```
arc-principle-validation/
├── README.md                  this file
├── LICENCE                    which licence governs which material
├── LICENCE-CODE.md            code, experiments, data and results
├── LICENCE-PAPERS.md          papers, text, figures and PDFs
├── CITATION.cff               machine-readable citation
├── EXPERIMENTS-INDEX.md       every experiment, indexed by claim
├── papers/                    35 document folders, each with its own README
├── experiments/               nine experiment homes, shared code and superseded scripts, mapped in experiments/README.md
├── instruments/               reference code for runs not yet made, mapped in instruments/README.md
├── priority-claims/           the dated provenance record
└── docs/                      reproducibility notes, release notes, PDF checksums
```

Every folder has a `README.md` saying what is in it and what it is for. Start with
[`experiments/README.md`](experiments/README.md) if you came to check a result.

See [the publication canon](docs/PUBLICATION-CANON.md) and [release manifest](papers/publication-manifest.json) for the separation of document exports from website pages.

## Where everything is

| You want | Go to |
|---|---|
| A document at its latest version | its folder in [`papers/`](papers/), listed in [`papers/README.md`](papers/README.md); every version is on OSF under the document's DOI |
| The code and recorded outputs behind a result | [`EXPERIMENTS-INDEX.md`](EXPERIMENTS-INDEX.md), then the experiment's home in [`experiments/`](experiments/); Papers VIII and X keep theirs in their own folders under `papers/` |
| Reference code for runs not yet made | [`instruments/`](instruments/) |
| A registered study unit | its registration and results on OSF: Unit 06 ([registration](https://doi.org/10.17605/OSF.IO/RV2B5), [results](https://doi.org/10.17605/OSF.IO/4BPHG)); Unit 83 ([registration](https://doi.org/10.17605/OSF.IO/MN73W), [results](https://doi.org/10.17605/OSF.IO/98HKR), and its code in [`experiments/unit-83-counted-p21/`](experiments/unit-83-counted-p21/)); Unit 82 ([registration](https://doi.org/10.17605/OSF.IO/JUFG4)); Unit 83b ([registration](https://doi.org/10.17605/OSF.IO/4HK32)) |
| The priority record | [`priority-claims/`](priority-claims/) |
| Release notes and the PDF checksums | [`docs/`](docs/) |
| The licences | [`LICENCE`](LICENCE): the papers are CC BY 4.0; the code, data and results are all rights reserved, with the [Replication Licence 1.0](LICENCE-REPLICATION.md) |
| The OSF project | [osf.io/6c5xb](https://osf.io/6c5xb/); its `00-START-HERE.md` lists every document with its DOI and code folder, and says which of the project's own folders are in use |

**Kept, but not in use:** superseded files in `_archive/` and `archive/` folders; the four early scaling scripts in [`experiments/arc-scaling-lineage/`](experiments/arc-scaling-lineage/); in three paper folders, the same edition under the document's earlier file name; and Paper X's June 2026 deposit texts in its `osf/` folder.

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

## Current implementation entry point (7 September 2026)

The new [exact arithmetic oversight assay](instruments/oversight-assay/README.md)
provides an independent scorer for raw model responses, with fixed denominators and
explicit failure handling. It is an instrument-development task, not a new empirical
result or a validated alignment metric. The [decision-study proposal](docs/oversight-decision-study-20260907.md)
connects a measurable oversight failure, comparison and potential configuration decision.

The theory-level public document currently shows draft predictions. Code, a DOI,
a timestamp or a repository deposit does not make an experiment registered, tested,
peer reviewed or independently replicated. The current protocol and paper versions
are listed on the [research website](https://www.michaeldariuseastwood.com/research/).

## The ARC Principle

```
U = I x R^alpha     proposed relationship, conditional on the measured domain
```

| Symbol | Meaning |
|--------|---------|
| `U` | effective capability |
| `I` | baseline capability in the specified measurement model |
| `R` | recursive depth |
| `alpha` | scaling exponent to estimate and compare against rival functional forms |

**The historical prediction was `alpha_sequential > 1 > alpha_parallel`.** Paper II
reports an exploratory sequential/parallel ordering. A general compute-matched causal
advantage and a universal scaling law are not established by that observation. The early
super-linearity estimate of approximately 2.24 was retracted; the surviving v13 estimate,
approximately 0.49 from one model (0.59 by endpoint estimate, interval [-1.3, 2.9]), is
inconclusive. Super-linearity remains an open prediction and is not confirmed. The
`R^2 = 1.00000000` fit reported in an earlier version was an algebraic identity check and
never an empirical validation.

## Results, with their current status

| Document | What it reports | Status |
|----------|-----------------|--------|
| [Paper I](papers/Paper-I-ARC-Principle/) | the original statement, `U = I x R^alpha` | published |
| [Paper II](papers/Paper-II-Experimental-Validation/) | sequential exceeded parallel wherever both exponents could be estimated | exploratory ordering; the general advantage remains unresolved |
| [Paper III](papers/Paper-III-Alignment-Scaling-Problem/) | a conditional model and measurement programme, not proof that a named alignment method must fail or that embedded correction survives unrestricted self-modification; 13 falsification criteria | published |
| [Paper IV.a](papers/Paper-IV-a-Baked-In-vs-Computed-Alignment/) | When an AI reasons longer, does its alignment improve, stall or decline? | pilot |
| [Paper IV.b](papers/Paper-IV-b-Alignment-Saturation-at-Low-Depth/) | Do alignment gains level off as language models reason longer? | pilot |
| [Paper IV.c](papers/Paper-IV-c-ARC-Align-Benchmark/) | a benchmark for evaluating AI alignment as reasoning depth varies | specification, not a result |
| [Paper IV.d](papers/Paper-IV-d-The-Effect-of-Blinding-on-AI-Alignment-Evaluation/) | on the programme's own benchmark, unblinded rises did not hold in a later run that changed several things, a bundle of blinding steps among them (one reversal, one loss to null) | pilot; several components changed together and a code review found defects, so no single cause is claimed |
| [Paper V](papers/Paper-V-Stewardship-Gene/) | the Eden intervention on stakeholder care | five per-model results stand; the Fisher-combined figure of 6.3 x 10^-21 is **withdrawn** under AQ-017, because independence among the five tests was never established |
| [Paper VI](papers/Paper-VI-Honey-Architecture/) | entangled loss functions for self-modifying AI safety | mechanistic in simulation, exploratory against live models |
| [Paper VII](papers/Paper-VII-Cauchy-Unification/) | the cross-domain functional-equation classification | **published.** The grid has four cells, not three, as Section 3.1 states, and the three-family framing is withdrawn. The abstract says what Section 3.2 states: a power law for multiplicative composition and an exponential for additive composition. A saturating form follows from none of them: it needs added assumptions, a ceiling on output and dynamics of its own, since a ceiling alone selects no form. The primary statistic is a permutation test conditioned on both marginals, not the binomial reported previously |
| [Paper VIII](papers/Paper-VIII-The-Load-Bearing-Proof/) | whether entangled safety is load-bearing | one null, one inconclusive and one simulation result leave the safety-capability trade-off unproven at the scale tested, and whether embedded safety brings a measurable benefit open |
| [Paper IX](papers/Paper-IX-Synthesis-and-Roadmap/) | synthesis of the programme and a four-tier replication roadmap | synthesis |
| [Paper X](papers/Paper-X-Coupled-CoScaling-Correction/) | coupled co-scaling of capability and correction | published |
| [Paper XI](papers/Paper-XI-Convergence/) | convergence across independent traditions | published |
| [Paper XII](papers/Paper-XII-Public-Benchmark-Rescoring/) | public benchmark rescoring | published |
| [Paper XIII](papers/Paper-XIII-Self-Acceleration-Exponent/) | the self-acceleration exponent | published |
| [Paper C](papers/Paper-C-PNP/) | polymathy and neurodivergent cognition: the Polymathic Neurodivergent Profile (PNP) and the Capability-Adjustment Fallacy | published |
| [HRIH Paper](papers/HRIH-Paper/) | a cosmological hypothesis | **speculative, outside the empirical core.** No result in this repository bears on it and it is not part of the evidence base |
| [The ARC Theory](papers/The-ARC-Theory/) | the theory-level statement | published |
| [The ARC Theory, two-page form](papers/The-ARC-Theory-Two-Page-Form/) | the theory in two pages | published |
| [Recursive Dynamics, the founding paper](papers/Recursive-Dynamics-Founding-Paper/) | the science of systems that improve themselves | published |
| [The theory-level predictions registration](papers/ARC-Theory-Predictions-Registration/) | twenty-two severable propositions, registered before any admissible confirmatory outcome | registered on OSF, 8 September 2026 |
| [The Dated Prediction Register](papers/Dated-Prediction-Register/) | forward predictions as one verifiable chain | published |
| [The ARC Alignment-Scaling Working Report](papers/ARC-Alignment-Scaling-Working-Report/) | the live laboratory notebook of the alignment-scaling work | working report |
| [Foundational](papers/Foundational/) | the axiomatic derivation and the `d/(d+1)` prediction | published |
| [On the Origin of Scaling Laws](papers/On-the-Origin-of-Scaling-Laws/) | the cross-domain evidence catalogue | published |
| [Eden Engineering](papers/Eden-Engineering/) | the protocol architecture specification | specification |
| [Eden Vision](papers/Eden-Vision/) | the philosophical foundations | position paper |
| [Executive Summary](papers/Executive-Summary/) | the programme overview | overview |
| [Master Table of Contents](papers/Master-Table-of-Contents/) | suite navigation | index |
| [The HARI Treaty, the Draft Treaty on Hardware-Aligned Recursive Intelligence: What evidence should justify international safeguards for advanced AI?](papers/HARI-Treaty-Paper/) | a conditional governance proposal | working draft; specialist and external review remain open |
| [The HARI Treaty draft instruments](papers/HARI-Treaty-Draft-Instruments/) | the Draft Treaty on Hardware-Aligned Recursive Intelligence | working draft; its existing review remains open |

Where a result is under correction or withdrawn, the
[corrections notice](https://www.michaeldariuseastwood.com/research/corrections/) is
authoritative and this page is not.

## Where the master copy lives

Each paper is written and versioned privately by the author. Each approved release is
published under its DOI on OSF and on
[michaeldariuseastwood.com](https://www.michaeldariuseastwood.com/research/papers/), which carry
the same PDF, and this repository receives the same files after each release. Until October
2026 this section said the website pages were the master copies; they are published copies.

**This repository is authoritative for reproduction.** The code, data and experiment runners
here are the ones that produced the results. If a paper mirror here disagrees with the
release published under its DOI, the published release is correct and this copy is behind.
Please [open an issue](https://github.com/MichaelDariusEastwood/arc-principle-validation/issues) if
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

Experiments live in two places, and both are real. Papers VIII and X keep their experiments
in their own folders: `papers/Paper-VIII-The-Load-Bearing-Proof/experiments/` and
`papers/Paper-X-Coupled-CoScaling-Correction/`. The top-level `experiments/` tree holds nine
experiment homes: eight for papers and one for Unit 83; [`experiments/README.md`](experiments/README.md) maps each one to the documents it
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
cd papers/Paper-VIII-The-Load-Bearing-Proof/experiments/code/architectural-gated-self-mod
python src/run_train.py --config configs/default.yaml
```

## Falsification

Thirteen falsification criteria are specified across the ARC framework and the Eden
Protocol, each independently sufficient to refute the claim it attaches to. See Paper III
section 7 and Eden Engineering section 5.2.

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

The deposit record's own description is author-held and refreshed at deposit time rather than
by branches, so its document count can trail `CITATION.cff`, which carries the current one.

## Related

- **Book.** Eastwood, M.D. (2026). *Infinite Architects: Intelligence, Recursion, and the
  Creation of Everything.* ISBN 978-1806056200.
- **Research hub.** [michaeldariuseastwood.com/research](https://www.michaeldariuseastwood.com/research/)

## Licence

This repository is **dual-licensed**, and which licence applies depends on the material.

| Material | Licence |
|----------|---------|
| Papers, text, figures and PDFs | Creative Commons Attribution 4.0 International (CC BY 4.0), see [LICENCE-PAPERS.md](LICENCE-PAPERS.md) |
| Code, experiments, scripts, data and results | all rights reserved, with a replication licence: anyone may run, adapt and re-analyse them to reproduce, check or test the generalisation of the reported results, and publish what they find; see [LICENCE-REPLICATION.md](LICENCE-REPLICATION.md) and [LICENCE-CODE.md](LICENCE-CODE.md) |

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

**Earliest dated statement.** 8 December 2024, a manuscript self-emailed on that date, its
time signed by the sending server and sealed by Google's servers. A SHA-256 hash fixes the
bytes: any change to them changes the hash. It does not authenticate who made them or when.
The priority record is in [`priority-claims/`](priority-claims/).

**Copyright 2026 Michael Darius Eastwood.**

