# Experiments

Every experiment in the programme, with the code that produced it and the output that
code recorded. This file is the map: what each folder is, which document it supports,
and what its evidence is worth.

## How a folder is named

`<what-it-measures>__<the-document-it-supports>`

The double underscore separates the two halves. A folder supporting more than one
document names them together, as `Papers-IV-a-b-c-d` does. Two folders carry no
document suffix because they support no single one: `shared/` and
`arc-scaling-lineage/`.

## What a folder contains

| Path | Holds |
|------|-------|
| `scripts/` | the code that was run |
| `results/` | what that code recorded, dated in the filename where a run is dated |
| `figures/` | plots generated from `results/`, where the document uses them |
| `REPRODUCE.md` | how to run it again, and what it needs |

A recorded result is never overwritten. A fresh collection writes a new dated file
beside the existing one.

## The experiments

| Folder | Supports | What it measures | Evidence tier |
|--------|----------|------------------|---------------|
| `paper-i-foundational__Paper-I` | Paper I | the original ARC Principle validation toolkit | supporting |
| `paper-ii-compute__Paper-II` | Paper II | sequential against parallel scaling, six models | pilot |
| `alignment-scaling__Papers-IV-a-b-c-d` | Papers IV.a to IV.d | alignment scaling across depth, versions v1 to v5 | pilot, blind-scored at v5 |
| `eden-intervention__Paper-V` | Paper V | the Eden intervention against a control, five analysable runs | pilot, single-scorer, not blinded |
| `honey-architecture__Paper-VI` | Paper VI | the honey architecture in simulation and against live models | mechanistic in simulation, exploratory on live models |
| `cauchy-unification__Paper-VII` | Paper VII | the cross-domain functional-equation classification | **under correction**, see below |
| `domain-validation__Foundational-and-Origin` | Foundational, On the Origin of Scaling Laws | the dimensional relation across physics and other domains | supporting, mathematical |
| `blind-prediction-test__Paper-III-and-Foundational` | Paper III, Foundational | a prediction recorded before the outcome was seen, with its forensic analysis | supporting |
| `analysis-tools__Cross-Programme` | Papers II, IV.a to IV.d, V | shared analysis code, not a result | tooling |
| `shared` | all experiments | the gateway adapter the experiment scripts share | tooling |
| `arc-scaling-lineage` | none directly | the four early standalone scaling scripts, kept as a record | superseded |

## Paper VII is under correction

`cauchy-unification__Paper-VII` records the runs as they were made. The classification
they were reported under has since been corrected: the functional-equation grid has four
cells rather than three, and the primary statistic is a permutation test conditioned on
both marginals rather than the binomial reported at the time. The recorded outputs stand;
the reading placed on them does not. Read the corrections notice before citing anything
from this folder.

## What the evidence tiers mean

| Tier | What it claims |
|------|----------------|
| pilot | real data with stated methodological limits, most often single-scorer or unblinded |
| blind-scored | the scorer could not see the condition, and blinding integrity was itself measured |
| mechanistic | a mechanism demonstrated in a toy system, with no claim about any deployed system |
| supporting | mathematical or computational validation of a step, not an empirical result |
| exploratory | pattern-finding, never confirmatory |
| tooling | code used by other experiments, producing no result of its own |
| superseded | kept for the record, replaced by later work, not to be cited as current |

No tier in this table means validated, replicated or independently confirmed. Nothing in
this repository has been independently replicated, and no result here has been through
peer review.

## Two of these paths are cited from outside

`cauchy-unification__Paper-VII/data/canonical_50_domain_manifest.json` and
`domain-validation__Foundational-and-Origin/phase0_evidence_pack/canonical_v5_alignment_table.md`
are linked from outside this repository. They do not move.
