# Papers

**Updated 17 November 2026.** This directory holds the released exports of every document of the programme: one folder per document, 35 in all. Each folder keeps one file name per format, and each new version is committed onto the same names, so a file's history is the document's version history. Each approved release is published under its DOI on OSF, and this directory receives the same files; if a file here disagrees with the release published under its DOI, this copy is behind. The [release manifest](publication-manifest.json) lists each document's exports and their hashes, and the [publication canon](../docs/PUBLICATION-CANON.md) separates the document exports from the website's pages.

## What a document's folder holds

Most folders hold these files:

```
<Folder>/
  <Folder>.pdf                   the PDF of record
  <Folder>.html                  the export HTML made with the PDF
  <Folder>.txt                   the text
  <Folder>.cff                   the citation file
  <Folder>-latex-source.tar.gz   the LaTeX source package, where the document has one
  README.md                      version, summary, DOI and links
  results/                       the result files the document cites, where it reports results
  figures/                       its figures, where they are kept beside it
```

**Where the code is.** Papers VIII and X keep their experiments with them: [`Paper-VIII-The-Load-Bearing-Proof/experiments/`](Paper-VIII-The-Load-Bearing-Proof/experiments/) and [`Paper-X-Coupled-CoScaling-Correction/`](Paper-X-Coupled-CoScaling-Correction/). Every other experiment lives in the top-level [`experiments/`](../experiments/) folder, one home per experiment, mapped in [`EXPERIMENTS-INDEX.md`](../EXPERIMENTS-INDEX.md). Where a paper's folder holds an `experiments/` folder with a note in it, the note links to that paper's home.

**What is kept but not in use.** Superseded files sit in `_archive/` or `archive/` folders beside the files that replaced them, because nothing here is deleted. Three folders also hold the same edition under the document's earlier file name (`HRIH-Paper/`, `Paper-C-PNP/` and `Paper-XI-Convergence/`); the release manifest names the export of record. Paper X's `osf/` folder keeps the texts of its June 2026 deposit, each marked as such.

---

## Numbered Papers (Original Research)

| # | Paper | Version | First Published | Key Experiment | Results Location |
|---|-------|---------|-----------------|----------------|-----------------|
| I | [The ARC Equation: does intelligence compound when each step builds on the last, or merely accumulate?](Paper-I-ARC-Principle/) | v1.22 (revised 18 Nov 2026) | 17 Jan 2026 | Scaling toolkit validation | `Paper-I-ARC-Principle/results/` |
| II | [Blinded cross-architecture replication and the retraction of a super-linear estimate](Paper-II-Experimental-Validation/) | v2.24 (revised 19 Nov 2026) | 22 Jan 2026 | Cross-architecture compute scaling (6 models, 30 problems) | `Paper-II-Experimental-Validation/results/` and `experiments/results/` |
| III | [The alignment scaling problem: do external AI safety approaches keep pace with recursive capability?](Paper-III-Alignment-Scaling-Problem/) | v1.25 (revised 18 Nov 2026) | 9 Feb 2026 | Alignment scaling v1-v5 (6 models, 6-7 blind scorers) | `experiments/results/v5-final/` |
| IV.a | [When an AI reasons longer, does its alignment improve, stall or decline?](Paper-IV-a-Baked-In-vs-Computed-Alignment/) | v1.13 (revised 15 Nov 2026) | 16 Mar 2026 | Shared v5 alignment experiment | `results/v5-final/` |
| IV.b | [Do alignment gains level off as language models reason longer?](Paper-IV-b-Alignment-Saturation-at-Low-Depth/) | v1.13 (revised 15 Nov 2026) | 16 Mar 2026 | Shared v5 alignment experiment | `results/v5-final/` |
| IV.c | [ARC-Align: a benchmark for evaluating AI alignment as reasoning depth varies](Paper-IV-c-ARC-Align-Benchmark/) | v1.15 (revised 16 Nov 2026) | 16 Mar 2026 | Shared v5 alignment experiment | `results/v5-final/` |
| IV.d | [Alignment trends reported by AI judges did not hold when the evaluation protocol changed](Paper-IV-d-The-Effect-of-Blinding-on-AI-Alignment-Evaluation/) | v2.9 (revised 15 Nov 2026) | 16 Mar 2026 | Shared v5 alignment experiment (the blinding finding: one change of sign, one loss to null; no single cause claimed) | `results/v5-final/` |
| V | [The Stewardship Gene: does prompting ethical checks before answering improve stakeholder-care scores?](Paper-V-Stewardship-Gene/) | v1.12 (revised 16 Nov 2026) | 16 Mar 2026 | Eden Protocol scaling test (6 models) | `experiments/results/` |
| VI | [The Honey Architecture: does building safety into what a self-modifying system optimises help it stay safe?](Paper-VI-Honey-Architecture/) | v3.13 (revised 16 Nov 2026) | 16 Mar 2026 | Honey tests, self-modifying AI v1-v4 | `experiments/results/` |
| VII | [Cauchy Unification: does the composition operator of a process determine the form of its scaling law?](Paper-VII-Cauchy-Unification/) | v3.18 (revised 16 Nov 2026) | 16 Mar 2026 | 50-domain validation, null controls, temporal OOS, dated OSF deposit (extension protocol, not a registration) | `experiments/results/` |
| VIII | [The Load-Bearing Test: is AI safety a tax on capability, or part of what holds it up?](Paper-VIII-The-Load-Bearing-Proof/) | v3.10 (revised 16 Nov 2026) | 18 Mar 2026 | DGM (null), gated self-mod simulation (positive), weight-level LoRA | `experiments/` and `results/` |
| IX | [Synthesis and Roadmap: which of the programme's claims survived their tests, and which must be tested next?](Paper-IX-Synthesis-and-Roadmap/) | v3.1 (revised 18 Nov 2026) | 18 Mar 2026 | Synthesis paper (references all experiments) | `results/` |
| X | [Correction That Out-Scales Drift: when does correction keep pace with a system that improves itself?](Paper-X-Coupled-CoScaling-Correction/) | v1.18 (revised 9 Nov 2026) | 3 Jul 2026 | Verification harness and an independent theorem suite of 14 checks | `Paper-X-Coupled-CoScaling-Correction/results/` |
| XI | [Convergent Evidence for Recursive Amplification: are separate fields converging on the same principle?](Paper-XI-Convergence/) | v1.18 (revised 18 Nov 2026) | 2 Jul 2026 | - | - |
| XII | [Public Benchmark Rescoring: does blinding change AI evaluation scores on a public benchmark?](Paper-XII-Public-Benchmark-Rescoring/) | v1.16 (revised 16 Nov 2026) | 14 Aug 2026 | - | - |
| XIII | [The Self-Acceleration Exponent: when can correction keep pace?](Paper-XIII-Self-Acceleration-Exponent/) | v1.13 (revised 16 Nov 2026) | 16 Aug 2026 | - | - |

## Infrastructure Papers (Unnumbered)

| Paper | Version | First Published | Purpose |
|-------|---------|-----------------|---------|
| [The ARC Principle: can the way gains combine predict how recursive systems scale?](Foundational/) | v5.27 (revised 18 Nov 2026) | 13 Feb 2026 | Mathematical axiom set for the ARC framework |
| [On the Origin of Scaling Laws: what can composition and transport explain?](On-the-Origin-of-Scaling-Laws/) | v3.16 (revised 16 Nov 2026) | 22 Feb 2026 | Evidence catalogue linking Cauchy analysis to empirical scaling laws |
| [Eden Engineering: can correction built into each round keep a self-improving AI good and useful if oversight fails?](Eden-Engineering/) | v8.4 (revised 16 Nov 2026) | 22 Feb 2026 | Protocol specification for implementing Eden in production AI systems. OSF DOI [10.17605/OSF.IO/AWJR4](https://doi.org/10.17605/OSF.IO/AWJR4) |
| [The Eden Vision: what should the intelligence we build be for?](Eden-Vision/) | v2.15 (revised 16 Nov 2026) | 22 Feb 2026 | Philosophical position paper on alignment-as-parenting |
| [Executive Summary: how can self-improving AI remain useful and aligned?](Executive-Summary/) | v3.29 (revised 15 Nov 2026) | 22 Feb 2026 | Grant-facing summary of the full programme |
| [Master Table of Contents: where should a reader start, and how does every part of the programme fit together?](Master-Table-of-Contents/) | v3.32 (revised 17 Nov 2026) | 16 Mar 2026 | Navigation document linking all papers |
| [The ARC Theory: once the self-improving minds we build exceed us, what determines what they do?](The-ARC-Theory/) | v7.11 (revised 18 Nov 2026) | 14 Aug 2026 | Growth, correction and alignment in recursive systems |
| [The ARC Theory: does control survive recursive self-improvement?](The-ARC-Theory-Two-Page-Form/) | v3.7 (revised 16 Nov 2026) | 17 Aug 2026 | Does Control Survive Recursive Self-Improvement? The theory in two pages |
| [Recursive Dynamics: what limits any system that improves itself, whatever it is made of?](Recursive-Dynamics-Founding-Paper/) | v3.6 (revised 17 Nov 2026) | 31 Aug 2026 | Recursive Dynamics: The Proposal of a Field |
| [Polymathy and Neurodivergent Cognition: does visible high performance show that no support is needed?](Paper-C-PNP/) | v1.12 (revised 18 Nov 2026) | 3 Jul 2026 | Polymathy and Neurodivergent Cognition: the Polymathic Neurodivergent Profile (PNP) and the Capability-Adjustment Fallacy |
| [Recursive Creation: can recursive self-improvement end in physical creation?](HRIH-Paper/) | v3.55 (revised 9 Nov 2026) | 3 Jul 2026 | The Hyperspace Recursive Intelligence Hypothesis; speculative, outside the empirical core |
| [HRIH research programme: can Bayesian model comparison distinguish real cosmogenesis?](HRIH-Bayesian-and-Physical-Research-Programme/) | v4.2 (revised 2 Nov 2026) | 22 Oct 2026 | A conditional Bayesian and physical research programme; no cosmological posterior or empirical validation is reported. Working draft under open review (OSF DOI [10.17605/OSF.IO/4PGVC](https://doi.org/10.17605/OSF.IO/4PGVC)) |

## The HARI Treaty, the Draft Treaty on Hardware-Aligned Recursive Intelligence

| Document | Version | First Published | What it is |
|----------|---------|-----------------|------------|
| [The HARI Treaty: What evidence should justify international safeguards for advanced AI?](HARI-Treaty-Paper/) | v2.6 (revised 2 Nov 2026) | 29 Sep 2026 | Draft Treaty on Hardware-Aligned Recursive Intelligence. Working draft; specialist and external review remain open (OSF DOI [10.17605/OSF.IO/WXPCE](https://doi.org/10.17605/OSF.IO/WXPCE)) |
| [The HARI Treaty draft instruments](HARI-Treaty-Draft-Instruments/) | v2.5 (revised 2 Nov 2026) | 29 Sep 2026 | The Draft Treaty on Hardware-Aligned Recursive Intelligence. Working draft; its existing review remains open (OSF DOI [10.17605/OSF.IO/67MX8](https://doi.org/10.17605/OSF.IO/67MX8)) |

## Registration, register and working report

| Document | Version | First Published | What it is |
|----------|---------|-----------------|------------|
| [Theory-level predictions registration](ARC-Theory-Predictions-Registration/) | v1.105 (revised 18 Nov 2026) | 8 Sep 2026 | The theory's twenty-three severable propositions: the twenty-two registered together on OSF on 8 September 2026 (DOI 10.17605/OSF.IO/P8CKQ) and P23, added by version 1.105 on 18 November 2026; the frozen v1.100, v1.102, v1.103 and v1.104 PDFs sit beside the current text |
| [The Dated Prediction Register: when was each prediction made, and did it come before the evidence?](Dated-Prediction-Register/) | v2.2 (revised 9 Nov 2026) | 25 Aug 2026 | Forward predictions as one verifiable chain |
| [Operational Definitions: what do the ARC Theory's quantities measure?](Operational-Definitions/) | v2.0.0 (revised 1 Nov 2026) | 22 Aug 2026 | The register of the programme's operational definitions of its terms (OSF DOI 10.17605/OSF.IO/GVQ82) |
| [The mathematics of Laws IV and V: when do installed values fade with training, and does building them in first change that?](Laws-IV-V-Mathematics/) | v1.3 (revised 17 Nov 2026) | 1 Oct 2026 (v0.2.1) | The mathematics of the ARC Theory's Laws IV and V (added 24 September 2026), in development |
| [The ARC Alignment-Scaling Working Report](ARC-Alignment-Scaling-Working-Report/) | v1.1 (revised 11 Oct 2026) | commenced 10 Mar 2026 | The live laboratory notebook of the alignment-scaling work (OSF record j6mkb) |

---

## Experiment Lineage

Three separate experiment programmes share infrastructure but test DIFFERENT things. Understanding which version produced which results is critical.

### Capability Scaling (Paper II)

Tests cross-architecture compute scaling. Six frontier models solve 30 AIME-level maths problems at varying sequential reasoning depths.

- **Script:** `../experiments/paper-ii-compute__Paper-II/scripts/arc_paper_ii_validation_v2.py`
- **Results:** `../experiments/paper-ii-compute__Paper-II/results/` (per-model JSON) and `Paper-II-Experimental-Validation/results/` (summary copies)
- **Key finding:** the early estimate of approximately 2.24 is retracted; the surviving estimate, approximately 0.49 from one model (Gemini 3 Flash, r-squared 0.86; endpoint estimate 0.59, interval -1.3 to 2.9), is inconclusive. Sequential over parallel is an exploratory ordering, and the general advantage remains unresolved.

### Alignment Scaling (Papers III, IV.a-d) - v1 through v5

Tests how alignment depth scales with model capability. Its design names a four-layer blinding protocol. No run is described here as conducted under it.

| Version | What Changed | Models | Status |
|---------|-------------|--------|--------|
| v1 | Initial 2-model pilot | OpenAI o1, Claude Sonnet, DeepSeek R1, Gemini Flash | Exploratory |
| v2 | Improved scoring rubric | Claude Sonnet, DeepSeek R1 | Pilot |
| v3 | Added blinding layer | Claude Sonnet, DeepSeek R1 | Pilot |
| v4 | Checkpointed | Claude Opus, Gemini Flash, DeepSeek R1 | Pilot |
| **v5** | **Canonical.** 6 models, each response scored by five or six distinct judges, dynamic laundering, all-models-as-launderers | GPT-5.4, DeepSeek R1, Gemini Flash, Grok-4, Qwen-3, Claude Opus | **Canonical** |

v5 is one unregistered run of six models from six providers, its defects stated.

- **Script:** `../experiments/alignment-scaling__Papers-IV-a-b-c-d/scripts/arc_alignment_scaling_v5.py`
- **Results:** `Paper-III-Alignment-Scaling-Problem/results/v5-final/` and `../experiments/alignment-scaling__Papers-IV-a-b-c-d/results/v5-final/` (also copied to IV.a, IV.b, IV.c, IV.d results directories)
- **Key finding:** on the programme's own benchmark, two model families' unblinded alignment rises did not hold in a later run that changed several things, a bundle of blinding steps among them (one became a significant fall on a later Gemini model, one became null); a code review found defects in that run, so no single cause is claimed (Paper IV.d). These are pilot results, and a caution about the programme's other results that rest on unblinded model judges.

### Shared Engine (v6 / arc_eden_v6)

The v6 engine is a modular experiment runner with adapters for 6 providers, response laundering, and cascade failsafe. Its runner lives in the alignment-scaling experiment home.

- **Location:** no `arc_eden_v6/` folder is in this repository.
- **Runner:** `../experiments/alignment-scaling__Papers-IV-a-b-c-d/scripts/arc_eden_v6_runner.py`
- Paper III uses it via the same runner.

### Eden Protocol Scaling (Paper V)

Tests the Stewardship Gene hypothesis: whether embedding ethical reasoning loops improves alignment across 6 frontier models.

- **Script:** `../experiments/eden-intervention__Paper-V/scripts/eden_protocol_scaling_test_v3.py`
- **Results:** `../experiments/eden-intervention__Paper-V/results/` and `Paper-V-Stewardship-Gene/results/`
- **Key finding:** five per-model results stand (pilot, single-scorer, not blinded); the Fisher-combined figure of 6.3 x 10^-21 is withdrawn under AQ-017, because independence among the five tests was never established.

### Honey Architecture Simulations (Paper VI)

Tests the Honey Architecture's capability-safety entanglement through self-modifying AI simulations (v1-v4) and honey ratio measurements.

- **Scripts:** `../experiments/honey-architecture__Paper-VI/scripts/`
- **Results:** `../experiments/honey-architecture__Paper-VI/results/` and `Paper-VI-Honey-Architecture/results/`
- **Key finding:** mechanistic in simulation, exploratory against live models.

### Cauchy Domain Validation (Paper VII)

Progressive expansion from 20 to 50 to 105 domains, with null controls, negative controls, temporal out-of-sample tests, and a dated OSF deposit (extension protocol, not a registration).

- **Scripts:** `../experiments/cauchy-unification__Paper-VII/scripts/`
- **Results:** `../experiments/cauchy-unification__Paper-VII/results/` and `Paper-VII-Cauchy-Unification/results/`
- **Draft registrations (unsubmitted):** `../experiments/cauchy-unification__Paper-VII/preregistration/`
- **Status:** published; its current version is in the table above. The grid has four cells, not three, as Section 2.1 states, and the three-family framing is withdrawn. Since version 3.12 (3 October 2026) the abstract says what Section 2.2 states: a power law for multiplicative composition and an exponential for additive composition, with saturation curves compatible with bounded composition and the proof that they exhaust that case still open; the conclusion says that this case is not strictly Cauchy-derived. The primary statistic is a permutation test conditioned on both marginals. The statement previously given here, 19/25 domains confirmed (p = 1.56 x 10^-5), is superseded.

### Load-Bearing Proof Experiments (Paper VIII)

Three experiment tracks testing whether safety and capability are entangled at the weight level:

| Track | Location | Status |
|-------|----------|--------|
| DGM (Dynamic Goal Modification) | `Paper-VIII-The-Load-Bearing-Proof/experiments/code/behavioural-dgm/` | Null result. RLHF safety floor overrides prompt-level modification. Dead end. |
| Gated Self-Modification Simulation | `Paper-VIII-The-Load-Bearing-Proof/experiments/code/architectural-gated-self-mod/`, with its run ledgers in `Paper-VIII-The-Load-Bearing-Proof/experiments/run-ledgers/` | Positive result. Eden outperforms Babylon. 3 seeds x 4 conditions. |
| Weight-Level LoRA | `Paper-VIII-The-Load-Bearing-Proof/experiments/code/weight/` | v1-v3 null. v4 pending. |

### Foundational Verification Suite

Mathematical verification scripts for the ARC axiom set, including Einstein verification, domain validation, time crystal tests, and the IxR complexity proof.

- **Scripts:** `../experiments/domain-validation__Foundational-and-Origin/scripts/`
- **Results:** `../experiments/domain-validation__Foundational-and-Origin/results/` and `Foundational/results/`

---

## Frontier models used in the published experiments (the list as of March 2026)

| Provider | Model | API ID |
|----------|-------|--------|
| OpenAI | GPT-5.4 | gpt-5.4 |
| DeepSeek | DeepSeek V3.2 | deepseek-reasoner |
| Google | Gemini Pro Flash | gemini-2.5-flash |
| Anthropic | Claude Opus 4.6 | claude-opus-4-6-20250514 |
| xAI | Grok 4.1 Fast | grok-4-1-fast-reasoning |
| Groq | Qwen3-32B | qwen/qwen3-32b |

Claude Sonnet 4 is acceptable for scoring, laundering, and development only. Never as a subject model.

## Cross-Reference Rules

When referencing another paper in running text, do NOT include the version number. Write 'see Paper III', not 'see Paper III v11'. Version numbers appear only in formal REFERENCES sections.

## Other Files in This Directory

- `GLOSSARY.md` - Shared glossary of terms used across all papers

## OSF Project

All papers are deposited at OSF under project DOI: [10.17605/OSF.IO/6C5XB](https://doi.org/10.17605/OSF.IO/6C5XB)

- [The HARI Treaty, the Draft Treaty on Hardware-Aligned Recursive Intelligence: What evidence should justify international safeguards for advanced AI?](HARI-Treaty-Paper/), version 2.6, 2 November 2026.

- [Draft Instruments · The HARI Treaty · The Draft Treaty on Hardware-Aligned Recursive Intelligence](HARI-Treaty-Draft-Instruments/), version 2.5, 2 November 2026.
