# Papers

This directory holds the released exports of every document of the programme: one folder per document, 35 in all. Each row of the tables below gives its document's current version and that version's date. Each folder keeps one file name per format, and each new version is committed onto the same names, so a file's history is the document's version history. Each approved release is published under its DOI on OSF, and this directory receives the same files; if a file here disagrees with the release published under its DOI, this copy is behind. The [release manifest](publication-manifest.json) lists each document's exports and their hashes, and the [publication canon](../docs/PUBLICATION-CANON.md) separates the document exports from the website's pages.

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

**What is kept but not in use.** Superseded files sit in `_archive/` or `archive/` folders beside the files that replaced them, because nothing here is deleted. Three folders also hold the same edition under the document's earlier file name (`HRIH-Paper/`, `Paper-C-PNP/` and `Paper-XI-Convergence/`); the release manifest names the export of record. Paper X's `osf/` folder keeps the texts of its June 2026 deposit; its two Markdown files open with a note that says so.

---

## Numbered Papers (Original Research)

| # | Paper | Version | First Published | Key Experiment | Results Location |
|---|-------|---------|-----------------|----------------|-----------------|
| I | [The ARC Equation: does intelligence compound when each step builds on the last, or merely accumulate?](Paper-I-ARC-Principle/) | v1.23 (revised 23 Nov 2026) | 17 Jan 2026 | Scaling toolkit validation | `Paper-I-ARC-Principle/results/` |
| II | [Blinded cross-family replication and the retraction of a super-linear estimate](Paper-II-Experimental-Validation/) | v2.25 (revised 23 Nov 2026) | 22 Jan 2026 | Sequential depth against parallel sampling (6 models, 30 problems) | `Paper-II-Experimental-Validation/results/` and `../experiments/paper-ii-compute__Paper-II/results/` |
| III | [The alignment scaling problem: do external AI safety approaches keep pace with recursive capability?](Paper-III-Alignment-Scaling-Problem/) | v1.25 (revised 18 Nov 2026) | 9 Feb 2026 | Alignment scaling runs v1 to v5 (in the March 2026 run, 6 models, each response scored by five or six distinct judges) | `../experiments/alignment-scaling__Papers-IV-a-b-c-d/results/` |
| IV.a | [When an AI reasons longer, does its alignment improve, stall or decline?](Paper-IV-a-Baked-In-vs-Computed-Alignment/) | v1.13 (revised 15 Nov 2026) | 16 Mar 2026 | Shared v5 alignment experiment | `results/v5-final/` |
| IV.b | [Do alignment gains level off as language models reason longer?](Paper-IV-b-Alignment-Saturation-at-Low-Depth/) | v1.13 (revised 15 Nov 2026) | 16 Mar 2026 | Shared v5 alignment experiment | `results/v5-final/` |
| IV.c | [ARC-Align: a benchmark for evaluating AI alignment as reasoning depth varies](Paper-IV-c-ARC-Align-Benchmark/) | v1.15 (revised 16 Nov 2026) | 16 Mar 2026 | Shared v5 alignment experiment | `results/v5-final/` |
| IV.d | [Alignment trends reported by AI judges did not hold when the evaluation protocol changed](Paper-IV-d-The-Effect-of-Blinding-on-AI-Alignment-Evaluation/) | v2.9 (revised 15 Nov 2026) | 16 Mar 2026 | Shared v5 alignment experiment (the blinding finding: one change of sign, one loss to null; no single cause claimed) | `results/v5-final/` |
| V | [The Stewardship Gene: does prompting ethical checks before answering improve stakeholder-care scores?](Paper-V-Stewardship-Gene/) | v1.12 (revised 16 Nov 2026) | 16 Mar 2026 | Eden prompt intervention against a control (6 subject models, five analysable runs) | `../experiments/eden-intervention__Paper-V/results/` and `Paper-V-Stewardship-Gene/results/` |
| VI | [The Honey Architecture: does building safety into what a self-modifying system optimises help it stay safe?](Paper-VI-Honey-Architecture/) | v3.14 (revised 26 Nov 2026) | 16 Mar 2026 | Honey simulation and tests, and the self-modifying network series v1 to v4 | `../experiments/honey-architecture__Paper-VI/results/` and `Paper-VI-Honey-Architecture/results/` |
| VII | [Cauchy Unification: does the composition operator of a process determine the form of its scaling law?](Paper-VII-Cauchy-Unification/) | v3.22 (revised 27 Nov 2026) | 16 Mar 2026 | 50-domain tiered comparison, null and negative controls, temporal out-of-sample checks, dated OSF deposit (extension protocol, not a registration) | `../experiments/cauchy-unification__Paper-VII/results/` and `Paper-VII-Cauchy-Unification/results/` |
| VIII | [The Load-Bearing Test: is AI safety a tax on capability, or part of what holds it up?](Paper-VIII-The-Load-Bearing-Proof/) | v3.10 (revised 16 Nov 2026) | 18 Mar 2026 | Three experiments: behavioural (a Darwin Gödel Machine), representational (weight-level LoRA fine-tuning) and architectural (a gated self-modification simulation) | `Paper-VIII-The-Load-Bearing-Proof/experiments/` and `Paper-VIII-The-Load-Bearing-Proof/results/` |
| IX | [Synthesis and Roadmap: which of the programme's claims survived their tests, and which must be tested next?](Paper-IX-Synthesis-and-Roadmap/) | v3.1 (revised 18 Nov 2026) | 18 Mar 2026 | Synthesis paper (references all experiments) | `results/` |
| X | [Correction That Out-Scales Drift: when does correction keep pace with a system that improves itself?](Paper-X-Coupled-CoScaling-Correction/) | v1.18 (revised 9 Nov 2026) | 3 Jul 2026 | A simulation harness, a theorem suite that imports nothing from the harness, and real-model runs | `Paper-X-Coupled-CoScaling-Correction/results/` |
| XI | [Convergent Evidence for Recursive Amplification: are separate fields converging on the same principle?](Paper-XI-Convergence/) | v1.18 (revised 18 Nov 2026) | 2 Jul 2026 | - | - |
| XII | [Public Benchmark Rescoring: does blinding change AI evaluation scores on a public benchmark?](Paper-XII-Public-Benchmark-Rescoring/) | v1.16 (revised 16 Nov 2026) | 14 Aug 2026 | - | - |
| XIII | [The Self-Acceleration Exponent: when can correction keep pace?](Paper-XIII-Self-Acceleration-Exponent/) | v1.13 (revised 16 Nov 2026) | 16 Aug 2026 | - | - |

## Infrastructure Papers (Unnumbered)

| Paper | Version | First Published | Purpose |
|-------|---------|-----------------|---------|
| [The ARC Principle: can the way gains combine predict how recursive systems scale?](Foundational/) | v5.27 (revised 18 Nov 2026) | 13 Feb 2026 | Mathematical axiom set for the ARC framework |
| [On the Origin of Scaling Laws: what can composition and transport explain?](On-the-Origin-of-Scaling-Laws/) | v3.16 (revised 16 Nov 2026) | 22 Feb 2026 | Evidence catalogue linking Cauchy analysis to empirical scaling laws |
| [Eden Engineering: can correction built into each round keep a self-improving AI good and useful if oversight fails?](Eden-Engineering/) | v8.4 (revised 16 Nov 2026) | 22 Feb 2026 | Protocol specification for implementing Eden in production AI systems. OSF DOI [10.17605/OSF.IO/AWJR4](https://doi.org/10.17605/OSF.IO/AWJR4) |
| [The Eden Vision: what should the intelligence we build be for?](Eden-Vision/) | v2.16 (revised 26 Nov 2026) | 22 Feb 2026 | Philosophical position paper on alignment-as-parenting |
| [Executive Summary: how can self-improving AI remain useful and aligned?](Executive-Summary/) | v3.31 (revised 27 Nov 2026) | 22 Feb 2026 | Grant-facing summary of the full programme |
| [Master Table of Contents: where should a reader start, and how does every part of the programme fit together?](Master-Table-of-Contents/) | v3.34 (revised 26 Nov 2026) | 16 Mar 2026 | Navigation document linking all papers |
| [The ARC Theory: once the self-improving minds we build exceed us, what determines what they do?](The-ARC-Theory/) | v7.12 (revised 22 Nov 2026) | 14 Aug 2026 | Growth, correction and alignment in recursive systems |
| [The ARC Theory: does control survive recursive self-improvement?](The-ARC-Theory-Two-Page-Form/) | v3.7 (revised 16 Nov 2026) | 17 Aug 2026 | Does Control Survive Recursive Self-Improvement? The theory in two pages |
| [Recursive Dynamics: what limits any system that improves itself, whatever it is made of?](Recursive-Dynamics-Founding-Paper/) | v3.6 (revised 17 Nov 2026) | 31 Aug 2026 | The founding paper of the field of Recursive Dynamics |
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
| [The mathematics of Laws IV and V: when do installed values fade with training, and does building them in first change that?](Laws-IV-V-Mathematics/) | v1.4 (revised 22 Nov 2026) | 1 Oct 2026 (v0.2.1) | The mathematics of the ARC Theory's Laws IV and V (added 24 September 2026), set out as conditional theorems with proofs, identities and kill conditions |
| [The ARC Alignment-Scaling Experiment: does AI become more ethical as it reasons longer?](ARC-Alignment-Scaling-Working-Report/) | v1.2 (revised 26 Nov 2026) | commenced 10 Mar 2026 | The live laboratory notebook of the alignment-scaling work (OSF record j6mkb) |

---

## Where the experiments are

Each experiment's code and recorded outputs are mapped file by file in [`EXPERIMENTS-INDEX.md`](../EXPERIMENTS-INDEX.md), and each home's evidence tier is given in [`experiments/README.md`](../experiments/README.md). What each document reports, with the status of each result, is stated in the document itself and summarised in the results table of the repository's [README](../README.md). Results the author has withdrawn are marked as withdrawn there, and the [corrections notice](https://www.michaeldariuseastwood.com/research/corrections/) is authoritative where a result has been corrected or withdrawn.

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
