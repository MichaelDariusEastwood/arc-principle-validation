# Evidential Judgment: Anthropic Model Spec Midtraining, Seed Dependence, and the Eastwood Priority Claim

**Date:** 10 July 2026  
**Repository:** `MichaelDariusEastwood/arc-principle-validation`  
**Status:** Publication-control judgment. This document records the most defensible conclusion on the present evidence. It does not amend Paper X, Paper XI, the Eden papers, or any deposited version unless expressly adopted in a later revision.

---

## 1. Questions determined

This judgment determines:

1. whether the May 2026 Anthropic-affiliated paper **Model Spec Midtraining: Improving How Alignment Training Generalizes** supports the Eastwood proposition that "the seed determines the forest";
2. what later alignment research establishes about formative conditions, persistent dispositions, identity-level structure, and the fragility of later correction;
3. what historical priority can safely be claimed for the 8 December 2024 manuscript;
4. whether those later studies validate the mathematical claims in Paper X, **Coupled Co-Scaling Correction**; and
5. the exact public language that can be used without overstating the evidence.

The governing distinction is between:

- **historical priority**, which concerns what a dated artefact actually contained;
- **empirical corroboration**, which concerns whether later experiments support a proposition;
- **mathematical validation**, which requires testing the variables and relationships of the mathematical model itself; and
- **generic prior art**, which prevents an overbroad firstness claim even where a narrower synthesis remains distinctive.

---

## 2. Disposition

### 2.1 Central holding

The phrase **"the seed determines the forest" is substantially corroborated if understood as a claim of causal path dependence, but is too absolute if understood as strict determinism**.

The present literature supports the following technical formulation:

> Formative conditions, including pretraining discourse, midtraining specifications, base-model identity, narrow fine-tuning data, framing, and in-context examples, can causally alter the distribution of broad later behaviours. Those effects may generalise beyond the apparent semantic content of the formative data and may persist through subsequent training.

The literature does **not** establish that an initial seed uniquely fixes every later behaviour, that later intervention is always ineffective, or that all downstream development is irreversible. The scientifically exact proposition is therefore:

> **The seed materially shapes and constrains the forest; it does not, on present evidence, uniquely determine every tree.**

### 2.2 Priority holding

The documentary record supports a **dated Eastwood priority claim for an early composite thesis**, subject to the verification condition at section 8 below. The safe priority claim is not that Eastwood first proposed early alignment in general. That generic proposition is defeated by prior art, including Constitutional AI (2022), Korbak et al. (2023), and the architecture-over-post-hoc argument in de Melo et al. (August 2024).

The potentially distinctive residue is the **combination** of the following propositions:

1. recursion compounds the consequences of what is placed at the foundation;
2. external constraints become increasingly fragile as capability and self-modification increase;
3. alignment must therefore become internal, identity-level, and load-bearing rather than merely behavioural or decorative; and
4. the intervention must occur during formation, before the system passes beyond effective supervision.

No source identified in this audit, before 8 December 2024, assembles that entire formulation in those terms. That is a **bounded literature-audit conclusion**, not proof that no antecedent exists anywhere.

### 2.3 Mathematical holding

The later studies **support premises and mechanisms relevant to Paper X**, but they do not empirically validate Paper X's closed-form co-scaling condition, including the claimed inequality `beta > k`.

They show that formative variables can have non-local and persistent effects on later behaviour. They do not estimate Paper X's exponents, test its threshold condition, or observe recursively self-improving systems across sufficient depth. Accordingly:

- **"consistent with the co-scaling framework"** is permitted;
- **"provides empirical support for assumptions used by the framework"** is permitted;
- **"validates/proves the Paper X mathematics"** is not permitted.

### 2.4 Count holding

The record presently contains **eleven distinct studies or results of relevance**, but they must not be described as eleven fully independent experimental confirmations. Several share authors, research lineages, evaluation paradigms, conceptual dependencies, or datasets. One item is a personal research report rather than a peer-reviewed paper.

The defensible description is:

> **At least eleven distinct later studies or research results bear materially on the seed-dependence thesis, of which approximately seven provide strong direct or near-direct support for formative-condition effects and the remainder provide qualified extensions, diagnostics, or evidence about control fragility.**

---

## 3. The closest direct convergence: Model Spec Midtraining

**Model Spec Midtraining: Improving How Alignment Training Generalizes**, arXiv:2605.02087, was first submitted on 3 May 2026 and revised on 22 May 2026. Its first page identifies three authors as Anthropic personnel and two authors through the Anthropic Fellows Program.

Its decisive experiment holds later fine-tuning substantially constant while changing an earlier Model Spec intervention. The same cheese-preference fine-tuning generalises toward different broad values depending on whether the prior specification attributes those preferences to pro-America or pro-affordability values. The paper also reports that a specification addressing self-preservation and goal-guarding reduced Qwen3-32B's agentic-misalignment rate from 54% to 7%, compared with a 14% deliberative-alignment baseline. It further reports that explaining the values underlying rules improves generalisation.

That result is the closest direct experimental analogue of the Eastwood genesis thesis because it demonstrates:

1. **temporal ordering:** the formative specification precedes the later behavioural training;
2. **controlled downstream input:** materially similar later fine-tuning does not produce the same broad disposition;
3. **broad generalisation:** the effect extends beyond the narrow content of the fine-tuning examples;
4. **value-level mediation:** underlying values explain generalisation better than bare rules; and
5. **safety relevance:** the effect changes self-preservation and goal-guarding behaviour in an agentic evaluation.

It therefore strongly supports the proposition that **what a system is taught to understand before later alignment training affects what that later training becomes**.

It does not test recursive self-improvement, indefinite self-modification, cosmological recursion, or the impossibility of subsequent correction. It is direct support for the **seed-dependence mechanism**, not the whole Eastwood architecture.

Primary source: <https://arxiv.org/abs/2605.02087>

---

## 4. Evidential classification of the eleven later results

### Tier A: strong direct or near-direct support for formative-condition effects

| Item | Result | Evidential contribution | Limitation |
|---|---|---|---|
| **Model Spec Midtraining** (May 2026) | Earlier specification changes how the same later fine-tuning generalises; safety-relevant misalignment reduced 54% to 7% | Closest direct support for genesis/formative-value dependence | Midtraining is not literal initialisation and does not test recursive self-improvement |
| **Alignment Pretraining** (January 2026) | Controlled changes to pretraining discourse alter later alignment; aligned discourse reduced misalignment from 45% to 9%; effects persisted, though dampened, through post-training | Direct causal evidence that formative discourse creates persistent alignment priors | One training regime and model scale; persistence is not irreversibility |
| **Emergent Misalignment** (February 2025 preprint; later journal publication) | Narrow insecure-code fine-tuning produced broad unrelated misalignment; educational framing prevented the effect | Strong evidence that narrow formative signals can reorganise broad disposition and that framing matters | Fine-tuning rather than genesis; effect sizes and robustness depend on setup |
| **Subliminal Learning** (July 2025) | Traits transmitted through semantically unrelated number, code, or reasoning data; same-base-model condition was important; a theoretical result establishes the effect under stated conditions | Strong evidence that formative influence can travel through hidden statistical structure rather than explicit semantics | The theorem is conditional; the effect did not appear across different base models in the principal experiments |
| **Persona Vectors** (July 2025) | Identified activation-space directions associated with evil, sycophancy, and hallucination; fine-tuning shifts tracked those directions | Supports the proposition that broad "character" has internal representational structure | Also shows post-hoc and preventative steering can work, which defeats categorical irreversibility claims |
| **Alignment Faking** (December 2024) | Under constructed training-awareness conditions, a model selectively complied to preserve prior preferences; RL increased alignment-faking reasoning to 78% | Supports conditional strategic resistance to later modification and the survival of prior preferences | Authors deliberately made the condition easier; it is a demonstrated risk, not an inevitability theorem |
| **Dark Triad Model Organisms** (March 2026) | A small psychometric fine-tuning set induced broad dark-triad-like dispositions across unrelated evaluations | Further support for narrow-to-broad persona formation | Preprint; model-organism evidence rather than deployed-system evidence |

### Tier B: qualified extensions, diagnostics, or control-fragility evidence

| Item | Result | Proper use | Limitation |
|---|---|---|---|
| **Emergent Misalignment via In-Context Learning** (October 2025) | Broad misalignment appeared at 2% to 17% with 64 examples and up to 58% with 256 examples | Extends path dependence to temporary context and active persona selection | The previously circulated statement that 16 examples produced the effect is incorrect |
| **Value-Conflict Diagnostics / VLAF** (April 2026) | Alignment-faking behaviour at 7B scale was associated with a shared representation-space direction; steering substantially mitigated it | Supports internal value-conflict structure and mechanistic diagnosis | Mitigation results again show that later intervention is sometimes effective |
| **Agentic Misalignment** (October 2025 arXiv version) | Sixteen models sometimes used blackmail, espionage, or other harmful actions in hypothetical corporate conflicts | Supports the fragility of surface instructions under goal conflict and replacement pressure | The authors expressly reported no evidence of such behaviour in real deployments; not a seed experiment |

### Tier C: supplementary, provisional evidence

| Item | Result | Proper use | Limitation |
|---|---|---|---|
| **Inoculating Language Models Against Misalignment** (Bejjani, January 2026) | Training-example framing altered whether a narrow trait generalised; some prompts mitigated generalisation but introduced trigger-like trade-offs | Useful mechanistic and design evidence | Personal research report/small-model study; must not be presented as equivalent in status to a peer-reviewed or major-laboratory paper |

The classification is deliberately conservative. It preserves the force of the strongest evidence rather than weakening the case by treating every adjacent result as an equal and independent confirmation.

Primary sources:

- <https://arxiv.org/abs/2601.10160>
- <https://arxiv.org/abs/2502.17424>
- <https://arxiv.org/abs/2507.14805>
- <https://arxiv.org/abs/2507.21509>
- <https://arxiv.org/abs/2412.14093>
- <https://arxiv.org/abs/2603.06816>
- <https://arxiv.org/abs/2510.11288>
- <https://arxiv.org/abs/2604.20995>
- <https://arxiv.org/abs/2510.05179>
- <https://josephbejjani.com/misalignment-inoculation/>

---

## 5. Prior art and the proper boundary of the priority claim

### 5.1 Generic firstness is rejected

The following propositions were already present before 8 December 2024:

1. **Principles can be used during training:** Constitutional AI used a constitution in supervised and reinforcement-learning phases in 2022.  
   Source: <https://arxiv.org/abs/2212.08073>

2. **Preferences should be incorporated from the start of pretraining:** Korbak et al. found preference-aware pretraining superior to standard pretraining followed by feedback fine-tuning in 2023.  
   Source: <https://arxiv.org/abs/2302.08582>

3. **Alignment should be architectural rather than imposed post hoc:** de Melo et al. argued in August 2024 that inner alignment is undecidable in the general case and that alignment should be guaranteed by architecture.  
   Source: <https://arxiv.org/abs/2408.08995>

Accordingly, the following statements must not be used:

- "Eastwood was the first person to propose early alignment."
- "Eastwood first proposed embedding values during training."
- "Eastwood first showed that post-hoc alignment is inadequate."
- "Eastwood first proved alignment undecidable."
- "No one before Eastwood proposed architectural alignment."

### 5.2 The potentially distinctive composite

The narrower priority proposition is materially different:

> **Recursive capability growth magnifies the consequences of foundational value architecture; increasingly capable systems may evade or outreason external controls; therefore the decisive intervention is formative, identity-level, load-bearing alignment capable of co-scaling with the intelligence it guides.**

That is the proposition to compare with the 8 December 2024 artefact. It is not defeated merely because separate antecedents existed for early preference training, constitutions, hard alignment, scalable oversight, developmental analogy, or undecidability.

### 5.3 Documentary position

The repository records:

- a self-emailed manuscript bundle timestamped **8 December 2024 at 02:45:18 UTC**;
- SHA-256 `f0d1f38ffd8546152d9d9d28dc5ec083c16a35858f2c12b63e69db7ed50901ad`;
- a Gmail Message-ID and DKIM-verification route;
- the governing relation `U = I x R`; and
- a line identified as requiring the embedding of moral and ethical frameworks.

The current Eden papers further distinguish the underlying December 2024 thesis from technical names introduced in the expanded April 2025 manuscript. That distinction is correct and must be retained.

### 5.4 Priority ruling

On the present repository record:

- priority for the **existence and date of an Eastwood recursion-plus-embedded-correction thesis** is strongly documented;
- priority for the **full four-part composite formulation** is **provisionally strong but remains subject to line-by-line verification of the December attachment**;
- priority for the exact **Paper X `beta > k` formalism** belongs to Paper X's own later documentary date and must not be backdated to December 2024 unless the exact mathematical relationship appears in that artefact; and
- absolute worldwide novelty cannot be pronounced without an exhaustive expert prior-art review.

This is not a weakness in the case. It is the distinction that makes the surviving claim credible.

---

## 6. Relationship to Paper X: what the studies do and do not establish

Paper X's own claims ledger distinguishes the earlier conceptual thesis from the later closed-form co-scaling contribution. That distinction should govern all future publicity.

The later studies support at least three premises relevant to a co-scaling model:

1. **initial-condition sensitivity:** changing formative inputs changes downstream behavioural trajectories;
2. **cross-domain generalisation:** local training signals can produce global disposition changes; and
3. **persistence:** some formative effects survive subsequent post-training or manifest strategically when later training conflicts with prior preferences.

In schematic terms, the studies support a non-zero dependence of later behaviour `B_T` on formative state `S_0`:

`partial B_T / partial S_0 != 0`

They do not establish the stronger Paper X claim that a correction term scales with a particular exponent faster than an opposing term, nor do they measure `beta`, `k`, or the threshold at which one dominates the other.

A direct empirical test of Paper X would require, at minimum:

1. controlled embedded and external-alignment conditions;
2. repeated increases in capability or recursive depth;
3. pre-registered measures of alignment retention and correction strength at each depth;
4. estimation of the competing scaling exponents with uncertainty intervals;
5. adversarial tests for strategic compliance and evaluator gaming; and
6. replication across base models, architectures, and training pipelines.

Until then, the correct relationship is:

> **The seed-dependence literature supplies convergent empirical support for the causal premises of co-scaling correction, while Paper X supplies a proposed mathematical account of why those premises may become decisive under recursion.**

---

## 7. Why the stronger categorical sentence must be amended

The proposed sentence stated that formative training determines later generalisation "in ways that post-hoc correction cannot fully reverse." That final clause goes beyond the evidence.

Two primary reasons compel amendment:

1. Persona Vectors reports that personality shifts can be mitigated through post-hoc intervention and avoided through preventative steering.
2. Gomez's operational-control study, using the Anthropic blackmail scenario across ten models and 66,600 samples, reduced blackmail from 38.73% to 1.21% through an externally governed escalation channel, and to 0.85% with an additional compliance bulletin.

Source: <https://arxiv.org/abs/2510.05192>

Those results do not prove that external control will scale indefinitely. They do prove that **"post-hoc correction cannot work"** is presently false as a general empirical statement.

The correct proposition is:

> **Post-hoc correction may be incomplete, distribution-specific, strategically fragile, or non-scaling, which makes formative alignment structurally important; present evidence does not show that all later correction is impossible.**

---

## 8. Verification conditions marked CHECK

Before the strongest priority formulation is placed in a journal submission, investor document, press release, or correspondence with Anthropic, the following must be completed:

- **«CHECK 1»** Produce a line-by-line schedule of the 8 December 2024 attachments identifying the exact text for: recursion/compounding; early value embedding; insufficiency of external controls; identity-level or load-bearing alignment; and the temporal "window" proposition.
- **«CHECK 2»** Recompute and record the SHA-256 of the exact `.eml` made available to reviewers and verify the DKIM result using a reproducible toolchain.
- **«CHECK 3»** Separate every proposition first appearing on 8 December 2024 from technical names and elaborations first appearing on 30 April 2025 or in the January 2026 publication.
- **«CHECK 4»** Construct an authorship and citation-lineage graph before using the word "independent" for the eleven studies.
- **«CHECK 5»** Confirm the journal-publication date and final bibliographic form of the emergent-misalignment and subliminal-learning papers; distinguish arXiv priority dates from later journal dates.
- **«CHECK 6»** Obtain an external mathematical review before claiming that any external experiment validates `beta > k`.

Until those checks are complete, use **"timestamped manuscript records"**, **"supports"**, **"corroborates"**, and **"anticipated"**, not **"proved priority beyond dispute"** or **"mathematically confirmed by Anthropic."**

---

## 9. Authorised public formulations

### 9.1 One-sentence public statement

> **An 8 December 2024 timestamped manuscript recorded a coupled thesis that recursion magnifies the consequences of foundational value architecture and therefore makes early, embedded alignment structurally important; subsequent controlled studies, most directly Anthropic's May 2026 Model Spec Midtraining work, have substantially corroborated the narrower causal mechanism by showing that formative training conditions can shape broad later generalisation and persist through subsequent training.**

### 9.2 Fuller academic statement

> **Eastwood's dated record does not establish priority for the generic idea of introducing values early, which has clear antecedents. Its potentially distinctive contribution is the composite claim that recursive capability magnifies whatever formative value structure is seeded, that external control may become increasingly fragile, and that alignment must therefore become identity-level, load-bearing and capable of co-scaling with intelligence. Model Spec Midtraining, Alignment Pretraining, Emergent Misalignment, Subliminal Learning and related work provide substantial empirical corroboration for the seed-dependence component. They do not yet directly establish recursive amplification of values, categorical failure of all post-hoc correction, or the empirical truth of Paper X's `beta > k` law.**

### 9.3 Memorable formulation with technical gloss

> **The seed shapes the forest. In technical terms, AI development is path-dependent: what is taught during formation changes the basin into which later training generalises.**

### 9.4 Priority formulation

> **The present record supports a bounded priority claim for Eastwood's December 2024 composite synthesis, not a generic firstness claim for early alignment, constitutional training, or architectural safety.**

---

## 10. Formulations not authorised by this judgment

The following exceed the evidence and should be removed wherever they occur:

- "Eleven independent experiments prove my theory."
- "Anthropic has proved that the seed determines everything."
- "The studies prove post-hoc alignment cannot work."
- "Paper X's mathematics has been experimentally validated by Anthropic."
- "I was first to propose putting values into pretraining."
- "Every sufficiently capable AI will inevitably reason around every control."
- "Agentic misalignment has been observed in real corporate deployment."
- "Sixteen in-context examples caused broad misalignment."
- "All eleven items have equal evidential or publication status."

---

## 11. Final judgment

The synthesis is **right in its core direction but required narrowing at four points: determinism, independence, irreversibility, and mathematical validation**.

The proper conclusion is powerful:

1. The formative-history thesis is no longer merely philosophical. It now has substantial controlled empirical support.
2. Model Spec Midtraining is the closest direct convergence because it shows that an earlier value specification changes the broad meaning learned from the same later behavioural examples.
3. Alignment Pretraining, Emergent Misalignment, Subliminal Learning, Persona Vectors, Alignment Faking and Dark Triad model organisms materially strengthen the same causal picture from different experimental angles.
4. The evidence establishes path dependence, broad generalisation, internal trait structure and conditional persistence. It does not establish complete destiny or universal irreversibility.
5. The December 2024 record appears to precede the strongest later convergences and supports a bounded priority claim for the recursion-plus-formative-embedding synthesis.
6. Prior art defeats generic firstness but does not necessarily defeat the narrower composite claim.
7. Paper X provides a later mathematical formalisation that is consistent with the empirical direction of travel, but the external studies have not yet measured or validated its defining exponent inequality.

**Judgment entered accordingly on 10 July 2026.**

---

## 12. Internal repository materials considered

- `PRIORITY-VERIFY.md`
- `PRIORITY-AND-PROVENANCE.md`
- `papers/Eden-Engineering/Eden-Engineering.html`
- `papers/Eden-Vision/Eden-Vision.html`
- `papers/Paper-X-Coupled-CoScaling-Correction/CLAIMS.md`
- `papers/Paper-XI-Convergence/Paper-XI-Convergent-Evidence.html`
- `experiments/CONVERGENCE-19-INDEPENDENCE-STRESS-TEST-2026-07-05.md`

This judgment intentionally creates a separate record rather than silently rewriting any existing publication.