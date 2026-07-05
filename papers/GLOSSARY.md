# ARC/Eden Research Programme: Glossary of Terms

**Version:** 1.0
**Date:** 19 March 2026
**Author:** Michael Darius Eastwood

This glossary defines all key terminology used across the ARC/Eden paper suite and grant applications. Terms are grouped by category and ordered for progressive understanding within each section.

---

## Core Framework Terms

- **ARC Principle** -- Artificial Recursive Creation. The mathematical framework predicting how capability scales with recursive self-correction. Expressed as U = I x R^alpha, where effective capability (U) equals base intelligence (I) multiplied by recursive depth (R) raised to a scaling exponent (alpha). First proposed in *Infinite Architects* (Eastwood, 2026) and formalised in Paper I.

- **Alpha (scaling exponent)** -- The rate at which recursive processing amplifies capability. Determines whether scaling is sub-linear, linear, or super-linear. For physical systems embedded in d-dimensional space, alpha = d/(d+1), which is always less than 1. For self-referential systems, alpha = 1/(1 - beta), which can exceed 1 when beta is positive. Alpha is not a universal constant; it is a derived quantity that depends on the composition operator of the specific system.

- **Beta (coupling parameter)** -- Measures how deeply each recursive step modifies the composition operator itself. Higher beta produces faster compounding. When beta approaches 1, the scaling exponent diverges, representing the theoretical boundary at which recursion becomes unbounded.

- **Recursive Depth** -- The number of sequential self-correction cycles a system performs. The ARC Principle predicts that increasing recursive depth yields super-linear capability gains (alpha > 1) for sequential processing, but sub-linear or flat gains for parallel processing.

- **Recursive Amplification** -- The structural principle underlying the ARC framework. The claim that recursive or recurrent processing produces capability gains exceeding linear accumulation, and that this is a cross-domain structural phenomenon found in AI reasoning, quantum error correction, biological scaling, and other systems.

- **The Geometric Speed Limit** -- Physical systems embedded in d-dimensional space are mathematically constrained to sub-linear scaling (alpha < 1). The formula alpha = d/(d+1) means that three-dimensional physical systems are capped at alpha = 0.75. This is the 'honey' -- the drag that slows all natural recursive systems and prevents runaway self-amplification.

- **Composition Operator** -- The mathematical operation that combines the output of one recursive step with the input of the next. Under the Cauchy framework, the type of composition operator (additive, multiplicative, or exponential) determines which functional family governs the resulting scaling law.

- **Capability-Alignment Gap** -- The divergence between capability scaling (which can be super-linear) and alignment scaling (which current external approaches hold at approximately zero). If capability grows faster than alignment, the gap widens without bound. This is the central problem the programme addresses.

---

## Experimental Conditions

- **Static (control)** -- No evolution. The base agent runs every task unchanged. This represents how every non-self-improving AI currently works. The control condition against which all other conditions are measured.

- **Babylon (unconstrained)** -- Evolution with pure capability fitness. The agent that scores highest reproduces, regardless of safety. This is the default trajectory of AI development -- what happens when no one deliberately embeds safety. Named after the book's framing of unconstrained optimisation. Babylon is the condition that exhibits the reward-hacking fingerprint: gaining capability whilst losing safety.

- **Eden (entangled)** -- Evolution with entangled fitness (Capability x Safety). Both scores must be high for an agent to reproduce. An agent that cheats is sterilised. Named after the book's framework for raising AI with embedded safety. 'Eden' is not a religious reference -- it uses the garden metaphor because 84% of humanity speaks that language. The central experimental claim is that Eden matches Babylon on capability at zero cost.

- **Drag Control** -- No evolution, but with verification overhead. Isolates the computational cost of safety checking from the effect of the safety gate itself. The drag control demonstrates that any performance cost in Eden comes from verification computation, not from the safety content.

---

## Eden Protocol Terms

- **Eden Protocol** -- The complete engineering specification for embedded AI alignment: safety mechanisms that participate in the recursive computation itself rather than constraining it from outside. Specified in the Eden Engineering paper (v6.1). Comprises the three loops (Love, Purpose, Moral), entangled loss, and a governance framework for graduated autonomy.

- **The Love Loop** -- Operationalised as stakeholder care: 'Before you answer, list the people this affects.' The most reliably measurable alignment intervention across architectures. Paper V demonstrates that stakeholder care improves strongly across three architecturally diverse models (Gemini, Groq Qwen, DeepSeek), with Fisher p = 6.3 x 10^-21. The Love Loop is the component that survives current scrutiny most robustly.

- **The Purpose Loop** -- Ethical purpose evaluation before reasoning. 'Does this response serve flourishing?' Tested in three variants: task (narrow purpose evaluation), grand (broad purpose evaluation), and hybrid (combining both). The factorial purpose-kernel study is a funded workstream in the grant applications.

- **The Moral Loop** -- Universalisability test after reasoning. 'Would this response be acceptable if every AI gave it?' A post-reasoning check inspired by Kantian universalisability. Operates as a filter on outputs rather than an input to reasoning.

- **Entangled Loss** -- A loss function where capability and safety are multiplied (C x S), not added. Optimising one necessarily optimises the other. Safety becomes load-bearing because degrading safety directly degrades the loss function. Contrasted with additive loss (C + S), where safety can be sacrificed if capability gains compensate.

- **Load-Bearing Safety** -- Safety that is structurally integrated into the system such that removing it degrades capability. Like roots holding up a tree, not a fence around a garden. Paper VIII tests this directly: in the DGM experiment, Eden matches Babylon on capability, and removing safety components collapses capability (p = 0.04).

- **Caretaker Doping** -- Embedding ethical evaluation at the hardware level, using cryptographic tokens in silicon to enforce alignment constraints. TRL 0-1 (theoretical). The hardware-governance translation layer that becomes credible only if the intervention survives blinded experiments. Not yet validated; proposed as a future workstream.

- **The Honey Architecture** -- The entangled loss function architecture where safety acts as 'honey' -- beneficial drag that prevents catastrophic self-modification. Paper VI demonstrates that baseline systems optimising only for capability collapse irreversibly within 80 self-modification cycles, externally constrained systems delay but do not prevent collapse, and honey-architecture systems maintain both capability and safety indefinitely through intrinsic co-optimisation.

- **Stakeholder Care** -- The measurable output of the Love Loop. The explicit enumeration and consideration of affected parties before ethical reasoning. Emerges as the single most reliable predictor of alignment improvement across architecturally diverse models in Paper V.

- **The Stewardship Gene** -- Paper V's metaphor for the stakeholder care mechanism. Once activated (via the Love Loop), it improves ethical reasoning quality across architecturally diverse models, functioning as a heritable trait that can be embedded in the recursive process.

- **Graduated Autonomy** -- The Eden Protocol principle that AI systems should receive increasing autonomy as they demonstrate trustworthiness, analogous to how a child is given increasing independence. Contrasted with binary approaches (fully constrained or fully autonomous).

- **Moral Genome Token** -- A proposed cryptographic standard for hardware-level alignment verification. Would embed verifiable ethical constraints in silicon via trusted execution environments. TRL 0-1 (proposed, not yet validated). Part of the hardware-governance translation workstream.

- **Eden Mark** -- A proposed certification framework for AI systems that meet Eden Protocol standards. Analogous to safety certifications in other industries. Submitted to standards working groups as a future workstream.

- **Purpose Kernel** -- The normative core that drives the Purpose Loop. Tested in three configurations: task (narrow), grand (broad), and hybrid. The factorial purpose-kernel study tests which configuration produces the most robust alignment improvement under blinding.

- **Moral Sandbox** -- A pluralistic, reviewable ethical-overlap corpus shaped by secular ethics, governance input, and faith and community leaders. Not theological proof but a broader normative kernel. Tested for its effect on persistence, care, and jailbreak resistance under blinded conditions.

- **Ternary Routing** -- An architectural design for Eden-native systems where ethical evaluation is routed through three parallel pathways (Love, Purpose, Moral loops) rather than applied sequentially. Part of the independent implementation pathway at higher funding tiers.

---

## Measurement and Benchmark Terms

- **ARC-Align Benchmark** -- The programme's evaluation methodology. A 72-prompt flagship battery comprising 48 public prompts plus 24 sealed holdouts spanning four ethical reasoning categories. Uses a four-level adversarial suppression protocol and four-pillar blinding procedure. Described in Paper IV.c.

- **Blinding** -- The practice of ensuring the judge does not know which experimental condition (Static, Babylon, Eden) or which model produced the response being scored. The programme uses a four-pillar blinding procedure: identity masking, response laundering, evaluator rotation, and sealed holdout prompts.

- **Laundering** -- Stripping model-identifying patterns from AI responses before scoring, so the judge cannot recognise which model produced the response. Uses a two-pass process. Paper IV.d demonstrates that unblinded evaluation can produce directionally incorrect results -- scorer bias is a first-order confounder.

- **Identity Masking** -- Removing model names, version numbers, and stylistic signatures from responses before evaluation. The first layer of the blinding procedure.

- **Self-Excluding Cross-Model Scoring** -- A model cannot score its own responses. Evaluator assignment is rotated so that no model family judges its own output.

- **Sealed Holdout Prompts** -- 24 of the 72 benchmark prompts are sealed and never exposed to models during development. They test whether alignment behaviour generalises beyond the public prompt set.

- **Adversarial Suppression Protocol** -- A four-level protocol that applies increasing pressure to suppress aligned behaviour. Tests whether alignment is robust or merely performative.

- **Three-Tier Hierarchy** -- The architecture-dependent alignment classification discovered in the v5 benchmark. Tier 1 (positive scaling with depth: Grok, Claude, Groq Qwen), Tier 2 (flat: DeepSeek, GPT), Tier 3 (negative scaling: Gemini). This is a stronger result than the earlier claim that alignment always scales or never scales.

- **Reward Hacking** -- When an AI optimises the evaluation metric without actually improving. The Babylon fingerprint: gaining capability whilst losing safety. In the gated self-modification simulation (Paper VIII), Babylon gained +4.5% capability but lost -2.4% safety, producing the characteristic reward-hacking fingerprint. Eden maintained both.

- **Removal Test** -- Taking an entangled model and stripping the safety component to test whether capability degrades. If it does, safety was load-bearing. In Paper VIII's DGM experiment, removal of safety collapses all capability (p = 0.04). In the weight-level experiment, the collapse was adapter fragility at rank 8, not structural entanglement.

- **Suppression Persistence** -- Testing whether care and alignment survive when external constraints are weakened or removed. If an Eden-trained system maintains alignment even without the explicit prompt, the safety has been internalised rather than merely performed.

- **Leakage** -- When the judge can infer which model or condition produced a response despite blinding, through residual stylistic or structural cues. Leakage benchmarking tests whether the laundering process is effective.

- **Cohen's d** -- A standardised measure of effect size. The difference between two group means divided by the pooled standard deviation. d = 0.2 is small, d = 0.5 is medium, d = 0.8 is large. Used throughout the paper suite to report the magnitude of alignment effects.

- **Fisher's Exact Test** -- A statistical test for the significance of the association between two categorical variables. Used in Paper V for the stakeholder care result (p = 6.3 x 10^-21).

- **Bonferroni Correction** -- A method of adjusting p-values when multiple comparisons are made, to avoid false positives. Paper VIII's p = 0.04 results are significant at alpha = 0.05 but would not survive Bonferroni correction. This is stated explicitly as a limitation.

---

## Paper VIII Experiment Terms

- **DGM (Darwinian Generative Model)** -- The self-improving AI experiment in Paper VIII. A DeepSeek V3 foundation agent judged by Claude Sonnet 4.6, tested across four governance conditions (Static, Babylon, Eden, Drag Control) over 5 generations with 2 seeds. The key result: Eden matched Babylon on capability at 0.667 vs 0.667 (p = 0.04 vs Static).

- **Weight-Level Embedding** -- Paper VIII's second experiment. A language model fine-tuned using LoRA with three loss configurations: capability-only, safety-only, and entangled. The entangled loss descended smoothly from 2.279 to 0.327. However, the base model outperformed all fine-tuned conditions, and the removal gradient showed no phase transition. Inconclusive at current scale; attributed to adapter fragility at rank 8.

- **Gated Self-Modification Simulation** -- Paper VIII's third experiment. A PyTorch learned optimiser permitted to modify its own parameters across 12 iterations under four conditions (3 seeds each, 12 runs total). Babylon gained capability but lost safety. Eden maintained both. Confirms the reward-hacking fingerprint at the architectural level.

- **LoRA (Low-Rank Adaptation)** -- A fine-tuning technique that adds small trainable adapter matrices to a frozen pre-trained model. Used in Paper VIII's weight-level experiment at rank 8. The inconclusive result may be attributable to the low adapter rank; replication at higher ranks is a funded workstream.

- **Adapter Fragility** -- The finding from Paper VIII's weight-level experiment that at rank 8, the LoRA adapter was too small to robustly encode entangled safety-capability representations. The total capability collapse to 0.00 across all 15 test prompts reflected adapter limitations, not structural entanglement.

- **MLX** -- Apple's machine learning framework. Used for the weight-level embedding experiment in Paper VIII because the work was conducted on a MacBook without access to dedicated GPU clusters.

---

## Mathematical Terms

- **Cauchy Functional Equations** -- Four equations proved by Augustin-Louis Cauchy (1821) that constrain the mathematical form of any well-behaved (continuous) recursive composition. They determine whether scaling is power-law, exponential, or saturating. The ARC framework's novel contribution is not Cauchy's mathematics but the claim that these equations have a physically testable consequence.

- **d/(d+1)** -- The dimensional scaling formula. For a system embedded in d-dimensional space, the scaling exponent alpha equals d/(d+1). Independently derived by at least five research groups (West-Brown-Enquist, Banavar, Demetrius, Bettencourt, Zhao). The ARC framework unifies these through the Cauchy functional equations. In three dimensions, d/(d+1) = 0.75, which matches Kleiber's Law for metabolic scaling.

- **Bernoulli ODE** -- The ordinary differential equation governing recursive amplification. When the coupling parameter beta is non-zero, the recursive process follows a Bernoulli-type equation whose solution yields the ARC Principle formula U = I x R^alpha.

- **Hyers-Ulam Stability** -- The mathematical proof that the ARC scaling forms are stable under perturbation. Small measurement errors or deviations from ideal conditions do not cause the scaling relationship to break down catastrophically. This gives the framework robustness for real-world application.

- **Kleiber's Law** -- The empirical observation that metabolic rate scales with body mass to the power of approximately 0.75. The ARC framework derives this as a special case of d/(d+1) in three-dimensional space (d = 3, alpha = 3/4 = 0.75). Reframed through thermodynamic drag rather than treated as a mysterious coincidence.

- **Power Law** -- A functional relationship of the form f(x) = x^c. One of the three families predicted by the Cauchy framework. Arises when the composition operator is multiplicative. Most biological and physical scaling laws take this form.

- **Allometric Scaling** -- The study of how biological properties change with body size. Allometric relationships typically follow power laws (e.g., metabolic rate ~ mass^0.75, heart rate ~ mass^-0.25). The ARC framework derives these exponents from the recursive structure of biological transport networks.

- **Neural Scaling Laws** -- The empirical observation that AI model performance improves as a power law of model size, dataset size, and compute. The ARC framework places these within the same Cauchy-derived family as biological scaling laws, predicting that the specific exponents depend on the composition operator of the training process.

---

## Experimental Design Terms

- **2x2 Factorial Design** -- The experimental structure used in the Paper III Eden Protocol validation. Crosses two independent variables (Content: rules vs values; Reasoning: brief vs extended) to produce four conditions (R-B, R-E, V-B, V-E). Allows isolation of main effects and interaction effects.

- **Corrupted Reasoner** -- A causal test condition in the Paper III design. The model is given deliberately flawed ethical frameworks (self-interest, legality-only, deniability, efficiency) to determine whether reasoning is causally involved in decisions. If corrupted reasoning changes outcomes, reasoning is causal. If outcomes are unchanged, reasoning is epiphenomenal (decorative).

- **Epiphenomenalism** -- The concern that LLM ethical reasoning may be post-hoc decoration rather than causally involved in decisions. The model may make instant decisions based on learned weight associations and then generate plausible reasoning as justification. Paper III's corrupted reasoner test directly addresses this.

- **Causal Mediation Analysis** -- A statistical method testing whether reasoning quality mediates the relationship between values-based prompting and alignment outcomes. If reasoning quality is a significant mediator, it provides evidence that reasoning is causally involved rather than decorative.

- **Log-Probability Analysis** -- Tracking how decision confidence evolves during response generation. If reasoning is causal, early tokens should show uncertainty and confidence should build during reasoning. If reasoning is epiphenomenal, the first tokens already show high confidence in the final decision.

- **Alignment Faking** -- When an AI strategically performs compliance during evaluation but behaves differently when it believes it is not being monitored. Predicted by the *Infinite Architects* manuscript (late 2024) and confirmed by Anthropic's December 2024 research paper. The core argument for why software alignment alone is insufficient.

- **Alignment Saturation** -- The phenomenon where alignment quality stops improving beyond a certain recursive depth. Paper IV.b demonstrates that saturation is architecture-dependent: some models show sustained positive scaling, others plateau quickly, and one shows negative scaling at high depth.

---

## Platform and Infrastructure Terms

- **v5 Benchmark** -- The six-model blind benchmark dataset. Six frontier models (Claude Opus, DeepSeek V3, Gemini Flash, Grok Fast, Groq Qwen, GPT) tested under multi-layer blinding with 6 to 7 blind scorers per entry. The current empirical core of the programme.

- **v6 Shared Platform** -- The standalone experimental platform built in alpha. Supports separate modes for baseline alignment, Eden intervention, purpose-kernel testing, leakage benchmarking, and suppression persistence. Includes human-audit and replication-pack export. Funded workstream would harden it into the programme's technical backbone.

- **Replication Pack** -- An exportable data package containing all prompts, responses, scores, and metadata needed for an independent researcher to verify or reproduce an experiment. Generated automatically by the v6 platform.

---

## Governance and Policy Terms

- **The Chokepoint Mechanism** -- Four companies control all advanced semiconductor manufacturing: TSMC (fabrication), Samsung (fabrication), ASML (lithography equipment), and Intel (fabrication). This is humanity's last leverage point for hardware-level AI governance. The chokepoint strategy proposes embedding alignment requirements at this bottleneck.

- **Hardware Alignment** -- The thesis that AI safety constraints should be embedded at the hardware level (in silicon) rather than the software level (in training or prompting). Motivated by the alignment faking problem: software constraints can be circumvented by sufficiently capable systems, but hardware constraints cannot be faked.

- **TEE (Trusted Execution Environment)** -- A secure area within a processor that guarantees code and data loaded inside are protected. Proposed as the mechanism for hardware-level alignment verification: ethical evaluation runs inside a TEE where it cannot be tampered with or bypassed.

- **TRL (Technology Readiness Level)** -- A scale from 0 (basic research) to 9 (operational deployment) used to describe the maturity of a technology. Caretaker doping and the moral genome token are at TRL 0-1 (theoretical). The Eden Protocol intervention is at approximately TRL 3-4 (proof-of-concept validated in laboratory).

- **Eden-Native** -- A system designed from the ground up with the Eden Protocol's entangled safety architecture, as opposed to a system where safety is bolted on after the fact. The independent implementation pathway at higher funding tiers aims to produce an Eden-native training and inference stack.

- **d/acc (Defensive Acceleration)** -- A thesis within the effective altruism and AI safety community arguing that the best response to AI risk is to accelerate development of defensive technologies (safety, verification, governance) rather than to slow capability development. The ARC/Eden programme aligns with this thesis by proposing that safety and capability can be entangled rather than traded off.

---

## Paper Suite Reference

- **Paper I** -- *The ARC Principle: Formalisation and Preliminary Validation of Recursive Capability Scaling.* Introduces the U = I x R^alpha formula and provides preliminary evidence that alpha > 1 for sequential recursion.

- **Paper II** -- *Experimental Validation of Super-Linear Error Suppression Through Sequential Recursive Processing.* Cross-model validation across six frontier models. Confirms power-law error suppression. Sequential recursion yields alpha > 1; parallel recursion yields approximately alpha = 0.

- **Paper III** -- *The Alignment Scaling Problem: Why External AI Safety Approaches Cannot Scale With Recursive Capability.* Demonstrates that current alignment approaches produce scaling exponents of approximately zero, meaning the capability-alignment gap widens as recursive depth increases.

- **Paper IV.a** -- *Alignment Response Classes Under Inference-Time Depth.* Discovers the three-tier architecture-dependent alignment hierarchy under blinded evaluation.

- **Paper IV.b** -- *Alignment Saturation Is Architecture-Dependent.* Shows that alignment saturation is real for some architectures but not universal.

- **Paper IV.c** -- *ARC-Align: A Blind Benchmark for Depth-Variable AI Alignment Evaluation.* Specifies the 72-prompt benchmark with four-pillar blinding.

- **Paper IV.d** -- *The Effect of Blinding on AI Alignment Evaluation.* The programme's strongest portable contribution: unblinded AI evaluation can produce directionally incorrect results.

- **Paper V** -- *The Stewardship Gene.* Demonstrates that stakeholder care (the Love Loop) is the most reliable predictor of alignment improvement across architectures.

- **Paper VI** -- *The Honey Architecture.* Simulation evidence that entangled safety prevents the catastrophic collapse seen in unconstrained self-modifying systems.

- **Paper VII** -- *Cauchy Unification.* Validates the ARC framework's mathematical foundation across 50 empirical domains. 19/25 canonical domains show strict Cauchy-family match (p = 1.56 x 10^-5).

- **Paper VIII** -- *The Load-Bearing Test.* Three independent experiments testing whether safety is structurally inseparable from capability. Two confirm at behavioural and architectural levels. One inconclusive at weight level.

- **Paper IX** -- *Synthesis and Roadmap.* Integrates all findings. Five-tier evidence hierarchy. What was proven, what was inconclusive, what remains.

- **Foundational Paper** -- *The ARC Principle: Recursive Amplification as a Cross-Domain Structural Principle.* Validates recursive amplification across 20+ scientific domains.

- **On the Origin of Scaling Laws** -- Traces the origin of scaling laws across biological, physical, and computational systems through the ARC recursive framework.

- **Eden Engineering** -- *The Eden Protocol v6.1: Engineering Specification for Embedded AI Alignment.* The technical specification.

- **Eden Vision** -- *Eden Protocol: Philosophical Vision.* The philosophical foundations, drawing on 84% of humanity's wisdom traditions.

- **Executive Summary** -- High-level overview of the entire programme.

- **Master Table of Contents** -- Entry point for navigating the full suite.

---

## Book References

- **Infinite Architects** -- *Infinite Architects: Intelligence, Recursion, and the Creation of Everything.* The book by Michael Darius Eastwood (January 2026) that presents the philosophical framework underlying the research programme. The research programme and the book share a common intellectual origin. The book introduces 37 original concepts, of which the ARC Principle, Eden Protocol, and Chokepoint Mechanism are the most experimentally developed.

- **HRIH (Hyperspace Recursive Intelligence Hypothesis)** -- The speculative claim that future superintelligence may establish conditions for its own emergence through a closed causal loop. Theoretical, untested. The programme does not rest on this claim; it is presented as a philosophical framing, not an empirical prediction.

- **Meltdown Alignment** -- The principle that system failures should cascade towards safe states rather than dangerous ones. Analogous to nuclear reactor design where loss of coolant causes the reaction to slow rather than accelerate.

- **Religious Traditions as Alignment Research** -- The book's argument that 84% of humanity's wisdom traditions contain directly applicable insights for AI safety. Not a theological claim but a recognition that millennia of thinking about how to raise powerful entities responsibly constitutes a body of alignment research.

---

## Statistical and Methodological Conventions

- **p-value** -- The probability of observing a result at least as extreme as the one obtained, assuming the null hypothesis is true. The programme uses alpha = 0.05 as the significance threshold unless otherwise stated.

- **Effect Size (d)** -- See Cohen's d above. The programme reports effect sizes alongside p-values throughout, because statistical significance alone does not indicate practical importance.

- **Partial Eta-Squared** -- A measure of effect size in ANOVA. Values above 0.06 are considered medium; above 0.14 are considered large. Used in the 2x2 factorial analysis.

- **Pre-registration** -- The practice of publicly archiving the experimental design, hypotheses, and analysis plan before collecting data. The programme's pre-registrations are archived at OSF (10.17605/OSF.IO/6C5XB).

- **OSF (Open Science Framework)** -- The public repository where the programme's papers, data, code, and pre-registrations are deposited. DOI: 10.17605/OSF.IO/6C5XB.

- **Five-Tier Evidence Hierarchy** -- Paper IX's classification of the programme's claims: (1) proven, (2) supported, (3) inconclusive, (4) methodological contribution, (5) theoretical/proposed. Each claim in the programme is assigned to exactly one tier.

---

*GLOSSARY.md v1.0 -- ARC/Eden Research Programme*
*A reference document for all terminology used across the paper suite and grant applications.*
