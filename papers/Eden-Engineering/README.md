# Eden Protocol: Engineering Specification

**Full title:** The Eden Protocol v6.1: Engineering Specification for Embedded AI Alignment
**Version:** v6.1
**First published:** 22 February 2026
**Author:** Michael Darius Eastwood

## Summary

Current alignment approaches produce alignment scaling exponents of approximately zero, meaning safety degrades relative to capability as recursive depth increases. If AI capability scales super-linearly while external alignment constraints do not participate in the recursive process, then the capability-alignment gap widens without bound. The Eden Protocol provides a complete engineering specification for embedded alignment -- safety mechanisms that participate in the recursive computation itself rather than constraining it from outside. This document specifies the technical architecture, governance framework, and implementation pathway.

## Experiments

Experiment data supporting this paper is distributed across the alignment-scaling and Eden intervention suites.
See [`../Paper-III-Alignment-Scaling-Problem/experiments/`](../Paper-III-Alignment-Scaling-Problem/experiments/) and [`../Paper-V-Stewardship-Gene/experiments/`](../Paper-V-Stewardship-Gene/experiments/).

## Links

- **OSF DOI:** https://doi.org/10.17605/OSF.IO/6C5XB
- **GitHub:** https://github.com/MichaelDariusEastwood/arc-principle-validation

## Declarations & Statement of Authorship

**1. Human Authorship & Intellectual Property Assertion**
The author, Michael Darius Eastwood, is the sole creator and copyright holder of this work. All core concepts, hypotheses, architectural frameworks, and conclusions originate exclusively from human ideation.

- **United Kingdom:** In accordance with the Copyright, Designs and Patents Act 1988 (including s.9(3)), the author asserts that they undertook the "necessary arrangements" for the creation of this work. The AI served strictly as an instrument to execute the author's specific instructions, and the work is a human-authored work assisted by a computer - not a computer-generated work.
- **United States:** In compliance with US Copyright Office guidance, the author certifies that the selection, coordination, and arrangement of all text were performed by the human author, rendering the final expression a product of human intellect.

**2. Nature of AI-Assisted Workflows**
Generative artificial-intelligence tools were used purely as assistive, high-velocity instruments to support the mechanical execution of the research process (analogous to advanced text editors or reference software). AI assistance was restricted to prose refinement, structural formatting, cross-referencing literature, and brainstorming counter-arguments, all under direct human oversight and manual verification. Every underlying idea, hypothesis, experimental design, test, synthesis, and final editorial judgement is human-driven; no content herein constitutes an unedited or unverified machine output, and nothing is relied upon without human checking.

**3. Inventions & Patent Rights**
Any novel technical contributions, structural designs, or algorithmic discoveries described in this work are the exclusive intellectual property of the human author. Consistent with UK and US authorities that an AI system cannot be a named inventor (e.g. Thaler v Comptroller-General), the AI functioned solely as a calculation and search utility and did not autonomously conceive or invent any solution presented; the conception is the author's.


## Declaration of AI Use

The author used Claude (Anthropic), GPT (OpenAI), Gemini (Google), and DeepSeek AI to draft sections, refine clarity, and check mathematical consistency. The research question, theoretical framework, formalism, experimental predictions, and scientific judgement are human work. The author takes full responsibility for all claims, interpretations, errors, and conclusions. AI models used as experimental *subjects* or *evaluators* are named in each paper's methods (e.g. Claude Opus, DeepSeek-V4, GPT-5.5, Gemini, gpt-3.5-turbo, gpt-4o-mini, as applicable). Don't believe - verify.

- **Licence:** CC BY-NC-ND 4.0 (paper text/figures) / proprietary (code) - see repo `LICENSE.md`.
