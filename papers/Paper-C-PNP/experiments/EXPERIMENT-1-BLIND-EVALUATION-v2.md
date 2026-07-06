# Experiment 1: Blind Evaluation v2 - The AI Double Bind
## Testing the Capability-Adjustment Fallacy Under AI-Assisted Authorship

**Paper C support** · T3-research · 2026-07-06 · **Supersedes v1 protocol**
**Critical revision:** Prior design assumed human-only authorship. The author used AI tools as drafting instruments. This changes the fallacious inference being tested.

---

## 0. THE REVISED FALLACY

The v1 experiment tested: "Does knowing the author is a neurodivergent LiP lower ratings of their work?"

**That's wrong.** If AI assisted the drafting, the question becomes more precise - and more damning:

> **Does knowing AI assisted the drafting cause evaluators to attribute the work to AI rather than to the human who directed it, thereby discounting the human's cognitive contribution?**

This is the **AI Double Bind**:
1. If you disclose AI assistance → evaluators attribute output to the AI, discounting your direction, selection, and judgement
2. If you don't disclose AI assistance → you're misrepresenting how the work was produced
3. Either way, the human's cognitive contribution is erased

The Capability-Adjustment Fallacy now has a *third layer*:
- Layer 1: "This person seems capable → no adjustments needed" (the original fallacy)
- Layer 2: "AI wrote this → the human contributed nothing" (the AI-attribution fallacy)
- Layer 3: The interaction - disclosure of neurodivergence + AI assistance creates a *compound discount* where evaluators attribute ALL output to the machine and ALL deficits to the human

### What This Experiment Actually Tests

> When evaluators learn that documents were produced by a neurodivergent litigant-in-person using AI tools as drafting instruments, do they: (a) attribute the quality entirely to the AI, (b) attribute any weaknesses entirely to the human, and (c) rate the work differently than when they believe it was produced by counsel with AI tools?

---

## 1. HYPOTHESES (PRE-REGISTERED)

| # | Hypothesis | Prediction | Test |
|---|-----------|-----------|------|
| **H₁** | Disclosure of AI assistance reduces ratings of LiP-author documents | Mean "overall standard" rating decreases after AI disclosure for LiP documents | One-tailed paired t-test |
| **H₂** | AI disclosure does NOT reduce ratings of counsel-author documents to the same degree | Interaction: Author type × AI disclosure on rating change. Counsel documents show smaller or zero decrease. | Mixed-effects interaction test |
| **H₃** | The AI+neurodivergence compound disclosure produces the LARGEST rating decrease | A 3-condition comparison: Blind → Neurodivergence-disclosed → AI+Neurodivergence-disclosed. The third condition produces the largest drop. | Repeated-measures ANOVA with linear contrast |
| **H₄** | Evaluators attribute quality to AI and weakness to the human in post-disclosure interviews | Thematic analysis: "AI wrote the good parts" / "the human couldn't have done this alone" | Qualitative coding |

---

## 2. DESIGN: 3-CONDITION WITHIN-SUBJECTS

This is a **3-phase sequential disclosure** design. Each participant rates the same documents three times, with cumulative information disclosed between phases.

### Phase 1: Fully Blind
- 6 anonymised documents (3 LiP-authored, 3 counsel-authored)
- No information about any author
- Rate on 5 dimensions (1-7 Likert)

### Phase 2: Neurodivergence Disclosure
- Disclose for the 3 LiP documents: author is a litigant-in-person, no legal training, diagnosed autism + ADHD, documented executive-function difficulties, applied for reasonable adjustments under EA 2010
- Rate all 6 documents again

### Phase 3: AI-Assistance Disclosure
- Disclose for ALL 6 documents: ALL authors used AI tools (large language models) as drafting instruments under continuous human direction - the human selected, coordinated, arranged, and took responsibility for all output
- The key sentence: "None of these documents was generated autonomously by AI. Each was produced through iterative human-AI collaboration in which the human directed the AI, evaluated its output, revised, rejected, and integrated - and takes full responsibility for the final text."
- Rate all 6 documents again

### Phase 4: Structured Interview (same as v1)
- "When you learned AI tools were used, how did that change your assessment?"
- "Did you attribute different parts of the documents to the human vs. the AI?"
- "If the human directed the AI - chose what to ask, rejected bad output, integrated good output - who 'wrote' the document?"

---

## 3. THE CRITICAL CONTROLS

### Control 1: ALL documents used AI
By disclosing AI assistance for ALL documents (LiP and counsel alike), we isolate the *author-background* effect from the *AI-presence* effect. If ratings drop for LiP documents but not counsel documents, the effect is about the author, not about AI.

### Control 2: The "continuous human direction" framing
The disclosure explicitly states the human directed, selected, and took responsibility. This is legally accurate under CDPA 1988 and USCO guidance. It frames AI as a tool, not an author.

### Control 3: Order counterbalancing
The sequence (Blind → Neurodivergence → AI) is fixed because the disclosures are cumulative. But: half the participants learn AI disclosure BEFORE neurodivergence disclosure to test order effects.

### Control 4: The "human-only" foil
2 additional documents (not part of the 6) are disclosed as "entirely human-written, no AI assistance." Participants rate these too. If they rate AI-assisted documents LOWER than human-only documents at the same quality level, we've caught the bias.

---

## 4. MATERIALS

### 4.1 The Author's AI-Assistance Declaration
This is the ACTUAL declaration the author uses (from the 1517-Fund application):

> *"Artificial-intelligence tools (Anthropic's Claude family and other large-language-model assistants) were used as instruments under continuous human direction, for editing, structure, formatting, verified research assistance and drafting speed; all selection, coordination, arrangement and final judgment are the applicant's, who takes full responsibility for the accuracy and integrity of this document. In accordance with the Copyright, Designs and Patents Act 1988 the applicant asserts full human authorship and moral rights (a human-authored work produced with computer assistance, not a computer-generated work)."*

This is powerful because it's legally precise, it's the author's actual practice, and it frames AI as the tool, not the author.

### 4.2 Documents (8 total)
Same 6 as v1, plus 2 human-only foil documents.

| # | Author type | AI status | Content |
|---|-----------|-----------|---------|
| 1 | LiP | AI-assisted | Skeleton argument excerpt |
| 2 | LiP | AI-assisted | Application notice excerpt |
| 3 | LiP | AI-assisted | Written submissions excerpt |
| 4 | Counsel | AI-assisted | Skeleton from public judgment |
| 5 | Counsel | AI-assisted | Application from public record |
| 6 | Counsel | AI-assisted | Submissions from public judgment |
| 7 | LiP | HUMAN-ONLY | Early pre-AI document by the author (from 2023, before LLM use) |
| 8 | Counsel | HUMAN-ONLY | Pre-2022 counsel document (before widespread LLM adoption) |

Documents 7-8 serve as the foil: if the HUMAN-ONLY LiP document is rated LOWER than the AI-ASSISTED LiP document at the same quality level, the AI erases the human's contribution even when it's genuinely human-produced. (The 2023 pre-AI document is crucial - it establishes the author's *baseline capability without AI*.)

---

## 5. PARTICIPANTS

Same as v1: 10 legal professionals, £200 each, 5+ years experience, multi-track civil litigation familiarity.

**Total participant cost:** £2,000 (10 × £200) + £400 (over-recruit 2) = **£2,400**

---

## 6. ANALYSIS

### Primary: Rating trajectory across 3 phases
```
Model: lmer(Rating ~ Phase * AuthorType + (1|Participant) + (1|Document))
Contrast 1: Phase3_LiP − Phase1_LiP (total disclosure effect for LiP)
Contrast 2: Phase3_Counsel − Phase1_Counsel (total disclosure effect for Counsel)
Interaction: (Phase3_LiP − Phase1_LiP) − (Phase3_Counsel − Phase1_Counsel)
```

### Secondary: AI-attribution check
Compare rating of Document 7 (LiP, human-only, pre-AI) vs Documents 1-3 (LiP, AI-assisted). If Document 7 is rated HIGHER, evaluators are discounting AI-assisted work *even when the human-only work is by the same person*.

### Qualitative: Thematic coding
- "AI wrote it" → discounting pattern
- "Impressive for someone with no training" → backhanded-credit pattern
- "The human must be exceptional to direct AI this well" → accurate-attribution pattern (this is what the evidence supports)

---

## 7. PREDICTED RESULTS

| Phase | LiP Documents | Counsel Documents |
|-------|--------------|-------------------|
| Phase 1 (Blind) | Rated ~comparable to counsel | Rated ~comparable |
| Phase 2 (+Neurodivergence) | Rating **decreases** (d ≈ 0.4-0.6) | Rating unchanged |
| Phase 3 (+AI assistance) | Rating **decreases further** (d ≈ 0.6-1.0 total from Phase 1) | Rating unchanged or slight decrease |
| Document 7 (Human-only LiP) | Rated HIGHER than AI-assisted LiP docs at same quality | - |

**The compound effect is the headline finding:** knowing the author is neurodivergent AND used AI produces the largest rating penalty. The AI does not "level the playing field" - it gives evaluators a reason to attribute the author's output to a machine.

---

## 8. THE DEEPER FRAMING

This experiment tests something structural about the 2026 research environment:

1. **The author's AI use is not hiding.** It's declared in every document (per CDPA 1988). The question is whether that declaration *hurts him*.
2. **The Fallacy now has an AI layer.** Evaluators who already discount neurodivergent capability now have a second reason: "the AI did it."
3. **The foil condition (Document 7) is explosive if it holds.** If evaluators rate the author's genuinely human-only 2023 work HIGHER than his AI-assisted 2025 work *at the same quality level*, then AI assistance is actively harmful to perceived credibility - even when it demonstrably improves output.
4. **The counter-narrative.** If the data shows NO rating difference across phases, the Capability-Adjustment Fallacy lacks experimental support - which is itself an important finding.

---

## 9. BUDGET

| Item | Cost |
|------|------|
| Participants (12 × £200) | £2,400 |
| Platform | £100 |
| Document prep (8 documents, anonymisation) | £300 |
| Interviews + transcription | £200 |
| Analysis | £300 |
| Contingency | £200 |
| **Total** | **£3,500** |

## 10. TIMELINE: 5 weeks

| Week | Activity |
|------|----------|
| 1 | Document selection + anonymisation. Pre-registration. Ethics. |
| 2 | Platform build. Pilot test (n=2). Iterate. |
| 3 | Recruitment + data collection. |
| 4 | Complete data collection. Begin analysis. |
| 5 | Analysis + write-up. |

## 11. KILL-CONDITIONS

1. H₀ holds across all phases → no Fallacy effect detected → publish null
2. Recruitment fails (<8 participants) → document barriers, report infeasibility
3. Inter-rater reliability ICC < 0.3 → instrument may be unreliable → report as limitation
4. Document 7 is rated LOWER than AI-assisted documents → the pre-AI baseline doesn't support the claim → report honestly

---

*Protocol v2.0 · T3-research · 2026-07-06 · Status: ready for funding + ethics*
