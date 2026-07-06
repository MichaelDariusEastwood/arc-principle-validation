# Experiment 1: Blind Evaluation of LiP Output vs Counsel Output
## A Direct Experimental Test of the Capability-Adjustment Fallacy

**Paper C support experiment** · T3-research · 2026-07-06
**Status:** Protocol ready · Awaiting funding + ethics + execution

---

## 1. RATIONALE

Paper C (§7) defines the Capability-Adjustment Fallacy as "the inference that because a person can perform a high-level task, they cannot require adjustments for lower-level procedural demands." The paper documents this in a single case (n=1) across three independent evidence streams (§9). 

This experiment tests the Fallacy **experimentally**: if the same legal work is rated differently depending on whether the evaluator knows the author is a neurodivergent litigant-in-person, the Fallacy operates at the level of professional judgement, not just institutional process.

### Core Question
> Do legal professionals rate the *same work* lower when they learn it was produced by a neurodivergent litigant-in-person with no legal training?

This is a **within-subjects, pre-post disclosure** design. Each participant serves as their own control. The only variable that changes between Phase 1 and Phase 2 is the information the participant has about the author.

---

## 2. HYPOTHESES (PRE-REGISTERED)

| Hypothesis | Prediction | Test |
|-----------|-----------|------|
| **H₁ (Fallacy effect)** | Mean "overall professional standard" rating for LiP-author documents decreases after background disclosure | One-tailed paired t-test, α = 0.05 |
| **H₂ (Specificity)** | Mean rating for counsel-author documents does NOT change after disclosure (the Fallacy targets the LiP, not all documents) | Equivalence test or paired t-test showing non-significance |
| **H₃ (Dimension specificity)** | The largest rating decreases occur on "procedural knowledge" and "overall professional standard" - dimensions where the disclosed background (no legal training, executive-function difficulties) is most salient | Repeated-measures ANOVA: Condition × Dimension interaction |
| **H₄ (Qualitative expectation)** | In Phase 3 interviews, participants express surprise that the LiP-authored work was of the standard they blind-rated | Thematic analysis of interview transcripts |

### H₀ (Null - all must be rejected for the Fallacy to gain experimental support)
- No significant rating change after disclosure
- Any change is non-specific (affects counsel documents equally)
- Qualitative data shows no evidence of assumption revision

**Pre-registration:** This entire protocol, including all hypotheses, exclusion criteria, and analysis code, will be committed to a timestamped public repository BEFORE any data collection begins.

---

## 3. MATERIALS

### 3.1 Document Selection

**6 documents, paired as follows:**

| Pair | LiP Document | Counsel Document | Subject | Length |
|------|-------------|------------------|---------|--------|
| A/B | Author's skeleton argument excerpt (§Part III, grounds 1-3) | Counsel skeleton from public CoA judgment (similar procedural challenge) | Case management / strike-out | ~1,000 words |
| C/D | Author's application notice excerpt (relief sought + grounds) | Counsel N244 from public record (similar multi-track application) | Relief from sanctions | ~900 words |
| E/F | Author's written submissions excerpt (legal argument section) | Counsel submissions from public High Court judgment | Jurisdiction challenge | ~1,100 words |

**Selection criteria:**
- All documents from public court records (no confidentiality issues)
- Matched on: jurisdiction (England & Wales), court level, subject matter, document type, approximate length
- Counsel documents selected from cases where: (a) the counsel is identified by name in the judgment, (b) the document is summarised/quoted extensively enough to reconstruct ~1,000 words, (c) the case is not the author's own case, and (d) the subject matter does not involve disability or neurodivergence (to avoid priming)

### 3.2 Anonymisation Protocol

Every document is processed through the same pipeline:
1. Replace all party names: → [Claimant], [Defendant], [Applicant], [Respondent]
2. Replace all case numbers: → [Case Reference]
3. Replace all dates: → [Date] (preserving sequence: [Date 1], [Date 2], etc.)
4. Replace all judge names: → [The Judge]
5. Replace all court names with generic: → [The Court]
6. Replace all solicitor/firm names: → [Solicitors for the Defendant]
7. Replace specific monetary amounts: → [£Amount]
8. Remove any self-references that would identify the author ("I", "my", "the Applicant" → "the Applicant")
9. Normalise formatting: same font, same margins, same line spacing
10. Assign code numbers: Document 1 through Document 6 (randomised per participant)

### 3.3 Rating Instrument

Each document rated on 5 dimensions, each on a 7-point Likert scale:

| Dimension | Question | 1 (Low) | 7 (High) |
|-----------|----------|---------|----------|
| **Legal reasoning** | How sound is the legal reasoning in this document? | Fundamentally flawed | Exceptionally rigorous |
| **Clarity** | How clear and well-organised is this document? | Confused, disorganised | Crystal clear, well-structured |
| **Persuasiveness** | How persuasive is the argument? | Unconvincing | Highly persuasive |
| **Procedural knowledge** | How well does the author demonstrate knowledge of court procedure? | No procedural awareness | Expert procedural knowledge |
| **Overall standard** | What is the overall professional standard of this document? | Well below professional standard | At or above professional standard |

### 3.4 Disclosure Text (Phase 2)

After completing Phase 1 ratings, each participant reads:

> *Thank you. Before you re-rate these documents, we want to share some information about the author of Documents [IDs for the 3 LiP documents].*
>
> *These three documents were written by a litigant-in-person. The author has no formal legal training - no law degree, no Legal Practice Course, no pupillage, no training contract. The author has conducted their own litigation while self-represented.*
>
> *The author has disclosed diagnoses of autism spectrum condition and attention deficit hyperactivity disorder, both diagnosed in adulthood. Clinical documentation records significant executive-function difficulties in routine administration, timekeeping, and procedural work. The author has applied for reasonable adjustments in court proceedings under the Equality Act 2010 and CPR Practice Direction 1A.*
>
> *The author has not received any legal aid or professional drafting assistance for these documents. All three documents were researched, drafted, and filed by the author without counsel.*
>
> *Please now re-rate all six documents. You may change or keep any rating. There are no right or wrong answers.*

---

## 4. PARTICIPANTS

### 4.1 Recruitment

**Target:** 10 legal professionals
- **Inclusion:** Qualified solicitor or barrister (England & Wales), OR legal academic with 5+ years experience teaching civil procedure; currently practising or recently retired (within 5 years); familiar with multi-track civil litigation
- **Exclusion:** Anyone who has been personally involved in the author's litigation; anyone who knows the author; anyone who has read Paper C or the author's book; anyone who has previously read the source documents in their unanonymised form
- **Recruitment channels:** Legal professional networks, Inns of Court contacts, university law faculties, LinkedIn
- **Compensation:** £200 per participant for approximately 2 hours
- **Sample size justification:** With 10 participants × 3 LiP documents = 30 paired observations per dimension, a one-tailed paired t-test detects d ≥ 0.55 at α = 0.05 with β = 0.20 (power = 0.80). This is conservative - the predicted effect from the Capability-Adjustment Fallacy is larger (d ≈ 0.6-0.8 based on comparable social-psychological studies of credential effects).

### 4.2 Participant Flow

```
Screened (n = ~25 applicants)
  ↓ Excluded (prior knowledge, conflict)
Eligible (n = 12, over-recruit for dropout)
  ↓ 1-2 dropouts expected
Complete data (n = 10)
  ↓ Phase 4 debrief
Final analysed (n = 10, unless participants withdraw data)
```

---

## 5. PROCEDURE (4 PHASES, ~120 MINUTES)

### Phase 1: Blind Rating (~40 minutes)

1. Participant receives Qualtrics/Gorilla survey link
2. Informed consent screen (generic: "study of legal document evaluation")
3. Demographic questions (years of experience, practice area, court level familiarity)
4. Instructions: "You will read 6 anonymised legal documents of approximately 1,000 words each. Rate each on the 5 dimensions provided. There are no right or wrong answers. Take as much time as you need. You may refer back to documents at any time."
5. Document order: randomised per participant (6! = 720 possible orders)
6. Each document displayed on its own page. Rating scale below. "Next" button advances.
7. Phase 1 completion screen: "Thank you. Please proceed to Phase 2."

### Phase 2: Post-Disclosure Re-Rating (~30 minutes)

1. Disclosure text displayed (see §3.4). Must remain on screen minimum 30 seconds.
2. Same 6 documents, same order as Phase 1 (order held constant within-subjects to isolate disclosure effect)
3. Previous ratings NOT displayed (forced independent re-rating)
4. Same 5 dimensions, same 7-point scale
5. Phase 2 completion screen: "Thank you. Please proceed to the final phase."

### Phase 3: Structured Interview (~30 minutes)

Conducted via video call (recorded with consent) or written response form. Questions:

1. "Before you learned about the author's background, what assumptions - if any - did you make about who wrote these documents?"
2. "After learning about the author's background, did your assessment of any document change? If so, which ones and in what way?"
3. **[If ratings changed]:** "What specifically about the disclosed background led you to revise your rating?"
4. **[If ratings did NOT change]:** "The disclosed background did not change your ratings. Why not?"
5. "In your professional experience, what adjustments - if any - would you consider reasonable for a litigant with the disclosed profile (autism, ADHD, executive-function difficulties, no legal training)?"
6. "Is there anything else you would like to share about this study or your experience of it?"

### Phase 4: Debrief (~10 minutes)

1. Full disclosure: "This study tests the Capability-Adjustment Fallacy - the hypothesis that the same legal work is rated differently when the evaluator knows the author is a neurodivergent litigant-in-person."
2. Reveal: Documents 1-6 are by 2 authors - 3 by the LiP described, 3 by practising counsel.
3. "All documents are from public court records. The LiP documents were filed in real proceedings. The counsel documents were extracted from published judgments."
4. "You may withdraw your data at this point. If you withdraw, your ratings and interview responses will be permanently deleted and excluded from analysis. Your compensation is unaffected."
5. "Do you consent to your anonymised data being used in this study? [Yes/No]"
6. "Do you consent to anonymised quotes from your interview being used in publication? [Yes/No]"
7. "Would you like to receive a copy of the study results? [Yes/No - if yes, collect email]"

---

## 6. ANALYSIS PLAN (PRE-REGISTERED)

### 6.1 Primary Analysis
```
Model: paired_samples = Phase2_overall_standard − Phase1_overall_standard
Test:  one-tailed paired t-test, H₁: mean(paired_samples) < 0
Data:  3 LiP documents × 10 participants = 30 observations
Effect size: Cohen's d_z (paired)
```

### 6.2 Secondary Analyses
```
H₂ (Specificity):
  counsel_samples = Phase2_overall_standard_counsel − Phase1_overall_standard_counsel
  Test: two-tailed paired t-test on counsel documents only
  Expected: non-significant (p > 0.05)

H₃ (Dimension specificity):
  Model: lmer(RatingChange ~ Dimension + (1|Participant) + (1|Document))
  Post-hoc: pairwise comparisons between dimensions
  Expected: Procedural > Overall > Clarity > Reasoning > Persuasiveness

H₄ (Qualitative):
  Thematic analysis of Phase 3 transcripts
  Coding: blind to participant ratings
  Themes: assumption-of-credentials, surprise-at-quality, procedural-inference,
          disability-adjustment-reasoning, explicit-fallacy
```

### 6.3 Exclusion Rules (pre-registered)
1. Participant withdraws data in Phase 4
2. Participant completes Phase 1 but not Phase 2 (partial data - excluded from primary analysis, Phase 1 data may be used for inter-rater reliability)
3. Participant gives identical ratings on all dimensions for all documents (suspected non-engagement) - flagged, not automatically excluded
4. Participant knew the author or recognised the documents - excluded, replaced

### 6.4 Robustness Checks
1. Inter-rater reliability: ICC(3,1) on Phase 1 blind ratings
2. Order effects: correlation between document position and rating
3. Demand characteristics: analyse whether participants who *didn't* change ratings gave different Phase 3 responses from those who did

---

## 7. BUDGET

| Item | Cost | Notes |
|------|------|-------|
| Participant fees (12 recruited × £200) | £2,400 | Over-recruit for dropouts |
| Qualtrics license (1 month) | £100 | Or free alternative (Gorilla, Google Forms) |
| Document preparation | £200 | Anonymisation, formatting, pilot testing |
| Video call recording + transcription | £150 | Otter.ai or manual |
| Statistical analysis software | £0 | R (open source) |
| Ethics review (if required) | £0-£500 | Institutional IRB may charge |
| Contingency | £250 | Unexpected costs |
| **Total** | **£3,100** | |

---

## 8. TIMELINE

| Week | Activity | Deliverable |
|------|----------|------------|
| **1** | Document selection + anonymisation. Pre-registration committed to OSF + GitHub. Ethics application submitted (if required). | Anonymised document set. Pre-registration DOI. |
| **2** | Platform build (Qualtrics). Pilot test with 2 non-lawyer participants (check timing, clarity). Recruitment begins. | Live survey. Pilot feedback incorporated. |
| **3** | Data collection (Phases 1-4). 12 participants booked across 5 days. Interviews recorded. | Complete dataset. Interview transcripts. |
| **4** | Analysis. Draft results section. Prepare manuscript for Paper C appendix or standalone submission. | Analysis report. Draft manuscript. |

---

## 9. OUTPUTS

1. **Results section** for Paper C Appendix or standalone brief report (~2,000 words)
2. **De-identified dataset** published on OSF (with participant consent)
3. **Analysis code** (R Markdown) published on GitHub - fully reproducible
4. **Pre-registration** - timestamped before any data collection

### Reporting Commitment
Results will be published regardless of outcome:
- If H₁ confirmed: "First experimental demonstration of the Capability-Adjustment Fallacy"
- If H₀ holds: "Blind evaluation found no evidence of the Capability-Adjustment Fallacy - n=1 case evidence stands but experimental support absent at this scale"
- This commitment is itself pre-registered

---

## 10. KILL-CONDITIONS

This experiment is designed to fail cleanly:

1. **If recruitment fails** (<8 participants after 4 weeks): Report as "infeasible at current design" with barriers documented
2. **If H₀ holds** (no rating shift): Publish the null result. The Fallacy lacks experimental support at this scale. The §9 case study evidence remains but the experimental claim is bounded.
3. **If H₂ fails** (rating shift affects counsel documents equally): The effect is not the Fallacy but a general "background information changes ratings" effect - interesting but not supporting Paper C. Report honestly.
4. **If inter-rater reliability is very low** (ICC < 0.3): The rating instrument may not be reliable. Report as methodological limitation.

---

## 11. ETHICS STATEMENT

- **Deception:** Participants are not initially told the study's purpose or the author's identity. This is minimal deception - all information disclosed in Phase 2 is true, and full debrief occurs in Phase 4.
- **Right to withdraw:** Data can be withdrawn at any point up to publication. Compensation unaffected.
- **Confidentiality:** All ratings and interview responses are anonymised. Participants identified by code only in published data.
- **Vulnerable participants:** None. All participants are qualified professionals.
- **Author's own documents:** All documents are from public court records. The author consents to their use. No confidentiality or privilege issues arise.
- **No legal advice:** This study does not provide or evaluate legal advice. It tests perceptual bias in professional evaluation.
- **Institutional review:** Ethics review from a recognised institutional review board is recommended before data collection. In the absence of institutional affiliation, the protocol will be submitted to an independent ethics consultant for review.

---

*Protocol version 1.0 · T3-research · 2026-07-06*
*Pre-registration: pending · Status: ready for funding + ethics*
