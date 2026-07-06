# Experiment 4: Cross-Domain Structural Pattern Transfer Test
## Testing PNP Component 2 — Recursive Pattern Transfer Between Fields

**Paper C §6 component 2, §6A, §11 support** · T3-research · 2026-07-06
**Status:** Protocol · Ready for funding + ethics

---

## 0. RATIONALE

Paper C §6 defines PNP component 2 as: "Extracting relational structure from one domain and using it to accelerate learning or reasoning in another." §6A describes the mechanism as "recursive questioning applied over decades to many domains accumulates by ordinary compounding" and states the author "was not born with cross-domain competence. The competence was ingrained, trained, and built up over decades."

This is the hardest PNP component to evidence observationally — self-report is unreliable, and credentials measure institutional access, not cognition. A controlled experiment is the only way to test whether cross-domain structural transfer is a real, measurable cognitive ability that varies between individuals.

### The Claim Being Tested
> Some individuals are systematically better than others at: (a) extracting the deep relational structure from a problem in one domain, (b) recognising that same structure when it appears in a different domain, and (c) using the mapped structure to solve a novel problem faster than someone encountering it fresh.

This is **not** about general intelligence. It's about a specific skill: structural analogy-making across domains. If it's real, it should be measurable independently of IQ.

---

## 1. HYPOTHESES (PRE-REGISTERED)

| # | Hypothesis | Prediction | Test |
|---|-----------|-----------|------|
| **H₁** | Self-identified polymaths (≥2 domains of documented output) outperform domain specialists on cross-domain transfer problems | Main effect of Group on transfer score | One-way ANOVA |
| **H₂** | The polymath advantage is specific to TRANSFER — not general problem-solving | Group × ProblemType interaction: polymaths better on transfer, not on within-domain novel problems | Mixed ANOVA |
| **H₃** | Cross-domain transfer ability is predicted by analogical reasoning (separate from IQ) | AMT score predicts transfer, controlling for ICAR | Hierarchical regression |
| **H₄** | The advantage persists when controlling for: IQ, education, age, domain familiarity | Group effect remains significant with covariates | ANCOVA |
| **H₅** | Within the polymath group, number of output domains predicts transfer performance (dose-response) | β > 0 for domain_count → transfer_score | Linear regression |

---

## 2. DESIGN: 3-GROUP BETWEEN-SUBJECTS

### 2.1 Groups

| Group | N | Definition |
|-------|---|-----------|
| **Polymath (PM)** | 20 | Documented output in ≥2 distinct domains (verified), no formal credentials in at least 1 of those domains |
| **Specialist (SP)** | 20 | Documented output in 1 domain, formal credentials in that domain, no output outside it |
| **General Population (GP)** | 20 | Any output in 0-1 domains, not self-identifying as polymath or specialist |

**Total N = 60**

### 2.2 Matching
Groups matched on: age band, education level, ICAR-16 cognitive ability score.
This is essential — the polymath group may have higher IQ; we need to control for this to isolate transfer ability.

---

## 3. THE TASK: STRUCTURAL ANALOGY TRANSFER BATTERY

### 3.1 Design

**3 problem structures × 3 domain pairs = 9 transfer problems**

Each problem follows the same format:
1. **Source problem** in Domain A (with full solution provided — tests extraction, not solving)
2. **Distractor** (unrelated problem in Domain B — filler)
3. **Target problem** in Domain B (structurally isomorphic to source — tests TRANSFER)

### 3.2 Problem Structures

| Structure | Description | Source Domain | Target Domain |
|-----------|-------------|---------------|---------------|
| **Convergence-to-threshold** | A quantity approaches but never exceeds a limit; the approach rate is the key variable | Physics (terminal velocity) | Economics (diminishing returns) |
| **Nested-recursion** | A process that calls itself with modified parameters; depth determines output quality | Computer Science (tree traversal) | Biology (gene regulation cascades) |
| **Correction-outscaling-drift** | Stability requires that error-correction strengthens faster than error-generation; the ratio of exponents determines outcome | Engineering (feedback control systems) | Ecology (predator-prey equilibrium) |

### 3.3 Control: Within-Domain Novel Problems

3 additional novel problems within a SINGLE domain (no transfer required). These control for general problem-solving ability.

### 3.4 Dependent Measures

For each of the 9 transfer problems:
- **Transfer score (0-10):** Did the participant apply the source structure to the target problem? Scored by blind raters using a rubric.
- **Recognition (0/1):** Did the participant explicitly note the structural similarity?
- **Solution time (seconds):** Time from starting target problem to submitting answer.
- **Solution quality (0-10):** Blind-rated quality of the target problem solution, independent of whether transfer was used.

---

## 4. THE ANALOGICAL MAPPING TEST (AMT)

All participants also complete a 20-item Analogical Mapping Test — a validated measure of relational reasoning that requires mapping structures between domains. Examples:

- "A fortress is to a siege as an immune system is to _____" (infection)
- "Thermodynamic entropy is to disorder as Shannon entropy is to _____" (uncertainty)
- "A keystone species is to an ecosystem as a _____ is to an argument" (premise)

The AMT score serves as the mechanism measure (H₃): if cross-domain transfer is driven by analogical reasoning ability, AMT should predict transfer performance beyond IQ.

Scored 0-20. Time limit: 15 minutes.

---

## 5. PROCEDURE (~2.5 HOURS)

```
0:00-0:15 — Consent, demographics, ICAR-16
0:15-0:30 — Analogical Mapping Test (AMT, 15 min)
0:30-0:50 — Source problems (3 problems, full solutions provided, read-only — testing extraction)
0:50-0:55 — Break
0:55-1:30 — Block 1: 3 transfer problems (Sources 1-3 → Targets)
1:30-1:35 — Break
1:35-2:10 — Block 2: 3 within-domain novel problems (control — no transfer)
2:10-2:15 — Break
2:15-2:50 — Block 3: 3 more transfer problems (counterbalanced domain pairs)
2:50-2:55 — Debrief + payment
```

**Counterbalancing:** Domain pairs rotated across participants (Latin square) so each source-target pairing appears equally often in each block position.

---

## 6. ANALYSIS

### 6.1 Primary
```
H₁-H₂: Mixed ANOVA
  Within: ProblemType (transfer vs within-domain)
  Between: Group (PM vs SP vs GP)
  DV: Solution quality (blind-rated, 0-10)
  Key: Group × ProblemType interaction

Post-hoc: PM > SP = GP on transfer; PM = SP = GP on within-domain
```

### 6.2 Mechanism (H₃)
```
Model 1: lm(TransferScore ~ AMT + ICAR + age + education)
Model 2: lm(TransferScore ~ AMT + ICAR + age + education + Group)
ΔR² from Model 1 to Model 2 tests whether Group adds beyond mechanism
```

### 6.3 Dose-Response (H₅)
```
Within PM group only:
  lm(TransferScore ~ domain_count + AMT + ICAR, data = PM_only)
```

### 6.4 Power Analysis
- N = 60 (20 per group), medium effect (f = 0.25), 3-group ANOVA: power ≈ 0.75
- For the interaction: medium effect (f = 0.20), power ≈ 0.65
- Adequately powered for primary effects, marginally powered for interaction — note in limitations

---

## 7. PREDICTED RESULTS

### Predicted Pattern
```
                Transfer Problems        Within-Domain Problems
              ┌──────────────────┐      ┌──────────────────┐
Polymath      │  ██████████████  │      │  ██████████      │
              │  (high)          │      │  (comparable)    │
              └──────────────────┘      └──────────────────┘

Specialist    │  ██████          │      │  ██████████████  │
              │  (moderate-low)  │      │  (high in domain)│
              └──────────────────┘      └──────────────────┘

General Pop   │  ██████          │      │  ████████        │
              │  (moderate-low)  │      │  (moderate)      │
              └──────────────────┘      └──────────────────┘
```

### What This Would Mean

If the polymath group outperforms on transfer but NOT on within-domain problems (where specialists excel), we have evidence for PNP component 2 as a specific, measurable cognitive ability — not "general smartness."

If the polymath group outperforms on EVERYTHING, it's IQ, not transfer — component 2 is not a distinct ability.

If there are no group differences, transfer ability is not measurable at n=60 — or doesn't exist as a distinct trait.

---

## 8. BLIND RATING

All 540 solutions (60 participants × 9 problems) are blind-rated:

1. Solutions anonymised, shuffled across participants and groups
2. Two independent raters per problem structure
3. Raters trained on a scoring rubric with anchor examples
4. Raters blind to: participant group, hypothesis, whether transfer was used
5. ICC(2,1) ≥ 0.7 required; if not, third rater adjudicates

**Rater cost:** 3 problem structures × 2 raters × £100 = **£600**

---

## 9. THE AUTHOR'S THEORY PREDICTS...

The author's own account (§6A) claims that cross-domain competence is TRAINED, not innate — "the competence was ingrained, trained, and built up over decades of the loop running."

If this is correct:
1. **Domain count should predict transfer** (H₅) — more domains of output = more practice at structural extraction = better transfer. This is a dose-response prediction.
2. **AMT should mediate** (H₃) — the mechanism is analogical reasoning, which is trainable. If AMT fully mediates the group effect, the polymath advantage is explained by practiced analogical reasoning, not by innate gift.
3. **Specialists should NOT show the transfer advantage** — because they've practiced within one domain, not across domains. Specialism doesn't train transfer.

### The Killer Test

If the dose-response (H₅) holds, the author's "trained recursion, not innate gift" claim (§6A) gains experimental support. A person with 5 domains of output should show better transfer than a person with 2, controlling for IQ and age.

If it doesn't hold — if domain count doesn't predict transfer — then cross-domain transfer may be an innate ability rather than a trained one. That wouldn't falsify PNP (the profile describes the pattern regardless of mechanism), but it would challenge the humility mechanism in §6A.

---

## 10. BUDGET

| Item | Cost |
|------|------|
| Participant compensation (60 × £25) | £1,500 |
| Blind raters (6 × £100) | £600 |
| Task materials development | £400 |
| Platform + venue | £300 |
| Analysis | £400 |
| Contingency | £300 |
| **Total** | **£3,500** |

---

## 11. TIMELINE: 6 weeks

| Week | Activity |
|------|----------|
| 1-2 | Problem development. Pilot test (n=5: 2 PM, 2 SP, 1 GP). Refine difficulty, timing, instructions. |
| 3 | Rater recruitment, rubric development, training. Pre-registration. |
| 4-5 | Data collection (10 participants per week). |
| 5-6 | Blind rating. Analysis. Write-up. |

---

## 12. KILL-CONDITIONS

1. **No group difference on transfer** → component 2 not measurable at n=60 → report honestly
2. **Polymath group outperforms on ALL problems** (transfer AND within-domain) → it's IQ, not transfer → component 2 as a distinct ability is NOT supported
3. **Dose-response absent** (domain count doesn't predict transfer) → "trained recursion" mechanism (§6A) challenged
4. **AMT fully explains group differences** (no residual Group effect in Model 2) → transfer ability IS analogical reasoning, which IS measurable — but it doesn't require PNP to explain it
5. **Blind rater ICC < 0.6** → scoring unreliable → findings qualified

---

## 13. CONNECTION TO PAPER C

| Paper C Section | This Experiment Tests |
|-----------------|----------------------|
| §6 component 2 | Is cross-domain pattern transfer a measurable, trainable ability? |
| §6A mechanism | Is it trained (dose-response by domain count) or innate? |
| §11 monoculture risk | Do polymaths detect structural patterns that specialists miss? |
| §12 falsifiability | Kill-condition 2 is a genuine falsification threat |

---

*Protocol v1.0 · T3-research · 2026-07-06*
