# Experiment 3: The Task-Asymmetry Protocol
## Measuring the Gap Between Complex Self-Directed and Routine Procedural Performance

**Paper C §5, §6, §7 support** · T3-research · 2026-07-06
**Status:** Protocol · Ready for funding + ethics

---

## 0. RATIONALE

Paper C's core empirical claim is that the same person can perform at radically different levels depending on task structure: high in complex, self-directed, high-interest domains; low in routine, externally-paced, low-stimulation procedural tasks. This is the asymmetry.

This experiment measures that gap directly, comparing neurodivergent adults with documented cross-domain output against matched controls.

The critical design feature: **both groups do the SAME tasks.** The prediction is not that the PNP group is better at everything - it's that their performance VARIANCE across task types is larger. The gap IS the finding.

---

## 1. HYPOTHESES (PRE-REGISTERED)

| # | Hypothesis | Prediction | Test |
|---|-----------|-----------|------|
| **H₁** | PNP-group participants show larger performance gap between complex-self-directed and routine-procedural tasks than controls | Group × Task interaction: PNP group higher variance | Mixed ANOVA: Group × Task interaction |
| **H₂** | PNP-group participants outperform controls on complex-self-directed tasks | Main effect of Group on Complex tasks | Independent t-test |
| **H₃** | PNP-group participants underperform controls on routine-procedural tasks | Main effect of Group on Routine tasks (reversed direction) | Independent t-test |
| **H₄** | Within the PNP group, monotropism score predicts the asymmetry gap | MQ × Task interaction within PNP group | Within-group regression |
| **H₅** | The asymmetry is domain-general - it appears across multiple task domains (legal, numerical, verbal, spatial) | Group × Task interaction replicates across domains | Meta-analytic combination across domains |

---

## 2. PARTICIPANTS

### 2.1 Groups

**PNP Group (n = 20):**
- Adults 18+ with self-reported ASC and/or ADHD diagnosis
- Documented output in ≥2 distinct domains (verified via CDOI from Experiment 2 or equivalent screening)
- Self-reported difficulty with routine procedural tasks

**Control Group (n = 20):**
- Adults 18+ matched on: age (±5 years), education level, estimated cognitive ability
- No neurodevelopmental diagnosis
- May have output in 0-1 domains (not excluded if 2+, but matched for domain count where possible)

**Total N = 40** · Between-subjects design with within-subject task manipulation

### 2.2 Matching

Each PNP participant matched to a control on:
1. Age (±5 years)
2. Education level (banded: secondary / undergraduate / postgraduate)
3. ICAR-16 cognitive ability score (±0.5 SD)
4. Employment status (employed / self-employed / not employed)

This isolates the neurodivergence + cross-domain-output variable from confounds.

### 2.3 Recruitment

- PNP group: recruited from Experiment 2 participants who scored in top quartile on CDOI domain count AND have diagnosis
- Control group: university participant pools, community recruitment, matched from Experiment 2 non-diagnosed participants
- Compensation: £30 each for ~3 hours = **£1,200** (40 × £30) + £100 travel = **£1,300**

---

## 3. TASK BATTERY

### 3.1 Complex Self-Directed Tasks (CSD)

These tasks are: open-ended, intrinsically interesting, self-paced, allow deep engagement, no time pressure.

| Task | Domain | Description | Duration | Measure |
|------|--------|-------------|----------|---------|
| **CSD-1: Legal Reasoning** | Legal | Analyse a novel legal scenario (fictional jurisdiction to control for prior knowledge). Identify relevant principles, construct an argument, anticipate counterarguments. No time limit. | 40 min | Quality score (blind-rated by 2 independent legal professionals on 1-7 scale) |
| **CSD-2: Pattern Discovery** | Numerical/Spatial | Given a dataset with a hidden structural pattern, identify the pattern and predict out-of-sample. Open-ended - no single correct answer. | 30 min | Pattern identification accuracy + prediction error |
| **CSD-3: Cross-Domain Analogy** | Verbal/Abstract | Read two passages from different domains (e.g., biology and economics). Identify the structural analogy. Explain the mapping. Generate a novel prediction in one domain based on the structure of the other. | 30 min | Analogy quality score (blind-rated on structural mapping accuracy, 1-7) |
| **CSD-4: Deep Research Mini-Task** | Self-Directed | "You have 45 minutes to research any topic you're curious about and produce a 500-word brief explaining what you learned and why it matters." Topic choice is free. | 45 min | Depth score (blind-rated), originality, synthesis quality |

### 3.2 Routine Procedural Tasks (RPT)

These tasks are: externally-structured, low-stimulation, time-pressured, require sequential procedural compliance, administratively demanding.

| Task | Domain | Description | Duration | Measure |
|------|--------|-------------|----------|---------|
| **RPT-1: Form Completion** | Administrative | Complete a complex multi-section form (modelled on a court N244 application notice) with specific procedural requirements: correct sections, correct fee calculation, correct service address, correct enclosures list. | 20 min | Accuracy score (% of required fields correctly completed) |
| **RPT-2: Email Triage** | Administrative | Process 15 emails requiring: sort by urgency, draft responses for 5, flag 3 for escalation, file 7. Time-pressured (25 min for all). | 25 min | Completion rate + accuracy of sorting + response quality |
| **RPT-3: Sequential Filing** | Procedural | Follow a 12-step filing protocol with specific ordering requirements. One error at step 3 cascades. Must track document versions. | 20 min | Error count + completion time |
| **RPT-4: Calendar Scheduling** | Time Management | Schedule 8 appointments across 5 people with availability constraints, room bookings, and 3 reschedules. Must avoid double-booking. "Urgent" interruptions every 5 minutes. | 20 min | Conflict count + completion rate |

---

## 4. PROCEDURE (~3 HOURS)

### Session Structure
```
0:00-0:15 - Consent + demographics + ICAR-16 + MQ
0:15-1:00 - CSD-1 (Legal Reasoning, 40 min)
1:00-1:30 - CSD-2 (Pattern Discovery, 30 min)
1:30-1:40 - Break
1:40-2:10 - CSD-3 (Cross-Domain Analogy, 30 min)
2:10-2:30 - RPT-1 (Form Completion, 20 min)
2:30-2:55 - RPT-2 (Email Triage, 25 min)
2:55-3:00 - Break
3:00-3:20 - RPT-3 (Sequential Filing, 20 min)
3:20-3:40 - RPT-4 (Calendar Scheduling, 20 min)
3:40-4:25 - CSD-4 (Deep Research Mini-Task, 45 min)
4:25-4:30 - Debrief + payment
```

**Order fixed:** CSD tasks first (to capture peak performance without fatigue), RPT tasks in the middle (to capture procedural performance under time pressure), CSD-4 last (to test whether deep engagement recovers after procedural drain - this IS the asymmetry).

**Critical methodological note:** CSD-4 is placed AFTER the procedural battery. The prediction: PNP-group participants recover into deep engagement for CSD-4 despite procedural drain; controls show flatter performance across the session. This within-session recovery IS evidence for the asymmetry.

---

## 5. BLIND RATING PROTOCOL

CSD-1, CSD-3, and CSD-4 outputs are blind-rated:

1. All outputs anonymised (participant code only)
2. Two independent raters per task (legal professionals for CSD-1, domain experts for CSD-3 and CSD-4)
3. Raters blind to: participant group, hypothesis, other task scores
4. Inter-rater reliability: ICC(2,1) computed; if < 0.6, third rater adjudicates
5. **Raters also blind to whether AI tools were used** - participants are permitted to use AI tools if they wish, and this is recorded as a covariate. (This directly addresses the user's concern: the experiment measures what the human CAN do, with or without tools, under different task structures.)

---

## 6. ANALYSIS

### Primary
```
H₁-H₃: Mixed ANOVA
  Within-subject: TaskType (CSD vs RPT)
  Between-subject: Group (PNP vs Control)
  DV: z-scored performance (pooled across tasks within type)
  Key test: Group × TaskType interaction

H₄: Within PNP group
  lm(AsymmetryGap ~ MQ_total + ICAR_score + age, data = PNP_only)
  AsymmetryGap = mean(z_CSD) − mean(z_RPT)

H₅: Meta-analytic
  Compute Group × TaskType interaction separately for each domain pair
  Combine using random-effects meta-analysis
```

### Power Analysis
- N = 40 (20 per group), mixed ANOVA with medium effect (f = 0.25): power ≈ 0.82 at α = 0.05
- H₁ (Group × Task interaction): adequately powered for medium-large effects
- H₄ (within-PNP regression, n = 20): underpowered for small effects; exploratory, not confirmatory

---

## 7. PREDICTED RESULTS

```
                    CSD Tasks              RPT Tasks
                 ┌─────────────────┐    ┌─────────────────┐
PNP Group        │  ████████████   │    │  ████           │
                 │  (high)         │    │  (low)          │
                 └─────────────────┘    └─────────────────┘
                          ↕ large gap
                 ┌─────────────────┐    ┌─────────────────┐
Control Group    │  ████████       │    │  ███████        │
                 │  (moderate)     │    │  (moderate)     │
                 └─────────────────┘    └─────────────────┘
                          ↕ small gap
```

The interaction is the finding. The PNP group doesn't outperform controls on everything - they outperform on complex-self-directed tasks and underperform on routine-procedural tasks. The gap between the two is the Capability-Adjustment Fallacy in measurable form.

---

## 8. BUDGET

| Item | Cost |
|------|------|
| Participant compensation (40 × £30) | £1,200 |
| Travel reimbursement | £100 |
| Blind raters (2 × CSD-1 legal, 2 × CSD-3 domain, 2 × CSD-4 general) - 6 raters × £100 | £600 |
| Task materials development | £300 |
| Venue/room hire (if in-person) or online platform | £200 |
| Data analysis | £300 |
| Contingency | £300 |
| **Total** | **£3,000** |

---

## 9. TIMELINE: 8 weeks

| Week | Activity |
|------|----------|
| 1-2 | Task development + pilot with 3 PNP and 3 control participants. Refine timing, difficulty, instructions. |
| 3 | Rater recruitment + training. Pre-registration. Ethics. |
| 4-5 | Data collection (5 participants per week, ~2 hours each + raters). |
| 6 | Blind rating completed. |
| 7 | Analysis. |
| 8 | Write-up. |

---

## 10. KILL-CONDITIONS

1. **Group × Task interaction non-significant** → asymmetry not detectable at n=40 → report honestly, note power limitations
2. **PNP group outperforms controls on BOTH task types** → the asymmetry claim is wrong - PNP confers general advantage, not specific asymmetry → this FALSIFIES the central Paper C claim
3. **PNP group underperforms controls on BOTH task types** → PNP is a general impairment profile, not an asymmetric one → also falsifies
4. **Blind rater ICC < 0.5** → rating instrument unreliable → findings qualified, instrument needs revision
5. **CSD-4 does NOT show recovery in PNP group** → the procedural-drain-then-deep-engagement pattern fails → bounded

---

## 11. THE AI COVARIATE

Every participant is asked: "Did you use any AI tools (e.g., ChatGPT, Claude) during any task? If yes, which tasks and how?"

This is recorded as a covariate, not a restriction. The experiment tests what humans CAN do - with whatever tools they normally use. The claim is about the *asymmetry between task types*, not about "unaided" performance. If a participant uses AI to help with a legal reasoning task, that's part of how they work. The question is whether the gap between CSD and RPT performance persists when tools are available.

**This directly addresses the user's point:** AI didn't write the papers - the user did, using AI as a tool. The experiment doesn't strip tools away. It measures the asymmetry in what the human can *produce*, with tools, under different task structures.

---

*Protocol v1.0 · T3-research · 2026-07-06*
