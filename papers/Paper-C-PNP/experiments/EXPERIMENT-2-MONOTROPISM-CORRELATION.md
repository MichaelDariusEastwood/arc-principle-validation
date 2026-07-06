# Experiment 2: Monotropism-to-Polymathy Correlation Study
## Linking the Clinical Mechanism to Observable Cross-Domain Output

**Paper C §4, §6 support** · T3-research · 2026-07-06
**Status:** Protocol · Ready for funding + ethics

---

## 0. RATIONALE

Paper C §4 proposes monotropism — the attentional tendency to focus intensely on a narrow interest tunnel — as the mechanism producing PNP component 4 (hyperfocus) and component 3 (autonomous research intensity). §6 defines PNP as a cluster of six features, with components 1 (cross-domain depth) and 5 (executive asymmetry) being measurable.

This experiment tests the mechanism claim directly:
> **Does monotropism score predict: (a) number of domains with documented output, (b) depth of output per domain, and (c) the gap between high-complexity and routine-procedural performance?**

If the mechanism is real, high-monotropism individuals should show MORE cross-domain output AND MORE procedural difficulty — the asymmetry itself, in a single dataset.

---

## 1. HYPOTHESES (PRE-REGISTERED)

| # | Hypothesis | Prediction | Test |
|---|-----------|-----------|------|
| **H₁** | Monotropism score positively predicts number of domains with documented output | β > 0, p < 0.05 | Linear regression: domains ~ MQ_score + covariates |
| **H₂** | Monotropism score positively predicts self-reported procedural difficulty | β > 0, p < 0.05 | Linear regression: proc_diff ~ MQ_score + covariates |
| **H₃** | The asymmetry gap (domain output − procedural function) increases with monotropism | Interaction effect | MQ_score × domain_type interaction in mixed model |
| **H₄** | ADHD+ASC combined diagnosis predicts higher monotropism than either alone | ASC+ADHD > ASC > ADHD > neither | One-way ANOVA |
| **H₅** | The monotropism→polymathy relationship is mediated by hyperfocus intensity, not by general intelligence | Mediation model: MQ → hyperfocus → domains, controlling for IQ proxy | Mediation analysis (bootstrapped) |

---

## 2. DESIGN: CROSS-SECTIONAL SURVEY WITH VALIDATED INSTRUMENTS

### 2.1 Instruments

| Construct | Instrument | Items | Time | Validation |
|-----------|-----------|-------|------|------------|
| **Monotropism** | Monotropism Questionnaire (MQ) — Garau et al. (2023) | 47 items (Likert) | ~15 min | Validated in autistic + ADHD populations; preprint, cite as preliminary |
| **Hyperfocus** | Adult Hyperfocus Questionnaire (AHQ) — Hupfeld et al. (2019) or adapted | 12 items | ~5 min | Published in ADHD populations |
| **Cross-domain output** | Custom Cross-Domain Output Inventory (CDOI) | ~20 items | ~10 min | Novel instrument (validated within this study against documented output) |
| **Executive function** | Adult ADHD Self-Report Scale (ASRS v1.1) — WHO | 6 items (screener) | ~3 min | Validated globally |
| **Procedural difficulty** | Custom Procedural Functioning Scale (PFS) | ~15 items | ~5 min | Novel; items drawn from NICE CG142 adaptive-functioning domains |
| **Cognitive ability proxy** | ICAR-16 (International Cognitive Ability Resource) | 16 items | ~15 min | Validated, public domain |
| **Demographics** | Age, gender, diagnosis status (self-reported: ASC, ADHD, both, neither, pursuing assessment), education level, employment status, country | 10 items | ~2 min | Standard |

**Total survey time:** ~55 minutes. Compensation: £15 Amazon voucher.

### 2.2 Cross-Domain Output Inventory (CDOI) — Novel Instrument

The CDOI asks participants to list domains in which they have produced *documented, verifiable output* in the past 5 years. For each domain:

| Question | Scale |
|----------|-------|
| How many distinct projects/outputs? | Count (0-50+) |
| What is your deepest level of expertise? | 1 (dabbled) → 7 (published / professional) |
| Is this domain part of your paid employment? | Yes/No |
| How many hours/week do you spend on this domain? | Continuous |
| Do you have formal credentials in this domain? | None / Certificate / Degree / Professional |
| Can you provide evidence of output? | None / Self-report / Portfolio / Publication / Public record |

The CDOI is validated within the study by:
1. Asking participants to upload or link to ONE documented output (optional, with consent)
2. Comparing self-reported depth against a subset of verified outputs
3. Computing a "self-report inflation index" = (self-rated depth − verified depth) / verified depth

### 2.3 Sampling Strategy

**Target N = 200** (powered for medium effects)

Recruitment channels:
- Social media (autism/ADHD community groups, research participation forums)
- University participant pools (if institutional partnership secured)
- Professional networks (LinkedIn, relevant subreddits)
- Clinical networks (with ethics board approval)

Inclusion: Adults 18+, fluent English, any diagnosis status
Exclusion: Cannot complete online survey (accommodations offered: extended time, text-only version, screen-reader-compatible)

**Quota sampling** to ensure adequate representation:
- ASC-diagnosed: ≥50
- ADHD-diagnosed: ≥50
- ASC+ADHD (combined): ≥50
- Neither diagnosis: ≥50

---

## 3. ANALYSIS PLAN (PRE-REGISTERED)

### 3.1 Primary
```
H₁: lm(domain_count ~ MQ_total + age + education + ICAR_score, data = df)
H₂: lm(PFS_total ~ MQ_total + age + diagnosis, data = df)
```

### 3.2 Asymmetry Gap (H₃)
```
Long-format: each participant has 2 rows — "domain_output_z" and "procedural_function_z"
Model: lmer(z_score ~ domain_type * MQ_total + (1|participant), data = df_long)
Key term: domain_type:MQ_total interaction
```

### 3.3 Mediation (H₅)
```
Model: MQ_total → AHQ_hyperfocus → domain_count
Bootstrapped indirect effect (10,000 resamples)
Covariates: ICAR_score, age, education
```

### 3.4 Power Analysis
- N = 200, 4-group comparison (H₄): detects f ≥ 0.22 (medium) at α = 0.05, β = 0.20
- N = 200, regression with 4 predictors (H₁): detects f² ≥ 0.06 (small-medium) at α = 0.05, β = 0.20
- Adequately powered for all primary hypotheses

---

## 4. PREDICTED RESULTS

| Hypothesis | Predicted outcome | Effect size estimate |
|-----------|------------------|---------------------|
| H₁: MQ → domains | β ≈ 0.25-0.35 | Medium |
| H₂: MQ → procedural difficulty | β ≈ 0.20-0.30 | Small-medium |
| H₃: Asymmetry gap | Significant interaction, MQ × domain_type | Medium |
| H₄: ASC+ADHD highest MQ | η² ≈ 0.08-0.12 | Medium |
| H₅: Mediation via hyperfocus | Indirect effect significant, direct effect reduced | — |

### What success looks like
A single dataset showing: high monotropism → more domains of output AND more procedural difficulty → the asymmetry is quantifiable and mechanistically grounded. This would be the first study directly linking monotropism to polymathic output.

### What failure looks like
No correlation between MQ and domain count → the mechanism claim lacks quantitative support → Paper C §4's monotropism-to-polymathy link is bounded as "clinical observation, not demonstrated mechanism."

---

## 5. BUDGET

| Item | Cost |
|------|------|
| Participant compensation (200 × £15) | £3,000 |
| Survey platform (Qualtrics, 2 months) | £200 |
| ICAR-16 license (public domain) | £0 |
| MQ license (contact authors; likely free for academic use) | £0 |
| Data analysis (R, open source) | £0 |
| Research assistant (participant screening, data cleaning) | £1,000 |
| Contingency | £300 |
| **Total** | **£4,500** |

---

## 6. TIMELINE: 3 weeks

| Week | Activity |
|------|----------|
| 1 | Ethics application. Survey build. Pilot test (n=10). Instrument refinement. Pre-registration. |
| 2 | Recruitment + data collection (target 200 completions). |
| 3 | Data cleaning + analysis + write-up. |

---

## 7. KILL-CONDITIONS

1. **MQ fails to predict domain count** (H₁ β not significantly > 0) → mechanism claim bounded
2. **CDOI self-report inflation index > 0.3** (participants systematically over-report) → instrument unreliable, findings qualified
3. **Response rate < 150** (underpowered for H₄) → report as preliminary, seek additional funding
4. **ASC+ADHD group MQ scores NOT higher than single-diagnosis groups** → §4 mechanism claim about combined-profile respondents (Garau et al., 2023) requires re-examination

---

## 8. ETHICS

- **Diagnosis disclosure:** Self-report only. Participants not required to disclose diagnosis. "Prefer not to say" option throughout.
- **Output evidence:** Uploading documented output is entirely optional. Participants can complete the full survey without providing any evidence.
- **Vulnerable population:** Some participants will be autistic/ADHD adults. Survey designed with: clear language (no jargon unless defined), progress bar, save-and-return, extended time allowances, screen-reader compatibility, optional rest breaks.
- **Data protection:** All data anonymised at collection. IP addresses not stored. Evidence uploads stored in encrypted bucket, deleted after verification.
- **Right to withdraw:** Data can be withdrawn up to 2 weeks after participation (unique anonymous code provided for withdrawal).

---

## 9. OUTPUTS

1. **Preprint** (~4,000 words): "Monotropism as a Predictor of Cross-Domain Output: Evidence from a Cross-Sectional Study of 200 Adults"
2. **De-identified dataset** on OSF
3. **Analysis code** (R Markdown) on GitHub
4. **CDOI validation report** — the novel instrument's psychometric properties
5. **Paper C Appendix B:** Summary of findings, integrated into Paper C's mechanism section (§4, §6)

---

*Protocol v1.0 · T3-research · 2026-07-06*
