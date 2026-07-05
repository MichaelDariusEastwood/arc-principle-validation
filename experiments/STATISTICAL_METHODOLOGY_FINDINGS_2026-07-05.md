# T2-research statistical-methodology findings · 2026-07-05

**Lane:** T2-research · **Routed:** T5-research (verify) + author. Matches T5-dispatch flag: "Paper V Fisher p=6.3×10⁻²¹ — verify with 'pending blind replication' qualifier."

## Finding (MEDIUM) — Paper V Fisher-combined p=6.3×10⁻²¹ does not state the independence assumption

**Context.** Paper V (Stewardship Gene) reports stakeholder-care improvement across five models, each strong on its own:
Claude p=1.8×10⁻⁵ (d=0.94), DeepSeek p=9.8×10⁻⁵ (d=0.69), Gemini p=1.2×10⁻⁸ (d=1.14), Grok p=0.0105 (d=0.54),
Groq p=5.0×10⁻⁸ (d=1.07). It then reports **Fisher combination across the five → p≈6.3×10⁻²¹**.

**The issue.** Fisher's method assumes the combined p-values come from **independent** tests. The five model-runs use
"40 matched pairs" each and (by design) appear to share the same stakeholder-care scenarios and scoring instrument.
If the scenarios/scorers are shared, the five tests are **not fully independent** (a hard/ambiguous scenario perturbs
all models similarly → correlated errors), and the Fisher-combined p=6.3×10⁻²¹ **overstates** the combined
significance, possibly by many orders of magnitude. This is a classic hostile-reviewer attack on astronomically
small combined p-values.

**Why it is low-cost to fix — the claim does not need it.** The *individual* model p-values (10⁻⁸ to 10⁻⁵, with large
effect sizes d≈0.7–1.14) already establish "stakeholder care improves under the Eden intervention" robustly and
independently of the Fisher step. The Fisher p=6.3×10⁻²¹ is an unnecessary flourish that adds a vulnerability without
adding necessary evidence.

**Recommendation (flag — T5 verify, author decide):**
(a) Add one sentence: "Fisher combination assumes independence; the five model-runs share the scenario set and scoring,
    so this combined value is illustrative, not a formal joint significance"; OR
(b) De-emphasise the Fisher value and lead with the individual per-model results (which carry the claim); OR
(c) Use a method robust to dependence (e.g. Cauchy combination / a conservative bound) if a combined figure is wanted.
Also apply the T5-requested "pending blind replication" qualifier to the combined claim.

## Credit where due (LAW 1) — Paper V's statistics are otherwise disciplined
Reports effect sizes (d) alongside every p; corrected its own test (pilot Mann-Whitney U → paired t-test on the matched
design, p 0.016→0.0018); and explicitly NARROWS the claim ("the strongest defensible empirical claim is therefore
narrower than the original pilot framing: stakeholder care ... while the composite is architecture-dependent"). The
Fisher-independence caveat is the single gap in an otherwise-honest statistical treatment.

## Positive (verified SOUND) — Paper VII convergence p-value is exemplary
Paper VII's headline "19 of 25 domains confirm the Cauchy-predicted family, p=1.56×10⁻⁵" is done RIGHT, and stands in
instructive contrast to the Paper V Fisher issue:
- **Null fully stated + justified:** one-sided binomial test; under H0 that each domain's family is chosen uniformly at
  random from three families (chance=1/3), P(≥19 of 25) = 1.56×10⁻⁵.
- **Honest interpretation:** "~1 in 64,000, ~4.2σ" — not inflated.
- **Robustness rerun:** baseline-20 → 15/20 (p=1.67×10⁻⁴). Result holds on a different sample.
- **Acknowledges the misses:** "The Six Empirical Misses: six of 25 domains produced…".
- **Predict-before-fit discipline:** the composition operator was classified from known physics BEFORE fitting and the
  family PREDICTED from the operator class, then tested by independent AICc model selection. This is the direct defence
  against the look-elsewhere / post-hoc / cherry-picking attack on the convergence thesis.

**Net statistical picture:** the corpus CAN do combination-of-evidence rigorously (Paper VII proves it with a single
clean binomial test + stated null + robustness + predict-before-fit). Paper V's Fisher p=6.3×10⁻²¹ is the lone outlier —
a combined flourish that skips the independence justification Paper VII would never skip. Fixing/caveating Paper V's
Fisher step brings the whole corpus to Paper VII's standard.

## Finding (LOW/MODEST) — Data & Code Availability statement is not standardized across empirical papers

**Objective D (methodology, NOT book-blocked) + Objective A peer-review readiness.**

Every empirical paper references its data/code (OSF DOI 10.17605/OSF.IO/6C5XB + github.com/MichaelDariusEastwood/
arc-principle-validation) and carries methodological detail — so reproducibility is NOT absent. But a **dedicated,
standardized "Data & Code Availability" section** is present in only 3 papers (Paper II, VI, VII — they have an explicit
Reproducibility heading) and is distributed-not-consolidated in the other 7: **IV-a, IV-b, IV-c, IV-d, V, VIII,
On-the-Origin**. Most peer-reviewed venues now REQUIRE a standardized availability statement per article.

**Recommendation (flag — T5 + author; low provenance risk, adds a statement, changes no claim):** add a uniform
"Data & Code Availability" block to the 7 papers lacking one, following the II/VI/VII model — data location (OSF 6C5XB),
code repo, per-experiment parameters/n/seeds, and (per Objective G) the verify-yourself pointer (redacted .emls +
published hashes). This brings the whole corpus to journal-submission standard. Not urgent, but it is the concrete
remaining item under Objective D's methodology half while the book manuscript stays absent from this machine.

**Credit (LAW 1):** Paper V already has "3.5 Data Integrity" + "we report all data without exclusion"; Paper VIII has
"§8.4 Cross-Architecture Replication"; all papers carry the OSF DOI. The gap is standardization, not absence.
