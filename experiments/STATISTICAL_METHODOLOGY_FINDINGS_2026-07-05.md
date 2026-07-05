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
