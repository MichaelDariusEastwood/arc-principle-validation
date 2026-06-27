# Prior-Art & Novelty Audit — the Cauchy Cluster (Paper VII, with Foundational & On the Origin of Scaling Laws)

**Method.** Four independent, web-enabled prior-art searches were run in parallel across the
distinct precedent clusters that bear on the cluster's central bridge — (1) functional equations /
invariance fixing the admissible form of a law; (2) the origin-of-power-laws / mechanism literature;
(3) the allometric-scaling unification literature; (4) a direct-hit search for the exact thesis. Each
returned closest precedents with citations and an independent novelty verdict. This is the same
adversarial treatment Paper X received (`Paper-X/results/redteam.md`).

**The thesis under audit (the "bridge").** *Cauchy's four functional equations are a single unifying
constraint that limits cross-domain scaling laws to three families — power, exponential, saturation —
with the system's **composition operator** selecting which family appears; and this Cauchy constraint
is the unifying reason that independent allometric derivations (West–Brown–Enquist, Banavar, Demetrius)
converge on the same d/(d+1) exponent.*

---

## Verdict: **PARTIALLY-ANTICIPATED**

The bridge is **not** wholesale novel and it is **not** un-original. It is a **novel synthesis and
empirical stress-test built on a well-developed prior tradition**, plus **one genuinely new sub-claim**.
All four searches reached this verdict independently. No source was found making the exact compound
thesis; but the load-bearing ideas each have specific, strong precedents — several of which the paper
does not cite.

| # | Component claim | Status | Closest precedent |
|---|---|---|---|
| C1 | Cauchy's four equations have unique continuous solutions (linear/power/exp/log) | **Anticipated** (textbook; author concedes) | Aczél, *Lectures on Functional Equations* (1966) |
| C2 | A functional-equation / invariance constraint fixes the *admissible functional form* of a law, before data, with no free parameters | **Anticipated** | **Luce (1959), "On the possible psychophysical laws"**; Aczél–Luce–Ng, "On the possible scientific laws" (1990); **S. A. Frank (2009, 2016)** invariance/max-entropy programme |
| C3 | The *composition operator* selects the form: multiplicative→power, additive→exponential, bounded→saturation | **Anticipated** | **Biró–Barnaföldi (2008)** composition-rule / formal-logarithm (additive→Boltzmann-Gibbs, non-additive→Tsallis power law); Mitzenmacher (2004); group entropies / formal group laws (Tempesta 2011); entropy trichotomy (arXiv:nlin/0408007) |
| C4 | Scale-invariance ⇒ power law as the cross-domain reason power laws are ubiquitous | **Anticipated** (textbook) | Sornette, *Critical Phenomena in Natural Sciences* (2000); Barenblatt (1996); regular variation (Bingham–Goldie–Teugels 1987) |
| C5 | **Cauchy's multiplicative equation is the *meta-reason* WBE / Banavar / Demetrius converge on d/(d+1)** | **NOVEL** (no source fuses these via a named functional equation) | Banavar et al. (2010) unify the *network* routes geometrically but exclude Demetrius and do not invoke Cauchy |
| C6 | The explicit four-Cauchy-equation taxonomy as a single cross-domain **predictive protocol**, empirically tested on 25/50 domains (AICc) | **Novel as a protocol/dataset** (weakened by author-classified operators; not pre-registered — see §12 of the paper) | No precedent for the test design; the *idea* of a cross-domain form-selection principle is Frank's |

---

## What is genuinely yours (the defensible residue)

1. **C5 — the Cauchy unification of the allometric derivations.** No located source claims that
   Cauchy's multiplicative functional equation is *the* object explaining why West's fractal-network
   route and Demetrius's quantum-oscillator route land on the same d/(d+1). Banavar (2010) unifies the
   network family geometrically but neither includes the quantum route nor names a functional equation.
   This sub-claim is the most defensibly original thing in the cluster — **provided** it is framed as
   doing explanatory work the geometric framing does not (i.e. that Cauchy multiplicativity + conservation
   in d-space is what forces the *same value* across a network model and a non-network model).
2. **The four-equation Cauchy taxonomy as one cross-domain predictive tool**, and **the 25/50-domain
   structured AICc test** (C6). These are contributions of *framing, synthesis and empirical reach* — real,
   but they sit on top of an old idea, not beside a vacuum.

## What is NOT yours (anticipated — must be conceded, not claimed as first)

- That a functional-equation / invariance constraint *fixes the admissible form of a law across domains,
  before data* is the explicit content of **Luce (1959)** (power/log/linear from scale-type invariance,
  with the physics/dimensional-analysis connection named) and the **Steven Frank invariance/maximum-entropy
  programme (2009–2016)** ("one principle selects the form across unrelated fields": mean→exponential,
  geometric-mean→power, etc.).
- That the **composition operator** selects power-vs-exponential is the **Biró–Barnaföldi / Hanel–Thurner–
  Gell-Mann composition-rule** result (additive→exponential, non-additive/multiplicative→power) and the
  **group-entropy / formal-group-law** literature; the multiplicative-vs-additive determinant of form is
  also **Mitzenmacher (2004)**.
- Scale-invariance ⇒ power law is **textbook** (Sornette; Barenblatt; regular variation).
- The d/(d+1) exponent and ¾/⅔ are **WBE (1997), Banavar (1999/2010), Demetrius (2006)** — the paper
  already cites these.

---

## Citations the paper currently MISSES and must add (priority order)

1. **R. D. Luce (1959), "On the possible psychophysical laws," *Psychological Review* 66:81–95** —
   **MANDATORY.** The single closest antecedent to C2; currently uncited. Omitting it while asserting
   "no prior work invokes [this principle]" is the audit's most serious exposure.
2. **T. S. Biró & G. G. Barnaföldi (2008) and the Hanel–Thurner–Gell-Mann composition-rule / group-entropy
   line (incl. Tempesta 2011)** — the closest antecedent to C3.
3. **S. A. Frank (2009) "The common patterns of nature," *J. Evol. Biol.* 22:1563; Frank (2016)
   "The invariances of power law size distributions"** — the nearest cross-domain "one principle selects
   the form" competitor.
4. **D. Sornette, *Critical Phenomena in Natural Sciences* (2000), ch. 14** and **M. Mitzenmacher (2004),
   "A brief history of generative models…"** — for C3/C4.
5. **Aczél, Luce & Ng (1990) "On the possible scientific laws"** and **Aczél & Dhombres (1989)** — these
   directly rebut the "no cross-domain unifying use" assertion.

## Required re-scoping of the §11 novelty assertion

The paper's §11 currently states: *"an exhaustive prior-art investigation found no previous work that
invokes [Cauchy's functional equations] as a unifying cross-domain principle constraining the forms of
scaling laws."* **This assertion does not survive** — Frank's invariance programme and the
composition-rule / group-entropy literature do essentially this (with a different selector). Re-scope to
the precise, defensible claim:

> *Prior work establishes that an invariance or composition constraint fixes the admissible functional
> form of a law (Luce 1959; Aczél; Frank 2009/2016), and that composition rules select power-vs-exponential
> forms (Biró–Barnaföldi 2008; group entropies). The contribution here is (i) casting the trichotomy
> explicitly through Cauchy's four classical equations as a single before-the-fit cross-domain predictor,
> (ii) the claim that Cauchy multiplicativity is the meta-reason the independent allometric derivations
> converge, and (iii) a 25/50-domain structured empirical test.*

This narrows the claim to exactly what is yours — and, done, the contribution stands as a legitimate
**synthesis-plus-one-new-sub-claim**, not an overclaimed "first."

## Methodological caveats (the paper already concedes these — keep them prominent)

- **Not pre-registered.** Paper VII §12 states plainly: *"No pre-registration artefact… not a formally
  pre-registered blind trial."* (This audit corrects an earlier informal description of VII as
  "preregistered" — it is an honest *structured prediction comparison*, not a blind pre-registered trial.)
- **Operator classification by the author**, not an independent panel — the central confound (the author
  picks the operator, then "predicts" the form).
- **Bounded-composition exhaustiveness is unproven** — the saturation leg is empirical convention, not theorem.
- **Family prediction only**, not exponent; six empirical misses; publication-bias risk on the 5 expansion domains.

## Bottom line

The Cauchy cluster is **partially anticipated**: its conceptual programme (a single
functional-equation/invariance/composition principle that selects the scaling form across domains) has
mature, specific precedents the paper must cite (Luce 1959; Frank 2009/2016; Biró–Barnaföldi 2008;
Sornette; regular variation), and its mathematics is textbook. What is **defensibly yours** is the
explicit Cauchy-four-equation cross-domain *predictive* taxonomy, the **novel** sub-claim that Cauchy
multiplicativity is the meta-reason the WBE/Banavar/Demetrius derivations converge, and the structured
25/50-domain test. The honest framing — which the paper's abstract already half-makes — is *"not the
maths, and not the first to use invariance to fix form, but the first to (a) unify these specific
derivations via Cauchy and (b) test the operator→form taxonomy at cross-domain scale."* Add the missing
citations, soften the §11 universal assertion, and the cluster is well-positioned rather than exposed.

---

*Generated by a 4-front parallel prior-art investigation (functional equations · power-law mechanisms ·
allometric unification · direct-hit search), web-enabled, June 2026. Verdicts were reached independently
per front and converged on PARTIALLY-ANTICIPATED. Full source lists are in each front's findings.*
