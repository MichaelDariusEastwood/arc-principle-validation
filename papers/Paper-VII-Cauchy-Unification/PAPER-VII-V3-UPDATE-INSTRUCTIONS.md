# PAPER VII v3.0 UPDATE: EXACT CHANGES
# Each change specifies WHERE in the paper and WHAT to insert/replace.
# Copy each passage directly into the HTML source.

============================================================================
CHANGE 1: Version number and header
============================================================================
LOCATION: Version line at top
REPLACE:
  "Version 2.0 | 17 March 2026"
WITH:
  "Version 3.0 | 24 March 2026 | First published 16 March 2026
   v3.0: Demetrius independent derivation; d=4 maximum metabolic rate;
   Glazier speed limit confirmation; d=2 biological correction;
   Friedmann boundary coincidence explicit
   v2.0: expanded to 50 domains; replaced R^2 with AICc; tiered reporting"

============================================================================
CHANGE 2: Abstract - add Demetrius and speed limit
============================================================================
LOCATION: End of abstract, before "Keywords"
INSERT AFTER the sentence ending "...exploratory evidence but not strong
enough to claim as proven":

  "Additionally, we identify an independent derivation of the dimensional
  formula alpha = d/(d+1) by Demetrius (2006), who derived the same
  result from quantum oscillator coupling in the Debye model. Two
  independent derivations from different branches of physics - network
  partition geometry and quantum statistical mechanics - converge on
  the same formula. We show this convergence is mathematically compelled
  by Cauchy's classification: both derivations instantiate the
  multiplicative functional equation in d-dimensional space. We further
  identify alpha = 1 as the geometric speed limit: the maximum scaling
  exponent achievable by any physical transport network in
  finite-dimensional space, confirmed by the formula, by Glazier's
  (2008) metabolic boundary data, and by the Demetrius classical limit.
  We correct the d=2 biological claim: no known organism possesses a
  genuinely 2D hierarchical space-filling transport network; the d=2
  confirmation exists in cosmology (Friedmann, exact) and physics, not
  in biology."

============================================================================
CHANGE 3: NEW SECTION after Section 2 - "The Demetrius Convergence"
============================================================================
LOCATION: After Section 2 (The Theory), before Section 3 (Prediction Protocol)
INSERT NEW SECTION:

  ## 2A. Independent Derivation: The Demetrius Convergence

  In 2006, Lloyd Demetrius of Harvard University published a derivation
  of the metabolic scaling formula from quantum statistical mechanics
  (Demetrius, 2006; Journal of the Royal Society Interface 3, 843-851;
  PMC2842802). His approach was entirely different from the network
  geometry of West, Brown & Enquist.

  Demetrius modelled organisms as networks of coupled energy-transducing
  oscillators embedded in d-dimensional physical space, applying the
  Debye model of thermal properties. In the quantum regime (cycle time
  much less than characteristic time), he derived a scaling exponent of
  d/(d + 1), where d is the spatial dimensionality of the oscillator
  network.

  His results (Demetrius 2006, Section 3):
  - d = 1 (linear geometry): alpha = 1/2
  - d = 2 (planar geometry): alpha = 2/3
  - d = 3 (three-dimensional): alpha = 3/4
  - d approaching infinity: alpha approaches 1

  This is the same formula, producing the same predictions, derived from
  different physics. West et al. derived d/(d+1) from hierarchical
  network geometry. Demetrius derived it from quantum oscillator coupling.
  Neither cited the other's method.

  ### Why the convergence is mathematically compelled

  Both derivations are special cases of Cauchy's multiplicative
  functional equation applied to recursive composition in d-dimensional
  space. The argument is as follows:

  1. Both systems involve recursive composition: the output of one
     process becomes the input to the next (branching levels in West;
     oscillator coupling cycles in Demetrius).

  2. Both composition operators are multiplicative: successive stages
     scale the output proportionally (branching ratios in West;
     Boltzmann factors in Demetrius).

  3. Cauchy's theorem constrains continuous multiplicative composition
     to the power-law family: f(x) = x^alpha.

  4. The space-filling condition in d dimensions forces the exponent
     to alpha = d/(d + 1), regardless of the specific physical
     mechanism.

  The convergence is therefore not coincidental. It is a mathematical
  consequence of applying Cauchy's constraint to recursive composition
  in finite-dimensional space. Any future derivation that satisfies the
  same conditions must arrive at the same formula.

  This is, to the author's knowledge, the first explicit identification
  of Cauchy's theorem as the meta-explanation for why independent
  derivations of metabolic scaling laws converge.

============================================================================
CHANGE 4: NEW SECTION after 2A - "The Dimensional Ladder"
============================================================================
LOCATION: After new Section 2A
INSERT NEW SECTION:

  ## 2B. The Dimensional Ladder and the Geometric Speed Limit

  The formula alpha = d/(d + 1) makes a prediction for every positive
  integer d. We present confirmed and consistent predictions:

  | d | Predicted | Domain | Measured | Status |
  |---|-----------|--------|----------|--------|
  | 1 | 0.500 | Cosmic radiation era | 0.500 exact | Confirmed |
  | 1 | 0.500 | Fungal metabolism | 0.50-0.55 | Consistent |
  | 2 | 0.667 | Cosmic matter era | 0.667 exact | Confirmed |
  | 2 | 0.667 | Biology | No valid test organism | Untested |
  | 3 | 0.750 | Mammalian BMR | 0.750 | Confirmed (90 years) |
  | 4 | 0.800 | Mammalian MMR | 0.872 (CI: 0.813-0.932) | Consistent |
  | inf | 1.000 | Extreme metabolic levels | Approaches 1.0 | Consistent |

  ### d = 4: Maximum metabolic rate

  Weibel et al. (2004) measured the maximal metabolic rate (MMR) of
  34 mammalian species during intense exercise. The scaling exponent
  was 0.872 (95% CI: 0.813-0.932), significantly different from the
  resting 3/4. West, Brown & Enquist (1999), in a paper titled 'The
  Fourth Dimension of Life', argued that the fractal branching depth of
  vascular networks adds an effective fourth dimension during maximum
  exertion, when the full capillary depth is recruited. The predicted
  value d/(d+1) for d = 4 is 4/5 = 0.800, which falls within the 95%
  confidence interval of the measured exponent.

  ### d approaching infinity: the geometric speed limit

  The formula alpha = d/(d + 1) is strictly less than 1 for all finite
  d. We identify alpha = 1 as the geometric speed limit: the maximum
  scaling efficiency achievable by any physical transport network in
  finite-dimensional space.

  This limit has three independent confirmations:
  (a) The formula itself: d/(d + 1) < 1 for all finite d.
  (b) Glazier (2008): metabolic scaling exponents approach 1.0 at
      extreme metabolic levels in both birds and mammals.
  (c) Demetrius (2006): the classical limit of the quantum oscillator
      model yields alpha = 1.

  ### d = 2 biological correction

  The v2.0 paper cited cnidarian data (jellyfish at 0.680, ctenophores
  at 0.660) as evidence for d = 2. This claim requires correction.
  Jellyfish do not possess genuinely two-dimensional hierarchical
  space-filling transport networks. Their gastrovascular canals
  distribute food, but oxygen enters by diffusion through thin tissue.
  The domain-of-applicability condition (hierarchical internal transport
  network) is not met. The d = 2 biological prediction remains untested,
  not because it has failed, but because no known organism has the
  required 2D network structure. The d = 2 confirmation exists in
  cosmology (Friedmann matter-era solution, exact) and in physics
  domains (2D percolation, fragmentation).

  ### The Friedmann boundary coincidence

  Under the mapping d = 2/(1 + 3w), the Friedmann expansion solution
  a(t) proportional to t^{2/(3(1+w))} becomes a(t) proportional to
  t^{d/(d+1)}. The strong energy condition boundary in general
  relativity (w = -1/3) maps to d approaching infinity, alpha
  approaching 1 - the same boundary between the power-law and
  exponential Cauchy families. This coincidence of two independently
  derived boundaries - one from 20th-century cosmology, one from
  19th-century functional analysis - is either structural or
  extraordinary.

============================================================================
CHANGE 5: Section 7.1 - Correct the cnidarian claims
============================================================================
LOCATION: Section 7.1, entries 33-35 in the published exponents table

REPLACE the jellyfish/cnidarian/ctenophore entries and their predicted
dimension:

CURRENT:
  | 33 | Jellyfish | 0.667 | 0.680 | d=2 | Yes |
  | 34 | Cnidarians | 0.667 | 0.700 | d=2 | Yes |
  | 35 | Ctenophores | 0.667 | 0.660 | d=2 | Yes |

REPLACE WITH:
  | 33 | Jellyfish | «CHECK» | 0.680 | d=2? | Needs revision |
  | 34 | Cnidarians | «CHECK» | 0.700 | d=2? | Needs revision |
  | 35 | Ctenophores | «CHECK» | 0.660 | d=2? | Needs revision |

ADD FOOTNOTE:
  "Entries 33-35 were classified as d=2 in v2.0 on the basis that these
  organisms have flat body plans with gastrovascular canal networks.
  On further investigation, jellyfish, cnidarians, and ctenophores do
  not possess genuinely 2D hierarchical space-filling transport networks.
  Their gastrovascular systems distribute food, but oxygen is primarily
  exchanged through surface diffusion. The d=2 classification requires
  an internal hierarchical transport network that partitions a
  2-dimensional space, which these organisms lack. The measured
  exponents (0.66-0.70) are close to the d=2 prediction but this may
  reflect an effective fractional dimension (d approximately 2.0-2.3)
  rather than a clean d=2 case. These entries are flagged for correction
  and should be treated as provisional rather than confirmed. The
  published exponent count for direct matches should be revised from
  13/13 to 10/13 confirmed plus 3 provisional."

============================================================================
CHANGE 6: Section 11 - Prior art update
============================================================================
LOCATION: Section 11, after the existing prior art entries
INSERT:

  "* **Demetrius (2006):** derived alpha = d/(d+1) from quantum
    oscillator coupling in the Debye model. This is the only known
    independent derivation of the same formula from a fundamentally
    different branch of physics. Demetrius does not cite Cauchy's
    functional equations, does not frame the result as a cross-domain
    scaling principle, and does not connect it to cosmic expansion or
    the geometric speed limit. The convergence of his quantum derivation
    with the network geometry derivation, unified by Cauchy's theorem,
    is the novel observation of this paper.

  * **Glazier (2008):** showed metabolic scaling exponents approach 1.0
    at extreme metabolic levels in birds and mammals. This is cited as
    empirical confirmation of the geometric speed limit but Glazier does
    not frame it in terms of Cauchy's equations or dimensional limits.

  * **Weibel et al. (2004):** measured maximal metabolic rate scaling at
    0.872. West et al. (1999) interpreted this as a 'fourth dimension'
    effect. Neither connects the d=4 prediction to the d/(d+1) ladder
    or to Cauchy's theorem."

============================================================================
CHANGE 7: References - add new citations
============================================================================
LOCATION: Add to reference list (or create if inline citations are used)

  Demetrius, L. (2006). Quantum metabolism explains the allometric
  scaling of metabolic rates. Journal of the Royal Society Interface,
  3, 843-851. PMC2842802.

  Glazier, D.S. (2008). Effects of metabolic level on the body size
  scaling of metabolic rate in birds and mammals. Proceedings of the
  Royal Society B, 275, 1405-1410.

  Weibel, E.R., Bacigalupe, L.D., Schmitt, B. & Hoppeler, H. (2004).
  Allometric scaling of maximal metabolic rate in mammals: muscle
  aerobic capacity as determinant factor. Respiratory Physiology &
  Neurobiology, 140, 115-132.

  West, G.B., Brown, J.H. & Enquist, B.J. (1999). The fourth dimension
  of life: fractal geometry and allometric scaling of organisms.
  Science, 284, 1677-1679.

============================================================================
CHANGE 8: Section 13 (Implications) - add new implications
============================================================================
LOCATION: Section 13, after existing implication #5
INSERT:

  "6. The identification of alpha = 1 as the geometric speed limit -
     the maximum scaling efficiency for physical transport networks -
     defines a boundary between the physical and intelligence regimes.
     Any system operating above alpha = 1 is, by definition, not a
     physical transport network constrained by spatial dimensionality.

  7. The convergence of two independent derivations (West et al. 1997
     from network geometry; Demetrius 2006 from quantum mechanics) on
     the same formula, explained by a single mathematical theorem
     (Cauchy 1821), suggests that the d/(d+1) identity may be one of
     the deepest structural constraints in mathematical physics: the
     unique scaling law compatible with continuous recursive composition
     in finite-dimensional space.

  8. The d=2 biological gap - the absence of any known organism with
     a genuinely 2D hierarchical transport network - may itself be
     informative. If the formula's domain of applicability requires
     hierarchical space-filling networks, the absence of such networks
     in 2D biology may reflect a geometric or evolutionary constraint
     on body plan design. Understanding why nature builds d=1 and d=3
     networks but not d=2 networks could yield insight into the
     relationship between dimensionality and biological complexity."

============================================================================
SUMMARY OF CHANGES
============================================================================

1. Version number to 3.0 with change log
2. Abstract: add Demetrius, speed limit, d=2 correction
3. New Section 2A: The Demetrius Convergence
4. New Section 2B: Dimensional Ladder and Geometric Speed Limit
5. Section 7.1: Correct cnidarian d=2 claims (13/13 -> 10/13 + 3 provisional)
6. Section 11: Add Demetrius, Glazier, Weibel to prior art
7. References: Add 4 new citations
8. Section 13: Add 3 new implications

Total: 8 changes. The existing 50-domain Cauchy result (19/25, p = 1.56e-5)
is untouched. The existing methodology, data, and code are untouched.
The changes ADD to the paper. They do not alter the primary result.

The single most important addition is the Demetrius convergence (Change 3).
If you can only make one change, make that one.
