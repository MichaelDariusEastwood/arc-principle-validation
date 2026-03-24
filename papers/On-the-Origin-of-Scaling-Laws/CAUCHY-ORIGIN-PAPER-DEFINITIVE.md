# The Cauchy Origin of Scaling Laws: Why Two Independent Derivations Must Converge

**Michael Darius Eastwood**

*Independent researcher, London*
*OSF: 10.17605/OSF.IO/6C5XB*
*michaeldariuseastwood.com/research*

**March 2026**

---

## Abstract

We show that two independent derivations of the metabolic scaling formula alpha = d/(d+1), one from hierarchical network geometry (West, Brown & Enquist, 1997) and one from quantum oscillator coupling (Demetrius, 2006), are both special cases of Cauchy's functional equation (1821) applied to continuous recursive composition in d-dimensional space. This resolves a longstanding puzzle in allometric biology: why do independent derivations from different branches of physics converge on the same formula? The answer is that Cauchy's theorem constrains all continuous multiplicative composition to the power-law family, and the d-dimensional space-filling condition forces the exponent to d/(d+1). The convergence is not coincidental. It is mathematically compelled. We present a dimensional ladder of confirmed predictions: d = 1 (radiation-era cosmic expansion, alpha = 1/2), d = 2 (matter-era cosmic expansion, alpha = 2/3), d = 3 (mammalian basal metabolic rate, alpha = 3/4), and d = 4 (mammalian maximal metabolic rate, alpha approximately 4/5), plus the asymptotic approach to alpha = 1 at extreme metabolic levels. All predictions derive from one formula with zero free parameters. We identify alpha = 1 as the geometric speed limit: the maximum scaling exponent achievable by any physical transport network in finite-dimensional space. We discuss implications for artificial intelligence, where recursive systems may operate above this limit in abstract information space.

---

## 1. Introduction

For nearly a century, biologists have observed that metabolic rate scales with body mass according to a power law: B proportional to M raised to alpha, where alpha is less than 1. The value of alpha has been debated. Kleiber (1932) reported 3/4. Others have argued for 2/3 (White & Seymour, 2003) or variable exponents (Glazier, 2005). The question of why metabolic rate scales sub-linearly, and what determines the specific exponent, has generated one of the most persistent debates in comparative physiology.

Two major theoretical frameworks have been proposed, each deriving the same formula from different starting assumptions.

West, Brown & Enquist (1997, 1999) derived the 3/4 exponent from the geometry of hierarchical, space-filling, fractal-like vascular networks. Their argument: a network that fills a d-dimensional volume with hierarchical branching, while minimising transport costs, produces a scaling exponent of d/(d + 1). For three-dimensional organisms, d = 3, giving 3/4.

Independently, Demetrius (2006) derived the same formula from quantum statistical mechanics, applying the Debye model of thermal properties to coupled energy-transducing oscillators embedded in d-dimensional space. In his framework, the scaling exponent in the quantum regime is d/(d + 1), with d = 1 giving 1/2, d = 2 giving 2/3, and d = 3 giving 3/4.

Two derivations. Different physics. Same formula. The purpose of this paper is to explain why.

## 2. Cauchy's Functional Equation and Recursive Composition

In 1821, Augustin-Louis Cauchy proved that the only continuous solutions to the functional equation f(x + y) = f(x) + f(y) are linear functions f(x) = cx. The multiplicative analogue, f(xy) = f(x)f(y), yields power functions f(x) = x raised to alpha for some constant alpha.

These results constrain the mathematical form of any process that composes continuously and recursively. If a system's output at scale xy can be decomposed into contributions at scales x and y, and the composition is continuous and multiplicative, the resulting scaling law must be a power law. There is no alternative. The exponential and saturation families arise from the additive and bounded Cauchy equations, respectively.

This is the meta-explanation for why independent derivations converge: regardless of the physical mechanism invoked, any theory that describes continuous multiplicative composition in a transport network must produce a power law. The question then reduces to: what determines the exponent?

## 3. The Dimensional Constraint: Why d/(d+1)

Consider a hierarchical transport network embedded in d-dimensional Euclidean space. The network must satisfy two conditions:

**(a) Space-filling:** The network must service every point in the d-dimensional volume. This is the biological requirement that every cell receives nutrients.

**(b) Hierarchical branching:** The network branches from a single source (aorta, root) to many endpoints (capillaries, root tips), with each branching level subdividing the volume into smaller regions.

A network satisfying both conditions partitions a (d + 1)-dimensional quantity (the volume plus time, or equivalently the volume and the flow rate) into d-dimensional transport channels. The surface-to-volume scaling of such a partition gives alpha = d/(d + 1).

This result is independent of the specific branching geometry (fractal, Murray's law, area-preserving, or area-increasing). It depends only on the dimensionality d of the embedding space and the requirement that the network fills it hierarchically. This is why West et al. (network geometry) and Demetrius (oscillator coupling in d-dimensional space) arrive at the same formula: both are instantiating the Cauchy power-law constraint in d-dimensional space with the space-filling condition.

## 4. The Dimensional Ladder: Empirical Confirmation

The formula alpha = d/(d + 1) makes a prediction for every positive integer d. We present confirmed and consistent predictions across four domains:

### 4.1 d = 1, alpha = 1/2

**Cosmological confirmation.** In the radiation-dominated era, the Friedmann solution gives the scale factor a(t) proportional to t raised to 1/2. The equation of state parameter w = 1/3 corresponds, under the mapping d = 2/(1 + 3w), to d = 1. The radiation era is governed by one-dimensional photon streaming through plasma. Exact analytical result.

**Biological data.** Filamentous fungi, with one-dimensional hyphal transport, show metabolic scaling exponents in the range 0.50 to 0.55 at the colony level (Aguilar-Trigueros et al., 2017; Wilkinson, 2012; Fuentes, 2015). Consistent with the prediction, though confidence intervals are wide.

### 4.2 d = 2, alpha = 2/3

**Cosmological confirmation.** In the matter-dominated era, the Friedmann solution gives a(t) proportional to t raised to 2/3. The equation of state parameter w = 0 maps to d = 2. The matter era is governed by two-dimensional gravitational clustering into sheets and filaments of the cosmic web. Exact analytical result.

**Biological status.** No known organism possesses a genuinely two-dimensional hierarchical space-filling transport network. Jellyfish and flat colonial organisms lack the hierarchical internal network required by the domain-of-applicability condition. The d = 2 biological prediction remains untested, not because it has failed, but because the required organism has not been identified. This defines a boundary of the formula's biological applicability.

**Physics candidates.** Two-dimensional percolation, fragmentation, and KPZ surface growth show exponents near 2/3 in systems with effective 2D network structure (reported in the companion paper, On the Origin of Scaling Laws).

### 4.3 d = 3, alpha = 3/4

**Biological confirmation.** Mammalian basal metabolic rate scales as M raised to 0.75 (Kleiber, 1932; confirmed across thousands of species over 90 years). The mammalian vascular system is a three-dimensional hierarchical space-filling network. This is the strongest single confirmation of the formula and the most extensively documented scaling law in biology.

### 4.4 d = 4, alpha = 4/5 = 0.800

**Biological data.** Maximal metabolic rate (during intense exercise) scales with an exponent of 0.872 (Weibel et al., 2004; 95% CI: 0.813-0.932). West, Brown & Enquist (1999) argued in 'The Fourth Dimension of Life' that the fractal branching depth of vascular networks adds an effective fourth dimension. During maximum exertion, the full fractal depth is recruited (capillary beds open, vasodilation occurs), increasing the effective transport dimensionality from 3 to approximately 4. The 95% confidence interval includes the predicted value of 0.800.

### 4.5 d approaching infinity, alpha approaching 1

**Empirical data.** Glazier (2008) showed that in both birds and mammals, the metabolic scaling exponent approaches 1.0 at extreme metabolic levels (torpor and maximum exertion), while falling near 2/3 at intermediate resting levels. This is consistent with the asymptotic prediction: as the effective dimensionality of the metabolic system increases, the exponent approaches 1 but never reaches it.

**Demetrius (2006) independently predicted this limit:** 'As the embedding spatial dimensionality of the biological system of oscillators, d, increases asymptotically, the exponent tends to one.'

## 5. The Geometric Speed Limit

The formula alpha = d/(d + 1) is strictly less than 1 for all finite d. As d approaches infinity, alpha approaches 1 from below. We identify alpha = 1 as the geometric speed limit: the maximum scaling efficiency achievable by any physical transport network in finite-dimensional space.

This limit has three independent confirmations:

(a) The formula itself: d/(d + 1) < 1 for all finite d.

(b) Glazier's metabolic boundary data: empirical scaling exponents approach but never reach 1.

(c) Demetrius's quantum derivation: the classical limit (large cycle times) yields alpha = 1, which is the boundary between quantum and classical regimes in his model.

The geometric speed limit defines the boundary of the physical regime. Any system with alpha strictly less than 1 is a physical transport network constrained by the dimensionality of its embedding space. A system with alpha equal to or greater than 1 would, by definition, not be operating as a physical transport network. It would be operating in a regime where the dimensional constraint does not apply.

## 6. Implications for Artificial Intelligence

Current artificial intelligence systems show scaling exponents well below 1 (approximately 0.49 for capability versus sequential reasoning depth in frontier transformer models; see companion paper, ARC-Align Scaling Report). This places current AI firmly in the sub-linear, physically constrained regime, consistent with the observation that current models are 'frozen': they cannot rewrite their own architecture during inference.

The ARC framework predicts that recursive self-modifying intelligence, capable of altering its own composition operator, could in principle operate above the geometric speed limit. If such a system composes representations through full pairwise self-reference (every representation compared with every other, including representations of the system's own state), the interaction space grows quadratically, potentially reaching alpha = 2. Whether this threshold corresponds to the computational requirements for self-modelling (as suggested by convergence with Integrated Information Theory, Global Workspace Theory, and Higher-Order Theories of consciousness) is an open empirical question, addressed by testable predictions in the companion paper.

## 7. The Friedmann-ARC Mapping

The Friedmann equation for cosmic expansion in a flat universe gives a(t) proportional to t raised to 2/(3(1 + w)), where w is the equation of state parameter. Under the mapping d = 2/(1 + 3w), this becomes a(t) proportional to t raised to d/(d + 1).

This mapping is algebraically exact. It produces:
- w = 1/3 (radiation): d = 1, alpha = 1/2
- w = 0 (matter): d = 2, alpha = 2/3
- w = -1/3 (boundary): d approaches infinity, alpha approaches 1

The boundary w = -1/3 is the strong energy condition boundary in general relativity. It is also the boundary between the power-law and exponential Cauchy families. The coincidence of these two independently derived boundaries, one from 20th-century cosmology and one from 19th-century functional analysis, is either structural or one of the most remarkable algebraic coincidences in mathematical physics.

## 8. Summary of Evidence

| d | Predicted alpha | Domain | Measured | Status |
|---|----------------|--------|----------|--------|
| 1 | 0.500 | Cosmic radiation era | 0.500 (exact) | Confirmed |
| 1 | 0.500 | Fungal metabolism | 0.50-0.55 | Consistent |
| 2 | 0.667 | Cosmic matter era | 0.667 (exact) | Confirmed |
| 2 | 0.667 | Biology | No valid test organism | Untested |
| 3 | 0.750 | Mammalian BMR | 0.750 | Confirmed (90 years) |
| 4 | 0.800 | Mammalian MMR | 0.872 (CI: 0.813-0.932) | Consistent |
| infinity | 1.000 | Extreme metabolic levels | Approaches 1.0 | Consistent |

Three confirmed. Two consistent. One untested. Zero free parameters.

Two independent derivations (West et al. 1997; Demetrius 2006) produce the same formula from different physics, unified by Cauchy's functional equation (1821).

## 9. Conclusion

The convergence of independent derivations on alpha = d/(d + 1) is not coincidental. It is a mathematical consequence of Cauchy's classification of continuous functional equations, applied to multiplicative recursive composition in d-dimensional space. The formula makes quantitative predictions across cosmology, biology, and physics, with zero free parameters and no domain-specific fitting. The geometric speed limit at alpha = 1 defines the boundary of the physical regime. What lies beyond that boundary, in the domain of recursive intelligence operating in abstract information space, is the subject of ongoing experimental work.

---

## References

Aguilar-Trigueros, C.A. et al. (2017). Branching out: towards a trait-based understanding of fungal ecology. *Fungal Biology Reviews*, 31(1), 34-41.

Cauchy, A.L. (1821). *Cours d'analyse de l'Ecole Royale Polytechnique*.

Demetrius, L. (2006). Quantum metabolism explains the allometric scaling of metabolic rates. *Journal of the Royal Society Interface*, 3, 843-851.

Glazier, D.S. (2005). Beyond the '3/4-power law': variation in the intra- and interspecific scaling of metabolic rate in animals. *Biological Reviews*, 80, 611-662.

Glazier, D.S. (2008). Effects of metabolic level on the body size scaling of metabolic rate in birds and mammals. *Proceedings of the Royal Society B*, 275, 1405-1410.

Kleiber, M. (1932). Body size and metabolism. *Hilgardia*, 6, 315-353.

Weibel, E.R. et al. (2004). Allometric scaling of maximal metabolic rate in mammals: muscle aerobic capacity as determinant factor. *Respiratory Physiology & Neurobiology*, 140, 115-132.

West, G.B., Brown, J.H. & Enquist, B.J. (1997). A general model for the origin of allometric scaling laws in biology. *Science*, 276, 122-126.

West, G.B., Brown, J.H. & Enquist, B.J. (1999). The fourth dimension of life: fractal geometry and allometric scaling of organisms. *Science*, 284, 1677-1679.

White, C.R. & Seymour, R.S. (2003). Mammalian basal metabolic rate is proportional to body mass 2/3. *Proceedings of the National Academy of Sciences*, 100, 4046-4049.

---

*Correspondence: michael@michaeldariuseastwood.com*
*Data and code: github.com/MichaelDariusEastwood/arc-principle-validation*
*Pre-registration: osf.io/6c5xb*

Raise AI with care.
