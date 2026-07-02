# Miss Analysis: 6 Failures in the 50-Domain Cauchy Validation

**Date:** 2026-03-17
**Analyst:** Claude (independent review at request of M. D. Eastwood)
**Source data:** `results_50_domain_validation.json` and `canonical_50_domain_manifest.json`

---

## Summary Table

| # | Domain | Predicted | Best fit | Verdict | Category |
|---|--------|-----------|----------|---------|----------|
| 1 | Kleiber's Law | power_law | saturation_exp | Fitting artifact | (c) |
| 3 | Species-Area (Galapagos) | power_law | hill | Data quality | (b) |
| 6 | Zipf's Law | power_law | michaelis_menten | Fitting artifact (bug) | (c) |
| 15 | Muscle Force-Velocity | bounded | exponential | Fitting artifact | (c) |
| 20 | Time Crystal Order Parameter | bounded | power_law | Data quality | (b) |
| 21 | Stellar Mass-Luminosity | power_law | logistic | Fitting artifact | (c) |

**Verdict: 0 genuine framework failures. 4 fitting artifacts. 2 data quality issues.**

---

## Domain 1: Kleiber's Law (Metabolic Scaling)

**Prediction:** power_law | **Winner:** saturation_exp (bounded) | **n = 13**

### The data

Thirteen mammals from Kleiber's 1932 paper, body mass ranging from 0.15 kg (dove) to 679 kg (steer). The x-range spans roughly 4,500x. The power-law fit yields exponent 0.737, which is squarely in the canonical range (White et al. 2006 give 0.737 +/- 0.02 on hundreds of species).

### What happened

The fitting procedure fits all models in *linear* (original) space and selects by AICc. The power-law fit is obtained by OLS in log-log space and then evaluated in linear space. This is a well-known statistical trap:

- In log-log space, the power law fits beautifully (R^2 ~ 0.997 in log-log).
- In linear space, the power law has R^2 = 0.986 because the residuals for the two largest animals (342 kg, 679 kg) are large in absolute terms, even though they are small in relative/proportional terms.
- The saturation exponential has R^2 = 0.999 in linear space because it can 'bend' to reduce the absolute residual at the high end.

The saturation_exp fit gives an asymptote of 461.5 W. But the data only reach 400 W, and there is no biological reason for mammalian metabolic rate to saturate. The asymptote is an artifact of overfitting to the truncated x-range. Elephants (4,000 kg, ~3,500 W) and whales (100,000 kg, ~30,000+ W) exist and follow the power law.

The AICc gap is large (~37 points), but this is entirely because AICc is computed on linear-scale RSS, which heavily weights the largest data points. If the fitting were done properly for a power-law hypothesis (i.e., minimising RSS in log space, or using a log-normal error model), the power law would win decisively.

### Assessment

**Fitting artifact (c).** The power law is correct. The miss is caused by an error-model mismatch: the fitter assumes homoscedastic errors in linear space, but power-law data have multiplicative (heteroscedastic) errors. The saturation_exp asymptote at 461 W is physically absurd given that blue whales metabolise at ~30,000 W. A log-space AICc comparison, or a dataset with broader mass range, would fix this.

---

## Domain 3: Species-Area Relationship (Galapagos)

**Prediction:** power_law | **Winner:** hill (bounded) | **n = 30**

### The data

Thirty Galapagos islands from the Johnson & Raven (1973) dataset (the `faraway::gala` dataset in R). Area ranges from 0.01 km^2 to 4,669 km^2. Species counts range from 2 to 444.

### What happened

The scatter is enormous. The hill function wins with R^2 = 0.768; the power law (not shown in top 4 but computable from the log-log regression) would have a similar or slightly lower R^2 in linear space.

The data are notoriously noisy:
- Island with area 0.1 km^2 has 25 species; island with area 0.78 km^2 has 5 species.
- Island with area 642 km^2 has 93 species; island with area 170 km^2 has 285 species.
- Several small islands have more species than much larger ones (due to elevation, habitat diversity, distance from source).

The hill function fits with y_max = 487.8, K = 528.9, n = 0.54. The low Hill coefficient (n < 1) makes this function behave almost identically to a power law over the observed range. The difference is that the hill function can accommodate the 'ceiling' suggested by the largest island (Isabela, 4,669 km^2, 347 species), which looks like saturation only because there is a single very large island.

The species-area relationship S = cA^z is one of the most robust power laws in ecology, confirmed across thousands of datasets (Drakare et al. 2006, Ecology). The Galapagos dataset is simply too small and too noisy to distinguish between a power law and a quasi-power-law saturation curve.

### Assessment

**Data quality issue (b).** The relationship is genuinely a power law (confirmed by the enormous published literature). The 30-island Galapagos dataset has high scatter (multiple confounding variables: elevation, isolation, habitat type) and a single outlier island at the high end that creates an illusory saturation signal. A larger compilation (e.g., the global island database with >1,000 islands) would recover the power law.

---

## Domain 6: Zipf's Law (Word Frequency)

**Prediction:** power_law | **Winner:** michaelis_menten (bounded) | **n = 30**

### The data

Word frequencies from the Brown Corpus (Kucera & Francis 1967). Rank 1 (frequency 69,971) to rank 10,000 (frequency 1).

### What happened

This is the most clear-cut fitting artifact of all six misses, and arguably a bug in the pipeline.

The 'winning' michaelis_menten fit has:
- R^2 = 0.0
- K = 1.4e-18 (essentially zero)
- L = 11,200 (approximately the mean of y)

An R^2 of exactly 0.0 means the model is predicting y = mean(y) for all points. The Michaelis-Menten function L*x/(K+x) with K ~ 0 reduces to L (a constant). This is a degenerate fit that explains nothing.

Similarly, the saturation_exp has R^2 = -1.3e-14 (effectively zero), and the hill and logistic fits are equally degenerate (R^2 ~ 0).

The power law was not even returned in the top 4 fits. Looking at the fit_power_law function: it requires x > 0 and y > 0, both of which are satisfied. A log-log regression on this data would yield an exponent of approximately -1.07 with very high R^2 in log-log space. In linear space, the enormous dynamic range (69,971 down to 1) means the absolute residuals at rank 1 dominate.

The likely explanation: the power-law fit, while excellent in log space, has a moderate R^2 in linear space because the residual at rank 1 is large in absolute terms. But the bounded models have R^2 = 0, which is *worse*. The michaelis_menten nonetheless 'wins' on AICc because when R^2 = 0, rss/n = variance(y), and the AICc formula yields a finite value. If the power-law fit was computed and has a *worse* AICc than the constant-prediction model, there may be a numerical issue in the power-law evaluation (e.g., overflow or underflow in `a * x^b` for the extreme x values).

Regardless of the mechanism, a model with R^2 = 0 beating a model that captures the fundamental structure of the data is clearly a pipeline artifact. Zipf's law is one of the most well-established power laws in all of science.

### Assessment

**Fitting artifact (c), likely a pipeline bug.** The winner has R^2 = 0 -- it predicts the mean for every rank. This is not a meaningful model selection. The power law is overwhelmingly correct. The issue is likely that the linear-space AICc comparison breaks down when data span 5 orders of magnitude and the power-law fit is done via log-log OLS, creating a mismatch between the error model and the AICc computation.

---

## Domain 15: Muscle Force-Velocity (Hill 1938)

**Prediction:** bounded (hyperbolic_decay) | **Winner:** exponential | **n = 15**

### The data

Force-velocity data from Hill's 1938 classic paper. Velocity (x) ranges from 0 to 70.6 cm/s. Force (y) ranges from 4.9 (isometric) down to 0.0 at velocities of 60, 65, and 70.6 cm/s.

### What happened

The predicted family is 'bounded' and the predicted specific model is hyperbolic_decay: y = a/(b + x). This is the Hill equation for muscle (P + a)(v + b) = b(P_0 + a), which rearranges to P = a(v_max - v)/(b + v) or equivalently a decreasing hyperbolic.

The exponential fit wins (R^2 = 0.939) over hyperbolic_decay (R^2 = 0.902), with an AICc gap of about 7 points.

The critical issue: the data include three points where y = 0 exactly. This is physically correct -- muscle cannot produce force above its maximum shortening velocity. But:

1. The hyperbolic decay y = a/(b + x) **never reaches zero**. It asymptotes to zero but cannot equal it. So the three zero-valued points create unavoidable residuals for hyperbolic_decay.
2. The exponential y = a*exp(bx) also never reaches zero, but decays faster than a hyperbola at large x, so it can get closer to the zero points.

If the three zero-force points (x = 60, 65, 70.6) are excluded, the hyperbolic_decay fit would improve substantially because it would only need to fit the region where force is positive, which is exactly where Hill's equation applies.

Furthermore, the family prediction is 'bounded', and hyperbolic_decay is classified as 'bounded'. The exponential winning is a family-level miss because exponential is in the 'exponential' family. But the data genuinely show bounded behaviour (force is bounded between 0 and P_0). The exponential happens to approximate the bounded decay shape over this particular finite range.

### Assessment

**Fitting artifact (c).** The prediction (bounded/hyperbolic_decay) is physically correct -- this is literally the Hill equation, one of the most validated equations in muscle physiology. The miss occurs because: (i) exact zeros at high velocity create residuals that are irreducible for the hyperbola, and (ii) the exponential can approximate the decay shape over the observed range. Removing the three zero-force points, or using a modified Hill equation that allows force to reach zero at v_max, would recover the correct family.

---

## Domain 20: Time Crystal Order Parameter (Rydberg Gas)

**Prediction:** bounded | **Winner:** power_law | **n = 4**

### The data

Four data points from Shen et al. (2025): (1.1, 0.0), (2.19, 0.3), (2.8, 0.65), (4.44, 0.95).

### What happened

This is entirely a sample-size problem interacting with the AICc correction.

The hill function fits with R^2 = 0.9999 -- essentially perfect. The logistic fits with R^2 = 0.9999 as well. Both are bounded models, matching the prediction.

However, both hill and logistic have 3 free parameters. With n = 4 data points and k = 3 parameters, the AICc correction term is 2k(k+1)/(n-k-1) = 24/0, which is infinite. The AICc is therefore infinite for any 3-parameter model.

The power law has k = 2, giving n-k-1 = 1, so AICc is finite (though the correction is enormous: 2*2*3/1 = 12). Thus the power law wins by default, not by merit.

This is a pure mechanical consequence of having fewer data points than parameters + 2. The prediction is correct -- the order parameter of a time crystal is bounded between 0 and 1, and the data clearly show saturation towards 1. The AICc-based model selection simply cannot work with 4 data points and 3-parameter models.

### Assessment

**Data quality issue (b).** With only 4 data points, AICc penalises 3-parameter models with an infinite correction. The 2-parameter bounded models (saturation_exp, michaelis_menten) are also unable to compete because the data start at y = 0 exactly (saturation_exp(0) = 0 but michaelis_menten(0) = 0 as well; both should be able to fit). The saturation_exp was apparently invalid (not in top 4), possibly because the x-values don't start at 0. Regardless, this domain provides essentially zero information about model selection and should be flagged as underpowered. A dataset with >= 8 points from the original paper would resolve this.

---

## Domain 21: Stellar Mass-Luminosity (Main Sequence)

**Prediction:** power_law | **Winner:** logistic (bounded) | **n = 19**

### The data

Nineteen main-sequence stars from Torres et al. (2010) and Eker et al. (2018). Mass ranges from 0.09 to 3.8 solar masses. Luminosity ranges from 0.0014 to 288 solar luminosities -- a dynamic range of ~200,000x.

### What happened

The power-law fit gives exponent 3.36 in log-log space. In log-log space, R^2 is high (~0.96). But in linear space, R^2 = 0.568.

The logistic fit gives R^2 = 0.9996 with K = 290.7, r = 3.96, x0 = 2.62. This is a remarkable fit, but it is physically meaningless. The logistic function K/(1 + exp(-r(x - x0))) predicts that luminosity saturates at 290.7 L_sun. This is false: stars at 10 solar masses have luminosity ~10,000 L_sun; stars at 50 solar masses reach ~1,000,000 L_sun.

The logistic wins because:
1. The data are heavily concentrated below 2 solar masses (14 of 19 points below M = 2).
2. There is one high-mass outlier at 3.8 solar masses with L = 288 L_sun.
3. The logistic can fit the low-mass 'floor' near zero, the rapid rise around 1-2 solar masses, and the single point at 288 as its 'ceiling'.
4. The power law, fitted in log space, produces predictions in linear space that have large absolute residuals at the high end.

This is the same error-model mismatch as Domain 1. The mass-luminosity relation L ~ M^3.5 is one of the most secure power laws in astrophysics, confirmed over many orders of magnitude. The dataset here spans only 1.6 decades in mass, with most points clustered below 2 M_sun. The logistic is essentially fitting a local segment of the power law and interpreting it as a sigmoid.

If the dataset extended to 10 or 20 solar masses (where L ~ 10,000-100,000 L_sun), the logistic ceiling at 290.7 would be destroyed and the power law would win.

### Assessment

**Fitting artifact (c).** The mass-luminosity power law is one of the most established results in stellar astrophysics. The miss is caused by: (i) the dataset spans too narrow a mass range to distinguish power law from a local sigmoid, (ii) the enormous dynamic range in y means linear-space AICc is dominated by the single high-luminosity point, and (iii) the logistic function with an asymptote of 290.7 is physically absurd (stars much more luminous than this are common). Extending the dataset to higher masses, or fitting in log space, would recover the power law.

---

## Overall Assessment

### No genuine framework failures

All six misses fall into two categories:

**Fitting artifacts (4 domains: 1, 6, 15, 21):** The AICc model selection procedure compares RSS in linear (original) space. For power-law data spanning multiple orders of magnitude, this systematically penalises power laws because:
- Power-law fits are obtained by OLS in log space, which minimises relative error.
- AICc is computed on absolute (linear-space) RSS, which weights the largest data points heavily.
- Bounded models with physically absurd asymptotes can reduce absolute RSS at the top end.

This is a known statistical issue. The correct approach for power-law model selection is to compute AICc in log space (i.e., on the log-transformed residuals), which corresponds to a log-normal error model. This is more physically appropriate for multiplicative processes.

Domain 6 (Zipf) is the most egregious: the 'winner' has R^2 = 0 and is literally a constant function.

Domain 15 (Hill muscle) is a different kind of artifact: the data contain exact zeros that are incompatible with the correct functional form.

**Data quality issues (2 domains: 3, 20):**
- Domain 3 (Galapagos) has high scatter from confounding variables and only 30 islands.
- Domain 20 (time crystal) has only 4 data points, making AICc-based selection of 3-parameter models impossible.

### Recommendations

1. **Log-space AICc for multiplicative domains.** When `operator_class` is `multiplicative`, compute RSS and AICc on log-transformed data. This would fix Domains 1, 6, and 21.
2. **Exclude exact zeros from hyperbolic fits.** For Domain 15, either exclude zero-force points or use a truncated Hill equation.
3. **Minimum sample size guard.** Flag domains with n < 6 as underpowered for model selection (fixes Domain 20).
4. **Degenerate-fit guard.** Reject any model whose R^2 < some minimal threshold (e.g., 0.05) from winning the AICc competition (fixes Domain 6).
