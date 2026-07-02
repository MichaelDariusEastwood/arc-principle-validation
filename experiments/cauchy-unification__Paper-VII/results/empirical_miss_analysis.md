# Empirical Miss Analysis

This report separates likely framework misses from domains where the current dataset is too weak, truncated, or regime-mixed to carry much evidential weight.

- Empirical misses in current canonical cohort: `6`
- Rule: keep every current miss as a miss in the canonical result; do not retroactively promote any domain without a preregistered follow-up dataset.

## 1. Kleiber's Law (Metabolic Scaling)

- Predicted family: `power_law`
- Current best family: `bounded`
- Current best model: `saturation_exp`
- Likely driver: Dataset choice / sample size
- Suspected issue: Small legacy dataset and wide body-mass span let flexible bounded curves overfit the upper tail.
- Recommended action: Refit against a larger modern mammal compilation such as White et al. or a comparable multi-hundred-species dataset.
- Follow-up posture: Keep as a miss in the canonical suite; test whether the miss survives on an updated compendium before changing any headline claim.

Top competing fits:

- `saturation_exp` (bounded): AICc `40.234`, R^2 `0.9992`
- `hill` (bounded): AICc `46.220`, R^2 `0.9990`
- `michaelis_menten` (bounded): AICc `46.847`, R^2 `0.9987`

## 3. Species-Area Relationship (Galapagos)

- Predicted family: `power_law`
- Current best family: `bounded`
- Current best model: `hill`
- Likely driver: Noisy empirical regime
- Suspected issue: Noisy small-island archipelago data with high scatter makes bounded alternatives competitive.
- Recommended action: Use a larger island system or preregister a replacement biodiversity dataset with broader area coverage and clearer sampling quality.
- Follow-up posture: Treat as an ambiguous ecological domain rather than a clean falsifier until replicated on a stronger species-area dataset.

Top competing fits:

- `hill` (bounded): AICc `246.528`, R^2 `0.7684`
- `michaelis_menten` (bounded): AICc `246.870`, R^2 `0.7456`
- `saturation_exp` (bounded): AICc `249.474`, R^2 `0.7225`

## 6. Zipf's Law (Word Frequency)

- Predicted family: `power_law`
- Current best family: `bounded`
- Current best model: `michaelis_menten`
- Likely driver: Finite-size artifact / corpus truncation
- Suspected issue: Finite-corpus truncation and extreme-rank zeros collapse the fit quality and allow bounded models to win trivially.
- Recommended action: Refit on a much larger corpus and preregister the usable rank window to avoid tail-collapse artifacts.
- Follow-up posture: Re-run with a modern large corpus and an explicit mid-rank inclusion rule rather than retrofitting the current Brown Corpus result.

Top competing fits:

- `michaelis_menten` (bounded): AICc `577.978`, R^2 `0.0000`
- `saturation_exp` (bounded): AICc `577.978`, R^2 `-0.0000`
- `hill` (bounded): AICc `580.456`, R^2 `-0.0000`

## 15. Muscle Force-Velocity (Hill 1938)

- Predicted family: `bounded`
- Current best family: `exponential`
- Current best model: `exponential`
- Likely driver: Incomplete regime coverage
- Suspected issue: The recorded series drops to zero before a clean asymptotic hyperbolic tail is expressed, making exponential decay competitive.
- Recommended action: Replace with a modern force-velocity dataset or a digitized raw series spanning the hyperbolic regime without zero-clipped tail compression.
- Follow-up posture: Keep as a miss now; the dataset likely under-resolves the exact bounded subfamily rather than disproving the bounded family itself.

Top competing fits:

- `exponential` (exponential): AICc `-25.126`, R^2 `0.9386`
- `hyperbolic_decay` (bounded): AICc `-18.092`, R^2 `0.9018`
- `logistic` (bounded): AICc `19.905`, R^2 `0.0000`

## 20. Time Crystal Order Parameter (Rydberg Gas)

- Predicted family: `bounded`
- Current best family: `power_law`
- Current best model: `power_law`
- Likely driver: Undersampled dataset
- Suspected issue: Only four points are available, so onset behavior dominates and saturation cannot be distinguished cleanly from a power-law rise.
- Recommended action: Obtain supplementary or follow-on time-crystal measurements with denser coverage across the control parameter.
- Follow-up posture: Classify as a weak-domain miss and do not overinterpret it until more points exist.

Top competing fits:

- `power_law` (power_law): AICc `-2.050`, R^2 `0.9144`
- `exponential` (exponential): AICc `0.731`, R^2 `0.8284`
- `hill` (bounded): AICc `inf`, R^2 `0.9999`

## 21. Stellar Mass-Luminosity (Main Sequence)

- Predicted family: `power_law`
- Current best family: `bounded`
- Current best model: `logistic`
- Likely driver: Mixed-regime physics
- Suspected issue: High-mass stellar scatter and mixed physical regimes make a single global power law unstable against bounded alternatives.
- Recommended action: Restrict to a preregistered homogeneous main-sequence subset or piecewise mass band rather than fitting one curve across mixed stellar regimes.
- Follow-up posture: Use as a case for regime splitting in the next extension, not as a retroactive patch to the canonical 25-domain cohort.

Top competing fits:

- `logistic` (bounded): AICc `18.925`, R^2 `0.9996`
- `hill` (bounded): AICc `24.235`, R^2 `0.9994`
- `power_law` (power_law): AICc `146.785`, R^2 `0.5678`

## Recommended order of attack

1. Time crystal: more points first.
2. Kleiber and Zipf: stronger modern datasets first.
3. Stellar mass-luminosity: preregistered regime split.
4. Species-area and force-velocity: replacement-quality datasets before interpretive escalation.
