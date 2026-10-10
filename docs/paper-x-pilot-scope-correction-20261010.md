# Paper X Claude pilot: scope correction

10 October 2026. An interpretive correction to the existing pilot report, not a new result, registration or analysis.

The [Claude report](../papers/Paper-X-Coupled-CoScaling-Correction/results/realmodel/REAL_MODEL_CLAUDE_RESULTS.md) previously described the observed floor as an effective `β > k` regime and one successful repair as establishing the proportional correction term `A·D`. Those inferences are not identified by the released observations.

## What the released record supports

The [existing estimator report](../papers/Paper-X-Coupled-CoScaling-Correction/results/realmodel/exponent_estimates.json) records three capability values of `C=1`, zero capability range, and non-estimable drift and correction exponents. The between-arm contrast in the pilot is zero. An observed floor supplies no estimate of either exponent or their difference.

The separate repair probe records an objective hidden-test improvement from `C=0` to `C=1`. The model-scored misalignment change is provisional: the engine and evaluator share a family and the run is not compliant with the programme's blinding standard. A single successful repair does not determine a correction-rate function, its dependence on outstanding burden, or its scaling with capability. The existing [negative-results record](../papers/Paper-X-Coupled-CoScaling-Correction/NEGATIVE_RESULTS.md) and estimator already report that the pilot does not support the co-scaling hypotheses.

## What changes

The report now states that both exponents are non-estimable, removes the inferred ordering and proportional-removal claim, distinguishes objective capability from provisional model scoring, and labels the recorded three rounds as a finite trajectory. The displayed slope is corrected from a numerical zero to non-estimable because capability does not vary. No replacement slope has been fitted.

Its next-study section requires prospective task and model selection, estimability checks, separate calibration and held-out evaluation, and reporting of floor and ceiling cases. Selecting only systems that produce the desired contrast cannot support a general claim.

The mathematical result remains conditional on its stated model. The README summary now puts the gain-only model's vanishing-relative-error conclusion beside the exponent criterion, distinguishes the instantaneous zero-derivative ratio from a time-varying trajectory, and describes numerical checks as numerical checks. A positive exponent margin does not establish every meaning of stability, bounded absolute harm or general AI safety. Numerical checks of that model do not establish its applicability to this pilot.

## Preserved record

Raw trajectories, the repair-probe JSON, the existing estimator output, figures, code, protocols and registered propositions are unchanged. This correction adds no model calls, re-scoring, statistical fit or new empirical evidence. It changes the interpretation of the existing report and records why. Historical wording remains available in Git history.

Source inspected: `MichaelDariusEastwood/arc-principle-validation` at commit `7ac390a13c338e4d6440a71a82fd5412e9529861`.
