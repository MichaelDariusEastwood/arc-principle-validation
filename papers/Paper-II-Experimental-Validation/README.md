# Paper II · The ARC Theory · Law I, the ARC Principle · Blinded Cross-Architecture Replication and the Retraction of a Super-Linear Estimate

**Earlier title:** Experimental Validation of Super-Linear Error Suppression (superseded: the super-linear estimate is retracted).

**Full title:** Paper II · The ARC Theory · Law I, the ARC Principle · Blinded Cross-Architecture Replication and the Retraction of a Super-Linear Estimate
**Version:** v2.17 (Working Paper)
**Version date:** revised 5 October 2026
**First published:** 22 January 2026
**Author:** Michael Darius Eastwood

## Summary

From the abstract (version 2.17):

> This paper presents experimental validation of the ARC Principle (Artificial Recursive Creation), a mathematical framework proposing that error rates in intelligent systems decrease according to a power law with recursive depth. ... Phase 1 (the initial single-model study) used DeepSeek R1 with visible reasoning tokens on 12 competition-level mathematics problems, finding $\alpha_{\text{sequential}} \approx 2.24$ (95% CI: 1.5-3.0), since retracted, and $\alpha_{\text{parallel}} \approx 0.0$. Phase 2 (the six-model study) extended testing to six frontier models on 18 AIME/Putnam-level problems (n=54 per depth per model) with bootstrap confidence intervals and 4-layer cross-verification. ... Only Gemini 3 Flash produced clean, monotonic scaling data: $\alpha_{\text{seq}} = 0.49$ (regression, $r^2 = 0.86$, SE = 0.20; endpoint estimate 0.59, boot CI [−1.3, 2.9]). ... The replicated finding is therefore that parallel scaling is at or near zero in every model measured, with one measured exception, not that it is universally zero. Sequential outperformed parallel for every model where both were measurable.

## Experiments

All experiment scripts, results, figures, and data are in [`experiments/`](./experiments/).

## Links

- **OSF DOI:** https://doi.org/10.17605/OSF.IO/6C5XB
- **GitHub:** https://github.com/MichaelDariusEastwood/arc-principle-validation

The HTML is the official document export, not the website page. The PDF is byte-identical to that export release on the website. New versions retain these filenames, with their own commits.
