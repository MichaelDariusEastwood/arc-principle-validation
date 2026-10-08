# Paper II: The ARC Equation Measured: Blinded Cross-Architecture Replication and the Retraction of a Super-Linear Estimate

**Earlier title:** Experimental Validation of Super-Linear Error Suppression (superseded: the super-linear estimate is retracted).

**Full title:** The ARC Equation Measured: Blinded Cross-Architecture Replication and the Retraction of a Super-Linear Estimate
**Version:** v2.16 (Working Paper)
**Version date:** revised 4 October 2026
**First published:** 22 January 2026
**Author:** Michael Darius Eastwood

## Summary

This paper measures the ARC Principle's scaling exponent across five frontier AI models (DeepSeek, Gemini, Grok, Groq Qwen, GPT). Only Gemini 3 Flash produced clean, monotonic, non-ceiling, non-floor data amenable to power-law fitting; its point estimate, approximately 0.49, is sub-linear, with a bootstrap interval of -1.3 to 2.9, and the early single-model estimate of approximately 2.24 is retracted. Parallel scaling is at or near zero in every model measured, with one measured exception, Gemini 3 Flash at 0.31. Sequential exceeded parallel for every model where both were measurable, and concurrent work by Sharma and Chopra (arXiv:2511.02309, 4 November 2025) reports sequential refinement beating parallel self-consistency at matched compute, on a different measure; a general compute-matched causal advantage and a universal scaling law are not established by this paper's observation.

## Experiments

All experiment scripts, results, figures, and data are in [`experiments/`](./experiments/).

## Links

- **OSF DOI:** https://doi.org/10.17605/OSF.IO/6C5XB
- **GitHub:** https://github.com/MichaelDariusEastwood/arc-principle-validation

Mirror refreshed 2026-08-13 from the site master (Option A: site HTML pages are the manuscript masters).
