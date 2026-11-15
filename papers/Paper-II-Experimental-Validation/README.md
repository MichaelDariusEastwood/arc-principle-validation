# Blinded cross-architecture replication and the retraction of a super-linear estimate

**Version:** 2.20
**Revised:** 15 November 2026
**First published:** 2026-01-22
**Author:** Michael Darius Eastwood

## Summary

From the abstract (version 2.20):

> How much more capable does AI become when it thinks longer? If capability compounded super-linearly with reasoning depth, a safeguard that did not grow with it would fall behind, so the answer bears on AI safety. Law I of the ARC Theory, the ARC Principle, a law under test, concerns capability; measured as error, $E(R)=E_0R^{-\alpha}$, only through a metric bridge, it first predicted super-linear suppression ($\alpha>1$) for sequential reasoning and sub-linear suppression for parallel sampling. We measured $\alpha$ on competition mathematics in two phases. On DeepSeek R1 with visible reasoning tokens (12 problems), the parallel exponent was $\alpha\approx0$, and the sequential endpoint estimate, $\alpha\approx2.24$ (quadratic), is retracted: it did not replicate in a blinded study across frontier models (18 AIME and Putnam-level problems, $n=54$ per depth per model, bootstrap intervals, and four-layer cross-verification in which no model checks its own answers). Two models were at or near ceiling, one switched from 54 to 100 per cent as a step and one stayed near 50 per cent; only Gemini 3 Flash scaled smoothly, with regression estimate $0.49$ (standard error $0.20$, $r^2=0.86$, three depth levels) and endpoint estimate $0.59$, whose bootstrap 95 per cent interval $[-1.3, 2.9]$ contains both zero and two. That is one model’s fit and inconclusive, and its closeness to a reference exponent of one half identifies no mechanism. More samples left the parallel exponent near zero in DeepSeek R1 and GPT-5.4, gave $0.31$ in Gemini 3 Flash and $0.22$ in Groq Qwen3 (all 30 problems), and gave none for Grok 4.1 Fast, at 100 per cent; sequential exceeded parallel wherever both exponents could be estimated. Alignment, measured in a companion arm, did not move with capability (one run, under a code-review caveat). We state what would refute the surviving claim and what a deciding measurement requires.

## Status and changes

Version 2.20 corrects the paper’s account of the record of 8 December 2024 and states Law I, the ARC Principle, as it is registered; Appendix B gives each correction. Internally source-checked and read in rendered form; not independent peer review.

The complete historical material and correction record remain in the lettered appendices.

## Files and sources

- [PDF](Paper-II-Experimental-Validation.pdf)
- [Document HTML](Paper-II-Experimental-Validation.html)
- [Text](Paper-II-Experimental-Validation.txt)
- [Citation metadata](Paper-II-Experimental-Validation.cff)
- [LaTeX source](Paper-II-Experimental-Validation-latex-source.tar.gz)
- [OSF record](https://doi.org/10.17605/OSF.IO/8FJMA)

These are the official document exports. Stable filenames preserve earlier versions in file history. Written research is CC BY 4.0; separately identified code, data and source records retain their stated licences.
