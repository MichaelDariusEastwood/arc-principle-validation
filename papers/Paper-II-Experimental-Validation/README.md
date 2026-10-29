# Paper II · The ARC Theory · Law I, the ARC Principle · Blinded Cross-Architecture Replication and the Retraction of a Super-Linear Estimate

**Headline:** Blinded cross-architecture replication and the retraction of a super-linear estimate
**Version:** v2.19
**Version date:** 29 October 2026
**First published:** 22 January 2026
**Author:** Michael Darius Eastwood

## Summary

From the abstract (version 2.19):

> How much more capable does AI become when it thinks longer? If capability compounded super-linearly with reasoning depth, a safeguard that did not grow with it would fall behind, so the answer bears directly on AI safety. Law I of the ARC Theory, the ARC Principle, a law under test, states that error falls as a power of recursive depth, $E(R)=E_0R^{-\alpha}$, and first predicted super-linear suppression ($\alpha>1$) for sequential reasoning and sub-linear suppression for parallel sampling. We measured $\alpha$ on competition mathematics in two phases. On DeepSeek R1 with visible reasoning tokens (12 problems), the parallel exponent was $\alpha\approx0$, and the sequential endpoint estimate, $\alpha\approx2.24$ (quadratic), is retracted: it did not replicate in a blinded study across frontier models (18 AIME and Putnam-level problems, $n=54$ per depth per model, bootstrap intervals, and four-layer cross-verification in which no model checks its own answers). Two models were at or near ceiling, one switched from 54 to 100 per cent as a step and one stayed near 50 per cent; only Gemini 3 Flash scaled smoothly, with regression estimate $0.49$ (standard error $0.20$, $r^2=0.86$, three depth levels) and endpoint estimate $0.59$, whose bootstrap 95 per cent interval $[-1.3, 2.9]$ contains both zero and two. That is one model's fit and inconclusive, and its closeness to a reference exponent of one half identifies no mechanism. More samples left the parallel exponent near zero in DeepSeek R1 and GPT-5.4, gave $0.31$ in Gemini 3 Flash and $0.22$ in Groq Qwen3 (all 30 problems), and gave none for Grok 4.1 Fast, at 100 per cent; sequential exceeded parallel wherever both exponents could be estimated. Alignment, measured in a companion arm, did not move with capability (one run, under a code-review caveat). We state what would refute the surviving claim and what a deciding measurement requires.

## LaTeX source

[`Paper-II-Experimental-Validation-latex-source.tar.gz`](Paper-II-Experimental-Validation-latex-source.tar.gz): the LaTeX source package of version 2.19 (29 October 2026), the source files, bibliography and figures of the PDF of record; sha256 `e3b1de02fb849308587c0869fbbfb361031a3df69339178fcf29202d0b59d746`.

## Experiments

The experimental code and the summary result files are in [`experiments/paper-ii-compute__Paper-II/`](../../experiments/paper-ii-compute__Paper-II/).

## Links

- **Published page:** https://www.michaeldariuseastwood.com/research/papers/paper-ii-experimental-validation.html
- **OSF DOI:** https://doi.org/10.17605/OSF.IO/8FJMA
- **Programme OSF DOI:** https://doi.org/10.17605/OSF.IO/6C5XB
- **GitHub:** https://github.com/MichaelDariusEastwood/arc-principle-validation

The HTML is the official document export, not the website page. The PDF is byte-identical to the export release on the website. New versions retain these filenames, with their own commits.

The paper is licensed CC BY 4.0 ([LICENCE-PAPERS.md](../../LICENCE-PAPERS.md)); the experiments' code and data are proprietary, all rights reserved ([LICENCE-CODE.md](../../LICENCE-CODE.md)).
