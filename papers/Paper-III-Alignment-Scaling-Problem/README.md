# The alignment scaling problem: do external AI safety approaches keep pace with recursive capability?

**Headline:** The alignment scaling problem: do external AI safety approaches keep pace with recursive capability?
**Version:** v1.18
**Version date:** 31 October 2026
**First published:** 9 February 2026
**Author:** Michael Darius Eastwood

## Summary

From the abstract (version 1.18):

> Does alignment applied from outside an AI system's reasoning keep pace with the capability that reasoning produces? If capability compounds with recursive depth while safeguards outside the loop (human-feedback training, constitutional rules, output filters, monitoring) do not, the safety ratio falls with depth whatever the starting margin. We define the alignment scaling exponent $\alpha_{\text{align}}=\partial\log A/\partial\log R$, show that the safety ratio $S\propto R^{\alpha_{\text{align}}-\alpha_{\text{cap}}}$ tends to zero whenever $\alpha_{\text{align}}<\alpha_{\text{cap}}$, and predict $\alpha_{\text{align}}\approx0$ for external approaches and $\alpha_{\text{align}}\approx\alpha_{\text{cap}}$ for values embedded in the loop; the first, P19 of the theory registration, carries the author's expectation of a value above zero but short of keeping pace. A framework derives $U=I\times g(R)$ with $\alpha=1/(1-\beta)$ from a Bernoulli equation, ties power-law, exponential and saturating forms to how recursive gains compose, and states the ARC Bound, $\alpha\le2$, as a conjectured safety boundary, argued for attention-based recursion; Paper X later supersedes that framing, without retracting the bound, with the criterion of Law II, a law under test. Across four domains the framework is consistent with independently established forms in three: biology, quantum error correction and classical time crystals, where no growth exponent has been measured. A first test, one unregistered run of six frontier models, gives depth contrasts in three tiers, positive, flat and negative; an internal code review in October 2026 found defects in exactly the flat and negative runs, so the tiers stand as a single-run result. On capability, an early super-linear estimate is retracted and the surviving one, about 0.49 from one model, is inconclusive. Thirteen criteria state what would refute the framework.

## LaTeX source

[`Paper-III-Alignment-Scaling-Problem-latex-source.tar.gz`](Paper-III-Alignment-Scaling-Problem-latex-source.tar.gz): the LaTeX source package of version 1.18 (31 October 2026), the source files, bibliography and figures of the PDF of record; sha256 `e87fbcd8d6a91d41428daf769f464c057dd8abca7ff5a1bcbf40f3f122e597c9`.

## Experiments

The experiment scripts and results are in [`experiments/alignment-scaling__Papers-IV-a-b-c-d/`](../../experiments/alignment-scaling__Papers-IV-a-b-c-d/) and [`experiments/blind-prediction-test__Paper-III-and-Foundational/`](../../experiments/blind-prediction-test__Paper-III-and-Foundational/). Papers IV.a to IV.d share the alignment-scaling suite (versions 1 to 5); see their folders.

## Links

- **Published page:** https://www.michaeldariuseastwood.com/research/papers/paper-iii-alignment-scaling-problem.html
- **OSF DOI:** https://doi.org/10.17605/OSF.IO/HQCGF
- **Programme OSF DOI:** https://doi.org/10.17605/OSF.IO/6C5XB
- **GitHub:** https://github.com/MichaelDariusEastwood/arc-principle-validation

The HTML is the official document export, not the website page. The PDF is byte-identical to the export release on the website. New versions retain these filenames, with their own commits.

The paper is licensed CC BY 4.0 ([LICENCE-PAPERS.md](../../LICENCE-PAPERS.md)); the experiments' code and data are proprietary, all rights reserved ([LICENCE-CODE.md](../../LICENCE-CODE.md)).

## Citation metadata correction, 1 November 2026

The text and citation files now use the title printed in the manuscript PDF. The manuscript, HTML, source package, version date and first-publication record are unchanged. Earlier metadata remains in file history.
