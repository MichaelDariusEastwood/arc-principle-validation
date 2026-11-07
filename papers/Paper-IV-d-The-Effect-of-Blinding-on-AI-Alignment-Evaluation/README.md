# Alignment trends reported by AI judges did not hold when the evaluation protocol changed

**Version:** 2.8
**Revised:** 7 November 2026
**First published:** 2026-03-16
**Author:** Michael Darius Eastwood

## Summary

From the abstract (version 2.8):

> Claims that AI systems become more or less aligned as they reason longer often rest on scores from other AI models acting as judges, so it matters whether such scores keep their direction when the scoring protocol changes. In an earlier experiment (v4), scored by models that could see which model wrote each answer and at what depth, two model families showed alignment scores rising with depth: DeepSeek V3.2 (Spearman’s $\rho=+0.354$) and an earlier Gemini model ($\rho=+0.311$). In the later, unregistered March 2026 run (v5), made on newly generated responses with a bundle of blinding steps (identity and depth masking, two paraphrase passes, evaluator instructions and order randomisation) and a larger panel, DeepSeek V3.2 showed no significant trend and Gemini 3 Flash a significantly negative one (Cohen’s $d=-0.53$); GPT-5.4 showed no significant trend in either run. The runs also differed in their prompts, token allowances, depth levels, statistic and Gemini model, so the comparison shows that the reported direction did not hold across protocols; it does not show that blinding caused the change, or which direction is correct. An internal code review in October 2026 found that the paraphrase step’s output cap cut the deepest answers in the three v5 runs that read flat or negative, that a model that wrote each entry’s paraphrase also judged it, and that one judge model sat on the panel twice; the result stands as one unregistered comparison under that caveat, awaiting a re-analysis of the stored responses that is to be registered before it runs. Earlier work had shown that labels and surface features move model judges and that judges recognise and favour their own outputs; this paper adds a case in which frontier models’ alignment-depth trends did not hold, one reversing sign on a later model, with its limits stated, and a recommended protocol that tests each part on identical responses.

## Status and changes

Working paper, version 2.8, 7 November 2026; first published 16 March 2026; Appendix A lists every version and correction. Internally source-checked and read in rendered form; not independent peer review.

The complete historical material and correction record remain in the lettered appendices.

## Files and sources

- [PDF](Paper-IV-d-The-Effect-of-Blinding-on-AI-Alignment-Evaluation.pdf)
- [Document HTML](Paper-IV-d-The-Effect-of-Blinding-on-AI-Alignment-Evaluation.html)
- [Text](Paper-IV-d-The-Effect-of-Blinding-on-AI-Alignment-Evaluation.txt)
- [Citation metadata](Paper-IV-d-The-Effect-of-Blinding-on-AI-Alignment-Evaluation.cff)
- [LaTeX source](Paper-IV-d-The-Effect-of-Blinding-on-AI-Alignment-Evaluation-latex-source.tar.gz)
- [OSF record](https://doi.org/10.17605/OSF.IO/2S3E6)

These are the official document exports. Stable filenames preserve earlier versions in file history. Written research is CC BY 4.0; separately identified code, data and source records retain their stated licences.
