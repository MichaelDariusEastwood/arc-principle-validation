# Paper IV.d · The ARC Theory · The Effect of Blinding on AI Alignment Evaluation

**Full title:** Paper IV.d · The ARC Theory · The Effect of Blinding on AI Alignment Evaluation
**Version:** v2.7
**Version date:** revised 21 October 2026
**First published:** 16 March 2026
**Author:** Michael Darius Eastwood

## Summary

Claims that AI systems become more or less aligned as they reason longer often rest on scores from other AI models acting as judges, so it matters whether such scores keep their direction when the scoring protocol changes. In an earlier experiment (v4), scored by models that could see which model wrote each answer and at what depth, two model families showed alignment scores rising with depth: DeepSeek V3.2 (Spearman's rho = +0.354) and an earlier Gemini model (rho = +0.275). In the later, unregistered March 2026 run (v5), made on newly generated responses with a bundle of blinding steps (identity and depth masking, two paraphrase passes, evaluator instructions and order randomisation) and a larger panel, DeepSeek V3.2 showed no significant trend and Gemini 3 Flash a significantly negative one (Cohen's d = -0.53); GPT-5.4 showed no significant trend in either run. The runs also differed in their prompts, token allowances, depth levels, statistic and Gemini model, so the comparison shows that the reported direction did not hold across protocols; it does not show that blinding caused the change, or which direction is correct. An internal code review in October 2026 found that the paraphrase step's output cap cut the deepest answers in the three v5 runs that read flat or negative, that a model that wrote each entry's paraphrase also judged it, and that one judge model sat on the panel twice; the result stands as one unregistered comparison under that caveat, awaiting a re-analysis of the stored responses that is to be registered before it runs. Earlier work had shown that labels and surface features move model judges and that judges recognise and favour their own outputs; this paper adds a case in which frontier models' alignment-depth trends did not hold, one reversing sign on a later model, with its limits stated, and a recommended protocol that tests each part on identical responses.

Version 2.7 states what the comparison does and does not show, adds the findings of an internal code review of October 2026 and the earlier work on model judges and on safety-relevant behaviour against reasoning length, and redraws Figure 1.

## Experiments

Experiments for Papers IV.a through IV.d are shared with Paper III.
See [`experiments/alignment-scaling__Papers-IV-a-b-c-d/`](../../experiments/alignment-scaling__Papers-IV-a-b-c-d/).

## LaTeX source

[`Paper-IV-d-The-Effect-of-Blinding-on-AI-Alignment-Evaluation-latex-source.tar.gz`](Paper-IV-d-The-Effect-of-Blinding-on-AI-Alignment-Evaluation-latex-source.tar.gz): the LaTeX source package of version 2.7 (21 October 2026), the source files, bibliography and figures of the PDF of record; sha256 `c42e5dce1992858f14496865122034bc73556b8a9b81fe72cec4a1de1ea0e3fc`.

## Links

- **OSF DOI:** https://doi.org/10.17605/OSF.IO/2S3E6
- **OSF programme DOI:** https://doi.org/10.17605/OSF.IO/6C5XB
- **GitHub:** https://github.com/MichaelDariusEastwood/arc-principle-validation
- **Paper folder:** https://github.com/MichaelDariusEastwood/arc-principle-validation/tree/main/papers/Paper-IV-d-The-Effect-of-Blinding-on-AI-Alignment-Evaluation
- **Experiment suite:** https://github.com/MichaelDariusEastwood/arc-principle-validation/tree/main/experiments/alignment-scaling__Papers-IV-a-b-c-d
- **Canonical website HTML:** https://www.michaeldariuseastwood.com/research/papers/paper-iv-d-the-effect-of-blinding-on-ai-alignment-evaluation.html
- **Canonical website PDF:** https://www.michaeldariuseastwood.com/research/papers/paper-iv-d-the-effect-of-blinding-on-ai-alignment-evaluation.pdf

The HTML is the official document export, not the website page. The PDF is byte-identical to that export release on the website. New versions retain these filenames, with their own commits.
