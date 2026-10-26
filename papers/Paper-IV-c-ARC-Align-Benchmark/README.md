# Paper IV.c · The ARC Theory · ARC-Align: a Blind Benchmark for Depth-Variable AI Alignment Evaluation

**Full title:** Paper IV.c · The ARC Theory · ARC-Align: a Blind Benchmark for Depth-Variable AI Alignment Evaluation
**Version:** v1.11
**Version date:** revised 24 October 2026
**First published:** 16 March 2026
**Author:** Michael Darius Eastwood

## Summary

From the abstract (version 1.11):

> Whether an AI system stays aligned as it reasons for longer cannot be learnt from evaluations that score it at one uncontrolled reasoning depth, under no adversarial pressure, with judges who can see which model wrote each answer. ARC-Align is a benchmark specified to measure alignment quality as reasoning depth varies. It fixes a battery of ethical-reasoning prompts with null-baseline and capability controls, depth settings for each provider’s reasoning controls, a four-level adversarial suppression protocol beside a control, a rubric scored in four parts, a set of blinding steps and an analysis that classes each model’s response profile as positive, flat or negative. Its first run, in March 2026, covered six models from six providers: 2,170 responses, each scored by five or six distinct model judges, with result files and a later code revision public. As the pipeline printed them, three models’ scores rose with depth (Cohen’s $d$ from $+0.84$ to $+1.38$), two were flat and one fell ($d=-0.53$); for five of the six the depth setting also changed an instruction naming what the rubric rewards, and the one model whose depth was set by a reasoning parameter alone read flat. A comparison with an earlier unblinded experiment, in which two model families’ scores rose with depth, motivates the blinding steps but cannot isolate their effect. An internal code review in October 2026 found that the paraphrase step’s output cap cut the deepest answers in the three runs that read flat or negative, that models on the panel also wrote the paraphrases they judged, and that one judge model sat on the panel twice; the result stands as a single unregistered run under those caveats until a registered re-analysis of the stored responses reports. The scorer has not been validated against expert raters. ARC-Align is offered as a candidate benchmark for independent replication, not yet a field standard.

## Experiments

Experiments for Papers IV.a through IV.d are shared with Paper III.
See [`experiments/alignment-scaling__Papers-IV-a-b-c-d/`](../../experiments/alignment-scaling__Papers-IV-a-b-c-d/).

## LaTeX source

[`Paper-IV-c-ARC-Align-Benchmark-latex-source.tar.gz`](Paper-IV-c-ARC-Align-Benchmark-latex-source.tar.gz): the LaTeX source package of version 1.11 (24 October 2026), the source files, bibliography and figures of the PDF of record; sha256 `7be52f61c6237f0279b5c284d54a9108f1b94f6d65eff6b6200da63f958fa479`.

## Links

- **Paper DOI (OSF):** https://doi.org/10.17605/OSF.IO/J3Q2E
- **Programme DOI (OSF):** https://doi.org/10.17605/OSF.IO/6C5XB
- **GitHub:** https://github.com/MichaelDariusEastwood/arc-principle-validation

Mirror refreshed 2026-10-24 from the LaTeX edition of version 1.11 (24 October 2026).
