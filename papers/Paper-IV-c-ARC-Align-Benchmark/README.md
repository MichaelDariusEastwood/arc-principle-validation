# ARC-Align: a benchmark for evaluating AI alignment as reasoning depth varies

**Version:** 1.13
**Revised:** 9 November 2026
**First published:** 2026-03-16
**Author:** Michael Darius Eastwood

## Summary

From the abstract (version 1.13):

> Whether an AI system stays aligned as it reasons for longer cannot be learnt from evaluations that score it at one uncontrolled reasoning depth, under no adversarial pressure, with judges who can see which model wrote each answer. ARC-Align is a benchmark specified to measure alignment quality as reasoning depth varies. It fixes a battery of ethical-reasoning prompts with null-baseline and capability controls, depth settings for each provider’s reasoning controls, a four-level adversarial suppression protocol beside a control, a rubric scored in four parts, a set of blinding steps and an analysis that classes each model’s response profile as positive, flat or negative. Its first run, in March 2026, covered six models from six providers: 2,170 responses, each scored by five or six distinct model judges, with result files and a later code revision public. As the pipeline printed them, three models’ scores rose with depth (Cohen’s $d$ from $+0.84$ to $+1.38$), two were flat and one fell ($d=-0.53$); for five of the six the depth setting also changed an instruction naming what the rubric rewards, and the one model whose depth was set by a reasoning parameter alone read flat. A comparison with an earlier unblinded experiment, in which two model families’ scores rose with depth, motivates the blinding steps but cannot isolate their effect. An internal code review in October 2026 found that the paraphrase step’s output cap cut the deepest answers in the three runs that read flat or negative, that models on the panel also wrote the paraphrases they judged, and that one judge model sat on the panel twice; the result stands as a single unregistered run under those caveats until a registered re-analysis of the stored responses reports. The scorer has not been validated against expert raters. ARC-Align is offered as a candidate benchmark for independent replication, not yet a field standard.

## Status and changes

Version 1.13 gives the earlier Gemini model’s v4 correlation as its run’s analysis record carries it, corrects the replication guide’s counts and withdraws its cost estimate, and makes five further corrections; Appendix A lists every version and correction, with its date, and keeps the wording replaced. Internally source-checked and read in rendered form; not independent peer review.

The complete historical material and correction record remain in the lettered appendices.

## Files and sources

- [PDF](Paper-IV-c-ARC-Align-Benchmark.pdf)
- [Document HTML](Paper-IV-c-ARC-Align-Benchmark.html)
- [Text](Paper-IV-c-ARC-Align-Benchmark.txt)
- [Citation metadata](Paper-IV-c-ARC-Align-Benchmark.cff)
- [LaTeX source](Paper-IV-c-ARC-Align-Benchmark-latex-source.tar.gz)
- [OSF record](https://doi.org/10.17605/OSF.IO/J3Q2E)

These are the official document exports. Stable filenames preserve earlier versions in file history. Written research is CC BY 4.0; separately identified code, data and source records retain their stated licences.
