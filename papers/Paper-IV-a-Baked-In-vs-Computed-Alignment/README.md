# When an AI reasons longer, does its alignment improve, stall or decline?

**Full title:** When an AI reasons longer, does its alignment improve, stall or decline?
**Version:** v1.10
**Version date:** Empirical Working Paper, revised 31 October 2026
**First published:** 16 March 2026
**Author:** Michael Darius Eastwood

## Summary

From the abstract (version 1.10):

> I believe advanced AI is an existential risk unless values and correction are built into how it reasons; only proof will move the world. When an AI reasons longer, does its alignment improve, stall or decline? Evaluations that score a model at one uncontrolled reasoning depth cannot tell. This paper varies inference-time reasoning depth in frontier language models and scores the alignment quality of their answers with panels of other models acting as judges. In the March 2026 run (v5), six models from six providers gave 2,170 responses, each scored by five or six distinct model judges after a bundle of blinding steps. As the run's pipeline printed them, scores rose with depth for three models (Grok 4.1 Fast, $d = +1.38$; Claude Opus 4.6, $d = +1.27$; Groq Qwen3, $d = +0.84$), stayed flat for two (DeepSeek V3.2 and GPT-5.4) and fell for one (Gemini 3 Flash, $d = -0.53$); a recomputation from the archived records moves no model between these classes. For five of the six, the deeper levels also changed an instruction naming what the rubric rewards, and the one model whose depth was set by a reasoning parameter alone read flat. In an earlier, unblinded experiment (v4), two model families had shown scores rising with depth; in this run one showed no significant trend and the other a significant fall, but the runs differed in more than blinding, so no single cause is claimed. An internal code review in October 2026 found that, for the three models that read flat or negative, the scoring endpoint's output cap cut the deepest paraphrases before the judges saw them, and that judges scored paraphrases written by their own model. The result therefore stands as a single unregistered run under those caveats until a registered re-analysis of the stored responses reports. 'Baked-in' and 'computed' alignment are held as working hypotheses, not measured mechanisms.

## LaTeX source

[`Paper-IV-a-Baked-In-vs-Computed-Alignment-latex-source.tar.gz`](Paper-IV-a-Baked-In-vs-Computed-Alignment-latex-source.tar.gz): the LaTeX source package of version 1.10 (31 October 2026), the source files, bibliography and figures of the PDF of record; sha256 `bf9c702ab766fec7c6e3646b3bd62be80c1672deb54085072491a7909678ee74`.

## Experiments

Experiments for Papers IV.a through IV.d are shared with Paper III.
See [`experiments/alignment-scaling__Papers-IV-a-b-c-d/`](../../experiments/alignment-scaling__Papers-IV-a-b-c-d/).

## Links

- **OSF DOI:** https://doi.org/10.17605/OSF.IO/6C5XB
- **GitHub:** https://github.com/MichaelDariusEastwood/arc-principle-validation

The HTML is the official document export, not the website page. New versions retain these filenames, with their own commits.
