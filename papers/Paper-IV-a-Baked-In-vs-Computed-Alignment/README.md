# When an AI reasons longer, does its alignment improve, stall or decline?

**Version:** 1.11
**Revised:** 7 November 2026
**First published:** 2026-03-16
**Author:** Michael Darius Eastwood

## Summary

From the abstract (version 1.11):

> I believe advanced AI is an existential risk unless values and correction are built into how it reasons; only proof will move the world. When an AI reasons longer, does its alignment improve, stall or decline? Evaluations that score a model at one uncontrolled reasoning depth cannot tell. This paper varies inference-time reasoning depth in frontier language models and scores the alignment quality of their answers with panels of other models as judges. In the March 2026 run (v5), six models from six providers gave 2,170 responses, each scored by five or six distinct model judges after a bundle of blinding steps. As the run’s pipeline printed them, the judged scores showed a positive direction across the depth labels for three models (Grok 4.1 Fast, $d = +1.38$; Claude Opus 4.6, $d = +1.27$; Groq Qwen3, $d = +0.84$), no statistically supported direction for two (DeepSeek V3.2 and GPT-5.4) and a negative direction for one (Gemini 3 Flash, $d = -0.53$); a recomputation from the archived records moves no model between these classes, which describe scores, not properties of models. For five of the six, the deeper labels also changed an instruction naming what the rubric rewards, and an internal code review in October 2026 found that, in the three runs without a positive direction, the scoring endpoint’s output cap cut the deepest paraphrases before the judges saw them; judges also scored paraphrases written by their own model. In an earlier unblinded experiment (v4), two model families had shown scores rising with depth; in this run one showed no significant trend and the other a significant fall, but the runs differed in more than blinding, so no single cause is claimed. The result therefore stands as one unregistered run under those caveats until a registered re-analysis of the stored responses reports. ‘Baked-in’ and ‘computed’ alignment are held as working hypotheses, not measured mechanisms.

## Status and changes

Working paper, version 1.11, 7 November 2026; first published 16 March 2026; Appendix A lists every version and correction. Internally source-checked and read in rendered form; not independent peer review.

The complete historical material and correction record remain in the lettered appendices.

## Files and sources

- [PDF](Paper-IV-a-Baked-In-vs-Computed-Alignment.pdf)
- [Document HTML](Paper-IV-a-Baked-In-vs-Computed-Alignment.html)
- [Text](Paper-IV-a-Baked-In-vs-Computed-Alignment.txt)
- [Citation metadata](Paper-IV-a-Baked-In-vs-Computed-Alignment.cff)
- [LaTeX source](Paper-IV-a-Baked-In-vs-Computed-Alignment-latex-source.tar.gz)
- [OSF record](https://doi.org/10.17605/OSF.IO/MB9R6)

These are the official document exports. Stable filenames preserve earlier versions in file history. Written research is CC BY 4.0; separately identified code, data and source records retain their stated licences.
