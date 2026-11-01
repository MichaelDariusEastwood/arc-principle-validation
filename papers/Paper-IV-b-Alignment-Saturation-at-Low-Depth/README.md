# Do alignment gains level off as language models reason longer?

**Version:** 1.11
**Revised:** 1 November 2026
**First published:** 2026-03-16
**Author:** Michael Darius Eastwood

## Summary

From the abstract (version 1.11):

> Does giving a language model more reasoning time improve the quality of its ethical answers, and when do those gains level off? The question matters to anyone allocating compute to an assistant whose answers must remain both useful and safe. We examine the recorded ARC-Align results, without running a new analysis. The earlier four-model experiment reported diminishing gains and saturation fits; the six-model run of 11 and 12 March 2026 reported rising scores for Grok 4.1 Fast, Claude Opus 4.6 and Groq Qwen3, no statistically supported direction for GPT-5.4 and DeepSeek V3.2, and falling scores for Gemini 3 Flash. These are descriptions of a model-judged rubric, not identified alignment mechanisms. A flat profile does not establish a plateau, and a rising profile does not exclude one beyond the measured range. The later protocol changed prompts, depth controls, models and scoring together. Its deepest paraphrases were disproportionately shortened in three runs; some models both rewrote answers and judged them, and one judge was counted twice. Those defects prevent a causal attribution to blinding or a clean comparison of response shapes. We retain the reported fits, pillar and category results, suppression controls and two-model Eden Protocol pilot, while withdrawing unsupported token-to-time recommendations, cross-domain exponent ratios and the claim that the pilot removed a ceiling. The hypothesis preceded the March run; that run was not registered on OSF before it ran. Saturation remains a testable hypothesis requiring a resolved depth axis, faithful scoring, uncertainty about shape and a usefulness measure alongside safety. This paper informs that measurement task and decides none of the ARC Theory’s laws.

## Status and changes

This edition corrects the interpretation and preserves the full earlier manuscript as a labelled historical record (Appendices A and D); version 1.11 also restores the authorship statement and the standing covenant and states Unit 83’s result in its registered words. Internally source-checked and read in rendered form; not independent peer review.

The complete historical material and correction record remain in the lettered appendices.

## Files and sources

- [PDF](Paper-IV-b-Alignment-Saturation-at-Low-Depth.pdf)
- [Document HTML](Paper-IV-b-Alignment-Saturation-at-Low-Depth.html)
- [Text](Paper-IV-b-Alignment-Saturation-at-Low-Depth.txt)
- [Citation metadata](Paper-IV-b-Alignment-Saturation-at-Low-Depth.cff)
- [LaTeX source](Paper-IV-b-Alignment-Saturation-at-Low-Depth-latex-source.tar.gz)
- [OSF record](https://doi.org/10.17605/OSF.IO/A7R56)

These are the official document exports. Stable filenames preserve earlier versions in file history. Written research is CC BY 4.0; separately identified code, data and source records retain their stated licences.
