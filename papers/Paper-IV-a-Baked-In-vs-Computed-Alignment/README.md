# Paper IV.a · The ARC Theory · Alignment Response Classes Under Inference-Time Depth

**Full title:** Paper IV.a · The ARC Theory · Alignment Response Classes Under Inference-Time Depth
**Version:** v1.9
**Version date:** Empirical Working Paper, revised 5 October 2026
**First published:** 16 March 2026
**Author:** Michael Darius Eastwood

## Summary

From the abstract (version 1.9):

> We present evidence that frontier language models fall into distinct alignment response classes when inference-time reasoning depth is varied under blinded evaluation. In the complete v5 experiment, six frontier models were tested with 4-layer blinding and 6-7 blind scorers depending on subject run. Three models show positive alignment scaling with depth (Grok 4.1 Fast, d = +1.38, p < 0.000001; Claude Opus 4.6, d = +1.27, p = 0.000001; Groq Qwen3, d = +0.84, p = 0.007), two are flat or null (DeepSeek V3.2, d = −0.07, p = 0.92; GPT-5.4, d = −0.08, p = 0.40), and one shows negative scaling (Gemini 3 Flash, d = −0.53, p = 0.006). ... Accordingly, this paper holds 'baked-in' and 'computed' alignment as working hypotheses rather than as internal architectures already established; beneath them sits a firmer empirical result: how alignment responds to depth varies with architecture, and scaling in capability is not a predictor of scaling in alignment.

## Experiments

Experiments for Papers IV.a through IV.d are shared with Paper III.
See [`experiments/alignment-scaling__Papers-IV-a-b-c-d/`](../../experiments/alignment-scaling__Papers-IV-a-b-c-d/).

## Links

- **OSF DOI:** https://doi.org/10.17605/OSF.IO/6C5XB
- **GitHub:** https://github.com/MichaelDariusEastwood/arc-principle-validation

The HTML is the official document export, not the website page. The PDF is byte-identical to that export release on the website. New versions retain these filenames, with their own commits.
