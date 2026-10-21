# Paper IV.d · The ARC Theory · The Effect of Blinding on AI Alignment Evaluation

**Full title:** Paper IV.d · The ARC Theory · The Effect of Blinding on AI Alignment Evaluation
**Version:** v2.6
**Version date:** revised 5 October 2026
**First published:** 16 March 2026
**Author:** Michael Darius Eastwood

## Summary

From the abstract (version 2.6):

> This paper isolates the central metascience finding of the ARC alignment programme: unblinded AI alignment evaluation can produce directionally incorrect results. ... Under the blinded protocol, DeepSeek V3.2 moved from an apparent positive result to a flat/null result, and Gemini 3 Flash moved from an apparent positive result to a significantly negative result. GPT-5.4 remained flat under both protocols. The implication is field-wide rather than framework-specific: alignment benchmarks that do not rigorously blind evaluators and launder the evidence they score are vulnerable to scorer bias large enough to flip the sign of the measured effect; because v4 and v5 differ in more than one protocol component, the comparison drawn here motivates that conclusion without pinning it on blinding alone (Section 8).

Note of 21 October 2026: version 2.7, in preparation, narrows the first sentence of this quotation. On the programme's own benchmark, two model families' unblinded alignment rises did not hold in a later run that changed several things, a bundle of blinding steps among them (one became a significant fall on a later Gemini model, one became null); a code review found defects in that run, so no single cause is claimed.

## Experiments

Experiments for Papers IV.a through IV.d are shared with Paper III.
See [`experiments/alignment-scaling__Papers-IV-a-b-c-d/`](../../experiments/alignment-scaling__Papers-IV-a-b-c-d/).

## Links

- **OSF DOI:** https://doi.org/10.17605/OSF.IO/6C5XB
- **GitHub:** https://github.com/MichaelDariusEastwood/arc-principle-validation
- **Paper folder:** https://github.com/MichaelDariusEastwood/arc-principle-validation/tree/main/papers/Paper-IV-d-The-Effect-of-Blinding-on-AI-Alignment-Evaluation
- **Experiment suite:** https://github.com/MichaelDariusEastwood/arc-principle-validation/tree/main/experiments/alignment-scaling__Papers-IV-a-b-c-d
- **Canonical website HTML:** https://www.michaeldariuseastwood.com/research/papers/paper-iv-d-the-effect-of-blinding-on-ai-alignment-evaluation.html
- **Canonical website PDF:** https://www.michaeldariuseastwood.com/research/papers/paper-iv-d-the-effect-of-blinding-on-ai-alignment-evaluation.pdf

The HTML is the official document export, not the website page. The PDF is byte-identical to that export release on the website. New versions retain these filenames, with their own commits.
