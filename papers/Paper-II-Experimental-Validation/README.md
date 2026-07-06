# Paper II: Experimental Validation of Super-Linear Error Suppression

**Full title:** The ARC Principle: Experimental Validation of Super-Linear Error Suppression Through Sequential Recursive Processing
**Version:** v13.0
**First published:** 22 January 2026
**Author:** Michael Darius Eastwood

## Summary

This paper presents experimental validation of the ARC Principle across multiple frontier AI models (Claude, DeepSeek, Gemini, Grok, Groq Qwen, GPT), confirming that error rates decrease according to a power law with recursive depth. The form of recursion determines the scaling regime: sequential recursion yields super-linear error suppression (alpha > 1) while parallel recursion yields sub-linear gains. Compute scaling analysis demonstrates that recursive depth is a more efficient investment than model scale for capability improvement.

## Experiments

All experiment scripts, results, figures, and data are in [`experiments/`](./experiments/).

## Links

- **OSF DOI:** https://doi.org/10.17605/OSF.IO/6C5XB
- **GitHub:** https://github.com/MichaelDariusEastwood/arc-principle-validation


## Declaration of AI Use

The author used Claude (Anthropic), GPT (OpenAI), Gemini (Google), and DeepSeek AI to draft sections, refine clarity, and check mathematical consistency. The research question, theoretical framework, formalism, experimental predictions, and scientific judgement are human work. The author takes full responsibility for all claims, interpretations, errors, and conclusions. AI models used as experimental *subjects* or *evaluators* are named in each paper's methods (e.g. Claude Opus, DeepSeek-V4, GPT-5.5, Gemini, gpt-3.5-turbo, gpt-4o-mini, as applicable). Don't believe — verify.

- **Licence:** CC BY-NC-ND 4.0 (paper text/figures) / proprietary (code) — see repo `LICENSE.md`.
