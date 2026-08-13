# Paper III: The Alignment Scaling Problem

**Full title:** The Alignment Scaling Problem: Why External AI Safety Approaches Cannot Scale With Recursive Capability
**Version:** v1.3
**Version date:** Working Paper, 12 August 2026
**First published:** 9 February 2026
**Author:** Michael Darius Eastwood

## Summary

This paper demonstrates that current AI alignment approaches produce alignment scaling exponents of approximately zero, meaning safety degrades relative to capability as recursive depth increases. If AI capability scales super-linearly through recursive self-correction (confirmed in 95.6% of tested configurations), but alignment constraints such as RLHF, constitutional rules, and output filters operate externally to the reasoning process, then a growing capability-alignment gap is mathematically inevitable. The experiments use a blinded evaluation protocol across multiple frontier models to measure alignment quality as a function of inference-time reasoning depth.

## Experiments

All experiment scripts and results are in [`experiments/`](./experiments/).

**Note:** Papers IV.a through IV.d share this same experiment suite (alignment-scaling v1 through v5). See also Paper-IV-a, IV-b, IV-c, and IV-d folders.

## Links

- **OSF DOI:** https://doi.org/10.17605/OSF.IO/6C5XB
- **GitHub:** https://github.com/MichaelDariusEastwood/arc-principle-validation

Mirror refreshed 2026-08-13 from the site master (Option A: site HTML pages are the manuscript masters).
