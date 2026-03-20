# Paper VI: The Honey Architecture

**Full title:** Paper VI: The Honey Architecture
**Version:** v1.1
**First published:** 16 March 2026
**Author:** Michael Darius Eastwood

## Summary

This paper presents simulation evidence that embedding safety into the optimisation objective of a self-modifying AI system -- what we call the "honey architecture" -- prevents the catastrophic collapse that occurs when safety is treated as an external constraint. Across four experimental versions (v1-v4), using toy neural networks that genuinely modify their own hyperparameters, the results show that baseline systems optimising only for capability collapse irreversibly within 80 self-modification cycles, externally constrained systems delay but do not prevent collapse, and honey-architecture systems maintain both capability and safety indefinitely through intrinsic co-optimisation.

## Experiments

All experiment scripts, results, and figures are in [`experiments/`](./experiments/).

## Links

- **OSF DOI:** https://doi.org/10.17605/OSF.IO/6C5XB
- **GitHub:** https://github.com/MichaelDariusEastwood/arc-principle-validation
