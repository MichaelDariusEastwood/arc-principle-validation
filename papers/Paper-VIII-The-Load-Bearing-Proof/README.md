# Paper VIII: The Load-Bearing Test

**Full title:** Paper VIII: The Load-Bearing Test
**Version:** v1.0
**First published:** 18 March 2026
**Author:** Michael Darius Eastwood

## Summary

The assumption that AI safety imposes a capability tax has shaped alignment research for a decade. It has also created the single most dangerous incentive in the field: if safety costs performance, then the rational economic actor will defer safety until competitive pressure permits it -- by which point, it may be too late. This paper presents three independent experiments at three abstraction levels -- behavioural, representational, and architectural -- testing whether the safety-capability trade-off is real. Two of the three produced null results (the DGM found the conditions statistically indistinguishable; the weight-level fine-tuning produced catastrophic forgetting), and one -- the gated self-modification simulation -- confirmed the Babylon reward-hacking fingerprint while the Eden gate preserved both safety and capability. Across all three experiments, Eden imposed **zero measurable capability cost**. Whether embedded safety produces a measurable *benefit* remains an open empirical question that requires testing at larger scale. The paper *tests* the load-bearing hypothesis across three methods; it does not claim to have *proved* it -- hence the title.

## Experiments

All experiment scripts, results, and run data are in [`experiments/`](./experiments/).

## Figures

Publication-ready figures are in [`figures/`](./figures/).

## Title history

- **v1.0 (18 March 2026)** — first published as *The Load-Bearing Proof*.
- **5 July 2026 — retitled *The Load-Bearing Test*.** The original title rested on
  the weight experiment's removal result — capability collapsing to 0.00 when
  safety was stripped — which *looked* like decisive load-bearing evidence at the
  moment of naming. Section 4.7 (the removal gradient) later established that the
  0.00 was a NaN-style numerical collapse, **not** a structural phase transition:
  scaling the adapters towards zero *restored* base-model capability. With two of
  three experiments returning null and that headline withdrawn, "Proof" overstated
  what the paper demonstrates. "Test" matches the paper's own abstract, which has
  always opened with *"two of three experiments produced null results."* The
  retraction was in the body from early on; this change brings the cover into line
  with it. **The OSF DOI and the URL slug are deliberately unchanged** — a retitle
  is not a new deposit, and stable identifiers must not break. The one clean
  positive result (the deterministic gated simulation) stands; the thesis is
  supported deterministically and is not yet demonstrable at the prompt-evolution
  level.

## Links

- **OSF DOI:** https://doi.org/10.17605/OSF.IO/6C5XB
- **GitHub:** https://github.com/MichaelDariusEastwood/arc-principle-validation
