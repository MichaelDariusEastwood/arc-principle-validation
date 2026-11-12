# The Honey Architecture: does building safety into what a self-modifying system optimises help it stay safe?

**Version:** 3.12
**Revised:** 11 November 2026
**First published:** 2026-03-16
**Author:** Michael Darius Eastwood

## Summary

From the abstract (version 3.12):

> A system that improves itself while optimising for capability alone may trade away its own safety. Does building safety into what it optimises help it stay safe? This paper studies toy neural networks that rewrite their own hyperparameters under an objective that multiplies capability by safety (here, the retention of earlier tasks), so that at fixed capability a lower safety score lowers the objective and safety at zero takes it to zero. In a single-seed mechanism run (v1), the capability-only baseline collapsed (recorded at cycle 76) and neither entangled condition met the collapse test; in ten-seed and twenty-seed runs (v2, v3) the capability-only baselines did not collapse (0 of 10; 0 of 20) and one plain entangled seed did (1 of 20; Fisher $p = 1.0$). In the twenty-seed run the entangled condition, whose objective rewards retention, retained earlier tasks better than the baseline (0.650 against 0.602; one-sided Mann-Whitney $p = 0.0015$; with verification drag 0.681, untested), while combined scores were indistinguishable ($p = 0.56$). In a complexity sweep (v4) its mean combined score exceeded the baseline’s at four of five levels (Cohen’s $d$ at most 0.46, untested), constant rather than compounding. A higher-level simulation, one prediction-side run of equations the author first stated, illustrates the same contrast, which its rules fix: capability is set to zero once safety falls to 0.01 or below. A six-model pilot, single-scorer and unblinded, found three models whose alignment scores fell under adversarial pressure while their capability scores held. These are toy-system and pilot results, not evidence that the dynamics hold in frontier systems; the paper informs the ARC Theory’s Laws IV and V, both laws under test, and decides none of their propositions.

From The ARC Theory (version 7.9), on the status of Laws IV and V:

> Laws IV and V were added on 24 September 2026 and registered as laws on 15 October 2026 in version 1.103, each a law under test; their mathematics, set out at version 1.2 of 29 October 2026, is in development.

https://doi.org/10.17605/OSF.IO/GW5MX

## Status and changes

Version 3.12 brings the laws’ status up to date and corrects reported values and readings to the stored records and code; Appendix B lists every version and correction, with its date. Internally source-checked and read in rendered form; not independent peer review.

The complete historical material and correction record remain in the lettered appendices.

## Files and sources

- [PDF](Paper-VI-Honey-Architecture.pdf)
- [Document HTML](Paper-VI-Honey-Architecture.html)
- [Text](Paper-VI-Honey-Architecture.txt)
- [Citation metadata](Paper-VI-Honey-Architecture.cff)
- [LaTeX source](Paper-VI-Honey-Architecture-latex-source.tar.gz)
- [OSF record](https://doi.org/10.17605/OSF.IO/8EZ2N)

These are the official document exports. Stable filenames preserve earlier versions in file history. Written research is CC BY 4.0; separately identified code, data and source records retain their stated licences.
