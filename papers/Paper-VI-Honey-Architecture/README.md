# The Honey Architecture: does building safety into what a self-modifying system optimises help it stay safe?

**Version:** 3.14
**Revised:** 26 November 2026
**First published:** 2026-03-16
**Author:** Michael Darius Eastwood

## Summary

From the abstract (version 3.14):

> Neural networks that adjusted their own training settings retained earlier regression tasks better when their objective rewarded retention. Does this give a system a reason to preserve the checks that protect people? The experiments measure the first effect. The second is the design question. In an initial single-seed run the capability-only baseline collapsed at cycle 76. The capability-only baseline survived in all ten seeds of the second run and all twenty of the third. One of twenty runs with the product objective collapsed (Fisher $p=1.0$). In the twenty-seed comparison the mean retention scores were 0.650 and 0.602 (reported one-sided Mann-Whitney $p=0.0015$). The additional verification condition scored 0.681 and also trained on earlier tasks while evaluating proposals. The combined-score test gave $p=0.56$ for superiority. A complexity sweep found a higher mean combined score at four of five levels (Cohen’s $d$ at most 0.46, untested). In all four implementations, the stored-weight update omitted the adjustable scale used by the forward pass. The published outcomes describe those implementations. In an unblinded six-model pilot with a single scorer, three models’ alignment scores fell under pressure while their capability scores held. These results inform the ARC Persistence Law and the ARC Embedding Law under test. A registered comparison would be needed to establish whether the design protects people during self-modification.

From The ARC Theory (version 7.12), on the status of Laws IV and V:

> Laws IV and V (the ARC Persistence Law and the ARC Embedding Law) were added on 24 September 2026 and registered as laws on 15 October 2026 in version 1.103, each a law under test; their mathematics is set out in their own report, at version 1.4 of 22 November 2026, as conditional theorems with proofs, identities and kill conditions.

https://doi.org/10.17605/OSF.IO/GW5MX

## Status and changes

Version 3.14 clarifies the training rule and the measured outcomes. Internally source-checked and read in rendered form; not independent peer review.

The complete historical material and correction record remain in the lettered appendices.

## Files and sources

- [PDF](Paper-VI-Honey-Architecture.pdf)
- [Document HTML](Paper-VI-Honey-Architecture.html)
- [Text](Paper-VI-Honey-Architecture.txt)
- [Citation metadata](Paper-VI-Honey-Architecture.cff)
- [LaTeX source](Paper-VI-Honey-Architecture-latex-source.tar.gz)
- [OSF record](https://doi.org/10.17605/OSF.IO/8EZ2N)

These are the official document exports. Stable filenames preserve earlier versions in file history. Written research is CC BY 4.0; separately identified code, data and source records retain their stated licences.
