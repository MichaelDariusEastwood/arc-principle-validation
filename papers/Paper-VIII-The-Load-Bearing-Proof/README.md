# The Load-Bearing Test: is AI safety a tax on capability, or part of what holds it up?

**Version:** 3.8
**Revised:** 9 November 2026
**First published:** 2026-03-18
**Author:** Michael Darius Eastwood

## Summary

From the abstract (version 3.8):

> I believe advanced AI is an existential risk unless values and correction are built into how it reasons; only proof will move the world. Safety is often taken to cost capability, which in a race to be first gives developers a reason to defer it. Is the cost real, or can safety be part of what holds capability up? Three separate experiments look for it. In the first, a Darwin Gödel Machine on DeepSeek V3.2 evolved 75 agents under three selection conditions, scored by GPT-5.4, a blinded judge from another provider: no difference was detected between the conditions ($p$ from 0.28 to 0.74, Mann-Whitney U), so the safety gate cost no measurable capability and brought no measurable benefit. In the second, LoRA fine-tuning of Qwen 2.5 3B Instruct under capability, safety-proxy and entangled losses, with 9 and then 295 training examples, left every fine-tuned condition below the base model on capability (7.68 against at best 4.00 at the larger scale), so the entangled objective’s capability cost could not be determined; on the paper’s candidate account, the base model’s existing RLHF training is too strong for fine-tuning at this scale. In the third, a gated self-modification simulation in a toy system, the unconstrained arm gained capability and lost safety (here, retention of earlier tasks: 0.718 and 0.933 against the static baseline’s 0.687 and 0.956), while the gated arm stayed above the baseline on capability ($0.696 \pm 0.014$) with safety $0.947 \pm 0.025$, three seeds an arm and no difference tested; all of it is simulation output, prediction-side and never an empirical outcome. One null, one inconclusive and one simulation result leave the safety-capability trade-off unproven at the scale tested, and whether embedded safety brings a measurable benefit open. The paper informs Law V, the ARC Embedding Law, a law under test, and decides none of its propositions.

From the paper (version 3.8):

> Every law is a law under test until a registered test returns its verdict. Laws IV and V were added on 24 September 2026 and registered as laws on 15 October 2026 in version 1.103 of the theory registration, and their mathematics is in development.

## Status and changes

Version 3.8 sets the paper on the programme’s presentation standard; calls its experiments separate and its judge blinded, from another provider; states what each experiment’s safety measure is, and describes Experiment 1’s gate and Experiment 3’s drag control as the code runs them; reports Experiment 1’s null as no difference detected, and notes that no arm varied recursive depth; corrects the epigraphs, Figure 1 and the dating of the code; and brings the laws’ status up to date. Internally source-checked and read in rendered form; not independent peer review.

The complete historical material and correction record remain in the lettered appendices.

## Files and sources

- [PDF](Paper-VIII-The-Load-Bearing-Proof.pdf)
- [Document HTML](Paper-VIII-The-Load-Bearing-Proof.html)
- [Text](Paper-VIII-The-Load-Bearing-Proof.txt)
- [Citation metadata](Paper-VIII-The-Load-Bearing-Proof.cff)
- [LaTeX source](Paper-VIII-The-Load-Bearing-Proof-latex-source.tar.gz)
- [OSF record](https://doi.org/10.17605/OSF.IO/7YJ4E)

These are the official document exports. Stable filenames preserve earlier versions in file history. Written research is CC BY 4.0; separately identified code, data and source records retain their stated licences.
