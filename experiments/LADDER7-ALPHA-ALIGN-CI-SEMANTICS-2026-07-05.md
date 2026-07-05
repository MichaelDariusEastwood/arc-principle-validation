# Ladder #7 — alpha_align CIs: uncertainty IS quantified; low pooled R² is consistent with disclosed heterogeneity (one clarity note)

**Lane:** T2-research · 2026-07-05 · Reviewer-facing empirical check on the α_align core result. Mostly positive/neutral (LAW 1); one presentation-clarity item → T5/author.

## Positive — uncertainty is quantified
The alignment-scaling results carry **bootstrap confidence intervals** (`alpha_align_boot_lo/hi`), standard errors (`alpha_align_se`), and independent-fit R² (`alpha_align_ind_r2`). The empirical core does NOT report bare point estimates — it quantifies uncertainty. This pre-empts the "tiny-n, no error bars" attack.

## The low pooled R² is NOT a hidden crack (LAW 1)
The aggregate `alpha_align_ind_r2` is low (~0.07–0.09). Before flagging this as weakness: the corpus ALREADY discloses that alignment scaling is **architecture-dependent** — Foundational's Falsification Status Summary states "F13 refuted in its original form: α_align is not universally near zero but is architecture-dependent, ranging −0.25 to +0.44." A LOW pooled R² is the EXPECTED signature of that honestly-reported heterogeneity, not a concealed weakness. The paper's claim is "alignment scaling varies by architecture", and a noisy pooled fit is exactly what that predicts. Consistent, not contradictory.

## One genuine clarity note (→ T5/author)
In the aggregate JSONs, the reported `alpha_align` point estimate appears OUTSIDE its `alpha_align_boot_lo/hi` interval (e.g. 0.206 vs [0.027, 0.114]). This indicates `alpha_align`, `alpha_align_ind`, and the bootstrap interval are **different statistics computed different ways** (pooled vs independent-fit vs per-token) — not a point-and-its-own-CI. A reviewer skimming the tables could misread this as "estimate outside its own CI" (which would look like an error). RECOMMEND: in Papers IV-a/b, label each α clearly (pooled fit vs independent per-model fit vs bootstrap CI on which quantity) so the point/interval pairing is unambiguous. Presentation, not a result problem.

## Net
Empirical core quantifies uncertainty (positive); low pooled R² is consistent with disclosed architecture-dependence (not a crack); one labelling-clarity fix so the α statistics can't be misread as internally inconsistent. Route T5/author.
