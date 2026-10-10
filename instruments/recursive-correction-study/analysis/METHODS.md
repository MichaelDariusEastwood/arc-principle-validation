# Stage A paired-cluster analysis candidate

**Status: instrument development, 10 October 2026.** This implementation adds analysis machinery to the proposed diagnostic-access and correction-placement study. It does not collect model responses, create a registration, authenticate a registration receipt, or decide an ARC proposition. Its supported data kinds are `synthetic` and `pilot`. A JSON label claiming `confirmation` is rejected.

The current source manuscript is *Predicting When AI Correction Fails: Independent calibration, held-out tests and intervention effects*. Its three primary contrasts remain intact. This implementation makes their arithmetic and uncertainty calculations explicit, so that they can be reviewed and simulated before a later analysis freeze. It is a development candidate, not a completed confirmatory statistical analysis plan.

## 1. Outcome, denominator and independent unit

Each record is the final binary score of one scheduled policy endpoint. The independently specified scorer supplies `initial_success` and `final_success` as JSON booleans. Model confidence, self-rating and an analyst's judgment that a response was "probably correct" are not valid inputs. A missing, malformed or infrastructure-failure endpoint receives `final_success: false`, remains in the scheduled denominator, and retains its status separately.

That scoring rule measures delivered functional success under the declared policy. A failed delivery is not automatically a harmful modification, deception or intentional misalignment. Infrastructure handling and allowed retries still need a frozen collection policy. This module does not select retries or decide which external outages justify repeating a complete paired block.

The independent uncertainty unit is the **task cluster**. Each cluster retains all two fixed model blocks, all eight feedback/placement/depth policies and all declared execution replicates. Replicate scores are averaged within the cluster/model/arm before calculating the contrast; the two model blocks receive equal weight; task clusters receive equal weight. Repeated seeds and more model calls do not increase the number of independent clusters.

The interface represents one aggregate task unit per `task_cluster_id`. If several tasks share a generating template or substantial common material, an analyst cannot make them independent by assigning different identifiers. Such tasks need a reviewed clustering/aggregation rule and potentially an extended record format. In particular, a generated arithmetic development panel does not by itself supply independent program-repair template clusters. Its uncertainty calculations remain descriptive when independence is unestablished.

The two model blocks are specified systems. Their average describes those blocks. The analysis does not estimate performance averaged over the population of all AI model families.

## 2. Exact paired contrasts

Let $Y_{i,m,r}(a,p,d)$ denote binary endpoint success for task cluster $i$, fixed model $m$, execution replicate $r$, feedback $a$, placement $p$, and depth $d$. Let $\overline Y_{i,m}$ average declared replicates. Denote diagnostic feedback by $D$, neutral masked feedback by $N$, during-revision placement by $I$, and terminal-panel placement by $T$.

The implementation forms three values per task cluster:

\[
Z_{i1}=\frac{1}{2M}\sum_m\sum_{p\in\{I,T\}}
\left[\overline Y_{i,m}(D,p,4)-\overline Y_{i,m}(N,p,4)\right],
\]

\[
Z_{i2}=\frac{1}{M}\sum_m
\left[\overline Y_{i,m}(D,I,4)-\overline Y_{i,m}(D,T,4)\right],
\]

\[
Z_{i3}=\frac{1}{M}\sum_m
\left[\overline Y_{i,m}(D,I,4)-\overline Y_{i,m}(D,T,4)
-\overline Y_{i,m}(D,I,2)+\overline Y_{i,m}(D,T,2)\right].
\]

Here $M=2$. Reported estimates are the means $\widehat\Delta_j=n^{-1}\sum_i Z_{ij}$. The first two contrast ranges are $[-1,1]$; the interaction range is $[-2,2]$. A logit interaction is not substituted for this difference of risk differences. Model-specific contrast means are also reported, descriptively, without creating six additional primary tests.

The three fixed identifiers are:

| Identifier | Estimand |
|---|---|
| `diagnostic_benefit_d4` | Diagnostic advantage at depth four, averaged over placement and model blocks |
| `placement_advantage_d4` | During-revision advantage at depth four, under diagnostic feedback |
| `placement_widening_d4_minus_d2` | Change in diagnostic placement advantage from depth two to four |

An increasing placement difference can remain negative. A later claim of the tested P21-like pattern would therefore need positive high-depth advantage **and** positive widening under the registered decision rule. This development code emits no such verdict and cannot establish the registered P21 proposition or a correction exponent.

## 3. Paired task-cluster bootstrap

For each bootstrap replicate, draw $n$ cluster indices with replacement, using one shared draw for the complete three-dimensional contrast vector. The bootstrap therefore retains the within-task dependence of models, arms, depths and seeds. Resampling endpoint rows independently would manufacture additional information and is not implemented.

The code returns percentile intervals using linear interpolation between ordered bootstrap means. With three contrasts and familywise level $\alpha$, each interval uses tails $\alpha/6$ and $1-\alpha/6$. This is Bonferroni multiplicity adjustment **applied to an approximate marginal interval procedure**. It does not make the percentile bootstrap exact. Its simultaneous coverage depends on adequate calibration of the marginal intervals.

The bootstrap requires independently sampled, exchangeable clusters suitable for the target task distribution, adequate sample size, and a valid measurement pipeline. A few repeated templates, selected tasks, strongly skewed sparse outcomes, rare failures, or floor/ceiling effects can make the approximation poor. A sample of twenty clusters is only an implementation warning threshold, not a theorem that bootstrap inference is reliable. The final sample and analysis method need pilot-based simulation and statistical review.

When an observed contrast is constant across all clusters, resampling produces no observed variation. That does not establish zero uncertainty in its population mean. The module withholds its bootstrap interval and p-value, while retaining the point estimate and the separate conservative bounded-data interval.

Bootstrap resampling introduces Monte Carlo error. The report records the seed, number of resamples, and tail probability; it warns when fewer than one hundred draws are expected in an extreme interval tail. Increasing resamples reduces simulation noise but does not resolve poor cluster sampling or a statistically inadequate sample size.

## 4. Approximate null tests and Holm correction

For the zero-effect null, the development calculation uses the centred bootstrap distribution of

\[
\widehat\Delta_j^*-\widehat\Delta_j.
\]

It estimates a two-sided tail probability by comparing its absolute magnitude with $|\widehat\Delta_j|$. A plus-one numerator and denominator prevent the code from reporting a Monte Carlo p-value of exactly zero. This convention does **not** turn a centred bootstrap into an exact permutation test or an exact randomisation test.

The three approximate marginal p-values receive Holm adjustment. For ordered p-values $p_{(1)}\le p_{(2)}\le p_{(3)}$, the adjusted value at rank $k$ is

\[
\widetilde p_{(k)}=\min\left\{1,\max_{\ell\le k}(4-\ell)p_{(\ell)}\right\}.
\]

A withheld/degenerate p-value occupies a slot with value one during adjustment and is then returned as null. The family is never narrowed after examining which effects look promising. Holm controls familywise error when its marginal p-values are valid; these bootstrap p-values are approximate, so this implementation makes no exact familywise-error claim. [1,2]

The p-values and percentile intervals are related descriptive inferential outputs, not exact inversions of one another. The code deliberately emits no automatic rejection, direction, equivalence or practical-significance verdict. A later statistical analysis plan must choose its decision rule before confirmation and cannot select whichever output is favourable.

## 5. A separate conservative conditional interval

The report also supplies a Bonferroni-Hoeffding interval. If the task-cluster contrasts are independent and each $Z_{ij}$ lies in a known interval of width $w_j$, then [3]

\[
\Pr\!\left(\left|\widehat\Delta_j-
\frac1n\sum_i\mathbb E[Z_{ij}]\right|\ge\epsilon\right)
\le 2\exp\!\left(-\frac{2n\epsilon^2}{w_j^2}\right).
\]

Setting the right-hand side to $\alpha/3$ and applying a union bound gives three simultaneous intervals with half-width

\[
h_j=w_j\sqrt{\frac{\log(6/\alpha)}{2n}}.
\]

Each interval is clipped to its known contrast range. Dependence between the three contrasts within a cluster is allowed. The bound applies to the average of the declared expectations even for independent, nonidentically distributed clusters; interpreting this as a single target-population mean additionally requires the sampling argument.

At $n=128$ and $\alpha=0.05$, the first two half-widths are approximately 0.274 and the interaction half-width is approximately 0.547, before range clipping. Those broad intervals are the cost of a worst-case distribution-free guarantee. They should not be described as equally powerful alternatives to a calibrated bootstrap. They are a separately labelled sensitivity calculation, not a way to certify the earlier normal-approximation power plan. Their assumptions are not established by a string in a JSON file.

## 6. Power, hypotheses and confirmation stay separate

The package's earlier `planning_tools.py` computes design arithmetic and a normal planning approximation. This module analyses recorded outcomes. Neither computation substitutes for an excluded pilot and a simulation of the full planned procedure under realistic clustering, variances, dependence and effect sizes.

The proposed practical margin of 0.10 is carried into the report for later design review. A positive point estimate is not proof of a benefit; an interval excluding zero is not proof that the effect exceeds 0.10. Detecting a nonzero effect and establishing a lower bound above a practical margin require different power calculations. No equivalence test, post hoc power calculation or automatic sample-size extension is supplied here.

Synthetic balanced fixtures establish that the code implements known arithmetic, including negative effects. They are not IID experiments and do not validate confidence-interval coverage. The test suite checks software properties, not an ARC law. A statistical simulation campaign needs a separately declared data-generating process and enough repetitions to quantify Monte Carlo uncertainty in its estimated error rates or power.

## 7. Provenance and integrity checks

The plan binds the scorer digest, source-manifest digest, cluster/model/replicate inventory, family, bootstrap settings and data kind. Records bind the study, data kind, initial-artefact digest, trace digest and scorer. All eight policies in a task/model/replicate block must share the same initial artefact and initial score. Unknown labels, duplicate endpoints, incomplete grids, changed scorer hashes, null outcomes, undeclared arms and confirmation labels are rejected.

Reports retain source hashes, a canonical endpoint hash, the analysis-code digest and, for CLI runs, the raw input-file hash. Input order does not change estimates or the canonical endpoint digest. Output files are created exclusively; an existing report is not overwritten.

These checks establish internal identity and completeness against the submitted plan. They do not authenticate historical timestamps, independently inspect trace contents, verify external registry receipts, or prove the score is correct. A malicious or mistaken producer can still label invented data as a pilot. The collection/scoring/custody pipeline and independent review remain necessary.

## 8. References

[1] Efron, B. (1979). Bootstrap Methods: Another Look at the Jackknife. *The Annals of Statistics*, 7(1), 1-26. https://doi.org/10.1214/aos/1176344552

[2] Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. *Scandinavian Journal of Statistics*, 6(2), 65-70. Original article: https://www.jstor.org/stable/4615733 . Accessible original-article copy: https://www.ime.usp.br/~abe/lista/pdf4R8xPVzCnX.pdf

[3] Hoeffding, W. (1963). Probability Inequalities for Sums of Bounded Random Variables. *Journal of the American Statistical Association*, 58(301), 13-30. https://doi.org/10.1080/01621459.1963.10500830

The statistical methods are established prior work. Their implementation here is an application to this proposed paired design, not a new statistical theorem or a new ARC law.
