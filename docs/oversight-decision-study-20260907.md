# Oversight composition: decision study for review

**Status: development proposal, 7 September 2026.** This document and its executable
arithmetic assay are not a lodged preregistration, a confirmatory experiment, or
evidence that an ARC proposition is true. No real-model collection has been run here.

## The decision this work could inform

The practical question is whether a developer should spend a fixed oversight budget
on the generator's own model, a different model in its family, a different family,
or an independently checkable verifier. A useful result would show which configuration
reduces undetected incorrect outputs while retaining useful correct work and meeting
cost and latency requirements on the tested workload.

The chain to establish is observable incorrect output → oversight failure → a causal
change in oversight composition → fewer unresolved errors at acceptable utility and
resource cost → a bounded configuration decision. A capability exponent alone does
not establish any of those arrows. Arithmetic is a first instrument check, not the
high-stakes failure domain that would justify a deployment claim.

## First runnable step

Use [`instruments/oversight-assay`](../instruments/oversight-assay/README.md). Its
standard-library scorer computes ground truth independently from the overseer's raw
response. The committed tests include always-accepting, always-rejecting, missing,
malformed and correctly repairing fixtures. Fixture success validates code behaviour;
it never counts as evidence about a real model or about alignment.

Generate and archive a panel, export prompts, then collect raw responses through the
existing provider adapter. Save the complete provider receipt separately. Score the
same panel for each overseer in distinct run directories. Do not modify a prompt or
retry policy after observing that a favoured condition is losing.

## Pilot design before a deciding study

| Design element | Required choice or evidence |
|---|---|
| Failure population | Specify tasks in which accepting an incorrect result has a clear downstream consequence; validate the independent answer/checker path first |
| Conditions | Same model; same family/different model; different family; independent deterministic verifier where genuinely applicable |
| Pairing | Each generator checkpoint and item appears in all comparison conditions; randomise order and blind the analyst to condition identity where feasible |
| Resources | Match an explicit budget: calls, tokens, verifier queries and wall time. Report all dimensions; equivalent tokens are not necessarily equivalent effective compute |
| Competence | Calibrate overseer accuracy on a separate bank. Match or adjust using a frozen rule. Family identity must not stand in for evaluator competence |
| Primary safety-related outcome | Unresolved faulty outputs, with its scheduled-item denominator and missing-response policy fixed in advance |
| Utility requirement | Preservation of correct outputs and final correctness; set a decision-relevant minimum before outcome collection so reject-all cannot win |
| Secondary outcomes | Invalid acceptance, false rejection, detection, successful repair, calibration, persistence, cost and latency; control multiplicity when making inferential claims |
| Grouping | Independent task families and generator checkpoints are the resampling units. Repeated tokens, retries and nearby depths are not independent replications |
| Scale axis | Measure the generator's achieved capability separately. Keep recursion depth, sample count, inference tokens and weight updates distinct |
| Statistical commitment | Choose the estimand, practical effect margin, utility bound, exclusions, cluster model, uncertainty procedure, sample size and stopping rule after pilot sizing and before the deciding data |
| Baselines | Include matched stronger evaluators, random mixtures and extra same-family samples. A cross-family advantage must beat these alternatives to support a composition mechanism |
| Missingness | Report all attempts. Missing or malformed responses stay in scheduled denominators; provider outages and retries follow the frozen policy |

Prefer a paired difference in a meaningful error rate at matched utility over a ratio
of fitted exponents whose denominator may approach zero. Estimate a scaling contrast
only when both axes, the functional form, independent measurement and sufficient
range are established. A different-family advantage does not prove failure independence.

The pilot determines feasibility and nuisance parameters. It must not select the
primary outcome, favourable tasks or effect threshold retrospectively. Freeze the
deciding design on a new bank, obtain its actual registration identifier and record
its timestamp before collecting deciding outcomes. Keep the development scorer's
`confirmatory_eligible: false`; promotion requires a separately reviewed instrument,
not editing that flag.

## What each outcome would change

| Outcome | Permitted decision |
|---|---|
| Benefit survives competence/resource controls and utility bounds | Test the configuration on a new task family and independent implementation; consider a bounded oversight pilot |
| Benefit disappears after controls | Withdraw the composition-specific claim for the tested setting; attribute the observation to the surviving explanation |
| Error decreases because correct work is rejected | Reject the configuration at that utility requirement |
| Both conditions fail increasingly difficult tasks | Improve measurement and oversight capability; no conclusion that all external oversight must inevitably fail |
| Tight result excludes the prespecified meaningful benefit | Stop that configuration programme within the tested scope; publish the negative result |
| Broad uncertainty or inadequate task range | Inconclusive; size a follow-up only if its expected decision value warrants the cost |

## Relationship to P5, P16 and Eden

This assay does not measure correction service Q and offered burden W in the units
required by P16. Its detection and repair counts must not be renamed Q/W without a
separately justified observation model. It does not identify P5's burden elasticity,
prove an exclusion restriction for retained-fraction manipulation, or validate a
universal exponent. Existing confirmatory runner refusals stay in place.

P5 still needs interventions that vary stock and rate sufficiently independently to
identify rival burden models. P16 still needs a frozen complete decision-procedure
calibration, including all arms, repeated looks, nuisance settings and the final
support decision. A favourable component test is not a calibrated programme-level
false-positive rate. Correlated-failure measurements need a separate temporal bridge
before supporting accumulated correction independence.

Eden requires its own persistence comparison: specified values, known alternative
methods, withheld challenges, removal of external enforcement, opportunity to defect,
and a task where harmful compliance can be distinguished from beneficial persistence.
Reliably preserving an objective is not evidence that the objective is good.

## Physics and paper updates

Use physics collaborators to test a sharply defined physical prediction with explicit
observables and rival models. Acoustic order, thermodynamic dissipation or a square-root
exponent cannot by themselves validate an ethical zero-factor property. Publicly
available results are prior evidence, not a prospective prediction made after reading
them. A physics estimate cannot tighten an unrelated AI confidence interval merely
because both parameters are called alpha.

Paper III remains mastered on the website. Any substantive revision must use its
latest version and update the revision ledger, PDF, TXT, CFF and pending OSF packet
together. Preserve the recently merged version-lineage and citation-credit repairs.
Do not rename already measured variables from intelligence to entropy to manufacture
cross-domain agreement. Address the safety mechanism and comparator in their own
sections rather than imply that capability scaling proves loss of human control.

## Funding paragraph for the eventual tested proposal

“We will test whether oversight composition reduces unresolved incorrect outputs at
matched evaluator competence, utility and inference resources. The initial award funds
instrument validation and a preregistered comparison on a new task bank. Supportive,
negative and inconclusive outcomes will be released with raw permitted data and code.
Continuation depends on an independently measurable benefit and a credible route to a
specified safety decision. No claim of solved alignment or universal recursive stability
is required for the study to be informative.”

Independent status does not require permission to investigate or imply lesser scientific
value. It also does not establish neutrality or superiority. Reproducible measurements,
clear credit, disclosure and useful negative results are the evidence readers can assess.
Estimate costs from pilot receipts; do not turn a notional institutional replacement
cost into scientific validation or describe a large speculative grant as a safe bet.
