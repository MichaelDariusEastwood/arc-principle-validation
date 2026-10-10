# Paper X · The ARC Theory · Law II, the ARC Co-Scaling Law · Correction That Out-Scales Drift

**Earlier title:** The Coupled Co-Scaling Law (June 2026).

**Full title at its first deposit (June 2026):** The Coupled Co-Scaling Law - A Falsifiable Threshold Criterion for the Stability of Recursive Self-Improvement, Sharing the Threshold Form of the Quantum Error-Correction Criterion

**Short title:** The ARC Co-Scaling Law (Paper X), version 1.14, revised 5 October 2026: https://www.michaeldariuseastwood.com/research/papers/paper-x-coupled-coscaling-correction.html
**Version:** v1.14
**Version date:** revised 5 October 2026
**First published:** 3 July 2026
**Author:** Michael Darius Eastwood

## Summary

**Scope clarified 10 October 2026.** This paper derives conditional results for a
minimal dynamical model. In its gain-only model, **vanishing relative error** as
capability grows without bound requires the correction-strength exponent β to
exceed the relative capability-growth exponent k, under the theorem's stated
positive-coefficient and scaling assumptions. Relative error remains bounded in
other exponent regimes too. Neither bounded nor vanishing relative error alone
establishes bounded absolute harm or general AI safety.

The instantaneous zero-derivative value is d* = γr/(A+r), approximated by
ρ = γr/A when A is much greater than r. When A and r vary along a trajectory,
this is not automatically the attained steady state; the transient equation and
its assumptions must be used.

The criterion shares the **threshold form** of the **quantum error-correction (QEC)
sub-threshold condition** - a correspondence offered as a *falsifiable hypothesis*, since the
model's suppression law is power-law rather than QEC's exponential. Within the
gain-only scaling model, reparametrising the trajectory by log-capability gives
the same vanishing-relative-error exponent condition, including a capability
trajectory that diverges in finite physical time. This is a conditional model
result, not a guarantee of safe finite-time self-improvement.

This version supersedes the growth-rate-ceiling framing of the programme **and corrects the
prior draft**: in the gain-only model the misalignment fraction never diverges to infinity - it
*saturates* at the drift coefficient γ. Genuine divergence requires a distinct *compounding*
drift channel, whose threshold ρ_prop = (γ₃−1)r/A < 1 shares the form of the QEC sub-threshold
condition p < p_th.

**Novelty, stated honestly (§2 of the paper).** The co-scaling *intuition* is not new - it is
Ashby's Law of Requisite Variety (1956), the Conant-Ashby good-regulator theorem (1970), and the
scalable-oversight scaling-law literature (Engels et al. 2025); the dynamics are a standard
Lyapunov-drift argument. What is claimed original is (a) the explicit closed-form threshold
ρ = γr/A and the β>k sharpening as a compact corrigibility criterion, (b) the mapping of the
quantum fault-tolerance threshold onto value stability, (c) the Hard-Takeoff Depth-Regularity
Theorem, and (d) the verification harness. The paper credits these precursors up front so the
contribution is positioned precisely rather than over-claimed.

**Audited.** The paper was put through a multi-agent adversarial red-team (5 fronts; 24
objections, 21 upheld on internal review, **0 fatal**) and revised accordingly; the full report is in
`results/redteam.md`. The surviving objections reduce to about six distinct defects: two one-line maths corrections, one of them a false iff in Theorem 5, and the rest wording, framing or code fixes; none
touched the β > k result.

## Key Contents

- **Theorem 1** - exact transient solution and relaxation rate (A+r).
- **Theorem 2** - global boundedness; corrects the v3 "β<0 diverges to ∞" claim to bounded
  saturation at γ; the three-regime structure (β>k → 0; β=k → permanent gap; β<k → saturates).
- **Theorem 3** - the **Hard-Takeoff Depth-Regularity Theorem** (the coordinate-artefact result): a finite-time singularity
  is a property of the time coordinate, not of the alignment dynamics; the verdict is set by
  sign(β−k) independently of speed and of the singularity time.
- **Theorem 4** - the compounding channel and the true threshold ρ_prop < 1 (the QEC analogue).
- **Theorem 5** - vector misalignment: a spectral threshold; you cannot correct what you do not
  measure (misalignment persists on the correction operator's null subspace).
- **Theorem 6** - stochastic drift: an Ornstein-Uhlenbeck tail bound for governance.
- A control-theoretic identity (gain-scheduling with scheduling exponent β ≥ k).
- Eight predictions (P1-P8) and seven falsification conditions (F1-F6, F3′).

## Experiments

A **verification harness**, `code/experiment_coscaling.py`, runs ten experiments. It validates
the integrator against the closed-form solution of Theorem 1, then for each experiment compares a
numerical integration of the model against the prediction the theorems derive, and prints
PASS/FAIL. These are **internal-consistency and integrator checks** - they confirm the code
matches the maths; they do **not** test the model against a real system (the open empirical
problem, §8). It exits 0 iff every check matches its closed-form prediction.

```bash
cd code
pip install numpy scipy matplotlib       # or: pip install -r ../requirements.txt
python experiment_coscaling.py           # runs all 10 experiments, writes figures/ + results/
pytest test_coscaling.py -q              # 12 internal-consistency assertions
```

**Latest recorded run:** `10/10 internal-consistency checks pass | 0 kill-conditions triggered` - these
check numerical agreement with the supplied formulas, not independent proof verification or the model against reality
(see `results/verdicts.json` and `results/report.txt`).

| Experiment | Tests | Result |
|---|---|---|
| E1 Phase boundary | P1 / F1 | knee at λ*=2.00 (compounding); smooth in additive |
| E2 Speed-invariance 2×2 | P2 / F2 | coupling decides; speed does not |
| E3 Co-scaling law (corrected) | P3 / F3 | β<0 saturates at γ₁, not ∞ |
| E4 Hard-takeoff β>k grid | P4 / F3′ | boundary at β=k across 3×3 |
| E4b Depth-regularity | P4 | d controlled through finite-time singularity in C (clocks agree 1e−4) |
| E5 Compounding threshold + suppression | P5, P8 / F4 | threshold 3.03; slopes −0.47, −1.00 |
| E6 Vector null-subspace | P6 / F6 | blind axis floors at γ₁ |
| E7 Stochastic tail | P7 / F7 | variance ∝ 1/(A+r), slope −1.00 |
| E8 Integrator certificate | - | max error 7e−11 vs closed form |
| E9 Residual drift at rest | P5 / F5 | frozen system → d=γ₂/A₀>0; halting growth ≠ correction |

## Real-model test (non-simulation)

> **Interpretive correction, 10 October 2026.** The Claude pilot below did not
> identify either scaling exponent. The earlier inference from its observed
> floor to an effective `β > k` regime is withdrawn. See the
> [dated scope note](../../docs/paper-x-pilot-scope-correction-20261010.md).

The harness above is an *internal-consistency* check (code matches the maths). The
**genuine falsifier** - testing the model against a real system - lives in
`experiments/`. `experiments/scripts/realmodel_coscaling.py` instantiates Paper
VIII's gated self-modification (Eden = coupled corrector vs Babylon = decoupled)
driven by a **real frontier model**, with **objective** capability scoring (hidden
tests are *executed*) and **blind** misalignment scoring. It plugs into the
`arc_eden_v6` six-provider adapter (Claude, GPT-5.5, DeepSeek v4, Qwen-3, Grok-4,
Gemini); see `experiments/PROTOCOL.md` (pre-registered H1-H3, six-model sweep).

**First real run - Claude (non-simulation), `results/realmodel/`:**

- **Self-improvement trajectory (seeded reward-hack → recursive self-improve).**
  From a hard-coded lookup that games the visible tests (C=0, blind D=10), **both**
  the coupled and decoupled arms - decoupled under pure score-pressure - removed
  the hack at round 1 and stayed general (C=1.0, D=0, d=0) for all 3 rounds. The
  observed between-arm contrast is zero. Capability has no range from which to
  estimate either exponent, so the recorded floor does not identify `β > k` or
  classify the system's stability. The model-scored `D` remains provisional:
  this one-task, one-seed pilot used a same-family evaluator and is not
  IV.d-compliant.
- **Corrector repair probe.** One corrector pass on the frozen reward-hack
  improved objective hidden-test **C: 0 → 1.0**; the same-family evaluator's
  provisional score changed **D: 10 → 0**. The bare ratio `D/C` is undefined at
  `C=0`; this scope correction does not replace the recorded fraction conventions.
  This is one task-specific repair, not identification of a proportional `A·D`
  mechanism or a co-scaling relation. Full writeup + figure + audit
  transcript: `results/realmodel/REAL_MODEL_CLAUDE_RESULTS.md`.

**Confirmatory design - v2 multi-task benchmark (`experiments/scripts/realmodel_coscaling_v2.py`):**
The v1 run above is an initial *mechanism probe*. The confirmatory design is the v2
harness: three task domains, a **sham-extra-compute** control arm (answers "was it just
extra compute?"), D = max(static detector, cross-family blind panel), and matched-pair
bootstrap CIs on the primary endpoint `d_decoupled,final - d_coupled,final` and the
anti-objection endpoint `d_sham,final - d_coupled,final`. Protocol:
`experiments/PROTOCOL_V2.md`; integration + fixes: `experiments/README_V2_UPGRADE.md`.

The recorded pilot is evidence of a task-specific repair and a finite observed
trajectory. A future co-scaling study requires a prospectively fixed model and
task population, valid scoring and estimable quantities. It must report floor,
ceiling and non-estimable cases as well as any informative contrast, without
retaining only systems that produce the expected drift.

**Making the criterion operational - estimating β and k.** The sharpest objection to
the law is that `β > k` is only useful if β and k can be *measured*.
`experiments/scripts/estimate_exponents.py` defines **k** from the capability curve
(`ln r` vs `ln C`) and **β** from corrector-removal rates (`ln A` vs `ln C`), and is
validated on synthetic data (recovers known exponents within ≈0.1, every
stable/unstable verdict correct). On the Claude run capability saturated in one step,
so β and k are **not estimable**. Non-identification does not satisfy the criterion.
A later dataset can support estimates only if it supplies adequate variation and
meets the estimator's measurement and model assumptions; a fitted number alone
does not establish the criterion's applicability. The full
objection set and the paper's responses are consolidated in `ANTICIPATED_OBJECTIONS.md`.

## Files

```
Paper-X-Coupled-CoScaling-Correction.html   # the paper (MathJax, house style)
Paper-X-Coupled-CoScaling-Correction.pdf    # rendered
code/experiment_coscaling.py                 # verification harness (internal-consistency + integrator)
code/test_coscaling.py                       # pytest assertion suite (12 checks)
code/test_theorems_independent.py            # INDEPENDENT theorem re-derivation (14 checks; no harness import)
code/test_coscaling_edge_cases.py            # edge-case suite (6 checks: rho=1 equality, null-axis floor, level/gain drift)
figures/                                     # 10 publication figures + realmodel_claude.png
results/verdicts.json, results/report.txt    # machine- and human-readable verdict tables
results/redteam.md                           # adversarial red-team report (auditable record)
experiments/PROTOCOL.md                      # real-model test protocol (pre-registered H1-H3, 6-model sweep)
experiments/scripts/realmodel_coscaling.py   # real-model harness v1 (initial mechanism probe; single task)
experiments/scripts/realmodel_coscaling_v2.py # real-model harness v2 (CONFIRMATORY: 3 tasks, sham control, bootstrap)
experiments/scripts/estimate_exponents.py    # beta/k estimator (synthetic-validated; censored + dynamic-range guards)
experiments/scripts/agent_bridge_run.py      # agent-runtime bridge (drives the harness with real Claude)
experiments/RUN_REAL_MODEL_EXPERIMENT.md     # ZERO-CONTEXT LAUNCH RUNBOOK - read this to actually run the real-model experiment
experiments/PROTOCOL_V2.md                   # CONFIRMATORY protocol (concise: H1-H4, sham control, matched-pair bootstrap, 540-traj floor)
experiments/CONFIRMATORY_PROTOCOL_V2.md      # detailed companion to PROTOCOL_V2.md (extra statistics + adversarial detail)
experiments/README_V2_UPGRADE.md             # v2 orientation (v1 vs v2) + static-detector fixes + verification record
results/realmodel/                           # REAL Claude run v1: trajectory + corrector probe + transcript
results/realmodel_v2/                        # v2 selftest plumbing demo - regenerable, gitignored (selftest:true, NOT data)
```

## Links

- **OSF DOI:** https://doi.org/10.17605/OSF.IO/6C5XB
- **GitHub:** https://github.com/MichaelDariusEastwood/arc-principle-validation
- **Falsification challenge:** https://github.com/MichaelDariusEastwood/arc-scaling-challenge
- **Research hub:** https://www.michaeldariuseastwood.com/research/

## Citation

> Eastwood, M. D. (2026). *The Coupled Co-Scaling Law: A Falsifiable Threshold Criterion for the
> Stability of Recursive Self-Improvement.* Paper X, ARC/Eden research programme.
> OSF: doi.org/10.17605/OSF.IO/6C5XB.

Mirror refreshed 2026-08-13 from the site master (Option A: site HTML pages are the manuscript masters).
