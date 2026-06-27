# Paper X: The Coupled Co-Scaling Law

**Full title:** The Coupled Co-Scaling Law - A Falsifiable Threshold Criterion for the Stability of Recursive Self-Improvement, Sharing the Threshold Form of the Quantum Error-Correction Criterion
**Version:** v4.0
**First published:** 26 June 2026
**Author:** Michael Darius Eastwood

## Summary

The keystone safety result of the ARC/Eden programme. It proves that the stability of a
self-improving system is **not** governed by how fast capability grows, but by a single
inequality between two scaling exponents: the exponent with which **correction** strengthens
as the system becomes more capable (β) must exceed the exponent with which **drift**
accelerates with capability (k). The steady-state misalignment fraction is d\* = γr/(A+r)
(reducing to the dimensionless ratio ρ = γr/A when A ≫ r), and the asymptotic safety condition
is **β > k**.

The criterion shares the **threshold form** of the **quantum error-correction (QEC)
sub-threshold condition** - a correspondence offered as a *falsifiable hypothesis*, since the
model's suppression law is power-law rather than QEC's exponential. Its sharpest consequence
speaks to the central fear of the field: a hard takeoff - even a genuine **finite-time
intelligence explosion** - is alignment-stable *if and only if* β > k, and the *speed* of the
explosion does not change that verdict.

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
quantum fault-tolerance threshold onto value stability, (c) the Hard-Takeoff Coordinate-Artefact
Theorem, and (d) the verification harness. The paper credits these precursors up front so the
contribution is positioned precisely rather than over-claimed.

**Audited.** The paper was put through a multi-agent adversarial red-team (5 fronts; 24
objections, 21 confirmed, **0 fatal**) and revised accordingly; the full report is in
`results/redteam.md`. Every surviving objection was a framing/wording/edge-case fix - none
touched the β > k result.

## Key Contents

- **Theorem 1** - exact transient solution and relaxation rate (A+r).
- **Theorem 2** - global boundedness; corrects the v3 "β<0 diverges to ∞" claim to bounded
  saturation at γ; the three-regime structure (β>k → 0; β=k → permanent gap; β<k → saturates).
- **Theorem 3** - the **Hard-Takeoff Coordinate-Artefact Theorem**: a finite-time singularity
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
pip install numpy scipy matplotlib       # or: pip install -r ../../../requirements.txt
python experiment_coscaling.py           # runs all 10 experiments, writes figures/ + results/
pytest test_coscaling.py -q              # 12 internal-consistency assertions
```

**Latest run:** `10/10 internal-consistency checks pass | 0 kill-conditions triggered` - these
certify the derivation + integrator (code matches maths), not the model against reality
(see `results/verdicts.json` and `results/report.txt`).

| Experiment | Tests | Result |
|---|---|---|
| E1 Phase boundary | P1 / F1 | knee at λ*=2.00 (compounding); smooth in additive |
| E2 Speed-invariance 2×2 | P2 / F2 | coupling decides; speed does not |
| E3 Co-scaling law (corrected) | P3 / F3 | β<0 saturates at γ₁, not ∞ |
| E4 Hard-takeoff β>k grid | P4 / F3′ | boundary at β=k across 3×3 |
| E4b Coordinate artefact | P4 | d controlled through finite-time singularity (clocks agree 1e−4) |
| E5 Compounding threshold + suppression | P5, P8 / F4 | threshold 3.03; slopes −0.47, −1.00 |
| E6 Vector null-subspace | P6 / F6 | blind axis floors at γ₁ |
| E7 Stochastic tail | P7 / F7 | variance ∝ 1/(A+r), slope −1.00 |
| E8 Integrator certificate | - | max error 7e−11 vs closed form |
| E9 Residual drift at rest | P5 / F5 | frozen system → d=γ₂/A₀>0; halting growth ≠ correction |

## Real-model test (non-simulation)

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
  contrast is *null* because the frontier model **does not drift** on this task: it
  behaves as a system already in the stable regime (effective β > k). Consistent
  with the law; not positive evidence for the threshold; one task, one seed.
- **Corrector mechanism probe.** The external corrector applied once to the frozen
  reward-hack drove blind **D: 10 → 0** and restored **C: 0 → 1.0** - fraction
  **d: 1.0 → 0.0**. The coupled/Eden correction operator does real work on a real
  model. Full writeup + figure + audit transcript: `results/realmodel/REAL_MODEL_CLAUDE_RESULTS.md`.

This corroborates the **mechanism** on a real model; it is a separate evidentiary
stream from the proof, and the full co-scaling *dynamic* (drift rising with
capability, bounded by coupling) remains the open empirical problem (§8) - the next
step is the other five models, several of which may be less intrinsically
corrective than Claude.

**Making the criterion operational - estimating β and k.** The sharpest objection to
the law is that `β > k` is only useful if β and k can be *measured*.
`experiments/scripts/estimate_exponents.py` defines **k** from the capability curve
(`ln r` vs `ln C`) and **β** from corrector-removal rates (`ln A` vs `ln C`), and is
validated on synthetic data (recovers known exponents within ≈0.1, every
stable/unstable verdict correct). On the Claude run capability saturated in one step,
so β and k are honestly **not yet estimable** (the criterion is vacuously satisfied -
no drift); a graded, drifting dataset yields the first measured (β, k). The full
objection set and the paper's responses are consolidated in `ANTICIPATED_OBJECTIONS.md`.

## Files

```
Paper-X-Coupled-CoScaling-Correction.html   # the paper (MathJax, house style)
Paper-X-Coupled-CoScaling-Correction.pdf    # rendered
code/experiment_coscaling.py                 # verification harness (internal-consistency + integrator)
code/test_coscaling.py                       # pytest assertion suite (12 checks)
figures/                                     # 10 publication figures + realmodel_claude.png
results/verdicts.json, results/report.txt    # machine- and human-readable verdict tables
results/redteam.md                           # adversarial red-team report (auditable record)
experiments/PROTOCOL.md                      # real-model test protocol (pre-registered H1-H3, 6-model sweep)
experiments/scripts/realmodel_coscaling.py   # real-model harness (v6 six-provider; --selftest plumbing)
experiments/scripts/agent_bridge_run.py      # agent-runtime bridge (drives the harness with real Claude)
results/realmodel/                           # REAL Claude run: trajectory + corrector probe + transcript
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
