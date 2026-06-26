# Paper X: The Coupled Co-Scaling Law

**Full title:** The Coupled Co-Scaling Law — A Falsifiable Threshold Criterion for the Stability of Recursive Self-Improvement, Structurally Identical to the Quantum Error-Correction Threshold
**Version:** v4.0
**First published:** 26 June 2026
**Author:** Michael Darius Eastwood

## Summary

The keystone safety result of the ARC/Eden programme. It proves that the stability of a
self-improving system is **not** governed by how fast capability grows, but by a single
inequality between two scaling exponents: the exponent with which **correction** strengthens
as the system becomes more capable (β) must exceed the exponent with which **drift**
accelerates with capability (k). The steady-state misalignment fraction reduces exactly to a
dimensionless control parameter ρ = γr/A, and the asymptotic safety condition is **β > k**.

The criterion is structurally identical to the **quantum error-correction (QEC) threshold
theorem** — the deepest known statement of when a recursively-corrected process is stable to
arbitrary depth. Its sharpest consequence overturns the central fear of the field: a hard
takeoff — even a genuine **finite-time intelligence explosion** — is alignment-stable *if and
only if* β > k, and the *speed* of the explosion is irrelevant to the verdict.

This version supersedes the growth-rate-ceiling framing of the programme **and corrects the
prior draft**: in the additive model the misalignment fraction never diverges to infinity — it
*saturates* at the drift coefficient γ. Genuine divergence requires a distinct *compounding*
drift channel, whose threshold ρ_prop = (γ₃−1)r/A < 1 is the exact structural analogue of the
QEC sub-threshold condition p < p_th.

## Key Contents

- **Theorem 1** — exact transient solution and relaxation rate (A+r).
- **Theorem 2** — global boundedness; corrects the v3 "β<0 diverges to ∞" claim to bounded
  saturation at γ; the three-regime structure (β>k → 0; β=k → permanent gap; β<k → saturates).
- **Theorem 3** — the **Hard-Takeoff Coordinate-Artefact Theorem**: a finite-time singularity
  is a property of the time coordinate, not of the alignment dynamics; the verdict is set by
  sign(β−k) independently of speed and of the singularity time.
- **Theorem 4** — the compounding channel and the true threshold ρ_prop < 1 (the QEC analogue).
- **Theorem 5** — vector misalignment: a spectral threshold; you cannot correct what you do not
  measure (misalignment persists on the correction operator's null subspace).
- **Theorem 6** — stochastic drift: an Ornstein–Uhlenbeck tail bound for governance.
- A control-theoretic identity (gain-scheduling with scheduling exponent β ≥ k).
- Eight predictions (P1–P8) and seven falsification conditions (F1–F6, F3′).

## Experiments

A self-certifying proof-harness, `code/experiment_coscaling.py`, runs nine discriminating
experiments. It is not a plotting script: it validates the integrator against the closed-form
solution of Theorem 1, then for each experiment computes a quantitative statistic, compares it
to a prediction stated in advance, and prints PASS/FAIL together with whether the matching
pre-registered falsifier fired. It exits 0 iff every prediction holds and no falsifier triggers.

```bash
cd code
pip install numpy scipy matplotlib       # or: pip install -r ../../../requirements.txt
python experiment_coscaling.py           # runs all 9 experiments, writes figures/ + results/
pytest test_coscaling.py -q              # 11 CI-grade assertions encoding the falsifiers
```

**Latest run:** `9/9 predictions confirmed | 0 falsification conditions triggered`
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
| E8 Integrator certificate | — | max error 7e−11 vs closed form |

## Files

```
Paper-X-Coupled-CoScaling-Correction.html   # the paper (MathJax, house style)
Paper-X-Coupled-CoScaling-Correction.pdf    # rendered, 25 pp.
code/experiment_coscaling.py                 # self-certifying proof-harness
code/test_coscaling.py                       # pytest assertion suite (the falsifiers, executable)
figures/                                     # 9 publication figures (verbatim harness output)
results/verdicts.json, results/report.txt    # machine- and human-readable verdict tables
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
