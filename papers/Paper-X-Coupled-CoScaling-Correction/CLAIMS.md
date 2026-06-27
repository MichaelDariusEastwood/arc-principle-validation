# Claim-status ledger - Paper X

Every load-bearing claim, the exact rung it sits on, and where the evidence is. Nothing
here may be read one rung higher than it is placed. Rungs: **P** proved-within-model ·
**V** internally-verified (code matches maths) · **S** synthetically-validated · **Pilot**
real-model pilot (limited, provisional) · **Open** open empirical problem · **Hyp**
falsifiable hypothesis · **Cond** conditional implication.

| # | Claim | Rung | Evidence |
|---|---|---|---|
| 1 | Stability of recursive self-improvement is governed by the exponent inequality **$\beta > k$** (correction must out-scale drift-acceleration). | **P** | Theorems 1, 3, 4; §3.4-3.8 |
| 2 | Exact steady-state misalignment fraction $d^* = \gamma_1 r/(A+r)$, reducing to $\rho = \gamma_1 r/A$ when $A \gg r$. | **P** | Theorem 1; §3.5 |
| 3 | In the gain-only model the fraction **saturates at $\gamma_1$, it does not diverge to infinity** (corrects the prior draft). | **P** | Theorem 2; §3.4 |
| 4 | A finite-time hard takeoff is alignment-stable iff $\beta > k$; the verdict $= \mathrm{sign}(\beta - k)$, independent of speed and of the singularity time. | **P** | Theorem 3 (Hard-Takeoff Coordinate-Artefact); §3.7 |
| 5 | Genuine divergence lives in a distinct compounding channel with threshold $\rho_{\mathrm{prop}} = (\gamma_3 - 1)r/A < 1$. | **P** | Theorem 4; §3.8 |
| 6 | Vector misalignment: an exact spectral-abscissa threshold; misalignment persists on the corrector's null subspace ("you cannot correct what you do not measure"). | **P** | Theorem 5; §3.9 |
| 7 | Stochastic drift: an Ornstein-Uhlenbeck tail bound for governance. | **P** | Theorem 6; §3.10 |
| 8 | The threshold **shares the form** of the quantum error-correction sub-threshold condition - **not** its mechanism (QEC suppression is exponential; this is power-law). | **Hyp** (falsifiable, F4) | §3.12; F4; abstract disanalogy |
| 9 | The numerical implementation matches the closed-form derivation, and the corrected theorems are **independently** re-verified from scratch (Thm 2 time-varying limits; Thm 4 equality = linear divergence; Thm 5 non-normal transient + null-axis floor; Thm 6 OU; gamma3=1 boundary). | **V** | code/experiment_coscaling.py; results/verdicts.json; test_coscaling.py (12); test_theorems_independent.py (14, no harness import) |
| 10 | The $\beta/k$ estimator recovers known exponents from model-generated data (to $\approx 0.1$) and fails honestly on saturated data. | **S** | experiments/scripts/estimate_exponents.py; results/realmodel/exponent_estimates.json |
| 11 | On a real model (Claude), the external corrector removes a seeded reward-hack: blind $D: 10 \to 0$, capability $C: 0 \to 1.0$. | **Pilot** (mechanism; **IV.d non-compliant** - same-family scoring) | results/realmodel/claude-opus_corrector_probe.json; REAL_MODEL_CLAUDE_RESULTS.md |
| 12 | The co-scaling **dynamic** ($H_1$ decoupled drifts up; $H_2$ coupling bounds $d$) on a real model. | **NOT shown** (null contrast; the model did not drift) | results/realmodel/claude-opus_2026*.json (H1=false, H2=false) |
| 13 | $\beta$ and $k$ measured on a real drifting self-improving system across capability levels. | **Open** | experiments/PROTOCOL.md (the named next experiment) |
| 14 | If 13 succeeds, $\beta - k$ becomes a measurable corrigibility audit margin for governance. | **Cond** | §9 (stated conditionally) |
| 15 | $\beta > k$ holds only in the **unbounded power-law corrector** regime; a finite-capacity corrector ($A \to A_{\max}$) fails it asymptotically for $k>0$ (since $r \to \infty$ outpaces a fixed $A_{\max}$). | **P** (scope limit) | §8 "Unbounded vs finite-capacity corrector"; Theorem 4 |
| 16 | $d=D/C$ is an operational, scoring-convention-dependent risk index, not a natural constant; the criterion is in scale-free exponents. | **P** (scope) | §8 "Metric normalisation" |

## Priority vs novelty (separate axes)

- **Priority** (when the author articulated it): the thesis was set out in *Infinite Architects*
  (copyright deposited 8 Dec 2024; published 2 Jan 2026; ISBN 978-1806056200), with each paper
  time-stamped on OSF (10.17605/OSF.IO/6C5XB). This is real and is the author's.
- **Novelty** (new to the literature): the co-scaling *intuition* has ancestors (Ashby 1956;
  Conant-Ashby 1970; scalable oversight; Lyapunov drift), credited in §2. The defensible *novel*
  residue is: the closed-form $\beta > k$ criterion + the Hard-Takeoff Coordinate-Artefact theorem
  (this paper); the IV.d blinding-reversal measurement law (companion); and the runnable $\beta/k$
  estimator + protocol. Programme-wide prior-art audits are committed under each paper's `results/`.

## What would falsify or materially weaken the central claim

- A stable system with $\beta < k$ under genuine acceleration, or an unstable one with $\beta > k$
  (kill conditions F1-F3', F6 in the stated model).
- A real drifting system whose measured $(\beta, k)$ does **not** predict whether coupled correction
  succeeds or fails.
- The QEC correspondence (claim 8) is downgraded - not the threshold result - if the suppression
  is shown to be exponential rather than power-law (F4).
