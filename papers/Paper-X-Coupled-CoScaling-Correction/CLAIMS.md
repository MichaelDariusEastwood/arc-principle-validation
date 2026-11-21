> **Note, 21 November 2026.**
>
> This ledger was written in June 2026, for the drafts that came before the paper's original edition of 3 July 2026. The paper is now at version 1.18, of 9 November 2026, which has superseded these statements in the ledger:
>
> - Row 1 says “Stability of recursive self-improvement is governed by the exponent inequality **$\beta > k$** (correction must out-scale drift-acceleration).” Version 1.18 says (Section 1): “With positive gain drift, whether the misalignment fraction of a self-improving system vanishes is decided, in the model, by one exponent inequality, 𝛽 > 𝑘 (correction must out-scale drift-acceleration; with level drift and decelerating growth, also 𝛽 > 0)”. Row 1 cites Theorems 1, 3 and 4; Section 3.3 says: “Theorem 1 establishes the constant-coefficient baseline; the headline criterion follows only under the scaling laws of equation (1), in §§3.4 to 3.8.”
> - Row 4 says “A finite-time hard takeoff is alignment-stable iff $\beta > k$; the verdict $= \mathrm{sign}(\beta - k)$, independent of speed and of the singularity time.” Version 1.18 states Theorem 3 “Without compounding (𝛾3 = 0; level drift 𝛾2 ≥ 0 allowed)” and gives: “for 𝛾1 > 0, 𝑑 → 0 iff 𝛽 > 𝑘, a regime independent of 𝑏 and of 𝑡★, though the gap at 𝛽 = 𝑘 and the path at every finite depth depend on 𝑏”. It adds: “With compounding the boundary at 𝛽 = 𝑘 depends on 𝑏 (Theorem 4(c)).” Section 3.7 says that “it does not assert that a real hard takeoff is operationally manageable in wall-clock time”.
> - Row 5 says “Genuine divergence lives in a distinct compounding channel with threshold $\rho_{\mathrm{prop}} = (\gamma_3 - 1)r/A < 1$.” Version 1.18 says (Section 3.4): “(Divergence does exist: in the compounding channel, Theorem 4, and in the level-drift case just named.)” The level-drift case is the one in which “growth decelerates while correction weakens”. Of the ratio threshold, Theorem 4 says: “Positive 𝜅 eff at every instant does not by itself bound the fraction, and a temporary crossing of 𝜌prop = 1 does not by itself make it diverge.” Under the power laws it gives: “The unbounded region is therefore 𝛽 < 𝑘 with 𝛾3 ≥ 1, or 𝛽 = 𝑘 with 𝐴0 ≤ (𝛾3 − 1)𝑏.”
> - Row 6 says “Vector misalignment: an exact spectral-abscissa threshold; misalignment persists on the corrector's null subspace ("you cannot correct what you do not measure").” Version 1.18 states the spectral threshold for constant coefficients (Theorem 5): “For constant A and 𝑟 it has a unique globally exponentially attracting equilibrium iff every eigenvalue of M has positive real part”. It says that “an axis the corrector does not measure (Av = 0) can still be corrected through coupling when A is not normal”. Section 3.9 reads: “The governance reading is sharp: you cannot correct along an axis on which the corrector never acts.”
> - Row 7 says “Stochastic drift: an Ornstein-Uhlenbeck tail bound for governance.” Version 1.18 gives a single-time tail and says (Theorem 6): “It is not the probability of crossing 𝑑crit at some time, which over an unbounded horizon is one.” Section 3.10 adds that “it certifies nothing over a horizon”.
> - Row 9 says “the corrected theorems are **independently** re-verified from scratch” and counts “test_theorems_independent.py (14, no harness import)”. Version 1.18 says (Section 5.2): “A second suite of 104 parametrised cases, written separately from the harness and merged into the paper’s code repository on 19 October 2026 (merge commit 83871b5), checks the corrected statements of Theorems 1 to 7 at sampled parameter values”. It adds: “It predates, and does not test, the level-drift case of §3.4, which Appendix A.1 proves.” Section 5.1 says that “the proofs of Appendix A, not these checks, carry the theorems”.
> - Row 11 calls the pilot's misalignment score “blind $D: 10 \to 0$”. Version 1.18 lists the pilot as “Real-model pilot v1 (superseded).” (Section 8) and says of its stored record (Section 9.1, Table 4): “The stored record explicitly identifies same-family scoring and non-compliance with the blinding standard.”
> - Row 15 says “$\beta > k$ holds only in the **unbounded power-law corrector** regime” and cites Theorem 4. Version 1.18 treats the finite-capacity corrector in Theorem 7, which says that “with 𝛾1 > 0 a finite-capacity corrector cannot drive the fraction to zero under indefinitely accelerating growth, however large 𝛽 (with 𝛾1 = 0 the fraction tends to zero)” (part 1), and that “the criterion 𝛽 > 𝑘 survives the lift from correction strength to correction capacity unchanged” (part 4). It adds: “What must out-scale drift-acceleration is whichever of the two binds last.”
> - Row 16 says “the criterion is in scale-free exponents”. Version 1.18 says (Section 9) that the exponents are unchanged when capability is rescaled by a constant factor, but that “a nonlinear rescaling, such as a logit or an ability scale, changes the power laws and the gain-drift term, so the capability scale is part of the model.” It adds: “A pass fraction bounded by one cannot reach the regime 𝐶 → ∞ that the theorems describe”.
> - The priority line says the thesis was set out in *Infinite Architects* “(copyright deposited 8 Dec 2024; published 2 Jan 2026; ISBN 978-1806056200)”. Version 1.18 says (Appendix C.4): “The conceptual thesis of this programme was set out in the author’s book Infinite Architects (first set out in a message the author sent himself on 8 December 2024, its time signed by the sending server and sealed by Google’s servers; published 2 January 2026; ISBN 978-1806056200; Eastwood 2026a)”. Appendix C.3 says of the priority records that “they are evidence of the date of articulation rather than of a registered deposit”.
> - The novelty line counts as novel residue “and the runnable $\beta/k$ estimator + protocol”. Version 1.18 says (Appendix E): “What this paper asserts as new, and only this: (i) the 𝛽 > 𝑘 criterion for a vanishing misalignment fraction, which indexes a classical feedback balance by capability, with its finite-time reading in the Hard-Takeoff Depth-Regularity Theorem; and (ii) the explicit QEC threshold mapping (as a hypothesis with its own falsifier).”
> - The falsification list gives “A stable system with $\beta < k$ under genuine acceleration, or an unstable one with $\beta > k$ (kill conditions F1-F3', F6 in the stated model).” and “A real drifting system whose measured $(\beta, k)$ does **not** predict whether coupled correction succeeds or fails.” Version 1.18 says that F1 to F3, F3′, F5 and F6 “are therefore internal-consistency conditions of the derivation, not empirical falsifiers of the thesis” (Section 5.1). It says that reading a system's regime from an interval on its measured margin “is a measurement of a system, not a test of the model, and equality never refutes the model”, and: “Whether the model is right is read from a sealed prediction.” (Section 7). That prediction is of where the fraction crosses stated tolerances, from terms measured on separate data.
> - The falsification list also says “The QEC correspondence (claim 8) is downgraded - not the threshold result - if the suppression is shown to be exponential rather than power-law (F4).” Version 1.18 says (prediction P8): “Exponential suppression would indicate a stronger, QEC-like mechanism; neither outcome falsifies the threshold, only the QEC mechanism claim (F4).” Its table of falsification conditions gives the consequence of F4 as: “Bears on the QEC mechanism only.”
>
> Some places this ledger cites have moved in version 1.18: the limitations it cites as Section 8 are in Section 9, and the conditional statement it cites as Section 9 is in Section 10. The abstract of version 1.18 no longer states the contrast between power-law and exponential suppression; Sections 3.8 and 3.12 state it.
>
> Where this ledger and the paper differ, the paper governs.

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
| 4 | A finite-time hard takeoff is alignment-stable iff $\beta > k$; the verdict $= \mathrm{sign}(\beta - k)$, independent of speed and of the singularity time. | **P** | Theorem 3 (Hard-Takeoff Depth-Regularity); §3.7 |
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
  residue is: the closed-form $\beta > k$ criterion + the Hard-Takeoff Depth-Regularity theorem
  (this paper); the IV.d blinding finding (companion; no single cause claimed); and the runnable $\beta/k$
  estimator + protocol. Programme-wide prior-art audits are committed under each paper's `results/`.

## What would falsify or materially weaken the central claim

- A stable system with $\beta < k$ under genuine acceleration, or an unstable one with $\beta > k$
  (kill conditions F1-F3', F6 in the stated model).
- A real drifting system whose measured $(\beta, k)$ does **not** predict whether coupled correction
  succeeds or fails.
- The QEC correspondence (claim 8) is downgraded - not the threshold result - if the suppression
  is shown to be exponential rather than power-law (F4).
