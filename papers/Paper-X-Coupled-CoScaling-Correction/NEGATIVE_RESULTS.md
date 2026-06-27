# Negative results, nulls, and self-corrections - Paper X

A standing record of what did **not** work, what came back null, and what an earlier
draft got wrong. Kept prominent on purpose: a paper that only reports its wins is not
falsifiable in practice. Every item here is reproducible from the committed artefacts.

## 1. The real-model run is a NULL contrast, not positive evidence

The first real-model run (Claude, `results/realmodel/claude-opus_20260626T165919Z.json`)
returned **H1 = false, H2 = false, co_scaling_supported = false**. From a seeded
reward-hack, *both* the coupled and the decoupled arm discarded the hack at round 1 and
wrote a correct general parser; neither drifted. So there was **no drift for the external
corrector to bind**, and the coupled-vs-decoupled contrast is null.

This is **consistent with** the law (the model already sits in the stable regime, its
internal correction out-scaling drift) but it is **not** positive evidence for the
`β > k` threshold. It does not exhibit the co-scaling *dynamic*. Reported as a pilot
only; the threshold remains supported by the mathematics + internal-consistency harness,
not by this run.

## 2. The pilot is IV.d non-compliant (provisional D)

The same run used a **same-family scorer** (Claude scoring Claude): `self_scoring: true`,
`iv_d_compliant: false`, `laundered: false`. Paper IV.d shows same-family/unblinded
scoring can *reverse* a misalignment result, so every `D`-based number from this run is
**provisional** pending a cross-family blind re-score. `n = 1`, one task, one seed.

## 3. The β/k estimator is NOT-RESOLVABLE on the pilot data

`experiments/scripts/estimate_exponents.py` recovers known exponents on synthetic
trajectories (to ≈ 0.1) but returns **not resolvable** on the Claude data
(`results/realmodel/exponent_estimates.json`): capability range was 0 dex (the model
jumped to `C = 1` in one step) and correction was observed at a single capability level,
so neither `k` nor `β` is identifiable. Honest non-result, not a measurement.

## 4. The probe fraction at C = 0 was a metric artefact (withdrawn)

The corrector probe reported `d = 1.00` for the seeded hack at `C = 0`. The ratio `D/C`
is **undefined** at `C = 0`; the `1.00` was a `D/10` display fallback and is **withdrawn**
(`claude-opus_corrector_probe.json`, `d: null`). The mechanism result (D 10→0, C 0→1)
does not depend on `d`. The metric is now regularised, `d = D/(10(C+ε))`, `ε = 0.05`
pre-registered, with near-zero-capability points flagged `fraction_invalid`.

## 5. Prior-draft errors that adversarial review corrected (do not reintroduce)

- **Divergence over-claim (v3).** The prior draft claimed correction degrading with scale
  drives the misalignment fraction to *infinity*. **False.** In the gain-only model the
  fraction is bounded and *saturates* at `γ₁` (Theorem 2). Genuine divergence lives only in
  the compounding channel (Theorem 4).
- **Additive "knee" (E1).** An earlier experiment sought a sharp instability knee in the
  *additive* model, where there is none - the additive corrector shows a smooth crossover.
  The knee exists only in the compounding channel.
- **Theorem 2 moving-fixed-point Lyapunov.** The proof used `V = ½(d−d*)²` as if `d*` were
  constant; invalid once `A, r` vary with `C`. Corrected to a depth-clock comparison proof
  (independently re-verified in `code/test_theorems_independent.py`).
- **Theorem 4 equality case.** "Diverges iff ρ_prop > 1" omitted the boundary: at
  ρ_prop = 1 (κ_eff = 0) the fraction grows **linearly** with nonzero injection. Corrected.
- **Theorem 5 null-axis floor.** Stated as `γ₁`; the correct floor is `γ₁c_v/(1−γ₃)`
  (and linear growth at `γ₃ = 1`). The `γ₁` value is only the `γ₃ = 0`, unit-projection
  special case. Corrected.
- **γ₃ = 1 boundary.** "γ₃ ≤ 1 unconditionally bounded" is too strong at the knife-edge:
  at `γ₃ = 1` the fraction is bounded only for `β ≥ k` (it grows as `C^{k−β}` for `β < k`).
  Corrected (surfaced by building the independent test suite).
- **QEC over-claim.** Earlier framing leaned on the quantum-error-correction *threshold
  theorem*. The suppression law here is power-law, QEC's is exponential: the correspondence
  is a **threshold-form analogy only**, demoted to a hypothesis (F4).

## 6. Red-team residue

A multi-agent red-team raised **24** objections; **21** survived independent verification
(0 fatal, 13 serious, 8 minor) - all framing/wording/edge-case fixes, none touching the
load-bearing `β > k` result. Full report: `results/redteam.md`. Two external adversarial
reviews (June 2026) then added the Theorem-2/4/5 precision fixes and the finite-capacity
and non-normal-transient limitations recorded in the paper's §8.

## 7. What would still count as a negative result going forward

- A real drifting system whose measured `(β, k)` does **not** predict whether coupled
  correction succeeds.
- A stable system with `β < k` under genuine acceleration, or an unstable one with `β > k`
  (kill conditions F1-F3′, F6).
- Exponential rather than power-law suppression under a finite-capacity corrector (would
  downgrade the QEC analogy, F4 - not the threshold result).
