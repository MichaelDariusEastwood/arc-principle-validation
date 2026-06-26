# Paper X — Adversarial Red-Team Report

**Method.** An independent multi-agent adversarial red-team attacked Paper X across five
fronts — **mathematics, numerics/code, the QEC correspondence, the safety inference, and
priority/originality**. Each objection raised by an attacker was adjudicated by a separate
skeptic agent (real / severity / invalid). This is the record the paper's §8 "Adversarial
audit" note refers to.

**Verdict: SURVIVES with revisions. No fatal, unfixable flaw.**

| | count |
|---|---|
| Objections raised | 24 |
| Survived independent verification | 21 |
| — fatal | **0** |
| — serious | 13 |
| — minor | 8 |

The load-bearing scientific contribution — the **β > k** stability criterion, derived from the
master equation and proved in Theorems 1, 3, 4 — is **untouched by every surviving objection**.
The 21 survivors de-duplicate to ~6 distinct defects, all of which "shrink the marketing, not
the science": two one-line maths corrections, one headline-wording error, and two over-sold
framing layers. None required retracting a theorem or re-running a result.

## The six must-fix defects, and how each was resolved in this version

1. **Theorem 5 stated a false `iff`** (Hermitian-part / numerical-range condition presented as
   necessary-and-sufficient; only the eigenvalue / spectral-abscissa condition follows from the
   proof, and they diverge for an admissible PSD non-normal `A`).
   → **Fixed.** Theorem 5 now states the exact criterion as `min Re λ(M) > 0` (positive spectral
   abscissa); the Hermitian-part condition is given as the stronger *sufficient* condition that
   also rules out transient growth. §8 reconciled.

2. **Theorem 2's boundedness bound `max(d0, γ1)` is false once level-drift γ₂ > 0**, which §3.6
   activates under the same "γ₃ = 0" banner.
   → **Fixed.** Theorem 2 is scoped to the **gain-only model (γ₂ = γ₃ = 0)**, matching Theorem 1;
   a note gives the correct γ₂ > 0 floor `(γ1 r + γ2)/(A+r) → γ2/A` at rest, still → 0 as C → ∞
   when β > 0 or k > 0. "Saturates at γ1" wording scoped to the gain-only model throughout.

3. **Headline `d* = ρ` "exactly" and "safety iff ρ < 1" were false as stated** (the exact steady
   state is `d* = γ1 r/(A+r)`; the additive fraction is bounded by γ1 < 1 for *all* ρ, so ρ < 1
   marks no transition — the genuine divergence threshold is the distinct `ρ_prop`). A stray
   algebra line `d* = ρ/(1+ρ)·(…)` was also wrong.
   → **Fixed.** Abstract, scope, §3.5 and conclusion now state `d* = γ1 r/(A+r)`, reducing to ρ
   only when A ≫ r; ρ < 1 is labelled the *instantaneous* injection-vs-correction balance,
   explicitly **not** the divergence boundary (that role is `ρ_prop < 1`, Theorem 4). The bogus
   algebra line is deleted.

4. **QEC over-claim** ("structurally identical" / "exact structural analogue" / "matches QEC's
   actual mechanism") contradicted the body's own §3.12/F4 concession that the suppression law
   differs (power-law vs QEC's exponential).
   → **Fixed.** Title, subtitle, meta, significance, §3.8 and conclusion now claim only that the
   threshold *condition* shares the ratio-crossing-unity *form*; the mechanism claim is removed;
   the power-law-vs-exponential disanalogy is carried up into the abstract/significance.

5. **"Self-certifying proof-harness that adjudicates every falsifier" over-sold the epistemics**:
   the harness integrates the model's own ODE, so a PASS certifies the algebra + solver, not that
   the model describes reality; no QEC object is simulated.
   → **Fixed.** Renamed to a **verification harness**; §5/§10 state plainly it is an
   internal-consistency + integrator check (code matches maths), not a test against reality; the
   "invited to make it print FAIL" line is removed; F1–F3/F3′/F6 relabelled internal-consistency
   conditions of the derivation.

6. **F5 / "Experiment 5(ii)" were cited but absent from the code** (γ₂ never set non-zero).
   → **Fixed.** A real residual-drift experiment (**Experiment 9**, F5) was added: it shows a
   frozen system relaxes to `d* = γ2/A0 > 0` when γ₂ > 0 and to 0 when γ₂ = 0. A matching pytest
   assertion was added (12 tests now pass).

## Should-fix (also addressed)

- Theorem 3 "speed irrelevant to the *trajectory*" scoped to the **asymptotic verdict** (the
  level-drift injection γ₂/r is genuinely speed-dependent at finite depth).
- "Diffeomorphism" → "orientation-preserving bijection of the open intervals, dτ/dt → ∞ at t*".
- Experiment 2 caption notes the cells fix the margin A₀/b; under a fixed budget, raising speed
  crosses the Theorem-4 threshold at `b* = A0/(γ3−1)`.
- F4 relabelled an analytic self-consistency check on the model's power-law suppression; the
  finite-capacity corrector that would discriminate exponential vs power-law is named as future
  work (§8) and is honestly not run.
- Harness verdict language no longer reads "framework survives".

## Genuine strengths the red-team confirmed

- The β > k criterion is correct, is what Theorems 1–4 prove, and is independent of every
  surviving objection. The skeptic independently reproduced 9/9 (now 10/10) and the Theorem-1
  closed form (errors 7×10⁻¹¹).
- Theorem 2's *corrected* content — the gain-only additive fraction saturates rather than
  diverges, bounded by γ1 — is true and is an honest correction of the prior draft.
- The Hard-Takeoff Coordinate-Artefact result's verdict-level content (finite-time singularity is
  a coordinate property; verdict = sign(β−k)) is correct.
- Theorem 4's compounding threshold `ρ_prop < 1` is correct and located to ~1% (3.03 vs 3.0).
- The paper's honest scaffolding — Scope box, §8 Limitations, Appendix C novelty ledger,
  mandatory blinding (§6), falsification table — is "its saving grace": the defensible positions
  it must retreat to were, in most cases, already written in the body.

---

*Generated by the Paper X adversarial red-team (5 fronts × find → independently verify →
synthesise). This report is committed as the auditable record referenced in §8.*
