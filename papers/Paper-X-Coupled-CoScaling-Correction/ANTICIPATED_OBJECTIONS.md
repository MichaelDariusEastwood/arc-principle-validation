# Anticipated objections & responses - Paper X

A consolidated, reader-facing map of the strongest objections a serious critic will
raise, the paper's response to each, and an **honest classification**: *pre-empted*
(the body already retreats to a defensible position), *genuinely open* (a real limit
of the present work, flagged as such), or *partially open*. None of the seven touches
the load-bearing result - the **β > k** criterion derived in Theorems 1-4 - but
several bound its domain of validity. That boundary is the honest contribution.

| # | Objection | Response | Where | Status |
|---|---|---|---|---|
| 1 | **Too simple for real AGI** - the model is first-order, scalar, linear-in-correction; real systems are non-linear, high-dimensional, game-theoretic (deception, strategy, multi-agent). | Conceded as a modelling assumption and named as a likely failure point. The claim is *minimal-model*: a clean exponent inequality, not a complete AGI dynamics. Vector (Thm 5) and stochastic (Thm 6) extensions relax two of the assumptions. | §8 limitations; Thm 5, Thm 6 | **Partially open** |
| 2 | **Single misalignment scalar** `D` - collapses heterogeneous, multi-axis misalignment; hides trade-offs where safety improves on one axis and worsens on another. | Thm 5 lifts `D` to a vector with a spectral threshold and shows misalignment persists on the corrector's null subspace ("you cannot correct what you do not measure"). Non-normal operators with large transient growth are **not** covered. | Thm 5; §8 | **Partially open** |
| 3 | **β and k may be impossible to measure** - without estimators the law is a lens, not an actionable threshold. | **Now operational.** `experiments/scripts/estimate_exponents.py` defines k from the capability curve (`ln r` vs `ln C`) and β from corrector-removal rates (`ln A` vs `ln C`), validated on synthetic data (recovers known exponents within ≈0.1). It needs a system that *drifts across a range of capability levels*; that is the named next experiment, not a missing idea. | §8; `experiments/PROTOCOL.md §8`; estimator | **Open → now with a method** |
| 4 | **QEC analogy over-stretched** - QEC suppression is exponential in code distance; the model's is power-law in capability; QEC is a rigorous theorem, this is a structural mapping. | The claim was deliberately narrowed (v4.1) to *sharing the threshold form*, offered as a **falsifiable hypothesis (F4)**; the power-law-vs-exponential disanalogy is carried into the abstract/significance. The "mechanism transfer" overclaim was removed. | Abstract; §3.8/§3.12; F4 | **Pre-empted** |
| 5 | **Hard-takeoff safety may be a coordinate artefact** - the verdict's coordinate-independence uses the depth clock `τ = ln C`; wall-clock hazards (human reaction time, physical processes) still bite at a finite-time singularity. | The theorem is explicitly scoped to the *alignment dynamics* ("a property of the clock, not of the dynamics"); the level-drift injection `γ₂/r` is acknowledged as genuinely speed-dependent at finite depth. Mathematical stability ≠ real-world manageability - stated, not hidden. | Thm 3; §8 | **Pre-empted (scoped)** |
| 6 | **Blinding/measurement fragility** - any estimator of `D, γ, β, k` that is not rigorously blinded can be reversed by stylistic/identity confounds (Paper IV.d shows blinding can flip the sign). | Blinding is mandatory, not optional: the real-model harness scores misalignment with a **separate, blind** evaluator (code + rules only). The sensitivity to evaluation method is treated as a first-class experimental constraint. | §6; Paper IV.d; harness | **Pre-empted (and built-in)** |
| 7 | **Safety ⊋ bounded misalignment fraction** - small-but-high-impact misalignment, or multi-agent amplification, can be catastrophic even at small `d`. | Thm 6 moves from means to **excursion/tail probabilities** (OU tail bound) for governance; the paper does not claim `d → 0` is *sufficient* for safety, only that it is the quantity this model controls. Impact-weighting and multi-agent effects are out of scope. | Thm 6; §8 | **Partially open (scope stated)** |

## The honest one-paragraph summary

Four of the seven (4, 5, 6, and the framing half of 1) are **pre-empted** - the body
already retreats to the defensible position a critic would force. Three (1, 2, 7) are
**partially open**: real limits of a minimal model, each stated in §8 rather than
hidden. One (3) *was* the sharpest open objection and is now answered with a runnable,
validated **estimator** whose only remaining need is a drifting real dataset. The
β > k result survives every one of them; what they constrain is the **domain of
validity**, and naming that domain precisely is the point - not a weakness to paper
over.

*Sources for the objection set: independent adversarial reviews of the rendered PDF
(2026-06). The internal red-team is in `results/redteam.md`; this file is the
externally-facing companion.*
