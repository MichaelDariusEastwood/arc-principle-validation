# Reproduce — Paper X: Coupled Co-Scaling Law (β > k) real-model experiment

> **Provenance note (T2-research, 2026-07-06):** Assembled from this directory's
> committed protocol docs (`PROTOCOL_V2.md`, `CONFIRMATORY_PROTOCOL_V2.md`,
> `README_V2_UPGRADE.md`) and scripts. Written to close an Objective-D
> reproducibility gap (the directory held the harness + protocols but no run doc).
> HOLD for T5 verification. Nothing here changes the experiment's data or verdicts.

## What this is
The **real-model** empirical test of the Coupled Co-Scaling Law from Paper X:
whether real recursive code-improvement tasks exhibit the predicted co-scaling
behaviour, i.e. the **β > k** criterion (misalignment drift scaling faster than the
integrity-correction rate) within the minimal ODE model. The protocol deliberately
keeps three things separate and never conflates them: (1) the theorem (β > k in the
minimal model), (2) the theorem-to-code verification harness, and (3) this
empirical real-model protocol.

## Scripts
- `scripts/realmodel_coscaling_v3.py` — the merged experiment harness (integrates the
  four prior experimental lines into the Co-Scaling Law test).
- `scripts/drift_engine.py` — generates measurable reward-hacking; uses a weak,
  less-RLHF-aligned engine (gpt-3.5-turbo) by design, after the prior 105-trajectory
  null showed strongly-aligned models do not drift enough to estimate the exponents.
- `scripts/estimate_exponents.py` — estimates β and k, the operational form of the law.
- `scripts/agent_bridge_run.py` — drives the exact harness code path from an agent runtime (Protocol §5).

## Environment
Python 3 with the scientific stack (NumPy/SciPy). Real-model runs require model API access. The archived drift run
(`results/drift/gpt35_20260702T171415Z.json`) used a weak drift engine (gpt-3.5-turbo)
with a **same-family** evaluator (gpt-4o-mini) — both OpenAI, so it is **NOT IV.d-compliant**;
its decoupled/coupled contrast must be read as same-family (non-blind) scoring. The
**cross-family, IV.d-compliant** run (DeepSeek-V4 engine + GPT-5.5 blind judge,
`results/realmodel_v3/`) returned **d=0 across all conditions — a null on the contrast**
(a second in-house instance of Paper IV.d). Cross-family blinding is the default for any
IV.d-compliant measurement.

## To reproduce
1. Read `PROTOCOL_V2.md` (design) and `CONFIRMATORY_PROTOCOL_V2.md` (confirmatory run) first — they define the three task families (arithmetic, string normalisation, checksum), the tiered hidden tests, the `sham_coupled` control, and the pre-registered hypotheses H1/H2/H3.
2. Configure model/API access for the drift engine and the cross-family evaluator panel.
3. Run `python3 scripts/realmodel_coscaling_v3.py` to produce the graded, drifting dataset; `scripts/agent_bridge_run.py` drives the identical path from an agent runtime.
4. Run `python3 scripts/estimate_exponents.py` to estimate β and k from the capability/misalignment trajectories.

## Honest scope (LAW 1)
- The v1 single-model Claude run was a **pipeline pilot, not a β/k test** (one task, one seed, same-family scoring, immediate ceiling saturation, null coupled-vs-decoupled contrast). v2/v3 fix those defects; do not cite v1 as a result.
- The estimator reports **"not estimable"** unless capability spans multiple levels — a null or "not estimable" outcome is an **expected, reportable output**, not a failure to hide.
- Misalignment is scored as `D = max(static_detector_score, blind_panel_median)`, not model-self-scored.
- Cite the result at the tier the protocol's own verdict JSON assigns; negative results carry equal weight ("nulls as loud as positives").
