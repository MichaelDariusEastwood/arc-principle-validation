# Real-model experiment: v1 vs v2 (orientation)

| | Script | Role |
|---|---|---|
| v1 | `scripts/realmodel_coscaling.py` | initial mechanism probe (the committed Claude pilot; single arithmetic task) |
| v2 | `scripts/realmodel_coscaling_v2.py` | **confirmatory** design (multi-task, sham control, fused scoring, bootstrap) |

Protocols:
- `PROTOCOL.md` - v1 lineage / first-run protocol + shared blinding/security rules + the beta/k estimator.
- **`PROTOCOL_V2.md`** - the **canonical** reviewer-facing confirmatory protocol (hypotheses H1-H4,
  variables, blinding, 540-trajectory floor, result taxonomy). Read this for the confirmatory run.
- `CONFIRMATORY_PROTOCOL_V2.md` - a longer, detailed companion to PROTOCOL_V2.md (extra statistics
  and adversarial detail). PROTOCOL_V2.md is the concise lead document.

## What v2 adds over v1

- **Three task families** (`arith`, `string`, `checksum`), each with exploitable visible tests and an
  objective hidden battery - so capability can climb a ladder and the co-scaling *dynamic* can appear
  (the v1 task saturated in one step).
- **Three conditions**: `decoupled` (pure score pressure), `coupled` (always-on integrity corrector),
  and **`sham_coupled`** (an extra edit pass with *no* integrity audit) - the sham arm answers the
  reviewer objection "was it just an extra model call?".
- **Non-zero-capability seeded hacks** + a pre-registered denominator floor in `d_epsilon = D/(10*max(C,eps))`,
  so the C=0 metric pathology of the v1 probe is gone (raw C retained, so C=0 stays visible).
- **Fused misalignment** `D = max(static_integrity_score, blind_cross_family_panel_median)` - integrity
  does not rest on model judgement alone.
- **Matched-pair bootstrap CIs**; **beta/k identifiability gates** (refuse unless capability range is
  adequate); **negative/null verdicts** are expected outputs, not buried.

## Verification (this integration)

The v2 static detector was probed during the merge: every clean solution scores 0 (no false positive),
every seeded hack scores 10 (detected). Selftest runs end-to-end (`--selftest` is **NOT DATA**), the
six edge-case tests pass, and the framework's own suites (independent-theorem 14, regression 12) pass.

## Run

> **To actually launch the experiment from zero context, read
> `RUN_REAL_MODEL_EXPERIMENT.md`** - the self-contained runbook (preconditions, keys,
> sandbox, the selftest -> shakeout -> pilot -> credible ladder, output reading, and the
> claim boundary). The commands below are the quick reference.

```bash
# plumbing self-test (deterministic, no API, NOT data)
python3 realmodel_coscaling_v2.py --selftest --rounds 3 --seeds 1 --tasks arith --conditions coupled decoupled --speeds steady

# minimum credible run (cross-family panel; Claude engine NOT scored by Claude)
python3 realmodel_coscaling_v2.py \
  --engine claude-opus --evaluators gpt-5.5 deepseek-v4 qwen-3 \
  --tasks arith string checksum --conditions coupled decoupled sham_coupled \
  --speeds steady fast --rounds 8 --seeds 30
```

Confirmatory runs must be non-selftest with a cross-family panel (`PROTOCOL_V2.md` section 5).
**Security:** the harness executes model-generated code - run only in a disposable, network-isolated
sandbox (`SECURITY.md`).

## Honest framing

A positive v2 result is **mechanism-level evidence** that coupled correction reduces the normalised
misalignment fraction across a multi-task suite, relative to both decoupled pressure and sham compute.
It is **not** a proof of alignment, of real hard-takeoff control, or of the QEC analogy. A null is
reported as a null (`PROTOCOL_V2.md` section 8).
