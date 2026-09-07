# Instrument-kit integration review — 7 September 2026

This change repairs the reference software deposited through public PR #38. It does not change a
theory prediction, release a deciding instrument, validate an assay, or report an empirical result.
The original review identified seven defects after the existing 515 tests had passed. That baseline
was independently rerun successfully before these repairs.

| Review finding | Repair | Remaining boundary |
|---|---|---|
| Submitted Python can inspect hidden checks and forge the grader's last output line | The shipped in-process and subprocess graders carry a development-only marker. Confirmatory preflight and sealed-apparatus checks refuse them, including marked partials and wrappers. Adding an exact-check marker does not remove this refusal. | No secure general-purpose Python judge is supplied. An external judging boundary must protect the test oracle, result channel and host, and be independently validated. A container around a shared interpreter alone does not do that. |
| Reusing an output directory truncates old evidence and mixes progress | Bundle directories are created exclusively. Individual final files are also created exclusively. The P5 initial manifest and later sealed manifest have separate names. Progress is flushed and its read cursor advances only after a successful write. | This is an evidence-preservation fix, not a tamper-proof storage service. Keep independently anchored copies. Use a fresh directory for each run. |
| P16 replay trusts derived arm summaries | Collection and replay use the same deterministic analysis function. Replay reconstructs every arm from its round readings and checks the arm, replicate, exposure and complete round schedule. Cached estimates, alarms and component inputs do not govern replay. | P5 still exposes separate bank re-estimation and saved-summary verdict replay; this change does not claim full P5 raw-data reanalysis. |
| A nonempty date can pass the prior-inspection gate | Parse explicit timezone-aware timestamps. Preflight rejects malformed or future attestations; final custody rejects attestations later than the seal. Equal timestamps are accepted at the legacy seal's one-second resolution. | Parsing cannot establish truthfulness or authenticate an attester. Independent receipts and review remain necessary. |
| Parity ignores extra code | Extra Python files now fail parity. Missing or empty trees also fail rather than passing vacuously. | Parity compares the requested file class; it does not validate scientific correctness or prove absence of non-Python executable material. |
| Supplied ratios can contradict Q and W | When Q and W are present, derive the declared ratio or log ratio from them during collection and replay. Wrong round identifiers are refused. A balance elasticity remains a window estimate. | The Q and W measurements themselves still need instrument validation. |
| An unvalidated 0.052 default can reach confirmatory scoring | Both preflight and P16 scoring require a complete-procedure calibration record bound to configuration and implementation. The record must cover the declared observation quantities and required adverse worlds, with whole-run counts for both false support and false refutation. Exact binomial upper bounds account for finite simulation size and simultaneous coverage. | No qualifying calibration is supplied here. The legacy alarm/run-pattern battery cannot be relabelled as calibration of the final P16 wrapper. |

## Calibration contract

`arc_runner/calibration_gate.py` documents the accepted record schema. The caller supplies the
registration reference, evidence location and digest, reviewer identity, numerical error limit and
simultaneous confidence level. This patch chooses none of those scientific thresholds. Each world
has a frozen specification digest and counts from independent complete runs; individual looks,
arms or repeated readings must not be counted as independent runs.

The declared worlds must include null, positive-control, wrong-location, wrong-timing, wrong-slope,
failed-delivery and correlated-noise cases for every declared observation quantity. These are minimum
software adversaries, not exhaustive scientific coverage. The unit owner must also register the
noise, missingness, delays, model rivals, population and parameter ranges the assay actually faces.
Counts are checked using simultaneous one-sided Clopper–Pearson upper bounds over all declared
worlds and both error directions. Schema checks cannot authenticate the evidence, the reviewer's
identity or the claimed independence of runs. Those are independent-review obligations.

Changing any configuration field or implementation invalidates the calibration binding. A new
record must describe that exact procedure; do not copy the old digest, set a boolean approval flag,
or delete the guard. Passing this record validator establishes only that the supplied record satisfies
the software contract. It does not establish universal type-I error control or scientific truth.

The shipped CLI's default Python grader therefore cannot start a confirmatory run. Use its development
commands for development. Release of a deciding adapter is separate work; the Python API can accept
an independently reviewed apparatus and calibration record. No provider is called by these tests.

## Evidence and adoption

The pre-repair battery artefacts retain their original bytes and hashes as historical development
evidence. They describe the previous implementation and do not calibrate this repaired one. The
test-only addition grader recognises a restricted authored fixture language without executing the
submitted source. The test-only calibration record uses explicitly invented counts. Neither is a
released instrument, real provider result or evidence that any ARC prediction holds.

Apply the same functional changes to the canonical kit and the public copy; preserve their intentional
publication wording differences. Before a deciding unit adopts this code, record the precise code
identity, reconcile its charter and schema, supply the validated apparatus and calibration, rerun the
required verification, and issue a fresh versioned package under the existing defect protocol.

No existing evidence, paper, preregistration, model response, deployment artefact or research outcome
should be silently rewritten to make it fit the repaired runner.
