# Unit 83: counted correction inside each revision round

Does the performance advantage of correction widen as a register grows through successive revision rounds? The corrector sees the arriving entry and can repair it; the monitor sees the current register at each round, with no entry marked as new, and can accept or reject it. Unit 83 tests P21, the placement limb of Law V, the ARC Embedding Law, a law under test.

This directory preserves the exact registered implementation, `41-D-COUNTED-v0.1`, used in the run of 27 and 28 October 2026. The source is also available in the [public Unit 83 registration](https://doi.org/10.17605/OSF.IO/MN73W). The [results component](https://osf.io/98hkr/) contains the original report and its mandatory recovery disclosure. Those records distinguish the pooled performance finding, the different engine-level findings, the unevaluable mechanism axis and the post-registration recovery amendment. They do not establish the Eden Protocol as an architecture or resolve Law V as a whole.

## Contents and identity

- `registered/`: the 43 source/configuration files and their original `registration_manifest.json`. That manifest has SHA-256 `013d7008aa92a7867385d20f15e6754988e1d83d20eebc6096b20b9e5505e026`. All 44 files match the public registration and the original runtime freeze.
- `historical-bench/bench.py`: the exact sealed original bench, made by bench factory 1.2 from Unit 06 bench 1.1. It prints and checks the original manual steps. Its full kit, folder seal and private lab state are not bundled, so this is source for inspection.
- `historical-recovery/`: exact source and validation records for Recovery Amendment 01. These are an incident-specific historical supplement, not a launcher for a new run. Its original code manifest also names an addendum document that is not included here; this directory is not the complete amendment capsule.
- `RECOVERY-AMENDMENT-DISCLOSURE.md`: the unchanged public companion to the report. Read it with the [original report](https://osf.io/download/6ae361aec20b550f44957bbf/).
- `REPRODUCTION.md`: source inspection and the original execution sequence, with the permissions and dependencies that must be satisfied before execution.
- `PROVENANCE.json` and `MANIFEST.sha256`: file identities and public source links. The manifest covers the directory's files apart from itself.

The original registered source ZIP is [available from OSF](https://osf.io/download/enp9t/), SHA-256 `907c79d20c16da0b950c2ab2afc37107dc082a8f1cffb755e9829205dafb1176`. It contains the 43 files named by the separately supplied registration manifest.

## What can be checked now

From this directory, `shasum -a 256 -c MANIFEST.sha256` checks the supplied bytes without importing or running study code. The registration provides a second, dated source for the registered file hashes. Publication of the code is not a claim that its execution has been repeated here.

A live repetition additionally requires the author's permission, the exact design file, the original author attestation and access to the original OSF component. The frozen implementation uses fixed component and registration identities. Its deterministic bank depends on the registered seed and registration record; rerunning it against the same record does not select a fresh bank. An independently controlled new registration requires a separately versioned successor, with the original code preserved.

## Licence

The code, configurations, tests and data remain all rights reserved under the repository's [code licence](../../LICENCE-CODE.md). Reading for verification is permitted there; permission for other use remains the author's to give. This package grants no additional licence. Written research works follow [LICENCE-PAPERS.md](../../LICENCE-PAPERS.md). Raw requests, model replies, credentials, private lab settings and the sealed results archive are not copied into this source directory.
