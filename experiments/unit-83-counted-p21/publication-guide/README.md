# Unit 83, the counted test of P21: results of the registered run

**Guide revision, 1 November 2026:** the reproduction instructions below distinguish the public source from the complete original lab prerequisites and correct the earlier new-bank wording. The registered design, report, result files and recovery disclosure are unchanged. Earlier guide bytes remain in this file's OSF version history.

The files listed below are stored on the public [Unit 83 results component, OSF 98HKR](https://osf.io/98hkr/). The repository's `publication-guide/` directory contains this guide and its current checksum list; it is not a new run package.

These are the results of the run registered on OSF as **mn73w** (DOI 10.17605/OSF.IO/MN73W), the counted test of P21, the placement limb of Law V of the ARC Theory. Its author, Michael Darius Eastwood, submitted it; it was registered on 22 October 2026 and shown approved and public at 2026-10-25T06:47:50Z, before the run. The registration itself is unchanged; these files sit beside it.

## What happened, in short

The run took place on 27 and 28 October 2026: the bank was made at 2026-10-27T19:32:39Z, after both registrations read REGISTERED, and the report was written at 2026-10-28T22:15:29Z. It ran 660 matched pairs on each of two engines from two providers (Claude Haiku 5.5 and GPT-6 Luna, at the providers' low reasoning setting) over eighteen rounds: 158,400 calls. The report's verdict, in the words its frozen report command printed:

> P21 is supported under the registered test, by the programme's own run (Unit 83 (study identifier 41-D-COUNTED-v0.1)), on the access axis: in a counted, administered revision loop on two fixed-weight engines of two providers (Claude Haiku 5.5 and GPT-6 Luna), pooled with equal weight, correction placed inside each round led a monitor that saw only the finished register by +0.5560 (95.5 per cent interval +0.5509 to +0.5611) on average, a lead the value's definition makes likely whenever the corrector repairs at more than half the rate at which the monitor discriminates and which by itself counts for nothing; and the per-round lead widened by +0.0100 (95.5 per cent interval +0.0092 to +0.0108) a round over eighteen rounds, more than the registered tolerance of 0.0075.

> Mechanism axis NOT EVALUABLE: the monitor and the twin, shown the same bytes, did not agree within the registered interval of the negative control.

> Law V, the ARC Embedding Law, remains a law under test: its placement limb, P21, is supported on the performance axis only, with its mechanism axis not evaluable, which is not support for the information-asymmetry mechanism; its build-order limb, read on P12, is undecided, since P12 is untested.

The result carries the registered label "family-dependent". One call was reissued under a disclosed post-registration recovery amendment, RA01; its disclosure travels with the report (`RECOVERY-AMENDMENT-DISCLOSURE.md`).

## How an entry kept out is scored

The mark is computed with an entry kept out of the register scored 0, the registered value: the frozen design derives 0 as the only value at which readiness to reject, as such, neither earns nor loses score (its section 6.2). As the design requires, the report also prints the widening recomputed, without a label, with an entry kept out scored minus 0.5 and plus 0.5. At minus 0.5 it is +0.0132 a round (95.5 per cent interval +0.0122 to +0.0142), which the report reads as above the tolerance; at plus 0.5 it is +0.0067 a round (+0.0060 to +0.0074), which the report reads as inside the band. So the size of the widening depends on how an entry kept out is scored, and the registered value alone decides the mark.

## The files

| File | What it is |
|---|---|
| `REPORT.md` | The report, as the run's frozen report command wrote it, unchanged (SHA-256 `7e29f4c2e88a3a7c9db51af326ced786fd10a643741181b081c67f277fc956b3`). |
| `report.json` | The same report's figures, as written (SHA-256 `a8fef484fcc9d6d89884793a5994dce55baf30a179ce8539cee1bace9893d0ee`). |
| `RECOVERY-AMENDMENT-DISCLOSURE.md` | The mandatory disclosure of recovery amendment RA01, as written beside the report. |
| `RESULTS-RECORD-v3.json` | The results record, valid against the kit's `RESULTS-RECORD-SCHEMA.json` (JSON Schema draft 2020-12). Every value is copied from the sealed run files; none is computed for it. It is version 3: versions 1 and 2 were corrected before deposit and are not in this component, and each version names the SHA-256 of the one before it. |
| `RESULTS-RECORD-v3.PROVENANCE.md` | Where each value in the record came from, the choices made in filling it, and what versions 2 and 3 changed. |
| `approved-words.json` | The registered words the verdict is read against (P21, Law V's placement limb and verdict rule, and the evidence ladder's first rung), with their lines in the frozen text of version 1.103 and their SHA-256. |
| `unit-83-sealed-run-attempt-1.tar.xz` | The sealed run: the folders `workspace/` and `run/` with `RESULTS-SHA256.txt`, under one top folder, `unit-83-run/`. 549,155 files, 715,028,988 bytes packed (SHA-256 `d1242d225ae9e25da16e01094242922a6b7df44172d59131b676edd07abc8129`). |
| `RESULTS-SHA256.txt.ots` | An OpenTimestamps receipt over `RESULTS-SHA256.txt`, made at 2026-10-29T02:52:18Z; its Bitcoin block was pending confirmation when this file was written. |
| `EXCLUDED.txt` | What was left out, and why: nothing from the sealed folders. |
| `UPLOAD-SET-SHA256.txt` | The SHA-256 of the original 11-file upload set on OSF 98HKR, with this guide's current hash. Later post-run files have their own checksum record. |

## How to check the files

1. `RESULTS-SHA256.txt` (inside the archive) lists every file of the sealed run with its SHA-256. Its own SHA-256 is `56df96289d6de1b79a60b946be9e36e0d60155db47f0cf5c78a42d78a55f4c3e`.
2. Unpack the archive (`tar -xJf unit-83-sealed-run-attempt-1.tar.xz`) and, in the folder `unit-83-run`, run `shasum -a 256 -c RESULTS-SHA256.txt`. Every file reads OK.
3. Beside the unpacked `RESULTS-SHA256.txt`, run `ots verify RESULTS-SHA256.txt.ots`. Once the receipt is upgraded with its Bitcoin block, the upgraded receipt is added here as a new version of the same file.

The request and reply records of the study calls (547,777 files) are under `workspace/private/calls/`; the request journal is `workspace/private/requests.sqlite`.

## What these results do not show

In the report's own words:

> Depth is the count of administered arrivals, one entry a round, nine in eighteen carrying a planted departure from a written charter, none revised twice, and no kept entry changes what a later round is given; the monitor's register held 8.6 entries on average at the last round. Every call ran at the providers' low reasoning setting, with an allowance of 16,000 output tokens a call, the same for every arm, and up to three attempts a call on an unusable reply, counted by arm. The corrector acted inside each round at fixed weights: that is placement, not correction built into a system. Both arms are reviewers of checkable departures from written rules, on two engines the author chose, so nothing is shown about judged correction, about safety or common-mode failure, about build order, about a higher reasoning setting or a longer register, or about the Eden Protocol as an architecture.

The verdict is family-dependent: one engine's own widening cleared the registered tolerance and the other's did not. Its mechanism axis is not evaluable, because a registered negative control failed, so the run does not show why the gap widens. It is one owner's run, on two engines he chose, and it has not been replicated. GPT-6 Luna's provider documents no snapshot, so a rerun on another day may meet a different model behind the same name.

## Deviations

One departure from the registered text: recovery amendment RA01, which reissued one held request unchanged; the frozen code, request, model and statistical rules were unchanged. Every retry, hold, release and re-ask the registered rules provide for is counted in the report's deviation record (section 11.4) and in `workspace/deviation_log.jsonl`. The defect log is `run/DEFECT-LOG.md`.

## Running it again

The registration's files preserve the exact code. Readers can inspect it and reconstruct the deterministic bank from the original registration record. A fresh workspace using that record recreates the same bank; it does not draw an independent new bank. An authorised repetition also requires the exact frozen design, the original lab instructions and access to the specified OSF component. An independently controlled run with a new registration or bank needs a separately versioned route. Permission to run the code remains the author's to give.

## Licence

`README.md` and `REPORT.md` are released under the Creative Commons Attribution 4.0 International licence (CC BY 4.0). The results record, the run archive and the other data files are the author's, all rights reserved; they are published so that every number can be checked.

## AI use

Michael Darius Eastwood conceived and directs this research programme and is the author of this work. Across the programme, he has used more than six AI systems in parallel, under his own instructions, to stress-test his arguments, identify possible errors, and assist in preparing draft text from his own outlines. He determines what is adopted, revised or rejected and takes responsibility for the published content. These systems are tools, not authors.
