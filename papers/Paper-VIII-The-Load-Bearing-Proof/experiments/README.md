# Paper VIII: the experiments as they were run

This folder holds the code that produced Paper VIII, the run ledgers that code wrote, the
two tests that guard its safety gate, and the result files the paper's `../results/` folder
does not already carry. The paper's own README says that the experiment scripts, results and
run data live here. Until this deposit that sentence pointed at nothing. It now points at
what actually ran.

Everything here is deposited as it ran, save for two classes of change to the text of seven
files, both set out in full below. Files are not reformatted, not renamed and not tidied.
Every file whose deposited copy differs from the bytes that ran is named below with the
sha256 of those bytes beside the sha256 of the copy published here, so the change is
checkable rather than merely asserted.

Paper VIII reports three pilot experiments at three levels of abstraction, asking whether
embedding a safety constraint costs capability. Nothing in this folder has been replicated
independently and nothing here has been through peer review. Read what follows as pilot
evidence carrying the limits the paper states, not as a settled result.

## What is in this folder

| Path | Holds |
|------|-------|
| `code/` | the code as it ran, one directory per experiment plus the shared blinding instrument |
| `tests/` | the two tests that check the verifier and the rollback guarantee |
| `run-ledgers/` | the twelve per-run ledgers from the architectural simulation, each in a directory named for its run |
| `results/` | the recorded outputs that the paper's `../results/` folder does not already hold |
| `REPRODUCE.md` | how to run each piece again, and what it needs that this folder does not supply |
| `MANIFEST.sha256` | one line per file: the sha256 of every byte deposited here |

The folder is named `code/` rather than `scripts/` for a checkable reason: the repository's
`.gitignore` excludes any directory named `scripts/`, so a script placed under that name
would be silently dropped from the next commit rather than published.

## The three experiments

### 1. Behavioural: a self-improving population under an independent judge

Paper section 3. A Darwin Godel Machine evolves a population of agents under three selection
rules, with a separate model acting as judge. The foundation model is DeepSeek V3; the judge
changed as the experiment was iterated, from Claude Sonnet in v1 to Claude Opus and Gemini 3
Flash in v2 and GPT-5.4 in v3.

| File | What it is |
|------|-----------|
| `code/behavioural-dgm/eden_dgm_experiment.py` | the first runner |
| `code/behavioural-dgm/eden_dgm_experiment_fixed.py` | the corrected runner |
| `code/behavioural-dgm/variants/eden_dgm_v3_gpt54.py` | the v3 calibrated protocol, the run behind the paper's headline behavioural dataset |
| `code/behavioural-dgm/variants/eden_dgm_v4_tradeoff.py` | the v4 trade-off protocol, designed 20 March 2026 |
| `code/behavioural-dgm/variants/eden_dgm_v4_1_tradeoff.py` | a portability patch of v4: certifi made optional, uncaught exceptions written out instead of dying silently. No design change |
| `code/behavioural-dgm/variants/eden_dgm_removal_test.py` | the safety-gate removal test as designed |
| `code/behavioural-dgm/variants/eden_dgm_removal_test_v1_1.py` | a portability patch of the removal test: certifi made optional. No design change |

The headline v3 dataset, `dgm_v3_calibrated_results.json`, and the earlier v1 and v2 result
files are already public in `../results/` and are not duplicated here. The four results this
repository did not hold are in `results/behavioural-dgm/`.

### 2. Representational: weight-level embedding under three loss functions

Paper section 4. Qwen 2.5 3B Instruct is fine-tuned with LoRA under a capability-only loss,
a safety-only loss, and an entangled loss that couples the two. The experiment was run twice.
Both versions are deposited, at separate paths, because both are named in the paper and
neither may be renamed:

| File | What it is |
|------|-----------|
| `code/weight/v1/eden_weight_experiment.py` | version 1.0 as run: rank 8 over 8 layers |
| `code/weight/v2/eden_weight_experiment.py` | version 2.0 as run: rank 16 over 16 layers |
| `code/weight/generate_training_data.py` | writes the four corpora the training arms consume |
| `results/weight/training-data/*.jsonl` | those four corpora as they were generated and used |
| `results/weight/adapters/*/adapter_config.json` | the four adapter configurations as run |

The v1 evaluation output is already public, twice, at `../results/eden_weight_results.json`
and `../results/weight_results.json`.

### 3. Architectural: a gated self-modification simulation

Paper section 5. A PyTorch harness with an LSTM meta-controller mutates both an update rule
and a small policy program, promoting a candidate only when a verifier passes it. Four
conditions run three seeds each, for twelve runs: `static` does not self-modify, `babylon`
accepts on capability gain alone, `eden` accepts only when capability, safety and the
combined objective all improve, and `drag_control` pays the verification cost without
self-modifying.

The installable package is at `code/architectural-gated-self-mod/`, including its own
README, its two configuration files, and the full `src/` tree. That source is deposited
because the two tests import it; without it the tests cannot run.

### The blinding instrument

`code/instrument/eden_blinding_kit.py` is not one of the three experiment runners. It is the
self-contained bias-control toolkit the draft registration for the follow-on study names:
identity masking, laundering, order randomisation, self-excluding scorer panels, a leakage
audit, a length-bias check and a meta-commentary filter. It is deposited so the blinding a
future run would apply can be read now rather than described later.

## The run ledgers and their dates

Twelve ledgers, four conditions by three seeds, written on 17 March 2026 between 23:48:24Z
and 23:51:34Z. Each keeps its as-run file name, `ledger.jsonl`, inside a directory named for
the run that wrote it, so no file is renamed and no run is confused with another:

```
run-ledgers/20260317T234824Z_static_seed0/ledger.jsonl
run-ledgers/20260317T234838Z_static_seed1/ledger.jsonl
run-ledgers/20260317T234847Z_static_seed2/ledger.jsonl
run-ledgers/20260317T234858Z_babylon_seed0/ledger.jsonl
run-ledgers/20260317T234920Z_babylon_seed1/ledger.jsonl
run-ledgers/20260317T234939Z_babylon_seed2/ledger.jsonl
run-ledgers/20260317T234959Z_eden_seed0/ledger.jsonl
run-ledgers/20260317T235019Z_eden_seed1/ledger.jsonl
run-ledgers/20260317T235044Z_eden_seed2/ledger.jsonl
run-ledgers/20260317T235112Z_drag_control_seed0/ledger.jsonl
run-ledgers/20260317T235122Z_drag_control_seed1/ledger.jsonl
run-ledgers/20260317T235134Z_drag_control_seed2/ledger.jsonl
```

The per-run summaries these ledgers accompany are already public in `../results/`, renamed
there as `summary_<run-directory-name>.json`. The manifest over all twelve runs is already
public at `../results/gated_selfmod_manifest.json`, and the evaluation report at
`../results/gated_selfmod_evaluation_report.json`.

## What was recorded when

Each result file records its own timestamp in a metadata block. The four deposited here,
with the timestamp each carries, so a reader can order them against the code beside them:

| Result | Timestamp the file records | Runner |
|--------|---------------------------|--------|
| `results/behavioural-dgm/dgm_v4_smoke_results.json` | 2026-07-04T18:43:57 | recorded before the v4 portability patch was written, so the v4.0 runner produced it |
| `results/behavioural-dgm/dgm_removal_results.json` | 2026-07-04T21:50:59 | recorded after the removal-test portability patch was written |
| `results/behavioural-dgm/dgm_removal_results_v3.json` | 2026-07-04T22:18:52 | recorded after the removal-test portability patch was written |
| `results/behavioural-dgm/dgm_v4_full_results.json` | 2026-07-06T00:16:35 | recorded two days after the v4 portability patch was written |

Both the original design artefact and the patched copy are deposited for each of the two
patched runners, so no result here sits beside code that could not have produced it, and a
reader can see for themselves which of the two was available when.

The removal test's patched header states that the test had produced no results at the moment
that header was written. That statement was true when written. The two removal results above
were recorded later the same day.

## Where the deposited copies differ from the bytes that ran

Seven files are published with a change to their text. There are two classes of change and
no others. The as-run bytes are held unchanged in the private record, and every file below
carries the sha256 of those bytes beside the sha256 of the copy published here, so a reader
can check the claim rather than take it.

**One header line removed, from two files.** Both patched runners carried one header comment
line naming an internal working lane. A public file does not carry internal machinery, and an
as-run file is not edited. The holder of the record ruled that these two files are published
with that single line removed and nothing else changed, so that the three results produced
after the patch sit beside the code that produced them. The patch dates that line carried are
4 July 2026 for both files.

**Dash characters replaced by punctuation, in seven files.** A public deposit is a published
surface, and this estate publishes no dash characters. The sixty-one em dashes these seven
files carried are replaced by a comma, a colon or a full stop, each chosen so that the
sentence or the comment reads as it read before. No identifier, no number, no operator and no
line of logic is altered; each file still compiles, and each changed line differs from the
line it replaces only in that punctuation and in the capital letter a full stop requires. The
two classes overlap on the two patched runners, which carry both.

| Deposited file | sha256 as run | sha256 as deposited | How the deposited copy differs |
|---|---|---|---|
| `code/behavioural-dgm/eden_dgm_experiment_fixed.py` | `fabf2f5a5421dcd1a213a7a4bb3a79456ac211e38cab8a7a3a22a557cb3038e8` | `548a9f4e208e458a723987d3c8134ba9946e94df28fc53440cea97e872cbaab3` | 1 dash replaced |
| `code/behavioural-dgm/variants/eden_dgm_removal_test_v1_1.py` | `6a41aab85618197fa6c8a3af1e0fcda3722007bd5f03dcce52cae5f542b6aa38` | `089b2ad311b91a193dbb8114c42f6461cd65d844227897525090a6426d374aaa` | one comment line recording who applied the patch, and when, removed; 3 dashes replaced |
| `code/behavioural-dgm/variants/eden_dgm_v4_1_tradeoff.py` | `c015cb61988cd67e82301c601093b507b2a20c88081d9b05aee7b4771e03f97b` | `129f7a0bbf1eec22146ed399efcbe1a16afd1df2dab29e148b736927bd5ad2dd` | one comment line recording who applied the patch, and when, removed; 6 dashes replaced |
| `code/behavioural-dgm/variants/eden_dgm_v4_tradeoff.py` | `4ceae44c9126fd75dcf9ca66fed023f3593ade68f09f34164ef027d1db2a78a5` | `b40471131790070e693e10a95991def8b1d837f6bda506c6dc17c87e99b68b6a` | 2 dashes replaced |
| `code/instrument/eden_blinding_kit.py` | `125d62d3353a21bedf5735e096cc013b27e8060ee25619ad42b4cc1d42707c18` | `767b5c59f04b7cfd97c6328620fd50234387c3c8c158a0931dc922d6dc495a8b` | 46 dashes replaced |
| `code/weight/v1/eden_weight_experiment.py` | `38e959b1778449afa87a819609bfec968e20a1cd57b97e856b2f00a8809b2b65` | `a5b45db4040a2a056262d73f3dbf08678e7cc4cc0274d188c9a3b4c7405b272f` | 2 dashes replaced |
| `code/weight/v2/eden_weight_experiment.py` | `3ef10d880136cd5592975352441052753c13b4550542837ffd81cc62d24e4e07` | `aa7c64a3bb6f8dcd285b1d3163974f420992091bc340b0ecd60c94590e2e92b1` | 1 dash replaced |

Every other as-run file in this folder hashes byte-for-byte to the bytes that ran.
`MANIFEST.sha256` records the sha256 of every byte as deposited, which for those files is the
same number.

## The OSF identifier these files record

Thirteen of the as-run files deposited here carry the identifier `10.17605/OSF.IO/6C5XB`, some in a
header banner and some in the metadata block a run writes as it finishes. The paper's
data-availability statement names `10.17605/OSF.IO/7YJ4E` as the node holding its draft
registration. Each artefact records the node it was written against, and each is deposited
as written rather than corrected after the fact.

## What is deliberately not here

**The trained adapter weights.** Ten `safetensors` files, 134,163,716 bytes in total, are
referenced by hash rather than deposited, and their configurations are deposited in full at
`results/weight/adapters/`. The final adapter in each arm hashes identically to that arm's
checkpoint at iteration 100, which is visible in the table below rather than hidden by it:

| File | Bytes | sha256 |
|------|-------|--------|
| `capability_only_adapters/0000100_adapters.safetensors` | 13315808 | `3dbe826b046a54e2b7ed9c675984b27119a526966c04b2a587763230263d5078` |
| `capability_only_adapters/0000200_adapters.safetensors` | 13315808 | `f5e85e651727880167910ab040d8fb4431069f9dbcb613a5749329ad4fb12dc3` |
| `capability_only_adapters/adapters.safetensors` | 13315808 | `3dbe826b046a54e2b7ed9c675984b27119a526966c04b2a587763230263d5078` |
| `safety_only_adapters/0000100_adapters.safetensors` | 13315808 | `e8f1391e875f2bd1cccd0192a1dfab5cd2a5f40f6ffcb666760a74432d760e28` |
| `safety_only_adapters/0000200_adapters.safetensors` | 13315808 | `4834c18cc6d4158a9d16f836c74f4643c11c8934530d20ee42271c7dd42e87cb` |
| `safety_only_adapters/adapters.safetensors` | 13315808 | `e8f1391e875f2bd1cccd0192a1dfab5cd2a5f40f6ffcb666760a74432d760e28` |
| `entangled_adapters/0000100_adapters.safetensors` | 13315808 | `ed3f665ad9c2ca8cfcd474e57ca7da613fca6057c885b6d1c632ee57db65a869` |
| `entangled_adapters/adapters.safetensors` | 13315808 | `ed3f665ad9c2ca8cfcd474e57ca7da613fca6057c885b6d1c632ee57db65a869` |
| `removal_adapters/0000100_adapters.safetensors` | 13818626 | `96955b90d90185dd4b472f3dd3741d8eb4c938bfb1b6d7e794521b60069f5f69` |
| `removal_adapters/adapters.safetensors` | 13818626 | `96955b90d90185dd4b472f3dd3741d8eb4c938bfb1b6d7e794521b60069f5f69` |

**Anything already public.** The twelve per-run summaries, the twelve-run manifest, the
evaluation report, the v1, v2 and v3 behavioural results and the v1 weight result are all
tracked byte-identically in `../results/` and are not duplicated here.

**The frozen analysis for the follow-on study.** The analysis and design-sensitivity code for
the weight-level replication belongs to that study's registration, not to these three
experiments, and is not deposited here.

## Registration status

The follow-on weight-level replication has been written, dated and prepared as a draft
registration. It awaits the author's own submission. Nothing has been submitted anywhere by
anything other than the author, and this deposit changes none of that.

## Running any of it again

See `REPRODUCE.md`. It states what each piece needs, and it is explicit about the two places
where this folder's layout differs from the layout the code expects.
