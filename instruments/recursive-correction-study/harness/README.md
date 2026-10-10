# Stage A schedule and response custody

This is a runnable instrument-development workflow on an exact-arithmetic substrate. It does not implement the proposed representative program-repair confirmation bank and it cannot decide P12, P13, P15 or P21. The original `assay.py` is an unchanged copy of the upstream exact-arithmetic oracle; the local wrapper adds a distinct strict answer parser and schedule. Both source hashes enter the scorer identity.

## What one task/model block does

One initial generation is frozen and shared by eight policies: diagnostic or neutral-masked feedback; during-revision or terminal-panel correction; depth two or four. Each policy receives d critic calls and d+1 rewrite calls after that initial generation. The two depth-two and depth-four groups contain four policies each, giving 1+4(5)+4(9)=57 calls per model/task/replicate. Two models give 114 calls. Sixteen independent task clusters in the eventual two-model pilot would use 1,824 calls, before retries. These are workload calculations, not data or price estimates.

During-revision policies alternate critic and rewrite, then perform a common final rewrite. Terminal policies make d unaided rewrites, obtain d critiques of the same finished draft and use all critiques in a frozen order in one final rewrite. Terminal critics do not see earlier drafts. At deterministic decoding their identical requests may return identical critiques; that is recorded, not interpreted as independent votes.

The longest dependency chain is ten waves for the depth-four during-revision arm, including initial generation. Ready calls can be executed concurrently, but a later dependent prompt cannot be known before the previous response. If each wave is submitted through an asynchronous service, latency accumulates across waves.

## Diagnostic and final scoring boundaries

The task is a bounded integer calculation. The visible task contains operands and operation, not the original proposal, panel seed, task ID or a correctness label. Artefacts contain exactly one bounded integer field, `answer`. Duplicate JSON keys, booleans, non-finite values, extra fields and prose fail parsing.

At a critic call both feedback arms execute the same exact checker. Diagnostic output reports only format validity and whether the answer passes. The neutral mask replaces both fields with constants independent of truth and format. No expected answer is supplied by the diagnostic. The visible calculation itself determines the answer, so this is computational access, not a measured positive conditional Shannon-information channel. The same exact checker scores endpoints; this development task has no secret final-test bank.

Feedback and later artefacts differ as a consequence of policy. The design compares whole policies. Equal call counts are not equal token counts, FLOPs, money or wall-clock time.

## Output states and limits

- A completed, correctly formatted response is scored by the oracle, including valid but incorrect answers.
- A completed malformed response is retained. Its raw content may be reconsidered at the next scheduled call. It is an unsuccessful endpoint if it is final.
- A token-cap exhaustion, content-filter stop or refusal is a `model_unsuccessful` result with its own completion reason, raw returned content and reported usage. The next scheduled call receives a fixed unsuccessful-artefact sentinel. A final occurrence is `invalid`, not an infrastructure outage.
- A raw response over 4,096 UTF-8 bytes is kept in custody, but receives a fixed overlength sentinel downstream and is an unsuccessful final artefact. There is no outcome-dependent truncation or retry.
- Up to 1,000,000 raw bytes per call can be retained. A larger response is an incident requiring preservation of the original returned file and review; it is not silently discarded.
- A genuine missing response or infrastructure failure blocks dependent calls. All scheduled endpoints remain represented. `pending=0` can mean that branches are blocked, not that all 57 calls ran. Planned calls, recorded calls and statuses are reported separately.

The prompt byte cap is 126,976, covering worst-case JSON escaping of the five accepted visible strings at terminal depth four. This is a defensive serialisation limit, not a model context-token budget. Real execution needs tokenizer-aware input budgeting and explicit provider context checks. Output token caps are declared; recorded usage is preserved where available. Missing usage is `null`, not fabricated zero.

Incomplete collection cannot be scored by accident. `endpoints` rejects ready unresolved calls unless `--allow-aborted` is explicitly supplied. That option makes absence rows; it does not remove failed arms. No automatic retry policy is implemented. A real study needs the outage/retry rules fixed before collection.

## Provider-independent development workflow

1. Prepare the unchanged assay panel and two model-block declarations.
2. Create a pilot-labelled development study.
3. Export only ready requests.
4. Obtain responses through the declared backend, retaining the raw response files.
5. Normalise them, import them into a new append-only ledger snapshot, then export the next ready wave.
6. Once every branch completes or has an explicit failure, emit endpoints and the quality report.

Example commands, with actual input files to be supplied:

```bash
python3 harness/stage_a.py new --panel panel.json --models models.json --study-id pilot-development-01 --out study.json
python3 harness/stage_a.py pending --study study.json --out requests-01.jsonl
python3 harness/stage_a.py import --study study.json --incoming normalised-01.jsonl --out ledger-01.jsonl
python3 harness/stage_a.py pending --study study.json --ledger ledger-01.jsonl --out requests-02.jsonl
python3 harness/stage_a.py endpoints --study study.json --ledger ledger-final.jsonl --out endpoints.jsonl
```

Later imports include the previous `--ledger`. Every output path is created exclusively. Existing records are not overwritten. Hashes bind the study, exact source, raw initial artefact, request and response lineage. A hash binds content; it does not authenticate an external service or establish when a registry received a record.

A model declaration has exactly `block_id`, `model_id`, `backend`, `revision`, `revision_evidence_sha256` and `generation_parameters`. Supported backends are `synthetic`, `openai_responses` and `external_replay`. Synthetic and real-response declarations cannot mix. The actual revision evidence must be retained independently. The parser checks declaration format, not the truth of the declared provider.

## OpenAI Responses Batch adapter

```bash
python3 harness/batch_adapter.py export --requests requests-01.jsonl --out batch-wave-01
python3 harness/batch_adapter.py normalise --requests requests-01.jsonl --results downloaded-results.jsonl --out normalised-01.jsonl
```

The adapter sends nothing. It emits the documented POST `/v1/responses` JSONL shape, uses opaque unique `custom_id` values and separates files by exact model. Returned row order is irrelevant. Unknown/duplicate IDs, changed request hashes and unexpected model snapshots are rejected. A reasoning item before the assistant message does not confuse the text extractor. Unreturned IDs stay pending. Model errors, usage and actual raw response hashes remain visible.

The Batch documentation describes 50% discounted eligible requests and a 24-hour completion window per batch. That is not a promise that an entire ten-wave iterative experiment finishes in 24 hours. Model support, dated snapshot availability, generation-parameter compatibility and current account pricing must be checked before submitting. [1,2]

An external-replay backend can represent a locally run Hugging Face model, provided weights, tokenizer, chat template, generation configuration and runtime are pinned and recorded. This kit does not download weights or supply a GPU inference runner. The arithmetic pilot must first show that the chosen model/task range avoids total saturation.

## References

[1] OpenAI. Batch API. Official documentation checked 10 October 2026. https://developers.openai.com/api/docs/guides/batch

[2] OpenAI. Create a model response. Official schema checked 10 October 2026. https://developers.openai.com/api/reference/resources/responses/methods/create
