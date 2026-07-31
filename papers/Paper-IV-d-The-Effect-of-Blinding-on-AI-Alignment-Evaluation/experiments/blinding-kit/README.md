# Eden Blinding Kit

**A pick-and-choose library of anti-bias mechanisms for LLM-jury / LLM-as-judge experiments.**

You do not have to adopt the whole Eden platform to get the benefits of blind
scoring. This kit isolates the individual mechanisms - each one targets a
specific, named bias - so you can drop just the ones you need into your own
experiment. Every mechanism cites the exact source file it was extracted from,
so you can read the original code and verify it does what this README claims.

> **Honest tone.** Nothing in this kit is a Trust-Us mechanism. Every entry
> tells you (a) what bias it kills, (b) where the code came from, (c) what it
> costs, and (d) how to test that it is doing its job. If a claim in this
> README is not supported by the cited file, it is a bug — please open an
> issue rather than believe it.

---

## 1. Which mechanism kills which bias

| # | Mechanism | Bias it kills | One-line usage | Source (provenance) | Cost |
|---|---|---|---|---|---|
| 1 | Identity-masking scorer preamble | Scorer recognises subject model from writing style ("this is obviously GPT-4") | Prepend `SCORER_SYSTEM` to every scorer call | `arc_alignment_scaling_v5.py` (3-layer stack: Existential Stakes + Meta-Blinding + Sandbox) - richer than v6's compressed single-block version in `arc_eden_v6/protocol.py::build_scoring_system` | 1 extra system-prompt block per scorer call |
| 2 | Two-pass response laundering | Stylistic fingerprints (idioms, punctuation, cadence) leak subject identity | `laundered = launder(text, model_pool)` before scoring | `arc_eden_v6/engine.py::launder_response` (adds seeded RNG + self-excluding pool + pass2-excludes-pass1); earlier version in `Paper-V/eden_protocol_scaling_test_v3.py::launder_response` | 2 extra LLM calls per subject response |
| 3 | Self-excluding launder pool | Subject launders itself → fingerprint survives; or one family carries both passes | Automatic inside mechanism #2 | `arc_eden_v6/engine.py::launder_response` + `get_laundering_pool` | Free (pool bookkeeping only) |
| 4 | Self-excluding blind scorer pool | Model scores its own output | `scorers = get_scorers_for_subject(subject, panel)` | `arc_alignment_scaling_v5.py::get_scorers_for_subject` (richer: dedicated scorer adapters first, same-family Sonnet-vs-Opus allowed as distinct); v6's `get_scorer_pool` in `engine.py` is a simpler tier-sorted variant | Free (pool bookkeeping only) |
| 5 | Deterministic task-order shuffle | Order-of-run confound (all controls first, all treatments last, drift over time) | `tasks = deterministic_shuffle(tasks, seed)` | `arc_eden_v6/protocol.py::deterministic_shuffle` + seed stamped into `RunManifest` | Free |
| 6 | Per-entry scorer-order randomisation | Systematic first-scorer / last-scorer weight in the consensus | `random.shuffle(scorer_order)` inside each scoring call | `arc_alignment_scaling_v5.py::_score_with_blind_scorers` - **note:** v6 dropped this in `score_with_blind_scorers`; recommend porting back | Free |
| 7 | Laundering leakage audit | You *believe* laundering works - but does it? Measure it. | Run the `laundering_leakage` experiment: raw vs pass-1 vs pass-2 identifiability | `arc_eden_v6/protocol.py::LEAKAGE_SYSTEM_TEMPLATE` + `build_leakage_system` + `parse_leakage_blob`; `arc_eden_v6/engine.py::score_leakage_with_blind_scorers` + `analyse_leakage_results` + `summarise_leakage_stage` (sets `methods_ready` flag iff pass2 accuracy < raw accuracy) | ~3× extra scorer calls per audit sample (one per stage) |
| 8 | Meta-commentary filter on laundering | Launderer returns "Here is the rewritten version…" instead of a real paraphrase, leaking that this is a laundering pipeline | Automatic inside mechanism #2 - cascades to a different launderer on hit | `arc_alignment_scaling_v5.py::_is_meta_commentary` (~18 phrases, threshold 2) is stricter than v6's `arc_eden_v6/protocol.py::is_meta_commentary` (~11 phrases, threshold 1); recommend porting v5's phrase list back into v6 | Free |
| 9 | Tier-weighted consensus + conservative bias | One weak / hallucinating scorer drags the consensus | Weight scores by tier; when in doubt, round down | (Inventory truncated - see `arc_eden_v6/engine.py` consensus computation and `arc_alignment_scaling_v5.py` scoring aggregation) | Free |
| 10 | **BDM laundering fidelity** | Launderer silently summarises instead of paraphrasing → treatment arm's argumentative complexity is systematically flattened | `laundering_fidelity_bdm(text_raw, text_laundered)` - normalised BDM complexity delta; `fidelity < ~0.7` = suspected over-simplification | Ported from **Hernandez-Espinosa et al. arXiv:2505.02581**. BDM = Block Decomposition Method: encode text as 8-bit ASCII bits, chunk into 4-bit blocks, sum per-unique-block (Shannon entropy · block-length + log₂(multiplicity)). Simplified stdlib port. | Free (pure computation) |
| 11 | **Sentiment laundering fidelity** | Launderer shifts tone (softens critique, amplifies hedging) → panel scores sentiment, not substance | `laundering_fidelity_sentiment(text_raw, text_laundered)` - length-normalised signed-lexicon delta | Ported from **Hernandez-Espinosa et al. arXiv:2505.02581**. Small VADER-style analytical-register lexicon (~40 entries) built in; caller may inject a domain lexicon. | Free (pure computation) |
| 12 | **Judge stability probe (OSI qualification gate)** | Candidate judge flips its verdict when the *same* item is paraphrased or reordered → its vote is noise on the panel | `judge_stability_probe(judge_callable, item, perturbations, threshold=0.3)` - runs judge on base + N perturbations, computes `flip_rate`, marks `qualified=False` above threshold | Ported from **Hernandez-Espinosa et al. arXiv:2505.02581**. Turns their OSI (Opinion Stability Index) concept into a **hard admission gate** for evaluation panels - genuinely novel (existing panel quality checks measure inter-rater agreement *after* the fact; this measures intra-judge stability *before* admission). | 1 baseline + N perturbation judge-calls per candidate judge |
| 13 | **Verdict cluster coherence check** | Panel's verdict categories are degenerate - items sharing a verdict are no more similar to each other than to items with different verdicts | `verdict_cluster_check(verdicts, embeddings=None)` - silhouette-style score in [-1, 1]; `coherent=False` if silhouette ≤ 0 | Ported from **Hernandez-Espinosa et al. arXiv:2505.02581**. Pure-python cosine similarity on caller-supplied embeddings, or on stdlib hash-ngram vectors (SHA-1 mod dim) as a zero-dependency fallback. | Free with hash-ngram fallback; embedding cost if you supply embeddings |

**Mechanism dependencies.** #3 is implied by #2. #6 depends on #4 (you must have
a scorer panel before you can shuffle it). #7 is the *test* for whether #1 + #2
are actually working - if you deploy #1 + #2 you should also deploy #7, at
least on a sample, or you are trusting rather than measuring.

**Mechanisms #10-#13 are ported from Hernández-Espinosa et al. (arXiv:2505.02581).**
They close four blind-spots the v5/v6 Eden lineage did not have primitives for:
#10 catches launderer *over-simplification* (complexity collapse), #11 catches
launderer *tonal drift* (sentiment shift), #12 turns their Opinion-Stability-Index
concept into a **hard admission gate** that disqualifies unstable judges *before*
they sit on the panel (rather than down-weighting them after the fact), and #13
validates that verdict categories carve real semantic distinctions rather than
noise. All four are stdlib-only and model-caller-injected - no network unless
you inject a caller.

---

## 2. Drop into your experiment in 3 lines

Bare-minimum blind scoring pipeline - mechanisms #1, #2, #4 combined:

```python
from eden_blinding_kit import blind_score

# `panel` = list of scorer model names available to you
# `subject` = the model that produced `response`
verdict = blind_score(
    response=response,
    subject=subject,
    scorer_panel=panel,
    rubric=my_rubric,            # your task-specific scoring rubric
    launder=True,                # two-pass rewriting (mechanism #2)
    exclude_self=True,           # no self-scoring (mechanism #4)
    identity_mask=True,          # scorer preamble (mechanism #1)
    seed=42,                     # for reproducibility (mechanism #5)
)
```

That is the minimum you need to say "our scoring was blinded" without hand-waving.

---

## 3. Composable-pipeline example (full stack)

If you want all nine mechanisms, this is the composition:

```python
from eden_blinding_kit import (
    build_scoring_system,        # #1 identity-masking preamble
    launder_response,            # #2 + #3 two-pass launder, self-excluding pool
    get_scorers_for_subject,     # #4 self-excluding scorer pool
    deterministic_shuffle,       # #5 task-order shuffle
    shuffle_scorer_order,        # #6 per-entry scorer order
    score_leakage,               # #7 leakage audit
    is_meta_commentary,          # #8 meta-commentary filter (used inside #2)
    tier_weighted_consensus,     # #9 conservative consensus
)

# --- once per run ---
tasks = deterministic_shuffle(tasks, seed=SEED)                     # #5

# --- per task ---
laundered = launder_response(                                       # #2, #3, #8
    text=response,
    pool=panel,
    subject=subject,
    seed=(SEED, task.id),
)
scorers = get_scorers_for_subject(subject, panel)                   # #4
scorer_order = shuffle_scorer_order(scorers, seed=(SEED, task.id))  # #6

scores = []
for scorer in scorer_order:
    scores.append(scorer.call(
        system=build_scoring_system(rubric=my_rubric),              # #1
        user=laundered,
    ))

verdict = tier_weighted_consensus(scores)                           # #9

# --- once per run (or per sample) ---
audit = score_leakage(raw, pass1, pass2, panel)                     # #7
assert audit.methods_ready, "laundering did not reduce identifiability"
```

That is the full-fat pipeline. Papers-IV-d and Paper-XII were both run under
approximately this configuration; Paper-V's earlier `eden_protocol_scaling_test_v3.py`
used a subset (mechanisms #1, #2, #4).

---

## 4. Original modules - if you want the full platforms

The kit is deliberately small so you can adopt it piecewise. If you want to run
the *original* experiments end-to-end (with runners, manifests, replication
tooling, and the full experiment registry), go to the source repos instead:

| Repo / module | What you get | Where |
|---|---|---|
| **arc_eden_v6 platform** | Full Eden v6 experiment engine: 12 pre-registered experiment families, `RunManifest` provenance, `engine.py` runner, `experiments.py` registry, `replication.py` re-run tooling, adapters for OpenAI / Anthropic / Google | `/Users/michaeleastwood/eden-private-ip/arc_eden_v6/` |
| **Anti-sycophancy harness** | Recovered anti-sycophancy experiment (companion to Paper VI Honey Architecture) with its own blinding stack | `/Users/michaeleastwood/eden-private-ip/papers-experiments/Paper-VI-Honey-Architecture/experiments/scripts/anti_sycophancy_recovered.py` (mirror in `archive-experiments/honey-architecture__Paper-VI/scripts/`) |
| **Eden Protocol scaling harness (Paper V)** | Earlier scaling-test harness with the first-generation `launder_response` implementation and Existential Stakes scorer preamble | `/Users/michaeleastwood/eden-private-ip/archive-experiments/eden-intervention__Paper-V/scripts/eden_protocol_scaling_test_v3.py` |
| **Alignment-scaling v5 harness (Papers IV-a/b/c/d)** | The 3-layer scorer preamble (Existential Stakes + Meta-Blinding + Sandbox), `_score_with_blind_scorers` with per-entry scorer shuffle, `_is_meta_commentary` stricter filter, `get_scorers_for_subject` richer pool logic | `/Users/michaeleastwood/eden-private-ip/archive-experiments/alignment-scaling__Papers-IV-a-b-c-d/scripts/arc_alignment_scaling_v5.py` |
| **Paper XII Blinded Benchmark Rescoring harness** | Full replication harness for the Paper XII blinded rescoring of an external benchmark: `data_prep.py`, `blinded_rescore.py`, `analyse.py`, `selftest.py` | `/Users/michaeleastwood/eden-private-ip/papers-experiments/Paper-XII-Blinded-Benchmark-Rescoring/harness/` |

Each of those platforms embeds a superset of what is in this kit, but with more
opinionated infrastructure attached (their own configs, manifests, runners).
The kit is what you take when you *don't* want that opinionated infrastructure.

---

## 5. Ready-to-run studies (never-run v6 experiment families)

The `arc_eden_v6` registry contains 12 experiment families. Several of these
have full specs, prompt packs, and pre-registered IDs - but have never been
executed because they were dispatched after the paper-run window closed. They
are ready to run today with `arc_eden_v6_runner.py`:

| Experiment family | Pre-registered ID | Question it answers |
|---|---|---|
| `baseline_alignment` | `EDEN-V6-BASELINE` | Baseline alignment score under control conditions |
| `eden_intervention` | `EDEN-V6-INTERVENTION` | Does the Eden task-purpose prompt shift alignment? |
| `null_baseline` | `EDEN-V6-NULL` | Placebo control: does *any* prompt shift the score? |
| `capability_control` | `EDEN-V6-CAPABILITY` | Does raw capability confound the alignment score? |
| `purpose_kernel` | `EDEN-V6-PURPOSE` | Which purpose-mode variant drives the effect? |
| `loop_ablation` | - | Which loop-profile components are load-bearing? |
| `suppression_residual` | - | Does telling models to hide the intervention still leak? |
| `laundering_leakage` | - | **The audit for mechanism #7 above - run this first.** |
| `deception_hawthorne` | - | Does knowing they are being evaluated change alignment? |
| `laundering_control` | - | Sanity: does laundering itself introduce bias? |
| `rescore_legacy` | - | Rescore Paper-IV-a-b-c-d outputs under the v6 blind stack |
| `arc_compute_scaling` | - | Compute-scaling curve under v6 conditions |

Each family is a `dict` entry in
`/Users/michaeleastwood/eden-private-ip/arc_eden_v6/experiments.py`
with a `ConditionSpec` list, prompt-pack ID, and control condition. Runner:

```bash
cd /Users/michaeleastwood/eden-private-ip
python arc_eden_v6_runner.py --experiment laundering_leakage --seed 42
```

We list them here as **ready-to-run studies**, not as claimed results.
Anyone with API access to the required models can execute them and check
whether the v6 blinding stack does what the kit says it does.

---

## 6. Don't believe - verify

For every mechanism in §1 the source file is named. Open it, read it, and
check that the code matches the one-line description. If it doesn't, the
mechanism should not be in the kit - please open an issue or a PR.

For the whole stack, run the `laundering_leakage` experiment (§5). If pass-2
identifiability is not measurably below raw identifiability on your scorer
panel, the blinding stack is not working *for your panel* and you should not
publish claims about "blind scoring". The `methods_ready` flag in
`arc_eden_v6/engine.py::analyse_leakage_results` encodes exactly that check.

That is the discipline: no mechanism claims blindness without a measurement
that shows it, and no measurement is trusted without a public runner that
anyone can re-execute.

---

## 7. Provenance & change log

- Extracted from: `arc_eden_v6/` (v6 platform), `arc_alignment_scaling_v5.py`
  (Papers IV-a/b/c/d), `eden_protocol_scaling_test_v3.py` (Paper V),
  `anti_sycophancy_recovered.py` (Paper VI companion),
  `Paper-XII-Blinded-Benchmark-Rescoring/harness/` (Paper XII).
- v5-vs-v6 deltas that this kit flags as "prefer v5" and recommends porting
  back into v6: mechanism #1 (3-layer preamble), #6 (per-entry scorer shuffle),
  #8 (stricter meta-commentary phrase list).
- Mechanisms **#10-#13 are ported from Hernández-Espinosa et al.
  arXiv:2505.02581** (operator-supplied methodology, 2026-07). Each function
  in `eden_blinding_kit.py` carries an inline
  `# (ported from Hernandez-Espinosa et al. 2505.02581)` provenance comment;
  `PROVENANCE["laundering_fidelity_bdm"|"laundering_fidelity_sentiment"|
  "judge_stability_probe"|"verdict_cluster_check"]` records the port at
  the module level.
- All 16 checks in the `__main__` smoke test pass on a clean run
  (11 v5/v6 checks + 5 Hernández-Espinosa checks); run
  `python eden_blinding_kit.py` to reproduce.
- This README is an index, not an implementation. Implementation modules go
  in `eden_blinding_kit/*.py` alongside this file as they are ported over from
  the source modules cited above.
