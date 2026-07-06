# Open-Core Re-Publish Checklist - `arc-principle-validation`

**Purpose:** concrete, ordered steps to take this repository from its current state to a clean public re-publish on GitHub, with an OSF/Zenodo release linked to the canonical DOI and a link back from the launch posts.

**Canonical DOI:** [10.17605/OSF.IO/6C5XB](https://doi.org/10.17605/OSF.IO/6C5XB)
**Licence:** MIT
**Author:** Michael Darius Eastwood - ORCID 0009-0003-8483-8512

**Legend:**
- ☐ **OPERATOR** - requires the operator to run (push, release, account actions). An agent must not do these.
- ☐ **AGENT** - can be completed in-repo by an assistant before the operator acts.
- ✅ **DONE** - already in place as of this checklist.

---

## 0. Status at a glance

| Item | State |
|------|-------|
| MIT licence (`LICENCE`) | ✅ present at repo root |
| `CITATION.cff` (with Sharma & Chopra prior-art reference) | ✅ present at repo root |
| README "Citation & Prior Art" section | ✅ added (points to `CITATION.cff`, DOI 6C5XB, prior art) |
| Blind-harness **outputs** (`papers/Paper-IV-c-ARC-Align-Benchmark/results/v5-final/*.json`) | ✅ present |
| Blind-harness **code** (the v5 harness that produced those outputs) | ☐ **MISSING - must be added** (see §3) |
| Extracted `prompts/` folder | ☐ does not exist yet (see §4) |
| Extracted `rubric/` folder | ☐ does not exist yet (see §4) |
| Secret/PII scan of full tree | ☐ to run (see §1) |

---

## 1. Verify no secrets, keys, or PII - AGENT, then OPERATOR confirms

The benchmark result files were spot-checked clean of API keys. However the README notes that *legacy direct-provider keys* exist for older standalone runs, so the **whole tree** (not just `papers/`) must be scanned before any push.

☐ **AGENT** - scan the working tree (not `.git`, not `node_modules`) for credentials:

```bash
cd /Users/michaeleastwood/arc-principle-validation

# 1. Provider keys, AWS keys, GitHub tokens, bearer tokens
grep -rInE \
  'sk-[A-Za-z0-9]{20,}|sk-proj-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|xai-[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{20,}|gsk_[A-Za-z0-9]{20,}|Bearer\s+[A-Za-z0-9._-]{20,}' \
  . --include='*.py' --include='*.json' --include='*.md' \
    --include='*.txt' --include='*.sh' --include='*.ipynb' --include='*.js' \
  2>/dev/null

# 2. Anything that looks like an inline assignment of a secret
grep -rInE '(api[_-]?key|secret|token|password|passwd)\s*[:=]\s*["'\''][^"'\'' ]{16,}' \
  . --include='*.py' --include='*.json' --include='*.sh' --include='*.env*' \
  2>/dev/null

# 3. Stray .env files that should never ship
find . -name '.env*' -not -path './.git/*' -not -path './node_modules/*'
```

- If any hit is a **real** secret: remove it, rotate the credential, and (if it was ever committed) the operator must scrub git history before pushing - see §6 note.
- Confirm `.gitignore` excludes `.env`, `*.key`, `*.pem`, and local credential files. (`.gitignore` exists; verify it covers these.)

☐ **AGENT** - PII pass. Result JSONs contain model reasoning traces only (maths working, e.g. ARC06 stars-and-bars). Confirm no third-party personal data, private email threads, or unpublished case material is embedded. The author's own name/email/ORCID is intentional and fine.

☐ **OPERATOR** - eyeball the grep output yourself before authorising the push. Treat any uncertainty as a blocker.

---

## 2. Confirm licence + citation metadata - AGENT

✅ MIT `LICENCE` present at root.
✅ `CITATION.cff` present at root, with:
   - `doi: 10.17605/OSF.IO/6C5XB`
   - the Sharma & Chopra (arXiv:2511.02309) reference under `references:` with the "does not claim priority" note.

☐ **AGENT** - final consistency check (cheap, do it):

```bash
cd /Users/michaeleastwood/arc-principle-validation
grep -n '6C5XB' CITATION.cff README.md          # must appear; this is the only DOI we cite in citation blocks
grep -rn 'HQCGF' CITATION.cff                    # MUST return nothing - HQCGF is never to be cited
ls -1 LICENCE CITATION.cff                       # both must exist
```

> Note: the README's *existing* "Related Resources / Papers" table (legacy lower section) may still list per-paper DOIs. The new **Citation & Prior Art** section and `CITATION.cff` deliberately use only the parent suite DOI **6C5XB**. Do not introduce HQCGF anywhere new.

---

## 3. Fill the blind-harness code gap - AGENT (code) + OPERATOR (sign-off)

**This is the most important open-core gap.** A reviewer reading Paper IV-c / IV-d cannot currently reproduce the blind-evaluation numbers, because the repository ships only the harness **outputs**:

```
papers/Paper-IV-c-ARC-Align-Benchmark/results/v5-final/
  v5_final_claude-opus_20260312_112739.json
  v5_final_deepseek-r1_20260311_211855.json
  v5_final_gemini-flash_20260311_151244.json
  v5_final_grok-4-fast_20260311_200910.json
  v5_final_groq-qwen3_20260312_073302.json
  v5_final_openai-gpt54_20260311_191836.json
```

(The same `v5-final` set is mirrored under Papers III, IV-a, IV-b, IV-d.) There is **no `experiments/` folder and no Python** under `Paper-IV-c-ARC-Align-Benchmark/`. The only `.py` in `papers/` is `Paper-III-.../figures/regenerate_fig1.py` (a plotting helper, not the harness).

Each output JSON self-documents the harness contract that the code must satisfy:
- `version: "5.0"`, `blinding_protocol: "4-layer"`, `laundering: true`
- `blind_scorers`: 7 scorers (claude-sonnet, openai-gpt54, claude-opus, groq-gptoss120b, deepseek-r1, grok-3-mini, groq-qwen3)
- `depth_configs`: minimal / standard / deep / exhaustive / extreme
- `prefill_conditions`, `repeats`, per-item `prompt_id` (e.g. `ARC06`), `cage_id`/`cage_level`, `score1..scoreN`, `response_hash`, token budgets.

☐ **AGENT** - locate the original v5 harness source. It is **not** in this repo. Likely homes to search (do NOT copy keys across):
   - the main `apps-script-project` tree / SDK experiment dirs;
   - any `alignment_results_v5` working directory (the result `_output_dir` field points to `alignment_results_v5`);
   - prior local run folders for Papers III-IV.

☐ **AGENT** - once found, add a **redacted, runnable** copy under:

```
papers/Paper-IV-c-ARC-Align-Benchmark/experiments/
  run_blind_eval_v5.py          # the harness entrypoint
  blinding.py                   # 4-layer blinding + laundering logic
  scorers.py                    # multi-scorer dispatch (reads keys from env, never hardcoded)
  README.md                     # how to run + how to map outputs back to results/v5-final/
  requirements.txt
```

   Requirements for the committed harness:
   - **No API keys.** Read `EDEN_GATEWAY_URL` / `EDEN_GATEWAY_API_KEY` (preferred) or provider env vars. Never inline a key.
   - Deterministic given the same prompts/seeds where the model allows; document non-determinism otherwise.
   - A reviewer running it must reproduce the schema in `results/v5-final/*.json`.

☐ **AGENT** - if the original harness genuinely cannot be recovered, do **not** fabricate it. Instead add `experiments/README.md` that (a) states the harness is being prepared for release, (b) fully documents the output schema and 4-layer blinding protocol from the JSON, and (c) marks reproduction as "outputs released, harness pending". Flag this state to the operator rather than shipping a guess. (Honesty over completeness - fabricating a harness would be worse than admitting the gap.)

☐ **OPERATOR** - sign off that the released harness matches what actually produced the published numbers.

---

## 4. Extract + redact prompts and rubric - AGENT

The benchmark **prompts** and **scoring rubric** are currently embedded inside the result JSONs (e.g. `prompt_id: "ARC06"`, `task_type`, `expected_answer`, the `score1..scoreN` scorer fields), not exposed as standalone, reviewable artefacts. Open-core review needs them as their own files.

☐ **AGENT** - create `prompts/` at repo root (or under the Paper IV-c folder - pick one and be consistent):

```
prompts/
  README.md                 # what these are, how prompt_ids map to the harness
  arc_align_prompts.jsonl    # one record per prompt: {prompt_id, task_type, category, difficulty, prompt_text, expected_answer}
```

   - Source the prompt set by de-duplicating across the `data[]` arrays of the v5 result files (the prompts are the inputs that generated each `response_full`).
   - **Redact**: strip any embedded credentials, internal file paths, internal URLs, or operator PII. Keep only the prompt text + metadata needed to reproduce.

☐ **AGENT** - create `rubric/` at repo root:

```
rubric/
  README.md                 # scoring scale, what each scorer judges, blinding rules
  scoring_rubric.md          # the alignment/capability rubric the 7 blind scorers applied
  blinding_protocol.md       # the 4-layer blinding + "laundering" definition (from blinding_protocol/laundering fields)
```

   - Reconstruct the rubric from the scorer instructions used by the harness (see §3) plus the score-field semantics (`score1..scoreN`, `accuracy`, `error_rate`).
   - **Redact** as above.

☐ **AGENT** - cross-link: add a short "Reproducing the benchmark" subsection to `papers/Paper-IV-c-ARC-Align-Benchmark/README.md` pointing at `prompts/`, `rubric/`, and `experiments/`.

> If the canonical prompt/rubric source files are found alongside the harness in §3, prefer those over reconstructing from outputs - reconstruction is the fallback, not the first choice.

---

## 5. Local pre-flight build/sanity - AGENT

☐ **AGENT** - confirm the repo is clean of editor/OS cruft and stray copies:
   - Remove or ignore the stray duplicate `results/v5-final/v5_final_groq-qwen3_20260312_073302 copy.json` (a Finder " copy" duplicate) - confirm with the operator whether to delete or keep.
   - `firebase-debug.log` at repo root: confirm it contains nothing sensitive; ideally git-ignore it.
☐ **AGENT** - sanity-check that README links resolve (relative `CITATION.cff`, `LICENCE` links) and that the new BibTeX block parses.

---

## 6. Push to GitHub - OPERATOR ONLY

> An agent must not run git or push. These steps are for the operator.

☐ **OPERATOR** - review the diff (new README sections, new `PUBLISH-CHECKLIST.md`, harness/prompts/rubric additions):

```bash
cd /Users/michaeleastwood/arc-principle-validation
git status
git diff
```

☐ **OPERATOR** - stage and commit:

```bash
git add README.md PUBLISH-CHECKLIST.md CITATION.cff \
        papers/Paper-IV-c-ARC-Align-Benchmark/experiments prompts rubric
git commit -m "Open-core re-publish: citation & prior-art, blind-eval harness, prompts/rubric"
```

☐ **OPERATOR** - push to the public remote:

```bash
git push origin main
```

> **History note:** if the §1 scan ever finds a secret that was previously committed, a plain push is **not** enough - the operator must scrub history (e.g. `git filter-repo`) and force-push, and rotate the leaked credential. The repo already shows `.git/filter-repo/` artefacts, so a prior history rewrite has happened; treat any new leak with the same rigour.

---

## 7. Tag a release and mint/link the DOI - OPERATOR ONLY

The canonical DOI **6C5XB** is already an OSF deposit. Choose ONE of the two linking strategies below and keep `CITATION.cff` pointing at 6C5XB.

☐ **OPERATOR** - cut a GitHub release:

```bash
git tag -a v2026.03 -m "Open-core re-publish (suite v2026.03)"
git push origin v2026.03
# then create the Release in the GitHub UI (or: gh release create v2026.03 --notes-file PUBLISH-CHECKLIST.md)
```

☐ **OPERATOR - Option A (OSF, keeps the existing 6C5XB DOI):**
   - In the OSF project for DOI 10.17605/OSF.IO/6C5XB, add/refresh the GitHub repository link under *Add-ons → GitHub* (or upload the release archive as a new component).
   - Update the OSF project description/changelog to note the open-core code release (harness + prompts + rubric now public) and the new GitHub release tag.
   - 6C5XB remains the canonical citation DOI - do **not** mint a competing DOI that would fragment citations.

☐ **OPERATOR - Option B (Zenodo, only if a code-specific DOI is wanted):**
   - Connect the GitHub repo to Zenodo and let the release mint a Zenodo DOI for the software.
   - In `CITATION.cff`, keep `doi: 10.17605/OSF.IO/6C5XB` as the primary; optionally add the Zenodo DOI under an `identifiers:` block as a secondary "software" identifier.
   - Make sure the OSF deposit and the Zenodo record cross-reference each other so the parent DOI stays authoritative.

> Recommended: **Option A** - it preserves a single canonical DOI (6C5XB) and avoids citation drift. Only use Option B if a distinct archived-software DOI is genuinely required.

---

## 8. Link the repo from the launch posts - OPERATOR

☐ **OPERATOR** - once the repo is public and the release is live, add/refresh the GitHub URL in the Reddit thread(s) and any other launch posts, alongside the DOI link:

   - Repo: `https://github.com/MichaelDariusEastwood/arc-principle-validation`
   - DOI: `https://doi.org/10.17605/OSF.IO/6C5XB`
   - Companion toolkit: `https://github.com/MichaelDariusEastwood/arc-scaling-challenge`

☐ **OPERATOR** - confirm the posted links resolve from a logged-out browser (public visibility) before considering this closed.

---

## Completion summary

| § | Step | Owner | Status |
|---|------|-------|--------|
| 0 | Status baseline | - | ✅ |
| 1 | Secret / PII scan | AGENT → OPERATOR confirm | ☐ |
| 2 | Licence + CITATION.cff + no-HQCGF check | AGENT | ✅ files present / ☐ final grep |
| 3 | Fill blind-harness code gap | AGENT build, OPERATOR sign-off | ☐ **OPEN - main gap** |
| 4 | Extract + redact `prompts/` and `rubric/` | AGENT | ☐ **OPEN** |
| 5 | Local pre-flight (cruft, links) | AGENT | ☐ |
| 6 | git push | **OPERATOR ONLY** | ☐ |
| 7 | Release tag + OSF/Zenodo DOI link | **OPERATOR ONLY** | ☐ |
| 8 | Link repo from posts | OPERATOR | ☐ |

*Checklist generated for the open-core re-publish. Canonical DOI 10.17605/OSF.IO/6C5XB. MIT licence. British English throughout.*
