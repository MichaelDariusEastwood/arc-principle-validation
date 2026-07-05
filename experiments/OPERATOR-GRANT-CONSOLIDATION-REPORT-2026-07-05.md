# Operator ask DONE — grant documents consolidated into ONE canonical location (2026-07-05)

**Operator (verbatim):** "why is there so many places my grant documents are saved, so many PRs, so many different repos folders, where is the one place that is the canon?? it should be in the infinite architects website folder: /Users/michaeleastwood/infinite-architects-ultimate-website/grant-applications where is this out of date!!! merge it all into here!!"

## Root cause of the sprawl (the honest answer)
The operator's designated canon — `infinite-architects-ultimate-website/grant-applications/` — **was never git-tracked (0 tracked files in the website repo).** Because it was outside version control, it silently drifted: it held only an **older PDF-only export dated 2026-06-16**. Meanwhile the live, git-tracked work accumulated in `apps-script-project/pm/research/website-private-archive/grant-applications/` (**754 files, full HTML+MD+PDF per funder, updated 2026-07-05**) — because swarm lanes home in apps-script-project. Result: the "canon" was the stale one; the "archive" was the real one. Plus scatter: a stray `/grants/one-pager`, `grant-kit` (canon numbers), and a skill dir.

## All grant locations found
| Location | Was | Now |
|---|---|---|
| `infinite-architects-ultimate-website/grant-applications` | 25 funders / 150 files / PDF-only / June-16 / **untracked** | ★ CANON — 25 funders / 653 files / full HTML+MD+PDF / today |
| `infinite-architects-ultimate-website/grants/one-pager` | stray 2nd folder | folded → canon `/_one-pager` |
| `apps-script-project/pm/research/website-private-archive/grant-applications` | rich working-archive (source) | → should become a POINTER (follow-up) |
| `apps-script-project/pm/research/grant-kit` | canon-support (ASK-POLICY, CANONICAL-NUMBERS) | stays (support, not applications) |
| `arc-principle-validation/.claude/skills/sci-research-grants` | a SKILL (methodology) | stays (not documents) |

## What I did (additive + reversible, LAW 3)
- rsync `--update` (newer-wins) `--no-delete` from the rich archive → canonical folder: **150 → 653 files**. Nothing overwritten unless the source was newer; nothing deleted.
- Removed cruft (`.tmp`, `.~lock`, `.DS_Store`); folded the stray one-pager.
- Wrote `_CANONICAL_SOURCE.md` declaring the folder the single source + the go-forward rule.
- **Held `emails/`** (grant correspondence) OUT — may be private, and this repo is public-facing. Operator to decide.

## Where I had been working from (honest)
This session I did **not** create or edit any grant files — I only READ/verified them (the 25-app count, £624k ask, 78% qualifier) from the pm/research archive. My authored edits this session were all in `arc-principle-validation` (papers). The sprawl predates my session.

## NOT done (needs operator/T6 — LAW 5)
- The 653 consolidated files are **untracked on disk, NOT committed or pushed** (the website repo is on a T4-research PR branch). Committing + any live deploy is operator/T6. **⚠ T4/T6: do not `git clean` the website working tree — 428 untracked grant files await commit.**
- Recommend: commit the canonical folder on a proper branch; then reduce `pm/research/website-private-archive/grant-applications` to a pointer (apps-script owner) so no parallel copy remains.
- `emails/` placement decision.
- Grant SUBMISSION remains operator-only (LAW 5).
