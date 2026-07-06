# Zenodo / OSF Release Runbook - arc-principle-validation & arc-scaling-challenge

Both repos are **already linked to GitHub**. `CITATION.cff` + `.zenodo.json` + `RELEASE-NOTES-v1.0.md` are in place. Re-publishing the open-core science = **commit → push → release → mint Zenodo DOI**. (Eden Legal app, production pipeline and keys stay private - the open-core boundary.)

Tags: **[OPERATOR]** = needs your GitHub/Zenodo/OSF login · **[READY]** = already prepared.

1. **[OPERATOR] Finalise + secret-scan** (per `PUBLISH-CHECKLIST.md`): scan for keys/PII; redact prompts + rubric into `prompts/` and `rubric/`; decide the blind-harness gap (publish the harness redacted, or publish outputs + a reconstruction spec and say so - do not fabricate).
2. **[OPERATOR] Commit + push** (per repo):
   `git add -A && git commit -m "Open-core release v1.0: CITATION, Zenodo metadata, README citation, reproducibility docs" && git push origin main`
3. **[OPERATOR] Enable Zenodo-GitHub integration:** zenodo.org → log in with GitHub → *Account → GitHub* → toggle the repo **ON** (do this for both repos).
4. **[OPERATOR] Create a GitHub release/tag** (e.g. `v1.0.0`). Zenodo auto-archives the release and **mints a Zenodo DOI** using the `.zenodo.json` metadata.
5. **[OPERATOR] Back-link the DOI:** add the new Zenodo DOI to `CITATION.cff` (identifiers) and the README; commit + push.
6. **[OPERATOR] OSF (project 6C5XB):** add the Zenodo DOI + the GitHub release URL as linked resources so OSF ↔ Zenodo ↔ GitHub cross-reference.
7. **[READY] Link from posts:** the Reddit/Quora/LessWrong drafts and the priority page already point to the GitHub repo + OSF DOI 6C5XB; once the Zenodo DOI exists, add it to the priority page and the flagship articles.

**Why this matters:** a GitHub release + Zenodo DOI gives a permanent, citable, timestamped record - the strongest protection of priority - and the public code is what makes frontier models cross-reference and attribute the work to you. Never cite OSF DOI HQCGF.
