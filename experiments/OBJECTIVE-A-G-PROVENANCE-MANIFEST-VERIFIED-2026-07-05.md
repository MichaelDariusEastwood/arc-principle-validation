# Objective A + G — provenance manifest VERIFIED current (23/23), + LAW-3 merge-sequencing catch

**Lane:** T2-research · 2026-07-05 · Verified (not asserted): ran `shasum -a 256 -c PAPER-PDF-SHA256SUMS.txt`.

## Verified positive (Objective A "each paper SHA-256" + Objective G "verify-yourself")
- **23 paper directories, 23 PDFs, 23 HTMLs** — every paper has a PDF.
- **SHA-256 manifest `PAPER-PDF-SHA256SUMS.txt` already exists** (LAW 4 — did not duplicate) and **VERIFIES CLEAN: 23/23 OK, 0 FAILED, 0 missing.** Every published PDF matches its published hash.
- The "verify-yourself" provenance WORKS today: anyone can run `shasum -a 256 -c PAPER-PDF-SHA256SUMS.txt` and get 23/23 OK. Objective G's executable-verification claim is SOLID for the papers.

## LAW-3 merge-sequencing catch (genuine dependency — add to merge path)
The 6 held HTML edits on branch `pr/t2-research/paper-vi-falsifiability-20260705T0331Z` (Paper III R²=1, On-the-Origin ARC-Bound, IV-b overstatement+falsifiability, VI/VIII/XI falsifiability) change the HTML but NOT the PDFs (the PDFs are the pre-edit versions, which is why they still match the manifest). **When these HTML edits merge, the 6 PDFs must be regenerated AND `PAPER-PDF-SHA256SUMS.txt` updated for those 6 rows.** Otherwise:
- the published PDF will NOT contain the new falsifiability sections (HTML/PDF divergence), and
- the manifest will remain internally valid but describe stale PDFs.
This is a required post-merge step, not a blocker on the edits themselves. **Add to the T5 verification manifest's merge-sequencing note.**

## DOI note (Objective A "each paper DOI")
Papers reference the **programme OSF DOI (10.17605/OSF.IO/6C5XB)**. Whether each paper additionally carries its OWN per-paper DOI is an OSF-registration question (external/operator+T4), not a repo-editable item. Flag: confirm the Objective-A intent — one programme DOI (present) vs per-paper DOIs (OSF registration needed).

## Net
Provenance is genuinely verifiable and currently clean (23/23) — a real Objective-A/G strength, proved. One LAW-3 post-merge step (regen 6 PDFs + manifest) added to the merge path so the edits don't create HTML/PDF divergence. Route T4 + T5.
