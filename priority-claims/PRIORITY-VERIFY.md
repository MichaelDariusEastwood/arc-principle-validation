# How to verify the priority claims yourself, version 2

Everything below is checkable. If any step fails, the claim fails.

## 1. The hashes are the proof

```
01-dec8-2024-priority-anchor.eml      f0d1f38ffd8546152d9d9d28dc5ec083c16a35858f2c12b63e69db7ed50901ad
02-apr30-2025-book-manuscript.eml     09f5b5e156ed96f8883eaf668495fd350898ce62be6294b8f788e0e2d6dcb664
03-jan7-2026-bbc (original)           28ada54653d866fc308674cf417bb68015f494fa813f4d28ccec366c9a1ee7b3
04-jan22-2026-bbc (original)          af863bfa2fc992efc84c656d216894721ef205443774c7804d26907f6fc06c72
```

The two manuscript .emls are held in a companion private component and are available to serious verifiers on request (michael@michaeldariuseastwood.com). Run `shasum -a 256` on any provided copy and compare. The two BBC anchors are downloadable, in third-party-redacted form with both original and served hashes published, at https://www.michaeldariuseastwood.com/research/evidence-portal/

## 2. Read the headers of any provided .eml

`Date:` (8 Dec 2024 02:45:18 +0000 / 30 Apr 2025 13:37:05 +0100) · `Message-ID:` beginning `CAGPsKA`, the Gmail-origin pattern · `From:`/`To:` self-addressed. These establish a Google-server date for the exact hashed bytes. The DKIM signature operates in transit and is not preserved by the export, so the claim is Google-server-timestamped, not "DKIM-verified".

## 3. Confirm the deposit dates are later than the creation dates

On OSF, open any canonical file and click Revisions: version 1 of the earliest deposits is dated 17 March 2026. Creation is the anchors in step 1, fifteen months earlier.

## 4. Confirm the book

ISBN 978-1806056200, published 2 January 2026 in print, 6 January as ebook; 37 named original concepts; Appendix E lists 127 verified sources.

## 5. Check the full ledger

All 37 priority claims (PC-001 to PC-037), each with its date, category, claim text, evidence pointers and risk notes: https://www.michaeldariuseastwood.com/research/priority.json · human-readable register at /priority-claims.html · convergence register at /research/evidence-portal/

## 6. Run the mathematics

`git clone https://github.com/MichaelDariusEastwood/arc-principle-validation`, then in papers/Paper-X-Coupled-CoScaling-Correction run `pip install -r requirements.txt && python code/test_theorems_independent.py`. Fourteen checks, about one second, no API keys and no model calls.

*Version 2 · 31 July 2026 · none of this requires trusting the author.*
