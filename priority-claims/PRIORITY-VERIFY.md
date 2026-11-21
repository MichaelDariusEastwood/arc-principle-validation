# How to verify the priority record yourself, version 3

Everything below can be checked by a stranger. If a step fails, the statement it supports fails.

## 1. The hashes

```
01-dec8-2024-priority-anchor.eml      f0d1f38ffd8546152d9d9d28dc5ec083c16a35858f2c12b63e69db7ed50901ad
02-apr30-2025-book-manuscript.eml     09f5b5e156ed96f8883eaf668495fd350898ce62be6294b8f788e0e2d6dcb664
03-jan7-2026-bbc (original)           28ada54653d866fc308674cf417bb68015f494fa813f4d28ccec366c9a1ee7b3
04-jan22-2026-bbc (original)          af863bfa2fc992efc84c656d216894721ef205443774c7804d26907f6fc06c72
```

The two message files (01 and 02) are held privately and are sent to any verifier on request, through the contact page of the author's website: https://www.michaeldariuseastwood.com/contact/. Run `shasum -a 256` on the copy you receive and compare. The two BBC anchors are held as sealed sources; the evidence record describes them: https://www.michaeldariuseastwood.com/research/evidence-portal/

## 2. Read the Bitcoin receipts

This folder holds an OpenTimestamps receipt for each of the two message files:

- `01-dec8-2024-priority-anchor.eml.ots`
- `02-apr30-2025-book-manuscript.eml.ots`

A receipt contains no part of a message. It contains the file's SHA-256 and a chain of hashes that leads from it to a Bitcoin block. With the OpenTimestamps client installed, run:

```
ots info 01-dec8-2024-priority-anchor.eml.ots
```

The output gives the file's SHA-256, which must equal the value in step 1, and names two attestations, `BitcoinBlockHeaderAttestation(961340)` and `BitcoinBlockHeaderAttestation(961402)`, each with its block's merkle root. Open each block on any public block explorer and compare:

| Block | Mined (UTC) | Merkle root |
|---|---|---|
| 961,340 | 6 August 2026, 21:36:23 | 038418c103704b8c307a645240ea54c874c8f758d39febd19f14a6fb8f7abe42 |
| 961,402 | 7 August 2026, 08:45:20 | cc48815facf6a6142d441e196e6a5eb1fafe661705735f840a11ffaf89b2ba54 |

The programme's own check of 21 November 2026 followed each receipt's chain of hashes to both merkle roots, as two public explorers give them.

What the receipts prove: files with exactly these SHA-256 values existed when block 961,340 was mined, on 6 August 2026. A receipt dates bytes from its own block and never earlier, so it does not carry the dates of December 2024 and April 2025. Those rest on step 3. With a copy of a message file and a Bitcoin node, `ots verify 01-dec8-2024-priority-anchor.eml.ots` checks the file itself against the block.

## 3. Check the mail servers' signatures on the message of 8 December 2024

The message of 8 December 2024 carries times signed by the mail servers. Its Date field, 02:45:18 UTC, is covered by the sender's DKIM signature (signed at 02:45:30 UTC) and by Google's ARC set (sealed at 02:45:31 UTC with Google's own key). Both were verified on 8 October 2026 against the keys published in DNS. The header block, the check script, a guide and the full report have been public on OSF since 15 October 2026, at https://doi.org/10.17605/OSF.IO/QASRK:

```
python3 verify_headers.py HEADERS-2024-12-08.txt
```

The report (VERIFY-REPORT.md) gives every signature, what it covers and its limits. Three points from it:

- The signatures are on the copy of the message as received, SHA-256 ecdc8999e5326235ad9a12810e864eefb6f305c04b15568c0c86a1a3a6a53b39. The copy in step 1 is the copy as sent. It has the same identifying header fields, and the same five attachments byte for byte, and it carries no server signature, because the sender's server signs a message as it leaves and Google adds its record on receipt.
- The ARC set is Google's own record of its check at receipt, timed by Google's clock and sealed with Google's key. It is not an independent timestamping service.
- Four of the five attachments are the manuscripts, public in the same OSF component with their SHA-256 values. The fifth is a question-and-answer file whose answers are not the author's words; it is available to verifiers on request.

The message of 30 April 2025 has no published header block. Its date, 30 April 2025 at 13:37:05 +0100, is the Date field of the sender's copy, and its SHA-256 in step 1 identifies the exact bytes.

## 4. Confirm that the deposits came later

On OSF, each document's oldest file keeps every version under Revisions, with OSF's own dates. The priority index lists each document's earliest file with its date and a link to that revision. The earliest it lists is Paper I's, deposited on 17 January 2026, more than thirteen months after the message of 8 December 2024: https://www.michaeldariuseastwood.com/research/priority-index/

## 5. Confirm the book

Infinite Architects, ISBN 978-1806056200, was published in print on 2 January 2026 and as an ebook on 6 January 2026. It names 37 concepts.

## 6. Check the full ledger

Every dated entry (PC-001 to PC-041 on 21 November 2026), each with its date, category, claim text, evidence pointers and risk notes: https://www.michaeldariuseastwood.com/research/priority.json. The readable register is at https://www.michaeldariuseastwood.com/priority-claims, and the earliest dated record of each major claim is in the priority index named in step 4.

## 7. Run the mathematics

```
git clone https://github.com/MichaelDariusEastwood/arc-principle-validation
cd arc-principle-validation/papers/Paper-X-Coupled-CoScaling-Correction
pip install -r requirements.txt
python code/test_theorems_independent.py
```

It runs Paper X's checks of its theorems. On 21 November 2026, at commit c5c46ff, it printed `104 passed` in about five seconds. No API keys and no model calls.

*Version 3 · 21 November 2026 · none of this requires trusting the author.*

## Changes from version 2 (31 July 2026)

- The two receipts are now in this folder. The website's verification page said they were here; they had not been deposited.
- Step 3 gives the mail servers' signatures as verified on 8 October 2026. Version 2 said the DKIM signature was not preserved by the export; it is preserved on the copy as received.
- The earliest OSF deposit is dated 17 January 2026. Version 2 gave 17 March 2026.
- The ledger has 41 entries. Version 2 counted 37.
- The book's 37 named concepts are described without the word original, which the programme now uses only with a recorded search behind it. The count of the book's sources is not carried, because it was not checked again for this version.
- Paper X's check count is dated and tied to a commit. The suite has grown from the fourteen checks version 2 named.
- The request route is the website's contact page, and the two BBC anchors are described as the evidence record now describes them.
